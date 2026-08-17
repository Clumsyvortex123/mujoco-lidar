#!/usr/bin/env python3
"""Flask 3D point cloud viewer for the MuJoCo 3D lidar.

Runs as a separate process from the simulator; the only coupling is zenoh.

    zenoh  mujoco_lidar/points  --> this server --> browser (WebGL)
    browser keys --> this server --> zenoh  mujoco_lidar/cmd --> simulator

The cloud is relayed to the browser as the *same* binary blob that came off
the wire. No JSON re-encoding: a 12k point frame is ~150 KB of float32, which
becomes ~2 MB and a parse stall per frame if you round trip it through JSON.
The browser hands the ArrayBuffer straight to a WebGL vertex buffer.
"""

import sys
import json
import time
import signal
import argparse
import threading

from flask import Flask, Response, request, jsonify, send_from_directory
from mujoco_lidar.cloud import unpack_header
from mujoco_lidar.comms import (
    TOPIC_POINTS, TOPIC_CMD, open_session, payload_bytes,
)

app = Flask(__name__, static_folder="static", static_url_path="/static")

import logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)


class CloudCache:
    """Holds the most recent frame. One writer (zenoh), many readers (HTTP)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._blob = None
        self._header = None
        self._recv_times = []
        self.frames = 0
        self.errors = 0
        self.last_error = None

    def update(self, blob):
        try:
            header = unpack_header(blob)
        except ValueError as exc:
            with self._lock:
                self.errors += 1
                self.last_error = str(exc)
            print(f"[viewer] dropping malformed cloud: {exc}", file=sys.stderr)
            return

        now = time.time()
        with self._lock:
            self._blob = blob
            self._header = header
            self.frames += 1
            self._recv_times.append(now)
            if len(self._recv_times) > 30:
                del self._recv_times[:-30]

    def get(self):
        with self._lock:
            return self._blob, self._header

    def stats(self):
        with self._lock:
            hz = 0.0
            if len(self._recv_times) >= 2:
                span = self._recv_times[-1] - self._recv_times[0]
                if span > 0:
                    hz = (len(self._recv_times) - 1) / span
            header = dict(self._header) if self._header else None
            age = (time.time() - self._recv_times[-1]) if self._recv_times else None
            return {
                "connected": header is not None,
                "frames": self.frames,
                "errors": self.errors,
                "last_error": self.last_error,
                "rate_hz": round(hz, 2),
                "age_s": round(age, 3) if age is not None else None,
                "header": header,
            }


cache = CloudCache()
cmd_pub = None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/cloud")
def api_cloud():
    """Latest frame as raw bytes.

    ``?since=<seq>`` returns 204 when the client already has that frame, so a
    polling client costs one empty response instead of a redundant 150 KB.
    """
    blob, header = cache.get()
    if blob is None:
        return Response(status=204)

    since = request.args.get("since")
    if since is not None:
        try:
            if int(since) == header["seq"]:
                return Response(status=204)
        except ValueError:
            pass

    return Response(blob, mimetype="application/octet-stream", headers={
        "Cache-Control": "no-store",
        "X-Cloud-Seq": str(header["seq"]),
        "X-Cloud-Points": str(header["n_points"]),
    })


@app.route("/api/cmd", methods=["POST"])
def api_cmd():
    """Forward a movement command to the simulator over zenoh."""
    if cmd_pub is None:
        return jsonify({"ok": False, "error": "no zenoh publisher"}), 503

    msg = request.get_json(silent=True) or {}
    key = str(msg.get("key", "")).lower()

    allowed = set("wasdikqerp")
    if key:
        if key not in allowed:
            return jsonify({"ok": False, "error": f"unknown key {key!r}"}), 400
        payload = {"key": key}
    else:
        payload = {
            "dx": float(msg.get("dx", 0.0)),
            "dy": float(msg.get("dy", 0.0)),
            "dz": float(msg.get("dz", 0.0)),
            "dyaw": float(msg.get("dyaw", 0.0)),
        }

    cmd_pub.put(json.dumps(payload).encode("utf-8"))
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(cache.stats())


def main():
    global cmd_pub

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--points-topic", default=TOPIC_POINTS)
    p.add_argument("--cmd-topic", default=TOPIC_CMD)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    session = open_session()

    def on_cloud(sample):
        cache.update(payload_bytes(sample))

    sub = session.declare_subscriber(args.points_topic, on_cloud)
    cmd_pub = session.declare_publisher(args.cmd_topic)

    print(f"[viewer] subscribing <- {args.points_topic}")
    print(f"[viewer] publishing  -> {args.cmd_topic}")
    print(f"[viewer] open http://{args.host}:{args.port}")

    def shutdown(_sig, _frm):
        try:
            sub.undeclare()
            session.close()
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    app.run(host=args.host, port=args.port, threaded=True,
            debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
