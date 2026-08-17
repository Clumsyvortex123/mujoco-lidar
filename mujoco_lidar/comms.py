"""Zenoh helpers shared by the simulator and the viewer.

Zenoh is only used by the demo applications. The library core
(:mod:`mujoco_lidar.scanner`, :mod:`mujoco_lidar.cloud`) has no transport
dependency at all, so you can drop this module and publish frames over
whatever you already use.
"""

import json
import os

import zenoh

__all__ = ["TOPIC_POINTS", "TOPIC_CMD", "open_session", "payload_bytes"]

TOPIC_POINTS = "mujoco_lidar/points"
TOPIC_CMD = "mujoco_lidar/cmd"


def zenoh_config():
    """Loopback-only peer by default.

    A demo that multicasts onto the LAN the moment you run it is a bad
    neighbour, so scouting is pinned to the loopback interface. Set
    ``MUJOCO_LIDAR_LOCALHOST=0`` to use zenoh's stock configuration instead.
    """
    conf = zenoh.Config()
    if os.environ.get("MUJOCO_LIDAR_LOCALHOST", "1") != "0":
        conf.insert_json5("mode", json.dumps("peer"))
        conf.insert_json5("listen/endpoints",
                          json.dumps(["tcp/127.0.0.1:0#iface=lo"]))
        conf.insert_json5("scouting", json.dumps({
            "multicast": {"enabled": True, "interface": "lo"},
            "gossip": {"enabled": True},
        }))
    return conf


def open_session():
    zenoh.try_init_log_from_env()
    return zenoh.open(zenoh_config())


def payload_bytes(sample):
    """Raw bytes from a zenoh Sample, across 1.x payload flavours."""
    payload = sample.payload
    to_bytes = getattr(payload, "to_bytes", None)
    return to_bytes() if to_bytes is not None else bytes(payload)
