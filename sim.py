#!/usr/bin/env python3
"""Demo simulator: drive the 3D lidar through a field of vertical obstacles.

Loads the example scene (which ``<include>``s ``model/lidar_3d.xml``), lets you
fly the sensor with the keyboard, and publishes each scan over zenoh.

Controls, in the MuJoCo window or forwarded from the browser viewer:

    W / S   forward / backward along the sensor heading
    A / D   strafe left / right
    I / K   up / down
    Q / E   yaw left / right
    R       reset pose
    P       print pose

The beam pattern comes from ``model/lidar_3d.xml``. The ``--rings`` style flags
below only *override* it, so the XML stays the single source of truth.
"""

import argparse
import json
import os
import signal
import sys
import threading
import traceback
import time
from collections import deque

import numpy as np
import mujoco

from mujoco_lidar import LidarScanner, LidarConfig, SceneSpec, load_scene, pack_cloud
from mujoco_lidar.comms import TOPIC_POINTS, TOPIC_CMD, open_session, payload_bytes

DRIVE_KEYS = "wasdikqerp"


class LidarSim:

    def __init__(self, args):
        self.args = args
        self.done = False
        self.seq = 0

        spec = SceneSpec(
            rows=args.rows, cols=args.cols, spacing=args.spacing,
            block=args.block, pillar_radius=args.pillar_radius,
            wall_height=args.wall_height, seed=args.seed,
            spawn_xy=(args.start_x, args.start_y),
            spawn_clearance=args.spawn_clearance,
        )
        self.model, self.data, n_pillars = load_scene(spec)

        if args.dump_xml:
            from mujoco_lidar.scene import build_scene_xml
            xml, _ = build_scene_xml(spec)
            with open(args.dump_xml, "w") as fd:
                fd.write(xml)
            print(f"[sim] wrote scene xml to {args.dump_xml} "
                  f"(it <include>s lidar_3d.xml)")

        # XML is the source of truth; CLI flags override only what was given.
        config = LidarConfig.from_model(self.model)
        for attr, value in (("rings", args.rings),
                            ("elev_min", args.elev_min),
                            ("elev_max", args.elev_max),
                            ("azimuth_step", args.azimuth_step),
                            ("max_range", args.max_range),
                            ("rate", args.publish_rate)):
            if value is not None:
                setattr(config, attr, value)

        self.scanner = LidarScanner(self.model, self.data,
                                    site=args.site, config=config)
        self.publish_rate = config.rate

        # The lidar body is a mocap body, so its pose is written directly
        # rather than driven by actuators or joints.
        body_id = int(self.model.site_bodyid[self.scanner.site_id])
        self.mocap_id = int(self.model.body_mocapid[body_id])
        if self.mocap_id < 0:
            raise RuntimeError(
                f"body holding site {args.site!r} is not a mocap body; "
                f"keyboard control needs mocap=\"true\" in the lidar XML")

        self.start_pose = (args.start_x, args.start_y, args.height, args.start_yaw)
        self.pos = np.array([args.start_x, args.start_y, args.height], dtype=np.float64)
        self.yaw = float(args.start_yaw)
        self.pose_lock = threading.Lock()
        self.cmd_queue = deque(maxlen=256)

        self.session = open_session()
        self.points_pub = self.session.declare_publisher(args.points_topic)
        self.cmd_sub = self.session.declare_subscriber(args.cmd_topic, self._on_cmd)

        print(f"[sim] scene   : {n_pillars} pillars + 4 walls, "
              f"{args.rows}x{args.cols} lattice at {args.spacing} m spacing")
        print(f"[sim] lidar   : {self.scanner.describe()}")
        print(f"[sim] rate    : {self.publish_rate} Hz")
        print(f"[sim] publish : {args.points_topic}")
        print(f"[sim] commands: {args.cmd_topic}")

    # -- pose control -------------------------------------------------------

    def _on_cmd(self, sample):
        """Zenoh handler. Runs on a zenoh thread, so it only queues."""
        try:
            msg = json.loads(payload_bytes(sample).decode("utf-8"))
        except Exception as exc:
            print(f"[sim] bad command payload: {exc}", file=sys.stderr)
            return
        self.cmd_queue.append(msg)

    def _drain_commands(self):
        """Apply queued commands. Called from the sim thread only."""
        while True:
            try:
                msg = self.cmd_queue.popleft()
            except IndexError:
                return
            key = str(msg.get("key", "")).lower()
            if key:
                self.apply_key(key)
            else:
                self.nudge(float(msg.get("dx", 0.0)), float(msg.get("dy", 0.0)),
                           float(msg.get("dz", 0.0)), float(msg.get("dyaw", 0.0)))

    def nudge(self, dx, dy, dz, dyaw):
        """Move (dx, dy) in the heading frame, dz in world, dyaw in degrees."""
        with self.pose_lock:
            self.yaw = (self.yaw + dyaw) % 360.0
            c, s = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
            self.pos[0] += dx * c - dy * s
            self.pos[1] += dx * s + dy * c
            self.pos[2] = max(0.05, self.pos[2] + dz)

    def reset_pose(self):
        with self.pose_lock:
            self.pos[:] = self.start_pose[:3]
            self.yaw = float(self.start_pose[3])

    def apply_key(self, key):
        step, yaw_step = self.args.step_size, self.args.yaw_step
        moves = {
            "w": (step, 0, 0, 0), "s": (-step, 0, 0, 0),
            "a": (0, step, 0, 0), "d": (0, -step, 0, 0),
            "i": (0, 0, step, 0), "k": (0, 0, -step, 0),
            "q": (0, 0, 0, yaw_step), "e": (0, 0, 0, -yaw_step),
        }
        if key in moves:
            self.nudge(*moves[key])
        elif key == "r":
            self.reset_pose()
        elif key == "p":
            with self.pose_lock:
                print(f"[sim] pose x={self.pos[0]:.2f} y={self.pos[1]:.2f} "
                      f"z={self.pos[2]:.2f} yaw={self.yaw:.1f}deg")

    def _key_callback(self, keycode):
        """MuJoCo viewer key hook. GLFW letter codes are ASCII uppercase."""
        if 0 <= keycode < 0x110000:
            self.apply_key(chr(keycode).lower())

    def _sync_mocap(self):
        with self.pose_lock:
            pos, yaw = self.pos.copy(), self.yaw
        half = np.radians(yaw) / 2.0
        self.data.mocap_pos[self.mocap_id] = pos
        self.data.mocap_quat[self.mocap_id] = [np.cos(half), 0.0, 0.0, np.sin(half)]

    # -- main loop ----------------------------------------------------------

    def publish(self):
        points = self.scanner.scan()
        blob = pack_cloud(points, self.scanner.origin(), self.scanner.quat(),
                          self.seq, time.time())
        self.points_pub.put(blob)
        self.seq += 1
        return points.shape[0]

    def run(self):
        period = 1.0 / self.publish_rate
        next_pub = time.time()
        last_report = time.time()
        frames = 0
        last_points = 0

        def step_once():
            nonlocal next_pub, last_report, frames, last_points
            self._drain_commands()
            self._sync_mocap()
            mujoco.mj_step(self.model, self.data)

            now = time.time()
            if now >= next_pub:
                last_points = self.publish()
                frames += 1
                # Skip missed slots rather than catching up in a burst.
                next_pub = max(now, next_pub + period)

            if self.args.stats and now - last_report >= 2.0:
                print(f"[sim] {frames / (now - last_report):5.1f} Hz  "
                      f"{last_points:6d} returns / {self.scanner.n_rays} beams")
                frames, last_report = 0, now

        if self.args.headless:
            print("[sim] headless: drive it from the browser viewer")
            while not self.done:
                step_once()
                time.sleep(max(0.0, self.model.opt.timestep * 0.5))
            return

        # Imported lazily under a distinct name: a bare `import mujoco.viewer`
        # would bind a local `mujoco` inside run(), which the step_once()
        # closure would then resolve to instead of the module global.
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            print("[sim] WASD move, I/K up-down, Q/E yaw, R reset, P print pose")
            while viewer.is_running() and not self.done:
                step_once()
                viewer.sync()

    def close(self):
        self.done = True
        try:
            self.cmd_sub.undeclare()
            self.points_pub.undeclare()
            self.session.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    scene = p.add_argument_group("scene")
    scene.add_argument("--rows", type=int, default=13)
    scene.add_argument("--cols", type=int, default=13)
    scene.add_argument("--spacing", type=float, default=3.0,
                       help="metres between lattice cells")
    scene.add_argument("--block", type=int, default=4,
                       help="every Nth row/col is left empty to form corridors")
    scene.add_argument("--pillar-radius", type=float, default=0.32,
                       help="nominal pillar half-width, jittered +/-15%%")
    scene.add_argument("--wall-height", type=float, default=6.0)
    scene.add_argument("--seed", type=int, default=7)
    scene.add_argument("--spawn-clearance", type=float, default=2.5,
                       help="radius cleared of pillars around the spawn point")
    scene.add_argument("--dump-xml", default=None,
                       help="write the generated scene MJCF here and continue")

    lidar = p.add_argument_group(
        "lidar overrides (default: whatever model/lidar_3d.xml declares)")
    lidar.add_argument("--site", default="lidar_origin",
                       help="site that defines the ray origin and frame")
    lidar.add_argument("--rings", type=int, default=None)
    lidar.add_argument("--elev-min", type=float, default=None)
    lidar.add_argument("--elev-max", type=float, default=None)
    lidar.add_argument("--azimuth-step", type=float, default=None)
    lidar.add_argument("--max-range", type=float, default=None)
    lidar.add_argument("--publish-rate", type=float, default=None)

    pose = p.add_argument_group("pose")
    pose.add_argument("--start-x", type=float, default=0.0)
    pose.add_argument("--start-y", type=float, default=0.0)
    pose.add_argument("--height", type=float, default=1.5)
    pose.add_argument("--start-yaw", type=float, default=0.0)
    pose.add_argument("--step-size", type=float, default=0.35,
                      help="metres moved per key press")
    pose.add_argument("--yaw-step", type=float, default=6.0,
                      help="degrees turned per key press")

    comms = p.add_argument_group("comms")
    comms.add_argument("--points-topic", default=TOPIC_POINTS)
    comms.add_argument("--cmd-topic", default=TOPIC_CMD)

    p.add_argument("--headless", action="store_true",
                   help="no MuJoCo window; drive from the browser viewer")
    p.add_argument("--stats", action="store_true",
                   help="print scan rate and return count every 2 s")

    args = p.parse_args()

    if args.headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    sim = LidarSim(args)
    signal.signal(signal.SIGINT, lambda *_: setattr(sim, "done", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(sim, "done", True))

    code = 0
    try:
        sim.run()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        sim.close()
        print("[sim] stopped")

    # MuJoCo's passive viewer can fault inside its own teardown during
    # interpreter shutdown (reproducible with mujoco 3.11 on Linux/GLFW), which
    # prints a "Segmentation fault (core dumped)" long after the simulator has
    # finished and closed its session. Everything we own is already shut down
    # and flushed at this point, so leave immediately rather than hand control
    # to that destructor.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
