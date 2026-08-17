#!/usr/bin/env python3
"""The smallest useful example: put the lidar in your own scene and scan.

Everything the sensor needs is the <include> and the site it defines. Run:

    ./.venv/bin/python examples/minimal_scan.py
"""

import numpy as np
import mujoco

from mujoco_lidar import LidarScanner, LIDAR_XML_PATH

SCENE = """
<mujoco model="minimal">
  <include file="lidar_3d.xml"/>
  <worldbody>
    <geom name="ground" type="plane" size="20 20 0.1" rgba="0.3 0.3 0.35 1"/>
    <geom name="wall"   type="box"  size="0.2 6 2" pos="6 0 2" rgba="0.7 0.3 0.2 1"/>
    <geom name="pillar" type="cylinder" size="0.5 1.5" pos="-3 2 1.5" rgba="0.2 0.5 0.7 1"/>
  </worldbody>
</mujoco>
"""

assets = {"lidar_3d.xml": LIDAR_XML_PATH.read_bytes()}
model = mujoco.MjModel.from_xml_string(SCENE, assets)
data = mujoco.MjData(model)

scanner = LidarScanner(model, data)
print("beam pattern from XML:", scanner.describe())

data.mocap_pos[0] = [0.0, 0.0, 1.2]
mujoco.mj_forward(model, data)

points = scanner.scan()
world = scanner.scan_world()

print(f"returns      : {len(points)} of {scanner.n_rays} beams "
      f"({100 * len(points) / scanner.n_rays:.0f}% hit)")
print(f"range        : {np.linalg.norm(points, axis=1).min():.2f} m to "
      f"{np.linalg.norm(points, axis=1).max():.2f} m")
print(f"world z span : {world[:, 2].min():.2f} m to {world[:, 2].max():.2f} m")

_, mask = scanner.scan(return_mask=True)
ring_of_beam = np.repeat(np.arange(scanner.n_rings), scanner.n_azimuth)
print(f"points in the lowest ring: {(ring_of_beam[mask] == 0).sum()}")
