#!/usr/bin/env python3
"""The smallest useful example: put the lidar in your own scene and scan.

Everything the sensor needs is the <include> and the site it defines. Run:

    ./.venv/bin/python examples/minimal_scan.py
"""

import numpy as np
import mujoco

from mujoco_lidar import LidarScanner, LIDAR_XML_PATH

# Your model. The only lidar-specific line is the <include>.
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

# Handing the include through the assets dict means the scene compiles from
# any working directory. from_xml_path() with the file on disk works too.
assets = {"lidar_3d.xml": LIDAR_XML_PATH.read_bytes()}
model = mujoco.MjModel.from_xml_string(SCENE, assets)
data = mujoco.MjData(model)

scanner = LidarScanner(model, data)
print("beam pattern from XML:", scanner.describe())

# Place the sensor. It is a mocap body in the shipped XML, so writing
# mocap_pos moves it; mj_forward updates the site pose the scanner reads.
data.mocap_pos[0] = [0.0, 0.0, 1.2]
mujoco.mj_forward(model, data)

points = scanner.scan()           # (N, 3) float32, SENSOR frame
world = scanner.scan_world()      # the same scan in world coordinates

print(f"returns      : {len(points)} of {scanner.n_rays} beams "
      f"({100 * len(points) / scanner.n_rays:.0f}% hit)")
print(f"range        : {np.linalg.norm(points, axis=1).min():.2f} m to "
      f"{np.linalg.norm(points, axis=1).max():.2f} m")
print(f"world z span : {world[:, 2].min():.2f} m to {world[:, 2].max():.2f} m")

# Recover which ring each point came from, if you need per-ring processing.
_, mask = scanner.scan(return_mask=True)
ring_of_beam = np.repeat(np.arange(scanner.n_rings), scanner.n_azimuth)
print(f"points in the lowest ring: {(ring_of_beam[mask] == 0).sum()}")
