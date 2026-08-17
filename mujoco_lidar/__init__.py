"""A reusable 3D lidar sensor for MuJoCo.

The sensor itself is ``model/lidar_3d.xml``; this package is the Python side
that reads it and turns beams into point clouds.

Typical use in your own model::

    import mujoco
    from mujoco_lidar import LidarScanner

    model = mujoco.MjModel.from_xml_path("my_scene.xml")  # <include>s lidar_3d.xml
    data = mujoco.MjData(model)
    scanner = LidarScanner(model, data)

    mujoco.mj_step(model, data)
    points = scanner.scan()          # (N, 3) float32, sensor frame
    world = scanner.scan_world()     # same scan in world coordinates
"""

from .cloud import (HEADER_SIZE, MAGIC, VERSION,
                    pack_cloud, unpack_header, unpack_points)
from .scanner import LidarConfig, LidarScanner, build_ray_pattern
from .scene import SceneSpec, build_scene_xml, load_scene, LIDAR_XML_PATH

__version__ = "1.0.0"

__all__ = [
    "LidarScanner", "LidarConfig", "build_ray_pattern",
    "pack_cloud", "unpack_header", "unpack_points",
    "HEADER_SIZE", "MAGIC", "VERSION",
    "SceneSpec", "build_scene_xml", "load_scene", "LIDAR_XML_PATH",
]
