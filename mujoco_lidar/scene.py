"""Demo world: a structured field of vertical obstacles.

This is the *example scene*, not part of the sensor. The sensor is defined
entirely by ``model/lidar_3d.xml``; everything here exists to give the beams
something interesting to hit.

The scene is generated rather than hand-written so the lattice size, spacing,
corridor pattern and pillar geometry are all parameters. It ``<include>``s the
lidar XML, which is the same thing you would do in your own model.
"""

from pathlib import Path

import numpy as np
import mujoco

__all__ = ["SceneSpec", "build_scene_xml", "load_scene", "LIDAR_XML_PATH"]

# The sensor definition ships inside the package, so it resolves the same way
# whether the repo was cloned or the package was pip installed.
MODEL_DIR = Path(__file__).resolve().parent / "model"
LIDAR_XML_PATH = MODEL_DIR / "lidar_3d.xml"
LIDAR_XML_NAME = "lidar_3d.xml"

# Six height tiers give the skyline vertical structure, so different elevation
# rings land on different surfaces instead of one flat silhouette.
HEIGHT_TIERS = (1.6, 2.6, 3.8, 5.2, 7.0, 9.5)

SCENE_TEMPLATE = """
<mujoco model="lidar_pillar_field">
  <include file="{lidar_xml}"/>

  <compiler angle="degree"/>
  <option timestep="0.004"/>

  <visual>
    <global offwidth="1600" offheight="900" azimuth="140" elevation="-25"/>
    <map znear="0.02" zfar="90"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.08 0.10 0.14" rgb2="0.02 0.02 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.10 0.12 0.15"
             rgb2="0.14 0.17 0.21" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="{tex_repeat} {tex_repeat}"
              reflectance="0.05"/>
    <material name="wall_mat" rgba="0.30 0.33 0.38 1"/>
{tier_materials}
  </asset>

  <worldbody>
    <light name="sun" pos="0 0 30" dir="0 0 -1" directional="true"
           diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <light name="fill" pos="{half_extent} {half_extent} 18" dir="-1 -1 -1"
           diffuse="0.30 0.30 0.35"/>

    <geom name="ground" type="plane" size="{ground_size} {ground_size} 0.1"
          material="grid"/>

{obstacles}
  </worldbody>
</mujoco>
"""


class SceneSpec:
    """Parameters of the demo obstacle field."""

    def __init__(self, rows=13, cols=13, spacing=3.0, block=4,
                 pillar_radius=0.32, wall_height=6.0, seed=7,
                 spawn_xy=(0.0, 0.0), spawn_clearance=2.5):
        if block < 2:
            raise ValueError("block must be >= 2 or every cell becomes corridor")
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.block = block
        self.pillar_radius = pillar_radius
        self.wall_height = wall_height
        self.seed = seed
        self.spawn_xy = spawn_xy
        self.spawn_clearance = spawn_clearance

    @property
    def half_extent(self):
        return (max(self.rows, self.cols) - 1) * self.spacing / 2.0

    @property
    def ground_size(self):
        return self.half_extent + 3.0 * self.spacing


def build_scene_xml(spec):
    """Render ``spec`` to MJCF. Returns ``(xml, n_pillars)``."""
    rng = np.random.default_rng(spec.seed)

    tier_materials = []
    for idx in range(len(HEIGHT_TIERS)):
        t = idx / max(len(HEIGHT_TIERS) - 1, 1)
        tier_materials.append(
            f'    <material name="tier{idx}" rgba="'
            f'{0.22 + 0.55 * t:.3f} {0.45 - 0.20 * t:.3f} '
            f'{0.70 - 0.35 * t:.3f} 1"/>')

    parts = []
    n_pillars = 0
    for i in range(spec.rows):
        for j in range(spec.cols):
            # Streets: skipping whole rows and columns leaves corridors you
            # can actually drive the sensor down.
            if i % spec.block == spec.block - 1 or j % spec.block == spec.block - 1:
                continue

            x = (i - (spec.rows - 1) / 2.0) * spec.spacing
            y = (j - (spec.cols - 1) / 2.0) * spec.spacing

            # With an odd row/column count a cell lands exactly on the origin
            # and the sensor would spawn *inside* it -- every beam then
            # returns about a metre and the cloud collapses to a ball.
            if spec.spawn_clearance > 0:
                dx = x - spec.spawn_xy[0]
                dy = y - spec.spawn_xy[1]
                if dx * dx + dy * dy < spec.spawn_clearance ** 2:
                    continue

            # Deterministic tier from a cheap integer hash of the cell, so a
            # given spec always produces the same world.
            tier = (i * 7 + j * 13 + (i * j) % 5) % len(HEIGHT_TIERS)
            height = HEIGHT_TIERS[tier]
            radius = spec.pillar_radius * (0.85 + 0.30 * float(rng.random()))

            if (i + j) % 2 == 0:
                geom = f'type="box" size="{radius:.3f} {radius:.3f} {height / 2:.3f}"'
            else:
                geom = f'type="cylinder" size="{radius:.3f} {height / 2:.3f}"'

            parts.append(
                f'    <geom name="pillar_{i}_{j}" {geom} '
                f'pos="{x:.3f} {y:.3f} {height / 2:.3f}" material="tier{tier}"/>')
            n_pillars += 1

    # Perimeter walls so distant beams terminate on something instead of
    # flying off to the cutoff and being dropped.
    w = spec.half_extent + spec.spacing
    t = 0.15
    h = spec.wall_height
    for name, size, pos in (
        ("wall_px", f"{t} {w + t} {h / 2}", f"{w} 0 {h / 2}"),
        ("wall_nx", f"{t} {w + t} {h / 2}", f"{-w} 0 {h / 2}"),
        ("wall_py", f"{w + t} {t} {h / 2}", f"0 {w} {h / 2}"),
        ("wall_ny", f"{w + t} {t} {h / 2}", f"0 {-w} {h / 2}"),
    ):
        parts.append(f'    <geom name="{name}" type="box" size="{size}" '
                     f'pos="{pos}" material="wall_mat"/>')

    xml = SCENE_TEMPLATE.format(
        lidar_xml=LIDAR_XML_NAME,
        tex_repeat=int(spec.ground_size),
        half_extent=f"{spec.half_extent:.3f}",
        ground_size=f"{spec.ground_size:.3f}",
        tier_materials="\n".join(tier_materials),
        obstacles="\n".join(parts),
    )
    return xml, n_pillars


def load_scene(spec):
    """Compile the demo scene. Returns ``(model, data, n_pillars)``.

    The lidar XML is handed to MuJoCo through the assets dict rather than
    resolved from disk, so the scene compiles no matter what the working
    directory is.
    """
    xml, n_pillars = build_scene_xml(spec)
    assets = {LIDAR_XML_NAME: LIDAR_XML_PATH.read_bytes()}
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return model, mujoco.MjData(model), n_pillars
