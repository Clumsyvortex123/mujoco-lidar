"""Batched 3D lidar scanning for MuJoCo models.

The scanner is stateless with respect to time: it does not own a thread, a
rate, or a publisher. You call :meth:`LidarScanner.scan` whenever you want a
frame, and it returns the points. That keeps it usable from a sim loop, a
notebook, a dataset generator, or a test.
"""

from dataclasses import dataclass

import numpy as np
import mujoco

__all__ = ["LidarConfig", "LidarScanner"]


def _numeric(model, name, default=None):
    """Read a <custom><numeric> array from the model, or return ``default``."""
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if idx < 0:
        if default is None:
            raise KeyError(
                f"model has no <numeric name=\"{name}\">; either add it to the "
                f"lidar XML or pass an explicit LidarConfig")
        return np.asarray(default, dtype=float)
    adr = model.numeric_adr[idx]
    size = model.numeric_size[idx]
    return np.array(model.numeric_data[adr:adr + size], dtype=float)


@dataclass
class LidarConfig:
    """Beam pattern. Normally loaded from the sensor XML, not hand-built."""

    rings: int = 32
    elev_min: float = -22.0
    elev_max: float = 15.0
    azimuth_step: float = 0.9
    max_range: float = 35.0
    rate: float = 10.0

    @classmethod
    def from_model(cls, model, prefix="lidar_"):
        """Read the ``<custom><numeric>`` block written by ``lidar_3d.xml``.

        Missing entries fall back to this dataclass's defaults, so a model
        that only overrides ``lidar_rings`` still loads.
        """
        d = cls()
        elevation = _numeric(model, prefix + "elevation", [d.elev_min, d.elev_max])
        if elevation.size != 2:
            raise ValueError(
                f"<numeric name=\"{prefix}elevation\"> needs exactly 2 values "
                f"(min max), got {elevation.size}")
        return cls(
            rings=int(_numeric(model, prefix + "rings", [d.rings])[0]),
            elev_min=float(elevation[0]),
            elev_max=float(elevation[1]),
            azimuth_step=float(_numeric(model, prefix + "azimuth_step",
                                        [d.azimuth_step])[0]),
            max_range=float(_numeric(model, prefix + "max_range",
                                     [d.max_range])[0]),
            rate=float(_numeric(model, prefix + "rate", [d.rate])[0]),
        )

    def validate(self):
        if self.rings < 1:
            raise ValueError("rings must be >= 1")
        if self.elev_min > self.elev_max:
            raise ValueError("elevation min must be <= max")
        if not 0 < self.azimuth_step <= 360:
            raise ValueError("azimuth_step must be in (0, 360]")
        if self.max_range <= 0:
            raise ValueError("max_range must be > 0")
        return self


def build_ray_pattern(config):
    """Unit ray directions in the sensor frame, shape (n_rays, 3), float64.

    Frame convention: **+X forward, +Y left, +Z up, positive elevation up.**

    Rays are ordered ring-major: all azimuths of ring 0, then ring 1, and so
    on, so ``pattern.reshape(rings, n_azimuth, 3)`` recovers the ring layout
    if you want per-ring processing.
    """
    config.validate()

    n_azimuth = int(round(360.0 / config.azimuth_step))
    if n_azimuth < 1:
        raise ValueError("azimuth_step too large, no beams generated")

    if config.rings == 1:
        elevations = np.array([(config.elev_min + config.elev_max) / 2.0])
    else:
        elevations = np.linspace(config.elev_min, config.elev_max, config.rings)

    azimuths = np.arange(n_azimuth) * (360.0 / n_azimuth)

    el = np.radians(elevations)[:, None]
    az = np.radians(azimuths)[None, :]
    cos_el = np.cos(el)

    dirs = np.stack([
        cos_el * np.cos(az),
        cos_el * np.sin(az),
        np.broadcast_to(np.sin(el), (config.rings, n_azimuth)),
    ], axis=-1)

    return np.ascontiguousarray(dirs.reshape(-1, 3), dtype=np.float64), n_azimuth


class LidarScanner:
    """Casts a lidar beam pattern into a MuJoCo model.

    Args:
        model: ``MjModel`` that contains the lidar site.
        data:  ``MjData`` the scanner reads poses from.
        site:  name of the site that defines the ray origin and frame.
        config: beam pattern. Defaults to ``LidarConfig.from_model(model)``.
        exclude_body: body id whose geoms rays ignore. Defaults to the site's
            own body, so the sensor never returns its own housing. Pass -1 to
            disable.
        geomgroup: optional length-6 mask over MuJoCo geom groups; ``None``
            scans every group.
        include_static: whether world/static geoms are scannable. Almost
            always ``True`` -- a ground plane and fixed obstacles are static.
    """

    def __init__(self, model, data, site="lidar_origin", config=None,
                 exclude_body=None, geomgroup=None, include_static=True):
        self.model = model
        self.data = data

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(
                f"site {site!r} not found; did you <include> the lidar XML?")

        self.config = (config or LidarConfig.from_model(model)).validate()
        self.pattern, self.n_azimuth = build_ray_pattern(self.config)
        self.n_rays = self.pattern.shape[0]
        self.n_rings = self.config.rings

        if exclude_body is None:
            exclude_body = int(model.site_bodyid[self.site_id])
        self.exclude_body = int(exclude_body)
        self.flg_static = bool(include_static)
        self.geomgroup = (None if geomgroup is None
                          else np.asarray(geomgroup, dtype=np.uint8))

        self._world_dirs = np.zeros((self.n_rays, 3), dtype=np.float64)
        self._dist = np.zeros(self.n_rays, dtype=np.float64)
        self._geomid = np.zeros(self.n_rays, dtype=np.int32)

    def origin(self):
        """Sensor origin in world coordinates, shape (3,)."""
        return np.array(self.data.site_xpos[self.site_id], dtype=np.float64)

    def rotation(self):
        """Sensor orientation in world coordinates as a 3x3 matrix."""
        return np.array(self.data.site_xmat[self.site_id],
                        dtype=np.float64).reshape(3, 3)

    def quat(self):
        """Sensor orientation in world coordinates as (w, x, y, z)."""
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(
            self.data.site_xmat[self.site_id], dtype=np.float64))
        return q

    def scan(self, return_mask=False):
        """Cast every beam once and return the hits.

        Returns:
            (N, 3) float32 array of points in the **sensor frame**, where N is
            the number of beams that hit geometry. Non-returns are dropped, so
            N varies frame to frame. With ``return_mask=True`` also returns the
            (n_rays,) boolean hit mask, which lets you recover which ring and
            azimuth each point came from.
        """
        origin = self.origin()
        rot = self.rotation()

        np.matmul(self.pattern, rot.T, out=self._world_dirs)

        mujoco.mj_multiRay(
            m=self.model,
            d=self.data,
            pnt=origin,
            vec=self._world_dirs.reshape(-1),
            geomgroup=self.geomgroup,
            flg_static=self.flg_static,
            bodyexclude=self.exclude_body,
            geomid=self._geomid,
            dist=self._dist,
            normal=None,
            nray=self.n_rays,
            cutoff=self.config.max_range,
        )

        hit = (self._geomid >= 0) & (self._dist > 0.0)

        points = (self.pattern[hit] * self._dist[hit, None]).astype(np.float32)

        return (points, hit) if return_mask else points

    def scan_world(self):
        """Convenience: the same scan expressed in world coordinates."""
        points = self.scan()
        if points.size == 0:
            return points
        return (points @ self.rotation().T + self.origin()).astype(np.float32)

    def describe(self):
        c = self.config
        return (f"{self.n_rings} rings x {self.n_azimuth} azimuth = "
                f"{self.n_rays} beams | elevation {c.elev_min:g} to "
                f"{c.elev_max:g} deg | range {c.max_range:g} m")
