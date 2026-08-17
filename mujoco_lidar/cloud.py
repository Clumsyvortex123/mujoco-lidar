"""Point cloud wire format.

One flat binary blob per frame: a 56 byte header followed by tightly packed
float32 XYZ. No JSON, no protobuf, no per-point parsing anywhere in the chain.

Layout (little endian)::

    offset  size  field
    0       4     magic   b'MJLC'
    4       2     version
    6       2     flags        (reserved)
    8       4     n_points
    12      4     seq
    16      8     stamp        unix seconds, float64
    24      12    origin       sensor position in world, 3 x float32
    36      16    quat         sensor orientation in world, w x y z, float32
    52      4     _pad
    56      ...   points       n_points x (x, y, z) float32, SENSOR FRAME

Two properties of that layout are deliberate:

* The header is 56 bytes, a multiple of 4, so a browser can do
  ``new Float32Array(buffer, 56, n * 3)`` with zero copying. A typed array
  view requires its byte offset to be aligned to the element size, and the
  float64 timestamp sits at offset 16 so it stays 8 byte aligned too.

* Points are in the **sensor frame** and the sensor pose rides in the header.
  Baking world coordinates into the payload would be simpler and strictly
  worse: the consumer could no longer recover range, incidence, or the sensor
  frame, and a cloud with no declared frame is ambiguous by construction.
"""

import struct

import numpy as np

__all__ = ["HEADER_SIZE", "MAGIC", "VERSION",
           "pack_cloud", "unpack_header", "unpack_points"]

MAGIC = b"MJLC"
VERSION = 1
HEADER_FMT = "<4sHHIId3f4fI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

if HEADER_SIZE != 56 or HEADER_SIZE % 4 != 0:
    raise RuntimeError(
        f"header must be 56 bytes and 4 byte aligned for zero-copy typed "
        f"array views, got {HEADER_SIZE}")


def pack_cloud(points, origin, quat, seq, stamp):
    """Serialise a sensor-frame cloud.

    Args:
        points: (N, 3) XYZ in the sensor frame.
        origin: (3,) sensor position in world.
        quat:   (4,) sensor orientation in world as w, x, y, z.
        seq:    monotonically increasing frame counter.
        stamp:  unix seconds.

    Returns:
        ``bytes`` ready for any transport.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    o = np.asarray(origin, dtype=np.float32).reshape(3)
    q = np.asarray(quat, dtype=np.float32).reshape(4)

    header = struct.pack(
        HEADER_FMT,
        MAGIC, VERSION, 0,
        pts.shape[0], seq & 0xFFFFFFFF, float(stamp),
        float(o[0]), float(o[1]), float(o[2]),
        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        0,
    )
    return header + pts.tobytes()


def unpack_header(buf):
    """Read and validate the header. Raises ``ValueError`` on a bad blob."""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"truncated cloud: {len(buf)} < {HEADER_SIZE} bytes")

    (magic, version, flags, n_points, seq, stamp,
     ox, oy, oz, qw, qx, qy, qz, _pad) = struct.unpack(
        HEADER_FMT, buf[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"unsupported wire version {version}")

    expected = HEADER_SIZE + n_points * 12
    if len(buf) != expected:
        raise ValueError(
            f"payload size mismatch: {len(buf)} bytes for {n_points} points, "
            f"expected {expected}")

    return {
        "version": version,
        "flags": flags,
        "n_points": n_points,
        "seq": seq,
        "stamp": stamp,
        "origin": (ox, oy, oz),
        "quat": (qw, qx, qy, qz),
    }


def unpack_points(buf):
    """Return the (N, 3) float32 sensor-frame points from a packed cloud."""
    header = unpack_header(buf)
    return np.frombuffer(
        buf, dtype=np.float32, count=header["n_points"] * 3, offset=HEADER_SIZE
    ).reshape(-1, 3)
