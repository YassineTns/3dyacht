"""Procedural, tileable knit textures (numpy only), written as PNG files.

* jersey: technical face of a single jersey - columns (wales) of V-shaped loops, 12 wales and
  16 courses per centimetre, which is what a heavy 240-280 g/m2 cotton jersey looks like.
* rib:    1x1 rib for the neckband - raised knit wales alternating with recessed purl wales.

Each texture is a height map (yarn relief, used for cavity/roughness) and a tangent-space
normal map (OpenGL / Blender convention, +Y up). One tile = 1 x 1 cm.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, img: np.ndarray, bits: int = 16) -> None:
    """Minimal PNG writer (grey / RGB / RGBA, 8 or 16 bit); row 0 of `img` is the BOTTOM."""
    img = np.asarray(img, np.float64)
    if img.ndim == 2:
        img = img[..., None]
    h, w, ch = img.shape
    ctype = {1: 0, 3: 2, 4: 6}[ch]
    q = np.clip(np.round(img[::-1] * (2 ** bits - 1)), 0, 2 ** bits - 1)
    raw = q.astype(">u2" if bits == 16 else "u1").reshape(h, -1).view(np.uint8)
    data = b"".join(b"\x00" + row.tobytes() for row in raw)

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, ctype, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(data, 9)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def _blur_wrap(a: np.ndarray, sigma_px: float) -> np.ndarray:
    """Periodic gaussian blur through the FFT (keeps the tile seamless)."""
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    g = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    return np.fft.irfft2(np.fft.rfft2(a) * g, s=a.shape)


def _legs(x, y, cx, cy, w, c, tilt, height, length=1.45, thick=0.52, twist=0.035):
    """Height of the two legs of one knit loop centred at (cx, cy)."""
    out = np.zeros_like(x)
    for side in (-1.0, 1.0):
        ax = np.array([side * np.sin(tilt), np.cos(tilt)])  # leg axis, bottom -> top
        nx = np.array([ax[1], -ax[0]])
        lx, ly = cx + side * 0.25 * w, cy
        dx, dy = x - lx, y - ly
        u = (dx * ax[0] + dy * ax[1]) / (0.5 * length * c)
        v = (dx * nx[0] + dy * nx[1]) / (0.5 * thick * w)
        r2 = u * u + v * v
        hleg = np.sqrt(np.clip(1.0 - r2, 0.0, None))
        # yarn twist: fine slanted striations across the leg
        hleg *= 1.0 + twist * np.sin(2 * np.pi * (u * 4.5 + v * 1.6))
        out = np.maximum(out, height * hleg)
    return out


def knit_height(size: int = 1024, wales: int = 12, courses: int = 16, rib: bool = False,
                seed: int = 11) -> np.ndarray:
    """Tileable height map in [0, 1] of a 1 x 1 cm patch of fabric."""
    rng = np.random.default_rng(seed)
    y, x = (np.mgrid[0:size, 0:size].astype(np.float32) + 0.5)
    w, c = size / wales, size / courses
    jitter = rng.normal(0.0, 1.0, (courses, wales, 4)).astype(np.float32)
    h = np.zeros((size, size), np.float32)
    ci, cj = np.floor(x / w).astype(int), np.floor(y / c).astype(int)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            i, j = ci + di, cj + dj
            jit = jitter[j % courses, i % wales]
            cx = (i + 0.5) * w + 0.04 * w * jit[..., 0]
            cy = (j + 0.5) * c + 0.05 * c * jit[..., 1]
            amp = 1.0 + 0.06 * jit[..., 2]
            if rib:
                purl = (i % 2) == 1
                # knit wales stand out, purl wales sit back and show horizontal loops
                knit = _legs(x, y, cx, cy, w, c, 0.30, amp, thick=0.62 + 0.03 * jit[..., 3])
                dxp, dyp = (x - cx) / (0.55 * w), (y - cy) / (0.36 * c)
                purl_h = 0.45 * amp * np.sqrt(np.clip(1.0 - dxp ** 2 - dyp ** 2, 0.0, None))
                h = np.maximum(h, np.where(purl, purl_h, knit))
            else:
                h = np.maximum(h, _legs(x, y, cx, cy, w, c, 0.36, amp,
                                        thick=0.52 + 0.03 * jit[..., 3]))
    # fibre fuzz: a little high-frequency noise, softened
    fuzz = _blur_wrap(rng.normal(0.0, 1.0, (size, size)), 1.2)
    h = h + 0.05 * fuzz / (np.abs(fuzz).max() + 1e-6)
    h = _blur_wrap(h, 0.8)
    h -= h.min()
    return (h / h.max()).astype(np.float32)


def normal_from_height(h: np.ndarray, depth_px: float) -> np.ndarray:
    """Tangent-space normal map (0..1 encoded) from a periodic height map; depth_px is the
    relief amplitude expressed in pixels."""
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * depth_px
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * depth_px
    n = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def generate(out_dir: Path, size: int = 1024, force: bool = False) -> dict[str, Path]:
    """Write jersey/rib height + normal maps once (they are deterministic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, kw, depth in (("jersey", dict(wales=12, courses=16), 18.0),
                            ("rib", dict(wales=10, courses=16, rib=True), 24.0)):
        hp, np_ = out_dir / f"{name}_height.png", out_dir / f"{name}_normal.png"
        if force or not (hp.exists() and np_.exists()):
            h = knit_height(size, **kw)
            write_png(hp, h, bits=8)
            write_png(np_, normal_from_height(h, depth), bits=8)
        files[f"{name}_height"], files[f"{name}_normal"] = hp, np_
    return files


if __name__ == "__main__":  # python scripts/tee3d/textures.py -> assets/textures
    root = Path(__file__).resolve().parents[2]
    for k, v in generate(root / "assets" / "textures", force=True).items():
        print(k, v)
