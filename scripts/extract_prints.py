#!/usr/bin/env python3
"""Step 1 - extract the two prints of the Tee_1.png mockup as transparent PNGs.

Outputs
  assets/prints/print_front.png  dragonfly on the rod tip + hanging line (photo, translucent wings)
  assets/prints/print_back.png   burgundy "Sport Fishing / Yacht Club Morocco / since 1985" logo
  assets/prints/prints.json      where each print sits on the garment (read by build_tee.py)
  renders/print_*_check.png      each print over checker / dark / fabric to judge the mattes

Method
  * Everything is un-mixed in linear light, the way Cycles will mix it back over the fabric.
  * The fabric under a print is estimated locally (normalised convolution around a dilated
    print mask), so the mockup's folds/shading and the faint green blotch behind the rod are
    removed instead of being baked into the print.
  * Front photo: "colour to alpha" against that background -> minimal alpha, true colours,
    translucent wings keep their partial alpha. Faint noise is cut with a hysteresis threshold.
  * Back logo: single ink, so alpha comes from luminance only (the webp's 4:2:0 chroma is
    blurry) and RGB is the flat ink colour -> no coloured fringes.
  * The rod and the line are cut by the garment edges in the mockup; their last clean cross
    section is extruded so the print bleeds past the side seam / hem.

Usage
  python scripts/extract_prints.py [--src Tee_1.png] [--scale 2] [--back-scale 4] [--ink #RRGGBB]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# --- Mockup geometry (pixels of the 2000x2000 Tee_1.png) -----------------------------------
# Front garment (bottom-left): body from x=382 to 961, shoulders (HPS) at y~715, hem at y=1755.
FRONT = dict(box=(716, 948, 996, 1756), center_x=671.5, hps_y=715.0, hem_y=1755.0,
             side_x=961.0,
             # last rows where the rod / line are cleanly inside the garment, before the
             # side-seam silhouette and the hem fold shading
             rod_clean_y=1596.0, line_clean_y=1722.0)
# Back garment (top-right): body from x=1003 to 1609, HPS at y~250, back neck y=286, hem y=1265.
BACK = dict(box=(1086, 366, 1516, 514), center_x=1306.0, hps_y=250.0, neck_y=286.0,
            hem_y=1265.0)

BLEED_PX = 40          # extra canvas below the hem / past the side seam (mockup pixels)
FABRIC_HEX = "#F2EFE6"  # garment colour, only used for the preview sheet


# --- colour helpers ------------------------------------------------------------------------
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=rgb.dtype)


def hex_to_linear(h: str) -> np.ndarray:
    h = h.lstrip("#")
    srgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
    return srgb_to_linear(srgb).astype(np.float32)


def linear_to_hex(c: np.ndarray) -> str:
    s = np.round(linear_to_srgb(np.asarray(c)) * 255).astype(int)
    return "#%02X%02X%02X" % tuple(s)


# --- image helpers -------------------------------------------------------------------------
def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    """Lanczos upscale of a float image (any channel count), overshoot clamped."""
    if scale == 1:
        return img.copy()
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    return np.clip(out, 0.0, 1.0)


def disk(radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def estimate_background(img: np.ndarray, fg: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth fabric colour under the print: normalised convolution of the non-print pixels,
    coarse-to-fine so large holes (the dragonfly) are filled from further away."""
    w = (~fg).astype(np.float32)
    bg = None
    for s in (sigma * 8, sigma * 4, sigma * 2, sigma):
        num = cv2.GaussianBlur(img * w[..., None], (0, 0), s)
        den = cv2.GaussianBlur(w, (0, 0), s)
        est = num / np.maximum(den, 1e-6)[..., None]
        if bg is None:
            bg = est
        else:
            t = np.clip(den / 0.3, 0.0, 1.0)[..., None]  # trust in the finer estimate
            bg = est * t + bg * (1.0 - t)
    return bg


def darkening_alpha(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """GIMP-style colour-to-alpha restricted to darkening: the smallest alpha for which
    img = a*F + (1-a)*bg has a valid F >= 0. Lighter-than-fabric pixels are webp ringing."""
    a = (bg - img) / np.maximum(bg, 1e-6)
    return np.clip(a.max(axis=2), 0.0, 1.0)


def hysteresis(a: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep pixels above `low` only when they are connected to a pixel above `high`."""
    n, lab = cv2.connectedComponents((a > low).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[a > high])] = True
    keep[0] = False
    return keep[lab]


def remap_alpha(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Noise floor below `lo`, fully opaque above `hi`, linear in between."""
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def unmix(img: np.ndarray, bg: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Foreground colour F from img = a*F + (1-a)*bg (linear light)."""
    f = bg + (img - bg) / np.maximum(a, 1e-4)[..., None]
    return np.clip(f, 0.0, 1.0)


def pad_colour(rgb: np.ndarray, a: np.ndarray, radius: float) -> np.ndarray:
    """Bleed the colour of visible pixels into transparent ones, so texture filtering and
    mip-mapping in Blender never pull the (meaningless) RGB of alpha=0 pixels into edges."""
    w = a.astype(np.float32)
    num = cv2.GaussianBlur(rgb * w[..., None], (0, 0), radius)
    den = cv2.GaussianBlur(w, (0, 0), radius)
    fill = num / np.maximum(den, 1e-6)[..., None]
    far = den < 1e-3
    if far.any():  # anything still empty gets the mean print colour
        mean = (rgb * w[..., None]).sum((0, 1)) / max(w.sum(), 1e-6)
        fill[far] = mean
    t = np.clip(a * 4.0, 0.0, 1.0)[..., None]
    return rgb * t + fill * (1.0 - t)


def extrude(rgba: np.ndarray, p0: np.ndarray, direction: np.ndarray, half_width: float,
            s_from: float, s_to: float) -> None:
    """Replace, inside a band around the axis p0 + s*direction, every pixel with s > s_from by
    the cross section found at s = s_from (in place, premultiplied RGBA)."""
    h, w = rgba.shape[:2]
    d = direction / np.linalg.norm(direction)
    n = np.array([-d[1], d[0]])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rel_x, rel_y = xx - p0[0], yy - p0[1]
    s = rel_x * d[0] + rel_y * d[1]
    t = rel_x * n[0] + rel_y * n[1]
    band = (np.abs(t) <= half_width) & (s > s_from) & (s <= s_to)
    src_x = p0[0] + s_from * d[0] + t * n[0]
    src_y = p0[1] + s_from * d[1] + t * n[1]
    sampled = cv2.remap(rgba, src_x.astype(np.float32), src_y.astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # soft 1px feather across the band edge so the extrusion does not show a hard border
    feather = np.clip(half_width + 0.5 - np.abs(t), 0.0, 1.0)[..., None]
    rgba[band] = (sampled * feather + rgba * (1.0 - feather))[band]


def fit_axis(a: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-weighted line fit x = m*y + c over a window; returns (point, unit direction)."""
    ys, xs = np.mgrid[y0:y1, x0:x1]
    wts = a[y0:y1, x0:x1] ** 2
    ys, xs, wts = ys.ravel(), xs.ravel(), wts.ravel()
    m, c = np.polyfit(ys, xs, 1, w=np.sqrt(wts) + 1e-6)
    d = np.array([m, 1.0])
    return np.array([m * y0 + c, float(y0)]), d / np.linalg.norm(d)


def save_rgba(path: Path, rgb_lin: np.ndarray, a: np.ndarray) -> None:
    rgb8 = np.round(linear_to_srgb(rgb_lin) * 255).astype(np.uint8)
    a8 = np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(np.dstack([rgb8, a8]), "RGBA").save(path, optimize=True)


def crop_to_content(rgb: np.ndarray, a: np.ndarray, margin: int, keep_bottom: bool = False):
    ys, xs = np.nonzero(a > 1.0 / 255)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, a.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, a.shape[1])
    if keep_bottom:
        y1 = a.shape[0]
    return rgb[y0:y1, x0:x1], a[y0:y1, x0:x1], (x0, y0)


# --- front print ---------------------------------------------------------------------------
def extract_front(src_rgb: np.ndarray, src_a: np.ndarray, scale: int):
    x0, y0, x1, y1 = FRONT["box"]
    # canvas = crop + bleed to the right (rod past the side seam) and below the hem
    cw, ch = (x1 - x0) + BLEED_PX, (y1 - y0) + BLEED_PX
    crop = np.zeros((ch, cw, 3), np.float32)
    garment = np.zeros((ch, cw), np.float32)
    crop[: y1 - y0, : x1 - x0] = src_rgb[y0:y1, x0:x1]
    garment[: y1 - y0, : x1 - x0] = src_a[y0:y1, x0:x1]

    img = upscale(srgb_to_linear(crop), scale)
    inside = upscale(garment, scale) > 0.999
    inside = cv2.erode(inside.astype(np.uint8), disk(3 * scale)).astype(bool)
    k = float(scale)

    # 1) rough print mask -> 2) fabric estimate -> refine once with the alpha itself
    g8 = np.round(linear_to_srgb(img) * 255).astype(np.uint8)
    med = cv2.medianBlur(g8, 2 * int(25 * k) + 1)
    dev = np.abs(g8.astype(np.int16) - med.astype(np.int16)).max(axis=2)
    fg = cv2.dilate((dev > 10).astype(np.uint8), disk(10 * k)).astype(bool) | ~inside
    bg = estimate_background(img, fg, 10 * k)
    for _ in range(2):
        a0 = darkening_alpha(img, bg)
        fg = cv2.dilate((a0 > 0.04).astype(np.uint8), disk(8 * k)).astype(bool) | ~inside
        bg = estimate_background(img, fg, 10 * k)

    a0 = darkening_alpha(img, bg)
    a0[~inside] = 0.0
    keep = hysteresis(a0, low=0.05, high=0.30)
    a = remap_alpha(a0, lo=0.05, hi=0.88) * keep
    rgb = unmix(img, bg, a)

    # hem fold shading: nothing but the line survives below the clean line
    y_clean_line = int((FRONT["line_clean_y"] - y0) * k)
    a[y_clean_line:, :] = 0.0

    # 3) extrude the rod past the side seam and the line past the hem (premultiplied RGBA)
    rgba = np.dstack([rgb * a[..., None], a]).astype(np.float32)
    ry = int((FRONT["rod_clean_y"] - y0) * k)
    rod_p, rod_d = fit_axis(a, ry - int(90 * k), ry, int(160 * k), int(250 * k))
    extrude(rgba, rod_p, rod_d, half_width=7.5 * k, s_from=(ry - rod_p[1]) / rod_d[1],
            s_to=1e9)
    line_p, line_d = fit_axis(a, y_clean_line - int(150 * k), y_clean_line,
                              int(100 * k), int(125 * k))
    extrude(rgba, line_p, line_d, half_width=2.5 * k,
            s_from=(y_clean_line - 1 - line_p[1]) / line_d[1], s_to=1e9)
    a = rgba[..., 3]
    rgb = np.where(a[..., None] > 1e-4, rgba[..., :3] / np.maximum(a, 1e-4)[..., None], 0)

    rgb, a, (ox, oy) = crop_to_content(rgb, a, margin=int(6 * k), keep_bottom=True)
    rgb = pad_colour(rgb, a, radius=4 * k)

    # placement in the mockup, in units of the garment length (HPS -> hem)
    L = FRONT["hem_y"] - FRONT["hps_y"]
    px0, py0 = x0 + ox / k, y0 + oy / k
    px1, py1 = px0 + a.shape[1] / k, py0 + a.shape[0] / k
    place = dict(
        panel="front",
        left_of_center=(px0 - FRONT["center_x"]) / L,
        right_of_center=(px1 - FRONT["center_x"]) / L,
        top_below_hps=(py0 - FRONT["hps_y"]) / L,
        bottom_below_hps=(py1 - FRONT["hps_y"]) / L,
        hem_below_hps=1.0,
        note="x > 0 is the wearer's left (viewer's right); units = garment length HPS->hem",
    )
    return rgb, a, place


# --- back print ----------------------------------------------------------------------------
def extract_back(src_rgb: np.ndarray, scale: int, ink_hex: str | None):
    """Flat one-colour logo: alpha is ink *coverage*. The mockup was composited in gamma
    space (its anti-aliased pixels sit on a straight sRGB line between fabric and ink), so
    coverage is measured on sRGB luminance, at the mockup's resolution, then upscaled and
    re-sharpened into clean ink edges."""
    x0, y0, x1, y1 = BACK["box"]
    img = src_rgb[y0:y1, x0:x1]  # sRGB-encoded
    Y = luminance(img)

    # fabric: everything clearly lighter than the ink, smoothed
    g = np.round(Y * 255).astype(np.uint8)
    med = cv2.medianBlur(g, 41).astype(np.int16)
    fg = cv2.dilate(((med - g.astype(np.int16)) > 8).astype(np.uint8), disk(4)).astype(bool)
    bg = estimate_background(img, fg, 8)
    Yb = luminance(bg)

    if ink_hex:
        ink_srgb = linear_to_srgb(hex_to_linear(ink_hex))
    else:
        # solid ink = the dominant luminance among clearly-inked pixels (the few darker ones
        # are webp ringing, the lighter ones anti-aliasing)
        inked = Y < 0.5 * Yb
        hist, edges = np.histogram(Y[inked], bins=40, range=(0.0, 0.5))
        peak = edges[np.argmax(hist)] + 0.5 * (edges[1] - edges[0])
        solid = inked & (np.abs(Y - peak) <= 0.02)
        ink_srgb = np.median(img[solid], axis=0)
    a0 = np.clip((Yb - Y) / np.maximum(Yb - luminance(ink_srgb), 1e-6), 0.0, 1.0)
    keep = hysteresis(a0, low=0.06, high=0.35)
    a = remap_alpha(a0, lo=0.06, hi=0.92) * keep

    # upscale the coverage, then steepen it -> crisp, print-like ink edges. The contour sits
    # at ~35 % coverage so the hairlines of the script and the small caps are not eaten.
    a = upscale(a, scale)
    t = np.clip((a - 0.1) / 0.5, 0.0, 1.0)
    a = t * t * (3.0 - 2.0 * t)

    k = float(scale)
    ink = srgb_to_linear(np.asarray(ink_srgb, np.float32))
    rgb = np.broadcast_to(ink, a.shape + (3,)).astype(np.float32).copy()
    rgb, a, (ox, oy) = crop_to_content(rgb, a, margin=int(8 * k))

    L = BACK["hem_y"] - BACK["hps_y"]
    px0, py0 = x0 + ox / k, y0 + oy / k
    px1, py1 = px0 + a.shape[1] / k, py0 + a.shape[0] / k
    place = dict(
        panel="back",
        left_of_center=(px0 - BACK["center_x"]) / L,
        right_of_center=(px1 - BACK["center_x"]) / L,
        top_below_hps=(py0 - BACK["hps_y"]) / L,
        bottom_below_hps=(py1 - BACK["hps_y"]) / L,
        hem_below_hps=1.0,
        ink=linear_to_hex(ink),
        note="seen from behind; units = garment length HPS->hem",
    )
    return rgb, a, place


# --- preview sheet -------------------------------------------------------------------------
def preview(rgb: np.ndarray, a: np.ndarray, path: Path, long_side: int = 1400) -> None:
    """The print over a checkerboard, a dark grey and the fabric colour. Tall prints are laid
    out side by side, wide ones stacked."""
    tall = a.shape[0] >= a.shape[1]
    s = long_side / max(a.shape)
    size = (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s)))
    rgb_s = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    a_s = np.clip(cv2.resize(a, size, interpolation=cv2.INTER_AREA), 0, 1)[..., None]
    h, w = a_s.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // 16) + (xx // 16)) % 2 == 0, 0.8, 0.55).astype(np.float32)
    backgrounds = (np.repeat(checker[..., None], 3, axis=2), hex_to_linear("#1E1E1E"),
                   hex_to_linear(FABRIC_HEX))
    gap = np.ones((h, 24, 3) if tall else (24, w, 3), np.float32)
    tiles = []
    for bgc in backgrounds:
        tiles += [rgb_s * a_s + np.broadcast_to(bgc, rgb_s.shape) * (1 - a_s), gap]
    sheet = np.concatenate(tiles[:-1], axis=1 if tall else 0)
    Image.fromarray(np.round(linear_to_srgb(sheet) * 255).astype(np.uint8)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "Tee_1.png"))
    ap.add_argument("--scale", type=int, default=2, help="upscale of the front photo print")
    ap.add_argument("--back-scale", type=int, default=4,
                    help="upscale of the back logo (flat ink -> alpha upscales cleanly)")
    ap.add_argument("--ink", default=None, help="force the back logo ink colour, e.g. #5E1A20")
    ap.add_argument("--out", default=str(ROOT / "assets" / "prints"))
    args = ap.parse_args()

    src = Image.open(args.src).convert("RGBA")
    if src.size != (2000, 2000):
        raise SystemExit(f"expected the 2000x2000 mockup, got {src.size}")
    arr = np.asarray(src).astype(np.float32) / 255.0
    src_rgb, src_a = arr[..., :3], arr[..., 3]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f_rgb, f_a, f_place = extract_front(src_rgb, src_a, args.scale)
    b_rgb, b_a, b_place = extract_back(src_rgb, args.back_scale, args.ink)
    save_rgba(out / "print_front.png", f_rgb, f_a)
    save_rgba(out / "print_back.png", b_rgb, b_a)

    meta = {
        "source": Path(args.src).name,
        "front": {"file": "print_front.png", "size_px": [f_a.shape[1], f_a.shape[0]],
                  "upscale": args.scale, **f_place},
        "back": {"file": "print_back.png", "size_px": [b_a.shape[1], b_a.shape[0]],
                 "upscale": args.back_scale, **b_place},
    }
    (out / "prints.json").write_text(json.dumps(meta, indent=2) + "\n")

    (ROOT / "renders").mkdir(exist_ok=True)
    preview(f_rgb, f_a, ROOT / "renders" / "print_front_check.png")
    preview(b_rgb, b_a, ROOT / "renders" / "print_back_check.png")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
