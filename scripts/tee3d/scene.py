"""Studio set-up: soft product lighting, cameras, Cycles settings, renders and the final
front/back composite laid out like the reference mockup."""
from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from .textures import write_png
from .util import log

BACKGROUND_SRGB = (1.0, 1.0, 1.0)  # e-commerce white, like the mockup


def _look_at(obj, target: Vector):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _area(name, rig, loc, target, size, energy, shape="RECTANGLE", size_y=None, color=(1, 1, 1)):
    data = bpy.data.lights.new(name, "AREA")
    data.shape = shape
    data.size = size
    if size_y:
        data.size_y = size_y
    data.energy = energy
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    _look_at(obj, target)
    obj.parent = rig
    return obj


def setup_render(quality: str) -> None:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = {"preview": 48, "final": 256}[quality]
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.015 if quality == "final" else 0.03
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces = 10, 5, 4
    cy.transparent_max_bounces = 8
    cy.caustics_reflective = cy.caustics_refractive = False
    cy.sample_clamp_indirect = 8.0
    sc.render.film_transparent = True
    sc.render.resolution_percentage = 100
    vs = sc.view_settings
    for vt in ("Khronos PBR Neutral", "AgX", "Standard"):  # colour-faithful first
        try:
            vs.view_transform = vt
            break
        except TypeError:
            continue
    vs.look = "None"
    vs.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"


def build_studio(tee: bpy.types.Object) -> dict:
    """Soft key/fill/top/rim rig parented to an empty, so it turns with the camera."""
    sc = bpy.context.scene
    world = bpy.data.worlds.new("Studio")
    sc.world = world
    if bpy.app.version < (5, 0, 0) or world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is None:  # make sure there is a Background -> World Output chain
        nt.nodes.clear()
        bg = nt.nodes.new("ShaderNodeBackground")
        nt.links.new(bg.outputs[0], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    bg.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    bg.inputs["Strength"].default_value = 0.12

    bb = [tee.matrix_world @ Vector(c) for c in tee.bound_box]
    lo = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    hi = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    centre = (lo + hi) / 2
    rig = bpy.data.objects.new("LightRig", None)
    sc.collection.objects.link(rig)
    rig.location = (centre.x, centre.y, 0.0)
    t = Vector((0.0, 0.0, centre.z))
    # powers give the lit fabric a scene-linear value of ~1; calibrate_exposure() trims it
    _area("Key", rig, (-2.0, -2.6, centre.z + 1.3), t, 2.2, 115.0, size_y=2.2,
          color=(1.0, 0.985, 0.96))
    _area("Fill", rig, (2.6, -2.2, centre.z + 0.2), t, 2.8, 45.0, size_y=2.8,
          color=(0.97, 0.985, 1.0))
    _area("Top", rig, (0.0, -0.6, centre.z + 2.4), t, 1.6, 40.0, size_y=1.6)
    _area("RimL", rig, (-1.6, 1.9, centre.z + 0.9), t, 0.5, 28.0, size_y=2.0)
    _area("RimR", rig, (1.6, 1.9, centre.z + 0.9), t, 0.5, 28.0, size_y=2.0)
    _area("Bounce", rig, (0.0, -1.6, centre.z - 1.6), t, 2.5, 10.0, size_y=1.2)
    return dict(rig=rig, centre=centre, lo=lo, hi=hi)


def camera(name: str, studio: dict, azimuth_deg: float, elevation_deg: float, lens=85.0,
           margin=1.12, aspect=1.0) -> bpy.types.Object:
    """Camera orbiting the garment; azimuth 0 = front (camera on -Y)."""
    sc = bpy.context.scene
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        sc.collection.objects.link(cam)
    cam.data.lens = lens
    cam.data.sensor_width = 36.0
    lo, hi, c = studio["lo"], studio["hi"], studio["centre"]
    size = max(hi.x - lo.x, hi.y - lo.y, (hi.z - lo.z) * aspect)
    half_fov = math.atan(18.0 / lens)
    dist = margin * 0.5 * size / math.tan(half_fov) + 0.5 * (hi.y - lo.y)
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    target = Vector((c.x, c.y, c.z + 0.01))
    cam.location = target + d * dist
    _look_at(cam, target)
    return cam


def detail_camera(name: str, studio: dict, hps_z_m: float) -> bpy.types.Object:
    """Close-up on the neck rib and the dragonfly (front, slightly from the right)."""
    sc = bpy.context.scene
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        sc.collection.objects.link(cam)
    cam.data.lens = 70.0
    cam.data.sensor_width = 36.0
    target = Vector((0.06, studio["lo"].y + 0.05, hps_z_m - 0.13))
    az, el = math.radians(16.0), math.radians(9.0)
    d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    cam.location = target + d * 0.74
    _look_at(cam, target)
    return cam


def calibrate_exposure(cam, studio: dict, target: float = 1.0, percentile: float = 90.0,
                       tmp_dir: Path | None = None) -> float:
    """Tiny linear (EXR) render of the garment; set the view exposure so that its lit fabric
    (given luminance percentile) lands on `target` scene-linear. With the Khronos PBR Neutral
    view transform this shows the #F2EFE6 fabric as itself in the lit areas."""
    import tempfile
    sc = bpy.context.scene
    r, cy, fmt = sc.render, sc.cycles, sc.render.image_settings
    saved = (r.resolution_x, r.resolution_y, cy.samples, cy.use_denoising, fmt.file_format,
             fmt.color_depth, r.filepath, sc.camera, studio["rig"].rotation_euler.copy())
    tmp = Path(tmp_dir or tempfile.gettempdir()) / "tee_exposure.exr"
    sc.camera = cam
    studio["rig"].rotation_euler = (0.0, 0.0, 0.0)
    r.resolution_x = r.resolution_y = 200
    cy.samples, cy.use_denoising = 16, False
    fmt.file_format, fmt.color_depth = "OPEN_EXR", "32"
    r.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    (r.resolution_x, r.resolution_y, cy.samples, cy.use_denoising, fmt.file_format,
     fmt.color_depth, r.filepath, sc.camera, studio["rig"].rotation_euler) = saved
    px = _load_rgba(tmp)
    tmp.unlink(missing_ok=True)
    lum = px[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    level = float(np.percentile(lum[px[..., 3] > 0.99], percentile))
    sc.view_settings.exposure = math.log2(target / max(level, 1e-6))
    log(f"exposure: fabric p{percentile:.0f} = {level:.3f} -> {sc.view_settings.exposure:+.2f} EV")
    return sc.view_settings.exposure


def render_view(cam, studio: dict, azimuth_deg: float, path: Path, res=(2000, 2000)) -> Path:
    sc = bpy.context.scene
    sc.camera = cam
    studio["rig"].rotation_euler = (0.0, 0.0, math.radians(azimuth_deg))
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.filepath = str(path)
    log(f"render {path.name} ({res[0]}x{res[1]}, {sc.cycles.samples} spp)")
    bpy.ops.render.render(write_still=True)
    return path


# --- compositing (numpy only) ----------------------------------------------------------------
def _load_rgba(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)  # row 0 = bottom, values as stored (sRGB)


def _resample_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Area-weighted (box) resampling matrix n_out x n_in, for down-sizing."""
    edges = np.linspace(0.0, n_in, n_out + 1)
    m = np.zeros((n_out, n_in), np.float32)
    for o in range(n_out):
        a, b = edges[o], edges[o + 1]
        i0, i1 = int(np.floor(a)), min(int(np.ceil(b)), n_in)
        for i in range(i0, i1):
            m[o, i] = min(b, i + 1) - max(a, i)
        m[o] /= m[o].sum()
    return m


def _resize(rgba: np.ndarray, size: int) -> np.ndarray:
    """Down-size a straight-alpha RGBA image in premultiplied space (no dark fringes)."""
    h, w = rgba.shape[:2]
    ry, rx = _resample_matrix(h, size), _resample_matrix(w, size)
    pre = np.concatenate([rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]], axis=2)
    out = np.stack([ry @ pre[..., c] @ rx.T for c in range(4)], axis=2)
    a = out[..., 3:]
    out[..., :3] = np.where(a > 1e-5, out[..., :3] / np.maximum(a, 1e-5), 0.0)
    return out


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    a = np.apply_along_axis(lambda r: np.convolve(r, g, mode="same"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(c, g, mode="same"), 0, a)


def _shadow_over(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, strength=0.16) -> None:
    """Paste `rgba` at (x, y) (bottom-left origin) with a soft drop shadow, in place."""
    h, w = rgba.shape[:2]
    a = rgba[..., 3]
    small = a[::4, ::4]
    sh = _blur(small, 9.0)
    sh = np.kron(sh, np.ones((4, 4)))[:h, :w]
    dy = int(0.012 * h)
    region = canvas[y - dy:y - dy + h, x:x + w]
    region[...] = region * (1.0 - strength * sh[..., None])
    region = canvas[y:y + h, x:x + w]
    region[...] = rgba[..., :3] * a[..., None] + region * (1.0 - a[..., None])


def on_background(src: Path, dst: Path) -> None:
    rgba = _load_rgba(src)
    canvas = np.ones(rgba.shape[:2] + (3,), np.float32) * np.array(BACKGROUND_SRGB, np.float32)
    pad = 60
    big = np.ones((rgba.shape[0] + 2 * pad, rgba.shape[1] + 2 * pad, 3), np.float32) * canvas[0, 0]
    _shadow_over(big, rgba, pad, pad)
    write_png(dst, big[pad:-pad, pad:-pad], bits=8)


def mockup(front: Path, back: Path, dst: Path, size=2000) -> None:
    """Front and back renders arranged like the reference Tee_1.png: back view top-right,
    front view bottom-left and overlapping it."""
    canvas = np.ones((size, size, 3), np.float32) * np.array(BACKGROUND_SRGB, np.float32)
    s = int(size * 0.62)
    fb, ff = _resize(_load_rgba(back), s), _resize(_load_rgba(front), s)
    _shadow_over(canvas, fb, int(size * 0.36), int(size * 0.36))
    _shadow_over(canvas, ff, int(size * 0.04), int(size * 0.03))
    write_png(dst, canvas, bits=8)
