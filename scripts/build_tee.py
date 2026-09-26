"""Yacht Club Morocco - Sport Fishing tee: photoreal 3D build, fully headless.

    blender -b -P scripts/build_tee.py -- [options]
    python  scripts/build_tee.py [options]          # with the `bpy` module from PyPI

Steps: ghost mannequin -> pattern pieces (front, back, 2 sleeves, neck rib) -> cloth
simulation with sewing springs -> welded garment + Subdivision + Solidify -> materials
(heavy cotton jersey, rib, stitching, prints) -> studio lighting -> renders.
Writes output/tee.blend and renders/*.png.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import bpy  # noqa: E402

from tee3d import garment, mannequin, materials, pattern, scene as studio_scene, textures  # noqa: E402
from tee3d.util import clear_scene, collection, log  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(prog="build_tee.py", description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=None, help="cloth simulation length")
    ap.add_argument("--quality", choices=("preview", "final"), default="final",
                    help="render samples / resolution (preview: 1000 px, 48 spp)")
    ap.add_argument("--views", default="front,back,three_quarter,detail",
                    help="comma separated: front, back, three_quarter, detail")
    ap.add_argument("--no-render", action="store_true", help="build and save the .blend only")
    ap.add_argument("--debug-renders", action="store_true",
                    help="quick workbench renders of the drape in renders/_debug")
    ap.add_argument("--blend", default=str(ROOT / "output" / "tee.blend"))
    ap.add_argument("--renders", default=str(ROOT / "renders"))
    return ap.parse_args(argv)


def debug_render(objs, name: str) -> None:
    """Fast solid-shaded front / side / back views, to check the drape."""
    from mathutils import Vector
    out = ROOT / "renders" / "_debug"
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.render.resolution_x, scene.render.resolution_y = 900, 1000
    cam = bpy.data.objects.get("DebugCam")
    if cam is None:
        cam = bpy.data.objects.new("DebugCam", bpy.data.cameras.new("DebugCam"))
        scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 1.25
    scene.camera = cam
    target = Vector((0.0, 0.0, 1.13))
    hidden = [o for o in objs if o.hide_render]
    for o in hidden:
        o.hide_render = False
    for view, direction in (("front", (0, -1, 0)), ("side", (1, 0, 0)), ("back", (0, 1, 0)),
                            ("top", (0.0001, -0.0001, 1))):
        d = Vector(direction).normalized()
        cam.location = target + d * 4.0
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out / f"{name}_{view}.png")
        bpy.ops.render.render(write_still=True)
    for o in hidden:
        o.hide_render = True


def main():
    args = parse_args()
    clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    sim_col = collection("Simulation")
    tee_col = collection("Tee")

    rig = mannequin.Rig()
    body = mannequin.build(rig, sim_col)
    body.color = (0.35, 0.55, 0.8, 1.0)

    pat = pattern.Pattern()
    params = garment.ClothParams()
    if args.frames:
        params.frames = args.frames
    prints_json = ROOT / "assets" / "prints" / "prints.json"

    # pass 1: sew front, back and sleeves on the mannequin and let them drape
    body_names = [n for n in garment.ORDER if not n.startswith("rib")]
    start = garment.starting_positions(pat, rig)
    sim1, _ = garment.build_sim_object(pat, body_names, start, sim_col, prints_json, "Tee_sim_pass1",
                                       body_pressure=params.pressure_body)
    garment.setup_cloth(sim1, params)
    if args.debug_renders:
        debug_render([sim1, body], "start")
    garment.simulate(sim1, params)
    if args.debug_renders:
        debug_render([sim1, body], "pass1")

    # pass 2: stand the neck rib on the draped neckline, sew it, let everything settle
    pos = garment.piece_positions(sim1, pat, body_names)
    pos.update(garment.sweep_rib(pat, rig, pos))
    bpy.data.objects.remove(sim1)
    sim, info = garment.build_sim_object(pat, garment.ORDER, pos, sim_col, prints_json,
                                         weld_virtual=True, body_pressure=params.pressure_body)
    settle = dataclasses.replace(params, frames=params.settle_frames, sew_frames=0)
    garment.setup_cloth(sim, settle)
    garment.simulate(sim, settle)
    sim.color = (0.95, 0.93, 0.88, 1.0)
    if args.debug_renders:
        debug_render([sim, body], "pass2")

    tee, stats = garment.build_tee(sim, tee_col)
    tee.color = (0.95, 0.93, 0.88, 1.0)
    if args.debug_renders:
        debug_render([tee], "tee")

    # --- step 3: materials ---
    tex = textures.generate(ROOT / "assets" / "textures")
    prints = {k: ROOT / "assets" / "prints" / f"print_{k}.png" for k in ("front", "back")}
    prints = {k: v for k, v in prints.items() if v.exists()}
    materials.assign(tee, tex, prints, info["atlas_size_cm"], pat.spec)

    # --- studio, cameras, render settings ---
    for o in list(bpy.data.objects):
        if o.name == "DebugCam":
            bpy.data.objects.remove(o)
    studio = studio_scene.build_studio(tee)
    studio_scene.setup_render(args.quality)
    views = {"front": (0.0, 6.0), "back": (180.0, 6.0), "three_quarter": (-35.0, 10.0)}
    cams = {v: studio_scene.camera(f"Cam_{v}", studio, *views[v]) for v in views}
    cams["detail"] = studio_scene.detail_camera("Cam_detail", studio, rig.hps_z * 0.01)
    views["detail"] = (0.0, 0.0)
    scene.camera = cams["front"]
    studio_scene.calibrate_exposure(cams["front"], studio)

    # keep the simulation set-up in the file (re-simulable), but with an empty cache, out of
    # the way and not re-evaluated on load
    sim.modifiers.remove(sim.modifiers["Cloth"])
    garment.setup_cloth(sim, settle)
    scene.frame_current = 1
    sim.hide_render = True
    vl = bpy.context.view_layer
    vl.layer_collection.children[sim_col.name].exclude = True
    blend = Path(args.blend).resolve()
    blend.parent.mkdir(parents=True, exist_ok=True)
    for img in bpy.data.images:  # textures/prints stay next to the project, not packed
        if img.filepath and not img.filepath.startswith("//"):
            img.filepath = bpy.path.relpath(bpy.path.abspath(img.filepath), start=str(blend.parent))
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)
    log(f"saved {args.blend}")

    if args.no_render:
        return
    out = Path(args.renders)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {"front": 2000, "back": 2000, "three_quarter": 1600, "detail": 1600}
    scale = 0.5 if args.quality == "preview" else 1.0
    done = {}
    for v in [v.strip() for v in args.views.split(",") if v.strip()]:
        px = int(sizes[v] * scale)
        done[v] = studio_scene.render_view(cams[v], studio, views[v][0], out / f"tee_{v}.png",
                                           (px, px))
        if v != "detail":
            studio_scene.on_background(done[v], out / f"tee_{v}_white.png")
    if "front" in done and "back" in done:
        studio_scene.mockup(done["front"], done["back"], out / "tee_mockup.png")
        log(f"composite {out / 'tee_mockup.png'}")
    studio["rig"].rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = cams["front"]


if __name__ == "__main__":
    main()
