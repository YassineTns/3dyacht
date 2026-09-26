"""Sew the pattern on the mannequin with Blender's cloth solver, then build the final tee.

* All simulation pieces live in ONE mesh object; each seam is a row of loose edges that the
  cloth solver turns into sewing springs.
* The Basis shape is the 3D starting position (flat front/back panels slightly curved around
  the body, sleeves lofted from the armholes to a ring around each arm, neck rib standing on
  the neckline). A "Flat" shape key holds the 2D pattern and is used as the cloth REST shape,
  so the fabric wants to recover the pattern dimensions, not the starting layout.
* After the simulation the seams are welded, giving one garment mesh with per-panel UVs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
import numpy as np

from .mannequin import Rig
from .pattern import Pattern
from .util import (CM, face_attr, log, mesh_object, point_attr, set_loop_uvs, set_smooth,
                   verts_co, evaluated_mesh_copy)

ORDER = ["front", "back", "sleeve_l_front", "sleeve_l_back", "sleeve_r_front", "sleeve_r_back",
         "rib_front", "rib_back"]
PATTERN_IDS = {"front": 0, "back": 1, "sleeve_l": 2, "sleeve_r": 3, "rib": 4}
NO_UV = -4.0  # print UV for faces that do not carry the print (outside the CLIP texture)


@dataclass
class ClothParams:
    frames: int = 110             # pass 1: sewing + draping (body and sleeves)
    settle_frames: int = 40       # pass 2: neck rib added, everything settles
    quality: int = 12
    mass: float = 0.35            # heavier than the 'cotton' preset: 260 g/m2 jersey
    tension: float = 25.0
    compression: float = 25.0
    shear: float = 12.0
    bending: float = 3.0          # thick jersey keeps soft, round folds
    rib_stiffness: float = 80.0   # neck rib and hems are double layers: stiffer
    rib_bending: float = 25.0
    air_damping: float = 2.0
    sewing_force: float = 12.0
    sew_frames: int = 22          # gravity off while the seams close...
    gravity_full: int = 40        # ...then ramps to full gravity at this frame
    collision_distance: float = 0.004
    self_distance: float = 0.0025
    pressure: float = 6.0         # uniform inside pressure: fills the sleeves like a ghost
                                  # mannequin (x kPa pressure scale: ~30 % of gravity here)
    pressure_body: float = 0.3    # relative pressure on front/back (rib: none)


# --- 3D starting positions -----------------------------------------------------------------
def _place_body(grid: np.ndarray, rig: Rig, back: bool) -> np.ndarray:
    """Flat panel bent on a large cylinder, in front of (or behind) the mannequin."""
    x, y = grid[..., 0], grid[..., 1]
    R = rig.wrap_radius
    X = R * np.sin(x / R)
    Y = -rig.panel_depth + R * (1.0 - np.cos(x / R))
    if back:  # the back panel is drawn seen from behind: its +x is the world -X
        X, Y = -X, -Y
    return np.stack([X, Y, rig.hps_z + y], axis=-1)


def _seam(pat: Pattern, name: str):
    return next(s for s in pat.seams if s.name == name)


def _hem_ring(pat: Pattern, rig: Rig, sx: float, shoulder_pt, underarm_pt):
    """Ring around the arm where each sleeve starts its hem. It slides along the arm and
    tilts so that the straight lines shoulder point -> ring top and underarm -> ring bottom
    have the pattern lengths of the sleeve fold and of the underarm seam: no fabric starts
    compressed under the arm (that is what crumpled into flaps at the sleeve hem)."""
    joint, d, up = rig.arm_frame(sx)
    r = pat.spec.sleeve_opening / (2 * math.pi)
    l_top, l_under = pat.spec.sleeve_length, pat.underarm_len
    best = None
    for t in np.arange(12.0, rig.arm_length, 0.25):
        r_arm = rig.arm_r0 + (rig.arm_r1 - rig.arm_r0) * t / rig.arm_length
        for beta in np.radians(np.arange(-45.0, 25.0, 1.0)):
            if r * math.cos(beta) < r_arm + 1.0:  # the tilted ring must clear the arm
                continue
            e1 = math.cos(beta) * up + math.sin(beta) * d
            c = joint + t * d
            err = ((np.linalg.norm(shoulder_pt - (c + r * e1)) - l_top) ** 2
                   + (np.linalg.norm(underarm_pt - (c - r * e1)) - l_under) ** 2)
            if best is None or err < best[0]:
                best = (err, t, math.degrees(beta), c, e1)
    err, t, beta, c, e1 = best
    log(f"sleeve hem ring: {t:.1f} cm down the arm, tilted {beta:+.0f} deg, "
        f"length error {math.sqrt(err):.1f} cm")
    return c, e1, r


def starting_positions(pat: Pattern, rig: Rig) -> dict[str, np.ndarray]:
    P = pat.pieces
    pos = {"front": _place_body(P["front"].grid, rig, back=False),
           "back": _place_body(P["back"].grid, rig, back=True)}
    flat = {k: v.reshape(-1, 3) for k, v in pos.items()}

    # sleeves: loft each half from its armhole to half of the hem ring
    for side, sx in (("l", 1.0), ("r", -1.0)):
        armhole = flat["front"][_seam(pat, f"armhole_sleeve_{side}_front").ia]  # underarm -> shoulder
        centre, e1, r_hem = _hem_ring(pat, rig, sx, armhole[-1], armhole[0])
        for role in ("front", "back"):
            pc = P[f"sleeve_{side}_{role}"]
            sm = _seam(pat, f"armhole_sleeve_{side}_{role}")
            body = flat[sm.a]
            cap = np.zeros((pc.nu + 1, 3))
            for ia, ib in zip(sm.ia, sm.ib):
                cap[ib - pc.idx(pc.nv, 0)] = body[ia]
            i = np.arange(pc.nu + 1) / pc.nu
            eps = 0.05  # keep the halves' edges ~4 mm apart at the start
            t = (1.0 - i) if pc.marks["half"] == "a" else i
            phi = eps + (np.pi - 2 * eps) * t
            e_side = np.array([0.0, -1.0 if role == "front" else 1.0, 0.0])
            hem = centre + r_hem * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e_side)
            # keep the cap 0.5 cm off the armhole so the two rows do not start coincident
            gap = hem - cap
            cap = cap + 0.5 * gap / np.linalg.norm(gap, axis=1, keepdims=True)
            w = (np.arange(pc.nv + 1) / pc.nv)[:, None, None]
            pos[pc.name] = (1.0 - w) * hem[None] + w * cap[None]

    return pos


def sweep_rib(pat: Pattern, rig: Rig, pos: dict[str, np.ndarray], lean: float = 0.45,
              gap_cm: float = 0.25) -> dict[str, np.ndarray]:
    """Neck rib standing on the DRAPED neckline: it continues the body surface upwards and
    leans towards the neck, like a real crew-neck band."""
    out = {}
    for part in ("front", "back"):
        body = pat.pieces[part]
        g = pos[part].reshape(body.nv + 1, body.nu + 1, 3)
        i0, i1 = body.marks["neck"]
        neck, below = g[-1, i0:i1 + 1], g[-3, i0:i1 + 1]
        up = neck - below
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        inward = np.array([0.0, 0.0, 0.0]) - neck
        inward[:, 2] = 0.0
        inward /= np.linalg.norm(inward, axis=1, keepdims=True)
        d = up + lean * inward
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rib = pat.pieces[f"rib_{part}"]
        h = gap_cm + pat.spec.rib_height * (np.arange(rib.nv + 1) / rib.nv)
        out[rib.name] = neck[None] + h[:, None, None] * d[None]
    return out


def piece_positions(obj, pat: Pattern, names) -> dict[str, np.ndarray]:
    """Current (evaluated) vertex positions of each piece of a simulation object, in cm."""
    me = evaluated_mesh_copy(obj, "_tmp")
    co = verts_co(me) / CM
    bpy.data.meshes.remove(me)
    out, n = {}, 0
    for name in names:
        pc = pat.pieces[name]
        k = (pc.nv + 1) * (pc.nu + 1)
        out[name] = co[n:n + k].reshape(pc.nv + 1, pc.nu + 1, 3)
        n += k
    return out


# --- atlas / UV layout ---------------------------------------------------------------------
def atlas_layout(pat: Pattern, margin: float = 3.0):
    """Shelf-pack the 5 pattern pieces; returns ({pattern: (dx, dy)} in cm, atlas size cm)."""
    boxes = {}
    for pc in pat.pieces.values():
        g = pc.grid.reshape(-1, 2)
        lo, hi = g.min(0), g.max(0)
        if pc.pattern in boxes:
            lo = np.minimum(lo, boxes[pc.pattern][0])
            hi = np.maximum(hi, boxes[pc.pattern][1])
        boxes[pc.pattern] = (lo, hi)
    rows = [["front", "back"], ["sleeve_l", "sleeve_r"], ["rib"]]
    off, y, width = {}, margin, 0.0
    for row in rows:
        x, h = margin, 0.0
        for name in row:
            lo, hi = boxes[name]
            off[name] = np.array([x - lo[0], y - lo[1]])
            x += (hi - lo)[0] + margin
            h = max(h, (hi - lo)[1])
        width = max(width, x)
        y += h + margin
    return off, max(width, y)


def print_rects(spec, prints_json: Path | None):
    """Print rectangles in pattern cm: {panel: (x0, x1, y0, y1)} from assets/prints/prints.json."""
    if not prints_json or not prints_json.exists():
        return {}
    meta = json.loads(prints_json.read_text())
    L = spec.length
    out = {}
    for panel in ("front", "back"):
        m = meta[panel]
        out[panel] = (m["left_of_center"] * L, m["right_of_center"] * L,
                      -m["bottom_below_hps"] * L, -m["top_below_hps"] * L)
    return out


def _neck_distance(pts: np.ndarray, neck: np.ndarray):
    """Distance (cm) of points to the neckline polyline, and arc length of the closest point."""
    a, b = neck[:-1], neck[1:]
    ab = b - a
    seg_len = np.linalg.norm(ab, axis=1)
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(-1) / (seg_len ** 2)[None], 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    dist = np.linalg.norm(pts[:, None] - closest, axis=-1)
    k = dist.argmin(1)
    s0 = np.concatenate([[0.0], np.cumsum(seg_len)])
    return dist[np.arange(len(pts)), k], s0[k] + t[np.arange(len(pts)), k] * seg_len[k]


# --- simulation mesh -----------------------------------------------------------------------
def build_sim_object(pat: Pattern, names, start: dict, col, prints_json: Path | None,
                     name: str = "Tee_sim", weld_virtual: bool = False, body_pressure: float = 0.3):
    """One cloth object for the given pieces. Seams become loose (sewing) edges. With
    weld_virtual, halves of the same pattern piece are merged along their fold line: both
    sides share the same flat rest coordinates there, so the fabric is continuous (it has
    bending stiffness across the fold instead of acting like a hinge)."""
    spec = pat.spec
    atlas_off, atlas_size = atlas_layout(pat)
    rects = print_rects(spec, prints_json)

    verts, rest, faces, face_piece = [], [], [], []
    uv_pat, uv_front, uv_back, uv_hem, uv_neck, thick, stiff = [], [], [], [], [], [], []
    press = []
    offset = {}
    n = 0
    for piece in names:
        pc = pat.pieces[piece]
        g = pc.grid.reshape(-1, 2)
        offset[piece] = n
        verts.append(start[piece].reshape(-1, 3))
        rest.append(np.concatenate([g + atlas_off[pc.pattern], np.zeros((len(g), 1))], axis=1))
        f = pc.faces() + n
        faces.append(f)
        face_piece.append(np.full(len(f), PATTERN_IDS[pc.pattern]))
        uv_pat.append((g + atlas_off[pc.pattern]) / atlas_size)
        for panel, store in (("front", uv_front), ("back", uv_back)):
            if pc.pattern == panel and panel in rects:
                x0, x1, y0, y1 = rects[panel]
                store.append(np.stack([(g[:, 0] - x0) / (x1 - x0), (g[:, 1] - y0) / (y1 - y0)], 1))
            else:
                store.append(np.full((len(g), 2), NO_UV))
        # detail coordinates for the stitching: (along the edge, distance to the edge) in cm
        far = np.full(len(g), 100.0)
        if pc.pattern in ("front", "back"):
            hem_d = g[:, 1] + spec.length
            uv_hem.append(np.stack([g[:, 0], hem_d], 1))
            i0, i1 = pc.marks["neck"]
            neck_line = pc.grid[-1, i0:i1 + 1]
            dist, along = _neck_distance(g, neck_line)
            uv_neck.append(np.stack([along, np.where(dist < 8.0, dist, 100.0)], 1))
        elif pc.pattern.startswith("sleeve"):
            hem_d = g[:, 1] + spec.sleeve_length
            uv_hem.append(np.stack([g[:, 0], hem_d], 1))
            uv_neck.append(np.stack([g[:, 0], far], 1))
        else:
            hem_d = far
            uv_hem.append(np.stack([g[:, 0], far], 1))
            uv_neck.append(np.stack([g[:, 0], far], 1))
        # Solidify weight (x 4 mm): body 1.5 mm, turned-up hems 3 mm, folded rib 4 mm
        is_rib = pc.pattern == "rib"
        in_hem = hem_d < spec.hem_depth
        thick.append(np.where(is_rib, 1.0, np.where(in_hem, 0.75, 0.375)))
        stiff.append(np.full(len(g), 1.0 if is_rib else 0.0))
        on_body = pc.pattern in ("front", "back")
        press.append(np.full(len(g), 0.0 if is_rib else (body_pressure if on_body else 1.0)))
        n += len(g)

    verts = np.concatenate(verts)
    faces = np.concatenate(faces)
    remap = np.arange(len(verts))
    if weld_virtual:
        for sm in pat.seams_between(names):
            if sm.virtual:
                for a, b in zip(sm.ia, sm.ib):
                    remap[offset[sm.b] + b] = offset[sm.a] + a
        while (remap[remap] != remap).any():
            remap = remap[remap]
    keep = remap == np.arange(len(verts))
    new_index = np.cumsum(keep) - 1
    vmap = new_index[remap]
    counts = np.bincount(vmap, minlength=keep.sum()).astype(float)
    merged = np.stack([np.bincount(vmap, weights=verts[:, k]) for k in range(3)], 1) / counts[:, None]

    sew = set()
    for sm in pat.seams_between(names):
        if weld_virtual and sm.virtual:
            continue
        for a, b in zip(sm.ia, sm.ib):
            e = tuple(sorted((vmap[offset[sm.a] + a], vmap[offset[sm.b] + b])))
            if e[0] != e[1]:
                sew.add(e)
    sew = sorted(sew)

    def per_vertex(chunks):
        return np.concatenate(chunks)[keep]

    obj = mesh_object(name, merged * CM, vmap[faces], edges=sew, col=col)
    me = obj.data
    assert len(me.vertices) == len(merged), "mesh validation dropped vertices"
    set_smooth(me)
    for layer, data in (("Pattern", uv_pat), ("PrintFront", uv_front), ("PrintBack", uv_back),
                        ("Hem", uv_hem), ("Neck", uv_neck)):
        set_loop_uvs(me, layer, per_vertex(data))
    me.uv_layers["Pattern"].active = True
    me.uv_layers["Pattern"].active_render = True
    face_attr(me, "piece", np.concatenate(face_piece))
    point_attr(me, "thickness", per_vertex(thick))
    me.polygons.foreach_set("material_index", (np.concatenate(face_piece) == PATTERN_IDS["rib"]).astype(np.int32))

    for group, data in (("stiff", stiff), ("pressure", press)):
        vg = obj.vertex_groups.new(name=group)
        w = per_vertex(data)
        for value in np.unique(w):
            if value > 0:
                vg.add(np.nonzero(w == value)[0].tolist(), float(value), "REPLACE")

    obj.shape_key_add(name="Basis", from_mix=False)
    flat = obj.shape_key_add(name="Flat", from_mix=False)
    flat.data.foreach_set("co", (per_vertex(rest) * CM).astype(np.float32).ravel())
    flat.value = 0.0
    info = dict(sewing_edges=len(sew), atlas_size_cm=float(atlas_size), print_rects=rects)
    log(f"sim mesh: {len(merged)} verts, {len(me.polygons)} quads, {len(sew)} sewing springs")
    return obj, info


def setup_cloth(obj, p: ClothParams) -> bpy.types.ClothModifier:
    mod = obj.modifiers.new("Cloth", "CLOTH")
    s = mod.settings
    s.quality = p.quality
    s.mass = p.mass
    s.air_damping = p.air_damping
    s.tension_stiffness = p.tension
    s.compression_stiffness = p.compression
    s.shear_stiffness = p.shear
    s.bending_stiffness = p.bending
    s.tension_damping = s.compression_damping = s.shear_damping = 5.0
    s.bending_damping = 0.5
    s.vertex_group_structural_stiffness = "stiff"
    s.tension_stiffness_max = p.rib_stiffness
    s.compression_stiffness_max = p.rib_stiffness
    s.vertex_group_bending = "stiff"
    s.bending_stiffness_max = p.rib_bending
    s.use_sewing_springs = True
    s.sewing_force_max = p.sewing_force
    s.rest_shape_key = obj.data.shape_keys.key_blocks["Flat"]
    if p.pressure:
        s.use_pressure = True
        s.uniform_pressure_force = p.pressure
        s.pressure_factor = 1.0
        s.vertex_group_pressure = "pressure"
    c = mod.collision_settings
    c.use_collision = True
    c.distance_min = p.collision_distance
    c.collision_quality = 4
    c.use_self_collision = True
    c.self_distance_min = p.self_distance
    c.self_friction = 5.0
    mod.point_cache.frame_start = 1
    mod.point_cache.frame_end = p.frames
    ew = s.effector_weights
    if p.sew_frames > 1:
        ew.gravity = 0.0
        ew.keyframe_insert("gravity", frame=1)
        ew.keyframe_insert("gravity", frame=p.sew_frames)
        ew.gravity = 1.0
        ew.keyframe_insert("gravity", frame=p.gravity_full)
    else:
        ew.gravity = 1.0
    return mod


def simulate(obj, p: ClothParams, on_frame=None) -> None:
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, p.frames
    for f in range(1, p.frames + 1):
        scene.frame_set(f)
        if on_frame:
            on_frame(f)
        if f % 10 == 0 or f == p.frames:
            log(f"cloth frame {f}/{p.frames}")


def weld_seams(me: bpy.types.Mesh) -> dict:
    """Merge the vertices joined by sewing springs (the loose edges) at their average."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    parent = list(range(len(bm.verts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    loose = [e for e in bm.edges if not e.link_faces]
    gaps = [e.calc_length() for e in loose]
    for e in loose:
        a, b = find(e.verts[0].index), find(e.verts[1].index)
        if a != b:
            parent[b] = a
    groups: dict[int, list] = {}
    for v in bm.verts:
        groups.setdefault(find(v.index), []).append(v)
    targetmap = {}
    for root, vs in groups.items():
        if len(vs) > 1:
            c = sum((v.co for v in vs), vs[0].co * 0)
            c /= len(vs)
            for v in vs:
                v.co = c
            for v in vs:
                if v is not bm.verts[root]:
                    targetmap[v] = bm.verts[root]
    bmesh.ops.weld_verts(bm, targetmap=targetmap)
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context="EDGES")
    stats = dict(springs=len(loose), max_gap_mm=1000 * max(gaps), mean_gap_mm=1000 * sum(gaps) / len(gaps),
                 merged=len(targetmap))
    bm.to_mesh(me)
    bm.free()
    me.update()
    return stats


def build_tee(sim_obj, col, thickness_m: float = 0.004):
    """Final garment: simulated + welded mesh, Subdivision then Solidify (1.5 mm body)."""
    me = evaluated_mesh_copy(sim_obj, "Tee")
    if me.shape_keys:
        raise RuntimeError("unexpected shape keys on the evaluated mesh")
    stats = weld_seams(me)
    log("weld: {springs} springs, gap max {max_gap_mm:.1f} mm / mean {mean_gap_mm:.1f} mm, "
        "{merged} verts merged".format(**stats))
    set_smooth(me)
    tee = bpy.data.objects.new("Tee", me)
    col.objects.link(tee)
    # thickness weights -> vertex group for Solidify
    w = np.empty(len(me.vertices), np.float32)
    me.attributes["thickness"].data.foreach_get("value", w)
    vg = tee.vertex_groups.new(name="thickness")
    for value in np.unique(np.round(w, 4)):
        vg.add(np.nonzero(np.isclose(w, value, atol=1e-3))[0].tolist(), float(value), "REPLACE")
    sub = tee.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    sub.uv_smooth = "PRESERVE_BOUNDARIES"
    sol = tee.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = thickness_m
    sol.offset = -1.0             # grow inwards: the simulated surface stays the outside
    sol.use_even_offset = True
    sol.use_thickness_angle_clamp = True
    sol.thickness_clamp = 1.0
    sol.use_quality_normals = True
    sol.use_rim = True
    sol.vertex_group = "thickness"
    sol.thickness_vertex_group = 0.0
    return tee, stats
