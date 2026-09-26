"""Invisible ("ghost") mannequin used as the cloth collider: torso + neck + arms in an A-pose.

Built from a lofted torso, two deltoids and two tapered arms, merged with a voxel remesh
and smoothed so the armpits and shoulders have no crease the cloth could catch on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import bmesh
import bpy
import numpy as np

from .util import CM, apply_modifiers, log, mesh_object, set_smooth


@dataclass
class Rig:
    """Mannequin pose and the initial placement of the panels around it (cm, degrees)."""
    hps_z: float = 145.0          # world height of the high point of shoulder line
    arm_angle: float = 45.0       # arms down-out, from vertical (mockup: ~45 deg)
    joint_x: float = 17.5         # shoulder joint, from the centre plane
    joint_drop: float = 10.5      # shoulder joint below the HPS line
    arm_length: float = 40.0
    arm_r0: float = 5.8           # upper arm radius at the shoulder
    arm_r1: float = 4.5           # ... near the elbow
    panel_depth: float = 15.0     # flat front/back panels start this far in front/behind
    wrap_radius: float = 60.0     # ... slightly curved around the body with this radius
    sleeve_hem_dist: float = 30.0  # arm joint -> sleeve hem ring, along the arm

    def arm_frame(self, sx: float):
        """Joint, arm direction and 'up' direction (perpendicular, towards the shoulder top)
        for the arm on the side sx (+1: world +X = wearer's left, -1: wearer's right)."""
        a = math.radians(self.arm_angle)
        joint = np.array([sx * self.joint_x, 0.0, self.hps_z - self.joint_drop])
        d = np.array([sx * math.sin(a), 0.0, -math.cos(a)])
        up = np.array([sx * math.cos(a), 0.0, math.sin(a)])
        return joint, d, up


# (height below/above the HPS line, half width, half depth, y offset of the centre, exponent)
TORSO = [
    (-86.0, 16.8, 11.8, 0.0, 2.2),
    (-75.0, 17.4, 12.4, 0.0, 2.3),
    (-62.0, 17.2, 12.0, 0.0, 2.3),   # hips
    (-50.0, 16.2, 11.3, -0.3, 2.3),  # waist
    (-40.0, 16.6, 11.6, -0.5, 2.3),
    (-30.0, 17.8, 12.4, -0.8, 2.4),
    (-22.0, 18.6, 12.8, -0.9, 2.5),  # chest
    (-15.0, 18.9, 12.2, -0.6, 2.5),
    (-10.0, 19.4, 11.2, -0.2, 2.5),
    (-7.0, 20.0, 10.2, 0.0, 2.4),
    (-5.0, 19.2, 9.4, 0.2, 2.3),     # acromion level: sloping shoulders
    (-3.0, 16.0, 8.4, 0.4, 2.2),
    (-1.2, 11.5, 7.3, 0.5, 2.1),
    (0.0, 7.6, 6.4, 0.6, 2.0),       # neck base, under the garment's HPS (x = 9.5 cm)
    (2.0, 6.2, 6.0, 0.8, 2.0),
    (10.0, 5.8, 5.8, 1.0, 2.0),
    (14.0, 5.5, 5.5, 1.2, 2.0),
]


def _superellipse(a, b, n, k=64):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = a * np.sign(c) * np.abs(c) ** (2.0 / n)
    y = b * np.sign(s) * np.abs(s) ** (2.0 / n)
    return x, y


def _loft(rings: list[np.ndarray]):
    """Closed tube through rings of k points each, capped at both ends."""
    k = len(rings[0])
    verts = np.concatenate(rings)
    faces = []
    for r in range(len(rings) - 1):
        for i in range(k):
            a, b = r * k + i, r * k + (i + 1) % k
            faces.append((a, b, b + k, a + k))
    faces.append(tuple(range(k - 1, -1, -1)))
    last = (len(rings) - 1) * k
    faces.append(tuple(range(last, last + k)))
    return verts, faces


def _torso(rig: Rig):
    rings = []
    for dz, a, b, yc, n in TORSO:
        x, y = _superellipse(a, b, n)
        rings.append(np.stack([x, y + yc, np.full_like(x, rig.hps_z + dz)], axis=1))
    return _loft(rings)


def _arm(rig: Rig, sx: float, k=40, rows=24):
    joint, d, up = rig.arm_frame(sx)
    side = np.cross(d, up)
    rings = []
    t = np.linspace(0.0, 1.0, rows)
    for ti in t:
        r = rig.arm_r0 + (rig.arm_r1 - rig.arm_r0) * ti
        c = joint + d * rig.arm_length * ti
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        rings.append(c + r * (np.cos(ang)[:, None] * up + np.sin(ang)[:, None] * side))
    # rounded end
    end = joint + d * rig.arm_length
    for q in np.linspace(0.3, 0.95, 4):
        r = rig.arm_r1 * math.cos(q * math.pi / 2)
        c = end + d * rig.arm_r1 * math.sin(q * math.pi / 2)
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        rings.append(c + r * (np.cos(ang)[:, None] * up + np.sin(ang)[:, None] * side))
    return _loft(rings)


def _ellipsoid(center, radii):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
    verts = np.array([v.co[:] for v in bm.verts]) * radii + center
    faces = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    return verts, faces


def build(rig: Rig, col: bpy.types.Collection, voxel_cm: float = 0.7) -> bpy.types.Object:
    parts = [_torso(rig)]
    for sx in (-1.0, 1.0):
        parts.append(_arm(rig, sx))
        parts.append(_ellipsoid(np.array([sx * 19.5, 0.3, rig.hps_z - 10.0]), np.array([6.3, 6.6, 7.0])))
    verts, faces, off = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces += [tuple(i + off for i in face) for face in f]
        off += len(v)
    obj = mesh_object("Mannequin", np.concatenate(verts) * CM, faces, col=col)

    rm = obj.modifiers.new("Remesh", "REMESH")  # union of the overlapping parts
    rm.mode = "VOXEL"
    rm.voxel_size = voxel_cm * CM
    sm = obj.modifiers.new("Smooth", "SMOOTH")  # round off armpits / shoulder junctions
    sm.factor = 0.8
    sm.iterations = 12
    apply_modifiers(obj)
    set_smooth(obj.data)
    log(f"mannequin: {len(obj.data.vertices)} verts")

    coll = obj.modifiers.new("Collision", "COLLISION")
    cs = obj.collision
    cs.thickness_outer = 0.004
    cs.thickness_inner = 0.02
    cs.cloth_friction = 8.0
    cs.damping = 0.2
    cs.use_culling = True   # single sided: cloth is always pushed out of the body
    cs.use_normal = True
    obj.hide_render = True  # ghost mannequin: never rendered
    return obj
