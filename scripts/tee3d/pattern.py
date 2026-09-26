"""Flat sewing pattern of the oversized tee, meshed as structured quad grids.

Pure numpy (no bpy) so it can be checked outside Blender.

Units are centimetres. Every piece is drawn as seen from the OUTSIDE of the garment:
x to the right, y up, the high-point-of-shoulder (HPS) line at y = 0 for front/back, the
top of the sleeve cap at y = 0 for sleeves, the sewn edge at y = 0 for the neck rib.

Pattern pieces (= UV islands): front, back, sleeve_l, sleeve_r, rib.
For the simulation, sleeves and rib are split in a front and a back half joined by
"virtual" seams (they are welded back into one piece after the simulation). This lets
every half start close to its seam partners instead of being torn across the shoulders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeeSpec:
    """Oversized / boxy fit with dropped shoulders (men's L), heavy jersey."""
    half_chest: float = 30.0         # 60 cm flat across chest and hem: boxy, straight sides
    length: float = 74.0             # HPS -> hem
    half_shoulder: float = 29.0      # dropped shoulder point: 58 cm seam to seam
    shoulder_drop: float = 4.0       # shoulder point below the HPS line
    half_neck: float = 9.5           # 19 cm neck width, HPS to HPS
    front_neck_drop: float = 9.5
    back_neck_drop: float = 2.5
    armhole_depth: float = 26.0      # shoulder point -> underarm, straight down
    sleeve_length: float = 23.0      # cap top -> sleeve hem: ends above the elbow
    sleeve_cap_height: float = 12.0  # cap height sets the hang angle: 12 cm -> ~45-50 deg
    sleeve_opening: float = 48.0     # circumference of the sleeve hem (24 cm flat): wide
    rib_height: float = 3.0          # visible height of the (folded) neck rib
    rib_ratio: float = 0.90          # rib length / neckline length: the rib is stretched on
    hem_depth: float = 2.5           # turned-up hem allowance, body and sleeves
    band_split: float = 18.0         # body grid is regular below this depth (cm under HPS)
    rib_rows: int = 4                # grid rows across the rib
    edge: float = 1.3                # target edge length of the simulation mesh


# --- curves --------------------------------------------------------------------------------
def bezier(p0, p1, p2, p3, n: int = 400) -> np.ndarray:
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def segment(a, b, n: int = 64) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) * np.asarray(a, float) + t * np.asarray(b, float)


def join(*curves: np.ndarray) -> np.ndarray:
    out = [curves[0]]
    for c in curves[1:]:
        out.append(c[1:])
    return np.concatenate(out)


def length(c: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def resample(c: np.ndarray, n: int) -> np.ndarray:
    """n segments (n+1 points) evenly spaced along the polyline c."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    s = np.linspace(0.0, d[-1], n + 1)
    return np.stack([np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])], axis=1)


def resample_at(c: np.ndarray, fracs: np.ndarray) -> np.ndarray:
    """Points at the given arc-length fractions (0..1) of the polyline c."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    s = np.asarray(fracs) * d[-1]
    return np.stack([np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])], axis=1)


def arc_fractions(p: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return d / d[-1]


def split_at_y(c: np.ndarray, y0: float):
    """Split a polyline going upwards at height y0 -> (below, above), sharing the cut point."""
    k = int(np.nonzero(c[:, 1] >= y0)[0][0])
    t = (y0 - c[k - 1, 1]) / (c[k, 1] - c[k - 1, 1])
    cut = c[k - 1] + t * (c[k] - c[k - 1])
    return np.vstack([c[:k], cut]), np.vstack([cut, c[k:]])


def segments(len_cm: float, edge: float, minimum: int = 2) -> int:
    return max(minimum, int(round(len_cm / edge)))


def coons(bottom: np.ndarray, top: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Bilinearly blended Coons patch. bottom/top: (nu+1, 2) left->right, left/right: (nv+1, 2)
    bottom->top. Returns the (nv+1, nu+1, 2) grid; its border is exactly the four curves."""
    nu, nv = len(bottom) - 1, len(left) - 1
    assert len(top) == nu + 1 and len(right) == nv + 1
    u = np.linspace(0.0, 1.0, nu + 1)[None, :, None]
    v = np.linspace(0.0, 1.0, nv + 1)[:, None, None]
    B, T = bottom[None], top[None]
    L, R = left[:, None], right[:, None]
    c00, c10, c01, c11 = bottom[0], bottom[-1], top[0], top[-1]
    g = ((1 - v) * B + v * T + (1 - u) * L + u * R
         - ((1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11))
    g[0], g[-1], g[:, 0], g[:, -1] = bottom, top, left, right
    return g


# --- pieces --------------------------------------------------------------------------------
@dataclass
class Piece:
    name: str                 # simulation piece
    pattern: str              # pattern piece / UV island it belongs to
    grid: np.ndarray          # (nv+1, nu+1, 2) pattern coordinates, cm
    marks: dict = field(default_factory=dict)  # named grid index ranges (see Pattern)

    @property
    def nu(self) -> int:
        return self.grid.shape[1] - 1

    @property
    def nv(self) -> int:
        return self.grid.shape[0] - 1

    def idx(self, j: int, i: int) -> int:
        return j * (self.nu + 1) + i

    def bottom(self, a=0, b=None):
        b = self.nu if b is None else b
        return [self.idx(0, i) for i in range(a, b + 1)]

    def top(self, a=0, b=None):
        b = self.nu if b is None else b
        return [self.idx(self.nv, i) for i in range(a, b + 1)]

    def left(self, a=0, b=None):
        b = self.nv if b is None else b
        return [self.idx(j, 0) for j in range(a, b + 1)]

    def right(self, a=0, b=None):
        b = self.nv if b is None else b
        return [self.idx(j, self.nu) for j in range(a, b + 1)]

    def faces(self) -> np.ndarray:
        """Quads, counter-clockwise in pattern space = normal pointing out of the garment."""
        j, i = np.mgrid[0:self.nv, 0:self.nu]
        a = j * (self.nu + 1) + i
        return np.stack([a, a + 1, a + self.nu + 2, a + self.nu + 1], axis=-1).reshape(-1, 4)


@dataclass
class Seam:
    a: str
    ia: list
    b: str
    ib: list
    virtual: bool = False     # fold line of a piece split for the simulation only
    name: str = ""


class Pattern:
    """Builds all pieces and seams from a TeeSpec."""

    def __init__(self, spec: TeeSpec | None = None):
        self.spec = s = spec or TeeSpec()
        e = s.edge
        W, L = s.half_chest, s.length
        u_y = -(s.shoulder_drop + s.armhole_depth)      # underarm height
        S = np.array([s.half_shoulder, -s.shoulder_drop])  # shoulder point (right)
        U = np.array([W, u_y])                              # underarm (right)

        # right armhole, underarm -> shoulder point: almost straight (boxy dropped shoulder)
        arm_r = bezier(U, U + [-0.8, 8.0], S + [-0.2, -8.0], S)
        self.armhole_len = length(arm_r)
        necks = {
            "front": join(bezier([-s.half_neck, 0], [-s.half_neck, -0.55 * s.front_neck_drop],
                                 [-0.55 * s.half_neck, -s.front_neck_drop], [0, -s.front_neck_drop]),
                          bezier([0, -s.front_neck_drop], [0.55 * s.half_neck, -s.front_neck_drop],
                                 [s.half_neck, -0.55 * s.front_neck_drop], [s.half_neck, 0])),
            "back": join(bezier([-s.half_neck, 0], [-s.half_neck + 0.3, -0.8 * s.back_neck_drop],
                                [-0.5 * s.half_neck, -s.back_neck_drop], [0, -s.back_neck_drop]),
                         bezier([0, -s.back_neck_drop], [0.5 * s.half_neck, -s.back_neck_drop],
                                [s.half_neck - 0.3, -0.8 * s.back_neck_drop], [s.half_neck, 0])),
        }
        self.neck_len = {k: length(v) for k, v in necks.items()}
        shoulder_r = segment([s.half_neck, 0], S)
        self.shoulder_len = length(shoulder_r)

        self.n_side = segments(L + u_y, e)
        self.n_arm = segments(self.armhole_len, e)
        self.n_sh = segments(self.shoulder_len, e)
        self.n_neck = {k: segments(v, e) for k, v in self.neck_len.items()}

        self.pieces: dict[str, Piece] = {}
        self.seams: list[Seam] = []
        self.outlines: dict[str, np.ndarray] = {}

        # --- body panels: regular grid up to the band, blended band under the neckline ---
        # (a single Coons patch would slant the grain lines from the HPS down to the hem,
        # and quad cloth creases along slanted grain lines)
        y_split = -s.band_split
        arm_lo, arm_up = split_at_y(arm_r, y_split)
        n_up = max(2, int(round(self.n_arm * length(arm_up) / self.armhole_len)))
        side_r_lo = np.concatenate([resample(segment([W, -L], U), self.n_side),
                                    resample(arm_lo, self.n_arm - n_up)[1:]])
        side_r_up = resample(arm_up, n_up)
        armhole_pts = np.concatenate([side_r_lo[self.n_side:], side_r_up[1:]])
        self.armhole_fracs = arc_fractions(armhole_pts)  # sleeve caps are sampled alike
        xs = side_r_up[0, 0]
        for name in ("front", "back"):
            nk = self.n_neck[name]
            mirror = lambda c: c * [-1, 1]
            top = np.concatenate([
                resample(segment(mirror(S), [-s.half_neck, 0]), self.n_sh),
                resample(necks[name], nk)[1:],
                resample(shoulder_r, self.n_sh)[1:],
            ])
            nu = len(top) - 1
            bottom = resample(segment([-W, -L], [W, -L]), nu)
            mid = resample(segment([-xs, y_split], [xs, y_split]), nu)
            lower = coons(bottom, mid, side_r_lo * [-1, 1], side_r_lo)
            upper = coons(mid, top, side_r_up * [-1, 1], side_r_up)
            grid = np.concatenate([lower, upper[1:]], axis=0)
            self.pieces[name] = Piece(name, name, grid, marks=dict(neck=(self.n_sh, self.n_sh + nk)))
            side_r = np.concatenate([side_r_lo, side_r_up[1:]])
            side_l = side_r * [-1, 1]
            self.outlines[name] = np.concatenate([bottom, side_r[1:], top[::-1][1:], side_l[::-1][1:]])

        # --- sleeve: solve the cap width so that half the cap = one armhole (no ease) ---
        h = s.sleeve_cap_height

        def cap_half(b):  # underarm (-b, -h) -> cap top (0, 0), bell shaped
            return bezier([-b, -h], [-0.55 * b, -h], [-0.45 * b, 0.0], [0.0, 0.0])

        lo, hi = 5.0, 60.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if length(cap_half(mid)) < self.armhole_len else (lo, mid)
        self.bicep_half = b = 0.5 * (lo + hi)
        hem_half = s.sleeve_opening / 2.0
        Ls = s.sleeve_length
        underarm = segment([-hem_half, -Ls], [-b, -h])
        self.underarm_len = length(underarm)
        centre = segment([0.0, -Ls], [0.0, 0.0])
        self.n_sleeve = segments(0.5 * (length(underarm) + length(centre)), e)
        cap = resample_at(cap_half(b), self.armhole_fracs)
        hem = resample(segment([-hem_half, -Ls], [0.0, -Ls]), self.n_arm)
        uarm = resample(underarm, self.n_sleeve)
        cline = resample(centre, self.n_sleeve)
        half_a = coons(hem, cap, uarm, cline)                                # x <= 0
        half_b = coons(hem[::-1] * [-1, 1], cap[::-1] * [-1, 1], cline, uarm * [-1, 1])  # x >= 0
        m = np.array([-1.0, 1.0])
        self.outlines["sleeve"] = np.concatenate([
            hem, (hem[::-1] * m)[1:], (uarm * m)[1:], (cap * m)[1:], cap[::-1][1:], uarm[::-1][1:]])
        # left sleeve (world +X): seen from outside, x > 0 is the back half; right: the front
        for side, (front_half, back_half) in (("l", ("a", "b")), ("r", ("b", "a"))):
            for tag, g in (("a", half_a), ("b", half_b)):
                role = "front" if tag == front_half else "back"
                self.pieces[f"sleeve_{side}_{role}"] = Piece(
                    f"sleeve_{side}_{role}", f"sleeve_{side}", g.copy(), marks=dict(half=tag))

        # --- neck rib: one strip, front part then back part (seen from outside) ---
        self.rib_len = s.rib_ratio * (self.neck_len["front"] + self.neck_len["back"])
        lf = s.rib_ratio * self.neck_len["front"]
        nr = s.rib_rows
        for name, x0, x1, n in (("rib_front", 0.0, lf, self.n_neck["front"]),
                                ("rib_back", lf, self.rib_len, self.n_neck["back"])):
            g = coons(resample(segment([x0, 0], [x1, 0]), n),
                      resample(segment([x0, s.rib_height], [x1, s.rib_height]), n),
                      resample(segment([x0, 0], [x0, s.rib_height]), nr),
                      resample(segment([x1, 0], [x1, s.rib_height]), nr))
            self.pieces[name] = Piece(name, "rib", g)
        self.outlines["rib"] = np.array([[0, 0], [self.rib_len, 0], [self.rib_len, s.rib_height],
                                         [0, s.rib_height], [0, 0]])

        self._seams()

    # --- seams ---------------------------------------------------------------------------
    def _seams(self):
        P, add = self.pieces, self.seams.append
        f, b = P["front"], P["back"]
        ns, nsh = self.n_side, self.n_sh
        nv = f.nv  # = n_side + n_arm, same for front and back
        # side seams: front right (world +X) <-> back left, front left <-> back right
        add(Seam("front", f.right(0, ns), "back", b.left(0, ns), name="side_l"))
        add(Seam("front", f.left(0, ns), "back", b.right(0, ns), name="side_r"))
        # shoulders, HPS -> shoulder point
        nf, nb = f.nu, b.nu
        add(Seam("front", [f.idx(nv, i) for i in range(nf - nsh, nf + 1)],
                 "back", [b.idx(nv, i) for i in range(nsh, -1, -1)], name="shoulder_l"))
        add(Seam("front", [f.idx(nv, i) for i in range(nsh, -1, -1)],
                 "back", [b.idx(nv, i) for i in range(nb - nsh, nb + 1)], name="shoulder_r"))
        # armholes <-> sleeve caps, underarm -> shoulder point
        caps = {  # (body piece, its armhole indices underarm->shoulder) per sleeve half
            "sleeve_l_front": f.right(ns, nv), "sleeve_l_back": b.left(ns, nv),
            "sleeve_r_front": f.left(ns, nv), "sleeve_r_back": b.right(ns, nv),
        }
        for name, arm in caps.items():
            sl = P[name]
            body = "front" if name.endswith("front") else "back"
            cap = sl.top()  # half a: underarm -> top ; half b: top -> underarm
            if sl.marks["half"] == "b":
                cap = cap[::-1]
            add(Seam(body, arm, name, cap, name=f"armhole_{name}"))
        for side in ("l", "r"):
            fa, ba = P[f"sleeve_{side}_front"], P[f"sleeve_{side}_back"]
            a, bb = (fa, ba) if fa.marks["half"] == "a" else (ba, fa)
            add(Seam(a.name, a.left(), bb.name, bb.right(), name=f"underarm_{side}"))
            add(Seam(a.name, a.right(), bb.name, bb.left(), virtual=True, name=f"sleeve_fold_{side}"))
        # neck rib bottom edge <-> necklines
        rf, rb = P["rib_front"], P["rib_back"]
        i0, i1 = f.marks["neck"]
        add(Seam("front", [f.idx(nv, i) for i in range(i0, i1 + 1)], "rib_front", rf.bottom(),
                 name="neck_front"))
        i0, i1 = b.marks["neck"]
        add(Seam("back", [b.idx(nv, i) for i in range(i0, i1 + 1)], "rib_back", rb.bottom(),
                 name="neck_back"))
        # rib: split at the left shoulder (virtual), closing seam at the right shoulder
        add(Seam("rib_front", rf.right(), "rib_back", rb.left(), virtual=True, name="rib_fold"))
        add(Seam("rib_front", rf.left(), "rib_back", rb.right(), name="rib_join"))
        for sm in self.seams:
            assert len(sm.ia) == len(sm.ib), (sm.name, len(sm.ia), len(sm.ib))

    def seams_between(self, names) -> list[Seam]:
        names = set(names)
        return [sm for sm in self.seams if sm.a in names and sm.b in names]

    # --- summaries -------------------------------------------------------------------------
    def check(self) -> dict:
        """Seam length mismatches (cm), for sanity."""
        out = {}
        for sm in self.seams:
            ga = self.pieces[sm.a].grid.reshape(-1, 2)[sm.ia]
            gb = self.pieces[sm.b].grid.reshape(-1, 2)[sm.ib]
            out[sm.name] = (round(length(ga), 2), round(length(gb), 2))
        return out

    def vertex_count(self) -> int:
        return sum(p.grid.shape[0] * p.grid.shape[1] for p in self.pieces.values())
