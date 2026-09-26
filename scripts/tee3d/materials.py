"""Shaders: heavy cotton jersey (#F2EFE6), 1x1 rib, double-needle stitching, printed artwork.

Everything is driven by the per-panel UV maps written by garment.py:
  Pattern     atlas of the flat pattern (1 UV unit = atlas_size cm)  -> knit / rib textures
  PrintFront  print_front.png placed on the front panel (CLIP outside) -> dragonfly print
  PrintBack   print_back.png placed on the back panel                -> logo
  Hem / Neck  (along the edge, distance to the edge) in cm           -> stitching relief
"""
from __future__ import annotations

from pathlib import Path

import bpy

FABRIC_SRGB = "#F2EFE6"


def srgb_hex_to_linear(h: str):
    h = h.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (*out, 1.0)


class Graph:
    """Tiny helper around a material node tree."""

    def __init__(self, mat: bpy.types.Material):
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def node(self, kind: str, loc=None, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        if loc:
            n.location = loc
        return n

    @staticmethod
    def inp(node, name: str, kind: str | None = None):
        for s in node.inputs:
            if s.name == name and s.enabled and (kind is None or s.type == kind):
                return s
        raise KeyError(f"{node.bl_idname} has no input {name!r} ({kind})")

    @staticmethod
    def out(node, name: str, kind: str | None = None):
        for s in node.outputs:
            if s.name == name and s.enabled and (kind is None or s.type == kind):
                return s
        raise KeyError(f"{node.bl_idname} has no output {name!r} ({kind})")

    def link(self, a, b):
        self.nt.links.new(a, b)

    def set(self, node, name, value, kind=None):
        self.inp(node, name, kind).default_value = value

    # --- small building blocks ---
    def value(self, v, loc=None):
        n = self.node("ShaderNodeValue", loc)
        n.outputs[0].default_value = v
        return n.outputs[0]

    def math(self, op, a, b=None, c=None, loc=None, clamp=False):
        n = self.node("ShaderNodeMath", loc, operation=op, use_clamp=clamp)
        for sock, v in zip(n.inputs, (a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                sock.default_value = v
            else:
                self.link(v, sock)
        return n.outputs[0]

    def smoothstep(self, x, e0, e1, loc=None):
        """0 at e0, 1 at e1 (e0 > e1 allowed), smooth and clamped."""
        n = self.node("ShaderNodeMapRange", loc, data_type="FLOAT", interpolation_type="SMOOTHSTEP",
                      clamp=True)
        self.link(x, self.inp(n, "Value", "VALUE"))
        self.set(n, "From Min", e0, "VALUE")
        self.set(n, "From Max", e1, "VALUE")
        return self.out(n, "Result", "VALUE")

    def mix_color(self, fac, a, b, blend="MIX", loc=None):
        n = self.node("ShaderNodeMix", loc, data_type="RGBA", blend_type=blend)
        for name, v in (("Factor", fac), ("A", a), ("B", b)):
            sock = self.inp(n, name, "VALUE" if name == "Factor" else "RGBA")
            if isinstance(v, (int, float)):
                sock.default_value = v
            elif isinstance(v, tuple):
                sock.default_value = v
            else:
                self.link(v, sock)
        return self.out(n, "Result", "RGBA")

    def uv(self, name, loc=None):
        return self.out(self.node("ShaderNodeUVMap", loc, uv_map=name), "UV")

    def image(self, img, vector, extension="REPEAT", interpolation="Linear", loc=None):
        n = self.node("ShaderNodeTexImage", loc, image=img, extension=extension,
                      interpolation=interpolation)
        self.link(vector, n.inputs["Vector"])
        return n


def load_image(path: Path, non_color: bool) -> bpy.types.Image:
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    if not non_color:
        img.alpha_mode = "STRAIGHT"
    return img


def _stitch_rows(g: Graph, uv_name: str, rows, period=0.25, thread_r=0.024, groove_w=0.11,
                 loc=(0, 0)):
    """Relief of rows of stitches parallel to an edge. The UV map holds (along, distance) in
    cm. Returns (height, thread_mask) sockets."""
    x, y = loc
    sep = g.node("ShaderNodeSeparateXYZ", (x, y))
    g.link(g.uv(uv_name, (x - 200, y)), sep.inputs[0])
    along, dist = sep.outputs[0], sep.outputs[1]
    # dashes along the seam: each stitch ~85 % of the pitch, rounded ends
    phase = g.math("FRACT", g.math("DIVIDE", along, period))
    dash = g.smoothstep(g.math("ABSOLUTE", g.math("SUBTRACT", phase, 0.5)), 0.47, 0.37)
    thread = None
    groove = None
    for r in rows:
        d = g.math("ABSOLUTE", g.math("SUBTRACT", dist, r))
        t = g.smoothstep(d, thread_r, 0.2 * thread_r)
        gr = g.smoothstep(d, groove_w, 0.0)
        thread = t if thread is None else g.math("MAXIMUM", thread, t)
        groove = gr if groove is None else g.math("MAXIMUM", groove, gr)
    thread = g.math("MULTIPLY", thread, dash)
    height = g.math("SUBTRACT", thread, g.math("MULTIPLY", groove, 0.45))
    return height, thread, dist


def jersey(name: str, tex: dict, prints: dict, atlas_cm: float, spec, inside=False, rib=False):
    """Principled BSDF cotton jersey; `prints` = {'front': Path, 'back': Path} (outside only)."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    g = Graph(mat)
    kind = "rib" if rib else "jersey"
    out = g.node("ShaderNodeOutputMaterial", (1400, 0))
    bsdf = g.node("ShaderNodeBsdfPrincipled", (1000, 0))
    g.link(g.out(bsdf, "BSDF"), out.inputs["Surface"])

    # fabric coordinates in cm (one texture tile = 1 cm)
    cm = g.node("ShaderNodeVectorMath", (-900, 0), operation="SCALE")
    g.link(g.uv("Pattern", (-1100, 0)), cm.inputs[0])
    g.set(cm, "Scale", atlas_cm)
    cm = g.out(cm, "Vector")
    nrm_img = g.image(load_image(tex[f"{kind}_normal"], True), cm, interpolation="Cubic", loc=(-600, -300))
    hgt_img = g.image(load_image(tex[f"{kind}_height"], True), cm, loc=(-600, 0))
    h_knit = g.out(hgt_img, "Color")

    base = g.node("ShaderNodeRGB", (-600, 600))
    base.outputs[0].default_value = srgb_hex_to_linear(FABRIC_SRGB)
    # yarn / dye irregularity: very low amplitude, low frequency mottling (+-2.5 %)
    noise = g.node("ShaderNodeTexNoise", (-600, 400))
    g.link(cm, noise.inputs["Vector"])
    g.set(noise, "Scale", 0.35)
    g.set(noise, "Detail", 4.0)
    mottle = g.math("MULTIPLY_ADD", noise.outputs[0], 0.05, 0.975)  # "Fac" (4.x) / "Factor" (5.x)
    color = g.mix_color(1.0, base.outputs[0], mottle, "MULTIPLY")

    ink = None
    if prints and not inside:
        for panel, uvname in (("front", "PrintFront"), ("back", "PrintBack")):
            if panel not in prints:
                continue
            img = g.image(load_image(prints[panel], False), g.uv(uvname), extension="CLIP",
                          interpolation="Cubic", loc=(-600, 700 if panel == "front" else 1000))
            a = g.out(img, "Alpha")
            color = g.mix_color(a, color, g.out(img, "Color"))
            ink = a if ink is None else g.math("MAXIMUM", ink, a)

    # knit cavities are a touch darker (self-shadowing of the yarn)
    cav = g.math("MULTIPLY_ADD", h_knit, 0.26 if not inside else 0.14, 0.74 if not inside else 0.86)
    color = g.mix_color(1.0, color, cav, "MULTIPLY")
    if inside:
        color = g.mix_color(1.0, color, (0.93, 0.93, 0.93, 1.0), "MULTIPLY")

    # stitching relief (outside only): double-needle hems, twin-needle neck seam
    height = None
    thread = None
    if not inside and not rib:
        r1 = spec.hem_depth - 0.95
        hem_h, hem_t, hem_d = _stitch_rows(g, "Hem", (r1, r1 + 0.64), loc=(-900, -900))
        band = g.smoothstep(hem_d, spec.hem_depth + 0.12, spec.hem_depth - 0.1)  # turned-up hem
        neck_h, neck_t, neck_d = _stitch_rows(g, "Neck", (0.55, 1.2), loc=(-900, -1300))
        seam = g.smoothstep(neck_d, 0.22, 0.0)  # the rib seam itself sinks a little
        height = g.math("ADD", g.math("ADD", hem_h, g.math("MULTIPLY", band, 0.35)),
                        g.math("SUBTRACT", neck_h, g.math("MULTIPLY", seam, 0.6)))
        thread = g.math("MAXIMUM", hem_t, neck_t)
        # tonal thread: the stitches read through their relief, barely through colour
        color = g.mix_color(g.math("MULTIPLY", thread, 0.35), color, (0.88, 0.86, 0.82, 1.0),
                            "MULTIPLY")

    # normals: knit normal map, then stitching bump on top
    nmap = g.node("ShaderNodeNormalMap", (-200, -300), space="TANGENT", uv_map="Pattern")
    g.link(g.out(nrm_img, "Color"), nmap.inputs["Color"])
    strength = 1.6 if rib else (0.6 if inside else 1.3)
    if ink is not None:  # ink fills the knit a little
        g.link(g.math("MULTIPLY_ADD", ink, -0.3 * strength, strength), nmap.inputs["Strength"])
    else:
        g.set(nmap, "Strength", strength)
    normal = g.out(nmap, "Normal")
    if height is not None:
        bump = g.node("ShaderNodeBump", (200, -300))
        g.set(bump, "Strength", 1.0)
        g.set(bump, "Distance", 0.0006)
        g.link(height, g.inp(bump, "Height"))
        g.link(normal, g.inp(bump, "Normal"))
        normal = g.out(bump, "Normal")
    g.link(normal, g.inp(bsdf, "Normal"))

    # cotton: very matte, soft sheen from the fibres; ink and thread are a bit smoother
    rough = g.math("MULTIPLY_ADD", h_knit, -0.08, 0.95)
    if ink is not None:
        rough = g.math("SUBTRACT", rough, g.math("MULTIPLY", ink, 0.12))
    if thread is not None:
        rough = g.math("SUBTRACT", rough, g.math("MULTIPLY", thread, 0.3))
    g.link(color, g.inp(bsdf, "Base Color"))
    g.link(rough, g.inp(bsdf, "Roughness"))
    g.set(bsdf, "Specular IOR Level", 0.35)
    sheen = 0.45 if not inside else 0.7
    if ink is not None:
        g.link(g.math("MULTIPLY_ADD", ink, -0.5 * sheen, sheen), g.inp(bsdf, "Sheen Weight"))
    else:
        g.set(bsdf, "Sheen Weight", sheen)
    g.set(bsdf, "Sheen Roughness", 0.5)
    return mat


def assign(tee: bpy.types.Object, tex: dict, prints: dict, atlas_cm: float, spec) -> None:
    """Slots: 0 jersey, 1 rib (outside); Solidify offsets the inner shell/rim by +2."""
    mats = [jersey("Tee_Jersey", tex, prints, atlas_cm, spec),
            jersey("Tee_Rib", tex, {}, atlas_cm, spec, rib=True),
            jersey("Tee_Jersey_Inside", tex, {}, atlas_cm, spec, inside=True),
            jersey("Tee_Rib_Inside", tex, {}, atlas_cm, spec, inside=True, rib=True)]
    tee.data.materials.clear()
    for m in mats:
        tee.data.materials.append(m)
    sol = tee.modifiers.get("Solidify")
    if sol:
        sol.material_offset = 2
        sol.material_offset_rim = 2
