"""Small bpy helpers shared by the build steps (Blender 4.2+ / 5.x, background mode)."""
from __future__ import annotations

import time

import bpy
import numpy as np

CM = 0.01  # the pattern is in centimetres, Blender scenes in metres

_t0 = time.time()


def log(msg: str) -> None:
    print(f"[tee {time.time() - _t0:7.1f}s] {msg}", flush=True)


def blender_at_least(major: int, minor: int = 0) -> bool:
    return bpy.app.version >= (major, minor, 0)


def clear_scene() -> None:
    """Start from an empty file (factory startup has a cube, a light and a camera)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if col.name not in parent.children:
        parent.children.link(col)
    return col


def mesh_object(name: str, verts: np.ndarray, faces, edges=(), col=None) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, float).tolist(), [tuple(e) for e in edges],
                   [tuple(int(i) for i in f) for f in faces])
    me.validate(clean_customdata=False)
    me.update()
    obj = bpy.data.objects.new(name, me)
    (col or bpy.context.scene.collection).objects.link(obj)
    return obj


def evaluated_mesh_copy(obj: bpy.types.Object, name: str) -> bpy.types.Mesh:
    """Mesh data of `obj` with its whole modifier stack applied (works headless)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    me.name = name
    return me


def apply_modifiers(obj: bpy.types.Object) -> None:
    me = evaluated_mesh_copy(obj, obj.data.name)
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def set_smooth(me: bpy.types.Mesh) -> None:
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    me.update()


def verts_co(me: bpy.types.Mesh) -> np.ndarray:
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def set_loop_uvs(me: bpy.types.Mesh, name: str, per_vertex_uv: np.ndarray) -> None:
    """UV map from a per-vertex (u, v) array (pattern pieces never share vertices)."""
    layer = me.uv_layers.get(name) or me.uv_layers.new(name=name)
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    layer.data.foreach_set("uv", np.asarray(per_vertex_uv, np.float32)[loop_v].ravel())


def point_attr(me: bpy.types.Mesh, name: str, values: np.ndarray, kind="FLOAT") -> None:
    attr = me.attributes.get(name) or me.attributes.new(name, kind, "POINT")
    attr.data.foreach_set("value", np.asarray(values).ravel())


def face_attr(me: bpy.types.Mesh, name: str, values: np.ndarray, kind="INT") -> None:
    attr = me.attributes.get(name) or me.attributes.new(name, kind, "FACE")
    attr.data.foreach_set("value", np.asarray(values).ravel())
