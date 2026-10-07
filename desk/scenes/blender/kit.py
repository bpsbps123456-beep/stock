"""Shared Blender toolkit for the AI 탐방 cutaway panels — runs inside Blender only (bpy).

A panel is modelled in metre-like units on a diorama slab, rendered by Cycles from a fixed
isometric camera onto a transparent background, and exported with what the page needs to put
labels on it: one anchor per zone (a point on the zone's *visible* pixels), a highlight mask for
hover, and a coarse occupancy grid so labels can be laid out over empty space
(desk/scenes/layout.py). Zones are the tour_map.json zone ids; the page draws names from there.
"""

from __future__ import annotations

import json
import math
import os

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

SCALE = 2          # rendered pixels per CSS pixel (sharp on high-DPI screens)
CELL = 8           # occupancy grid cell, CSS px
MASK = 2           # highlight mask cell, CSS px


def lin(hexstr: str) -> tuple[float, float, float]:
    """'#rrggbb' (sRGB) -> linear RGB, which is what Blender colour sockets take."""
    h = hexstr.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


# name: (sRGB colour, roughness, metallic, emission strength, alpha)
PALETTE = {
    "ground":   ("#c7c0ad", 0.9, 0, 0, 1),
    "grass":    ("#9fb586", 0.95, 0, 0, 1),
    "asphalt":  ("#6d7176", 0.9, 0, 0, 1),
    "gravel":   ("#b3ab98", 0.95, 0, 0, 1),
    "soil1":    ("#9a7b56", 0.95, 0, 0, 1),
    "soil2":    ("#80643f", 0.95, 0, 0, 1),
    "soil3":    ("#66503a", 0.95, 0, 0, 1),
    "rock":     ("#6e665e", 0.9, 0, 0, 1),
    "concrete": ("#d7d2c7", 0.85, 0, 0, 1),
    "wall":     ("#e6e1d6", 0.85, 0, 0, 1),
    "cut":      ("#9b5b45", 0.8, 0, 0, 1),     # cut face of a sectioned wall/slab
    "roof":     ("#bdb7ab", 0.85, 0, 0, 1),
    "floor":    ("#d9d6ce", 0.7, 0, 0, 1),
    "metal":    ("#b9bec5", 0.45, 0.5, 0, 1),
    "metal_dk": ("#5b616a", 0.5, 0.4, 0, 1),
    "metal_bk": ("#2c3036", 0.5, 0.3, 0, 1),
    "white":    ("#eef0f1", 0.6, 0, 0, 1),
    "rack":     ("#23272d", 0.55, 0.2, 0, 1),
    "rack_fr":  ("#33383f", 0.6, 0.2, 0, 1),
    "blade":    ("#454b54", 0.55, 0.3, 0, 1),
    "led_g":    ("#3fd98b", 0.5, 0, 6, 1),
    "led_b":    ("#4aa3ff", 0.5, 0, 6, 1),
    "led_a":    ("#ffb020", 0.5, 0, 6, 1),
    "screen":   ("#5fd4c4", 0.4, 0, 2.5, 1),
    "power":    ("#f0b429", 0.4, 0.1, 0, 1),
    "cold":     ("#3b8fe0", 0.35, 0.1, 0, 1),
    "hot":      ("#e8603c", 0.35, 0.1, 0, 1),
    "fiber":    ("#25bfe0", 0.4, 0, 0.2, 1),
    "fibertray": ("#f2c94c", 0.5, 0, 0, 1),
    "glass":    ("#a8d0e0", 0.15, 0, 0, 0.55),
    "window":   ("#8fb6c9", 0.2, 0.1, 0, 1),
    "solar":    ("#27467f", 0.25, 0.2, 0, 1),
    "chip":     ("#2a2e35", 0.4, 0.2, 0, 1),
    "die":      ("#2f7f6a", 0.25, 0.3, 0, 1),
    "cpu_die":  ("#4b6a9b", 0.25, 0.3, 0, 1),
    "hbm":      ("#8f86e8", 0.35, 0.2, 0, 1),
    "pcb":      ("#2f6f4a", 0.6, 0, 0, 1),
    "substrate": ("#5c8f4f", 0.6, 0, 0, 1),
    "silicon":  ("#8d93a0", 0.3, 0.4, 0, 1),
    "copper":   ("#c98a4b", 0.35, 0.8, 0, 1),
    "gold":     ("#d9b14a", 0.35, 0.8, 0, 1),
    "solder":   ("#c8c9cc", 0.3, 0.7, 0, 1),
    "mlcc":     ("#b98d4a", 0.5, 0.1, 0, 1),
    "transformer": ("#8c9aa8", 0.55, 0.3, 0, 1),
    "porcelain": ("#a9764c", 0.4, 0, 0, 1),
    "genset":   ("#d2d6da", 0.6, 0.1, 0, 1),
    "crane":    ("#f2c12e", 0.5, 0.1, 0, 1),
    "steel":    ("#c0673b", 0.6, 0.2, 0, 1),
    "tree":     ("#6e9a5c", 0.95, 0, 0, 1),
    "trunk":    ("#7a5c3e", 0.95, 0, 0, 1),
    "water":    ("#5f9fd6", 0.15, 0, 0, 1),
    "steam":    ("#f4f5f6", 0.9, 0, 0, 0.92),
    "smoke":    ("#9aa0a6", 0.9, 0, 0, 0.85),
    "uranium":  ("#d6c13a", 0.6, 0, 0.15, 1),
    "ore":      ("#8b7a52", 0.95, 0, 0, 1),
    "wafer":    ("#7f88c4", 0.2, 0.7, 0, 1),
    "euv":      ("#b06cf0", 0.4, 0, 3, 1),
    "plasma":   ("#c690ff", 0.4, 0, 8, 1),
    "red":      ("#d9534f", 0.5, 0, 0, 1),
}


def _line_colour(rgb):  # darker shade of the body colour for its outline
    return (rgb[0] * 0.28, rgb[1] * 0.28, rgb[2] * 0.3, 1.0)


class Part:
    """Builds one multi-material mesh out of boxes, rods and rings, so a whole machine (a rack,
    a transformer) is a single object that can be instanced many times cheaply."""

    def __init__(self, kit: "Kit", name: str):
        self.kit, self.name = kit, name
        self.bm = bmesh.new()
        self.mats: list[str] = []

    def _mi(self, key):
        if key not in self.mats:
            self.mats.append(key)
        return self.mats.index(key)

    def _paint(self, verts, mat, top=None, smooth=False):
        faces = {f for v in verts for f in v.link_faces}
        mi = self._mi(mat)
        ti = self._mi(top) if top else None
        best, bz = None, -2.0
        for f in faces:
            f.material_index = mi
            f.smooth = smooth and len(f.verts) == 4
            f.normal_update()
            if f.normal.z > bz:
                best, bz = f, f.normal.z
        if ti is not None and best is not None:
            best.material_index = ti
        return faces

    def box(self, c, s, mat, rz=0.0, top=None, rx=0.0, ry=0.0):
        """Box with its bottom centre at c (x, y, z), size s, optional rotations (radians)."""
        m = (Matrix.Translation(Vector(c) + Vector((0, 0, s[2] / 2)))
             @ Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X")
             @ Matrix.Diagonal((s[0], s[1], s[2], 1)))
        v = bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)["verts"]
        self._paint(v, mat, top)
        return self

    def rod(self, a, b, r, mat, verts=12, r2=None, cap=True):
        a, b = Vector(a), Vector(b)
        d = b - a
        if d.length < 1e-6:
            return self
        m = Matrix.Translation((a + b) / 2) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4()
        v = bmesh.ops.create_cone(self.bm, cap_ends=cap, cap_tris=False, segments=verts, radius1=r,
                                  radius2=r if r2 is None else r2, depth=d.length, matrix=m)["verts"]
        faces = self._paint(v, mat, smooth=True)
        for f in faces:
            for e in f.edges:
                if len(e.link_faces) == 2 and e.link_faces[0].smooth != e.link_faces[1].smooth:
                    e.smooth = False
        return self

    def ring(self, c, r, t, h, mat, verts=24, axis="Z"):
        """Short open-ended tube (fan shroud, pipe flange) as a lathe of a rectangle."""
        prof = [(r, 0), (r, h), (r - t, h), (r - t, 0)]
        rot = {"Z": Matrix.Identity(4), "X": Matrix.Rotation(math.pi / 2, 4, "Y"), "Y": Matrix.Rotation(-math.pi / 2, 4, "X")}[axis]
        m = Matrix.Translation(c) @ rot
        rings = []
        for rr, zz in prof:
            rings.append([self.bm.verts.new(m @ Vector((rr * math.cos(2 * math.pi * i / verts), rr * math.sin(2 * math.pi * i / verts), zz)))
                          for i in range(verts)])
        rings.append(rings[0])
        vs = []
        for a, b in zip(rings, rings[1:]):
            for i in range(verts):
                j = (i + 1) % verts
                f = self.bm.faces.new((a[i], a[j], b[j], b[i]))
                vs.extend(f.verts)
        self._paint(set(vs), mat)
        return self

    def done(self, loc=(0, 0, 0), rz=0.0, zone=None, bevel=0.0):
        me = bpy.data.meshes.new(self.name)
        self.bm.to_mesh(me)
        self.bm.free()
        ob = self.kit.obj(self.name, me, self.mats, loc=loc, rot=(0, 0, rz), zone=zone)
        if bevel:
            self.kit.bevel(ob, bevel)
        return ob


class Kit:
    """One panel: geometry helpers, zone tags, camera, render and export."""

    def __init__(self, panel: str):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.panel = panel
        self.sc = bpy.context.scene
        self.col = self.sc.collection
        self.mats: dict[str, bpy.types.Material] = {}
        self.zones: list[str] = []          # label order
        self.anchor_at: dict[str, Vector] = {}
        self.cam = None

    # ------------------------------------------------------------------ materials
    def mat(self, key: str) -> bpy.types.Material:
        if key in self.mats:
            return self.mats[key]
        hexc, rough, metal, emit, alpha = PALETTE[key]
        m = bpy.data.materials.new(key)
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        rgb = lin(hexc)
        b.inputs["Base Color"].default_value = (*rgb, 1)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        if emit:
            b.inputs["Emission Color"].default_value = (*rgb, 1)
            b.inputs["Emission Strength"].default_value = emit
        if alpha < 1:
            b.inputs["Alpha"].default_value = alpha
        m.diffuse_color = (*rgb, 1)
        m.line_color = _line_colour(rgb)
        self.mats[key] = m
        return m

    # ------------------------------------------------------------------ objects
    def obj(self, name, mesh, mats=(), loc=(0, 0, 0), rot=(0, 0, 0), zone=None, smooth=False):
        for m in mats:
            mesh.materials.append(self.mat(m) if isinstance(m, str) else m)
        if smooth:
            for p in mesh.polygons:
                p.use_smooth = True
        ob = bpy.data.objects.new(name, mesh)
        ob.location = loc
        ob.rotation_euler = rot
        self.col.objects.link(ob)
        if zone:
            ob["zone"] = zone
        return ob

    def tag(self, obs, zone):
        for ob in obs if isinstance(obs, (list, tuple)) else [obs]:
            ob["zone"] = zone
        return obs

    def inst(self, ob, loc, rot=(0, 0, 0), zone=None):
        """Linked duplicate (shares the mesh) at a new place."""
        dup = ob.copy()
        dup.location = loc
        dup.rotation_euler = rot
        self.col.objects.link(dup)
        if zone:
            dup["zone"] = zone
        elif "zone" in dup:
            del dup["zone"]
        return dup

    def bevel(self, ob, width=0.03, segments=1):
        md = ob.modifiers.new("bevel", "BEVEL")
        md.width = width
        md.segments = segments
        md.limit_method = "ANGLE"
        return ob

    def box(self, c, s, mat, top=None, bevel=0.0, rz=0.0, zone=None, name="box", side=None):
        """Box standing on z=c[2], centred at (c[0], c[1]); s = (sx, sy, sz). `top` colours the
        top face (a cut wall shows its section colour), `side` the four sides."""
        hx, hy, sz = s[0] / 2, s[1] / 2, s[2]
        v = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0),
             (-hx, -hy, sz), (hx, -hy, sz), (hx, hy, sz), (-hx, hy, sz)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(v, [], f)
        mats = [mat] + ([top] if top else []) + ([side] if side else [])
        ob = self.obj(name, me, mats, loc=c, rot=(0, 0, rz), zone=zone)
        if top:
            me.polygons[1].material_index = 1
        if side:
            for i in (2, 3, 4, 5):
                me.polygons[i].material_index = len(mats) - 1
        if bevel:
            self.bevel(ob, bevel)
        return ob

    def rod(self, a, b, r, mat, verts=16, zone=None, name="rod", cap=True, r2=None):
        """Cylinder (or cone with r2) from point a to point b."""
        a, b = Vector(a), Vector(b)
        d = b - a
        L = d.length
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=verts,
                              radius1=r, radius2=r if r2 is None else r2, depth=L)
        for f in bm.faces:
            f.smooth = len(f.verts) == 4
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.link_faces[0].smooth != e.link_faces[1].smooth:
                e.smooth = False
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        rot = d.to_track_quat("Z", "Y").to_euler()
        return self.obj(name, me, [mat], loc=a + d / 2, rot=rot, zone=zone)

    def cyl(self, c, r, h, mat, verts=24, zone=None, name="cyl", r2=None):
        """Upright cylinder standing on c."""
        x, y, z = c
        return self.rod((x, y, z), (x, y, z + h), r, mat, verts, zone, name, r2=r2)

    def lathe(self, profile, mat, c=(0, 0, 0), verts=48, zone=None, name="lathe", close=True, arc=None, cap="cut"):
        """Surface of revolution around the z axis through c; profile = [(radius, z), ...] bottom to
        top. `arc` = (a0, a1) in degrees revolves only part way (a cutaway), leaving the cut open."""
        bm = bmesh.new()
        full = arc is None
        a0, a1 = (0.0, 360.0) if full else arc
        n = verts if full else max(2, int(verts * (a1 - a0) / 360) + 1)
        ang = [math.radians(a0 + (a1 - a0) * i / (verts if full else n - 1)) for i in range(n)]
        rings = []
        for r, z in profile:
            if r < 1e-5:
                rings.append([bm.verts.new((0, 0, z))] * n)
            else:
                rings.append([bm.verts.new((r * math.cos(t), r * math.sin(t), z)) for t in ang])
        steps = n if full else n - 1
        for a, b in zip(rings, rings[1:]):
            for i in range(steps):
                j = (i + 1) % n
                quad = [a[i], a[j], b[j], b[i]]
                uniq = list(dict.fromkeys(quad))
                if len(uniq) >= 3:
                    f = bm.faces.new(uniq)
                    f.smooth = True
        if close and full:
            for ring, up in ((rings[0], False), (rings[-1], True)):
                if ring[0].co.xy.length > 1e-4:
                    f = bm.faces.new(ring if up else list(reversed(ring)))
                    f.smooth = False
                    for e in f.edges:
                        e.smooth = False
        caps = []
        if close and not full:                    # section faces where the cut opens the shell
            loop = rings[:-1] if tuple(profile[0]) == tuple(profile[-1]) else rings
            for i in (0, n - 1):
                ring = list(dict.fromkeys(rg[i] for rg in loop))
                if len(ring) >= 3:
                    f = bm.faces.new(ring)
                    f.smooth = False
                    f.material_index = 1
                    caps.append(f)
                    for e in f.edges:
                        e.smooth = False
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        return self.obj(name, me, [mat] + ([cap] if caps else []), loc=c, zone=zone)

    def dome(self, c, r, mat, h=None, zone=None, name="dome", verts=48):
        h = r if h is None else h
        prof = [(r * math.cos(t), h * math.sin(t)) for t in np.linspace(0, math.pi / 2, 14)]
        prof[-1] = (0.0, h)
        return self.lathe(prof, mat, c, verts, zone, name)

    def tube(self, pts, r, mat, zone=None, name="tube", res=3):
        """Pipe/cable along a polyline (curve with a round bevel)."""
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = r
        cu.bevel_resolution = res
        cu.use_fill_caps = True
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, q in zip(sp.points, pts):
            p.co = (*q, 1)
        cu.materials.append(self.mat(mat) if isinstance(mat, str) else mat)
        ob = bpy.data.objects.new(name, cu)
        self.col.objects.link(ob)
        if zone:
            ob["zone"] = zone
        return ob

    def wire(self, a, b, sag, r, mat, zone=None, n=18):
        a, b = Vector(a), Vector(b)
        pts = []
        for i in range(n + 1):
            t = i / n
            p = a.lerp(b, t)
            p.z -= sag * 4 * t * (1 - t)
            pts.append(tuple(p))
        return self.tube(pts, r, mat, zone, "wire", res=1)

    def struts(self, segs, r, mat, zone=None, name="struts"):
        """Many thin square bars in one mesh (lattice towers, steel frames, cranes)."""
        bm = bmesh.new()
        for a, b in segs:
            a, b = Vector(a), Vector(b)
            d = b - a
            if d.length < 1e-6:
                continue
            q = d.to_track_quat("Z", "Y")
            m = Matrix.Translation((a + b) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r * 2, r * 2, d.length, 1))
            bmesh.ops.create_cube(bm, size=1.0, matrix=m)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        return self.obj(name, me, [mat], zone=zone)

    def join(self, obs, name=None):
        """Merge objects into the first (keeps materials); returns it."""
        obs = [o for o in obs if o]
        if len(obs) < 2:
            return obs[0] if obs else None
        ctx = {"active_object": obs[0], "selected_editable_objects": obs, "selected_objects": obs}
        with bpy.context.temp_override(**ctx):
            bpy.ops.object.join()
        ob = obs[0]
        ob.data.transform(ob.matrix_basis)       # bake so the result sits at the world origin
        ob.matrix_basis = Matrix.Identity(4)
        if name:
            ob.name = name
        return ob

    # ------------------------------------------------------------------ common parts
    def fan(self, c, r, mat="metal_dk", ring="metal", zone=None):
        """Horizontal fan seen from above: shroud ring, five blades, hub."""
        x, y, z = c
        parts = [self.lathe([(r, 0), (r, 0.12), (r * 0.92, 0.12), (r * 0.92, 0.02)], ring, (x, y, z), 32, zone, "shroud", close=False)]
        for k in range(5):
            a = 2 * math.pi * k / 5
            b = self.box((x + math.cos(a) * r * 0.45, y + math.sin(a) * r * 0.45, z + 0.03), (r * 0.8, r * 0.22, 0.03), mat, rz=a + 0.35, zone=zone, name="blade")
            parts.append(b)
        parts.append(self.cyl((x, y, z + 0.02), r * 0.16, 0.08, "metal_bk", 16, zone, "hub"))
        return parts

    def leds(self, origin, axis_u, axis_v, nu, nv, du, dv, size, mats=("led_g",), every=1, seed=0, zone=None):
        """A grid of small emissive squares on a face (server fronts, switch ports)."""
        ox, oy, oz = origin
        out = []
        rng = np.random.default_rng(seed)
        base = {}
        for i in range(nu):
            for j in range(nv):
                if every > 1 and rng.integers(0, every):
                    continue
                m = mats[int(rng.integers(0, len(mats)))]
                p = Vector((ox, oy, oz)) + Vector(axis_u) * (i * du) + Vector(axis_v) * (j * dv)
                if m not in base:
                    base[m] = self.box(tuple(p), size, m, zone=zone, name="led")
                    out.append(base[m])
                else:
                    out.append(self.inst(base[m], tuple(p), zone=zone))
        return out

    def slab(self, x0, x1, y0, y1, top="ground", depth=2.4, layers=("soil1", "soil2", "soil3"), z=0.0):
        """The diorama base: a top surface over soil strata whose cut faces show on the near sides."""
        h = depth / len(layers)
        for k, m in enumerate(layers):
            self.ground(self.box(((x0 + x1) / 2, (y0 + y1) / 2, z - depth + k * h), (x1 - x0, y1 - y0, h), m, name=f"soil{k}"))
        self.ground(self.box(((x0 + x1) / 2, (y0 + y1) / 2, z - 0.12), (x1 - x0, y1 - y0, 0.12), top, name="top"))

    def ground(self, ob):
        """Mark flat ground (slab, pads, roads): labels may sit over it at a small cost."""
        ob["ground"] = True
        return ob

    def catcher(self, c, size):
        """Invisible floor that only receives shadows (indoor objects float on the page)."""
        x, y, z = c
        me = bpy.data.meshes.new("catcher")
        me.from_pydata([(-size, -size, 0), (size, -size, 0), (size, size, 0), (-size, size, 0)], [], [(0, 1, 2, 3)])
        ob = self.obj("catcher", me, ["white"], loc=(x, y, z))
        ob.is_shadow_catcher = True
        ob["catcher"] = True
        return self.no_lines(ob)

    def no_lines(self, ob):
        """Keep an object out of the ink outlines (shadow catchers, light beams, steam)."""
        if "noline" not in bpy.data.collections:
            self.sc.collection.children.link(bpy.data.collections.new("noline"))
        nl = bpy.data.collections["noline"]
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        nl.objects.link(ob)
        return ob

    # ------------------------------------------------------------------ zones
    def spot(self, zone: str, at=None):
        """Declare that `zone` is labelled on this panel (label order = call order). `at` pins the
        anchor to a 3D point; otherwise the deepest visible pixel of the zone is used."""
        if zone not in self.zones:
            self.zones.append(zone)
        if at is not None:
            self.anchor_at[zone] = Vector(at)

    # ------------------------------------------------------------------ camera, light, render
    def camera(self, rot=(60, 0, 45), width=1200, margin=(0.05, 0.08), min_h=0, aspect=None):
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        self.col.objects.link(cam)
        self.sc.camera = cam
        cam.data.type = "ORTHO"
        cam.data.sensor_fit = "HORIZONTAL"
        cam.rotation_euler = tuple(math.radians(a) for a in rot)
        R = cam.rotation_euler.to_matrix()
        right, up, back = R @ Vector((1, 0, 0)), R @ Vector((0, 1, 0)), R @ Vector((0, 0, 1))
        xs, ys = [], []
        dg = bpy.context.evaluated_depsgraph_get()
        axes = np.array([tuple(right), tuple(up)])
        for ob in self.sc.objects:                   # real vertices: a bounding box overshoots spread-out parts
            if ob.type not in ("MESH", "CURVE") or ob.get("catcher"):
                continue
            ev = ob.evaluated_get(dg)
            me = ev.to_mesh()
            n = len(me.vertices)
            if n:
                co = np.empty(n * 3, dtype=np.float64)
                me.vertices.foreach_get("co", co)
                co = co.reshape(n, 3)
                mw = np.array(ev.matrix_world)
                w = co @ mw[:3, :3].T + mw[:3, 3]
                pr = w @ axes.T
                xs += [pr[:, 0].min(), pr[:, 0].max()]
                ys += [pr[:, 1].min(), pr[:, 1].max()]
            ev.to_mesh_clear()
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cw, ch = x1 - x0, y1 - y0
        fw = cw * (1 + 2 * margin[0])
        fh = ch + 2 * margin[1] * cw
        if aspect:
            fh = max(fh, fw * aspect)
            fw = max(fw, fh / aspect)
        W = width * SCALE
        H = max(int(round(W * fh / fw / 8)) * 8, min_h * SCALE)
        fh = fw * H / W
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cam.data.ortho_scale = fw
        cam.location = right * cx + up * cy + back * 200
        cam.data.clip_end = 1000
        self.sc.render.resolution_x = W
        self.sc.render.resolution_y = H
        self.sc.render.resolution_percentage = 100
        self.cam = cam
        return W // SCALE, H // SCALE

    def lights(self, sun_rot=(48, 8, -28), sun=2.6, sky=0.55, sky_color="#e4e9ef", angle=6):
        s = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        self.col.objects.link(s)
        s.data.energy = sun
        s.data.angle = math.radians(angle)
        s.rotation_euler = tuple(math.radians(a) for a in sun_rot)
        w = bpy.data.worlds.new("w")
        self.sc.world = w
        w.use_nodes = True
        bg = w.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (*lin(sky_color), 1)
        bg.inputs[1].default_value = sky

    def _cycles(self, samples, denoise=True):
        sc = self.sc
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        dev = None
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    dev = kind
                    break
            except TypeError:
                continue
        sc.cycles.device = "GPU" if dev else "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.use_denoising = denoise
        if denoise:
            sc.cycles.denoiser = "OPTIX" if dev == "OPTIX" else "OPENIMAGEDENOISE"
        sc.cycles.max_bounces = 6
        sc.cycles.transparent_max_bounces = 8
        sc.render.film_transparent = True
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.look = "None"
        sc.view_settings.exposure = 0
        sc.view_settings.gamma = 1

    def render(self, path, samples=128, line=1.3):
        sc = self.sc
        self._cycles(samples)
        sc.cycles.filter_width = 1.5
        sc.render.use_freestyle = line > 0
        if line > 0:
            sc.render.line_thickness_mode = "ABSOLUTE"
            sc.render.line_thickness = line
            fs = sc.view_layers[0].freestyle_settings
            fs.crease_angle = math.radians(120)
            ls = fs.linesets[0] if fs.linesets else fs.linesets.new("lines")
            ls.select_by_visibility = True
            ls.select_by_edge_types = True
            if "noline" in bpy.data.collections:
                ls.select_by_collection = True
                ls.collection = bpy.data.collections["noline"]
                ls.collection_negation = "EXCLUSIVE"
            ls.select_silhouette = True
            ls.select_border = True
            ls.select_crease = True
            ls.select_external_contour = True
            st = ls.linestyle or bpy.data.linestyles.new("ink")
            ls.linestyle = st
            st.color = (0.12, 0.12, 0.14)
            st.thickness = line
            if not any(m.type == "MATERIAL" for m in st.color_modifiers):
                cm = st.color_modifiers.new("by_material", "MATERIAL")
                cm.material_attribute = "LINE"
                cm.blend = "MIX"
                cm.influence = 1.0
        im = sc.render.image_settings
        im.file_format = "WEBP"
        im.color_mode = "RGBA"
        im.quality = 88
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)

    # ------------------------------------------------------------------ id pass + export
    def _id_colour(self, k):  # k >= 1; four linear levels per channel -> 63 distinct ids
        lv = (0.0, 0.2, 0.5, 1.0)
        return (lv[k % 4], lv[(k // 4) % 4], lv[(k // 16) % 4], 1.0)

    def export(self, json_path, id_png, img_name):
        """Render an object-colour pass (exact zone per pixel, occlusion respected) and write the
        anchor, highlight mask and occupancy grid for the page."""
        sc = self.sc
        order = list(self.zones)
        # one id per (zone, is-ground); id 1 = other objects, id 2 = bare ground
        keys = [(None, False), (None, True)] + [(z, False) for z in order] + [(z, True) for z in order]
        kid = {k: i + 1 for i, k in enumerate(keys)}
        for ob in self.sc.objects:
            if ob.type in ("MESH", "CURVE"):
                if ob.get("catcher"):
                    ob.hide_render = True
                    continue
                z = ob.get("zone")
                ob.color = self._id_colour(kid[(z if z in order else None, bool(ob.get("ground")))])
        em = bpy.data.materials.new("id")
        em.use_nodes = True
        nt = em.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        info = nt.nodes.new("ShaderNodeObjectInfo")
        emis = nt.nodes.new("ShaderNodeEmission")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(info.outputs["Color"], emis.inputs["Color"])
        nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
        vl = sc.view_layers[0]
        vl.material_override = em
        self._cycles(1, denoise=False)
        sc.cycles.filter_width = 0.01
        sc.cycles.use_adaptive_sampling = False
        sc.render.use_freestyle = False
        im = sc.render.image_settings
        im.file_format = "PNG"
        im.color_mode = "RGBA"
        im.color_depth = "8"
        sc.render.filepath = id_png
        bpy.ops.render.render(write_still=True)
        vl.material_override = None

        img = bpy.data.images.load(id_png)
        W, H = img.size
        px = np.empty(W * H * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(H, W, 4)[::-1]          # Blender stores rows bottom-up
        enc = np.array([0, 0.2, 0.5, 1.0]) ** (1 / 2.2)  # approx sRGB of the four levels
        def level(ch):
            return np.abs(ch[..., None] - enc[None, None, :]).argmin(-1)
        r, g, b, a = level(px[..., 0]), level(px[..., 1]), level(px[..., 2]), px[..., 3]
        ids = np.where(a > 0.5, r + 4 * g + 16 * b, 0)
        weight = np.zeros(len(keys) + 2)
        for k, i in kid.items():
            weight[i] = 0.3 if k[1] else 1.0                  # ground is cheap to cover, things are not
        occ = weight[np.clip(ids, 0, len(weight) - 1)]
        s = SCALE
        spots = []
        for z in order:
            m = (ids == kid[(z, False)]) | (ids == kid[(z, True)])
            n = int(m.sum())
            if not n:
                spots.append({"zone": z, "hidden": True})
                continue
            ys, xs = np.nonzero(m)
            anchor = self._anchor(z, m, W, H)
            spots.append({
                "zone": z,
                "anchor": [round(anchor[0] / s, 1), round(anchor[1] / s, 1)],
                "bbox": [round(xs.min() / s, 1), round(ys.min() / s, 1), round(xs.max() / s, 1), round(ys.max() / s, 1)],
                "area": round(n / s / s),
                "path": _mask_path(m, MASK * s, s),
            })
        cw = CELL * s
        rows = []
        for y in range(0, H, cw):
            row = ""
            for x in range(0, W, cw):
                f = occ[y:y + cw, x:x + cw].mean()
                row += "0123456789abcdef"[min(15, int(f * 15.99))]
            rows.append(row)
        data = {"panel": self.panel, "img": img_name, "w": W // s, "h": H // s,
                "spots": spots, "occ": {"cell": CELL, "rows": rows}}
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        return data

    def _anchor(self, zone, m, W, H):
        """A point well inside the zone's visible pixels: the pinned 3D point if it lands on the
        zone (else the nearest zone pixel), otherwise the deepest pixel (last to survive erosion)."""
        if zone in self.anchor_at:
            co = world_to_camera_view(self.sc, self.cam, self.anchor_at[zone])
            x, y = co.x * W, (1 - co.y) * H
            ys, xs = np.nonzero(m)
            d = (xs - x) ** 2 + (ys - y) ** 2
            k = int(d.argmin())
            if d[k] > (6 * SCALE) ** 2:
                print(f"[scenes] {self.panel}: anchor for {zone} is off its pixels; moved {math.sqrt(d[k]) / SCALE:.0f}px")
                return xs[k], ys[k]
            return x, y
        f = 4  # erode on a quarter-resolution mask
        h, w = H // f, W // f
        sm = m[:h * f, :w * f].reshape(h, f, w, f).mean((1, 3)) > 0.5
        if not sm.any():
            ys, xs = np.nonzero(m)
            return xs.mean(), ys.mean()
        cur = sm
        while True:
            nxt = cur.copy()
            nxt[1:, :] &= cur[:-1, :]
            nxt[:-1, :] &= cur[1:, :]
            nxt[:, 1:] &= cur[:, :-1]
            nxt[:, :-1] &= cur[:, 1:]
            if not nxt.any():
                break
            cur = nxt
        ys, xs = np.nonzero(cur)
        cy, cx = ys.mean(), xs.mean()
        k = int(((xs - cx) ** 2 + (ys - cy) ** 2).argmin())
        return (xs[k] + 0.5) * f, (ys[k] + 0.5) * f


def _mask_path(m, cell, s):
    """SVG path (CSS px) of the zone's pixels on a `cell` grid, as merged rectangles."""
    H, W = m.shape
    h, w = H // cell, W // cell
    g = m[:h * cell, :w * cell].reshape(h, cell, w, cell).mean((1, 3)) > 0.35
    rects, open_ = [], {}
    for y in range(h):
        runs, x = [], 0
        row = g[y]
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                runs.append((x0, x))
            else:
                x += 1
        nxt = {}
        for run in runs:
            if run in open_:
                r = open_.pop(run)
                r[3] += 1
                nxt[run] = r
            else:
                nxt[run] = [run[0], y, run[1] - run[0], 1]
        rects.extend(open_.values())
        open_ = nxt
    rects.extend(open_.values())
    k = cell / s
    return "".join(f"M{x * k:g} {y * k:g}h{w_ * k:g}v{h_ * k:g}h{-w_ * k:g}z" for x, y, w_, h_ in rects)
