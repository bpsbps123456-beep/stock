"""전력 — from fuel to the data centre's switchgear (stop 1 전력).

Left to right: a uranium mine (ore seen in the cut face of the ground) and an enrichment hall
of centrifuges; a nuclear plant (containment cut open to the reactor and steam generators,
turbine hall, cooling tower) with a row of small modular reactors in their pool; a gas-turbine
plant with the turbine opened; solar and battery containers; transmission towers into a
substation; the utility's control centre; and the data centre with fuel cells beside it.
"""

from __future__ import annotations

import math

from kit import PALETTE, Part

PALETTE.setdefault("tool", ("#eef0f2", 0.5, 0.05, 0, 1))
PALETTE.setdefault("containment", ("#e4e0d6", 0.7, 0.0, 0, 1))
PALETTE.setdefault("reactor", ("#d98a3d", 0.4, 0.3, 0.6, 1))
PALETTE.setdefault("pool", ("#4aa6d9", 0.1, 0.0, 0.3, 1))
PALETTE.setdefault("turbine", ("#9aa3ad", 0.35, 0.7, 0, 1))
PALETTE.setdefault("gaspipe", ("#e0b04c", 0.5, 0.2, 0, 1))
PALETTE.setdefault("cell", ("#f2f3f4", 0.5, 0.05, 0, 1))


def lattice(K, x, y, h=11.0, base=2.0, top=0.6, arm=3.0, zone=None, rot=0.0):
    """Transmission tower; returns the three conductor attachment points."""
    c, s = math.cos(rot), math.sin(rot)
    def P(dx, dy, z):
        return (x + dx * c - dy * s, y + dx * s + dy * c, z)
    segs = []
    b, t = base / 2, top / 2
    def at(z):
        return b + (t - b) * z / h
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for cx, cy in corners:
        segs.append((P(cx * b, cy * b, 0), P(cx * t, cy * t, h)))
    z = 0
    while z < h - 0.1:
        z2 = min(h, z + 1.5)
        a1, a2 = at(z), at(z2)
        for (cx, cy), (dx, dy) in zip(corners, corners[1:] + corners[:1]):
            segs.append((P(cx * a1, cy * a1, z), P(dx * a2, dy * a2, z2)))
        z = z2
    za = h - 1.4
    segs.append((P(0, -arm, za), P(0, arm, za)))
    segs.append((P(0, -arm, za), P(0, -at(za), za + 1.0)))
    segs.append((P(0, arm, za), P(0, at(za), za + 1.0)))
    K.struts(segs, 0.05, "metal_dk", zone, "tower")
    return [P(0, -arm, za - 0.3), P(0, 0, h + 0.2), P(0, arm, za - 0.3)]


def line(K, a_pts, b_pts, zone, sag=0.9):
    for a, b in zip(a_pts, b_pts):
        K.wire(a, b, sag, 0.05, "power", zone)


def power(K):
    K.slab(-50, 50, -14, 14, top="grass")
    # ------------------------------------------------------------ uranium: mine + enrichment (fuel)
    fuel = Part(K, "mine")
    fuel.box((-43.5, -14.02, -2.0), (7.0, 0.04, 1.4), "uranium")             # ore body in the cut face
    fuel.box((-41.0, -14.02, -1.0), (3.5, 0.04, 0.6), "uranium")
    fuel.box((-46.0, -14.03, -2.1), (0.9, 0.04, 2.1), "metal_bk")             # shaft seen in section
    fuel.box((-45.0, -14.03, -1.6), (2.2, 0.04, 0.35), "metal_bk")            # drift to the ore
    fuel.box((-46.0, -12.6, 0), (1.2, 1.2, 0.3), "concrete")
    K.tag(fuel.done(), "fuel")
    hf = []                                                                   # headframe over the shaft
    for dx in (-0.6, 0.6):
        hf.append(((-46.0 + dx, -12.6, 0.3), (-46.0 + dx * 0.4, -12.6, 6.0)))
    hf.append(((-46.6, -12.6, 2.5), (-45.4, -12.6, 2.5)))
    hf.append(((-46.6, -12.6, 4.5), (-45.4, -12.6, 4.5)))
    hf.append(((-45.6, -12.6, 6.0), (-44.0, -12.6, 0.3)))
    K.struts(hf, 0.09, "metal_dk", "fuel", "headframe")
    K.tag(K.rod((-46.0, -12.9, 6.0), (-46.0, -12.3, 6.0), 0.75, "metal", 20), "fuel")
    drums = Part(K, "yellowcake")
    for i in range(3):
        for j in range(2):
            drums.rod((-42.8 + i * 0.8, -11.6 + j * 0.8, 0), (-42.8 + i * 0.8, -11.6 + j * 0.8, 1.0), 0.36, "uranium", 16)
    K.tag(drums.done(), "fuel")
    enr = Part(K, "enrichment")                                               # hall opened to its centrifuges
    enr.box((-41.0, 4.0, 0), (14.0, 9.0, 0.2), "concrete")
    enr.box((-41.0, 8.4, 0.2), (14.0, 0.3, 5.0), "wall", top="cut")
    enr.box((-47.9, 4.0, 0.2), (0.3, 9.0, 5.0), "wall", top="cut")
    enr.box((-41.0, -0.4, 0.2), (14.0, 0.3, 0.7), "wall", top="cut")
    for i in range(10):
        for j in range(4):
            enr.rod((-46.5 + i * 1.2, 1.4 + j * 1.6, 0.2), (-46.5 + i * 1.2, 1.4 + j * 1.6, 3.6), 0.3, "white", 14)
            enr.rod((-46.5 + i * 1.2, 1.4 + j * 1.6, 3.6), (-46.5 + i * 1.2, 1.4 + j * 1.6, 3.75), 0.22, "metal_dk", 10)
    K.tag(enr.done(), "fuel")

    # ------------------------------------------------------------ nuclear + SMRs (nuclear)
    cx, cy = -26.0, 2.0
    K.tag(K.lathe([(4.2, 0), (4.2, 7.0), (4.0, 7.0), (4.0, 0.3), (0, 0.3), (0, 0), (4.2, 0)], "containment",
                  (cx, cy, 0), 64, name="containment", arc=(-15, 255)), "nuclear")
    K.tag(K.lathe([(4.2 * math.cos(t), 7.0 + 4.2 * math.sin(t)) for t in [i * math.pi / 2 / 12 for i in range(13)]]
                  + [(4.0 * math.cos(t), 7.0 + 4.0 * math.sin(t)) for t in [math.pi / 2 - i * math.pi / 2 / 12 for i in range(13)]]
                  + [(4.2, 7.0)], "containment", (cx, cy, 0), 64, name="dome", arc=(-15, 255)), "nuclear")
    core = Part(K, "reactor")
    core.rod((cx, cy, 0.3), (cx, cy, 4.6), 1.3, "reactor", 32)                  # reactor vessel
    core.rod((cx, cy, 4.6), (cx, cy, 5.0), 1.0, "metal_dk", 32)
    for dx, dy in ((-2.4, 1.2), (2.2, 1.6)):                                  # steam generators
        core.rod((cx + dx, cy + dy, 0.3), (cx + dx, cy + dy, 6.0), 0.75, "metal", 24)
        core.rod((cx + dx * 0.5, cy + dy * 0.5, 2.0), (cx + dx, cy + dy, 2.0), 0.25, "hot", 12)
    K.tag(core.done(), "nuclear")
    th = Part(K, "turbine_hall")
    th.box((-18.5, 3.0, 0), (8.0, 7.0, 6.5), "containment", top="roof")
    th.box((-18.5, -0.52, 1.0), (6.0, 0.04, 2.5), "window")
    K.tag(th.done(), "nuclear")
    prof = []
    for k in range(13):
        z = 14.0 * k / 12
        r = 3.1 + 2.3 * ((z - 9.5) / 9.5) ** 2 if z < 9.5 else 3.1 + 0.9 * ((z - 9.5) / 4.5) ** 2
        prof.append((r, z))
    K.tag(K.lathe(prof + [(prof[-1][0] - 0.25, 14.0), (prof[0][0] - 0.25, 0.4)], "containment", (-31.0, 9.0, 0), 64,
                  name="cooling_tower", close=False), "nuclear")
    for dx, dy, dz, r in ((0, 0, 15.5, 3.0), (1.4, 0.6, 17.6, 3.3), (-0.8, 1.2, 19.4, 2.8), (2.4, -0.4, 20.6, 2.2)):
        K.no_lines(K.dome((-31.0 + dx, 9.0 + dy, dz - r * 0.6), r, "steam", h=r * 1.2, zone="nuclear", name="steam", verts=24))
    smr = Part(K, "smr")                                                    # small modular reactors in a pool
    smr.box((-24.0, -9.0, 0), (9.0, 6.0, 0.25), "concrete")
    smr.box((-24.0, -6.15, 0.25), (9.0, 0.3, 4.5), "wall", top="cut")
    smr.box((-28.35, -9.0, 0.25), (0.3, 6.0, 4.5), "wall", top="cut")
    smr.box((-24.0, -11.85, 0.25), (9.0, 0.3, 0.8), "wall", top="cut")
    smr.box((-24.0, -9.0, 0.25), (7.6, 4.6, 0.9), "concrete")
    smr.box((-24.0, -9.0, 1.15), (7.2, 4.2, 0.05), "pool")
    for k in range(3):
        x = -26.6 + k * 2.6
        smr.rod((x, -9.0, 0.6), (x, -9.0, 3.2), 0.7, "metal", 24)
        smr.rod((x, -9.0, 3.2), (x, -9.0, 3.6), 0.45, "reactor", 20)
    K.tag(smr.done(), "nuclear")

    # ------------------------------------------------------------ gas turbine plant (gen)
    gx = -5.0
    gt = Part(K, "gas_plant")
    gt.box((gx, 1.0, 0), (12.0, 8.0, 0.25), "concrete")
    gt.box((gx, 4.85, 0.25), (12.0, 0.3, 6.0), "wall", top="cut")
    gt.box((gx - 5.85, 1.0, 0.25), (0.3, 8.0, 6.0), "wall", top="cut")
    gt.box((gx, -2.85, 0.25), (12.0, 0.3, 0.8), "wall", top="cut")
    gt.rod((gx - 4.2, 0.5, 2.0), (gx + 1.8, 0.5, 2.0), 1.2, "turbine", 32, r2=1.6)      # compressor → turbine
    for k in range(9):
        x = gx - 3.8 + k * 0.65
        gt.ring((x, 0.5, 2.0), 1.25 + k * 0.05, 0.12, 0.25, "metal_dk", 32, axis="X")   # blade rows
    gt.box((gx + 3.4, 0.5, 0.25), (2.6, 2.6, 3.3), "metal")                  # generator
    gt.box((gx + 3.4, 0.5, 0.25), (2.8, 2.8, 0.6), "metal_dk")
    gt.box((gx - 5.0, 0.5, 0.25), (1.6, 2.2, 3.4), "metal")                  # air intake
    gt.box((gx + 9.0, 1.0, 0), (4.0, 5.5, 8.0), "metal")                     # heat-recovery boiler
    gt.rod((gx + 10.2, 1.0, 8.0), (gx + 10.2, 1.0, 16.0), 0.9, "metal_dk", 24)   # stack
    gt.rod((gx + 1.8, 0.5, 2.0), (gx + 7.0, 0.5, 2.0), 1.0, "metal_dk", 20)
    gt.rod((gx - 8.0, -6.0, 0.6), (gx - 8.0, 0.5, 0.6), 0.3, "gaspipe", 12)        # gas pipeline in
    gt.rod((gx - 8.0, 0.5, 0.6), (gx - 4.6, 0.5, 0.6), 0.3, "gaspipe", 12)
    gt.rod((gx - 8.0, -6.0, -1.0), (gx - 8.0, -6.0, 0.6), 0.3, "gaspipe", 12)
    K.tag(gt.done(), "gen")
    for dz, r in ((17.2, 1.4), (18.9, 1.9), (20.4, 1.5)):
        K.no_lines(K.dome((gx + 10.2 + (dz - 17) * 0.4, 1.0, dz - r * 0.6), r, "smoke", h=r * 1.1, zone="gen", name="smoke", verts=20))

    # ------------------------------------------------------------ solar and batteries (storage_power)
    sol = Part(K, "solar")
    for i in range(5):
        for j in range(3):
            x, y = 9.0 + i * 3.0, -1.0 + j * 3.2
            sol.box((x, y, 1.0), (2.6, 2.0, 0.08), "solar", rx=0.35)
            sol.box((x, y - 0.3, 0), (0.1, 0.1, 1.0), "metal_dk")
            sol.box((x, y + 0.5, 0), (0.1, 0.1, 1.4), "metal_dk")
    for k in range(4):
        x = 9.5 + k * 3.4
        sol.box((x, -8.5, 0), (2.6, 5.0, 2.6), "white")
        sol.box((x, -11.02, 0.6), (2.0, 0.04, 1.6), "metal_dk")
        sol.box((x, -8.5, 2.6), (1.4, 1.6, 0.6), "metal")
        sol.box((x + 0.9, -11.03, 2.0), (0.2, 0.02, 0.2), "led_g")
    K.tag(sol.done(), "storage_power")

    # ------------------------------------------------------------ transmission and substation (grid)
    towers = [lattice(K, x, 9.0, zone="grid", rot=math.pi / 2) for x in (-12.0, 4.0, 20.0)]
    for a, b in zip(towers, towers[1:]):
        line(K, a, b, "grid")
    line(K, [(p[0] - 14.0, p[1], p[2] - 2.0) for p in towers[0]], towers[0], "grid")
    sub = Part(K, "substation")
    sub.box((30.0, 4.0, 0), (10.0, 8.0, 0.08), "gravel")
    for k, x in enumerate((27.5, 32.0)):
        sub.box((x, 4.0, 0.08), (2.6, 3.0, 0.3), "concrete")
        sub.box((x, 4.0, 0.38), (2.0, 2.6, 2.2), "transformer")
        for side in (-1, 1):
            for f in range(8):
                sub.box((x + side * 1.35, 2.9 + f * 0.32, 0.6), (0.6, 0.07, 1.6), "transformer")
        sub.rod((x - 0.7, 5.0, 3.0), (x + 0.7, 5.0, 3.0), 0.32, "transformer", 16)
        for b in (-0.6, 0.0, 0.6):
            sub.rod((x + b, 3.4, 2.58), (x + b, 3.4, 4.2), 0.1, "porcelain", 10, r2=0.06)
    for gx2 in (25.0, 35.0):
        sub.box((gx2, 7.5, 0.08), (0.25, 0.25, 6.0), "metal_dk")
    sub.box((30.0, 7.5, 6.0), (10.4, 0.25, 0.25), "metal_dk")
    K.tag(sub.done(), "grid")
    fence = []
    for a, b in (((24.5, -0.3), (35.5, -0.3)), ((35.5, -0.3), (35.5, 8.3)), ((24.5, -0.3), (24.5, 8.3))):
        n = max(2, int(math.dist(a, b) / 1.5))
        for i in range(n + 1):
            t = i / n
            fence.append(((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0), (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 1.4)))
        fence.append(((a[0], a[1], 1.35), (b[0], b[1], 1.35)))
    K.struts(fence, 0.03, "metal", None, "fence")
    line(K, towers[-1], [(25.0, 7.5, 5.8), (30.0, 7.5, 6.2), (35.0, 7.5, 5.8)], "grid", sag=0.6)
    K.tube([(33.5, 4.0, 0.15), (37.0, 1.0, 0.15), (40.5, 1.0, 0.15)], 0.16, "power", "grid")

    # ------------------------------------------------------------ utility control centre (utility)
    ut = Part(K, "control_centre")
    ut.box((29.5, -8.0, 0), (9.0, 6.0, 0.2), "concrete")
    ut.box((29.5, -5.15, 0.2), (9.0, 0.3, 4.2), "wall", top="cut")
    ut.box((25.15, -8.0, 0.2), (0.3, 6.0, 4.2), "wall", top="cut")
    ut.box((29.5, -10.85, 0.2), (9.0, 0.3, 0.8), "wall", top="cut")
    ut.box((29.5, -5.4, 1.0), (7.0, 0.1, 2.6), "metal_bk")                  # the grid map wall
    for i in range(6):
        for j in range(3):
            ut.box((26.6 + i * 1.15, -5.47, 1.3 + j * 0.8), (1.0, 0.04, 0.6), "screen")
    for i in range(4):
        ut.box((26.8 + i * 1.8, -8.2, 0.2), (1.4, 0.8, 0.8), "white")
        ut.box((26.8 + i * 1.8, -8.0, 1.0), (1.0, 0.06, 0.5), "screen", rx=-0.25)
    K.tag(ut.done(), "utility")

    # ------------------------------------------------------------ data centre + fuel cells (onsite, power_dist)
    dc = Part(K, "datacentre")
    dc.box((44.5, 4.0, 0), (9.0, 12.0, 0.2), "concrete")
    dc.box((44.5, 9.85, 0.2), (9.0, 0.3, 5.0), "wall", top="cut")
    dc.box((40.15, 4.0, 0.2), (0.3, 12.0, 5.0), "wall", top="cut")
    dc.box((44.5, -1.85, 0.2), (9.0, 0.3, 0.8), "wall", top="cut")
    for i in range(6):
        for j in range(2):
            dc.box((42.0 + i * 1.15, 3.0 + j * 4.0, 0.2), (1.0, 1.3, 2.4), "rack")
            dc.box((42.0 + i * 1.15, 2.33 + j * 4.0, 1.0), (0.1, 0.02, 0.1), "led_g")
    K.tag(dc.done(), "power_dist")
    sw = Part(K, "dc_switchgear")
    for k in range(4):
        sw.box((41.2, -0.6 + k * 1.0, 0.2), (0.9, 0.9, 2.2), "metal")
    sw.box((41.2, 1.0, 2.4), (0.6, 4.2, 0.3), "power")
    K.tag(sw.done(), "power_dist")
    fc = Part(K, "fuel_cells")
    for k in range(5):
        x = 41.5 + k * 1.6
        fc.box((x, -8.0, 0), (1.3, 3.2, 2.0), "cell")
        fc.box((x, -9.62, 0.4), (0.9, 0.04, 1.2), "metal_dk")
        fc.box((x + 0.45, -9.63, 1.6), (0.12, 0.02, 0.12), "led_g")
    fc.rod((40.0, -6.2, 0.4), (49.0, -6.2, 0.4), 0.18, "gaspipe", 10)
    K.tag(fc.done(), "onsite")
    K.tube([(44.5, -6.2, 0.4), (44.5, -2.2, 0.4), (41.2, -2.2, 0.4), (41.2, -0.6, 0.4)], 0.14, "power", "onsite")

    for tx, ty_, s in ((-49, 12.5, 1.2), (-36, 12.5, 1.0), (-1, 12.6, 1.1), (14, 12.4, 1.0), (38, 12.6, 1.2), (49, -12.6, 1.0)):
        K.cyl((tx, ty_, 0), 0.15 * s, 1.0 * s, "trunk", 8)
        K.dome((tx, ty_, 0.8 * s), 1.1 * s, "tree", h=2.0 * s, verts=16)

    for zone in ("fuel", "nuclear", "gen", "storage_power", "grid", "utility", "onsite", "power_dist"):
        K.spot(zone)
    return {"width": 1200, "margin": (0.03, 0.06), "rot": (60, 0, 30)}


PANELS = {"power": power}
