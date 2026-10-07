"""반도체 공장 — fab and packaging cutaway, then one chip package exploded
(stops 2 소재, 3 장비, 4 반도체 공장, 5 패키징).

fab: a two-level front-end fab (cleanroom over the sub-fab) and a back-end hall beside it. A wafer
goes left to right: EUV lithography, etch/deposition cluster tools (one chamber opened to its
consumable rings), inspection, wafer probe; gases and chemicals come up from the sub-fab and
the gas yard; the back-end stacks HBM, assembles CoWoS packages and tests them.
fab_pkg: one GPU package pulled apart layer by layer — dies and HBM stacks, micro-bumps,
silicon interposer, C4 bumps, ABF substrate (glass-cloth core) with its capacitors and balls.
"""

from __future__ import annotations

import math

from kit import PALETTE, Part

PALETTE.setdefault("tile", ("#cfd2d4", 0.7, 0.05, 0, 1))
PALETTE.setdefault("cleanfloor", ("#dfe3e6", 0.6, 0.05, 0, 1))
PALETTE.setdefault("tool", ("#eef0f2", 0.5, 0.05, 0, 1))
PALETTE.setdefault("tool_dk", ("#3e4a5c", 0.5, 0.2, 0, 1))
PALETTE.setdefault("ring", ("#8e3b3b", 0.45, 0.2, 0, 1))
PALETTE.setdefault("pump", ("#5b7fae", 0.5, 0.2, 0, 1))
PALETTE.setdefault("gas1", ("#7fb069", 0.5, 0.2, 0, 1))
PALETTE.setdefault("gas2", ("#e0b04c", 0.5, 0.2, 0, 1))
PALETTE.setdefault("gas3", ("#b55d8f", 0.5, 0.2, 0, 1))
PALETTE.setdefault("abf", ("#4f8a52", 0.55, 0.0, 0, 1))
PALETTE.setdefault("glasscloth", ("#d8d2b8", 0.6, 0.0, 0, 1))

SUB = 4.2          # sub-fab height
SLAB = 0.6         # waffle slab
FL = SUB + SLAB + 0.4   # cleanroom floor level (raised floor on top of the slab)
TOP = 11.2         # cleanroom ceiling


def foup(p, x, y, z, rz=0.0):
    p.box((x, y, z), (0.46, 0.42, 0.38), "white", rz=rz)
    p.box((x, y, z + 0.38), (0.3, 0.2, 0.06), "metal_dk", rz=rz)


# ------------------------------------------------------------------------------ tools
def euv(K, x0, y0):
    """ASML-like scanner: long white body, source vessel, cut window with mirrors and light."""
    z = FL
    body = Part(K, "euv")
    body.box((x0, y0, z), (8.6, 4.6, 4.4), "tool")
    body.box((x0 + 4.35, y0, z + 0.4), (0.12, 4.0, 3.2), "tool_dk")          # service side panels
    body.box((x0 - 1.0, y0 - 2.32, z + 0.3), (6.0, 0.06, 0.5), "tool_dk")
    body.box((x0 + 3.6, y0 - 2.6, z), (1.6, 0.6, 1.0), "tool")               # wafer handler
    foup(body, x0 + 3.6, y0 - 3.1, z + 1.0)
    body.rod((x0 - 5.6, y0 + 0.4, z + 2.2), (x0 - 3.9, y0 + 0.4, z + 2.2), 1.15, "metal", 24)   # source vessel
    body.box((x0 - 4.9, y0 + 0.4, z - 0.1), (1.6, 1.6, 1.1), "tool")
    sc = body.done()
    K.tag(sc, "litho")
    cut = Part(K, "euv_window")                                            # the opened front shows the optics
    cut.box((x0 - 0.6, y0 - 2.34, z + 1.0), (5.0, 0.08, 3.0), "metal_bk")
    for mx, mz in ((-2.4, 1.6), (-0.9, 3.5), (0.6, 1.5), (1.8, 3.2)):
        cut.rod((x0 + mx, y0 - 2.45, z + mz), (x0 + mx, y0 - 2.3, z + mz), 0.42, "metal", 20)
    K.tag(cut.done(), "litho")
    pts = [(x0 - 3.9, y0 - 2.5, z + 2.2), (x0 - 2.4, y0 - 2.5, z + 1.6), (x0 - 0.9, y0 - 2.5, z + 3.5),
           (x0 + 0.6, y0 - 2.5, z + 1.5), (x0 + 1.8, y0 - 2.5, z + 3.2), (x0 + 1.9, y0 - 2.5, z + 1.1)]
    K.no_lines(K.tube(pts, 0.07, "euv", "litho", "euv_light"))


def cluster(K, x0, y0, opened=False):
    """Etch/deposition cluster tool: transfer hub, four chambers, front end with load ports."""
    z = FL
    t = Part(K, "cluster")
    t.box((x0, y0 - 2.5, z), (3.2, 1.3, 2.2), "tool")                      # EFEM
    for k in (-1, 0, 1):
        t.box((x0 + k * 1.0, y0 - 3.35, z), (0.7, 0.45, 0.95), "tool_dk")    # load ports
        foup(t, x0 + k * 1.0, y0 - 3.35, z + 0.95)
    t.rod((x0, y0, z), (x0, y0, z + 1.1), 1.25, "metal", 6)                 # hexagonal transfer chamber
    t.box((x0, y0, z + 1.1), (1.2, 1.2, 0.12), "metal_dk", rz=0.5)
    chambers = [(-1.95, 0.6), (-1.3, 2.1), (1.3, 2.1), (1.95, 0.6)]
    for k, (dx, dy) in enumerate(chambers):
        if opened and k == 3:
            continue
        t.rod((x0 + dx, y0 + dy, z), (x0 + dx, y0 + dy, z + 1.6), 0.8, "tool", 24)
        t.rod((x0 + dx, y0 + dy, z + 1.6), (x0 + dx, y0 + dy, z + 1.85), 0.62, "metal", 24)
        t.box((x0 + dx, y0 + dy, z + 1.85), (0.5, 0.5, 0.6), "tool_dk")      # RF match / gas box
    K.tag(t.done(), "equip")
    if opened:                                                              # chamber cut open: the consumables
        dx, dy = chambers[3]
        c = (x0 + dx, y0 + dy, z)
        K.tag(K.lathe([(0.8, 0), (0.8, 1.6), (0.7, 1.6), (0.7, 0.12), (0.0, 0.12), (0.0, 0), (0.8, 0)], "tool", c, 48,
                      name="chamber_cut", arc=(0, 270)), "equip")
        parts = Part(K, "chamber_parts")
        parts.rod((c[0], c[1], z + 0.12), (c[0], c[1], z + 0.55), 0.46, "metal_dk", 32)          # electrostatic chuck
        parts.rod((c[0], c[1], z + 0.55), (c[0], c[1], z + 0.58), 0.4, "wafer", 32)            # the wafer
        parts.ring((c[0], c[1], z + 0.5), 0.6, 0.18, 0.12, "ring", 40)                           # focus ring
        parts.rod((c[0], c[1], z + 1.25), (c[0], c[1], z + 1.36), 0.55, "silicon", 32)          # showerhead
        K.tag(parts.done(), "tool_parts")
        glow = K.rod((c[0], c[1], z + 0.62), (c[0], c[1], z + 1.2), 0.44, "plasma", 32, "tool_parts", "plasma")
        K.no_lines(glow)


def metrology(K, x0, y0):
    z = FL
    t = Part(K, "inspect")
    t.box((x0, y0, z), (2.6, 2.2, 1.5), "tool")
    t.box((x0, y0 - 1.15, z + 0.4), (2.0, 0.06, 0.7), "tool_dk")
    t.rod((x0 + 0.3, y0 + 0.2, z + 1.5), (x0 + 0.3, y0 + 0.2, z + 3.4), 0.42, "tool_dk", 24)   # optical column
    t.rod((x0 + 0.3, y0 + 0.2, z + 3.4), (x0 + 0.3, y0 + 0.2, z + 3.6), 0.55, "metal", 24)
    t.box((x0 - 1.6, y0 - 0.4, z), (0.6, 1.6, 0.9), "tool")
    foup(t, x0 - 1.6, y0 - 0.6, z + 0.9)
    t.box((x0 - 1.0, y0 - 1.25, z + 1.6), (0.9, 0.08, 0.6), "screen", rx=-0.2)
    t.box((x0 - 1.0, y0 - 1.15, z + 1.5), (0.08, 0.08, 0.15), "metal_dk")
    K.tag(t.done(), "metrology")


def prober(K, x0, y0):
    """Wafer probe: prober with the probe card exposed, test head swung up, tester beside it."""
    z = FL
    p = Part(K, "prober")
    p.box((x0, y0, z), (2.4, 2.4, 1.3), "tool")
    p.box((x0 - 1.7, y0 - 0.3, z), (0.9, 1.5, 0.9), "tool")
    foup(p, x0 - 1.7, y0 - 0.5, z + 0.9)
    K.tag(p.done(), "test")
    card = Part(K, "probe_card")
    card.rod((x0, y0, z + 1.3), (x0, y0, z + 1.42), 0.95, "pcb", 40)
    card.ring((x0, y0, z + 1.42), 0.35, 0.12, 0.08, "gold", 24)
    for k in range(16):
        a = 2 * math.pi * k / 16
        card.box((x0 + 0.62 * math.cos(a), y0 + 0.62 * math.sin(a), z + 1.42), (0.12, 0.12, 0.08), "gold")
    K.tag(card.done(), "socket")
    head = Part(K, "test_head")
    head.box((x0 + 0.2, y0 + 1.7, z + 2.0), (2.2, 1.0, 1.9), "tool_dk")       # test head, swung up
    head.box((x0 + 0.2, y0 + 1.7, z + 1.3), (0.6, 0.6, 0.7), "metal_dk")
    head.box((x0 + 2.6, y0 + 1.6, z), (1.6, 1.4, 2.6), "tool_dk")            # tester mainframe
    head.box((x0 + 2.6, y0 + 0.88, z + 1.9), (1.0, 0.04, 0.5), "screen")
    head.rod((x0 + 1.3, y0 + 1.7, z + 3.4), (x0 + 2.2, y0 + 1.6, z + 2.6), 0.25, "metal_bk", 12)
    K.tag(head.done(), "test")


# ------------------------------------------------------------------------------ fab
def fab(K):
    K.slab(-28, 34, -16, 13, top="grass")
    K.ground(K.box((3, -1.5, 0), (60, 28, 0.02), "gravel", name="site"))
    X0, X1, Y0, Y1 = -24, 12, -8, 9
    CF = -4.6                                   # the cleanroom floor stops here: a stepped section
    # ---- building (foundry): sub-fab floor, waffle slab cut at CF, raised floor, back/left walls
    shell = Part(K, "fab_shell")
    shell.box(((X0 + X1) / 2, (Y0 + Y1) / 2, 0), (X1 - X0, Y1 - Y0, 0.2), "concrete")
    shell.box(((X0 + X1) / 2, (CF + Y1) / 2, SUB), (X1 - X0, Y1 - CF, SLAB), "concrete", top="concrete")
    for k in range(int(X1 - X0)):                                          # waffle holes seen in section
        shell.box((X0 + 0.5 + k, CF - 0.01, SUB + 0.12), (0.5, 0.04, SLAB - 0.24), "metal_bk")
    shell.box(((X0 + X1) / 2, (CF + Y1) / 2, SUB + SLAB), (X1 - X0, Y1 - CF, 0.4), "cleanfloor")
    shell.box(((X0 + X1) / 2, Y1 - 0.2, 0), (X1 - X0, 0.4, TOP + 1.0), "wall", top="cut")
    shell.box((X0 + 0.2, (Y0 + Y1) / 2, 0), (0.4, Y1 - Y0, TOP + 1.0), "wall", top="cut")
    shell.box(((X0 + X1) / 2, Y0 + 0.2, 0), (X1 - X0, 0.4, 0.6), "wall", top="cut")       # front, cut low
    shell.box((X1 - 0.2, (Y0 + Y1) / 2, 0), (0.4, Y1 - Y0, 0.6), "wall", top="cut")
    for x in range(int(X0) + 4, int(X1), 6):                               # columns through both levels
        shell.box((x, Y1 - 0.7, 0), (0.6, 0.6, TOP + 1.0), "concrete", top="cut")
    K.tag(shell.done(), "foundry")
    # hero wafer on a stand at the start of the line: what the foundry makes
    wf = Part(K, "hero_wafer")
    wx, wy = X0 + 2.6, CF + 1.6
    wf.rod((wx, wy, FL), (wx, wy, FL + 1.0), 0.08, "metal_dk", 8)
    wf.rod((wx, wy - 0.05, FL + 2.3), (wx, wy + 0.05, FL + 2.3), 1.4, "wafer", 64)
    for i in range(-5, 6):
        for j in range(-5, 6):
            if (i * 0.24) ** 2 + (j * 0.24) ** 2 < 1.25 ** 2:
                wf.box((wx + i * 0.24, wy - 0.07, FL + 2.3 - 0.1 + j * 0.24), (0.2, 0.02, 0.2), "die")
    K.tag(wf.done(), "foundry")

    # ---- cleanroom tools along the line
    euv(K, -14.0, 0.6)
    cluster(K, -4.8, 0.3, opened=True)
    cluster(K, 0.9, 0.3)
    metrology(K, 5.8, -1.0)
    prober(K, 8.4, -1.8)

    # ---- overhead transport (OHT) and the fan-filter ceiling (fab_facility)
    fac = Part(K, "oht")
    for y in (-3.3, 3.4):
        fac.box(((X0 + X1) / 2 + 1.0, y, TOP - 1.3), (X1 - X0 - 3, 0.25, 0.18), "metal")
        for x in range(int(X0) + 3, int(X1) - 1, 4):
            fac.box((x, y, TOP - 1.12), (0.08, 0.08, 1.12 + 1.0), "metal_dk")
    for x in (-19.5, -11.0, -4.8, 0.9, 7.0):
        fac.box((x, -3.3, TOP - 1.85), (0.7, 0.5, 0.55), "tool")
        foup(fac, x, -3.3, TOP - 2.3)
    for x in range(int(X0) + 1, int(X1) - 1, 2):
        for y in (4.8, 6.8):
            fac.box((x + 0.9, y + 0.9, TOP), (1.8, 1.8, 0.25), "white")
            fac.rod((x + 0.9, y + 0.9, TOP - 0.02), (x + 0.9, y + 0.9, TOP + 0.01), 0.45, "metal_dk", 16)
    for x in range(int(X0) + 1, int(X1), 4):                               # ceiling joists over the cleanroom
        fac.box((x, (CF + Y1) / 2, TOP + 0.9), (0.25, Y1 - CF, 0.3), "metal_dk")
    K.tag(fac.done(), "fab_facility")

    # ---- sub-fab, in the open strip in front of the slab: pumps, scrubbers, chillers (fab_facility)
    sf = Part(K, "subfab")
    for x in (-6.6, -3.0, -0.9, 2.7):
        sf.box((x, -6.4, 0.2), (1.3, 1.0, 1.1), "pump")
        sf.box((x, -6.4, 1.3), (0.9, 0.6, 0.2), "metal_dk")
        sf.rod((x + 0.3, -6.4, 1.5), (x + 0.3, -6.4, 3.6), 0.09, "metal", 8)
        sf.rod((x + 0.3, -6.4, 3.6), (x + 0.3, CF + 0.4, 3.6), 0.09, "metal", 8)
    for x in (7.6, 9.3):
        sf.rod((x, -6.4, 0.2), (x, -6.4, 3.1), 0.62, "white", 24)
        sf.box((x, -6.4, 3.1), (0.6, 0.6, 0.6), "metal_dk")
    sf.box((5.0, -6.4, 0.2), (1.6, 1.2, 1.5), "metal")                        # chiller
    K.tag(sf.done(), "fab_facility")
    for k, x in enumerate((-5.9, -2.3, -0.2, 3.4)):
        K.rod((x, -6.0, 1.3), (x, -6.0, 3.95), 0.08, ("gas1", "gas2", "gas3")[k % 3], 8, "fab_facility", "riser")
    # gas cabinets and chemicals (materials)
    mat = Part(K, "chem")
    for k, x in enumerate((-22.2, -20.7, -19.2)):
        mat.box((x, -6.5, 0.2), (1.3, 1.0, 2.6), "tool")
        mat.box((x, -7.02, 0.6), (1.0, 0.04, 1.8), "glass")
        for c in range(3):
            mat.rod((x - 0.32 + c * 0.32, -6.6, 0.3), (x - 0.32 + c * 0.32, -6.6, 1.9), 0.12, ("gas1", "gas2", "gas3")[(c + k) % 3], 12)
    for k in range(4):
        mat.rod((-11.0 + k * 0.75, -7.0, 0.2), (-11.0 + k * 0.75, -7.0, 1.0), 0.3, "cold" if k % 2 else "white", 16)
    K.tag(mat.done(), "materials")
    yard = Part(K, "gas_yard")                                             # bulk gases outside
    for k, x in enumerate((-22.0, -19.4, -16.8)):
        yard.rod((x, -12.6, 0.0), (x, -12.6, 1.0), 0.15, "metal_dk", 8)
        yard.rod((x, -12.6, 1.0), (x, -12.6, 6.5), 1.0, "white", 32)
        yard.rod((x, -12.6, 6.5), (x, -12.6, 6.8), 0.6, "white", 24)
    yard.rod((-22.0, -11.5, 0.6), (-22.0, Y0 + 0.3, 0.6), 0.12, "metal", 8)
    K.tag(yard.done(), "materials")
    # the EUV source's drive laser lives down here too and fires up through the slab (litho)
    las = Part(K, "drive_laser")
    las.box((-16.5, -6.4, 0.3), (6.8, 1.5, 1.5), "tool_dk")
    las.rod((-19.6, -6.4, 1.8), (-19.6, -6.4, 3.8), 0.22, "metal", 12)
    las.rod((-19.6, -6.4, 3.8), (-19.6, 1.0, 3.8), 0.22, "metal", 12)
    K.tag(las.done(), "litho")

    # ---- back-end hall (packaging) beside the fab
    B0, B1 = 14.0, 32.0
    hall = Part(K, "pkg_hall")
    hall.box(((B0 + B1) / 2, (Y0 + Y1) / 2, 0), (B1 - B0, Y1 - Y0, 0.3), "cleanfloor")
    hall.box(((B0 + B1) / 2, Y1 - 0.2, 0), (B1 - B0, 0.4, 6.5), "wall", top="cut")
    hall.box(((B0 + B1) / 2, Y0 + 0.2, 0), (B1 - B0, 0.4, 0.6), "wall", top="cut")
    hall.box((B1 - 0.2, (Y0 + Y1) / 2, 0), (0.4, Y1 - Y0, 0.6), "wall", top="cut")
    hall.done()
    sub = Part(K, "substrate_cart")                                         # ABF substrates arriving
    sub.box((16.0, -4.2, 0.3), (1.4, 1.1, 0.12), "metal_dk")
    for sx in (-1, 1):
        for sy in (-1, 1):
            sub.box((16.0 + sx * 0.6, -4.2 + sy * 0.45, 0.3), (0.06, 0.06, 1.3), "metal")
    for k in range(14):
        sub.box((16.0, -4.2, 0.45 + k * 0.06), (1.1, 0.9, 0.04), "abf")
    K.tag(sub.done(), "substrate")
    tc = Part(K, "tc_bonder")                                              # HBM thermo-compression bonder
    tc.box((20.5, -2.5, 0.3), (3.2, 2.2, 1.1), "tool")
    for sx in (-1, 1):
        tc.box((20.5 + sx * 1.3, -2.5, 1.4), (0.25, 0.25, 1.9), "metal")
    tc.box((20.5, -2.5, 3.1), (2.85, 0.4, 0.3), "metal")
    tc.box((20.5, -2.5, 2.2), (0.6, 0.6, 0.9), "tool_dk")                     # bond head
    for k in range(8):
        tc.box((20.5, -2.5, 1.4 + k * 0.08), (0.5, 0.5, 0.07), "hbm")       # the stack being built
    for i in range(3):
        for j in range(2):
            tc.box((19.4 + i * 0.35, -3.3 + j * 0.35, 1.4), (0.3, 0.3, 0.05), "hbm")
    K.tag(tc.done(), "back_tools")
    pk = Part(K, "cowos_line")                                             # assembly + finished packages
    pk.box((26.0, -2.0, 0.3), (2.8, 2.4, 1.9), "tool")
    pk.box((26.0, -3.22, 0.9), (2.0, 0.04, 0.9), "glass")
    pk.box((26.0, -4.4, 0.3), (4.6, 1.0, 0.85), "metal")                      # conveyor
    for k in range(4):
        x = 24.3 + k * 1.1
        pk.box((x, -4.4, 1.15), (0.8, 0.8, 0.06), "abf")
        pk.box((x, -4.4, 1.21), (0.5, 0.42, 0.05), "silicon")
        pk.box((x - 0.08, -4.4, 1.26), (0.16, 0.3, 0.05), "die")
        pk.box((x + 0.08, -4.4, 1.26), (0.16, 0.3, 0.05), "die")
        for dy in (-0.17, 0.17):
            pk.box((x + 0.22, -4.4 + dy, 1.26), (0.06, 0.08, 0.07), "hbm")
            pk.box((x - 0.22, -4.4 + dy, 1.26), (0.06, 0.08, 0.07), "hbm")
    K.tag(pk.done(), "package")
    ft = Part(K, "final_test")                                             # handler + tester
    ft.box((28.6, 2.6, 0.3), (2.6, 2.0, 1.6), "tool")
    ft.box((28.6, 1.58, 0.8), (1.6, 0.04, 0.8), "glass")
    ft.box((30.3, 4.8, 0.3), (1.6, 1.4, 2.4), "tool_dk")
    ft.box((30.3, 4.08, 1.8), (0.9, 0.04, 0.45), "screen")
    K.tag(ft.done(), "test")
    so = Part(K, "test_sockets")
    so.box((28.6, -0.6, 0.3), (1.6, 1.2, 0.8), "metal")
    for i in range(3):
        for j in range(2):
            so.box((28.1 + i * 0.5, -0.9 + j * 0.55, 1.1), (0.4, 0.4, 0.12), "pcb")
            so.box((28.1 + i * 0.5, -0.9 + j * 0.55, 1.22), (0.22, 0.22, 0.06), "metal_bk")
    K.tag(so.done(), "socket")

    for tx, ty_, s in ((-26.5, 11.8, 1.2), (-23.5, 12.2, 1.0), (32.6, 11.6, 1.1), (29.8, 12.0, 0.9)):
        K.cyl((tx, ty_, 0), 0.15 * s, 1.0 * s, "trunk", 8)
        K.dome((tx, ty_, 0.8 * s), 1.1 * s, "tree", h=2.0 * s, verts=16)

    for zone in ("materials", "substrate", "litho", "equip", "metrology", "test", "socket", "tool_parts",
                 "fab_facility", "foundry", "package", "back_tools"):
        K.spot(zone)
    K.spot("foundry", at=(wx, wy - 0.08, FL + 2.3))                          # the wafer, not the walls
    return {"width": 1200, "margin": (0.04, 0.06)}


# ------------------------------------------------------------------------------ package, exploded
def package(K):
    K.catcher((0, 0, -0.01), 120)
    G = 9.0                                    # gap between exploded layers
    # ABF substrate with its cut corner: build-up layers, glass-cloth core (substrate)
    sub = Part(K, "abf")
    S, T = 60.0, 4.0
    sub.box((0, 0, 0), (S, S, T), "abf")
    for k in range(5):
        sub.box((S / 2 + 0.02, 0, 0.4 + k * 0.7), (0.04, S - 0.2, 0.18), "copper")
        sub.box((0, -S / 2 - 0.02, 0.4 + k * 0.7), (S - 0.2, 0.04, 0.18), "copper")
    sub.box((S / 2 + 0.03, 0, 1.55), (0.06, S - 0.4, 0.9), "glasscloth")
    sub.box((0, -S / 2 - 0.03, 1.55), (S - 0.4, 0.06, 0.9), "glasscloth")
    for i in range(10):                                                    # BGA balls underneath
        for j in range(10):
            sub.rod((-25 + i * 5.5, -25 + j * 5.5, -1.2), (-25 + i * 5.5, -25 + j * 5.5, 0.0), 1.1, "solder", 12)
    K.tag(sub.done(), "substrate")
    caps = Part(K, "caps")
    for i in range(14):
        for side in (-1, 1):
            caps.box((-22 + i * 3.4, side * 27.0, T), (1.6, 0.9, 0.9), "mlcc")
    K.tag(caps.done(), "mlcc")
    # C4 bumps, silicon interposer with TSVs and wiring, micro-bumps (package = CoWoS)
    z1 = T + G * 0.5
    pk = Part(K, "cowos")
    for i in range(12):
        for j in range(10):
            pk.rod((-20 + i * 3.6, -16 + j * 3.6, z1), (-20 + i * 3.6, -16 + j * 3.6, z1 + 1.0), 0.75, "solder", 10)
    z2 = z1 + G * 0.45
    pk.box((0, 0, z2), (46.0, 38.0, 1.6), "silicon")
    for k in range(9):
        pk.box((-18 + k * 4.5, 0, z2 + 1.6), (0.3, 34.0, 0.04), "copper")
    for i in range(10):
        pk.box((-20 + i * 4.4, -19.02, z2 + 0.2), (0.3, 0.04, 1.2), "copper")   # TSVs on the cut edge
    z3 = z2 + 1.6 + G * 0.35
    for i in range(30):
        for j in range(22):
            if abs(-14 + i * 1.0) < 15 and abs(-10.5 + j * 1.0) < 11:
                pk.box((-14.5 + i * 1.0, -10.5 + j * 1.0, z3), (0.45, 0.45, 0.45), "solder")
    K.tag(pk.done(), "package")
    # the chips on top: two GPU dies, eight HBM stacks, one stack pulled up to show its layers
    z4 = z3 + G * 0.7
    gp = Part(K, "gpu_dies")
    for dx in (-6.6, 6.6):
        gp.box((dx, 0, z4), (12.6, 15.0, 2.0), "die")
        for k in range(4):                                                 # a hint of the floorplan
            gp.box((dx, -5.4 + k * 3.6, z4 + 2.0), (10.6, 0.25, 0.04), "metal_dk")
        gp.box((dx - 2.6, 0, z4 + 2.0), (0.25, 13.0, 0.04), "metal_dk")
    K.tag(gp.done(), "gpu")
    hb = Part(K, "hbm_stacks")
    for i, dx in enumerate((-16.5, -5.5, 5.5, 16.5)):
        for dy in (-12.5, 12.5):
            if i == 3 and dy < 0:
                continue                                                   # this one is lifted out below
            for k in range(9):
                hb.box((dx, dy, z4 + k * 0.42), (9.0, 8.0, 0.36), "hbm" if k else "metal_dk")
    lx, ly, lz = 38.0, -26.0, z4 - 4.0                                     # pulled out to the side
    for k in range(13):                                                    # 12 DRAM dies + base die, apart
        hb.box((lx, ly, lz + k * 1.35), (9.0, 8.0, 0.4), "hbm" if k else "metal_dk")
    for i in range(4):
        for j in range(3):
            hb.rod((lx - 3 + i * 2, ly - 2.4 + j * 2.4, lz), (lx - 3 + i * 2, ly - 2.4 + j * 2.4, lz + 12 * 1.35 + 0.4), 0.16, "copper", 6)
    K.tag(hb.done(), "hbm")
    for dx, dy in ((-24, -21), (24, -21), (-24, 21), (24, 21)):            # dashed guides between layers
        for k in range(int((z4 - T) / 1.6)):
            K.no_lines(K.box((dx, dy, T + 0.4 + k * 1.6), (0.25, 0.25, 0.8), "metal_dk", name="guide"))
    for zone in ("gpu", "hbm", "package", "substrate", "mlcc"):
        K.spot(zone)
    return {"width": 1200, "margin": (0.1, 0.06)}


PANELS = {"fab": fab, "fab_pkg": package}
