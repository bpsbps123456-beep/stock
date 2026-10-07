"""AI 서버 — one GPU rack, then one compute tray opened (stops 6 AI 칩, 7 부품, 8 서버 조립).

rack: an NVL72-style rack with its right side and front off — power shelves at top and bottom,
compute trays around the NVLink switch trays, the copper spine, busbar and coolant manifolds at
the back, one compute tray pulled out. tray: that tray with its lid off — four GPUs (two dies
and eight HBM stacks each), two CPUs, power stages, MLCCs, NICs with optics at the front,
NVLink connectors at the back, cold plates — and a custom-ASIC board beside it.
"""

from __future__ import annotations

from kit import PALETTE, Part

PALETTE.setdefault("asic", ("#5d63c8", 0.25, 0.3, 0, 1))
PALETTE.setdefault("tile", ("#cfd2d4", 0.7, 0.05, 0, 1))
PALETTE.setdefault("tile_dk", ("#b4b8bc", 0.7, 0.05, 0, 1))

U = 0.5            # one rack unit in panel units (10 cm)
DT = 9.4           # full tray depth (the pulled-out one)
CUT = 0.8          # trays in the rack are cut here (half section) so the back of the rack shows
FRONT = -5.7       # tray fronts (y)


# ---------------------------------------------------------------------------- rack
def compute_tray(K, pulled=False, D=DT):
    """1U compute tray seen from outside; `pulled` shows its insides from above (lid off)."""
    p = Part(K, "ctray")
    W, Hh = 5.2, 0.44
    if not pulled:
        p.box((0, 0, 0), (W, D, Hh), "blade")
    else:
        p.box((0, 0, 0), (W, D, 0.06), "metal")
        for sx in (-1, 1):
            p.box((sx * (W / 2 - 0.03), 0, 0), (0.06, D, Hh), "metal", top="metal_dk")
        p.box((0, 0.2, 0.06), (W - 0.3, D - 1.2, 0.04), "pcb")
        for gx in (-1.2, 1.2):                       # four GPU cold plates, two CPU plates
            for gy in (0.2, 2.3):
                p.box((gx, gy, 0.1), (1.6, 1.6, 0.2), "copper")
            p.box((gx, -1.9, 0.1), (1.2, 1.2, 0.16), "copper")
            p.rod((gx - 0.4, -2.4, 0.3), (gx - 0.4, 4.4, 0.3), 0.07, "cold", 8)
            p.rod((gx + 0.4, -2.4, 0.34), (gx + 0.4, 4.4, 0.34), 0.07, "hot", 8)
        for k in range(4):
            p.box((-1.95 + k * 1.3, -3.6, 0.1), (0.7, 0.7, 0.16), "metal_bk")
    fy = -D / 2 - 0.01
    p.box((0, fy, 0.04), (W - 0.1, 0.04, Hh - 0.08), "metal_dk")
    for k in range(4):                               # OSFP cages on the front
        p.box((-1.95 + k * 1.3, fy - 0.02, 0.12), (0.62, 0.04, 0.22), "metal")
    p.box((2.35, fy - 0.03, 0.18), (0.1, 0.02, 0.1), "led_g")
    tray = p.done()
    q = Part(K, "ctray_optics")                      # transceivers + fibre tails (zone optics)
    for k in range(4):
        x = -1.95 + k * 1.3
        q.box((x, fy - 0.35, 0.13), (0.5, 0.7, 0.2), "metal")
        q.box((x, fy - 0.75, 0.18), (0.16, 0.12, 0.08), "fiber")
    return tray, q.done()


def switch_tray(K, D=DT):
    p = Part(K, "nvswitch")
    W, Hh = 5.2, 0.44
    p.box((0, 0, 0), (W, D, Hh), "metal_bk")
    fy = -D / 2 - 0.01
    p.box((0, fy, 0.05), (W - 0.2, 0.04, Hh - 0.1), "rack_fr")
    for k in range(10):
        p.box((-2.1 + k * 0.42, fy - 0.02, 0.15), (0.26, 0.03, 0.12), "metal_dk")
    p.box((2.3, fy - 0.03, 0.18), (0.1, 0.02, 0.1), "led_a")
    return p.done()


def power_shelf(K, D=DT):
    p = Part(K, "psu")
    W, Hh = 5.2, 0.44
    p.box((0, 0, 0), (W, D, Hh), "metal_dk")
    fy = -D / 2 - 0.01
    for k in range(6):
        x = -2.15 + k * 0.86
        p.box((x, fy, 0.04), (0.8, 0.04, Hh - 0.08), "white")
        p.rod((x - 0.12, fy - 0.02, 0.22), (x - 0.12, fy - 0.05, 0.22), 0.13, "metal_bk", 16)
        p.box((x + 0.27, fy - 0.03, 0.26), (0.08, 0.02, 0.08), "led_g")
    return p.done()


def rack(K):
    W, D, H = 6.0, 12.0, 18.6
    x0, y0 = -W / 2, -D / 2
    # raised floor around the rack
    for i in range(-3, 4):
        for j in range(-3, 4):
            K.ground(K.box((i * 3.0, j * 3.0, -0.3), (2.94, 2.94, 0.3), "tile" if (i + j) % 2 else "tile_dk", name="tile"))
    K.ground(K.box((0, -9.0, -0.02), (2.9, 2.9, 0.04), "metal", name="perf_tile"))
    frame = Part(K, "rack_frame")
    frame.box((0, 0, 0), (W, D, 0.6), "rack")
    frame.box((0, 0, H - 0.3), (W, D, 0.3), "rack")
    frame.box((x0 + 0.06, 0, 0.6), (0.12, D, H - 0.9), "rack_fr")          # left side panel
    frame.box((0, D / 2 - 0.06, 0.6), (W, 0.12, H - 0.9), "rack_fr")      # back panel
    for sx in (-1, 1):
        for sy in (-1, 1):
            frame.box((sx * (W / 2 - 0.18), sy * (D / 2 - 0.18), 0.6), (0.36, 0.36, H - 0.9), "rack")
    for z in (0.6, H - 0.4):
        frame.box((W / 2 - 0.18, 0, z), (0.36, D, 0.12), "rack")
    K.tag(frame.done(), "server")

    DC = CUT - FRONT                                   # depth of the sectioned trays
    ty = (FRONT + CUT) / 2
    tray, optics = compute_tray(K, D=DC)
    sw = switch_tray(K, D=DC)
    ps = power_shelf(K, D=DC)
    first = {"tray": True, "sw": True, "ps": True}

    def put(kind, z):
        nonlocal tray, optics, sw, ps
        if kind == "ps":
            ob = ps if first["ps"] else K.inst(ps, (0, ty, z))
            ob.location = (0, ty, z)
            K.tag(ob, "board")
        elif kind == "sw":
            ob = sw if first["sw"] else K.inst(sw, (0, ty, z))
            ob.location = (0, ty, z)
            K.tag(ob, "rack_link")
        else:
            if first["tray"]:
                tray.location = optics.location = (0, ty, z)
                K.tag(tray, "server")
                K.tag(optics, "optics")
            else:
                K.inst(tray, (0, ty, z), zone="server")
                K.inst(optics, (0, ty, z), zone="optics")
        first[kind] = False

    z = 0.6
    stack = ["ps"] * 3 + ["tray"] * 8 + ["sw"] * 9 + ["tray"] * 10 + ["ps"] * 3
    pulled_at = 3 + 8 + 9 + 6                          # one of the upper compute trays
    slot_z = None
    for k, kind in enumerate(stack):
        if k == pulled_at:
            slot_z = z
        else:
            put(kind, z)
        z += U
    # the pulled-out tray (lid off) slides forward on its rails
    pt, po = compute_tray(K, pulled=True)
    pt.location = po.location = (0, FRONT + DT / 2 - 6.2, slot_z)
    K.tag(pt, "server")
    K.tag(po, "optics")
    for sx in (-1, 1):
        K.box((sx * 2.55, FRONT - 1.0, slot_z - 0.05), (0.1, 8.0, 0.06), "metal", zone="server", name="rail")

    # back of the rack: NVLink copper spine, busbar, coolant manifolds
    zb, zt = 0.6 + 3 * U, 0.6 + 30 * U
    spine = Part(K, "spine")
    for cx in (-1.95, -0.65, 0.65, 1.95):
        spine.box((cx, 4.95, zb), (1.1, 0.8, zt - zb), "metal_bk")
        for k in range(7):
            spine.rod((cx - 0.42 + k * 0.14, 4.52, zb + 0.1), (cx - 0.42 + k * 0.14, 4.52, zt - 0.1), 0.05, "copper", 6)
    K.tag(spine.done(), "rack_link")
    bus = Part(K, "busbar")
    bus.box((0, 4.3, 0.6), (0.34, 0.18, H - 1.3), "copper")
    for k in range(6):
        z = 0.6 + (k if k < 3 else 27 + k) * U + 0.15
        bus.box((0, 4.15, z), (0.5, 0.2, 0.16), "copper")
    K.tag(bus.done(), "board")
    for x, y, mat in ((2.2, 3.7, "cold"), (2.65, 4.15, "hot")):      # supply and return manifolds
        K.tube([(x, y, 0.2), (x, y, H - 0.8)], 0.17, mat, "cooling")
        for k, kind in enumerate(stack):
            if kind == "tray" and k != pulled_at:
                zz = 0.6 + k * U + (0.16 if mat == "cold" else 0.3)
                K.tube([(x, y, zz), (x - 0.35, y - 0.4, zz), (x - 0.35, CUT + 0.03, zz)], 0.045, mat, "cooling")
        K.tube([(x, y, 0.25), (x, y, -0.2), (x + 1.2, y + 4.0, -0.2), (x + 2.5, y + 7.5, -0.2)], 0.17, mat, "cooling")
    # fibres from the compute-tray optics gather up the front-right post and leave over the top
    tail = FRONT - 0.8
    for k, kind in enumerate(stack):
        if kind == "tray" and k != pulled_at:
            zz = 0.6 + k * U + 0.25
            K.tube([(1.95, tail, zz), (2.8, tail - 0.25, zz), (2.85, tail - 0.3, H + 0.2)], 0.035, "fiber", "optics")
    K.tube([(2.85, tail - 0.3, H + 0.2), (2.85, tail - 0.3, H + 1.6), (2.85, 2.0, H + 1.6)], 0.22, "fiber", "optics")

    for zone in ("server", "rack_link", "board", "cooling", "optics"):
        K.spot(zone)
    K.spot("rack_link", at=(1.6, 4.5, 8.4))                                  # on the copper spine
    return {"width": 1200, "margin": (0.22, 0.05)}


# ---------------------------------------------------------------------------- tray
def gpu_package(K, cx, cy, plate=True):
    """Substrate, two GPU dies, eight HBM stacks; with a cold plate unless the chip is shown bare."""
    sub = Part(K, "gpu_sub")
    sub.box((cx, cy, 0.5), (11.5, 11.5, 0.35), "substrate")
    sub.box((cx, cy, 0.85), (10.4, 9.6, 0.12), "silicon")             # interposer
    gpu = K.tag(sub.done(), "gpu")
    dies = Part(K, "gpu_die")
    for dx in (-2.35, 2.35):
        dies.box((cx + dx, cy, 0.97), (4.4, 4.2, 0.3), "die")
    K.tag(dies.done(), "gpu")
    hbm = Part(K, "hbm")
    for dx in (-3.5, -1.2, 1.2, 3.5):
        for dy in (-3.35, 3.35):
            hbm.box((cx + dx, cy + dy, 0.97), (1.9, 1.9, 0.6), "hbm")
            for k in range(1, 4):
                hbm.box((cx + dx, cy + dy, 0.97 + k * 0.15), (1.94, 1.94, 0.02), "metal_bk")
    K.tag(hbm.done(), "hbm")
    vr = Part(K, "vrm")                                                  # power stages + inductors
    for side in (-1, 1):
        for k in range(6):
            y = cy - 5.0 + k * 2.0
            vr.box((cx + side * 7.4, y, 0.5), (1.0, 1.0, 0.3), "chip")
            vr.box((cx + side * 8.9, y, 0.5), (1.2, 1.3, 0.8), "metal")
    K.tag(vr.done(), "board")
    caps = Part(K, "caps")
    for k in range(12):
        for row in (0, 1):
            caps.box((cx - 5.0 + k * 0.9, cy - 7.0 - row * 0.8, 0.5), (0.55, 0.32, 0.32), "mlcc")
    K.tag(caps.done(), "mlcc")
    if plate:
        cp = Part(K, "coldplate")
        cp.box((cx, cy, 1.6), (11.0, 10.4, 1.3), "copper")
        cp.box((cx, cy, 2.9), (9.0, 8.4, 0.25), "metal")
        K.tag(cp.done(), "cooling")
        K.tube([(cx - 2.0, cy, 3.1), (cx - 2.0, cy, 4.0), (cx - 2.0, 79.0, 4.0)], 0.45, "cold", "cooling")
        K.tube([(cx + 2.0, cy, 3.1), (cx + 2.0, cy, 4.6), (cx + 2.0, 79.0, 4.6)], 0.45, "hot", "cooling")
    return gpu


def cpu_package(K, cx, cy, plate=True):
    sub = Part(K, "cpu_sub")
    sub.box((cx, cy, 0.5), (8.0, 8.0, 0.35), "substrate")
    sub.box((cx, cy, 0.85), (4.6, 4.6, 0.3), "cpu_die")
    K.tag(sub.done(), "cpu")
    mem = Part(K, "lpddr")
    for dx, dy in ((-5.6, -2.2), (-5.6, 2.2), (5.6, -2.2), (5.6, 2.2), (-2.2, 5.6), (2.2, 5.6), (-2.2, -5.6), (2.2, -5.6)):
        mem.box((cx + dx, cy + dy, 0.5), (2.4 if abs(dx) > 3 else 3.0, 3.0 if abs(dx) > 3 else 2.4, 0.3), "metal_bk")
    K.tag(mem.done(), "cpu")
    if plate:
        cp = Part(K, "cpu_plate")
        cp.box((cx, cy, 1.25), (7.6, 7.6, 1.1), "copper")
        K.tag(cp.done(), "cooling")
        K.tube([(cx - 1.5, cy, 2.35), (cx - 1.5, cy, 3.4), (cx - 4.2, cy, 3.4), (cx - 4.2, 79.0, 3.4)], 0.4, "cold", "cooling")
        K.tube([(cx + 1.5, cy, 2.35), (cx + 1.5, cy, 5.2), (cx + 4.2, cy, 5.2), (cx + 4.2, 79.6, 5.2)], 0.4, "hot", "cooling")


def tray(K):
    W, D = 44.0, 80.0
    K.catcher((W / 2 + 10, D / 2, -0.01), 70)
    ch = Part(K, "chassis")
    ch.box((W / 2, D / 2, 0), (W, D, 0.3), "metal")
    for x in (0.15, W - 0.15):
        ch.box((x, D / 2, 0.3), (0.3, D, 4.1), "metal", top="metal_dk")
    ch.box((W / 2, D - 0.15, 0.3), (W, 0.3, 4.1), "metal", top="metal_dk")
    ch.box((W / 2, 0.15, 0.3), (W, 0.3, 4.1), "metal_dk")               # front panel
    K.tag(ch.done(), "server")
    board = Part(K, "board")
    board.box((W / 2, 40.5, 0.3), (42.4, 70.0, 0.2), "pcb")
    for k in range(9):                                                   # copper traces to show it is a board
        board.box((4 + k * 4.5, 40.5, 0.5), (0.12, 66.0, 0.01), "copper")
    K.tag(board.done(), "pcb")
    gpu_package(K, 11.5, 44.0)
    gpu_package(K, 32.5, 44.0, plate=False)          # nearest GPU left bare
    gpu_package(K, 11.5, 62.0)
    gpu_package(K, 32.5, 62.0)
    cpu_package(K, 11.5, 25.0)
    cpu_package(K, 32.5, 25.0, plate=False)
    # NICs behind the front panel, OSFP cages in it, transceivers sticking out (optics)
    nic = Part(K, "nic")
    cage = Part(K, "cages")
    opt = Part(K, "osfp")
    for k, x in enumerate((7.0, 17.0, 27.0, 37.0)):
        nic.box((x, 11.0, 0.5), (4.4, 4.4, 0.35), "substrate")
        nic.box((x, 11.0, 0.85), (2.6, 2.6, 0.25), "chip")
        for f in range(7):
            nic.box((x - 1.8 + f * 0.6, 11.0, 1.1), (0.18, 3.6, 1.4), "metal")
        cage.box((x, 3.0, 0.5), (3.0, 6.0, 2.2), "metal")
        opt.box((x, -3.2, 0.7), (2.4, 7.0, 1.7), "metal")
        opt.box((x, -7.0, 1.0), (1.2, 1.0, 0.5), "fiber")
        for fx in (-0.4, 0.0, 0.4):
            opt.rod((x + fx, -7.4, 1.25), (x + fx * 3, -14.0, 1.25 - 0.4), 0.14, "fiber", 6)
    K.tag(nic.done(), "switch")
    K.tag(cage.done(), "optics")
    K.tag(opt.done(), "optics")
    # NVLink connectors and the busbar clip at the back
    lk = Part(K, "nvlink")
    for x in (10.0, 34.0):
        lk.box((x, 77.0, 0.5), (12.0, 3.2, 2.6), "metal_bk")
        for k in range(20):
            lk.box((x - 5.4 + k * 0.57, 78.65, 0.9), (0.22, 0.12, 1.8), "copper")
    K.tag(lk.done(), "rack_link")
    pw = Part(K, "power_in")
    pw.box((W / 2, 77.0, 0.5), (5.0, 3.0, 2.0), "copper")
    for k in range(3):
        pw.box((W / 2 - 1.6 + k * 1.6, 79.0, 0.8), (0.9, 1.6, 1.4), "copper")
    K.tag(pw.done(), "board")
    qd = Part(K, "quick_disconnect")
    for x, m in ((40.0, "cold"), (41.8, "hot")):
        qd.rod((x, 76.0, 4.0 if m == "cold" else 4.6), (x, 81.0, 4.0 if m == "cold" else 4.6), 0.6, m, 16)
    K.tag(qd.done(), "cooling")
    K.tube([(9.5, 79.0, 4.0), (40.0, 79.0, 4.0)], 0.45, "cold", "cooling")
    K.tube([(13.5, 79.6, 4.6), (41.8, 79.6, 4.6)], 0.45, "hot", "cooling")

    # a custom-accelerator board beside the tray: the GPU's alternative
    ax0 = W + 14.0
    ab = Part(K, "asic_board")
    ab.box((ax0 + 13, 50.0, 0), (26.0, 34.0, 0.25), "pcb")
    for cy in (41.0, 59.0):
        ab.box((ax0 + 13, cy, 0.25), (11.0, 11.0, 0.35), "substrate")
        ab.box((ax0 + 13, cy, 0.6), (6.0, 6.0, 0.35), "asic")
        for dx, dy in ((-4.2, -2.4), (-4.2, 2.4), (4.2, -2.4), (4.2, 2.4)):
            ab.box((ax0 + 13 + dx, cy + dy, 0.6), (1.8, 2.0, 0.6), "hbm")
    for k in range(14):
        ab.box((ax0 + 1.5 + k * 1.7, 34.0, 0.25), (0.5, 0.9, 0.35), "chip")
    K.tag(ab.done(), "asic")

    for zone in ("gpu", "hbm", "cpu", "asic", "pcb", "mlcc", "board", "cooling", "switch", "optics", "rack_link"):
        K.spot(zone)
    K.spot("board", at=(23.6, 42.0, 1.3))                                    # the power stages beside the bare GPU
    return {"width": 1200, "margin": (0.04, 0.06)}


PANELS = {"rack": rack, "tray": tray}
