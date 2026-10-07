"""AI 데이터센터 — campus cutaway (stops 9 네트워크, 10 저장, 11 데이터센터).

Power comes in at the front-left (grid → substation → electrical room, backup generators in
front), runs along the busway over two rows of GPU racks, coolant comes from the dry-cooler yard
on the right through CDUs, and data leaves through the network room and the fibre conduit at the
front-right. Phase 2 is being built at the back-right on land the developer secured.
Units are metres, exaggerated where a thing would otherwise vanish (racks, transceivers).
"""

from __future__ import annotations

import math

from kit import PALETTE, Part

PALETTE.setdefault("platter", ("#cfd4dc", 0.12, 0.9, 0, 1))

FLOOR = 0.15
H = 5.4


# ---------------------------------------------------------------- equipment parts
def gpu_rack(K):
    p = Part(K, "gpu_rack")
    w, d, h = 1.1, 1.3, 2.6
    p.box((0, 0, 0), (w, d, h), "rack")
    fy = -d / 2 - 0.01
    for k in range(9):                                   # tray bands on the front
        z = 0.18 + k * 0.25
        mat = "blade" if k not in (4,) else "metal_bk"
        p.box((0, fy, z), (w - 0.16, 0.03, 0.19), mat)
        p.box((-w / 2 + 0.17, fy - 0.02, z + 0.07), (0.05, 0.02, 0.05), "led_a" if k == 4 else "led_g")
        if k % 2:
            p.box((w / 2 - 0.2, fy - 0.02, z + 0.07), (0.05, 0.02, 0.05), "led_b")
    p.box((0, 0, h), (w - 0.1, d - 0.1, 0.05), "metal_bk")
    return p.done()


def switch_rack(K):
    """Network rack: switch chassis full of ports, with transceivers (separate object, zone optics)."""
    p = Part(K, "switch_rack")
    w, d, h = 1.1, 1.3, 2.6
    p.box((0, 0, 0), (w, d, h), "rack")
    fy = -d / 2 - 0.01
    for k in range(5):
        z = 0.25 + k * 0.45
        p.box((0, fy, z), (w - 0.12, 0.04, 0.32), "metal_dk")
        for i in range(8):
            p.box((-0.4 + i * 0.115, fy - 0.025, z + 0.21), (0.05, 0.02, 0.04), "led_b" if (i + k) % 3 else "led_g")
    sw = p.done()
    q = Part(K, "optics")
    for k in range(5):
        z = 0.25 + k * 0.45
        for i in range(8):
            q.box((-0.4 + i * 0.115, fy - 0.06, z + 0.05), (0.07, 0.12, 0.09), "metal")
            q.box((-0.4 + i * 0.115, fy - 0.13, z + 0.07), (0.03, 0.04, 0.03), "fiber")
    return sw, q.done()


def flash_rack(K):
    p = Part(K, "flash_rack")
    w, d, h = 1.1, 1.3, 2.6
    p.box((0, 0, 0), (w, d, h), "rack")
    fy = -d / 2 - 0.01
    for k in range(7):
        z = 0.2 + k * 0.33
        p.box((0, fy, z), (w - 0.12, 0.04, 0.26), "white")
        for i in range(12):
            p.box((-0.42 + i * 0.077, fy - 0.025, z + 0.04), (0.04, 0.02, 0.18), "metal_dk")
        p.box((0.44, fy - 0.03, z + 0.1), (0.04, 0.02, 0.04), "led_b")
    return p.done()


def disk_rack(K):
    p = Part(K, "disk_rack")
    w, d, h = 1.1, 1.3, 2.6
    p.box((0, 0, 0), (w, d, h), "rack")
    fy = -d / 2 - 0.01
    for k in range(4):
        z = 0.2 + k * 0.6
        if k == 2:                                       # one drawer pulled out: drives from above
            p.box((0, fy - 0.45, z), (w - 0.12, 0.9, 0.42), "metal_dk")
            for i in range(5):
                for j in range(4):
                    p.box((-0.38 + i * 0.19, fy - 0.8 + j * 0.2, z + 0.42), (0.15, 0.17, 0.02), "metal")
            continue
        p.box((0, fy, z), (w - 0.12, 0.04, 0.5), "metal_dk")
        for i in range(4):
            for j in range(3):
                p.box((-0.36 + i * 0.24, fy - 0.025, z + 0.05 + j * 0.15), (0.2, 0.02, 0.11), "metal")
        p.box((0.44, fy - 0.03, z + 0.4), (0.04, 0.02, 0.04), "led_g")
    return p.done()


def controller_rack(K):
    p = Part(K, "ctrl_rack")
    w, d, h = 1.1, 1.3, 2.6
    p.box((0, 0, 0), (w, d, h), "rack")
    fy = -d / 2 - 0.01
    for k in range(5):
        z = 0.25 + k * 0.45
        p.box((0, fy, z), (w - 0.12, 0.05, 0.3), "white")
        p.box((0, fy - 0.03, z + 0.12), (w - 0.3, 0.02, 0.06), "red")
        p.box((-0.4, fy - 0.03, z + 0.22), (0.04, 0.02, 0.04), "led_g")
    p.box((0, fy - 0.02, 2.45), (0.5, 0.02, 0.1), "screen")
    return p.done()


def dci_rack(K):
    p = Part(K, "dci_rack")
    w, d, h = 1.1, 1.3, 2.6
    p.box((0, 0, 0), (w, d, h), "rack")
    fy = -d / 2 - 0.01
    for k in range(4):
        z = 0.3 + k * 0.55
        p.box((0, fy, z), (w - 0.12, 0.05, 0.42), "metal_dk")
        for i in range(10):
            p.box((-0.42 + i * 0.093, fy - 0.03, z + 0.08), (0.03, 0.02, 0.26), "fiber" if i % 2 else "led_b")
    return p.done()


def cdu(K):
    p = Part(K, "cdu")
    p.box((0, 0, 0), (0.9, 1.2, 2.2), "white")
    p.box((0, -0.61, 1.5), (0.5, 0.02, 0.3), "screen")
    p.box((0, -0.61, 0.3), (0.6, 0.02, 0.8), "metal")
    return p.done()


def switchgear(K, n=5):
    p = Part(K, "switchgear")
    for i in range(n):
        x = (i - (n - 1) / 2) * 1.0
        p.box((x, 0, 0), (0.96, 1.0, 2.3), "metal")
        p.box((x, -0.51, 0.15), (0.8, 0.02, 1.0), "white")
        p.box((x, -0.51, 1.25), (0.8, 0.02, 0.9), "white")
        p.box((x + 0.3, -0.53, 1.0), (0.04, 0.03, 0.3), "metal_dk")
        p.box((x - 0.2, -0.53, 1.85), (0.08, 0.02, 0.08), "led_g" if i % 2 else "led_a")
    p.box((0, 0, 2.3), (n * 1.0, 0.6, 0.12), "metal_dk")      # cable tray on top
    return p.done()


def ups(K, n=3):
    p = Part(K, "ups")
    for i in range(n):
        x = (i - (n - 1) / 2) * 0.95
        p.box((x, 0, 0), (0.9, 0.9, 2.1), "white")
        p.box((x, -0.46, 1.4), (0.36, 0.02, 0.24), "screen")
        for k in range(5):
            p.box((x, -0.46, 0.2 + k * 0.18), (0.7, 0.02, 0.06), "metal_dk")
    return p.done()


def battery(K, n=4):
    p = Part(K, "battery")
    for i in range(n):
        x = (i - (n - 1) / 2) * 0.85
        p.box((x, 0, 0), (0.8, 0.8, 2.0), "metal_bk")
        for k in range(6):
            p.box((x, -0.41, 0.2 + k * 0.3), (0.62, 0.02, 0.2), "metal_dk")
            p.box((x + 0.25, -0.42, 0.33 + k * 0.3), (0.04, 0.02, 0.04), "led_g")
    return p.done()


def dry_cooler(K):
    p = Part(K, "dry_cooler")
    L, W, Hh = 5.2, 2.4, 1.9
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((sx * (W / 2 - 0.15), sy * (L / 2 - 0.2), 0), (0.12, 0.12, 0.7), "metal_dk")
    p.box((0, 0, 0.7), (W, L, Hh - 0.7), "white")
    for sx in (-1, 1):                                     # coil faces on the long sides
        p.box((sx * (W / 2 + 0.005), 0, 0.85), (0.02, L - 0.3, Hh - 1.0), "metal_dk")
        for k in range(12):
            p.box((sx * (W / 2 + 0.02), -L / 2 + 0.35 + k * 0.41, 0.85), (0.02, 0.05, Hh - 1.0), "metal")
    ob = p.done()
    fans = []
    for k in (-1, 0, 1):
        fans += K.fan((0, k * 1.65, Hh), 0.72)
    fan = K.join(fans, "fans")
    return ob, fan


def genset(K):
    p = Part(K, "genset")
    W, L, Hh = 2.3, 5.0, 2.5
    p.box((0, 0, 0), (W, L, 0.25), "metal_dk")
    p.box((0, 0, 0.25), (W, L, Hh - 0.25), "genset")
    for k in range(7):                                     # louvers on the front long side
        p.box((-W / 2 - 0.01, -1.6 + k * 0.5, 0.7), (0.02, 0.32, 1.2), "metal_dk")
    p.box((0, L / 2 + 0.01, 0.6), (W - 0.5, 0.02, 1.5), "metal_dk")   # radiator end
    p.rod((0.4, -1.2, Hh), (0.4, -1.2, Hh + 1.3), 0.17, "metal_dk")
    p.rod((0.4, -1.2, Hh + 1.3), (0.4, -1.2, Hh + 1.42), 0.26, "metal_dk")
    return p.done()


def transformer(K):
    p = Part(K, "transformer")
    W, L, Hh = 2.2, 3.0, 2.3
    p.box((0, 0, 0), (W + 0.4, L + 0.4, 0.3), "concrete")
    p.box((0, 0, 0.3), (W, L, Hh), "transformer")
    for sx in (-1, 1):                                       # radiator banks
        for k in range(9):
            p.box((sx * (W / 2 + 0.38), -1.2 + k * 0.3, 0.55), (0.7, 0.07, 1.75), "transformer")
        p.box((sx * (W / 2 + 0.06), 0, 0.5), (0.12, 2.8, 0.12), "metal_dk")
        p.box((sx * (W / 2 + 0.06), 0, 2.15), (0.12, 2.8, 0.12), "metal_dk")
    p.rod((-0.9, 1.1, Hh + 0.75), (0.9, 1.1, Hh + 0.75), 0.36, "transformer", 20)   # conservator
    p.box((0, 1.1, Hh + 0.3), (0.2, 0.2, 0.2), "metal_dk")
    for i in range(3):                                       # high-voltage bushings
        x = -0.65 + i * 0.65
        p.rod((x, -0.6, Hh + 0.3), (x, -0.6, Hh + 1.9), 0.12, "porcelain", 12, r2=0.07)
        for k in range(5):
            p.rod((x, -0.6, Hh + 0.55 + k * 0.27), (x, -0.6, Hh + 0.62 + k * 0.27), 0.17, "porcelain", 12)
        p.rod((x, -0.6, Hh + 1.9), (x, -0.6, Hh + 2.0), 0.08, "metal", 8)
    for i in range(3):
        x = -0.5 + i * 0.5
        p.rod((x, 0.5, Hh + 0.3), (x, 0.5, Hh + 0.8), 0.07, "porcelain", 10)
    return p.done()


def lattice_tower(K, x, y, h=12.5, base=2.4, top=0.7, zone=None):
    segs = []
    b, t = base / 2, top / 2
    def at(z):
        f = z / h
        return b + (t - b) * f
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for cx, cy in corners:
        segs.append(((x + cx * b, y + cy * b, 0), (x + cx * t, y + cy * t, h)))
    z = 0
    step = 1.6
    while z < h - 0.1:
        z2 = min(h, z + step)
        a1, a2 = at(z), at(z2)
        for (cx, cy), (dx, dy) in zip(corners, corners[1:] + corners[:1]):
            segs.append(((x + cx * a1, y + cy * a1, z), (x + dx * a2, y + dy * a2, z2)))
            segs.append(((x + dx * a1, y + dy * a1, z), (x + cx * a2, y + cy * a2, z2)))
        z = z2
    arms = []
    for za, span in ((h - 1.2, 2.6), (h - 3.4, 3.2)):
        a = at(za)
        segs.append(((x - span, y, za), (x + span, y, za)))
        segs.append(((x - span, y, za), (x - a, y, za + 0.8)))
        segs.append(((x + span, y, za), (x + a, y, za + 0.8)))
        arms += [(x - span, y, za), (x + span, y, za)]
    ob = K.struts(segs, 0.045, "metal_dk", zone, "tower")
    return ob, arms


def tree(K, x, y, s=1.0):
    K.cyl((x, y, 0), 0.12 * s, 1.0 * s, "trunk", 8)
    K.dome((x, y, 0.8 * s), 1.0 * s, "tree", h=1.9 * s, verts=16)


# ---------------------------------------------------------------- the campus
def campus(K):
    K.slab(-30, 30, -15, 15, top="grass")
    K.ground(K.box((-6.5, 0.5, 0), (49, 25, 0.02), "gravel", name="site"))          # graded campus pad
    K.ground(K.box((-4.5, -10.5, 0), (3.2, 9, 0.04), "asphalt", name="road"))

    # ---- building shell (back and left walls full height, front/right cut low)
    x0, x1, y0, y1 = -20, 6, -6, 10
    K.box(((x0 + x1) / 2, (y0 + y1) / 2, 0), (x1 - x0, y1 - y0, FLOOR), "floor", name="slab")
    K.box(((x0 + x1) / 2, y1 - 0.15, FLOOR), (x1 - x0, 0.3, H), "wall", top="cut", name="wall_back")
    K.box((x0 + 0.15, (y0 + y1) / 2, FLOOR), (0.3, y1 - y0 - 0.3, H), "wall", top="cut", name="wall_left")
    K.box(((x0 + x1) / 2, y0 + 0.15, FLOOR), (x1 - x0, 0.3, 0.8), "wall", top="cut", name="wall_front")
    K.box((x1 - 0.15, (y0 + y1) / 2, FLOOR), (0.3, y1 - y0 - 0.6, 0.8), "wall", top="cut", name="wall_right")
    for xp in (-14, 0):
        K.box((xp, (y0 + y1) / 2, FLOOR), (0.2, y1 - y0 - 0.6, 0.8), "wall", top="cut", name="partition")
    for xc in range(-17, 6, 6):                                            # columns on the back wall
        K.box((xc, y1 - 0.45, FLOOR), (0.35, 0.3, H), "concrete", top="cut", name="column")

    # ---- electrical room (power_dist)
    sg = switchgear(K, 5)
    sg.location = (-17, 8.85, FLOOR)
    K.tag(sg, "power_dist")
    u = ups(K, 3)
    u.location = (-17, 4.2, FLOOR)
    K.tag(u, "power_dist")
    bt = battery(K, 4)
    bt.location = (-17, 1.0, FLOOR)
    K.tag(bt, "power_dist")

    # ---- data hall: two rows of GPU racks + CDUs (server, cooling)
    rack = gpu_rack(K)
    rows = (-1.6, 4.6)
    first = True
    for ry in rows:
        for i in range(10):
            x = -13.2 + i * 1.2
            if first:
                rack.location = (x, ry, FLOOR)
                K.tag(rack, "server")
                first = False
            else:
                K.inst(rack, (x, ry, FLOOR), zone="server")
    c = cdu(K)
    for k, ry in enumerate(rows):
        ob = c if k == 0 else K.inst(c, (-0.9, ry, FLOOR))
        ob.location = (-0.9, ry, FLOOR)
        K.tag(ob, "cooling")

    # busway (power) from the switchgear over each row, with tap-offs
    bz = 3.35
    K.tube([(-17.6, 8.6, 2.42), (-17.6, 8.6, bz), (-14.6, 8.6, bz), (-14.6, -1.6, bz)], 0.13, "power", "power_dist")
    for ry in rows:
        K.box((-8, ry, bz - 0.1), (13.4, 0.26, 0.2), "power", zone="power_dist", name="busway")
        for i in range(10):
            x = -13.2 + i * 1.2
            K.box((x, ry, bz - 0.32), (0.2, 0.16, 0.22), "power", zone="power_dist", name="tap")
            K.tube([(x, ry, bz - 0.32), (x, ry, 2.65 + FLOOR)], 0.03, "metal_bk")
    # cables from outside (substation and generators) into the electrical room
    K.tube([(-22.6, -7.6, 0.08), (-19.4, -7.6, 0.08), (-19.4, 7.9, 0.2)], 0.14, "power", "grid")
    K.tube([(-13.6, -8.9, 0.08), (-13.6, -6.6, 0.08), (-18.8, -5.2, 0.2), (-18.8, 7.9, 0.2)], 0.1, "power", "backup")

    # coolant: dry coolers -> CDUs (primary), CDUs -> racks (secondary)
    pz = 4.55
    sup = [(12.0, 4.0, 1.2), (12.0, 4.0, pz), (6.3, 4.0, pz), (-0.6, 4.0, pz), (-0.6, 4.6, pz), (-0.6, 4.6, 2.3)]
    ret = [(12.6, 3.4, 1.2), (12.6, 3.4, pz + 0.35), (6.3, 3.4, pz + 0.35), (-1.2, 3.4, pz + 0.35), (-1.2, 4.6, pz + 0.35), (-1.2, 4.6, 2.3)]
    K.tube(sup, 0.13, "cold", "cooling")
    K.tube(ret, 0.13, "hot", "cooling")
    K.tube([(-0.6, 4.0, pz), (-0.6, -1.6, pz), (-0.6, -1.6, 2.3)], 0.11, "cold", "cooling")
    K.tube([(-1.2, 3.4, pz + 0.35), (-1.2, -1.6, pz + 0.35), (-1.2, -1.6, 2.3)], 0.11, "hot", "cooling")
    for ry in rows:
        sz, rz = 3.0, 3.12
        K.tube([(-0.9, ry - 0.25, 2.35), (-0.9, ry - 0.25, sz), (-13.4, ry - 0.25, sz)], 0.05, "cold", "cooling")
        K.tube([(-0.9, ry + 0.25, 2.35), (-0.9, ry + 0.25, rz), (-13.4, ry + 0.25, rz)], 0.05, "hot", "cooling")

    # ---- network room: switches with optics, DCI shelf, storage row
    sw, op = switch_rack(K)
    for k, x in enumerate((1.3, 2.5, 3.7)):
        if k == 0:
            sw.location = (x, -1.6, FLOOR)
            op.location = (x, -1.6, FLOOR)
            K.tag(sw, "switch")
            K.tag(op, "optics")
        else:
            K.inst(sw, (x, -1.6, FLOOR), zone="switch")
            K.inst(op, (x, -1.6, FLOOR), zone="optics")
    dci = dci_rack(K)
    dci.location = (4.9, -1.6, FLOOR)
    K.tag(dci, "dci")
    fl = flash_rack(K)
    fl.location = (1.3, 4.6, FLOOR)
    K.tag(fl, "flash")
    K.inst(fl, (2.5, 4.6, FLOOR), zone="flash")
    dk = disk_rack(K)
    dk.location = (3.7, 4.6, FLOOR)
    K.tag(dk, "disk")
    ct = controller_rack(K)
    ct.location = (4.9, 4.6, FLOOR)
    K.tag(ct, "systems")

    # fibre: yellow overhead trays with cyan bundles from every row to the switches
    fz = 4.0
    for ry in rows:
        K.box((-6.4, ry, fz), (14.2, 0.5, 0.05), "metal", zone="fiber", name="tray")
        for sy in (-1, 1):
            K.box((-6.4, ry + sy * 0.25, fz), (14.2, 0.04, 0.2), "metal", zone="fiber", name="tray_lip")
        K.tube([(-13.3, ry, fz + 0.1), (0.4, ry, fz + 0.1)], 0.07, "fiber", "fiber")
        for i in range(10):
            x = -13.2 + i * 1.2 + 0.35
            K.tube([(x, ry, fz), (x, ry, 2.62 + FLOOR)], 0.025, "fiber", "fiber")
    K.box((0.9, 1.5, fz), (0.5, 6.7, 0.05), "metal", zone="fiber", name="tray_x")
    K.box((3.1, -1.6, fz), (4.4, 0.5, 0.05), "metal", zone="fiber", name="tray_net")
    K.tube([(0.9, -1.6, fz + 0.1), (0.9, 4.6, fz + 0.1)], 0.07, "fiber", "fiber")
    K.tube([(0.9, -1.6, fz + 0.1), (4.9, -1.6, fz + 0.1)], 0.07, "fiber", "fiber")
    for x in (1.3, 2.5, 3.7, 4.9):
        K.tube([(x, -1.85, fz), (x, -1.85, 2.62 + FLOOR)], 0.06, "fiber", "fiber")
    # out of the building to other data centres (dci)
    K.tube([(4.9, -1.0, 2.7), (4.9, -1.0, 3.4), (5.6, -1.0, 3.4), (5.6, -1.0, 0.25), (6.6, -1.0, 0.05),
            (30.2, -1.0, 0.05)], 0.12, "fiber", "dci")
    K.tube([(30.0, -1.0, 0.05), (30.2, -1.0, -0.8)], 0.12, "fiber", "dci")
    for x in (10, 16, 22, 28):                                             # buried-fibre marker posts
        K.cyl((x, -1.6, 0), 0.07, 0.9, "fiber", 8, zone="dci")
        K.cyl((x, -1.6, 0.9), 0.1, 0.12, "red", 8, zone="dci")

    # ---- substation and transmission line (grid)
    K.ground(K.box((-25.5, -9.5, 0), (8.5, 9.5, 0.05), "gravel", name="pad"))
    tf = transformer(K)
    tf.location = (-27.0, -10.8, 0.05)
    K.tag(tf, "grid")
    K.inst(tf, (-23.5, -10.8, 0.05), zone="grid")
    gantry = []
    for gx in (-28.6, -21.9):
        gantry.append(((gx, -6.4, 0), (gx, -6.4, 6.4)))
    gantry.append(((-28.8, -6.4, 6.4), (-21.7, -6.4, 6.4)))
    K.struts(gantry, 0.12, "metal_dk", "grid", "gantry")
    tw, arms = lattice_tower(K, -27.2, 2.5, zone="grid")
    for k, (ax, ay, az) in enumerate(arms[:2]):
        gx = -27.5 + k * 3.0
        K.wire((ax, ay, az), (gx, -6.4, 6.35), 0.6, 0.035, "metal_bk", "grid")
        K.wire((-30.0, ay + 0.8, az + 1.0), (ax, ay, az), 0.4, 0.035, "metal_bk", "grid")
        for tx in (-27.0, -23.5):
            K.wire((gx, -6.4, 6.3), (tx - 0.65 + k * 0.65, -11.4, 4.2), 0.3, 0.03, "metal_bk", "grid")
    fence = []
    fx0, fx1, fy0, fy1 = -29.6, -21.4, -14.2, -5.0
    for a, b in (((fx0, fy0), (fx1, fy0)), ((fx1, fy0), (fx1, fy1)), ((fx0, fy0), (fx0, fy1)), ((fx0, fy1), (fx1, fy1))):
        n = max(2, int(math.dist(a, b) / 1.6))
        for i in range(n + 1):
            t = i / n
            px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            fence.append(((px, py, 0), (px, py, 1.6)))
        fence.append(((a[0], a[1], 1.55), (b[0], b[1], 1.55)))
        fence.append(((a[0], a[1], 0.8), (b[0], b[1], 0.8)))
    K.struts(fence, 0.035, "metal", None, "fence")

    # ---- backup generators and their fuel (backup)
    gs = genset(K)
    for k, x in enumerate((-17.2, -13.9, -10.6)):
        ob = gs if k == 0 else K.inst(gs, (x, -11.8, 0.02))
        ob.location = (x, -11.8, 0.02)
        K.tag(ob, "backup")
    tank = K.rod((-7.6, -13.6, 0.75), (-7.6, -10.2, 0.75), 0.62, "genset", 24, "backup", "tank")
    for yy in (-13.0, -10.8):
        K.box((-7.6, yy, 0), (1.0, 0.25, 0.4), "metal_dk", zone="backup", name="saddle")

    # ---- dry-cooler yard (cooling)
    dc_body, dc_fans = dry_cooler(K)
    spots = [(9.6, 8.6), (12.6, 8.6), (15.6, 8.6)]
    for k, (x, y) in enumerate(spots):
        if k == 0:
            dc_body.location = (x, y, 0)
            dc_fans.location = (x, y, 0)
            K.tag([dc_body, dc_fans], "cooling")
        else:
            K.inst(dc_body, (x, y, 0), zone="cooling")
            K.inst(dc_fans, (x, y, 0), zone="cooling")
    K.tube([(9.6, 5.7, 1.0), (15.6, 5.7, 1.0)], 0.16, "cold", "cooling")
    K.tube([(9.6, 5.2, 1.4), (15.6, 5.2, 1.4)], 0.16, "hot", "cooling")
    K.tube([(12.0, 5.7, 1.0), (12.0, 4.0, 1.0), (12.0, 4.0, 1.2)], 0.13, "cold", "cooling")
    K.tube([(12.6, 5.2, 1.4), (12.6, 3.4, 1.4), (12.6, 3.4, 1.2)], 0.13, "hot", "cooling")
    K.box((9.0, 4.2, 0), (1.6, 1.4, 1.1), "metal_dk", zone="cooling", name="pumps")

    # ---- phase 2 under construction (build)
    segs = []
    gx = [19.5, 22.8, 26.1, 29.4]
    gy = [0.0, 4.2, 8.4, 12.6]
    for x in gx:
        for y in gy:
            segs.append(((x, y, 0), (x, y, 5.6)))
    for z in (2.8, 5.6):
        for x in gx:
            segs.append(((x, gy[0], z), (x, gy[-1], z)))
        for y in gy:
            segs.append(((gx[0], y, z), (gx[-1], y, z)))
    for x in gx[:2]:                                                       # bracing
        segs.append(((x, gy[0], 0), (x, gy[1], 2.8)))
    K.struts(segs, 0.13, "steel", "build", "frame")
    K.box((22.8 - 1.65 + 1.65, 6.3, 5.6), (6.6, 12.6, 0.12), "concrete", zone="build", name="deck")
    K.ground(K.box((24.45, 6.3, 0), (10.6, 13.4, 0.12), "concrete", zone="build", name="footing"))
    # tower crane
    cx, cy = 30.4, 10.5
    crane = []
    for dx in (-0.45, 0.45):
        for dy in (-0.45, 0.45):
            crane.append(((cx + dx, cy + dy, 0), (cx + dx, cy + dy, 13.5)))
    for z in range(0, 13, 1):
        for (ax, ay), (bx, by) in ((((-0.45, -0.45), (0.45, -0.45))), (((0.45, -0.45), (0.45, 0.45))),
                                   (((0.45, 0.45), (-0.45, 0.45))), (((-0.45, 0.45), (-0.45, -0.45)))):
            crane.append(((cx + ax, cy + ay, z), (cx + bx, cy + by, z + 1)))
    jz = 13.5
    crane += [((cx - 11, cy + 0.0, jz), (cx + 4, cy, jz)), ((cx - 11, cy, jz), (cx, cy, jz + 2.2)),
              ((cx + 4, cy, jz), (cx, cy, jz + 2.2)), ((cx, cy - 0.45, jz), (cx, cy, jz + 2.2))]
    for k in range(11):
        crane.append(((cx - k, cy, jz), (cx - k - 1, cy, jz - 0.0)))
        crane.append(((cx - k, cy, jz), (cx - k - 0.5, cy, jz + 0.7)))
    crane.append(((cx - 11, cy, jz + 0.7), (cx, cy, jz + 0.7)))
    K.struts(crane, 0.07, "crane", "build", "crane")
    K.box((cx + 3.2, cy, jz - 0.9), (1.4, 1.2, 0.9), "metal_dk", zone="build", name="counterweight")
    K.box((cx + 0.9, cy - 0.9, jz - 1.3), (1.0, 0.9, 1.1), "crane", zone="build", name="cab")
    K.tube([(cx - 7, cy, jz), (cx - 7, cy, 7.6)], 0.025, "metal_bk", "build")
    K.box((cx - 7, cy, 7.2), (0.25, 3.0, 0.35), "steel", zone="build", name="load")
    for k in range(4):                                                     # stacked beams
        K.box((24.0, -3.0 + k * 0.32, 0), (5.0, 0.26, 0.3), "steel", zone="build", name="beams")

    # ---- the plot (land): staked, signed, fenced
    K.ground(K.box((19.0, -9.8, 0), (21.0, 9.0, 0.05), "soil1", zone="land", name="plot"))
    stakes = [(8.8, -5.6), (29.2, -5.6), (29.2, -14.2), (8.8, -14.2), (19.0, -5.6), (19.0, -14.2)]
    for x, y in stakes:
        K.cyl((x, y, 0), 0.06, 1.1, "trunk", 6, zone="land")
        K.box((x + 0.18, y, 0.85), (0.32, 0.03, 0.22), "red", zone="land", name="flag")
    for x0_, x1_ in ((9.2, 18.6), (19.4, 28.8)):                            # painted outline of the next hall
        K.ground(K.box(((x0_ + x1_) / 2, -6.2, 0.05), (x1_ - x0_, 0.12, 0.01), "white", zone="land", name="line"))
        K.ground(K.box(((x0_ + x1_) / 2, -13.6, 0.05), (x1_ - x0_, 0.12, 0.01), "white", zone="land", name="line"))
    K.box((13.5, -12.0, 0), (0.12, 0.12, 2.0), "metal_dk", zone="land", name="signpost")
    K.box((15.1, -12.0, 0), (0.12, 0.12, 2.0), "metal_dk", zone="land", name="signpost")
    K.box((14.3, -12.0, 1.3), (2.4, 0.08, 1.1), "white", zone="land", name="sign")
    K.box((14.3, -12.05, 1.45), (2.0, 0.02, 0.2), "power", zone="land", name="sign_stripe")

    for x, y, s in ((-29, 13.5, 1.2), (-26, 13.8, 1.0), (-22.5, 13.5, 1.3), (-2, 13.6, 1.1), (3, 13.4, 0.9),
                    (-29, 9, 1.0), (29, 14, 1.1)):
        tree(K, x, y, s)

    for z in ("land", "build", "grid", "backup", "power_dist", "server", "cooling", "fiber",
              "switch", "optics", "flash", "disk", "systems", "dci"):
        K.spot(z)
    return {"width": 1200, "margin": (0.06, 0.07)}


# ---------------------------------------------------------------- inside one transceiver
def optics(K):
    """An 800G-class pluggable module half out of a switch port, lid lifted: electrical edge →
    DSP → laser drivers → lasers and lenses → fibre array → MPO cable. The switch chip sits
    behind the faceplate."""
    K.catcher((14, 0, -6.01), 80)
    # switch side: faceplate with cages, board and the switch ASIC behind it (switch)
    fp = Part(K, "faceplate")
    fp.box((-0.6, 0, -6), (1.2, 19, 15), "metal_dk")
    for y in (-5.5, 5.5):
        for z in (-1.2, 5.0):
            if (y, z) != (-5.5, -1.2):
                fp.box((0.05, y, z), (0.2, 8.0, 5.0), "metal_bk")             # empty cage openings
    K.tag(fp.done(), "switch")
    sb = Part(K, "switch_board")
    sb.box((-13, 0, -6), (24, 19, 0.6), "pcb")
    sb.box((-13.5, 0, -5.4), (11, 11, 0.6), "substrate")
    sb.box((-13.5, 0, -4.8), (7, 7, 0.5), "chip")
    for k in range(10):                                                  # heatsink fins over the chip
        sb.box((-18.0 + k * 1.0, 0, -4.3), (0.35, 10, 5.5), "metal")
    for k in range(12):
        sb.box((-5.5, -6 + k * 1.0, -5.4), (1.4, 0.5, 0.3), "chip")      # retimers on the lanes
    K.tag(sb.done(), "switch")

    # the module (optics): shell with its near wall cut low, board, gold fingers
    L = 40.0
    yc = -5.5                                                            # the module sits in the lower-left cage
    shell = Part(K, "module")
    shell.box((L / 2 - 8, yc, -3.7), (L + 8, 10.4, 0.5), "metal")
    shell.box((L / 2 - 8, yc + 5.0, -3.7), (L + 8, 0.4, 4.8), "metal", top="metal_dk")       # far wall
    shell.box((L / 2 - 8, yc - 5.0, -3.7), (L + 8, 0.4, 1.2), "metal", top="cut")           # near wall, cut
    shell.box((L - 0.4, yc, -3.7), (0.8, 10.4, 4.8), "metal")                                 # nose
    shell.box((L - 1, yc, -2.7), (1.2, 4.4, 2.6), "metal_bk")
    K.tag(shell.done(), "optics")
    z0 = -3.2
    pcb = Part(K, "module_pcb")
    pcb.box((8, yc, z0), (34, 8.6, 0.35), "pcb")
    for k in range(12):
        pcb.box((-6.3, yc - 3.85 + k * 0.7, z0 + 0.35), (2.6, 0.4, 0.04), "gold")
    for k in range(8):
        pcb.box((9 + k * 1.5, yc - 3.6, z0 + 0.35), (0.9, 0.5, 0.25), "mlcc")
    K.tag(pcb.done(), "optics")

    # DSP: turns the switch's electrical lanes into laser drive signals (and back)
    dz = z0 + 0.35                                                       # top of the module board
    dsp = Part(K, "dsp")
    dsp.box((5.5, yc, dz), (8.0, 8.0, 0.45), "substrate")
    dsp.box((5.5, yc, dz + 0.45), (5.4, 5.4, 0.55), "chip")
    dsp.box((5.5, yc, dz + 1.0), (3.4, 3.4, 0.06), "silicon")
    K.tag(dsp.done(), "dsp")
    drv = Part(K, "drivers")
    for y in (-2.0, 2.0):
        drv.box((13.5, yc + y, dz), (2.8, 2.8, 0.45), "chip")
    K.tag(drv.done(), "dsp")

    # lasers and receivers on a ceramic bench, lenses, silicon photonics (laser)
    las = Part(K, "lasers")
    las.box((19.5, yc, dz), (6.0, 8.2, 0.55), "white")
    for k in range(4):
        y = yc - 2.7 + k * 1.8
        las.box((18.8, y, dz + 0.55), (2.0, 1.2, 0.9), "gold")
        las.box((19.9, y, dz + 0.85), (0.3, 0.5, 0.35), "plasma")
    for k in range(4):
        las.box((21.6, yc - 2.7 + k * 1.8, dz + 0.55), (0.9, 0.9, 0.6), "chip")      # photodiodes
    K.tag(las.done(), "laser")
    lens = Part(K, "lenses")
    for k in range(8):
        lens.rod((23.6, yc - 3.15 + k * 0.9, dz + 1.05), (24.8, yc - 3.15 + k * 0.9, dz + 1.05), 0.42, "glass", 12)
    lens.box((24.2, yc, dz), (1.8, 8.2, 0.6), "white")
    K.tag(lens.done(), "laser")
    for k in range(4):                                                    # light from each laser
        y = yc - 2.7 + k * 1.8
        K.no_lines(K.rod((20.1, y, dz + 1.05), (30.5, yc + (y - yc) * 0.55, dz + 1.0), 0.1, "plasma", 6, "laser", "beam"))
    sip = Part(K, "photonics")
    sip.box((27.5, yc, dz), (5.0, 7.0, 0.6), "silicon")
    for k in range(6):
        sip.box((27.5, yc - 2.5 + k * 1.0, dz + 0.61), (4.6, 0.12, 0.02), "gold")
    K.tag(sip.done(), "laser")

    # fibre: array block, fibres to the MPO port, cable out of the nose (fiber)
    fb = Part(K, "fibre_array")
    fb.box((31.4, yc, dz), (2.2, 5.0, 1.5), "glass")
    for k in range(8):
        y = -1.75 + k * 0.5
        fb.rod((32.4, yc + y, dz + 0.75), (38.8, yc + y * 0.7, dz + 1.2), 0.13, "fiber", 6)
    fb.box((L + 1.6, yc, -2.7), (3.0, 4.0, 2.6), "fiber")
    K.tag(fb.done(), "fiber")
    K.tube([(L + 3.0, yc, -1.4), (L + 8.0, yc, -1.4), (L + 11.5, yc - 1.2, -3.0), (L + 15.0, yc - 3.0, -5.6)], 0.9, "fiber", "fiber")

    for zone in ("switch", "optics", "dsp", "laser", "fiber"):
        K.spot(zone)
    K.spot("switch", at=(-15.0, 0, 1.0))                                     # the switch chip's heatsink
    return {"width": 1200, "margin": (0.06, 0.06)}


# ---------------------------------------------------------------- inside the storage
def storage(K):
    """An all-flash storage system with its lid off (controllers behind a midplane, drives in
    front), one SSD pulled out and opened (NAND packages, controller, DRAM), and an opened hard
    disk (platters, spindle, actuator arm and heads)."""
    K.catcher((20, 30, -0.01), 120)
    # storage system (systems): chassis, controller modules, midplane, drive bays
    W, D, Hs = 44.0, 72.0, 8.8
    ch = Part(K, "array")
    ch.box((W / 2, D / 2, 0), (W, D, 0.3), "metal")
    for x in (0.2, W - 0.2):
        ch.box((x, D / 2, 0.3), (0.4, D, Hs - 0.3), "metal", top="metal_dk")
    ch.box((W / 2, D - 0.2, 0.3), (W, 0.4, Hs - 0.3), "metal_dk")
    ch.box((W / 2, 30.0, 0.3), (W - 0.8, 0.6, Hs - 0.6), "pcb")               # midplane
    for k in range(2):                                                    # two controller modules
        y = 40.0 + k * 15.5
        ch.box((W / 2, y, 0.4), (W - 1.6, 14.0, 0.4), "pcb")
        for cx in (12.0, 32.0):
            ch.box((cx, y, 0.8), (6.0, 6.0, 0.4), "substrate")
            for f in range(9):
                ch.box((cx - 2.8 + f * 0.7, y, 1.2), (0.25, 5.6, 3.0), "metal")
        for d in range(6):
            ch.box((19.5 + d * 0.9, y - 4.0, 0.8), (0.35, 6.0, 2.6), "metal_bk")   # memory
        ch.box((W / 2, y + 6.3, 0.8), (W - 3, 1.0, 2.0), "metal_dk")
    ch.box((W / 2, -0.9, 0), (W, 1.8, Hs), "white")                       # bezel
    ch.box((W / 2, -1.82, 3.6), (W - 6, 0.04, 1.2), "red")
    K.tag(ch.done(), "systems")
    bay = Part(K, "drives")
    for k in range(24):                                                   # E1/U.2 drives standing in their bays
        if k == 9:
            continue
        x = 1.6 + k * 1.7
        bay.box((x, 14.0, 0.4), (1.3, 28.0, 7.6), "metal_dk")
        bay.box((x, 0.1, 0.4), (1.3, 0.2, 7.6), "metal_bk")
        bay.box((x, -0.05, 6.6), (0.5, 0.1, 0.5), "led_b")
    K.tag(bay.done(), "flash")
    # the SSD pulled out, and its insides opened beside it (flash)
    ssd = Part(K, "ssd")
    sx = 1.6 + 9 * 1.7
    ssd.box((sx, -16.0, 0.4), (1.3, 28.0, 7.6), "metal_dk")
    ssd.box((sx, -30.1, 0.4), (1.3, 0.2, 7.6), "metal_bk")
    K.tag(ssd.done(), "flash")
    op = Part(K, "ssd_open")
    ox, oy = -16.0, -14.0                                                 # the board, laid flat and enlarged
    op.box((ox, oy, 0), (14.0, 32.0, 0.6), "metal")                       # tray of the casing
    op.box((ox, oy, 0.6), (12.6, 30.4, 0.3), "pcb")
    op.box((ox, oy - 10.5, 0.9), (5.4, 5.4, 0.45), "chip")                # controller
    op.box((ox, oy - 10.5, 1.35), (3.0, 3.0, 0.05), "silicon")
    op.box((ox + 4.3, oy - 10.5, 0.9), (2.6, 3.4, 0.35), "metal_bk")      # DRAM cache
    for i in range(2):                                                    # NAND packages
        for j in range(4):
            op.box((ox - 3.0 + i * 6.0, oy - 4.0 + j * 5.6, 0.9), (5.0, 4.6, 0.45), "chip")
            op.box((ox - 3.0 + i * 6.0, oy - 4.0 + j * 5.6, 1.35), (3.6, 3.2, 0.04), "hbm")
    for k in range(8):
        op.box((ox - 5.6 + k * 1.6, oy - 15.9, 0.9), (0.8, 0.5, 0.3), "mlcc")
    for k in range(14):
        op.box((ox - 5.2 + k * 0.8, oy + 15.5, 0.6), (0.45, 1.2, 0.08), "gold")   # edge connector
    K.tag(op.done(), "flash")
    for k in range(6):                                                    # dashed guide from the pulled drive
        K.no_lines(K.box((sx - 3.5 - k * 3.3, -22.0, 4.5), (1.8, 0.25, 0.25), "metal_dk", name="guide"))

    # an opened hard disk (disk), enlarged
    hx, hy = 70.0, 20.0
    hd = Part(K, "hdd")
    hd.box((hx, hy, 0), (22.0, 30.0, 0.8), "metal")                        # base casting: floor
    hd.box((hx, hy + 14.7, 0.8), (22.0, 0.6, 4.2), "metal", top="metal_dk")   # far walls full height
    hd.box((hx - 10.7, hy, 0.8), (0.6, 30.0, 4.2), "metal", top="metal_dk")
    hd.box((hx, hy - 14.7, 0.8), (22.0, 0.6, 1.2), "metal", top="cut")        # near walls cut low
    hd.box((hx + 10.7, hy, 0.8), (0.6, 30.0, 1.2), "metal", top="cut")
    for k in range(4):                                                    # platters on the spindle
        hd.rod((hx - 1.0, hy - 4.0, 1.6 + k * 0.8), (hx - 1.0, hy - 4.0, 1.72 + k * 0.8), 9.0, "platter", 64)
    hd.rod((hx - 1.0, hy - 4.0, 0.8), (hx - 1.0, hy - 4.0, 5.0), 1.8, "metal", 32)       # spindle hub
    hd.rod((hx - 1.0, hy - 4.0, 5.0), (hx - 1.0, hy - 4.0, 5.2), 1.2, "metal_dk", 32)
    hd.rod((hx + 7.0, hy + 9.0, 0.8), (hx + 7.0, hy + 9.0, 5.0), 1.1, "metal", 24)        # actuator pivot
    for k in range(4):                                                    # arms reaching over the platters
        z = 2.0 + k * 0.8
        hd.box((hx + 3.4, hy + 3.6, z), (1.4, 13.0, 0.12), "metal", rz=-0.58)
        hd.box((hx - 0.2, hy - 1.8, z - 0.1), (0.7, 0.9, 0.14), "gold", rz=-0.58)          # head
    hd.box((hx + 8.2, hy + 12.0, 0.8), (4.2, 3.2, 2.2), "metal_bk")        # voice-coil magnet
    hd.box((hx, hy - 15.6, 0.0), (18.0, 1.0, 4.2), "pcb")                  # board on the connector side
    K.tag(hd.done(), "disk")

    for zone in ("systems", "flash", "disk"):
        K.spot(zone)
    K.spot("flash", at=(ox - 3.0, oy + 7.2, 1.4))                          # on the NAND packages
    return {"width": 1200, "margin": (0.06, 0.06)}


PANELS = {"dc": campus, "dc_optics": optics, "dc_storage": storage}
