"""Place the AI 탐방 scene labels  —  py -m desk.scenes.layout

Reads the Blender export (desk/scenes/anchors/<panel>.json: zone anchors, hover masks and an
occupancy grid) and the tour map (zone names, room numbers, ★ leads), then puts every label
next to its zone with a short leader line: over empty space where it can, never touching
another label, never covering a zone's anchor. Writes desk/static/scenes/scenes.json.
Run it again whenever a zone name, lead or bottleneck changes (tests/test_scenes.py checks).
"""

from __future__ import annotations

import json
import math
import random
import sys

from desk import tour
from desk.scenes import ANCHORS, SCENES_JSON

# scene id (tour_map.json stop "scene") -> title and its panels in reading order; each panel has
# its own caption and legend (keys = flow colours baked into that image, tour.js TR_INK).
SCENES = {
    "power": {"title": "전력 단면도", "panels": {
        "power": ("", "발전소에서 만든 전기가 송전탑과 변전소를 거쳐 오른쪽 데이터센터로 갑니다. 원자력은 우라늄 연료가, "
                      "가스터빈은 천연가스가 필요합니다.", ["power"]),
    }},
    "fab": {"title": "반도체 공장 단면도", "panels": {
        "fab": ("팹(전공정)과 패키징 공장", "웨이퍼는 노광기에서 회로를 새기고 식각·증착 장비에서 깎고 쌓기를 수십 번 되풀이한 뒤 "
                "검사와 시험을 거칩니다. 아래층(서브팹)이 가스·약품과 진공을 대고, 오른쪽 후공정동에서 HBM을 쌓아 패키지로 묶습니다.", []),
        "fab_pkg": ("칩 패키지 속", "GPU 칩 하나를 층별로 떼어 본 모습입니다. GPU 다이 옆에 HBM을 붙이고(CoWoS), 실리콘 인터포저와 "
                    "기판이 작은 범프 수만 개로 이들을 서버 기판까지 잇습니다.", []),
    }},
    "server": {"title": "AI 서버 단면도", "panels": {
        "rack": ("랙 하나", "GPU 72개가 뒤쪽 구리선 묶음(NVLink 스파인)으로 이어져 한 대처럼 일합니다. 위아래는 전원 선반, "
                 "뒤쪽 양옆은 냉각수 관, 앞쪽 하늘색 선은 바깥 네트워크로 가는 광섬유입니다.", ["cold", "hot", "fiber"]),
        "tray": ("연산 트레이 하나", "랙에서 꺼낸 트레이 하나입니다. GPU 4개와 CPU 2개가 냉각판 아래에 있고, 앞쪽 NIC·광모듈로 "
                 "바깥과, 뒤쪽 커넥터로 랙 안 스위치와 이어집니다. 오른쪽은 GPU 대신 쓰는 맞춤형 칩 보드입니다.", ["cold", "hot", "fiber"]),
    }},
    "dc": {"title": "AI 데이터센터 단면도", "panels": {
        "dc": ("캠퍼스 전체", "전기는 왼쪽 변전소에서 전기실과 노란 버스웨이를 거쳐 GPU 랙으로, 냉각수는 오른쪽 냉각기에서 파란 관으로 "
               "들어가 빨간 관으로 나옵니다. 데이터는 하늘색 광섬유로 네트워크실을 지나 다른 데이터센터로 나갑니다.",
               ["power", "cold", "hot", "fiber"]),
        "dc_optics": ("광모듈 속", "스위치 칩의 전기 신호를 DSP가 정리하고 레이저가 빛으로 바꿔 광섬유로 보냅니다. 받을 때는 "
                      "빛을 다시 전기로 바꿉니다.", ["light", "fiber"]),
        "dc_storage": ("저장장치 속", "가운데 저장 시스템은 컨트롤러와 드라이브 수십 개로 이뤄집니다. 왼쪽은 SSD를 열어 본 모습"
                       "(낸드 칩·컨트롤러), 오른쪽은 HDD를 열어 본 모습(자기 원판·헤드)입니다.", []),
    }},
}
LEGEND = {"power": "전력", "cold": "차가운 냉각수", "hot": "데워진 냉각수", "fiber": "광섬유(데이터)", "light": "레이저 빛"}

# label metrics in CSS px at 1:1 — keep in step with tour.css (.scp-lab)
NUM_PX, NAME_PX, SUB_PX = 10.5, 12.5, 10.5
PAD_X, PAD_Y, LINE1, LINE2, NUM_GAP = 7, 5, 15, 13, 5
GAP = 6          # clear space between two labels
EDGE = 4         # keep labels this far inside the picture


def _em(ch: str) -> float:
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3 or 0x3130 <= o <= 0x318F:
        return 1.0
    if ch == " ":
        return 0.3
    if ch in "·.,:;'":
        return 0.34
    if ch in "()[]-/":
        return 0.42
    if ch == "★":
        return 1.0
    if ch.isdigit():
        return 0.62
    if ch.isupper():
        return 0.7
    return 0.58 if ch.islower() else 0.64


def text_w(s: str, px: float, bold: bool = False) -> float:
    return sum(_em(c) for c in s) * px * (1.06 if bold else 1.0)


def label_text(m: dict, zone: str) -> tuple[str, str, str]:
    """(room number, zone name, ★ lead · 병목) exactly as the page prints them."""
    for s in m["stops"]:
        for k, z in enumerate(s["zones"]):
            if z["id"] == zone:
                lead = z.get("lead")
                short = (m["companies"].get(lead) or {}).get("short") or lead if lead else ""
                sub = " · ".join(x for x in ((f"★ {short}" if short else ""), ("병목" if z.get("bottleneck") else "")) if x)
                return f"{s['no']}-{k + 1}", z["name"], sub
    raise KeyError(f"zone {zone} is not on the tour map")


def label_size(num: str, name: str, sub: str) -> tuple[float, float]:
    w1 = text_w(num, NUM_PX) * 0.95 + NUM_GAP + text_w(name, NAME_PX, bold=True)
    w2 = text_w(sub, SUB_PX) if sub else 0
    return math.ceil(max(w1, w2) + 2 * PAD_X), PAD_Y * 2 + LINE1 + (LINE2 if sub else 0)


class Occupancy:
    """Summed-area table over the export's occupancy grid (0 = empty .. 1 = drawn thing)."""

    def __init__(self, occ: dict):
        self.cell = occ["cell"]
        rows = [[int(c, 16) / 15 for c in r] for r in occ["rows"]]
        self.h, self.w = len(rows), len(rows[0])
        s = [[0.0] * (self.w + 1) for _ in range(self.h + 1)]
        for y in range(self.h):
            acc = 0.0
            for x in range(self.w):
                acc += rows[y][x]
                s[y + 1][x + 1] = s[y][x + 1] + acc
        self.s = s

    def mean(self, x0, y0, x1, y1) -> float:
        c = self.cell
        a, b = max(0, int(x0 // c)), max(0, int(y0 // c))
        e, f = min(self.w, int(math.ceil(x1 / c))), min(self.h, int(math.ceil(y1 / c)))
        if e <= a or f <= b:
            return 0.0
        s = self.s
        return (s[f][e] - s[b][e] - s[f][a] + s[b][a]) / ((e - a) * (f - b))


def _seg_hit_box(p, q, box, pad=2.0) -> bool:
    """Does segment p-q pass through the box (Liang-Barsky clip)?"""
    x0, y0, w, h = box
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return t1 - t0 > 0.02


def _cross(a, b, c, d) -> bool:
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def _near_point(box, p):
    x0, y0, w, h = box
    return (min(max(p[0], x0), x0 + w), min(max(p[1], y0), y0 + h))


class _Cand:
    __slots__ = ("i", "cost", "box", "end", "anchor", "reg")

    def __init__(self, i, cost, box, end, anchor):
        self.i, self.cost, self.box, self.end, self.anchor = i, cost, box, end, anchor
        x0, y0, w, h = box
        self.reg = (min(x0, anchor[0]) - GAP, min(y0, anchor[1]) - GAP,
                    max(x0 + w, anchor[0]) + GAP, max(y0 + h, anchor[1]) + GAP)


def _pair(a: _Cand, b: _Cand) -> float:
    ra, rb = a.reg, b.reg
    if ra[2] <= rb[0] or rb[2] <= ra[0] or ra[3] <= rb[1] or rb[3] <= ra[1]:
        return 0.0                                       # far apart: nothing can touch
    ba, bb = a.box, b.box
    if not (ba[0] + ba[2] + GAP <= bb[0] or bb[0] + bb[2] + GAP <= ba[0]
            or ba[1] + ba[3] + GAP <= bb[1] or bb[1] + bb[3] + GAP <= ba[1]):
        return 1e5                                       # labels overlap
    cost = 0.0
    if _seg_hit_box(a.anchor, a.end, bb):
        cost += 300
    if _seg_hit_box(b.anchor, b.end, ba):
        cost += 300
    if _cross(a.anchor, a.end, b.anchor, b.end):
        cost += 70
    return cost


def place(W: int, H: int, occ: Occupancy, items: list[dict], seed: int = 7, keep: int = 90) -> list[dict]:
    """items: {zone, anchor, w, h}. Returns each with box [x, y, w, h] and leader end [x, y]."""
    anchors = [tuple(it["anchor"]) for it in items]
    cands = []
    for i, it in enumerate(items):
        ax, ay = anchors[i]
        w, h = it["w"], it["h"]
        opts = []
        for r in (14, 26, 40, 58, 80, 106, 136, 170, 210, 260, 320):
            for k in range(32):
                t = 2 * math.pi * k / 32
                ux, uy = math.cos(t), math.sin(t)
                ext = min(w / 2 / max(abs(ux), 1e-6), h / 2 / max(abs(uy), 1e-6))
                cx, cy = ax + ux * (r + ext), ay + uy * (r + ext)
                box = (round(cx - w / 2, 1), round(cy - h / 2, 1), w, h)
                if box[0] < EDGE or box[1] < EDGE or box[0] + w > W - EDGE or box[1] + h > H - EDGE:
                    continue
                end = _near_point(box, (ax, ay))
                length = math.dist(end, (ax, ay))
                cost = 0.045 * length + 60 * occ.mean(box[0], box[1], box[0] + w, box[1] + h)
                for j, (bx, by) in enumerate(anchors):          # never sit on an anchor
                    if box[0] - 8 <= bx <= box[0] + w + 8 and box[1] - 8 <= by <= box[1] + h + 8:
                        cost += 400
                    elif j != i and _seg_hit_box((ax, ay), end, (bx - 5, by - 5, 10, 10), 0):
                        cost += 40
                opts.append(_Cand(i, cost, box, end, (ax, ay)))
        opts.sort(key=lambda o: o.cost)
        cands.append(opts[:keep])

    n = len(items)
    rng = random.Random(seed)
    best, best_cost = None, float("inf")
    for _ in range(10):
        order = list(range(n))
        rng.shuffle(order)
        pick: list[_Cand | None] = [None] * n
        for i in order:                                  # greedy start
            others = [p for p in pick if p is not None]
            pick[i] = min(cands[i], key=lambda o: o.cost + sum(_pair(o, p) for p in others))
        for _ in range(8):                               # then improve one label at a time
            changed = False
            for i in order:
                others = [pick[j] for j in range(n) if j != i]
                def total(o):
                    return o.cost + sum(_pair(o, p) for p in others)
                nb = min(cands[i], key=total)
                if total(nb) + 1e-9 < total(pick[i]):
                    pick[i], changed = nb, True
            if not changed:
                break
        cost = sum(o.cost for o in pick) + sum(_pair(pick[i], pick[j]) for i in range(n) for j in range(i + 1, n))
        if cost < best_cost:
            best, best_cost = list(pick), cost
    return [{**it, "box": list(c.box), "end": [round(c.end[0], 1), round(c.end[1], 1)]} for it, c in zip(items, best)]


def build(m: dict | None = None) -> dict:
    m = m or tour.load_map()
    panels, problems = {}, []
    for sid, spec in SCENES.items():
        for pid, (ptitle, caption, legend) in spec["panels"].items():
            path = ANCHORS / f"{pid}.json"
            if not path.exists():
                problems.append(f"{pid}: 렌더 결과가 없습니다 (desk/scenes/blender/build.py {pid})")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            items = []
            for sp in data["spots"]:
                if sp.get("hidden"):
                    problems.append(f"{pid}: {sp['zone']} 분야가 그림에서 보이지 않습니다")
                    continue
                num, name, sub = label_text(m, sp["zone"])
                w, h = label_size(num, name, sub)
                items.append({"zone": sp["zone"], "anchor": sp["anchor"], "w": w, "h": h,
                              "n": num, "text": [num, name, sub], "path": sp["path"], "bbox": sp["bbox"]})
            placed = place(data["w"], data["h"], Occupancy(data["occ"]), items)
            spots = [{k: v for k, v in p.items() if k not in ("w", "h")} for p in placed]
            panels[pid] = {"scene": sid, "title": ptitle, "caption": caption, "legend": [[k, LEGEND[k]] for k in legend],
                           "img": data["img"], "w": data["w"], "h": data["h"], "spots": spots}
    scenes = {sid: {"title": s["title"], "panels": [p for p in s["panels"] if p in panels]} for sid, s in SCENES.items()}
    return {"version": 1, "scenes": scenes, "panels": panels, "problems": problems}


def main() -> int:
    out = build()
    SCENES_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCENES_JSON.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for pid, p in out["panels"].items():
        print(f"{pid}: {len(p['spots'])}개 라벨 · {p['w']}x{p['h']}")
    for msg in out["problems"]:
        print("문제:", msg)
    print(f"→ {SCENES_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
