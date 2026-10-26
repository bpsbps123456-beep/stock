"""구조물 지도 Pine 지표의 파이썬 기준 구현 (검증용).

Pine(`structure_map.pine`)과 같은 규칙으로 BTCUSDT 봉에서 사건을 뽑아
작가(쉽알남) 차트의 표시와 맞는지 본다. 기본값은 Pine 입력 기본값과 같다.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import sys
from dataclasses import dataclass, field

CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache"

# ── 기본값 (Pine 입력과 같게 유지) ─────────────────────────
PIV = 5            # 추세선 기준점 피벗 강도
MIN_GAP = 10       # 두 기준점 최소 간격(봉)
MAX_LOOK = 300     # 기준점 최대 거리(봉)
MAX_SLOPE = 0.5    # 기울기 상한 (ATR/봉)
TOL_B = 0.3        # 꼬리 이탈로 보는 최소 폭 (ATR 배수)
TOL_T = 0.0        # 리테스트 닿음 허용 폭 (ATR 배수)
K_F = 3            # 이 봉 수 이하로 밖에서 마감 후 복귀 → Fake out
K_T = 12           # 이 봉 수 이하 → Trap, 초과 → 돌파 확정
LIQ_PIV = 10       # 유동성 피벗 강도
OB_RATIO = 2.0     # 오더블럭: 장대봉 몸통 ≥ 앞 봉 몸통 × 2
DOJI_ATR = 0.1     # 오더블럭: 앞 봉 몸통이 ATR × 0.1 미만이면 도지로 제외
FVG_RATIO = 2.0    # FVG: 가운데 봉 몸통 ≥ 앞·뒤 봉 몸통 × 2
SWEEP_LOOK = 20    # 별표(유동성 흡수): 직전 20봉 저점/고점 갱신
STRUCT_PIV = 5     # 상위 봉 구조 피벗 강도

EV_NAME = {1: "마감 돌파", 2: "안착", 3: "Fake out", 4: "Trap", 5: "Fake out(꼬리)",
           6: "돌파 확정", 7: "돌파 확정(Fake out 무효)", 8: "리테스트"}


def load(iv: str):
    rows = json.loads((CACHE / f"BTCUSDT_{iv}.json").read_text())
    return [dict(t=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in rows]


def kst(ms: int, year: bool = False) -> str:
    fmt = "%Y-%m-%d %H:%M" if year else "%m-%d %H:%M"
    return (dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc) + dt.timedelta(hours=9)).strftime(fmt)


def atr_series(b, n=14):
    out, prev = [], None
    for i, r in enumerate(b):
        tr = r["h"] - r["l"] if i == 0 else max(r["h"] - r["l"], abs(r["h"] - b[i - 1]["c"]), abs(r["l"] - b[i - 1]["c"]))
        prev = tr if prev is None else (prev * (n - 1) + tr) / n   # Wilder(RMA), Pine ta.atr 와 같음
        out.append(prev)
    return out


def sma(xs, n):
    return [None if i < n - 1 else sum(xs[i - n + 1:i + 1]) / n for i in range(len(xs))]


def ema(xs, n):
    a, out, e = 2 / (n + 1), [], None
    for i, x in enumerate(xs):
        if i < n - 1:
            out.append(None)
            continue
        e = sum(xs[:n]) / n if e is None else a * x + (1 - a) * e
        out.append(e)
    return out


def pivot_at(b, i, p, hi=True):
    """bar i 가 좌우 p 봉 피벗인지 (왼쪽은 초과, 오른쪽은 이상. 같은 값 처리는 Pine ta.pivothigh와 다를 수 있으나 가격 동률은 드물다)."""
    if i - p < 0 or i + p >= len(b):
        return False
    key = "h" if hi else "l"
    v = b[i][key]
    left = [b[k][key] for k in range(i - p, i)]
    right = [b[k][key] for k in range(i + 1, i + p + 1)]
    if hi:
        return v > max(left) and v >= max(right)
    return v < min(left) and v <= min(right)


# ── 돌파 판정기 (Pine type Watch / method step 과 같은 규칙) ──
@dataclass
class Watch:
    s: int                 # +1: 위로 뚫는지 본다(저항), -1: 아래로(지지)
    ep: int = 0            # 0 없음, 1 꼬리만, 2 밖에서 마감 중, 3 복귀 뒤 꼬리
    cnt: int = 0
    ext: float = math.nan
    ext_i: int = -1
    prior: float = math.nan
    last_ev: int = 0
    rt_done: bool = False

    def step(self, r, lv, atr, tol_b, tol_t, i):
        s = self.s
        y = s * lv
        ext = r["h"] if s > 0 else -r["l"]
        c, o = s * r["c"], s * r["o"]
        bc = c > y
        br = ext > y + tol_b * atr
        ev = 0
        if self.ep == 0:
            if bc:
                self.ep, self.cnt, self.ext, self.ext_i = 2, 1, ext, i
                ev = 1
                if not math.isnan(self.prior) and ext > self.prior:
                    ev = 7
            elif br:
                self.ep, self.ext, self.ext_i = 1, ext, i
            elif self.last_ev > 0 and not self.rt_done and ext >= y - tol_t * atr:
                ev, self.rt_done = 8, True
        elif self.ep == 1:
            if ext > self.ext:
                self.ext, self.ext_i = ext, i
            if bc:
                self.ep, self.cnt = 2, 1
                ev = 1
                if not math.isnan(self.prior) and self.ext > self.prior:
                    ev = 7
            elif not br:
                ev = 5
                self.prior = self.ext if math.isnan(self.prior) else max(self.prior, self.ext)
                self.last_ev, self.rt_done, self.ep = 1, False, 0
        elif self.ep == 2:
            if ext > self.ext:
                self.ext, self.ext_i = ext, i
            if bc:
                self.cnt += 1
                if self.cnt == 2 and o > y:
                    ev = 2
                if not math.isnan(self.prior) and self.ext > self.prior:
                    ev = 7
                elif self.cnt > K_T:
                    ev = 6
            else:
                ev = 3 if self.cnt <= K_F else 4
                self.prior = self.ext if math.isnan(self.prior) else max(self.prior, self.ext)
                self.last_ev, self.rt_done = 1, False
                self.ep = 3 if br else 0
        else:  # 3: 복귀 뒤 꼬리
            if bc:
                self.ep, self.cnt, self.ext, self.ext_i = 2, 1, ext, i
                ev = 1
                if ext > self.prior:
                    ev = 7
            elif br:
                if ext > self.ext:
                    self.ext, self.ext_i = ext, i
                self.prior = max(self.prior, ext)
            else:
                self.ep = 0
        if ev in (6, 7):
            self.s = -self.s
            self.ep, self.cnt, self.prior, self.last_ev, self.rt_done = 0, 0, math.nan, 2, False
        return ev


@dataclass
class TL:
    d: int            # -1 하락 추세선(고점), +1 상승 추세선(저점)
    ia: int
    pa: float
    ib: int
    pb: float
    w: float = 0.0
    broken: bool = False
    est: bool = False  # 확립: 기준점 B 뒤 PROV 봉 동안 다시 긋지 않았으면 판정을 시작한다
    wm: Watch = None
    ws: Watch = None
    four: bool = False

    def val(self, i):
        return self.pa + (self.pb - self.pa) * (i - self.ia) / (self.ib - self.ia)

    def sub(self, i):
        return self.val(i) - self.w if self.d < 0 else self.val(i) + self.w


def find_line(b, piv, ib, pb, d, now, atr):
    """기준점 B(ib, pb)에 대해 가장 먼 유효 기준점 A를 찾는다. 없으면 None."""
    for ia, pa in piv:                      # 오래된 것부터 → 처음 맞는 것이 가장 먼 것
        gap = ib - ia
        if gap < MIN_GAP or gap > MAX_LOOK:
            continue
        if (d < 0 and pa <= pb) or (d > 0 and pa >= pb):
            continue
        if abs(pb - pa) / gap > MAX_SLOPE * atr:
            continue
        t = TL(d, ia, pa, ib, pb)
        ok = True
        for k in range(ia + 1, ib):
            if (d < 0 and b[k]["h"] > t.val(k)) or (d > 0 and b[k]["l"] < t.val(k)):
                ok = False
                break
        if ok:
            for k in range(ib + 1, now + 1):
                if (d < 0 and b[k]["c"] > t.val(k)) or (d > 0 and b[k]["c"] < t.val(k)):
                    ok = False
                    break
        if not ok:
            continue
        t.w = max((t.val(k) - b[k]["l"]) if d < 0 else (b[k]["h"] - t.val(k)) for k in range(ia, ib + 1))
        t.wm = Watch(s=1 if d < 0 else -1)
        t.ws = Watch(s=-1 if d < 0 else 1)
        return t
    return None


PROV = 20         # 다시 긋기 허용 기간(기준점 B 뒤 봉 수)
STALE = 5.0       # 확립된 선이 현재가에서 이 ATR 배수보다 멀면 새 선으로 바꿀 수 있다
DISCARD = 1.0     # 확립 전 선을 종가가 이 ATR 배수 넘게 뚫으면 버린다


def run_trendlines(b, atr, show_from, label=""):
    out = []
    piv = {-1: [], 1: []}
    active = {-1: None, 1: None}
    flipped = {-1: None, 1: None}
    for i in range(len(b)):
        r = b[i]
        # 0) 돌파 확정된 선(전환 선): 리테스트·꼬리 거짓 돌파만 보고, 반대로 마감하면 '돌파 실패'로 끝낸다
        for d in (-1, 1):
            t = flipped[d]
            if t is None:
                continue
            name = f"{'하락' if d < 0 else '상승'}선(전환)"
            if abs(r["c"] - t.val(i)) > STALE * atr[i]:
                flipped[d] = None
                continue
            ev = t.wm.step(r, t.val(i), atr[i], TOL_B, TOL_T, i)
            if ev == 8:
                out.append((i, name, "리테스트(지지·저항 전환)", t, None))
            elif ev == 5:
                out.append((i, name, "Fake out(꼬리, 전환 확인)", t, (t.wm.ext_i, abs(t.wm.ext))))
            elif ev:
                out.append((i, name, "돌파 실패(Trap)", t, None))
                flipped[d] = None
        # 1) 사건 판정 (이번 봉 마감 기준)
        for d in (-1, 1):
            t = active[d]
            if t is not None and not t.est:
                beyond = (r["c"] - t.val(i)) * (1 if d < 0 else -1)
                if beyond > DISCARD * atr[i]:
                    active[d] = None
                    out.append((i, f"{'하락' if d < 0 else '상승'}선", "확립 전 이탈 → 버림", t, None))
                elif i - t.ib >= PROV:
                    t.est = True
                    out.append((i, f"{'하락' if d < 0 else '상승'}선", "확립", t, None))
                continue
            for t in [x for x in (active[d],) if x is not None]:
                name = f"{'하락' if d < 0 else '상승'}선"
                ev = t.wm.step(r, t.val(i), atr[i], TOL_B, TOL_T, i)
                if ev:
                    exi = (t.wm.ext_i, abs(t.wm.ext)) if ev in (3, 4, 5) else None
                    out.append((i, name, EV_NAME[ev], t, exi))
                    if ev in (6, 7):
                        t.broken = True
                        flipped[d] = t
                        active[d] = None
                        continue
                if not t.broken:
                    sv = t.sub(i)
                    if not t.four and ((d < 0 and r["l"] <= sv) or (d > 0 and r["h"] >= sv)):
                        t.four = True
                        out.append((i, f"{'하락' if d < 0 else '상승'}채널", "4포인트", t, None))
                    ev2 = t.ws.step(r, sv, atr[i], TOL_B, TOL_T, i)
                    if ev2:
                        out.append((i, f"{'하락' if d < 0 else '상승'}채널 반대선", EV_NAME[ev2], t, None))
        # 2) 피벗 확정 → 새 추세선
        j = i - PIV
        for d, hi in ((-1, True), (1, False)):
            if j >= 0 and pivot_at(b, j, PIV, hi):
                p = b[j]["h"] if hi else b[j]["l"]
                cur = active[d]
                if cur is not None and not cur.broken and i - cur.ia > MAX_LOOK:
                    active[d] = cur = None
                stale = cur is not None and cur.est and not cur.broken and abs(r["c"] - cur.val(i)) > STALE * atr[i]
                if cur is None or cur.broken or stale or not cur.est:
                    t = find_line(b, piv[d], j, p, d, i, atr[i])
                    # 확립 전 선은 같은 시작점이나 더 먼 시작점으로만 다시 긋는다(안쪽의 짧은 선으로 바꾸지 않음)
                    if t is not None and (cur is None or cur.broken or stale or t.ia <= cur.ia):
                        active[d] = t
                        why = "새 추세선" if cur is None or cur.broken else ("오래된 선 교체" if stale else "다시 긋기")
                        out.append((i, f"{'하락' if d < 0 else '상승'}선", why, t, None))
                piv[d].append((j, p))
                piv[d] = [(a, q) for a, q in piv[d] if i - a <= MAX_LOOK + PIV][-40:]
    return [e for e in out if e[0] >= show_from]


def run_liquidity(b, atr, show_from):
    out, lv = [], []
    for i in range(len(b)):
        r = b[i]
        keep = []
        for L in lv:
            ev = L["w"].step(r, L["p"], atr[i], 0.0, 0.0, i)
            if ev in (3, 4, 5):
                out.append((i, "유동성", f"스윕 {L['p']:.0f} ({kst(b[L['i']]['t'])})"))
            elif ev in (6, 7):
                out.append((i, "유동성", f"돌파 {L['p']:.0f} ({kst(b[L['i']]['t'])})"))
            else:
                keep.append(L)
        lv = keep
        j = i - LIQ_PIV
        if j >= 0:
            if pivot_at(b, j, LIQ_PIV, True):
                lv.append(dict(p=b[j]["h"], i=j, w=Watch(s=1)))
            if pivot_at(b, j, LIQ_PIV, False):
                lv.append(dict(p=b[j]["l"], i=j, w=Watch(s=-1)))
    return [e for e in out if e[0] >= show_from], lv


def run_zones(b, atr, show_from):
    """오더블럭·FVG 생성/터치/무효/만료 수 세기."""
    stats = dict(ob=0, ob_star=0, fvg=0, fvg_star=0)
    events = []
    zones = []
    for i in range(3, len(b)):
        r0, r1, r2 = b[i], b[i - 1], b[i - 2]
        body = lambda r: abs(r["c"] - r["o"])
        lowest_prev = min(x["l"] for x in b[max(0, i - 2 - SWEEP_LOOK):i - 2]) if i - 2 > 0 else math.inf
        highest_prev = max(x["h"] for x in b[max(0, i - 2 - SWEEP_LOOK):i - 2]) if i - 2 > 0 else -math.inf
        # 상승 OB: 음봉(1) → 장대 양봉(0)
        if r1["c"] < r1["o"] and r0["c"] > r0["o"] and body(r1) >= DOJI_ATR * atr[i] and body(r0) >= OB_RATIO * body(r1) and r0["c"] > r1["o"]:
            stop = min(r1["l"], r0["l"], r2["l"] if r2["c"] > r2["o"] else math.inf)
            star = stop < lowest_prev
            zones.append(dict(k="OB", d=1, top=r1["o"], bot=r1["c"], stop=stop, i0=i - 1, star=star, touched=False))
            stats["ob"] += 1
            stats["ob_star"] += star
            if i >= show_from:
                events.append((i, "상승 OB" + ("★" if star else ""), f"{r1['c']:.0f}~{r1['o']:.0f}"))
        if r1["c"] > r1["o"] and r0["c"] < r0["o"] and body(r1) >= DOJI_ATR * atr[i] and body(r0) >= OB_RATIO * body(r1) and r0["c"] < r1["o"]:
            stop = max(r1["h"], r0["h"], r2["h"] if r2["c"] < r2["o"] else -math.inf)
            star = stop > highest_prev
            zones.append(dict(k="OB", d=-1, top=r1["c"], bot=r1["o"], stop=stop, i0=i - 1, star=star, touched=False))
            stats["ob"] += 1
            stats["ob_star"] += star
            if i >= show_from:
                events.append((i, "하락 OB" + ("★" if star else ""), f"{r1['o']:.0f}~{r1['c']:.0f}"))
        # FVG
        if r0["l"] > r2["h"] and r1["c"] > r1["o"] and body(r1) >= FVG_RATIO * body(r0) and body(r1) >= FVG_RATIO * body(r2):
            stats["fvg"] += 1
            if i >= show_from:
                events.append((i, "상승 FVG", f"{r2['h']:.0f}~{r0['l']:.0f}"))
        if r0["h"] < r2["l"] and r1["c"] < r1["o"] and body(r1) >= FVG_RATIO * body(r0) and body(r1) >= FVG_RATIO * body(r2):
            stats["fvg"] += 1
            if i >= show_from:
                events.append((i, "하락 FVG", f"{r0['h']:.0f}~{r2['l']:.0f}"))
    return stats, events


def run_structure(b, show_from):
    """상위 봉 구조: 마지막 고점/저점, 방향, 추세 전환 유동성."""
    out = []
    last_ph = last_pl = None
    ph_broken = pl_broken = False
    d = 0
    for i in range(len(b)):
        j = i - STRUCT_PIV
        if j >= 0 and pivot_at(b, j, STRUCT_PIV, True):
            last_ph, ph_broken = (b[j]["h"], j), False
        if j >= 0 and pivot_at(b, j, STRUCT_PIV, False):
            last_pl, pl_broken = (b[j]["l"], j), False
        c = b[i]["c"]
        if last_ph and not ph_broken and c > last_ph[0]:
            out.append((i, "구조", f"{'추세 전환' if d == -1 else '구조 돌파'} ↑ {last_ph[0]:.0f}"))
            d, ph_broken = 1, True
        if last_pl and not pl_broken and c < last_pl[0]:
            out.append((i, "구조", f"{'추세 전환' if d == 1 else '구조 돌파'} ↓ {last_pl[0]:.0f}"))
            d, pl_broken = -1, True
        flip = (last_ph if d == -1 and not ph_broken else last_pl if d == 1 and not pl_broken else None)
        b[i]["dir"], b[i]["flip"] = d, flip
    return [e for e in out if e[0] >= show_from]


def main(iv: str, since: str):
    b = load(iv)
    atr = atr_series(b)
    show_from = next(i for i, r in enumerate(b) if kst(r["t"], True) >= since)
    print(f"\n===== {iv}  (ATR 마지막 {atr[-1]:.0f}) =====")
    for i, who, what, t, exi in run_trendlines(b, atr, show_from):
        extra = ""
        if what in ("새 추세선", "오래된 선 교체", "다시 긋기"):
            extra = f" A {kst(b[t.ia]['t'])} {t.pa:.0f} → B {kst(b[t.ib]['t'])} {t.pb:.0f}, 채널 폭 {t.w:.0f}"
        elif exi:
            extra = f" (극점 {kst(b[exi[0]]['t'])} {exi[1]:.0f}, 선 {t.val(i):.0f})"
        else:
            extra = f" (선 {t.val(i):.0f}, 종가 {b[i]['c']:.0f})"
        print(f"{kst(b[i]['t'])} {who} {what}{extra}")
    ev, lv = run_liquidity(b, atr, show_from)
    for i, who, what in ev:
        print(f"{kst(b[i]['t'])} {who} {what}")
    print("남은 유동성:", sorted(((round(L['p']), kst(b[L['i']]['t'])) for L in lv), key=lambda x: x[0]))
    stats, zev = run_zones(b, atr, show_from)
    print("OB/FVG 전체 개수:", stats, f"(봉 {len(b)}개)")
    for i, who, what in zev[-12:]:
        print(f"{kst(b[i]['t'])} {who} {what}")
    for i, who, what in run_structure(b, show_from):
        print(f"{kst(b[i]['t'])} {who} {what}")
    print("마지막 구조:", b[-1]["dir"], b[-1]["flip"] and (round(b[-1]["flip"][0]), kst(b[b[-1]['flip'][1]]['t'])))


if __name__ == "__main__":
    since = sys.argv[2] if len(sys.argv) > 2 else "2026-10-01 00:00"
    for iv in sys.argv[1].split(","):
        main(iv, since)
