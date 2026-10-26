"""구조물 지도 '대응' 백테스트 — 강의(쉽알남 비밀 생존 전략 1~5장)의 진입·손절·익절을 기계 규칙으로 옮겨 잰다.

구조물 판정(추세선·전환 선·채널·유동성·오더블럭·FVG)은 structure_ref.py(= structure_map.pine)와 같은 규칙이다.
판단은 봉 마감 뒤에만 하고 주문은 다음 봉부터 체결한다. 목표 구조물은 체결 직전 봉까지 보인 것만 쓴다.
같은 봉에서 손절과 목표가 함께 닿으면 손절로 본다. 지정가가 체결된 봉에서는 손절만 보고 목표는 다음 봉부터 본다.

셋업(괄호는 강의 근거)
  A-TL/A-CH/A-LQ/A-MA  거짓 돌파 확인 매매(5장 전략 A): 추세선·채널 반대선·고점저점 유동성·상위 이동평균에서
                       Fake out·Trap이 나면 반대로. 손절 = 거짓 돌파 극점 바로 밖. 진입 = 복귀 뒤 그 선 리테스트(정석) 또는 다음 봉 시가.
  B-OB   오더블럭(1장): 구간에 닿으면 진입, 손절 = 만든 봉들 꼬리 끝, 1차 목표 = 그 파동 고점 탈환.
  C-FVG  FVG(2장): 구간에 닿으면 진입(되돌림 파동 안에서만), 손절 = 세 봉 꼬리 끝, 1차 목표 = 그 파동 끝.
  D-RT   추세선 돌파 뒤 리테스트(3장 전략 2): 손절 = 돌파 파동 시작점(직전 20봉 극점).
  D1-BN  추세선 터치 반등(3장 전략 1): 손절 = 선 0.5 ATR 밖.
  E-CH4  채널 4포인트(4장 전략 1): 반대선 첫 닿음에 진입, 채널 밖 봉마감이면 손절(비상 손절 1 ATR), 반대편 닿으면 익절.
익절 기본(5장): 1차 목표(가장 가까운 반대 구조물)에서 절반, 손절을 본전으로, 나머지는 다음 구조물.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from array import array
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import structure_ref as sr  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
HIST = ROOT / "cache" / "hist"
MS = {"5m": 300_000, "40m": 2_400_000, "1h": 3_600_000, "2h": 7_200_000, "160m": 9_600_000, "4h": 14_400_000, "12h": 43_200_000, "1d": 86_400_000}
CODES = ["LQ", "OB", "FVG", "TL", "CH", "12H20", "D20", "D200E", "D200S", "FLIP12"]
CODE = {k: i for i, k in enumerate(CODES)}
MAS = {"h12_20": CODE["12H20"], "d20": CODE["D20"], "d200e": CODE["D200E"], "d200s": CODE["D200S"]}
MAXZ = 3            # 오더블럭·FVG 종류·방향별 최대 개수 (Pine maxZ)
RET_BARS = 10       # 거짓 돌파 뒤 리테스트 지정가 유효 봉 수
RT_BARS = 20        # 추세선 돌파 확정 뒤 리테스트 지정가 유효 봉 수
SWING_BARS = 20     # 돌파 파동 시작점 = 직전 이 봉 수 극점
REARM_ATR = 2.0     # 추세선 터치 뒤 다시 노리려면 선에서 이만큼 멀어져야 함
TP_GAP = 0.25       # 목표 구조물 최소 간격 (ATR 배수)
CONF_BAND = 0.6     # 겹침 반폭 (ATR 배수, Pine confK)
MAXH = 120          # 최대 보유 봉 수
MIN_RISK = 0.0005   # 손절 폭 하한(가격 비율) — 이보다 좁으면 거래하지 않음


def load(sym, iv):
    return [dict(t=r[0], o=r[1], h=r[2], l=r[3], c=r[4], v=(r[5] if len(r) > 5 else 0.0)) for r in json.loads((HIST / f"{sym}_{iv}.json").read_text())]


def htf_values(b, tf, sym):
    """차트 봉마다 '그 봉이 끝날 때까지 완성된' 상위 봉 값(12H·일봉 이동평균, 12H 구조 방향·전환 유동성).
    미국 주식처럼 12시간봉이 없으면(정규장 6.5시간이라 12시간봉 = 일봉) 구조는 일봉으로 보고 12H 20SMA는 뺀다."""
    has12 = (HIST / f"{sym}_12h.json").exists()
    h12, d1 = load(sym, "12h" if has12 else "1d"), load(sym, "1d")
    ms12 = MS["12h"] if has12 else MS["1d"]
    s12 = sr.sma([r["c"] for r in h12], 20)
    sr.run_structure(h12, 0)
    cd = [r["c"] for r in d1]
    d20, d200e, d200s = sr.sma(cd, 20), sr.ema(cd, 200), sr.sma(cd, 200)
    out = {k: [None] * len(b) for k in ("h12_20", "d20", "d200e", "d200s", "flip12", "dir12")}
    j12 = jd = -1
    for i, r in enumerate(b):
        end = r["t"] + MS[tf]
        while j12 + 1 < len(h12) and h12[j12 + 1]["t"] + ms12 <= end:
            j12 += 1
        while jd + 1 < len(d1) and d1[jd + 1]["t"] + MS["1d"] <= end:
            jd += 1
        if j12 >= 0:
            out["h12_20"][i] = s12[j12] if has12 else None
            out["dir12"][i] = h12[j12]["dir"]
            out["flip12"][i] = h12[j12]["flip"][0] if h12[j12]["flip"] else None
        if jd >= 0:
            out["d20"][i], out["d200e"][i], out["d200s"][i] = d20[jd], d200e[jd], d200s[jd]
    return out


@dataclass
class Zone:
    kind: int        # 1 오더블럭, 2 FVG
    d: int
    top: float
    bot: float
    stop: float
    i0: int
    ic: int          # 생긴 봉
    star: bool
    touched: bool = False
    wave: float | None = None
    end_i: int | None = None


@dataclass
class Setup:
    kind: str
    i: int                    # 신호 봉(주문을 거는 봉)
    d: int                    # +1 매수, -1 매도
    lim: object               # None = 다음 봉 시가, 숫자 = 지정가, 함수 j -> 지정가
    stop: float | None        # 손절 기준 가격(버퍼 전). None이면 체결 때 선으로 정함(D1·E)
    start: int
    end: int | None           # 주문 유효 마지막 봉. None이면 owner.end_i(없으면 마지막 봉)
    code: int                 # 자기 구조물 종류(겹침 계산에서 뺌)
    ref: float                # 겹침 계산 기준 가격
    star: bool = False
    wave_tp: bool = False     # 1차 목표 = 생긴 뒤 파동 극점(오더블럭·FVG)
    line: object = None       # 추세선 객체(D1·E)
    owner: object = None      # 주문 수명 객체(end_i)
    conf: int = 0
    dir12: int = 0


class Engine:
    def __init__(self, sym, tf):
        self.sym, self.tf = sym, tf
        self.b = load(sym, tf)
        self.n = len(self.b)
        self.atr = sr.atr_series(self.b)
        self.htf = htf_values(self.b, tf, sym)
        self.setups: list[Setup] = []
        self.snap = [None] * self.n
        self.tl_events = []          # (i, 이름, 사건) — structure_ref 대조용(주 추세선·전환 선)
        self.breaks = []             # 처음 종가로 넘은 돌파와 그 결말(진짜·가짜) — breaks.py 연구용
        self._run()

    # ── 겹침: 기준 가격 ± 0.6 ATR 안에 있는 다른 구조물 종류 수 ──
    def conf_at(self, i, price, own):
        los, his, codes = self.snap[i]
        band = CONF_BAND * self.atr[i]
        seen = set()
        for lo, hi, c in zip(los, his, codes):
            if c != own and lo - band <= price <= hi + band:
                seen.add(c)
        return len(seen)

    def _add(self, s: Setup):
        self.setups.append(s)

    def _run(self):
        b, n, atr, H = self.b, self.n, self.atr, self.htf
        piv = {-1: [], 1: []}
        active = {-1: None, 1: None}
        flipped = {-1: None, 1: None}
        lv, zones = [], []
        maw = {k: None for k in MAS}
        pend = []                    # 이번 봉 끝에 겹침을 계산할 셋업
        for i in range(n):
            r, a = b[i], atr[i]
            # 0) 돌파 확정된 선: 리테스트·꼬리 거짓 돌파만 보고, 반대로 마감하면 돌파 실패
            for d in (-1, 1):
                t = flipped[d]
                if t is None:
                    continue
                if abs(r["c"] - t.val(i)) > sr.STALE * a:
                    t.end_i = i
                    flipped[d] = None
                    continue
                ev = t.wm.step(r, t.val(i), a, sr.TOL_B, sr.TOL_T, i)
                if ev == 8:
                    self.tl_events.append((i, d, "전환 리테스트"))
                elif ev == 5:
                    self.tl_events.append((i, d, "전환 꼬리"))
                elif ev:
                    self.tl_events.append((i, d, "돌파 실패"))
                    t.end_i = i
                    flipped[d] = None
            # 1) 활성 추세선
            for d in (-1, 1):
                t = active[d]
                if t is None:
                    continue
                if not t.est:
                    beyond = (r["c"] - t.val(i)) * (1 if d < 0 else -1)
                    if beyond > sr.DISCARD * a:
                        t.end_i = i
                        active[d] = None
                    elif i - t.ib >= sr.PROV:
                        t.est = True
                        # 3장 전략 1(터치 반등)과 4장 전략 1(4포인트)은 확립된 선에서만 노린다
                        self._add(Setup("D1-BN", i, t.d, (lambda j, t=t: t.val(j)), None, i + 1, None, CODE["TL"], t.val(i), line=t, owner=t))
                        t.armed = True
                        self._add(Setup("E-CH4", i, -t.d, (lambda j, t=t: t.sub(j)), None, i + 1, None, CODE["CH"], t.sub(i), line=t, owner=t))
                        pend += self.setups[-2:]
                    continue
                s0, pb = t.wm.s, t.wm.prior
                ev = t.wm.step(r, t.val(i), a, sr.TOL_B, sr.TOL_T, i)
                self._brk(t.wm, ev, i, "TL", t.val(i), s0, pb)
                if ev:
                    self.tl_events.append((i, d, sr.EV_NAME[ev]))
                if ev in (3, 4, 5):
                    ext = s0 * t.wm.ext
                    self._fake("A-TL", i, -s0, ext, (lambda j, t=t: t.val(j)), CODE["TL"], t.val(i), pend)
                if ev in (6, 7):
                    t.broken = True
                    t.end_act = i
                    if flipped[d] is not None:
                        flipped[d].end_i = i
                    flipped[d] = t
                    active[d] = None
                    t.end_i = None
                    dd = -t.d          # 하락선(저항)을 위로 뚫으면 매수
                    sw = min(x["l"] for x in b[max(0, i - SWING_BARS):i + 1]) if dd > 0 else max(x["h"] for x in b[max(0, i - SWING_BARS):i + 1])
                    s = Setup("D-RT", i, dd, (lambda j, t=t: t.val(j)), sw, i + 1, i + RT_BARS, CODE["TL"], t.val(i), owner=t)
                    self._add(s)
                    pend.append(s)
                    t.armed = False
                    continue
                # 추세선 터치 반등: 닿으면 그 셋업은 끝, 선에서 2 ATR 멀어지면 다시 노린다
                lvl = t.val(i)
                touch = r["l"] <= lvl if t.d > 0 else r["h"] >= lvl
                if getattr(t, "armed", False) and touch:
                    t.armed = False
                elif not getattr(t, "armed", False) and abs(r["c"] - lvl) >= REARM_ATR * a:
                    t.armed = True
                    s = Setup("D1-BN", i, t.d, (lambda j, t=t: t.val(j)), None, i + 1, None, CODE["TL"], lvl, line=t, owner=t)
                    self._add(s)
                    pend.append(s)
                # 채널 (Pine과 같게: 4포인트 뒤에만 반대선 판정, 이탈 확정 뒤 멈춤)
                if not getattr(t, "subBrk", False):
                    sv = t.sub(i)
                    tsub = r["l"] <= sv if d < 0 else r["h"] >= sv
                    if not t.four and tsub:
                        t.four = True
                    if t.four:
                        s1, pb = t.ws.s, t.ws.prior
                        ev2 = t.ws.step(r, sv, a, sr.TOL_B, sr.TOL_T, i)
                        self._brk(t.ws, ev2, i, "CH", sv, s1, pb)
                        if ev2 in (3, 4, 5):
                            self._fake("A-CH", i, -s1, s1 * t.ws.ext, (lambda j, t=t: t.sub(j)), CODE["CH"], sv, pend)
                        if ev2 in (6, 7):
                            t.subBrk = True
            # 2) 유동성 (좌우 10봉 고점·저점)
            keep = []
            for L in lv:
                s0, pb = L["w"].s, L["w"].prior
                ev = L["w"].step(r, L["p"], a, 0.0, 0.0, i)
                self._brk(L["w"], ev, i, "LQ", L["p"], s0, pb)
                if ev in (3, 4, 5):
                    self._fake("A-LQ", i, -s0, s0 * L["w"].ext, L["p"], CODE["LQ"], L["p"], pend)
                elif ev in (6, 7):
                    pass
                else:
                    keep.append(L)
            lv = keep
            # 3) 상위 이동평균(12H 20SMA·일봉 20SMA·200EMA·200SMA)
            for k, code in MAS.items():
                v = H[k][i]
                if v is None:
                    continue
                if maw[k] is None:
                    maw[k] = sr.Watch(s=-1 if r["c"] > v else 1)
                    continue
                s0, pb = maw[k].s, maw[k].prior
                ev = maw[k].step(r, v, a, sr.TOL_B, sr.TOL_T, i)
                self._brk(maw[k], ev, i, "MA", v, s0, pb)
                if ev in (3, 4, 5):
                    self._fake("A-MA", i, -s0, s0 * maw[k].ext, (lambda j, k=k: H[k][j - 1]), code, v, pend)
            # 4) 오더블럭·FVG 수명 (Pine과 같게: 종가 무효, 첫 닿음, 파동 끝 넘으면 정리)
            wph = b[i - 3]["h"] if i >= 3 and sr.pivot_at(b, i - 3, 3, True) else None
            wpl = b[i - 3]["l"] if i >= 3 and sr.pivot_at(b, i - 3, 3, False) else None
            alive = []
            for z in zones:
                bull = z.d > 0
                dead = r["c"] < z.stop if bull else r["c"] > z.stop
                if not dead and not z.touched and i > z.i0 + 1 and (r["l"] <= z.top if bull else r["h"] >= z.bot):
                    z.touched = True
                if z.wave is None and i - 3 > z.i0:
                    if bull and wph is not None:
                        z.wave = wph
                    elif not bull and wpl is not None:
                        z.wave = wpl
                done = z.wave is not None and (r["c"] > z.wave if bull else r["c"] < z.wave) and (z.kind == 2 or z.touched)
                if dead or done or i - z.i0 > 500:
                    z.end_i = i
                else:
                    alive.append(z)
            zones = alive
            if i >= 3:
                r1, r2 = b[i - 1], b[i - 2]
                body = lambda x: abs(x["c"] - x["o"])
                lowp = min(x["l"] for x in b[max(0, i - 22):i - 2])
                highp = max(x["h"] for x in b[max(0, i - 22):i - 2])
                new = []
                ob_bull = r1["c"] < r1["o"] and r["c"] > r["o"] and body(r1) >= sr.DOJI_ATR * a and body(r) >= sr.OB_RATIO * body(r1) and r["c"] > r1["o"]
                ob_bear = r1["c"] > r1["o"] and r["c"] < r["o"] and body(r1) >= sr.DOJI_ATR * a and body(r) >= sr.OB_RATIO * body(r1) and r["c"] < r1["o"]
                if ob_bull:
                    st = min(r["l"], r1["l"], r2["l"] if r2["c"] > r2["o"] else r1["l"])
                    new.append(Zone(1, 1, r1["o"], r1["c"], st, i - 1, i, st < lowp))
                if ob_bear:
                    st = max(r["h"], r1["h"], r2["h"] if r2["c"] < r2["o"] else r1["h"])
                    new.append(Zone(1, -1, r1["c"], r1["o"], st, i - 1, i, st > highp))
                prev_ob_bull = getattr(self, "_ob_bull_prev", False)
                prev_ob_bear = getattr(self, "_ob_bear_prev", False)
                if r["l"] > r2["h"] and r1["c"] > r1["o"] and body(r1) >= sr.FVG_RATIO * body(r) and body(r1) >= sr.FVG_RATIO * body(r2):
                    new.append(Zone(2, 1, r["l"], r2["h"], min(r["l"], r1["l"], r2["l"]), i - 2, i, min(r1["l"], r2["l"]) < lowp or prev_ob_bull))
                if r["h"] < r2["l"] and r1["c"] < r1["o"] and body(r1) >= sr.FVG_RATIO * body(r) and body(r1) >= sr.FVG_RATIO * body(r2):
                    new.append(Zone(2, -1, r2["l"], r["h"], max(r["h"], r1["h"], r2["h"]), i - 2, i, max(r1["h"], r2["h"]) > highp or prev_ob_bear))
                self._ob_bull_prev, self._ob_bear_prev = ob_bull, ob_bear
                for z in new:
                    same = [q for q in zones if q.kind == z.kind and q.d == z.d]
                    if len(same) >= MAXZ:
                        old = same[0]
                        old.end_i = i
                        zones.remove(old)
                    zones.append(z)
                    edge = z.top if z.d > 0 else z.bot
                    s = Setup("B-OB" if z.kind == 1 else "C-FVG", i, z.d, edge, z.stop, i + 1, None, CODE["OB"] if z.kind == 1 else CODE["FVG"], edge, star=z.star, wave_tp=True, owner=z)
                    self._add(s)
                    pend.append(s)
            # 5) 피벗 확정 → 새 추세선 / 다시 긋기 / 교체 (structure_ref.run_trendlines와 같은 순서)
            j = i - sr.PIV
            for d, hi in ((-1, True), (1, False)):
                if j >= 0 and sr.pivot_at(b, j, sr.PIV, hi):
                    p = b[j]["h"] if hi else b[j]["l"]
                    cur = active[d]
                    if cur is not None and i - cur.ia > sr.MAX_LOOK:
                        cur.end_i = cur.end_act = i
                        active[d] = cur = None
                    stale = cur is not None and cur.est and abs(r["c"] - cur.val(i)) > sr.STALE * a
                    if cur is None or stale or not cur.est:
                        t = sr.find_line(b, piv[d], j, p, d, i, a)
                        if t is not None and (cur is None or stale or t.ia <= cur.ia):
                            if cur is not None:
                                cur.end_i = cur.end_act = i
                            t.end_i = None
                            active[d] = t
                            self.tl_events.append((i, d, "새 선"))
                    piv[d].append((j, p))
                    piv[d] = [(x, q) for x, q in piv[d] if i - x <= sr.MAX_LOOK + sr.PIV][-40:]
            # 6) 유동성 새 피벗
            jl = i - sr.LIQ_PIV
            if jl >= 0:
                if sr.pivot_at(b, jl, sr.LIQ_PIV, True):
                    lv.append(dict(p=b[jl]["h"], i=jl, w=sr.Watch(s=1)))
                if sr.pivot_at(b, jl, sr.LIQ_PIV, False):
                    lv.append(dict(p=b[jl]["l"], i=jl, w=sr.Watch(s=-1)))
                if len(lv) > 40:
                    lv = lv[-40:]
            # 7) 이 봉 끝의 구조물 사진(목표·겹침용): (아래값, 위값, 종류)
            los, his, codes = array("d"), array("d"), bytearray()

            def put(lo, hi, c):
                los.append(lo)
                his.append(hi)
                codes.append(c)

            for L in lv:
                put(L["p"], L["p"], CODE["LQ"])
            for z in zones:
                put(z.bot, z.top, CODE["OB"] if z.kind == 1 else CODE["FVG"])
            for d in (-1, 1):
                t = active[d]
                if t is not None and t.est:
                    v = t.val(i)
                    put(v, v, CODE["TL"])
                    if not getattr(t, "subBrk", False):
                        sv = t.sub(i)
                        put(sv, sv, CODE["CH"])
                f = flipped[d]
                if f is not None:
                    v = f.val(i)
                    put(v, v, CODE["TL"])
            for k, code in MAS.items():
                if H[k][i] is not None:
                    put(H[k][i], H[k][i], code)
            if H["flip12"][i] is not None:
                put(H["flip12"][i], H["flip12"][i], CODE["FLIP12"])
            self.snap[i] = (los, his, bytes(codes))
            for s in pend:
                s.conf = self.conf_at(i, s.ref, s.code)
                s.dir12 = H["dir12"][i] or 0
            pend = []
        # 주문 수명 마감 처리: 끝나지 않은 객체는 마지막 봉까지
        for s in self.setups:
            if s.kind in ("D1-BN", "E-CH4"):
                e = getattr(s.line, "end_act", None)          # 선이 깨지거나 바뀌면 반등·4포인트 주문도 끝
                s.end = e if e is not None else n - 1
            elif s.kind == "D-RT":
                e = getattr(s.owner, "end_i", None)           # 전환 선이 지워지면 리테스트 주문도 끝
                if e is not None:
                    s.end = min(s.end, e)
            elif s.end is None:
                e = getattr(s.owner, "end_i", None) if s.owner is not None else None
                s.end = e if e is not None else n - 1

    def _brk(self, w, ev, i, kind, lv, s0, pb):
        """돌파 기록: 처음 종가로 넘은 봉(사건 1)을 남기고, 같은 판정기가 가짜(3 Fake out·4 Trap)나 진짜(6·7 확정)로 끝내면 결말을 적는다.
        직전 거짓 돌파 극점을 바로 넘은 재돌파(첫 마감에서 사건 7)는 rebreak로 따로 남긴다. 선이 먼저 지워지면 결말 없음."""
        rec = getattr(w, "rec", None)
        if ev == 1 or (ev == 7 and rec is None):
            rec = dict(kind=kind, i=i, s=s0, lv=lv, prior=not math.isnan(pb), rebreak=(ev == 7), out=("real" if ev == 7 else None), res_i=(i if ev == 7 else None))
            self.breaks.append(rec)
            w.rec = None if ev == 7 else rec
        elif rec is not None and ev in (3, 4, 6, 7):
            rec["out"] = "real" if ev in (6, 7) else ("fake" if ev == 3 else "trap")
            rec["res_i"] = i
            w.rec = None

    def _fake(self, kind, i, d, ext, level, code, ref, pend):
        """거짓 돌파 셋업 두 가지: 리테스트 지정가(정석)와 다음 봉 시가(봉마감 확인 진입)."""
        lim = level
        s1 = Setup(kind, i, d, lim, ext, i + 1, i + RET_BARS, code, ref)
        s2 = Setup(kind + ":mkt", i, d, None, ext, i + 1, i + 1, code, ref)
        self._add(s1)
        self._add(s2)
        pend += [s1, s2]


# ── 모의 체결 ────────────────────────────────────────────
@dataclass
class Var:
    buf: float = 0.1          # 손절 버퍼 (ATR 배수)
    exit: str = "half"        # half: 1차 절반+본전+2차 / full: 1차 전량 / r2: 2R 전량
    minrr: float = 0.0        # 1차 목표 손익비 하한(못 미치면 거래 안 함)
    stopmode: str = "touch"   # touch: 닿으면 손절 / close: 손절 기준 밖에서 마감하면 그 종가에 손절 (오더블럭·FVG 2번째 기준)
    nostop: bool = False      # 점검용: 손절 없이 목표나 시간(120봉)으로만 청산 — '대응'을 뺐을 때와 비교
    ch_close: bool = True     # 채널 4포인트: 반대선 밖 봉마감이면 손절. False면 지표처럼 비상 손절선(1 ATR 밖)만


def targets(eng, s, fj, px, ru):
    """체결 직전 봉(fj-1)까지 보인 구조물로 1·2차 목표를 정한다."""
    a = eng.atr[fj - 1]
    los, his, _ = eng.snap[fj - 1]
    if s.d > 0:
        cands = sorted(lo for lo in los if lo > px + TP_GAP * a)
    else:
        cands = sorted((hi for hi in his if hi < px - TP_GAP * a), reverse=True)
    tp1 = None
    if s.wave_tp:
        seg = eng.b[s.i:fj]
        w = max(x["h"] for x in seg) if s.d > 0 else min(x["l"] for x in seg)
        if (w - px) * s.d > TP_GAP * a:
            tp1 = w
    fallback = tp1 is None and not cands
    if tp1 is None:
        tp1 = cands[0] if cands else px + s.d * 2 * ru
    nxt = [c for c in cands if (c - tp1) * s.d > TP_GAP * a]
    tp2 = nxt[0] if nxt else tp1 + (tp1 - px)
    return tp1, tp2, fallback


def fill(eng, s: Setup):
    """체결 봉과 가격. 손절·목표 설정과 무관하므로 셋업마다 한 번만 찾는다."""
    b, n, d = eng.b, eng.n, s.d
    if s.lim is None:
        fj = s.i + 1
        return (fj, b[fj]["o"], True) if fj < n else None
    for j in range(s.start, min(s.end, n - 1) + 1):
        L = s.lim(j) if callable(s.lim) else s.lim
        if L is None:
            continue
        r = b[j]
        if d > 0 and r["l"] <= L:
            return j, min(r["o"], L), False
        if d < 0 and r["h"] >= L:
            return j, max(r["o"], L), False
    return None


class Sub:
    """봉 안 순서를 가리는 하위 봉(코인 5분봉, 미국 주식 1시간봉). 차트 봉 [시작, 시작+길이) 안의 하위 봉 범위를 돌려준다."""

    def __init__(self, sym, iv="5m"):
        rows = json.loads((HIST / f"{sym}_{iv}.json").read_text())
        self.t = [r[0] for r in rows]
        self.o = [r[1] for r in rows]
        self.h = [r[2] for r in rows]
        self.l = [r[3] for r in rows]
        self.c = [r[4] for r in rows]

    def range(self, t0, t1):
        from bisect import bisect_left
        return bisect_left(self.t, t0), bisect_left(self.t, t1)


def tv3(o, h, l, c):
    """트레이딩뷰 전략 테스터 가정의 봉 안 경로: 시가가 고가에 더 가까우면 시가→고가→저가→종가, 아니면 시가→저가→고가→종가.
    한 방향으로만 움직이는 세 구간 [(시작가, 고가, 저가, 방향)]으로 돌려준다."""
    pts = (o, h, l, c) if (h - o) < (o - l) else (o, l, h, c)
    return [(pts[k], max(pts[k], pts[k + 1]), min(pts[k], pts[k + 1]), 1 if pts[k + 1] > pts[k] else -1) for k in range(3)]


def simulate(eng, s: Setup, v: Var, fee: float, f=None, sub: Sub | None = None, path: str = "stop"):
    """sub(하위 봉)를 주면 손절·목표가 닿은 봉과 지정가 체결 봉을 하위 봉 순서로 따라간다. 하위 봉 안에서 둘 다 닿으면 손절로 본다.
    하위 봉이 없는 봉은 path="stop"이면 손절 먼저, path="tv"면 트레이딩뷰 전략 테스터 가정(시가에 가까운 극값 먼저)."""
    b, n, atr = eng.b, eng.n, eng.atr
    d = s.d
    f = f if f is not None else fill(eng, s)
    if f is None:
        return None
    fj, px, mkt = f
    a = atr[fj - 1]
    # 손절·목표
    if s.kind == "D1-BN":
        stp = s.line.val(fj) - d * (0.5 + v.buf) * a
    elif s.kind == "E-CH4":
        stp = s.line.sub(fj) - d * (1.0 + v.buf) * a
    else:
        stp = s.stop - d * v.buf * a
    ru = (px - stp) * d
    if ru <= px * MIN_RISK:
        return None
    if s.kind == "E-CH4":
        t1f = (lambda j: (s.line.val(j) + s.line.sub(j)) / 2) if v.exit == "half" else (lambda j: s.line.val(j))
        t2f = (lambda j: s.line.val(j)) if v.exit == "half" else None
        fallback = False
        if v.exit == "r2":
            t1f, t2f = (lambda j: px + d * 2 * ru), None
        tp1_now = t1f(fj)
    else:
        tp1, tp2, fallback = targets(eng, s, fj, px, ru)
        if v.exit == "r2":
            tp1, tp2 = px + d * 2 * ru, None
        elif v.exit == "full":
            tp2 = None
        t1f = lambda j, x=tp1: x
        t2f = (lambda j, x=tp2: x) if tp2 is not None else None
        tp1_now = tp1
    rr1 = (tp1_now - px) * d / ru
    if rr1 <= 0 or rr1 < v.minrr:
        return None
    close_stop = v.stopmode == "close" and s.kind in ("B-OB", "C-FVG")
    close_lvl = ((lambda j: s.line.sub(j)) if v.ch_close else None) if s.kind == "E-CH4" else ((lambda j, x=stp: x) if close_stop else None)
    if v.nostop:
        close_lvl = None
    st = {"pos": 1.0, "got": 0.0, "stop": stp, "half": False, "why": "time", "xj": None}

    def events(o, h, l, j, allow_tp, mono=0):
        """한 봉(또는 하위 봉·구간) 안의 손절·목표. 끝났으면 True.
        mono: 한 방향으로만 움직이는 구간이면 +1(오름)·-1(내림). 유리한 쪽으로만 가는 구간에서는 1차 익절 뒤 본전 복귀가 없다."""
        if (not close_stop or st["half"]) and not v.nostop:
            sn = st["stop"]
            if (d > 0 and l <= sn) or (d < 0 and h >= sn):
                xp = min(o, sn) if d > 0 else max(o, sn)
                st["got"] += st["pos"] * (xp - px) * d / ru
                st.update(pos=0.0, why="be" if st["half"] else "stop", xj=j)
                return True
        if allow_tp:
            if not st["half"]:
                t1 = t1f(j)
                if (d > 0 and h >= t1) or (d < 0 and l <= t1):
                    xp = max(o, t1) if d > 0 else min(o, t1)
                    if t2f is not None:
                        st["got"] += 0.5 * (xp - px) * d / ru
                        st.update(pos=0.5, half=True, stop=px)
                        if mono != d and ((d > 0 and l <= px) or (d < 0 and h >= px)):
                            st.update(pos=0.0, why="tp1+be", xj=j)      # 같은 봉에서 본전 복귀는 손해 쪽으로
                            return True
                    else:
                        st["got"] += st["pos"] * (xp - px) * d / ru
                        st.update(pos=0.0, why="tp1", xj=j)
                        return True
            if st["half"] and t2f is not None:
                t2 = t2f(j)
                if (d > 0 and h >= t2) or (d < 0 and l <= t2):
                    xp = max(o, t2) if d > 0 else min(o, t2)
                    st["got"] += st["pos"] * (xp - px) * d / ru
                    st.update(pos=0.0, why="tp2", xj=j)
                    return True
        return False

    def touches(r, j):
        if (d > 0 and r["l"] <= st["stop"]) or (d < 0 and r["h"] >= st["stop"]):
            return True
        t = t1f(j) if not st["half"] else (t2f(j) if t2f is not None else None)
        return t is not None and ((d > 0 and r["h"] >= t) or (d < 0 and r["l"] <= t))

    last = min(n - 1, fj + MAXH)
    ms = MS[eng.tf]
    for j in range(fj, last + 1):
        r = b[j]
        lim_bar = j == fj and not mkt
        segs = None                                  # 봉 안 순서: [(시작가, 고가, 저가), ...]
        if sub is not None:
            k0, k1 = sub.range(r["t"], r["t"] + ms)
            if k1 > k0:
                if j == fj or touches(r, j):
                    segs = []
                    for k in range(k0, k1):
                        segs += tv3(sub.o[k], sub.h[k], sub.l[k], sub.c[k]) if path == "tv" else [(sub.o[k], sub.h[k], sub.l[k], 0)]
                else:
                    segs = []
        if segs is None and path == "tv":
            segs = tv3(r["o"], r["h"], r["l"], r["c"])
        if segs is not None:
            if lim_bar:
                L = s.lim(j) if callable(s.lim) else s.lim
                ld = getattr(s, "lim_dir", d)       # 점검용 반대 방향 셋업은 원래 셋업과 같은 하위 봉에서 체결
                kf = next((k for k, g in enumerate(segs) if (ld > 0 and g[2] <= L) or (ld < 0 and g[1] >= L)), 0)
                if segs and events(segs[kf][0], segs[kf][1], segs[kf][2], j, False, segs[kf][3]):      # 체결된 구간에서는 손절만
                    break
                segs = segs[kf + 1:]
            if any(events(o_, h_, l_, j, True, m_) for o_, h_, l_, m_ in segs):
                break
        elif events(r["o"], r["h"], r["l"], j, not lim_bar):
            break
        # 봉마감 손절(오더블럭·FVG 2번째 기준, 채널 밖 마감)
        if close_lvl is not None and not st["half"]:
            cl = close_lvl(j)
            if (d > 0 and r["c"] < cl) or (d < 0 and r["c"] > cl):
                st["got"] += st["pos"] * (r["c"] - px) * d / ru
                st.update(pos=0.0, why="close", xj=j)
                break
    if st["pos"] > 0:
        st["got"] += st["pos"] * (b[last]["c"] - px) * d / ru
        st["xj"] = last
    got, xj, why = st["got"], st["xj"], st["why"]
    cost = 2 * fee * px / ru
    return dict(kind=s.kind, sym=eng.sym, tf=eng.tf, i=s.i, t=b[s.i]["t"], d=d, R=got - cost, Rg=got, cost=cost,
                rr1=rr1, conf=s.conf, star=s.star, dir12=s.dir12, why=why, hold=xj - fj, fb=fallback, open=(xj == n - 1 and why == "time"),
                t_fill=b[fj]["t"], t_exit=b[xj]["t"])
