"""BRIEF3 L1 소진 묶음 — 전략·격자·고르기(sel 2026-01-02~06-30만, 잠금 켜지 않음).
사용: py lay_L1.py → results/lay_L1_sel.csv, results/lay_L1_frozen.json

신호(BRIEF3 H0에서 출발, 격자 실행 전에 정함):
- 종목 s(SOXL·SOXS)마다 기준 B_s를 잰다.
  Lmin : 09:30 봉(또는 앞 매매 청산 다음 봉)부터 지금 봉까지 s 저가의 최저(H0 그대로).
  low15/low30 : 최근 15/30봉(지금 봉 포함) 저가의 최저. 창은 09:30 봉·청산 다음 봉 앞으로 넘어가지 않는다.
  open : 오늘 09:30 봉 시가.  pclose : 전날 정규장 종가(ctx.prev_day, 반일장 13:00 반영).
- 조건: s 1분 종가 ≥ B_s × (1 + x%) 그리고 (켠 것만)
  speed N : Lmin에서만. 지금 최저를 만든 마지막 봉부터 신호 봉까지 분(i − j_L) ≤ N('오름이 N분 안').
  vol     : 신호 봉 s 거래량 ÷ 직전 정규장 20봉 평균(날 넘어 이음, 빈 분 0). lt1 <1, lt2 <2, ge2 ≥2, ge4 ≥4.
  rsi R   : s 1분 Wilder RSI14(정규장 체결 봉만 날 넘어 이음, 연장 끔, 전전날부터 씨앗) ≥ R.
  vwap W  : s 종가 ≥ 세션 VWAP(09:30부터, 봉 VWAP × 거래량) × (1 + W%).
- 조건이 맞으면 반대 종목을 산다(엔진: 다음 봉 종가 + 반 호가, 1분 지연). 목표 +1% 지정가(관통 1.0002),
  손절 없음, 못 닿으면 창 끝(엔진: w90 10:59, std 09:59 봉 종가 − 반 호가). H0 청산 그대로.
- 한 번에 하나: 드는 동안 신호 무시. 청산 뒤 두 종목의 Lmin·low창을 청산 다음 봉부터 다시 잰다.
- open·pclose 기준은 오른 종목마다 하루 한 번만 진입한다(드는 동안 맞은 봉은 쓰지 않은 것으로 본다).
- 진입 마감: 결정 봉 마감 ≤ 창 끝 15분 전(w90 10:45, std 09:45) = 봉 i ≤ cut − 16.
- 같은 봉에 두 종목이 다 맞으면 (종가 ÷ 기준)이 큰 쪽(드묾).
- 거울(mirror=True): 같은 신호에 오른 종목을 산다(SOXL↔SOXS 뒤집기 = 추종).
전략은 ctx로만 자료를 본다. 지표 캐시는 (날, 종목)별로 봉 j 값을 봉 j까지의 ctx 자료로만 채운다(설정끼리 공유).

고르기(BRIEF3): 점수 = sel 하루 평균 t(전체 sel 날, 활동일 20일 이상만 후보). 최종 점수 = 자기 + 격자 이웃 t의 중앙값.
이웃 = 순서 있는 축(x·speed·vol·rsi·vwap) 하나만 다르고 나머지가 같은 설정 중 그 축에서 바로 아래·위 값(격자에 있는 것).
기준(base)은 범주라 이웃이 아니다. w90 1위·std 1위(같으면 1개)를 얼린다.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import (Ctx, Data, Engine, Order, RTH0, RTH1, ROOT, TOUCH, bucket, hm, idx,  # noqa: E402
                 load_costs, summarize)

S2 = ("SOXL", "SOXS")
SIGN = {"SOXL": 1, "SOXS": -1}
RSI_N, VR_N, TARGET = 14, 20, 0.01
KEYS = ["base", "x", "speed", "vol", "rsi", "vwap"]
DEFAULTS = dict(base="Lmin", x=3.0, speed=None, vol=None, rsi=None, vwap=None)
BASES = ["Lmin", "low15", "low30", "open", "pclose"]
AX = {"x": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
      "speed": [None, 30, 15, 10, 5],                 # 느슨 → 빡빡
      "vol": ["lt1", "lt2", None, "ge2", "ge4"],      # 조용한 오름만 ← 제한 없음 → 거래량 실린 오름만
      "rsi": [None, 60, 70, 80],
      "vwap": [None, 1.0, 2.0, 3.0, 4.0]}
WINB = {"low15": 15, "low30": 30}
FIXED = ("open", "pclose")


# ------------------------------------------------------------------ 지표(봉 j 값은 봉 j까지의 자료로만)

def _rth_end(v_soxl):
    """lib.Data와 같은 반일장 규칙(SOXL 13:00~15:59 거래량 < 09:30~12:59의 15%면 13:00 마감)."""
    return idx("13:00") if v_soxl[idx("13:00"):RTH1].sum() < 0.15 * v_soxl[RTH0:idx("13:00")].sum() else RTH1


class Feat:
    """(날, 종목) 지표: 세션 VWAP, 거래량 배수, RSI14. upto(ctx, i)가 봉 i까지 채운다."""

    def __init__(self, ctx, sym):
        self.sym = sym
        seqs, tail = [], None
        for k in (2, 1):
            c, v = ctx.prev_bars(sym, "Close", k), ctx.prev_bars(sym, "Volume", k)
            e = _rth_end(ctx.prev_bars("SOXL", "Volume", k))
            m = v[RTH0:e] > 0
            seqs.append(c[RTH0:e][m])
            if k == 1:
                tail = v[RTH0:e][-VR_N:].astype(float)
        s = np.concatenate(seqs)
        d = np.diff(s)
        g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
        ag, al = g[:RSI_N].mean(), l[:RSI_N].mean()
        a = 1.0 / RSI_N
        for x_, y_ in zip(g[RSI_N:], l[RSI_N:]):
            ag += a * (x_ - ag)
            al += a * (y_ - al)
        self.ag, self.al, self.last = float(ag), float(al), float(s[-1])
        self.tail = tail
        self.cpv = self.cv = 0.0
        self.k = RTH0 - 1
        self.vwap = np.full(960, np.nan)
        self.vr = np.full(960, np.nan)
        self.rsi = np.full(960, np.nan)

    def upto(self, ctx, i):
        if i <= self.k:
            return
        c, v, p = ctx.c(self.sym), ctx.bars(self.sym, "Volume"), ctx.bars(self.sym, "VWAP")
        a = 1.0 / RSI_N
        for j in range(self.k + 1, i + 1):
            vj = float(v[j])
            self.cpv += float(p[j]) * vj
            self.cv += vj
            self.vwap[j] = self.cpv / self.cv if self.cv > 0 else np.nan
            n_today = j - RTH0
            if n_today >= VR_N:
                prev = v[j - VR_N:j]
            else:
                prev = np.concatenate([self.tail[len(self.tail) - (VR_N - n_today):], v[RTH0:j]])
            mu = float(np.mean(prev))
            self.vr[j] = vj / mu if mu > 0 else np.nan
            if vj > 0:
                dl = float(c[j]) - self.last
                self.ag += a * (max(dl, 0.0) - self.ag)
                self.al += a * (max(-dl, 0.0) - self.al)
                self.last = float(c[j])
            self.rsi[j] = 100.0 if self.al == 0 else 100.0 - 100.0 / (1.0 + self.ag / self.al)
        self.k = i


_FEAT, _PCLOSE = {}, {}


def feat(ctx, sym, i):
    k = (ctx.day, sym)
    f = _FEAT.get(k)
    if f is None:
        f = _FEAT[k] = Feat(ctx, sym)
    f.upto(ctx, i)
    return f


def pclose(ctx, sym):
    k = (ctx.day, sym)
    v = _PCLOSE.get(k)
    if v is None:
        v = _PCLOSE[k] = float(ctx.prev_day(sym)["close"])
    return v


def vol_ok(band, r):
    if band is None:
        return True
    if not np.isfinite(r):
        return False
    return {"lt1": r < 1, "lt2": r < 2, "ge2": r >= 2, "ge4": r >= 4}[band]


# ------------------------------------------------------------------ 전략

class Exhaust:
    def __init__(self, p, mirror=False):
        self.p, self.flip = p, (-1 if mirror else 1)
        self.thr = 1.0 + p["x"] / 100.0
        self.win = WINB.get(p["base"])
        self.need_feat = p["vol"] is not None or p["rsi"] is not None or p["vwap"] is not None
        self.st = "flat"
        self.r0 = RTH0
        self.L = {s: np.inf for s in S2}
        self.jL = {s: -1 for s in S2}
        self.used = set()

    def _reset(self, i):
        self.r0 = i + 1
        for s in S2:
            self.L[s], self.jL[s] = np.inf, -1

    def on_bar(self, ctx, i):
        p = self.p
        if self.st == "entering":                    # 체결 봉(1분 지연)
            self.st = "in"
        if self.st == "in":
            if ctx.pos:
                return None
            self.st = "flat"                         # 이 봉에서 청산됨 → 다음 봉부터 기준 다시 잼
            self._reset(i)
            return None
        best = None
        late = i > ctx.cut - 16
        for s in S2:
            c = ctx.bars(s, "Close")[i]
            if p["base"] == "Lmin":
                lo = ctx.bars(s, "Low")[i]
                if lo <= self.L[s]:
                    self.L[s], self.jL[s] = lo, i
                B = self.L[s]
            elif self.win:
                B = ctx.bars(s, "Low")[max(i - self.win + 1, self.r0):i + 1].min()
            elif p["base"] == "open":
                B = ctx.bars(s, "Open")[RTH0]
            else:
                B = pclose(ctx, s)
            if late or c < B * self.thr:
                continue
            if p["base"] in FIXED and s in self.used:
                continue
            if p["speed"] is not None and i - self.jL[s] > p["speed"]:
                continue
            if self.need_feat:
                f = feat(ctx, s, i)
                if not vol_ok(p["vol"], f.vr[i]):
                    continue
                if p["rsi"] is not None and not f.rsi[i] >= p["rsi"]:
                    continue
                if p["vwap"] is not None and not c >= f.vwap[i] * (1 + p["vwap"] / 100):
                    continue
            sc = c / B
            if best is None or sc > best[0]:
                best = (sc, s)
        if best is None:
            return None
        s = best[1]
        if p["base"] in FIXED:
            self.used.add(s)
        self.st = "entering"
        return Order(-SIGN[s] * self.flip, target=TARGET)


def make(params, mirror=False):
    p = dict(DEFAULTS, **params)
    return lambda: Exhaust(p, mirror)


# ------------------------------------------------------------------ 격자

def grid():
    out = {}

    def add(**kw):
        p = dict(DEFAULTS, **kw)
        out.setdefault(tuple(p[k] for k in KEYS), p)
    X = AX["x"]
    # A 기준 × 문턱(× 속도, Lmin만): 30 + 4 × 6 = 54
    for x, sp in itertools.product(X, AX["speed"]):
        add(base="Lmin", x=x, speed=sp)
    for b, x in itertools.product(["low15", "low30", "open", "pclose"], X):
        add(base=b, x=x)
    # B Lmin(속도 제한 없음) × 문턱 6 × 거래량 5 × VWAP 5 = 150, A와 겹침 6 → 144
    for x, v, w in itertools.product(X, AX["vol"], AX["vwap"]):
        add(base="Lmin", x=x, vol=v, vwap=w)
    # C Lmin × 문턱 6 × RSI(60·70·80) × VWAP(없음·2·4) = 54
    for x, r, w in itertools.product(X, [60, 70, 80], [None, 2.0, 4.0]):
        add(base="Lmin", x=x, rsi=r, vwap=w)
    # D Lmin × 속도(30·15·10·5) × 문턱 6 × VWAP(2·4) = 48
    for sp, x, w in itertools.product([30, 15, 10, 5], X, [2.0, 4.0]):
        add(base="Lmin", x=x, speed=sp, vwap=w)
    # E 다른 기준 4 × 문턱 6 × 조건 하나(VWAP 2·VWAP 4·거래량 <2·RSI 70) = 96
    for b, x in itertools.product(["low15", "low30", "open", "pclose"], X):
        for kw in ({"vwap": 2.0}, {"vwap": 4.0}, {"vol": "lt2"}, {"rsi": 70}):
            add(base=b, x=x, **kw)
    return out


GRID_DOC = ("A 54 = Lmin x6 x speed5 + {low15,low30,open,pclose} x6; "
            "B 144 = Lmin x6 x vol{lt1,lt2,any,ge2,ge4} x vwap{any,1,2,3,4} (A와 겹침 6 뺌); "
            "C 54 = Lmin x6 x rsi{60,70,80} x vwap{any,2,4}; "
            "D 48 = Lmin x speed{30,15,10,5} x6 x vwap{2,4}; "
            "E 96 = {low15,low30,open,pclose} x6 x one of {vwap2, vwap4, vol<2, rsi70}")


def cid(p):
    f = lambda v: "N" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
    return f"L1-{p['base']}-x{p['x']:g}-s{f(p['speed'])}-v{f(p['vol'])}-r{f(p['rsi'])}-w{f(p['vwap'])}"


def neighbors(key, keys):
    """순서 있는 축 하나만 다른 설정 중 그 축 순서에서 바로 아래·위(격자에 있는 것)."""
    out = []
    for a, vals in AX.items():
        j = KEYS.index(a)
        k = vals.index(key[j])
        same = [q for q in keys if q[j] != key[j] and all(q[t] == key[t] for t in range(len(KEYS)) if t != j)]
        lo = [q for q in same if vals.index(q[j]) < k]
        hi = [q for q in same if vals.index(q[j]) > k]
        if lo:
            out.append(max(lo, key=lambda q: vals.index(q[j])))
        if hi:
            out.append(min(hi, key=lambda q: vals.index(q[j])))
    return out


def flat(s, tr, pre):
    row = {f"{pre}_{k}": v for k, v in s.items() if k != "ci95"}
    row[f"{pre}_ci_lo"], row[f"{pre}_ci_hi"] = s.get("ci95", [np.nan, np.nan])
    n = len(tr)
    row[f"{pre}_avg_trade_pct"] = float(tr["ret"].mean() * 100) if n else np.nan
    row[f"{pre}_target_rate"] = float((tr["why"] == "target").mean()) if n else np.nan
    row[f"{pre}_miss_avg_pct"] = float(tr.loc[tr["why"] != "target", "ret"].mean() * 100) if (tr["why"] != "target").any() else np.nan
    row[f"{pre}_n_buy_soxs"] = int((tr["sym"] == "SOXS").sum()) if n else 0
    return row


# ------------------------------------------------------------------ 얼린 설정 보조 측정(독립 계산)

def leg(D, costs, sym, d, f, cut):
    """봉 f 종가 + 반 호가 매수 → 봉 f+1부터 +1% 지정가(관통 1.0002) / 창 끝(cut−1 봉 종가 − 반 호가). %."""
    A = D.A[sym]
    e = A["Close"][d, f] * (1 + costs[sym][bucket(f)])
    hi = A["High"][d, f + 1:cut]
    k = np.nonzero(hi >= e * (1 + TARGET) * TOUCH)[0]
    if len(k):
        return TARGET * 100
    return (A["Close"][d, cut - 1] * (1 - costs[sym][bucket(cut - 1)]) / e - 1) * 100


def boot_mean(vals, groups, nb=10000, seed=20261029):
    """날 묶음 부트스트랩: 매매 가중 평균과 95% 구간."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    if not len(vals):
        return None
    u, inv = np.unique(groups, return_inverse=True)
    S = np.bincount(inv, weights=vals, minlength=len(u))
    N = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(u), (nb, len(u)))
    bs = S[ix].sum(1) / N[ix].sum(1)
    return [round(float(vals.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def same_minute(D, costs, tr, cut):
    """얼린 매매 각각: 같은 체결 분에 오른 종목(추종)을 산 결과와 비교. 반대 다리는 엔진 손익과 대조."""
    other = {"SOXL": "SOXS", "SOXS": "SOXL"}
    opp, fol, eng, days = [], [], [], []
    for r in tr.itertuples(index=False):
        d = D.pos[r.day]
        f = idx(r.entry)
        opp.append(leg(D, costs, r.sym, d, f, cut))
        fol.append(leg(D, costs, other[r.sym], d, f, cut))
        eng.append(r.ret * 100)
        days.append(d)
    opp, fol, eng = map(np.array, (opp, fol, eng))
    if not len(opp):
        return {"n": 0}
    return {"n": int(len(opp)), "engine_vs_independent_max_abs_pct": float(np.abs(opp - eng).max()),
            "opp_mean": boot_mean(eng, days), "follow_same_minute_mean": boot_mean(fol, days),
            "same_minute_random_mean": boot_mean((eng + fol) / 2, days),
            "opp_minus_random": boot_mean((eng - fol) / 2, days),
            "note": "추종 다리는 반대 다리 체결 분에 오른 종목을 같은 규칙(+1% 지정가·창 끝)으로 산 값, 서로 독립 계산"}


def feature_check(D):
    """events_sel_events.csv(사건 분석, 날 넘어 이은 지표)의 신호 봉 vratio·rsi·vwap_pct와 Feat 값 대조."""
    p = os.path.join(ROOT, "results", "events_sel_events.csv")
    if not os.path.exists(p):
        return None
    ev = pd.read_csv(p, usecols=["d", "rise", "i", "kind", "x", "win", "vratio", "rsi", "vwap_pct", "close"])
    ev = ev[(ev.win == "w90") & (ev.kind == "first")].drop_duplicates(["d", "rise", "i"])
    diffs = {"vratio": [], "rsi": [], "vwap_pct": []}
    for r in ev.itertuples(index=False):
        ctx = Ctx(D, int(r.d))
        ctx.i = int(r.i)
        f = feat(ctx, r.rise, int(r.i))
        diffs["vratio"].append(abs(f.vr[r.i] - r.vratio) if np.isfinite(r.vratio) else 0.0)
        diffs["rsi"].append(abs(f.rsi[r.i] - r.rsi))
        diffs["vwap_pct"].append(abs((r.close / f.vwap[r.i] - 1) * 100 - r.vwap_pct))
    return {"n_bars": int(len(ev)), **{k: float(np.nanmax(v)) for k, v in diffs.items()}}


# ------------------------------------------------------------------ 실행

def main():
    t0 = time.time()
    D = Data()
    print(f"data {time.time() - t0:.1f}s", flush=True)
    G = grid()
    assert len(G) <= 400, len(G)
    assert len({cid(p) for p in G.values()}) == len(G)
    print("configs", len(G), flush=True)
    E = {"w90": Engine(D, scheme="w90"), "std": Engine(D, scheme="std")}
    keys = list(G)
    rows = []
    for n, k in enumerate(keys):
        p = G[k]
        row = {"id": cid(p), **{kk: p[kk] for kk in KEYS}}
        for sc, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            assert np.isfinite(daily["ret"]).all(), (cid(p), sc)
            row.update(flat(summarize(daily), tr, sc))
        rows.append(row)
        if n % 40 == 0:
            print(n, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    kset = set(keys)
    pos = {k: n for n, k in enumerate(keys)}
    NB = {k: neighbors(k, kset) for k in keys}
    for sc in E:
        med, nn = [], []
        for k in keys:
            ts = [df.at[pos[k], f"{sc}_t"]] + [df.at[pos[q], f"{sc}_t"] for q in NB[k]]
            med.append(float(np.median(ts)))
            nn.append(len(NB[k]))
        df[f"{sc}_nbr_med_t"] = med
        df[f"{sc}_n_nbr"] = nn
        df[f"{sc}_eligible"] = df[f"{sc}_active_days"] >= 20
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "lay_L1_sel.csv"), index=False, encoding="utf-8-sig")
    print(f"grid done {time.time() - t0:.0f}s", flush=True)

    frozen, landscape, top = [], {}, {}
    for sc in E:
        el = df[df[f"{sc}_eligible"]]
        ts = df[f"{sc}_t"]
        landscape[sc] = {"n_eligible": int(len(el)), "t_max_all": float(ts.max()), "t_median_all": float(ts.median()),
                         "t_pos_frac_all": float((ts > 0).mean()),
                         "t_max_eligible": float(el[f"{sc}_t"].max()) if len(el) else None,
                         "t_median_eligible": float(el[f"{sc}_t"].median()) if len(el) else None,
                         "t_pos_frac_eligible": float((el[f"{sc}_t"] > 0).mean()) if len(el) else None,
                         "nbr_med_max_eligible": float(el[f"{sc}_nbr_med_t"].max()) if len(el) else None,
                         "mean_pct_median_eligible": float(el[f"{sc}_mean_pct"].median()) if len(el) else None,
                         "ci_lo_pos_eligible": int((el[f"{sc}_ci_lo"] > 0).sum()) if len(el) else 0,
                         "by_base_t_median_eligible": el.groupby("base")[f"{sc}_t"].median().round(3).to_dict()}
        cols = ["id", f"{sc}_nbr_med_t", f"{sc}_t", f"{sc}_mean_pct", f"{sc}_active_days", f"{sc}_trades",
                f"{sc}_target_rate", f"{sc}_avg_trade_pct"]
        top[sc] = el.sort_values([f"{sc}_nbr_med_t", f"{sc}_t"], ascending=False)[cols].head(10).round(4).to_dict("records")
        if not len(el):
            continue
        best = el.sort_values([f"{sc}_nbr_med_t", f"{sc}_t"], ascending=False).iloc[0]
        if any(f["id"] == best["id"] for f in frozen):
            frozen[[f["id"] for f in frozen].index(best["id"])]["scheme"] += "+" + sc
            continue
        frozen.append({"id": best["id"], "scheme": sc})

    costs = load_costs()
    id2k = {cid(G[k]): k for k in keys}
    h0k = tuple(dict(DEFAULTS, base="Lmin", x=3.0)[kk] for kk in KEYS)
    h0row = df.loc[pos[h0k]]
    out = []
    for f in frozen:
        k = id2k[f["id"]]
        p = dict(G[k])
        sel, mir, sm, halves = {}, {}, {}, {}
        for sc, eng in E.items():
            daily, tr = eng.run(make(p), period="sel")
            s = summarize(daily)
            s["avg_trade_pct"] = float(tr["ret"].mean() * 100) if len(tr) else None
            s["target_rate"] = float((tr["why"] == "target").mean()) if len(tr) else None
            s["exits"] = tr["why"].value_counts().to_dict() if len(tr) else {}
            s["buys_by_sym"] = tr["sym"].value_counts().to_dict() if len(tr) else {}
            s["nbr_med_t"] = float(df.at[pos[k], f"{sc}_nbr_med_t"])
            miss = tr[tr["why"] != "target"]["ret"] * 100
            s["miss"] = {"n": int(len(miss)), "mean": float(miss.mean()) if len(miss) else None,
                         "min": float(miss.min()) if len(miss) else None}
            sel[sc] = s
            dm, trm = eng.run(make(p, mirror=True), period="sel")
            mir[sc] = {kk: v for kk, v in summarize(dm).items()}
            mir[sc]["avg_trade_pct"] = float(trm["ret"].mean() * 100) if len(trm) else None
            sm[sc] = same_minute(D, costs, tr, idx("11:00") if sc == "w90" else idx("10:00"))
            dd = daily.copy()
            dd["m"] = pd.to_datetime(dd["day"]).dt.month
            halves[sc] = {"1~3월": float(dd[dd.m <= 3]["ret"].mean() * 100),
                          "4~6월": float(dd[dd.m >= 4]["ret"].mean() * 100)}
            tr.to_csv(os.path.join(ROOT, "results", f"lay_L1_trades_sel_{f['id']}_{sc}.csv"), index=False, encoding="utf-8-sig")
        out.append({"id": f["id"], "scheme": f["scheme"], "params": p,
                    "factory": f"lay_L1.make({json.dumps(p)})".replace("null", "None"),
                    "code": os.path.join(HERE, "lay_L1.py"),
                    "sel_w90": sel["w90"], "sel_std": sel["std"], "sel_halves_daily_mean_pct": halves,
                    "mirror_sel": mir, "same_minute_sel": sm,
                    "neighbors": [{"id": cid(G[q]), "w90_t": float(df.at[pos[q], "w90_t"]),
                                   "std_t": float(df.at[pos[q], "std_t"])} for q in NB[k]]})
    h0ref = {sc: {kk: (float(h0row[f"{sc}_{kk}"]) if pd.notna(h0row[f"{sc}_{kk}"]) else None)
                  for kk in ["mean_pct", "t", "ci_lo", "ci_hi", "active_days", "trades", "avg_trade_pct",
                             "target_rate", "miss_avg_pct", "nbr_med_t"]} for sc in E}
    res = {"family": "L1 소진", "n_configs": len(G), "period": "sel 2026-01-02~06-30 (123일), 잠금 켜지 않음",
           "engine": "Engine(D, scheme='w90'|'std'), fill='close', cost_table 1x, 목표 +1% 지정가, 손절 없음",
           "score": "sel 하루 평균 t(전체 sel 날), 활동일 20일 이상만 후보, 최종 = 자기+격자 이웃 t 중앙값(순서 축 하나 바로 옆, base는 범주)",
           "grid": GRID_DOC, "axes": {k: [str(v) for v in vs] for k, vs in AX.items()}, "bases": BASES,
           "h0_on_sel_reference": {"id": cid(G[h0k]), **h0ref},
           "landscape": landscape, "top10_by_score": top, "frozen": out,
           "feature_check_vs_events_sel": feature_check(D),
           "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(ROOT, "results", "lay_L1_frozen.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"landscape": landscape, "frozen": [{k: o[k] for k in ["id", "scheme"]} for o in out],
                      "h0": h0ref}, ensure_ascii=False, default=str), flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
