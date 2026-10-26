"""8단계 검증 p8_V_AIFG(2026-10-12): 갈래 AIFG의 대표 후보를 깨려고 처음부터 다시 잰다.

대표 후보: 7종목 쏠림판 S1 '평소 60%, AI쏠림 ≤25이면 100%, ≥40이면 60%'(그날 종가 상태 → 다음 거래일 시가).
부후보(갈래가 사후에 본 것, 선택편향 큼): 쏠림판 S2 '평소 100%, ≥75이면 50%, ≤60이면 100%'.

독립 재계산 원칙
  - p8_AIFG·harness·trade_lab·stock_lab·lev5를 import하지 않는다. 읽는 것: cache/p8_AIFG_px.parquet(가격·거래량),
    cache/base_2026.json·base_hold.json(7종목 시가 경로), cache/p7_idx.parquet(^GSPC·^SOX), fg_data.fg()(CNN 자료 읽기),
    저장소 watchlist.AI_UNIVERSE(종목 목록). 갈래 지수 cache/p8_AIFG_index.parquet는 '대조'로만 읽는다.
  - 지수·상태·시점·지표(수익·낙폭·고정초과·효율선초과)를 새로 짰다. 효율선은 고정 비중 0~1을 0.001 간격(갈래 0.005)으로 훑는다.

사전 고정(이 파일을 처음 돌리기 전에 정함, 확인 결과를 보고 바꾸지 않음)
  이웃: 규칙 모수마다 앞뒤 2칸. S1 w0 {0.4..0.8 / 0.1}, T {15..35 / 5}, 해제 {30..50 / 5} → 한 칸씩 12개 + 정육면체 125개.
        S2 w1 {0.3..0.7 / 0.1}, G {65..85 / 5}, 복귀 {50..70 / 5} → 12개 + 125개.
        지수 모수: 백분위 창 {126,189,252,315,378}, 50일선 {30..70 / 10}, 신고가 창 {126..378 / 63}, A5 평균 {1,3,5,7,9}일,
        요소 빼기 3개(A2·A3 / A2·A5 / A3·A5). 통과 = 2026 뒤 효율선초과 > 0이고 고정초과 > 0(갈래와 같은 기준).
  위약: (1) 지수 원형 밀기 63×k일(k=1..20) (2) 무작위 날짜 20개: 그 구간의 실제 노출 경로에서 '평소와 다른 구간'의 개수·길이를
        그대로 두고 시작일만 무작위로 옮긴다(평균 노출 같음 → 시점 정보만 시험). 순위 = 1 + (위약 > 실제)인 수, 21개 중.
        무작위 2,000개 분포에서 실제보다 큰 비율도 적는다.
  시점 감사: CNN 일별 변화와 같은 날·앞뒤 하루 S&P 종가 수익의 상관(보관본 2011~2021-01 / 공식 2021-02~ 따로),
        쏠림판 일별 변화와 ^SOX 수익의 상관. 하루 늦춤(종가 t → 시가 t+2), 미끄러짐 0.5%(노출 변화 1당 편도), 현금 이자 4%.
  시대별: 2011, 2015-06~2016-06, 2018 4분기, 2020, 2022, 2025, 2026 앞·뒤·7월 쇼크. 7종목 경로가 없는 해는 대리 대상
        (AI 유니버스 120일 모멘텀 상위 7 동일가중 주간 교체, SMH)로 잰다. 쏠림판은 2013-01부터라 2011은 CNN판만.
지표: 효율선초과 = 규칙 수익 − (규칙 최대 낙폭 이하가 나오는 고정 비중 중 최고 수익). 2025·2024는 연율 %p, 나머지 구간 %p.
      고정초과 = 규칙 수익 − 같은 구간 평균 노출로 고정한 수익.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fg_data  # noqa: E402  (자료 읽기 전용)

CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ROOT = HERE.parents[4]
ID = "p8_V_AIFG"

# ============================================================================ 사전 고정 상수
REP = {"kind": "S1", "w0": 0.6, "T": 25, "Tx": 40}
SUB = {"kind": "S2", "w1": 0.5, "G": 75, "Gb": 60}
IDX_DEF = {"pct": 252, "ma": 50, "hl": 252, "a5": 5, "comps": ("A2", "A3", "A5")}
BAD_UP, BAD_DN = 1.5, -0.7
A5_MIN = 10
W26 = {"2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "2026전체": ("2026-01-02", "2026-10-09")}
WHOLD = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31")}
ANNUAL = {"2025", "2024"}
NB_S1 = {"w0": (0.4, 0.5, 0.6, 0.7, 0.8), "T": (15, 20, 25, 30, 35), "Tx": (30, 35, 40, 45, 50)}
NB_S2 = {"w1": (0.3, 0.4, 0.5, 0.6, 0.7), "G": (65, 70, 75, 80, 85), "Gb": (50, 55, 60, 65, 70)}
NB_IDX = {"pct": (126, 189, 252, 315, 378), "ma": (30, 40, 50, 60, 70), "hl": (126, 189, 252, 315, 378), "a5": (1, 3, 5, 7, 9)}
LOO = (("A2", "A3"), ("A2", "A5"), ("A3", "A5"))
SHIFTS = tuple(63 * k for k in range(1, 21))
N_RAND, N_RAND_BIG, SEED = 20, 2000, 20261012
EG = np.round(np.arange(0.0, 1.0000001, 0.001), 3)
SLIP, CASH = 0.005, 0.04
LAGS = (-2, -1, 0, 1, 2)
MOM, TOPN = 120, 7
ERAS = {"2011": ("2011-01-03", "2011-12-30"), "2015~16": ("2015-06-01", "2016-06-30"), "2018 4분기": ("2018-10-01", "2018-12-31"),
        "2020": ("2020-01-02", "2020-12-31"), "2022": ("2022-01-03", "2022-12-30"), "2025": ("2025-01-02", "2025-12-31"),
        "2026앞": W26["2026앞"], "2026뒤": W26["2026뒤"], "AI쇼크": W26["AI쇼크"]}
CLAIM = {"2026앞": (1.236, -0.115, 0.207), "2026뒤": (0.077, -0.177, 0.031), "AI쇼크": (-0.160, None, None),
         "2026전체": (1.407, -0.177, -0.411), "2025": (0.414, -0.434, 0.005), "2024": (None, None, 0.048)}


# ============================================================================ 자료
class Panel:
    def __init__(self):
        px = pd.read_parquet(CACHE / "p8_AIFG_px.parquet")
        px["date"] = px["date"].astype(str)
        self.px = px
        sys.path.insert(0, str(ROOT))
        import watchlist
        uni = sorted({t for g in watchlist.AI_UNIVERSE.values() for lst in g.values() for t in lst if "." not in t})
        self.universe = uni
        u = px[px.sym.isin(uni)]
        piv = lambda f: u.pivot_table(index="date", columns="sym", values=f).sort_index()  # noqa: E731
        c, o, ac = piv("Close"), piv("Open"), piv("Adj Close")
        bad = []
        for s in ac.columns:
            x = ac[s].dropna().to_numpy()
            r = x[1:] / x[:-1] - 1
            if len(r) and ((r > BAD_UP).any() or (r < BAD_DN).any()):
                bad.append(s)
        keep = [s for s in ac.columns if s not in bad]
        self.bad = bad
        self.ac = ac[keep]
        self.ao = (o * (ac / c))[keep]
        self.days = list(self.ac.index)

    def etf(self, sym, field):
        e = self.px[self.px.sym == sym].set_index("date")[field].astype(float).sort_index()
        return e[~e.index.duplicated()]


def idx_series(sym, field="Close"):
    d = pd.read_parquet(CACHE / "p7_idx.parquet")
    d["date"] = d["date"].astype(str)
    s = d[d.sym == sym].set_index("date")[field].astype(float).sort_index()
    return s[~s.index.duplicated()]


def pct_rank(x: pd.Series, win: int) -> pd.Series:
    """과거 win개 관측(그날 포함) 안에서 그날 값의 백분위(같은 값은 평균 순위) × 100."""
    x = x.dropna()
    v = x.to_numpy(float)
    out = np.full(len(v), np.nan)
    if len(v) >= win:
        sw = np.lib.stride_tricks.sliding_window_view(v, win)
        last = sw[:, -1][:, None]
        less = (sw < last).sum(axis=1)
        eq = (sw == last).sum(axis=1)
        out[win - 1:] = (less + (eq + 1) / 2) / win * 100
    return pd.Series(out, index=x.index)


def build_index(P: Panel, pct=252, ma=50, hl=252, a5=5, comps=("A2", "A3", "A5"), with_parts=False):
    ac = P.ac
    days = ac.index
    m = ac.rolling(ma, min_periods=ma).mean()
    v = m.notna() & ac.notna()
    raw = pd.DataFrame(index=days)
    raw["A2"] = ((ac > m) & v).sum(axis=1) / v.sum(axis=1).replace(0, np.nan)
    hi, lo = ac.rolling(hl, min_periods=hl).max(), ac.rolling(hl, min_periods=hl).min()
    v2 = hi.notna() & ac.notna()
    raw["A3"] = (((ac >= hi - 1e-12) & v2).sum(axis=1) - ((ac <= lo + 1e-12) & v2).sum(axis=1)) / v2.sum(axis=1).replace(0, np.nan)
    vl = pd.DataFrame({s: P.etf(s, "Volume") for s in ("SOXL", "SOXS")}).reindex(days)
    cl = pd.DataFrame({s: P.etf(s, "Close") for s in ("SOXL", "SOXS")}).reindex(days)
    good = vl["SOXS"].rolling(20, min_periods=20).min() >= A5_MIN
    start = good[~good].index.max()
    dv = cl * vl
    raw["A5"] = (dv["SOXL"] / dv["SOXS"].replace(0, np.nan)).rolling(a5, min_periods=a5).mean().where(days > start)
    pc = pd.DataFrame({k: pct_rank(raw[k], pct) for k in ("A2", "A3", "A5")}).reindex(days)
    cols = list(comps)
    n = pc[cols].notna().sum(axis=1)
    ix = pc[cols].mean(axis=1)[n >= len(cols) - 1]
    if len(cols) == 2:          # 요소 빼기 판: 두 요소 중 A5가 빠진 날은 남은 하나만(있는 요소만 같은 가중)
        ix = pc[cols].mean(axis=1)[n >= 1]
        if "A5" in cols:
            other = [c for c in cols if c != "A5"][0]
            ix = ix[pc[other].notna()]
    ix = ix.rename("AI쏠림")
    return (ix, raw, pc, start) if with_parts else ix


# ============================================================================ 대상(시가 → 다음 시가)
class Tgt:
    def __init__(self, name, days, opens):
        self.name = name
        self.days = list(days)
        o = np.asarray(opens, float)
        self.ret = pd.Series(o[1:] / o[:-1] - 1, index=self.days[:-1])
        self._fr = {}


def seven(bn):
    b = json.loads((CACHE / f"base_{bn}.json").read_text(encoding="utf-8"))
    return Tgt(f"7종목_{bn}", b["days"], b["open"])


def smh(P: Panel):
    o, c, a = P.etf("SMH", "Open"), P.etf("SMH", "Close"), P.etf("SMH", "Adj Close")
    ao = (o * a / c).dropna()
    return Tgt("SMH", ao.index, ao.to_numpy())


def ai7(P: Panel):
    """대리 대상: AI 유니버스(자료 오류 제외) 120일 모멘텀 상위 7 동일가중, 주 첫 거래일 시가 교체(직전 종가로 고름)."""
    ac, ao = P.ac, P.ao.ffill(limit=3)
    days = list(ac.index)
    A, AO = ac.to_numpy(float), ao.to_numpy(float)
    mom = np.full_like(A, np.nan)
    mom[MOM + 1:] = A[MOM:-1] / A[:-MOM - 1] - 1          # 날 i: 종가 i-1 / 종가 i-1-MOM
    wk = [date.fromisoformat(d).isocalendar()[:2] for d in days]
    reb = [i for i in range(len(days)) if i == 0 or wk[i] != wk[i - 1]]
    reb.append(len(days) - 1)
    val = np.full(len(days), np.nan)
    v0 = 1.0
    for a, b in zip(reb[:-1], reb[1:]):
        mm = mom[a].copy()
        mm[~np.isfinite(AO[a])] = np.nan
        ok = np.where(np.isfinite(mm))[0]
        if len(ok) < TOPN:
            continue
        pick = ok[np.argsort(-mm[ok])[:TOPN]]
        rel = AO[a:b + 1, pick] / AO[a, pick]
        rel = pd.DataFrame(rel).ffill().to_numpy()
        path = v0 * np.nanmean(rel, axis=1)
        val[a:b + 1] = path
        v0 = path[-1]
    s = pd.Series(val, index=days).dropna()
    return Tgt("AI7대리", s.index, s.to_numpy())


# ============================================================================ 규칙·시점
def state(x: pd.Series, rule: dict) -> pd.Series:
    v = x.to_numpy(float)
    out = np.empty(len(v))
    if rule["kind"] == "S1":
        cur = rule["w0"]
        for i, z in enumerate(v):
            if z <= rule["T"]:
                cur = 1.0
            elif z >= rule["Tx"]:
                cur = rule["w0"]
            out[i] = cur
    else:
        cur = 1.0
        for i, z in enumerate(v):
            if z >= rule["G"]:
                cur = rule["w1"]
            elif z <= rule["Gb"]:
                cur = 1.0
            out[i] = cur
    return pd.Series(out, index=x.index)


def base_w(rule):
    return rule["w0"] if rule["kind"] == "S1" else 1.0


def exposure(tg: Tgt, x: pd.Series, rule: dict, lag: int = 0) -> pd.Series:
    """종가 t 상태 → t보다 엄격히 뒤의 첫 대상 거래일 시가부터. lag=1이면 하루 더 늦춘다."""
    x = x.dropna().sort_index()
    st = state(x, rule)
    if lag:
        st = pd.Series(st.to_numpy()[:-lag], index=st.index[lag:])
    keys = st.index.to_numpy()
    pos = np.searchsorted(keys, np.asarray(tg.days), side="left") - 1
    vals = np.where(pos >= 0, st.to_numpy()[np.clip(pos, 0, None)], base_w(rule))
    return pd.Series(vals, index=tg.days).reindex(tg.ret.index)


def frontier(tg: Tgt, lo, hi, cash):
    k = (lo, hi, cash)
    if k not in tg._fr:
        rb = tg.ret[(tg.ret.index >= lo) & (tg.ret.index <= hi)].to_numpy()
        d = np.outer(EG, rb) + np.outer(1 - EG, np.full(len(rb), cash / 252))
        cur = np.concatenate([np.ones((len(EG), 1)), np.cumprod(1 + d, axis=1)], axis=1)
        tg._fr[k] = (cur[:, -1] - 1, (cur / np.maximum.accumulate(cur, axis=1) - 1).min(axis=1), len(rb))
    return tg._fr[k]


def wstats(tg: Tgt, e: pd.Series, lo, hi, annual=False, slip=0.0, cash=0.0):
    m = (tg.ret.index >= lo) & (tg.ret.index <= hi)
    rb, ee = tg.ret[m].to_numpy(), e[m].to_numpy()
    if not len(rb):
        return None
    turn = np.abs(np.diff(ee, prepend=ee[0]))
    d = ee * rb + (1 - ee) * cash / 252 - slip * turn
    cur = np.concatenate([[1.0], np.cumprod(1 + d)])
    r, dd = cur[-1] - 1, (cur / np.maximum.accumulate(cur) - 1).min()
    avg = ee.mean()
    fixed = np.prod(1 + avg * rb + (1 - avg) * cash / 252) - 1
    fr, fd, n = frontier(tg, lo, hi, cash)
    j = int(np.argmax(np.where(fd >= dd - 1e-12, fr, -np.inf)))
    yrs = n / 252
    an = (lambda z: (1 + z) ** (1 / yrs) - 1) if annual else (lambda z: z)
    return {"수익": float(r), "낙폭": float(dd), "노출": float(avg), "고정초과": float(an(r) - an(fixed)),
            "효율선초과": float(an(r) - an(fr[j])), "효율선w": float(EG[j]), "매매수": int((np.abs(np.diff(ee)) > 1e-9).sum()),
            "보유수익": float(np.prod(1 + rb) - 1)}


def run(tg, x, rule, wins, lag=0, slip=0.0, cash=0.0):
    e = exposure(tg, x, rule, lag)
    return {w: wstats(tg, e, lo, hi, w in ANNUAL, slip, cash) for w, (lo, hi) in wins.items()}


def trace(tg, x, rule, lo, hi):
    e = exposure(tg, x, rule)
    e = e[(e.index >= lo) & (e.index <= hi)]
    ch = e[e.ne(e.shift())]
    xs = x.dropna().sort_index()
    out = []
    for d, v in ch.items():
        p = xs[xs.index < d]
        out.append([d, float(v), round(float(p.iloc[-1]), 1), p.index[-1]])
    return out


# ============================================================================ 위약
def placebo_shift(tg, x, rule, win, key="효율선초과"):
    lo, hi = win[1]
    ann = win[0] in ANNUAL
    act = wstats(tg, exposure(tg, x, rule), lo, hi, ann)[key]
    xs = x.dropna().sort_index()
    vals = []
    for s in SHIFTS:
        y = pd.Series(np.roll(xs.to_numpy(), s), index=xs.index)
        vals.append(wstats(tg, exposure(tg, y, rule), lo, hi, ann)[key])
    vals = np.array(vals)
    return {"실제": float(act), "순위": int(1 + (vals > act + 1e-12).sum()), "개수": len(vals) + 1, "위약중앙": float(np.median(vals))}


def _runs(ee, basev):
    """평소 값과 다른 연속 구간의 길이 목록."""
    on = np.abs(ee - basev) > 1e-9
    L, vals, i = [], [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            L.append(j - i)
            vals.append(ee[i])
            i = j
        else:
            i += 1
    return L, vals


def placebo_random(tg, x, rule, win, n, seed, key="효율선초과"):
    lo, hi = win[1]
    ann = win[0] in ANNUAL
    e = exposure(tg, x, rule)
    m = (e.index >= lo) & (e.index <= hi)
    ee = e[m].to_numpy()
    act = wstats(tg, e, lo, hi, ann)[key]
    bv = base_w(rule)
    L, vals = _runs(ee, bv)
    k = len(L)
    rng = np.random.default_rng(seed)
    out = []
    if k == 0:
        return {"실제": float(act), "순위": 1, "개수": n + 1, "큰비율": float("nan"), "구간수": 0}
    free = (len(ee) - sum(L)) - (k - 1)
    for _ in range(n):
        order = rng.permutation(k)
        cuts = np.sort(rng.choice(free + k, size=k, replace=False))
        bins = np.diff(np.concatenate([[-1], cuts, [free + k]])) - 1
        gaps = bins.copy()
        gaps[1:-1] += 1
        z = []
        for i in range(k):
            z += [bv] * int(gaps[i]) + [vals[order[i]]] * L[order[i]]
        z += [bv] * int(gaps[-1])
        ez = e.copy()
        ez[m] = np.array(z)
        out.append(wstats(tg, ez, lo, hi, ann)[key])
    out = np.array(out)
    return {"실제": float(act), "순위": int(1 + (out[:N_RAND] > act + 1e-12).sum()), "개수": N_RAND + 1,
            "큰비율": float((out > act + 1e-12).mean()), "구간수": k, "위약중앙": float(np.median(out))}


# ============================================================================ 시점 감사
def timing_audit(ix: pd.Series):
    fg = fg_data.fg()
    spx = idx_series("^GSPC")
    sox = idx_series("^SOX")
    out = {}
    for name, (lo, hi) in {"보관본 2011~2021-01": ("2011-01-01", "2021-01-31"), "공식 2021-02~": ("2021-02-01", "2026-12-31")}.items():
        j = pd.concat([fg.rename("fg"), spx.rename("px")], axis=1, join="inner").sort_index()
        j = j[(j.index >= lo) & (j.index <= hi)]
        dfg, r = j.fg.diff(), j.px.pct_change()
        out[name] = {"일수": len(j), "상관": {L: float(dfg.corr(r.shift(-L))) for L in LAGS},
                     "S&P휴장일값": int(len(set(fg[(fg.index >= lo) & (fg.index <= hi)].index) - set(spx.index)))}
    j = pd.concat([ix.rename("ix"), sox.rename("px")], axis=1, join="inner").sort_index()
    out["쏠림판-SOX"] = {"일수": len(j), "상관": {L: float(j.ix.diff().corr(j.px.pct_change().shift(-L))) for L in LAGS}}
    # 보관본과 공식이 겹치는 2020-09~2021-01(공식 50.0 자리표시 제외)에서 날짜 밀림 확인
    arc = pd.read_csv(CACHE / "fg_archive.csv")
    arc = pd.Series(arc["Fear Greed"].astype(float).to_numpy(), index=arc["Date"].astype(str)).sort_index()
    jj = json.loads((CACHE / "fg_cnn.json").read_text())
    pts = jj["fear_and_greed_historical"]["data"]
    off = pd.Series([p["y"] for p in pts], index=pd.to_datetime([p["x"] for p in pts], unit="ms").strftime("%Y-%m-%d"), dtype=float)
    off = off[~off.index.duplicated(keep="last")].sort_index()
    ov = pd.concat([arc.rename("a"), off.rename("o")], axis=1, join="inner")
    ov = ov[(ov.o != 50.0)]
    out["보관본-공식 겹침"] = {"일수": len(ov), "같은날 평균차": float((ov.a - ov.o).abs().mean()),
                          "보관본을 하루 당긴 평균차": float((ov.a.shift(-1) - ov.o).abs().mean()),
                          "보관본을 하루 늦춘 평균차": float((ov.a.shift(1) - ov.o).abs().mean())}
    return out


# ============================================================================ 실행
def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p".replace("-", "−")


def pc_(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+,.{d}f}%".replace("-", "−")


def main():
    P = Panel()
    ix, raw, pc, a5s = build_index(P, with_parts=True)
    ref = pd.read_parquet(CACHE / "p8_AIFG_index.parquet")
    ref.index = ref.index.astype(str)
    j = pd.concat([ix, ref["AI쏠림"].rename("ref")], axis=1).dropna()
    rec = {"제외종목": P.bad, "지수종목수": int(P.ac.shape[1]), "A5시작(이 날 뒤)": a5s, "지수시작": ix.index[0], "지수끝": ix.index[-1],
           "갈래지수대비_최대차": float((j.iloc[:, 0] - j.ref).abs().max()), "겹친날": len(j),
           "2026-10-09": round(float(ix.iloc[-1]), 1),
           "A5백분위시작": pc["A5"].dropna().index[0]}
    print("재구성", rec)
    cnn = fg_data.fg()
    t26, th = seven("2026"), seven("hold")

    def both(x, rule, **kw):
        return {**run(t26, x, rule, W26, **kw), **run(th, x, rule, WHOLD, **kw)}

    res = {"대표_쏠림": both(ix, REP), "대표_CNN": both(cnn, REP), "대표_갈래지수": both(ref["AI쏠림"], REP),
           "부후보_쏠림": both(ix, SUB), "부후보_CNN": both(cnn, SUB),
           "기본": {**{w: wstats(t26, pd.Series(1.0, index=t26.ret.index), lo, hi) for w, (lo, hi) in W26.items()},
                  **{w: wstats(th, pd.Series(1.0, index=th.ret.index), lo, hi, True) for w, (lo, hi) in WHOLD.items()}}}
    for k in ("대표_쏠림", "대표_CNN", "부후보_쏠림"):
        print(k, {w: (round(v["수익"], 4), round(v["낙폭"], 4), round(v["효율선초과"], 4), round(v["고정초과"], 4)) for w, v in res[k].items()})
    sens = {}
    for nm, rule in (("대표", REP), ("부후보", SUB)):
        sens[nm] = {"하루늦춤": both(ix, rule, lag=1), "미끄러짐0.5%": both(ix, rule, slip=SLIP), "이자4%": both(ix, rule, cash=CASH),
                    "늦춤+미끄러짐": both(ix, rule, lag=1, slip=SLIP)}
    tr = {"대표": trace(t26, ix, REP, "2026-01-01", "2026-10-09"), "대표_hold": trace(th, ix, REP, "2024-01-01", "2025-12-31"),
          "부후보": trace(t26, ix, SUB, "2026-01-01", "2026-10-09")}
    print("대표 2026 노출 변화", tr["대표"])

    # ---------------- 위약
    plac = {}
    for nm, rule in (("대표", REP), ("부후보", SUB)):
        plac[nm] = {}
        for w in ("2026앞", "2026뒤", "2026전체", "2025", "2024"):
            tg = t26 if w.startswith("2026") else th
            win = (w, (W26 | WHOLD)[w])
            plac[nm][w] = {"원형밀기": placebo_shift(tg, ix, rule, win), "무작위날짜": placebo_random(tg, ix, rule, win, N_RAND_BIG, SEED),
                           "원형밀기_고정초과": placebo_shift(tg, ix, rule, win, "고정초과"),
                           "무작위날짜_고정초과": placebo_random(tg, ix, rule, win, N_RAND_BIG, SEED, "고정초과")}
    print("위약", json.dumps(plac, ensure_ascii=False)[:1500])

    # ---------------- 이웃
    nb = {}
    for nm, rule, grid in (("대표", REP, NB_S1), ("부후보", SUB, NB_S2)):
        one, cube = {}, {}
        keys = list(grid)
        for k in keys:
            for v in grid[k]:
                r = {**rule, k: v}
                one[f"{k}={v}"] = both(ix, r)
        import itertools
        for vals in itertools.product(*[grid[k] for k in keys]):
            r = {**rule, **dict(zip(keys, vals))}
            rr = both(ix, r)
            cube[" ".join(f"{k}={v}" for k, v in zip(keys, vals))] = {w: {a: rr[w][a] for a in ("수익", "낙폭", "효율선초과", "고정초과")} for w in rr}
        nb[nm] = {"한칸씩": one, "정육면체": cube}
    nbi = {}
    for k, vals in NB_IDX.items():
        for v in vals:
            if v == IDX_DEF[k]:
                continue
            x = build_index(P, **{**{a: b for a, b in IDX_DEF.items() if a != "comps"}, k: v})
            nbi[f"{k}={v}"] = {"대표": both(x, REP), "부후보": both(x, SUB)}
    for c in LOO:
        x = build_index(P, comps=c)
        nbi["요소=" + "·".join(c)] = {"대표": both(x, REP), "부후보": both(x, SUB)}

    # ---------------- 시점 감사
    ta = timing_audit(ix)
    print("시점", json.dumps(ta, ensure_ascii=False))

    # ---------------- 시대별
    tsm, tai = smh(P), ai7(P)
    corr = {}
    for nm, t in (("2026", t26), ("hold", th)):
        jj = pd.concat([t.ret.rename("a"), tai.ret.rename("b"), tsm.ret.rename("c")], axis=1, join="inner")
        corr[nm] = {"AI7대리": float(jj.a.corr(jj.b)), "SMH": float(jj.a.corr(jj.c)), "일수": len(jj)}
    eras = {}
    for era, (lo, hi) in ERAS.items():
        row = {}
        tgs = [("AI7대리", tai), ("SMH", tsm)]
        if era.startswith("2026") or era == "AI쇼크":
            tgs = [("7종목", t26)] + tgs
        elif era == "2025":
            tgs = [("7종목", th)] + tgs
        for tn, tg in tgs:
            b = wstats(tg, pd.Series(1.0, index=tg.ret.index), lo, hi)
            cell = {"기본": b}
            for xn, x in (("쏠림", ix), ("CNN", cnn)):
                if x.dropna().index[0] > lo:
                    cell[xn] = None
                    continue
                cell[xn] = wstats(tg, exposure(tg, x, REP), lo, hi)
                cell[xn + "_부후보"] = wstats(tg, exposure(tg, x, SUB), lo, hi)
            row[tn] = cell
        eras[era] = row

    # ---------------- 갈래와 차이 원인: 효율선 격자(갈래 0.005, 여기 0.001)
    global EG
    keep = EG
    EG = np.round(np.arange(0.0, 1.0000001, 0.005), 3)
    t26b, thb = seven("2026"), seven("hold")
    g5 = {"쏠림": {**run(t26b, ix, REP, W26), **run(thb, ix, REP, WHOLD)}, "CNN": run(t26b, cnn, REP, W26)}
    EG = keep
    gridchk = {k: {w: v["효율선초과"] for w, v in d.items()} for k, d in g5.items()}
    # ---------------- 사건별 기여: 그 100% 구간만 남기고 나머지 100%는 평소(60%)로
    e = exposure(t26, ix, REP)
    epis = {}
    for wn, (wl, wh), spans in (("2026뒤", W26["2026뒤"], (("2026-07-14", "2026-08-12"), ("2026-08-20", "2026-09-21"))),
                                ("2026앞", W26["2026앞"], (("2026-03-09", "2026-03-17"),))):
        for lo, hi in spans:
            ee = e.copy()
            ee[((ee.index < lo) | (ee.index > hi)) & (ee > REP["w0"] + 1e-9)] = REP["w0"]
            s = wstats(t26, ee, wl, wh)
            rr = t26.ret[(t26.ret.index >= lo) & (t26.ret.index <= hi)]
            epis[f"{wn} {lo}~{hi}"] = {"효율선초과": s["효율선초과"], "고정초과": s["고정초과"], "7종목수익": float(np.prod(1 + rr) - 1), "일수": len(rr)}

    payload = {"재구성": rec, "res": res, "sens": sens, "trace": tr, "plac": plac, "nb": nb, "nbi": nbi, "timing": ta,
               "대리상관": corr, "eras": eras, "격자0.005": gridchk, "사건기여": epis,
               "grid": {"이웃_S1_한칸": 12, "이웃_S1_정육면체": 125, "이웃_S2_한칸": 12, "이웃_S2_정육면체": 125,
                        "지수모수": sum(len(v) - 1 for v in NB_IDX.values()) + len(LOO), "원형밀기": len(SHIFTS), "무작위": N_RAND_BIG}}
    RESULTS.mkdir(exist_ok=True)
    txt = json.dumps(payload, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    (RESULTS / f"{ID}.json").write_text(txt, encoding="utf-8")
    print("저장", RESULTS / f"{ID}.json")
    write_md(json.loads(txt))


# ============================================================================ 보고서
def _cell(v, eff=True):
    if not v:
        return "—"
    t = f"{pc_(v['수익'])}/{pc_(-abs(v['낙폭']), 1)}"
    return t + (f" ({pp(v['효율선초과'])})" if eff else "")


def _ok(r):
    return r["2026뒤"]["효율선초과"] > 0 and r["2026뒤"]["고정초과"] > 0


def write_md(P: dict):
    res, sens, plac, nb, nbi, ta, eras = P["res"], P["sens"], P["plac"], P["nb"], P["nbi"], P["timing"], P["eras"]
    R, C, B, S, SC = res["대표_쏠림"], res["대표_CNN"], res["기본"], res["부후보_쏠림"], res["부후보_CNN"]
    dl, sl = sens["대표"]["하루늦춤"], sens["대표"]["미끄러짐0.5%"]
    ep = P["사건기여"]
    cube, cube2 = nb["대표"]["정육면체"], nb["부후보"]["정육면체"]
    npass = sum(_ok(r) for r in cube.values())
    ni = sum(_ok(v["대표"]) for v in nbi.values())
    rnd = plac["대표"]["2026뒤"]["무작위날짜"]
    g5 = P["격자0.005"]
    L = []
    L.append("# 8단계 검증 V_AIFG: 7종목 쏠림판 '평소 60%, ≤25이면 100%, ≥40이면 60%' 깨 보기 (2026-10-12)\n")
    L.append("> 대상: 갈래 AIFG 대표 후보(AI쏠림 지수 = AI 유니버스 50일선 위 비율·52주 신고가−신저가·SOXL/SOXS 거래대금 5일 평균의 252일 백분위 평균)와 "
             "부후보 S2 '평소 100%, ≥75이면 50%, ≤60이면 100%'.  ")
    L.append("> 방법: 갈래 코드·공유 엔진을 import하지 않고 parquet·json만 읽어 지수·상태·지표를 새로 짰다. 효율선 격자 0.001(갈래 0.005). "
             f"코드 `code/{ID}.py`, 결과 `results/{ID}.json`. 시험 범위: 규칙 이웃 S1·S2 각 12+125개, 지수 모수 이웃 19개, 위약 원형 밀기 20개·무작위 날짜 2,000개, 시대 9구간 × 대상 3.\n")

    L.append("## 결론\n")
    L.append(f"1. **숫자는 재현됐고 시점 문제도 없다.** 다시 만든 지수는 갈래 지수와 3,464일 모두 같다(최대 차 {P['재구성']['갈래지수대비_최대차']:.1f}점). "
             f"2026 앞 {_cell(R['2026앞'])}, 뒤 {_cell(R['2026뒤'])}(고정 대비 {pp(R['2026뒤']['고정초과'])}), 7월 쇼크 {pc_(R['AI쇼크']['수익'])}. "
             f"2026 전체 효율선 {pp(R['2026전체']['효율선초과'])}는 갈래 −41.1%p와 0.6%p 다른데, 효율선 격자를 갈래처럼 0.005로 하면 {pp(g5['쏠림']['2026전체'])}로 같다. "
             f"CNN·쏠림판 모두 일별 변화가 같은 날 지수 수익과 가장 크게 움직였다(하루 밀림 없음).")
    L.append(f"2. **'뒤 통과'는 모수에는 튼튼하지만 사건 두 번에 기댄 것이고 우연과 구별되지 않는다.** 규칙 이웃 125개 중 {npass}개, 지수 모수 이웃 {len(nbi)}개 중 {ni}개가 뒤를 통과했고, "
             f"하루 늦춤 {pp(dl['2026뒤']['효율선초과'])}, 미끄러짐 0.5% {pp(sl['2026뒤']['효율선초과'])}도 플러스다. "
             f"그러나 앞 이익은 3/9~3/17 한 번(7거래일, 7종목 {pc_(ep['2026앞 2026-03-09~2026-03-17']['7종목수익'])}), 뒤 이익은 8/20~9/21 한 번({pp(ep['2026뒤 2026-08-20~2026-09-21']['효율선초과'])})에서 나왔다. "
             f"7월 쇼크 중 7/14 100% 전환은 {pp(ep['2026뒤 2026-07-14~2026-08-12']['효율선초과'])}로 손해였다. 같은 길이의 100% 구간을 무작위 날짜에 놓은 위약은 "
             f"{rnd['큰비율'] * 100:.0f}%가 더 나았다(20개 중 {rnd['순위']}위). 하루 늦추면 앞 이익이 {pp(dl['2026앞']['효율선초과'])}로 줄고 2026 전체 고정 대비는 {pp(dl['2026전체']['고정초과'])}가 된다.")
    L.append(f"3. **판정: 맞지만 취약(매매 규칙으로 쓸 근거 없음).** 2026 전체는 같은 낙폭 고정 비중보다 {pp(R['2026전체']['효율선초과'])}이고, 쇼크 구간은 같은 평균 노출 고정보다 "
             f"{pp(R['AI쇼크']['고정초과'])}다(덜 맞은 것은 평소 60%라서다). 대리 대상(AI 모멘텀 상위 7·SMH)으로 본 다른 해는 2022·2025에 고정 대비 마이너스, 2020은 엇갈림(−9.1%p·+3.2%p), 2015~16·2018 4분기는 ±2.3%p 안이었다. "
             f"부후보 S2는 2025에 튼튼했지만(이웃 125개 모두 플러스, 위약 2/21) 2026 앞은 이웃 125개 중 {sum(r['2026앞']['효율선초과'] > 0 for r in cube2.values())}개만 플러스, "
             f"하루 늦추면 2026 전체 {pp(sens['부후보']['하루늦춤']['2026전체']['효율선초과'])}라 2026 기준으로 기각이다.\n")

    # ---------------- 재현
    L.append("## 재현 표 (갈래 주장 → 다시 잰 값)\n")
    L.append("| 항목 | 갈래 주장 | 재계산 | 차이 원인 |")
    L.append("|---|---|---|---|")
    rows = [("2026 앞 수익/낙폭 (효율선)", "+123.6%/−11.5% (+20.7%p)", _cell(R["2026앞"]), ""),
            ("2026 뒤 수익/낙폭 (효율선)", "+7.7%/−17.7% (+3.1%p)", _cell(R["2026뒤"]), ""),
            ("2026 뒤 고정 대비", "+3.2%p", pp(R["2026뒤"]["고정초과"]), ""),
            ("7월 쇼크 (기본)", "−16.0% (−18.7%)", f"{pc_(R['AI쇼크']['수익'])} ({pc_(B['AI쇼크']['수익'])})", ""),
            ("2026 전체 (효율선)", "+140.7%/−17.7% (−41.1%p)", _cell(R["2026전체"]), f"격자 0.005면 {pp(g5['쏠림']['2026전체'])}"),
            ("2025 (효율선, 연)", "+41.4%/−43.4% (+0.5%p)", _cell(R["2025"]), f"격자 0.005면 {pp(g5['쏠림']['2025'])}"),
            ("2024 효율선 (연)", "+4.8%p", pp(R["2024"]["효율선초과"]), f"격자 0.005면 {pp(g5['쏠림']['2024'])}"),
            ("CNN판 뒤 / 전체 효율선", "−2.2%p / −62.8%p", f"{pp(C['2026뒤']['효율선초과'])} / {pp(C['2026전체']['효율선초과'])}", f"격자 0.005면 전체 {pp(g5['CNN']['2026전체'])}"),
            ("하루 늦춤 / 미끄러짐 0.5% (뒤)", "+5.3%p / +2.2%p", f"{pp(dl['2026뒤']['효율선초과'])} / {pp(sl['2026뒤']['효율선초과'])}", ""),
            ("위약 원형 밀기 순위 뒤 / 전체", "6/21 / 17/21", f"{plac['대표']['2026뒤']['원형밀기']['순위']}/21 / {plac['대표']['2026전체']['원형밀기']['순위']}/21", ""),
            ("노출 변경일(시가)", "7/14 100%, 8/13 60%, 8/20 100%",
             ", ".join(f"{d[5:]} {v * 100:.0f}%" for d, v, _, _ in P["trace"]["대표"][1:]), "3/9·3/18·9/22도 있음"),
            ("부후보 S2 7종목 2025 (기본)", "+80.7% (+42.2%)", f"{pc_(S['2025']['수익'])} ({pc_(B['2025']['수익'])})", ""),
            ("부후보 위약 2025 / 하루 늦춤 2026 전체", "2/21 / −59.8%p",
             f"{plac['부후보']['2025']['원형밀기']['순위']}/21 / {pp(sens['부후보']['하루늦춤']['2026전체']['효율선초과'])}", "")]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L.append(f"\n- 지수: 미국 115종목 중 자료 오류 {', '.join(P['재구성']['제외종목'])} 제외 → {P['재구성']['지수종목수']}종목. A5는 {P['재구성']['A5시작(이 날 뒤)']} 뒤부터, "
             f"백분위는 {P['재구성']['A5백분위시작']}부터(그 전은 A2·A3 평균). 지수는 2013-01-02부터라 2011은 잴 수 없다.\n")

    # ---------------- 사용자 표
    L.append("## 사용자 표\n")
    L.append("칸 = 구간 수익/최대 낙폭 (효율선초과). 2026·쇼크는 구간 %p, 2025·2024는 연 %p. 그날 종가 → 다음 거래일 시가, 수수료·현금 이자 0. 매매 수 = 2026 노출 변경 횟수.\n")
    L.append("| 규칙 | 2026 앞 | 2026 뒤 | 7월 AI 쇼크 | 2026 전체 | 2025 | 2024 | 2026 매매 수 |")
    L.append("|---|---|---|---|---|---|---|---|")
    tab = [("7종목 그대로", B, False), ("**대표** 쏠림 평소 60%, ≤25 → 100%, ≥40 → 60%", R, True),
           ("대표, 하루 늦춤", dl, True), ("대표, 미끄러짐 0.5%", sl, True), ("대표, 현금 이자 4%", sens["대표"]["이자4%"], True),
           ("대표, 하루 늦춤 + 미끄러짐", sens["대표"]["늦춤+미끄러짐"], True),
           ("같은 규칙 CNN판", C, True), ("부후보 쏠림 평소 100%, ≥75 → 50%, ≤60 → 100%", S, True),
           ("부후보, 하루 늦춤", sens["부후보"]["하루늦춤"], True), ("부후보 같은 규칙 CNN판", SC, True)]
    for lab, r, eff in tab:
        L.append(f"| {lab} | {_cell(r['2026앞'], eff)} | {_cell(r['2026뒤'], eff)} | {pc_(r['AI쇼크']['수익'])} | {_cell(r['2026전체'], eff)} | "
                 f"{_cell(r['2025'], eff)} | {_cell(r['2024'], eff)} | {r['2026전체']['매매수']} |")
    L.append("\n사건별 기여(그 100% 구간만 남기고 나머지 100%를 60%로 바꾼 판, 구간 효율선초과)\n")
    L.append("| 100% 구간(시가) | 7종목 그 구간 수익 | 구간 효율선초과 | 고정 대비 |")
    L.append("|---|---|---|---|")
    for k, v in ep.items():
        L.append(f"| {k} ({v['일수']}일) | {pc_(v['7종목수익'])} | {pp(v['효율선초과'])} | {pp(v['고정초과'])} |")

    # ---------------- 시점 감사
    L.append("\n## 시점 감사\n")
    L.append("일별 변화와 지수 종가 수익의 상관(수익 날짜 = 지수 날짜 + 시차). 같은 날(0)이 가장 커야 그날 장 마감 값이다.\n")
    L.append("| 계열 | 일수 | −2일 | −1일 | 같은 날 | +1일 | +2일 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("보관본 2011~2021-01", "공식 2021-02~", "쏠림판-SOX"):
        c = ta[k]["상관"]
        lab = {"보관본 2011~2021-01": "CNN 보관본 vs S&P", "공식 2021-02~": "CNN 공식 vs S&P", "쏠림판-SOX": "쏠림판 vs ^SOX"}[k]
        L.append(f"| {lab} | {ta[k]['일수']} | " + " | ".join(f"{c[str(x)]:.2f}" for x in LAGS) + " |")
    o = ta["보관본-공식 겹침"]
    L.append(f"\n- 보관본과 공식이 겹치는 {o['일수']}일: 같은 날 평균 차 {o['같은날 평균차']:.2f}점, 보관본을 하루 당기면 {o['보관본을 하루 당긴 평균차']:.1f}점, 늦추면 {o['보관본을 하루 늦춘 평균차']:.1f}점 → 날짜 밀림 없음. "
             f"CNN 값에 S&P 휴장일 날짜는 없다.")
    L.append("- 쏠림판 재료는 모두 그날 종가·거래량이라 장 마감 뒤 확정된다. 사후 정보는 두 가지뿐이다: 종목 목록(2026-10 기준, 생존편향), 자료 오류 종목(NVTS)과 A5 시작일을 전 기간 자료로 정한 것. 2026 신호 날짜에는 영향이 없다.")
    L.append(f"- 하루 늦춤(종가 t → 시가 t+2): 뒤 {pp(dl['2026뒤']['효율선초과'])}, 앞 {pp(dl['2026앞']['효율선초과'])}(고정 대비 {pp(dl['2026앞']['고정초과'])}), "
             f"전체 고정 대비 {pp(dl['2026전체']['고정초과'])}. 3월 사건은 7거래일짜리라 하루 늦으면 이익 대부분이 사라진다.\n")

    # ---------------- 조합 격자(이웃)
    L.append("## 조합 격자 표 (이웃, 사전 고정 앞뒤 2칸)\n")
    L.append("한 칸씩 바꾼 판(효율선초과 / 고정 대비). 통과 = 뒤 효율선초과 > 0이고 고정 대비 > 0.\n")
    L.append("| 바꾼 모수 | 2026 앞 | 2026 뒤 | 7월 쇼크 수익 | 2026 전체 | 2025 | 2024 |")
    L.append("|---|---|---|---|---|---|---|")
    f2 = lambda v: f"{pp(v['효율선초과'])} / {pp(v['고정초과'])}"  # noqa: E731
    for k, r in nb["대표"]["한칸씩"].items():
        lab = k.replace("w0=", "평소 ").replace("Tx=", "해제 ").replace("T=", "문턱 ")
        L.append(f"| {lab} | {f2(r['2026앞'])} | {f2(r['2026뒤'])} | {pc_(r['AI쇼크']['수익'])} | {f2(r['2026전체'])} | {f2(r['2025'])} | {f2(r['2024'])} |")
    for k, v in nbi.items():
        r = v["대표"]
        lab = (k.replace("pct=", "백분위 창 ").replace("ma=", "이동평균 ").replace("hl=", "신고가 창 ").replace("a5=", "A5 평균 ")
               .replace("요소=", "요소 "))
        L.append(f"| 지수: {lab} | {f2(r['2026앞'])} | {f2(r['2026뒤'])} | {pc_(r['AI쇼크']['수익'])} | {f2(r['2026전체'])} | {f2(r['2025'])} | {f2(r['2024'])} |")
    L.append("\n정육면체(모수 3개 × 5칸 = 125개) 요약: 칸 = 효율선초과 > 0인 개수 (중앙값)\n")
    L.append("| 규칙 | 뒤 통과 | 2026 앞 | 2026 뒤 | 2026 전체 | 2025 | 2024 |")
    L.append("|---|---|---|---|---|---|---|")
    for nm, cb in (("대표 S1", cube), ("부후보 S2", cube2)):
        cnt = lambda w: f"{sum(r[w]['효율선초과'] > 0 for r in cb.values())} ({pp(float(np.median([r[w]['효율선초과'] for r in cb.values()])))})"  # noqa: E731
        L.append(f"| {nm} | {sum(_ok(r) for r in cb.values())}/125 | {cnt('2026앞')} | {cnt('2026뒤')} | {cnt('2026전체')} | {cnt('2025')} | {cnt('2024')} |")
    L.append("\n- 뒤 구간은 이웃 모두가 통과한다. 모수 선택에는 튼튼하다는 뜻이지만, 모든 이웃이 같은 두 날(7/13·8/19 종가 공포)을 잡았기 때문이기도 하다.")
    L.append("- 2026 전체는 대표 이웃 125개 중 소수만 효율선 위다. '평소 60%'라는 바닥이 상승장 수익을 깎는다.\n")

    # ---------------- 고르기 표(판정)
    L.append("## 고르기 표 (판정)\n")
    L.append("위약: 원형 밀기 = 지수를 63×k일(k=1..20) 밀어 같은 규칙. 무작위 날짜 = 그 구간 실제 '평소와 다른 구간'의 개수·길이를 그대로 두고 시작일만 무작위(평균 노출 같음). 순위는 21개 중(1위가 가장 좋음).\n")
    L.append("| 판정 | 규칙 | 구간 | 실제 효율선초과 | 원형 밀기 순위 | 무작위 날짜 순위 | 무작위 2,000개 중 더 나은 비율 | 바뀐 구간 수 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nm, lab in (("대표", "대표 S1"), ("부후보", "부후보 S2")):
        for w in ("2026앞", "2026뒤", "2026전체", "2025", "2024"):
            p = plac[nm][w]
            j = "2026" if w.startswith("2026") else "보조"
            L.append(f"| {j} | {lab} | {w} | {pp(p['원형밀기']['실제'])} | {p['원형밀기']['순위']}/21 | {p['무작위날짜']['순위']}/21 | "
                     f"{p['무작위날짜']['큰비율'] * 100:.0f}% | {p['무작위날짜']['구간수']} |")
    L.append("\n시대별(긴 역사 대신). 7종목 경로는 2023-05부터라 그 전은 대리 대상: AI 유니버스 120일 모멘텀 상위 7 동일가중 주간 교체(7종목과 일별 상관 "
             f"2026 {P['대리상관']['2026']['AI7대리']:.2f}·2023~2025 {P['대리상관']['hold']['AI7대리']:.2f}), SMH(상관 {P['대리상관']['2026']['SMH']:.2f}·{P['대리상관']['hold']['SMH']:.2f}). "
             "칸 = 수익/낙폭, 고정 대비(같은 평균 노출), 구간 %p.\n")
    L.append("| 구간 | 대상 | 그대로 | 대표(쏠림) | 대표 고정 대비 | 대표 효율선초과 | CNN판 고정 대비 | 부후보(쏠림) 고정 대비 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for era, row in eras.items():
        for tn, c in row.items():
            a = c.get("쏠림")
            L.append(f"| {era} | {tn} | {_cell(c['기본'], False)} | {_cell(a, False) if a else '지수 없음'} | {pp(a['고정초과']) if a else '—'} | "
                     f"{pp(a['효율선초과']) if a else '—'} | {pp(c['CNN']['고정초과']) if c.get('CNN') else '—'} | "
                     f"{pp(c['쏠림_부후보']['고정초과']) if c.get('쏠림_부후보') else '—'} |")
    L.append("\n- 2026 판정: 사전 기준(뒤 효율선초과 > 0, 고정 대비 > 0)은 재현되고 이웃·하루 늦춤·미끄러짐에도 남는다. 그러나 무작위 날짜 위약과 구별되지 않고(29%), 이익이 3/9·8/20 두 번에서 나온다.")
    L.append("- 긴 역사 판정: 쏠림판은 2013년부터라 2011·2008 등은 잴 수 없다. 대리 대상(AI7·SMH) 고정 대비는 2015~16 +1.6·−0.0%p, 2018 4분기 +1.8·+2.3%p, "
             "2020 −9.1·+3.2%p, 2022 −3.2·−1.8%p, 2025 −11.0·−9.0%p(실제 7종목은 +5.0%p)로 일관된 이익이 없다. "
             "공포 때 100%로 올리는 규칙은 하락이 몇 달 이어지면(2022) 일찍 들어가고, 큰 반등 뒤에는(2020 AI7) 평소 60%가 수익을 깎는다(장세 전환 위험).\n")

    # ---------------- 한계
    L.append("## 한계\n")
    L.append("- 7종목 실제 경로는 2023-05~2025와 2026뿐이다. 그 전 해는 대리 대상이라 반도체 50% 한도·TIME 편입 종목이 반영되지 않는다(2026 뒤 대리 −11% 대 실제 +5%).")
    L.append("- AI 유니버스는 2026-10 목록(생존편향). 시장폭(A2·A3)이 옛날일수록 지금 살아남은 종목으로만 잰 값이다.")
    L.append("- 무작위 날짜 위약은 구간 안에서 100% 구간을 옮긴다. 2026 앞은 바뀐 구간이 1개라 위약 분포가 거칠다. 원형 밀기는 다른 해 지수를 가져오므로 평균 노출이 달라진다.")
    L.append("- 효율선은 같은 최대 낙폭의 고정 비중 대비다. 손실 구간(7월 쇼크)에서는 현금이 효율선이 되어 모든 규칙이 마이너스로 나온다 → 고정 대비를 함께 봤다.")
    L.append("- 이웃·위약·시대 격자는 이 파일 맨 위 상수로 미리 정했고 결과를 본 뒤 바꾸지 않았다. 시험 칸 수: 규칙 이웃 274개, 지수 이웃 19개, 위약 20+2,000개 × 구간 5 × 규칙 2, 시대 9 × 대상 3. 매수 추천·주문 수량이 아니다.")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("저장", RESULTS / f"{ID}.md")


if __name__ == "__main__":
    if sys.argv[1:] == ["md"]:
        write_md(json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8")))
    else:
        main()
