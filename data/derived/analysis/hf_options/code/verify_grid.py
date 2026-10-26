"""verify_grid — 격자 칸 독립 재계산 (harness.py·grid.py를 쓰지 않음).

원자료 캐시에서 결과(QQQ R/M), 백분위 신호, 칸 통계(front/back/y2025 차이·신호일 수·Welch t)를
직접 다시 만들고 results/grid_cells.parquet와 비교한다.

정의(BRIEF.md·QA 메모 기준, 내 해석):
- 달력: options_sentiment yahoo_ohlc의 QQQ 거래일.
- 진입일 e0: 장 마감 뒤 공개 일별 자료(CBOE·FINRA·DIX·ProShares 흐름)는 측정일 다음 거래일,
  그 밖(CFTC·AAII·Yahoo 가격 지표)은 known 다음 거래일.
- 백분위: 원자료 자기 빈도 행 기준 지난 N행(자기 포함, 일 250 / 주 52), 최소 N/2.
  기본 = 평균 순위 / n (pandas rank pct). 변형 = (순위-1)/(n-1).
- 묵은 값: e0부터 일별 3거래일, 주간 15거래일까지 이어 쓴다(기본: 나이 <= 한도).
- 결과: R_h = C(d+h-1)/O(d)-1, M_h = [O(d), C(d..d+h-1)] 경로의 고점 대비 최저(%).
- 칸: 단독 >=0.8 / <=0.2, 쌍 각각 >=0.7 / <=0.3. 차이 = 신호일 평균 - 나머지 평균.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]           # hf_options
OS_CACHE = ROOT.parent / "options_sentiment" / "cache"
CACHE = ROOT / "cache"
RES = ROOT / "results"

PERIODS = {
    "front": ("2026-01-02", "2026-05-29"),
    "back": ("2026-06-01", "2099-12-31"),
    "y2025": ("2025-01-01", "2025-12-31"),
}

# ---------------------------------------------------------------- calendar / outcomes

def load_ohlc():
    o = pd.read_parquet(OS_CACHE / "yahoo_ohlc.parquet")
    o = o.dropna(subset=["QQQ_o", "QQQ_c"])
    o.index = pd.Index([str(i)[:10] for i in o.index])
    return o.sort_index()


def outcomes(o: pd.DataFrame, tk: str = "QQQ", hs=(5, 10, 21)) -> pd.DataFrame:
    op = o[f"{tk}_o"].to_numpy(float)
    cl = o[f"{tk}_c"].to_numpy(float)
    n = len(op)
    out = {}
    for h in hs:
        R = np.full(n, np.nan)
        M = np.full(n, np.nan)       # 고점 대비 최저 (peak-to-trough)
        M2 = np.full(n, np.nan)      # 진입가 대비 최저 (min close / open - 1, 0 상한)
        for i in range(n):
            j = i + h - 1
            if j >= n:
                continue
            R[i] = (cl[j] / op[i] - 1) * 100
            path = np.concatenate([[op[i]], cl[i:j + 1]])
            peak = np.maximum.accumulate(path)
            M[i] = (path / peak - 1).min() * 100
            M2[i] = min(0.0, path[1:].min() / op[i] - 1) * 100
        out[f"R{h}"] = R
        out[f"M{h}"] = M
        out[f"Mo{h}"] = M2
    return pd.DataFrame(out, index=o.index)


# ---------------------------------------------------------------- percentile

def rolling_pct(x: pd.Series, n: int, method: str = "avg") -> pd.Series:
    """x: 원자료 자기 빈도 행(결측 없음). 지난 n행 안 자기 순위."""
    v = x.to_numpy(float)
    out = np.full(len(v), np.nan)
    mp = n // 2
    for i in range(len(v)):
        lo = max(0, i - n + 1)
        w = v[lo:i + 1]
        m = len(w)
        if m < mp:
            continue
        less = (w < v[i]).sum()
        eq = (w == v[i]).sum()
        if method == "avg":            # pandas rank(pct=True, method='average')
            out[i] = (less + (eq + 1) / 2) / m
        elif method == "le":           # 이하 비율
            out[i] = (less + eq) / m
        elif method == "lt":           # 미만 비율
            out[i] = less / m
        elif method == "n1":           # (순위-1)/(n-1), 평균 순위
            out[i] = (less + (eq - 1) / 2) / (m - 1) if m > 1 else np.nan
        else:
            raise ValueError(method)
    return pd.Series(out, index=x.index)


# ---------------------------------------------------------------- mapping to entry calendar

def to_entry(vals: pd.Series, avail: pd.Series, cal: pd.Index, limit: int, inclusive: bool = True) -> pd.Series:
    """vals·avail: 같은 색인(원자료 행 순서). avail = 공개 기준일(문자열).
    e0 = avail 다음 거래일. 진입일 d에는 e0 <= d인 마지막 행을 쓰고, 나이(거래일)가 한도를 넘으면 비운다."""
    calv = np.array(cal, dtype=object)
    av = avail.astype(str).str[:10].to_numpy(dtype=object)
    e0 = np.searchsorted(calv, av, side="right")     # 다음 거래일 위치
    v = vals.to_numpy(float)
    ok = ~np.isnan(v)
    e0, v = e0[ok], v[ok]
    # 같은 e0이면 원자료 순서상 마지막 행
    out = np.full(len(calv), np.nan)
    order = np.argsort(e0, kind="stable")
    e0s, vs = e0[order], v[order]
    for d in range(len(calv)):
        k = np.searchsorted(e0s, d, side="right") - 1
        if k < 0:
            continue
        age = d - e0s[k]
        if (age <= limit) if inclusive else (age < limit):
            out[d] = vs[k]
    return pd.Series(out, index=cal)


# ---------------------------------------------------------------- raw signals (native frequency)

def raw_signals(chg_mode: str = "ratio"):
    """반환: {이름: (값 Series(원자료 행), 공개 기준일 Series, 빈도 'd'|'w')}"""
    sig = {}

    # CBOE 주식 옵션 풋/콜 5일 합 (측정일 D, 장 마감 뒤 → D 다음 거래일)
    rows = []
    for f in sorted(os.listdir(OS_CACHE / "cboe")):
        j = json.load(open(OS_CACHE / "cboe" / f, encoding="utf-8"))
        vol = [r for r in j["EQUITY OPTIONS"] if r["name"] == "VOLUME"][0]
        rows.append((f[:-5], float(vol["put"]), float(vol["call"])))
    cb = pd.DataFrame(rows, columns=["d", "put", "call"]).set_index("d").sort_index()
    eq = cb["put"].rolling(5).sum() / cb["call"].rolling(5).sum()
    eq = eq.dropna()
    sig["eq_pcr5"] = (eq, pd.Series(eq.index, index=eq.index), "d")

    # FINRA shvr_ai_eq 5일 평균 (asof D, 장 마감 뒤 → D 다음 거래일)
    fs = pd.read_parquet(CACHE / "finra_shvol.parquet")
    fs = fs.reset_index().sort_values("asof").set_index("asof")
    sh = fs["shvr_ai_eq"].astype(float).rolling(5).mean().dropna()
    sig["shvr_ai5"] = (sh, pd.Series(sh.index, index=sh.index), "d")

    # CBOE SKEW (Yahoo ^SKEW 종가, known = 그날 → 다음 거래일)
    y = pd.read_parquet(OS_CACHE / "yahoo.parquet")
    y.index = pd.Index([str(i)[:10] for i in y.index])
    sk = y["^SKEW"].dropna()
    sig["skew"] = (sk, pd.Series(sk.index, index=sk.index), "d")

    # DIX (asof D, 장 마감 뒤 → D 다음 거래일)
    sq = pd.read_parquet(CACHE / "squeeze.parquet").reset_index().sort_values("asof").set_index("asof")
    dx = sq["dix"].astype(float).dropna()
    sig["dix"] = (dx, pd.Series(dx.index, index=dx.index), "d")

    # CFTC TFF (known 다음 거래일), 주간 행
    c = pd.read_parquet(CACHE / "cftc_tff.parquet").sort_index()
    kn = pd.Series(c.index, index=c.index)
    r_es_lev = c["es_lev_net"] / c["es_oi"]
    r_vx_lev = c["vx_lev_net"] / c["vx_oi"]
    r_nq_am = c["nq_am_net"] / c["nq_oi"]
    r_nq_lev = c["nq_lev_net"] / c["nq_oi"]
    if chg_mode == "ratio":           # 비율의 4주 변화
        nq_am_chg4 = r_nq_am - r_nq_am.shift(4)
        nq_lev_chg4 = r_nq_lev - r_nq_lev.shift(4)
    else:                             # 순포지션 4주 변화 ÷ 현재 OI
        nq_am_chg4 = (c["nq_am_net"] - c["nq_am_net"].shift(4)) / c["nq_oi"]
        nq_lev_chg4 = (c["nq_lev_net"] - c["nq_lev_net"].shift(4)) / c["nq_oi"]
    for nm, s in [("es_lev_net_oi", r_es_lev), ("vx_lev_net_oi", r_vx_lev),
                  ("nq_am_chg4", nq_am_chg4), ("nq_lev_chg4", nq_lev_chg4)]:
        s = s.dropna()
        sig[nm] = (s, kn.loc[s.index], "w")

    # AAII 강세-약세 (known 다음 거래일), 주간 행
    sv = pd.read_parquet(CACHE / "surveys.parquet").sort_index()
    aa = sv["aaii_spread"].dropna()
    sig["aaii_spread"] = (aa, pd.Series(aa.index, index=aa.index), "w")

    # ProShares TQQQ-SQQQ 순유입 20일 합 ÷ 두 펀드 순자산 (asof T, 장 마감 뒤 → T 다음 거래일)
    hp = pd.read_parquet(CACHE / "hf_proxies.parquet").sort_index()
    fl = hp[["asof", "flow_tqqq", "flow_sqqq", "aum_tqqq", "aum_sqqq"]].dropna()
    fl = fl.set_index("asof").sort_index()
    f20 = (fl["flow_tqqq"] - fl["flow_sqqq"]).rolling(20).sum()
    fp = (f20 / (fl["aum_tqqq"] + fl["aum_sqqq"])).dropna()
    sig["lev_qqq_flow20p"] = (fp, pd.Series(fp.index, index=fp.index), "d")

    # GVIP/QQQ 수정종가 비율 20일 변화율 (Yahoo 가격, known = 그날 → 다음 거래일). 종목별 표에서 다시 계산
    bs = pd.read_parquet(CACHE / "hf_proxies_by_symbol.parquet")
    px = bs.pivot_table(index="known", columns="symbol", values="adj_close").sort_index()
    rel = (px["GVIP"] / px["QQQ"]).dropna()
    g20 = (rel / rel.shift(20) - 1).dropna()
    sig["gvip_rel20"] = (g20, pd.Series(g20.index, index=g20.index), "d")
    return sig, hp


def build_panel(cal, pct_method="avg", stale_incl=True, chg_mode="ratio", limits=None):
    limits = limits or {"d": 3, "w": 15}
    sig, hp = raw_signals(chg_mode)
    win = {"d": 250, "w": 52}
    P = {}
    for nm, (vals, avail, fq) in sig.items():
        pc = rolling_pct(vals, win[fq], pct_method)
        P[nm] = to_entry(pc, avail, cal, limits[fq], stale_incl)
    return pd.DataFrame(P, index=cal), sig, hp


# ---------------------------------------------------------------- cells

def parse_cell(cell):
    parts = cell.split("&")
    out = []
    for p in parts:
        out.append((p[:-1], p[-1]))
    return out


def cell_mask(panel: pd.DataFrame, cell: str):
    parts = parse_cell(cell)
    single = len(parts) == 1
    hi, lo = (0.8, 0.2) if single else (0.7, 0.3)
    on = pd.Series(True, index=panel.index)
    defined = pd.Series(True, index=panel.index)
    for nm, d in parts:
        x = panel[nm]
        defined &= x.notna()
        on &= (x >= hi) if d == "+" else (x <= lo)
    return on, defined


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def cell_stats(panel, outc, cell, h, y, rest="all", mcol="M"):
    on, defined = cell_mask(panel, cell)
    col = f"{y}{h}" if y == "R" else f"{mcol}{h}"
    res = {"cell": cell, "h": h, "y": y}
    for per, (a, b) in PERIODS.items():
        idx = (panel.index >= a) & (panel.index <= b)
        yy = outc[col]
        base = idx & yy.notna().to_numpy()
        s = base & on.to_numpy()
        r = base & ~on.to_numpy()
        if rest == "defined":
            r &= defined.to_numpy()
        sv, rv = yy[s].to_numpy(), yy[r].to_numpy()
        res[f"{per}_n"] = int(s.sum())
        res[f"{per}_d"] = (sv.mean() - rv.mean()) if len(sv) and len(rv) else np.nan
        res[f"{per}_t"] = welch(sv, rv)
    return res


TARGETS = [
    ("eq_pcr5+&shvr_ai5-", 21, "M"),
    ("skew+&shvr_ai5-", 21, "M"),
    ("skew+&nq_am_chg4-", 21, "M"),
    ("es_lev_net_oi+", 21, "R"),
    ("nq_lev_chg4-", 10, "M"),
    ("dix-", 5, "R"),
    ("aaii_spread+&lev_qqq_flow20p+", 21, "R"),
    ("gvip_rel20-&vx_lev_net_oi+", 10, "M"),
]


def canon(cell, names):
    """grid_cells의 칸 이름(쌍 순서)과 맞춘다."""
    if cell in names:
        return cell
    p = cell.split("&")
    if len(p) == 2 and f"{p[1]}&{p[0]}" in names:
        return f"{p[1]}&{p[0]}"
    return None


def compare(variant: dict, gc: pd.DataFrame, cal, outc):
    panel, _, _ = build_panel(cal, variant.get("pct", "avg"), variant.get("incl", True),
                              variant.get("chg", "ratio"), variant.get("limits"))
    names = set(gc["cell"])
    rows = []
    for cell, h, y in TARGETS:
        st = cell_stats(panel, outc, cell, h, y, variant.get("rest", "all"), variant.get("mcol", "M"))
        for per in PERIODS:                      # grid 관례: 신호일 3일 미만이면 차이·t 비움
            if st[f"{per}_n"] < 3:
                st[f"{per}_d"] = st[f"{per}_t"] = np.nan
        cn = canon(cell, names)
        g = gc[(gc.cell == cn) & (gc.h == h) & (gc.y == y)].iloc[0]
        row = {"cell": cell, "grid_name": cn, "h": h, "y": y}
        for per in PERIODS:
            for k in ("d", "t", "n"):
                row[f"{per}_{k}_mine"] = st[f"{per}_{k}"]
                row[f"{per}_{k}_grid"] = g[f"{per}_{k}"]
        rows.append(row)
    return pd.DataFrame(rows), panel


def maxdiff(df):
    out = {}
    for _, r in df.iterrows():
        dd = []
        dn = []
        dt = []
        for per in PERIODS:
            a, b = r[f"{per}_d_mine"], r[f"{per}_d_grid"]
            if not (pd.isna(a) and pd.isna(b)):
                dd.append(abs(a - b) if not (pd.isna(a) or pd.isna(b)) else np.inf)
            a, b = r[f"{per}_t_mine"], r[f"{per}_t_grid"]
            if not (pd.isna(a) and pd.isna(b)):
                dt.append(abs(a - b) if not (pd.isna(a) or pd.isna(b)) else np.inf)
            dn.append(abs(int(r[f"{per}_n_mine"]) - int(r[f"{per}_n_grid"])))
        out[f"{r['cell']}|h{r['h']}{r['y']}"] = {
            "max_abs_d": float(max(dd)) if dd else 0.0,
            "max_abs_t": float(max(dt)) if dt else 0.0,
            "max_abs_n": int(max(dn)),
        }
    return out


def definition_variants():
    """정의 민감도: 관례 맞춘 달력(10-12까지)·나머지=정의된 날 위에서 한 가지씩 바꿔 grid와의 차이."""
    o = load_ohlc().loc[:"2026-10-12"]
    cal = pd.Index(o.index)
    outc = outcomes(o, "QQQ")
    gc = pd.read_parquet(RES / "grid_cells.parquet")
    base = {"rest": "defined"}
    variants = {
        "aligned": {},
        "rest_all": {"rest": "all"},
        "M_entry_based": {"mcol": "Mo"},
        "pct_le": {"pct": "le"},
        "pct_n1": {"pct": "n1"},
        "stale_lt": {"incl": False},
        "chg4_netdiff_over_oi": {"chg": "netdiff"},
    }
    res = {}
    for nm, var in variants.items():
        df, _ = compare({**base, **var}, gc, cal, outc)
        md = maxdiff(df)
        res[nm] = {"max_abs_d_over_cells": max(v["max_abs_d"] for v in md.values()),
                   "max_abs_n_over_cells": max(v["max_abs_n"] for v in md.values()),
                   "cells_exact(d<1e-9,n=0)": sum(1 for v in md.values() if v["max_abs_d"] < 1e-9 and v["max_abs_n"] == 0)}
    return res


# ---------------------------------------------------------------- extra checks (my own design)

def episodes(days, cal, gap=5):
    """신호일을 거래일 간격 gap 이하로 이어지면 한 묶음으로."""
    eps = []
    for d in days:
        p = cal.get_loc(d)
        if eps and p - eps[-1]["last_pos"] <= gap:
            eps[-1].update(last=d, last_pos=p, n=eps[-1]["n"] + 1)
        else:
            eps.append({"first": d, "last": d, "last_pos": p, "n": 1})
    return [{k: e[k] for k in ("first", "last", "n")} for e in eps]


def shift_p(panel, outc, cell, h, y, side, rest="defined", min_shift=21):
    """내 순열 설계: 2026 진입일 안에서 신호열(켜짐·정의 마스크)을 원형 이동(모든 s in [min_shift, N-min_shift])해
    back 차이의 한쪽 p = (1 + 관측 이상으로 극단인 이동 수) / (1 + 유효 이동 수). 신호일·나머지 3일 미만 이동은 뺀다."""
    yr = panel.index[panel.index >= "2026-01-01"]
    on, defined = cell_mask(panel.loc[yr], cell)
    on, defined = on.to_numpy(), defined.to_numpy()
    yy = outc.loc[yr, f"{y}{h}"]
    back = (np.asarray(yr) >= "2026-06-01") & yy.notna().to_numpy()

    def stat(on_, def_):
        s_ = back & on_
        r_ = back & ~on_
        if rest == "defined":
            r_ &= def_
        if s_.sum() < 3 or r_.sum() < 3:
            return np.nan
        return yy[s_].mean() - yy[r_].mean()

    obs = stat(on, defined)
    N = len(yr)
    null = np.array([stat(np.roll(on, k), np.roll(defined, k)) for k in range(min_shift, N - min_shift + 1)])
    null = null[~np.isnan(null)]
    if np.isnan(obs) or not len(null):
        return obs, np.nan, int(len(null))
    ext = (null <= obs).sum() if side == "low" else (null >= obs).sum()
    return obs, (1 + ext) / (1 + len(null)), int(len(null))


def _f(x):
    return None if pd.isna(x) else float(x)


# ---------------------------------------------------------------- selection-level re-implementation (BRIEF '판정' 문장 기준)

def all_cells(P: pd.DataFrame, rule: str):
    """단독 48 + 쌍 1,104칸(이름 순서는 BRIEF 이름 규칙). rule='brief' = '-'를 <= 0.2 / <= 0.3(경계 포함),
    rule='float' = 1-0.8(=0.19999999999999996)·1-0.7(=0.30000000000000004)을 그대로 문턱으로(grid의 부동소수 결과 재현)."""
    lo1 = 0.2 if rule == "brief" else 1 - 0.8
    lo2 = 0.3 if rule == "brief" else 1 - 0.7
    X = P.to_numpy(float)
    cols, names = [], []
    for j, s in enumerate(P.columns):
        x = X[:, j]
        for sg, cond in (("+", x >= 0.8), ("-", x <= lo1)):
            cols.append(np.where(np.isnan(x), np.nan, cond.astype(float)))
            names.append(f"{s}{sg}")
    m = len(P.columns)
    for a in range(m):
        for b in range(a + 1, m):
            xa, xb = X[:, a], X[:, b]
            na = np.isnan(xa) | np.isnan(xb)
            for sa, ca in (("+", xa >= 0.7), ("-", xa <= lo2)):
                for sb, cb in (("+", xb >= 0.7), ("-", xb <= lo2)):
                    cols.append(np.where(na, np.nan, (ca & cb).astype(float)))
                    names.append(f"{P.columns[a]}{sa}&{P.columns[b]}{sb}")
    return np.column_stack(cols), names


def vec_stats(S, y, mask, min_n=3):
    S = np.where(mask[:, None], S, np.nan)
    ok = ~np.isnan(S) & ~np.isnan(y)[:, None]
    sig = (np.nan_to_num(S) > 0) & ok
    rst = ok & ~sig
    yy = np.nan_to_num(y)[:, None]
    n1, n0 = sig.sum(0), rst.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m0 = (yy * sig).sum(0) / n1, (yy * rst).sum(0) / n0
        v1 = ((yy - m1) ** 2 * sig).sum(0) / (n1 - 1)
        v0 = ((yy - m0) ** 2 * rst).sum(0) / (n0 - 1)
        d = m1 - m0
        t = d / np.sqrt(v1 / n1 + v0 / n0)
    bad = (n1 < min_n) | (n0 < min_n)
    d[bad] = np.nan
    t[bad] = np.nan
    return d, t, n1


def p_shift_vec(sig, y, side, min_shift=21):
    """sig·y: 같은 길이(NaN = 제외). 남은 날 안에서 원형 이동(모든 k in [21, n-21])."""
    m = ~np.isnan(sig) & ~np.isnan(y)
    s, v = sig[m] > 0, y[m]
    n = len(s)
    if s.sum() < 3 or n < 3 * min_shift:
        return np.nan, 0
    obs = v[s].mean() - v[~s].mean()
    null = []
    for k in range(min_shift, n - min_shift + 1):
        r = np.roll(s, k)
        if 0 < r.sum() < n:
            null.append(v[r].mean() - v[~r].mean())
    null = np.array(null)
    hit = (null <= obs) if side == "low" else (null >= obs)
    return (hit.sum() + 1) / (len(null) + 1), len(null)


def p_shift_2026(sig26, y26, back, side, min_shift=21):
    """대안: 2026 전체 신호열을 원형 이동하고 back 날의 차이만 잰다(신호일·나머지 3일 미만 이동은 뺌)."""
    defined = ~np.isnan(sig26)
    on = np.nan_to_num(sig26) > 0
    yok = back & ~np.isnan(y26)

    def st(o_, d_):
        s_ = yok & o_ & d_
        r_ = yok & ~o_ & d_
        if s_.sum() < 3 or r_.sum() < 3:
            return np.nan
        return y26[s_].mean() - y26[r_].mean()

    obs = st(on, defined)
    N = len(on)
    null = np.array([st(np.roll(on, k), np.roll(defined, k)) for k in range(min_shift, N - min_shift + 1)])
    null = null[~np.isnan(null)]
    if np.isnan(obs) or not len(null):
        return np.nan, 0
    hit = (null <= obs) if side == "low" else (null >= obs)
    return (hit.sum() + 1) / (len(null) + 1), len(null)


def selection(P, Y, rule, topk=20, min_front=5):
    S, names = all_cells(P, rule)
    idx = P.index
    y26 = np.asarray(idx >= "2026-01-02")
    S26, i26 = S[y26], idx[y26]
    fr = np.asarray((i26 >= "2026-01-02") & (i26 <= "2026-05-29"))
    bk = np.asarray(i26 >= "2026-06-01")
    recs = []
    for h in (5, 10, 21):
        for y in ("R", "M"):
            yy = Y.loc[i26, f"{y}{h}"].to_numpy(float)
            d, t, n = vec_stats(S26, yy, fr)
            recs.append(pd.DataFrame({"k": np.arange(len(names)), "h": h, "y": y, "t": t, "n": n, "d": d}))
    g = pd.concat(recs, ignore_index=True)
    g = g[(g["n"] >= min_front) & g["t"].notna()]
    out = {}
    for side in ("top", "bottom"):
        sub = g if side == "top" else g[g["y"] == "R"]
        pick = sub.nsmallest(topk, "t") if side == "top" else sub.nlargest(topk, "t")
        rows = []
        for _, r in pick.iterrows():
            k, h, y = int(r["k"]), int(r["h"]), r["y"]
            yy = Y.loc[i26, f"{y}{h}"].to_numpy(float)
            sig = S26[:, k]
            bs, by = np.where(bk, sig, np.nan), np.where(bk, yy, np.nan)
            d, _, nb = vec_stats(bs[:, None], by, np.ones(len(by), bool))
            d, nb = float(d[0]), int(nb[0])
            want = (d < 0) if side == "top" else (d > 0)
            pdir = "low" if side == "top" else "high"
            p_b, ns_b = p_shift_vec(bs, by, pdir) if (nb >= 3 and want) else (np.nan, 0)
            p_26, ns_26 = p_shift_2026(sig, yy, bk, pdir) if (nb >= 3 and want) else (np.nan, 0)
            rows.append({"cell": names[k], "h": h, "y": y, "front_t": float(r["t"]), "front_n": int(r["n"]),
                         "back_d": _f(d), "back_n": nb, "p_backonly": _f(p_b), "n_shifts_backonly": ns_b,
                         "p_2026shift": _f(p_26), "n_shifts_2026": ns_26,
                         "confirmed_backonly": bool(want and p_b < 0.05),
                         "confirmed_2026shift": bool(want and p_26 < 0.05)})
        out[side] = rows
    return out


def run():
    gc = pd.read_parquet(RES / "grid_cells.parquet")
    gj = json.load(open(RES / "grid.json", encoding="utf-8"))
    picks = {(p["cell"], p["h"], p["y"]): p for side in ("top", "bottom") for p in gj["picks"][side]}
    names = set(gc["cell"])
    o_full = load_ohlc()
    out = {"specs": {}}
    specs = {
        # A: 내 해석 그대로 — 가격 달력 2026-10-13까지, 나머지 = 신호 아닌 모든 결과일, 신호일 1일 이상이면 차이 보고
        "A_literal": {"o": o_full, "rest": "all", "nmin": 1},
        # B: 관례 맞춤 — 달력 2026-10-12까지, 나머지 = 신호가 정의된 날 중 꺼진 날, 신호일 3일 이상일 때만 차이
        "B_aligned": {"o": o_full.loc[:"2026-10-12"], "rest": "defined", "nmin": 3},
    }
    panels = {}
    for sn, sp in specs.items():
        cal = pd.Index(sp["o"].index)
        outc = outcomes(sp["o"], "QQQ")
        panel, _, _ = build_panel(cal)
        panels[sn] = (panel, outc, cal)
        rows, md = [], {}
        for cell, h, y in TARGETS:
            st = cell_stats(panel, outc, cell, h, y, sp["rest"])
            for per in PERIODS:
                if st[f"{per}_n"] < sp["nmin"]:
                    st[f"{per}_d"] = np.nan
                if st[f"{per}_n"] < max(sp["nmin"], 2):
                    st[f"{per}_t"] = np.nan
            cn = canon(cell, names)
            g = gc[(gc.cell == cn) & (gc.h == h) & (gc.y == y)].iloc[0]
            row = {"cell": cell, "grid_name": cn, "h": h, "y": y}
            dd, dt, dn, nan_mis = [], [], [], 0
            for per in PERIODS:
                for k in ("n", "d", "t"):
                    row[f"{per}_{k}"] = _f(st[f"{per}_{k}"])
                    row[f"{per}_{k}_grid"] = _f(g[f"{per}_{k}"])
                for k, lst in (("d", dd), ("t", dt)):
                    a, b = row[f"{per}_{k}"], row[f"{per}_{k}_grid"]
                    if (a is None) != (b is None):
                        nan_mis += 1
                    elif a is not None:
                        lst.append(abs(a - b))
                dn.append(abs(row[f"{per}_n"] - row[f"{per}_n_grid"]))
            rows.append(row)
            md[f"{cell} h{h} {y}"] = {"max_abs_d": max(dd) if dd else 0.0, "max_abs_t": max(dt) if dt else 0.0,
                                      "max_abs_n": int(max(dn)), "nan_mismatch": nan_mis}
        out["specs"][sn] = {"rows": rows, "maxdiff": md}

    out["definition_variants"] = definition_variants()

    # 대상 칸: 신호일 묶음, 내 순열 p (B 관례 위에서)
    panel, outc, cal = panels["B_aligned"]
    conf = []
    for cell, h, y in TARGETS:
        cn = canon(cell, names)
        pk = picks.get((cn, h, y))
        g = gc[(gc.cell == cn) & (gc.h == h) & (gc.y == y)].iloc[0]
        side = "low" if g["front_d"] < 0 else "high"    # front 방향 = 가설 방향
        on, _ = cell_mask(panel, cell)
        info = {"cell": cell, "h": h, "y": y, "in_picks": pk is not None,
                "grid_confirmed": bool(pk["confirmed"]) if pk else None,
                "grid_back_p": _f(pk["back_p"]) if pk else None, "side": side}
        for per, (a, b) in PERIODS.items():
            m = on & (panel.index >= a) & (panel.index <= b) & outc[f"{y}{h}"].notna()
            info[f"{per}_episodes"] = episodes(list(panel.index[m]), cal)
        obs, p, ns = shift_p(panel, outc, cell, h, y, side)
        info.update({"my_back_d": _f(obs), "my_back_p_shift": _f(p), "my_n_shifts": ns})
        conf.append(info)
    out["cell_checks"] = conf

    # 고점 쪽 확인 3칸이 같은 날을 공유하는지
    top3 = ["eq_pcr5+&shvr_ai5-", "skew+&shvr_ai5-", "skew+&nq_am_chg4-"]
    ov = {}
    for per in ("front", "back"):
        a, b = PERIODS[per]
        sets = {}
        for c in top3:
            on, _ = cell_mask(panel, c)
            m = on & (panel.index >= a) & (panel.index <= b) & outc["M21"].notna()
            sets[c] = set(panel.index[m])
        ov[per] = {"n": {c: len(s) for c, s in sets.items()}, "union": len(set().union(*sets.values())),
                   "all_three": len(set.intersection(*sets.values())),
                   "union_days": sorted(set().union(*sets.values()))}
    fr_days = ov["front"]["union_days"]
    ov["front_M21_on_union_days"] = {d: round(float(outc.loc[d, "M21"]), 2) for d in fr_days}
    out["top3_overlap"] = ov

    # 사후: 내 패널과 harness panel.parquet 비교(2025~, 내가 만든 11개 신호)
    hp = pd.read_parquet(CACHE / "panel.parquet")
    idx = [d for d in panel.index if d >= "2025-01-01" and d in hp.index]
    pc = {}
    for c in panel.columns:
        a, b = panel.loc[idx, c], hp.loc[idx, c].astype(float)
        both = a.notna() & b.notna()
        diff = (a[both] - b[both]).abs()
        flips = {}
        for th in (0.2, 0.3, 0.7, 0.8):
            ma = (a[both] <= th) if th < 0.5 else (a[both] >= th)
            mb = (b[both] <= th) if th < 0.5 else (b[both] >= th)
            flips[str(th)] = int((ma != mb).sum())
        pc[c] = {"max_abs": _f(diff.max()) if len(diff) else None, "n_diff": int((diff > 1e-12).sum()),
                 "nan_mismatch": int((a.isna() != b.isna()).sum()), "threshold_flips": flips,
                 "days_pct_exactly_0.2": [d for d in idx if pd.notna(a[d]) and abs(a[d] - 0.2) < 1e-12],
                 "days_pct_exactly_0.3": int(sum(1 for d in idx if pd.notna(a[d]) and abs(a[d] - 0.3) < 1e-12))}
    out["panel_vs_harness"] = pc

    # 고르기·확인 단계 재구현(내 코드, harness 패널 24신호 + 내 결과 B). grid.json 고른 칸과 비교, 문턱 규칙·순열 설계 민감도
    sel = {}
    for rule in ("float", "brief"):
        sel[rule] = selection(hp, outc, rule)
    gpk = {side: [(p["cell"], p["h"], p["y"], bool(p["confirmed"])) for p in gj["picks"][side]] for side in ("top", "bottom")}
    cmp = {}
    for side in ("top", "bottom"):
        mine = [(r["cell"], r["h"], r["y"], r["confirmed_backonly"]) for r in sel["float"][side]]
        cmp[side] = {"same_picks_in_order": [m[:3] for m in mine] == [g[:3] for g in gpk[side]],
                     "same_confirmed_flags": mine == gpk[side],
                     "confirmed_backonly_float": sum(r["confirmed_backonly"] for r in sel["float"][side]),
                     "confirmed_2026shift_float": sum(r["confirmed_2026shift"] for r in sel["float"][side]),
                     "confirmed_backonly_brief": sum(r["confirmed_backonly"] for r in sel["brief"][side]),
                     "confirmed_2026shift_brief": sum(r["confirmed_2026shift"] for r in sel["brief"][side]),
                     "picks_changed_by_rule": [r["cell"] + f" h{r['h']} {r['y']}" for r in sel["brief"][side]
                                               if (r["cell"], r["h"], r["y"]) not in [(a, b, c) for a, b, c, _ in mine]]}
    out["selection"] = {"compare_to_grid_json": cmp, "rows": sel}
    json.dump(out, open(RES / "verify_grid.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return out


if __name__ == "__main__":
    res = run()
    for sn, v in res["specs"].items():
        print("====", sn)
        for k, m in v["maxdiff"].items():
            print(f"  {k:42s} d {m['max_abs_d']:.3g}  t {m['max_abs_t']:.3g}  n {m['max_abs_n']}  nan {m['nan_mismatch']}")
    print("==== definition variants (aligned base)")
    for k, v in res["definition_variants"].items():
        print("  ", k, v)
    print("==== cell checks")
    for c in res["cell_checks"]:
        print(c["cell"], c["h"], c["y"], "grid_p", c["grid_back_p"], "my_p", c["my_back_p_shift"],
              "shifts", c["my_n_shifts"], "side", c["side"])
        for per in PERIODS:
            print("   ", per, [(e["first"], e["last"], e["n"]) for e in c[f"{per}_episodes"]])
    print("==== selection re-implementation")
    for side, v in res["selection"]["compare_to_grid_json"].items():
        print("  ", side, v)
    for side in ("top", "bottom"):
        for r in res["selection"]["rows"]["float"][side]:
            if r["p_backonly"] is not None:
                print(f"   {side} {r['cell']:40s} h{r['h']} {r['y']} back_d {r['back_d']:+.3f} n{r['back_n']} "
                      f"p_back {r['p_backonly']:.4f}({r['n_shifts_backonly']}) p_2026 {r['p_2026shift']}({r['n_shifts_2026']})")
    print("==== panel vs harness")
    for k, v in res["panel_vs_harness"].items():
        print(f"  {k:16s} max {v['max_abs']}  ndiff {v['n_diff']}  nanmis {v['nan_mismatch']}  flips {v['threshold_flips']}"
              f"  eq0.2 {v['days_pct_exactly_0.2']}  eq0.3 {v['days_pct_exactly_0.3']}")
