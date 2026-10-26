"""지지·저항선(60분 정규장 스윙 저점·고점) 매수/목표 + 기존 최선 매수 자리에 저항선 목표를 더한 조합."""
import itertools
import numpy as np
import pandas as pd
from numba import njit
from core import *
from evalcfg import data
from grid_short import portfolio, window_metrics
from run_on import ctx, day_arrays, levels, FILTERS
from sim_on import sim_overnight

D, HALF = data()


@njit(cache=True)
def pivots(lo, hi, N):
    """확정된 스윙 저점·고점: 봉 t가 좌우 N봉 중 최저(최고)면 t+N봉이 끝난 뒤 알 수 있다. 반환: (확정 봉 번호, 가격)."""
    m = len(lo)
    pli = np.empty(m, np.int64)
    plp = np.empty(m)
    phi = np.empty(m, np.int64)
    php = np.empty(m)
    a, b = 0, 0
    for t in range(N, m - N):
        isl, ish = True, True
        for k in range(t - N, t + N + 1):
            if k == t:
                continue
            if lo[k] <= lo[t]:
                isl = False
            if hi[k] >= hi[t]:
                ish = False
        if isl:
            pli[a] = t + N
            plp[a] = lo[t]
            a += 1
        if ish:
            phi[b] = t + N
            php[b] = hi[t]
            b += 1
    return pli[:a], plp[:a], phi[:b], php[:b]


def sr_levels(S, s, N, lookback_days=10):
    """세션별 개장 전: 전일 종가 아래 가장 가까운 지지 S1, 그다음 S2, 위쪽 가장 가까운 저항 R1 (최근 lookback 세션 안의 확정 피벗)."""
    A = S[s]
    tb = tf_bars(D[(s, "1h")], 60, False, HALF)
    pli, plp, phi, php = pivots(tb.L.values, tb.H.values, N)
    end = tb.end_ns.values
    reg = A["is_reg"]
    df = pd.DataFrame({"d": A["day_idx"][reg], "c": A["c"][reg], "st": A["start_ns"][reg]})
    g = df.groupby("d")
    daily = pd.DataFrame({"c": g.c.last(), "st": g.st.first()})
    pc = daily.c.shift(1).values
    n_days = len(A["days"])
    S1, S2, R1 = (np.full(n_days, np.nan) for _ in range(3))
    for idx, (d, st) in enumerate(zip(daily.index.values, daily.st.values)):
        if np.isnan(pc[idx]):
            continue
        last = np.searchsorted(end, st, side="right") - 1          # 개장 전 마지막 완성 60분봉
        first = max(0, last - lookback_days * 7)
        lows = plp[(pli <= last) & (pli >= first)]
        highs = php[(phi <= last) & (phi >= first)]
        below = np.sort(lows[lows < pc[idx]])[::-1]
        above = np.sort(highs[highs > pc[idx]])
        if len(below) >= 1:
            S1[d] = below[0]
        if len(below) >= 2:
            S2[d] = below[1]
        if len(above) >= 1:
            R1[d] = above[0]
    return S1, S2, R1


def run(win, mcost=0.001, lcost=0.0, margin=0.0):
    S, common, sel = ctx(win)
    for s in SYMS:
        day_arrays(S, s)
    base = {s: levels(S, s, win) for s in SYMS}
    SR = {(s, N): sr_levels(S, s, N) for s in SYMS for N in (3, 5)}
    # 필터 마스크는 기존 levels의 '종가-1%' 후보에서 NaN 여부로 재사용할 수 없으니 따로 만든다
    filt = {}
    for s in SYMS:
        noflt = base[s][("종가-1%", "없음")]
        for fi in FILTERS:
            f = base[s][("종가-1%", fi)]
            filt[(s, fi)] = ~np.isnan(f) | np.isnan(noflt)   # 필터가 막은 날만 False
    rows = []
    entries = [("지지1(N3)", 3, 0), ("지지1(N5)", 5, 0), ("지지2(N3)", 3, 1), ("지지2(N5)", 5, 1),
               ("ATR0.5", None, None), ("종가-2%", None, None), ("전일저가", None, None)]
    tps = [("0.5%", None, 0.005), ("1%", None, 0.01), ("저항(N3)", 3, None), ("저항(N5)", 5, None)]
    for (en, eN, which), fi, (tn, tN, tg), hd in itertools.product(entries, FILTERS, tps, [1, 2]):
        eqs, ntr, nw, sr = [], 0, 0, 0.0
        for s in SYMS:
            A = S[s]
            if eN is None:
                lv = base[s][(en, fi)]
            else:
                lv = SR[(s, eN)][which].copy()
                lv = np.where(filt[(s, fi)], lv, np.nan)
            tpl = SR[(s, tN)][2] if tN else np.full(len(A["days"]), np.nan)
            e, t, w, r = sim_overnight(A["o"], A["h"], A["l"], A["c"], A["is_reg"], A["is_pre"], A["is_awake"], A["is_end"],
                                       A["first_reg"], A["day_idx"], A["j0"], lv, tg if tg else 0.0, hd, mcost, lcost, margin,
                                       tpl, 0.005, 0.03)
            eqs.append(e); ntr += t; nw += w; sr += r
        rows.append({"win": win, "entry": en, "filter": fi, "tp": tn, "hold": hd, "trades_per_day": ntr / len(common),
                     "win_rate": nw / ntr if ntr else np.nan, "avg_trade": sr / ntr if ntr else np.nan,
                     **window_metrics(portfolio(eqs, S), sel)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    kw = {"base": {}, "stress": dict(mcost=0.002, lcost=0.001, margin=0.001), "zero": dict(mcost=0.0, lcost=0.0, margin=0.0),
          "zero_strict": dict(mcost=0.0, lcost=0.0, margin=0.001)}[tag]
    out = pd.concat([run("long", **kw), run("y2025", **kw), run("short", **kw)], ignore_index=True)
    out.to_parquet(f"run_sr_{tag}.parquet")
    print(out.groupby("win").size())
