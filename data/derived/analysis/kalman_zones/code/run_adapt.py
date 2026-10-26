"""추세 따라 매수 깊이를 바꾸는 규칙: 상승 추세면 얕게(전일 종가 -a%), 아니면 깊게(-b%) 또는 안 삼. 프리장·본장만.
세 창(2026, 2025, 최근 60일), 닿으면 체결 / 봉 종가 체결. 정리는 다음 날 프리장(목표 미달 시)."""
import itertools
import numpy as np, pandas as pd
from core import *
from evalcfg import data, indicator
from grid_short import portfolio, window_metrics
from run_on import ctx, day_arrays
from diag_pre import sim3

D, H = data()


def day_info(S, s):
    A = S[s]
    reg = A["is_reg"]
    df = pd.DataFrame({"d": A["day_idx"][reg], "c": A["c"][reg], "st": A["start_ns"][reg]})
    g = df.groupby("d")
    daily = pd.DataFrame({"c": g.c.last(), "st": g.st.first()})
    pc = daily.c.shift(1).values
    tb = tf_bars(D[(s, "1h")], 240, False, H)
    K, u, z = indicator(tb, 21, "rms", 50, 0.75, 1.75)
    bi = np.searchsorted(tb.end_ns.values, daily.st.values, side="right") - 1
    ok = bi >= 5
    Kd = np.where(ok, K[np.maximum(bi, 0)], np.nan)
    zd = np.where(ok, z[np.maximum(bi, 0)], -9)
    rising = np.where(ok, K[np.maximum(bi, 0)] > K[np.maximum(bi - 4, 0)], False)
    trend = {"구간 매수·과매수": np.isin(zd, [1, 2]), "칼만선 상승": rising, "종가>칼만선": pc > Kd}
    return daily.index.values, pc, trend, len(A["days"])


def run(win, close_fill):
    S, common, sel = ctx(win)
    for s in SYMS:
        day_arrays(S, s)
    INFO = {s: day_info(S, s) for s in SYMS}
    PRE = {}
    for s in SYMS:
        t = pd.to_datetime(S[s]["start_ns"], utc=True).tz_convert("America/New_York")
        mins = np.asarray(t.hour * 60 + t.minute)
        PRE[(s, 4)] = (mins >= 240) & (mins < 570)
        PRE[(s, 7)] = (mins >= 420) & (mins < 570)
    rows = []
    trends = ["구간 매수·과매수", "칼만선 상승", "종가>칼만선", "(추세 무관)"]
    for tr, up, dn, tgt, pf in itertools.product(trends, [0.0, 0.005, 0.01], [0.01, 0.015, 0.02, 0.03, 0.04, None],
                                                 [0.005, 0.01, 0.02], [4, 7]):
        if tr == "(추세 무관)" and (up != 0.01 or dn is None):
            continue    # 추세 무관은 '항상 -dn%' 한 줄씩만
        eqs, nt = [], 0
        for s in SYMS:
            A = S[s]
            idx, pc, trd, nd = INFO[s]
            lv = np.full(nd, np.nan)
            if tr == "(추세 무관)":
                lv[idx] = pc * (1 - dn)
            else:
                t = trd[tr]
                lv_up = pc * (1 - up)
                lv_dn = pc * (1 - dn) if dn is not None else np.full(len(pc), np.nan)
                lv[idx] = np.where(t, lv_up, lv_dn)
            e, n1, _ = sim3(A["o"], A["h"], A["l"], A["c"], A["is_reg"], PRE[(s, pf)], A["is_awake"], A["is_end"],
                            A["first_reg"], A["day_idx"], A["j0"], lv, tgt, 1, True, True, close_fill)
            eqs.append(e); nt += n1
        m = window_metrics(portfolio(eqs, S), sel)
        rows.append({"win": win, "close_fill": close_fill, "trend": tr, "up": up, "dn": dn if dn is not None else -1,
                     "tgt": tgt, "pre_from": pf, "trades_per_day": nt / len(common), **m})
    return rows


if __name__ == "__main__":
    rows = []
    for w in ("long", "y2025", "short"):
        for cf in (False, True):
            rows += run(w, cf)
    pd.DataFrame(rows).to_parquet("run_adapt.parquet")
    print(len(rows))
