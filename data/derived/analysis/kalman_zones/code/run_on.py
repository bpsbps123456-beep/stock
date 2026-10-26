"""초단타 조합 400가지 x 3창(2026 1h, 2025 1h, 최근 60일 5m) + 대조군. 체결: 정규장 지정가 매수, 깨어 있는 프리장·정규장 매도."""
import itertools
from datetime import date

import numpy as np
import pandas as pd

from core import *
from evalcfg import data, indicator, WINDOWS
from grid_short import portfolio, window_metrics
from run_v3 import ctx
from sim_on import sim_overnight, sim_close_to_open

D, HALF = data()
WINDOWS["y2025"] = dict(base_key="1h", eval_start=date(2025, 1, 2), split=date(2025, 7, 1))
LEVELS = ["칼만중심", "매도라인", "패닉라인", "종가-1%", "종가-2%", "종가-3%", "전일저가", "3일최저", "ATR0.5", "ATR1.0"]
FILTERS = ["없음", "평범·매수만", "칼만상승만", "둘다"]
TARGETS = [0.005, 0.01, 0.02, 0.03, 0.0]
HOLDS = [1, 2]


def day_arrays(S, s):
    A = S[s]
    t = pd.to_datetime(A["start_ns"], utc=True).tz_convert("America/New_York")
    mins = np.asarray(t.hour * 60 + t.minute)
    A["is_pre"] = (mins >= 240) & (mins < 570)
    reg = A["is_reg"]
    di = A["day_idx"]
    first = np.zeros(len(reg), bool)
    seen = set()
    for j in np.flatnonzero(reg):
        if di[j] not in seen:
            seen.add(di[j]); first[j] = True
    A["first_reg"] = first
    return A


def levels(S, s, win):
    """세션별(day_idx) 개장 전 지정가 후보와 필터."""
    A = S[s]
    reg = A["is_reg"]
    df = pd.DataFrame({"d": A["day_idx"][reg], "o": A["o"][reg], "h": A["h"][reg], "l": A["l"][reg], "c": A["c"][reg],
                       "st": A["start_ns"][reg]})
    g = df.groupby("d")
    daily = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "st": g.st.first()})
    pc = daily.c.shift(1)
    tr = np.maximum(daily.h - daily.l, np.maximum((daily.h - pc).abs(), (daily.l - pc).abs()))
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1)
    tb = tf_bars(D[(s, "1h")], 240, False, HALF)
    K, u, z = indicator(tb, 21, "rms", 50, 0.75, 1.75)
    bi = np.searchsorted(tb.end_ns.values, daily.st.values, side="right") - 1
    ok = bi >= 5
    Kd = np.where(ok, K[np.maximum(bi, 0)], np.nan)
    ud = np.where(ok, u[np.maximum(bi, 0)], np.nan)
    zd = np.where(ok, z[np.maximum(bi, 0)], -9)
    rising = np.where(ok, K[np.maximum(bi, 0)] > K[np.maximum(bi - 4, 0)], False)
    cand = {"칼만중심": Kd, "매도라인": Kd - 0.75 * ud, "패닉라인": Kd - 1.75 * ud,
            "종가-1%": pc * 0.99, "종가-2%": pc * 0.98, "종가-3%": pc * 0.97,
            "전일저가": daily.l.shift(1).values, "3일최저": daily.l.rolling(3).min().shift(1).values,
            "ATR0.5": pc - 0.5 * atr14, "ATR1.0": pc - 1.0 * atr14}
    filt = {"없음": np.ones(len(daily), bool), "평범·매수만": np.isin(zd, [0, 1]), "칼만상승만": rising,
            "둘다": np.isin(zd, [0, 1]) & rising}
    n_days = len(A["days"])
    out = {}
    for name, v in cand.items():
        v = np.asarray(v, float)
        v = np.where(v < pc.values, v, np.nan)  # 지지선은 전일 종가 아래일 때만
        for fn, fm in filt.items():
            arr = np.full(n_days, np.nan)
            arr[daily.index.values] = np.where(fm, v, np.nan)
            out[(name, fn)] = arr
    return out


def run(win, mcost=0.001, lcost=0.0, margin=0.0):
    S, common, sel = ctx(win)
    for s in SYMS:
        day_arrays(S, s)
    LV = {s: levels(S, s, win) for s in SYMS}
    rows = []
    eq = portfolio([sim_close_to_open(S[s]["o"], S[s]["c"], S[s]["is_reg"], S[s]["is_end"], S[s]["first_reg"], S[s]["j0"], mcost)
                    for s in SYMS], S)
    rows.append({"win": win, "entry": "참고: 종가매수→다음시가", **window_metrics(eq, sel)})
    for lv, fi, tg, hd in itertools.product(LEVELS, FILTERS, TARGETS, HOLDS):
        eqs, ntr, nw, sr = [], 0, 0, 0.0
        for s in SYMS:
            A = S[s]
            e, t, w, r = sim_overnight(A["o"], A["h"], A["l"], A["c"], A["is_reg"], A["is_pre"], A["is_awake"], A["is_end"],
                                       A["first_reg"], A["day_idx"], A["j0"], LV[s][(lv, fi)], tg, hd, mcost, lcost, margin,
                                       np.full(len(S[s]['days']), np.nan), 0.005, 0.03)
            eqs.append(e); ntr += t; nw += w; sr += r
        rows.append({"win": win, "entry": lv, "filter": fi, "tgt": tg, "hold": hd, "trades": ntr,
                     "trades_per_day": ntr / len(common), "win_rate": nw / ntr if ntr else np.nan,
                     "avg_trade": sr / ntr if ntr else np.nan, **window_metrics(portfolio(eqs, S), sel)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    kw = {"base": {}, "stress": dict(mcost=0.002, lcost=0.001, margin=0.001), "zero": dict(mcost=0.0, lcost=0.0, margin=0.0),
          "zero_strict": dict(mcost=0.0, lcost=0.0, margin=0.001)}[tag]
    out = pd.concat([run("long", **kw), run("y2025", **kw), run("short", **kw)], ignore_index=True)
    out.to_parquet(f"run_on_{tag}.parquet")
    print(out.groupby("win").size())
