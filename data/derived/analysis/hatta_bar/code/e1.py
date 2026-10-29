"""E1(실측 호가) 손익: 신호 마감 + 10초 매도호가에 사고, 30분 안 +1% 지정가(다음 봉은 종가 ≥ 목표일 때만, 그 뒤는 고가),
못 닿으면 창 끝 매수호가에 판다. 사용: py e1.py   → results/e1.json"""
import os, json
import numpy as np
import pandas as pd
import engine as E
from build import SYMS
import evaluate as V

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")


def r1_e1(rows, q):
    out = []
    bars = {}
    for r in rows.itertuples():
        sym = SYMS[int(r.sid)]
        if sym not in bars:
            d = E.load(sym)
            bars[sym] = E.prep(d)
        a = bars[sym]
        t = int(r.t)
        qq = q.get((sym, int(r.ts)))
        if qq is None or not np.isfinite(qq[0]) or qq[0] <= 0:
            out.append((np.nan, np.nan))
            continue
        ask_in, bid_in, ask_out, bid_out = qq
        tlim = a["ts"][t] + 30
        z = t + 1
        while z + 1 < len(a["ts"]) and a["ts"][z + 1] <= tlim and a["day"][z + 1] == a["day"][t]:
            z += 1
        tg = ask_in * 1.01
        hit = a["c"][t + 1] >= tg or (z >= t + 2 and np.max(a["h"][t + 2:z + 1]) >= tg)
        ex = bid_out if np.isfinite(bid_out) and bid_out > 0 else a["c"][z]
        spread_in = (ask_in - bid_in) / ((ask_in + bid_in) / 2) if np.isfinite(bid_in) and bid_in > 0 else np.nan
        out.append((0.01 if hit else ex / ask_in - 1, spread_in))
    return np.array(out)


def main():
    qd = pd.read_parquet(os.path.join(E.CACHE, "quotes.parquet"))
    q = {(s, int(t)): (a, b, c, d) for s, t, a, b, c, d in qd[["sym", "ts", "ask_in", "bid_in", "ask_out", "bid_out"]].itertuples(index=False)}
    rng = np.random.default_rng(1029)
    res = {}
    for f in sorted(os.listdir(RES)):
        if not (f.startswith("signals_") and f.endswith(".csv")):
            continue
        name = f[8:-4]
        s = pd.read_csv(os.path.join(RES, f))
        v = r1_e1(s, q)
        s["r1_e1"], s["spread_in"] = v[:, 0], v[:, 1]
        back = s[(s.day > V.SPLIT) & s.r1_e1.notna()]
        days = back.day.unique()
        agg = back.groupby("day").r1_e1.agg(["sum", "count"]).reindex(days)
        draws = []
        for _ in range(V.B_BOOT):
            w = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
            draws.append((agg["sum"].to_numpy() * w).sum() / (agg["count"].to_numpy() * w).sum())
        draws = np.array(draws)
        res[name] = {"n_back": int(len(back)), "quoted_share": float(s.r1_e1.notna().mean()),
                     "r1_e1_back": float(back.r1_e1.mean()) if len(back) else None,
                     "p_r1_e1_le0": float((1 + np.sum(~(draws > 0))) / (V.B_BOOT + 1)) if len(back) else None,
                     "spread_median_by_sess": {E.SESS_NAMES[k]: float(g.spread_in.median()) for k, g in s.groupby("sess")},
                     "r1_e1_back_by_sess": {E.SESS_NAMES[k]: float(g.r1_e1.mean()) for k, g in back.groupby("sess")}}
        s.to_csv(os.path.join(RES, f), index=False)
        print(name, res[name]["n_back"], res[name]["r1_e1_back"])
    json.dump(res, open(os.path.join(RES, "e1.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
