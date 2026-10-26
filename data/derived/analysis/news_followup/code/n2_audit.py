"""N2 감사: 독립 재계산·미리보기·SOXX 지그재그 대조.

py code/n2_audit.py → results/n2_audit.json

(가) 단서 재계산: merge_asof(그날까지 마지막 ^TNX)와 SPY 거래일 달력으로 R1·R2·M을 다른 길로 다시 구해 사건표와 비교.
(나) 미리보기: SOXX 사건 30개·주식 사건 30개를 확정일까지로 자른 가격·금리로 다시 계산(지그재그 확정·극점·직전 극점·R1·R2·M).
(다) 칸 차이 재계산: 3분위 경계·위/아래 평균을 pandas로 따로 계산해 n2_cells.csv의 고르기·확인 차이와 비교.
(라) SOXX 지그재그 대조: n2 가격(2001~)과 swing_trend 가격(2022-06~)의 2023-07 이후 확정일·방향 일치율.
(마) SOXX 결과 재계산: y1·y10·y20을 날짜 조회로 다시 계산.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
os.environ["NUMBA_CACHE_DIR"] = str(HERE / "cache" / "numba")
sys.dont_write_bytecode = True
SWING = HERE.parent / "swing_trend"
sys.path.insert(0, str(SWING / "code"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from swing import atr, load, zigzag  # noqa: E402

RES = HERE / "results"


def asof(series: pd.Series, dates) -> np.ndarray:
    q = pd.DataFrame({"d": pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]")}).reset_index()
    s = series.rename("v").reset_index().rename(columns={series.index.name or "index": "d"})
    s["d"] = pd.to_datetime(s["d"]).astype("datetime64[ns]")
    m = pd.merge_asof(q.sort_values("d"), s.sort_values("d"), on="d", direction="backward").sort_values("index")
    return m.v.values


def main():
    raw = pd.read_parquet(HERE / "cache" / "n2_prices_raw.parquet")
    adj = pd.read_parquet(HERE / "cache" / "n2_prices_adj.parquet")
    for x in (raw, adj):
        x["date"] = pd.to_datetime(x.date)
    tnx = raw[raw.sym == "^TNX"].set_index("date").C
    tnx.index.name = "date"
    sxr = raw[raw.sym == "SOXX"].set_index("date").C
    sxr.index.name = "date"
    spy_cal = pd.DatetimeIndex(sorted(raw[raw.sym == "SPY"].date))
    out = {}

    # (가)
    se = pd.read_parquet(RES / "n2_events_stock.parquet")
    xe = pd.read_parquet(RES / "n2_events_soxx.parquet")
    pos = {d: i for i, d in enumerate(spy_cal)}
    for nm, ev in (("stock", se), ("soxx", xe)):
        d = pd.to_datetime(ev.date)
        i = np.array([pos[x] for x in d])
        t0 = asof(tnx, d)
        t20 = asof(tnx, spy_cal[i - 20])
        t5 = asof(tnx, spy_cal[i - 5])
        s0 = sxr.reindex(d).values
        s5 = sxr.reindex(spy_cal[i - 5]).values
        r1 = t0 - t20
        m = ((s0 / s5 - 1 < 0) & (t0 - t5 < 0)).astype(float)
        r2 = asof(tnx, pd.to_datetime(ev.pivot_date)) - asof(tnx, pd.to_datetime(ev.prev_pivot_date))
        out[f"recalc_{nm}"] = dict(n=len(ev), R1_maxabs=float(np.nanmax(np.abs(r1 - ev.R1.values))),
                                   R2_maxabs=float(np.nanmax(np.abs(r2 - ev.R2.values))),
                                   M_mismatch=int((m != ev.M.values).sum()))

    # (나) 미리보기: 확정일까지 자른 자료
    rng = np.random.default_rng(7)
    g = adj[adj.sym == "SOXX"].sort_values("date").reset_index(drop=True)
    look = []
    for _, r in xe.sample(30, random_state=3).iterrows():
        c = int(r.conf_idx)
        gg = g.iloc[: c + 1]
        H, L, C = (gg[k].values.astype(np.float64) for k in ("H", "L", "C"))
        st, pi, pc, pt = zigzag(C, 0.0, 3.0 * atr(H, L, C, 20))
        ok = len(pc) > 1 and pc[-1] == c and pt[-1] == r.type
        cal = pd.DatetimeIndex(gg.date)
        tt = tnx[tnx.index <= cal[-1]].reindex(cal).ffill().values
        sx = sxr[sxr.index <= cal[-1]].reindex(cal).values
        r1 = tt[c] - tt[c - 20]
        r2 = tt[pi[-1]] - tt[pi[-2]]
        m = float((sx[c] / sx[c - 5] - 1 < 0) and (tt[c] - tt[c - 5] < 0))
        look.append(dict(kind="soxx", ok_conf=bool(ok), pivot_same=bool(cal[pi[-1]].date() == r.pivot_date),
                         prev_same=bool(cal[pi[-2]].date() == r.prev_pivot_date),
                         dR1=abs(r1 - r.R1), dR2=abs(r2 - r.R2), dM=abs(m - r.M)))
    D = load()
    cal_s = pd.DatetimeIndex(sorted(pd.to_datetime(D["SOXX"].date)))
    for _, r in se.sample(30, random_state=5).iterrows():
        gs = D[r.sym]
        c = int(r.conf_idx)
        gg = gs.iloc[: c + 1]
        H, L, C = (gg[k].values.astype(np.float64) for k in ("H", "L", "C"))
        st, pi, pc, pt = zigzag(C, 0.0, 3.0 * atr(H, L, C, 20))
        dates = pd.to_datetime(gg.date)
        ok = len(pc) > 1 and pc[-1] == c and pt[-1] == r.type
        cc = cal_s[cal_s <= dates.iloc[-1]]
        tt = tnx[tnx.index <= cc[-1]].reindex(cc).ffill()
        sx = sxr[sxr.index <= cc[-1]].reindex(cc)
        k = len(cc) - 1
        r1 = tt.iat[k] - tt.iat[k - 20]
        m = float((sx.iat[k] / sx.iat[k - 5] - 1 < 0) and (tt.iat[k] - tt.iat[k - 5] < 0))
        r2 = tt.loc[dates.iloc[pi[-1]]] - tt.loc[dates.iloc[pi[-2]]]
        look.append(dict(kind="stock", ok_conf=bool(ok), pivot_same=bool(dates.iloc[pi[-1]].date() == r.pivot_date),
                         prev_same=bool(dates.iloc[pi[-2]].date() == r.prev_pivot_date),
                         dR1=abs(r1 - r.R1), dR2=abs(r2 - r.R2), dM=abs(m - r.M)))
    lk = pd.DataFrame(look)
    out["lookahead"] = {k: dict(n=len(v), conf_ok=int(v.ok_conf.sum()), pivot_same=int(v.pivot_same.sum()), prev_same=int(v.prev_same.sum()),
                                max_dR1=float(v.dR1.max()), max_dR2=float(v.dR2.max()), max_dM=float(v.dM.max()))
                        for k, v in lk.groupby("kind")}

    # (다) 칸 차이 재계산
    cells = pd.read_csv(RES / "n2_cells.csv")
    b = cells[cells.feats == "base"]
    errs = []
    for nm, ev, sel, conf in (("stock", se, "P1", "P2"), ("soxx", xe, "L1", "L2")):
        for t in (1, -1):
            s1 = ev[(ev.period == sel) & (ev.type == t)]
            s2 = ev[(ev.period == conf) & (ev.type == t)]
            for f in ("R1", "R2", "M"):
                for o in ("y1", "y10", "y20"):
                    def diff(s):
                        s = s.dropna(subset=[o])
                        if f == "M":
                            return s[s.M == 1][o].mean() - s[s.M == 0][o].mean()
                        lo, hi = np.quantile(s1[f].dropna(), [1 / 3, 2 / 3])
                        return s[s[f] > hi][o].mean() - s[s[f] <= lo][o].mean()
                    row = b[(b.set == nm) & (b.type == t) & (b.feature == f) & (b.outcome == o)].iloc[0]
                    for per, s in ((sel, s1), (conf, s2)):
                        a, c_ = diff(s), row[f"{per}_diff"]
                        if not (np.isnan(a) and np.isnan(c_)):
                            errs.append(abs(a - c_))
    out["cell_recalc"] = dict(compared=len(errs), max_abs_err=float(np.nanmax(errs)))

    # (라) SOXX 지그재그 대조
    sw = D["SOXX"]
    H, L, C = (sw[k].values.astype(np.float64) for k in ("H", "L", "C"))
    _, pi2, pc2, pt2 = zigzag(C, 0.0, 3.0 * atr(H, L, C, 20))
    d2 = pd.to_datetime(sw.date)
    A = {(d2.iloc[c].date(), int(t)) for c, t in zip(pc2, pt2) if d2.iloc[c] >= pd.Timestamp("2023-07-01")}
    H, L, C = (g[k].values.astype(np.float64) for k in ("H", "L", "C"))
    _, pi1, pc1, pt1 = zigzag(C, 0.0, 3.0 * atr(H, L, C, 20))
    d1 = pd.to_datetime(g.date)
    B = {(d1.iloc[c].date(), int(t)) for c, t in zip(pc1, pt1) if pd.Timestamp("2023-07-01") <= d1.iloc[c] <= pd.Timestamp("2026-10-22")}
    out["soxx_zigzag_vs_swing"] = dict(swing=len(A), n2=len(B), both=len(A & B), only_swing=sorted(map(str, A - B)), only_n2=sorted(map(str, B - A)))

    # (마) SOXX 결과 재계산
    gi = g.set_index("date")
    errs = []
    for _, r in xe.iterrows():
        dd = list(gi.index)
        c = int(r.conf_idx)
        o_in = gi.O.iat[c + 1]
        if not r.censored:
            c2 = dd.index(pd.Timestamp(r.exit_date))
            y1 = r.type * np.log(gi.O.iat[c2 + 1] / o_in)
            errs.append(abs(y1 - r.y1))
        if c + 21 < len(dd):
            errs.append(abs(r.type * np.log(gi.O.iat[c + 21] / o_in) - r.y20))
    out["soxx_outcome_recalc"] = dict(compared=len(errs), max_abs_err=float(max(errs)))
    (RES / "n2_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
