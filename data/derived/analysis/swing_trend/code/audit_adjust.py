"""audit 4: 수정주가(auto_adjust=True) 영향.
- yfinance auto_adjust=False(분할만 반영, 배당 미반영 = 차트 기본 화면) 일봉을 받아 cache와 비교한다.
- 배당이 있는 종목: 미반영 가격으로 판정 28개 상태를 다시 계산해 P1·P2·Y25 안에서 상태가 다른 날 수와,
  '미반영 가격 판정 + 수정 가격 손익(배당 포함)' 수익이 '수정 가격 판정' 수익과 얼마나 다른지 본다.
- 분할: 기간 안 분할 목록, cache에 분할 계단이 남았는지, 전체 배율(×7.3)에 상태가 변하지 않는지(분할 전 시점 화면과 같은 판정).
→ cache/audit_raw_daily.parquet, results/audit_adjust_states.csv, results/audit_adjust_summary.csv"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import swing as S  # noqa: E402

raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
CP = ROOT / "cache" / "audit_raw_daily.parquet"
if not CP.exists():
    import yfinance as yf
    syms = sorted(G)
    d = yf.download(syms, start="2022-06-01", end="2026-10-23", interval="1d", auto_adjust=False, actions=True,
                    progress=False, group_by="ticker", threads=True)
    rows = []
    for s in syms:
        x = d[s].dropna(subset=["Close"]).copy()
        x = x.rename(columns={"Open": "O", "High": "H", "Low": "L", "Close": "C", "Adj Close": "AC", "Volume": "V",
                              "Dividends": "div", "Stock Splits": "split"})
        x["sym"] = s
        x["date"] = pd.DatetimeIndex(x.index).date
        rows.append(x[["sym", "date", "O", "H", "L", "C", "AC", "V", "div", "split"]].reset_index(drop=True))
    pd.concat(rows, ignore_index=True).to_parquet(CP)
U = {s: g.sort_values("date").reset_index(drop=True) for s, g in pd.read_parquet(CP).groupby("sym")}

summ = []
st_rows = []
for s in sorted(G):
    g = G[s]; u = U[s]
    m = g.merge(u, on="date", suffixes=("", "_u"), how="left")
    ndiv = int((m["div"] > 0).sum()); nsplit = int((m["split"] > 0).sum())
    fac = m["C"] / m["C_u"]  # 배당 조정 배율(분할은 둘 다 반영)
    # 배율이 배당 날 아닌 날 바뀌었는지
    jump = (fac / fac.shift(1) - 1).abs()
    divday = m["div"].fillna(0) > 0
    nondiv_jump = int(((jump > 1e-6) & ~divday).sum())
    adjclose_gap = float((m["C"] / m["AC"] - 1).abs().max())
    split_step = float((g.C / g.C.shift(1)).dropna().pipe(lambda r: max(r.max(), 1 / r.min())))
    summ.append(dict(sym=s, ndiv=ndiv, nsplit=nsplit, fac_first=float(fac.iloc[0]), nondiv_fac_jump=nondiv_jump,
                     cache_vs_adjclose=adjclose_gap, max_day_ratio=split_step,
                     split_dates=";".join(str(x) for x in m.loc[m["split"] > 0, "date"])))
    # 상태 비교(배당 있는 종목만)
    if ndiv == 0 or fac.isna().any():
        continue
    gu = pd.DataFrame({"O": m["O_u"], "H": m["H_u"], "L": m["L_u"], "C": m["C_u"]})
    sa = S.states(g); su = S.states(gu)
    O = g.O.to_numpy(float); C = g.C.to_numpy(float)
    for per in ("P1", "P2", "Y25"):
        ix = S.period_index(g, per)
        if len(ix) == 0 or ix[0] < 60:
            continue
        j0, j1 = int(ix[0]), int(ix[-1])
        for nm in sa:
            ra = S.sim_period(O, C, sa[nm], j0, j1, 0.0005)[0]
            ru = S.sim_period(O, C, su[nm], j0, j1, 0.0005)[0]
            st_rows.append(dict(sym=s, per=per, det=nm, days=j1 - j0 + 1,
                                state_diff_days=int((sa[nm][j0 - 1:j1] != su[nm][j0 - 1:j1]).sum()),
                                ret_adj=ra, ret_rawsig=ru, dret=ru - ra, div_yield_period=float(fac.iloc[j0] / fac.iloc[j1] - 1)))
sm = pd.DataFrame(summ)
sm.to_csv(ROOT / "results" / "audit_adjust_summary.csv", index=False)
sr = pd.DataFrame(st_rows)
sr.to_csv(ROOT / "results" / "audit_adjust_states.csv", index=False)

print("cache C vs 지금 받은 Adj Close 최대 차이:", f"{sm.cache_vs_adjclose.max():.2e}")
print("배당 있는 종목:", int((sm.ndiv > 0).sum()), "/", len(sm), " 기간 안 분할 있는 종목:", int((sm.nsplit > 0).sum()))
print(sm[sm.nsplit > 0][["sym", "split_dates", "max_day_ratio"]].to_string(index=False))
print("배당 날 아닌 날 배율이 바뀐 날(합):", int(sm.nondiv_fac_jump.sum()))
print("첫날 배당 배율(작을수록 배당 큼) 하위 10:")
print(sm.sort_values("fac_first").head(10)[["sym", "ndiv", "fac_first"]].to_string(index=False))
if len(sr):
    print(f"\n상태 비교 {len(sr)}건(배당 종목 × 구간 × 판정 28, 예열 60일 이상)")
    agg = sr.groupby("per").agg(n=("dret", "size"), any_diff=("state_diff_days", lambda x: int((x > 0).sum())),
                                diff_days=("state_diff_days", "sum"), med_abs_dret=("dret", lambda x: float(x.abs().median())),
                                p95_abs_dret=("dret", lambda x: float(x.abs().quantile(0.95))), max_abs_dret=("dret", lambda x: float(x.abs().max())))
    print(agg.to_string())
    print("판정별 상태 다른 날 합(상위):")
    print(sr.groupby("det").state_diff_days.sum().sort_values(ascending=False).head(10).to_string())
    print("|수익 차이| 큰 10건:")
    print(sr.reindex(sr.dret.abs().sort_values(ascending=False).index).head(10).to_string(index=False))
    for s in ("MU", "NVDA", "SPY", "BE", "NBIS"):
        q = sr[(sr.sym == s) & (sr.det.isin([S.DEFAULT, S.DEFAULT2, "Z5"]))]
        if len(q):
            print(s, q[["per", "det", "state_diff_days", "dret"]].to_string(index=False))

# 배율 불변(분할 전 시점 화면): 전 종목 × 28 판정
bad = 0; tot = 0
for s, g in G.items():
    g2 = g.copy()
    for k in ("O", "H", "L", "C"):
        g2[k] = g2[k] * 7.3
    a = S.states(g); b = S.states(g2)
    for nm in a:
        tot += 1
        bad += int((a[nm] != b[nm]).sum())
print(f"\n전체 배율 ×7.3 불변 시험: 종목 {len(G)} × 판정 28 = {tot}, 상태 다른 날 {bad}")
