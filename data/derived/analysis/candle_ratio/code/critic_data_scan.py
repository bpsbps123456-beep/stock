"""critic: 144종목 캐시 이음매(옛 10/10 캐시 vs 새로 받은 값) 일치와 큰 봉간 점프, 중복 시각 점검."""
import numpy as np, pandas as pd, pathlib
H = pathlib.Path(__file__).resolve().parent.parent
OLD = H.parent / "trend_zones" / "cache" / "dip_raw_1010"
rows = []
for f in sorted((H / "cache").glob("*_5m.parquet")):
    s = f.stem[:-3]
    d = pd.read_parquet(f); d.index = d.index.tz_convert("America/New_York")
    d = d[d.index.date <= pd.Timestamp("2026-10-22").date()]
    o = pd.read_parquet(OLD / f"5m_{s}.parquet")
    if isinstance(o.columns, pd.MultiIndex): o.columns = o.columns.get_level_values(0)
    o.index = o.index.tz_convert("America/New_York")
    ov = d.index.intersection(o.index)
    newstart = d.index[~d.index.isin(o.index)].min()
    ratio_med = np.nan; maxdiff = np.nan; nov = len(ov)
    if nov:
        r = np.log(d.loc[ov, "Close"].astype(float).values / o.loc[ov, "Close"].astype(float).values)
        ratio_med = float(np.median(r)); maxdiff = float(np.max(np.abs(r)))
    # 이음매가 실제로 바뀐 첫 시각: 옛 파일 마지막 + 새 파일 첫 시각
    lr = np.log(d.Open.astype(float).values[1:] / d.Close.astype(float).values[:-1])
    big = np.flatnonzero(np.abs(lr) > 0.15)
    rows.append(dict(sym=s, n=len(d), dup=int(d.index.duplicated().sum()), old_last=o.index.max(), overlap=nov,
                     ov_med_logdiff=ratio_med, ov_max_logdiff=maxdiff,
                     n_gap15=len(big), gaps=";".join(f"{d.index[i+1]:%m-%d %H:%M}:{lr[i]:+.2f}" for i in big[:4])))
r = pd.DataFrame(rows)
r.to_csv(H / "results" / "critic_data_scan.csv", index=False)
print("종목", len(r), "중복 시각 있는 종목", (r.dup > 0).sum())
print("겹침 구간 중앙 로그차 |x|>0.001 종목:", (r.ov_med_logdiff.abs() > 1e-3).sum())
print(r.sort_values("ov_max_logdiff", ascending=False).head(8)[["sym","overlap","ov_med_logdiff","ov_max_logdiff"]].to_string())
print(r[r.n_gap15 > 0][["sym", "n_gap15", "gaps"]].to_string())
