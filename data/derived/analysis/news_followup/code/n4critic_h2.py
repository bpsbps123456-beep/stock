"""N4 반박 검산(h2): R 의 은행별 차이(GS PB 대 BofA 등 고객 흐름)와 GS 안 중복·첫 신호. 읽기만: results/n4h14_signals.csv
py code/n4critic_h2.py"""
import pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
s = pd.read_csv(HERE / "results" / "n4h14_signals.csv", parse_dates=["entry"])
r = s[s.family == "R"].sort_values("entry").copy()
for h in (5, 10, 20): r[f"ex{h}"] = r[f"SOXX_r{h}"] - r[f"SOXX_A{h}"]
def first(df):
    keep, last = [], None
    for e in df.entry:
        keep.append(last is None or np.busday_count(last.date(), e.date()) > 5); last = e if keep[-1] else last
    return df[np.array(keep)]
rng = np.random.default_rng(1)
def boot(df, h):
    v = df[f"ex{h}"].dropna().to_numpy()
    m = np.array([rng.choice(v, len(v)).mean() for _ in range(5000)])
    return f"{v.mean():+.4f} [{np.quantile(m,.025):+.4f},{np.quantile(m,.975):+.4f}] n={len(v)}"
for tag, df in [("GS 전부", r[r.bank == "GS"]), ("GS 중복의심 제외", r[(r.bank == "GS") & r.uncertain_dup.isna()]),
                ("GS 첫 신호", first(r[r.bank == "GS"])), ("GS 외 전부", r[r.bank != "GS"]), ("BofA", r[r.bank == "BofA"])]:
    print(f"{tag:14s} 5일 {boot(df,5)} | 10일 {boot(df,10)}")
