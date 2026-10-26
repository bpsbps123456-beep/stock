"""이격매수매도 AI 궁합 4부 보충(2026-10-23, 마지막 점검 단계에서 만든 것을 옮김). 입력 ../results/ai_fit_trades.csv, 출력은 표준출력 → ../results/ai_fit_walk.txt"""
# 최종 점검 보충: 날마다 '그 전날까지' 성적으로 종목을 고르고 그날 거래로 확인(전진 검증)
import pandas as pd, numpy as np, sys
from scipy import stats
P = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/dip_fit/results/ai_fit_trades.csv"
T = pd.read_csv(P, encoding="utf-8-sig")
T = T[T.ret.notna()].copy()
T["day"] = pd.to_datetime(T.day)
def add_e(D, keys):
    D = D.copy(); D["e"] = D.ret - D.groupby(keys).ret.transform("mean"); return D
samples = {"정규": add_e(T[T.ses=="정규"], ["day"]), "전체": add_e(T, ["day","ses"])}
out = []
def tt(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3: return np.nan, np.nan, len(x)
    t = x.mean()/(x.std(ddof=1)/np.sqrt(len(x))); return x.mean(), t, len(x)
res = {}
for sname, D in samples.items():
    days = sorted(D.day.unique())
    for start_name, start in [("8/3부터(P0 5분봉 포함)", "2026-08-03"), ("9/10부터(3분봉만)", "2026-09-10")]:
        for k in (3, 5):
            for score_col in ("e", "ret"):
                rows = []
                for d in days:
                    per = D.loc[D.day==d, "period"].iloc[0]
                    if per == "P0": continue
                    H = D[(D.day < d) & (D.day >= pd.Timestamp(start))]
                    g = H.groupby("sym")[score_col].agg(["mean","count"])
                    g = g[g["count"] >= k]
                    if len(g) < 15: continue
                    lo, hi = g["mean"].quantile([1/3, 2/3])
                    top = set(g.index[g["mean"] >= hi]); bot = set(g.index[g["mean"] <= lo])
                    X = D[D.day==d]
                    xt = X[X.sym.isin(top)]; xb = X[X.sym.isin(bot)]
                    # 그날 종목별 e와 사전 점수의 순위 상관
                    m = X.groupby("sym").e.mean().to_frame("y").join(g["mean"], how="inner")
                    rho = stats.spearmanr(m.y, m["mean"]).statistic if len(m) >= 5 else np.nan
                    rows.append(dict(day=d, per=per, n=len(X), nt=len(xt), nb=len(xb),
                        diff=(xt.ret.mean()-xb.ret.mean()) if len(xt) and len(xb) else np.nan,
                        gain=(xt.ret.mean()-X.ret.mean()) if len(xt) else np.nan,
                        top_ret=xt.ret.mean() if len(xt) else np.nan, all_ret=X.ret.mean(), rho=rho))
                R = pd.DataFrame(rows)
                for per in ("P1", "P2", "P1+P2"):
                    S = R if per=="P1+P2" else R[R.per==per]
                    if start_name.startswith("9/10") and per in ("P1","P1+P2"):
                        pass
                    md, td, nd = tt(S["diff"]); mg, tg, _ = tt(S["gain"]); mr, tr, nr = tt(S["rho"])
                    out.append(dict(표본=sname, 시작=start_name, 최소=k, 점수=score_col, 구간=per, 날짜=nd,
                        상하차=md, t상하=td, 상위_전체대비=mg, t이득=tg, 상위거래당=S.top_ret.mean(), 전체거래당=S.all_ret.mean(),
                        상위건=int(S.nt.sum()), 평균rho=mr, t_rho=tr))
O = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
fmt = O.copy()
for c in ["상하차","상위_전체대비","상위거래당","전체거래당"]: fmt[c] = (fmt[c]*100).map(lambda v: f"{v:+.2f}%")
for c in ["t상하","t이득","평균rho","t_rho"]: fmt[c] = fmt[c].map(lambda v: f"{v:+.2f}")
print(fmt.to_string(index=False))
O.to_csv(sys.argv[1] if len(sys.argv)>1 else "walk_out.csv", index=False, encoding="utf-8-sig")
