"""이격매수매도 AI 궁합 4부 보충(2026-10-23, 마지막 점검 단계에서 만든 것을 옮김). 입력 ../results/ai_fit_trades.csv, 출력은 표준출력 → ../results/ai_fit_fitlook.txt"""
# '잘 맞아 보임'(선 닿고 RSI로 튀어 오름) 지표: RSI 매도 비율·승률 — NBIS·IREN과 유니버스 지속성
import pandas as pd, numpy as np
from scipy import stats
P = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/dip_fit/results/ai_fit_trades.csv"
T = pd.read_csv(P, encoding="utf-8-sig"); T = T[T.ret.notna()].copy()
T["rsi"] = (T.why=="RSI").astype(float); T["win"] = (T.ret>0).astype(float)
T["e"] = T.ret - T.groupby(["day","ses"]).ret.transform("mean")
T["rsi_e"] = T.rsi - T.groupby(["day","ses"]).rsi.transform("mean")
T["P"] = T.period.where(T.period!="P0","P0")
print("== NBIS·IREN: 건수 / RSI 매도 비율 / 승률 / 거래당 ==")
for ses_name, S in [("정규", T[T.ses=="정규"]), ("전체", T)]:
    for per in ["P0","P1","P2"]:
        for s in ["NBIS","IREN"]:
            X = S[(S.sym==s)&(S.period==per)]
            print(f"{ses_name} {per} {s}: {len(X):3d}건 RSI {X.rsi.mean():.0%} 승률 {X.win.mean():.0%} 거래당 {X.ret.mean():+.2%}  (같은 날·세션 평균 대비 RSI비율 {X.rsi_e.mean():+.0%})")
    for s in ["NBIS","IREN"]:
        X = S[(S.sym==s)&(S.period.isin(["P1","P2"]))]
        print(f"{ses_name} P1+P2(3분봉) {s}: {len(X)}건 RSI {X.rsi.mean():.0%} 승률 {X.win.mean():.0%} 거래당 {X.ret.mean():+.2%}")
# 날짜 군집 차이: NBIS−IREN RSI 비율 (전체, P0~P2)
def ct(D, col):
    a = D[D.sym=="NBIS"]; b = D[D.sym=="IREN"]
    d = a[col].mean()-b[col].mean()
    # 날짜 군집 회귀
    X = D[D.sym.isin(["NBIS","IREN"])].copy(); X["x"] = (X.sym=="NBIS").astype(float)
    Xm = np.c_[np.ones(len(X)), X.x.values]; y = X[col].values
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]; u = y - Xm@beta
    bread = np.linalg.inv(Xm.T@Xm); meat = np.zeros((2,2))
    for _, idx in X.groupby("day").indices.items():
        g = Xm[idx].T@u[idx]; meat += np.outer(g,g)
    G = X.day.nunique(); V = bread@meat@bread*G/(G-1)
    return d, beta[1]/np.sqrt(V[1,1]), G
for ses_name, S in [("정규", T[T.ses=="정규"]), ("전체", T)]:
    for per, pl in [("P0+P1",["P0","P1"]),("P2",["P2"]),("전체기간",["P0","P1","P2"])]:
        D = S[S.period.isin(pl)]
        d, t, G = ct(D, "rsi"); d2, t2, _ = ct(D, "win")
        print(f"{ses_name} {per}: NBIS−IREN RSI비율 {d:+.0%}(t {t:+.2f}) 승률 {d2:+.0%}(t {t2:+.2f}) 날짜 {G}")
print("\n== 유니버스: 종목별 RSI 매도 비율이 이어지나, 그리고 수익을 맞히나 (정규장, 앞≥3·뒤≥1) ==")
for ses_name, S in [("정규", T[T.ses=="정규"]), ("전체", T)]:
    A = S[S.period.isin(["P0","P1"])].groupby("sym").agg(n=("ret","size"), rsi=("rsi_e","mean"), e=("e","mean"), win=("win","mean"))
    B = S[S.period=="P2"].groupby("sym").agg(n=("ret","size"), rsi=("rsi_e","mean"), e=("e","mean"), win=("win","mean"))
    A1 = S[S.period=="P0"].groupby("sym").agg(n=("ret","size"), rsi=("rsi_e","mean"), e=("e","mean"))
    B1 = S[S.period=="P1"].groupby("sym").agg(n=("ret","size"), rsi=("rsi_e","mean"), e=("e","mean"))
    for nm, a, b in [("P0→P1", A1, B1), ("P0+P1→P2", A, B)]:
        m = a[a.n>=3].join(b[b.n>=1], lsuffix="_a", rsuffix="_b", how="inner")
        r1 = stats.spearmanr(m.rsi_a, m.rsi_b); r2 = stats.spearmanr(m.rsi_a, m.e_b); r3 = stats.spearmanr(m.rsi_b, m.e_b)
        print(f"{ses_name} {nm} ({len(m)}종목): RSI비율 앞→뒤 ρ {r1.statistic:+.2f}(p {r1.pvalue:.3f}) | 앞 RSI비율→뒤 e ρ {r2.statistic:+.2f}(p {r2.pvalue:.3f}) | 같은 구간 RSI비율↔e ρ {r3.statistic:+.2f}")
