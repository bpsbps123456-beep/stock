"""이격매수매도 AI 궁합 4부 보충(2026-10-23, 마지막 점검 단계에서 만든 것을 옮김). 입력 ../results/ai_fit_trades.csv, 출력은 표준출력 → ../results/ai_fit_walk_jk.txt"""
import pandas as pd, numpy as np
P = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/dip_fit/results/ai_fit_trades.csv"
T = pd.read_csv(P, encoding="utf-8-sig"); T = T[T.ret.notna()&(T.ses=="정규")].copy(); T["day"]=pd.to_datetime(T.day)
T["e"] = T.ret - T.groupby("day").ret.transform("mean")
for start in ["2026-08-03","2026-09-10"]:
  rows=[]
  for d in sorted(T.day.unique()):
    per = T.loc[T.day==d,"period"].iloc[0]
    if per=="P0": continue
    H = T[(T.day<d)&(T.day>=pd.Timestamp(start))]; g = H.groupby("sym").e.agg(["mean","count"]); g=g[g["count"]>=3]
    if len(g)<15: continue
    lo,hi = g["mean"].quantile([1/3,2/3]); X=T[T.day==d]
    xt=X[X.sym.isin(g.index[g["mean"]>=hi])]; xb=X[X.sym.isin(g.index[g["mean"]<=lo])]
    rows.append((d.strftime("%m/%d"),per,xt.ret.mean()-xb.ret.mean(),xt.ret.mean(),X.ret.mean(),len(xt),len(xb)))
  R=pd.DataFrame(rows,columns=["day","per","diff","top","all","nt","nb"])
  print("시작",start)
  for per in ["P1","P2"]:
    S=R[R.per==per].dropna(); x=S["diff"].values
    ts=[]
    for i in range(len(x)):
      y=np.delete(x,i); ts.append(y.mean()/(y.std(ddof=1)/np.sqrt(len(y))))
    print(f" {per}: 날짜 {len(x)} 평균 {x.mean():+.2%} 양(+)인 날 {(x>0).sum()}/{len(x)} 하루 빼기 t 최소 {min(ts):+.2f} 최대 {max(ts):+.2f}")
  print(R[R.per=="P2"].assign(diff=lambda d:(d["diff"]*100).round(2),top=lambda d:(d.top*100).round(2),all=lambda d:(d["all"]*100).round(2)).to_string(index=False))
