import pandas as pd, numpy as np
exec(open('rare_scout.py', encoding='utf-8').read().split("# 1)")[0])
def eps2(ddv, X):
    """에피소드: 로그낙폭이 ln(1-X) 아래로 처음 내려간 날 시작, 낙폭이 그 절반 위로 회복하면 끝."""
    thr = np.log(1-X); rec = np.log(1-X/2); n=0; on=False
    for v in ddv:
        if not np.isfinite(v): continue
        if not on and v <= thr: n+=1; on=True
        elif on and v > rec: on=False
    return n
def x4(series, yrs):
    s = series.dropna().values
    best = np.nan
    for X in np.arange(0.02,0.60,0.0025):
        if eps2(s, X)/yrs >= 4: best = X
    return best
mask = (C.index>='2024-01-02')&(C.index<='2025-12-31')
st = [s for s in C.columns if s not in etf]
th = {s: x4(dd[s][mask], 2.0) for s in st+sorted(etf)}
for grp,names in (('주식',st),('ETF',sorted(etf))):
    vals=[]; n_ep=[]
    for s in names:
        X=th[s]
        if not np.isfinite(X): continue
        sel=(dd[s]<=np.log(1-X))&(C.index>='2026-01-02')&Z[s].notna()
        vals += list(Z[s][sel].values)
        n_ep.append(eps2(dd[s][C.index>='2026-01-02'].values, X)/(196/252))
    v=np.array(vals); print(f'{grp}: 2024-25 기준 연4번 낙폭에 2026 도달한 {len(v)}칸 — 하방계열 {(v<0).mean():.1%} 애매 {(v==0).mean():.1%} 상방계열 {(v>0).mean():.1%}, 2026 실제 연환산 {np.median(n_ep):.1f}번(중앙)')
for s in ['SOXX','EWY','SMH','QQQ','SPY','MU','NVDA','SNDK']:
    print(f'{s}: 연4번 낙폭 {th[s]*100:.1f}% / 지금 {(np.exp(dd[s].iloc[-1])-1)*100:.1f}%  (20일 최고 종가 대비)')
print('주식 연4번 낙폭 중앙', round(np.nanmedian([th[s] for s in st])*100,1),'%, 25~75%', np.round(np.nanpercentile([th[s] for s in st],[25,75])*100,1))
