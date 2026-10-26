import pandas as pd, numpy as np
x = pd.read_parquet('panel_1d_20261012.parquet')
C = x.pivot_table(index='ts', columns='sym', values='Close').sort_index()
etf = set(x[x.is_etf].sym)
lc = np.log(C); lr = lc.diff()
sd20 = lr.rolling(20).std(); sd252 = lr.rolling(252, min_periods=200).std()
k = np.sqrt(19*39/120)
T3 = (lc - lc.rolling(20).mean())/(sd20*k)
d = C.diff(); up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-d).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
rsi = 100-100/(1+up/dn)
z = np.select([rsi>=70, rsi<=30, T3>=0.5, T3<=-0.5],[2,-2,1,-1],0); Z = pd.DataFrame(z, index=C.index, columns=C.columns).where(T3.notna() & rsi.notna())
dd = np.log(C/C.rolling(20).max())          # 20일 최고 종가 대비(로그)
ddz = dd/(sd252*np.sqrt(10))                 # 1년 변동성 단위
def eps_rate(hit, gap=10):
    n=0; last=-999
    for i,h in enumerate(hit):
        if h:
            if i-last>gap: n+=1
            last=i
    return n
# 1) 1년 변동성 단위 m별 종목당 연 횟수(2024-01~2025-12) 중앙값
mask = (C.index>='2024-01-02')&(C.index<='2025-12-31'); yrs=2.0
for m in (1.0,1.25,1.5,1.75,2.0):
    rates=[eps_rate((ddz[s][mask]<=-m).values)/yrs for s in C.columns if s not in etf and ddz[s][mask].notna().sum()>400]
    print(f'm={m}: 종목당 연 {np.median(rates):.1f}번(중앙), 25~75% {np.percentile(rates,25):.1f}~{np.percentile(rates,75):.1f}')
# 2) 2026에 그 낙폭 도달한 날 B0 단계 분포 (m=1.5, 1.75)
m26 = C.index>='2026-01-02'
for m in (1.25,1.5,1.75):
    sel = (ddz<=-m)&m26[:,None]&Z.notna()
    v = Z[sel].stack()
    st = [s for s in C.columns if s not in etf]
    vs = Z[st][sel[st]].stack()
    print(f'm={m} 2026 주식 {len(vs)}칸: 하방계열 {(vs<0).mean():.1%} 애매 {(vs==0).mean():.1%} 상방계열 {(vs>0).mean():.1%}')
for s in ['SOXX','EWY','SMH','MU','NVDA']:
    print(s, '지금 ddz', round(ddz[s].iloc[-1],2), 'dd%', round((np.exp(dd[s].iloc[-1])-1)*100,1), 'sd252%', round(sd252[s].iloc[-1]*100,2))
print('--- 고친 집계')
st = [s for s in C.columns if s not in etf]
Zs = Z[st].to_numpy(); Ds = ddz[st].to_numpy(); rows = (C.index>='2026-01-02')
for m in (1.25,1.5,1.75):
    sel = (Ds<=-m) & rows[:,None] & np.isfinite(Zs)
    v = Zs[sel]
    print(f'm={m} 2026 주식 {sel.sum()}칸: 하방계열 {(v<0).mean():.1%} 애매 {(v==0).mean():.1%} 상방계열 {(v>0).mean():.1%}')
# 종목 자기 이력으로 '연 4번' 낙폭(2024-25) 정하고 2026 그 낙폭 날 B0 분포
def x4(series):
    s = series.dropna()
    for X in np.arange(0.03,0.60,0.005):
        if eps_rate((s<=np.log(1-X)).values)/2.0 <= 4: return X
    return np.nan
cov=[]; th={}
for s in st+sorted(etf):
    th[s] = x4(dd[s][mask])
for grp,names in (('주식',st),('ETF',sorted(etf))):
    vals=[]
    for s in names:
        X=th[s]
        if not np.isfinite(X): continue
        sel=(dd[s]<=np.log(1-X))&(C.index>='2026-01-02')&Z[s].notna()
        vals += list(Z[s][sel].values)
    v=np.array(vals); print(f'{grp} 자기 이력 연4번 낙폭 도달일(2026) {len(v)}칸: 하방계열 {(v<0).mean():.1%} 애매 {(v==0).mean():.1%} 상방계열 {(v>0).mean():.1%}')
for s in ['SOXX','EWY','SMH','QQQ','MU','NVDA']:
    print(s, '연4번 낙폭', round(th[s]*100,1),'% / 지금', round((np.exp(dd[s].iloc[-1])-1)*100,1),'%')
print('주식 연4번 낙폭 중앙', round(np.nanmedian([th[s] for s in st])*100,1), '%')
