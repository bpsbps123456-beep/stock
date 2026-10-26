"""rare_compare.py 핵심 함수 대조: 벡터화 에피소드 vs 단순 루프, 백분위 vs np.percentile, X4 루프 재계산."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rare_compare as RC

def eps_loop(dd, thr, rec):
    n = 0; on = False
    for v, t, r in zip(dd, thr, rec):
        if not (np.isfinite(v) and np.isfinite(t) and np.isfinite(r)):
            continue
        if not on and v <= t: n += 1; on = True
        elif on and v > r: on = False
    return n

rng = np.random.default_rng(0)
bad = 0
for _ in range(300):
    n = 300
    lc = np.cumsum(rng.normal(0, 0.03, n))
    C = np.exp(lc)
    dd = np.log(C / pd.Series(C).rolling(20, min_periods=20).max().to_numpy())
    a = -np.abs(rng.normal(0.1, 0.03, n))
    a[rng.random(n) < 0.05] = np.nan
    t, r = RC.thr_from_log(a)
    nv, _, _ = RC.episodes(dd[None, :], t[None, :], r[None, :])
    if nv[0] != eps_loop(dd, t, r): bad += 1
print('episodes 불일치', bad, '/300')
x = rng.normal(size=600); x[:5] = np.nan
S = RC.roll_pct_sorted(x)
for q in (0.5, 12.5, 87.5, 99.5):
    v = RC.pct_from_sorted(S, q)
    ref = np.array([np.percentile(x[i-251:i+1], q, method='linear') if i >= 251 and np.isfinite(x[i-251:i+1]).all() else np.nan for i in range(600)])
    print('pct', q, 'maxdiff', np.nanmax(np.abs(v - ref)), 'nan일치', (np.isnan(v) == np.isnan(ref)).all())
# X4 루프 재계산(주식 전부, 2026 평가용 이력)
raw = pd.read_parquet(RC.PANEL); raw['ts'] = pd.to_datetime(raw['ts'])
mis = 0; tot = 0
for sym, g in raw.groupby('sym'):
    g = g.sort_values('ts'); C = g['Close'].to_numpy(); d = g['ts'].dt.strftime('%Y-%m-%d').to_numpy()
    dd = np.log(C / pd.Series(C).rolling(20, min_periods=20).max().to_numpy())
    w = (d >= '2024-01-02') & (d <= '2025-12-31') & np.isfinite(dd)
    if w.sum() < 400: continue
    yrs = w.sum() / 252; best = np.nan
    for X in np.round(np.arange(0.0025, 0.8000001, 0.0025), 4):
        k = eps_loop(dd[w], np.full(w.sum(), np.log(1-X)), np.full(w.sum(), np.log(1-X/2)))
        if k / yrs >= 4: best = X
    tot += 1
    if not np.isclose(best, RC.x4_of(dd[w], yrs)): mis += 1; print('X4 차이', sym, best, RC.x4_of(dd[w], yrs))
print('X4 대조', tot, '종목, 불일치', mis)
