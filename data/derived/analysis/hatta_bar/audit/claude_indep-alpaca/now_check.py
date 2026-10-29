"""사용자 실시간 예시 NOW 2026-10-29 09:38(①)·09:39(②)를 내 구현이 잡는지, 그리고 ① 경계값(아랫꼬리 정확히 0.5) 수."""
import numpy as np, pandas as pd
import recompute as R
df = pd.read_parquet(R.BASE + '/cache/now_live.parquet')
rows = pd.DataFrame(R.process('NOW', -1, None, df), columns=R.COLS)
rows['et'] = pd.to_datetime(rows.ts * 60, unit='s', utc=True).dt.tz_convert('America/New_York').dt.strftime('%m-%d %H:%M')
d = rows[(rows.day == 20261029)]
print(d[['arm', 'et', 'S', 'level', 'touch', 'dd', 'openok', 'wick', 'shape', 'fixed', 'valid', 'tradable', 'hit', 'r1']].to_string(index=False))

m = pd.read_csv(R.OUT + '/mine_A.csv')
b = m[np.isclose(m.wick, 0.5, atol=1e-12)]
print('① 신호 중 아랫꼬리 == 0.5(1e-12 안):', len(b), '/', len(m))
# 곱셈식 (min(O,C)-L) >= 0.5*(H-L) 로 바꾸면 뒤집히는 수
flip = 0
for sym, g in b.groupby('sym'):
    x = pd.read_parquet(R.BASE + f'/cache/alp1m/{sym}.parquet')
    for t in g.t:
        o, h, l, c = x.Open.iat[t], x.High.iat[t], x.Low.iat[t], x.Close.iat[t]
        a1 = (min(o, c) - l) / (h - l) >= 0.5
        a2 = (min(o, c) - l) >= 0.5 * (h - l)
        flip += a1 != a2
print('나눗셈식 vs 곱셈식 판정이 뒤집히는 경계 신호:', flip)
