"""진단: 손절선(종가가 매수가 -X%)에 처음 닿은 순간 40분봉 RSI가 30 이하였는지에 따라, 그때 팔았을 때와 끝까지 들고 갔을 때를 비교한다.
사용자 가설(10/7): 40분봉 RSI 30 이하(이미 과매도)면 손절하면 안 좋다. 사용: py stop_rsi40_diag.py"""
import numpy as np, pandas as pd
from stop_lab import load, base_trades, htf_rsi, PNAME, USER78

for p in ("early", "late"):
    S = load(p); b = base_trades(p)
    for X in (0.02, 0.03, 0.05):
        rows = []
        for x in b.itertuples():
            A = S[x.sym]; c, o = A["c"], A["o"]
            dev, conf = htf_rsi(A)
            for i in range(x.ei + 1, x.xi):            # 들고 있는 동안 봉 i 시가에 판단(i-1 종가 기준)
                if c[i - 1] <= x.ent * (1 - X):
                    rows.append(dict(sym=x.sym, reg=A["reg"][i - 1], dev=dev[i], conf=conf[i],
                                     stop=o[i] / x.ent - 1, hold=x.ret))
                    break
        d = pd.DataFrame(rows)
        d["득실"] = d.stop - d.hold                       # 팔았을 때 - 들고 갔을 때 (+면 손절이 이득)
        out = []
        for kind in ("dev", "conf"):
            for g in (25, 30, 35, 40):
                lo, hi = d[d[kind] <= g], d[d[kind] > g]
                out.append(f"{kind}≤{g}: {len(lo)}건 손절 득실 {lo.득실.mean():+.2%}(합 {lo.득실.sum():+.2f}) | >{g}: {len(hi)}건 {hi.득실.mean():+.2%}(합 {hi.득실.sum():+.2f})")
        print(f"\n[{PNAME[p]}] 종가 -{X:.0%} 첫 도달 {len(d)}건 (137종목), 전체 손절 득실 평균 {d.득실.mean():+.2%} 합 {d.득실.sum():+.2f}")
        print("  " + "\n  ".join(out))
