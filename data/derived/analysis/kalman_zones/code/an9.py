"""최종 설정의 현재 구간·라인(대조용), 구간 체류 비율, 7월 급락 때 동작."""
import numpy as np, pandas as pd
from core import *
from evalcfg import data, indicator
NAME = {2: "과매수", 1: "매수", 0: "평범", -1: "매도", -2: "패닉셀"}
D, H = data()
L, NORM, N, A, B = 21, "rms", 50, 0.75, 1.75
print(f"칼만 Process Noise = {gain_to_q(length_to_gain(L)):.5f} (Measurement Noise 3.0, 정상상태 이득 {length_to_gain(L):.4f})")
rows, share = [], []
for s in SYMS:
    tb = tf_bars(D[(s, "1h")], 240, False, H)
    K, u, z = indicator(tb, L, NORM, N, A, B)
    i = len(tb) - 1
    c = tb.C.values[i]
    lines = {"과매수": K[i] + B * u[i], "매수": K[i] + A * u[i], "평범(중심)": K[i], "매도": K[i] - A * u[i], "패닉셀": K[i] - B * u[i]}
    rows.append({"종목": s, "종가": c, "구간": NAME[int(z[i])], **{k: v for k, v in lines.items()},
                 "매도라인까지": lines["매도"] / c - 1, "패닉라인까지": lines["패닉셀"] / c - 1, "과매수라인까지": lines["과매수"] / c - 1})
    yr = pd.to_datetime(tb.end_ns.values, utc=True).year == 2026
    zz = z[yr]; share.append(pd.Series(zz).map(lambda v: NAME.get(int(v), "-")).value_counts(normalize=True).rename(s))
t = pd.DataFrame(rows)
print("\n마지막 240분 정규장 봉(9/25 13:30~16:00) 기준")
print(t.to_string(index=False, formatters={**{c: "{:,.2f}".format for c in ["종가", "과매수", "매수", "평범(중심)", "매도", "패닉셀"]},
                                           **{c: "{:+.1%}".format for c in ["매도라인까지", "패닉라인까지", "과매수라인까지"]}}))
print("\n2026년 240분봉의 구간 체류 비율")
print((pd.concat(share, axis=1).T * 100).round(0).reindex(columns=["과매수", "매수", "평범", "매도", "패닉셀"]).to_string())
