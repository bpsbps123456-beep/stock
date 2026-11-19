"""갈래 exit_ma 진단: 3분봉 EMA60 종가 하향 이탈이 왜 무작위 탈출보다 나쁜가 (TRAIN만).

- 이탈 순간의 단기 과매도(RSI14·Stoch K·직전 20봉 저점 거리·직전 20봉 고점 낙폭)
- 이탈 뒤 r15/r30/r60(체결가 기준)과 같은 날 ±30분 대조의 차
- RSI 구간별 쪼개기
- EMA 1봉 기울기 꺾임 = 종가 교차와 수학적으로 같은지 확인
결과: results/exit_ma_diag.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

d1 = H.load_1m()
f = H._fwd_tables()
out = {}
for tf in (3, 1):
    b = H.resample(tf)
    c = b["Close"]
    e60 = H.ema(c, 60)
    cross = (c < e60) & (c.shift(1) >= e60.shift(1))
    slope = (e60 < e60.shift(1)) & (e60.shift(1) >= e60.shift(2))
    out[f"slope1_equals_cross_{tf}m"] = bool((cross.fillna(False) == slope.fillna(False)).all())
    r = H.rsi(c, 14)
    K, D = H.stoch_rsi(c)
    lo20 = b["Low"].rolling(20, min_periods=1).min()
    hi20 = b["High"].rolling(20, min_periods=1).max()
    feat = pd.DataFrame({"rsi": r, "K": K, "dlo20": c / lo20 - 1, "dhi20": c / hi20 - 1})
    sig = cross.fillna(False)
    pos = H.exec_index(sig, b)
    fe = feat[sig.values].copy()
    fe["pos"] = pos
    fe = fe[fe.pos >= 0]
    fe["date"] = d1["date"].values[fe.pos]
    fe["reg"] = d1["reg"].values[fe.pos]
    fe = fe[fe.reg & H.in_range(fe.date.values, H.TRAIN)]
    # 대조: TRAIN 정규장 같은 tf 봉 전체의 특징 분포
    bs = b.reg & H.in_range(b["date"].values, H.TRAIN)
    base_feat = feat[bs.values]
    ev = H.event_metrics(fe.pos.values)
    for h in ("r15", "r30", "r60"):
        dm = H.day_matched(ev, h)
        fe[h] = ev[h].values
        fe[h + "_xdm"] = ev[h].values - dm
    blk = {"n_raw_cross(no dedup)": int(len(fe)),
           "feat_at_cross_median": {k: round(float(fe[k].median()), 4) for k in ("rsi", "K", "dlo20", "dhi20")},
           "feat_all_bars_median": {k: round(float(base_feat[k].median()), 4) for k in ("rsi", "K", "dlo20", "dhi20")},
           "pct_rsi_below35_at_cross": round(100 * float((fe.rsi < 35).mean()), 1),
           "pct_rsi_below35_all": round(100 * float((base_feat.rsi < 35).mean()), 1),
           "pct_K_below20_at_cross": round(100 * float((fe.K < 20).mean()), 1),
           "pct_K_below20_all": round(100 * float((base_feat.K < 20).mean()), 1),
           "fwd_mean%(price after exit; +=rebound=bad exit)": {h: round(100 * float(fe[h].mean()), 3) for h in ("r15", "r30", "r60")},
           "fwd_xdm_mean%": {h: round(100 * float(fe[h + "_xdm"].mean()), 3) for h in ("r15", "r30", "r60")}}
    buckets = {}
    for lab, m in (("rsi<35", fe.rsi < 35), ("35<=rsi<45", (fe.rsi >= 35) & (fe.rsi < 45)), ("rsi>=45", fe.rsi >= 45)):
        s = fe[m]
        buckets[lab] = {"n": int(len(s)), **{h + "_mean%": round(100 * float(s[h].mean()), 3) for h in ("r15", "r30", "r60")},
                        **{h + "_xdm%": round(100 * float(s[h + "_xdm"].mean()), 3) for h in ("r15", "r30", "r60")}}
    blk["by_rsi_at_cross"] = buckets
    out[f"ema60_cross_{tf}m_TRAIN"] = blk

# 대조 확인: 1분봉 '아무 시점'의 단기 자기상관(15분 하락 뒤 다음 30분) — 평균회귀 장세인지
d = d1
sel = d.reg.values & H.in_range(d["date"].values, H.TRAIN)
C = d["Close"].to_numpy(float)
past15 = pd.Series(C).pct_change(15).to_numpy()
fw = f["r30"]
idx = np.nonzero(sel)[0]
q = pd.qcut(past15[idx], 5, labels=False, duplicates="drop")
out["TRAIN_mean_reversion_check"] = {
    "desc": "정규장 1분봉, 직전 15분 수익 5분위별 다음 봉 시가 체결 기준 30분 수익 평균%(1=가장 많이 빠진 분위)",
    **{f"q{int(k)+1}": round(100 * float(np.nanmean(fw[idx][q == k])), 3) for k in np.unique(q[~np.isnan(q)])}}

with open(os.path.join(H.ROOT, "results", "exit_ma_diag.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
