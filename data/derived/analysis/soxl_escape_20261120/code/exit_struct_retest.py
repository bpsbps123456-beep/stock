"""갈래 exit_struct 4단계: '이탈 뒤 되돌림에서 팔기'와 '이탈 뒤 2차 하락에서 팔기' (2026-11-20).

1~3단계(exit_struct_scan.py)에서 이탈 순간 매도는 무작위 탈출보다 대체로 나빴다(이탈 순간 = 단기 저점).
그래서 이탈 사건을 '경보'로만 쓰고, 실제 매도는 그 뒤 60분 안의 다음 자리에서 한다.
  rt_touch : 이탈 때 선 값(고정)까지 고가가 되돌아온 봉(되돌림 매도)
  rt_close : 종가가 이탈선 값 이상으로 다시 올라선 봉(회복 매도)
  bounce05 / bounce10 : 이탈 뒤 최저가 대비 종가 +0.5% / +1.0% 반등 봉
  second   : 이탈 뒤 최저가(직전 봉까지)를 종가가 다시 깬 봉(2차 하락)
기준선 6종(hl9m, pivot15m, donch60m, vwap, or30_low, pm_low; 각 계열 대표를 1단계 전에 고정) x 시간봉 2 x 매도 자리 5 = 60 설정.
확인방식은 종가 이탈(c0) 하나, 거래량 거르개 없음, 창 60분 하나로 고정.
실행: py code/exit_struct_retest.py → results/exit_struct.json 의 'stage4'와 최종 후보를 갱신.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import exit_struct_scan as S  # noqa: E402

LINES = ("hl9m", "pivot15m", "donch60m", "vwap", "or30_low", "pm_low")
MODES = ("rt_touch", "rt_close", "bounce05", "bounce10", "second")
WIN_MIN = 60


def follow_signal(bars: pd.DataFrame, line: pd.Series, tf: int, mode: str) -> pd.Series:
    """이탈(종가<선, 직전 봉은 아님) 뒤 WIN_MIN분 안 첫 매도 자리 봉 = True. 같은 날 안에서만. 한 번 쏘면 무장 해제."""
    c = bars["Close"].to_numpy(float)
    h = bars["High"].to_numpy(float)
    lo = bars["Low"].to_numpy(float)
    ln = line.to_numpy(float)
    dates = bars["date"].values
    below = c < ln
    brk = below & ~np.r_[False, below[:-1]]
    W = WIN_MIN // tf
    out = np.zeros(len(c), dtype=bool)
    armed, lvl, start, mn, dt = False, np.nan, -1, np.nan, None
    for i in range(len(c)):
        if armed and (dates[i] != dt or i - start > W):
            armed = False
        if armed:
            if mode == "rt_touch":
                fire = h[i] >= lvl
            elif mode == "rt_close":
                fire = c[i] >= lvl
            elif mode == "bounce05":
                fire = c[i] >= mn * 1.005
            elif mode == "bounce10":
                fire = c[i] >= mn * 1.010
            else:  # second
                fire = c[i] < mn
            if fire:
                out[i] = True
                armed = False
            mn = min(mn, lo[i]) if armed else mn
        if brk[i] and not np.isnan(ln[i]):
            armed, lvl, start, mn, dt = True, ln[i], i, lo[i], dates[i]
    return pd.Series(out, index=bars.index)


def main():
    with open(S.OUT, encoding="utf-8") as fh:
        res = json.load(fh)
    res["stage4"] = {}
    for tf in (1, 3):
        bars = S.TF[tf]
        lines = S.build_lines(tf)
        for ln in LINES:
            for mode in MODES:
                sig = follow_signal(bars, lines[ln], tf, mode)
                res["stage4"][f"{tf}m|{ln}|c0|{mode}|event"] = S.score(sig, bars, "event")
        print("stage4 tf", tf, flush=True)
    allc = {}
    for st in ("stage1", "stage2", "stage3", "stage4"):
        allc.update(res[st])
    res["configs_tested"] = len(allc)
    elig = {k: v for k, v in allc.items() if S.ok_freq(v)}
    top3 = sorted(elig, key=lambda k: -elig[k]["train"]["primary"])[:3]
    res["final_top3"] = {k: allc[k] for k in top3}
    pr = np.array([v["train"]["primary"] for v in elig.values()])
    pt = np.array([v["test"]["primary"] for v in elig.values()])
    res["train_primary_dist"] = {"n_eligible": int(len(pr)), "mean": round(float(pr.mean()), 3),
                                 "p90": round(float(np.quantile(pr, 0.9)), 3), "max": round(float(pr.max()), 3),
                                 "share_positive": round(float((pr > 0).mean()), 3)}
    res["train_test_rank_corr"] = round(float(pd.Series(pr).corr(pd.Series(pt), method="spearman")), 3)
    s4 = {k: v for k, v in res["stage4"].items() if S.ok_freq(v)}
    by_mode = {}
    for k, v in s4.items():
        m = k.split("|")[3]
        by_mode.setdefault(m, []).append((v["train"]["primary"], v["test"]["primary"], v["pre"]["primary"]))
    res["stage4_mode_mean"] = {m: [round(float(np.mean([x[j] for x in xs])), 3) for j in range(3)] + [len(xs)]
                               for m, xs in by_mode.items()}
    with open(S.OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print("configs", len(allc), res["stage4_mode_mean"])
    for k in top3:
        print(k, json.dumps(allc[k]["train"]), "\n   TEST", json.dumps(allc[k]["test"]), "\n   PRE", allc[k]["pre"]["primary"])


if __name__ == "__main__":
    main()
