"""soxl_lines.pine 의 core() 상태 기계를 파이썬으로 옮겨 엔진(engine.simulate)의 전환과 대조한다(2026-11-20).
기준 봉마다: 보유 중이면 (닿으면: 저가 <= 직전 봉 경고선 / 종가 확인: 종가 < 경고선) 대피, 대피 중이면 종가 > 재매수선 복귀. 한 봉 한 번.
경고선 = min(EMA a, EMA b)×(1−띠), 재매수선 = max(EMA a, EMA b)×(1+띠). 빠른 선은 a=b=200, 띠 2%.
엔진 전환(1분봉 위치)을 기준 봉 번호로 바꿔(닿으면 대피 = 그 1분봉이 든 봉, 나머지 = 체결 전에 마지막으로 끝난 봉) 미러 전환 목록과 비교한다.
결과: ../results/pine_mirror.json"""
import json, os
import numpy as np, pandas as pd
import engine as E

OUT = os.path.join(E.ROOT, "results", "pine_mirror.json")
TN = pd.DatetimeIndex(E.load("SOXL").index).tz_convert("UTC").tz_localize(None).to_numpy()


def lines(b, a, c, band):
    ea, ec = E.ema(b["Close"], a), E.ema(b["Close"], c)
    return np.minimum(ea, ec) * (1 - band), np.maximum(ea, ec) * (1 + band)


def mirror(b, a, c, stop, band=0.0):
    C, Lo = b["Close"].to_numpy(float), b["Low"].to_numpy(float)
    W, R = (x.to_numpy() for x in lines(b, a, c, band))
    st, out = True, np.zeros(len(b), bool)
    for j in range(len(b)):
        if st:
            if (stop and j > 0 and Lo[j] <= W[j - 1]) or (not stop and C[j] < W[j]):
                st = False
        elif C[j] > R[j]:
            st = True
        out[j] = st
    return out, W, R


def parity(tf, sess, a, c, stop, band):
    b = E.bars("SOXL", tf, sess)
    W, R = lines(b, a, c, band)
    kw = dict(exit_touch=W.shift(1)) if stop else dict(exit_close=b["Close"] < W)
    pos, corr, sw = E.simulate(b, **kw, entry_close=b["Close"] > R, exec_sess=sess)
    starts = pd.DatetimeIndex(b.index).tz_convert("UTC").tz_localize(None).to_numpy()
    ends = pd.DatetimeIndex(b["end"]).tz_convert("UTC").tz_localize(None).to_numpy()
    eng = []
    for p in sw:
        if pos[p] == 0 and stop:
            eng.append((int(np.searchsorted(starts, TN[p], side="right") - 1), "out"))
        else:
            eng.append((int(np.searchsorted(ends, TN[p], side="right") - 1), "in" if pos[p] else "out"))
    m, Wm, Rm = mirror(b, a, c, stop, band)
    chg = np.nonzero(np.diff(np.r_[True, m].astype(int)) != 0)[0]
    mir = [(int(j), "in" if m[j] else "out") for j in chg]
    diff = [(str(b.index[x[0]])[:16], x[1]) for x in sorted(set(eng) ^ set(mir))]
    return {"engine_switches": len(eng), "mirror_switches": len(mir), "same": eng == mir, "diff": diff[:10],
            "now": {"bar": str(b.index[-1])[:16], "close": round(float(b["Close"].iloc[-1]), 2), "경고선": round(float(Wm[-1]), 2),
                    "재매수선": round(float(Rm[-1]), 2), "보유": bool(m[-1])}}


res = {}
for tag, args in (("느린 선 종가 확인(일봉 EMA10·20)", ("D", "reg", 10, 20, False, 0.0)),
                  ("느린 선 닿으면(일봉 EMA10·20)", ("D", "reg", 10, 20, True, 0.0)),
                  ("빠른 선 종가 확인(5분 EMA200 ±2%)", (5, "ext", 200, 200, False, 0.02)),
                  ("참고: 연장 240분 EMA10·50 닿으면", (240, "ext", 10, 50, True, 0.0))):
    res[tag] = parity(*args)
    print(tag, {k: v for k, v in res[tag].items() if k != "now"}, res[tag]["now"])
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
