"""최종 두 규칙의 2026 대피·복귀 일지와 현재 상태 (2026-11-20). 결과: ../results/final_state.json"""
import json, os
import numpy as np, pandas as pd
import engine as E

d1 = E.load("SOXL"); O = d1["Open"].to_numpy(float)
out = {}
# 느린 선: 정규 일봉 EMA10·20 종가
bd = E.bars("SOXL", "D", "reg"); Cd = bd.Close; a, c = E.ema(Cd, 10), E.ema(Cd, 20); W, R = np.minimum(a, c), np.maximum(a, c)
pos, corr, sw = E.simulate(bd, exit_close=Cd < W, entry_close=Cd > R, exec_sess="reg")
log = []
for p in sw:
    if d1.index[p].year == 2026:
        log.append({"체결": str(d1.index[p])[:16], "행동": "재매수" if pos[p] else "대피", "가격": round(float(O[p]), 2)})
# 대피 구간별: 판 가격 → 되산 가격
ep = []
s26 = [p for p in sw if d1.index[p].year == 2026]
for i, p in enumerate(s26):
    if pos[p] == 0:
        q = s26[i + 1] if i + 1 < len(s26) else None
        back = float(O[q]) if q is not None else float(d1["Close"].iloc[-1])
        ep.append({"대피": str(d1.index[p])[:10], "대피가": round(float(O[p]), 2), "재매수": str(d1.index[q])[:10] if q is not None else "아직 대피 중",
                   "재매수가(또는 현재가)": round(back, 2), "대피 중 SOXL 변동%": round(100 * (back / O[p] - 1), 1)})
out["slow"] = {"rule": "정규 일봉 EMA10·20: 종가 < 낮은 선 대피, 종가 > 높은 선 재매수, 다음 정규장 시가 체결", "log_2026": log, "episodes_2026": ep,
               "now": {"date": str(bd.index[-1].date()), "close": float(Cd.iloc[-1]), "경고선": round(float(W.iloc[-1]), 2), "재매수선": round(float(R.iloc[-1]), 2), "보유": bool(pos[-1])}}
# 빠른 선: 연장 5분 EMA200 ±2%
b = E.bars("SOXL", 5, "ext"); C = b.Close; e = E.ema(C, 200)
pos2, corr2, sw2 = E.simulate(b, exit_close=C < e * 0.98, entry_close=C > e * 1.02, exec_sess="ext")
conf = [p for p in sw2 if d1.index[p] >= pd.Timestamp("2026-10-01", tz=E.ET)]
out["fast"] = {"rule": "연장 5분봉 EMA200: 종가 < 선×0.98 대피, 종가 > 선×1.02 재매수, 다음 1분봉 시가 체결",
               "switches_since_1001": len(conf),
               "now": {"time": str(b.index[-1])[:16], "close": float(C.iloc[-1]), "EMA200": round(float(e.iloc[-1]), 2), "경고선": round(float(e.iloc[-1] * 0.98), 2),
                       "재매수선": round(float(e.iloc[-1] * 1.02), 2), "보유": bool(pos2[-1])}}
json.dump(out, open(os.path.join(E.ROOT, "results", "final_state.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(pd.DataFrame(ep).to_string(index=False)); print(out["slow"]["now"]); print(out["fast"]["now"], out["fast"]["switches_since_1001"])
