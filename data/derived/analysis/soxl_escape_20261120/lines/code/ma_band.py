"""Claude 갈래 2차 — 두 이평선 띠(사전 선언 60개, 2026-11-20). 1차(ma_scan.py)에서 빠른 선은 비용에 무너지고 느린 선(2시간~일봉)이 세 구간에서 버텼다.
사용자 개념(경고선 아래면 대피, 재매수선 위로 올라타면 복귀)에 맞는 깔끔한 두 줄: 경고선 W = min(EMA a, EMA b), 재매수선 R = max(EMA a, EMA b).
  종가 < W 대피, 종가 > R 복귀, 두 선 사이에서는 직전 상태 유지(왔다 갔다 없음).
 R1 종가 판정 30: 쌍{(10,20),(10,50),(20,50),(20,100),(50,100)} × tf{120,240,D} × 세션{reg,ext}
 R2 닿으면 대피(스탑, 직전 봉의 W) + 종가 > R 복귀 30: 같은 격자
결과: ../results/ma_band.json"""
import json, os
import numpy as np, pandas as pd
import engine as E

OUT = os.path.join(E.ROOT, "results", "ma_band.json")
WINS = ("SEL", "CONF", "Y2025", "B1", "B2")
rows = []
for tf in (120, 240, "D"):
    for sess in ("reg", "ext"):
        b = E.bars("SOXL", tf, sess)
        C = b["Close"]
        for a, c in ((10, 20), (10, 50), (20, 50), (20, 100), (50, 100)):
            ea, ec = E.ema(C, a), E.ema(C, c)
            W, R = np.minimum(ea, ec), np.maximum(ea, ec)
            for mode in ("close", "touch"):
                if mode == "close":
                    pos, corr, sw = E.simulate(b, exit_close=C < W, entry_close=C > R, exec_sess=sess)
                else:
                    pos, corr, sw = E.simulate(b, exit_touch=W.shift(1), entry_close=C > R, exec_sess=sess)
                sc = E.score_all(pos, corr, sw, WINS)
                rows.append({"name": f"R:{mode}:tf={tf},sess={sess},pair=({a},{c})", "cfg": dict(fam="R", mode=mode, tf=tf, sess=sess, a=a, c=c), **sc})
json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
L = lambda x: 100 * np.log1p(x / 100)
t = pd.DataFrame([dict(name=r["name"], S_sw=r["SEL"]["switches"], S_expo=r["SEL"]["expo%"], S_tvnet=round(r["SEL"]["tv%p"] - 0.1 * r["SEL"]["switches"], 1),
                       S_t=r["SEL"]["tv_t"], S_ret=r["SEL"]["ret%"], S_mdd=r["SEL"]["mdd%"], C_sw=r["CONF"]["switches"], C_expo=r["CONF"]["expo%"],
                       C_ret=r["CONF"]["ret%"], C_mdd=r["CONF"]["mdd%"],
                       C_tvL=round(L(r["CONF"]["ret%"]) - r["SEL"]["expo%"] / 100 * L(r["CONF"]["bh%"]) - 0.1 * r["CONF"]["switches"], 1),
                       Y_tvnet=round(r["Y2025"]["tv%p"] - 0.1 * r["Y2025"]["switches"], 1), Y_ret=r["Y2025"]["ret%"], Y_mdd=r["Y2025"]["mdd%"]) for r in rows])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(t.sort_values("S_tvnet", ascending=False).to_string(index=False))
print("세 구간 모두 양수", ((t.S_tvnet > 0) & (t.C_tvL > 0) & (t.Y_tvnet > 0)).mean().round(2), "| 중앙값", t[["S_tvnet", "C_tvL", "Y_tvnet", "S_mdd", "Y_mdd"]].median().round(1).to_dict())
