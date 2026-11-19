"""갈래 exit_ma 기전 확인(선택 대상 아님): 약세 이탈 대신 '강세(상향 교차)에서 파는' 거울 규칙의 TRAIN x_rand.
가격 하락이 방아쇠인 손절이 무작위보다 나쁜 이유가 짧은 평균회귀인지 확인용. 4설정(EMA25·60 × 1·3분)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
out = {}
for tf in (1, 3):
    b = H.resample(tf); c = b["Close"]
    for n in (25, 60):
        e = H.ema(c, n)
        up = (c > e) & (c.shift(1) <= e.shift(1))
        p = H.exec_index(up, b); p = p[p >= 0]
        out[f"ema{n}_cross_UP_event_{tf}m"] = {lab: H.hold_sim(p, rng)["x_rand_mean%"] for rng, lab in ((H.TRAIN, "TRAIN"),)}
        out[f"ema{n}_cross_UP_event_{tf}m"]["TRAIN_t"] = H.hold_sim(p, H.TRAIN)["x_rand_t_days"]
json.dump(out, open(os.path.join(H.ROOT, "results", "exit_ma_mirror.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(out)
