"""broad.py 의 일봉 평가가 1분봉 엔진(soxl_escape_20261120/lines/code/engine.py)과 같은 값을 내는지 대조한다(2026-11-23).
느린 선(띠 EMA10·20 종가)과 보통 선(띠 EMA3·15 닿으면)을 9종목에서 같은 구간(2026-01-02~10-14)으로 비교한다.
엔진 구간은 마지막 날 장 마감 뒤까지, broad 는 다음 날 시가까지라 마지막 밤사이 갭만큼 다를 수 있다.
결과: ../results/check_engine.json"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../soxl_escape_20261120/lines/code"))
import broad as B  # noqa: E402
import engine as E  # noqa: E402

RNG = ("2026-01-02", "2026-10-14")
out = {}
for sym in ["SOXL", "MU", "AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"]:
    if not os.path.exists(os.path.join(B.CACHE, f"{sym}.parquet")):
        continue
    D = B.load(sym)
    E.use(sym)
    bd = E.bars(sym, "D", "reg"); C = bd["Close"]
    row = {}
    for nm, (a, b_, touch) in {"느린 선": (10, 20, False), "보통 선": (3, 15, True)}.items():
        W, R, t = B.lines(D, ("band", a, b_, touch))
        ev = B.evaluate(D, *B.simulate(D, W, R, t), RNG)
        ea, eb = E.ema(C, a), E.ema(C, b_)
        We, Re = np.minimum(ea, eb), np.maximum(ea, eb)
        kw = dict(exit_touch=We.shift(1)) if touch else dict(exit_close=C < We)
        ee = E.evaluate(*E.simulate(bd, **kw, entry_close=C > Re, exec_sess="reg"), RNG, B.COST)
        row[nm] = {"broad 수익%": round(ev["ret%"], 2), "엔진 수익%": ee["ret%"], "broad 전환": ev["sw"], "엔진 전환": ee["switches"],
                   "broad 비중%": round(ev["expo%"], 1), "엔진 비중%": ee["expo%"]}
    out[sym] = row
    print(sym, row, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "results", "check_engine.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
