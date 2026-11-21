"""40분봉 격자의 이동 전방 검증 (2026-11-22, AGENTS.md 연구 보고 규칙).
달마다 직전 L개월(3·6)의 시점 값(편도 0.1% 차감) 종목 중앙값이 가장 큰 설정 하나를 골라 다음 달 9종목 모두에 쓴다.
비교: 고정 느린 선(일봉 EMA10·20 종가), 고정 A(40분봉 EMA100 닿으면·장 전 판단), 보유. 결과: ../results/walkforward40.json
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid40 as G  # noqa: E402
E = G.E

COST = 0.001
WIN = {"2026": ("2026-01-02", "2026-11-20"), "최근 분기": ("2026-08-20", "2026-11-20"), "급락 구간": ("2026-10-15", "2026-11-20")}
cfgs = G.configs()
months = pd.period_range("2025-01", "2026-11", freq="M")
NM = len(months)
store = {}
tv = np.zeros((len(cfgs), len(G.SYMS), NM))
for s_i, sym in enumerate(G.SYMS):
    E.use(sym)
    B = E._base(); N1 = len(B["r"]); dates = pd.DatetimeIndex(B["date"])
    starts = [int(np.searchsorted(dates.values, np.datetime64(p.start_time.date()))) for p in months] + [N1 - 1]
    cache = {}
    for i, c in enumerate(cfgs):
        pos, corr, sw = G.run(sym, c, cache)
        inc = pos * B["r"] + corr
        cs, cp, cr = (np.r_[0.0, np.cumsum(x)] for x in (inc, pos, B["r"]))
        for m in range(NM):
            a, z = starts[m], starts[m + 1]
            n = max(z - a, 1)
            nsw = int(((sw >= a) & (sw < z)).sum())
            tv[i, s_i, m] = (cs[z] - cs[a]) - (cp[z] - cp[a]) / n * (cr[z] - cr[a]) - COST * nsw
    store[sym] = starts
    print(sym, flush=True)

res = {"variants": {}}
for LB in (3, 6):
    sel = {m: int(np.argmax(np.median(tv[:, :, m - LB:m].sum(axis=2), axis=1))) for m in range(LB, NM)}
    per = {}
    for s_i, sym in enumerate(G.SYMS):
        E.use(sym)
        B = E._base(); N1 = len(B["r"]); starts = store[sym]
        cache = {}
        paths = {i: G.run(sym, cfgs[i], cache) for i in set(sel.values())}
        pos = np.ones(N1); corr = np.zeros(N1); sws = []; prev = None
        for m in range(LB, NM):
            p, c_, s = paths[sel[m]]
            a, z = starts[m], starts[m + 1]
            pos[a:z] = p[a:z]; corr[a:z] = c_[a:z]
            sws += [int(x) for x in s[(s >= a) & (s < z)]]
            if prev is not None and p[a] != prev:
                sws.append(a)
            prev = p[z - 1]
        sws = np.array(sorted(set(sws)), dtype=int)
        per[sym] = {w: E.evaluate(pos, corr, sws, rng, COST) for w, rng in WIN.items()}
    res["variants"][f"직전 {LB}개월 1위"] = {"per_symbol": per, "picks": [{"month": str(months[m]), "pick": G.name(cfgs[sel[m]])} for m in range(LB, NM)]}

# 고정 규칙
fixed = {"느린 선": {}, "A 40분 EMA100 닿으면": {}, "보유": {}}
A = dict(d=10, b=0.0, a=0.0, mode="touch", tm="T2")
for sym in G.SYMS:
    E.use(sym)
    slow = G.refs(sym)["느린 선(일봉 EMA10·20 종가)"]
    fixed["느린 선"][sym] = {w: E.evaluate(*slow, rng, COST) for w, rng in WIN.items()}
    pa = G.run(sym, A, {})
    fixed["A 40분 EMA100 닿으면"][sym] = {w: E.evaluate(*pa, rng, COST) for w, rng in WIN.items()}
res["fixed"] = fixed

rows = []
for nm, per in [(k, v["per_symbol"]) for k, v in res["variants"].items()] + [(k, v) for k, v in fixed.items() if k != "보유"]:
    r = {"규칙": nm}
    for w in WIN:
        r[f"{w} 수익% 중앙"] = round(float(np.median([per[s][w]["ret%"] for s in G.SYMS])), 1)
        r[f"{w} 낙폭% 중앙"] = round(float(np.median([per[s][w]["mdd%"] for s in G.SYMS])), 1)
    rows.append(r)
r = {"규칙": "보유"}
for w in WIN:
    r[f"{w} 수익% 중앙"] = round(float(np.median([fixed["느린 선"][s][w]["bh%"] for s in G.SYMS])), 1)
    r[f"{w} 낙폭% 중앙"] = round(float(np.median([fixed["느린 선"][s][w]["bh_mdd%"] for s in G.SYMS])), 1)
rows.append(r)
res["table"] = rows
json.dump(res, open(os.path.join(HERE, "..", "results", "walkforward40.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
for k, v in res["variants"].items():
    print(k, "고른 설정 종류", len(set(p["pick"] for p in v["picks"])), "| 2026 고른 것:", sorted(set(p["pick"] for p in v["picks"] if p["month"].startswith("2026")))[:4])
