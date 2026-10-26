"""감사 보조: SMCI 복원 효과의 출처, 라벨 공정 데스크, desk_system §5 수치 대조 (2026-10-10).

verify_bucket_holdout.py의 독립 엔진을 그대로 쓴다. 쓰기: results/bucket/verify_holdout_smci.json 만.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/verify_bucket_holdout_smci.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_bucket_holdout as V  # noqa: E402
from desk.strategies import UNKNOWN  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "bucket", "verify_holdout_smci.json")


def contributions(D, v, log):
    """시작 가치 1 기준 금액 기여(티커별)."""
    days = {d["day"]: d["tgt"] for d in log}
    k0 = D.days.index(v.index[0])
    w, out = {}, {}
    vals = v.to_numpy()
    for t in range(len(v)):
        k = k0 + t
        if t > 0:
            new = {}
            for key, wt in w.items():
                j = D.j[key]
                a, b = D.O[k - 1, j], D.O[k, j]
                r = b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
                out[D.tk[key]] = out.get(D.tk[key], 0.0) + vals[t - 1] * wt * r
                new[key] = wt * (1 + r)
            tot = sum(new.values()) + 1 - sum(w.values())
            w = {key: x / tot for key, x in new.items()}
        if D.days[k] in days:
            w = dict(days[D.days[k]])
    assert abs(sum(out.values()) - (vals[-1] - 1)) < 1e-8
    return out


def label_fair_desk(semis):
    """데스크 규칙인데 '미분류'를 종목마다 각자 섹터로 본다(반도체 예산에서도 빠짐). 라벨 공백의 불이익을 없앤 대조."""
    base = V.desk_rule(semis=semis)

    def f(D, k, pdf, cur, exclude):
        saved = D.sector
        D.sector = {x: (UNKNOWN + ":" + x if s == UNKNOWN else s) for x, s in saved.items()}
        try:
            return base(D, k, pdf, cur, exclude)
        finally:
            D.sector = saved
    return f


def main():
    out = {}
    D0, D1 = V.Data(False), V.Data(True)
    names = ["desk50", "c774_60", "sub-N5-eq", "sub-N5-fund", "sector-N5-eq", "sector-N5-fund", "sector-N7-fund"]
    rules = dict(V.specs())

    print("1. 복원 자료에서 SMCI 기여(시작 가치 1 기준)와 보유 주 수")
    S = {}
    for mode in ("strict", "same", "lag1"):
        for n in names:
            v, log = V.simulate(D1, rules[n], mode)
            c = contributions(D1, v, log)
            weeks = sum("SMCI US EQUITY" in d["tgt"] for d in log)
            top = sorted(c.items(), key=lambda x: -x[1])[:4]
            S[f"{mode}:{n}"] = {"full": float(v.iloc[-1] - 1), "smci": c.get("SMCI", 0.0), "smci_weeks": weeks,
                                "top4": [(t, round(x, 3)) for t, x in top]}
            print(f"  {mode:6s} {n:15s} 전체 {v.iloc[-1] - 1:+.3f} SMCI {c.get('SMCI', 0.0):+.3f} ({weeks}주) 상위 {S[f'{mode}:{n}']['top4']}")
    out["smci"] = S

    print("\n2. 라벨 공정 데스크(미분류를 반도체 예산·60% 묶음에서 뺌) vs 2026 앞 선택(sub-N5-eq)")
    L = {}
    for dtag, D in (("base", D0), ("restored", D1)):
        for mode in ("strict", "same", "lag1"):
            row = {}
            for n, r in (("desk50_fair", label_fair_desk(0.50)), ("c774_fair", label_fair_desk(0.60)),
                         ("desk_nocaps", V.desk_rule(caps=False)), ("sub-N5-eq", rules["sub-N5-eq"]),
                         ("desk50", rules["desk50"])):
                v, log = V.simulate(D, r, mode)
                s = V.stats(D, v, log)
                row[n] = {"full": s["full"], "mdd": s["mdd"], "years": s["years"]}
            L[f"{dtag}|{mode}"] = row
            print(f"  {dtag:8s} {mode:6s} " + " | ".join(f"{n} {x['full']:+.3f} ({x['mdd']:+.3f})" for n, x in row.items()))
    out["label_fair"] = L

    print("\n3. desk_system §5 대조: #774, 실험실 타이밍(시가+0/+1), 배당 반영 시가, 원화, 끝 2025-12-31 (§5: +244.7%/+205.1%)")
    C = {}
    for dtag, D in (("base", D0), ("restored", D1)):
        for mode, st in (("lag0", "2023-05-22"), ("lag1", "2023-05-23")):
            for n in ("c774_60", "desk50"):
                v, log = V.simulate(D, rules[n], mode, start=st, end="2025-12-31", div=True)
                k0, k1 = D.days.index(v.index[0]), D.days.index(v.index[-1])
                krw = float(v.iloc[-1] * D.fx[k1] / D.fx[k0] - 1)
                C[f"{dtag}|{mode}|{n}"] = {"krw": krw, "x_etf": krw / 2.2817}
            print(f"  {dtag:8s} {mode}: #774 {C[f'{dtag}|{mode}|c774_60']['krw']:+.3f} ({C[f'{dtag}|{mode}|c774_60']['x_etf']:.2f}배), "
                  f"데스크50 {C[f'{dtag}|{mode}|desk50']['krw']:+.3f} ({C[f'{dtag}|{mode}|desk50']['x_etf']:.2f}배)")
    out["desk_s5"] = C

    print("\n4. 16개 전부: 복원 자료 3시점에서 데스크50보다 수익이 높은 횟수(0~3)와 낙폭도 얕은 횟수")
    W = {}
    M = {}
    for mode in ("strict", "same", "lag1"):
        for n, r in V.specs():
            v, log = V.simulate(D1, r, mode)
            M[(mode, n)] = V.stats(D1, v, log)
    for n, _ in V.specs()[2:]:
        wins = sum(M[(m, n)]["full"] > M[(m, "desk50")]["full"] for m in ("strict", "same", "lag1"))
        both = sum(M[(m, n)]["full"] > M[(m, "desk50")]["full"] and M[(m, n)]["mdd"] > M[(m, "desk50")]["mdd"]
                   for m in ("strict", "same", "lag1"))
        W[n] = {"wins": wins, "wins_and_mdd": both,
                "full": {m: M[(m, n)]["full"] for m in ("strict", "same", "lag1")},
                "mdd": {m: M[(m, n)]["mdd"] for m in ("strict", "same", "lag1")}}
        print(f"  {n:15s} 수익 승 {wins}/3, 둘다 {both}/3 | " +
              " ".join(f"{m} {M[(m, n)]['full']:+.3f}({M[(m, n)]['mdd']:+.3f})" for m in ("strict", "same", "lag1")))
    print("  데스크50          " + " ".join(f"{m} {M[(m, 'desk50')]['full']:+.3f}({M[(m, 'desk50')]['mdd']:+.3f})"
                                         for m in ("strict", "same", "lag1")))
    out["restored_wins"] = W
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("저장", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
