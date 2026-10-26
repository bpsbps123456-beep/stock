"""데스크 #774 집중도 완화(caps) 6개 변형 비교(2026-10-10, 연구용 스크래치).

변형: semis_cap 0.40 / 0.50, stock_cap 0.15 / 0.20, sector_cap 0.40, n=10(buffer 5). 나머지는 #774 그대로.
선택: 앞 1/2~5/29 수익만으로 최선 변형을 고르고, 그 뒤 구간·월별·낙폭을 보고한다.
종목 기여: 각 시가에 든 비중 × 그 종목 시가→다음 시가 수익 × 그날 계좌 가치(시작 1) 를 더한 값.
          종목 기여 합 = 전체 최종 가치 - 1. 상위 3 비중 = 상위 3 기여 합 / (최종 가치 - 1).
실행: py deskx_caps.py   (code 폴더에서, PYTHONIOENCODING=utf-8)
"""
import json, os, sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskx import Engine, Params, PERIODS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/deskx/caps.json")
FRONT, BACK, SEP, FULL = (p[0] for p in PERIODS)

VARIANTS = [
    ("semis_cap 0.40", Params(semis_cap=0.40)),
    ("semis_cap 0.50", Params(semis_cap=0.50)),
    ("stock_cap 0.15", Params(stock_cap=0.15)),
    ("stock_cap 0.20", Params(stock_cap=0.20)),
    ("sector_cap 0.40", Params(sector_cap=0.40)),
    ("n=10 (buffer 5)", Params(n=10, buffer=5)),
]


def contributions(E, v, rows):
    """종목별 기여(시작 가치 1 기준 금액). rows[i]['w']는 i일 시가부터 i+1일 시가까지 든 비중."""
    k0 = E.days.index(rows[0]["day"])
    vals = v.to_numpy()
    contrib: dict[str, float] = {}
    for i in range(len(rows) - 1):
        k = k0 + i + 1
        for key, wt in rows[i]["w"].items():
            o = E.col("opn", key)
            ret = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                and o[k - 1] > 0 else 0.0
            tk = E.ticker[key]
            contrib[tk] = contrib.get(tk, 0.0) + vals[i] * wt * ret
    gain = float(vals[-1] - 1)
    check = sum(contrib.values())
    assert abs(check - gain) < 1e-9 * max(1.0, abs(gain)), (check, gain)
    top = sorted(contrib.items(), key=lambda x: -x[1])
    top3 = sum(c for _, c in top[:3])
    return {"상위3": [t for t, _ in top[:3]], "상위3비중": top3 / gain if gain else float("nan"),
            "상위5기여": {t: round(c, 4) for t, c in top[:5]},
            "하위3기여": {t: round(c, 4) for t, c in top[-3:]},
            "전체기여": {t: round(c, 4) for t, c in top}}


def one(E, name, p):
    v, rows = E.run(p)
    res = E.evaluate(name, v, rows)
    res["params"] = asdict(p)
    res["평균투자비중"] = float(np.mean([sum(r["w"].values()) for r in rows]))
    res.update(contributions(E, v, rows))
    return res


def main():
    E = Engine()
    base = one(E, "기본 #774", Params())
    out = [one(E, name, p) for name, p in VARIANTS]

    best = max(out, key=lambda r: r[FRONT]["수익"])
    both = [r["name"] for r in out if r[FRONT]["수익"] > base[FRONT]["수익"] and r[BACK]["수익"] > base[BACK]["수익"]]
    mdd_ok = [r["name"] for r in out if r[FULL]["낙폭"] > base[FULL]["낙폭"] + 1e-3   # 0.1%p 이상 개선만
              and r[FULL]["수익"] >= 0.9 * base[FULL]["수익"]]

    months = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
    head = f"{'변형':<16}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'전체MDD':>9}{'뒤MDD':>8}" \
           + "".join(f"{m[-2:] + '월':>7}" for m in months) + f"{'반도체':>7}{'투자':>6}{'상위3':>7}  상위3종목"
    print(head)
    for r in [base] + out:
        print(f"{r['name']:<16}{r[FRONT]['수익']:>8.3f}{r[BACK]['수익']:>8.3f}{r[SEP]['수익']:>8.3f}"
              f"{r[FULL]['수익']:>8.3f}{r[FULL]['낙폭']:>9.3f}{r[BACK]['낙폭']:>8.3f}"
              + "".join(f"{r['월별'][m]:>7.3f}" for m in months)
              + f"{r['반도체평균']:>7.3f}{r['평균투자비중']:>6.2f}{r['상위3비중']:>7.2f}  {','.join(r['상위3'])}")
    print(f"\n앞 기준 최선: {best['name']}  | 앞·뒤 모두 기본 초과: {both or '없음'}"
          f"  | 전체 MDD 개선 & 전체 수익 ≥ 기본의 90%: {mdd_ok or '없음'}")
    print("시험 범위: 단일 파라미터 변경 6개(조합 없음), 기본 #774 대비.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"family": "caps", "grid": len(VARIANTS), "base": base, "variants": out,
                   "best_by_front": best["name"], "beats_base_both_halves": both, "lower_mdd_small_cost": mdd_ok},
                  f, ensure_ascii=False, indent=1)
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
