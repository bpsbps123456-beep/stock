"""데스크 #774 위 '시점·거르기' 묶음(2026-10-10): 매일 갱신, 60일 모멘텀, 신고가 근처만 후보(80%·90%).

미리 정한 4개 변형 + 기본 1개 = 5개. 고르기는 앞 1/2~5/29 수익만, 뒤 6/1~10/9·9/1~10/9·월별·낙폭은 확인용.
  daily   rebalance='daily'      매 거래일 시가에 같은 순위·버퍼(상위 12 안이면 유지) 규칙으로 7자리와 비중을 다시 맞춘다
  lb60    lookback=60            모멘텀 창 120 → 60거래일
  nh80    min_near_high=0.80     재조정 때 k-1일 종가 / k-1일까지 252거래일 최고 종가 < 0.80이면 후보에서 뺀다
  nh90    min_near_high=0.90     같은 기준 0.90
시점: k일 시가 결정은 k일보다 앞선 날짜의 PDF와 k-1일 종가까지만 쓴다(deskx.Engine과 같음).
회전율: rows(재조정 뒤 비중)에서 신규 편입=매수, 편출=매도 횟수를 센다. 편도 회전율은 전날 비중을 시가 수익으로 굴린 값과
당일 비중의 차 절반을 합한 것(비중만 바꾸는 거래 포함, 수수료 0 가정의 민감도 참고용).
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/deskx_timing.py
"""
import os, sys, json
from dataclasses import asdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, START          # noqa: E402

OUT = os.path.join(HERE, "..", "results", "deskx", "timing.json")
FRONT, BACK, SEPOCT, FULL = "앞 1/2~5/29", "뒤 6/1~10/9", "9/1~10/9", "전체"
SHOW_MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]

VARIANTS = [
    ("base(#774)", Params()),
    ("daily", Params(rebalance="daily")),
    ("lb60", Params(lookback=60)),
    ("nh80", Params(min_near_high=0.80)),
    ("nh90", Params(min_near_high=0.90)),
]


def turnover(E, rows):
    """신규 편입·편출 횟수와 편도 회전율 합(첫날 매수 제외)."""
    buys = sells = 0
    one_way = 0.0
    k0 = E.days.index(rows[0]["day"])
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1]["w"], rows[i]["w"]
        buys += len(set(cur) - set(prev))
        sells += len(set(prev) - set(cur))
        k = k0 + i
        rets = {}
        for key in prev:
            o = E.col("opn", key)
            ok = o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0
            rets[key] = o[k] / o[k - 1] - 1 if ok else 0.0
        tot = sum(wt * (1 + rets[x]) for x, wt in prev.items()) + (1 - sum(prev.values()))
        drift = {x: wt * (1 + rets[x]) / tot for x, wt in prev.items()}
        one_way += 0.5 * sum(abs(cur.get(x, 0.0) - drift.get(x, 0.0)) for x in set(cur) | set(drift))
    return buys, sells, one_way


def held_names(E, rows):
    """보유 일수 순 종목 목록."""
    cnt = {}
    for row in rows:
        for key in row["w"]:
            cnt[E.ticker[key]] = cnt.get(E.ticker[key], 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: -kv[1]))


def main():
    E = Engine()
    hist = E.days.index([d for d in E.days if d >= START][0])
    results = []
    for name, p in VARIANTS:
        v, rows = E.run(p)
        res = E.evaluate(name, v, rows)
        b, s, ow = turnover(E, rows)
        res["params"] = {k: x for k, x in asdict(p).items() if x != getattr(Params(), k)}
        res["매수횟수"], res["매도횟수"], res["편도회전율합"] = b, s, round(ow, 3)
        res["보유일수"] = held_names(E, rows)
        results.append(res)

    base = results[0]
    cands = results[1:]
    best = max(cands, key=lambda r: r[FRONT]["수익"])
    both = [r["name"] for r in cands if r[FRONT]["수익"] > base[FRONT]["수익"] and r[BACK]["수익"] > base[BACK]["수익"]]
    mdd_ok = [r["name"] for r in cands if r[FULL]["낙폭"] > base[FULL]["낙폭"]
              and r[FULL]["수익"] >= 0.9 * base[FULL]["수익"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {"family": "timing/filter", "grid": "미리 정한 4개 변형(daily·lb60·nh80·nh90) + 기본 1개",
               "history_sessions_before_start": hist, "best_by_front": best["name"],
               "beats_base_both_halves": both, "lower_mdd_within_10pct_of_base_full": mdd_ok, "results": results}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    head = f"{'변형':<11}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'MDD':>8}" + "".join(f"{m[5:]:>7}" for m in SHOW_MONTHS) \
        + f"{'반도체':>7}{'종목':>5}{'매수':>5}{'매도':>5}{'회전':>7}"
    print(head)
    for r in results:
        line = f"{r['name']:<11}" + "".join(f"{r[k]['수익']:>8.3f}" for k in (FRONT, BACK, SEPOCT, FULL)) \
            + f"{r[FULL]['낙폭']:>8.3f}" + "".join(f"{r['월별'][m]:>7.3f}" for m in SHOW_MONTHS) \
            + f"{r['반도체평균']:>7.3f}{r['거친종목수']:>5}{r['매수횟수']:>5}{r['매도횟수']:>5}{r['편도회전율합']:>7.2f}"
        print(line)
    print("앞 구간 최고:", best["name"], "| 앞·뒤 모두 기본 초과:", both or "없음",
          "| 낙폭 개선+전체 90% 이상:", mdd_ok or "없음", "| 시작 전 이력 거래일:", hist)
    for r in results:
        print(r["name"], "마지막보유", r["마지막보유"], "| MU 보유일", r["보유일수"].get("MU", 0),
              "| PLTR 보유일", r["보유일수"].get("PLTR", 0))
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
