"""데스크 #774 위에 추적 손절(trailing stop)을 얹는 'stops' 계열(2026-10-10 연구, desk/ 코드는 고치지 않는다).

격자: trail ∈ {0.10, 0.15, 0.20} × trail_refill ∈ {False, True} = 6개 + 기본(#774).
  trail x       : 산 뒤 최고 종가 대비 k-1일 종가가 (1-x) 이하이면 k일 시가에 판다(종가 기준이라 손절 주문보다 보수적).
  trail_refill  : False = 판 자리는 다음 주 재조정까지 현금, True = 그날 순위에서 다음 종목으로 바로 채움.
고르기: 앞 1/2~5/29 수익만으로 최선 변형을 고른 뒤 뒤 6/1~10/9·9/1~10/9·월별·낙폭을 보고한다.
손절 판정: 재조정일이 아닌 날 보유에서 빠진 종목 = 손절(재조정 없는 날 비중이 바뀌는 길은 손절뿐).

실행: PYTHONIOENCODING=utf-8 py deskx_stops.py   (이 폴더에서)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskx import Engine, Params, MONTHS
from desk.strategies import iso_week

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/deskx/stops.json"))
TRAILS = (0.10, 0.15, 0.20)
REFILLS = (False, True)
P_FRONT, P_BACK, P_SEP, P_ALL = "앞 1/2~5/29", "뒤 6/1~10/9", "9/1~10/9", "전체"


def stop_events(E, rows):
    """재조정일이 아닌 날 빠진 종목 = 손절. 판 시가에서 다음 재조정 시가까지 그 종목 수익(음수면 손절이 도움)도 잰다."""
    days = [r["day"] for r in rows]
    reb_idx = [i for i in range(1, len(rows)) if iso_week(days[i]) != iso_week(days[i - 1])]
    out = []
    for i in range(1, len(rows)):
        if iso_week(days[i]) != iso_week(days[i - 1]):
            continue
        for key in set(rows[i - 1]["w"]) - set(rows[i]["w"]):
            nxt = next((j for j in reb_idx if j > i), None)
            o = E.col("opn", key)
            k = E.days.index(days[i])
            after = None
            if nxt is not None and o is not None:
                kn = E.days.index(days[nxt])
                if np.isfinite(o[k]) and np.isfinite(o[kn]) and o[k] > 0:
                    after = float(o[kn] / o[k] - 1)
            out.append({"day": days[i], "ticker": E.ticker[key], "weight_before": round(float(rows[i - 1]["w"][key]), 4),
                        "next_reb": days[nxt] if nxt is not None else None, "ret_to_next_reb": after,
                        "rebought_next_reb": bool(nxt is not None and key in rows[nxt]["w"])})
    return sorted(out, key=lambda e: (e["day"], e["ticker"]))


def main():
    E = Engine()
    grid = [("기본 #774", Params())] + [
        (f"trail{int(t * 100)}_{'refill' if r else 'cash'}", Params(trail=t, trail_refill=r)) for t in TRAILS for r in REFILLS]
    res = []
    for name, p in grid:
        v, rows = E.run(p)
        ev = E.evaluate(name, v, rows)
        ev["params"] = {"trail": p.trail, "trail_refill": p.trail_refill}
        ev["stops"] = stop_events(E, rows) if p.trail is not None else []
        ev["stop_count"] = len(ev["stops"])
        ev["stop_count_jul"] = sum(e["day"].startswith("2026-07") for e in ev["stops"])
        res.append(ev)

    base, var = res[0], res[1:]
    best = max(var, key=lambda e: e[P_FRONT]["수익"])
    both = [e["name"] for e in var if e[P_FRONT]["수익"] > base[P_FRONT]["수익"] and e[P_BACK]["수익"] > base[P_BACK]["수익"]]
    mdd_ok = [e["name"] for e in var if e[P_ALL]["낙폭"] > base[P_ALL]["낙폭"] and e[P_ALL]["수익"] >= 0.9 * base[P_ALL]["수익"]]

    hdr = f"{'변형':<20}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'전체MDD':>8}{'뒤MDD':>8}" + \
          "".join(f"{m[5:]+'월':>7}" for m in MONTHS[5:]) + f"{'손절':>5}{'7월':>4}{'반도체':>7}"
    print(hdr)
    for e in res:
        print(f"{e['name']:<20}{e[P_FRONT]['수익']:>8.3f}{e[P_BACK]['수익']:>8.3f}{e[P_SEP]['수익']:>8.3f}"
              f"{e[P_ALL]['수익']:>8.3f}{e[P_ALL]['낙폭']:>8.3f}{e[P_BACK]['낙폭']:>8.3f}"
              + "".join(f"{e['월별'][m]:>7.3f}" for m in MONTHS[5:])
              + f"{e['stop_count']:>5}{e['stop_count_jul']:>4}{e['반도체평균']:>7.3f}")
    print(f"\n격자 6개(trail 3 × refill 2). 앞 기준 최선: {best['name']}")
    print("앞·뒤 모두 기본 초과:", both or "없음")
    print("전체 MDD 개선 + 전체 수익 기본의 90% 이상:", mdd_ok or "없음")
    for e in var:
        names = {}
        for s in e["stops"]:
            names[s["ticker"]] = names.get(s["ticker"], 0) + 1
        print(f"  {e['name']}: 손절 {e['stop_count']}회 —", ", ".join(f"{t}×{n}" if n > 1 else t for t, n in sorted(names.items())))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"family": "stops", "grid": "trail {0.10,0.15,0.20} x trail_refill {False,True} = 6",
                   "selection": "best by 앞 1/2~5/29 수익", "best_by_front": best["name"],
                   "beats_base_both_halves": both, "lower_mdd_small_cost": mdd_ok,
                   "lower_mdd_rule": "전체 낙폭이 기본보다 얕고 전체 수익 >= 기본 전체 수익 x 0.9",
                   "results": res}, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
