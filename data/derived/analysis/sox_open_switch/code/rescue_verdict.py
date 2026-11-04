"""BRIEF7 놓친 후보 되살리기 — 판정(구현 A·B 대조 + Holm m = 6 + 다섯 조건).

읽는 것: results/rescue_daily.json(K1·K2), rescue_k3.json(K3), rescue_k45.json(K4·K5), rescue_k6.json(K6)
         와 각 눈가림 구현 B(rescue_*_blind.json). 새로 재는 값은 없다. 판정 숫자는 사전 등록 구현 A
         (원래 연구 코드를 그대로 불러 쓴 쪽)에서 가져오고, B와 평균·건수가 맞는지 확인한다.
판정 칸(BRIEF7): K1 1995~2010 합성 D 시가→D+1 시가, K2 주 1995~2010 합성 D 시가→D 종가,
         K3 2016~2022 SOXS 09:31→09:59(날 평균), K4·K5 2016~2022 하루 평균(사건 없는 날 0), K6 2016~2022 하루 평균.
다섯 조건: ① 평균 > 0 ② Holm(m = 6) 보정 p ≤ 0.05 ③ 상위 2건(2일) 빼도 평균 > 0 ④ 비용 2배 > 0
         ⑤ 사건이 있는 해의 절반 이상이 양(+). 다섯 모두 통과해야 '되살린 후보'.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
R = BASE / "results"
OUT = R / "rescue_verdict.json"
ALPHA = 0.05
MEAN_TOL = 0.02  # %p, A·B 평균 차 허용
# A·B p가 소수 셋째 자리에서 다른 까닭(판정자 확인, 2026-11-04). 평균·건수는 여섯 후보 모두 같아 코드를 고치지 않았다.
AB_P_NOTES = {
    "K1": "A 0.9625 / B 0.9643. 1996 풀이 A 254일, B 253일(B는 D+1 시가가 시가 규칙으로 뺀 1997년인 1996-12-31을 풀에서도 뺌). "
          "풀 길이가 달라 난수 흐름이 갈림. 사건 36건은 같고 값 차 5e-5(B 반올림).",
    "K2": "A·B 0.2681로 같음(풀 같음).",
    "K3": "A 0.0626 / B 0.0689. 2016년 7일(02-29·03-03·03-16·04-07·04-12·05-11·07-19)은 그날 09:31까지 SOXS 봉이 없어 "
          "엔진에서 체결 불가(NaN). B는 그날 진입가를 전날 종가로 채워 무작위 풀에 넣음. B 코드로 그 7일만 풀에서 빼면 "
          "p 0.0626·귀무 평균 −0.16399로 A와 정확히 같음 → A가 맞음. 사건 125건은 같음.",
    "K4": "A 0.7083 / B 0.7084(대조 뽑기 구현 차, 같은 정의).",
    "K5": "A·B 0.9994.",
    "K6": "A 0.1511(10,000회) / B 0.1515(2,000회). 하루 수익 1,762일 최대 차 1e-14%.",
}


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def holm(pvals: dict) -> dict:
    """Holm 단계 하향 보정 p(단조, 1로 자름)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def years_pos(by_year: dict, key: str) -> tuple:
    vals = [v[key] for v in by_year.values() if v.get(key) is not None and (v.get("n", 1) or v.get("trades", 1))]
    vals = [x for x in vals if x is not None]
    return sum(1 for x in vals if x > 0), len(vals)


def main():
    daily, daily_b = load("rescue_daily.json"), load("rescue_daily_blind.json")
    k3, k3_b = load("rescue_k3.json"), load("rescue_k3_blind.json")
    k45, k45_b = load("rescue_k45.json"), load("rescue_k45_blind.json")
    k6, k6_b = load("rescue_k6.json"), load("rescue_k6_blind.json")

    C = {}
    # K1·K2 (사건당 %, 순수익 = 합성 − 0.2%p, 비용 2배 = −0.4%p)
    for K in ("K1", "K2"):
        a = daily["candidates"][K]
        b = daily_b["main"][K]["summary"]
        by = {y: v for y, v in a["by_year"].items() if v.get("n")}
        C[K] = {"window": "1995~2010 일봉 합성 SOXS(1995~2001 ^SOX 1996만 시가 규칙 통과, 2002~2010 SOXX)",
                "unit": "사건당 %",
                "n": a["n"], "mean": a["mean"], "drop_top2": a["drop_top2"], "cost2x": a["cost2x"],
                "p_raw": a["p_one_sided"], "p_plus1": a["p_plus1"],
                "yp": years_pos(by, "mean"),
                "B": {"n": b["n"], "mean": b["mean"], "p": b["p_one_sided"], "drop_top2": b["drop_top2"],
                      "cost2x": b["cost2x"], "years_positive": b["years_positive"]},
                "p_method": "같은 해 아무 거래일(합성값 있는 날)에서 같은 수 비복원 10,000회, seed 20261104, 평균 ≥ 관측 비율"}
    # K3 (날 평균 = 사건당 %)
    a, b = k3["K3"], k3_b["K3"]["all"]
    by = {y: v for y, v in a["by_year"].items() if v.get("n")}
    C["K3"] = {"window": "2016~2022 Alpaca 1분봉, SOXS 09:31 봉 종가→09:59 봉 종가, cost_table_2016.json",
               "unit": "사건당 %",
               "n": a["n"], "mean": a["mean"], "drop_top2": a["drop_top2"]["mean"], "cost2x": a["cost2x_mean"],
               "p_raw": a["p_random"], "p_plus1": a["p_random_plus1"], "yp": years_pos(by, "mean"),
               "B": {"n": b["n"], "mean": b["mean"], "p": b["p_one_sided"], "drop_top2": b["drop_top2"],
                     "cost2x": b["cost2x"], "years_positive": b["years_positive"]},
               "p_method": "같은 해 실행 가능한 거래일에서 같은 수 비복원 10,000회, seed 20261104"}
    # K4·K5 (하루 평균, 사건 없는 날 0, 1,762일)
    for K, pkey in (("K5", "boot_day"), ("K4", "ctrl_same_year_subset")):
        a, b = k45[K], k45_b[K]
        if K == "K5":
            p_raw, p1 = a["boot_day"]["p_one_sided"], None
        else:
            p_raw, p1 = a["ctrl_same_year_subset"]["p_one_sided"], a["ctrl_same_year_subset"]["p_plus1"]
        bn = b.get("n") or b.get("trades")
        C[K] = {"window": "2016~2022 Alpaca 1분봉, 사용자 방향 x = 3% 사슬(ev4_build), cost_table_2016.json",
                "unit": "하루 %(사건 없는 날 0)",
                "n": a["trades"], "days": a["days"], "mean": a["daily_mean"],
                "drop_top2": a["drop_top2_days"]["daily_mean"], "cost2x": a["cost2x"]["daily_mean"],
                "p_raw": p_raw, "p_plus1": p1, "yp": years_pos(a["by_year"], "daily_mean"),
                "B": {"n": bn, "mean": b.get("mean", b.get("daily_mean")), "p": b.get("p_one_sided"),
                      "drop_top2": b.get("drop_top2"), "cost2x": b.get("cost2x"),
                      "years_positive": b.get("years_positive")},
                "p_method": ("날 묶음 부트스트랩 10,000회, 평균 ≤ 0 비율" if K == "K5" else
                             "같은 해 조건 없는 사용자 방향 x3 사건에서 같은 수 비복원 10,000회, 하루 평균 ≥ 관측 비율")}
    # K6 (하루 평균, 1,762일)
    a = k6["base"]
    ball = k6_b["all"]
    C["K6"] = {"window": "2016~2022 Alpaca 1분봉, F1-A(N1·confirm·2R·손절 없음), 마감 actual, cost_table_2016.json",
               "unit": "하루 %(매매 없는 날 0)",
               "n": a["days"], "trades": a["trades"], "mean": a["mean_pct"], "drop_top2": a["drop_top2_mean_pct"],
               "cost2x": k6["cost2x"]["mean_pct"], "p_raw": a["p_one_sided"], "p_plus1": None,
               "yp": years_pos(k6["by_year"], "mean_pct"),
               "B": {"n": ball.get("n", ball.get("days")), "mean": ball.get("mean"), "p": ball.get("p_one_sided"),
                     "drop_top2": ball.get("drop_top2"), "cost2x": ball.get("cost2x"),
                     "years_positive": ball.get("years_positive"), "boot_draws": 2000},
               "p_method": "날 묶음 부트스트랩 10,000회, seed 20261104, 평균 ≤ 0 비율"}

    order = ["K1", "K2", "K3", "K4", "K5", "K6"]
    ph = holm({k: C[k]["p_raw"] for k in order})
    ph_b = holm({k: C[k]["B"]["p"] for k in order})
    ph_max = holm({k: max(C[k]["p_raw"], C[k]["B"]["p"]) for k in order})
    ph_p1 = holm({k: (C[k]["p_plus1"] if C[k]["p_plus1"] is not None else C[k]["p_raw"]) for k in order})

    table, revived = [], []
    for k in order:
        c = C[k]
        pos, ny = c["yp"]
        ab_mean_diff = abs(c["mean"] - c["B"]["mean"])
        ab_match = (c["n"] == c["B"]["n"]) and ab_mean_diff <= MEAN_TOL
        crit = {"1_mean_gt0": c["mean"] > 0,
                "2_holm_p_le_0.05": ph[k] <= ALPHA,
                "3_drop_top2_gt0": c["drop_top2"] > 0,
                "4_cost2x_gt0": c["cost2x"] > 0,
                "5_half_years_positive": 2 * pos >= ny}
        npass = sum(crit.values())
        ok = all(crit.values())
        if ok:
            revived.append(k)
        row = {"id": k, "window": c["window"], "unit": c["unit"], "n": c["n"], "mean": c["mean"],
               "drop_top2": c["drop_top2"], "cost2x": c["cost2x"], "p_raw": c["p_raw"], "p_holm": ph[k],
               "years_positive": f"{pos}/{ny}", "criteria": crit, "n_pass": npass,
               "verdict": "되살림" if ok else f"되살리지 못함({npass}/5 통과)",
               "p_method": c["p_method"],
               "ab": {"match": ab_match, "n_A": c["n"], "n_B": c["B"]["n"], "mean_A": c["mean"],
                      "mean_B": c["B"]["mean"], "mean_abs_diff": ab_mean_diff, "p_A": c["p_raw"], "p_B": c["B"]["p"]},
               "holm_sens": {"B_p": ph_b[k], "max_AB_p": ph_max[k], "p_plus1": ph_p1[k]}}
        for extra in ("days", "trades"):
            if extra in c:
                row[extra] = c[extra]
        table.append(row)

    # 감도(판정 칸 밖): 채운 봉 매매 뺀 값
    sens = {
        "K3_채운봉_뺌": {k: k3["K3_sensitivity_no_filled"].get(k) for k in ("n", "mean", "cost2x_mean", "p_random")},
        "K3_채운봉_뺌_drop_top2": k3["K3_sensitivity_no_filled"]["drop_top2"]["mean"],
        **{f"{K}_채운봉_뺌": {"trades": k45[K]["excl_filled"]["trades"],
                             "daily_mean": k45[K]["excl_filled"]["daily_mean"],
                             "drop_top2": k45[K]["excl_filled"]["drop_top2_days"]["daily_mean"],
                             "p": (k45[K]["excl_filled"]["ctrl_same_year_subset"]["p_one_sided"] if K == "K4"
                                   else k45[K]["excl_filled"]["boot_day"]["p_one_sided"])} for K in ("K4", "K5")},
        **{name: {k: k6_b[key].get(k) for k in ("n", "mean", "p_one_sided", "drop_top2", "cost2x", "years_positive")}
           for name, key in (("K6_채운봉_날_뺌", "excl_filled_drop_days"), ("K6_채운봉_매매_0", "excl_filled_as_zero"))},
    }
    aux = k3["K2_aux"]
    out = {
        "brief": "BRIEF7.md(2026-11-04 사전 등록, 결과 보기 전 커밋 b241633)",
        "family": "Holm m = 6(K1~K6 판정 칸 원 p), 보정 p ≤ 0.05",
        "p_source": "구현 A(원래 연구 코드 lib·stage5·redfirst·fam_F1·ev4_build를 불러 쓴 쪽). B는 평균·건수 대조와 Holm 감도용",
        "revived": revived,
        "table": table,
        "holm_raw_order": sorted(((k, C[k]["p_raw"]) for k in order), key=lambda kv: kv[1]),
        "sensitivity_filled_bars": sens,
        "ab_p_notes": AB_P_NOTES,
        "K2_aux_1m_out_of_family": {
            "note": "BRIEF7 보조 칸(판정 밖). 2016~2022 SOXS 09:31→10:59, BRIEF6 일봉 근사로 일부 본 구간",
            "n": aux["n"], "mean": aux["mean"], "p_raw": aux["p_random"], "drop_top2": aux["drop_top2"]["mean"],
            "cost2x": aux["cost2x_mean"],
            "holm_if_added_as_7th_smallest": min(1.0, 7 * aux["p_random"])},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in table:
        print(r["id"], r["n"], round(r["mean"], 4), "p", r["p_raw"], "holm", round(r["p_holm"], 4),
              "drop2", round(r["drop_top2"], 4), "c2", round(r["cost2x"], 4), r["years_positive"],
              r["verdict"], "AB", r["ab"]["match"], round(r["ab"]["mean_abs_diff"], 6), r["ab"]["p_B"])
    print("revived", revived)
    print("sens", json.dumps(sens, ensure_ascii=False)[:1500])
    print("aux", out["K2_aux_1m_out_of_family"])


if __name__ == "__main__":
    main()
