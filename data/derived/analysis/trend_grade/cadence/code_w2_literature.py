"""W2 문헌 조사 — 논문 표에서 옮긴 숫자로 신호 소멸 속도·회전율 효과를 계산한다.

입력은 모두 논문 원문 표를 손으로 옮긴 값이다(출처는 각 상수 옆 주석).
결과: results_w2_literature.json (results_w2_literature.md 의 계산 근거)
실행: py data/derived/analysis/trend_grade/cadence/code_w2_literature.py
"""
import json
from pathlib import Path

OUT = Path(__file__).with_name("results_w2_literature.json")

# Jegadeesh & Titman (1993) JF 48(1) Table VII: 6개월 형성 승자-패자(10분위) 포트폴리오의
# 형성 뒤 t개월째 월 수익률(1965-01~1989-12, NYSE·AMEX). 출처 PDF:
# https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf (JSTOR 사본, p.84)
JT_TABLE_VII_MONTHLY = [
    -0.0025, 0.0124, 0.0116, 0.0110, 0.0093, 0.0091, 0.0134, 0.0115, 0.0085, 0.0048, 0.0045, 0.0013,
    -0.0036, -0.0039, -0.0034, -0.0038, -0.0047, -0.0056, -0.0026, -0.0032, -0.0032, -0.0034, -0.0011, -0.0010,
    -0.0035, -0.0030, -0.0024, -0.0032, -0.0032, -0.0030, -0.0001, 0.0008, 0.0013, 0.0008, 0.0010, -0.0005,
]
# 같은 표의 누적 수익률(논문 값; 단순 합과 소수점 차이 확인용)
JT_TABLE_VII_CUM = {12: 0.0951, 24: 0.0556, 36: 0.0406}

# Jegadeesh & Titman (1993) Table I: 승자-패자 월평균 수익률. 행 J(형성 개월), 열 K(보유 개월 3,6,9,12).
# Panel A = 형성 직후 매수, Panel B = 형성 뒤 1주 건너뛰고 매수. 같은 PDF p.70.
JT_TABLE_I_A = {3: [0.0032, 0.0058, 0.0061, 0.0069], 6: [0.0084, 0.0095, 0.0102, 0.0086],
                9: [0.0109, 0.0121, 0.0105, 0.0082], 12: [0.0131, 0.0114, 0.0093, 0.0068]}
JT_TABLE_I_B = {3: [0.0073, 0.0078, 0.0074, 0.0077], 6: [0.0114, 0.0110, 0.0108, 0.0090],
                9: [0.0135, 0.0130, 0.0109, 0.0085], 12: [0.0149, 0.0121, 0.0096, 0.0069]}
K_LIST = [3, 6, 9, 12]

# Novy-Marx & Velikov (2016 RFS; NBER w20721, 2014-12) 미국 가치가중 10분위 모멘텀(직전 1개월 제외 12개월).
# 월 %: 총수익, 한쪽 월 회전율(%), 거래비용, 순수익. 표 3·7·8·10의 Momentum 행.
# PDF: https://www.nber.org/system/files/working_papers/w20721/w20721.pdf
NMV_MOMENTUM = {
    "기본(매월 전체 재정렬)": {"gross": 1.33, "turnover": 34.52, "tcost": 0.65, "net": 0.68, "table": "Table 3"},
    "분할 재정렬(매월 1/3만 교체)": {"gross": 1.25, "turnover": 16.66, "tcost": 0.34, "net": 0.91, "table": "Table 7"},
    "이력 띠(상위 10% 진입, 20% 밖이면 퇴출)": {"gross": 1.20, "turnover": 18.82, "tcost": 0.35, "net": 0.85, "table": "Table 8"},
    "세 기법 모두": {"gross": 1.27, "turnover": 15.48, "tcost": 0.26, "net": 1.01, "table": "Table 10"},
}


def jt_decay():
    m = JT_TABLE_VII_MONTHLY
    cum, run = [], 0.0
    for r in m:
        run += r
        cum.append(run)
    core = m[1:7]  # 2~7개월째(1개월째는 단기 반전으로 음수)
    core_avg = sum(core) / len(core)
    half_month = next(t for t in range(8, 37) if m[t - 1] < core_avg / 2)
    first_neg_after_peak = next(t for t in range(8, 37) if m[t - 1] < 0)
    peak_t = max(range(1, 37), key=lambda t: cum[t - 1])
    return {
        "month1_return": m[0],
        "months2_7_avg_monthly": round(core_avg, 5),
        "first_month_below_half_of_months2_7_avg": half_month,
        "first_negative_month_after_month7": first_neg_after_peak,
        "peak_cum_month": peak_t,
        "peak_cum_simple_sum": round(cum[peak_t - 1], 4),
        "cum_paper": JT_TABLE_VII_CUM,
        "share_of_peak_lost_by_month36_paper": round(1 - JT_TABLE_VII_CUM[36] / JT_TABLE_VII_CUM[12], 3),
        "year2_sum": round(sum(m[12:24]), 4),
    }


def jt_skip_week():
    rows = []
    for j in JT_TABLE_I_A:
        for i, k in enumerate(K_LIST):
            a, b = JT_TABLE_I_A[j][i], JT_TABLE_I_B[j][i]
            rows.append({"J": j, "K": k, "no_skip": a, "skip_1w": b, "diff": round(b - a, 4)})
    diffs = [r["diff"] for r in rows]
    by_k = {k: round(sum(JT_TABLE_I_A[j][i] for j in JT_TABLE_I_A) / 4, 5) for i, k in enumerate(K_LIST)}
    return {
        "grid": rows,
        "skip_week_avg_gain": round(sum(diffs) / len(diffs), 5),
        "skip_week_positive_cells": sum(d > 0 for d in diffs),
        "cells": len(diffs),
        "panelA_avg_by_K": by_k,
        "best_A": max(rows, key=lambda r: r["no_skip"]),
        "best_B": max(rows, key=lambda r: r["skip_1w"]),
    }


def nmv_effect():
    base = NMV_MOMENTUM["기본(매월 전체 재정렬)"]
    out = {}
    for name, v in NMV_MOMENTUM.items():
        out[name] = dict(v)
        out[name]["gross_change_vs_base"] = round(v["gross"] - base["gross"], 2)
        out[name]["net_change_vs_base"] = round(v["net"] - base["net"], 2)
        out[name]["turnover_ratio_vs_base"] = round(v["turnover"] / base["turnover"], 2)
    return out


def main():
    res = {
        "source_note": "모든 입력은 논문 표에서 옮긴 값. URL은 코드 주석과 results_w2_literature.md 참고",
        "jt1993_event_time_decay": jt_decay(),
        "jt1993_skip_week": jt_skip_week(),
        "nmv2016_momentum_cost_mitigation": nmv_effect(),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "jt1993_skip_week"}, ensure_ascii=False, indent=2))
    s = res["jt1993_skip_week"]
    print("skip-week avg gain", s["skip_week_avg_gain"], s["skip_week_positive_cells"], "/", s["cells"],
          "panelA by K", s["panelA_avg_by_K"], "best A", s["best_A"], "best B", s["best_B"])


if __name__ == "__main__":
    main()
