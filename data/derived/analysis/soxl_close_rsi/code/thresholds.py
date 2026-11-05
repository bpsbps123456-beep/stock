"""All integer RSI cutoffs1..30 on existing frozen1m/3m events; no new feed.

Run after study.py, quotes.py: py .../code/thresholds.py
Uses existing return/statistics implementations rather than another fill engine.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FRAMES = ("3m", "1m")
LEVELS = tuple(range(1, 31))
FIXED = {"3m": "18:00", "1m": "19:00"}
PERIODS = {"all": ("2026-01-01", "2026-11-03"),
           "H1": ("2026-01-01", "2026-06-30"), "H2": ("2026-07-01", "2026-11-03")}


def shared():
    spec = importlib.util.spec_from_file_location("threshold_shared_study", ROOT / "code/study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eligible(value, cutoff):
    return value <= cutoff


def load_events():
    t = pd.read_csv(OUT / "trades.csv")
    t = t[t.tf.isin(FRAMES) & t.delay.eq(1) & t.exit_hm.isin(shared().EXITS)].copy()
    q = pd.read_csv(OUT / "quotes_trades.csv")
    fields = ["day", "tf", "exit_hm", "quote_valid", "quote_reason", "ratio_flag", "quote_return_pct", "quote_return_cost2_pct"]
    q = q[q.tf.isin(FRAMES) & q.exit_hm.isin(shared().EXITS)][fields]
    merged = t.merge(q, on=["day", "tf", "exit_hm"], how="left", validate="one_to_one")
    assert merged.loc[merged.rsi.le(30), "quote_valid"].notna().all(), "lower cutoffs need existing quote audit"
    return merged


def analyze(events):
    study = shared()
    cells = []
    for tf in FRAMES:
        for threshold in LEVELS:
            for ex in study.EXITS:
                for period, (first, last) in PERIODS.items():
                    signal = events[events.tf.eq(tf) & events.exit_hm.eq(ex)
                                    & events.day.between(first, last) & eligible(events.rsi, threshold)]
                    good = signal[signal.quote_valid.eq(True)]
                    assert len(signal) == signal.day.nunique()
                    cells.append({"tf": tf, "threshold": threshold, "exit_hm": ex, "period": period,
                                  "n_signal": len(signal), "n_valid_quote": len(good),
                                  "signal_days": signal.day.tolist(), "valid_quote_days": good.day.tolist(),
                                  "model_gross": study.stats(signal.gross_pct),
                                  "model_net10bp": study.stats(signal.net_10bp_pct),
                                  "quote": study.stats(good.quote_return_pct),
                                  "quote_cost2": study.stats(good.quote_return_cost2_pct),
                                  "mae_worst_pct": float(signal.mae_pct.min()) if len(signal) else None,
                                  "excluded_quote_reasons": signal.loc[~signal.quote_valid.eq(True), "quote_reason"].value_counts().to_dict()})
    lookup = {(c["tf"], c["threshold"], c["exit_hm"], c["period"]): c for c in cells}
    wf = []
    for tf in FRAMES:
        for threshold in LEVELS:
            choices = [lookup[(tf, threshold, ex, "H1")] for ex in study.EXITS
                       if lookup[(tf, threshold, ex, "H1")]["n_signal"] > 0]
            if not choices:
                wf.append({"tf": tf, "threshold": threshold, "selected_exit": None, "H1": None, "H2": None,
                           "passes_frozen_gate": False, "reason": "no_H1_signal"})
                continue
            best = max(choices, key=lambda c: c["model_net10bp"]["mean_pct"])
            test = lookup[(tf, threshold, best["exit_hm"], "H2")]
            st = test["model_net10bp"]
            # Match the original gate; unconditional comparator is from all available days.
            baseline = events[events.tf.eq(tf) & events.exit_hm.eq(best["exit_hm"]) & events.day.between(*PERIODS["H2"])].net_10bp_pct.mean()
            passed = st["n"] >= 20 and st["mean_pct"] > 0 and st["mean_ci95_pct"][0] > 0 and st["mean_pct"] > baseline
            wf.append({"tf": tf, "threshold": threshold, "selected_exit": best["exit_hm"],
                       "H1": best, "H2": test, "passes_frozen_gate": bool(passed),
                       "reason": "no_H2_signal" if not test["n_signal"] else "H2_available"})
    return cells, wf


def fmt(x):
    return "—" if x is None else f"{x:+.2f}%"


def report(result):
    cells = result["cells"]
    lookup = {(c["tf"], c["threshold"], c["exit_hm"], c["period"]): c for c in cells}
    lines = ["# SOXL 마감 전 RSI 문턱1~30 전체 비교", "",
             "2026-11-05 · Path-Q · [원 연구](report.md) · [계산 규약](BRIEF.md)", "",
             "> **매수 시점 정정:** 아래는15:30 딱 한 번 검사한 이전 연구다. 사용자 확정 조건은15:30~16:00 구간의 첫 신호 매수이며, 아래 수익률과 희소성 결론을 그 조건에 적용하지 않는다. [정정된 매수 흐름·실제 차트](strategy_visual.html).", "",
             "이번 요청은 3분봉 RSI14를 중심으로 보고, 기존 1분봉 RSI14도 함께 비교했다. 정규장 마감 30분 전인 15:30 ET에 확정된 RSI≤문턱을 판단하고 15:31에 매수한다. 정규장만 연결한 Wilder RSI이며 2026-01-02~11-03의 211거래일을 사용했다.",
             "수수료0. 아래 표는 실제 SIP 매수ask/매도bid 체결 근사이며 추가 슬리피지·시장충격·환율·세금은 제외했다.0건은0%로 채우지 않았다.", "",
             "## 고정 청산으로 문턱만 비교", "",
             "3분18:00·1분19:00은 앞선≤30 실험에서 상반기로 고른 시각이다. 이 표에서는 문턱을 낮출 때 청산 시각을 바꾸지 않는다. 각 문턱은 서로 겹치는 거래의 부분집합이다.", "",
             "| RSI≤ |3분 신호/유효 |3분 평균18:00 |승률 |최악 |하반기 신호 |1분 평균19:00(유효) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for level in (30, 25, 20, 15, 10, 5):
        a, b = [lookup[(tf, level, FIXED[tf], "all")] for tf in FRAMES]
        h2 = lookup[("3m", level, FIXED["3m"], "H2")]
        s = a["quote"]
        win = "—" if not s["n"] else f'{s["win_rate_pct"]:.1f}%'
        lines.append(f'| {level} | {a["n_signal"]}/{a["n_valid_quote"]} | {fmt(s.get("mean_pct"))} | {win} | {fmt(s.get("worst_pct"))} | {h2["n_signal"]} | {fmt(b["quote"].get("mean_pct"))} ({b["n_valid_quote"]}) |')
    lines += ["", "### 해석", "",
              "3분봉의 고정 18:00 청산에서는 ≤30→≤25→≤20으로 낮출 때 평균이 +1.01%→+0.46%→+0.33%로 감소했다. ≤20의 승률 100%는 단 2건이며, ≤25 이하에는 하반기 신호가 없어 전방 확인이 안 된다. ≤15·≤10·≤5는 거래가 없었다.",
              "낮은 문턱에서 남는 몇 거래는 여러 행에서 반복된다. 행이 많아져도 독립 거래 수가 늘지 않는다. 문턱을 낮추었을 때 평균이 높아지지 않거나 하반기 거래가 없으면 개선으로 판정하지 않는다.",
              "전체 구간의 최고 평균 문턱을 고른 뒤 확정 규칙으로 쓰지 않는다. 이미 확인한2026년 가격을 다시 나눈 탐색이며 문턱 선택 편향이 있다.", "",
              "## 문턱별 상반기 청산 선택 → 하반기 적용", "",
              "상반기 한쪽0.10% 모델 비용 후 평균으로4시각 중 하나를 고르고 하반기에 고정한다. 문턱 자체를 하반기 수익으로 고르지 않는다. 상반기0건이면 시각선택없음,하반기0건이면 확인없음이다.", "",
              "|차트 |RSI≤ |선택 매도ET |상반기 건수 |하반기 유효 호가 |하반기 평균 |95% 구간 |",
              "|---|---:|---|---:|---:|---:|---|"]
    for tf in FRAMES:
        for level in (30, 25, 20, 15, 10, 5):
            w = next(w for w in result["walk_forward"] if w["tf"] == tf and w["threshold"] == level)
            s = w["H2"]["quote"] if w["H2"] else {"n": 0}
            ci = s.get("mean_ci95_pct")
            ci_text = "—" if ci is None else f'{fmt(ci[0])}~{fmt(ci[1])}'
            lines.append(f'| {tf} | {level} | {w["selected_exit"] or "—"} | {w["H1"]["n_signal"] if w["H1"] else 0} | {s["n"]} | {fmt(s.get("mean_pct"))} | {ci_text} |')
    lines += ["", "## 정수1~30 전체 결과", "",
              "각 평균 뒤 괄호는 유효 호가 거래 수다.19:00의1/13 오래된 호가 제외는 원 실험과 같다.1분봉과3분봉은 각각14분·42분 명목 시간창이므로 직접적인 봉 주기 우월성 비교가 아니다."]
    for tf in FRAMES:
        lines += ["", f"### {tf} RSI14", "", "|RSI≤ |신호일 |16:30 평균(n) |17:00 평균(n) |18:00 평균(n) |19:00 평균(n) |", "|---|---:|---:|---:|---:|---:|"]
        for level in reversed(LEVELS):
            row = [lookup[(tf, level, ex, "all")] for ex in shared().EXITS]
            lines.append(f'| {level} | {row[0]["n_signal"]} | ' + " | ".join(f'{fmt(c["quote"].get("mean_pct"))} ({c["n_valid_quote"]})' for c in row) + " |")
    meta = result["meta"]
    lines += ["", "## 범위·검증·한계", "",
              "- 2차트×30문턱×4청산=240칸(기존≤30의8칸 포함). 이번새232칸,이전누적20칸을 포함해 전체누적252칸. 비용·지연의 추가 최적화는 하지 않았다.",
              f'- 문턱1~30의 고유 신호 집합 수(빈 집합 포함): {meta["distinct_signal_sets"]}. 동일 거래를 재사용한 문턱끼리는 독립이지 않다.',
              f'- 관측 최저 RSI: {meta["minimum_rsi"]}. 기준 미달 거래가 없는 문턱은 성과를 평가할 수 없다.',
              "- 거래일 단위10,000회 부트스트랩(seed20261105)은 원 study.stats를 재사용했다.1~2건 구간은 관측하지 못한 큰 손실을 표현하지 못한다. 여러 문턱의 최고 평균 선택 편향을 보정한 구간이 아니다.",
              "- 원 연구의 최소20건·평균양수·95%하한양수·무조건매수대조 우위 기준을 통과한 문턱은 없다. 낮은 문턱에서0건은 안전하거나 수익0이라는 의미가 아니다.",
              "- 기존 주문 가능 시간과 매매 전략은 바꾸지 않는다. 데이터·신호 시점·애프터장 체결 제약은 원 보고서와 같다.", "",
              "재현: `py data/derived/analysis/soxl_close_rsi/code/thresholds.py`, 독립검산: `py data/derived/analysis/soxl_close_rsi/code/audit_thresholds.py`. [전체 기계 결과](results/thresholds_summary.json), [전체 요약 CSV](results/thresholds_cells.csv), [원시 봉·호가 독립 검증](results/thresholds_audit.json).", ""]
    (ROOT / "thresholds_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    events = load_events()
    cells, wf = analyze(events)
    signals = pd.read_csv(OUT / "signals.csv")
    meta = {"thresholds": list(LEVELS), "timeframes": list(FRAMES), "exits": list(shared().EXITS),
            "matrix_cells_excluding_periods": 240, "new_cells": 232, "cumulative_cells": 252,
            "end": shared().END, "minimum_rsi": {}, "distinct_signal_sets": {},
            "input_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                             for name in ("signals.csv", "trades.csv", "quotes_trades.csv")}}
    for tf in FRAMES:
        s = signals[signals.tf.eq(tf)]
        meta["minimum_rsi"][tf] = float(s.rsi.min())
        meta["distinct_signal_sets"][tf] = len({tuple(s[eligible(s.rsi, level)].day) for level in LEVELS})
    result = {"meta": meta, "comparison_fixed_exits": FIXED, "cells": cells, "walk_forward": wf}
    (OUT / "thresholds_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    flat = []
    for c in cells:
        flat.append({k: v for k, v in c.items() if k in ("tf", "threshold", "exit_hm", "period", "n_signal", "n_valid_quote", "mae_worst_pct")}
                    | {f'{group}_{key}': value for group in ("model_gross", "model_net10bp", "quote", "quote_cost2")
                       for key, value in c[group].items() if key != "mean_ci95_pct"})
    pd.DataFrame(flat).to_csv(OUT / "thresholds_cells.csv", index=False)
    report(result)
    print(json.dumps({"cells": len(cells), "meta": meta, "passing": sum(w["passes_frozen_gate"] for w in wf)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
