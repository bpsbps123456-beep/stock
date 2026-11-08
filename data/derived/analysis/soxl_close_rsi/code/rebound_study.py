"""Compare first oversold RSI with first rising RSI still below the cutoff.

Uses visualize.build_inputs() for observed RSI and study.make_trades()/stats()
for the frozen price and cost model. No quotes or actual fills are assumed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FRAMES = ("1m", "3m")
RULES = ("level", "rising")
LEVELS = tuple(range(1, 31))
PERIODS = {"all": ("2026-01-01", "2026-11-03"),
           "H1": ("2026-01-01", "2026-06-30"), "H2": ("2026-07-01", "2026-11-03")}
FIXED = {"1m": "19:00", "3m": "18:00"}


def load_visual():
    spec = importlib.util.spec_from_file_location("rebound_shared_visual", ROOT / "code/visualize.py")
    visual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visual)
    return visual


def signal_rows(inputs, visual):
    study = inputs["study"]
    rows = []
    for tf in FRAMES:
        for cutoff in LEVELS:
            for rule in RULES:
                for day in inputs["exact_days"]:
                    minute = visual.first_signal(day, tf, cutoff, rising=rule == "rising")
                    if minute is None:
                        continue
                    pos = day["times"][tf].index(minute)
                    decision = pd.Timestamp(day["date"], tz=study.ET) + pd.Timedelta(minutes=minute)
                    rows.append({"day": day["date"], "tf": tf, "threshold": cutoff, "rule": rule,
                                 "decision": decision.isoformat(), "decision_minute": minute,
                                 "rsi": day["r"][tf][pos], "previous_rsi": day["r"][tf][pos - 1], "signal": True})
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["day", "tf", "threshold", "rule"]).any()
    for tf in FRAMES:
        for cutoff in LEVELS:
            selected = frame[frame.tf.eq(tf) & frame.threshold.eq(cutoff)]
            base = selected[selected.rule.eq("level")].set_index("day")
            rising = selected[selected.rule.eq("rising")].set_index("day")
            assert set(rising.index) <= set(base.index)
            assert (rising.decision_minute >= base.loc[rising.index, "decision_minute"]).all()
            assert (rising.rsi > rising.previous_rsi).all()
            assert rising.rsi.le(cutoff).all()
    return frame


def model_trades(inputs, signals):
    study = inputs["study"]
    unique = signals.drop_duplicates(["day", "decision"])[["day", "decision", "rsi", "signal"]].copy()
    unique["tf"] = "window"
    # Reuse the existing engine once for each unique observed decision.
    engine, missing = study.make_trades(inputs["full"], unique)
    engine = engine[engine.delay.eq(1) & engine.exit_hm.isin(study.EXITS)]
    fields = ["day", "decision", "exit_hm", "entry", "exit", "entry_price", "exit_price",
              "gross_pct", "mae_pct", "mfe_pct", *[f"net_{int(c * 10000)}bp_pct" for c in study.COSTS]]
    expected = signals.merge(pd.DataFrame({"exit_hm": study.EXITS}), how="cross")
    frame = expected.merge(engine[fields], on=["day", "decision", "exit_hm"], how="left", validate="many_to_one")
    frame["valid_model_price"] = frame.gross_pct.notna()
    return frame, {"unique_decisions": len(unique), "unique_valid_event_horizons": len(engine),
                   "engine_missing_delay1_fixed_exits": sum(r["delay"] == 1 and r["exit_hm"] in study.EXITS for r in missing)}


def delta_stats(study, values):
    st = study.stats(values)
    if not st["n"]:
        return {"n": 0}
    return {"n": st["n"], "mean_pp": st["mean_pct"], "median_pp": st["median_pct"],
            "positive_share_pct": st["win_rate_pct"], "mean_ci95_pp": st["mean_ci95_pct"],
            "worst_pp": st["worst_pct"], "best_pp": st["best_pct"]}


def analyze(inputs, trades):
    study = inputs["study"]
    cells, pairs = [], []
    for tf in FRAMES:
        for cutoff in LEVELS:
            for hm in study.EXITS:
                for period, (start, finish) in PERIODS.items():
                    population = trades[trades.tf.eq(tf) & trades.threshold.eq(cutoff)
                                        & trades.exit_hm.eq(hm) & trades.day.between(start, finish)]
                    for rule in RULES:
                        all_ = population[population.rule.eq(rule)].sort_values("day")
                        valid = all_[all_.valid_model_price]
                        cells.append({"tf": tf, "threshold": cutoff, "rule": rule, "exit_hm": hm, "period": period,
                                      "n_signal": len(all_), "n_valid_model": len(valid),
                                      "signal_days": all_.day.tolist(), "valid_days": valid.day.tolist(),
                                      "missing_price_days": all_.loc[~all_.valid_model_price, "day"].tolist(),
                                      "gross": study.stats(valid.gross_pct),
                                      "costs": {str(int(c * 10000)): study.stats(valid[f"net_{int(c * 10000)}bp_pct"])
                                                for c in study.COSTS},
                                      "mae_worst_pct": float(valid.mae_pct.min()) if len(valid) else None})
                    base_signals = population[population.rule.eq("level")]
                    rise_signals = population[population.rule.eq("rising")]
                    base = base_signals[base_signals.valid_model_price]
                    rise = rise_signals[rise_signals.valid_model_price]
                    same = base.merge(rise, on="day", suffixes=("_level", "_rising"), validate="one_to_one").sort_values("day")
                    baseline_only = base[~base.day.isin(rise_signals.day)]
                    pairs.append({"tf": tf, "threshold": cutoff, "exit_hm": hm, "period": period,
                                  "common_days": same.day.tolist(), "n_common_valid": len(same),
                                  "same_day_level_net10bp": study.stats(same.net_10bp_pct_level),
                                  "same_day_rising_net10bp": study.stats(same.net_10bp_pct_rising),
                                  "rising_minus_level_pp": delta_stats(study, same.net_10bp_pct_rising - same.net_10bp_pct_level),
                                  "mean_entry_wait_minutes": float((same.decision_minute_rising - same.decision_minute_level).mean()) if len(same) else None,
                                  "baseline_only_days": baseline_only.day.tolist(),
                                  "baseline_only_net10bp": study.stats(baseline_only.net_10bp_pct),
                                  "rising_signal_but_missing_price_days": rise_signals.loc[~rise_signals.valid_model_price, "day"].tolist()})
    by_cell = {(x["tf"], x["threshold"], x["rule"], x["exit_hm"], x["period"]): x for x in cells}
    by_pair = {(x["tf"], x["threshold"], x["exit_hm"], x["period"]): x for x in pairs}
    walk_forward = []
    for tf in FRAMES:
        for cutoff in LEVELS:
            candidates = [by_cell[(tf, cutoff, "level", hm, "H1")] for hm in study.EXITS
                          if by_cell[(tf, cutoff, "level", hm, "H1")]["n_valid_model"]]
            if not candidates:
                walk_forward.append({"tf": tf, "threshold": cutoff, "selected_exit": None,
                                     "selection": "baseline_H1_net10bp_mean_only", "H1": None, "H2": None,
                                     "reason": "no_baseline_H1_signal_with_valid_price"})
                continue
            best = max(candidates, key=lambda c: c["costs"]["10"]["mean_pct"])
            hm = best["exit_hm"]
            walk_forward.append({"tf": tf, "threshold": cutoff, "selected_exit": hm,
                                 "selection": "baseline_H1_net10bp_mean_only",
                                 **{period: {"level": by_cell[(tf, cutoff, "level", hm, period)],
                                             "rising": by_cell[(tf, cutoff, "rising", hm, period)],
                                             "paired": by_pair[(tf, cutoff, hm, period)]} for period in ("H1", "H2")},
                                 "reason": "H2_not_used_for_selection"})
    return cells, pairs, walk_forward


def pct(value):
    return "—" if value is None else f"{value:+.2f}%"


def pp(value):
    return "—" if value is None else f"{value:+.2f}%p"


def report(result):
    by_cell = {(x["tf"], x["threshold"], x["rule"], x["exit_hm"], x["period"]): x for x in result["cells"]}
    by_pair = {(x["tf"], x["threshold"], x["exit_hm"], x["period"]): x for x in result["paired"]}
    wf30 = {tf: next(w for w in result["walk_forward"] if w["tf"] == tf and w["threshold"] == 30)
            for tf in FRAMES}
    h2_text = []
    for tf in FRAMES:
        w = wf30[tf]
        a, b = (w["H2"][rule] for rule in RULES)
        h2_text.append(f'{tf} {w["selected_exit"]} ET: 기준 {a["n_valid_model"]}건 {pct(a["costs"]["10"].get("mean_pct"))} → 상승 {b["n_valid_model"]}건 {pct(b["costs"]["10"].get("mean_pct"))}')
    lines = ["# SOXL 마감 전 RSI: 기준 이하 첫 진입 vs 기준 이하에서 상승 확인", "",
             "2026-11-08 · Path-Q 연구 · 사용자 조건 정본: `docs/desk_system.md` §7", "",
             "**요약: 같은 날끼리 비교하면 상승 확인 후 진입 수익률은 개선됐지만, 전략 전체가 좋아졌다는 근거는 없다.** 기준30에서 상반기 기준 규칙으로 고른 청산 시각을 하반기 두 규칙에 똑같이 적용하면 다음과 같다.",
             " / ".join(h2_text) + ".",
             "상승 규칙의 하반기 표본은 각각7·5건으로 원 연구의 최소20건에도 미달한다. 거래하지 않은 날짜가 달라진 효과를 함께 봐야 하며, 새 진입 시각의 실측 호가는 아직 검증하지 않았다.", "",
             "15:30~16:00 ET에 진입 기회를 찾는 규칙의 후속 비교다. ‘상승’은 **현재 완성 RSI가 기준 이하이고 직전 완성 RSI보다 높음** 한 가지로 정의했다. 연속 상승 봉 수·기울기 크기·미래 저점 피벗을 추가 탐색하지 않았다.",
             "실행 가정은 기존과 같이 판단 후 1분 지연, 매수 시각<16:00이다. 따라서 마지막 판단은 1분봉15:58, 3분봉15:57이다. 15:30에 이미 상승 중이며 기준 이하라면 그때 진입할 수 있다. 매도는16:30·17:00·18:00·19:00 ET 중 미리 정한 시각이다.", "",
             "**아래는 분봉 시가와 고정 비용모형의 수익률이며 실제 호가·체결 수익률이 아니다.** 새 진입 시각의 호가를 재수집하지 않았다. 수수료0, 주 비교는 편도0.10%(10bp), 감도는0·5·10·20bp다. 추가 시장충격·환율·세금은 제외했다.", "",
             "## 같은 청산 시각에서 비교", "",
             "1분봉은19:00, 3분봉은18:00을 고정했다. 이는 이전 연구의 비교 시각이며 새 상승 조건에 맞춰 바꾸지 않았다. 두 차트의 RSI14는 명목14분·42분 창이므로 봉 주기 자체의 우월성 비교로 읽지 않는다.",
             "아래 n은 진입·청산 분봉 가격이 모두 있는 건수다. 신호일 수와 다를 수 있다. 예를 들어1분봉≤30은 신호64/23일이지만19:00 가격 유효 건수는63/22건이다. 누락 분 가격을 보간하지 않았다.", ""]
    for tf in FRAMES:
        hm = FIXED[tf]
        lines += [f"### {tf} RSI14 · {hm} ET 고정 매도", "",
                  "| RSI≤ |기준 첫 진입 n |평균 순수익 |상승 확인 n |평균 순수익 |동일 날짜 n |같은 날 상승−기준 |95% 구간 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for level in (30, 25, 20, 15, 10, 5):
            a, b = (by_cell[(tf, level, rule, hm, "all")] for rule in RULES)
            pair = by_pair[(tf, level, hm, "all")]
            ds = pair["rising_minus_level_pp"]
            ci = ds.get("mean_ci95_pp")
            ci_text = "—" if ci is None else f"{pp(ci[0])}~{pp(ci[1])}"
            lines.append(f'| {level} | {a["n_valid_model"]} | {pct(a["costs"]["10"].get("mean_pct"))} | {b["n_valid_model"]} | {pct(b["costs"]["10"].get("mean_pct"))} | {pair["n_common_valid"]} | {pp(ds.get("mean_pp"))} | {ci_text} |')
        lines += ["", "평균 차이에는 진입 시점뿐 아니라 진입하지 않은 날의 차이도 섞인다. ‘같은 날 상승−기준’은 양쪽 가격이 있는 공통 날짜에서만 뺀 값이다.", ""]
    lines += ["## 상반기 기준 규칙으로 매도 시각 선택 → 하반기에 두 규칙 동일 적용", "",
              "각 문턱에서 기준 이하 첫 진입 규칙의 상반기10bp 비용 후 평균으로 청산 시각을 고른다. 상승 규칙이나 하반기 성과를 보고 청산 시각을 다시 고르지 않는다. 동률이면 먼저 나오는 이른 시각이다.", "",
              "|차트 |RSI≤ |상반기 선택 ET |하반기 기준 n |평균 |하반기 상승 n |평균 |공통 날짜 차이 |95% 구간 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for tf in FRAMES:
        for level in (30, 25, 20, 15, 10, 5):
            w = next(x for x in result["walk_forward"] if x["tf"] == tf and x["threshold"] == level)
            if w["H2"] is None:
                lines.append(f'| {tf} | {level} | — | 0 | — | 0 | — | — | — |')
                continue
            a, b, pair = (w["H2"][key] for key in ("level", "rising", "paired"))
            ds = pair["rising_minus_level_pp"]
            ci = ds.get("mean_ci95_pp")
            ci_text = "—" if ci is None else f"{pp(ci[0])}~{pp(ci[1])}"
            lines.append(f'| {tf} | {level} | {w["selected_exit"]} | {a["n_valid_model"]} | {pct(a["costs"]["10"].get("mean_pct"))} | {b["n_valid_model"]} | {pct(b["costs"]["10"].get("mean_pct"))} | {pp(ds.get("mean_pp"))} | {ci_text} |')
    lines += ["", "## 문턱30의 날짜 선택 효과와 비용 감도", ""]
    for tf in FRAMES:
        hm = FIXED[tf]
        pair = by_pair[(tf, 30, hm, "all")]
        same_a, same_b = pair["same_day_level_net10bp"], pair["same_day_rising_net10bp"]
        dropped = pair["baseline_only_net10bp"]
        lines += [f'### {tf} · {hm}', "",
                  f'- 공통 {pair["n_common_valid"]}일: 기준 진입 평균 {pct(same_a.get("mean_pct"))}, 상승 확인 평균 {pct(same_b.get("mean_pct"))}. 평균 대기 {f"{pair['mean_entry_wait_minutes']:.2f}" if pair["mean_entry_wait_minutes"] is not None else "—"}분.',
                  f'- 상승 조건이 끝내 없어서 빠진 기준 규칙 {dropped["n"]}일의 평균은 {pct(dropped.get("mean_pct"))}. 빠진 날의 성과를 진입 가격 개선으로 부르지 않는다.',
                  "", "|조건 |비용0 |편도5bp |편도10bp |편도20bp |", "|---|---:|---:|---:|---:|"]
        for rule in RULES:
            cell = by_cell[(tf, 30, rule, hm, "all")]
            lines.append(f'| {rule} | ' + " | ".join(pct(cell["costs"][cost].get("mean_pct")) for cost in ("0", "5", "10", "20")) + " |")
        lines += [""]
    lines += ["", "## 자료·탐색 범위·해석 제한", "",
              "- 기간은 기존과 같은2026-01-02~11-03, 211거래일이다. 2025년은 RSI 초기화용이며 이전 연도 성과로 후보를 고르지 않았다. 상반기 선택·하반기 적용도 이미 본2026년 자료의 재사용이지 새 독립 검증은 아니다.",
              "- 2차트×30문턱×2진입 규칙×4청산=480조합, 기간별 통계1440칸. 이전 단일 시점 연구252조합을 포함하면 누적732조합이다. 비용 감도4개는 별도이며 추가 최적화하지 않았다. 문턱끼리 신호일이 겹치므로 독립 증거로 더하지 않는다.",
              "- 상승 확인은 신호를 거르거나 진입을 늦추는 조건이다. 거래하지 않은 날의 구성과 표본 감소 때문에 전체 평균이 바뀔 수도 있다. RSI 상승이 가격의 미래 상승을 보장하지 않는다.",
              "- 95% 구간은 원 study.stats의 거래일 단위10,000회 부트스트랩(seed20261105)이다. 여러 문턱 중 최고값 선택 편향을 보정하지 않는다. 1~2건의 좁은 구간은 관측하지 못한 손실을 반영하지 못한다. 빈 표본은0%로 채우지 않았다.",
              "- 1분 지연은 구현 가정이다. 봉이 완성된 뒤 알 수 있는 RSI와 직전 봉만 사용하며, 사후 최저 RSI·미래 피벗으로 진입을 정하지 않는다. 손절·익절은 넣지 않았다.",
              "- 동일 날짜에서도 서로 다른 진입 시각이므로 수익률 차이는 같은 주식수 손익과 다를 수 있다. 각 진입 가격을 기준으로 정규화한 거래 수익률을 비교했다.",
              "- 실제 매매 규칙·주문 시간·수량은 변경하지 않는다. 새 진입 시각의 호가·유동성·실제 체결은 별도 검증이 필요하다.", "",
              "재현: `py data/derived/analysis/soxl_close_rsi/code/rebound_study.py`. [실제 RSI·진입 비교 그림](strategy_visual.html), [전체 결과](results/rebound_summary.json), [일별 신호](results/rebound_signals.csv), [가격모형 거래](results/rebound_trades.csv), [독립 검산](results/rebound_audit.json).", ""]
    (ROOT / "rebound_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    visual = load_visual()
    inputs = visual.build_inputs()
    signals = signal_rows(inputs, visual)
    trades, engine_meta = model_trades(inputs, signals)
    cells, pairs, wf = analyze(inputs, trades)
    result = {"meta": {"as_of": "2026-11-08", "start": "2026-01-02", "end": "2026-11-03", "sessions": 211,
                       "rules": list(RULES), "timeframes": list(FRAMES), "thresholds": list(LEVELS),
                       "rule_definition": "rising: RSI_t <= cutoff and RSI_t > RSI_previous_completed_bar",
                       "matrix_cells": 480, "period_cells": len(cells), "paired_period_cells": len(pairs),
                       "cumulative_parameter_cells": 732, "cost_model_one_way_bp": [0, 5, 10, 20],
                       "actual_quote_returns": False, "fixed_comparison_exits": FIXED, **engine_meta},
              "cells": cells, "paired": pairs, "walk_forward": wf}
    (OUT / "rebound_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    signals.to_csv(OUT / "rebound_signals.csv", index=False)
    trades.to_csv(OUT / "rebound_trades.csv", index=False)
    report(result)
    display = []
    for tf in FRAMES:
        for rule in RULES:
            c = next(c for c in cells if c["tf"] == tf and c["threshold"] == 30 and c["rule"] == rule
                     and c["exit_hm"] == FIXED[tf] and c["period"] == "all")
            display.append({"tf": tf, "rule": rule, "n": c["n_valid_model"], "net10bp": c["costs"]["10"]})
    print(json.dumps({"meta": result["meta"], "threshold30": display}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
