"""Repeat the fixed ETF research grid on US listings and actual US sessions.

    py -m gbrc.etf.research_us_experiment [--reuse-grid] [--threads 8]

The original global experiment remains unchanged.  This is additional adaptive
research after its results were seen, not a new independent holdout.  Reuse of
the existing engine intentionally preserves full-fund CU flows and NAV weights.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from . import research, research_robust, research_stress, research_subgroups
from .research_engine import (
    FREQUENCIES, REGIMES, WEIGHTINGS, build_rankings, simulate_detail,
)
from .research_us_only import _filter_us, _hash
from .research_validation import _one_metrics
from .storage import atomic_write_json, atomic_write_text


STATUS = (
    "US-only adaptive additional exploration after viewing original global research results. "
    "The historical August-September holdout is reused, not independent validation. "
    "Candidate selection remains January-May training and June-July validation only."
)
TIME_AXIS = ("dates", "prices", "tradable", "price_age_days", "fx_age_days", "benchmarks")
SOURCE_FILES = (
    "bundle.npz", "metadata.json", "raw_prices.json", "protocol.json",
    "configurations.json", "grid_returns.npy", "evaluation.json", "stress.json",
    "subgroup_evaluation.json", "robust_research.json",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _us_context(value: Any) -> Any:
    """Clarify reused artifact wording without changing computed metrics or IDs."""
    if isinstance(value, dict):
        return {key: _us_context(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_us_context(item) for item in value]
    if isinstance(value, str):
        return (value.replace("global valuation-session", "US trading session")
                .replace("global valuation session", "US trading session")
                .replace("observed valuation dates", "observed US trading dates")
                .replace("Global daily local/FX closes", "US stock/FX daily closes"))
    return value


def prepare(source: Path, out: Path) -> dict[str, Any]:
    """Write a US security/date slice, preserving all KRX disclosure inputs."""
    hashes = {name: _hash(source / name) for name in SOURCE_FILES}
    original_meta = _load(source / "metadata.json")
    if hashes["bundle.npz"] != original_meta["bundle_sha256"]:
        raise ValueError("Original bundle checksum does not match its metadata")
    if hashes["raw_prices.json"] != original_meta["raw_prices_sha256"]:
        raise ValueError("Original raw price checksum does not match its metadata")
    engine_hash = _hash(Path(__file__).with_name("research_engine.py"))
    validation_hash = _hash(Path(__file__).with_name("research_validation.py"))
    if (out / "bundle.npz").exists():
        meta = _load(out / "metadata.json")
        audit = meta["us_only_experiment"]
        if (audit["source_artifact_sha256"] != hashes
                or audit["engine_sha256"] != engine_hash
                or audit["validation_sha256"] != validation_hash):
            raise ValueError("Existing US experiment has different source artifacts or engine")
        if _hash(out / "bundle.npz") != meta["bundle_sha256"]:
            raise ValueError("Existing US bundle checksum changed")
        return meta

    original = dict(np.load(source / "bundle.npz", allow_pickle=False))
    raw = _load(source / "raw_prices.json")
    arrays, securities, security_mask = _filter_us(original, original_meta)
    original_dates = original["dates"].astype(str)
    date_mask = np.asarray([day in raw["QQQ"] for day in original_dates], dtype=bool)
    if not date_mask.any():
        raise ValueError("No observed QQQ sessions inside the original sample")
    for key in TIME_AXIS:
        if arrays[key].shape[0] != len(original_dates):
            raise ValueError("Unexpected valuation date axis: " + key)
        arrays[key] = arrays[key][date_mask].copy()
    dates = arrays["dates"].astype(str)
    expected = sorted(day for day in raw["QQQ"]
                      if original_dates[0] <= day <= original_dates[-1])
    if dates.tolist() != expected:
        raise ValueError("US date axis omits an observed QQQ session")
    for key in ("snapshot_dates", "cash", "excluded_weight", "flow_base_dates", "lookbacks"):
        np.testing.assert_equal(arrays[key], original[key])
    for key in ("weights", "flow_pp", "flow_kind"):
        np.testing.assert_equal(arrays[key], original[key][..., security_mask])
    # Keep the omitted original NAV fraction explicitly; never renormalize it.
    arrays["non_us_nav_weight"] = original["weights"][..., ~security_mask].sum(axis=2)
    partition = (arrays["weights"].sum(axis=2) + arrays["cash"]
                 + arrays["excluded_weight"] + arrays["non_us_nav_weight"])
    if np.max(np.abs(partition - 1)) > 1e-7:
        raise ValueError("Filtered US securities and original NAV partition do not reconcile")
    first = int(np.searchsorted(dates, arrays["snapshot_dates"][0], side="right"))
    fresh = int(np.flatnonzero(dates >= "2026-08-01")[0])
    for day in (str(dates[first]), str(dates[fresh])):
        for symbol in ("456600.KS", "0015B0.KS", "QQQ"):
            if day not in raw[symbol]:
                raise ValueError("Benchmark entry lacks an observed close: " + symbol + " " + day)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "bundle.npz", **arrays)
    shutil.copyfile(source / "raw_prices.json", out / "raw_prices.json")
    removed_dates = original_dates[~date_mask].tolist()
    audit = {
        "research_status": STATUS,
        "source_artifact_sha256": hashes,
        "source_directory": str(source),
        "engine_sha256": engine_hash,
        "validation_sha256": validation_hash,
        "original_security_count": len(security_mask),
        "us_security_count": int(security_mask.sum()),
        "removed_non_us_securities": [item for item in original_meta["securities"]
                                       if item["market"] != "US"],
        "original_valuation_date_count": len(original_dates),
        "us_session_count": len(dates),
        "removed_non_us_session_dates": removed_dates,
        "calendar_rule": "actual observed QQQ Close dates within original sample only",
        "flow_windows": "1/3/5/10/20/40 KRX sessions; original full-fund CU/common factors",
        "price_windows": "20 actual US sessions, causal features using prior closes only",
        "extra_delay": "one additional US session for both disclosure and price signals",
        "weekly_monthly_schedule": "regenerated from US dates; US holidays move to next US session",
        "weights": "original NAV weights retained, not renormalized after US filtering",
        "cash": "original fund cash inputs unchanged; simulated residual cash in KRW",
        "benchmark_entry_actual_closes_verified": [str(dates[first]), str(dates[fresh])],
        "benchmark_marks": "direct Korean ETFs sampled on the same US date axis; KRX holidays carry prior close",
        "retained_original_us_nav_weight_pct": {
            name: {"mean": float(arrays["weights"][:, i].sum(axis=1).mean() * 100),
                   "min": float(arrays["weights"][:, i].sum(axis=1).min() * 100),
                   "max": float(arrays["weights"][:, i].sum(axis=1).max() * 100)}
            for i, name in enumerate(("time", "koact"))},
    }
    meta = {
        **original_meta,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_start": str(dates[0]), "sample_end": str(dates[-1]),
        "security_count": len(securities), "trading_date_count": len(dates),
        "securities": securities,
        "bundle_sha256": _hash(out / "bundle.npz"),
        "raw_prices_sha256": _hash(out / "raw_prices.json"),
        "us_only_experiment": audit,
        "nav_partition_max_abs_error": float(np.max(np.abs(partition - 1))),
        "array_contract": {**original_meta["array_contract"],
            "non_us_nav_weight": "[S,2] original non-US security NAV fraction omitted from candidate universe"},
        "calendar_and_execution_limitation": (
            "Actual US stock sessions only. US Close and daily FX marks are not synchronized quotes. "
            "Fractional same-close fills are research assumptions; not guaranteed order returns."),
    }
    if isinstance(meta.get("price_coverage"), dict):
        meta["price_coverage"] = {key: value for key, value in meta["price_coverage"].items()
                                  if key in set(arrays["symbols"].astype(str))}
    atomic_write_json(out / "metadata.json", meta)
    protocol = _load(source / "protocol.json")
    protocol.update({
        "created_before_grid_execution": datetime.now(timezone.utc).isoformat(),
        "research_status": STATUS,
        "experiment": "US-listed securities only; maximum ten; actual US session calendar",
        "period": [str(dates[0]), str(dates[-1])],
        "security_count": len(securities), "us_session_count": len(dates),
        "calendar": audit["calendar_rule"],
        "flow_policy": audit["flow_windows"] + "; no recomputation on US subset",
        "nav_weight_policy": audit["weights"],
        "regime": "QQQ USD prior 20 actual US sessions, +/-3% plus MA20 confirmation; first21 unknown",
        "execution": "Same-US-close fractional fills after earning existing holdings return; "
            "latest disclosure strictly before execution date; no leverage, KRW cash zero interest",
        "price_basis": "Split-adjusted Close in KRW; dividends and taxes excluded; daily FX marks",
        "selection": "Jan-May log return minus half absolute MDD; training MDD <=-50% excluded; "
            "top1% training eligible, June-July objective winner; no August-September selection",
        "robust_scenarios": {"base_15bps": {"cost": .0015, "extra_us_session_lag": 0},
                             "delayed_30bps": {"cost": .003, "extra_us_session_lag": 1}},
        "order_access": "User confirmed orders can be submitted; no further access approval needed",
        "order_return_limitation": "Historical assumed fills do not guarantee actual order or account returns",
        "stress": ["0/15/30/50bps", "+1 actual US session", "fresh cash August start",
                   "full grid at +1US session/30bps", "regime and top-contributor sensitivity"],
        "limitations": [STATUS, original_meta["backfill_temporal_limitation"],
            "Maximum holdings is a cap; fewer eligible names can cause concentration.",
            "20-session price features and +1-session delay now refer to US sessions; flows remain KRX sessions.",
            "Split-adjusted close fills, fractional positions and daily FX are assumptions; taxes/dividends excluded.",
            "ETF benchmarks use US-date marks; this maximum drawdown samples that same date axis.",
            "There is no guarantee of future 2x/3x ETF returns or exact executable order returns."],
        "source_artifact_sha256": hashes,
    })
    atomic_write_json(out / "protocol.json", protocol)
    return meta


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}%"


def _rule_ko(config: dict) -> str:
    families = {
        "conviction": "기존 보유비중", "net_flow": "수량 기준 순매매",
        "buy_only": "실매수", "joint_buy": "두 펀드 동시 매수",
        "joint_hold": "두 펀드 공통 보유", "new_entry": "신규 편입",
        "repeated_buy": "반복 매수", "avoid_sell": "매도 신호 회피",
        "divergence": "두 펀드 매매 방향 엇갈림", "buy_momentum": "실매수와 가격상승",
        "buy_dip": "실매수와 가격눌림", "holding_momentum": "보유비중과 가격상승",
        "sold_reversal": "실매도 종목 역발상",
    }
    gates = {"none": "추가 가격필터 없음", "positive_momentum": "원화 환산 최근20미국거래일 수익률 양수",
             "above_ma20": "원화 환산 직전종가가20미국거래일 평균보다 높음"}
    frequency = {"daily": "매 미국거래일", "weekly": "매주 첫 미국거래일",
                 "monthly": "매월 첫 미국거래일"}
    weighting = {"equal": "동일비중", "conviction": "선정종목의 혼합 펀드비중 비례",
                 "inverse_volatility": "최근20미국거래일 변동성이 낮을수록 높은 비중"}
    cash = {"always": "주식 목표100%", "trend_half": "교체일 QQQ하락국면이면 주식50%·현금50%",
            "trend_cash": "교체일 QQQ하락국면이면 전액현금, 그 외 주식 목표100%",
            "manager_cash": "두 펀드의 평균 현금비중 반영"}
    return (f"{families[config['family']]} 신호, D-{config['lookback']}(KRX거래일), "
            f"TIME/KoAct 신호배합 {config['time_mix']:.0%}/{1-config['time_mix']:.0%}, "
            f"{gates[config['gate']]}, {frequency[config['rebalance']]} 교체, "
            f"최대{config['max_stocks']}종목, {weighting[config['weighting']]} 배분, "
            f"{cash[config['risk_rule']]}(적격종목이 없으면 현금)")


def write_report(out: Path, meta: dict, evaluation: dict, subgroups: dict,
                 robust: dict, stress: dict) -> dict[str, Any]:
    delayed = np.load(out / "delayed_grid_returns.npy", mmap_mode="r")
    arrays = dict(np.load(out / "bundle.npz", allow_pickle=False))
    raw = _load(out / "raw_prices.json")
    arrays["qqq_usd"] = np.asarray([raw["QQQ"][str(day)] for day in arrays["dates"]])
    delayed_ranking = build_rankings(arrays, extra_lag=1)
    fresh_start = int(np.flatnonzero(arrays["dates"] >= "2026-08-01")[0])
    labels = [
        ("전체 후보군 선택", "primary_selected"),
        ("최대 3종목 집단 선택", "max_stocks_3"),
        ("최대 5종목 집단 선택", "max_stocks_5"),
        ("두 펀드 혼합·최대 3~10종목·강건 선택", "robust_mixed_max_stocks_3_to_10"),
        ("두 펀드 혼합·최대 5~10종목·강건 선택", "robust_mixed_max_stocks_5_to_10"),
    ]
    rows, selected = [], {}
    for name in ("time", "koact", "qqq_krw"):
        bm = evaluation["segments"]["full"]["benchmarks"][name]
        fresh = stress["fresh_holdout_benchmarks"][name]
        rows.append([name.upper(), "—", "—", _pct(bm["return_pct"]), _pct(bm["max_drawdown_pct"]),
                     _pct(fresh["return_pct"]), "—", "—"])
    for title, label in labels:
        row = stress["candidates"][label]
        joint_stress = _one_metrics(delayed[row["index"]])
        config = row["config"]
        rank_config = {key: config[key] for key in ("family", "lookback", "time_mix", "gate")}
        rid = delayed_ranking["rank_configs"].index(rank_config)
        params = [rid, config["max_stocks"], FREQUENCIES.index(config["rebalance"]),
                  WEIGHTINGS.index(config["weighting"]), REGIMES.index(config["risk_rule"])]
        fresh_delayed = simulate_detail(params, arrays, delayed_ranking, cost=.003, start=fresh_start)
        selected[label] = {
            "label": title, "index": row["index"], "config": row["config"],
            "base_full": row["full"], "continuing_holdout": row["holdout"],
            "fresh_holdout": row["fresh_holdout"], "extra_us_lag_15bps": row["extra_lag"],
            "extra_us_lag_30bps_full": joint_stress,
            "extra_us_lag_30bps_fresh_holdout": _one_metrics(fresh_delayed["returns"]),
            "rule_korean": _rule_ko(config),
            "max_observed_stocks": row["max_observed_stocks"],
            "average_observed_stocks": row["average_observed_stocks"],
            "without_top_contributor": row["without_top_contributor"],
        }
        if label.startswith("robust_"):
            scope = robust["scopes"][label.removeprefix("robust_")]
            selected[label]["robust_scenarios"] = scope["selected"]["scenarios"]
        rows.append([title, row["index"], f"{row['max_observed_stocks']}/{row['average_observed_stocks']:.2f}",
                     _pct(row["full"]["return_pct"]),
                     _pct(row["full"]["max_drawdown_pct"]), _pct(row["fresh_holdout"]["return_pct"]),
                     _pct(joint_stress["return_pct"]),
                     _pct(selected[label]["extra_us_lag_30bps_fresh_holdout"]["return_pct"])])
    audit = meta["us_only_experiment"]
    summary = {
        "research_status": STATUS, "security_count": meta["security_count"],
        "us_session_count": meta["trading_date_count"], "candidate_count_per_scenario": 252720,
        "scenario_count": 2, "total_grid_paths": 505440,
        "unique_base_paths_rounded_1e_7": evaluation["unique_return_paths_rounded_1e_7"],
        "sample_start": meta["sample_start"], "sample_end": meta["sample_end"],
        "benchmarks": {name: {"full": evaluation["segments"]["full"]["benchmarks"][name],
                              "fresh_holdout": stress["fresh_holdout_benchmarks"][name]}
                       for name in ("time", "koact", "qqq_krw")},
        "candidates": selected,
        "both_scenario_full_target_counts": robust["scopes"]["all"]["full_period_joint_target_counts"],
        "input_preservation": audit,
    }
    lines = [
        "# 미국 상장 주식·최대 10종목 추가 조합 연구", "",
        f"**2026-01-02~09-23, 미국 상장 {meta['security_count']}종목, 실제 미국 거래일 "
        f"{meta['trading_date_count']}개**에서 252,720개 규칙을 다시 계산했다. 기본 편도15bp와 "
        "+1미국거래일 지연·편도30bp를 모두 전수 계산해 총505,440개 경로를 평가했다.", "",
        "이 연구는 기존 글로벌 결과를 본 뒤의 추가 탐색이다. 1~5월 훈련과6~7월 검증만으로 "
        "각 후보를 선택했지만, 8~9월은 이미 알려진 과거 자료이므로 새로운 독립 검증으로 부르지 않는다.", "",
        "| 선택 기준 / 비교 대상 | 번호 | 실제 최대/평균 종목수 | 전체 수익률 | 최대낙폭 | 8월 현금 출발 | +1미국일·30bp 전체 | +1미국일·30bp 8월 현금 출발 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines += ["", "전체·최대낙폭은1월부터의 계좌, ‘8월 현금 출발’은8월 첫 미국 거래일에 현금으로 "
              "다시 시작한 별도 계산이다. 지연 열은 같은 규칙에 하루의 정보 지연과 편도30bp 비용을 적용했다. "
              "강건 선택은 두 비용·지연 시나리오 중 더 나쁜 훈련·검증 점수로 고른 후보를 뜻한다.", "",
              "## 선택된 규칙", ""]
    for label, row in selected.items():
        lines.append(f"- **{row['label']} #{row['index']}**: {row['rule_korean']}.")
    hindsight = evaluation["hindsight_best"]
    lines += ["", "개별 종목의 가격 필터·변동성은 원화 환산 가격으로 계산한다. 시장 방어 신호는 "
        "QQQ 달러 가격으로 계산하며, 직전 20미국거래일 수익률이 -3% 미만이고 직전 종가가 "
        "20일 이동평균보다 낮을 때 하락국면이다. 신호는 교체일에만 매매에 반영한다. "
        "#246798의 TIME/KoAct 100%/0%는 순위·배분 가중치다. 두 펀드 모두 매수·신규라는 "
        "편입 조건은 그대로 적용한다.", "",
        f"참고: 전체 기간 결과를 보고 고른 사후 최고값은 #{hindsight['index']}의 "
              f"{_pct(hindsight['segments']['full']['return_pct'])}다. 이는 과거 최고값이며 미래 추천이나 "
              "독립 검증 후보가 아니다."]
    counts = summary["both_scenario_full_target_counts"]
    lines += ["", f"두 시나리오 모두에서 더 높은 수익의 ETF 대비 전체 기간2배를 넘긴 조합은 "
        f"{counts['at_least_2x_stronger_etf_in_both_scenarios']:,}개, 3배는 "
        f"{counts['at_least_3x_stronger_etf_in_both_scenarios']:,}개다. 서로 유사하거나 중복된 조합이 "
        "포함되므로 독립적인 성공 사례 수가 아니다.", "", "## 계산 범위", "",
        "- 미국 상장 주식과 ADR만 허용했다. 원래171종목에서150종목을 남겼고, 국내·일본·홍콩·중국 "
        "현지 상장 주식은 제외했다. 현금은 원화, 차입·레버리지는 없다.",
        f"- KRX공시179일의 전체 펀드 수량·공통배율·원래 NAV비중을 유지했다. 비중을 미국 종목만으로 "
        "다시 정규화하지 않았다. 가격지표20일과 추가 지연1일은 미국 거래일 기준이다.",
        "- 주간·월간 리밸런싱은 미국 달력으로 다시 만들었다. 미국 휴장일이면 다음 미국 거래일에 "
        "교체한다. ETF 비교도 같은 미국 날짜 축의 원화 평가액으로 맞췄다.",
        "- 분할 반영 종가, 소수 단위 포지션과 종가 체결, 일별 환율 평가를 가정했다. 비용은 편도15/30bp "
        "등의 시나리오이며 세금·배당·현금이자는 제외했다. 주문 가능 여부는 사용자 확인을 반영했다.",
        "- 이 수치는 가정한 체결의 역사적 결과다. 실제 주문·계좌 수익률 또는 앞으로 ETF의2~3배 수익을 "
        "보장하지 않는다. 최대종목 수는 상한이라 신호가 적으면1종목에 집중될 수 있다.", "",
        "기존 글로벌 결과는 보존했다. 전체 조합은 `all_results.csv`, 선택 근거는 `evaluation.json`, "
        "하위집단은 `subgroup_evaluation.json`, 두 시나리오는 `robust_research.json`, 매매 이력과 추가 "
        "스트레스는 `stress.json` 및 `*_historical_ledger.csv`에 있다.", "",
    ]
    atomic_write_json(out / "summary.json", summary)
    atomic_write_text(out / "report.md", "\n".join(lines))
    return summary


def run(source: Path = research.OUT, reuse: bool = False, threads: int = 8) -> dict[str, Any]:
    source = Path(source).resolve()
    out = (source / "us_only_experiment").resolve()
    if out.parent != source or out.name != "us_only_experiment":
        raise ValueError("US experiment output must be the isolated child directory")
    print("Preparing US-only universe and actual US-session calendar", flush=True)
    meta = prepare(source, out)
    if (out / "grid_returns.npy").exists() and not reuse:
        raise ValueError("US grid already exists; use --reuse-grid to preserve and reuse it")
    print(f"US securities={meta['security_count']}, sessions={meta['trading_date_count']}; "
          "full-fund KRX disclosure inputs preserved", flush=True)
    # reuse=True preserves the custom protocol; the engine simulates when no grid exists.
    base_result = research.run(out, reuse=True, threads=threads)
    evaluation = base_result[6]
    evaluation["holdout_status"] = STATUS
    evaluation["research_status"] = STATUS
    evaluation["universe"] = "US-listed shares and ADRs only, maximum ten positions"
    evaluation["execution_calendar"] = "actual QQQ observed US sessions"
    atomic_write_json(out / "evaluation.json", evaluation)
    del base_result
    gc.collect()
    print("Computing parameter-stratum diagnostics with original temporal selection", flush=True)
    subgroups = research_subgroups.run(out)
    print("Running full +1-US-session / 30-bps robustness grid", flush=True)
    robust = research_robust.run(out, reuse=reuse and (out / "delayed_grid_returns.npy").exists(),
                                threads=threads)
    robust = _us_context(robust)
    robust["research_status"] = STATUS
    robust["execution_calendar"] = "actual QQQ observed US sessions; legacy extra_global_session_lag means US sessions here"
    for scenario in robust["scenarios"].values():
        scenario["extra_us_session_lag"] = scenario["extra_global_session_lag"]
    atomic_write_json(out / "robust_research.json", robust)
    print("Replaying selected candidates: fresh cash, costs, delay and contribution", flush=True)
    with (out / "stress_execution.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        stress = research_stress.run(out)
    stress = _us_context(stress)
    stress["research_status"] = STATUS
    stress["assumptions"]["calendar"] = "actual observed QQQ US sessions only"
    stress["assumptions"]["universe"] = "US listings only; at most ten stocks"
    atomic_write_json(out / "stress.json", stress)
    summary = write_report(out, meta, evaluation, subgroups, robust, stress)
    original_hashes = meta["us_only_experiment"]["source_artifact_sha256"]
    if {name: _hash(source / name) for name in SOURCE_FILES} != original_hashes:
        raise ValueError("Original global experiment changed during US-only research")
    stats = np.load(out / "grid_stats.npy")
    delayed_stats = np.load(out / "delayed_grid_stats.npy")
    audit = {
        "original_source_artifacts_unchanged": True,
        "us_listing_count": meta["security_count"],
        "all_security_metadata_markets_us": all(x["market"] == "US" for x in meta["securities"]),
        "actual_us_session_count": meta["trading_date_count"],
        "full_fund_quantity_inputs_preserved": True,
        "original_nav_input_weights_not_renormalized": True,
        "original_krx_snapshot_calendar_preserved": True,
        "maximum_base_holdings": int(stats[:, 1].max()),
        "maximum_delayed_holdings": int(delayed_stats[:, 1].max()),
        "delayed_grid": robust["delayed_grid_audit"],
        "selected_all_index": evaluation["selected_index"],
        "selected_cap3_index": subgroups["subgroups"]["max_stocks_3"]["selected_global_index"],
        "selected_cap5_index": subgroups["subgroups"]["max_stocks_5"]["selected_global_index"],
        "selection_uses_holdout": False,
        "holdout_is_new_independent_validation": False,
    }
    if max(audit["maximum_base_holdings"], audit["maximum_delayed_holdings"]) > 10:
        raise ValueError("US experiment violated maximum ten holdings")
    atomic_write_json(out / "validation.json", audit)
    print(json.dumps({"out": str(out), "audit": audit,
                      "candidates": {name: {"index": row["index"], "base": row["base_full"],
                                            "fresh": row["fresh_holdout"],
                                            "lag30": row["extra_us_lag_30bps_full"]}
                                     for name, row in summary["candidates"].items()}},
                     ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=research.OUT)
    parser.add_argument("--reuse-grid", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run(args.source, args.reuse_grid, args.threads)


if __name__ == "__main__":
    main()
