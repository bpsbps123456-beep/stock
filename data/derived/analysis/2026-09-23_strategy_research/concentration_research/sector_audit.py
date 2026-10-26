"""Frozen watchlist-sector mapping and concentration audit; no strategy search.

Run: py data/derived/analysis/2026-09-23_strategy_research/concentration_research/sector_audit.py
Use --map-only to emit the engine's ordered sector map without the path audit.
Only sector_map.json, sector_audit.json and sector_audit.md beside this file are written.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
from watchlist import resolve_sector
from gbrc.etf.storage import atomic_write_json, atomic_write_text

COMMON = EXP / "strategy_comparison/results.json"
MIXTURE = EXP / "strategy_comparison/section16_review/mixture_audit.json"
METADATA = EXP / "us_only_experiment/metadata.json"
PANELS = EXP.parent / "strategy_lab/lab_panels.json"
WATCHLIST = REPO / "watchlist.py"
UNKNOWN = "미분류"
GROUPS = {
    "broad_semiconductor": {
        "label": "반도체 묶음",
        "source_sectors": ["AI 반도체", "메모리", "반도체 장비", "반도체 장비(적자)", "반도체(적자)", "반도체 소재"],
    },
    "ai_hardware_infrastructure": {
        "label": "반도체·AI 하드웨어/인프라",
        "source_sectors": ["AI 반도체", "메모리", "반도체 장비", "반도체 장비(적자)", "반도체(적자)", "반도체 소재",
                           "전자부품/AI 하드웨어", "네트워크/광통신", "AI 인프라"],
    },
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(paths):
    return {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in paths}


def make_map(common, metadata, panels):
    """Resolve only canonical labels and explicitly recorded historical aliases."""
    tickers = common["tickers"]
    securities = metadata["securities"]
    assert len(tickers) == len(securities) == 150
    records = []
    for i, (ticker, security) in enumerate(zip(tickers, securities)):
        assert security["market"] == "US"
        assert panels["ticker"][security["company_key"]] == ticker
        keys = list(dict.fromkeys([ticker, security["symbol"], security["company_key"].split()[0]]
                                 + [alias.split()[0] for alias in security["aliases"]]))
        lookups = {key: resolve_sector(key, default=UNKNOWN) for key in keys}
        known = {label for label in lookups.values() if label != UNKNOWN}
        if len(known) > 1:
            raise ValueError(f"Conflicting canonical classifications: {ticker}: {lookups}")
        label = next(iter(known)) if known else UNKNOWN
        matched = next((key for key in keys if lookups[key] == label), None) if known else None
        records.append({"index": i, "ticker": ticker, "metadata_symbol": security["symbol"],
                        "company_key": security["company_key"], "name": security["name"],
                        "sector": label, "resolved_listing_key": matched, "lookup_results": lookups,
                        "alias_recovery": resolve_sector(ticker, default=UNKNOWN) == UNKNOWN and label != UNKNOWN})
    labels = sorted({row["sector"] for row in records})
    ids = [labels.index(row["sector"]) for row in records]
    masks = {name: [row["sector"] in group["source_sectors"] for row in records] for name, group in GROUPS.items()}
    for row, sid in zip(records, ids):
        row["sector_id"] = sid
        row["risk_groups"] = [name for name, mask in masks.items() if mask[row["index"]]]
    unknown = [r["ticker"] for r in records if r["sector"] == UNKNOWN]
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "watchlist.py::resolve_sector only; metadata aliases resolve spelling, no new company classification",
        "source_sha256": source_hashes([COMMON, METADATA, PANELS, WATCHLIST]),
        "ordered_tickers": tickers, "ordered_company_keys": [s["company_key"] for s in securities],
        "detail_ids": ids, "detail_labels": labels, "unknown_id": labels.index(UNKNOWN),
        "risk_group_definitions": GROUPS, "risk_group_masks": masks,
        "records": records,
        "coverage": {"total": len(records), "classified": len(records)-len(unknown),
                     "classified_pct": 100*(len(records)-len(unknown))/len(records),
                     "unknown": len(unknown), "unknown_tickers": unknown,
                     "alias_recoveries": [{k: r[k] for k in ("ticker", "resolved_listing_key", "sector")} for r in records if r["alias_recovery"]],
                     "detail_counts": dict(sorted(Counter(r["sector"] for r in records).items())),
                     "risk_group_counts": {name: sum(mask) for name, mask in masks.items()}},
        "limitations": [
            "These are research risk groups made by joining existing labels, not GICS classifications.",
            "Current watchlist labels have no historical effective timestamps; applying them to 2026 is retrospective tagging.",
            "All unclassified names share one unknown bucket. They must not be interpreted as zero risk or confirmed non-semiconductor.",
            "Known broad-semiconductor exposure is a lower bound on that economic risk. Known group plus unknown is an explicit uncertainty bound.",
            "AI infrastructure includes equipment and data-center operators; the wider group represents AI-capex-related exposure, not pure hardware manufacturing.",
            "Risk-group sets overlap: semiconductor is a subset of the broader AI group. Do not add both exposures together.",
            "No classifications or group membership were optimized against candidate returns.",
        ],
    }


def holdings_matrix(holdings, n):
    weights = np.zeros((len(holdings), n))
    for t, rows in enumerate(holdings):
        for index, weight in rows:
            assert 0 <= int(index) < n and np.isfinite(weight) and weight >= 0
            weights[t, int(index)] += weight
    assert np.all(weights.sum(axis=1) <= 1 + 1e-9)
    return weights


def concentration(holdings, dates, sector_map):
    weights = holdings_matrix(holdings, len(sector_map["ordered_tickers"]))
    ids = np.asarray(sector_map["detail_ids"])
    labels = sector_map["detail_labels"]
    sector = np.stack([weights[:, ids == i].sum(axis=1) for i in range(len(labels))], axis=1)
    np.testing.assert_allclose(sector.sum(axis=1), weights.sum(axis=1), atol=1e-12)
    cash = 1-weights.sum(axis=1)
    np.testing.assert_allclose(sector.sum(axis=1)+cash, 1., atol=1e-12)
    descending = np.sort(sector, axis=1)[:, ::-1]
    top1 = descending[:, 0]
    top2 = descending[:, :2].sum(axis=1)
    groups = {name: weights[:, np.asarray(mask)].sum(axis=1) for name, mask in sector_map["risk_group_masks"].items()}
    unknown = sector[:, sector_map["unknown_id"]]
    dates = np.asarray(dates)

    def stat(mask):
        matrix = sector[mask]
        span = dates[mask]
        a = top1[mask]; b = top2[mask]
        flat = np.unravel_index(np.argmax(matrix), matrix.shape)
        return {
            "days": int(mask.sum()), "max_single_sector_pct": float(a.max()*100),
            "mean_daily_max_single_sector_pct": float(a.mean()*100),
            "maximum_sector": {"label": labels[int(flat[1])], "date": str(span[int(flat[0])]), "weight_pct": float(matrix[flat]*100)},
            "max_top2_sectors_pct": float(b.max()*100), "mean_daily_top2_sectors_pct": float(b.mean()*100),
            "mean_cash_pct": float(cash[mask].mean()*100),
            "unknown_max_pct": float(unknown[mask].max()*100), "unknown_mean_pct": float(unknown[mask].mean()*100),
            "sectors": {label: {"mean_nav_pct": float(matrix[:, i].mean()*100), "max_nav_pct": float(matrix[:, i].max()*100)}
                        for i, label in enumerate(labels) if np.any(matrix[:, i] > 0)},
            "risk_groups": {name: {"max_nav_pct": float(value[mask].max()*100), "mean_nav_pct": float(value[mask].mean()*100),
                                   "max_known_plus_unknown_pct": float((value[mask]+unknown[mask]).max()*100),
                                   "mean_known_plus_unknown_pct": float((value[mask]+unknown[mask]).mean()*100)} for name, value in groups.items()},
        }

    masks = {"full": np.ones(len(dates), bool), "train": dates < "2026-06-01",
             "june_july": (dates >= "2026-06-01") & (dates < "2026-08-01"), "august_september": dates >= "2026-08-01"}
    return {
        "segments": {name: stat(mask) for name, mask in masks.items() if mask.any()},
        "unknown_held_tickers": [sector_map["ordered_tickers"][i] for i in np.flatnonzero((ids == sector_map["unknown_id"]) & np.any(weights > 0, axis=0))],
        "daily": {"top1_sector_pct": (top1*100).tolist(), "top2_sector_pct": (top2*100).tolist(),
                  "unknown_pct": (unknown*100).tolist(), "cash_pct": (cash*100).tolist(),
                  "risk_group_pct": {name: (value*100).tolist() for name, value in groups.items()}},
    }


def run(map_only=False):
    files = [COMMON, METADATA, PANELS, WATCHLIST] + ([] if map_only else [MIXTURE])
    before = source_hashes(files)
    common, metadata, panels = read(COMMON), read(METADATA), read(PANELS)
    sector_map = make_map(common, metadata, panels)
    atomic_write_json(HERE / "sector_map.json", sector_map)
    if map_only:
        assert source_hashes(files) == before
        print(json.dumps(sector_map["coverage"], ensure_ascii=False))
        return
    mixture = read(MIXTURE)
    dates = common["dates"]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dates": dates,
        "protocol": {"source_sha256": before, "audit_code_sha256": sha(Path(__file__)),
                     "fixed_strategies": ["H6e", "H6", "201609", "239120"],
                     "fixed_mixtures": ["239120+H6e", "239120+H6"],
                     "allocation": "Mixes start with 50% each, then no sleeve transfers; actual same-stock exposures add at daily drifting sleeve NAV weights.",
                     "denominator": "Whole portfolio KRW close NAV including cash; cash is shown separately and is not a sector.",
                     "mean_definition": "Equal-weight mean across observed daily close weights; full-period includes all-cash days.",
                     "top2_definition": "Sum of the largest two sector weights on each day; labels may change daily. Max and time-average are distinct.",
                     "mapping_policy": sector_map["limitations"], "no_new_selection": True},
        "coverage": sector_map["coverage"], "strategies": {}, "mixtures": {},
    }
    for name in result["protocol"]["fixed_strategies"]:
        result["strategies"][name] = {}
        for scenario, row in common["strategies"][name]["scenarios"].items():
            audit = concentration(row["daily_holdings"], dates, sector_map)
            fresh = row["fresh_starts"]["august"]
            audit["fresh_august"] = concentration(fresh["daily_holdings"], [d for d in dates if d >= fresh["date"]], sector_map)
            result["strategies"][name][scenario] = audit
    for name in result["protocol"]["fixed_mixtures"]:
        result["mixtures"][name] = {}
        for scenario, row in mixture["pairs"][name].items():
            other = name.split("+")[1]
            a = common["strategies"]["239120"]["scenarios"][scenario]
            b = common["strategies"][other]["scenarios"][scenario]
            sleeve = np.asarray(a["daily_nav"])/(np.asarray(a["daily_nav"])+np.asarray(b["daily_nav"]))
            reconstructed = holdings_matrix(a["daily_holdings"], 150)*sleeve[:, None] + holdings_matrix(b["daily_holdings"], 150)*(1-sleeve[:, None])
            np.testing.assert_allclose(reconstructed, holdings_matrix(row["daily_holdings"], 150), atol=1e-12)
            audit = concentration(row["daily_holdings"], dates, sector_map)
            fresh = row["fresh_august"]
            audit["fresh_august"] = concentration(fresh["daily_holdings"], [d for d in dates if d >= fresh["start"]], sector_map)
            result["mixtures"][name][scenario] = audit
    result["validation"] = {"input_hashes_unchanged": source_hashes(files) == before,
                            "150_us_companies_aligned": True, "canonical_alias_conflicts": 0,
                            "unknown_is_one_bucket": True, "sector_plus_cash_equals_NAV": True,
                            "mixture_actual_sleeve_weights_reconstructed": True,
                            "semiconductor_subset_of_AI_group": bool(np.all(~np.asarray(sector_map["risk_group_masks"]["broad_semiconductor"]) | np.asarray(sector_map["risk_group_masks"]["ai_hardware_infrastructure"]))),
                            "classification_or_candidate_selection_changed": False}
    assert result["validation"]["input_hashes_unchanged"]
    atomic_write_json(HERE / "sector_audit.json", result)
    lines = ["# 기존 고정 후보의 섹터 집중 감사", "", "2026-01-02~09-23의 미국 종가 평가 NAV 비중이다. 같은 종목은 합산하고 현금은 별도로 둔다. 새 후보·비율·분류 최적화는 하지 않았다.", "",
             f"정본 `watchlist.resolve_sector`와 검증된 티커 별칭으로 150개 중 {sector_map['coverage']['classified']}개({sector_map['coverage']['classified_pct']:.1f}%)를 분류했다. 나머지 {sector_map['coverage']['unknown']}개는 하나의 **미분류** 버킷이다. ECHO→SATS, MOG-A→MOG/A는 기존 분류를 별칭으로 복원했으며 정본을 변경하지 않았다.", "",
             "## 공시 1미국거래일 지연: 집중도", "", "`평균 최대`는 매일 가장 큰 섹터 비중의 기간 평균이며, `최대`는 관측된 최고치다. 상위 2개는 매일 비중이 큰 두 섹터의 합이고 섹터 이름은 날짜마다 달라질 수 있다. 모두 현금을 포함한 전체 NAV 대비 비중이다.", "",
             "| 고정 후보 | 단일 섹터 최대 | 평균 최대 | 상위2 최대 | 상위2 평균 | 반도체 최대/평균 | AI설비 연관 최대/평균 | 미분류 최대/평균 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for collection in ("strategies", "mixtures"):
        for name, scenarios in result[collection].items():
            s = scenarios["publication_lag1"]["segments"]["full"]
            groups = s["risk_groups"]
            lines.append(f"| {name} | {s['max_single_sector_pct']:.2f}% | {s['mean_daily_max_single_sector_pct']:.2f}% | {s['max_top2_sectors_pct']:.2f}% | {s['mean_daily_top2_sectors_pct']:.2f}% | {groups['broad_semiconductor']['max_nav_pct']:.2f}/{groups['broad_semiconductor']['mean_nav_pct']:.2f}% | {groups['ai_hardware_infrastructure']['max_nav_pct']:.2f}/{groups['ai_hardware_infrastructure']['mean_nav_pct']:.2f}% | {s['unknown_max_pct']:.2f}/{s['unknown_mean_pct']:.2f}% |")
    lines += ["", "혼합은 시작자금만 50:50으로 나눠 이후 두 계좌 간 자금이동을 하지 않은 기존 경로다. 매일 50:50으로 재설정한 결과가 아니다. 모든 지연 0/1/2 시나리오와 1~5월·6~7월·8~9월 및 8월 현금출발 집중도는 JSON에 있다.", "",
              "## 분류를 합쳐 보는 이유", ""]
    for name, group in GROUPS.items():
        lines.append(f"- **{group['label']}** (`{name}`): " + ", ".join(group["source_sectors"]) + ".")
    lines += ["", "두 묶음은 공식 GICS 분류가 아니라 기존 정본 라벨을 합친 연구용 위험축이다. 반도체는 넓은 AI 묶음의 부분집합이므로 두 수치를 더하지 않는다. 넓은 묶음의 AI 인프라는 장비 회사와 데이터센터 운영사를 함께 포함하며, 순수 제조업이 아닌 AI 설비투자 연관 위험을 본다.", "",
              "현재 분류에는 과거 적용 시작 시각이 없다. 이를 2026년 과거에 적용한 사후 진단이며, 당시에도 동일한 분류를 알 수 있었다고 입증하지 않는다. 미분류 종목의 반도체 마스크가 false여도 비반도체로 확정된 뜻이 아니다. 알려진 반도체 비중과 `알려진 반도체+미분류` 불확실성 상한을 JSON에 함께 남겼다. 섹터 상한을 시험할 때도 미분류를 0위험으로 처리해서는 안 된다.", "",
              "## 미분류와 검증", "", ", ".join(sector_map["coverage"]["unknown_tickers"]), "",
              "150종목 순서·미국성·별칭 충돌 없음, 섹터+현금=전체 NAV, 혼합의 실제 계좌 비중 재구성, 원본 해시 불변을 확인했다. 기존 공통 엔진의 시가 체결·환율 근사·비용 0·세금 및 배당 제외 가정이 그대로 적용된다.", "",
              "[엔진용 섹터 맵](sector_map.json) · [전체 감사 JSON](sector_audit.json) · [재실행 코드](sector_audit.py)", "",
              "재실행: `py data/derived/analysis/2026-09-23_strategy_research/concentration_research/sector_audit.py`", "",
              "### ✅ 자기검증 (Self-Check)", "", "- 트리거 매칭: [Path-F ✅]", "- Gate B 통과: [ETF 정본·섹터 정본 Read + 기존 고정 경로 감사 Y]", "- 가정으로 채운 숫자: [현재 분류의 과거 적용과 기존 체결 가정 명시; 관측 비중 직접 집계]", "- 돌발 호재 확인: [불필요 ✅]", "- 규칙 위반: [없음 ✅ — 원본 분류·결과 불변]", ""]
    atomic_write_text(HERE / "sector_audit.md", "\n".join(lines))
    print(json.dumps({"coverage": sector_map["coverage"], "validation": result["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    run(map_only="--map-only" in sys.argv)
