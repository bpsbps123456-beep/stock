"""Small frozen concentration study; all portfolio mechanics reuse desk.strategies.

Run: py data/derived/analysis/concentration_review_20261105/review.py
Forward: add --end YYYY-MM-DD --price-dir data/derived/desk after cache refresh.
No quotes, orders, scheduled tasks or changes to operating settings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from desk import strategies as S
from desk import config

PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
FIELDS = {"Adj Close": "prices_adj_close.csv", "Open": "prices_open.csv", "Close": "prices_close.csv"}


def load_book(price_dir, end):
    """Read-only prices: never refresh or overwrite the shared desk cache."""
    prices = {f: pd.read_csv(price_dir / name, index_col=0, parse_dates=True)
              for f, name in FIELDS.items()}
    # US closes beyond end must not enter the ranking or the model.
    for f, df in prices.items():
        uscols = [c for c in df if c not in config.REFERENCES]
        df.loc[df.index > end, uscols + ["QQQ", "KRW=X"]] = np.nan
    original_loader = S.load_fund_snapshots
    def snapshots(fund, ignored):
        return original_loader(fund, end)
    # The PDF on the following Korean date cannot affect the completed US model.
    with patch.object(S.marketdata, "history", return_value=prices), patch.object(S, "load_fund_snapshots", snapshots):
        return S.load_book()


@contextmanager
def variant(candidate, activation="2026-01-02"):
    """Process-local research override. Restore settings even after a failure."""
    original_project, original_weekly = S.project_caps, S.weekly_target
    active = False
    def project(raw, sectors):
        rule = dict(config.RULE)
        if active:
            rule["stock_cap"] = candidate["stock_cap"]
        with patch.object(config, "RULE", rule):
            out = original_project(raw, sectors)
        if active and candidate["raw_ceiling"]:
            out = {k: min(v, raw[k]) for k, v in out.items()}
        return out
    def weekly(book, k, session, current):
        nonlocal active
        active = session >= activation
        return original_weekly(book, k, session, current)
    with patch.object(S, "project_caps", project), patch.object(S, "weekly_target", weekly):
        yield


def metrics(curve):
    curve = np.asarray(curve, float)
    return {"return_pct": float((curve[-1] / curve[0] - 1) * 100),
            "mdd_pct": float((curve / np.maximum.accumulate(curve) - 1).min() * 100)}


def exposure(path):
    # After each session's open trades; only a descriptive exposure statistic.
    return np.array([sum(w.values()) for w in path.weights])


def compare(book, paths, begin, end):
    days = paths["baseline"].days
    indices = [i for i, d in enumerate(days) if begin <= d <= end]
    if not indices:
        return {}
    lo, hi = indices[0], indices[-1]
    baseline_curve = S.model_index(book, paths["baseline"])
    rows = {}
    for name, path in paths.items():
        curve = S.model_index(book, path)
        # Include previous close, or initial money before the first session.
        segment = np.r_[curve[lo-1] if lo else 100., curve[lo:hi+1]]
        row = metrics(segment)
        row["mean_open_exposure_pct"] = float(exposure(path)[lo:hi+1].mean()*100)
        # Approximate USD cash-scaling diagnostic; open exposure omits overnight drift.
        baseline = paths["baseline"]
        bclose = np.r_[baseline.close[lo-1] if lo else 1., baseline.close[lo:hi+1]]
        bret = bclose[1:]/bclose[:-1]-1
        ratio = np.divide(exposure(path)[lo:hi+1], exposure(baseline)[lo:hi+1],
                          out=np.zeros(len(indices)), where=exposure(baseline)[lo:hi+1] > 0)
        control = np.r_[1., np.cumprod(1+bret*ratio)]
        actual = np.r_[path.close[lo-1] if lo else 1., path.close[lo:hi+1]]
        row["usd_exposure_control"] = metrics(control)
        row["usd_actual"] = metrics(actual)
        rows[name] = row
    return {"first_us": days[lo], "last_us": days[hi], "sessions": len(indices), "rows": rows}


def checks(book, paths, candidates):
    base = paths["baseline"]
    for candidate in candidates:
        p = paths[candidate["id"]]
        assert list(p.rebalances) == list(base.rebalances)
        for day, weights in p.rebalances.items():
            assert set(weights) == set(base.rebalances[day])
            assert max(weights.values(), default=0) <= candidate["stock_cap"]+1e-9
            assert sum(weights.values()) <= 1+1e-9
            sector = defaultdict(float)
            for k, w in weights.items():
                sector[book.sector[k]] += w
            assert max(sector.values(), default=0) <= .6+1e-9
            assert sum(w for k,w in weights.items() if S.in_semis_budget(book.sector[k])) <= .5+1e-9
    return "passed: all rebalance caps, NAV and rebalance dates"


def forward_decision(book, paths, end):
    start = PROTOCOL["forward"]["activation_us"]
    days = [d for d in paths["baseline"].days if start <= d <= end]
    if len(days) < 60:
        return {"status": "not_started" if not days else "observation_only",
                "sessions": len(days), "required": 60}
    assessment = compare(book, paths, start, days[59])
    base, primary = assessment['rows']['baseline'], assessment['rows'][PROTOCOL['forward']['primary']]
    passed = primary['return_pct'] >= base['return_pct'] and primary['mdd_pct'] > base['mdd_pct']
    return {"status": "candidate_only" if passed else "not_passed", "sessions": 60,
            "primary": PROTOCOL['forward']['primary'], "assessment": assessment,
            "operating_rule_changed": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=PROTOCOL["historical_end_us"])
    ap.add_argument("--price-dir", type=Path, default=HERE / "inputs")
    args = ap.parse_args()
    book = load_book(args.price_dir, args.end)
    if book.days[-1] != args.end:
        raise SystemExit(f"Requested completed US session {args.end}; cache ends {book.days[-1]}")
    candidates = PROTOCOL["candidates"]
    historical, forward = {}, {}
    for candidate in candidates:
        with variant(candidate):
            historical[candidate["id"]] = S.run_model(book)
        with variant(candidate, PROTOCOL["forward"]["activation_us"]):
            forward[candidate["id"]] = S.run_model(book)
    validation = checks(book, historical, candidates)
    # Before activation every forward replay must be identical to the baseline.
    cutoff = np.array(forward["baseline"].days) < PROTOCOL["forward"]["activation_us"]
    for p in forward.values():
        np.testing.assert_allclose(p.close[cutoff], forward["baseline"].close[cutoff], atol=1e-12)
    periods = {name: compare(book,historical,a,z) for name,a,z in [
        ("2026_1_5", "2026-01-02", "2026-05-29"),
        ("2026_6_latest", "2026-06-01", args.end),
        ("october_diagnostic", "2026-10-12", args.end),
        ("2026_full", "2026-01-02", args.end)]}
    fwd = compare(book, forward, PROTOCOL["forward"]["activation_us"], args.end)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / "data/normalized/etf_holdings").glob("*_456600.json"))
              if p.name[:10] <= args.end}
    hashes.update({name: hashlib.sha256((args.price_dir/name).read_bytes()).hexdigest() for name in FIELDS.values()})
    result = {"end_us": book.days[-1], "protocol_sha256": hashlib.sha256((HERE/"protocol.json").read_bytes()).hexdigest(),
              "periods": periods, "forward": fwd or {"status": "not_started", "sessions": 0},
              "forward_decision": forward_decision(book, forward, args.end),
              "validation": validation, "forward_common_start": "passed", "input_sha256": hashes}
    suffix = "" if args.end == PROTOCOL["historical_end_us"] else "_"+args.end
    (HERE / ("results"+suffix+".json")).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines = ["# 7종목 집중도 검토 — 2026-11-05", "", "현행1+후보3. 아래는 이미 본 2026 자료의 회고 진단이며 새 독립 검증이 아니다.",
             "주 후보 cap20은 결과 확인 전에 고정했다. 운영 규칙은 유지한다.", "",
             "| 구간 | 설정 | 원화 수익 | 최대 낙폭 | 평균 투자비중 |", "|---|---|---:|---:|---:|"]
    for label, period in periods.items():
        for name,r in period.get("rows",{}).items():
            lines.append(f"| {label} | {name} | {r['return_pct']:+.2f}% | {r['mdd_pct']:.2f}% | {r['mean_open_exposure_pct']:.1f}% |")
    lines += ["", "수익·낙폭은 구간 직전 종가(연초는 최초 투자금)를 포함한다. 종목 선택·가격·상한 계산은 기존 데스크 엔진을 재사용했다.",
              "각 PDF는 거래일 이전 것만, 순위는 전일 수정종가만 사용. 시장 시가에 교체하고 종가 원화 평가. 수수료0, 호가·환전·세금·현금이자 제외.",
              "raw_ceiling25는 기존 상한 배분과 원래 TIME 비례 비중의 종목별 최솟값만 투자한다. 단순 균등비중 또는 최적 무재배분 해가 아니다.",
              "현금 효과 대조(달러)는 results.json의 usd_exposure_control에 별도 보존한다. 후보의 실제 일별 시가 노출을 사후에 쓴 근사 진단이며 야간 비중 변화를 완전히 맞추지 못한다. 사전 매매 전략이 아니다.",
              "2025는 평가에 쓰지 않았고 2025-06 이후 가격은 120일 순위 준비에만 썼다. 현재 섹터 분류의 사후성과 누적 선택편향이 남는다.",
              "", "## 진단 결론", "",
              "종목 상한 축소는 10월 손실을 줄였지만, 1~5월 수익도 깎았다. 6월 이후 전체 원화 최대 낙폭 개선은 작고, 현금 대조는 오히려 낙폭이 깊었다. 즉시 운영 변경을 뒷받침하지 않는다.",
              "15% 후보가 최근 구간에 더 나았다는 이유로 주 후보를 바꾸지 않는다. 손실 집중 사실과 상한 축소의 유효성은 구분한다.",
              "", "## 앞으로 확인", "", f"11/9 이전에는 모든 후보를 현행으로 동일 재생하고 11/9 시가부터 분기한다. 판정 상태: {result['forward_decision']['status']}, 누적 {result['forward_decision']['sessions']}거래일.",
              "20일은 중간 관찰, 60일은 cap20의 수익이 현행 이상이고 최대 낙폭이 더 얕은지 판정한다. 후보를 과거 성과 순위로 교체하지 않는다.",
              "자동 예약·실계좌 변경 없이, 가격 캐시 갱신 뒤 아래 명령으로 실제 쌓인 날짜까지만 재계산한다.",
              "```", "py data/derived/analysis/concentration_review_20261105/review.py --end YYYY-MM-DD --price-dir data/derived/desk", "```",
              "", f"검증: {validation}; 전방 공통 출발 동일성 통과. 입력 해시는 results JSON에 보존."]
    (HERE / ("report"+suffix+".md")).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"periods":periods,"forward":result['forward'],"validation":validation},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
