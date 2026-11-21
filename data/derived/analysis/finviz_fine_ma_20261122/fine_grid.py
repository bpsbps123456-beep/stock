"""FINVIZ 30·40·175 주변을 1일 간격으로 다시 잰다(2026-11-22 사용자 "29일 39일 174일선으로 하면 되는 거 아니야?").

finviz_trends_20261120의 입력·엔진·체결·비용·SOXX 100일선 전환을 고치지 않고 불러 쓴다. 바꾸는 것은 종목 이평선
길이뿐이다: 단기 25~35, 중기 36~45, 장기 165~185(모두 1일 간격, 11 × 10 × 21 = 2,310개). 나머지는 §3.2 후보와
같다(기본 추세 = 단기 > 중기, 종가 > 중기·장기; 유동비율 > 1.5, 0 ≤ 부채/자본 < 1, 상대거래량 > 1, RSI > 40;
상대거래량 순 7자리; 월초 교체). 비교 기준은 현행 §3.1(20·50·200, 부채/자본 < 0.5)이다.
엔진의 주간 데스크 구간은 seven_sleeves baselines.desk_targets가 지금 데스크 규칙(60일 잔차)으로 낸다. 그래서
현행·30-40-175 두 줄은 2026-11-21 재측정 원장(desk_etf_validation_20261121 paired_daily.csv의 switch·trend,
11/18 미국 날)과 같아야 하고, 실행 때 그것을 확인한다.

실행: py data/derived/analysis/finviz_fine_ma_20261122/fine_grid.py   (가격은 finviz_rotation 캐시를 읽는다)
"""
from itertools import product
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRENDS = HERE.parent / "finviz_trends_20261120"
sys.path.insert(0, str(TRENDS))
import features  # noqa: E402
from gates import load_gates, load_inputs  # noqa: E402  (gates keeps its own SOXX periods)
import importlib.util  # noqa: E402


def _trends(name):
    """Load a finviz_trends module by file: seven_sleeves/recipes put same-named modules on sys.path."""
    spec = importlib.util.spec_from_file_location(f"trends_{name}", TRENDS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FastEngine = _trends("engine").FastEngine
_experiment = _trends("experiment")
WINDOWS, summaries = _experiment.WINDOWS, _experiment.summaries
top_seven_days = _trends("ranking").top_seven_days

SHORT = range(25, 36)
MEDIUM = range(36, 46)
LONG = range(165, 186)
LOOSE = (1.5, 1.0, 1.0, 40.0)    # §3.2 후보
BALANCE = (1.5, 0.5, 1.0, 40.0)  # 현행 §3.1
CASES = [("timely_10bp", 0, .001), ("lag1_10bp", 1, .001)]
SPANS_USED = ["year", "h2", "q4"]
# 2026-11-21 재측정 원장(정시·10bp, 11/18 미국 날 누적)과 같아야 같은 엔진·입력·주간 규칙이다
LEDGER = HERE.parent / "desk_etf_validation_20261121/results/paired_daily.csv"


def calibration():
    row = pd.read_csv(LEDGER).set_index("us_date").loc["2026-11-18"]
    return {("20-50-200", "balance", "timely_10bp"): row.switch - 1, ("30-40-175", "loose", "timely_10bp"): row.trend - 1}


def table_with(periods):
    features.PERIODS = tuple(sorted(periods))  # add_technical reads the module global
    src = load_inputs()
    reviews = features.review_indices(src)
    base = features.source_candidates(src, reviews)
    table = features.add_filing_ratios(features.add_technical(base, src)).sort_values(["trade_open", "symbol"])
    return src, reviews, table


def panels(src, reviews, table, periods):
    dates = np.asarray(reviews, dtype=int)
    day = pd.Index(np.asarray(src.days)[dates]).get_indexer(table.trade_open)
    slot = table.groupby("trade_open").cumcount().to_numpy()
    shape = (len(dates), slot.max() + 1)

    def panel(field):
        values = np.full(shape, np.nan)
        values[day, slot] = table[field]
        return values

    cols = np.full(shape, -1, dtype=np.int16)
    cols[day, slot] = table.col
    p = {name: panel(name) for name in ("close", "rv", "rsi", "current_ratio", "debt_equity")}
    p["sma"] = {n: panel(f"sma{n}") for n in periods}
    order = np.argsort(-p["rv"], axis=1, kind="stable")
    p["order"], p["sorted_cols"] = order, np.take_along_axis(cols, order, axis=1)
    p["dates"] = dates
    return p


def ranks(p, src, short, medium, long, profile):
    c, d, v, r = profile
    sma, close = p["sma"], p["close"]
    mask = (p["current_ratio"] > c) & (p["debt_equity"] >= 0) & (p["debt_equity"] < d) & (p["rv"] > v) & (p["rsi"] > r)
    passed = (sma[short] > sma[medium]) & (close > sma[medium]) & (close > sma[long]) & mask
    return top_seven_days(passed, p["order"], p["sorted_cols"], p["dates"], len(src.days))


def main():
    periods = set(SHORT) | set(MEDIUM) | set(LONG) | {20, 50, 200}
    src, reviews, table = table_with(periods)
    p = panels(src, reviews, table, periods)
    engine = FastEngine(src)
    gates = {0: load_gates(src)["sma100"], 1: load_gates(src, lag=1)["sma100"]}
    days = pd.Index(src.days)
    spans = [(days.searchsorted(WINDOWS[w][0]), days.searchsorted(WINDOWS[w][1], side="right")) for w in SPANS_USED]
    monthly = engine.reviews["monthly1"]
    combos = [("20-50-200", 20, 50, 200, "balance", BALANCE)]
    combos += [(f"{s}-{m}-{l}", s, m, l, "loose", LOOSE) for s, m, l in product(SHORT, MEDIUM, LONG)]
    rows, cache = [], {}
    for name, s, m, l, prof, profile in combos:
        rk = ranks(p, src, s, m, l, profile)[monthly]
        sig = rk.tobytes()
        for case, lag, fee in CASES:
            key = (sig, case)
            if key not in cache:
                path = engine.evaluate(rk, "monthly1", gates[lag], fee=fee, delay=lag)
                cache[key] = summaries(path.nav, path.exposure, spans)[0]
            v = cache[key]
            rows.append(dict(key=name, short=s, medium=m, long=l, profile=prof, test=case, basket=hash(sig),
                             **{f"{w}_{f}": v[i, j] for i, w in enumerate(SPANS_USED) for j, f in enumerate(("ret", "dd"))}))
    out = pd.DataFrame(rows)
    for (name, prof, case), want in calibration().items():
        got = out[(out.key == name) & (out.profile == prof) & (out.test == case)].year_ret.iloc[0]
        assert abs(got - want) < 5e-6, (name, case, got, want)
    out.to_csv(HERE / "fine_grid.csv", index=False, float_format="%.10g")
    summary = {}
    for case, _, _ in CASES:
        g = out[(out.test == case) & (out.profile == "loose")]
        base = out[(out.test == case) & (out.key == "20-50-200")].iloc[0]
        ref = g[g.key == "30-40-175"].iloc[0]
        ask = g[g.key == "29-39-174"].iloc[0]
        five = (g.year_ret > base.year_ret) & (g.h2_ret > base.h2_ret) & (g.h2_dd > base.h2_dd) \
            & (g.q4_ret > base.q4_ret) & (g.q4_dd > base.q4_dd)
        summary[case] = dict(
            n=len(g), distinct_baskets=int(g.basket.nunique()),
            current={w: float(base[w]) for w in ("year_ret", "h2_ret", "q4_ret", "h2_dd")},
            r30_40_175={w: float(ref[w]) for w in ("year_ret", "h2_ret", "q4_ret", "h2_dd")},
            r29_39_174={w: float(ask[w]) for w in ("year_ret", "h2_ret", "q4_ret", "h2_dd")},
            same_basket_29_39_174_as_30_40_175=bool(ask.basket == ref.basket),
            share_beat_current_year=float((g.year_ret > base.year_ret).mean()),
            share_beat_current_h2=float((g.h2_ret > base.h2_ret).mean()),
            share_beat_current_q4=float((g.q4_ret > base.q4_ret).mean()),
            share_beat_current_all_five=float(five.mean()),
            year_ret_quantiles={q: float(g.year_ret.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            q4_ret_quantiles={q: float(g.q4_ret.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            rank_of_30_40_175_year=int((g.year_ret > ref.year_ret).sum()) + 1,
            rank_of_29_39_174_year=int((g.year_ret > ask.year_ret).sum()) + 1)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
