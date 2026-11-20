"""Measure §3.1/§3.2 on the 2026-11-21 weekly rule and write results/ and report.md.

py v1121_report.py   (offline; uses the frozen research inputs and results/issuer_closes.csv)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v1121_ledgers as L  # noqa: E402
import v1121_extend as X  # noqa: E402
import v1121_etf as ETF  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
OLD = ETF.OLD
END = X.MARK
PERIODS = {"ytd": ("2026-01-01", END), "h1": ("2026-01-01", "2026-06-30"), "h2": ("2026-07-01", END),
           "recent": ("2026-09-01", END), "q4": ("2026-10-01", END)}
SERIES = ("krw10", "usd10", "krw", "usd", "krw25")
ORDER = ["switch", "trend", "original", "exit", "soxx", "base"]
SHORT = {"switch": "§3.1 전환(재무 강화 FINVIZ)", "trend": "§3.2 30·40·175 FINVIZ", "original": "원본 FINVIZ 전환",
         "exit": "대피 때 달러 현금", "soxx": "대피 때 SOXX 보유", "base": "주간 모델만"}
EXEC = {"timely": "정시", "lag1": "SOXX 전환·FINVIZ 하루 지연", "lag1_all": "주간 포함 전부 하루 지연"}


def window(curve, start, end):
    ix = np.flatnonzero((curve.index >= start) & (curve.index <= end))
    initial = curve.iat[ix[0] - 1] if ix[0] else 1.0
    values = np.r_[initial, curve.iloc[ix].to_numpy(float)]
    return values[-1] / values[0] - 1, float(np.min(values / np.maximum.accumulate(values) - 1))


def performance(builds):
    rows = []
    for rule, built in builds.items():
        for execution, frames in built.ledgers.items():
            for name, frame in frames.items():
                for series in SERIES:
                    if series not in frame:
                        continue
                    for period, (a, b) in PERIODS.items():
                        ret, dd = window(frame[series], a, b)
                        rows.append(dict(rule=rule, execution=execution, strategy=name, series=series,
                                         period=period, ret=ret, mdd=dd))
    return pd.DataFrame(rows)


def spells(built, execution, series="krw10"):
    lag = 0 if execution == "timely" else 1
    gate = built.gate_days[lag]
    frames = built.ledgers[execution]
    days = built.days
    starts = np.flatnonzero(~gate & np.r_[True, gate[:-1]])
    ends = np.flatnonzero(~gate & np.r_[gate[1:], True])
    rows = []
    base = frames["base"][series].to_numpy(float)
    for n, (first, end) in enumerate(zip(starts, ends), 1):
        is_open = end == len(days) - 1
        mark = end if is_open else end + 1  # through the close after the desk is restored
        prev = lambda c: c[first - 1] if first else 1.0  # noqa: E731
        base_ratio = base[mark] / prev(base)
        for name in ORDER[:-1]:
            curve = frames[name][series].to_numpy(float)
            ratio = curve[mark] / prev(curve)
            rows.append(dict(rule=built.rule, execution=execution, series=series, strategy=name, episode=n,
                             exit_open=days[first], last_off_open=days[end], end_close=days[mark],
                             sessions_off=int(end - first + 1), status="open" if is_open else "closed",
                             strategy_return=ratio - 1, base_return=base_ratio - 1,
                             relative=ratio / base_ratio - 1))
    out = pd.DataFrame(rows)
    for name in ORDER[:-1]:
        total = (1 + out[out.strategy.eq(name)].relative).prod()
        np.testing.assert_allclose(total, frames[name][series].iat[-1] / base[-1], rtol=1e-9)
    return out


def paired_metrics(built, kr, labels):
    """desk_etf_validation_20261120 method: KRW, US anchors 12/31 and 6/30, first KRX close after."""
    us = pd.DataFrame({name: built.ledgers["timely"][name].krw10 for name in labels if name in built.ledgers["timely"]})
    us.index = pd.to_datetime(us.index)
    us = pd.concat([pd.DataFrame(1.0, index=pd.to_datetime(["2025-12-31"]), columns=us.columns), us])
    us_dates, krx = ETF.pair(kr, us.index)
    paired = us.loc[us_dates].copy()
    paired.index.name = "us_date"
    paired.insert(0, "krx_date", krx)
    for key in kr:
        paired[key] = kr[key].loc[krx].to_numpy()
    end_us, end_kr = paired.index[-1], paired.krx_date.iat[-1]
    rows, detail = [], []
    anchors = {"ytd": "2025-12-31", "h2": "2026-06-30", "recent": "2026-08-31"}
    for key, label in labels.items():
        row = {"label": label}
        for period, anchor in anchors.items():
            a = pd.Timestamp(anchor)
            start_kr = paired.at[a, "krx_date"]
            s = kr.loc[start_kr:end_kr, key] if key in kr else us.loc[a:end_us, key]
            ret = float(s.iat[-1] / s.iat[0] - 1)
            assert abs(ret - (paired[key].iat[-1] / paired.at[a, key] - 1)) < 1e-10
            dd = float((s / s.cummax() - 1).min())
            row[period] = ret
            row[f"{period}_mdd"] = dd
            detail.append(dict(key=key, period=period, return_krw=ret, max_drawdown=dd, us_anchor=anchor,
                               krx_anchor=str(start_kr.date()), observations=len(s)))
        rows.append(row)
    return dict(through_us=str(end_us.date()), through_krx=str(end_kr.date()), rows=rows), paired, pd.DataFrame(detail)


def rolling(builds, rule="res60"):
    """Moving forward check: each month (quarter) hold the strategy that did best the month (quarter) before."""
    frames = builds[rule].ledgers["timely"]
    curves = pd.DataFrame({k: frames[k].krw10 for k in ORDER})
    curves.index = pd.to_datetime(curves.index)
    out = {}
    for label, freq in [("monthly", "M"), ("quarterly", "Q")]:
        ends = curves.groupby(curves.index.to_period(freq)).tail(1)
        rets = ends.pct_change()
        rets.iloc[0] = ends.iloc[0] - 1
        picks, chain = [], 1.0
        for i in range(1, len(rets)):
            previous = rets.iloc[i - 1]
            pick = max(ORDER[::-1], key=lambda k: (previous[k], k == "base"))  # tie -> base
            chain *= 1 + rets.iloc[i][pick]
            picks.append(dict(period=str(rets.index[i].to_period(freq)), pick=pick, ret=rets.iloc[i][pick],
                              base=rets.iloc[i]["base"], switch=rets.iloc[i]["switch"]))
        first = rets.index[1]
        fixed = {k: float(ends[k].iat[-1] / ends[k].iat[0]) - 1 for k in ("base", "switch")}
        out[label] = dict(picks=picks, chained=chain - 1, from_=str(first.date()), fixed_same_span=fixed)
    return out


def fmt(x, digits=1):
    return f"{x * 100:+.{digits}f}%".replace("-", "−")


def pp(x):
    return f"{x * 100:+.1f}%p".replace("-", "−")


def main():
    builds = {rule: L.build(rule) for rule in L.RULES}
    checks = L.reproduce_old(builds["old120"])  # 1/2~11/18 ledgers, before the 11/19 mark is added
    builds = {rule: X.extend(b) for rule, b in builds.items()}
    checks.update(X.reproduce_forward_mark(builds["old120"]))
    # The SOXX gate and the three monthly FINVIZ baskets do not depend on the weekly rule.
    for lag in (0, 1):
        np.testing.assert_array_equal(builds["old120"].gate_days[lag], builds["res60"].gate_days[lag])
    for name in ("switch", "trend", "original"):
        a = builds["old120"].ledgers["timely"][name]
        b = builds["res60"].ledgers["timely"][name]
        off = ~builds["res60"].gate_days[0]
        same = (a.holdings.fillna("").to_numpy() == b.holdings.fillna("").to_numpy())[off]
        checks[f"{name}: off-gate holdings identical across rules (share of off sessions)"] = float(same.mean())

    perf = performance(builds)
    RESULTS.mkdir(exist_ok=True)
    perf.to_csv(RESULTS / "performance.csv", index=False, float_format="%.12g", lineterminator="\n")
    episodes = pd.concat([spells(b, e, s) for b in builds.values() for e in L.EXECUTIONS for s in ("krw10", "usd10")])
    episodes.to_csv(RESULTS / "episodes.csv", index=False, float_format="%.12g", lineterminator="\n")
    daily = []
    for rule, b in builds.items():
        for e, frames in b.ledgers.items():
            for name, f in frames.items():
                cols = [c for c in ("krw", "krw10", "krw25", "usd", "usd10", "exposure", "turn", "gate_on", "holdings")
                        if c in f]
                daily.append(f[cols].assign(rule=rule, execution=e, strategy=name).rename_axis("date").reset_index())
    pd.concat(daily).to_csv(RESULTS / "daily.csv", index=False, float_format="%.11g", lineterminator="\n")

    kr = ETF.closes()
    # Same code, old rule, old KRX endpoint: must give the frozen 11/20 metrics.json exactly.
    old_labels = {"switch": "재무 강화 FINVIZ", "original": "기존 FINVIZ", "base": "기존 데스크", "time": "TIME",
                  "koact": "KoAct"}
    old_metrics, _, _ = paired_metrics(builds["old120"], kr.loc[:"2026-11-19"], old_labels)
    frozen = json.loads((OLD / "metrics.json").read_text(encoding="utf-8"))
    assert old_metrics["through_us"] == frozen["through_us"] and old_metrics["through_krx"] == frozen["through_krx"]
    assert len(frozen["rows"]) == len(old_metrics["rows"])  # same five rows in the same order (labels may be renamed)
    for mine, theirs in zip(old_metrics["rows"], frozen["rows"]):
        np.testing.assert_allclose([mine["ytd"], mine["h2"]], [theirs["ytd"], theirs["h2"]], atol=1e-12)
    checks["desk_etf_validation_20261120/metrics.json rows (max abs diff)"] = float(max(
        abs(m[p] - t[p]) for m, t in zip(old_metrics["rows"], frozen["rows"]) for p in ("ytd", "h2")))

    labels = {"switch": "재무 강화 FINVIZ 전환", "trend": "30·40·175 FINVIZ 후보", "original": "기존 FINVIZ 전환",
              "exit": "대피 때 달러 현금", "base": "주간 모델만", "time": "TIME", "koact": "KoAct"}
    metrics, paired, detail = paired_metrics(builds["res60"], kr, labels)
    metrics = {"through_us": metrics["through_us"], "through_krx": metrics["through_krx"],
               "note": "2026-11-21 60일 잔차 주간 규칙 기준", "rows": metrics["rows"]}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paired.to_csv(RESULTS / "paired_daily.csv", float_format="%.12g", lineterminator="\n")
    detail.to_csv(RESULTS / "etf_paired_performance.csv", index=False, float_format="%.12g", lineterminator="\n")
    roll = rolling(builds)
    (RESULTS / "rolling.json").write_text(json.dumps(roll, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "reproduction.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(builds, perf, episodes, metrics, roll, checks)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


def write_report(builds, perf, episodes, metrics, roll, checks):
    P = perf.set_index(["rule", "execution", "strategy", "series", "period"])

    def v(rule, strategy, period, series="krw10", execution="timely", field="ret"):
        return P.at[(rule, execution, strategy, series, period), field]

    E = episodes
    def ep(rule, strategy, execution="timely", series="krw10"):
        return E[(E.rule == rule) & (E.strategy == strategy) & (E.execution == execution) & (E.series == series)]

    new_sw, new_ex = ep("res60", "switch"), ep("res60", "exit")
    closed = lambda d: (1 + d[d.status == "closed"].relative).prod() - 1  # noqa: E731
    opened = lambda d: (1 + d[d.status == "open"].relative).prod() - 1  # noqa: E731
    sw_closed, sw_open = closed(new_sw), opened(new_sw)
    old_sw = ep("old120", "switch")
    gain_ytd = (1 + v("res60", "switch", "ytd")) / (1 + v("res60", "base", "ytd")) - 1
    wins_h2 = v("res60", "switch", "h2") > v("res60", "base", "h2")
    wins_recent = v("res60", "switch", "recent") > v("res60", "base", "recent")
    lead = (f"**새 주간 규칙(60일 잔차)에서도 §3.1 전환이 주간 모델만보다 나았다(원화·편도 10bp, 연초 이후 "
            f"{fmt(v('res60', 'switch', 'ytd'))} 대 {fmt(v('res60', 'base', 'ytd'))}). 그러나 이득은 아직 열린 "
            f"10/13 대피 하나에서만 나온다. 끝난 대피 {int((new_sw.status == 'closed').sum())}번은 합쳐 "
            f"{fmt(sw_closed)}이고, 열린 대피는 {fmt(sw_open)}이다.**")
    # Sentences below state these orderings; stop instead of printing a false statement.
    for execution, series in [("timely", "krw25"), ("lag1", "krw10"), ("lag1_all", "krw10")]:
        assert v("res60", "switch", "ytd", series, execution) > v("res60", "base", "ytd", series, execution)
    recent_gap = np.log((1 + v("res60", "switch", "recent")) / (1 + v("res60", "base", "recent")))
    assert np.log(1 + sw_open) / recent_gap > 0.8
    trend_gap = {r: (1 + v(r, "trend", "ytd")) / (1 + v(r, "switch", "ytd")) for r in L.RULES}
    assert abs(trend_gap["res60"] - trend_gap["old120"]) < 0.005
    detail_lead = (f"최근 구간(9/1 이후)은 전환 {fmt(v('res60', 'switch', 'recent'))}, 주간 모델만 "
                   f"{fmt(v('res60', 'base', 'recent'))}이다. 이 차이도 거의 열린 대피 기간에서 생겼다. 7월 이후는 "
                   f"{fmt(v('res60', 'switch', 'h2'))} 대 {fmt(v('res60', 'base', 'h2'))}다. 120일 규칙 때 전환 이득은 "
                   f"상대 {fmt((1 + v('old120', 'switch', 'ytd')) / (1 + v('old120', 'base', 'ytd')) - 1)}였다. 지금은 "
                   f"{fmt(gain_ytd)}로 줄었다. 줄어든 몫은 모두 주간 모델 교체에서 나왔다(§3). §3.2 30·40·175는 §3.1보다 "
                   f"상대 {fmt((1 + v('res60', 'trend', 'ytd')) / (1 + v('res60', 'switch', 'ytd')) - 1)} 높았다. 그 차이도 "
                   "대피 기간 FINVIZ 바구니에서만 생기므로 규칙 교체의 영향을 받지 않았다.")
    lines = ["# 새 주간 규칙에서 SOXX 100일선 전환(§3.1)·30·40·175(§3.2) 재측정", "", lead, "", detail_lead, ""]
    lines += [f"비교 기간은 2026-01-02~{END} 미국 정규장 {len(builds['res60'].days)}거래일이다. 연구 시점은 "
              "뉴욕 2026-11-20 12시라 11/20 정규장이 아직 끝나지 않았고, 마지막 완료 세션은 11/19다. 주간 모델은 지금 "
              "데스크 규칙이다. `desk/strategies.py`의 `run_model`이 `desk/config.py` RULE(60일 잔차 모멘텀, 7종목, "
              "버퍼 5, 상한 25%·60%·반도체 50%)로 내는 경로를 그대로 썼다. 전환·바구니·원장·비용·환율은 앞선 연구 "
              "코드를 고치지 않고 불러 썼다. 연구 코드는 finviz_trends·finviz_recipes·seven_sleeves·desk_defense다. "
              "11/19는 결정이 없는 보유일이다. 주 첫날·월 첫날이 아니고 SOXX 상태도 그대로임을 검사했다. 그래서 "
              "11/18 마감 보유를 고정 시세(finviz_recipes extension_quotes.csv)로 평가했다.", ""]
    lines += ["## 1. 결과 (원화, 편도 10bp, 미국 달력)", "",
              "| 전략 | 연초 이후 | 상반기 | 7월 이후 | 9/1 이후 | 10/1 이후 | 연중 최대낙폭 | 7월 이후 최대낙폭 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ORDER:
        lines.append("| " + " | ".join([SHORT[s], *(fmt(v("res60", s, p)) for p in ("ytd", "h1", "h2", "recent", "q4")),
                                          fmt(v("res60", s, "ytd", field="mdd")), fmt(v("res60", s, "h2", field="mdd"))]) + " |")
    lines += ["", "같은 표를 달러로 보면 다음과 같다. 원화 표는 매일 KRW=X로 환산했다.", "",
              "| 전략 | 연초 이후 | 7월 이후 | 9/1 이후 | 연중 최대낙폭 |", "|---|---:|---:|---:|---:|"]
    for s in ORDER:
        lines.append("| " + " | ".join([SHORT[s], *(fmt(v("res60", s, p, "usd10")) for p in ("ytd", "h2", "recent")),
                                          fmt(v("res60", s, "ytd", "usd10", field="mdd"))]) + " |")
    lines += ["", "### 체결 지연·비용 민감도 (원화, 연초 이후 / 9/1 이후)", "",
              "| 전략 | 정시 0bp | 정시 10bp | 정시 25bp | 전환·FINVIZ 하루 지연 10bp | 주간까지 전부 하루 지연 10bp |",
              "|---|---:|---:|---:|---:|---:|"]
    for s in ORDER:
        cells = [f"{fmt(v('res60', s, 'ytd', ser))} / {fmt(v('res60', s, 'recent', ser))}" for ser in ("krw", "krw10", "krw25")]
        cells += [f"{fmt(v('res60', s, 'ytd', execution=e))} / {fmt(v('res60', s, 'recent', execution=e))}"
                  for e in ("lag1", "lag1_all")]
        lines.append("| " + " | ".join([SHORT[s], *cells]) + " |")
    lines += ["", f"전환과 주간 모델의 격차는 연초 이후 상대 {fmt(gain_ytd)}다. 7월 이후 전환이 "
              f"{'앞섰다' if wins_h2 else '뒤졌다'}({fmt(v('res60', 'switch', 'h2'))} 대 {fmt(v('res60', 'base', 'h2'))}). "
              f"9/1 이후도 {'앞섰다' if wins_recent else '뒤졌다'}({fmt(v('res60', 'switch', 'recent'))} 대 "
              f"{fmt(v('res60', 'base', 'recent'))}). SOXX 전환과 FINVIZ 체결을 하루 늦추면 전환은 연초 이후 "
              f"{fmt(v('res60', 'switch', 'ytd', execution='lag1'))}, 9/1 이후 "
              f"{fmt(v('res60', 'switch', 'recent', execution='lag1'))}이다. 주간 모델만은 각각 "
              f"{fmt(v('res60', 'base', 'ytd', execution='lag1'))}, {fmt(v('res60', 'base', 'recent', execution='lag1'))}다. "
              "주간 결정까지 하루 늦추면 주간 모델만 "
              f"{fmt(v('res60', 'base', 'ytd', execution='lag1_all'))}, 전환 {fmt(v('res60', 'switch', 'ytd', execution='lag1_all'))}이다. "
              "지연에서도 순서는 같았다. 편도 0.25%에서도 전환이 앞섰다.", ""]
    lines += ["## 2. 대피 회차별 상대 수익 (주간 모델만 대비)", "",
              "구간은 이탈 다음 시가부터 주간 모델 복귀일 종가까지다. 복귀 시가의 갭과 비용을 포함한다. 열린 회차는 "
              f"{END} 종가까지 평가했다. 각 회차의 (1+상대 수익)을 곱하면 연말 상대 자산과 같다(검사함).", "",
              "| 회차 | 이탈 시가 | 마지막 이탈일 | 평가 종가 | 이탈 거래일 | 상태 | 주간 모델 | §3.1 전환 | 전환 상대 | 달러 현금 상대 | 30·40·175 상대 | SOXX 보유 상대 | 120일 규칙 때 전환 상대 |",
              "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in new_sw.iterrows():
        n = r.episode
        get = lambda name, rule="res60": ep(rule, name).set_index("episode").at[n, "relative"]  # noqa: E731
        lines.append("| " + " | ".join([str(n), r.exit_open, r.last_off_open, r.end_close, str(r.sessions_off),
                                          "열림" if r.status == "open" else "끝남", fmt(r.base_return),
                                          fmt(r.strategy_return), fmt(r.relative), fmt(get("exit")), fmt(get("trend")),
                                          fmt(get("soxx")), fmt(get("switch", "old120"))]) + " |")
    rows = []
    for name in ("switch", "exit", "trend", "original", "soxx"):
        d_new, d_old = ep("res60", name), ep("old120", name)
        rows.append("| " + " | ".join([SHORT[name], fmt(closed(d_new)), fmt(opened(d_new)),
                                          fmt((1 + d_new.relative).prod() - 1), fmt(closed(d_old)), fmt(opened(d_old)),
                                          fmt((1 + d_old.relative).prod() - 1)]) + " |")
    lines += ["", "| 대피 때 대안 | 끝난 회차 합(60일 잔차) | 열린 회차(60일 잔차) | 전체(60일 잔차) | 끝난 회차 합(120일) | 열린 회차(120일) | 전체(120일) |",
              "|---|---:|---:|---:|---:|---:|---:|", *rows, ""]
    lag_closed = closed(ep("res60", "switch", "lag1"))
    lines += [f"끝난 대피만 보면 전환은 주간 모델보다 {fmt(sw_closed)} 뒤졌다. 120일 규칙 때도 "
              f"{fmt(closed(old_sw))}였다. 하루 지연 체결에서는 끝난 회차 합이 {fmt(lag_closed)}다. 연초 이후 우위는 "
              "모두 10/13에 시작해 아직 열린 대피에서 나온다. 이 대피의 결과는 SOXX가 100일선을 회복하는 날 확정된다. "
              "그때까지 FINVIZ 바구니와 주간 모델의 차이가 줄면 우위도 줄어든다. [회차 원장](results/episodes.csv)", ""]
    inside = {r: (1 + ep(r, "switch").base_return).prod() for r in L.RULES}
    outside = {r: (1 + v(r, "base", "ytd")) / inside[r] for r in L.RULES}
    base_ep = {r: ep(r, "switch").set_index("episode").base_return.to_dict() for r in L.RULES}
    ob, nb = v("old120", "base", "ytd"), v("res60", "base", "ytd")
    os_, ns = v("old120", "switch", "ytd"), v("res60", "switch", "ytd")
    lines += ["## 3. 규칙 교체 때문인가: 같은 코드로 120일 규칙 재현", "",
              "같은 코드로 RULE만 교체 전 값(120일 수익률, 버퍼 5, 상한 25%·60%·반도체 50%)으로 바꿨다. 그러자 "
              "11/20 연구 파일의 일별 원장이 그대로 재현됐다. 비교한 파일은 finviz_recipes selected_daily.csv의 "
              "base·exit·balance·original 일별 원화·달러·비용·회전율이다. finviz_trends finalists.csv의 "
              "30·40·175와 현행 전환(정시·하루 지연)도 맞췄다. forward_mark.csv의 11/19 평가와 "
              "desk_etf_validation_20261120/metrics.json의 다섯 줄도 같았다. 최대 오차는 "
              f"{max(checks[k] for k in checks if 'identical' not in k):.1e}이다. SOXX 신호와 세 FINVIZ 바구니는 "
              "두 규칙에서 같다. 대피 중 보유 종목도 모든 날 같음을 확인했다. 그러므로 아래 차이는 주간 모델 교체에서만 "
              "나온다. [재현 점검](results/reproduction.json)", "",
              "| 원화 10bp, 1/2~11/19 | 120일 규칙 | 60일 잔차 |", "|---|---:|---:|",
              f"| 주간 모델만 연초 이후 | {fmt(ob)} | {fmt(nb)} |",
              f"| §3.1 전환 연초 이후 | {fmt(os_)} | {fmt(ns)} |",
              f"| 전환 − 주간 모델 (상대) | {fmt((1+os_)/(1+ob)-1)} | {fmt((1+ns)/(1+nb)-1)} |",
              f"| 주간 모델만 7월 이후 | {fmt(v('old120', 'base', 'h2'))} | {fmt(v('res60', 'base', 'h2'))} |",
              f"| §3.1 전환 7월 이후 | {fmt(v('old120', 'switch', 'h2'))} | {fmt(v('res60', 'switch', 'h2'))} |",
              f"| §3.2 30·40·175 연초 이후 | {fmt(v('old120', 'trend', 'ytd'))} | {fmt(v('res60', 'trend', 'ytd'))} |",
              f"| §3.2 30·40·175 − §3.1 (상대) | {fmt((1+v('old120','trend','ytd'))/(1+os_)-1)} | {fmt((1+v('res60','trend','ytd'))/(1+ns)-1)} |",
              f"| 주간 모델만 연중 최대낙폭 | {fmt(v('old120', 'base', 'ytd', field='mdd'))} | {fmt(v('res60', 'base', 'ytd', field='mdd'))} |",
              f"| §3.1 전환 연중 최대낙폭 | {fmt(v('old120', 'switch', 'ytd', field='mdd'))} | {fmt(v('res60', 'switch', 'ytd', field='mdd'))} |",
              "", f"주간 모델만의 자산을 대피 구간 안과 밖으로 나눴다. 대피 구간 밖에서는 120일 규칙 대비 "
              f"{fmt(outside['res60'] / outside['old120'] - 1)} 더 벌었다. 대피 구간 안에서는 "
              f"{fmt(inside['old120'] - 1)} 대신 {fmt(inside['res60'] - 1)}였다. 대피 중 전환 쪽 보유 종목은 두 규칙에서 "
              "같다. 그래도 회차 수익이 다른 것은 이탈 시가까지의 밤사이 갭과 복귀일 하루를 주간 모델로 보내기 때문이다(4회차는 8/31 갭과 9/22 하루에서만 갈렸다). "
              "그래서 전환의 상대 이득을 바꾼 것은 같은 기간 주간 모델 자체의 성적이다. 4회차(8/31~9/22)는 "
              f"{fmt(base_ep['old120'][4])}에서 {fmt(base_ep['res60'][4])}로, 열린 5회차는 {fmt(base_ep['old120'][5])}에서 "
              f"{fmt(base_ep['res60'][5])}로 바뀌었다.", ""]
    lines += ["## 4. 두 ETF와 원화 비교 (desk_etf_validation_20261120 방식)", "",
              f"미국 {metrics['through_us']} 종가를 그 뒤 첫 한국 {metrics['through_krx']} 종가와 맞췄다. 미국 12/31·6/30·8/31 "
              "마감 뒤 첫 한국 종가(1/2·7/1·9/1)가 시작점이다. ETF는 운용사 시장 종가로 계산했다. 11/20 종가는 "
              "네이버 KRX 일봉으로 채웠고, 앞선 218거래일이 운용사 값과 모두 같음을 확인했다. TIME은 운용사 표에서도 "
              "같았다.", "",
              "| 전략·ETF | 연초 이후 | 7월 이후 | 9/1 이후 | 연중 최대낙폭 | 7월 이후 최대낙폭 |", "|---|---:|---:|---:|---:|---:|"]
    for r in metrics["rows"]:
        lines.append("| " + " | ".join([r["label"], fmt(r["ytd"], 2), fmt(r["h2"], 2), fmt(r["recent"], 2),
                                          fmt(r["ytd_mdd"], 2), fmt(r["h2_mdd"], 2)]) + " |")
    m = roll["monthly"]
    q = roll["quarterly"]
    picks = ", ".join(f"{p['period']} {SHORT[p['pick']]}" for p in m["picks"][-4:])
    lines += ["", "## 5. 이동 전방 점검", "",
              "매달 직전 달 원화 10bp 수익이 가장 좋았던 대안(여섯 개 중)을 다음 달에 쓰는 방식으로 점검했다. "
              f"2월~11/19 연결 수익은 {fmt(m['chained'])}다. 같은 구간에 주간 모델만 쓰면 "
              f"{fmt(m['fixed_same_span']['base'])}, §3.1 전환을 고정하면 {fmt(m['fixed_same_span']['switch'])}다. "
              f"분기 단위로는 2분기~11/19 {fmt(q['chained'])}이고, 고정 시 주간 모델 "
              f"{fmt(q['fixed_same_span']['base'])}, 전환 {fmt(q['fixed_same_span']['switch'])}다. 대안 사이를 옮길 때 드는 "
              f"매매비용은 넣지 않았다. 최근 넉 달의 선택은 {picks}였다. 상반기 이탈은 3/30~31 이틀뿐이라 앞 구간에서 "
              "고르고 뒤 구간에서 확인하는 방식이 이 전환에는 거의 작동하지 않는다. [선택 기록](results/rolling.json)", ""]
    lines += ["## 6. 언제 알 수 있었나, 무엇을 몇 개 봤나", "",
              "- SOXX 신호: 직전 확정 종가가 그날 포함 100거래일 단순평균 이상인지 본다. 다음 정규장 시가에 체결한다. "
              "FINVIZ 월초 명단은 월말 종가까지의 재무(공시일 다음 날부터)·기관 지분·거래량으로 만든다. 주간 순위의 "
              "잔차 베타·모멘텀은 결정일 전날 종가까지 쓴다. 사후 값은 쓰지 않았다.",
              "- 이번 재측정은 새 탐색이 아니다. 이미 고정된 두 규칙을 합쳤다. 주간 잔차 60일은 10,004개 설정에서 "
              "골랐고(momentum_variants_20261120), FINVIZ 재무 강화식은 170개 결과에서, 30·40·175는 89,856개 조합에서 "
              "모두 2026 자료를 본 뒤 골랐다. 여기서 비교한 것은 대안 6개 × 체결 3가지 × 비용 3단계 × 규칙 2개뿐이다. "
              "두 구성 요소의 선택편향은 그대로 남는다. 숫자는 확인이 아니라 같은 표본 안의 서술이다.",
              f"- 2026 이탈은 {len(new_sw)}번, {int(new_sw.sessions_off.sum())}거래일이다. 상반기 이탈은 "
              f"{int(new_sw[new_sw.exit_open < '2026-07-01'].sessions_off.sum())}일뿐이다. 열린 10/13 대피가 "
              f"{int(new_sw[new_sw.status == 'open'].sessions_off.sum())}거래일을 차지한다.",
              "- 장세 위험 한 줄: 2025 모방 바구니에서는 SOXX 대피가 +14.5%를 −36.0%로 바꿨다(desk_defense_check_20261119).",
              "- 모의 가격 수익이다. 배당·세금·현금 이자는 뺐고, 빈자리는 달러 현금이다. 실제 계좌 수익과 다르다.", ""]
    lines += ["## 7. 다시 돌리기와 파일", "",
              "`py code/v1121_report.py`는 고정 입력에서 오프라인으로 모든 결과와 이 보고서를 다시 만든다. "
              "`py code/v1121_etf.py --refresh`는 ETF 종가를 다시 받는다. `py code/v1121_ledgers.py`는 120일 재현만 "
              "점검한다. desk/config.py RULE이 2026-11-21 값과 다르면 실행을 멈춘다.", "",
              "[성과 표](results/performance.csv) · [회차](results/episodes.csv) · [일별 원장](results/daily.csv) · "
              "[ETF 날짜 대응](results/paired_daily.csv) · [ETF 비교 지표](results/metrics.json) · "
              "[ETF 종가](results/issuer_closes.csv)"]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
