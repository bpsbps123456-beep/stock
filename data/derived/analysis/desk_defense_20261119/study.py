"""Three desk panels and defensive exits, using completed closes and next-open fills."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from desk import strategies as S
from inputs import load_book
from signals import build_signals
from evaluate import evaluate, summarize
from chart import draw
from switch import switched_targets


def market_gates(book, days):
    gates = {"none": np.ones(len(days), dtype=bool)}
    for symbol in ("QQQ", "SOXX"):
        close = book.cls[symbol]
        for n in (10, 20, 50, 100):
            line = close.rolling(n).mean()
            above = (close >= line).where(close.notna() & line.notna())
            for confirmation in ((1, 3) if n in (20, 50) else (1,)):
                # Exit on one close below, re-enter after consecutive recoveries.
                run = above.astype(float).rolling(confirmation).min()
                state = pd.Series(np.nan, index=above.index)
                state.loc[above.eq(False)] = 0
                state.loc[run.eq(1)] = 1
                allowed = state.ffill().shift().reindex(days)
                if allowed.isna().any():
                    raise ValueError(f"Missing market warmup {symbol} SMA{n}")
                gates[f"{symbol}{n}r{confirmation}"] = allowed.astype(bool).to_numpy()
    return gates


def selectors():
    # strategy name, frequency, disposition on failure, eligibility test
    yield "weekly", "week", "rotate", "any"
    yield "fast", "change", "rotate", "any"
    for grade in ("trendA", "giantA", "bothA", "bothB"):
        for freq, method in (("week", "rotate"), ("change", "rotate"), ("change", "cash")):
            yield f"{grade}_{freq}_{method}", freq, method, grade
    for n in (10, 20, 50, 100):
        for method in ("cash", "rotate"):
            yield f"stock{n}_{method}", "change", method, f"sma{n}"


def eligibility(book, days, tickers):
    trend, giant = build_signals(book)
    def previous(frame):
        return frame.reindex(index=book.days, columns=tickers).shift().loc[days]
    trend, giant = previous(trend), previous(giant)
    ok = {"any": np.ones(trend.shape, dtype=bool), "trendA": (trend >= 3).values,
          "giantA": (giant >= 3).values, "bothA": ((trend >= 3) & (giant >= 3)).values,
          "bothB": ((trend >= 2) & (giant >= 2)).values}
    for n in (10, 20, 50, 100):
        close = book.cls.reindex(columns=tickers)
        ok[f"sma{n}"] = previous(close >= close.rolling(n).mean()).to_numpy(bool)
    return ok


def plan(book, days, rankings, col, eligible, freq, method):
    """Publish target weights on weekly dates or name changes, otherwise let them drift."""
    prev, last_pass = set(), set()
    targets = {}
    started = False
    for j, (day, ranked) in enumerate(zip(days, rankings)):
        week = j == 0 or S.iso_week(day) != S.iso_week(days[j-1])
        if not ranked or (freq == "week" and started and not week):
            continue
        # Cash overlay uses the actual weekly desk basket as its reference.
        if method == "cash" and started and not week:
            passed = {k for k in prev if eligible[j, col[k]]}
            if passed == last_pass:
                continue
        else:
            pool = ranked if method == "cash" else [r for r in ranked if eligible[j, col[r["key"]]]]
            selected = set(S.select(pool, prev))
            if method == "rotate" and started and not week and selected == prev:
                continue
            prev = selected
            raw = {r["key"]: r["time_weight"] for r in pool if r["key"] in selected}
            total = sum(raw.values())
            target = S.project_caps({k: v / total for k, v in raw.items()}, book.sector)
            passed = {k for k in prev if eligible[j, col[k]]}
        weights = np.zeros(len(col))
        for k in passed:
            weights[col[k]] = target.get(k, 0.)
        targets[j] = weights
        last_pass, started = passed, True
    return targets


def replay(days, opn, close, fx, targets, gate, scale=1.):
    # Shares marked to actual opens/closes. The shadow basket remains defined in cash
    # periods, so a market recovery can enter at the next open using then-current weights.
    shares, shadow = np.zeros(opn.shape[1]), np.zeros(opn.shape[1])
    cash, reserve, drag = 1., 1., 1.
    rows = []
    for j, day in enumerate(days):
        o, c = opn[j], close[j]
        active = (shares != 0) | (shadow != 0)
        if not np.isfinite(o[active]).all() or not np.isfinite(c[active]).all():
            raise ValueError(f"Held security missing market price on {day}")
        o_mark, c_mark = np.nan_to_num(o), np.nan_to_num(c)
        nav = cash + shares @ o_mark
        shadow_nav = reserve + shadow @ o_mark
        trade = j in targets
        if trade:
            w = targets[j]
            if not (o[w > 0] > 0).all():
                raise ValueError(f"Missing executable open on {day}")
            shadow = np.divide(shadow_nav*w, o, out=np.zeros_like(w), where=w > 0)
            reserve = shadow_nav * (1 - w.sum())
        turn = 0.
        if (gate[j] and trade) or (j > 0 and gate[j] != gate[j-1]):
            weights = scale * (shadow * o_mark / shadow_nav) if gate[j] else np.zeros_like(shares)
            turn = np.abs(weights - shares * o_mark / nav).sum()
            shares = np.divide(nav * weights, o, out=np.zeros_like(weights), where=weights > 0)
            cash = nav * (1 - weights.sum())
            drag *= 1 - .001 * turn
        if not np.isfinite(c[(shares != 0) | (shadow != 0)]).all():
            raise ValueError(f"Missing closing mark on {day}")
        usd = cash + shares @ c_mark
        krw = usd * fx[j]
        rows.append((krw, usd, turn, 1 - cash/nav, krw*drag))
    return pd.DataFrame(rows, index=days, columns=["krw", "usd", "turn", "exposure", "krw10"])


def write_cash_spells(days, baseline, gates):
    """US dollar return forgone between the executable exit and re-entry opens."""
    rows = []
    for market, flags in gates.items():
        start = None
        for j, invested in enumerate(flags):
            if not invested and start is None:
                start = j
            if start is not None and (invested or j == len(days)-1):
                end_value = baseline.value[j] if invested else baseline.close[j]
                rows.append(dict(market=market, exit_open=days[start], end=days[j],
                                 end_mark="reentry_open" if invested else "last_close",
                                 cash_sessions=j-start+int(not invested),
                                 base_usd_return=end_value/baseline.value[start]-1))
                start = None
    pd.DataFrame(rows).to_csv(HERE / "market_cash_spells.csv", index=False)


def diagnostics(days, opn, close, fx, weekly, weekly_trend, curves, selected):
    # Choose a fixed exposure after seeing the holdout solely to explain the result.
    gate = np.ones(len(days), dtype=bool)
    target = curves[selected].loc["2026-07-01":, "exposure"].mean()
    lo, hi = 0., 1.
    for _ in range(35):
        scale = (lo+hi) / 2
        matched = replay(days, opn, close, fx, weekly, gate, scale)
        if matched.loc["2026-07-01":, "exposure"].mean() < target:
            lo = scale
        else:
            hi = scale
    breadth = pd.read_csv(HERE / "breadth.csv", index_col=0).breadth.shift().reindex(days)
    assert breadth.notna().all()
    diag = {"선택 규칙": curves[selected], "같은 평균 비중의 주간 고정 투자": matched,
            "달러 현금만 보유": replay(days, opn, close, fx, {}, gate)}
    for label, targets in [("시장 폭 19% + 기존", weekly), ("시장 폭 19% + 추세 A 이상", weekly_trend)]:
        diag[label] = replay(days, opn, close, fx, targets, (breadth >= 19).values)
    stats = pd.DataFrame({k: summarize(v) for k, v in diag.items()}).T
    stats.to_csv(HERE / "diagnostics.csv")
    lines = ["", "## 추가 진단", "",
             f"상반기 수익 기준 선택과 하반기 평균 주식 비중을 맞추기 위해, 기존 주간 목표를 "
             f"매주 {scale:.2%}로 축소해 같은 시가에 체결했다. 이 고정 배율은 하반기 자료로 맞춘 설명용이다. "
             "거인의 어깨 시장 폭은 전체 AI 종목 중 전고점 10% 안에 있는 비율이다. 19% 미만이면 다음 시가부터 달러 현금, "
             "19% 이상으로 회복하면 다음 시가에 복귀한다. 이 경계는 이미 2~9월 자료로 정해져 두 경로를 앞 구간 선택에서 제외했다.",
             "", "| 진단 | 상반기 원화 수익 | 7월 이후 원화 수익 | 7월 이후 최대낙폭 | 7월 이후 평균 주식 비중 |",
             "|---|---:|---:|---:|---:|"]
    for key, row in stats.iterrows():
        lines.append(f"| {key} | {row.h1_return:+.2%} | {row.h2_return:+.2%} | "
                     f"{row.h2_mdd:+.2%} | {row.h2_exposure:.2%} |")
    lines.extend(["", "![현행과 수익 우선 후보, 고정 절반 대조의 원화 자산](comparison.png)", ""])
    with (HERE / "report.md").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def main():
    book = load_book()
    baseline = S.run_model(book)
    days = baseline.days
    start = book.days.index(days[0])
    col = {key: i for i, key in enumerate(book.keys)}
    tickers = [book.ticker[key] for key in col]
    rankings = [S.momentum_ranking(book, start+j, day)[1] for j, day in enumerate(days)]
    ok = eligibility(book, days, tickers)
    gates = market_gates(book, days)
    opn, close = [frame.loc[days, tickers].to_numpy() for frame in (book.opn, book.cls)]
    fx = book.fx[start:] / book.fx[start-1]
    curves, configs = {}, {}
    for mode, freq, method, condition in selectors():
        targets = plan(book, days, rankings, col, ok[condition], freq, method)
        if mode == "weekly":
            weekly = targets
        if mode == "trendA_week_rotate":
            weekly_trend = targets
        for market, gate in gates.items():
            key = f"{mode}_{market}"
            family = ("개별가격선" if condition.startswith("sma") else
                      "추세와회복" if condition.startswith("both") else
                      "추세" if condition == "trendA" else "회복" if condition == "giantA"
                      else "빠른교체" if mode == "fast" else "기존")
            if market != "none":
                family = "시장이탈" if family == "기존" else family + "+시장이탈"
            elif mode == "weekly":
                key = "base"
            configs[key] = dict(family=family, mode=mode, market=market)
            curves[key] = replay(days, opn, close, fx, targets, gate)
        if freq == "change" and method == "rotate" and condition in ("trendA", "giantA", "bothA", "bothB"):
            for market in ("QQQ20r1", "QQQ50r1", "SOXX20r1", "SOXX50r1"):
                key = f"switch_{condition}_{market}"
                configs[key] = dict(family="이탈시교체", mode=mode, market=market)
                switched = switched_targets(days, opn, weekly, targets, gates[market])
                curves[key] = replay(days, opn, close, fx, switched, gates["none"])
        if mode == "weekly":
            configs["half"] = dict(family="대조", mode="weekly50", market="none")
            curves["half"] = replay(days, opn, close, fx, targets, gates["none"], .5)
    np.testing.assert_allclose(curves["base"].usd, baseline.close, rtol=1e-10)
    np.testing.assert_allclose(curves["base"].krw, S.model_index(book, baseline)/100, rtol=1e-10)
    for name, frame in curves.items():
        assert np.isfinite(frame.values).all() and frame.exposure.between(-1e-9, 1+1e-9).all(), name
    results = evaluate(curves, configs)
    diagnostics(days, opn, close, fx, weekly, weekly_trend, curves, results["global"])
    write_cash_spells(days, baseline, gates)
    draw()


if __name__ == "__main__":
    main()
