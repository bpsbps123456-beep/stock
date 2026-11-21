"""클로드의 40분봉 실제 후보 신호에 SOXX 대피 경고를 겹쳐 재생한다."""
import sys

import numpy as np
import pandas as pd

import study

SOURCE = study.HERE.parent / "exit40_20261122"
sys.path.insert(0, str(SOURCE / "code"))
import grid40

# 1/2~10/14로 정한 원 연구 상위 두 후보. 사후에 가장 잘 맞는 대피 조합을 고르지 않는다.
PRESETS = {
    "rank1": dict(d=10, b=0., a=0., mode="touch", tm="T2"),
    "rank2": dict(d=3, b=.75, a=0., mode="close", tm="T1"),
    "slow": dict(mode="close", tm="T2"),  # 클로드가 40분봉 경고를 얹어 확인 중인 기존 보유 규칙
}
WINDOWS = {"train": grid40.WIN["SEL"], "confirm": grid40.WIN["CONF"],
           "recent": grid40.WIN["Q"], "aug_oct": ("2026-08-03", "2026-10-30"),
           "nov": ("2026-11-02", "2026-11-20")}
COST = grid40.COST


def sector_states():
    grid = study.session_grid()
    market = grid.join(study.download().loc["SOXX"]).assign(end=grid.index + study.BAR)
    h = study.confirmed_band(market, 60)
    d = study.confirmed_band(market, None)
    times = h.index.union(d.index)
    states = pd.DataFrame({"hour": study.at_times(h, times), "day": study.at_times(d, times)})
    states["both"] = study.coupled_gate(states.hour, states.day)
    return states


def carry(values, index):
    return pd.Series(values, index=index).ffill().fillna(1).to_numpy(bool)


def paths(quotes, original, correction, market, preset, manual):
    idx = quotes.index
    prior = np.r_[original[0], original[:-1]]
    entry = (original == 1) & (prior == 0)
    sell = (original == 0) & (prior == 1)
    regular = quotes.reg.to_numpy(bool)
    available = regular & ((idx.tz_convert("Asia/Seoul").hour >= 8) if manual else True)
    # T2의 손절선은 개장 전 확정되어 밤에도 유지할 수 있다. T1 종가는 수면 중 실행하지 않는다.
    standing_stop = (preset["mode"] == "touch" and preset["tm"] == "T2")
    eligible = available | (sell & standing_stop)
    sample = np.maximum.accumulate(np.where(eligible, np.arange(len(idx)), 0))
    positions = {"base": original[sample].astype(bool)}
    gates = market.reindex(idx, method="ffill").fillna(1).astype(bool)
    for name in ("hour", "day", "both"):
        gate = gates[name].to_numpy()
        desired = carry(np.select([(original == 0) | ~gate, entry], [0., 1.], np.nan), idx)
        positions[name] = desired[sample]
    positions["entry_only"] = carry(np.select([~positions["base"], positions["both"]],
                                              [0., 1.], np.nan), idx)
    result = {}
    for name, held in positions.items():
        previous = np.r_[held[0], held[:-1]]
        exit_now = previous & ~held
        # 시장 경고가 이미 알려졌고 실행할 수 있으면 그 분 시가로 대피한다.
        # 그 외 같은 분에 체결된 기존 손절은 원 엔진의 시가-손절가 차이를 보존한다.
        market_exit = (available & ~gates[name].to_numpy()) if name in gates else False
        adjust = np.where(exit_now & sell & ~np.asarray(market_exit), correction, 0.)
        changed = np.flatnonzero(held != previous)
        result[name] = (held.astype(float), adjust, changed)
    return result


def blocked_trades(quotes, paths):
    held, adj, switches = paths["base"]
    veto = paths["entry_only"][0]
    rows = []
    for start, end in zip(switches[:-1], switches[1:]):
        if not held[start] or veto[start:end].any():
            continue
        gain = quotes.Open.iloc[end] * np.exp(adj[end]) / quotes.Open.iloc[start] - 1
        rows.append({"entry": quotes.index[start], "exit": quotes.index[end], "price_return_pct": 100 * gain})
    return rows


def chart_warning(symbol, quotes, raw, market):
    """현 Pine의 40분 경고를 실제 손절로 바꿀 때. 선과 시장 허용 여부는 개장 전 고정."""
    bars = grid40.E.bars(symbol, 40, "reg")
    level = grid40.E.ema(bars.Close, 100).groupby(bars.date).last().shift(1)
    line = quotes.date.map(level).to_numpy()
    touched = quotes.reg.to_numpy() & (quotes.Low.to_numpy() <= line)
    gate = market.both.reindex(quotes.index, method="ffill").fillna(1).astype(bool)
    weak_at_open = ~gate.groupby(quotes.date).transform("first").to_numpy()
    prev = np.r_[False, raw[:-1]].astype(bool)
    starts = np.flatnonzero(raw.astype(bool) & ~prev)
    ends = np.flatnonzero(~raw.astype(bool) & prev)
    ends = np.r_[ends, len(raw)] if len(ends) < len(starts) else ends
    out = {}
    for name, sector in [("warn_only", False), ("warn_market", True)]:
        hits = np.flatnonzero(touched & (weak_at_open if sector else True))
        held, adjust = raw.copy(), np.zeros(len(raw))
        for begin, end in zip(starts, ends):
            if sector and not gate.iloc[begin]:
                held[begin:end] = 0
                continue
            # 신규 체결 후 손절은 다음 분부터. 기존 보유의 갭은 시가와 선 중 낮은 가격이다.
            j = hits.searchsorted(begin + 1)
            if j < len(hits) and hits[j] < end:
                hit = hits[j]
                held[hit:end] = 0
                adjust[hit] = min(0., np.log(line[hit] / quotes.Open.iloc[hit]))
        out[name] = (held, adjust, np.flatnonzero(held[1:] != held[:-1]) + 1)
    return out


def rolling_overlay(quotes, paths):
    """원 신호 선정 뒤 10/15와 11/2 개장 때 직전 완결 3개월로 대피 정책을 고른다."""
    hold, adj, _ = paths["base"]
    hold, adj = hold.copy(), adj.copy()
    choices = []
    for start, last, train in [("2026-10-15", "2026-10-30", ("2026-07-01", "2026-09-30")),
                               ("2026-11-02", "2026-11-20", ("2026-08-03", "2026-10-30"))]:
        scores = {name: grid40.E.evaluate(*paths[name], train, COST) for name in study.MODES}
        eligible = [name for name, s in scores.items() if s["mdd%"] >= scores["base"]["mdd%"]]
        winner = max(eligible, key=lambda name: scores[name]["ret%"])
        # 첫 정규장 시가까지 기존 정책을 유지하고, 선택 경로의 당시 보유로 전환한다.
        first = quotes.loc[start].loc[lambda d: d.reg].index[0]
        use = quotes.index >= first
        hold[use] = paths[winner][0][use]
        adj[use] = paths[winner][1][use]
        choices.append(dict(start=start, end=last, mode=winner))
    previous = np.r_[hold[0], hold[:-1]]
    return (hold, np.where((previous == 1) & (hold == 0), adj, 0.),
            np.flatnonzero(hold != previous)), choices


def score(quotes, paths):
    index = quotes.index
    logprice = np.log(quotes.Open.to_numpy())
    change = np.diff(logprice, append=logprice[-1])
    days = pd.DatetimeIndex(index.date)
    rows = []
    for window, (left, right) in WINDOWS.items():
        where = np.flatnonzero((days >= left) & (days <= right))
        sel = slice(where[0], where[-1])  # upstream excludes the final minute's forward return
        base, corr, sw = paths["base"]
        base_net = base * change + corr
        base_net[sw] -= COST
        base_daily = np.expm1(pd.Series(base_net[sel], index=days[sel]).groupby(level=0).sum())
        active = base[sel].mean()
        for mode, (hold, adj, switches) in paths.items():
            measured = grid40.E.evaluate(hold, adj, switches, (left, right), COST)
            ratio = hold[sel].mean() / active if active else 0.
            control = np.expm1(np.log1p(ratio * base_daily).sum())
            extra = None
            if mode in ("both", "warn_market", "warn_only"):
                other = paths["base" if mode == "warn_only" else "entry_only"][0]
                exits = np.r_[False, (hold[:-1] == 1) & (hold[1:] == 0)]
                extra = int((exits[sel] & (other[sel] == 1)).sum())
            rows.append({"window": window, "mode": mode, "return_pct": measured["ret%"],
                         "drawdown_pct": measured["mdd%"], "switches": measured["switches"],
                         "exposure_pct": measured["expo%"], "exposure_control_pct": 100 * control,
                         "extra_forced_exits": extra})
    return rows


def main():
    market = sector_states()
    upstream = pd.read_csv(SOURCE / "results" / "grid40.csv", dtype={"i": str}).set_index(["sym", "i"])
    results, blocked, rolling = [], [], []
    for symbol in grid40.SYMS:
        grid40.E.use(symbol)
        quotes = grid40.E.load(symbol)
        for rank, preset in PRESETS.items():
            if rank == "slow":
                source_id = "느린 선(일봉 EMA10·20 종가)"
                original, correction, switches = grid40.refs(symbol)[source_id]
                warnings = chart_warning(symbol, quotes, original, market)
            else:
                original, correction, switches = grid40.run(symbol, preset, {})
                source_id = str(grid40.configs().index(preset))
            for manual in (True, False):
                current = paths(quotes, original, correction, market, preset, manual)
                if rank == "slow":
                    current.update(warnings)
                if not np.all(current["both"][0] <= current["entry_only"][0]):
                    raise ValueError("강제 대피 경로가 신규 차단 경로보다 많이 보유")
                if not manual or rank in ("rank1", "slow"):
                    h, a, sw = current["base"]
                    if not (np.array_equal(h, original) and np.array_equal(a, correction)
                            and np.array_equal(sw, switches)):
                        raise ValueError(f"원본 체결 불일치 {symbol} {rank}")
                if manual:
                    current["rolling3"], choices = rolling_overlay(quotes, current)
                    rolling.extend(dict(symbol=symbol, preset=rank, **row) for row in choices)
                for row in score(quotes, current):
                    if row["mode"] == "rolling3" and row["window"] not in ("confirm", "nov"):
                        continue
                    if not manual and row["mode"] == "base" and row["window"] == "confirm":
                        expected = upstream.loc[(symbol, source_id)]
                        for key, col in [("return_pct", "CONF|ret%"), ("drawdown_pct", "CONF|mdd%")]:
                            if abs(row[key] - expected[col]) > .011:
                                raise ValueError(f"원본 원장 성적 불일치 {symbol} {rank} {key}")
                    results.append(dict(symbol=symbol, preset=rank, execution="awake" if manual else "all", **row))
                if manual:
                    for row in blocked_trades(quotes, current):
                        blocked.append(dict(symbol=symbol, preset=rank, **row))
    for filename, rows in [("indicator_scores.csv", results), ("indicator_blocked.csv", blocked),
                           ("indicator_rolling.csv", rolling)]:
        pd.DataFrame(rows).to_csv(study.HERE / filename, index=False, float_format="%.6f")
    print(pd.DataFrame(results).query(
        "preset=='rank1' and execution=='awake' and window=='confirm' and symbol in ['MU','SOXL']"
    ).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
