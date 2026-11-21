"""기존 일봉 신호에 SOXX 시간봉·일봉 약세와 회복 대기 1~3회를 겹쳐 재생한다."""
import numpy as np
import pandas as pd

from inputs import FIRST, HERE, LAST, ORIGINAL, grid40, replay_panel, sip_bars, study
from check_indicator import sector_states

COST = grid40.COST  # 원 연구와 같은 체결당 로그비용
WINDOWS = {
    "train": ("2026-01-02", "2026-05-29"), "forward": ("2026-06-01", LAST),
    "quarter": ("2026-08-20", LAST), "oct15": ("2026-10-15", LAST),
}
# 순서를 동률 처리에도 쓴다. 기준, 짧은 확인 대기, 같은 날수면 새 신호 우선.
ENTRY_RULES = ["base", "fresh1", "resume1", "fresh2", "resume2", "fresh3", "resume3"]
FORCED = ["exit_hour", "exit_day", "exit_both", "exit_both_resume", "exit_matched", "exit_qqq", "exit_spy"]
OTHER_MARKETS = ["new_qqq", "new_spy"]


def matched_hourly_band():
    """시간봉을 일봉 EMA10·20과 같은 거래일 감쇠로 평활한다. 마지막 30분봉은 짧게 반영한다."""
    grid = study.session_grid()
    quotes = grid.join(study.download().loc["SOXX"])
    groups = quotes.groupby([quotes.day, quotes.slot // 6], sort=False)
    close = groups.c.last()
    sizes = groups.size()
    if not groups.c.count().eq(sizes).all():
        raise ValueError("거래일 길이 일치 대조에서 SOXX 10분봉 누락")
    fractions = sizes / sizes.groupby(level=0).transform("sum")
    smooth = {}
    for span in (10, 20):
        decay = (1 - 2 / (span + 1)) ** fractions.to_numpy()
        line = np.empty(len(close))
        last = close.iloc[0]
        for i, price in enumerate(close):
            last = decay[i] * last + (1 - decay[i]) * price
            line[i] = last
        smooth[span] = line
    flag = np.select([close > np.maximum(smooth[10], smooth[20]),
                      close < np.minimum(smooth[10], smooth[20])], [1., 0.], default=np.nan)
    return pd.Series(flag, index=close.index).ffill().groupby(level=0).last()


def market_at_closes(dates):
    saved = HERE / "market_states.csv"
    if saved.exists():
        return pd.read_csv(saved, parse_dates=["date"]).set_index("date").loc[dates]
    cal = study.market_calendar().schedule.loc[FIRST:LAST]
    ends = pd.DatetimeIndex(cal.close)
    market = sector_states().reindex(ends, method="ffill").set_axis(cal.index).fillna(1).astype(int)
    market["matched"] = matched_hourly_band().reindex(cal.index).fillna(1).astype(int)
    grid = study.session_grid()
    broad = sip_bars(["QQQ", "SPY"], "10Min")
    for symbol in ("QQQ", "SPY"):
        quotes = grid.join(broad.loc[symbol]).assign(end=grid.index + study.BAR)
        if quotes.c.isna().any():
            raise ValueError(f"{symbol} 정규장 10분봉 누락")
        hour = study.confirmed_band(quotes, 60)
        day = study.confirmed_band(quotes, None)
        times = hour.index.union(day.index)
        both = study.coupled_gate(study.at_times(hour, times), study.at_times(day, times))
        market[symbol.lower()] = both.reindex(ends, method="ffill").to_numpy(int)
    # 갈라진 시간·일봉에서는 기존 두 주기 결합 상태를 잇는다. 두 주기가 따로 켜진 날도 있다.
    market["streak"] = market.both.groupby(market.both.eq(0).cumsum()).cumsum()
    market.rename_axis("date").to_csv(saved)
    return market.loc[dates]


def positions(own, market, policies):
    """각 행은 당시 정규장 개장. own은 이미 직전 종가까지의 원 지표 체결 상태다."""
    before = market.shift(1).fillna(0)
    n, count = own.shape
    path = np.zeros((n, count), dtype=bool)
    held = np.zeros(count, dtype=bool)
    previous = np.zeros(count, dtype=bool)
    # 새 종목 신호가 없는 종목은 기다리고, 보유 중이면 원 지표 매도까지 유지한다.
    for i, (rule, state) in enumerate(zip(policies, before.itertuples())):
        active = own[i]
        fresh = active & ~previous
        if rule == "base":
            held = active
        elif rule in OTHER_MARKETS:
            allowed = bool(getattr(state, rule.split("_")[1]))
            held = active & (held | (allowed & fresh))
        elif rule in FORCED:
            market_name = rule.split("_")[1]
            eligible = active if rule == "exit_both_resume" else fresh
            held = bool(getattr(state, market_name)) & active & (held | eligible)
        else:
            days = int(rule[-1])
            eligible = fresh if rule.startswith("fresh") else active
            held = active & (held | ((state.streak >= days) & eligible))
        path[i] = held
        previous = active
    return path


def return_legs(hold, opening, closing):
    """두 평가 시점은 비용을 낸 개장 직후와 종가. 갭은 전일 보유자 몫."""
    prev = np.vstack((np.zeros(hold.shape[1], dtype=bool), hold[:-1]))
    prev_close = np.vstack((opening[0], closing[:-1]))
    open_leg = prev * np.log(opening / prev_close) - (hold != prev) * COST
    close_leg = hold * np.log(closing / opening)
    return np.stack([open_leg, close_leg], axis=1)


def score(legs, mask):
    selected = legs[mask].reshape(-1, legs.shape[-1])
    curve = np.vstack((np.zeros(legs.shape[-1]), selected.cumsum(axis=0)))
    dd = (curve - np.maximum.accumulate(curve, axis=0)).min(axis=0)
    return 100 * np.expm1(curve[-1]), 100 * np.expm1(dd)


def window(dates, left, right):
    return (dates >= left) & (dates <= right)


def pick(all_legs, mask, originals):
    scores = {}
    for name in ENTRY_RULES:
        ret, dd = score(all_legs[name], mask)
        scores[name] = (np.median(ret[originals]), np.median(dd[originals]))
    eligible = [name for name in ENTRY_RULES if scores[name][1] >= scores["base"][1] - 1e-9]
    return max(eligible, key=lambda name: round(scores[name][0], 8))


def denied_trades(own, dates, symbols, opening, closing, market, paths):
    """첫날 허용된 신규 회차 중 2·3일 대기가 걸러낸 거래. 원래 청산 시가까지의 독립 손익."""
    rows = []
    first_session = pd.Timestamp(WINDOWS["train"][0])
    for col, symbol in enumerate(symbols):
        on = own[:, col]
        prev = np.r_[False, on[:-1]]
        starts = np.flatnonzero(on & ~prev)
        exits = np.flatnonzero(~on & prev)
        for start in starts:
            if dates[start] < first_session or not paths["fresh1"][start, col]:
                continue
            if paths["fresh3"][start, col]:
                continue
            idx = exits.searchsorted(start)
            closed = idx < len(exits)
            end = exits[idx] if closed else len(on) - 1
            end_price = opening[end, col] if closed else closing[end, col]
            net = np.log(end_price / opening[start, col]) - COST * (1 + int(closed))
            streak = int(market.streak.iloc[start - 1])
            rows.append(dict(symbol=symbol, cohort="original9" if symbol in ORIGINAL else "extra10",
                             entry=dates[start].date(), exit_or_mark=dates[end].date(), closed=closed,
                             recovery_close=dates[start - streak].date(), confirmation_at_entry=streak,
                             blocked_by_two=not paths["fresh2"][start, col], trade_net_pct=100*np.expm1(net)))
    return pd.DataFrame(rows)


def save(name, frame):
    frame.to_csv(HERE / name, index=False, float_format="%.6f")


def main():
    panel = replay_panel()
    opening = panel.pivot(columns="symbol", values="open")
    dates, symbols = opening.index, opening.columns
    closing = panel.pivot(columns="symbol", values="close").to_numpy()
    own = panel.pivot(columns="symbol", values="original_at_open").to_numpy(bool)
    market = market_at_closes(dates)
    expected = study.market_calendar().schedule.loc[FIRST:LAST].index
    if (not dates.equals(expected) or len(symbols) != 19
            or not panel.original_at_open.isin([0, 1]).all()
            or not market.drop(columns="streak").isin([0, 1]).all().all()
            or not np.isfinite(opening).all().all() or not np.isfinite(closing).all()):
        raise ValueError("거래일·종목 가격 검산 오류")
    opening = opening.to_numpy()
    originals = symbols.isin(ORIGINAL)
    paths = {name: positions(own, market, [name]*len(dates)) for name in ENTRY_RULES + FORCED + OTHER_MARKETS}
    if not (np.array_equal(own, paths["base"]) and np.all(paths["fresh3"] <= paths["fresh2"])
            and np.all(paths["fresh2"] <= paths["fresh1"]) and np.all(paths["fresh1"] <= paths["resume1"])
            and np.all(paths["exit_both"] <= paths["fresh1"])):
        raise ValueError("보유 경로 포함 관계 검산 오류")
    legs = {name: return_legs(held, opening, closing) for name, held in paths.items()}
    selected = np.full(len(dates), "base", dtype=object)
    choices = []
    for month in pd.period_range("2026-06", "2026-11", freq="M"):
        train_start, train_end = (month - 3).start_time, (month - 1).end_time
        winner = pick(legs, window(dates, train_start, train_end), originals)
        selected[window(dates, month.start_time, month.end_time)] = winner
        choices.append(dict(application=str(month), training_start=train_start.date(),
                            training_end=train_end.date(), rule=winner))
    # 월초 정책만 바꾼다. 기존 보유와 진행 중 신호에서 실제 필요한 주문만 한 번 계산한다.
    paths["rolling"] = positions(own, market, selected)
    legs["rolling"] = return_legs(paths["rolling"], opening, closing)
    left, right = WINDOWS["train"]
    winner = pick(legs, window(dates, left, right), originals)
    choices.append(dict(application="06~11 fixed", training_start=left, training_end=right, rule=winner))
    # 5월 말에 고른 고정 정책도 실제로 6월 첫 개장부터 적용한다. 직전 보유는 기준 지표에서 승계한다.
    fixed = np.where(dates >= WINDOWS["forward"][0], winner, "base")
    paths["fixed"] = positions(own, market, fixed)
    legs["fixed"] = return_legs(paths["fixed"], opening, closing)

    scores = []
    base_daily = np.expm1(legs["base"].sum(axis=1))
    base_prev = np.vstack((np.zeros(len(symbols), dtype=bool), own[:-1]))
    for label, (left, right) in WINDOWS.items():
        mask = window(dates, left, right)
        base_count = own[mask].sum(axis=0) + base_prev[mask].sum(axis=0)
        for name, value in legs.items():
            if label == "train" and name in ("fixed", "rolling"):
                continue
            ret, dd = score(value, mask)
            held = paths[name]
            prev = np.vstack((np.zeros(len(symbols), dtype=bool), held[:-1]))
            starts = held & ~prev
            # 단순히 현금을 늘린 효과와 비교한다. 구간이 끝난 뒤, 기준·실험의 보유 구간 수를 맞춘다.
            # 매일 두 구간(전일 종가~시가, 시가~종가)을 센다.
            held_count = held[mask].sum(axis=0) + prev[mask].sum(axis=0)
            ratio = np.divide(held_count, base_count, out=np.zeros(len(symbols)), where=base_count > 0)
            scaled = 100 * np.expm1(np.log1p(base_daily[mask] * ratio).sum(axis=0))
            for i, symbol in enumerate(symbols):
                scores.append(dict(window=label, mode=name, symbol=symbol,
                                   cohort="original9" if originals[i] else "extra10",
                                   return_pct=ret[i], drawdown_pct=dd[i], sessions=int(mask.sum()),
                                   new_entries=int(starts[mask, i].sum()), held_days=int(held[mask, i].sum()),
                                   relative_exposure_pct=100 * ratio[i], scaled_base_pct=scaled[i],
                                   diff_scaled_pp=ret[i]-scaled[i]))
    scores = pd.DataFrame(scores)
    base_returns = scores.query("mode=='base'").set_index(["window", "symbol"]).return_pct
    scores["diff_pp"] = scores.return_pct.to_numpy() - base_returns.reindex(
        pd.MultiIndex.from_frame(scores[["window", "symbol"]])).to_numpy()
    summary = scores.groupby(["window", "mode", "cohort"], sort=False).agg(
        median_return_pct=("return_pct", "median"), median_drawdown_pct=("drawdown_pct", "median"),
        median_paired_diff_pp=("diff_pp", "median"), better=("diff_pp", lambda a: (a > 1e-8).sum()),
        worse=("diff_pp", lambda a: (a < -1e-8).sum()), trades=("new_entries", "sum"),
        median_diff_scaled_pp=("diff_scaled_pp", "median")).reset_index()
    save("scores.csv", scores)
    save("summary.csv", summary)
    save("choices.csv", pd.DataFrame(choices))
    save("blocked.csv", denied_trades(own, dates, symbols, opening, closing, market, paths))
    print(pd.DataFrame(choices).to_string(index=False))


if __name__ == "__main__":
    main()
