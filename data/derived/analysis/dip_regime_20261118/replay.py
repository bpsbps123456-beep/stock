"""Replay the existing indicator and six single changes on observed 3-minute bars."""
import sys

import numpy as np
import pandas as pd

from collect import CACHE, HERE, SYMBOLS

sys.path.insert(0, str(HERE.parent / "dip_fit/code"))
from ai_fit_build import dead_times, sim
from build import LEV, sim as original_sim
from events import prep

START = pd.Timestamp("2026-07-01")
SPLIT = pd.Timestamp("2026-10-06")
END = pd.Timestamp("2026-11-17")
RULES = ("base", "skip_weak", "skip_drop", "weak_gap3", "weak_rsi60", "gap3", "rsi60")
MODELS = ("chart", "manual", "stress")


def session_days(index):
    dates = index.tz_localize(None).normalize()
    return dates + pd.to_timedelta((index.hour >= 20).astype(int), unit="D")


def load_bars(sym, calendar):
    sip = pd.read_parquet(CACHE / f"{sym}_sip_3Min.parquet")
    boats = pd.read_parquet(CACHE / f"{sym}_boats_3Min.parquet")
    bars = pd.concat([sip[(sip.index.hour >= 4) & (sip.index.hour < 20)],
                      boats[(boats.index.hour < 4) | (boats.index.hour >= 20)]]).sort_index()
    days = session_days(bars.index)
    off_calendar = ~days.isin(calendar)
    bars = bars[~off_calendar]
    if (not bars.index.is_unique or bars.isna().any().any()
            or (bars[["Open", "High", "Low", "Close"]] <= 0).any().any()
            or (bars.Low > bars[["Open", "Close"]].min(axis=1)).any()
            or (bars.High < bars[["Open", "Close"]].max(axis=1)).any()):
        raise ValueError(f"Invalid prices or timestamps: {sym}")
    return bars, int(off_calendar.sum())


def market():
    daily = pd.read_parquet(CACHE / "QQQ_sip_1Day.parquet").Close
    daily.index = daily.index.tz_localize(None).normalize()
    daily = daily.sort_index()
    if daily.index[-1] != END or not daily.index.is_unique:
        raise ValueError("Incomplete QQQ calendar")
    # All regular sessions from the 6/22 warm-up through 11/17 had a 16:00 ET close.
    import exchange_calendars as xc
    schedule = xc.get_calendar("XNYS").schedule.loc["2026-06-22":END]
    assert schedule.index.equals(daily.loc[schedule.index[0]:].index)
    assert (schedule.close.dt.tz_convert("America/New_York").dt.hour == 16).all()
    qbars, _ = load_bars("QQQ", daily.index)
    prev = daily.shift()
    mean20 = daily.rolling(20).mean().shift()
    market_days = pd.DataFrame({"prev": prev, "weak": prev < mean20, "close": daily,
                                "day_return": daily / prev - 1})
    assert mean20.loc["2026-06-22":].notna().all()
    # Label a market bar by when its close became available.
    qclose = qbars.Close.copy()
    qclose.index += pd.Timedelta(minutes=3)
    return market_days, qclose


def symbol_events(sym, market_days, qclose):
    bars, off_calendar = load_bars(sym, market_days.index)
    days = session_days(bars.index)
    regular = (bars.index.time >= pd.Timestamp("09:30").time()) & (bars.index.hour < 16)
    seen_days = days[regular].unique()
    study_days = market_days.loc[START:END].index
    missing = study_days.difference(seen_days)
    coverage = dict(sym=sym, regular_days=len(study_days) - len(missing),
                    missing_regular_days=" ".join(missing.strftime("%Y-%m-%d")),
                    off_calendar_bars=off_calendar, bars=len(bars), baseline_closed=0)
    if len(bars) <= 61:
        return [], coverage
    d = prep(bars, 60, 14)
    minutes = np.asarray(d["mod"])
    session = np.select([regular, (minutes >= 240) & (minutes < 570),
                         (minutes >= 960) & (minutes < 1200)],
                        ["정규", "프리", "애프터"], default="주간")
    regular_close = pd.Series(d.c.to_numpy()[regular], index=days[regular]).groupby(level=0).last()
    hot = regular_close.pct_change().shift().reindex(days).ge(0.07).to_numpy() & regular
    weak = market_days.weak.reindex(days).to_numpy(bool)
    prev_market = market_days.prev.reindex(days).to_numpy()
    # Missing/stale market observations cannot establish a -1% drop.
    known_close = qclose.reindex(d.index, method="ffill", tolerance=pd.Timedelta(minutes=6))
    drop = (known_close.to_numpy() / prev_market - 1) <= -0.01
    kr = d.index.tz_convert("Asia/Seoul")
    awake = (kr.hour >= 8) & (session != "애프터")
    dead = dead_times(d)
    ema_before = d.ema.shift().to_numpy()
    split_end = days.searchsorted(SPLIT, side="right") - 1
    frames = []
    for rule in RULES:
        widen = (rule == "gap3") | ((rule == "weak_gap3") & weak)
        early_exit = (rule == "rsi60") | ((rule == "weak_rsi60") & weak)
        gap = np.where(widen, 0.03, 0.02)
        rsi = np.where(early_exit, 60, 70)
        blocked = hot.copy()
        if rule == "skip_weak":
            blocked |= weak
        elif rule == "skip_drop":
            blocked |= drop
        for model in MODELS:
            mask = blocked.copy()
            if model != "chart":
                mask |= ~awake
            if model == "stress":
                mask |= d.l.to_numpy() > ema_before * (1 - gap) * 0.999
            events = sim(d, dead, mask, gap, rsi, exit_allowed=None if model == "chart" else awake,
                         mark_end=True)
            if rule == "base" and model == "chart":
                old = original_sim(d, dead, hot, 0.02)
                assert old == [event for event in events if event[-1] != "평가"], sym
                coverage["baseline_closed"] = len(old)
            if not events:
                continue
            entry, exit_, prices, proceeds, reason = zip(*events)
            ei, xi = np.array(entry), np.array(exit_)
            price, value = np.array(prices), np.array(proceeds)
            if model != "chart":
                assert awake[ei].all() and awake[xi[np.array(reason) != "평가"]].all()
            # Freeze open training outcomes at the split. Later exits can never influence selection.
            cross = (ei <= split_end) & (xi > split_end)
            score_value = value.copy()
            if cross.any():
                score_value[cross] = d.c.iat[split_end]
            cost = 0.20 if model == "stress" else 0
            rows = pd.DataFrame(dict(sym=sym, model=model, rule=rule, day=days[ei],
                                     entry=d.index[ei], exit=d.index[xi], price=price, proceeds=value,
                                     session=session[ei], exit_session=session[xi], weak=weak[ei],
                                     reason=reason, split_mark=cross, known_qqq=known_close.notna().to_numpy()[ei],
                                     pct=(value / price - 1) * 100 - cost,
                                     score_pct=(score_value / price - 1) * 100 - cost))
            frames.append(rows[rows.day.between(START, END)])
    return frames, coverage


def main():
    assert len(SYMBOLS) == 78 and not LEV.intersection(SYMBOLS)
    market_days, qclose = market()
    trades, coverage = [], []
    for sym in SYMBOLS:
        frames, counts = symbol_events(sym, market_days, qclose)
        trades.extend(frames)
        coverage.append(counts)
    result = pd.concat(trades, ignore_index=True)
    assert not result.duplicated(["sym", "rule", "model", "entry"]).any()
    result.to_parquet(CACHE / "trades.parquet", index=False)
    pd.DataFrame(coverage).to_csv(HERE / "coverage.csv", index=False)
    print(result.groupby(["model", "rule"]).size())


if __name__ == "__main__":
    main()
