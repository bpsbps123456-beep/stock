"""Summarize the fixed train/forward comparison. Units of returns are percentage points."""
import json

import numpy as np
import pandas as pd

from replay import CACHE, HERE, MODELS, RULES, START, SPLIT, END, market


def stats(g, field="pct"):
    r = g[field]
    return dict(n=len(g), days=g.day.nunique(), avg=r.mean(), median=r.median(),
                win=100 * (r > 0).mean(), p05=r.quantile(.05), worst=r.min(),
                loss3=100 * (r <= -3).mean(), day_avg=g.groupby("day")[field].mean().mean(),
                open=int((g.reason == "평가").sum()))


def by(g, columns):
    records = []
    for key, group in g.groupby(columns, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(columns, key)) | stats(group))
    return records


def score(g, calendar):
    days = g.groupby(["day", "rule"]).score_pct.agg(["sum", "count"])
    totals = days["sum"].unstack().reindex(index=calendar, columns=RULES).fillna(0)
    counts = days["count"].unstack().reindex(index=calendar, columns=RULES).fillna(0)
    delta = totals.subtract(totals.base, axis=0)
    return delta, counts.base


def block_interval(delta, base_n):
    rng = np.random.default_rng(20261118)
    length = len(delta)
    starts = rng.integers(length, size=(10000, (length + 4) // 5))
    positions = ((starts[:, :, None] + np.arange(5)) % length).reshape(10000, -1)[:, :length]
    effect = delta.to_numpy()[positions].sum(axis=1) / base_n.to_numpy()[positions].sum(axis=1)
    return np.quantile(effect, [.025, .975]).tolist()


def composition(base):
    past = base[base.day < "2026-11-01"]
    nov = base[base.day >= "2026-11-01"]
    past_rates = past.groupby("weak").pct.mean()
    weights = nov.weak.value_counts(normalize=True)
    expected = (past_rates.reindex(weights.index) * weights).sum()
    return dict(pre=past.pct.mean(), nov=nov.pct.mean(), expected=expected,
                pre_weak_pct=100 * past.weak.mean(), nov_weak_pct=100 * nov.weak.mean(),
                mix_effect=expected - past.pct.mean(), within_effect=nov.pct.mean() - expected)


def next_open_check(t, calendar, selected):
    """Settle 11/17 pending manual exits only. Do not introduce any 11/18 entries."""
    g = t[(t.model != "chart") & (t.day > SPLIT)].copy()
    marked = g.reason == "평가"
    first = {}
    for sym in g.loc[marked, "sym"].unique():
        boats = pd.read_parquet(CACHE / "next" / f"{sym}_boats_3Min.parquet")
        sip = pd.read_parquet(CACHE / "next" / f"{sym}_sip_3Min.parquet")
        obs = pd.concat([boats[(boats.index.hour >= 20) | (boats.index.hour < 4)],
                         sip[(sip.index.hour >= 4) & (sip.index.hour < 16)]]).sort_index()
        obs = obs[obs.index.tz_convert("Asia/Seoul").hour >= 8]
        first[sym] = (obs.index[0], obs.Open.iat[0])
    # All these positions had already reached their Korean 08:00 deadline by the first eligible bar.
    local_entry = g.loc[marked, "entry"].dt.tz_convert("Asia/Seoul")
    at_eight = local_entry.dt.normalize() + pd.Timedelta(hours=8)
    deadline = at_eight + pd.to_timedelta((local_entry >= at_eight).astype(int), unit="D")
    next_bar = g.loc[marked, "sym"].map(lambda s:first[s][0]).dt.tz_convert("Asia/Seoul")
    assert (next_bar >= deadline).all()
    open_price = g.loc[marked, "sym"].map(lambda s:first[s][1])
    g.loc[marked, "pct"] = 100 * (open_price / g.loc[marked, "price"] - 1) - 0.20 * (g.loc[marked, "model"] == "stress")
    g["score_pct"] = g.pct
    g.loc[marked, "reason"] = "이월 청산"
    nov_base = g[(g.day.dt.month == 11) & (g.rule == "base")]
    manual = nov_base[nov_base.model == "manual"]
    daily = manual.groupby("day").pct.sum()
    worst_day = daily.idxmin()
    worst = manual[manual.day == worst_day]
    result = {
        "n_base_manual_settled": int((marked & (g.model == "manual") & (g.rule == "base")).sum()),
        "last_fill_et": str(max(bar for bar, _ in first.values())),
        "nov_base": by(nov_base, ["model"]),
        "nov_market_direction": by(manual, ["down_today"]),
        "nov_exits": by(manual, ["reason"]),
        "nov_win_avg": manual.loc[manual.pct > 0, "pct"].mean(),
        "nov_loss_avg": manual.loc[manual.pct < 0, "pct"].mean(),
        "worst_day": {"date": worst_day.date().isoformat(), "n": len(worst), "syms": worst.sym.nunique(),
                      "avg": worst.pct.mean(), "other_days_avg": manual.loc[manual.day != worst_day, "pct"].mean()},
    }
    delta, n = score(g[g.model == "stress"], calendar[calendar > SPLIT])
    result["selected_forward_delta"] = float(delta[selected].sum() / n.sum())
    result["selected_forward_ci"] = block_interval(delta[selected], n)
    return result


def main():
    t = pd.read_parquet(CACHE / "trades.parquet")
    market_days, _ = market()
    cal = market_days.loc[START:END]
    t["month"] = t.day.dt.strftime("%Y-%m")
    t["stage"] = np.where(t.day <= SPLIT, "train", "forward")
    t["down_today"] = (market_days.day_return.reindex(t.day).to_numpy() < 0)
    base = t[t.rule == "base"]
    month_rows = []
    for month, group in cal.groupby(cal.index.strftime("%Y-%m")):
        month_rows.append(dict(month=month, start=group.index[0].date().isoformat(),
                               end=group.index[-1].date().isoformat(), n=len(group),
                               qqq_pct=100 * (group.close.iloc[-1] / group.prev.iloc[0] - 1),
                               weak_days=int(group.weak.sum()), down_days=int((group.day_return < 0).sum())))
    effects = []
    selected = None
    selected_ci = None
    for model in ("stress", "manual", "chart"):
        for stage, sample in (("train", cal.index[cal.index <= SPLIT]),
                              ("forward", cal.index[cal.index > SPLIT]),
                              ("nov", cal.index[cal.index.month == 11])):
            g = t[(t.model == model) & t.day.isin(sample)]
            delta, count = score(g, sample)
            gain = delta.sum() / count.sum()
            if model == "stress" and stage == "train":
                selected = gain.idxmax()  # base comes first and scores exactly zero
            if model == "stress" and stage == "forward" and selected != "base":
                selected_ci = block_interval(delta[selected], count)
            for rule in RULES:
                group = g[g.rule == rule]
                effects.append(dict(model=model, stage=stage, rule=rule, delta=gain[rule],
                                    split_mark=int(group.split_mark.sum()) if stage == "train" else 0)
                               | stats(group, "score_pct"))
    diagnosis = []
    for label, group in (("all", base), ("forward", base[base.day > SPLIT]),
                         ("nov", base[base.day.dt.month == 11])):
        for factor in ("weak", "down_today", "session", "reason"):
            diagnosis.extend(dict(period=label, factor=factor, **row) for row in by(group, ["model", factor]))
    coverage = pd.read_csv(HERE / "coverage.csv")
    full = coverage.loc[coverage.regular_days == len(cal), "sym"]
    fixed = base[base.sym.isin(full)]
    output = dict(
        start=str(START.date()), split=str(SPLIT.date()), end=str(END.date()), n_days=len(cal),
        qqq_pct=100 * (cal.close.iloc[-1] / cal.prev.iloc[0] - 1),
        qqq_last={"close":float(cal.close.iloc[-1]), "prior_close":float(cal.prev.iloc[-1]),
                  "prior_ma20":float(market_days.close.rolling(20).mean().shift().loc[END]),
                  "since_high_pct":100 * (cal.close.iloc[-1] / cal.close.max() - 1),
                  "highest_close_day":cal.close.idxmax().date().isoformat()},
        month_market=month_rows, monthly=by(base, ["model","month"]),
        month_sessions=by(base, ["model", "month", "session"]),
        diagnosis=diagnosis, regime_session=by(base, ["model","session","weak"]),
        balanced_count=len(full), balanced=by(fixed, ["model","month"]),
        missing_qqq_prior_pct=100 * (1 - base.loc[base.model == "chart", "known_qqq"].mean()),
        split_comparisons=effects, selected=selected, forward_ci=selected_ci,
        composition={m:composition(base[base.model == m]) for m in MODELS},
        # 11/18 prices are used only for this endpoint sensitivity, after the 11/17 comparison is frozen.
        next_open=next_open_check(t, cal.index, selected))
    # Pandas converts nullable/empty cells to valid JSON null and timestamps are already ISO dates.
    normalized = json.loads(pd.Series(output, dtype=object).to_json(force_ascii=False, double_precision=8))
    (HERE / "results.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Selected:", selected, "forward 95% interval:", selected_ci)
    print(pd.DataFrame(month_rows).round(2).to_string(index=False))
    print(pd.DataFrame(output["monthly"])[["model", "month", "n", "avg", "win", "p05"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
