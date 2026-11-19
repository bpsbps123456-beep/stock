"""2026 seasons: which species thrived when (hindsight map), and how late point-in-time signals saw it.

The map is a hindsight optimum: hold one species per contiguous season, pay a fixed log penalty for
every change (DP over daily USD log returns), so the penalty controls how many seasons there are.
It is a description, not a strategy. Detection then asks, for each season change, on which session a
point-in-time detector first pointed at the new season's species and how much of the switch benefit
was still ahead at that moment.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from species import Lab, HERE, CHECK

OUT = HERE / "out"
UNLEV = ["MOM", "REV", "PULL", "NONSEMI", "EW", "SOXX", "QQQ", "CASH"]
FULL = UNLEV + ["SOXL", "SOXS", "PSQ"]
PENALTIES = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
SHOW = (0.08, 0.12)


def log_returns(lab, names):
    usd = np.column_stack([lab.ledgers[n].usd.to_numpy() for n in names])
    prev = np.vstack([np.ones(len(names)), usd[:-1]])
    return np.log(usd / prev)


def oracle(R, penalty):
    """Best piecewise-constant species path with a log penalty per switch (Viterbi)."""
    T, K = R.shape
    value, back = R[0].copy(), np.zeros((T, K), dtype=int)
    for t in range(1, T):
        best = int(value.argmax())
        switch = value[best] - penalty
        move = switch > value
        back[t] = np.where(move, best, np.arange(K))
        value = R[t] + np.where(move, switch, value)
    path = np.empty(T, dtype=int)
    path[-1] = int(value.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    return path, float(value.max())


def segments(path):
    cut = np.flatnonzero(np.diff(path)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut - 1, len(path) - 1]
    return list(zip(starts, ends, path[starts]))


def ranked_features(lab):
    """Daily point-in-time market descriptors (value at session j uses closes through j-1)."""
    c = lab.ctx
    soxx = c.book.cls["SOXX"]
    k = np.arange(c.start, c.start + len(c.days)) - 1
    px = soxx.to_numpy(float)
    sma = {n: soxx.rolling(n).mean().to_numpy(float) for n in (10, 20, 50)}
    ret = np.log(px[1:] / px[:-1])
    vol20 = pd.Series(np.r_[np.nan, ret]).rolling(20).std().to_numpy() * np.sqrt(252)
    hi60 = soxx.rolling(60).max().to_numpy(float)
    breadth = pd.read_csv(CHECK.parent / "desk_defense_20261119" / "breadth.csv", index_col=0).breadth
    bb1 = pd.DataFrame(json.loads((CHECK / "cache" / "crowd" / "bb1_calendar.json").read_text(encoding="utf-8")))
    bb1 = bb1.set_index("day").on.reindex(c.days)
    return pd.DataFrame({
        "soxx_vs_sma10": px[k] / sma[10][k] - 1, "soxx_vs_sma20": px[k] / sma[20][k] - 1,
        "soxx_vs_sma50": px[k] / sma[50][k] - 1, "soxx_ret20": px[k] / px[k - 20] - 1,
        "soxx_dd60": px[k] / hi60[k] - 1, "soxx_vol20": vol20[k],
        "breadth": breadth.reindex(c.book.days).to_numpy(float)[k],
        "bb1": bb1.map({True: 1.0, False: 0.0}).to_numpy(float),
    }, index=c.days)


def momentum_ic(lab, horizon):
    """ic[j]: rank correlation of the 120-session momentum known at the open of j with the return
    from the close of j-1 to the close of j-1+horizon. It is known only after that later close."""
    c = lab.ctx
    adj = c.book.adj.to_numpy(float)
    where = {t: i for i, t in enumerate(c.book.adj.columns)}
    out = np.full(len(c.days), np.nan)
    for j, ranked in enumerate(c.rankings):
        k = c.start + j
        if len(ranked) < 8 or k - 1 + horizon >= adj.shape[0]:
            continue
        cols = [where[c.book.ticker[r["key"]]] for r in ranked]
        fwd = adj[k - 1 + horizon, cols] / adj[k - 1, cols] - 1
        mom = np.array([r["mom120"] for r in ranked])
        ok = np.isfinite(fwd) & np.isfinite(mom)
        if ok.sum() >= 8:
            out[j] = pd.Series(mom[ok]).rank().corr(pd.Series(fwd[ok]).rank())
    return out


def known_ic(ic, horizon, window):
    """Point-in-time trailing mean: at session t only ic[j] with j <= t - horizon has matured."""
    s = pd.Series(ic)
    return s.rolling(window, min_periods=max(3, window // 2)).mean().shift(horizon).to_numpy()


def species_momentum(usd, lookback):
    """Index of the species with the best return over `lookback` sessions through the previous close."""
    T = usd.shape[0]
    pick = np.zeros(T, dtype=int)
    for t in range(T):
        hi = t - 1
        lo = hi - lookback
        if hi < 0:
            continue
        base = usd[lo] if lo >= 0 else np.ones(usd.shape[1])
        pick[t] = int(np.argmax(usd[hi] / base))
    return pick


def season_table(lab, names, path, R, feats, ic5):
    days = lab.ctx.days
    usd = {n: lab.ledgers[n].usd.to_numpy() for n in names}
    rows = []
    for i, (a, b, w) in enumerate(segments(path)):
        seg = {n: float(np.exp(R[a:b + 1, names.index(n)].sum()) - 1) for n in names}
        order = sorted(names, key=lambda n: -seg[n])
        rows.append({
            "season": i + 1, "start": days[a], "end": days[b], "sessions": int(b - a + 1),
            "winner": names[w], "winner_ret": seg[names[w]], "second": order[1], "second_ret": seg[order[1]],
            "MOM": seg["MOM"], "REV": seg["REV"], "SOXX": seg["SOXX"], "CASH": seg["CASH"],
            "mom_ic5_realized": float(np.nanmean(ic5[a:b + 1])) if np.isfinite(ic5[a:b + 1]).any() else np.nan,
            **{f"start_{k}": float(feats[k].iloc[a]) for k in ("soxx_vs_sma20", "soxx_vs_sma50", "soxx_vol20",
                                                               "soxx_dd60", "breadth")},
            "bb1_days": int(np.nansum(feats.bb1.to_numpy()[a:b + 1])),
        })
    return pd.DataFrame(rows)


def detection(lab, names, path, R, lookbacks=(5, 10, 20)):
    """For each season change: first session a species-momentum detector names the new winner."""
    usd = np.column_stack([lab.ledgers[n].usd.to_numpy() for n in names])
    picks = {L: species_momentum(usd, L) for L in lookbacks}
    days = lab.ctx.days
    segs = segments(path)
    rows = []
    for i in range(1, len(segs)):
        a, b, w = segs[i]
        p = segs[i - 1][2]
        edge = R[a:b + 1, w] - R[a:b + 1, p]
        row = {"season": i + 1, "start": days[a], "sessions": int(b - a + 1), "from": names[p], "to": names[w],
               "switch_edge": float(np.expm1(edge.sum()))}
        for L, pick in picks.items():
            hit = np.flatnonzero(pick[a:b + 1] == w)
            if len(hit):
                t = a + int(hit[0])
                row[f"L{L}_lag"] = int(hit[0])
                row[f"L{L}_left"] = float(edge[t - a:].sum() / edge.sum()) if edge.sum() > 0 else np.nan
            else:
                row[f"L{L}_lag"], row[f"L{L}_left"] = np.nan, 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    lab = Lab()
    OUT.mkdir(exist_ok=True)
    feats = ranked_features(lab)
    ic = {h: momentum_ic(lab, h) for h in (5, 10, 20)}
    pd.DataFrame({f"ic{h}": v for h, v in ic.items()}, index=lab.ctx.days).join(feats).to_csv(
        OUT / "daily_features.csv", index_label="session")
    summary = {}
    for label, names in (("unlevered", UNLEV), ("with_etf", FULL)):
        R = log_returns(lab, names)
        grid = []
        for pen in PENALTIES:
            path, total = oracle(R, pen)
            lengths = [b - a + 1 for a, b, _ in segments(path)]
            grid.append({"penalty": pen, "seasons": len(lengths), "median_len": float(np.median(lengths)),
                         "min_len": int(min(lengths)), "oracle_usd": float(np.expm1(R[np.arange(len(path)), path].sum())),
                         "winners": "/".join(sorted({names[x] for x in path}))})
            if label == "unlevered" and pen in SHOW:
                tag = f"{label}_p{int(round(pen * 100)):02d}"
                season_table(lab, names, path, R, feats, ic[5]).to_csv(OUT / f"seasons_{tag}.csv", index=False)
                detection(lab, names, path, R).to_csv(OUT / f"detection_{tag}.csv", index=False)
                pd.DataFrame({"session": lab.ctx.days, "species": [names[x] for x in path]}).to_csv(
                    OUT / f"oracle_path_{tag}.csv", index=False)
        pd.DataFrame(grid).to_csv(OUT / f"oracle_grid_{label}.csv", index=False)
        summary[label] = grid
    (OUT / "oracle_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:+.3f}")
    for label in ("unlevered", "with_etf"):
        print(f"\n== oracle grid {label}\n", pd.read_csv(OUT / f"oracle_grid_{label}.csv").to_string(index=False))
    for pen in SHOW:
        tag = f"unlevered_p{int(round(pen * 100)):02d}"
        print(f"\n== seasons {tag}\n", pd.read_csv(OUT / f"seasons_{tag}.csv").to_string(index=False))
        print(f"\n== detection {tag}\n", pd.read_csv(OUT / f"detection_{tag}.csv").to_string(index=False))


if __name__ == "__main__":
    main()
