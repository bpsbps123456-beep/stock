"""2025 auxiliary check: the same pre-declared season dials on 2025 data (not used to choose anything).

Universe: US holdings of the latest weekly TIME PDF dated before each session (strategy_lab holdout
files, 2025 weekly). Prices: the holdout open/close/adjusted files. MOM imitates the desk rule (120-session
momentum, top 7 with keep-buffer, TIME weights, today's caps); REV = 7 worst 5-session names, equal
weight, caps. Gates: SOXX (grade_bars closes) and QQQ. Fills at the next open, USD, no costs.
Families: M2 with ALT=REV (9) and M5 (24), exactly as in meta.py.
"""
from __future__ import annotations

import glob
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from species import HERE, CHECK, multi_switch
from season import known_ic
import study  # noqa: E402
from desk import strategies as S  # noqa: E402

LAB = HERE.parent / "strategy_lab" / "holdout"
GRADE = HERE.parent / "desk_defense_20261119" / "cache" / "grade_bars"
OUT = HERE / "out"
START, END = "2025-01-02", "2025-12-31"


def load():
    px = {f: pd.read_csv(LAB / f"prices_{f}.csv", index_col=0) for f in ("open", "close", "adj_close")}
    # The holdout files carry rows on US holidays (QQQ blank); keep US sessions only so that lookbacks count
    # trading days and nothing is filled at a carried-forward price.
    us = px["close"]["QQQ"].notna()
    px = {f: frame.loc[us] for f, frame in px.items()}
    soxx = pd.read_parquet(GRADE / "SOXX.parquet").Close
    soxx.index = soxx.index.strftime("%Y-%m-%d")
    pdfs = []
    for path in sorted(glob.glob(str(LAB / "time_weekly" / "*.json"))):
        o = json.loads(open(path, encoding="utf-8").read())
        rows = [(h["code"], h["code"].split()[0], float(h["weight"])) for h in o["holdings"]
                if str(h["code"]).endswith("US EQUITY")]
        pdfs.append((o["date"], rows))
    return px, soxx, pdfs


def build():
    """2025 book, plans, momentum IC and gates as SimpleNamespace (shared with other checks)."""
    px, soxx, pdfs = load()
    allday = list(px["adj_close"].index)
    days = [d for d in allday if START <= d <= END]
    k0 = allday.index(days[0])
    tickers = sorted({t for _, rows in pdfs for _, t, _ in rows if t in px["adj_close"].columns})
    col = {t: i for i, t in enumerate(tickers)}
    adj = px["adj_close"][tickers].to_numpy(float)
    opn = px["open"][tickers].ffill().loc[days].to_numpy(float)
    close = px["close"][tickers].ffill().loc[days].to_numpy(float)
    dates = [d for d, _ in pdfs]
    sector = {}

    rankings = []
    for j, day in enumerate(days):
        i = int(np.searchsorted(dates, day)) - 1  # latest PDF strictly before the session
        k = k0 + j
        rows = []
        for code, t, w in (pdfs[i][1] if i >= 0 else []):
            if t not in col:
                continue
            a, b, o = adj[k - 121, col[t]], adj[k - 1, col[t]], opn[j, col[t]]
            r5 = adj[k - 1, col[t]] / adj[k - 6, col[t]] - 1
            if np.isfinite(a) and np.isfinite(b) and a > 0 and np.isfinite(o):
                sector.setdefault(t, S.sector_of(code, t))
                rows.append({"key": t, "mom120": b / a - 1, "r5": r5, "time_weight": w, "sector": sector[t]})
        rows.sort(key=lambda r: -r["mom120"])
        for n, r in enumerate(rows, 1):
            r["rank"] = n
        rankings.append(rows)

    width = len(tickers)

    def weekly(pick):
        targets, prev, started = {}, set(), False
        for j, (day, ranked) in enumerate(zip(days, rankings)):
            week = j == 0 or S.iso_week(day) != S.iso_week(days[j - 1])
            if not ranked or (started and not week):
                continue
            raw = pick(ranked, prev)
            tot = sum(raw.values())
            capped = S.project_caps({k: v / tot for k, v in raw.items()}, sector)
            vec = np.zeros(width)
            for key, v in capped.items():
                vec[col[key]] = v
            targets[j], prev, started = vec, set(raw), True
        return targets

    def mom(ranked, prev):
        keep = S.select(ranked, prev)
        return {r["key"]: r["time_weight"] for r in ranked if r["key"] in keep}

    def rev(ranked, prev):
        ok = sorted((r for r in ranked if np.isfinite(r["r5"])), key=lambda r: r["r5"])[:7]
        return {r["key"]: 1.0 for r in ok}

    plans = {"MOM": weekly(mom), "REV": weekly(rev)}
    plans["HALF"] = {j: .5 * w for j, w in plans["MOM"].items()}
    plans["CASH"] = {}

    # Momentum IC over the PIT universe, as in season.momentum_ic.
    ic_raw = {}
    for h in (5, 10, 20):
        out = np.full(len(days), np.nan)
        for j, ranked in enumerate(rankings):
            k = k0 + j
            if len(ranked) < 8 or k - 1 + h >= adj.shape[0]:
                continue
            idx = [col[r["key"]] for r in ranked]
            fwd = adj[k - 1 + h, idx] / adj[k - 1, idx] - 1
            m = np.array([r["mom120"] for r in ranked])
            ok = np.isfinite(fwd)
            if ok.sum() >= 8:
                out[j] = pd.Series(m[ok]).rank().corr(pd.Series(fwd[ok]).rank())
        ic_raw[h] = out
    market = pd.DataFrame({"SOXX": soxx.reindex(allday), "QQQ": px["close"]["QQQ"]}, index=allday)
    missing = {c: int(market.loc[days, c].isna().sum()) for c in market}
    book = SimpleNamespace(cls=market.ffill())  # holidays of one source carry the last close
    print("market closes missing on 2025 sessions (carried forward):", missing)
    gates = study.market_gates(book, days)
    ones, always = np.ones(len(days)), np.ones(len(days), dtype=bool)

    def ledger(targets, gate=always):
        return study.replay(days, opn, close, ones, targets, gate)

    return SimpleNamespace(days=days, opn=opn, close=close, plans=plans, gates=gates, ic_raw=ic_raw,
                           ledger=ledger, tickers=tickers, rankings=rankings, px=px, soxx=soxx)


def stats(curve):
    res = {}
    for w, (a, b) in {"y2025": (START, END), "h1": ("2025-01-01", "2025-06-30"), "h2": ("2025-07-01", END)}.items():
        ix = np.flatnonzero((curve.index >= a) & (curve.index <= b))
        v = np.r_[curve.usd.iloc[ix[0] - 1] if ix[0] else 1.0, curve.usd.to_numpy()[ix]]
        res[f"{w}_usd"], res[f"{w}_mdd"] = v[-1] / v[0] - 1, (v / np.maximum.accumulate(v) - 1).min()
    res["expo"] = float(curve.exposure.mean())
    return res


def main():
    aux = build()
    days, opn, plans, gates, ic_raw, ledger = aux.days, aux.opn, aux.plans, aux.gates, aux.ic_raw, aux.ledger
    tickers, rankings = aux.tickers, aux.rankings

    rows = {"MOM": stats(ledger(plans["MOM"])), "REV": stats(ledger(plans["REV"])),
            "HALF": stats(ledger(plans["HALF"]))}
    for g in ("SOXX10r1", "SOXX20r1", "SOXX20r3"):
        rows[f"gate_{g}"] = stats(ledger(plans["MOM"], gates[g]))
    for h in (5, 10, 20):
        for K in (5, 10, 20):
            works = ~(known_ic(ic_raw[h], h, K) < 0)
            ch = np.where(works, 0, 1)
            rows[f"M2_h{h}_K{K}_REV"] = {**stats(ledger(multi_switch(opn, [plans["MOM"], plans["REV"]], ch))),
                                          "family": "M2_REV", "switches": int((np.diff(ch) != 0).sum())}
    for g in ("SOXX10r1", "SOXX20r1", "SOXX20r3"):
        for h in (5, 10):
            for K in (10, 20):
                works = ~(known_ic(ic_raw[h], h, K) < 0)
                for down in ("CASH", "HALF"):
                    ch = np.where(~gates[g], 2, np.where(works, 0, 1))
                    led = ledger(multi_switch(opn, [plans["MOM"], plans["REV"], plans[down]], ch))
                    rows[f"M5_{g}_h{h}_K{K}_{down}"] = {**stats(led), "family": "M5",
                                                       "switches": int((np.diff(ch) != 0).sum())}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"
    table.to_csv(OUT / "aux2025.csv")
    monthly_ic = pd.Series(ic_raw[5], index=pd.to_datetime(days)).resample("ME").mean()
    monthly_ic.index = monthly_ic.index.strftime("%Y-%m")
    monthly_ic.to_csv(OUT / "aux2025_monthly_ic5.csv", header=["ic5"])

    pd.set_option("display.width", 220, "display.max_rows", 80, "display.float_format", lambda v: f"{v:+.3f}")
    print(f"universe tickers {len(tickers)}, sessions {len(days)}, first ranked {next(d for d, r in zip(days, rankings) if r)}")
    print(table.to_string())
    fam = table.dropna(subset=["family"])
    m = rows["MOM"]
    for f, part in fam.groupby("family"):
        print(f"{f}: n {len(part)}, median 2025 {part.y2025_usd.median():+.3f} (MOM {m['y2025_usd']:+.3f}), "
              f"beat MOM on return {int((part.y2025_usd > m['y2025_usd']).sum())}, "
              f"on MDD {int((part.y2025_mdd > m['y2025_mdd']).sum())}")
    print("monthly 5-session momentum IC 2025:", monthly_ic.round(3).to_dict())


if __name__ == "__main__":
    main()
