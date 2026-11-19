"""Cross-family forward selection over every saved ledger (Codex's 303 + extension families).

Run after common.py and the ext_*.py modules:  py data/derived/analysis/desk_defense_check_20261119/merge.py
Writes out/merged_stats.csv, out/selection.csv, out/walkforward.csv and prints the tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common

PROTOCOLS = (("h1", "h2"), ("jan_aug", "sep_nov"))


def load():
    """All ledgers as {name: DataFrame}; each family's own copy of base must equal Codex's."""
    curves, family_of = {}, {}
    for path in sorted(common.CACHE.glob("*_daily.parquet")):
        family = path.stem[:-len("_daily")]
        frame = pd.read_parquet(path)
        for key, part in frame.groupby("strategy", sort=False):
            ledger = part.set_index("date")[common.LEDGER].astype(float)
            if family != "codex" and key == "base":
                continue
            name = key if family == "codex" else f"{family}:{key}"
            curves[name], family_of[name] = ledger, family
    base = curves["base"]
    for path in sorted(common.CACHE.glob("*_daily.parquet")):
        frame = pd.read_parquet(path)
        own = frame[frame.strategy == "base"]
        if len(own):
            np.testing.assert_allclose(own.usd.to_numpy(), base.usd.to_numpy(), rtol=2e-6)
    return curves, family_of


def describe(table, base_curve, key, test):
    row = table.loc[key]
    ctrl_ret, ctrl_mdd = common.matched_control(base_curve, row[f"{test}_expo"], test)
    return {"strategy": key, f"test_krw": row[f"{test}_krw"], "test_krw_mdd": row[f"{test}_krw_mdd"],
            "test_usd": row[f"{test}_usd"], "test_usd_mdd": row[f"{test}_usd_mdd"],
            "test_expo": row[f"{test}_expo"], "test_turn": row[f"{test}_turn"],
            "matched_usd": ctrl_ret, "matched_usd_mdd": ctrl_mdd}


def selections(table, family_of, base_curve):
    rows = []
    families = sorted(set(family_of.values()))
    scopes = {f: [k for k, v in family_of.items() if v == f or k == "base"] for f in families}
    scopes["ALL"] = list(table.index)
    for train, test in PROTOCOLS:
        for currency in ("krw", "usd"):
            for scope, keys in scopes.items():
                sub = table.loc[[k for k in keys if k != "half"]]
                for rule, key in common.select(sub, train, currency).items():
                    rows.append({"train": train, "test": test, "select_in": currency, "scope": scope,
                                 "rule": rule, "n": len(sub), "train_krw": sub.loc[key, f"{train}_krw"],
                                 "train_krw_mdd": sub.loc[key, f"{train}_krw_mdd"],
                                 **describe(table, base_curve, key, test)})
    return pd.DataFrame(rows)


def rank_persistence(table, family_of):
    """Does the training ranking say anything about the test window? Spearman and top-decile means."""
    rows = []
    for train, test in PROTOCOLS:
        for scope in [*sorted(set(family_of.values())), "ALL"]:
            keys = [k for k in table.index if scope == "ALL" or family_of.get(k) == scope]
            sub = table.loc[keys]
            if len(sub) < 20:
                continue
            for currency in ("krw", "usd"):
                a, b = sub[f"{train}_{currency}"], sub[f"{test}_{currency}"]
                top = a.nlargest(max(1, len(sub) // 10)).index
                shallow = sub[f"{train}_{currency}_mdd"].nlargest(max(1, len(sub) // 10)).index
                rows.append({"train": train, "test": test, "scope": scope, "currency": currency, "n": len(sub),
                             "spearman_ret": a.rank().corr(b.rank()),
                             "spearman_mdd_to_ret": sub[f"{train}_{currency}_mdd"].rank().corr(b.rank()),
                             "test_all_median": b.median(), "test_top10pct_by_train_ret": b[top].mean(),
                             "test_top10pct_by_train_mdd": b[shallow].mean(),
                             "test_base": table.loc["base", f"{test}_{currency}"]})
    return pd.DataFrame(rows)


def walk_forward(curves, table_keys, first_month=4, currency="krw"):
    """Month-end re-selection (Codex 'return' rule), expanding year-to-date and trailing 60 sessions.

    Chains the picked rule's own daily returns; the switch between two rules' holdings at the
    month boundary is not charged (zero commission, close-to-close hand-over) - an approximation.
    """
    days = curves["base"].index
    nav = pd.DataFrame({k: curves[k][currency] for k in table_keys})
    month = pd.Index(days).str[:7]
    out = []
    for mode in ("ytd", "roll60"):
        value, picks = 1.0, []
        for m in sorted(set(month)):
            if int(m[5:]) < first_month:
                continue
            ix = np.flatnonzero(month == m)
            cut = ix[0] - 1
            lo = 0 if mode == "ytd" else max(0, cut - 60)
            window = nav.iloc[lo:cut + 1]
            start = nav.iloc[lo - 1] if lo else pd.Series(1.0, index=nav.columns)
            path = pd.concat([start.to_frame().T, window])
            ret = path.iloc[-1] / path.iloc[0] - 1
            mdd = (path / path.cummax() - 1).min()
            pool = ret.index[(mdd >= mdd["base"] - 1e-7)]
            pick = sorted(pool, key=lambda k: (-ret[k], k))[0]
            seg = nav[pick].iloc[ix[-1]] / nav[pick].iloc[cut]
            base_seg = nav["base"].iloc[ix[-1]] / nav["base"].iloc[cut]
            value *= seg
            picks.append({"mode": mode, "currency": currency, "month": m, "pick": pick,
                          "pick_ret": seg - 1, "base_ret": base_seg - 1, "chained": value})
        out.extend(picks)
    return pd.DataFrame(out)


def main():
    curves, family_of = load()
    table = common.stats_table(curves)
    table.insert(0, "family", [family_of[k] for k in table.index])
    base_curve = curves["base"]
    common.OUT.mkdir(exist_ok=True)
    table.to_csv(common.OUT / "merged_stats.csv", index_label="strategy")
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 400,
                  "display.float_format", lambda v: f"{v:+.3f}")
    print(f"ledgers: {len(curves)} in families {table.family.value_counts().to_dict()}")
    b = table.loc["base"]
    print("base:", {k: round(float(b[k]), 4) for k in table.columns if k != "family" and k.split("_")[-1] in ("krw", "usd")})

    sel = selections(table, family_of, base_curve)
    sel.to_csv(common.OUT / "selection.csv", index=False)
    cols = ["train", "select_in", "scope", "rule", "n", "strategy", "train_krw", "test_krw", "test_usd",
            "test_usd_mdd", "test_expo", "matched_usd", "matched_usd_mdd"]
    print("\n== forward selection ==")
    print(sel[cols].to_string(index=False))

    print("\n== does the training rank persist? ==")
    persist = rank_persistence(table, family_of)
    persist.to_csv(common.OUT / "rank_persistence.csv", index=False)
    print(persist.to_string(index=False))

    print("\n== beats base and beats the exposure-matched control (USD) ==")
    rows = []
    for test in ("h2", "sep_nov", "full"):
        ctrl = np.array([common.matched_control(base_curve, e, test) for e in table[f"{test}_expo"]])
        table[f"{test}_matched_usd"], table[f"{test}_matched_usd_mdd"] = ctrl[:, 0], ctrl[:, 1]
        for family, sub in [*table.groupby("family"), ("ALL", table)]:
            sub = sub.drop(index=[k for k in ("base", "half") if k in sub.index])
            both = (sub[f"{test}_usd"] > b[f"{test}_usd"] + 1e-9) & (sub[f"{test}_usd_mdd"] > b[f"{test}_usd_mdd"] + 1e-9)
            skill = both & (sub[f"{test}_usd"] > sub[f"{test}_matched_usd"] + 1e-9)
            skill2 = skill & (sub[f"{test}_usd_mdd"] > sub[f"{test}_matched_usd_mdd"] + 1e-9)
            rows.append({"test": test, "family": family, "n": len(sub), "beat_base_ret_and_mdd": int(both.sum()),
                         "also_beat_matched_ret": int(skill.sum()), "also_beat_matched_ret_and_mdd": int(skill2.sum()),
                         "median_test_usd": sub[f"{test}_usd"].median(), "best_test_usd": sub[f"{test}_usd"].max()})
    counts = pd.DataFrame(rows)
    counts.to_csv(common.OUT / "counts.csv", index=False)
    print(counts.to_string(index=False))
    table.to_csv(common.OUT / "merged_stats.csv", index_label="strategy")

    keys = [k for k in table.index if k != "half"]
    wf = pd.concat([walk_forward(curves, keys, currency=c) for c in ("krw", "usd")])
    wf.to_csv(common.OUT / "walkforward.csv", index=False)
    print("\n== monthly walk-forward (pick at month end, hold next month) ==")
    for (mode, currency), part in wf.groupby(["mode", "currency"]):
        late = part[part.month >= "2026-07"]
        print(f"[{mode}/{currency}] Jul-Nov chained pick {np.prod(1 + late.pick_ret) - 1:+.3f} vs base "
              f"{np.prod(1 + late.base_ret) - 1:+.3f}; distinct picks {part.pick.nunique()}")
        print(part[["month", "pick", "pick_ret", "base_ret"]].to_string(index=False))


if __name__ == "__main__":
    main()
