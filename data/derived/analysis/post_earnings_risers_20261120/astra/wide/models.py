"""Forward-selected multi-output earnings models for PREREG3. All training uses S."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / "results/panel_wide.csv"
TARGETS = [f"{entry}_h{h}" for entry in ("e1", "e6") for h in (5, 10, 20)]
TAPE = "pre20_x pre60_x dist_hi vol60 vsurge5 dvol20 intra20 over20 eff20 r0_x gap o2c clv vol0 r0_sig".split()
FLOW = "eps_surprise occ_tot_ab occ_call_ab occ_putshare dtc si_ch kr_days_pre kr_buy_share".split()
REGIME = "qqq_pre20 qqq_above50 vix soxx_rel20 react_mon".split()
BUNDLES = {"tape": TAPE, "flow": TAPE + FLOW, "regime": TAPE + FLOW + REGIME}
INTERACTIONS = [tuple(s.split("*")) for s in (
    "pre20_x*r0_x", "pre60_x*vol60", "intra20*over20", "clv*vol0", "r0_x*vol0",
    "eps_surprise*r0_sig", "occ_call_ab*vol0", "dtc*si_ch", "pre20_x*qqq_pre20",
    "vol60*vix", "d5_x*str5", "d5_x*r0_x", "qqq_pre20*soxx_rel20",
)]
HYPERS = ("tree2", "tree4", "extra4", "ridge30", "ridge150")
TAILS = (.025, .05, .10, .20, .33)
SEED = 20261120


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def panel(train_only=False):
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if not train_only or
                (r["src"] == "wide" and r["reaction_date"] < "2026-07-01")]
    df = pd.DataFrame(rows)
    assert not df.duplicated(["src", "ticker", "reaction_date"]).any()
    for name in df:
        if name in ("src", "ticker", "reaction_date", "sector", "etf"):
            continue
        if name in ("fund_ticker", "qqq_above50", "react_mon"):
            df[name] = df[name].map({"True": True, "False": False})
        else:
            df[name] = pd.to_numeric(df[name], errors="raise").replace([np.inf, -np.inf], np.nan)
    return df


def split(df):
    date = df.reaction_date
    fund = df.src.eq("fund")
    first_half = date.between("2026-01-01", "2026-06-30")
    gates = date.between("2026-07-01", "2026-10-21")
    return {k: v.to_numpy(bool) for k, v in {
        "S": ~fund & first_half, "A": ~fund & ~df.fund_ticker & gates,
        "B": fund & gates, "early": fund & date.between("2026-10-22", "2026-11-12"),
    }.items()}


def quantile_rank(x, reference):
    reference = np.sort(reference[np.isfinite(reference)])
    assert len(reference)
    left = np.searchsorted(reference, x, side="left")
    right = np.searchsorted(reference, x, side="right")
    rank = (left + right) / len(reference) - 1.
    return np.where(np.isfinite(x), rank, 0.)


class Design:
    def __init__(self, df):
        self.df = df
        self.part = split(df)
        self.q = pd.to_datetime(df.reaction_date).dt.quarter.to_numpy()
        self.k = df.k.to_numpy()
        self.s = self.part["S"]
        self.end_s = self.k[self.s].max()
        self.start_q2 = self.k[self.s & (self.q == 2)].min()
        # All three horizons must be known by the Q2 boundary for a joint Q1 fit.
        self.train = self.s & (self.q == 1) & (self.k + 25 < self.start_q2)
        self.ready6 = df.d5_x.notna().to_numpy()
        self.mature = {t: self.k + self.exit_offset(t) <= self.end_s for t in TARGETS}
        u = df[TARGETS].to_numpy()
        self.y = np.stack([u, df[[t+"_s" for t in TARGETS]].to_numpy()], axis=2)
        assert np.isfinite(self.y[self.s]).all()
        columns = list(dict.fromkeys(TAPE + FLOW + REGIME + ["d5_x", "str5"]))
        ranks = {}
        for name in columns:
            x = df[name].to_numpy(dtype=float)
            ranks[name] = quantile_rank(x, x[self.train])
        self.designs = {}
        self.sizes = {}
        self.column_names = {}
        self.scale_sources = {"train_rows": int(self.train.sum()), "last_Q1_fit_date": df.reaction_date[self.train].max(),
                              "last_S_event": df.reaction_date[self.s].max()}
        levels = sorted(set(df.loc[self.train, "sector"]) - {""})
        for entry in ("e1", "e6"):
            for bundle, base in BUNDLES.items():
                raw = base + (["d5_x", "str5"] if entry == "e6" else [])
                cols = [ranks[c] for c in raw]
                names = raw.copy()
                if bundle == "regime":
                    names.extend("sector="+s for s in levels)
                    cols.extend(df.sector.eq(s).to_numpy(float) for s in levels)
                base_count = len(names)
                for a, b in INTERACTIONS:
                    if a in raw and b in raw:
                        cols.append(ranks[a] * ranks[b])
                        names.append(a+"*"+b)
                self.designs[entry, bundle] = np.column_stack(cols)
                self.sizes[entry, bundle] = base_count
                self.column_names[entry, bundle] = names

    @staticmethod
    def exit_offset(target):
        entry, horizon = target.split("_h")
        return int(horizon) + (5 if entry == "e6" else 0)

    def model_scores(self, label_tensor):
        for entry, offset in (("e1", 0), ("e6", 3)):
            y = (label_tensor[self.train, offset:offset+3, 0] > 0).astype(float)
            for bundle in BUNDLES:
                all_x = self.designs[entry, bundle]
                base_x = all_x[:, :self.sizes[entry, bundle]]
                for hyper in HYPERS:
                    if hyper.startswith("ridge"):
                        x = all_x
                        model = Ridge(alpha=float(hyper[5:]), solver="cholesky")
                    elif hyper == "extra4":
                        x = base_x
                        model = ExtraTreesRegressor(n_estimators=32, max_depth=4, min_samples_leaf=25,
                                                    max_features=.75, random_state=SEED, n_jobs=1)
                    else:
                        x = base_x
                        model = DecisionTreeRegressor(max_depth=int(hyper[-1]), min_samples_leaf=35,
                                                      random_state=SEED)
                    model.fit(x[self.train], y)
                    scores = model.predict(x)
                    for i, target in enumerate(TARGETS[offset:offset+3]):
                        yield target, offset+i, bundle, hyper, scores[:, i]


def wilson(w, n):
    if n == 0:
        return 0.
    p = w/n
    return (p + .5/n - np.sqrt(p*(1-p)/n + .25/n**2)) / (1+1/n)


def candidate_mask(scores, ready, cut, side):
    return ready & ((scores >= cut) if side == "up" else (scores <= cut))


def select_models(design, tensor):
    best = {}
    ranks = {}
    all_pass = {}
    frozen_scores = {}
    for target, j, bundle, hyper, scores in design.model_scores(tensor):
        ready = design.ready6 if target.startswith("e6") else np.ones(len(scores), bool)
        observation_lag = 5 if target.startswith("e6") else 0
        calib = design.s & ready & (design.k+observation_lag <= design.end_s)
        qs = np.quantile(scores[calib], list(TAILS) + [1-x for x in TAILS])
        usable = design.s & design.mature[target] & np.isfinite(tensor[:, j, 0])
        for side in ("up", "down"):
            key = target+"_"+side
            all_pass.setdefault(key, 0)
            sign = 1 if side == "up" else -1
            wins = sign*tensor[:, j, 0] > 0.
            for i, tail in enumerate(TAILS):
                lo, hi = qs[i], qs[i+len(TAILS)]
                if lo >= hi:
                    continue
                cut = float(hi if side == "up" else lo)
                m = candidate_mask(scores, ready, cut, side) & usable
                q1 = m & (design.q == 1)
                q2 = m & (design.q == 2)
                n1, n2 = int(q1.sum()), int(q2.sum())
                w1, w2 = int(wins[q1].sum()), int(wins[q2].sum())
                n, w = n1+n2, w1+w2
                if n < 60 or min(n1,n2) < 20:
                    continue
                shortfall = max(0, .65-w1/n1) + max(0, .65-w2/n2) + max(0, .70-w/n)
                passed = w*10 >= 7*n and w1*100 >= 65*n1 and w2*100 >= 65*n2
                all_pass[key] += int(passed)
                rank = (passed, -round(shortfall, 12), wilson(w2,n2),
                        min(wilson(w1,n1), wilson(w2,n2)), wilson(w,n))
                if key in ranks and rank <= ranks[key]:
                    continue
                ranks[key] = rank
                best[key] = dict(target=target, side=side, bundle=bundle, model=hyper, tail=tail, cut=cut,
                                 selection_q1=[w1,n1], selection_q2=[w2,n2], selection_pass=bool(passed))
                frozen_scores[key] = scores
    assert len(best) == 12
    for key, rec in best.items():
        rec["eligible_settings"] = all_pass[key]
        rec["score_sha256"] = hashlib.sha256(np.round(frozen_scores[key][design.s], 10).astype("<f8").tobytes()).hexdigest()
    return best, frozen_scores

