"""Compare the independent flags with the engine's zone_flags.parquet: 400-signal sample (seed 20261103) and full population."""
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEED

KEY = ["period", "sym", "ts", "pat"]
REQ = ["z_hit", "z_str", "z_near", "pdl_hit"]
BONUS = ["z_rank", "n_zones"]

zf = pd.read_parquet(CACHE / "zone_flags.parquet")
mine = pd.read_parquet(OUT / "mine_all.parquet")
assert not zf.duplicated(KEY).any() and not mine.duplicated(KEY).any()
m = zf.merge(mine, on=KEY, how="outer", suffixes=("_e", "_m"), indicator=True)
print("join:", m["_merge"].value_counts().to_dict())
m = m[m["_merge"] == "both"].copy()


def rates(d: pd.DataFrame, cols) -> dict:
    return {c: {"match": int((d[f"{c}_e"] == d[f"{c}_m"]).sum()), "n": len(d),
                "rate": float((d[f"{c}_e"] == d[f"{c}_m"]).mean())} for c in cols}


# ---- 400-signal sample (rows of zone_flags in stored order) ----
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(zf), size=400, replace=False))
samp = zf.iloc[idx][KEY].merge(m, on=KEY, how="left")
res = {"sample": {"n": len(samp), "by_period": samp["period"].value_counts().to_dict(),
                  "by_pat": samp["pat"].value_counts().to_dict(),
                  "engine_z_hit_true": int(samp["z_hit_e"].sum()), "engine_pdl_true": int(samp["pdl_hit_e"].sum()),
                  "rates": rates(samp, REQ + BONUS)}}
samp.to_csv(OUT / "sample400.csv", index=False)

# ---- full population ----
res["full"] = {"n": len(m), "rates": rates(m, REQ + BONUS)}
res["full"]["z_sweep_guess_rate"] = float((m["z_sweep"] == m["z_sweep_guess"]).mean())
res["full"]["multi_zone_touch"] = int((m["n_hit"] > 1).sum())
res["full"]["t_index_matches_ts"] = float(m["t_idx_ok"].mean())

# ---- power checks: how plausible wrong implementations would score against the engine ----
pw = {}
for name, col in [("approx_bars(12:30/14:30)", "v_approx_hit"), ("halfday_trim", "v_trim_hit"),
                  ("patternB_L=low[t]", "v_bt_hit")]:
    d = m if name != "patternB_L=low[t]" else m[m["pat"] == "B"]
    pw[name] = {"z_hit_match": float((d["z_hit_e"] == d[col]).mean()), "n_diff": int((d["z_hit_e"] != d[col]).sum()),
                "n": len(d)}
pw["halfday_trim_str"] = {"z_str_match": float((m["z_str_e"] == m["v_trim_str"]).mean())}
pw["halfday_trim_near"] = {"z_near_match": float((m["z_near_e"] == m["v_trim_near"]).mean())}
ah = m["mins"] >= 960
pw["pdl_same_day_for_19h"] = {"pdl_match_19h": float((m.loc[ah, "pdl_hit_e"] == m.loc[ah, "v_pdl_same"]).mean()),
                              "n_19h": int(ah.sum()), "n_diff": int((m.loc[ah, "pdl_hit_e"] != m.loc[ah, "v_pdl_same"]).sum())}
pw["pdl_halfday_trim"] = {"pdl_match": float((m["pdl_hit_e"] == m["v_pdl_trim"]).mean()),
                          "n_diff": int((m["pdl_hit_e"] != m["v_pdl_trim"]).sum())}
res["power_checks"] = pw

# ---- mismatch classes (full population) ----
mis = {}
for c in REQ + BONUS:
    d = m[m[f"{c}_e"] != m[f"{c}_m"]]
    mis[c] = {"n": len(d)}
    if len(d):
        cls = []
        for r in d.itertuples(index=False):
            tag = []
            if c == "z_hit":
                if getattr(r, "v_trim_hit") == getattr(r, "z_hit_e"):
                    tag.append("explained_by_halfday_trim")
                if getattr(r, "v_approx_hit") == getattr(r, "z_hit_e"):
                    tag.append("explained_by_approx_bars")
                if r.pat == "B" and getattr(r, "v_bt_hit") == getattr(r, "z_hit_e"):
                    tag.append("explained_by_patB_low_t")
            if c == "pdl_hit":
                if getattr(r, "v_pdl_same") == getattr(r, "pdl_hit_e"):
                    tag.append("explained_by_pdl_same_day")
                if getattr(r, "v_pdl_trim") == getattr(r, "pdl_hit_e"):
                    tag.append("explained_by_pdl_trim")
            cls.append("+".join(tag) or "unexplained")
        d = d.assign(cls=cls)
        mis[c]["classes"] = d["cls"].value_counts().to_dict()
        cols = KEY + [f"{c}_e", f"{c}_m", "L", "C", "zlo", "zhi", "ref", "dec_day", "pdl", "mins"]
        mis[c]["examples"] = d[cols].head(8).astype(str).to_dict("records")
        d[KEY + [f"{c}_e", f"{c}_m", "cls"]].to_csv(OUT / f"mismatch_{c}.csv", index=False)
res["mismatch_full"] = mis
# mismatches inside the sample
res["sample"]["mismatch_rows"] = {c: samp.loc[samp[f"{c}_e"] != samp[f"{c}_m"], KEY].astype(str).to_dict("records")
                                  for c in REQ}

(OUT / "s2_compare.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str)[:6000])
