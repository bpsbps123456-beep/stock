"""시운전 병렬판(run.py의 dry와 같은 절차, 프로세스 10개). 고정 파일(run.py·harness.py)은 그대로 가져다 쓴다.

난수: 무관 대리 b는 seed 20261026 + b, 효과 심기는 seed 7_000_000 + 1000·k + b(k = ρ 순번).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import harness as hz
from run import PCT_SIGS, build, planted, setup

RES = Path(__file__).resolve().parents[1] / "results"
RHOS = [0.1, 0.2, 0.35, 0.5]
_S = {}


def _init():
    raw, px, entries, front, back, sd_front, steps, stepped = setup()
    _S.update(raw=raw, px=px, entries=entries, front=front, back=back, sd=sd_front, stepped=stepped,
              cells=hz.all_cells(), X=raw[PCT_SIGS + ["K1"]].copy(),
              cens=raw[[c for c in raw.columns if c.startswith("cens_")]], phi=float(raw["S1"].autocorr(1)))


def _surr(rng):
    raw, X = _S["raw"], _S["X"]
    S = hz.surrogate(X, rng)
    S["K1"] = (S["K1"] >= S["K1"].quantile(1 - raw["K1"].mean())).astype(float).where(S["K1"].notna())
    return S


def null_task(b):
    rng = np.random.default_rng(20261026 + b)
    S = _surr(rng).join(_S["cens"])
    pct, ons = build(S)
    r = hz.judge(_S["cells"], ons, _S["front"], _S["back"], _S["sd"])
    fam = hz.family_null(_S["cells"], ons, _S["px"], _S["entries"], hz.FRONT, hz.BACK, _S["sd"], step=8,
                         stepped=_S["stepped"]) if r["n_pass_explore"] > 0 else np.array([])
    p_fam = float((1 + (fam >= r["n_pass_explore"]).sum()) / (1 + len(fam))) if len(fam) else np.nan
    pr = r["primary"]
    return {"b": b, "primary_pass": r["n_pass_primary"], "explore_pass": r["n_pass_explore"], "p_family": p_fam,
            "concl_distinct": bool(r["n_pass_primary"] > 0 or (r["n_pass_explore"] > 0 and p_fam < 0.05)),
            "primary_p_min": float(np.nanmin(pr["p"])) if pr["p"].notna().any() else np.nan,
            "primary_judgeable": int(pr["judgeable"].sum()),
            "explore_judgeable": int(r["explore"]["judgeable"].sum()),
            "explore_raw_p_lt05": int(((r["explore"]["p"] < 0.05) & r["explore"]["judgeable"]).sum())}


def plant_task(arg):
    k, b = arg
    rho = RHOS[k]
    rng = np.random.default_rng(7_000_000 + 1000 * k + b)
    S = _surr(rng)
    S["S1"] = planted(_S["raw"], _S["px"], _S["entries"], rho, rng, _S["phi"]).where(_S["raw"]["S1"].notna())
    S = S.join(_S["cens"])
    pct, ons = build(S)
    r = hz.judge(_S["cells"], ons, _S["front"], _S["back"], _S["sd"])
    pr = r["primary"]
    row = pr[(pr["signal"] == "S1") & (pr["outcome"] == "R5")].iloc[0]
    ex = r["explore"]
    return {"rho": rho, "b": b, "primary_S1R5_pass": bool(row["pass"]), "p_raw": float(row["p"]),
            "p_holm": float(row["p_holm"]) if pd.notna(row["p_holm"]) else np.nan,
            "judgeable": bool(row["judgeable"]),
            "explore_has_S1": bool((ex["signal"] == "S1").any()),
            "explore_S1_pass": bool(((ex["signal"] == "S1") & ex["pass"]).any())}


def main(B=200, P=100):
    RES.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=10, initializer=_init) as ex:
        nul = pd.DataFrame(list(ex.map(null_task, range(B))))
        nul.to_csv(RES / "dryrun_null.csv", index=False)
        print("null done", nul[["primary_pass", "explore_pass", "concl_distinct"]].mean().to_dict(), flush=True)
        pl = pd.DataFrame(list(ex.map(plant_task, [(k, b) for k in range(len(RHOS)) for b in range(P)])))
        pl.to_csv(RES / "dryrun_planted.csv", index=False)
    summ = {
        "B_null": B, "P_planted_per_rho": P, "family_step": 8,
        "null_rate_primary_any_pass": float((nul["primary_pass"] > 0).mean()),
        "null_rate_explore_any_pass": float((nul["explore_pass"] > 0).mean()),
        "null_rate_conclusion_distinct": float(nul["concl_distinct"].mean()),
        "null_mean_primary_judgeable": float(nul["primary_judgeable"].mean()),
        "null_mean_explore_judgeable": float(nul["explore_judgeable"].mean()),
        "null_rate_explore_raw_p_lt05_per_judgeable_cell": float(nul["explore_raw_p_lt05"].sum() / max(1, nul["explore_judgeable"].sum())),
        "planted": pl.groupby("rho")[["judgeable", "primary_S1R5_pass", "explore_has_S1", "explore_S1_pass"]].mean().round(3).to_dict(),
        "planted_median_p_raw": pl.groupby("rho")["p_raw"].median().round(3).to_dict(),
    }
    (RES / "dryrun_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
