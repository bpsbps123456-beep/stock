"""재계산(recheck_loo_K_R.csv, recheck_lead_K_R.csv)과 실행 결과 tests/lead csv 비교."""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parent
TESTS = {("KOSPI", "8"): "run_KOSPI_8_tests.csv", ("KOSPI", "12"): "run_KOSPI_12/tests.csv",
         ("SKHYNIX", "8"): "run_SKHYNIX_8/tests.csv", ("SKHYNIX", "12"): "run_SKHYNIX_12_tests.csv",
         ("SOXX", "8"): "run_SOXX_8_tests.csv", ("SOXX", "12"): "run_SOXX_12_tests.csv"}
LEADS = {("KOSPI", "8"): "run_KOSPI_8_lead.csv", ("KOSPI", "12"): "run_KOSPI_12/lead.csv",
         ("SKHYNIX", "8"): "run_SKHYNIX_8/lead.csv", ("SKHYNIX", "12"): "run_SKHYNIX_12_lead.csv",
         ("SOXX", "8"): "run_SOXX_8_lead.csv", ("SOXX", "12"): "run_SOXX_12_lead.csv"}


def side_col(d):
    c = "kind" if "kind" in d.columns else "contrast"
    s = d[c].astype(str)
    return np.where(s.str.contains("bottom|바닥"), "bottom", "top")


def load_rep():
    out = []
    for (a, x), f in TESTS.items():
        d = pd.read_csv(RES / f)
        d["side"] = side_col(d); d["asset"] = a; d["X"] = x
        out.append(d[["asset", "X", "side", "feature", "offset", "n_events", "n_fakes", "ba", "p", "mean_z_events", "mean_z_fakes",
                      "ba_all_fake_days_run_groups", "candidate", "status_feature", "leak_suspect"]])
    return pd.concat(out)


def load_rep_lead():
    out = []
    for (a, x), f in LEADS.items():
        d = pd.read_csv(RES / f)
        d["lead_side"] = np.where(d["side"].astype(str).str.contains("상위"), "top10", "bot10")
        d["side"] = side_col(d); d["asset"] = a; d["X"] = x
        out.append(d[["asset", "X", "side", "feature", "lead_side", "n_events", "n_fakes", "rate_events", "rate_fakes", "p",
                      "rate_placebo_mean", "rate_placebo_q975", "p_vs_placebo", "candidate"]])
    return pd.concat(out)


if __name__ == "__main__":
    mine = pd.read_csv(OUT / "recheck_loo_K_R.csv", dtype={"X": str})
    rep = load_rep(); rep["X"] = rep["X"].astype(str)
    summ = []
    for tag in ["K", "R"]:
        m = mine[mine.tag == tag].merge(rep, on=["asset", "X", "side", "feature", "offset"], suffixes=("", "_rep"))
        m = m[m.ba_P.notna() & m.ba.notna()]
        for (a, x), g in m.groupby(["asset", "X"]):
            summ.append(dict(tag=tag, combo=f"{a}_{x}", cells=len(g),
                             same_n=float(((g.n_events == g.n_events_rep) & (g.n_fakes == g.n_fakes_rep)).mean()),
                             ba_equal_P=float(np.isclose(g.ba_P, g.ba).mean()), ba_equal_J=float(np.isclose(g.ba_J, g.ba).mean()),
                             rho_p_P=float(g[["p_P", "p"]].corr("spearman").iloc[0, 1]),
                             rho_meanz=float(np.corrcoef(g.mean_z_events, g.mean_z_events_rep)[0, 1]),
                             mean_abs_dz=float((g.mean_z_events - g.mean_z_events_rep).abs().mean()),
                             p05_rep=int((g.p < 0.05).sum()), p05_mine_P=int((g.p_P < 0.05).sum()), p05_mine_J=int((g.p_J < 0.05).sum()),
                             p05_both_P=int(((g.p < 0.05) & (g.p_P < 0.05)).sum())))
        m.to_csv(OUT / f"compare_loo_{tag}.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "compare_summary.csv", index=False, encoding="utf-8-sig")
    print(s.to_string())
    # 앞섬
    ml = pd.read_csv(OUT / "recheck_lead_K_R.csv", dtype={"X": str})
    rl = load_rep_lead(); rl["X"] = rl["X"].astype(str)
    for tag in ["K", "R"]:
        m = ml[ml.tag == tag].merge(rl, on=["asset", "X", "side", "feature", "lead_side"], suffixes=("", "_rep"))
        m = m[m.rate_events.notna() & m.rate_events_rep.notna()]
        m.to_csv(OUT / f"compare_lead_{tag}.csv", index=False, encoding="utf-8-sig")
        print(tag, "lead cells", len(m), "rate_events equal", float(np.isclose(m.rate_events, m.rate_events_rep).mean()),
              "rho p", float(m[["p_fake_perm", "p"]].corr("spearman").iloc[0, 1]))
