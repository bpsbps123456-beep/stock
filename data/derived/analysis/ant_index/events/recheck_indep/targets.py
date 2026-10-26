"""후보(조합마다 상위 3개, 없으면 가장 가까운 3개) 칸만 뽑아 실행값과 재계산값을 나란히 놓는다."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
LOO = [
    ("KOSPI", "8", "bottom", "vw_top10_share", 0, "근접"), ("KOSPI", "8", "bottom", "tp_KX_A", 1, "근접"), ("KOSPI", "8", "bottom", "pr_P_strong1", 0, "근접"),
    ("KOSPI", "8", "top", "act_share_night", 1, "근접"), ("KOSPI", "8", "top", "act_new30_share", 0, "근접"), ("KOSPI", "8", "top", "tp_MEM_A", 1, "근접"),
    ("KOSPI", "12", "bottom", "vw_ln_b_median", 1, "후보"), ("KOSPI", "12", "bottom", "act_ln_saved_per_h", 0, "근접"), ("KOSPI", "12", "bottom", "act_comments_per_post", 0, "근접"),
    ("KOSPI", "12", "top", "act_new30_share", 1, "후보"), ("KOSPI", "12", "top", "act_share_night", 1, "후보"), ("KOSPI", "12", "top", "vw_up_gini", 0, "후보"),
    ("SKHYNIX", "8", "bottom", "act_ln_saved_per_h", 0, "앞섬후보의 가르기"), ("SKHYNIX", "8", "bottom", "act_ln_saved_per_h", 1, "앞섬후보의 가르기"),
    ("SKHYNIX", "8", "bottom", "tp_MEM_A", 0, "앞섬후보의 가르기"), ("SKHYNIX", "8", "bottom", "tp_MEM_A", 1, "앞섬후보의 가르기"),
    ("SKHYNIX", "8", "bottom", "act_ln_posts_per_h", 0, "앞섬후보의 가르기"), ("SKHYNIX", "8", "bottom", "act_ln_posts_per_h", 1, "앞섬후보의 가르기"),
    ("SKHYNIX", "8", "top", "vw_top10_share", 1, "후보"), ("SKHYNIX", "8", "top", "act_ln_comments_pre_per_h", 1, "후보"), ("SKHYNIX", "8", "top", "pr_P_share", 0, "후보"),
    ("SKHYNIX", "12", "bottom", "tp_LS_A", 0, "후보"),
    ("SKHYNIX", "12", "top", "tp_ANG_A", 0, "후보"), ("SKHYNIX", "12", "top", "vw_ln_b_median", 1, "후보"), ("SKHYNIX", "12", "top", "act_new30_share", 1, "후보"),
    ("SOXX", "8", "bottom", "tp_LL_US_A", 0, "후보"), ("SOXX", "8", "bottom", "act_share_night", 1, "후보"), ("SOXX", "8", "bottom", "pr_P_share", 1, "후보"),
    ("SOXX", "8", "top", "pr_P_bar", 0, "근접"), ("SOXX", "8", "top", "pr_P_share", 1, "근접"), ("SOXX", "8", "top", "tp_LS_A", 0, "근접"),
    ("SOXX", "12", "bottom", "tp_LL_US_A", 1, "후보"), ("SOXX", "12", "bottom", "tp_MC_A", 0, "근접"), ("SOXX", "12", "bottom", "pr_P_share", 0, "근접"),
    ("SOXX", "12", "top", "tp_LS_A", 0, "근접"), ("SOXX", "12", "top", "vw_view_gini", 1, "근접"), ("SOXX", "12", "top", "em_mean_FEAR", 1, "근접(설명용)"),
]
LEAD = [("SKHYNIX", "8", "bottom", "act_ln_saved_per_h"), ("SKHYNIX", "8", "bottom", "tp_MEM_A"), ("SKHYNIX", "8", "bottom", "act_ln_posts_per_h"),
        ("SKHYNIX", "12", "bottom", "tp_MC_A"), ("SKHYNIX", "12", "bottom", "act_ln_saved_per_h")]

if __name__ == "__main__":
    K = pd.read_csv(OUT / "compare_loo_K.csv", dtype={"X": str}); R = pd.read_csv(OUT / "compare_loo_R.csv", dtype={"X": str})
    rows = []
    for a, x, s, f, o, role in LOO:
        k = K[(K.asset == a) & (K.X == x) & (K.side == s) & (K.feature == f) & (K.offset == o)]
        r = R[(R.asset == a) & (R.X == x) & (R.side == s) & (R.feature == f) & (R.offset == o)]
        k = k.iloc[0]; r = r.iloc[0]
        rows.append(dict(combo=f"{a}_{x}", side=s, cell=f"{f}@{'+' if o else ''}{o}", role=role,
                         rep_n=f"{int(k.n_events_rep)}/{int(k.n_fakes_rep)}", rep_ba=round(k.ba, 3), rep_p=round(k.p, 3),
                         n=f"{k.n_events}/{k.n_fakes}",
                         KP_hit=f"{k.hitE_P}/{k.n_events}·{k.hitF_P}/{k.n_fakes}", KP_ba=round(k.ba_P, 3), KP_p=round(k.p_P, 3),
                         KJ_ba=round(k.ba_J, 3), KJ_p=round(k.p_J, 3),
                         RP_hit=f"{r.hitE_P}/{r.n_events}·{r.hitF_P}/{r.n_fakes}", RP_ba=round(r.ba_P, 3), RP_p=round(r.p_P, 3),
                         KP_allfake_ba=round(k.ba_all_fake_days, 3) if pd.notna(k.ba_all_fake_days) else np.nan,
                         rep_allfake_ba=round(k.ba_all_fake_days_run_groups, 3) if pd.notna(k.ba_all_fake_days_run_groups) else np.nan,
                         mz=f"{k.mean_z_events:+.2f}/{k.mean_z_fakes:+.2f}", rep_mz=f"{k.mean_z_events_rep:+.2f}/{k.mean_z_fakes_rep:+.2f}",
                         miss_events=k.miss_events if isinstance(k.miss_events, str) else "", miss_fakes=k.miss_fakes if isinstance(k.miss_fakes, str) else ""))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "targets_loo.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.drop(columns=["miss_events", "miss_fakes"]).to_string())
    print(t[["combo", "side", "cell", "miss_events", "miss_fakes"]].to_string())
    LK = pd.read_csv(OUT / "compare_lead_K.csv", dtype={"X": str}); LR = pd.read_csv(OUT / "compare_lead_R.csv", dtype={"X": str})
    lr = []
    for a, x, s, f in LEAD:
        k = LK[(LK.asset == a) & (LK.X == x) & (LK.side == s) & (LK.feature == f) & (LK.lead_side == "top10")].iloc[0]
        r = LR[(LR.asset == a) & (LR.X == x) & (LR.side == s) & (LR.feature == f) & (LR.lead_side == "top10")].iloc[0]
        lr.append(dict(combo=f"{a}_{x}", cell=f"{f} 상위10%",
                       rep=f"{k.rate_events_rep:.2f}({int(k.n_events_rep)}) / {k.rate_fakes_rep:.2f}({int(k.n_fakes_rep)}) / 위약 {k.rate_placebo_mean:.2f}",
                       rep_p=round(k.p, 3), rep_p_pl=round(k.p_vs_placebo, 4), rep_q975=round(k.rate_placebo_q975, 2),
                       K=f"{k.rate_events:.2f}({k.n_events}) / {k.rate_fakes:.2f}({k.n_fakes}) / 위약 {k.rate_placebo:.2f}",
                       K_p=round(k.p_fake_perm, 3), K_p_pl=round(k.p_vs_placebo, 4), K_q975=round(k.placebo_q975, 2),
                       R=f"{r.rate_events:.2f} / {r.rate_fakes:.2f} / 위약 {r.rate_placebo:.2f}", R_p=round(r.p_fake_perm, 3), R_p_pl=round(r.p_vs_placebo, 4)))
    l = pd.DataFrame(lr); l.to_csv(OUT / "targets_lead.csv", index=False, encoding="utf-8-sig")
    print(l.to_string())
