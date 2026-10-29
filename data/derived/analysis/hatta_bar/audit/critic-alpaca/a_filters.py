"""(a) 위생 거르기·겹침 빼기가 좋은 신호를 지웠는가 + (c) 격자 고르기 통계 진단."""
from common import *


def summ(X):
    if len(X) == 0:
        return {"n": 0}
    d = {"n": int(len(X)), "r1_e2": float(X.r1_e2.mean()), "r1_e0": float(X.r1_e0.mean()), "hit_e2": float(X.hit_e2.mean()),
         "d_bt": float((X.r1_e2 - X.bt_r1_e2).mean()), "r1net_e2": float(X.r1net_e2.mean())}
    if len(X) > 30 and X.day.nunique() > 2:
        d["t_r1"] = cluster_t(X, "r1_e2")[2]
        d["t_dbt"] = cluster_t(X.assign(x=X.r1_e2 - X.bt_r1_e2), "x")[2]
    return d


def main():
    out = {}
    sp = specs()
    for name in CELLS:
        pat, spec = sp[name]
        X = V.table("cand_alp1m", pat, spec)
        sm, cm = V.masks(X, pat, spec)
        S = X[sm & X.tradable.to_numpy()]
        back = S.day > SPLIT
        r = {}
        # 위생 통과/탈락(각각 겹침 빼기)
        r["sane_back"] = summ(V.dedup_rows(S[back & S.sane]))
        r["insane_back"] = summ(V.dedup_rows(S[back & ~S.sane]))
        # 겹침: 빼기 전 전체, 남긴 것, 뺀 것
        Ss = S[back & S.sane]
        kept = V.dedup_rows(Ss)
        dropped = Ss.drop(index=kept.index)
        r["nodedup_back"] = summ(Ss)
        r["dedup_kept_back"] = summ(kept)
        r["dedup_dropped_back"] = summ(dropped)
        out[name] = r
        print(name, {k: {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in r.items()})
    # (c) 격자 진단
    diag = {}
    for pat in ("A", "B"):
        g = pd.read_parquet(os.path.join(RES, f"grid_{pat}.parquet"),
                            columns=["f_n", "f_t", "f_r1", "f_lift", "f_r1net", "b_r1", "b_t", "b_n", "b_lift", "elig"])
        g = g[g.elig]
        bvf = g.f_r1 - g.f_lift
        diag[pat] = {
            "eligible": int(len(g)),
            "share_f_t_gt_3": float((g.f_t > 3).mean()),
            "share_f_lift_gt_0": float((g.f_lift > 0).mean()),
            "f_lift_mean": float(g.f_lift.mean()), "f_lift_sd": float(g.f_lift.std()),
            "f_r1_mean": float(g.f_r1.mean()), "f_r1_sd": float(g.f_r1.std()),
            "f_bv_mean": float(bvf.mean()), "f_bv_sd": float(bvf.std()),
            "spearman_ft_sqrtn": float(g.f_t.corr(np.sqrt(g.f_n), method="spearman")),
            "spearman_ft_fr1": float(g.f_t.corr(g.f_r1, method="spearman")),
            "spearman_ft_minus_bv": float(g.f_t.corr(-bvf, method="spearman")),
            "spearman_flift_fr1": float(g.f_lift.corr(g.f_r1, method="spearman")),
            "spearman_flift_minus_bv": float(g.f_lift.corr(-bvf, method="spearman")),
            "spearman_fr1_br1": float(g.f_r1.corr(g.b_r1, method="spearman")),
            "spearman_ft_bt": float(g.f_t.corr(g.b_t, method="spearman")),
            "spearman_bt_sqrtbn": float(g.b_t.corr(np.sqrt(g.b_n), method="spearman")),
            "share_var_flift_from_bv": float(np.var(bvf) / np.var(g.f_lift)),
        }
        print(pat, {k: round(v, 4) for k, v in diag[pat].items()})
    out["grid_diag"] = diag
    json.dump(out, open(os.path.join(OUT, "a_filters.json"), "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
