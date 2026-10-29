"""(d) 비용 전 R1(E2·E0)이 0, 같은 분 다른 종목, 맥락-신호와 구별되는가 — 사전 등록 4칸, 뒤 구간 중심(앞·전체도)."""
import json
from common import *


def run():
    out = {}
    for name in CELLS:
        sig, ctx = sig_ctx(name)
        sig = sig.assign(d_bt_e2=sig.r1_e2 - sig.bt_r1_e2, d_bt_e0=sig.r1_e0 - sig.bt_r1_e0,
                         d_bv_e2=sig.r1_e2 - sig.bv_r1_e2, d_btnet=sig.r1net_e2 - sig.bt_r1net_e2)
        res = {}
        for per, msk_s, msk_c in (("front", sig.day <= SPLIT, ctx.day <= SPLIT), ("back", sig.day > SPLIT, ctx.day > SPLIT),
                                  ("all", sig.day > 0, ctx.day > 0)):
            S, C = sig[msk_s], ctx[msk_c]
            r = {"n": len(S), "days": int(S.day.nunique())}
            for col in ("r1_e2", "r1_e0", "d_bt_e2", "d_bt_e0", "d_bv_e2", "bt_r1_e2", "bv_r1_e2", "r1net_e2", "d_btnet"):
                m, se, t, n, g = cluster_t(S, col)
                pt, p, lo, hi = day_boot([S], lambda d, w, days, c=col: wmean(d[0], c, w, days), B=2000, seed=7)
                r[col] = dict(mean=m, se=se, t=t, p_boot_gt0=p, ci95=(lo, hi))
            for col in ("r1_e2", "r1_e0", "race_e2"):
                f = lambda d, w, days, c=col: wmean(d[0], c, w, days) - wmean(d[1], c, w, days)
                pt, p, lo, hi = day_boot([S, C], f, B=2000, seed=11)
                r[f"vs_ctx_{col}"] = dict(diff=pt, p_boot_gt0=p, ci95=(lo, hi), ctx_mean=float(C[col].mean()), ctx_n=len(C))
            res[per] = r
        out[name] = res
    # Holm(4칸) — 뒤 구간, 비용 전 R1 > 0, 비용 전 R1 − 같은 분 > 0
    for key in ("r1_e2", "d_bt_e2", "r1_e0"):
        ps = {k: out[k]["back"][key]["p_boot_gt0"] for k in CELLS}
        out[f"holm_back_{key}"] = V.holm(ps)
    ps = {k: out[k]["back"]["vs_ctx_r1_e2"]["p_boot_gt0"] for k in CELLS}
    out["holm_back_vs_ctx_r1_e2"] = V.holm(ps)
    json.dump(out, open(os.path.join(OUT, "d_gross.json"), "w"), ensure_ascii=False, indent=1, default=float)
    for k in CELLS:
        b = out[k]["back"]
        print(k, "n", b["n"], "| R1e2 %.4f%% t=%.2f p=%.3f" % (b["r1_e2"]["mean"] * 100, b["r1_e2"]["t"], b["r1_e2"]["p_boot_gt0"]),
              "| R1e0 %.4f%% t=%.2f" % (b["r1_e0"]["mean"] * 100, b["r1_e0"]["t"]),
              "| -bt %.4f%%p t=%.2f p=%.3f" % (b["d_bt_e2"]["mean"] * 100, b["d_bt_e2"]["t"], b["d_bt_e2"]["p_boot_gt0"]),
              "| bt %.4f%%" % (b["bt_r1_e2"]["mean"] * 100),
              "| -ctx %.4f%%p p=%.3f" % (b["vs_ctx_r1_e2"]["diff"] * 100, b["vs_ctx_r1_e2"]["p_boot_gt0"]),
              "| bv %.4f%%" % (b["bv_r1_e2"]["mean"] * 100))
        for per in ("front", "all"):
            f = out[k][per]
            print("   ", per, "n", f["n"], "R1e2 %.4f%% t=%.2f | -bt %.4f%%p t=%.2f | -ctx %.4f%%p p=%.3f" % (
                f["r1_e2"]["mean"] * 100, f["r1_e2"]["t"], f["d_bt_e2"]["mean"] * 100, f["d_bt_e2"]["t"],
                f["vs_ctx_r1_e2"]["diff"] * 100, f["vs_ctx_r1_e2"]["p_boot_gt0"]))
    for key in ("holm_back_r1_e2", "holm_back_d_bt_e2", "holm_back_r1_e0", "holm_back_vs_ctx_r1_e2"):
        print(key, {k: round(v, 3) for k, v in out[key].items()})


if __name__ == "__main__":
    run()
