"""recompute 결과 집계: 기준선 분해, 대안 기준선, 엄격 체결, 비용 민감도, 무조건 기준, 유동성 3분위(탐색)."""
from common import *

R = pd.read_csv(os.path.join(OUT, "recompute_signals.csv"))
U = pd.read_parquet(os.path.join(OUT, "uncond_bars.parquet"))
out = {}
print("bv 재현 최대 오차", float((R.bv_csv - R.bv_re).abs().max()))
B = R[R.day > SPLIT]
res = {}
for nm in CELLS:
    X = B[B.cell == nm].copy()
    r = {"n": len(X), "r1": X.r1_e2.mean(), "bv": X.bv_csv.mean(), "ter_top_share": float((X.ter == 2).mean())}
    for k in ("pre_done", "pre_overlap", "self", "post"):
        w = X[f"bvn_{k}"]
        r[f"bv_{k}"] = float(X[f"bv_{k}"].mean())
        r[f"bvn_{k}_mean"] = float(w.mean())
    # 신호마다 묶음 안 앞(끝남+겹침) 봉 비중
    tot = X[["bvn_pre_done", "bvn_pre_overlap", "bvn_self", "bvn_post"]].sum(1)
    r["share_pre_bars"] = float(((X.bvn_pre_done + X.bvn_pre_overlap) / tot).mean())
    r["bv_post_only_r1_minus"] = float((X.r1_e2 - X.bv_post).mean())
    # 과거 20일 ATR 맞춤
    Y = X.dropna(subset=["pv_r1"])
    r["pv_cov"] = len(Y) / len(X)
    r["pv_r1"] = float(Y.pv_r1.mean())
    r["ps_r1"] = float(X.ps_r1.mean())
    m, se, t, n, g = cluster_t(Y.assign(x=Y.r1_e2 - Y.pv_r1), "x")
    r["r1_minus_pv"] = m
    r["t_r1_minus_pv"] = t
    # 엄격 체결
    for c in ("r1_tick", "r1_011"):
        m, se, t, n, g = cluster_t(X, c)
        r[c] = m
        r[f"t_{c}"] = t
    r["strict_drop_tick"] = float((X.r1_e2 - X.r1_tick).mean())
    res[nm] = r
    print(nm, {k: (round(v * 100, 4) if isinstance(v, float) and abs(v) < 0.05 else (round(v, 3) if isinstance(v, float) else v)) for k, v in r.items()})
out["baseline_fill"] = res

# 비용 민감도 — 신호마다 진입·청산 봉의 실제 시간대로
def net(X, cost):
    ci = np.array(cost)[X.sess_e.to_numpy()]
    co = np.array(cost)[X.sess_z.to_numpy()]
    return np.where(X.hit.to_numpy() > 0.5, X.r1_e2 - ci, X.r1_e2 - ci - co)

sched = {"brief(0.3/0.1/0.05/0.3)": [0.003, 0.001, 0.0005, 0.003], "half": [0.0015, 0.0005, 0.00025, 0.0015],
         "flat5bp": [0.0005] * 4, "flat2bp": [0.0002] * 4, "zero": [0.0] * 4}
cs = {}
for nm in CELLS:
    X = B[B.cell == nm]
    d = {}
    for k, c in sched.items():
        x = X.assign(nn=net(X, c))
        m, se, t, n, g = cluster_t(x, "nn")
        d[k] = (round(m * 100, 4), round(t, 2))
    miss = 1 - X.hit.mean()
    d["break_even_flat_oneway_%"] = round(X.r1_e2.mean() / (1 + miss) * 100, 4)
    # 시간대 구성
    d["entry_sess_share"] = {E.SESS_NAMES[s]: round(float((X.sess_e == s).mean()), 3) for s in range(4)}
    cs[nm] = d
    print(nm, d)
out["cost"] = cs

# 무조건 기준(판정 모집단 봉 전체)
for per, m in (("front", U.day <= SPLIT), ("back", U.day > SPLIT)):
    Z = U[m]
    mm, se, t, n, g = cluster_t(Z, "r1_e2")
    print("uncond", per, "n", n, "R1e2 %.4f%% t=%.2f" % (mm * 100, t), "| net %.4f%%" % (Z.r1net_e2.mean() * 100),
          {E.SESS_NAMES[s]: round(Z[Z.sess == s].r1_e2.mean() * 100, 4) for s in range(4)},
          "| ATR top tercile %.4f%%" % (Z[Z.ter == 2].r1_e2.mean() * 100))
    out[f"uncond_{per}"] = {"n": n, "r1": mm, "t": t, "r1net": float(Z.r1net_e2.mean()),
                            "by_sess": {E.SESS_NAMES[s]: float(Z[Z.sess == s].r1_e2.mean()) for s in range(4)},
                            "ter_top": float(Z[Z.ter == 2].r1_e2.mean())}

# 유동성 3분위(앞 구간 정규장 분당 거래대금 중앙값, 종목 단위) — 탐색: 3조각 × 4칸 = 12조각
liq = R.drop_duplicates("sid").set_index("sid").liq
q = liq.quantile([1 / 3, 2 / 3]).to_numpy()
lt = {}
for nm in CELLS:
    X = B[B.cell == nm]
    sig = pd.read_csv(os.path.join(RES, f"signals_{nm}.csv"), usecols=["sid", "ts", "bt_r1_e2"])
    X = X.merge(sig, on=["sid", "ts"], how="left")
    b = np.digitize(X.liq, q)
    d = {}
    for k in range(3):
        Y = X[b == k]
        mm, se, t, n, g = cluster_t(Y, "r1_e2")
        m2, se2, t2, _, _ = cluster_t(Y.assign(x=Y.r1_e2 - Y.bt_r1_e2), "x")
        be = Y.r1_e2.mean() / (2 - Y.hit.mean())
        d[["low", "mid", "high"][k]] = dict(n=n, r1=round(mm * 100, 4), t=round(t, 2), d_bt=round(m2 * 100, 4), t_dbt=round(t2, 2),
                                             break_even_oneway=round(be * 100, 4))
    lt[nm] = d
    print(nm, d)
out["liq_tercile_exploratory"] = {"cut_usd_per_min": q.tolist(), "cells": lt}
json.dump(out, open(os.path.join(OUT, "analyze.json"), "w"), ensure_ascii=False, indent=1, default=float)
