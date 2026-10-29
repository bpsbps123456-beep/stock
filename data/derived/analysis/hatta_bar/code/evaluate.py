"""주 가설 4개(원안 ①·②, 앞 1위 ①·②) 판정과 보조 보고.
사용: py evaluate.py            → results/primary.json, results/eval_tables.json, results/signals_<칸>.csv"""
import os, sys, json
import numpy as np
import pandas as pd
import engine as E
import grid as G

HERE = os.path.dirname(os.path.abspath(__file__))
RES = G.RES
SPLIT = 20260831
B_BOOT = 2000
ORIG = {
    "A": dict(N=120, tol=0.0005, K="level3_2", ema="persist", D=0.02, v=0.0, rm="wick", r=0.5, m=1.0, above=False),
    "B": dict(N=120, tol=0.0005, K="k1", ema="persist", D=0.02, v=0.0, below=True, eng="basic"),
}


def kopt(name):
    if name == "k1":
        return ("k1", None, 1, 0)
    mode_p, k = name.split("_")
    mode = "touch" if mode_p.startswith("touch") else "level"
    return (name, mode, int(k), int(mode_p[len(mode):]))


def table(cand, pat, spec, plain=False):
    f = f"plain_{pat}_{int(spec['N'])}.parquet" if plain else f"{pat}_{int(spec['N'])}_{spec['tol']}.parquet"
    X = pd.read_parquet(os.path.join(E.CACHE, cand, f))
    return X


def masks(X, pat, spec, plain=False):
    """(신호, 맥락) 마스크. 맥락 = 이탈 정의·K·추세·거래량, 신호 = 맥락 & 봉 모양."""
    km = np.ones(len(X), bool) if plain else G.k_mask(X, kopt(spec["K"]))
    ctx = km & G.ema_mask(X, spec["ema"]) & (X["dd"].to_numpy() >= spec["D"])
    if spec["v"] > 0:
        ctx &= X["volr"].to_numpy() >= spec["v"]
    if pat == "A":
        sh = (spec["rm"], spec["r"], spec["m"], bool(spec["above"]))
        ctx &= G.context_ok(X, "A", sh)
        shape = G.shape_mask_A(X, sh)
    else:
        sh = (bool(spec["below"]), spec["eng"])
        ctx &= G.context_ok(X, "B", sh)
        shape = G.shape_mask_B(X, sh)
    return ctx & shape, ctx & ~shape


def dedup_rows(X):
    if len(X) == 0:
        return X
    X = X.sort_values(["sid", "ts"], kind="stable")
    keep = E.dedup(X.sid.to_numpy().astype(np.int64), X.ts.to_numpy().astype(np.int64), 30)
    return X[keep]


def pop(X, tradable=True):
    X = X[X.sane]
    return X[X.tradable] if tradable else X


def summary(X):
    if len(X) == 0:
        return {"n": 0}
    rc = X.race_e2.dropna()
    return {"n": int(len(X)), "days": int(X.day.nunique()), "syms": int(X.sid.nunique()),
            "hit_e2": X.hit_e2.mean(), "bv_hit_e2": X.bv_hit_e2.mean(), "hit_strict_e2": X.hits_e2.mean(),
            "r1_e2": X.r1_e2.mean(), "r1net_e2": X.r1net_e2.mean(), "bv_r1net_e2": X.bv_r1net_e2.mean(),
            "bt_r1net_e2": X.bt_r1net_e2.mean(), "race_e2": rc.mean() if len(rc) else np.nan,
            "race_resolved": len(rc) / len(X), "r2_e2": X.r2_e2.mean(), "amb_e2": X.amb_e2.mean(),
            "hit_e0": X.hit_e0.mean(), "r1_e0": X.r1_e0.mean(), "bv_hit_e0": X.bv_hit_e0.mean(),
            "mfe_e2": X.mfe_e2.median(), "mae_e2": X.mae_e2.median()}


def day_agg(X, cols):
    g = X.groupby("day")
    out = pd.DataFrame({"n": g.size()})
    for c in cols:
        v = X[c]
        out[f"s_{c}"] = v.fillna(0).groupby(X.day).sum()
        out[f"c_{c}"] = v.notna().groupby(X.day).sum()
    return out


def boot(sig, ctx, rng):
    """거래일 짝 부트스트랩: 단측 p값."""
    sig = sig.assign(d_bv=sig.r1net_e2 - sig.bv_r1net_e2, d_bt=sig.r1net_e2 - sig.bt_r1net_e2)
    a = day_agg(sig, ["r1net_e2", "d_bv", "d_bt", "race_e2"])
    b = day_agg(ctx, ["r1net_e2", "race_e2"])
    days = np.union1d(a.index.to_numpy(), b.index.to_numpy())
    a = a.reindex(days, fill_value=0)
    b = b.reindex(days, fill_value=0)

    def stats(w):
        def m(df, c):
            den = (df[f"c_{c}"].to_numpy() * w).sum()
            return (df[f"s_{c}"].to_numpy() * w).sum() / den if den > 0 else np.nan
        r_sig, r_ctx = m(a, "r1net_e2"), m(b, "r1net_e2")
        q_sig, q_ctx = m(a, "race_e2"), m(b, "race_e2")
        return np.array([r_sig, r_sig - r_ctx, m(a, "d_bv"), m(a, "d_bt"), q_sig - 0.5, q_sig - q_ctx])

    point = stats(np.ones(len(days)))
    draws = np.array([stats(np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)))
                      for _ in range(B_BOOT)])
    names = ["r1net", "vs_ctx", "vs_bv", "vs_bt", "race_minus_half", "race_vs_ctx"]
    p = {nm: float((1 + np.sum(~(draws[:, i] > 0))) / (B_BOOT + 1)) for i, nm in enumerate(names)}
    lo = {nm: float(np.nanpercentile(draws[:, i], 5)) for i, nm in enumerate(names)}
    return {"point": dict(zip(names, map(float, point))), "p": p, "lo5": lo, "days": int(len(days))}


def holm(ps):
    order = sorted(ps, key=lambda k: ps[k])
    m, out, run = len(ps), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, ps[k] * (m - i)))
        out[k] = run
    return out


def cell_rows(cand, pat, spec, plain=False, tradable=True):
    X = table(cand, pat, spec, plain)
    sm, cm = masks(X, pat, spec, plain)
    sig = dedup_rows(pop(X[sm], tradable))
    ctx = dedup_rows(pop(X[cm], tradable))
    return sig, ctx


def halt_share(sig, src="alp1m"):
    """정규장(09:30~16:00) 30분 창 안에 5분 이상 빈 곳이 있는 신호(거래 정지 의심) 몫과 그 신호의 R1."""
    from build import SYMS
    flags = np.zeros(len(sig), bool)
    for sid, idx in sig.groupby("sid").groups.items():
        d = E.load(SYMS[int(sid)], src)
        a = E.prep(d)
        for q in idx:
            r = sig.loc[q]
            t = int(r.t)
            w = a["ts"][t:t + 40]
            m = a["mins"][t:t + 40]
            w, m = w[w <= a["ts"][t] + 30], m[: (w <= a["ts"][t] + 30).sum()]
            reg = (m >= 570) & (m < 960)
            gaps = np.diff(w) if len(w) > 1 else np.array([])
            flags[sig.index.get_loc(q)] = bool(len(gaps) and np.any((gaps >= 5) & reg[1:]))
    return {"share": float(flags.mean()) if len(flags) else None,
            "r1net_flagged": float(sig[flags].r1net_e2.mean()) if flags.any() else None, "n_flagged": int(flags.sum())}


def spec_of(row, pat):
    keys = ["N", "tol", "K", "ema", "D", "v"] + (["rm", "r", "m", "above"] if pat == "A" else ["below", "eng"])
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in keys}


def main(cand="cand_alp1m"):
    rng = np.random.default_rng(20261029)
    picks = json.load(open(os.path.join(RES, "pick.json"), encoding="utf-8"))
    cells = {"orig_A": ("A", ORIG["A"]), "orig_B": ("B", ORIG["B"])}
    for pat in ("A", "B"):
        if picks.get(pat):
            cells[f"top_{pat}"] = (pat, spec_of(picks[pat], pat))
    primary, tables = {}, {}
    for name, (pat, spec) in cells.items():
        sig, ctx = cell_rows(cand, pat, spec)
        sig.to_csv(os.path.join(RES, f"signals_{name}.csv"), index=False)
        back_s, back_c = sig[sig.day > SPLIT], ctx[ctx.day > SPLIT]
        front_s = sig[sig.day <= SPLIT]
        res = {"spec": spec, "front": summary(front_s), "back": summary(back_s), "all": summary(sig),
               "ctx_back": summary(back_c), "boot_back": boot(back_s, back_c, rng)}
        res["p_hyp"] = max(res["boot_back"]["p"].values())
        primary[name] = res
        print(name, "back n", len(back_s), "p_hyp", round(res["p_hyp"], 4), flush=True)
    adj = holm({k: v["p_hyp"] for k, v in primary.items()})
    for k in primary:
        primary[k]["p_holm"] = adj[k]
        primary[k]["pass"] = bool(adj[k] < 0.05)
    json.dump(primary, open(os.path.join(RES, "primary.json"), "w"), ensure_ascii=False, indent=1, default=float)

    # 보조 보고
    for name, (pat, spec) in cells.items():
        allsess, _ = cell_rows(cand, pat, spec, tradable=False)
        t = {}
        for per, Y in (("front", allsess[allsess.day <= SPLIT]), ("back", allsess[allsess.day > SPLIT])):
            t[per] = {E.SESS_NAMES[s]: summary(Y[Y.sess == s]) for s in range(4)}
        sig = pd.read_csv(os.path.join(RES, f"signals_{name}.csv"))
        back = sig[sig.day > SPLIT]
        if len(back):
            contrib = back.groupby("day").r1net_e2.sum()
            worst = contrib.abs().idxmax()
            t["drop_biggest_day"] = {"day": int(worst), "r1net_e2": back[back.day != worst].r1net_e2.mean()}
            same = back.groupby("ts").size()
            t["cluster5_share"] = float(back.ts.map(same).ge(5).mean())
            mins = ((pd.to_datetime(back.ts * 60, unit="s", utc=True).dt.tz_convert(E.ET)).dt.hour * 60
                    + (pd.to_datetime(back.ts * 60, unit="s", utc=True).dt.tz_convert(E.ET)).dt.minute)
            t["open5_share"] = float(((mins >= 570) & (mins < 575)).mean())
            t["pre_first10_share"] = float(((mins >= 240) & (mins < 250)).mean())
            sess_key = back.day.astype(str) + "_" + back.sess.astype(str) + "_" + back.sid.astype(str)
            first = ~sess_key.duplicated()
            t["first_vs_later"] = {"first": summary(back[first.values]), "later": summary(back[~first.values])}
            t["morning_to_1029"] = summary(back[~((mins >= 630) & (mins < 660)).values])
            t["halt_window"] = halt_share(back)
        pl_sig, _ = cell_rows(cand, pat, spec, plain=True)
        t["plain_dip_back"] = summary(pl_sig[pl_sig.day > SPLIT])
        tables[name] = t
    # 순위가 이어지는가
    carry = {}
    for pat in ("A", "B"):
        g = pd.read_parquet(os.path.join(RES, f"grid_{pat}.parquet"))
        g = g[g.elig].sort_values("rank")
        carry[pat] = {"eligible": int(len(g)),
                      "ranks": {int(r): {k: float(g.iloc[r - 1][k]) for k in ("f_t", "f_n", "f_r1net", "b_n", "b_r1net",
                                                                            "b_liftnet", "b_tnet", "b_hit", "b_bvhit", "b_race")}
                                for r in (1, 5, 20, 100, 1000) if r <= len(g)},
                      "spearman_front_t_back_t": float(g.f_t.corr(g.b_t, method="spearman")),
                      "top20_median_b_r1net": float(g.head(20).b_r1net.median()),
                      "top20_median_b_liftnet": float(g.head(20).b_liftnet.median())}
    tables["carry"] = carry
    json.dump(tables, open(os.path.join(RES, "eval_tables.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main(*sys.argv[1:])
