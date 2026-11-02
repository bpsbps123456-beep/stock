"""3차 판정(BRIEF3.md): H1 = ② & QQQ 당일 < −1%, H2 = ① & VWAP 이격 < −3%를 처음 보는 2026 1~6월(주)·2025 1~6월(보조)로 확인한다.
사용: py h3_eval.py judge   → results/h3_primary.json, results/h3_tables.json, results/h3_signals.csv
      py h3_eval.py e1      → 주 확인 신호의 실측 호가 손익(h3_primary.json에 더함)"""
import os, sys, json
import numpy as np
import pandas as pd
import engine as E
import cond_eval as CE

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
B_BOOT = 2000
HYP = {
    "H1": dict(pat="B", col="qqq_day", thr=-0.01, label="② & QQQ 당일 < −1%"),
    "H2": dict(pat="A", col="vwap_gap", thr=-0.03, label="① & VWAP 이격 < −3%"),
}


def kst_awake(ts):
    h = pd.to_datetime(np.asarray(ts) * 60, unit="s", utc=True).tz_convert("Asia/Seoul").hour
    return np.asarray(h >= 8)


def dedup(X):
    X = X.sort_values(["sid", "ts"], kind="stable")
    return X[E.dedup(X.sid.to_numpy().astype(np.int64), X.ts.to_numpy().astype(np.int64), 30)]


def summ(X):
    if len(X) == 0:
        return {"n": 0}
    rc = X.race_e2.dropna()
    vb = (X.r1_e2 - X.bt2_r1).dropna()
    return {"n": int(len(X)), "days": int(X.day.nunique()), "syms": int(X.sid.nunique()), "hit": float(X.hit_e2.mean()),
            "r1": float(X.r1_e2.mean()), "r1n": float(X.r1n.mean()), "vs_bt": float(vb.mean()) if len(vb) else None,
            "race": float(rc.mean()) if len(rc) else None, "mfe_med": float(X.mfe_e2.median()), "mae_med": float(X.mae_e2.median())}


def day_boot(X, rng, cols=("r1n", "vsbt")):
    """거래일 묶음 부트스트랩 단측 p(평균 ≤ 0)와 묶음 표준오차."""
    X = X.assign(vsbt=X.r1_e2 - X.bt2_r1)
    days = np.sort(X.day.unique())
    pos = np.searchsorted(days, X.day.to_numpy())
    out = {}
    for c in cols:
        v = X[c].to_numpy()
        ok = ~np.isnan(v)
        s = np.bincount(pos[ok], weights=v[ok], minlength=len(days))
        n = np.bincount(pos[ok], minlength=len(days)).astype(float)
        m = s.sum() / n.sum()
        G = len(days)
        se = np.sqrt(((s - m * n) ** 2).sum() * G / (G - 1)) / n.sum() if G > 1 else np.nan
        draws = np.empty(B_BOOT)
        for b in range(B_BOOT):
            w = np.bincount(rng.integers(0, G, G), minlength=G)
            draws[b] = (s * w).sum() / max((n * w).sum(), 1)
        out[c] = {"mean": float(m), "se": float(se), "p": float((1 + np.sum(draws <= 0)) / (B_BOOT + 1)),
                  "mde80": float((1.96 + 0.84) * se)}
    return out


def paired_diff(A, B, rng):
    """같은 날 짝 부트스트랩: mean(A.r1n) − mean(B.r1n) ≤ 0 단측 p."""
    days = np.union1d(A.day.unique(), B.day.unique())

    def agg(X):
        p = np.searchsorted(days, X.day.to_numpy())
        return (np.bincount(p, weights=X.r1n.to_numpy(), minlength=len(days)),
                np.bincount(p, minlength=len(days)).astype(float))
    sa, na = agg(A)
    sb, nb = agg(B)
    d0 = sa.sum() / na.sum() - sb.sum() / nb.sum()
    draws = []
    for _ in range(B_BOOT):
        w = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
        draws.append((sa * w).sum() / max((na * w).sum(), 1) - (sb * w).sum() / max((nb * w).sum(), 1))
    return {"diff": float(d0), "p": float((1 + np.sum(np.array(draws) <= 0)) / (B_BOOT + 1))}


def holm(ps):
    order = sorted(ps, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, ps[k] * (len(ps) - i)))
        out[k] = run
    return out


def hyp_rows(R, h, thr=None, extra=None):
    m = (R.pat == h["pat"]) & (R[h["col"]] < (h["thr"] if thr is None else thr))
    if extra is not None:
        m &= extra(R)
    return R[m]


def comp_rows(R, h):
    return R[(R.pat == h["pat"]) & (R[h["col"]] >= h["thr"])]


VARIANTS = {
    "H1+연쇄 두번째": ("H1", lambda R: R.chain >= 1),
    "H1+상대낙폭<-5%": ("H1", lambda R: R.rel < -0.05),
    "H1+120분하락>5%": ("H1", lambda R: R.dd >= 0.05),
    "H2+연쇄 두번째": ("H2", lambda R: R.chain >= 1),
    "H2+시간대거래량 2~5배": ("H2", lambda R: (R.vol_tod >= 2) & (R.vol_tod < 5)),
}
THRESH = {"H1": [-0.005, -0.015, -0.02], "H2": [-0.02, -0.04]}


def judge():
    cost = json.load(open(os.path.join(RES, "cost_model.json"), encoding="utf-8"))
    data = {y: CE.load(y, cost) for y in ("2026h1", "2025h1", "2026", "2025")}
    rng = np.random.default_rng(20261102)
    prim, tabs, sig_out = {}, {}, []
    for hk, h in HYP.items():
        R = data["2026h1"]
        X = dedup(hyp_rows(R, h))
        b = day_boot(X, rng)
        aux = dedup(hyp_rows(data["2025h1"], h))
        prim[hk] = {"label": h["label"], "y2026h1": summ(X), "boot": b, "y2025h1": summ(aux),
                    "kst_awake_2026h1": summ(dedup(hyp_rows(R[kst_awake(R.ts)], h))),
                    "ref_2026_julOct": summ(dedup(hyp_rows(data["2026"], h))),
                    "ref_2025_julOct": summ(dedup(hyp_rows(data["2025"], h)))}
        prim[hk]["p_hyp"] = max(b["r1n"]["p"], b["vsbt"]["p"])
        Xo = X.assign(hyp=hk)
        sig_out.append(Xo[["hyp", "sym", "sid", "ts", "day", "t", "mins", "r1_e2", "r1n", "hit_e2", "c_in", "c_out"]])
        # 보조 보고
        t = {}
        C = dedup(comp_rows(R, h))
        t["반대쪽"] = {"summ": summ(C), "diff": paired_diff(X, C, rng)}
        t["반대쪽_2025h1"] = {"summ": summ(dedup(comp_rows(data["2025h1"], h))),
                           "diff": paired_diff(aux, dedup(comp_rows(data["2025h1"], h)), rng)}
        t["문턱"] = {str(th): {"2026h1": summ(dedup(hyp_rows(R, h, th))), "2025h1": summ(dedup(hyp_rows(data["2025h1"], h, th)))}
                   for th in THRESH[hk]}
        X["month"] = X.day // 100
        t["달별"] = {str(mn): summ(g) for mn, g in X.groupby("month")}
        X["tier"] = X.cap_b.map(lambda c: "<50억$" if c < 5 else ("50억~200억$" if c < 20 else ("200억~1000억$" if c < 100 else "1000억$+")))
        t["시총"] = {k: summ(g) for k, g in X.groupby("tier")}
        big = X.groupby("day").r1n.sum().abs().idxmax() if len(X) else None
        t["큰하루뺌"] = {"day": int(big) if big is not None else None, "summ": summ(X[X.day != big])}
        tabs[hk] = t
    adj = holm({k: v["p_hyp"] for k, v in prim.items()})
    for k in prim:
        prim[k]["p_holm"] = adj[k]
        prim[k]["pass_stat"] = bool(adj[k] < 0.05)
        prim[k]["aux_pos"] = bool(prim[k]["y2025h1"].get("r1n", -1) > 0)
    # 겹친 변형(설명용)
    var = {}
    for name, (hk, f) in VARIANTS.items():
        h = HYP[hk]
        var[name] = {y: summ(dedup(hyp_rows(data[y], h, extra=f))) for y in ("2026h1", "2025h1", "2026", "2025")}
    tabs["변형"] = var
    tabs["기본"] = {y: {"①": summ(dedup(data[y][data[y].pat == "A"])), "②": summ(dedup(data[y][data[y].pat == "B"]))}
                  for y in ("2026h1", "2025h1")}
    json.dump(prim, open(os.path.join(RES, "h3_primary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    json.dump(tabs, open(os.path.join(RES, "h3_tables.json"), "w"), ensure_ascii=False, indent=1, default=float)
    pd.concat(sig_out).to_csv(os.path.join(RES, "h3_signals.csv"), index=False)
    for k, v in prim.items():
        print(k, v["label"], v["y2026h1"], "p_holm", round(v["p_holm"], 3))


def e1():
    from concurrent.futures import ThreadPoolExecutor
    from fetch_alpaca import keys, Pace
    from fetch_quotes import last_quote
    S = pd.read_csv(os.path.join(RES, "h3_signals.csv"))
    U = S.drop_duplicates(["sym", "ts"])[["sym", "ts"]]
    head, pace = keys(), Pace(3.0)

    def one(r):
        t = pd.Timestamp(int(r[1]) * 60, unit="s", tz="UTC")
        ai, bi = last_quote(r[0], t + pd.Timedelta(seconds=70), head, pace)
        ao, bo = last_quote(r[0], t + pd.Timedelta(minutes=31), head, pace)
        return r[0], int(r[1]), ai, bi, ao, bo

    with ThreadPoolExecutor(6) as ex:
        Q = pd.DataFrame(list(ex.map(one, U.itertuples(index=False, name=None))),
                         columns=["sym", "ts", "ask_in", "bid_in", "ask_out", "bid_out"])
    S = S.merge(Q, on=["sym", "ts"], how="left")
    bars, r = {}, []
    for x in S.itertuples():
        if x.sym not in bars:
            bars[x.sym] = E.prep(E.load(x.sym, "alp1m_2026h1"))
        a = bars[x.sym]
        t = int(x.t)
        if not (x.ask_in > 0):
            r.append(np.nan)
            continue
        z = t + 1
        while z + 1 < len(a["ts"]) and a["ts"][z + 1] <= a["ts"][t] + 30 and a["day"][z + 1] == a["day"][t]:
            z += 1
        tg = x.ask_in * 1.01
        hit = a["c"][t + 1] >= tg or (z >= t + 2 and np.max(a["h"][t + 2:z + 1]) >= tg)
        r.append(0.01 if hit else (x.bid_out if x.bid_out > 0 else a["c"][z]) / x.ask_in - 1)
    S["r1_e1"] = r
    S.to_csv(os.path.join(RES, "h3_signals_e1.csv"), index=False)
    P = json.load(open(os.path.join(RES, "h3_primary.json"), encoding="utf-8"))
    for k, g in S.groupby("hyp"):
        e = {"n": int(g.r1_e1.notna().sum()), "r1_e1": float(g.r1_e1.mean()), "hit_e1": float((g.r1_e1 == 0.01).mean()),
             "half_spread_med": float(((g.ask_in - g.bid_in) / (g.ask_in + g.bid_in)).median())}
        P[k]["e1"] = e
        P[k]["accept"] = bool(P[k]["pass_stat"] and e["r1_e1"] > 0 and P[k]["aux_pos"])
        print(k, e, "채택", P[k]["accept"])
    json.dump(P, open(os.path.join(RES, "h3_primary.json"), "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    {"judge": judge, "e1": e1}[sys.argv[1]]()
