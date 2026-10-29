"""(b) 사전 등록 보조 보고 중 빠진 것(야후 같은 분 저가 대조) + 목표 체결 과대·과소(넘어 시작한 봉) + 격자 순위 지속 진단."""
from common import *
from build import SYMS

R = pd.read_csv(os.path.join(OUT, "recompute_signals.csv"))
out = {}
# 1) 야후 같은 분 저가 대조: 9/30~10/28 뒤 구간 신호. ①은 신호 봉, ②는 이탈 봉(t−1)의 저가
yf = {}
rows = []
for nm in CELLS:
    pat = nm[-1]
    S = pd.read_csv(os.path.join(RES, f"signals_{nm}.csv"))
    S = S[(S.day >= 20260930) & (S.day <= 20261028)]
    for sid, grp in S.groupby("sid"):
        sym = SYMS[int(sid)]
        if sym not in yf:
            d = E.load(sym, "yf1m")
            a = E.prep(E.load(sym, "alp1m"))
            yf[sym] = (d, a)
        d, a = yf[sym]
        if d is None:
            continue
        ylow = d.Low
        for _, r in grp.iterrows():
            k = int(r.t) if pat == "A" else int(r.t) - 1
            ts = pd.Timestamp(int(a["ts"][k]) * 60, unit="s", tz="UTC").tz_convert(E.ET)
            yl = ylow.get(ts, np.nan)
            al = a["l"][k]
            tolr = max(0.0015, 0.1 * a["atr"][k] / a["c"][k]) if np.isfinite(a["atr"][k]) else 0.0015
            rows.append(dict(cell=nm, sid=sid, ts=r.ts, day=r.day, r1_e2=r.r1_e2, bt=r.bt_r1_e2, r1net=r.r1net_e2,
                             has_yf=np.isfinite(yl), agree=bool(np.isfinite(yl) and abs(yl - al) / al <= tolr),
                             rel=(yl - al) / al if np.isfinite(yl) else np.nan))
Y = pd.DataFrame(rows)
yres = {}
for nm in CELLS:
    X = Y[Y.cell == nm]
    A_ = X[X.agree]
    D_ = X[~X.agree]
    yres[nm] = dict(n=len(X), has_yf=float(X.has_yf.mean()), agree=float(X.agree.mean()),
                    r1_agree=float(A_.r1_e2.mean()), r1net_agree=float(A_.r1net.mean()), dbt_agree=float((A_.r1_e2 - A_.bt).mean()),
                    r1_disagree=float(D_.r1_e2.mean()) if len(D_) else None, n_disagree=len(D_),
                    yf_low_above_alp_share=float((X.rel > 0.0015).mean()))
    print("yahoo", nm, {k: round(v, 5) if isinstance(v, float) else v for k, v in yres[nm].items()})
out["yahoo_lowcheck_0930_1028"] = yres

# 2) 목표 도달 봉이 목표 위에서 시작(갭)한 비율·평균 초과분 — 엔진은 +1%만 적는다(실제 지정가는 더 받음)
over = {}
cache = {}
for nm in CELLS:
    S = pd.read_csv(os.path.join(RES, f"signals_{nm}.csv"))
    S = S[(S.day > SPLIT) & (S.hit_e2 > 0.5)]
    gaps, extra = 0, []
    for sid, grp in S.groupby("sid"):
        sym = SYMS[int(sid)]
        if sym not in cache:
            cache[sym] = E.prep(E.load(sym, "alp1m"))
        a = cache[sym]
        for _, r in grp.iterrows():
            t = int(r.t)
            e = t + 1
            en = a["c"][e]
            up = en * 1.01
            for j in range(e + 1, min(e + 40, len(a["h"]))):
                if a["ts"][j] > a["ts"][t] + 30:
                    break
                if a["h"][j] >= up:
                    if a["o"][j] > up:
                        gaps += 1
                        extra.append(a["o"][j] / en - 1.01)
                    break
    over[nm] = dict(hits=len(S), gap_open_above=gaps, mean_extra_if_gap=float(np.mean(extra)) if extra else 0.0,
                    r1_add_per_signal=float(np.sum(extra) / max(1, (pd.read_csv(os.path.join(RES, f"signals_{nm}.csv")).day > SPLIT).sum())))
    print("overshoot", nm, over[nm])
out["overshoot_back"] = over

# 3) 격자 순위 지속(전 칸 상관 — 칸을 다시 고르지 않음)
pers = {}
for pat in ("A", "B"):
    g = pd.read_parquet(os.path.join(RES, f"grid_{pat}.parquet"),
                        columns=["f_r1", "b_r1", "f_r1net", "b_r1net", "f_race", "b_race", "f_hit", "f_bvhit", "b_hit", "b_bvhit",
                                 "f_r1e0", "b_r1e0", "f_t", "b_t", "f_n", "b_n", "elig"])
    g = g[g.elig]
    sp = lambda x, y: float(x.corr(y, method="spearman"))
    pers[pat] = dict(r1=sp(g.f_r1, g.b_r1), r1net=sp(g.f_r1net, g.b_r1net), race=sp(g.f_race, g.b_race),
                     hit_lift=sp(g.f_hit - g.f_bvhit, g.b_hit - g.b_bvhit), r1e0=sp(g.f_r1e0, g.b_r1e0),
                     t_vs_bv=sp(g.f_t, g.b_t), n=sp(g.f_n, g.b_n),
                     share_b_r1net_gt0=float((g.b_r1net > 0).mean()), share_b_r1_gt0=float((g.b_r1 > 0).mean()))
    print("persist", pat, {k: round(v, 3) for k, v in pers[pat].items()})
out["grid_rank_persistence"] = pers
json.dump(out, open(os.path.join(OUT, "b_checks.json"), "w"), ensure_ascii=False, indent=1, default=float)
