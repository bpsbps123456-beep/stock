"""profile·contrast 갈래 적대 검증(독립 재계산). derived/만 읽는다. 출력은 화면(JSON 한 줄)뿐이다.

py -B code/verify_profile_contrast.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
WINS = {"W1": (-20, -6), "W2": (-5, -1), "W3": (0, 0), "W4": (1, 5), "W5": (6, 20)}
MEAS = ["A1", "A2", "A3", "A4", "X1", "X2", "X3", "X4", "X5", "X6", "H1", "H2", "H3", "H4",
        "L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4", "R5", "C1", "C2", "C3"]
rng = np.random.default_rng(7)

mw = pd.read_parquet(DER / "measures_win.parquet")
pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date").sort_index()
ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
S = pf.index
POS = pd.Series(np.arange(len(S)), index=S)
FEND = mw["date"].max()
B15 = [pd.Timestamp(x) for x in ev["B15"]]
B10 = [pd.Timestamp(x) for x in ev["B10"]]
F = pd.DataFrame(ev["F"])
F["date"] = pd.to_datetime(F["date"])
F = F[F["has_flows"]]
# 넓은 표: (col) → date × (L, measure)
WIDE = {c: mw.pivot_table(index="date", columns=["L", "measure"], values=c, dropna=False) for c in ("P", "E_full", "E_exp", "y", "n_cens", "n_merged")}


def end_dates(days, w):
    z = WINS[w][1]
    out = []
    for d in days:
        i = POS[pd.Timestamp(d)] + z
        out.append(S[i] if i < len(S) and S[i] <= FEND else pd.NaT)
    return out


def vals(days, w, m, col):
    L = WINS[w][1] - WINS[w][0] + 1
    s = WIDE[col][(L, m)]
    return np.array([s.get(e, np.nan) if pd.notna(e) else np.nan for e in end_dates(days, w)], float)


# ---------- 1. profile: 공통 칸 다시 세기 ----------
def p_flag(v, thr):
    v = v[np.isfinite(v)]
    return ("hi" if (v >= 80).sum() >= thr else ("lo" if (v <= 20).sum() >= thr else "")), int((v >= 80).sum()), int((v <= 20).sum())


def e_flag(v, thr):
    v = v[np.isfinite(v)]
    npos, nneg = int((v > 0).sum()), int((v < 0).sum())
    ok = np.median(np.abs(v)) >= 0.5 if len(v) else False
    return ("+" if npos >= thr and ok else ("-" if nneg >= thr and ok else "")), npos, nneg


def count_flags(days, thr):
    cp = ce = 0
    for w in WINS:
        for m in MEAS:
            cp += p_flag(vals(days, w, m, "P"), thr)[0] != ""
            ce += e_flag(vals(days, w, m, "E_full"), thr)[0] != ""
    return cp, ce


R = {}
R["flag_counts"] = {"B15": count_flags(B15, 5), "B10": count_flags(B10, 9)}


def share(days, w, m, col, side):
    v = vals(days, w, m, col)
    v = v[np.isfinite(v)]
    h = {"hi": v >= 80, "lo": v <= 20, "+": v > 0, "-": v < 0}[side]
    return int(h.sum()), len(v), round(float(np.median(v)), 3) if len(v) else None


def share_ep(w, m, col, side):
    v = vals(F["date"], w, m, col)
    ok = np.isfinite(v)
    h = {"hi": v >= 80, "lo": v <= 20, "+": v > 0, "-": v < 0}[side]
    return round(float(pd.Series(h[ok]).groupby(F["episode"].values[ok]).mean().mean()), 3)


# 바닥 닮은 날 모음: 10세션 최저 & dd60 ≤ −10%(F ①②), 흐름 끝 안, B10 바닥 자신은 뺌
pool_all = pf[(pf["is_10d_low"]) & (pf["dd60"] <= -0.10) & (pf.index <= FEND) & (pf.index >= pd.Timestamp("2023-02-01"))].index
pool = pool_all.difference(B10)
# 큰 하락일: 하루 수익 하위 10%(흐름 기간)
r1 = pf.loc[(pf.index <= FEND) & (pf.index >= pd.Timestamp("2023-02-01")), "ret_1"]
bigdown = r1[r1 <= r1.quantile(0.10)].index.difference(B10)

HEAD = [("X1", "W3", "P", "hi"), ("X2", "W3", "P", "hi"), ("L1", "W3", "P", "hi"), ("H2", "W3", "P", "hi"),
        ("C1", "W3", "P", "hi"), ("X6", "W3", "P", "lo"), ("X1", "W3", "E_full", "-"), ("X3", "W3", "E_full", "+"),
        ("X3", "W2", "E_full", "+"), ("R5", "W3", "E_full", "-"), ("X5", "W3", "P", "hi"), ("X5", "W3", "E_full", "+"),
        ("X1", "W4", "E_full", "+"), ("X6", "W4", "E_full", "-"), ("X3", "W4", "P", "hi"), ("L1", "W2", "P", "hi"),
        ("X1", "W3", "P", "lo"), ("X3", "W3", "P", "hi"), ("H1", "W3", "P", "hi"), ("H1", "W3", "P", "lo"),
        ("A1", "W3", "P", "hi"), ("A1", "W3", "E_full", "+"), ("X6", "W3", "P", "lo")]
rows = []
for m, w, col, side in HEAD:
    b10 = share(B10, w, m, col, side)
    b15 = share(B15, w, m, col, side)
    f = share(F["date"], w, m, col, side)
    pl = share(pool, w, m, col, side)
    bd = share(bigdown, w, m, col, side) if w == "W3" else (None, None, None)
    rows.append({"cell": f"{m}·{w}·{col[0]}{side}", "B15": f"{b15[0]}/{b15[1]}", "B10": f"{b10[0]}/{b10[1]}", "B10_med": b10[2],
                 "F": f"{f[0]}/{f[1]}", "F_ep": share_ep(w, m, col, side), "pool": f"{pl[0]}/{pl[1]}",
                 "big_down": f"{bd[0]}/{bd[1]}" if bd[1] else ""})
R["head"] = rows
R["pool_n"], R["bigdown_n"] = len(pool), len(bigdown)

# 바닥 닮은 날에서 11개(서로 10세션 이상 떨어짐)를 뽑았을 때 공통 칸 수(우연이 아니라 '급락 바닥 닮음' 기준)


def spaced_draw(cands, n, gap=10, tries=2000):
    p = POS[cands].values
    for _ in range(tries):
        idx = rng.permutation(len(p))
        pick = []
        for j in idx:
            if all(abs(p[j] - p[k]) >= gap for k in pick):
                pick.append(j)
            if len(pick) == n:
                return [cands[k] for k in pick]
    return None


def flag_count_E_P(days, thr):
    return count_flags(days, thr)


NDRAW = 200
null_pool = []
for _ in range(NDRAW):
    d = spaced_draw(pool, 11)
    if d is None:
        continue
    null_pool.append(flag_count_E_P(d, 9))
null_pool = np.array(null_pool)
all_days = mw.loc[(mw["L"] == 15) & mw["E_full"].notna(), "date"].unique()
all_days = pd.DatetimeIndex(all_days)[30:-25]
null_rand = np.array([flag_count_E_P(spaced_draw(all_days, 11), 9) for _ in range(NDRAW)])
R["null_B10_flags"] = {"obs": R["flag_counts"]["B10"], "pool_mean": null_pool.mean(0).round(2).tolist(),
                       "pool_p_ge_obs": [float((null_pool[:, i] >= R["flag_counts"]["B10"][i]).mean()) for i in range(2)],
                       "rand_mean": null_rand.mean(0).round(2).tolist(), "n_pool_draw": len(null_pool)}

# 2026 확인: 2023~25 B10 7개로 W1~W3 고르기, 2026 같은 쪽(E_full / E_exp)
old = [b for b in B10 if b.year <= 2025]
new = [b for b in B10 if b.year == 2026]
F26 = F[F["date"].dt.year == 2026]["date"]
ho = []
for w in ("W1", "W2", "W3"):
    for m in MEAS:
        pv = vals(old, w, m, "P")
        if (pv >= 80).sum() >= 6 or (pv <= 20).sum() >= 6:
            side = "hi" if (pv >= 80).sum() >= 6 else "lo"
            n26 = share(new, w, m, "P", side)[0]
            ho.append((f"{m}·{w}·P{side}", n26, share(F26, w, m, "P", side)[0], None))
        ev_ = vals(old, w, m, "E_full")
        npos, nneg = (ev_ > 0).sum(), (ev_ < 0).sum()
        if (npos >= 6 or nneg >= 6) and np.median(np.abs(ev_)) >= 0.5:
            side = "+" if npos >= 6 else "-"
            ho.append((f"{m}·{w}·E{side}", share(new, w, m, "E_full", side)[0], share(F26, w, m, "E_full", side)[0],
                       share(new, w, m, "E_exp", side)[0]))
R["holdout"] = {"n_sel_W1W3": len(ho), "rows": ho, "nF26": len(F26)}

# 하락 누적(10% 고점 다음 날 ~ 바닥, 일별 L=1 합)
md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date")
dec = []
for b in ev["B10_events"]:
    seg = md.loc[(md.index > pd.Timestamp(b["top10"])) & (md.index <= pd.Timestamp(b["bottom"]))]
    seg2 = md.loc[(md.index >= pd.Timestamp(b["top10"])) & (md.index <= pd.Timestamp(b["bottom"]))]
    dec.append((b["bottom"], round(seg["X1"].sum() / 1e6), round(seg["H1"].sum() / 1e6), round(seg["L1"].sum() / 1e6),
                round(100 * seg2["X1"].sum() / seg2["A1"].sum(), 1), len(seg2)))
R["decline"] = dec

# ---------- 2. contrast: 짝 비교 ----------
eps = [e["episode"] for e in ev["episodes"] if e["episode"] != "current" and len(F[F["episode"] == e["episode"]])]
fmap = {e: list(F.loc[F["episode"] == e, "date"]) for e in eps}
CELLS = [(m, w, n) for n in ("P", "E_full") for w in ("W1", "W2", "W3") for m in MEAS] + [(m, "W4", "E_full") for m in MEAS]


def ep_d(cols_by_day, ep, m, w, col):
    b = vals([pd.Timestamp(ep)], w, m, col)[0]
    f = vals(fmap[ep], w, m, col)
    f = f[np.isfinite(f)]
    return b - f.mean() if np.isfinite(b) and len(f) else np.nan


SIGNS = {n: np.array(list(itertools.product([1, -1], repeat=n))) for n in range(1, 9)}


def wil_p(d):
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return 1.0
    r = pd.Series(np.abs(d)).rank().values
    tp = r[d > 0].sum()
    mu = r.sum() / 2
    T = (SIGNS[n] == 1) @ r
    return float((np.abs(T - mu) >= abs(tp - mu) - 1e-9).mean())


def sign_p(d):
    d = d[np.isfinite(d) & (d != 0)]
    n, k = len(d), int((d > 0).sum())
    from scipy.stats import binomtest
    return binomtest(k, n, 0.5).pvalue if n else 1.0


def build_D(col_map):
    D = np.full((len(CELLS), len(eps)), np.nan)
    for i, (m, w, n) in enumerate(CELLS):
        for j, e in enumerate(eps):
            D[i, j] = ep_d(None, e, m, w, col_map.get(n, n))
    return D


Dlin = build_D({})
wp = np.array([wil_p(r) for r in Dlin])
sp = np.array([sign_p(r) for r in Dlin])
isE = np.array([c[2] == "E_full" for c in CELLS])
flipnull = []
for sg in SIGNS[len(eps)]:
    flipnull.append([(np.array([wil_p(r * sg) for r in Dlin]) < 0.05).sum(), (np.array([wil_p(r * sg) for r in Dlin[isE]]) < 0.05).sum()])
flipnull = np.array(flipnull)


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


C = {"eps": eps, "n_wil05": int((wp < 0.05).sum()), "n_sign05": int((sp < 0.05).sum()), "n_wil05_E": int((wp[isE] < 0.05).sum()),
     "n_wil05_P": int((wp[~isE] < 0.05).sum()), "min_q_wil": round(float(bh(wp).min()), 3),
     "flip_P_ge_all": float((flipnull[:, 0] >= (wp < 0.05).sum()).mean()), "flip_mean_all": float(flipnull[:, 0].mean()),
     "flip_P_ge_E": float((flipnull[:, 1] >= (wp[isE] < 0.05).sum()).mean())}
top = np.argsort(wp, kind="stable")[:12]
C["top"] = [(f"{CELLS[i][0]}·{CELLS[i][1]}·{CELLS[i][2][0]}", round(wp[i], 4), int((Dlin[i] > 0).sum()), round(float(np.nanmedian(Dlin[i])), 2))
            for i in top]
iX1 = CELLS.index(("X1", "W3", "P"))
C["X1W3P"] = (round(float(np.nanmedian(Dlin[iX1])), 2), int((Dlin[iX1] > 0).sum()), round(wil_p(Dlin[iX1]), 3))

# 제곱항 메아리: (a) 전체 적합(갈래와 같은 식) (b) 바닥·F ±20세션을 적합에서 빼고 그 날들을 예측(밖 적합)


def fit_resid(L, m, quad, excl_days=None):
    idx = WIDE["y"].index
    y = WIDE["y"][(L, m)].values
    X = pd.DataFrame(index=idx)
    X["c"] = 1.0
    X["ret"] = pf[f"ret_{L}"].reindex(idx).values
    X["pre20"] = pf[f"pre20_{L}"].reindex(idx).values
    X["dd60"] = pf["dd60"].reindex(idx).values
    X["rv20"] = pf["rv20"].reindex(idx).values
    reg = pf["regime"].reindex(idx).values
    for k in range(1, 5):
        X[f"g{k}"] = (reg == k).astype(float)
    if quad:
        for c in ("ret", "pre20", "dd60", "rv20"):
            X[c + "2"] = X[c] ** 2
        X["rxd"] = X["ret"] * X["dd60"]
    XX = X.values
    ok = np.isfinite(y) & np.isfinite(XX).all(1)
    fit_ok = ok.copy()
    if excl_days is not None:
        fit_ok &= ~excl_days
    b, *_ = np.linalg.lstsq(XX[fit_ok], y[fit_ok], rcond=None)
    r = y - XX @ b
    sd = np.sqrt(np.nansum(r[fit_ok] ** 2) / (fit_ok.sum() - XX.shape[1]))
    out = np.where(ok, r / sd, np.nan)
    return pd.Series(out, index=idx)


ix = WIDE["y"].index
ipos = POS.reindex(ix).values
ev_pos = np.array([POS[d] for d in list(B10) + list(F["date"])])
excl = (np.abs(ipos[:, None] - ev_pos[None, :]) <= 25).any(1)  # 창 끝이 사건 ±25 안이면 적합에서 뺌
for tag, quad, ex in (("Eq_in", True, None), ("Eq_out", True, excl), ("El_out", False, excl)):
    cols = {}
    for L in (1, 5, 15):
        for m in MEAS:
            cols[(L, m)] = fit_resid(L, m, quad, ex)
    WIDE[tag] = pd.DataFrame(cols)
for tag in ("Eq_in", "Eq_out", "El_out"):
    Dq = build_D({"E_full": tag})
    wq = np.array([wil_p(r) for r in Dq[isE]])
    C[f"n_wil05_{tag}"] = int((wq < 0.05).sum())
    C[f"top3_{tag}"] = [round(wil_p(Dq[CELLS.index(c)]), 3) for c in (("X6", "W4", "E_full"), ("X1", "W4", "E_full"), ("H4", "W3", "E_full"))]
    if tag == "Eq_in":
        wqall = np.array([wil_p(r) for r in Dq])
        C["quad_in_survivors"] = [f"{CELLS[i][0]}·{CELLS[i][1]}" for i in range(len(CELLS)) if isE[i] and wqall[i] < 0.05]

# 날짜 섞기(B10 11 대 F 28)
pool_days = list(B10) + list(F["date"])
isB = np.array([1] * len(B10) + [0] * len(F))
LAB = np.array([rng.permutation(isB) for _ in range(10000)])
pperm = []
for (m, w, n) in CELLS:
    x = vals(pool_days, w, m, n)
    ok = np.isfinite(x)
    xb, lb, ll = x[ok], isB[ok], LAB[:, ok]
    obs = xb[lb == 1].mean() - xb[lb == 0].mean()
    nb = ll.sum(1)
    sim = (ll @ xb) / nb - ((1 - ll) @ xb) / (len(xb) - nb)
    pperm.append((np.sum(np.abs(sim) >= abs(obs) - 1e-12) + 1) / (len(sim) + 1))
pperm = np.array(pperm)
C["n_perm05"] = int((pperm < 0.05).sum())
C["min_q_perm"] = round(float(bh(pperm).min()), 3)
iX3W2 = CELLS.index(("X3", "W2", "E_full"))
C["X3W2E"] = {"perm_p": round(float(pperm[iX3W2]), 4), "wil_p": round(wil_p(Dlin[iX3W2]), 3), "n_pos": int((Dlin[iX3W2] > 0).sum()),
              "d": np.round(Dlin[iX3W2], 2).tolist()}
iR5 = CELLS.index(("R5", "W3", "E_full"))
C["R5W3E"] = {"perm_p": round(float(pperm[iR5]), 4), "wil_p": round(wil_p(Dlin[iR5]), 3), "d": np.round(Dlin[iR5], 2).tolist()}
iX3 = CELLS.index(("X3", "W3", "E_full"))
C["X3W3E"] = {"perm_p": round(float(pperm[iX3]), 4), "wil_p": round(wil_p(Dlin[iX3]), 3), "d": np.round(Dlin[iX3], 2).tolist()}

# 2026 확인: 2023~25 짝 4국면으로 W1~W3 156칸 → 5칸
oe = [i for i, e in enumerate(eps) if e[:4] <= "2025"]
ne = [i for i, e in enumerate(eps) if e[:4] == "2026"]
cand = []
for i, (m, w, n) in enumerate(CELLS):
    if w == "W4":
        continue
    d = Dlin[i, oe]
    t = np.nanmean(d) / (np.nanstd(d, ddof=1) / np.sqrt(np.isfinite(d).sum()))
    cand.append((wil_p(d), sign_p(d), -abs(t), i))
cand.sort()
sel = [c[3] for c in cand[:5]]
C["n_tied_min"] = sum(1 for c in cand if c[0] <= cand[0][0] + 1e-12)
C["holdout"] = [(f"{CELLS[i][0]}·{CELLS[i][1]}·{CELLS[i][2][0]}", int(np.sign(np.nanmedian(Dlin[i, oe]))),
                 [int(np.sign(Dlin[i, j]) == np.sign(np.nanmedian(Dlin[i, oe]))) for j in ne]) for i in sel]

# 겹침: F 날의 바닥 기준 k(같은 국면)
ov = []
for e in eps:
    for f in fmap[e]:
        ov.append((e, str(f.date()), int(POS[f] - POS[pd.Timestamp(e)])))
C["F_k"] = ov
# 깊이: 바닥 대 F dd60
C["dd60"] = [(e, round(float(pf.loc[pd.Timestamp(e), "dd60"]), 3), round(float(np.mean([pf.loc[f, "dd60"] for f in fmap[e]])), 3)) for e in eps]

R["contrast"] = C
for k, v in R.items():
    print(k, json.dumps(v, ensure_ascii=False, default=str))

# ---------- 3. 추가 점검 ----------
X = {}
# (a) 창별 B10 E 공통 칸 수: 바닥 대 바닥 닮은 날 뽑기(평균)


def flags_by_window(days, thr):
    return {w: sum(e_flag(vals(days, w, m, "E_full"), thr)[0] != "" for m in MEAS) for w in WINS}


X["E_flags_by_W_B10"] = flags_by_window(B10, 9)
acc = {w: [] for w in WINS}
for _ in range(100):
    d = spaced_draw(pool, 11)
    for w, v in flags_by_window(d, 9).items():
        acc[w].append(v)
X["E_flags_by_W_pool_mean"] = {w: round(float(np.mean(v)), 2) for w, v in acc.items()}
X["E_flags_by_W_pool_p"] = {w: float(np.mean(np.array(v) >= X["E_flags_by_W_B10"][w])) for w, v in acc.items()}

# (b) 국면 하나 빼고 적합한 메아리(밖 예측): 국면(10% 고점 −25 ~ 바닥 +25)을 적합에서 빼고 그 국면 바닥·F를 예측
epinfo = {e["episode"]: e for e in ev["episodes"]}
LOO = {"lin": {}, "quad": {}}
for e in eps:
    top = pd.Timestamp(epinfo[e]["start"])
    a = POS[top] - 25 if top is not None else POS[pd.Timestamp(e)] - 80
    span = (ipos >= a) & (ipos <= POS[pd.Timestamp(e)] + 25)
    for tag, q in (("lin", False), ("quad", True)):
        LOO[tag][e] = {(L, m): fit_resid(L, m, q, span) for L in (1, 5, 15) for m in MEAS}
for tag in ("lin", "quad"):
    D = np.full((len(CELLS), len(eps)), np.nan)
    for i, (m, w, n) in enumerate(CELLS):
        if n != "E_full":
            continue
        L = WINS[w][1] - WINS[w][0] + 1
        for j, e in enumerate(eps):
            s = LOO[tag][e][(L, m)]
            b = s.get(end_dates([pd.Timestamp(e)], w)[0], np.nan)
            f = np.array([s.get(x, np.nan) if pd.notna(x) else np.nan for x in end_dates(fmap[e], w)])
            f = f[np.isfinite(f)]
            D[i, j] = b - f.mean() if np.isfinite(b) and len(f) else np.nan
    wl = np.array([wil_p(r) for r in D[isE]])
    X[f"loo_{tag}_n05"] = int((wl < 0.05).sum())
    X[f"loo_{tag}_top3"] = [round(wil_p(D[CELLS.index(c)]), 3) for c in (("X6", "W4", "E_full"), ("X1", "W4", "E_full"), ("H4", "W3", "E_full"))]
    X[f"loo_{tag}_list"] = [f"{CELLS[i][0]}·{CELLS[i][1]}" for i in np.flatnonzero(isE) if wil_p(D[i]) < 0.05]
X["excl_share_pm25"] = round(float(excl.mean()), 3)

# (c) 2026-09-14 하락 누적의 고점 날 X1
t9 = pd.Timestamp(epinfo["2026-09-14"]["start"])
X["top_day_2026_09"] = (str(t9), round(md.loc[t9, "X1"] / 1e6) if t9 is not None else None)
R["extra"] = X
for k, v in X.items():
    print(k, json.dumps(v, ensure_ascii=False, default=str))

# (d) profile E 공통 칸이 제곱항 메아리(전체 적합)에서도 남는가: B10 같은 쪽 수·중앙 |z|, F·큰 하락일 비율
Q = []
for m, w, side in (("X3", "W3", "+"), ("X3", "W2", "+"), ("R5", "W3", "-"), ("X5", "W3", "+"), ("X1", "W4", "+"),
                   ("X6", "W4", "-"), ("X1", "W3", "-"), ("A3", "W3", "+"), ("C3", "W3", "+")):
    for col in ("E_full", "Eq_in"):
        v = vals(B10, w, m, col)
        Q.append((f"{m}·{w}·{side}", col, share(B10, w, m, col, side)[0], round(float(np.median(np.abs(v))), 2),
                  "%d/%d" % share(F["date"], w, m, col, side)[:2],
                  "%d/%d" % share(bigdown, w, m, col, side)[:2] if w == "W3" else ""))
print("quad_profile", json.dumps(Q, ensure_ascii=False))
print("B10_E_flags_quad", sum(e_flag(vals(B10, w, m, "Eq_in"), 9)[0] != "" for w in WINS for m in MEAS),
      "B15", sum(e_flag(vals(B15, w, m, "Eq_in"), 5)[0] != "" for w in WINS for m in MEAS))

# (e) 강건성 상위 10칸: (a) 07-29 뺌 방향 (b) B15 국면만 방향 (c) 하나 빼기 부호 뒤집힘
b15e = [j for j, e in enumerate(eps) if pd.Timestamp(e) in B15]
j729 = eps.index("2026-07-29")
rb = []
for i in np.argsort(wp, kind="stable")[:10]:
    d = Dlin[i]
    dr = np.sign(np.nanmedian(d))
    a = np.sign(np.nanmedian(np.delete(d, j729))) == dr
    b = np.sign(np.nanmedian(d[b15e])) == dr
    c = sum(np.sign(np.nanmedian(np.delete(d, j))) != dr for j in range(len(d)))
    rb.append((f"{CELLS[i][0]}·{CELLS[i][1]}·{CELLS[i][2][0]}", bool(a), bool(b), int(c)))
print("robust", json.dumps(rb, ensure_ascii=False))
