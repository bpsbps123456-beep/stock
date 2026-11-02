"""BRIEF5 감사 — 구현 A(stage5.py) 결과를 깎을 결함 찾기 (2026-11-03).

입력: ../results/stage5_*.csv, stage5.json, ev4_events.parquet, ../cache/soxx_1d_yf_20210101_20261028.csv.
엔진을 다시 돌리지 않는다(표 A 하루 수익은 stage5_daily_A.csv, 사건 하루 손익은 stage5_daily_B.csv).
단계는 이 파일에서 따로 다시 계산해 대조한다(Wilder RSI 루프, 창 루프 z).
실행: cd code && py audit5.py → ../results/audit5.json
      4년 합친 원형 이동(10절)까지: AUDIT5_POOLED=1 py audit5.py
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
CACHE = os.path.normpath(os.path.join(HERE, "..", "cache"))
SEED, NB, NSH, MINSH = 20261103, 10000, 5000, 20
STAGES = (-2, -1, 0, 1, 2)
NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}
COLS = ["SOXL_w90", "SOXS_w90", "SOXL_std", "SOXS_std"]
YEARS = (2023, 2024, 2025, 2026)


def f(v, n=3):
    if v is None:
        return None
    v = float(v)
    return round(v, n) if np.isfinite(v) else None


# ---------------------------------------------------------------- 1. 단계 독립 재계산·미래 참조
def stages_indep(c):
    lc = np.log(c.values)
    lr = np.r_[np.nan, np.diff(lc)]
    N = 20
    k = np.sqrt((N - 1) * (2 * N - 1) / (6 * N))
    z = np.full(len(c), np.nan)
    for i in range(N, len(c)):
        w = lr[i - N + 1:i + 1]
        z[i] = (lc[i] - lc[i - N + 1:i + 1].mean()) / (w.std(ddof=1) * k)
    d = pd.Series(c.values).diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    # Wilder RMA, 첫 값 = 첫 14개 단순 평균(Pine ta.rsi)
    au = np.full(len(c), np.nan); ad = np.full(len(c), np.nan)
    au[14], ad[14] = up[1:15].mean(), dn[1:15].mean()
    for i in range(15, len(c)):
        au[i] = (au[i - 1] * 13 + up[i]) / 14
        ad[i] = (ad[i - 1] * 13 + dn[i]) / 14
    rsi = np.where(ad == 0, 100, np.where(au == 0, 0, 100 - 100 / (1 + au / np.where(ad == 0, 1, ad))))
    st = np.select([rsi >= 70, rsi <= 30, z >= 0.5, z <= -0.5], [2, -2, 1, -1], 0).astype(float)
    st[(np.arange(len(c)) < 129) | ~np.isfinite(z) | ~np.isfinite(rsi)] = np.nan
    return pd.Series(st, index=c.index), pd.Series(rsi, index=c.index), pd.Series(z, index=c.index)


def runs_of(st):
    """연속 같은 단계 묶음 번호."""
    st = np.asarray(st)
    return np.r_[0, np.cumsum(st[1:] != st[:-1])]


def boot_ci(v, seed=SEED, nb=NB):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (nb, len(v)))].mean(1)
    return [f(np.percentile(b, 2.5)), f(np.percentile(b, 97.5))]


def ep_boot_ci(vals, eps, seed=SEED, nb=NB):
    """구간 묶음 부트스트랩: 구간을 복원 추출, 날 가중 평균."""
    vals, eps = np.asarray(vals, float), np.asarray(eps)
    u = np.unique(eps)
    if len(u) < 2:
        return [None, None]
    s = np.array([vals[eps == e].sum() for e in u]); n = np.array([(eps == e).sum() for e in u])
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(u), (nb, len(u)))
    b = s[ix].sum(1) / n[ix].sum(1)
    return [f(np.percentile(b, 2.5)), f(np.percentile(b, 97.5))]


def episode_stats(vals, eps):
    vals, eps = np.asarray(vals, float), np.asarray(eps)
    u = np.unique(eps)
    em = np.array([vals[eps == e].mean() for e in u])
    en = np.array([(eps == e).sum() for e in u])
    s = np.array([vals[eps == e].sum() for e in u])
    tot = vals.sum()
    loo = [(tot - s[j]) / (len(vals) - en[j]) for j in range(len(u))] if len(u) > 1 else []
    out = {"episodes": int(len(u)), "days": int(len(vals)),
           "day_mean": f(vals.mean()),
           "episode_mean_of_means": f(em.mean()),
           "episode_t": f(em.mean() / (em.std(ddof=1) / np.sqrt(len(em)))) if len(em) > 2 and em.std(ddof=1) > 0 else None,
           "episodes_positive": int((em > 0).sum()),
           "ep_block_ci95": ep_boot_ci(vals, eps),
           "day_iid_ci95": boot_ci(vals),
           "loo_episode_min": f(min(loo)) if loo else None, "loo_episode_max": f(max(loo)) if loo else None,
           "largest_episode_share_of_sum": f(s[np.argmax(np.abs(s))] / tot) if tot != 0 else None,
           "largest_episode_days": int(en[np.argmax(np.abs(s))])}
    return out


def shift_null(st, v, s, n_all=True):
    """원형 이동(모든 k ∈ [20, n−20])으로 단계 s 날 평균의 귀무 분포."""
    n = len(st)
    ks = np.arange(MINSH, n - MINSH + 1)
    R = (np.arange(n)[None, :] - ks[:, None]) % n
    m = (st[R] == s)
    return (m * v[None, :]).sum(1) / m.sum(1)


def main():
    out = {}
    A = pd.read_csv(os.path.join(RES, "stage5_daily_A.csv"), parse_dates=["day", "prev_date"])
    B = pd.read_csv(os.path.join(RES, "stage5_daily_B.csv"), parse_dates=["day"])
    C = pd.read_csv(os.path.join(RES, "stage5_daily_C.csv"), parse_dates=["day"])
    S = pd.read_csv(os.path.join(RES, "stage5_stages.csv"), parse_dates=["date"]).set_index("date")
    J = json.load(open(os.path.join(RES, "stage5.json"), encoding="utf-8"))
    c = pd.read_csv(os.path.join(CACHE, "soxx_1d_yf_20210101_20261028.csv"), index_col=0, parse_dates=True)["Close"]

    # ---- 1. 단계·미래 참조
    st_i, rsi_i, z_i = stages_indep(c)
    both = st_i.notna() & S["stage"].notna()
    chk = {"indep_stage_agree": f((st_i[both] == S["stage"][both]).mean(), 6),
           "indep_rsi_max_diff": float(np.nanmax(np.abs(rsi_i.values - S["rsi14"].values))),
           "indep_z_max_diff": float(np.nanmax(np.abs(z_i.values - S["z20"].values)))}
    # D−1: prev_date가 D보다 앞이고, yfinance 날짜열에서 D 바로 앞 날이며, 그 날 단계와 같은가
    yd = list(c.index)
    pos = {d: k for k, d in enumerate(yd)}
    ok_prev = [(r.prev_date < r.day) and pos.get(r.day) is not None and yd[pos[r.day] - 1] == r.prev_date for r in A.itertuples()]
    chk["prev_date_is_yf_previous_session"] = f(np.mean(ok_prev), 6)
    chk["stage_prev_equals_stage_at_prev_date"] = f(np.mean(st_i.reindex(A["prev_date"]).values == A["stage_prev"].values), 6)
    # 미래 참조였다면(D 종가 단계) 얼마나 달라지나: 같은 날 단계 쓰면 표 A가 크게 갈려야 정상
    A["stage_same_day"] = st_i.reindex(A["day"]).values
    chk["stage_prev_vs_same_day_agree"] = f((A["stage_prev"] == A["stage_same_day"]).mean())
    la = {}
    a26 = A[A.year == 2026]
    for s in (-1, 0, 1, 2):
        la[NAMES[s]] = {"D-1(쓴 값)": f(a26.loc[a26.stage_prev == s, "SOXL_w90"].mean()),
                        "D(미래 참조라면)": f(a26.loc[a26.stage_same_day == s, "SOXL_w90"].mean())}
    chk["lookahead_placebo_2026_SOXL_w90"] = la
    # 반일장: 데이터 날 중 미국 반일장(달력)
    half = pd.to_datetime(["2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
                           "2025-07-03", "2025-11-28", "2025-12-24"])
    chk["half_days_in_data"] = int(A["day"].isin(half).sum())
    chk["half_days_note"] = "반일장은 13:00 마감이라 10:59(w90)·09:59(std) 청산과 겹치지 않는다. 단계는 그날 13:00 공식 종가(yfinance)로 정해지고 다음 날 쓴다."
    chk["soxx_split_2024"] = "SOXX 3:1 분할(2024-03) — auto_adjust=False Close는 분할 반영. RSI·z는 같은 배율 창 안에서 배율 불변이라 영향 없음."
    out["1_lookahead_stage"] = chk

    # ---- 2·3. 표 A: 구간 단위·몰림·재현
    A["month"] = A["day"].dt.strftime("%Y-%m")
    tabA = {}
    for y in YEARS:
        a = A[A.year == y].reset_index(drop=True)
        st = a["stage_prev"].values.astype(int)
        ep = runs_of(st)
        tabA[str(y)] = {}
        for s in STAGES:
            m = st == s
            if not m.any():
                continue
            row = {}
            for col in COLS:
                v = a.loc[m, col].values
                es = episode_stats(v, ep[m])
                es["year_mean"] = f(a[col].mean())
                es["minus_year_mean"] = f(v.mean() - a[col].mean())
                if y == 2026:
                    sub = a.loc[m, [col, "month", "day"]]
                    ms = sub.groupby("month")[col].sum()
                    fav = 1 if v.mean() >= 0 else -1                   # 칸 부호 쪽으로 가장 기여한 달·5일 빼기
                    bm = (ms * fav).idxmax()
                    es["drop_best_month"] = {"month": bm, "mean": f(sub.loc[sub.month != bm, col].mean()),
                                             "days_left": int((sub.month != bm).sum())}
                    order = np.argsort(-fav * v)
                    keep = np.ones(len(v), bool); keep[order[:5]] = False
                    es["drop_top5_days_toward_sign"] = f(v[keep].mean())
                    es["top5_days"] = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sub["day"].values[order[:5]]]
                    nul = shift_null(st, a[col].values, s)
                    es["shift_null_mean_all_k"] = f(nul.mean())
                    es["shift_p_two_sided_all_k_vs_null_mean"] = f(np.mean(np.abs(nul - nul.mean()) >= abs(v.mean() - nul.mean()) - 1e-12))
                row[col] = es
            # 방향 성분 d = (SOXL − SOXS)/2(양이면 SOXL 쪽), 비용 성분 = −(SOXL + SOXS)/2
            for h in ("w90", "std"):
                dv = (a.loc[m, f"SOXL_{h}"].values - a.loc[m, f"SOXS_{h}"].values) / 2
                cv = -(a.loc[m, f"SOXL_{h}"].values + a.loc[m, f"SOXS_{h}"].values) / 2
                row[f"dir_{h}"] = {"mean": f(dv.mean()), "ep_block_ci95": ep_boot_ci(dv, ep[m]),
                                   "cost_mean": f(cv.mean()), "episodes_dir_negative": int(sum(dv[ep[m] == e].mean() < 0 for e in np.unique(ep[m])))}
            tabA[str(y)][NAMES[s]] = row
        # 짝 상관
        tabA[str(y)]["corr_SOXL_SOXS_w90"] = f(np.corrcoef(a["SOXL_w90"], a["SOXS_w90"])[0, 1], 4)
        tabA[str(y)]["corr_w90_std_SOXL"] = f(np.corrcoef(a["SOXL_w90"], a["SOXL_std"])[0, 1], 4)
    out["2_3_tableA_episodes"] = tabA

    # 4년 합친 단계별(구간 묶음, 해 경계에서 구간 끊음)
    pooled = {}
    A["ep"] = -1
    off = 0
    for y in YEARS:
        ix = A.index[A.year == y]
        A.loc[ix, "ep"] = runs_of(A.loc[ix, "stage_prev"].values) + off
        off = A["ep"].max() + 1
    for s in STAGES:
        m = A["stage_prev"].values == s
        if not m.any():
            continue
        row = {}
        for h in ("w90", "std"):
            dv = (A.loc[m, f"SOXL_{h}"].values - A.loc[m, f"SOXS_{h}"].values) / 2
            # 해 평균 빼기(해마다 장세 차를 지움): 칸 − 그 해 전체 평균
            ym = A.groupby("year")[[f"SOXL_{h}", f"SOXS_{h}"]].transform("mean")
            dvx = ((A[f"SOXL_{h}"] - ym[f"SOXL_{h}"]) - (A[f"SOXS_{h}"] - ym[f"SOXS_{h}"])).values[m] / 2
            row[h] = {"days": int(m.sum()), "episodes": int(A.loc[m, "ep"].nunique()),
                      "SOXL": f(A.loc[m, f"SOXL_{h}"].mean()), "SOXS": f(A.loc[m, f"SOXS_{h}"].mean()),
                      "SOXS_ep_ci95": ep_boot_ci(A.loc[m, f"SOXS_{h}"].values, A.loc[m, "ep"].values),
                      "dir": f(dv.mean()), "dir_ep_ci95": ep_boot_ci(dv, A.loc[m, "ep"].values),
                      "dir_minus_year": f(dvx.mean()), "dir_minus_year_ep_ci95": ep_boot_ci(dvx, A.loc[m, "ep"].values)}
        pooled[NAMES[s]] = row
    out["4_pooled_2023_2026_tableA"] = pooled

    # 해마다 같은 부호 칸(4개 단계 × 4열 = 16칸, 극하방 제외)
    sign = {}
    for s in (-1, 0, 1, 2):
        for col in COLS:
            vals = [tabA[str(y)][NAMES[s]][col]["day_mean"] for y in YEARS]
            minus = [tabA[str(y)][NAMES[s]][col]["minus_year_mean"] for y in YEARS]
            sign[f"{NAMES[s]}_{col}"] = {"means": vals, "same_sign_4y": bool(len(set(np.sign(vals))) == 1),
                                         "minus_year_mean": minus, "minus_year_same_sign_4y": bool(len(set(np.sign(minus))) == 1)}
    out["5_sign_consistency_tableA"] = sign
    # 기대 개수(칸 독립 가정): 16 × 2 × (1/2)^4 = 2. 해마다 열의 해 평균 부호가 같으면 기대가 더 높다.
    yr_sign = {col: [f(A.loc[A.year == y, col].mean()) for y in YEARS] for col in COLS}
    out["5_year_means"] = yr_sign

    # ---- 6. 표 A·B 칸 p 분포
    ps = []
    for y in YEARS:
        for r in J["tableA"][str(y)]["cells"]:
            if r["p_shift"] is not None:
                ps.append(("A", y, r["stage_name"], r["col"], r["p_shift"], r["p_family"], r["mean"]))
        for r in J["tableB"][str(y)]["cells"]:
            if r["p_shift"] is not None:
                ps.append(("B", y, r["stage_name"], r["side"], r["p_shift"], r["p_family"], r["daily_mean"]))
    P = pd.DataFrame(ps, columns=["tab", "year", "stage", "col", "p", "pf", "mean"])
    multi = {}
    for (t, y), g in P.groupby(["tab", "year"]):
        multi[f"{t}{y}"] = {"cells": int(len(g)), "p<=0.05": int((g.p <= 0.05).sum()), "p<=0.10": int((g.p <= 0.10).sum()),
                            "expected_0.05": f(0.05 * len(g), 2), "min_p": f(g.p.min()), "min_pf": f(g.pf.min())}
    g26 = P[P.year == 2026]
    multi["2026_A+B"] = {"cells": int(len(g26)), "p<=0.05": int((g26.p <= 0.05).sum()), "p<=0.10": int((g26.p <= 0.10).sum()),
                         "expected_0.05": f(0.05 * len(g26), 2), "expected_0.10": f(0.10 * len(g26), 2)}
    multi["all_years_A+B"] = {"cells": int(len(P)), "p<=0.05": int((P.p <= 0.05).sum()), "expected_0.05": f(0.05 * len(P), 2),
                              "list_p<=0.05": P[P.p <= 0.05].round(4).values.tolist()}
    # 2026 칸 p의 균등성(KS)
    from scipy import stats
    multi["2026_p_KS_vs_uniform"] = f(stats.kstest(g26.p, "uniform").pvalue)
    out["6_multiplicity"] = multi

    # ---- 7. 표 B 2026 구간 단위·몰림
    B["month"] = B["day"].dt.strftime("%Y-%m")
    b26 = B[B.year == 2026].reset_index(drop=True)
    st = b26["stage_prev"].values.astype(int)
    ep = runs_of(st)
    n = len(b26)
    tabB = {}
    for side in ("user", "mirror"):
        for s in (-1, 0, 1, 2):
            m = st == s
            v = b26.loc[m, f"{side}_r_w90"].values
            es = episode_stats(v, ep[m])            # 단계 날당 평균(사건 없는 날 0)
            es["contrib_to_year_daily"] = f(v.sum() / n)
            ms = b26.loc[m].groupby("month")[f"{side}_r_w90"].sum()
            fav = 1 if v.mean() >= 0 else -1
            bm = (ms * fav).idxmax()
            es["drop_best_month"] = {"month": bm, "contrib_left": f(b26.loc[m & (b26.month != bm).values, f"{side}_r_w90"].sum() / n)}
            tabB[f"{side}_{NAMES[s]}"] = es
    # 면 전체(단계 무관) — 단계로 나누기 전 효과
    for side in ("user", "mirror"):
        v = b26[f"{side}_r_w90"].values
        tabB[f"{side}_all"] = {"daily_mean": f(v.mean()), "ci95": boot_ci(v), "trades": int(b26[f"{side}_n"].sum())}
    out["7_tableB_2026_episodes"] = tabB

    # ---- 8. C 재현·구간 단위
    c26 = C[C.year == 2026].reset_index(drop=True)
    for h in ("trend", "counter"):
        side = np.where(c26["stage_prev"] <= -1, "user" if h == "trend" else "mirror",
                        np.where(c26["stage_prev"] >= 1, "mirror" if h == "trend" else "user", ""))
        rec = np.where(side == "user", b26["user_r_w90"], np.where(side == "mirror", b26["mirror_r_w90"], 0.0))
        assert np.allclose(rec, c26[f"{h}_r_w90"].values)
    stC = c26["stage_prev"].values.astype(int)
    epC = runs_of(np.sign(stC))                      # 같은 면을 쓰는 연속 날 묶음
    resC = {}
    for h in ("trend", "counter"):
        v = c26[f"{h}_r_w90"].values
        resC[h] = {"daily_mean": f(v.mean()), "recomputed_from_B": True,
                   "sign_run_block_ci95": ep_boot_ci(v, epC), "sign_runs": int(len(np.unique(epC)))}
    out["8_C_2026"] = resC
    # 2026 SOXS_std 해 전체(단계 없이) — 단계 나누기 전에 이미 있는 기울기
    a26 = A[A.year == 2026]
    out["9_no_stage_2026"] = {col: {"mean": f(a26[col].mean()), "ci95": boot_ci(a26[col].values)} for col in COLS}
    out["9_no_stage_by_year_dir_std"] = {str(y): f(((A.loc[A.year == y, "SOXL_std"] - A.loc[A.year == y, "SOXS_std"]) / 2).mean()) for y in YEARS}

    json.dump(out, open(os.path.join(RES, "audit5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()


def pooled_shift(nsh=5000, seed=SEED):
    """4년 합친 칸(단계 4 × 열 4, 극하방 제외): 해마다 따로 원형 이동(k ∈ [20, n−20])해 합친 평균의 귀무, 칸 p와 16칸 최대 |t| 보정 p."""
    A = pd.read_csv(os.path.join(RES, "stage5_daily_A.csv"), parse_dates=["day"])
    rng = np.random.default_rng(seed)
    st_y = {y: A.loc[A.year == y, "stage_prev"].values.astype(int) for y in YEARS}
    V_y = {y: A.loc[A.year == y, COLS].values for y in YEARS}
    cells = [(s, j) for s in (-1, 0, 1, 2) for j in range(len(COLS))]

    def stat(stmap):
        out = []
        for s, j in cells:
            num = sum(((stmap[y] == s) * V_y[y][:, j]).sum() for y in YEARS)
            den = sum((stmap[y] == s).sum() for y in YEARS)
            out.append(num / den)
        return np.array(out)
    obs = stat(st_y)
    null = np.empty((nsh, len(cells)))
    for b in range(nsh):
        rot = {}
        for y in YEARS:
            n = len(st_y[y]); k = rng.integers(MINSH, n - MINSH + 1)
            rot[y] = np.roll(st_y[y], k)
        null[b] = stat(rot)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    t = (obs - mu) / sd
    tn = np.abs((null - mu) / sd)
    p = (1 + (tn >= np.abs(t)).sum(0)) / (nsh + 1)
    pf = (1 + (tn.max(1)[:, None] >= np.abs(t)[None, :]).sum(0)) / (nsh + 1)
    return {f"{NAMES[s]}_{COLS[j]}": {"pooled_mean": f(obs[c]), "null_mean": f(mu[c]), "t": f(t[c], 2), "p": f(p[c]), "p_family16": f(pf[c])}
            for c, (s, j) in enumerate(cells)}


if __name__ == "__main__" and os.environ.get("AUDIT5_POOLED") == "1":
    r = pooled_shift()
    o = json.load(open(os.path.join(RES, "audit5.json"), encoding="utf-8"))
    o["10_pooled_shift_2023_2026"] = r
    json.dump(o, open(os.path.join(RES, "audit5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(r, ensure_ascii=False, indent=0))
