"""BRIEF4 '통과 뒤 검증' 감사 — C3 H4(사용자 방향 x = 3%, 신호 봉 SOXL 윗꼬리 ÷ 봉 길이 ≥ 0.5, r_w90).

감사 담당이 결과를 깎을 결함을 찾으려고 다시 잰다. 판정 규칙(BRIEF4)은 바꾸지 않고, 깎는 근거만 수치로 낸다.
엔진(Engine)·sim_trade를 쓰지 않는 자체 체결 계산(sim)으로 사건 사슬을 다시 만들어 사건 표와 맞춘 뒤(부분 독립 재구현),
같은 계산으로 감도를 잰다. 자료 배열과 비용표만 lib에서 읽는다.

사용: cd code && py audit4_H4.py → results/audit4_H4.json
"""
import hashlib, json, os, sys, time
from datetime import datetime, timedelta

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF4: 2023~2026 전체
import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, RTH0, idx, hm, cutoff_idx, load_costs, bucket, ET, KST  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
EPS, X, TGT, WK = 1e-9, 0.03, 0.01, 0.5
CUT = idx("11:00")
COSTS = load_costs()
SEED = 20261103
T0 = time.time()


def fnum(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


# ----------------------------------------------------------------------------- 자체 체결 계산
def sim(D, d, dec, sym, cut, target=TGT, stop=None, lag=1, fill="close", hs_in=None, hs_out_mult=1.0, cm=1.0, touch=1.0002):
    """결정 봉 dec → 체결 봉 f = dec + lag(종가 또는 시가) + 반 호가. +target 지정가(고가 ≥ 목표 × touch), 손절 우선, 못 닿으면 cut−1 종가 − 반 호가."""
    A = D.A[sym]
    f = dec + lag
    if f >= cut:
        return None
    base = A["Close"][d, f] if fill == "close" else A["Open"][d, f]
    h_in = (COSTS[sym][bucket(f)] * cm) if hs_in is None else hs_in
    px = base * (1 + h_in)
    tg = px * (1 + target) if target else None
    sp = px * (1 - stop) if stop else None
    j0 = f + 1 if fill == "close" else f
    hs_out = lambda j: COSTS[sym][bucket(j)] * cm * hs_out_mult   # noqa: E731
    for j in range(j0, cut):
        if sp is not None and A["Low"][d, j] <= sp:
            return (min(A["Open"][d, j], sp) * (1 - hs_out(j)) / px - 1) * 100, f, j, "stop", px
        if tg is not None and A["High"][d, j] >= tg * touch:
            return target * 100, f, j, "target", px
    j = cut - 1
    return (A["Close"][d, j] * (1 - hs_out(j)) / px - 1) * 100, f, j, "cutoff", px


def wick(D, d, i, sym="SOXL"):
    A = D.A[sym]
    h, l, o, c = A["High"][d, i], A["Low"][d, i], A["Open"][d, i], A["Close"][d, i]
    return (h - max(o, c)) / (h - l) if h > l else 0.0


def chain_day(D, d, mode, cut=CUT, x=X, wk=WK, **kw):
    """사용자 방향 하루 사슬. mode:
    'all'   = 조건 없는 사슬(사건 표와 같은 규칙) — 모든 첫 교차에서 산다(조건 값은 기록만).
    'phantom' = 사건 표 거르기와 같음: 첫 교차에서 윗꼬리 < wk면 가상 매매 청산까지 기다렸다 L을 다시 잰다(c3의 암묵 규칙).
    'reset' = 첫 교차가 조건 불만족이면 그 봉 다음부터 L을 다시 잰다(가상 매매 없음).
    'any'   = L은 그대로 두고, 종가 ≥ L × (1 + x)이고 윗꼬리 ≥ wk인 아무 봉에서나 산다(사람이 화면을 보며 하는 판)."""
    A = D.A["SOXL"]
    out = []
    s0, i = RTH0, RTH0
    last_dec = cut - 16
    while i <= last_dec:
        L = A["Low"][d, s0:i + 1].min()
        c = A["Close"][d, i]
        if np.isfinite(c) and L > 0 and c >= L * (1 + x):
            w = wick(D, d, i)
            ok = w >= wk - EPS
            if mode == "any" and not ok:
                i += 1
                continue
            if mode == "reset" and not ok:
                s0, i = i + 1, i + 1
                continue
            r = sim(D, d, i, "SOXS", cut, **kw)
            if r is None:
                break
            ret, f, j, why, px = r
            if mode in ("all", "phantom") or ok:
                out.append({"d": d, "dec_i": i, "wick": w, "ok": bool(ok), "r": ret, "why": why, "exit_i": j,
                            "fill_i": f, "L": L, "ratio": c / L, "s0": s0})
            s0, i = j + 1, j + 1
            continue
        i += 1
    return out


def day_series(trades, days_idx, col="r"):
    """날 색인 목록마다 하루 손익(곱 − 1, %)."""
    m = {}
    for t in trades:
        m.setdefault(t["d"], []).append(t[col])
    return np.array([(np.prod([1 + v / 100 for v in m[d]]) - 1) * 100 if d in m else 0.0 for d in days_idx])


def boot(dly, nb=10000, seed=20261102):
    ix = np.random.default_rng(seed).integers(0, len(dly), (nb, len(dly)))
    bd = dly[ix].mean(1)
    return {"daily_mean": fnum(dly.mean()), "ci95": [fnum(np.percentile(bd, 2.5)), fnum(np.percentile(bd, 97.5))],
            "p_one_sided": fnum((bd <= 0).mean())}


def tsum(tr, days_idx, b=True):
    r = np.array([t["r"] for t in tr]) if tr else np.array([])
    dly = day_series(tr, days_idx)
    o = {"trades": int(len(r)), "trade_mean": fnum(r.mean()) if len(r) else None,
         "hit_rate": fnum(np.mean([t["why"] == "target" for t in tr])) if tr else None,
         "worst": fnum(r.min()) if len(r) else None, "daily_mean": fnum(dly.mean()), "sum_pct": fnum(r.sum()) if len(r) else 0.0}
    if b and len(r):
        o.update({k: v for k, v in boot(dly).items() if k != "daily_mean"})
    return o


def main():
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    D = Data("alp1m_long")
    print(f"자료 {D.days[0]}~{D.days[-1]} {len(D.days)}일 ({time.time() - T0:.0f}s)", flush=True)
    years = {y: [k for k, dd in enumerate(D.days) if dd.year == y] for y in (2023, 2024, 2025, 2026)}
    d26 = years[2026]
    assert len(d26) == 207
    out = {"brief": "BRIEF4 '통과 뒤 검증' — C3 H4 감사", "units": "% (1.0 = 1%)",
           "code_sha256_16": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]}

    u = ev[(ev["side"] == "user") & np.isclose(ev["x"], X)].copy()
    u["h4"] = u["c10_wick"] >= WK - EPS
    u26 = u[u["year"] == 2026]
    h = u26[u26["h4"]]
    assert len(h) == 9, len(h)

    # ------------------------------------------------------------------ 0. 사슬 다시 만들기(자체 계산) = 사건 표?
    rec = {y: [] for y in years}
    for y, dl in years.items():
        for d in dl:
            rec[y] += chain_day(D, d, "all")
    re26 = pd.DataFrame(rec[2026])
    tab = u26[["d", "dec_i", "r_w90", "c10_wick", "exit_i"]].reset_index(drop=True)
    m = re26.merge(tab, on=["d", "dec_i"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    out["s0_rebuild"] = {
        "events_rebuilt_2026": int(len(re26)), "events_table_2026": int(len(tab)),
        "only_rebuilt": int((m["_merge"] == "left_only").sum()), "only_table": int((m["_merge"] == "right_only").sum()),
        "max_abs_diff_r": fnum((both["r"] - both["r_w90"]).abs().max()),
        "max_abs_diff_wick": fnum((both["wick"] - both["c10_wick"]).abs().max()),
        "all_years_rebuilt": {y: len(v) for y, v in rec.items()},
        "all_years_table": {int(y): int((u["year"] == y).sum()) for y in years}}
    print("s0", out["s0_rebuild"], flush=True)

    # ------------------------------------------------------------------ 1. 부트스트랩 퇴화: p ≈ (1 − k/207)^207
    k_win = int((h["r_w90"] > 0).sum())
    p_analytic = (1 - k_win / 207) ** 207
    out["s1_bootstrap_degeneracy"] = {
        "win_days": k_win, "loss_days": int((h["r_w90"] <= 0).sum()),
        "analytic_p_no_win_day_drawn": p_analytic, "reported_p": 0.0004,
        "mirror_5of5_analytic": (1 - 5 / 207) ** 207, "mirror_reported_p": 0.0067,
        "note": "이긴 날 8개가 하나도 안 뽑힐 확률 = 보고 p. 진 매매의 크기·모집단의 손실 꼬리와 무관하게 '이긴 날 수'만으로 p가 정해진다"}

    # ------------------------------------------------------------------ 2. 작은 표본에 맞는 귀무들
    r_all = u26["r_w90"].values
    n = len(h)
    obs = float(h["r_w90"].mean())
    rng = np.random.default_rng(SEED)
    NP = 100000
    perm = np.array([rng.choice(len(r_all), n, replace=False) for _ in range(NP)])
    p_perm_all = float((r_all[perm].mean(1) >= obs - 1e-12).mean())
    rest = u26.loc[~u26["h4"], "r_w90"].values
    perm2 = np.array([rng.choice(len(rest), n, replace=False) for _ in range(50000)])
    p_perm_rest = float((rest[perm2].mean(1) >= obs - 1e-12).mean())
    pool_all = u["r_w90"].values                          # 2023~2026 사용자 방향 x = 3 전체 716건
    perm3 = rng.integers(0, len(pool_all), (NP, n))
    p_pool_all = float((pool_all[perm3].mean(1) >= obs - 1e-12).mean())
    hits = int(h["hit"].sum())
    hr26, hr_all = float(u26["hit"].mean()), float(u["hit"].mean())
    sd_g = float(u26["r_w90"].std(ddof=1))
    out["s2_small_sample_nulls"] = {
        "obs_trade_mean": obs, "n": n, "hits": hits,
        "perm_9_of_258_2026": p_perm_all, "perm_9_of_249_rest_2026": p_perm_rest,
        "draw_9_from_2023_2026_pool_716": p_pool_all,
        "binom_ge8_at_2026_hit_0.713": float(binom.sf(hits - 1, n, hr26)),
        "binom_ge8_at_2023_2026_hit": {"rate": hr_all, "p": float(binom.sf(hits - 1, n, hr_all))},
        "c1_floor_t": {"sd_group_2026": sd_g, "se_floor": sd_g / np.sqrt(n), "t": obs / (sd_g / np.sqrt(n)),
                       "note": "C1이 같은 +1.0 점 질량 퇴화를 피하려 쓴 se 바닥 σ_g/√N. C1 귀무 최대 t 95% = 3.89(9,676칸), C3 참고 최대 t 95% = 2.36"}}
    print("s2", json.dumps(out["s2_small_sample_nulls"], ensure_ascii=False), flush=True)

    # ------------------------------------------------------------------ 3. 10개 가설 가족에서 사건 섞기 minP 보정
    HM = {
        "H1": u26["c01_time"].between(20 - EPS, 40 + EPS), "H2": u26["c17_qqq_lv"] == "<0.5%",
        "H3": u26["c18_nvda_lv"] == "<x/3", "H4": u26["c10_wick"] >= 0.5 - EPS, "H5": u26["c09_rsidiv_lv"] == "예",
        "H6": u26["c12_risevol_lv"] == "<1", "H7": u26["c15_vs_prevhigh_lv"] == "±0.5% 안",
        "H8": (u26["c05_gap"] <= -2 + EPS) & (u26["c04_vs_prevclose"].abs() <= 0.5 + EPS),
        "H9": u26["c22_preveff_lv"] == "≥0.6", "H10": u26["c07_vwap_lv"] == "≥3%"}
    M = np.vstack([HM[k].fillna(False).astype(bool).values for k in HM]).astype(float)   # 10 × 258
    nh = M.sum(1)
    obs_h = (M @ r_all) / nh
    B = 20000
    rr = np.vstack([r_all[rng.permutation(len(r_all))] for _ in range(B)]).T              # 258 × B
    null = (M @ rr) / nh[:, None]                                                         # 10 × B
    p_h = (null >= obs_h[:, None] - 1e-12).mean(1)
    srt = np.sort(null, axis=1)
    pb = np.vstack([1 - np.searchsorted(srt[k], null[k], side="left") / B for k in range(len(nh))])   # 각 섞기의 가설별 p
    minp = pb.min(0)
    adj = {k: float((minp <= p_h[j] + 1e-12).mean()) for j, k in enumerate(HM)}
    out["s3_family_permutation_minP"] = {
        "B": B, "stat": "가설 사건 r_w90 평균(사건 수 고정이라 하루 평균 합과 같은 순서)",
        "per_hyp": {k: {"n": int(nh[j]), "obs_trade_mean": fnum(obs_h[j]), "perm_p": fnum(p_h[j]), "minP_adj_p": adj[k]}
                    for j, k in enumerate(HM)},
        "note": "사건 결과를 2026 사용자 방향 258건 안에서 섞는다(손실 꼬리 포함). 같은 날 사건 상관은 무시"}
    print("s3 H4", out["s3_family_permutation_minP"]["per_hyp"]["H4"], flush=True)

    # ------------------------------------------------------------------ 4. 칸 수 대비: C1식 한·두 겹 작은 칸에서 H4만큼 좋은 칸이 우연보다 많은가
    cols = [c for c in u26.columns if c.endswith("_lv")]
    singles, names = [], []
    for c in cols:
        for lv in u26[c].cat.categories:
            mk = (u26[c] == lv).fillna(False).values
            if mk.sum() >= 1:
                singles.append(mk)
                names.append((c, lv))
    S = np.array(singles)
    cells = [S[a] for a in range(len(S))]
    cname = [names[a] for a in range(len(S))]
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            if names[a][0] != names[b][0]:
                cells.append(S[a] & S[b])
                cname.append((names[a], names[b]))
    C = np.array(cells, dtype=float)
    cn = C.sum(1)
    keep = (cn >= 9) & (cn <= 24)
    Ck, cnk = C[keep], cn[keep]
    obs_cell = (Ck @ r_all) / cnk
    thr = obs
    n_obs_good = int((obs_cell >= thr - 1e-12).sum())
    BB = 300
    rr2 = np.vstack([r_all[rng.permutation(len(r_all))] for _ in range(BB)]).T
    cnt = (((Ck @ rr2) / cnk[:, None]) >= thr - 1e-12).sum(0)
    good_ids = [str(cname[np.nonzero(keep)[0][q]]) for q in np.nonzero(obs_cell >= thr - 1e-12)[0]][:15]
    out["s4_cell_multiplicity"] = {
        "cells_user_x3_2026_singles_pairs": int(len(C)), "cells_n_9_to_24": int(keep.sum()),
        "threshold_trade_mean": thr, "observed_cells_ge_threshold": n_obs_good,
        "perm_expected_cells_ge_threshold": {"mean": fnum(cnt.mean()), "q05": fnum(np.percentile(cnt, 5)), "q95": fnum(np.percentile(cnt, 95))},
        "examples": good_ids,
        "note": "BRIEF4 조건 28개의 한·두 겹 칸(C1 칸 정의, 사용자 방향 x = 3, 2026) 중 사건 9~24건짜리. H4 평균 이상인 칸 수를 사건 섞기 기대와 비교"}
    print("s4", out["s4_cell_multiplicity"]["observed_cells_ge_threshold"], out["s4_cell_multiplicity"]["perm_expected_cells_ge_threshold"], flush=True)

    # ------------------------------------------------------------------ 5. 월 몰림·상하위 날
    hh = h.copy()
    bym = hh.groupby("month")["r_w90"].sum().sort_values(ascending=False)
    dly26 = pd.Series(0.0, index=[D.days[k] for k in d26])
    for _, e in hh.iterrows():
        dly26[e["day"].date()] = e["r_w90"]
    def drop_months(ms):
        keepd = [dd for dd in dly26.index if f"{dd.year}-{dd.month:02d}" not in ms]
        s = dly26[keepd]
        sub = hh[~hh["month"].isin(ms)]
        return {"months_dropped": list(ms), "days": len(keepd), "trades": int(len(sub)),
                "trade_mean": fnum(sub["r_w90"].mean()) if len(sub) else None, "daily_mean": fnum(s.mean())}
    bym_l = list(bym.index)
    # 달별 매매 수가 같으면 합이 같아서 가장 좋은 달이 여럿이다 → 매매 수가 가장 많은 달부터
    cnt_m = hh.groupby("month").size().sort_values(ascending=False)
    srt_d = dly26.sort_values()
    out["s5_concentration"] = {
        "by_month_sum": {k: fnum(v) for k, v in bym.items()}, "by_month_trades": {k: int(v) for k, v in cnt_m.items()},
        "drop_best1_by_count": drop_months(list(cnt_m.index[:1])), "drop_best2_by_count": drop_months(list(cnt_m.index[:2])),
        "drop_top5_days": {"daily_mean": fnum(srt_d.iloc[:-5].mean()), "trades_left": int((srt_d.iloc[:-5] != 0).sum()),
                           "trade_mean_left": fnum(srt_d.iloc[:-5][srt_d.iloc[:-5] != 0].mean())},
        "drop_bottom5_days": {"daily_mean": fnum(srt_d.iloc[5:].mean())},
        "trim_both5": {"daily_mean": fnum(srt_d.iloc[5:-5].mean())},
        "note": "이긴 8건이 모두 정확히 +1.0이라 '가장 좋은 달'은 매매 수가 가장 많은 달(10월 3건)이다"}

    # ------------------------------------------------------------------ 6. 체결 감도(자체 sim, H4 2026 사건 9개 + 2023~2025 16개)
    def resim(sub, **kw):
        res = []
        for _, e in sub.iterrows():
            r = sim(D, int(e["d"]), int(e["dec_i"]), "SOXS", CUT, **kw)
            res.append(None if r is None else {"d": int(e["d"]), "r": r[0], "why": r[3]})
        res = [t for t in res if t is not None]
        return res
    hold = u[(u["year"] < 2026) & u["h4"]]
    sens = {}
    variants = {
        "base": {}, "cost2x": {"cm": 2.0}, "cost3x": {"cm": 3.0},
        "soxs_hs_p75_0.116%": {"hs_in": 0.00116, "hs_out_mult": 0.00116 / 0.00052},
        "soxs_hs_max_0.288%": {"hs_in": 0.00288, "hs_out_mult": 0.00288 / 0.00052},
        "delay2_lag2": {"lag": 2}, "delay3_lag3": {"lag": 3}, "next_open_lag0": {"fill": "open"},
        "stop1": {"stop": 0.01}, "stop1.5": {"stop": 0.015}, "stop2": {"stop": 0.02}, "stop3": {"stop": 0.03},
        "touch1.0005": {"touch": 1.0005}, "touch1.001": {"touch": 1.001}, "touch1.002": {"touch": 1.002},
        "delay2_cost2x_touch1.001": {"lag": 2, "cm": 2.0, "touch": 1.001}}
    for nm, kw in variants.items():
        t26 = resim(h, **kw)
        told = resim(hold, **kw)
        sens[nm] = {"2026": tsum(t26, d26), "2023_2025_pooled": tsum(told, years[2023] + years[2024] + years[2025], b=False),
                    "2026_trades": [round(t["r"], 3) for t in t26]}
    base_chk = max(abs(a - b) for a, b in zip([t["r"] for t in resim(h)], h["r_w90"].values))
    out["s6_execution_sensitivity"] = {"base_vs_table_max_abs_diff": base_chk, "variants": sens,
                                       "note": "같은 9개 결정 봉에서 체결만 바꿈(지연 = 결정 1·2봉 늦춤과 같음: 체결 봉 dec+2·dec+3). 반 호가 0.116%=SOXS 실측 p75, 0.288%=날별 최대(report.md §7), 청산 쪽도 같은 배수"}
    print("s6", {k: (v["2026"]["trade_mean"], v["2026"]["daily_mean"], v["2026"].get("p_one_sided"), v["2023_2025_pooled"]["trade_mean"]) for k, v in sens.items()}, flush=True)

    # 목표 닿음 여유: MFE − 1.02%
    marg = []
    for _, e in h[h["hit"] == 1].iterrows():
        marg.append({"day": str(e["day"].date()), "mfe_pct": fnum(e["mfe"]), "margin_over_touch_pct": fnum(e["mfe"] - (1.01 * 1.0002 - 1) * 100),
                     "min_to_target": fnum(e["min_to_target"]), "mae": fnum(e["mae"])})
    out["s6b_target_touch_margin"] = marg

    # ------------------------------------------------------------------ 7. 2023~2025 재현과 그 해 무작위 대비
    rep = {}
    for y in (2023, 2024, 2025, 2026):
        uy = u[u["year"] == y]
        hy = uy[uy["h4"]]
        ry = uy["r_w90"].values
        ny, oy = len(hy), float(hy["r_w90"].mean())
        pp = np.array([ry[rng.choice(len(ry), ny, replace=False)].mean() for _ in range(50000)])
        rep[y] = {"events": int(len(uy)), "h4_trades": ny, "h4_trade_mean": oy, "h4_hit": fnum(hy["hit"].mean()),
                  "year_all_trade_mean": fnum(ry.mean()), "year_all_hit": fnum(uy["hit"].mean()),
                  "perm_p_h4_ge_random_same_n": fnum((pp >= oy - 1e-12).mean()),
                  "direction_component_trade_mean": fnum(((hy["r_w90"] - hy["r_follow"]) / 2).mean()),
                  "follow_trade_mean": fnum(hy["r_follow"].mean())}
    # 2023~2026 합친 층화 섞기(해마다 그 해 안에서)
    obs_pool = float(u[u["h4"]]["r_w90"].mean())
    stats_p = []
    groups = [(u[u["year"] == y]["r_w90"].values, int(u[(u["year"] == y) & u["h4"]].shape[0])) for y in (2023, 2024, 2025, 2026)]
    for _ in range(50000):
        s = 0.0
        for ry, ny in groups:
            s += ry[rng.choice(len(ry), ny, replace=False)].sum()
        stats_p.append(s / sum(g[1] for g in groups))
    stats_p = np.array(stats_p)
    old = u[(u["year"] < 2026)]
    obs_old = float(old[old["h4"]]["r_w90"].mean())
    stats_o = []
    for _ in range(50000):
        s = 0.0
        for ry, ny in groups[:3]:
            s += ry[rng.choice(len(ry), ny, replace=False)].sum()
        stats_o.append(s / sum(g[1] for g in groups[:3]))
    stats_o = np.array(stats_o)
    out["s7_replication"] = {"by_year": rep,
                             "pooled_2023_2026": {"trades": int(u["h4"].sum()), "trade_mean": obs_pool, "hit": fnum(u[u["h4"]]["hit"].mean()),
                                                  "strat_perm_p": fnum((stats_p >= obs_pool - 1e-12).mean())},
                             "pooled_2023_2025": {"trades": int(old["h4"].sum()), "trade_mean": obs_old,
                                                  "strat_perm_p": fnum((stats_o >= obs_old - 1e-12).mean())}}
    print("s7", json.dumps(out["s7_replication"], ensure_ascii=False)[:900], flush=True)

    # 윗꼬리 이웃 문턱 × 해
    nb = {}
    for t in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6):
        row = {}
        for y in (2023, 2024, 2025, 2026):
            uy = u[u["year"] == y]
            s = uy[uy["c10_wick"] >= t - EPS]
            row[y] = {"n": int(len(s)), "trade_mean": fnum(s["r_w90"].mean()) if len(s) else None, "hit": fnum(s["hit"].mean()) if len(s) else None}
        nb[f"≥{t}"] = row
    near = u[(u["c10_wick"] - 0.5).abs() <= 0.05 + EPS]
    out["s7b_wick_threshold_by_year"] = nb
    out["s7c_near_threshold"] = {"events_wick_0.45_0.55_all_years": int(len(near)),
                                 "exactly_0.5": int((u["c10_wick"].sub(0.5).abs() <= EPS).sum()),
                                 "h4_2026_within_0.05": int(((h["c10_wick"] - 0.5).abs() <= 0.05 + EPS).sum()),
                                 "h4_2026_wicks": [round(v, 4) for v in sorted(h["c10_wick"].values)]}

    # ------------------------------------------------------------------ 8. 같은 사건 추종 대비(방향 성분)
    both_hit = int(((h["hit"] == 1) & (h["r_follow"] >= 1.0 - 1e-9)).sum())
    out["s8_direction"] = {"h4_trade_mean": obs, "follow_trade_mean": fnum(h["r_follow"].mean()),
                           "direction_component": fnum(((h["r_w90"] - h["r_follow"]) / 2).mean()),
                           "both_legs_hit_+1%": both_hit, "of": n,
                           "follow_losses": [round(v, 3) for v in h.loc[h["r_follow"] < 0, "r_follow"].values]}

    # ------------------------------------------------------------------ 9. 사람이 실제로 하는 판(단독 실행)
    alone = {}
    for mode in ("phantom", "reset", "any"):
        row = {}
        for y, dl in years.items():
            tr = []
            for d in dl:
                tr += [t for t in chain_day(D, d, mode) if t["ok"]]
            row[y] = tsum(tr, dl, b=(y == 2026))
        alone[mode] = row
    # 'any'에 −2% 손절, 지연 2분
    for nm, kw in {"any_stop2": {"stop": 0.02}, "any_delay2": {"lag": 2}}.items():
        row = {}
        for y, dl in years.items():
            tr = []
            for d in dl:
                tr += [t for t in chain_day(D, d, "any", **kw) if t["ok"]]
            row[y] = tsum(tr, dl, b=(y == 2026))
        alone[nm] = row
    out["s9_standalone"] = {"modes": alone,
                            "note": "phantom = 사건 표 거르기(첫 교차 불만족이면 가상 SOXS 매매가 끝날 때까지 기다림, 사람은 이렇게 안 함), "
                                    "reset = 첫 교차 불만족이면 다음 봉부터 L 다시, any = L 유지·조건 맞는 아무 봉"}
    print("s9", {m_: (v[2026]["trades"], v[2026]["trade_mean"], v[2026]["daily_mean"], v[2026].get("p_one_sided"),
                      [v[y]["trade_mean"] for y in (2023, 2024, 2025)]) for m_, v in alone.items()}, flush=True)

    # ------------------------------------------------------------------ 10. 한국 시각·주문 가능 시간(08~24 KST)
    rows = []
    for _, e in h.iterrows():
        dd = e["day"].date()
        cut_a = cutoff_idx(dd, "actual")
        def kst(i):
            t = datetime(dd.year, dd.month, dd.day, tzinfo=ET) + timedelta(minutes=i + 240)
            return t.astimezone(KST).strftime("%m-%d %H:%M")
        ok = int(e["dec_i"]) <= cut_a - 16
        ra = sim(D, int(e["d"]), int(e["dec_i"]), "SOXS", cut_a) if ok else None
        rows.append({"day": str(dd), "dst": cut_a == idx("11:00"), "signal_close_et": hm(int(e["dec_i"]) + 1),
                     "entry_fill_et": hm(int(e["fill_i"]) + 1), "exit_et": hm(int(e["exit_i"]) + 1),
                     "entry_kst": kst(int(e["fill_i"]) + 1), "exit_kst": kst(int(e["exit_i"]) + 1),
                     "r_w90": fnum(e["r_w90"]), "allowed_by_24kst": ok, "r_actual_24kst": fnum(ra[0]) if ra else None})
    act = [r_["r_actual_24kst"] for r_ in rows if r_["r_actual_24kst"] is not None]
    out["s10_kst"] = {"trades": rows, "actual_24kst_trades": len(act), "actual_24kst_trade_mean": fnum(np.mean(act)) if act else None,
                      "actual_24kst_daily_mean": fnum(sum(act) / 207)}
    # 겨울판(매일 10:00 ET 마감) 단독 any
    row = {}
    for y, dl in years.items():
        tr = []
        for d in dl:
            tr += [t for t in chain_day(D, d, "any", cut=idx("10:00")) if t["ok"]]
        row[y] = tsum(tr, dl, b=(y == 2026))
    out["s10b_std_any"] = row
    # 실제 24:00 KST 마감 단독 any
    row = {}
    for y, dl in years.items():
        tr = []
        for d in dl:
            tr += [t for t in chain_day(D, d, "any", cut=cutoff_idx(D.days[d], "actual")) if t["ok"]]
        row[y] = tsum(tr, dl, b=(y == 2026))
    out["s10c_actual_any"] = row

    # 신호 봉 크기(사람 눈으로 0.5를 가를 수 있나): 봉 길이·윗꼬리 센트
    sb = []
    for _, e in pd.concat([h, hold]).iterrows():
        d, i = int(e["d"]), int(e["dec_i"])
        A = D.A["SOXL"]
        sb.append({"day": str(e["day"].date()), "range_cents": round((A["High"][d, i] - A["Low"][d, i]) * 100, 2),
                   "wick_cents": round((A["High"][d, i] - max(A["Open"][d, i], A["Close"][d, i])) * 100, 2),
                   "close": round(float(A["Close"][d, i]), 2), "wick": round(float(e["c10_wick"]), 4), "r": round(float(e["r_w90"]), 3),
                   "vol": int(A["Volume"][d, i])})
    out["s11_signal_bars"] = sb

    with open(os.path.join(RES, "audit4_H4.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    print(f"완료 ({time.time() - T0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
