"""BRIEF6 감사 — 추세 5단계 빨강 첫날 SOXS(구현 A redfirst.py 결과를 깎을 결함 찾기, 2026-11-03).

정본: ../BRIEF6.md. 구현 A 결과: ../results/redfirst.json·redfirst_events.csv. 엔진 lib.py(고치지 않음), 단계 stage5.zone_calc.
감사 항목(모두 판정 밖, BRIEF6 규칙은 바꾸지 않는다):
 0. 재계산 대조: 단계·R1을 단순 벡터식으로 다시 만들고 O1~O4를 직접 계산해 구현 A 사건 값과 대조.
 1. 미래 참조·위약: (a) 등록 = 단계(D−1), (b) 위약 = 단계(D) 첫 빨강(그날 종가를 미리 앎), (c) 하루 늦게 = 단계(D−2) 첫 빨강,
    (d) 장중 다시 그리는 지표 = D 시가를 종가로 넣은 단계. 빨강이 보이기 시작한 때(D−1 16:00 ET)부터 들어가는 진입 시각 감도.
 2. 사건 독립성: 2026 사건 간격, 빨강 구간 길이(깜빡임), 5거래일 안 사건을 한 묶음으로 본 묶음 평균·묶음 부트스트랩.
 3. 몰림: 상위 1·2·3건 빼기, 달별 합, 긴 재현에서 하락장 해(2018·2022) 빼기.
 4. 장세: SOXX D−1 종가가 200일선 위·아래(사후 분할).
 5. 첫날 대 나머지·'전날 하락' 대조군: R1이 아닌데 전날 SOXX가 그만큼 빠진 날의 SOXS.
 6. 긴 재현 SOXS 일봉 시가 질: 해마다 시가 = 전일 종가 비율, 시가가 고저 밖, SOXS 시가→종가와 −3×SOXX 시가→종가 차이.
 7. 비용: 손익분기 호가 배수, 사건 날 09:31 가격의 1틱 반 호가, 호가 표본 날, 09:31 봉 고가 진입·10:59 봉 저가 청산(최악).
    진입·청산 시각 감도.
 8. 사람 실행: 한국 시각, scheme='actual'(24:00 KST 마감) 결과, 겨울(표준시) O1만 가능.
 9. 통계: t·윌콕슨·부호 검정, 2023~2026 모은 해 층화 무작위 p.
실행: cd code && py audit6.py → ../results/audit6.json
"""
import json, os, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF6 허용
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, Engine, RTH0, idx, bucket, load_costs, cutoff_idx  # noqa: E402
import stage5 as S5  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
CACHE = os.path.normpath(os.path.join(HERE, "..", "cache"))
SEED, NR = 20261103, 10000
ET, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")
T0 = time.time()
f = S5.fnum


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def r2(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def ld(sym, start):
    return pd.read_csv(os.path.join(CACHE, f"{sym.lower()}_1d_yf_ohlc_{start}_20261029.csv"), index_col=0, parse_dates=True)


def null_p(vals, pool, seed=SEED):
    vals, pool = np.asarray(vals, float), np.asarray(pool, float)
    n = len(vals)
    if n == 0 or len(pool) < n:
        return None, None
    rng = np.random.default_rng(seed)
    keys = rng.random((NR, len(pool)))
    pick = np.argpartition(keys, n - 1, axis=1)[:, :n]
    nm = pool[pick].mean(1)
    return float((nm >= vals.mean() - 1e-12).mean()), float(nm.mean())


def first_red(st, lag):
    """사건 t: st[t−lag] 빨강, st[t−lag−1] 값 있고 빨강 아님."""
    n = len(st)
    fin = np.isfinite(st)
    red = fin & (st <= -1)
    e = np.zeros(n, bool)
    for t in range(lag + 1, n):
        e[t] = red[t - lag] and fin[t - lag - 1] and not red[t - lag - 1]
    return e


def live_open_stage(o, c):
    """t일 단계를 t일 시가를 종가로 넣어 다시 잼(장중에 다시 그리는 일봉 지표가 09:30에 보일 값)."""
    n = len(c)
    N, k = S5.N_TR, np.sqrt((S5.N_TR - 1.0) * (2.0 * S5.N_TR - 1.0) / (6.0 * S5.N_TR))
    lc = np.log(c)
    # RSI 상태(au, ad)를 t−1까지 정확히 다시 만든다(stage5.rsi_wilder와 같은 재귀)
    d = np.r_[np.nan, np.diff(c)]
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au = np.full(n, np.nan)
    ad = np.full(n, np.nan)
    for i in range(S5.RL, n):
        if i == S5.RL:
            au[i], ad[i] = up[1:S5.RL + 1].mean(), dn[1:S5.RL + 1].mean()
        else:
            au[i] = au[i - 1] + (up[i] - au[i - 1]) / S5.RL
            ad[i] = ad[i - 1] + (dn[i] - ad[i - 1]) / S5.RL
    out = np.full(n, np.nan)
    for t in range(max(N + 1, S5.RL + 1, S5.MIN_IDX), n):
        lo = np.log(o[t])
        w_lc = np.r_[lc[t - N + 1:t], lo]
        w_lr = np.diff(np.r_[lc[t - N:t], lo])
        z = (lo - w_lc.mean()) / (np.std(w_lr, ddof=1) * k)
        dd = o[t] - c[t - 1]
        a_u = au[t - 1] + (max(dd, 0) - au[t - 1]) / S5.RL
        a_d = ad[t - 1] + (max(-dd, 0) - ad[t - 1]) / S5.RL
        r = 100.0 if a_d == 0 else (0.0 if a_u == 0 else 100 - 100 / (1 + a_u / a_d))
        out[t] = 2 if r >= S5.RHI else -2 if r <= S5.RLO else 1 if z >= S5.TH else -1 if z <= -S5.TH else 0
    return out


def main():
    out = {"brief": "BRIEF6.md", "audit_of": "redfirst.py (A)", "units": "%"}
    A = json.load(open(os.path.join(RES, "redfirst.json"), encoding="utf-8"))
    EVA = pd.read_csv(os.path.join(RES, "redfirst_events.csv"))
    X, S = ld("SOXX", "20010101"), ld("SOXS", "20100101")
    dX = pd.DatetimeIndex(X.index)
    cX, oX = X["Close"].to_numpy(float), X["Open"].to_numpy(float)
    zn, rsi, z = S5.zone_calc(cX)
    tX = {d: t for t, d in enumerate(dX)}
    yrX = np.array([d.year for d in dX])

    # ------------------------------------------------ 0. 재계산 대조
    R1 = first_red(zn, 1)
    a26 = EVA[(EVA.variant == "R1")]
    mine = set(str(d.date()) for d in dX[R1])
    theirs = set(a26.date)
    out["0_recompute"] = {"R1_dates_equal_A_all_years": mine == theirs, "n_mine": len(mine), "n_A": len(theirs)}

    D = Data("alp1m_long")
    days = [pd.Timestamp(d) for d in D.days]
    nd = len(days)
    tD = np.array([tX[d] for d in days])
    yrD = np.array([d.year for d in days])
    monD = np.array([f"{d.year}-{d.month:02d}" for d in days])
    C, O, H, L = (D.A["SOXS"][k] for k in ("Close", "Open", "High", "Low"))
    cost = load_costs()["SOXS"]

    def hs(i, cm=1.0):
        return cost[bucket(i)] * cm

    def ret(i_ent, i_x, h, cm=1.0, ent_px=None, ex_px=None):
        e = (C[:, i_ent] if ent_px is None else ent_px) * (1 + hs(i_ent, cm))
        x = (C[:, i_x] if ex_px is None else ex_px) * (1 - hs(i_x, cm))
        r = np.full(nd, np.nan)
        r[:nd - h] = (x[h:] / e[:nd - h] - 1) * 100
        return r

    IE, I59, I1059 = idx("09:31"), idx("09:59"), idx("10:59")
    ARR = {"O1": ret(IE, I59, 0), "O2": ret(IE, I1059, 0), "O3": ret(IE, I1059, 1), "O4": ret(IE, I1059, 5)}
    ev26 = a26[a26.year == 2026].set_index("date")
    k_of = {str(d.date()): k for k, d in enumerate(days)}
    diffs = {}
    for o in ARR:
        dd = [abs(ARR[o][k_of[d]] - ev26.loc[d, o]) for d in ev26.index if np.isfinite(ev26.loc[d, o])]
        diffs[o] = f(max(dd))
    out["0_recompute"]["direct_vs_A_2026_R1_max_abs_diff"] = diffs
    # yfinance 날짜 정렬(하루 밀리면 미래 참조): 같은 날 Alpaca 정규장 종가와 비교, 하루 민 것과도 비교
    dd_ = D.daily.loc["SOXX"].copy()
    dd_.index = pd.to_datetime(dd_.index)
    out["0_recompute"]["yf_soxx_close_vs_alpaca_same_day_median_abs_rel_pct"] = r2(float((X["Close"].reindex(dd_.index) / dd_["close"] - 1).abs().median() * 100), 4)
    out["0_recompute"]["yf_soxx_close_shifted1_vs_alpaca_median_abs_rel_pct"] = r2(float((X["Close"].shift(1).reindex(dd_.index) / dd_["close"] - 1).abs().median() * 100), 4)
    log("0", out["0_recompute"])

    isR1D = R1[tD]

    def evstats(mask, o="O2", year=2026, arr=None, seed=SEED):
        a = ARR[o] if arr is None else arr
        sel = mask & (yrD == year) & np.isfinite(a) if year != "all" else mask & np.isfinite(a)
        v = a[sel]
        pool = a[((yrD == year) if year != "all" else np.ones(nd, bool)) & np.isfinite(a)]
        p, nmn = null_p(v, pool, seed)
        return {"n": int(len(v)), "mean": r2(v.mean()) if len(v) else None, "median": r2(np.median(v)) if len(v) else None,
                "hit": r2((v > 0).mean()) if len(v) else None, "p_random": p, "null_mean": r2(nmn)}

    # ------------------------------------------------ 1. 미래 참조·위약
    R_same = first_red(zn, 0)          # 위약: D 종가 단계를 D 장초에 안다고 가정
    R_late = first_red(zn, 2)          # 하루 늦게(빨강 둘째 날)
    zl = live_open_stage(oX, cX)
    finp = np.r_[False, np.isfinite(zn[:-1])]
    prev_not_red = np.r_[False, np.isfinite(zn[:-1]) & (zn[:-1] > -1)]
    R_live = (np.isfinite(zl) & (zl <= -1)) & prev_not_red & finp
    timing = {}
    for nm_, ev in (("registered_stage_D-1", R1), ("placebo_stage_D_sameday_close(lookahead)", R_same),
                    ("late_stage_D-2(red_day2)", R_late), ("live_open_stage_D(repainting)", R_live)):
        m = ev[tD]
        timing[nm_] = {str(y): {o: evstats(m, o, y) for o in ("O1", "O2", "O3")} for y in (2023, 2024, 2025, 2026)}
    # 장중 다시 그리는 지표의 첫 빨강과 등록 사건이 겹치는 정도
    timing["overlap_live_vs_registered_2026"] = {"live_n": int((R_live & (yrX == 2026)).sum()),
                                                 "registered_n": int((R1 & (yrX == 2026)).sum()),
                                                 "both_same_D": int((R_live & R1 & (yrX == 2026)).sum()),
                                                 "registered_already_live_red_on_D-1": int((R1 & np.r_[False, R_live[:-1]] & (yrX == 2026)).sum())}
    # 일봉 근사 위약(2011~2025, SOXS 시가→종가)
    jS = {d: j for j, d in enumerate(pd.DatetimeIndex(S.index))}
    so, sc = S["Open"].to_numpy(float), S["Close"].to_numpy(float)
    oc_S = (sc / so - 1) * 100
    tS = np.array([jS.get(d, -1) for d in dX])

    def daily_vals(ev, y0=2011, y1=2025, arr=oc_S, extra=None):
        m = ev & (tS >= 0) & (yrX >= y0) & (yrX <= y1)
        if extra is not None:
            m &= extra
        v = arr[tS[m]]
        v = v[np.isfinite(v)]
        return {"n": int(len(v)), "mean": r2(v.mean()) if len(v) else None}

    timing["daily_oc_2011_2025"] = {"registered": daily_vals(R1), "placebo_sameday": daily_vals(R_same),
                                    "late": daily_vals(R_late), "live_open": daily_vals(R_live)}
    # 진입 시각 감도(빨강은 D−1 16:00 ET부터 보인다): D−1 정규장 마지막 봉 종가, D 08:00·09:00 ET(프리장), 등록 09:31
    rth_last = np.array([C[k, e - 1] for k, e in enumerate(D.rth_end)])
    ent_alt = {}
    for lab, px_fn, i_cost in (("D-1_16:00_close", lambda: np.r_[np.nan, rth_last[:-1]], idx("15:59")),
                               ("D_08:00ET_pre", lambda: C[:, idx("07:59")], idx("07:59")),
                               ("D_09:00ET_pre", lambda: C[:, idx("08:59")], idx("08:59")),
                               ("D_09:31_registered", lambda: C[:, IE], IE),
                               ("D_09:32", lambda: C[:, idx("09:32")], idx("09:32")),
                               ("D_09:35", lambda: C[:, idx("09:35")], idx("09:35")),
                               ("D_09:31_bar_open", lambda: O[:, IE], IE),
                               ("D_09:31_bar_high(worst)", lambda: H[:, IE], IE)):
        e = px_fn() * (1 + hs(i_cost))
        r = (C[:, I1059] * (1 - hs(I1059)) / e - 1) * 100
        ent_alt[lab] = {str(y): evstats(isR1D, "O2", y, arr=r) for y in (2023, 2024, 2025, 2026)}
    # 청산 감도
    ex_alt = {}
    for lab, i_x in (("10:30", idx("10:29")), ("10:55", idx("10:54")), ("10:59_registered", I1059), ("11:05", idx("11:04")),
                     ("11:30", idx("11:29"))):
        ex_alt[lab] = {str(y): evstats(isR1D, "O2", y, arr=ret(IE, i_x, 0)) for y in (2025, 2026)}
    worst = ret(IE, I1059, 0, ent_px=H[:, IE], ex_px=L[:, I1059])
    ex_alt["worst_09:31high_10:59low"] = {str(y): evstats(isR1D, "O2", y, arr=worst) for y in (2025, 2026)}
    timing["entry_time_O2_exit"] = ent_alt
    timing["exit_time"] = ex_alt
    out["1_lookahead_placebo_timing"] = timing
    log("1 끝")

    # ------------------------------------------------ 2. 사건 독립성
    t26 = np.flatnonzero(R1 & (yrX == 2026))
    red = np.isfinite(zn) & (zn <= -1)
    rows = []
    for q, t in enumerate(t26):
        run = 0
        while t - 1 + run < len(zn) and red[t - 1 + run]:
            run += 1
        k = int(np.searchsorted(tD, t))
        prev_red_any = bool(red[max(0, t - 6):t - 1].any())
        rows.append({"date": str(dX[t].date()), "gap_td_from_prev_event": int(t - t26[q - 1]) if q else None,
                     "red_run_len_from_D-1": run, "red_in_D-6..D-2": prev_red_any,
                     "soxx_D-1_ret": r2((cX[t - 1] / cX[t - 2] - 1) * 100), "soxx_D-1_rsi": r2(rsi[t - 1], 1),
                     "soxx_D-1_z": r2(z[t - 1]), "above_sma200_D-1": None,
                     **{o: r2(ARR[o][k]) for o in ("O1", "O2", "O3")}})
    sma200 = pd.Series(cX).rolling(200).mean().to_numpy()
    above = cX > sma200
    for r_, t in zip(rows, t26):
        r_["above_sma200_D-1"] = bool(above[t - 1])
    # 묶음: 앞 사건과 5거래일 이하면 같은 묶음
    ep, cur = [], [0]
    for q in range(1, len(t26)):
        if t26[q] - t26[q - 1] <= 5:
            cur.append(q)
        else:
            ep.append(cur)
            cur = [q]
    ep.append(cur)
    v2 = np.array([rows[q]["O2"] for q in range(len(rows))], float)
    epm = np.array([v2[g].mean() for g in ep])
    rng = np.random.default_rng(SEED)
    bs = epm[rng.integers(0, len(epm), (NR, len(epm)))].mean(1)
    first_in_ep = np.array([v2[g[0]] for g in ep])
    later_in_ep = np.array([v2[q] for g in ep for q in g[1:]])
    out["2_independence"] = {
        "events_2026": rows,
        "episodes_gap<=5td": [[rows[q]["date"] for q in g] for g in ep],
        "n_episodes": len(ep), "episode_mean_O2": r2(epm.mean()), "episode_boot95": [r2(np.percentile(bs, 2.5)), r2(np.percentile(bs, 97.5))],
        "episode_boot_P(mean<=0)": float((bs <= 0).mean()),
        "first_event_of_episode_O2": {"n": len(first_in_ep), "mean": r2(first_in_ep.mean())},
        "later_events_in_episode_O2": {"n": len(later_in_ep), "mean": r2(later_in_ep.mean()) if len(later_in_ep) else None},
        "flicker_red_run_len1": int(sum(1 for r_ in rows if r_["red_run_len_from_D-1"] == 1)),
        "R3_vs_reRed_from_A": A["primary_2026"]["posthoc_description_only"]["O2"]}
    log("2 끝", out["2_independence"]["n_episodes"], out["2_independence"]["episode_mean_O2"])

    # ------------------------------------------------ 3. 몰림
    srt = np.sort(v2)[::-1]
    conc = {"O2_sum": r2(v2.sum()), "drop_top1": r2(srt[1:].mean()), "drop_top2": r2(srt[2:].mean()), "drop_top3": r2(srt[3:].mean()),
            "top3_share_of_sum": r2(srt[:3].sum() / v2.sum()),
            "by_month_sum_O2": pd.Series(v2, index=[r_["date"][:7] for r_ in rows]).groupby(level=0).sum().round(2).to_dict()}
    jul = np.array([r_["date"][:7] == "2026-07" for r_ in rows])
    conc["july_share_of_sum"] = r2(v2[jul].sum() / v2.sum())
    conc["leave_one_month_out_min_mean"] = r2(min(v2[np.array([r_["date"][:7] != m for r_ in rows])].mean()
                                                  for m in set(r_["date"][:7] for r_ in rows)))
    # 긴 재현: 해마다 SOXS oc·o1, 하락장 해(2018·2022) 빼기, 연도별
    o1_S = np.r_[(so[1:] / so[:-1] - 1) * 100, np.nan]
    by_year = {}
    for y in range(2011, 2026):
        by_year[str(y)] = {"oc": daily_vals(R1, y, y)["mean"], "o1": daily_vals(R1, y, y, arr=o1_S)["mean"], "n": daily_vals(R1, y, y)["n"]}
    ex_bear = (yrX != 2018) & (yrX != 2022)
    conc["long_2011_2025"] = {"oc_all": daily_vals(R1), "oc_ex_2018_2022": daily_vals(R1, extra=ex_bear),
                              "o1_all": daily_vals(R1, arr=o1_S), "o1_ex_2018_2022": daily_vals(R1, arr=o1_S, extra=ex_bear),
                              "by_year": by_year}
    out["3_concentration"] = conc
    log("3 끝", conc["drop_top3"], conc["july_share_of_sum"])

    # ------------------------------------------------ 4. 장세(SOXX D−1 200일선)
    abvD = np.r_[False, above[:-1]][tD]
    reg = {}
    for y in (2023, 2024, 2025, 2026):
        reg[str(y)] = {"above": evstats(isR1D & abvD, "O2", y), "below": evstats(isR1D & ~abvD, "O2", y),
                       "all_days_above_O2": r2(np.nanmean(ARR["O2"][(yrD == y) & abvD])),
                       "all_days_below_O2": r2(np.nanmean(ARR["O2"][(yrD == y) & ~abvD])) if ((yrD == y) & ~abvD).any() else None,
                       "share_days_above": r2(abvD[yrD == y].mean())}
    abv_prev = np.r_[False, above[:-1]]
    reg["long_2011_2025_oc"] = {"above": daily_vals(R1, extra=abv_prev), "below": daily_vals(R1, extra=~abv_prev),
                                "all_days_above": daily_vals(np.ones(len(dX), bool), extra=abv_prev),
                                "all_days_below": daily_vals(np.ones(len(dX), bool), extra=~abv_prev)}
    reg["long_2011_2025_o1"] = {"above": daily_vals(R1, arr=o1_S, extra=abv_prev), "below": daily_vals(R1, arr=o1_S, extra=~abv_prev)}
    # 해 층화 무작위 p(같은 장세 날 안에서 뽑음)
    rngr = np.random.default_rng(SEED)
    for nm_, arr_ in (("oc", oc_S), ("o1", o1_S)):
        for lab, cond in (("above", abv_prev), ("below", ~abv_prev)):
            vals, ns = [], np.zeros(NR)
            for y in range(2011, 2026):
                m = R1 & cond & (yrX == y) & (tS >= 0)
                v = arr_[tS[m]]
                v = v[np.isfinite(v)]
                pm = cond & (yrX == y) & (tS >= 0)
                pool = arr_[tS[pm]]
                pool = pool[np.isfinite(pool)]
                if not len(v):
                    continue
                keys = rngr.random((NR, len(pool)))
                ns += pool[np.argpartition(keys, len(v) - 1, axis=1)[:, :len(v)]].sum(1)
                vals += list(v)
            va = np.array(vals)
            reg[f"long_{nm_}_{lab}_p_random_year_stratified"] = {"n": len(va), "mean": r2(va.mean()), "null_mean": r2((ns / len(va)).mean()),
                                                                 "p": float((ns / len(va) >= va.mean() - 1e-12).mean())}
    out["4_regime_sma200_posthoc"] = reg
    log("4 끝")

    # ------------------------------------------------ 5. 첫날 대 나머지·전날 하락 대조
    rprev = np.r_[np.nan, np.nan, (cX[1:-1] / cX[:-2] - 1) * 100]      # t에서 D−1 종가 대비 D−2 종가(SOXX)
    rpD = rprev[tD]
    ctrl = {}
    for y in (2023, 2024, 2025, 2026):
        thr = float(np.median(rpD[isR1D & (yrD == y)]))
        dn = (rpD <= thr)
        ctrl[str(y)] = {"R1_median_soxx_D-1_ret": r2(thr), "R1": evstats(isR1D, "O2", y),
                        "nonR1_with_D-1_ret<=thr": evstats(dn & ~isR1D, "O2", y),
                        "nonR1_with_D-1_ret<=thr_O1": evstats(dn & ~isR1D, "O1", y),
                        "nonR1_with_D-1_ret<=thr_O3": evstats(dn & ~isR1D, "O3", y),
                        "all_D-1_ret<=-2%": evstats(rpD <= -2, "O2", y)}
    thrL = float(np.nanmedian(rprev[R1 & (yrX >= 2011) & (yrX <= 2025)]))
    ctrl["long_2011_2025"] = {"R1_median_soxx_D-1_ret": r2(thrL), "R1_oc": daily_vals(R1), "nonR1_down_oc": daily_vals(~R1 & (rprev <= thrL)),
                              "R1_o1": daily_vals(R1, arr=o1_S), "nonR1_down_o1": daily_vals(~R1 & (rprev <= thrL), arr=o1_S)}
    # 범주 분할 점검(첫날 + 나머지 빨강 + 빨강 아님 = 단계 값 있는 날 전부)
    fin = np.isfinite(zn)
    first = R1
    rest = np.r_[False, False, red[1:-1] & red[:-2]]
    nonred = np.r_[False, fin[:-1] & ~red[:-1]]
    valid = np.r_[False, False, fin[1:-1] & fin[:-2]]
    ctrl["partition_ok_since_2002"] = bool(((first.astype(int) + rest + nonred)[valid & (yrX >= 2002)] == 1).all())
    # 첫날 O3·O4 '나머지'는 보유 창이 서로 겹친다: 나머지 빨강 날의 O3 값 중 인접 날과 창을 공유하는 비율
    out["5_first_vs_rest_and_downday_control"] = ctrl
    log("5 끝", ctrl["2026"]["nonR1_with_D-1_ret<=thr"], ctrl["long_2011_2025"])

    # ------------------------------------------------ 6. SOXS 일봉 시가 질
    dS = pd.DatetimeIndex(S.index)
    yrS = np.array([d.year for d in dS])
    sh, sl = S["High"].to_numpy(float), S["Low"].to_numpy(float)
    stale = np.r_[False, np.abs(so[1:] / sc[:-1] - 1) < 1e-6]
    outside = (so > sh * (1 + 1e-9)) | (so < sl * (1 - 1e-9))
    xo_on_S = X["Open"].reindex(dS).to_numpy(float)
    xc_on_S = X["Close"].reindex(dS).to_numpy(float)
    resid = oc_S - (-3 * (xc_on_S / xo_on_S - 1) * 100)
    evS = np.zeros(len(dS), bool)
    for t in np.flatnonzero(R1):
        if tS[t] >= 0:
            evS[tS[t]] = True
    q6 = {}
    for y in range(2011, 2027):
        m = yrS == y
        q6[str(y)] = {"stale_open_share": r2(stale[m].mean(), 4), "open_outside_HL": int(outside[m].sum()),
                      "abs_resid_median_pctp": r2(np.nanmedian(np.abs(resid[m])), 3),
                      "abs_resid_p95": r2(np.nanpercentile(np.abs(resid[m]), 95), 3),
                      "R1_days_abs_resid_median": r2(np.nanmedian(np.abs(resid[m & evS])), 3) if (m & evS).any() else None,
                      "R1_days_abs_resid_max": r2(np.nanmax(np.abs(resid[m & evS])), 3) if (m & evS).any() else None}
    # 큰 차이가 난 사건 날(시가 의심) 빼고 다시
    bad = evS & (np.abs(resid) > 1.5)
    good_ev = np.array([tS[t] >= 0 and not bad[tS[t]] for t in range(len(dX))])
    q6["R1_2011_2025_oc_excluding_resid>1.5pp"] = daily_vals(R1 & good_ev)
    q6["R1_2011_2025_n_resid>1.5pp"] = int(sum(1 for t in np.flatnonzero(R1 & (yrX >= 2011) & (yrX <= 2025)) if tS[t] >= 0 and bad[tS[t]]))
    Lx = ld("SOXL", "20100101")
    det = []
    for t in np.flatnonzero(R1 & (yrX >= 2011) & (yrX <= 2025)):
        if tS[t] >= 0 and bad[tS[t]]:
            dte = dX[t]
            det.append({"date": str(dte.date()), "soxs_oc": r2(oc_S[tS[t]]), "minus3_soxx_oc": r2(-3 * (cX[t] / oX[t] - 1) * 100),
                        "minus_soxl_oc": r2(-(Lx["Close"][dte] / Lx["Open"][dte] - 1) * 100)})
    q6["R1_resid>1.5pp_days"] = det
    out["6_soxs_daily_open_quality"] = q6
    log("6 끝")

    # ------------------------------------------------ 7. 비용
    rt_cost = (cost["open5"] + cost["late"]) * 100
    k26 = np.flatnonzero(isR1D & (yrD == 2026))
    px = C[k26, IE]
    tick_hs = 0.005 / px * 100
    q = pd.read_parquet(os.path.join(CACHE, "quotes.parquet")).dropna()
    q = q[(q.ask > q.bid) & (q.bid > 0) & (q.sym == "SOXS")]
    q["hs"] = (q.ask - q.bid) / 2 / ((q.ask + q.bid) / 2) * 100
    tt = pd.to_datetime(q.t)
    q["day"] = tt.dt.tz_localize(None).dt.normalize() if tt.dt.tz is None else tt.dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    mm = tt.dt.tz_convert(ET).dt.hour * 60 + tt.dt.tz_convert(ET).dt.minute
    q31 = q[(mm >= 571) & (mm <= 575)]
    qdays = sorted(set(q.day.dt.strftime("%Y-%m-%d")))
    r1all = set(EVA[EVA.variant == "R1"].date)
    rng_ = (H[:, IE] - L[:, IE]) / C[:, IE] * 100
    c2 = ret(IE, I1059, 0, cm=2.0)
    c5 = ret(IE, I1059, 0, cm=5.0)
    out["7_cost"] = {
        "round_trip_cost_pct": r2(rt_cost, 4), "breakeven_cost_multiple_O2_2026": r2((v2.mean() + rt_cost) / rt_cost, 1),
        "soxs_0931_price_2026_events": {"min": r2(px.min()), "median": r2(np.median(px)), "max": r2(px.max())},
        "one_tick_half_spread_pct_2026_events_max": r2(tick_hs.max(), 4), "table_half_spread_open5_pct": r2(cost["open5"] * 100, 4),
        "quote_sample_days": len(qdays), "quote_days_that_are_R1": sorted(set(qdays) & r1all),
        "quote_0931_0935_half_spread_pct": {"median": r2(q31.hs.median(), 4), "p90": r2(q31.hs.quantile(0.9), 4), "max": r2(q31.hs.max(), 4)},
        "bar_0931_range_pct": {"R1_2026_median": r2(np.median(rng_[k26])), "all_2026_median": r2(np.nanmedian(rng_[yrD == 2026]))},
        "O2_2026_cost1x_2x_5x": [evstats(isR1D, "O2", 2026)["mean"], evstats(isR1D, "O2", 2026, arr=c2)["mean"], evstats(isR1D, "O2", 2026, arr=c5)["mean"]],
        "O2_2026_worst_fill_09:31high_10:59low": evstats(isR1D, "O2", 2026, arr=worst),
        "quote_R1_days_half_spread_pct": {d_: {"open_0931_0935_median": r2(q31[q31.day.dt.strftime("%Y-%m-%d") == d_].hs.median(), 4),
                                               "late_1055_1100_median": r2(q[(q.day.dt.strftime("%Y-%m-%d") == d_) & (mm >= 655) & (mm <= 660)].hs.median(), 4)}
                                          for d_ in sorted(set(qdays) & r1all)}}
    log("7 끝", out["7_cost"]["breakeven_cost_multiple_O2_2026"])

    # ------------------------------------------------ 8. 사람 실행
    E_act = Engine(D, scheme="actual")
    dA, trA = E_act.run(lambda: S5.Always(-1), days=list(range(nd)))
    act = dA["ret"].to_numpy() * 100
    cut = np.array([cutoff_idx(d.date(), "actual") for d in days])
    winter = cut == idx("10:00")
    ex_rows = []
    for k in k26:
        d = days[k]
        e_et = datetime(d.year, d.month, d.day, 9, 31, tzinfo=ET)
        x_et = datetime(d.year, d.month, d.day, 11, 0, tzinfo=ET)
        ex_rows.append({"date": str(d.date()), "EST(winter)": bool(winter[k]), "entry_KST": e_et.astimezone(KST).strftime("%H:%M"),
                        "O2_exit_KST": x_et.astimezone(KST).strftime("%H:%M"), "O2": r2(ARR["O2"][k]), "actual_cutoff_ret": r2(act[k])})
    # O3 actual: D+1 청산을 그날 마감 1분 전 봉으로
    x_act = np.array([C[kk, cut[kk] - 1] * (1 - hs(cut[kk] - 1)) for kk in range(nd)])
    o3_act = np.full(nd, np.nan)
    o3_act[:-1] = (x_act[1:] / (C[:-1, IE] * (1 + hs(IE))) - 1) * 100
    by_year_O1 = {str(y): evstats(isR1D, "O1", y) for y in (2023, 2024, 2025, 2026)}
    out["8_human_execution"] = {
        "events": ex_rows,
        "actual_24KST_2026": evstats(isR1D, "O2", 2026, arr=act),
        "actual_24KST_by_year": {str(y): evstats(isR1D, "O2", y, arr=act) for y in (2023, 2024, 2025)},
        "O3_actual_24KST_2026": evstats(isR1D, "O3", 2026, arr=o3_act),
        "winter_feasible_only_O1_by_year": by_year_O1,
        "winter_R1_events_all_years_O1_vs_O2": {"O1": evstats(isR1D & winter, "O1", "all"), "O2_infeasible": evstats(isR1D & winter, "O2", "all")},
        "summer_R1_events_all_years_O2": evstats(isR1D & ~winter, "O2", "all"),
        "note": "주문 가능 08:00~24:00 KST(desk §1). 표준시(11월 첫 일요일~3월 둘째 일요일)에는 11:00 ET = 01:00 KST라 O2·O3·O4 청산을 직접 못 낸다."}
    log("8 끝", out["8_human_execution"]["actual_24KST_2026"])

    # ------------------------------------------------ 9. 통계 재점검
    pool26 = ARR["O2"][(yrD == 2026) & np.isfinite(ARR["O2"])]
    hit0 = float((pool26 > 0).mean())
    kpos = int((v2 > 0).sum())
    pooled = {}
    vals_all, null_sum, n_all = [], np.zeros(NR), 0
    rngp = np.random.default_rng(SEED)
    for y in (2023, 2024, 2025, 2026):
        a = ARR["O2"]
        v = a[isR1D & (yrD == y) & np.isfinite(a)]
        pool = a[(yrD == y) & np.isfinite(a)]
        keys = rngp.random((NR, len(pool)))
        pick = np.argpartition(keys, len(v) - 1, axis=1)[:, :len(v)]
        null_sum += pool[pick].sum(1)
        n_all += len(v)
        vals_all += list(v)
    va = np.array(vals_all)
    pooled = {"n": n_all, "mean": r2(va.mean()), "p_random_year_stratified": float((null_sum / n_all >= va.mean() - 1e-12).mean())}
    out["9_stats"] = {
        "O2_2026_t_test_p_one_sided": r2(stats.ttest_1samp(v2, 0, alternative="greater").pvalue, 4),
        "O2_2026_wilcoxon_p_one_sided": r2(stats.wilcoxon(v2, alternative="greater").pvalue, 4),
        "O2_2026_sign": {"pos": kpos, "n": len(v2), "null_hit": r2(hit0, 3),
                         "binom_p_one_sided": r2(stats.binomtest(kpos, len(v2), hit0, alternative="greater").pvalue, 4)},
        "O2_pooled_2023_2026": pooled,
        "years_positive_O2_1m": [y for y in ("2023", "2024", "2025", "2026") if A["years_1m"][y]["O2"]["mean"] > 0]}
    log("9 끝", out["9_stats"])
    json.dump(out, open(os.path.join(RES, "audit6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    log("끝")


if __name__ == "__main__":
    main()
