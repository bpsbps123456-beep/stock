"""BRIEF4 C3 — 이론 가설 10개(사용자 방향 x = 3%, r_w90, 2026) 판정.

정본: BRIEF4.md 'C3 이론 가설 10개'. 사건 표 results/ev4_events.parquet(ev4_build.py)를 거르기만 한다(새 체결 없음).

결과 보기 전에 정한 것(이 파일 첫 실행 전 고정):
- 표본: side == 'user', x == 0.03, 2026-01-02~10-28 거래일 207일(Data('alp1m_long').days).
- 주 통계량: 하루 손익 평균(그날 매매 손익의 곱 − 1, 사건 없는 날 0) — H0 판정(BRIEF3)과 같은 '하루 평균'.
- 날 묶음 부트스트랩: 그 해 거래일을 복원 추출 10,000회(seed 20261102, 날 집합마다 새 rng). 95% 구간 = 2.5·97.5 분위,
  단측 p = 부트스트랩 평균 ≤ 0 비율(lib.summarize·confirm3.boot_ci와 같은 방식).
- 판정: 주 통계량의 단측 p에 Holm(m = 10) 보정, 보정 p ≤ 0.05면 통과. 매매당 평균의 날 묶음 p는 함께 적고 판정에 쓰지 않는다.
- 조건(결정 봉 마감 때 아는 값만, 수치 비교는 ev4 수준 규칙과 같은 1e-9 여유, 결측이면 불만족):
  H1 c01_time ∈ [20, 40](신호 봉 마감 09:50~10:10, 조건 1과 같은 '마감 시각' 기준)
  H2 c17_qqq_lv '<0.5%'      H3 c18_nvda_lv '<x/3'(x = 3 → 1%)     H4 c10_wick ≥ 0.5
  H5 c09_rsidiv_lv '예'       H6 c12_risevol_lv '<1'                H7 c15_vs_prevhigh_lv '±0.5% 안'
  H8 c05_gap ≤ −2 그리고 |c04_vs_prevclose| ≤ 0.5                    H9 c22_preveff_lv '≥0.6'
  H10 c07_vwap_lv '≥3%'
  거울 면(설명용)은 같은 열을 그대로 쓴다(ev4 조건 열이 이미 오른 종목 R 기준이라, 거울에서는 SOXS 기준으로 같은 뜻).
- 추가(판정 밖, 표만): 두 반(1~6월 123일·7~10월 84일), 가장 좋은 달(하루 손익 합 최대) 빼기, 비용 2배(r_w90_c2),
  −2% 손절(r_w90_s2), std(r_std, 결정 ≤ 09:44 사건만), 2023·2024·2025 같은 조건, 같은 사건의 추종(r_follow)·방향 성분
  (r_w90 − r_follow)/2, 거울 면 같은 조건, H2·H3의 09:30부터 창 판(alt_*), 조건 불만족 사건과의 매매당 차이.
- 참고: 10개 가설 주 통계량의 중심화 부트스트랩 최대 t(단측) 95% 분위.

한계: 사건 표는 조건 없는 사슬(한 번에 하나, 청산 뒤 L 다시 잼)의 사건이다. '조건이 맞을 때만 산다'를 단독 실행하면
건너뛴 사건 동안 L이 다시 잡히지 않아 뒤 신호가 달라질 수 있다. 통과 가설은 BRIEF4 '통과 뒤 검증'에서 따로 잰다.
"""
import hashlib, json, os, sys, time

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF4: 2023~2026 전체를 쓴다
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
SEED, NB, EPS, ALPHA, M = 20261102, 10000, 1e-9, 0.05, 10
X = 0.03
HALVES = {"H1_jan_jun": ("2026-01-01", "2026-06-30"), "H2_jul_oct": ("2026-07-01", "2026-10-31")}
T0 = time.time()

HYP = [
    ("H1", "10시 되돌림", "신호 봉 마감 09:50~10:10 (c01_time 20~40)",
     lambda e: e["c01_time"].between(20 - EPS, 40 + EPS)),
    ("H2", "반도체만 오름", "QQQ 같은 창 저점 대비 < 0.5% (c17_qqq_lv '<0.5%')",
     lambda e: e["c17_qqq_lv"] == "<0.5%"),
    ("H3", "대장 미확인", "NVDA 같은 창 저점 대비 < 1% (c18_nvda_lv '<x/3', x = 3)",
     lambda e: e["c18_nvda_lv"] == "<x/3"),
    ("H4", "윗꼬리 거부", "신호 봉 윗꼬리 ÷ 봉 길이 ≥ 0.5 (c10_wick)",
     lambda e: e["c10_wick"] >= 0.5 - EPS),
    ("H5", "RSI 다이버전스", "조건 9 = 예 (c09_rsidiv_lv)",
     lambda e: e["c09_rsidiv_lv"] == "예"),
    ("H6", "빈 거래량 오름", "오름 구간 평균 거래량 ÷ L 앞 20봉 < 1 (c12_risevol_lv '<1')",
     lambda e: e["c12_risevol_lv"] == "<1"),
    ("H7", "전날 고가 저항", "전날 정규장 고가 대비 ±0.5% 안 (c15_vs_prevhigh_lv)",
     lambda e: e["c15_vs_prevhigh_lv"] == "±0.5% 안"),
    ("H8", "갭 메우기 끝", "갭 ≤ −2% (c05_gap) 그리고 신호 봉 종가가 전날 종가 ±0.5% 안 (|c04_vs_prevclose| ≤ 0.5)",
     lambda e: (e["c05_gap"] <= -2 + EPS) & (e["c04_vs_prevclose"].abs() <= 0.5 + EPS)),
    ("H9", "추세일 다음 날", "전날 장초 90분 방향 효율 ≥ 0.6 (c22_preveff_lv '≥0.6')",
     lambda e: e["c22_preveff_lv"] == "≥0.6"),
    ("H10", "VWAP 과열", "VWAP 위 거리 ≥ 3% (c07_vwap_lv '≥3%')",
     lambda e: e["c07_vwap_lv"] == "≥3%"),
]
ALT = {"H2": ("H2_alt0930", "QQQ 09:30부터 저점 대비 < 0.5% (alt_c17_qqq_0930)",
              lambda e: e["alt_c17_qqq_0930"] < 0.5 - EPS),
       "H3": ("H3_alt0930", "NVDA 09:30부터 저점 대비 < 1% (alt_c18_nvda_0930)",
              lambda e: e["alt_c18_nvda_0930"] < 1.0 - EPS)}

_IDX = {}


def boot_idx(n):
    """날 수 n마다 같은 seed로 한 번 뽑아 같은 날 집합의 모든 통계가 같은 복원 추출을 쓴다."""
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(SEED).integers(0, n, (NB, n))
    return _IDX[n]


def fnum(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def day_arrays(sub, days, col):
    """days(정렬된 Timestamp 목록)마다 하루 손익(곱 − 1, %), 손익 합, 건수."""
    s = sub[["day", col]].dropna()
    g = s.groupby("day")[col]
    comp = g.apply(lambda r: (np.prod(1 + r.values / 100) - 1) * 100) if len(s) else pd.Series(dtype=float)
    dly = comp.reindex(days, fill_value=0.0).values.astype(float)
    S = g.sum().reindex(days, fill_value=0.0).values.astype(float) if len(s) else np.zeros(len(days))
    N = g.size().reindex(days, fill_value=0).values.astype(float) if len(s) else np.zeros(len(days))
    return dly, S, N


def stats(sub, days, col="r_w90", boot=True):
    dly, S, N = day_arrays(sub, days, col)
    n = int(N.sum())
    dset = set(days)
    assert n == int(sub.loc[sub["day"].isin(dset), col].notna().sum()), ("날 맞춤 실패", col, n)
    act = N > 0
    out = {"days": len(days), "trades": n, "active_days": int(act.sum()),
           "trade_mean": fnum(S.sum() / n) if n else None,
           "active_day_mean": fnum(dly[act].mean()) if act.any() else None,
           "daily_mean": fnum(dly.mean()) if len(days) else None,
           "sum_daily": fnum(dly.sum())}
    if boot and n and len(days) > 1:
        ix = boot_idx(len(days))
        bd = dly[ix].mean(1)
        out["daily_ci95"] = [fnum(np.percentile(bd, 2.5)), fnum(np.percentile(bd, 97.5))]
        out["daily_p_one_sided"] = fnum((bd <= 0).mean())
        out["daily_boot_sd"] = fnum(bd.std(ddof=1))
        sn, nn = S[ix].sum(1), N[ix].sum(1)
        ok = nn > 0
        bt = np.where(ok, sn / np.where(ok, nn, 1), np.nan)
        bt = bt[ok]
        out["trade_ci95"] = [fnum(np.percentile(bt, 2.5)), fnum(np.percentile(bt, 97.5))]
        out["trade_p_one_sided"] = fnum((bt <= 0).mean())
        out["boot_draws_with_trades"] = int(ok.sum())
    return out


def trade_desc(sub):
    if not len(sub):
        return {}
    miss = sub[sub["hit"] == 0]
    return {"hit_rate": fnum(sub["hit"].mean()), "miss_n": int(len(miss)),
            "miss_mean": fnum(miss["r_w90"].mean()) if len(miss) else None,
            "worst_trade": fnum(sub["r_w90"].min()), "best_trade": fnum(sub["r_w90"].max()),
            "mae_median": fnum(sub["mae"].median()), "mae_p10": fnum(sub["mae"].quantile(0.10)),
            "min_to_target_median": fnum(sub.loc[sub["hit"] == 1, "min_to_target"].median()) if sub["hit"].any() else None,
            "by_month_trades": {k: int(v) for k, v in sub.groupby("month").size().items()}}


def by_month(sub, days, col="r_w90"):
    dly, _, N = day_arrays(sub, days, col)
    m = pd.Series(dly, index=days).groupby(lambda t: f"{t.year}-{t.month:02d}")
    n = pd.Series(N, index=days).groupby(lambda t: f"{t.year}-{t.month:02d}")
    return {k: {"sum_daily": fnum(v.sum()), "daily_mean": fnum(v.mean()), "days": int(v.size), "trades": int(n.get_group(k).sum())}
            for k, v in m}


def drop_best_month(sub, days, col="r_w90"):
    bm = by_month(sub, days, col)
    best = max(bm, key=lambda k: bm[k]["sum_daily"])
    keep = [d for d in days if f"{d.year}-{d.month:02d}" != best]
    s2 = sub[sub["month"] != best]
    st = stats(s2, keep, col, boot=True)
    return {"best_month": best, "best_month_sum_daily": bm[best]["sum_daily"],
            "share_of_total_sum": fnum(bm[best]["sum_daily"] / sum(v["sum_daily"] for v in bm.values()))
            if sum(v["sum_daily"] for v in bm.values()) != 0 else None,
            "without": {k: st.get(k) for k in ["days", "trades", "trade_mean", "daily_mean", "daily_ci95", "daily_p_one_sided"]}}


def holm(ps):
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


def dir_component(sub, days):
    s = sub.copy()
    s["r_dir"] = (s["r_w90"] - s["r_follow"]) / 2
    st = stats(s, days, "r_dir")
    return {k: st.get(k) for k in ["trades", "trade_mean", "trade_ci95", "trade_p_one_sided", "daily_mean", "daily_ci95"]}


def diff_vs_rest(sub, rest, days):
    """조건 만족 − 불만족 매매당 평균 차(같은 날 복원 추출)."""
    _, S1, N1 = day_arrays(sub, days, "r_w90")
    _, S0, N0 = day_arrays(rest, days, "r_w90")
    if N1.sum() == 0 or N0.sum() == 0:
        return None
    ix = boot_idx(len(days))
    a, b = S1[ix].sum(1), N1[ix].sum(1)
    c, d = S0[ix].sum(1), N0[ix].sum(1)
    ok = (b > 0) & (d > 0)
    diff = a[ok] / b[ok] - c[ok] / d[ok]
    return {"in_trade_mean": fnum(S1.sum() / N1.sum()), "rest_trade_mean": fnum(S0.sum() / N0.sum()), "rest_trades": int(N0.sum()),
            "diff": fnum(S1.sum() / N1.sum() - S0.sum() / N0.sum()),
            "diff_ci95": [fnum(np.percentile(diff, 2.5)), fnum(np.percentile(diff, 97.5))],
            "diff_p_one_sided": fnum((diff <= 0).mean())}


NEIGH = {   # 사후 이웃 문턱(결과를 본 뒤, 통과 가설에만 계산, 판정 밖)
    "H1": {f"c01 {a}~{b}": (lambda a, b: lambda e: e["c01_time"].between(a - EPS, b + EPS))(a, b)
           for a, b in [(15, 35), (25, 45), (20, 30), (30, 40)]},
    "H2": {f"c17 < {t}": (lambda t: lambda e: e["c17_qqq"] < t - EPS)(t) for t in (0.3, 0.7)},
    "H3": {f"c18 < {t}": (lambda t: lambda e: e["c18_nvda"] < t - EPS)(t) for t in (0.75, 1.25)},
    "H4": {f"c10 ≥ {t}": (lambda t: lambda e: e["c10_wick"] >= t - EPS)(t) for t in (0.3, 0.4, 0.45, 0.55, 0.6, 0.7)},
    "H5": {},
    "H6": {f"c12 < {t}": (lambda t: lambda e: e["c12_risevol"] < t - EPS)(t) for t in (0.8, 1.2)},
    "H7": {f"|c15| ≤ {t}": (lambda t: lambda e: e["c15_vs_prevhigh"].abs() <= t + EPS)(t) for t in (0.3, 1.0)},
    "H8": {f"c05 ≤ {g}, |c04| ≤ {b}": (lambda g, b: lambda e: (e["c05_gap"] <= g + EPS) & (e["c04_vs_prevclose"].abs() <= b + EPS))(g, b)
           for g, b in [(-1.5, 0.5), (-3, 0.5), (-2, 1.0)]},
    "H9": {f"c22 ≥ {t}": (lambda t: lambda e: e["c22_preveff"] >= t - EPS)(t) for t in (0.5, 0.7)},
    "H10": {f"c07 ≥ {t}": (lambda t: lambda e: e["c07_vwap"] >= t - EPS)(t) for t in (2.5, 3.5)},
}


def robustness(hid, fn, u26, base, ev, d26, yd):
    """통과 가설의 사후 점검(판정 밖): 작은 표본에서 날 부트스트랩이 못 보는 손실 꼬리를 다른 귀무로 다시 잰다."""
    mk = fn(u26).fillna(False).astype(bool).values
    sub = u26[mk]
    n, obs = len(sub), float(sub["r_w90"].mean())
    pool = u26["r_w90"].values
    rng = np.random.default_rng(SEED)
    NN = 100000
    # (a) 같은 해 사용자 방향 전체 사건 258건을 평균 0으로 옮긴 뒤 n건 복원 추출: 평균 ≥ 관측일 확률
    shifted = pool - pool.mean()
    pa = float((shifted[rng.integers(0, len(pool), (NN, n))].mean(1) >= obs - 1e-12).mean())
    # (b) 258건에서 n건 비복원 무작위 고르기(조건 없는 사건 대비): 평균 ≥ 관측일 확률
    pb = float((np.array([pool[rng.permutation(len(pool))[:n]].mean() for _ in range(20000)]) >= obs - 1e-12).mean())
    # (c) 도달 수 이항: 손익분기 도달률(전체 사건의 도달 +1%·못 닿은 평균 기준)과 전체 도달률
    from scipy.stats import binom
    hit_all = float(u26["hit"].mean())
    miss_mean = float(u26.loc[u26["hit"] == 0, "r_w90"].mean())
    hit_ret = float(u26.loc[u26["hit"] == 1, "r_w90"].mean())
    p_be = -miss_mean / (hit_ret - miss_mean)
    k = int(sub["hit"].sum())
    pc = {"hits": k, "n": n, "breakeven_hit_rate": p_be, "p_ge_k_at_breakeven": float(binom.sf(k - 1, n, p_be)),
          "baseline_hit_rate": hit_all, "p_ge_k_at_baseline": float(binom.sf(k - 1, n, hit_all))}
    neigh = {}
    for lab, nf in NEIGH.get(hid, {}).items():
        s2 = u26[nf(u26).fillna(False).astype(bool).values]
        st = stats(s2, d26)
        neigh[lab] = {kk: st.get(kk) for kk in ["trades", "trade_mean", "daily_mean", "daily_ci95", "daily_p_one_sided", "trade_ci95"]}
        neigh[lab]["hit_rate"] = fnum(s2["hit"].mean()) if len(s2) else None
    other_x = {}
    for xx in (0.025, 0.035, 0.04):
        e = ev[(ev["side"] == "user") & np.isclose(ev["x"], xx) & (ev["year"] == 2026)]
        s2 = e[fn(e).fillna(False).astype(bool).values]
        st = stats(s2, d26)
        other_x[f"{xx * 100:g}%"] = {kk: st.get(kk) for kk in ["trades", "trade_mean", "daily_mean", "daily_ci95", "daily_p_one_sided"]}
        other_x[f"{xx * 100:g}%"]["hit_rate"] = fnum(s2["hit"].mean()) if len(s2) else None
    u_old = base["user"][base["user"]["year"].isin([2023, 2024, 2025])]
    s_old = u_old[fn(u_old).fillna(False).astype(bool).values]
    pooled = stats(s_old, yd[2023] + yd[2024] + yd[2025])
    cols = ["day", "dec_close", "entry", "exit", "c10_wick", "c01_time", "c07_vwap", "c12_risevol", "ratio",
            "r_w90", "hit", "min_to_target", "mae", "mfe", "r_follow", "r_std", "r_w90_c2"]
    tl = sub[cols].copy()
    tl["day"] = tl["day"].dt.strftime("%Y-%m-%d")
    tl_old = s_old[cols].copy()
    tl_old["day"] = tl_old["day"].dt.strftime("%Y-%m-%d")
    return {"note": "결과를 본 뒤 더한 점검이라 판정에 쓰지 않는다",
            "null_shifted_pool_p": pa, "null_random_subset_p": pb, "binomial": pc,
            "neighbors_2026": neigh, "other_x_2026": other_x,
            "pooled_2023_2025": {kk: pooled.get(kk) for kk in ["days", "trades", "trade_mean", "daily_mean", "daily_ci95", "daily_p_one_sided", "trade_ci95"]}
                                | {"hit_rate": fnum(s_old["hit"].mean()) if len(s_old) else None},
            "trades_2026": json.loads(tl.to_json(orient="records", force_ascii=False)),
            "trades_2023_2025": json.loads(tl_old.to_json(orient="records", force_ascii=False))}


def main():
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    D = Data("alp1m_long")
    alld = [pd.Timestamp(d) for d in D.days]
    print(f"자료 {D.days[0]}~{D.days[-1]} {len(D.days)}일 ({time.time() - T0:.0f}s)", flush=True)
    yd = {y: [d for d in alld if d.year == y] for y in (2023, 2024, 2025, 2026)}
    d26 = yd[2026]
    half_days = {h: [d for d in d26 if pd.Timestamp(a) <= d <= pd.Timestamp(b)] for h, (a, b) in HALVES.items()}
    assert len(d26) == 207 and len(half_days["H1_jan_jun"]) == 123 and len(half_days["H2_jul_oct"]) == 84, \
        (len(d26), {k: len(v) for k, v in half_days.items()})
    assert set(ev["day"]).issubset(set(alld))

    base = {s: ev[(ev["side"] == s) & np.isclose(ev["x"], X)] for s in ("user", "mirror")}
    u26 = base["user"][base["user"]["year"] == 2026]
    m26 = base["mirror"][base["mirror"]["year"] == 2026]
    assert len(u26) == 258, len(u26)

    baseline = {"user_all_2026": stats(u26, d26), "user_all_2026_desc": trade_desc(u26),
                "mirror_all_2026": stats(m26, d26)}
    for y in (2023, 2024, 2025):
        baseline[f"user_all_{y}"] = stats(base["user"][base["user"]["year"] == y], yd[y])

    rows, daily_cols = {}, {}
    for hid, name, rule, fn in HYP:
        mk = fn(u26).fillna(False).astype(bool).values
        sub, rest = u26[mk], u26[~mk]
        r = {"id": hid, "name": name, "rule": rule,
             "stats_2026": stats(sub, d26), "desc_2026": trade_desc(sub),
             "halves": {h: stats(sub[sub["day"].isin(set(dd))], dd) for h, dd in half_days.items()},
             "drop_best_month": drop_best_month(sub, d26) if len(sub) else None,
             "by_month_2026": by_month(sub, d26),
             "cost2x": stats(sub, d26, "r_w90_c2"),
             "stop2": stats(sub, d26, "r_w90_s2"),
             "std_0959": stats(sub, d26, "r_std"),
             "follow": stats(sub, d26, "r_follow"),
             "direction_component": dir_component(sub, d26),
             "vs_rest_2026": diff_vs_rest(sub, rest, d26),
             "replication": {}, "mirror_2026": None}
        for y in (2023, 2024, 2025):
            uy = base["user"][base["user"]["year"] == y]
            sy = uy[fn(uy).fillna(False).astype(bool).values]
            r["replication"][str(y)] = {**stats(sy, yd[y]), "hit_rate": fnum(sy["hit"].mean()) if len(sy) else None}
        sm = m26[fn(m26).fillna(False).astype(bool).values]
        r["mirror_2026"] = {**stats(sm, d26), "hit_rate": fnum(sm["hit"].mean()) if len(sm) else None}
        if hid in ALT:
            aid, arule, afn = ALT[hid]
            sa = u26[afn(u26).fillna(False).astype(bool).values]
            r["alt_window_0930"] = {"rule": arule, "overlap_with_main": int((afn(u26).fillna(False).astype(bool).values & mk).sum()),
                                    **stats(sa, d26)}
        dly, _, _ = day_arrays(sub, d26, "r_w90")
        daily_cols[hid] = dly
        rows[hid] = r
        s = r["stats_2026"]
        print(f"{hid}: n={s['trades']} 날 {s['active_days']} 매매당 {s['trade_mean']} 하루 {s['daily_mean']} p {s.get('daily_p_one_sided')} "
              f"({time.time() - T0:.0f}s)", flush=True)

    ps = {h: (rows[h]["stats_2026"].get("daily_p_one_sided") if rows[h]["stats_2026"]["trades"] else 1.0) for h in rows}
    adj = holm(ps)
    ps_tr = {h: (rows[h]["stats_2026"].get("trade_p_one_sided") if rows[h]["stats_2026"]["trades"] else 1.0) for h in rows}
    adj_tr = holm(ps_tr)
    for h in rows:
        rows[h]["p_one_sided"] = ps[h]
        rows[h]["holm_p"] = adj[h]
        rows[h]["pass"] = bool(adj[h] <= ALPHA and (rows[h]["stats_2026"]["daily_mean"] or 0) > 0)
        rows[h]["trade_p_holm_ref"] = adj_tr[h]
        st = rows[h]
        st["extra_checks"] = {
            "half1_gt0": (st["halves"]["H1_jan_jun"]["daily_mean"] or 0) > 0,
            "half2_gt0": (st["halves"]["H2_jul_oct"]["daily_mean"] or 0) > 0,
            "drop_best_month_gt0": bool(st["drop_best_month"] and (st["drop_best_month"]["without"]["daily_mean"] or 0) > 0),
            "cost2x_gt0": (st["cost2x"]["daily_mean"] or 0) > 0}

    # 사후 강건성(결과를 본 뒤 더함, 판정 밖): 통과 가설만
    fns = {hid: fn for hid, _, _, fn in HYP}
    for h in rows:
        if rows[h]["pass"]:
            rows[h]["posthoc_robustness"] = robustness(h, fns[h], u26, base, ev, d26, yd)

    # 참고: 10개 가설 주 통계량의 중심화 최대 t(단측)
    ix = boot_idx(len(d26))
    T = []
    tobs = {}
    for h in rows:
        dly = daily_cols[h]
        bd = dly[ix].mean(1)
        sd = bd.std(ddof=1)
        tobs[h] = fnum(dly.mean() / sd) if sd > 0 else None
        T.append((bd - dly.mean()) / sd if sd > 0 else np.zeros(NB))
    maxt = np.max(np.vstack(T), axis=0)
    ref_maxt = {"q95": fnum(np.percentile(maxt, 95)), "q99": fnum(np.percentile(maxt, 99)), "t_obs": tobs,
                "note": "참고만. t = 하루 평균 ÷ 부트스트랩 표준편차, 귀무 = 가설마다 평균을 뺀 부트스트랩 최대값(단측)"}

    order = sorted(rows, key=lambda h: (rows[h]["holm_p"], rows[h]["p_one_sided"]))
    out = {"brief": "BRIEF4.md 'C3 이론 가설 10개'",
           "data": f"ev4_events.parquet (Data('alp1m_long') {D.days[0]}~{D.days[-1]} {len(D.days)}일)",
           "sample": "side = user(SOXL 오름 → SOXS 매수), x = 3%, r_w90, 2026-01-02~10-28 207일",
           "primary_stat": "하루 손익 평균(그날 매매 곱 − 1, 사건 없는 날 0), 날 복원 추출 10,000회 seed 20261102, 단측 p = 부트스트랩 평균 ≤ 0 비율, Holm m = 10",
           "units": "% (1.0 = 1%)",
           "code_sha256_16": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16],
           "events_sha256_16": hashlib.sha256(open(os.path.join(RES, "ev4_events.parquet"), "rb").read()).hexdigest()[:16],
           "baseline": baseline, "holm_order": order, "n_pass": int(sum(rows[h]["pass"] for h in rows)),
           "ref_max_t": ref_maxt, "hypotheses": rows,
           "limits": ["사건 표는 조건 없는 사슬의 사건을 거른 것이다. 조건 단독 실행은 L 재설정이 달라 사건이 다를 수 있다.",
                      "r_std는 결정 ≤ 09:44 사건만 값이 있어 H1(09:50~10:10)은 std 사건이 0건이다.",
                      "H6·H10·신호 시각 축은 앞 sel(2026 1~6월) 사건 분석에서 단서가 보인 축이라 일부 표본 안이다(BRIEF4)."]}
    with open(os.path.join(RES, "c3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    pd.DataFrame(daily_cols, index=[d.date() for d in d26]).rename_axis("day").to_csv(
        os.path.join(RES, "c3_daily_2026.csv"), float_format="%.10g")
    print("Holm:", {h: round(adj[h], 4) for h in order}, flush=True)
    print(f"완료 ({time.time() - T0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
