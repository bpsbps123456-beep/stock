"""구조 진단(BRIEF2 '원리' ①) — sel 구간(2026-01-02~06-30) 날만 쓴다. 결과: results/structure.json

a) SOXL 1분 로그수익 분산비 VR(q), 09:30~10:00 / 10:00~11:00 (+09:30~11:00)
b) 이어짐: 결정 시각 t의 지난 움직임 r(시가→t) 부호 × 앞으로 r(t→t+h) (부호 맞춤 수익·부호 일치율, |r| 3분위)
c) 갭·프리장(SOXL·SOXX·QQQ·NVDA 08:00→09:29) → r(09:30→10:00)·r(09:30→11:00)·r(09:35→10:00) (+r(09:31→10:00))
d) 한 방향 날 비율(방향 효율 ≥ 0.6), 09:30~11:00 / 09:30~10:00, 같은 날 1분 수익 순서 섞기 기준선
e) 거울: SOXS 1분 수익 대 −SOXL, 평균 차(감쇠), 크기 비
f) 고저폭 중앙값(첫 30·90분), 1분 |수익| 시간대별 중앙값
g) 전날 장초 방향 효율 → 오늘 장초 이어짐(b)
h) (보조) 개장 전 상태 8개 3분위 × 이어짐 3칸 — F5 후보 상태 진단

모든 구간 = 날 묶음 부트스트랩(10,000회, seed 20261029, 모든 통계가 같은 재추출 날을 쓴다) 95% 백분위 구간.
수익 단위 %(1.0 = 1%). 시각 규칙: 't의 가격' = t에 끝나는 봉(t−1분 시작)의 종가, '09:30 가격' = 09:30 봉 시가.
개장 전 상태는 전날까지의 정규장과 그날 09:29 봉 종가까지만 쓴다(첫날의 전날 = 2025-12-31 봉, prev_day 허용 범위).
사용: py structure.py
"""
import json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)   # 프리장 빈 날(SOXS 1일)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, idx, PERIODS, RTH0, cutoff_idx  # noqa: E402

NB, SEED = 10000, 20261029
OUT = os.path.join(HERE, "..", "results", "structure.json")

D = Data()
SEL = D.days_in("sel")
_a, _b = PERIODS["sel"]
assert SEL and all(_a <= str(D.days[k]) <= _b for k in SEL)
N = len(SEL)
PREV = [k - 1 for k in SEL]
BI = np.random.default_rng(SEED).integers(0, N, (NB, N))
SYM4 = ["SOXL", "SOXS", "SOXX", "QQQ", "NVDA"]
O = {s: D.A[s]["Open"][SEL] for s in SYM4}
H = {s: D.A[s]["High"][SEL] for s in SYM4}
L = {s: D.A[s]["Low"][SEL] for s in SYM4}
C = {s: D.A[s]["Close"][SEL] for s in SYM4}
I0930, I1000, I1100 = RTH0, idx("10:00"), idx("11:00")


def rnd(v, k=4):
    return None if v is None or not np.isfinite(v) else round(float(v), k)


def px(sym, t):
    """t 시각에 아는 가격(09:30 = 시가)."""
    return O[sym][:, I0930] if t == "09:30" else C[sym][:, idx(t) - 1]


def _ci(b):
    b = b[np.isfinite(b)]
    return (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)


def mean_ci(x, mask=None, scale=1.0):
    """날마다 한 값(x: (N,)), nan 제외. 평균과 날 묶음 구간."""
    m = np.isfinite(x)
    if mask is not None:
        m &= mask
    if m.sum() == 0:
        return {"n": 0}
    num, den = np.where(m, x * scale, 0.0), m.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num[BI].sum(1) / den[BI].sum(1)
    lo, hi = _ci(b)
    return {"mean": rnd(num.sum() / den.sum()), "lo": rnd(lo), "hi": rnd(hi), "n": int(den.sum())}


def diff_ci(x, m1, m0, scale=1.0):
    """mean(x|m1) − mean(x|m0)."""
    ok = np.isfinite(x)
    a, b_ = ok & m1, ok & m0
    if a.sum() == 0 or b_.sum() == 0:
        return {"n1": int(a.sum()), "n0": int(b_.sum())}
    n1, d1 = np.where(a, x * scale, 0.0), a.astype(float)
    n0, d0 = np.where(b_, x * scale, 0.0), b_.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = n1[BI].sum(1) / d1[BI].sum(1) - n0[BI].sum(1) / d0[BI].sum(1)
    lo, hi = _ci(bs)
    return {"diff": rnd(n1.sum() / d1.sum() - n0.sum() / d0.sum()), "lo": rnd(lo), "hi": rnd(hi),
            "n1": int(a.sum()), "n0": int(b_.sum())}


def median_ci(x, scale=1.0):
    x = x * scale
    b = np.nanmedian(x[BI], axis=1)
    lo, hi = _ci(b)
    return {"median": rnd(np.nanmedian(x)), "lo": rnd(lo), "hi": rnd(hi), "n": int(np.isfinite(x).sum())}


def corr_ci(x, y):
    """순위 상관(Spearman, 순위는 전체 표본에서 고정)."""
    m = np.isfinite(x) & np.isfinite(y)
    rx, ry = np.full(N, np.nan), np.full(N, np.nan)
    rx[m] = np.argsort(np.argsort(x[m])).astype(float)
    ry[m] = np.argsort(np.argsort(y[m])).astype(float)

    def c(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
        n = ok.sum(-1)
        ma, mb = a.sum(-1) / n, b.sum(-1) / n
        cov = (a * b).sum(-1) / n - ma * mb
        va, vb = (a * a).sum(-1) / n - ma ** 2, (b * b).sum(-1) / n - mb ** 2
        return cov / np.sqrt(va * vb)
    lo, hi = _ci(c(rx[BI], ry[BI]))
    return {"rho": rnd(c(rx, ry)), "lo": rnd(lo), "hi": rnd(hi), "n": int(m.sum())}


def bins(x, q):
    """분위 칸 번호(0..q−1), nan은 −1."""
    m = np.isfinite(x)
    edges = np.quantile(x[m], np.arange(1, q) / q)
    out = np.full(N, -1)
    out[m] = np.searchsorted(edges, x[m], side="right")
    return out, [rnd(e * 100) for e in edges]


R = {"meta": {
    "period": "sel", "days": N, "first": str(D.days[SEL[0]]), "last": str(D.days[SEL[-1]]),
    "days_dst_cut1100": int(sum(cutoff_idx(D.days[k]) == I1100 for k in SEL)),
    "days_std_cut1000": int(sum(cutoff_idx(D.days[k]) == I1000 for k in SEL)),
    "boot": {"n": NB, "seed": SEED, "unit": "day"},
    "units": "수익 %(1.0 = 1%), 효율·비율·상관은 그대로",
    "time_rule": "t의 가격 = t에 끝나는 봉 종가, 09:30 = 09:30 봉 시가",
    "note": "구조 진단. 3분위·5분위 문턱은 sel 표본 안에서 정한 사후 값이라 매매 문턱이 아니다."}}

# ───────── a) 분산비 ─────────
QS = [2, 3, 5, 10, 15, 30]


def window_logret(sym, a, b):
    first = O[sym][:, a] if a == RTH0 else C[sym][:, a - 1]
    P = np.concatenate([first[:, None], C[sym][:, a:b]], axis=1)
    return np.diff(np.log(P), axis=1)


def vr_table(r):
    n_day, m = r.shape
    S1, S2 = r.sum(1), (r * r).sum(1)
    cs = np.concatenate([np.zeros((n_day, 1)), np.cumsum(r, 1)], 1)
    out = {}
    for q in QS:
        if q > m:
            continue
        W = cs[:, q:] - cs[:, :-q]
        SW, SW2, nq = W.sum(1), (W * W).sum(1), W.shape[1]

        def vr(s1, s2, sw, sw2, nd):
            n1, nqt = m * nd, nq * nd
            mu = s1 / n1
            v1 = s2 / n1 - mu ** 2
            vq = sw2 / nqt - 2 * q * mu * sw / nqt + (q * mu) ** 2
            return vq / (q * v1)
        est = vr(S1.sum(), S2.sum(), SW.sum(), SW2.sum(), n_day)
        bs = vr(S1[BI].sum(1), S2[BI].sum(1), SW[BI].sum(1), SW2[BI].sum(1), n_day)
        lo, hi = _ci(bs)
        out[str(q)] = {"vr": rnd(est), "lo": rnd(lo), "hi": rnd(hi), "windows_per_day": int(nq)}
    return out


a = {}
for name, (s, e) in {"0930_1000": (I0930, I1000), "1000_1100": (I1000, I1100), "0930_1100": (I0930, I1100)}.items():
    r = window_logret("SOXL", s, e)
    rs = r / np.sqrt((r * r).mean(0))                       # 분(分)별 변동성으로 나눈 판(첫 분 지배 제거)
    rho1 = (r[:, 1:] * r[:, :-1]).sum() / np.sqrt((r[:, 1:] ** 2).sum() * (r[:, :-1] ** 2).sum())
    a[name] = {"returns_per_day": int(r.shape[1]), "raw": vr_table(r), "minute_std": vr_table(rs),
               "rho1_uncentered": rnd(rho1)}
R["a_variance_ratio"] = {"desc": "SOXL 1분 로그수익(첫 수익 = 09:30 봉 시가→종가), 날 안 겹침 창, 표본 평균 제거. "
                                 "raw = 원 수익, minute_std = 분별 RMS로 나눈 수익(주 판). VR>1 추세, <1 되돌림. "
                                 "raw는 창 앞쪽 큰 분산(09:30 봉)이 겹침 창에 덜 들어가 q가 클수록 아래로 치우친다",
                         "windows": a}

# ───────── b) 이어짐 ─────────
TS = ["09:31", "09:35", "09:45", "10:00"]
HZ = [("h5", 5), ("h15", 15), ("h30", 30), ("to1000", "10:00"), ("to1100", "11:00")]
open_px = px("SOXL", "09:30")
B, PAST, CELL = {}, {}, {}
for t in TS:
    it = idx(t)
    Pt = px("SOXL", t)
    past = Pt / open_px - 1
    s = np.sign(past)
    s = np.where(s == 0, np.nan, s)
    PAST[t] = past
    terc, edges = bins(np.abs(past), 3)
    cells = {}
    for hn, hv in HZ:
        ie = it + hv if isinstance(hv, int) else idx(hv)
        if ie <= it:
            continue
        f = C["SOXL"][:, ie - 1] / Pt - 1
        al = s * f
        hit = np.where(np.isfinite(s) & (f != 0), (np.sign(f) == s).astype(float), np.nan)
        fd = C["SOXL"][:, ie - 1] / C["SOXL"][:, it] - 1 if ie - 1 > it else np.full(N, np.nan)
        fs = C["SOXS"][:, ie - 1] / C["SOXS"][:, it - 1] - 1
        trade = np.where(s > 0, f, np.where(s < 0, fs, np.nan))
        CELL[(t, hn)] = al
        cells[hn] = {"end": f"{(ie + 240) // 60:02d}:{(ie + 240) % 60:02d}",
                     "aligned": mean_ci(al, scale=100), "hit": mean_ci(hit),
                     "drift_part": rnd(np.nanmean(s) * np.mean(f) * 100),
                     "aligned_delay1": mean_ci(s * fd, scale=100),
                     "trade_soxl_or_soxs": mean_ci(trade, scale=100),
                     "by_abs_past_tercile": {f"T{k + 1}": {"aligned": mean_ci(al, terc == k, 100),
                                                           "hit": mean_ci(hit, terc == k)} for k in range(3)},
                     "T3_minus_T1": diff_ci(al, terc == 2, terc == 0, 100)}
    B[t] = {"abs_past_tercile_edges_pct": edges, "abs_past_median_pct": rnd(np.median(np.abs(past)) * 100),
            "up_share": rnd((past > 0).mean()), "cells": cells}
R["b_continuation"] = {"desc": "aligned = sign(r(09:30 시가→t)) × r(t→끝) [SOXL %]. 양수 = 이어짐, 음수 = 되돌림. "
                                "aligned_delay1 = 끝은 같고 시작을 t+1분 가격으로(1분 지연 체결). drift_part = mean(부호) × mean(앞 수익)(추세 기울기 몫). "
                                "trade_soxl_or_soxs = 위면 SOXL, 아래면 SOXS의 실제 수익(비용 전). hit = 부호 일치율",
                       "times": B}

# ───────── c) 갭·프리장 ─────────
prev_close = np.array([D.daily.loc[("SOXL", D.days[k]), "close"] for k in PREV])
gap = open_px / prev_close - 1


def pre_ret(sym):
    p0 = C[sym][:, idx("08:00") - 1].copy()             # 08:00 가격 = 07:59 봉 종가(같은 날 앞 종가 채움)
    for r_ in np.where(~np.isfinite(p0))[0]:
        hs = np.where(D.has[sym][SEL[r_], idx("08:00"):I0930])[0]
        if len(hs):
            p0[r_] = O[sym][r_, idx("08:00") + hs[0]]
    return C[sym][:, I0930 - 1] / p0 - 1                 # 09:29 봉 종가(09:30에 앎)


PRED = {"SOXL_gap": gap, "SOXL_pre": pre_ret("SOXL"), "SOXX_pre": pre_ret("SOXX"),
        "QQQ_pre": pre_ret("QQQ"), "NVDA_pre": pre_ret("NVDA")}
TGT = {"r0930_1000": px("SOXL", "10:00") / open_px - 1, "r0930_1100": px("SOXL", "11:00") / open_px - 1,
       "r0935_1000": px("SOXL", "10:00") / px("SOXL", "09:35") - 1,
       "r0931_1000": px("SOXL", "10:00") / px("SOXL", "09:31") - 1}
lo30 = L["SOXL"][:, I0930:I1000].min(1)
hi30 = H["SOXL"][:, I0930:I1000].max(1)
lo90 = L["SOXL"][:, I0930:I1100].min(1)
hi90 = H["SOXL"][:, I0930:I1100].max(1)
fill30 = np.where(gap > 0, lo30 <= prev_close, np.where(gap < 0, hi30 >= prev_close, np.nan)).astype(float)
fill90 = np.where(gap > 0, lo90 <= prev_close, np.where(gap < 0, hi90 >= prev_close, np.nan)).astype(float)
c = {}
for pn, x in PRED.items():
    q5, e5 = bins(x, 5)
    t3, e3 = bins(np.abs(x), 3)
    sx = np.sign(x)
    sx = np.where(sx == 0, np.nan, sx)
    ent = {"quintile_edges_pct": e5, "abs_tercile_edges_pct": e3,
           "corr_with_SOXL_gap": corr_ci(x, gap)["rho"] if pn != "SOXL_gap" else 1.0, "targets": {}}
    for tn, y in TGT.items():
        al = sx * y
        hit = np.where(np.isfinite(sx) & (y != 0), (np.sign(y) == sx).astype(float), np.nan)
        ent["targets"][tn] = {
            "spearman": corr_ci(x, y),
            "by_quintile_mean_target": {f"Q{k + 1}": mean_ci(y, q5 == k, 100) for k in range(5)},
            "aligned_follow": mean_ci(al, scale=100), "hit_follow": mean_ci(hit),
            "aligned_by_abs_tercile": {f"T{k + 1}": mean_ci(al, t3 == k, 100) for k in range(3)}}
    if pn == "SOXL_gap":
        ent["gap_fill_touch_prev_close"] = {
            "by_10_00": {f"Q{k + 1}": mean_ci(fill30, q5 == k) for k in range(5)},
            "by_11_00": {f"Q{k + 1}": mean_ci(fill90, q5 == k) for k in range(5)},
            "all_by_10_00": mean_ci(fill30), "all_by_11_00": mean_ci(fill90)}
    c[pn] = ent
R["meta"]["unconditional_pct"] = {
    **{f"SOXL_{k}": mean_ci(v, scale=100) for k, v in TGT.items()},
    "SOXS_r0930_1000": mean_ci(px("SOXS", "10:00") / px("SOXS", "09:30") - 1, scale=100),
    "SOXS_r0930_1100": mean_ci(px("SOXS", "11:00") / px("SOXS", "09:30") - 1, scale=100),
    "SOXL_gap": mean_ci(gap, scale=100)}
R["c_gap_premarket"] = {"desc": "갭 = 09:30 시가 ÷ 전날 정규장 종가 − 1, 프리장 = 08:00 가격 → 09:29 봉 종가. "
                                 "aligned_follow = sign(예측 변수) × 목표 수익(양수 = 따라가기, 음수 = 메우기). "
                                 "r0931_1000은 09:30에 알고 1분 지연 체결하는 실행 가능판(추가)",
                        "predictors": c}

# ───────── d) 한 방향 날 ─────────


def eff_hl(rows_o, rows_h, rows_l, rows_c, a, b):
    net = rows_c[:, b - 1] - rows_o[:, a]
    rng = rows_h[:, a:b].max(1) - rows_l[:, a:b].min(1)
    return net / rng


def eff_path(P):
    return (P[:, -1] - P[:, 0]) / (P.max(1) - P.min(1))


def perm_baseline(sym, a, b, nperm=200, seed=SEED):
    """같은 날 1분 로그수익의 순서만 섞은 기준선(순 움직임·변동성 같음)."""
    rng_ = np.random.default_rng(seed)
    r = window_logret(sym, a, b)
    share = np.zeros(N)
    for _ in range(nperm):
        rp = rng_.permuted(r, axis=1)
        P = np.exp(np.concatenate([np.zeros((N, 1)), np.cumsum(rp, 1)], 1))
        share += np.abs(eff_path(P)) >= 0.6
    return share / nperm


d = {}
for name, (s, e) in {"0930_1100": (I0930, I1100), "0930_1000": (I0930, I1000)}.items():
    ef = eff_hl(O["SOXL"], H["SOXL"], L["SOXL"], C["SOXL"], s, e)
    P = np.concatenate([O["SOXL"][:, [s]], C["SOXL"][:, s:e]], 1)
    efp = eff_path(P)
    base = perm_baseline("SOXL", s, e)
    d[name] = {"one_way_share_hl": mean_ci((np.abs(ef) >= 0.6).astype(float)),
               "one_way_up_share": rnd((ef >= 0.6).mean()), "one_way_down_share": rnd((ef <= -0.6).mean()),
               "median_abs_eff_hl": median_ci(np.abs(ef)),
               "one_way_share_closepath": mean_ci((np.abs(efp) >= 0.6).astype(float)),
               "perm_baseline_share_closepath": mean_ci(base),
               "closepath_minus_perm": mean_ci((np.abs(efp) >= 0.6).astype(float) - base)}
R["d_one_way_days"] = {"desc": "방향 효율 = (끝 가격 − 09:30 시가) ÷ 고저폭. hl = 봉 고가·저가 폭(앞 연구 정의), "
                                "closepath = 시가+1분 종가 경로 폭. perm = 같은 날 1분 수익 순서 섞기 200회 기준선. "
                                "날 성격은 사후 값(설명용)",
                       "windows": d}

# ───────── e) 거울 ─────────


def min_ret(sym, a, b):
    first = O[sym][:, a] if a == RTH0 else C[sym][:, a - 1]
    P = np.concatenate([first[:, None], C[sym][:, a:b]], axis=1)
    return P[:, 1:] / P[:, :-1] - 1


e_out = {}
for name, (s, e) in {"0930_0935": (I0930, idx("09:35")), "0935_1000": (idx("09:35"), I1000),
                     "1000_1100": (I1000, I1100), "0930_1100": (I0930, I1100),
                     "pre_0800_0930": (idx("08:00"), I0930)}.items():
    xl, xs = min_ret("SOXL", s, e), min_ret("SOXS", s, e)
    both = D.has["SOXL"][SEL][:, s:e] & D.has["SOXS"][SEL][:, s:e] & np.isfinite(xl) & np.isfinite(xs)
    y, x = np.where(both, xs, 0.0), np.where(both, -xl, 0.0)
    n = both.sum(1).astype(float)
    Sx, Sy, Sxx, Syy, Sxy = x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1), (x * y).sum(1)
    Ax, Ay = np.abs(x).sum(1), np.abs(y).sum(1)

    def stats(sx, sy, sxx, syy, sxy, nn, ax, ay):
        mx, my = sx / nn, sy / nn
        cov = sxy / nn - mx * my
        vx, vy = sxx / nn - mx ** 2, syy / nn - my ** 2
        return cov / np.sqrt(vx * vy), (sy - sx) / nn, ay / ax, cov / vx

    est = stats(Sx.sum(), Sy.sum(), Sxx.sum(), Syy.sum(), Sxy.sum(), n.sum(), Ax.sum(), Ay.sum())
    bs = stats(*(v[BI].sum(1) for v in (Sx, Sy, Sxx, Syy, Sxy, n, Ax, Ay)))
    keys = ["corr_soxs_vs_neg_soxl", "mean_diff_soxs_minus_negsoxl_pct_per_min", "abs_size_ratio_soxs_over_soxl",
            "slope_soxs_on_neg_soxl"]
    ent = {"minutes_both": int(n.sum()), "minutes_total": int(both.size)}
    for k_, ev, bv in zip(keys, est, bs):
        sc = 100 if "pct" in k_ else 1
        lo, hi = _ci(bv * sc)
        ent[k_] = {"est": rnd(ev * sc, 5), "lo": rnd(lo, 5), "hi": rnd(hi, 5)}
    # 창 전체: SOXL r + SOXS r (둘 다 들고 있으면 남는 것 — 거울이면 0)
    pl = C["SOXL"][:, e - 1] / (O["SOXL"][:, s] if s == RTH0 else C["SOXL"][:, s - 1]) - 1
    ps = C["SOXS"][:, e - 1] / (O["SOXS"][:, s] if s == RTH0 else C["SOXS"][:, s - 1]) - 1
    ent["window_soxl_plus_soxs_pct"] = mean_ci(pl + ps, scale=100)
    ent["window_soxs_over_neg_soxl_median_ratio"] = median_ci(np.where(np.abs(pl) > 0.005, -ps / pl, np.nan))
    e_out[name] = ent
R["e_mirror"] = {"desc": "1분 단순 수익. 두 종목 모두 실제 봉이 있는 분만. mean_diff = SOXS − (−SOXL) 분당 평균(%), "
                         "abs_size_ratio = Σ|SOXS| ÷ Σ|SOXL|. window_* = 창 시작→끝 수익(감쇠·비대칭), "
                         "median_ratio는 |SOXL 창 수익| > 0.5%인 날만",
                 "windows": e_out}

# ───────── f) 고저폭·시간대별 1분 |수익| ─────────
f_out = {"range_pct": {}}
for sym in ["SOXL", "SOXS"]:
    for name, (s, e) in {"first30": (I0930, I1000), "first90": (I0930, I1100)}.items():
        rg = (H[sym][:, s:e].max(1) - L[sym][:, s:e].min(1)) / O[sym][:, s]
        f_out["range_pct"][f"{sym}_{name}"] = {**median_ci(rg, 100), "mean": rnd(rg.mean() * 100)}
buckets = [(f"{hh}", idx(hh), idx(hh) + 1) for hh in
           ["09:30", "09:31", "09:32", "09:33", "09:34", "09:35", "09:36", "09:37", "09:38", "09:39"]]
for a_ in range(idx("09:40"), I1100, 10):
    buckets.append((f"{(a_ + 240) // 60:02d}:{(a_ + 240) % 60:02d}+10m", a_, a_ + 10))
pre_b = [("08:00+30m", idx("08:00"), idx("08:30")), ("08:30+30m", idx("08:30"), idx("09:00")),
         ("09:00+15m", idx("09:00"), idx("09:15")), ("09:15+15m", idx("09:15"), I0930)]
f_out["abs_1min_ret_median_pct"] = {}
for sym in ["SOXL", "SOXS"]:
    allr = np.abs(min_ret(sym, idx("08:00"), I1100))
    hasm = D.has[sym][SEL][:, idx("08:00"):I1100]
    allr = np.where(hasm, allr, np.nan)                    # 빈 분(채운 값)은 뺀다
    off = idx("08:00")
    tab = {}
    for nm, s, e in pre_b + buckets:
        tab[nm] = median_ci(np.nanmedian(allr[:, s - off:e - off], axis=1), 100)
    f_out["abs_1min_ret_median_pct"][sym] = tab
R["f_range_and_abs_returns"] = {"desc": "고저폭 = (창 최고가 − 최저가) ÷ 09:30 시가. 1분 |수익|: 날마다 칸 안 중앙값 → 날들의 중앙값. "
                                        "첫 수익은 09:30 봉 시가→종가, 프리장 빈 분 제외",
                                "values": f_out}

# ───────── g) 전날 장초 추세 → 오늘 이어짐 ─────────
Dp = {k: D.A["SOXL"][k][PREV] for k in ["Open", "High", "Low", "Close"]}
prev_eff = eff_hl(Dp["Open"], Dp["High"], Dp["Low"], Dp["Close"], I0930, I1100)
prev_trend = np.abs(prev_eff) >= 0.6
pt3, pe3 = bins(np.abs(prev_eff), 3)
g_cells = {}
for (t, hn), al in CELL.items():
    g_cells[f"{t}|{hn}"] = {"prev_trend": mean_ci(al, prev_trend, 100), "prev_chop": mean_ci(al, ~prev_trend, 100),
                            "trend_minus_chop": diff_ci(al, prev_trend, ~prev_trend, 100),
                            "by_prev_abs_eff_tercile": {f"T{k + 1}": mean_ci(al, pt3 == k, 100) for k in range(3)}}
sp = np.sign(prev_eff)
g_dir = {tn: {"aligned_prev_morning_dir": mean_ci(sp * TGT[tn], scale=100),
              "same_on_prev_trend_days": mean_ci(sp * TGT[tn], prev_trend, 100)}
         for tn in ["r0930_1000", "r0930_1100", "r0931_1000"]}
R["g_prev_morning_trend"] = {"desc": "전날 09:30~11:00 방향 효율(hl) ≥ 0.6 = 추세. 오늘 b의 부호 맞춤 수익을 나눠 본다. "
                                     "direction = sign(전날 장초 순 움직임) × 오늘 장초 수익",
                             "prev_trend_share": rnd(prev_trend.mean()), "prev_abs_eff_tercile_edges": [
                                 rnd(v / 100) for v in pe3],
                             "cells": g_cells, "direction": g_dir}

# ───────── h) (보조) 개장 전 상태 × 이어짐 ─────────
dl = D.daily.loc["SOXL"]
dclose = {d_: v for d_, v in dl["close"].items()}
prev_rng = np.array([(dl.loc[D.days[k], "high"] - dl.loc[D.days[k], "low"]) / dl.loc[D.days[k], "open"] for k in PREV])
prev_ret = np.array([dclose[D.days[k]] / dclose[D.days[k - 1]] - 1 for k in PREV])
trend5 = np.array([dclose[D.days[k]] / dclose[D.days[k - 5]] - 1 for k in PREV])
STATES = {"prev_range": prev_rng, "abs_gap": np.abs(gap), "prev_ret": prev_ret, "trend5": trend5,
          "prev_abs_eff": np.abs(prev_eff), "abs_QQQ_pre": np.abs(PRED["QQQ_pre"]),
          "abs_NVDA_pre": np.abs(PRED["NVDA_pre"])}
HC = [("09:31", "to1000"), ("09:35", "to1000"), ("09:45", "to1100")]
h_out = {}
for sn, x in STATES.items():
    t3, e3 = bins(x, 3)
    h_out[sn] = {"tercile_edges_pct": e3, "cells": {
        f"{t}|{hn}": {**{f"T{k + 1}": mean_ci(CELL[(t, hn)], t3 == k, 100) for k in range(3)},
                      "T3_minus_T1": diff_ci(CELL[(t, hn)], t3 == 2, t3 == 0, 100)} for t, hn in HC}}
wd = np.array([D.days[k].weekday() for k in SEL])
h_out["weekday"] = {"cells": {f"{t}|{hn}": {["Mon", "Tue", "Wed", "Thu", "Fri"][w]: mean_ci(CELL[(t, hn)], wd == w, 100)
                                            for w in range(5)} for t, hn in HC}}
R["h_prestate_x_continuation"] = {
    "desc": "요청 밖 보조. F5 후보 상태 8개(요일 5칸, 나머지 3분위) × 이어짐 3칸(b의 부호 맞춤 수익). "
            "평균 수 = 7×3×3 + 5×3 = 78. 대량 비교라 개별 칸 구간이 0을 벗어나도 근거로 쓰지 않는다",
    "states": h_out}

# ───────── i) (보조) F1 손절 진단: 첫 N분 범위의 반대쪽 끝에 닿는 비율 ─────────
i_out = {}
for nmin in [1, 5, 15, 30]:
    e_ = I0930 + nmin
    rh, rl = H["SOXL"][:, I0930:e_].max(1), L["SOXL"][:, I0930:e_].min(1)
    up = C["SOXL"][:, e_ - 1] > O["SOXL"][:, I0930]
    dn = C["SOXL"][:, e_ - 1] < O["SOXL"][:, I0930]
    ent = {"range_width_pct": median_ci((rh - rl) / O["SOXL"][:, I0930], 100), "dir_days": int((up | dn).sum())}
    for nm, end_ in [("by_10_00", I1000), ("by_11_00", I1100)]:
        if end_ <= e_:
            continue
        lo_, hi_ = L["SOXL"][:, e_:end_].min(1), H["SOXL"][:, e_:end_].max(1)
        touch = np.where(up, lo_ <= rl, np.where(dn, hi_ >= rh, np.nan)).astype(float)
        ent[nm] = mean_ci(touch)
    i_out[f"N{nmin}"] = ent
R["i_orb_opposite_touch"] = {"desc": "요청 밖 보조(F1 손절 설계용). 첫 N분 범위 봉이 양봉이면 범위 저가, 음봉이면 범위 고가에 "
                                     "범위가 끝난 뒤 닿은 날 비율(SOXL 봉 고저). 방향 수익은 b의 t = 09:31·09:35·09:45·10:00 칸과 같다",
                             "N": i_out}

R["search_scope"] = {
    "a": "창 3 × q 6 × 판 2 = 36 VR(창 0930_1000의 q=30은 하루 1창)",
    "b": "t 4 × 지평 5 − 1(10:00→10:00) = 19칸, 칸마다 전체·3분위 3·지연·실매매",
    "c": "예측 변수 5 × 목표 4 = 20쌍, 쌍마다 5분위·따라가기·|x| 3분위",
    "d": "창 2", "e": "창 5", "f": "종목 2 × (고저폭 2 + 시간대 22)",
    "g": "b의 19칸 × (추세/오락가락, 3분위) + 방향 3", "h": "78 평균", "i": "N 4 × 마감 2"}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("saved", os.path.normpath(OUT), "days", N)
