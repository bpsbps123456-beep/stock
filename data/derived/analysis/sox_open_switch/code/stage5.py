"""BRIEF5 — SOXX 추세 5단계별로 나눠 보기 (구현 A, 2026-11-03).

정본: ../BRIEF5.md(결과 보기 전 고정). 사건 표 ../results/ev4_events.parquet(BRIEF4), 엔진 lib.py(고치지 않음).

결과 보기 전에 이 파일에서 정한 해석(첫 실행 전 고정):
- 단계: yfinance SOXX 일봉 2021-01-01~2026-10-28(auto_adjust=False Close = 분할 반영·배당 미반영).
  lc = ln C, lr = lc − lc[1], sd = lr 20일 표본 표준편차, k = √((N−1)(2N−1)/(6N)), z = (lc − lc 20일 SMA) ÷ (sd·k),
  RSI14 = 파인 ta.rsi(첫 14개 변화 단순 평균으로 시작하는 RMA, dn = 0이면 100, up = 0이면 0).
  RSI ≥ 70 → +2, ≤ 30 → −2, 그 밖에 z ≥ 0.5 → +1, z ≤ −0.5 → −1, 나머지 0. 내려받은 첫 130봉(bar_index < 129)은 값 없음.
  D일(Data 날)에는 yfinance 날짜열에서 D 바로 앞 거래일 종가의 단계를 붙인다.
- 대조: ../trend_zones/code/pine_port.pine_zone(c, L=20)(RSI·10-13 이전 z 정의)과
  ../trend_zones/k3_20261013/turn_A/turn_compare(rsi_pine·kmult·stage5, 10-13 정의 = 지금 Pine)에 같은 종가를 넣어 비교.
- 표 A: Engine(D, scheme ∈ {w90, std}) × 늘 SOXL(+1)·늘 SOXS(−1). 09:30 봉 마감 결정 → 09:31 봉 종가 체결(+반 호가)
  → 마감 1분 전 봉(w90 10:59, std 09:59) 종가 청산(−반 호가). 2023-01-03~2026-10-28 959일 전부.
  칸 = 해 × 단계 5 × 종목 2 × 마감 2(해마다 20칸). 칸 통계 = 그 단계 날의 하루 수익 평균.
  95% 구간 = 칸의 날을 복원 추출 10,000회(seed 20261103, 칸마다 새 rng)의 2.5·97.5 분위.
- 원형 이동 귀무(모든 표 공통): 그 해 날들(거래일 순) 안에서 단계 열을 k칸 np.roll, k ~ 정수 균등 [20, n − 20],
  5,000회(해마다 np.random.default_rng(20261103)에서 새로 뽑음 → 같은 해의 표 A·B·C는 같은 k 열).
  칸 t = (관측 − 귀무 평균) ÷ 귀무 표준편차. 칸 p(양측) = (1 + #{|t*| ≥ |t|}) ÷ 5,001.
  가족 보정 p = (1 + #{max_칸 |t*| ≥ |t|}) ÷ 5,001(해·표마다: A 20칸, B 10칸, 단일 단계 최대 t).
- 표 B: x = 0.03(np.isclose), 면 2(user = SOXL 오름 → SOXS, mirror = SOXS 오름 → SOXL) × 단계 5, r_w90.
  하루 손익 = 그 면 그날 사건 r_w90의 곱 − 1, 사건 없는 날 0, 분모 = 그 해 거래일 전체. p용 칸 통계 = 하루 평균.
  칸마다 건수·날 수·매매당 r_w90·도달률·매매당 r_follow·방향 성분 (r_w90 − r_follow)/2(매매당)·하루 평균과 그 95% 구간
  (그 해 날 전체 복원 추출 10,000회, seed 20261103).
- C: 주 가설 'trend'(추세 쪽 되돌리기) = 단계 ≤ −1인 날 user 면 사건만, ≥ +1인 날 mirror 면 사건만, 0인 날 없음.
  둘째 'counter'(추세 반대 쪽) = 단계 ≤ −1 → mirror, ≥ +1 → user, 0 → 없음.
  한 번에 한 종목: 그날 고른 사건을 체결 봉 순으로 놓고, 앞 매매 청산 봉 이하(fill_i ≤ 앞 exit_i)에 체결되는 사건은 건너뜀.
  하루 손익 = 곱 − 1(%), 사건 없는 날 0. 2026 207일 평균.
  p(단측) = (1 + #{귀무 평균 ≥ 관측}) ÷ 5,001, 2026 날들 안 원형 이동. Holm m = 2(trend·counter).
  판정: (1) 평균 > 0 (2) Holm 보정 p ≤ 0.05 (3) 1~6월·7~10월 하루 평균 모두 > 0 (4) r_w90_c2(반 호가 2배) 하루 평균 > 0.
  넷 다 만족이면 통과. 함께(판정 밖): 날 묶음 95% 구간, 가장 좋은 달(하루 손익 합 최대) 빼기, 상위 5일(하루 손익 큰 5일)을
  표본에서 빼기, 추종·방향 성분, 하방 쪽·상방 쪽 기여, 2023·2024·2025(그 해 날들 안 원형 이동, 같은 판정 항목).
- 이어진 구간: 같은 단계가 연속한 날 묶음 하나를 한 구간으로 센다(D에 붙인 단계 열, 해 안에서).
실행: cd code && py stage5.py
결과: ../results/stage5.json, stage5_stages.csv, stage5_daily_A.csv, stage5_daily_B.csv, stage5_daily_C.csv
"""
import hashlib, json, os, sys, time

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF5 허용: 2023~2026 전체를 쓴다
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, Engine, Order, RTH0  # noqa: E402

TZ_CODE = os.path.normpath(os.path.join(HERE, "..", "..", "trend_zones", "code"))
TURN_A = os.path.normpath(os.path.join(HERE, "..", "..", "trend_zones", "k3_20261013", "turn_A"))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
CACHE = os.path.normpath(os.path.join(HERE, "..", "cache"))
SEED, NB, NSH, MINSH = 20261103, 10000, 5000, 20
N_TR, TH, RL, RHI, RLO, MIN_IDX = 20, 0.5, 14, 70.0, 30.0, 129
X = 0.03
YEARS = (2023, 2024, 2025, 2026)
STAGES = (-2, -1, 0, 1, 2)
NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}
SIDES = ("user", "mirror")
T0 = time.time()


def fnum(v):
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- 단계
def soxx_daily():
    p = os.path.join(CACHE, "soxx_1d_yf_20210101_20261028.csv")
    if os.path.exists(p):
        s = pd.read_csv(p, index_col=0, parse_dates=True)["Close"]
        return s
    import yfinance as yf
    df = yf.download("SOXX", start="2021-01-01", end="2026-10-29", auto_adjust=False, progress=False, actions=False)
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
    c = c.dropna().astype(float)
    c.name = "Close"
    os.makedirs(CACHE, exist_ok=True)
    c.to_frame().to_csv(p)
    return c


def rsi_wilder(c, n=RL):
    out = np.full(len(c), np.nan)
    d = np.r_[np.nan, np.diff(c)]
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au = ad = None
    for i in range(1, len(c)):
        if i == n:
            au, ad = up[1:n + 1].mean(), dn[1:n + 1].mean()
        elif i > n:
            au = au + (up[i] - au) / n
            ad = ad + (dn[i] - ad) / n
        else:
            continue
        out[i] = 100.0 if ad == 0 else (0.0 if au == 0 else 100 - 100 / (1 + au / ad))
    return out


def zone_calc(c):
    lc = np.log(c)
    s = pd.Series(lc)
    lr = s.diff()
    sd = lr.rolling(N_TR, min_periods=N_TR).std(ddof=1).to_numpy()
    k = np.sqrt((N_TR - 1.0) * (2.0 * N_TR - 1.0) / (6.0 * N_TR))
    z = (lc - s.rolling(N_TR, min_periods=N_TR).mean().to_numpy()) / (sd * k)
    r = rsi_wilder(c)
    zn = np.select([r >= RHI, r <= RLO, z >= TH, z <= -TH], [2, -2, 1, -1], 0).astype(float)
    ok = np.isfinite(z) & np.isfinite(r) & (np.arange(len(c)) >= MIN_IDX)
    return np.where(ok, zn, np.nan), r, z


def z_loop(c):
    """독립 확인용: 창마다 np.mean·np.std(ddof=1)로 z를 다시 잰다."""
    lc = np.log(c)
    lr = np.r_[np.nan, np.diff(lc)]
    k = np.sqrt((N_TR - 1.0) * (2.0 * N_TR - 1.0) / (6.0 * N_TR))
    z = np.full(len(c), np.nan)
    for i in range(N_TR, len(c)):
        w = lr[i - N_TR + 1:i + 1]
        z[i] = (lc[i] - lc[i - N_TR + 1:i + 1].mean()) / (np.std(w, ddof=1) * k)
    return z


def runs(a):
    a = np.asarray(a)
    if not len(a):
        return 0
    return int(1 + (a[1:] != a[:-1]).sum())


def stage_check(c, dates, zn, r, z):
    sys.path.insert(0, TZ_CODE)
    import pine_port  # noqa: E402
    sys.path.insert(0, TURN_A)
    import turn_compare as TC  # noqa: E402
    out = {}
    zp, rp, zold = pine_port.pine_zone(c, L=N_TR, th=TH, rl=RL, hi=RHI, lo=RLO)
    both = np.isfinite(r) & np.isfinite(rp)
    out["pine_port_rsi_max_abs_diff"] = fnum(np.abs(r[both] - rp[both]).max())
    out["pine_port_rsi_n"] = int(both.sum())
    ev = np.isfinite(zn) & np.isfinite(zp)
    out["pine_port_zone_agree_all"] = fnum((zn[ev] == zp[ev]).mean())
    m23 = ev & (dates >= pd.Timestamp("2022-12-01"))
    out["pine_port_zone_agree_2022_12_on"] = fnum((zn[m23] == zp[m23]).mean())
    ext = m23 & ((r >= RHI) | (r <= RLO))
    out["pine_port_zone_agree_on_rsi_extreme_days"] = fnum((zn[ext] == zp[ext]).mean()) if ext.any() else None
    out["pine_port_valid_same"] = bool((np.isfinite(zp) == np.isfinite(zn)).all())
    out["pine_port_note"] = ("pine_port.pine_zone의 z는 10-13 이전 정의(ln(C/C[L]) ÷ (sd·√L), N일 전 한 점 기준)라 "
                             "지금 Pine(20일 SMA 기준, k 배율)과 z가 다르다. RSI·극단(±2)·130봉 규칙은 같은 함수로 대조된다.")
    # 10-13 정의 이식(turn_compare): rsi_pine, kmult, T3_20, stage5
    rt = TC.rsi_pine(c)
    s = pd.Series(np.log(c))
    sd = s.diff().rolling(N_TR, min_periods=N_TR).std(ddof=1)
    t3 = ((s - s.rolling(N_TR, min_periods=N_TR).mean()) / (sd * TC.kmult(N_TR))).to_numpy()
    st = TC.stage5(rt, t3, TH).astype(float)
    valid = (np.arange(len(c)) >= TC.MIN_IDX) & np.isfinite(rt) & np.isfinite(t3)
    st = np.where(valid, st, np.nan)
    b2 = np.isfinite(st) & np.isfinite(zn)
    out["turn_compare_rsi_max_abs_diff"] = fnum(np.nanmax(np.abs(rt - r)))
    out["turn_compare_z_max_abs_diff"] = fnum(np.nanmax(np.abs(t3 - z)))
    out["turn_compare_zone_agree"] = fnum((st[b2] == zn[b2]).mean())
    out["turn_compare_valid_same"] = bool((np.isfinite(st) == np.isfinite(zn)).all())
    zl = z_loop(c)
    out["z_loop_max_abs_diff"] = fnum(np.nanmax(np.abs(zl - z)))
    out["first_valid_date"] = str(dates[np.argmax(np.isfinite(zn))].date())
    out["z_within_1e-6_of_threshold"] = int((np.abs(np.abs(z) - TH) < 1e-6).sum())
    return out


# ---------------------------------------------------------------- 원형 이동
_KS = {}


def shifts(n):
    if n not in _KS:
        _KS[n] = np.random.default_rng(SEED).integers(MINSH, n - MINSH + 1, NSH)
    return _KS[n]


def rotated(st):
    """(NSH, n) 단계 행렬: 행 b = np.roll(st, k_b)."""
    n = len(st)
    ks = shifts(n)
    R = (np.arange(n)[None, :] - ks[:, None]) % n
    return st[R]


def maxT(obs, null):
    """obs (C,), null (NSH, C) → 칸 t, 칸 양측 p, 가족 보정 p, 귀무 평균·표준편차."""
    mu = np.nanmean(null, 0)
    sd = np.nanstd(null, 0, ddof=1)
    ok = np.isfinite(obs) & (sd > 0)
    den = np.where(sd > 0, sd, 1)
    t = np.where(ok, (obs - mu) / den, np.nan)
    tn = np.where(ok[None, :], (null - mu) / den, np.nan)
    at = np.abs(tn)
    mx = np.nanmax(at, 1)
    p = np.where(ok, (1 + (at >= np.abs(t)[None, :] - 1e-12).sum(0)) / (NSH + 1), np.nan)
    pf = np.where(ok, (1 + (mx[:, None] >= np.abs(t)[None, :] - 1e-12).sum(0)) / (NSH + 1), np.nan)
    return t, p, pf, mu, sd


def boot_mean(v, seed=SEED):
    v = np.asarray(v, float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    return [fnum(np.percentile(bs, 2.5)), fnum(np.percentile(bs, 97.5))]


# ---------------------------------------------------------------- 표 A
class Always:
    """09:30 봉 마감 결정 → 다음 봉(09:31) 종가 체결 → 마감까지 한 종목(confirm.Always와 같은 정의)."""
    start = "09:30"

    def __init__(self, pos):
        self.pos = pos

    def on_bar(self, ctx, i):
        return Order(self.pos) if i == RTH0 else None


A_COLS = [f"{inst}_{sch}" for sch in ("w90", "std") for inst in ("SOXL", "SOXS")]


def table_A(dA, yidx):
    out = {}
    for y in YEARS:
        ix = yidx[y]
        st = dA["stage_prev"].values[ix].astype(int)
        V = dA[A_COLS].values[ix]
        rot = rotated(st)
        cells, obs, null = [], [], []
        for s in STAGES:
            m = st == s
            cnt = int(m.sum())
            mr = (rot == s).astype(float)         # 회전해도 단계 날 수는 같다
            for j, cn in enumerate(A_COLS):
                cells.append((s, cn, cnt))
                obs.append(V[m, j].mean() if cnt else np.nan)
                null.append(mr @ V[:, j] / cnt if cnt else np.full(NSH, np.nan))
        obs, null = np.array(obs), np.array(null).T
        t, p, pf, mu, sd = maxT(obs, null)
        rows = []
        for c_, (s, cn, cnt) in enumerate(cells):
            m = st == s
            j = A_COLS.index(cn)
            rows.append({"stage": s, "stage_name": NAMES[s], "col": cn, "days": cnt,
                         "mean": fnum(obs[c_]), "ci95": boot_mean(V[m, j]) if cnt > 1 else None,
                         "hit_days": fnum((V[m, j] > 0).mean()) if cnt else None,
                         "null_mean": fnum(mu[c_]), "t_shift": fnum(t[c_]), "p_shift": fnum(p[c_]), "p_family": fnum(pf[c_])})
        out[str(y)] = {"days": len(ix), "year_mean": {cn: fnum(V[:, j].mean()) for j, cn in enumerate(A_COLS)},
                       "distinct_shifts": int(len(ix) - 2 * MINSH + 1), "cells": rows}
    return out


# ---------------------------------------------------------------- 표 B·C
def comp_daily(e, col, nd):
    comp = e.groupby("d")[col].apply(lambda r: (np.prod(1 + r.values / 100) - 1) * 100)
    return comp.reindex(range(nd), fill_value=0.0).values.astype(float)


def side_daily(ev3, nd):
    """면마다 Data 날 색인 길이 배열: 하루 손익(곱 − 1, %) r_w90·r_w90_c2·r_follow, 건수."""
    out = {}
    for sd_ in SIDES:
        e = ev3[ev3["side"] == sd_]
        a = {col: comp_daily(e, col, nd) for col in ("r_w90", "r_w90_c2", "r_follow")}
        a["n"] = e.groupby("d").size().reindex(range(nd), fill_value=0).values.astype(int)
        out[sd_] = a
    return out


def chain_overlaps(ev3):
    """면 안 사건 사슬이 겹치는 수(다음 체결 봉 ≤ 앞 청산 봉)."""
    bad = 0
    for _, g in ev3.sort_values(["side", "d", "fill_i"]).groupby(["side", "d"]):
        f, x = g["fill_i"].values, g["exit_i"].values
        bad += int((f[1:] <= x[:-1]).sum())
    return bad


def table_B(ev3, SD, stage_prev, yidx):
    out = {}
    for y in YEARS:
        ix = np.asarray(yidx[y])
        n = len(ix)
        st = stage_prev[ix].astype(int)
        rot = rotated(st)
        e_y = ev3[ev3["d"].isin(set(ix.tolist()))]
        cells, obs, null = [], [], []
        for sd_ in SIDES:
            dly = SD[sd_]["r_w90"][ix]
            for s in STAGES:
                cells.append((sd_, s))
                obs.append(dly[st == s].sum() / n)
                null.append(((rot == s) * dly[None, :]).sum(1) / n)
        obs, null = np.array(obs), np.array(null).T
        t, p, pf, mu, sdv = maxT(obs, null)
        rows = []
        for c_, (sd_, s) in enumerate(cells):
            e = e_y[(e_y["side"] == sd_) & (e_y["stage_prev"] == s)]
            dly = SD[sd_]["r_w90"][ix] * (st == s)
            dirc = (e["r_w90"] - e["r_follow"]) / 2
            ne = len(e)
            rows.append({"side": sd_, "stage": s, "stage_name": NAMES[s], "stage_days": int((st == s).sum()),
                         "trades": int(ne), "trade_days": int(e["d"].nunique()),
                         "trade_mean_r_w90": fnum(e["r_w90"].mean()) if ne else None,
                         "hit_rate": fnum(e["hit"].mean()) if ne else None,
                         "trade_mean_r_follow": fnum(e["r_follow"].mean()) if ne else None,
                         "trade_mean_dir": fnum(dirc.mean()) if ne else None,
                         "trade_mean_r_w90_c2": fnum(e["r_w90_c2"].mean()) if ne else None,
                         "daily_mean": fnum(obs[c_]), "daily_ci95": boot_mean(dly) if ne else None,
                         "mean_per_stage_day": fnum(SD[sd_]["r_w90"][ix][st == s].mean()) if (st == s).any() else None,
                         "null_mean": fnum(mu[c_]), "t_shift": fnum(t[c_]), "p_shift": fnum(p[c_]), "p_family": fnum(pf[c_])})
        out[str(y)] = {"days": n, "distinct_shifts": int(n - 2 * MINSH + 1), "cells": rows,
                       "side_total_daily_mean": {sd_: fnum(SD[sd_]["r_w90"][ix].mean()) for sd_ in SIDES},
                       "side_total_trades": {sd_: int(SD[sd_]["n"][ix].sum()) for sd_ in SIDES}}
    return out


def pick_side(st, hyp):
    """단계 → 그날 쓸 면(0 없음, 1 user, 2 mirror)."""
    st = np.asarray(st)
    if hyp == "trend":
        return np.where(st <= -1, 1, np.where(st >= 1, 2, 0))
    return np.where(st <= -1, 2, np.where(st >= 1, 1, 0))


def c_daily_filtered(ev3, stage_prev, hyp, nd):
    """한 번에 한 종목 규칙을 지켜 그날 고른 사건으로 하루 손익을 낸다(관측용, 건너뛴 수도 센다)."""
    side_of = pick_side(stage_prev, hyp)
    smap = {1: "user", 2: "mirror"}
    e = ev3.copy()
    e["want"] = [smap.get(int(side_of[int(d)]), "") for d in e["d"]]
    e = e[e["side"] == e["want"]].sort_values(["d", "fill_i", "side"])
    keep, skipped = [], 0
    for _, g in e.groupby("d"):
        last_exit = -1
        for k, row in g.iterrows():
            if row["fill_i"] <= last_exit:
                skipped += 1
                continue
            keep.append(k)
            last_exit = row["exit_i"]
    sel = e.loc[keep]
    res = {col: comp_daily(sel, col, nd) for col in ("r_w90", "r_w90_c2", "r_follow")}
    res["n"] = sel.groupby("d").size().reindex(range(nd), fill_value=0).values.astype(int)
    res["side"] = np.array([smap.get(int(v), "") for v in side_of])
    return res, sel, skipped


def c_stats(SD, sel, Cd, stage_prev, hyp, ix, days, halves_def, months):
    n = len(ix)
    st = stage_prev[ix].astype(int)
    dly = Cd["r_w90"][ix]
    obs = dly.mean()
    u, m = SD["user"]["r_w90"][ix], SD["mirror"]["r_w90"][ix]
    ps0 = pick_side(st, hyp)
    re = np.where(ps0 == 1, u, np.where(ps0 == 2, m, 0.0))
    assert np.allclose(re, dly), "필터판과 면 하루 손익이 다르다"
    ps = pick_side(rotated(st), hyp)
    null = np.where(ps == 1, u[None, :], np.where(ps == 2, m[None, :], 0.0)).mean(1)
    p = (1 + (null >= obs - 1e-12).sum()) / (NSH + 1)
    allk = np.arange(MINSH, n - MINSH + 1)
    R = (np.arange(n)[None, :] - allk[:, None]) % n
    psa = pick_side(st[R], hyp)
    nulla = np.where(psa == 1, u[None, :], np.where(psa == 2, m[None, :], 0.0)).mean(1)
    p_exact = float((nulla >= obs - 1e-12).mean())
    s_y = sel[sel["d"].isin(set(ix.tolist()))]
    c2 = Cd["r_w90_c2"][ix]
    fol = Cd["r_follow"][ix]
    dirc = (dly - fol) / 2
    halves = {}
    for nm, (a, b) in halves_def.items():
        mk = np.array([(a <= d.month <= b) for d in days])
        halves[nm] = {"days": int(mk.sum()), "daily_mean": fnum(dly[mk].mean()) if mk.any() else None,
                      "cost2x_daily_mean": fnum(c2[mk].mean()) if mk.any() else None,
                      "trades": int(Cd["n"][ix][mk].sum())}
    msum = pd.Series(dly, index=months).groupby(level=0).sum()
    best = msum.idxmax()
    keepm = np.array([mm != best for mm in months])
    top5 = np.sort(np.argsort(-dly, kind="stable")[:5])
    keep5 = np.ones(n, bool)
    keep5[top5] = False
    down, up = st <= -1, st >= 1
    return {"days": n, "trades": int(Cd["n"][ix].sum()), "active_days": int((Cd["n"][ix] > 0).sum()),
            "daily_mean": fnum(obs), "daily_ci95": boot_mean(dly),
            "trade_mean": fnum(s_y["r_w90"].mean()) if len(s_y) else None,
            "hit_rate": fnum(s_y["hit"].mean()) if len(s_y) else None,
            "p_shift_one_sided": fnum(p), "p_shift_exact_all_rotations": fnum(p_exact),
            "distinct_shifts": int(len(allk)),
            "null_mean": fnum(null.mean()), "null_q95": fnum(np.percentile(null, 95)),
            "halves": halves,
            "cost2x_daily_mean": fnum(c2.mean()), "cost2x_ci95": boot_mean(c2),
            "follow_daily_mean": fnum(fol.mean()),
            "dir_component_daily_mean": fnum(dirc.mean()), "dir_component_ci95": boot_mean(dirc),
            "dir_component_trade_mean": fnum(((s_y["r_w90"] - s_y["r_follow"]) / 2).mean()) if len(s_y) else None,
            "contrib_down_days": {"days": int(down.sum()), "trades": int(Cd["n"][ix][down].sum()),
                                  "share_of_daily_mean": fnum(dly[down].sum() / n)},
            "contrib_up_days": {"days": int(up.sum()), "trades": int(Cd["n"][ix][up].sum()),
                                "share_of_daily_mean": fnum(dly[up].sum() / n)},
            "by_month_sum": {k: fnum(v) for k, v in msum.items()},
            "drop_best_month": {"month": best, "daily_mean": fnum(dly[keepm].mean()), "days": int(keepm.sum())},
            "drop_top5_days": {"days_dropped": [str(days[i].date()) for i in top5],
                               "values": [fnum(dly[i]) for i in top5],
                               "daily_mean": fnum(dly[keep5].mean()), "days": int(keep5.sum())},
            "worst_day": fnum(dly.min()), "best_day": fnum(dly.max())}


def holm(ps):
    items = sorted(ps.items(), key=lambda kv: kv[1])
    mm, run, out = len(items), 0.0, {}
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (mm - j) * p))
        out[k] = run
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    # 1. 단계
    c = soxx_daily()
    dates = pd.DatetimeIndex(c.index)
    cv = c.to_numpy(float)
    zn, r, z = zone_calc(cv)
    log(f"SOXX {dates[0].date()}~{dates[-1].date()} {len(cv)}봉")
    chk = stage_check(cv, dates, zn, r, z)
    log("대조", chk)
    stg = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": cv, "rsi14": r, "z20": z, "stage": zn,
                        "stage_name": [NAMES.get(int(v), "") if np.isfinite(v) else "" for v in zn]})
    stg.to_csv(os.path.join(RES, "stage5_stages.csv"), index=False, encoding="utf-8-sig")

    # 2. 자료·D−1 단계
    D = Data("alp1m_long")
    days = [pd.Timestamp(d) for d in D.days]
    nd = len(days)
    log(f"Data {days[0].date()}~{days[-1].date()} {nd}일")
    pos = np.searchsorted(dates.values, np.array(days, dtype="datetime64[ns]"), side="left") - 1
    assert (pos >= 0).all()
    prev_date = dates[pos]
    stage_prev = zn[pos]
    assert np.isfinite(stage_prev).all()
    yset, dset_ = set(dates[dates >= days[0]]), set(days)
    data_chk = {"data_days_missing_in_yf": [str(d.date()) for d in sorted(dset_ - yset)],
                "yf_days_missing_in_data": [str(d.date()) for d in sorted(yset - dset_) if d <= days[-1]],
                "prev_date_equals_data_prev_day": int(sum(prev_date[k] == days[k - 1] for k in range(1, nd))),
                "n_checked": nd - 1}
    al = D.daily.loc["SOXX"]["close"].values
    yc = c.reindex(days).values
    rel = np.abs(yc / al - 1) * 100
    data_chk["yf_vs_alpaca_close_abs_pct"] = {"median": fnum(np.nanmedian(rel)), "p99": fnum(np.nanpercentile(rel, 99)),
                                             "max": fnum(np.nanmax(rel))}
    log("날짜 대조", {k: (v if not isinstance(v, list) else len(v)) for k, v in data_chk.items()})
    yidx = {y: np.array([k for k, d in enumerate(days) if d.year == y]) for y in YEARS}

    stage_counts = {}
    for y in YEARS:
        st = stage_prev[yidx[y]].astype(int)
        stage_counts[str(y)] = {"days": int(len(st)), "runs_total": runs(st),
                                "by_stage": {NAMES[s]: {"days": int((st == s).sum()),
                                                        "runs": int(sum(1 for k in range(len(st)) if st[k] == s and (k == 0 or st[k - 1] != s)))}
                                             for s in STAGES}}
    st26 = stage_prev[yidx[2026]].astype(int)
    months26 = [f"{days[k].year}-{days[k].month:02d}" for k in yidx[2026]]
    stage_counts["2026"]["by_month"] = {mo: {NAMES[s]: int(sum(1 for mm, v in zip(months26, st26) if mm == mo and v == s)) for s in STAGES}
                                        for mo in sorted(set(months26))}
    log("2026 단계", stage_counts["2026"]["by_stage"], "구간", stage_counts["2026"]["runs_total"])

    # 3. 표 A — 엔진
    allk = list(range(nd))
    dA = pd.DataFrame({"day": [d.strftime("%Y-%m-%d") for d in days], "year": [d.year for d in days],
                       "prev_date": prev_date.strftime("%Y-%m-%d"), "stage_prev": stage_prev})
    eng_chk = {}
    for sch in ("w90", "std"):
        E = Engine(D, scheme=sch)
        for nm, p_ in (("SOXL", 1), ("SOXS", -1)):
            daily, tr = E.run(lambda p_=p_: Always(p_), days=allk)
            assert len(daily) == nd and (daily["n"] == 1).all(), (sch, nm, daily["n"].value_counts().to_dict())
            dA[f"{nm}_{sch}"] = daily["ret"].values * 100
            eng_chk[f"{nm}_{sch}"] = {"entry": sorted(tr["entry"].unique().tolist()), "exit": sorted(tr["exit"].unique().tolist()),
                                      "why": tr["why"].value_counts().to_dict()}
    log("엔진", eng_chk)
    dA["stage_name"] = [NAMES[int(v)] for v in dA["stage_prev"]]
    dA.to_csv(os.path.join(RES, "stage5_daily_A.csv"), index=False, encoding="utf-8-sig")
    A = table_A(dA, yidx)
    log("표 A 끝")

    # 4. 표 B
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    ev3 = ev[np.isclose(ev["x"], X)].copy()
    ev3["side"] = ev3["side"].astype(str)
    ev3["d"] = ev3["d"].astype(int)
    assert all(days[d] == pd.Timestamp(t) for d, t in zip(ev3["d"], ev3["day"]))
    ev3["stage_prev"] = stage_prev[ev3["d"].values].astype(int)
    bad = chain_overlaps(ev3)
    SD = side_daily(ev3, nd)
    B = table_B(ev3, SD, stage_prev, yidx)
    dB = pd.DataFrame({"day": dA["day"], "year": dA["year"], "stage_prev": stage_prev, "stage_name": dA["stage_name"]})
    for sd_ in SIDES:
        for col in ("n", "r_w90", "r_w90_c2", "r_follow"):
            dB[f"{sd_}_{col}"] = SD[sd_][col]
    dB.to_csv(os.path.join(RES, "stage5_daily_B.csv"), index=False, encoding="utf-8-sig")
    log("표 B 끝, 면 안 겹침", bad, "x 3% 사건", len(ev3))

    # 5. C
    C, Cd_all, skipped = {}, {}, {}
    for hyp in ("trend", "counter"):
        Cd, sel, sk = c_daily_filtered(ev3, stage_prev, hyp, nd)
        Cd_all[hyp], skipped[hyp] = Cd, sk
        res = {}
        for y in YEARS:
            ix = yidx[y]
            dd = [days[k] for k in ix]
            months = [f"{d.year}-{d.month:02d}" for d in dd]
            halves = {"jan_jun": (1, 6), "jul_oct": (7, 10)} if y == 2026 else {"jan_jun": (1, 6), "jul_dec": (7, 12)}
            res[str(y)] = c_stats(SD, sel, Cd, stage_prev, hyp, ix, dd, halves, months)
        C[hyp] = res
    praw = {h: C[h]["2026"]["p_shift_one_sided"] for h in C}
    ph = holm(praw)
    for h in C:
        r26 = C[h]["2026"]
        crit = {"(1) mean>0": bool(r26["daily_mean"] > 0),
                "(2) holm_p<=0.05": bool(ph[h] <= 0.05),
                "(3) both_halves>0": bool(all(v["daily_mean"] is not None and v["daily_mean"] > 0 for v in r26["halves"].values())),
                "(4) cost2x>0": bool(r26["cost2x_daily_mean"] > 0)}
        C[h]["judgement_2026"] = {"p_raw": praw[h], "p_holm_m2": ph[h], "criteria": crit, "pass": all(crit.values()),
                                  "skipped_overlap_events": skipped[h]}
        for y in (2023, 2024, 2025):
            ry = C[h][str(y)]
            ry["same_sign_as_2026"] = bool(np.sign(ry["daily_mean"]) == np.sign(r26["daily_mean"]))
    dC = pd.DataFrame({"day": dA["day"], "year": dA["year"], "stage_prev": stage_prev, "stage_name": dA["stage_name"]})
    for h in ("trend", "counter"):
        Cd = Cd_all[h]
        dC[f"{h}_side"] = Cd["side"]
        for col in ("n", "r_w90", "r_w90_c2", "r_follow"):
            dC[f"{h}_{col}"] = Cd[col]
    dC.to_csv(os.path.join(RES, "stage5_daily_C.csv"), index=False, encoding="utf-8-sig")
    log("C", {h: (C[h]["2026"]["daily_mean"], C[h]["judgement_2026"]) for h in C})

    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    out = {"brief": "BRIEF5.md", "impl": "A (stage5.py)", "code_sha256_16": code_sha,
           "units": "% (1.0 = 1%)",
           "settings": {"seed": SEED, "boot": NB, "shifts": NSH, "min_shift": MINSH, "x": X,
                        "stage": "RSI14(Wilder) ≥70 +2, ≤30 −2, z20 ≥0.5 +1, ≤−0.5 −1, else 0; 첫 130봉 없음; D일 = D−1 종가 단계",
                        "soxx": "yfinance SOXX 2021-01-01~2026-10-28 auto_adjust=False Close",
                        "data": f"Data('alp1m_long') {days[0].date()}~{days[-1].date()} {nd}일"},
           "stage_check": chk, "date_check": data_chk, "engine_check": eng_chk,
           "event_chain_overlaps_within_side": bad,
           "stage_counts": stage_counts, "tableA": A, "tableB": B, "C": C,
           "files": ["stage5_stages.csv", "stage5_daily_A.csv", "stage5_daily_B.csv", "stage5_daily_C.csv"]}
    json.dump(out, open(os.path.join(RES, "stage5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    log("끝")


if __name__ == "__main__":
    main()
