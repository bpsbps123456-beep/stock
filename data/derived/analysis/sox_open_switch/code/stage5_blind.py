"""BRIEF5 눈가림 구현 B (2026-11-03). SOXX 추세 5단계별 장초 방향(A)·사용자 규칙(B)·주 가설(C).

stage5.py·lib.py는 쓰지 않는다. 단계 정의는 ../../trend_zones/trend_5zone.pine의 zoneCalc()와 BRIEF5만 보고 짰다.
자료는 사건 표 results/ev4_events.parquet, 1분봉 cache/alp1m_long/*.parquet, 반 호가 cost_table.json을 직접 읽는다.

정의(이 구현이 정한 것)
- 단계: SOXX 일봉(yfinance 2021-01-01~2026-10-28, auto_adjust=False Close = 분할만 반영).
  lc = ln C, lr = lc − lc[1], sd = lr 20개 표본 표준편차, k = √((N−1)(2N−1)/(6N)), z = (lc − SMA20(lc)) / (sd·k),
  RSI14 = Wilder(첫 14개 변화 단순평균으로 시작, 그 뒤 1/14 지수평활). RSI ≥ 70 → +2, ≤ 30 → −2, z ≥ 0.5 → +1,
  z ≤ −0.5 → −1, 나머지 0. bar_index < 129(파인 그대로)는 값 없음. D일 매매에는 D−1 거래일 종가의 단계를 쓴다.
- 표 A: 09:30 봉 마감에 결정 → 09:31 봉 종가 × (1 + 반 호가 open5)에 산다. w90은 10:59 봉 종가 × (1 − 반 호가 late),
  std는 09:59 봉 종가 × (1 − 반 호가 early)에 판다. 반 호가 시간대는 체결 봉 시작 시각 기준.
  봉이 비면 같은 날 09:30 이후 그 시각 이전의 마지막 봉을 쓴다(개수 보고).
- 표 B: 사건 표 x = 3%, r_w90. 칸 = 단계 × 면. 하루 평균은 그 단계 날 전부로 나눈다(사건 없는 날 0).
  하루 손익 = 그날 그 면 사건 (1 + r/100)의 곱 − 1(BRIEF3·4 '하루 평균'과 같은 정의).
  [2026-11-03 A 대조 후 고침] 처음에는 같은 날 사건을 단순 합했다(같은 면 같은 날 2~5건인 날 234개에서 A와 어긋남).
- 표 C: 하방·극하방 날은 사용자 면(SOXL 오름 → SOXS)만, 상방·극상방 날은 거울 면(SOXS 오름 → SOXL)만, 애매 날은 쉼.
  반대 가설은 면을 맞바꾼다. 하루에 한 면만 쓰고 한 면 안 사건은 앞 사건 청산 뒤에만 생기므로 한 번에 한 종목이다(검사함).
- 귀무(원형 이동): 그해 거래일 단계 열을 s ∈ [20, n−20]일 원형으로 민다. 해마다 seed 20261103으로 5,000번.
  [2026-11-03 A 대조 후 고침] 처음에는 2023~2025를 seed [20261103, 해]로 뽑았다. BRIEF5가 적은 seed는 20261103 하나다.
  C는 단측 p = (1 + #{m_s ≥ m_obs}) / (1 + B). A·B 칸은 중심 t = (칸 평균 − 그 열 전체 평균) / (칸 표준편차/√칸 날 수)의
  양측 p, 가족 보정은 매 이동의 표 안 최대 |t|.
실행: cd code && py stage5_blind.py → results/stage5_blind.json, stage5_blind_daily.csv, stage5_blind_soxx_stage.csv
"""
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ["SOS_UNLOCK"] = "frozen"  # BRIEF5 허용(이 구현은 lib를 import하지 않으므로 실제로 쓰이지 않는다)

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
CACHE = os.path.join(BASE, "cache", "alp1m_long")
RES = os.path.join(BASE, "results")
TZ = "America/New_York"
SEED = 20261103
NSHIFT = 5000
NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}
STAGES = [2, 1, 0, -1, -2]
YEARS = [2026, 2025, 2024, 2023]
SOXX_CSV = os.path.join(RES, "stage5_blind_soxx_1d.csv")


# ------------------------------------------------------------------ 1. SOXX 단계
def soxx_close():
    try:
        import yfinance as yf
        df = yf.download("SOXX", start="2021-01-01", end="2026-10-29", auto_adjust=False, progress=False, actions=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna().astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s.name = "Close"
        s.to_frame().to_csv(SOXX_CSV, index_label="date")
        src = "yfinance(새로 받음)"
    except Exception as e:  # 받기 실패 때만 앞서 저장한 사본
        s = pd.read_csv(SOXX_CSV, index_col=0, parse_dates=True)["Close"]
        src = f"저장 사본({type(e).__name__})"
    return s.sort_index(), src


def rsi_wilder(c, n=14):
    d = np.diff(c)
    up, dn = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    au, ad = np.full(len(c), np.nan), np.full(len(c), np.nan)
    au[n], ad[n] = up[:n].mean(), dn[:n].mean()
    for i in range(n + 1, len(c)):
        au[i] = (au[i - 1] * (n - 1) + up[i - 1]) / n
        ad[i] = (ad[i - 1] * (n - 1) + dn[i - 1]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + au / ad)
    r = np.where(ad == 0, 100.0, np.where(au == 0, 0.0, r))
    return np.where(np.isfinite(au) & np.isfinite(ad), r, np.nan)


def stage_series(close: pd.Series, N=20, th=0.5):
    c = close.to_numpy(float)
    lc = np.log(c)
    lr = np.r_[np.nan, np.diff(lc)]
    sd = pd.Series(lr).rolling(N, min_periods=N).std(ddof=1).to_numpy()
    ma = pd.Series(lc).rolling(N, min_periods=N).mean().to_numpy()
    k = np.sqrt((N - 1.0) * (2.0 * N - 1.0) / (6.0 * N))
    z = (lc - ma) / (sd * k)
    r = rsi_wilder(c, 14)
    st = np.select([r >= 70, r <= 30, z >= th, z <= -th], [2, -2, 1, -1], 0).astype(float)
    ok = np.isfinite(z) & np.isfinite(r) & (np.arange(len(c)) >= 129)
    st = np.where(ok, st, np.nan)
    return pd.DataFrame({"close": c, "rsi": r, "z": z, "stage": st}, index=close.index)


def weekly_warn(close: pd.Series):
    """SOXX 주간 경고(사후 표시용): 이번 주 첫 거래일 직전 종가가 20일선 아래면 그 주 내내."""
    below = close < close.rolling(20, min_periods=20).mean()
    wk = close.index - pd.to_timedelta(close.index.weekday, unit="D")
    out, cur, prev_below = {}, None, None
    for d, w, b in zip(close.index, wk, below):
        if w != cur:
            cur, state = w, bool(prev_below) if prev_below is not None else False
        out[d] = state
        prev_below = b
    return pd.Series(out)


# ------------------------------------------------------------------ 2. 표 A 원자료
def load_min(sym):
    d = pd.read_parquet(os.path.join(CACHE, sym + ".parquet"))
    idx = d.index.tz_convert(TZ)
    d = d.assign(date=idx.tz_localize(None).normalize(), hm=idx.hour * 100 + idx.minute)
    return d


def bar_close_at(d, T):
    sub = d[(d.hm >= 930) & (d.hm <= T)]
    last = sub.groupby("date").tail(1).set_index("date")
    return last["Close"], last["hm"] == T


def table_a_raw(days, cost):
    out, miss = {}, {}
    for sym in ("SOXL", "SOXS"):
        d = load_min(sym)
        hs = cost[sym]
        px = {}
        for T in (931, 959, 1059):
            c, exact = bar_close_at(d, T)
            px[T] = c.reindex(days)
            miss[f"{sym}_{T}"] = int((~exact.reindex(days).fillna(False).astype(bool)).sum())
        for mult, tag in ((1.0, ""), (2.0, "_c2")):
            buy = px[931] * (1 + mult * hs["open5"])
            out[f"{sym}_w90{tag}"] = (px[1059] * (1 - mult * hs["late"]) / buy - 1) * 100
            out[f"{sym}_std{tag}"] = (px[959] * (1 - mult * hs["early"]) / buy - 1) * 100
    return pd.DataFrame(out, index=days), miss


# ------------------------------------------------------------------ 통계 도우미
def shift_matrix(n, rng, B=NSHIFT):
    s = rng.integers(20, n - 20 + 1, size=B)
    return (np.arange(n)[None, :] - s[:, None]) % n


def centered_t(y, mask):
    """(칸 평균 − 열 전체 평균) / (칸 표준편차/√n). mask: (..., n)."""
    m = mask.astype(float)
    nk = m.sum(-1)
    s1 = m @ y
    s2 = m @ (y * y)
    mean = s1 / nk
    var = (s2 - nk * mean ** 2) / (nk - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean - y.mean()) / np.sqrt(var / nk)
    return np.where(np.isfinite(t), t, 0.0)


def cell_tests(series: dict, st, idxm):
    """series: {이름: (n,) 날별 값}. 칸 = (이름, 단계). 반환: {칸: (t, p, p_fam)}."""
    cells, t_obs, t_null = [], [], []
    st_sh = st[idxm]
    for name, y in series.items():
        for k in STAGES:
            if (st == k).sum() < 2:
                continue
            cells.append((name, k))
            t_obs.append(centered_t(y, st == k))
            t_null.append(centered_t(y, st_sh == k))
    t_obs, t_null = np.array(t_obs), np.array(t_null)  # (C,), (C, B)
    B = t_null.shape[1]
    mx = np.abs(t_null).max(0)
    res = {}
    for i, c in enumerate(cells):
        p = (1 + (np.abs(t_null[i]) >= abs(t_obs[i])).sum()) / (1 + B)
        pf = (1 + (mx >= abs(t_obs[i])).sum()) / (1 + B)
        res[c] = (float(t_obs[i]), float(p), float(pf))
    return res


def holm(ps):
    order = np.argsort(ps)
    m, adj, run = len(ps), np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def runs(st):
    out = {NAMES[k]: 0 for k in STAGES}
    prev = None
    for v in st:
        if v != prev:
            out[NAMES[int(v)]] += 1
        prev = v
    return out


# ------------------------------------------------------------------ 본 계산
def main():
    cost = json.load(open(os.path.join(BASE, "cost_table.json"), encoding="utf-8"))
    close, src = soxx_close()
    S = stage_series(close)
    S.assign(stage_name=S.stage.map(lambda v: NAMES.get(int(v)) if np.isfinite(v) else "")).to_csv(
        os.path.join(RES, "stage5_blind_soxx_stage.csv"), index_label="date", float_format="%.6f")

    # 거래일(1분봉 정규장이 있는 날) — D일 단계 = D−1 일봉 단계
    soxl = load_min("SOXL")
    days = pd.DatetimeIndex(sorted(soxl.loc[(soxl.hm >= 930) & (soxl.hm < 1600), "date"].unique())).astype("datetime64[ns]")
    del soxl
    pos = S.index.get_indexer(days)
    not_in_daily = [str(d.date()) for d, p in zip(days, pos) if p < 0]
    assert not not_in_daily, f"일봉에 없는 거래일: {not_in_daily[:5]}"
    prev = S.iloc[pos - 1]
    D = pd.DataFrame({"prev_date": prev.index, "stage": prev.stage.to_numpy(), "rsi_prev": prev.rsi.to_numpy(),
                      "z_prev": prev.z.to_numpy()}, index=days)
    assert D.stage.notna().all()
    D["stage"] = D.stage.astype(int)
    D["year"] = D.index.year
    warn = weekly_warn(close)
    D["warn"] = warn.reindex(days).to_numpy()

    # 표 A 원자료
    A, miss = table_a_raw(days, cost)
    D = D.join(A)

    # 사건(x = 3%)
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    e = ev[np.isclose(ev.x, 0.03)].copy()
    e["day"] = pd.to_datetime(e.day).dt.normalize().astype("datetime64[ns]")
    e["side"] = e.side.astype(str)
    assert e.day.isin(days).all()
    # 한 면 안 겹침 없음 검사
    overlap = 0
    for (_, _), g in e.sort_values("fill_i").groupby(["day", "side"]):
        f, x = g.fill_i.to_numpy(), g.exit_i.to_numpy()
        overlap += int((f[1:] <= x[:-1]).sum())
    e["dir"] = (e.r_w90 - e.r_follow) / 2
    e["stage"] = D.stage.reindex(e.day).to_numpy()
    def comp(r):  # 하루 손익 = 그날 사건 (1 + r/100)의 곱 − 1 (%)
        return (np.prod(1.0 + r.to_numpy(float) / 100.0) - 1.0) * 100.0

    for sd in ("user", "mirror"):
        g = e[e.side == sd].groupby("day")
        D[f"y_{sd}"] = g.r_w90.apply(comp).reindex(days).fillna(0.0)
        D[f"y_{sd}_c2"] = g.r_w90_c2.apply(comp).reindex(days).fillna(0.0)
        D[f"n_{sd}"] = g.size().reindex(days).fillna(0).astype(int)

    # 표 C 날별
    down, up = D.stage <= -1, D.stage >= 1
    D["C_trend"] = np.where(down, D.y_user, np.where(up, D.y_mirror, 0.0))
    D["C_trend_c2"] = np.where(down, D.y_user_c2, np.where(up, D.y_mirror_c2, 0.0))
    D["C_counter"] = np.where(down, D.y_mirror, np.where(up, D.y_user, 0.0))
    D["C_counter_c2"] = np.where(down, D.y_mirror_c2, np.where(up, D.y_user_c2, 0.0))
    D.to_csv(os.path.join(RES, "stage5_blind_daily.csv"), index_label="date", float_format="%.6f")

    out = {"meta": {"soxx_source": src, "days_total": len(days), "days_by_year": D.year.value_counts().sort_index().to_dict(),
                    "bar_fallback_counts": miss, "events_x3": int(len(e)), "one_side_overlap": overlap,
                    "shifts": NSHIFT, "seed_2026": SEED}}

    # ---- 단계 날 수·구간
    sc = {}
    for y in YEARS:
        Dy = D[D.year == y]
        sc[str(y)] = {"days": {NAMES[k]: int((Dy.stage == k).sum()) for k in STAGES}, "runs": runs(Dy.stage.to_numpy()),
                      "switches": int((Dy.stage.diff().fillna(0) != 0).sum())}
    out["stage_counts"] = sc

    # ---- 지표 이식 대조
    out["port_check"] = port_check(close, S)

    # ---- 해별 이동 행렬
    shifts = {}
    for y in YEARS:
        n = int((D.year == y).sum())
        rng = np.random.default_rng(SEED)
        shifts[y] = shift_matrix(n, rng)

    # ---- 표 A
    A_series = ["SOXL_w90", "SOXS_w90", "SOXL_std", "SOXS_std"]
    tabA = {}
    for y in YEARS:
        Dy = D[D.year == y]
        st = Dy.stage.to_numpy()
        tests = cell_tests({nm: Dy[nm].to_numpy() for nm in A_series}, st, shifts[y]) if y == 2026 else {}
        rows = {}
        for k in STAGES:
            m = Dy.stage == k
            row = {"days": int(m.sum())}
            for nm in A_series:
                v = Dy.loc[m, nm]
                row[nm] = r4(v.mean()) if len(v) else None
                if y == 2026 and len(v):
                    row[nm + "_c2"] = r4(Dy.loc[m, nm + "_c2"].mean())
                    row[nm + "_win"] = r4((v > 0).mean())
                    t, p, pf = tests.get((nm, k), (np.nan, np.nan, np.nan))
                    row[nm + "_t"], row[nm + "_p"], row[nm + "_pfam"] = r4(t), r4(p), r4(pf)
            rows[NAMES[k]] = row
        rows["전체"] = {"days": len(Dy), **{nm: r4(Dy[nm].mean()) for nm in A_series}}
        tabA[str(y)] = rows
    out["tableA"] = tabA

    # ---- 표 B
    tabB = {}
    for y in YEARS:
        Dy = D[D.year == y]
        ey = e[e.day.dt.year == y]
        st = Dy.stage.to_numpy()
        tests = cell_tests({sd: Dy[f"y_{sd}"].to_numpy() for sd in ("user", "mirror")}, st, shifts[y]) if y == 2026 else {}
        rows = {}
        for k in STAGES:
            nd = int((Dy.stage == k).sum())
            for sd in ("user", "mirror"):
                g = ey[(ey.stage == k) & (ey.side == sd)]
                row = {"stage_days": nd, "trades": int(len(g)), "event_days": int(g.day.nunique()),
                       "r_w90": r4(g.r_w90.mean()) if len(g) else None,
                       "hit": r4(g.hit.mean()) if len(g) else None,
                       "r_follow": r4(g.r_follow.mean()) if len(g) else None,
                       "dir": r4(g.dir.mean()) if len(g) else None,
                       "daily": r4(Dy.loc[Dy.stage == k, f"y_{sd}"].sum() / nd) if nd else None}
                if y == 2026:
                    row["daily_c2"] = r4(Dy.loc[Dy.stage == k, f"y_{sd}_c2"].sum() / nd) if nd else None
                    t, p, pf = tests.get((sd, k), (np.nan, np.nan, np.nan))
                    row["t"], row["p"], row["pfam"] = r4(t), r4(p), r4(pf)
                rows[f"{NAMES[k]}|{sd}"] = row
        for sd in ("user", "mirror"):
            g = ey[ey.side == sd]
            rows[f"전체|{sd}"] = {"stage_days": len(Dy), "trades": int(len(g)), "r_w90": r4(g.r_w90.mean()),
                                 "hit": r4(g.hit.mean()), "daily": r4(Dy[f"y_{sd}"].mean())}
        tabB[str(y)] = rows
    out["tableB"] = tabB

    # ---- 표 C
    C = {}
    for hyp in ("C_trend", "C_counter"):
        C[hyp] = {}
        for y in YEARS:
            Dy = D[D.year == y]
            yv = Dy[hyp].to_numpy()
            st = Dy.stage.to_numpy()
            idxm = shifts[y]
            st_sh = st[idxm]
            if hyp == "C_trend":
                null = np.where(st_sh <= -1, Dy.y_user.to_numpy()[None, :],
                                np.where(st_sh >= 1, Dy.y_mirror.to_numpy()[None, :], 0.0)).mean(1)
            else:
                null = np.where(st_sh <= -1, Dy.y_mirror.to_numpy()[None, :],
                                np.where(st_sh >= 1, Dy.y_user.to_numpy()[None, :], 0.0)).mean(1)
            mobs = yv.mean()
            p = (1 + (null >= mobs).sum()) / (1 + len(null))
            side_today = np.where(st <= -1, "user" if hyp == "C_trend" else "mirror",
                                  np.where(st >= 1, "mirror" if hyp == "C_trend" else "user", ""))
            key = pd.DataFrame({"day": Dy.index, "side": side_today})
            trades = e.merge(key, on=["day", "side"])
            h1, h2 = Dy.index.month <= 6, Dy.index.month >= 7
            ser = pd.Series(yv, index=Dy.index)
            mon = ser.groupby(ser.index.to_period("M")).sum()
            best = mon.idxmax()
            row = {"days": len(Dy), "daily": r4(mobs), "p_shift": r4(p), "null_mean": r4(null.mean()),
                   "trades": int(len(trades)), "trade_days": int((yv != 0).sum()),
                   "per_trade": r4(trades.r_w90.mean()) if len(trades) else None,
                   "hit": r4(trades.hit.mean()) if len(trades) else None,
                   "dir": r4(trades.dir.mean()) if len(trades) else None,
                   "h1_JanJun": r4(yv[h1].mean()) if h1.any() else None,
                   "h2_JulOct": r4(yv[h2].mean()) if h2.any() else None,
                   "cost2x": r4(Dy[hyp + "_c2"].mean()),
                   "best_month": str(best), "best_month_sum": r4(mon.max()),
                   "ex_best_month": r4(ser[ser.index.to_period("M") != best].mean()),
                   "ex_top5_days": r4(np.sort(yv)[:-5].mean()),
                   "by_stage_daily": {NAMES[k]: r4(yv[st == k].mean()) if (st == k).any() else None for k in STAGES},
                   "monthly_sum": {str(k): r4(v) for k, v in mon.items()}}
            if y == 2026:
                w = Dy.warn.to_numpy().astype(bool)
                row["posthoc_warn"] = {"warn_days": int(w.sum()), "daily_warn_on": r4(yv[w].mean()) if w.any() else None,
                                       "daily_warn_off": r4(yv[~w].mean()) if (~w).any() else None}
            C[hyp][str(y)] = row
    ps = np.array([C["C_trend"]["2026"]["p_shift"], C["C_counter"]["2026"]["p_shift"]], float)
    adj = holm(ps)
    for i, hyp in enumerate(("C_trend", "C_counter")):
        r = C[hyp]["2026"]
        r["p_holm"] = r4(adj[i])
        crit = {"(1) 평균>0": r["daily"] > 0, "(2) Holm p≤0.05": adj[i] <= 0.05,
                "(3) 두 반>0": (r["h1_JanJun"] or 0) > 0 and (r["h2_JulOct"] or 0) > 0, "(4) 비용2배>0": r["cost2x"] > 0}
        r["criteria"] = {k: bool(v) for k, v in crit.items()}
        r["verdict"] = "통과" if all(crit.values()) else "불통과"
    out["tableC"] = C

    with open(os.path.join(RES, "stage5_blind.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    report(out)


def port_check(close, S):
    """지표 이식 코드와 대조: pine_port.py(옛 정의: 60일 전 한 점, 10-13 전)와 port_k3.py(지금 정의)."""
    res = {}
    try:
        sys.path.insert(0, os.path.join(BASE, "..", "trend_zones", "code"))
        import pine_port as PP
        st_old, r_old, _ = PP.pine_zone(close.to_numpy(float))
        ok = np.isfinite(st_old) & S.stage.notna().to_numpy()
        res["pine_port"] = {"rsi_max_abs_diff": r4(np.nanmax(np.abs(r_old - S.rsi.to_numpy()))),
                            "stage_agree_pct_vs_old_def": r4((st_old[ok] == S.stage.to_numpy()[ok]).mean() * 100),
                            "note": "pine_port.pine_zone은 10-13 전 옛 z(60일 전 한 점 기준)라 단계 자체는 다르다; RSI만 같은 계산"}
    except Exception as ex:
        res["pine_port"] = {"error": repr(ex)}
    try:
        sys.path.insert(0, os.path.join(BASE, "..", "trend_zones", "k3_20261013", "pine"))
        import port_k3 as PK
        st_new, _, r, z3, _ = PK.pine_zone(close.to_numpy(float), 20)
        a, b = st_new, S.stage.to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        res["port_k3"] = {"rows_compared": int(both.sum()), "stage_agree_pct": r4((a[both] == b[both]).mean() * 100),
                          "na_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
                          "rsi_max_abs_diff": r4(np.nanmax(np.abs(r - S.rsi.to_numpy()))),
                          "z_max_abs_diff": float(np.nanmax(np.abs(z3 - S.z.to_numpy())))}
    except Exception as ex:
        res["port_k3"] = {"error": repr(ex)}
    try:
        p = pd.read_parquet(os.path.join(BASE, "..", "trend_zones", "k3_20261013", "panel_1d_20261012.parquet"))
        p = p[p.sym == "SOXX"].copy()
        p.index = pd.to_datetime(p.ts).dt.tz_localize(None).dt.normalize()
        j = pd.concat([p.Close, close], axis=1, keys=["panel", "yf"]).dropna()
        res["close_vs_panel_1d_20261012"] = {"rows": len(j), "max_rel_diff": float((j.panel / j.yf - 1).abs().max())}
    except Exception as ex:
        res["close_vs_panel_1d_20261012"] = {"error": repr(ex)}
    return res


def report(o):
    print(json.dumps(o["meta"], ensure_ascii=False))
    print("port_check", json.dumps(o["port_check"], ensure_ascii=False))
    for y in ("2026",):
        print(y, "단계 날", o["stage_counts"][y])
    print("\n표 A 2026 (하루 평균 %, p, p_fam)")
    for k, r in o["tableA"]["2026"].items():
        print(f"  {k:4} n={r['days']:3}", "  ".join(
            f"{nm} {r.get(nm)} (p {r.get(nm + '_p')}, f {r.get(nm + '_pfam')})" for nm in
            ["SOXL_w90", "SOXS_w90", "SOXL_std", "SOXS_std"]))
    for y in ("2025", "2024", "2023"):
        print(" ", y, {k: (r["days"], r["SOXL_w90"], r["SOXS_w90"], r["SOXL_std"], r["SOXS_std"]) for k, r in o["tableA"][y].items()})
    print("\n표 B 2026")
    for k, r in o["tableB"]["2026"].items():
        print(" ", k, r)
    for y in ("2025", "2024", "2023"):
        print(" ", y, {k: (r["trades"], r["r_w90"], r["daily"]) for k, r in o["tableB"][y].items()})
    print("\n표 C")
    for hyp, d in o["tableC"].items():
        for y, r in d.items():
            print(" ", hyp, y, {k: v for k, v in r.items() if k != "monthly_sum"})


if __name__ == "__main__":
    main()
