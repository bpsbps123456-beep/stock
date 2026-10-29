"""SOXL·SOXS 장초 1분봉 RSI 70 반대 매수 — 구현 A (BRIEF.md 사전 등록 명세 그대로, pandas/numpy 벡터 연산 위주).

32칸(신호 T1·T1F·T2·T3 × RSI 기준 R·X × 체결 E1·E2 × 창 W60·W30) × 반대·추종·B0 를 재고,
주 칸(T1|R|E2|W60) 반대 매수 매매 목록과 판정(날 묶음 부트스트랩·앞뒤 반·날 성격)·보조 5분봉 확인을 낸다.
결과: results/impl_a.json(정본), results/events_a.csv. 사용: py code/impl_a.py
구현 B(code/impl_b.py)와의 대조는 py code/compare_ab.py.
"""
import json
import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(OUT_DIR, "impl_a.json")
OUT_CSV = os.path.join(OUT_DIR, "events_a.csv")

TICKERS = ["SOXL", "SOXS"]
OTHER = {"SOXL": "SOXS", "SOXS": "SOXL"}
RSI_N = 14
LEVEL = 70.0
COST = 0.01            # 진입 +$0.01, 시간 청산 −$0.01, 목표·손절 관통 $0.01
TGT = 0.01             # +1% 목표(역경주는 ±1%)
HOLD = 30              # 보유 봉 수(진입 봉 포함)
OPEN_MIN = 9 * 60 + 30  # 09:30
NBAR = 390             # 정규장 봉 수(09:30~15:59)
WINDOWS = {"W60": 60, "W30": 30}   # 창 안 봉 시작 분(09:30부터) 개수
FILLS = {"E1": 1, "E2": 2}         # 신호 봉 시작 + k분 시가 체결
SIGNALS = ["T1", "T1F", "T2", "T3"]
BASES = ["R", "X"]
MAIN = "T1|R|E2|W60"
HALF_SPLIT = pd.Timestamp("2026-10-13").date()   # 앞 반 9/30~10/13, 뒤 반 10/14~10/28
SEED = 20261029
NBOOT = 10_000
# 보조 5분봉(BRIEF '보조 5분봉'): 5분 RSI(14, 기준 R) T1, 창 09:30~10:25 봉 시작(12봉), 다음 5분봉 시가 + 0.01, 6봉 보유
AUX_STEP = 5
AUX_NWIN = 12
AUX_HOLD = 6
AUX_PERIODS = {"2026-08-31~09-29": ("2026-08-31", "2026-09-29"),
               "2026-09-30~10-28": ("2026-09-30", "2026-10-28")}


# ---------------------------------------------------------------- 자료

def load(sym, iv="1m"):
    d = pd.read_parquet(os.path.join(ROOT, "cache", iv, f"{sym}.parquet"))
    d = d[~d.index.duplicated()].sort_index()
    d.index = d.index.tz_convert("America/New_York")
    return d[["Open", "High", "Low", "Close"]].astype(float)


def minute_of_day(idx):
    return idx.hour * 60 + idx.minute


# ---------------------------------------------------------------- RSI

def wilder_rsi(close, n=RSI_N):
    """Wilder RSI: 처음 n개 변화의 단순평균으로 시작, 이후 RMA(α=1/n). 손실 평균 0이면 100. 앞 n봉은 NaN."""
    c = np.asarray(close, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    dlt = np.diff(c)                      # dlt[i] = 봉 i+1 의 변화
    g = np.clip(dlt, 0, None)
    l = np.clip(-dlt, 0, None)
    ag = pd.Series(np.r_[g[:n].mean(), g[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    al = pd.Series(np.r_[l[:n].mean(), l[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + ag / al)
    out[n:] = np.where(al == 0, 100.0, r)
    return out


def wilder_rsi_loop(close, n=RSI_N):
    """자기 점검용 단순 반복 구현."""
    c = list(map(float, close))
    out = [float("nan")] * len(c)
    if len(c) <= n:
        return np.array(out)
    gs = [max(c[i] - c[i - 1], 0.0) for i in range(1, len(c))]
    ls = [max(c[i - 1] - c[i], 0.0) for i in range(1, len(c))]
    ag, al = sum(gs[:n]) / n, sum(ls[:n]) / n
    for i in range(n, len(c)):
        if i > n:
            ag = (ag * (n - 1) + gs[i - 1]) / n
            al = (al * (n - 1) + ls[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return np.array(out)


def rsi_series(df, base):
    """기준 R: 정규장 봉만 날을 넘어 이어 붙임. 기준 X: 04:00~19:59 모든 봉."""
    mod = minute_of_day(df.index)
    if base == "R":
        m = (mod >= OPEN_MIN) & (mod <= 15 * 60 + 59)
    else:
        m = (mod >= 4 * 60) & (mod <= 19 * 60 + 59)
    s = df.loc[m, "Close"]
    r = wilder_rsi(s.to_numpy())
    chk = wilder_rsi_loop(s.to_numpy())
    assert np.allclose(r, chk, equal_nan=True, atol=1e-9, rtol=0), "RSI 벡터·반복 구현 불일치"
    out = pd.DataFrame({"rsi": r}, index=s.index)
    out["prev"] = out["rsi"].shift(1)     # 같은 기준 계열의 바로 앞 봉
    return out


# ---------------------------------------------------------------- 정규장 격자와 매매 표

def regular_grid(df, days, step=1):
    """(날 수, 390 / step) 격자: 정규장 봉 O/H/L/C, 없는 봉은 NaN. 열 = 09:30부터 봉 번호(step분 봉)."""
    mod = minute_of_day(df.index)
    reg = df[(mod >= OPEN_MIN) & (mod <= 15 * 60 + 59)]
    di = {d: i for i, d in enumerate(days)}
    r = np.array([di.get(d, -1) for d in reg.index.date])
    m = ((minute_of_day(reg.index) - OPEN_MIN) // step).to_numpy()
    keep = r >= 0
    g = {}
    for col in ["Open", "High", "Low", "Close"]:
        a = np.full((len(days), NBAR // step), np.nan)
        a[r[keep], m[keep]] = reg[col].to_numpy()[keep]
        g[col[0]] = a
    return g


def trade_table(g, hold=HOLD):
    """모든 (날, 진입 봉 e)의 hold봉 보유 결과를 한 번에 계산. e = 0..열 수−hold."""
    O, H, L, C = g["O"], g["H"], g["L"], g["C"]
    ne = O.shape[1] - hold + 1
    P = O[:, :ne] + COST
    up_thr = P * (1 + TGT) + COST
    dn_thr = P * (1 - TGT) - COST
    Hw = sliding_window_view(H, hold, axis=1)          # (날, ne, hold)
    Lw = sliding_window_view(L, hold, axis=1)
    up = Hw >= up_thr[..., None]
    dn = Lw <= dn_thr[..., None]
    hit = up.any(-1)
    j_up = np.where(hit, up.argmax(-1), hold)          # 처음 목표 관통 봉(없으면 hold)
    has_dn = dn.any(-1)
    j_dn = np.where(has_dn, dn.argmax(-1), hold)
    exit_off = np.where(hit, j_up, hold - 1)
    last_close = C[:, hold - 1:hold - 1 + ne]          # 마지막 보유 봉(1분이면 진입 + 29분) 종가
    pnl = np.where(hit, TGT * 100.0, ((last_close - COST) / P - 1.0) * 100.0)
    cmin = np.minimum.accumulate(Lw, axis=-1)
    mae = (np.take_along_axis(cmin, exit_off[..., None], -1)[..., 0] / P - 1.0) * 100.0
    race = np.full(P.shape, "none", dtype=object)
    race[(j_up < j_dn)] = "win"
    race[(j_dn < j_up)] = "loss"
    race[(j_up == j_dn) & hit] = "ambig"               # 같은 봉에 +1%·−1% 둘 다
    return {"P": P, "hit": hit, "exit_off": exit_off, "pnl": pnl, "mae": mae, "race": race,
            "open": O[:, :ne]}


# ---------------------------------------------------------------- 신호

def signals_for(rsi_df, days, nwin, step=1):
    """창 안 봉(09:30부터 nwin개 step분 봉)의 T1·T1F·T2·T3 신호.
    반환: {T: DataFrame(d, s)} (d = 날 번호, s = 09:30부터 봉 번호, 1분봉이면 분)."""
    di = {d: i for i, d in enumerate(days)}
    mod = minute_of_day(rsi_df.index) - OPEN_MIN
    w = rsi_df[(mod >= 0) & (mod < nwin * step)].copy()
    w["d"] = [di.get(x, -1) for x in w.index.date]
    w["s"] = ((minute_of_day(w.index) - OPEN_MIN) // step).to_numpy()
    w = w[w["d"] >= 0]
    t1 = (w["prev"] <= LEVEL) & (w["rsi"] > LEVEL)     # NaN 비교는 False → 9/30 예열 구간 신호 없음
    t3 = (w["prev"] > LEVEL) & (w["rsi"] <= LEVEL)
    a = w.loc[t1, ["d", "s"]].sort_values(["d", "s"])
    k = a.groupby("d").cumcount()
    return {"T1": a.reset_index(drop=True),
            "T1F": a[k == 0].reset_index(drop=True),
            "T2": a[k == 1].reset_index(drop=True),
            "T3": w.loc[t3, ["d", "s"]].sort_values(["d", "s"]).reset_index(drop=True)}


# ---------------------------------------------------------------- 매매 실행(겹침 금지)

def run_trades(sig_by_x, tables, grids, k, mode):
    """sig_by_x: {X: DataFrame(d, s)}. mode 'fade' → Y = 반대 종목, 'follow' → Y = X.
    같은 날 같은 매수 종목 Y에 열린 매매가 있으면(새 진입 분 < 앞 청산 봉 시작 + 1분) 건너뛰고 센다."""
    rows = []
    for X, sg in sig_by_x.items():
        if len(sg):
            t = sg.copy()
            t["X"] = X
            rows.append(t)
    if not rows:
        return pd.DataFrame(columns=["d", "s", "X", "Y", "e", "P", "hit", "exit_off", "pnl", "mae", "race"]), 0, 0
    sig = pd.concat(rows, ignore_index=True)
    sig["Y"] = sig["X"].map(OTHER) if mode == "fade" else sig["X"]
    sig["e"] = sig["s"] + k
    sig = sig.sort_values(["d", "Y", "e", "X"]).reset_index(drop=True)

    recs, skip_ov, skip_nb = [], 0, 0
    for (d, Y), grp in sig.groupby(["d", "Y"], sort=True):
        tb, O = tables[Y], grids[Y]["O"]
        last_exit = -10**9
        for s, X, e in zip(grp["s"], grp["X"], grp["e"]):
            if e >= tb["P"].shape[1] or np.isnan(O[d, e]):
                skip_nb += 1                        # 체결 봉 없음
                continue
            if e < last_exit + 1:
                skip_ov += 1                        # 열린 매매와 겹침
                continue
            last_exit = e + int(tb["exit_off"][d, e])
            recs.append((d, s, X, Y, e, tb["P"][d, e], bool(tb["hit"][d, e]), int(tb["exit_off"][d, e]),
                         float(tb["pnl"][d, e]), float(tb["mae"][d, e]), tb["race"][d, e]))
    tr = pd.DataFrame(recs, columns=["d", "s", "X", "Y", "e", "P", "hit", "exit_off", "pnl", "mae", "race"])
    return tr, skip_ov, skip_nb


def b0_trades(tables, k, nwin, ndays):
    """B0: 창 안 모든 분 s × Y ∈ {SOXL, SOXS}, 같은 체결·보유, 겹침 허용."""
    parts = []
    for Y in TICKERS:
        tb = tables[Y]
        e = np.arange(nwin) + k
        dd, ee = np.meshgrid(np.arange(ndays), e, indexing="ij")
        ok = ~np.isnan(tb["open"][dd, ee])
        parts.append(pd.DataFrame({"d": dd[ok], "e": ee[ok], "Y": Y,
                                   "hit": tb["hit"][dd, ee][ok], "pnl": tb["pnl"][dd, ee][ok],
                                   "race": tb["race"][dd, ee][ok]}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- 집계

def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def trade_stats(tr, skip_ov, skip_nb):
    n = len(tr)
    out = {"n": n, "n_days": int(tr["d"].nunique()) if n else 0,
           "skipped_overlap": int(skip_ov), "skipped_nobar": int(skip_nb)}
    if n == 0:
        out.update({k: None for k in ["hit_rate", "ev_mean_pct", "ev_median_pct", "miss_mean_pct",
                                      "mae_median_pct", "race_win", "race_loss", "race_none"]})
        out["race_ambig_n"] = 0
        return out
    miss = tr.loc[~tr["hit"].astype(bool), "pnl"]
    rc = tr["race"].value_counts()
    out.update({
        "hit_rate": _f(tr["hit"].mean()),
        "ev_mean_pct": _f(tr["pnl"].mean()),
        "ev_median_pct": _f(tr["pnl"].median()),
        "miss_mean_pct": _f(miss.mean()) if len(miss) else None,
        "mae_median_pct": _f(tr["mae"].median()),
        "race_win": _f(rc.get("win", 0) / n),
        "race_loss": _f((rc.get("loss", 0) + rc.get("ambig", 0)) / n),   # 모호는 패로 셈
        "race_ambig_n": int(rc.get("ambig", 0)),
        "race_none": _f(rc.get("none", 0) / n),
    })
    return out


def b0_stats(b):
    n = len(b)
    rc = b["race"].value_counts()
    return {"n": n, "hit_rate": _f(b["hit"].mean()), "ev_mean_pct": _f(b["pnl"].mean()),
            "race_win": _f(rc.get("win", 0) / n),
            "race_loss": _f((rc.get("loss", 0) + rc.get("ambig", 0)) / n)}


def per_day(df, ndays):
    """날별 (건수, 손익 합, 도달 합)."""
    n = np.bincount(df["d"].to_numpy(dtype=int), minlength=ndays).astype(float)
    s = np.bincount(df["d"].to_numpy(dtype=int), weights=df["pnl"].to_numpy(dtype=float), minlength=ndays)
    h = np.bincount(df["d"].to_numpy(dtype=int), weights=df["hit"].to_numpy(dtype=float), minlength=ndays)
    return n, s, h


def bootstrap(fade, follow, b0, ndays):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, ndays, size=(NBOOT, ndays))
    agg = {}
    for name, df in [("fade", fade), ("follow", follow), ("b0", b0)]:
        n, s, h = per_day(df, ndays)
        N = n[idx].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            agg[name] = {"ev": s[idx].sum(1) / N, "hit": h[idx].sum(1) / N}
    stats = {
        "fade_ev_mean_pct": agg["fade"]["ev"],
        "fade_hit_rate": agg["fade"]["hit"],
        "follow_ev_mean_pct": agg["follow"]["ev"],
        "b0_ev_mean_pct": agg["b0"]["ev"],
        "b0_hit_rate": agg["b0"]["hit"],
        "fade_minus_b0_hit": agg["fade"]["hit"] - agg["b0"]["hit"],
        "fade_minus_b0_ev_pct": agg["fade"]["ev"] - agg["b0"]["ev"],
        "fade_minus_follow_hit": agg["fade"]["hit"] - agg["follow"]["hit"],
        "fade_minus_follow_ev_pct": agg["fade"]["ev"] - agg["follow"]["ev"],
    }
    out = {}
    for k, v in stats.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        out[k] = {"lo": _f(lo), "hi": _f(hi), "nan_draws": int(np.isnan(v).sum())}
    return out


def hhmm(minute_from_open):
    m = OPEN_MIN + int(minute_from_open)
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------- 보조 5분봉

def aux_5m():
    """BRIEF '보조 5분봉': 1분 경로와 같은 함수를 봉 번호(5분) 단위로 쓴다.
    겹침 '진입 봉 < 앞 청산 봉 + 1봉'은 5분 격자에서 '진입 시각 < 청산 봉 시작 + 1분'과 같다."""
    data = {t: load(t, "5m") for t in TICKERS}
    dsets = [sorted(set(data[t].index[(minute_of_day(data[t].index) >= OPEN_MIN)
                                      & (minute_of_day(data[t].index) <= 959)].date)) for t in TICKERS]
    assert dsets[0] == dsets[1], "5분봉 SOXL·SOXS 거래일 불일치"
    days = dsets[0]
    grids = {t: regular_grid(data[t], days, step=AUX_STEP) for t in TICKERS}
    for t in TICKERS:   # 창 12봉 + 진입 1봉 뒤 6봉 보유가 닿는 09:30~10:55 봉이 모두 있어야 한다
        assert not np.isnan(grids[t]["C"][:, :AUX_NWIN + AUX_HOLD]).any(), f"{t} 5분봉 장초 봉 누락"
    tables = {t: trade_table(grids[t], hold=AUX_HOLD) for t in TICKERS}
    sig = {t: signals_for(rsi_series(data[t], "R"), days, AUX_NWIN, step=AUX_STEP)["T1"] for t in TICKERS}
    b0_all = b0_trades(tables, 1, AUX_NWIN, len(days))
    out = {"rule": "5분 RSI(14, 기준 R, 정규장 5분봉을 날을 넘어 이음) T1, 창 09:30~10:25 봉 시작(12봉), "
                   "다음 5분봉 시가 + 0.01 진입, 6봉 보유, 같은 목표·청산·비용·겹침 금지, B0 = 창 안 12봉 × 2종목",
           "periods": {}}
    for name, (a, z) in AUX_PERIODS.items():
        a, z = pd.Timestamp(a).date(), pd.Timestamp(z).date()
        idx = [i for i, d in enumerate(days) if a <= d <= z]
        remap = {i: j for j, i in enumerate(idx)}
        sbx = {t: sig[t][sig[t]["d"].isin(idx)] for t in TICKERS}
        fade, fo, fn = run_trades(sbx, tables, grids, 1, "fade")
        foll, lo, ln = run_trades(sbx, tables, grids, 1, "follow")
        b0 = b0_all[b0_all["d"].isin(idx)]
        local = lambda df: df.assign(d=df["d"].map(remap).astype(int))
        out["periods"][name] = {"n_days": len(idx),
                                "fade": trade_stats(fade, fo, fn), "follow": trade_stats(foll, lo, ln),
                                "b0": b0_stats(b0),
                                "ci95": bootstrap(local(fade), local(foll), local(b0), len(idx))}
    return out


# ---------------------------------------------------------------- 본 실행

def main():
    data = {t: load(t) for t in TICKERS}
    soxl_days = sorted(set(data["SOXL"].index[(minute_of_day(data["SOXL"].index) >= OPEN_MIN)
                                              & (minute_of_day(data["SOXL"].index) <= 959)].date))
    soxs_days = sorted(set(data["SOXS"].index[(minute_of_day(data["SOXS"].index) >= OPEN_MIN)
                                              & (minute_of_day(data["SOXS"].index) <= 959)].date))
    assert soxl_days == soxs_days, "SOXL·SOXS 거래일 불일치"
    days = soxl_days
    ndays = len(days)

    grids = {t: regular_grid(data[t], days) for t in TICKERS}
    for t in TICKERS:   # 장초~11:00(보유가 닿는 구간) 봉이 모두 있어야 마지막 봉 종가 청산이 정의된다
        assert not np.isnan(grids[t]["C"][:, :91]).any(), f"{t} 09:30~11:00 봉 누락"
    tables = {t: trade_table(grids[t]) for t in TICKERS}
    rsi = {(b, t): rsi_series(data[t], b) for b in BASES for t in TICKERS}

    cells, keep = {}, {}
    sigs = {(b, w): {t: signals_for(rsi[(b, t)], days, nw) for t in TICKERS}
            for b in BASES for w, nw in WINDOWS.items()}
    b0s = {(e, w): b0_trades(tables, k, nw, ndays) for e, k in FILLS.items() for w, nw in WINDOWS.items()}
    for T in SIGNALS:
        for b in BASES:
            for e, k in FILLS.items():
                for w in WINDOWS:
                    key = f"{T}|{b}|{e}|{w}"
                    sbx = {t: sigs[(b, w)][t][T] for t in TICKERS}
                    fade, fo, fn = run_trades(sbx, tables, grids, k, "fade")
                    foll, lo, ln = run_trades(sbx, tables, grids, k, "follow")
                    cells[key] = {"fade": trade_stats(fade, fo, fn),
                                  "follow": trade_stats(foll, lo, ln),
                                  "b0": b0_stats(b0s[(e, w)])}
                    keep[key] = (fade, foll, b0s[(e, w)])

    # ---- 주 칸 매매 목록
    fade, foll, b0 = keep[MAIN]
    ev = pd.DataFrame({
        "day": [days[d].isoformat() for d in fade["d"]],
        "signal_ticker": fade["X"],
        "signal_time": [hhmm(s) for s in fade["s"]],
        "buy": fade["Y"],
        "entry_time": [hhmm(e) for e in fade["e"]],
        "entry_px": fade["P"].round(4),
        "hit": fade["hit"].astype(int),
        "exit_time": [hhmm(e + x) for e, x in zip(fade["e"], fade["exit_off"])],
        "pnl_pct": fade["pnl"].round(6),
        "mae_pct": fade["mae"].round(6),
    }).sort_values(["day", "signal_time", "signal_ticker"])
    os.makedirs(OUT_DIR, exist_ok=True)
    ev.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # ---- 판정(주 칸)
    boot = bootstrap(fade, foll, b0, ndays)
    m = cells[MAIN]
    c1 = (m["fade"]["ev_mean_pct"] or -1) > 0 and (boot["fade_ev_mean_pct"]["lo"] or -1) > 0
    hit_diff = (m["fade"]["hit_rate"] or 0) - m["b0"]["hit_rate"]
    c2 = hit_diff > 0 and (boot["fade_minus_b0_hit"]["lo"] or -1) > 0
    first = np.array([days[d] <= HALF_SPLIT for d in range(ndays)])
    halves = {}
    for name, mask in [("first_9/30-10/13", first), ("second_10/14-10/28", ~first)]:
        dset = set(np.flatnonzero(mask))
        f = fade[fade["d"].isin(dset)]
        bb = b0[b0["d"].isin(dset)]
        fl = foll[foll["d"].isin(dset)]
        halves[name] = {"n_days": int(mask.sum()), "fade_n": len(f),
                        "fade_ev_mean_pct": _f(f["pnl"].mean()) if len(f) else None,
                        "fade_hit_rate": _f(f["hit"].mean()) if len(f) else None,
                        "follow_ev_mean_pct": _f(fl["pnl"].mean()) if len(fl) else None,
                        "b0_ev_mean_pct": _f(bb["pnl"].mean()), "b0_hit_rate": _f(bb["hit"].mean())}
    c3 = all((h["fade_ev_mean_pct"] or -1) > 0 for h in halves.values())
    if not c1:
        verdict = "불가"
    elif c2 and c3:
        verdict = "후보"
    elif not c2:
        verdict = "변동성 장세 덕"
    else:
        verdict = "1·2 통과, 3 실패 — BRIEF 미정의(후보 아님)"

    # ---- 날 성격(설명용): SOXL 09:30 시가 → 11:00 시가 ÷ 09:30~10:59 고저 폭
    gS = grids["SOXL"]
    rng_ = np.nanmax(gS["H"][:, :90], 1) - np.nanmin(gS["L"][:, :90], 1)
    eff = (gS["O"][:, 90] - gS["O"][:, 0]) / rng_
    label = np.where(np.abs(eff) >= 0.6, "한 방향", "오락가락")
    day_rows = [{"day": days[i].isoformat(), "eff": _f(eff[i]), "type": str(label[i])} for i in range(ndays)]
    by_type = {}
    for lab in ["한 방향", "오락가락"]:
        dset = set(np.flatnonzero(label == lab))
        f = fade[fade["d"].isin(dset)]
        fl = foll[foll["d"].isin(dset)]
        bb = b0[b0["d"].isin(dset)]
        by_type[lab] = {"n_days": len(dset), "fade_n": len(f),
                        "fade_ev_mean_pct": _f(f["pnl"].mean()) if len(f) else None,
                        "fade_hit_rate": _f(f["hit"].mean()) if len(f) else None,
                        "follow_ev_mean_pct": _f(fl["pnl"].mean()) if len(fl) else None,
                        "b0_hit_rate": _f(bb["hit"].mean()), "b0_ev_mean_pct": _f(bb["pnl"].mean())}

    # ---- 나머지 31칸 범위
    others = [k for k in cells if k != MAIN]
    def rng_of(f):
        v = [f(cells[k]) for k in others]
        v = [x for x in v if x is not None]
        return {"min": _f(min(v)), "max": _f(max(v)), "cells_with_value": len(v)}
    other_range = {
        "fade_ev_mean_pct": rng_of(lambda c: c["fade"]["ev_mean_pct"]),
        "fade_hit_rate": rng_of(lambda c: c["fade"]["hit_rate"]),
        "fade_minus_b0_hit": rng_of(lambda c: None if c["fade"]["hit_rate"] is None
                                    else c["fade"]["hit_rate"] - c["b0"]["hit_rate"]),
        "fade_minus_follow_ev_pct": rng_of(lambda c: None if c["fade"]["ev_mean_pct"] is None
                                           or c["follow"]["ev_mean_pct"] is None
                                           else c["fade"]["ev_mean_pct"] - c["follow"]["ev_mean_pct"]),
        "fade_n": rng_of(lambda c: c["fade"]["n"]),
        "cells_fade_ev_pos": sum(1 for k in others if (cells[k]["fade"]["ev_mean_pct"] or -1) > 0),
    }

    out = {
        "cells": cells,
        "events": "results/events_a.csv",
        "extra": {
            "impl": "A (pandas/numpy 벡터)",
            "days": [d.isoformat() for d in days],
            "main_cell": MAIN,
            "judgement": {
                "c1_fade_ev_pos_and_ci_lo_pos": bool(c1),
                "c2_fade_minus_b0_hit_pos_and_ci_lo_pos": bool(c2),
                "c3_both_halves_ev_pos": bool(c3),
                "fade_minus_b0_hit_point": _f(hit_diff),
                "verdict": verdict,
            },
            "bootstrap_main": {"n_boot": NBOOT, "seed": SEED, "rng": "numpy.default_rng.integers(0,21,(10000,21))",
                               "ci95": boot},
            "halves_main": halves,
            "day_type_main": {"rule": "|SOXL(11:00 시가 − 09:30 시가) ÷ (09:30~10:59 고저 폭)| ≥ 0.6 → 한 방향",
                              "by_type": by_type, "days": day_rows},
            "other_31_cells_range": other_range,
            "aux_5m": aux_5m(),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    # ---- 화면 요약
    print("거래일", ndays, days[0], "~", days[-1])
    print(f"{'cell':16s} {'fade n':>6s} {'hit':>6s} {'ev':>7s} | {'foll n':>6s} {'hit':>6s} {'ev':>7s} | {'b0 hit':>6s} {'b0 ev':>7s}")
    for k, c in cells.items():
        fa, fl, bb = c["fade"], c["follow"], c["b0"]
        fmt = lambda x: "  n/a" if x is None else f"{x:6.3f}"
        print(f"{k:16s} {fa['n']:6d} {fmt(fa['hit_rate'])} {fmt(fa['ev_mean_pct']):>7s} | "
              f"{fl['n']:6d} {fmt(fl['hit_rate'])} {fmt(fl['ev_mean_pct']):>7s} | {fmt(bb['hit_rate'])} {fmt(bb['ev_mean_pct']):>7s}")
    print("주 칸", MAIN, json.dumps(cells[MAIN], ensure_ascii=False))
    print("부트스트랩", json.dumps(boot, ensure_ascii=False))
    print("앞뒤 반", json.dumps(halves, ensure_ascii=False))
    print("판정", verdict, "c1", c1, "c2", c2, "c3", c3)
    print("날 성격", json.dumps(by_type, ensure_ascii=False))
    for name, p in out["extra"]["aux_5m"]["periods"].items():
        print("보조 5분봉", name, "fade", p["fade"]["n"], p["fade"]["hit_rate"], p["fade"]["ev_mean_pct"],
              "| b0", p["b0"]["hit_rate"], p["b0"]["ev_mean_pct"], "| CI", p["ci95"]["fade_ev_mean_pct"])


if __name__ == "__main__":
    main()
