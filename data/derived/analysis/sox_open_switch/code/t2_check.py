"""T2 가설 확인(BRIEF2.md '확인' 절 마지막 항목) — 앞 연구(../soxl_rsi_fade) 주 규칙의 T2판을 Alpaca SIP 1분봉으로 잰다.

정의(앞 연구 BRIEF.md·impl_a.py 그대로, 비용만 가격 비례로 바꿈):
- RSI 기준 R: 정규장 봉(09:30~15:59, 반일장은 09:30~12:59)만 날을 넘어 이은 Wilder RSI14(1분 종가). 거래가 있는 봉만 잇는다.
- T2: 그날 그 티커(X)의 창 W60(신호 봉 시작 09:30~10:29) 안 두 번째 70 상향 돌파(RSI[t−1] ≤ 70 < RSI[t]).
- 반대 매수: X가 SOXL이면 SOXS, SOXS면 SOXL(Y)을 산다. 체결 E2 = 신호 봉 시작 + 2분 봉 시가.
- 가격 비례 비용 h: cost_table.json(실측 반 호가, 시간대별)이 있으면 그 값, 없으면 0.03%.
  진입 P = 시가 × (1 + h_진입), 목표 = P × 1.01, 도달 = 봉 고가 ≥ 목표 × (1 + h_그 봉)(관통).
  진입 봉 포함 30봉 안 처음 도달이면 +1.00%(지정가). 못 닿으면 진입 + 29분 봉 종가 × (1 − h_그 봉)에 판다.
  시간대(lib.bucket과 같은 기준): 시가 체결은 그 시각에 끝나는 앞 봉의 시간대, 봉 안 관통·종가 체결은 그 봉의 시간대.
- 겹침 금지: 같은 날 같은 매수 종목에 열린 매매가 있으면 건너뛴다(T2는 날마다 종목당 신호 1개라 실제로 0건).
- 대조: 같은 진입 분 무작위 = 그 기간 모든 날 × 두 종목이 같은 진입 분에 산 결과의 평균(각 T2 매매의 진입 분에 맞춰 평균).
- 날 묶음 부트스트랩: 기간의 모든 날을 복원 추출 10,000회(seed 20261029), 평균 = 뽑힌 날 손익 합 ÷ 매매 수.
기간: 2026-01-02~09-29(주 판정), 2025(보조), 2026-09-30~10-28(앞 연구 구간 재현 대조).
판정: 주 기간 평균 > 0이고 95% 하한 > 0이면 통과. 결과: results/t2_check.json. 사용: py t2_check.py
"""
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import lib as L  # noqa: E402  시간대(bucket)·주문 마감(cutoff_idx)만 빌린다. 체결 엔진·잠금 구간은 쓰지 않는다

PRIOR = os.path.normpath(os.path.join(ROOT, "..", "soxl_rsi_fade", "code"))
OUT = os.path.join(ROOT, "results", "t2_check.json")
TK = ["SOXL", "SOXS"]
OTHER = {"SOXL": "SOXS", "SOXS": "SOXL"}
RSI_N, LEVEL = 14, 70.0
TGT, HOLD, NWIN, K = 0.01, 30, 60, 2          # +1%, 30봉, 창 60분, E2(신호 봉 시작 + 2분 시가)
NB = NWIN + K + HOLD - 1                      # 09:30부터 91봉(09:30~11:00) — 가장 늦은 보유 봉까지
NE = NB - HOLD + 1                            # 진입 분 0..61
FLAT = 0.0003
SEED, NBOOT = 20261029, 10_000
HALF = {date(2024, 12, 24), date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24)}   # 13:00 마감(자료로 확인)
PERIODS = {"main_2026": ("2026-01-02", "2026-09-29"),
           "aux_2025": ("2025-01-02", "2025-12-31"),
           "repl_2026_09_30_10_28": ("2026-09-30", "2026-10-28")}
PRIOR_T2 = {"n": 12, "n_days": 11, "hit_rate": 0.916667, "ev_mean_pct": 0.852288,
            "source": "../soxl_rsi_fade/results/impl_a.json cells['T2|R|E2|W60'].fade (야후, 한쪽 $0.01)"}


def r6(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


# ------------------------------------------------------------------ 자료

def load(sym):
    d = pd.read_parquet(os.path.join(ROOT, "cache", "alp1m", f"{sym}.parquet"))
    d = d[~d.index.duplicated()].sort_index()
    if str(d.index.tz) != "America/New_York":
        d.index = d.index.tz_convert("America/New_York")
    return d[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def mod(ix):
    return ix.hour * 60 + ix.minute


def rth_mask(df):
    m = mod(df.index)
    end = np.where(pd.Index(df.index.date).isin(list(HALF)), 13 * 60, 16 * 60)
    return (m >= 570) & (m < end)


def check_half_days(df):
    """13시대 거래량 ÷ 12시대 거래량이 0.25 미만인 날 = 반일장. HALF와 같아야 한다."""
    m = mod(df.index)
    a = df[(m >= 720) & (m < 780)].groupby(df[(m >= 720) & (m < 780)].index.date)["Volume"].sum()
    b = df[(m >= 780) & (m < 840)].groupby(df[(m >= 780) & (m < 840)].index.date)["Volume"].sum()
    found = set((b / a).dropna().loc[lambda s: s < 0.25].index)
    assert found == HALF, f"반일장 목록 불일치: {sorted(found)}"


# ------------------------------------------------------------------ RSI(앞 연구 impl_a와 같은 식)

def wilder_rsi(c, n=RSI_N):
    c = np.asarray(c, float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    dl = np.diff(c)
    g, l = np.clip(dl, 0, None), np.clip(-dl, 0, None)
    ag = pd.Series(np.r_[g[:n].mean(), g[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    al = pd.Series(np.r_[l[:n].mean(), l[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + ag / al)
    out[n:] = np.where(al == 0, 100.0, r)
    return out


def wilder_rsi_loop(c, n=RSI_N):
    c = list(map(float, c))
    out = [np.nan] * len(c)
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


def rsi_R(df, start=None, ffill=False):
    """기준 R. start를 주면 그날부터 새로 시작(야후 자료처럼 예열 없음). ffill=True면 정규장 빈 분을 앞 종가로 채운 뒤 잇는다(감도)."""
    s = df.loc[rth_mask(df), "Close"]
    if start is not None:
        s = s[s.index >= pd.Timestamp(start, tz="America/New_York")]
    if ffill:
        parts = []
        for dd, g in s.groupby(s.index.date):
            end = "12:59" if dd in HALF else "15:59"
            full = pd.date_range(f"{dd} 09:30", f"{dd} {end}", freq="1min", tz="America/New_York")
            parts.append(g.reindex(full).ffill().bfill())
        s = pd.concat(parts)
    r = wilder_rsi(s.to_numpy())
    assert np.allclose(r, wilder_rsi_loop(s.to_numpy()), equal_nan=True, atol=1e-9, rtol=0), "RSI 벡터·반복 불일치"
    out = pd.DataFrame({"rsi": r}, index=s.index)
    out["prev"] = out["rsi"].shift(1)
    return out


def t2_signals(rsi_df, days):
    """창 안(09:30~10:29 봉 시작) 70 상향 돌파 중 그날 두 번째. 반환 DataFrame(d, s), s = 09:30부터 분."""
    pos = {d: i for i, d in enumerate(days)}
    m = mod(rsi_df.index) - 570
    w = rsi_df[(m >= 0) & (m < NWIN)].copy()
    w["d"] = [pos.get(x, -1) for x in w.index.date]
    w["s"] = (mod(w.index) - 570).to_numpy()
    w = w[w["d"] >= 0]
    t1 = (w["prev"] <= LEVEL) & (w["rsi"] > LEVEL)
    a = w.loc[t1, ["d", "s"]].sort_values(["d", "s"])
    k = a.groupby("d").cumcount()
    return a[k == 1].reset_index(drop=True), a.reset_index(drop=True)


# ------------------------------------------------------------------ 격자·비용·매매 표

def grid(df, days):
    m = mod(df.index) - 570
    sub = df[(m >= 0) & (m < NB)]
    pos = {d: i for i, d in enumerate(days)}
    r = np.array([pos.get(x, -1) for x in sub.index.date])
    c = (mod(sub.index) - 570).to_numpy()
    ok = r >= 0
    g = {}
    for f in ["Open", "High", "Low", "Close"]:
        a = np.full((len(days), NB), np.nan)
        a[r[ok], c[ok]] = sub[f].to_numpy()[ok]
        g[f[0]] = a
    assert not np.isnan(g["C"]).any(), "09:30~11:00 봉 빠짐"
    return g


def load_cost_table():
    p = os.path.join(ROOT, "cost_table.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def cost_model(kind, table=None, mult=1.0):
    """kind: 'rel_table'(실측 표) / 'rel_flat'(0.03%) / 'abs'(한쪽 $0.01, 앞 연구). 반환: 종목별 (진입 분 e의 h, 봉 j의 h)."""
    if kind == "abs":
        return {"kind": "abs", "c": 0.01 * mult}
    out = {"kind": "rel"}
    for s in TK:
        if kind == "rel_table":
            cin = np.array([table[s][L.bucket(L.RTH0 + e - 1)] for e in range(NB)]) * mult   # 시가 = 앞 봉 끝 시각
            cbar = np.array([table[s][L.bucket(L.RTH0 + j)] for j in range(NB)]) * mult
        else:
            cin = cbar = np.full(NB, FLAT * mult)
        out[s] = (cin, cbar)
    return out


def trade_table(g, sym, cm):
    """모든 (날, 진입 분 e=0..NE−1)의 30봉 보유 결과(벡터)."""
    O, H, Lw_, C = g["O"], g["H"], g["L"], g["C"]
    Hw = sliding_window_view(H, HOLD, axis=1)[:, :NE]          # (날, NE, HOLD)
    Lw = sliding_window_view(Lw_, HOLD, axis=1)[:, :NE]
    last = C[:, HOLD - 1:HOLD - 1 + NE]
    if cm["kind"] == "abs":
        P = O[:, :NE] + cm["c"]
        thr = (P * (1 + TGT) + cm["c"])[..., None]
        ex = last - cm["c"]
    else:
        cin, cbar = cm[sym]
        P = O[:, :NE] * (1 + cin[:NE])
        cw = sliding_window_view(cbar, HOLD)[:NE]                # (NE, HOLD)
        thr = (P * (1 + TGT))[..., None] * (1 + cw[None])
        ex = last * (1 - cbar[HOLD - 1:HOLD - 1 + NE])
    up = Hw >= thr
    hit = up.any(-1)
    off = np.where(hit, up.argmax(-1), HOLD - 1)
    pnl = np.where(hit, TGT * 100, (ex / P - 1) * 100)
    cmin = np.minimum.accumulate(Lw, axis=-1)
    mae = (np.take_along_axis(cmin, off[..., None], -1)[..., 0] / P - 1) * 100
    return {"P": P, "hit": hit, "off": off, "pnl": pnl, "mae": mae}


def sim_one(g, sym, d, e, cm, last=None):
    """반복문 모의 체결(표 대조·주문 마감 감도용). last = 마지막 보유 봉(기본 e + 29)."""
    last = e + HOLD - 1 if last is None else last
    O, H, C = g["O"][d], g["H"][d], g["C"][d]
    if cm["kind"] == "abs":
        P = O[e] + cm["c"]
        thr = lambda j: P * (1 + TGT) + cm["c"]
        out = lambda j: C[j] - cm["c"]
    else:
        cin, cbar = cm[sym]
        P = O[e] * (1 + cin[e])
        thr = lambda j: P * (1 + TGT) * (1 + cbar[j])
        out = lambda j: C[j] * (1 - cbar[j])
    for j in range(e, last + 1):
        if H[j] >= thr(j):
            return True, TGT * 100, j
    return False, (out(last) / P - 1) * 100, last


# ------------------------------------------------------------------ 매매

def run_fade(sig, tabs, mode="fade"):
    """sig: {X: DataFrame(d, s)}. 반대(fade)면 Y = 반대 종목, 추종(follow)이면 Y = X."""
    rows = []
    for X, sg in sig.items():
        for d, s in zip(sg["d"], sg["s"]):
            rows.append((int(d), int(s), X, OTHER[X] if mode == "fade" else X, int(s) + K))
    rows.sort(key=lambda r: (r[0], r[3], r[4], r[2]))
    recs, last_exit, skipped = [], {}, 0
    for d, s, X, Y, e in rows:
        if e >= NE:
            skipped += 1
            continue
        if e < last_exit.get((d, Y), -10**9) + 1:
            skipped += 1
            continue
        tb = tabs[Y]
        last_exit[(d, Y)] = e + int(tb["off"][d, e])
        recs.append((d, s, X, Y, e, float(tb["P"][d, e]), bool(tb["hit"][d, e]), int(tb["off"][d, e]),
                     float(tb["pnl"][d, e]), float(tb["mae"][d, e])))
    return pd.DataFrame(recs, columns=["d", "s", "X", "Y", "e", "P", "hit", "off", "pnl", "mae"]), skipped


def hhmm(m):
    m = 570 + int(m)
    return f"{m // 60:02d}:{m % 60:02d}"


def stats_block(tr, dsel, tabs, nboot=NBOOT):
    """dsel: 그 기간 날 번호 목록. 매매 통계 + 같은 진입 분 무작위 + 날 묶음 부트스트랩."""
    nd = len(dsel)
    loc = {d: k for k, d in enumerate(dsel)}
    t = tr[tr["d"].isin(loc)].copy()
    t["ld"] = t["d"].map(loc)
    # 무작위: 이 기간 모든 날 × 두 종목, 진입 분 e = K..K+NWIN−1
    Hd = sum(tabs[y]["hit"][dsel].astype(float) for y in TK)        # (nd, NE)
    Pd = sum(tabs[y]["pnl"][dsel] for y in TK)
    b0_hit_e, b0_pnl_e = Hd.mean(0) / 2, Pd.mean(0) / 2
    ew = slice(K, K + NWIN)
    out = {"n_days_period": nd, "n": int(len(t)), "n_days_with_trade": int(t["d"].nunique()),
           "b0_window": {"n": int(nd * NWIN * 2), "hit_rate": r6(b0_hit_e[ew].mean()), "ev_mean_pct": r6(b0_pnl_e[ew].mean())}}
    if len(t) == 0:
        return out, t
    fe = t["e"].to_numpy(int)
    fh, fp = t["hit"].to_numpy(float), t["pnl"].to_numpy(float)
    miss = fp[fh == 0]
    dsum = t.groupby("ld")["pnl"].sum()
    out.update({
        "hit_rate": r6(fh.mean()), "ev_mean_pct": r6(fp.mean()), "ev_median_pct": r6(np.median(fp)),
        "miss_n": int(len(miss)), "miss_mean_pct": r6(miss.mean()) if len(miss) else None,
        "worst_pct": r6(fp.min()), "mae_median_pct": r6(t["mae"].median()),
        "days_pos": int((dsum > 0).sum()), "days_neg": int((dsum < 0).sum()),
        "days_both_tickers": int((t.groupby("ld")["Y"].nunique() > 1).sum()),
        "by_buy": {y: {"n": int((t["Y"] == y).sum()),
                       "hit_rate": r6(t.loc[t["Y"] == y, "hit"].mean()) if (t["Y"] == y).any() else None,
                       "ev_mean_pct": r6(t.loc[t["Y"] == y, "pnl"].mean()) if (t["Y"] == y).any() else None} for y in TK},
        "matched_b0": {"rule": "각 매매의 진입 분에서 기간 모든 날 × 두 종목 평균",
                       "hit_rate": r6(b0_hit_e[fe].mean()), "ev_mean_pct": r6(b0_pnl_e[fe].mean())},
    })
    # 날 묶음 부트스트랩(기간 모든 날, 매매 없는 날 포함)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nd, size=(nboot, nd))
    cnt = np.zeros((nboot, nd))
    np.add.at(cnt, (np.repeat(np.arange(nboot), nd), idx.ravel()), 1.0)
    ld = t["ld"].to_numpy(int)
    w = cnt[:, ld]                                                  # (B, n)
    N = w.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ev = (w * fp).sum(1) / N
        hr = (w * fh).sum(1) / N
        b0h = (cnt @ Hd) / (2 * cnt.sum(1, keepdims=True))         # 뽑힌 날로 다시 잰 진입 분별 무작위
        b0p = (cnt @ Pd) / (2 * cnt.sum(1, keepdims=True))
        dh = hr - (w * b0h[:, fe]).sum(1) / N
        dp = ev - (w * b0p[:, fe]).sum(1) / N
    pc = lambda v: [r6(x) for x in np.nanpercentile(v, [2.5, 97.5])]
    out["boot"] = {"n_boot": nboot, "seed": SEED, "nan_draws": int(np.isnan(ev).sum()),
                   "ev_mean_ci95": pc(ev), "hit_rate_ci95": pc(hr),
                   "ev_p_le_0": r6(np.nanmean(ev <= 0)),
                   "minus_matched_b0_hit": r6(fh.mean() - b0_hit_e[fe].mean()), "minus_matched_b0_hit_ci95": pc(dh),
                   "minus_matched_b0_ev_pct": r6(fp.mean() - b0_pnl_e[fe].mean()), "minus_matched_b0_ev_ci95": pc(dp)}
    return out, t


def events(t, days):
    return [{"day": days[d].isoformat(), "signal": X, "signal_time": hhmm(s), "buy": Y, "entry": hhmm(e),
             "hit": bool(h), "exit": hhmm(e + o), "pnl_pct": r6(p), "mae_pct": r6(m)}
            for d, s, X, Y, e, h, o, p, m in zip(t["d"], t["s"], t["X"], t["Y"], t["e"], t["hit"], t["off"], t["pnl"], t["mae"])]


def cutoff_variant(t, days, grids, cm):
    """감도(판정 밖): 주문 마감(24:00 KST, lib.cutoff_idx actual) 반영 — 진입 분이 마감 이후면 못 사고, 보유는 마감 봉 종가에서 끊는다."""
    rows = []
    for d, Y, e in zip(t["d"], t["Y"], t["e"]):
        cut = L.cutoff_idx(days[d], "actual") - L.RTH0                 # 09:30부터 분(서머타임 90, 표준시 30)
        if e > cut - 1:
            rows.append((d, None, None))
            continue
        h, p, j = sim_one(grids[Y], Y, d, e, cm, last=min(e + HOLD - 1, cut - 1))
        rows.append((d, h, p))
    ok = [(h, p) for _, h, p in rows if h is not None]
    std_days = sum(1 for d in t["d"] if L.cutoff_idx(days[d], "actual") == L.idx("10:00"))
    return {"rule": "진입 분 ≥ 마감이면 매매 불가, 보유는 min(진입 + 29분, 마감 1분 전) 봉 종가에서 청산",
            "trades_on_std_time_days": int(std_days), "dropped": int(len(rows) - len(ok)), "n": len(ok),
            "hit_rate": r6(np.mean([h for h, _ in ok])) if ok else None,
            "ev_mean_pct": r6(np.mean([p for _, p in ok])) if ok else None}


# ------------------------------------------------------------------ 앞 연구 재현(야후)

def prior_yahoo_t2():
    """앞 연구 impl_a 함수를 고치지 않고 불러 야후 1분봉 T2|R|E2|W60 반대 매수 목록을 만든다."""
    sys.path.insert(0, PRIOR)
    import impl_a as A                                                      # noqa: E402
    data = {t: A.load(t) for t in TK}
    days = sorted(set(data["SOXL"].index[(A.minute_of_day(data["SOXL"].index) >= 570)
                                         & (A.minute_of_day(data["SOXL"].index) <= 959)].date))
    grids = {t: A.regular_grid(data[t], days) for t in TK}
    tabs = {t: A.trade_table(grids[t]) for t in TK}
    sig = {t: A.signals_for(A.rsi_series(data[t], "R"), days, 60)["T2"] for t in TK}
    fade, _, _ = A.run_trades(sig, tabs, grids, 2, "fade")
    return [{"day": days[d].isoformat(), "signal": X, "signal_time": A.hhmm(s), "buy": Y,
             "hit": bool(h), "pnl_pct": r6(p)} for d, s, X, Y, h, p in
            zip(fade["d"], fade["s"], fade["X"], fade["Y"], fade["hit"], fade["pnl"])]


def compare_lists(a, b):
    ka = {(x["day"], x["signal"], x["signal_time"]): x for x in a}
    kb = {(x["day"], x["signal"], x["signal_time"]): x for x in b}
    both = sorted(set(ka) & set(kb))
    return {"n_a": len(a), "n_b": len(b), "matched": len(both),
            "only_a": [list(k) for k in sorted(set(ka) - set(kb))],
            "only_b": [list(k) for k in sorted(set(kb) - set(ka))],
            "hit_disagree": [list(k) for k in both if ka[k]["hit"] != kb[k]["hit"]],
            "pnl_max_abs_diff_pct": r6(max((abs(ka[k]["pnl_pct"] - kb[k]["pnl_pct"]) for k in both), default=0.0))}


# ------------------------------------------------------------------ 본 실행

def main():
    data = {t: load(t) for t in TK}
    check_half_days(data["SOXL"])
    dsets = [sorted(set(data[t].index[rth_mask(data[t])].date)) for t in TK]
    assert dsets[0] == dsets[1], "SOXL·SOXS 거래일 불일치"
    days = dsets[0]
    grids = {t: grid(data[t], days) for t in TK}
    table = load_cost_table()
    prim_kind = "rel_table" if table else "rel_flat"
    cms = {"primary": cost_model(prim_kind, table), "cost_2x": cost_model(prim_kind, table, 2.0),
           "flat_0.03pct": cost_model("rel_flat"), "zero_cost": cost_model("rel_flat", mult=0.0),
           "abs_0.01usd": cost_model("abs")}
    tabs = {k: {t: trade_table(grids[t], t, cm) for t in TK} for k, cm in cms.items()}

    rsi = {t: rsi_R(data[t]) for t in TK}
    sig_all = {t: t2_signals(rsi[t], days) for t in TK}
    sig = {t: sig_all[t][0] for t in TK}
    rsi_ff = {t: rsi_R(data[t], ffill=True) for t in TK}
    sig_ff = {t: t2_signals(rsi_ff[t], days)[0] for t in TK}
    ff_diff = {t: int(len(set(map(tuple, sig[t].values)) ^ set(map(tuple, sig_ff[t].values)))) for t in TK}

    trades = {k: run_fade(sig, tabs[k]) for k in cms}
    follow, _ = run_fade(sig, tabs["primary"], "follow")
    assert all(v[1] == 0 for v in trades.values()), "T2 겹침·봉 없음 건너뜀이 생김"

    # 표(벡터) ↔ 반복문 모의 체결 대조(모든 비용 모형, 모든 매매)
    nchk = 0
    for k, cm in cms.items():
        for d, Y, e, h, p in zip(*(trades[k][0][c] for c in ["d", "Y", "e", "hit", "pnl"])):
            h2, p2, _ = sim_one(grids[Y], Y, d, e, cm)
            assert h2 == h and abs(p2 - p) < 1e-9, (k, d, Y, e)
            nchk += 1

    out = {"rule": {"signal": "T2 = 창(신호 봉 09:30~10:29) 안 그날 그 티커의 두 번째 RSI14 70 상향 돌파, 기준 R(정규장 봉만 날을 넘어 이음, 반일장 12:59까지)",
                    "trade": "반대 종목을 신호 봉 시작 + 2분 시가에 매수, +1% 지정가(관통), 진입 봉 포함 30봉 안 못 닿으면 29분 뒤 봉 종가 매도",
                    "cost": "primary = " + ("cost_table.json 시간대별 실측 반 호가" if table else "cost_table.json 없음 → 0.03% 고정"),
                    "cost_table": table, "data": "Alpaca SIP 1분봉(분할 반영) cache/alp1m", "half_days": sorted(d.isoformat() for d in HALF),
                    "bootstrap": f"날 묶음(기간 모든 날) {NBOOT}회, numpy default_rng({SEED}).integers(0, nd, ({NBOOT}, nd))"},
           "checks": {"vector_vs_loop_trades_checked": nchk, "rsi_vector_vs_loop": "일치(assert)",
                      "t2_signal_diff_if_rth_gaps_ffilled": ff_diff,
                      "rth_days": len(days), "first_day": days[0].isoformat(), "last_day": days[-1].isoformat()},
           "periods": {}}
    for pname, (a, b) in PERIODS.items():
        a_, b_ = pd.Timestamp(a).date(), pd.Timestamp(b).date()
        dsel = [i for i, d in enumerate(days) if a_ <= d <= b_]
        blk, t = stats_block(trades["primary"][0], dsel, tabs["primary"])
        sens = {}
        for k in ["cost_2x", "flat_0.03pct", "zero_cost"] + (["abs_0.01usd"] if pname.startswith("repl") else []):
            s_, _ = stats_block(trades[k][0], dsel, tabs[k], nboot=NBOOT)
            sens[k] = {x: s_.get(x) for x in ["n", "hit_rate", "ev_mean_pct"]}
            sens[k]["ev_mean_ci95"] = s_.get("boot", {}).get("ev_mean_ci95")
        fo, _ = stats_block(follow, dsel, tabs["primary"], nboot=NBOOT)
        blk["follow_same_signal"] = {x: fo.get(x) for x in ["n", "hit_rate", "ev_mean_pct"]}
        blk["follow_same_signal"]["ev_mean_ci95"] = fo.get("boot", {}).get("ev_mean_ci95")
        blk["sensitivity"] = sens
        blk["order_cutoff_variant"] = cutoff_variant(t, days, grids, cms["primary"]) if len(t) else None
        blk["events"] = events(t, days)
        out["periods"][pname] = blk

    # 주 기간 반(설명용, 판정 밖): 날 수로 반 나눔
    a_, b_ = (pd.Timestamp(x).date() for x in PERIODS["main_2026"])
    dm = [i for i, d in enumerate(days) if a_ <= d <= b_]
    h1, h2 = dm[: len(dm) // 2], dm[len(dm) // 2:]
    out["periods"]["main_2026"]["halves_descriptive"] = {
        f"{days[h[0]]}~{days[h[-1]]}": {x: stats_block(trades["primary"][0], h, tabs["primary"])[0].get(x)
                                         for x in ["n", "hit_rate", "ev_mean_pct"]} for h in (h1, h2)}

    # 앞 연구 구간 재현: 야후(impl_a) ↔ Alpaca(한쪽 $0.01, RSI를 9/30부터 새로 시작 = 야후와 같은 예열)
    yh = prior_yahoo_t2()
    rs0 = {t: rsi_R(data[t], start="2026-09-30") for t in TK}
    sig0 = {t: t2_signals(rs0[t], days)[0] for t in TK}
    tr0, _ = run_fade(sig0, tabs["abs_0.01usd"])
    a_, b_ = (pd.Timestamp(x).date() for x in PERIODS["repl_2026_09_30_10_28"])
    dr = [i for i, d in enumerate(days) if a_ <= d <= b_]
    s0, t0 = stats_block(tr0, dr, tabs["abs_0.01usd"])
    ev0 = events(t0, days)
    out["replication"] = {
        "prior_yahoo_reported": PRIOR_T2,
        "prior_yahoo_recomputed": {"n": len(yh), "hit_rate": r6(np.mean([x["hit"] for x in yh])),
                                   "ev_mean_pct": r6(np.mean([x["pnl_pct"] for x in yh])), "events": yh},
        "alpaca_abs001_rsi_restart_0930": {x: s0.get(x) for x in ["n", "hit_rate", "ev_mean_pct", "matched_b0"]} |
                                          {"ev_mean_ci95": s0.get("boot", {}).get("ev_mean_ci95"), "events": ev0},
        "yahoo_vs_alpaca_restart": compare_lists(yh, ev0),
        "alpaca_abs001_full_history_rsi": out["periods"]["repl_2026_09_30_10_28"]["sensitivity"]["abs_0.01usd"],
        "yahoo_vs_alpaca_full_history_abs001": compare_lists(
            yh, events(trades["abs_0.01usd"][0][trades["abs_0.01usd"][0]["d"].isin(dr)], days)),
    }

    m = out["periods"]["main_2026"]
    ok = (m.get("ev_mean_pct") or -1) > 0 and ((m.get("boot") or {}).get("ev_mean_ci95") or [-1])[0] > 0
    out["verdict"] = {"rule": "2026-01-02~09-29 평균 > 0이고 날 묶음 95% 하한 > 0이면 통과",
                      "main_ev_mean_pct": m.get("ev_mean_pct"), "main_ci95": (m.get("boot") or {}).get("ev_mean_ci95"),
                      "pass": bool(ok)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)

    # 화면 요약
    print("비용:", out["rule"]["cost"], "| 표↔반복 대조", nchk, "건 | 빈 분 채움 시 T2 차이", ff_diff)
    for p, b in out["periods"].items():
        bt = b.get("boot", {})
        print(f"{p}: 날 {b['n_days_period']} 건 {b['n']}({b.get('n_days_with_trade')}일) 도달 {b.get('hit_rate')} "
              f"평균 {b.get('ev_mean_pct')} CI {bt.get('ev_mean_ci95')} | 같은 분 무작위 {b.get('matched_b0')} | "
              f"창 무작위 {b['b0_window']} | 차이 도달 {bt.get('minus_matched_b0_hit')} {bt.get('minus_matched_b0_hit_ci95')} "
              f"손익 {bt.get('minus_matched_b0_ev_pct')} {bt.get('minus_matched_b0_ev_ci95')}")
        print("   감도", json.dumps(b["sensitivity"], ensure_ascii=False), "| 추종", b["follow_same_signal"],
              "| 마감 반영", b["order_cutoff_variant"])
    print("반", json.dumps(out["periods"]["main_2026"]["halves_descriptive"], ensure_ascii=False))
    rp = out["replication"]
    print("재현: 야후 재계산", {k: rp["prior_yahoo_recomputed"][k] for k in ["n", "hit_rate", "ev_mean_pct"]},
          "| Alpaca $0.01·9/30 새 RSI", {k: rp["alpaca_abs001_rsi_restart_0930"][k] for k in ["n", "hit_rate", "ev_mean_pct"]},
          "| 대조", rp["yahoo_vs_alpaca_restart"], "| 전체 이력 RSI 대조", rp["yahoo_vs_alpaca_full_history_abs001"])
    print("판정", out["verdict"])


if __name__ == "__main__":
    main()
