"""Features, score matrices, and the numba backtest kernel for the strategy lab.

Calendar: US trading days (QQQ). Backtest window is US close 2026-01-02 -> 2026-09-22,
which is the same information span as buying the ETFs at the KRX 2026-01-05 close
and selling at the KRX 2026-09-23 close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

BT_START = "2026-01-02"
BT_END = "2026-09-22"
IS_END = "2026-05-29"
SOURCES = ("T", "K", "U", "I")  # TIME, KoAct, union, intersection
PREFILTERS = ("all", "top20")
FLOW_WINDOWS = (1, 3, 5, 10, 20)


@dataclass
class Market:
    days: list            # backtest US days (YYYY-MM-DD)
    companies: list
    tickers: list
    R: np.ndarray         # (T, J) daily simple returns, NaN -> 0
    fxr: np.ndarray       # (T,) USD/KRW daily return
    dec_row: np.ndarray   # (T,) decision row index or -1
    dec_days: list        # decision day per row
    dec_pdf: list         # PDF date per row
    reb: dict             # name -> (T,) bool rebalance mask
    held: dict            # source -> (n_dec, J) bool
    fw: dict              # source -> (n_dec, J) fund weight used for FW weighting
    iv: np.ndarray        # (n_dec, J) inverse 20d vol
    feat: dict = field(default_factory=dict)
    ref: dict = field(default_factory=dict)   # reference price series on backtest days


def _pct_rank(mat):
    """Row-wise percentile rank in (0, 1]; NaN stays NaN."""
    out = np.full(mat.shape, np.nan)
    for r in range(mat.shape[0]):
        row = mat[r]
        ok = ~np.isnan(row)
        n = ok.sum()
        if n == 0:
            continue
        order = row[ok].argsort().argsort()
        out[r, ok] = (order + 1) / n
    return out


def build_market(d, exec_mode="close", exclude=()):
    """exec_mode 'close': trade at the US close of the decision day (default).
    'open': trade at the US open of the decision day; returns run open-to-open.
    Features always use closes up to the previous US day."""
    import pandas as pd

    adj = d["prices"]["Adj Close"]
    us_all = adj["QQQ"].dropna().index
    days_all = us_all.strftime("%Y-%m-%d").tolist()
    companies = [c for c in d["companies"] if d["ticker"][c] in adj.columns]
    tickers = [d["ticker"][c] for c in companies]
    P = adj.reindex(us_all)[tickers].to_numpy(float)
    t0 = next(i for i, x in enumerate(days_all) if x >= BT_START)
    t1 = max(i for i, x in enumerate(days_all) if x <= BT_END)
    days = days_all[t0:t1 + 1]
    T, J = len(days), len(companies)

    X = P
    if exec_mode == "open":
        raw_open = d["prices"]["Open"].reindex(us_all)[tickers].to_numpy(float)
        raw_close = d["prices"]["Close"].reindex(us_all)[tickers].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            X = raw_open * (P / raw_close)  # dividend-adjusted open
    with np.errstate(invalid="ignore", divide="ignore"):
        R = X[t0:t1 + 1] / X[t0 - 1:t1] - 1.0
    R = np.nan_to_num(R, nan=0.0)
    R[0] = 0.0  # positions are first opened at the close of day 0

    fx = adj.reindex(us_all)["KRW=X"].ffill().to_numpy(float)
    fxr = fx[t0:t1 + 1] / fx[t0 - 1:t1] - 1.0
    fxr[0] = 0.0

    # decision rows: first US day on/after each PDF, latest PDF wins
    pdfs = d["pdf_dates"]
    dec_row = np.full(T, -1, np.int64)
    dec_days, dec_pdf, dec_t = [], [], []
    last_pdf = None
    for t, day in enumerate(days):
        latest = max(p for p in pdfs if p <= day)
        if latest != last_pdf:
            dec_row[t] = len(dec_days)
            dec_days.append(day)
            dec_pdf.append(latest)
            dec_t.append(t0 + t)  # index into days_all
            last_pdf = latest
    n_dec = len(dec_days)

    # rebalance masks on backtest days
    stamp = pd.to_datetime(days)
    iso = [(s.isocalendar().year, s.isocalendar().week) for s in stamp]
    reb = {name: np.zeros(T, np.bool_) for name in ("D", "W", "2W", "M")}
    prev_week, prev_month, week_count = None, None, -1
    for t in range(T):
        if dec_row[t] < 0:
            continue
        reb["D"][t] = True
        if iso[t] != prev_week:
            week_count += 1
            reb["W"][t] = True
            reb["2W"][t] = week_count % 2 == 0
            prev_week = iso[t]
        if days[t][:7] != prev_month:
            reb["M"][t] = True
            prev_month = days[t][:7]
    for mask in reb.values():
        mask[0] = True

    # fund panels on decision rows
    panels = d["panels"]
    col = {c: j for j, c in enumerate(companies)}
    pdf_index = {p: i for i, p in enumerate(pdfs)}

    def fund_mat(tag, getter):
        m = np.zeros((n_dec, J))
        for r, p in enumerate(dec_pdf):
            for c, v in getter(tag, p).items():
                if c in col:
                    m[r, col[c]] = v
        return m

    wT = fund_mat("T", lambda f, p: panels[f]["weights"][p])
    wK = fund_mat("K", lambda f, p: panels[f]["weights"][p])
    flow = {(f, L): fund_mat(f, lambda ff, p, L=L: panels[ff]["flow_pp"][str(L)][p])
            for f in "TK" for L in FLOW_WINDOWS}
    new = {(f, L): fund_mat(f, lambda ff, p, L=L: {c: 1.0 for c in panels[ff]["new_in"][str(L)][p]})
           for f in "TK" for L in FLOW_WINDOWS}

    def lag_weights(tag, L):
        m = np.zeros((n_dec, J))
        for r, p in enumerate(dec_pdf):
            q = pdfs[max(0, pdf_index[p] - L)]
            for c, v in panels[tag]["weights"][q].items():
                if c in col:
                    m[r, col[c]] = v
        return m

    wT_lag = {L: lag_weights("T", L) for L in (5, 20)}
    wK_lag = {L: lag_weights("K", L) for L in (5, 20)}

    # price features from closes up to the day before each decision day
    dt = np.array(dec_t)
    tradeable = ~np.isnan(P[dt])
    for j, tk in enumerate(tickers):
        if tk in exclude:
            tradeable[:, j] = False  # leave-winner-out robustness runs
    last = P[dt - 1]
    mom = {}
    for k in (5, 10, 20, 40, 60, 120):
        with np.errstate(invalid="ignore", divide="ignore"):
            mom[k] = last / P[dt - 1 - k] - 1.0
    logret = np.log(P[1:] / P[:-1])
    vol20 = np.array([np.nanstd(logret[i - 21:i - 1], axis=0) * np.sqrt(252) for i in dt])
    vol20[np.sum(~np.isnan(np.array([logret[i - 21:i - 1] for i in dt])), axis=1) < 15] = np.nan
    high = {}
    for k in (60, 120):
        high[k] = np.array([last[r] / np.nanmax(P[i - k:i], axis=0) for r, i in enumerate(dt)])
    ma20 = np.array([np.nanmean(P[i - 20:i], axis=0) for i in dt])

    held = {"T": wT > 0, "K": wK > 0, "U": (wT > 0) | (wK > 0), "I": (wT > 0) & (wK > 0)}
    wavg = (wT + wK) / 2.0
    fw = {"T": wT, "K": wK, "U": wavg, "I": wavg}
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = 1.0 / vol20

    # reference series on backtest days
    ref = {}
    for tk in ("QQQ", "SMH", "^VIX"):
        ref[tk] = adj.reindex(us_all)[tk].ffill().to_numpy(float)
    ref_idx = (t0, t1)

    feat = {"wT": wT, "wK": wK, "flow": flow, "new": new, "wT_lag": wT_lag, "wK_lag": wK_lag,
            "mom": mom, "vol20": vol20, "high": high, "tradeable": tradeable,
            "ma20_above": last > ma20, "dec_t": dt, "P": P, "days_all": days_all,
            "ref_idx": ref_idx}
    return Market(days, companies, tickers, R, fxr, dec_row, dec_days, dec_pdf, reb, held,
                  fw, iv, feat, ref)


def build_scores(m: Market):
    """Return (names, stack) where stack[i] is an (n_dec, J) score matrix, NaN = not eligible."""
    f = m.feat
    wT, wK = f["wT"], f["wK"]
    names, mats = [], []
    for src in SOURCES:
        held = m.held[src] & f["tradeable"]
        if src == "T":
            w, fl = wT, {L: f["flow"][("T", L)] for L in FLOW_WINDOWS}
            nw = {L: f["new"][("T", L)] > 0 for L in FLOW_WINDOWS}
            dw = {L: wT - f["wT_lag"][L] for L in (5, 20)}
            cons = {L: (fl[L] > 0).astype(float) for L in (5, 20)}
        elif src == "K":
            w, fl = wK, {L: f["flow"][("K", L)] for L in FLOW_WINDOWS}
            nw = {L: f["new"][("K", L)] > 0 for L in FLOW_WINDOWS}
            dw = {L: wK - f["wK_lag"][L] for L in (5, 20)}
            cons = {L: (fl[L] > 0).astype(float) for L in (5, 20)}
        else:
            w = (wT + wK) / 2.0
            fl = {L: f["flow"][("T", L)] + f["flow"][("K", L)] for L in FLOW_WINDOWS}
            nw = {L: (f["new"][("T", L)] > 0) | (f["new"][("K", L)] > 0) for L in FLOW_WINDOWS}
            dw = {L: (wT - f["wT_lag"][L] + wK - f["wK_lag"][L]) / 2.0 for L in (5, 20)}
            cons = {L: (f["flow"][("T", L)] > 0).astype(float) + (f["flow"][("K", L)] > 0)
                    for L in (5, 20)}
        both = ((wT > 0) & (wK > 0)).astype(float)

        for pre in PREFILTERS:
            mask = held.copy()
            if pre == "top20":
                ww = np.where(held, w, -np.inf)
                rank = (-ww).argsort(axis=1).argsort(axis=1)
                mask &= rank < 20

            def elig(x):
                return np.where(mask & ~np.isnan(x), x, np.nan)

            pw = _pct_rank(elig(w))
            tie = w * 1e-9
            base = {}
            base["W"] = w
            for L in FLOW_WINDOWS:
                base[f"FLOW{L}"] = fl[L] + tie
            for L in (5, 20):
                base[f"DW{L}"] = dw[L] + tie
            for L in (5, 10, 20):
                base[f"NEW{L}"] = w + 1000.0 * nw[L]
            for k in (5, 10, 20, 40, 60, 120):
                base[f"MOM{k}"] = f["mom"][k]
            for k in (20, 60, 120):
                base[f"RADJ{k}"] = f["mom"][k] / f["vol20"]
            for k in (60, 120):
                base[f"HIGH{k}"] = f["high"][k] + tie
            base["REV5"] = -f["mom"][5]
            base["PULL"] = _pct_rank(elig(f["mom"][60])) - _pct_rank(elig(f["mom"][5]))
            for k in (20, 60):
                pm = _pct_rank(elig(f["mom"][k]))
                for lam in (25, 50, 75):
                    base[f"WMOM{k}_{lam}"] = lam / 100 * pw + (1 - lam / 100) * pm
            for L in (5, 20):
                pf = _pct_rank(elig(fl[L]))
                for k in (20, 60):
                    base[f"FMOM{L}_{k}"] = 0.5 * pf + 0.5 * _pct_rank(elig(f["mom"][k]))
                base[f"WFLOW{L}"] = 0.5 * pw + 0.5 * pf
                base[f"CONS{L}"] = cons[L] + 0.5 * pw
            base["BOTH"] = both + 0.5 * pw
            for name, mat in base.items():
                names.append((src, pre, name))
                mats.append(elig(mat))
    return names, np.stack(mats).astype(np.float64)


def apply_delay(m: Market, stack, delay):
    """Use each decision row's information `delay` rows later (unverified publication
    time test). Scores, fund weights, held masks and vols all shift together; the
    first `delay` rows have no signal (the strategy waits in cash)."""
    if delay <= 0:
        return stack
    out = np.full_like(stack, np.nan)
    out[:, delay:] = stack[:, :-delay]
    for src in list(m.fw):
        fw = np.zeros_like(m.fw[src])
        fw[delay:] = m.fw[src][:-delay]
        m.fw[src] = fw
        held = np.zeros_like(m.held[src])
        held[delay:] = m.held[src][:-delay]
        m.held[src] = held
    iv = np.full_like(m.iv, np.nan)
    iv[delay:] = m.iv[:-delay]
    m.iv = iv
    return out


# ----------------------------------------------------------------------------
# numba kernel
# ----------------------------------------------------------------------------

@njit(cache=True)
def _select(score_row, w_cur, N, buffer, out_sel):
    J = score_row.shape[0]
    idx = np.empty(J, np.int64)
    n = 0
    for j in range(J):
        if not np.isnan(score_row[j]):
            idx[n] = j
            n += 1
    if n == 0:
        return 0
    vals = np.empty(n)
    for i in range(n):
        vals[i] = -score_row[idx[i]]
    ranked = idx[:n][np.argsort(vals, kind="mergesort")]
    k = 0
    if buffer > 0:
        limit = min(n, N + buffer)
        for r in range(limit):
            j = ranked[r]
            if w_cur[j] > 0.0 and k < N:
                out_sel[k] = j
                k += 1
    for r in range(n):
        if k >= N:
            break
        j = ranked[r]
        dup = False
        for q in range(k):
            if out_sel[q] == j:
                dup = True
                break
        if not dup:
            out_sel[k] = j
            k += 1
    return k


@njit(cache=True)
def _weights(sel, k, fw_row, iv_row, weighting, out_w):
    out_w[:] = 0.0
    if k == 0:
        return
    if weighting == 0:
        for q in range(k):
            out_w[sel[q]] = 1.0 / k
        return
    src = fw_row if weighting == 1 else iv_row
    tot, good = 0.0, 0
    for q in range(k):
        v = src[sel[q]]
        if v > 0.0 and not np.isnan(v):
            tot += v
            good += 1
    if good == 0:
        for q in range(k):
            out_w[sel[q]] = 1.0 / k
        return
    fill = tot / good  # names without a value get the average value
    tot2 = 0.0
    for q in range(k):
        v = src[sel[q]]
        if not (v > 0.0) or np.isnan(v):
            v = fill
        out_w[sel[q]] = v
        tot2 += v
    for q in range(k):
        out_w[sel[q]] /= tot2


@njit(cache=True)
def _run(R, dec_row, reb, score, fw, iv, held, N, weighting, buffer, exit_rule, stop,
         gross, turn, nh):
    T, J = R.shape
    w = np.zeros(J)
    grow = np.ones(J)
    peak = np.ones(J)
    sel = np.empty(J, np.int64)
    tw = np.zeros(J)
    for t in range(T):
        if t > 0:
            g = 0.0
            for j in range(J):
                if w[j] > 0.0:
                    g += w[j] * R[t, j]
            gross[t] = g
            inv = 1.0 / (1.0 + g)
            for j in range(J):
                if w[j] > 0.0:
                    w[j] = w[j] * (1.0 + R[t, j]) * inv
                    grow[j] *= 1.0 + R[t, j]
                    if grow[j] > peak[j]:
                        peak[j] = grow[j]
        row = dec_row[t]
        traded = 0.0
        if row >= 0 and reb[t]:
            k = _select(score[row], w, N, buffer, sel)
            _weights(sel, k, fw[row], iv[row], weighting, tw)
            for j in range(J):
                traded += abs(tw[j] - w[j])
                if tw[j] > 0.0 and w[j] == 0.0:
                    grow[j] = 1.0
                    peak[j] = 1.0
                w[j] = tw[j]
        else:
            if exit_rule == 1 and row >= 0:
                for j in range(J):
                    if w[j] > 0.0 and not held[row, j]:
                        traded += w[j]
                        w[j] = 0.0
            elif exit_rule == 2:
                for j in range(J):
                    if w[j] > 0.0 and grow[j] / peak[j] < 1.0 - stop:
                        traded += w[j]
                        w[j] = 0.0
        turn[t] = traded
        c = 0
        for j in range(J):
            if w[j] > 0.0:
                c += 1
        nh[t] = c


@njit(parallel=True, cache=True)
def run_batch(R, dec_row, reb_stack, score_stack, fw_stack, iv, held_stack, params, stop,
              gross, turn, nh):
    """params columns: score_id, N, weighting, buffer, reb_id, exit_rule, source_id."""
    S = params.shape[0]
    for s in prange(S):
        _run(R, dec_row, reb_stack[params[s, 4]], score_stack[params[s, 0]],
             fw_stack[params[s, 6]], iv, held_stack[params[s, 6]], params[s, 1], params[s, 2],
             params[s, 3], params[s, 5], stop, gross[s], turn[s], nh[s])


@njit(cache=True)
def run_weights(R, dec_row, reb, score, fw, iv, held, N, weighting, buffer, exit_rule, stop, W):
    """Same rules as _run; stores end-of-day weights W[t] (after that day's trades)."""
    T, J = R.shape
    gross = np.zeros(T)
    turn = np.zeros(T)
    nh = np.zeros(T, np.int64)
    w = np.zeros(J)
    grow = np.ones(J)
    peak = np.ones(J)
    sel = np.empty(J, np.int64)
    tw = np.zeros(J)
    for t in range(T):
        if t > 0:
            g = 0.0
            for j in range(J):
                if w[j] > 0.0:
                    g += w[j] * R[t, j]
            gross[t] = g
            inv = 1.0 / (1.0 + g)
            for j in range(J):
                if w[j] > 0.0:
                    w[j] = w[j] * (1.0 + R[t, j]) * inv
                    grow[j] *= 1.0 + R[t, j]
                    if grow[j] > peak[j]:
                        peak[j] = grow[j]
        row = dec_row[t]
        traded = 0.0
        if row >= 0 and reb[t]:
            k = _select(score[row], w, N, buffer, sel)
            _weights(sel, k, fw[row], iv[row], weighting, tw)
            for j in range(J):
                traded += abs(tw[j] - w[j])
                if tw[j] > 0.0 and w[j] == 0.0:
                    grow[j] = 1.0
                    peak[j] = 1.0
                w[j] = tw[j]
        else:
            if exit_rule == 1 and row >= 0:
                for j in range(J):
                    if w[j] > 0.0 and not held[row, j]:
                        traded += w[j]
                        w[j] = 0.0
            elif exit_rule == 2:
                for j in range(J):
                    if w[j] > 0.0 and grow[j] / peak[j] < 1.0 - stop:
                        traded += w[j]
                        w[j] = 0.0
        turn[t] = traded
        for j in range(J):
            W[t, j] = w[j]
    return gross, turn
