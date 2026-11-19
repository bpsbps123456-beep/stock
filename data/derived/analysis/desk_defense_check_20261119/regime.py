"""Regime study: why 6/22-7/29 and 9/22-11/17 were "hell", what each desk tab showed, and tab-signal gates.

    py data/derived/analysis/desk_defense_check_20261119/regime.py [declines|attrib|signals|gates|all]

Reads the frozen Codex engine through common.py. Writes only cache/regime_*.{csv,json} and, for the
gate family, common.save("tabsig", ...). Every series is dated by the close it is known at; a gate
built from it is shifted one session (decision at the open of j uses values through close j-1; fund
PDFs dated strictly before j, the base rule's own convention).
"""
from __future__ import annotations

import json
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import study, S  # noqa: E402
from desk import config as desk_config  # noqa: E402
from gbrc.etf import flows as qflows  # noqa: E402
from gbrc.etf.market import is_cash_holding, listing_market, yahoo_ticker  # noqa: E402
from gbrc.etf.storage import load_fund_snapshots  # noqa: E402
import watchlist  # noqa: E402

CACHE = common.CACHE
CODEX = common.CODEX
HELL = {"jul": ("2026-06-22", "2026-07-29"), "octnov": ("2026-09-22", "2026-11-17"),
        "octnov_leg": ("2026-10-13", "2026-11-17")}
OPTICAL = {"AAOI", "LITE", "COHR", "CIEN", "IPGP", "VIAV", "AXTI", "FN", "POET", "GLW", "CRDO", "ALAB"}


# ---------------------------------------------------------------------------
# base ledger with per-holding shares
# ---------------------------------------------------------------------------

def holdings_replay(ctx):
    """study.replay(base) keeping share counts: per-session overnight and intraday dollar P&L by column."""
    days, opn, close = ctx.days, ctx.opn, ctx.close
    n = opn.shape[1]
    shares, cash = np.zeros(n), 1.0
    night, intra = np.zeros((len(days), n)), np.zeros((len(days), n))
    nav_open, nav_close, held = np.zeros(len(days)), np.zeros(len(days)), np.zeros((len(days), n))
    prev_close = np.zeros(n)
    for j in range(len(days)):
        o, c = np.nan_to_num(opn[j]), np.nan_to_num(close[j])
        if j:
            night[j] = shares * (o - prev_close)
        nav = cash + shares @ o
        nav_open[j] = nav
        if j in ctx.weekly:
            w = ctx.weekly[j]
            shares = np.divide(nav * w, opn[j], out=np.zeros(n), where=w > 0)
            cash = nav * (1 - w.sum())
        intra[j] = shares * (c - o)
        nav_close[j] = cash + shares @ c
        held[j] = shares * c
        prev_close = c
    return dict(night=night, intra=intra, nav_open=nav_open, nav_close=nav_close, held=held)


def group_of(ctx, key):
    """semis / memory / optical / other from the cap sector (watchlist) plus an optical ticker list."""
    tk = ctx.book.ticker[key]
    sector = ctx.book.sector[key]
    if tk in OPTICAL or "광" in (S.subsector_of(key, tk) or ""):
        return "optical"
    if sector == "메모리":
        return "memory"
    if sector in desk_config.SEMIS_SECTORS:
        return "semis"
    return "other"


# ---------------------------------------------------------------------------
# 1) declines
# ---------------------------------------------------------------------------

def zigzag_declines(nav, days, depth=.08):
    """Peak-to-trough legs of >= depth on closes. A leg ends when the close regains the leg's peak
    or rebounds by `depth` from the low (then a new peak search starts from that rebound)."""
    out, i, n = [], 0, len(nav)
    peak = 0
    while i < n:
        if nav[i] >= nav[peak]:
            peak = i
            i += 1
            continue
        # in a decline from peak: follow until recovered or rebound >= depth from the running low
        low, j = i, i
        while j < n and nav[j] < nav[peak] and nav[j] < nav[low] * (1 + depth):
            if nav[j] < nav[low]:
                low = j
            j += 1
        d = nav[low] / nav[peak] - 1
        if d <= -depth:
            rec = next((k for k in range(low, n) if nav[k] >= nav[peak]), None)
            half = next((k for k in range(low, n) if nav[k] >= (nav[peak] + nav[low]) / 2), None)
            out.append(dict(peak=days[peak], trough=days[low], depth=d, sessions_to_trough=low - peak,
                            recovered=days[rec] if rec is not None else "not recovered",
                            sessions_trough_to_recover=(rec - low) if rec is not None else np.nan,
                            sessions_peak_to_recover=(rec - peak) if rec is not None else np.nan,
                            half_recovered=days[half] if half is not None else "not",
                            sessions_to_half=(half - low) if half is not None else np.nan,
                            peak_i=peak, trough_i=low))
        if j >= n:
            break
        peak = j if nav[j] < nav[peak] else j  # restart the peak search at the rebound / recovery bar
        i = j + 1
    return pd.DataFrame(out)


def hwm_episodes(nav, days, depth=.08):
    """High-water-mark episodes: from a record close to the next record close."""
    out, peak = [], 0
    low = 0
    for i in range(1, len(nav) + 1):
        if i == len(nav) or nav[i] >= nav[peak]:
            if low != peak and nav[low] / nav[peak] - 1 <= -depth:
                out.append(dict(peak=days[peak], trough=days[low], depth=nav[low] / nav[peak] - 1,
                                sessions_to_trough=low - peak,
                                recovered=days[i] if i < len(nav) else "not recovered",
                                sessions_trough_to_recover=(i - low) if i < len(nav) else np.nan))
            if i < len(nav):
                peak = low = i
        elif nav[i] < nav[low]:
            low = i
    return pd.DataFrame(out)


def declines(ctx, base):
    days = ctx.days
    nav = np.r_[base.usd.to_numpy()]
    zz = zigzag_declines(nav, days)
    hw = hwm_episodes(nav, days)
    krw = base.krw.to_numpy()
    soxx, qqq = (ctx.book.cls[s].reindex(days).to_numpy() for s in ("SOXX", "QQQ"))
    for frame in (zz, hw):
        ix = [(days.index(a), days.index(b)) for a, b in zip(frame.peak, frame.trough)]
        frame["krw_depth"] = [krw[b] / krw[a] - 1 for a, b in ix]
        frame["soxx"] = [soxx[b] / soxx[a] - 1 for a, b in ix]
        frame["qqq"] = [qqq[b] / qqq[a] - 1 for a, b in ix]
    zz.drop(columns=["peak_i", "trough_i"]).to_csv(CACHE / "regime_declines_zigzag.csv", index=False)
    hw.to_csv(CACHE / "regime_declines_hwm.csv", index=False)
    return zz, hw


# ---------------------------------------------------------------------------
# 2) attribution
# ---------------------------------------------------------------------------

def attribution(ctx, base, hold, a, b):
    """Dollar P&L of [close a .. close b] by holding, as a fraction of the NAV at close a."""
    days = ctx.days
    ia, ib = days.index(a), days.index(b)
    nav0 = hold["nav_close"][ia]
    keys = list(ctx.col)
    night = hold["night"][ia + 1:ib + 1].sum(0) / nav0
    intra = hold["intra"][ia + 1:ib + 1].sum(0) / nav0
    total = night + intra
    rows = []
    for key, i in ctx.col.items():
        if abs(total[i]) < 1e-12 and not hold["held"][ia:ib + 1, i].any():
            continue
        px = ctx.close[:, i]
        rows.append(dict(ticker=ctx.tickers[i], group=group_of(ctx, key), sector=ctx.book.sector[key],
                         contrib=total[i], overnight=night[i], intraday=intra[i],
                         sessions_held=int((hold["held"][ia + 1:ib + 1, i] > 0).sum()),
                         weight_at_start=hold["held"][ia, i] / nav0,
                         price_change=px[ib] / px[ia] - 1))
    table = pd.DataFrame(rows).sort_values("contrib")
    usd = hold["nav_close"][ib] / nav0 - 1
    assert abs(table.contrib.sum() - usd) < 1e-9, (table.contrib.sum(), usd)
    assert abs(usd - (base.usd.iloc[ib] / base.usd.iloc[ia] - 1)) < 1e-9
    fx = ctx.fx[ib] / ctx.fx[ia] - 1
    krw = base.krw.iloc[ib] / base.krw.iloc[ia] - 1
    summary = dict(window=f"{a}..{b}", usd=usd, fx=fx, krw=krw, fx_interaction=krw - usd - fx,
                   overnight=table.overnight.sum(), intraday=table.intraday.sum(),
                   avg_exposure=float(base.exposure.iloc[ia + 1:ib + 1].mean()),
                   n_names=len(table), top3_loss_share=table.contrib.nsmallest(3).sum() / usd if usd < 0 else np.nan,
                   top3_gain_share=table.contrib.nlargest(3).sum() / usd if usd > 0 else np.nan)
    by_group = table.groupby("group")[["contrib", "overnight", "intraday", "weight_at_start"]].sum()
    return summary, table, by_group


def spells(ctx, hold):
    """Every holding spell of the base: entry/exit sessions, 120-session rank at entry and exit, return."""
    days = ctx.days
    held = hold["held"] > 0
    rank = {}
    for j, ranked in enumerate(ctx.rankings):
        rank[j] = {r["key"]: (r["rank"], r["mom120"], len(ranked)) for r in ranked}
    rows = []
    for key, i in ctx.col.items():
        h = held[:, i]
        if not h.any():
            continue
        starts = np.flatnonzero(h & ~np.r_[False, h[:-1]])
        ends = np.flatnonzero(h & ~np.r_[h[1:], False])
        for s, e in zip(starts, ends):
            sold = e + 1 if e + 1 < len(days) else None            # sold at the open of session e+1
            entry_px = ctx.opn[s, i]
            exit_px = ctx.opn[sold, i] if sold is not None else ctx.close[e, i]
            hi = np.nanmax(ctx.close[s:e + 1, i])
            hi_j = s + int(np.nanargmax(ctx.close[s:e + 1, i]))
            r_in = rank[s].get(key, (np.nan, np.nan, np.nan))
            r_out = rank[sold].get(key, (np.nan, np.nan, np.nan)) if sold is not None else rank[e].get(key, (np.nan,) * 3)
            pnl = (hold["night"][s:(sold or e) + 1, i].sum() + hold["intra"][s:e + 1, i].sum())
            rows.append(dict(ticker=ctx.tickers[i], group=group_of(ctx, key), entry=days[s],
                             exit=days[sold] if sold is not None else "held",
                             sessions=e - s + 1, rank_in=r_in[0], mom120_in=r_in[1], n_ranked_in=r_in[2],
                             rank_out=r_out[0], mom120_out=r_out[1],
                             ret=exit_px / entry_px - 1, pnl_nav_units=pnl,
                             peak_close=days[hi_j], give_back=exit_px / hi - 1,
                             sessions_after_own_peak=(e - hi_j)))
    out = pd.DataFrame(rows).sort_values(["entry", "ticker"])
    out.to_csv(CACHE / "regime_spells.csv", index=False)
    return out


def run_attribution(ctx, base, hold):
    res = {}
    wins = dict(HELL, h1=("2026-01-02", "2026-06-30"), full=("2026-01-02", "2026-11-17"),
                jan_sep22=("2026-01-02", "2026-09-22"), aug_sep_rally=("2026-07-29", "2026-09-22"))
    for name, (a, b) in wins.items():
        summary, table, by_group = attribution(ctx, base, hold, a, b)
        table.to_csv(CACHE / f"regime_attrib_{name}.csv", index=False)
        res[name] = dict(summary=summary, by_group=by_group.round(5).to_dict("index"),
                         worst=table.head(8).round(4).to_dict("records"),
                         best=table.tail(5).round(4).to_dict("records"))
    return res


# ---------------------------------------------------------------------------
# 3) signal panel. Row d = what is known after the close of US session d (usable at the open of d+1).
# ---------------------------------------------------------------------------

ROOT = common.ROOT
END = "2026-11-17"


def fund_panel(fund, days):
    """Per-PDF fund series, then mapped to sessions: row d uses the latest PDF dated strictly before
    the next session (the base rule's convention: a PDF dated D is first used at the open after D)."""
    snaps = load_fund_snapshots(fund, END)
    pdfs = sorted(snaps)
    rows, prev = [], None
    for day in pdfs:
        snap = snaps[day]
        rec = dict(cash_w=0., us_w=0., fut_w=0., semis_w=0., semis_us_w=0., memory_w=0., sw_w=0., n_names=0, n_us=0)
        for row in snap["holdings"]:
            w = float(row["weight"])
            if is_cash_holding(row):
                rec["cash_w"] += w
                continue
            key = S.research_key(row["code"])
            market = listing_market(row["code"])
            sec = S.sector_of(key, yahoo_ticker(row["code"], day) or "")
            rec["n_names"] += 1
            if market == "FUT":
                rec["fut_w"] += w
            if market == "US":
                rec["us_w"] += w
                rec["n_us"] += 1
            if sec in desk_config.SEMIS_SECTORS:
                rec["semis_w"] += w
                if market == "US":
                    rec["semis_us_w"] += w
            if sec == "메모리":
                rec["memory_w"] += w
            if sec == "엔터프라이즈 SW":
                rec["sw_w"] += w
        net = dict(net_all=0., net_semis=0., net_us=0., net_us_semis=0., sells=0, buys=0, common=np.nan)
        if prev is not None:
            flow = qflows.quantity_flows([snaps[prev], snap])
            net["common"] = flow.common_factor
            for f in flow.flows.values():
                market = listing_market(f.code)
                if market == "FUT" or f.trade_pp is None or f.kind in ("유지", "N/A"):
                    continue                     # 유지 = below the desk's 5% real-trade threshold
                sec = S.sector_of(S.research_key(f.code), yahoo_ticker(f.code, day) or "")
                net["net_all"] += f.trade_pp
                net["sells"] += f.kind in qflows.SELL_KINDS
                net["buys"] += f.kind in qflows.BUY_KINDS
                if market == "US":
                    net["net_us"] += f.trade_pp
                if sec in desk_config.SEMIS_SECTORS:
                    net["net_semis"] += f.trade_pp
                    if market == "US":
                        net["net_us_semis"] += f.trade_pp
        rows.append(dict(pdf=day, **rec, **net))
        prev = day
    pdf = pd.DataFrame(rows).set_index("pdf")
    for n in (5, 10, 20):
        for c in ("net_all", "net_semis", "net_us", "net_us_semis"):
            pdf[f"{c}_{n}"] = pdf[c].rolling(n).sum()
        for c in ("cash_w", "semis_w", "us_w"):
            pdf[f"{c}_chg{n}"] = pdf[c] - pdf[c].shift(n)
    pdf["cash_fut"] = pdf.cash_w - pdf.fut_w      # cash not covered by index futures
    nxt = days[1:] + ["9999-12-31"]
    pick = [pdfs[bisect_left(pdfs, d) - 1] if bisect_left(pdfs, d) > 0 else None for d in nxt]
    blank = pd.Series(np.nan, index=pdf.columns)
    out = pd.DataFrame([pdf.loc[k] if k is not None else blank for k in pick], index=days)
    out["pdf_used"] = pick
    return pdf, out


def price_panel(ctx):
    days = ctx.days
    out = pd.DataFrame(index=days)
    for sym in ("SOXX", "QQQ"):
        c = ctx.book.cls[sym]
        for n in (10, 20, 50, 100):
            out[f"{sym.lower()}_sma{n}"] = (c / c.rolling(n).mean() - 1).reindex(days)
        out[f"{sym.lower()}_dd60"] = (c / c.rolling(60).max() - 1).reindex(days)
        out[f"{sym.lower()}_ret5"] = (c / c.shift(5) - 1).reindex(days)
    return out


def tab_panel(ctx, base, hold):
    days = ctx.days
    out = pd.DataFrame(index=days)
    b = pd.read_csv(CODEX / "breadth.csv", index_col=0).breadth
    out["breadth"] = b.reindex(days)
    out["breadth_chg10"] = (b - b.shift(10)).reindex(days)
    out["breadth_vs_max20"] = (b - b.rolling(20).max()).reindex(days)
    trend = pd.read_csv(CODEX / "trend_grades.csv", index_col=0).reindex(days)
    giant = pd.read_csv(CODEX / "giant_grades.csv", index_col=0).reindex(days)
    out["trend_SA_share"] = (trend >= 3).sum(axis=1) / trend.notna().sum(axis=1)
    out["trend_D_share"] = (trend == 0).sum(axis=1) / trend.notna().sum(axis=1)
    out["trend_n"] = trend.notna().sum(axis=1)
    out["giant_SA_share"] = (giant >= 3).sum(axis=1) / giant.notna().sum(axis=1)
    out["giant_D_share"] = (giant == 0).sum(axis=1) / giant.notna().sum(axis=1)
    out["giant_mean"] = giant.mean(axis=1)
    out["giant_n"] = giant.notna().sum(axis=1)
    held = hold["held"] > 0                                   # the 7 names the base holds at that close
    tg, gg = trend[ctx.tickers].to_numpy(), giant[ctx.tickers].to_numpy()

    def mean_of(grid, j):
        x = grid[j, held[j]]
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    out["held_trend_mean"] = [mean_of(tg, j) for j in range(len(days))]
    out["held_trend_SA"] = [float(np.nansum(tg[j, held[j]] >= 3)) if held[j].any() else np.nan for j in range(len(days))]
    out["held_giant_mean"] = [mean_of(gg, j) for j in range(len(days))]
    # cross-section of TIME's US roster (as ranked for that session: PDF before the session)
    full = ctx.book.cls[ctx.tickers]
    ret = full.pct_change(fill_method=None)
    pos = {d: i for i, d in enumerate(full.index)}
    line50 = full.rolling(50).mean()
    sma50 = (full >= line50).where(line50.notna() & full.notna()).astype(float)
    disp, corr_t, corr_h, above50 = [], [], [], []
    for j, day in enumerate(days):
        roster = [ctx.col[r["key"]] for r in ctx.rankings[j]]
        i = pos[day]
        win = ret.iloc[i - 19:i + 1]

        def avg_corr(cols):
            x = win.iloc[:, cols].dropna(axis=1)
            if x.shape[1] < 3:
                return np.nan
            c = np.corrcoef(x.to_numpy().T)
            return (c.sum() - len(c)) / (len(c) * (len(c) - 1))
        disp.append(win.iloc[:, roster].std(axis=1).mean() if roster else np.nan)
        corr_t.append(avg_corr(roster) if roster else np.nan)
        corr_h.append(avg_corr(list(np.flatnonzero(held[j]))) if held[j].sum() >= 3 else np.nan)
        above50.append(sma50.iloc[i, roster].mean() if roster else np.nan)
    out["time_us_disp20"], out["time_us_corr20"], out["held_corr20"] = disp, corr_t, corr_h
    out["time_us_above_sma50"] = above50
    r = base.usd.pct_change()
    out["base_vol20"] = r.rolling(20).std() * np.sqrt(252)
    out["base_dd"] = base.usd / base.usd.cummax() - 1
    out["base_vs_sma20"] = base.usd / base.usd.rolling(20).mean() - 1
    out["base_ret5"] = base.usd / base.usd.shift(5) - 1
    return out


def other_panel(days):
    """Other tabs with a daily history. All of these detectors were DESIGNED in Oct-Nov 2026 (hindsight)."""
    out = pd.DataFrame(index=days)
    cal = ROOT / "data/derived/analysis/bottom_zone_20261111/belt/now/bb1_calendar.json"
    if cal.exists():
        c = pd.DataFrame(json.loads(cal.read_text(encoding="utf-8"))).set_index("day")
        out["bb1_on"] = c["on"].astype(float).reindex(days)
        out["bb1_soxl_percentile"] = c["percentile"].reindex(days)
    ant = ROOT / "data/derived/analysis/ant_index/final/results/feat_activity.parquet"
    if ant.exists():
        a = pd.read_parquet(ant)
        a = a[a.market == "us"].assign(P=lambda x: pd.to_datetime(x.P).dt.strftime("%Y-%m-%d"))
        out["ant_activity_z"] = a.drop_duplicates("P").set_index("P").A_rz8.reindex(days)
    bot = ROOT / "data/derived/desk/bottom.json"
    if bot.exists():
        sig = [s for s in json.loads(bot.read_text(encoding="utf-8"))["signals"] if s["date"] >= "2026"]
        out["bottom_signal"] = pd.Series({s["date"]: 1.0 for s in sig}).reindex(days).fillna(0.)
    return out


# name -> +1 if HIGH values are the alarm, -1 if LOW values are the alarm
DIRECTION = {
    **{f"{s}_sma{n}": -1 for s in ("soxx", "qqq") for n in (10, 20, 50, 100)},
    "soxx_dd60": -1, "qqq_dd60": -1, "soxx_ret5": -1, "qqq_ret5": -1,
    "breadth": -1, "breadth_chg10": -1, "breadth_vs_max20": -1,
    "trend_SA_share": -1, "trend_D_share": 1, "giant_SA_share": -1, "giant_D_share": 1, "giant_mean": -1,
    "held_trend_mean": -1, "held_trend_SA": -1, "held_giant_mean": -1,
    "time_us_disp20": 1, "time_us_corr20": 1, "held_corr20": 1, "time_us_above_sma50": -1,
    "base_vol20": 1, "base_vs_sma20": -1, "base_ret5": -1,
    "bb1_soxl_percentile": 1, "ant_activity_z": 1,
}
for _f in ("time", "koact"):
    DIRECTION.update({f"{_f}_cash_w": 1, f"{_f}_cash_fut": 1, f"{_f}_us_w": -1, f"{_f}_semis_w": -1,
                      f"{_f}_semis_us_w": -1, f"{_f}_memory_w": -1, f"{_f}_n_names": 1, f"{_f}_fut_w": -1})
    for _n in (5, 10, 20):
        DIRECTION.update({f"{_f}_net_all_{_n}": -1, f"{_f}_net_semis_{_n}": -1, f"{_f}_net_us_{_n}": -1,
                          f"{_f}_net_us_semis_{_n}": -1, f"{_f}_cash_w_chg{_n}": 1,
                          f"{_f}_semis_w_chg{_n}": -1, f"{_f}_us_w_chg{_n}": -1})


def build_panel(ctx, base, hold):
    days = ctx.days
    parts = [price_panel(ctx), tab_panel(ctx, base, hold), other_panel(days)]
    pdfs = {}
    for fund, label in (("456600", "time"), ("2ETFQ1", "koact")):
        pdf, mapped = fund_panel(fund, days)
        pdfs[label] = pdf
        parts.append(mapped.add_prefix(f"{label}_"))
        pdf.to_csv(CACHE / f"regime_fund_{label}_by_pdf.csv")
    panel = pd.concat(parts, axis=1)
    panel.to_csv(CACHE / "regime_signal_panel.csv")
    return panel, pdfs


def first_dd(nav, peak_i, level, end_i):
    for k in range(peak_i, end_i + 1):
        if nav[k] / nav[peak_i] - 1 <= -level:
            return k
    return None


def event_table(ctx, base, panel, zz):
    """Signal values at P-10, P-5, P, first close 5% and 10% below the peak, and the trough."""
    nav = base.usd.to_numpy()
    rows, marks = [], {}
    for _, d in zz.iterrows():
        p, t = int(d.peak_i), int(d.trough_i)
        m = {"P-10": max(p - 10, 0), "P-5": max(p - 5, 0), "P": p, "d5": first_dd(nav, p, .05, t),
             "d10": first_dd(nav, p, .10, t), "T": t}
        marks[d.peak] = m
        for col in DIRECTION:
            if col not in panel:
                continue
            v = panel[col].to_numpy(float)
            rows.append(dict(decline=d.peak, trough=d.trough, depth=d.depth, signal=col,
                             **{k: (v[i] if i is not None else np.nan) for k, i in m.items()}))
    table = pd.DataFrame(rows)
    table.to_csv(CACHE / "regime_event_table.csv", index=False)
    return table, marks


def alarm_stats(ctx, base, panel, marks, hells, zz):
    """Loosest threshold that is in alarm at some close in [P, d10] of every hell decline; then its
    alarms in the rest of 2026. In-sample by construction (the threshold is read off the declines)."""
    days, nav = ctx.days, base.usd.to_numpy()
    nxt = np.r_[nav[1:] / nav[:-1] - 1, 0.]          # close d -> close d+1
    op = ctx.baseline.value                           # base NAV at each open
    inside = np.zeros(len(days), bool)
    for h in hells:
        inside[days.index(h[0]):days.index(h[1]) + 1] = True
    any_decl = np.zeros(len(days), bool)
    for _, d in zz.iterrows():
        any_decl[int(d.peak_i):int(d.trough_i) + 1] = True
    rows = []
    for col, sign in DIRECTION.items():
        if col not in panel:
            continue
        v = panel[col].to_numpy(float) * sign          # high = alarm
        thr_h = []
        for h in hells:
            m = marks[h[0]]
            seg = v[m["P"]:m["d10"] + 1]
            thr_h.append(np.nanmax(seg) if np.isfinite(seg).any() else np.nan)
        if not np.isfinite(thr_h).all():
            rows.append(dict(signal=col, note="no data in a hell window"))
            continue
        thr = min(thr_h)
        on = np.nan_to_num(v, nan=-np.inf) >= thr - 1e-12
        onset = on & ~np.r_[False, on[:-1]]
        rec = dict(signal=col, direction="high" if sign > 0 else "low", threshold=thr * sign,
                   valid_sessions=int(np.isfinite(v).sum()), on_sessions=int(on.sum()),
                   onsets_total=int(onset.sum()), false_onsets=int((onset & ~inside).sum()),
                   false_onsets_outside_any_8pct_decline=int((onset & ~any_decl).sum()),
                   false_on_sessions=int((on & ~inside).sum()),
                   base_logret_next_on_outside=float(np.log1p(nxt[on & ~inside]).sum()),
                   base_logret_next_on_inside=float(np.log1p(nxt[on & inside]).sum()))
        for h in hells:
            m = marks[h[0]]
            first = next((k for k in range(max(m["P"] - 10, 0), m["T"] + 1) if on[k]), None)
            tag = h[0][5:]
            rec[f"first_on_{tag}"] = days[first] if first is not None else None
            rec[f"sessions_vs_peak_{tag}"] = (first - m["P"]) if first is not None else np.nan
            ok = first is not None and first + 1 < len(days)
            rec[f"dd_at_fill_{tag}"] = (op[first + 1] / nav[m["P"]] - 1) if ok else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "regime_alarm_stats.csv", index=False)
    return out


def separation(table, hell_peaks, mark="d5"):
    """Does the value at the first -5% (or -10%) close separate hell declines from the V-shaped ones?"""
    rows = []
    for sig, g in table.groupby("signal"):
        g = g.dropna(subset=[mark])
        s = DIRECTION[sig]
        h = g[g.decline.isin(hell_peaks)][mark] * s
        v = g[~g.decline.isin(hell_peaks)][mark] * s
        if len(h) < len(hell_peaks) or len(v) < 3:
            continue
        rows.append(dict(signal=sig, mark=mark, n_v=len(v), hell_values=[round(float(x * s), 4) for x in h],
                         v_min=(v * s).min(), v_max=(v * s).max(), v_median=(v * s).median(),
                         separates=bool(h.min() > v.max()), separates_wrong_way=bool(h.max() < v.min()),
                         v_as_alarming=[int((v >= x).sum()) for x in h]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) family "tabsig": tab and fund-manager series as gates on the base plan
# ---------------------------------------------------------------------------

def tabsig_grid():
    """(signal column, op, threshold). Grid written after the panels around both declines had been
    printed (hindsight in the choice of series and scale); values are round numbers, not fitted."""
    grid = []
    for n in (5, 10, 20):                                   # TIME real trades, NAV %p summed over n PDFs
        grid += [(f"time_net_semis_{n}", "<=", -x) for x in (1., 2., 4.)]
        grid += [(f"time_net_all_{n}", "<=", -x) for x in (1., 2., 4.)]
        grid += [(f"koact_net_semis_{n}", "<=", -x) for x in (2., 4., 8.)]
        grid += [(f"time_semis_w_chg{n}", "<=", -x) for x in (2., 4.)]
    grid += [("time_cash_w", ">=", x) for x in (3., 4., 5.)]
    grid += [(f"time_cash_w_chg{n}", ">=", x) for n in (5, 10) for x in (1., 2.)]
    grid += [("koact_cash_w", ">=", x) for x in (2., 4.)]
    grid += [(f"koact_semis_w_chg{n}", "<=", -x) for n in (5, 10) for x in (3., 6.)]
    grid += [(f"both_net_semis_{n}", "<=", 0.) for n in (5, 10, 20)]     # both funds net sellers of semis
    grid += [("breadth", "<=", x) for x in (10., 15., 19.)]
    grid += [("breadth_chg10", "<=", x) for x in (-8., -15.)]
    grid += [("trend_SA_share", "<=", x) for x in (.20, .23)]
    grid += [("trend_D_share", ">=", x) for x in (.25, .30)]
    grid += [("held_trend_mean", "<=", x) for x in (2.5, 3.0)]
    grid += [("time_us_above_sma50", "<=", x) for x in (.40, .50)]
    grid += [("held_corr20", ">=", x) for x in (.60, .70)]
    grid += [("time_us_corr20", ">=", x) for x in (.25, .30)]
    grid += [("base_vol20", ">=", x) for x in (.60, .70)]
    return grid


def alarm_of(panel, column, op, threshold):
    if column.startswith("both_net_semis_"):
        n = column.rsplit("_", 1)[1]
        a, b = panel[f"time_net_semis_{n}"].to_numpy(float), panel[f"koact_net_semis_{n}"].to_numpy(float)
        return (a < threshold) & (b < threshold)             # NaN compares False
    v = panel[column].to_numpy(float)
    return (v <= threshold) if op == "<=" else (v >= threshold)


def run_gates(ctx, base, panel):
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    curves, configs = {"base": base}, {"base": dict(signal="none", op="", threshold=np.nan, level=1., source="base")}
    for column, op, thr in tabsig_grid():
        alarm = alarm_of(panel, column, op, thr)
        source = ("fund" if column.split("_")[0] in ("time", "koact", "both") and "above" not in column
                  and "corr" not in column else "tab")
        for level in (0., .5):
            frac = np.ones(len(ctx.days))
            frac[1:] = np.where(alarm[:-1], level, 1.)        # alarm known at close d acts at the open of d+1
            key = f"{column}{'le' if op == '<=' else 'ge'}{thr:g}_x{level:g}"
            curves[key] = common.replay_frac(*args, ctx.weekly, frac)
            configs[key] = dict(signal=column, op=op, threshold=thr, level=level, source=source,
                                alarm_sessions=int(alarm.sum()), alarm_onsets=int((alarm & ~np.r_[False, alarm[:-1]]).sum()))
    check = common.replay_frac(*args, ctx.weekly, np.ones(len(ctx.days)))
    np.testing.assert_allclose(check.to_numpy(), base.to_numpy(), rtol=1e-12, atol=1e-12)
    common.save("tabsig", curves, configs)
    table = common.stats_table(curves)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    out = cfg.join(table)
    b = table.loc["base"]
    for test in ("h2", "sep_nov", "oct_nov"):
        ctl = [common.matched_control(base, e, test) for e in table[f"{test}_expo"]]
        out[f"{test}_ctl_usd"] = [c[0] for c in ctl]
        out[f"{test}_ctl_usd_mdd"] = [c[1] for c in ctl]
        out[f"{test}_beats_base"] = (table[f"{test}_usd"] > b[f"{test}_usd"]) & (table[f"{test}_usd_mdd"] > b[f"{test}_usd_mdd"])
        out[f"{test}_beats_ctl"] = (table[f"{test}_usd"] > out[f"{test}_ctl_usd"]) & (table[f"{test}_usd_mdd"] > out[f"{test}_ctl_usd_mdd"])
    out.to_csv(CACHE / "regime_tabsig_stats.csv")
    picks = []
    for train, test in (("h1", "h2"), ("jan_aug", "sep_nov")):
        for currency in ("krw", "usd"):
            for rule, key in common.select(table, train, currency).items():
                r = out.loc[key]
                picks.append(dict(protocol=f"{train}->{test}", currency=currency, rule=rule, pick=key,
                                  train_ret=r[f"{train}_{currency}"], train_mdd=r[f"{train}_{currency}_mdd"],
                                  test_krw=r[f"{test}_krw"], test_usd=r[f"{test}_usd"], test_usd_mdd=r[f"{test}_usd_mdd"],
                                  test_expo=r[f"{test}_expo"], ctl_usd=r[f"{test}_ctl_usd"], ctl_usd_mdd=r[f"{test}_ctl_usd_mdd"],
                                  base_test_usd=b[f"{test}_usd"], base_test_usd_mdd=b[f"{test}_usd_mdd"]))
    picks = pd.DataFrame(picks)
    picks.to_csv(CACHE / "regime_tabsig_picks.csv", index=False)
    return out, picks


def controls(ctx, base):
    """Constant exposure and semiconductor-cap variants of the base (diagnostics, not in the family)."""
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    curves = {"base": base}
    for scale in (.75, .5, .25):
        curves[f"const_{scale:g}"] = study.replay(*args, ctx.weekly, ctx.gates["none"], scale)
    saved = dict(desk_config.RULE)
    try:
        for cap in (.0, .25, .35, .60):
            desk_config.RULE["semis_cap"] = cap                # in-memory only; restored below
            targets = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok["any"], "week", "rotate")
            curves[f"semis_cap_{cap:g}"] = study.replay(*args, targets, ctx.gates["none"])
        desk_config.RULE["semis_cap"] = saved["semis_cap"]
        for cap in (.15, .20):
            desk_config.RULE["stock_cap"] = cap
            targets = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok["any"], "week", "rotate")
            curves[f"stock_cap_{cap:g}"] = study.replay(*args, targets, ctx.gates["none"])
    finally:
        desk_config.RULE.clear()
        desk_config.RULE.update(saved)
    again = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok["any"], "week", "rotate")
    np.testing.assert_allclose(study.replay(*args, again, ctx.gates["none"]).to_numpy(), base.to_numpy(), rtol=1e-12)
    table = common.stats_table(curves)
    table.to_csv(CACHE / "regime_controls.csv")
    return table


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    ctx = common.context()
    base = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)
    hold = holdings_replay(ctx)
    np.testing.assert_allclose(hold["nav_close"], base.usd.to_numpy(), rtol=1e-10)
    if what in ("declines", "all"):
        zz, hw = declines(ctx, base)
        print("== zigzag declines >= 8% (USD closes)\n", zz.drop(columns=["peak_i", "trough_i"]).round(4).to_string())
        print("== high-water-mark episodes >= 8%\n", hw.round(4).to_string())
    if what in ("attrib", "all"):
        res = run_attribution(ctx, base, hold)
        for name, r in res.items():
            print(f"\n== attribution {name}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["summary"].items()})
            print(pd.DataFrame(r["by_group"]).T.to_string())
            print(pd.DataFrame(r["worst"]).to_string())
            print(pd.DataFrame(r["best"]).to_string())
        sp = spells(ctx, hold)
        print("\n== spells\n", sp.round(4).to_string())
    if what in ("signals", "all"):
        zz, hw = declines(ctx, base)
        panel, pdfs = build_panel(ctx, base, hold)
        table, marks = event_table(ctx, base, panel, zz)
        for peak, m in marks.items():
            print(peak, {k: (ctx.days[i] if i is not None else None) for k, i in m.items()})
        hells = [("2026-06-22", "2026-07-29"), ("2026-09-22", "2026-11-17")]
        # the 9/22 decline's zigzag trough is 10/16; alarms are judged against the whole stretch
        marks_h = {h[0]: dict(marks[h[0]], T=ctx.days.index(h[1])) for h in hells}
        alarms = alarm_stats(ctx, base, panel, marks_h, hells, zz)
        print("\n== alarm stats (threshold = loosest that fires within [P, d10] of both hell declines)")
        print(alarms.round(4).to_string())
        show = table[table.decline.isin([h[0] for h in hells] + ["2026-07-22", "2026-10-28"])]
        print("\n== event table (hell declines and their second legs)")
        print(show.drop(columns=["trough", "depth"]).round(4).to_string())
        hell_peaks = ["2026-06-22", "2026-09-22"]
        legs = table[~table.decline.isin(["2026-07-22", "2026-10-28"])]   # second legs are not independent dips
        for mark in ("P", "d5", "d10"):
            sep = separation(legs, hell_peaks, mark)
            sep.to_csv(CACHE / f"regime_separation_{mark}.csv", index=False)
            print(f"\n== separation at {mark}: {int(sep.separates.sum())} of {len(sep)} series separate; "
                  f"{int(sep.separates_wrong_way.sum())} the wrong way")
            print(sep.round(4).to_string())
    if what in ("gates", "all"):
        panel, _ = build_panel(ctx, base, hold)
        out, picks = run_gates(ctx, base, panel)
        print(f"\n== tabsig: {len(out)} candidates incl. base ({len(tabsig_grid())} signals x 2 levels)")
        for test in ("h2", "sep_nov", "oct_nov"):
            cand = out.drop(index="base")
            print(test, "beat base (ret & mdd, USD):", int(cand[f"{test}_beats_base"].sum()), "of", len(cand),
                  "| beat exposure-matched control:", int(cand[f"{test}_beats_ctl"].sum()))
        print(picks.round(4).to_string())
        cols = ["source", "alarm_sessions", "alarm_onsets", "h1_usd", "h1_usd_mdd", "jul_aug_usd", "h2_usd", "h2_usd_mdd",
                "h2_expo", "h2_ctl_usd", "sep_nov_usd", "sep_nov_usd_mdd", "sep_nov_ctl_usd", "oct_nov_usd", "full_usd", "full_krw",
                "h2_beats_ctl", "sep_nov_beats_ctl"]
        print(out[cols].round(3).to_string())
        ctl = controls(ctx, base)
        print("\n== controls\n", ctl[[c for c in ctl.columns if c.endswith(("_usd", "_usd_mdd", "_expo")) or c in ("full_krw", "h1_krw", "h2_krw")]].round(3).T.to_string())
