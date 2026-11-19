"""Family "hourly": can acting on 60-minute bars (earlier market exits, resting stops, hourly rotation)
improve the weekly 7-stock rule?

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_hourly.py

Rules : T1 SOXX/QQQ versus its 20/50/100-hour mean (decided at the daily close, the 09:00 ET pre-market
        price, 10:30 ET, every hourly close) with daily-close references; T2 per-stock resting stops and a
        SOXX-drop basket stop; T3 hourly-momentum rotation; T4 picked feasible rules x SOXX10r1 / SOXX20r3.
Label : "feasible" = traded at the 09:30 open on any session, or at 10:30 ET on EDT sessions
        (2026-03-09..10-30, 23:30 KST), or a resting stop order. Everything else "needs automation".

Data  : yfinance 60-minute bars (prepost=True) frozen in cache/hourly/<SYM>.parquet for the 86 book
        tickers + SOXX, QQQ, SOXS, SOXL. Regular bars start 09:30, 10:30 ... 15:30 ET (7 per session).
Grid  : slot k=0 trades at the official open of the frozen book; slots k=1..6 trade at the open of the
        hourly bar that starts at 10:30 ... 15:30 ET; NAV is marked at the official book close.
        The 10:00 / 11:00 ET decision times of the brief do not exist on this grid: the nearest
        boundary is 10:30 ET, which the user can trade live only on EDT sessions (24:00 KST = 11:00 ET).
Timing: a decision executed at slot k uses hourly closes through bar k-1 of the same session
        (k=0: closes through the previous session, plus the cleaned pre-market price as of 09:00 ET).
Ledger: standard five columns at zero cost (krw10 = 10bp one way), plus usd_hs1 / usd_hs2 (half-spread
        5bp or 15bp per trade, and double) written to cache/hourly_spread.parquet.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import study  # noqa: E402

HOURLY = common.CACHE / "hourly"
EXTRA = ["SOXX", "QQQ", "SOXS", "SOXL"]
K = 7                         # regular hourly bars per session
T_BUMP = 150                  # look-ahead self-test: prices after this session index are multiplied by 1.5
TOL = 0.007                   # reconciliation tolerance of the hourly close against the book close
ET = ZoneInfo("America/New_York")
SLOT_ET = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
LEDGER = common.LEDGER
GATES = ("SOXX10r1", "SOXX20r3")
DECLINES = {"dd_jul": ("2026-06-23", "2026-07-29"), "dd_oct": ("2026-10-13", "2026-11-17")}
REPORT_WINDOWS = ("full", "h1", "h2", "jan_aug", "sep_nov", "dd_jul", "dd_oct")


# ----------------------------------------------------------------------------- data
def fetch(symbols):
    """Download missing symbols once; later runs read the frozen parquet files."""
    missing = [s for s in symbols if not (HOURLY / f"{s}.parquet").exists()]
    if not missing:
        return
    import yfinance as yf
    HOURLY.mkdir(parents=True, exist_ok=True)
    for s in missing:
        kw = dict(interval="1h", prepost=True, auto_adjust=False, progress=False, multi_level_index=False)
        f = yf.download(s, period="730d", **kw)
        if f is None or f.empty:      # names listed <730 days ago by Yahoo's own start date
            f = yf.download(s, start="2025-06-01", **kw)
        f = f[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        f.index = f.index.tz_convert("America/New_York")
        f = f[f.index >= "2025-06-01"]
        f.columns.name = None
        if f.empty:
            raise RuntimeError(f"No hourly data for {s}")
        f.to_parquet(HOURLY / f"{s}.parquet")
        time.sleep(0.5)


def daily_hilo(tickers):
    """Official daily High/Low used only to clip bad hourly prints (frozen copy inside cache/hourly)."""
    path = HOURLY / "_daily_hilo.parquet"
    if path.exists():
        return pd.read_parquet(path)
    src = HERE / "frozen_ohlc" / "ohlc.csv"
    if src.exists():
        d = pd.read_csv(src)
        d = d[d.ticker.isin(tickers)][["date", "ticker", "high", "low"]]
    else:
        import yfinance as yf
        raw = yf.download(tickers, start="2025-12-01", end="2026-11-18", auto_adjust=False,
                          progress=False, group_by="ticker")
        d = pd.concat([raw[t][["High", "Low"]].dropna().rename(columns=str.lower).assign(ticker=t)
                       for t in tickers]).reset_index(names="date")
        d["date"] = pd.to_datetime(d.date).dt.strftime("%Y-%m-%d")
    d = d[d.date <= "2026-11-17"].reset_index(drop=True)
    d.to_parquet(path, index=False)
    return d


def bars(sym, dpos):
    """Regular-session arrays [D, K] (Open, High, Low, Close, Volume) and the pre-market frame."""
    f = pd.read_parquet(HOURLY / f"{sym}.parquet")
    mins = np.asarray(f.index.hour * 60 + f.index.minute)
    day = np.array([dpos.get(x, -1) for x in f.index.strftime("%Y-%m-%d")])
    reg = (mins >= 570) & (mins < 960) & ((mins - 570) % 60 == 0) & (day >= 0)
    out = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        a = np.full((len(dpos), K), np.nan)
        a[day[reg], (mins[reg] - 570) // 60] = f[field].to_numpy()[reg]
        out[field] = a
    pre = (mins < 570) & (day >= 0)
    out["pre"] = pd.DataFrame({"d": day[pre], "m": mins[pre], "low": f.Low.to_numpy()[pre],
                               "close": f.Close.to_numpy()[pre]})
    return out


def premarket(pre, D):
    """Cleaned pre-market values per session.

    pm0900: close of the 08:00 bar (price as of 09:00 ET), accepted only when the 07:00 bar also exists
            and the two closes agree within 1.5% (a single bad print cannot set the price).
    pmlow : lowest level at which two consecutive pre-market bars (04:00..09:00) both closed
            (min over neighbouring pairs of the higher close); known by 09:30 ET.
    """
    pm, low = np.full(D, np.nan), np.full(D, np.nan)
    for d, g in pre.groupby("d"):
        c = dict(zip(g.m, g.close))
        if 480 in c and 420 in c and abs(c[480] / c[420] - 1) <= 0.015:
            pm[d] = c[480]
        ms = sorted(c)
        pairs = [max(c[a], c[b]) for a, b in zip(ms, ms[1:]) if b - a == 60 or (a, b) == (540, 570)]
        if pairs:
            low[d] = min(pairs)
    return pm, low


def build_market(ctx, bump=None):
    """All price arrays the engine and the signals need. bump=T multiplies every price after session T by 1.5."""
    book = ctx.book
    days_all = list(book.days)
    D, J, start = len(days_all), len(ctx.days), ctx.start
    dpos = {d: i for i, d in enumerate(days_all)}
    tick = ctx.tickers
    N = len(tick)
    mult = np.ones(D)
    if bump is not None:
        mult[start + bump + 1:] = 1.5
    bo = book.opn.loc[days_all, tick].to_numpy() * mult[:, None]
    bc = book.cls.loc[days_all, tick].to_numpy() * mult[:, None]
    hilo = daily_hilo(tick)
    dlo = hilo.pivot(index="date", columns="ticker", values="low").reindex(index=days_all, columns=tick).to_numpy() * mult[:, None]
    dhi = hilo.pivot(index="date", columns="ticker", values="high").reindex(index=days_all, columns=tick).to_numpy() * mult[:, None]

    HO, HL, HC = (np.full((D, K, N), np.nan) for _ in range(3))
    dollar = np.full(N, np.nan)
    recon = []
    for i, s in enumerate(tick):
        b = bars(s, dpos)
        HO[:, :, i], HL[:, :, i], HC[:, :, i] = b["Open"], b["Low"], b["Close"]
        dv = np.nansum(b["Volume"] * b["Close"], axis=1)[start:]
        dollar[i] = np.nanmedian(np.where(dv > 0, dv, np.nan))
        have = np.isfinite(bo[start:, i] / mult[start:])
        do = np.abs(b["Open"][start:, 0] / (bo[start:, i] / mult[start:]) - 1)
        dc = np.abs(b["Close"][start:, K - 1] / (bc[start:, i] / mult[start:]) - 1)
        full = np.isfinite(b["Close"][start:]).all(axis=1)
        recon.append(dict(ticker=s, book_sessions=int(have.sum()), no_hourly=int((have & ~np.isfinite(b["Close"][start:]).any(axis=1)).sum()),
                          incomplete=int((have & ~full).sum()), open_max=float(np.nanmax(do)), open_median=float(np.nanmedian(do)),
                          open_gt_tol=int((do > TOL).sum()), close_max=float(np.nanmax(dc)), close_median=float(np.nanmedian(dc)),
                          close_gt_tol=int((dc > TOL).sum())))
    HO, HL, HC = (a * mult[:, None, None] for a in (HO, HL, HC))
    complete = np.isfinite(HC).all(axis=1)
    usable = complete & (np.abs(HC[:, K - 1, :] / bc - 1) <= TOL)             # [D, N]
    # A whole-market data hole (SOXX itself incomplete) disables intraday decisions on that session.
    lo_b, hi_b = dlo[:, None, :], dhi[:, None, :]
    clip = np.isfinite(lo_b) & np.isfinite(hi_b)
    HO = np.where(clip, np.clip(HO, lo_b, hi_b), HO)
    HC = np.where(clip, np.clip(HC, lo_b, hi_b), HC)
    HL = np.where(clip, np.maximum(HL, lo_b), HL)
    HC[:, K - 1, :] = np.where(usable, bc, HC[:, K - 1, :])                  # official close on the last bar

    # slot prices for the 2026 sessions
    SO = np.full((J, K, N), np.nan)
    stale = np.zeros((J, K, N), dtype=bool)
    SO[:, 0, :] = bo[start:]
    for k in range(1, K):
        px = HO[start:, k, :]
        alt = np.where(np.isfinite(HC[start:, k - 1, :]), HC[start:, k - 1, :], SO[:, k - 1, :])
        bad = ~np.isfinite(px)
        SO[:, k, :] = np.where(bad, alt, px)
        stale[:, k, :] = (bad | ~usable[start:]) & np.isfinite(SO[:, k, :])
    BL = HL[start:].copy()
    un = ~usable[start:]
    BL = np.where(un[:, None, :], np.nan, BL)                               # unusable session: one daily bar
    BL[:, 0, :] = np.where(un, dlo[start:], BL[:, 0, :])
    X = pd.DataFrame(HC.reshape(D * K, N)).ffill()                           # continuous hourly closes
    ma20 = X.rolling(20).mean()
    above20 = (X >= ma20).to_numpy() & np.isfinite(ma20.to_numpy())

    sig = {}
    for s in ("SOXX", "QQQ"):
        b = bars(s, dpos)
        off = book.cls[s].reindex(days_all).to_numpy() * mult
        c = b["Close"] * mult[:, None]
        ok = np.isfinite(c).all(axis=1) & (np.abs(c[:, K - 1] / off - 1) <= TOL)
        c[:, K - 1] = np.where(ok, off, c[:, K - 1])
        pm, pmlow = premarket(b["pre"], D)
        do = np.abs(b["Open"][start:, 0] / book.opn[s].reindex(days_all).to_numpy()[start:] - 1)
        dc = np.abs(b["Close"][start:, K - 1] / book.cls[s].reindex(days_all).to_numpy()[start:] - 1)
        recon.append(dict(ticker=s, book_sessions=J, no_hourly=0, incomplete=int((~np.isfinite(b["Close"][start:]).all(axis=1)).sum()),
                          open_max=float(np.nanmax(do)), open_median=float(np.nanmedian(do)), open_gt_tol=int((do > TOL).sum()),
                          close_max=float(np.nanmax(dc)), close_median=float(np.nanmedian(dc)), close_gt_tol=int((dc > TOL).sum())))
        sig[s] = SimpleNamespace(close=c, low=b["Low"] * mult[:, None], has=np.isfinite(b["Close"]), flat=pd.Series(c.reshape(-1)).ffill(),
                                 pm=pm * mult, pmlow=pmlow * mult, daily=pd.Series(off, index=days_all), ok=ok)
    edt = np.array([datetime(int(d[:4]), int(d[5:7]), int(d[8:]), 12, tzinfo=ET).utcoffset().total_seconds() == -4 * 3600
                    for d in ctx.days])
    hs = np.where(dollar >= 1e9, 0.0005, 0.0015)                             # half-spread: 5bp if median day >= $1bn traded
    gates = study.market_gates(SimpleNamespace(cls=book.cls.mul(pd.Series(mult, index=days_all), axis=0)), ctx.days)
    return SimpleNamespace(SO=SO, SOm=np.nan_to_num(SO), BL=BL, C=bc[start:], Cm=np.nan_to_num(bc[start:]), fx=ctx.fx, hs=hs,
                           stale=stale, usable=usable, X=X.to_numpy(), above20=above20, sig=sig, edt=edt, gates=gates,
                           start=start, D=D, J=J, N=N, dollar=dollar, recon=pd.DataFrame(recon).set_index("ticker"),
                           HO0=HO[start:, 0, :], HCraw_last=None, omega=0.5, ones=np.ones((J, K)), days=ctx.days,
                           market_ok=sig["SOXX"].has[start:])


# ----------------------------------------------------------------------------- engine
def engine(M, targets, F=None, stop=None):
    """Share-level replay on the 7-slot grid.

    targets : {(j, k): weight vector}; the shadow basket is rebalanced there (study.replay semantics).
    F       : [J, K] basket exposure 0/1 decided before slot k; real holdings follow F x shadow weights.
    stop    : None or dict(x=, mode="week"|"ma20"); per-stock resting stop x below the highest official
              close since entry (level fixed before the session, entry day: entry price), filled at
              min(bar open, stop). mode week: stopped names return at the next published target;
              mode ma20: they return when M.re_ok[j, k] is true (hourly close above its 20-bar mean).
    """
    SO, SOm, BL, C, Cm, fx, hs = M.SO, M.SOm, M.BL, M.C, M.Cm, M.fx, M.hs
    J, _, N = SO.shape
    F = M.ones if F is None else F
    shares, shadow = np.zeros(N), np.zeros(N)
    cash = reserve = 1.0
    d10 = d1 = d2 = 1.0
    use_stop = stop is not None
    if use_stop:
        x, mode = stop["x"], stop["mode"]
        held, ref = np.ones(N, dtype=bool), np.zeros(N)
    prev_f, carried, stale_turn = F[0, 0], 0.0, 0.0
    out = np.empty((J, 8))
    for j in range(J):
        turn = inv = 0.0
        active = (shares != 0) | (shadow != 0)
        if not np.isfinite(SO[j, 0][active]).all() or not np.isfinite(C[j][active]).all():
            raise ValueError(f"Held security missing market price on {M.days[j]}")
        if use_stop:
            lvl = np.where(shares > 0, (1 - x) * ref, -np.inf)
        for k in range(K):
            o = SOm[j, k]
            f = F[j, k]
            trade = (j, k) in targets
            nav = cash + shares @ o
            if k == 0:
                nav_open = nav
            if not trade and f == prev_f and not use_stop:
                inv += 1 - cash / nav
                continue
            dws = None
            if use_stop and k == 0:
                if trade and mode == "week":
                    held[:] = True
                gap = (shares > 0) & (o <= lvl)
                if gap.any():                                   # opened through the stop: sold at the open
                    dws = np.where(gap, shares * o / nav, 0.0)
                    cash += (shares[gap] * o[gap]).sum()
                    shares = np.where(gap, 0.0, shares)
                    held[gap] = False
            if trade:
                w = targets[(j, k)]
                if not (SO[j, k][w > 0] > 0).all():
                    raise ValueError(f"Missing executable price on {M.days[j]} slot {k}")
                snav = reserve + shadow @ o
                if use_stop:
                    held[w <= 0] = True
                shadow = np.divide(snav * w, o, out=np.zeros(N), where=w > 0)
                reserve = snav * (1 - w.sum())
            dw = None
            if (f > 0 and trade) or f != prev_f:
                snav = reserve + shadow @ o
                wt = f * (shadow * o / snav)
                if use_stop:
                    wt = wt * held
                new = np.divide(nav * wt, o, out=np.zeros(N), where=wt > 0)
                dw = np.abs(wt - shares * o / nav)
                if use_stop:
                    ent = (new > 0) & (shares == 0)
                    ref[ent] = o[ent]
                    lvl = np.where(ent, (1 - x) * o, np.where(new > 0, lvl, -np.inf))
                shares, cash = new, nav * (1 - wt.sum())
            elif use_stop and mode == "ma20" and f > 0:
                re = ~held & (shadow > 0) & M.re_ok[j, k] & (o > 0)
                if re.any():
                    snav = reserve + shadow @ o
                    want = np.where(re, nav * f * shadow * o / snav, 0.0)
                    want *= min(1.0, max(cash, 0.0) / want.sum())
                    dw = want / nav
                    shares = shares + np.divide(want, o, out=np.zeros(N), where=want > 0)
                    cash -= want.sum()
                    held[re] = True
                    ref[re] = o[re]
                    lvl = np.where(re, (1 - x) * o, lvl)
            for part in (dws, dw):
                if part is not None:
                    t = part.sum()
                    c = part @ hs
                    turn += t
                    d10 *= 1 - 0.001 * t
                    d1 *= 1 - c
                    d2 *= 1 - 2 * c
                    stale_turn += part @ M.stale[j, k]
            prev_f = f
            inv += (shares @ o) / nav
            if use_stop:
                hit = (shares > 0) & (BL[j, k] <= lvl)
                if hit.any():                                    # resting stop filled inside the bar
                    fill = np.where(hit, np.minimum(o, lvl), 0.0)
                    part = shares * fill / nav
                    t, c = part.sum(), part @ hs
                    turn += t
                    d10 *= 1 - 0.001 * t
                    d1 *= 1 - c
                    d2 *= 1 - 2 * c
                    cash += (shares * fill).sum()
                    shares = np.where(hit, 0.0, shares)
                    held[hit] = False
        pos = shares @ Cm[j]
        usd = cash + pos
        if not np.isfinite(C[j][(shares != 0) | (shadow != 0)]).all():
            raise ValueError(f"Missing closing mark on {M.days[j]}")
        if use_stop:
            m = shares > 0
            ref[m] = np.maximum(ref[m], Cm[j][m])
        expo = M.omega * carried + (1 - M.omega) * inv / K
        carried = pos / usd
        out[j] = (usd * fx[j], usd, turn, expo, usd * fx[j] * d10, usd * d1, usd * d2, nav_open)
    frame = pd.DataFrame(out, index=M.days, columns=LEDGER + ["usd_hs1", "usd_hs2", "nav_open"])
    frame.attrs["stale_turn"] = float(stale_turn)
    return frame


# ----------------------------------------------------------------------------- signals
def walk(M, decide):
    """F[j, k] from decide(j, k, state) -> new state (0/1) or None to keep the previous one."""
    F = np.ones((M.J, K))
    state = 1.0
    for j in range(M.J):
        for k in range(K):
            new = decide(j, k, state)
            if new is not None:
                state = float(new)
            F[j, k] = state
    F[0, :] = 1.0   # nothing is held before the first target; keeps prev_f neutral
    return F


def f_hourly_ma(M, sym, n, sched):
    """T1: sym versus its n-hour mean of regular hourly closes.

    close: decided from the last hourly close of the previous session (the daily close), traded at the open.
    c1030f / c1030a: also re-decided from the first-hour close and traded at 10:30 ET (f: EDT sessions only).
    pm...: the 09:00 ET pre-market price replaces the previous close at the open; all: every hourly close.
    """
    s = M.sig[sym]
    flat = s.flat
    above = (flat >= flat.rolling(n).mean()).to_numpy()
    ma = flat.rolling(n).mean().to_numpy()
    slots = {"close": (), "c1030f": (1,), "c1030a": (1,), "pm": (), "pm1030f": (1,), "pm1030a": (1,), "all": (1, 2, 3, 4, 5, 6)}[sched]
    use_pm = sched in ("pm", "pm1030f", "pm1030a", "all")      # c* schedules: no pre-market check, only the first-hour close

    def decide(j, k, state):
        d = M.start + j
        last = (d - 1) * K + K - 1
        if k == 0:
            if use_pm and np.isfinite(s.pm[d]):
                return s.pm[d] >= ma[last]
            return above[last]
        if k in slots and s.has[d, k - 1] and (not sched.endswith("1030f") or M.edt[j]):
            return above[d * K + k - 1]
        return None
    return walk(M, decide)


def f_daily_sma(M, sym, n):
    """Daily-close gate with the same construction as study.market_gates (one close below exits)."""
    close = M.sig[sym].daily
    line = close.rolling(n).mean()
    above = (close >= line).where(close.notna() & line.notna())
    allowed = above.ffill().shift().reindex(M.days)
    if allowed.isna().any():
        raise ValueError("daily SMA warm-up")
    return np.repeat(allowed.to_numpy(dtype=float)[:, None], K, axis=1)


def f_soxx_drop(M, x, reentry, scope, week_slots):
    """T2 portfolio: SOXX trades x below yesterday's official close -> basket to cash at the next slot."""
    s = M.sig["SOXX"]
    gate = M.gates["SOXX10r1"]
    pending = {"v": False}

    def decide(j, k, state):
        d = M.start + j
        level = (1 - x) * s.daily.iloc[d - 1]
        if k == 0:
            new = state
            if pending["v"]:                       # triggered in yesterday's last bar
                new, pending["v"] = 0.0, False
            if new == 0 and (reentry == "day" or (reentry == "week" and j in week_slots) or (reentry == "gate10" and gate[j])):
                new = 1.0
            if np.isfinite(s.pm[d]) and s.pm[d] <= level:
                new = 0.0
            return new
        watched = scope == "all" or (k == 1 and M.edt[j])
        if watched and s.low[d, k - 1] <= level:
            return 0
        if scope == "all" and k == K - 1 and reentry != "day" and s.low[d, k] <= level:
            pending["v"] = True                    # last bar: can only act at the next open
        return None
    return walk(M, decide)


def t3_targets(M, ctx, L, n, sched):
    """T3: equal-weight top n of the PIT TIME universe by the return over the last L hourly closes."""
    X = M.X
    slots = {"open": (0,), "1030f": None, "1030a": None, "every2h": (0, 2, 4, 6)}[sched]
    targets, current = {}, None
    for j in range(1, M.J):
        cols = np.array([ctx.col[r["key"]] for r in ctx.rankings[j]], dtype=int)
        if not len(cols):
            continue
        d = M.start + j
        late = M.market_ok[j, 0] and (sched == "1030a" or M.edt[j])
        day_slots = slots if slots is not None else ((1,) if late else (0,))
        for k in day_slots:
            if k and not M.market_ok[j, k - 1]:
                continue
            t = (d - 1) * K + K - 1 if k == 0 else d * K + k - 1
            score = X[t, cols] / X[t - L, cols] - 1
            ok = np.isfinite(score) & (M.SO[j, k, cols] > 0)
            if k:
                ok &= ~M.stale[j, k, cols]
            if not ok.any():
                continue
            order = sorted(zip(-score[ok], cols[ok]))
            pick = frozenset(c for _, c in order[:n])
            if pick != current:
                w = np.zeros(M.N)
                w[list(pick)] = 1 / len(pick)
                targets[(j, k)] = w
                current = pick
    return targets


def re_ok(M):
    """ma20 re-entry mask: slot 0 from the previous session's last hourly close, slot 1 on EDT sessions."""
    ok = np.zeros((M.J, K, M.N), dtype=bool)
    for j in range(1, M.J):
        d = M.start + j
        ok[j, 0] = M.above20[(d - 1) * K + K - 1]
        if M.edt[j]:
            ok[j, 1] = M.above20[d * K] & M.usable[d]
    return ok


# ----------------------------------------------------------------------------- candidates
T1_SCHEDULES = ("close", "c1030f", "c1030a", "pm", "pm1030f", "pm1030a", "all")


def gate_f(M, name):
    return np.repeat(M.gates[name].astype(float)[:, None], K, axis=1)


def candidates(M, ctx, picks=None):
    """Every rule as key -> (targets, F, stop, config). Deterministic order. picks: {key: how} for T4."""
    weekly = {(j, 0): w for j, w in ctx.weekly.items()}
    week_slots = set(ctx.weekly)
    M.re_ok = re_ok(M)
    spec = {"base": (weekly, None, None, dict(subfamily="base", feasible="feasible", hourly=False, piece="base", gate="none"))}

    def add(key, targets, F, stop, **cfg):
        cfg.setdefault("gate", "none")
        cfg.setdefault("piece", key)
        cfg.setdefault("hourly", True)
        spec[key] = (targets, F, stop, cfg)

    for sym in ("SOXX", "QQQ"):                                                   # T1: 42 hourly + 10 daily references
        for n, nd in zip((20, 50, 100), (3, 7, 14)):
            for sched in T1_SCHEDULES:
                add(f"T1_{sym}{n}h_{sched}", weekly, f_hourly_ma(M, sym, n, sched), None, subfamily="T1_hourly_ma",
                    feasible="feasible" if sched in ("close", "c1030f", "pm", "pm1030f") else "needs automation",
                    sym=sym, horizon=n, schedule=sched)
            add(f"T1_{sym}{nd}d_daily", weekly, f_daily_sma(M, sym, nd), None, subfamily="T1_daily_ref",
                feasible="feasible", hourly=False, sym=sym, horizon=nd, schedule="daily_close")
        for nd in (10, 20):
            add(f"T1_{sym}{nd}r1_gate", weekly, gate_f(M, f"{sym}{nd}r1"), None, subfamily="T1_daily_ref",
                feasible="feasible", hourly=False, sym=sym, horizon=nd, schedule="daily_close", gate=f"{sym}{nd}r1")
    for x in (5, 8, 10, 12, 15):                                                  # T2: 5 per-stock resting stops (8, 12 asked)
        add(f"T2_stop{x}_week", weekly, None, dict(x=x / 100, mode="week"), subfamily="T2_stock_stop",
            feasible="feasible", x=x, schedule="week")
    for x in (3, 4, 5):                                                           # T2: 18 SOXX-drop basket stops
        for reentry in ("day", "week", "gate10"):
            for scope in ("edt", "all"):
                add(f"T2_soxx{x}_{reentry}_{scope}", weekly, f_soxx_drop(M, x / 100, reentry, scope, week_slots), None,
                    subfamily="T2_soxx_drop", feasible="feasible" if scope == "edt" else "needs automation",
                    x=x, schedule=reentry, scope=scope)
    for L in (7, 35, 70):                                                         # T3: 24 rotations
        for n in (5, 7):
            for sched in ("open", "1030f", "1030a", "every2h"):
                add(f"T3_L{L}_N{n}_{sched}", t3_targets(M, ctx, L, n, sched), None, None, subfamily="T3_rotation",
                    feasible="feasible" if sched in ("open", "1030f") else "needs automation",
                    horizon=L, x=n, schedule=sched)
    if picks:                                                                     # T4: picked feasible rules x 2 gates
        for key, how in picks.items():
            targets, F, stop, cfg = spec[key]
            for g in GATES:
                add(f"T4_{key[3:]}__{g}", targets, gate_f(M, g) if F is None else F * gate_f(M, g), stop,
                    **{**cfg, "subfamily": "T4_cross", "gate": g, "piece": key, "pick": how})
        for g in GATES:
            add(f"T4_base__{g}", weekly, gate_f(M, g), None, subfamily="T4_gate_only", feasible="feasible",
                hourly=False, gate=g, piece="base")
    return spec


def run_all(M, spec, done=None):
    done = {} if done is None else done
    for key, (targets, F, stop, _) in spec.items():
        if key not in done:
            done[key] = engine(M, targets, F, stop)
    return done


# ----------------------------------------------------------------------------- reporting helpers
def win(series, window):
    a, b = common.WINDOWS[window]
    ix = np.flatnonzero((series.index >= a) & (series.index <= b))
    before = series.iloc[ix[0] - 1] if ix[0] else 1.0
    v = np.r_[before, series.to_numpy()[ix]]
    return float(v[-1] / before - 1), float(np.min(v / np.maximum.accumulate(v) - 1))


def control(base_curve, expo, window):
    """Exposure-matched control return; a rule holding more than the base is compared with the base itself."""
    c = common.matched_control(base_curve, expo, window)[0]
    return win(base_curve.usd, window)[0] if np.isnan(c) else float(c)


def decline_edges(frame, base_curve, cost="usd_hs1"):
    out = {}
    for w in DECLINES:
        expo = common.window_stats(frame, *common.WINDOWS[w])["expo"]
        out[w] = win(frame[cost], w)[0] - control(base_curve, expo, w)
    return out


def choose(full, spec):
    """T4 pieces: three best feasible hourly rules on the H1 training window (Codex's return rule) and,
    labelled as hindsight, the three best by H2 USD after the 5/15bp half-spread."""
    keys = [k for k, v in spec.items() if v[3]["feasible"] == "feasible" and v[3]["hourly"]]
    st = common.stats_table({k: full[k][LEDGER] for k in keys + ["base"]})
    pool = [k for k in keys if st.loc[k, "h1_krw_mdd"] >= st.loc["base", "h1_krw_mdd"] - 1e-7]
    ranked = sorted(pool, key=lambda k: (-st.loc[k, "h1_krw"], k)) + sorted(keys, key=lambda k: (-st.loc[k, "h1_krw"], k))
    picks = {k: "train_h1" for k in list(dict.fromkeys(ranked))[:3]}
    h2 = pd.Series({k: win(full[k].usd_hs1, "h2")[0] for k in keys}).sort_values(ascending=False, kind="stable")
    for k in h2.index[:3]:
        picks[k] = "train_h1+hindsight_h2" if k in picks else "hindsight_h2"
    return picks


def shift_f(F, s):
    return np.r_[np.ones(s), F.reshape(-1)[:-s]].reshape(F.shape)


def robustness(M, spec, key, base_curve, real, real_h2):
    """Fill-delay and placebo checks for one rule (edges at the 5/15bp half-spread)."""
    targets, F, stop, cfg = spec[key]
    out = dict(strategy=key, jul_edge=real["dd_jul"], oct_edge=real["dd_oct"])
    for label, s in (("delay_1h", 1), ("delay_1d", K)):
        try:
            if F is not None:
                f = engine(M, targets, shift_f(F, s), stop)
            elif cfg["piece"].startswith("T3_") and stop is None:
                moved = {((j * K + k + s) // K, (j * K + k + s) % K): w for (j, k), w in targets.items() if j * K + k + s < M.J * K}
                f = engine(M, moved, None, None)
            else:
                continue
        except ValueError:
            continue
        e = decline_edges(f, base_curve)
        out[f"{label}_jul"], out[f"{label}_oct"] = e["dd_jul"], e["dd_oct"]
        out[f"{label}_h2"] = win(f.usd_hs1, "h2")[0]
    if F is not None and (F != 1).any():
        rows = []
        for s in range(PLACEBO_STEP, M.J - PLACEBO_STEP + 1, PLACEBO_STEP):     # same spells, moved in time
            f = engine(M, targets, np.roll(F, s, axis=0), stop)
            e = decline_edges(f, base_curve)
            rows.append((e["dd_jul"], e["dd_oct"], win(f.usd_hs1, "h2")[0]))
        p = np.array(rows)
        out.update(placebo_n=len(p), placebo_both_positive=float(((p[:, 0] > 0) & (p[:, 1] > 0)).mean()),
                   placebo_min_edge_ge_real=float((p[:, :2].min(axis=1) >= min(real.values())).mean()),
                   placebo_jul_ge_real=float((p[:, 0] >= real["dd_jul"]).mean()),
                   placebo_oct_ge_real=float((p[:, 1] >= real["dd_oct"]).mean()),
                   placebo_h2_ge_real=float((p[:, 2] >= real_h2).mean()),
                   placebo_h2_median=float(np.median(p[:, 2])))
    return out



def random_baskets(M, ctx, n, draws, seed):
    """T3 control: n names drawn at random from the same PIT universe at every weekly rebalance (zero cost),
    plus the equal-weight universe as row 0. Columns: full, h1, h2, dd_jul, dd_oct USD returns."""
    rng = np.random.default_rng(seed)
    pools = {j: np.array([ctx.col[r["key"]] for r in ctx.rankings[j]], dtype=int) for j in sorted(ctx.weekly)}
    pools = {j: c[M.SO[j, 0, c] > 0] for j, c in pools.items()}
    rows, skipped = [], 0
    for i in range(draws + 1):
        targets = {}
        for j, c in pools.items():
            w = np.zeros(M.N)
            pick = c if i == 0 else rng.choice(c, size=min(n, len(c)), replace=False)
            w[pick] = 1 / len(pick)
            targets[(j, 0)] = w
        try:
            f = engine(M, targets)
        except ValueError:
            skipped += 1
            continue
        rows.append([win(f.usd, w)[0] for w in ("full", "h1", "h2", "dd_jul", "dd_oct")])
    return np.array(rows), skipped


def show(title, frame, cols=None):
    f = frame if cols is None else frame[cols]
    print(f"\n== {title}")
    print(f.to_string(float_format=lambda v: f"{v:+.3f}"))


PLACEBO_STEP = 5
RANDOM_DRAWS = 300


def main():
    t0 = time.time()
    common.WINDOWS.update(DECLINES)     # this process only: the two decline windows for stats_table / matched_control
    ctx = common.context()
    fetch(ctx.tickers + EXTRA)
    M = build_market(ctx)
    weekly = {(j, 0): w for j, w in ctx.weekly.items()}

    # --- base reproduction
    ref = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    base0 = engine(M, weekly)
    on = np.log(base0.nav_open.to_numpy()[2:] / base0.usd.to_numpy()[1:-1])
    intr = np.log(base0.usd.to_numpy()[2:] / base0.nav_open.to_numpy()[2:])
    M.omega = float(on.var() / (on.var() + intr.var()))        # share of base variance earned overnight
    base = engine(M, weekly)
    for c in ("krw", "usd", "turn", "krw10"):
        np.testing.assert_allclose(base[c], ref[c], rtol=1e-10, atol=1e-12)
    book_rel = float(np.abs(base.usd / ref.usd - 1).max())
    expo_diff = float(np.abs(base.exposure - ref.exposure).iloc[2:].max())
    for name, g in M.gates.items():
        assert (g == ctx.gates[name]).all(), name
    for g in ("SOXX100r1", "SOXX10r1", "SOXX20r3"):
        gated = engine(M, weekly, gate_f(M, g))
        ref_g = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates[g])
        for c in ("krw", "usd", "turn", "krw10"):
            np.testing.assert_allclose(gated[c], ref_g[c], rtol=1e-10, atol=1e-12)
    # pure-hourly reconciliation: first hourly print as the open, last hourly close as the mark
    P = SimpleNamespace(**vars(M))
    us = M.usable[M.start:]
    raw_last = np.stack([bars(s, {d: i for i, d in enumerate(ctx.book.days)})["Close"][M.start:, K - 1] for s in ctx.tickers], axis=1)
    P.SO = M.SO.copy()
    P.SO[:, 0, :] = np.where(us & np.isfinite(M.HO0), M.HO0, M.SO[:, 0, :])
    P.SOm = np.nan_to_num(P.SO)
    P.C = np.where(us & np.isfinite(raw_last), raw_last, M.C)
    P.Cm = np.nan_to_num(P.C)
    pure = engine(P, weekly)
    rel_nav = float(np.abs(pure.usd / base.usd - 1).max())
    rel_ret = float(np.abs(pure.usd.pct_change() - base.usd.pct_change()).iloc[1:].max())

    # --- candidates (T1-T3, then T4 from the picked pieces)
    spec = candidates(M, ctx)
    full = run_all(M, spec)
    picks = choose(full, spec)
    spec = candidates(M, ctx, picks)
    full = run_all(M, spec, full)
    full = {k: full[k] for k in spec}
    np.testing.assert_allclose(full["base"].usd, ref.usd, rtol=1e-10)
    curves = {k: f[LEDGER].copy() for k, f in full.items()}
    configs = {k: dict(v[3]) for k, v in spec.items()}
    assert len(curves) - 1 <= 300
    for k, f in curves.items():
        assert f.exposure.between(-1e-9, 1 + 1e-9).all() and (f.usd > 0).all(), k
    common.save("hourly", curves, configs)
    spread = pd.concat({k: f[["usd_hs1", "usd_hs2"]] for k, f in full.items()}, names=["strategy", "date"]).reset_index()
    spread.to_parquet(common.CACHE / "hourly_spread.parquet", index=False)

    # --- look-ahead self-test: every price after session T_BUMP x1.5, ledgers through T_BUMP must not move
    M2 = build_market(ctx, bump=T_BUMP)
    M2.omega = M.omega
    spec2 = candidates(M2, ctx, picks)
    assert list(spec2) == list(spec)
    worst, moved = 0.0, 0
    for key, (targets, F, stop, _) in spec2.items():
        a, (t1, F1, _, _) = engine(M2, targets, F, stop), spec[key]
        assert {x for x in targets if x[0] <= T_BUMP} == {x for x in t1 if x[0] <= T_BUMP}, key
        for x in targets:
            if x[0] <= T_BUMP:
                assert np.array_equal(targets[x], t1[x]), key
        if F is not None:
            assert np.array_equal(F[:T_BUMP + 1], F1[:T_BUMP + 1]), key
        for c in ("usd", "turn", "exposure"):
            diff = np.abs(a[c].to_numpy()[:T_BUMP + 1] - full[key][c].to_numpy()[:T_BUMP + 1]).max()
            worst = max(worst, float(diff))
            assert diff < 1e-10, (key, c, diff)
        moved += abs(a.usd.iloc[-1] / full[key].usd.iloc[-1] - 1) > 1e-9

    # --- statistics
    st = common.stats_table(curves)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    T = st.join(cfg)
    bc = curves["base"]
    for w in REPORT_WINDOWS:
        T[f"{w}_usd_hs1"] = pd.Series({k: win(full[k].usd_hs1, w)[0] for k in st.index})
        T[f"{w}_usd_hs2"] = pd.Series({k: win(full[k].usd_hs2, w)[0] for k in st.index})
    for w in ("full", "h2", "dd_jul", "dd_oct"):
        T[f"{w}_ctrl"] = pd.Series({k: control(bc, st.loc[k, f"{w}_expo"], w) for k in st.index})
        for cost in ("", "_hs1", "_hs2"):
            T[f"{w}_edge{cost}"] = T[f"{w}_usd{cost}"] - T[f"{w}_ctrl"]
    for cost in ("", "_hs1", "_hs2"):
        T[f"both{cost}"] = (T[f"dd_jul_edge{cost}"] > 0) & (T[f"dd_oct_edge{cost}"] > 0)
    T["both_hs1_1pt"] = (T.dd_jul_edge_hs1 > 0.01) & (T.dd_oct_edge_hs1 > 0.01)
    T["beats_base_both_hs1"] = (T.dd_jul_usd_hs1 > T.loc["base", "dd_jul_usd_hs1"]) & (T.dd_oct_usd_hs1 > T.loc["base", "dd_oct_usd_hs1"])
    T["stale_turn"] = pd.Series({k: float(f.attrs["stale_turn"]) for k, f in full.items()})
    T.to_csv(common.CACHE / "hourly_stats.csv")
    R = T[T.index != "base"]
    hourly_feas = R[(R.feasible == "feasible") & R.hourly]

    rc = M.recon
    stock = rc.loc[ctx.tickers]
    held_any = np.zeros(M.N, dtype=bool)
    for w in ctx.weekly.values():
        held_any |= w > 0
    un = ~M.usable[M.start:] & np.isfinite(M.C)
    report = dict(n_candidates=len(curves) - 1, omega=M.omega, picks=picks, runtime_seconds=None,
                  base_reproduction=dict(book_prices_max_rel_error=book_rel, expo_diff_after_entry=expo_diff,
                                         hourly_prints_max_rel_nav_error=rel_nav, hourly_prints_max_abs_daily_return_error=rel_ret,
                                         full_usd_hourly_prints=win(pure.usd, "full")[0], full_usd_book=win(base.usd, "full")[0],
                                         h2_usd_hourly_prints=win(pure.usd, "h2")[0], h2_usd_book=win(base.usd, "h2")[0]),
                  selftest=dict(T=T_BUMP, candidates=len(spec2), worst_abs_diff_through_T=worst, final_nav_moved=int(moved)),
                  edt_sessions=int(M.edt.sum()), est_sessions=int((~M.edt).sum()),
                  feasible_1030_sessions_in=dict(dd_jul=int(sum(M.edt[j] for j, d in enumerate(ctx.days) if "2026-06-23" <= d <= "2026-07-29")),
                                                 dd_oct=int(sum(M.edt[j] for j, d in enumerate(ctx.days) if "2026-10-13" <= d <= "2026-11-17")),
                                                 dd_oct_total=int(sum("2026-10-13" <= d <= "2026-11-17" for d in ctx.days))),
                  half_spread=dict(n_5bp=int((M.hs == 0.0005).sum()), n_15bp=int((M.hs == 0.0015).sum())),
                  stale_turn_max=[float(T.stale_turn.max()), str(T.stale_turn.idxmax())])
    report["reconcile"] = dict(
        tickers=M.N, tickers_any_close_gt_tol=int((stock.close_gt_tol > 0).sum()), ticker_sessions_close_gt_tol=int(stock.close_gt_tol.sum()),
        tickers_any_open_gt_tol=int((stock.open_gt_tol > 0).sum()), ticker_sessions_open_gt_tol=int(stock.open_gt_tol.sum()),
        close_max=float(stock.close_max.max()), open_first_print_max=float(stock.open_max.max()),
        ticker_sessions_total=int(np.isfinite(M.C).sum()), ticker_sessions_unusable=int(un.sum()),
        tickers_any_unusable=int((un.sum(axis=0) > 0).sum()),
        unusable_by_session_top={ctx.days[j]: int(un[j].sum()) for j in np.argsort(-un.sum(axis=1))[:5]},
        unusable_by_ticker_top={ctx.tickers[i]: int(un[:, i].sum()) for i in np.argsort(-un.sum(axis=0))[:8]},
        tickers_gt5pct_unusable=[ctx.tickers[i] for i in range(M.N) if un[:, i].sum() > 0.05 * np.isfinite(M.C[:, i]).sum()],
        base_names_unusable_sessions=int(un[:, held_any].sum()), soxx=rc.loc["SOXX"].to_dict(), qqq=rc.loc["QQQ"].to_dict())

    def pick_row(key):
        r = T.loc[key]
        cols = [f"{w}_{m}" for w in REPORT_WINDOWS for m in ("krw", "usd", "usd_hs1", "usd_mdd", "expo", "turn")]
        return dict(strategy=key, feasible=r.feasible, subfamily=r.subfamily, **{c: float(r[c]) for c in cols})

    sel = []
    for scope, keys in (("all", list(T.index)), ("feasible", ["base"] + list(R.index[R.feasible == "feasible"])),
                        ("feasible_hourly", ["base"] + list(hourly_feas.index))):
        for train, test in (("h1", "h2"), ("jan_aug", "sep_nov")):
            for rule, key in common.select(st.loc[keys], train).items():
                r = T.loc[key]
                sel.append(dict(scope=scope, split=f"{train}->{test}", rule=rule, strategy=key, feasible=r.feasible,
                                train_krw=r[f"{train}_krw"], train_krw_mdd=r[f"{train}_krw_mdd"], test_krw=r[f"{test}_krw"],
                                test_usd=r[f"{test}_usd"], test_usd_hs1=r[f"{test}_usd_hs1"], test_usd_mdd=r[f"{test}_usd_mdd"],
                                base_test_usd=T.loc["base", f"{test}_usd"], base_test_usd_mdd=T.loc["base", f"{test}_usd_mdd"],
                                dd_jul_usd_hs1=r.dd_jul_usd_hs1, dd_oct_usd_hs1=r.dd_oct_usd_hs1, full_usd_hs1=r.full_usd_hs1))
    sel = pd.DataFrame(sel)
    report["selections"] = sel.to_dict("records")
    report["base"] = pick_row("base")

    med_cols = ["full_usd", "full_usd_hs1", "full_usd_hs2", "h2_usd", "h2_usd_hs1", "dd_jul_usd", "dd_jul_usd_hs1", "dd_oct_usd",
                "dd_oct_usd_hs1", "full_usd_mdd", "h2_usd_mdd", "full_expo", "h2_expo", "full_turn", "full_edge_hs1", "h2_edge_hs1",
                "dd_jul_edge_hs1", "dd_oct_edge_hs1"]
    sub = R.groupby(["subfamily", "feasible"])[med_cols].median()
    sub.insert(0, "n", R.groupby(["subfamily", "feasible"]).size())
    for c in ("both", "both_hs1", "both_hs2", "both_hs1_1pt", "beats_base_both_hs1"):
        sub[f"n_{c}"] = R.groupby(["subfamily", "feasible"])[c].sum()
    report["subfamily"] = sub.reset_index().to_dict("records")

    # T1: same indicator, different decision times (5/15bp half-spread)
    t1 = []
    for sym in ("SOXX", "QQQ"):
        for n in (20, 50, 100):
            for s in T1_SCHEDULES:
                r, c = T.loc[f"T1_{sym}{n}h_{s}"], T.loc[f"T1_{sym}{n}h_close"]
                t1.append(dict(sym=sym, hours=n, sched=s, **{w: r[f"{w}_usd_hs1"] for w in ("full", "h1", "h2", "dd_jul", "dd_oct")},
                               h2_mdd=r.h2_usd_mdd, turn=r.full_turn, h2_expo=r.h2_expo, jul_edge=r.dd_jul_edge_hs1, oct_edge=r.dd_oct_edge_hs1,
                               **{f"d_{w}": r[f"{w}_usd_hs1"] - c[f"{w}_usd_hs1"] for w in ("full", "h2", "dd_jul", "dd_oct")},
                               **{f"d0_{w}": r[f"{w}_usd"] - c[f"{w}_usd"] for w in ("full", "h2", "dd_jul", "dd_oct")}))
    t1 = pd.DataFrame(t1)
    report["t1_timing"] = t1.to_dict("records")
    dcols = [c for c in t1.columns if c.startswith("d_") or c.startswith("d0_")]
    t1m = t1[t1.sched != "close"].groupby("sched")[dcols].median()
    for w in ("full", "h2", "dd_jul", "dd_oct"):
        t1m[f"n_pos_{w}"] = t1[t1.sched != "close"].groupby("sched")[f"d_{w}"].apply(lambda v: int((v > 0).sum()))
    report["t1_timing_median_vs_close"] = t1m.reset_index().to_dict("records")
    ref_cols = ["full_usd_hs1", "h1_usd_hs1", "h2_usd_hs1", "dd_jul_usd_hs1", "dd_oct_usd_hs1", "h2_usd_mdd", "full_turn", "h2_expo",
                "dd_jul_edge_hs1", "dd_oct_edge_hs1"]
    t1ref = T.loc[T.subfamily == "T1_daily_ref", ref_cols]
    report["t1_daily_refs"] = t1ref.reset_index().to_dict("records")

    # T3: hourly ranking versus the weekly 120-session ranking (base)
    t3 = T.loc[T.subfamily == "T3_rotation", ["feasible", "full_usd", "full_usd_hs1", "full_usd_hs2", "h1_usd_hs1", "h2_usd_hs1",
                                                "dd_jul_usd_hs1", "dd_oct_usd_hs1", "full_usd_mdd", "full_turn"]]
    b = T.loc["base"]
    report["t3_beats_base"] = {f"{w}{cost}": int((T.loc[t3.index, f"{w}_usd{cost}"] > b[f"{w}_usd{cost}"]).sum())
                               for w in ("full", "h1", "h2", "dd_jul", "dd_oct") for cost in ("", "_hs1", "_hs2")}
    report["t3_n"] = len(t3)
    t3["d_full_hs1"], t3["d_h1_hs1"] = t3.full_usd_hs1 - b.full_usd_hs1, t3.h1_usd_hs1 - b.h1_usd_hs1
    rb, rnd = {}, []
    for n in (5, 7):
        arr, skipped = random_baskets(M, ctx, n, RANDOM_DRAWS, 20261119 + n)
        ew, arr = arr[0], arr[1:]
        rb[n] = arr
        rnd.append(dict(n=n, draws=len(arr), skipped=skipped, ew_universe_full=ew[0], ew_universe_h1=ew[1], ew_universe_h2=ew[2],
                        ew_universe_dd_jul=ew[3], ew_universe_dd_oct=ew[4],
                        **{f"median_{w}": float(np.median(arr[:, i])) for i, w in enumerate(("full", "h1", "h2", "dd_jul", "dd_oct"))},
                        share_beat_base_dd_jul=float((arr[:, 3] > b.dd_jul_usd).mean()), share_beat_base_dd_oct=float((arr[:, 4] > b.dd_oct_usd).mean()),
                        share_beat_base_both=float(((arr[:, 3] > b.dd_jul_usd) & (arr[:, 4] > b.dd_oct_usd)).mean()),
                        share_beat_base_h2=float((arr[:, 2] > b.h2_usd).mean()), share_beat_base_full=float((arr[:, 0] > b.full_usd).mean())))
    rnd = pd.DataFrame(rnd).set_index("n")
    report["t3_random_baskets"] = rnd.reset_index().to_dict("records")
    for i, w in enumerate(("full", "h1", "h2", "dd_jul", "dd_oct")):          # share of random baskets at least as good as the rule
        t3[f"rand_ge_{w}"] = [float((rb[int(T.loc[k, "x"])][:, i] >= T.loc[k, f"{w}_usd_hs1"]).mean()) for k in t3.index]
    t3["rand_ge_both"] = [float(((rb[int(T.loc[k, "x"])][:, 3] >= T.loc[k, "dd_jul_usd_hs1"]) &
                                 (rb[int(T.loc[k, "x"])][:, 4] >= T.loc[k, "dd_oct_usd_hs1"])).mean()) for k in t3.index]
    report["t3"] = t3.reset_index().to_dict("records")

    # both declines versus the exposure-matched control
    counts = {}
    for label, g in (("all", R), ("feasible_hourly", hourly_feas), ("needs_automation", R[R.feasible != "feasible"]),
                     ("daily_refs_and_gate_only", R[~R.hourly])):
        counts[label] = dict(n=len(g), **{c: int(g[c].sum()) for c in ("both", "both_hs1", "both_hs2", "both_hs1_1pt", "beats_base_both_hs1")},
                             both_hs1_and_h2_edge_pos=int((g.both_hs1 & (g.h2_edge_hs1 > 0)).sum()),
                             both_hs1_and_full_ge_base=int((g.both_hs1 & (g.full_usd_hs1 >= b.full_usd_hs1)).sum()))
    report["both_declines"] = counts
    passed = hourly_feas[hourly_feas.both_hs1].copy()
    passed["min_edge"] = passed[["dd_jul_edge_hs1", "dd_oct_edge_hs1"]].min(axis=1)
    passed = passed.sort_values("min_edge", ascending=False)
    rob = [robustness(M, spec, k, bc, {"dd_jul": float(passed.loc[k, "dd_jul_edge_hs1"]), "dd_oct": float(passed.loc[k, "dd_oct_edge_hs1"])},
                      float(passed.loc[k, "h2_usd_hs1"]))
           for k in passed.index[:12]]
    for k in dict.fromkeys(sel.strategy):                                      # the forward-selected rules as well
        if k != "base" and k not in set(passed.index[:12]):
            rob.append(robustness(M, spec, k, bc, {"dd_jul": float(T.loc[k, "dd_jul_edge_hs1"]), "dd_oct": float(T.loc[k, "dd_oct_edge_hs1"])},
                                  float(T.loc[k, "h2_usd_hs1"])))
    rob = pd.DataFrame(rob)
    report["robustness"] = rob.to_dict("records")
    show_cols = ["feasible", "full_usd_hs1", "h1_usd_hs1", "h2_usd_hs1", "dd_jul_usd_hs1", "dd_oct_usd_hs1", "full_usd_mdd", "h2_expo",
                 "full_turn", "h2_edge_hs1", "dd_jul_edge_hs1", "dd_oct_edge_hs1", "dd_jul_edge_hs2", "dd_oct_edge_hs2", "stale_turn"]
    report["passed_feasible_hourly"] = passed[show_cols].reset_index().to_dict("records")
    report["runtime_seconds"] = time.time() - t0
    (common.CACHE / "hourly_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 400)
    print(json.dumps({k: report[k] for k in ("n_candidates", "omega", "picks", "base_reproduction", "selftest", "reconcile", "edt_sessions",
                                             "est_sessions", "feasible_1030_sessions_in", "half_spread", "stale_turn_max", "both_declines",
                                             "t3_beats_base", "t3_random_baskets", "runtime_seconds")}, indent=1, default=float))
    show("base", T.loc[["base"], [f"{w}_usd" for w in REPORT_WINDOWS] + ["full_usd_hs1", "full_usd_mdd", "h2_usd_mdd", "full_expo", "full_turn"]])
    show("selections", sel)
    show("subfamily medians", sub.T)
    show("T1 timing (hs1)", t1)
    show("T1 median of (schedule - close)", t1m)
    show("T1 daily references", t1ref)
    show("T2", T.loc[T.subfamily.isin(["T2_stock_stop", "T2_soxx_drop"]), show_cols])
    show("T3", t3)
    show("T4", T.loc[T.subfamily.isin(["T4_cross", "T4_gate_only"]), show_cols])
    show("feasible hourly rules beating the matched control in both declines (hs1)", passed[show_cols])
    show("robustness", rob.set_index("strategy") if len(rob) else rob)
    return report


if __name__ == "__main__":
    main()
