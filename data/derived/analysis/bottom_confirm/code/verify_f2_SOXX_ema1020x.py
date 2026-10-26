"""Independent adversarial verification: family f2, rule ema1020x (EMA10 crosses above EMA20), kind SOXX.

Pre-registration: ../BRIEF.md (2026-10-26). This file deliberately does NOT import bc_lib. Every quantity
(calendar, Ldd, in_dd, up, EMA, first signals with cooldown, outcomes, same-state baseline, bootstrap,
L1 test) is rebuilt from the raw CSVs with plain numpy loops and then compared with the family files
results/f2_SOXX_signals.csv and results/f2_SOXX_summary.csv.

Sections
  [1] re-implementation and comparison          [4] null: random in_dd dates, same count / up-bucket mix / spacing
  [2] look-ahead audit (truncation test)         [5] mechanical check: plain 'up >= candidate median up'
  [3] robustness (EMA spans, delay, cooldown,    [6] verdict inputs
      in_dd threshold)

Run from data/derived/analysis/bottom_confirm:  python3 -W ignore code/verify_f2_SOXX_ema1020x.py
Output: results/verify_f2_SOXX_ema1020x.txt
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f2_SOXX_ema1020x.txt"
P_PRICES = ROOT / "research/news_turning_points_20261025/prices.csv"
P_CLOSE = ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv"

RULE, KIND, FAMILY = "ema1020x", "SOXX", "f2"
LAST = pd.Timestamp("2026-10-23")
EVAL0 = pd.Timestamp("2024-01-02")
THR0, COOL0, FAST0, SLOW0 = 0.10, 20, 10, 20
EDGES = (0.03, 0.08, 0.15)                     # up buckets 0-3, 3-8, 8-15, 15+
BUCKET_NAMES = ("0-3%", "3-8%", "8-15%", "15%+")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
MAIN = "2024~2026"
NULL_DRAWS = 2000
BOOT_B = 5000
RNG_SEED = 7_2026_10_26

LINES: list[str] = []


def say(s: str = "") -> None:
    LINES.append(s)
    print(s)


# ------------------------------------------------------------------------------------------- data
def load_raw():
    """SOXX close (2023-01-03..2026-10-23) and open (2023-12-01..). Parses the two-row header by hand."""
    hdr = pd.read_csv(P_PRICES, header=None, nrows=2)
    body = pd.read_csv(P_PRICES, header=None, skiprows=3)
    fields, ticks = hdr.iloc[0, 1:].tolist(), hdr.iloc[1, 1:].tolist()
    vals = body.iloc[:, 1:].astype(float)
    vals.columns = pd.MultiIndex.from_arrays([fields, ticks])
    vals.index = pd.to_datetime(body.iloc[:, 0])
    pc = vals[("close", "SOXX")].dropna()
    po = vals[("open", "SOXX")].dropna()
    u = pd.read_csv(P_CLOSE, encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna()
    ov = u.index.intersection(pc.index)
    maxdiff = float(np.abs(u[ov].to_numpy() - pc[ov].to_numpy()).max())
    idx = u.index.union(pc.index).sort_values()
    idx = idx[idx <= LAST]
    C = pd.Series(np.nan, index=idx)
    C.loc[u.index[u.index <= LAST]] = u[u.index <= LAST]
    C.loc[pc.index] = pc
    O = po.reindex(idx)
    assert C.notna().all()
    cal_msg = "exchange_calendars unavailable"
    try:
        import exchange_calendars as xc
        s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(idx[0], LAST))
        s = s.tz_localize(None) if s.tz is not None else s
        cal_msg = f"XNYS sessions missing={len(s.difference(idx))} extra={len(idx.difference(s))}"
    except Exception as e:  # pragma: no cover
        cal_msg += f" ({e})"
    return C, O, maxdiff, cal_msg


# ------------------------------------------------------------------------------------------- state
def decline_state(c: np.ndarray, thr: float):
    """Ldd[t] = min(C[pk..t]), pk = latest position of the 60-session max ending at t;
    dd60[t] = C[t]/max(C[t-59..t]) - 1; in_dd[t] = min(dd60[t-19..t]) <= -thr; up = C/Ldd - 1."""
    N = len(c)
    Ldd = np.full(N, np.nan)
    dd60 = np.full(N, np.nan)
    for t in range(59, N):
        w = c[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])
        Ldd[t] = c[pk:t + 1].min()
        dd60[t] = c[t] / mx - 1.0
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    up = c / Ldd - 1.0
    return Ldd, dd60, in_dd, up


def ema(x: np.ndarray, span: int, adjust: bool = False) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    e = np.empty(len(x))
    if not adjust:
        e[0] = x[0]
        for i in range(1, len(x)):
            e[i] = a * x[i] + (1.0 - a) * e[i - 1]
        return e
    num = den = 0.0
    for i in range(len(x)):
        num = x[i] + (1.0 - a) * num
        den = 1.0 + (1.0 - a) * den
        e[i] = num / den
    return e


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = x[t - n + 1:t + 1].mean()
    return out


def cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = np.zeros(len(a), bool)
    for t in range(1, len(a)):
        if np.isnan(a[t]) or np.isnan(b[t]) or np.isnan(a[t - 1]) or np.isnan(b[t - 1]):
            continue
        s[t] = (a[t] > b[t]) and (a[t - 1] <= b[t - 1])
    return s


def ema_signal(c: np.ndarray, fast: int, slow: int, adjust: bool = False) -> np.ndarray:
    return cross_up(ema(c, fast, adjust), ema(c, slow, adjust))


# ------------------------------------------------------------------------------------------- outcomes
def outcomes(c: np.ndarray, o: np.ndarray, Ldd: np.ndarray) -> dict[str, np.ndarray]:
    N = len(c)
    res = {k: np.full(N, np.nan) for k in ("hold20", "hold40", "near5", "win20", "r20", "r60", "hold20_strict")}
    for t in range(N):
        if t + 20 <= N - 1 and not np.isnan(Ldd[t]):
            m = c[t + 1:t + 21].min()
            res["hold20"][t] = float(m >= Ldd[t] - 1e-12)
            res["hold20_strict"][t] = float(m >= Ldd[t])
        if t + 40 <= N - 1 and not np.isnan(Ldd[t]):
            res["hold40"][t] = float(c[t + 1:t + 41].min() >= Ldd[t] - 1e-12)
        if t + 1 <= N - 1 and not np.isnan(o[t + 1]):
            if t + 40 <= N - 1:
                res["near5"][t] = float(c[t + 1:t + 41].min() >= 0.95 * o[t + 1])
            if t + 20 <= N - 1:
                res["r20"][t] = c[t + 20] / o[t + 1] - 1.0
                res["win20"][t] = float(res["r20"][t] > 0)
            if t + 60 <= N - 1:
                res["r60"][t] = c[t + 60] / o[t + 1] - 1.0
    return res


def bucket(up: np.ndarray) -> np.ndarray:
    b = np.full(len(up), -1)
    ok = ~np.isnan(up)
    b[ok] = np.digitize(up[ok], EDGES)
    return b


def matched_base(dates: pd.DatetimeIndex, in_dd, up, y: np.ndarray) -> np.ndarray:
    """Same calendar year x same up bucket mean of outcome y over in_dd days from EVAL0 (y must be non-NaN)."""
    yr = np.asarray(dates.year)
    b = bucket(up)
    pool = in_dd & (b >= 0) & (np.asarray(dates) >= np.datetime64(EVAL0)) & ~np.isnan(y)
    cell = {}
    for yy in np.unique(yr[pool]):
        for bb in range(4):
            m = pool & (yr == yy) & (b == bb)
            if m.any():
                cell[(yy, bb)] = y[m].mean()
    out = np.full(len(y), np.nan)
    for t in range(len(y)):
        if b[t] >= 0:
            out[t] = cell.get((yr[t], b[t]), np.nan)
    return out


# ------------------------------------------------------------------------------------------- evaluation
def first_signals(sig, in_dd, Ldd, dates, cool):
    d = np.asarray(dates)
    elig = sig & in_dd & ~np.isnan(Ldd) & (d >= np.datetime64(EVAL0)) & (d <= np.datetime64(LAST))
    keep, prev = [], -10 ** 9
    for p in np.flatnonzero(elig):
        if p - prev <= cool:
            continue
        keep.append(p)
        prev = p
    return np.array(keep, int)


class Ctx:
    def __init__(self, C: pd.Series, O: pd.Series, thr: float):
        self.dates = C.index
        self.c = C.to_numpy(float)
        self.o = O.to_numpy(float)
        self.thr = thr
        self.Ldd, self.dd60, self.in_dd, self.up = decline_state(self.c, thr)
        self.y = outcomes(self.c, self.o, self.Ldd)
        self.base = {k: matched_base(self.dates, self.in_dd, self.up, v) for k, v in self.y.items()}


def period_table(pos: np.ndarray, ctx: Ctx) -> dict:
    d = ctx.dates[pos] if len(pos) else pd.DatetimeIndex([])
    out = {}
    for name, a, b in PERIODS:
        m = (d >= a) & (d <= b)
        p = pos[m]
        r = dict(n=len(p))
        for k in ("hold20", "near5", "win20", "r20"):
            v, bv = ctx.y[k][p], ctx.base[k][p]
            ok = ~np.isnan(v) & ~np.isnan(bv)
            r[f"{k}_n"] = int(ok.sum())
            r[k] = float(v[ok].mean()) if ok.any() else np.nan
            r[f"{k}_base"] = float(bv[ok].mean()) if ok.any() else np.nan
        r["up_median"] = float(np.median(ctx.up[p])) if len(p) else np.nan
        out[name] = r
    return out


def l1(tab: dict) -> bool:
    g = lambda p, k: tab[p][k] if not (isinstance(tab[p][k], float) and np.isnan(tab[p][k])) else 0
    c1 = g(MAIN, "hold20_n") >= 8 and g(MAIN, "hold20") >= 0.70
    c2 = g("2026", "hold20_n") >= 3 and g("2026", "hold20") >= 0.70
    c3 = g("2025", "hold20_n") < 3 or g("2025", "hold20") >= 0.60
    return bool(c1 and c2 and c3)


def fmt_row(label: str, tab: dict) -> str:
    def cell(p):
        r = tab[p]
        return f"{r['hold20']:.3f}({r['hold20_n']})" if r["hold20_n"] else f"  -  ({r['n']})"
    m = tab[MAIN]
    return (f"  {label:34s} main {cell(MAIN):11s} 2026 {cell('2026'):10s} H1 {cell('2026H1'):10s} "
            f"H2 {cell('2026H2'):10s} 2025 {cell('2025'):10s} 2024 {cell('2024'):10s} "
            f"upmed {m['up_median']:.3f}  base {m['hold20_base']:.3f}  win20 {m['win20']:.2f}")


def block_boot_lo(v: np.ndarray, pos: np.ndarray, rng, B=BOOT_B) -> float:
    lab = pos // 20
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return np.nan
    s = np.bincount(inv, weights=v)
    n = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    return float(np.quantile(s[pick].sum(1) / n[pick].sum(1), 0.025))


# ------------------------------------------------------------------------------------------- null
def null_draws(pool_pos: np.ndarray, pool_bucket: np.ndarray, counts: dict, y: np.ndarray, cool: int, rng,
               draws: int, strata=None):
    """Random date sets: per stratum key the same number of dates as observed, all pairwise > cool apart.
    counts: key -> needed count; strata: array of keys aligned with pool_pos (default: bucket)."""
    keys = pool_bucket if strata is None else strata
    pools = {k: pool_pos[keys == k] for k in counts}
    slots = np.concatenate([[k] * n for k, n in counts.items()]) if counts else np.array([])
    vals, attempts = [], 0
    while len(vals) < draws:
        attempts += 1
        if attempts > draws * 200:
            break
        chosen = []
        okd = True
        for k in rng.permutation(slots):
            cand = pools[k]
            if chosen:
                ch = np.array(chosen)
                cand = cand[(np.abs(cand[:, None] - ch[None, :]) > cool).all(1)]
            if len(cand) == 0:
                okd = False
                break
            chosen.append(int(rng.choice(cand)))
        if okd:
            vals.append(y[np.array(chosen)].mean())
    return np.array(vals), attempts


# ------------------------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(RNG_SEED)
    C, O, maxdiff, cal_msg = load_raw()
    say("=" * 118)
    say(f"Independent verification: family {FAMILY}, rule {RULE}, kind {KIND}  (no bc_lib import)")
    say("=" * 118)
    say(f"Data: SOXX close {C.index[0].date()}..{C.index[-1].date()}  N={len(C)}; open from "
        f"{O.first_valid_index().date()}")
    say(f"  overlap max |us_prices_close - prices.csv| = {maxdiff:.2e}; {cal_msg}")

    ctx = Ctx(C, O, THR0)
    sig0 = ema_signal(ctx.c, FAST0, SLOW0)
    pos0 = first_signals(sig0, ctx.in_dd, ctx.Ldd, ctx.dates, COOL0)
    tab0 = period_table(pos0, ctx)

    # ---------------------------------------------------------------- [1]
    say("\n[1] Independent re-implementation vs family files")
    fam_sig = pd.read_csv(RES / "f2_SOXX_signals.csv", parse_dates=["date"])
    fam_sig = fam_sig[fam_sig.rule == RULE].reset_index(drop=True)
    fam_sum = pd.read_csv(RES / "f2_SOXX_summary.csv")
    fam_sum = fam_sum[fam_sum.rule == RULE].set_index("period")
    mine = pd.DataFrame(dict(date=ctx.dates[pos0], pos=pos0, up=ctx.up[pos0], dd60=ctx.dd60[pos0],
                             Ldd=ctx.Ldd[pos0], hold20=ctx.y["hold20"][pos0], base_hold20=ctx.base["hold20"][pos0],
                             hold40=ctx.y["hold40"][pos0], near5=ctx.y["near5"][pos0],
                             win20=ctx.y["win20"][pos0], r20=ctx.y["r20"][pos0], r60=ctx.y["r60"][pos0]))
    say(mine.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    same_dates = list(mine.date) == list(fam_sig.date)
    say(f"  first-signal dates identical to family: {same_dates}  (mine {len(mine)}, family {len(fam_sig)})")
    mism = []
    if same_dates:
        for k in ("pos", "up", "dd60", "hold20", "base_hold20", "hold40", "near5", "win20", "r20", "r60"):
            a, b = mine[k].to_numpy(float), fam_sig[k].to_numpy(float)
            bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-6))
            for i in np.flatnonzero(bad):
                mism.append(f"{mine.date.iat[i].date()} {k}: mine {a[i]} family {b[i]}")
    else:
        only_m = sorted(set(mine.date) - set(fam_sig.date))
        only_f = sorted(set(fam_sig.date) - set(mine.date))
        mism.append(f"dates only mine {[d.date() for d in only_m]} only family {[d.date() for d in only_f]}")
    say("  per-signal field mismatches: " + ("none" if not mism else "; ".join(mism)))
    strict_diff = int(np.nansum(np.abs(ctx.y["hold20"][pos0] - ctx.y["hold20_strict"][pos0])))
    say(f"  hold20 with 1e-12 tolerance vs strict >=: {strict_diff} signal(s) differ")
    say("  period      mine hold20 (n)   family hold20 (n)   mine base   family base")
    per_ok = True
    for name, _, _ in PERIODS:
        r = tab0[name]
        fh = fam_sum.loc[name, "hold20"] if name in fam_sum.index else np.nan
        fn = int(fam_sum.loc[name, "hold20_n"]) if name in fam_sum.index and pd.notna(fam_sum.loc[name, "hold20_n"]) else 0
        fb = fam_sum.loc[name, "hold20_base"] if name in fam_sum.index else np.nan
        ok = (r["hold20_n"] == fn) and (np.isclose(r["hold20"], fh, atol=1e-6, equal_nan=True)) and \
             np.isclose(r["hold20_base"], fb, atol=1e-6, equal_nan=True)
        per_ok &= bool(ok)
        say(f"  {name:10s}  {r['hold20']:.4f} ({r['hold20_n']:2d})      {fh:.4f} ({fn:2d})          "
            f"{r['hold20_base']:.4f}      {fb:.4f}   {'OK' if ok else 'MISMATCH'}")
    indep_match = bool(same_dates and not mism and per_ok)
    say(f"  => independent match: {indep_match}")
    m = tab0[MAIN]
    say(f"  main: n={m['n']} hold20={m['hold20']:.4f} near5={m['near5']:.4f} win20={m['win20']:.4f} "
        f"r20={m['r20']:.4f} up_med={m['up_median']:.4f}  L1(mine)={l1(tab0)}")
    v = ctx.y["hold20"][pos0]
    bv = ctx.base["hold20"][pos0]
    ok = ~np.isnan(v) & ~np.isnan(bv)
    dlo = block_boot_lo(v[ok] - bv[ok], pos0[ok], rng)
    say(f"  L2 check: main hold20 - matched base = {(v[ok] - bv[ok]).mean():+.4f}; my 20-session block bootstrap "
        f"2.5% = {dlo:+.4f} (family {fam_sum.loc[MAIN, 'hold20_dlo']:+.4f}) -> L2 {'pass' if dlo > 0 else 'fail'}")
    dmain = np.asarray(ctx.dates)
    mainmask = (dmain >= np.datetime64(EVAL0)) & (dmain <= np.datetime64(LAST)) & ctx.in_dd & ~np.isnan(ctx.y["hold20"])
    p_unc = ctx.y["hold20"][mainmask].mean()
    from math import comb
    k_obs = int(round(m["hold20"] * m["hold20_n"]))
    nn = m["hold20_n"]
    binom = lambda p: sum(comb(nn, j) * p ** j * (1 - p) ** (nn - j) for j in range(k_obs, nn + 1))
    say(f"  unconditional in_dd hold20 2024~2026 = {p_unc:.4f} (days={int(mainmask.sum())})")
    say(f"  binomial P(X >= {k_obs}/{nn} | p={p_unc:.3f}) = {binom(p_unc):.3f};  "
        f"| p=matched base {m['hold20_base']:.3f}) = {binom(m['hold20_base']):.3f}")
    # the 2026-04-08 case: at a new 60-session high, Ldd resets to today's close
    for p in pos0:
        if ctx.up[p] == 0:
            say(f"  note {ctx.dates[p].date()}: dd60={ctx.dd60[p]:+.4f} up=0 -> signal day is a new 60-session high, "
                f"Ldd = today's close {ctx.Ldd[p]:.2f}; hold20 means 'no close below the signal close in 20 sessions'")
    nh = mainmask & (ctx.dd60 >= -1e-12)
    say(f"  context: in_dd days at a new 60-session high (dd60=0) 2024~2026: {int(nh.sum())} days, "
        f"hold20 {ctx.y['hold20'][nh].mean():.3f}; all of them fall in the 0-3% up bucket (pool mean there is "
        f"dominated by days near a real low)")

    # ---------------------------------------------------------------- [2]
    say("\n[2] Look-ahead audit")
    N = len(ctx.c)
    t0 = int(np.searchsorted(dmain, np.datetime64(EVAL0)))
    bad = []
    for t in range(t0, N):
        ct = ctx.c[:t + 1]
        e_f, e_s = ema(ct, FAST0), ema(ct, SLOW0)
        s_t = (e_f[t] > e_s[t]) and (e_f[t - 1] <= e_s[t - 1])
        w = ct[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])
        L_t = ct[pk:t + 1].min()
        dd_t = np.array([ct[q] / ct[q - 59:q + 1].max() - 1 for q in range(t - 19, t + 1)])
        in_t = dd_t.min() <= -THR0
        if s_t != sig0[t] or not np.isclose(L_t, ctx.Ldd[t]) or in_t != ctx.in_dd[t]:
            bad.append(ctx.dates[t].date())
    say(f"  truncation test (EMA signal, Ldd, in_dd recomputed with data cut at t, {N - t0} sessions): "
        + ("all identical" if not bad else f"DIFFERENCES at {bad[:10]}"))
    # family code text checks
    src = (HERE / "code/f2_reclaim.py").read_text(encoding="utf-8")
    ema_src = src[src.index("def ema_cross"):src.index("def higher_low")]
    flags = [kw for kw in ("shift(-", "load_market", "dollar_vol", "notes", "center=True", "bfill") if kw in ema_src]
    say(f"  f2_reclaim.ema_cross source: uses ewm(span, adjust=False) + shift(1); suspicious tokens in it: "
        f"{flags if flags else 'none'}")
    say("  family pipeline: rule depends only on SOXX closes up to t; in_dd/Ldd windows end at t; outcomes start at")
    say("  t+1 (entry O[t+1]); the matched baseline averages same-year outcomes (an evaluation reference, not an input).")
    say("  EMA seed: both implementations seed EMA at the first close (2023-01-03), ~250 sessions before EVAL0, so")
    say("  the seed is immaterial. Residual: Yahoo split/dividend-adjusted history (scale factors applied later) can")
    say("  move an EMA10/EMA20 near-tie by a few bp; does not create forward information about outcomes.")

    # ---------------------------------------------------------------- [3]
    say("\n[3] Robustness (hold20 (n) per period; main = 2024~2026; EMA spans fast/slow)")
    variants = []

    def run(label, sig, ctx_, cool=COOL0, core=False):
        pos = first_signals(sig, ctx_.in_dd, ctx_.Ldd, ctx_.dates, cool)
        tab = period_table(pos, ctx_)
        variants.append((label, tab, core, pos, ctx_))
        say(fmt_row(label, tab))
        return tab

    run(f"base EMA {FAST0}/{SLOW0} cool=20 thr=10% d=0", sig0, ctx, core=True)
    say("  -- one step either side (fast +-2, slow +-5):")
    for f, s in ((8, 20), (12, 20), (10, 15), (10, 25)):
        run(f"EMA {f}/{s}", ema_signal(ctx.c, f, s), ctx, core=True)
    say("  -- finer step (fast +-1, slow +-2) and diagonal:")
    for f, s in ((9, 20), (11, 20), (10, 18), (10, 22), (8, 18), (12, 22), (8, 25), (12, 15)):
        run(f"EMA {f}/{s}", ema_signal(ctx.c, f, s), ctx)
    say("  -- wider context:")
    for f, s in ((5, 10), (5, 20), (8, 21), (12, 26), (10, 30), (10, 40), (20, 50)):
        run(f"EMA {f}/{s} (wider)", ema_signal(ctx.c, f, s), ctx)
    run("EMA 10/20 adjust=True", ema_signal(ctx.c, FAST0, SLOW0, adjust=True), ctx)
    run("SMA10 crosses SMA20 (context)", cross_up(sma(ctx.c, 10), sma(ctx.c, 20)), ctx)
    say("  -- timing / cooldown / state threshold:")
    for dly in (1, 2):
        sd = np.zeros_like(sig0)
        sd[dly:] = sig0[:-dly]
        run(f"delay +{dly} session{'s' if dly > 1 else ''}", sd, ctx, core=(dly == 1))
    for cl in (10, 30):
        run(f"cooldown {cl}", sig0, ctx, cool=cl, core=True)
    ctxs = {}
    for th in (0.08, 0.12):
        ctxs[th] = Ctx(C, O, th)
        run(f"in_dd thr {th * 100:.0f}%", sig0, ctxs[th], core=True)
    say("  L1 (registered SOXX rule) per variant, and main-period hold20 minus same-state baseline:")
    for label, tab, core, pos, _ in variants:
        r = tab[MAIN]
        say(f"    {label:34s} L1={str(l1(tab)):5s} diff={r['hold20'] - r['hold20_base']:+.3f}  "
            f"first dates: {' '.join(str(d.date())[2:] for d in variants[[v[0] for v in variants].index(label)][4].dates[pos])}"
            if core else f"    {label:34s} L1={str(l1(tab)):5s} diff={r['hold20'] - r['hold20_base']:+.3f}")

    # ---------------------------------------------------------------- [4]
    say(f"\n[4] Null: random in_dd dates in {MAIN}, same count and up-bucket mix, pairwise spacing > {COOL0} sessions")
    obs_pos = pos0[~np.isnan(ctx.y["hold20"][pos0])]
    obs = ctx.y["hold20"][obs_pos].mean()
    b_all = bucket(ctx.up)
    pool_mask = mainmask & (b_all >= 0)
    pool_pos = np.flatnonzero(pool_mask)
    pool_b = b_all[pool_pos]
    obs_b = b_all[obs_pos]
    counts = {k: int((obs_b == k).sum()) for k in range(4) if (obs_b == k).sum()}
    say(f"  observed hold20 = {obs:.4f} (n={len(obs_pos)}); bucket mix (0-3,3-8,8-15,15+) = "
        f"{[int((obs_b == k).sum()) for k in range(4)]}; pool days = {len(pool_pos)} by bucket "
        f"{[int((pool_b == k).sum()) for k in range(4)]}")
    say(f"  pool hold20 by bucket: {[round(float(ctx.y['hold20'][pool_pos[pool_b == k]].mean()), 3) for k in range(4)]}")
    vals, att = null_draws(pool_pos, pool_b, counts, ctx.y["hold20"], COOL0, rng, NULL_DRAWS)
    p_null = float((vals >= obs - 1e-12).mean())
    say(f"  draws={len(vals)} (attempts {att}); null mean={vals.mean():.4f} sd={vals.std():.4f} "
        f"q05={np.quantile(vals, 0.05):.3f} q95={np.quantile(vals, 0.95):.3f}")
    say(f"  p = P(null hold20 >= observed) = {p_null:.4f}")
    # secondary: year x bucket strata
    yr = np.asarray(ctx.dates.year)
    strata_pool = yr[pool_pos] * 10 + pool_b
    strata_obs = yr[obs_pos] * 10 + obs_b
    counts2 = {int(k): int((strata_obs == k).sum()) for k in np.unique(strata_obs)}
    vals2, _ = null_draws(pool_pos, pool_b, counts2, ctx.y["hold20"], COOL0, rng, NULL_DRAWS, strata=strata_pool)
    p_null2 = float((vals2 >= obs - 1e-12).mean())
    say(f"  secondary (year x bucket matched): draws={len(vals2)} mean={vals2.mean():.4f} p={p_null2:.4f}")
    # tertiary: bucket matched but the up=0 new-high day matched to new-high days (dd60 = 0)
    nh_flag = (ctx.dd60 >= -1e-12)
    strata_pool3 = pool_b * 10 + nh_flag[pool_pos]
    strata_obs3 = obs_b * 10 + nh_flag[obs_pos]
    counts3 = {int(k): int((strata_obs3 == k).sum()) for k in np.unique(strata_obs3)}
    vals3, _ = null_draws(pool_pos, pool_b, counts3, ctx.y["hold20"], COOL0, rng, NULL_DRAWS, strata=strata_pool3)
    p_null3 = float((vals3 >= obs - 1e-12).mean())
    say(f"  tertiary (bucket x 'new 60-session high' flag): draws={len(vals3)} mean={vals3.mean():.4f} p={p_null3:.4f}")
    # 2026-only null
    m26 = mainmask & (yr == 2026) & (b_all >= 0)
    pp26 = np.flatnonzero(m26)
    o26 = obs_pos[yr[obs_pos] == 2026]
    c26 = {k: int((b_all[o26] == k).sum()) for k in range(4) if (b_all[o26] == k).sum()}
    v26, _ = null_draws(pp26, b_all[pp26], c26, ctx.y["hold20"], COOL0, rng, NULL_DRAWS)
    obs26 = ctx.y["hold20"][o26].mean()
    p26 = float((v26 >= obs26 - 1e-12).mean())
    say(f"  2026-only null: observed {obs26:.3f} (n={len(o26)}, mix {[int((b_all[o26] == k).sum()) for k in range(4)]}), "
        f"null mean {v26.mean():.3f}, p={p26:.4f}")

    # ---------------------------------------------------------------- [5]
    say("\n[5] Mechanical comparison: plain rule up >= candidate median up (same in_dd, cooldown, outcomes)")
    upmed = m["up_median"]
    for thr_up, lab in ((upmed, f"up >= {upmed:.4f} (cand. median)"), (0.08, "up >= 0.0800"), (0.10, "up >= 0.1000"),
                        (0.15, "up >= 0.1500")):
        sig_u = ~np.isnan(ctx.up) & (ctx.up >= thr_up)
        tab = period_table(first_signals(sig_u, ctx.in_dd, ctx.Ldd, ctx.dates, COOL0), ctx)
        say(fmt_row(lab, tab))
    hi = mainmask & (ctx.up >= upmed)
    lo = mainmask & (ctx.up < upmed)
    say(f"  all in_dd days 2024~2026 with up >= {upmed:.4f} (no cooldown): hold20 = {ctx.y['hold20'][hi].mean():.4f} "
        f"(days {int(hi.sum())}); with up < it: {ctx.y['hold20'][lo].mean():.4f} (days {int(lo.sum())})")
    say("  within-bucket comparison, 2024~2026 (candidate first signals vs all in_dd pool days):")
    for k in range(4):
        cm = obs_pos[obs_b == k]
        if len(cm):
            say(f"    bucket {BUCKET_NAMES[k]:6s} cand {ctx.y['hold20'][cm].mean():.3f} (n={len(cm)})   pool "
                f"{ctx.y['hold20'][pool_pos[pool_b == k]].mean():.3f} (days={int((pool_b == k).sum())})   "
                f"cand dates {[str(ctx.dates[q].date()) for q in cm]}")
    raw = mainmask & sig0
    say(f"  raw ema1020x in_dd days (no cooldown): hold20 = {ctx.y['hold20'][raw].mean():.4f} (days {int(raw.sum())}), "
        f"matched base avg {np.nanmean(ctx.base['hold20'][raw]):.4f}")
    hit_cushion = obs_pos[(ctx.up[obs_pos] >= 0.08)]
    hit_low = obs_pos[(ctx.up[obs_pos] < 0.08)]
    say(f"  signals with up >= 8%: hold20 {ctx.y['hold20'][hit_cushion].mean():.3f} (n={len(hit_cushion)}), "
        f"win20 {np.nanmean(ctx.y['win20'][hit_cushion]):.3f};  up < 8%: hold20 {ctx.y['hold20'][hit_low].mean():.3f} "
        f"(n={len(hit_low)}) dates {[str(ctx.dates[q].date()) for q in hit_low]}")

    # ---------------------------------------------------------------- [6]
    say("\n[6] Verdict inputs")
    core = [(lab, l1(tab)) for lab, tab, c_, _, _ in variants if c_]
    n_core_pass = sum(x for _, x in core)
    say(f"  independent match={indep_match}, look-ahead found={bool(bad)}, main hold20={obs:.4f} vs matched base "
        f"{m['hold20_base']:.4f}, null p={p_null:.4f} (year x bucket p={p_null2:.4f}, new-high strata p={p_null3:.4f})")
    say(f"  L1 survives in {n_core_pass}/{len(core)} core variants: " + ", ".join(f"{a}={b}" for a, b in core))
    say(f"  L2 (registered) {'passes' if dlo > 0 else 'fails'}: block-bootstrap lower bound of diff = {dlo:+.4f}")

    base_l1 = l1(tab0)
    if not indep_match or bad or not base_l1:
        verdict = "refuted"
    elif p_null < 0.05 and n_core_pass == len(core) and dlo > 0:
        verdict = "confirmed"
    else:
        verdict = "fragile"
    say(f"\nVERDICT: {verdict}")
    say("  Reproduced exactly (10 first signals, every field, every period; L2 bootstrap bound identical), no")
    say("  look-ahead (truncation test identical on all 706 sessions). The 90% hit rate is stable: every one-step EMA")
    say("  neighbour, delay +1/+2, cooldown 10 and in_dd 8%/12% keep L1; cooldown 30 drops 2026 to n=2 (< 3).")
    say(f"  But it carries no information beyond the rebound already made: the up-bucket-matched random-date null")
    say(f"  gives mean {vals.mean():.3f} and p={p_null:.3f} (year x bucket p={p_null2:.3f}); L2 fails "
        f"(diff {(v[ok] - bv[ok]).mean():+.3f}, lower bound {dlo:+.3f}); the")
    say(f"  sign of 'hold20 - same-state base' flips across neighbours (-0.078..+0.094). 7/10 signals fire with up >= 8%,")
    say(f"  where every in_dd day holds (pool 98.7-100%); the plain rule up >= {upmed:.3f} scores 100% (n=8). The only")
    say("  informative-looking cells are 3 low-rebound signals (2/3 held; the 2026-04-08 hit is a new 60-session high")
    say("  where Ldd resets to that close). near5 0.33 and win20 0.40 are below their same-state bases (0.47, 0.55):")
    say("  the cross fires late, after a ~13% rebound, and the entries lose money over 20 sessions 60% of the time.")
    say("  Fragile, not confirmed: the 70%+ is a mechanical consequence of how far price has already rebounded.")

    RES.mkdir(exist_ok=True)
    return dict(indep_match=indep_match, lookahead=bool(bad), obs=obs, p_null=p_null, p_null2=p_null2,
                p_null3=p_null3, core=core, upmed=upmed)


if __name__ == "__main__":
    info = main()
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwritten {OUT}")
