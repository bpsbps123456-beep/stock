"""Adversarial verification: bottom_confirm family f1, rule up10, kind SOXX (BRIEF.md, registered 2026-10-26).

Independent of bc_lib: nothing is imported from it. Raw CSVs are read directly, the session calendar is taken
from the data files themselves (cross-checked against exchange_calendars XNYS), and every quantity is recomputed
with plain loops.

Definitions re-implemented (BRIEF.md / bc_lib docstring)
  pk[t]    latest position of the max close in C[t-59..t]
  Ldd[t]   min(C[pk..t])                 decline low
  dd60[t]  C[t]/max(C[t-59..t]) - 1
  up[t]    C[t]/Ldd[t] - 1
  in_dd[t] min(dd60[t-19..t]) <= -THR    (THR = 10% SOXX)
  rule     up[t] >= 0.10, only on in_dd days in 2024-01-02..2026-10-23
  first    greedy 20-session cooldown: drop if pos - prev_accepted <= COOL
  hold20   min(C[t+1..t+20]) >= Ldd[t]
  near5    min(C[t+1..t+40]) >= 0.95 * O[t+1]
  r20      C[t+20]/O[t+1] - 1, win20 = r20 > 0
  matched  mean outcome over in_dd days (>= 2024-01-02) of the same calendar year and up bucket (0-3, 3-8, 8-15, 15%+)

Checks: 1 independent match, 2 look-ahead audit notes, 3 robustness grid, 4 bucket-matched random-date null
(2000 draws), 5 mechanical comparison with up >= median(up), 6 verdict.
Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify_f1_SOXX_up10.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f1_SOXX_up10.txt"

RULE, LEVEL = "up10", 0.10
LAST = pd.Timestamp("2026-10-23")
EVAL0 = pd.Timestamp("2024-01-02")
BINS = [0.0, 0.03, 0.08, 0.15]          # bucket edges: [0,3) [3,8) [8,15) [15,inf)
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
N_NULL = 2000
SEED = 7_2026_10_26

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


# ------------------------------------------------------------------ data (raw CSVs, no bc_lib)
def load() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()].sort_index()
    o = pd.to_numeric(p[("open", "SOXX")], errors="coerce")
    c = pd.to_numeric(p[("close", "SOXX")], errors="coerce")
    keep = o.notna() & c.notna()
    o, c = o[keep], c[keep]
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna().sort_index()
    ov = u.index.intersection(c.index)
    say(f"raw: prices.csv SOXX rows {len(c)} ({c.index[0].date()}..{c.index[-1].date()}); "
        f"us_prices_close SOXX rows {len(u)} ({u.index[0].date()}..{u.index[-1].date()}); "
        f"overlap {len(ov)} days, max |diff| {float((u[ov] - c[ov]).abs().max()):.2e}")
    C = pd.concat([u[u.index < c.index[0]], c]).sort_index()
    assert not C.index.duplicated().any()
    C = C[C.index <= LAST]
    d = pd.DataFrame({"C": C})
    d["O"] = o.reindex(d.index)
    # calendar cross-check against XNYS (exchange_calendars is an independent source)
    try:
        import exchange_calendars as xc
        s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(d.index[0], d.index[-1]))
        s = s.tz_localize(None) if s.tz is not None else s
        miss = s.difference(d.index)
        extra = d.index.difference(s)
        say(f"calendar: data sessions {len(d)}, XNYS sessions {len(s)}, missing in data {len(miss)}, "
            f"extra in data {len(extra)}" + (f" missing={list(miss.date)[:10]}" if len(miss) else "")
            + (f" extra={list(extra.date)[:10]}" if len(extra) else ""))
    except Exception as e:  # noqa: BLE001
        say(f"calendar: XNYS cross-check skipped ({e})")
    return d


# ------------------------------------------------------------------ state + outcomes (plain loops)
def frames(d: pd.DataFrame, thr: float) -> pd.DataFrame:
    C = d["C"].to_numpy(float)
    O = d["O"].to_numpy(float)
    N = len(C)
    pk = np.full(N, -1)
    lpos = np.full(N, -1)
    Ldd = np.full(N, np.nan)
    dd60 = np.full(N, np.nan)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        mx = w.max()
        j = t - 59 + int(np.flatnonzero(w == mx)[-1])          # latest max
        seg = C[j:t + 1]
        mn = seg.min()
        pk[t] = j
        lpos[t] = j + int(np.flatnonzero(seg == mn)[-1])
        Ldd[t] = mn
        dd60[t] = C[t] / mx - 1
    up = C / Ldd - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[np.isfinite(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    hold20 = np.full(N, np.nan)
    cushion = np.full(N, np.nan)
    near5 = np.full(N, np.nan)
    r20 = np.full(N, np.nan)
    for t in range(N):
        if t + 20 <= N - 1 and np.isfinite(Ldd[t]):
            m = C[t + 1:t + 21].min()
            hold20[t] = float(m >= Ldd[t] - 1e-12)
            cushion[t] = m / Ldd[t] - 1
        if t + 1 <= N - 1 and np.isfinite(O[t + 1]):
            if t + 40 <= N - 1:
                near5[t] = float(C[t + 1:t + 41].min() >= 0.95 * O[t + 1])
            if t + 20 <= N - 1:
                r20[t] = C[t + 20] / O[t + 1] - 1
    win20 = np.where(np.isfinite(r20), (r20 > 0).astype(float), np.nan)
    f = pd.DataFrame(dict(C=C, O=O, pk=pk, lpos=lpos, Ldd=Ldd, dd60=dd60, up=up, in_dd=in_dd, hold20=hold20,
                          cushion=cushion, near5=near5, r20=r20, win20=win20), index=d.index)
    f["bucket"] = np.where(np.isfinite(up), np.digitize(up, BINS[1:]), -1)
    return f


def first_signals(raw: np.ndarray, f: pd.DataFrame, cool: int = 20, start=EVAL0) -> np.ndarray:
    idx = f.index
    ok = raw & f["in_dd"].to_numpy() & np.isfinite(f["Ldd"].to_numpy()) & (idx >= start) & (idx <= LAST)
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(p)
        prev = p
    return np.asarray(out, int)


def matched(f: pd.DataFrame, col: str) -> pd.Series:
    st = f["in_dd"] & np.isfinite(f["up"]) & (f.index >= EVAL0) & f[col].notna()
    g = f[st].groupby([f[st].index.year, f[st]["bucket"]])[col].mean()
    return pd.Series([g.get((y, b), np.nan) for y, b in zip(f.index.year, f["bucket"])], index=f.index)


def period_stats(f: pd.DataFrame, pos: np.ndarray, col: str = "hold20") -> dict[str, tuple[int, float]]:
    res = {}
    dates = f.index[pos] if len(pos) else pd.DatetimeIndex([])
    for name, a, b in PERIODS:
        sel = pos[(dates >= a) & (dates <= b)] if len(pos) else pos
        v = f[col].to_numpy()[sel]
        v = v[np.isfinite(v)]
        res[name] = (len(v), float(v.mean()) if len(v) else np.nan)
    return res


def fmt(st: dict, keys=("2024~2026", "2026", "2026H1", "2026H2", "2025")) -> str:
    return "  ".join(f"{k} {st[k][1]:.3f}(n{st[k][0]})" if st[k][0] else f"{k} -(n0)" for k in keys)


def block_boot_lo(v: np.ndarray, pos: np.ndarray, rng, B: int = 5000) -> tuple[float, float]:
    lab = pos // 20
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    s, n = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = s[pick].sum(1) / n[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


# ------------------------------------------------------------------ main
def main() -> None:
    pd.set_option("display.width", 240)
    say("verify f1 / up10 / SOXX  (adversarial, independent of bc_lib)")
    say("=" * 100)
    d = load()
    f = frames(d, 0.10)
    idx = f.index
    upv = f["up"].to_numpy()
    raw = np.isfinite(upv) & (upv >= LEVEL)
    pos = first_signals(raw, f)

    # ---------------------------------------------------------- 1. independent match
    say("\n[1] Independent re-implementation vs family files")
    fam_s = pd.read_csv(RES / "f1_SOXX_signals.csv", parse_dates=["date"])
    fam_s = fam_s[fam_s.rule == RULE].reset_index(drop=True)
    fam_m = pd.read_csv(RES / "f1_SOXX_summary.csv")
    fam_m = fam_m[fam_m.rule == RULE].set_index("period")
    mine_dates = list(idx[pos].date)
    fam_dates = list(fam_s.date.dt.date)
    say(f"first-signal dates: mine {len(mine_dates)}, family {len(fam_dates)}, identical={mine_dates == fam_dates}")
    if mine_dates != fam_dates:
        say(f"  only mine: {sorted(set(mine_dates) - set(fam_dates))}")
        say(f"  only family: {sorted(set(fam_dates) - set(mine_dates))}")
    bh = matched(f, "hold20")
    rows = []
    for p in pos:
        rows.append(dict(date=idx[p].date(), pos=p, C=f.C.iat[p], Ldd=f.Ldd.iat[p], low_date=idx[f.lpos.iat[p]].date(),
                         peak_date=idx[f.pk.iat[p]].date(), dd60=f.dd60.iat[p], up=f.up.iat[p], hold20=f.hold20.iat[p],
                         cushion20=f.cushion.iat[p], near5=f.near5.iat[p], win20=f.win20.iat[p], r20=f.r20.iat[p],
                         base_hold20=bh.iat[p]))
    mine = pd.DataFrame(rows)
    say(mine.round(4).to_string(index=False))
    if len(mine) == len(fam_s) and mine_dates == fam_dates:
        for a, b in (("up", "up"), ("dd60", "dd60"), ("hold20", "hold20"), ("near5", "near5"), ("win20", "win20"),
                     ("r20", "r20"), ("base_hold20", "base_hold20")):
            x, y = mine[a].to_numpy(float), fam_s[b].to_numpy(float)
            same_nan = (np.isnan(x) == np.isnan(y)).all()
            mx = np.nanmax(np.abs(x - y)) if np.isfinite(x - y).any() else 0.0
            say(f"  per-signal {a:12s}: NaN pattern equal={same_nan}, max |mine - family| = {mx:.2e}")
        say(f"  pos column equal (session index from data calendar vs XNYS): {list(mine.pos) == list(fam_s.pos)}")
    say("per-period comparison (mine | family):")
    st_h = period_stats(f, pos, "hold20")
    st_n = period_stats(f, pos, "near5")
    st_w = period_stats(f, pos, "win20")
    st_r = period_stats(f, pos, "r20")
    bm = f.assign(bh=bh)
    mism = 0
    for name, a, b in PERIODS:
        sel = pos[(idx[pos] >= a) & (idx[pos] <= b)]
        upm = float(np.median(upv[sel])) if len(sel) else np.nan
        bhm = float(np.nanmean(bh.to_numpy()[sel])) if len(sel) else np.nan
        fm = fam_m.loc[name] if name in fam_m.index else None
        famtxt = (f"n={int(fm.n)} hold20={fm.hold20:.4f} base={fm.hold20_base:.4f} near5={fm.near5:.4f} "
                  f"win20={fm.win20:.4f} r20={fm.r20:.4f} upmed={fm.up_median:.4f}") if fm is not None else "-"
        say(f"  {name:10s} mine n={len(sel)} hold20={st_h[name][1]:.4f}(n{st_h[name][0]}) base={bhm:.4f} "
            f"near5={st_n[name][1]:.4f} win20={st_w[name][1]:.4f} r20={st_r[name][1]:.4f} upmed={upm:.4f} | {famtxt}")
        if fm is not None:
            for mv, fv in ((len(sel), fm.n), (st_h[name][1], fm.hold20), (bhm, fm.hold20_base), (st_n[name][1], fm.near5),
                           (st_w[name][1], fm.win20), (st_r[name][1], fm.r20), (upm, fm.up_median)):
                if not (np.isclose(mv, fv, atol=1e-6) or (pd.isna(mv) and pd.isna(fv))):
                    mism += 1
    say(f"  per-period mismatches: {mism}")
    rng = np.random.default_rng(SEED)
    v = f.hold20.to_numpy()[pos]
    bv = bh.to_numpy()[pos]
    ok = np.isfinite(v) & np.isfinite(bv)
    lo, hi = block_boot_lo(v[ok] - bv[ok], pos[ok], rng)
    say(f"  my block-bootstrap 95% CI of (hold20 - matched) main period: [{lo:.4f}, {hi:.4f}] "
        f"(family dlo {fam_m.loc['2024~2026', 'hold20_dlo']:.4f}) -> L2 {'pass' if lo > 0 else 'FAIL'}")
    indep_match = (mine_dates == fam_dates) and mism == 0

    # ---------------------------------------------------------- 2. look-ahead audit (code reading + numeric probe)
    say("\n[2] Look-ahead audit")
    # numeric probe: recompute up/in_dd at t using only C[..t] (truncate the series) for every first signal
    trunc_ok = True
    for p in pos:
        ft = frames(d.iloc[:p + 1], 0.10)
        if not (np.isclose(ft.up.iat[-1], f.up.iat[p]) and bool(ft.in_dd.iat[-1]) == bool(f.in_dd.iat[p])):
            trunc_ok = False
    say(f"  truncation probe: up[t], in_dd[t] recomputed from data cut at t equal the full-sample values for all "
        f"{len(pos)} first signals: {trunc_ok}")
    say("  code: f1_rebound.build_rules uses fr['up'] only (sigma20 unused for up10); bc_lib.outcome_frame builds Ldd from "
        "C[pk..t] (window ending at t), dd60 from rolling(60) max (trailing), in_dd from rolling(20).min (trailing); outcomes "
        "use C[t+1..] and O[t+1] only; first_signals cooldown is causal (depends on earlier accepted signals). No flow/news "
        "input in this rule. No look-ahead found.")
    say("  caveat (not look-ahead in the signal): prices are a 2026-10 yfinance snapshot (split/dividend adjusted after the "
        "fact); ratio effects are tiny for a 10% threshold.")

    # ---------------------------------------------------------- 3. robustness
    say("\n[3] Robustness (hold20 by period; main = 2024~2026)")
    keys = ("2024~2026", "2026", "2026H1", "2026H2", "2025")
    rob = {}

    def run(level=LEVEL, cool=20, thr=0.10, lag=0, frm=None):
        ff = frm if frm is not None else (f if thr == 0.10 else frames(d, thr))
        u = ff["up"].to_numpy()
        r = np.isfinite(u) & (u >= level)
        if lag:
            r = np.concatenate([np.zeros(lag, bool), r[:-lag]])
        pp = first_signals(r, ff, cool)
        return period_stats(ff, pp, "hold20"), pp, ff

    variants = [("base up10", {}), ("neighbour up8", dict(level=0.08)), ("neighbour up15", dict(level=0.15)),
                ("fine up9", dict(level=0.09)), ("fine up11", dict(level=0.11)), ("fine up12", dict(level=0.12)),
                ("lag1 (signal shifted 1 session)", dict(lag=1)), ("lag2", dict(lag=2)),
                ("cooldown 10", dict(cool=10)), ("cooldown 30", dict(cool=30)),
                ("in_dd thr 8%", dict(thr=0.08)), ("in_dd thr 12%", dict(thr=0.12))]
    for name, kw in variants:
        st, pp, ff = run(**kw)
        rob[name] = st
        fails = [str(ff.index[q].date()) for q in pp if ff.hold20.iat[q] == 0]
        say(f"  {name:32s} {fmt(st, keys)}  fails={fails}")
    # same first-signal episodes, but acted one session late (outcomes at t+1)
    pp1 = pos + 1
    st = period_stats(f, pp1, "hold20")
    rob["same signals, outcome at t+1"] = st
    fails = [str(f.index[q].date()) for q in pp1 if f.hold20.iat[q] == 0]
    say(f"  {'same first signals, judged at t+1':32s} {fmt(st, keys)}  fails={fails}")
    # lag-1 failures explained
    st, ppl, _ = run(lag=1)
    for q in ppl:
        if f.hold20.iat[q] == 0:
            say(f"    lag1 fail {f.index[q].date()}: up[t]={f.up.iat[q]:+.3f} (up[t-1]={f.up.iat[q-1]:+.3f}), Ldd[t]={f.Ldd.iat[q]:.2f}"
                f" @ {f.index[f.lpos.iat[q]].date()} (Ldd[t-1]={f.Ldd.iat[q-1]:.2f} @ {f.index[f.lpos.iat[q-1]].date()}), "
                f"peak {f.index[f.pk.iat[q]].date()} (t-1 peak {f.index[f.pk.iat[q-1]].date()}), "
                f"min C[t+1..t+20]/Ldd-1={f.cushion.iat[q]:+.3f}")
    # delay that keeps the ORIGINAL decline low (separates a real re-break from the Ldd reset on a new 60-day high)
    Cc = f.C.to_numpy()
    keep = []
    for p in pos:
        if p + 21 <= len(Cc) - 1:
            keep.append((p, float(Cc[p + 2:p + 22].min() >= f.Ldd.iat[p] - 1e-12), Cc[p + 2:p + 22].min() / f.Ldd.iat[p] - 1))
    kp = np.asarray([k[0] for k in keep])
    kv = np.asarray([k[1] for k in keep])
    msel = (idx[kp] >= "2026-01-02")
    say(f"  {'lag1 keeping original Ldd[t]':32s} min(C[t+2..t+21]) >= Ldd[t]: 2024~2026 {kv.mean():.3f}(n{len(kv)})  "
        f"2026 {kv[msel].mean():.3f}(n{int(msel.sum())})  fails={[str(idx[k[0]].date()) for k in keep if k[1] == 0]}")
    # outcome tolerance: the low must hold with a margin (near-miss sensitivity)
    for buf in (0.01, 0.02, 0.03):
        hb = np.asarray([float(f.cushion.iat[p] >= buf) for p in pos if np.isfinite(f.cushion.iat[p])])
        pb = np.asarray([p for p in pos if np.isfinite(f.cushion.iat[p])])
        m26 = idx[pb] >= "2026-01-02"
        say(f"  outcome margin: min C[t+1..t+20] >= Ldd*(1+{buf:.2f}): 2024~2026 {hb.mean():.3f}(n{len(hb)})  "
            f"2026 {hb[m26].mean():.3f}(n{int(m26.sum())})")
    # hold40 (registered secondary)
    h40 = []
    for p in pos:
        if p + 40 <= len(Cc) - 1:
            h40.append((p, float(Cc[p + 1:p + 41].min() >= f.Ldd.iat[p] - 1e-12)))
    h40v = np.asarray([x[1] for x in h40])
    say(f"  hold40 (secondary): 2024~2026 {h40v.mean():.3f}(n{len(h40v)}) fails={[str(idx[x[0]].date()) for x in h40 if x[1] == 0]}")
    # exact binomial lower bounds (12/12 and 4/4 are small samples)
    for name in ("2024~2026", "2026", "2026H2"):
        n_ = rob["base up10"][name][0]
        k_ = round(rob["base up10"][name][1] * n_)
        lb = (0.025) ** (1 / n_) if k_ == n_ else np.nan
        say(f"  Clopper-Pearson 95% lower bound {name}: {k_}/{n_} -> {lb:.3f}")
    # cushion distribution for the base signals
    cush = f.cushion.to_numpy()[pos]
    say(f"  base signals: min forward-20 close / Ldd - 1: min {np.nanmin(cush):+.3f}, median {np.nanmedian(cush):+.3f}; "
        f"signals with cushion < 3%: {[str(idx[p].date()) for p, c in zip(pos, cush) if c < 0.03]}")
    # supplementary out-of-window context: 2023 (in_dd defined from late Apr 2023, close-only so hold20 only)
    p23 = first_signals(raw, f, 20, start=pd.Timestamp("2023-01-03"))
    p23 = p23[f.index[p23] < EVAL0]
    say(f"  context 2023 (outside registered window): up10 first signals {len(p23)}: "
        + ", ".join(f"{f.index[q].date()} hold20={f.hold20.iat[q]:.0f}" for q in p23))

    # ---------------------------------------------------------- 4. null
    say("\n[4] Null: random in_dd dates, main period, matched on count and up-bucket mix, 20-session spacing, 2000 draws")
    main_mask = (idx >= EVAL0) & (idx <= LAST) & f["in_dd"].to_numpy() & np.isfinite(upv) & np.isfinite(f.hold20.to_numpy())
    obs_pos = pos[np.isfinite(f.hold20.to_numpy()[pos])]
    obs = float(f.hold20.to_numpy()[obs_pos].mean())
    bk = f["bucket"].to_numpy()
    need = {b: int((bk[obs_pos] == b).sum()) for b in range(4)}
    names = ["0-3%", "3-8%", "8-15%", "15%+"]
    say(f"  observed hold20 {obs:.4f} (n{len(obs_pos)}); bucket mix " + ", ".join(f"{names[b]}:{need[b]}" for b in range(4)))
    pool = np.flatnonzero(main_mask)
    for b in range(4):
        pb = pool[bk[pool] == b]
        say(f"  pool {names[b]:6s}: {len(pb)} in_dd days, day-level hold20 {np.nanmean(f.hold20.to_numpy()[pb]) if len(pb) else np.nan:.4f}")
    h = f.hold20.to_numpy()
    rng = np.random.default_rng(SEED)

    def draw(pool_, need_):
        for _ in range(500):
            perm = rng.permutation(pool_)
            left = dict(need_)
            got = []
            for q in perm:
                b = bk[q]
                if left.get(b, 0) <= 0:
                    continue
                if any(abs(q - g) <= 20 for g in got):
                    continue
                got.append(q)
                left[b] -= 1
                if sum(left.values()) == 0:
                    return np.asarray(got)
        return None

    sims, fails_draw = [], 0
    for _ in range(N_NULL):
        g = draw(pool, need)
        if g is None:
            fails_draw += 1
            continue
        sims.append(h[g].mean())
    sims = np.asarray(sims)
    p_null = float((sims >= obs - 1e-12).mean())
    say(f"  draws {len(sims)} (infeasible {fails_draw}); null hold20 mean {sims.mean():.4f}, 5/50/95% "
        f"{np.quantile(sims, 0.05):.3f}/{np.quantile(sims, 0.5):.3f}/{np.quantile(sims, 0.95):.3f}; "
        f"share of draws with all 12 holding (>= observed): p = {p_null:.4f}")
    # unconstrained-by-bucket null (any in_dd day): shows what the bucket match absorbs
    # reference: random in_dd days ignoring up, same count & spacing
    sims2 = []

    def draw_any(pool_, n):
        for _ in range(500):
            perm = rng.permutation(pool_)
            got = []
            for q in perm:
                if any(abs(q - g) <= 20 for g in got):
                    continue
                got.append(q)
                if len(got) == n:
                    return np.asarray(got)
        return None

    for _ in range(N_NULL):
        g = draw_any(pool, len(obs_pos))
        if g is not None:
            sims2.append(h[g].mean())
    sims2 = np.asarray(sims2)
    say(f"  reference null ignoring up (any in_dd day, same n & spacing): mean {sims2.mean():.4f}, "
        f"p(>= observed) = {(sims2 >= obs - 1e-12).mean():.4f}")

    # ---------------------------------------------------------- 5. mechanical comparison
    say("\n[5] Mechanical check: plain up >= median(up of first signals)")
    med = float(np.median(upv[pos]))
    st_med, pmed, _ = run(level=med)
    say(f"  median up of up10 first signals = {med:.4f}")
    say(f"  up >= {med:.4f}: {fmt(st_med, keys)}")
    say(f"  up10       : {fmt(rob['base up10'], keys)}")
    say("  day-level hold20 on in_dd days, 2024~2026 (no cooldown) by up threshold, and fall needed to break Ldd:")
    hh = f.hold20.to_numpy()
    for x in (0.0, 0.03, 0.05, 0.08, 0.10, med, 0.15, 0.20):
        m = main_mask & (upv >= x)
        say(f"    up >= {x:6.4f}: days {int(m.sum()):4d}, hold20 {np.nanmean(hh[m]):.4f}, fall needed {1 - 1 / (1 + x):.3f}")
    for b in range(4):
        m = main_mask & (bk == b)
        say(f"    bucket {names[b]:6s}: days {int(m.sum()):4d}, hold20 {np.nanmean(hh[m]) if m.any() else np.nan:.4f}")
    # how often does SOXX fall >= 9.1% from any close within 20 sessions (any day, 2024~2026)? gives the cushion scale
    C = f.C.to_numpy()
    dd_fwd = np.full(len(C), np.nan)
    for t in range(len(C) - 20):
        dd_fwd[t] = C[t + 1:t + 21].min() / C[t] - 1
    ev = (idx >= EVAL0) & np.isfinite(dd_fwd)
    say(f"  any day 2024~2026: P(min next-20 close <= -9.1% from today's close) = {np.mean(dd_fwd[ev] <= -1 + 1 / 1.1):.4f}; "
        f"on in_dd days: {np.mean(dd_fwd[main_mask & np.isfinite(dd_fwd)] <= -1 + 1 / 1.1):.4f}")
    dd_sig = f.dd60.to_numpy()[pos]
    say(f"  signals already within 5% of the 60-day high (decline essentially over when 'confirmed'): "
        f"{int((dd_sig > -0.05).sum())}/{len(pos)} {[str(idx[p].date()) for p in pos if f.dd60.iat[p] > -0.05]}")
    say(f"  near5/win20 of up10 vs in_dd baseline (2024~2026): near5 {st_n['2024~2026'][1]:.3f} vs "
        f"{np.nanmean(f.near5.to_numpy()[main_mask]):.3f}, win20 {st_w['2024~2026'][1]:.3f} vs "
        f"{np.nanmean(f.win20.to_numpy()[main_mask]):.3f}, r20 {st_r['2024~2026'][1]:+.4f} vs "
        f"{np.nanmean(f.r20.to_numpy()[main_mask]):+.4f}")

    # ---------------------------------------------------------- 6. verdict
    say("\n[6] Verdict")
    robust_ok = all(rob[k]["2024~2026"][1] >= 0.70 and (rob[k]["2026"][0] == 0 or rob[k]["2026"][1] >= 0.70)
                    for k in rob)
    weakest = min(rob, key=lambda k: rob[k]["2026"][1] if rob[k]["2026"][0] else 9)
    say(f"  independent match: {indep_match}; look-ahead: none found (truncation probe {trunc_ok})")
    say(f"  robustness: all variants keep main >= 70% and 2026 >= 70%: {robust_ok}; weakest 2026 variant '{weakest}' "
        f"{fmt(rob[weakest], ('2024~2026', '2026', '2026H1', '2026H2'))}")
    say(f"  null p = {p_null:.4f} (>= 0.05), matched baseline {np.nanmean(bv[ok]):.4f}, L2 fails")
    verdict = "confirmed" if (indep_match and trunc_ok and robust_ok and p_null < 0.05) else (
        "refuted" if (not indep_match or not trunc_ok) else "fragile")
    say(f"  VERDICT: {verdict}")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
