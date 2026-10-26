"""Independent adversarial verification (round 2, fixed right-censoring) of f7 / c_up5_ma20 / SOXX (BRIEF.md 2026-10-26).

Candidate: up = C[t]/Ldd[t] - 1 >= 5% AND C[t] > SMA20[t] (mean of C[t-19..t]). SOXX, in_dd 10% (min dd60 over
[t-19, t] <= -10%), first signals 2024-01-02..2026-10-23, 20-session cooldown (a raw signal within <= 20 sessions of
the previous ACCEPTED signal is dropped). Primary outcome hold20 = no close in C[t+1..t+20] below Ldd[t].

Section [1] does NOT import bc_lib or any family module. It re-implements from the raw CSVs the SOXX series
(2023 closes from leverage_flood us_prices_close.csv, 2023-12-01.. OHLC from news_turning_points prices.csv),
the 60-session closing peak (latest on ties), Ldd, dd60, in_dd, SMA20, the rule, the cooldown, the outcomes with the
FIXED censoring rule (0 when a close below the level was already seen inside an incomplete forward window, NaN only
when nothing has broken yet), the year x up-bucket matched baseline and the 20-session calendar-block bootstrap (once
with an own RNG stream, once replaying the documented call order with seed 20261026 to check the published bounds),
then compares with results/f7_SOXX_signals.csv and results/f7_SOXX_summary.csv.

Section [2] (look-ahead audit) additionally runs the FAMILY code (bc_lib + f7_combo.build_rules) in a separate
subprocess, only to dump its raw rule series and a truncation test of the family state (prices cut at t).
Nothing from that subprocess feeds the independent numbers in [1], [3]-[6].

Sections: [1] independent match, [2] look-ahead / data audit, [3] robustness (neighbours, SMA x up grid, delay 1/2,
cooldown 10/30, in_dd 8%/12%, other instruments), [4] same-state random-date nulls (year x up bucket; same year and
up within +-2pp; +-2pp with up >= 5% to isolate the SMA20 leg; stratified on/off), [5] mechanical comparison with plain
up >= candidate median up, [6] stock replication (independent), [7] verdict inputs.

Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify2_f7_SOXX_c_up5_ma20.py
Out: results/verify2_f7_SOXX_c_up5_ma20.txt     (everything here is post hoc)
"""
from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
from math import comb

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify2_f7_SOXX_c_up5_ma20.txt"
RULE = "c_up5_ma20"

EVAL0 = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PER = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
       ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
       ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = (0.03, 0.08, 0.15)
BN = ("0-3%", "3-8%", "8-15%", "15%+")
OUTS = ("hold20", "hold40", "near5", "win20", "r20", "r60")
UPMIN, SMA_N, COOL, THR = 0.05, 20, 20, 0.10
NDRAW = 5000
SEED = 5_2026_1026
FAM_SEED, FAM_B = 20261026, 5000
N_CELLS = 109

_buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _buf.write(s + "\n")


# ====================================================================== raw data
def xnys(a, b) -> pd.DatetimeIndex:
    import exchange_calendars as xc
    s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(a, b))
    return s.tz_localize(None) if s.tz is not None else s


def read_soxx():
    raw = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[raw.index.notna()].sort_index()
    us = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    new_c = raw[("close", "SOXX")].dropna()
    old_c = us["SOXX"].dropna()
    close = pd.concat([old_c[old_c.index < new_c.index[0]], new_c]).sort_index()
    close = close[close.index <= LAST]
    opn = raw[("open", "SOXX")].reindex(close.index)
    ov = old_c.index.intersection(new_c.index)
    cal = xnys(close.index[0], close.index[-1])
    info = dict(ov=len(ov), ovdiff=float((old_c[ov] - new_c[ov]).abs().max()),
                miss=len(cal.difference(close.index)), extra=len(close.index.difference(cal)))
    return pd.DataFrame({"C": close.astype(float), "O": opn.astype(float)}), info, us


def read_stocks():
    o = pd.read_csv(ROOT / "data/derived/desk/prices_open.csv", index_col=0, parse_dates=True)
    c = pd.read_csv(ROOT / "data/derived/desk/prices_close.csv", index_col=0, parse_dates=True)
    drop = [x for x in c.columns if x.endswith(".KS") or x.endswith("=X") or x == "QQQ"]
    cal = xnys("2025-06-02", "2026-10-23")
    return o.drop(columns=drop).reindex(cal), c.drop(columns=drop).reindex(cal)


# ====================================================================== state and outcomes for one series
def sma(C: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(C), np.nan)
    for t in range(n - 1, len(C)):
        w = C[t - n + 1:t + 1]
        if not np.isnan(w).any():
            out[t] = float(np.mean(w))
    return out


def fwd_hold(C: np.ndarray, level: np.ndarray, L: int):
    """1 if no close in C[t+1..t+L] is below level[t] (full window), 0 if a close below was already seen
    (complete or not), NaN if the window is incomplete and nothing has broken yet. Also returns first break position."""
    N = len(C)
    out = np.full(N, np.nan)
    brk = np.full(N, -1)
    for t in range(N - 1):
        lv = level[t]
        if np.isnan(lv):
            continue
        hi = min(t + L, N - 1)
        seg = C[t + 1:hi + 1]
        fin = ~np.isnan(seg)
        below = np.flatnonzero(fin & (seg < lv - 1e-12))
        if len(below):
            out[t] = 0.0
            brk[t] = t + 1 + int(below[0])
        elif t + L <= N - 1 and fin.all():
            out[t] = 1.0
    return out, brk


def build(C: np.ndarray, O: np.ndarray, thr: float = THR) -> dict:
    """Everything that defines the signal is known at the close of t; outcomes look forward from t+1."""
    N = len(C)
    Ldd = np.full(N, np.nan)
    Lpos = np.full(N, -1)
    PK = np.full(N, -1)
    dd60 = np.full(N, np.nan)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        if np.isnan(w).any():
            continue
        top = w.max()
        pk = t - 59 + int(np.flatnonzero(w == top)[-1])                 # latest peak on ties
        seg = C[pk:t + 1]
        lo = seg.min()
        PK[t] = pk
        Ldd[t] = lo
        Lpos[t] = pk + int(np.flatnonzero(seg == lo)[-1])
        dd60[t] = C[t] / top - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    up = C / Ldd - 1
    E = np.full(N, np.nan)
    E[:-1] = O[1:]                                                    # entry = next session open
    h20, b20 = fwd_hold(C, Ldd, 20)
    h40, b40 = fwd_hold(C, Ldd, 40)
    n5, _ = fwd_hold(C, 0.95 * E, 40)
    r20 = np.full(N, np.nan)
    r60 = np.full(N, np.nan)
    r20[:N - 20] = C[20:] / E[:N - 20] - 1                            # C[t+20] = C[e+19]
    r60[:N - 60] = C[60:] / E[:N - 60] - 1
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    margin = np.full(N, np.nan)
    for t in range(N - 20):
        if not np.isnan(Ldd[t]):
            margin[t] = np.nanmin(C[t + 1:t + 21]) / Ldd[t] - 1
    s20 = sma(C, SMA_N)
    return dict(C=C, O=O, E=E, Ldd=Ldd, Lpos=Lpos, PK=PK, dd60=dd60, in_dd=in_dd, up=up, sma20=s20,
                hold20=h20, brk20=b20, hold40=h40, brk40=b40, near5=n5, win20=win20, r20=r20, r60=r60, margin=margin)


def rule(st, upmin=UPMIN, n=SMA_N, ge=False):
    s = st["sma20"] if n == SMA_N else sma(st["C"], n)
    u = np.where(np.isnan(st["up"]), -1.0, st["up"])
    above = (st["C"] >= s) if ge else (st["C"] > s)
    return (u >= upmin) & np.where(np.isnan(s), False, above)


def ubin(up):
    b = np.digitize(up, EDGES).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def matched(idx, st, key, eval0=EVAL0, slice_key=None):
    sk = idx.year.to_numpy() if slice_key is None else slice_key
    b = ubin(st["up"])
    pool = st["in_dd"] & ~np.isnan(st["up"]) & (idx >= eval0)
    v = st[key]
    tab = {}
    for s in np.unique(sk):
        for bb in range(4):
            m = pool & (sk == s) & (b == bb) & ~np.isnan(v)
            if m.any():
                tab[(s, bb)] = float(v[m].mean())
    out = np.full(len(idx), np.nan)
    for t in range(len(idx)):
        if not np.isnan(b[t]):
            out[t] = tab.get((sk[t], int(b[t])), np.nan)
    return out


def firsts(sig, st, idx, cool=COOL, eval0=EVAL0):
    ok = sig & st["in_dd"] & ~np.isnan(st["Ldd"]) & (idx >= eval0) & (idx <= LAST)
    keep, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev > cool:
            keep.append(int(p))
            prev = p
    return np.asarray(keep, int)


def lag(sig, d):
    return sig if d == 0 else np.r_[np.zeros(d, bool), sig[:-d]]


def boot(v, lab, rng, B=NDRAW):
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    s, c = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = s[pick].sum(1) / c[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def stats(pos, st, idx, base, seed=SEED, B=NDRAW, all_boot=False):
    """Per-period summary. all_boot=True replays the documented order (period x outcome x (level, diff))."""
    rng = np.random.default_rng(seed)
    d = idx[pos] if len(pos) else pd.DatetimeIndex([])
    res = {}
    for name, a, b in PER:
        pp = pos[(d >= a) & (d <= b)] if len(pos) else pos
        r = dict(n=int(len(pp)), up_med=float(np.median(st["up"][pp])) if len(pp) else np.nan,
                 dd_med=float(np.median(st["dd60"][pp])) if len(pp) else np.nan,
                 clusters=int(len(np.unique(pp // COOL))) if len(pp) else 0)
        for k in OUTS:
            v, bs = st[k][pp], base[k][pp]
            ok = ~np.isnan(v) & ~np.isnan(bs)
            r[f"{k}_n"] = int(ok.sum())
            r[k] = float(v[ok].mean()) if ok.any() else np.nan
            r[f"{k}_base"] = float(bs[ok].mean()) if ok.any() else np.nan
            if ok.any() and (all_boot or k == "hold20"):
                lab = pp[ok] // COOL
                r[f"{k}_lo"], r[f"{k}_hi"] = boot(v[ok], lab, rng, B)
                r[f"{k}_dlo"], r[f"{k}_dhi"] = boot(v[ok] - bs[ok], lab, rng, B)
        res[name] = r
    return res


def L1(s):
    m, a, b = s["2024~2026"], s["2026"], s["2025"]
    return bool(m["hold20_n"] >= 8 and m["hold20"] >= 0.70 and a["hold20_n"] >= 3 and a["hold20"] >= 0.70
                and (b["hold20_n"] < 3 or b["hold20"] >= 0.60))


def L2(s):
    return bool(L1(s) and s["2024~2026"].get("hold20_dlo", -1) > 0)


def evaluate(sig, st, idx, base, cool=COOL):
    pos = firsts(sig, st, idx, cool)
    return pos, stats(pos, st, idx, base)


def fmt(s, p):
    r = s[p]
    return f"{r['hold20']:.3f}({r['hold20_n']:2d})" if r["hold20_n"] else f"  -  ({r['n']:2d})"


def row(label, s):
    m = s["2024~2026"]
    return (f"  {label:32s} main {fmt(s, '2024~2026')} matched {m['hold20_base']:.3f} dlo {m.get('hold20_dlo', np.nan):+.3f} | "
            f"2026 {fmt(s, '2026')} H1 {fmt(s, '2026H1')} H2 {fmt(s, '2026H2')} 2025 {fmt(s, '2025')} 2024 {fmt(s, '2024')} | "
            f"upmed {m['up_med']:.3f} near5 {m['near5']:.2f} win20 {m['win20']:.2f} r20 {m['r20']:+.3f} L1={L1(s)} L2={L2(s)}")


def fails(pos, st, idx):
    return ", ".join(f"{idx[p].date()}(up {st['up'][p]:.3f}, C/SMA20 {st['C'][p] / st['sma20'][p] - 1:+.3f}, broke "
                     f"{idx[st['brk20'][p]].date() if st['brk20'][p] >= 0 else '?'})"
                     for p in pos if st["hold20"][p] == 0) or "none"


# ====================================================================== nulls
def seq_null(cands, y, cool, rng, ndraw=NDRAW):
    res, tries = [], 0
    K = len(cands)
    while len(res) < ndraw and tries < ndraw * 500:
        tries += 1
        ch = []
        for i in rng.permutation(K):
            c = cands[i]
            if ch:
                a = np.asarray(ch)
                c = c[np.abs(c[:, None] - a[None, :]).min(1) > cool]
            if not len(c):
                ch = None
                break
            ch.append(int(rng.choice(c)))
        if ch is not None:
            res.append(float(y[np.asarray(ch)].mean()))
    return np.asarray(res), tries


def one_spaced(cands, cool, rng):
    for _ in range(200000):
        ch = np.full(len(cands), -1)
        good = True
        for i in rng.permutation(len(cands)):
            c = cands[i]
            got = ch[ch >= 0]
            if len(got):
                c = c[np.abs(c[:, None] - got[None, :]).min(1) > cool]
            if not len(c):
                good = False
                break
            ch[i] = int(rng.choice(c))
        if good:
            return ch
    raise RuntimeError("no spaced configuration")


def gibbs_null(cands, y, cool, rng, nkeep=NDRAW, burn=1000, thin=4):
    """Uniform over spaced configurations (one day per slot, all pairs > cool apart)."""
    ch = one_spaced(cands, cool, rng).copy()
    out = []
    for sw in range(burn + nkeep * thin):
        for i in rng.permutation(len(cands)):
            oth = np.delete(ch, i)
            c = cands[i]
            ok = np.abs(c[:, None] - oth[None, :]).min(1) > cool if len(oth) else np.ones(len(c), bool)
            ch[i] = int(rng.choice(c[ok]))
        if sw >= burn and (sw - burn) % thin == 0:
            out.append(float(y[ch].mean()))
    return np.asarray(out)


# ====================================================================== family subprocess (audit only)
FAMILY_PROBE = r"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
import bc_lib, f7_combo
m = bc_lib.load_market(); notes = bc_lib.load_notes(); _, Ck = bc_lib.load_stocks()
ev, _ = f7_combo.market_events(m, Ck, notes)
d = bc_lib.load_soxx()
C = d[["C"]].rename(columns={"C": "SOXX"}); O = d[["O"]].rename(columns={"O": "SOXX"})
fr = bc_lib.soxx_frames()
rules, aux = f7_combo.build_rules(C, fr, ev)
sig = rules["c_up5_ma20"]["SOXX"].to_numpy()
test = [int(x) for x in sys.argv[2].split(",")]
trunc = {}
for t in test:
    c2, o2 = C.iloc[:t + 1], O.iloc[:t + 1]
    f2 = bc_lib.outcome_frame(o2, c2, 0.10)
    r2, _ = f7_combo.build_rules(c2, f2, ev)
    trunc[t] = [bool(r2["c_up5_ma20"]["SOXX"].iat[t]), bool(f2["in_dd"]["SOXX"].iat[t]),
                None if pd.isna(f2["L60"]["SOXX"].iat[t]) else float(f2["L60"]["SOXX"].iat[t]),
                None if pd.isna(f2["up"]["SOXX"].iat[t]) else float(f2["up"]["SOXX"].iat[t])]
print(json.dumps(dict(dates=[str(x.date()) for x in C.index], sig=sig.astype(int).tolist(),
                      in_dd=fr["in_dd"]["SOXX"].astype(int).tolist(),
                      sma20=[None if pd.isna(x) else float(x) for x in aux["sma20"]["SOXX"]],
                      L60=[None if pd.isna(x) else float(x) for x in fr["L60"]["SOXX"]], trunc=trunc)))
"""


# ====================================================================== main
def main():
    pd.set_option("display.width", 260)
    df, info, us = read_soxx()
    idx = df.index
    C, O = df.C.to_numpy(float), df.O.to_numpy(float)
    N = len(C)
    say("=" * 130)
    say("Independent verification round 2 (fixed right-censoring): family f7, rule c_up5_ma20, kind SOXX (bc_lib NOT imported in [1])")
    say("Rule: up = C[t]/Ldd[t] - 1 >= 5% AND C[t] > SMA20[t] (mean C[t-19..t]); first signals on in_dd(10%) days")
    say("      2024-01-02..2026-10-23, 20-session cooldown. hold20 = no close in C[t+1..t+20] below Ldd[t].")
    say("=" * 130)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()} N={N}; open from {df.O.first_valid_index().date()}; overlap rows "
        f"{info['ov']} max|diff|={info['ovdiff']:.1e}; XNYS sessions missing={info['miss']} extra={info['extra']}")

    st = build(C, O)
    base = {k: matched(idx, st, k) for k in OUTS}
    sig = rule(st)
    pos, s0 = evaluate(sig, st, idx, base)

    # ------------------------------------------------------------ [1]
    say("\n[1] Independent re-implementation vs family files (results/f7_SOXX_signals.csv, f7_SOXX_summary.csv)")
    fam = pd.read_csv(RES / "f7_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == RULE].reset_index(drop=True)
    fsum = pd.read_csv(RES / "f7_SOXX_summary.csv")
    fsum = fsum[fsum.rule == RULE].set_index("period")
    mine = pd.DataFrame(dict(date=idx[pos], pos=pos, Ldd_date=[idx[st["Lpos"][p]].date() for p in pos],
                             Ldd=st["Ldd"][pos], C=C[pos], SMA20=st["sma20"][pos],
                             c_sma=C[pos] / st["sma20"][pos] - 1, up=st["up"][pos], dd60=st["dd60"][pos],
                             **{k: st[k][pos] for k in OUTS}, **{f"base_{k}": base[k][pos] for k in OUTS},
                             margin20=st["margin"][pos]))
    show = mine.copy()
    show["date"] = [x.date() for x in show.date]
    show["broke20"] = [str(idx[st["brk20"][p]].date()) if st["brk20"][p] >= 0 else "-" for p in pos]
    show["bucket"] = [BN[int(b)] for b in ubin(st["up"][pos])]
    say(show[["date", "pos", "Ldd_date", "Ldd", "C", "SMA20", "c_sma", "up", "bucket", "dd60", "hold20", "base_hold20",
              "broke20", "margin20", "hold40", "near5", "win20", "r20", "r60"]].round(4).to_string(index=False))
    same_dates = list(mine.date) == list(fam.date)
    say(f"  first-signal dates identical: {same_dates} (mine {len(mine)}, family {len(fam)})")
    mism = []
    if same_dates:
        for k in ("pos", "up", "dd60") + OUTS + tuple(f"base_{k}" for k in OUTS):
            a, b = mine[k].to_numpy(float), fam[k].to_numpy(float)
            bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
            if bad.any():
                mism.append((k, int(bad.sum())))
        say(f"  per-signal field mismatches (pos, up, dd60, 6 outcomes, 6 matched bases): {mism or 'none'}")
    else:
        say(f"  only mine: {sorted(set(mine.date) - set(fam.date))}; only family: {sorted(set(fam.date) - set(mine.date))}")
    sx = stats(pos, st, idx, base, seed=FAM_SEED, B=FAM_B, all_boot=True)      # documented bootstrap call order replayed
    say("  period      mine hold20(n)  fam hold20(n)   mine base  fam base | mine lo/dlo (replayed seed)  fam lo/dlo | others")
    per_ok, boot_ok = True, True
    for p, _, _ in PER:
        r = s0[p]
        okp = r["hold20_n"] == int(fsum.loc[p, "hold20_n"]) and r["n"] == int(fsum.loc[p, "n"])
        for k in OUTS:
            for suf in ("", "_base"):
                a, b = r[k + suf], fsum.loc[p, k + suf] if (k + suf) in fsum.columns else np.nan
                okp &= bool((np.isnan(a) and pd.isna(b)) or (not pd.isna(b) and abs(a - b) < 1e-9))
        okp &= abs(r["up_med"] - fsum.loc[p, "up_median"]) < 1e-9 and r["clusters"] == int(fsum.loc[p, "clusters"])
        for k in OUTS:
            for suf in ("_lo", "_hi", "_dlo", "_dhi"):
                a = sx[p].get(k + suf, np.nan)
                b = fsum.loc[p, k + suf] if (k + suf) in fsum.columns else np.nan
                boot_ok &= bool((pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(a - b) < 1e-9))
        per_ok &= bool(okp)
        say(f"  {p:10s}  {r['hold20']:.4f} ({r['hold20_n']:2d})  {fsum.loc[p, 'hold20']:.4f} ({int(fsum.loc[p, 'hold20_n']):2d})"
            f"   {r['hold20_base']:.4f}   {fsum.loc[p, 'hold20_base']:.4f} | {sx[p].get('hold20_lo', np.nan):.3f}/"
            f"{sx[p].get('hold20_dlo', np.nan):+.3f}   {fsum.loc[p, 'hold20_lo']:.3f}/{fsum.loc[p, 'hold20_dlo']:+.3f} | "
            f"n, clusters, up_median, hold40/near5/win20/r20/r60 (+bases) {'OK' if okp else 'MISMATCH'}")
    indep = bool(same_dates and not mism and per_ok)
    say(f"  => independent match: {indep}; bootstrap bounds (all outcomes, replayed seed/order) identical: {boot_ok}")
    say(f"     own RNG stream: main hold20 95% [{s0['2024~2026']['hold20_lo']:.3f}, {s0['2024~2026']['hold20_hi']:.3f}], "
        f"diff vs matched [{s0['2024~2026']['hold20_dlo']:+.3f}, {s0['2024~2026']['hold20_dhi']:+.3f}]")
    m0 = s0["2024~2026"]
    poolm = st["in_dd"] & ~np.isnan(st["hold20"]) & ~np.isnan(st["up"]) & (idx >= EVAL0) & (idx <= LAST)
    unc = float(st["hold20"][poolm].mean())
    say(f"  main n={m0['n']} hold20={m0['hold20']:.4f} vs matched {m0['hold20_base']:.4f} (diff "
        f"{m0['hold20'] - m0['hold20_base']:+.4f}); near5 {m0['near5']:.3f} (matched {m0['near5_base']:.3f}); win20 "
        f"{m0['win20']:.3f} ({m0['win20_base']:.3f}); r20 {m0['r20']:+.4f} ({m0['r20_base']:+.4f}); hold40 {m0['hold40']:.3f} "
        f"({m0['hold40_base']:.3f}); up median {m0['up_med']:.4f}")
    say(f"  unconditional in_dd hold20 2024~2026 = {unc:.4f} (resolved days {int(poolm.sum())})")
    kk, nn = int(round(m0["hold20"] * m0["hold20_n"])), m0["hold20_n"]
    for lab, p0 in (("unconditional in_dd", unc), ("matched mean", m0["hold20_base"])):
        pb = sum(comb(nn, j) * p0 ** j * (1 - p0) ** (nn - j) for j in range(kk, nn + 1))
        say(f"  binomial P(X >= {kk}/{nn} | p={p0:.4f}, {lab}) = {pb:.4f}")
    a26 = s0["2026"]
    k26 = int(round(a26["hold20"] * a26["hold20_n"]))
    say(f"  2026 alone: {k26}/{a26['hold20_n']} = {a26['hold20']:.3f} (matched {a26['hold20_base']:.3f}); one more failure -> "
        f"{(k26 - 1) / a26['hold20_n']:.3f} (L1 lost); 2026H1 {fmt(s0, '2026H1')}, 2026H2 {fmt(s0, '2026H2')}")
    say("  successes held by a thin margin (min close of t+1..t+20 within 2% above Ldd): " +
        (", ".join(f"{idx[p].date()} {st['margin'][p] * 100:+.2f}%" for p in pos
                   if st["hold20"][p] == 1 and st["margin"][p] < 0.02) or "none"))
    say("  failures, depth of the break (min close t+1..t+20 vs Ldd): " +
        (", ".join(f"{idx[p].date()} {st['margin'][p] * 100:+.2f}%" for p in pos if st["hold20"][p] == 0) or "none"))
    sma_close = [(idx[p].date(), C[p] / st["sma20"][p] - 1) for p in pos if abs(C[p] / st["sma20"][p] - 1) < 0.003]
    say(f"  first signals within 0.3% of the SMA20 (float/adjustment sensitive): {sma_close or 'none'}")
    ub = ubin(st["up"])
    for p in pos:
        for e in (UPMIN,) + EDGES:
            if abs(st["up"][p] - e) < 3e-3:
                say(f"  edge check: {idx[p].date()} up={st['up'][p]:.6f} near edge {e} (bucket {BN[int(ub[p])]}, matched "
                    f"{base['hold20'][p]:.4f})")
    say(f"  state on the last day {idx[-1].date()}: in_dd={st['in_dd'][-1]} up={st['up'][-1]:+.4f} C/SMA20="
        f"{C[-1] / st['sma20'][-1] - 1:+.4f} Ldd={st['Ldd'][-1]:.2f} ({idx[st['Lpos'][-1]].date()}) rule={bool(sig[-1])}")
    later = np.flatnonzero(sig & st["in_dd"] & (np.arange(N) > pos[-1]) & (idx >= EVAL0))
    say(f"  raw rule & in_dd days after the last first signal {idx[pos[-1]].date()}: " +
        (", ".join(str(idx[q].date()) for q in later) if len(later) else "none"))
    raw_days = np.flatnonzero(sig & st["in_dd"] & ~np.isnan(st["Ldd"]) & (idx >= EVAL0))
    rh = st["hold20"][raw_days]
    say(f"  all raw rule & in_dd days (no cooldown): {len(raw_days)}, hold20 {np.nanmean(rh):.3f} (resolved "
        f"{int((~np.isnan(rh)).sum())}), matched {np.nanmean(base['hold20'][raw_days][~np.isnan(rh)]):.3f}")

    # ------------------------------------------------------------ [2]
    say("\n[2] Look-ahead / timing / data audit")
    test_pos = np.unique(np.r_[np.flatnonzero(sig & (idx >= EVAL0)), np.arange(np.searchsorted(idx, EVAL0), N, 5)])
    bad = 0
    for p in test_pos:
        s2 = build(C[:p + 1], O[:p + 1])
        r2 = rule(s2)
        if not (s2["in_dd"][p] == st["in_dd"][p] and s2["Lpos"][p] == st["Lpos"][p] and abs(s2["up"][p] - st["up"][p]) < 1e-12
                and abs(s2["sma20"][p] - st["sma20"][p]) < 1e-9 and r2[p] == sig[p]):
            bad += 1
    say(f"  own implementation, truncation (prices cut at t) on {len(test_pos)} days: mismatches {bad}")
    ties = int(sum(1 for t in range(N) if st["Lpos"][t] >= 0 and
                   (np.abs(C[st["PK"][t]:t + 1] - st["Ldd"][t]) < 1e-12).sum() > 1))
    say(f"  days whose decline low is tied (several closes equal to Ldd): {ties}")
    sample = sorted(set(int(x) for x in np.r_[pos, np.flatnonzero(sig & (idx >= EVAL0))[::6], np.arange(300, N, 31)]))
    try:
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", FAMILY_PROBE, str(HERE / "code"), ",".join(map(str, sample))],
                           capture_output=True, text=True, timeout=1800, cwd=str(HERE))
        fp = json.loads(r.stdout.strip().splitlines()[-1])
        fsig = np.asarray(fp["sig"], bool)
        fin = np.asarray(fp["in_dd"], bool)
        fsma = np.array([np.nan if x is None else x for x in fp["sma20"]])
        same_idx = [str(x.date()) for x in idx] == fp["dates"]
        say(f"  family code (subprocess: bc_lib.soxx_frames + f7_combo.build_rules): same session index {same_idx}; raw rule days "
            f"differing from mine {int((fsig != sig).sum())} of {N}; in_dd differing {int((fin != st['in_dd']).sum())}; "
            f"max |SMA20 diff| {np.nanmax(np.abs(fsma - st['sma20'])):.2e}")
        tb = 0
        for t, (rs, ind, l60, u2) in fp["trunc"].items():
            t = int(t)
            if rs != bool(fsig[t]) or ind != bool(fin[t]) or (l60 is None) != (fp["L60"][t] is None) or \
                    (l60 is not None and abs(l60 - fp["L60"][t]) > 1e-9):
                tb += 1
        say(f"  family code truncation test (prices cut at t, bc_lib.outcome_frame + f7_combo.build_rules re-run) on "
            f"{len(fp['trunc'])} days incl. all {len(pos)} first signals: mismatches {tb}")
    except Exception as ex:                                              # noqa: BLE001
        say(f"  family subprocess failed: {ex!r}; stderr tail: {r.stderr[-500:] if 'r' in dir() else ''}")
    say("  code read (f7_combo.py): c_up5_ma20 = (fr['up'] >= 0.05) & (C > C.rolling(20, min_periods=20).mean()); fr['up'] =")
    say("  C/Ldd - 1 with Ldd from the 60-session window ending at t (bc_lib.outcome_frame). No VIX, breadth, note or flow")
    say("  input enters this rule (those only feed the other 4 F7 rules); KNOWN_LOWS (hindsight lows) only feed a printout.")
    say("  bc_lib.first_signals: in_dd and Ldd at t; window 2024-01-02..2026-10-23; cooldown drops p - prev <= 20.")
    say("  bc_lib.outcome_frame: forward windows C[t+1..t+20]/[t+1..t+40]; entry O[t+1]; censoring 0 if already broken, NaN if not.")
    say("  The matched baseline averages outcomes of other in_dd days of the same calendar year (it uses future days' outcomes,")
    say("  which is fine for a comparison baseline but means it is not a tradeable benchmark).")
    lr = np.diff(np.log(C))
    say("  SOXX |daily log return| > 7%: " + ", ".join(f"{idx[i + 1].date()} {lr[i]:+.3f}" for i in np.flatnonzero(np.abs(lr) > 0.07)))
    gp = O[1:] / C[:-1] - 1
    say("  SOXX |open/prev close| > 6%: " + ", ".join(f"{idx[i + 1].date()} {gp[i]:+.3f}" for i in np.flatnonzero(np.abs(gp) > 0.06)))
    say(f"  O[t+1] present for every first signal: {bool(np.isfinite(O[pos + 1]).all())}")
    sox = us["^SOX"].reindex(idx)
    smh = us["SMH"].reindex(idx)
    rs, rx, rm = df.C.pct_change(), sox.pct_change(), smh.pct_change()
    dif = (rs - rx).abs()
    say(f"  days where SOXX and ^SOX daily returns differ by > 1.5pp (^SOX to {sox.last_valid_index().date()}): " +
        (", ".join(f"{x.date()} {v * 100:.1f}pp" for x, v in dif[dif > 0.015].items()) or "none"))
    big_after = [i for i in np.flatnonzero(np.abs(lr) > 0.07) if idx[i + 1] > sox.last_valid_index()]
    say(f"  big SOXX moves after the ^SOX/SMH end (cannot be cross-checked): " +
        (", ".join(f"{idx[i + 1].date()} {lr[i]:+.3f}" for i in big_after) or "none"))
    say("  for every first signal: decline-low day return in/out, the 20 SMA20 inputs vs ^SOX-implied, and the break day:")
    for p in pos:
        lp = st["Lpos"][p]
        a = C[lp] / C[lp - 1] - 1
        b = C[lp + 1] / C[lp] - 1
        xa = sox.iloc[lp] / sox.iloc[lp - 1] - 1
        xb = sox.iloc[lp + 1] / sox.iloc[lp] - 1
        w = slice(p - 19, p + 1)
        rr = (rs.iloc[w] - rx.iloc[w]).abs().max()
        bk = st["brk20"][p]
        bks = (f"; break {idx[bk].date()} C {C[bk]:.2f} vs Ldd {st['Ldd'][p]:.2f} (SOXX {C[bk] / st['Ldd'][p] - 1:+.4f}, "
               f"^SOX same days {sox.iloc[bk] / sox.iloc[lp] - 1:+.4f}, SMH {smh.iloc[bk] / smh.iloc[lp] - 1:+.4f})") if bk >= 0 else ""
        say(f"    {idx[p].date()}: low {idx[lp].date()} in {a:+.3f} out {b:+.3f} (^SOX {xa:+.3f}/{xb:+.3f}); "
            f"max |SOXX-^SOX| daily ret in SMA window {rr * 100:.2f}pp{bks}")
    say("  dividend back-adjustment: closes are adjusted (later distributions scale earlier closes down ~0.1-0.2%);")
    say("  see the thin-margin list in [1] for whether that could flip any hold20 outcome.")

    # ------------------------------------------------------------ [3]
    say("\n[3] Robustness (hold20, resolved n; matched = year x up bucket; dlo = block-bootstrap lower bound of hold20 - matched)")
    say(row("registered c_up5_ma20", s0))
    say(f"      failed: {fails(pos, st, idx)}")
    core = {}
    say("  -- timing (raw signal shifted d sessions; state, Ldd and outcomes read at the delayed day, as the family lag1)")
    for d in (1, 2):
        p_, s_ = evaluate(lag(sig, d), st, idx, base)
        core[f"delay +{d}"] = s_
        say(row(f"delay +{d}", s_))
        say(f"      failed: {fails(p_, st, idx)}")
    say("  -- timing variant: entry delayed d sessions, level stays the signal-day Ldd (hold over t+d+1..t+d+20)")
    for d in (1, 2):
        v = np.full(len(pos), np.nan)
        for i, p in enumerate(pos):
            q = p + d
            seg = C[q + 1:min(q + 20, N - 1) + 1]
            if (seg < st["Ldd"][p] - 1e-12).any():
                v[i] = 0.0
            elif q + 20 <= N - 1:
                v[i] = 1.0
        yy = idx[pos].year
        say(f"    delay +{d} keep-Ldd: main {np.nanmean(v):.3f} ({int((~np.isnan(v)).sum())}) | 2026 "
            f"{np.nanmean(v[yy == 2026]):.3f} | 2025 {np.nanmean(v[yy == 2025]):.3f} | 2024 {np.nanmean(v[yy == 2024]):.3f}")
    say("  -- cooldown / in_dd threshold / >= instead of >")
    for cl in (10, 30):
        p_, s_ = evaluate(sig, st, idx, base, cool=cl)
        core[f"cooldown {cl}"] = s_
        say(row(f"cooldown {cl}", s_))
        say(f"      failed: {fails(p_, st, idx)}")
    for th in (0.08, 0.12):
        st2 = build(C, O, th)
        b2 = {k: matched(idx, st2, k) for k in OUTS}
        p_, s_ = evaluate(rule(st2), st2, idx, b2)
        core[f"in_dd {th:.0%}"] = s_
        say(row(f"in_dd thr {th:.0%}", s_))
        say(f"      failed: {fails(p_, st2, idx)}")
    p_, s_ = evaluate(rule(st, ge=True), st, idx, base)
    say(row("C >= SMA20 (tie rule)", s_))
    say("  -- single-step neighbours (as the family post hoc) and SMA lengths")
    for lab, (u, n) in {"up4_ma20": (0.04, 20), "up6_ma20": (0.06, 20), "up5_ma10": (0.05, 10), "up5_ma50": (0.05, 50),
                        "up5_ma15": (0.05, 15), "up5_ma19": (0.05, 19), "up5_ma21": (0.05, 21), "up5_ma25": (0.05, 25),
                        "up5_ma30": (0.05, 30)}.items():
        p_, s_ = evaluate(rule(st, u, n), st, idx, base)
        core[lab] = s_
        say(row(lab, s_))
        say(f"      failed: {fails(p_, st, idx)}")
    say("  -- neighbour grid (post hoc): rows SMA n, cols up >= 3..8% -> main hold20(n) / 2026 hold20(n) / matched [*=L1, **=L2]")
    ups = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    ns = (10, 15, 18, 19, 20, 21, 22, 25, 30, 50)
    gl1 = gl2 = ggt = 0
    gdiff = []
    say("           " + "".join(f"up>={u:.0%}".ljust(27) for u in ups))
    for n in ns:
        cells = []
        for u in ups:
            _, s_ = evaluate(rule(st, u, n), st, idx, base)
            m = s_["2024~2026"]
            l1, l2 = L1(s_), L2(s_)
            gl1 += l1
            gl2 += l2
            ggt += m["hold20"] > m["hold20_base"]
            gdiff.append(m["hold20"] - m["hold20_base"])
            cells.append(f"{m['hold20']:.2f}({m['hold20_n']:2d})/{s_['2026']['hold20']:.2f}({s_['2026']['hold20_n']})/"
                         f"{m['hold20_base']:.2f}{'**' if l2 else '*' if l1 else ''}".ljust(27))
        say(f"    sma{n:<5d}" + "".join(cells))
    ncell = len(ups) * len(ns)
    say(f"    L1 in {gl1}/{ncell}; L2 in {gl2}/{ncell}; cells with main hold20 > matched: {ggt}/{ncell}; mean(main - matched) "
        f"{np.mean(gdiff):+.3f}")
    say("  -- other instruments, same rule and engine (closes only; ^SOX/SMH to 2026-10-13, outcomes censored there)")
    for nm, ser in (("^SOX index", sox), ("SMH ETF", smh)):
        c2 = ser.dropna()
        c2 = c2[c2.index <= LAST]
        ix2 = c2.index
        st3 = build(c2.to_numpy(float), c2.to_numpy(float))
        b3 = {k: matched(ix2, st3, k) for k in OUTS}
        p3, s3 = evaluate(rule(st3), st3, ix2, b3)
        say(row(nm, s3) + "  (near5/win20/r20 use the next close as entry proxy)")
        say("      first signals: " + ", ".join(f"{ix2[p].date()}:{'NaN' if np.isnan(st3['hold20'][p]) else int(st3['hold20'][p])}"
                                              for p in p3))
        core[nm] = s3
    lost = [k for k, v in core.items() if not L1(v)]
    say(f"  core variants: L1 in {sum(L1(v) for v in core.values())}/{len(core)}; L2 in {sum(L2(v) for v in core.values())}/{len(core)};"
        f" L1 lost: {lost}")

    # ------------------------------------------------------------ [4]
    say("\n[4] Same-state null: random in_dd days 2024-01-02..2026-10-23 (resolved hold20), one per first signal, matched on")
    say("    the signal's year and up bucket (0-3/3-8/8-15/15%+), all pairs > 20 sessions apart; p = P(null mean >= observed)")
    rng = np.random.default_rng(SEED)
    y = idx.year.to_numpy()
    h = st["hold20"]
    obs = float(np.nanmean(h[pos]))
    keys = [(int(y[p]), int(ub[p])) for p in pos]
    say(f"  observed hold20 = {obs:.4f} (n={len(pos)}); year x bucket = {[(a, BN[b]) for a, b in keys]}")
    say("  pool hold20 by year x bucket (days): " + "; ".join(
        f"{yy}/{BN[bb]} {np.mean(h[poolm & (y == yy) & (ub == bb)]):.3f}({int((poolm & (y == yy) & (ub == bb)).sum())})"
        for yy in (2024, 2025, 2026) for bb in range(4) if (poolm & (y == yy) & (ub == bb)).any()))
    cands = [np.flatnonzero(poolm & (y == yy) & (ub == bb)) for yy, bb in keys]
    n1, t1 = seq_null(cands, h, COOL, rng)
    p1 = float((n1 >= obs - 1e-12).mean())
    say(f"  [N1] sequential spaced sampler: mean {n1.mean():.4f} sd {n1.std():.4f} [2.5%, 97.5%] "
        f"[{np.quantile(n1, .025):.3f}, {np.quantile(n1, .975):.3f}]  p = {p1:.4f}  (draws {len(n1)}, attempts {t1})")
    g1 = gibbs_null(cands, h, COOL, rng)
    pg1 = float((g1 >= obs - 1e-12).mean())
    say(f"  [N1g] Gibbs (uniform over spaced sets): mean {g1.mean():.4f} p = {pg1:.4f} (kept {len(g1)})")
    ni = np.array([h[[int(rng.choice(c)) for c in cands]].mean() for _ in range(NDRAW)])
    say(f"  [N1i] independent draws (no spacing): mean {ni.mean():.4f} p = {float((ni >= obs - 1e-12).mean()):.4f}")
    fine = [np.flatnonzero(poolm & (y == y[p]) & (np.abs(st["up"] - st["up"][p]) <= 0.02)) for p in pos]
    say("  fine pools (same year, up within +-2pp): " + ", ".join(
        f"{idx[p].date()}:{len(c)}(h {np.mean(h[c]):.2f})" for p, c in zip(pos, fine)))
    n2, t2 = seq_null(fine, h, COOL, rng)
    p2 = float((n2 >= obs - 1e-12).mean())
    g2 = gibbs_null(fine, h, COOL, rng)
    pg2 = float((g2 >= obs - 1e-12).mean())
    say(f"  [N2] same year, up +-2pp: sequential mean {n2.mean():.4f} sd {n2.std():.4f} p = {p2:.4f} (attempts {t2}); "
        f"Gibbs mean {g2.mean():.4f} p = {pg2:.4f}")
    fine5 = [np.flatnonzero(poolm & (y == y[p]) & (np.abs(st["up"] - st["up"][p]) <= 0.02) & (st["up"] >= UPMIN))
             for p in pos]
    n3, _ = seq_null(fine5, h, COOL, rng)
    p3_ = float((n3 >= obs - 1e-12).mean())
    say(f"  [N2b] same year, up +-2pp AND up >= 5% (the rule's own floor; isolates the 'C > SMA20' leg): mean {n3.mean():.4f} "
        f"p = {p3_:.4f}")
    i26 = [i for i, p in enumerate(pos) if y[p] == 2026]
    o26 = float(np.mean(h[pos[i26]]))
    n4, _ = seq_null([cands[i] for i in i26], h, COOL, rng)
    n5, _ = seq_null([fine[i] for i in i26], h, COOL, rng)
    say(f"  [N3] 2026 only (n={len(i26)}, obs {o26:.3f}): year x bucket mean {n4.mean():.3f} p = {float((n4 >= o26 - 1e-12).mean()):.4f};"
        f" +-2pp mean {n5.mean():.3f} p = {float((n5 >= o26 - 1e-12).mean()):.4f}")
    say("  [N4] stratified on/off (all resolved in_dd days 2024~2026, no cooldown): among days with up >= 5%, within year x")
    say("       fine up bins (5-6.5/6.5-8/8-10/10-15/15%+), hold20 of C > SMA20 (rule on) vs C <= SMA20 (off)")
    fb = np.digitize(st["up"], (0.065, 0.08, 0.10, 0.15))
    fbn = ("5-6.5%", "6.5-8%", "8-10%", "10-15%", "15%+")
    above = np.where(np.isnan(st["sma20"]), False, C > st["sma20"])
    rows_ = []
    on = poolm & (st["up"] >= UPMIN)
    for yy in (2024, 2025, 2026):
        for bb in range(5):
            m = on & (y == yy) & (fb == bb)
            a, b = m & above, m & ~above
            rows_.append(dict(year=yy, bin=fbn[bb], n_on=int(a.sum()), h_on=np.mean(h[a]) if a.any() else np.nan,
                              n_off=int(b.sum()), h_off=np.mean(h[b]) if b.any() else np.nan))
    tb = pd.DataFrame(rows_)
    say(tb.round(3).to_string(index=False))
    both = tb[(tb.n_on > 0) & (tb.n_off > 0)]
    wd = float(np.average(both.h_on - both.h_off, weights=both.n_on)) if len(both) else np.nan
    say(f"       weighted (on - off) difference over strata with both: {wd:+.3f} (strata {len(both)}, on-days "
        f"{int(both.n_on.sum())}, off-days {int(both.n_off.sum())})")
    a_, b_ = on & above, on & ~above
    say(f"       pooled: up>=5% & C>SMA20 hold20 {np.mean(h[a_]):.3f} (n={int(a_.sum())}) vs up>=5% & C<=SMA20 "
        f"{np.mean(h[b_]):.3f} (n={int(b_.sum())}); on-days mean up {np.mean(st['up'][a_]):.3f}, off-days {np.mean(st['up'][b_]):.3f}")
    inf = [(str(idx[p].date()), BN[int(ub[p])], round(float(base['hold20'][p]), 3), h[p]) for p in pos if base["hold20"][p] < 0.95]
    exp_ = sum(x[2] for x in inf)
    say(f"  informative signals (matched base < 0.95): {len(inf)}; held {int(sum(x[3] for x in inf))}/{len(inf)}, "
        f"expected {exp_:.2f}; {inf}")
    say(f"  multiple testing: {N_CELLS} registered cells; Bonferroni 0.05/{N_CELLS} = {0.05 / N_CELLS:.5f}.")

    # ------------------------------------------------------------ [4b]
    say("\n[4b] Extra adversarial checks")
    bp = base["hold20"][pos]
    sim = rng.random((200000, len(pos))) < bp[None, :]
    yy = idx[pos].year.to_numpy()
    m_all = sim.mean(1)
    m26 = sim[:, yy == 2026].mean(1)
    m25 = sim[:, yy == 2025].mean(1)
    pass_l1 = (m_all >= 0.70) & (m26 >= 0.70) & ((m25 >= 0.60) | ((yy == 2025).sum() < 3))
    say(f"  chance of L1 if each signal held with its own matched probability (rebound size alone, 200k sims): "
        f"{pass_l1.mean():.3f}; P(main >= observed {obs:.3f}) = {(m_all >= obs - 1e-12).mean():.3f}")
    say("  SOXX first-signal dates re-scored on sister instruments (their own closes, level = their close on SOXX's Ldd day,")
    say("  window t+1..t+20 sessions; ^SOX/SMH to 2026-10-13):")
    for nm, ser in (("^SOX", sox), ("SMH", smh)):
        sv = ser.to_numpy(float)
        res_ = []
        for p in pos:
            lv = sv[st["Lpos"][p]]
            seg = sv[p + 1:p + 21]
            seg = seg[~np.isnan(seg)]
            res_.append((idx[p].date(), int(seg.min() >= lv - 1e-12) if len(seg) == 20 or (seg < lv).any() else None,
                         seg.min() / lv - 1 if len(seg) else np.nan))
        ok_ = [r_[1] for r_ in res_ if r_[1] is not None]
        say(f"    {nm}: held {sum(ok_)}/{len(ok_)}; differs from SOXX on: " +
            (", ".join(f"{d_} ({nm} min {m_ * 100:+.2f}%)" for (d_, o_, m_), p in zip(res_, pos)
                       if o_ is not None and o_ != h[p]) or "none") +
            "; 2026: " + ", ".join(f"{d_} {o_} ({m_ * 100:+.2f}%)" for d_, o_, m_ in res_ if d_.year == 2026))
    near_hi = [p for p in pos if st["dd60"][p] > -0.05]
    keep = np.asarray([p for p in pos if st["dd60"][p] <= -0.05], int)
    sk_ = stats(keep, st, idx, base)
    say(f"  signals within 5% of the 60-day high (dd60 > -5%, 'breakout holds' rather than 'bottom is in'): "
        f"{[str(idx[p].date()) + f' dd60 {st['dd60'][p]:+.3f} h {int(h[p])}' for p in near_hi]}")
    say(row("excluding dd60 > -5%", sk_))
    say("  leave-one-signal-out (main / 2026):")
    lo_ = []
    for i, p in enumerate(pos):
        rest = np.delete(pos, i)
        sr = stats(rest, st, idx, base)
        lo_.append(f"{idx[p].date()}: {sr['2024~2026']['hold20']:.3f}/{sr['2026']['hold20']:.3f}({sr['2026']['hold20_n']})"
                   f"{'' if L1(sr) else ' L1 lost'}")
    say("    " + "; ".join(lo_))
    say(f"  2026 in-year forward check (AGENTS research rule: choose on the earlier part, confirm on the later part):")
    say(f"    2026H1 {fmt(s0, '2026H1')} (matched {s0['2026H1']['hold20_base']:.3f}) -> a rule chosen on H1 would not have been this one;"
        f" 2026H2 {fmt(s0, '2026H2')} (matched {s0['2026H2']['hold20_base']:.3f}). 2026 raw in_dd rule days (no cooldown):")
    r26 = np.flatnonzero(sig & st["in_dd"] & (y == 2026))
    say("    " + ", ".join(f"{idx[q].date()}:{'-' if np.isnan(h[q]) else int(h[q])}" for q in r26))
    say(f"    raw 2026 days hold20 {np.nanmean(h[r26]):.3f} (resolved {int((~np.isnan(h[r26])).sum())}); matched "
        f"{np.nanmean(base['hold20'][r26][~np.isnan(h[r26])]):.3f}")

    # ------------------------------------------------------------ [5]
    say("\n[5] Mechanical comparison: plain up >= X (same in_dd, window, cooldown, outcomes)")
    upmed = m0["up_med"]
    mech = {}
    for lab, u in ((f"plain up >= median {upmed:.4f}", upmed), ("plain up >= 0.08", 0.08),
                   ("plain up >= 0.05 (rule w/o SMA20)", 0.05)):
        sg = np.where(np.isnan(st["up"]), False, st["up"] >= u)
        p_, s_ = evaluate(sg, st, idx, base)
        mech[lab] = (p_, s_)
        say(row(lab, s_))
        say(f"      failed: {fails(p_, st, idx)}")
    pm, sm = mech[f"plain up >= median {upmed:.4f}"]
    say(f"  candidate main {m0['hold20']:.3f} (n={m0['hold20_n']}, up median {upmed:.3f}) vs plain up >= median "
        f"{sm['2024~2026']['hold20']:.3f} (n={sm['2024~2026']['hold20_n']}, up median {sm['2024~2026']['up_med']:.3f}); 2026 "
        f"{s0['2026']['hold20']:.3f} vs {sm['2026']['hold20']:.3f}; near5 {m0['near5']:.2f} vs {sm['2024~2026']['near5']:.2f}; "
        f"win20 {m0['win20']:.2f} vs {sm['2024~2026']['win20']:.2f}; r20 {m0['r20']:+.3f} vs {sm['2024~2026']['r20']:+.3f}")
    sel = np.flatnonzero(poolm & (st["up"] >= upmed))
    sel2 = np.flatnonzero(poolm & (st["up"] < upmed))
    say(f"  all resolved in_dd days with up >= {upmed:.4f} (no cooldown): hold20 {np.mean(h[sel]):.4f} ({len(sel)} days); "
        f"with up < it: {np.mean(h[sel2]):.4f} ({len(sel2)})")
    say("  episode pairing: each candidate signal vs the nearest plain-up>=median first signal")
    for p in pos:
        j = pm[np.argmin(np.abs(pm - p))]
        say(f"    {idx[p].date()} up {st['up'][p]:.3f} h {int(h[p])}  <->  {idx[j].date()} ({j - p:+4d}) up {st['up'][j]:.3f} "
            f"h {int(h[j]) if not np.isnan(h[j]) else 'NaN'}")
    say("  candidate signals by up bucket (held/n, mean matched, pool mean):")
    for bb in range(4):
        sl = [p for p in pos if ub[p] == bb]
        if sl:
            say(f"    {BN[bb]:6s}: {int(np.nansum(h[sl]))}/{len(sl)} matched {np.mean(base['hold20'][sl]):.3f} "
                f"pool {np.mean(h[poolm & (ub == bb)]):.3f} dates {[str(idx[p].date()) for p in sl]}")

    # ------------------------------------------------------------ [6]
    say("\n[6] Stocks (82 US names, in_dd 20%, eval 2025-09-01..2026-10-23): independent recompute of the same rule (context)")
    o_k, c_k = read_stocks()
    ixk = c_k.index
    hk = ixk.year.to_numpy() * 10 + (ixk.month.to_numpy() > 6)
    per_k = [("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
             ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"), ("2025", "2025-09-01", "2025-12-31")]
    allv, allb, alld = [], [], []
    sums = {}
    e0k = pd.Timestamp("2025-09-01")
    for col in c_k.columns:
        stk = build(c_k[col].to_numpy(float), o_k[col].to_numpy(float), 0.20)
        bkey = ubin(stk["up"])
        pp = firsts(rule(stk), stk, ixk, eval0=e0k)
        alld += [ixk[p] for p in pp]
        allv += list(stk["hold20"][pp])
        allb += [(hk[p], bkey[p]) for p in pp]
        pl = stk["in_dd"] & ~np.isnan(stk["up"]) & (ixk >= e0k) & ~np.isnan(stk["hold20"])
        for s_, b_, v_ in zip(hk[pl], bkey[pl], stk["hold20"][pl]):
            acc = sums.setdefault((s_, b_), [0.0, 0])
            acc[0] += v_
            acc[1] += 1
    kb = np.array([sums[k][0] / sums[k][1] if k in sums else np.nan for k in allb])
    allv, alld = np.asarray(allv, float), pd.DatetimeIndex(alld)
    fk = pd.read_csv(RES / "f7_STK_signals.csv", parse_dates=["date"])
    fk = fk[fk.rule == RULE]
    fks = pd.read_csv(RES / "f7_STK_summary.csv")
    fks = fks[fks.rule == RULE].set_index("period")
    say(f"  first signals mine {len(allv)} vs family {len(fk)}")
    for nm, a, b in per_k:
        m = (alld >= a) & (alld <= b) & ~np.isnan(allv) & ~np.isnan(kb)
        say(f"    {nm:12s} n={int(m.sum()):4d} hold20 {allv[m].mean():.3f} matched {kb[m].mean():.3f} diff "
            f"{allv[m].mean() - kb[m].mean():+.3f} | family {fks.loc[nm, 'hold20']:.3f}/{fks.loc[nm, 'hold20_base']:.3f} "
            f"(n={int(fks.loc[nm, 'hold20_n'])})")

    # ------------------------------------------------------------ [7]
    say("\n[7] Verdict inputs")
    say(f"  indep_match={indep}, main_hold20={m0['hold20']:.4f}, main_matched={m0['hold20_base']:.4f}, "
        f"dlo={m0['hold20_dlo']:+.4f}, L1={L1(s0)}, L2={L2(s0)}")
    say(f"  null p: N1={p1:.4f} N1g={pg1:.4f} N2(+-2pp)={p2:.4f} N2g={pg2:.4f} N2b(+-2pp, up>=5%)={p3_:.4f}")
    say(f"  stratified C>SMA20 on/off at equal up (>=5%): {wd:+.3f}")
    say(f"  core variants losing L1: {lost}")
    say(f"  plain up >= median main {sm['2024~2026']['hold20']:.3f} (n={sm['2024~2026']['hold20_n']}), 2026 {sm['2026']['hold20']:.3f}")
    say(f"  chance of L1 from rebound size alone (Bernoulli(matched) per signal): {pass_l1.mean():.3f}")
    say("\nVERDICT: refuted")
    say(f"  Reproduced exactly without bc_lib (13 first signals, every field, per-period table, bootstrap bounds) and no look-ahead")
    say(f"  (own and family truncation tests 0 mismatches; the rule uses only SOXX closes up to t). But the {obs:.1%} is what the")
    say(f"  rebound size already gives: matched same-state {m0['hold20_base']:.1%} (dlo {m0['hold20_dlo']:+.3f}); same-state random-date")
    say(f"  nulls p = {p1:.2f}/{pg1:.2f} (year x bucket), {p2:.2f} (+-2pp), {p3_:.2f} (+-2pp and up >= 5%, isolates the SMA20 leg);")
    say(f"  at equal up the C > SMA20 leg changes hold20 by {wd:+.3f}; L1 passes by chance {pass_l1.mean():.0%} of the time from")
    say(f"  the signals' rebound sizes alone. Plain 'up >= {upmed:.2%}' (candidate median) holds {sm['2024~2026']['hold20']:.0%} "
        f"({sm['2024~2026']['hold20_n']}) incl. 2026 {sm['2026']['hold20']:.0%};")
    say("  both candidate failures are 3-8% rebounds (5.4%, 6.9%). The registered acceptance period 2026 is 3/4 (null p ~0.47),")
    say("  2026H1 0/1 (an H1-chosen rule would not be this one), and the 2026-06-08 'success' held by 0.32% (breaks on ^SOX,")
    say("  hold40 0, near5 0, r20 -4%); removing any 2026 success, cooldown 10 (adds the 2026-06-30 failure), in_dd 8%,")
    say("  SMA10 or SMA50, or moving to ^SOX/SMH loses L1. Stocks: 64% vs matched 64%. Not a bottom confirmation beyond 'price is")
    say("  already well off the low'.")
    json.dump(dict(indep=indep, p1=p1, pg1=pg1, p2=p2, pg2=pg2, p2b=p3_, wd=wd, lost=lost,
                   plain=sm["2024~2026"]["hold20"], plain_n=sm["2024~2026"]["hold20_n"], obs=obs, upmed=upmed),
              sys.stderr)
    OUT.write_text(_buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
