"""Independent adversarial verification (round 2, fixed right-censoring) of f7 / c_up5_since5 / SOXX (BRIEF.md 2026-10-26).

Candidate: up = C[t]/Ldd[t] - 1 >= 5% AND sessions since the decline low (LAST position of the min close since the latest
60-session closing high) >= 5. SOXX, in_dd 10%, first signals 2024-01-02..2026-10-23, 20-session cooldown.

Section [1] does NOT import bc_lib or any family module: it re-implements from the raw CSVs the SOXX series, the 60-session
peak (latest on ties), Ldd and its last position, dd60, in_dd, the rule, the cooldown, the outcomes with the FIXED
censoring rule (0 when a close below the level was already seen inside an incomplete forward window, NaN only when
nothing has broken yet), the year x up-bucket matched baseline and the 20-session block bootstrap, then compares with
results/f7_SOXX_signals.csv and f7_SOXX_summary.csv.

Section [2] (look-ahead audit) additionally runs the FAMILY code (bc_lib + f7_combo.decline_low_pos) in a separate
subprocess, only to dump its full raw rule series and a truncation test of the family state (prices cut at t). Nothing
from that subprocess feeds the independent numbers in [1], [3]-[5].

Sections: [1] independent match, [2] look-ahead / data audit, [3] robustness (neighbours, delay 1/2, cooldown 10/30,
in_dd 8%/12%, other instruments), [4] same-state random-date nulls, [5] mechanical comparison with plain up >= median up,
[6] stocks replication (independent), [7] verdict inputs.

Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify2_f7_SOXX_c_up5_since5.py
Out: results/verify2_f7_SOXX_c_up5_since5.txt     (everything here is post hoc)
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
OUT = RES / "verify2_f7_SOXX_c_up5_since5.txt"

EVAL0 = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PER = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
       ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
       ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = (0.03, 0.08, 0.15)
BN = ("0-3%", "3-8%", "8-15%", "15%+")
OUTS = ("hold20", "hold40", "near5", "win20", "r20", "r60")
UPMIN, SINCEMIN, COOL, THR = 0.05, 5, 20, 0.10
NDRAW, NBOOT = 5000, 5000
SEED = 7_2026_1026
N_CELLS = 109

_buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _buf.write(s + "\n")


# ====================================================================== raw data
def sessions(a, b) -> pd.DatetimeIndex:
    import exchange_calendars as xc
    s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(a, b))
    return s.tz_localize(None) if s.tz is not None else s


def read_soxx():
    raw = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[raw.index.notna()].sort_index()
    us = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                     encoding="utf-8-sig", index_col=0, parse_dates=True).sort_index()
    cn = raw[("close", "SOXX")].dropna()
    old = us["SOXX"].dropna()
    close = pd.concat([old[old.index < cn.index[0]], cn]).sort_index()
    close = close[close.index <= LAST]
    opn = raw[("open", "SOXX")].dropna().reindex(close.index)
    ov = old.index.intersection(cn.index)
    cal = sessions(close.index[0], close.index[-1])
    info = dict(n=len(close), ov=len(ov), ovdiff=float((old[ov] - cn[ov]).abs().max()),
                miss=len(cal.difference(close.index)), extra=len(close.index.difference(cal)))
    return pd.DataFrame({"C": close, "O": opn}), info, us


def read_stocks():
    o = pd.read_csv(ROOT / "data/derived/desk/prices_open.csv", index_col=0, parse_dates=True)
    c = pd.read_csv(ROOT / "data/derived/desk/prices_close.csv", index_col=0, parse_dates=True)
    drop = [x for x in c.columns if x.endswith(".KS") or x.endswith("=X") or x == "QQQ"]
    cal = sessions("2025-06-02", "2026-10-23")
    return o.drop(columns=drop).reindex(cal), c.drop(columns=drop).reindex(cal)


# ====================================================================== state / outcomes (one series)
def build(C: np.ndarray, O: np.ndarray, thr: float = THR) -> dict:
    """Everything known at the close of t; outcomes look forward from t+1."""
    N = len(C)
    Ldd = np.full(N, np.nan)
    Lpos = np.full(N, -1)
    pk_ = np.full(N, -1)
    dd = np.full(N, np.nan)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        if np.isnan(w).any():
            continue
        hi = np.max(w)
        pk = t - 59 + max(i for i in range(60) if w[i] == hi)            # latest peak on ties
        seg = C[pk:t + 1]
        lo = np.min(seg)
        Ldd[t] = lo
        Lpos[t] = pk + max(i for i in range(len(seg)) if seg[i] == lo)    # latest low on ties
        pk_[t] = pk
        dd[t] = C[t] / hi - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = bool(len(w)) and float(w.min()) <= -thr
    up = C / Ldd - 1
    since = np.where(Lpos >= 0, np.arange(N) - Lpos, -1)
    E = np.full(N, np.nan)
    E[:-1] = O[1:]

    def hold(level, L):
        out = np.full(N, np.nan)
        first_break = np.full(N, -1)
        for t in range(N - 1):
            if np.isnan(level[t]):
                continue
            seg = C[t + 1:min(t + L, N - 1) + 1]
            okv = ~np.isnan(seg)
            br = np.flatnonzero(okv & (seg < level[t] - 1e-12))
            if len(br):
                out[t] = 0.0
                first_break[t] = t + 1 + int(br[0])
            elif t + L <= N - 1 and okv.all():
                out[t] = 1.0
        return out, first_break

    h20, b20 = hold(Ldd, 20)
    h40, _ = hold(Ldd, 40)
    n5, _ = hold(0.95 * E, 40)
    r20 = np.full(N, np.nan)
    r60 = np.full(N, np.nan)
    for t in range(N - 20):
        r20[t] = C[t + 20] / E[t] - 1
    for t in range(N - 60):
        r60[t] = C[t + 60] / E[t] - 1
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    margin = np.full(N, np.nan)                                            # min forward close vs Ldd (complete windows)
    for t in range(N - 20):
        if not np.isnan(Ldd[t]):
            margin[t] = np.nanmin(C[t + 1:t + 21]) / Ldd[t] - 1
    return dict(C=C, O=O, Ldd=Ldd, Lpos=Lpos, pk=pk_, dd60=dd, in_dd=in_dd, up=up, since=since, hold20=h20, brk20=b20,
                hold40=h40, near5=n5, win20=win20, r20=r20, r60=r60, margin=margin)


def ubin(up):
    b = np.digitize(up, EDGES).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def matched(idx, st, key, eval0=EVAL0, slice_key=None):
    sk = idx.year.to_numpy() if slice_key is None else slice_key
    b = ubin(st["up"])
    pool = st["in_dd"] & ~np.isnan(st["up"]) & (idx >= eval0)
    v = st[key]
    out = np.full(len(idx), np.nan)
    tab = {}
    for s in np.unique(sk):
        for bb in range(4):
            m = pool & (sk == s) & (b == bb) & ~np.isnan(v)
            if m.any():
                tab[(s, bb)] = float(v[m].mean())
    for t in range(len(idx)):
        if not np.isnan(b[t]):
            out[t] = tab.get((sk[t], int(b[t])), np.nan)
    return out


def rule(st, upmin=UPMIN, smin=SINCEMIN):
    u = np.where(np.isnan(st["up"]), -1.0, st["up"])
    return (u >= upmin) & (st["since"] >= smin)


def firsts(sig, st, idx, cool=COOL, eval0=EVAL0):
    ok = sig & st["in_dd"] & ~np.isnan(st["Ldd"]) & (idx >= eval0) & (idx <= LAST)
    keep, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev > cool:
            keep.append(p)
            prev = p
    return np.asarray(keep, int)


def lag(sig, d):
    return sig if d == 0 else np.r_[np.zeros(d, bool), sig[:-d]]


def boot(v, lab, rng, B=NBOOT):
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    s, c = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = s[pick].sum(1) / c[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def stats(pos, st, idx, base):
    rng = np.random.default_rng(SEED)
    d = idx[pos] if len(pos) else pd.DatetimeIndex([])
    res = {}
    for name, a, b in PER:
        pp = pos[(d >= a) & (d <= b)]
        r = dict(n=int(len(pp)), up_med=float(np.median(st["up"][pp])) if len(pp) else np.nan,
                 dd_med=float(np.median(st["dd60"][pp])) if len(pp) else np.nan)
        for k in OUTS:
            v, bs = st[k][pp], base[k][pp]
            ok = ~np.isnan(v) & ~np.isnan(bs)
            r[f"{k}_n"] = int(ok.sum())
            r[k] = float(v[ok].mean()) if ok.any() else np.nan
            r[f"{k}_base"] = float(bs[ok].mean()) if ok.any() else np.nan
            if k == "hold20":
                lab = pp[ok] // COOL
                r["hold20_lo"], r["hold20_hi"] = boot(v[ok], lab, rng)
                r["hold20_dlo"], r["hold20_dhi"] = boot(v[ok] - bs[ok], lab, rng)
        res[name] = r
    return res


def L1(s):
    m, a, b = s["2024~2026"], s["2026"], s["2025"]
    return bool(m["hold20_n"] >= 8 and m["hold20"] >= 0.70 and a["hold20_n"] >= 3 and a["hold20"] >= 0.70
                and (b["hold20_n"] < 3 or b["hold20"] >= 0.60))


def L2(s):
    return bool(L1(s) and s["2024~2026"]["hold20_dlo"] > 0)


def row(label, s):
    def f(p):
        r = s[p]
        return f"{r['hold20']:.3f}({r['hold20_n']:2d})" if r["hold20_n"] else f"  -  ({r['n']:2d})"
    m = s["2024~2026"]
    return (f"  {label:30s} main {f('2024~2026')} matched {m['hold20_base']:.3f} dlo {m['hold20_dlo']:+.3f} | 2026 "
            f"{f('2026')} H1 {f('2026H1')} H2 {f('2026H2')} 2025 {f('2025')} 2024 {f('2024')} | upmed {m['up_med']:.3f} "
            f"near5 {m['near5']:.2f} win20 {m['win20']:.2f} r20 {m['r20']:+.3f} L1={L1(s)} L2={L2(s)}")


def fails(pos, st, idx):
    return ", ".join(f"{idx[p].date()}(up {st['up'][p]:.3f}, since {st['since'][p]}, broke "
                     f"{idx[st['brk20'][p]].date() if st['brk20'][p] >= 0 else '?'})"
                     for p in pos if st["hold20"][p] == 0) or "none"


def evaluate(sig, st, idx, base, cool=COOL):
    pos = firsts(sig, st, idx, cool)
    return pos, stats(pos, st, idx, base)


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


def gibbs_null(cands, y, cool, rng, start, nkeep=NDRAW, burn=1000, thin=4):
    """Uniform over spaced configurations (one day per slot, all pairs > cool apart)."""
    ch = np.asarray(start, int).copy()
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


def one_spaced(cands, cool, rng):
    for _ in range(100000):
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


# ====================================================================== family subprocess (audit only)
FAMILY_PROBE = r"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
import bc_lib, f7_combo
fr = bc_lib.soxx_frames()
C = bc_lib.load_soxx()[["C"]].rename(columns={"C": "SOXX"})
O = bc_lib.load_soxx()[["O"]].rename(columns={"O": "SOXX"})
PK, LL = f7_combo.decline_low_pos(C)
N = len(C)
since = np.where(LL >= 0, np.arange(N)[:, None] - LL, -1)[:, 0]
sig = ((fr["up"]["SOXX"].to_numpy() >= 0.05) & (since >= 5))
test = [int(x) for x in sys.argv[2].split(",")]
trunc = {}
for t in test:
    c2, o2 = C.iloc[:t + 1], O.iloc[:t + 1]
    f2 = bc_lib.outcome_frame(o2, c2, 0.10)
    pk2, ll2 = f7_combo.decline_low_pos(c2)
    s2 = (t - ll2[t, 0]) if ll2[t, 0] >= 0 else -1
    u2 = f2["up"]["SOXX"].iat[t]
    trunc[t] = [bool((u2 >= 0.05) and (s2 >= 5)), bool(f2["in_dd"]["SOXX"].iat[t]),
                None if pd.isna(f2["L60"]["SOXX"].iat[t]) else float(f2["L60"]["SOXX"].iat[t]), int(s2)]
print(json.dumps(dict(dates=[str(d.date()) for d in C.index], sig=sig.astype(int).tolist(),
                      in_dd=fr["in_dd"]["SOXX"].astype(int).tolist(), since=since.tolist(),
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
    say("Independent verification round 2 (fixed right-censoring): family f7, rule c_up5_since5, kind SOXX (bc_lib NOT imported in [1])")
    say("Rule: up = C[t]/Ldd[t] - 1 >= 5% AND sessions since the decline low (last position of the min close since the latest")
    say("      60-session closing high) >= 5; first signals on in_dd(10%) days 2024-01-02..2026-10-23, 20-session cooldown.")
    say("=" * 130)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()} N={N}; open from {df.O.first_valid_index().date()}; overlap "
        f"rows {info['ov']} max|diff|={info['ovdiff']:.1e}; XNYS sessions missing={info['miss']} extra={info['extra']}")

    st = build(C, O)
    base = {k: matched(idx, st, k) for k in OUTS}
    sig = rule(st)
    pos, s0 = evaluate(sig, st, idx, base)

    # ------------------------------------------------------------ [1]
    say("\n[1] Independent re-implementation vs family files (results/f7_SOXX_signals.csv, f7_SOXX_summary.csv)")
    fam = pd.read_csv(RES / "f7_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "c_up5_since5"].reset_index(drop=True)
    fsum = pd.read_csv(RES / "f7_SOXX_summary.csv")
    fsum = fsum[fsum.rule == "c_up5_since5"].set_index("period")
    mine = pd.DataFrame(dict(date=idx[pos], pos=pos, low=[idx[st["Lpos"][p]].date() for p in pos], Ldd=st["Ldd"][pos],
                             C=C[pos], since=st["since"][pos], up=st["up"][pos], dd60=st["dd60"][pos],
                             **{k: st[k][pos] for k in OUTS}, **{f"base_{k}": base[k][pos] for k in OUTS},
                             margin20=st["margin"][pos]))
    show = mine.copy()
    show["date"] = [d.date() for d in show.date]
    show["broke"] = [str(idx[st["brk20"][p]].date()) if st["brk20"][p] >= 0 else "-" for p in pos]
    say(show[["date", "pos", "low", "Ldd", "C", "since", "up", "dd60", "hold20", "base_hold20", "broke", "margin20",
              "hold40", "near5", "win20", "r20"]].round(4).to_string(index=False))
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
    say("  period      mine hold20(n)  fam hold20(n)   mine base  fam base | mine dlo  fam dlo | other outcomes")
    per_ok = True
    for p, _, _ in PER:
        r = s0[p]
        okp = r["hold20_n"] == int(fsum.loc[p, "hold20_n"]) and r["n"] == int(fsum.loc[p, "n"])
        for k in OUTS:
            for suf in ("", "_base"):
                a, b = r[k + suf], fsum.loc[p, k + suf] if (k + suf) in fsum.columns else np.nan
                okp &= bool((np.isnan(a) and pd.isna(b)) or abs(a - b) < 1e-9)
        per_ok &= bool(okp)
        say(f"  {p:10s}  {r['hold20']:.4f} ({r['hold20_n']:2d})  {fsum.loc[p, 'hold20']:.4f} ({int(fsum.loc[p, 'hold20_n']):2d})"
            f"   {r['hold20_base']:.4f}   {fsum.loc[p, 'hold20_base']:.4f} | {r['hold20_dlo']:+.3f}   "
            f"{fsum.loc[p, 'hold20_dlo']:+.3f} | hold40/near5/win20/r20/r60 all {'OK' if okp else 'MISMATCH'}")
    indep = bool(same_dates and not mism and per_ok)
    say(f"  => independent match: {indep}  (bootstrap bounds: different RNG stream, compared for sign only)")
    m0 = s0["2024~2026"]
    poolm = st["in_dd"] & ~np.isnan(st["hold20"]) & ~np.isnan(st["up"]) & (idx >= EVAL0) & (idx <= LAST)
    unc = float(st["hold20"][poolm].mean())
    say(f"  main n={m0['n']} hold20={m0['hold20']:.4f} vs matched {m0['hold20_base']:.4f} (diff "
        f"{m0['hold20'] - m0['hold20_base']:+.4f}, block-boot dlo {m0['hold20_dlo']:+.4f}); near5 {m0['near5']:.3f} (matched "
        f"{m0['near5_base']:.3f}); win20 {m0['win20']:.3f} ({m0['win20_base']:.3f}); r20 {m0['r20']:+.4f} ({m0['r20_base']:+.4f});"
        f" up median {m0['up_med']:.4f}")
    say(f"  unconditional in_dd hold20 2024~2026 = {unc:.4f} (resolved days {int(poolm.sum())})")
    kk, nn = int(round(m0["hold20"] * m0["hold20_n"])), m0["hold20_n"]
    for lab, p0 in (("unconditional in_dd", unc), ("matched mean", m0["hold20_base"])):
        pb = sum(comb(nn, j) * p0 ** j * (1 - p0) ** (nn - j) for j in range(kk, nn + 1))
        say(f"  binomial P(X >= {kk}/{nn} | p={p0:.4f}, {lab}) = {pb:.4f}")
    a26 = s0["2026"]
    say(f"  2026 alone: {int(round(a26['hold20'] * a26['hold20_n']))}/{a26['hold20_n']} = {a26['hold20']:.3f}; one more "
        f"failure would give {(a26['hold20'] * a26['hold20_n'] - 1) / a26['hold20_n']:.3f} (< 0.70 -> L1 lost)")
    say("  successes held by a thin margin (min close of t+1..t+20 within 2% of Ldd): " +
        (", ".join(f"{idx[p].date()} {st['margin'][p] * 100:+.2f}%" for p in pos
                   if st["hold20"][p] == 1 and st["margin"][p] < 0.02) or "none"))
    later = np.flatnonzero(sig & st["in_dd"] & (np.arange(N) > pos[-1]))
    say(f"  raw rule days in in_dd after the last first signal {idx[pos[-1]].date()}: "
        f"{len(later)} ({idx[later[0]].date()}..{idx[later[-1]].date()})" if len(later) else "  none after last signal")
    say(f"  state on the last day {idx[-1].date()}: in_dd={st['in_dd'][-1]} up={st['up'][-1]:+.4f} since={st['since'][-1]} "
        f"Ldd={st['Ldd'][-1]:.2f} ({idx[st['Lpos'][-1]].date()}) rule={bool(sig[-1])}")
    ub = ubin(st["up"])
    for p in pos:
        for e in EDGES:
            if abs(st["up"][p] - e) < 2e-3:
                say(f"  bucket-edge: {idx[p].date()} up={st['up'][p]:.6f} (edge {e}) bucket {BN[int(ub[p])]} matched "
                    f"{base['hold20'][p]:.4f}")

    # ------------------------------------------------------------ [2]
    say("\n[2] Look-ahead / timing / data audit")
    test_pos = np.unique(np.r_[np.flatnonzero(sig & (idx >= EVAL0)), np.arange(np.searchsorted(idx, EVAL0), N, 7)])
    bad = 0
    for p in test_pos:
        s2 = build(C[:p + 1], O[:p + 1])
        if not (s2["in_dd"][p] == st["in_dd"][p] and s2["Lpos"][p] == st["Lpos"][p] and
                abs(s2["up"][p] - st["up"][p]) < 1e-12 and rule(s2)[p] == sig[p]):
            bad += 1
    say(f"  own implementation, truncation (prices cut at t) on {len(test_pos)} days: mismatches {bad}")
    sample = [int(x) for x in np.r_[pos, np.flatnonzero(sig & (idx >= EVAL0))[::9], np.arange(300, N, 37)]]
    sample = sorted(set(sample))
    try:
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", FAMILY_PROBE, str(HERE / "code"),
                            ",".join(map(str, sample))], capture_output=True, text=True, timeout=1200, cwd=str(HERE))
        fp = json.loads(r.stdout.strip().splitlines()[-1])
        fsig = np.asarray(fp["sig"], bool)
        fin = np.asarray(fp["in_dd"], bool)
        same_idx = [str(d.date()) for d in idx] == fp["dates"]
        say(f"  family code (subprocess: bc_lib.soxx_frames + f7_combo.decline_low_pos): same session index {same_idx}; "
            f"raw rule days differing from mine {int((fsig != sig).sum())} of {N}; in_dd days differing "
            f"{int((fin != st['in_dd']).sum())}; since differing {int((np.asarray(fp['since']) != st['since']).sum())}")
        tb = 0
        for t, (rs, ind, l60, sn) in fp["trunc"].items():
            t = int(t)
            if rs != bool(fsig[t]) or ind != bool(fin[t]) or sn != int(fp["since"][t]) or \
                    (l60 is None) != (fp["L60"][t] is None) or (l60 is not None and abs(l60 - fp["L60"][t]) > 1e-9):
                tb += 1
        say(f"  family code truncation test (prices cut at t, family functions re-run) on {len(fp['trunc'])} days incl. "
            f"all {len(pos)} first signals: mismatches {tb}")
    except Exception as ex:                                              # noqa: BLE001
        say(f"  family subprocess failed: {ex!r}")
    say("  code read (f7_combo.py): decline_low_pos uses w = c[t-59:t+1] (ends at t), pk = t - argmax(w[::-1]) (latest peak),")
    say("  LL = t - argmin(c[pk:t+1][::-1]) (latest low); since = pos - LL; c_up5_since5 = (fr['up'] >= 0.05) & (since >= 5).")
    say("  No market, VIX, breadth or news input in this rule; KNOWN_LOWS (hindsight lows) only feed the printed spot check.")
    say("  bc_lib.first_signals: in_dd, Ldd at t; window 2024-01-02..2026-10-23; cooldown drops p - prev <= 20.")
    say("  bc_lib.outcome_frame: fwd window C[t+1..t+20]/[t+1..t+40]; entry O[t+1]; censoring 0 if already broken, NaN if not.")
    lr = np.diff(np.log(C))
    big = [(idx[i + 1].date(), lr[i]) for i in np.flatnonzero(np.abs(lr) > 0.07)]
    say("  SOXX |daily log return| > 7%: " + ", ".join(f"{d} {v:+.3f}" for d, v in big))
    gp = O[1:] / C[:-1] - 1
    say("  SOXX |open/prev close| > 6%: " + ", ".join(f"{idx[i + 1].date()} {gp[i]:+.3f}" for i in np.flatnonzero(np.abs(gp) > 0.06)))
    say(f"  O[t+1] present for every first signal: {bool(np.isfinite(O[pos + 1]).all())}")
    sox = us["^SOX"].reindex(idx)
    smh = us["SMH"].reindex(idx)
    rs, rx = df.C.pct_change(), sox.pct_change()
    dif = (rs - rx).abs()
    say(f"  days where SOXX and ^SOX daily returns differ by > 1.5pp (^SOX to {sox.last_valid_index().date()}): "
        + (", ".join(f"{d.date()} {v * 100:.1f}pp" for d, v in dif[dif > 0.015].items()) or "none"))
    say("  decline-low days of the first signals (return into / out of the low; SOXX vs ^SOX):")
    for p in pos:
        lp = st["Lpos"][p]
        a = C[lp] / C[lp - 1] - 1
        b = C[lp + 1] / C[lp] - 1
        xa = sox.iloc[lp] / sox.iloc[lp - 1] - 1
        xb = sox.iloc[lp + 1] / sox.iloc[lp] - 1
        say(f"    {idx[p].date()}: low {idx[lp].date()} in {a:+.3f} out {b:+.3f} (^SOX {xa:+.3f}/{xb:+.3f})")
    say("  dividend back-adjustment: closes are adjusted (later dividends scale earlier prices down by ~0.1-0.2%);")
    say("  the thinnest hold margin above shows whether that could flip any hold20 outcome.")

    # ------------------------------------------------------------ [3]
    say("\n[3] Robustness (hold20, resolved n; matched = year x up-bucket; dlo = block-bootstrap lower bound of hold20 - matched)")
    say(row("registered c_up5_since5", s0))
    say(f"      failed: {fails(pos, st, idx)}")
    core = {}
    say("  -- timing (raw signal shifted d sessions; state, Ldd and outcomes read at the delayed day, as the family lag1)")
    for d in (1, 2):
        p_, s_ = evaluate(lag(sig, d), st, idx, base)
        core[f"delay +{d}"] = s_
        say(row(f"delay +{d}", s_))
        say(f"      failed: {fails(p_, st, idx)}")
        for p in p_:
            if st["hold20"][p] == 0 and st["since"][p] == 0:
                say(f"      note {idx[p].date()}: Ldd reset on the delayed day (60-session peak now {idx[st['pk'][p]].date()}, "
                    f"C {C[p]:.2f}, dd60 {st['dd60'][p]:+.3f}) -> the outcome tests 'this new low holds', not the original low")
    say("  -- timing variant: entry delayed d sessions but the level stays the signal-day Ldd (hold over t+d+1..t+d+20)")
    for d in (1, 2):
        v = np.full(len(pos), np.nan)
        for i, p in enumerate(pos):
            q = p + d
            if q + 20 <= N - 1:
                v[i] = float(np.min(C[q + 1:q + 21]) >= st["Ldd"][p] - 1e-12)
            elif np.min(C[q + 1:]) < st["Ldd"][p] - 1e-12:
                v[i] = 0.0
        y = idx[pos].year
        say(f"    delay +{d} keep-Ldd: main {np.nanmean(v):.3f} ({int((~np.isnan(v)).sum())}) | 2026 "
            f"{np.nanmean(v[y == 2026]):.3f} | 2025 {np.nanmean(v[y == 2025]):.3f} | 2024 {np.nanmean(v[y == 2024]):.3f}"
            f"  failed: {', '.join(str(idx[p].date()) for p, x in zip(pos, v) if x == 0)}")
    say("    (2024-10-16 close equals 2024-10-15 close exactly, 227.360001, while ^SOX +0.2%: possible stale print; it only"
        " matters for the delay +2 Ldd-reset row)")
    say("  -- cooldown / in_dd threshold")
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
    say("  -- single-step neighbours (as family post hoc)")
    for lab, (u, n) in {"up4_since5": (0.04, 5), "up6_since5": (0.06, 5), "up5_since4": (0.05, 4),
                        "up5_since6": (0.05, 6)}.items():
        p_, s_ = evaluate(rule(st, u, n), st, idx, base)
        core[lab] = s_
        say(row(lab, s_))
        say(f"      failed: {fails(p_, st, idx)}")
    say("  -- neighbour grid (post hoc): rows since >= 3..10, cols up >= 3..8% -> main hold20(n) / 2026 hold20(n) / matched [*=L1, **=L2]")
    grid_l1 = grid_l2 = grid_gt = 0
    gdiff = []
    ups = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    say("           " + "".join(f"up>={u:.0%}".ljust(28) for u in ups))
    for n in range(3, 11):
        cells = []
        for u in ups:
            _, s_ = evaluate(rule(st, u, n), st, idx, base)
            m = s_["2024~2026"]
            l1, l2 = L1(s_), L2(s_)
            grid_l1 += l1
            grid_l2 += l2
            grid_gt += m["hold20"] > m["hold20_base"]
            gdiff.append(m["hold20"] - m["hold20_base"])
            cells.append(f"{m['hold20']:.2f}({m['hold20_n']:2d})/{s_['2026']['hold20']:.2f}({s_['2026']['hold20_n']})/"
                         f"{m['hold20_base']:.2f}{'**' if l2 else '*' if l1 else ''}".ljust(28))
        say(f"    since{n:<3d}" + "".join(cells))
    say(f"    L1 in {grid_l1}/48; L2 in {grid_l2}/48; cells with main hold20 > matched: {grid_gt}/48; mean(main - matched) "
        f"{np.mean(gdiff):+.3f}")
    say("  -- other instruments, same rule and engine (closes only; ^SOX/SMH to 2026-10-13, outcomes censored there)")
    for nm, ser in (("^SOX index", sox), ("SMH ETF", smh)):
        c2 = ser.dropna()
        c2 = c2[c2.index <= LAST]
        ix2 = c2.index
        st3 = build(c2.to_numpy(float), c2.to_numpy(float))
        b3 = {k: matched(ix2, st3, k) for k in OUTS}
        p3, s3 = evaluate(rule(st3), st3, ix2, b3)
        say(row(nm, s3) + "  (near5/win20/r20 use next close as entry proxy)")
        say(f"      first signals: {', '.join(f'{ix2[p].date()}:{int(st3['hold20'][p]) if not np.isnan(st3['hold20'][p]) else None}' for p in p3)}")
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
        for yy in (2024, 2025, 2026) for bb in range(4)))
    cands = [np.flatnonzero(poolm & (y == yy) & (ub == bb)) for yy, bb in keys]
    n1, t1 = seq_null(cands, h, COOL, rng)
    p1 = float((n1 >= obs - 1e-12).mean())
    say(f"  [N1] sequential spaced sampler: mean {n1.mean():.4f} sd {n1.std():.4f} [2.5%, 97.5%] "
        f"[{np.quantile(n1, .025):.3f}, {np.quantile(n1, .975):.3f}]  p = {p1:.4f}  (draws {len(n1)}, attempts {t1})")
    g1 = gibbs_null(cands, h, COOL, rng, one_spaced(cands, COOL, rng))
    pg1 = float((g1 >= obs - 1e-12).mean())
    say(f"  [N1g] Gibbs (uniform over spaced sets): mean {g1.mean():.4f} p = {pg1:.4f} (kept {len(g1)})")
    ni = np.array([h[[int(rng.choice(c)) for c in cands]].mean() for _ in range(NDRAW)])
    say(f"  [N1i] independent draws (no spacing): mean {ni.mean():.4f} p = {float((ni >= obs - 1e-12).mean()):.4f}")
    fine = [np.flatnonzero(poolm & (y == y[p]) & (np.abs(st["up"] - st["up"][p]) <= 0.02)) for p in pos]
    say("  fine pools (same year, up within +-2pp): " + ", ".join(
        f"{idx[p].date()}:{len(c)}(h {np.mean(h[c]):.2f})" for p, c in zip(pos, fine)))
    n2, t2 = seq_null(fine, h, COOL, rng)
    p2 = float((n2 >= obs - 1e-12).mean())
    g2 = gibbs_null(fine, h, COOL, rng, one_spaced(fine, COOL, rng))
    pg2 = float((g2 >= obs - 1e-12).mean())
    say(f"  [N2] same year, up +-2pp: sequential mean {n2.mean():.4f} sd {n2.std():.4f} p = {p2:.4f} (attempts {t2}); "
        f"Gibbs mean {g2.mean():.4f} p = {pg2:.4f}")
    fine5 = [np.flatnonzero(poolm & (y == y[p]) & (np.abs(st["up"] - st["up"][p]) <= 0.02) & (st["up"] >= UPMIN))
             for p in pos]
    n3, _ = seq_null(fine5, h, COOL, rng)
    p3_ = float((n3 >= obs - 1e-12).mean())
    say(f"  [N2b] same year, up +-2pp AND up >= 5% (the rule's own floor; isolates 'since >= 5'): mean {n3.mean():.4f} "
        f"p = {p3_:.4f}")
    i26 = [i for i, p in enumerate(pos) if y[p] == 2026]
    o26 = float(np.mean(h[pos[i26]]))
    n4, _ = seq_null([cands[i] for i in i26], h, COOL, rng)
    n5, _ = seq_null([fine[i] for i in i26], h, COOL, rng)
    say(f"  [N3] 2026 only (n={len(i26)}, obs {o26:.3f}): year x bucket mean {n4.mean():.3f} p = {float((n4 >= o26 - 1e-12).mean()):.4f};"
        f" +-2pp mean {n5.mean():.3f} p = {float((n5 >= o26 - 1e-12).mean()):.4f}")
    say("  [N4] stratified on/off (all resolved in_dd days 2024~2026, no cooldown): among days with up >= 5%, within year x")
    say("       fine up bins (5-6.5/6.5-8/8-10/10-15/15%+), hold20 of since >= 5 vs since < 5")
    fb = np.digitize(st["up"], (0.065, 0.08, 0.10, 0.15))
    fbn = ("5-6.5%", "6.5-8%", "8-10%", "10-15%", "15%+")
    rows_ = []
    on = poolm & (st["up"] >= UPMIN)
    for yy in (2024, 2025, 2026):
        for bb in range(5):
            m = on & (y == yy) & (fb == bb)
            a, b = m & (st["since"] >= SINCEMIN), m & (st["since"] < SINCEMIN)
            rows_.append(dict(year=yy, bin=fbn[bb], n_on=int(a.sum()), h_on=np.mean(h[a]) if a.any() else np.nan,
                              n_off=int(b.sum()), h_off=np.mean(h[b]) if b.any() else np.nan))
    tb = pd.DataFrame(rows_)
    say(tb.round(3).to_string(index=False))
    both = tb[(tb.n_on > 0) & (tb.n_off > 0)]
    wd = float(np.average(both.h_on - both.h_off, weights=both.n_on)) if len(both) else np.nan
    say(f"       weighted (on - off) difference over strata with both: {wd:+.3f} (strata {len(both)}, on-days "
        f"{int(both.n_on.sum())}, off-days {int(both.n_off.sum())})")
    a_ = on & (st["since"] >= SINCEMIN)
    b_ = on & (st["since"] < SINCEMIN)
    say(f"       pooled: up>=5% & since>=5 hold20 {np.mean(h[a_]):.3f} (n={int(a_.sum())}) vs up>=5% & since<5 "
        f"{np.mean(h[b_]):.3f} (n={int(b_.sum())})")
    inf = [(str(idx[p].date()), BN[int(ub[p])], round(float(base['hold20'][p]), 3), h[p]) for p in pos if base["hold20"][p] < 0.95]
    exp_ = sum(x[2] for x in inf)
    say(f"  informative signals (matched base < 0.95): {len(inf)}; held {int(sum(x[3] for x in inf))}/{len(inf)}, "
        f"expected {exp_:.2f}; {inf}")
    say(f"  multiple testing: {N_CELLS} registered cells; Bonferroni 0.05/{N_CELLS} = {0.05 / N_CELLS:.5f}.")

    # ------------------------------------------------------------ [5]
    say("\n[5] Mechanical comparison: plain up >= X (same in_dd, window, cooldown, outcomes)")
    upmed = m0["up_med"]
    mech = {}
    for lab, u in ((f"plain up >= median {upmed:.4f}", upmed), ("plain up >= 0.08", 0.08), ("plain up >= 0.05 (rule w/o since)", 0.05)):
        sg = np.where(np.isnan(st["up"]), False, st["up"] >= u)
        p_, s_ = evaluate(sg, st, idx, base)
        mech[lab] = (p_, s_)
        say(row(lab, s_))
        say(f"      failed: {fails(p_, st, idx)}")
    pm, sm = mech[f"plain up >= median {upmed:.4f}"]
    say(f"  candidate main {m0['hold20']:.3f} (n={m0['hold20_n']}, up median {upmed:.3f}) vs plain up >= median "
        f"{sm['2024~2026']['hold20']:.3f} (n={sm['2024~2026']['hold20_n']}); 2026 {s0['2026']['hold20']:.3f} vs "
        f"{sm['2026']['hold20']:.3f}; near5 {m0['near5']:.2f} vs {sm['2024~2026']['near5']:.2f}; win20 {m0['win20']:.2f} vs "
        f"{sm['2024~2026']['win20']:.2f}; r20 {m0['r20']:+.3f} vs {sm['2024~2026']['r20']:+.3f}")
    say("  episode pairing: each candidate signal vs the nearest plain-up>=median first signal")
    for p in pos:
        j = pm[np.argmin(np.abs(pm - p))]
        say(f"    {idx[p].date()} up {st['up'][p]:.3f} h {int(h[p])}  <->  {idx[j].date()} ({j - p:+4d}) up {st['up'][j]:.3f} "
            f"h {int(h[j]) if not np.isnan(h[j]) else 'NaN'}")
    say("  candidate signals by up bucket (held/n, mean matched):")
    for bb in range(4):
        sel = [p for p in pos if ub[p] == bb]
        if sel:
            say(f"    {BN[bb]:6s}: {int(np.nansum(h[sel]))}/{len(sel)} matched {np.mean(base['hold20'][sel]):.3f} "
                f"dates {[str(idx[p].date()) for p in sel]}")

    # ------------------------------------------------------------ [6]
    say("\n[6] Stocks (82 US names, in_dd 20%, eval 2025-09-01..2026-10-23): independent recompute of the same rule")
    o_k, c_k = read_stocks()
    ixk = c_k.index
    hk = (ixk.year.to_numpy() * 10 + (ixk.month.to_numpy() > 6))
    per_k = [("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
             ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"), ("2025", "2025-09-01", "2025-12-31")]
    allv, allb, allp, alld = [], [], [], []
    pools = []
    for col in c_k.columns:
        stk = build(c_k[col].to_numpy(float), o_k[col].to_numpy(float), 0.20)
        bkey = ubin(stk["up"])
        pools.append((stk, bkey))
        sg = rule(stk)
        pp = firsts(sg, stk, ixk, eval0=pd.Timestamp("2025-09-01"))
        allp += list(pp)
        alld += [ixk[p] for p in pp]
        allv += list(stk["hold20"][pp])
        allb += [(hk[p], bkey[p]) for p in pp]
    sums = {}
    for stk, bkey in pools:
        pl = stk["in_dd"] & ~np.isnan(stk["up"]) & (ixk >= pd.Timestamp("2025-09-01")) & ~np.isnan(stk["hold20"])
        for s_, b_, v_ in zip(hk[pl], bkey[pl], stk["hold20"][pl]):
            a = sums.setdefault((s_, b_), [0.0, 0])
            a[0] += v_
            a[1] += 1
    kb = np.array([sums[k][0] / sums[k][1] if k in sums else np.nan for k in allb])
    allv, alld = np.asarray(allv), pd.DatetimeIndex(alld)
    fk = pd.read_csv(RES / "f7_STK_signals.csv", parse_dates=["date"])
    fk = fk[fk.rule == "c_up5_since5"]
    say(f"  first signals mine {len(allv)} vs family {len(fk)}")
    fks = pd.read_csv(RES / "f7_STK_summary.csv")
    fks = fks[fks.rule == "c_up5_since5"].set_index("period")
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
    say(f"  stratified since>=5 on/off at equal up (>=5%): {wd:+.3f}")
    say(f"  core variants losing L1: {lost}")
    say(f"  plain up >= median main {sm['2024~2026']['hold20']:.3f} (n={sm['2024~2026']['hold20_n']})")
    say("\n[8] Verdict: FRAGILE (L1 reproduced exactly, but it is rebound size, not a bottom-confirmation signal)")
    say("  - Reproduced: independent code gives the same 14 first signals and every outcome/base value; no look-ahead found")
    say("    (own and family truncation tests 0 mismatches; rule uses closes only; outcomes start at t+1).")
    say("  - Not information: main hold20 0.857 is BELOW its year x up-bucket matched value 0.886 (dlo -0.15, L2 fails);")
    say("    same-state random-date nulls give p 0.63-0.84; at equal rebound size 'since >= 5' adds +0.01 (N4).")
    say("  - Mechanical: all 10 signals with up >= 8% held; the 4 signals in 5-8% held 2/4 (matched 0.62). Plain up >= 9.9%")
    say("    (the candidate's own median) holds 12/12 with no 'since' condition; the candidate is a weaker version of it.")
    say("  - Fragile: 2026 passes on 3/4 (one more miss -> 0.50). L1 lost with since 4 or 6 (knife-edge at 5), delay +1/+2")
    say("    (engine lag), in_dd 8% and on ^SOX closes. The main-period rate stays 0.71-0.91 in every variant, i.e. the")
    say("    70%+ level is what the up mix delivers, not what the rule adds.")
    say("  - Practical cost: near5 0.50, win20 0.57, r20 +3.6% (all below matched win20 0.65 / r20 +5.2%); 82 stocks 0.56 in 2026.")
    OUT.write_text(_buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
