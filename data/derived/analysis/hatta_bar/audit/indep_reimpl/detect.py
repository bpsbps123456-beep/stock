"""Independent re-implementation of the hatta_bar v2 detector from BRIEF.md only.

Outputs mine_signals.csv (cell, sym, ts_et, pattern, k, S, touch, level_anchor, level_chain)
plus a per-candidate diagnostic table (diag.csv) recording each condition for every
first-break bar so mismatches can be traced. No outcome / performance computation.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "cache", "yf1m")
OUT = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    # name, pat, N, tol, kmode, p, K, trend, D, v, shape
    dict(name="orig_A", pat="A", N=120, tol=0.0005, kmode="level", p=3, K=2, trend="persist", D=0.02, v=0,
         rec="wick", r=0.5, m=1.0, above=False),
    dict(name="orig_B", pat="B", N=120, tol=0.0005, kmode=None, p=3, K=1, trend="persist", D=0.02, v=0,
         below=True, engulf="basic"),
    dict(name="alt_A1", pat="A", N=60, tol=0.0, kmode="touch", p=2, K=3, trend="now", D=0.0, v=2,
         rec="closepos", r=0.65, m=0.0, above=True),
    dict(name="alt_A2", pat="A", N=390, tol=0.001, kmode="level", p=5, K=3, trend="none", D=0.03, v=0,
         rec="wick", r=0.65, m=2.0, above=False),
    dict(name="alt_B1", pat="B", N=240, tol=0.001, kmode="level", p=2, K=2, trend="now", D=0.01, v=2,
         below=False, engulf="full"),
    dict(name="alt_B2", pat="B", N=60, tol=0.0, kmode="touch", p=5, K=2, trend="none", D=0.0, v=0,
         below=False, engulf="reclaim"),
]


def ema(x, n):
    return pd.Series(x).ewm(alpha=2.0 / (n + 1), adjust=False).mean().to_numpy()


def atr14(h, l, c):
    n = len(h)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    a = np.full(n, np.nan)
    if n >= 14:
        a[13] = tr[:14].mean()
        for i in range(14, n):
            a[i] = (a[i - 1] * 13 + tr[i]) / 14
    return a


def swing_mask(low, p):
    n = len(low)
    m = np.zeros(n, bool)
    for j in range(p, n - p):
        lj = low[j]
        ok = True
        for i in range(1, p + 1):
            if not (lj < low[j - i]) or not (lj <= low[j + i]):
                ok = False
                break
        m[j] = ok
    return m


def cluster_count(prices, mode):
    if len(prices) == 0:
        return 0
    ps = np.sort(prices)
    cnt = 1
    anchor = ps[0]
    prev = ps[0]
    for x in ps[1:]:
        ref = anchor if mode == "anchor" else prev
        if x > ref * 1.0015:
            cnt += 1
            anchor = x
        prev = x
    return cnt


def load(sym):
    d = pd.read_parquet(os.path.join(DATA, sym + ".parquet"))
    return d


def per_symbol(sym, d, swing_cache):
    o = d["Open"].to_numpy(float)
    h = d["High"].to_numpy(float)
    l = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    v = d["Volume"].to_numpy(float)
    idx = d.index
    n = len(d)
    e10, e25, e60 = ema(c, 10), ema(c, 25), ema(c, 60)
    stack = (e10 < e25) & (e25 < e60)
    atr = atr14(h, l, c)
    hs = pd.Series(h, index=idx)
    maxh120 = hs.rolling("120min").max().to_numpy()
    dens = hs.rolling("30min").count().to_numpy()
    sane = (~((o == h) & (h == l) & (l == c))) & (v * c >= 25000) & ((h - l) >= 0.002 * c)
    dates = np.array(idx.strftime("%Y-%m-%d"))
    tss = np.array(idx.strftime("%Y-%m-%d %H:%M"))
    vs = pd.Series(v)
    vmean20 = vs.rolling(20).mean().shift(1).to_numpy()  # mean(vol[k-20..k-1])
    stack_frac30 = pd.Series(stack.astype(float)).rolling(30).mean().to_numpy()  # bars b-29..b

    sigs, diag = [], []
    for cell in CELLS:
        N, tol, p = cell["N"], cell["tol"], cell["p"]
        key = p
        if key not in swing_cache:
            swing_cache[key] = swing_mask(l, p)
        sw = swing_cache[key]
        ls = pd.Series(l)
        # S_t = min(low[t-N .. t-2]) : window of N-1 bars ending at t-2
        S = ls.rolling(N - 1).min().shift(2).to_numpy()
        thr = S * (1 - tol)
        prev_l = np.r_[np.nan, l[:-1]]
        brk = (prev_l >= thr) & (l < thr) & (l < prev_l)
        brk &= ~np.isnan(S)
        cand = np.nonzero(brk)[0]
        for k in cand:
            Sk = S[k]
            thk = thr[k]
            # swing points j in [k-N, k-2], j + p <= k-1
            lo_j, hi_j = k - N, min(k - 2, k - 1 - p)
            js = np.nonzero(sw[lo_j:hi_j + 1])[0] + lo_j
            pts = list(l[js])
            if not np.any(l[js] == Sk):
                pts.append(Sk)
            pts = np.array(pts)
            touch = int(np.sum(pts <= Sk * 1.003))
            lvp = pts[pts <= Sk * 1.01]
            lv_a = cluster_count(lvp, "anchor")
            lv_c = cluster_count(lvp, "chain")
            if cell["kmode"] is None:
                kok_a = kok_c = True
            elif cell["kmode"] == "touch":
                kok_a = kok_c = touch >= cell["K"]
            else:
                kok_a = lv_a >= cell["K"]
                kok_c = lv_c >= cell["K"]
            b = k - 1
            if cell["trend"] == "none":
                tok = True
            elif cell["trend"] == "now":
                tok = bool(stack[b])
            else:
                tok = (b >= 30) and (stack_frac30[b] >= 0.8) and (e60[b] < e60[b - 30])
            Dv = (maxh120[k] - l[k]) / maxh120[k]
            dok = Dv >= cell["D"]
            if cell["v"] > 0:
                vr = v[k] / vmean20[k] if (k >= 20) else np.nan
                vok = bool(vr >= cell["v"]) if not np.isnan(vr) else False
            else:
                vr = np.nan
                vok = True
            rng = h[k] - l[k]
            rec = dict(cell=cell["name"], sym=sym, k=int(k), S=Sk, touch=touch, lv_a=lv_a, lv_c=lv_c,
                       kok_a=kok_a, kok_c=kok_c, tok=tok, dok=dok, Dv=Dv, vok=vok, vr=vr)
            if cell["pat"] == "A":
                t = k
                openok = o[k] >= thk
                if rng > 0:
                    recv = ((min(o[k], c[k]) - l[k]) / rng) if cell["rec"] == "wick" else ((c[k] - l[k]) / rng)
                else:
                    recv = np.nan
                rok = bool(recv >= cell["r"]) if not np.isnan(recv) else False
                if cell["m"] > 0:
                    mok = bool(rng >= cell["m"] * atr[k - 1]) if not np.isnan(atr[k - 1]) else False
                else:
                    mok = True
                aok = (c[k] >= Sk) if cell["above"] else True
                shape = openok and rok and mok and aok
                hyg = bool(sane[k]) and dens[k] >= 15
                rec.update(openok=openok, rok=rok, mok=mok, aok=aok, shape=shape, hyg=hyg,
                           ts=tss[k])
            else:
                t = k + 1
                red = c[k] < o[k]
                bok = (c[k] < thk) if cell["below"] else True
                if t < n:
                    same = dates[t] == dates[k]
                    green = c[t] > o[t]
                    eg = cell["engulf"]
                    if eg == "basic":
                        e_ok = c[t] >= o[k] - 0.1 * rng
                    elif eg == "full":
                        e_ok = c[t] >= h[k]
                    elif eg == "reclaim":
                        e_ok = c[t] >= Sk
                    else:
                        e_ok = c[t] >= o[k]
                    shape = red and bok and same and green and e_ok
                    hyg = bool(sane[k]) and bool(sane[t]) and dens[t] >= 15
                    ts = tss[t]
                else:
                    same = green = e_ok = shape = hyg = False
                    ts = ""
                rec.update(red=red, bok=bok, same=same, green=green, e_ok=e_ok, shape=shape, hyg=hyg, ts=ts)
            rec["pass_a"] = bool(kok_a and tok and dok and vok and rec["shape"] and rec["hyg"])
            rec["pass_c"] = bool(kok_c and tok and dok and vok and rec["shape"] and rec["hyg"])
            diag.append(rec)
    return diag


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    syms = [os.path.splitext(os.path.basename(f))[0] for f in files]
    syms = [s for s in syms if s not in ("SPY", "QQQ")]
    alld = []
    for s in syms:
        d = load(s)
        alld.extend(per_symbol(s, d, {}))
    diag = pd.DataFrame(alld)
    diag.to_csv(os.path.join(OUT, "diag.csv"), index=False)
    for mode in ("a", "c"):
        sg = diag[diag["pass_" + mode]][["cell", "sym", "ts", "k", "S", "touch", "lv_a", "lv_c"]]
        sg.to_csv(os.path.join(OUT, f"mine_signals_{mode}.csv"), index=False)
    print(len(syms), "symbols;", len(diag), "first-break candidates")


if __name__ == "__main__":
    main()
