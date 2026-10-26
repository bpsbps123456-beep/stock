"""verify_breakout.py — adversarial audit of variants/breakout.py (2026-10-10). Does not modify the builder's files.

1. Rerun the 9 variants with the builder's build() + harness and compare to results/variants/breakout.json.
2. Independent rebuild from raw cache (1d, 1d_long, 1h parquet): close, prior 252/756-day high, MA20, rs20, 160m RSI,
   own simulator (no entry_exit_picker) -> compare daily held sets and period returns.
3. Look-ahead perturbation: scramble every input row >= T (close, QQQ close, long High, r160); decisions on days <= T
   must not change. Negative control: a deliberately leaky entry (row k) must be caught.
4. Spot checks on 3 dates of the selected variant with raw numbers.
5. Robustness in 뒤: per-name contribution, top-3 share, ex-top-1/2 names, ex-best-1/2 days.
6. PLTR claims.
Run: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py variants/verify_breakout.py
"""
import copy, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
from harness import Harness, PERIODS  # noqa: E402
import breakout as BO  # noqa: E402
from rsi_breakout import RES, CACHE  # noqa: E402

FRONT, BACK = "앞 1/2~5/29", "뒤 6/1~10/9"
PER = [p[0] for p in PERIODS]


def tot(df, a, b, cost=0.0):
    x = df[(df.day >= a) & (df.day <= b)]
    r = x.ret - cost * x.turn if cost else x.ret
    return float((1 + r).prod() - 1)


# ------------------------------------------------------------------ independent data
def indep_tables(days):
    raw = pd.read_parquet(os.path.join(CACHE, "1d.parquet"))
    raw["d"] = raw.ts.dt.strftime("%Y-%m-%d")
    C = raw.pivot(index="d", columns="sym", values="Close").reindex(days)
    O = raw.pivot(index="d", columns="sym", values="Open").reindex(days)
    lg = pd.read_parquet(os.path.join(CACHE, "1d_long.parquet"))
    lg["d"] = lg.ts.dt.strftime("%Y-%m-%d")
    cal = sorted(lg[lg.sym == "QQQ"].d)                     # long trading calendar
    LH = lg.pivot(index="d", columns="sym", values="High").reindex(cal)
    LC = lg.pivot(index="d", columns="sym", values="Close").reindex(cal)
    # prior n-day high through the PREVIOUS session, on the long calendar, then mapped to days
    prev = {n: LH.rolling(n, min_periods=1).max().shift(1).reindex(days) for n in (252, 756)}
    # adjustment consistency 1d vs long
    common = [d for d in days if d in LC.index]
    rel = (C.loc[common] / LC.loc[common, C.columns] - 1).abs()
    hrel = (raw.pivot(index="d", columns="sym", values="High").reindex(common) / LH.loc[common, C.columns] - 1).abs()
    adj = dict(close_max_rel_diff=float(np.nanmax(rel.to_numpy())), high_max_rel_diff=float(np.nanmax(hrel.to_numpy())))
    # 160m RSI from 1h, own code: regular hours, 3 bars/day ending 12:30, 14:30, 16:00; Wilder RSI 14 seeded by SMA
    h1 = pd.read_parquet(os.path.join(CACHE, "1h.parquet"))
    m = h1.ts.dt.hour * 60 + h1.ts.dt.minute
    h1 = h1[(m >= 570) & (m < 960)].copy()
    m = h1.ts.dt.hour * 60 + h1.ts.dt.minute
    h1["g"] = np.select([m < 750, m < 870], [0, 1], 2)
    h1["d"] = h1.ts.dt.strftime("%Y-%m-%d")
    h1 = h1.sort_values("ts")
    r160 = {}
    for s, g in h1.groupby("sym"):
        b = g.groupby(["d", "g"]).Close.last()
        x = b.to_numpy(float)
        d = np.diff(x)
        up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
        r = np.full(len(x), np.nan)
        if len(d) >= 14:
            au, ad = up[:14].mean(), dn[:14].mean()
            for i in range(14, len(d) + 1):
                if i > 14:
                    au = (au * 13 + up[i - 1]) / 14
                    ad = (ad * 13 + dn[i - 1]) / 14
                r[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
        ser = pd.Series(r, index=b.index.get_level_values(0))
        r160[s] = ser.groupby(level=0).last().reindex(days)
    R160 = pd.DataFrame(r160)
    return C, O, prev, R160, adj, cal, LH


def indep_sim(days, k0, syms, C, O, entry_tbl, exit_kind, prio_tbl, MA, elig, slots=7):
    """Own simulator. entry_tbl[j, s] = signal known at close j. exit_kind: ('trail', x) or ('ma',)."""
    Cv, Ov, Ev, Pv, Mv = (t.reindex(columns=syms).to_numpy(float) if t.dtypes.iloc[0] != bool
                          else t.reindex(columns=syms).fillna(False).to_numpy(bool) for t in (C, O, entry_tbl, prio_tbl, MA))
    idx = {s: i for i, s in enumerate(syms)}
    held, info, rows, events = [], {}, [], []
    for k in range(k0, len(days) - 1):
        keep = []
        for s in held:
            j = idx[s]
            p = info[s]
            c = Cv[k - 1, j]
            if k - 1 >= p["k0"] and np.isfinite(c):
                p["peak"] = max(p["peak"], c)
            if exit_kind[0] == "trail":
                out = c < (1 - exit_kind[1]) * p["peak"]
            else:
                out = c < Mv[k - 1, j]
            if out:
                events.append(("sell", days[k], s))
            else:
                keep.append(s)
        for s in [s for s in info if s not in keep]:
            info.pop(s)
        el = elig(days[k])
        cand = [s for s in syms if s not in keep and s in el and np.isfinite(Ov[k, idx[s]]) and Ev[k - 1, idx[s]]]
        cand.sort(key=lambda s: -(Pv[k - 1, idx[s]] if np.isfinite(Pv[k - 1, idx[s]]) else -1e9))
        add = cand[:max(0, slots - len(keep))]
        for s in add:
            info[s] = {"k0": k, "peak": Ov[k, idx[s]]}
            events.append(("buy", days[k], s))
        held = keep + add
        w = 1 / max(slots, len(held)) if held else 0
        r = 0.0
        for s in held:
            a, b = Ov[k, idx[s]], Ov[k + 1, idx[s]]
            r += w * (b / a - 1) if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
        rows.append(dict(day=days[k], ret=r, held=",".join(sorted(held)), n=len(held)))
    return pd.DataFrame(rows), events


# ------------------------------------------------------------------ robustness helpers
def contrib_in(H, df, a, b):
    x = df[(df.day >= a) & (df.day <= b)]
    pos = {d: i for i, d in enumerate(H.days)}
    val, c, daily = 1.0, {}, []
    for _, r in x.iterrows():
        hs = r.held.split(",") if r.held else []
        k = pos[r.day]
        w = 1 / max(7, len(hs)) if hs else 0
        day_r = {}
        for s in hs:
            o0, o1 = H.opn[s].iloc[k], H.opn[s].iloc[k + 1]
            day_r[s] = o1 / o0 - 1 if np.isfinite(o0) and np.isfinite(o1) and o0 > 0 else 0.0
            c[s] = c.get(s, 0.0) + val * w * day_r[s]
        daily.append((r.day, w, day_r))
        val *= 1 + sum(w * v for v in day_r.values())
    gain = val - 1
    top = sorted(c.items(), key=lambda kv: -kv[1])

    def excl_names(names):
        v = 1.0
        for _, w, dr in daily:
            v *= 1 + sum(w * rr for s, rr in dr.items() if s not in names)
        return v - 1
    dr = sorted((sum(w * v for v in d.values()) for _, w, d in daily), reverse=True)
    full = np.prod([1 + sum(w * v for v in d.values()) for _, w, d in daily]) - 1

    def excl_days(n):
        rs = [sum(w * v for v in d.values()) for _, w, d in daily]
        drop = sorted(range(len(rs)), key=lambda i: -rs[i])[:n]
        return float(np.prod([1 + r for i, r in enumerate(rs) if i not in drop]) - 1)
    return dict(ret=round(float(full), 4), top5=[(s, round(v, 4)) for s, v in top[:5]],
                top3_share=(round(sum(v for _, v in top[:3]) / gain, 3) if gain > 0 else None),
                ex_top1=round(float(excl_names({top[0][0]})), 4) if top else None,
                ex_top2=round(float(excl_names({top[0][0], top[1][0]})), 4) if len(top) > 1 else None,
                ex_best1day=round(excl_days(1), 4), ex_best2days=round(excl_days(2), 4),
                best2days=[round(float(v), 4) for v in dr[:2]])


def perturb(H, T, seed):
    rng = np.random.default_rng(seed)
    kT = H.days.index(T)
    H2 = copy.copy(H)
    H2._cache = {}
    c = H.cls.copy()
    c.iloc[kT:] = c.iloc[kT:].to_numpy() * np.exp(rng.normal(0, 0.4, c.iloc[kT:].shape))
    H2.cls = c
    q = H.qqq_c.copy()
    q.iloc[kT:] = q.iloc[kT:].to_numpy() * np.exp(rng.normal(0, 0.4, len(q) - kT))
    H2.qqq_c = q
    r = H.r160.copy()
    r.iloc[kT:] = rng.uniform(0, 100, r.iloc[kT:].shape)
    H2.r160 = r
    lh = {}
    for s, h in H._long_high.items():
        h = h.copy()
        msk = h.index >= T
        h[msk] = h[msk].to_numpy() * np.exp(rng.normal(0, 0.4, msk.sum()))
        lh[s] = h
    H2._long_high = lh
    return H2


def main():
    out = {}
    H = Harness()
    days = H.days
    k0 = next(i for i, d in enumerate(days) if d >= "2026-01-02")
    stored = {r["name"]: r for r in json.load(open(os.path.join(RES, "variants", "breakout.json"), encoding="utf-8"))}
    entries, exits, sig = BO.build(H)
    prio = lambda k: H.rs(20).iloc[k - 1]  # noqa: E731

    # ---------------- 1. rerun builder variants
    print("\n[1] rerun builder variants vs stored json (max abs diff over periods)")
    dfs = {}
    for ek, (en, ef) in entries.items():
        for xk, (xn, xf) in exits.items():
            name = f"돌파 {ek}:{en} / {xn}"
            df = H.simulate(H.entry_exit_picker(ef, xf, priority=prio))
            dfs[(ek, xk)] = df
            ev = H.evaluate(name, df)
            diff = max(abs(ev[p]["수익"] - stored[name][p]["수익"]) for p in PER)
            print(f"  {name:<32} maxdiff {diff:.2e}")
    out["rerun_maxdiff"] = "see stdout"

    # ---------------- 2. independent rebuild
    C, O, prev, R160, adj, cal, LH = indep_tables(days)
    print("\n[2] adjustment consistency 1d vs 1d_long:", adj)
    print("    calendar: H.days contiguous in long calendar?",
          all(cal[cal.index(days[i]) + 1] == days[i + 1] for i in range(len(days) - 1)))
    syms = H.syms
    Cs = C[syms]
    # compare harness tables to independent
    hc = (H.cls[syms] - Cs).abs().max().max()
    hp252 = (H.hi(252)[syms].shift(1) - prev[252][syms]).abs().iloc[k0 - 2:].max().max()
    hp756 = (H.hi(756)[syms].shift(1) - prev[756][syms]).abs().iloc[k0 - 2:].max().max()
    rr = (H.r160.reindex(columns=syms) - R160.reindex(columns=syms)).abs().iloc[k0 - 2:]
    print(f"    cls maxdiff {hc:.3g} | prev252 maxdiff {hp252:.3g} | prev756 maxdiff {hp756:.3g} | "
          f"r160 maxdiff (2026) {np.nanmax(rr.to_numpy()):.3g}, r160 NaN mismatch "
          f"{int((H.r160.reindex(columns=syms).iloc[k0 - 2:].isna() != R160.reindex(columns=syms).iloc[k0 - 2:].isna()).sum().sum())}")
    MA = Cs.rolling(20, min_periods=20).mean()
    Q = C["QQQ"]
    RS = (Cs / Cs.shift(20)).div(Q / Q.shift(20), axis=0) - 1
    EA = Cs > prev[252][syms]
    EB = Cs > prev[756][syms]
    EC = EA & (R160.reindex(columns=syms) > 70)
    ent = {"A": EA, "B": EB, "C": EC}
    exk = {"T10": ("trail", 0.10), "T15": ("trail", 0.15), "MA20": ("ma",)}
    print("\n[2b] independent simulator vs builder (held-set mismatching days, period returns indep / builder)")
    indep = {}
    for ek in "ABC":
        for xk in ("T10", "T15", "MA20"):
            df_i, ev_i = indep_sim(days, k0, syms, Cs, O[syms], ent[ek], exk[xk], RS, MA, H.eligible)
            indep[(ek, xk)] = (df_i, ev_i)
            df_b = dfs[(ek, xk)]
            mism = int((df_i.held.to_numpy() != df_b.held.to_numpy()).sum())
            cells = " ".join(f"{tot(df_i, a, b):+.4f}/{tot(df_b, a, b):+.4f}" for _, a, b in PERIODS)
            print(f"  {ek}/{xk:<4} mismatch days {mism:3d} | {cells}")

    # ---------------- 3. look-ahead perturbation
    print("\n[3] look-ahead test: scramble rows >= T, compare held sets on days <= T (should be 0 mismatches)")
    Ts = ["2026-02-17", "2026-04-15", "2026-06-24", "2026-08-20", "2026-10-02"]
    la = []
    for i, T in enumerate(Ts):
        H2 = perturb(H, T, seed=100 + i)
        e2, x2, _ = BO.build(H2)
        p2 = lambda k, H2=H2: H2.rs(20).iloc[k - 1]  # noqa: E731
        kT = days.index(T)
        bad, changed_after = 0, 0
        for ek, (_, ef) in e2.items():
            for xk, (_, xf) in x2.items():
                d2 = H2.simulate(H2.entry_exit_picker(ef, xf, priority=p2))
                d1 = dfs[(ek, xk)]
                m_ = (d1.day <= T).to_numpy()
                bad += int((d1.held.to_numpy()[m_] != d2.held.to_numpy()[m_]).sum())
                changed_after += int((d1.held.to_numpy()[~m_] != d2.held.to_numpy()[~m_]).sum())
        # negative control: leaky entry using row k (same-day close) must differ on/before T
        ea2 = (H2.cls[H2.syms] > H2.hi(252)[H2.syms].shift(1)).to_numpy()
        col = {s: j for j, s in enumerate(H2.syms)}
        leak = lambda s, k, e=ea2: bool(e[k, col[s]])  # noqa: E731
        ea1 = (H.cls[H.syms] > H.hi(252)[H.syms].shift(1)).to_numpy()
        leak1 = lambda s, k, e=ea1: bool(e[k, col[s]])  # noqa: E731
        xf = x2["T10"][1]
        l2 = H2.simulate(H2.entry_exit_picker(leak, xf, priority=p2))
        l1 = H.simulate(H.entry_exit_picker(leak1, exits["T10"][1], priority=prio))
        m_ = (l1.day <= T).to_numpy()
        ctrl = int((l1.held.to_numpy()[m_] != l2.held.to_numpy()[m_]).sum())
        la.append((T, bad, changed_after, ctrl))
        print(f"  T={T}: mismatches on days<=T {bad} (9 variants) | after T {changed_after} | leaky control mismatches<=T {ctrl}")
    out["lookahead"] = la

    # ---------------- 4. spot checks (selected variant C/MA20)
    print("\n[4] spot checks C/MA20 (entry: close[b-1] > prior-252 high through b-2 AND r160[b-1] > 70; "
          "exit at open s when close[s-1] < MA20[s-1])")
    dfb = dfs[("C", "MA20")]
    _, evs = indep[("C", "MA20")]
    pos = {d: i for i, d in enumerate(days)}
    runs = [set(h.split(",")) if h else set() for h in dfb.held]
    dlist = dfb.day.tolist()
    for D in ["2026-03-02", "2026-07-20", "2026-10-06"]:
        i = dlist.index(D)
        print(f"  {D} held: {sorted(runs[i])}")
        for s in sorted(runs[i]):
            j = i
            while j > 0 and s in runs[j - 1]:
                j -= 1
            b = pos[dlist[j]]
            nxt = i
            while nxt + 1 < len(runs) and s in runs[nxt + 1]:
                nxt += 1
            sell = dlist[nxt + 1] if nxt + 1 < len(runs) else None
            c1, p1, r1 = C[s].iloc[b - 1], prev[252][s].iloc[b - 1], R160[s].iloc[b - 1]
            no_exit = all(C[s].iloc[t - 1] >= MA[s].iloc[t - 1] for t in range(b + 1, pos[D] + 1))
            line = (f"    {s:<5} buy {days[b]} (signal {days[b - 1]}: close {c1:.2f} > prev252 {p1:.2f}? {c1 > p1}; "
                    f"r160 {r1:.1f}>70? {r1 > 70}; elig {s in H.eligible(days[b])}) no exit until {D}: {no_exit}")
            if sell:
                ks = pos[sell]
                line += (f" | sold {sell}: close[{days[ks - 1]}] {C[s].iloc[ks - 1]:.2f} < MA20 {MA[s].iloc[ks - 1]:.2f}? "
                         f"{C[s].iloc[ks - 1] < MA[s].iloc[ks - 1]}; first breach before? "
                         f"{any(C[s].iloc[t - 1] < MA[s].iloc[t - 1] for t in range(b + 1, ks))}")
            else:
                line += " | still held at end"
            print(line)
    # slot binding check: days where signals exceeded free slots
    print("  priority check: days with more signals than slots, chosen vs rejected rs20[k-1]")
    shown = 0
    for k in range(k0, len(days) - 1):
        if shown >= 3:
            break
        d = days[k]
        i = dlist.index(d)
        prevh = runs[i - 1] if i else set()
        keep = prevh & runs[i]
        el = H.eligible(d)
        cand = [s for s in syms if s not in keep and s in el and np.isfinite(O[s].iloc[k]) and bool(EC[s].iloc[k - 1])]
        free = 7 - len(keep)
        if len(cand) > free > 0:
            chosen = runs[i] - keep
            rej = set(cand) - chosen
            mn = min(RS[s].iloc[k - 1] for s in chosen)
            mx = max(RS[s].iloc[k - 1] for s in rej)
            print(f"    {d}: free {free}, cand {len(cand)}, min chosen rs20 {mn:+.3f} >= max rejected {mx:+.3f}? {mn >= mx}")
            shown += 1

    # ---------------- 5. robustness
    print("\n[5] robustness: 뒤 / 9/1~ / 10/1~ concentration")
    nm = {("A", "T10"): "A/T10", ("A", "T15"): "A/T15", ("A", "MA20"): "A/MA20", ("B", "T10"): "B/T10",
          ("B", "T15"): "B/T15", ("B", "MA20"): "B/MA20", ("C", "T10"): "C/T10", ("C", "T15"): "C/T15",
          ("C", "MA20"): "C/MA20"}
    rob = {}
    for key, df in dfs.items():
        rb = contrib_in(H, df, "2026-06-01", "2026-10-09")
        rb9 = contrib_in(H, df, "2026-09-01", "2026-10-09")
        rob[nm[key]] = dict(back=rb, sep=dict(ret=rb9["ret"], top3=rb9["top5"][:3], top3_share=rb9["top3_share"]))
        print(f"  {nm[key]:<7} 뒤 {rb['ret']:+.4f} top5 {rb['top5']} top3몫 {rb['top3_share']} | ex top1 {rb['ex_top1']:+.4f} "
              f"ex top2 {rb['ex_top2']:+.4f} | ex best day {rb['ex_best1day']:+.4f} ex best 2 days {rb['ex_best2days']:+.4f}"
              f" (best days {rb['best2days']})")
        print(f"           9/1~ {rb9['ret']:+.4f} top3 {rb9['top5'][:3]} share {rb9['top3_share']}")
    out["robust"] = rob

    # trade stats (indep events)
    print("\n  avg holding days per trade (independent, rows held):")
    for key, df in dfs.items():
        tr = BO.trade_list(H, df)
        print(f"   {nm[key]:<7} trades {len(tr)} mean days {tr.days.mean():.1f} median {tr.days.median():.0f}")

    # post-IPO 'new high' share: trades whose buy is within 252 sessions of first long-data date
    lg_first = {s: h.index.min() for s, h in H._long_high.items()}
    tr = BO.trade_list(H, dfs[("C", "MA20")])
    young = []
    for _, t in tr.iterrows():
        f = lg_first.get(t.sym)
        if f and f > "2023-01-03":
            n_since = sum(1 for d in cal if f <= d < t.buy)
            if n_since < 252:
                young.append((t.sym, t.buy, n_since, round(t.ret, 3)))
    print("  C/MA20 trades in names listed <252 sessions before buy (52w high = since-listing high):", young)

    # ---------------- sensitivity (audit only, not a variant / not used for selection)
    print("\n[5b] audit sensitivity around the selected point (C entry RSI thr x MA exit length); NOT selection")
    for thr in (65, 70, 75):
        for n in (15, 20, 25):
            ec = (EA & (R160.reindex(columns=syms) > thr))
            man = Cs.rolling(n, min_periods=n).mean()
            d_i, _ = indep_sim(days, k0, syms, Cs, O[syms], ec, ("ma",), RS, man, H.eligible)
            print(f"   RSI>{thr} MA{n}: 앞 {tot(d_i, '2026-01-02', '2026-05-29'):+.4f} 뒤 {tot(d_i, '2026-06-01', '2026-10-09'):+.4f}"
                  f" 9/1~ {tot(d_i, '2026-09-01', '2026-10-09'):+.4f} 10/1~ {tot(d_i, '2026-10-01', '2026-10-09'):+.4f}")
    for x in (0.08, 0.10, 0.12, 0.15, 0.20):
        d_i, _ = indep_sim(days, k0, syms, Cs, O[syms], EA, ("trail", x), RS, MA, H.eligible)
        print(f"   A trail {x:.2f}: 앞 {tot(d_i, '2026-01-02', '2026-05-29'):+.4f} 뒤 {tot(d_i, '2026-06-01', '2026-10-09'):+.4f}"
              f" 9/1~ {tot(d_i, '2026-09-01', '2026-10-09'):+.4f} 10/1~ {tot(d_i, '2026-10-01', '2026-10-09'):+.4f}")

    # ---------------- 6. PLTR
    print("\n[6] PLTR independent")
    s = "PLTR"
    for d in [d for d in days if "2026-09-01" <= d <= "2026-10-09"]:
        k = pos[d]
        a, b_, c_ = bool(EA[s].iloc[k]), bool(EB[s].iloc[k]), bool(EC[s].iloc[k])
        if a or b_ or c_ or d >= "2026-10-06":
            print(f"   {d} close {C[s].iloc[k]:.2f} prev252 {prev[252][s].iloc[k]:.2f} prev756 {prev[756][s].iloc[k]:.2f}"
                  f" r160 {R160[s].iloc[k]:.1f} rs20 {RS[s].iloc[k]:+.3f} A {a} B {b_} C {c_}")
    lh = LH[s].dropna()
    print("   argmax High before 10/9:", lh[lh.index < "2026-10-09"].idxmax(), float(lh[lh.index < "2026-10-09"].max()))
    print("   PLTR eligible 10/12:", s in H.eligible("2026-10-12"), "| held by any variant:",
          any(s in h for df in dfs.values() for h in df.held))
    json.dump(out, open(os.path.join(RES, "variants", "verify_breakout.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)


def extra():
    """Second pass: stronger leak control, independent 10/12 decision, best-day dates, MRVL trades in 뒤."""
    H = Harness()
    days = H.days
    n = len(days)
    k0 = next(i for i, d in enumerate(days) if d >= "2026-01-02")
    entries, exits, _ = BO.build(H)
    prio = lambda k: H.rs(20).iloc[k - 1]  # noqa: E731
    syms = H.syms
    col = {s: j for j, s in enumerate(syms)}
    print("\n[3b] stronger negative control: leaky trailing-10% exit using close[k] (same day) with entry A")
    for i, T in enumerate(["2026-02-17", "2026-04-15", "2026-06-24", "2026-08-20", "2026-10-02"]):
        H2 = perturb(H, T, seed=100 + i)
        e2, x2, _ = BO.build(H2)
        p2 = lambda k, H2=H2: H2.rs(20).iloc[k - 1]  # noqa: E731
        C1, C2 = H.cls[syms].to_numpy(), H2.cls[syms].to_numpy()
        lx1 = lambda s, k, p, C=C1: bool(C[k, col[s]] < 0.9 * p["peak"])  # noqa: E731
        lx2 = lambda s, k, p, C=C2: bool(C[k, col[s]] < 0.9 * p["peak"])  # noqa: E731
        l1 = H.simulate(H.entry_exit_picker(entries["A"][1], lx1, priority=prio))
        l2 = H2.simulate(H2.entry_exit_picker(e2["A"][1], lx2, priority=p2))
        m_ = (l1.day <= T).to_numpy()
        print(f"  T={T}: leaky-exit mismatches on days<=T {int((l1.held.to_numpy()[m_] != l2.held.to_numpy()[m_]).sum())}")

    # independent 10/12 decision from the simulated held set
    C, O, prev, R160, adj, cal, LH = indep_tables(days)
    Cs = C[syms]
    MA = Cs.rolling(20, min_periods=20).mean()
    Q = C["QQQ"]
    RS = (Cs / Cs.shift(20)).div(Q / Q.shift(20), axis=0) - 1
    EA = Cs > prev[252][syms]
    ent = {"A": EA, "B": Cs > prev[756][syms], "C": EA & (R160.reindex(columns=syms) > 70)}
    exk = {"T10": ("trail", 0.10), "T15": ("trail", 0.15), "MA20": ("ma",)}
    stored = {r["name"]: r for r in json.load(open(os.path.join(RES, "variants", "breakout.json"), encoding="utf-8"))}
    print("\n[6b] independent 10/9-open and 10/12-open decisions (display only)")
    elig12 = H.eligible("2026-10-12")
    for ek, (en, _) in entries.items():
        for xk, (xn, _) in exits.items():
            name = f"돌파 {ek}:{en} / {xn}"
            # extend own sim through 10/9 open by appending a dummy day for the loop end
            df_i, ev = indep_sim(days + ["2026-10-12"], k0, syms, Cs.reindex(days + ["2026-10-12"]),
                                 O[syms].reindex(days + ["2026-10-12"]),
                                 ent[ek].reindex(days + ["2026-10-12"]).fillna(False).astype(bool),
                                 exk[xk], RS.reindex(days + ["2026-10-12"]), MA.reindex(days + ["2026-10-12"]), H.eligible)
            h9 = df_i.held.iloc[-1].split(",") if df_i.held.iloc[-1] else []     # held from 10/9 open
            # peaks: rebuild from events
            last_buy = {}
            for kind, d, s in ev:
                if kind == "buy":
                    last_buy[s] = d
            keep, sold = [], []
            for s in h9:
                kb = days.index(last_buy[s])
                peak = max(O[s].iloc[kb], np.nanmax(Cs[s].iloc[kb:n].to_numpy()))
                c = Cs[s].iloc[n - 1]
                out = c < (1 - exk[xk][1]) * peak if exk[xk][0] == "trail" else c < MA[s].iloc[n - 1]
                (sold if out else keep).append(s)
            cand = [s for s in syms if s not in keep and s in elig12 and bool(ent[ek][s].iloc[n - 1])]
            cand.sort(key=lambda s: -(RS[s].iloc[n - 1] if np.isfinite(RS[s].iloc[n - 1]) else -1e9))
            buy = cand[:max(0, 7 - len(keep))]
            st = stored[name]["10/12결정(표시만)"]
            same = (sorted(h9) == st["held_1009"] and sorted(sold) == st["sell_1012"] and buy == st["buy_1012"])
            print(f"  {name:<30} held10/9 {sorted(h9)} sell {sold} signals {cand[:6]} buy {buy} | matches builder {same}")
            if ek == "C" and xk == "MA20":
                print("    rs20 at 10/9 of held:", {s: round(float(RS[s].iloc[n - 1]), 3) for s in sorted(h9)},
                      "PLTR", round(float(RS["PLTR"].iloc[n - 1]), 3))

    # best days in 뒤 and MRVL trades
    print("\n[5c] best 뒤 days and MRVL trades in 뒤")
    for ek, xk in (("C", "MA20"), ("A", "T10"), ("A", "T15")):
        df = H.simulate(H.entry_exit_picker(entries[ek][1], exits[xk][1], priority=prio))
        b = df[(df.day >= "2026-06-01")]
        top = b.sort_values("ret", ascending=False).head(3)
        print(f"  {ek}/{xk} best 뒤 days: {list(zip(top.day, top.ret.round(4), top.n))}")
        tr = BO.trade_list(H, df)
        print("    MRVL trades:", tr[tr.sym == "MRVL"][["buy", "sell", "days", "ret"]].to_dict("records"))
        print("    month 뒤:", {m: round(tot(df[df.day.str.startswith(m)], "0", "9"), 4) for m in
                               ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10")})
        qo = H.qqq_o
        print("    QQQ open->open on those days:",
              {d: round(float(qo.iloc[days.index(d) + 1] / qo.iloc[days.index(d)] - 1), 4) for d in top.day})


def fresh():
    """Robustness: start each variant with an EMPTY book on 6/1, 7/1, 9/1, 10/1 (no positions carried from 앞)."""
    from rsi_breakout import desk_curve
    H = Harness()
    days = H.days
    syms = H.syms
    C, O, prev, R160, adj, cal, LH = indep_tables(days)
    Cs = C[syms]
    MA = Cs.rolling(20, min_periods=20).mean()
    Q = C["QQQ"]
    RS = (Cs / Cs.shift(20)).div(Q / Q.shift(20), axis=0) - 1
    EA = Cs > prev[252][syms]
    ent = {"A": EA, "B": Cs > prev[756][syms], "C": EA & (R160.reindex(columns=syms) > 70)}
    exk = {"T10": ("trail", 0.10), "T15": ("trail", 0.15), "MA20": ("ma",)}
    starts = ["2026-06-01", "2026-07-01", "2026-09-01", "2026-10-01"]
    qo = H.qqq_o
    end = len(days) - 1                                   # last return is open(10/8)->open(10/9)
    print("\n[5d] fresh start (empty book) -> 10/9 open; carried = harness run from 1/2 over same window")
    hdr = "  variant  " + " ".join(f"{s[5:]}:fresh/carried".rjust(22) for s in starts)
    print(hdr)
    rows = {}
    for ek in "ABC":
        for xk in ("T10", "T15", "MA20"):
            full, _ = indep_sim(days, next(i for i, d in enumerate(days) if d >= "2026-01-02"), syms, Cs, O[syms],
                                ent[ek], exk[xk], RS, MA, H.eligible)
            cells, rr = [], {}
            for s0 in starts:
                k0 = days.index(s0)
                d_i, _ = indep_sim(days, k0, syms, Cs, O[syms], ent[ek], exk[xk], RS, MA, H.eligible)
                f = tot(d_i, s0, "2026-10-09")
                c = tot(full, s0, "2026-10-09")
                rr[s0] = (round(f, 4), round(c, 4), round(float(d_i.n.mean()), 2))
                cells.append(f"{f:+.4f}/{c:+.4f}(n{d_i.n.mean():.1f})".rjust(22))
            rows[f"{ek}/{xk}"] = rr
            print(f"  {ek}/{xk:<6} " + " ".join(cells))
    qcells = []
    for s0 in starts:
        k0 = days.index(s0)
        qcells.append(f"{float(qo.iloc[end] / qo.iloc[k0] - 1):+.4f}".rjust(22))
    print("  QQQ      " + " ".join(qcells))
    try:
        d = desk_curve(H.P)
        print("  desk     " + " ".join(f"{tot(d, s0, '2026-10-09'):+.4f}".rjust(22) for s0 in starts))
    except Exception as exc:  # noqa: BLE001
        print("  desk failed", exc)
    json.dump(rows, open(os.path.join(RES, "variants", "verify_breakout_fresh.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["extra"]:
        extra()
    elif sys.argv[1:] == ["fresh"]:
        fresh()
    else:
        main()
