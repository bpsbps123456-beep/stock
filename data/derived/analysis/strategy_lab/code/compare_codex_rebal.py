"""Is monthly sleeve rebalancing too slow versus weekly? (#239120 + H6e / H6)

Rebalancing here only resets the capital split between the two sleeves to 50:50;
each sleeve keeps trading on its own schedule. Every phase of each schedule is run
(k-th trading day of the month, each ISO weekday) so a schedule is judged by its
spread of outcomes, not by one lucky calendar day. Both sleeves open-executed, 0 cost.
"""

import numpy as np

import common as C
import compare_codex as X
import compare_codex_blend as B
import data as D
import engine as E


def sleeves():
    d = D.load()
    dates, symbols, cx = X.codex_paths()
    dates = list(dates)
    base = cx[("239120", "base")]
    alloc = B.alloc_matrix(base, len(dates), len(symbols))
    ra_full = X.open_returns_for(alloc, dates, symbols, d)
    m = E.build_market(d, exec_mode="open")
    names, stack = E.build_scores(m)
    stack = E.apply_delay(m, stack, 1)
    common = [x for x in m.days if x in set(dates)]
    ia = [dates.index(x) for x in common]
    ib = [m.days.index(x) for x in common]
    wa = [{B.ALIAS.get(symbols[j], symbols[j]): alloc[i - 1, j] for j in np.nonzero(alloc[i - 1])[0]} if i else {}
          for i in ia]
    out = {}
    for label, (s, sc, N, rb, buf, wt) in {"H6e": ("T", "MOM120", 5, 1, 5, 0),
                                           "H6": ("T", "MOM120", 5, 0, 5, 1)}.items():
        p = [names.index((s, "all", sc)), N, wt, buf, rb, 0, E.SOURCES.index(s)]
        r = C.simulate(m, stack, p, cost=0.0)[0][ib]
        W, _, _ = C.weights_path(m, stack, p)
        wb = [{m.tickers[j]: W[ib[k] - 1, j] for j in np.nonzero(W[ib[k] - 1])[0]} if k else {}
              for k in range(len(ib))]
        out[label] = (r, wb)
    return common, ra_full[ia], wa, out


def run(ra, rb, wa, wb, mask):
    sa = sb = 0.5
    r = np.zeros(len(ra))
    dev, peak_name = [], 0.0
    for t in range(1, len(ra)):
        if mask[t]:
            sa = sb = (sa + sb) / 2
        before = sa + sb
        sa *= 1 + ra[t]
        sb *= 1 + rb[t]
        r[t] = (sa + sb) / before - 1
        fa = sa / (sa + sb)
        dev.append(abs(fa - 0.5))
        combined = {}
        for n, w in wa[t].items():
            combined[n] = combined.get(n, 0.0) + fa * w
        for n, w in wb[t].items():
            combined[n] = combined.get(n, 0.0) + (1 - fa) * w
        if combined:
            peak_name = max(peak_name, max(combined.values()))
    st = C.stats(r)
    return st["total"], st["mdd"], max(dev), float(np.mean(dev)), peak_name


def masks(dates):
    ts = np.array(dates, dtype="datetime64[D]")
    month = [x[:7] for x in dates]
    iso = [np.datetime64(x).astype(object).isocalendar()[:2] for x in dates]
    weekday = [np.datetime64(x).astype(object).weekday() for x in dates]
    out = {"never": [np.zeros(len(dates), bool)], "daily": [np.ones(len(dates), bool)]}
    out["monthly"] = []
    for k in range(0, 15, 2):
        mask = np.zeros(len(dates), bool)
        for mo in sorted(set(month)):
            idx = [i for i, x in enumerate(month) if x == mo]
            if k < len(idx):
                mask[idx[k]] = True
        out["monthly"].append(mask)
    out["weekly"] = []
    for wd in range(5):
        mask = np.zeros(len(dates), bool)
        for wk in sorted(set(iso)):
            idx = [i for i, x in enumerate(iso) if x == wk and weekday[i] >= wd]
            if idx:
                mask[idx[0]] = True
        out["weekly"].append(mask)
    del ts
    return out


def main():
    dates, ra, wa, mine = sleeves()
    schedules = masks(dates)
    print("#239120 alone: " + " ".join(f"{v:+.1f}" for v in [C.stats(ra)['total'] * 100, C.stats(ra)['mdd'] * 100]))
    for label, (rb, wb) in mine.items():
        st = C.stats(rb)
        print(f"\n== #239120 + {label} ({label} alone {st['total'] * 100:+.1f}%, MDD {st['mdd'] * 100:.1f}%) ==")
        print("schedule | phases | return min / median / max      | MDD min / median / max    | 50:50 drift max (mean) | peak single name")
        for name, ms in schedules.items():
            res = np.array([run(ra, rb, wa, wb, mk) for mk in ms])
            tot, mdd, dmax, dmean, peak = res.T
            print(f"{name:8s} | {len(ms):6d} | {tot.min() * 100:+6.1f}% / {np.median(tot) * 100:+6.1f}% / {tot.max() * 100:+6.1f}% |"
                  f" {mdd.min() * 100:5.1f}% / {np.median(mdd) * 100:5.1f}% / {mdd.max() * 100:5.1f}% |"
                  f" {dmax.max() * 100:4.1f}%p ({dmean.mean() * 100:4.1f}%p)      | {peak.max() * 100:4.1f}%")


if __name__ == "__main__":
    main()
