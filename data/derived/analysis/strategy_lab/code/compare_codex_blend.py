"""Audit of the section-16 claims after Codex's review (2026-09-27).

1. Contributions of #239120 on Codex's close/base path, as capital (prior equity x daily
   return) rather than the earlier simple sum of daily returns.
2. #239120 delay sensitivity with the SAME open execution (base vs lag1 decisions).
3. #239120 + H6e blends under explicit rebalancing rules (never / monthly / weekly /
   daily back to 50:50) and the largest combined single-stock weight.
Open paths execute each day's decision at that day's US open, 0 cost, KRW.
"""

import numpy as np

import common as C
import compare_codex as X
import data as D
import engine as E

ALIAS = {"GOOG": "GOOGL", "SATS": "ECHO"}


def capital_contrib(dates, returns, holdings, symbols, lo=None, hi=None):
    """Per-name gain in % of the equity at the window start (1-stock path)."""
    eq, out = 1.0, {}
    start_eq = None
    for t in range(1, len(dates)):
        if lo and dates[t] < lo:
            eq *= 1 + returns[t]
            continue
        if hi and dates[t] > hi:
            break
        if start_eq is None:
            start_eq = eq
        j = int(holdings[t - 1, 0])
        name = symbols[j] if j >= 0 else "현금"
        out[name] = out.get(name, 0.0) + eq * returns[t]
        eq *= 1 + returns[t]
    base = start_eq if start_eq else 1.0
    return {k: v / base for k, v in out.items()}, eq / base - 1


def alloc_matrix(det, n_days, n_sym):
    a = np.zeros((n_days, n_sym))
    for t in range(n_days):
        for s in range(det["holdings"].shape[1]):
            j = int(det["holdings"][t, s])
            if j >= 0:
                a[t, j] += float(det["allocations"][t, s])
    return a


def blend(ra, rb, wa_names, wb_names, dates, rule):
    """Two sleeves starting 50:50; rebalance back to 50:50 on the rule's days."""
    sa, sb = 0.5, 0.5
    out = np.zeros(len(ra))
    max_name, max_where = 0.0, ""
    prev_key = None
    for t in range(1, len(ra)):
        key = {"never": None, "monthly": dates[t][:7], "weekly": np.datetime64(dates[t], "W").astype(str),
               "daily": dates[t]}[rule]
        if rule != "never" and key != prev_key:
            total = sa + sb
            sa, sb = total / 2, total / 2
        prev_key = key
        total0 = sa + sb
        sa *= 1 + ra[t]
        sb *= 1 + rb[t]
        out[t] = (sa + sb) / total0 - 1
        frac_a = sa / (sa + sb)
        combined = {}
        for name, w in wa_names[t].items():
            combined[name] = combined.get(name, 0.0) + frac_a * w
        for name, w in wb_names[t].items():
            combined[name] = combined.get(name, 0.0) + (1 - frac_a) * w
        if combined:
            name, w = max(combined.items(), key=lambda x: x[1])
            if w > max_name:
                max_name, max_where = w, f"{dates[t]} {name}"
    return out, max_name, max_where


def main():
    d = D.load()
    dates, symbols, cx = X.codex_paths()
    dates = list(dates)
    base = cx[("239120", "base")]

    print("== 1. #239120 contributions on Codex close/base path, 2026-01-02..09-22 ==")
    cut = dates.index("2026-09-22") + 1
    contrib, total = capital_contrib(dates[:cut], base["returns"][:cut], base["holdings"][:cut], symbols)
    top = sorted(contrib.items(), key=lambda x: -x[1])
    print(f"  total gain {total * 100:+.2f}% | capital contribution %p of initial: "
          + ", ".join(f"{k} {v * 100:+.2f}" for k, v in top))
    litearm = contrib.get("LITE", 0) + contrib.get("ARM", 0)
    print(f"  LITE+ARM = {litearm * 100:+.2f}%p = {litearm / total * 100:.1f}% of the total gain")
    jj, jj_total = capital_contrib(dates, base["returns"], base["holdings"], symbols, "2026-06-01", "2026-07-31")
    print(f"  Jun-Jul total {jj_total * 100:+.2f}% split (vs June-start equity): "
          + ", ".join(f"{k} {v * 100:+.2f}%p" for k, v in sorted(jj.items(), key=lambda x: -x[1])))

    print("\n== 2. #239120 delay with the same open execution (0원) ==")
    for name in ("base", "lag1"):
        r = X.open_returns_for(alloc_matrix(cx[("239120", name)], len(dates), len(symbols)), dates, symbols, d)
        st = X.metrics(dates, r)
        print(f"  open fill, Codex {name:4s} decisions: {st['total'] * 100:+6.1f}%  MDD {st['mdd'] * 100:5.1f}%")

    # sleeves on the common open path
    ra_full = X.open_returns_for(alloc_matrix(base, len(dates), len(symbols)), dates, symbols, d)
    alloc = alloc_matrix(base, len(dates), len(symbols))
    m = E.build_market(d, exec_mode="open")
    names, stack = E.build_scores(m)
    stack = E.apply_delay(m, stack, 1)
    common = [x for x in m.days if x in set(dates)]
    ia = [dates.index(x) for x in common]
    ib = [m.days.index(x) for x in common]
    ra = ra_full[ia]
    wa = [{ALIAS.get(symbols[j], symbols[j]): alloc[i - 1, j] for j in np.nonzero(alloc[i - 1])[0]} if i else {}
          for i in ia]
    print("\n== 3. #239120 + my rule, both sleeves open-executed, start 50:50 (0원, 1/2..9/22) ==")
    for label, spec in (("H6e", ("T", "MOM120", 5, 1, 5, 0)), ("H6", ("T", "MOM120", 5, 0, 5, 1))):
        s, sc, N, rb, buf, wt = spec
        p = [names.index((s, "all", sc)), N, wt, buf, rb, 0, E.SOURCES.index(s)]
        rb_series = C.simulate(m, stack, p, cost=0.0)[0][ib]
        W, _, _ = C.weights_path(m, stack, p)
        wb = [{m.tickers[j]: W[ib[k] - 1, j] for j in np.nonzero(W[ib[k] - 1])[0]} if k else {}
              for k in range(len(ib))]
        for tag, r in (("#239120 alone", ra), (f"{label} alone", rb_series)):
            st = C.stats(r)
            print(f"  {tag:16s} {st['total'] * 100:+6.1f}%  MDD {st['mdd'] * 100:5.1f}%")
        for rule in ("never", "monthly", "weekly", "daily"):
            r, mx, where = blend(ra, rb_series, wa, wb, common, rule)
            st = C.stats(r)
            print(f"  blend {label} rebalance {rule:7s} {st['total'] * 100:+6.1f}%  MDD {st['mdd'] * 100:5.1f}%"
                  f"  max single-name weight {mx * 100:4.1f}% ({where})")


if __name__ == "__main__":
    main()
