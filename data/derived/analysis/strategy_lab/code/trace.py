"""Trace one strategy: holdings at each rebalance and per-period returns."""
import numpy as np
import data as D, engine as E, grid as G

def simulate(m, stack, names, param, cost=G.COST):
    reb_stack = np.stack([m.reb[r] for r in G.REBS])
    fw = np.stack([m.fw[s] for s in E.SOURCES]); held = np.stack([m.held[s] for s in E.SOURCES])
    iv = np.nan_to_num(m.iv, nan=0.0, posinf=0.0)
    T = len(m.days); P = np.array([param], np.int64)
    g = np.zeros((1, T)); tu = np.zeros((1, T)); nh = np.zeros((1, T), np.int64)
    E.run_batch(m.R, m.dec_row, reb_stack, stack, fw, iv, held, P, G.STOP, g, tu, nh)
    net = g[0] - cost * tu[0]
    return net, tu[0], nh[0]

def holdings_path(m, stack, param):
    """Re-run in python to list picks at each rebalance (same selection rule)."""
    sid, N, wi, buf, ri, ex, src = param
    reb = m.reb[G.REBS[ri]]; out = []
    w = np.zeros(len(m.companies)); sel = np.empty(len(m.companies), np.int64); tw = np.zeros_like(w)
    for t in range(len(m.days)):
        r = m.dec_row[t]
        if r >= 0 and reb[t]:
            k = E._select(stack[sid][r], w, N, buf, sel)
            picks = [m.tickers[j] for j in sel[:k]]
            out.append((m.days[t], m.dec_pdf[r], picks))
            w[:] = 0; w[sel[:k]] = 1.0
    return out

if __name__ == "__main__":
    d = D.load(); m = E.build_market(d); names, stack = E.build_scores(m)
    param = [names.index(("T", "all", "RADJ120")), 1, 0, 0, 3, 0, 0]
    net, tu, nh = simulate(m, stack, names, param)
    krw = (1 + net) * (1 + m.fxr) - 1
    eq = np.cumprod(1 + krw)
    print("total", round((eq[-1] - 1) * 100, 1))
    for day, pdf, picks in holdings_path(m, stack, param):
        t = m.days.index(day)
        print(day, "pdf", pdf, picks)
    # monthly returns
    for mo in sorted(set(x[:7] for x in m.days)):
        idx = [i for i, x in enumerate(m.days) if x[:7] == mo]
        print(mo, round((np.prod(1 + krw[idx]) - 1) * 100, 1))
