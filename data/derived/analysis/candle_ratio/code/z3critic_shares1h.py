"""z3critic: 1시간봉 창 k=0에서 절반 보유 회계(frac 대 주식 수 고정)에 따른 고르기 1위 변화."""
from multiprocessing import Pool
import numpy as np, pandas as pd
from engine import WIN
from zones3 import prepare, sim_frac, static_path, targets
from z3critic_shares import STRATS, sim_shares, SYMS
from z3strat_run import eth_days

def work(sym):
    P = prepare(sym, "1h", "dst")
    O, C, seg, aw, s = P["O"], P["C"], P["seg"], P["awake"], P["s"]
    rows = []; po = np.zeros(len(O))
    for nm, f, ai in STRATS:
        if f in ("HTF_tf160", "HTF_tf320"):
            continue
        t = targets(s, P["flags"][f] if f else np.zeros(len(O), np.bool_), ai)
        for acc in ("frac", "shares"):
            o = sim_frac(O, C, t, seg, aw, 0.001, po) if acc == "frac" else sim_shares(O, C, t, seg, aw, 0.001)
            x = o[2:4] / o[4:6]
            r = dict(sym=sym, sid=f"{nm}|{'A3' if ai == 3 else 'A%d' % ai}", acc=acc)
            for w, kk in (("sel", 0), ("conf", 1)):
                st, stm, _, _ = static_path(O, C, seg, x[kk], kk)
                r[f"ex_{w}"] = float(np.expm1(o[kk])) - st; r[f"ret_{w}"] = float(np.expm1(o[kk])); r[f"static_{w}"] = st
            rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    syms = [s for s in SYMS if min(eth_days(s, "1h")) <= WIN["1h"]["sel"][0]]
    with Pool(3) as pool:
        g = pd.concat(pool.map(work, syms, chunksize=4))
    m = g.groupby(["acc", "sid"])[["ex_sel", "ex_conf", "ret_conf", "static_conf"]].median()
    for acc in ("frac", "shares"):
        c = m.loc[acc].drop(index="BASE|A3").sort_values("ex_sel", ascending=False)
        print(acc, len(syms)); print((c.head(4) * 100).round(2).to_string())
