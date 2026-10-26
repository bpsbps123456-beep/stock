"""z3critic: 절반 보유를 '주식 수 고정(비중 표류), 목표가 바뀔 때만 깨어 있는 봉에서 거래'로 다시 잰다.
zones3.sim_frac(매 봉 비용 없이 비중 재조정, 잘 때도)과 비교해 z3strat의 고르기 1위·행동별 귀무 분위가 바뀌는지 본다.
실제 판(깨어 있음 서머타임 + 편도 0.1%)만. 귀무는 z3strat_run.shifter와 같은 달력 이동.
"""
from __future__ import annotations
from multiprocessing import Pool
import numpy as np, pandas as pd
from numba import njit
from engine import CACHE, HERE, WIN
from zones3 import CLASSIFIERS, cname, prepare, sim_frac, static_path, targets
from z3strat_run import calendar, shifter, shifted_flags, GUARD

SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
STRATS = [("BASE", None, 3)] + [(cname(k, p), cname(k, p), ai) for k, p in CLASSIFIERS for ai in range(3)]

@njit(cache=True)
def sim_shares(o, c, tgt, seg, awake, cost):
    n = len(o)
    out = np.zeros(8)  # [0:2] 로그수익, [2:4] 보유 합, [4:6] 봉 수, [6:8] 낙폭
    w = 0.0; exe = 0.0; target = 0.0
    eq = np.zeros(2); pk = np.zeros(2)
    for j in range(n):
        r = 0.0
        if awake[j] and target != exe:
            d = abs(target - w)
            if d > 0:
                r += np.log(1.0 - cost * d)
            w = target; exe = target
        nxt = o[j + 1] if j + 1 < n else c[j]
        R = nxt / o[j] - 1.0
        r += np.log(1.0 + w * R)
        sg = seg[j]
        if sg >= 0:
            out[sg] += r; out[2 + sg] += w; out[4 + sg] += 1
            eq[sg] += r; pk[sg] = max(pk[sg], eq[sg]); out[6 + sg] = min(out[6 + sg], eq[sg] - pk[sg])
        w = w * (1.0 + R) / (1.0 + w * R)
        target = tgt[j]
    return out

def work(args):
    sym, cal, ks = args
    P = prepare(sym, "5m", "dst")
    O, C, seg, aw, s = P["O"], P["C"], P["seg"], P["awake"], P["s"]
    src = shifter(P["tb"], cal, "5m")
    rows = []
    po = np.zeros(len(O))
    for k in [0] + ks:
        fl = P["flags"] if k == 0 else shifted_flags(P["flags"], src(k)[0])
        for nm, f, ai in STRATS:
            flag = fl[f] if f else np.zeros(len(O), np.bool_)
            t = targets(s, flag, ai)
            for acc in ("frac", "shares"):
                if acc == "frac":
                    o = sim_frac(O, C, t, seg, aw, 0.001, po); ret = o[0:2]; x = o[2:4] / o[4:6]; mdd = o[6:8]
                else:
                    o = sim_shares(O, C, t, seg, aw, 0.001); ret = o[0:2]; x = o[2:4] / o[4:6]; mdd = o[6:8]
                r = dict(sym=sym, k=k, sid=f"{nm}|{'A3' if ai == 3 else 'A%d' % ai}", acc=acc)
                for w, kk in (("sel", 0), ("conf", 1)):
                    st, stm, _, _ = static_path(O, C, seg, x[kk], kk)
                    r[f"ret_{w}"] = float(np.expm1(ret[kk])); r[f"mdd_{w}"] = float(np.expm1(mdd[kk]))
                    r[f"static_{w}"] = st; r[f"static_mdd_{w}"] = stm
                rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    cal = calendar("5m", SYMS)
    D = len(cal); ks = list(range(GUARD, D - GUARD + 1))
    with Pool(3) as pool:
        g = pd.concat(pool.map(work, [(s, cal, ks) for s in SYMS], chunksize=4), ignore_index=True)
    g.to_parquet(HERE / "results" / "z3critic_shares_rows.parquet")
    print("D", D, "ks", ks[0], ks[-1], len(ks), "rows", len(g))
