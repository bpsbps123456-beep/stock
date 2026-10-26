"""mtfaudit 4: 예열 영향. 7/17 전 1시간봉 예열을 빼고(5분봉 7/17부터만) 같은 335개 전략 × 판 3개를 다시 재서 mtf_rows(k=0)와 비교.

- zones_mtf.WARM1H를 START5로 바꾸면 1시간봉 예열이 비고(bars_with_warm의 h1이 빈 표) 나머지는 그대로다. 일봉 판정은 일봉 자체 이력(2025-10~)이라 바뀌지 않는다.
- 같은 패치를 독립 재계산(mtfaudit_core.build(warm=False))과 MU·BE·NBIS 몇 칸에서 대조한다.

py code/mtfaudit_warm.py [프로세스 수 ≤ 3] → results/mtfaudit_warm_rows.parquet, results/mtfaudit_warm_summary.csv, results/mtfaudit_warm.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402
import zones_mtf as Z  # noqa: E402

OUT = engine.HERE / "results"
BOARDS = {"real": (0.001, False), "real0": (0.0, False), "ideal": (0.0, True)}
KEY = ["sym", "b", "clf", "action", "board"]


def init():
    Z.WARM1H = Z.START5


def work(sym):
    init()
    rows = []
    for b in Z.BASE_TFS:
        P = Z.prepare(sym, b)
        assert not P["tb"].warm.any() and P["tb"].day.min() >= Z.START5
        for bd, (cost, ideal) in BOARDS.items():
            for r in Z.run(P, cost, ideal):
                r.update(board=bd, sym=sym)
                rows.append(r)
    return pd.DataFrame(rows)


def reimpl_check():
    import mtfaudit_core as M
    init()
    cells = [(40, "BASE", "A3"), (40, "z160_AGREE", "A0"), (40, "z320_ER", "A0"), (240, "z960_AGREE", "A2"), (10, "z80_ADX", "A0")]
    worst = 0.0
    n = 0
    for sym in ("MU", "BE", "NBIS"):
        raw = M.read_raw(sym)
        for b, clf, act in cells:
            P = M.build(raw, b, warm=False)
            flag = P["flags"][clf] if clf != "BASE" else np.zeros(len(P["O"]), bool)
            mine, _ = M.metrics(P, M.target(P["s"], flag, act), 0.001, False)
            Pz = Z.prepare(sym, b)
            ref = [r for r in Z.run(Pz, 0.001, False) if r["clf"] == clf and r["action"] == act][0]
            for c in ("ret_sel", "ret_conf", "mdd_sel", "mdd_conf", "x_sel", "x_conf"):
                worst = max(worst, abs(mine[c] - ref[c]))
                n += 1
    return dict(values=n, max_abs_diff=worst)


def med(df, col):
    return df.groupby(["b", "clf", "action"], observed=True)[col].median()


def main():
    nproc = min(3, int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    syms = sorted(p.stem[:-3] for p in engine.CACHE.glob("*_5m.parquet"))
    with Pool(nproc) as pool:
        nw = pd.concat(pool.map(work, syms), ignore_index=True)
    for c in ("sym", "clf", "action", "board"):
        nw[c] = nw[c].astype(str)
    nw.to_parquet(OUT / "mtfaudit_warm_rows.parquet")
    w = pd.read_parquet(OUT / "mtf_rows.parquet", filters=[("k", "==", 0)])
    for c in ("sym", "clf", "action", "board"):
        w[c] = w[c].astype(str)
    m = w.merge(nw, on=KEY, suffixes=("", "_nw"))
    assert len(m) == len(w) == len(nw), (len(m), len(w), len(nw))
    for s in ("", "_nw"):
        for wd in ("sel", "conf"):
            m[f"ex_{wd}{s}"] = m[f"ret_{wd}{s}"] - m[f"static_{wd}{s}"]
            m[f"dd_{wd}{s}"] = m[f"mdd_{wd}{s}"] - m[f"static_mdd_{wd}{s}"]
    out = {}
    summ = []
    for bd in BOARDS:
        mb = m[m.board == bd]
        g = mb.groupby(["b", "clf", "action"])
        t = pd.DataFrame({c: g[c].median() for c in ("ex_sel", "ex_sel_nw", "ex_conf", "ex_conf_nw", "dd_conf", "dd_conf_nw",
                                                      "ret_conf", "ret_conf_nw")}).reset_index()
        t["board"] = bd
        t["d_sel"] = t.ex_sel_nw - t.ex_sel
        t["d_conf"] = t.ex_conf_nw - t.ex_conf
        summ.append(t)
        per = dict(abs_d_sel_median=float(t.d_sel.abs().median()), abs_d_sel_max=float(t.d_sel.abs().max()),
                   abs_d_conf_median=float(t.d_conf.abs().median()), abs_d_conf_max=float(t.d_conf.abs().max()),
                   sym_abs_d_ret_conf_p99=float((mb.ret_conf_nw - mb.ret_conf).abs().quantile(0.99)),
                   sym_abs_d_ret_conf_max=float((mb.ret_conf_nw - mb.ret_conf).abs().max()),
                   sym_abs_d_ret_sel_median=float((mb.ret_sel_nw - mb.ret_sel).abs().median()),
                   sym_abs_d_ret_sel_max=float((mb.ret_sel_nw - mb.ret_sel).abs().max()))
        for lab, sub in (("all335", t), ("zones326", t[t.clf != "BASE"])):
            a = sub.loc[sub.ex_sel.idxmax()]
            b_ = sub.loc[sub.ex_sel_nw.idxmax()]
            per[f"top_{lab}_warm"] = dict(b=int(a.b), clf=a.clf, action=a.action, sel=float(a.ex_sel), conf=float(a.ex_conf))
            per[f"top_{lab}_nowarm"] = dict(b=int(b_.b), clf=b_.clf, action=b_.action, sel=float(b_.ex_sel_nw), conf=float(b_.ex_conf_nw))
            rk = sub.ex_sel.rank(ascending=False)
            rk2 = sub.ex_sel_nw.rank(ascending=False)
            per[f"rank_spearman_sel_{lab}"] = float(np.corrcoef(rk, rk2)[0, 1])
            per[f"top10_overlap_{lab}"] = int(len(set(sub.nlargest(10, "ex_sel").index) & set(sub.nlargest(10, "ex_sel_nw").index)))
        out[bd] = per
    S = pd.concat(summ, ignore_index=True)
    S.to_csv(OUT / "mtfaudit_warm_summary.csv", index=False)
    pick = S[(S.board == "real") & (((S.clf == "BASE")) | ((S.b == 40) & S.clf.isin(["z160_AGREE", "z320_ER"]) & (S.action == "A0")))]
    out["named_real"] = pick[["b", "clf", "action", "ex_sel", "ex_sel_nw", "ex_conf", "ex_conf_nw"]].round(4).to_dict("records")
    out["reimpl_nowarm_check"] = reimpl_check()
    json.dump(out, open(OUT / "mtfaudit_warm.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
