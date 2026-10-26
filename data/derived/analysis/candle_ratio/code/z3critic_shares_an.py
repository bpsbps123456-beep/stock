import numpy as np, pandas as pd
from engine import HERE
g = pd.read_parquet(HERE / "results" / "z3critic_shares_rows.parquet")
for w in ("sel", "conf"):
    g[f"ex_{w}"] = g[f"ret_{w}"] - g[f"static_{w}"]; g[f"mex_{w}"] = g[f"mdd_{w}"] - g[f"static_mdd_{w}"]
med = g.groupby(["acc", "k", "sid"])[["ret_sel", "ex_sel", "mex_sel", "ret_conf", "mdd_conf", "static_conf", "static_mdd_conf", "ex_conf", "mex_conf"]].median().reset_index()
med["action"] = med.sid.str.split("|").str[1]
def q(a, z): z = np.asarray(z); return ((z < a).sum() + 0.5 * (z == a).sum()) / len(z)
for acc in ("frac", "shares"):
    m = med[med.acc == acc]
    m0 = m[m.k == 0].set_index("sid")
    c = m0.drop(index="BASE|A3")
    top = c.ex_sel.idxmax(); topm = c.mex_sel.idxmax()
    print(f"\n[{acc}] 고르기 1위 {top}  보조 1위 {topm}")
    print(c.sort_values("ex_sel", ascending=False).head(5)[["ex_sel", "ex_conf", "mex_conf", "ret_conf", "mdd_conf"]].round(4).to_string())
    b = m0.loc["BASE|A3"]
    for nm in (top, "ER_K20_th0.3|A2"):
        r = m0.loc[nm]
        # 귀무: k마다 같은 고르기
        nulls = []
        for k, mk in m[m.k != 0].groupby("k"):
            mk = mk.set_index("sid"); ck = mk.drop(index="BASE|A3")
            sid = ck.ex_sel.idxmax() if nm == top else nm
            nulls.append(mk.loc[sid].ex_conf)
        print(f"  {nm}: 확인 {r.ret_conf*100:.2f}/{r.mdd_conf*100:.2f} 같은비중 {r.static_conf*100:.2f}/{r.static_mdd_conf*100:.2f} ex {r.ex_conf*100:.2f}%p q {q(r.ex_conf, nulls):.2f} p95 {np.quantile(nulls,0.95)*100:.2f} | 원안 {b.ret_conf*100:.2f}/{b.mdd_conf*100:.2f}")
    # 행동별
    for a in ("A0", "A1", "A2"):
        act = m[m.action == a].groupby("k")[["ex_sel", "ex_conf", "mex_conf"]].median()
        a0 = act.loc[0]; an = act.drop(index=0)
        print(f"  행동 {a}: ex_sel {a0.ex_sel*100:.2f} ex_conf {a0.ex_conf*100:.2f} (q {q(a0.ex_conf, an.ex_conf):.2f}) mex_conf {a0.mex_conf*100:.2f} (q {q(a0.mex_conf, an.mex_conf):.2f})")
    # 기준2 (a)(b) 동시 만족, 구간 실제 쓰는 전략
    ok = c[(c.ret_conf > b.ret_conf) & (c.mdd_conf > b.mdd_conf) & (c.ret_conf >= c.static_conf) & (c.mdd_conf >= c.static_mdd_conf)]
    print("  기준2 (a)(b) 만족:", list(ok.index))
