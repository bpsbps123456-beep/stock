"""이웃 평균(L·a·b ±1칸) 선택 구간 샤프로 강건한 설정을 고르고 확인 구간 성적을 본다."""
import numpy as np, pandas as pd
pd.set_option("display.width", 260)
LS = [5, 8, 13, 21, 34, 55, 89]; AS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]; BS = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

def robust(g, key="sel_sh"):
    g = g.copy()
    g["li"] = g.L.map({v: i for i, v in enumerate(LS)}); g["ai"] = g.a.map({v: i for i, v in enumerate(AS)})
    g["bi"] = g.b.map({v: i for i, v in enumerate(BS)})
    idx = {(r.li, r.ai, r.bi): getattr(r, key) for r in g.itertuples()}
    sc = []
    for r in g.itertuples():
        vals = [idx[(r.li + dl, r.ai + da, r.bi + db)] for dl in (-1, 0, 1) for da in (-1, 0, 1) for db in (-1, 0, 1)
                if (r.li + dl, r.ai + da, r.bi + db) in idx]
        sc.append(np.mean(vals) if len(vals) >= 6 else np.nan)
    g["rob"] = sc
    return g

def table(name, rules=(True, True)):
    df = pd.read_parquet(f"{name}.parquet")
    bh = df[df.kind == "buyhold"].iloc[0]; og = df[df.kind == "original"]
    z = df[(df.kind == "zone5") & (df.tp_half == rules[0]) & (df.panic_hold == rules[1])]
    out = []
    for (tf, eth), g in z.groupby(["tf", "eth"]):
        best = None
        for (norm, n), gg in g.groupby(["norm", "n"]):
            gg = robust(gg)
            c = gg.sort_values("rob", ascending=False).iloc[0]
            if best is None or c.rob > best.rob:
                best = c
        o = og[(og.tf == tf) & (og.eth == eth)].iloc[0]
        out.append({"tf": int(tf), "ses": "ETH" if eth else "RTH", "cfg": f"L{best.L:.0f} {best.norm}{best.n:.0f} a{best.a} b{best.b}",
                    "rob_selsh": best.rob, "sel_sh": best.sel_sh, "sel_ret": best.sel_ret, "val_ret": best.val_ret,
                    "val_sh": best.val_sh, "val_mdd": best.val_mdd, "all_ret": best.all_ret, "all_mdd": best.all_mdd,
                    "trades": int(best.trades), "expo": best.expo,
                    "orig_all": o.all_ret, "orig_val": o.val_ret, "orig_valmdd": o.val_mdd})
    t = pd.DataFrame(out)
    pct = [c for c in t.columns if c.endswith(("_ret", "_mdd", "orig_all", "orig_val", "expo"))]
    print(f"\n===== {name} 규칙 절반익절={rules[0]} 패닉보유={rules[1]} | 보유: 선택 {bh.sel_ret:+.1%} 확인 {bh.val_ret:+.1%} (샤프 {bh.val_sh:.2f}, 낙폭 {bh.val_mdd:.1%}) 전체 {bh.all_ret:+.1%} 낙폭 {bh.all_mdd:.1%}")
    print(t.to_string(index=False, formatters={**{c: "{:+.1%}".format for c in pct}, **{c: "{:.2f}".format for c in ["rob_selsh", "sel_sh", "val_sh"]}}))
    return t

tl = table("grid_long2"); ts = table("grid_short2")
tl.to_csv("pick_long.csv", index=False); ts.to_csv("pick_short.csv", index=False)
