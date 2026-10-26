import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for name in ["grid_short", "grid_long"]:
    df = pd.read_parquet(f"{name}.parquet")
    bh = df[df.kind == "buyhold"].iloc[0]
    print(f"\n===== {name}  보유(B&H): 전체 {bh.all_ret:+.1%} 샤프 {bh.all_sh:.2f} 낙폭 {bh.all_mdd:.1%} | 선택 {bh.sel_ret:+.1%} ({bh.sel_sh:.2f}) | 확인 {bh.val_ret:+.1%} ({bh.val_sh:.2f}) 낙폭 {bh.val_mdd:.1%}")
    og = df[df.kind == "original"]
    z = df[df.kind == "zone5"]
    rows = []
    for (tf, eth), g in z.groupby(["tf", "eth"]):
        o = og[(og.tf == tf) & (og.eth == eth)].iloc[0]
        best = g.sort_values("sel_sh", ascending=False).iloc[0]
        top = g.sort_values("sel_sh", ascending=False).head(max(10, len(g) // 100))
        rows.append({"tf": tf, "ses": "ETH" if eth else "RTH", "n": len(g),
                     "orig_sel": o.sel_ret, "orig_val": o.val_ret, "orig_valsh": o.val_sh,
                     "best_sel": best.sel_ret, "best_selsh": best.sel_sh, "best_val": best.val_ret, "best_valsh": best.val_sh,
                     "best_valmdd": best.val_mdd, "top1%_val_med": top.val_ret.median(), "top1%_valsh_med": top.val_sh.median(),
                     "all_val_med": g.val_ret.median(), "cfg": f"L{best.L:.0f} {best.norm}{best.n:.0f} a{best.a} b{best.b} half{int(best.tp_half)} ph{int(best.panic_hold)}"})
    t = pd.DataFrame(rows)
    fmt = {c: "{:+.1%}".format for c in t.columns if c.endswith(("_sel", "_val", "val_med", "valmdd")) }
    print(t.to_string(index=False, formatters={**fmt, **{c: "{:.2f}".format for c in t.columns if "sh" in c}}))
