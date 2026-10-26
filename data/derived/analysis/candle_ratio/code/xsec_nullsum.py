"""xsec 갈래: 실제 vs 원형 이동 귀무 비교표.

py code/xsec_nullsum.py → results/xsec_null_compare.json
"""
import json

from xsec_analyze import HERE

RES = HERE / "results"


def row(sub):
    k = json.loads((RES / f"xsec_key_{sub}.json").read_text(encoding="utf-8"))
    f = json.loads((RES / f"xsec_followup_{sub}.json").read_text(encoding="utf-8"))
    a = k["all"]
    return {
        "top1_beat_bh": round(a["all"]["top1_beat_bh"], 3), "top1_xpos": round(a["all"]["top1_xpos"], 3),
        "top1_xconf_med%": round(a["all"]["top1_xconf_med%"], 2),
        "rth_top1_beat_bh": round(a["rth"]["top1_beat_bh"], 3), "rth_top1_xpos": round(a["rth"]["top1_xpos"], 3),
        "cells_beat_conf_med": round(a["all"]["cells_beat_conf_med"], 3),
        "rho_med": round(a["all"]["rho_med"], 3),
        "x_fade_sel_med%": round(a["x_fade_sel_med%"], 2), "x_fade_conf_med%": round(a["x_fade_conf_med%"], 2),
        "x_follow_sel_med%": round(a["x_follow_sel_med%"], 2), "x_follow_conf_med%": round(a["x_follow_conf_med%"], 2),
        "region_gross": f["gross"]["region_n"], "region_net": f["net"]["region_n"],
        "region_gross_rth_fade_rate": f["gross"]["region_rate"].get("sess_dir", {}).get("rth_fade"),
        "fam_rth_fade_tf<=15_rate": f["gross"]["family_rth_fade_tf<=15"]["region_rate"],
        "fwd_top20_gross_xc%": f["gross"]["forward"]["top20"]["xc_med%"],
        "fwd_top20_net_xc%": f["net"]["forward"]["top20"]["xc_med%"],
        "fwd_top100_gross_xc%": f["gross"]["forward"]["top100"]["xc_med%"],
        "net_top1_beat_bh": f["net"]["per_stock_top1"]["all"]["beat_bh"], "net_top1_xpos": f["net"]["per_stock_top1"]["all"]["xpos"],
    }


if __name__ == "__main__":
    out = {}
    for p in sorted(RES.glob("xsec_followup_*.json")):
        sub = p.stem[len("xsec_followup_"):]
        if (RES / f"xsec_key_{sub}.json").exists():
            out[sub] = row(sub)
    (RES / "xsec_null_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = list(next(iter(out.values())).keys())
    print("항목".ljust(28) + "".join(s.ljust(13) for s in out))
    for k in keys:
        print(k.ljust(28) + "".join(str(out[s][k]).ljust(13) for s in out))
