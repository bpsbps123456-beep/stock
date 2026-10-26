"""z3 comp 갈래 3단계: z3comp_analyze.py 결과를 사용자 말로 옮길 핵심 숫자로 줄인다 → results/z3comp_summary.json"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import HERE  # noqa: E402

RES = HERE / "results"
A = pd.read_csv(RES / "z3comp_agree.csv")
T = pd.read_csv(RES / "z3comp_tests.csv")
C = pd.read_csv(RES / "z3comp_cells.csv")
RB = pd.read_csv(RES / "z3comp_robust27.csv")
META = json.loads((RES / "z3comp_meta.json").read_text(encoding="utf-8"))
REPS = ["ER_K20_th0.3", "FLIP_K40_m3", "EXT_n20_d0.1", "HTF_tfD", "S_only"]


def r(x, k=2):
    return None if pd.isna(x) else round(float(x), k)


out = {"meta": META, "agree": {}, "z3_in_z5": {}, "z5_in_z3": {}, "combo": {}, "robust27": {}}
for c in REPS:
    a = A[(A.clf == c) & (A.seg == "both")].iloc[0]
    out["agree"][c] = dict(kappa=r(a.kappa, 3), kappa_null_p95=r(a.kappa_null_p95, 3), same_dir=r(a.same_dir, 3), same_dir_chance=r(a.same_dir_exp, 3),
                           up_given_plus=r(a.p_up_given_p, 3), base_up=r(a.base_up, 3), dn_given_minus=r(a.p_dn_given_m, 3), base_dn=r(a.base_dn, 3),
                           mid_given_zero=r(a.p_mid_given_z, 3), base_mid=r(a.base_mid, 3), share_z3_zero=r(a.base_z, 3),
                           run_bars=r(META["run_length_bars_median"][c], 1))


def pick(clf, sub, h, ms, seg, test, con):
    x = T[(T.clf == clf) & (T.subset == sub) & (T.h == h) & (T.measure == ms) & (T.seg == seg) & (T.test == test) & (T.contrast == con)]
    return x.iloc[0] if len(x) else None


for c in REPS:
    for sub in ("all", "aw_dst", "aw_std"):
        for ms in ("rank", "xs", "raw"):
            for h in (1, 5):
                for seg in ("sel", "conf"):
                    key = f"{c}|{sub}|{ms}|h{h}|{seg}"
                    t1, t0 = pick(c, sub, h, ms, seg, "T1", "p_m"), pick(c, sub, h, ms, seg, "T0", "p_m")
                    t2, t2m = pick(c, sub, h, ms, seg, "T2", "u_d"), pick(c, sub, h, ms, seg, "T2m", "u_d")
                    out["z3_in_z5"][key] = dict(est=r(t1.est), t=r(t1.t), null_pct=r(t1.L_pct), flagonly_pct=r(t1.get("F_pct")), alone=r(t0.est), alone_null_pct=r(t0.L_pct))
                    out["z5_in_z3"][key] = dict(est=r(t2.est), t=r(t2.t), null_pct=r(t2.Z_pct), alone=r(t2m.est), alone_null_pct=r(t2m.Z_pct))

# 9칸(5단계 3묶음 × 3단계) 빈도·평균
C["z5c"] = C.z5.clip(-1, 1)
C["sum"] = C["mean"].fillna(0) * C.n
g9 = C.groupby(["clf", "subset", "h", "measure", "seg", "z5c", "z3"]).agg(n=("n", "sum"), s=("sum", "sum")).reset_index()
g9["mean"] = g9.s / g9.n
g9["share"] = g9.n / g9.groupby(["clf", "subset", "h", "measure", "seg"]).n.transform("sum")
names = {(1, 1): "둘 다 상승", (-1, -1): "둘 다 하락", (1, -1): "엇갈림(5상방·3하락)", (-1, 1): "엇갈림(5하방·3상승)", (0, 0): "둘 다 애매"}
for c in REPS:
    for sub in ("all", "aw_dst"):
        for h in (1, 5):
            for seg in ("sel", "conf"):
                d = {}
                for (a, b), nm in names.items():
                    row = {}
                    for ms in ("rank", "raw", "xs"):
                        x = g9[(g9.clf == c) & (g9.subset == sub) & (g9.h == h) & (g9.measure == ms) & (g9.seg == seg) & (g9.z5c == a) & (g9.z3 == b)]
                        if len(x):
                            row[ms] = r(x["mean"].iloc[0])
                            row["share"] = r(x.share.iloc[0], 3)
                    d[nm] = row
                for con in ("bothup_vs_up", "bothup_vs_up_m", "bothdn_vs_dn", "bothdn_vs_dn_p", "bothup_vs_bothdn"):
                    t = pick(c, sub, h, "rank", seg, "COMBO", con)
                    d[con] = dict(est=r(t.est), t=r(t.t), null_pct=r(t.L_pct))
                out["combo"][f"{c}|{sub}|h{h}|{seg}"] = d

rb = RB[RB.test == "T1"]
for (sub, ms, h), d in rb.groupby(["subset", "measure", "h"]):
    a = d.pivot(index="clf", columns="seg", values="est")
    p = d.pivot(index="clf", columns="seg", values="L_pct")
    out["robust27"][f"{sub}|{ms}|h{h}"] = dict(n=int(len(a)), sel_pos=int((a.sel > 0).sum()), conf_pos=int((a.conf > 0).sum()),
                                              both_pos=int(((a.sel > 0) & (a.conf > 0)).sum()), both_neg=int(((a.sel < 0) & (a.conf < 0)).sum()),
                                              sel_ge95=int((p.sel >= 0.95).sum()), sel_le05=int((p.sel <= 0.05).sum()),
                                              conf_ge95=int((p.conf >= 0.95).sum()), conf_le05=int((p.conf <= 0.05).sum()),
                                              both_ge95=int(((p.sel >= 0.95) & (p.conf >= 0.95)).sum()))
(RES / "z3comp_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# 화면용
print("일치율(전체 구간, 144종목 합)")
for c, a in out["agree"].items():
    print(f"  {c:14s} 카파 {a['kappa']:.3f}(귀무95 {a['kappa_null_p95']:.3f}) 같은 방향 {a['same_dir']:.1%}(우연 {a['same_dir_chance']:.1%}) "
          f"+1→5상방 {a['up_given_plus']:.1%}(평소 {a['base_up']:.1%}) −1→5하방 {a['dn_given_minus']:.1%}({a['base_dn']:.1%}) "
          f"0→5애매 {a['mid_given_zero']}({a['base_mid']:.1%}) 3단계 애매 비중 {a['share_z3_zero']:.1%} 상태 지속 {a['run_bars']}봉")
print("5단계 안에서 3단계 +1 − −1 (순위 %p, t, 귀무 분위) / 3단계 혼자")
for c in REPS:
    for sub in ("all", "aw_dst"):
        s = []
        for h in (1, 5):
            for seg in ("sel", "conf"):
                v = out["z3_in_z5"][f"{c}|{sub}|rank|h{h}|{seg}"]
                s.append(f"h{h} {seg} {v['est']:+.2f}(t {v['t']:+.1f}, {v['null_pct']:.2f}) 혼자 {v['alone']:+.2f}")
        print(f"  {c:14s} {sub:6s} " + " | ".join(s))
print("3단계 안에서 5단계 상방 − 하방 (순위 %p)")
for sub in ("all", "aw_dst"):
    s = []
    for h in (1, 5):
        for seg in ("sel", "conf"):
            v = out["z5_in_z3"][f"ER_K20_th0.3|{sub}|rank|h{h}|{seg}"]
            s.append(f"h{h} {seg} {v['est']:+.2f}(t {v['t']:+.1f}, {v['null_pct']:.3f}) 혼자 {v['alone']:+.2f}")
    print(f"  {sub:6s} " + " | ".join(s))
print("같이 띄우기(ER 기본값, all)")
for h in (1, 5):
    for seg in ("sel", "conf"):
        d = out["combo"][f"ER_K20_th0.3|all|h{h}|{seg}"]
        print(f"  h{h} {seg}: " + " / ".join(f"{k} {v.get('share', 0):.1%} 순위 {v.get('rank', np.nan):+.2f} 날것 {v.get('raw', np.nan):+.2f}%" for k, v in d.items() if "share" in v))
        print("        " + " / ".join(f"{k} {v['est']:+.2f}(t {v['t']:+.1f}, {v['null_pct']:.2f})" for k, v in d.items() if "est" in v))
print("27개 판정 전체")
for k, v in out["robust27"].items():
    print(" ", k, v)
