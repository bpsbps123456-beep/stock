"""robust 갈래 분해 요약: robust_decomp.npy·robust_holds.csv·robust_sleep.csv·robust_realloss.csv → 표.

몫은 로그 기준 %p(더하면 맞는다). 덜 잃은 몫 L = S − B = 현금 (x−1)B + 쏠림 Σ(x_c−x)B_c + 시간대 안 타이밍 Σ(S_c − x_c B_c).
노출 맞춘 초과 E = S − xB = 쏠림 + 타이밍.
py code/robust_decomp_summary.py → results/robust_decomp_summary.csv, robust_decomp_summary.json (+ 화면 표)
"""
import json

import numpy as np
import pandas as pd

from robust_common import RES

meta = json.loads((RES / "robust_decomp_meta.json").read_text(encoding="utf-8"))
A = np.load(RES / "robust_decomp.npy")  # [sym, k, mode, win, F]
F = {f: i for i, f in enumerate(meta["feat"])}
syms = meta["syms"]
COMPS = ("reg", "pre", "post", "closed")


def derive(a):
    """a[..., F] → 몫 dict (로그 %p)."""
    g = lambda f: a[..., F[f]]
    x = g("x")
    d = dict(S=g("S"), B=g("B"), x=x, L=g("L"), E=g("E"), cash=(x - 1) * g("B"),
             E_rm3=g("E_rm3"), L_rm3=g("L_rm3"), E_exearn=g("E_exearn"), L_exearn=g("L_exearn"), cost=g("cost"),
             S_rm3s=g("S_rm3s"))
    for c in COMPS:
        d[f"tilt_{c}"] = (g(f"x_{c}") - x) * g(f"B_{c}")
        d[f"timing_{c}"] = g(f"S_{c}") - g(f"x_{c}") * g(f"B_{c}")
        d[f"E_{c}"] = g(f"S_{c}") - x * g(f"B_{c}")
        d[f"S_{c}"] = g(f"S_{c}"); d[f"B_{c}"] = g(f"B_{c}"); d[f"x_{c}"] = g(f"x_{c}")
    d["E_night"] = d["E_pre"] + d["E_post"] + d["E_closed"]
    d["S_night"] = d["S_pre"] + d["S_post"] + d["S_closed"]
    d["B_night"] = d["B_pre"] + d["B_post"] + d["B_closed"]
    d["tilt_night"] = d["tilt_pre"] + d["tilt_post"] + d["tilt_closed"]
    d["timing_night"] = d["timing_pre"] + d["timing_post"] + d["timing_closed"]
    return d


KEYS = ["S", "B", "L", "cash", "E", "E_reg", "tilt_reg", "timing_reg", "E_night", "tilt_night", "timing_night",
        "E_pre", "E_post", "E_closed", "S_reg", "B_reg", "S_night", "B_night", "x", "x_reg", "x_pre", "x_post", "x_closed",
        "E_rm3", "L_rm3", "E_exearn", "L_exearn", "cost", "S_rm3s"]
rows = []
for mi, mode in enumerate(meta["modes"]):
    for wi, win in enumerate(meta["windows"]):
        d = derive(A[:, :, mi, wi])  # [sym, k]
        for k in KEYS:
            v = d[k]
            sc = 1.0 if k.startswith("x") else 100.0
            r = dict(mode=mode, window=win, key=k, **{s: float(v[syms.index(s), 0] * sc) for s in ("MU", "BE", "NBIS")})
            med = np.nanmedian(v, axis=0) * sc
            r["med144"] = float(med[0])
            r["pos144"] = float(np.nanmean(v[:, 0] > 0)) if not k.startswith("x") else np.nan
            r["null_med"] = float(np.median(med[1:]))
            r["null_pct"] = float((med[1:] < med[0]).mean())
            rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(RES / "robust_decomp_summary.csv", index=False)
pd.set_option("display.width", 220)
for mode in ("ideal", "real"):
    for win in ("full", "conf", "sel"):
        print(f"\n### 분해 {mode} {win} (로그 %p, x는 비율) — MU BE NBIS | 144 중앙 (+ 종목) | 귀무 중앙 [분위]")
        x = S[(S["mode"] == mode) & (S.window == win)]
        for _, r in x.iterrows():
            print(f"| {r.key} | {r.MU:+.1f} | {r.BE:+.1f} | {r.NBIS:+.1f} | {r.med144:+.2f} ({'' if np.isnan(r.pos144) else round(r.pos144 * 100)}) | "
                  f"{r.null_med:+.2f} [{r.null_pct * 100:.0f}] |")

# 보유 구간·자는 동안
H = pd.read_csv(RES / "robust_holds.csv")
H = H[H.seg >= 0]
SL = pd.read_csv(RES / "robust_sleep.csv")
out = {}
print("\n### 보유 구간 길이 (창 안 진입, 시가→시가 달력 시간)")
for mode in ("ideal", "real"):
    for grp, h in (("144전체", H[H["mode"] == mode]), *[(s, H[(H["mode"] == mode) & (H.sym == s)]) for s in ("MU", "BE", "NBIS")]):
        q = h.hours.quantile([0.25, 0.5, 0.75, 0.9]).round(1).tolist()
        o = dict(n=len(h), n_per_sym=round(len(h) / max(h.sym.nunique(), 1), 1), q25_50_75_90_h=q,
                 bars_med=float(h.bars.median()), over_night=round(float((h.nights >= 1).mean()), 3),
                 under_2h=round(float((h.hours < 2).mean()), 3), over_24h=round(float((h.hours >= 24).mean()), 3))
        out[f"{mode}_{grp}"] = o
        print(mode, grp, o)
print("\n### 자는 동안(00~08 KST = 11:00~19:00 ET) 보유")
for mode in ("ideal", "real"):
    s = SL[SL["mode"] == mode]
    o = dict(held_share_in_sleep_med=round(s.held_share_in_sleep.median(), 3),
             sleep_share_of_session=round(s.sleep_share_of_session.median(), 3),
             held_while_asleep_med=round(s.held_while_asleep.median(), 3),
             held_while_awake_med=round(s.held_while_awake.median(), 3),
             named={r.sym: [round(r.held_share_in_sleep, 3), round(r.held_while_asleep, 3), round(r.held_while_awake, 3)]
                    for r in s[s.sym.isin(["MU", "BE", "NBIS"])].itertuples()})
    out[f"sleep_{mode}"] = o
    print(mode, o)
print("\n### 실제 − 이상적 손해 출처 (로그 %p)")
LO = pd.read_csv(RES / "robust_realloss.csv")
for win in ("full", "conf"):
    l = LO[LO.window == win].copy()
    l["total"] = l[["late_sell", "kept_through", "late_buy", "missed_trip", "cost"]].sum(axis=1)
    cols = ["late_sell", "kept_through", "late_buy", "missed_trip", "cost", "total"]
    o = {"med144": {c: round(float(l[c].median() * 100), 2) for c in cols},
         "pos144": {c: round(float((l[c] > 0).mean()), 2) for c in cols},
         "mean144": {c: round(float(l[c].mean() * 100), 2) for c in cols},
         "n_med": {c: float(l["n_" + c].median()) for c in cols[:4]}}
    for s in ("MU", "BE", "NBIS"):
        r = l[l.sym == s].iloc[0]
        o[s] = {c: round(float(r[c] * 100), 1) for c in cols} | {"n_" + c: int(r["n_" + c]) for c in cols[:4]}
    out[f"realloss_{win}"] = o
    print(win, json.dumps(o, ensure_ascii=False))
TOP = pd.read_csv(RES / "robust_topdays.csv")
print("\n### 가장 큰 하루 3개 (전체 창, E = 노출 맞춘 초과, L = 보유 대비)")
print(TOP.assign(val=(TOP.val * 100).round(1)).to_string(index=False))
out["earn"] = {e["sym"]: e for e in meta["earn"] if e["sym"] in ("MU", "BE", "NBIS", "AEHR")}
out["earn_src_counts"] = pd.Series([e["earn_src"] for e in meta["earn"]]).value_counts().to_dict()
out["n_earn_days_med"] = float(np.median(A[:, 0, 0, 0, F["n_earn_days"]]))
(RES / "robust_decomp_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
