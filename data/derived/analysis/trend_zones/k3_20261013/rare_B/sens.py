"""민감도(보조): R1 m=1.55(보정 동률), R3·R4 p=12.5(C=1/2 백분위 관례에서 보정값). 결과: sens.json"""
import json, os, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rare_compare as RC
CT = RC.CT
f, etf = CT.load()
C = f["Close"]; dates = list(C.index); syms = list(C.columns); nD, nS = C.shape
stock = np.array([s not in etf for s in syms]); etfm = ~stock
F = CT.features(C)
mx20 = C.rolling(20, min_periods=20).max(); mn20 = C.rolling(20, min_periods=20).min()
dd20 = np.log(C / mx20).to_numpy(); ru20 = np.log(C / mn20).to_numpy()
u252 = F["lr"].rolling(252, min_periods=252).std(ddof=1).to_numpy() * np.sqrt(10)
Z0 = CT.five(F, F["T3"][20], 0.5)
rm_ly = CT.pmask(dates, CT.P_LY); rm_all = CT.pmask(dates, CT.P_ALL)
lC = F["lC"]
tt = pd.DataFrame(np.tile(np.arange(nD, dtype=float)[:, None], (1, nS)), index=C.index, columns=C.columns)
slope60 = (lC.rolling(60, min_periods=60).cov(tt) / tt.rolling(60, min_periods=60).var() * 60).to_numpy()
r2 = F["r2"].to_numpy()
L, _ = CT.labels(F)
rms = {k: CT.pmask(dates, v) for k, v in CT.PER.items()}
X4 = {}
rare = {}
for ev, (a, b) in RC.HIST.items():
    inwin = np.array([a <= d <= b for d in dates])
    X4[ev] = np.array([RC.x4_for(dd20[:, j], inwin)[0] for j in range(nS)])
for ev, (a, b) in RC.EVAL.items():
    em = np.array([a <= d <= b for d in dates])
    with np.errstate(invalid="ignore"):
        rare[ev] = em[:, None] & np.isfinite(X4[ev])[None, :] & (dd20 <= np.log(1 - X4[ev])[None, :])
with np.errstate(invalid="ignore"):
    cdn155 = dd20 <= -1.55 * u252
    qdn = np.column_stack([RC.roll_pct(dd20[:, j], [12.5])[:, 0] for j in range(nS)])
    qup = np.column_stack([RC.roll_pct(ru20[:, j], [87.5])[:, 0] for j in range(nS)])
    cdn3 = (qdn < 0) & (dd20 <= qdn); cup3 = (qup > 0) & (ru20 >= qup)
Zs = {"R1_m1.55": RC.override(Z0, cdn155), "R3_p12.5": RC.override(Z0, cdn3),
      "R4_p12.5": RC.override(Z0, cdn3, cup3, -dd20, ru20)}
out = {}
for n, Z in Zs.items():
    ao = {th: {h: CT.agree_opp(Z, L[th], stock, rms[h]) for h in ("2026H1", "2026H2")} for th in CT.THETAS}
    out[n] = {"cover26": RC.cover(Z, rare["2026"], stock)["down"], "cover25": RC.cover(Z, rare["2025"], stock)["down"],
              "etf_cover26": RC.cover(Z, rare["2026"], etfm)["down"],
              "flips": RC.flips_ps(Z, stock, rm_ly), "direct": RC.direct_ps(Z, stock, rm_ly),
              "steady_up": RC.steady_up(Z, r2, slope60, L["Z10"], stock, rm_all)[0],
              "score": float(np.mean([ao[th][h]["agree"] - 2 * ao[th][h]["opp"] for th in CT.THETAS for h in ("2026H1", "2026H2")]))}
    print(n, {k: round(v, 4) for k, v in out[n].items()})
json.dump(out, open(os.path.join(HERE, "sens.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
