import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "turn_B")))
import numpy as np, pandas as pd
import compare_turn as CT
f, etf = CT.load()
C = f["Close"]; dates = list(C.index); syms = list(C.columns)
stock = np.array([s not in etf for s in syms])
F = CT.features(C)
Z = CT.five(F, F["T3"][20], 0.5)
rm = CT.pmask(dates, CT.P_LY)
print("n days LY", rm.sum(), "stocks", stock.sum())
print("turnB flips", CT.flips(Z, stock, rm), "direct", CT.direct(Z, stock, rm))
z = Z[:, stock].astype(np.int16)
zt, zp = z[1:], z[:-1]
ok = (zt != -9) & (zp != -9) & rm[1:, None]
ch = (ok & (zt != zp))
vd = ((z != -9) & rm[:, None]).sum(0)
# per stock mean
rate = ch.sum(0) / (vd / 252)
print("per-stock mean", np.nanmean(rate[vd > 0]), "median", np.nanmedian(rate[vd>0]))
print("pairs denom", ch.sum() / (ok.sum() / 252))
print("count per stock / 1 year (all stocks)", ch.sum() / stock.sum())
print("count / stocks with full year", ch.sum()/ (vd.sum()/rm.sum()))
print("n stocks partial", (vd < rm.sum()).sum(), vd[vd<rm.sum()])
# denominators by 250/251 etc
print("count/stock-years using days", ch.sum()/(vd.sum()/rm.sum()))
# direct per-stock
def direct_ps(Z, colmask, rm, gap=3):
    z = Z[:, colmask].astype(np.int16)
    nD, nS = z.shape
    out = []
    for j in range(nS):
        cnt = 0
        last_s, last_i = 0, -10**9
        for i in range(nD):
            v = z[i, j]
            if v == -9:
                last_s, last_i = 0, -10**9; continue
            sg = int(np.sign(v))
            if sg == 0: continue
            if last_s != 0 and sg == -last_s and i - last_i <= gap and rm[i]: cnt += 1
            last_s, last_i = sg, i
        vd = ((z[:, j] != -9) & rm).sum()
        if vd: out.append(cnt / (vd/252))
    return np.mean(out)
print("direct per-stock", direct_ps(Z, stock, rm))
# steady with regression slope
lC = F["lC"]
tt = pd.DataFrame(np.tile(np.arange(len(C), dtype=float)[:, None], (1, C.shape[1])), index=C.index, columns=C.columns)
cov = lC.rolling(60, min_periods=60).cov(tt)
vt = tt.rolling(60, min_periods=60).var()
slope = cov / vt
L, SW = CT.labels(F)
for per in [CT.P_ALL, CT.P_LY, CT.P_26]:
    rmp = CT.pmask(dates, per)
    for useslope in (True, False):
        r2 = F["r2"].to_numpy()[:, stock]
        cond_s = (slope*60).to_numpy()[:, stock] >= np.log(1.15) if useslope else F["ret60"].to_numpy()[:, stock] >= 0.15
        lab = L["Z10"][:, stock]
        zz = Z[:, stock]
        sel = (r2 >= 0.8) & cond_s & (lab == 1) & (zz != -9) & rmp[:, None]
        print(per, "slope" if useslope else "ret60", sel.sum(), (zz[sel] > 0).mean())
