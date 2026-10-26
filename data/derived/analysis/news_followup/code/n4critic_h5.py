"""N4 반박 검산(h5): 'R 사전 -> 매수 확인 V자 실패'가 두 폭락 바닥(2025-04-24, 2026-04-07) 사건과 시기 편중에 기대는지.
py code/n4critic_h5.py  (읽기만: results/n4h5_soxx_events.csv, cache/n3_raw.parquet, ../swing_trend/results/events_ZZA3.parquet)
"""
import pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "code"))
import n4h5_run as H

ev = pd.read_csv(HERE / "results" / "n4h5_soxx_events.csv", parse_dates=["conf_date", "pivot_date"])
raw = pd.read_parquet(HERE / "cache" / "n3_raw.parquet")
z = H.zz_runs(raw, "SOXX"); dates = z["dates"]; C = z["C"]
nt = H.note_table(dates); R = H.note_sets(nt)[("R", "all")].kb.to_numpy()
i0 = int(dates.searchsorted(H.START)); i1 = int(dates.searchsorted(H.END, side="right")) - 1; S = i1 - i0 + 1

e = ev[ev.det == "ZZA3"].reset_index(drop=True)
# 직전 하락 다리 깊이: 앞 사건(매도 확인)의 극점(고점) -> 이 사건 극점(저점)
piv = [int(dates.searchsorted(d)) for d in e.pivot_date]
e["prior_leg"] = [np.nan] + [float(np.log(C[piv[i]] / C[piv[i - 1]])) for i in range(1, len(e))]
b = e[e.type == 1].copy()
lab = b.R_all_pre.to_numpy(bool); y = b.y1.to_numpy()
def show(tag, m):
    l, yy = lab[m], y[m]
    if l.sum() == 0 or (~l).sum() == 0:
        print(tag, "한쪽 0"); return
    u = mannwhitneyu(yy[l], yy[~l], alternative="less")
    print(f"{tag:42s} n1={l.sum()} n0={(~l).sum()} m1={yy[l].mean():+.3f} m0={yy[~l].mean():+.3f} diff={yy[l].mean()-yy[~l].mean():+.3f} "
          f"med1={np.median(yy[l]):+.3f} med0={np.median(yy[~l]):+.3f} MW p={u.pvalue:.3f}")
print(b[["conf_date", "prior_leg", "y1", "R_all_pre", "D_all_pre"]].to_string())
allm = np.ones(len(b), bool)
show("전체", allm)
crash = b.conf_date.isin(pd.to_datetime(["2025-04-24", "2026-04-07"])).to_numpy()
show("두 폭락 바닥 제외", ~crash)
late = (b.conf_date >= "2024-07-01").to_numpy()
show("2024-07 이후(R 사전 8건 모두 이 구간)", late)
show("2024-07 이후, 두 바닥 제외", late & ~crash)
show("직전 하락 다리 > -15%(얕은 조정만)", (b.prior_leg > -0.15).to_numpy())
print("R 사전 여부별 직전 하락 다리:", b.groupby("R_all_pre").prior_leg.describe().round(3).to_string())

# 원형 이동 귀무를 두 바닥 제외 표본에 다시
m = ~crash
cidx = b.conf_idx.to_numpy()[m]
NL = H.shifted_labels(cidx, R, -5, 0, i0, S)
nd = H.null_diffs(y[m], NL); obs = y[m][lab[m]].mean() - y[m][~lab[m]].mean()
print(f"두 바닥 제외 원형 이동 하단 p = {np.mean(nd[np.isfinite(nd)] <= obs):.3f} (obs {obs:+.3f}, K={np.isfinite(nd).sum()})")
m2 = late
NL = H.shifted_labels(b.conf_idx.to_numpy()[m2], R, -5, 0, i0, S)
nd = H.null_diffs(y[m2], NL); obs = y[m2][lab[m2]].mean() - y[m2][~lab[m2]].mean()
print(f"2024-07 이후 원형 이동 하단 p = {np.mean(nd[np.isfinite(nd)] <= obs):.3f} (obs {obs:+.3f})")
m3 = late & ~crash
NL = H.shifted_labels(b.conf_idx.to_numpy()[m3], R, -5, 0, i0, S)
nd = H.null_diffs(y[m3], NL); obs = y[m3][lab[m3]].mean() - y[m3][~lab[m3]].mean()
print(f"2024-07 이후·두 바닥 제외 원형 이동 하단 p = {np.mean(nd[np.isfinite(nd)] <= obs):.3f} (obs {obs:+.3f})")

# 주식: 두 SOXX 바닥 전후 창을 뺀 매수 확인 y20 차이
st = pd.read_parquet(H.SWINGR / "events_ZZA3.parquet"); st["date"] = pd.to_datetime(st.date)
st["sidx"] = dates.searchsorted(st.date)
bs = st[st.type == 1].reset_index(drop=True)
for per in ("ALL", "Y2026", "Y25"):
    sub = bs if per == "ALL" else (bs[bs.period.isin(["P1", "P2"])] if per == "Y2026" else bs[bs.period == "Y25"])
    sub = sub.reset_index(drop=True)
    U = np.unique(sub.sidx); di = np.searchsorted(U, sub.sidx)
    LU = H.counts(U, R, -5, 0) > 0
    for W in (0, 20, 40):
        bad = np.zeros(len(U), bool)
        for d in ("2025-04-08", "2026-03-30"):   # 두 바닥 극점
            p = int(dates.searchsorted(pd.Timestamp(d)))
            bad |= (U >= p) & (U <= p + W) if W else False
        keepU = ~bad
        k = keepU[di]
        yy = sub.y20.to_numpy(float)[k]; ll = LU[di][k]; dk = di[k]
        ok = ~np.isnan(yy)
        diff = yy[ok & ll].mean() - yy[ok & ~ll].mean()
        # 날짜 회전 귀무(남은 날짜)
        Uk = np.flatnonzero(keepU); remap = -np.ones(len(U), int); remap[Uk] = np.arange(len(Uk)); dd = remap[dk]
        L2 = LU[Uk]; D = len(Uk)
        sy = np.bincount(dd[ok], weights=yy[ok], minlength=D); cn = np.bincount(dd[ok], minlength=D).astype(float)
        nds = []
        for kk in range(1, D):
            Lr = np.roll(L2, kk).astype(float)
            s1, c1 = Lr @ sy, Lr @ cn
            nds.append(s1 / c1 - (sy.sum() - s1) / (cn.sum() - c1))
        nds = np.array(nds)
        print(f"주식 매수 y20 {per:5s} 바닥 극점 뒤 {W:2d}세션 제외: 날짜 {D} (R {int(L2.sum())}), diff={diff:+.4f}, 하단 p={np.mean(nds<=diff):.3f}")
