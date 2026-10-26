"""N4 반박 검산(h5b): 주식 매수 확인 R 사전 효과가 같은 날 SOXX 20일 움직임(시장)으로 설명되는지, R 사전 날짜가 몇 개 일화로 묶이는지.
py code/n4critic_h5b.py (읽기만)
"""
import pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "code"))
import n4h5_run as H
raw = pd.read_parquet(HERE / "cache" / "n3_raw.parquet")
z = H.zz_runs(raw, "SOXX"); dates = z["dates"]; O = z["O"]
nt = H.note_table(dates); R = H.note_sets(nt)[("R", "all")].kb.to_numpy()
st = pd.read_parquet(H.SWINGR / "events_ZZA3.parquet"); st["date"] = pd.to_datetime(st.date)
st["sidx"] = dates.searchsorted(st.date)
n = len(O)
st["mkt20"] = [np.log(O[i + 21] / O[i + 1]) if i + 21 < n else np.nan for i in st.sidx]
bs = st[st.type == 1].reset_index(drop=True)
def rot_p(sub, col):
    U = np.unique(sub.sidx); di = np.searchsorted(U, sub.sidx); LU = H.counts(U, R, -5, 0) > 0
    y = sub[col].to_numpy(float); ok = ~np.isnan(y); l = LU[di]
    diff = y[ok & l].mean() - y[ok & ~l].mean()
    D = len(U); sy = np.bincount(di[ok], weights=y[ok], minlength=D); cn = np.bincount(di[ok], minlength=D).astype(float)
    nds = np.array([(np.roll(LU, k).astype(float) @ sy) / (np.roll(LU, k).astype(float) @ cn)
                    - (sy.sum() - np.roll(LU, k).astype(float) @ sy) / (cn.sum() - np.roll(LU, k).astype(float) @ cn) for k in range(1, D)])
    # R 사전 날짜 일화: 연속한 표지 날짜 사이 간격 > 10세션이면 새 일화
    Ul = U[LU]; ep = 1 + int((np.diff(Ul) > 10).sum()) if len(Ul) else 0
    Un = U[~LU]; ep0 = 1 + int((np.diff(Un) > 10).sum()) if len(Un) else 0
    return diff, float(np.mean(nds <= diff)), ep, ep0
for per in ("ALL", "Y2026", "P1", "P2", "Y25", "Y24"):
    sub = bs if per == "ALL" else (bs[bs.period.isin(["P1", "P2"])] if per == "Y2026" else bs[bs.period == per])
    sub = sub.dropna(subset=["y20", "mkt20"]).copy()
    sub["ex20"] = sub.y20 - sub.mkt20
    d1, p1, ep, ep0 = rot_p(sub, "y20"); d2, p2, _, _ = rot_p(sub, "mkt20"); d3, p3, _, _ = rot_p(sub, "ex20")
    print(f"{per:5s} 사건 {len(sub):4d}  y20 diff {d1:+.4f}(p {p1:.3f}) | 같은 날 SOXX 20일 diff {d2:+.4f}(p {p2:.3f}) | 시장 뺀 초과 diff {d3:+.4f}(p {p3:.3f}) | R 사전 일화 {ep}, 없음 일화 {ep0}")
