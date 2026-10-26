"""dip_scalper.pine 지지·저항 구역 계산을 그대로 옮겨, TEM 10/6 매수 시점(09:57 ET)에 보였을 구역을 다시 그린다.
피벗 좌우 5봉, 0.5% 안은 한 구역, 구역 사이 1.2% 미만이면 약한 쪽 제외, 최근 스윙 60개."""
import sys
import numpy as np, pandas as pd
R = sys.argv[1]
TF = int(sys.argv[2]) if len(sys.argv) > 2 else 40
m = pd.read_parquet(f"{R}/1m_TEM.parquet")
b = m.resample(f"{TF}min", origin=pd.Timestamp("2026-09-09 04:00", tz="America/New_York"), label="left", closed="left").agg(
    {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
N, KEEP, TOL, GAP, CNT = 5, 60, 0.005, 0.012, 3
h, l, t = b.High.values, b.Low.values, b.index

def zones_at(T, close):
    last = np.searchsorted(t, T, side="right") - 1          # T에 진행 중인 봉
    px, bi = [], []
    for i in range(N, last - N + 1):                         # 확정된 피벗만(오른쪽 5봉이 끝나야 앎)
        if i + N >= last:                                    # 진행 중인 봉은 아직 닫히지 않음
            break
        lo_win, hi_win = l[i - N:i + N + 1], h[i - N:i + N + 1]
        if l[i] == lo_win.min() and (l[i - N:i] > l[i]).all():
            px.append(l[i]); bi.append(i)
        if h[i] == hi_win.max() and (h[i - N:i] < h[i]).all():
            px.append(h[i]); bi.append(i)
    px, bi = px[-KEEP:], bi[-KEEP:]
    order = np.argsort(px, kind="stable")
    Z = []                                                   # [lo, hi, cnt, last_bar]
    for k in order:
        p, q = px[k], bi[k]
        if Z and (p - Z[-1][0]) / Z[-1][0] <= TOL:
            Z[-1][1] = p; Z[-1][2] += 1; Z[-1][3] = max(Z[-1][3], q)
        else:
            Z.append([p, p, 1, q])
    keep = []
    for z in sorted(Z, key=lambda z: z[2] * 1e7 + z[3], reverse=True):
        mid = (z[0] + z[1]) / 2
        if all(abs(mid - (k[0] + k[1]) / 2) / ((k[0] + k[1]) / 2) >= GAP for k in keep):
            keep.append(z)
    keep.sort(key=lambda z: (z[0] + z[1]) / 2)
    sup = [z for z in keep if (z[0] + z[1]) / 2 < close]
    res = [z for z in keep if (z[0] + z[1]) / 2 >= close]
    return sup, res, keep

T = pd.Timestamp("2026-10-06 09:57", tz="America/New_York")
sup, res, allz = zones_at(T, 83.30)
day = m.loc["2026-10-06 09:57":"2026-10-06 19:59"]
print(f"TEM {TF}분봉, 10/6 09:57 ET(22:57 KST) 매수 83.30 시점 구역 — 전체 {len(allz)}개")
print("화면에 그려지는 지지(가까운 3개) / 그 아래 구역까지:")
for z in sorted(sup, key=lambda z: -(z[0] + z[1]) / 2):
    lo, hi, cnt, q = z
    brk = day[day.Low < lo * 0.999]
    bt = brk.index[0] if len(brk) else None
    after = day.loc[bt:] if bt is not None else None
    # 뚫린 뒤 그 구역 위로 다시 올라온 적 있나(종가 기준)
    back = (after.Close > hi).any() if after is not None else None
    print(f"  {'지지' if z in sup[-CNT:] else '(화면 밖)'} {lo:.2f}~{hi:.2f} ×{cnt}  마지막 스윙 {t[q].tz_convert('Asia/Seoul'):%m/%d %H:%M}KST"
          f"  → 뚫림 {bt.tz_convert('Asia/Seoul'):%H:%M}KST" if bt is not None else "  → 안 뚫림", f" 다시 위로 {back}" if bt is not None else "")
print("저항(가까운 3개):", [f"{z[0]:.2f}~{z[1]:.2f}×{z[2]}" for z in res[:CNT]])
print(f"그날 저가 {day.Low.min():.2f} ({day.Low.idxmin().tz_convert('Asia/Seoul'):%H:%M}KST), 종가(정규) {m.loc['2026-10-06 15:59'].Close:.2f}")
