"""verify_tf_recompute.py 보조(검증용, 2026-10-11): 60·120분 h5 극상방의 날짜 수준 Newey-West t(5일 겹침, 시차 5)와
'모든 종목' 기준선(상승확률·평균 대비 뒤짐 비율·초과 중앙·>+10% 비율), 이상 거래일 영향.
결과: results/verify_tf_sig.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_tf_recompute as V  # noqa: E402

OUT = os.path.join(V.A, "trend_zones/results/verify_tf_sig.txt")


def nw_t(x, lag=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan, n
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n), n


def main():
    bars = V.load_hourly()
    for tf in (60, 120):
        W, last = V.tf_panel(bars, tf)
        day = np.array(W["C"].index.get_level_values("day"))
        D = V.defs(W)
        f, exday, ent = V.forward(bars, last, 5)
        rk = V.crank(f)
        fdf = pd.DataFrame(f)
        excu = fdf.sub(fdf.mean(axis=1), axis=0).to_numpy()
        valid = (W["C"].notna().cumsum() >= V.WARM).to_numpy() & W["C"].notna().to_numpy()
        V.p(f"\n== {tf}분 h5")
        for p_, (a, z) in V.PER.items():
            s = (day >= a) & (day <= z)
            M = valid[s] & np.isfinite(f[s])
            n = M.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                up = np.where(n > 0, (M & (f[s] > 0)).sum(1) / n, np.nan)
                lag = np.where(n > 0, (M & (excu[s] < 0)).sum(1) / n, np.nan)
            pool = excu[s][M]
            V.p(f"  기준선(모든 종목) {p_}: 상승확률 {V.daymean(up, day[s]).mean() * 100:.0f}% · 평균 대비 뒤짐 {V.daymean(lag, day[s]).mean() * 100:.0f}% · "
                f"초과 중앙 {np.median(pool) * 100:+.2f}% · >+10% {np.mean(pool > 0.10) * 100:.1f}%")
        # 날짜 수준 시계열과 NW t
        comp = {p_: [] for p_ in V.PER}
        want = ("RSI14 70/30", "RSI3 90/10", "볼린저50 ±1.5", "칼만식 EMA20·atr14 ±2.0", "칼만식 EMA50·rms60 ±1.5", "5일 수익 z ±2.0")
        for nm, t, b in D:
            cells = []
            for p_, (a, z) in V.PER.items():
                s = (day >= a) & (day <= z)
                if p_ == "앞":
                    s &= np.array([e <= z for e in exday])
                fin = np.isfinite(rk[s])
                Mr = t[s] & fin
                sh = Mr.sum() / max(fin.sum(), 1)
                n = Mr.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    m = np.where(n > 0, np.where(Mr, rk[s], 0).sum(1) / n, np.nan)
                dm = -V.daymean(m, day[s])
                if 0.01 <= sh <= 0.20:
                    comp[p_].append(dm.rename(nm))
                if nm in want:
                    tt, nn = nw_t(dm.to_numpy())
                    cells.append(f"{p_} {dm.mean() * 100:+.2f} t{tt:+.1f} (날{nn})")
            if nm in want:
                V.p(f"  {nm:<24} 위 h5 " + " | ".join(cells))
        cells = []
        for p_ in V.PER:
            c = pd.concat(comp[p_], axis=1).mean(axis=1)
            tt, nn = nw_t(c.to_numpy())
            cells.append(f"{p_} {c.mean() * 100:+.2f} t{tt:+.1f} (정의{len(comp[p_])})")
        V.p("  자격 정의 평균(합성)       위 h5 " + " | ".join(cells))
        # 자격 정의끼리 상관(날짜 수준) — 독립 증거가 아님을 보이기
        c = pd.concat(comp["2025"], axis=1)
        cm = c.corr().to_numpy()
        V.p(f"  2025 자격 정의 {c.shape[1]}개 날짜 수준 상관 중앙 {np.nanmedian(cm[np.triu_indices_from(cm, 1)]):.2f}")
    # 이상 거래일: 원본 Base 칸 번호 청산이 시계와 어긋나는 신호
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tf_extremes as T
    B = T.load_hour()
    O, Hh, Lo, C, end, days = T.resample(B, 1)
    ent = end + 1
    okk = ent < len(B.day)
    ent = ent[okk]
    ex = B.exit_index(ent, 5)
    good = ex >= 0
    t0 = B.O.index[ent[good]].strftime("%H:%M")
    t1 = B.O.index[ex[good]].strftime("%H:%M")
    bad = np.flatnonzero(np.array(t0) != np.array(t1))
    V.p(f"\n원본 60분 h5: 청산 시각 ≠ 진입 시각인 신호 행 {len(bad)} / {good.sum()} (반일장·2026-01-30·02-02 칸 어긋남), "
        f"예: {[str(B.O.index[ent[good][i]])[:16] + '→' + str(B.O.index[ex[good][i]])[:16] for i in bad[:3]]}")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(V.L))


if __name__ == "__main__":
    main()
