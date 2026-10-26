"""trend_5zone.pine 10/11 기본값(추세 20일)의 트레이딩뷰 대조값. pine_port.pine_zone(파인과 같은 계산)을 L=20으로 돌린다.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/pine_check20.py"""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
from pine_port import pine_zone, CHECK  # noqa: E402

NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}


def main():
    O, H, L, C = Zm.load()
    dates = list(C.index)
    print("== 10/9 종가 기준(추세 20일)")
    for s in CHECK:
        c = C[s].dropna()
        zp, r, z = pine_zone(c.to_numpy(float), L=20)
        print(f"  {s:<5} 종가 {c.iloc[-1]:9,.2f}  RSI14 {r[-1]:5.1f}  20일 z {z[-1]:+.2f}  → {NAMES.get(int(zp[-1]), '—') if np.isfinite(zp[-1]) else '—'}")
    c = C["MU"].dropna()
    zp, r, z = pine_zone(c.to_numpy(float), L=20)
    idx = {d: i for i, d in enumerate(c.index)}
    print("== MU 하락 구간(추세 20일)")
    for d in ("2026-06-26", "2026-07-01", "2026-07-15", "2026-07-20", "2026-07-28", "2026-07-29", "2026-08-12", "2026-09-22"):
        i = idx[d]
        print(f"  {d} 종가 {c.iloc[i]:8.2f}  RSI14 {r[i]:5.1f}  20일 z {z[i]:+.2f}  → {NAMES.get(int(zp[i]), '—')}")


if __name__ == "__main__":
    main()
