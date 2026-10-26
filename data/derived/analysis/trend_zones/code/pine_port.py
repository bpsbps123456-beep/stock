"""trend_5zone.pine 계산을 파이썬으로 그대로 옮겨 연구 계산(zones.py 사전 등록 1위)과 대조한다(2026-10-10).

파인 쪽 차이: ta.rsi는 첫 14개 평균으로 시작하는 RMA, ta.stdev(…, false)는 표본 표준편차, bar_index < 129는 계산 안 함.
결과: 날짜×종목 단계 일치율, 그리고 사용자가 트레이딩뷰에서 대조할 10/9 종가 기준 값(RSI·60일 z·단계).
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/pine_port.py
"""
import os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402

CHECK = ["SNDK", "MU", "NVDA", "AMD", "INTC", "SNOW", "STX", "DELL", "PLTR", "TSLA", "ASTS", "TEM"]


def pine_rma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    start = None
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        if start is None:
            start = i
        k = i - start
        if k == n - 1:
            out[i] = np.mean(x[start:i + 1])
        elif k >= n:
            out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def pine_zone(c: np.ndarray, L=60, th=0.5, rl=14, hi=70, lo=30):
    lr = np.r_[np.nan, np.log(c[1:] / c[:-1])]
    sd = pd.Series(lr).rolling(L, min_periods=L).std().to_numpy()
    z = np.log(c / np.r_[np.full(L, np.nan), c[:-L]]) / (sd * np.sqrt(L))
    d = np.r_[np.nan, np.diff(c)]
    up, dn = pine_rma(np.where(np.isfinite(d), np.maximum(d, 0), np.nan), rl), pine_rma(np.where(np.isfinite(d), np.maximum(-d, 0), np.nan), rl)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    r = np.where(np.isfinite(up) & np.isfinite(dn), r, np.nan)
    zn = np.select([r >= hi, r <= lo, z >= th, z <= -th], [2, -2, 1, -1], 0).astype(float)
    ok = np.isfinite(z) & np.isfinite(r) & (np.arange(len(c)) >= 129)
    return np.where(ok, zn, np.nan), r, z


def main():
    I = Zm.Ind(*Zm.load())
    res = __import__("json").load(open(os.path.join(Zm.OUT, "zones.json"), encoding="utf-8"))
    best = next(g["cfg"] for g in res["grid"] if g["name"] == res["best"])
    Zr = Zm.zones(I, best).astype(float)
    Zr[Zr == -9] = np.nan
    agree, n = 0, 0
    rows = [i for i, d in enumerate(I.dates) if d >= "2024-01-02"]
    out = []
    for j, s in enumerate(I.syms):
        col = I.C[s]
        first = col.first_valid_index()
        c = col.loc[first:].to_numpy(float)
        zp, r, z = pine_zone(c)
        off = I.dates.index(first)
        zp_full = np.full(len(I.dates), np.nan)
        zp_full[off:] = zp
        both = np.isfinite(zp_full[rows]) & np.isfinite(Zr[rows, j])
        agree += int((zp_full[rows][both] == Zr[rows, j][both]).sum())
        n += int(both.sum())
        if s in CHECK:
            out.append((s, I.dates[-1], c[-1], r[-1], z[-1], Zm.NAMES.get(int(zp[-1])) if np.isfinite(zp[-1]) else "—",
                        Zm.NAMES.get(int(Zr[-1, j])) if np.isfinite(Zr[-1, j]) else "—"))
    print(f"2024-01-02 이후 날짜×종목 {n}개 중 파인 이식과 연구 단계 일치 {agree / n * 100:.2f}%")
    print("트레이딩뷰 대조값(정규장 일봉 종가 기준):")
    for s, d, c, r, z, zp, zr in out:
        print(f"  {s:<5} {d} 종가 {c:,.2f}  RSI14 {r:5.1f}  60일 z {z:+.2f}  → 파인 {zp} / 연구 {zr}")


if __name__ == "__main__":
    main()
