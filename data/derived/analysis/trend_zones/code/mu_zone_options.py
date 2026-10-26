"""MU 6/26~7/28 하락인데 띠가 '상방'인 이유 + 더 빨리 반응하는 상방·하방 정의 비교(2026-10-11, 연구용).

사용자(10/11, 트레이딩뷰 MU 일봉 캡처): "MU가 6/26 1,200 부근에서 7/28로 갈 때도 왜 연빨강(상방)이야?"
지금 정의: 극상방 RSI14 ≥ 70, 극하방 ≤ 30, 그 밖에는 60거래일 흐름 z = ln(C/C[60]) / (일 로그수익 표준편차 60 × √60) ≥ 0.5 상방, ≤ -0.5 하방.
비교(극단은 RSI 70/30 그대로, 가운데 세 칸만 바꿈):
  A 지금: 60일 흐름 z ±0.5
  B 20일 흐름 z ±0.5
  C 20일선−60일선 거리(ATR 배수) ±1.0 (연구 격자 cross(20,60))
  D 20일선 5일 기울기(ATR/일) ±0.1 (연구 격자 slope(20,5))
  E 사용자 차트 이평선 배열: 종가 > EMA25 > EMA60 상방, 종가 < EMA25 < EMA60 하방, 나머지 애매 (격자 밖, 사후)
보고: MU 6/15~10/9 날짜별 단계(띠가 어떻게 보이는지), 구간별 단계 일수, 단계가 바뀐 횟수(깜빡임), 그리고 150종목 연구 점수
     (zones.py와 같은 S·T1, 앞/뒤/2025, h10) — 결과를 본 뒤의 비교라 고르기 근거가 아니라 '손해가 없는지' 확인용.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/mu_zone_options.py
"""
import os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402

NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}
SHORT = {2: "▲", 1: "+", 0: "·", -1: "-", -2: "▼"}


def middle(I: Zm.Ind, kind):
    C = I.C
    if kind == "A":
        t = I.trend("mz", 60); return np.select([t >= 0.5, t <= -0.5], [1, -1], 0), t
    if kind == "B":
        t = I.trend("mz", 20); return np.select([t >= 0.5, t <= -0.5], [1, -1], 0), t
    if kind in ("B30", "B40"):
        t = I.trend("mz", int(kind[1:])); return np.select([t >= 0.5, t <= -0.5], [1, -1], 0), t
    if kind == "C":
        t = I.trend("cross", (20, 60)); return np.select([t >= 1.0, t <= -1.0], [1, -1], 0), t
    if kind == "D":
        t = I.trend("slope", (20, 5)); return np.select([t >= 0.1, t <= -0.1], [1, -1], 0), t
    if kind == "E":
        e25, e60 = I.ema(25), I.ema(60)
        up = (C > e25) & (e25 > e60)
        dn = (C < e25) & (e25 < e60)
        return np.select([up, dn], [1, -1], 0), (C / e25 - 1)
    raise ValueError(kind)


def zone_of(I, kind):
    m, t = middle(I, kind)
    r = I.rsi.to_numpy()
    z = np.select([r >= 70, r <= 30], [2, -2], m)
    ok = I.valid.to_numpy() & np.isfinite(np.asarray(t, float)) & np.isfinite(r)
    return np.where(ok, z, -9).astype(np.int8)


def main():
    Zm.MIN_BARS = 130
    I = Zm.Ind(*Zm.load())
    F = Zm.Fwd(I, 10)
    j = I.syms.index("MU")
    dates = I.dates
    sel = [i for i, d in enumerate(dates) if "2026-06-15" <= d <= "2026-10-09"]
    kinds = {"A 지금(60일 흐름)": "A", "B 20일 흐름": "B", "C 20·60일선 거리": "C", "D 20일선 기울기": "D",
             "E 종가·EMA25·EMA60 배열": "E", "F 30일 흐름": "B30", "G 40일 흐름": "B40"}
    Z = {k: zone_of(I, v) for k, v in kinds.items()}
    lines = ["== MU 날짜별 단계(▲극상방 +상방 ·애매 -하방 ▼극하방), 6/15~10/9", "날짜       종가    RSI14  60일z  20일z | " + " ".join(k[0] for k in kinds)]
    C = I.C["MU"].to_numpy(); r = I.rsi["MU"].to_numpy()
    z60 = I.trend("mz", 60)["MU"].to_numpy(); z20 = I.trend("mz", 20)["MU"].to_numpy()
    for i in sel:
        lines.append(f"{dates[i]} {C[i]:8.2f} {r[i]:6.1f} {z60[i]:+6.2f} {z20[i]:+6.2f} | "
                     + " ".join(SHORT.get(int(Z[k][i, j]), "?") for k in kinds))
    lines += ["", "== MU 구간별 단계 일수 · 바뀐 횟수"]
    for lab, a, b in (("6/26~7/28 하락", "2026-06-26", "2026-07-28"), ("7/29~10/9", "2026-07-29", "2026-10-09")):
        idx = [i for i in sel if a <= dates[i] <= b]
        for k in kinds:
            v = [int(Z[k][i, j]) for i in idx]
            cnt = " ".join(f"{NAMES[z]} {v.count(z)}" for z in (2, 1, 0, -1, -2) if v.count(z))
            flips = sum(1 for x, y in zip(v, v[1:]) if x != y)
            lines.append(f"  {lab:<14} {k:<22} {cnt}  (바뀜 {flips}번)")
    lines += ["", "== 150종목 연구 점수(h10, 날짜 평균, %p) — S / T1(상방−하방), 앞·뒤·2025 · 종목당 1년 평균 단계 바뀜 횟수"]
    for k, Zk in Z.items():
        cells = []
        for p in ("앞", "뒤", "2025"):
            e = Zm.evaluate(Zk, F, Zm.period_rows(I, p, 10))
            cells.append(f"{p} S {e['S'] * 100:+5.2f} T1 {e['terms']['T1 상방-하방'] * 100:+5.2f}")
        yr = [i for i, d in enumerate(dates) if "2025-10-01" <= d <= "2026-10-09"]
        Zy = Zk[yr]
        flips = np.mean([np.sum((Zy[1:, c] != Zy[:-1, c]) & (Zy[1:, c] != -9) & (Zy[:-1, c] != -9)) for c in range(Zy.shape[1])])
        lines.append(f"  {k:<22} " + " | ".join(cells) + f" | 1년 바뀜 {flips:5.1f}번")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "mu_zone_options.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
