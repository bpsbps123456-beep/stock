"""추세선만 남긴 지표에서 선이 얼마나 자주 떠 있나 (2026-10-23 사용자 "이렇게 하면 거의 선이 없나?").

structure_ref.py(= 지표와 같은 긋기·확립·돌파 규칙)를 1시간봉에 돌리며 봉마다 떠 있는 선을 센다.
  확립 선(굵은 선, 기본 표시) / 작도 중 선(가는 선) / 돌파된 선(흐린 선)
최근 180일 봉 비율로 보고, 확립 선이 하나도 없는 구간이 한 번에 얼마나 이어졌는지도 본다.
"""
from __future__ import annotations

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import structure_ref as sr  # noqa: E402


def states(b, atr):
    piv = {-1: [], 1: []}
    active = {-1: None, 1: None}
    flipped = {-1: None, 1: None}
    out = []
    for i in range(len(b)):
        r = b[i]
        for d in (-1, 1):
            t = flipped[d]
            if t is None:
                continue
            if abs(r["c"] - t.val(i)) > sr.STALE * atr[i]:
                flipped[d] = None
                continue
            ev = t.wm.step(r, t.val(i), atr[i], sr.TOL_B, sr.TOL_T, i)
            if ev and ev not in (5, 8):
                flipped[d] = None
        for d in (-1, 1):
            t = active[d]
            if t is None:
                continue
            if not t.est:
                beyond = (r["c"] - t.val(i)) * (1 if d < 0 else -1)
                if beyond > sr.DISCARD * atr[i]:
                    active[d] = None
                elif i - t.ib >= sr.PROV:
                    t.est = True
                continue
            ev = t.wm.step(r, t.val(i), atr[i], sr.TOL_B, sr.TOL_T, i)
            if ev in (6, 7):
                t.broken = True
                flipped[d] = t
                active[d] = None
        j = i - sr.PIV
        for d, hi in ((-1, True), (1, False)):
            if j >= 0 and sr.pivot_at(b, j, sr.PIV, hi):
                p = b[j]["h"] if hi else b[j]["l"]
                cur = active[d]
                if cur is not None and i - cur.ia > sr.MAX_LOOK:
                    active[d] = cur = None
                stale = cur is not None and cur.est and abs(r["c"] - cur.val(i)) > sr.STALE * atr[i]
                if cur is None or stale or not cur.est:
                    t = sr.find_line(b, piv[d], j, p, d, i, atr[i])
                    if t is not None and (cur is None or stale or t.ia <= cur.ia):
                        active[d] = t
                piv[d].append((j, p))
                piv[d] = [(a, q) for a, q in piv[d] if i - a <= sr.MAX_LOOK + sr.PIV][-40:]
        est = {d: active[d] is not None and active[d].est for d in (-1, 1)}
        prov = {d: active[d] is not None and not active[d].est for d in (-1, 1)}
        flp = {d: flipped[d] is not None for d in (-1, 1)}
        out.append((est, prov, flp))
    return out


def main(syms):
    print("| 종목 | 180일 60분봉 | 굵은 선 1개 이상 | 굵은 선 2개(빨강·초록) | 굵은 선 없음 | 없음이 이어진 길이 중앙값·최장 | 가는 선까지 켜면 1개 이상 | 흐린 선까지 켜면 1개 이상 |")
    print("|---|---|---|---|---|---|---|---|")
    for sym in syms:
        b = bt.load(sym, "1h")
        atr = sr.atr_series(b)
        st = states(b, atr)
        start = b[-1]["t"] - 180 * 86400000
        idx = [i for i in range(len(b)) if b[i]["t"] >= start]
        n = len(idx)
        one = sum(1 for i in idx if any(st[i][0].values()))
        two = sum(1 for i in idx if all(st[i][0].values()))
        prov1 = sum(1 for i in idx if any(st[i][0].values()) or any(st[i][1].values()))
        flp1 = sum(1 for i in idx if any(st[i][0].values()) or any(st[i][1].values()) or any(st[i][2].values()))
        gaps, run = [], 0
        for i in idx:
            if any(st[i][0].values()):
                if run:
                    gaps.append(run)
                run = 0
            else:
                run += 1
        if run:
            gaps.append(run)
        g = f"{statistics.median(gaps):.0f}봉·{max(gaps)}봉" if gaps else "-"
        print(f"| {sym} | {n} | {one/n*100:.0f}% | {two/n*100:.0f}% | {(n-one)/n*100:.0f}% | {g} | {prov1/n*100:.0f}% | {flp1/n*100:.0f}% |")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["IREN", "PANW", "MU", "NVDA", "BTCUSDT", "ETHUSDT"])
