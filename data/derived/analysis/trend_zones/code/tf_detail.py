"""tf_extremes.py 결과 풀어 보기(2026-10-11): 시간봉별 중앙값(h 1·5·10일), 네 구간 견고한 정의, 수익 %p로 환산.

B(1시간봉 2024~2026)의 60·120·180·240·300·360분·하루에서
  1) 위·아래 타당도(순위 %p) 구간별 중앙값, h = 1·5·10일
  2) 60·120분 극상방에서 네 구간 최소값이 가장 큰 정의 상위 8(설명용 — 고른 것이 아니다)
  3) RSI14 70/30, 칼만식 EMA20·atr14 ±2.0, 볼린저20 ±2.0을 60·120·240분·하루에서 '5일 초과 수익(같은 시각 종목 평균 대비, 1·99% 자름)'과 '5일 상승 확률'로
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/tf_detail.py
"""
import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import tf_extremes as T  # noqa: E402

P = list(T.PERIODS_B)


def main():
    out = json.load(open(os.path.join(Zm.OUT, "tf_extremes.json"), encoding="utf-8"))
    lines = ["== B 중앙값(순위 %p, 자격 정의) — 위 = 극상방 뒤 덜 오름(+면 팔 자리), 아래 = 극하방 뒤 더 오름"]
    for tf, res in out["B"].items():
        for side in ("위", "아래"):
            cells = []
            for h in (1, 5, 10):
                med = []
                for p in P:
                    v = np.array([r[f"{side}{h}{p}"] for r in res if r[f"{side}{h}{p}"] is not None])
                    med.append(f"{np.median(v) * 100:+5.2f}" if len(v) else "  —  ")
                cells.append(f"h{h}: " + " ".join(med))
            lines.append(f"{tf:>4} {side:<2} | " + " | ".join(cells))
    for tf in ("60", "120"):
        res = out["B"][tf]
        ok = [r for r in res if all(r[f"위5{p}"] is not None for p in P)]
        ok.sort(key=lambda r: -min(r[f"위5{p}"] for p in P))
        lines += ["", f"== {tf}분 극상방, 네 구간 최소값 큰 순(h5, 설명용)"]
        for r in ok[:8]:
            lines.append(f"{r['name']:<28} " + " ".join(f"{p} {r[f'위5{p}'] * 100:+5.2f}" for p in P))
    # 수익 %p 환산
    B = T.load_hour()
    lines += ["", "== 대표 정의의 5일 초과 수익(같은 시각 종목 평균 대비, %) · 5일 상승 확률 — 구간: " + "/".join(P)]
    names = ("RSI14 70/30", "칼만식 EMA20·atr14 ±2.0", "볼린저20 ±2.0")
    for tf, per in ((60, 1), (120, 2), (240, 4), ("하루", None)):
        O, Hh, L, C, end, days = T.resample(B, per or 1, whole_day=per is None)
        D, V = T.defs_for(O, Hh, L, C, 130, False)
        entry = np.minimum(end + 1, len(B.day) - 1)
        ex = B.exit_index(entry, 5)
        Ob = B.O.to_numpy()
        f = pd.DataFrame(np.where((ex >= 0)[:, None], Ob[np.maximum(ex, 0)] / Ob[entry] - 1, np.nan))
        lo, hi = f.quantile(0.01, axis=1), f.quantile(0.99, axis=1)
        fw = f.clip(lower=lo, upper=hi, axis=0)
        exr = fw.sub(fw.mean(axis=1), axis=0).to_numpy()
        raw = f.to_numpy()
        for nm in names:
            t, b = next((t, b) for n2, t, b in D if n2 == nm)
            cells = []
            for side, M in (("극상방", t), ("극하방", b)):
                vals = []
                for p, (a, z, purge) in T.PERIODS_B.items():
                    s = (days >= a) & (days <= z)
                    Mr = M[s] & np.isfinite(exr[s])
                    n = Mr.sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        e = np.where(n > 0, np.where(Mr, exr[s], 0).sum(axis=1) / n, np.nan)
                        u = np.where(n > 0, (Mr & (raw[s] > 0)).sum(axis=1) / n, np.nan)
                    e = pd.Series(e).groupby(days[s]).mean().mean()
                    u = pd.Series(u).groupby(days[s]).mean().mean()
                    vals.append(f"{e * 100:+5.2f}%·{u * 100:3.0f}%")
                cells.append(f"{side} " + " / ".join(vals))
            lines.append(f"{str(tf):>4} {nm:<24} " + " | ".join(cells))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "tf_detail.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
