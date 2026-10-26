"""verify_tf_recompute.py 보조(검증용, 2026-10-11): A-조합 '일봉 + 분봉 극하방'(7/17~10/9)을 자체 코드로 다시 잰다.
5분봉 → 시계 기준 40·160·320분, 봉 40개 뒤부터. 일봉 상태는 1d_long 전날 종가 기준. h = 5일. 짧은 정의 88개(RSI2~14, 볼린저10·20, 칼만식 EMA5·10·20×rms20·atr14).
결과: results/verify_tf_combo.txt
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_tf_recompute as V  # noqa: E402

OUT = os.path.join(V.A, "trend_zones/results/verify_tf_combo.txt")
SHORT = re.compile(r"^(RSI(2|3|5|7|9|14) |볼린저(10|20) |칼만식 EMA(5|10|20)·(rms20|atr14) )")


def main():
    x = pd.read_parquet(f"{V.A}/rsi_breakout/cache/5m.parquet")
    x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(V.NY)
    x = x[x.sym != "QQQ"]
    bars = {s: g.set_index("ts")[["Open", "High", "Low", "Close"]].sort_index() for s, g in x.groupby("sym")}
    d = pd.read_parquet(f"{V.A}/rsi_breakout/cache/1d_long.parquet")
    d = d[d.sym != "QQQ"].copy()
    d["day"] = pd.to_datetime(d.ts).dt.strftime("%Y-%m-%d")
    Wd = {k: d.pivot_table(index="day", columns="sym", values=c).sort_index() for k, c in zip("OHLC", ("Open", "High", "Low", "Close"))}
    V.WARM = 130
    Dd = {nm: (t, b) for nm, t, b in V.defs(Wd) if SHORT.match(nm)}
    dday = list(Wd["C"].index)
    dsyms = list(Wd["C"].columns)
    V.WARM = 40
    for tf in (40, 160, 320):
        W, last = V.tf_panel(bars, tf)
        day = np.array(W["C"].index.get_level_values("day"))
        syms = list(W["C"].columns)
        cols = [dsyms.index(s) for s in syms]
        prev = np.array([dday.index(q) - 1 for q in day])          # 전날 일봉
        f, exday, _ = V.forward(bars, last, 5)
        rk = V.crank(f)
        res_i, res_c = [], []
        for nm, t, b in V.defs(W):
            if not SHORT.match(nm):
                continue
            db = Dd[nm][1][:, cols][prev]
            sel = np.ones(len(day), bool)
            sh, v = V.score(b, rk, day, sel, 1)
            res_i.append((nm, sh, v))
            sh2, v2 = V.score(b & db, rk, day, sel, 1)
            nd = len(set(day[(b & db).any(axis=1)]))
            res_c.append((nm, sh2, v2, nd))
        vi = np.array([v for nm, sh, v in res_i if 0.01 <= sh <= 0.2])
        vc = np.array([v for nm, sh, v, nd in res_c if sh >= 0.002 and nd >= 5])
        r14i = next(r for r in res_i if r[0] == "RSI14 70/30")
        r14c = next(r for r in res_c if r[0] == "RSI14 70/30")
        V.p(f"{tf}분 극하방 h5 단독: 양수 {np.mean(vi > 0) * 100:.0f}% 중앙 {np.median(vi) * 100:+.2f} (n{len(vi)}) RSI14 30 {r14i[2] * 100:+.2f} | "
            f"+일봉: 양수 {np.mean(vc > 0) * 100:.0f}% 중앙 {np.median(vc) * 100:+.2f} (n{len(vc)}) RSI14 30 {r14c[2] * 100:+.2f} (비중 {r14c[1] * 100:.2f}%, 날 {r14c[3]})")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(V.L))


if __name__ == "__main__":
    main()
