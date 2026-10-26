"""검증 #2 보강 2 (2026-10-11, 연구용): K3 효과가 'SOXX가 SMA20 아래' 전체인지, 깊은 하락 주에만 있는지.
verify_2_k3_extra.py 6절(다섯 구간)에서 Q1(SOXX ≤ SMA20 -3%)만 T3 양수였다. 그 구간 경계는 결과에서 나왔으므로
절단값을 -2%·-3%·-5% 셋으로 함께 보고, 구간별·사건별로 나눈다(선택편향 공개: 절단 3개 × 구간 5개를 한 번에 본다).
경고 주를 얕은 경고(SMA20 아래, 거리 > 절단) / 깊은 경고(거리 ≤ 절단)로 나눠 평상 주와 비교. h10 초과 수익 T3 = 상방-극상방.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_2_k3_depth.py
결과: results/verify_2_k3_depth.txt, verify_2_k3_depth.json
"""
import json, os, sys, warnings
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import zones_regime as ZR  # noqa: E402
import verify_2_k3_extra as VX  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
pct = VX.pct


def main():
    O, Hh, L_, C = Zm.load()
    I = Zm.Ind(O, Hh, L_, C)
    dates = I.dates
    res = json.load(open(os.path.join(Zm.OUT, "zones.json"), encoding="utf-8"))
    cfgs = [g["cfg"] for g in res["grid"] if g["eligible"]]
    wi = [Zm.cname(c) for c in cfgs].index(res["best"])
    F = Zm.Fwd(I, 10)
    o = pd.read_parquet(ZR.SOXX_SRC, columns=["date", "sym", "Close"])
    s = o[o.sym == "SOXX"].copy()
    s["d"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
    sc = s.drop_duplicates("d").set_index("d").sort_index()["Close"].astype(float)
    warn, dist = VX.warn_indep(dates, sc)
    D = np.vstack([VX.t3_daily(Zm.zones(I, c), F.ex)[0] for c in cfgs])
    seg = np.r_[0, np.cumsum(warn[1:] != warn[:-1])]
    lines, J = [], {}
    P = ("2023하", "앞", "뒤", "2025", "2024")
    allrows = np.unique(np.concatenate([VX.rows_of(dates, p, 10) for p in P]))

    def st(rr):
        t = np.array([VX.nm(D[k], rr) for k in range(len(cfgs))])
        f = t[np.isfinite(t)]
        return {"1위": float(t[wi]), "격자중앙": float(np.median(f)) if len(f) else np.nan,
                "양수": float(np.mean(f > 0)) if len(f) else np.nan, "날짜": int(len(rr)),
                "사건": int(len(set(seg[rr]))) if len(rr) else 0}

    f = lambda x: f"1위 {pct(x['1위'])} 격자 {pct(x['격자중앙'])} 양수 {x['양수'] * 100:3.0f}% ({x['날짜']}일·사건 {x['사건']})"
    for cut in (-0.02, -0.03, -0.05):
        lines.append(f"== 절단 {cut * 100:.0f}%: 평상 | 얕은 경고({cut * 100:.0f}% < 거리 < 0) | 깊은 경고(거리 ≤ {cut * 100:.0f}%)")
        J[str(cut)] = {}
        for p in P + ("전체",):
            rp = allrows if p == "전체" else VX.rows_of(dates, p, 10)
            ok = rp[np.isfinite(dist[rp])]
            n_ = ok[~warn[ok]]
            sh = ok[warn[ok] & (dist[ok] > cut)]
            dp = ok[warn[ok] & (dist[ok] <= cut)]
            a, b, c = st(n_), st(sh), st(dp)
            J[str(cut)][p] = {"평상": a, "얕은": b, "깊은": c}
            lines.append(f"  [{p:>5}] 평상 {f(a)} | 얕은 {f(b)} | 깊은 {f(c)}")
    # 깊은 경고(-3%) 사건별
    cut = -0.03
    ok = allrows[np.isfinite(dist[allrows])]
    dp = ok[warn[ok] & (dist[ok] <= cut)]
    lines += ["", "== 깊은 경고(-3%) 사건별: 사건 기간(신호일) · 날짜 · 1위 T3 · 격자 중앙 T3 · 빼면 깊은 경고 격자 중앙"]
    J["사건"] = []
    for sg in sorted(set(seg[dp])):
        rr = dp[seg[dp] == sg]
        rest = dp[seg[dp] != sg]
        a, b = st(rr), st(rest)
        J["사건"].append({"기간": [dates[rr[0]], dates[rr[-1]]], **a, "빼면격자": b["격자중앙"], "빼면1위": b["1위"]})
        lines.append(f"  {dates[rr[0]]}~{dates[rr[-1]]} {len(rr):3d}일: 1위 {pct(a['1위'])} 격자 {pct(a['격자중앙'])} "
                     f"| 빼면 1위 {pct(b['1위'])} 격자 {pct(b['격자중앙'])}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "verify_2_k3_depth.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(Zm.OUT, "verify_2_k3_depth.json"), "w", encoding="utf-8") as fh:
        json.dump(J, fh, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


if __name__ == "__main__":
    main()
