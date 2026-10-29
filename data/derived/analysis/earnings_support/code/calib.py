"""닿을 확률 보정표(부모 작업의 예측 기록용, 사전 등록 '일찍 넘길 것' 2).

실행: py data/derived/analysis/earnings_support/code/calib.py
출력: results/touch_calibration.csv  — P(반응일 저가가 기준 종가 × (1 − m·s) 이하에 닿음)
      (m = 0.25~3.0, s = 직전 실적 8번(최소 4번)의 반응일 |r0| 평균), 체결 창 B·C, 구간별. % 판(d = 1~20%)도 같은 파일.
      results/z_quantiles.csv  — z = r0/s 분위수(구간·시총별)
      results/up_share.csv     — 반응일 종가 > 기준 종가 비율(구간·시총별)
'닿음'은 백테스트 체결 규칙과 같다(프리장 오류 체결 거름, 정규장 0.1% 관통).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from events import load  # noqa: E402
from lib import RES  # noqa: E402
from sim import fills  # noqa: E402

M = np.round(np.arange(0.25, 3.01, 0.25), 2)
D = np.arange(1, 21)


def cap_bucket(x: float) -> str:
    if not np.isfinite(x):
        return "na"
    return "<10B" if x < 10 else "10-50B" if x < 50 else "50-200B" if x < 200 else ">=200B"


def main() -> None:
    meta, bars, _ = load()
    rows = []
    for r in meta.itertuples():
        b = bars.get(r.eid)
        if b is None:
            continue
        arr = {k: b[k].to_numpy() for k in ("seg", "o", "h", "l", "c", "nl")}
        rec = {"eid": r.eid, "period": r.period, "cap": cap_bucket(r.mcap_b), "amc": r.amc}
        for w in ("B", "C"):
            pos, _ = fills(arr, r.base_close * (1 - D / 100), w, r.amc)
            for d, p in zip(D, pos):
                rec[f"{w}_d{d}"] = p >= 0
            if r.s is not None and np.isfinite(r.s):
                pos, _ = fills(arr, r.base_close * (1 - M * r.s / 100), w, r.amc)
                for m, p in zip(M, pos):
                    rec[f"{w}_m{m}"] = p >= 0
        rows.append(rec)
    t = pd.DataFrame(rows)
    out = []
    groups = [("all", t)] + [(p, g) for p, g in t.groupby("period")] + [("2026", t[t.period.str.startswith("2026")])]
    for name, g in groups:
        for cap_name, gg in [("all", g), (">=10B", g[g.cap.isin(["10-50B", "50-200B", ">=200B"])]),
                             (">=50B", g[g.cap.isin(["50-200B", ">=200B"])])]:
            for w in ("B", "C"):
                for m in M:
                    col = f"{w}_m{m}"
                    v = gg[col].dropna() if col in gg else pd.Series(dtype=float)
                    out.append({"period": name, "cap": cap_name, "window": w, "unit": "s", "x": m,
                                "p_touch": float(v.astype(float).mean()) if len(v) else np.nan, "n": len(v)})
                for d in D:
                    col = f"{w}_d{d}"
                    v = gg[col].dropna()
                    out.append({"period": name, "cap": cap_name, "window": w, "unit": "pct", "x": d,
                                "p_touch": float(v.astype(float).mean()) if len(v) else np.nan, "n": len(v)})
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(RES / "touch_calibration.csv", index=False)

    meta = meta.assign(cap=meta.mcap_b.map(cap_bucket), z=meta.r0 / meta.s)
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    zq, up = [], []
    for name, g in [("all", meta)] + list(meta.groupby("period")) + [("2026", meta[meta.period.str.startswith("2026")])]:
        for cap_name, gg in [("all", g), (">=10B", g[g.mcap_b >= 10]), (">=50B", g[g.mcap_b >= 50])]:
            z = gg.z.dropna()
            zq.append({"period": name, "cap": cap_name, "n": len(z), **{f"q{int(q * 100):02d}": z.quantile(q) for q in qs},
                       "mean_abs_z": z.abs().mean(), "share_abs_z_gt2": (z.abs() > 2).mean()})
            up.append({"period": name, "cap": cap_name, "n": len(gg), "up_share": (gg.r0 > 0).mean(),
                       "mean_r0": gg.r0.mean(), "median_r0": gg.r0.median()})
    pd.DataFrame(zq).to_csv(RES / "z_quantiles.csv", index=False)
    pd.DataFrame(up).to_csv(RES / "up_share.csv", index=False)
    print(pd.DataFrame(zq).round(3).to_string())
    print(pd.DataFrame(up).round(3).to_string())
    c = pd.DataFrame(out)
    piv = c[(c.cap == ">=10B") & (c.unit == "s")].pivot_table(index="x", columns=["period", "window"], values="p_touch")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
