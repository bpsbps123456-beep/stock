"""이동 전방 검증(AGENTS 2026-11-20 '가장 최근 장세 우선'): 최근에 통한 방향을 바로 다음에 쓰면 맞나.

매주 결정일 w마다 신호 4개의 '상위 − 하위' 다음 20일 초과수익 차(spread)를 잰다.
- 적응 방향 = 결과가 이미 끝난 최근 8주(w−12 ~ w−5주, 20거래일 결과가 w 전에 확정)의 spread 평균 부호
- 적응 성적 = 적응 방향 × 이번 주 spread  (양수면 '최근 트렌드 따라가기'가 맞은 주)
신호: crowd(K1 ≥ 0.3% 쏠림 − 나머지), mom(20일 모멘텀 상위 1/3 − 하위 1/3), lowvol(60일 변동성 하위 1/3 − 상위 1/3),
      size(거래대금 상위 1/3 − 하위 1/3). 넓은 표본. 결과: results/adaptive.csv, results/adaptive_weekly.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def spreads(p):
    out = []
    for d, g in p.groupby("date"):
        g = g.dropna(subset=["y20"])
        if len(g) < 100:
            continue
        r = {"date": d}
        hi = g[g.K1 >= 0.3]
        r["crowd"] = hi.y20.mean() - g[g.K1 < 0.3].y20.mean() if len(hi) >= 3 else np.nan
        for nm, col, sign in (("mom", "mom20", 1), ("lowvol", "vol60", -1), ("size", "logdv", 1)):
            a, b = g[col].quantile(1 / 3), g[col].quantile(2 / 3)
            top, bot = g[g[col] >= b].y20.mean(), g[g[col] <= a].y20.mean()
            r[nm] = sign * (top - bot)
        out.append(r)
    return pd.DataFrame(out).set_index("date").sort_index()


def main() -> None:
    p = pd.read_csv(BASE / "cache/panel.csv", parse_dates=["date"])
    s = spreads(p)
    rows, weekly = [], []
    for sig in s.columns:
        x = s[sig]
        past = x.shift(5).rolling(8, min_periods=6).mean()  # w−12 ~ w−5주: 결정일에 이미 결과가 끝난 주만
        adj = np.sign(past) * x
        w = pd.DataFrame({"spread": x, "trend_sign": np.sign(past), "adaptive": adj}).assign(signal=sig)
        weekly.append(w)
        for nm, lo, hi in (("2025", "2025-01-01", "2025-12-31"), ("2026 상반기", "2026-01-01", "2026-06-30"),
                           ("2026 7~10월", "2026-07-01", "2026-10-31"), ("가장 최근 8주", "2026-08-21", "2026-10-31")):
            a = adj.loc[lo:hi].dropna()
            f = x.loc[lo:hi].dropna()
            rows.append({"신호": sig, "구간": nm, "주 수": len(a),
                         "최근 따라가기 맞은 주": f"{int((a > 0).sum())}/{len(a)}" if len(a) else "",
                         "따라가기 평균(%p)": round(float(a.mean()), 2) if len(a) else np.nan,
                         "고정 방향 평균(%p)": round(float(f.mean()), 2) if len(f) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "results/adaptive.csv", index=False)
    pd.concat(weekly).to_csv(BASE / "results/adaptive_weekly.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
