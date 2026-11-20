"""탐색(사전 등록 밖, 결과 본 뒤): 한국인 매수 비중이 극단으로 높은 종목(K1 ≥ 0.3%, 거래대금의 0.3% 이상)의 다음 20·60일.
2026 펀드 종목 예시에서 K1 0.3% 이상 종목이 크게 빠진 것을 보고 정한 문턱이다 → 2025는 문턱을 정하는 데 쓰지 않은 확인 구간.
결과: results/extreme.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def t4(s):
    ts = []
    for k in range(4):
        x = s.iloc[k::4]
        ts.append(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan)
    return round(float(np.nanmedian(ts)), 2) if not all(np.isnan(ts)) else np.nan


def main() -> None:
    p = pd.read_csv(BASE / "cache/panel.csv", parse_dates=["date"])
    rows = []
    for uname, u in (("넓은 표본", p), ("펀드 161", p[p.fund])):
        for half in ("2025", "H1", "H2"):
            g = u[u.half == half]
            for y in ("y20", "y60"):
                wk = []
                for d, gg in g.groupby("date"):
                    gg = gg.dropna(subset=[y])
                    hi = gg[gg.K1 >= 0.3]
                    rest = gg[gg.K1 < 0.3]
                    if len(hi) >= 3:
                        wk.append((d, hi[y].mean() - rest[y].mean(), (hi[y] > 0).mean(), (rest[y] > 0).mean(), len(hi)))
                w = pd.DataFrame(wk, columns=["date", "diff", "win_hi", "win_rest", "n_hi"])
                rows.append({"표본": uname, "구간": half, "기간": y, "주 수": len(w), "주당 극단 종목": round(w.n_hi.mean(), 1) if len(w) else np.nan,
                             "극단 − 나머지(%p)": round(w["diff"].mean(), 2) if len(w) else np.nan, "t": t4(w["diff"]) if len(w) > 8 else np.nan,
                             "극단 QQQ 이긴 비율": round(w.win_hi.mean(), 3) if len(w) else np.nan, "나머지": round(w.win_rest.mean(), 3) if len(w) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "results/extreme.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
