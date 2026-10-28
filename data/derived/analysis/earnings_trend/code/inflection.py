"""실적 추세 연구 — '실적 발표가 추세의 변곡점인가'(1년 4번 변곡) 검정.

실행: py data/derived/analysis/earnings_trend/code/inflection.py  → results/inflection.json

같은 종목에서 기준일 앞 60거래일 초과수익(QQQ 대비)과 뒤 60거래일 초과수익의 관계를
① 실적 반응일 기준, ② 실적 반응일에서 15거래일 이상 떨어진 보통 날 기준으로 비교한다.
실적이 변곡점이면 ①에서 앞뒤 부호가 같은 비율(추세 지속)이 ②보다 낮아야 한다.
보통 날은 종목마다 10거래일 간격으로 모두 쓴다(무작위 추출 없음 → 재현 가능).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sec import RESULTS  # noqa: E402

N = 60


def main() -> None:
    px = pd.concat([pd.read_csv(RESULTS / "prices_adj_close.csv", index_col=0),
                    pd.read_csv(RESULTS / "prices_adj_close_broad.csv", index_col=0).drop(columns=["QQQ", "SPY", "SOXX", "IGV", "XLV", "XLU"], errors="ignore")], axis=1)
    px.index = pd.to_datetime(px.index)
    px = px.loc[:, ~px.columns.duplicated()]
    ev = pd.concat([pd.read_csv(RESULTS / "events.csv"), pd.read_csv(RESULTS / "events_broad.csv")])
    lr = np.log(px).diff()
    b = lr["QQQ"]
    days = px.index
    out = {}
    for period, lo, hi in [("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2026-08-01")]:
        recs = {"earn": [], "plain": []}
        for t in px.columns:
            if t in ("QQQ", "SPY", "SOXX", "IGV", "XLV", "XLU"):
                continue
            x = (lr[t] - b).values
            rd = sorted(days.searchsorted(pd.to_datetime(ev.loc[ev["ticker"] == t, "reaction_date"])))
            rd = [k for k in rd if k < len(days)]
            near = np.zeros(len(days), bool)
            for k in rd:
                near[max(0, k - 15):k + 16] = True
            for kind, anchors in [("earn", rd), ("plain", [k for k in range(N + 1, len(days) - N, 10) if not near[k]])]:
                for k in anchors:
                    if k - N < 1 or k + N >= len(days) or not (lo <= str(days[k].date()) < hi):
                        continue
                    pre = np.nansum(x[k - N:k])  # 기준일 전날까지
                    post = np.nansum(x[k + 1:k + 1 + N]) if kind == "earn" else np.nansum(x[k:k + N])  # 실적은 반응일 다음 날부터
                    if np.isfinite(pre) and np.isfinite(post) and pre != 0 and post != 0:
                        recs[kind].append((pre, post))
        res = {}
        for kind, v in recs.items():
            a = np.array(v)
            res[kind] = {"n": len(a), "same_sign_pct": round(float((np.sign(a[:, 0]) == np.sign(a[:, 1])).mean() * 100), 1),
                         "spearman": round(float(stats.spearmanr(a[:, 0], a[:, 1]).statistic), 3)}
        out[period] = res
    (RESULTS / "inflection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
