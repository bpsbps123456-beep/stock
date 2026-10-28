"""실적 추세 연구 — 반응일 점수(3개 숫자) 구간별 실제 반응과 문턱값(프롬프트 v2의 환산표 근거).

실행: py data/derived/analysis/earnings_trend/code/calibrate.py  → results/calibration.json
문턱값은 2025-01~2026-09 핵심 32종목 발표(222건)의 3분위 경계다. 시즌 안 순위가 아니라 절대값이라
새 발표 한 건에도 바로 쓸 수 있다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sec import RESULTS  # noqa: E402

COMP = ["eps_surprise", "beat_guide", "guide_accel"]


def main() -> None:
    d = pd.read_csv(RESULTS / "scored_events.csv")
    cut = {c: [round(float(x), 2) for x in d[c].quantile([1 / 3, 2 / 3])] for c in COMP}
    # 각 숫자를 −1/0/+1로(하위·중간·상위 1/3), 결측은 0, 합계 −3..+3
    pts = sum(np.select([d[c] <= cut[c][0], d[c] >= cut[c][1]], [-1, 1], 0) * d[c].notna() for c in COMP)
    d["pts"] = pts
    d["band"] = pd.cut(d["pts"], [-4, -2, 0, 1, 4], labels=["−3~−2", "−1~0", "+1", "+2~+3"])
    tab = d.groupby(["band", "window"], observed=True).agg(
        n=("r0_x", "size"), r0_x=("r0_x", "mean"), r0_up_pct=("r0_x", lambda s: (s > 0).mean() * 100),
        drift_x=("drift_x", "mean"), drift_med=("drift_x", "median")).round(1)
    out = {"cutoffs_tercile": cut, "n": len(d), "table": tab.reset_index().astype({"band": str}).to_dict("records")}
    (RESULTS / "calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(cut, ensure_ascii=False))
    print(tab.to_string())


if __name__ == "__main__":
    main()
