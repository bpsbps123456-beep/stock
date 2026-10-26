"""사건 연구 전처리(창 종류 보정·직전 20창 z·직전 60창 백분위)가 앞 창 값만 쓰는지: 실제 특징 판에서 t 뒤 원값을 무작위로 바꿔도
t 이하 행의 Z·PCT가 같아야 한다. 시각 t = 실시간 감사 T0~T3과 같은 창 + 무작위 20개. 쓰기: events/results/audit_preproc.json
실행: py -B audit_preproc.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_study as es  # noqa: E402

F, meta = es.load_features()
rng = np.random.default_rng(20261023)
out = {"cuts": [], "max_abs_diff_le_t": 0.0, "changed_gt_t_cells": 0}
for market in ("kr", "us"):
    g = F[F["market"] == market].sort_values("close_kst").reset_index(drop=True)
    P0 = es.build_panel(F, meta, market)
    cuts = [int(x) for x in rng.integers(40, len(g) - 5, 20)]
    for wid in ("K2026-07-30", "K2026-07-31", "K2026-10-14", "U2026-07-29", "U2026-07-30", "U2026-10-13"):
        if wid in P0.pos:
            cuts.append(P0.pos[wid])
    for t in cuts:
        X = g[P0.feats].to_numpy(float).copy()
        X[t + 1:] = rng.normal(0, 10, X[t + 1:].shape)
        P1 = es.build_panel(F, meta, market, X_override=X)
        a, b = P0.Z[: t + 1], P1.Z[: t + 1]
        c, d = P0.PCT[: t + 1], P1.PCT[: t + 1]
        diff = max(float(np.nanmax(np.abs(a - b), initial=0)), float(np.nanmax(np.abs(c - d), initial=0)))
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum() + (np.isnan(c) != np.isnan(d)).sum())
        changed_after = int((~np.isclose(P0.Z[t + 1:], P1.Z[t + 1:], equal_nan=True)).sum())
        out["cuts"].append({"market": market, "t_win": str(P0.win_id[t]), "max_abs_diff_le_t": diff,
                            "nan_mismatch_le_t": nan_mismatch, "changed_gt_t_cells": changed_after})
        out["max_abs_diff_le_t"] = max(out["max_abs_diff_le_t"], diff + nan_mismatch)
        out["changed_gt_t_cells"] += changed_after
out["n_cuts"] = len(out["cuts"])
(es.RESULTS / "audit_preproc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
print(out["n_cuts"], out["max_abs_diff_le_t"], out["changed_gt_t_cells"])
