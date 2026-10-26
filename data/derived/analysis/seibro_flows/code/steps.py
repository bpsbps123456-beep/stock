"""구조 단절 계단 검사(BRIEF v2 §3, SEIBRO만 사용).

신호마다 단절 전 40세션(8주) 중앙값과 단절 후 40세션 중앙값의 차이 ÷ 단절 전 250세션 IQR = d. |d| ≥ 0.5이면 계단.
단절 시점은 신호 창이 담은 마지막 미국 거래일(t_end, 보관은 보관 날짜)이 단절일 이상이 되는 첫 진입이다.
출력: derived/steps.json, 그리고 계단 신호의 체제 내 정규화 계열을 만드는 함수 regime_adjust.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
BREAKS = {"B0": "2025-12-15", "B1": "2026-05-22", "B2": "2026-07-31"}
SIGS = ["S1", "S2", "S3", "S4", "S5n", "S5b", "S7", "S8", "S9n", "S9b", "S10n", "S10b", "L1", "L2", "C1", "C2", "C3",
        "C4"]


def clock(df: pd.DataFrame, s: str) -> pd.Series:
    return pd.to_datetime(df["cdate"] if s.startswith("C") else df["t_end"])


def step_table(df: pd.DataFrame) -> dict:
    out = {}
    for s in SIGS:
        rec = {}
        for b, day in BREAKS.items():
            t = clock(df, s)
            x = df[s]
            pre = x[t < pd.Timestamp(day)].dropna()
            post = x[t >= pd.Timestamp(day)].dropna()
            if len(pre) < 60 or len(post) < 40:
                rec[b] = None
                continue
            iqr = pre.iloc[-250:].quantile(0.75) - pre.iloc[-250:].quantile(0.25)
            d = (post.iloc[:40].median() - pre.iloc[-40:].median()) / iqr if iqr > 0 else np.nan
            rec[b] = {"d": round(float(d), 3), "step": bool(abs(d) >= 0.5)}
        out[s] = rec
    return out


def regime_adjust(df: pd.DataFrame, s: str, steps: dict) -> pd.Series:
    """계단이 있는 단절마다 구간을 나누고, 구간 안 확장 중앙값(첫 구간은 250세션 이동 중앙값)을 뺀다. 단절 뒤 20세션은 NaN."""
    x = df[s].astype(float)
    t = clock(df, s)
    cuts = [pd.Timestamp(BREAKS[b]) for b, r in steps.get(s, {}).items() if r and r["step"]]
    if not cuts:
        return x
    cuts = sorted(cuts)
    out = pd.Series(np.nan, index=x.index)
    seg_id = np.searchsorted(np.array(cuts, dtype="datetime64[ns]"), t.values.astype("datetime64[ns]"), side="right")
    for j in np.unique(seg_id):
        m = seg_id == j
        seg = x[m]
        if j == 0:
            out[m] = seg - seg.rolling(250, min_periods=60).median()
        else:
            med = seg.expanding(min_periods=20).median()
            out[m] = seg - med
    return out


def main() -> None:
    df = pd.read_parquet(DER / "sig_daily.parquet")
    res = step_table(df)
    path = DER / "steps.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({s: {b: (r["d"] if r else None) for b, r in v.items()} for s, v in res.items()}, ensure_ascii=False))
    print("stepped:", {s: [b for b, r in v.items() if r and r["step"]] for s, v in res.items()})
    print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
