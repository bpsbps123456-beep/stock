"""체결·매도 시뮬레이션과 집계(사전 등록 BRIEF.md 그대로).

체결: 창 안 1시간봉을 시간순으로 본다. 프리·애프터 봉은 저가 ≤ L이고 (그 봉 종가 ≤ L×1.005 또는 다음 봉 저가 ≤ L×1.005),
정규장 봉은 저가 ≤ L×0.999. 체결가 = min(L, 처음 조건을 채운 봉의 시가).
매도: X1 반응일 종가, X2 다음 거래일 시가, X3 다음 거래일 종가, X4 5번째 거래일 종가(반응일 = 1번째),
X5 체결가 +3% 지정가(체결 봉 뒤 정규장 봉, 시가가 목표 위면 시가) 못 닿으면 X4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXITS = ("x1", "x2", "x3", "x4", "x5")
TP = 0.03


def window_mask(seg: np.ndarray, window: str, amc: bool) -> np.ndarray:
    if window == "A":
        return (seg == 1) | (seg == 2) | ((seg == 0) & amc)
    if window == "B":
        return (seg == 1) | (seg == 2)
    return seg == 2


def fills(b: dict, levels: np.ndarray, window: str, amc: bool) -> tuple[np.ndarray, np.ndarray]:
    """b: 창 봉 배열(seg,o,h,l,c,nl). 반환: 체결 봉 위치(없으면 −1), 체결가(없으면 nan)."""
    seg = b["seg"]
    ii = np.flatnonzero(window_mask(seg, window, amc))
    levels = np.asarray(levels, float)
    if not len(ii) or not len(levels):
        return np.full(len(levels), -1), np.full(len(levels), np.nan)
    lo, c, nl, o = b["l"][ii], b["c"][ii], b["nl"][ii], b["o"][ii]
    ext = seg[ii] < 2
    L = levels[:, None]
    with np.errstate(invalid="ignore"):
        q_ext = (lo[None, :] <= L) & ((c[None, :] <= L * 1.005) | (nl[None, :] <= L * 1.005))
        q_reg = lo[None, :] <= L * 0.999
    q = np.where(ext[None, :], q_ext, q_reg)
    hit = q.any(axis=1)
    first = np.where(hit, q.argmax(axis=1), 0)
    pos = np.where(hit, ii[first], -1)
    px = np.where(hit, np.minimum(levels, o[first]), np.nan)
    return pos, px


def take_profit(b: dict, pos: np.ndarray, px: np.ndarray, x4: float) -> np.ndarray:
    """체결 봉 뒤 정규장 봉에서 +3% 익절가. 못 닿으면 X4 가격."""
    reg = np.flatnonzero(b["seg"] >= 2)
    out = np.full(len(pos), np.nan)
    if not len(reg):
        return out
    H, O = b["h"][reg], b["o"][reg]
    tgt = px * (1 + TP)
    with np.errstate(invalid="ignore"):
        after = reg[None, :] > pos[:, None]
        hitm = after & (H[None, :] >= tgt[:, None])
    anyhit = hitm.any(axis=1)
    first = np.where(anyhit, hitm.argmax(axis=1), 0)
    sell = np.where(O[first] >= tgt, O[first], tgt)
    out = np.where(anyhit, sell, x4)
    out[pos < -1] = np.nan
    return np.where(np.isnan(px), np.nan, out)


def returns(b: dict, meta_row, pos: np.ndarray, px: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for k in ("x1", "x2", "x3", "x4"):
        v = getattr(meta_row, k)
        out[k] = v / px - 1 if np.isfinite(v) else np.full(len(px), np.nan)
    x4 = meta_row.x4 if np.isfinite(meta_row.x4) else np.nan
    out["x5"] = take_profit(b, pos, px, x4) / px - 1
    return out


def cluster_t(x: np.ndarray, g: np.ndarray) -> float:
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    if n < 5:
        return np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum().to_numpy()
    se = np.sqrt((s ** 2).sum()) / n
    return float(m / se) if se > 0 else np.nan


def summarize(df: pd.DataFrame, ret: str) -> dict:
    """df: 사건 한 줄(filled, ret 열, reaction_date). 체결 뒤 통계 + 사건당 기대값(미체결 0)."""
    valid = df[df[f"ok_{ret}"]] if f"ok_{ret}" in df else df
    valid = valid[~(valid["filled"] & ~np.isfinite(valid[ret]))]
    f = valid[valid["filled"]]
    r = f[ret].to_numpy(float)
    ev = np.where(valid["filled"], valid[ret].fillna(0), 0.0).astype(float)
    return {
        "n_events": len(valid), "n_fill": len(f), "fill_rate": len(f) / len(valid) if len(valid) else np.nan,
        "win": float((r > 0).mean()) if len(r) else np.nan, "mean": float(r.mean()) if len(r) else np.nan,
        "median": float(np.median(r)) if len(r) else np.nan,
        "t": cluster_t(r, f["reaction_date"].to_numpy()) if len(r) else np.nan,
        "p5": float(np.percentile(r, 5)) if len(r) else np.nan, "worst": float(r.min()) if len(r) else np.nan,
        "per_event": float(ev.mean()) if len(ev) else np.nan,
        "per_event_t": cluster_t(ev, valid["reaction_date"].to_numpy()) if len(ev) else np.nan,
    }
