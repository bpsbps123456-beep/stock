"""SOXX 향후 10거래일 최저 종가의 경험 범위와 시간순 재생 (데스크 §13.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .news import market_calendar

START = pd.Timestamp("2026-01-02")
CHECK_FROM = pd.Timestamp("2026-07-01")
HORIZON = 10
VOL_WINDOW = 20
MIN_TRAIN = 30
QUANTILES = (0.1, 0.5, 0.9)
BASE = "과거 10거래일 최저 종가 변화율"
VOL = "최근 20거래일 변동성 반영"


def _panel(close: pd.Series) -> pd.DataFrame:
    from .bottom import frame

    # Apply the exchange calendar before taking windows. A missing session must
    # invalidate its windows, never shift the following label one session later.
    dates = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    series = pd.Series(close.to_numpy(dtype=float), index=dates).sort_index()
    series = series.loc["2025-08-01":]
    if series.empty:
        return pd.DataFrame()
    sessions = market_calendar().sessions_in_range(series.index[0], series.index[-1])
    c = series[~series.index.duplicated(keep="last")].reindex(sessions)
    c = c.where(np.isfinite(c) & (c > 0))
    vol = c.pct_change(fill_method=None).rolling(VOL_WINDOW).std()
    fwd_min = c.rolling(HORIZON).min().shift(-HORIZON)
    detector = frame(c)
    return pd.DataFrame({"close": c, "vol": vol, "target": fwd_min / c - 1,
                         "observed_low": detector["low"] / c - 1, "in_dd": detector["in_dd"]}).loc[START:]


def _bands(panel: pd.DataFrame, pos: int) -> tuple[int, dict]:
    """Fit paired bands using labels whose tenth close is already known."""
    matured = panel.iloc[:max(0, pos - HORIZON + 1)]
    train = matured.loc[(matured["vol"] > 0) & matured["target"].notna()]
    current_vol = panel["vol"].iat[pos]
    if len(train) < MIN_TRAIN or not np.isfinite(current_vol) or current_vol <= 0:
        return len(train), {}
    raw = train["target"].to_numpy()
    scaled = raw / train["vol"].to_numpy()
    return len(train), {BASE: np.quantile(raw, QUANTILES),
                        VOL: np.quantile(scaled, QUANTILES) * current_vol}


def _replay(panel: pd.DataFrame, start: int, stop: int) -> list[dict]:
    rows = []
    # Sampled ten sessions apart, so realized windows in each period are disjoint.
    for pos in range(start, stop - HORIZON, HORIZON):
        actual = panel["target"].iat[pos]
        _, bands = _bands(panel, pos)
        if not bands or not np.isfinite(actual):
            continue
        rows.append({"date": panel.index[pos].date().isoformat(),
                     "end": panel.index[pos + HORIZON].date().isoformat(),
                     "in_dd": bool(panel["in_dd"].iat[pos]),
                     "observed_low": float(panel["observed_low"].iat[pos]),
                     "actual": float(actual), "bands": bands})
    return rows


def _metrics(rows: list[dict], method: str) -> dict:
    n = len(rows)
    summary = {"n": n, "hits": 0, "below": 0, "above": 0, "coverage": None,
               "mae_pct": None, "mean_width_pct": None, "interval_score": None,
               "start": rows[0]["date"] if n else None, "end": rows[-1]["end"] if n else None}
    if not n:
        return summary
    lo, med, hi = np.array([row["bands"][method] for row in rows]).T
    actual = np.array([row["actual"] for row in rows])
    below, above = int((actual < lo).sum()), int((actual > hi).sum())
    hits = n - below - above
    # Central 80% interval score (smaller is better). Width and misses both cost.
    score = hi - lo + 10 * (np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0))
    return dict(summary, hits=hits, below=below, above=above, coverage=hits / n,
                mae_pct=float(np.abs(actual - med).mean() * 100),
                mean_width_pct=float((hi - lo).mean() * 100), interval_score=float(score.mean() * 100))


def replay_outlook(close: pd.Series) -> tuple[dict, list[dict]]:
    """Return a JSON-ready outlook and individual verification cases for research."""
    panel = _panel(close)
    empty = _metrics([], BASE)
    out = {"available": False, "as_of": None, "horizon": HORIZON, "close": None,
           "lower": None, "center": None, "upper": None, "method": None, "n_train": 0,
           "reason": "2026년 상반기 비교에 필요한 완료 종가가 부족합니다.",
           "validation": dict(empty, baseline_mae_pct=None, drawdown=empty)}
    if panel.empty:
        return out, []
    out["as_of"] = panel.index[-1].date().isoformat()
    if pd.notna(panel["close"].iat[-1]):
        out["close"] = float(panel["close"].iat[-1])
    split = int(panel.index.searchsorted(CHECK_FROM))
    if split == len(panel) or panel.index[0] != START:
        return out, []
    selection = _replay(panel, 0, split)
    if len(selection) < 3:
        return out, []
    calibration = {key: _metrics(selection, key) for key in (BASE, VOL)}
    method = min(calibration, key=lambda key: calibration[key]["interval_score"])
    checks = _replay(panel, split, len(panel))
    validation = _metrics(checks, method)
    baseline = _metrics(checks, BASE)
    past_low_errors = [abs(r["observed_low"] - r["actual"]) for r in checks if np.isfinite(r["observed_low"])]
    validation.update(baseline_mae_pct=baseline["mae_pct"], baseline=baseline,
                      observed_low_n=len(past_low_errors),
                      observed_low_mae_pct=float(np.mean(past_low_errors) * 100) if past_low_errors else None,
                      drawdown=_metrics([r for r in checks if r["in_dd"]], method))
    n_train, bands = _bands(panel, len(panel) - 1)
    out.update(validation=validation, method=method, selection=calibration, n_train=n_train)
    if not bands or out["close"] is None:
        out["reason"] = "최근 20거래일 종가 또는 학습에 필요한 완료 가격창이 부족합니다."
        return out, checks
    lower, center, upper = out["close"] * (1 + bands[method])
    out.update(available=True, reason=None, lower=float(lower), center=float(center), upper=float(upper))
    return out, checks


def build_outlook(close: pd.Series) -> dict:
    """Use split-checked, completed SOXX closes, exactly as in the historical replay."""
    return replay_outlook(close)[0]
