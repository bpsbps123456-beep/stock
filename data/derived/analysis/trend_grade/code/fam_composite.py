"""추세 등급 계열 'composite' (2026-10-12): 서로 다른 가격·이평선 재료의 단면 백분위 평균.

재료마다 그날 보유 종목군(H.universe_mask & 값 있음) 안에서 백분위를 낸 뒤 가중 평균한다.
한 재료가 NaN이면 남은 재료의 가중 평균을 쓴다(가중치도 남은 것만으로 나눈다).
사전 등록 격자: 조합 18개(E1~E18) + 평활 8개(E2·E7·E11·E14 × 5·10일 평균) = 26개.
가격(OHLC)·이평선만 쓴다. 거래량·재무·뉴스 없음.

창 처리
- SMA n: rolling(n, min_periods=min(n,120)) → SMA200은 봉 120개부터. 보유 종목군은 봉 130개 이상이라 항상 있다.
- 252일 최고 고가: rolling(252, min_periods=120) 고가 최대.
- 수익률 r_k = C/C.shift(k)-1: 자료가 k+1봉보다 짧으면 NaN(그 재료에서 빠짐). RS는 r189·r252 중 하나라도 NaN이면 NaN.
- Q(클레노우 126일): ln C를 126개 연속 종가에 회귀한 기울기 b와 R²로 (exp(252b)-1)*R². 126봉 모두 있어야 값.
- V: ln(C/C.shift(126)) / (σ60·√126), σ60 = 일간 로그수익 60일 표준편차(min_periods=60).
- RSI14: Wilder(ewm alpha=1/14, adjust=False, min_periods=14). 재료는 -RSI(과열 회피 방향).
- 평활: 조합 점수 자체를 rolling(n, min_periods=1).mean() 한 뒤 그날 원점수가 있는 칸만 남긴다.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import harness as H  # noqa: E402

FAMILY = "composite"


def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=min(n, 120)).mean()


def clenow(C: pd.DataFrame, n: int = 126) -> pd.DataFrame:
    y = np.log(C.to_numpy(float))
    out = np.full(y.shape, np.nan)
    if len(y) >= n:
        w = np.lib.stride_tricks.sliding_window_view(y, n, axis=0)  # (T-n+1, N, n)
        x = np.arange(n, dtype=float) - (n - 1) / 2.0
        sxx = float(x @ x)
        ym = w.mean(axis=2, keepdims=True)
        yc = w - ym
        b = (yc * x).sum(axis=2) / sxx
        syy = (yc * yc).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(syy > 0, b * b * sxx / syy, np.nan)
        out[n - 1:] = (np.exp(252.0 * b) - 1.0) * r2
    return pd.DataFrame(out, index=C.index, columns=C.columns)


def rsi_wilder(C: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = C.diff()
    g = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    l_ = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = g / l_
    r = 100.0 - 100.0 / (1.0 + rs)
    r = r.where(~(l_ == 0) | g.isna(), 100.0)
    return r


def raw_ingredients(P: dict, need: set[str]) -> dict:
    C, Hi = P["C"], P["H"]
    out = {}
    s = {}

    def S(n):
        if n not in s:
            s[n] = sma(C, n)
        return s[n]

    def ret(k):
        return C / C.shift(k) - 1.0

    if "L" in need:
        out["L"] = C / S(200) - 1.0
    if "M" in need:
        out["M"] = C / S(50) - 1.0
    if "S" in need:
        out["S"] = C / S(20) - 1.0
    if "T" in need:
        ns = (10, 20, 50, 100, 200)
        cnt = sum((C > S(n)).astype(float) for n in ns)
        cnt = cnt + sum((S(a) > S(b)).astype(float) for a, b in zip(ns[:-1], ns[1:]))
        ok = C.notna()
        for n in ns:
            ok &= S(n).notna()
        out["T"] = cnt.where(ok)
    if "H" in need:
        hh = Hi.rolling(252, min_periods=120).max()
        out["H"] = C / hh - 1.0
    if "R" in need:
        out["R"] = ret(126)
    if "RS" in need:
        out["RS"] = 0.4 * ret(63) + 0.2 * ret(126) + 0.2 * ret(189) + 0.2 * ret(252)
    if "Q" in need:
        out["Q"] = clenow(C, 126)
    if "V" in need:
        lr = np.log(C / C.shift(1))
        sig = lr.rolling(60, min_periods=60).std()
        out["V"] = np.log(C / C.shift(126)) / (sig * np.sqrt(126.0))
    if "SR" in need:
        out["SR"] = -ret(5)
    if "RSI" in need:
        out["RSI"] = -rsi_wilder(C, 14)
    return out


def composite(P: dict, weights: dict) -> pd.DataFrame:
    raw = raw_ingredients(P, set(weights))
    um = H.universe_mask(P)
    num, den = None, None
    for k, w in weights.items():
        x = raw[k].replace([np.inf, -np.inf], np.nan)
        pr = H.pct_rank(x, um & x.notna())
        has = pr.notna().astype(float)
        num = pr.fillna(0.0) * w if num is None else num + pr.fillna(0.0) * w
        den = has * w if den is None else den + has * w
    return (num / den).where(den > 0)


def make(weights: dict, smooth: int = 0):
    def fn(P):
        sc = composite(P, weights)
        if smooth:
            sc = sc.rolling(smooth, min_periods=1).mean().where(sc.notna())
        return sc
    return fn


GRID = {
    "E1": {"L": 1, "M": 1, "S": 1},
    "E2": {"L": 1, "M": 1, "S": 1, "H": 1},
    "E3": {"T": 1, "H": 1},
    "E4": {"R": 1, "H": 1},
    "E5": {"L": 1, "Q": 1},
    "E6": {"L": 1, "H": 1, "Q": 1},
    "E7": {"L": 1, "M": 1, "S": 1, "H": 1, "Q": 1, "R": 1},
    "E8": {"RS": 1, "H": 1, "T": 1},
    "E9": {"L": 1, "M": 1, "SR": 0.5},
    "E10": {"R": 1, "H": 1, "SR": 0.5},
    "E11": {"L": 1, "M": 1, "S": 1, "H": 1, "SR": 0.5},
    "E12": {"V": 1, "H": 1},
    "E13": {"L": 1, "H": 1},
    "E14": {"L": 1, "M": 1, "H": 1, "Q": 1},
    "E15": {"L": 1, "T": 1},
    "E16": {"R": 1, "Q": 1},
    "E17": {"L": 1, "RSI": 1},
    "E18": {"L": 1, "M": 1, "RSI": 0.5},
}
SMOOTHED = ("E2", "E7", "E11", "E14")

# 격자 밖(결과를 본 뒤 추가, 선택편향 있음): front 1위 E6에 사전 등록 평활 규칙과 ½·SR 규칙을 그대로 적용.
EXTRAS = (
    ("extra_E6_s5", "E6 점수의 최근 5일 평균 (격자 밖, E6이 front 1위라 사후 추가)", GRID["E6"], 5),
    ("extra_E6_s10", "E6 점수의 최근 10일 평균 (격자 밖, 사후 추가)", GRID["E6"], 10),
    ("extra_E6_SR", "pct_rank 가중 평균 L×1, H×1, Q×1, SR×0.5 (격자 밖, 사후 추가)",
     {"L": 1, "H": 1, "Q": 1, "SR": 0.5}, 0),
)


def variants() -> list[tuple[str, str, object]]:
    out = []
    for name, w in GRID.items():
        spec = "pct_rank 가중 평균 " + ", ".join(f"{k}×{v}" for k, v in w.items())
        out.append((name, spec, make(w)))
    for name in SMOOTHED:
        for n in (5, 10):
            w = GRID[name]
            spec = f"{name} 점수의 최근 {n}일 평균(rolling min_periods=1, 그날 원점수 있는 칸만)"
            out.append((f"{name}_s{n}", spec, make(w, n)))
    for name, spec, w, n in EXTRAS:
        out.append((name, spec, make(w, n)))
    return out


def brief(r: dict) -> dict:
    g = lambda w, h, k: r.get(w, {}).get(h, {}).get(k)
    return {
        "front_ic20": g("front", "h20", "ic"), "front_t20": g("front", "h20", "ic_t"),
        "front_top20": g("front", "h20", "top20"),
        "back_ic20": g("back", "h20", "ic"), "back_t20": g("back", "h20", "ic_t"),
        "back_top20": g("back", "h20", "top20"),
        "y2025_ic20": g("y2025", "h20", "ic"), "y2024_ic20": g("y2024", "h20", "ic"),
        "late_ic10": g("late", "h10", "ic"),
        "portS_front": r.get("front", {}).get("port_S", {}).get("ret"),
        "portS_back": r.get("back", {}).get("port_S", {}).get("ret"),
        "turnover_front": r.get("front", {}).get("turnover"),
    }


if __name__ == "__main__":
    P = H.load_panel()
    rows = []
    for name, spec, fn in variants():
        t0 = time.time()
        nl = H.no_lookahead(fn)
        res = H.evaluate(fn(P))
        rows.append({"name": name, "spec": spec, "result": res, "no_lookahead": bool(nl)})
        print(H.summary_line(name + ("" if nl else " [LOOKAHEAD]"), res),
              f"| turn {res['front']['turnover']} | {time.time() - t0:.1f}s", flush=True)
    path = H.save_results(FAMILY, rows)
    print("saved", os.path.abspath(path))
