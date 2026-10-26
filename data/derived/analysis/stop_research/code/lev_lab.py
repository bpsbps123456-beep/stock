"""4단계 레버리지 상품 엔진(2026-10-11): 일일 재설정 레버리지 ETF를 기초자산으로 합성하고 실제 상품과 대조한다.

사용자: "특히 레버리지 주식도 함." 실제 상품(NVDL 2022-12~, TSLL 2022-08~, MSTU 2024-09~ 등)은 역사가 짧아
2018·2020·2022 폭락을 담지 못한다. 그래서 기초자산(NVDA·TSLA·QQQ·SOXX…)으로 2014년부터 합성한다.

합성(일일 재설정):
  r_L[t] = L × r_u[t] − (L − 1) × rf[t]/252 − fee/252   (r_u = 기초 종가 수익, rf = ^IRX/100(13주 국채), fee = 연 보수)
  시가·고가·저가 = 전일 합성 종가 × (1 + L × (기초 시가·고가·저가 / 기초 전일 종가 − 1)). 저가는 0.01배 아래로 내리지 않는다.
  실제 상품은 스왑 비용·현금 배당·괴리 때문에 조금 다르다 → validate()가 겹치는 기간의 일수익 상관·연 추적차·최대 낙폭 차를 보인다.
반환은 stock_lab.Bars라서 stock_lab.simulate·Policy, tranche_lab.simulate를 그대로 쓸 수 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_lab as S
from trade_lab import _bars_from

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
FEE = {2: 0.0105, 3: 0.0095}       # 대표 보수(연). 단일 종목 2배 ≈ 1.05~1.3%, 지수 3배 ≈ 0.9~1.0%

_LEV = None


def load() -> dict:
    """p4_lev.parquet의 실제 상품·기초자산·참고 지수 Bars(배당 수정)."""
    global _LEV
    if _LEV is None:
        df = pd.read_parquet(CACHE / "p4_lev.parquet")
        df = df[df.sym != "^IRX"]
        out = {}
        for sym, g in df.sort_values(["sym", "date"]).groupby("sym"):
            g = g.dropna(subset=["Open", "High", "Low", "Close"])
            g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0)]
            if len(g) < 30:
                continue
            f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
            out[sym] = S.Bars(sym, g.date.to_numpy(), g.Open.to_numpy() * f, g.High.to_numpy() * f,
                              g.Low.to_numpy() * f, g.Close.to_numpy() * f, g.Volume.to_numpy(float))
        _LEV = out
    return _LEV


def products() -> dict:
    return json.loads((CACHE / "p4_meta.json").read_text(encoding="utf-8"))["lev"]


def rf_series() -> pd.Series:
    df = pd.read_parquet(CACHE / "p4_lev.parquet")
    s = df[df.sym == "^IRX"].set_index("date")["Close"].astype(float).sort_index()
    return s[~s.index.duplicated()] / 100.0


def synth(under: str, L: float, fee: float | None = None) -> S.Bars:
    u = load()[under]
    fee = FEE.get(int(L), 0.01) if fee is None else fee
    rf = rf_series().reindex(u.days).ffill().fillna(0.0).to_numpy()
    n = len(u.c)
    c = np.empty(n); o = np.empty(n); h = np.empty(n); l = np.empty(n)
    c[0] = o[0] = h[0] = l[0] = 100.0
    for t in range(1, n):
        pc = u.c[t - 1]
        r = L * (u.c[t] / pc - 1) - (L - 1) * rf[t] / 252 - fee / 252
        c[t] = c[t - 1] * max(1 + r, 0.01)
        o[t] = c[t - 1] * max(1 + L * (u.o[t] / pc - 1), 0.01)
        hi = c[t - 1] * max(1 + L * (u.h[t] / pc - 1), 0.01)
        lo = c[t - 1] * max(1 + L * (u.l[t] / pc - 1), 0.01)
        h[t], l[t] = max(hi, o[t], c[t]), min(lo, o[t], c[t])
    return S.Bars(f"{under}x{L:g}", u.days.copy(), o, h, l, c, u.v.copy())


def validate(product: str) -> dict:
    """실제 상품과 합성의 겹치는 기간 비교."""
    info = products()[product]
    a = load().get(product)
    if a is None:
        return {"product": product, "자료": "없음"}
    s = synth(info["under"], info["L"])
    common = np.intersect1d(a.days, s.days)
    if len(common) < 60:
        return {"product": product, "자료": f"겹침 {len(common)}일"}
    ia, is_ = np.searchsorted(a.days, common), np.searchsorted(s.days, common)
    ra, rs = a.c[ia][1:] / a.c[ia][:-1] - 1, s.c[is_][1:] / s.c[is_][:-1] - 1
    yrs = len(common) / 252
    ca, cs = a.c[ia][-1] / a.c[ia][0], s.c[is_][-1] / s.c[is_][0]
    mdd = lambda v: float((v / np.maximum.accumulate(v) - 1).min())
    # 연 비용은 비율(로그) 기준으로 낸다. CAGR 차(%p)는 고성장 구간에서 비용을 부풀린다(NVDL CAGR 차 −47%p ≈ 로그 비용 연 18%).
    return {"product": product, "L": info["L"], "under": info["under"], "시작": str(common[0]), "일수": int(len(common)),
            "일수익상관": float(np.corrcoef(ra, rs)[0, 1]), "연비용_로그": float(-np.log(ca / cs) / yrs),
            "CAGR차_참고": float(ca ** (1 / yrs) - cs ** (1 / yrs)),
            "실제_누적": float(ca - 1), "합성_누적": float(cs - 1), "실제_낙폭": mdd(a.c[ia]), "합성_낙폭": mdd(s.c[is_])}


def check():
    rows = [validate(p) for p in sorted(products())]
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    check()
