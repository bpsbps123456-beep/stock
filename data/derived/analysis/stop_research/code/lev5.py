"""5단계 레버리지 긴 역사 도우미(2026-10-11): cache/p5_lev_long.parquet로 3배 상품을 기초 지수에서 1990년대부터 합성한다.

synth(under, L=3, fee, extra): lev_lab.synth와 같은 식(일일 재설정, 금융비용 ^IRX, 보수, 추가 비용 extra는 연 로그 비용).
splice(...): 두 기초를 이어 붙인다(예: 2001-07 전 ^SOX, 뒤 SOXX). 이어 붙인 날의 수익은 앞 기초 기준.
validate(product, bars): 실제 상품과 겹치는 기간의 일수익 상관, 연 로그 비용 차(−ln(실제/합성)/년), 최대 낙폭.
const_path(bars, w): 상품을 w만큼(나머지 현금 0%) 매일 맞춰 든 종가 경로 = '덜 담기' 효율선 재료.
반환 Bars는 stock_lab.Bars라 stock_lab.simulate/Policy(비율 목표·장중 손절)를 그대로 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stock_lab as S
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
_RAW = None


def raw() -> pd.DataFrame:
    global _RAW
    if _RAW is None:
        _RAW = pd.read_parquet(CACHE / "p5_lev_long.parquet")
    return _RAW


def bars(sym: str) -> S.Bars:
    g = raw()
    g = g[g.sym == sym].sort_values("date").dropna(subset=["Close"])
    g = g[g.Close > 0]
    f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
    o, h, l = (g[c].fillna(g.Close).to_numpy() for c in ("Open", "High", "Low"))
    return S.Bars(sym, g.date.to_numpy(), o * f, h * f, l * f, g.Close.to_numpy() * f, g.Volume.fillna(0).to_numpy(float))


def rf(days) -> np.ndarray:
    g = raw()
    s = g[g.sym == "^IRX"].set_index("date")["Close"].astype(float).sort_index()
    s = s[~s.index.duplicated()] / 100.0
    return s.reindex(sorted(set(s.index) | set(days))).ffill().reindex(days).fillna(0.0).to_numpy()


def splice(first: str, second: str, at: str) -> S.Bars:
    """at 전에는 first, at부터는 second의 일수익으로 이은 가상 기초(가격 수준은 second에 맞춤)."""
    a, b = bars(first), bars(second)
    ia, ib = a.idx(at), b.idx(at)
    # second의 at 이후를 그대로 쓰고, 그 전은 first의 수익률로 거꾸로 이어 붙인다
    scale = b.c[ib] / a.c[ia]
    cat = lambda x, y: np.concatenate([x[:ia] * scale, y[ib:]])
    return S.Bars(f"{first}+{second}", np.concatenate([a.days[:ia], b.days[ib:]]), cat(a.o, b.o), cat(a.h, b.h),
                  cat(a.l, b.l), cat(a.c, b.c), np.concatenate([a.v[:ia], b.v[ib:]]))


def synth(u: S.Bars, L: float = 3.0, fee: float = 0.0095, extra: float = 0.0) -> S.Bars:
    r_f = rf(u.days)
    n = len(u.c)
    c = np.empty(n); o = np.empty(n); h = np.empty(n); l = np.empty(n)
    c[0] = o[0] = h[0] = l[0] = 100.0
    for t in range(1, n):
        pc = u.c[t - 1]
        r = L * (u.c[t] / pc - 1) - (L - 1) * r_f[t] / 252 - (fee + extra) / 252
        c[t] = c[t - 1] * max(1 + r, 0.001)
        o[t] = c[t - 1] * max(1 + L * (u.o[t] / pc - 1), 0.001)
        hi = c[t - 1] * max(1 + L * (u.h[t] / pc - 1), 0.001)
        lo = c[t - 1] * max(1 + L * (u.l[t] / pc - 1), 0.001)
        h[t], l[t] = max(hi, o[t], c[t]), min(lo, o[t], c[t])
    return S.Bars(f"{u.sym}x{L:g}", u.days.copy(), o, h, l, c, u.v.copy())


def validate(product: str, syn: S.Bars) -> dict:
    a = bars(product)
    common = np.intersect1d(a.days, syn.days)
    ia, isy = np.searchsorted(a.days, common), np.searchsorted(syn.days, common)
    ra, rs = a.c[ia][1:] / a.c[ia][:-1] - 1, syn.c[isy][1:] / syn.c[isy][:-1] - 1
    yrs = len(common) / 252
    ca, cs = a.c[ia][-1] / a.c[ia][0], syn.c[isy][-1] / syn.c[isy][0]
    mdd = lambda v: float((v / np.maximum.accumulate(v) - 1).min())
    return {"product": product, "시작": str(common[0]), "일수": int(len(common)), "일수익상관": float(np.corrcoef(ra, rs)[0, 1]),
            "연비용_로그": float(-np.log(ca / cs) / yrs), "실제_낙폭": mdd(a.c[ia]), "합성_낙폭": mdd(syn.c[isy])}


def const_path(b: S.Bars, w: float) -> np.ndarray:
    """상품을 매일 비율 w로 맞춰 든 종가 가치(시작 1)."""
    r = np.concatenate([[0.0], b.c[1:] / b.c[:-1] - 1])
    return np.cumprod(1 + w * r)


def check():
    sox = splice("^SOX", "SOXX", "2001-07-16")
    s3 = synth(sox, 3)
    k3 = synth(bars("EWY"), 3)
    for name, prod, syn in (("SOXL ← ^SOX+SOXX", "SOXL", s3), ("KORU ← EWY", "KORU", k3)):
        v = validate(prod, syn)
        print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    for name, syn in (("SOX 3배 합성", s3), ("EWY 3배 합성", k3)):
        c = syn.c
        print(name, syn.days[0], "~", syn.days[-1], "최대 낙폭", round(float((c / np.maximum.accumulate(c) - 1).min()), 4),
              "| 2000-03~2002-10 낙폭", round(float(c[syn.idx("2002-10-09")] / c[:syn.idx("2002-10-09")].max() - 1), 4)
              if syn.days[0] < "2000-03-01" else "")


if __name__ == "__main__":
    check()
