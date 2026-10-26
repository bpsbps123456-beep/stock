"""추세 등급 계열 'trend_quality' — 추세의 매끄러움·일관성 (2026-10-12).

사전 등록 격자 25개(D1~D6)와 결과를 보기 전에 정한 extra_ 3개를 공통 채점기(harness.py)로 잰다.
모든 점수는 t일 종가까지의 OHLC만 쓴다(거래량·재무·뉴스 미사용). 패널에 상장 뒤 빈 봉은 없다.

NaN 처리
- 회귀·수익률 창(D1·D2·D4·D5 ret): 창 길이만큼 자료가 없으면 NaN(그 점수에서 그 종목 제외).
- 이평선·비율 창(D3·D5 MDD·D6): rolling(n, min_periods=min(n, 120)).
- 단면 백분위: H.pct_rank(x, H.universe_mask(P) & 모든 구성 요소 notna).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import harness as H


# ---------------------------------------------------------------- 공통 계산

def mp(n: int) -> int:
    return min(n, 120)


def rolling_ols(C: pd.DataFrame, n: int, min_obs: int | None = None):
    """ln C를 거래일 번호에 회귀. 창 안 유효 관측 수 >= min_obs(기본 n)일 때만 값.
    반환: 일 기울기 b, R²."""
    min_obs = n if min_obs is None else min_obs
    y = np.log(C)
    valid = y.notna()
    x = pd.DataFrame(np.arange(len(C), dtype=float)[:, None].repeat(C.shape[1], 1),
                     index=C.index, columns=C.columns).where(valid)
    r = lambda d: d.rolling(n, min_periods=1).sum()
    cnt = valid.astype(float).rolling(n, min_periods=1).sum()
    sx, sy = r(x), r(y)
    sxx, syy, sxy = r(x * x), r(y * y), r(x * y)
    Sxx = sxx - sx * sx / cnt
    Syy = syy - sy * sy / cnt
    Sxy = sxy - sx * sy / cnt
    ok = cnt >= min_obs
    b = (Sxy / Sxx).where(ok)
    r2 = (Sxy * Sxy / (Sxx * Syy)).where(ok & (Syy > 0))
    return b, r2.clip(0, 1)


def ret(C: pd.DataFrame, n: int) -> pd.DataFrame:
    return C / C.shift(n) - 1.0


def xs_pct(P: dict, *parts: pd.DataFrame) -> list[pd.DataFrame]:
    """같은 마스크(유니버스 & 모든 구성 요소 존재)로 각 요소의 단면 백분위."""
    m = H.universe_mask(P)
    for p in parts:
        m = m & p.notna()
    return [H.pct_rank(p, m) for p in parts]


def rolling_mdd(C: pd.DataFrame, n: int) -> pd.DataFrame:
    """최근 n일(t 포함) 종가 경로 안의 최대 낙폭(양수, 0~1). 창 안 유효 봉 >= min(n,120)."""
    a = C.to_numpy(float)
    T, K = a.shape
    out = np.full((T, K), np.nan)
    need = mp(n)
    for t in range(T):
        w = a[max(0, t - n + 1): t + 1]
        cnt = np.isfinite(w).sum(0)
        peak = np.fmax.accumulate(w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = 1.0 - w / peak
        dd = np.where(np.isfinite(dd), dd, -np.inf)
        m = dd.max(0)
        m = np.where(cnt >= need, m, np.nan)
        out[t] = m
    return pd.DataFrame(out, index=C.index, columns=C.columns)


def frac_true(ind: pd.DataFrame, valid: pd.DataFrame, n: int) -> pd.DataFrame:
    """최근 n일 중 ind가 참인 날 비율(valid인 날만 센다, 유효일 >= min(n,120))."""
    v = ind.astype(float).where(valid)
    return v.rolling(n, min_periods=mp(n)).mean()


# ---------------------------------------------------------------- D1 클레노우

def clenow(n: int, min_obs: int | None = None):
    def f(P):
        b, r2 = rolling_ols(P["C"], n, min_obs)
        return (np.exp(252 * b) - 1.0) * r2
    return f


def slope_only(n: int):
    def f(P):
        b, _ = rolling_ols(P["C"], n)
        return b
    return f


def r2_signed(n: int):
    def f(P):
        b, r2 = rolling_ols(P["C"], n)
        return r2 * np.sign(b)
    return f


# ---------------------------------------------------------------- D2 효율비

def er(C: pd.DataFrame, n: int) -> pd.DataFrame:
    net = C - C.shift(n)
    path = C.diff().abs().rolling(n, min_periods=n).sum()
    e = (net.abs() / path).where(path > 0)
    return e.where(net.notna()), net


def er_signed(n: int):
    def f(P):
        e, net = er(P["C"], n)
        return np.sign(net) * e
    return f


def er_x_ret(n: int):
    def f(P):
        e, _ = er(P["C"], n)
        return e * ret(P["C"], n)
    return f


# ---------------------------------------------------------------- D3 이평 위 날 비율

def above_ma(ma: int, n: int):
    def f(P):
        C = P["C"]
        sma = C.rolling(ma, min_periods=mp(ma)).mean()
        return frac_true(C > sma, sma.notna() & C.notna(), n)
    return f


# ---------------------------------------------------------------- D4 frog-in-pan

def fip(look: int, skip: int = 21):
    """r = C[t-skip]/C[t-look] - 1, ID = sign(r)·(%음수일 − %양수일) (같은 창의 일 수익 look-skip개).
    점수 = 단면 백분위(r) − 단면 백분위(ID)."""
    def f(P):
        C = P["C"]
        r = C.shift(skip) / C.shift(look) - 1.0
        d = C.pct_change(fill_method=None)
        k = look - skip
        neg = (d < 0).astype(float).where(d.notna()).rolling(k, min_periods=k).mean().shift(skip)
        pos = (d > 0).astype(float).where(d.notna()).rolling(k, min_periods=k).mean().shift(skip)
        idv = np.sign(r) * (neg - pos)
        idv = idv.where(r.notna())
        pr_r, pr_id = xs_pct(P, r, idv)
        return pr_r - pr_id
    return f


# ---------------------------------------------------------------- D5 낙폭

def neg_mdd(n: int):
    def f(P):
        return -rolling_mdd(P["C"], n)
    return f


def ret126_plus_negmdd126(P):
    r = ret(P["C"], 126)
    m = -rolling_mdd(P["C"], 126)
    a, b = xs_pct(P, r, m)
    return a + b


# ---------------------------------------------------------------- D6 상승일 비율

def up_days(n: int):
    def f(P):
        C = P["C"]
        prev = C.shift(1)
        return frac_true(C > prev, C.notna() & prev.notna(), n)
    return f


# ---------------------------------------------------------------- extra_ (결과 보기 전 등록)

def clenow90_filtered(P):
    """클레노우 원형 필터: C < SMA100 이거나 90일 안 |일 수익| > 15%면 맨 아래로(점수 − 100, 순서는 유지)."""
    C = P["C"]
    s = clenow(90)(P)
    sma = C.rolling(100, min_periods=100).mean()
    gap = C.pct_change(fill_method=None).abs().rolling(90, min_periods=90).max()
    bad = (C < sma) | (gap > 0.15)
    return s.where(~bad, s - 100.0).where(sma.notna() & gap.notna())


REGISTRY = [
    # D1 클레노우 (회귀 창 전체 n개 봉 필요, 짧으면 NaN)
    ("D1_clenow63", "(exp(252b)-1)*R2, ln C ~ 거래일 OLS, N=63, 63봉 전부 필요", clenow(63)),
    ("D1_clenow90", "(exp(252b)-1)*R2, N=90, 90봉 전부 필요", clenow(90)),
    ("D1_clenow126", "(exp(252b)-1)*R2, N=126, 126봉 전부 필요", clenow(126)),
    ("D1_clenow252", "(exp(252b)-1)*R2, N=252, 252봉 전부 필요(상장 252봉 미만 NaN)", clenow(252)),
    ("D1_slope63", "OLS 기울기 b만, N=63", slope_only(63)),
    ("D1_slope126", "OLS 기울기 b만, N=126", slope_only(126)),
    ("D1_r2sign63", "R2*sign(b), N=63", r2_signed(63)),
    ("D1_r2sign126", "R2*sign(b), N=126", r2_signed(126)),
    # D2 효율비
    ("D2_er20_signed", "sign(C-C[-20])*|C-C[-20]|/sum20|dC|", er_signed(20)),
    ("D2_er60_signed", "sign(C-C[-60])*ER60", er_signed(60)),
    ("D2_er120_signed", "sign(C-C[-120])*ER120", er_signed(120)),
    ("D2_er20_x_ret20", "ER20*(C/C[-20]-1)", er_x_ret(20)),
    ("D2_er60_x_ret60", "ER60*(C/C[-60]-1)", er_x_ret(60)),
    ("D2_er120_x_ret120", "ER120*(C/C[-120]-1)", er_x_ret(120)),
    # D3 이평 위 날 비율 (SMA min_periods=min(n,120), 비율 창 min_periods=창 길이)
    ("D3_aboveSMA50_20d", "최근 20일 중 C>SMA50 비율", above_ma(50, 20)),
    ("D3_aboveSMA50_60d", "최근 60일 중 C>SMA50 비율", above_ma(50, 60)),
    ("D3_aboveSMA50_120d", "최근 120일 중 C>SMA50 비율(유효 120일 필요 → 상장 169봉부터)", above_ma(50, 120)),
    ("D3_aboveSMA20_20d", "최근 20일 중 C>SMA20 비율", above_ma(20, 20)),
    ("D3_aboveSMA20_60d", "최근 60일 중 C>SMA20 비율", above_ma(20, 60)),
    # D4 frog-in-pan
    ("D4_fip252_skip21", "pct(r)-pct(ID), r=C[-21]/C[-252]-1, ID=sign(r)(%neg-%pos) 같은 창 231일, 253봉 미만 NaN", fip(252, 21)),
    ("D4_fip126_skip21", "pct(r)-pct(ID), r=C[-21]/C[-126]-1, ID 같은 창 105일", fip(126, 21)),
    # D5 낙폭 (종가 경로, 창 min_periods=min(n,120))
    ("D5_negMDD63", "-(최근 63일 종가 최대 낙폭)", neg_mdd(63)),
    ("D5_negMDD126", "-(최근 126일 종가 최대 낙폭), 유효 120봉 이상", neg_mdd(126)),
    ("D5_pct_ret126_plus_negMDD126", "pct(C/C[-126]-1)+pct(-MDD126)", ret126_plus_negmdd126),
    # D6
    ("D6_updays60", "최근 60일 중 종가 상승일 비율", up_days(60)),
    # extra_ (격자 밖, 결과 보기 전 등록)
    ("extra_clenow252_mp120", "클레노우 N=252, 창 안 유효 120봉 이상이면 부분 창 회귀(신규 주도주 포함)", clenow(252, 120)),
    ("extra_clenow90_filt", "클레노우 N=90 + 원형 필터(C<SMA100 또는 90일 |일수익|>15%면 −100, 순서 유지)", clenow90_filtered),
    ("extra_fip126_noskip", "frog-in-pan, r=C/C[-126]-1, ID 같은 창 126일(최근 21일 건너뛰지 않음)", fip(126, 0)),
]


def main(only: list[str] | None = None) -> None:
    P = H.load_panel()
    rows = []
    for name, spec, fn in REGISTRY:
        if only and name not in only:
            continue
        t0 = time.time()
        sc = fn(P)
        nl = bool(H.no_lookahead(fn))
        res = H.evaluate(sc)
        rows.append({"name": name, "spec": spec, "result": res, "no_lookahead": nl})
        print(("" if nl else "[실격 lookahead] ") + H.summary_line(name, res), f"({time.time() - t0:.1f}s)", flush=True)
    if not only:
        print(H.save_results("trend_quality", rows))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
