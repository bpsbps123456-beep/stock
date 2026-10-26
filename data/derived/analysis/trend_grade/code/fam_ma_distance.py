"""추세 등급 계열 'ma_distance' — 이평선과의 거리·이평선 간격·기울기(연속 점수).

사전 등록 격자 37개(B1~B5) + 격자 밖 extra_ 6개(결과를 보기 전에 이 파일에 함께 정의해 한 번에 잰다).
공통 채점기 harness.py를 그대로 쓴다. 가격(OHLC)과 이평선만 쓴다(거래량·재무·뉴스 없음).

창 처리
    SMA_n = C.rolling(n, min_periods=min(n,120)).mean()  → 상장 120~n봉 종목은 그동안의 전체 평균(확장 평균)
    EMA_n = C.ewm(span=n, adjust=False, min_periods=min(n,120)).mean()  → 첫 종가에서 시작
    σ60   = 일 로그수익 rolling(60, min_periods=60).std(ddof=1)
    ATR20 = 진폭 rolling(20, min_periods=20).mean(), 진폭 = fmax(H, C전일) − fmin(L, C전일)
            (전일 종가가 없는 첫 봉은 H−L)
    단면 백분위 = H.pct_rank(x, H.universe_mask(P) & x.notna())  (t일까지 아는 종목군)
    리본 평균은 네 성분이 모두 있을 때만 값(하나라도 NaN이면 NaN).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import harness as H

NS_B1 = (10, 20, 50, 100, 150, 200)
NS = (20, 50, 100, 200)


def sma(C: pd.DataFrame, n: int) -> pd.DataFrame:
    return C.rolling(n, min_periods=min(n, 120)).mean()


def ema(C: pd.DataFrame, n: int) -> pd.DataFrame:
    return C.ewm(span=n, adjust=False, min_periods=min(n, 120)).mean()


def sigma60(C: pd.DataFrame) -> pd.DataFrame:
    r = np.log(C).diff()
    return r.rolling(60, min_periods=60).std(ddof=1)


def atr20(P: dict) -> pd.DataFrame:
    C, Hh, L = P["C"], P["H"], P["L"]
    cp = C.shift(1)
    tr = pd.DataFrame(np.fmax(Hh.to_numpy(float), cp.to_numpy(float)) - np.fmin(L.to_numpy(float), cp.to_numpy(float)),
                      index=C.index, columns=C.columns)
    return tr.rolling(20, min_periods=20).mean()


def z_dist(P: dict, n: int) -> pd.DataFrame:
    C = P["C"]
    return (np.log(C) - np.log(sma(C, n))) / sigma60(C)


def xs_pct(P: dict, x: pd.DataFrame) -> pd.DataFrame:
    return H.pct_rank(x, H.universe_mask(P) & x.notna())


def mean_strict(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0].copy()
    for f in frames[1:]:
        out = out + f
    return out / len(frames)


# ---------------------------------------------------------------- 변형 정의
def build_variants() -> list[tuple[str, str, callable]]:
    v = []
    # B1 C/MA_n − 1
    for kind, fn in (("SMA", sma), ("EMA", ema)):
        for n in NS_B1:
            v.append((f"B1_C_{kind}{n}", f"C/{kind}{n}-1; {kind} min_periods=min(n,120)",
                      (lambda P, n=n, fn=fn: P["C"] / fn(P["C"], n) - 1)))
    # B2 변동성으로 나눈 거리
    for n in NS:
        v.append((f"B2_z_SMA{n}", f"(lnC-lnSMA{n})/sigma60; sigma60=log-ret rolling60 std ddof1",
                  (lambda P, n=n: z_dist(P, n))))
    for n in NS:
        v.append((f"B2_atr_SMA{n}", f"(C-SMA{n})/ATR20; TR=fmax(H,Cprev)-fmin(L,Cprev), ATR20=SMA20(TR)",
                  (lambda P, n=n: (P["C"] - sma(P["C"], n)) / atr20(P))))
    # B3 리본
    v.append(("B3_ribbon_raw", "mean_n(C/SMA_n-1), n=20,50,100,200; NaN if any missing",
              lambda P: mean_strict([P["C"] / sma(P["C"], n) - 1 for n in NS])))
    v.append(("B3_ribbon_pct", "mean_n(xs pct_rank(C/SMA_n-1) within universe_mask), n=20,50,100,200",
              lambda P: mean_strict([xs_pct(P, P["C"] / sma(P["C"], n) - 1) for n in NS])))
    v.append(("B3_ribbon_z", "mean_n((lnC-lnSMA_n)/sigma60), n=20,50,100,200",
              lambda P: mean_strict([z_dist(P, n) for n in NS])))
    # B4 이평 간격
    for a, b in ((20, 50), (50, 200), (10, 50), (20, 200)):
        v.append((f"B4_SMA{a}_SMA{b}", f"SMA{a}/SMA{b}-1",
                  (lambda P, a=a, b=b: sma(P["C"], a) / sma(P["C"], b) - 1)))
    v.append(("B4_macd_C", "(EMA12-EMA26)/C",
              lambda P: (ema(P["C"], 12) - ema(P["C"], 26)) / P["C"]))
    v.append(("B4_macd_C_sig", "((EMA12-EMA26)/C)/sigma60",
              lambda P: (ema(P["C"], 12) - ema(P["C"], 26)) / P["C"] / sigma60(P["C"])))
    # B5 이평 기울기
    for n in NS:
        for k in (5, 20):
            v.append((f"B5_slope_SMA{n}_k{k}", f"SMA{n}/SMA{n}.shift({k})-1",
                      (lambda P, n=n, k=k: sma(P["C"], n) / sma(P["C"], n).shift(k) - 1)))

    # 격자 밖(extra_, 사전 정의 6개)
    def stack_align(P):
        C = P["C"]
        s = {n: sma(C, n) for n in NS}
        cnt = ((C > s[20]).astype(float) + (s[20] > s[50]).astype(float)
               + (s[50] > s[100]).astype(float) + (s[100] > s[200]).astype(float))
        cnt = cnt.where(C.notna() & s[200].notna())
        tie = mean_strict([xs_pct(P, C / s[n] - 1) for n in NS])
        return cnt + 0.5 * tie  # 정배열 단계(0~4)가 우선, 같은 단계 안에서 리본 백분위

    v.append(("extra_stack_align", "정배열 수(C>S20,S20>S50,S50>S100,S100>S200) + 0.5*B3_ribbon_pct(동점 깨기)",
              stack_align))
    v.append(("extra_z50_x_slope50", "mean(xs pct(z_SMA50), xs pct(SMA50/SMA50.shift20-1))",
              lambda P: mean_strict([xs_pct(P, z_dist(P, 50)),
                                     xs_pct(P, sma(P["C"], 50) / sma(P["C"], 50).shift(20) - 1)])))
    def min_z(P):
        zs = [z_dist(P, n) for n in NS]
        out = zs[0]
        for z in zs[1:]:
            out = out.where(out <= z, z)  # NaN이 섞이면 아래 mask로 지운다
        return out.where(mean_strict(zs).notna())

    v.append(("extra_min_z", "min_n z_SMA_n, n=20,50,100,200 (모든 이평 위 거리의 최솟값); NaN if any missing",
              min_z))
    v.append(("extra_slope50_sig", "(SMA50/SMA50.shift20-1)/sigma60",
              lambda P: (sma(P["C"], 50) / sma(P["C"], 50).shift(20) - 1) / sigma60(P["C"])))
    v.append(("extra_ema_ribbon_pct", "mean_n(xs pct(C/EMA_n-1)), n=20,50,100,200",
              lambda P: mean_strict([xs_pct(P, P["C"] / ema(P["C"], n) - 1) for n in NS])))
    v.append(("extra_ribbon_z_long", "mean_n z_SMA_n, n=50,100,200 (짧은 20일 제외)",
              lambda P: mean_strict([z_dist(P, n) for n in (50, 100, 200)])))
    return v


def main():
    P = H.load_panel()
    variants = build_variants()
    print("variants", len(variants), "grid", sum(not n.startswith("extra_") for n, _, _ in variants))
    rows = []
    for name, spec, fn in variants:
        t0 = time.time()
        sc = fn(P)
        nl = H.no_lookahead(fn)
        res = H.evaluate(sc)
        rows.append({"name": name, "spec": spec, "result": res, "no_lookahead": bool(nl)})
        print(H.summary_line(name, res), "| NL", nl, f"| {time.time() - t0:.1f}s", flush=True)
    path = H.save_results("ma_distance", rows)
    print("saved", path)


if __name__ == "__main__":
    main()
