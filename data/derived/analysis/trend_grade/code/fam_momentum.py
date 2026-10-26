"""추세 등급 연구 — 계열 'momentum' (기간 수익률·상대강도·고점 근접), 2026-10-12.

사전 등록 격자 32개(C1~C5)와 격자 밖 extra_ 4개를 공통 채점기(harness.py)로 잰다.
extra_ 4개는 결과를 보기 전에 정했다. 가격(OHLC)과 QQQ 종가만 쓴다.

창 처리 규칙
- 수익률 창(C1·C2·C3·C4의 L창): 자료가 창보다 짧으면 NaN(그 종목은 그 점수에서 빠짐).
- 고점 창(C5 252일): rolling(252, min_periods=120) → 상장 120봉 이후 신규 종목도 들어감.
- σ60: 일 로그수익 rolling(60, min_periods=60) 표본 표준편차(ddof=1).
- 잔차 회귀 창 W: rolling(W, min_periods=min(W,120)).

실행: cd 이 폴더 && PYTHONIOENCODING=utf-8 py fam_momentum.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import harness as H

FAMILY = "momentum"
HI_MIN = 120  # 고점·회귀 창 최소 봉 수


# ---------- 공통 조각 ----------
def ret(C: pd.DataFrame, L: int, s: int = 0) -> pd.DataFrame:
    return C.shift(s) / C.shift(s + L) - 1.0


def logret(C: pd.DataFrame) -> pd.DataFrame:
    return np.log(C).diff()


def sigma60(C: pd.DataFrame) -> pd.DataFrame:
    return logret(C).rolling(60, min_periods=60).std()


def roll_max(X: pd.DataFrame, n: int = 252) -> pd.DataFrame:
    return X.rolling(n, min_periods=min(n, HI_MIN)).max()


# ---------- C1 기간 수익률 ----------
def make_ret(L: int, s: int):
    return lambda P: ret(P["C"], L, s)


# ---------- C2 변동성 조정 ----------
def make_voladj(L: int):
    def f(P):
        C = P["C"]
        return np.log(C / C.shift(L)) / (sigma60(C) * np.sqrt(L))
    return f


# ---------- C3 IBD RS ----------
def ibd_rs(C: pd.DataFrame, s: int = 0) -> pd.DataFrame:
    return 0.4 * ret(C, 63, s) + 0.2 * ret(C, 126, s) + 0.2 * ret(C, 189, s) + 0.2 * ret(C, 252, s)


def make_ibd(s: int):
    return lambda P: ibd_rs(P["C"], s)


# ---------- C4 잔차 모멘텀 ----------
def resid_mom(P: dict, W: int, L: int) -> pd.DataFrame:
    """t일 창 W로 OLS(r_i = a + b·r_QQQ) → 그 a·b로 최근 L일 잔차 e를 만든다.
    점수 = sum(e) / std(e, ddof=1). 닫힌 식(정확, 근사 아님):
      sum(e) = Σr − L·a − b·Σm
      var(e) = var_L(r) + b²·var_L(m) − 2b·cov_L(r,m)   (ddof=1, 절편은 분산에 무관)
    m은 종목 r이 있는 날로만 맞춘다(짝 결측 제거)."""
    C = P["C"]
    r = logret(C)
    mq = np.log(P["bench"]["C"]).diff().reindex(C.index)
    m = pd.DataFrame(np.repeat(mq.to_numpy()[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)
    m = m.where(r.notna())
    r = r.where(m.notna())
    mpW = min(W, HI_MIN)
    rw, mw = r.rolling(W, min_periods=mpW), m.rolling(W, min_periods=mpW)
    mr, mm = rw.mean(), mw.mean()
    mrm = (r * m).rolling(W, min_periods=mpW).mean()
    mmm = (m * m).rolling(W, min_periods=mpW).mean()
    cov_w = mrm - mr * mm
    var_w = mmm - mm * mm
    b = cov_w / var_w
    a = mr - b * mm
    sr = r.rolling(L, min_periods=L).sum()
    sm = m.rolling(L, min_periods=L).sum()
    srr = (r * r).rolling(L, min_periods=L).sum()
    smm = (m * m).rolling(L, min_periods=L).sum()
    srm = (r * m).rolling(L, min_periods=L).sum()
    var_r = (srr - sr * sr / L) / (L - 1)
    var_m = (smm - sm * sm / L) / (L - 1)
    cov_rm = (srm - sr * sm / L) / (L - 1)
    sum_e = sr - L * a - b * sm
    var_e = var_r + b * b * var_m - 2 * b * cov_rm
    return sum_e / np.sqrt(var_e.where(var_e > 0))


def make_resid(W: int, L: int):
    return lambda P: resid_mom(P, W, L)


def _check_resid(P: dict) -> None:
    """닫힌 식이 '창 전체 회귀 후 마지막 L일 잔차'와 같은지 몇 점 직접 확인."""
    C = P["C"]
    r_all = logret(C)
    mq = np.log(P["bench"]["C"]).diff()
    for W, L in ((126, 63), (252, 126)):
        sc = resid_mom(P, W, L)
        for sym, day in (("NVDA", "2026-03-13"), ("MU", "2026-07-22"), ("AVGO", "2025-06-02")):
            if sym not in C.columns:
                continue
            t = C.index.get_loc(pd.Timestamp(day))
            y = r_all[sym].iloc[t - W + 1:t + 1].to_numpy()
            x = mq.iloc[t - W + 1:t + 1].to_numpy()
            ok = np.isfinite(y) & np.isfinite(x)
            bb, aa = np.polyfit(x[ok], y[ok], 1)
            e = (y - aa - bb * x)[-L:]
            want = e.sum() / e.std(ddof=1)
            got = sc[sym].iloc[t]
            assert np.isclose(want, got, rtol=1e-6), (W, L, sym, day, want, got)


# ---------- C5 고점 ----------
def hi_dist_H(P):
    return P["C"] / roll_max(P["H"]) - 1.0


def hi_dist_C(P):
    return P["C"] / roll_max(P["C"]) - 1.0


def ath_dist_H(P):
    return P["C"] / P["H"].cummax() - 1.0  # 자료 시작(2023-01-03) 이후 누적 최고 고가


def days_since_high(P, n: int = 252):
    """−(최근 n일 최고 고가가 나온 뒤 지난 거래일 수). 같은 값이 여럿이면 가장 최근 날 기준.
    창 안 유효 봉이 120개 미만이면 NaN."""
    Hh = P["H"].to_numpy(float)
    T, N = Hh.shape
    pad = np.vstack([np.full((n - 1, N), np.nan), Hh])
    win = sliding_window_view(pad, n, axis=0)  # (T, N, n), 마지막 축이 시간(오래된→최근)
    cnt = np.isfinite(win).sum(axis=2)
    rev = np.where(np.isfinite(win), win, -np.inf)[:, :, ::-1]
    since = rev.argmax(axis=2).astype(float)  # 뒤집은 창의 첫 최대 = 가장 최근 최고점
    since[cnt < min(n, HI_MIN)] = np.nan
    out = pd.DataFrame(-since, index=P["H"].index, columns=P["H"].columns)
    return out.where(P["C"].notna())


def new_high_count(P, k: int):
    """최근 k일 중 종가가 252일 최고 종가(그날 포함, min_periods 120)를 새로 쓴 날 수.
    252일 최고가가 아직 없는 날(봉 120개 미만)은 0으로 센다. k일 창은 다 차야 한다."""
    C = P["C"]
    mx = roll_max(C)
    flag = (C >= mx).astype(float).where(C.notna())
    return flag.rolling(k, min_periods=k).sum()


# ---------- extra_ (결과 보기 전에 정함) ----------
def _pr(P, s):
    return H.pct_rank(s, H.universe_mask(P) & s.notna())


def extra_rs_high(P):
    a = _pr(P, ibd_rs(P["C"]))
    b = _pr(P, hi_dist_H(P))
    return 0.5 * a + 0.5 * b


def extra_highdist_voladj(P):
    return np.log(P["C"] / roll_max(P["H"])) / sigma60(P["C"])


def extra_fip126(P):
    C = P["C"]
    r = logret(C)
    pos = (r > 0).astype(float).where(r.notna()).rolling(126, min_periods=126).mean()
    neg = (r < 0).astype(float).where(r.notna()).rolling(126, min_periods=126).mean()
    r126 = ret(C, 126)
    ID = np.sign(r126) * (neg - pos)
    return 0.5 * _pr(P, r126) + 0.5 * _pr(P, -ID)


def extra_accel63(P):
    C = P["C"]
    return ret(C, 63, 0) - ret(C, 63, 63)


# ---------- 격자 ----------
def variants():
    v = []
    for s in (0, 5):
        for L in (21, 42, 63, 126, 189, 252):
            v.append((f"C1_ret{L}_s{s}", f"C.shift({s})/C.shift({s + L})-1; 자료<{s + L + 1}봉이면 NaN", make_ret(L, s)))
    for L in (63, 126, 189, 252):
        v.append((f"C1_ret{L}_s21", f"C.shift(21)/C.shift({21 + L})-1; 자료 부족 NaN", make_ret(L, 21)))
    for L in (21, 63, 126, 252):
        v.append((f"C2_voladj{L}", f"ln(C/C.shift({L}))/(σ60·√{L}); σ60=일 로그수익 rolling(60,mp=60) std ddof1",
                  make_voladj(L)))
    for s in (0, 5):
        v.append((f"C3_ibd_rs_s{s}", f"0.4·r63+0.2·r126+0.2·r189+0.2·r252 (r=C.shift({s})/C.shift({s}+L)-1); "
                  "하나라도 NaN이면 NaN(252+s봉 필요)", make_ibd(s)))
    for W in (126, 252):
        for L in (63, 126):
            v.append((f"C4_resid_W{W}_L{L}",
                      f"일 로그수익~QQQ 로그수익 OLS(창 {W}, min_periods {min(W, HI_MIN)}, t까지) → 그 a·b로 최근 {L}일 잔차; "
                      f"sum/std(ddof1). 정확한 닫힌 식(롤링 모멘트), {L}일 창은 다 차야 함",
                      make_resid(W, L)))
    v += [
        ("C5_hidist_H252", "C/rolling(252,mp=120) 최고 고가 −1", hi_dist_H),
        ("C5_hidist_C252", "C/rolling(252,mp=120) 최고 종가 −1", hi_dist_C),
        ("C5_athdist_H", "C/자료 시작(2023-01-03) 이후 누적 최고 고가 −1", ath_dist_H),
        ("C5_days_since_H252", "−(252일 최고 고가 뒤 지난 거래일 수); 동률이면 가장 최근; 창 유효봉<120 NaN",
         days_since_high),
        ("C5_newhigh_cnt20", "최근 20일 중 C ≥ rolling(252,mp=120) 최고 종가(그날 포함)인 날 수; 최고가 미정 날은 0; 20일 창 다 차야 함",
         lambda P: new_high_count(P, 20)),
        ("C5_newhigh_cnt60", "최근 60일 중 같은 날 수; 60일 창 다 차야 함", lambda P: new_high_count(P, 60)),
        # extra_ (격자 밖, 실행 전 정함)
        ("extra_rs_high", "0.5·pct(IBD RS s0) + 0.5·pct(C/252일 최고 고가); pct=H.pct_rank(universe_mask)", extra_rs_high),
        ("extra_highdist_voladj", "ln(C/252일 최고 고가)/σ60 (고점 거리를 변동성 단위로)", extra_highdist_voladj),
        ("extra_fip126", "0.5·pct(r126) + 0.5·pct(−ID126), ID=sgn(r126)·(%하락일−%상승일) (frog-in-the-pan)",
         extra_fip126),
        ("extra_accel63", "r63(s0) − r63(s63) (모멘텀 가속)", extra_accel63),
    ]
    return v


def main(only=None):
    P = H.load_panel()
    _check_resid(P)
    rows = []
    t0 = time.time()
    for name, spec, fn in variants():
        if only and name not in only:
            continue
        sc = fn(P)
        res = H.evaluate(sc)
        nl = H.no_lookahead(fn)
        rows.append({"name": name, "spec": spec, "result": res, "no_lookahead": bool(nl)})
        print(("" if nl else "[실격] ") + H.summary_line(name, res), flush=True)
    path = H.save_results(FAMILY, rows)
    print("saved", path, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
