"""과제 E1 — 추세 등급의 시간 단위(바탕 봉)와 리밸런싱(등급 갱신) 주기 비교 (2026-10-12).

질문(사용자 2026-10-12): 큰 펀드는 일·주·월·분기 단위로 리밸런싱한다. 3월까지 S였다가 4월부터 아닌 종목은
급락이 아니라 서서히 빠지기도 한다. 등급을 나누는 최소 시간 단위를 우리 자료로 잰다.

사전 등록 격자(측정 전에 고정, 측정 뒤 바꾸지 않음)
  A 바탕 봉(신호) 5개
    a1   일봉 FINAL: composite E14(L·M·H·Q 단면 백분위 평균)의 10일 평균 (fam_composite.make(GRID['E14'], 10))
    a2   일봉 FINAL 평활 없음 (make(GRID['E14'], 0))
    a3n  주봉 대응, 평활 없음: 주(W-FRI) 종가로 C/SMA40주−1, C/SMA10주−1, C/52주 최고 고가−1, Clenow 26주(연율 52)
    a3s2 주봉 대응, 2주 평균: a3n 점수와 직전 완성 주 점수의 평균
    a4   월봉 대응: 월말 종가로 C/SMA10개월−1, C/SMA2개월−1, C/12개월 최고 고가−1, Clenow 6개월(연율 12)
    등급: 점수의 그날 백분위(universe_mask 안) → 피라미드 10/20/40/20/10 + 같은 봉의 관문
          (S·A는 C>짧은 이평>긴 이평, 아니면 B; C<긴 이평 그리고 짧은<긴이면 D). 일봉은 fam_ladder.hyb_grades 그대로.
  B 갱신 주기: D 매일 / W 매주(첫 거래일 시가) / M 매월(첫 거래일 시가) / Q 매분기(1·4·7·10월 첫 거래일 시가).
    결정 = 직전 기간 마지막 거래일 종가, 체결 = 다음 거래일 시가. 바탕 봉보다 잦은 조합은 뺀다 → 신호×주기 16개.
  C 보유 집합: S / SA(S+A) / BUF(들어올 때 S, S·A인 동안 유지, B 이하·등급 없음이면 나감) → 48개.
  추가 (i)  월 k번째 거래일 종가 결정(k=1·5·10·15·20; 그 달 거래일이 k보다 적으면 마지막 거래일) × 5 신호 × 3 집합 = 75개
  추가 (ii) 분기 둘째 달(2·5·8·11월) 10번째 거래일 종가 결정(≈15일 무렵, 실적 시즌 뒤) × 5 × 3 = 15개
  합계 138개. 진단(D1 등급 유지율, D2 신호 지연별 IC, D3 S 이탈 경로, D4 바탕 봉별 S 겹침, D5 분기 바구니)은 변형이 아니며 선택에 쓰지 않는다.

주봉·월봉 '마지막 봉' 처리(측정 전 결정)
  결정일 t의 주봉·월봉 지표는 t까지의 일봉만으로 만든다. t가 주·월 마지막 거래일이 아니면 마지막 봉은 t 종가까지의
  진행 중 봉이다(종가 = t 종가, 고가 = 그 봉 안 t까지 최고 고가). 주 단위 갱신(W)의 결정일은 늘 금요일(휴장이면 그 주
  마지막 거래일)이라 완성 봉이고, 월 단위 갱신(M)에서 주봉 신호는 월말이 주 중간이면 진행 중 주봉을 쓴다.
  봉 최소 개수: 일봉 min_periods=min(n,120)과 같은 비율로 주봉 min(n,24), 월봉 min(n,6). Clenow는 봉 n개가 모두 있어야 값.

성과 채점
  같은 비중 보유, 기간 동안은 사서 들고 있기(비중 표류), 갱신일에 다시 같은 비중. 체결 시가가 없는 종목은 뺀다.
  매도 시가가 없으면 직전 종가. 보유 0이면 현금(0%).
  구간: front 2026-01-02~05-29, back 2026-06-01~끝, y2026 2026-01-02~끝, y2025, y2024(DRUG 제외; 보유·EW 모두).
  구간 포함 = 결정일 ≥ 구간 시작 그리고 다음 체결일(매도) ≤ 구간 끝(다 채운 기간만, 구간 시작 전 신호 안 씀).
  보유 종목군 = universe_mask(결정일 이하 최신 PDF, 봉 130개 이상; 2026 이전은 패널 150종목 전체 — 생존 편향).
  비교: 같은 주기·같은 기간으로 보유 종목 전체 같은 비중(EW), 그리고 첫 체결 시가~마지막 매도 시가 QQQ 보유.
  최대 낙폭: 시작 1.0, 매일 종가 평가, 마지막 매도 시가.
  회전율: 갱신마다 Σ|목표 비중−표류 비중|(매수+매도). 연 회전율(편도) = ½×(첫 매수·마지막 청산 제외 합)/연수.
  스프레드 민감도: 거래 금액마다 0.1%(첫 매수·마지막 청산 포함) — 수수료 0, 결과에 따로 적는다.

주 지표(순위·순위상관, 측정 전 고정): ex_EW_m = ln((1+R)/(1+R_EW)) × 21 / 보유 거래일 수 × 100
  (같은 주기·같은 기간 EW 대비 로그 초과수익의 21거래일 환산 %, 스프레드 0). 주기마다 구간 길이가 달라 환산한다.

사전 등록 밖(결과를 본 뒤 추가, 선택에 쓰지 않음)
  보조 채점 'start'(구간 시작 운용, start_run): 사전 등록 채점은 주기·날짜마다 덮는 달이 달라 주기끼리 비교가
  왜곡돼(예: 월말 갱신은 1~5월에서 1월이 빠짐), 같은 시작·끝으로 다시 잰다. 처음 시도한 '2023-09부터 연속 운용'은
  2025-12 결정이 2026 이전 보유 기준(150종목 전체)을 써 2026년 1월 바구니가 오염돼 버렸다.
  phase_vs_cadence(날짜 폭 대 주기 차이), 진단 D4(바탕 봉별 S 겹침)·D5(분기 바구니)도 결과를 본 뒤 추가했다.

실행: cd data/derived/analysis/trend_grade/cadence && PYTHONIOENCODING=utf-8 py code_cadence.py
출력: results_cadence.json, results_cadence_tables.md(표). 서술은 results_cadence.md.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import harness as H  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOWS = {
    "front": H.WINDOWS["front"],
    "back": H.WINDOWS["back"],
    "y2026": ("2026-01-02", "2026-12-31"),
    "y2025": H.WINDOWS["y2025"],
    "y2024": H.WINDOWS["y2024"],
}
EXCLUDE = {"y2024": {"DRUG"}}
SPREAD = 0.001
BASES = ("a1", "a2", "a3n", "a3s2", "a4")
BAR = {"a1": "D", "a2": "D", "a3n": "W", "a3s2": "W", "a4": "M"}
ALLOWED = {"D": ("D", "W", "M", "Q"), "W": ("W", "M", "Q"), "M": ("M", "Q")}
SETS = ("S", "SA", "BUF")
OFFSETS = (1, 5, 10, 15, 20)
QMID_DAY = 10
BASE_SPEC = {
    "a1": "일봉 FINAL: E14(C/SMA200, C/SMA50, C/252일 최고 고가, Clenow126) 백분위 평균의 10일 평균, 관문 C>SMA50>SMA200",
    "a2": "일봉 E14 평활 없음, 관문 C>SMA50>SMA200",
    "a3n": "주봉(W-FRI) C/SMA40주, C/SMA10주, C/52주 최고 고가, Clenow 26주 백분위 평균, 평활 없음, 관문 C>SMA10주>SMA40주",
    "a3s2": "a3n 점수와 직전 완성 주 점수의 평균(2주 평균), 관문 같음",
    "a4": "월봉 C/SMA10개월, C/SMA2개월, C/12개월 최고 고가, Clenow 6개월 백분위 평균, 관문 C>SMA2개월>SMA10개월",
}


# ---------------------------------------------------------------- 주봉·월봉 진행 중 봉 지표
def _codes(cal: pd.DatetimeIndex, freq: str) -> tuple[np.ndarray, int]:
    per = cal.to_period("W-FRI" if freq == "W" else "M")
    codes, uniq = pd.factorize(per)
    assert np.all(np.diff(codes) >= 0)
    return codes, len(uniq)


def _prev_windows(X: np.ndarray, m: int) -> np.ndarray:
    """(K,N) → (K,N,m): k번째 칸은 X[k-m .. k-1](직전 완성 봉 m개). 앞쪽은 NaN."""
    Xp = np.vstack([np.full((m, X.shape[1]), np.nan), X])
    return np.lib.stride_tricks.sliding_window_view(Xp, m, axis=0)[: X.shape[0]]


def live_bars(P: dict, freq: str) -> dict:
    """결정일 t마다 t까지의 자료로 만든 주봉(W)·월봉(M) 지표(날짜 × 종목). 마지막 봉은 진행 중 봉."""
    C, Hi = P["C"], P["H"]
    cal = C.index
    k, K = _codes(cal, freq)
    Cn = C.to_numpy(float)
    BC = C.groupby(k).last().to_numpy(float)       # 봉 종가(봉 안 마지막 유효 종가)
    BH = Hi.groupby(k).max().to_numpy(float)       # 봉 고가
    curH = Hi.groupby(k).cummax().to_numpy(float)  # 진행 중 봉의 t까지 최고 고가
    if freq == "W":
        n_s, n_l, n_h, n_q, ann, mp = 10, 40, 52, 26, 52.0, 24
    else:
        n_s, n_l, n_h, n_q, ann, mp = 2, 10, 12, 6, 12.0, 6
    out = {"codes": k}

    def sma_live(n):
        w = _prev_windows(BC, n - 1)
        s, c = np.nansum(w, axis=2), np.isfinite(w).sum(axis=2)
        s, c = s[k], c[k]
        val = (s + Cn) / (c + 1)
        return np.where(np.isfinite(Cn) & (c + 1 >= min(n, mp)), val, np.nan)

    def high_live(n):
        w = _prev_windows(BH, n - 1)
        c = np.isfinite(w).sum(axis=2)
        mx = np.where(c > 0, np.nanmax(np.where(np.isfinite(w), w, -np.inf), axis=2), np.nan)
        mx, c = mx[k], c[k]
        cur_ok = np.isfinite(curH)
        val = np.fmax(mx, curH)
        return np.where(cur_ok & (c + cur_ok >= min(n, mp)), val, np.nan)

    def clenow_live(n):
        m = n - 1
        x = np.arange(n, dtype=float) - (n - 1) / 2.0
        sxx = float(x @ x)
        w = _prev_windows(np.log(BC), m)                 # (K,N,m) 시간 순서
        W1 = (w * x[:m]).sum(axis=2)                      # NaN 하나라도 있으면 NaN
        S1 = w.sum(axis=2)
        S2 = (w * w).sum(axis=2)
        W1, S1, S2 = W1[k], S1[k], S2[k]
        y = np.log(Cn)
        sxy = W1 + x[m] * y
        sy = S1 + y
        syy = S2 + y * y - sy * sy / n
        b = sxy / sxx
        r2 = np.where(syy > 0, b * b * sxx / syy, np.nan)
        return (np.exp(ann * b) - 1.0) * r2

    idx, cols = C.index, C.columns
    df = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    out["s_short"] = df(sma_live(n_s))
    out["s_long"] = df(sma_live(n_l))
    out["high"] = df(high_live(n_h))
    out["Q"] = df(clenow_live(n_q))
    return out


def avg_pct(raw: dict, um: pd.DataFrame) -> pd.DataFrame:
    """fam_composite.composite와 같은 평균(가중 1, NaN 재료는 빼고 남은 것으로 나눔)."""
    num, den = None, None
    for x in raw.values():
        x = x.replace([np.inf, -np.inf], np.nan)
        pr = H.pct_rank(x, um & x.notna())
        has = pr.notna().astype(float)
        num = pr.fillna(0.0) if num is None else num + pr.fillna(0.0)
        den = has if den is None else den + has
    return (num / den).where(den > 0)


def bar_score(P: dict, freq: str, smooth2: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """주봉·월봉 대응 점수와 관문용 짧은·긴 이평(진행 중 봉)."""
    C = P["C"]
    lb = live_bars(P, freq)
    raw = {"L": C / lb["s_long"] - 1.0, "M": C / lb["s_short"] - 1.0, "H": C / lb["high"] - 1.0, "Q": lb["Q"]}
    sc = avg_pct(raw, H.universe_mask(P))
    if smooth2:
        k = lb["codes"]
        arr = sc.to_numpy(float)
        last_pos = pd.Series(np.arange(len(k))).groupby(k).max().to_numpy()
        bar_sc = arr[last_pos]                                   # 봉 마지막 날 점수(완성 봉)
        prev = np.vstack([np.full((1, arr.shape[1]), np.nan), bar_sc])[k]  # 직전 완성 봉 점수
        both = np.stack([arr, prev])
        avg = np.nanmean(both, axis=0)
        sc = pd.DataFrame(np.where(np.isfinite(arr), avg, np.nan), index=sc.index, columns=sc.columns)
    return sc, lb["s_short"], lb["s_long"]


def hyb_custom(score: pd.DataFrame, C: pd.DataFrame, s_short: pd.DataFrame, s_long: pd.DataFrame,
               mask: pd.DataFrame) -> pd.DataFrame:
    """fam_ladder.hyb_grades와 같은 규칙, 관문 이평만 바꿔 넣는다."""
    g = FL.rel_grades(score, H.PYRAMID, mask).to_numpy(object).copy()
    gate = ((C > s_short) & (s_short > s_long)).to_numpy()
    dcond = ((C < s_long) & (s_short < s_long)).to_numpy()
    graded = pd.notna(g)
    top = graded & ((g == "S") | (g == "A"))
    g[top & ~gate] = "B"
    g[graded & dcond] = "D"
    return pd.DataFrame(g, index=C.index, columns=C.columns)


def score_fn(base: str):
    if base == "a1":
        return FC.make(FC.GRID["E14"], 10)
    if base == "a2":
        return FC.make(FC.GRID["E14"], 0)
    if base == "a3n":
        return lambda P: bar_score(P, "W")[0]
    if base == "a3s2":
        return lambda P: bar_score(P, "W", True)[0]
    if base == "a4":
        return lambda P: bar_score(P, "M")[0]
    raise ValueError(base)


def grades_of(base: str, P: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    um = H.universe_mask(P)
    if base in ("a1", "a2"):
        sc = score_fn(base)(P)
        return FL.hyb_grades(sc, P, um), sc
    sc, ss, sl = bar_score(P, "W" if base.startswith("a3") else "M", base == "a3s2")
    return hyb_custom(sc, P["C"], ss, sl, um), sc


def grade_code_fn(base: str):
    return lambda P: FL.code_of(grades_of(base, P)[0])


# ---------------------------------------------------------------- 갱신 일정
def schedule(cal: pd.DatetimeIndex, kind: str) -> np.ndarray:
    """결정일 위치(정수). 체결은 다음 거래일."""
    T = len(cal)
    pos = pd.Series(np.arange(T), index=cal)
    m = cal.to_period("M")
    if kind == "D":
        dec = np.arange(T)
    elif kind == "W":
        dec = pos.groupby(cal.to_period("W-FRI")).max().to_numpy()
    elif kind == "M":
        dec = pos.groupby(m).max().to_numpy()
    elif kind == "Q":
        last = pos.groupby(m).max()
        dec = last[[p.month in (3, 6, 9, 12) for p in last.index]].to_numpy()
    elif kind.startswith("M@"):
        kk = int(kind[2:])
        dec = pos.groupby(m).apply(lambda s: s.iloc[min(kk, len(s)) - 1]).to_numpy()
    elif kind == "Qmid":
        g = pos.groupby(m).apply(lambda s: s.iloc[min(QMID_DAY, len(s)) - 1])
        dec = g[[p.month in (2, 5, 8, 11) for p in g.index]].to_numpy()
    else:
        raise ValueError(kind)
    dec = np.asarray(sorted(int(d) for d in dec))
    return dec[dec + 1 < T]


def window_periods(cal, dec: np.ndarray, wname: str) -> list[tuple[int, int, int]]:
    """(결정 위치, 체결 위치, 매도 위치). 결정일 ≥ 시작, 매도일 ≤ 끝."""
    a, b = WINDOWS[wname]
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    out = []
    for j in range(len(dec) - 1):
        d, e, x = dec[j], dec[j] + 1, dec[j + 1] + 1
        if x >= len(cal):
            break
        if cal[d] >= a and cal[x] <= b:
            out.append((int(d), int(e), int(x)))
    return out


# ---------------------------------------------------------------- 보유 성과
class Book:
    def __init__(self, P: dict):
        self.P = P
        self.cal = P["C"].index
        self.cols = list(P["C"].columns)
        self.O = P["O"].to_numpy(float)
        self.Cff = P["C"].ffill().to_numpy(float)
        self.QO = P["bench"]["O"].reindex(self.cal).to_numpy(float)
        self.QC = P["bench"]["C"].reindex(self.cal).to_numpy(float)
        self.um = H.universe_mask(P).to_numpy(bool)

    def tradeable(self, e: int, idx: np.ndarray) -> np.ndarray:
        return idx[np.isfinite(self.O[e, idx])]

    def run(self, periods: list[tuple[int, int, int, np.ndarray]]) -> dict:
        N = len(self.cols)
        eq, net = 1.0, 1.0
        curve = [1.0]
        w_pre = np.zeros(N)
        traded, names, rets = [], [], []
        for j, (d, e, x, idx) in enumerate(periods):
            w_new = np.zeros(N)
            if len(idx):
                w_new[idx] = 1.0 / len(idx)
            tr = float(np.abs(w_new - w_pre).sum())
            traded.append(tr)
            net *= 1.0 - SPREAD * tr
            names.append(len(idx))
            if not len(idx):
                curve += [eq] * (x - e)
                w_pre = np.zeros(N)
                rets.append(0.0)
                continue
            o0 = self.O[e, idx]
            marks = np.nanmean(self.Cff[e:x, idx] / o0, axis=1)
            curve += list(eq * marks)
            ex = self.O[x, idx]
            ex = np.where(np.isfinite(ex), ex, self.Cff[x - 1, idx])
            ratio = ex / o0
            g = float(np.nanmean(ratio))
            eq *= g
            net *= g
            rets.append(g - 1.0)
            w_pre = np.zeros(N)
            w_pre[idx] = ratio / np.nansum(ratio)
        final_tr = float(w_pre.sum())
        net *= 1.0 - SPREAD * final_tr
        curve.append(eq)
        curve = np.asarray(curve)
        mdd = float((curve / np.maximum.accumulate(curve) - 1.0).min())
        days = periods[-1][2] - periods[0][1]
        years = days / 252.0
        inner = sum(traded[1:])
        return {"ret": eq - 1.0, "ret_net": net - 1.0, "mdd": mdd, "days": int(days),
                "turn_yr": 0.5 * inner / years if years > 0 else float("nan"),
                "traded_all": sum(traded) + final_tr,
                "n_mean": float(np.mean(names)), "n_min": int(np.min(names)), "n_max": int(np.max(names)),
                "cash_periods": int(sum(1 for n in names if n == 0)), "period_rets": rets}

    def qqq(self, e0: int, x1: int) -> dict:
        curve = np.concatenate([[1.0], self.QC[e0:x1] / self.QO[e0], [self.QO[x1] / self.QO[e0]]])
        return {"ret": float(curve[-1] - 1.0), "mdd": float((curve / np.maximum.accumulate(curve) - 1).min())}


def baskets(G: np.ndarray, book: Book, wperiods, set_name: str, excl_idx: set) -> list:
    out, state = [], set()
    for d, e, x in wperiods:
        row = G[d]
        if set_name == "S":
            names = set(np.where(row == "S")[0])
        elif set_name == "SA":
            names = set(np.where((row == "S") | (row == "A"))[0])
        elif set_name == "BUF":
            keep = {i for i in state if row[i] in ("S", "A")}
            names = keep | set(np.where(row == "S")[0])
            state = names
        elif set_name == "EW":
            names = set(np.where(book.um[d])[0])
        else:
            raise ValueError(set_name)
        idx = np.array(sorted(names - excl_idx), dtype=int)
        idx = book.tradeable(e, idx) if len(idx) else idx
        out.append((d, e, x, idx))
    return out


def r2(x, k=2):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def metrics(book: Book, G: np.ndarray, kind: str, set_name: str, wname: str, keep_log: bool = False) -> dict:
    cal = book.cal
    dec = SCHED[kind]
    wp = window_periods(cal, dec, wname)
    if not wp:
        return {"n_periods": 0}
    excl = {book.cols.index(s) for s in EXCLUDE.get(wname, set()) if s in book.cols}
    bk = baskets(G, book, wp, set_name, excl)
    ew = book.run(baskets(G, book, wp, "EW", excl))
    st = book.run(bk)
    q = book.qqq(wp[0][1], wp[-1][2])
    days = st["days"]
    ex_m = (np.log1p(st["ret"]) - np.log1p(ew["ret"])) * 21.0 / days * 100 if days else float("nan")
    ex_m_net = (np.log1p(st["ret_net"]) - np.log1p(ew["ret_net"])) * 21.0 / days * 100 if days else float("nan")
    o = {"n_periods": len(wp), "first_exec": str(cal[wp[0][1]].date()), "last_exit": str(cal[wp[-1][2]].date()),
         "days": days,
         "ret": r2(st["ret"] * 100), "mdd": r2(st["mdd"] * 100), "ret_net01": r2(st["ret_net"] * 100),
         "turn_yr_pct": r2(st["turn_yr"] * 100, 0), "n_mean": r2(st["n_mean"], 1), "n_min": st["n_min"],
         "n_max": st["n_max"], "cash_periods": st["cash_periods"],
         "EW_ret": r2(ew["ret"] * 100), "EW_mdd": r2(ew["mdd"] * 100), "EW_ret_net01": r2(ew["ret_net"] * 100),
         "QQQ_ret": r2(q["ret"] * 100), "QQQ_mdd": r2(q["mdd"] * 100),
         "ex_EW": r2((st["ret"] - ew["ret"]) * 100), "ex_QQQ": r2((st["ret"] - q["ret"]) * 100),
         "ex_EW_m": r2(ex_m, 3), "ex_EW_m_net01": r2(ex_m_net, 3),
         "period_hit_vs_EW": r2(float(np.mean(np.array(st["period_rets"]) > np.array(ew["period_rets"]))) * 100, 1)}
    if keep_log:
        o["log"] = [{"dec": str(cal[d].date()), "exec": str(cal[e].date()), "exit": str(cal[x].date()),
                     "names": [book.cols[i] for i in idx], "ret": r2(rr * 100), "EW": r2(er * 100)}
                    for (d, e, x, idx), rr, er in zip(bk, st["period_rets"], ew["period_rets"])]
    return o


# ---------------------------------------------------------------- 보조 채점: 구간 시작 운용(격자 밖 채점 방식, 선택에 안 씀)
def start_run(book: Book, G: np.ndarray, kind: str, set_name: str, wname: str) -> dict:
    """구간 첫 거래일 종가에 처음 고르고(다음 날 시가 매수), 이후 그 주기 일정대로 갱신, 구간 마지막 날 종가로 평가.
    모든 변형이 같은 시작(구간 둘째 거래일 시가)·같은 끝(마지막 날 종가)을 가진다. 결정은 모두 구간 안(시작 전 신호 없음).
    주 격자의 '다 채운 기간만' 규칙은 주기·날짜마다 덮는 달이 달라 생기는 차이를 없애려는 보조 채점.
    처음 매수와 마지막 청산(종가)도 스프레드에 넣는다. 연 회전율은 처음 매수·마지막 청산을 뺀다."""
    cal = book.cal
    T, N = len(cal), len(book.cols)
    a, b = (pd.Timestamp(v) for v in WINDOWS[wname])
    i_first = int(np.searchsorted(cal, a))
    i_end = int(np.searchsorted(cal, b, side="right")) - 1
    sched = SCHED[kind]
    decs = [i_first] + [int(d) for d in sched if i_first < d < i_end]
    excl = {book.cols.index(x) for x in EXCLUDE.get(wname, set()) if x in book.cols}
    eq = net = 1.0
    curve = [1.0]
    w_pre = np.zeros(N)
    state: set = set()
    traded, names_n = [], []
    for j, d in enumerate(decs):
        e = d + 1
        x = decs[j + 1] + 1 if j + 1 < len(decs) else None   # None: 마지막 기간은 i_end 종가까지
        row = G[d]
        if set_name == "S":
            names = set(np.where(row == "S")[0])
        elif set_name == "SA":
            names = set(np.where((row == "S") | (row == "A"))[0])
        elif set_name == "BUF":
            names = {i for i in state if row[i] in ("S", "A")} | set(np.where(row == "S")[0])
            state = names
        else:
            names = set(np.where(book.um[d])[0])
        idx = np.array(sorted(names - excl), dtype=int)
        idx = book.tradeable(e, idx) if len(idx) else idx
        w_new = np.zeros(N)
        if len(idx):
            w_new[idx] = 1.0 / len(idx)
        tr = float(np.abs(w_new - w_pre).sum())
        traded.append(tr)
        names_n.append(len(idx))
        net *= 1.0 - SPREAD * tr
        stop = x if x is not None else i_end + 1
        if not len(idx):
            curve += [eq] * (stop - e)
            w_pre = np.zeros(N)
            continue
        o0 = book.O[e, idx]
        marks = np.nanmean(book.Cff[e:stop, idx] / o0, axis=1)
        curve += list(eq * marks)
        if x is not None:
            ex = book.O[x, idx]
            ex = np.where(np.isfinite(ex), ex, book.Cff[x - 1, idx])
            ratio = ex / o0
        else:
            ratio = book.Cff[i_end, idx] / o0
        g = float(np.nanmean(ratio))
        eq *= g
        net *= g
        w_pre = np.zeros(N)
        w_pre[idx] = ratio / np.nansum(ratio)
    net *= 1.0 - SPREAD * float(w_pre.sum())
    curve = np.asarray(curve)
    e0 = decs[0] + 1
    days = i_end - e0 + 1
    return {"ret": eq - 1.0, "ret_net": net - 1.0, "mdd": float((curve / np.maximum.accumulate(curve) - 1).min()),
            "days": int(days), "turn_yr": 0.5 * sum(traded[1:]) / (days / 252.0),
            "n_mean": float(np.mean(names_n)), "n_decisions": len(decs),
            "from_open": str(cal[e0].date()), "to_close": str(cal[i_end].date()), "e0": e0, "i_end": i_end}


def start_metrics(book: Book, G: np.ndarray, kind: str, set_name: str, wname: str) -> dict:
    st = start_run(book, G, kind, set_name, wname)
    ew = start_run(book, G, kind, "EW", wname)
    e0, i1 = st["e0"], st["i_end"]
    q = np.concatenate([[1.0], book.QC[e0:i1 + 1] / book.QO[e0]])
    R, E_, days = st["ret"], ew["ret"], st["days"]
    return {"from_open": st["from_open"], "to_close": st["to_close"], "n_days": days, "n_decisions": st["n_decisions"],
            "ret": r2(R * 100), "mdd": r2(st["mdd"] * 100), "ret_net01": r2(st["ret_net"] * 100),
            "turn_yr_pct": r2(st["turn_yr"] * 100, 0), "n_mean": r2(st["n_mean"], 1),
            "EW_ret": r2(E_ * 100), "EW_mdd": r2(ew["mdd"] * 100), "EW_ret_net01": r2(ew["ret_net"] * 100),
            "QQQ_ret": r2((q[-1] - 1) * 100), "QQQ_mdd": r2(float((q / np.maximum.accumulate(q) - 1).min()) * 100),
            "ex_EW": r2((R - E_) * 100), "ex_QQQ": r2((R - (q[-1] - 1)) * 100),
            "ex_EW_m": r2((np.log1p(R) - np.log1p(E_)) * 21.0 / days * 100, 3),
            "ex_EW_m_net01": r2((np.log1p(st["ret_net"]) - np.log1p(ew["ret_net"])) * 21.0 / days * 100, 3)}


# ---------------------------------------------------------------- 진단(격자 밖, 선택에 안 씀)
def diag_persistence(Gs: dict, cal) -> dict:
    """D1: 그 봉 끝날 S였던 종목이 1주·1개월·1분기 뒤 같은 봉 끝날 어떤 등급인지(비율 %)."""
    out = {}
    lag_spec = {"D": (("1주", 5), ("1개월", 21), ("1분기", 63)),
                "W": (("1주", 1), ("1개월", 4), ("1분기", 13)),
                "M": (("1개월", 1), ("1분기", 3))}
    for base in ("a1", "a2", "a3n", "a4"):
        G = Gs[base]
        bar = BAR[base]
        pts = schedule(cal, bar) if bar != "D" else np.arange(len(cal))
        res = {}
        for wname in ("front", "back", "y2025", "y2024"):
            a, b = (pd.Timestamp(v) for v in WINDOWS[wname])
            r = {}
            for lab, L in lag_spec[bar]:
                cnt = {"S": 0, "A": 0, "B이하": 0, "등급없음": 0}
                for j, p in enumerate(pts):
                    if j + L >= len(pts) or not (a <= cal[p] <= b) or cal[pts[j + L]] > b:
                        continue
                    now, later = G[p], G[pts[j + L]]
                    for i in np.where(now == "S")[0]:
                        v = later[i]
                        key = v if v in ("S", "A") else ("B이하" if v in ("B", "C", "D") else "등급없음")
                        cnt[key] += 1
                tot = sum(cnt.values())
                r[lab] = {"n": tot, **{k: r2(v / tot * 100, 1) if tot else None for k, v in cnt.items()}}
            res[wname] = r
        out[base] = res
    return out


def diag_ic_decay(P: dict, sc: pd.DataFrame, lags=(0, 5, 10, 21, 42, 63), h=20) -> dict:
    """D2: t일 점수 → t+1+L 시가 매수 → t+1+L+h 시가 매도의 날짜별 Spearman 평균. 결정일 ≥ 시작, 매도일 ≤ 끝."""
    O = P["O"]
    cal = O.index
    um = H.universe_mask(P)
    pr = H.pct_rank(sc, um & sc.notna())
    out = {}
    for wname in ("front", "back", "y2025", "y2024"):
        a, b = (pd.Timestamp(v) for v in WINDOWS[wname])
        cols = [c for c in O.columns if c not in EXCLUDE.get(wname, set())]
        r = {}
        for L in lags:
            fr = O.shift(-(1 + L + h)) / O.shift(-(1 + L)) - 1.0
            ex = pd.Series(cal, index=cal).shift(-(1 + L + h))
            ok = (cal >= a) & ex.notna().to_numpy() & (ex.fillna(pd.Timestamp("2100-01-01")) <= b).to_numpy()
            dates = cal[ok]
            if len(dates) < 5:
                r[f"L{L}"] = {"ic": None, "days": int(len(dates))}
                continue
            ic = H._row_spearman(pr.loc[dates, cols], fr.loc[dates, cols].where(um.loc[dates, cols])).dropna()
            r[f"L{L}"] = {"ic": r2(ic.mean(), 3), "t_nw": r2(H.newey_west_t(ic, h), 2), "days": int(len(ic))}
        out[wname] = r
    return out


def diag_exit(P: dict, G: pd.DataFrame, h=21) -> dict:
    """D3: 일봉 a1에서 S였다가 다음 날 S가 아닌 사건. 어디로 갔나(A=한 단계, B이하=여러 단계)와
    이탈 다음 날 시가부터 h일 수익의 그날 보유 종목 평균 대비 초과(%p)."""
    O = P["O"]
    um = H.universe_mask(P)
    fr = (O.shift(-(1 + h)) / O.shift(-1) - 1.0)
    fr_u = fr.where(um)
    base = fr_u.mean(axis=1)
    Gv = G.to_numpy(object)
    cal = O.index
    out = {}
    for wname in ("front", "back", "y2025", "y2024"):
        a, b = (pd.Timestamp(v) for v in WINDOWS[wname])
        ex = pd.Series(cal, index=cal).shift(-(1 + h))
        rec = {"A": [], "B이하": [], "등급없음": 0}
        for t in range(1, len(cal)):
            if not (a <= cal[t] <= b) or pd.isna(ex.iloc[t]) or ex.iloc[t] > b:
                continue
            was = np.where(Gv[t - 1] == "S")[0]
            for i in was:
                v = Gv[t][i]
                if v == "S":
                    continue
                if v not in ("A", "B", "C", "D"):
                    rec["등급없음"] += 1
                    continue
                x = fr.iat[t, i] - base.iat[t]
                if np.isfinite(x):
                    rec["A" if v == "A" else "B이하"].append(x)
        out[wname] = {"to_A": {"n": len(rec["A"]), "ex21": r2(np.mean(rec["A"]) * 100) if rec["A"] else None},
                      "to_B_or_lower": {"n": len(rec["B이하"]),
                                        "ex21": r2(np.mean(rec["B이하"]) * 100) if rec["B이하"] else None},
                      "ungraded": rec["등급없음"]}
    return out


def diag_overlap(Gs: dict, cal) -> dict:
    """D4: 같은 결정일(월말)에 바탕 봉만 다른 신호들의 S 집합이 a1 S와 얼마나 겹치는가(겹친 수/a1 S 수, 자카드)."""
    dec = SCHED["M"]
    out = {}
    for wname in ("front", "back", "y2025", "y2024"):
        a, b = (pd.Timestamp(v) for v in WINDOWS[wname])
        ds = [d for d in dec if a <= cal[d] <= b]
        r = {}
        for base in ("a2", "a3n", "a3s2", "a4"):
            share, jac = [], []
            for d in ds:
                s1 = set(np.where(Gs["a1"][d] == "S")[0])
                s2 = set(np.where(Gs[base][d] == "S")[0])
                if s1:
                    share.append(len(s1 & s2) / len(s1))
                if s1 | s2:
                    jac.append(len(s1 & s2) / len(s1 | s2))
            r[base] = {"months": len(ds), "a1_S_kept_pct": r2(np.mean(share) * 100, 1) if share else None,
                       "jaccard_pct": r2(np.mean(jac) * 100, 1) if jac else None}
        out[wname] = r
    return out


def diag_quarter_baskets(book: Book, G: np.ndarray, base_set=("a1", "S"), windows=("y2025", "y2026")) -> dict:
    """D5: 같은 시작·끝 채점에서 Q·Qmid 바구니와 기간 수익(같은 비중, %), 그리고 같은 기간 EW. 설명용."""
    cal = book.cal
    out = {}
    for w in windows:
        a, b = (pd.Timestamp(v) for v in WINDOWS[w])
        i0 = int(np.searchsorted(cal, a))
        i1 = int(np.searchsorted(cal, b, side="right")) - 1
        for kind in ("Q", "Qmid"):
            decs = [i0] + [int(d) for d in SCHED[kind] if i0 < d < i1]
            rows = []
            for j, d in enumerate(decs):
                e = d + 1
                x = decs[j + 1] + 1 if j + 1 < len(decs) else None
                S = np.where(G[d] == "S")[0]
                S = S[np.isfinite(book.O[e, S])]
                U = np.where(book.um[d])[0]
                U = U[np.isfinite(book.O[e, U])]

                def ret(idx):
                    if not len(idx):
                        return 0.0
                    o0 = book.O[e, idx]
                    if x is None:
                        ex = book.Cff[i1, idx]
                    else:
                        ex = np.where(np.isfinite(book.O[x, idx]), book.O[x, idx], book.Cff[x - 1, idx])
                    return float(np.nanmean(ex / o0) - 1) * 100
                rows.append({"dec": str(cal[d].date()), "exec": str(cal[e].date()),
                             "exit": str(cal[x].date()) if x is not None else f"{cal[i1].date()} 종가",
                             "S_ret": r2(ret(S), 1), "EW_ret": r2(ret(U), 1), "names": [book.cols[i] for i in S]})
            out[f"{w}|{kind}"] = rows
    return out


# ---------------------------------------------------------------- 실행
def variants() -> list[dict]:
    rows = []
    for base in BASES:
        for kind in ALLOWED[BAR[base]]:
            for s in SETS:
                rows.append({"name": f"{base}|{kind}|{s}", "family": "main", "base": base, "cadence": kind, "set": s})
    for base in BASES:
        for k in OFFSETS:
            for s in SETS:
                rows.append({"name": f"{base}|M@{k}|{s}", "family": "offset", "base": base, "cadence": f"M@{k}", "set": s})
    for base in BASES:
        for s in SETS:
            rows.append({"name": f"{base}|Qmid|{s}", "family": "qmid", "base": base, "cadence": "Qmid", "set": s})
    return rows


def spearman(a, b) -> float | None:
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    ok = a.notna() & b.notna()
    if ok.sum() < 4:
        return None
    return r2(a[ok].rank().corr(b[ok].rank()), 3)


def rank_report(rows: list[dict], fam: tuple[str, ...], field: str = "windows") -> dict:
    sel = [dict(r, windows=r[field]) for r in rows if r["family"] in fam]
    f = {r["name"]: r["windows"]["front"].get("ex_EW_m") for r in sel}
    out = {"n_variants": len(sel)}
    both = [r for r in sel if f[r["name"]] is not None and r["windows"]["back"].get("ex_EW_m") is not None]
    out["n_front_and_back"] = len(both)
    for other in ("back", "y2025", "y2024"):
        pairs = [r for r in sel if f[r["name"]] is not None and r["windows"][other].get("ex_EW_m") is not None]
        out[f"spearman_front_vs_{other}"] = spearman([f[r["name"]] for r in pairs],
                                                     [r["windows"][other]["ex_EW_m"] for r in pairs])
        out[f"n_pairs_{other}"] = len(pairs)
    if both:
        fr = sorted(both, key=lambda r: -f[r["name"]])
        bk = sorted(both, key=lambda r: -r["windows"]["back"]["ex_EW_m"])
        top = fr[0]
        out["front_top1"] = {"name": top["name"], "front_ex_EW_m": f[top["name"]],
                             "back_ex_EW_m": top["windows"]["back"]["ex_EW_m"],
                             "back_rank": 1 + [r["name"] for r in bk].index(top["name"]), "of": len(both),
                             "back_ex_EW": top["windows"]["back"].get("ex_EW"),
                             "back_ret": top["windows"]["back"].get("ret"),
                             "back_EW_ret": top["windows"]["back"].get("EW_ret")}
        out["front_top5"] = [{"name": r["name"], "front": f[r["name"]], "back": r["windows"]["back"]["ex_EW_m"],
                              "back_rank": 1 + [x["name"] for x in bk].index(r["name"])} for r in fr[:5]]
        out["back_top5"] = [{"name": r["name"], "back": r["windows"]["back"]["ex_EW_m"], "front": f[r["name"]]}
                            for r in bk[:5]]
        out["back_positive"] = int(sum(1 for r in both if r["windows"]["back"]["ex_EW_m"] > 0))
        out["front_positive"] = int(sum(1 for r in both if f[r["name"]] > 0))
    return out


def by_group(rows, key: str, fam=("main",), field: str = "windows") -> dict:
    """주기·바탕·집합별 주 지표 중앙값(구간마다)."""
    out = {}
    for r in rows:
        if r["family"] not in fam:
            continue
        g = out.setdefault(r[key], {w: [] for w in WINDOWS})
        for w in WINDOWS:
            v = r[field][w].get("ex_EW_m")
            if v is not None:
                g[w].append(v)
    return {k: {w: {"median": r2(float(np.median(v)), 3) if v else None, "n": len(v),
                    "pos": int(sum(1 for x in v if x > 0))} for w, v in d.items()} for k, d in out.items()}


def phase_vs_cadence(rows: list[dict], field: str) -> dict:
    """같은 바탕×집합(15개)에서 (가) 월별 날짜만 바꾼 6개(M, M@1~M@20)의 폭, (나) 분기 Q와 Qmid 차이,
    (다) 주기끼리 차이(|D−W|, |W−M|, |M−Q|)를 주 지표(ex_EW_m)로 비교한 중앙값. 그리고 주기별 1위 횟수."""
    idx = {r["name"]: r for r in rows}
    out = {}
    for w in WINDOWS:
        ph, qq, dw, wm, mq, w_minus_mavg, mavg_l = [], [], [], [], [], [], []
        wins = {"D": 0, "W": 0, "M": 0, "Q": 0}
        wins_n = 0
        for base in BASES:
            for st in SETS:
                g = lambda c: idx[f"{base}|{c}|{st}"][field][w].get("ex_EW_m") if f"{base}|{c}|{st}" in idx else None
                mon = [g("M")] + [g(f"M@{k}") for k in OFFSETS]
                mon = [v for v in mon if v is not None]
                if len(mon) >= 2:
                    ph.append(max(mon) - min(mon))
                    mavg_l.append(float(np.mean(mon)))
                    if g("W") is not None:
                        w_minus_mavg.append(g("W") - float(np.mean(mon)))
                if g("Q") is not None and g("Qmid") is not None:
                    qq.append(abs(g("Q") - g("Qmid")))
                if g("D") is not None and g("W") is not None:
                    dw.append(abs(g("D") - g("W")))
                if g("W") is not None and g("M") is not None:
                    wm.append(abs(g("W") - g("M")))
                if g("M") is not None and g("Q") is not None:
                    mq.append(abs(g("M") - g("Q")))
                cands = {c: g(c) for c in ALLOWED[BAR[base]] if g(c) is not None}
                if len(cands) >= 2:
                    wins[max(cands, key=cands.get)] += 1
                    wins_n += 1
        med = lambda v: r2(float(np.median(v)), 3) if v else None
        out[w] = {"monthly_phase_range_median": med(ph), "n_monthly": len(ph),
                  "Q_vs_Qmid_absdiff_median": med(qq), "n_q": len(qq),
                  "D_vs_W_absdiff_median": med(dw), "W_vs_M_absdiff_median": med(wm), "M_vs_Q_absdiff_median": med(mq),
                  "best_cadence_count": wins, "n_combos": wins_n,
                  "monthly_6phase_mean_median": med(mavg_l),
                  "W_minus_monthly_6phase_mean_median": med(w_minus_mavg),
                  "W_beats_monthly_6phase_mean": int(sum(1 for v in w_minus_mavg if v > 0)), "n_w": len(w_minus_mavg)}
    return out


SCHED: dict = {}


def main():
    t0 = time.time()
    P = H.load_panel()
    cal = P["C"].index
    book = Book(P)
    for kind in ("D", "W", "M", "Q", "Qmid") + tuple(f"M@{k}" for k in OFFSETS):
        SCHED[kind] = schedule(cal, kind)

    # 1) 미래 정보 점검: 점수와 등급 코드를 3개 시점에서 잘라 다시 계산
    look = {}
    for base in BASES:
        look[base] = {"score": bool(H.no_lookahead(score_fn(base))), "grade": bool(H.no_lookahead(grade_code_fn(base)))}
        print("lookahead", base, look[base], f"{time.time()-t0:.0f}s", flush=True)

    # 2) 등급
    Gdf, scores = {}, {}
    for base in BASES:
        Gdf[base], scores[base] = grades_of(base, P)
    Gs = {b: g.to_numpy(object) for b, g in Gdf.items()}

    # 3) 대조: a1 등급 = 연구 FINAL(final.py: eligible 안 백분위 + hyb) — eligible은 다음 날 시가까지 보므로 마지막 날 제외
    E = H.eligible()
    fin = FL.hyb_grades(FC.make(FC.GRID["E14"], 10)(P), P, E)
    rng = (cal >= pd.Timestamp("2026-01-02")) & (cal < cal[-1])
    a, b = Gdf["a1"].loc[rng].where(E.loc[rng]), fin.loc[rng].where(E.loc[rng])
    both = a.notna() & b.notna()
    sanity = {"a1_vs_research_FINAL_2026_cells": int(both.to_numpy().sum()),
              "a1_vs_research_FINAL_2026_diff": int(((a != b) & both).to_numpy().sum())}
    # 주봉 진행 중 봉 대조: 금요일 결정일 값 = 완성 주봉을 직접 resample해 계산한 값
    sanity["weekly_live_vs_resample"] = check_resample(P, "W")
    sanity["monthly_live_vs_resample"] = check_resample(P, "M")
    print("sanity", sanity, flush=True)

    # 4) 격자
    rows = variants()
    for r in rows:
        keep = r["family"] == "main" and r["cadence"] in ("M", "Q")
        r["windows"] = {w: metrics(book, Gs[r["base"]], r["cadence"], r["set"], w, keep_log=keep) for w in WINDOWS}
    print("grid done", len(rows), f"{time.time()-t0:.0f}s", flush=True)

    # 4b) 보조: 구간 시작 운용 채점(같은 시작·끝)
    for r in rows:
        r["start"] = {w: start_metrics(book, Gs[r["base"]], r["cadence"], r["set"], w) for w in WINDOWS}
    print("start-run done", f"{time.time()-t0:.0f}s", flush=True)

    # 5) 순위
    ranks = {"main": rank_report(rows, ("main",)), "all": rank_report(rows, ("main", "offset", "qmid")),
             "start_main": rank_report(rows, ("main",), "start"),
             "start_all": rank_report(rows, ("main", "offset", "qmid"), "start")}
    groups = {"cadence": by_group(rows, "cadence"), "base": by_group(rows, "base"), "set": by_group(rows, "set"),
              "offset_cadence": by_group(rows, "cadence", ("offset",)),
              "start_cadence": by_group(rows, "cadence", ("main",), "start"),
              "start_offset_cadence": by_group(rows, "cadence", ("offset", "qmid"), "start"),
              "start_base": by_group(rows, "base", ("main",), "start"),
              "start_set": by_group(rows, "set", ("main",), "start")}
    phase = {"strict": phase_vs_cadence(rows, "windows"), "start": phase_vs_cadence(rows, "start")}

    # 6) 진단
    diag = {"D1_persistence": diag_persistence(Gs, cal),
            "D2_ic_decay": {"a1": diag_ic_decay(P, scores["a1"]), "a2": diag_ic_decay(P, scores["a2"])},
            "D3_exit_from_S_a1": diag_exit(P, Gdf["a1"]),
            "D4_S_overlap_vs_a1_month_end": diag_overlap(Gs, cal),
            "D5_quarter_baskets_a1_S_start": diag_quarter_baskets(book, Gs["a1"])}

    sched_info = {k: {"n_total": int(len(v)), "dates_2026": [str(cal[d].date()) for d in v if cal[d].year == 2026][:40]}
                  for k, v in SCHED.items() if k != "D"}
    res = {"meta": {"doc": __doc__, "bases": BASE_SPEC, "windows": WINDOWS, "spread": SPREAD,
                    "panel": [str(cal[0].date()), str(cal[-1].date()), int(P["C"].shape[1])],
                    "n_variants": {"main": sum(r["family"] == "main" for r in rows),
                                   "offset": sum(r["family"] == "offset" for r in rows),
                                   "qmid": sum(r["family"] == "qmid" for r in rows), "total": len(rows)},
                    "schedules": sched_info},
           "lookahead": look, "sanity": sanity, "rank": ranks, "groups": groups, "phase_vs_cadence": phase,
           "diagnostics": diag, "rows": rows}
    with open(os.path.join(HERE, "results_cadence.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    write_tables(res)
    print("saved", f"{time.time()-t0:.0f}s")
    return res


def check_resample(P: dict, freq: str) -> dict:
    """봉 마지막 거래일의 진행 중 봉 지표가, 일봉을 주·월 단위로 직접 묶어 rolling으로 낸 값과 같은지."""
    C, Hi = P["C"], P["H"]
    cal = C.index
    lb = live_bars(P, freq)
    k, _ = _codes(cal, freq)
    last = pd.Series(np.arange(len(cal))).groupby(k).max().to_numpy()
    WC = C.groupby(k).last()
    WH = Hi.groupby(k).max()
    n_s, n_l, n_h, n_q, ann, mp = (10, 40, 52, 26, 52.0, 24) if freq == "W" else (2, 10, 12, 6, 12.0, 6)
    s_s = WC.rolling(n_s, min_periods=min(n_s, mp)).mean().to_numpy()
    s_l = WC.rolling(n_l, min_periods=min(n_l, mp)).mean().to_numpy()
    hh = WH.rolling(n_h, min_periods=min(n_h, mp)).max().to_numpy()
    lw = np.log(WC.to_numpy(float))  # Clenow는 연율(52·12)로 직접 회귀(fam_composite.clenow는 연율 252 고정)
    x = np.arange(n_q) - (n_q - 1) / 2.0
    qq = np.full(lw.shape, np.nan)
    for i in range(n_q - 1, len(lw)):
        y = lw[i - n_q + 1:i + 1]
        yc = y - y.mean(axis=0)
        bb = (x @ yc) / (x @ x)
        syy = (yc * yc).sum(axis=0)
        qq[i] = (np.exp(ann * bb) - 1) * np.where(syy > 0, bb * bb * (x @ x) / syy, np.nan)
    Cl = C.to_numpy(float)[last]
    out = {}
    for name, live, ref in (("SMA_short", lb["s_short"], s_s), ("SMA_long", lb["s_long"], s_l),
                            ("High", lb["high"], hh), ("Clenow", lb["Q"], qq)):
        lv = live.to_numpy(float)[last]
        ok = np.isfinite(lv) & np.isfinite(ref) & np.isfinite(Cl)
        out[name] = {"cells": int(ok.sum()),
                     "max_abs_rel_diff": float(np.nanmax(np.abs(lv[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-12)))
                     if ok.any() else None,
                     "nan_mismatch": int((np.isfinite(lv) != np.isfinite(ref))[np.isfinite(Cl)].sum())}
    return out


def fmt(v, nd=1):
    return "—" if v is None else (f"{v:+.{nd}f}" if isinstance(v, (int, float)) else str(v))


def write_tables(res: dict):
    rows = res["rows"]
    L = ["# E1 표 (code_cadence.py 자동 생성)", "",
         "주 지표 ex_EW_m = 같은 주기·같은 기간 보유 종목 전체 같은 비중(EW) 대비 로그 초과수익의 21거래일 환산(%). "
         "수익·낙폭은 %, 회전율은 연 편도 %.", ""]
    for w in WINDOWS:
        L += [f"## {w} {WINDOWS[w][0]}~{WINDOWS[w][1]}", "",
              "| 변형 | 기간 수 | 첫 체결~마지막 매도 | 수익 | 낙폭 | 스프레드0.1% | 연회전 | 평균 종목 | EW 수익 | QQQ 수익 | EW 대비 | QQQ 대비 | ex_EW_m |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["family"] != "main":
                continue
            m = r["windows"][w]
            if not m.get("n_periods"):
                L.append(f"| {r['name']} | 0 | — | | | | | | | | | | |")
                continue
            L.append(f"| {r['name']} | {m['n_periods']} | {m['first_exec']}~{m['last_exit']} | {fmt(m['ret'])} | "
                     f"{fmt(m['mdd'])} | {fmt(m['ret_net01'])} | {m['turn_yr_pct']:.0f} | {m['n_mean']} | {fmt(m['EW_ret'])} | "
                     f"{fmt(m['QQQ_ret'])} | {fmt(m['ex_EW'])} | {fmt(m['ex_QQQ'])} | {fmt(m['ex_EW_m'], 2)} |")
        L.append("")
    L += ["## 보조: 구간 시작 운용(같은 시작·끝) — 격자 밖 채점 방식", "",
          "구간 첫 거래일 종가에 처음 고르고 다음 날 시가 매수, 이후 각 주기 일정대로 갱신, 구간 마지막 날 종가 평가. "
          "모든 결정이 구간 안. 선택에 쓰지 않는다.", ""]
    for w in WINDOWS:
        L += [f"### {w}", "", "| 변형 | 수익 | 낙폭 | 스프레드0.1% | 연회전 | 평균 종목 | EW 수익 | QQQ 수익 | EW 대비 | ex_EW_m |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            m = r["start"][w]
            if r["family"] != "main" or not m.get("n_days"):
                continue
            L.append(f"| {r['name']} | {fmt(m['ret'])} | {fmt(m['mdd'])} | {fmt(m['ret_net01'])} | {m['turn_yr_pct']:.0f} | "
                     f"{m['n_mean']} | {fmt(m['EW_ret'])} | {fmt(m['QQQ_ret'])} | {fmt(m['ex_EW'])} | {fmt(m['ex_EW_m'], 2)} |")
        L.append("")
    L += ["## 날짜 운(같은 월별 갱신의 날짜만 바꾼 폭) 대 주기 차이 — ex_EW_m 중앙값(15개 바탕×집합)", "",
          "| 채점 | 구간 | 월별 6개 날짜 폭 | Q·Qmid 차이 | D·W 차이 | W·M 차이 | M·Q 차이 | 1위 횟수 D/W/M/Q | W − 월별 6날짜 평균 (W 우세) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for kind in ("strict", "start"):
        for w, x in res["phase_vs_cadence"][kind].items():
            bc = x["best_cadence_count"]
            L.append(f"| {kind} | {w} | {fmt(x['monthly_phase_range_median'], 2)} | {fmt(x['Q_vs_Qmid_absdiff_median'], 2)} | "
                     f"{fmt(x['D_vs_W_absdiff_median'], 2)} | {fmt(x['W_vs_M_absdiff_median'], 2)} | {fmt(x['M_vs_Q_absdiff_median'], 2)} | "
                     f"{bc['D']}/{bc['W']}/{bc['M']}/{bc['Q']} (of {x['n_combos']}) | "
                     f"{fmt(x['W_minus_monthly_6phase_mean_median'], 2)} ({x['W_beats_monthly_6phase_mean']}/{x['n_w']}) |")
    L.append("")
    L += ["## (i) 월 k번째 거래일 결정 — ex_EW_m", "",
          "| 바탕|집합 | 구간 | M(월말) | M@1 | M@5 | M@10 | M@15 | M@20 | 폭(최대−최소) |", "|---|---|---|---|---|---|---|---|---|"]
    idx = {r["name"]: r for r in rows}
    for base in BASES:
        for s in SETS:
            for w in ("front", "back", "y2025", "y2024"):
                vals = [idx[f"{base}|M|{s}"]["windows"][w].get("ex_EW_m")] + \
                       [idx[f"{base}|M@{k}|{s}"]["windows"][w].get("ex_EW_m") for k in OFFSETS]
                fin = [v for v in vals if v is not None]
                spread = (max(fin) - min(fin)) if len(fin) > 1 else None
                L.append(f"| {base}|{s} | {w} | " + " | ".join(fmt(v, 2) for v in vals) + f" | {fmt(spread, 2)} |")
    L += ["", "## (ii) 분기: 분기 첫날(Q) 대 둘째 달 10번째 거래일(Qmid) — ex_EW_m (기간 수)", "",
          "| 바탕|집합 | 구간 | Q | Qmid |", "|---|---|---|---|---|"]
    for base in BASES:
        for s in SETS:
            for w in ("front", "back", "y2026", "y2025", "y2024"):
                q, qm = idx[f"{base}|Q|{s}"]["windows"][w], idx[f"{base}|Qmid|{s}"]["windows"][w]
                L.append(f"| {base}|{s} | {w} | {fmt(q.get('ex_EW_m'), 2)} ({q.get('n_periods', 0)}) | "
                         f"{fmt(qm.get('ex_EW_m'), 2)} ({qm.get('n_periods', 0)}) |")
    L += ["", "## 보조(구간 시작 운용): 주기·날짜별 ex_EW_m", "",
          "| 바탕|집합 | 구간 | D | W | M | Q | M@1 | M@5 | M@10 | M@15 | M@20 | Qmid |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for base in BASES:
        for st in SETS:
            for w in WINDOWS:
                vals = []
                for c in ("D", "W", "M", "Q") + tuple(f"M@{k}" for k in OFFSETS) + ("Qmid",):
                    n = f"{base}|{c}|{st}"
                    vals.append(idx[n]["start"][w].get("ex_EW_m") if n in idx else None)
                L.append(f"| {base}|{st} | {w} | " + " | ".join(fmt(v, 2) for v in vals) + " |")
    with open(os.path.join(HERE, "results_cadence_tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
