"""7단계 갈래 MASAM(2026-10-11): 마삼룰 원형과 변형을 실제 보유 대상에 걸면 '덜 담기'보다 나은가.

사용자: "마삼룰이라고 있거든? 이거랑 컨셉이 비슷한데, 폭락 이후에는 어느 정도 기간 뒤에 기계적으로 매수하는 방법을 연구해볼래요. 다양하게 조합해서."
마삼룰 원문(sondoctor.co.kr/664, 조던·JD 부자연구소): 나스닥 종합 하루 −3% 이상 → 전량 매도, 마지막 −3% 날로부터 한 달 + 1일 뒤 재매수
(그 사이 −3%가 또 나오면 그날부터 다시 셈), 한 달 안에 −3%가 4번 이상이면 '공황' → 두 달 + 1일. 수정판: 8거래일 연속 상승이면 더 빨리 재매수.
원문은 '달'이 거래일인지 달력일인지 밝히지 않아 둘 다 잰다. 말뚝박기(낙폭 단계 매수)는 다른 갈래 몫이라 여기서 다루지 않는다.

상태 기계(사전 고정, 결과를 본 뒤 바꾸지 않는다)
  - 신호 r[t] = 신호 지수 t일 종가 / t−1일 종가 − 1. t일 종가에 판단 → 대상의 t 다음 거래일 시가 체결(harness.at_open과 같은 규칙:
    대상 거래일 k에는 k보다 엄격히 앞선 신호일의 상태를 쓴다).
  - 보유 상태에서 r[t] ≤ X → 현금 상태(매도 비율 s: 100%면 전량, 50%면 절반만 팔고 절반은 그대로 든다).
    현금 상태에서 r[t] ≤ X가 또 나오면 대기를 그날부터 다시 센다.
  - 재매수(거래일판): 마지막 X 날 D 뒤 N거래일 동안 X가 없으면 D+N+1 거래일 시가에 다시 산다(N=21 = '한 달 + 1일').
    달력판: D + 1개월 + 1일 이후 첫 거래일 시가.
  - 공황(k, W): D를 포함한 21거래일(달력판 1개월) 안의 X 날이 k번 이상이면 그 현금 기간 동안 대기를 W(거래일, 달력판 2개월)로.
    원문 '마지막 −3% 날로부터 두 달'을 따라 공황은 그 현금 기간 내내 유지하고 다시 사면 풀린다.
  - 수정판(8일): 현금 상태에서 신호 지수가 8거래일 연속 종가 상승하면 다음 시가에 바로 다시 산다.
  - 매도 비율 50%는 판 뒤 남은 주식을 그대로 들고(매일 맞추지 않음) 재매수 때 100%로 되돌린다.

격자(사전 고정) — 대상·신호 쌍마다 같은 93개(선언 94, 공황 K4W42는 원형 P1과 같아 1개 중복)
  원형 4: P1 21거래일·공황(4번/21거래일 → 42거래일), P2 달력 1개월·공황(4번/1개월 → 2개월), P3 = P1 + 8일, P4 = P2 + 8일. X = −3%, s = 100%.
  변형 72: X ∈ {−2, −2.5, −3, −3.5, −4, −5}% × N ∈ {5, 10, 15, 21, 30, 42}거래일 × s ∈ {100%, 50%} (공황·8일 없음).
  공황 6: k ∈ {3, 4, 5}회/21거래일 × 대기 {42, 63}거래일 (X = −3%, N = 21, s = 100%).
  8일 조기 재매수 12: X 6 × s 2, N = 21 (공황 없음). 8일 없는 짝은 변형 72 안에 있다.

대상(신호는 모두 나스닥 종합 ^IXIC, 보조로 대상 자체 지수)
  IXIC1  ^IXIC 1배(1971-03~).
  시가 규칙(모든 기초): 시가가 없거나 종가와 같거나 전날 종가와 같은 날(옛 지수의 '묵은 시가': ^IXIC 1971~84, ^NDX 1985~99의 약 90%,
         ^SOX 1994~2001의 약 90%)은 시가 = 종가로 둬 '다음 날 종가 체결'. 묵은 시가를 쓰면 갭 하락 전 가격(전날 종가)에 파는 셈이라
         1987-10-19 같은 날을 공짜로 피한다(첫 실행에서 확인, 결과를 보고 바꾼 것이 아니라 자료 오류 수정).
  NDX3   ^NDX(1985-10~1999-03-09) + QQQ(배당 수정, 1999-03-10~)를 이은 기초의 3배 합성(1985-11~, TQQQ 대리). 보수 0.95% + 추가 1.3%,
         금융비용 p7 ^IRX(1971~, lev5의 ^IRX는 1990~라 직접 씀). ^NDX만 쓰면 배당이 빠져 실제 TQQQ보다 연 2.8% 덜 벌어(점검 1회) QQQ를
         이었다. 1999년 전은 배당이 빠져 있다(연 0.5~1% × 3배 과소).
  SOXL3  lev5.splice('^SOX', 'SOXX', '2001-07-16') 3배 합성(1994-06~), 보수 0.95% + 추가 1.5%. 보조 신호 ^SOX −X%.
  KORU3  EWY(배당 수정) 3배 합성(2000-06~), 보수 0.95% + 추가 2.6%. 보조 신호 EWY(배당 수정 종가) −X%.
  PORT7  사용자 7종목(harness base('2026')·base('hold')). 시가→시가, 노출 e를 at_open으로 맞춘 overlay.
  실제 상품 대조: TQQQ(2010-02~)·SOXL(2010-03~)·KORU(2013-04~)에 ^IXIC 신호로 같은 93개.

평가(사전 고정)
  - 효율선초과 = 규칙 − (규칙의 최대 낙폭보다 깊지 않은 '덜 담기' w(0~1, 0.01 간격, 매일 맞춤) 가운데 최고). 짧은 창은 기간 수익 %p,
    긴 창은 연율 %p. 같은노출초과 = 규칙 − 같은 평균 노출 w의 덜 담기.
  - 2026 판정(G26): 2026 앞(1/2~5/29) 효율선초과 최대 → 뒤(6/1~10/9)·2025 확인. ^IXIC 신호는 2026 앞에 −2% 날만 5번이라 정보가 거의 없다.
  - 긴 역사 전방 판정(GL): 대상 시작~2005 효율선초과(연) 최대 → 2006~2025 확인. 2026은 고정한 GL의 결과만 보고.
  - 대표 후보 = NDX3(^IXIC 신호)의 GL.
  - 비용: 수수료 0, 미끄러짐 0.5%(|Δw|에 곱함, 효율선도 같은 조건). 현금 이자 0, 4%·^IRX 판 별도. 하루 늦게 읽기(lag 1) 민감도.
  - 휩소 = 판 뒤 다시 산 가격이 판 가격보다 비쌌던 주기 비율. 현금 기간 = 노출 < 1인 날 비율. 연 매매 = |Δw| ≥ 0.1인 날/년.
  - 사건형(^IXIC 1배·NDX3): 주기마다 '바로 매수'(첫 X 날 다음 시가) vs '마삼 재매수'(재매수 시가) vs '아무 날 매수'의 126거래일 수익·
    승률·그 안에 진입가 대비 −20% 이하로 빠진 비율. 1971~2025 주기만(2026은 따로).
"""
from __future__ import annotations

import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import lev5  # noqa: E402
import stock_lab as S  # noqa: E402

CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p7_MASAM"

# ============================================================================ 사전 고정 상수
X_GRID = (-0.02, -0.025, -0.03, -0.035, -0.04, -0.05)
N_GRID = (5, 10, 15, 21, 30, 42)
S_GRID = (1.0, 0.5)
PANIC_K, PANIC_W = (3, 4, 5), (42, 63)
UP_RUN = 8
FEE = 0.0095
EXTRA = {"NDX3": 0.013, "SOXL3": 0.015, "KORU3": 0.026, "TQQQ": 0.0, "SOXL": 0.0, "KORU": 0.0}
SLIP, CASH_ALT = 0.005, 0.04
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
TRADE_MIN = 0.1
END = "2026-10-09"
PICK_END, CHECK = "2005-12-31", ("2006-01-01", "2025-12-31")
H_EVT = 126
DD_EVT = -0.20

RUN0 = {"IXIC1": "1971-03-01", "NDX3": "1985-11-01", "SOXL3": "1994-06-01", "KORU3": "2000-06-01",
        "TQQQ": "2010-03-15", "SOXL": "2010-04-12", "KORU": "2013-05-10"}
PAIRS = [("IXIC1", "IXIC"), ("NDX3", "IXIC"), ("NDX3", "NDX"), ("SOXL3", "IXIC"), ("SOXL3", "SOX"),
         ("KORU3", "IXIC"), ("KORU3", "EWY")]
REAL = ["TQQQ", "SOXL", "KORU"]
TNAME = {"IXIC1": "나스닥 종합 1배", "NDX3": "3배 나스닥100 합성(TQQQ 대리)", "SOXL3": "SOXL 합성", "KORU3": "KORU 합성",
         "PORT7": "7종목", "TQQQ": "실제 TQQQ", "SOXL": "실제 SOXL", "KORU": "실제 KORU"}
SNAME = {"IXIC": "^IXIC", "NDX": "^NDX", "SOX": "^SOX", "EWY": "EWY"}

WIN = {  # [첫날, 끝날]: 첫날 직전 종가 → 끝날 종가
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", END), "2026_7월": ("2026-06-23", "2026-07-29"),
    "2025": ("2025-01-02", "2025-12-31"), "2022": ("2022-01-03", "2022-12-30"), "2020": ("2020-01-02", "2020-12-31"),
    "2010~19": ("2010-01-04", "2019-12-31"), "2008": ("2008-01-02", "2008-12-31"), "2000~02": ("2000-03-01", "2002-10-31"),
    "1990s": ("1990-01-02", "1999-12-31"), "1987": ("1987-01-02", "1987-12-31"), "1971~86": ("1971-03-01", "1986-12-31"),
    "고르기_긴": (None, PICK_END), "확인_긴": CHECK, "긴역사": (None, END),
}
LONGW = {"고르기_긴", "확인_긴", "긴역사", "1971~86", "1990s", "2010~19", "실제전체"}
ERAS = [("1971~86", "1971-01-01", "1986-12-31"), ("1987", "1987-01-01", "1987-12-31"), ("1988~99", "1988-01-01", "1999-12-31"),
        ("2000~02", "2000-01-01", "2002-12-31"), ("2003~07", "2003-01-01", "2007-12-31"), ("2008~09", "2008-01-01", "2009-12-31"),
        ("2010~19", "2010-01-01", "2019-12-31"), ("2020~21", "2020-01-01", "2021-12-31"), ("2022~24", "2022-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]


# ============================================================================ 규칙
def specs():
    out = []
    base = {"X": -0.03, "s": 1.0, "up8": False}
    out.append({**base, "id": "P1", "fam": "원형", "mode": "td", "N": 21, "panic": (4, 42),
                "label": "원형 P1: −3% → 21거래일, 공황 4번/21일 → 42거래일"})
    out.append({**base, "id": "P2", "fam": "원형", "mode": "cal", "N": 1, "panic": (4, 2),
                "label": "원형 P2: −3% → 달력 1개월+1일, 공황 4번/1개월 → 2개월"})
    out.append({**base, "id": "P3", "fam": "원형", "mode": "td", "N": 21, "panic": (4, 42), "up8": True,
                "label": "원형 P3: P1 + 8일 연속 상승 즉시 재매수"})
    out.append({**base, "id": "P4", "fam": "원형", "mode": "cal", "N": 1, "panic": (4, 2), "up8": True,
                "label": "원형 P4: P2 + 8일 연속 상승 즉시 재매수"})
    for X in X_GRID:
        for N in N_GRID:
            for s in S_GRID:
                out.append({"id": f"G{-X * 100:g}_{N}_{int(s * 100)}", "fam": "격자", "mode": "td", "X": X, "N": N, "s": s,
                            "panic": None, "up8": False, "label": f"{X * 100:g}% → {N}거래일, {s:.0%} 매도"})
    for k in PANIC_K:
        for w in PANIC_W:
            out.append({"id": f"K{k}W{w}", "fam": "공황", "mode": "td", "X": -0.03, "N": 21, "s": 1.0, "panic": (k, w),
                        "up8": False, "label": f"−3% → 21거래일, 공황 {k}번/21일 → {w}거래일"})
    for X in X_GRID:
        for s in S_GRID:
            out.append({"id": f"U{-X * 100:g}_{int(s * 100)}", "fam": "8일", "mode": "td", "X": X, "N": 21, "s": s,
                        "panic": None, "up8": True, "label": f"{X * 100:g}% → 21거래일 + 8일 연속 상승, {s:.0%} 매도"})
    declared = len(out)
    seen, uniq = {}, []
    for sp in out:
        key = (sp["mode"], sp["X"], sp["N"], sp["s"], sp["panic"], sp["up8"])
        if key in seen:
            seen[key]["dup"] = seen[key].get("dup", []) + [sp["id"]]
            continue
        seen[key] = sp
        uniq.append(sp)
    return uniq, declared


# ============================================================================ 자료
_P7 = None


def p7() -> pd.DataFrame:
    global _P7
    if _P7 is None:
        _P7 = pd.read_parquet(CACHE / "p7_idx.parquet")
    return _P7


def fix_open(b: S.Bars) -> S.Bars:
    """시가가 없거나(≤0), 종가와 같거나, 전날 종가와 같으면(옛 지수 자료의 '묵은 시가') 그날 시가 = 종가로 둔다.
    그런 날 시가 체결은 곧 '그날 종가 체결'(= 신호 다음 날 종가)이 된다. 묵은 시가를 그대로 쓰면 갭 하락 전 가격에 파는 셈이 된다."""
    pc = np.concatenate([[np.nan], b.c[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = ~np.isfinite(b.o) | (b.o <= 0) | (np.abs(b.o / b.c - 1) < 1e-9) | (np.abs(b.o / pc - 1) < 1e-6)
    o = np.where(bad, b.c, b.o)
    h = np.where(np.isfinite(b.h), b.h, b.c)
    lo = np.where(np.isfinite(b.l), b.l, b.c)
    nb = S.Bars(b.sym, b.days, o, np.maximum.reduce([h, o, b.c]), np.minimum.reduce([lo, o, b.c]), b.c, b.v)
    nb._cache["bad_open"] = bad
    return nb


def p7_bars(sym: str, adj: bool = False) -> S.Bars:
    g = p7()
    g = g[g.sym == sym].sort_values("date").dropna(subset=["Close"])
    g = g[g.Close > 0]
    f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy() if adj else np.ones(len(g))
    return fix_open(S.Bars(sym, g.date.to_numpy(), g.Open.to_numpy(float) * f, g.High.to_numpy(float) * f,
                           g.Low.to_numpy(float) * f, g.Close.to_numpy(float) * f, g.Volume.fillna(0).to_numpy(float)))


def splice2(a: S.Bars, b: S.Bars, at: str, name: str) -> S.Bars:
    """at 전에는 a, at부터는 b(lev5.splice와 같은 방식, 가격 수준은 b에 맞춤)."""
    ia, ib = a.idx(at), b.idx(at)
    sc = b.c[ib] / a.c[ia]
    cat = lambda x, y: np.concatenate([x[:ia] * sc, y[ib:]])
    nb = S.Bars(name, np.concatenate([a.days[:ia], b.days[ib:]]), cat(a.o, b.o), cat(a.h, b.h), cat(a.l, b.l),
                cat(a.c, b.c), np.concatenate([a.v[:ia], b.v[ib:]]))
    nb._cache["bad_open"] = np.concatenate([a._cache["bad_open"][:ia], b._cache["bad_open"][ib:]])
    return nb


def irx(days) -> np.ndarray:
    g = p7()
    s = g[g.sym == "^IRX"].set_index("date")["Close"].astype(float).sort_index()
    s = s[~s.index.duplicated()] / 100.0
    return s.reindex(sorted(set(s.index) | set(days))).ffill().reindex(days).fillna(0.0).to_numpy()


def synth(u: S.Bars, L=3.0, fee=FEE, extra=0.0) -> S.Bars:
    """lev5.synth와 같은 식, 금융비용만 p7 ^IRX(1971~)."""
    r_f = irx(u.days)
    n = len(u.c)
    c = np.empty(n); o = np.empty(n); h = np.empty(n); lo = np.empty(n)
    c[0] = o[0] = h[0] = lo[0] = 100.0
    for t in range(1, n):
        pc = u.c[t - 1]
        r = L * (u.c[t] / pc - 1) - (L - 1) * r_f[t] / 252 - (fee + extra) / 252
        c[t] = c[t - 1] * max(1 + r, 0.001)
        o[t] = c[t - 1] * max(1 + L * (u.o[t] / pc - 1), 0.001)
        hi = c[t - 1] * max(1 + L * (u.h[t] / pc - 1), 0.001)
        lw = c[t - 1] * max(1 + L * (u.l[t] / pc - 1), 0.001)
        h[t], lo[t] = max(hi, o[t], c[t]), min(lw, o[t], c[t])
    return S.Bars(f"{u.sym}x{L:g}", u.days.copy(), o, h, lo, c, u.v.copy())


def ndx_under(at: str = "1999-03-10") -> S.Bars:
    """^NDX(가격 지수)를 QQQ(배당 수정) 첫날 전까지 쓰고, 그 뒤는 QQQ."""
    return splice2(p7_bars("^NDX"), p7_bars("QQQ", adj=True), at, "^NDX+QQQ")


def sox_under(at: str = "2001-07-16") -> S.Bars:
    """lev5.splice('^SOX', 'SOXX', '2001-07-16')와 같은 기초. 묵은 시가만 fix_open으로 고침."""
    return splice2(p7_bars("^SOX"), fix_open(lev5.bars("SOXX")), at, "^SOX+SOXX")


def signal(name: str):
    """신호 지수 (날짜, 종가 일 수익률). EWY는 배당 수정 종가."""
    b = lev5.bars("EWY") if name == "EWY" else p7_bars(SNAME[name])
    days, c = b.days, b.c
    r = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
    return np.asarray(days), r


def targets():
    T = {}
    T["IXIC1"] = p7_bars("^IXIC")
    T["NDX3"] = synth(ndx_under(), 3, FEE, EXTRA["NDX3"])
    T["SOXL3"] = synth(sox_under(), 3, FEE, EXTRA["SOXL3"])
    T["KORU3"] = synth(fix_open(lev5.bars("EWY")), 3, FEE, EXTRA["KORU3"])
    for k, u in (("NDX3", ndx_under()), ("SOXL3", sox_under()), ("KORU3", fix_open(lev5.bars("EWY")))):
        T[k]._cache["bad_open"] = u._cache["bad_open"]
    for p in REAL:
        T[p] = lev5.bars(p)
    return T


# ============================================================================ 상태 기계
class Cal:
    def __init__(self, days):
        d = pd.to_datetime(pd.Series(days))
        self.rel = {m: np.searchsorted(days, (d + pd.DateOffset(months=m) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").to_numpy(),
                                       "left") for m in (1, 2)}
        self.cstart = np.searchsorted(days, (d - pd.DateOffset(months=1)).dt.strftime("%Y-%m-%d").to_numpy(), "right")


def states(r: np.ndarray, sp: dict, cal: Cal | None) -> np.ndarray:
    """St[t] = t일 종가 판단 뒤 상태(1 보유, 0 현금 상태). 다음 거래일 시가에 적용."""
    n = len(r)
    St = np.ones(n, dtype=np.int8)
    X, N, mode, panic, up8 = sp["X"], sp["N"], sp["mode"], sp["panic"], sp["up8"]
    inv, pan, release, up = True, False, -1, 0
    trig: list = []
    for t in range(n):
        x = r[t]
        if x != x:
            St[t] = 1 if inv else 0
            continue
        up = up + 1 if x > 0 else 0
        if x <= X:
            trig.append(t)
            if inv:
                inv, pan = False, False
            if panic:
                k, w = panic
                lo = t - 20 if mode == "td" else int(cal.cstart[t])
                if len(trig) - bisect.bisect_left(trig, lo) >= k:
                    pan = True
            if mode == "td":
                release = t + (panic[1] if pan else N) + 1
            else:
                release = int(cal.rel[panic[1] if pan else N][t])
        elif not inv:
            if (up8 and up >= UP_RUN) or t + 1 >= release:
                inv = True
        St[t] = 1 if inv else 0
    return St


def map_states(sdays, St, tdays, lag=0) -> np.ndarray:
    """대상 거래일 k에 쓸 상태 = k보다 엄격히 앞선 신호일의 상태(lag만큼 더 늦게)."""
    j = np.searchsorted(sdays, tdays, "left") - 1 - lag
    return np.where(j >= 0, St[np.clip(j, 0, None)], 1).astype(float)


# ============================================================================ 엔진(변형을 열로 묶어 한 번에)
def engine(o, c, a, z, tgt, slip=0.0, cash=None):
    """a..z 거래일. a−1 종가에 전량 보유(가치 1). tgt[t, v] = t일 시가 목표 비율(nan = 그대로).
    반환 E(종가 가치, [n+1, V]), W(종가 비율), TO(그날 |Δw|)."""
    V = tgt.shape[1]
    n = z - a + 1
    E = np.empty((n + 1, V)); W = np.empty((n, V)); TO = np.zeros((n, V))
    E[0] = 1.0
    csh = np.zeros(V)
    sh = np.full(V, 1.0 / c[a - 1])
    for k in range(n):
        t = a + k
        ot = o[t]
        tg = tgt[t]
        nn = ~np.isnan(tg)
        if nn.any():
            eq = csh + sh * ot
            cur = sh * ot / eq
            d = np.where(nn, np.nan_to_num(tg) - cur, 0.0)
            d[np.abs(d) <= 1e-9] = 0.0
            neg = d < 0
            if neg.any():
                sell = np.where(neg, -d * eq / ot, 0.0)
                csh = csh + sell * ot * (1 - slip)
                sh = sh - sell
                sh[neg & (np.nan_to_num(tg, nan=1.0) <= 1e-9)] = 0.0
            pos = d > 0
            if pos.any():
                buy = np.where(pos, np.minimum(d * eq, csh), 0.0)
                sh = sh + buy / (ot * (1 + slip))
                csh = csh - buy
            TO[k] = np.abs(d)
        if cash is not None:
            csh = csh * (1 + cash[t] / 252)
        E[k + 1] = csh + sh * c[t]
        W[k] = sh * c[t] / E[k + 1]
    return {"E": E, "W": W, "TO": TO, "a": a, "z": z}


def to_tgt(ex: np.ndarray, a: int) -> np.ndarray:
    """노출 상태 ex[t, v] → 바뀐 날만 목표(nan = 그대로). 첫날(a)은 상태 그대로."""
    tg = np.full(ex.shape, np.nan)
    ch = np.zeros(ex.shape, bool)
    ch[a + 1:] = ex[a + 1:] != ex[a:-1]
    ch[a] = True
    tg[ch] = ex[ch]
    return tg


# ============================================================================ 창 통계
def win_bounds(P, a, z, wn, w):
    d0, d1 = w
    if d0 is None:
        d0 = P.days[a]
    i0, i1 = P.idx(d0), P.idx(d1, "right") - 1
    i1 = min(i1, z)
    if i1 < a or i1 < i0:
        return None
    part = i0 < a
    i0 = max(i0, a)
    if i1 - i0 < 15:
        return None
    return i0, i1, part


def wstats(res, P, ex, wb):
    i0, i1, part = wb
    a = res["a"]
    k0, k1 = i0 - a, i1 - a
    E = res["E"][k0:k1 + 2]
    n = k1 - k0 + 1
    yrs = n / 252
    g = E[-1] / E[0]
    out = {"수익": g - 1, "낙폭": (E / np.maximum.accumulate(E, axis=0) - 1).min(axis=0),
           "연율": np.where(g > 0, np.abs(g) ** (1 / yrs) - 1, -1.0),
           "평균노출": res["W"][k0:k1 + 1].mean(axis=0), "연매매": (res["TO"][k0:k1 + 1] >= TRADE_MIN).sum(axis=0) / yrs,
           "부분": part, "n": n}
    if ex is not None:
        out["현금기간"] = (ex[i0:i1 + 1] < 0.999).mean(axis=0)
    return out


def excess(st, fs, long):
    kf = fs["연율"] if long else fs["수익"]
    val = st["연율"] if long else st["수익"]
    ok = fs["낙폭"][None, :] >= st["낙폭"][:, None] - 1e-12
    M = np.where(ok, kf[None, :], -np.inf)
    j = M.argmax(axis=1)
    jj = np.clip(np.round(np.clip(st["평균노출"], 0, 1) * 100).astype(int), 0, 100)
    return val - M.max(axis=1), W_GRID[j], val - kf[jj]


def cycles(ex_col, px, a, z, days):
    """노출 열(대상 거래일) → 주기 [(판 날, 판 가격, 산 날, 산 가격)]. 산 날 없으면 None."""
    e = ex_col[a:z + 1]
    prev = np.concatenate([[1.0], e[:-1]])
    out, cur = [], None
    for k in np.nonzero(e != prev)[0]:
        t = a + k
        if e[k] < prev[k] and cur is None:
            cur = [t, px[t]]
        elif e[k] > prev[k] and cur is not None:
            out.append((days[cur[0]], cur[1], days[t], px[t], cur[0], t))
            cur = None
    if cur is not None:
        out.append((days[cur[0]], cur[1], None, None, cur[0], None))
    return out


def whipsaw(cyc, d0, d1):
    cc = [x for x in cyc if d0 <= x[0] <= d1 and x[2] is not None]
    if not cc:
        return {"주기": 0, "휩소율": float("nan"), "되산가차": float("nan")}
    g = np.array([x[3] / x[1] - 1 for x in cc])
    return {"주기": len(cc), "휩소율": float((g > 0).mean()), "되산가차": float(g.mean())}


# ============================================================================ 대상·신호 쌍 실행
def run_pair(P, tname, sname, SP, sig_cache, cfgs=("base", "slip", "cash4", "irx", "lag1"), frontier_cache=None):
    sdays, r, St_all = sig_cache[sname]
    a = P.idx(RUN0[tname])
    z = P.idx(END, "right") - 1
    ex = np.stack([np.where(map_states(sdays, St_all[i], P.days) > 0.5, 1.0, 1.0 - sp["s"]) for i, sp in enumerate(SP)], 1)
    ex1 = np.stack([np.where(map_states(sdays, St_all[i], P.days, lag=1) > 0.5, 1.0, 1.0 - sp["s"]) for i, sp in enumerate(SP)], 1)
    rfz = irx(P.days)
    cash_of = {"base": None, "slip": None, "cash4": np.full(len(P.c), CASH_ALT), "irx": rfz, "lag1": None}
    slip_of = {"base": 0.0, "slip": SLIP, "cash4": 0.0, "irx": 0.0, "lag1": 0.0}
    wins = {wn: win_bounds(P, a, z, wn, w) for wn, w in WIN.items()}
    if tname in REAL:
        wins["실제전체"] = win_bounds(P, a, z, "실제전체", (None, END))
    wins = {k: v for k, v in wins.items() if v is not None}
    out = {"cfg": {}, "wins": {k: (P.days[v[0]], P.days[v[1]], v[2]) for k, v in wins.items()}}
    for cf in cfgs:
        fkey = (tname, "base" if cf == "lag1" else cf)
        if fkey not in frontier_cache:
            fr = engine(P.o, P.c, a, z, np.tile(W_GRID, (len(P.c), 1)), slip_of[fkey[1]], cash_of[fkey[1]])
            frontier_cache[fkey] = {wn: wstats(fr, P, None, wb) for wn, wb in wins.items()}
        fst = frontier_cache[fkey]
        e_use = ex1 if cf == "lag1" else ex
        res = engine(P.o, P.c, a, z, to_tgt(e_use, a), slip_of[cf], cash_of[cf])
        rows = {}
        for wn, wb in wins.items():
            st = wstats(res, P, e_use, wb)
            ex_, w_, same = excess(st, fst[wn], wn in LONGW)
            st["효율선초과"], st["효율선_w"], st["같은노출초과"] = ex_, w_, same
            rows[wn] = st
        out["cfg"][cf] = rows
        if cf == "base":
            out["E_base"] = res
            out["E_hold_frontier"] = fst
    out["ex"] = ex
    out["a"], out["z"] = a, z
    out["cycles"] = [cycles(ex[:, i], P.o, a, z, P.days) for i in range(len(SP))]
    return out


def run_port(SP, sig_cache):
    """7종목: harness 기본 경로에 노출 e(at_open)를 곱한다. 현금 0%(4% 판 따로), 미끄러짐은 cost × 회전."""
    sdays, r, St_all = sig_cache["IXIC"]
    out = {}
    for bn in ("2026", "hold"):
        b = H.base(bn)
        ret = b.ret
        E, E1 = {}, {}
        for i, sp in enumerate(SP):
            s = pd.Series(St_all[i].astype(float), index=sdays)
            E[sp["id"]] = H.at_open(s, b).map(lambda v: 1.0 if v > 0.5 else 1.0 - sp["s"])
            s1 = pd.Series(np.concatenate([[1.0], St_all[i][:-1].astype(float)]), index=sdays)   # 하루 늦게 읽기
            E1[sp["id"]] = H.at_open(s1, b).map(lambda v: 1.0 if v > 0.5 else 1.0 - sp["s"])
        rows = {}
        for cf, cost, cash in (("base", 0.0, 0.0), ("slip", SLIP, 0.0), ("cash4", 0.0, CASH_ALT), ("lag1", 0.0, 0.0)):
            fr = {}
            for w in W_GRID:
                rr = w * ret + (1 - w) * cash / 252
                fr[w] = {k: H.stats(rr, *v) for k, v in b.periods.items()}
            res = {}
            for sp in SP:
                e = (E1 if cf == "lag1" else E)[sp["id"]].reindex(ret.index)
                rr = H.overlay(b, e, cost=cost) + (1 - e) * cash / 252
                m = {}
                for k, v in b.periods.items():
                    st = H.stats(rr, *v)
                    ew = e[(e.index >= v[0]) & (e.index <= v[1])]
                    st["평균노출"] = float(ew.mean()) if len(ew) else float("nan")
                    st["현금기간"] = float((ew < 0.999).mean()) if len(ew) else float("nan")
                    st["매도횟수"] = int(((ew.shift(1).fillna(1.0) > ew + 1e-9)).sum())
                    okw = [w for w in W_GRID if fr[w][k]["낙폭"] >= st["낙폭"] - 1e-12]
                    bw = max(okw, key=lambda w: fr[w][k]["수익"])
                    st["효율선초과"] = st["수익"] - fr[bw][k]["수익"]
                    st["효율선_w"] = float(bw)
                    jw = W_GRID[int(round(min(max(st["평균노출"], 0), 1) * 100))]
                    st["같은노출초과"] = st["수익"] - fr[jw][k]["수익"]
                    m[k] = st
                res[sp["id"]] = m
            rows[cf] = res
        out[bn] = {"rows": rows, "E": E, "open": pd.Series(b.open, index=b.days)}
    return out


# ============================================================================ 사건형
def event_study(P, ex_col, a, z, d_lo="1900-01-01", d_hi="2025-12-31"):
    o, c, days = P.o, P.c, P.days
    last = len(c) - 1

    def fwd(k):
        if k is None or k + H_EVT - 1 > last:
            return None
        seg = c[k:k + H_EVT]
        return seg[-1] / o[k] - 1, seg.min() / o[k] - 1

    cyc = cycles(ex_col, o, a, z, days)
    rows = []
    for (d_s, p_s, d_b, p_b, ks, kb) in cyc:
        if not (d_lo <= d_s <= d_hi) or kb is None:
            continue
        fi, fr = fwd(ks), fwd(kb)
        if fi is None or fr is None:
            continue
        rows.append({"판날": d_s, "산날": d_b, "되산가차": p_b / p_s - 1, "바로": fi[0], "바로_dd": fi[1], "마삼": fr[0], "마삼_dd": fr[1]})
    ks = [k for k in range(a, last - H_EVT + 2) if d_lo <= days[k] <= d_hi]
    base = np.array([c[k + H_EVT - 1] / o[k] - 1 for k in ks])
    based = np.array([c[k:k + H_EVT].min() / o[k] - 1 for k in ks])
    return rows, {"n": len(ks), "평균": float(base.mean()), "승률": float((base > 0).mean()), "dd20": float((based <= DD_EVT).mean())}


def ev_summary(rows):
    if not rows:
        return {"n": 0}
    f = lambda k: np.array([r[k] for r in rows])
    return {"n": len(rows), "되산가차": float(f("되산가차").mean()), "싸게되산": float((f("되산가차") < 0).mean()),
            "마삼": float(f("마삼").mean()), "마삼_승률": float((f("마삼") > 0).mean()), "마삼_dd20": float((f("마삼_dd") <= DD_EVT).mean()),
            "바로": float(f("바로").mean()), "바로_승률": float((f("바로") > 0).mean()), "바로_dd20": float((f("바로_dd") <= DD_EVT).mean())}


# ============================================================================ 점검
def checks(T, SP, sig_cache):
    msgs = []
    sd, r = signal("IXIC")
    n3 = int(np.nansum(r <= -0.03))
    d26 = [d for d, x in zip(sd, r) if d >= "2026-01-01" and x <= -0.03]
    msgs.append(f"^IXIC −3% 이하 날 {n3}번(1971~2026), 2026: {d26}")
    bad = T["IXIC1"]._cache["bad_open"]
    yrs = pd.Series(bad, index=pd.Series(T["IXIC1"].days).str[:4].to_numpy()).groupby(level=0).mean()
    msgs.append("^IXIC 시가 없음·묵은 시가(=종가 체결로 둠) 비율: 1971~1983 " + f"{yrs.loc['1971':'1983'].mean():.0%}, 1984 {yrs.loc['1984']:.0%}, "
                f"1985~2026 {bad[T['IXIC1'].idx('1985-01-01'):].mean():.1%}")
    for k, lo, hi in (("NDX3", "1985-10-01", "1999-03-09"), ("NDX3", "1999-03-10", END), ("SOXL3", "1994-05-01", "2001-07-13"),
                      ("SOXL3", "2001-07-16", END), ("KORU3", "2000-05-01", END)):
        P = T[k]
        bb = P._cache["bad_open"][P.idx(lo):P.idx(hi, "right")]
        msgs.append(f"{k} 기초 {lo[:4]}~{hi[:4]} 묵은·없는 시가(→ 다음 날 종가 체결) 비율 {bb.mean():.0%}")
    # 1) 엔진 대조: stock_lab.simulate(시가 목표 비율)과 같은 값인지
    P = T["NDX3"]
    a, z = P.idx("2015-01-02"), P.idx(END, "right") - 1
    sdays, _, St_all = sig_cache["IXIC"]
    for i in (0, 1, 5, 41):
        sp = SP[i]
        ex = np.where(map_states(sdays, St_all[i], P.days) > 0.5, 1.0, 1.0 - sp["s"])
        tg = to_tgt(ex[:, None], a)[:, 0]

        class Pol(S.Policy):
            def on_open(self, b, t, st):
                v = tg[t]
                return None if v != v else float(v)
        v1, _, _ = S.simulate(P, a, z, Pol(), SLIP)
        v2 = engine(P.o, P.c, a, z, tg[:, None], SLIP)["E"][:, 0]
        msgs.append(f"엔진 대조 {sp['id']}(미끄러짐 0.5%): stock_lab.simulate와 가치 최대 상대차 {np.max(np.abs(v1 / v2 - 1)):.1e}")
    # 2) 합성 대조
    for name, prod in (("NDX3", "TQQQ"), ("SOXL3", "SOXL"), ("KORU3", "KORU")):
        v = lev5.validate(prod, T[name])
        msgs.append(f"합성 {name} vs 실제 {prod}: 일수익 상관 {v['일수익상관']:.4f}, 남은 연 로그 비용 차 {v['연비용_로그']:+.4f}, "
                    f"낙폭 실제 {v['실제_낙폭']:.3f} / 합성 {v['합성_낙폭']:.3f} ({v['시작']}~, {v['일수']}일)")
    s5 = lev5.synth(lev5.splice("^SOX", "SOXX", "2001-07-16"), 3, FEE, EXTRA["SOXL3"])  # 종가만 비교(시가는 묵은 시가 고침)
    j = T["SOXL3"].idx("1995-01-03")
    msgs.append(f"SOXL3 이 스크립트 합성 vs lev5.synth(1995~): 종가 최대 상대차 "
                f"{np.max(np.abs((T['SOXL3'].c[j:] / T['SOXL3'].c[j]) / (s5.c[j:] / s5.c[j]) - 1)):.1e}")
    # 3) 미래 교란: 2020-03-02 뒤 신호를 흔들어도 그날까지 상태·가치가 같아야 한다
    sd, r = signal("IXIC")
    Tk = int(np.searchsorted(sd, "2020-03-02"))
    rng = np.random.default_rng(7)
    r2 = r.copy()
    r2[Tk + 1:] = rng.normal(0, 0.03, len(r) - Tk - 1)
    cal = Cal(sd)
    worst = 0
    for sp in SP:
        s1, s2 = states(r, sp, cal), states(r2, sp, cal)
        worst = max(worst, int(np.abs(s1[:Tk + 1].astype(int) - s2[:Tk + 1].astype(int)).max()))
    msgs.append(f"미래 교란(2020-03-02 뒤 ^IXIC 수익을 무작위로) 규칙 {len(SP)}개: 그날까지 상태 차 {worst}(0이어야 함)")
    # 4) 보유(전 기간 1)는 대상 종가 비율과 같아야
    P = T["IXIC1"]
    a, z = P.idx(RUN0["IXIC1"]), P.idx(END, "right") - 1
    e = engine(P.o, P.c, a, z, to_tgt(np.ones((len(P.c), 1)), a))["E"][:, 0]
    msgs.append(f"보유 재현(^IXIC): 가치/종가비 최대 차 {np.max(np.abs(e[1:] - P.c[a:z + 1] / P.c[a - 1])):.1e}")
    return msgs


# ============================================================================ 본 실행
def main():
    t0 = time.time()
    SP, declared = specs()
    ids = [sp["id"] for sp in SP]
    sig_cache = {}
    for sn in ("IXIC", "NDX", "SOX", "EWY"):
        sd, r = signal(sn)
        cal = Cal(sd)
        sig_cache[sn] = (sd, r, [states(r, sp, cal) for sp in SP])
    print(f"상태 기계 {len(SP)}개 × 신호 4: {time.time() - t0:.1f}s")
    T = targets()
    fcache = {}
    R = {}
    for tn, sn in PAIRS:
        R[(tn, sn)] = run_pair(T[tn], tn, sn, SP, sig_cache, frontier_cache=fcache)
        print(f"{tn}·{sn} 완료 {time.time() - t0:.1f}s")
    for p in REAL:
        R[(p, "IXIC")] = run_pair(T[p], p, "IXIC", SP, sig_cache, cfgs=("base", "slip"), frontier_cache=fcache)
    PORT = run_port(SP, sig_cache)
    print(f"7종목 완료 {time.time() - t0:.1f}s")
    msgs = checks(T, SP, sig_cache)
    for m in msgs:
        print(" ·", m)

    # ---------------- 고르기
    def col(tn, sn, cf, wn, key):
        return R[(tn, sn)]["cfg"][cf][wn][key]

    picks = {}
    shares = {}
    for (tn, sn) in PAIRS:
        rows = R[(tn, sn)]["cfg"]["base"]
        exL = rows["고르기_긴"]["효율선초과"]
        gl = int(np.argmax(exL))
        ex26 = rows["2026앞"]["효율선초과"]
        g26 = int(np.argmax(ex26))
        picks[(tn, sn)] = {"GL": gl, "G26": g26, "G26_max": float(ex26[g26]),
                           "G26_traded": int((rows["2026앞"]["연매매"] > 0).sum())}
        okL = exL > 0
        ok26 = ex26 > 1e-9
        shares[(tn, sn)] = {
            "긴_앞>0": int(okL.sum()), "긴_앞>0중_확인>0": int((okL & (rows["확인_긴"]["효율선초과"] > 0)).sum()),
            "긴_무조건_확인>0": int((rows["확인_긴"]["효율선초과"] > 0).sum()),
            "26_앞>0": int(ok26.sum()), "26_앞>0중_뒤>0": int((ok26 & (rows["2026뒤"]["효율선초과"] > 1e-9)).sum()),
            "26_앞>0중_2025>0": int((ok26 & (rows["2025"]["효율선초과"] > 1e-9)).sum()),
            "26_무조건_뒤>0": int((rows["2026뒤"]["효율선초과"] > 1e-9).sum()),
            "26_무조건_2025>0": int((rows["2025"]["효율선초과"] > 1e-9).sum())}
    pr = PORT["2026"]["rows"]["base"]
    pex = np.array([pr[i]["앞"]["효율선초과"] for i in ids])
    hr = PORT["hold"]["rows"]["base"]
    ok26 = pex > 1e-9
    pb = np.array([pr[i]["뒤"]["효율선초과"] for i in ids]) > 1e-9
    p25 = np.array([hr[i]["2025"]["효율선초과"] for i in ids]) > 1e-9
    shares[("PORT7", "IXIC")] = {"26_앞>0": int(ok26.sum()), "26_앞>0중_뒤>0": int((ok26 & pb).sum()),
                                 "26_앞>0중_2025>0": int((ok26 & p25).sum()), "26_무조건_뒤>0": int(pb.sum()),
                                 "26_무조건_2025>0": int(p25.sum())}
    picks[("PORT7", "IXIC")] = {"G26": int(np.argmax(pex)), "G26_max": float(pex.max()),
                                "G26_traded": int(sum(pr[i]["앞"]["매도횟수"] > 0 for i in ids))}
    rep = picks[("NDX3", "IXIC")]["GL"]

    # ---------------- 사건형
    EV = {}
    for tn in ("IXIC1", "NDX3"):
        Rr = R[(tn, "IXIC")]
        P = T[tn]
        bs = None
        cells = {}
        for i, sp in enumerate(SP):
            if not (sp["fam"] in ("원형",) or (sp["fam"] == "격자" and sp["s"] == 1.0)):
                continue
            rows, bs = event_study(P, Rr["ex"][:, i], Rr["a"], Rr["z"])
            cells[sp["id"]] = ev_summary(rows)
            if sp["id"] == "P1":
                cells["P1_rows"] = rows
                era = {}
                for en, lo, hi in ERAS:
                    era[en] = ev_summary([x for x in rows if lo <= x["판날"] <= hi])
                cells["P1_era"] = era
                rows26, _ = event_study(P, Rr["ex"][:, i], Rr["a"], Rr["z"], "2026-01-01", "2026-12-31")
                cells["P1_2026"] = rows26
        era_base = {}
        for en, lo, hi in ERAS:
            _, b_ = event_study(P, Rr["ex"][:, 0], Rr["a"], Rr["z"], lo, hi)
            era_base[en] = b_
        EV[tn] = {"cells": cells, "base": bs, "era_base": era_base}

    # ---------------- 2026 날짜 추적
    def trace(tn, sn, i, lo="2025-12-01"):
        Rr = R[(tn, sn)]
        P = T[tn]
        ex = Rr["ex"][:, i]
        sdays, rs, St = sig_cache[sn]
        out = []
        for k in range(P.idx(lo), Rr["z"] + 1):
            if ex[k] != ex[k - 1]:
                j = int(np.searchsorted(sdays, P.days[k], "left") - 1)
                # 그 상태를 만든 신호일과 직전 X 날
                out.append({"체결": P.days[k], "노출": float(ex[k]), "판단일": sdays[j], "판단일_신호수익": float(rs[j]),
                            "가격": float(P.o[k])})
        return out

    TR = {}
    for (tn, sn) in [("IXIC1", "IXIC"), ("NDX3", "IXIC"), ("SOXL3", "SOX"), ("KORU3", "EWY"), ("SOXL3", "IXIC")]:
        TR[f"{tn}·{sn}"] = {SP[i]["id"]: trace(tn, sn, i) for i in sorted({0, 1, 2, 3, rep, picks[(tn, sn)]["G26"]})}
    ptr = {}
    for i in sorted({0, 1, 2, 3, rep, picks[("PORT7", "IXIC")]["G26"]}):
        e = PORT["2026"]["E"][SP[i]["id"]]
        ch = e[e.ne(e.shift())]
        ptr[SP[i]["id"]] = [(d, float(v)) for d, v in ch.items()][1:] if len(ch) else []
    TR["PORT7·IXIC"] = ptr

    payload = build_payload(SP, declared, R, PORT, picks, shares, EV, TR, msgs, T)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_jd), encoding="utf-8")
    md = build_md(SP, declared, R, PORT, picks, shares, EV, TR, msgs, T)
    (RESULTS / f"{ID}.md").write_text(md, encoding="utf-8")
    print("저장", RESULTS / f"{ID}.json", RESULTS / f"{ID}.md", f"{time.time() - t0:.1f}s")


def _jd(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def _f(x, d=4):
    try:
        x = float(x)
    except Exception:
        return x
    return None if x != x else round(x, d)


def row_of(R, key, cf, i, wins=None):
    out = {}
    for wn, st in R[key]["cfg"][cf].items():
        if wins and wn not in wins:
            continue
        out[wn] = {k: _f(v[i] if isinstance(v, np.ndarray) else v) for k, v in st.items() if k not in ("n",)}
    return out


def build_payload(SP, declared, R, PORT, picks, shares, EV, TR, msgs, T):
    ids = [sp["id"] for sp in SP]
    pay = {"family": "MASAM", "grid_declared": declared, "grid_unique": len(SP),
           "grid_total_runs": len(SP) * (len(PAIRS) + 1), "real_product_runs": len(SP) * len(REAL),
           "specs": [{k: v for k, v in sp.items()} for sp in SP], "checks": msgs, "pairs": {}, "picks": {}, "shares": {},
           "port7": {}, "events": {}, "trace_2026": TR, "windows": {}}
    for key in list(PAIRS) + [(p, "IXIC") for p in REAL]:
        k = f"{key[0]}·{key[1]}"
        pay["windows"][k] = R[key]["wins"]
        pay["pairs"][k] = {}
        for i, sp in enumerate(SP):
            rec = {cf: row_of(R, key, cf, i) for cf in R[key]["cfg"]}
            cyc = R[key]["cycles"][i]
            rec["휩소_긴역사"] = whipsaw(cyc, "1900-01-01", "2026-12-31")
            rec["휩소_고르기"] = whipsaw(cyc, "1900-01-01", PICK_END)
            rec["휩소_확인"] = whipsaw(cyc, *CHECK)
            rec["주기_2026"] = [(c[0], c[1], c[2], c[3]) for c in cyc if c[0] >= "2026-01-01" or (c[2] or "") >= "2026-01-01"]
            pay["pairs"][k][sp["id"]] = rec
        hold = R[key]["E_hold_frontier"]
        pay["pairs"][k]["보유"] = {wn: {kk: _f(v[-1]) for kk, v in st.items() if isinstance(v, np.ndarray)} for wn, st in hold.items()}
    for key, p in picks.items():
        k = f"{key[0]}·{key[1]}"
        pay["picks"][k] = {kk: (ids[v] if kk in ("GL", "G26") else v) for kk, v in p.items()}
    for key, s in shares.items():
        pay["shares"][f"{key[0]}·{key[1]}"] = s
    for bn in ("2026", "hold"):
        pay["port7"][bn] = {cf: {sid: {wn: {kk: _f(vv) for kk, vv in st.items()} for wn, st in m.items()}
                                  for sid, m in rows.items()} for cf, rows in PORT[bn]["rows"].items()}
        b = H.base(bn)
        pay["port7"][bn]["보유"] = {k: H.stats(b.ret, *v) for k, v in b.periods.items()}
    for tn, ev in EV.items():
        pay["events"][tn] = {"base": ev["base"], "era_base": ev["era_base"],
                             "cells": {k: v for k, v in ev["cells"].items() if k not in ("P1_rows",)},
                             "P1_rows": ev["cells"].get("P1_rows")}
    return pay


# ============================================================================ 보고서
def pct(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%p"


def p0(x):
    return "—" if x is None or x != x else f"{x * 100:.0f}%"


def build_md(SP, declared, R, PORT, picks, shares, EV, TR, msgs, T):
    ids = [sp["id"] for sp in SP]
    rep = picks[("NDX3", "IXIC")]["GL"]
    kN, kI = ("NDX3", "IXIC"), ("IXIC1", "IXIC")
    gl_I = picks[kI]["GL"]
    g26P = picks[("PORT7", "IXIC")]["G26"]
    L = []
    A = L.append

    def g(key, cf, wn, k, i):
        rows = R[key]["cfg"][cf]
        if wn not in rows:
            return float("nan")
        return float(rows[wn][k][i])

    def hold(key, wn, k):
        f = R[key]["E_hold_frontier"]
        return float(f[wn][k][-1]) if wn in f else float("nan")

    def rm(key, wn, i, long=False):
        if wn not in R[key]["cfg"]["base"]:
            return "—"
        st = R[key]["cfg"]["base"][wn]
        star = "*" if st["부분"] else ""
        if long:
            return f"연 {pct(st['연율'][i])}/{pct(st['낙폭'][i], 0)}{star}"
        return f"{pct(st['수익'][i], 0)}/{pct(st['낙폭'][i], 0)}{star}"

    def rh(key, wn, long=False):
        f = R[key]["E_hold_frontier"]
        if wn not in f:
            return "—"
        st = f[wn]
        star = "*" if R[key]["cfg"]["base"][wn]["부분"] else ""
        if long:
            return f"연 {pct(st['연율'][-1])}/{pct(st['낙폭'][-1], 0)}{star}"
        return f"{pct(st['수익'][-1], 0)}/{pct(st['낙폭'][-1], 0)}{star}"

    def pm(bn, sid, k, key="수익", cf="base"):
        return PORT[bn]["rows"][cf][sid][k][key]

    def bhp(bn, k):
        return H.stats(H.base(bn).ret, *H.base(bn).periods[k])

    def cyc_in(key, i, lo, hi):
        return [(c[0], c[2]) for c in R[key]["cycles"][i] if lo <= c[0] <= hi]

    P = T["IXIC1"]
    PN = T["NDX3"]
    ev = EV["IXIC1"]
    e1 = ev["cells"]["P1"]
    c00 = [c for c in R[kI]["cycles"][0] if "1999-12-01" <= c[0] <= "2000-06-30"][0]
    c08 = [c for c in R[kI]["cycles"][0] if "2008-08-01" <= c[0] <= "2008-12-31"][-1]
    c25 = cyc_in(kI, 0, "2025-01-01", "2025-12-31")
    real_ex = [g(("TQQQ", "IXIC"), "base", "실제전체", "효율선초과", i) for i in range(4)]
    n10 = [g(kN, "base", "2010~19", "효율선초과", i) for i in range(4)]
    nb26 = [g(kN, "base", "2026뒤", "효율선초과", i) for i in range(4)]
    sI, sN = shares[kI], shares[kN]
    sP = shares[("PORT7", "IXIC")]
    io = lambda P_, d: float(P_.o[P_.idx(d)])
    ic = lambda P_, d: float(P_.c[P_.idx(d)])
    back_I = ic(P, "2026-07-29") / io(P, "2026-07-09") - 1
    back_N = ic(PN, "2026-07-29") / io(PN, "2026-07-09") - 1
    back_N2 = ic(PN, "2026-07-29") / io(PN, "2026-07-06") - 1
    wipe = lambda key, i: g(key, "base", "긴역사", "낙폭", i) <= -0.90
    fmt0 = lambda x: "—" if not x else f"{x['가격']:,.0f}"
    md_ = lambda d: "—" if not d else f"{int(d[5:7])}/{int(d[8:10])}"

    A("# 7단계 MASAM: 마삼룰(나스닥 −3% → 팔고 한 달 뒤 되사기)을 실제 보유 대상에 걸면 (2026-10-11)")
    A("")
    A("> 사용자: \"마삼룰이라고 있거든? 폭락 이후에는 어느 정도 기간 뒤에 기계적으로 매수하는 방법을 연구해볼래요. 다양하게 조합해서.\"")
    A("> 원문 sondoctor.co.kr/664. 신호 = 나스닥 종합(^IXIC) 종가, 다음 거래일 시가 체결(옛 지수의 묵은 시가·시가 없는 날은 다음 날 종가). "
      "수수료 0, 현금 이자 0(4%·^IRX 판 따로). 스크립트 `code/p7_MASAM.py`, 숫자 `results/p7_MASAM.json`.")
    A(f"> 시험 범위: 대상·신호 쌍마다 같은 규칙 {len(SP)}개(선언 {declared}: 원형 4 + 격자 72 + 공황 6 + 8일 12, 공황 1개가 원형 P1과 같음) "
      f"× 쌍 {len(PAIRS) + 1}개 = {len(SP) * (len(PAIRS) + 1)}회, 실제 TQQQ·SOXL·KORU 대조 {len(SP) * len(REAL)}회. 격자는 실행 전에 고정했다.")
    A("")
    A("## 결론 3줄")
    A("")
    u2 = picks[kN]["G26"]
    sells26 = [x for x in TR["NDX3·IXIC"].get(ids[u2], []) if x["체결"] >= "2026-01-01" and x["노출"] < 1]
    A(f"1. **원형 마삼룰은 2026에 한 번 팔고 한 번 샀고, 결과는 같은 낙폭의 '덜 담기'와 비슷했다.** "
      f"6/5(−4.2%) → 6/8 시가 매도 → 7/9(거래일판)·7/6(달력판) 재매수. 7월 쇼크 앞 절반(6/22~7/8)만 피하고 뒤 절반(^IXIC 7/9 시가 → 7/29 {pct(back_I)})은 맞아 "
      f"3배 나스닥100 7월 쇼크 {pct(hold(kN, '2026_7월', '수익'))} → {pct(g(kN, 'base', '2026_7월', '수익', 0))}, "
      f"7종목 {pct(bhp('2026', 'AI쇼크')['수익'])} → {pct(pm('2026', 'P1', 'AI쇼크'))}였다(3배 2026 뒤 효율선초과 {pp(min(nb26))}~{pp(max(nb26))}). "
      f"1~5월엔 −3% 날이 없어 원형으로 고를 정보가 없었고, 2025년엔 공황 규칙으로 {md_(c25[1][0])}에 팔아 {md_(c25[1][1])}에야 되사 4~6월 반등을 놓쳤다"
      f"(7종목 {pct(bhp('hold', '2025')['수익'], 0)} → {pct(pm('hold', 'P1', '2025'), 0)}).")
    A(f"2. **긴 역사에선 마삼형이 거의 다 '덜 담기'를 이겼지만, 그 몫은 두 번의 긴 약세장에서 나왔다.** "
      f"2006~2025 효율선초과 > 0: 1배 나스닥 {sI['긴_무조건_확인>0']}/{len(SP)}, 3배 나스닥100 {sN['긴_무조건_확인>0']}/{len(SP)}. "
      f"원형은 {c00[0]} → {c00[2]}(닷컴 붕괴 통째로)와 {c08[0]} → {c08[2]} 구간을 현금으로 비켰다. "
      f"그런 약세장이 없던 2010~19년엔 3배 원형이 연 {pp(min(n10))}~{pp(max(n10))}, 실제 TQQQ 2010~2026은 연 {pp(min(real_ex))}~{pp(max(real_ex))}였다. "
      f"대기의 값은 싸게 되사는 데 있지 않았다: 원형 재매수 값은 판 값보다 평균 {pct(e1['되산가차'])} 비쌌고(싸게 되산 주기 {p0(e1['싸게되산'])}), "
      f"126일 수익은 바로 산 것과 같았다({pct(e1['마삼'])} vs {pct(e1['바로'])}). "
      f"대표 후보(3배, 긴 역사로 고른 {SP[rep]['id']})는 1985~2005 연 {pp(g(kN, 'base', '고르기_긴', '효율선초과', rep))} → 2006~2025 연 {pp(g(kN, 'base', '확인_긴', '효율선초과', rep))}, "
      f"전멸(보유 {pct(hold(kN, '긴역사', '낙폭'), 0)})을 {pct(g(kN, 'base', '긴역사', '낙폭', rep), 0)}로 막았지만 2026 앞 정보 없음·뒤 {pp(g(kN, 'base', '2026뒤', '효율선초과', rep))}·"
      f"2025 {pp(g(kN, 'base', '2025', '효율선초과', rep))}라 관찰이다.")
    A(f"3. **2026 앞으로 고를 수 있었던 건 '나스닥 −2% → 21거래일 대기' 계열뿐이었고 뒤에서도 버텼다. 다만 2026 매도가 {len(sells26)}번뿐이다.** "
      f"3배 나스닥100에서 고른 '{SP[u2]['label']}' 규칙은 앞 {pp(g(kN, 'base', '2026앞', '효율선초과', u2))} → 뒤 {pp(g(kN, 'base', '2026뒤', '효율선초과', u2))}·"
      f"2025 {pp(g(kN, 'base', '2025', '효율선초과', u2))}(−2% 날 6/5·6/23·7/23이 이어져 7월 쇼크 내내 현금, 미끄러짐 0.5% 뒤 {pp(g(kN, 'slip', '2026뒤', '효율선초과', u2))}·"
      f"하루 늦게 뒤 {pp(g(kN, 'lag1', '2026뒤', '효율선초과', u2))}), 7종목에서 고른 '{SP[g26P]['label']}' 규칙은 앞 {pp(pm('2026', ids[g26P], '앞', '효율선초과'))} → "
      f"뒤 {pp(pm('2026', ids[g26P], '뒤', '효율선초과'))}·2025 {pp(pm('hold', ids[g26P], '2025', '효율선초과'))}였다. "
      f"그러나 같은 3배 규칙이 2010~19엔 연 {pp(g(kN, 'base', '2010~19', '효율선초과', u2))}, 실제 TQQQ 2010~2026 연 {pp(g(('TQQQ', 'IXIC'), 'base', '실제전체', '효율선초과', u2))}, "
      f"7종목 규칙은 2024 {pp(pm('hold', ids[g26P], '2024', '효율선초과'))}라 장세 전환 위험이 커 관찰만 한다. "
      f"SOXL·KORU는 나스닥 신호든 자기 지수 신호든 2006~2025 효율선 아래(> 0인 규칙 SOXL {shares[('SOXL3', 'IXIC')]['긴_무조건_확인>0']}·{shares[('SOXL3', 'SOX')]['긴_무조건_확인>0']}개, "
      f"KORU {shares[('KORU3', 'IXIC')]['긴_무조건_확인>0']}·{shares[('KORU3', 'EWY')]['긴_무조건_확인>0']}개/{len(SP)})였고 전멸도 못 막아 기각.")
    A("")

    # ---- 사용자 표
    A("## 사용자 표 (수익/최대 낙폭, 미끄러짐 0, 현금 이자 0)")
    A("")
    A("짧은 창은 기간 수익, '긴 역사'는 대상 시작~2026-10-09 연율. * = 대상이 창 중간에 시작. 7월 쇼크 = 6/22 종가 → 7/29 종가. "
      "2000~02 = 2000-03-01~2002-10-31, 2008·2020·2022·2025·1987은 달력 연도.")
    A("")
    cols = ["2026앞", "2026뒤", "2026_7월", "2025", "2022", "2020", "2008", "2000~02", "1987"]
    A("| 대상·신호 | 규칙 | 2026 앞 | 2026 뒤 | 7월 쇼크 | 2025 | 2022 | 2020 | 2008 | 2000~02 | 1987 | 긴 역사 | 전멸(≤−90%) | 연 매매 | 현금 기간 | 휩소 |")
    A("|---|---|" + "---|" * 14)
    for key in PAIRS:
        tn, sn = key
        A(f"| {TNAME[tn]} · {SNAME[sn]} 신호 | 그냥 보유 | " + " | ".join(rh(key, w) for w in cols) + f" | {rh(key, '긴역사', True)} | "
          f"{'예' if hold(key, '긴역사', '낙폭') <= -0.9 else '아니오'} | 0.0 | 0% | — |")
        gl = picks[key]["GL"]
        for i in sorted({0, 1, 2, 3, gl}):
            nm2 = SP[i]["id"] if SP[i]["fam"] == "원형" else SP[i]["label"]
            if i == gl:
                nm2 = f"GL: {nm2}"
            wh = whipsaw(R[key]["cycles"][i], "1900-01-01", "2026-12-31")
            A(f"| | {nm2} | " + " | ".join(rm(key, w, i) for w in cols) + f" | {rm(key, '긴역사', i, True)} | "
              f"{'예' if wipe(key, i) else '아니오'} | {g(key, 'base', '긴역사', '연매매', i):.1f} | "
              f"{p0(g(key, 'base', '긴역사', '현금기간', i))} | {p0(wh['휩소율'])} ({wh['주기']}번) |")
    A("")
    A("원형: P1 21거래일·공황(21일 안 4번 → 42거래일), P2 달력 1개월+1일·공황(1개월 안 4번 → 2개월), P3·P4 = P1·P2 + 8거래일 연속 상승이면 즉시 재매수. "
      "휩소 = 다시 산 가격이 판 가격보다 비쌌던 주기 비율(괄호 = 주기 수). GL = 긴 역사 전방 검증(시작~2005 효율선초과 최대)으로 고른 변형. "
      "보조 신호(^NDX·^SOX·EWY)는 그 지수의 하루 −X%를 쓴 판.")
    A("")

    # ---- 7종목
    A("### 7종목(데스크 규칙, 시가→시가, ^IXIC 신호)")
    A("")
    pk = [("2026", "앞"), ("2026", "뒤"), ("2026", "AI쇼크"), ("2026", "7~9월"), ("hold", "2025"), ("hold", "2024"), ("hold", "관세")]
    A("| 규칙 | 2026 앞 | 2026 뒤 | 7월 쇼크(6/22~7/29) | 7~9월 | 2025 | 2024 | 25년 관세(2/18~4/4) | 효율선초과 앞·뒤·2025·2024 | 하루 늦게 앞·뒤·2025 | 현금 기간 2026·2025 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    A("| 그냥 보유 | " + " | ".join(f"{pct(bhp(bn, k)['수익'], 0)}/{pct(bhp(bn, k)['낙폭'], 0)}" for bn, k in pk) + " | 0 | 0 | 0% · 0% |")
    for i in sorted({0, 1, 2, 3, gl_I, rep, g26P}):
        sid = ids[i]
        tag = sid if SP[i]["fam"] == "원형" else SP[i]["label"]
        if i == gl_I:
            tag += " (1배 GL)"
        if i == rep:
            tag += " (3배 GL = 대표)"
        if i == g26P:
            tag += " (7종목 G26)"
        A(f"| {tag} | " + " | ".join(f"{pct(pm(bn, sid, k), 0)}/{pct(pm(bn, sid, k, '낙폭'), 0)}" for bn, k in pk) +
          f" | {pp(pm('2026', sid, '앞', '효율선초과'))} · {pp(pm('2026', sid, '뒤', '효율선초과'))} · {pp(pm('hold', sid, '2025', '효율선초과'))} · {pp(pm('hold', sid, '2024', '효율선초과'))} | "
          f"{pp(pm('2026', sid, '앞', '효율선초과', 'lag1'))} · {pp(pm('2026', sid, '뒤', '효율선초과', 'lag1'))} · {pp(pm('hold', sid, '2025', '효율선초과', 'lag1'))} | "
          f"{p0(pm('2026', sid, '전체', '현금기간'))} · {p0(pm('hold', sid, '2025', '현금기간'))} |")
    A("")
    A("7종목 이웃(나스닥 −2%·−2.5%, 대기 거래일별) 효율선초과 앞 / 뒤 / 2025 (%p, 굵게 = 7종목 G26):")
    A("")
    A("| 규칙 | " + " | ".join(f"{n}일" for n in N_GRID) + " | 21일+8일 |")
    A("|---|" + "---|" * (len(N_GRID) + 1))
    for X in (-0.02, -0.025):
        for s_ in S_GRID:
            cells = []
            for sid in [f"G{-X * 100:g}_{n}_{int(s_ * 100)}" for n in N_GRID] + [f"U{-X * 100:g}_{int(s_ * 100)}"]:
                mk = "**" if sid == ids[g26P] else ""
                cells.append(f"{mk}{pm('2026', sid, '앞', '효율선초과') * 100:+.1f} / {pm('2026', sid, '뒤', '효율선초과') * 100:+.1f} / "
                             f"{pm('hold', sid, '2025', '효율선초과') * 100:+.1f}{mk}")
            A(f"| {X * 100:g}%, {s_:.0%} 매도 | " + " | ".join(cells) + " |")
    A("")

    # ---- 효율선 표
    A("## 효율선초과 (같은 최대 낙폭의 '덜 담기'보다 더 번 몫)")
    A("")
    A("짧은 창은 기간 %p, 긴 창(고르기·확인·2010~19·긴 역사)은 연율 %p. 7월 쇼크는 하락만 있는 창이라 효율선 대신 '보유 대비' %p. "
      "미끄러짐 0.5%·현금 4%·^IRX·하루 늦게는 긴 역사 연율 %p(효율선도 같은 조건).")
    A("")
    A("| 대상·신호 | 규칙 | 2026 앞 | 2026 뒤 | 7월 쇼크(보유 대비) | 2025 | 고르기(시작~2005) | 확인(2006~2025) | 2010~19 | 긴 역사 | 같은 노출 대비 | 미끄러짐 0.5% | 현금 4% | 현금 ^IRX | 하루 늦게 |")
    A("|---|---|" + "---|" * 13)
    for key in PAIRS:
        gl = picks[key]["GL"]
        for i in sorted({0, 1, 2, 3, gl}):
            tag = SP[i]["id"] + (" GL" if i == gl else "")
            A(f"| {TNAME[key[0]]} · {SNAME[key[1]]} | {tag} | {pp(g(key, 'base', '2026앞', '효율선초과', i))} | {pp(g(key, 'base', '2026뒤', '효율선초과', i))} | "
              f"{pp(g(key, 'base', '2026_7월', '수익', i) - hold(key, '2026_7월', '수익'))} | " +
              " | ".join(pp(g(key, "base", w, "효율선초과", i)) for w in ("2025", "고르기_긴", "확인_긴", "2010~19", "긴역사")) +
              f" | {pp(g(key, 'base', '긴역사', '같은노출초과', i))} | {pp(g(key, 'slip', '긴역사', '효율선초과', i))} | "
              f"{pp(g(key, 'cash4', '긴역사', '효율선초과', i))} | {pp(g(key, 'irx', '긴역사', '효율선초과', i))} | {pp(g(key, 'lag1', '긴역사', '효율선초과', i))} |")
    A("")
    A("GL 규칙: " + "; ".join(f"{TNAME[k[0]]}·{SNAME[k[1]]} = {SP[picks[k]['GL']]['label']}" for k in PAIRS) + ".")
    A("")

    # ---- 조합 격자 표
    A("## 조합 격자 표 (행 = 사건 정의 '나스닥 하루 하락', 열 = 대기 거래일)")
    A("")
    for key, ttl in ((kN, "3배 나스닥100 합성 · ^IXIC 신호"), (kI, "나스닥 종합 1배 · ^IXIC 신호")):
        for s_ in (1.0, 0.5):
            A(f"**{ttl}, {s_:.0%} 매도** — 칸 = 고르기(시작~2005) / 확인(2006~2025) / 2010~19 효율선초과 연 %p (굵게 = GL)")
            A("")
            A("| 하루 하락 | " + " | ".join(f"{n}일" for n in N_GRID) + " | 21일+8일 |")
            A("|---|" + "---|" * (len(N_GRID) + 1))
            for X in X_GRID:
                cells = []
                for sid in [f"G{-X * 100:g}_{n}_{int(s_ * 100)}" for n in N_GRID] + [f"U{-X * 100:g}_{int(s_ * 100)}"]:
                    i = ids.index(sid)
                    mk = "**" if i == picks[key]["GL"] else ""
                    cells.append(f"{mk}{g(key, 'base', '고르기_긴', '효율선초과', i) * 100:+.1f} / {g(key, 'base', '확인_긴', '효율선초과', i) * 100:+.1f} / "
                                 f"{g(key, 'base', '2010~19', '효율선초과', i) * 100:+.1f}{mk}")
                A(f"| {X * 100:g}% | " + " | ".join(cells) + " |")
            A("")
    A("공황 변형(−3%, 21거래일, 100% 매도) 고르기 / 확인 / 2010~19 효율선초과 연 %p (4번/21일→42일 = 원형 P1):")
    A("")
    A("| 대상 | " + " | ".join(f"{k}번/21일→{w}일" for k in PANIC_K for w in PANIC_W) + " |")
    A("|---|" + "---|" * 6)
    for key in (kN, kI):
        cells = []
        for k in PANIC_K:
            for w in PANIC_W:
                i = ids.index(f"K{k}W{w}") if f"K{k}W{w}" in ids else 0
                cells.append(f"{g(key, 'base', '고르기_긴', '효율선초과', i) * 100:+.1f} / {g(key, 'base', '확인_긴', '효율선초과', i) * 100:+.1f} / "
                             f"{g(key, 'base', '2010~19', '효율선초과', i) * 100:+.1f}")
        A(f"| {TNAME[key[0]]} | " + " | ".join(cells) + " |")
    A("")

    # ---- 고르기 표
    A("## 고르기 표")
    A("")
    A("| 대상·신호 | 판정 | 고른 규칙 | 고른 구간 효율선초과 | 확인 효율선초과 | 2026 앞 | 2026 뒤 | 7월 쇼크 | 2025 | '고른 구간 > 0' 중 확인 > 0 (무조건) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for key in PAIRS:
        p, s = picks[key], shares[key]
        i = p["GL"]
        A(f"| {TNAME[key[0]]} · {SNAME[key[1]]} | 긴 역사(시작~2005 → 2006~2025) | {SP[i]['label']} | 연 {pp(g(key, 'base', '고르기_긴', '효율선초과', i))} | "
          f"연 {pp(g(key, 'base', '확인_긴', '효율선초과', i))} | {rm(key, '2026앞', i)} ({pp(g(key, 'base', '2026앞', '효율선초과', i))}) | "
          f"{rm(key, '2026뒤', i)} ({pp(g(key, 'base', '2026뒤', '효율선초과', i))}) | {pct(g(key, 'base', '2026_7월', '수익', i))} | "
          f"{rm(key, '2025', i)} ({pp(g(key, 'base', '2025', '효율선초과', i))}) | {s['긴_앞>0중_확인>0']}/{s['긴_앞>0']} ({s['긴_무조건_확인>0']}/{len(SP)}) |")
        i = p["G26"]
        if p["G26_max"] > 1e-9:
            A(f"| | 2026(앞 → 뒤·2025) | {SP[i]['label']} | {pp(p['G26_max'])} | 뒤 {pp(g(key, 'base', '2026뒤', '효율선초과', i))}, 2025 {pp(g(key, 'base', '2025', '효율선초과', i))} | "
              f"{rm(key, '2026앞', i)} | {rm(key, '2026뒤', i)} | {pct(g(key, 'base', '2026_7월', '수익', i))} | {rm(key, '2025', i)} | "
              f"뒤 {s['26_앞>0중_뒤>0']}/{s['26_앞>0']} ({s['26_무조건_뒤>0']}/{len(SP)}), 2025 {s['26_앞>0중_2025>0']}/{s['26_앞>0']} ({s['26_무조건_2025>0']}/{len(SP)}) |")
        else:
            A("| | 2026(앞 → 뒤·2025) | 고를 수 없음(앞에서 효율선을 이긴 규칙 0개) | — | — | — | — | — | — | — |")
    p, s = picks[("PORT7", "IXIC")], sP
    i = p["G26"]
    sid = ids[i]
    A(f"| 7종목 · ^IXIC | 2026(앞 → 뒤·2025) | {SP[i]['label']} | {pp(p['G26_max'])} | 뒤 {pp(pm('2026', sid, '뒤', '효율선초과'))}, 2025 {pp(pm('hold', sid, '2025', '효율선초과'))} | "
      f"{pct(pm('2026', sid, '앞'), 0)}/{pct(pm('2026', sid, '앞', '낙폭'), 0)} | {pct(pm('2026', sid, '뒤'), 0)}/{pct(pm('2026', sid, '뒤', '낙폭'), 0)} | "
      f"{pct(pm('2026', sid, 'AI쇼크'))} | {pct(pm('hold', sid, '2025'), 0)}/{pct(pm('hold', sid, '2025', '낙폭'), 0)} | "
      f"뒤 {s['26_앞>0중_뒤>0']}/{s['26_앞>0']} ({s['26_무조건_뒤>0']}/{len(SP)}), 2025 {s['26_앞>0중_2025>0']}/{s['26_앞>0']} ({s['26_무조건_2025>0']}/{len(SP)}) |")
    A("")
    A("- ^IXIC 신호로 2026 앞(1/2~5/29)에 거래한 규칙은 −2% 날(1/20·2/12·3/20·3/26·3/27)을 쓰는 14개뿐이다. 원형(−3%)은 앞에서 그냥 보유와 같다.")
    A("- 3배 나스닥100에서 '고른 구간 > 0' 집합의 확인 성공률은 무조건(전 규칙)과 비슷하다 → 긴 역사에서는 '어떤 변형이냐'보다 '마삼형이냐'가 결과를 갈랐다.")
    A("")

    # ---- 2026 날짜
    A("## 2026년에 마삼룰이 실제로 한 일 (^IXIC 신호)")
    A("")
    A("| 규칙 | 판 날(신호 → 체결) | 되산 날(판단 → 체결) | ^IXIC 판 값 → 되산 값 | 3배 나스닥100 되산 값/판 값 | 7종목 7월 쇼크(보유 " + pct(bhp('2026', 'AI쇼크')['수익']) + ") |")
    A("|---|---|---|---|---|---|")
    for i in sorted({0, 1, 2, 3, rep}):
        tr = TR["IXIC1·IXIC"][ids[i]]
        sells = [x for x in tr if x["노출"] < 1 and x["체결"] >= "2026-01-01"]
        buys = [x for x in tr if x["노출"] >= 1 and x["체결"] >= "2026-01-01"]
        if not sells:
            continue
        sv, bu = sells[0], (buys[0] if buys else None)
        trN = TR["NDX3·IXIC"][ids[i]]
        sN_ = [x for x in trN if x["노출"] < 1 and x["체결"] >= "2026-01-01"][0]
        bN_ = [x for x in trN if x["노출"] >= 1 and x["체결"] >= "2026-01-01"]
        A(f"| {ids[i]}{' (대표)' if i == rep else ''} | {md_(sv['판단일'])} {pct(sv['판단일_신호수익'])} → {md_(sv['체결'])} 시가 | "
          f"{(md_(bu['판단일']) + ' → ' + md_(bu['체결']) + ' 시가') if bu else '아직'} | "
          f"{sv['가격']:,.0f} → {fmt0(bu)} ({pct(bu['가격'] / sv['가격'] - 1) if bu else '—'}) | "
          f"{pct(bN_[0]['가격'] / sN_['가격'] - 1) if bN_ else '—'} | {pct(pm('2026', ids[i], 'AI쇼크'))} |")
    A("")
    A("- 거래일판: 6/5 다음 21거래일(6/19·7/3 휴장 제외) 동안 −3% 날이 없어 7/8 종가에 판단 → 7/9 시가. 달력판: 6/5 + 1개월 + 1일 = 7/6(월) 시가. "
      "그 사이 8거래일 연속 상승은 없어 수정판(P3·P4)도 같다.")
    A(f"- 7월 쇼크(6/22~7/29)와의 관계: 앞 절반(6/22~7/8)은 현금이라 피했고 뒤 절반에는 다시 들어갔다. ^IXIC 7/9 시가 → 7/29 종가 {pct(back_I)}, "
      f"3배 나스닥100 {pct(back_N)}(달력판 7/6 → 7/29 {pct(back_N2)}). 7/23(−2.2%)은 −3%에 못 미쳐 다시 팔지 않았다.")
    a26 = TR["NDX3·IXIC"].get(ids[picks[kN]["G26"]], [])
    if a26:
        A(f"- 참고(2026 앞으로 고른 '{SP[picks[kN]['G26']]['label']}', 3배): " +
          ", ".join(f"{md_(x['체결'])} {'매도' if x['노출'] < 1 else '재매수'}" for x in a26 if x["체결"] >= "2026-01-01") +
          ". −2% 날(6/5·6/23·7/23)이 21거래일 안에 이어져 7월 쇼크 내내 현금이었다.")
    A("- 2025년(원형 P1): " + ", ".join(f"{md_(a)} 매도 → {md_(b)} 재매수" for a, b in c25) +
      ". 3/10~4/16 −3% 5번이 공황(21거래일 안 4번)을 켜 마지막 날(4/16)부터 42거래일을 기다렸다.")
    A("")

    # ---- 사건형
    A("## 사건형: 폭락 날 바로 사기 vs 마삼 대기 뒤 사기 vs 아무 날 (나스닥 종합 1배, 1971~2025, 126거래일 보유)")
    A("")
    A(f"아무 날 매수 기저율: 126일 평균 {pct(ev['base']['평균'])}, 승률 {p0(ev['base']['승률'])}, 그 안에 진입가 대비 −20% 이하 {p0(ev['base']['dd20'])}.")
    A("")
    A("칸 = 되산 가격 − 판 가격 평균(−면 싸게 되삼) / 싸게 되산 비율 / 주기 수. 100% 매도, 공황·8일 없음.")
    A("")
    A("| 하루 하락 | " + " | ".join(f"{n}일" for n in N_GRID) + " |")
    A("|---|" + "---|" * len(N_GRID))
    for X in X_GRID:
        cells = []
        for n in N_GRID:
            c = ev["cells"].get(f"G{-X * 100:g}_{n}_100", {"n": 0})
            cells.append("—" if not c["n"] else f"{pct(c['되산가차'])} / {p0(c['싸게되산'])} / {c['n']}")
        A(f"| {X * 100:g}% | " + " | ".join(cells) + " |")
    A("")
    A("원형 P1(−3%, 21거래일, 공황 42) 시대별: 마삼 재매수 vs 첫 −3% 다음 날 바로 매수(126거래일 수익 평균, −20% 이하 비율)")
    A("")
    A("| 시대 | 주기 | 되산−판 | 마삼 재매수 126일 | 바로 매수 126일 | 아무 날 126일 | 마삼 −20%↓ | 바로 −20%↓ |")
    A("|---|---|---|---|---|---|---|---|")
    for en, lo, hi in ERAS:
        c = ev["cells"]["P1_era"][en]
        b_ = ev["era_base"][en]
        if not c["n"]:
            A(f"| {en} | 0 | — | — | — | {pct(b_['평균']) if en != '2026' else '— (126일 미경과)'} | — | — |")
            continue
        A(f"| {en} | {c['n']} | {pct(c['되산가차'])} | {pct(c['마삼'])} (승 {p0(c['마삼_승률'])}) | {pct(c['바로'])} (승 {p0(c['바로_승률'])}) | "
          f"{pct(b_['평균'])} | {p0(c['마삼_dd20'])} | {p0(c['바로_dd20'])} |")
    A(f"| 전체 | {e1['n']} | {pct(e1['되산가차'])} | {pct(e1['마삼'])} (승 {p0(e1['마삼_승률'])}) | {pct(e1['바로'])} (승 {p0(e1['바로_승률'])}) | "
      f"{pct(ev['base']['평균'])} | {p0(e1['마삼_dd20'])} | {p0(e1['바로_dd20'])} |")
    e3, b3 = EV["NDX3"]["cells"]["P1"], EV["NDX3"]["base"]
    A("")
    A(f"- 3배 나스닥100 합성(1985~2025, 주기 {e3['n']}): 마삼 재매수 126일 {pct(e3['마삼'])}(−20%↓ {p0(e3['마삼_dd20'])}), "
      f"바로 매수 {pct(e3['바로'])}(−20%↓ {p0(e3['바로_dd20'])}), 아무 날 {pct(b3['평균'])}(−20%↓ {p0(b3['dd20'])}). 되산−판 {pct(e3['되산가차'])}.")
    A(f"- 2000~02 주기 하나({c00[0]} 매도 → {c00[2]} 재매수, 되산−판 {pct(c00[3] / c00[1] - 1)})가 긴 역사 이익의 큰 몫이다. "
      "나머지 주기는 평균적으로 판 값보다 비싸게 되샀다.")
    A("")

    # ---- 실제 상품
    A("## 실제 상품 대조 (^IXIC 신호, 상장~2026-10-09)")
    A("")
    A("| 상품 | 규칙 | 2026 앞 | 2026 뒤 | 7월 쇼크 | 2025 | 2022 | 2020 | 상장 뒤 연율/낙폭 | 상장 뒤 효율선초과(연) | 미끄러짐 0.5% |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for p_ in REAL:
        key = (p_, "IXIC")
        A(f"| {p_} | 그냥 보유 | " + " | ".join(rh(key, w) for w in ("2026앞", "2026뒤", "2026_7월", "2025", "2022", "2020")) +
          f" | {rh(key, '실제전체', True)} | 0 | — |")
        for i in sorted({0, 1, 2, 3, rep}):
            A(f"| | {ids[i]}{' (대표)' if i == rep else ''} | " + " | ".join(rm(key, w, i) for w in ("2026앞", "2026뒤", "2026_7월", "2025", "2022", "2020")) +
              f" | {rm(key, '실제전체', i, True)} | {pp(g(key, 'base', '실제전체', '효율선초과', i))} | {pp(g(key, 'slip', '실제전체', '효율선초과', i))} |")
    A("")

    # ---- 한계
    A("## 한계")
    A("")
    A(f"- 시험 범위: 규칙 {len(SP)}개 × 대상·신호 {len(PAIRS) + 1}쌍(+ 실제 상품 3). 최고값에는 선택편향이 있어 "
      "'고른 구간 > 0 집합의 확인 성공률 vs 무조건'을 함께 적었다.")
    A("- 긴 역사 이익은 2000~02·2008 두 번에 몰려 있다. 2010~19·2022·2025·실제 TQQQ 2010~2026에서는 원형이 덜 담기보다 나빴다. "
      "앞으로 그런 긴 약세장이 오느냐에 달린 보험이지 2026 장세에서 검증된 규칙이 아니다(합격 기준은 2026).")
    A("- 신호 시점: t일 종가 → t+1 시가. 옛 지수(^IXIC 1971~84, ^NDX 1985~99, ^SOX 1994~2001)는 시가가 없거나 전날 종가와 같아(묵은 시가) "
      "그날 종가 체결로 바꿨다. 첫 실행에서 묵은 시가를 그대로 써 1987년 3배 P1이 +80%로 부풀었던 자료 오류를 고친 것이다(고친 뒤 −1%).")
    A("- 3배 합성: 보수 0.95% + 추가 비용(TQQQ 1.3%·SOXL 1.5%·KORU 2.6%)으로 실제와 맞췄다(남은 차 연 −0.2~+0.1%). "
      "NDX3 기초는 1999-03 전 ^NDX 가격 지수라 배당(연 0.5~1% × 3배)이 빠져 그 구간을 조금 낮게 잡는다(처음 ^NDX만 썼을 때 실제 TQQQ보다 연 2.8% 낮아 QQQ를 이었다).")
    A(f"- 현금 이자 0%가 기본이다. 마삼형은 현금 기간이 20~40%다. 덜 담기에도 같은 이자를 주면 효율선초과는 조금 줄고 부호는 같았다"
      f"(3배 원형 P1 긴 역사 연 {pp(g(kN, 'base', '긴역사', '효율선초과', 0))} → 4% {pp(g(kN, 'cash4', '긴역사', '효율선초과', 0))}·"
      f"^IRX {pp(g(kN, 'irx', '긴역사', '효율선초과', 0))}). 이자 없이 덜 담기 쪽만 현금이 노는 비교가 아니다.")
    A("- 달력판 '한 달'은 월 더하기(1/31 + 1개월 = 2/28) + 1일 뒤 첫 거래일. 공황은 그 현금 기간 내내 유지(원문 '마지막 −3% 날로부터 두 달' 해석). "
      "말뚝박기(낙폭 단계 매수)는 이 갈래에서 다루지 않았다.")
    A("- 7종목은 2023-05 이후 자료뿐이라 긴 역사 판정이 없다. 2026 앞으로 고른 7종목 규칙은 앞 > 0 규칙 3개가 모두 −2%·50% 매도·15~21일(사실상 한 규칙)이고 "
      "대기 10·30일 이웃은 앞에서 졌으며 2024에선 효율선 아래였다.")
    A("- '−2% → 21거래일' 계열의 2026 성과는 매도 세 번(1/21·3/23·6/8)에 기댄다. −2% 날이 몰리는 고변동 장세(2025~26)에선 이겼고, "
      "−2% 날이 흩어진 2010년대에는 들락거림으로 크게 졌다(3배 연 −24%p). 같은 격자 안에서 '−2%·절반 매도·15~21일'이 1배·3배·실제 TQQQ·7종목에 두루 "
      "괜찮아 보이는 것은 표를 본 뒤의 관찰이라 선택편향이 있다.")
    A("")
    A("## 점검")
    A("")
    for m in msgs:
        A(f"- {m}")
    A("")
    return "\n".join(L)

if __name__ == "__main__":
    main()
