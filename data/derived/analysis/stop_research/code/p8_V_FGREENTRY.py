"""8단계 반박 검증 p8_V_FGREENTRY(2026-10-12): 갈래 FGREENTRY 대표 후보를 독립 재계산으로 깨 보려 한다.

후보: SOXL 추적 −25% 손절(산 뒤 최고 종가 대비, 장중) + 'R5 먼저' 재매수(손절일 + 20거래일 또는 직전 종가 F&G ≥ 45 중 먼저, 다음 시가).
기준 B: 같은 손절 + 손절일 + 20거래일 시가 재매수.

독립성: p8_FGREENTRY.py·trade_lab·stock_lab·harness·lev5·lev_lab을 import하지 않는다. 읽는 것은
  cache/p5_lev_long.parquet(^SOX·SOXX·SOXL·^IRX), cache/p7_idx.parquet(^GSPC, 시점 감사용), fg_data.fg()(F&G 자료 읽기),
  cache/p8_FGREENTRY_recon.parquet·cache/p8_RECON_index.parquet(재구성 F&G 값만, 1994~2010 구간용)뿐이다.
3배 합성·이어 붙이기·손절·재매수 루프는 이 파일에서 새로 짰다(재매수 조건은 '트리거 날 배열'로 넣어 위약·무작위 날짜를 같은 엔진에 넣는다).

사전 고정(확인 구간을 본 뒤 바꾸지 않는다)
  합성: ^SOX(2001-07-13까지) → SOXX(2001-07-16부터) 일수익, 3배 일일 재설정, 금융비용 2 × ^IRX/252, 보수 0.95% + 추가 1.5%(연, 일할),
        2010-03-11부터 실제 SOXL(배당 조정 계수 적용). 시가·고가·저가는 기초의 전일 종가 대비 비율 × 3.
  시점: F&G는 그날 장 마감 뒤 확정 → t일 시가 판단에는 날짜 < t인 최신 값(FGP). 장중 손절선은 t−1까지 최고 종가.
        같은 날 순서 = 시가 재매수 → 손절 점검. 1994~2010 재구성 구간과 비교용 2011~2025는 종가 모드(t 종가 판단 → t+1 종가 체결).
  이웃 격자(각 파라미터 앞뒤 2칸): 손절 폭 15·20·25·30·35%, 시간 N 10·15·20·25·30거래일, F&G 문턱 X 35·40·45·50·55 = 125조합.
        비교는 두 가지: (1) B(같은 손절 폭, 20일) 대비, (2) 같은 손절 폭·같은 N의 '시간만' 규칙 대비(= F&G가 더한 몫).
  위약: (a) F&G 값을 원형으로 민 20판(이동량 63 + round(i × (n − 126)/19), i = 0..19, n = 2011~2026 F&G 일수; 재구성 앞부분은 그대로)
        (b) 무작위 날짜 20판(해마다 실제 F&G ≥ 45 날 수와 같은 수의 날을 그해 거래일에서 무작위로 골라 트리거로 쓴다, 시드 0..19).
  시대: 2011, 2015~16, 2018 4분기, 2020, 2022, 2025, 2026 앞·뒤·7월 쇼크(2026 연속 경로를 자름), 재구성 2000~02·2008(종가 모드).
  시작일 민감도: 2026 경로 시작을 2025-10-01 ~ 2026-01-30 거래일마다 바꿔 2026 앞(→ 5/29)과 뒤(6/1 → 10/9) 우위를 다시 잰다.
  비용: 수수료 0, 미끄러짐 0.5%(체결마다 불리), 현금 이자 0(4% 판 한 줄).

실행: cd code && PYTHONIOENCODING=utf-8 py p8_V_FGREENTRY.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fg_data  # noqa: E402  (자료 읽기만)

CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p8_V_FGREENTRY"

# ---------------------------------------------------------------- 사전 고정 상수
FEE, EXTRA, LEV = 0.0095, 0.015, 3.0
SPLICE_AT = "2001-07-16"
X0, N0, T0 = 0.25, 20, 45.0
MAXWAIT = 126
NB_X = [0.15, 0.20, 0.25, 0.30, 0.35]
NB_N = [10, 15, 20, 25, 30]
NB_T = [35.0, 40.0, 45.0, 50.0, 55.0]
PLACEBO_N, PLACEBO_MIN = 20, 63
SLIP, CASH = 0.005, 0.04
FRONT, BACK, SHOCK = ("2026-01-02", "2026-05-29"), ("2026-06-01", "2026-10-09"), ("2026-06-22", "2026-07-29")
D26 = ("2026-01-02", "2026-10-09")
RUNS = {"2025": ("2025-01-02", "2025-12-31", 0, False), "LP": ("2011-01-03", "2018-12-31", 0, True),
        "LC": ("2019-01-02", "2025-12-31", 0, True), "RP": ("1991-01-02", "2010-12-31", 1, True),
        "RC": ("2011-01-03", "2025-12-31", 1, True)}
ERAS = {"2011": ("2011-01-03", "2011-12-30", 0), "2015~16": ("2015-01-02", "2016-12-30", 0),
        "2018 4분기": ("2018-10-01", "2018-12-31", 0), "2020": ("2020-01-02", "2020-12-31", 0),
        "2022": ("2022-01-03", "2022-12-30", 0), "2025": ("2025-01-02", "2025-12-31", 0),
        "재구성 2000~02": ("2000-01-03", "2002-12-31", 1), "재구성 2008": ("2008-01-02", "2008-12-31", 1)}
START_SENS = ("2025-10-01", "2026-01-30")


# ---------------------------------------------------------------- 자료
def _raw() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "p5_lev_long.parquet")


def ohlc(raw: pd.DataFrame, sym: str) -> pd.DataFrame:
    g = raw[raw.sym == sym].sort_values("date").dropna(subset=["Close"])
    g = g[g.Close > 0].drop_duplicates("date", keep="last").set_index("date")
    k = (g["Adj Close"] / g["Close"]).fillna(1.0)
    out = pd.DataFrame({c: g[c].fillna(g["Close"]) * k for c in ("Open", "High", "Low")})
    out["Close"] = g["Close"] * k
    return out


def soxl_composite(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """^SOX→SOXX 기초의 3배 합성(실제 SOXL 전) + 실제 SOXL. 기초 비율(시가·고가·저가·종가 / 전일 종가)로 만든다."""
    sox, soxx = ohlc(raw, "^SOX"), ohlc(raw, "SOXX")

    def ratios(df):
        pc = df["Close"].shift(1)
        return pd.DataFrame({c: df[c] / pc for c in ("Open", "High", "Low", "Close")})

    ra, rb = ratios(sox), ratios(soxx)
    und = pd.concat([ra[ra.index < SPLICE_AT], rb[rb.index >= SPLICE_AT]]).iloc[1:]   # 첫 줄(전일 없음) 제외
    irx = raw[raw.sym == "^IRX"].drop_duplicates("date").set_index("date")["Close"].sort_index() / 100.0
    rf = irx.reindex(sorted(set(irx.index) | set(und.index))).ffill().reindex(und.index).fillna(0.0).to_numpy()
    n = len(und)
    o = np.empty(n + 1); h = np.empty(n + 1); lo = np.empty(n + 1); c = np.empty(n + 1)
    o[0] = h[0] = lo[0] = c[0] = 100.0
    R = und.to_numpy()
    for t in range(n):
        ro, rh, rl, rc = R[t]
        day = LEV * (rc - 1) - (LEV - 1) * rf[t] / 252 - (FEE + EXTRA) / 252
        c[t + 1] = c[t] * max(1 + day, 0.001)
        o[t + 1] = c[t] * max(1 + LEV * (ro - 1), 0.001)
        hh = c[t] * max(1 + LEV * (rh - 1), 0.001)
        ll = c[t] * max(1 + LEV * (rl - 1), 0.001)
        h[t + 1] = max(hh, o[t + 1], c[t + 1])
        lo[t + 1] = min(ll, o[t + 1], c[t + 1])
    days = [sox.index[0]] + list(und.index)
    syn = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c}, index=days)
    act = ohlc(raw, "SOXL")
    a0 = act.index[0]
    sc = act["Close"].iloc[0] / syn.loc[a0, "Close"]
    comp = pd.concat([syn[syn.index < a0] * sc, act])
    # 합성 점검: 실제와 겹치는 기간 일수익 상관·연 로그 비용 차
    common = syn.index.intersection(act.index)
    rs, ra_ = syn.loc[common, "Close"].pct_change().dropna(), act.loc[common, "Close"].pct_change().dropna()
    yrs = len(common) / 252
    chk = {"합성_실제_일수익상관": float(np.corrcoef(rs, ra_)[0, 1]),
           "실제_대비_합성_연로그차": float(-np.log((act.loc[common, "Close"].iloc[-1] / act.loc[common, "Close"].iloc[0]) /
                                             (syn.loc[common, "Close"].iloc[-1] / syn.loc[common, "Close"].iloc[0])) / yrs),
           "시작": str(comp.index[0]), "실제_시작": str(a0)}
    return comp, chk


def fg_series() -> dict:
    fga = fg_data.fg().sort_index()
    rc = pd.read_parquet(CACHE / "p8_FGREENTRY_recon.parquet")["recon"].dropna()
    alt = pd.read_parquet(CACHE / "p8_RECON_index.parquet")["R252"].dropna()
    alt.index = pd.Index(alt.index.astype(str).str[:10])
    return {"fga": fga, "pre": rc[rc.index < "2011-01-03"], "pre_alt": alt[alt.index < "2011-01-03"]}


def fgp_on(days: np.ndarray, s: pd.Series) -> np.ndarray:
    """t일 시가에 쓸 F&G: 날짜가 t보다 앞선 최신 값."""
    d = np.asarray(s.index, dtype="U10")
    v = s.to_numpy(float)
    j = np.searchsorted(d, days, side="left") - 1
    return np.where(j >= 0, v[np.clip(j, 0, None)], np.nan)


# ---------------------------------------------------------------- 엔진(새로 짬)
@njit(cache=True)
def sim(O, H, L, C, trig, a, z, mode, x, N, maxwait, slip, cash_d, use_stop, rebuy, out_v, out_ev):
    """a−1일 종가에 들고 시작(가치 1) → z일 종가. trig[t] = t일(시가 모드) 또는 t일 종가(종가 모드)에 재매수 신호.
    mode 0: 장중 손절(시가 ≤ 선이면 시가, 저가 ≤ 선이면 선) + 시가 재매수. mode 1: 종가 ≤ 선 → 다음 날 종가 매도, 재매수 종가.
    out_v[k] = (a−1+k)일 종가 가치. out_ev 행 = (판 t, 산 t). 반환 = 사건 수."""
    sh = 1.0 / C[a - 1]
    cash = 0.0
    held = True
    peak = C[a - 1]
    ex = -1
    pend = False
    ne = 0
    out_v[0] = 1.0
    for t in range(a, z + 1):
        if mode == 1 and held and pend:
            cash = sh * C[t] * (1.0 - slip)
            sh = 0.0
            held = False
            pend = False
            ex = t
            out_ev[ne, 0] = t
            out_ev[ne, 1] = -1
            ne += 1
        if (not held) and rebuy and ex >= 0 and t > ex:
            d = t - ex
            if d >= N or d >= maxwait or trig[t]:
                px = (O[t] if mode == 0 else C[t]) * (1.0 + slip)
                sh = cash / px
                cash = 0.0
                held = True
                peak = px
                out_ev[ne - 1, 1] = t
        if mode == 0 and held and use_stop:
            line = peak * (1.0 - x)
            px = -1.0
            if O[t] <= line:
                px = O[t]
            elif L[t] <= line:
                px = line
            if px > 0:
                cash = sh * px * (1.0 - slip)
                sh = 0.0
                held = False
                ex = t
                out_ev[ne, 0] = t
                out_ev[ne, 1] = -1
                ne += 1
        if held:
            if C[t] > peak:
                peak = C[t]
            if mode == 1 and use_stop and C[t] <= peak * (1.0 - x):
                pend = True
        else:
            cash *= 1.0 + cash_d
        out_v[t - a + 1] = cash + sh * C[t]
    return ne


class Path_:
    def __init__(self, df: pd.DataFrame):
        self.days = np.asarray(df.index, dtype="U10")
        self.O, self.H, self.L, self.C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))

    def i(self, d, right=False):
        return int(np.searchsorted(self.days, d, side="right" if right else "left")) - (1 if right else 0)

    def run(self, d0, d1, trig, mode=0, x=X0, N=N0, slip=0.0, cash=0.0, stop=True, rebuy=True):
        a, z = max(self.i(d0), 1), self.i(d1, True)
        v = np.empty(z - a + 2)
        ev = np.full((z - a + 3, 2), -1, np.int64)
        n = sim(self.O, self.H, self.L, self.C, trig, a, z, mode, x, N, MAXWAIT, slip, cash / 252, stop, rebuy, v, ev)
        return {"a": a, "z": z, "v": v, "ev": ev[:n]}

    def seg(self, r, d0, d1):
        """r 경로를 d0 직전 종가 → d1 종가로 자른 누적 수익."""
        i0 = self.i(d0) - 1 - (r["a"] - 1)
        i1 = self.i(d1, True) - (r["a"] - 1)
        v = r["v"][i0:i1 + 1]
        return float(v[-1] / v[0] - 1), float((v / np.maximum.accumulate(v) - 1).min())

    def hold(self, d0, d1):
        return float(self.C[self.i(d1, True)] / self.C[self.i(d0) - 1] - 1)

    def events(self, r):
        return [(str(self.days[e]), None if b < 0 else str(self.days[b])) for e, b in r["ev"]]


def cagr(ret, n_days):
    return (1 + ret) ** (252 / n_days) - 1 if ret > -1 else -1.0


def window_stats(P: Path_, trig_open, trig_close, x=X0, N=N0, slip=0.0, cash=0.0, stop=True, rebuy=True) -> dict:
    """후보 측정 창 전부: 앞·뒤·AI(2026 연속 경로), 2025, LP·LC(연율), RP·RC(종가 모드, 연율)."""
    out = {}
    r = P.run(*D26, trig_open, 0, x, N, slip, cash, stop, rebuy)
    for wn, (d0, d1) in {"앞": FRONT, "뒤": BACK, "AI": SHOCK, "2026": D26}.items():
        out[wn], md = P.seg(r, d0, d1)
        out[wn + "_낙폭"] = md
        out[wn + "_손절"] = sum(1 for e, _ in P.events(r) if d0 <= e <= d1)
    out["_ev26"] = P.events(r)
    for wn, (d0, d1, mode, ann) in RUNS.items():
        rr = P.run(d0, d1, trig_open if mode == 0 else trig_close, mode, x, N, slip, cash, stop, rebuy)
        ret = float(rr["v"][-1] - 1)
        nd = rr["z"] - rr["a"] + 1
        out[wn] = cagr(ret, nd) if ann else ret
        out[wn + "_낙폭"] = float((rr["v"] / np.maximum.accumulate(rr["v"]) - 1).min())
        out[wn + "_시작"] = str(P.days[rr["a"]])
        out[wn + "_손절"] = len(rr["ev"])
    return out


WINS = ["앞", "뒤", "AI", "2026", "2025", "LP", "LC", "RP", "RC"]


def diff(a: dict, b: dict) -> dict:
    return {w: a[w] - b[w] for w in WINS if w in a and w in b}


# ---------------------------------------------------------------- 트리거
def trig_from(P: Path_, fg_full: pd.Series, T: float, lag: int = 0) -> np.ndarray:
    f = fgp_on(P.days, fg_full)
    if lag:
        f = np.concatenate([np.full(lag, np.nan), f[:-lag]])
    return np.nan_to_num(f, nan=-1.0) >= T


def trig_close_from(P: Path_, fg_full: pd.Series, T: float, lag: int = 0) -> np.ndarray:
    """종가 모드: t일 종가 체결은 t−1일 종가까지 안 F&G로 판단(엔진은 t일 신호를 그날 종가에 체결) → 시가 모드와 같은 배열."""
    return trig_from(P, fg_full, T, lag)


def full_fg(S: dict, fga: pd.Series | None = None, pre: str = "pre") -> pd.Series:
    return pd.concat([S[pre], S["fga"] if fga is None else fga])


def shifted(fga: pd.Series, k: int) -> pd.Series:
    return pd.Series(np.roll(fga.to_numpy(), k), index=fga.index)


def random_trig(P: Path_, fga: pd.Series, T: float, seed: int) -> np.ndarray:
    """해마다 실제 F&G ≥ T 날 수(그해 거래일 기준)만큼 무작위 날을 트리거로. 2011 전(재구성 구간)도 재구성 값의 해별 수를 쓴다."""
    rng = np.random.default_rng(seed)
    real = trig_from(P, fga, T)
    out = np.zeros(len(P.days), bool)
    yrs = np.array([d[:4] for d in P.days])
    for y in np.unique(yrs):
        idx = np.where(yrs == y)[0]
        k = int(real[idx].sum())
        if k:
            out[rng.choice(idx, size=k, replace=False)] = True
    return out


# ---------------------------------------------------------------- 시점 감사
def timing_audit(S: dict) -> dict:
    idx = pd.read_parquet(CACHE / "p7_idx.parquet")
    spx = idx[idx.sym == "^GSPC"].drop_duplicates("date").set_index("date")["Close"].sort_index()
    spx.index = pd.Index(pd.to_datetime(spx.index).strftime("%Y-%m-%d"))
    fga = S["fga"]
    common = fga.index.intersection(spx.index)
    f = fga.reindex(common)
    r = spx.pct_change().reindex(common)
    df = pd.DataFrame({"dF": f.diff(), "r0": r, "r_next": r.shift(-1), "r_prev": r.shift(1)}).dropna()
    out = {}
    for lab, (a, b) in {"보관본 2011~2020": ("2011-01-01", "2020-12-31"), "공식 2021-02~2026": ("2021-02-01", "2026-12-31"),
                        "2026": ("2026-01-01", "2026-12-31")}.items():
        d = df[(df.index >= a) & (df.index <= b)]
        out[lab] = {"일수": len(d), "같은날": float(d.dF.corr(d.r0)), "다음날(값이 하루 이름)": float(d.dF.corr(d.r_next)),
                    "전날(값이 하루 늦음)": float(d.dF.corr(d.r_prev))}
    by_year = {}
    for y in range(2011, 2027):
        d = df[df.index.str[:4] == str(y)]
        by_year[str(y)] = [round(float(d.dF.corr(d.r0)), 3), round(float(d.dF.corr(d.r_next)), 3), round(float(d.dF.corr(d.r_prev)), 3)]
    out["해마다 [같은날, 다음날, 전날]"] = by_year
    miss = [d for d in spx.index[(spx.index >= "2011-01-03") & (spx.index <= "2026-10-09")] if d not in set(fga.index)]
    out["S&P 거래일 중 F&G 빈 날"] = len(miss)
    out["빈 날 2026"] = [d for d in miss if d >= "2026"]
    return out


# ---------------------------------------------------------------- 실행
def main():
    raw = _raw()
    comp, chk = soxl_composite(raw)
    P = Path_(comp)
    S = fg_series()
    fga = S["fga"]
    FULL = full_fg(S)
    res = {"합성점검": chk}

    tR5 = trig_from(P, FULL, T0)
    tB = np.zeros(len(P.days), bool)
    B = window_stats(P, tB, tB)
    R5 = window_stats(P, tR5, tR5)
    H = window_stats(P, tB, tB, stop=False)
    res["재현"] = {"B": B, "R5": R5, "보유": H, "R5-B": diff(R5, B), "보유-B": diff(H, B)}
    print("R5−B", {k: round(v * 100, 2) for k, v in diff(R5, B).items()})
    print("R5 사건", R5["_ev26"])
    print("B 사건", B["_ev26"])

    # 다른 재구성(RP)
    FULL_ALT = full_fg(S, pre="pre_alt")
    tR5a = trig_from(P, FULL_ALT, T0)
    R5a = window_stats(P, tR5a, tR5a)
    res["다른재구성 RP·RC"] = {"RP": R5a["RP"] - B["RP"], "RC": R5a["RC"] - B["RC"]}

    # 민감도
    sens = {}
    tR5_l1 = trig_from(P, FULL, T0, lag=1)
    sens["하루늦춤"] = diff(window_stats(P, tR5_l1, tR5_l1), B)
    sens["하루늦춤_사건26"] = window_stats(P, tR5_l1, tR5_l1)["_ev26"]
    Bs = window_stats(P, tB, tB, slip=SLIP)
    sens["미끄러짐0.5%"] = diff(window_stats(P, tR5, tR5, slip=SLIP), Bs)
    Bc = window_stats(P, tB, tB, cash=CASH)
    sens["이자4%"] = diff(window_stats(P, tR5, tR5, cash=CASH), Bc)
    tR5_l2 = trig_from(P, FULL, T0, lag=2)
    sens["이틀늦춤"] = diff(window_stats(P, tR5_l2, tR5_l2), B)
    res["민감도"] = sens
    print("하루늦춤", {k: round(v * 100, 2) for k, v in sens["하루늦춤"].items() if not k.startswith("_")})

    # 시점 감사
    res["시점감사"] = timing_audit(S)
    print(json.dumps(res["시점감사"], ensure_ascii=False)[:600])

    # 이웃 격자
    nb = {}
    base_by = {}
    for x in NB_X:
        Bx = window_stats(P, tB, tB, x=x, N=20)
        for N in NB_N:
            Tx = window_stats(P, tB, tB, x=x, N=N)
            base_by[(x, N)] = Tx
            for T in NB_T:
                tr = trig_from(P, FULL, T)
                Rx = window_stats(P, tr, tr, x=x, N=N)
                nb[f"{x:.2f}|{N}|{T:.0f}"] = {"B20대비": diff(Rx, Bx), "시간만대비": diff(Rx, Tx)}
    res["이웃"] = nb
    # 위약
    n = len(fga)
    shifts = [int(PLACEBO_MIN + round(i * (n - 2 * PLACEBO_MIN) / (PLACEBO_N - 1))) for i in range(PLACEBO_N)]
    pl_shift, pl_rand = [], []
    same26 = []
    for k in shifts:
        ff = full_fg(S, shifted(fga, k))
        tr = trig_from(P, ff, T0)
        ws = window_stats(P, tr, tr)
        pl_shift.append(diff(ws, B))
        if ws["_ev26"] == R5["_ev26"]:
            same26.append(k)
    # 2026 재매수마다 무엇이 켰나(F&G ≥ 45 또는 20일)와 그날 쓴 F&G
    fgp = fgp_on(P.days, FULL)
    why = []
    for e, b in R5["_ev26"]:
        if b is None:
            continue
        ie, ib = P.i(e), P.i(b)
        why.append({"손절": e, "재매수": b, "대기일": ib - ie, "쓴F&G": float(fgp[ib]),
                    "이유": "20일" if ib - ie >= N0 else "F&G≥45"})
    res["재매수이유2026"] = why
    for sd in range(PLACEBO_N):
        tr = random_trig(P, FULL, T0, sd)
        pl_rand.append(diff(window_stats(P, tr, tr), B))
    res["위약"] = {"이동량": shifts, "원형밀기": pl_shift, "무작위날짜": pl_rand, "2026거래가같은이동량": same26}

    # 시대별
    eras = {}
    for en, (d0, d1, mode) in ERAS.items():
        rB = P.run(d0, d1, tB, mode)
        rR = P.run(d0, d1, tR5, mode)
        rL = P.run(d0, d1, tR5_l1, mode)
        eras[en] = {"B": float(rB["v"][-1] - 1), "R5": float(rR["v"][-1] - 1), "R5_하루늦춤": float(rL["v"][-1] - 1),
                    "보유": P.hold(d0, d1), "R5사건": P.events(rR), "B사건": P.events(rB)}
    for wn, (d0, d1) in {"2026 앞": FRONT, "2026 뒤": BACK, "7월 쇼크": SHOCK}.items():
        k = {"2026 앞": "앞", "2026 뒤": "뒤", "7월 쇼크": "AI"}[wn]
        eras[wn] = {"B": B[k], "R5": R5[k], "R5_하루늦춤": B[k] + sens["하루늦춤"][k], "보유": H[k]}
    res["시대"] = eras

    # 시작일 민감도
    st = []
    for d in P.days[(P.days >= START_SENS[0]) & (P.days <= START_SENS[1])]:
        rB = P.run(d, D26[1], tB)
        rR = P.run(d, D26[1], tR5)
        fB, _ = P.seg(rB, FRONT[0], FRONT[1]) if d <= FRONT[0] else P.seg(rB, d, FRONT[1])
        fR, _ = P.seg(rR, FRONT[0], FRONT[1]) if d <= FRONT[0] else P.seg(rR, d, FRONT[1])
        bB, _ = P.seg(rB, *BACK)
        bR, _ = P.seg(rR, *BACK)
        st.append({"시작": str(d), "앞": fR - fB, "뒤": bR - bB})
    res["시작일"] = st

    # 해마다(그해 첫날 들고 시작) R5 − B, 하루 늦춘 R5 − B, 보유 − B
    yr = {}
    for y in range(2011, 2026):
        d0, d1 = f"{y}-01-01", f"{y}-12-31"
        rB, rR, rL = (float(P.run(d0, d1, t)["v"][-1] - 1) for t in (tB, tR5, tR5_l1))
        yr[str(y)] = {"B": rB, "R5-B": rR - rB, "하루늦춤-B": rL - rB, "보유-B": P.hold(d0, d1) - rB}
    res["해마다"] = yr

    # 효율선: 같은 최대 낙폭이 나오는 SOXL 고정 비중(매일 맞춤, 현금 0) 대비
    fr = {}
    for wn, (d0, d1, ann) in {"앞": (*FRONT, False), "뒤": (*BACK, False), "2025": ("2025-01-02", "2025-12-31", False),
                              "LP": ("2011-01-03", "2018-12-31", True), "LC": ("2019-01-02", "2025-12-31", True)}.items():
        i0, i1 = P.i(d0) - 1, P.i(d1, True)
        ru = P.C[i0 + 1:i1 + 1] / P.C[i0:i1] - 1
        row = {}
        for nm, t in (("R5", tR5), ("B", tB)):
            r = P.run(d0, d1, t)
            ret, md = float(r["v"][-1] - 1), float((r["v"] / np.maximum.accumulate(r["v"]) - 1).min())
            best, bw = -1.0, 0.0
            for w in np.arange(0, 1.0001, 0.01):
                v = np.concatenate([[1.0], np.cumprod(1 + w * ru)])
                if (v / np.maximum.accumulate(v) - 1).min() >= md - 1e-12 and v[-1] - 1 > best:
                    best, bw = float(v[-1] - 1), float(w)
            nd = len(ru)
            row[nm] = {"수익": cagr(ret, nd) if ann else ret, "낙폭": md, "같은낙폭비중": bw,
                       "그비중수익": cagr(best, nd) if ann else best,
                       "효율선초과": (cagr(ret, nd) - cagr(best, nd)) if ann else ret - best}
        fr[wn] = row
    res["효율선"] = fr

    # 이웃 격자를 하루 늦춘 F&G로(긴 역사 LP·LC만: 2011~2012 보관본 시점 의심 때문)
    nb_l = {}
    for x in NB_X:
        Bx = window_stats(P, tB, tB, x=x, N=20)
        for N in NB_N:
            for T in NB_T:
                tr = trig_from(P, FULL, T, lag=1)
                Rx = window_stats(P, tr, tr, x=x, N=N)
                nb_l[f"{x:.2f}|{N}|{T:.0f}"] = diff(Rx, Bx)
    res["이웃_하루늦춤"] = nb_l

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장", RESULTS / f"{ID}.json")


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


if __name__ == "__main__":
    main()
