"""8단계 갈래 AIFG(2026-10-11): 사용자 종목(AI·반도체)에 맞춘 'AI 공포탐욕 지수'를 만들고 CNN F&G와 같은 규칙으로 나란히 잰다.

질문: CNN F&G는 S&P 500 기준이라 7월 반도체 쇼크 같은 업종 사건을 늦게·약하게 잡을 수 있다. 반도체·AI 재료로 만든
지수가 (1) 7종목 비중 조절, (2) AI 유니버스 종목 사건형 매수, (3) SOXL 비중 조절에서 CNN판보다 나은가.

지수(사전 고정, 가중을 고르지 않음. 모두 그날 종가로 알 수 있는 값 → 다음 거래일 시가부터 사용)
  A1 ^SOX 종가/125일 평균 − 1                         (높을수록 탐욕)
  A2 AI 유니버스 50일선 위 비율                          (높을수록 탐욕)
  A3 AI 유니버스 52주(252일) 신고가 − 신저가 순비율          (높을수록 탐욕)
  A4 ^SOX 20일 실현 변동성 / 그 50일 평균                  (높을수록 공포 → 뒤집어 씀)
  A5 SOXL/SOXS 달러 거래대금 비의 5일 평균                  (높을수록 탐욕: 개인 레버리지 쏠림, 사전 고정 방향)
  A6 SMH 20일 수익 − TLT 20일 수익                       (높을수록 탐욕)
  A7 HYG 20일 수익 − IEF 20일 수익                       (높을수록 탐욕)
  각 요소를 과거 252거래일(그날 포함) 백분위 0~100으로 바꾼 뒤 같은 가중 평균 = 'AI'. 대안 하나: A2·A3·A5만 = 'AI쏠림'.
  A5는 yfinance SOXS 거래량이 역분할 수정 반올림으로 2020년 전 0~몇 주라 품질 규칙(20일 최소 10주) 뒤부터만 쓴다 →
  백분위는 2021-07경부터. 그 전 지수는 있는 요소만 같은 가중(AI = 6개, AI쏠림 = A2·A3).
  AI 유니버스 = watchlist.AI_UNIVERSE 미국 상장 115종목(2026-10 목록 → 생존편향), 하루 +150%/−70% 종가 변화가 있는 종목은 자료 오류로 뺀다.
  자료: cache/p8_AIFG_px.parquet(`py p8_AIFG.py fetch`, yfinance 2011~), ^SOX는 cache/p7_idx.parquet.

규칙(FGDIRECT 갈래와 같은 형태·같은 격자를 세 지수 AI·AI쏠림·CNN에 그대로)
  S1 공포 추가: 평소 w0, 지수 ≤ T이면 1.0, 지수 ≥ T_exit이면 다시 w0.   w0 3 × T 4 × T_exit 3 = 36
  S2 탐욕 덜기: 평소 1.0, 지수 ≥ G이면 w1, 지수 ≤ G_back이면 다시 1.0.   w1 2 × G 3 × G_back 2 = 12
  E1 레벨 진입: 지수가 T 이하로 처음 내려간 날(직전에 T+10 위였던 뒤에만 새 사건).  T 6개
  E2 공포 탈출: T 이하에 있다가 T+δ 위로 올라선 첫날.  T 4 × δ 3 = 12
  사건형 보유 H ∈ {21, 63, 126}거래일: 신호일 s 종가 → s+1 시가 매수 → s+H 종가 평가(trade_lab과 같은 시점).
쓰임
  (1) 7종목 경로(harness.base('2026')·base('hold')) 노출 = S1·S2 상태(at_open). 2026 앞 고르기 → 뒤 확인, 2025·2024 보조.
  (2) AI 유니버스 종목 사건형(동일가중 바구니) 대 같은 종목 '아무 날'. 긴 역사 2014~2019 고르기 → 2020~2025 확인.
  (3) SOXL(실제 상장 상품, lev5.bars) 노출 = S1·S2 상태. 2026 앞 → 뒤, 긴 역사 2014~2019 고르기 → 2020~2025 확인.
지표(사전 고정)
  전략형: 구간 수익·최대 낙폭, 평균 노출, 고정초과(같은 구간 평균 노출로 고정했을 때 대비), 효율선초과(같은 최대 낙폭이
  나오는 고정 비중 0~1 중 최고 수익 대비; 2026 구간은 구간 %p, 긴 구간은 연 %p).
  사건형: 바구니 수익 − 같은 구간 아무 날 바구니 평균(초과), 승률(바구니 > 0), 추가 −20%(보유 중 종가가 산 값의 80% 이하로
  간 종목 비율) − 아무 날의 같은 값, 사건 수.
고르기(사전 고정)
  전략형: 지수마다 48개 중 고르기 구간 효율선초과 최대 → 확인 구간 효율선초과 > 0이고 고정초과 > 0이면 '확인 통과'.
  사건형: 18 정의 × 3 H = 54칸 중 고르기 구간 초과 최대(사건 5건 이상) → 확인 구간 초과 > 0이면 통과.
  대표 후보 = 2026 앞 고르기 → 뒤 확인 통과이면서 같은 규칙의 CNN판보다 뒤·2026 전체 효율선초과가 큰 것.
민감도: 하루 늦춤(신호를 하루 더 늦게 씀), 미끄러짐 0.5%(노출 변화 1당 편도), 현금 이자 4%, 위약(지수를 63×k일 원형으로 밀기, k=1..20).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fg_data  # noqa: E402
import harness as HN  # noqa: E402
import lev5  # noqa: E402

ID = "p8_AIFG"
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"

# ============================================================================ 사전 고정 상수·격자
PCT_WIN = 252
COMP_ALL = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
COMP_CROWD = ("A2", "A3", "A5")
IDX = ("AI", "AI쏠림", "CNN")
S1_W0 = (0.6, 0.7, 0.8)
S1_T = (10, 20, 25, 30)
S1_TEXIT = (40, 50, 60)
S2_W1 = (0.5, 0.75)
S2_G = (75, 80, 85)
S2_GBACK = (50, 60)
E1_T = (5, 10, 15, 20, 25, 30)
E1_RESET = 10
E2_T = (10, 15, 20, 25)
E2_D = (0, 5, 10)
HOLD = (21, 63, 126)
MIN_EV_SELECT = 5
TAIL = -0.20
EG = np.round(np.arange(0.0, 1.0001, 0.005), 3)
SLIP = 0.005
CASH = 0.04
PLACEBO = tuple(63 * k for k in range(1, 21))
BAD_JUMP = 1.5
A5_MIN_SHARES = 10
FEAR, GREED = 25, 75

W26 = {"2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "AI쇼크": ("2026-06-22", "2026-07-29"),
       "2026전체": ("2026-01-02", "2026-10-09")}
WHOLD = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"), "관세": ("2025-02-18", "2025-04-04"),
         "24여름": ("2024-06-18", "2024-08-05")}
WLONG = {"긴고르기": ("2014-01-02", "2019-12-31"), "긴확인": ("2020-01-02", "2025-12-31")}
WCNN_LONG = {"CNN긴고르기": ("2011-01-03", "2018-12-31"), "CNN긴확인": ("2019-01-02", "2025-12-31")}
EPISODES = {"2024-08 엔캐리": ("2024-06-01", "2024-09-30"), "2025-04 관세": ("2025-01-15", "2025-05-31"),
            "2026-03": ("2026-01-15", "2026-04-30"), "2026-06~07 AI쇼크": ("2026-05-15", "2026-08-31")}
ANNUAL = {"긴고르기", "긴확인", "CNN긴고르기", "CNN긴확인", "2025", "2024"}


def grid_size() -> dict:
    s = len(S1_W0) * len(S1_T) * len(S1_TEXIT) + len(S2_W1) * len(S2_G) * len(S2_GBACK)
    e = (len(E1_T) + len(E2_T) * len(E2_D)) * len(HOLD)
    return {"전략형_규칙": s, "전략형_대상": 3, "사건형_칸": e, "지수": len(IDX),
            "합계": s * len(IDX) * 2 + e * len(IDX)}   # 7종목·SOXL 두 대상, 사건형 AI 유니버스


# ============================================================================ 자료
_PX = None


def px_long() -> pd.DataFrame:
    global _PX
    if _PX is None:
        _PX = pd.read_parquet(CACHE / "p8_AIFG_px.parquet")
    return _PX


def wide(field: str, syms=None) -> pd.DataFrame:
    d = px_long()
    if syms is not None:
        d = d[d.sym.isin(list(syms))]
    return d.pivot_table(index="date", columns="sym", values=field).sort_index()


def universe() -> list:
    sys.path.insert(0, str(HERE.parents[4]))
    import watchlist
    out = []
    for g in watchlist.AI_UNIVERSE.values():
        for lst in g.values():
            out += [t for t in lst if "." not in t]
    return sorted(set(out))


def stock_panel():
    """AI 유니버스 수정 시가·종가(wide) + 자료 오류로 뺀 종목."""
    syms = universe()
    c, o, ac = wide("Close", syms), wide("Open", syms), wide("Adj Close", syms)
    f = (ac / c)
    ao = o * f
    bad = []
    for s in ac.columns:
        x = ac[s].dropna()
        r = x.to_numpy()[1:] / x.to_numpy()[:-1] - 1
        if len(r) and (np.any(r > BAD_JUMP) or np.any(r < -0.7)):
            bad.append(s)
    keep = [s for s in ac.columns if s not in bad]
    return ao[keep], ac[keep], bad


def sox() -> pd.Series:
    d = pd.read_parquet(CACHE / "p7_idx.parquet")
    s = d[d.sym == "^SOX"].set_index("date")["Close"].astype(float).sort_index()
    return s[~s.index.duplicated()]


# ============================================================================ 지수
def pct_rank(x: pd.Series, sign: int = 1) -> pd.Series:
    """과거 252거래일(그날 포함) 백분위 0~100. 미래 값 없음."""
    return (sign * x).rolling(PCT_WIN, min_periods=PCT_WIN).rank(pct=True) * 100


def components() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    ao, ac, bad = stock_panel()
    days = ac.index
    s = sox().reindex(days)
    raw = pd.DataFrame(index=days)
    raw["A1"] = s / s.rolling(125, min_periods=125).mean() - 1
    ma50 = ac.rolling(50, min_periods=50).mean()
    v50 = ma50.notna() & ac.notna()
    raw["A2"] = ((ac > ma50) & v50).sum(axis=1) / v50.sum(axis=1).replace(0, np.nan)
    hi, lo = ac.rolling(252, min_periods=252).max(), ac.rolling(252, min_periods=252).min()
    v252 = hi.notna() & ac.notna()
    nh = ((ac >= hi - 1e-12) & v252).sum(axis=1)
    nl = ((ac <= lo + 1e-12) & v252).sum(axis=1)
    raw["A3"] = (nh - nl) / v252.sum(axis=1).replace(0, np.nan)
    lr = np.log(s).diff()
    rv = lr.rolling(20, min_periods=20).std() * np.sqrt(252)
    raw["A4"] = rv / rv.rolling(50, min_periods=50).mean()
    vol = wide("Volume", ["SOXL", "SOXS"]).reindex(days)
    e = wide("Close", ["SOXL", "SOXS"]).reindex(days) * vol
    # yfinance 거래량은 역분할 수정 뒤 정수로 반올림돼 SOXS 옛 거래량이 0~몇 주로 뭉개진다(누적 역분할 약 1/96억).
    # 자료 품질 규칙(사전 고정): SOXS 수정 거래량의 20일 최솟값이 A5_MIN_SHARES 이상인 날부터만 A5를 쓴다(반올림 오차 ≤ 5%).
    ok = vol["SOXS"].rolling(20, min_periods=20).min() >= A5_MIN_SHARES
    a5_start = ok[~ok].index.max()                      # 이 날 뒤로는 늘 품질 기준을 넘는다
    ok = pd.Series(ok.index > a5_start, index=ok.index)
    raw["A5"] = (e["SOXL"] / e["SOXS"].replace(0, np.nan)).rolling(5, min_periods=5).mean().where(ok)
    a = wide("Adj Close", ["SMH", "TLT", "HYG", "IEF"]).reindex(days)
    r20 = a / a.shift(20) - 1
    raw["A6"] = r20["SMH"] - r20["TLT"]
    raw["A7"] = r20["HYG"] - r20["IEF"]
    sign = {"A4": -1}
    pc = pd.DataFrame({k: pct_rank(raw[k].dropna(), sign.get(k, 1)) for k in COMP_ALL}).reindex(days)
    return raw, pc, bad


def build_indices():
    raw, pc, bad = components()
    # 있는 요소만 같은 가중(사전 고정): A5가 없는 2021년 전은 AI = 나머지 6개, AI쏠림 = A2·A3.
    n_all, n_cr = pc[list(COMP_ALL)].notna().sum(axis=1), pc[list(COMP_CROWD)].notna().sum(axis=1)
    ai = pc[list(COMP_ALL)].mean(axis=1)[n_all >= len(COMP_ALL) - 1]
    crowd = pc[list(COMP_CROWD)].mean(axis=1)[n_cr >= len(COMP_CROWD) - 1]
    cnn = fg_data.fg()
    out = {"AI": ai.rename("AI"), "AI쏠림": crowd.rename("AI쏠림"), "CNN": cnn.rename("CNN")}
    return out, raw, pc, bad


# ============================================================================ 규칙 상태
def rules_s() -> list[dict]:
    out = []
    for w0 in S1_W0:
        for t in S1_T:
            for tx in S1_TEXIT:
                out.append({"id": f"S1 w0={w0} T={t} 해제={tx}", "kind": "S1", "w0": w0, "T": t, "Tx": tx})
    for w1 in S2_W1:
        for g in S2_G:
            for gb in S2_GBACK:
                out.append({"id": f"S2 w1={w1} G={g}복귀={gb}", "kind": "S2", "w1": w1, "G": g, "Gb": gb})
    return out


def state(x: pd.Series, rule: dict) -> pd.Series:
    v = x.to_numpy(float)
    out = np.empty(len(v))
    if rule["kind"] == "S1":
        cur = rule["w0"]
        for i, z in enumerate(v):
            if z <= rule["T"]:
                cur = 1.0
            elif z >= rule["Tx"]:
                cur = rule["w0"]
            out[i] = cur
    else:
        cur = 1.0
        for i, z in enumerate(v):
            if z >= rule["G"]:
                cur = rule["w1"]
            elif z <= rule["Gb"]:
                cur = 1.0
            out[i] = cur
    return pd.Series(out, index=x.index)


def init_w(rule):
    return rule["w0"] if rule["kind"] == "S1" else 1.0


def rules_e() -> list[dict]:
    out = [{"id": f"E1 T={t}", "kind": "E1", "T": t} for t in E1_T]
    out += [{"id": f"E2 T={t} δ={d}", "kind": "E2", "T": t, "d": d} for t in E2_T for d in E2_D]
    return out


def events(x: pd.Series, rule: dict) -> list:
    v, days = x.to_numpy(float), list(x.index)
    ev = []
    if rule["kind"] == "E1":
        armed = False
        for i, z in enumerate(v):
            if z > rule["T"] + E1_RESET:
                armed = True
            elif armed and z <= rule["T"]:
                ev.append(days[i])
                armed = False
    else:
        low = False
        for i, z in enumerate(v):
            if z <= rule["T"]:
                low = True
            elif low and z > rule["T"] + rule["d"]:
                ev.append(days[i])
                low = False
    return ev


# ============================================================================ 전략형 평가
class Target:
    """일별 수익 경로: ret[k] = k일 시가 → k+1일 시가(마지막 날 없음)."""

    def __init__(self, name: str, days: list, ret: pd.Series, windows: dict):
        self.name, self.days, self.ret, self.windows = name, list(days), ret, windows
        self._fr = {}

    def frontier(self, lo, hi, cash):
        key = (lo, hi, cash)
        if key not in self._fr:
            rb = self.ret[(self.ret.index >= lo) & (self.ret.index <= hi)].to_numpy()
            daily = np.outer(EG, rb) + np.outer(1 - EG, np.full(len(rb), cash / 252))
            cur = np.concatenate([np.ones((len(EG), 1)), np.cumprod(1 + daily, axis=1)], axis=1)
            self._fr[key] = (cur[:, -1] - 1, (cur / np.maximum.accumulate(cur, axis=1) - 1).min(axis=1), len(rb))
        return self._fr[key]


def seven(bn: str) -> Target:
    b = HN.base(bn)
    w = W26 if bn == "2026" else WHOLD
    return Target(f"7종목_{bn}", b.days, b.ret, w)


def soxl() -> Target:
    br = lev5.bars("SOXL")
    o = pd.Series(br.o, index=list(br.days))
    r = (o.shift(-1) / o - 1).dropna()
    w = {**W26, **WHOLD, **WLONG, **WCNN_LONG}
    return Target("SOXL", list(br.days), r, w)


def exposure(tg: Target, x: pd.Series, rule: dict, lag: int = 0) -> pd.Series:
    st = state(x, rule)
    if lag:
        st = st.shift(lag).dropna()
    e = HN.at_open(st, types.SimpleNamespace(days=tg.days), fill=init_w(rule))
    return e.reindex(tg.ret.index).astype(float)


def wstats(tg: Target, e: pd.Series, lo: str, hi: str, annual: bool, slip: float = 0.0, cash: float = 0.0) -> dict:
    m = (tg.ret.index >= lo) & (tg.ret.index <= hi)
    rb, ee = tg.ret[m].to_numpy(), e[m].to_numpy()
    if not len(rb):
        return {}
    turn = np.abs(np.diff(np.concatenate([[ee[0]], ee])))
    daily = ee * rb + (1 - ee) * cash / 252 - slip * turn
    cur = np.concatenate([[1.0], np.cumprod(1 + daily)])
    ret, dd = float(cur[-1] - 1), float((cur / np.maximum.accumulate(cur) - 1).min())
    avg = float(ee.mean())
    fixed = float(np.prod(1 + avg * rb + (1 - avg) * cash / 252) - 1)
    fret, fdd, n = tg.frontier(lo, hi, cash)
    ok = fdd >= dd - 1e-12
    j = int(np.argmax(np.where(ok, fret, -np.inf)))
    yrs = n / 252
    if annual:
        a = lambda z: (1 + z) ** (1 / yrs) - 1  # noqa: E731
        fx, ef = a(ret) - a(fixed), a(ret) - a(fret[j])
    else:
        fx, ef = ret - fixed, ret - fret[j]
    bh = float(np.prod(1 + rb) - 1)
    return {"수익": ret, "낙폭": dd, "노출": avg, "고정초과": float(fx), "효율선초과": float(ef), "효율선w": float(EG[j]),
            "매매수_연": float((turn > 1e-9).sum() / yrs), "보유수익": bh, "연율": annual}


def run_rule(tg: Target, x: pd.Series, rule: dict, lag=0, slip=0.0, cash=0.0, wins=None) -> dict:
    e = exposure(tg, x, rule, lag)
    return {w: wstats(tg, e, lo, hi, w in ANNUAL, slip, cash) for w, (lo, hi) in (wins or tg.windows).items()}


def trace(tg: Target, x: pd.Series, rule: dict, lo="2026-06-01", hi="2026-08-31") -> list:
    e = exposure(tg, x, rule)
    e = e[(e.index >= lo) & (e.index <= hi)].round(3)
    ch = e[e.ne(e.shift())]
    xv = x.copy()
    out = []
    for d, v in ch.items():
        prev = xv[xv.index < d]
        out.append({"날짜(시가)": d, "노출": float(v), "전날지수": round(float(prev.iloc[-1]), 1) if len(prev) else None})
    return out


def select(rows: dict, sel: str, conf: str) -> dict:
    """rows: {rule_id: {window: stats}}. 고르기 구간 효율선초과 최대 → 확인."""
    rid = max(rows, key=lambda k: rows[k][sel]["효율선초과"])
    r = rows[rid]
    passed = r[conf]["효율선초과"] > 0 and r[conf]["고정초과"] > 0
    pos = [k for k in rows if rows[k][sel]["효율선초과"] > 0]
    info = {"고르기>0": len(pos), "전체": len(rows),
            "확인>0|고르기>0": float(np.mean([rows[k][conf]["효율선초과"] > 0 for k in pos])) if pos else float("nan"),
            "확인>0 무조건": float(np.mean([rows[k][conf]["효율선초과"] > 0 for k in rows]))}
    return {"id": rid, "통과": bool(passed), "정보": info}


def placebo_rank(tg: Target, x: pd.Series, rule: dict, win: str) -> dict:
    lo, hi = tg.windows[win]
    act = run_rule(tg, x, rule, wins={win: (lo, hi)})[win]["효율선초과"]
    vals = []
    for s in PLACEBO:
        xs = pd.Series(np.roll(x.to_numpy(), s), index=x.index)
        vals.append(run_rule(tg, xs, rule, wins={win: (lo, hi)})[win]["효율선초과"])
    rank = 1 + int(np.sum(np.array(vals) > act))
    return {"구간": win, "실제": act, "순위": rank, "개수": len(vals) + 1, "위약중앙": float(np.median(vals))}


# ============================================================================ 사건형 평가
class Basket:
    def __init__(self):
        ao, ac, _ = stock_panel()
        self.days = list(ac.index)
        self.pos = {d: i for i, d in enumerate(self.days)}
        A, C = ao.to_numpy(float), ac.to_numpy(float)
        self.R, self.T = {}, {}
        n = len(self.days)
        for H in HOLD:
            ent = np.vstack([A[1:], np.full((1, A.shape[1]), np.nan)])           # s+1 시가
            ex = np.vstack([C[H:], np.full((H, C.shape[1]), np.nan)])            # s+H 종가
            mn = pd.DataFrame(C).rolling(H, min_periods=H).min().shift(-H).to_numpy()   # s+1..s+H 최저 종가
            r = ex / ent - 1
            t = (mn / ent - 1 <= TAIL).astype(float)
            ok = np.isfinite(r) & np.isfinite(mn)
            r[~ok] = np.nan
            t[~ok] = np.nan
            cnt = ok.sum(axis=1)
            with np.errstate(all="ignore"):
                import warnings
                warnings.simplefilter("ignore", RuntimeWarning)
                R = np.where(cnt > 0, np.nanmean(r, axis=1), np.nan)
                T = np.where(cnt > 0, np.nanmean(t, axis=1), np.nan)
            self.R[H] = pd.Series(R, index=self.days)
            self.T[H] = pd.Series(T, index=self.days)

    def stats(self, evs: list, H: int, lo: str, hi: str) -> dict:
        R, T = self.R[H], self.T[H]
        d = np.asarray(self.days)
        # 확인 구간 밖으로 결과가 새지 않게: 신호일 s + H거래일이 구간 끝 이하인 날만
        last = int(np.searchsorted(d, hi, side="right")) - 1
        allm = (d >= lo) & (d <= hi) & R.notna().to_numpy() & (np.arange(len(d)) + H <= last)
        base_r, base_t = R[allm], T[allm]
        ev = [d for d in evs if d in self.pos and allm[self.pos[d]]]
        er, et = R.reindex(ev), T.reindex(ev)
        if not len(ev):
            return {"사건": 0, "초과": float("nan"), "승률": float("nan"), "기준승률": float(np.mean(base_r > 0)) if len(base_r) else float("nan"),
                    "꼬리초과": float("nan"), "평균": float("nan"), "기준평균": float(base_r.mean()) if len(base_r) else float("nan")}
        return {"사건": len(ev), "초과": float(er.mean() - base_r.mean()), "승률": float(np.mean(er > 0)),
                "기준승률": float(np.mean(base_r > 0)), "꼬리초과": float(et.mean() - base_t.mean()), "평균": float(er.mean()),
                "기준평균": float(base_r.mean()), "날짜": ev}


def event_rows(bk: Basket, x: pd.Series, wins: dict) -> dict:
    out = {}
    for rule in rules_e():
        evs = events(x, rule)
        for H in HOLD:
            out[f"{rule['id']} H={H}"] = {w: bk.stats(evs, H, lo, hi) for w, (lo, hi) in wins.items()}
    return out


def event_select(rows: dict, sel: str, conf: str) -> dict:
    ok = [k for k in rows if rows[k][sel]["사건"] >= MIN_EV_SELECT and np.isfinite(rows[k][sel]["초과"])]
    if not ok:
        return {"id": None, "통과": False}
    rid = max(ok, key=lambda k: rows[k][sel]["초과"])
    c = rows[rid][conf]
    pos = [k for k in ok if rows[k][sel]["초과"] > 0]
    fin = [k for k in pos if np.isfinite(rows[k][conf]["초과"])]
    return {"id": rid, "통과": bool(c["사건"] > 0 and c["초과"] > 0),
            "정보": {"고르기>0": len(pos), "후보": len(ok),
                   "확인>0|고르기>0": float(np.mean([rows[k][conf]["초과"] > 0 for k in fin])) if fin else float("nan")}}


def event_placebo(bk: Basket, x: pd.Series, rid: str, win: tuple) -> dict:
    name, H = rid.rsplit(" H=", 1)
    H = int(H)
    rule = next(r for r in rules_e() if r["id"] == name)
    act = bk.stats(events(x, rule), H, *win)["초과"]
    vals = []
    for s in PLACEBO:
        xs = pd.Series(np.roll(x.to_numpy(), s), index=x.index)
        v = bk.stats(events(xs, rule), H, *win)["초과"]
        vals.append(v if np.isfinite(v) else -np.inf)
    return {"실제": act, "순위": 1 + int(np.sum(np.array(vals) > act)), "개수": len(vals) + 1}


# ============================================================================ 지수 비교
def compare(ix: dict) -> dict:
    a, c, w = ix["AI"], ix["CNN"], ix["AI쏠림"]
    j = pd.concat([a, w, c], axis=1).dropna()
    out = {"상관_수준": {"AI-CNN": float(j.AI.corr(j.CNN)), "쏠림-CNN": float(j["AI쏠림"].corr(j.CNN)), "AI-쏠림": float(j.AI.corr(j["AI쏠림"]))},
           "상관_20일변화": {"AI-CNN": float(j.AI.diff(20).corr(j.CNN.diff(20)))},
           "해별상관_AI-CNN": {y: float(g.AI.corr(g.CNN)) for y, g in j.groupby(j.index.str[:4])},
           "분포": {}}
    for k, s in ix.items():
        s = s[s.index >= "2014-01-01"]
        out["분포"][k] = {"일수": len(s), "≤10": int((s <= 10).sum()), "≤20": int((s <= 20).sum()), "≤25": int((s <= 25).sum()),
                        "≥75": int((s >= 75).sum()), "≥80": int((s >= 80).sum()), "평균": float(s.mean()), "표준편차": float(s.std())}
    sx = sox()
    b26 = HN.base("2026")
    seven_c = pd.Series(b26.close, index=b26.days)
    ep = {}
    for name, (lo, hi) in EPISODES.items():
        s = sx[(sx.index >= lo) & (sx.index <= hi)]
        tr = s.idxmin()
        pk = s[s.index <= tr].idxmax()
        row = {"SOX고점": pk, "SOX저점": tr, "SOX낙폭": float(s[tr] / s[pk] - 1)}
        for k, x in ix.items():
            xx = x[(x.index >= lo) & (x.index <= hi)]
            fear = xx[(xx <= FEAR) & (xx.index >= pk)]
            greed = xx[(xx >= GREED) & (xx.index <= pk)]
            f1 = fear.index[0] if len(fear) else None
            row[k] = {"최저": round(float(xx.min()), 1), "최저일": xx.idxmin(),
                      "첫공포≤25(고점뒤)": f1, "그때SOX낙폭": float(sx[f1] / s[pk] - 1) if f1 else None,
                      "고점전마지막탐욕≥75": greed.index[-1] if len(greed) else None,
                      "고점날값": round(float(x[x.index <= pk].iloc[-1]), 1)}
        ep[name] = row
    out["사건비교"] = ep
    last = {k: round(float(v.iloc[-1]), 1) for k, v in ix.items()}
    out["2026-10-09"] = last
    _ = seven_c
    return out


# ============================================================================ 실행
def main():
    ix, raw, pc, bad = build_indices()
    print("자료 오류로 뺀 종목:", bad)
    print("AI 지수", ix["AI"].index[0], "~", ix["AI"].index[-1], len(ix["AI"]))
    comp = compare(ix)
    comp["요소_10-09"] = {k: round(float(pc[k].dropna().iloc[-1]), 1) for k in COMP_ALL}
    comp["요소상관_A1"] = {k: float(pc[k].corr(pc["A1"])) for k in COMP_ALL}
    comp["요소상관_나머지평균"] = {k: float(pc[k].corr(pc[[c for c in COMP_ALL if c != k]].mean(axis=1))) for k in COMP_ALL}
    print(json.dumps(comp, ensure_ascii=False, indent=1, default=str)[:6000])

    tg7 = {"2026": seven("2026"), "hold": seven("hold")}
    tgs = soxl()
    RS = rules_s()
    res = {"7종목": {}, "SOXL": {}}
    for k, x in ix.items():
        r7 = {}
        for rule in RS:
            a = run_rule(tg7["2026"], x, rule)
            b = run_rule(tg7["hold"], x, rule)
            r7[rule["id"]] = {**a, **b}
        res["7종목"][k] = r7
        res["SOXL"][k] = {rule["id"]: run_rule(tgs, x, rule) for rule in RS}
    base7 = {**{w: wstats(tg7["2026"], pd.Series(1.0, index=tg7["2026"].ret.index), lo, hi, w in ANNUAL) for w, (lo, hi) in W26.items()},
             **{w: wstats(tg7["hold"], pd.Series(1.0, index=tg7["hold"].ret.index), lo, hi, w in ANNUAL) for w, (lo, hi) in WHOLD.items()}}
    baseS = {w: wstats(tgs, pd.Series(1.0, index=tgs.ret.index), lo, hi, w in ANNUAL) for w, (lo, hi) in tgs.windows.items()}

    sel = {"7종목": {}, "SOXL": {}, "SOXL긴": {}}
    for k in IDX:
        sel["7종목"][k] = select(res["7종목"][k], "2026앞", "2026뒤")
        sel["SOXL"][k] = select(res["SOXL"][k], "2026앞", "2026뒤")
        sel["SOXL긴"][k] = select(res["SOXL"][k], "긴고르기", "긴확인")
    rmap = {r["id"]: r for r in RS}

    def cnn_same(target, rid):
        return res[target]["CNN"][rid]

    # 대표 후보: AI·AI쏠림 7종목/SOXL 2026 고르기 통과 + 같은 규칙 CNN판보다 뒤·전체 효율선초과 큼
    cands = []
    for target in ("7종목", "SOXL"):
        for k in ("AI", "AI쏠림"):
            s = sel[target][k]
            r, c = res[target][k][s["id"]], cnn_same(target, s["id"])
            better = r["2026뒤"]["효율선초과"] > c["2026뒤"]["효율선초과"] and r["2026전체"]["효율선초과"] > c["2026전체"]["효율선초과"]
            cands.append({"대상": target, "지수": k, "규칙": s["id"], "통과": s["통과"], "CNN보다나음": bool(better),
                          "뒤효율": r["2026뒤"]["효율선초과"], "CNN뒤효율": c["2026뒤"]["효율선초과"],
                          "전체효율": r["2026전체"]["효율선초과"], "CNN전체효율": c["2026전체"]["효율선초과"]})
    # 민감도·위약(지수마다 2026 고른 규칙)
    sens = {}
    for target, tg in (("7종목", tg7["2026"]), ("SOXL", tgs)):
        for k in IDX:
            rid = sel[target][k]["id"]
            rule, x = rmap[rid], ix[k]
            sens[f"{target}|{k}"] = {
                "규칙": rid,
                "하루늦춤": {w: run_rule(tg, x, rule, lag=1)[w] for w in ("2026앞", "2026뒤", "AI쇼크")},
                "미끄러짐0.5%": {w: run_rule(tg, x, rule, slip=SLIP)[w] for w in ("2026앞", "2026뒤")},
                "이자4%": {w: run_rule(tg, x, rule, cash=CASH)[w] for w in ("2026앞", "2026뒤")},
                "위약_뒤": placebo_rank(tg, x, rule, "2026뒤"),
                "위약_전체": placebo_rank(tg, x, rule, "2026전체"),
                "7월": trace(tg, x, rule),
            }
            if target == "SOXL":
                sens[f"{target}|{k}"]["위약_긴확인"] = placebo_rank(tg, x, rule, "긴확인")
    for k in IDX:
        rid = sel["SOXL긴"][k]["id"]
        rule, x = rmap[rid], ix[k]
        sens[f"SOXL긴|{k}"] = {"규칙": rid, "위약_긴확인": placebo_rank(tgs, x, rule, "긴확인"),
                              "하루늦춤": {w: run_rule(tgs, x, rule, lag=1)[w] for w in ("긴고르기", "긴확인")},
                              "미끄러짐0.5%": {w: run_rule(tgs, x, rule, slip=SLIP)[w] for w in ("긴고르기", "긴확인")},
                              "이자4%": {w: run_rule(tgs, x, rule, cash=CASH)[w] for w in ("긴고르기", "긴확인")}}

    # 사건형
    bk = Basket()
    EW = {**WLONG, "2026앞": W26["2026앞"], "2026뒤": W26["2026뒤"], "2026전체": W26["2026전체"], "2025": WHOLD["2025"]}
    ev = {k: event_rows(bk, x, EW) for k, x in ix.items()}
    ev_cnn_long = event_rows(bk, ix["CNN"], WCNN_LONG)
    esel = {k: event_select(ev[k], "긴고르기", "긴확인") for k in IDX}
    esel["CNN(2011~2018→2019~2025)"] = event_select(ev_cnn_long, "CNN긴고르기", "CNN긴확인")
    eplac = {k: event_placebo(bk, ix[k], esel[k]["id"], WLONG["긴확인"]) for k in IDX if esel[k]["id"]}
    # 사건형 하루 늦춤(신호 다음다음 날 시가 매수): 지수를 한 칸 밀어 같은 정의
    edelay = {}
    for k in IDX:
        rid = esel[k]["id"]
        if rid:
            name, H = rid.rsplit(" H=", 1)
            rule = next(r for r in rules_e() if r["id"] == name)
            xs = ix[k].shift(1).dropna()
            edelay[k] = {w: bk.stats(events(xs, rule), int(H), *WLONG[w]) for w in WLONG}
    # 대표 사건 정의를 세 지수에 같은 정의로(나란히) + 2026 사건 날짜
    eside = {}
    for k in IDX:
        rid = esel[k]["id"]
        if rid:
            eside[rid] = {kk: {w: {a: b for a, b in ev[kk][rid][w].items() if a != "날짜"} for w in EW} for kk in IDX}
    ev26 = {k: {r["id"]: [d for d in events(ix[k], r) if d >= "2026-01-01"] for r in rules_e()} for k in IDX}
    # 2026 사건형 앞 고르기 → 뒤 확인(사건 5건 이상 조건)
    esel26 = {k: event_select(ev[k], "2026앞", "2026뒤") for k in IDX}

    # 사후 관찰(선택편향 있음): 대상·지수마다 2026 전체 효율선초과 최대 규칙 + 쏠림판 S2 w1=0.5 G=75 복귀=60
    post = {}
    named = "S2 w1=0.5 G=75복귀=60"
    for target, tg in (("7종목", tg7["2026"]), ("SOXL", tgs)):
        for k in IDX:
            rows = res[target][k]
            best = max(rows, key=lambda i: rows[i]["2026전체"]["효율선초과"])
            for rid in dict.fromkeys([best, named]):
                rule, x = rmap[rid], ix[k]
                d = {"규칙": rid, "사후최고": rid == best,
                     "하루늦춤": {w: run_rule(tg, x, rule, lag=1)[w] for w in ("2026앞", "2026뒤", "2026전체")},
                     "위약_전체": placebo_rank(tg, x, rule, "2026전체"), "7월": trace(tg, x, rule)}
                if target == "7종목":
                    d["하루늦춤"].update({w: run_rule(tg7["hold"], x, rule, lag=1)[w] for w in ("2025", "2024")})
                    d["위약_2025"] = placebo_rank(tg7["hold"], x, rule, "2025")
                else:
                    d["하루늦춤"].update({w: run_rule(tg, x, rule, lag=1)[w] for w in ("2025", "긴확인")})
                    d["위약_긴확인"] = placebo_rank(tg, x, rule, "긴확인")
                post[f"{target}|{k}|{rid}"] = d
    # 7월 요소 경로(설명용)
    jul = pd.concat([pc.round(1), pd.concat([ix["AI"], ix["AI쏠림"], ix["CNN"]], axis=1).round(1)], axis=1)
    jul = jul[(jul.index >= "2026-06-15") & (jul.index <= "2026-08-05")]
    comp["7월요소"] = {d: {c: (None if pd.isna(v) else float(v)) for c, v in r.items()} for d, r in jul.iterrows()}

    traces = {}
    rep_id = sel["7종목"]["AI쏠림"]["id"]
    for k in IDX:
        traces[f"7종목|{k}|{rep_id}"] = trace(tg7["2026"], ix[k], rmap[rep_id])
    payload = {"family": "AIFG", "grid": grid_size(), "bad": bad, "compare": comp, "base7": base7, "baseSOXL": baseS,
               "sel": sel, "cands": cands, "sens": sens, "res": res, "traces": traces,
               "events": ev, "events_cnn_long": ev_cnn_long, "esel": esel, "eplac": eplac, "edelay": edelay,
               "eside": eside, "ev26": ev26, "esel26": esel26, "post": post}
    RESULTS.mkdir(exist_ok=True)
    txt = json.dumps(payload, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    (RESULTS / f"{ID}.json").write_text(txt, encoding="utf-8")
    ix_df = pd.concat([ix["AI"], ix["AI쏠림"], ix["CNN"]], axis=1)
    ix_df.to_parquet(CACHE / f"{ID}_index.parquet")
    pd.concat([raw.add_prefix("raw_"), pc.add_prefix("pct_")], axis=1).to_parquet(CACHE / f"{ID}_components.parquet")
    print("저장", RESULTS / f"{ID}.json")
    write_md(json.loads(txt))


# ============================================================================ 보고서
def _p(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    t = f"{x * 100:+,.{d}f}%" if sign else f"{x * 100:,.{d}f}%"
    return t.replace("-", "−")


def _pp(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%p".replace("-", "−")


def _cell(st, eff=True):
    if not st:
        return "—"
    t = f"{_p(st['수익'])}/{_p(-abs(st['낙폭']), 0)}"
    return t + (f" ({_pp(st['효율선초과'])})" if eff else "")


def _ev(st):
    if not st or not st.get("사건"):
        return "사건 0"
    return f"{_p(st['초과'])} ({st['사건']}건, 승률 {st['승률'] * 100:.0f}%)"


def _ko_rule(rid: str) -> str:
    """규칙 id → 사람이 읽는 말. 예: S1 w0=0.6 T=25 해제=40 → '평소 60%, ≤25면 100%, ≥40이면 60%'."""
    if rid.startswith("S1"):
        a = dict(kv.split("=") for kv in rid.split()[1:])
        w = f"{float(a['w0']) * 100:.0f}%"
        return f"평소 {w}, ≤{a['T']}이면 100%, ≥{a['해제']}이면 {w}"
    if rid.startswith("S2"):
        a = dict(kv.split("=") for kv in rid.replace("복귀", " 복귀").split()[1:])
        return f"평소 100%, ≥{a['G']}이면 {float(a['w1']) * 100:.0f}%, ≤{a['복귀']}이면 100%"
    n, H = rid.split(" H=") if " H=" in rid else (rid, None)
    tail = f", {H}일 보유" if H else ""
    if rid.startswith("E1"):
        return f"≤{n.split('T=')[1]} 첫날 매수{tail}"
    if rid.startswith("E2"):
        a = dict(kv.split("=") for kv in n.split()[1:])
        return f"≤{a['T']}에서 {int(a['T']) + int(a['δ'])} 위로 올라선 날 매수{tail}"
    return rid


def write_md(P: dict):
    res, sel, sens, post, base7, baseS = P["res"], P["sel"], P["sens"], P["post"], P["base7"], P["baseSOXL"]
    comp, ev, esel, eplac, edelay, ecl = P["compare"], P["events"], P["esel"], P["eplac"], P["edelay"], P["events_cnn_long"]
    g = P["grid"]
    L = []
    rep_t, rep_k = "7종목", "AI쏠림"
    rep = sel[rep_t][rep_k]["id"]
    R, C, A7 = res[rep_t][rep_k][rep], res[rep_t]["CNN"][rep], res[rep_t]["AI"][rep]
    S2 = "S2 w1=0.5 G=75복귀=60"
    Q, QC = res["7종목"]["AI쏠림"][S2], res["7종목"]["CNN"][S2]
    pq = post[f"7종목|AI쏠림|{S2}"]
    sx = sens[f"{rep_t}|{rep_k}"]
    ep = comp["사건비교"]
    fe = ep["2026-06~07 AI쇼크"]

    def npos(t, k, w):
        return sum(res[t][k][i][w]["효율선초과"] > 1e-9 for i in res[t][k])

    L.append("# 8단계 AIFG: 반도체·AI 전용 공포탐욕 지수 대 CNN F&G (2026-10-11)\n")
    L.append("> 질문: CNN F&G는 S&P 기준이라 7월 반도체 쇼크를 늦게·약하게 잡을 수 있다. 반도체·AI 재료로 만든 지수가 7종목·SOXL 비중 조절과 AI 종목 매수 시점에서 CNN판보다 나은가.  ")
    L.append("> 지수(사전 고정): A1 ^SOX/125일선, A2 AI 유니버스 50일선 위 비율, A3 52주 신고가−신저가 순비율, A4 SOX 20일 변동성/50일 평균(뒤집음), A5 SOXL/SOXS 거래대금 5일 평균, "
             "A6 SMH−TLT 20일, A7 HYG−IEF 20일 → 각각 과거 252일 백분위, 같은 가중 = **AI(7요소)**. 대안 **쏠림판** = A2·A3·A5.  ")
    L.append(f"> 코드 `code/p8_AIFG.py`(자료 받기 `py p8_AIFG.py fetch`), 결과 `results/p8_AIFG.json`, 지수 `cache/p8_AIFG_index.parquet`. "
             f"시험 {g['합계']}칸 = 전략형 {g['전략형_규칙']}개 × 지수 3 × 대상 2(7종목·SOXL) + 사건형 {g['사건형_칸']}칸 × 지수 3.\n")

    L.append("## 결론\n")
    L.append(f"1. **AI 지수는 '업종 과열'은 CNN보다 잘 보였지만, 7월 쇼크의 '공포'는 더 늦거나 못 봤다.** 6/22 반도체 고점 날 쏠림판은 {fe['AI쏠림']['고점날값']}(극단 탐욕), "
             f"CNN은 {fe['CNN']['고점날값']}(공포)이었다. 그런데 쏠림판은 다음 날 50으로 내려가 '덜기'가 이틀 만에 풀렸다. 공포(≤25)는 7/13에야 왔고(SOX 이미 {_p(fe['AI쏠림']['그때SOX낙폭'], 0)}), "
             f"7요소판은 끝내 25 아래로 안 갔다(최저 {fe['AI']['최저']}, 7/29). 2024-08·2025-01·2026-03에는 AI판이 CNN보다 1~20거래일 먼저 공포였다.")
    L.append(f"2. **사전 고르기(2026 앞 → 뒤)를 통과한 것은 7종목 쏠림판 '{_ko_rule(rep)}' 하나다.** 뒤 효율선초과 {_pp(R['2026뒤']['효율선초과'])}로 같은 규칙 CNN판({_pp(C['2026뒤']['효율선초과'])})보다 낫다. "
             f"하지만 2026 전체로는 같은 낙폭 고정 비중보다 {_pp(R['2026전체']['효율선초과'])}이고, 위약 순위가 뒤 {sx['위약_뒤']['순위']}/21·전체 {sx['위약_전체']['순위']}/21이라 우연과 구별되지 않는다. "
             f"SOXL은 2026 뒤에서 세 지수 모두 48개 중 {npos('SOXL', 'AI', '2026뒤')}·{npos('SOXL', 'AI쏠림', '2026뒤')}·{npos('SOXL', 'CNN', '2026뒤')}개만 플러스였다. "
             f"긴 역사(2014~2019 → 2020~2025) 확인은 AI·쏠림 0개, CNN {npos('SOXL', 'CNN', '긴확인')}개다.")
    L.append(f"3. **판정: 기각(매매 규칙으로).** 사후에 눈에 띈 쏠림판 '{_ko_rule(S2)}'은 7종목 2025 {_p(Q['2025']['수익'])}(기본 {_p(base7['2025']['수익'])}), "
             f"2026 전체 효율선 {_pp(Q['2026전체']['효율선초과'])}였다. 그러나 2026 앞 구간에서 져서 사전 고르기에 안 뽑혔고, 하루 늦추면 2026 전체가 {_pp(pq['하루늦춤']['2026전체']['효율선초과'])}로 무너진다. "
             f"사건형 최상(7요소판 ≤5 첫날 → 126일)은 확인 {_p(ev['AI'][esel['AI']['id']]['긴확인']['초과'])}였지만 사건 3건(2020-02·2024-08·2025-04)뿐이고, 2026에는 한 번도 켜지지 않았다.\n")

    # ---------------- 사용자 표
    L.append("## 사용자 표\n")
    L.append("칸 = 구간 수익/최대 낙폭 (효율선초과: 같은 최대 낙폭이 나오는 고정 비중 대비. 2026·쇼크는 구간 %p, 2025·긴 역사는 연 %p). 다음 거래일 시가 체결, 수수료·현금 이자 0.\n")
    L.append("| 대상 | 규칙(지수) | 2026 앞(1/2~5/29) | 2026 뒤(6/1~10/9) | 7월 AI 쇼크 | 2025 | 긴 역사 고르기 / 확인 | 연 매매 수 |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append(f"| 7종목 | 그대로 | {_cell(base7['2026앞'], False)} | {_cell(base7['2026뒤'], False)} | {_p(base7['AI쇼크']['수익'])} | {_cell(base7['2025'], False)} | — | 0 |")
    for label, k, r in (("**대표** " + _ko_rule(rep), "쏠림", R), ("같은 규칙", "CNN", C), ("같은 규칙", "AI 7요소", A7),
                        ("사후 관찰 " + _ko_rule(S2), "쏠림", Q), ("같은 규칙", "CNN", QC)):
        L.append(f"| 7종목 | {label} ({k}) | {_cell(r['2026앞'])} | {_cell(r['2026뒤'])} | {_p(r['AI쇼크']['수익'])} | {_cell(r['2025'])} | — | {r['2026전체']['매매수_연']:.0f} |")
    L.append(f"| SOXL | 그대로 | {_cell(baseS['2026앞'], False)} | {_cell(baseS['2026뒤'], False)} | {_p(baseS['AI쇼크']['수익'])} | {_cell(baseS['2025'], False)} | "
             f"{_cell(baseS['긴고르기'], False)} / {_cell(baseS['긴확인'], False)} | 0 |")
    for t, why in (("SOXL", "2026 앞으로 고름"), ("SOXL긴", "긴 역사로 고름")):
        for k in IDX:
            rid = sel[t][k]["id"]
            r = res["SOXL"][k][rid]
            L.append(f"| SOXL | {why}: {_ko_rule(rid)} ({k}) | {_cell(r['2026앞'])} | {_cell(r['2026뒤'])} | {_p(r['AI쇼크']['수익'])} | {_cell(r['2025'])} | "
                     f"{_cell(r['긴고르기'])} / {_cell(r['긴확인'])} | {r['긴확인']['매매수_연']:.0f} |")
    L.append("\n사건형(AI 유니버스 동일가중 바구니, 같은 종목 '아무 날 매수' 대비 초과, 긴 역사 2014~2019 고르기 → 2020~2025 확인)\n")
    L.append("| 규칙(지수) | 2026 사건 수 | 2025 | 긴 역사 고르기 | 긴 역사 확인 | 하루 늦춤 확인 | 위약 순위 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in IDX:
        rid = esel[k]["id"]
        for kk in IDX:
            st = P["eside"][rid][kk]
            b = "**" if kk == k else ""
            d26 = len(P["ev26"][kk][rid.rsplit(" H=", 1)[0]])
            L.append(f"| {b}{_ko_rule(rid)} ({kk}){b} | {d26}건 | {_ev(st['2025'])} | {_ev(st['긴고르기'])} | {_ev(st['긴확인'])} | "
                     f"{_ev(edelay[k]['긴확인']) if kk == k else ''} | {str(eplac[k]['순위']) + '/21' if kk == k else ''} |")
    cl = esel["CNN(2011~2018→2019~2025)"]
    st = ecl[cl["id"]]
    L.append(f"| CNN 규약 구간(2011~2018 → 2019~2025): {_ko_rule(cl['id'])} | | | {_ev(st['CNN긴고르기'])} | {_ev(st['CNN긴확인'])} | | |")
    L.append("\n굵게 = 그 지수로 고른 정의. 같은 정의를 다른 두 지수에 그대로 돌린 판을 붙였다. 2026 사건은 결과 기간(H일)이 끝나지 않아 수익을 매기지 않았다.\n")

    # ---------------- 지수 비교
    L.append("## AI 지수와 CNN: 언제 공포·탐욕을 보였나\n")
    c = comp
    dd = c["분포"]
    L.append(f"- 상관(2013~2026 일별 수준): AI–CNN {c['상관_수준']['AI-CNN']:.2f}, 쏠림판–CNN {c['상관_수준']['쏠림-CNN']:.2f}, AI–쏠림 {c['상관_수준']['AI-쏠림']:.2f}. "
             f"20일 변화 상관 AI–CNN {c['상관_20일변화']['AI-CNN']:.2f}. 2026은 {c['해별상관_AI-CNN']['2026']:.2f}로 해마다 가운데 가장 낮다(업종과 시장이 갈린 해).")
    L.append(f"- 2014~ 극단 일수: ≤10 AI {dd['AI']['≤10']}·쏠림 {dd['AI쏠림']['≤10']}·CNN {dd['CNN']['≤10']}일, ≥75 AI {dd['AI']['≥75']}·쏠림 {dd['AI쏠림']['≥75']}·CNN {dd['CNN']['≥75']}일. "
             f"쏠림판은 3요소 평균이라 더 출렁인다(표준편차 {dd['AI쏠림']['표준편차']:.0f}, CNN {dd['CNN']['표준편차']:.0f}).")
    e10 = c["2026-10-09"]
    L.append(f"- 2026-10-09: AI {e10['AI']}, 쏠림판 {e10['AI쏠림']}, CNN {e10['CNN']}. 요소 백분위 {', '.join(f'{k} {v:.0f}' for k, v in c['요소_10-09'].items())}. "
             f"SOX/125일선이 1년 중 하위 {c['요소_10-09']['A1']:.0f}%인 것은 큰 상승 뒤라 평범한 수준도 낮게 읽히기 때문이다.\n")
    L.append("| 사건 | SOX 고점 → 저점(낙폭) | 첫 공포 ≤25(그때 SOX 낙폭): AI / 쏠림 / CNN | 고점 날 값: AI / 쏠림 / CNN | 고점 전 마지막 ≥75: AI / 쏠림 / CNN | 최저(날짜): AI / 쏠림 / CNN |")
    L.append("|---|---|---|---|---|---|")
    for name, v in ep.items():
        f1 = " / ".join((v[k]["첫공포≤25(고점뒤)"] or "없음") + (f"({_p(v[k]['그때SOX낙폭'], 0)})" if v[k]["그때SOX낙폭"] is not None else "") for k in IDX)
        lo = " / ".join(f"{v[k]['최저']}({v[k]['최저일'][5:]})" for k in IDX)
        L.append(f"| {name} | {v['SOX고점']} → {v['SOX저점']} ({_p(v['SOX낙폭'], 0)}) | {f1} | {' / '.join(str(v[k]['고점날값']) for k in IDX)} | "
                 f"{' / '.join(str(v[k]['고점전마지막탐욕≥75'] or '없음') for k in IDX)} | {lo} |")
    d29 = c["7월요소"].get("2026-07-29", {})
    L.append(f"\n- 7요소판이 7월에 덜 떨어진 이유(7/29 SOX 저점 날 백분위): 모멘텀 A1 {d29.get('A1', 0):.0f}, 50일선 위 A2 {d29.get('A2', 0):.0f}, 신고가−신저가 A3 {d29.get('A3', 0):.0f}, "
             f"SMH−TLT A6 {d29.get('A6', 0):.0f}는 극단 공포였다. 반면 변동성 A4 {d29.get('A4', 0):.0f}(서서히 빠져 20일 변동성이 평소보다 낮음), 신용 A7 {d29.get('A7', 0):.0f}(거시 사건 아님), "
             f"SOXL/SOXS A5 {d29.get('A5', 0):.0f}(개인이 SOXL을 계속 삼)는 공포를 보이지 않았다. 거시·변동성 재료가 업종 안 쇼크에서 꺼져 있었다는 1단계 결론과 같다.\n")

    # ---------------- 조합 격자
    L.append("## 조합 격자 표\n")
    L.append("전략형 48개(S1 공포 추가: 평소 w0 {60, 70, 80%} × 문턱 T {10, 20, 25, 30} × 해제 {40, 50, 60} = 36, S2 탐욕 덜기: w1 {50, 75%} × G {75, 80, 85} × 복귀 {50, 60} = 12)를 "
             "세 지수 × 두 대상에 그대로. 칸 = 효율선초과 > 0인 규칙 수(48개 중).\n")
    L.append("| 대상 | 지수 | 2026 앞 > 0 | 그중 뒤 > 0 | 뒤 > 0(전체) | 2026 전체 > 0 | 2025 > 0 | 2024 > 0 | 긴 고르기 > 0 | 긴 확인 > 0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in ("7종목", "SOXL"):
        for k in IDX:
            rows = res[t][k]
            front = [i for i in rows if rows[i]["2026앞"]["효율선초과"] > 1e-9]
            both = sum(rows[i]["2026뒤"]["효율선초과"] > 1e-9 for i in front)
            has = next(iter(rows.values()))

            def f(w):
                return str(npos(t, k, w)) if w in has else "—"
            L.append(f"| {t} | {k} | {len(front)} | {both} | {f('2026뒤')} | {f('2026전체')} | {f('2025')} | {f('2024')} | {f('긴고르기')} | {f('긴확인')} |")
    L.append("\n7종목 S2(탐욕 덜기) 12개: 쏠림판 대 CNN (효율선초과). CNN은 2026에 71을 넘은 적이 없어 2026에는 아무것도 하지 않았다.\n")
    L.append("| 규칙 | 쏠림 2026 앞 | 쏠림 2026 뒤 | 쏠림 2026 전체 | 쏠림 2025 | 쏠림 2024 | CNN 2025 | CNN 2024 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rid in [i for i in res["7종목"]["AI쏠림"] if i.startswith("S2")]:
        a, b = res["7종목"]["AI쏠림"][rid], res["7종목"]["CNN"][rid]
        L.append(f"| {_ko_rule(rid)} | {_pp(a['2026앞']['효율선초과'])} | {_pp(a['2026뒤']['효율선초과'])} | {_pp(a['2026전체']['효율선초과'])} | {_pp(a['2025']['효율선초과'])} | "
                 f"{_pp(a['2024']['효율선초과'])} | {_pp(b['2025']['효율선초과'])} | {_pp(b['2024']['효율선초과'])} |")
    L.append("\n사건형 54칸(E1 ≤T 첫날 T 6개 + E2 ≤T에서 T+δ 위로 T 4 × δ 3 = 18정의 × 보유 21·63·126일), 긴 역사\n")
    L.append("| 지수 | 고르기 후보(사건 ≥5) | 고르기 > 0 | 그중 확인 > 0 | 고른 칸 |")
    L.append("|---|---|---|---|---|")
    for k in IDX:
        i = esel[k].get("정보", {})
        L.append(f"| {k} | {i.get('후보', '—')} | {i.get('고르기>0', '—')} | {_p(i.get('확인>0|고르기>0'), 0, False)} | {_ko_rule(esel[k]['id'])} |")

    # ---------------- 고르기 표
    L.append("\n## 고르기 표\n")
    L.append("사전 고정: 고르기 구간 효율선초과 최대 → 확인 구간 효율선초과 > 0이고 같은 평균 노출 고정 대비 > 0이면 통과. 사건형은 고르기 초과 최대(사건 5건 이상) → 확인 초과 > 0.\n")
    L.append("| 판정 | 대상 | 지수 | 고른 규칙 | 고르기 | 확인 효율선 / 고정 대비 | 통과 | 같은 규칙 CNN 확인 | 위약 순위(확인) | 하루 늦춤 확인 | 미끄러짐 0.5% 확인 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, sw, cw, lab in (("7종목", "2026앞", "2026뒤", "2026"), ("SOXL", "2026앞", "2026뒤", "2026"), ("SOXL긴", "긴고르기", "긴확인", "긴 역사")):
        tt = "SOXL" if t.startswith("SOXL") else t
        for k in IDX:
            s_ = sel[t][k]
            r, cn, sv = res[tt][k][s_["id"]], res[tt]["CNN"][s_["id"]], sens[f"{t}|{k}"]
            plc = sv.get("위약_뒤" if lab == "2026" else "위약_긴확인")
            L.append(f"| {lab} | {tt} | {k} | {_ko_rule(s_['id'])} | {_pp(r[sw]['효율선초과'])} | {_pp(r[cw]['효율선초과'])} / {_pp(r[cw]['고정초과'])} | "
                     f"{'통과' if s_['통과'] else '탈락'} | {_pp(cn[cw]['효율선초과'])} | {plc['순위']}/21 | {_pp(sv['하루늦춤'][cw]['효율선초과'])} | "
                     f"{_pp(sv['미끄러짐0.5%'][cw]['효율선초과'])} |")
    for k in IDX:
        rid = esel[k]["id"]
        st = ev[k][rid]
        L.append(f"| 긴 역사 | AI 종목 사건형 | {k} | {_ko_rule(rid)} | {_p(st['긴고르기']['초과'])}({st['긴고르기']['사건']}건) | {_p(st['긴확인']['초과'])}({st['긴확인']['사건']}건) | "
                 f"{'통과' if esel[k]['통과'] else '탈락'} | {_p(P['eside'][rid]['CNN']['긴확인']['초과'])} | {eplac[k]['순위']}/21 | {_p(edelay[k]['긴확인']['초과'])} | — |")
    L.append("| 2026 | AI 종목 사건형 | 세 지수 | 2026 앞에 결과가 끝난 사건 5건 이상인 정의 없음 → 판정 불가 | | | | | | | |")
    L.append(f"\n- 대표 후보 조건(2026 앞 → 뒤 통과 + 같은 규칙 CNN판보다 뒤·2026 전체 효율선초과가 큼)을 채운 것은 7종목 쏠림판 '{_ko_rule(rep)}' 하나다. "
             f"하루 늦춤 뒤 {_pp(sx['하루늦춤']['2026뒤']['효율선초과'])}, 미끄러짐 0.5% 뒤 {_pp(sx['미끄러짐0.5%']['2026뒤']['효율선초과'])}, 이자 4% 뒤 {_pp(sx['이자4%']['2026뒤']['효율선초과'])}. "
             f"평균 노출 {R['2026전체']['노출']:.2f}의 '덜 담기'에 공포 때 채우기를 더한 규칙이라 2026 전체로는 같은 낙폭 고정 비중보다 {_pp(R['2026전체']['효율선초과'])}다.")
    a7s = sens["7종목|AI"]
    L.append(f"- 같은 규칙의 7요소판은 2026 전체 효율선 {_pp(A7['2026전체']['효율선초과'])}·위약 {a7s['위약_전체']['순위']}/21로 가장 좋았지만, 이익이 3월 한 번(3/6 종가 ≤25 → 3/9 시가 100%)에서 나왔고 "
             f"뒤 구간에는 신호가 없어(늘 60%) 확인할 수 없다. 하루 늦추면 앞 구간 {_pp(a7s['하루늦춤']['2026앞']['효율선초과'])}, 2025는 {_pp(A7['2025']['효율선초과'])}다.")
    L.append(f"- 사후 관찰 쏠림판 '{_ko_rule(S2)}': 위약 순위 2026 전체 {pq['위약_전체']['순위']}/21·2025 {pq['위약_2025']['순위']}/21, "
             f"하루 늦춤 2026 전체 {_pp(pq['하루늦춤']['2026전체']['효율선초과'])}·2025 {_pp(pq['하루늦춤']['2025']['효율선초과'])}·2024 {_pp(pq['하루늦춤']['2024']['효율선초과'])}. "
             "2025·2024에는 버텼지만 2026에는 하루 차이로 무너진다(6/22 고점 날 덜고 6/24 되사는 식의 짧은 왕복이 날짜에 민감). 48개 × 3지수를 다 본 뒤 고른 것이라 선택편향이 크다.")
    i7, iq = sel["7종목"]["AI"]["정보"], sel["7종목"]["AI쏠림"]["정보"]
    L.append(f"- 고르기에 정보가 있었나(앞 > 0 중 뒤 > 0): 7종목 AI {i7['확인>0|고르기>0'] * 100:.0f}%(무조건 {i7['확인>0 무조건'] * 100:.0f}%), 쏠림 {iq['확인>0|고르기>0'] * 100:.0f}%(무조건 {iq['확인>0 무조건'] * 100:.0f}%). "
             "CNN은 앞 > 0이 0개. SOXL은 세 지수 모두 0~6%. 쏠림판의 100%는 뒤 구간에서 쏠림판 S1이 대부분 플러스였기 때문이라 고르기의 힘이라고 보기 어렵다.\n")

    # ---------------- 7월
    L.append("## 7월 AI 쇼크(6/22~7/29)에서 규칙이 한 일\n")
    L.append("| 규칙 | 노출이 바뀐 날(시가) ← 전날 종가 지수 | 쇼크 구간 수익 |")
    L.append("|---|---|---|")
    s_s, s_c = sel["SOXL"]["AI쏠림"]["id"], sel["SOXL"]["CNN"]["id"]
    rows7 = [("7종목 대표 " + _ko_rule(rep) + " (쏠림)", sx["7월"], R["AI쇼크"]["수익"]),
             ("7종목 같은 규칙 (CNN)", P["traces"][f"7종목|CNN|{rep}"], C["AI쇼크"]["수익"]),
             ("7종목 같은 규칙 (AI 7요소)", P["traces"][f"7종목|AI|{rep}"], A7["AI쇼크"]["수익"]),
             ("7종목 사후 " + _ko_rule(S2) + " (쏠림)", pq["7월"], Q["AI쇼크"]["수익"]),
             ("SOXL 2026 앞으로 고름 " + _ko_rule(s_s) + " (쏠림)", sens["SOXL|AI쏠림"]["7월"], res["SOXL"]["AI쏠림"][s_s]["AI쇼크"]["수익"]),
             ("SOXL 2026 앞으로 고름 " + _ko_rule(s_c) + " (CNN)", sens["SOXL|CNN"]["7월"], res["SOXL"]["CNN"][s_c]["AI쇼크"]["수익"])]
    for lab, tr, rr in rows7:
        t = ", ".join(f"{x['날짜(시가)'][5:]} {x['노출'] * 100:.0f}%←{x['전날지수']}" for x in (tr or []))
        L.append(f"| {lab} | {t or '변화 없음'} | {_p(rr)} |")
    L.append(f"\n- 비교: 7종목 그대로 {_p(base7['AI쇼크']['수익'])}, SOXL 그대로 {_p(baseS['AI쇼크']['수익'])}. 6/1 칸은 5월 말 상태가 이어진 값이다(6/1~8/31 경로).")
    L.append("- 쏠림판 탐욕 덜기는 6/22(고점 날) 시가에 반으로 줄였다가 6/24 시가에 다시 채웠다(6/23 종가 50.1 ≤ 60). 그 뒤 하락을 그대로 맞았다. "
             "공포 추가(S1) 쏠림판은 7/14 시가에 100%로 올렸는데 SOX 저점은 7/29였다. 덜 맞은 것은 평소 60%라서다. "
             "같은 규칙 CNN판은 6/29에 100%로 올렸다가 7/7에 60%로 돌아와 본 하락(7/7~7/29)을 60%로 맞았다(−12.9%). 7요소판은 신호 없이 60%였다.\n")

    # ---------------- 한계
    L.append("## 한계\n")
    L.append("- 생존편향: AI 유니버스는 2026-10 목록(미국 115종목, 자료 오류 의심 NVTS 제외). 2014~2019 시장폭·사건형 수익은 지금 살아남은 AI 승자로 잰 값이라 실제보다 좋게 나온다. 같은 종목 '아무 날'과 비교해 일부만 상쇄된다.")
    L.append("- A5(SOXL/SOXS 거래대금)는 yfinance SOXS 거래량이 역분할 수정 뒤 정수 반올림으로 2020년 전 0~몇 주라 2020-01-24부터만(백분위 2021-01부터) 썼다. "
             "그 전 7요소판은 6요소, 쏠림판은 A2·A3 두 요소다(사전에 정한 '있는 요소만 같은 가중'). 긴 역사 고르기 구간의 지수는 확인 구간과 구성이 다르다.")
    L.append(f"- 선택편향: 시험 {g['합계']}칸. 2026 앞·뒤는 각각 약 100·90거래일이고 공포 사건은 3월·7월 두 번뿐이다. 사건형 극단(≤5) 사건은 긴 역사 고르기 6건·확인 3건이다.")
    L.append("- SOXL 2026 뒤(−32%)처럼 손실 구간에서는 효율선이 현금(노출 0)이 되어 '효율선초과 > 0' = 절대 수익 플러스라는 높은 기준이 된다. 같은 규칙 CNN판·고정 대비를 함께 봤다.")
    L.append("- 정규화는 CNN식 252일 백분위로 고정했다(가중·창 길이를 고르지 않음). 큰 상승 뒤에는 평범한 되돌림도 낮은 백분위로 읽힌다(10/9 A1). 2011~2020 CNN 값은 개편 때 거꾸로 채운 값일 수 있다.")
    L.append("- 다른 해: 7요소판 S1은 2025에 같은 낙폭 고정 비중보다 대부분 낮았다(관세처럼 몇 달 끄는 하락에서 공포 추가가 일찍 들어감). 장세가 바뀌면 같은 문턱이 반대로 작동할 수 있다.")
    L.append("- 시점: 모든 재료는 그날 종가로 확정 → 다음 거래일 시가(harness.at_open). 하루 늦춘 판을 고르기 표에 붙였다. 매수 추천·주문 수량이 아니다.")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("저장", RESULTS / f"{ID}.md")


ETF = ["SOXL", "SOXS", "SMH", "TLT", "HYG", "IEF", "SOXX"]


def fetch():
    """cache/p8_AIFG_px.parquet: AI 유니버스 미국 115종목 + 재료 ETF, yfinance 2011-01-01~2026-10-09 일봉(분할 수정, 배당 미수정)."""
    import time
    import yfinance as yf
    syms = universe() + ETF
    frames = []
    for i in range(0, len(syms), 40):
        chunk = syms[i:i + 40]
        for _ in range(3):
            try:
                f = yf.download(chunk, start="2011-01-01", end="2026-10-10", auto_adjust=False, progress=False,
                                threads=True, group_by="column")
                break
            except Exception as e:  # noqa: BLE001
                print("재시도", e)
                time.sleep(5)
        cols = [f[c].stack(future_stack=True).rename(c) for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
        df = pd.concat(cols, axis=1).reset_index()
        df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        frames.append(df.dropna(subset=["Close"]))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_parquet(CACHE / f"{ID}_px.parquet")
    print("받은 종목", df.sym.nunique(), "/", len(syms), "| 빠진 종목", sorted(set(syms) - set(df.sym)))


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        main()
