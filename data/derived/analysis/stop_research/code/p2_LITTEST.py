"""2단계 갈래 LIT-TEST(2026-10-10): 문헌에서 나왔지만 아직 시험하지 않은 손절·재진입 규칙 C1~C8 검증.

후보 출처는 results/p2_LIT.md('아직 시험하지 않은 후보'). 모수는 문헌 값 그대로이고, 이웃 값은 후보당 2개까지만 사전에 적었다.
평가는 2단계 공통 규약(stock_lab): AI 종목 전체를 창 시작 직전 종가에 들고(C4만 현금 시작), 규칙을 얹은 동일가중 포트폴리오와
손절 주기(판 가격/다시 산 가격 − 1)로 잰다. C8과 Han-Zhou-Zhu 7종목판은 데스크 7종목 경로(harness.StockEngine) 위에서 잰다.

시점(stock_lab 규약 그대로)
  - 종가 신호는 t-1 종가까지로 계산 → t일 시가 체결. 장중 손절가는 t-1까지 정보(+그날 체결한 매수가)로 정해 t일 스탑으로 건다.
  - 매수 스탑(C5 완전판의 피벗 돌파, C6 r2의 20일 고가 돌파)은 t-1까지의 값에 t일 장중 체결(시가가 위면 시가).
    stock_lab.simulate에는 매수 스탑 훅이 없어 sim_ext가 같은 코드에 그 단계 하나만 더했다(p2_LINE과 같은 방식, check()로 동치 확인).
  - C2 월 기준가 P0 = 그 달 첫 거래일 시가(손절 주문은 시가 확인 뒤 건다; 이웃 'prevclose'는 전월 말 종가라 밤사이 걸 수 있음).
  - C4 β는 252일 고점일 h 이전 252일 일간 수익 회귀(h는 t-1까지의 252일 최고 종가일). 실적일은 이미 지난 날짜만 쓴다.
  - C7 공황 = t-1 기준 QQQ 504거래일 수익 < 0 그리고 126일 실현변동성 > 그 앞 1,260일 중앙값.
고르기(사전 고정): G1 앞 효율선초과 최대 / G2 앞 주기 기대값 최대(주기 30개 이상) / G3 앞 칼마 최대. 격자 38개 안에서 고른다.
실행: PYTHONIOENCODING=utf-8 py p2_LITTEST.py fetch   (QQQ 1999~ 일봉·실적일을 yfinance로 받아 cache/p2_LITTEST_*에 저장, 한 번)
      PYTHONIOENCODING=utf-8 py p2_LITTEST.py check   (엔진 동치·미래 교란·LINE 재현 점검)
      PYTHONIOENCODING=utf-8 py p2_LITTEST.py         (전체) / md (보고서만 다시)
"""
from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stock_lab as SL  # noqa: E402
from stock_lab import Policy  # noqa: E402

ID = "p2_LITTEST"
CACHE = HERE.parent / "cache"
QQQ_LONG = CACHE / "p2_LITTEST_qqq.parquet"
EARN = CACHE / "p2_LITTEST_earn.json"

# ============================================================================ 사전 고정 격자(뒤 구간·2025를 본 뒤 바꾸지 않음)
C1_MAIN = [(60, -1.0), (60, -0.5), (120, -1.0), (120, -0.5)]      # Kaminski-Lo: (J, γ)
C1_NB = [(60, -1.5), (250, -1.0)]                                    # 이웃 2(문헌 범위 γ −1.5, 창 10개월)
C1_SD_N, C1_RE_N = 252, 20                                          # J일 수익 표준편차 창, 재진입 = 직전 20일 누적수익 > 0
C2_MAIN = [0.10, 0.15, 0.20]                                         # Han-Zhou-Zhu L
C2_NB = [("all", 0.10), ("prevclose", 0.10)]                         # 이웃 2: 모멘텀 조건 없이 다음 달 재매수 / P0 = 전월 말 종가(원문)
C2_MOM_N, C2_TOP = 120, 0.10                                        # 120일 수익 상위 10%(원문 승자 10분위)
BASE_TRAIL = 0.20                                                    # C3·C5 손절 = 추적 20%(Dai 외 2021 비용 뒤 최선)
C3_K = [0.85, 0.90, 0.95]                                            # 52주 고점 근접 재진입
C4_C = [0.10, 0.15, 0.20]
C4_ETF = ["SOXX", "QQQ"]
C4_DD, C4_HOLD, C4_BETA_N, C4_BETA_MIN, C4_EARN_W = 0.40, 60, 252, 200, 1
C5_V = [0.6, 0.7, 0.8]
C5_S, C5_L, C5_SMA = 10, 50, 10
VCP_LOOK, VCP_SW, VCP_DEC, VCP_LAST = 60, 5, 2, 0.10                # 직전 60일, 5일 국소 고·저, 2회 연속 감소, 마지막 ≤ 10%
C6_M = [0.5, 1.0]
C6_RE = ["r1", "r2"]
C6_ATR, C6_K, BRK_N = 20, 2.0, 20
C7_BASES = [("sma", 200), ("trail", 0.20)]                           # SMA200(Clare 외·Faber). 추적 20%는 이웃(켬·끔 2개)
PANIC_RET_N, PANIC_VOL_N, PANIC_MED_N = 504, 126, 1260
C8_ATR = 20                                                          # 7종목 위험 균등 비중
HZZ7_L = 0.10                                                        # Han-Zhou-Zhu 7종목판
SLIP_SENS = 0.005
G2_MIN_CYCLES = 30
CONTEXT_YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
RUNS = {k: SL.RUNS[k] for k in ["2026", "2025"] + CONTEXT_YEARS}
AI_SHOCK = ("2026-06-22", "2026-07-29")
# 자료 이상: QUBT는 2018~19 장외 껍데기 시절 하루 −99.5%·+2,033%(2018-07-03·06) 같은 값이 있어 동일가중 포트를 혼자 +22.6% 올린다.
# 2018·2019 실행에서만 뺀다(첫 실행 뒤 발견, 격자·정의와 무관; 다른 갈래 2018 매수·보유 +30.2%와 달라짐).
DATA_EXCLUDE = {"2018": {"QUBT"}, "2019": {"QUBT"}}


def grid() -> list[dict]:
    g = []
    for J, gm in C1_MAIN + C1_NB:
        g.append({"id": f"C1_J{J}_g{gm:+.1f}", "cand": "C1", "J": J, "g": gm, "nb": (J, gm) in C1_NB,
                  "label": f"KL J{J} γ{gm:+.1f}σ → 20일 수익>0 재진입"})
    for L in C2_MAIN:
        g.append({"id": f"C2_L{int(L*100)}", "cand": "C2", "L": L, "mode": "top", "nb": False,
                  "label": f"HZZ 월초 시가 −{L:.0%} → 다음 달 120일 상위 10%면 재매수"})
    for mode, L in C2_NB:
        g.append({"id": f"C2_L{int(L*100)}_{mode}", "cand": "C2", "L": L, "mode": mode, "nb": True,
                  "label": f"HZZ −{L:.0%} " + ("→ 다음 달 무조건 재매수" if mode == "all" else "(P0 = 전월 말 종가) → 상위 10% 재매수")})
    for k in C3_K:
        g.append({"id": f"C3_k{int(k*100)}", "cand": "C3", "k": k, "nb": False,
                  "label": f"추적 20% → 종가 ≥ 52주 고점×{k:.2f} 재진입"})
    for etf in C4_ETF:
        for c in C4_C:
            for earn in (False, True):
                g.append({"id": f"C4_{etf}_c{int(c*100)}" + ("_e" if earn else ""), "cand": "C4", "etf": etf, "c": c,
                          "earn": earn, "filt": True, "nb": False, "start": "shift",
                          "label": f"낙폭≤−40% & 고유낙폭≥−{c:.0%}({etf}) 매수 60일" + (" + 실적일 제외" if earn else "")})
    for v in C5_V:
        g.append({"id": f"C5_v{int(v*100)}", "cand": "C5", "v": v, "vcp": False, "nb": False,
                  "label": f"추적 20% → 10일 거래량 ≤ {v:.1f}×50일 & 종가>SMA10 재진입"})
    g.append({"id": "C5_vcp", "cand": "C5", "v": None, "vcp": True, "nb": False,
              "label": "추적 20% → VCP 수축 2회 연속 감소·마지막 ≤10% 피벗 돌파 재진입"})
    for re in C6_RE:
        for m in C6_M:
            g.append({"id": f"C6_{re}_m{m:g}", "cand": "C6", "re": re, "m": m, "half": True, "nb": False,
                      "label": f"2N 손절 → {re} 재진입 1/2, +{m:g}ATR에 나머지 1/2"})
    for base in C7_BASES:
        for f in (False, True):
            nm = "SMA200" if base[0] == "sma" else "추적 20%"
            g.append({"id": f"C7_{base[0]}{base[1]:g}_{'on' if f else 'off'}", "cand": "C7", "base": base, "filter": f,
                      "nb": base[0] == "trail", "label": f"{nm} 손절" + (" + 공황 땐 끔" if f else "(필터 없음)")})
    return g


def controls() -> list[dict]:
    """비교 기준(고르기에 넣지 않음)."""
    return [
        {"id": "CTL_T20_brk", "cand": "CTL", "label": "추적 20% → 20일 돌파(종가>그 전 20일 최고 고가) 재진입", "re": "brk"},
        {"id": "CTL_T20_none", "cand": "CTL", "label": "추적 20%, 창 안에서 다시 사지 않음", "re": "none"},
        {"id": "CTL_C4_nofilt", "cand": "CTL", "label": "낙폭≤−40% 매수 60일(고유낙폭 조건 없음)", "c4": True, "start": "shift"},
        {"id": "CTL_C6_r1_full", "cand": "CTL", "label": "2N 손절 → r1 전량 재진입(LINE s8·r1)", "c6": ("r1", False)},
        {"id": "CTL_C6_r2_full", "cand": "CTL", "label": "2N 손절 → r2 전량 재진입(LINE s8·r2)", "c6": ("r2", False)},
    ]


# ============================================================================ 엔진 확장(매수 스탑 한 단계, p2_LINE과 같은 방식)
def sim_ext(b, a, z, policy, slip: float = 0.0):
    """stock_lab.simulate와 같은 코드 + '손절 → 매수 스탑(stop_buys) → 지정가 매수' 순서의 매수 스탑 단계."""
    c, o, h, l = b.c, b.o, b.h, b.l
    cash, sh = 0.0, 1.0 / c[a - 1]
    st = {"pos": 1.0, "entry_px": c[a - 1], "entry_t": a - 1, "peak": c[a - 1], "high_peak": h[a - 1],
          "exit_px": None, "exit_t": None, "trough": None, "days_out": 0, "n_exits": 0, "last_reason": None,
          "start_t": a}
    vals = [1.0]
    pos_close = []
    cycles, events = [], []

    def equity(px):
        return cash + sh * px

    def set_pos(target, px, t, reason):
        nonlocal cash, sh
        eq = equity(px)
        cur = sh * px / eq if eq > 0 else 0.0
        if abs(target - cur) < 1e-9:
            return
        if target < cur:
            fill = px * (1 - slip)
            sell_sh = sh - target * eq / px
            cash += sell_sh * fill
            sh -= sell_sh
            if target <= 1e-9:
                sh = 0.0
                st.update(exit_px=fill, exit_t=t, trough=fill, days_out=0, n_exits=st["n_exits"] + 1, last_reason=reason)
                events.append((b.days[t], "sell", reason, fill))
        else:
            fill = px * (1 + slip)
            buy_val = (target - cur) * eq
            buy_val = min(buy_val, cash)
            sh += buy_val / fill
            cash -= buy_val
            if cur <= 1e-9:
                if st["exit_px"] is not None:
                    cycles.append({"exit_day": b.days[st["exit_t"]], "exit_px": st["exit_px"], "re_day": b.days[t],
                                   "re_px": fill, "gain": st["exit_px"] / fill - 1, "days_out": t - st["exit_t"],
                                   "reason": st["last_reason"], "open": False})
                st.update(entry_px=fill, entry_t=t, peak=fill, high_peak=fill, exit_px=None, exit_t=None, trough=None)
            events.append((b.days[t], "buy", reason, fill))
        eqn = equity(px)
        st["pos"] = sh * px / eqn if eqn > 0 else 0.0

    sb = getattr(policy, "stop_buys", None)
    for t in range(a, z + 1):
        tgt = policy.on_open(b, t, st)
        if tgt is not None and np.isfinite(o[t]):
            set_pos(float(np.clip(tgt, 0.0, 1.0)), o[t], t, "open")
        if st["pos"] > 1e-9:
            L = policy.stop(b, t, st)
            if L is not None and np.isfinite(L):
                if o[t] <= L:
                    set_pos(0.0, o[t], t, "stop_gap")
                elif l[t] <= L:
                    set_pos(0.0, L, t, "stop")
        if sb is not None and st["pos"] <= 1e-9:
            for lvl, tg in sb(b, t, st):
                if lvl is not None and np.isfinite(lvl) and h[t] >= lvl:
                    set_pos(tg, max(o[t], lvl), t, "stop_buy")
                    break
        for lvl, tg in sorted(policy.limit_buys(b, t, st), key=lambda x: -x[0]):
            if lvl is None or not np.isfinite(lvl):
                continue
            if l[t] <= lvl and st["pos"] < tg - 1e-9:
                set_pos(tg, min(o[t], lvl), t, "limit")
        vals.append(equity(c[t]))
        pos_close.append(st["pos"])
        if st["pos"] > 1e-9:
            st["peak"] = max(st["peak"], c[t])
            st["high_peak"] = max(st["high_peak"], h[t])
        else:
            st["days_out"] += 1
            st["trough"] = min(st["trough"], c[t]) if st["trough"] is not None else c[t]
        if hasattr(policy, "after_close"):
            policy.after_close(b, t, st)
    if st["pos"] <= 1e-9 and st["exit_px"] is not None:
        cycles.append({"exit_day": b.days[st["exit_t"]], "exit_px": st["exit_px"], "re_day": None, "re_px": c[z],
                       "gain": st["exit_px"] / c[z] - 1, "days_out": z - st["exit_t"], "reason": st["last_reason"],
                       "open": True})
    sim_ext.last_pos = np.array(pos_close)
    return np.array(vals), cycles, events


_ORIG_SIM = SL.simulate
SL.simulate = sim_ext      # 이 프로세스 안에서만 run_universe가 확장판을 쓰게 한다(파일은 고치지 않음)


# ============================================================================ 외부 자료(yfinance, 한 번 받아 캐시)
def fetch():
    import yfinance as yf
    q = yf.Ticker("QQQ").history(period="max", auto_adjust=False)
    df = pd.DataFrame({"date": q.index.strftime("%Y-%m-%d"), "close": q["Close"].to_numpy(), "adj": q["Adj Close"].to_numpy()})
    df.to_parquet(QQQ_LONG, index=False)
    print("QQQ", df.date.iloc[0], df.date.iloc[-1], len(df))
    syms = sorted(s for s in SL.load() if s not in SL.ETFS)
    earn = {}
    for i, s in enumerate(syms):
        for attempt in range(3):
            try:
                e = yf.Ticker(s).get_earnings_dates(limit=60)
                earn[s] = sorted({ts.strftime("%Y-%m-%d") for ts in e.index}) if e is not None else []
                break
            except Exception as ex:  # noqa: BLE001
                earn[s] = None
                time.sleep(2 + 3 * attempt)
        time.sleep(0.4)
        if i % 20 == 0:
            print(i, s, None if earn[s] is None else len(earn[s]))
    EARN.write_text(json.dumps(earn, ensure_ascii=False), encoding="utf-8")
    print("실적일 없음:", [s for s, v in earn.items() if not v])


_PANIC = None
_EARN = None
_TOP = None


def panic_map() -> dict:
    """날짜 → 공황 여부(그날 종가 기준). QQQ 1999~(yfinance, 배당 수정)."""
    global _PANIC
    if _PANIC is None:
        q = pd.read_parquet(QQQ_LONG)
        s = pd.Series(q.adj.to_numpy(), index=q.date)
        r = s.pct_change()
        ret2y = s / s.shift(PANIC_RET_N) - 1
        vol = r.rolling(PANIC_VOL_N, min_periods=PANIC_VOL_N).std() * np.sqrt(252)
        med = vol.shift(1).rolling(PANIC_MED_N, min_periods=PANIC_MED_N).median()
        p = (ret2y < 0) & (vol > med)
        p[ret2y.isna() | med.isna()] = False
        _PANIC = p.to_dict()
    return _PANIC


def earn_idx(b) -> np.ndarray:
    """그 종목 실적일(yfinance) → 그 날 또는 그 다음 거래일 인덱스."""
    global _EARN
    if _EARN is None:
        _EARN = json.loads(EARN.read_text(encoding="utf-8")) if EARN.exists() else {}

    def f():
        ds = _EARN.get(b.sym) or []
        return np.unique(np.searchsorted(b.days, np.array(ds, dtype=b.days.dtype), side="left")) if ds else np.array([], int)
    return b._memo(("p2lt_earn",), f)


def top_sets() -> dict:
    """'YYYY-MM' → 그 달 첫 거래일 전날 종가 기준 120일 수익 상위 10% 종목(ETF 제외 전 종목에서 순위)."""
    global _TOP
    if _TOP is None:
        data = SL.load()
        cal = data["SPY"].days
        firsts = [i for i in range(1, len(cal)) if cal[i][:7] != cal[i - 1][:7]]
        out = {}
        for i in firsts:
            prev = cal[i - 1]
            rows = []
            for s, b in data.items():
                if s in SL.ETFS:
                    continue
                j = b.idx(prev, "right") - 1
                if j < C2_MOM_N or b.days[j] != prev:
                    continue
                r = b.c[j] / b.c[j - C2_MOM_N] - 1
                if np.isfinite(r):
                    rows.append((r, s))
            rows.sort(reverse=True)
            n = math.ceil(C2_TOP * len(rows))
            out[cal[i][:7]] = {s for _, s in rows[:n]}
        _TOP = out
    return _TOP


# ============================================================================ 지표
def kl_arrays(b, J):
    def f():
        s = pd.Series(b.c)
        rJ = s / s.shift(J) - 1
        sd = rJ.rolling(C1_SD_N, min_periods=C1_SD_N).std()
        return rJ.to_numpy(), sd.to_numpy()
    return b._memo(("p2lt_kl", J), f)


def vol_avg(b, n):
    return b._memo(("p2lt_vol", n), lambda: pd.Series(b.v).rolling(n, min_periods=n).mean().to_numpy())


def etf_aligned(b, etf):
    def f():
        e = SL.load()[etf]
        return pd.Series(e.c, index=e.days).reindex(b.days).ffill().to_numpy()
    return b._memo(("p2lt_etf", etf), f)


def swings(b):
    def f():
        w = 2 * VCP_SW + 1
        hmax = pd.Series(b.h).rolling(w, center=True, min_periods=w).max().to_numpy()
        lmin = pd.Series(b.l).rolling(w, center=True, min_periods=w).min().to_numpy()
        return b.h >= hmax, b.l <= lmin     # [i]는 h[i-5..i+5]만 본다 → i+5 ≤ t-1일 때만 쓴다
    return b._memo(("p2lt_sw",), f)


def vcp_pivot(b, t):
    """t일 매수 스탑 가격(t-1까지 정보) 또는 None."""
    sh, sl = swings(b)
    i0, i1 = max(VCP_SW, t - VCP_LOOK), t - 1 - VCP_SW
    if i1 <= i0:
        return None
    pts = []
    for i in range(i0, i1 + 1):
        if sh[i]:
            pts.append((i, 0, b.h[i]))
        if sl[i]:
            pts.append((i, 1, b.l[i]))
    seq = []
    for p in pts:
        if seq and seq[-1][1] == p[1]:
            if (p[1] == 0 and p[2] >= seq[-1][2]) or (p[1] == 1 and p[2] <= seq[-1][2]):
                seq[-1] = p
        else:
            seq.append(p)
    con = [(seq[j][0], 1 - seq[j + 1][2] / seq[j][2]) for j in range(len(seq) - 1) if seq[j][1] == 0 and seq[j + 1][1] == 1]
    if len(con) < VCP_DEC + 1:
        return None
    d = [x[1] for x in con[-(VCP_DEC + 1):]]
    if not all(d[k] > d[k + 1] for k in range(VCP_DEC)) or d[-1] > VCP_LAST:
        return None
    lastH = [p for p in seq if p[1] == 0][-1]
    piv_i, piv = lastH[0], lastH[2]
    if piv_i + 1 <= t - 1 and np.nanmax(b.c[piv_i + 1:t]) >= piv:
        return None
    return piv


_BETA: dict = {}


def idio_dd(b, i, etf):
    """i = t-1. (종목 낙폭, 고유 낙폭, h) — h = 직전 252일 최고 종가일, β = h까지 252일 일간 회귀."""
    lo = i - 251
    if lo < 1:
        return None
    h = lo + int(np.argmax(b.c[lo:i + 1]))
    E = etf_aligned(b, etf)
    key = (b.sym, etf, h)
    if key not in _BETA:
        a0 = h - C4_BETA_N + 1
        if a0 < 1:
            _BETA[key] = np.nan
        else:
            rs = b.c[a0:h + 1] / b.c[a0 - 1:h] - 1
            re = E[a0:h + 1] / E[a0 - 1:h] - 1
            m = np.isfinite(rs) & np.isfinite(re)
            _BETA[key] = float(np.cov(rs[m], re[m])[0, 1] / np.var(re[m], ddof=1)) if m.sum() >= C4_BETA_MIN else np.nan
    beta = _BETA[key]
    if not np.isfinite(beta) or not (np.isfinite(E[i]) and np.isfinite(E[h])):
        return None
    Ri = b.c[i] / b.c[h] - 1
    Rs = E[i] / E[h] - 1
    return Ri, Ri - beta * Rs, h


def earn_hit(b, i, h):
    """h+1..i 최대 일간 하락일이 실적일 ±1거래일이면 True."""
    if i <= h:
        return False
    r = b.c[h + 1:i + 1] / b.c[h:i] - 1
    j = h + 1 + int(np.nanargmin(r))
    e = earn_idx(b)
    return bool(len(e) and np.min(np.abs(e - j)) <= C4_EARN_W)


# ============================================================================ 정책
class KL(Policy):
    """C1 Kaminski-Lo: J일 누적수익 < γ×σ_J(직전 252일) → 다음 시가 매도, 직전 20일 누적수익 > 0 → 다음 시가 재매수."""

    def __init__(self, J, g):
        self.J, self.g = J, g

    def on_open(self, b, t, st):
        if st["pos"] > 1e-9:
            rJ, sd = kl_arrays(b, self.J)
            x, s = rJ[t - 1], sd[t - 1]
            return 0.0 if np.isfinite(x) and np.isfinite(s) and x < self.g * s else None
        if st["exit_px"] is None or t - 1 - C1_RE_N < 0:
            return None
        return 1.0 if b.c[t - 1] / b.c[t - 1 - C1_RE_N] - 1 > 0 else None


class HZZ(Policy):
    """C2 Han-Zhou-Zhu: 그 달 P0 대비 −L 장중 손절, 남은 달 현금, 다음 달 첫 거래일 시가에 (상위 10%면) 재매수."""

    def __init__(self, L, mode):
        self.L, self.mode, self.P0 = L, mode, None

    @staticmethod
    def mstart(b, t):
        return b.days[t][:7] != b.days[t - 1][:7]

    def on_open(self, b, t, st):
        if st["pos"] <= 1e-9 and st["exit_px"] is not None and self.mstart(b, t):
            if self.mode == "all" or b.sym in top_sets().get(b.days[t][:7], ()):
                return 1.0
        return None

    def stop(self, b, t, st):
        if self.P0 is None or self.mstart(b, t) or st["entry_t"] == t:
            self.P0 = b.c[t - 1] if self.mode == "prevclose" else b.o[t]
        return self.P0 * (1 - self.L)


class Trail(Policy):
    """추적 20% 장중 손절(산 뒤 최고 종가 × 0.8) + 재진입 규칙. C3·C5·통제."""

    def __init__(self, re, p=None):
        self.re, self.p = re, p

    def stop(self, b, t, st):
        return st["peak"] * (1 - BASE_TRAIL)

    def on_open(self, b, t, st):
        if st["pos"] > 1e-9 or st["exit_px"] is None or self.re in ("none", "vcp"):
            return None
        c, i = b.c, t - 1
        if self.re == "hi52":
            return 1.0 if c[i] >= self.p * b.close_high(252)[i] else None
        if self.re == "brk":
            return 1.0 if i >= 1 and c[i] > b.high_n(BRK_N)[i - 1] else None
        if self.re == "vol":
            v10, v50, m = vol_avg(b, C5_S)[i], vol_avg(b, C5_L)[i], b.sma(C5_SMA)[i]
            return 1.0 if np.isfinite(v10) and np.isfinite(v50) and np.isfinite(m) and v10 <= self.p * v50 and c[i] > m else None
        return None

    def stop_buys(self, b, t, st):
        if self.re != "vcp" or st["exit_px"] is None or st["days_out"] < 1:
            return ()
        p = vcp_pivot(b, t)
        return ((p, 1.0),) if p is not None else ()


class SectorDeep(Policy):
    """C4: 창 시작 현금. t-1 종가 낙폭 ≤ −40%이고 (고유 낙폭 ≥ −c)(실적일 하락 아님)이면 t 시가 매수 → 60거래일 뒤 시가 매도."""

    def __init__(self, c=None, etf=None, earn=False, filt=True, slip=0.0):
        self.cc, self.etf, self.earn, self.filt, self.slip = c, etf, earn, filt, slip
        self.trades, self.buy_t, self.tr = [], None, None

    def on_open(self, b, t, st):
        if t == st["start_t"]:
            return 0.0
        i = t - 1
        if st["pos"] > 1e-9:
            if self.buy_t is not None and t - self.buy_t >= C4_HOLD:
                self.tr["exit"] = (b.days[t], float(b.o[t] * (1 - self.slip)))
                self.tr, self.buy_t = None, None
                return 0.0
            return None
        dd = b.dd252()[i]
        if not (np.isfinite(dd) and dd <= -C4_DD):
            return None
        if self.filt or self.earn:
            x = idio_dd(b, i, self.etf or "QQQ")
            if x is None:
                return None
            _, idio, h = x
            if self.filt and idio < -self.cc:
                return None
            if self.earn and earn_hit(b, i, h):
                return None
        self.buy_t = t
        self.tr = {"sym": b.sym, "start": b.days[t], "buy": float(b.o[t] * (1 + self.slip)), "exit": None}
        self.trades.append(self.tr)
        return 1.0


class Pyr(Policy):
    """C6 터틀 피라미딩: 손절선 = 마지막 매수가 − 2×ATR20(매수 전날 값). 재진입(r1 손절선 회복 종가 / r2 20일 고가 돌파)에 1/2,
    종가 > 첫 매수가 + m×ATR20(그날 값)이면 다음 시가에 나머지 1/2. half=False면 LINE s8·r1/r2와 같은 규칙(전량 재진입)."""

    def __init__(self, re, m=None, half=True, slip=0.0):
        self.re, self.m, self.half, self.slip = re, m, half, slip
        self.seen_et, self.units, self.first, self.last_buy, self.N = None, 2, None, None, None
        self.add_t, self.last_L, self.exit_line = None, None, None

    def _sync(self, b, st):
        if st["pos"] > 1e-9 and st["entry_t"] != self.seen_et:
            et = self.seen_et = st["entry_t"]
            ref = et if et < st["start_t"] else et - 1
            self.last_buy, self.N, self.first = st["entry_px"], b.atr(C6_ATR)[ref], st["entry_px"]
            self.units = 2 if (et < st["start_t"] or not self.half) else 1

    def on_open(self, b, t, st):
        self._sync(b, st)
        c = b.c
        if st["pos"] > 1e-9:
            if self.half and self.units == 1 and self.first is not None:
                a = b.atr(C6_ATR)[t - 1]
                if np.isfinite(a) and c[t - 1] > self.first + self.m * a:
                    self.add_t, self.units = t, 2
                    return 1.0
            return None
        if st["exit_px"] is None or self.re != "r1":
            return None
        if self.exit_line is not None and c[t - 1] > self.exit_line:
            return 0.5 if self.half else 1.0
        return None

    def stop(self, b, t, st):
        self._sync(b, st)
        if self.add_t == t:
            self.last_buy, self.N = b.o[t] * (1 + self.slip), b.atr(C6_ATR)[t - 1]
        L = self.last_buy - C6_K * self.N if self.last_buy is not None and np.isfinite(self.N) else None
        self.last_L = L
        return L

    def stop_buys(self, b, t, st):
        if self.re != "r2" or st["exit_px"] is None or st["days_out"] < 1:
            return ()
        return ((b.high_n(BRK_N)[t - 1], 0.5 if self.half else 1.0),)

    def after_close(self, b, t, st):
        if st["pos"] <= 1e-9:
            self.units = 0
            if st["exit_t"] == t and st["last_reason"] in ("stop", "stop_gap"):
                self.exit_line = self.last_L
        else:
            self._sync(b, st)


class PanicGate(Policy):
    """C7: 가격선 손절(SMA200 종가 이탈 → 다음 시가 / 추적 20% 장중)을 공황 상태(t-1)에서 끈다. 재진입 = 같은 선 회복 → 다음 시가."""

    def __init__(self, base, use_filter):
        self.kind, self.p = base
        self.f = use_filter
        self.last_L, self.exit_line = None, None

    def panic(self, b, t):
        return self.f and bool(panic_map().get(b.days[t - 1], False))

    def on_open(self, b, t, st):
        c = b.c
        if self.kind == "sma":
            m = b.sma(self.p)[t - 1]
            if not np.isfinite(m):
                return None
            if st["pos"] > 1e-9:
                return 0.0 if c[t - 1] < m and not self.panic(b, t) else None
            return 1.0 if st["exit_px"] is not None and c[t - 1] > m else None
        if st["pos"] <= 1e-9 and st["exit_px"] is not None and self.exit_line is not None and c[t - 1] > self.exit_line:
            return 1.0
        return None

    def stop(self, b, t, st):
        if self.kind != "trail" or self.panic(b, t):
            self.last_L = None
            return None
        self.last_L = st["peak"] * (1 - self.p)
        return self.last_L

    def after_close(self, b, t, st):
        if st["pos"] <= 1e-9 and st["exit_t"] == t and st["last_reason"] in ("stop", "stop_gap"):
            self.exit_line = self.last_L


def make_policy(sp: dict, slip: float = 0.0):
    c = sp["cand"]
    if c == "C1":
        return KL(sp["J"], sp["g"])
    if c == "C2":
        return HZZ(sp["L"], sp["mode"])
    if c == "C3":
        return Trail("hi52", sp["k"])
    if c == "C4":
        return SectorDeep(sp["c"], sp["etf"], sp["earn"], True, slip)
    if c == "C5":
        return Trail("vcp") if sp["vcp"] else Trail("vol", sp["v"])
    if c == "C6":
        return Pyr(sp["re"], sp["m"], True, slip)
    if c == "C7":
        return PanicGate(sp["base"], sp["filter"])
    if sp.get("c4"):
        return SectorDeep(None, None, False, False, slip)
    if sp.get("c6"):
        return Pyr(sp["c6"][0], None, False, slip)
    return Trail(sp["re"])


# ============================================================================ 평가
def cyc_stats(cycles, d0, d1, data):
    """그 창에서 판 주기. 창 끝까지 못 되산 주기는 창 끝 종가로 평가(p2_LINE과 같음)."""
    g, days = [], []
    for cy in cycles:
        if not (d0 <= cy["exit_day"] <= d1):
            continue
        if cy["re_day"] is not None and cy["re_day"] <= d1:
            g.append(cy["gain"])
        else:
            b = data[cy["sym"]]
            z = b.idx(d1, "right") - 1
            g.append(cy["exit_px"] / b.c[z] - 1)
        days.append(cy["days_out"])
    return _gstats(np.array(g), days)


def trade_stats(trades, d0, d1, data):
    """C4 매수 1회 = 판 값/산 값 − 1(창 끝까지 안 판 것은 창 끝 종가)."""
    from datetime import date
    g, days, weeks, done = [], [], set(), 0
    for tr in trades:
        if not (d0 <= tr["start"] <= d1):
            continue
        b = data[tr["sym"]]
        if tr["exit"] is not None and tr["exit"][0] <= d1:
            px, end = tr["exit"][1], tr["exit"][0]
            done += 1
        else:
            z = b.idx(d1, "right") - 1
            px, end = b.c[z], b.days[z]
        g.append(px / tr["buy"] - 1)
        days.append(int(b.idx(end) - b.idx(tr["start"])))
        weeks.add(date.fromisoformat(tr["start"]).isocalendar()[:2])
    out = _gstats(np.array(g), days)
    out["매수_주수"] = len(weeks)                       # 매수가 몰린 정도(서로 다른 ISO 주 수)
    out["완료비율"] = done / len(g) if g else float("nan")   # 창 끝 전에 60일 보유를 마친 비율
    return out


def _gstats(g, days):
    win, loss = g[g > 0], g[g <= 0]
    return {
        "손절주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
        "평균이익": float(win.mean()) if len(win) else float("nan"),
        "평균손실": float(loss.mean()) if len(loss) else float("nan"),
        "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
        "기대값": float(g.mean()) if len(g) else float("nan"),
        "기대값_중앙": float(np.median(g)) if len(g) else float("nan"),
        "밖_일수_중앙": float(np.median(days)) if days else float("nan"),
    }


_CAL = None


def prev_day(d0: str) -> str:
    global _CAL
    if _CAL is None:
        _CAL = SL.load()["SPY"].days
    return str(_CAL[int(np.searchsorted(_CAL, d0)) - 1])


def run_variant(args):
    sp, slip, runs = args
    data = SL.load()
    out = {}
    for rname, (d0, d1) in runs.items():
        pols = []

        def fac():
            p = make_policy(sp, slip)
            pols.append(p)
            return p

        r0 = prev_day(d0) if sp.get("start") == "shift" else d0
        syms = [s for s in SL.universe(252, r0) if s not in DATA_EXCLUDE[rname]] if rname in DATA_EXCLUDE else None
        res = SL.run_universe(fac, syms, {rname: (r0, d1)}, slip)
        ws = SL.window_stats(res)
        trades = [tr for p in pols for tr in getattr(p, "trades", [])]
        for w in ws:
            wd0, wd1 = SL.WINDOWS[w]
            if isinstance(pols[0], SectorDeep):
                ws[w].update(trade_stats(trades, wd0, wd1, data))
            else:
                ws[w].update(cyc_stats(res[rname]["cycles"], wd0, wd1, data))
            ws[w]["손절_종목당"] = float(ws[w]["손절주기"] / max(ws[w]["종목수"], 1))
        if rname == "2026":
            r = res[rname]
            m = (r["R"].index >= AI_SHOCK[0]) & (r["R"].index <= AI_SHOCK[1])
            pr = np.concatenate([[1.0], (1 + r["R"][m].mean(axis=1).fillna(0)).cumprod().to_numpy()])
            pb = np.concatenate([[1.0], (1 + r["B"][m].mean(axis=1).fillna(0)).cumprod().to_numpy()])
            ws["2026"]["AI쇼크_수익"], ws["2026"]["AI쇼크_BH"] = float(pr[-1] - 1), float(pb[-1] - 1)
        out.update(ws)
    return sp["id"], out


def calmar(s):
    """수익/|낙폭|. 낙폭 0이면 수익 > 0일 때만 무한대, 거래가 없어 수익도 0이면 고르기에서 뺀다(−무한대).
    (첫 실행에서 앞 구간 거래 0회인 C4가 0/0 = 무한대로 G3에 뽑힌 버그를 고침 — 격자·정의는 그대로)"""
    if s["포트_낙폭"] < 0:
        return s["포트_수익"] / abs(s["포트_낙폭"])
    return float("inf") if s["포트_수익"] > 0 else float("-inf")


# ============================================================================ 7종목(데스크 경로) 실험: C8 위험 균등 비중, Han-Zhou-Zhu 7종목판
def _port_windows():
    import harness as H
    return {"2026": dict(H.P26), "hold": {k: H.PHOLD[k] for k in ("2025", "2024", "관세", "24여름", "25가을")}}


def port_metrics(r_by_bn: dict) -> dict:
    import harness as H
    out = {}
    for bn, wins in _port_windows().items():
        b = H.base(bn)
        r = r_by_bn[bn]
        out[bn] = {}
        for w, (lo, hi) in wins.items():
            x = r[(r.index >= lo) & (r.index <= hi)].to_numpy()
            y = b.ret[(b.ret.index >= lo) & (b.ret.index <= hi)].to_numpy()
            cur = np.concatenate([[1.0], np.cumprod(1 + x)])
            exc, e = excess_tol(cur, y)
            out[bn][w] = {"수익": float(cur[-1] - 1), "낙폭": SL._mdd(cur), "효율선초과": exc, "효율선_e": e}
    return out


def excess_tol(rule_curve, bh_daily, tol=1e-6):
    """SL.same_mdd_excess와 같고 낙폭 비교 허용치만 1e-6(StockEngine 경로와 기본 ret가 1e-9 수준으로 달라 기본 자신이
    e=1.0에서 탈락하던 문제를 고침; 첫 실행 뒤 발견)."""
    target = SL._mdd(rule_curve)
    best_ret, best_e = -1.0, 0.0
    for e in np.arange(0.0, 1.0001, 0.005):
        cur = np.concatenate([[1.0], np.cumprod(1 + e * bh_daily)])
        if SL._mdd(cur) >= target - tol and cur[-1] - 1 > best_ret:
            best_ret, best_e = cur[-1] - 1, e
    return float(rule_curve[-1] - 1 - best_ret), float(best_e)


def risk_weights(eng):
    """재조정일마다 고른 종목(기본 목표의 종목)에 원 비중 ∝ 1/(ATR20/종가)(k-1 값)을 주고 데스크 상한(project_caps)을 그대로 적용."""
    root = HERE.parents[4]
    sys.path.insert(0, str(root))
    from desk import strategies as S  # noqa: E402  (읽기만)
    b = eng.b
    new, log = [], []
    for i, d in enumerate(b.days):
        w = b.weights[i]
        if d not in eng.reb or not w:
            new.append(dict(w))
            continue
        k = eng.k_of[d]
        inv = {}
        for key in w:
            a, c = eng.atr(key, k, C8_ATR), eng.C[key][k - 1]
            inv[key] = c / a if np.isfinite(a) and a > 0 and np.isfinite(c) else np.nan
        ok = [v for v in inv.values() if np.isfinite(v)]
        fill = float(np.mean(ok)) if ok else 1.0
        inv = {k2: (v if np.isfinite(v) else fill) for k2, v in inv.items()}
        tot = sum(inv.values())
        raw = {k2: v / tot for k2, v in inv.items()}
        # 섹터는 기본 경로를 만들 때 고정한 표(cache/base_*.json)를 쓴다. 지금 watchlist(세션 중 d5ce794가 미분류 56종목을 분류)를
        # 쓰면 기본 경로와 다른 상한이 걸려 비교가 어긋난다(2025 +68.0% 대 고정 표 결과).
        tgt = S.project_caps(raw, b.sector)
        new.append(tgt)
        log.append({"day": d, "base": {eng.tk[k2]: round(v, 3) for k2, v in w.items()},
                    "risk": {eng.tk[k2]: round(v, 3) for k2, v in tgt.items()},
                    "sum_base": round(sum(w.values()), 4), "sum_risk": round(sum(tgt.values()), 4)})
    return new, log


class TwoN:
    """C8 이웃: 진입가 − 2×ATR20(진입 전날 값) 장중 손절, 재진입은 다음 주간 재조정(현금 대기)."""

    def allow(self, key, k, st, eng): return True
    def exit_open(self, key, k, st, eng): return False
    def reenter(self, key, k, st, eng): return False

    def stop(self, key, k, st, eng):
        if "N" not in st:
            st["N"] = eng.atr(key, st["entry_k"], C8_ATR)
        return st["entry"] - C6_K * st["N"] if np.isfinite(st["N"]) else None


def hzz7_policy(eng, variant):
    """Han-Zhou-Zhu 7종목판. entry = fam_B '장중 고정 10% / 재조정까지 현금' 그대로(진입가 대비),
    week = 매 재조정일 시가를 P0로 다시 잡음, month = 그 달 첫 거래일 시가를 P0로, 손절되면 다음 달 첫 재조정까지 현금."""
    import fam_B_stock_stops as FB

    class P(FB.StopPolicy):
        def __init__(self):
            super().__init__(eng, "intraday", "fixed", HZZ7_L, "cash")
            days = eng.book.days
            self.bdays = days
            self.mfirst = {}
            for d in eng.b.days:
                self.mfirst.setdefault(d[:7], eng.k_of[d])

        def level(self, key, k, st):
            if variant == "entry":
                return st["entry"] * (1 - self.lvl)
            if variant == "week":
                kr = int(self.reb_k[self.reb_k <= k].max())
            else:
                kr = self.mfirst[self.bdays[k][:7]]
            ref = eng.O[key][kr] if st["entry_k"] <= kr else st["entry"]
            return ref * (1 - self.lvl)

        def allow(self, key, k, st, e):
            if variant != "month" or key not in self.last_stop:
                return True
            return self.bdays[self.last_stop[key]][:7] != self.bdays[k][:7]
    return P()


def hzz7_cycles(eng, pol, events, variant, lo, hi):
    """손절 1회 = 손절가 / (다시 살 수 있는 첫 재조정 시가) − 1. week·entry는 다음 재조정, month는 다음 달 첫 재조정."""
    inv = {v: k for k, v in eng.tk.items()}
    reb_days = sorted(eng.reb)
    g = []
    for ev in events:
        if ev["type"] != "stop" or not (lo <= ev["day"] <= hi):
            continue
        key = inv[ev["key"]]
        nxt = [d for d in reb_days if d > ev["day"] and (variant != "month" or d[:7] != ev["day"][:7])]
        if not nxt:
            continue
        o = eng.O[key][eng.k_of[nxt[0]]]
        if np.isfinite(o) and o > 0:
            g.append(ev["px"] / o - 1)
    return _gstats(np.array(g), [])


def run_port(slip=0.0):
    import harness as H
    rows = {}
    logs = {}
    for name in ("base", "C8_risk", "C8_risk_2N", "HZZ7_entry", "HZZ7_week", "HZZ7_month"):
        rb, cyc, nst = {}, {}, {}
        for bn in ("2026", "hold"):
            eng = H.StockEngine(bn)
            if name.startswith("C8"):
                w, lg = risk_weights(eng)
                if bn == "2026":
                    logs["C8_weights_2026"] = lg
                eng.b = SimpleNamespace(weights=w, days=eng.b.days)
            if name in ("base", "C8_risk"):
                pol = H.NoPolicy()
            elif name == "C8_risk_2N":
                pol = TwoN()
            else:
                pol = hzz7_policy(eng, name.split("_")[1])
            r, ev = eng.run(pol, slip=slip)
            rb[bn] = r
            nst[bn] = sum(1 for e in ev if e["type"] == "stop")
            if name.startswith("HZZ7"):
                wins = _port_windows()[bn]
                cyc[bn] = {w: hzz7_cycles(eng, pol, ev, name.split("_")[1], *wins[w]) for w in wins}
        m = port_metrics(rb)
        rows[name] = {"metrics": m, "stops": nst, "cycles": cyc}
        print(name, {bn: {w: f"{v['수익']:+.1%}/{v['낙폭']:.1%} 초과 {v['효율선초과']:+.1%}" for w, v in m[bn].items()} for bn in m})
    return rows, logs


# ============================================================================ 점검
def check():
    data = SL.load()
    syms = SL.universe(252, "2026-01-02")[:30]
    worst = 0.0
    specs = [s for s in grid() if s["cand"] in ("C1", "C2", "C3", "C7")] + [s for s in controls() if s["id"] == "CTL_T20_brk"]
    for sp in specs:
        for s in syms[:12]:
            b = data[s]
            a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
            v1, c1, _ = _ORIG_SIM(b, a, z, make_policy(sp))
            v2, c2, _ = sim_ext(b, a, z, make_policy(sp))
            worst = max(worst, float(np.max(np.abs(v1 - v2))), abs(len(c1) - len(c2)))
    print("매수 스탑 없는 정책: 확장판 vs 원본 최대 차(0이어야 함)", worst)
    rng = np.random.default_rng(0)
    worst = 0.0
    for sp in grid() + controls():
        for s in syms[:6]:
            b = data[s]
            a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
            v1, _, _ = sim_ext(b, a, z, make_policy(sp))
            for T in (a + 40, a + 120):
                b2 = SL.Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v.copy())
                for arr in (b2.o, b2.h, b2.l, b2.c, b2.v):
                    arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
                b2.h[T + 1:] = np.maximum.reduce([b2.h[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
                b2.l[T + 1:] = np.minimum.reduce([b2.l[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
                v2, _, _ = sim_ext(b2, a, z, make_policy(sp))
                worst = max(worst, float(np.max(np.abs(v1[:T - a + 2] - v2[:T - a + 2]))))
    print("미래 교란 뒤 t까지 가치 최대 차(0이어야 함; C2 상위 집합·C7 공황·C4 ETF는 다른 종목 자료라 교란 안 함)", worst)
    # LINE 재현: half=False C6 = LINE s8·r1/r2, C7 SMA200 끔 = LINE s4·r1
    L = json.loads((SL.RESULTS / "p2_LINE.json").read_text(encoding="utf-8"))["results"]
    runs = {"2026": SL.RUNS["2026"]}
    for sp, ln in [(controls()[3], "s8·r1"), (controls()[4], "s8·r2"),
                   (next(s for s in grid() if s["id"] == "C7_sma200_off"), "s4·r1")]:
        _, ws = run_variant((sp, 0.0, runs))
        print(f"LINE {ln} 재현: 2026 {ws['2026']['포트_수익']:+.4f} (LINE {L[ln]['2026']['포트_수익']:+.4f}), "
              f"주기 {ws['2026']['손절주기']} (LINE {L[ln]['2026']['손절주기']})")
    pm = panic_map()
    for y in ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]:
        ds = [d for d in pm if d.startswith(y)]
        print("공황일", y, sum(pm[d] for d in ds), "/", len(ds))


# ============================================================================ 본 실행
def main():
    t0 = time.time()
    G, C = grid(), controls()
    print("격자", len(G), {c: sum(1 for g in G if g["cand"] == c) for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")},
          "통제", len(C))
    with Pool(12) as pool:
        out = dict(pool.map(run_variant, [(sp, 0.0, RUNS) for sp in G + C], chunksize=1))
    print(f"유니버스 {len(G) + len(C)}개 완료 {time.time() - t0:.0f}s")
    names = [g["id"] for g in G]
    spec = {s["id"]: s for s in G + C}
    ref = out["CTL_T20_none"]
    bh = {w: {k: ref[w][k] for k in ("포트BH_수익", "포트BH_낙폭", "종목수")} for w in ref}
    bh_shift = {w: {k: out["CTL_C4_nofilt"][w][k] for k in ("포트BH_수익", "포트BH_낙폭", "종목수")} for w in ref}

    front = {n: out[n]["2026앞"] for n in names}
    g1 = max(names, key=lambda n: front[n]["효율선초과"])
    elig = [n for n in names if front[n]["손절주기"] >= G2_MIN_CYCLES and np.isfinite(front[n]["기대값"])]
    g2 = max(elig, key=lambda n: front[n]["기대값"])
    g3 = max(names, key=lambda n: calmar(front[n]))
    picks = {"G1": g1, "G2": g2, "G3": g3}
    cand_best = {}
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        ns = [n for n in names if spec[n]["cand"] == c]
        cand_best[c] = max(ns, key=lambda n: front[n]["효율선초과"])

    uniq = sorted(set(picks.values()) | set(cand_best.values()))
    with Pool(min(12, len(uniq))) as pool:
        slip = dict(pool.map(run_variant, [(spec[n], SLIP_SENS, {k: SL.RUNS[k] for k in ("2026", "2025")}) for n in uniq]))

    def frac(ns, f):
        return (sum(1 for n in ns if f(n)) / len(ns)) if ns else float("nan")
    pos_front = [n for n in names if front[n]["효율선초과"] > 0]
    back_ok = lambda n: out[n]["2026뒤"]["효율선초과"] > 0
    y25_ok = lambda n: out[n]["2025"]["효율선초과"] > 0
    pass_set = {"앞>0 개수": len(pos_front), "전체": len(names),
                "뒤>0 | 앞>0": frac(pos_front, back_ok), "뒤>0 무조건": frac(names, back_ok),
                "2025>0 | 앞>0": frac(pos_front, y25_ok), "2025>0 무조건": frac(names, y25_ok),
                "뒤·2025 둘다>0 | 앞>0": frac(pos_front, lambda n: back_ok(n) and y25_ok(n)),
                "뒤·2025 둘다>0 무조건": frac(names, lambda n: back_ok(n) and y25_ok(n)), "앞>0 집합": pos_front}
    consist = {n: all(np.isfinite(out[n][w]["기대값"]) and out[n][w]["기대값"] > 0 and out[n][w]["손절주기"] >= 10
                      for w in ("2026앞", "2026뒤", "2025")) for n in names}
    ctx = {n: {y: out[n][y]["효율선초과"] for y in CONTEXT_YEARS} for n in names + [c["id"] for c in C]}

    t1 = time.time()
    port, port_logs = run_port(0.0)
    port_slip, _ = run_port(SLIP_SENS)
    print(f"7종목 완료 {time.time() - t1:.0f}s")
    # 1단계 B(같은 엔진) 비교 행
    Bj = json.loads((SL.RESULTS / "B_stock_stops.json").read_text(encoding="utf-8"))
    b_rows = {r["name"]: r for r in Bj["rows"] if r["name"] in ("장중 고정 8% / 재조정까지 현금", "장중 고정 12% / 재조정까지 현금",
                                                                 "장중 추적 15% / 재조정까지 현금")}
    # 짝 비교용 다른 갈래 숫자(LINE·DEEP json에서 읽기만)
    Lj = json.loads((SL.RESULTS / "p2_LINE.json").read_text(encoding="utf-8"))["results"]
    Dj = json.loads((SL.RESULTS / "p2_DEEP.json").read_text(encoding="utf-8"))["p3"]["results"]
    ext = {"LINE": {k: Lj[k] for k in ("s8·r1", "s8·r2", "s4·r1", "s10·r2", "s11·r2", "s12·r1")},
           "DEEP": {k: Dj[k] for k in ("D1_s20_d30_trail", "D1_s20_d40_trail", "D1_s20_d50_trail", "D1ctl_s20", "D4_d40_60")
                    if k in Dj}}
    pm = panic_map()
    panic_days = {y: int(sum(1 for d, v in pm.items() if d.startswith(y) and v)) for y in ["2018", "2019", "2020", "2021", "2022",
                                                                                            "2023", "2024", "2025", "2026"]}
    payload = {
        "grid": {"n": len(G), "specs": G, "controls": C,
                 "consts": {"C1_SD_N": C1_SD_N, "C1_RE_N": C1_RE_N, "C2_MOM_N": C2_MOM_N, "C2_TOP": C2_TOP, "BASE_TRAIL": BASE_TRAIL,
                            "C4": [C4_DD, C4_HOLD, C4_BETA_N, C4_BETA_MIN, C4_EARN_W], "C5": [C5_S, C5_L, C5_SMA],
                            "VCP": [VCP_LOOK, VCP_SW, VCP_DEC, VCP_LAST], "C6": [C6_ATR, C6_K, BRK_N],
                            "PANIC": [PANIC_RET_N, PANIC_VOL_N, PANIC_MED_N], "C8_ATR": C8_ATR, "HZZ7_L": HZZ7_L,
                            "SLIP_SENS": SLIP_SENS, "G2_MIN_CYCLES": G2_MIN_CYCLES}},
        "bh": bh, "bh_shift": bh_shift, "picks": picks, "cand_best": cand_best, "slip": slip, "pass_set": pass_set,
        "consist": consist, "context": ctx, "panic_days": panic_days, "results": out,
        "port": port, "port_slip": port_slip, "port_logs": port_logs, "B_rows": b_rows, "ext": ext,
        "earn_missing": [s for s, v in (json.loads(EARN.read_text(encoding="utf-8")) if EARN.exists() else {}).items() if not v],
    }
    p = SL.save(ID, payload)
    print("저장", p, f"{time.time() - t0:.0f}s")
    write_md(payload)


# ============================================================================ 보고서
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=0):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def rr(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


PORT_LAB = {"base": "기본(7종목 현행)", "C8_risk": "C8 위험 균등 비중(손절 없음)",
            "C8_risk_2N": "C8 + 진입가 − 2×ATR20 손절, 다음 재조정 재매수",
            "HZZ7_entry": "HZZ 10%, 기준가 = 산 가격(1단계 B 방식)", "HZZ7_week": "HZZ 10%, 기준가 = 매주 재조정 시가",
            "HZZ7_month": "HZZ 10%, 기준가 = 월 첫 거래일 시가, 다음 달까지 현금"}


def write_md(P):
    out, picks, bh, G = P["results"], P["picks"], P["bh"], P["grid"]["specs"]
    C = P["grid"]["controls"]
    spec = {s["id"]: s for s in G + C}
    names = [s["id"] for s in G]
    ps, port, pslip = P["pass_set"], P["port"], P["port_slip"]
    W = ("2026앞", "2026뒤", "2025")
    g1, g2 = picks["G1"], picks["G2"]
    o1, o2 = out[g1], out[g2]
    front_ev = [n for n in names if out[n]["2026앞"]["기대값"] > 0 and out[n]["2026앞"]["손절주기"] >= 10]
    n_cons = sum(1 for v in P["consist"].values() if v)
    pos_both = [n for n in ps["앞>0 집합"] if out[n]["2026뒤"]["효율선초과"] > 0]
    c4 = [n for n in names if spec[n]["cand"] == "C4"]
    c4m = [n for n in c4 if not spec[n]["earn"]]

    def rng(ns, w, k, d=0):
        v = [out[n][w][k] for n in ns if np.isfinite(out[n][w][k]) and out[n][w]["손절주기"] > 0]
        return f"{pct(min(v), d)}~{pct(max(v), d)}" if v else "—"

    ctl4 = out["CTL_C4_nofilt"]
    c2a = out["C2_L10_all"]
    hm = port["HZZ7_month"]
    hm26, hmh = hm["metrics"]["2026"], hm["metrics"]["hold"]
    hmc = hm["cycles"]["2026"]
    base26 = port["base"]["metrics"]["2026"]
    Lx = []
    A = Lx.append
    A("# 2단계 LIT-TEST: 문헌 규칙 C1~C8 검증 (2026-10-10)\n")
    A(f"> 코드 `code/p2_LITTEST.py`(`py p2_LITTEST.py check`: 엔진 동치·미래 교란·LINE 재현), 전 결과 `results/p2_LITTEST.json`. "
      f"AI 종목 전체(2026 {bh['2026']['종목수']}종목, 2025 {bh['2025']['종목수']}, 2022 {bh['2022']['종목수']}, 2018 {bh['2018']['종목수']})를 "
      "창 시작에 들고 규칙을 얹은 동일가중 포트폴리오다. C8과 HZZ 7종목판은 데스크 7종목 경로 위다. 수수료 0, 배당 수정, 현금 이자 0. 웹 검색 0회.\n")
    A("## 결론\n")
    A(f"1. **기각.** 문헌 값 그대로인 규칙과 사전 선언 이웃을 합친 38개 중 앞 구간 효율선초과가 플러스인 것은 {ps['앞>0 개수']}개였다. "
      f"그중 뒤 구간도 플러스는 {len(pos_both)}개({pp(ps['뒤>0 | 앞>0'])}, 고르지 않은 전체 {pp(ps['뒤>0 무조건'])})이고, 둘 다 앞 거래가 1번뿐인 C4다. "
      f"앞 1등 G1(=G3)은 SMA200 손절이다(C7 기준 규칙; 공황 필터는 2026에 0일 켜져 끔·켬이 같다). 앞 {pct(o1['2026앞']['포트_수익'])}/{pct(o1['2026앞']['포트_낙폭'])} → "
      f"뒤 {pct(o1['2026뒤']['포트_수익'])}(매수·보유 {pct(bh['2026뒤']['포트BH_수익'])}), 2025 {pct(o1['2025']['포트_수익'])}(매수·보유 {pct(bh['2025']['포트BH_수익'])})였다. "
      f"G2 Kaminski-Lo(J60, −1.5σ)는 앞 주기 기대값 {pct(o2['2026앞']['기대값'])} → 뒤 {pct(o2['2026뒤']['기대값'])}, 2025 {pct(o2['2025']['기대값'])}였다.")
    A(f"2. **'승률 30%라도 손익비로 이기는' 손절은 문헌 규칙에서도 2026에 없었다.** 앞 주기 기대값 플러스는 {len(front_ev)}/38(KL 두 개, "
      f"{pct(min(out[n]['2026앞']['기대값'] for n in front_ev))}~{pct(max(out[n]['2026앞']['기대값'] for n in front_ev))})이고, 앞·뒤·2025 모두 플러스는 {n_cons}/38이다. "
      f"원문에 가장 가까운 월 단위 Han-Zhou-Zhu(다음 달 무조건 재매수)는 7월 쇼크를 {pct(c2a['2026']['AI쇼크_BH'])} → {pct(c2a['2026']['AI쇼크_수익'])}로 줄였다. "
      f"뒤 주기도 승률 {pp(c2a['2026뒤']['승률'])}·손익비 {rr(c2a['2026뒤']['손익비'])}였다. 그러나 앞 주기 기대값 {pct(c2a['2026앞']['기대값'])}, 앞 효율선초과 {pct(c2a['2026앞']['효율선초과'])}였다. "
      f"7종목판도 같은 모양이다(쇼크 {pct(base26['AI쇼크']['수익'])} → {pct(hm26['AI쇼크']['수익'])}, 앞 {pct(hm26['앞']['효율선초과'])}, 2025 {pct(hmh['2025']['효율선초과'])}).")
    A(f"3. **'−40% 기계 매수'는 하락이 업종 전체로 설명될 때만 사면 매매 1회 성적이 좋아졌다(관찰, C4).** 고유 조건 없는 −40% 매수 대비 기대값이다. "
      f"뒤 {rng(c4m, '2026뒤', '기대값')}(조건 없음 {pct(ctl4['2026뒤']['기대값'])}), 2025 {rng(c4m, '2025', '기대값')}({pct(ctl4['2025']['기대값'])}), "
      f"2022 {rng(c4m, '2022', '기대값')}({pct(ctl4['2022']['기대값'])}). 다만 매수가 한두 주(2025-04, 2026-07 말)에 몰렸고 뒤 구간 매수는 아직 60일을 못 채웠다. "
      "노출이 0~5%라 포트 효과는 작다. 앞 구간 거래가 0~1번이라 앞→뒤 검증도 못 한다.\n")

    A("## 격자 (사전 고정: 유니버스 38개 + 통제 5개, 7종목 실험 6개)\n")
    A("| 후보 | 출처 | 규칙(문헌 값) | 이웃(사전 선언) | 개수 |\n|---|---|---|---|---|")
    A("| C1 | Kaminski-Lo 2014 | J일 누적수익 < γ×σ_J(직전 252일, 겹친 J일 수익) → 다음 시가 매도, 직전 20일 누적수익 > 0 → 다음 시가 재매수. J {60, 120} × γ {−1.0, −0.5} | J60·γ−1.5, J250·γ−1.0 | 6 |")
    A("| C2 | Han-Zhou-Zhu 2016 | 그 달 첫 거래일 시가 P0, 장중 저가 ≤ (1−L)P0 → 손절가(시가가 아래면 시가), 남은 달 현금. 다음 달 첫 거래일에 120일 수익 상위 10%(ETF 뺀 143종목 순위)면 시가 재매수, 아니면 다음 달 다시 확인. L {10, 15, 20%} | L10% 무조건 재매수, L10% P0 = 전월 말 종가(원문) | 5 |")
    A("| C3 | George-Hwang 2004 | 추적 20% 장중 손절(산 뒤 최고 종가) → 종가 ≥ k × 직전 252일 최고 종가면 다음 시가 재매수. k {0.85, 0.90, 0.95} | — | 3 |")
    A("| C4 | Chan 2003; Odean 1998 | 현금 시작. t−1 종가 252일 고점 대비 낙폭 ≤ −40%이고 고유 낙폭 R_i − β·R_업종 ≥ −c면 t 시가 매수, 60거래일 뒤 시가 매도. β는 고점일까지 252일 일간 회귀. c {10, 15, 20%} × 업종 ETF {SOXX, QQQ} × 보조 '고점 뒤 최대 일간 하락이 실적일 ±1거래일이면 제외' {없음, 있음} | — | 12 |")
    A("| C5 | Minervini VCP; 눌림목 게시물 | 추적 20% → (간단판) 10일 평균 거래량 ≤ v × 50일 평균이고 종가 > SMA10이면 다음 시가, v {0.6, 0.7, 0.8} / (완전판) 직전 60일 스윙(앞뒤 5일 국소 고·저) 조정 폭이 2회 연속 줄고 마지막 ≤ 10%면 마지막 스윙 고점에 매수 스탑 | — | 4 |")
    A("| C6 | 터틀 피라미딩; 불타기 게시물 | 손절선 = 마지막 매수가 − 2×ATR20(매수 전날 값). 재진입 r1(종가 > 손절된 선 → 다음 시가) 또는 r2(직전 20일 최고 고가 장중 돌파)에 1/2. 종가 > 첫 매수가 + m×ATR20이면 다음 시가에 나머지 1/2. m {0.5, 1.0} × {r1, r2} | — | 4 |")
    A("| C7 | Daniel-Moskowitz 2016 | 공황(QQQ 504일 수익 < 0이고 126일 변동성 > 그 앞 1,260일 중앙)이면 가격선 손절을 끈다. 기준 손절 SMA200(종가 이탈 → 다음 시가, 회복 → 다음 시가) × {끔, 켬} | 추적 20%(장중) × {끔, 켬} | 4 |")
    A("| 통제 | — | 추적 20% + 20일 돌파 재진입(= DEEP D1ctl_s20) / 추적 20% 재진입 없음 / C4 고유 조건 없음 / C6 전량 재진입(= LINE s8·r1, s8·r2) | — | 5(고르기 제외) |")
    A("| C8 (7종목) | Tharp R; 터틀 단위; 총자금 3% | 재조정일에 고른 종목 원 비중 ∝ 1/(ATR20/종가)(전날 값). 데스크 상한(한 종목 25%·섹터 60%·반도체 50%)은 그대로 | + 진입가 − 2×ATR20 장중 손절, 다음 재조정 재매수 | 2 |")
    A("| HZZ 7종목판 | Han-Zhou-Zhu 2016 | 1단계 B 코드 '장중 고정 10% / 재조정까지 현금'(기준가 = 산 가격) | 기준가 = 매주 재조정 시가 / 월 첫 거래일 시가 + 다음 달 첫 재조정까지 현금 | 3 |")
    A("\n- 유니버스 6+5+3+12+4+4+4 = **38**(그중 이웃 6). 미끄러짐 0으로 고르고, 고른 변형과 후보별 앞 1등은 0.5%로 다시 쟀다. "
      "7종목 실험은 기본 포함 6개이고 고르기 없이 전부 보고한다. 결과를 본 뒤 격자·정의를 바꾸지 않았다(버그 수정 2건은 한계에 적음).\n")

    A("## 변수 시점\n")
    A("| 변수 | 언제 아는가 | 체결 |\n|---|---|---|")
    A("| J일 누적수익·σ_J(직전 252일), 20일 누적수익(C1) | t−1 종가 | t 시가 |")
    A("| 월 P0(C2) | 그 달 첫 거래일 시가(이웃은 전월 말 종가) | 그 달 장중 스탑. 시가 확인 뒤 주문(KST 22:30, 주문 가능 시간 안). 전월 말 종가판은 밤사이 주문 가능 |")
    A("| 120일 수익 상위 10%(C2) | 그 달 첫 거래일 전날 종가 | 그날 시가 |")
    A("| 산 뒤 최고 종가(추적 20%), 252일 최고 종가(C3), 거래량 10·50일 평균·SMA10(C5) | t−1 종가 | 손절은 t 장중, 재진입은 t 시가 |")
    A("| 스윙 고·저(C5 완전판) | 국소점 i는 i+5일 종가에 확정. i+5 ≤ t−1만 쓴다 | 피벗 t 장중 매수 스탑(시가가 위면 시가) |")
    A("| 252일 고점일 h, R_i, R_업종(h~t−1), β(h까지 252일)(C4) | t−1 종가 | t 시가 |")
    A("| 실적일(C4 보조) | 이미 지난 발표일(그 하락 전에 공지됨, yfinance) | — |")
    A("| ATR20(C6·C8) | 매수 전날 종가 / 재조정 전날 종가 | 손절선은 매수 체결 뒤 그날부터 |")
    A("| QQQ 504일 수익·126일 변동성·1,260일 중앙(C7) | t−1 종가 | t 판단부터 |")
    A("| 7월 AI 쇼크 구간·연도 경계 | 사후 | 설명용. 신호에 쓰지 않음 |\n")

    def cells(ws):
        a, f, k, y5 = ws["2026"], ws["2026앞"], ws["2026뒤"], ws["2025"]
        x = (f"{pct(a['포트_수익'])} / {pct(a['포트_낙폭'])} | {pct(f['포트_수익'])} / {pct(f['포트_낙폭'])} ({pct(f['효율선초과'])}) | "
             f"{pct(k['포트_수익'])} / {pct(k['포트_낙폭'])} ({pct(k['효율선초과'])}) | {pct(y5['포트_수익'])} / {pct(y5['포트_낙폭'])} ({pct(y5['효율선초과'])})")
        x += f" | {pct(ws['2022']['포트_수익'])} / {pct(ws['2022']['포트_낙폭'])}" if "2022" in ws else " | —"
        x += f" | {pct(ws['2018']['포트_수익'])}" if "2018" in ws else " | —"
        return x + f" | {a['평균노출']:.2f} |"

    A("## 고르기 결과 (동일가중 포트폴리오 수익 / 최대 낙폭 (효율선초과))\n")
    hdr = "| 고르기 | 규칙 | 2026 전체 | 앞(고르기) | 뒤(확인) | 2025(보조) | 2022(맥락) | 2018(맥락) | 노출 |"
    A(hdr)
    A("|---|---|---|---|---|---|---|---|---|")
    b26, b25 = out["CTL_T20_none"]["2026"], out["CTL_T20_none"]["2025"]
    A(f"| 매수·보유 | — | {pct(b26['포트BH_수익'])} / {pct(b26['포트BH_낙폭'])} | {pct(bh['2026앞']['포트BH_수익'])} / {pct(bh['2026앞']['포트BH_낙폭'])} "
      f"| {pct(bh['2026뒤']['포트BH_수익'])} / {pct(bh['2026뒤']['포트BH_낙폭'])} | {pct(b25['포트BH_수익'])} / {pct(b25['포트BH_낙폭'])} "
      f"| {pct(bh['2022']['포트BH_수익'])} / {pct(bh['2022']['포트BH_낙폭'])} | {pct(bh['2018']['포트BH_수익'])} | 1.00 |")
    for g, n in picks.items():
        A(f"| {g} | `{n}` {spec[n]['label']} | " + cells(out[n]))
    for g in ("G1", "G2"):
        n = picks[g]
        A(f"| {g}{'(=G3)' if picks['G3'] == n else ''} 미끄러짐 0.5% | `{n}` | " + cells(P["slip"][n]))
    A(f"\n- 고르기에 정보가 있나: 앞 효율선초과 > 0 변형 {ps['앞>0 개수']}/38. 그중 뒤 > 0 {pp(ps['뒤>0 | 앞>0'])}(무조건 {pp(ps['뒤>0 무조건'])}), "
      f"2025 > 0 {pp(ps['2025>0 | 앞>0'])}(무조건 {pp(ps['2025>0 무조건'])}), 둘 다 {pp(ps['뒤·2025 둘다>0 | 앞>0'])}(무조건 {pp(ps['뒤·2025 둘다>0 무조건'])}). "
      f"뒤·2025 둘 다 플러스인 앞>0 변형은 {', '.join('`' + n + '`' for n in pos_both)}뿐이다. 앞 거래는 1번(CIFR 3/30)이고 노출 2%다.")
    A(f"- 뒤 구간은 매수·보유가 {pct(bh['2026뒤']['포트BH_수익'])}라 효율선 최선이 현금이다. 그래서 뒤 효율선초과 = 뒤 절대 수익이다. "
      f"7월 AI 쇼크(6/22~7/29) 매수·보유 {pct(out['CTL_T20_none']['2026']['AI쇼크_BH'])}, G1 {pct(o1['2026']['AI쇼크_수익'])}, G2 {pct(o2['2026']['AI쇼크_수익'])}.\n")

    A("### 후보별 요약 (변형 범위; 앞 1등은 같은 후보 안 앞 효율선초과 최대)\n")
    A("| 후보 | 변형 | 앞 효율선초과 | 뒤 수익 | 2025 효율선초과 | 2022 수익 | 노출 | 앞 1등 → 뒤 / 2025 초과 (미끄러짐 0.5%) |")
    A("|---|---|---|---|---|---|---|---|")
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        ns = [n for n in names if spec[n]["cand"] == c]

        def r2(w, k, d=1):
            v = [out[n][w][k] for n in ns]
            return f"{pct(min(v), d)} ~ {pct(max(v), d)}"
        ex = [out[n]["2026"]["평균노출"] for n in ns]
        nb = P["cand_best"][c]
        sl = P["slip"][nb]
        A(f"| {c} | {len(ns)} | {r2('2026앞', '효율선초과')} | {r2('2026뒤', '포트_수익')} | {r2('2025', '효율선초과')} | {r2('2022', '포트_수익')} "
          f"| {min(ex):.2f}~{max(ex):.2f} | `{nb}` {pct(out[nb]['2026뒤']['포트_수익'])} / {pct(out[nb]['2025']['효율선초과'])} "
          f"({pct(sl['2026뒤']['포트_수익'])} / {pct(sl['2025']['효율선초과'])}) |")
    A(f"| 매수·보유 | — | 0 | {pct(bh['2026뒤']['포트BH_수익'])} | 0 | {pct(bh['2022']['포트BH_수익'])} | 1.00 | — |\n")

    A("## 손절 주기 승률·손익비\n")
    A("주기 = 판 가격/다시 산 가격 − 1(+면 싸게 되삼). 창 끝까지 못 되산 주기는 그 창 끝 종가로 잰다. C4는 매수 1회 = 판 값/산 값 − 1이다(60일 못 채운 것은 창 끝 종가).\n")
    A("| 규칙 | 앞 n · 승률 · 손익비 · 기대값 | 뒤 n · 승률 · 손익비 · 기대값 | 2025 n · 승률 · 손익비 · 기대값 | 2022 기대값 | 수익 나은 종목(2026) | 최악 5% 종목(2026, 매수·보유) |")
    A("|---|---|---|---|---|---|---|")
    rows = list(dict.fromkeys(list(picks.values()) + list(P["cand_best"].values()) + ["C2_L10_all", "C4_SOXX_c20"]
                              + [c["id"] for c in C]))
    for n in rows:
        ws = out[n]
        cl = []
        for w in W:
            s = ws[w]
            cl.append(f"{s['손절주기']} · {pp(s['승률'])} · {rr(s['손익비'])} · {pct(s['기대값'])}")
        a = ws["2026"]
        A(f"| `{n}` | " + " | ".join(cl) + f" | {pct(ws['2022']['기대값'])} | {pp(a['수익_나은종목비율'])} | {pct(a['최악5%_종목수익'], 0)} ({pct(a['최악5%_BH'], 0)}) |")
    cnt = {w: sum(1 for n in names if out[n][w]["기대값"] > 0 and out[n][w]["손절주기"] >= 10) for w in W}
    wr = [out[n][w]["승률"] for n in names for w in W[:2] if out[n][w]["손절주기"] >= 30 and spec[n]["cand"] != "C4"]
    pr = [out[n][w]["손익비"] for n in names for w in W[:2] if out[n][w]["손절주기"] >= 30 and spec[n]["cand"] != "C4"
          and np.isfinite(out[n][w]["손익비"])]
    A(f"\n- 38개 전체: 주기 기대값 플러스(주기 10개 이상)는 앞 {cnt['2026앞']}, 뒤 {cnt['2026뒤']}, 2025 {cnt['2025']}, 세 구간 모두 {n_cons}개다. "
      f"2026 손절형(C4 제외, 주기 30개 이상) 승률은 {pp(min(wr))}~{pp(max(wr))}, 손익비는 {min(pr):.2f}~{max(pr):.2f}였다.")
    A("- 손익분기 손익비 = (1 − 승률)/승률이다. 승률 20%면 4.0, 30%면 2.33, 40%면 1.5가 필요하다. "
      "뒤 구간에 플러스가 많은 것은 7월 하락이 5주 이어져 판 뒤 더 싸게 되산 경우가 많았기 때문이다. 앞 구간의 V자 하락에서는 반대였다.\n")

    A("## 짝 비교\n")
    A("### C3 52주 고점 근처 재진입 vs DEEP 먼 자리 재매수 (같은 추적 20% 손절)\n")
    A("| 재진입 | 앞 수익 / 낙폭 (초과) | 뒤 수익 / 낙폭 | 2025 수익 (초과) | 2022 수익 | 뒤 밖에 있던 날(중앙) |")
    A("|---|---|---|---|---|---|")
    dj = P["ext"]["DEEP"]

    def prow(lab, ws, days=None):
        f, k, y5, y2 = ws["2026앞"], ws["2026뒤"], ws["2025"], ws["2022"]
        A(f"| {lab} | {pct(f['포트_수익'])} / {pct(f['포트_낙폭'])} ({pct(f['효율선초과'])}) | {pct(k['포트_수익'])} / {pct(k['포트_낙폭'])} "
          f"| {pct(y5['포트_수익'])} ({pct(y5['효율선초과'])}) | {pct(y2['포트_수익'])} | {days if days is not None else '—'} |")
    for n in ("C3_k85", "C3_k90", "C3_k95"):
        prow(f"C3 종가 ≥ 52주 고점×{spec[n]['k']:.2f}", out[n], f"{out[n]['2026뒤']['밖_일수_중앙']:.0f}일")
    prow("통제: 20일 돌파(= DEEP D1ctl_s20)", out["CTL_T20_brk"], f"{out['CTL_T20_brk']['2026뒤']['밖_일수_중앙']:.0f}일")
    for k, lab in (("D1_s20_d30_trail", "DEEP: 252일 고점 −30% 지정가"), ("D1_s20_d40_trail", "DEEP: −40% 지정가(사용자 아이디어)"),
                   ("D1_s20_d50_trail", "DEEP: −50% 지정가")):
        if k in dj:
            prow(lab, dj[k])
    prow("통제: 다시 사지 않음", out["CTL_T20_none"], f"{out['CTL_T20_none']['2026뒤']['밖_일수_중앙']:.0f}일")
    A(f"\n- 고점 근처까지 기다리면 뒤 구간 낙폭은 얕아졌다(−9~−12%, 돌파 −12.3%). 대신 늦게 사서 2025 효율선초과가 −1% 안팎이었다. "
      f"돌파 재진입은 {pct(out['CTL_T20_brk']['2025']['효율선초과'])}, DEEP −40%는 +17.1%였다. 앞 구간도 −1~−7%였다. "
      "George-Hwang의 '고점 근처 종목이 이긴다'를 재진입 신호로 옮기면, 2026처럼 빨리 회복하는 장에서는 너무 늦게 산다.\n")

    A("### C4 업종이 설명하는 −40% 하락만 매수 vs 조건 없는 −40% 매수 (매수 1회 손익, 60거래일 보유)\n")
    A("| 규칙 | 앞 n · 기대값 | 뒤 n · 승률 · 기대값 (매수 주 수) | 2025 n · 승률 · 기대값 (주 수) | 2022 n · 승률 · 기대값 | 2020 n · 기대값 | 노출 2026 |")
    A("|---|---|---|---|---|---|---|")

    def c4row(lab, ws):
        f, k, y5, y2, y0 = ws["2026앞"], ws["2026뒤"], ws["2025"], ws["2022"], ws["2020"]
        wk = lambda s: f" ({s.get('매수_주수', '—')}주)" if s["손절주기"] else ""
        A(f"| {lab} | {f['손절주기']} · {pct(f['기대값'])} | {k['손절주기']} · {pp(k['승률'])} · {pct(k['기대값'])}{wk(k)} "
          f"| {y5['손절주기']} · {pp(y5['승률'])} · {pct(y5['기대값'])}{wk(y5)} | {y2['손절주기']} · {pp(y2['승률'])} · {pct(y2['기대값'])} "
          f"| {y0['손절주기']} · {pct(y0['기대값'])} | {ws['2026']['평균노출']:.2f} |")
    for n in c4m:
        c4row(f"`{n}`", out[n])
    c4row("통제: 고유 조건 없음(종가 판단)", ctl4)
    if "D4_d40_60" in dj:
        d4 = dj["D4_d40_60"]
        A(f"| DEEP D4: −40% 지정가, 60일 | {d4['2026앞']['손절주기']} · {pct(d4['2026앞']['기대값'])} | {d4['2026뒤']['손절주기']} · {pp(d4['2026뒤']['승률'])} · {pct(d4['2026뒤']['기대값'])} "
          f"| {d4['2025']['손절주기']} · {pp(d4['2025']['승률'])} · {pct(d4['2025']['기대값'])} | {d4['2022']['손절주기']} · {pp(d4['2022']['승률'])} · {pct(d4['2022']['기대값'])} | — | — |")
    ce = [n for n in c4 if spec[n]["earn"]]
    red = [1 - out[n + "_e"][w]["손절주기"] / out[n][w]["손절주기"] for n in c4m for w in ("2026뒤", "2025", "2022")
           if out[n][w]["손절주기"] >= 5]
    A(f"\n- 실적일 제외(보조)를 더한 6개는 매수 수가 {pp(min(red))}~{pp(max(red))} 줄었고(매수 5번 이상인 창) 기대값은 비슷했다"
      f"(뒤 {rng(ce, '2026뒤', '기대값')}, 2025 {rng(ce, '2025', '기대값')}, 2022 {rng(ce, '2022', '기대값')}).")
    A("- 2026 뒤 구간 매수는 모두 7/17~9/15에 있고, 대부분 7/28~7/30 한 주다. 60일 보유를 마친 것은 0건이라 10/9 종가로 쟀다. "
      "2025 매수는 3/7~4/30에 몰렸다. 효과는 '업종 전체 급락 뒤 반등' 사건 한두 번이 만든 것이다. 2022처럼 1년 내내 빠진 해에는 매수가 20~38주에 흩어졌고 기대값이 0 근처였다.")
    A(f"- 앞 구간(1~5월)에는 −40% 종목이 82번 나왔다. 대부분 업종이 오르는데 혼자 빠진 하락이라 C4는 사지 않았다. 그런데 이 매수들도 60일 기대값 {pct(ctl4['2026앞']['기대값'])}로 좋았다. "
      "Chan(2003)의 '뉴스 있는 개별 하락은 이어진다'는 2026 앞 구간에서는 맞지 않았다.\n")

    A("### C6 나눠 다시 사기·불타기 vs 전량 재진입 (같은 2N 손절, LINE s8)\n")
    A("| 규칙 | 앞 (초과) | 뒤 수익 / 낙폭 | 2025 (초과) | 2022 수익 / 낙폭 | 노출 |")
    A("|---|---|---|---|---|---|")
    for n in ("C6_r1_m0.5", "C6_r1_m1", "CTL_C6_r1_full", "C6_r2_m0.5", "C6_r2_m1", "CTL_C6_r2_full"):
        ws = out[n]
        A(f"| `{n}` | {pct(ws['2026앞']['포트_수익'])} ({pct(ws['2026앞']['효율선초과'])}) | {pct(ws['2026뒤']['포트_수익'])} / {pct(ws['2026뒤']['포트_낙폭'])} "
          f"| {pct(ws['2025']['포트_수익'])} ({pct(ws['2025']['효율선초과'])}) | {pct(ws['2022']['포트_수익'])} / {pct(ws['2022']['포트_낙폭'])} | {ws['2026']['평균노출']:.2f} |")
    def dif(re, w, k):
        v = [out[f"C6_{re}_m{m:g}"][w][k] - out[f"CTL_C6_{re}_full"][w][k] for m in C6_M]
        return f"{min(v) * 100:+.1f}~{max(v) * 100:+.1f}%p"
    A(f"\n- 절반 재진입 − 전량 재진입: r2(돌파)는 뒤 수익 {dif('r2', '2026뒤', '포트_수익')}, 2022 수익 {dif('r2', '2022', '포트_수익')}, "
      f"2025 효율선초과 {dif('r2', '2025', '효율선초과')}였다. r1(선 회복)은 뒤 {dif('r1', '2026뒤', '포트_수익')}, 2022 {dif('r1', '2022', '포트_수익')}, "
      f"2025 {dif('r1', '2025', '효율선초과')}였다. 돌파 재진입에서는 노출을 줄인 만큼 빠진 장의 손실이 줄고 오른 장을 덜 탔다. "
      "손절 주기의 모양(낮은 승률, 손익비 2 안팎)은 바뀌지 않았다.\n")

    A("### C7 공황 필터, C1·C5 한 줄씩\n")
    pdn = P["panic_days"]
    A(f"- C7: 공황일은 2022 {pdn['2022']}일, 2023 {pdn['2023']}일이고 2018~2021·2024~2026은 0일이었다. 그래서 2026·2025 결과는 끔과 켬이 같다. "
      f"SMA200 손절에서 켜면 2023 {pct(out['C7_sma200_on']['2023']['포트_수익'])}(끔 {pct(out['C7_sma200_off']['2023']['포트_수익'])}, "
      f"매수·보유 {pct(bh['2023']['포트BH_수익'])})였다. 약세장 뒤 반등을 덜 놓친 것으로 문헌과 같은 방향이다. 2022는 "
      f"{pct(out['C7_sma200_on']['2022']['포트_수익'])} 대 {pct(out['C7_sma200_off']['2022']['포트_수익'])}로 거의 같았다.")
    A(f"- C1(Kaminski-Lo 긴 창): 6개 모두 앞 효율선초과가 플러스였다({rng([n for n in names if spec[n]['cand'] == 'C1'], '2026앞', '효율선초과')}). "
      f"그러나 뒤 수익은 {rng([n for n in names if spec[n]['cand'] == 'C1'], '2026뒤', '포트_수익')}로 매수·보유({pct(bh['2026뒤']['포트BH_수익'])})보다 나빴다. "
      f"7월 쇼크도 {rng([n for n in names if spec[n]['cand'] == 'C1'], '2026', 'AI쇼크_수익')}로 막지 못했다. 60~120일 창은 5주짜리 하락에 늦게 걸렸고, '20일 수익 > 0' 재진입은 반등 초입에 다시 샀다.")
    A(f"- C5: 거래량 마름 재진입은 앞 효율선초과 {rng([n for n in names if spec[n]['cand'] == 'C5'], '2026앞', '효율선초과')}로 가장 나빴다. "
      "VCP 완전판은 2026 손절 135번 중 14번만 다시 샀다(2025 149번 중 37번). 사실상 '다시 사지 않음'과 같다(앞 −19.5% 대 −19.3%).\n")

    A("## 7종목 경로 위 실험 (C8 위험 균등 비중, Han-Zhou-Zhu 7종목판)\n")
    A("| 변형 | 2026 전체 | 앞 수익 (초과) | 뒤 수익 / 낙폭 (초과) | AI 쇼크 | 2025 수익 / 낙폭 (초과) | 2024 (초과) | 2026 손절 수 | 2026 주기 n · 승률 · 손익비 · 기대값 |")
    A("|---|---|---|---|---|---|---|---|---|")

    def portrow(lab, row, slipped=False):
        m26, mh = row["metrics"]["2026"], row["metrics"]["hold"]
        cy = (row.get("cycles") or {}).get("2026", {}).get("전체")
        cys = f"{cy['손절주기']} · {pp(cy['승률'])} · {rr(cy['손익비'])} · {pct(cy['기대값'])}" if cy and cy["손절주기"] else "—"
        A(f"| {lab} | {pct(m26['전체']['수익'])} / {pct(m26['전체']['낙폭'])} | {pct(m26['앞']['수익'])} ({pct(m26['앞']['효율선초과'])}) "
          f"| {pct(m26['뒤']['수익'])} / {pct(m26['뒤']['낙폭'])} ({pct(m26['뒤']['효율선초과'])}) | {pct(m26['AI쇼크']['수익'])} "
          f"| {pct(mh['2025']['수익'])} / {pct(mh['2025']['낙폭'])} ({pct(mh['2025']['효율선초과'])}) | {pct(mh['2024']['수익'])} ({pct(mh['2024']['효율선초과'])}) "
          f"| {'—' if slipped else row['stops']['2026']} | {'—' if slipped else cys} |")
    for nm, row in port.items():
        portrow(PORT_LAB[nm], row)
    for nm in ("C8_risk_2N", "HZZ7_month"):     # C8_risk는 손절이 없어 미끄러짐(손절 체결에만 붙음) 영향이 0이라 뺀다
        portrow(PORT_LAB[nm] + ", 미끄러짐 0.5%", pslip[nm], True)
    for nm, r in P["B_rows"].items():
        A(f"| 1단계 B: {nm} | {pct(r['2026_전체_수익'])} / {pct(r['2026_전체_낙폭'])} | {pct(r['2026_앞_수익'])} | "
          f"{pct(r['2026_뒤_수익'])} / {pct(r['2026_뒤_낙폭'])} | {pct(r['2026_AI쇼크_수익'])} | {pct(r['hold_2025_수익'])} / {pct(r['hold_2025_낙폭'])} "
          f"| {pct(r['hold_2024_수익'])} | — | — |")
    c8 = port["C8_risk"]["metrics"]
    he, hw = port["HZZ7_entry"], port["HZZ7_week"]
    hec = he["cycles"]["2026"]["전체"]
    A("")
    A(f"- **C8 위험 균등 비중**: 앞 낙폭이 {pct(base26['앞']['낙폭'])} → {pct(c8['2026']['앞']['낙폭'])}로 얕아졌고 효율선초과는 {pct(c8['2026']['앞']['효율선초과'])}였다. "
      f"2025도 {pct(c8['hold']['2025']['효율선초과'])}였다. 그러나 뒤는 {pct(c8['2026']['뒤']['효율선초과'])}, 7월 쇼크는 {pct(c8['2026']['AI쇼크']['수익'])}로 기본과 같았다. "
      f"2026 전체는 {pct(c8['2026']['전체']['수익'])}로 기본 {pct(base26['전체']['수익'])}보다 낮고 낙폭은 같았다(효율선초과 {pct(c8['2026']['전체']['효율선초과'])}). "
      f"2024도 {pct(c8['hold']['2024']['효율선초과'])}였다. 변동성이 큰 종목(메모리, BE 등) 비중을 줄여도, 7월처럼 7종목이 함께 빠지는 하락은 막지 못했다. 손절은 주지 않고 비중만 바꾸는 규칙이다. "
      "엔진은 미끄러짐을 손절 체결에만 붙이므로 손절 없는 C8에는 0.5% 민감도가 0이다(재조정 체결 비용은 기본과 같게 0).")
    A(f"- **Han-Zhou-Zhu와 우리 결과가 다른 이유.** 원문은 세 가지가 1단계 B와 다르다. 기준가를 매달 새로 잡고(월초 가격 대비 −10%), 손절되면 그 달 끝까지 현금이며, "
      f"표본이 1926~2013년 88년이다. 그 기간 승자 10분위의 큰 손실은 몇 달씩 이어진 하락(1930년대, 2000~02, 2008~09)에 몰려 있었다. "
      f"1단계 B식 번역(산 가격 대비 −10%, 다음 월요일 재매수)은 크게 오른 종목에서는 손절선이 너무 아래에 있어 2026에 {he['stops']['2026']}번만 걸렸다. "
      f"걸리면 장중 저점 근처에서 팔고 다음 월요일에 더 비싸게 되샀다(주기 승률 {pp(hec['승률'])}, 손익비 {rr(hec['손익비'])}, 기대값 {pct(hec['기대값'])}). "
      f"기준가를 매주 다시 잡으면 {hw['stops']['2026']}번 걸려 2026 수익이 {pct(hw['metrics']['2026']['전체']['수익'])}로 무너졌다. "
      f"원문에 가장 가까운 '월 기준가 + 다음 달까지 현금'판만 5주 넘게 이어진 7월 하락에서 효과가 났다(쇼크 {pct(base26['AI쇼크']['수익'])} → {pct(hm26['AI쇼크']['수익'])}; "
      f"뒤 주기 승률 {pp(hmc['뒤']['승률'])}, 손익비 {rr(hmc['뒤']['손익비'])}, 기대값 {pct(hmc['뒤']['기대값'])}). "
      f"그러나 1~5월처럼 1~2주 만에 되돌린 하락에서는 졌다(앞 주기 기대값 {pct(hmc['앞']['기대값'])}, 앞 효율선초과 {pct(hm26['앞']['효율선초과'])}). "
      f"2025 효율선초과도 {pct(hmh['2025']['효율선초과'])}였다. 미끄러짐 0.5%를 넣으면 2026 전체 효율선초과가 {pct(hm26['전체']['효율선초과'])} → "
      f"{pct(pslip['HZZ7_month']['metrics']['2026']['전체']['효율선초과'])}로 뒤집힌다. 문헌의 이득은 '한 달 넘게 이어지는 하락'이 표본에 많아서 나온 것이다. "
      "2026처럼 되돌림이 빠른 장에서는 보험료(놓친 반등)가 더 컸다. 원문도 롱숏 개선의 대부분이 패자(공매도) 쪽에서 나왔다고 적는다. "
      "유니버스 C2도 같은 모양이었다(무조건 재매수판: 뒤 +3.1%, 쇼크 −7.3%, 앞 효율선초과 −6.6%). 상위 10% 조건을 붙이면 손절된 종목을 거의 다시 사지 않아 노출이 0.2~0.6으로 줄었다.\n")

    A("## 맥락: 2018~2024 (장세 전환 위험)\n")
    ctx = P["context"]
    yb = ", ".join(f"{y} {pct(bh[y]['포트BH_수익'], 0)}" for y in CONTEXT_YEARS)
    A(f"매수·보유: {yb}. 2018·2019는 QUBT 자료 이상을 빼서 다른 갈래(2018 +30.2%)와 다르다.\n")
    A("| 해 | " + " | ".join(CONTEXT_YEARS) + " |")
    A("|---|" + "---|" * len(CONTEXT_YEARS))
    A("| 효율선초과 + (38개 중) | " + " | ".join(str(sum(1 for n in names if ctx[n][y] > 0)) for y in CONTEXT_YEARS) + " |")
    A("| 매수·보유 수익 이김 (38개 중) | " + " | ".join(str(sum(1 for n in names if out[n][y]["포트_수익"] > bh[y]["포트BH_수익"]))
                                          for y in CONTEXT_YEARS) + " |")
    A(f"\n- 매수·보유를 이긴 해는 떨어진 해(2018, 2022)뿐이다. 오른 해(2019~2021, 2023~2024)에는 38개 중 0~1개만 이겼다. 장세가 2022처럼 바뀌면 손절이 손실을 줄인다. "
      f"예: SMA200 {pct(out['C7_sma200_off']['2022']['포트_수익'])}(매수·보유 {pct(bh['2022']['포트BH_수익'])}). 지금 장세에서는 비용이 더 컸다.\n")

    A("## 선택편향·한계\n")
    lim = [
        "- 시험 범위: 유니버스 38개(문헌 값 32 + 이웃 6)와 통제 5개, 7종목 6개다. 문헌 모수를 그대로 써서 이 자료로 고른 것은 아니다. 하지만 38개 중 앞 1등을 고르면 앞 수치는 위로 치우친다. 뒤·2025를 본 뒤 격자·정의를 바꾸지 않았다.",
        f"- 생존편향: 종목 목록이 2026-10 기준(+TIME 보유 이력)이다. C4처럼 깊이 빠진 종목을 사는 규칙은 실제보다 좋게 나온다(2025 매수 1회 기대값 {rng(c4m, '2025', '기대값', 0)}). 망해서 빠진 종목에서 손절이 막아 줬을 손실도 빠져 있다.",
        "- C4 표본은 건수보다 훨씬 작다. 2026 뒤는 사실상 7월 말 한 주이고 60일을 못 채웠다. 2025는 3~4월 한 사건이다. '업종 전체 급락 뒤 V자 반등'이 두 번 있었다는 뜻에 가깝다.",
        f"- 실적일은 yfinance 발표일(지난 날짜)이다. {', '.join(P['earn_missing']) or '없음'}은 실적일이 없어 보조 조건이 걸리지 않는다.",
        "- C2 월 P0(첫 거래일 시가)은 시가를 본 뒤 손절 주문을 건다(KST 22:30, 주문 가능 시간 안). 전월 말 종가판(이웃)은 밤사이 걸 수 있고 결과는 비슷했다.",
        "- C5 스윙 정의(앞뒤 5일 국소 고·저, 같은 종류가 이어지면 더 극단값만 남김)와 C6의 '나머지 1/2'(현금 전부)은 문헌 문장을 구현하며 정한 것이다.",
        "- HZZ 7종목 주기 = 손절가 / 다시 살 수 있는 첫 재조정 시가 − 1이다(그 주에 목록에서 빠졌어도 기본 경로도 그날 팔았으므로 같은 비교). 7종목 2025는 2023-05부터 이어 돌린 보유 기간 경로를 자른 값이다(미분류 종목 때문에 반도체 한도가 과하게 걸림, 1단계와 같음).",
        "- 버그 수정 2건(첫 실행 뒤, 격자·정의 그대로): ① 앞 구간 거래 0회인 C4가 칼마 0/0 = 무한대로 G3에 뽑혀 → 낙폭 0·수익 0은 고르기에서 뺌. "
        "② 7종목 효율선 비교 허용치 1e-12 때문에 기본 자신이 효율선초과 +1.6%로 나와 → 1e-6으로 고침. 자료 이상 1건: QUBT 2018-07-03 −99.5%, 07-06 +2,033%(장외 껍데기 시절)가 2018 동일가중 포트를 혼자 +22.6% 올려 2018·2019에서 뺐다.",
        "- 점검(`py p2_LITTEST.py check`): 매수 스탑이 없는 정책에서 확장 엔진 = 원본(차 0). t 뒤 가격·거래량을 흔들어도 t까지 가치 차 0(2026에 −40% 넘게 빠진 8종목에서 C2·C4·C5·C6도 0). "
        "통제 3개가 LINE s8·r1(2026 +49.61%, 주기 240), s8·r2(+45.94%, 242), s4·r1(+46.43%, 376)을 그대로 재현했다. 통제 '20일 돌파'는 DEEP D1ctl_s20과 같은 숫자였다.",
        "- C8 상한은 기본 경로를 만들 때 고정한 섹터 표(cache/base_*.json)로 계산했다. 이 작업 중 다른 커밋(d5ce794)이 watchlist의 미분류 56종목을 분류했다. "
        "지금 표로 계산하면 2026 +200.6%, 앞 효율선초과 +14.5%, 2025 +68.0%(효율선초과 +28.5%)로 방향은 같다.",
        "- 수수료·현금 이자 0, 정규장 고가·저가 기준 스탑이다. 프리·애프터 체결과 스탑 지연은 넣지 않았다. 미끄러짐은 고른 변형과 후보별 앞 1등만 0.5%로 다시 쟀다.",
    ]
    for line in lim:
        A(line)
    (SL.RESULTS / f"{ID}.md").write_text("\n".join(Lx) + "\n", encoding="utf-8")
    print("md", SL.RESULTS / f"{ID}.md")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "fetch":
        fetch()
    elif arg == "check":
        check()
    elif arg == "md":
        write_md(json.loads((SL.RESULTS / f"{ID}.json").read_text(encoding="utf-8")))
    else:
        main()
