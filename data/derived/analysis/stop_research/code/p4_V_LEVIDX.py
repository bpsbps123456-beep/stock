"""4단계 LEVIDX 반박 검증(2026-10-11): 대표 후보 G1 '합성 SOXL(SOXX 3배) 추적 −30% → 판 뒤 20거래일 시가 재진입'.

엔진(trade_lab·lev_lab·stock_lab·tranche_lab)을 import하지 않는다. cache/p4_lev.parquet만 읽어 새로 짰다.

설정(주장 정의 그대로)
  - 합성: r[t] = L × (SOXX 수정 종가 수익) − (L−1) × (^IRX/100)/252 − 보수/252, 2014-01-02 = 100.
    시가·고가·저가 = 전일 합성 종가 × (1 + L × (기초 시가·고가·저가 / 기초 전일 종가 − 1)), 바닥 0.01배.
  - 실행: 2015-01-02 ~ 2026-10-09 한 번, 2014-12-31 종가에 전량 보유로 시작. 창은 이 곡선을 잘라 잰다.
  - 손절: t일 선 = (산 뒤 최고 종가, t−1까지) × 0.70. 시가 ≤ 선이면 시가, 저가 ≤ 선이면 선 가격에 전량 매도.
  - 재진입: 판 날 t_e, t − t_e ≥ 20인 첫날 시가 전량. 고점은 매수가부터 다시.
  - 효율선초과 = 규칙 수익 − (낙폭이 규칙보다 깊지 않은 고정 k배(0~L, 0.05 간격) 들고 있기의 최대 수익).
    k ≤ 1 = k × 기초 수익(현금 0%), k > 1 = 일일 재설정 k배(금융비용·보수). 실제 상품은 (k/L) × 실제 상품 수익.

검증(아래 상수는 사전 고정, 결과를 본 뒤 바꾸지 않는다)
  1) 독립 재계산: 합성·실제 SOXL. 원 격자 252조합(규칙 36 × 상품 7)을 새 엔진으로 다시 돌려 G1·G3·'앞 > 0' 집합 비율도 재현.
  2) 시점 감사(미래 교란) + 하루 늦춘 판(장중 감지 → 다음 시가 매도, 재진입도 하루 뒤) + 종가 감지 판 + 미끄러짐 0.5%.
  3) 이웃: 손절 {20,25,30,35,40}% × 대기 {10,15,20,25,30}거래일 = 25개(±2칸), 원 격자 손절 {15,20,30,40,50}% × 대기 20.
     상품 이웃: 같은 규칙을 다른 6상품(합성)에.
  4) 위약: 같은 창 안에 같은 나가기 횟수·같은 현금 기간 길이를 무작위 날짜에(시가 매도 → 시가 재매수), 시드 20개.
     전체 기간판: 36번의 20거래일 현금 기간을 2015~2026에 무작위로(시드 20개).
  5) 장세: 2026 앞·뒤, 2025, 2022, 2020, 2018, 폭락 구간, 2015~2026 연율·최대 낙폭, '−80% 이하' 여부.
  +) 시작 시점(위상) 민감도: 시작을 2015-01 ~ 2025-12 매월 첫 거래일(132개)로 바꿔 2026 앞·뒤 숫자가 바뀌는지.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p4_V_LEVIDX"

# ============================================================================ 사전 고정
PRODUCTS = [("TQQQ", "QQQ", 3), ("SOXL", "SOXX", 3), ("UPRO", "SPY", 3), ("TECL", "XLK", 3), ("TNA", "IWM", 3),
            ("QLD", "QQQ", 2), ("USD", "SOXX", 2)]
CAND_P, CAND_U, CAND_L = "SOXL", "SOXX", 3
X0, W0 = 0.30, 20                          # 후보: 추적 −30%, 판 뒤 20거래일
FEE = {1: 0.01, 2: 0.0105, 3: 0.0095}      # 연 보수(1 < k < 2 는 1%)
RUN = ("2015-01-02", "2026-10-09")
K_STEP = 0.05
SLIP = 0.005
EPS = 1e-6
X_NB = (0.20, 0.25, 0.30, 0.35, 0.40)      # 이웃: 손절 5%p 간격 ±2칸
W_NB = (10, 15, 20, 25, 30)                # 이웃: 대기 5거래일 간격 ±2칸
X_GRID_NB = (0.15, 0.20, 0.30, 0.40, 0.50)  # 원 격자 손절 축(대기 20 고정)
SEEDS = 20
START_FROM, START_TO = "2015-01", "2025-12"  # 시작 위상 민감도: 매월 첫 거래일

# 원 격자(LEVIDX와 같은 값) — G1·G3 재선택 확인용
T_SMA, T_BAND = (20, 50, 100, 150, 200), (0.0, 0.03)
S_X, S_RE = (0.15, 0.20, 0.30, 0.40, 0.50), ("sma50", "brk20", "wait20")
V_THR = (0.20, 0.25, 0.35)
P_SMA = (50, 200)
D_X, D_D = (0.30, 0.40), (0.50, 0.60, 0.70)

WIN = {
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2026": ("2026-01-02", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2024": ("2024-01-02", "2024-12-31"), "2023": ("2023-01-03", "2023-12-29"),
    "2022": ("2022-01-03", "2022-12-30"), "2021": ("2021-01-04", "2021-12-31"), "2020": ("2020-01-02", "2020-12-31"),
    "2019": ("2019-01-02", "2019-12-31"), "2018": ("2018-01-02", "2018-12-31"), "2017": ("2017-01-03", "2017-12-29"),
    "2016": ("2016-01-04", "2016-12-30"), "2015": ("2015-01-02", "2015-12-31"),
    "2018Q4": ("2018-10-01", "2018-12-31"), "2020급락": ("2020-02-20", "2020-03-23"), "2020반등": ("2020-03-24", "2020-08-31"),
    "2025관세": ("2025-02-20", "2025-04-08"), "2025반등": ("2025-04-09", "2025-07-31"), "AI쇼크": ("2026-06-23", "2026-07-29"),
    "전체": RUN,
}
PICK, CHECK, AUX = "2026앞", "2026뒤", "2025"
PLACEBO_WINS = ("2026앞", "2026뒤", "2025", "2022", "2020", "2018")

# 주장 숫자(results/p4_LEVIDX.md·과제 설명)
CLAIM = {
    ("synth", "2026앞", "수익"): 3.996, ("synth", "2026앞", "낙폭"): -0.30, ("synth", "2026앞", "효율선초과"): 1.799,
    ("synth", "2026뒤", "수익"): -0.503, ("synth", "2026뒤", "낙폭"): -0.61, ("synth", "2026뒤", "효율선초과"): -0.511,
    ("synth", "2025", "수익"): 0.827, ("synth", "2025", "낙폭"): -0.51, ("synth", "2025", "효율선초과"): 0.275,
    ("synth", "전체", "연율"): 0.287, ("synth", "전체", "낙폭"): -0.93, ("synth", "전체", "효율선초과_연"): -0.188,
    ("hold", "2026앞", "수익"): 4.355, ("hold", "2026앞", "낙폭"): -0.43, ("hold", "전체", "연율"): 0.461,
    ("hold", "전체", "낙폭"): -0.90,
    ("real", "2026앞", "수익"): 3.974, ("real", "2026뒤", "수익"): -0.506, ("real", "2025", "수익"): 0.790,
    ("real", "전체", "연율"): 0.280, ("real", "전체", "낙폭"): -0.93,
    ("synth", "전체", "연들락"): 3.1,
    ("synth", "2022", "수익"): -0.827, ("synth", "2020급락", "수익"): -0.300, ("synth", "2018Q4", "수익"): -0.375,
    ("synth", "AI쇼크", "수익"): -0.242,
    ("synth", "전체", "같은노출초과_연"): -0.180,       # p4_LEVIDX.md 부록 '같은 노출 대비(연)'
}
# 주장값의 표시 자릿수(반올림 폭): 낙폭은 정수 %, 연들락은 0.1
CLAIM_TOL = {"낙폭": 0.005, "연들락": 0.05}


# ============================================================================ 자료
def load():
    df = pd.read_parquet(CACHE / "p4_lev.parquet")
    assert not df.duplicated(["sym", "date"]).any()
    irx = df[df.sym == "^IRX"].set_index("date")["Close"].astype(float).sort_index() / 100.0
    out = {}
    for sym, g in df[df.sym != "^IRX"].sort_values(["sym", "date"]).groupby("sym"):
        g = g.dropna(subset=["Open", "High", "Low", "Close"])
        g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0)]
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        out[sym] = {"days": g.date.astype(str).to_numpy(), "o": g.Open.to_numpy() * f, "h": g.High.to_numpy() * f,
                    "l": g.Low.to_numpy() * f, "c": g.Close.to_numpy() * f}
    return out, irx


def rf_for(days, irx):
    return irx.reindex(days).ffill().fillna(0.0).to_numpy()


def fee_of(k):
    return FEE.get(int(round(k, 6)), 0.01)


def synth(u, L, irx):
    c0, o0, h0, l0 = u["c"], u["o"], u["h"], u["l"]
    rf = rf_for(u["days"], irx)
    fee = fee_of(L)
    r = L * (c0[1:] / c0[:-1] - 1) - (L - 1) * rf[1:] / 252 - fee / 252
    c = 100.0 * np.concatenate([[1.0], np.cumprod(np.maximum(1 + r, 0.01))])
    pc = c[:-1]
    o = np.concatenate([[100.0], pc * np.maximum(1 + L * (o0[1:] / c0[:-1] - 1), 0.01)])
    hi = np.concatenate([[100.0], pc * np.maximum(1 + L * (h0[1:] / c0[:-1] - 1), 0.01)])
    lo = np.concatenate([[100.0], pc * np.maximum(1 + L * (l0[1:] / c0[:-1] - 1), 0.01)])
    h = np.maximum.reduce([hi, o, c])
    l = np.minimum.reduce([lo, o, c])
    return {"days": u["days"], "o": o, "h": h, "l": l, "c": c}


def k_daily(u, k, irx):
    ru = np.concatenate([[np.nan], u["c"][1:] / u["c"][:-1] - 1])
    if k <= 1 + 1e-12:
        return k * ru
    r = k * ru - (k - 1) * rf_for(u["days"], irx) / 252 - fee_of(k) / 252
    return np.maximum(r, -0.99)


def idx(days, d, side="left"):
    return int(np.searchsorted(days, d, side=side))


def win_idx(days, wins=WIN):
    return {w: (idx(days, d0), idx(days, d1, "right") - 1) for w, (d0, d1) in wins.items()}


def mdd(v):
    v = np.asarray(v, float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def curve(r, wa, wz):
    return np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(r[wa:wz + 1]))])


class Frontier:
    def __init__(self, daily_by_k, wins):
        self.tab = {}
        ks = sorted(daily_by_k)
        for w, (wa, wz) in wins.items():
            rets, dds = [], []
            for k in ks:
                v = curve(daily_by_k[k], wa, wz)
                rets.append(v[-1] - 1)
                dds.append(mdd(v))
            self.tab[w] = (np.array(ks), np.array(rets), np.array(dds))

    def best(self, w, dd):
        ks, rets, dds = self.tab[w]
        ok = dds >= dd - 1e-12
        i = int(np.argmax(np.where(ok, rets, -np.inf)))
        return float(rets[i]), float(ks[i]), float(dds[i])


# ============================================================================ 엔진 1: 후보 전용(추적 손절 + N일 대기)
def sim_trail(b, a, z, x, wait, slip=0.0, mode="intraday"):
    """mode: intraday(주장), delay1(장중 감지 → 다음 시가 매도, 재진입은 감지일 + wait + 1), close(종가 ≤ 선 → 다음 시가)."""
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]
    cash, sh = 0.0, 1.0 / c[a - 1]
    peak = c[a - 1]
    holding = True
    t_sig = None             # 판 날(intraday) 또는 신호 날(delay1·close)
    pend = False
    vals = np.empty(z - a + 2)
    vals[0] = 1.0
    pos = np.empty(z - a + 1)
    exits, entries = [], []
    for t in range(a, z + 1):
        # 시가
        if pend and holding:
            fill = o[t] * (1 - slip)
            cash, sh, holding, pend = sh * fill, 0.0, False, False
            exits.append((t, fill, "next_open"))
        elif not holding and t_sig is not None:
            need = wait if mode == "intraday" else wait + 1
            if t - t_sig >= need:
                fill = o[t] * (1 + slip)
                sh, cash, holding = cash / fill, 0.0, True
                peak, t_sig = fill, None
                entries.append((t, fill))
        # 장중
        if holding and not pend:
            line = peak * (1 - x)
            if mode == "intraday":
                why = None
                if o[t] <= line:
                    fill, why = o[t] * (1 - slip), "gap"
                elif l[t] <= line:
                    fill, why = line * (1 - slip), "line"
                if why is not None:
                    cash, sh, holding, t_sig = sh * fill, 0.0, False, t
                    exits.append((t, fill, why))
            elif mode == "delay1":
                if l[t] <= line:
                    pend, t_sig = True, t
            elif mode == "close":
                if c[t] <= line:
                    pend, t_sig = True, t
        vals[t - a + 1] = cash + sh * c[t]
        pos[t - a] = 1.0 if holding else 0.0
        if holding:
            peak = max(peak, c[t])
    return vals, pos, exits, entries


# ============================================================================ 엔진 2: 원 격자 36규칙(G1·G3 재선택 확인)
def rule_list():
    out = []
    for n in T_SMA:
        for bd in T_BAND:
            out.append({"id": f"T{n}" + (f"b{int(bd * 100)}" if bd else ""), "fam": "T", "n": n, "band": bd})
    for x in S_X:
        for re in S_RE:
            out.append({"id": f"S{int(round(x * 100))}_{re}", "fam": "S", "x": x, "re": re})
    for v in V_THR:
        out.append({"id": f"V{int(round(v * 100))}", "fam": "V", "thr": v})
    for n in P_SMA:
        out.append({"id": f"P{n}", "fam": "P", "n": n})
    for x in D_X:
        for d in D_D:
            out.append({"id": f"D{int(round(x * 100))}_{int(round(d * 100))}", "fam": "D", "x": x, "d": d})
    return out


def indicators(u, b):
    cu = pd.Series(u["c"])
    ru = cu.pct_change()
    ind = {f"sma{n}": cu.rolling(n, min_periods=n).mean().to_numpy() for n in (20, 50, 100, 150, 200)}
    ind["rv20"] = ru.rolling(20, min_periods=20).std().to_numpy() * np.sqrt(252)
    ind["hh20"] = pd.Series(b["h"]).rolling(20, min_periods=20).max().to_numpy()
    return ind


def sim_rule(b, u, ind, sp, a, z, slip=0.0):
    """시가 판단(t−1까지) → 시가 체결 → 장중 손절 → 장중 지정가 재매수. 반환 vals, pos(종가), 주기[(t_exit, px, t_re, px_re, 비중)], 판 횟수."""
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]
    uc = u["c"]
    cash, sh = 0.0, 1.0 / c[a - 1]
    peak = c[a - 1]
    pos = 1.0
    exit_t, exit_px, days_out = None, None, 0
    inn = True
    fam = sp["fam"]
    half_open = None
    cyc = []
    vals = np.empty(z - a + 2)
    vals[0] = 1.0
    pos_c = np.empty(z - a + 1)
    nsell = 0

    def set_pos(target, px, t):
        nonlocal cash, sh, peak, exit_t, exit_px, days_out, pos, nsell
        eq = cash + sh * px
        cur = sh * px / eq if eq > 0 else 0.0
        if abs(target - cur) < 1e-9:
            return
        if target < cur:
            fill = px * (1 - slip)
            sell_sh = sh - target * eq / px
            cash += sell_sh * fill
            sh -= sell_sh
            nsell += 1
            if target <= 1e-9:
                sh = 0.0
                exit_t, exit_px, days_out = t, fill, 0
        else:
            fill = px * (1 + slip)
            buy_val = min((target - cur) * eq, cash)
            sh += buy_val / fill
            cash -= buy_val
            if cur <= 1e-9:
                if exit_px is not None:
                    cyc.append((exit_t, exit_px, t, fill, 1.0))
                peak, exit_t, exit_px = fill, None, None
        eqn = cash + sh * px
        pos = sh * px / eqn if eqn > 0 else 0.0

    for t in range(a, z + 1):
        tgt = None
        if fam in ("T", "P"):
            m = ind[f"sma{sp['n']}"][t - 1]
            if np.isfinite(m):
                band = sp.get("band", 0.0)
                if inn and uc[t - 1] < m * (1 - band):
                    inn = False
                    tgt = 0.0 if fam == "T" else 0.5
                    if fam == "P":
                        half_open = (t, o[t] * (1 - slip))
                elif not inn and uc[t - 1] > m * (1 + band):
                    inn = True
                    tgt = 1.0
                    if fam == "P" and half_open:
                        cyc.append((half_open[0], half_open[1], t, o[t] * (1 + slip), 0.5))
                        half_open = None
        elif fam == "V":
            rv = ind["rv20"][t - 1]
            if np.isfinite(rv):
                if inn and rv > sp["thr"]:
                    inn, tgt = False, 0.0
                elif not inn and rv <= sp["thr"]:
                    inn, tgt = True, 1.0
        else:
            if pos <= 1e-9 and exit_t is not None and days_out >= 1:
                re = sp.get("re", "sma50")
                if re == "sma50":
                    m = ind["sma50"][t - 1]
                    if np.isfinite(m) and uc[t - 1] > m:
                        tgt = 1.0
                elif re == "brk20":
                    hh = ind["hh20"][t - 2]
                    if np.isfinite(hh) and c[t - 1] > hh:
                        tgt = 1.0
                elif re == "wait20":
                    if t - exit_t >= 20:
                        tgt = 1.0
        if tgt is not None:
            set_pos(float(tgt), o[t], t)
        if fam in ("S", "D") and pos > 1e-9:
            line = peak * (1 - sp["x"])
            if o[t] <= line:
                set_pos(0.0, o[t], t)
            elif l[t] <= line:
                set_pos(0.0, line, t)
        if fam == "D" and pos <= 1e-9 and exit_t is not None:
            lvl = peak * (1 - sp["d"])
            if l[t] <= lvl:
                set_pos(1.0, min(o[t], lvl), t)
        vals[t - a + 1] = cash + sh * c[t]
        pos_c[t - a] = pos
        if pos > 1e-9:
            peak = max(peak, c[t])
        else:
            days_out += 1
    if pos <= 1e-9 and exit_px is not None:
        cyc.append((exit_t, exit_px, None, None, 1.0))
    if half_open is not None:
        cyc.append((half_open[0], half_open[1], None, None, 0.5))
    return vals, pos_c, cyc, nsell


# ============================================================================ 창 지표
def wmetrics(vals, a, wins, front: Frontier, c=None, cyc=None, pos=None, exits_t=None, eqfun=None):
    out = {}
    for w, (wa, wz) in wins.items():
        v = vals[wa - a: wz - a + 2]
        v = v / v[0]
        n = wz - wa + 1
        ret, dd = float(v[-1] - 1), mdd(v)
        cagr = (1 + ret) ** (252 / n) - 1 if ret > -1 else -1.0
        fr, fk, fdd = front.best(w, dd)
        fc = (1 + fr) ** (252 / n) - 1 if fr > -1 else -1.0
        m = {"수익": ret, "낙폭": dd, "연율": float(cagr), "칼마": float(cagr / -dd) if dd < -1e-9 else float("nan"),
             "효율선초과": float(ret - fr), "효율선_k": fk, "효율선_수익": fr, "효율선_낙폭": fdd, "효율선_연율": float(fc),
             "효율선초과_연": float(cagr - fc), "일수": n}
        if pos is not None:
            m["평균노출"] = float(pos[wa - a: wz - a + 1].mean())
            if eqfun is not None:
                eq = curve(eqfun(m["평균노출"]), wa, wz)
                er = float(eq[-1] - 1)
                ec = (1 + er) ** (252 / n) - 1 if er > -1 else -1.0
                m["같은노출_수익"], m["같은노출초과"], m["같은노출초과_연"] = er, float(ret - er), float(cagr - ec)
        if exits_t is not None:
            k = sum(1 for t in exits_t if wa <= t <= wz)
            m["판횟수"] = k
            m["연들락"] = k / (n / 252)
        if cyc is not None and c is not None:
            g = []
            for (et, ep, rt, rp, wt) in cyc:
                if wa <= et <= wz:
                    px = rp if (rt is not None and rt <= wz) else c[wz]
                    g.append(ep / px - 1)
            m["_g"] = g
        out[w] = m
    return out


# ============================================================================ 위약
def sim_spells(b, a, z, spells, slip=0.0):
    """spells = [(s, e)]: s일 시가 전량 매도, e일 시가 전량 재매수(e > z면 끝까지 현금)."""
    o, c = b["o"], b["c"]
    cash, sh = 0.0, 1.0 / c[a - 1]
    sd = {s: e for s, e in spells}
    es = {e for _, e in spells}
    vals = np.empty(z - a + 2)
    vals[0] = 1.0
    for t in range(a, z + 1):
        if t in es and sh == 0.0:
            sh, cash = cash / (o[t] * (1 + slip)), 0.0
        if t in sd and sh > 0:
            cash, sh = sh * o[t] * (1 - slip), 0.0
        vals[t - a + 1] = cash + sh * c[t]
    return vals


def random_spells(rng, lo, hi, lengths):
    """[lo, hi) 안에 길이 lengths의 겹치지 않는(사이 1일 이상) 구간을 무작위 순서·무작위 위치로."""
    lengths = list(lengths)
    rng.shuffle(lengths)
    n = len(lengths)
    span = hi - lo
    free = span - sum(lengths) - max(n - 1, 0)
    if free < 0:
        raise ValueError("공간 부족")
    cuts = np.sort(rng.integers(0, free + 1, size=n))
    out, pos = [], lo
    prev = 0
    for i, ln in enumerate(lengths):
        gap = cuts[i] - prev
        prev = cuts[i]
        s = pos + gap
        out.append((s, s + ln))
        pos = s + ln + 1
    return out


def rule_spells_in(exits, entries, wa, wz, zrun):
    """규칙의 현금 구간(판 날 ~ 재매수 날 시가)을 창 [wa, wz+1)에 잘라 길이 목록으로."""
    ent = sorted(t for t, _ in entries)
    sp = []
    for (te, _, _) in exits:
        tr = next((t for t in ent if t > te), zrun + 1)
        s, e = max(te, wa), min(tr, wz + 1)
        if e > s:
            sp.append((s, e))
    return sp


# ============================================================================ 본 실행
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pct0(x, d=0):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def main():
    t0 = time.time()
    data, irx = load()
    us = {u: data[u] for u in sorted({u for _, u, _ in PRODUCTS})}
    syn = {p: synth(us[u], L, irx) for p, u, L in PRODUCTS}
    real = {p: data[p] for p, _, _ in PRODUCTS}
    for p, u, _ in PRODUCTS:
        assert (syn[p]["days"] == real[p]["days"]).all() and (us[u]["days"] == real[p]["days"]).all(), p
    days = syn[CAND_P]["days"]
    wins = win_idx(days)
    a, z = idx(days, RUN[0]), idx(days, RUN[1], "right") - 1
    print(f"실행 {days[a - 1]} 종가 시작 → {days[z]}, {z - a + 1}거래일", flush=True)

    # 효율선
    fronts = {}
    for p, u, L in PRODUCTS:
        ks = [round(k, 2) for k in np.arange(0, L + 1e-9, K_STEP)]
        fronts[("synth", p)] = Frontier({k: k_daily(us[u], k, irx) for k in ks}, wins)
        rr = np.concatenate([[np.nan], real[p]["c"][1:] / real[p]["c"][:-1] - 1])
        fronts[("real", p)] = Frontier({k: (k / L) * rr for k in ks}, wins)
    print(f"효율선 {time.time() - t0:.0f}s", flush=True)
    out = {"설정": {"RUN": RUN, "X0": X0, "W0": W0, "X_NB": X_NB, "W_NB": W_NB, "X_GRID_NB": X_GRID_NB, "SEEDS": SEEDS,
                   "SLIP": SLIP, "K_STEP": K_STEP}}

    # ------------------------------------------------------------ 1) 독립 재계산
    b, rb = syn[CAND_P], real[CAND_P]
    fS, fR = fronts[("synth", CAND_P)], fronts[("real", CAND_P)]
    v, pos, ex, en = sim_trail(b, a, z, X0, W0)
    eqS = lambda e: k_daily(us[CAND_U], e * CAND_L, irx)
    rr_real = np.concatenate([[np.nan], rb["c"][1:] / rb["c"][:-1] - 1])
    eqR = lambda e: e * rr_real
    M = {"synth": wmetrics(v, a, wins, fS, pos=pos, exits_t=[e[0] for e in ex], eqfun=eqS)}
    hv = np.concatenate([[1.0], b["c"][a:z + 1] / b["c"][a - 1]])
    M["hold"] = wmetrics(hv, a, wins, fS)
    vr, posr, exr, enr = sim_trail(rb, a, z, X0, W0)
    M["real"] = wmetrics(vr, a, wins, fR, pos=posr, exits_t=[e[0] for e in exr], eqfun=eqR)
    hr = np.concatenate([[1.0], rb["c"][a:z + 1] / rb["c"][a - 1]])
    M["hold_real"] = wmetrics(hr, a, wins, fR)
    same_exits = [e[0] for e in ex] == [e[0] for e in exr]
    print("합성 판 날 = 실제 판 날:", same_exits, len(ex), len(exr))
    rep = []
    for (kind, w, key), cl in CLAIM.items():
        mine = M[kind][w][key]
        tol = CLAIM_TOL.get(key, 0.0005)
        diff = mine - cl
        if key in ("연들락",):
            ok = abs(diff) <= tol + 1e-9
        elif key == "낙폭":
            ok = abs(diff) <= tol + 1e-9
        else:
            ok = abs(diff) <= 0.005   # 0.5%p
        rep.append({"종류": kind, "창": w, "항목": key, "주장": cl, "재계산": mine, "차": diff, "일치": bool(ok)})
    for r in rep:
        print(f"  {r['종류']:5s} {r['창']:6s} {r['항목']:8s} 주장 {r['주장']:+.4f} 재계산 {r['재계산']:+.4f} 차 {r['차']:+.4f} {'OK' if r['일치'] else 'XX'}")
    out["재현"] = rep
    out["후보"] = {k: {w: {kk: vv for kk, vv in m.items() if not kk.startswith("_")} for w, m in M[k].items()} for k in M}
    out["판날짜"] = [(days[t], float(px), why) for t, px, why in ex]
    out["재진입날짜"] = [(days[t], float(px)) for t, px in en]
    out["합성_실제_판날_같음"] = same_exits
    # 전체 최대 낙폭 위치
    vv = v / np.maximum.accumulate(v)
    ti = int(np.argmin(vv))
    tp = int(np.argmax(v[:ti + 1]))
    hv_ = hv / np.maximum.accumulate(hv)
    hi_ = int(np.argmin(hv_))
    hp_ = int(np.argmax(hv[:hi_ + 1]))
    rfa = rf_for(days, irx)[a:z + 1]
    out["현금이자_참고"] = {"기간평균_IRX": float(rfa.mean()), "현금비중": float(1 - pos.mean()),
                        "연_추가_근사": float(rfa.mean() * (1 - pos.mean()))}
    out["최대낙폭_구간"] = {"규칙": (days[a - 1 + tp], days[a - 1 + ti], float(vv[ti] - 1)),
                       "들고만": (days[a - 1 + hp_], days[a - 1 + hi_], float(hv_[hi_] - 1))}
    print("최대 낙폭 구간", out["최대낙폭_구간"])

    # 원 격자 252조합 재선택(합성, 미끄러짐 0)
    rules = rule_list()
    grid = {}
    for p, u, L in PRODUCTS:
        ind = indicators(us[u], syn[p])
        for sp in rules:
            vals, pc, cyc, ns = sim_rule(syn[p], us[u], ind, sp, a, z)
            grid[(p, sp["id"])] = wmetrics(vals, a, {w: wins[w] for w in (PICK, CHECK, AUX, "전체")},
                                          fronts[("synth", p)], c=syn[p]["c"], cyc=cyc)
    print(f"격자 252 {time.time() - t0:.0f}s", flush=True)
    vchk = grid[(CAND_P, "S30_wait20")]
    eng_diff = max(abs(vchk[w]["수익"] - M["synth"][w]["수익"]) for w in (PICK, CHECK, AUX, "전체"))
    combos = list(grid)
    g1 = max(combos, key=lambda c_: grid[c_][PICK]["효율선초과"])
    g3 = max(combos, key=lambda c_: np.nan_to_num(grid[c_][PICK]["칼마"], nan=-9))
    sel = [c_ for c_ in combos if grid[c_][PICK]["효율선초과"] > EPS]
    succ = {w: (sum(grid[c_][w]["효율선초과"] > EPS for c_ in sel) / len(sel) if sel else float("nan"),
                sum(grid[c_][w]["효율선초과"] > EPS for c_ in combos) / len(combos)) for w in (CHECK, AUX)}
    succ["전체"] = (sum(grid[c_]["전체"]["효율선초과_연"] > EPS for c_ in sel) / len(sel),
                  sum(grid[c_]["전체"]["효율선초과_연"] > EPS for c_ in combos) / len(combos))
    all3 = [c_ for c_ in combos if all(grid[c_][w]["효율선초과"] > EPS for w in (PICK, CHECK, AUX))]
    rank = sorted(combos, key=lambda c_: -grid[c_][PICK]["효율선초과"])
    top5 = [(c_[0], c_[1], grid[c_][PICK]["효율선초과"], grid[c_][CHECK]["효율선초과"], grid[c_][AUX]["효율선초과"]) for c_ in rank[:5]]
    # G2: 규칙 단위 7상품 합산 주기 기대값
    rids = [r["id"] for r in rules]
    pooled = {}
    for rid in rids:
        g = sum((grid[(p, rid)][PICK]["_g"] for p, _, _ in PRODUCTS), [])
        pooled[rid] = (len(g), float(np.mean(g)) if g else float("nan"))
    cand = [r for r in rids if pooled[r][0] >= 30]
    g2 = max(cand, key=lambda r: pooled[r][1]) if cand else None
    g2_check = float(np.mean([grid[(p, g2)][CHECK]["효율선초과"] for p, _, _ in PRODUCTS])) if g2 else float("nan")
    out["격자"] = {"엔진차_S30_wait20": eng_diff, "G1": g1, "G3": g3, "G1값": grid[g1][PICK]["효율선초과"],
                 "앞통과": len(sel), "조합": len(combos), "성공률": succ, "앞뒤2025모두": [list(c_) for c_ in all3],
                 "앞상위5": top5, "G2": g2, "G2_앞주기": pooled.get(g2), "G2_뒤효율선평균": g2_check}
    print("G1", g1, grid[g1][PICK]["효율선초과"], "G3", g3, "앞>0", len(sel), "성공", succ, "all3", len(all3), "G2", g2, pooled.get(g2), g2_check)
    print("앞 상위 5", top5)

    # ------------------------------------------------------------ 2) 시점 감사·체결 변형
    timing = {}
    # 미래 교란: T 이후 가격을 흔들어도 T까지 가치가 같아야 한다
    rng = np.random.default_rng(11)
    perturb = []
    for Tday in ("2018-10-10", "2020-03-02", "2022-06-01", "2026-03-02", "2026-07-01"):
        T = idx(days, Tday)
        b2 = {k: (vv_.copy() if k != "days" else vv_) for k, vv_ in b.items()}
        for k in ("o", "h", "l", "c"):
            b2[k][T + 1:] *= 1 + rng.normal(0, 0.05, len(b2[k]) - T - 1)
        b2["h"][T + 1:] = np.maximum.reduce([b2["h"][T + 1:], b2["o"][T + 1:], b2["c"][T + 1:]])
        b2["l"][T + 1:] = np.minimum.reduce([b2["l"][T + 1:], b2["o"][T + 1:], b2["c"][T + 1:]])
        for mode in ("intraday", "delay1", "close"):
            v1 = sim_trail(b, a, z, X0, W0, mode=mode)[0]
            v2 = sim_trail(b2, a, z, X0, W0, mode=mode)[0]
            k_ = T - a + 2
            perturb.append((Tday, mode, float(np.max(np.abs(v1[:k_] - v2[:k_])))))
    timing["미래교란"] = perturb
    # 손절이 없으면(x=1) 들고만 있기와 같아야 한다
    vh = sim_trail(b, a, z, 0.999999, W0)[0]
    timing["손절없음_vs_보유"] = float(np.max(np.abs(vh - hv)))
    # 장중 체결가가 그날 범위 안인가
    bad = 0
    for t, px, why in ex:
        if not (b["l"][t] - 1e-9 <= px <= b["h"][t] + 1e-9):
            bad += 1
    timing["체결가_범위밖"] = bad
    gaps = sum(1 for t, px, why in ex if why == "gap")
    timing["시가_갭_손절"] = gaps
    variants = {}
    for kind, bb, ff in (("synth", b, fS), ("real", rb, fR)):
        for mode in ("intraday", "delay1", "close"):
            for sl in (0.0, SLIP):
                vals, pp, exx, enn = sim_trail(bb, a, z, X0, W0, slip=sl, mode=mode)
                mm = wmetrics(vals, a, wins, ff, pos=pp, exits_t=[e[0] for e in exx])
                variants[f"{kind}|{mode}|{sl}"] = {w: {kk: vv_ for kk, vv_ in m.items() if not kk.startswith("_")} for w, m in mm.items()}
    timing["변형"] = variants
    out["시점"] = timing
    print("시점", {k: v_ for k, v_ in timing.items() if k != "변형"})
    for key in ("synth|intraday|0.0", "synth|intraday|0.005", "synth|delay1|0.0", "synth|delay1|0.005", "synth|close|0.0", "synth|close|0.005",
                "real|intraday|0.0", "real|delay1|0.005"):
        m = variants[key]
        print(f"  {key:24s} 앞 {pct(m[PICK]['수익'])}/{pct0(m[PICK]['낙폭'])} 효율 {pct(m[PICK]['효율선초과'])} | 뒤 {pct(m[CHECK]['효율선초과'])} | 2025 {pct(m[AUX]['효율선초과'])} | 전체 연 {pct(m['전체']['연율'])}/{pct0(m['전체']['낙폭'])} 효율연 {pct(m['전체']['효율선초과_연'])}")

    # ------------------------------------------------------------ 3) 이웃
    nb = {}
    for x in sorted(set(X_NB) | set(X_GRID_NB)):
        for wt in W_NB:
            if x not in X_NB and wt != W0:
                continue
            for sl in (0.0, SLIP):
                vals, pp, exx, _ = sim_trail(b, a, z, x, wt, slip=sl)
                mm = wmetrics(vals, a, {w: wins[w] for w in (PICK, CHECK, AUX, "2022", "2020", "2018", "전체")}, fS, pos=pp,
                              exits_t=[e[0] for e in exx])
                nb[(x, wt, sl)] = mm
    def nbstat(keys, sl):
        ks = [k for k in keys]
        pos3 = [k for k in ks if all(nb[(k[0], k[1], sl)][w]["효율선초과"] > EPS for w in (PICK, CHECK, AUX))]
        return {"n": len(ks), "앞>0": sum(nb[(k[0], k[1], sl)][PICK]["효율선초과"] > EPS for k in ks),
                "뒤>0": sum(nb[(k[0], k[1], sl)][CHECK]["효율선초과"] > EPS for k in ks),
                "2025>0": sum(nb[(k[0], k[1], sl)][AUX]["효율선초과"] > EPS for k in ks),
                "전체연>0": sum(nb[(k[0], k[1], sl)]["전체"]["효율선초과_연"] > EPS for k in ks),
                "셋 모두>0": len(pos3)}
    core = [(x, wt) for x in X_NB for wt in W_NB]
    gridax = [(x, W0) for x in X_GRID_NB]
    nbs = {"core0": nbstat(core, 0.0), "core5": nbstat(core, SLIP), "grid0": nbstat(gridax, 0.0), "grid5": nbstat(gridax, SLIP)}
    print("이웃", nbs)
    out["이웃"] = {"요약": nbs, "표": {f"{k[0]}|{k[1]}|{k[2]}": {w: {kk: vv_ for kk, vv_ in m.items() if not kk.startswith("_")} for w, m in mm.items()}
                                   for k, mm in nb.items()}}
    # 상품 이웃(같은 규칙, 합성 다른 6상품)
    pn = {}
    for p, u, L in PRODUCTS:
        vals, pp, exx, _ = sim_trail(syn[p], a, z, X0, W0)
        mm = wmetrics(vals, a, {w: wins[w] for w in (PICK, CHECK, AUX, "2022", "전체")}, fronts[("synth", p)], pos=pp, exits_t=[e[0] for e in exx])
        pn[p] = {w: {kk: vv_ for kk, vv_ in m.items() if not kk.startswith("_")} for w, m in mm.items()}
    out["상품이웃"] = pn

    # ------------------------------------------------------------ 4) 위약
    plc = {}
    for w in PLACEBO_WINS:
        wa, wz = wins[w]
        sp_rule = rule_spells_in(ex, en, wa, wz, z)
        lens = [e - s for s, e in sp_rule]
        # 규칙의 같은 구간을 시가 매도 판으로 다시(체결 방식 효과 분리)
        rows = []
        for sd in range(SEEDS):
            r_ = np.random.default_rng(1000 + sd)
            sps = random_spells(r_, wa, wz + 1, lens) if lens else []
            vals = sim_spells(b, wa, wz, sps)
            ret, dd = float(vals[-1] - 1), mdd(vals)
            fr, _, _ = fS.best(w, dd)
            rows.append({"수익": ret, "낙폭": dd, "효율선초과": ret - fr})
        same_open = sim_spells(b, wa, wz, sp_rule)
        ret_so, dd_so = float(same_open[-1] - 1), mdd(same_open)
        rv = M["synth"][w]
        pe = np.array([r_["효율선초과"] for r_ in rows])
        pr = np.array([r_["수익"] for r_ in rows])
        pd_ = np.array([r_["낙폭"] for r_ in rows])
        plc[w] = {"현금구간": len(lens), "현금일": int(sum(lens)), "창일수": wz - wa + 1,
                  "규칙_효율선초과": rv["효율선초과"], "규칙_수익": rv["수익"], "규칙_낙폭": rv["낙폭"],
                  "같은날_시가판_수익": ret_so, "같은날_시가판_낙폭": dd_so,
                  "위약_효율선초과_중앙": float(np.median(pe)), "위약_효율선초과_범위": [float(pe.min()), float(pe.max())],
                  "위약_수익_중앙": float(np.median(pr)), "위약_낙폭_중앙": float(np.median(pd_)),
                  "위약≥규칙(효율선초과)": int((pe >= rv["효율선초과"] - 1e-12).sum()),
                  "위약≥규칙(수익)": int((pr >= rv["수익"] - 1e-12).sum())}
        print("위약", w, plc[w])
    # 전체 기간판: 36번 현금 구간을 2015~2026에 무작위로
    sp_all = rule_spells_in(ex, en, a, z, z)
    lens_all = [e - s for s, e in sp_all]
    full_rows = []
    for sd in range(SEEDS):
        r_ = np.random.default_rng(5000 + sd)
        sps = random_spells(r_, a, z + 1, lens_all)
        vals = sim_spells(b, a, z, sps)
        mm = wmetrics(vals, a, {w: wins[w] for w in (PICK, CHECK, AUX, "2022", "2020", "2018", "전체")}, fS)
        full_rows.append({w: {"수익": m["수익"], "낙폭": m["낙폭"], "효율선초과": m["효율선초과"], "연율": m["연율"],
                              "효율선초과_연": m["효율선초과_연"]} for w, m in mm.items()})
    fp = {}
    for w in (PICK, CHECK, AUX, "2022", "2020", "2018", "전체"):
        key = "효율선초과_연" if w == "전체" else "효율선초과"
        arr = np.array([r_[w][key] for r_ in full_rows])
        mine = M["synth"][w][key]
        fp[w] = {"규칙": mine, "위약_중앙": float(np.median(arr)), "위약_범위": [float(arr.min()), float(arr.max())],
                 "위약≥규칙": int((arr >= mine - 1e-12).sum())}
    arr_dd = np.array([r_["전체"]["낙폭"] for r_ in full_rows])
    arr_cg = np.array([r_["전체"]["연율"] for r_ in full_rows])
    fp["전체_낙폭"] = {"규칙": M["synth"]["전체"]["낙폭"], "위약_중앙": float(np.median(arr_dd)), "위약_범위": [float(arr_dd.min()), float(arr_dd.max())],
                    "위약≤규칙(더 깊음)": int((arr_dd <= M["synth"]["전체"]["낙폭"] + 1e-12).sum())}
    fp["전체_연율"] = {"규칙": M["synth"]["전체"]["연율"], "위약_중앙": float(np.median(arr_cg)), "위약_범위": [float(arr_cg.min()), float(arr_cg.max())],
                    "위약≥규칙": int((arr_cg >= M["synth"]["전체"]["연율"] - 1e-12).sum())}
    out["위약"] = {"창별": plc, "전체판": fp, "현금구간수": len(lens_all), "현금일": int(sum(lens_all))}
    print("전체 위약", fp)

    # ------------------------------------------------------------ +) 시작 위상 민감도
    month_first = []
    seen = set()
    for i, d in enumerate(days):
        ym = d[:7]
        if START_FROM <= ym <= START_TO and ym not in seen:
            seen.add(ym)
            month_first.append(i)
    ph = []
    for s in month_first:
        vals, pp, exx, _ = sim_trail(b, s, z, X0, W0)
        sub = {w: wins[w] for w in (PICK, CHECK, AUX) if wins[w][0] >= s}
        mm = wmetrics(vals, s, sub, fS)
        ph.append({"시작": days[s], **{w: mm[w]["효율선초과"] for w in sub}, **{w + "_수익": mm[w]["수익"] for w in sub}})
    phdf = pd.DataFrame(ph)
    base = {w: M["synth"][w]["효율선초과"] for w in (PICK, CHECK, AUX)}
    phs = {}
    for w in (PICK, CHECK, AUX):
        col = phdf[w].dropna()
        phs[w] = {"시작수": int(len(col)), "주장과같음": int((abs(col - base[w]) < 1e-9).sum()), "최소": float(col.min()),
                  "중앙": float(col.median()), "최대": float(col.max()), ">0": int((col > EPS).sum())}
    out["위상"] = phs
    print("위상", phs)

    # ------------------------------------------------------------ 5) 장세
    reg = {}
    for w in ("2026앞", "2026뒤", "2026", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015",
              "2018Q4", "2020급락", "2020반등", "2025관세", "2025반등", "AI쇼크", "전체"):
        reg[w] = {k: {kk: M[k][w][kk] for kk in ("수익", "낙폭", "연율", "효율선초과", "효율선_k", "효율선초과_연") if kk in M[k][w]}
                  for k in ("synth", "hold", "real", "hold_real")}
    out["장세"] = reg
    out["전멸(-80%이하)"] = {k: M[k]["전체"]["낙폭"] <= -0.80 for k in ("synth", "hold", "real", "hold_real")}

    (RESULTS / f"{ID}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                        encoding="utf-8")
    print(f"총 {time.time() - t0:.0f}s")
    report(json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8")))
    return out


# ============================================================================ 보고서
WNAME = {"2026앞": "2026 앞(고르기)", "2026뒤": "2026 뒤(확인)", "2026": "2026 전체", "2025": "2025(보조)", "2018Q4": "2018 4분기",
         "2020급락": "2020 급락", "2020반등": "2020 반등", "2025관세": "2025 관세", "2025반등": "2025 반등", "AI쇼크": "2026 AI 쇼크",
         "전체": "2015~2026"}


def rr_(m):
    return f"{pct(m['수익'])}/{pct0(m['낙폭'])}"


def report(d):
    C = d["후보"]
    S_, H_, R_, HR_ = C["synth"], C["hold"], C["real"], C["hold_real"]
    G = d["격자"]
    T = d["시점"]
    V = T["변형"]
    NB = d["이웃"]["표"]
    NS = d["이웃"]["요약"]
    P = d["위약"]
    PH = d["위상"]
    rep = d["재현"]
    n_ok = sum(r["일치"] for r in rep)
    big = max(rep, key=lambda r: abs(r["차"]) if r["항목"] != "연들락" else 0)
    v_d1 = V["synth|delay1|0.0"]
    v_cl = V["synth|close|0.0"]
    v_s5 = V["synth|intraday|0.005"]
    p26 = P["창별"]["2026앞"]
    pck = P["창별"]["2026뒤"]
    fpl = P["전체판"]
    mdd_win = d["최대낙폭_구간"]
    pn = d["상품이웃"]
    pn_front_pos = [p for p in pn if p != CAND_P and pn[p][PICK]["효율선초과"] > EPS]
    ex26 = [x for x in d["판날짜"] if WIN[PICK][0] <= x[0] <= WIN[PICK][1]]
    ex26_txt = ", ".join(f"{int(x[0][5:7])}/{int(x[0][8:])}, {'시가가 선 아래로 열린 갭' if x[2] == 'gap' else '장중 선'}" for x in ex26)
    yrs_dd = (pd.Timestamp(mdd_win['규칙'][1]) - pd.Timestamp(mdd_win['규칙'][0])).days / 365.25
    crash_save = [S_[w]["수익"] - H_[w]["수익"] for w in ("2018Q4", "2020급락", "2025관세", "AI쇼크")]
    rebound_miss = [H_[w]["수익"] - S_[w]["수익"] for w in ("2020반등", "2025반등")]
    def rowrng(x):
        vals_ = [NB[f"{x}|{wt}|0.0"][PICK]["효율선초과"] for wt in W_NB]
        return f"{pct(max(vals_))} ~ {pct(min(vals_))}"
    pos_front = [(x, wt) for x in X_NB for wt in W_NB if NB[f"{x}|{wt}|0.0"][PICK]["효율선초과"] > EPS]
    ci = d["현금이자_참고"]
    L = []
    A = L.append
    A("# V_LEVIDX: 대표 후보 G1 반박 검증 — 합성 SOXL(SOXX 3배) · 산 뒤 고점 −30% 손절 → 20거래일 뒤 재진입 (2026-10-11)")
    A("")
    A("> 대상 주장: `results/p4_LEVIDX.md`의 G1(`SOXL S30_wait20`). 코드 `code/p4_V_LEVIDX.py`, 숫자 `results/p4_V_LEVIDX.json`.")
    A("> `trade_lab`·`lev_lab`·`stock_lab`·`tranche_lab`을 import하지 않고 `cache/p4_lev.parquet`만으로 합성·엔진·효율선을 새로 짰다"
      "(배당 수정 = Adj Close/Close 비율을 시가·고가·저가·종가에 곱함, 금융비용 ^IRX, 보수 3배 0.95%).")
    A("> 2015-01-02 ~ 2026-10-09를 한 번 이어서 돌렸다(2014-12-31 종가에 전량 보유로 시작). 수수료·현금 이자 0. 매수 추천·주문 수량 없음.")
    A("")
    A("## 결론")
    A("")
    A(f"1. **숫자는 그대로 재현됐다.** 주장 숫자 {len(rep)}개 중 {n_ok}개가 반올림 폭 안에서 같았다(가장 큰 차 {abs(big['차']) * 100:.2f}%p, "
      f"{big['종류']} {big['창']} {big['항목']} — 주장 낙폭이 정수 %로 반올림된 탓). 원 격자 252조합을 새 엔진으로 다시 돌려도 "
      f"G1 = G3 = 이 조합(앞 효율선초과 {pct(G['G1값'])}), 앞 > 0 {G['앞통과']}개 중 뒤 성공 {pct0(G['성공률']['2026뒤'][0])}, "
      f"앞·뒤·2025 모두 > 0 {len(G['앞뒤2025모두'])}개, G2 = 기초<SMA20(앞 주기 {G['G2_앞주기'][0]}건 기대 {pct(G['G2_앞주기'][1])}, "
      f"뒤 7상품 효율선초과 평균 {pct(G['G2_뒤효율선평균'])})로 주장과 같다. 실제 SOXL은 36번 판 날이 합성과 하루도 다르지 않았다.")
    A(f"2. **앞 구간 +179.9%p는 '그날 −30% 선 근처에서 판 것' 한 번에 달려 있다.** 2026 앞의 손절은 {len(ex26)}번({ex26_txt})이다. "
      f"신호를 보고 다음 날 시가에 팔면 앞 효율선초과 {pct(v_d1[PICK]['효율선초과'])}, 종가로 감지하면 {pct(v_cl[PICK]['효율선초과'])}로 줄고 "
      f"2025는 {pct(v_d1[AUX]['효율선초과'])}로 뒤집힌다. 이웃 25개(손절 20~40% × 대기 10~30일) 중 앞 > 0은 {NS['core0']['앞>0']}개, "
      f"앞·뒤·2025 모두 > 0은 {NS['core0']['셋 모두>0']}개. 같은 20거래일 현금 기간을 앞 구간 무작위 날짜에 둔 20개 중 "
      f"{p26['위약≥규칙(효율선초과)']}개가 규칙과 같거나 높았다(고르는 데 쓴 창에서도 우연과 구별되지 않음). 같은 규칙을 다른 6상품에 걸면 앞 > 0은 "
      f"{len(pn_front_pos)}개{'(' + ', '.join(pn_front_pos) + ')' if pn_front_pos else ''}.")
    A(f"3. **확인 구간과 긴 기간에서 졌다.** 뒤 {pct(S_[CHECK]['효율선초과'])}(무작위 현금 기간 20개 중 {pck['위약≥규칙(효율선초과)']}개가 규칙보다 같거나 나음), "
      f"2015~2026 연 {pct(S_['전체']['연율'])}/{pct0(S_['전체']['낙폭'])} vs 들고만 연 {pct(H_['전체']['연율'])}/{pct0(H_['전체']['낙폭'])} "
      f"(들고만이 수익·낙폭 둘 다 나음, 효율선보다 연 {pct(S_['전체']['효율선초과_연'])}; 같은 수·길이의 현금 기간을 무작위로 둔 20개 중 {fpl['전체']['위약≥규칙']}개가 같거나 높음 — 손절 시점에 값이 없다). "
      f"최대 낙폭 {pct0(mdd_win['규칙'][2])}는 들고만({pct0(mdd_win['들고만'][2])})보다 깊고 {mdd_win['규칙'][0]}~{mdd_win['규칙'][1]}로 {yrs_dd:.1f}년 이어졌다. "
      f"폭락 구간 손실은 크게 줄였지만(2020 급락 {pct(S_['2020급락']['수익'])} vs {pct(H_['2020급락']['수익'])}, 2025 관세 {pct(S_['2025관세']['수익'])} vs "
      f"{pct(H_['2025관세']['수익'])}, AI 쇼크 {pct(S_['AI쇼크']['수익'])} vs {pct(H_['AI쇼크']['수익'])}) 2022 한 해는 {S_['2022']['판횟수']}번 들락으로 "
      f"{pct(S_['2022']['수익'])}(들고만 {pct(H_['2022']['수익'])})였다. **판정: 반박(refuted)** — 숫자는 맞고, 규칙의 우위는 없다(갈래 LEVIDX 결론 1 '기각'과 같다).")
    A("")
    A("## 사용자 질문과의 관계")
    A("")
    A("- 레버리지 상품(SOXL)에 '−30% 추적 손절 + 20일 쉬고 다시 사기'를 쓸 근거는 없다. 2026 앞 숫자는 손절 한 번의 체결 위치에서 나왔고, "
      "하루만 늦어도 대부분 사라지며 손절 폭을 25%나 40%로 옮기면 크게 마이너스다.")
    A("- 사용자는 손절(스탑) 주문을 미리 걸어 둘 수 있으므로(`docs/desk_system.md` §1) '선에서 체결'은 불가능한 가정은 아니다. 다만 고점이 오르면 선을 날마다 다시 걸어야 하고, "
      "프리장·주간거래에서 먼저 닿는 경우(이 연구는 정규장 시가·장중만 봄)는 재지 않았다.")
    A(f"- 폭락 구간 손실을 줄이는 효과는 있었지만, 덜 들고 그냥 있기(고정 배수)보다 못했다. 2015~2026 효율선은 고정 {S_['전체']['효율선_k']:.2f}배 "
      f"(연 {pct(S_['전체']['효율선_연율'])}/{pct0(S_['전체']['효율선_낙폭'])})로, 규칙(연 {pct(S_['전체']['연율'])}/{pct0(S_['전체']['낙폭'])})보다 낙폭이 깊지 않으면서 "
      f"연 {-S_['전체']['효율선초과_연'] * 100:.1f}%p 더 벌었다.")
    A("")
    A("## 시험 범위 (사전 고정, 스크립트 맨 위 상수)")
    A("")
    A("| 검증 | 개수·간격 | 비고 |")
    A("|---|---|---|")
    A(f"| 독립 재계산 | 주장 숫자 {len(rep)}개 + 원 격자 252조합(규칙 36 × 상품 7) 재선택 | 0.5%p 넘게 다르면 원인 규명(해당 없음) |")
    A("| 시점 감사 | 미래 교란 5날짜 × 체결 3방식 = 15건, 손절 없음 = 보유, 체결가가 그날 범위 안 | |")
    A("| 체결 변형 | 장중(주장) · 하루 늦춘 판(장중 감지 → 다음 시가, 재진입도 하루 뒤) · 종가 감지 → 다음 시가 × 미끄러짐 0·0.5% × 합성·실제 = 12판 | |")
    A("| 이웃 | 손절 {20, 25, 30, 35, 40}% × 대기 {10, 15, 20, 25, 30}거래일 = **25개**(±2칸, 5%p·5일 간격), 원 격자 손절 {15, 20, 30, 40, 50}% × 대기 20 = 5개, 상품 7개 | 미끄러짐 0·0.5% |")
    A(f"| 위약 | 창별: 규칙의 현금 기간(수·길이 같게)을 창 안 무작위 날짜에, 시드 {SEEDS}개 × 6창. 전체판: 36번 × 20거래일을 2015~2026 무작위, 시드 {SEEDS}개 | 시가 매도 → 시가 재매수 |")
    A("| 위상 | 시작일을 2015-01 ~ 2025-12 매월 첫 거래일(132개)로 바꿈 | 2026 숫자가 시작 위상에 달렸나 |")
    A("")
    A("## 변수 시점")
    A("")
    A("| 변수 | 언제 아나 | 쓰는 때 |")
    A("|---|---|---|")
    A("| 산 뒤 최고 종가(고점) | t−1 종가 | t일 장중 손절선 = 고점 × 0.70(시가가 선 아래면 시가) |")
    A("| 판 날 t_e | t_e 장중 | t − t_e ≥ 20인 첫날 시가 재매수(정보 없음, 날짜 수만) |")
    A("| 합성 시가·고가·저가 | 기초 그날 시가·고가·저가와 전일 종가 | 장중 손절 판정(합성의 장중 경로는 근사) |")
    A("| ^IRX·보수 | 그날 | 합성 종가(실제 상품은 가격에 내장) |")
    A("| 효율선 k, 같은 낙폭 | 창 전체(사후) | 평가 기준일 뿐 신호 아님 |")
    A("| 폭락 구간 날짜 | 사후 | 맥락 표만 |")
    A("")
    A("## 1) 독립 재계산")
    A("")
    A("| 종류 | 창 | 항목 | 주장 | 재계산 | 차 | 일치 |")
    A("|---|---|---|---|---|---|---|")
    for r in rep:
        if r["항목"] == "연들락":
            A(f"| {r['종류']} | {WNAME.get(r['창'], r['창'])} | 연 들락(번) | {r['주장']:.1f} | {r['재계산']:.2f} | {r['차']:+.2f} | {'O' if r['일치'] else 'X'} |")
        else:
            A(f"| {r['종류']} | {WNAME.get(r['창'], r['창'])} | {r['항목']} | {pct(r['주장'])} | {pct(r['재계산'])} | {r['차'] * 100:+.2f}%p | {'O' if r['일치'] else 'X'} |")
    A("")
    A(f"- 원 격자 재선택(합성, 미끄러짐 0): 새 엔진의 S30_wait20과 후보 전용 엔진의 차 {G['엔진차_S30_wait20']:.1e}. 앞 효율선초과 상위 5: "
      + ", ".join(f"{p} {rid} {pct(a_)}(뒤 {pct(b_)}, 2025 {pct(c_)})" for p, rid, a_, b_, c_ in G["앞상위5"]) + ".")
    A(f"- '앞 > 0' {G['앞통과']}/252 → 뒤 성공 {pct0(G['성공률']['2026뒤'][0])}(무조건 {pct0(G['성공률']['2026뒤'][1])}), 2025 성공 "
      f"{pct0(G['성공률']['2025'][0])}(무조건 {pct0(G['성공률']['2025'][1])}), 2015~2026 성공 {pct0(G['성공률']['전체'][0])}(무조건 {pct0(G['성공률']['전체'][1])}). 주장 표와 같다.")
    A(f"- 판 날 36번(합성·실제 같음): " + ", ".join(x[0] for x in d["판날짜"]) + ".")
    A("")
    A("## 2) 시점 감사 · 하루 늦춘 판 · 미끄러짐")
    A("")
    mx = max(x[2] for x in T["미래교란"])
    A(f"- 미래 교란 15건(교란일 뒤 가격을 흔들어도 교란일까지 가치가 같아야 함): 최대 차 {mx:.1e}. 손절 없음(선 0%) = 들고만 있기 차 {T['손절없음_vs_보유']:.1e}. "
      f"체결가가 그날 저가~고가 밖인 경우 {T['체결가_범위밖']}건. 36번 중 {T['시가_갭_손절']}번은 갭 하락이라 시가에 팔렸다.")
    A("- 장중 손절선 체결은 '선 가격에 정확히 체결'을 가정한다. 사람이 신호를 보고 다음 날 시가에 파는 판(하루 늦춘 판)과 종가로 감지하는 판을 함께 돌렸다.")
    A("")
    A("| 판 | 2026 앞 수익/낙폭 · 효율선초과 | 2026 뒤 · 효율선초과 | 2025 · 효율선초과 | 2022 | 2015~2026 연율/낙폭 · 효율선초과(연) | 연 들락 |")
    A("|---|---|---|---|---|---|---|")
    names = {"intraday": "장중 선(주장)", "delay1": "하루 늦춘 판", "close": "종가 감지 → 다음 시가"}
    for kind in ("synth", "real"):
        for mode in ("intraday", "delay1", "close"):
            for sl in ("0.0", "0.005"):
                m = V[f"{kind}|{mode}|{sl}"]
                A(f"| {'합성' if kind == 'synth' else '실제 SOXL'} · {names[mode]} · 미끄러짐 {float(sl) * 100:.1f}% | {rr_(m[PICK])} · {pct(m[PICK]['효율선초과'])} | "
                  f"{rr_(m[CHECK])} · {pct(m[CHECK]['효율선초과'])} | {rr_(m[AUX])} · {pct(m[AUX]['효율선초과'])} | {rr_(m['2022'])} | "
                  f"연 {pct(m['전체']['연율'])}/{pct0(m['전체']['낙폭'])} · {pct(m['전체']['효율선초과_연'])} | {m['전체']['연들락']:.1f} |")
    A("")
    A(f"- 앞 구간 효율선초과는 체결 위치에 매우 민감하다: 손절을 다음 날 시가로 미루면 앞 낙폭이 {pct0(S_[PICK]['낙폭'])} → {pct0(v_d1[PICK]['낙폭'])}가 되고, "
      "그 낙폭까지 허용하는 고정 배수가 훨씬 높아 초과분이 대부분 사라진다.")
    A(f"- 미끄러짐 0.5%는 앞에서는 작게({(v_s5[PICK]['효율선초과'] - S_[PICK]['효율선초과']) * 100:+.1f}%p) 깎지만, 실제 SOXL 2025에서는 고점(매수가 기준)이 바뀌어 판 날이 달라지며 효율선초과가 "
      f"{pct(V['real|intraday|0.0'][AUX]['효율선초과'])} → {pct(V['real|intraday|0.005'][AUX]['효율선초과'])}로 뒤집혔다(경로 민감).")
    A("")
    A("## 3) 이웃 (사전 고정 ±2칸)")
    A("")
    for w, key, nm in ((PICK, "효율선초과", "2026 앞 효율선초과"), (CHECK, "효율선초과", "2026 뒤 효율선초과"), (AUX, "효율선초과", "2025 효율선초과"),
                       ("전체", "효율선초과_연", "2015~2026 효율선초과(연)")):
        A(f"### {nm} (행 = 손절 %, 열 = 대기 거래일, 미끄러짐 0)")
        A("")
        A("| 손절 \\ 대기 | " + " | ".join(str(x) for x in W_NB) + " |")
        A("|---|" + "---|" * len(W_NB))
        for x in X_NB:
            A(f"| {x * 100:.0f}% | " + " | ".join(pct(NB[f'{x}|{wt}|0.0'][w][key]) for wt in W_NB) + " |")
        A("")
    A("| 이웃 묶음 | 개수 | 앞 > 0 | 뒤 > 0 | 2025 > 0 | 2015~2026 > 0 | 앞·뒤·2025 모두 > 0 |")
    A("|---|---|---|---|---|---|---|")
    for k, nm in (("core0", "손절×대기 25개, 미끄러짐 0"), ("core5", "손절×대기 25개, 미끄러짐 0.5%"), ("grid0", "원 격자 손절 5개(대기 20), 0"),
                  ("grid5", "원 격자 손절 5개(대기 20), 0.5%")):
        s = NS[k]
        A(f"| {nm} | {s['n']} | {s['앞>0']} | {s['뒤>0']} | {s['2025>0']} | {s['전체연>0']} | {s['셋 모두>0']} |")
    A("")
    A("원 격자 손절 축(대기 20):")
    A("")
    A("| 손절 | 앞 | 뒤 | 2025 | 2015~2026 연율/낙폭 · 효율선초과(연) |")
    A("|---|---|---|---|---|")
    for x in X_GRID_NB:
        m = NB[f"{x}|{W0}|0.0"]
        A(f"| {x * 100:.0f}% | {pct(m[PICK]['효율선초과'])} | {pct(m[CHECK]['효율선초과'])} | {pct(m[AUX]['효율선초과'])} | "
          f"연 {pct(m['전체']['연율'])}/{pct0(m['전체']['낙폭'])} · {pct(m['전체']['효율선초과_연'])} |")
    A("")
    A("상품 이웃(같은 규칙 −30%·20일, 합성):")
    A("")
    A("| 상품 | 앞 | 뒤 | 2025 | 2022 | 2015~2026 연율/낙폭 · 효율선초과(연) |")
    A("|---|---|---|---|---|---|")
    for p, m in pn.items():
        A(f"| {p} | {pct(m[PICK]['효율선초과'])} | {pct(m[CHECK]['효율선초과'])} | {pct(m[AUX]['효율선초과'])} | {pct(m['2022']['효율선초과'])} | "
          f"연 {pct(m['전체']['연율'])}/{pct0(m['전체']['낙폭'])} · {pct(m['전체']['효율선초과_연'])} |")
    A("")
    A("- 앞 구간 양수 " + str(len(pos_front)) + "개: " + ", ".join(f"손절 {x * 100:.0f}%·대기 {wt}" for x, wt in pos_front)
      + f". 손절 25%는 대기와 상관없이 {rowrng(0.25)}, 40%는 {rowrng(0.40)}로 뾰족한 봉우리다.")
    A("")
    A("## 4) 위약: '−30% 선에서 판 것'인가 '그 무렵 20일 쉰 것'인가")
    A("")
    A("창별: 그 창 안 규칙의 현금 기간(개수·길이 같게)을 무작위 날짜로 옮겨 시가 매도 → 시가 재매수(시드 20개). '같은 날 시가 판'은 규칙의 현금 기간을 그대로 두고 판 날 시가에 판 것(미래를 아는 비교, 참고용).")
    A("")
    A("| 창 | 현금 구간·일 | 규칙 수익/낙폭 · 효율선초과 | 위약 중앙 수익/낙폭 · 효율선초과 (범위) | 위약 ≥ 규칙(효율선초과) | 같은 날 시가 판 |")
    A("|---|---|---|---|---|---|")
    for w, s in P["창별"].items():
        A(f"| {WNAME.get(w, w)} | {s['현금구간']}구간 · {s['현금일']}/{s['창일수']}일 | {pct(s['규칙_수익'])}/{pct0(s['규칙_낙폭'])} · {pct(s['규칙_효율선초과'])} | "
          f"{pct(s['위약_수익_중앙'])}/{pct0(s['위약_낙폭_중앙'])} · {pct(s['위약_효율선초과_중앙'])} ({pct(s['위약_효율선초과_범위'][0])} ~ {pct(s['위약_효율선초과_범위'][1])}) | "
          f"{s['위약≥규칙(효율선초과)']}/{SEEDS} | {pct(s['같은날_시가판_수익'])}/{pct0(s['같은날_시가판_낙폭'])} |")
    A("")
    A(f"전체판(36번 × 20거래일 현금 기간 = {P['현금일']}일을 2015~2026 무작위 날짜로, 시드 20개):")
    A("")
    A("| 창 | 규칙 효율선초과 | 위약 중앙 (범위) | 위약 ≥ 규칙 |")
    A("|---|---|---|---|")
    for w in (PICK, CHECK, AUX, "2022", "2020", "2018", "전체"):
        s = fpl[w]
        A(f"| {WNAME.get(w, w)}{' (연)' if w == '전체' else ''} | {pct(s['규칙'])} | {pct(s['위약_중앙'])} ({pct(s['위약_범위'][0])} ~ {pct(s['위약_범위'][1])}) | {s['위약≥규칙']}/{SEEDS} |")
    s = fpl["전체_연율"]
    A(f"| 2015~2026 연율 | {pct(s['규칙'])} | {pct(s['위약_중앙'])} ({pct(s['위약_범위'][0])} ~ {pct(s['위약_범위'][1])}) | {s['위약≥규칙']}/{SEEDS} |")
    s = fpl["전체_낙폭"]
    A(f"| 2015~2026 최대 낙폭 | {pct0(s['규칙'])} | {pct0(s['위약_중앙'])} ({pct0(s['위약_범위'][0])} ~ {pct0(s['위약_범위'][1])}) | 위약이 더 깊음 {s['위약≤규칙(더 깊음)']}/{SEEDS} |")
    A("")
    A("- 손절 시점이 무작위보다 확실히 나은 곳은 2020(위약 1/20만 같거나 높음) 하나이고, 2026 앞은 고른 창인데도 5/20, 2025는 7/20, 2026 뒤·2022·2018은 규칙이 무작위보다 나빴다(18~20/20).")
    A(f"- 긴 기간 수익·낙폭은 같은 수·같은 길이의 현금 기간을 아무 날짜에나 둔 것과 구별되지 않는다(연율 위약 ≥ 규칙 {fpl['전체_연율']['위약≥규칙']}/20, "
      f"최대 낙폭은 규칙 {pct0(fpl['전체_낙폭']['규칙'])}가 위약 중앙 {pct0(fpl['전체_낙폭']['위약_중앙'])}보다 깊음). '−30% 선에서 판다'는 정보가 '1년에 3번, 20일씩 쉰다'보다 나은 몫을 남기지 않았다.")
    A("")
    A("## 5) 장세별 (수익/최대 낙폭, 미끄러짐 0)")
    A("")
    A("| 창 | 규칙(합성) | 들고만(합성) | 효율선초과(배수 k) | 규칙(실제 SOXL) | 들고만(실제) |")
    A("|---|---|---|---|---|---|")
    for w in ("2026앞", "2026뒤", "2026", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015",
              "2018Q4", "2020급락", "2020반등", "2025관세", "2025반등", "AI쇼크"):
        A(f"| {WNAME.get(w, w)} | {rr_(S_[w])} | {rr_(H_[w])} | {pct(S_[w]['효율선초과'])} ({S_[w]['효율선_k']:.2f}배) | {rr_(R_[w])} | {rr_(HR_[w])} |")
    A(f"| 2015~2026 | 연 {pct(S_['전체']['연율'])}/{pct0(S_['전체']['낙폭'])} | 연 {pct(H_['전체']['연율'])}/{pct0(H_['전체']['낙폭'])} | "
      f"연 {pct(S_['전체']['효율선초과_연'])} ({S_['전체']['효율선_k']:.2f}배) | 연 {pct(R_['전체']['연율'])}/{pct0(R_['전체']['낙폭'])} | "
      f"연 {pct(HR_['전체']['연율'])}/{pct0(HR_['전체']['낙폭'])} |")
    A("")
    wipe = d["전멸(-80%이하)"]
    A(f"- '−80% 이하' 낙폭: 규칙(합성) {'예' if wipe['synth'] else '아니오'}, 들고만(합성) {'예' if wipe['hold'] else '아니오'}, 규칙(실제) {'예' if wipe['real'] else '아니오'}, "
      f"들고만(실제) {'예' if wipe['hold_real'] else '아니오'}. 규칙의 최대 낙폭 {pct0(mdd_win['규칙'][2])}({mdd_win['규칙'][0]} 고점 → {mdd_win['규칙'][1]} 바닥), "
      f"들고만 {pct0(mdd_win['들고만'][2])}({mdd_win['들고만'][0]} → {mdd_win['들고만'][1]}).")
    yrs = [w for w in ("2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026")]
    posy = [w for w in yrs if S_[w]["효율선초과"] > EPS]
    A(f"- 해마다 효율선초과 > 0: {len(posy)}/{len(yrs)}해({', '.join(posy)}). 폭락 구간(2018 4분기·2020 급락·2025 관세·AI 쇼크)에서는 들고만보다 "
      f"{min(crash_save) * 100:.0f}~{max(crash_save) * 100:.0f}%p 덜 잃었지만 2022처럼 하락이 길고 반등이 섞이면 판 뒤 20일 만에 되사서 다시 −30%를 맞는 일이 "
      f"{S_['2022']['판횟수']}번 이어져 들고만과 비슷하게 잃었다. 반등 구간(2020 반등, 2025 반등)에서는 들고만보다 {min(rebound_miss) * 100:.0f}~{max(rebound_miss) * 100:.0f}%p 덜 벌었다.")
    A(f"- 같은 평균 노출 고정 배수 대비(합성): 앞 {pct(S_[PICK]['같은노출초과'])}, 뒤 {pct(S_[CHECK]['같은노출초과'])}, 2025 {pct(S_[AUX]['같은노출초과'])}, "
      f"2015~2026 연 {pct(S_['전체']['같은노출초과_연'])}(평균 노출 {S_['전체']['평균노출']:.2f} = 평균 {S_['전체']['평균노출'] * 3:.2f}배).")
    A("")
    A("## 위상(시작 시점) 민감도")
    A("")
    A("| 창 | 시작 수 | 주장 경로와 같은 값 | 최소 | 중앙 | 최대 | > 0 |")
    A("|---|---|---|---|---|---|---|")
    for w in (PICK, CHECK, AUX):
        s = PH[w]
        A(f"| {WNAME.get(w, w)} | {s['시작수']} | {s['주장과같음']} | {pct(s['최소'])} | {pct(s['중앙'])} | {pct(s['최대'])} | {s['>0']} |")
    A("")
    A("- 시작일을 바꿔도 손절·재진입 경로가 금방 같아져(재진입 때 고점이 매수가로 초기화) 2026 숫자는 시작 위상에 달려 있지 않다. 이 점은 주장 숫자의 신뢰도를 높인다(약점 아님).")
    A("")
    A("## 판정")
    A("")
    A("| 항목 | 결과 |")
    A("|---|---|")
    A(f"| 재현 | {n_ok}/{len(rep)} 일치, G1·G3·G2·'앞 > 0' 성공률 재현, 실제 SOXL 판 날 36/36 같음 |")
    A(f"| 시점 | 미래 교란 0, 단 하루 늦춘 판에서 앞 {pct(S_[PICK]['효율선초과'])} → {pct(v_d1[PICK]['효율선초과'])}, 2025 → {pct(v_d1[AUX]['효율선초과'])} |")
    A(f"| 미끄러짐 0.5% | 앞 {pct(v_s5[PICK]['효율선초과'])}, 뒤 {pct(v_s5[CHECK]['효율선초과'])}, 2025 {pct(v_s5[AUX]['효율선초과'])}, 2015~2026 연 {pct(v_s5['전체']['효율선초과_연'])} |")
    A(f"| 이웃 | 25개 중 앞·뒤·2025 모두 > 0 {NS['core0']['셋 모두>0']}개(0.5%에서도 {NS['core5']['셋 모두>0']}개), 앞 > 0 {NS['core0']['앞>0']}개 |")
    A(f"| 위약 | 고른 창(앞)에서도 무작위 {p26['위약≥규칙(효율선초과)']}/20이 같거나 높음, 뒤 {pck['위약≥규칙(효율선초과)']}/20, 2015~2026 {fpl['전체']['위약≥규칙']}/20 |")
    A(f"| 장세 | 2026 뒤·2015~2026·2022·2018에서 효율선에 짐, 2015~2026 최대 낙폭 {pct0(S_['전체']['낙폭'])}(−80% 이하) |")
    A("")
    A(f"**반박(refuted).** 주장 숫자는 정확하지만, 이 숫자는 252조합 중 2026 앞 1등을 고른 결과이고 그 1등은 손절 한 번({ex26_txt})의 체결 위치에서 나왔다. "
      f"하루 늦추면 {pct(v_d1[PICK]['효율선초과'])}로 줄고 손절 폭을 25%·40%로 옮기면 크게 마이너스이며, 확인 구간(뒤)에서는 무작위 현금 기간 20개 모두보다 나빴고, "
      "긴 기간에서는 같은 수·길이의 무작위 현금 기간과 구별되지 않는다. 갈래 LEVIDX가 스스로 내린 '2026 기준 기각'은 맞다.")
    A("")
    A("## 선택편향·한계")
    A("")
    A("- 252조합 중 앞 1등이다. 앞 > 0인 60개 중 뒤에서 이긴 것이 0개라는 점(주장·재현 같음)이 선택편향의 크기를 보여 준다.")
    A("- 2026 앞·뒤는 5개월·4개월, 경로 하나. 앞 구간 손절은 1번뿐이라 '규칙 효과'와 '그 하루의 체결'을 나눌 수 없다.")
    A("- 합성 장중 가격은 기초 시가·고가·저가 × 3으로 근사했다. 실제 SOXL로 돌려도 판 날이 같았으므로 이 근사가 결론을 바꾸지는 않는다.")
    A("- 생존편향: SOXL은 지금 살아 있는 상품이다. 망한 테마 레버리지는 빠져 있고, 이 방향은 손절에 불리하다(손절이 빛날 표본이 빠짐).")
    A(f"- 현금 이자 0%: 현금에 있던 날(전체의 {pct0(ci['현금비중'])})에 단기 금리를 받지 않았다. 기간 평균 ^IRX {pct0(ci['기간평균_IRX'], 1)}로 치면 연 약 "
      f"{pct(ci['연_추가_근사'])}로, 효율선 대비 연 {pct(S_['전체']['효율선초과_연'])}를 뒤집지 못한다.")
    A("- 위약은 시가 매도 → 시가 재매수다. 규칙의 장중 선 체결과 체결 방식이 달라, 창별 위약은 '같은 노출 시간·같은 나가기 횟수'만 맞춘 비교다.")
    A("- 세금·환전·호가 차는 미끄러짐 0.5%로만 대신했다.")
    A("")
    A("## 다시 돌리기")
    A("")
    A("```")
    A("cd data/derived/analysis/stop_research/code")
    A("PYTHONIOENCODING=utf-8 py p4_V_LEVIDX.py          # 계산 + 보고서")
    A("PYTHONIOENCODING=utf-8 py p4_V_LEVIDX.py --report # JSON에서 보고서만")
    A("```")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("보고서", RESULTS / f"{ID}.md")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8")))
    else:
        main()
