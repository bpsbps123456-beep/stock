"""5단계 반박 검증 V_SOXL(2026-10-11): 갈래 SOXL의 대표 후보 '목표 변동성 45%(Vt45)'를 깨 본다.

후보: w = clip(0.45 / (3 × σ20), 0, 1). σ20 = 기초(2001-07-16 전 ^SOX, 뒤 SOXX) 20일 실현변동성(연율, 표준편차 ddof=1 × √252).
      t일 종가로 정해 t+1일 시가에 매일 비율을 맞춘다. 나머지 현금 0%.

독립성: lev5·stock_lab·p5_SOXL을 import하지 않는다. cache/p5_lev_long.parquet만 읽어 기초 이음·3배 합성·비용 맞춤·
        엔진·효율선을 새로 짠다. 엔진은 K개 전략을 한 번에 도는 벡터 일별 루프다(효율선 101칸·이웃·위약을 같이 돈다).

사전 고정(결과를 본 뒤 바꾸지 않는다)
  - 합성: 상품 종가 수익 = 3·r_u − 2·rf(t−1)/252 − (보수 0.95% + 추가 비용)/252, 상품 시가 = 전일 종가 × (1 + 3·기초 갭).
    추가 비용 = 실제 상품과 겹치는 기간의 연 로그 비용 차(추가 비용 0 합성 대비)로 상품마다 따로 잰다.
  - 이음: 2001-07-13까지 ^SOX 수익, 2001-07-16부터 SOXX 수익(SOXX 07-13 종가 기준). 원 갈래와 하루 다르다(의도한 독립 선택).
  - 엔진: 시가에 목표 w로 맞춤(미끄러짐 = slip × |w − 시가 직전 비율| × 가치), 종가까지 상품 수익, 현금 이자 0%(연 4% 판 한 줄).
  - 창: 2026 앞 01-02~05-29, 2026 뒤 06-01~10-09, 2025, 긴 역사 1995-05-01~2025-12-31(1995~2025)·~2026-10-09, 실제 SOXL 2010-03-12~.
  - 효율선초과: 규칙 최대 낙폭보다 깊지 않은 고정 w(0~1, 0.01, 매일 맞춤) 중 최고값을 뺀다. 짧은 창은 기간 수익, 긴 창은 연율.
  - 이웃(앞뒤 2칸): σ* ∈ {0.35, 0.40, 0.45, 0.50, 0.55} × 창 n ∈ {10, 15, 20, 25, 30} = 25개(중심 포함).
    맞춤 빈도 변형(따로 보고): 매일 / 주 1회 / 띠 5%·10%(목표와 차이가 띠를 넘을 때만 맞춤).
  - 위약: 창마다 규칙의 목표 w 열을 원형으로 밀어(시드 20개, 창 길이의 10~90%) 같은 평균 w·거의 같은 매매 횟수로 날짜만 섞는다.
    보조: 21일(긴 창)·10일(짧은 창) 묶음 순서 섞기 20개.
  - 다른 상품(파라미터 고정): TQQQ←^NDX, UPRO←^GSPC, TNA←IWM, TECL←XLK, YINN←FXI, EDC←EEM (+ 참고 KORU←EWY, FAS←XLF,
    LABU←XBI, NUGT←GDX). 합성은 기초 시작 252거래일 뒤부터 2026-10-09까지 한 번 이어 돌린다.
  - 판정 규칙: 재현(0.5%p 안) 실패 → refuted 후보. 시점·지연·미끄러짐·이웃·위약·다른 상품 중 핵심 주장이 깨지면 fragile.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PQ = ROOT / "cache" / "p5_lev_long.parquet"
RESULTS = ROOT / "results"
ID = "p5_V_SOXL"

FEE = 0.0095
L = 3.0
RUN0, END = "1995-05-01", "2026-10-09"
REAL0 = "2010-03-12"
SLIP = 0.005
CASH_ALT = 0.04
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
EPS = 1e-6
SSTAR, NWIN = 0.45, 20
NB_S = (0.35, 0.40, 0.45, 0.50, 0.55)
NB_N = (10, 15, 20, 25, 30)
SEEDS = 20

WIN = {"2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31"),
       "1995~2025": (RUN0, "2025-12-31"), "1995~2026": (RUN0, END)}
CRASH = {"2000~02": ("2000-03-01", "2002-10-31"), "2008": ("2007-10-01", "2009-03-31"), "2011": ("2011-01-03", "2011-12-30"),
         "2015~16": ("2015-05-01", "2016-02-29"), "2018Q4": ("2018-10-01", "2018-12-31"),
         "2020급락": ("2020-02-01", "2020-03-31"), "2022": ("2022-01-03", "2022-12-30"),
         "2025관세": ("2025-02-20", "2025-04-08"), "2026_7월": ("2026-06-23", "2026-07-29")}
LONGW = {"1995~2025", "1995~2026", "실제전체", "긴역사"}

# 원 갈래 주장(p5_SOXL.json에서 옮김, 대조용)
CLAIM = {"1995~2026_연율": 0.1727, "1995~2026_낙폭": -0.8986, "평균노출": 0.5401, "연매매": 6.88, "연회전": 5.75,
         "1995~2025_초과": 0.0362, "1995~2025_초과_slip": 0.0196, "1995~2025_초과_lag": 0.0258,
         "실제_초과": 0.0618, "실제_초과_slip": 0.0341, "2026뒤_수익": 0.0311, "2026뒤_낙폭": -0.1999,
         "2026_7월": -0.1999, "2026앞_초과": -0.147, "2025_초과": -0.0771, "2026뒤_초과": 0.0309,
         "보유_2026뒤": -0.36, "보유_2026_7월": -0.6931}

RAW = pd.read_parquet(PQ)


# ============================================================================ 자료
def load(sym: str) -> pd.DataFrame:
    g = RAW[RAW.sym == sym].dropna(subset=["Close"]).sort_values("date").drop_duplicates("date", keep="last")
    g = g[g.Close > 0]
    f = (g["Adj Close"] / g["Close"]).fillna(1.0)
    o = g.Open.where(g.Open > 0, g.Close)
    return pd.DataFrame({"o": (o * f).to_numpy(), "c": (g.Close * f).to_numpy()}, index=g.date.to_numpy())


def under(spec) -> pd.DataFrame:
    """기초의 종가 수익 r, 시가 갭 g(전일 종가 대비)."""
    if isinstance(spec, tuple):
        a, b, cut = spec
        A, B = load(a), load(b)
        ra, ga = A.c.pct_change(), A.o / A.c.shift() - 1
        rb, gb = B.c.pct_change(), B.o / B.c.shift() - 1
        ma, mb = A.index < cut, B.index >= cut
        df = pd.DataFrame({"r": pd.concat([ra[ma], rb[mb]]), "g": pd.concat([ga[ma], gb[mb]])})
    else:
        A = load(spec)
        df = pd.DataFrame({"r": A.c.pct_change(), "g": A.o / A.c.shift() - 1})
    df = df.sort_index()
    df.iloc[0] = 0.0
    return df.fillna(0.0)


def rf_prev(days: np.ndarray) -> np.ndarray:
    s = RAW[RAW.sym == "^IRX"].dropna(subset=["Close"]).drop_duplicates("date").set_index("date")["Close"].sort_index() / 100
    s = s.reindex(sorted(set(s.index) | set(days))).ffill().reindex(days).fillna(0.0)
    return s.shift(1).fillna(0.0).to_numpy()


def synth(U: pd.DataFrame, extra: float) -> tuple[np.ndarray, np.ndarray]:
    days = U.index.to_numpy()
    rf = rf_prev(days)
    R = L * U.r.to_numpy() - (L - 1) * rf / 252 - (FEE + extra) / 252
    R = np.maximum(R, -0.999)
    R[0] = 0.0
    C = 100 * np.cumprod(1 + R)
    Cp = np.concatenate([[100.0], C[:-1]])
    O = Cp * np.maximum(1 + L * U.g.to_numpy(), 0.001)
    O[0] = C[0]
    return O, C


def mdd(v) -> float:
    v = np.asarray(v, float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def calib(U: pd.DataFrame, real: str) -> dict:
    """추가 비용 0 합성과 실제 상품의 겹친 기간 연 로그 비용 차 → 추가 비용."""
    R = load(real)
    O0, C0 = synth(U, 0.0)
    s = pd.Series(C0, index=U.index)
    common = R.index.intersection(s.index).sort_values()
    cs, cr = s.loc[common].to_numpy(), R.c.loc[common].to_numpy()
    yrs = (len(common) - 1) / 252
    extra = float((np.log(cs[-1] / cs[0]) - np.log(cr[-1] / cr[0])) / yrs)
    O1, C1 = synth(U, extra)
    s1 = pd.Series(C1, index=U.index).loc[common].to_numpy()
    rs, rr = s1[1:] / s1[:-1] - 1, cr[1:] / cr[:-1] - 1
    return {"real": real, "시작": str(common[0]), "일수": int(len(common)), "추가비용": extra,
            "일수익상관": float(np.corrcoef(rs, rr)[0, 1]), "실제_낙폭": mdd(cr), "합성_낙폭": mdd(s1),
            "맞춘뒤_연로그차": float((np.log(s1[-1] / s1[0]) - np.log(cr[-1] / cr[0])) / yrs)}


# ============================================================================ 엔진(K개 동시)
def engine(O: np.ndarray, C: np.ndarray, W: np.ndarray, slip: float = 0.0, cash: float = 0.0, x0=1.0, at_close=False):
    """O, C: 길이 n(0번 = 시작 종가). W: (n, K) 목표 비율. 시가 모드: W[t] = t일 시가 목표(nan = 그대로).
    종가 모드(at_close): W[t] = t일 종가에 맞출 목표. 반환 E(n,K) 종가 가치, X(n,K) 종가 비율, TO(n,K) |Δw|."""
    n, K = W.shape
    E = np.empty((n, K)); X = np.empty((n, K)); TO = np.zeros((n, K))
    v = np.ones(K); x = np.broadcast_to(np.asarray(x0, float), (K,)).copy()
    E[0], X[0] = v, x
    rc = cash / 252
    for t in range(1, n):
        if not at_close:
            gt = O[t] / C[t - 1] - 1
            ht = C[t] / O[t] - 1
            hold = v * x * (1 + gt)
            vo = hold + v * (1 - x) * (1 + rc)
            xo = np.divide(hold, vo, out=np.zeros(K), where=vo > 0)
            w = W[t]
            w = np.where(np.isnan(w), xo, w)
            d = np.abs(w - xo)
            vo = vo * (1 - slip * d)
            v = vo * (1 + w * ht)
            x = np.divide(vo * w * (1 + ht), v, out=np.zeros(K), where=v > 0)
        else:
            Rt = C[t] / C[t - 1] - 1
            hold = v * x * (1 + Rt)
            vn = hold + v * (1 - x) * (1 + rc)
            xc = np.divide(hold, vn, out=np.zeros(K), where=vn > 0)
            w = W[t]
            w = np.where(np.isnan(w), xc, w)
            d = np.abs(w - xc)
            v = vn * (1 - slip * d)
            x = w
        E[t], X[t], TO[t] = v, x, d
    return E, X, TO


# ============================================================================ 규칙
def sigma(U: pd.DataFrame, n: int) -> np.ndarray:
    return (U.r.rolling(n, min_periods=n).std() * np.sqrt(252)).to_numpy()


def vt_dec(U: pd.DataFrame, s: float = SSTAR, n: int = NWIN) -> np.ndarray:
    """t일 종가에 정한 비율(기초 날짜 기준)."""
    sg = sigma(U, n)
    w = s / (L * sg)
    return np.clip(np.nan_to_num(w, nan=1.0), 0, 1)


def rule_dec(U: pd.DataFrame, rid: str) -> np.ndarray:
    """전멸 회피 '유일' 주장 점검용 T·V·VS·TV 18개(S 3개는 장중 손절이라 제외 — 원 갈래 2000~02 −99.7~−99.9%)."""
    c = np.cumprod(1 + U.r.to_numpy())
    cs = pd.Series(c)
    sg = sigma(U, 20)
    med = pd.Series(sg).expanding(min_periods=20).median().shift(1).to_numpy()
    n_ = len(c)
    if rid.startswith("T") and rid != "TV":
        n = int(rid[1:].split("b")[0].rstrip("h"))
        bd = 0.03 if "b3" in rid else 0.0
        low = 0.5 if rid.endswith("h") else 0.0
        m = cs.rolling(n, min_periods=n).mean().to_numpy()
        W = np.ones(n_); inn = True
        for t in range(n_):
            if not np.isfinite(m[t]):
                continue
            if inn and c[t] < m[t] * (1 - bd):
                inn = False
            elif not inn and c[t] > m[t] * (1 + bd):
                inn = True
            W[t] = 1.0 if inn else low
        return W
    if rid in ("Vm100", "Vm75", "Vm100w"):
        k = 0.75 if rid == "Vm75" else 1.0
        return np.clip(np.nan_to_num(k * med / sg, nan=1.0), 0, 1)
    if rid in ("Vt45", "Vt60", "Vt60w"):
        return np.clip(np.nan_to_num((0.45 if rid == "Vt45" else 0.60) / (3 * sg), nan=1.0), 0, 1)
    if rid.startswith("VS"):
        k = 1.5 if rid == "VS15" else 2.0
        return np.where(np.nan_to_num(sg) > k * np.nan_to_num(med, nan=np.inf), 0.0, 1.0)
    if rid == "TV":
        m = cs.rolling(200, min_periods=200).mean().to_numpy()
        a = c > np.nan_to_num(m, nan=-np.inf)
        b = np.nan_to_num(sg) < 1.5 * np.nan_to_num(med, nan=np.inf)
        return (a.astype(float) + b.astype(float)) / 2
    raise ValueError(rid)


OTHER18 = ["T50", "T50b3", "T100", "T100b3", "T150", "T150b3", "T200", "T200b3", "T200h",
           "Vm100", "Vm75", "Vt45", "Vt60", "Vm100w", "Vt60w", "VS15", "VS20", "TV"]


def targets(dec_u: np.ndarray, udays: np.ndarray, pdays: np.ndarray, lag: int = 0, weekly=False, close_exec=False) -> np.ndarray:
    """상품 날짜 pdays의 t일 목표 = 기초에서 (상품 t−1일) 이하 마지막 날의 결정(lag일 더 늦게). close_exec면 t일 종가 체결."""
    j = np.searchsorted(udays, pdays, side="right") - 1          # 상품 t일과 같은 날(또는 직전) 기초 인덱스
    prev = np.concatenate([[j[0] - 1], j[:-1]])                   # 상품 t−1일의 기초 인덱스
    idx = np.clip(prev - lag, 0, None)
    tg = dec_u[idx].astype(float)
    if weekly:
        wk = pd.to_datetime(pd.Series(pdays)).dt.isocalendar()
        key = (wk.year.astype(int) * 100 + wk.week.astype(int)).to_numpy()
        first = np.concatenate([[True], key[1:] != key[:-1]])
        tg = np.where(first, tg, np.nan)
    return tg


def banded(tg: np.ndarray, O, C, band: float) -> np.ndarray:
    """띠 맞춤: 시가 직전 비율과 목표 차가 band 이하면 거래 안 함(1열 전용, 경로 의존이라 루프)."""
    out = tg.copy(); x = 1.0
    for t in range(1, len(C)):
        gt = O[t] / C[t - 1] - 1
        xo = x * (1 + gt) / (1 + x * gt)
        w = tg[t]
        if not np.isfinite(w) or abs(w - xo) <= band:
            out[t] = np.nan; w = xo
        ht = C[t] / O[t] - 1
        x = w * (1 + ht) / (1 + w * ht)
    return out


# ============================================================================ 창 통계·효율선
def span(days, d0, d1):
    i0 = int(np.searchsorted(days, d0, side="left"))
    i1 = int(np.searchsorted(days, d1, side="right")) - 1
    return max(i0, 1), i1


def wst(E, X, TO, days, d0, d1, full=False) -> dict:
    """E·X·TO는 1열(배열). 창 직전 종가 = 기준 1."""
    i0, i1 = span(days, d0, d1)
    seg = E[i0 - 1:i1 + 1]
    n = i1 - i0 + 1
    g = seg[-1] / seg[0]
    yrs = n / 252
    out = {"수익": float(g - 1), "낙폭": mdd(seg), "연율": float(g ** (1 / yrs) - 1) if g > 0 else -1.0,
           "평균노출": float(X[i0:i1 + 1].mean()), "연회전": float(TO[i0:i1 + 1].sum() / yrs),
           "연매매": float((TO[i0:i1 + 1] >= 0.1).sum() / yrs), "거래일비율": float((TO[i0:i1 + 1] > 1e-3).mean()), "일수": n}
    if full:
        r = seg[1:] / seg[:-1] - 1
        out["연변동성"] = float(r.std() * np.sqrt(252))
        r252 = seg[252:] / seg[:-252] - 1
        out["최악1년"] = float(r252.min()) if len(r252) else float("nan")
        pk = np.maximum.accumulate(seg)
        ar = np.arange(len(seg))
        last = np.maximum.accumulate(np.where(seg >= pk * (1 - 1e-12), ar, 0))
        out["최장물밑_년"] = float((ar - last).max() / 252)
        out["전멸"] = bool(out["낙폭"] <= -0.90)
    return out


class Front:
    def __init__(self, O, C, days, wins, slip=0.0, cash=0.0):
        W = np.tile(W_GRID, (len(C), 1)).astype(float)
        E, X, TO = engine(O, C, W, slip, cash, x0=1.0)
        self.days, self.tab, self.E, self.X = days, {}, E, X
        for wn, (d0, d1) in wins.items():
            i0, i1 = span(days, d0, d1)
            if i1 <= i0:
                continue
            seg = E[i0 - 1:i1 + 1]
            g = seg[-1] / seg[0]
            n = i1 - i0 + 1
            cagr = np.where(g > 0, np.abs(g) ** (252 / n) - 1, -1.0)
            md = (seg / np.maximum.accumulate(seg, axis=0) - 1).min(axis=0)
            r = seg[1:] / seg[:-1] - 1
            vol = r.std(axis=0) * np.sqrt(252)
            self.tab[wn] = (g - 1, cagr, md, vol)

    def excess(self, wn, st):
        ret, cagr, md, vol = self.tab[wn]
        lw = wn in LONGW
        key = cagr if lw else ret
        val = st["연율"] if lw else st["수익"]
        ok = md >= st["낙폭"] - 1e-12
        i = int(np.argmax(np.where(ok, key, -np.inf)))
        j = int(round(min(max(st["평균노출"], 0), 1) * 100))
        out = {"초과": float(val - key[i]), "효율선_w": float(W_GRID[i]), "같은노출초과": float(val - key[j])}
        if "연변동성" in st:
            k = int(np.argmin(np.abs(vol - st["연변동성"])))
            out["같은변동성초과"] = float(val - key[k]); out["같은변동성_w"] = float(W_GRID[k])
        return out


def evaluate(O, C, days, tg_cols: dict, wins, slip=0.0, cash=0.0, front=None, at_close=False, full_keys=LONGW):
    front = front or Front(O, C, days, wins, slip, cash)
    names = list(tg_cols)
    W = np.column_stack([tg_cols[k] for k in names])
    E, X, TO = engine(O, C, W, slip, cash, x0=1.0, at_close=at_close)
    out = {}
    for k, nm in enumerate(names):
        row = {}
        for wn, (d0, d1) in wins.items():
            i0, i1 = span(days, d0, d1)
            if i1 <= i0:
                continue
            st = wst(E[:, k], X[:, k], TO[:, k], days, d0, d1, full=wn in full_keys)
            if wn in front.tab:
                st.update(front.excess(wn, st))
            row[wn] = st
        out[nm] = row
    return out, front, (E, X, TO, names)


# ============================================================================ 위약
def placebo(O, C, days, tg, wins, front, slip=0.0, mode="roll"):
    """창마다 목표 열을 섞어(원형 밀기 또는 묶음 순서 섞기) 같은 엔진으로 돌린다. 실제와 같은 시작 비율."""
    out = {}
    for wn, (d0, d1) in wins.items():
        i0, i1 = span(days, d0, d1)
        seg_t = tg[i0:i1 + 1]
        n = len(seg_t)
        cols = [seg_t]
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            if mode == "roll":
                k = int(rng.integers(int(0.1 * n), int(0.9 * n)))
                cols.append(np.roll(seg_t, k))
            else:
                B = 21 if wn in LONGW else 10
                blocks = [seg_t[i:i + B] for i in range(0, n, B)]
                order = rng.permutation(len(blocks))
                cols.append(np.concatenate([blocks[q] for q in order]))
        W = np.vstack([np.full(len(cols), np.nan), np.column_stack(cols)])
        E, X, TO = engine(O[i0 - 1:i1 + 1], C[i0 - 1:i1 + 1], W, slip, 0.0, x0=float(seg_t[0]))
        d = days[i0 - 1:i1 + 1]
        rows = []
        for k in range(len(cols)):
            st = wst(E[:, k], X[:, k], TO[:, k], d, d[1], d[-1], full=wn in LONGW)
            st.update(front.excess(wn, st))
            rows.append(st)
        key = "연율" if wn in LONGW else "수익"
        act, pl = rows[0], rows[1:]
        out[wn] = {"실제_값": act[key], "실제_초과": act["초과"], "위약_값_중앙": float(np.median([r[key] for r in pl])),
                   "위약_초과_중앙": float(np.median([r["초과"] for r in pl])),
                   "순위_값": int(1 + sum(r[key] > act[key] for r in pl)), "순위_초과": int(1 + sum(r["초과"] > act["초과"] for r in pl)),
                   "실제_연매매": act["연매매"], "위약_연매매_중앙": float(np.median([r["연매매"] for r in pl])),
                   "실제_평균w": act["평균노출"], "위약_평균w_중앙": float(np.median([r["평균노출"] for r in pl])),
                   "실제_낙폭": act["낙폭"], "위약_낙폭_중앙": float(np.median([r["낙폭"] for r in pl]))}
    return out


# ============================================================================ 시점 점검
def timing_audit(U, extra):
    msgs = []
    O, C = synth(U, extra)
    days = U.index.to_numpy()
    dec = vt_dec(U)
    tg = targets(dec, days, days)
    # (1) 손 계산: 몇 날짜에서 t일 시가 목표 = t−1일까지 20개 수익의 표준편차로 만든 값
    r = U.r.to_numpy()
    worst = 0.0
    for t in np.random.default_rng(3).integers(300, len(days) - 1, 25):
        s = np.std(r[t - 20:t], ddof=1) * np.sqrt(252)            # r[t−20..t−1]
        worst = max(worst, abs(min(max(0.45 / (3 * s), 0), 1) - tg[t]))
    msgs.append(f"손 계산 대조 25일: t일 시가 목표 = r[t−20..t−1]로 만든 값, 최대 차 {worst:.1e}(0이어야 함)")
    # (2) 미래 교란: 2020-03-02 뒤 기초 수익·갭에 잡음 → 합성·신호 다시 → 그날까지 가치 동일해야 함
    T = int(np.searchsorted(days, "2020-03-02"))
    U2 = U.copy()
    rng = np.random.default_rng(11)
    U2.iloc[T + 1:, 0] = U2.r.to_numpy()[T + 1:] + rng.normal(0, 0.03, len(U2) - T - 1)
    U2.iloc[T + 1:, 1] = U2.g.to_numpy()[T + 1:] + rng.normal(0, 0.01, len(U2) - T - 1)
    O2, C2 = synth(U2, extra)
    tg2 = targets(vt_dec(U2), days, days)
    a = int(np.searchsorted(days, RUN0))
    E1, _, _ = engine(O[a - 1:], C[a - 1:], tg[a - 1:, None])
    E2, _, _ = engine(O2[a - 1:], C2[a - 1:], tg2[a - 1:, None])
    k = T - a + 2
    msgs.append(f"미래 교란(2020-03-02 뒤 기초 잡음): 교란일까지 가치 최대 차 {np.max(np.abs(E1[:k] - E2[:k])):.1e}(0이어야 함), "
                f"다음 날 차 {abs(E1[k, 0] - E2[k, 0]):.2e}(0이 아니어야 교란이 들어간 것)")
    return msgs


# ============================================================================ 본 실행
def p(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%p"


def rm(st, cagr=False):
    if st is None:
        return "—"
    v = st["연율"] if cagr else st["수익"]
    return f"{'연 ' if cagr else ''}{v * 100:+.1f}%/{st['낙폭'] * 100:.{1 if cagr else 0}f}%"


def main():
    t0 = time.time()
    U = under(("^SOX", "SOXX", "2001-07-16"))
    days = U.index.to_numpy()
    cal = calib(U, "SOXL")
    EXTRA = cal["추가비용"]
    print("SOXL 맞춤", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cal.items()})
    O, C = synth(U, EXTRA)
    a = int(np.searchsorted(days, RUN0))
    Os, Cs, ds = O[a - 1:], C[a - 1:], days[a - 1:]
    allw = {**WIN, **CRASH}

    dec = vt_dec(U)
    T = lambda **kw: targets(dec, days, days, **kw)[a - 1:]
    base_tg = T()

    # ---------------------------------------------------------------- 1) 재현 + 변형(지연·종가 체결·미끄러짐·이자)
    hold = np.ones(len(Cs))
    cols = {"Vt45": base_tg, "보유": hold, "F33": np.full(len(Cs), 0.33), "F50": np.full(len(Cs), 0.5)}
    rep0, fr0, _ = evaluate(Os, Cs, ds, cols, allw)
    rep5, fr5, _ = evaluate(Os, Cs, ds, {"Vt45": base_tg}, allw, slip=SLIP)
    repL, _, _ = evaluate(Os, Cs, ds, {"Vt45_lag1": T(lag=1), "Vt45_lag2": T(lag=2)}, allw, front=fr0)
    repL5, _, _ = evaluate(Os, Cs, ds, {"Vt45_lag1": T(lag=1)}, allw, slip=SLIP, front=fr5)
    # 종가 체결: t일 종가에 정한 비율을 t+1일 종가에 맞춤(시가 자료 없는 1999~2001 영향 제거). 효율선도 종가 맞춤으로.
    W_close = targets(dec, days, days)[a - 1:]                   # t일 값 = (t−1)일 결정 → t일 종가 체결
    Wc = np.tile(W_GRID, (len(Cs), 1)).astype(float)
    Ec, Xc, TOc = engine(Os, Cs, Wc, 0.0, 0.0, 1.0, at_close=True)
    frClose = Front.__new__(Front); frClose.days = ds; frClose.tab = {}
    for wn, (d0, d1) in WIN.items():
        i0, i1 = span(ds, d0, d1); seg = Ec[i0 - 1:i1 + 1]; g = seg[-1] / seg[0]; n = i1 - i0 + 1
        frClose.tab[wn] = (g - 1, np.where(g > 0, np.abs(g) ** (252 / n) - 1, -1.0),
                           (seg / np.maximum.accumulate(seg, axis=0) - 1).min(axis=0), (seg[1:] / seg[:-1] - 1).std(axis=0) * np.sqrt(252))
    repC, _, _ = evaluate(Os, Cs, ds, {"Vt45_종가": W_close}, allw, front=frClose, at_close=True)
    repI, frI, _ = evaluate(Os, Cs, ds, {"Vt45": base_tg, "F33": np.full(len(Cs), 0.33)}, {k: WIN[k] for k in ("1995~2025", "1995~2026")}, cash=CASH_ALT)

    # 실제 SOXL
    RS = load("SOXL")
    ra = int(np.searchsorted(RS.index.to_numpy(), REAL0))
    Or, Cr, dr = RS.o.to_numpy()[ra - 1:], RS.c.to_numpy()[ra - 1:], RS.index.to_numpy()[ra - 1:]
    tg_r = targets(dec, days, dr)
    rwin = {"실제전체": (REAL0, END), "2026앞": WIN["2026앞"], "2026뒤": WIN["2026뒤"], "2025": WIN["2025"],
            "2022": CRASH["2022"], "2026_7월": CRASH["2026_7월"]}
    realR, frR, _ = evaluate(Or, Cr, dr, {"Vt45": tg_r, "보유": np.ones(len(Cr)), "Vt45_lag1": targets(dec, days, dr, lag=1)}, rwin)
    realR5, frR5, _ = evaluate(Or, Cr, dr, {"Vt45": tg_r, "Vt45_lag1": targets(dec, days, dr, lag=1)}, rwin, slip=SLIP)
    # 실제 SOXL 날짜 그대로, 합성으로도(같은 날 시작) — 합성·실제 차
    js = np.searchsorted(days, dr)
    synR, _, _ = evaluate(O[js], C[js], dr, {"Vt45": tg_r}, rwin)
    print("재현 끝", round(time.time() - t0, 1))

    # ---------------------------------------------------------------- 1b) '유일한 전멸 회피' 주장: T·V·VS·TV 18개 + F
    o18 = {rid: targets(rule_dec(U, rid), days, days, weekly=rid.endswith("w"))[a - 1:] for rid in OTHER18}
    lw = {"1995~2026": WIN["1995~2026"], "1995~2025": WIN["1995~2025"]}
    u0, _, _ = evaluate(Os, Cs, ds, o18, lw, front=Front(Os, Cs, ds, lw))
    u5, _, _ = evaluate(Os, Cs, ds, o18, lw, slip=SLIP, front=Front(Os, Cs, ds, lw, SLIP))
    o18L = {rid: targets(rule_dec(U, rid), days, days, lag=1, weekly=rid.endswith("w"))[a - 1:] for rid in OTHER18}
    uL, _, _ = evaluate(Os, Cs, ds, o18L, lw, front=Front(Os, Cs, ds, lw))
    uniq = {rid: {"낙폭0": u0[rid]["1995~2026"]["낙폭"], "낙폭_slip": u5[rid]["1995~2026"]["낙폭"], "낙폭_lag": uL[rid]["1995~2026"]["낙폭"],
                  "연율0": u0[rid]["1995~2026"]["연율"]} for rid in OTHER18}
    fixed_mdd = {w: float(fr0.tab["1995~2026"][2][int(round(w * 100))]) for w in (0.30, 0.33, 0.36, 0.40, 0.45, 0.50)}
    fixed_cagr = {w: float(fr0.tab["1995~2026"][1][int(round(w * 100))]) for w in (0.30, 0.33, 0.36, 0.40, 0.45, 0.50)}
    fixed_best = float(fr0.tab["1995~2026"][1].max()); fixed_best_w = float(W_GRID[int(np.argmax(fr0.tab["1995~2026"][1]))])

    # ---------------------------------------------------------------- 2) 시점 감사
    tmsgs = timing_audit(U, EXTRA)
    for m in tmsgs:
        print(m)

    # ---------------------------------------------------------------- 3) 이웃
    nb_cols, nb_r = {}, {}
    for s in NB_S:
        for n in NB_N:
            d_ = vt_dec(U, s, n)
            nb_cols[(s, n)] = targets(d_, days, days)[a - 1:]
            nb_r[(s, n)] = targets(d_, days, dr)
    key = lambda k: f"σ*{k[0]:.2f}·n{k[1]}"
    nb0, _, _ = evaluate(Os, Cs, ds, {key(k): v for k, v in nb_cols.items()}, WIN, front=fr0)
    nb5, _, _ = evaluate(Os, Cs, ds, {key(k): v for k, v in nb_cols.items()}, WIN, slip=SLIP, front=fr5)
    nbR, _, _ = evaluate(Or, Cr, dr, {key(k): v for k, v in nb_r.items()}, {"실제전체": rwin["실제전체"]}, front=frR)
    # 맞춤 빈도 변형
    fq = {"매일": base_tg, "주 1회": T(weekly=True), "띠 5%": banded(base_tg, Os, Cs, 0.05), "띠 10%": banded(base_tg, Os, Cs, 0.10)}
    fq0, _, _ = evaluate(Os, Cs, ds, fq, allw, front=fr0)
    fq5, _, _ = evaluate(Os, Cs, ds, fq, WIN, slip=SLIP, front=fr5)
    print("이웃 끝", round(time.time() - t0, 1))

    # ---------------------------------------------------------------- 4) 위약
    pw = {k: WIN[k] for k in ("2026앞", "2026뒤", "2025", "1995~2025")}
    plc = placebo(Os, Cs, ds, base_tg, pw, fr0, mode="roll")
    plb = placebo(Os, Cs, ds, base_tg, pw, fr0, mode="block")
    plR = placebo(Or, Cr, dr, tg_r, {"실제전체": rwin["실제전체"]}, frR, mode="roll")
    plRb = placebo(Or, Cr, dr, tg_r, {"실제전체": rwin["실제전체"]}, frR, mode="block")
    print("위약 끝", round(time.time() - t0, 1))

    # ---------------------------------------------------------------- 5) 다른 상품
    MAP = {"TQQQ": "^NDX", "UPRO": "^GSPC", "TNA": "IWM", "TECL": "XLK", "YINN": "FXI", "EDC": "EEM",
           "KORU": "EWY", "FAS": "XLF", "LABU": "XBI", "NUGT": "GDX"}
    xprod = {}
    for prod, base in MAP.items():
        Ux = under(base)
        cx = calib(Ux, prod)
        Ox, Cx = synth(Ux, cx["추가비용"])
        dx = Ux.index.to_numpy()
        ax = 252
        Oxs, Cxs, dxs = Ox[ax - 1:], Cx[ax - 1:], dx[ax - 1:]
        decx = vt_dec(Ux)
        tgx = targets(decx, dx, dx)[ax - 1:]
        long0 = (str(dxs[1]), "2025-12-31")
        xw = {"2026앞": WIN["2026앞"], "2026뒤": WIN["2026뒤"], "2025": WIN["2025"], "긴역사": long0}
        e0, f0, _ = evaluate(Oxs, Cxs, dxs, {"Vt45": tgx, "보유": np.ones(len(Cxs))}, xw)
        e5, _, _ = evaluate(Oxs, Cxs, dxs, {"Vt45": tgx}, xw, slip=SLIP)
        eL, _, _ = evaluate(Oxs, Cxs, dxs, {"Vt45": targets(decx, dx, dx, lag=1)[ax - 1:]}, xw, front=f0)
        RX = load(prod)
        rx = 21
        Orx, Crx, drx = RX.o.to_numpy()[rx - 1:], RX.c.to_numpy()[rx - 1:], RX.index.to_numpy()[rx - 1:]
        er, _, _ = evaluate(Orx, Crx, drx, {"Vt45": targets(decx, dx, drx), "보유": np.ones(len(Crx))},
                            {"실제전체": (str(drx[1]), END)})
        er5, _, _ = evaluate(Orx, Crx, drx, {"Vt45": targets(decx, dx, drx)}, {"실제전체": (str(drx[1]), END)}, slip=SLIP)
        xprod[prod] = {"기초": base, "맞춤": cx, "긴역사_시작": long0[0], "합성": e0, "합성_slip": e5, "합성_lag": eL,
                       "실제": er, "실제_slip": er5}
        print(prod, round(time.time() - t0, 1))

    # ================================================================ 저장
    def rnd(o):
        if isinstance(o, dict):
            return {str(k): rnd(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rnd(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    payload = {"설정": {"후보": "w = clip(0.45/(3σ20), 0, 1), 기초 ^SOX(~2001-07-13)+SOXX(2001-07-16~)", "보수": FEE,
                      "추가비용_SOXL": EXTRA, "실행": [RUN0, END], "미끄러짐": SLIP, "이웃": [NB_S, NB_N], "위약_시드": SEEDS},
               "맞춤": cal, "재현": rep0, "미끄러짐": rep5, "지연": repL, "지연_미끄러짐": repL5, "종가체결": repC, "현금4%": repI,
               "실제": realR, "실제_미끄러짐": realR5, "합성_실제날짜": synR, "전멸점검18": uniq,
               "덜담기_낙폭": fixed_mdd, "덜담기_연율": fixed_cagr, "덜담기_최고연율": [fixed_best_w, fixed_best],
               "시점": tmsgs, "이웃": nb0, "이웃_미끄러짐": nb5, "이웃_실제": nbR, "빈도": fq0, "빈도_미끄러짐": fq5,
               "위약_원형": plc, "위약_묶음": plb, "위약_실제_원형": plR, "위약_실제_묶음": plRb, "다른상품": xprod, "원주장": CLAIM}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(rnd(payload), ensure_ascii=False, indent=1), encoding="utf-8")
    print("json 저장", round(time.time() - t0, 1))
    report(rnd(payload))
    print("md 저장", round(time.time() - t0, 1))
    return payload


# ============================================================================ 보고서
def report(d: dict):
    A = []
    a = A.append
    r0, r5, rL, rL5, rC = d["재현"], d["미끄러짐"], d["지연"], d["지연_미끄러짐"], d["종가체결"]
    V = r0["Vt45"]
    real, real5, ns, n5 = d["실제"], d["실제_미끄러짐"], d["이웃"], d["이웃_미끄러짐"]
    lg = "1995~2025"
    keys = list(ns)
    cnt = lambda tab, wn: sum(1 for k in keys if tab[k][wn]["초과"] > EPS)
    all4 = lambda tab: sum(1 for k in keys if all(tab[k][w]["초과"] > EPS for w in ("2026앞", "2026뒤", "2025", lg)))
    surv = lambda tab: sum(1 for k in keys if tab[k]["1995~2026"]["낙폭"] > -0.90)
    nbR, xp = d["이웃_실제"], d["다른상품"]
    six = ["TQQQ", "UPRO", "TNA", "TECL", "YINN", "EDC"]
    sg = lambda x: "0" if abs(x) < 0.002 else ("+" if x > 0 else "−")
    fq = d["빈도"]
    variants = [("기준(시가, 미끄러짐 0)", V, real["Vt45"]), ("미끄러짐 0.5%", r5["Vt45"], real5["Vt45"]),
                ("하루 늦게", rL["Vt45_lag1"], real["Vt45_lag1"]), ("하루 늦게 + 미끄러짐 0.5%", rL5["Vt45_lag1"], real5["Vt45_lag1"]),
                ("이틀 늦게", rL["Vt45_lag2"], None), ("다음 날 종가 체결", rC["Vt45_종가"], None),
                ("주 1회 맞춤", fq["주 1회"], None), ("띠 5%(목표와 5%p 넘게 벌어질 때만)", fq["띠 5%"], None), ("띠 10%", fq["띠 10%"], None)]
    plc, plR, pb, pbR = d["위약_원형"], d["위약_실제_원형"], d["위약_묶음"], d["위약_실제_묶음"]
    pos_long = sum(1 for k in six if xp[k]["합성"]["Vt45"]["긴역사"]["초과"] > EPS)
    pos_real = sum(1 for k in six if xp[k]["실제"]["Vt45"]["실제전체"]["초과"] > EPS)
    mdd_ = lambda s: f"{s['1995~2026']['낙폭'] * 100:.1f}%"

    a("# 5단계 반박 검증: SOXL 목표 변동성 45%(Vt45) (2026-10-11)")
    a("")
    a("> 후보: w = clip(0.45 ÷ (3 × σ20), 0, 1) = 0.15 ÷ 기초 20일 변동성(상한 1). 기초 = ^SOX(~2001-07-13) + SOXX. "
      "t일 종가로 정해 t+1일 시가에 매일 맞춤, 나머지 현금 0%.")
    a(f"> 독립 재계산 `code/{ID}.py`(lev5·stock_lab 미사용, parquet만). 합성 추가 비용은 실제 SOXL로 다시 재서 연 {d['맞춤']['추가비용'] * 100:.2f}%"
      f"(원 갈래 1.5%), 일수익 상관 {d['맞춤']['일수익상관']:.3f}. 숫자 `results/{ID}.json`.")
    a("")
    a("## 결론 3줄")
    a("")
    a(f"1. **숫자는 맞다.** 1995~2026 연 {p(V['1995~2026']['연율'])}/{mdd_(V)}, 1995~2025 효율선초과 {pp(V[lg]['초과'])}, "
      f"실제 SOXL 2010~2026 {pp(real['Vt45']['실제전체']['초과'])}, 2026 뒤 {p(V['2026뒤']['수익'])}/{V['2026뒤']['낙폭'] * 100:.0f}%, "
      f"7월 쇼크 {p(V['2026_7월']['수익'])}가 모두 0.1%p 안에서 재현됐고 미래 정보 누수도 없었다.")
    a(f"2. **'21개 중 유일한 전멸 회피'는 깨진다.** 최대 낙폭 {mdd_(V)}가 −90% 문턱 바로 위라 미끄러짐 0.5%({mdd_(r5['Vt45'])}), "
      f"하루 늦게({mdd_(rL['Vt45_lag1'])}), 다음 날 종가 체결({mdd_(rC['Vt45_종가'])}), 주 1회({mdd_(fq['주 1회'])})에서 모두 −90%를 넘는다. "
      f"이웃 25개 중 전멸 회피는 σ* 0.45 이하 {surv(ns)}개(미끄러짐 0.5%면 {surv(n5)}개)로 사실상 '평균 비중이 낮다'는 효과다"
      f"(그냥 덜 담기 36%도 {d['덜담기_낙폭']['0.36'] * 100:.1f}%). 2026 앞·뒤·2025·긴 역사 4창을 모두 이긴 이웃은 {all4(ns)}/25개.")
    a(f"3. **남는 것은 SOXL 긴 역사 우위뿐이고 상품을 탄다.** 1995~2025 효율선초과는 이웃 {cnt(ns, lg)}/25·지연·미끄러짐·종가 체결에서 모두 플러스"
      f"({pp(rL5['Vt45_lag1'][lg]['초과'])}~{pp(max(ns[k][lg]['초과'] for k in keys))}), 날짜만 섞은 위약 20개 중 {plc[lg]['순위_값']}위다. "
      f"그러나 실제 SOXL 2010~2026은 하루 늦게+미끄러짐에서 {pp(real5['Vt45_lag1']['실제전체']['초과'])}, 같은 평균 비중 대비 {pp(real['Vt45']['실제전체']['같은노출초과'])}이고, "
      f"다른 3배 상품 6개에서 긴 역사 플러스 {pos_long}/6·실제 플러스 {pos_real}/6, KORU는 {pp(xp['KORU']['합성']['Vt45']['긴역사']['초과'])}"
      f"(실제 {pp(xp['KORU']['실제']['Vt45']['실제전체']['초과'])})이다. 판정 **confirmed_fragile**.")
    a("")
    a("## 재현 (원 갈래 주장 vs 독립 재계산)")
    a("")
    C = d["원주장"]
    rows = [("1995~2026 연율", C["1995~2026_연율"], V["1995~2026"]["연율"], "%"), ("1995~2026 최대 낙폭", C["1995~2026_낙폭"], V["1995~2026"]["낙폭"], "%"),
            ("1995~2025 효율선초과(연)", C["1995~2025_초과"], V[lg]["초과"], "%p"),
            ("　미끄러짐 0.5%", C["1995~2025_초과_slip"], r5["Vt45"][lg]["초과"], "%p"),
            ("　하루 늦게", C["1995~2025_초과_lag"], rL["Vt45_lag1"][lg]["초과"], "%p"),
            ("실제 SOXL 2010~2026 효율선초과(연)", C["실제_초과"], real["Vt45"]["실제전체"]["초과"], "%p"),
            ("　미끄러짐 0.5%", C["실제_초과_slip"], real5["Vt45"]["실제전체"]["초과"], "%p"),
            ("2026 앞 효율선초과", C["2026앞_초과"], V["2026앞"]["초과"], "%p"), ("2026 뒤 수익", C["2026뒤_수익"], V["2026뒤"]["수익"], "%"),
            ("2026 뒤 최대 낙폭", C["2026뒤_낙폭"], V["2026뒤"]["낙폭"], "%"), ("2025 효율선초과", C["2025_초과"], V["2025"]["초과"], "%p"),
            ("2026 7월 쇼크(6/23~7/29)", C["2026_7월"], V["2026_7월"]["수익"], "%"),
            ("그냥 보유 2026 뒤", C["보유_2026뒤"], r0["보유"]["2026뒤"]["수익"], "%"),
            ("그냥 보유 7월 쇼크", C["보유_2026_7월"], r0["보유"]["2026_7월"]["수익"], "%"),
            ("평균 비중(1995~2026)", C["평균노출"], V["1995~2026"]["평균노출"], "x"), ("연 매매(비율 변화 10%p 이상인 날)", C["연매매"], V["1995~2026"]["연매매"], "x"),
            ("연 회전(비율 변화 합)", C["연회전"], V["1995~2026"]["연회전"], "x")]
    a("| 항목 | 원 갈래 | 독립 재계산 | 차 |")
    a("|---|---|---|---|")
    for nm, c, m, u in rows:
        if u == "x":
            a(f"| {nm} | {c:.2f} | {m:.2f} | {m - c:+.2f} |")
        else:
            a(f"| {nm} | {c * 100:+.1f}{u} | {m * 100:+.1f}{u} | {(m - c) * 100:+.2f}%p |")
    a("")
    a(f"- '연 매매 6.9회'는 비율 변화가 10%p 이상인 날 수다. 매일 맞춤이라 비율을 실제로 고치는 날은 {V['1995~2026']['거래일비율'] * 100:.0f}%"
      f"(연 약 {V['1995~2026']['거래일비율'] * 252:.0f}일)다. 띠 5%로 바꾸면 {fq['띠 5%']['1995~2026']['거래일비율'] * 100:.0f}%로 준다(아래 표).")
    a("")
    a("## 흔들기 (합성 SOXL, 효율선초과 = 같은 최대 낙폭 덜 담기 대비)")
    a("")
    a("| 판 | 2026 앞 | 2026 뒤 | 2025 | 1995~2025(연) | 실제 SOXL 2010~2026(연) | 1995~2026 연율/최대 낙폭 | 전멸(≤−90%) | 연 매매 | 비율 고치는 날 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for nm, s, rr in variants:
        L_ = s["1995~2026"]
        a(f"| {nm} | {pp(s['2026앞']['초과'])} | {pp(s['2026뒤']['초과'])} | {pp(s['2025']['초과'])} | {pp(s[lg]['초과'])} | "
          f"{pp(rr['실제전체']['초과']) if rr else '—'} | {rm(L_, True)} | {'예' if L_['낙폭'] <= -0.9 else '아니오'} | {L_['연매매']:.1f} | "
          f"{L_['거래일비율'] * 100:.0f}% |")
    I = d["현금4%"]["Vt45"]
    a(f"| 현금 이자 연 4% | — | — | — | {pp(I[lg]['초과'])} | — | {rm(I['1995~2026'], True)} | {'예' if I['1995~2026']['낙폭'] <= -0.9 else '아니오'} | — | — |")
    a("")
    a(f"- 같은 변동성 덜 담기 대비(위험 맞춘 비교): 1995~2025 {pp(V[lg]['같은변동성초과'])}(미끄러짐 {pp(r5['Vt45'][lg]['같은변동성초과'])}, "
      f"하루 늦게 {pp(rL['Vt45_lag1'][lg]['같은변동성초과'])}), 실제 SOXL {pp(real['Vt45']['실제전체']['같은변동성초과'])}"
      f"(미끄러짐 {pp(real5['Vt45']['실제전체']['같은변동성초과'])}, 하루 늦게 {pp(real['Vt45_lag1']['실제전체']['같은변동성초과'])}).")
    a(f"- 같은 평균 비중 덜 담기 대비: 1995~2025 {pp(V[lg]['같은노출초과'])}, 실제 SOXL {pp(real['Vt45']['실제전체']['같은노출초과'])}. "
      f"실제 구간의 효율선 우위는 수익이 아니라 낙폭을 줄인 몫이다(최대 낙폭 {real['Vt45']['실제전체']['낙폭'] * 100:.0f}% vs 그냥 보유 "
      f"{real['보유']['실제전체']['낙폭'] * 100:.0f}%).")
    a(f"- 2026 앞은 체결 시점에 크게 흔들린다(기준 {pp(V['2026앞']['초과'])}, 하루 늦게 {pp(rL['Vt45_lag1']['2026앞']['초과'])}, "
      f"종가 체결 {pp(rC['Vt45_종가']['2026앞']['초과'])}). 같은 평균 비중 대비로는 세 판 모두 음수"
      f"({pp(V['2026앞']['같은노출초과'])}, {pp(rL['Vt45_lag1']['2026앞']['같은노출초과'])}, {pp(rC['Vt45_종가']['2026앞']['같은노출초과'])}).")
    a("")
    a("폭락 구간(합성, 수익/최대 낙폭):")
    a("")
    ck = ["2000~02", "2008", "2011", "2015~16", "2018Q4", "2020급락", "2022", "2025관세", "2026_7월"]
    a("| 판 | " + " | ".join(ck) + " |")
    a("|---|" + "---|" * len(ck))
    for nm, s in [("그냥 보유", r0["보유"]), ("덜 담기 33%", r0["F33"]), ("덜 담기 50%", r0["F50"]), ("Vt45 기준", V),
                  ("Vt45 미끄러짐 0.5%", r5["Vt45"]), ("Vt45 하루 늦게", rL["Vt45_lag1"]), ("Vt45 종가 체결", rC["Vt45_종가"])]:
        a(f"| {nm} | " + " | ".join(rm(s[k]) for k in ck) + " |")
    a("")
    a("## 이웃 (σ* 앞뒤 2칸 × 창 n 앞뒤 2칸 = 25개, 사전 고정)")
    a("")
    a("칸 = 1995~2025 효율선초과(연) / 1995~2026 최대 낙폭. 굵게 = 후보.")
    a("")
    a("| σ* \\ n | " + " | ".join(str(n) for n in NB_N) + " |")
    a("|---|" + "---|" * len(NB_N))
    for s_ in NB_S:
        cells = []
        for n_ in NB_N:
            k = f"σ*{s_:.2f}·n{n_}"
            c_ = f"{pp(ns[k][lg]['초과'])} / {ns[k]['1995~2026']['낙폭'] * 100:.1f}%"
            cells.append(f"**{c_}**" if (s_, n_) == (SSTAR, NWIN) else c_)
        a(f"| {s_:.2f} | " + " | ".join(cells) + " |")
    a("")
    a("| 기준 | 2026 앞 > 0 | 2026 뒤 > 0 | 2025 > 0 | 1995~2025 > 0 | 실제 SOXL > 0 | 4창 모두 > 0 | 전멸 회피 |")
    a("|---|---|---|---|---|---|---|---|")
    a(f"| 미끄러짐 0 | {cnt(ns, '2026앞')}/25 | {cnt(ns, '2026뒤')}/25 | {cnt(ns, '2025')}/25 | {cnt(ns, lg)}/25 | "
      f"{sum(1 for k in keys if nbR[k]['실제전체']['초과'] > EPS)}/25 | {all4(ns)}/25 | {surv(ns)}/25 |")
    a(f"| 미끄러짐 0.5% | {cnt(n5, '2026앞')}/25 | {cnt(n5, '2026뒤')}/25 | {cnt(n5, '2025')}/25 | {cnt(n5, lg)}/25 | — | {all4(n5)}/25 | {surv(n5)}/25 |")
    a("")
    a("- 2026 뒤는 n ≥ 20에서만 플러스(n 10·15는 음수), 2025는 n = 30에서 주로 플러스. 긴 역사만 25개 모두 플러스다.")
    a("")
    a("## 위약 (같은 평균 비율·비슷한 매매 수로 날짜만 원형으로 민 판 20개)")
    a("")
    a("| 창 | 실제 | 위약 중앙 | 실제 순위(21개 중) | 효율선초과 실제 / 위약 중앙 | 연 매매 실제 / 위약 중앙 | 최대 낙폭 실제 / 위약 중앙 |")
    a("|---|---|---|---|---|---|---|")
    for wn, v in [*plc.items(), *plR.items()]:
        f_ = (lambda x: f"연 {x * 100:+.1f}%") if wn in LONGW else p
        a(f"| {wn} | {f_(v['실제_값'])} | {f_(v['위약_값_중앙'])} | {v['순위_값']}위 | {pp(v['실제_초과'])} / {pp(v['위약_초과_중앙'])} | "
          f"{v['실제_연매매']:.1f} / {v['위약_연매매_중앙']:.1f} | {v['실제_낙폭'] * 100:.0f}% / {v['위약_낙폭_중앙'] * 100:.0f}% |")
    a("")
    a(f"- 보조(10·21일 묶음 순서 섞기, 매매 수는 늘어남): 2026 앞 {pb['2026앞']['순위_값']}위, 뒤 {pb['2026뒤']['순위_값']}위, 2025 {pb['2025']['순위_값']}위, "
      f"1995~2025 {pb[lg]['순위_값']}위, 실제 SOXL {pbR['실제전체']['순위_값']}위(효율선초과 {pbR['실제전체']['순위_초과']}위).")
    a("- 변동성 시점은 긴 역사(폭락 때 비중을 빨리 줄임)와 2026 뒤에서 무작위보다 나았고, 2026 앞은 무작위보다 못했고 2025는 중간이었다. "
      "실제 SOXL 2010~2026은 연율로 중간이고 낙폭만 크게 줄였다.")
    a("")
    a("## 다른 3배 상품 (파라미터 고정, 기초 시작 1년 뒤부터)")
    a("")
    a("| 상품 ← 기초 | 추가 비용 | 2026 앞 | 2026 뒤 | 2025 | 긴 역사(연) | 긴 역사 미끄러짐 0.5% | 긴 역사 Vt45 / 그냥 보유 | 실제 상품(연) | 실제 미끄러짐 0.5% |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for k in [*six, "KORU", "FAS", "LABU", "NUGT"]:
        v = xp[k]; s = v["합성"]["Vt45"]; h = v["합성"]["보유"]; s5 = v["합성_slip"]["Vt45"]
        rr, rr5 = v["실제"]["Vt45"]["실제전체"], v["실제_slip"]["Vt45"]["실제전체"]
        a(f"| {k} ← {v['기초']}{'' if k in six else ' (참고)'} | {v['맞춤']['추가비용'] * 100:+.1f}% | {pp(s['2026앞']['초과'])} | "
          f"{pp(s['2026뒤']['초과'])} | {pp(s['2025']['초과'])} | {pp(s['긴역사']['초과'])} ({v['긴역사_시작'][:4]}~) | {pp(s5['긴역사']['초과'])} | "
          f"{rm(s['긴역사'], True)} / {rm(h['긴역사'], True)} | {pp(rr['초과'])} | {pp(rr5['초과'])} |")
    a("")
    a("- 지정 6개 부호(긴 역사 합성): " + ", ".join(f"{k} {sg(xp[k]['합성']['Vt45']['긴역사']['초과'])}" for k in six) +
      ". 실제 상품: " + ", ".join(f"{k} {sg(xp[k]['실제']['Vt45']['실제전체']['초과'])}" for k in six) + ".")
    a("- 미국 대형·기술(나스닥100·S&P500·기술·금융·반도체)에서는 플러스, 소형주·신흥국·중국·한국·금광·바이오에서는 0 근처이거나 마이너스였다. "
      "'변동성이 오르면 줄이기'는 하락할 때 변동성이 튀는 성질이 강한 기초에서만 통했다.")
    a("")
    a("## 시점 감사")
    a("")
    for m in d["시점"]:
        a(f"- {m}")
    a("- ^SOX 시가는 1999~2001년 85~91%가 전일 종가와 같다. 그 구간의 '다음 날 시가 체결'은 사실상 '신호 낸 종가 체결'이라 조금 유리하다. "
      "다음 날 종가 체결 판(위 표)에서 긴 역사 효율선초과는 남지만 최대 낙폭은 −90%를 넘는다.")
    a("- 추가 비용(연 1.55%)은 2010~2026 전체로 맞춘 값이다. 규칙과 덜 담기에 똑같이 들어가 비교에는 중립이다.")
    a("")
    a("## 판정")
    a("")
    a("**confirmed_fragile.** 숫자 재현과 시점은 문제없다. 'SOXL 1995~2025에서 같은 낙폭 덜 담기보다 낫다'는 이웃·지연·미끄러짐·종가 체결에 버텼다. "
      "그러나 대표 주장인 '유일한 전멸 회피'는 실전 마찰 하나만 넣어도 사라진다. 2026 앞은 이웃 25개 모두 실패했고, 실제 SOXL 우위는 하루 늦게+미끄러짐에서 0이다. "
      "다른 상품에서는 절반만 플러스였고, 사용자가 함께 물은 KORU에서는 마이너스다.")
    a("")
    a("규칙으로 남길 수 있는 것(주문 수량 아님): SOXL 비율 w = 0.15 ÷ SOXX 20일 변동성(상한 1)은 '긴 역사에서 같은 낙폭 덜 담기보다 조금 나은 비중 조절'까지만 말할 수 있다. "
      "전멸을 막은 핵심은 평균 비중이 1/3~1/2이었다는 점이고, 2026 장세 합격 규칙은 아니다. KORU에는 이 규칙의 근거가 없다.")
    a("")
    a("## 한계")
    a("")
    a(f"- 시험 범위: 후보 1개 × 흔들기 10판, 이웃 25개(σ* 5 × n 5) + 맞춤 빈도 4판, 위약 2종 × 시드 {SEEDS}개 × 창 5개, 다른 상품 10개(지정 6 + 참고 4), "
      "효율선 w 101칸. '유일한 전멸 회피'는 T·V·VS·TV 18개만 독립 재계산했다(S 3개는 원 갈래 2000~02 −99.7~−99.9%로 문턱과 멀어 생략).")
    a("- 후보 자체가 21개 중 결과를 보고 골라 강조한 것이라 선택편향이 있다. 위약 3위/21은 단측 약 14% 수준이라 우연과 확실히 갈리지 않는다.")
    a("- 2001-07 전 기초는 배당 없는 가격 지수이고, ^NDX·^GSPC 기초 상품은 추가 비용이 음수(배당 3배 몫)로 맞춰졌다. 비용을 시대마다 같게 둔 근사다.")
    a("- 덜 담기도 매일 맞춤이라 미끄러짐 판에서 효율선도 같이 깎인다(원 갈래와 같은 조건).")
    a("")
    (RESULTS / f"{ID}.md").write_text("\n".join(A), encoding="utf-8")


if __name__ == "__main__":
    main()
