"""7단계 STKBUY(2026-10-11): 개별 종목이 폭락한 뒤 N거래일 지나 기계적으로 산다(마삼룰의 '기다렸다 산다'를 종목에).

사용자: "마삼룰이라고 있거든? 이거랑 컨셉이 비슷한데, 폭락 이후에는 어느 정도 기간 뒤에 기계적으로 매수하는 방법을 연구해볼래요. 다양하게 조합해서."
마삼룰(sondoctor.co.kr/664): 나스닥 −3% 날 전량 매도, 마지막 −3% 날로부터 한 달 + 1일 뒤 재매수(사이에 또 나오면 다시 셈).
이 갈래는 그 '기다림'을 개별 종목 폭락에 붙인다: 종목이 폭락한 날(사건)로부터 N거래일 기다렸다가 산다.

엔진: trade_lab(load_stocks·bad_bars·trade·entries). 엔진 파일은 고치지 않는다.
  - 속도를 위해 종목마다 '모든 진입일 e의 H일 거래 수익'을 미리 벡터로 계산한다(보유만·고정 −25%, 미끄러짐 0·0.5%).
    check()에서 trade_lab.trade와 같은 값인지(오차 0) 표본으로 대조한다.
  - 사건 S1·S2를 C0·N0으로 둔 진입은 trade_lab.entries('dip7'·'deep40')와 같은 날인지 대조한다.

사건(그날 종가로 판단, 같은 종목 같은 사건은 20거래일 이상 띄워 '사건 1번'으로 묶음)
  S1_x  : 하루 종가 변화 ≤ −x (x = 7·10·15%)
  S2_x  : 252일 최고 종가 대비 −x를 처음 넘은 날 (x = 30·40·50%)
  S3m_x : S1_x 이면서 같은 날 QQQ ≤ −2%(시장 동반 폭락) / S3i_x: 같은 날 QQQ > −0.5%(종목 고유 폭락)
  S4m_x : S2_x 이면서 그 종목 252일 고점일 → 사건일 QQQ ≤ −10%(시장 동반) / S4i_x: QQQ > −2%(고유)  ← 과제 격자에 더한 것(사전 선언).
          1·2단계 관찰('업종이 설명하는 −40%만 산 경우 매매 1회 성적이 좋았다')을 S2에서 직접 잰다.
대기·조건(사건일 s, 진입 e = 마지막 기준일 + 1 + N, e일 시가 체결)
  N ∈ {0, 1, 3, 5, 10, 21, 42}거래일.  N0 = 사건 다음 날 시가 바로 매수.
  C0 무조건: e = s + 1 + N
  C1 마삼룰식: 기다리는 N일 안에 같은 사건이 또 나오면 그날부터 다시 센다. e = 마지막 사건일 + 1 + N
  C2 싸게만: s + N일 종가 ≤ 사건일 종가일 때만 e = s + 1 + N에 산다(아니면 건너뜀)
  N0에서는 C1·C2가 C0와 같아 하나로 센다 → (N, C) 고유 조합 1 + 6 × 3 = 19.
보유·청산: H ∈ {21, 63, 126}거래일(2026 판정 21·63, 2025·맥락 63·126). 다리 = 보유만 | 고정 −25%(진입가 대비 장중).

비교(사건형)
  - 아무 날 매수(같은 종목): 같은 창에서 그 종목을 아무 날(모든 거래일) 샀을 때 H일 평균 수익. 거래마다 그 종목의 값과 짝으로 뺀다.
  - 사건 다음 날 바로 매수(N0): 같은 사건을 기다리지 않고 샀을 때와 짝으로 뺀다.
  - 같은 날 아무 종목(보조): 같은 진입일 전 종목 평균 대비.
  - 추가 −20% 확률: 산 뒤 H일 안에 저가가 진입가 −20% 이하로 내려간 비율(보유 경로 기준).
  - t: 진입 월 묶음(같은 달 거래는 서로 엮여 있다).

평가 규약(사전 고정, 확인 구간을 본 뒤 바꾸지 않는다)
  - 창(진입일 기준): 2026앞 1/2~5/29(청산도 5/29 안, 고르기), 2026뒤 6/1~10/9(확인), 2025(보조),
    맥락 2015~2024 각 해, 2020급락(2/19~3/23), 묶음 2015~2025.
  - 긴 역사 전방(대체): 종목 자료가 2014년부터라 1971~2005 고르기는 불가. 2015~2019(청산도 2019 안)로 고르고 2020~2025로 확인한다(H63).
  - 대표 후보 G1 = 2026앞 짝 평균차(규칙 − 아무 날 매수) 최대(거래 100건 이상), H ∈ {21, 63}, 다리 2.
    확인: 2026뒤 같은 H·다리, 2025는 H63(규약) + 같은 H 참고, 미끄러짐 0.5%.
  - 대체 긴 역사 GL = 2015~2019 짝 평균차 최대(H63, 100건 이상) → 2020~2025 확인, 2026은 결과만.
  - 비용: 수수료 0, 미끄러짐 0.5% 민감도(진입·손절에 적용), 달력 포트폴리오 현금 이자 0%(4% 한 줄).
  - 생존편향 점검: 사건 전 60일 평균 거래대금이 그날 종목표 상위 1/3인 사건만(대형) 따로.

실행: py p7_STKBUY.py          (전체 계산 → results/p7_STKBUY.json, .md)
      py p7_STKBUY.py --check  (대조 점검만)
      py p7_STKBUY.py --md     (json에서 md만 다시 쓰기)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

import trade_lab as T

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
OUT_JSON = RESULTS / "p7_STKBUY.json"
OUT_MD = RESULTS / "p7_STKBUY.md"

# ============================================================================ 사전 고정 격자
DROP1 = (7, 10, 15)                 # S1 하루 하락 %
DD2 = (30, 40, 50)                  # S2 252일 고점 대비 %
Q_MKT, Q_IDIO = -0.02, -0.005       # S3 같은 날 QQQ
SPAN_MKT, SPAN_IDIO = -0.10, -0.02  # S4 고점일→사건일 QQQ
WAITS = (0, 1, 3, 5, 10, 21, 42)
CONDS = ("C0", "C1", "C2")
HS = (21, 63, 126)
H26, HOTHER = (21, 63), (63, 126)
LEGS = ("hold", "fix25")
FIX_P = 0.25
SLIP = 0.005
GAP = 20
MIN_N = 100
START = "2014-01-01"

DEFS = ([f"S1_{x}" for x in DROP1] + [f"S2_{x}" for x in DD2]
        + [f"S3m_{x}" for x in DROP1] + [f"S3i_{x}" for x in DROP1]
        + [f"S4m_{x}" for x in DD2] + [f"S4i_{x}" for x in DD2])
NC = [(0, "C0")] + [(n, c) for n in WAITS if n > 0 for c in CONDS]
RULES = [f"{d}|N{n}|{c}" for d in DEFS for n, c in NC]

DEF_KO = {**{f"S1_{x}": f"하루 −{x}%" for x in DROP1}, **{f"S2_{x}": f"고점 −{x}% 첫날" for x in DD2},
          **{f"S3m_{x}": f"하루 −{x}% · 시장 동반(QQQ ≤ −2%)" for x in DROP1},
          **{f"S3i_{x}": f"하루 −{x}% · 고유(QQQ > −0.5%)" for x in DROP1},
          **{f"S4m_{x}": f"고점 −{x}% · 시장 동반(같은 기간 QQQ ≤ −10%)" for x in DD2},
          **{f"S4i_{x}": f"고점 −{x}% · 고유(같은 기간 QQQ > −2%)" for x in DD2}}
COND_KO = {"C0": "무조건", "C1": "재발 시 다시 셈", "C2": "사건일 종가 이하일 때만"}
LEG_KO = {"hold": "보유만", "fix25": "고정 −25%"}

# 창: 이름 → (진입 시작, 진입 끝, 청산 마감 또는 None)
WINDOWS = {
    "2026앞": ("2026-01-02", "2026-05-29", "2026-05-29"),
    "2026뒤": ("2026-06-01", "2026-10-09", None),
    "2025": ("2025-01-01", "2025-12-31", None),
    **{str(y): (f"{y}-01-01", f"{y}-12-31", None) for y in range(2015, 2025)},
    "2020급락": ("2020-02-19", "2020-03-23", None),
    "2015~2025": ("2015-01-01", "2025-12-31", None),
    "2015~2019": ("2015-01-01", "2019-12-31", "2019-12-31"),
    "2020~2025": ("2020-01-01", "2025-12-31", None),
}
SLIP_WINDOWS = ("2026앞", "2026뒤", "2025")
BIG_WINDOWS = ("2026앞", "2026뒤", "2025", "2015~2025")
N_CAND_2026 = len(RULES) * len(H26) * len(LEGS)
N_CAND_LONG = len(RULES) * len(LEGS)


# ============================================================================ 자료
def load_universe():
    data = T.load_stocks()
    allsyms = sorted(data)
    bad = [s for s in allsyms if T.bad_bars(data[s])]
    badset = set(bad)
    syms = [s for s in allsyms if s not in badset]
    return data, allsyms, bad, syms


def load_qqq() -> pd.Series:
    x = pd.read_parquet(CACHE / "p7_idx.parquet")
    q = x[x.sym == "QQQ"].set_index("date")["Adj Close"].sort_index()
    return q


class Pre:
    """종목마다 모든 진입일 e의 H일 거래 결과를 평평한 배열(float32)로. 위치 = off[종목] + e."""

    def __init__(self, data, syms):
        self.syms = syms
        self.bars = [data[s] for s in syms]
        ns = np.array([len(b.c) for b in self.bars])
        self.off = np.concatenate([[0], np.cumsum(ns)[:-1]])
        tot = int(ns.sum())
        self.R = {}          # (H, leg, slip) → 수익
        self.MAE = {}
        for H in HS:
            for leg in LEGS:
                self.R[(H, leg, 0.0)] = np.full(tot, np.nan, np.float32)
            self.R[(H, "fix25", SLIP)] = np.full(tot, np.nan, np.float32)
            self.MAE[H] = np.full(tot, np.nan, np.float32)
        for i, b in enumerate(self.bars):
            o, h, l, c = b.o, b.h, b.l, b.c
            n = len(c)
            for H in HS:
                if n < H + 2:
                    continue
                m = n - H + 1
                ow, lw = swv(o, H), swv(l, H)
                cend = c[H - 1:]
                sl = slice(self.off[i], self.off[i] + m)
                self.R[(H, "hold", 0.0)][sl] = cend / o[:m] - 1
                self.MAE[H][sl] = lw.min(1) / o[:m] - 1
                for slip in (0.0, SLIP):
                    entry = o[:m] * (1 + slip)
                    L = entry * (1 - FIX_P)
                    hit = (ow <= L[:, None]) | (lw <= L[:, None])
                    anyh = hit.any(1)
                    t = hit.argmax(1)
                    ot = ow[np.arange(m), t]
                    px = np.where(ot <= L, ot, L) * (1 - slip)
                    px = np.where(anyh, px, cend)
                    self.R[(H, "fix25", slip)][sl] = px / entry - 1

    def ret(self, H, leg, slip):
        if leg == "hold" and slip > 0:
            return (1 + self.R[(H, "hold", 0.0)]) / (1 + slip) - 1
        return self.R[(H, leg, slip)]


def grid(pre):
    G = np.array(sorted({d for b in pre.bars for d in b.days if d >= START}))
    gidx = [np.searchsorted(G, b.days) for b in pre.bars]
    pre.GI = np.concatenate(gidx)
    pre.NL = np.array([len(b.c) for b in pre.bars])
    elig = np.zeros(len(pre.GI), bool)                       # 252일 이력 뒤(e ≥ 253)
    for i, b in enumerate(pre.bars):
        elig[pre.off[i] + 253:pre.off[i] + len(b.c)] = True
    pre.ELIG = elig
    return G, gidx


# ============================================================================ 사건 → 진입
def raw_events(b, qret_s, qq_s):
    """사건 정의별 사건일(s) 배열. s는 252 ≤ s ≤ n−2(다음 날 거래 가능)."""
    c = b.c
    n = len(c)
    r1 = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
    dd = b.dd252()
    ok = np.zeros(n, bool)
    ok[252:n - 1] = True
    out = {}
    for x in DROP1:
        m = ok & (r1 <= -x / 100)
        out[f"S1_{x}"] = m
        out[f"S3m_{x}"] = m & (qret_s <= Q_MKT)
        out[f"S3i_{x}"] = m & (qret_s > Q_IDIO)
    for x in DD2:
        cond = dd <= -x / 100
        m = ok & np.concatenate([[False], cond[1:] & ~cond[:-1]])
        out[f"S2_{x}"] = m
        idx = np.flatnonzero(m)
        span = np.full(n, np.nan)
        for s in idx:
            p = s - 251 + int(np.argmax(c[s - 251:s + 1]))
            span[s] = qq_s[s] / qq_s[p] - 1
        out[f"S4m_{x}"] = m & (span <= SPAN_MKT)
        out[f"S4i_{x}"] = m & (span > SPAN_IDIO)
    return {k: np.flatnonzero(v) for k, v in out.items()}


def episodes(R: np.ndarray) -> np.ndarray:
    picked, last = [], -10 ** 9
    for s in R:
        if s - last >= GAP:
            picked.append(s)
            last = s
    return np.array(picked, int)


def entries_for(R: np.ndarray, P: np.ndarray, c: np.ndarray, N: int, C: str):
    """반환 (s0, e): s0 = 사건 1번의 첫 사건일, e = 진입일(시가). e ≤ n−1만."""
    n = len(c)
    if N == 0 or C == "C0":
        s0, e = P, P + 1 + N
    elif C == "C1":
        s0l, el, seen = [], [], set()
        for p in P:
            t = p
            while True:
                j = np.searchsorted(R, t, "right")
                if j < len(R) and R[j] <= t + N:
                    t = R[j]
                else:
                    break
            ee = t + 1 + N
            if ee in seen:
                continue
            seen.add(ee)
            s0l.append(p)
            el.append(ee)
        s0, e = np.array(s0l, int), np.array(el, int)
    elif C == "C2":
        k = P[P + N < n]
        k = k[c[k + N] <= c[k]]
        s0, e = k, k + 1 + N
    else:
        raise ValueError(C)
    keep = e <= n - 1
    return s0[keep], e[keep]


def build_trades(pre: Pre, G, gidx, qq: pd.Series, big_thr):
    """규칙별 진입 목록(종목 id, s0, e, 대형 여부)."""
    qret = qq.pct_change()
    out = {r: {"sid": [], "s0": [], "e": [], "big": []} for r in RULES}
    for i, b in enumerate(pre.bars):
        qret_s = qret.reindex(b.days).to_numpy()
        qq_s = qq.reindex(b.days).to_numpy()
        ev = raw_events(b, qret_s, qq_s)
        bigflag = big_thr[i]
        for d in DEFS:
            R = ev[d]
            if not len(R):
                continue
            P = episodes(R)
            for N, C in NC:
                s0, e = entries_for(R, P, b.c, N, C)
                if not len(e):
                    continue
                key = f"{d}|N{N}|{C}"
                out[key]["sid"].append(np.full(len(e), i))
                out[key]["s0"].append(s0)
                out[key]["e"].append(e)
                out[key]["big"].append(bigflag[s0])
    for r in RULES:
        o = out[r]
        for k in ("sid", "s0", "e", "big"):
            o[k] = np.concatenate(o[k]) if o[k] else np.array([], bool if k == "big" else int)
    return out


def big_flags(pre: Pre, G, gidx):
    """사건 전 60일 평균 거래대금(종가 × 거래량)이 그날 종목표 상위 1/3 이상이면 True. 종목별 [날] 배열."""
    M = np.full((len(G), len(pre.bars)), np.nan, np.float32)
    dvs = []
    for i, b in enumerate(pre.bars):
        dv = pd.Series(b.c * b.v).rolling(60, min_periods=60).mean().to_numpy()
        dvs.append(dv)
        gi = gidx[i]
        k = gi < len(G)
        k &= b.days >= START
        M[gi[k], i] = dv[k]
    thr = np.nanquantile(M, 2 / 3, axis=1)
    return [np.where(np.isfinite(dvs[i]), dvs[i] >= thr[np.minimum(gidx[i], len(G) - 1)], False) for i in range(len(pre.bars))]


# ============================================================================ 기준: 아무 날 · 같은 날
def window_bounds(b, w, H):
    d0, d1, ex = WINDOWS[w]
    a = max(b.idx(d0), 253)
    z = b.idx(d1, "right")             # 진입 e < z
    if ex is not None:
        z = min(z, b.idx(ex, "right") - H + 1)   # e + H − 1 ≤ 청산 마감
    return a, z


def any_day(pre: Pre):
    """창·H·다리·미끄러짐별: 종목마다 아무 날 평균(짝 기준), 전체 묶음 통계."""
    base, pooled = {}, {}
    keys = [(H, leg, 0.0) for H in HS for leg in LEGS] + [(H, leg, SLIP) for H in H26 for leg in LEGS]
    for w in WINDOWS:
        for (H, leg, slip) in keys:
            if slip > 0 and w not in SLIP_WINDOWS:
                continue
            R = pre.ret(H, leg, slip)
            mae = pre.MAE[H]
            per = np.full(len(pre.bars), np.nan)
            sx = sw = s20 = s30 = 0.0
            cnt = 0
            for i, b in enumerate(pre.bars):
                a, z = window_bounds(b, w, H)
                if z <= a:
                    continue
                x = R[pre.off[i] + a:pre.off[i] + z]
                mm = mae[pre.off[i] + a:pre.off[i] + z]
                k = np.isfinite(x)
                if not k.any():
                    continue
                x, mm = x[k].astype(float), mm[k].astype(float)
                per[i] = x.mean()
                sx += x.sum(); sw += (x > 0).sum(); s20 += (mm <= -0.2).sum(); s30 += (x <= -0.3).sum()
                cnt += len(x)
            base[(w, H, leg, slip)] = per
            pooled[(w, H, leg, slip)] = {"n": int(cnt), "mean": sx / cnt if cnt else np.nan, "win": sw / cnt if cnt else np.nan,
                                         "p20": s20 / cnt if cnt else np.nan, "p30": s30 / cnt if cnt else np.nan,
                                         "stock_mean": float(np.nanmean(per))}
    return base, pooled


def same_day(pre: Pre, G, gidx):
    """같은 진입일 전 종목 평균(보조 기준). (H, leg) → 날 배열."""
    out = {}
    for H in HS:
        for leg in LEGS:
            x = pre.ret(H, leg, 0.0).astype(float)
            k = np.isfinite(x) & pre.ELIG
            S = np.bincount(pre.GI[k], x[k], minlength=len(G))
            C = np.bincount(pre.GI[k], minlength=len(G)).astype(float)
            out[(H, leg)] = np.where(C > 0, S / np.maximum(C, 1), np.nan)
    return out


def stock_alpha(pre: Pre, xsd):
    """사후 추가 기준(결과를 본 뒤 더함): 종목 고유 평소 초과수익 alpha_i = 전 기간(2015~2026) 아무 날 (수익 − 같은 날 전 종목 평균)의 평균.
    d_adj = 규칙 수익 − 같은 날 전 종목 평균 − alpha_i. '같은 창 같은 종목' 기준은 폭락 자체를 품어 규칙에 유리하게 기운다."""
    out = {}
    lo = np.array([max(b.idx("2015-01-01"), 253) for b in pre.bars])
    for H in HS:
        for leg in LEGS:
            x = pre.ret(H, leg, 0.0).astype(float)
            ex = x - xsd[(H, leg)][pre.GI]
            a = np.full(len(pre.bars), np.nan)
            for i, b in enumerate(pre.bars):
                v = ex[pre.off[i] + lo[i]:pre.off[i] + len(b.c)]
                v = v[np.isfinite(v)]
                if len(v):
                    a[i] = v.mean()
            out[(H, leg)] = a
    return out


# ============================================================================ 칸 통계
def tstat(d, mon):
    if len(d) < 3:
        return np.nan
    u, inv = np.unique(mon, return_inverse=True)
    if len(u) < 3:
        return np.nan
    g = np.bincount(inv, d) / np.bincount(inv)
    sd = g.std(ddof=1)
    return float(g.mean() / (sd / np.sqrt(len(g)))) if sd > 0 else np.nan


def cell(ret, base, imm, xs, mae, mon, sid, alpha):
    n = len(ret)
    if n == 0:
        return {"n": 0}
    adj = ret - xs - alpha
    ka = np.isfinite(adj)
    d = ret - base
    di = ret - imm
    ki = np.isfinite(di)
    kx = np.isfinite(xs)
    return {"n": int(n), "nsym": int(len(np.unique(sid))), "mon": int(len(np.unique(mon))),
            "mean": float(ret.mean()), "med": float(np.median(ret)), "win": float((ret > 0).mean()),
            "p30": float((ret <= -0.3).mean()), "p20": float((mae <= -0.2).mean()),
            "base": float(base.mean()), "d_any": float(d.mean()), "t_any": tstat(d, mon),
            "d_imm": float(di[ki].mean()) if ki.any() else np.nan, "t_imm": tstat(di[ki], mon[ki]) if ki.any() else np.nan,
            "d_xs": float((ret - xs)[kx].mean()) if kx.any() else np.nan,
            "d_adj": float(adj[ka].mean()) if ka.any() else np.nan, "t_adj": tstat(adj[ka], mon[ka]) if ka.any() else np.nan}


def compute(pre: Pre, G, gidx, trades, base, xsd, alpha):
    month = np.array([int(d[:4]) * 12 + int(d[5:7]) for d in G])
    rows = []
    wb = {w: (WINDOWS[w][0], WINDOWS[w][1], WINDOWS[w][2]) for w in WINDOWS}
    t0 = time.time()
    for ri, r in enumerate(RULES):
        tr = trades[r]
        if not len(tr["e"]):
            continue
        sid, s0, e, big = tr["sid"], tr["s0"], tr["e"], tr["big"]
        off = pre.off[sid]
        ge = pre.GI[off + e]
        dates = G[ge]
        mon = month[ge]
        for H in HS:
            mae = pre.MAE[H][off + e].astype(float)
            # 청산일(e + H − 1) 날짜: 같은 종목 안에서
            zi = e + H - 1
            nlen = pre.NL[sid]
            valid_len = zi < nlen
            zdate = np.where(valid_len, G[pre.GI[off + np.minimum(zi, nlen - 1)]], "9999")
            for leg in LEGS:
                for slip in (0.0, SLIP):
                    if slip > 0 and H not in H26:
                        continue
                    R = pre.ret(H, leg, slip)
                    ret = R[off + e].astype(float)
                    imm = R[off + s0 + 1].astype(float)
                    xs = xsd[(H, leg)][ge] if slip == 0 else np.full(len(e), np.nan)
                    al = alpha[(H, leg)][sid]
                    ok = np.isfinite(ret) & valid_len
                    for w, (d0, d1, ex) in wb.items():
                        if slip > 0 and w not in SLIP_WINDOWS:
                            continue
                        m = ok & (dates >= d0) & (dates <= d1)
                        if ex is not None:
                            m &= zdate <= ex
                        bvec = base[(w, H, leg, slip)][sid]
                        m &= np.isfinite(bvec)
                        subs = [("all", m)]
                        if slip == 0 and H in H26 and w in BIG_WINDOWS:
                            subs.append(("big", m & big))
                        for sub, mm in subs:
                            st = cell(ret[mm], bvec[mm], imm[mm], xs[mm], mae[mm], mon[mm], sid[mm], al[mm])
                            d, nc = r.split("|")[0], r.split("|")[1:]
                            rows.append({"rule": r, "def": d, "N": int(nc[0][1:]), "C": nc[1], "w": w, "H": H, "leg": leg,
                                         "slip": slip, "sub": sub, **st})
        if ri % 40 == 0:
            print(f"  규칙 {ri + 1}/{len(RULES)} {r} 거래 {len(e)} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# ============================================================================ 달력 포트폴리오(긴 역사 연율·낙폭)
CAL_SPAN = ("2015-01-02", "2025-12-31")


def calendar(pre: Pre, G, gidx, tr, H, leg, cash=0.0, w="2015~2025"):
    d0, d1, _ = WINDOWS[w]
    S = np.zeros(len(G)); C = np.zeros(len(G))
    rc = cash / 252
    ntr = 0
    for i, s0, e in zip(tr["sid"], tr["s0"], tr["e"]):
        b = pre.bars[i]
        if e + H - 1 >= len(b.c) or not (d0 <= b.days[e] <= d1):
            continue
        entry = b.o[e]
        path = b.c[e:e + H] / entry
        k_exit = None
        if leg == "fix25":
            L = entry * (1 - FIX_P)
            hit = np.flatnonzero((b.o[e:e + H] <= L) | (b.l[e:e + H] <= L))
            if len(hit):
                k_exit = hit[0]
                ot = b.o[e + k_exit]
                path = path.copy()
                path[k_exit:] = (ot if ot <= L else L) / entry
        r = path / np.concatenate([[1.0], path[:-1]]) - 1
        if k_exit is not None and rc:
            r[k_exit + 1:] = rc
        gi = gidx[i][e:e + H]
        S[gi] += r
        C[gi] += 1
        ntr += 1
    span = (G >= CAL_SPAN[0]) & (G <= CAL_SPAN[1])
    r = np.where(C > 0, S / np.maximum(C, 1), rc)[span]
    eq = np.cumprod(1 + r)
    eq1 = np.concatenate([[1.0], eq])
    mdd = float((eq1 / np.maximum.accumulate(eq1) - 1).min())
    yrs = len(r) / 252
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1), "mdd": mdd, "tot": float(eq[-1] - 1), "expo": float((C[span] > 0).mean()),
            "per_year": ntr / 11.0, "avg_open": float(C[span].mean())}


def calendar_ew(pre: Pre, G, gidx, cash=0.0):
    """아무 날 매수 기준: 종목표 동일가중(매일 재조정, 252일 이력 있는 종목)."""
    r = np.concatenate([np.concatenate([[np.nan], b.c[1:] / b.c[:-1] - 1]) for b in pre.bars])
    k = np.isfinite(r) & pre.ELIG
    S = np.bincount(pre.GI[k], r[k], minlength=len(G))
    C = np.bincount(pre.GI[k], minlength=len(G)).astype(float)
    span = (G >= CAL_SPAN[0]) & (G <= CAL_SPAN[1])
    r = np.where(C > 0, S / np.maximum(C, 1), cash / 252)[span]
    eq1 = np.concatenate([[1.0], np.cumprod(1 + r)])
    return {"cagr": float(eq1[-1] ** (252 / len(r)) - 1), "mdd": float((eq1 / np.maximum.accumulate(eq1) - 1).min()),
            "tot": float(eq1[-1] - 1)}


# ============================================================================ 점검
def check(data, syms, pre: Pre, trades=None):
    rng = np.random.default_rng(7)
    # 1) 미리 계산한 거래 수익 = trade_lab.trade
    worst = 0.0
    nchk = 0
    longs = [i for i, b in enumerate(pre.bars) if len(b.c) > 600]
    for i in rng.choice(longs, 40, replace=False):
        b = pre.bars[i]
        for e in rng.integers(253, len(b.c) - 130, 15):
            for H in HS:
                for leg in LEGS:
                    for slip in (0.0, SLIP):
                        lg = [T.Leg(1.0, "hold")] if leg == "hold" else [T.Leg(1.0, "fixed", FIX_P)]
                        ref = T.trade(b, int(e), H, lg, slip)
                        mine = pre.ret(H, leg, slip)[pre.off[i] + e]
                        worst = max(worst, abs(ref["ret"] - float(mine)))
                        if leg == "hold" and slip == 0:
                            worst = max(worst, abs(ref["mae"] - float(pre.MAE[H][pre.off[i] + e])))
                        nchk += 1
    print(f"점검1 미리 계산 수익 vs trade_lab.trade 최대 오차(float32 반올림 수준이어야 함): {worst:.2e} ({nchk}건)")
    # 2) S1_7·S2_40 N0 진입 = trade_lab.entries(dip7·deep40)
    qq = load_qqq()
    qret = qq.pct_change()
    mism = 0
    tot = 0
    for i in rng.choice(len(pre.bars), 200, replace=False):
        b = pre.bars[i]
        ev = raw_events(b, qret.reindex(b.days).to_numpy(), qq.reindex(b.days).to_numpy())
        for d, kind in (("S1_7", "dip7"), ("S2_40", "deep40")):
            mine = entries_for(ev[d], episodes(ev[d]), b.c, 0, "C0")[1]
            ref = T.entries(b, kind, "2014-01-01", "2026-10-09", GAP, 10)
            tot += 1
            mism += int(not np.array_equal(np.sort(mine), np.sort(ref)))
    print(f"점검2 N0 진입 vs trade_lab.entries 불일치 종목(0이어야 함): {mism}/{tot}")
    # 3) C1 하루씩 도는 독립 구현
    bad3 = 0
    n3 = 0
    for i in rng.choice(len(pre.bars), 200, replace=False):
        b = pre.bars[i]
        ev = raw_events(b, qret.reindex(b.days).to_numpy(), qq.reindex(b.days).to_numpy())
        for d in ("S1_7", "S1_10", "S2_30"):
            R = ev[d]
            if not len(R):
                continue
            israw = np.zeros(len(b.c), bool)
            israw[R] = True
            P = episodes(R)
            for N in (5, 21, 42):
                _, e1 = entries_for(R, P, b.c, N, "C1")
                ref = []
                for p in P:
                    t, k = p, p + 1
                    while k <= t + N and k < len(b.c):
                        if israw[k]:
                            t = k
                        k += 1
                    ee = t + 1 + N
                    if ee <= len(b.c) - 1 and ee not in ref:
                        ref.append(ee)
                n3 += 1
                bad3 += int(sorted(ref) != sorted(e1.tolist()))
    print(f"점검3 C1(재발 시 다시 셈) 하루씩 독립 구현 불일치(0이어야 함): {bad3}/{n3}")
    # 4) 미래 교란: T일 뒤 가격을 바꿔도 e ≤ T 진입은 같아야 한다
    bad4 = 0
    n4 = 0
    for i in rng.choice(len(pre.bars), 60, replace=False):
        b = pre.bars[i]
        if len(b.c) < 800:
            continue
        Tm = int(rng.integers(500, len(b.c) - 100))
        b2 = type(b)(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
        f = 1 + rng.normal(0, 0.08, len(b.c) - Tm)
        for arr in (b2.o, b2.h, b2.l, b2.c):
            arr[Tm:] *= f
        ev1 = raw_events(b, qret.reindex(b.days).to_numpy(), qq.reindex(b.days).to_numpy())
        ev2 = raw_events(b2, qret.reindex(b.days).to_numpy(), qq.reindex(b.days).to_numpy())
        for d in ("S1_7", "S2_40", "S3i_7"):
            for N, C in NC:
                e1 = entries_for(ev1[d], episodes(ev1[d]), b.c, N, C)[1]
                e2 = entries_for(ev2[d], episodes(ev2[d]), b2.c, N, C)[1]
                n4 += 1
                bad4 += int(not np.array_equal(e1[e1 <= Tm], e2[e2 <= Tm]))
    print(f"점검4 미래 교란 뒤 진입 불일치(0이어야 함): {bad4}/{n4}")
    return {"ret_err": worst, "n_ret": nchk, "entries_mismatch": mism, "c1_mismatch": bad3, "future_mismatch": bad4}


# ============================================================================ 고르기
def sel_table(cells: pd.DataFrame):
    a = cells[(cells["sub"] == "all") & (cells["slip"] == 0.0)]
    return a.set_index(["rule", "w", "H", "leg"])


def getc(A, rule, w, H, leg, key):
    try:
        v = A.loc[(rule, w, H, leg), key]
        return float(v) if np.isfinite(v) else np.nan
    except KeyError:
        return np.nan


def selection(cells: pd.DataFrame) -> dict:
    A = sel_table(cells)
    S = cells[(cells["sub"] == "all") & (cells["slip"] == SLIP)].set_index(["rule", "w", "H", "leg"])
    cand = [(r, H, leg) for r in RULES for H in H26 for leg in LEGS]
    rows = []
    for r, H, leg in cand:
        g = lambda w, k, h=H: getc(A, r, w, h, leg, k)
        rows.append({"rule": r, "H": H, "leg": leg, "앞_n": g("2026앞", "n"), "앞_차": g("2026앞", "d_any"), "앞_t": g("2026앞", "t_any"),
                     "앞_평균": g("2026앞", "mean"), "앞_즉시차": g("2026앞", "d_imm"),
                     "뒤_n": g("2026뒤", "n"), "뒤_차": g("2026뒤", "d_any"), "뒤_t": g("2026뒤", "t_any"), "뒤_평균": g("2026뒤", "mean"),
                     "뒤_즉시차": g("2026뒤", "d_imm"),
                     "y25_n": g("2025", "n", 63), "y25_차": g("2025", "d_any", 63), "y25_t": g("2025", "t_any", 63),
                     "y25_차_같은H": g("2025", "d_any"),
                     "뒤_차_slip": float(S.loc[(r, "2026뒤", H, leg), "d_any"]) if (r, "2026뒤", H, leg) in S.index else np.nan})
    df = pd.DataFrame(rows)
    pool = df[df["앞_n"] >= MIN_N]
    g1 = pool.sort_values("앞_차", ascending=False).iloc[0].to_dict()

    def rate(sub, col):
        sub = sub[np.isfinite(sub[col])]
        return [float((sub[col] > 0).mean()) if len(sub) else np.nan, int(len(sub))]

    pos = pool[pool["앞_차"] > 0]
    succ = {"후보": int(len(df)), "100건+": int(len(pool)), "앞>0": int(len(pos)),
            "뒤>0|앞>0": rate(pos, "뒤_차"), "뒤>0|무조건": rate(pool, "뒤_차"),
            "2025>0|앞>0": rate(pos, "y25_차"), "2025>0|무조건": rate(pool, "y25_차")}
    top = pool.sort_values("앞_차", ascending=False).head(10).to_dict("records")
    # 긴 역사 대체: 2015~2019 → 2020~2025 (H63)
    rows = []
    for r in RULES:
        for leg in LEGS:
            g = lambda w, k: getc(A, r, w, 63, leg, k)
            rows.append({"rule": r, "H": 63, "leg": leg, "앞_n": g("2015~2019", "n"), "앞_차": g("2015~2019", "d_any"),
                         "앞_t": g("2015~2019", "t_any"), "뒤_n": g("2020~2025", "n"), "뒤_차": g("2020~2025", "d_any"),
                         "뒤_t": g("2020~2025", "t_any"), "뒤_평균": g("2020~2025", "mean"),
                         "y26앞_n": g("2026앞", "n"), "y26앞_차": g("2026앞", "d_any"), "y26뒤_n": g("2026뒤", "n"),
                         "y26뒤_차": g("2026뒤", "d_any"), "y26뒤_t": g("2026뒤", "t_any")})
    dl = pd.DataFrame(rows)
    pl = dl[dl["앞_n"] >= MIN_N]
    gl = pl.sort_values("앞_차", ascending=False).iloc[0].to_dict()
    posl = pl[pl["앞_차"] > 0]
    succl = {"후보": int(len(dl)), "100건+": int(len(pl)), "앞>0": int(len(posl)),
             "뒤>0|앞>0": rate(posl, "뒤_차"), "뒤>0|무조건": rate(pl, "뒤_차"),
             "2026뒤>0|앞>0": rate(posl, "y26뒤_차"), "2026뒤>0|무조건": rate(pl, "y26뒤_차")}
    topl = pl.sort_values("앞_차", ascending=False).head(10).to_dict("records")
    # 사후 참고(결과를 본 뒤 더함): 같은 절차를 d_adj(같은 날 전 종목·종목 평소 초과 뺌)로
    rows = []
    for r, H, leg in cand:
        g = lambda w, k, h=H: getc(A, r, w, h, leg, k)
        rows.append({"rule": r, "H": H, "leg": leg, "앞_n": g("2026앞", "n"), "앞_adj": g("2026앞", "d_adj"), "앞_t": g("2026앞", "t_adj"),
                     "뒤_n": g("2026뒤", "n"), "뒤_adj": g("2026뒤", "d_adj"), "뒤_t": g("2026뒤", "t_adj"),
                     "y25_adj": g("2025", "d_adj", 63), "y25_t": g("2025", "t_adj", 63), "y25_adj_같은H": g("2025", "d_adj"),
                     "앞_차": g("2026앞", "d_any"), "뒤_차": g("2026뒤", "d_any")})
    da = pd.DataFrame(rows)
    pa = da[da["앞_n"] >= MIN_N]
    ga = pa.sort_values("앞_adj", ascending=False).iloc[0].to_dict()
    posa = pa[pa["앞_adj"] > 0]
    succa = {"후보": int(len(da)), "100건+": int(len(pa)), "앞>0": int(len(posa)),
             "뒤>0|앞>0": rate(posa, "뒤_adj"), "뒤>0|무조건": rate(pa, "뒤_adj"),
             "2025>0|앞>0": rate(posa, "y25_adj"), "2025>0|무조건": rate(pa, "y25_adj")}
    topa = pa.sort_values("앞_adj", ascending=False).head(10).to_dict("records")
    rows = []
    for r in RULES:
        for leg in LEGS:
            g = lambda w, k: getc(A, r, w, 63, leg, k)
            rows.append({"rule": r, "H": 63, "leg": leg, "앞_n": g("2015~2019", "n"), "앞_adj": g("2015~2019", "d_adj"),
                         "앞_t": g("2015~2019", "t_adj"), "뒤_n": g("2020~2025", "n"), "뒤_adj": g("2020~2025", "d_adj"),
                         "뒤_t": g("2020~2025", "t_adj"), "y26뒤_n": g("2026뒤", "n"), "y26뒤_adj": g("2026뒤", "d_adj"),
                         "y26앞_adj": g("2026앞", "d_adj")})
    dla = pd.DataFrame(rows)
    pla = dla[dla["앞_n"] >= MIN_N]
    gla = pla.sort_values("앞_adj", ascending=False).iloc[0].to_dict()
    posla = pla[pla["앞_adj"] > 0]
    succla = {"후보": int(len(dla)), "100건+": int(len(pla)), "앞>0": int(len(posla)),
              "뒤>0|앞>0": rate(posla, "뒤_adj"), "뒤>0|무조건": rate(pla, "뒤_adj"),
              "2026뒤>0|앞>0": rate(posla, "y26뒤_adj"), "2026뒤>0|무조건": rate(pla, "y26뒤_adj")}
    return {"G1": g1, "succ": succ, "top10": top, "GL": gl, "succL": succl, "top10L": topl,
            "G1adj": ga, "succAdj": succa, "top10Adj": topa, "GLadj": gla, "succLAdj": succla}


# ============================================================================ 표 만들기
H_UT = 63
UT_WINDOWS = [("2026앞", "2026 앞"), ("2026뒤", "2026 뒤"), ("2025", "2025"), ("2022", "2022"), ("2020", "2020"),
              ("2015~2019", "2015~19")]


def _ucell(V, rule, w, H, leg):
    m, d, a, n = (V.g(rule, w, H, leg, k) for k in ("mean", "d_any", "d_adj", "n"))
    if not np.isfinite(n) or n == 0:
        return "—"
    s = f"{pct(m)} ({pp(d)} / {pp(a)})"
    return s + (f" n{int(n)}" if w.startswith("2026") else "")


def user_table(J, V):
    sel, cal = J["selection"], J["calendar"]
    head = ("| 규칙 (H = 보유 거래일) | " + " | ".join(f"{lbl} ({H_UT}일)" for _, lbl in UT_WINDOWS)
            + " | 2008 · 2000~02 · 1987 | 2015~2025 달력 연율 / 최대 낙폭 / 연 매매 수 |")
    out = [head, "|" + "---|" * (len(UT_WINDOWS) + 3)]
    ew = cal["EW"]["c0"]
    out.append("| 아무 날 매수(기저율, H63, 전 종목 평균) | "
               + " | ".join(pct(V.pooled(w, 63, "hold", "mean")) for w, _ in UT_WINDOWS)
               + f" | 자료 없음 | 동일가중 보유 {pct(ew['cagr'])} / {pct(ew['mdd'])} / 매일 |")
    rows = [(r, H, leg, "") for r, H, leg in USER_RULES]
    g1, gl = sel["G1"], sel["GL"]
    rows.append((g1["rule"], int(g1["H"]), g1["leg"], "**대표 후보 G1**(2026 앞 고르기) "))
    rows.append((gl["rule"], 63, gl["leg"], "**긴 역사 대체 GL**(2015~19 고르기) "))
    for r, H, leg, tag in rows:
        k = f"{r}#{H}#{leg}"
        c = cal.get(k, {}).get("c0")
        longc = f"{pct(c['cagr'])} / {pct(c['mdd'])} / {c['per_year']:,.0f}" if c else "—"
        out.append(f"| {tag}{rule_ko(r, H, leg)} | " + " | ".join(_ucell(V, r, w, H, leg) for w, _ in UT_WINDOWS)
                   + f" | 자료 없음 | {longc} |")
    return out


def grid_tables(J, V):
    out = []
    Ns = WAITS
    hdr = "| 사건 | " + " | ".join(f"N{n}" for n in Ns) + " |"
    sep = "|" + "---|" * (len(Ns) + 1)
    out.append("**① 2015~2025 묶음, 63일 보유** (칸 = 아무 날 대비 %p, 굵게 = 월 묶음 t ≥ 2; 행 끝 = N0 거래 수)\n")
    out += [hdr[:-1] + "| N0 거래 |", sep + "---|"]
    for d in DEFS:
        cells = []
        for n in Ns:
            r = f"{d}|N{n}|C0"
            x, t = V.g(r, "2015~2025", 63, "hold", "d_any"), V.g(r, "2015~2025", 63, "hold", "t_any")
            s = pp(x)
            cells.append(f"**{s}**" if np.isfinite(t) and abs(t) >= 2 else s)
        out.append(f"| {DEF_KO[d]} | " + " | ".join(cells) + f" | {V.g(f'{d}|N0|C0', '2015~2025', 63, 'hold', 'n'):,.0f} |")
    out.append("")
    out.append("**② 2026 앞(고르기) / 뒤(확인), 21일 보유** (칸 = 앞 / 뒤, 아무 날 대비 %p)\n")
    out += [hdr, sep]
    for d in DEFS:
        cells = []
        for n in Ns:
            r = f"{d}|N{n}|C0"
            cells.append(f"{pp(V.g(r, '2026앞', 21, 'hold', 'd_any'))} / {pp(V.g(r, '2026뒤', 21, 'hold', 'd_any'))}")
        out.append(f"| {DEF_KO[d]} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def wait_table(J, V):
    """기다린 값(C0만 공정): 같은 사건을 다음 날 바로 산 것 대비 %p."""
    Ns = [n for n in WAITS if n > 0]
    out = ["칸 = 2015~25(63일) · 2026 앞(21일) · 2026 뒤(21일) · 2025(63일), 같은 사건을 다음 날 바로 산 거래 대비 %p(무조건 C0). "
           "굵게 = 네 구간 모두 마이너스.\n",
           "| 사건 | " + " | ".join(f"N{n}" for n in Ns) + " |", "|" + "---|" * (len(Ns) + 1)]
    for d in DEFS:
        cells = []
        for n in Ns:
            r = f"{d}|N{n}|C0"
            v = [V.g(r, "2015~2025", 63, "hold", "d_imm"), V.g(r, "2026앞", 21, "hold", "d_imm"),
                 V.g(r, "2026뒤", 21, "hold", "d_imm"), V.g(r, "2025", 63, "hold", "d_imm")]
            s = " · ".join(pp(x) for x in v)
            allneg = all(np.isfinite(x) and x < 0 for x in v)
            cells.append(f"**{s}**" if allneg else s)
        out.append(f"| {DEF_KO[d]} | " + " | ".join(cells) + " |")
    return out


def adj_table(J, V):
    out = ["칸 = 2015~25(63일) · 2026 앞(21일) · 2026 뒤(21일) · 2025(63일), d_adj = 수익 − 같은 날 전 종목 평균 − 그 종목 평소 초과(%p, 무조건 C0 · 보유만).\n",
           "| 사건 | " + " | ".join(f"N{n}" for n in WAITS) + " |", "|" + "---|" * (len(WAITS) + 1)]
    for d in DEFS:
        cells = []
        for n in WAITS:
            r = f"{d}|N{n}|C0"
            v = [V.g(r, "2015~2025", 63, "hold", "d_adj"), V.g(r, "2026앞", 21, "hold", "d_adj"),
                 V.g(r, "2026뒤", 21, "hold", "d_adj"), V.g(r, "2025", 63, "hold", "d_adj")]
            cells.append(" · ".join(pp(x) for x in v))
        out.append(f"| {DEF_KO[d]} | " + " | ".join(cells) + " |")
    return out


MKT_WINDOWS = [("2015~2025", 63, "all"), ("2015~2025", 63, "big"), ("2020급락", 63, "all"), ("2022", 63, "all"), ("2025", 63, "all"),
               ("2026앞", 21, "all"), ("2026뒤", 21, "all")]


def mkt_table(J, V):
    out = ["칸 = 시장 동반 / 고유, d_adj(같은 날 전 종목 평균과 그 종목 평소 초과를 뺀 %p, 보유만·무조건). 위 6행: 다음 날 바로(N0), 아래 6행: 21거래일 뒤(N21). "
           "'대형' = 사건 전 60일 거래대금이 그날 상위 1/3인 사건만.\n",
           "| 사건 · 대기 | " + " | ".join(f"{w}{' 대형' if sub == 'big' else ''} ({H}일)" for w, H, sub in MKT_WINDOWS) + " | 거래 수 2015~25 동반/고유 |",
           "|" + "---|" * (len(MKT_WINDOWS) + 2)]
    pairs = ([("S3m_" + str(x), "S3i_" + str(x), f"하루 −{x}%") for x in DROP1]
             + [("S4m_" + str(x), "S4i_" + str(x), f"고점 −{x}%") for x in DD2])
    for N in (0, 21):
        for m, i, lbl in pairs:
            rm, ri = f"{m}|N{N}|C0", f"{i}|N{N}|C0"
            cells = [f"{pp(V.g(rm, w, H, 'hold', 'd_adj', sub=sub))} / {pp(V.g(ri, w, H, 'hold', 'd_adj', sub=sub))}" for w, H, sub in MKT_WINDOWS]
            nn = f"{V.g(rm, '2015~2025', 63, 'hold', 'n'):,.0f} / {V.g(ri, '2015~2025', 63, 'hold', 'n'):,.0f}"
            out.append(f"| {lbl} · N{N} | " + " | ".join(cells) + f" | {nn} |")
    return out


def select_table(J, V):
    sel = J["selection"]
    g1, gl, ga, gla = sel["G1"], sel["GL"], sel["G1adj"], sel["GLadj"]
    s1, sl, sa, sla = sel["succ"], sel["succL"], sel["succAdj"], sel["succLAdj"]
    out = ["| 판정 | 고른 규칙 | 고르기 구간 | 확인 구간 | 보조 | 미끄러짐 0.5% (뒤) |", "|---|---|---|---|---|---|"]
    out.append(f"| **2026 (사전 고정)** 앞 짝 평균차 최대, 100건+ | {rule_ko(g1['rule'], int(g1['H']), g1['leg'])} | "
               f"2026 앞 {pp(g1['앞_차'])}%p (t {tv(g1['앞_t'])}, n {int(g1['앞_n'])}) | 2026 뒤 {pp(g1['뒤_차'])}%p (n {int(g1['뒤_n'])}) | "
               f"2025(63일) {pp(g1['y25_차'])}%p (t {tv(g1['y25_t'])}) | {pp(g1['뒤_차_slip'])}%p |")
    out.append(f"| **긴 역사 대체 (사전 고정)** 2015~19 짝 평균차 최대 | {rule_ko(gl['rule'], 63, gl['leg'])} | "
               f"2015~19 {pp(gl['앞_차'])}%p (t {tv(gl['앞_t'])}, n {int(gl['앞_n'])}) | 2020~25 {pp(gl['뒤_차'])}%p (t {tv(gl['뒤_t'])}, n {int(gl['뒤_n'])}) | "
               f"2026 63일 칸 0건. 21일: 앞 {pp(V.g(gl['rule'], '2026앞', 21, gl['leg'], 'd_any'))}%p (n {V.g(gl['rule'], '2026앞', 21, gl['leg'], 'n'):.0f}) · "
               f"뒤 {pp(V.g(gl['rule'], '2026뒤', 21, gl['leg'], 'd_any'))}%p (n {V.g(gl['rule'], '2026뒤', 21, gl['leg'], 'n'):.0f}), "
               f"d_adj 앞 {pp(V.g(gl['rule'], '2026앞', 21, gl['leg'], 'd_adj'))} · 뒤 {pp(V.g(gl['rule'], '2026뒤', 21, gl['leg'], 'd_adj'))}%p | — |")
    out.append(f"| 사후 참고: 2026 앞 d_adj 최대 | {rule_ko(ga['rule'], int(ga['H']), ga['leg'])} | "
               f"2026 앞 {pp(ga['앞_adj'])}%p (t {tv(ga['앞_t'])}, n {int(ga['앞_n'])}) | 2026 뒤 {pp(ga['뒤_adj'])}%p (n {int(ga['뒤_n'])}) | "
               f"2025(63일) {pp(ga['y25_adj'])}%p (t {tv(ga['y25_t'])}) | — |")
    out.append(f"| 사후 참고: 2015~19 d_adj 최대 | {rule_ko(gla['rule'], 63, gla['leg'])} | "
               f"2015~19 {pp(gla['앞_adj'])}%p (t {tv(gla['앞_t'])}, n {int(gla['앞_n'])}) | 2020~25 {pp(gla['뒤_adj'])}%p (t {tv(gla['뒤_t'])}, n {int(gla['뒤_n'])}) | "
               f"2026 뒤 {pp(gla['y26뒤_adj'])}%p (n {int(gla['y26뒤_n'])}) | — |")
    out.append("")
    out.append("고르기가 정보를 주나: '고르기 구간 > 0' 규칙이 확인 구간에서도 > 0인 비율 vs 아무 규칙이나(무조건).\n")
    out.append("| 기준 · 구간 | 고르기 > 0 / 100건+ 후보 | 확인 > 0: 고르기>0 집합 / 무조건 | 보조 > 0: 고르기>0 집합 / 무조건 |\n|---|---|---|---|")
    f = lambda x: pct(x[0], 0, False)
    out.append(f"| 같은 종목 아무 날(사전) · 2026 앞→뒤, 2025 | {s1['앞>0']} / {s1['100건+']} | {f(s1['뒤>0|앞>0'])} / {f(s1['뒤>0|무조건'])} | "
               f"{f(s1['2025>0|앞>0'])} / {f(s1['2025>0|무조건'])} |")
    out.append(f"| 같은 종목 아무 날(사전) · 2015~19→2020~25, 2026 뒤 | {sl['앞>0']} / {sl['100건+']} | {f(sl['뒤>0|앞>0'])} / {f(sl['뒤>0|무조건'])} | "
               f"{f(sl['2026뒤>0|앞>0'])} / {f(sl['2026뒤>0|무조건'])} |")
    out.append(f"| d_adj(사후) · 2026 앞→뒤, 2025 | {sa['앞>0']} / {sa['100건+']} | {f(sa['뒤>0|앞>0'])} / {f(sa['뒤>0|무조건'])} | "
               f"{f(sa['2025>0|앞>0'])} / {f(sa['2025>0|무조건'])} |")
    out.append(f"| d_adj(사후) · 2015~19→2020~25, 2026 뒤 | {sla['앞>0']} / {sla['100건+']} | {f(sla['뒤>0|앞>0'])} / {f(sla['뒤>0|무조건'])} | "
               f"{f(sla['2026뒤>0|앞>0'])} / {f(sla['2026뒤>0|무조건'])} |")
    return out


def appendix(J, V):
    sel = J["selection"]
    out = ["### 2026 앞 짝 평균차 상위 10 (사전 기준, 100건+)\n",
           "| 순위 | 규칙 | 앞 차 (t, n) | 뒤 차 (n) | 2025 63일 차 | 앞 즉시 대비 | 뒤 즉시 대비 |", "|---|---|---|---|---|---|---|"]
    for i, p in enumerate(sel["top10"], 1):
        out.append(f"| {i} | {rule_ko(p['rule'], int(p['H']), p['leg'])} | {pp(p['앞_차'])} ({tv(p['앞_t'])}, {int(p['앞_n'])}) | "
                   f"{pp(p['뒤_차'])} ({int(p['뒤_n'])}) | {pp(p['y25_차'])} | {pp(p['앞_즉시차'])} | {pp(p['뒤_즉시차'])} |")
    out += ["", "### 2026 앞 d_adj 상위 10 (사후 참고, 100건+)\n",
            "| 순위 | 규칙 | 앞 d_adj (t, n) | 뒤 d_adj (t, n) | 2025 63일 d_adj (t) | 앞·뒤 사전 기준 차 |", "|---|---|---|---|---|---|"]
    for i, p in enumerate(sel["top10Adj"], 1):
        out.append(f"| {i} | {rule_ko(p['rule'], int(p['H']), p['leg'])} | {pp(p['앞_adj'])} ({tv(p['앞_t'])}, {int(p['앞_n'])}) | "
                   f"{pp(p['뒤_adj'])} ({tv(p['뒤_t'])}, {int(p['뒤_n'])}) | {pp(p['y25_adj'])} ({tv(p['y25_t'])}) | {pp(p['앞_차'])} · {pp(p['뒤_차'])} |")
    out += ["", "### 2015~19 짝 평균차 상위 10 (긴 역사 대체, 63일)\n",
            "| 순위 | 규칙 | 2015~19 차 (t, n) | 2020~25 차 (t, n) | 2026 뒤 차 (n) |", "|---|---|---|---|---|"]
    for i, p in enumerate(sel["top10L"], 1):
        out.append(f"| {i} | {rule_ko(p['rule'], 63, p['leg'])} | {pp(p['앞_차'])} ({tv(p['앞_t'])}, {int(p['앞_n'])}) | "
                   f"{pp(p['뒤_차'])} ({tv(p['뒤_t'])}, {int(p['뒤_n'])}) | {pp(p['y26뒤_차'])} ({int(p['y26뒤_n'])}) |")
    out += ["", "### 조건·청산 다리 효과 (2015~2025 63일, d_adj 차이 %p; 마지막 열은 2026 뒤 21일)\n",
            "| 사건 | C1 − C0 (N21 · N42) | C2 − C0 (N21 · N42) | 고정 −25% − 보유만 (N0 · N21) | 2026 뒤 21일: 고정 −25% − 보유만 (N0) |",
            "|---|---|---|---|---|"]
    for d in DEFS:
        c = lambda n, cc, w="2015~2025", H=63, leg="hold", k="d_adj": V.g(f"{d}|N{n}|{cc}", w, H, leg, k)
        out.append(f"| {DEF_KO[d]} | {pp(c(21, 'C1') - c(21, 'C0'))} · {pp(c(42, 'C1') - c(42, 'C0'))} | "
                   f"{pp(c(21, 'C2') - c(21, 'C0'))} · {pp(c(42, 'C2') - c(42, 'C0'))} | "
                   f"{pp(V.g(f'{d}|N0|C0', '2015~2025', 63, 'fix25', 'd_adj') - c(0, 'C0'))} · "
                   f"{pp(V.g(f'{d}|N21|C0', '2015~2025', 63, 'fix25', 'd_adj') - c(21, 'C0'))} | "
                   f"{pp(V.g(f'{d}|N0|C0', '2026뒤', 21, 'fix25', 'd_adj') - V.g(f'{d}|N0|C0', '2026뒤', 21, 'hold', 'd_adj'))} |")
    return out


def limits(J, V):
    meta = J["meta"]
    c4max = max(v["c4"]["cagr"] - v["c0"]["cagr"] for k, v in J["calendar"].items() if "c4" in v)
    return [
        "- **생존편향(가장 큼)**: 종목표는 2026-10 현재 생존·시총 10억 달러 이상이다. 폭락 뒤 상장폐지·파산·소형화한 종목이 없어 "
        "'폭락 뒤 매수'가 좋게 나온다. 사건 전 60일 거래대금 상위 1/3(대형)만 보면 사후 기준 차가 대부분 1~6%p 줄었다(시장 동반·고유 표의 '대형' 열, json `sub=big`).",
        "- **사전 기준의 구조 편향**: '같은 종목·같은 창 아무 날'은 그 창 안의 폭락 자체를 품은 매수까지 평균하므로 폭락 뒤에 산 거래가 "
        "기계적으로 유리하다. 그래서 거의 모든 규칙이 + 였고(고르기 표 '무조건' 비율) 사전 고르기는 정보를 주지 못했다. "
        "결과를 본 뒤 d_adj(같은 날 전 종목 평균과 그 종목 평소 초과를 뺌)를 더했다. d_adj도 생존편향은 못 뺀다.",
        "- **기다린 값은 C0만 공정**: C1(재발 시 다시 셈)·C2(사건일 종가 이하일 때만)는 기다리는 동안의 가격 경로로 사건을 골라서 "
        "'다음 날 바로 산 거래'와의 짝 비교가 C1·C2 쪽으로 기운다. 조건 효과는 같은 N의 C0와 비교했다(부록).",
        f"- **탐색 범위**: 사건 18 × 대기·조건 19 = {meta['rules']}규칙, 2026 후보 {meta['n_cand_2026']}개, 긴 역사 대체 후보 {meta['n_cand_long']}개, "
        f"계산 칸 {meta['cells']:,}개. 대표 후보는 이 가운데 최댓값이라 앞 구간 숫자는 부풀어 있다.",
        "- **2026 표본**: 2026 앞 63일 칸은 청산이 5/29 안이어야 해서 진입이 1/2~3/3뿐이다. 시장 동반 고점 −30~−50%(S4m)는 2026 앞 21일 8~15건, "
        "63일 0건, 하루 −15% 시장 동반(S3m_15)은 14건이라 2026 판정에 쓸 수 없다.",
        "- **시장 동반·고유 문턱**(QQQ −2%/−0.5%, 고점일 이후 QQQ −10%/−2%)은 사전에 하나로 정했다. S4(고점 낙폭의 시장 동반·고유)는 과제 격자에 더한 것이다.",
        "- **긴 역사**: 종목 자료가 2014년부터라 1971~2005 고르기·2008·2000~02·1987은 잴 수 없다. 대체로 2015~19(청산도 2019 안)로 고르고 2020~25로 확인했다.",
        "- **달력 포트폴리오**는 그날 열린 거래를 같은 돈으로 매일 재조정한 근사다. 사건이 잦으면 거의 늘 꽉 차 있어(노출 83~100%) 연율은 생존 종목표의 상승을 주로 반영한다. "
        f"현금 4% 판은 노출이 낮은 규칙(시장 동반 고점 −40·−50%, 노출 37~42%)에서 연율이 최대 {pct(c4max)}p 올랐고 나머지는 연 1%p 안쪽이었다(json `calendar.c4`).",
        "- 같은 달 거래는 서로 엮여 있어 t는 진입 월 묶음으로 냈다. 같은 종목 같은 사건은 20거래일 이상 띄웠지만 63·126일 보유는 서로 겹친다.",
        "- 수수료 0, 미끄러짐 0.5%(진입·손절)는 차를 0.5%p 안쪽으로만 바꿨다. 가격은 yfinance 배당 수정 일봉(TradingView 미사용).",
    ]


# ============================================================================ 결론(모두 json 수치에서)
def build_text(J, V):
    sel = J["selection"]
    g1, gl, sa = sel["G1"], sel["GL"], sel["succAdj"]
    s1 = sel["succ"]
    gi = lambda r, w, H, k, leg="hold", sub="all": V.g(r, w, H, leg, k, sub=sub)
    # 기다린 값(C0, 다음 날 바로 대비)
    w21 = {w: [gi(f"{d}|N21|C0", w, H, "d_imm") for d in DEFS] for w, H in (("2015~2025", 63), ("2026뒤", 21), ("2025", 63), ("2026앞", 21))}
    w42 = [gi(f"{d}|N42|C0", "2015~2025", 63, "d_imm") for d in DEFS]
    neg = lambda xs: sum(1 for x in xs if np.isfinite(x) and x < 0)
    rng = lambda xs: (np.nanmin(xs), np.nanmax(xs))
    short_pool = [gi(f"{d}|N{n}|C0", "2015~2025", 63, "d_imm") for d in ("S1_10", "S1_15", "S3m_10", "S3m_15") for n in (1, 3, 5)]
    short_26b = [gi(f"{d}|N{n}|C0", "2026뒤", 21, "d_imm") for d in ("S1_10", "S1_15", "S3m_10", "S3m_15") for n in (1, 3, 5)]
    c1 = [gi(f"{d}|N{n}|C1", "2015~2025", 63, "d_adj") - gi(f"{d}|N{n}|C0", "2015~2025", 63, "d_adj") for d in DEFS for n in (21, 42)]
    c2 = {w: [gi(f"{d}|N{n}|C2", w, H, "d_adj") - gi(f"{d}|N{n}|C0", w, H, "d_adj") for d in DEFS for n in (5, 10, 21, 42)]
          for w, H in (("2015~2025", 63), ("2026앞", 21), ("2026뒤", 21), ("2025", 63))}
    pos = lambda xs: np.mean([x > 0 for x in xs if np.isfinite(x)])
    s3i_26a = [gi(f"S3i_{x}|N{n}|C0", "2026앞", 21, "d_imm") for x in DROP1 for n in (21, 42)]
    s3i_26b = [gi(f"S3i_{x}|N{n}|C0", "2026뒤", 21, "d_imm") for x in DROP1 for n in (21, 42)]
    # 공정 기준
    adj = lambda d, w="2015~2025", H=63, sub="all", n=0: gi(f"{d}|N{n}|C0", w, H, "d_adj", sub=sub)
    g1_adj_back = gi(g1["rule"], "2026뒤", int(g1["H"]), "d_adj", g1["leg"])
    g1_adj_front = gi(g1["rule"], "2026앞", int(g1["H"]), "d_adj", g1["leg"])
    s3m_26a = [adj(f"S3m_{x}", "2026앞", 21) for x in (7, 10, 15)]
    s3m_26b = [adj(f"S3m_{x}", "2026뒤", 21) for x in (7, 10, 15)]
    big = lambda d, w="2015~2025", H=63: adj(d, w, H, "big")
    concl = [
        f"1. **한두 달 기다렸다 사는 마삼룰식 매수는 개별 종목에서 손해였다.** 같은 폭락을 다음 날 바로 산 것보다 21거래일 뒤 매수가 "
        f"2015~25(63일 보유) 18종 중 {neg(w21['2015~2025'])}종, 2026 뒤(21일) {neg(w21['2026뒤'])}종, 2025 {neg(w21['2025'])}종에서 낮았다"
        f"(2026 뒤 {pp(rng(w21['2026뒤'])[0])}~{pp(rng(w21['2026뒤'])[1])}%p). '또 폭락하면 다시 센다'(C1)도 무조건 기다림보다 "
        f"{pp(np.nanmin(c1))}~{pp(np.nanmax(c1))}%p로 나아지지 않았고, 1~5일 기다림은 2015~25엔 {pp(np.nanmin(short_pool))}~{pp(np.nanmax(short_pool))}%p였지만 "
        f"2026 뒤엔 {pp(np.nanmin(short_26b))}~{pp(np.nanmax(short_26b))}%p였다.",
        f"2. **'폭락 뒤 사면 아무 날보다 낫다'는 숫자는 대부분 비교 기준과 생존편향 착시였다.** 사전 기준(같은 종목·같은 창 아무 날)은 폭락 자체를 품어 "
        f"후보의 {pct(s1['뒤>0|무조건'][0], 0, False)}가 확인 구간에서도 +였다(고르기 정보 없음). 같은 날 전 종목과 그 종목 평소 초과를 뺀 기준(d_adj)으로 "
        f"2015~25 하루 −10% 바로 매수는 {pp(adj('S1_10'))}%p, 고점 −40%는 {pp(adj('S2_40'))}%p였고 거래대금 상위 1/3 종목만 보면 "
        f"{pp(big('S1_10'))}·{pp(big('S2_40'))}%p로 사라졌다. 사전 대표 후보 G1({rule_ko(g1['rule'], int(g1['H']), g1['leg'])})은 "
        f"사전 기준 뒤 {pp(g1['뒤_차'])}%p로 '통과'했지만 d_adj로는 뒤 {pp(g1_adj_back)}%p({int(g1['뒤_n'])}건)였다.",
        f"3. **남는 것은 '시장이 같이 무너진 날 바로 사기'뿐인데 2026엔 뒤집혔다.** 대형주만으로도 시장 동반 하루 −10·−15%는 2015~25 "
        f"{pp(big('S3m_10'))}·{pp(big('S3m_15'))}%p, 2025 {pp(big('S3m_10', '2025'))}·{pp(big('S3m_15', '2025'))}%p였지만, 2026 앞 "
        f"{pp(min(s3m_26a))}~{pp(max(s3m_26a))}%p → 뒤(7월 반도체 쇼크) {pp(min(s3m_26b))}~{pp(max(s3m_26b))}%p였고, 2026 앞에서 + 인 규칙이 뒤에서도 + 인 비율은 "
        f"{pct(sa['뒤>0|앞>0'][0], 0, False)}(무조건 {pct(sa['뒤>0|무조건'][0], 0, False)})였다. 기다린다면 'N일 뒤에도 사건일 종가 이하일 때만'(C2)이 "
        f"같은 N 무조건보다 대체로 나았다(2015~25 {pct(pos(c2['2015~2025']), 0, False)} · 2026 뒤 {pct(pos(c2['2026뒤']), 0, False)}의 칸, 2026 앞 {pct(pos(c2['2026앞']), 0, False)}).",
    ]
    surv = (f"사건 전 60일 거래대금 상위 1/3(대형)만 남기면 2015~25 63일 d_adj가 하루 −7·−10% {pp(adj('S1_7'))}·{pp(adj('S1_10'))} → "
            f"{pp(adj('S1_7', sub='big'))}·{pp(adj('S1_10', sub='big'))}%p, 고유 하루 −10% {pp(adj('S3i_10'))} → {pp(adj('S3i_10', sub='big'))}%p, "
            f"고유 고점 −40% {pp(adj('S4i_40'))} → {pp(adj('S4i_40', sub='big'))}%p로 줄었다. 시장 동반 하루 −10·−15%만 "
            f"{pp(adj('S3m_10'))}·{pp(adj('S3m_15'))} → {pp(adj('S3m_10', sub='big'))}·{pp(adj('S3m_15', sub='big'))}%p로 남았다. 상폐 종목이 없어 이것도 상한이다.")
    ut = ["칸 = 거래 1건 평균 수익 (괄호: 사전 기준 '같은 종목 아무 날' 대비 %p / 사후 공정 기준 d_adj %p). 모든 칸 63거래일 보유, 2026 앞은 진입 1/2~3/3(청산 5/29 안), "
          "2026 뒤는 진입 6/1~7/9. 긴 역사 열 = 2015~2025 진입 거래를 같은 돈으로 나눠 든 달력 포트폴리오(연율 / 최대 낙폭 / 종목표 전체 연 매매 수).\n"]
    ut += user_table(J, V)
    ut.append("")
    ut.append("- 2008·2000~02·1987: 종목 자료가 2014년부터라 잴 수 없다. 달력 포트폴리오는 거의 늘 수백 종목을 들고 있어(노출 83~100%) 생존 종목표의 상승을 주로 반영한다"
              f"(같은 기간 종목표 동일가중 보유 연 {pct(J['calendar']['EW']['c0']['cagr'])}).")
    grid = ["### 기다린 값 (핵심: 같은 사건을 다음 날 바로 산 것 대비)\n"] + wait_table(J, V) + [""]
    grid += ["### 사후 공정 기준 d_adj 격자\n"] + adj_table(J, V) + [""]
    grid += ["### 사전 기준(같은 종목·같은 창 아무 날 대비) 격자 — 폭락 자체를 품은 기준이라 거의 모두 + (고르기 정보 없음)\n"] + grid_tables(J, V)
    return {"conclusion": concl, "surv_line": surv, "user_table": ut, "grid": grid, "mkt": mkt_table(J, V),
            "select": select_table(J, V), "limits": limits(J, V), "appendix": appendix(J, V)}


# ============================================================================ md
def pct(x, d=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}"


def tv(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.1f}"


class View:
    def __init__(self, J):
        df = pd.DataFrame(J["cells"])
        self.A = df.set_index(["rule", "w", "H", "leg", "slip", "sub"]).sort_index()
        self.J = J

    def g(self, rule, w, H, leg, key, slip=0.0, sub="all"):
        try:
            v = self.A.loc[(rule, w, H, leg, slip, sub), key]
            v = float(v)
            return v if np.isfinite(v) else np.nan
        except (KeyError, TypeError, ValueError):
            return np.nan

    def pooled(self, w, H, leg, key, slip=0.0):
        v = self.J["pooled"].get(f"{w}|{H}|{leg}|{slip}")
        return float(v[key]) if v and v.get(key) is not None else np.nan


def rule_ko(rule, H=None, leg=None):
    d, n, c = rule.split("|")
    N = int(n[1:])
    wait = "다음 날 바로" if N == 0 else f"{N}거래일 뒤" + ("" if c == "C0" else f"({COND_KO[c]})")
    s = f"{DEF_KO[d]} → {wait}"
    if H is not None:
        s += f" · {H}일 보유"
    if leg is not None and leg != "hold":
        s += f" · {LEG_KO[leg]}"
    return s


def write_md(J: dict):
    V = View(J)
    sel, cal, meta = J["selection"], J["calendar"], J["meta"]
    TX = build_text(J, V)
    L = []
    L.append("# 7단계 STKBUY: 종목이 폭락한 뒤 N일 기다렸다 기계적으로 사기 (2026-10-11)\n")
    L.append("> 사용자: \"마삼룰이라고 있거든? 이거랑 컨셉이 비슷한데, 폭락 이후에는 어느 정도 기간 뒤에 기계적으로 매수하는 방법을 연구해볼래요. 다양하게 조합해서.\"")
    L.append(f"> 대상: 데스크 급락 탭 종목표(현재 생존, 시총 10억 달러 이상) {meta['n_all']}종목 − 자료 오류 의심 {meta['n_bad']} = {meta['n_used']}종목, "
             "2014~2026 일봉. 코드 `code/p7_STKBUY.py`, 수치 `results/p7_STKBUY.json`.\n")
    L.append("## 결론 3줄\n")
    L.extend(TX["conclusion"])
    L.append("")
    L.append("> **생존편향 경고(크게).** 종목표는 2026-10 지금 살아 있고 시총 10억 달러 이상인 종목뿐이다. 폭락한 뒤 상장폐지·파산·소형주로 "
             "떨어진 종목은 빠져 있고, 폭락 뒤 회복한 종목만 남아 있다. 그래서 '폭락 뒤 매수' 숫자는 실제보다 좋게 나온다(특히 고점 −50%, 하루 −15%). "
             + TX["surv_line"] + "\n")
    # ---------------------------------------------------------------- 사용자 표
    L.append("## 사용자 표: 규칙 → 구간별 거래 1건 평균 수익 (괄호: 사전 기준 / 공정 기준 d_adj, %p)\n")
    L.extend(TX["user_table"])
    L.append("")
    # ---------------------------------------------------------------- 조합 격자
    L.append("## 조합 격자 (행 = 사건, 열 = 기다린 거래일 N, 무조건 C0 · 보유만)\n")
    L.extend(TX["grid"])
    L.append("")
    # ---------------------------------------------------------------- 시장 동반 vs 고유
    L.append("## 시장 동반 폭락 vs 종목 고유 폭락\n")
    L.extend(TX["mkt"])
    L.append("")
    # ---------------------------------------------------------------- 고르기
    L.append("## 고르기 표\n")
    L.extend(TX["select"])
    L.append("")
    # ---------------------------------------------------------------- 시험 범위
    L.append("## 시험 범위(사전 고정, 확인 구간을 본 뒤 바꾸지 않음)\n")
    L.append("| 항목 | 값 |\n|---|---|")
    L.append("| 사건 18 | 하루 −7·−10·−15%(S1) · 252일 고점 −30·−40·−50% 첫날(S2) · S1을 같은 날 QQQ ≤ −2%(시장 동반)/> −0.5%(고유)로 나눔(S3 6) · "
             "S2를 고점일→사건일 QQQ ≤ −10%(시장 동반)/> −2%(고유)로 나눔(S4 6, 과제 격자에 더함). 같은 종목 같은 사건은 20거래일 이상 띄움 |")
    L.append("| 대기 N 7 × 조건 3 | N = 0·1·3·5·10·21·42거래일. C0 무조건 · C1 기다리는 중 같은 사건이 또 나오면 그날부터 다시 셈(마삼룰식) · "
             "C2 N일째 종가가 사건일 종가 이하일 때만. N0은 하나로 셈 → 19조합 |")
    L.append("| 보유 H · 다리 | 21·63·126거래일 × 보유만 / 고정 −25% |")
    L.append(f"| 규칙 수 | 사건 18 × 대기·조건 19 = {meta['rules']}개. 2026 고르기 후보 {meta['n_cand_2026']}개(× H 21·63 × 다리 2), "
             f"긴 역사 대체 후보 {meta['n_cand_long']}개(H63 × 다리 2). 계산 칸 {meta['cells']:,}개 |")
    L.append("| 창(진입일 기준) | 2026앞 1/2~5/29(청산도 5/29 안) · 2026뒤 6/1~10/9 · 2025 · 2015~2024 각 해 · 2020급락(2/19~3/23) · 2015~2025 · "
             "2015~2019(청산도 2019 안) · 2020~2025 |")
    L.append("| 시점 | 사건은 그날 종가로 판단 → N거래일 뒤 다음 날 시가에 산다. C2의 가격 비교도 산 날 전날 종가까지. 고정 −25%는 진입가 기준 장중(시가가 선 아래면 시가) |")
    L.append("| 비용 | 수수료 0, 미끄러짐 0.5% 민감도(진입·손절). 달력 포트폴리오 현금 이자 0%(4% 한 줄) |\n")
    L.append("## 변수 시점\n")
    L.append("| 변수 | 언제 아나 | 매매 근거로 쓰나 |\n|---|---|---|")
    L.append("| 하루 하락률·252일 고점 대비 낙폭·같은 날 QQQ | 사건일 종가 | 예 |")
    L.append("| 고점일→사건일 QQQ(S4 시장 동반·고유) | 사건일 종가(과거 252일만 봄) | 예 |")
    L.append("| C1 재발·C2 사건일 종가 이하 | 산 날 전날 종가까지 | 예 |")
    L.append("| 대형(사건 전 60일 거래대금 상위 1/3) | 사건일 종가 | 예(점검용 부분집합) |")
    L.append("| d_adj의 '그 종목 평소 초과'(2015~2026 전체 평균) | **사후** | 아니오 — 비교 기준(설명용)일 뿐 |")
    L.append("| 종목표(현재 생존·시총 10억 달러 이상) | **2026-10 현재(사후)** | 아니오 — 생존편향 원인 |\n")
    # ---------------------------------------------------------------- 한계
    L.append("## 한계\n")
    L.extend(TX["limits"])
    L.append("")
    # ---------------------------------------------------------------- 부록
    L.append("## 부록\n")
    L.extend(TX["appendix"])
    ch = meta["check"]
    L.append(f"\n점검: 미리 계산한 거래 수익 vs trade_lab.trade 최대 오차 {ch['ret_err']:.1e}({ch['n_ret']}건, float32 반올림), "
             f"N0 진입 vs trade_lab.entries 불일치 {ch['entries_mismatch']}, C1 하루씩 독립 구현 불일치 {ch['c1_mismatch']}, "
             f"미래 교란 뒤 진입 불일치 {ch['future_mismatch']}. 계산 {meta['secs']:.0f}초.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("md →", OUT_MD)


# ============================================================================ 본문
def main():
    if "--md" in sys.argv:
        J = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        write_md(J)
        return
    t0 = time.time()
    data, allsyms, bad, syms = load_universe()
    print("종목", len(allsyms), "· 자료 오류 의심 제외", len(bad), "· 사용", len(syms), flush=True)
    pre = Pre(data, syms)
    print(f"미리 계산 {time.time() - t0:.0f}s", flush=True)
    chk = check(data, syms, pre)
    if "--check" in sys.argv:
        return
    G, gidx = grid(pre)
    big = big_flags(pre, G, gidx)
    qq = load_qqq()
    trades = build_trades(pre, G, gidx, qq, big)
    print(f"진입 목록 {time.time() - t0:.0f}s", {r: len(trades[r]['e']) for r in RULES[:19:6]}, flush=True)
    base, pooled = any_day(pre)
    xsd = same_day(pre, G, gidx)
    alpha = stock_alpha(pre, xsd)
    print(f"기준 {time.time() - t0:.0f}s", flush=True)
    cells = compute(pre, G, gidx, trades, base, xsd, alpha)
    print(f"칸 {len(cells)} {time.time() - t0:.0f}s", flush=True)
    sel = selection(cells)
    # 달력: 사용자 표 규칙 + 대표 후보
    cal_rules = USER_RULES + [(sel["G1"]["rule"], int(sel["G1"]["H"]), sel["G1"]["leg"]),
                              (sel["GL"]["rule"], 63, sel["GL"]["leg"]),
                              (sel["G1adj"]["rule"], int(sel["G1adj"]["H"]), sel["G1adj"]["leg"]),
                              (sel["GLadj"]["rule"], 63, sel["GLadj"]["leg"])]
    cal = {}
    for r, H, leg in cal_rules:
        k = f"{r}#{H}#{leg}"
        if k in cal:
            continue
        cal[k] = {"c0": calendar(pre, G, gidx, trades[r], H, leg, 0.0), "c4": calendar(pre, G, gidx, trades[r], H, leg, 0.04)}
    cal["EW"] = {"c0": calendar_ew(pre, G, gidx, 0.0)}
    print(f"달력 {time.time() - t0:.0f}s", flush=True)
    J = {"meta": {"n_all": len(allsyms), "n_bad": len(bad), "bad": bad, "n_used": len(syms), "check": chk,
                  "secs": time.time() - t0, "defs": DEFS, "def_ko": DEF_KO, "nc": NC, "rules": len(RULES),
                  "n_cand_2026": N_CAND_2026, "n_cand_long": N_CAND_LONG, "cells": int(len(cells)),
                  "windows": WINDOWS, "trades_total": {r: int(len(trades[r]["e"])) for r in RULES}},
         "pooled": {f"{w}|{H}|{leg}|{slip}": v for (w, H, leg, slip), v in pooled.items()},
         "cells": {c: (cells[c].round(6).tolist() if cells[c].dtype.kind == "f" else cells[c].tolist()) for c in cells.columns},
         "selection": sel, "calendar": cal}
    OUT_JSON.write_text(json.dumps(J, ensure_ascii=False, default=float), encoding="utf-8")
    print("json →", OUT_JSON)
    write_md(J)


# 사용자 표 규칙(사전 고정): (규칙, H, 다리)
USER_RULES = [
    ("S1_10|N0|C0", 63, "hold"), ("S1_10|N21|C0", 63, "hold"), ("S1_10|N21|C1", 63, "hold"),
    ("S3m_10|N21|C0", 63, "hold"), ("S3i_10|N21|C0", 63, "hold"),
    ("S2_40|N0|C0", 63, "hold"), ("S2_40|N21|C0", 63, "hold"),
    ("S4m_40|N21|C0", 63, "hold"), ("S4i_40|N21|C0", 63, "hold"),
    ("S1_10|N21|C2", 63, "hold"),     # 결과를 본 뒤 더한 행(C2 효과 예시)
]

if __name__ == "__main__":
    main()
