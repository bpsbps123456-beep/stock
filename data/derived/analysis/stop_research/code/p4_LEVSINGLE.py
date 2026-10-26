"""4단계 LEVSINGLE(2026-10-11): 단일 종목 2배 레버리지 상품에 추세·손절·변동성 규칙을 얹으면 '전멸'을 막고 효율이 나아지나.

사용자: "일반적인 주식도 한단 말야. 거기에 쓸 만한 게 없어? 특히 레버리지 주식도 함!"
대상: 실제 2배 상품이 있는 기초 종목 전부(lev_lab 상품표에서 L=2이고 기초가 개별 종목인 것). 2014년부터 일일 재설정 2배를 합성한다.
  - 기본 합성: lev_lab.synth(u, 2) = 2×기초 − 1×단기금리 − 보수 1.05%(연).
  - 보정 합성: 위 + 연 추가 비용(실제 상품의 스왑 금융비용 등). 추가 비용 = 실제 상품과 겹치는 최근 252거래일의
    연 로그 추적차(−ln(실제 배율/합성 배율)/연수), 상품이 여럿이면 중앙값, 0 미만은 0, 겹침 60일 미만이면 연 10%.
    (lev_lab.validate의 '연추적차'는 CAGR 차라서 고성장 구간에서 비용을 부풀린다: NVDL CAGR 차 −47%p ≈ 로그 비용 연 18%.)
규칙 묶음(T·S·V·P·D + 조합 C). 모두 '창 시작 직전 종가에 2배 상품을 다 들고 시작'하고, 판 돈은 현금(이자 0)으로 둔다(D만 기초 1배).
  T 추세   : 신호원 종가 < SMA_n 이면 다음 시가에 전량 매도, 종가 ≥ SMA_n 이면 다음 시가에 전량 재매수. 신호원 = 기초 종목 자기 | QQQ.
  S 추적손절: 상품 '산 뒤 최고 종가'(t−1까지) × (1−X) 장중 손절. 재매수 = 기초 종가가 20일 최고 종가를 경신한 다음 시가 | 판 날+20거래일 시가.
  V 변동성 : 기초 20일 실현 변동성 > 그 종목 252일 중앙(최소 126일) × m 이면 다음 시가에 매도, 아래로 내려오면 다음 시가에 재매수.
  P 일부   : 기본 규칙을 상품의 50%에만 적용, 나머지 50%는 끝까지 보유(조각별 따로 계산 = 0.5×규칙 곡선 + 0.5×보유 곡선, 3단계와 같은 정의).
  D 배수 낮추기: T·V 신호 때 현금 대신 기초 종목 1배로 갈아탄다(다음 시가에 상품 매도·기초 매수), 신호가 꺼지면 다시 2배로.
  C 조합   : TS = 자기 SMA_n 이탈 또는 추적 X 손절이면 매도, 재매수는 추세 위 + (손절로 나갔으면) 20일 신고가.
             TV = 자기 SMA_200 아래 또는 변동성 > 중앙 × m 이면 매도, 둘 다 풀리면 재매수.
시점: 지표는 t−1 종가까지, 종가 신호 → t 시가 체결. 장중 손절가는 t−1까지의 최고 종가로 정한다(stock_lab 규약).
평가: 레버리지 효율선초과 = 규칙 수익 − (같은 최대 낙폭이 나오는 고정 배수 k 보유 가운데 최고 수익), k = 0~2(0.05 간격).
  k ≤ 1: k × 기초(현금 0%). k > 1: 2배 상품 (k−1) + 기초 (2−k)를 매일 맞춘 것 = k×기초 − (k−1)×금리 − (k−1)×(보수+추가 비용).
  같은 평균 노출 대비 = 규칙 수익 − (규칙의 평균 배수 k̄로 고정 보유한 수익).
창: 2026 앞(1/2~5/29, 고르기)·뒤(6/1~10/9, 확인), 2025(보조). 2015~2024 연도별·2018 4분기·2020 2~3월은 맥락. 2015~2026 이어 돌리기 CAGR·낙폭.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lev_lab as LL
import stock_lab as S
import trade_lab as TL

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# ============================================================================ 사전 고정(뒤·2025를 본 뒤 바꾸지 않는다)
L = 2
FEE = LL.FEE[2]                                   # 연 1.05%
SPAC_START = {"IONQ": "2021-10-01", "RGTI": "2022-03-02", "OKLO": "2024-05-10"}   # 합병 전 SPAC($10 껍데기) 구간 제외
MIN_HIST = 126                                    # 창 시작 전 기초 자료 최소 거래일(지표가 아직 없으면 신호 없음 = 보유)
TRACK_DAYS, TRACK_MIN, EXTRA_FALLBACK = 252, 60, 0.10
T_N = (20, 50, 100, 200)
T_SRC = ("own", "qqq")
S_X = (0.15, 0.25, 0.35, 0.50)
S_RE = ("hi20", "t20")
V_M = (1.5, 2.0)
VOL_N, VOL_MED_N, VOL_MED_MIN = 20, 252, 126
P_Q = 0.5
P_BASE = (("T", "own", 50), ("T", "own", 200), ("S", 0.25, "hi20"), ("S", 0.35, "hi20"))
D_BASE = (("T", "own", 50), ("T", "own", 100), ("T", "own", 200), ("T", "qqq", 50), ("T", "qqq", 200), ("V", 1.5), ("V", 2.0))
C_TS = tuple((n, x) for n in (50, 200) for x in (0.25, 0.35))
C_TV = tuple((200, m) for m in V_M)
K_GRID = np.round(np.arange(0, L + 1e-9, 0.05), 2)   # 41개
SLIP = 0.005
G2_MIN_N = 30
DEATH = -0.80

RUNS = {"2026": ("2026-01-02", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31")}
for _y in range(2024, 2014, -1):
    RUNS[str(_y)] = S.WINDOWS.get(str(_y), (f"{_y}-01-01", f"{_y}-12-31"))
WIN = {"2026앞": ("2026", "2026-01-02", "2026-05-29"), "2026뒤": ("2026", "2026-06-01", "2026-10-09"),
       "2026": ("2026", "2026-01-02", "2026-10-09")}
for _r, (_a, _b) in RUNS.items():
    if _r != "2026":
        WIN[_r] = (_r, _a, _b)
WIN["2018Q4"] = ("2018", "2018-10-01", "2018-12-31")
WIN["2020급락"] = ("2020", "2020-02-19", "2020-03-31")
FULL = ("2015-01-02", "2026-10-09")
SEL_W, CHK_W = "2026앞", ("2026뒤", "2025")


def rule_specs():
    """이름 → spec. 개수는 main에서 보고한다."""
    sp = {"보유(2배)": ("H",)}
    for src in T_SRC:
        for n in T_N:
            sp[f"T {'자기' if src == 'own' else 'QQQ'} SMA{n}"] = ("T", src, n)
    for x in S_X:
        for re_ in S_RE:
            sp[f"S 추적{int(x * 100)}% → {'20일 신고가' if re_ == 'hi20' else '20일 뒤'}"] = ("S", x, re_)
    for m in V_M:
        sp[f"V 변동성>중앙×{m}"] = ("V", m)
    for b in P_BASE:
        sp[f"P 50%만 [{_bname(b)}]"] = ("P", b)
    for b in D_BASE:
        sp[f"D 1배로 [{_bname(b)}]"] = ("D", b)
    for n, x in C_TS:
        sp[f"C 자기SMA{n} + 추적{int(x * 100)}%"] = ("TS", n, x)
    for n, m in C_TV:
        sp[f"C 자기SMA{n} + 변동성×{m}"] = ("TV", n, m)
    return sp


def _bname(b):
    if b[0] == "T":
        return f"{'자기' if b[1] == 'own' else 'QQQ'} SMA{b[2]}"
    if b[0] == "S":
        return f"추적{int(b[1] * 100)}%→신고가"
    if b[0] == "V":
        return f"변동성×{b[1]}"
    return str(b)


# ============================================================================ 자료
def sub(b: S.Bars, i0: int, sym=None) -> S.Bars:
    return S.Bars(sym or b.sym, b.days[i0:].copy(), b.o[i0:].copy(), b.h[i0:].copy(), b.l[i0:].copy(), b.c[i0:].copy(),
                  b.v[i0:].copy())


def extra_cost(u: str, d: dict, prods: dict) -> dict:
    rows = []
    s = LL.synth(u, L)
    for p, info in prods.items():
        if info["under"] != u or info["L"] != L or p not in d:
            continue
        a = d[p]
        com = np.intersect1d(a.days, s.days)
        if len(com) < TRACK_MIN:
            rows.append({"product": p, "겹침": int(len(com)), "drag": None})
            continue
        com = com[-(TRACK_DAYS + 1):]
        ia, is_ = np.searchsorted(a.days, com), np.searchsorted(s.days, com)
        yrs = (len(com) - 1) / 252
        drag = -(np.log(a.c[ia][-1] / a.c[ia][0]) - np.log(s.c[is_][-1] / s.c[is_][0])) / yrs
        full = np.intersect1d(a.days, s.days)
        fa, fs = np.searchsorted(a.days, full), np.searchsorted(s.days, full)
        fy = (len(full) - 1) / 252
        drag_full = -(np.log(a.c[fa][-1] / a.c[fa][0]) - np.log(s.c[fs][-1] / s.c[fs][0])) / fy
        cagr_diff = (a.c[fa][-1] / a.c[fa][0]) ** (1 / fy) - (s.c[fs][-1] / s.c[fs][0]) ** (1 / fy)
        rows.append({"product": p, "겹침": int(len(full)), "시작": str(full[0]), "drag": float(drag),
                     "drag_full": float(drag_full), "cagr_diff": float(cagr_diff)})
    vals = [r["drag"] for r in rows if r["drag"] is not None]
    extra = max(0.0, float(np.median(vals))) if vals else EXTRA_FALLBACK
    return {"extra": extra, "products": rows, "fallback": not vals}


def signals(ub: S.Bars, q: S.Bars) -> dict:
    """종가 시점 신호 배열(인덱스 i = i일 종가에 앎). 정책은 [t−1]을 읽는다."""
    c = ub.c
    qi = np.searchsorted(q.days, ub.days)
    assert np.all(qi < len(q.days)) and np.all(q.days[qi] == ub.days), ub.sym
    sig = {}
    for n in T_N:
        sma = ub.sma(n)
        sig[f"off_own{n}"] = np.isfinite(sma) & (c < sma)
        qs = q.sma(n)[qi]
        sig[f"off_qqq{n}"] = np.isfinite(qs) & (q.c[qi] < qs)
    rv = ub.rvol(VOL_N)
    med = pd.Series(rv).rolling(VOL_MED_N, min_periods=VOL_MED_MIN).median().to_numpy()
    for m in V_M:
        sig[f"off_vol{m}"] = np.isfinite(rv) & np.isfinite(med) & (rv > m * med)
    sig["hi20"] = c >= ub.close_high(20) - 1e-12
    return sig


def off_array(sig: dict, b) -> np.ndarray:
    if b[0] == "T":
        return sig[f"off_{b[1]}{b[2]}"]
    if b[0] == "V":
        return sig[f"off_vol{b[1]}"]
    raise ValueError(b)


def prepare():
    d = LL.load()
    prods = LL.products()
    q = d["QQQ"]
    unders = sorted({v["under"] for v in prods.values() if v["L"] == L} - {"QQQ", "SPY", "SOXX"})
    excluded = {u: "trade_lab.bad_bars(하루 +150% 또는 −70%)" for u in unders if TL.bad_bars(d[u])}
    rf_full = LL.rf_series()
    U = {}
    for u in unders:
        if u in excluded:
            continue
        ub_full = d[u]
        i0 = ub_full.idx(SPAC_START[u]) if u in SPAC_START else 0
        ec = extra_cost(u, d, prods)
        rf = rf_full.reindex(ub_full.days).ffill().fillna(0.0).to_numpy()
        ub = sub(ub_full, i0)
        U[u] = {"und": ub, "rf": rf[i0:], "extra": ec["extra"], "track": ec, "start": str(ub.days[0]),
                "syn": {"basic": sub(LL.synth(u, L), i0, f"{u}x2"),
                        "cal": sub(LL.synth(u, L, FEE + ec["extra"]), i0, f"{u}x2c")},
                "sig": signals(ub, q)}
        assert np.array_equal(U[u]["syn"]["cal"].days, ub.days)
    return U, excluded, prods


# ============================================================================ 정책·엔진
class Rule(S.Policy):
    def __init__(self, spec, sig):
        self.k, self.sig = spec, sig

    def _off(self, t):
        k = self.k
        if k[0] == "T":
            return bool(self.sig[f"off_{k[1]}{k[2]}"][t - 1])
        if k[0] == "V":
            return bool(self.sig[f"off_vol{k[1]}"][t - 1])
        if k[0] == "TS":
            return bool(self.sig[f"off_own{k[1]}"][t - 1])
        if k[0] == "TV":
            return bool(self.sig[f"off_own{k[1]}"][t - 1] or self.sig[f"off_vol{k[2]}"][t - 1])
        return False

    def on_open(self, b, t, st):
        k = self.k[0]
        held = st["pos"] > 1e-9
        if k in ("T", "V", "TV"):
            off = self._off(t)
            if held and off:
                return 0.0
            if not held and not off:
                return 1.0
            return None
        if k == "S":
            if held:
                return None
            if self.k[2] == "hi20":
                return 1.0 if (t - 1 >= st["exit_t"] and self.sig["hi20"][t - 1]) else None
            return 1.0 if t - st["exit_t"] >= 20 else None
        if k == "TS":
            off = self._off(t)
            if held:
                return 0.0 if off else None
            if off:
                return None
            if st["last_reason"] == "open":
                return 1.0
            return 1.0 if (t - 1 >= st["exit_t"] and self.sig["hi20"][t - 1]) else None
        return None

    def stop(self, b, t, st):
        if self.k[0] == "S":
            return st["peak"] * (1 - self.k[1])
        if self.k[0] == "TS":
            return st["peak"] * (1 - self.k[2])
        return None


def sim_switch(p: S.Bars, u: S.Bars, a: int, z: int, off: np.ndarray, slip: float = 0.0):
    """D: a−1 종가에 상품 전량. off[t−1]이면 t 시가에 상품→기초 1배, 꺼지면 t 시가에 기초→상품."""
    sh_p, sh_u, state = 1.0 / p.c[a - 1], 0.0, "P"
    vals, mult, cycles, cur = [1.0], [], [], None
    for t in range(a, z + 1):
        o = bool(off[t - 1])
        if state == "P" and o:
            sell, buy = p.o[t] * (1 - slip), u.o[t] * (1 + slip)
            sh_u, sh_p, state = sh_p * sell / buy, 0.0, "U"
            cur = {"exit_day": p.days[t], "prod_sell": sell, "und_buy": buy, "kind": "D"}
        elif state == "U" and not o:
            sell, buy = u.o[t] * (1 - slip), p.o[t] * (1 + slip)
            sh_p, sh_u, state = sh_u * sell / buy, 0.0, "P"
            cur.update(re_day=p.days[t], gain=(sell / cur["und_buy"]) * (cur["prod_sell"] / buy) - 1, open=False)
            cycles.append(cur)
            cur = None
        vals.append(sh_p * p.c[t] + sh_u * u.c[t])
        mult.append(float(L) if state == "P" else 1.0)
    if cur is not None:
        cur.update(re_day=None, gain=(u.c[z] / cur["und_buy"]) * (cur["prod_sell"] / p.c[z]) - 1, open=True)
        cycles.append(cur)
    return np.array(vals), np.array(mult), cycles


def run_rule(spec, p: S.Bars, u: S.Bars, sig: dict, a: int, z: int, slip: float = 0.0, _hold=None):
    """반환 (가치 곡선[z−a+2], 종가 배수[z−a+1], 주기 목록). 주기 gain = 판 가격/되산 가격 − 1(+면 판 것이 이득)."""
    k = spec[0]
    if k == "D":
        return sim_switch(p, u, a, z, off_array(sig, spec[1]), slip)
    if k == "P":
        vf, mf, cf = run_rule(spec[1], p, u, sig, a, z, slip)
        vh = _hold if _hold is not None else p.c[a - 1:z + 1] / p.c[a - 1]
        v = P_Q * vf + (1 - P_Q) * vh
        prod_val = P_Q * vf[1:] * (mf / L) + (1 - P_Q) * vh[1:]
        return v, L * prod_val / v[1:], [dict(cy, w=P_Q) for cy in cf]
    pol = S.Policy() if k == "H" else Rule(spec, sig)
    v, cyc, _ = S.simulate(p, a, z, pol, slip)
    m = L * np.asarray(S.simulate.last_pos, float)
    for cy in cyc:
        cy["kind"] = "S"
    return v, m, cyc


# ============================================================================ 평가
def _mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def kpath(ru, rfd, k, extra, rp=None):
    """고정 배수 k의 일수익. rp(실제 상품 일수익)가 있으면 k>1은 실제 상품 (k−1) + 기초 (2−k) 혼합."""
    if k <= 1:
        return k * ru
    if rp is not None:
        return (k - 1) * rp + (2 - k) * ru
    return np.maximum(k * ru - (k - 1) * rfd - (FEE + extra) * (k - 1) / (L - 1) / 252, -0.99)


def frontier(ru, rfd, extra, rp=None):
    out = []
    for k in K_GRID:
        cur = np.concatenate([[1.0], np.cumprod(1 + kpath(ru, rfd, k, extra, rp))])
        out.append((float(cur[-1] - 1), _mdd(cur)))
    return np.array(out)


def eff_excess(ret, mdd, fr):
    ok = fr[:, 1] >= mdd - 1e-12
    j = int(np.argmax(np.where(ok, fr[:, 0], -np.inf)))
    return float(ret - fr[j, 0]), float(K_GRID[j])


def slice_idx(b: S.Bars, a: int, z: int, w0: str, w1: str):
    i0, i1 = max(b.idx(w0), a), min(b.idx(w1, "right") - 1, z)
    return i0, i1


def cyc_in(cycles, p: S.Bars, u: S.Bars, w0, w1, i1):
    """창 [w0,w1]에 판 주기. 창 끝까지 못 되샀으면 창 끝 종가로 값을 매긴다(미래 정보 차단)."""
    g, wt = [], []
    for cy in cycles:
        if not (w0 <= cy["exit_day"] <= w1):
            continue
        closed = (not cy.get("open")) and cy.get("re_day") is not None and cy["re_day"] <= w1
        if closed:
            x = cy["gain"]
        elif cy["kind"] == "D":
            x = (u.c[i1] / cy["und_buy"]) * (cy["prod_sell"] / p.c[i1]) - 1
        else:
            x = cy["exit_px"] / p.c[i1] - 1
        g.append(x)
        wt.append(cy.get("w", 1.0))
    return g, wt


def evaluate(U, specs, variants=("cal", "basic"), runs=None, slip=0.0, wins=None, keep_curves=False, full=True):
    """제품×규칙×창 행(DataFrame)과 주기, (원하면) 창별 일수익(포트폴리오용)."""
    runs = runs or RUNS
    wins = wins or {w: v for w, v in WIN.items() if v[0] in runs}
    rows, cyc_rows, daily = [], [], {}
    t0 = time.time()
    for var in variants:
        for u, D in U.items():
            ub, p, sig = D["und"], D["syn"][var], D["sig"]
            ex = D["extra"] if var == "cal" else 0.0
            for rname, (d0, d1) in runs.items():
                a, z = ub.idx(d0), ub.idx(d1, "right") - 1
                if a < MIN_HIST or z <= a:
                    continue
                hold = p.c[a - 1:z + 1] / p.c[a - 1]
                res = {}
                for name, spec in specs.items():
                    res[name] = run_rule(spec, p, ub, sig, a, z, slip, _hold=hold if slip == 0 else None)
                for w, (rn, w0, w1) in wins.items():
                    if rn != rname:
                        continue
                    i0, i1 = slice_idx(ub, a, z, w0, w1)
                    if i1 <= i0:
                        continue
                    ru = ub.c[i0:i1 + 1] / ub.c[i0 - 1:i1] - 1
                    rfd = D["rf"][i0:i1 + 1] / 252
                    fr = frontier(ru, rfd, ex)
                    hv = p.c[i0 - 1:i1 + 1] / p.c[i0 - 1]
                    for name, (v, m, cyc) in res.items():
                        vv = v[i0 - a:i1 - a + 2]
                        vv = vv / vv[0]
                        mm = m[i0 - a:i1 - a + 1]
                        ret, mdd = float(vv[-1] - 1), _mdd(vv)
                        exc, k_eq = eff_excess(ret, mdd, fr)
                        kbar = float(mm.mean())
                        rk = np.concatenate([[1.0], np.cumprod(1 + kpath(ru, rfd, kbar, ex))])
                        g, wt = cyc_in(cyc, p, ub, w0, w1, i1)
                        rows.append({"var": var, "under": u, "rule": name, "win": w, "ret": ret, "mdd": mdd,
                                     "hold_ret": float(hv[-1] - 1), "hold_mdd": _mdd(hv), "exc": exc, "k_eq": k_eq,
                                     "kbar": kbar, "exp_exc": float(ret - (rk[-1] - 1)), "n_cyc": len(g),
                                     "days": int(i1 - i0 + 1)})
                        for x, ww in zip(g, wt):
                            cyc_rows.append({"var": var, "under": u, "rule": name, "win": w, "gain": x, "w": ww})
                        if keep_curves:
                            dd = ub.days[i0:i1 + 1]
                            daily.setdefault((var, name, w), {})[u] = pd.Series(vv[1:] / vv[:-1] - 1, index=dd)
                            daily.setdefault((var, "__k1", w), {})[u] = pd.Series(ru, index=dd)
            if full:
                a = max(ub.idx(FULL[0]), MIN_HIST)
                z = ub.idx(FULL[1], "right") - 1
                hold = p.c[a - 1:z + 1] / p.c[a - 1]
                yrs = (z - a + 1) / 252
                for name, spec in specs.items():
                    v, m, cyc = run_rule(spec, p, ub, sig, a, z, slip, _hold=hold if slip == 0 else None)
                    rows.append({"var": var, "under": u, "rule": name, "win": "전체", "ret": float(v[-1] - 1),
                                 "mdd": _mdd(v), "hold_ret": float(hold[-1] - 1), "hold_mdd": _mdd(hold),
                                 "cagr": float(max(v[-1], 1e-9) ** (1 / yrs) - 1),
                                 "hold_cagr": float(max(hold[-1], 1e-9) ** (1 / yrs) - 1), "start": str(ub.days[a]),
                                 "kbar": float(m.mean()), "n_cyc": len(cyc), "days": int(z - a + 1)})
        print(f"  {var} 끝 {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(cyc_rows), daily


def port_stats(daily, var, rule, w):
    R = pd.DataFrame(daily.get((var, rule, w), {})).sort_index()
    if not len(R):
        return {}
    pr = np.concatenate([[1.0], (1 + R.mean(axis=1).fillna(0)).cumprod().to_numpy()])
    return {"포트_수익": float(pr[-1] - 1), "포트_낙폭": _mdd(pr), "칼마": float((pr[-1] - 1) / max(-_mdd(pr), 0.01))}


def agg(rows: pd.DataFrame, cyc: pd.DataFrame, daily: dict) -> pd.DataFrame:
    out = []
    for (var, rule, w), g in rows[rows.win != "전체"].groupby(["var", "rule", "win"]):
        c = cyc[(cyc["var"] == var) & (cyc["rule"] == rule) & (cyc["win"] == w)] if len(cyc) else cyc
        x = c["gain"].to_numpy() if len(c) else np.array([])
        win, loss = x[x > 0], x[x <= 0]
        r = {"var": var, "rule": rule, "win": w, "상품수": int(len(g)),
             "효율선초과_평균": float(g.exc.mean()), "효율선초과_중앙": float(g.exc.median()),
             "효율선초과_양수비율": float((g.exc > 1e-12).mean()), "노출대비_평균": float(g.exp_exc.mean()),
             "수익_평균": float(g.ret.mean()), "수익_중앙": float(g.ret.median()), "낙폭_평균": float(g.mdd.mean()),
             "보유수익_평균": float(g.hold_ret.mean()), "보유수익_중앙": float(g.hold_ret.median()),
             "보유낙폭_평균": float(g.hold_mdd.mean()), "평균배수": float(g.kbar.mean()),
             "최악수익": float(g.ret.min()), "보유최악수익": float(g.hold_ret.min()),
             "전멸수": int((g.mdd <= DEATH).sum()), "보유전멸수": int((g.hold_mdd <= DEATH).sum()),
             "주기n": int(len(x)), "승률": float(len(win) / len(x)) if len(x) else np.nan,
             "평균이익": float(win.mean()) if len(win) else np.nan, "평균손실": float(loss.mean()) if len(loss) else np.nan,
             "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
             "기대값": float(x.mean()) if len(x) else np.nan,
             "주기하위5%": float(np.quantile(x, 0.05)) if len(x) else np.nan}
        r.update(port_stats(daily, var, rule, w))
        out.append(r)
    return pd.DataFrame(out)


# ============================================================================ 실제 상품 대조(2026)
def actual_check(U, prods, specs, d):
    """2026: 기초마다 2025-12-31 이전부터 있는 실제 상품 중 가장 오래된 것. 효율선 k>1은 실제 상품(k−1)+기초(2−k)."""
    rows = []
    for u, D in U.items():
        cands = [(d[p].days[0], p) for p, i in prods.items() if i["under"] == u and i["L"] == L and p in d
                 and d[p].days[0] < "2025-12-31"]
        if not cands:
            continue
        p = d[sorted(cands)[0][1]]
        ub = D["und"]
        ui = np.searchsorted(ub.days, p.days)
        ok = (ui < len(ub.days))
        ok[ok] = ub.days[ui[ok]] == p.days[ok]
        if not ok.all():
            keep = np.flatnonzero(ok)
            p = S.Bars(p.sym, p.days[keep], p.o[keep], p.h[keep], p.l[keep], p.c[keep], p.v[keep])
            ui = ui[keep]
        # 기초를 상품 날짜에 맞춘다(신호도 같은 날짜로)
        u2 = S.Bars(u, ub.days[ui], ub.o[ui], ub.h[ui], ub.l[ui], ub.c[ui], ub.v[ui])
        sig2 = {k: v[ui] for k, v in D["sig"].items()}
        d0, d1 = RUNS["2026"]
        a, z = p.idx(d0), p.idx(d1, "right") - 1
        if a < 1 or z <= a:
            continue
        hold = p.c[a - 1:z + 1] / p.c[a - 1]
        res = {n: run_rule(sp, p, u2, sig2, a, z, 0.0, _hold=hold) for n, sp in specs.items()}
        for w in ("2026앞", "2026뒤", "2026"):
            _, w0, w1 = WIN[w]
            i0, i1 = slice_idx(p, a, z, w0, w1)
            ru = u2.c[i0:i1 + 1] / u2.c[i0 - 1:i1] - 1
            rp = p.c[i0:i1 + 1] / p.c[i0 - 1:i1] - 1
            fr = frontier(ru, None, 0.0, rp)
            hv = p.c[i0 - 1:i1 + 1] / p.c[i0 - 1]
            for n, (v, m, cyc) in res.items():
                vv = v[i0 - a:i1 - a + 2]
                vv = vv / vv[0]
                exc, _ = eff_excess(float(vv[-1] - 1), _mdd(vv), fr)
                rows.append({"product": p.sym, "under": u, "rule": n, "win": w, "ret": float(vv[-1] - 1), "mdd": _mdd(vv),
                             "hold_ret": float(hv[-1] - 1), "exc": exc})
    return pd.DataFrame(rows)


# ============================================================================ 기준(기초 1배)·고르기
def ref_1x(U) -> pd.DataFrame:
    rows = []
    for u, D in U.items():
        ub = D["und"]
        for w, (rn, w0, w1) in WIN.items():
            d0, d1 = RUNS[rn]
            a, z = ub.idx(d0), ub.idx(d1, "right") - 1
            if a < MIN_HIST or z <= a:
                continue
            i0, i1 = slice_idx(ub, a, z, w0, w1)
            v = ub.c[i0 - 1:i1 + 1] / ub.c[i0 - 1]
            rows.append({"under": u, "win": w, "ret": float(v[-1] - 1), "mdd": _mdd(v)})
        a, z = max(ub.idx(FULL[0]), MIN_HIST), ub.idx(FULL[1], "right") - 1
        v = ub.c[a - 1:z + 1] / ub.c[a - 1]
        rows.append({"under": u, "win": "전체", "ret": float(v[-1] - 1), "mdd": _mdd(v),
                     "cagr": float(v[-1] ** (252 / (z - a + 1)) - 1), "start": str(ub.days[a])})
    return pd.DataFrame(rows)


def select(A: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """사전 고정 고르기(보정 합성, 2026 앞): G1 효율선초과 상품 평균 최대, G2 주기 기대값(n≥30) 최대, G3 동일가중 포트 칼마 최대.
    성공 = 같은 지표(효율선초과 상품 평균) > 0. 보조(사후 추가로 표시): 상품 중앙값 기준, 보유 대비(규칙 − 보유 효율선초과) 기준."""
    out = {}
    for var in ("cal", "basic"):
        a = A[(A["var"] == var) & (A.rule != "보유(2배)")]
        fr = a[a.win == SEL_W].set_index("rule")
        g = {"G1": fr["효율선초과_평균"].idxmax(), "G2": fr[fr["주기n"] >= G2_MIN_N]["기대값"].idxmax(), "G3": fr["칼마"].idxmax()}
        succ = {}
        for key, col in (("평균", "효율선초과_평균"), ("중앙", "효율선초과_중앙")):
            f0 = fr[col]
            pos = f0.index[f0 > 0]
            for w in CHK_W:
                ww = a[a.win == w].set_index("rule")[col]
                succ[f"{key}_{w}"] = {"앞>0": int(len(pos)), "조건부": float(np.mean(ww[pos] > 0)) if len(pos) else None,
                                      "무조건": float(np.mean(ww > 0)), "규칙수": int(len(ww))}
        r = rows[(rows["var"] == var) & rows.win.isin([SEL_W, *CHK_W])]
        h = r[r.rule == "보유(2배)"].set_index(["win", "under"])["exc"]
        r = r.assign(dh=r.exc.to_numpy() - h.loc[list(zip(r.win, r.under))].to_numpy())
        dh = r[r.rule != "보유(2배)"].groupby(["win", "rule"]).dh.mean()
        pos = dh[SEL_W].index[dh[SEL_W] > 0]
        for w in CHK_W:
            succ[f"보유대비_{w}"] = {"앞>0": int(len(pos)), "조건부": float(np.mean(dh[w][pos] > 0)) if len(pos) else None,
                                   "무조건": float(np.mean(dh[w] > 0)), "규칙수": int(len(dh[w]))}
        out[var] = {"고름": g, "성공": succ, "중앙G1": fr["효율선초과_중앙"].idxmax(), "보유대비G1": dh[SEL_W].idxmax()}
    return out


# ============================================================================ 점검
def check(U):
    print("== 엔진 점검 ==")
    u = "NVDA"
    D = U[u]
    ub, p, sig = D["und"], D["syn"]["cal"], D["sig"]
    a, z = ub.idx("2026-01-02"), ub.idx("2026-10-09", "right") - 1
    v, m, _ = run_rule(("H",), p, ub, sig, a, z)
    print("보유 정책 vs 상품 B&H 최대 차(0)", float(np.abs(v - p.c[a - 1:z + 1] / p.c[a - 1]).max()))
    ru = ub.c[a:z + 1] / ub.c[a - 1:z] - 1
    k2 = np.concatenate([[1.0], np.cumprod(1 + kpath(ru, D["rf"][a:z + 1] / 252, 2.0, D["extra"]))])
    print("고정 k=2 공식 vs 보정 합성 보유 최대 차(0)", float(np.abs(k2 - v).max()))
    pb = D["syn"]["basic"]
    k2b = np.concatenate([[1.0], np.cumprod(1 + kpath(ru, D["rf"][a:z + 1] / 252, 2.0, 0.0))])
    print("고정 k=2 공식 vs 기본 합성 보유 최대 차(0)", float(np.abs(k2b - pb.c[a - 1:z + 1] / pb.c[a - 1]).max()))
    never = {k: np.zeros_like(x) if k.startswith("off") else x for k, x in sig.items()}
    v2, _, _ = run_rule(("T", "own", 50), p, ub, never, a, z)
    v3, _, _ = run_rule(("D", ("T", "own", 50)), p, ub, never, a, z)
    print("신호 없음 T·D vs 보유 최대 차(0)", float(np.abs(v2 - v).max()), float(np.abs(v3 - v).max()))
    always = {k: np.ones_like(x) if k.startswith("off") else x for k, x in sig.items()}
    v4, m4, _ = run_rule(("D", ("T", "own", 50)), p, ub, always, a, z)
    man = np.concatenate([[1.0], [p.o[a] / p.c[a - 1] * ub.c[t] / ub.o[a] for t in range(a, z + 1)]])
    print("늘 신호 D vs 첫 시가에 기초로 갈아탄 수동 계산 최대 차(0)", float(np.abs(v4 - man).max()), "배수", m4.min(), m4.max())
    # D 엔진(현금 대신 기초)과 stock_lab: 기초 대신 '가격 고정 자산'을 넣으면 T 규칙과 같아야 한다
    flat = S.Bars("CASH", ub.days, np.ones_like(ub.c), np.ones_like(ub.c), np.ones_like(ub.c), np.ones_like(ub.c), ub.v)
    v5, _, _ = run_rule(("D", ("T", "own", 50)), p, flat, sig, a, z)
    v6, _, _ = run_rule(("T", "own", 50), p, ub, sig, a, z)
    print("D 엔진(현금 자산) vs stock_lab T 규칙 최대 차(0)", float(np.abs(v5 - v6).max()))
    # P = 0.5 규칙 + 0.5 보유
    vp, mp, _ = run_rule(("P", ("T", "own", 50)), p, ub, sig, a, z)
    print("P 곡선 = 0.5·T + 0.5·보유 최대 차(0)", float(np.abs(vp - (0.5 * v6 + 0.5 * v)).max()), "P 배수 범위", round(mp.min(), 3), round(mp.max(), 3))
    # 미래 교란: T 뒤 가격을 흔들어도 T까지 가치가 같아야 한다(신호도 다시 계산)
    T = a + 80
    rng = np.random.default_rng(3)
    ub2 = S.Bars(ub.sym, ub.days, ub.o.copy(), ub.h.copy(), ub.l.copy(), ub.c.copy(), ub.v)
    for arr in (ub2.o, ub2.h, ub2.l, ub2.c):
        arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
    ub2.h[T + 1:] = np.maximum.reduce([ub2.h[T + 1:], ub2.o[T + 1:], ub2.c[T + 1:]])
    ub2.l[T + 1:] = np.minimum.reduce([ub2.l[T + 1:], ub2.o[T + 1:], ub2.c[T + 1:]])
    p2 = S.Bars(p.sym, p.days, p.o.copy(), p.h.copy(), p.l.copy(), p.c.copy(), p.v)
    for t in range(T + 1, len(p.c)):     # 합성 상품도 교란된 기초로 다시 만든다(같은 공식)
        pc = ub2.c[t - 1]
        r = L * (ub2.c[t] / pc - 1) - (L - 1) * D["rf"][t] / 252 - (FEE + D["extra"]) / 252
        p2.c[t] = p2.c[t - 1] * max(1 + r, 0.01)
        p2.o[t] = p2.c[t - 1] * max(1 + L * (ub2.o[t] / pc - 1), 0.01)
        hi = p2.c[t - 1] * max(1 + L * (ub2.h[t] / pc - 1), 0.01)
        lo = p2.c[t - 1] * max(1 + L * (ub2.l[t] / pc - 1), 0.01)
        p2.h[t], p2.l[t] = max(hi, p2.o[t], p2.c[t]), min(lo, p2.o[t], p2.c[t])
    sig2 = signals(ub2, LL.load()["QQQ"])
    worst = 0.0
    for spec in (("T", "own", 20), ("S", 0.15, "hi20"), ("V", 1.5), ("D", ("T", "qqq", 50)), ("TS", 50, 0.25), ("TV", 200, 1.5)):
        w1, _, _ = run_rule(spec, p, ub, sig, a, z)
        w2, _, _ = run_rule(spec, p2, ub2, sig2, a, z)
        worst = max(worst, float(np.abs(w1[:T - a + 2] - w2[:T - a + 2]).max()))
    print("미래 교란 뒤 t까지 가치 최대 차(0)", worst)
    print("2026 합성 vs 실제(보정 합성은 실제 상품 대조 표 참조)")


# ============================================================================ 실행
def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"


def main():
    t0 = time.time()
    U, excluded, prods = prepare()
    specs = rule_specs()
    n_rules = len(specs) - 1
    print(f"기초 {len(U)}개(제외 {excluded}) · 규칙 {n_rules}개 + 보유 · 변형 2(기본·보정)")
    for u, D in U.items():
        print(f"  {u:5s} 시작 {D['start']} 추가비용 {D['extra']:.3f} 상품 {[r['product'] for r in D['track']['products']]}")
    if "check" in sys.argv:
        check(U)
        return
    check(U)
    rows, cyc, daily = evaluate(U, specs, keep_curves=True)
    A = agg(rows, cyc, daily)
    # 미끄러짐 0.5%: 보정 합성, 2026·2025
    rows_s, cyc_s, daily_s = evaluate(U, specs, variants=("cal",), runs={k: RUNS[k] for k in ("2026", "2025")},
                                      slip=SLIP, keep_curves=True, full=False)
    A_s = agg(rows_s, cyc_s, daily_s)
    d = LL.load()
    act = actual_check(U, prods, specs, d)
    print(f"계산 끝 {time.time() - t0:.0f}s")

    out = {"config": {"L": L, "FEE": FEE, "SPAC_START": SPAC_START, "MIN_HIST": MIN_HIST, "T_N": T_N, "T_SRC": T_SRC,
                      "S_X": S_X, "S_RE": S_RE, "V_M": V_M, "P_Q": P_Q, "P_BASE": P_BASE, "D_BASE": D_BASE, "C_TS": C_TS,
                      "C_TV": C_TV, "K_GRID": [float(k) for k in K_GRID], "SLIP": SLIP, "G2_MIN_N": G2_MIN_N, "DEATH": DEATH,
                      "n_rules": n_rules, "excluded": excluded},
           "unders": {u: {"start": D["start"], "extra": D["extra"], "track": D["track"]} for u, D in U.items()}}
    out["select"] = select(A, rows)
    print("고르기", json.dumps(out["select"], ensure_ascii=False))
    out["ref_1x"] = ref_1x(U).to_dict("records")
    out["agg"] = A.to_dict("records")
    out["agg_slip"] = A_s.to_dict("records")
    out["rows"] = rows.to_dict("records")
    out["actual"] = act.to_dict("records")
    (RESULTS / "p4_LEVSINGLE.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("저장", RESULTS / "p4_LEVSINGLE.json")


if __name__ == "__main__":
    main()
