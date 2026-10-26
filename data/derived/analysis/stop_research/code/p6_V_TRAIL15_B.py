"""6단계 반박 검증 V_TRAIL15 관점 B(2026-10-11): 단일 종목 2배 '상품 고점 대비 추적 −15% 손절 → 판 날부터 20거래일 뒤 시가 재매수'.

사용자: "단일종목2배레버리지는 20일 거래일 뒤에사라고했는데..." 4단계 LEVSINGLE 35개 격자의 한 칸이다(대표 후보가 아니어서 검증을 안 받음).
관점 B = 위약 · 실제 상품 · 다른 재진입 · 사용자가 실제로 고를 대안.
  0) 재현: 이 파일의 독립 엔진(sim_trail)이 갈래 엔진(p4_LEVSINGLE.run_rule)·stock_lab 정책과 같은 곡선을 내는지, 갈래 주장 숫자.
  1) 위약: 같은 상품·같은 실행 창·같은 판 횟수 N·같은 20거래일 현금 기간을 무작위 날짜(시가 매도 → 20거래일 뒤 시가 재매수)에 둔 판.
     시드 20. 날짜는 상품마다 따로 뽑는다(손절은 상품마다 다른 날 나므로). 순위 = 규칙 + 위약 20개 = 21개 중 몇 등.
  2) 실제 상품 28개(NVDL·NVDX·NVDU·TSLL·TSLT·MSTU·MSTX·CONL·AMDL·PLTU·MUU·AVL·AVGX·SMCX·TSMX·METU·AAPU·GGLL·AMZU·MSFU·NFXL·
     HOOX·IONX·RGTX·SNDU·ARMG·CRWG·OKLL 전부 자료 있음): 상장 후 전체(상장 5거래일 뒤 시작)·2026 앞·뒤·2025. 같은 날짜 보정 합성과 비교.
  3) 재진입 4개(손절 −15% 고정, 사전 고정): 20거래일 뒤 / 20거래일 뒤 단 기초 종가 > SMA50일 때만(아니면 매일 기다림) /
     기초 20일 최고 종가 경신 / 판 가격 지정가(판 다음 날부터, 20거래일 안에 안 닿으면 20일째 시가).
  4) 비교 기준: 같은 종목 1배 보유, 2배 보유, 2배 1/2만 담기(나머지 현금 0%, 매일 맞춤 = 일수익 0.5 × 2배 상품).
  5) 이웃(사용자 질문 '몇 일·몇 %'): 손절 {10, 12.5, 15, 20, 25%} × 기다림 {10, 15, 20, 30, 40거래일} = 25칸(후보 포함), 재진입은 'W거래일 뒤'.
시점: 손절선 = 상품 산 뒤 최고 종가(t−1까지) × (1 − X), t일 장중(시가가 선 아래면 시가). 재매수 = 판 날 + W거래일 시가.
     SMA50·20일 신고가는 t−1 종가. 지정가는 판 가격(이미 아는 값). 현금 이자 0, 수수료 0(미끄러짐 0.5% 민감도 따로).
자료: p4_LEVSINGLE.prepare()(보정 합성 = 2×기초 − 금리 − 보수 1.05% − 실제 상품 최근 252일 로그 추적차)를 그대로 읽는다. 엔진·캐시는 고치지 않는다.
실행: cd code && PYTHONIOENCODING=utf-8 py p6_V_TRAIL15_B.py  → results/p6_V_TRAIL15_B.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lev_lab as LL
import p4_LEVSINGLE as P4
import stock_lab as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않는다)
X0, W0 = 0.15, 20
MODES = ("t20", "t20sma", "hi20", "limit")
MODE_NAME = {"t20": "20거래일 뒤(후보)", "t20sma": "20거래일 뒤 + 기초>SMA50", "hi20": "기초 20일 신고가",
             "limit": "판 가격 지정가(20일째 시가)"}
GRID_X = (0.10, 0.125, 0.15, 0.20, 0.25)
GRID_W = (10, 15, 20, 30, 40)
N_SEED = 20
SLIP = 0.005
DEATH = -0.80
ACT = tuple("NVDL NVDX NVDU TSLL TSLT MSTU MSTX CONL AMDL PLTU MUU AVL AVGX SMCX TSMX METU AAPU GGLL AMZU MSFU NFXL "
            "HOOX IONX RGTX SNDU ARMG CRWG OKLL".split())
ACT_SKIP = 5                     # 상장 첫 며칠 값 이상(SMCX 둘째 날 +68% vs 기초 −3%) → 상장 5거래일 뒤 종가에 들고 시작
YEAR_WINS = [str(y) for y in range(2015, 2027)]
MAIN = ("2026앞", "2026뒤", "2025")
CTX = ("2022", "2020급락", "2018Q4")
RUNS, WIN, FULL, MIN_HIST = P4.RUNS, P4.WIN, P4.FULL, P4.MIN_HIST

CLAIM = {
    "상품해_전멸": {"보유": "37/196", "규칙": "1/196", "1배": "2/196"}, "2022_전멸": {"보유": "11/16", "규칙": "0/16"},
    "중앙_수익/낙폭": {"2026앞": (-0.10, -0.31, 0.25), "2026뒤": (-0.09, -0.30, -0.30), "2025": (0.54, -0.38, -0.01),
                    "2022": (-0.35, -0.50, -0.83)},
    "연율14": {"규칙": (0.160, -0.78), "보유": (0.321, None), "1배": (0.264, None)},
    "판1회": {"2026앞": (82, 0.37, 1.59, -0.013), "2026뒤": (74, 0.59, 2.02, 0.142), "2025": (168, 0.54, 2.42, 0.190)},
    "실제22_효율선초과": {"2026앞": (-0.477, -0.230, "9/22"), "2026뒤": (-0.116, -0.163, "14/22")},
}


# ============================================================================ 엔진
def sim_trail(o, h, l, c, a, z, x, wait, mode="t20", up50=None, hi20=None, slip=0.0):
    """a−1 종가에 상품 100%(가치 1). 반환 (가치[z−a+2], 종가 보유 여부[z−a+1], 주기 [exit_t, exit_px, re_t, re_px])."""
    held, sh, cash = True, 1.0 / c[a - 1], 0.0
    peak, exit_t, exit_px = c[a - 1], -1, 0.0
    vals = [1.0]
    pos = []
    cyc = []
    for t in range(a, z + 1):
        lim_buy = False
        if not held:
            px = None
            if mode == "t20":
                if t - exit_t >= wait:
                    px = o[t]
            elif mode == "t20sma":
                if t - exit_t >= wait and up50[t - 1]:
                    px = o[t]
            elif mode == "hi20":
                if t - 1 >= exit_t and hi20[t - 1]:
                    px = o[t]
            elif mode == "limit":
                if t - exit_t >= wait:
                    px = o[t]
                elif l[t] <= exit_px:
                    px, lim_buy = min(o[t], exit_px), True
            if px is not None:
                fill = px * (1 + slip)
                sh, cash, held, peak = cash / fill, 0.0, True, fill
                cyc[-1][2], cyc[-1][3] = t, fill
        if held and not lim_buy:
            stop = peak * (1 - x)
            sp = o[t] if o[t] <= stop else (stop if l[t] <= stop else None)
            if sp is not None:
                fill = sp * (1 - slip)
                cash, sh, held, exit_t, exit_px = sh * fill, 0.0, False, t, fill
                cyc.append([t, fill, None, None])
        vals.append(cash + sh * c[t])
        pos.append(held)
        if held and c[t] > peak:
            peak = c[t]
    return np.array(vals), np.array(pos, bool), cyc


def curve_cash(o, c, a, z, cash_rel):
    """cash_rel[i] = a+i일 종가에 현금. 매도는 그 구간 첫날 시가, 재매수는 구간 다음 날 시가."""
    held = ~cash_rel
    h0 = np.concatenate([[True], held[:-1]])
    cc = c[a:z + 1] / c[a - 1:z]
    so = o[a:z + 1] / c[a - 1:z]
    oc = c[a:z + 1] / o[a:z + 1]
    g = np.where(h0 & held, cc, np.where(h0 & ~held, so, np.where(~h0 & held, oc, 1.0)))
    return np.concatenate([[1.0], np.cumprod(g)]), held


PLACEBO_NOTE = {"간격20": 0, "줄임": 0, "전체": 0}


def placebo_cash(n_days, n_exit, wait, rng):
    """[0, n_days)에 시작점 n_exit개(간격 ≥ wait+1, 끝은 잘림)를 고르게 뽑는다. 안 되면 간격 ≥ wait."""
    cash = np.zeros(n_days, bool)
    PLACEBO_NOTE["전체"] += 1
    if n_exit <= 0:
        return cash, 0
    gap = wait if n_days - wait * (n_exit - 1) >= n_exit else wait - 1
    span = n_days - gap * (n_exit - 1)
    if gap < wait:
        PLACEBO_NOTE["간격20"] += 1
    if span < n_exit:                                     # 그래도 안 되면 가능한 만큼(보고)
        PLACEBO_NOTE["줄임"] += 1
        n_exit = max(1, (n_days + gap) // (gap + 1))
        span = n_days - gap * (n_exit - 1)
    xs = np.sort(rng.choice(span, n_exit, replace=False))
    starts = xs + gap * np.arange(n_exit)
    for s in starts:
        cash[s:s + wait] = True
    return cash, n_exit


def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def excess(ret, dd, fr):
    ok = fr[:, 1] >= dd - 1e-12
    j = int(np.argmax(np.where(ok, fr[:, 0], -np.inf)))
    return float(ret - fr[j, 0]), float(P4.K_GRID[j])


# ============================================================================ 자료
def build():
    U, excluded, prods = P4.prepare()
    for u, D in U.items():
        ub = D["und"]
        s50 = ub.sma(50)
        D["up50"] = np.isfinite(s50) & (ub.c > s50)
        D["hi20"] = D["sig"]["hi20"]
        p = D["syn"]["cal"]
        D["L"] = {k: getattr(p, k).tolist() for k in ("o", "h", "l", "c")}
    d = LL.load()
    A = {}
    for pn in ACT:
        if pn not in d:
            continue
        u = prods[pn]["under"]
        if u not in U:
            continue
        pb, ub = d[pn], U[u]["und"]
        ui = np.searchsorted(ub.days, pb.days)
        ok = ui < len(ub.days)
        ok[ok] = ub.days[ui[ok]] == pb.days[ok]
        keep = np.flatnonzero(ok)
        ui = ui[keep]
        A[pn] = {"under": u, "days": pb.days[keep], "o": pb.o[keep], "h": pb.h[keep], "l": pb.l[keep], "c": pb.c[keep],
                 "uo": ub.o[ui], "uc": ub.c[ui], "up50": U[u]["up50"][ui], "hi20": U[u]["hi20"][ui], "ui": ui,
                 "gaps": int((np.diff(ui) != 1).sum()), "dropped": int(len(pb.days) - len(keep))}
        A[pn]["L"] = {k: A[pn][k].tolist() for k in ("o", "h", "l", "c")}
    return U, excluded, prods, A


# ============================================================================ 전략 곡선
def strat_curve(spec, Lp, cnp, onp, uc, up50, hi20, a, z, n_exit=None, rng=None):
    """spec → (가치, 배수[종가], 주기). Lp = 상품 o/h/l/c 리스트, cnp/onp = 상품 종가·시가 배열, uc = 기초 종가 배열."""
    k = spec[0]
    if k == "hold2":
        return cnp[a - 1:z + 1] / cnp[a - 1], np.full(z - a + 1, 2.0), []
    if k == "hold1":
        return uc[a - 1:z + 1] / uc[a - 1], np.full(z - a + 1, 1.0), []
    if k == "half":
        r = cnp[a:z + 1] / cnp[a - 1:z] - 1
        return np.concatenate([[1.0], np.cumprod(1 + 0.5 * r)]), np.full(z - a + 1, 1.0), []
    if k == "trail":
        _, x, w, mode, slip = spec
        v, pos, cyc = sim_trail(Lp["o"], Lp["h"], Lp["l"], Lp["c"], a, z, x, w, mode, up50, hi20, slip)
        return v, 2.0 * pos, cyc
    if k == "placebo":
        cash, _ = placebo_cash(z - a + 1, n_exit, spec[2], rng)
        v, held = curve_cash(onp, cnp, a, z, cash)
        return v, 2.0 * held, []
    raise ValueError(spec)


def cyc_gains(cyc, i0, i1, c):
    g = []
    for et, ep, rt, rp in cyc:
        if not (i0 <= et <= i1):
            continue
        g.append(ep / rp - 1 if (rt is not None and rt <= i1) else ep / c[i1] - 1)
    return g


# ============================================================================ 합성 평가
_FR = {}


def frontier_syn(u, w, ru, rfd, extra):
    key = ("syn", u, w)
    if key not in _FR:
        _FR[key] = P4.frontier(ru, rfd, extra)
    return _FR[key]


def eval_syn(U, specs: dict, n_exits=None, seed=None, want_cyc=True):
    """specs 이름 → spec. 반환 rows(DataFrame), cyc(이름·창별 판 1회 손익 목록), exits[(이름, 상품, 실행)] = 판 횟수."""
    rows, cyc_rows, exits = [], {}, {}
    run_list = list(RUNS.items()) + [("FULL", FULL)]
    for ui_, (u, D) in enumerate(U.items()):
        ub, p = D["und"], D["syn"]["cal"]
        for ri, (rn, (d0, d1)) in enumerate(run_list):
            if rn == "FULL":
                a, z = max(ub.idx(d0), MIN_HIST), ub.idx(d1, "right") - 1
            else:
                a, z = ub.idx(d0), ub.idx(d1, "right") - 1
                if a < MIN_HIST or z <= a:
                    continue
            hold = p.c[a - 1:z + 1] / p.c[a - 1]
            for name, spec in specs.items():
                rng = None
                ne = None
                if spec[0] == "placebo":
                    ne = n_exits[(spec[3], u, rn)]
                    rng = np.random.default_rng([spec[1], ui_, ri])
                v, m, cyc = strat_curve(spec, D["L"], p.c, p.o, ub.c, D["up50"], D["hi20"], a, z, ne, rng)
                exits[(name, u, rn)] = len(cyc)
                if rn == "FULL":
                    yrs = (z - a + 1) / 252
                    rows.append({"name": name, "under": u, "win": "전체", "ret": float(v[-1] - 1), "mdd": mdd(v),
                                 "cagr": float(max(v[-1], 1e-9) ** (1 / yrs) - 1), "start": str(ub.days[a]),
                                 "kbar": float(m.mean()), "n_exit": len(cyc), "hold_mdd": mdd(hold)})
                    continue
                for w, (wr, w0, w1) in WIN.items():
                    if wr != rn:
                        continue
                    i0, i1 = P4.slice_idx(ub, a, z, w0, w1)
                    if i1 <= i0:
                        continue
                    ru = ub.c[i0:i1 + 1] / ub.c[i0 - 1:i1] - 1
                    rfd = D["rf"][i0:i1 + 1] / 252
                    fr = frontier_syn(u, w, ru, rfd, D["extra"])
                    vv = v[i0 - a:i1 - a + 2]
                    vv = vv / vv[0]
                    hv = hold[i0 - a:i1 - a + 2] / hold[i0 - a]
                    ret, dd = float(vv[-1] - 1), mdd(vv)
                    exc, keq = excess(ret, dd, fr)
                    hexc, _ = excess(float(hv[-1] - 1), mdd(hv), fr)
                    rows.append({"name": name, "under": u, "win": w, "ret": ret, "mdd": dd, "exc": exc, "k_eq": keq,
                                 "hold_ret": float(hv[-1] - 1), "hold_mdd": mdd(hv), "hold_exc": hexc,
                                 "kbar": float(m[i0 - a:i1 - a + 1].mean())})
                    if want_cyc and cyc:
                        cyc_rows.setdefault((name, w), []).extend(cyc_gains(cyc, i0, i1, p.c))
    return pd.DataFrame(rows), cyc_rows, exits


def cyc_stats(x):
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0}
    win, loss = x[x > 0], x[x <= 0]
    return {"n": int(len(x)), "승률": float(len(win) / len(x)), "평균이익": float(win.mean()) if len(win) else None,
            "평균손실": float(loss.mean()) if len(loss) else None,
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else None,
            "기대값": float(x.mean())}


def summarize(rows, cyc, name):
    """이름 하나의 요약: 창별 중앙 수익·낙폭·효율선초과, 상품-해 전멸, 14종목 연율."""
    g = rows[rows.name == name]
    out = {}
    for w in (*MAIN, "2026", *CTX, *YEAR_WINS):
        gw = g[g.win == w]
        if not len(gw) or w in out:
            continue
        out[w] = {"n": int(len(gw)), "ret_med": float(gw.ret.median()), "mdd_med": float(gw.mdd.median()),
                  "exc_mean": float(gw.exc.mean()), "exc_med": float(gw.exc.median()),
                  "vs_hold_mean": float((gw.exc - gw.hold_exc).mean()), "vs_hold_win": int(((gw.exc - gw.hold_exc) > 1e-12).sum()),
                  "beat_hold_ret": int((gw.ret > gw.hold_ret + 1e-12).sum()),
                  "hold_ret_med": float(gw.hold_ret.median()), "hold_mdd_med": float(gw.hold_mdd.median()),
                  "dead": int((gw.mdd <= DEATH).sum()), "kbar": float(gw.kbar.mean()),
                  "ret_mean_exSNDK": float(gw[gw.under != "SNDK"].ret.mean())}
        if (name, w) in cyc:
            out[w]["cyc"] = cyc_stats(cyc[(name, w)])
    py = g[g.win.isin(YEAR_WINS)]
    out["상품해"] = {"n": int(len(py)), "dead": int((py.mdd <= DEATH).sum()), "dd50": int((py.mdd <= -0.5).sum()),
                   "ret_med": float(py.ret.median()), "worst": float(py.ret.min())}
    f = g[g.win == "전체"]
    f14 = f[f.start == FULL[0]]
    out["전체"] = {"n14": int(len(f14)), "cagr14_med": float(f14.cagr.median()), "mdd14_med": float(f14.mdd.median()),
                 "dead23": int((f.mdd <= DEATH).sum()), "n23": int(len(f)), "dead14": int((f14.mdd <= DEATH).sum()),
                 "kbar14": float(f14.kbar.mean())}
    return out


# ============================================================================ 실제 상품
def eval_act(A, U, specs: dict):
    """실제 상품 × 실행(상장후·2026·2025). 같은 날짜 보정 합성도 같은 spec으로."""
    rows = []
    for pn, P in A.items():
        u = P["under"]
        D = U[u]
        days = P["days"]
        n = len(days)
        e = int(np.searchsorted(days, FULL[1], side="right")) - 1
        runs = {"상장후": (ACT_SKIP, e)}
        for rn, d0 in (("2026", "2026-01-02"), ("2025", "2025-01-02")):
            a = int(np.searchsorted(days, d0))
            if a >= ACT_SKIP and a < n:
                z = int(np.searchsorted(days, RUNS[rn][1], side="right")) - 1
                runs[rn] = (a, z)
        ps = D["syn"]["cal"]
        for rn, (a, z) in runs.items():
            # 같은 날짜 합성(기초 인덱스 공간)
            sa, sz = int(P["ui"][a]), int(P["ui"][z])
            wins = {"상장후": [("상장후", days[a], days[z])], "2025": [("2025", *WIN["2025"][1:])],
                    "2026": [(w, *WIN[w][1:]) for w in ("2026앞", "2026뒤", "2026")]}[rn]
            ru_all = P["uc"]
            for name, spec in specs.items():
                v, m, cyc = strat_curve(spec, P["L"], P["c"], P["o"], P["uc"], P["up50"], P["hi20"], a, z)
                vs, ms, cycs = strat_curve(spec, D["L"], ps.c, ps.o, D["und"].c, D["up50"], D["hi20"], sa, sz)
                hold = P["c"][a - 1:z + 1] / P["c"][a - 1]
                for w, w0, w1 in wins:
                    i0 = max(int(np.searchsorted(days, w0)), a)
                    i1 = min(int(np.searchsorted(days, w1, side="right")) - 1, z)
                    if i1 <= i0:
                        continue
                    ru = ru_all[i0:i1 + 1] / ru_all[i0 - 1:i1] - 1
                    rp = P["c"][i0:i1 + 1] / P["c"][i0 - 1:i1] - 1
                    key = ("act", pn, rn, w)
                    if key not in _FR:
                        _FR[key] = P4.frontier(ru, None, 0.0, rp)
                    fr = _FR[key]
                    vv = v[i0 - a:i1 - a + 2]
                    vv = vv / vv[0]
                    hv = hold[i0 - a:i1 - a + 2] / hold[i0 - a]
                    ret, dd = float(vv[-1] - 1), mdd(vv)
                    exc, _ = excess(ret, dd, fr)
                    hexc, _ = excess(float(hv[-1] - 1), mdd(hv), fr)
                    j0, j1 = int(P["ui"][i0]), int(P["ui"][i1])
                    sv = vs[j0 - sa:j1 - sa + 2]
                    sv = sv / sv[0]
                    rows.append({"name": name, "product": pn, "under": u, "run": rn, "win": w, "start": str(days[i0 - 1]),
                                 "days": int(i1 - i0 + 1), "ret": ret, "mdd": dd, "exc": exc, "hold_ret": float(hv[-1] - 1),
                                 "hold_mdd": mdd(hv), "hold_exc": hexc, "syn_ret": float(sv[-1] - 1), "syn_mdd": mdd(sv),
                                 "n_exit": len([cy for cy in cyc if i0 <= cy[0] <= i1]),
                                 "syn_n_exit": len([cy for cy in cycs if j0 <= cy[0] <= j1]),
                                 "gains": cyc_gains(cyc, i0, i1, P["c"])})
    return pd.DataFrame(rows)


def summ_act(R, name, w, subset=None):
    g = R[(R.name == name) & (R.win == w)]
    if subset is not None:
        g = g[g["product"].isin(subset)]
    if not len(g):
        return {}
    x = [y for gg in g.gains for y in gg]
    return {"n": int(len(g)), "ret_med": float(g.ret.median()), "mdd_med": float(g.mdd.median()),
            "exc_mean": float(g.exc.mean()), "exc_med": float(g.exc.median()),
            "vs_hold_win": int(((g.exc - g.hold_exc) > 1e-12).sum()), "beat_hold_ret": int((g.ret > g.hold_ret + 1e-12).sum()),
            "shallower": int((g.mdd > g.hold_mdd + 1e-12).sum()),
            "hold_ret_med": float(g.hold_ret.median()), "hold_mdd_med": float(g.hold_mdd.median()),
            "syn_ret_med": float(g.syn_ret.median()), "syn_mdd_med": float(g.syn_mdd.median()),
            "act_minus_syn_med": float((np.log1p(g.ret) - np.log1p(g.syn_ret)).median()),
            "dead": int((g.mdd <= DEATH).sum()), "hold_dead": int((g.hold_mdd <= DEATH).sum()), "cyc": cyc_stats(x)}


# ============================================================================ 점검
class _Pol(S.Policy):
    """stock_lab 정책으로 같은 규칙을 다시 짠 것(엔진 대조용)."""

    def __init__(self, x, w, mode, up50, hi20):
        self.x, self.w, self.mode, self.up50, self.hi20 = x, w, mode, up50, hi20

    def on_open(self, b, t, st):
        if st["pos"] > 1e-9:
            return None
        k = t - st["exit_t"]
        if self.mode in ("t20", "limit"):
            return 1.0 if k >= self.w else None
        if self.mode == "t20sma":
            return 1.0 if (k >= self.w and self.up50[t - 1]) else None
        if self.mode == "hi20":
            return 1.0 if (t - 1 >= st["exit_t"] and self.hi20[t - 1]) else None

    def stop(self, b, t, st):
        return st["peak"] * (1 - self.x)

    def limit_buys(self, b, t, st):
        if self.mode == "limit" and st["pos"] <= 1e-9 and st["exit_t"] is not None and 0 < t - st["exit_t"] < self.w:
            return ((st["exit_px"], 1.0),)
        return ()


class _Sched(S.Policy):
    """정해진 날 시가에 팔고 W거래일 뒤 시가에 되산다(위약 곡선 대조용)."""

    def __init__(self, starts, w):
        self.starts, self.w = set(starts), w
        self.back = {s + w for s in starts}

    def on_open(self, b, t, st):
        if t in self.starts and st["pos"] > 1e-9:
            return 0.0
        if t in self.back and st["pos"] <= 1e-9 and (t not in self.starts):
            return 1.0
        return None


def check(U, A):
    out = {}
    worst_p4, worst_sl = 0.0, {m: 0.0 for m in MODES}
    for u, D in U.items():
        ub, p = D["und"], D["syn"]["cal"]
        for rn in ("2026", "2025", "2022"):
            d0, d1 = RUNS[rn]
            a, z = ub.idx(d0), ub.idx(d1, "right") - 1
            if a < MIN_HIST:
                continue
            v, _, _ = sim_trail(D["L"]["o"], D["L"]["h"], D["L"]["l"], D["L"]["c"], a, z, X0, W0, "t20")
            vp, _, _ = P4.run_rule(("S", X0, "t20"), p, ub, D["sig"], a, z)
            worst_p4 = max(worst_p4, float(np.abs(v - vp).max()))
            for mode in MODES:
                v1, _, _ = sim_trail(D["L"]["o"], D["L"]["h"], D["L"]["l"], D["L"]["c"], a, z, X0, W0, mode, D["up50"], D["hi20"])
                v2, _, _ = S.simulate(p, a, z, _Pol(X0, W0, mode, D["up50"], D["hi20"]))
                worst_sl[mode] = max(worst_sl[mode], float(np.abs(v1 - v2).max()))
    out["독립엔진_vs_p4_run_rule(후보, 23상품×2026·2025·2022)"] = worst_p4
    out["독립엔진_vs_stock_lab_정책(재진입 4개)"] = worst_sl
    # 위약 곡선 vs stock_lab 일정 정책
    worst = 0.0
    rng = np.random.default_rng(5)
    for u in list(U)[:8]:
        D = U[u]
        ub, p = D["und"], D["syn"]["cal"]
        a, z = ub.idx("2026-01-02"), ub.idx("2026-10-09", "right") - 1
        cash, ne = placebo_cash(z - a + 1, 5, W0, rng)
        v1, _ = curve_cash(p.o, p.c, a, z, cash)
        starts = [a + i for i in range(z - a + 1) if cash[i] and (i == 0 or not cash[i - 1])]
        v2, _, _ = S.simulate(p, a, z, _Sched(starts, W0))
        worst = max(worst, float(np.abs(v1 - v2).max()))
    out["위약곡선_vs_stock_lab_일정정책"] = worst
    # 미래 교란: T 뒤 상품 가격을 흔들어도 T까지 가치가 같아야 한다(재진입 신호도 같이 흔듦)
    rng = np.random.default_rng(9)
    worst = 0.0
    for u in list(U)[:6]:
        D = U[u]
        Lp = {k: np.array(v) for k, v in D["L"].items()}
        a, z = D["und"].idx("2026-01-02"), D["und"].idx("2026-10-09", "right") - 1
        for T in (a + 40, a + 120):
            L2 = {k: v.copy() for k, v in Lp.items()}
            for k in L2:
                L2[k][T + 1:] *= 1 + rng.normal(0, 0.06, len(L2[k]) - T - 1)
            L2["h"][T + 1:] = np.maximum.reduce([L2["h"][T + 1:], L2["o"][T + 1:], L2["c"][T + 1:]])
            L2["l"][T + 1:] = np.minimum.reduce([L2["l"][T + 1:], L2["o"][T + 1:], L2["c"][T + 1:]])
            up2, hi2 = D["up50"].copy(), D["hi20"].copy()
            up2[T + 1:] = rng.random(len(up2) - T - 1) > 0.5
            hi2[T + 1:] = rng.random(len(hi2) - T - 1) > 0.5
            for mode in MODES:
                v1, _, _ = sim_trail(Lp["o"], Lp["h"], Lp["l"], Lp["c"], a, z, X0, W0, mode, D["up50"], D["hi20"])
                v2, _, _ = sim_trail(L2["o"], L2["h"], L2["l"], L2["c"], a, z, X0, W0, mode, up2, hi2)
                worst = max(worst, float(np.abs(v1[:T - a + 2] - v2[:T - a + 2]).max()))
    out["미래교란_T까지_가치차"] = worst
    out["실제상품_자료"] = {pn: {"시작": str(P["days"][0]), "기초날짜와_안맞아_뺀날": P["dropped"], "중간_빈날": P["gaps"]}
                        for pn, P in A.items()}
    for k, v in out.items():
        print(" ", k, v if not isinstance(v, dict) or len(v) < 6 else "…")
    return out


# ============================================================================ 실행
def pct(x, d=0):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"


def main():
    t0 = time.time()
    U, excluded, prods, A = build()
    print(f"기초 {len(U)}개 · 실제 상품 {len(A)}개 · {time.time() - t0:.0f}s")
    chk = check(U, A)
    if "check" in sys.argv:
        return

    base = ("trail", X0, W0, "t20", 0.0)
    specs = {"1배 보유": ("hold1",), "2배 보유": ("hold2",), "2배 1/2만(매일 맞춤)": ("half",)}
    for mode in MODES:
        specs[f"손절15% · {MODE_NAME[mode]}"] = ("trail", X0, W0, mode, 0.0)
    for mode in MODES:
        specs[f"손절15% · {MODE_NAME[mode]} · 미끄러짐0.5%"] = ("trail", X0, W0, mode, SLIP)
    for x in GRID_X:
        for w in GRID_W:
            nm = f"격자 손절{x * 100:g}% · {w}거래일 뒤"
            specs[nm] = ("trail", x, w, "t20", 0.0)
    BASE = f"손절15% · {MODE_NAME['t20']}"
    rows, cyc, exits = eval_syn(U, specs)
    print(f"합성 평가 {len(specs)}개 {time.time() - t0:.0f}s")

    # 위약(후보의 판 횟수를 상품·실행마다 그대로)
    n_ex = {(BASE, u, rn): n for (nm, u, rn), n in exits.items() if nm == BASE}
    pl_specs = {f"위약 시드{s}": ("placebo", s, W0, BASE) for s in range(N_SEED)}
    prow, _, _ = eval_syn(U, pl_specs, n_exits=n_ex, want_cyc=False)
    print(f"위약 {N_SEED}개 {time.time() - t0:.0f}s")

    SUM = {nm: summarize(rows, cyc, nm) for nm in specs}
    PSUM = {nm: summarize(prow, {}, nm) for nm in pl_specs}

    # 위약 순위
    def metric(sm):
        return {"상품해_전멸": sm["상품해"]["dead"], "2022_전멸": sm["2022"]["dead"], "연율14_중앙": sm["전체"]["cagr14_med"],
                "장기낙폭14_중앙": sm["전체"]["mdd14_med"], "장기_전멸23": sm["전체"]["dead23"],
                **{f"{w}_수익중앙": sm[w]["ret_med"] for w in MAIN}, **{f"{w}_효율선초과평균": sm[w]["exc_mean"] for w in MAIN},
                **{f"{w}_효율선초과중앙": sm[w]["exc_med"] for w in MAIN}, **{f"{w}_낙폭중앙": sm[w]["mdd_med"] for w in MAIN},
                "2026앞_노출": sm["2026앞"]["kbar"] / 2, "2026뒤_노출": sm["2026뒤"]["kbar"] / 2, "2025_노출": sm["2025"]["kbar"] / 2,
                "장기_노출14": sm["전체"]["kbar14"] / 2}
    lower_better = {"상품해_전멸", "2022_전멸", "장기_전멸23"}
    rm = metric(SUM[BASE])
    pm = pd.DataFrame([metric(s) for s in PSUM.values()])
    rank = {}
    for k, rv in rm.items():
        col = pm[k].to_numpy()
        if "노출" in k:
            rank[k] = {"규칙": rv, "위약_중앙": float(np.median(col)), "위약_범위": [float(col.min()), float(col.max())]}
            continue
        better = (col < rv - 1e-12) if k in lower_better else (col > rv + 1e-12)
        ties = np.abs(col - rv) <= 1e-12
        rank[k] = {"규칙": rv, "위약_중앙": float(np.median(col)), "위약_범위": [float(col.min()), float(col.max())],
                   "순위(21개 중, 1=최선)": int(1 + better.sum()), "동률": int(ties.sum())}
    print("위약 순위", json.dumps({k: v.get("순위(21개 중, 1=최선)") for k, v in rank.items()}, ensure_ascii=False))

    # 위약 상품별: 같은 상품-해에서 위약이 −80%에 빠진 비율
    pr_y = prow[prow.win.isin(YEAR_WINS)]
    per_py = pr_y.assign(dead=pr_y.mdd <= DEATH).groupby(["under", "win"]).dead.mean()
    rb = rows[(rows.name == BASE) & rows.win.isin(YEAR_WINS)].set_index(["under", "win"])
    hold_dead_idx = rows[(rows.name == "2배 보유") & rows.win.isin(YEAR_WINS) & (rows.mdd <= DEATH)].set_index(["under", "win"]).index
    pl_dead_in_hold_dead = float(per_py.loc[hold_dead_idx].mean()) if len(hold_dead_idx) else None
    rule_dead_in_hold_dead = float((rb.loc[hold_dead_idx].mdd <= DEATH).mean()) if len(hold_dead_idx) else None

    # 실제 상품
    act_specs = {k: specs[k] for k in ("1배 보유", "2배 보유", "2배 1/2만(매일 맞춤)")}
    for mode in MODES:
        act_specs[f"손절15% · {MODE_NAME[mode]}"] = specs[f"손절15% · {MODE_NAME[mode]}"]
    for x in GRID_X:
        for w in GRID_W:
            act_specs[f"격자 손절{x * 100:g}% · {w}거래일 뒤"] = ("trail", x, w, "t20", 0.0)
    R = eval_act(A, U, act_specs)
    print(f"실제 상품 {time.time() - t0:.0f}s")
    oldest = []
    for u in sorted({P["under"] for P in A.values()}):
        c = sorted((str(A[p]["days"][0]), p) for p in A if A[p]["under"] == u and str(A[p]["days"][0]) < "2025-12-31")
        if c:
            oldest.append(c[0][1])
    ASUM = {}
    for nm in act_specs:
        ASUM[nm] = {w: summ_act(R, nm, w) for w in ("2026앞", "2026뒤", "2026", "2025", "상장후")}
        ASUM[nm]["22_2026앞"] = summ_act(R, nm, "2026앞", oldest)
        ASUM[nm]["22_2026뒤"] = summ_act(R, nm, "2026뒤", oldest)
    per_prod = R[(R.win == "상장후") & R.name.isin(["1배 보유", "2배 보유", "2배 1/2만(매일 맞춤)", BASE])]
    per_prod = per_prod.pivot_table(index=["product", "start", "days"], columns="name", values=["ret", "mdd"]).reset_index()
    per_prod.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in per_prod.columns]
    syn_same = R[(R.win == "상장후") & (R.name == BASE)][["product", "syn_ret", "syn_mdd", "n_exit", "syn_n_exit"]]
    per_prod = per_prod.merge(syn_same, on="product")

    out = {"config": {"X0": X0, "W0": W0, "MODES": MODE_NAME, "GRID_X": GRID_X, "GRID_W": GRID_W, "N_SEED": N_SEED,
                      "SLIP": SLIP, "ACT": ACT, "ACT_SKIP": ACT_SKIP, "excluded": excluded, "n_specs": len(specs),
                      "oldest22": oldest},
           "check": chk, "placebo_note": PLACEBO_NOTE, "claim": CLAIM, "summary": SUM, "placebo_summary": PSUM, "placebo_rank": rank,
           "placebo_dead_in_hold_dead": {"위약_평균비율": pl_dead_in_hold_dead, "규칙_비율": rule_dead_in_hold_dead,
                                         "보유_전멸_상품해": int(len(hold_dead_idx))},
           "actual_summary": ASUM, "actual_per_product": per_prod.to_dict("records"),
           "actual_rows": R.drop(columns=["gains"]).to_dict("records"),
           "syn_rows_main": rows[rows.win.isin([*MAIN, "전체", *YEAR_WINS, *CTX])].to_dict("records")}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "p6_V_TRAIL15_B.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("저장", RESULTS / "p6_V_TRAIL15_B.json", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
