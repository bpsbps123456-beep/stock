"""반박 검증 V_DEEP(2026-10-10): 갈래 DEEP의 대표 후보 D1_s25_d30_floor.

목적: '진짜 우위이고 이웃·다른 해에서도 버틴다'는 주장을 깨 보기. stock_lab.py를 import하지 않고
cache/p2_long.parquet만으로 엔진을 새로 짠다(배당 수정 = Adj Close/Close 비율을 시가·고가·저가·종가에 곱함).
ETF 판정은 cache/p2_universe.json의 'ETF' 꼬리표로 한다.

후보 규칙(종목마다 따로, 창 첫날 직전 종가에 전량 보유, 추적 기준 최고 = 그 종가)
  보유 중(일반 모드) t일 장중 손절가 = 마지막 매수 뒤 t−1까지 최고 종가(매수 체결가 포함, 엔진과 같게) × (1−s).
     시가 ≤ 손절가면 시가, 아니면 저가 ≤ 손절가일 때 손절가에 전량 매도.
  현금 중 (1) 시가: 매도 다음 날부터 c[t−1] > max(h[t−21..t−2]) 이면 t일 시가 전량 매수(일반 모드).
          (2) 장중 지정가: H = max(c[t−252..t−1]), lvl = (1−d)·H. lvl < c[t−1] 일 때만 걸고, 손절한 날에는 lvl < 손절 체결가일 때만.
              저가 ≤ lvl 이면 min(시가, lvl)에 전량 매수 → 바닥 모드: 손절가 = H·(1−d−gap), 종가 ≥ rec·H 이면 일반 모드.
  같은 날 순서: 시가 매수 → 손절 → 지정가 매수.
주기 = 판 체결가/다시 산 체결가 − 1(창 끝까지 못 되샀으면 창 끝 종가). 매도일이 창 안인 주기만 그 창에 센다.

할 일 1) 독립 재계산 2) 시점 감사·하루 늦춘 판·미끄러짐 3) 이웃 4) 위약 5) 장세 → results/p2_V_DEEP.md·json

실행: cd code && PYTHONIOENCODING=utf-8 py p2_V_DEEP.py        (check 인자를 주면 미래 교란 점검만)
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

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않음)
CAND = {"stop": 0.25, "depth": 0.30, "gap": 0.10, "rec": 0.90, "brk": 20, "hlook": 252}
# 파라미터마다 후보 앞뒤 2칸씩. 손절·깊이는 DEEP 격자 간격 5%p, 바닥 간격도 5%p, 회복 5%p, 돌파 5일, 고점 창 63일(분기)
NEIGH = {"stop": [0.15, 0.20, 0.30, 0.35], "depth": [0.20, 0.25, 0.35, 0.40], "gap": [0.0, 0.05, 0.15, 0.20],
         "rec": [0.80, 0.85, 0.95, 1.00], "brk": [10, 15, 25, 30], "hlook": [126, 189, 315, 378]}
GRID2 = {"stop": [0.15, 0.20, 0.25, 0.30, 0.35], "depth": [0.20, 0.25, 0.30, 0.35, 0.40]}   # 손절×깊이 5×5(후보 포함)
SLIP = 0.005
N_SEEDS = 20
EFF_STEP = 0.005
MIN_HIST = 252
WIN = {
    "2026": ("2026-01-02", "2026-10-09"), "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2022": ("2022-01-03", "2022-12-30"), "2018": ("2018-01-02", "2018-12-31"),
    "2018Q4": ("2018-10-01", "2018-12-31"),
}
RUNS = {"2026": WIN["2026"], "2025": WIN["2025"], "2022": WIN["2022"], "2018": WIN["2018"], "2018Q4": WIN["2018Q4"]}
SLICES = {"2026": ["2026", "2026앞", "2026뒤"]}
CAND_ID = "D1_s25_d30_floor"


# ============================================================================ 자료
def load():
    df = pd.read_parquet(CACHE / "p2_long.parquet")
    tags = json.loads((CACHE / "p2_universe.json").read_text(encoding="utf-8"))
    etf = {s for s, v in tags.items() if "ETF" in v}
    data = {}
    for sym, g in df.sort_values(["sym", "date"]).groupby("sym"):
        if sym in etf:
            continue
        g = g.dropna(subset=["Open", "High", "Low", "Close"])
        g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0)]
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        o, h, l, c = (g[k].to_numpy() * f for k in ("Open", "High", "Low", "Close"))
        S = {"sym": sym, "days": g.date.to_numpy().astype(str), "o": o.tolist(), "h": h.tolist(), "l": l.tolist(),
             "c": c.tolist(), "v": g.Volume.fillna(0).to_numpy(float).tolist(), "ch": {}, "hh": {}}
        cs, hs = pd.Series(c), pd.Series(h)
        for n in sorted({CAND["hlook"], *NEIGH["hlook"]}):
            S["ch"][n] = cs.rolling(n, min_periods=1).max().tolist()
        for n in sorted({CAND["brk"], *NEIGH["brk"]}):
            S["hh"][n] = hs.rolling(n, min_periods=n).max().tolist()
        data[sym] = S
    return data


JUNK = 0.05      # 실행 창에서 거래량 0이거나 고가=저가인 날이 5% 넘으면 '정제' 판에서 뺀다(2018 QUBT 0.015→0.32 같은 껍데기 가격)


def junk_frac(S, a, z):
    v, h, l = S["v"][a:z + 1], S["h"][a:z + 1], S["l"][a:z + 1]
    return float(np.mean([(x <= 0) or (y == w) for x, y, w in zip(v, h, l)]))


def span(S, d0, d1):
    a = int(np.searchsorted(S["days"], d0, "left"))
    z = int(np.searchsorted(S["days"], d1, "right")) - 1
    return a, z


# ============================================================================ 엔진(독립)
def sim(S, a, z, p, slip_stop=0.0, slip_buy=0.0, reentry=None, rng=None):
    """후보 규칙. reentry=None이면 원래 재진입(돌파·깊은 지정가). reentry=(durs, p_deep)면 위약 B:
    같은 손절, 재진입만 '나가 있던 기간' 분포에서 뽑은 k일 뒤 시가(확률 p_deep로 바닥 모드 흉내)."""
    o, h, l, c = S["o"], S["h"], S["l"], S["c"]
    CH, HH = S["ch"][p["hlook"]], S["hh"][p["brk"]]
    s, d, gap, rec = p["stop"], p["depth"], p["gap"], p["rec"]
    cash, sh, hold = 0.0, 1.0 / c[a - 1], True
    peak, mode, Href, floor = c[a - 1], "n", None, None
    exit_px, days_out, re_day = None, 0, None
    vals, pos, cyc = [1.0], [], []
    for t in range(a, z + 1):
        # 1) 시가 재진입
        if not hold:
            go = False
            if reentry is None:
                if days_out >= 1:
                    hn = HH[t - 2]
                    go = hn == hn and c[t - 1] > hn
            else:
                go = re_day is not None and t == re_day
            if go:
                fill = o[t] * (1 + slip_buy)
                sh, cash, hold, peak = cash / fill, 0.0, True, (None if p.get("pk_close") else fill)
                mode = "n"
                if reentry is not None and rng.random() < reentry[1]:
                    mode, Href = "f", fill / (1 - d)
                    floor = Href * (1 - d - gap)
                cyc[-1].update(re_t=t, re_px=fill, how="brk" if reentry is None else "rand")
        # 2) 손절(손절가는 t−1까지 정보)
        stopped = False
        if hold:
            Lv = floor if mode == "f" else (peak * (1 - s) if peak is not None else -1.0)
            px = o[t] if o[t] <= Lv else (Lv if l[t] <= Lv else None)
            if px is not None:
                fill = px * (1 - slip_stop)
                cash, sh, hold = sh * fill, 0.0, False
                exit_px, days_out, stopped = fill, 0, True
                cyc.append({"exit_t": t, "exit_px": fill, "re_t": None, "re_px": None, "how": None, "mode": mode})
                if reentry is not None:
                    re_day = t + int(rng.choice(reentry[0]))
        # 3) 장중 지정가(깊은 재매수)
        if not hold and reentry is None and d is not None:
            H = CH[t - 1]
            lvl = (1 - d) * H
            if lvl < c[t - 1] and (not stopped or lvl < exit_px) and l[t] <= lvl:
                fill = min(o[t], lvl) * (1 + slip_buy)
                sh, cash, hold, peak = cash / fill, 0.0, True, (None if p.get("pk_close") else fill)
                mode, Href, floor = "f", H, H * (1 - d - gap)
                cyc[-1].update(re_t=t, re_px=fill, how="deep")
        vals.append(cash + sh * c[t])
        pos.append(1.0 if hold else 0.0)
        if hold:
            if peak is None or c[t] > peak:
                peak = c[t]
            if mode == "f" and c[t] >= rec * Href:
                mode = "n"
        else:
            days_out += 1
    return vals, pos, cyc


def sim_delay(S, a, z, p):
    """하루 늦춘 판: 모든 신호를 하루 늦게 '시가'에 체결. 장중 손절가 닿음(t) → t+1 시가 매도,
    지정가 닿음(t) → t+1 시가 매수(바닥 모드), 돌파 신호(c[t−1]) → t+1 시가 매수."""
    o, h, l, c = S["o"], S["h"], S["l"], S["c"]
    CH, HH = S["ch"][p["hlook"]], S["hh"][p["brk"]]
    s, d, gap, rec = p["stop"], p["depth"], p["gap"], p["rec"]
    cash, sh, hold = 0.0, 1.0 / c[a - 1], True
    peak, mode, Href, floor = c[a - 1], "n", None, None
    exit_px, days_out = None, 0
    pend_sell, pend_buy = False, None
    vals, pos, cyc = [1.0], [], []
    for t in range(a, z + 1):
        sold = False
        if hold and pend_sell:
            fill = o[t]
            cash, sh, hold, exit_px, days_out, sold = sh * fill, 0.0, False, fill, 0, True
            cyc.append({"exit_t": t, "exit_px": fill, "re_t": None, "re_px": None, "how": None, "mode": mode})
        elif not hold and pend_buy is not None:
            fill = o[t]
            sh, cash, hold, peak = cash / fill, 0.0, True, fill
            if pend_buy[0] == "deep":
                mode, Href, floor = "f", pend_buy[1], pend_buy[1] * (1 - d - gap)
            else:
                mode = "n"
            cyc[-1].update(re_t=t, re_px=fill, how=pend_buy[0])
        pend_sell, pend_buy = False, None
        if hold:
            Lv = floor if mode == "f" else peak * (1 - s)
            if l[t] <= Lv:
                pend_sell = True
        else:
            hn = HH[t - 2]
            if days_out >= 1 and hn == hn and c[t - 1] > hn:
                pend_buy = ("brk", None)
            else:
                H = CH[t - 1]
                lvl = (1 - d) * H
                if lvl < c[t - 1] and (not sold or lvl < exit_px) and l[t] <= lvl:
                    pend_buy = ("deep", H)
        vals.append(cash + sh * c[t])
        pos.append(1.0 if hold else 0.0)
        if hold:
            if c[t] > peak:
                peak = c[t]
            if mode == "f" and c[t] >= rec * Href:
                mode = "n"
        else:
            days_out += 1
    return vals, pos, cyc


def sim_schedule(S, a, z, outs):
    """위약 A: 정해진 날 시가에 팔고(s) 정해진 날 시가에 되산다(e). outs = [(s, e)], e > z면 끝까지 현금."""
    o, c = S["o"], S["c"]
    sells = {s_: e_ for s_, e_ in outs}
    buys = {e_ for _, e_ in outs}
    cash, sh, hold = 0.0, 1.0 / c[a - 1], True
    vals, pos = [1.0], []
    for t in range(a, z + 1):
        if not hold and t in buys:
            sh, cash, hold = cash / o[t], 0.0, True
        if hold and t in sells:
            cash, sh, hold = sh * o[t], 0.0, False
        vals.append(cash + sh * c[t])
        pos.append(1.0 if hold else 0.0)
    return vals, pos


# ============================================================================ 집계
def mdd(v):
    v = np.asarray(v, float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def excess(curve, bh_daily):
    tgt = mdd(curve)
    best, be = -1.0, 0.0
    for e in np.arange(0.0, 1.0 + 1e-9, EFF_STEP):
        cur = np.concatenate([[1.0], np.cumprod(1 + e * bh_daily)])
        if mdd(cur) >= tgt - 1e-12 and cur[-1] - 1 > best:
            best, be = cur[-1] - 1, e
    return float(curve[-1] - 1 - best), float(be)


def cyc_stats(gains):
    g = np.asarray(gains, float)
    if not len(g):
        return {"손절주기": 0, "승률": float("nan"), "평균이익": float("nan"), "평균손실": float("nan"),
                "손익비": float("nan"), "기대값": float("nan")}
    w, lo = g[g > 0], g[g <= 0]
    return {"손절주기": int(len(g)), "승률": float(len(w) / len(g)), "평균이익": float(w.mean()) if len(w) else float("nan"),
            "평균손실": float(lo.mean()) if len(lo) else float("nan"),
            "손익비": float(w.mean() / -lo.mean()) if len(w) and len(lo) and lo.mean() < 0 else float("nan"),
            "기대값": float(g.mean())}


def cycle_gains(per_sym, d0, d1, override=None):
    """per_sym: {sym: (S, z_run, cycles)}. 매도일이 [d0, d1]인 주기 손익. override(sym, i, cy) → (re_t, re_px) 대체."""
    out = []
    for sym, (S, z, cycles) in per_sym.items():
        days, c = S["days"], S["c"]
        zz = int(np.searchsorted(days, d1, "right")) - 1
        for i, cy in enumerate(cycles):
            if not (d0 <= days[cy["exit_t"]] <= d1):
                continue
            re_t, re_px = (cy["re_t"], cy["re_px"]) if override is None else override(sym, i, cy)
            if re_t is not None and re_t <= zz:
                out.append(cy["exit_px"] / re_px - 1)
            else:
                out.append(cy["exit_px"] / c[zz] - 1)
    return out


def run(data, rname, p, mode="base", slip_stop=0.0, slip_buy=0.0, schedule=None, reentry=None, seed=0, clean=False,
        static=None):
    d0, d1 = RUNS[rname]
    R, B, E, per, A0 = {}, {}, {}, {}, {}
    rng = np.random.default_rng(seed)
    for sym, S in data.items():
        a, z = span(S, d0, d1)
        if a < MIN_HIST or z <= a:
            continue
        if clean and junk_frac(S, a, z) > JUNK:
            continue
        if mode == "base":
            v, ps, cy = sim(S, a, z, p, slip_stop, slip_buy)
        elif mode == "delay":
            v, ps, cy = sim_delay(S, a, z, p)
        elif mode == "sched":
            v, ps = sim_schedule(S, a, z, schedule.get(sym, []))
            cy = []
        elif mode == "rand":
            v, ps, cy = sim(S, a, z, p, reentry=reentry, rng=rng)
        elif mode == "static":                       # 진단: 종목별 실제 평균 노출을 매일 일정하게(시점 없음, 사후 기울기)
            e = static.get(sym, 1.0)
            cc = np.asarray(S["c"][a - 1:z + 1])
            v = np.concatenate([[1.0], np.cumprod(1 + e * (cc[1:] / cc[:-1] - 1))]).tolist()
            ps, cy = [e] * (z - a + 1), []
        dd = S["days"][a:z + 1]
        v = np.asarray(v)
        cc = np.asarray(S["c"][a - 1:z + 1])
        R[sym] = pd.Series(v[1:] / v[:-1] - 1, index=dd)
        B[sym] = pd.Series(cc[1:] / cc[:-1] - 1, index=dd)
        E[sym] = pd.Series(ps, index=dd)
        per[sym] = (S, z, cy)
        A0[sym] = a
    return {"R": pd.DataFrame(R).sort_index(), "B": pd.DataFrame(B).sort_index(), "E": pd.DataFrame(E).sort_index(),
            "per": per, "a": A0}


def stats(res, rname, with_cyc=True):
    out = {}
    for w in SLICES.get(rname, [rname]):
        d0, d1 = WIN[w]
        m = (res["R"].index >= d0) & (res["R"].index <= d1)
        R, B, E = res["R"][m], res["B"][m], res["E"][m]
        pr = np.concatenate([[1.0], (1 + R.mean(axis=1).fillna(0)).cumprod().to_numpy()])
        pb = np.concatenate([[1.0], (1 + B.mean(axis=1).fillna(0)).cumprod().to_numpy()])
        sr, sb = (1 + R.fillna(0)).prod() - 1, (1 + B.fillna(0)).prod() - 1
        exc, e_eq = excess(pr, B.mean(axis=1).fillna(0).to_numpy())
        o = {"종목수": int(R.shape[1]), "포트_수익": float(pr[-1] - 1), "포트_낙폭": mdd(pr), "포트BH_수익": float(pb[-1] - 1),
             "포트BH_낙폭": mdd(pb), "효율선초과": exc, "효율선_e": e_eq, "평균노출": float(E.mean().mean()),
             "수익_나은종목비율": float((sr > sb + 1e-12).mean()), "최악5%_종목수익": float(sr.quantile(0.05)),
             "최악5%_BH": float(sb.quantile(0.05))}
        if with_cyc:
            o.update(cyc_stats(cycle_gains(res["per"], d0, d1)))
            allc = [cy for (S, z, cyc) in res["per"].values() for cy in cyc if d0 <= S["days"][cy["exit_t"]] <= d1]
            done = [cy for cy in allc if cy["re_t"] is not None]
            o["깊은재매수_비율"] = float(np.mean([cy["how"] == "deep" for cy in done])) if done else float("nan")
        out[w] = o
    return out


def full(data, p, runs=("2026", "2025"), **kw):
    out = {}
    for r in runs:
        out.update(stats(run(data, r, p, **kw), r))
    return out


# ============================================================================ 위약
def out_intervals(res):
    """종목별 실제 나가 있던 구간 [(매도일 t, 나가 있던 종가 수)] (끝까지 못 샀으면 z+1−t)."""
    out = {}
    for sym, (S, z, cyc) in res["per"].items():
        out[sym] = [(cy["exit_t"], (cy["re_t"] if cy["re_t"] is not None else z + 1) - cy["exit_t"]) for cy in cyc]
    return out


def random_schedule(res, rng):
    sched = {}
    for sym, (S, z, cyc) in res["per"].items():
        a = res["a"][sym]
        N = z - a + 1
        durs = [((cy["re_t"] if cy["re_t"] is not None else z + 1) - cy["exit_t"]) for cy in cyc]
        if not durs:
            continue
        durs = list(rng.permutation(durs))
        n, D = len(durs), sum(durs)
        free = N - D - (n - 1)
        extra = 1
        if free < 0:
            free, extra = N - D, 0
        cuts = np.sort(rng.integers(0, free + 1, n))       # 별·막대: n+1 칸으로 나누기
        parts = np.diff(np.concatenate([[0], cuts, [free]]))
        t = a + int(parts[0])
        outs = []
        for i, du in enumerate(durs):
            outs.append((t, t + int(du)))
            t = t + int(du) + extra + int(parts[i + 1])
        sched[sym] = outs
    return sched


def rank_of(actual, plac):
    plac = np.asarray(plac, float)
    return int(1 + np.sum(plac > actual + 1e-12)), float(np.median(plac)), float(np.mean(plac > actual + 1e-12))


# ============================================================================ 점검
def check(data=None):
    data = data or load()
    syms = [s for s in sorted(data) if span(data[s], *RUNS["2026"])[0] >= MIN_HIST][:30]
    worst = 0.0
    for p in (CAND, {**CAND, "stop": 0.15, "depth": 0.40}):
        for s in syms:
            S = data[s]
            a, z = span(S, *RUNS["2026"])
            for T in (a + 40, a + 100):
                for fn in (lambda X: sim(X, a, z, p)[0], lambda X: sim_delay(X, a, z, p)[0]):
                    v1 = fn(S)
                    S2 = {**S, "o": list(S["o"]), "h": list(S["h"]), "l": list(S["l"]), "c": list(S["c"])}
                    rng = np.random.default_rng(T)
                    o, h, l, c = (np.array(S2[k]) for k in "ohlc")
                    for arr in (o, h, l, c):
                        arr[T + 1:] *= 1 + rng.normal(0, 0.08, len(arr) - T - 1)
                    h[T + 1:] = np.maximum.reduce([h[T + 1:], o[T + 1:], c[T + 1:]])
                    l[T + 1:] = np.minimum.reduce([l[T + 1:], o[T + 1:], c[T + 1:]])
                    S2.update(o=o.tolist(), h=h.tolist(), l=l.tolist(), c=c.tolist(),
                              ch={k: pd.Series(c).rolling(k, min_periods=1).max().tolist() for k in S["ch"]},
                              hh={k: pd.Series(h).rolling(k, min_periods=k).max().tolist() for k in S["hh"]})
                    v2 = fn(S2)
                    worst = max(worst, float(np.max(np.abs(np.array(v1[:T - a + 2]) - np.array(v2[:T - a + 2])))))
    # 같은 날 종가를 바꿔도 그날 체결(매매 기록)은 그대로여야 한다(종가는 다음 날 판단에만 쓰임)
    same_day = 0
    for s in syms:
        S = data[s]
        a, z = span(S, *RUNS["2026"])
        _, _, cy1 = sim(S, a, z, CAND)
        for cy in cy1:
            T = cy["exit_t"]
            c = list(S["c"])
            c[T] = c[T] * 0.5
            S2 = {**S, "c": c, "h": [max(x, y) for x, y in zip(S["h"], c)], "l": [min(x, y) for x, y in zip(S["l"], c)],
                  "ch": {k: pd.Series(c).rolling(k, min_periods=1).max().tolist() for k in S["ch"]}}
            S2["l"][T] = S["l"][T] if c[T] >= S["l"][T] else c[T]
            _, _, cy2 = sim(S2, a, z, CAND)
            hit = [x for x in cy2 if x["exit_t"] == T]
            if not hit or abs(hit[0]["exit_px"] - cy["exit_px"]) > 1e-9:
                same_day += 1
    return {"미래교란_최대차": worst, "종가교란_매도변화": same_day}


# ============================================================================ 본체
def deep_fill_quality(res):
    """깊은 지정가 체결일 저가가 시가·종가보다 10% 넘게 아래(꼬리 이상치 의심)인 경우."""
    bad, n = [], 0
    for sym, (S, z, cyc) in res["per"].items():
        for cy in cyc:
            if cy["how"] == "deep":
                n += 1
                t = cy["re_t"]
                if S["l"][t] < 0.9 * min(S["o"][t], S["c"][t]):
                    bad.append((sym, S["days"][t], round(S["l"][t] / min(S["o"][t], S["c"][t]) - 1, 3)))
    return {"깊은체결": n, "꼬리의심": len(bad), "목록": bad[:15]}


def main():
    t0 = time.time()
    data = load()
    print("종목", len(data), "2026 대상", sum(1 for S in data.values() if span(S, *RUNS["2026"])[0] >= MIN_HIST))
    claim_all = json.loads((RESULTS / "p2_DEEP.json").read_text(encoding="utf-8"))["p3"]
    claim, claim_slip = claim_all["results"][CAND_ID], claim_all["slip"][CAND_ID]
    ctl = claim_all["results"]["D1ctl_s25"]

    # 1) 재계산 ----------------------------------------------------------------
    base_res = {r: run(data, r, CAND) for r in RUNS}
    base = {}
    for r, res in base_res.items():
        base.update(stats(res, r))
    keys = ("포트_수익", "포트_낙폭", "포트BH_수익", "포트BH_낙폭", "효율선초과", "평균노출", "손절주기", "승률", "평균이익",
            "평균손실", "손익비", "기대값")
    cmp_rows, maxdiff = [], 0.0
    for w in ("2026", "2026앞", "2026뒤", "2025", "2022", "2018Q4"):
        for k in keys:
            mine, cl = base[w].get(k), claim.get(w, {}).get(k)
            if cl is None:
                continue
            dif = float(mine - cl)
            if k not in ("손절주기", "손익비"):
                maxdiff = max(maxdiff, abs(dif))
            cmp_rows.append({"창": w, "항목": k, "주장": cl, "재계산": mine, "차": dif})
    slip_all = full(data, CAND, slip_stop=SLIP, slip_buy=SLIP)
    ctl_mine = full(data, {**CAND, "depth": None}, runs=("2026", "2025"))
    for w in ("2026", "2026앞", "2026뒤", "2025"):
        for k in ("포트_수익", "효율선초과", "손절주기", "기대값"):
            cl = claim_slip.get(w, {}).get(k)
            if cl is not None:
                d = float(slip_all[w][k] - cl)
                if k != "손절주기":
                    maxdiff = max(maxdiff, abs(d))
                cmp_rows.append({"창": w, "항목": "미끄럼0.5%·" + k, "주장": cl, "재계산": slip_all[w][k], "차": d})
            cl = ctl.get(w, {}).get(k)
            if cl is not None:
                cmp_rows.append({"창": w, "항목": "통제(돌파만)·" + k, "주장": cl, "재계산": ctl_mine[w][k],
                                 "차": float(ctl_mine[w][k] - cl)})
    print(f"재계산 최대 차(개수 제외) {maxdiff:.5f}  ({time.time()-t0:.0f}s)")
    quality = {r: deep_fill_quality(base_res[r]) for r in ("2026", "2025")}

    # 2) 시점 감사 --------------------------------------------------------------
    audit = check(data)
    delay = full(data, CAND, mode="delay", runs=("2026", "2025", "2022"))
    slip_stop = full(data, CAND, slip_stop=SLIP)
    print("시점", audit, f"({time.time()-t0:.0f}s)")

    # 3) 이웃 --------------------------------------------------------------------
    variants = {}
    for k, vals in NEIGH.items():
        for v in vals:
            variants[f"{k}={v}"] = {**CAND, k: v}
    for s in GRID2["stop"]:
        for d in GRID2["depth"]:
            pv = {**CAND, "stop": s, "depth": d}
            if pv != CAND and not any(pv == x for x in variants.values()):
                variants[f"stop={s},depth={d}"] = pv
    neigh = {}
    for name, pv in variants.items():
        st = full(data, pv, runs=("2026", "2025"))
        neigh[name] = {w: {k: st[w][k] for k in ("포트_수익", "포트_낙폭", "효율선초과", "손절주기", "승률", "손익비", "기대값")}
                       for w in ("2026", "2026앞", "2026뒤", "2025")}
    names = list(neigh)
    one = [n for n in names if "," not in n]
    allpos = lambda lst: float(np.mean([all(neigh[n][w]["효율선초과"] > 0 for w in ("2026앞", "2026뒤", "2025")) for n in lst]))  # noqa: E731
    frac = lambda lst, w: float(np.mean([neigh[n][w]["효율선초과"] > 0 for n in lst]))  # noqa: E731
    neigh_sum = {"개수": len(names), "한칸씩": len(one), "손절×깊이_추가": len(names) - len(one),
                 "앞뒤2025_모두>0": allpos(names), "앞>0": frac(names, "2026앞"), "뒤>0": frac(names, "2026뒤"),
                 "2025>0": frac(names, "2025"), "2026>0": frac(names, "2026"),
                 "앞초과_중앙": float(np.median([neigh[n]["2026앞"]["효율선초과"] for n in names])),
                 "뒤초과_중앙": float(np.median([neigh[n]["2026뒤"]["효율선초과"] for n in names])),
                 "2025초과_중앙": float(np.median([neigh[n]["2025"]["효율선초과"] for n in names])),
                 "앞초과_후보보다큼": int(sum(neigh[n]["2026앞"]["효율선초과"] > base["2026앞"]["효율선초과"] for n in names)),
                 "뒤수익_최고": max((neigh[n]["2026뒤"]["포트_수익"], n) for n in names)}
    print("이웃", neigh_sum, f"({time.time()-t0:.0f}s)")

    # 4) 위약 --------------------------------------------------------------------
    plac = {}
    for r in ("2026", "2025"):
        res0 = base_res[r]
        done = [du for lst in out_intervals(res0).values() for (t_, du) in lst]
        comp = [((cy["re_t"]) - cy["exit_t"]) for (S, z, cyc) in res0["per"].values() for cy in cyc if cy["re_t"] is not None]
        p_deep = float(np.mean([cy["how"] == "deep" for (S, z, cyc) in res0["per"].values() for cy in cyc if cy["re_t"] is not None]))
        A, Bp, C = [], [], []
        A0 = {}                                     # 창마다 그 창의 종목별 실제 평균 노출(사후)로 고정
        for w in SLICES.get(r, [r]):
            Ew = res0["E"][(res0["E"].index >= WIN[w][0]) & (res0["E"].index <= WIN[w][1])]
            stat_e = {sym: float(Ew[sym].dropna().mean()) for sym in res0["per"]}
            A0[w] = stats(run(data, r, CAND, mode="static", static=stat_e), r, with_cyc=False)[w]
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + seed)
            A.append(stats(run(data, r, CAND, mode="sched", schedule=random_schedule(res0, rng)), r, with_cyc=False))
            Bp.append(stats(run(data, r, CAND, mode="rand", reentry=(np.array(comp), p_deep), seed=2000 + seed), r))
            # C: 같은 매도, 재진입만 무작위 k일 뒤 시가(주기 손익만)
            rng2 = np.random.default_rng(3000 + seed)
            draw = {(sym, i): int(rng2.choice(comp)) for sym, (S, z, cyc) in res0["per"].items() for i in range(len(cyc))}

            def ov(sym, i, cy, draw=draw):
                S, z, _ = res0["per"][sym]
                t = cy["exit_t"] + draw[(sym, i)]
                return (t, S["o"][t]) if t <= z else (None, None)
            C.append({w: cyc_stats(cycle_gains(res0["per"], *WIN[w], override=ov)) for w in SLICES.get(r, [r])})
        for w in SLICES.get(r, [r]):
            act = base[w]
            plac[w] = {
                "A_수익": rank_of(act["포트_수익"], [x[w]["포트_수익"] for x in A]),
                "A_효율선초과": rank_of(act["효율선초과"], [x[w]["효율선초과"] for x in A]),
                "A_낙폭": rank_of(act["포트_낙폭"], [x[w]["포트_낙폭"] for x in A]),
                "A_노출_실제_위약": (act["평균노출"], float(np.mean([x[w]["평균노출"] for x in A]))),
                "A0_고정노출_수익_낙폭_초과": (A0[w]["포트_수익"], A0[w]["포트_낙폭"], A0[w]["효율선초과"]),
                "B_노출_실제_위약": (act["평균노출"], float(np.mean([x[w]["평균노출"] for x in Bp]))),
                "B_수익_중앙_낙폭_중앙": (float(np.median([x[w]["포트_수익"] for x in Bp])),
                                    float(np.median([x[w]["포트_낙폭"] for x in Bp]))),
                "B_수익": rank_of(act["포트_수익"], [x[w]["포트_수익"] for x in Bp]),
                "B_효율선초과": rank_of(act["효율선초과"], [x[w]["효율선초과"] for x in Bp]),
                "B_기대값": rank_of(act["기대값"], [x[w]["기대값"] for x in Bp]),
                "C_기대값": rank_of(act["기대값"], [x[w]["기대값"] for x in C]),
                "C_승률": rank_of(act["승률"], [x[w]["승률"] for x in C]),
                "C_손익비_중앙": float(np.nanmedian([x[w]["손익비"] for x in C])),
                "나간기간_중앙": float(np.median(done)), "p_deep": p_deep,
            }
    print("위약", {w: {k: v for k, v in x.items() if k.endswith("초과") or k.startswith("C_기대")} for w, x in plac.items()},
          f"({time.time()-t0:.0f}s)")

    # 5) 장세 ---------------------------------------------------------------------
    clean = {}
    for r in ("2022", "2018", "2018Q4"):
        res_c = run(data, r, CAND, clean=True)
        clean.update({w + "_정제": v for w, v in stats(res_c, r).items()})
        clean[r + "_정제"]["뺀종목"] = sorted(set(base_res[r]["per"]) - set(res_c["per"]))
    base.update(clean)
    pkc = full(data, {**CAND, "pk_close": True}, runs=("2026", "2025"))
    regimes = {w: {k: base[w][k] for k in ("포트_수익", "포트_낙폭", "포트BH_수익", "포트BH_낙폭", "효율선초과", "평균노출",
                                         "손절주기", "승률", "평균이익", "평균손실", "손익비", "기대값", "수익_나은종목비율",
                                         "최악5%_종목수익", "최악5%_BH")}
               for w in ("2026앞", "2026뒤", "2026", "2025", "2022", "2018", "2018Q4", "2022_정제", "2018_정제", "2018Q4_정제")}

    payload = {"grid": {"CAND": CAND, "NEIGH": NEIGH, "GRID2": GRID2, "N_NEIGH": len(names), "SLIP": SLIP, "N_SEEDS": N_SEEDS,
                        "EFF_STEP": EFF_STEP},
               "재계산": {"최대차_개수제외": maxdiff, "표": cmp_rows}, "base": base, "slip_all": slip_all,
               "slip_stop": slip_stop, "delay": delay, "pk_close": pkc, "control_mine": ctl_mine, "audit": audit, "fill_quality": quality,
               "neighbors": neigh, "neighbors_summary": neigh_sum, "placebo": plac, "regimes": regimes}
    p = RESULTS / "p2_V_DEEP.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("저장", p, f"({time.time()-t0:.0f}s)")
    for w in ("2026", "2026앞", "2026뒤", "2025", "2022", "2018", "2018Q4", "2022_정제", "2018_정제", "2018Q4_정제"):
        b = base[w]
        print(f"{w}: {b['포트_수익']:+.3f}/{b['포트_낙폭']:.3f} BH {b['포트BH_수익']:+.3f}/{b['포트BH_낙폭']:.3f} 초과 {b['효율선초과']:+.3f} "
              f"n{b['손절주기']} 승 {b['승률']:.3f} 이익 {b['평균이익']:+.3f} 손실 {b['평균손실']:+.3f} 손익비 {b['손익비']:.2f} 기대 {b['기대값']:+.4f}")
    print("정제 뺀 종목", {w: base[w].get("뺀종목") for w in ("2022_정제", "2018_정제", "2018Q4_정제")})
    for lab, dct in (("지연", delay), ("손절미끄럼", slip_stop), ("전체미끄럼", slip_all), ("체결가제외최고", pkc)):
        for w in ("2026", "2026앞", "2026뒤", "2025", "2022"):
            if w in dct:
                b = dct[w]
                print(f" {lab} {w}: {b['포트_수익']:+.3f}/{b['포트_낙폭']:.3f} 초과 {b['효율선초과']:+.3f} n{b['손절주기']} "
                      f"승 {b['승률']:.3f} 손익비 {b['손익비']:.2f} 기대 {b['기대값']:+.4f}")
    print("품질", quality)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print(check())
    else:
        main()
