"""반박 검증 V_LADDER(2026-10-11): 3단계 갈래 LADDER의 대표 후보 G1 'B3 50·100·150·200일선 하나 깰 때마다 25% 매도'.

후보 정의: 종가(t−1)가 SMA50·100·150·200 중 하나를 깰 때마다 25%를 t일 시가에 판다(깬 선 수 k → 보유 1 − 0.25k).
  그 선 위 종가가 되면 그 25%를 다음 시가에 되산다. 선 상태가 바뀐 날에만 목표 비율로 맞추고, 그 밖의 날은 가격대로 흘러간다.
  대상은 AI 종목 동일가중 포트폴리오(매일 재조정).

독립성: stock_lab·tranche_lab·harness·p3_LADDER를 import하지 않는다. cache/p2_long.parquet만 읽고(배당 수정 = Adj Close/Close
  비율을 시가·종가에 곱함), 가치 계산은 '노출 비율' 방식으로 새로 짰다(갈래는 주식 수·현금 방식).

할 일
  1) 재계산: 주장 숫자(앞·뒤·2026·2025·2022·7월 쇼크·미끄러짐) 재현, 0.5%p 넘게 다르면 원인 규명.
  2) 시점 감사(t일 종가·그 뒤 가격을 흔들어도 t일 보유·t−1일까지 수익 불변) + 하루 늦춘 판 + 미끄러짐 0.5%.
  3) 이웃(사전 고정, 아래 상수): 선 위치·간격 5 × 5, 판매 단위, 되사기 띠, 팔기 띠, 확인 일수, 선 수 → '앞·뒤·2025 효율선초과 모두 > 0' 비율.
  4) 위약: 같은 종목의 목표 비율 경로(판 비율)를 날짜만 원형 이동(P1 종목마다 따로, P2 모든 종목 같은 이동, 시드 20),
     종목별 사후 평균 노출 고정(P0), 같은 평균 노출 고정(타이밍값의 기준). 실력 0인 가짜 경로(일 단위 iid 재표본)로 위약의 치우침 보정.
  5) 장세: 2026 앞·뒤, 2025, 2022, 2018(QUBT 제외)의 효율선초과·타이밍값·주기 승률·손익비·기대값.
  6) 판정.

실행: PYTHONIOENCODING=utf-8 py p3_V_LADDER.py        (전체, json·md 저장)
      PYTHONIOENCODING=utf-8 py p3_V_LADDER.py md     (json → md 다시 쓰기)
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
CAND = {"lines": (50, 100, 150, 200), "u": 0.25, "bb": 0.0, "bs": 0.0, "confirm": 1, "lag": 1}
NEIGH_START = (30, 40, 50, 60, 70)        # 첫 선 길이: 후보 50의 ±2칸(10 간격)
NEIGH_GAP = (40, 45, 50, 55, 60)          # 선 간격: 후보 50의 ±2칸(5 간격). 최장 선 70 + 3·60 = 250 ≤ 창 앞 자료 252일
NEIGH_U = (0.15, 0.20, 0.30, 0.35)        # 판매 단위: 후보 25%의 ±2칸(5%p). 30%·35%는 네 번째 선에서 보유 0으로 잘림
NEIGH_BB = (0.01, 0.02)                   # 되사기 띠(종가 > 선 × (1 + b)): 후보 0의 한쪽 2칸. 2%는 갈래의 B8과 같음(교차 확인)
NEIGH_BS = (0.01, 0.02)                   # 팔기 띠(종가 < 선 × (1 − b)): 후보 0의 한쪽 2칸
NEIGH_CONFIRM = (2, 3)                    # 연속 종가 확인 일수: 후보 1의 한쪽 2칸
NEIGH_NLINES = (2, 3, 5)                  # 선 수 n(50·i, i = 1..n), 단위 1/n: 후보 4의 ±2칸 중 6(SMA300)은 자료 252일을 넘어 뺌
SLIP = 0.005                              # 미끄러짐 민감도(팔 때 −, 살 때 +)
N_SEEDS = 20                              # 위약 시드
N_NULL, N_NULL_PLAC = 20, 5               # 귀무 보정: iid 재표본 가짜 경로 20개 × 위약 시드 5
N_BOOT = 500                              # 종목 재표본(신뢰 구간)
MIN_HIST = 252                            # 창 시작 전 필요한 거래일
EFF_STEP = 0.005                          # 효율선 노출 격자 간격
TOL = 0.005                               # 재현 허용 오차 0.5%p
WIN = {
    "2026": ("2026-01-02", "2026-10-09"), "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2024": ("2024-01-02", "2024-12-31"), "2023": ("2023-01-03", "2023-12-29"),
    "2022": ("2022-01-03", "2022-12-30"), "2021": ("2021-01-04", "2021-12-31"), "2020": ("2020-01-02", "2020-12-31"),
    "2019": ("2019-01-02", "2019-12-31"), "2018": ("2018-01-02", "2018-12-31"),
}
RUNS = {k: WIN[k] for k in ("2026", "2025", "2022", "2018", "2024", "2023", "2021", "2020", "2019")}
MAIN_RUNS = ("2026", "2025", "2022", "2018")
SLICES = {"2026": ["2026", "2026앞", "2026뒤"]}
EXCLUDE = {"2018": {"QUBT"}, "2019": {"QUBT"}}     # 2020년 이전 QUBT 장외 자료
SHOCK = ("2026-06-22", "2026-07-29")
PL_WINS = ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")

# 주장(갈래 요약·대표 후보 설명의 반올림 숫자)
CLAIM = [
    ("base", "2026앞", "포트_수익", 0.511), ("base", "2026앞", "포트_낙폭", -0.064), ("base", "2026앞", "효율선초과", 0.204),
    ("base", "2026앞", "타이밍값", 0.146), ("base", "2026앞", "손절주기", 1257), ("base", "2026앞", "승률", 0.20),
    ("base", "2026앞", "손익비", 2.27), ("base", "2026앞", "기대값", -0.022),
    ("base", "2026뒤", "포트_수익", -0.077), ("base", "2026뒤", "포트BH_수익", -0.041), ("base", "2026뒤", "효율선초과", -0.077),
    ("base", "2026뒤", "타이밍값", -0.061), ("base", "2026뒤", "손절주기", 1449), ("base", "2026뒤", "승률", 0.30),
    ("base", "2026뒤", "손익비", 2.45), ("base", "2026뒤", "기대값", 0.003),
    ("base", "2026", "포트_수익", 0.395), ("base", "2026", "포트_낙폭", -0.175), ("base", "2026", "포트BH_수익", 0.586),
    ("base", "2026", "포트BH_낙폭", -0.255), ("base", "2026", "평균노출", 0.59), ("base", "2026", "주기_종목당", 19.6),
    ("base", "2026", "승률", 0.25), ("base", "2026", "손익비", 2.53), ("base", "2026", "기대값", -0.007),
    ("base", "2025", "포트_수익", 0.456), ("base", "2025", "포트BH_수익", 0.743), ("base", "2025", "효율선초과", 0.151),
    ("base", "2025", "타이밍값", 0.011), ("base", "2025", "손절주기", 3006), ("base", "2025", "승률", 0.23),
    ("base", "2025", "손익비", 2.11), ("base", "2025", "기대값", -0.016),
    ("base", "2022", "포트_수익", -0.175), ("base", "2022", "포트BH_수익", -0.306),
    ("shock", "AI쇼크", "수익", -0.145), ("shock", "AI쇼크", "BH수익", -0.218), ("shock", "AI쇼크", "고정노출_수익", -0.132),
    ("slip", "2026", "포트_수익", 0.332), ("slip", "2026앞", "포트_수익", 0.479), ("slip", "2026뒤", "포트_수익", -0.099),
    ("slip", "2025", "포트_수익", 0.380), ("slip", "2025", "타이밍값", -0.064),
]
TOL_BY_KEY = {"손절주기": 0, "손익비": 0.05, "평균노출": 0.01, "주기_종목당": 0.1}
E_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, EFF_STEP), 6)


# ============================================================================ 자료(독립 적재)
def load():
    df = pd.read_parquet(CACHE / "p2_long.parquet")
    tags = json.loads((CACHE / "p2_universe.json").read_text(encoding="utf-8"))
    etf = {s for s, v in tags.items() if "ETF" in v}
    out = {}
    for sym, g in df.groupby("sym", sort=True):
        if sym in etf:
            continue
        g = g.sort_values("date")
        ok = np.isfinite(g[["Open", "High", "Low", "Close"]].to_numpy(float)).all(axis=1)
        ok &= (g.Close.to_numpy() > 0) & (g.Open.to_numpy() > 0) & (g.Low.to_numpy() > 0)
        g = g[ok]
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        out[sym] = {"d": g.date.to_numpy().astype(str), "o": g.Open.to_numpy() * f, "c": g.Close.to_numpy() * f, "_sma": {}}
    return out


def sma(s, n):
    if n not in s["_sma"]:
        c = s["c"]
        cs = np.concatenate([[0.0], np.cumsum(c)])
        m = np.full(len(c), np.nan)
        if len(c) >= n:
            m[n - 1:] = (cs[n:] - cs[:-n]) / n
        s["_sma"][n] = m
    return s["_sma"][n]


def bounds(s, d0, d1):
    return int(np.searchsorted(s["d"], d0, "left")), int(np.searchsorted(s["d"], d1, "right")) - 1


def universe(data, rk):
    d0 = RUNS[rk][0] if rk in RUNS else rk
    ex = EXCLUDE.get(rk, set())
    return [k for k, s in data.items() if k not in ex and int(np.searchsorted(s["d"], d0, "left")) >= MIN_HIST]


# ============================================================================ 규칙: 선 상태 → 목표 비율 → 가치
def line_states(s, a, z, P):
    """H[i, q] = t=a+i일 시가 판단 뒤 선 q의 '보유 몫' 상태. 판단은 c[t−lag]·SMA[t−lag](lag=1: 전날 종가)만 쓴다."""
    c, lines, lag, cf = s["c"], P["lines"], P.get("lag", 1), P.get("confirm", 1)
    bb, bs = P.get("bb", 0.0), P.get("bs", 0.0)
    sm = [sma(s, n) for n in lines]
    L = len(lines)
    H = np.zeros((z - a + 1, L), bool)
    held = [True] * L
    for i in range(z - a + 1):
        j = a + i - lag
        for q in range(L):
            m = sm[q]
            if held[q]:
                if all(np.isfinite(m[j - k]) and c[j - k] < m[j - k] * (1 - bs) for k in range(cf)):
                    held[q] = False
            else:
                if all(np.isfinite(m[j - k]) and c[j - k] > m[j - k] * (1 + bb) for k in range(cf)):
                    held[q] = True
        H[i] = held
    return H


def targets(H, u):
    k = H.shape[1] - H.sum(axis=1)
    tgt = np.clip(1.0 - u * k, 0.0, 1.0)
    prev = np.vstack([np.ones((1, H.shape[1]), bool), H[:-1]])
    chg = (H != prev).any(axis=1)
    return tgt, chg


def sim_path(o, c, a, tgt, chg, ss=0.0, sb=0.0, x0=1.0):
    """노출 비율 방식 가치 계산. a−1 종가에 노출 x0(가치 1)로 시작. t일 시가: chg[i]면 목표 tgt[i]로 맞춘다(그 밖의 날은 흘러감).
    반환: 일수익 r(길이 m), 종가 노출 e(길이 m)."""
    m = len(tgt)
    r, e = np.empty(m), np.empty(m)
    V, x = 1.0, x0
    for i in range(m):
        t = a + i
        g = o[t] / c[t - 1]
        Vo = V * (1 + x * (g - 1))
        St = V * x * g                       # 시가 주식 가치
        Ca = Vo - St                         # 현금
        if chg[i]:
            T = tgt[i]
            cur = St / Vo if Vo > 0 else 0.0
            if abs(T - cur) >= 1e-9:
                if T < cur:
                    sold = St - T * Vo
                    Ca += sold * (1 - ss)
                    St = T * Vo
                else:
                    buy = min((T - cur) * Vo, Ca)
                    St += buy / (1 + sb)
                    Ca -= buy
        Sc = St * c[t] / o[t]
        Vn = Sc + Ca
        r[i] = Vn / V - 1
        e[i] = Sc / Vn if Vn > 0 else 0.0
        V, x = Vn, (Sc / Vn if Vn > 0 else 0.0)
    return r, e


def line_cycles(H, a, o, ss=0.0, sb=0.0):
    """선별 주기 (판 날 t, 판 값, 되산 날 t 또는 None, 되산 값). 주기 손익 = 판 값/되산 값 − 1."""
    out = []
    for q in range(H.shape[1]):
        prev, ex = True, None
        for i in range(H.shape[0]):
            cur = bool(H[i, q])
            t = a + i
            if prev and not cur:
                ex = (t, o[t] * (1 - ss))
            elif cur and not prev:
                out.append((ex[0], ex[1], t, o[t] * (1 + sb)))
                ex = None
            prev = cur
        if ex is not None:
            out.append((ex[0], ex[1], None, None))
    return out


def run(data, rk, P, ss=0.0, sb=0.0, path_override=None):
    """한 실행 창. path_override(k, a, z, H, tgt, chg) → (tgt, chg, x0)이면 그 경로로 가치만 다시 계산(위약)."""
    d0, d1 = RUNS[rk]
    R, B, E, cyc, paths = {}, {}, {}, [], {}
    for k in universe(data, rk):
        s = data[k]
        a, z = bounds(s, d0, d1)
        if a < MIN_HIST or z <= a:
            continue
        H = line_states(s, a, z, P)
        tgt, chg = targets(H, P["u"])
        r, e = sim_path(s["o"], s["c"], a, tgt, chg, ss, sb)
        idx = s["d"][a:z + 1]
        R[k] = pd.Series(r, idx)
        B[k] = pd.Series(s["c"][a:z + 1] / s["c"][a - 1:z] - 1, idx)
        E[k] = pd.Series(e, idx)
        cyc += [(k,) + x for x in line_cycles(H, a, s["o"], ss, sb)]
        paths[k] = (a, z, tgt, chg)
    return {"R": pd.DataFrame(R).sort_index(), "B": pd.DataFrame(B).sort_index(), "E": pd.DataFrame(E).sort_index(),
            "cyc": cyc, "paths": paths, "rk": rk}


# ============================================================================ 평가
def curve(daily):
    return np.concatenate([[1.0], np.cumprod(1 + np.asarray(daily, float))])


def mdd(v):
    v = np.asarray(v, float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def eff_excess(pr, bp):
    tgt = mdd(pr)
    cv = np.cumprod(1 + E_GRID[:, None] * np.asarray(bp, float)[None, :], axis=1)
    cv = np.concatenate([np.ones((len(E_GRID), 1)), cv], axis=1)
    dd = (cv / np.maximum.accumulate(cv, axis=1) - 1).min(axis=1)
    rets = cv[:, -1] - 1
    ok = dd >= tgt - 1e-12
    j = int(np.argmax(np.where(ok, rets, -np.inf)))
    return float(pr[-1] - 1 - rets[j]), float(E_GRID[j])


def cyc_stats(gains, n_stocks):
    g = np.asarray(gains, float)
    if not len(g):
        return {"손절주기": 0, "주기_종목당": 0.0, "승률": np.nan, "평균이익": np.nan, "평균손실": np.nan, "손익비": np.nan,
                "기대값": np.nan}
    w, l_ = g[g > 0], g[g <= 0]
    return {"손절주기": int(len(g)), "주기_종목당": float(len(g) / max(n_stocks, 1)), "승률": float(len(w) / len(g)),
            "평균이익": float(w.mean()) if len(w) else np.nan, "평균손실": float(l_.mean()) if len(l_) else np.nan,
            "손익비": float(w.mean() / -l_.mean()) if len(w) and len(l_) and l_.mean() < 0 else np.nan,
            "기대값": float(g.mean())}


def port(R, B, E):
    rp = R.mean(axis=1).fillna(0).to_numpy()
    bp = B.mean(axis=1).fillna(0).to_numpy()
    pr, pb = curve(rp), curve(bp)
    e_avg = float(E.mean(axis=1).mean())
    pf = curve(e_avg * bp)
    f = E.mean()                                                   # 종목별 사후 평균 노출(P0)
    p0 = curve((B * f).mean(axis=1).fillna(0).to_numpy())
    exc, e_eq = eff_excess(pr, bp)
    sr, sbh = (1 + R.fillna(0)).prod() - 1, (1 + B.fillna(0)).prod() - 1
    return {"종목수": int(R.shape[1]), "포트_수익": float(pr[-1] - 1), "포트_낙폭": mdd(pr),
            "포트BH_수익": float(pb[-1] - 1), "포트BH_낙폭": mdd(pb), "효율선초과": exc, "효율선_e": e_eq, "평균노출": e_avg,
            "고정노출_수익": float(pf[-1] - 1), "고정노출_낙폭": mdd(pf), "타이밍값": float(pr[-1] - pf[-1]),
            "P0_수익": float(p0[-1] - 1), "P0_낙폭": mdd(p0), "종목내타이밍": float(pr[-1] - p0[-1]),
            "종목배분_사후": float(p0[-1] - pf[-1]), "수익_나은종목비율": float((sr > sbh + 1e-12).mean())}


def wstats(data, res, w, with_cyc=True):
    d0, d1 = WIN[w]
    m = (res["R"].index >= d0) & (res["R"].index <= d1)
    out = port(res["R"][m], res["B"][m], res["E"][m])
    if with_cyc:
        gains = []
        for (k, et, ef, rt, rf) in res["cyc"]:
            s = data[k]
            if not (d0 <= s["d"][et] <= d1):
                continue
            _, iz = bounds(s, d0, d1)
            gains.append(ef / rf - 1 if rt is not None and rt <= iz else ef / s["c"][iz] - 1)
        out.update(cyc_stats(gains, out["종목수"]))
    return out


def all_stats(data, res_by_run, with_cyc=True):
    out = {}
    for rk, res in res_by_run.items():
        for w in SLICES.get(rk, [rk]):
            out[w] = wstats(data, res, w, with_cyc)
    return out


def shock_stats(res, e26):
    m = (res["R"].index >= SHOCK[0]) & (res["R"].index <= SHOCK[1])
    rp = res["R"][m].mean(axis=1).fillna(0).to_numpy()
    bp = res["B"][m].mean(axis=1).fillna(0).to_numpy()
    pr, pb, pf = curve(rp), curve(bp), curve(e26 * bp)
    return {"수익": float(pr[-1] - 1), "낙폭": mdd(pr), "BH수익": float(pb[-1] - 1), "BH낙폭": mdd(pb),
            "고정노출_수익": float(pf[-1] - 1), "쇼크_평균노출": float(res["E"][m].mean(axis=1).mean())}


# ============================================================================ 위약: 목표 비율 경로 원형 이동
def placebo_window(data, res, w, mode, rng):
    """창 w 안에서 종목별 실제 (목표 비율, 바꾼 날) 경로를 날짜만 원형 이동해 가치를 다시 잰다.
    mode 'P1' = 종목마다 다른 이동, 'P2' = 모든 종목 같은 이동(포트 노출 경로를 통째로 옮김).
    시작: 창 직전 종가에 이동한 경로 첫 목표 비율로 들고 있다. 이동 뒤 목표가 바뀐 날·원래 바꾼 날 시가에 맞춘다."""
    d0, d1 = WIN[w]
    R, E = {}, {}
    common = None
    for k, (a, z, tgt, chg) in res["paths"].items():
        s = data[k]
        dd = s["d"][a:z + 1]
        i0 = int(np.searchsorted(dd, d0, "left"))
        i1 = int(np.searchsorted(dd, d1, "right")) - 1
        L = i1 - i0 + 1
        if L < 2:
            continue
        if mode == "P1":
            sh = int(rng.integers(1, L))
        else:
            if common is None:
                common = int(rng.integers(1, L))
            sh = common % L
        tw = np.roll(tgt[i0:i1 + 1], sh)
        cw = np.roll(chg[i0:i1 + 1], sh)
        prevt = np.concatenate([[tw[0]], tw[:-1]])
        cw = cw | (np.abs(tw - prevt) > 1e-12)
        r, e = sim_path(s["o"], s["c"], a + i0, tw, cw, 0.0, 0.0, x0=float(tw[0]))
        R[k] = pd.Series(r, dd[i0:i1 + 1])
        E[k] = pd.Series(e, dd[i0:i1 + 1])
    R, E = pd.DataFrame(R).sort_index(), pd.DataFrame(E).sort_index()
    m = (res["B"].index >= d0) & (res["B"].index <= d1)
    B = res["B"][m][R.columns]
    return port(R, B, E)


def rank_vs(actual, plac):
    a = np.asarray(plac, float)
    a = a[np.isfinite(a)]
    if not len(a) or not np.isfinite(actual):
        return np.nan
    return float(((actual > a).sum() + 0.5 * (actual == a).sum()) / len(a))


def null_data(data, rk, rng):
    """실력 0 가짜 자료: 그 실행의 대상 종목을 같은 달력(창 + 앞 252거래일)에 맞추고, 매일 (밤사이, 장중) 로그 수익 쌍을
    그 달력에서 iid로 다시 뽑는다(모든 종목에 같은 날짜 뽑기 → 종목 간 동시 움직임은 남고, 추세·되돌림 예측력은 0)."""
    d0, d1 = RUNS[rk]
    ref = max(data, key=lambda k: len(data[k]["d"]))
    rd = data[ref]["d"]
    i0 = int(np.searchsorted(rd, d0, "left")) - MIN_HIST
    i1 = int(np.searchsorted(rd, d1, "right")) - 1
    cal = rd[i0:i1 + 1]
    n = len(cal)
    draw = rng.integers(1, n, size=n)
    out = {}
    for k in universe(data, rk):
        s = data[k]
        o = pd.Series(s["o"], s["d"]).reindex(cal)
        c = pd.Series(s["c"], s["d"]).reindex(cal).ffill().bfill().to_numpy()
        o = np.where(np.isfinite(o.to_numpy()), o.to_numpy(), c)
        on = np.concatenate([[0.0], np.log(o[1:] / c[:-1])])
        intra = np.log(c / o)
        cs, os_ = np.empty(n), np.empty(n)
        cs[0], os_[0] = c[0], o[0]
        for t in range(1, n):
            os_[t] = cs[t - 1] * np.exp(on[draw[t]])
            cs[t] = os_[t] * np.exp(intra[draw[t]])
        out[k] = {"d": cal, "o": os_, "c": cs, "_sma": {}}
    return out


def null_calibration(data):
    keys = ("포트_수익", "효율선초과", "타이밍값", "종목내타이밍", "종목배분_사후")
    gaps = {w: {k: [] for k in keys} for w in PL_WINS}
    rule = {w: {k: [] for k in keys} for w in PL_WINS}
    rng = np.random.default_rng(777)
    for _ in range(N_NULL):
        for rk in MAIN_RUNS:
            nd = null_data(data, rk, rng)
            # null 자료의 universe는 각 종목이 정확히 252일 앞 자료를 가지므로 모두 들어간다(제외 목록은 그대로).
            br = run(nd, rk, CAND)
            for w in SLICES.get(rk, [rk]):
                bs = wstats(nd, br, w, with_cyc=False)
                ps = [placebo_window(nd, br, w, "P1", rng) for _ in range(N_NULL_PLAC)]
                for k in keys:
                    gaps[w][k].append(bs[k] - float(np.median([p[k] for p in ps])))
                    rule[w][k].append(bs[k])
    return gaps, rule


# ============================================================================ 종목 재표본 신뢰 구간
def boot_ci(res, w, rng):
    d0, d1 = WIN[w]
    m = (res["R"].index >= d0) & (res["R"].index <= d1)
    R, B, E = res["R"][m], res["B"][m], res["E"][m]
    cols = np.array(R.columns)
    tv, ex, ret = [], [], []
    for _ in range(N_BOOT):
        pick = rng.choice(len(cols), len(cols), replace=True)
        Rb, Bb, Eb = R.iloc[:, pick], B.iloc[:, pick], E.iloc[:, pick]
        rp = Rb.mean(axis=1).fillna(0).to_numpy()
        bp = Bb.mean(axis=1).fillna(0).to_numpy()
        pr = curve(rp)
        pf = curve(float(Eb.mean(axis=1).mean()) * bp)
        tv.append(pr[-1] - pf[-1])
        ex.append(eff_excess(pr, bp)[0])
        ret.append(pr[-1] - curve(bp)[-1])
    q = lambda v: {"5%": float(np.percentile(v, 5)), "중앙": float(np.median(v)), "95%": float(np.percentile(v, 95)),
                   ">0 비율": float(np.mean(np.asarray(v) > 0))}
    return {"타이밍값": q(tv), "효율선초과": q(ex), "BH대비수익": q(ret)}


# ============================================================================ 시점 감사
def timing_audit(data):
    """t일 종가와 그 뒤 가격(시가는 t+1부터)을 흔들어도 (1) t일까지 선 상태·목표 비율이 같고 (2) t−1일까지 일수익이 같아야 한다."""
    rng = np.random.default_rng(7)
    syms = universe(data, "2026")[:40]
    worst_h, worst_r = 0, 0.0
    for lag in (1, 2):
        P = dict(CAND, lag=lag)
        for k in syms:
            s = data[k]
            a, z = bounds(s, *WIN["2026"])
            H1 = line_states(s, a, z, P)
            t1, c1 = targets(H1, P["u"])
            r1, _ = sim_path(s["o"], s["c"], a, t1, c1)
            for T in (a + 20, a + 80, a + 150):
                if T > z:
                    continue
                s2 = {"d": s["d"], "o": s["o"].copy(), "c": s["c"].copy(), "_sma": {}}
                s2["c"][T:] *= 1 + rng.normal(0, 0.08, len(s2["c"]) - T)
                s2["o"][T + 1:] *= 1 + rng.normal(0, 0.08, len(s2["o"]) - T - 1)
                H2 = line_states(s2, a, z, P)
                t2, c2 = targets(H2, P["u"])
                r2, _ = sim_path(s2["o"], s2["c"], a, t2, c2)
                worst_h = max(worst_h, int((H1[:T - a + 1] != H2[:T - a + 1]).any()))
                if T > a:
                    worst_r = max(worst_r, float(np.abs(r1[:T - a] - r2[:T - a]).max()))
    return {"t일까지_선상태_다른_경우": worst_h, "t−1일까지_일수익_최대차": worst_r, "점검종목": len(syms), "lag": [1, 2]}


def cmp_branch(base, shock, slip):
    """갈래 결과 파일(results/p3_LADDER.json, 코드는 import하지 않음)의 B|B3 값과 반올림 없이 비교."""
    p = RESULTS / "p3_LADDER.json"
    if not p.exists():
        return {"비교": "갈래 json 없음"}
    L = json.loads(p.read_text(encoding="utf-8"))
    out, worst, n = {}, 0.0, 0
    for src, mine in (("results", base), ("slip", slip)):
        theirs = L[src]["B|B3"]
        for w, d in theirs.items():
            if w == "AI쇼크":
                pairs = [(k, d[k], shock[k]) for k in ("수익", "낙폭", "BH수익", "고정노출_수익")] if src == "results" else []
            elif w in mine:
                pairs = [(k, d[k], mine[w][k]) for k in ("포트_수익", "포트_낙폭", "포트BH_수익", "효율선초과", "타이밍값", "평균노출",
                                                          "고정노출_수익", "수익_나은종목비율", "손절주기", "승률", "손익비", "기대값")
                         if k in d and k in mine[w]]
            else:
                pairs = []
            for k, a, b in pairs:
                if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
                    continue
                n += 1
                worst = max(worst, abs(a - b))
    return {"비교한_값": n, "최대_절대차": worst}


# ============================================================================ 이웃 목록
def neighbor_specs():
    sp = []
    for s0 in NEIGH_START:
        for g in NEIGH_GAP:
            sp.append((f"선 {s0}+{g}k", "grid", dict(CAND, lines=tuple(s0 + g * i for i in range(4)))))
    sp += [(f"단위 {int(u * 100)}%", "u", dict(CAND, u=u)) for u in NEIGH_U]
    sp += [(f"되사기 띠 {int(b * 100)}%", "bb", dict(CAND, bb=b)) for b in NEIGH_BB]
    sp += [(f"팔기 띠 {int(b * 100)}%", "bs", dict(CAND, bs=b)) for b in NEIGH_BS]
    sp += [(f"확인 {k}일", "confirm", dict(CAND, confirm=k)) for k in NEIGH_CONFIRM]
    sp += [(f"선 {n}개(50 간격)·단위 1/{n}", "nlines", dict(CAND, lines=tuple(50 * (i + 1) for i in range(n)), u=1.0 / n))
           for n in NEIGH_NLINES]
    return sp


# ============================================================================ 본 실행
def main():
    t0 = time.time()
    data = load()
    P = {"grid": {"CAND": CAND, "NEIGH_START": NEIGH_START, "NEIGH_GAP": NEIGH_GAP, "NEIGH_U": NEIGH_U, "NEIGH_BB": NEIGH_BB,
                  "NEIGH_BS": NEIGH_BS, "NEIGH_CONFIRM": NEIGH_CONFIRM, "NEIGH_NLINES": NEIGH_NLINES, "SLIP": SLIP,
                  "N_SEEDS": N_SEEDS, "N_NULL": N_NULL, "N_NULL_PLAC": N_NULL_PLAC, "N_BOOT": N_BOOT, "EFF_STEP": EFF_STEP,
                  "EXCLUDE": {k: sorted(v) for k, v in EXCLUDE.items()}}}

    # 1) 재계산
    base_res = {rk: run(data, rk, CAND) for rk in RUNS}
    base = all_stats(data, base_res)
    shock = shock_stats(base_res["2026"], base["2026"]["평균노출"])
    slip_res = {rk: run(data, rk, CAND, SLIP, SLIP) for rk in ("2026", "2025")}
    slip = all_stats(data, slip_res)
    P.update(base=base, shock=shock, slip=slip)
    rep = []
    for kind, w, key, v in CLAIM:
        mine = shock[key] if kind == "shock" else {"base": base, "slip": slip}[kind][w][key]
        tol = TOL_BY_KEY.get(key, TOL)
        rep.append({"구분": kind, "창": w, "지표": key, "주장": v, "재계산": mine, "차": mine - v,
                    "일치": bool(abs(mine - v) <= tol + 1e-9)})
    P["reproduce"] = rep
    P["vs_branch_json"] = cmp_branch(base, shock, slip)
    print(f"[1] 재계산 {time.time() - t0:.0f}s  불일치 {sum(not r['일치'] for r in rep)}/{len(rep)}", P["vs_branch_json"])
    for r in rep:
        if not r["일치"]:
            print("   불일치", r)

    # 2) 시점
    P["timing"] = timing_audit(data)
    lag_res = {rk: run(data, rk, dict(CAND, lag=2)) for rk in MAIN_RUNS}
    P["lag2"] = all_stats(data, lag_res)
    P["sell_slip"] = all_stats(data, {rk: run(data, rk, CAND, SLIP, 0.0) for rk in ("2026", "2025")})
    P["lag2_slip"] = all_stats(data, {rk: run(data, rk, dict(CAND, lag=2), SLIP, SLIP) for rk in ("2026", "2025")})
    P["shock_lag2"] = shock_stats(lag_res["2026"], P["lag2"]["2026"]["평균노출"])
    print(f"[2] 시점 {time.time() - t0:.0f}s", P["timing"])

    # 3) 이웃
    neigh = {}
    keep = ("포트_수익", "포트_낙폭", "효율선초과", "타이밍값", "평균노출", "손절주기", "승률", "손익비", "기대값")
    for name, kind, spec in neighbor_specs():
        rr = {rk: run(data, rk, spec) for rk in ("2026", "2025", "2022")}
        st = all_stats(data, rr)
        rs = {rk: run(data, rk, spec, SLIP, SLIP) for rk in ("2026", "2025")}
        ss_ = all_stats(data, rs, with_cyc=False)
        neigh[name] = {"kind": kind, "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
                       **{w: {k: st[w][k] for k in keep} for w in ("2026", "2026앞", "2026뒤", "2025", "2022")},
                       "slip": {w: {k: ss_[w][k] for k in ("포트_수익", "효율선초과", "타이밍값")} for w in ("2026앞", "2026뒤", "2025")}}
    P["neigh"] = neigh
    names = list(neigh)
    M3 = ("2026앞", "2026뒤", "2025")
    P["neigh_summary"] = {
        "개수": len(names),
        "효율선초과 앞·뒤·2025 모두>0": [n for n in names if all(neigh[n][w]["효율선초과"] > 0 for w in M3)],
        "타이밍값 앞·뒤·2025 모두>0": [n for n in names if all(neigh[n][w]["타이밍값"] > 0 for w in M3)],
        "미끄러짐 타이밍값 앞·뒤·2025 모두>0": [n for n in names if all(neigh[n]["slip"][w]["타이밍값"] > 0 for w in M3)],
        **{f"{w} 효율선초과>0": sum(1 for n in names if neigh[n][w]["효율선초과"] > 0) for w in M3},
        **{f"{w} 타이밍값>0": sum(1 for n in names if neigh[n][w]["타이밍값"] > 0) for w in M3},
        "뒤 BH수익 이김": sum(1 for n in names if neigh[n]["2026뒤"]["포트_수익"] > base["2026뒤"]["포트BH_수익"]),
        "뒤 주기기대값>0": sum(1 for n in names if np.isfinite(neigh[n]["2026뒤"]["기대값"]) and neigh[n]["2026뒤"]["기대값"] > 0),
        "앞 주기기대값>0": sum(1 for n in names if np.isfinite(neigh[n]["2026앞"]["기대값"]) and neigh[n]["2026앞"]["기대값"] > 0),
        "2025 주기기대값>0": sum(1 for n in names if np.isfinite(neigh[n]["2025"]["기대값"]) and neigh[n]["2025"]["기대값"] > 0),
    }
    print(f"[3] 이웃 {time.time() - t0:.0f}s", {k: (len(v) if isinstance(v, list) else v) for k, v in P["neigh_summary"].items()})

    # 4) 위약
    keys = ("포트_수익", "포트_낙폭", "효율선초과", "타이밍값", "평균노출")
    pl = {mode: {w: {k: [] for k in keys} for w in PL_WINS} for mode in ("P1", "P2")}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        for mode in ("P1", "P2"):
            for w in PL_WINS:
                rk = "2026" if w.startswith("2026") else w
                st = placebo_window(data, base_res[rk], w, mode, rng)
                for k in keys:
                    pl[mode][w][k].append(st[k])
    P["placebo"] = {mode: {w: {k: {"실제": base[w][k], "위약_중앙": float(np.median(v)), "위약_5%": float(np.percentile(v, 5)),
                                    "위약_95%": float(np.percentile(v, 95)), "실제가_이긴_비율": rank_vs(base[w][k], v)}
                                for k, v in d.items()} for w, d in pl[mode].items()} for mode in pl}
    print(f"[4] 위약 {time.time() - t0:.0f}s")
    gaps, rule0 = null_calibration(data)
    P["null"] = {}
    for w in PL_WINS:
        P["null"][w] = {}
        for k in gaps[w]:
            g = np.asarray(gaps[w][k], float)
            rv = np.asarray(rule0[w][k], float)
            real_gap = (base[w][k] - P["placebo"]["P1"][w][k]["위약_중앙"]) if k in P["placebo"]["P1"][w] else np.nan
            real = base[w][k]
            P["null"][w][k] = {"실제_규칙값": float(real), "귀무_규칙값_중앙": float(np.median(rv)),
                               "귀무_규칙값_5%": float(np.percentile(rv, 5)), "귀무_규칙값_95%": float(np.percentile(rv, 95)),
                               "귀무_규칙값이_실제보다_낮은_비율": float((rv < real).mean()),
                               "실제차_P1": float(real_gap), "귀무차_중앙": float(np.median(g)),
                               "귀무차_5%": float(np.percentile(g, 5)), "귀무차_95%": float(np.percentile(g, 95)),
                               "귀무차가_실제차보다_낮은_비율": float((g < real_gap).mean()) if np.isfinite(real_gap) else np.nan}
    print(f"[4b] 귀무 보정 {time.time() - t0:.0f}s")

    # 4c) 종목 재표본
    rng = np.random.default_rng(55)
    P["boot"] = {w: boot_ci(base_res["2026" if w.startswith("2026") else w], w, rng) for w in ("2026앞", "2026뒤", "2025")}
    print(f"[4c] 재표본 {time.time() - t0:.0f}s")

    # 5) 장세(맥락 해 포함, 2018·2019는 QUBT 제외)
    P["regimes"] = {w: {k: base[w][k] for k in ("포트_수익", "포트_낙폭", "포트BH_수익", "포트BH_낙폭", "효율선초과", "타이밍값",
                                                "종목내타이밍", "종목배분_사후", "평균노출", "손절주기", "주기_종목당", "승률", "손익비",
                                                "기대값", "종목수")}
                    for w in base}

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "p3_V_LADDER.json"
    p.write_text(json.dumps(P, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("저장", p, f"{time.time() - t0:.0f}s")
    write_md(json.loads(p.read_text(encoding="utf-8")))


# ============================================================================ 보고서
def pc(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p"


def pu(x, d=0):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def f2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def wr(w):
    return {"2026앞": "앞", "2026뒤": "뒤"}.get(w, w)


def write_md(P):
    B, S, N, PL, NU = P["base"], P["slip"], P["neigh"], P["placebo"], P["null"]
    NS = P["neigh_summary"]
    rep = P["reproduce"]
    L = []
    A = L.append
    bad = [r for r in rep if not r["일치"]]
    pct_keys = [r for r in rep if r["지표"] not in ("손절주기", "손익비", "주기_종목당")]
    worst = max(abs(r["차"]) for r in pct_keys)
    vb = P.get("vs_branch_json", {})
    tv = lambda w: NU[w]["타이밍값"]
    gp = lambda w: NU[w]["포트_수익"]
    names = list(N)
    M3 = ("2026앞", "2026뒤", "2025")
    rng_ = lambda w, k: (min(N[n][w][k] for n in names), max(N[n][w][k] for n in names))
    n_pl_lost = sum(1 for w in PL_WINS if PL["P1"][w]["포트_수익"]["실제가_이긴_비율"] == 0)
    gap_meds = [-gp(w)["귀무차_중앙"] for w in PL_WINS]
    gap_lo, gap_hi = min(gap_meds), max(gap_meds)

    A("# 반박 검증 V_LADDER: G1 'B3 50·100·150·200일선 하나 깰 때마다 25% 매도' (2026-10-11)\n")
    A(f"스크립트 `code/p3_V_LADDER.py`(stock_lab·tranche_lab·p3_LADDER를 import하지 않은 독립 엔진, 노출 비율 방식), 전 결과 "
      f"`results/p3_V_LADDER.json`. 대상: 2026 {B['2026']['종목수']}종목 · 2025 {B['2025']['종목수']} · 2022 {B['2022']['종목수']} · "
      f"2018 {B['2018']['종목수']}(QUBT 제외) AI 종목 동일가중 포트폴리오(매일 재조정). **판정: refuted(우위 없음)**\n")

    # ---------------------------------------------------------------- 결론
    A("## 결론 3줄\n")
    lag = P["lag2"]
    A(f"1. **숫자는 맞고 시점 누수도 없다.** 주장 {len(rep)}개 중 허용 오차(0.5%p) 밖 {len(bad)}개, 최대 차 {worst * 100:.2f}%p(주장 반올림 때문). "
      f"갈래 결과 파일과 반올림 없이 비교한 {vb.get('비교한_값', 0)}개 값의 최대 차는 {vb.get('최대_절대차', float('nan')):.1e}다. "
      f"t일 종가 이후 가격을 흔들어도 t일까지 선 상태·t−1일까지 수익이 같고, 하루 늦춰도 앞 {pc(lag['2026앞']['포트_수익'])}·뒤 {pc(lag['2026뒤']['포트_수익'])}로 거의 같다(느린 규칙이라 체결 하루 차이에 둔하다).")
    lo, hi = rng_("2026뒤", "효율선초과")
    A(f"2. **확인 구간에서 무너지고 이웃 {len(names) - 1}개도 모두 같다.** 후보의 뒤 효율선초과 {pp(B['2026뒤']['효율선초과'])}, "
      f"같은 평균 노출 고정 대비(타이밍값) {pp(B['2026뒤']['타이밍값'])}. 선 위치·간격·파는 몫·띠·확인 일수·선 수를 바꾼 {len(names)}개(후보 포함)는 "
      f"모두 앞·2025 효율선초과가 +이고 뒤는 −다({pp(lo)}~{pp(hi)}). 앞·뒤·2025 모두 + 인 것은 {len(NS['효율선초과 앞·뒤·2025 모두>0'])}/{len(names)}개, "
      f"타이밍값 기준도 {len(NS['타이밍값 앞·뒤·2025 모두>0'])}/{len(names)}개다. 미끄러짐 0.5%를 넣으면 뒤 {pc(S['2026뒤']['포트_수익'])}, 2025 타이밍값 {pp(S['2025']['타이밍값'])}로 더 나빠진다.")
    A(f"3. **'선을 깰 때 판 것'의 값은 앞 구간(고른 구간)에만 있었다.** 같은 평균 노출 고정 대비 타이밍값을 실력 0인 가짜 경로 {P['grid']['N_NULL']}개와 견주면 "
      f"앞 {pp(B['2026앞']['타이밍값'])}는 가짜 경로 범위(5~95% {pp(tv('2026앞')['귀무_규칙값_5%'])}~{pp(tv('2026앞')['귀무_규칙값_95%'])}) 위로 벗어났다. "
      f"그러나 뒤 {pp(B['2026뒤']['타이밍값'])}는 가짜 경로의 하위 {pu(tv('2026뒤')['귀무_규칙값이_실제보다_낮은_비율'])}, "
      f"2022 {pp(B['2022']['타이밍값'])}는 하위 {pu(tv('2022')['귀무_규칙값이_실제보다_낮은_비율'])}에 있다. "
      f"2025 {pp(B['2025']['타이밍값'])}와 2018 {pp(B['2018']['타이밍값'])}는 가짜 경로와 구별되지 않았다. "
      f"판 비율 경로를 날짜만 원형 이동한 위약은 {n_pl_lost}/{len(PL_WINS)}개 창에서 모든 시드가 실제보다 수익이 높았다. "
      f"다만 이 위약은 실력 0에서도 규칙을 {gap_lo * 100:.0f}~{gap_hi * 100:.0f}%p(창별 중앙) 이기는 구조적 치우침이 있다(아래 6절). "
      f"이 치우침을 빼면 앞만 우위(가짜 경로 {pu(gp('2026앞')['귀무차가_실제차보다_낮은_비율'])}보다 나음)였고, 뒤는 가짜 경로 {pu(gp('2026뒤')['귀무차가_실제차보다_낮은_비율'])}보다만 나았다.\n")

    # ---------------------------------------------------------------- 1. 재현
    A("## 1. 재현 (주장 → 독립 재계산)\n")
    A("| 창 | 수익 / 낙폭 | 매수·보유 | 효율선초과 | 타이밍값 | 주기 n · 승률 · 손익비 · 기대값 |\n|---|---|---|---|---|---|")
    cl = {(r["구분"], r["창"], r["지표"]): r for r in rep}

    def cv(kind, w, k, f):
        r = cl.get((kind, w, k))
        if r is None:
            return f({"base": B, "slip": S}[kind][w][k])
        return f"{f(r['주장'])} → {f(r['재계산'])}"
    nf = lambda x: f"{x:.0f}"
    for w in ("2026", "2026앞", "2026뒤", "2025"):
        A(f"| {wr(w)} | {cv('base', w, '포트_수익', pc)} / {cv('base', w, '포트_낙폭', pc)} | {cv('base', w, '포트BH_수익', pc)} | "
          f"{cv('base', w, '효율선초과', pp)} | {cv('base', w, '타이밍값', pp)} | {cv('base', w, '손절주기', nf)} · {cv('base', w, '승률', pu)} · "
          f"{cv('base', w, '손익비', f2)} · {cv('base', w, '기대값', pc)} |")
    A(f"| 2022(맥락) | {cv('base', '2022', '포트_수익', pc)} / {pc(B['2022']['포트_낙폭'])} | {cv('base', '2022', '포트BH_수익', pc)} | {pp(B['2022']['효율선초과'])} | {pp(B['2022']['타이밍값'])} | |")
    sh = P["shock"]
    A(f"| 7월 쇼크 6/22~7/29 | {cl[('shock', 'AI쇼크', '수익')]['주장'] * 100:+.1f}% → {pc(sh['수익'])} | {cl[('shock', 'AI쇼크', 'BH수익')]['주장'] * 100:+.1f}% → {pc(sh['BH수익'])} | "
      f"같은 비중({B['2026']['평균노출']:.2f}) 고정 {cl[('shock', 'AI쇼크', '고정노출_수익')]['주장'] * 100:+.1f}% → {pc(sh['고정노출_수익'])} | | |")
    A(f"| 미끄러짐 0.5% | 2026 {cv('slip', '2026', '포트_수익', pc)}, 앞 {cv('slip', '2026앞', '포트_수익', pc)}, 뒤 {cv('slip', '2026뒤', '포트_수익', pc)}, 2025 {cv('slip', '2025', '포트_수익', pc)} | | | 2025 {cv('slip', '2025', '타이밍값', pp)} | |\n")
    A(f"- 2026 평균 노출 {B['2026']['평균노출']:.3f}(주장 0.59), 주기 {B['2026']['주기_종목당']:.1f}회/종목(주장 19.6), 2026 주기 승률·손익비·기대값 "
      f"{pu(B['2026']['승률'])}·{f2(B['2026']['손익비'])}·{pc(B['2026']['기대값'])}(주장 25%·2.53·−0.7%). "
      f"수익이 매수·보유보다 나은 종목 비율 앞 {pu(B['2026앞']['수익_나은종목비율'])} · 뒤 {pu(B['2026뒤']['수익_나은종목비율'])} · 2025 {pu(B['2025']['수익_나은종목비율'])}. 깊은 재매수는 이 규칙에 없다(0%).")
    A("- 앞·뒤 주기는 그 창 끝까지 못 되산 몫을 그 창 끝 종가로 평가했다(앞 고르기에 뒤 가격이 섞이지 않음). 주기 하나의 비중이 모두 25%라 비중 가중 승률·기대값은 조각 수 기준과 같다.")
    A("- 교차 확인: 이웃 '되사기 띠 2%'는 갈래의 B8과 같은 규칙이다. 독립 엔진 값 앞 "
      f"{pc(N['되사기 띠 2%']['2026앞']['포트_수익'])} · 뒤 {pc(N['되사기 띠 2%']['2026뒤']['포트_수익'])} · 2025 {pc(N['되사기 띠 2%']['2025']['포트_수익'])}가 갈래 B8(+50.1% · −7.4% · +44.3%)과 같다.\n")

    # ---------------------------------------------------------------- 2. 격자
    A("## 2. 격자 (스크립트 맨 위 상수, 결과를 본 뒤 바꾸지 않음)\n")
    A("| 부 | 내용 | 개수 |\n|---|---|---|")
    A(f"| 이웃: 선 위치 × 간격 | 첫 선 {list(NEIGH_START)}(10 간격, 후보 50의 ±2칸) × 선 간격 {list(NEIGH_GAP)}(5 간격, 후보 50의 ±2칸), 선 4개 = 첫 선 + 간격 × 0·1·2·3. 최장 선 250일(창 앞 자료 252일 안) | 25 |")
    A(f"| 이웃: 파는 몫 | 선 하나당 {[int(u * 100) for u in NEIGH_U]}%(후보 25%의 ±2칸, 5%p 간격). 30·35%는 네 번째 선에서 보유 0으로 잘림 | 4 |")
    A(f"| 이웃: 되사기 띠 | 종가 > 선 × (1 + b), b = {[int(b * 100) for b in NEIGH_BB]}%(후보 0의 한쪽 2칸). 2%는 갈래 B8과 같음 | 2 |")
    A(f"| 이웃: 팔기 띠 | 종가 < 선 × (1 − b), b = {[int(b * 100) for b in NEIGH_BS]}% | 2 |")
    A(f"| 이웃: 확인 일수 | 연속 {list(NEIGH_CONFIRM)}일 종가(후보 1의 한쪽 2칸) | 2 |")
    A(f"| 이웃: 선 수 | {list(NEIGH_NLINES)}개(50일 간격), 파는 몫 = 1/선 수. 후보 4개의 ±2칸 중 6개(SMA300)는 창 앞 자료 252일을 넘어 뺌 | 3 |")
    A(f"| 시점 | 하루 늦춘 판(c[t−2]·SMA[t−2] → t 시가), 매도만 0.5%, 양쪽 0.5%, 늦춤 + 양쪽 0.5% | 4 |")
    A(f"| 위약 | P1 종목마다 판 비율 경로를 창 안에서 원형 이동(이동량 무작위), P2 모든 종목 같은 이동량, 각 시드 {P['grid']['N_SEEDS']}. P0 종목별 사후 평균 노출 고정, 같은 평균 노출 고정 | 2 × {P['grid']['N_SEEDS']} + 2 |")
    A(f"| 위약 귀무 보정 | 가짜 경로(창 + 앞 252일의 (밤사이, 장중) 수익 쌍을 iid 재표본, 모든 종목 같은 날짜 뽑기) {P['grid']['N_NULL']}개 × P1 시드 {P['grid']['N_NULL_PLAC']} | {P['grid']['N_NULL']} × {P['grid']['N_NULL_PLAC']} |")
    A(f"| 종목 재표본 | 종목을 복원 추출해 타이밍값·효율선초과 신뢰 구간 | {P['grid']['N_BOOT']} |\n")
    A(f"이웃 합계 {len(names)}개(후보 포함, 이웃 {len(names) - 1}개). 창: 2026 앞(1/2~5/29)·뒤(6/1~10/9)는 2026을 한 번 이어 돌린 곡선을 자른다. 2025는 보조, 2022·2018은 맥락이다.\n")

    # ---------------------------------------------------------------- 3. 변수 시점
    A("## 3. 변수 시점\n")
    A("| 변수 | 언제 아는가 | 쓰임 |\n|---|---|---|")
    A("| 종가 c[t−1], SMA50·100·150·200[t−1](t−1까지 배당 수정 종가) | t−1 종가 | 깬 선 수 k → t일 시가에 보유 1 − 0.25k로 맞춤 |")
    A("| 장중 손절가·지정가 | 없음(이 규칙은 장중 주문이 없다) | 감사 해당 없음 |")
    A("| 하루 늦춘 판 c[t−2], SMA[t−2] | t−2 종가 | t일 시가 |")
    A("| 효율선 노출 e(같은 낙폭의 고정 노출), 같은 평균 노출 | 사후(창 전체) | 비교 기준만, 매매 근거 아님 |")
    A("| 위약 P0·P1·P2의 종목별 노출 분포 | 사후(창 전체 규칙 결과) | 위약 쪽에 유리한 정보(6절 귀무 보정) |\n")
    ta = P["timing"]
    A(f"감사: {ta['점검종목']}종목 × 시점 3곳 × 판 2개(하루 전·이틀 전 종가)에서 t일 종가와 그 뒤 가격을 흔들었다. t일까지 선 상태가 달라진 경우 {ta['t일까지_선상태_다른_경우']}, "
      f"t−1일까지 일수익 최대 차 {ta['t−1일까지_일수익_최대차']:.1e}. 판단은 시가 직전 한 번이고 같은 날 종가를 쓰지 않는다.\n")

    # ---------------------------------------------------------------- 4. 후보와 시점 변형
    A("## 4. 후보와 시점 변형 (동일가중 포트 수익 / 최대 낙폭 (효율선초과 · 타이밍값))\n")
    A("| 판 | 2026 | 앞(고르기) | 뒤(확인) | 2025(보조) | 2022(맥락) | 2018(맥락) |\n|---|---|---|---|---|---|---|")
    cols = ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")
    A("| 매수·보유 | " + " | ".join(f"{pc(B[w]['포트BH_수익'])} / {pc(B[w]['포트BH_낙폭'])}" for w in cols) + " |")

    def row(name, d):
        cells = []
        for w in cols:
            if w in d:
                cells.append(f"{pc(d[w]['포트_수익'])} / {pc(d[w]['포트_낙폭'])} ({pp(d[w]['효율선초과'])} · {pp(d[w]['타이밍값'])})")
            else:
                cells.append("—")
        A(f"| {name} | " + " | ".join(cells) + " |")
    row("후보 B3", B)
    row("하루 늦춤", P["lag2"])
    row("매도만 0.5%", P["sell_slip"])
    row("양쪽 0.5%", S)
    row("늦춤 + 양쪽 0.5%", P["lag2_slip"])
    A("")
    shl = P["shock_lag2"]
    A(f"- 7월 쇼크(6/22~7/29): 후보 {pc(sh['수익'])}, 하루 늦춤 {pc(shl['수익'])}, 매수·보유 {pc(sh['BH수익'])}, 같은 비중({B['2026']['평균노출']:.2f}) 고정 {pc(sh['고정노출_수익'])}. "
      + ("쇼크 구간만 봐도 선을 깰 때마다 판 것이 그냥 그만큼 덜 들고 있기보다 더 잃었다." if sh["수익"] < sh["고정노출_수익"]
         else "쇼크 구간에서는 선을 깰 때마다 판 것이 그냥 그만큼 덜 들고 있기보다 덜 잃었다."))
    A(f"- 하루 늦춰도 결과가 거의 같다(앞 {pc(lag['2026앞']['포트_수익'])}, 뒤 {pc(lag['2026뒤']['포트_수익'])}, 2025 {pc(lag['2025']['포트_수익'])}). 체결 시점을 노린 착시는 아니다. "
      f"미끄러짐은 매매가 많아(2026 종목당 {B['2026']['주기_종목당']:.1f}주기, 선이 바뀐 날마다 비율 재조정) 앞 타이밍값 {pp(B['2026앞']['타이밍값'])} → {pp(S['2026앞']['타이밍값'])}, 2026 효율선초과 {pp(B['2026']['효율선초과'])} → {pp(S['2026']['효율선초과'])}로 깎인다.\n")

    # ---------------------------------------------------------------- 5. 이웃
    A("## 5. 이웃 — 조합을 한눈에\n")
    A("### 5a. 선 위치 × 간격: 효율선초과(같은 최대 낙폭의 고정 노출 대비, %p) 앞 / 뒤 / 2025\n")
    A("행 = 첫 선(일), 열 = 선 간격(일). 선 4개 = 첫 선, +간격, +2간격, +3간격. 각 선을 깰 때마다 25% 매도. 후보는 50 / 50(굵게).\n")
    A("| 첫 선 \\ 간격 | " + " | ".join(str(g) for g in NEIGH_GAP) + " |\n|---|" + "---|" * len(NEIGH_GAP))
    for s0 in NEIGH_START:
        cells = []
        for g in NEIGH_GAP:
            d = N[f"선 {s0}+{g}k"]
            txt = " / ".join(f"{d[w]['효율선초과'] * 100:+.1f}" for w in M3)
            cells.append(f"**{txt}**" if (s0, g) == (50, 50) else txt)
        A(f"| {s0} ({s0}·{s0}+간격·…) | " + " | ".join(cells) + " |")
    A("\n### 5b. 같은 격자: 타이밍값(같은 평균 노출 고정 대비, %p) 앞 / 뒤 / 2025\n")
    A("| 첫 선 \\ 간격 | " + " | ".join(str(g) for g in NEIGH_GAP) + " |\n|---|" + "---|" * len(NEIGH_GAP))
    for s0 in NEIGH_START:
        cells = []
        for g in NEIGH_GAP:
            d = N[f"선 {s0}+{g}k"]
            txt = " / ".join(f"{d[w]['타이밍값'] * 100:+.1f}" for w in M3)
            cells.append(f"**{txt}**" if (s0, g) == (50, 50) else txt)
        A(f"| {s0} | " + " | ".join(cells) + " |")
    A("\n### 5c. 한 가지씩 바꾼 이웃\n")
    A("| 이웃 | 2026 수익 / 낙폭 | 평균 노출 | 효율선초과 앞 / 뒤 / 2025 (%p) | 타이밍값 앞 / 뒤 / 2025 (%p) | 미끄러짐 0.5% 타이밍값 앞 / 뒤 / 2025 | 주기 승률·손익비·기대값 뒤 |\n|---|---|---|---|---|---|---|")
    order = ["선 50+50k"] + [n for n in names if N[n]["kind"] != "grid"]
    for n in order:
        d = N[n]
        lab = "후보(50·100·150·200, 25%)" if n == "선 50+50k" else n
        A(f"| {lab} | {pc(d['2026']['포트_수익'])} / {pc(d['2026']['포트_낙폭'])} | {d['2026']['평균노출']:.2f} | "
          + " / ".join(f"{d[w]['효율선초과'] * 100:+.1f}" for w in M3) + " | "
          + " / ".join(f"{d[w]['타이밍값'] * 100:+.1f}" for w in M3) + " | "
          + " / ".join(f"{d['slip'][w]['타이밍값'] * 100:+.1f}" for w in M3) + " | "
          + f"{pu(d['2026뒤']['승률'])} · {f2(d['2026뒤']['손익비'])} · {pc(d['2026뒤']['기대값'])} |")
    A("")
    A("- 파는 몫만 바꾼 이웃은 같은 날 같은 값에 팔고 되사므로 주기 승률·손익비·기대값이 후보와 똑같다. 달라지는 것은 그 손익이 포지션에 미치는 크기뿐이다.")
    A(f"- 요약({len(names)}개): 효율선초과 > 0 — 앞 {NS['2026앞 효율선초과>0']}개, 뒤 {NS['2026뒤 효율선초과>0']}개, 2025 {NS['2025 효율선초과>0']}개, "
      f"**앞·뒤·2025 모두 {len(NS['효율선초과 앞·뒤·2025 모두>0'])}개**. 타이밍값 > 0 — 앞 {NS['2026앞 타이밍값>0']}, 뒤 {NS['2026뒤 타이밍값>0']}, 2025 {NS['2025 타이밍값>0']}, 모두 {len(NS['타이밍값 앞·뒤·2025 모두>0'])}. "
      f"미끄러짐 0.5%에서 타이밍값 세 구간 모두 + {len(NS['미끄러짐 타이밍값 앞·뒤·2025 모두>0'])}개. 뒤 구간 수익이 매수·보유({pc(B['2026뒤']['포트BH_수익'])})보다 나은 이웃 {NS['뒤 BH수익 이김']}개.")
    lo2, hi2 = rng_("2026앞", "효율선초과")
    u15 = N["단위 15%"]
    A(f"- 앞 효율선초과는 이웃 전체가 {pp(lo2)}~{pp(hi2)}로 고르게 높다. 후보가 운 좋은 한 점이 아니라 이동평균 사다리 전체가 앞 장세"
      "(매수·보유 기준 1/28 고점 → 3/30 −12.4% 조정 뒤 4~5월 급등, 규칙 노출은 3월 말 0.35 → 5월 말 0.82)에 맞았다는 뜻이다. "
      "뒤(7월 쇼크 뒤 8~9월 반등, 규칙 노출 0.4~0.5에 머묾)의 실패도 이웃 전체에 똑같이 나타났다. "
      f"파는 몫을 15%로 줄이면 뒤 효율선초과 {pp(u15['2026뒤']['효율선초과'])}로 손실은 줄지만 앞 {pp(u15['2026앞']['효율선초과'])}로 이익도 같이 줄어, '덜 들기'의 크기만 바뀐다.\n")

    # ---------------------------------------------------------------- 6. 위약
    A("## 6. 위약 — '선을 깰 때 팔아서'인가, '평균적으로 덜 들어서'인가\n")
    A("판 비율 경로 = 날마다의 목표 보유 비율(1 − 0.25 × 깬 선 수). 위약은 이 경로를 그 종목·그 창 안에서 날짜만 원형으로 밀어(예: 37일 뒤로, 끝은 앞으로 돌림) 같은 종목에 다시 입힌다. "
      "평균 노출·노출 수준 분포·매매 횟수는 거의 같고, '언제 팔았나'만 가격과 떨어진다.\n")
    A("| 창 | 실제 수익 | 같은 평균 노출 고정 | P0 종목별 사후 노출 고정 | P1 종목마다 이동: 중앙 (5~95%) · 실제가 이긴 비율 | P2 모두 같은 이동: 중앙 (5~95%) · 이긴 비율 | 실제 최대 낙폭 vs P1 중앙 |\n|---|---|---|---|---|---|---|")
    for w in PL_WINS:
        p1, p2 = PL["P1"][w]["포트_수익"], PL["P2"][w]["포트_수익"]
        A(f"| {wr(w)} | {pc(B[w]['포트_수익'])} | {pc(B[w]['고정노출_수익'])} (노출 {B[w]['평균노출']:.2f}) | {pc(B[w]['P0_수익'])} | "
          f"{pc(p1['위약_중앙'])} ({pc(p1['위약_5%'])}~{pc(p1['위약_95%'])}) · {pu(p1['실제가_이긴_비율'])} | "
          f"{pc(p2['위약_중앙'])} ({pc(p2['위약_5%'])}~{pc(p2['위약_95%'])}) · {pu(p2['실제가_이긴_비율'])} | "
          f"{pc(B[w]['포트_낙폭'])} vs {pc(PL['P1'][w]['포트_낙폭']['위약_중앙'])} |")
    A("")
    A("**위약의 치우침과 보정.** 이동평균 규칙의 노출은 '지난 가격'으로 정해진다. 그 경로를 날짜만 옮기면 옮긴 날의 노출이 그 날 이후 가격을 반영한 값이 된다(미래를 본 노출). "
      "그래서 위약은 실력 0인 가짜 경로에서도 규칙을 이긴다. 가짜 경로(일 단위 iid 재표본, 추세·되돌림 예측력 0)에서 같은 규칙과 같은 위약을 돌려 이 치우침을 쟀다.\n")
    A("| 창 | 타이밍값 실제 | 실력 0 가짜 경로의 타이밍값: 중앙 (5~95%) | 가짜가 실제보다 낮은 비율 | 규칙 − P1 실제 | 실력 0에서 규칙 − P1: 중앙 (5~95%) | 가짜가 실제보다 낮은 비율 |\n|---|---|---|---|---|---|---|")
    for w in PL_WINS:
        t, g = tv(w), gp(w)
        A(f"| {wr(w)} | {pp(t['실제_규칙값'])} | {pp(t['귀무_규칙값_중앙'])} ({pp(t['귀무_규칙값_5%'])}~{pp(t['귀무_규칙값_95%'])}) | {pu(t['귀무_규칙값이_실제보다_낮은_비율'])} | "
          f"{pp(g['실제차_P1'])} | {pp(g['귀무차_중앙'])} ({pp(g['귀무차_5%'])}~{pp(g['귀무차_95%'])}) | {pu(g['귀무차가_실제차보다_낮은_비율'])} |")
    A("")
    A("- 읽는 법: '가짜가 실제보다 낮은 비율'이 95% 이상이면 실력 0으로는 나오기 힘든 좋은 타이밍, 5% 이하면 실력 0보다 나쁜 타이밍이다. 50% 근처면 구별되지 않는다.")
    dd_win = {w: PL["P1"][w]["포트_낙폭"]["실제가_이긴_비율"] for w in PL_WINS}
    A("- 낙폭만 보면 실제가 모든 P1 위약보다 작았던 창: " + ", ".join(wr(w) for w in PL_WINS if dd_win[w] == 1.0)
      + "; 모든 위약보다 컸던 창: " + ", ".join(wr(w) for w in PL_WINS if dd_win[w] == 0.0)
      + ". P2(포트 노출 경로를 통째로 옮김)는 시장 전체의 오르내림과 겹치는 위치에 따라 결과가 크게 흔들려 범위가 넓다(귀무 보정은 P1만 했다).")
    A(f"- 타이밍값 자체는 실력 0에서 중앙이 0 근처({pp(tv('2026앞')['귀무_규칙값_중앙'])}~{pp(tv('2025')['귀무_규칙값_중앙'])})라 치우침이 작다. 그래서 '선을 깰 때 판 값'은 타이밍값으로 읽는 것이 가장 공정하다. "
      f"앞 {pp(B['2026앞']['타이밍값'])}는 가짜 경로 {P['grid']['N_NULL']}개 중 {pu(tv('2026앞')['귀무_규칙값이_실제보다_낮은_비율'])}보다 높았다. "
      f"뒤 {pp(B['2026뒤']['타이밍값'])}({pu(tv('2026뒤')['귀무_규칙값이_실제보다_낮은_비율'])})·2022 {pp(B['2022']['타이밍값'])}({pu(tv('2022')['귀무_규칙값이_실제보다_낮은_비율'])})는 아래쪽 꼬리, "
      f"2025({pu(tv('2025')['귀무_규칙값이_실제보다_낮은_비율'])})·2018({pu(tv('2018')['귀무_규칙값이_실제보다_낮은_비율'])})은 가운데다.")
    bt = P["boot"]
    A(f"- 종목 재표본({P['grid']['N_BOOT']}회) 타이밍값 5~95%: 앞 {pp(bt['2026앞']['타이밍값']['5%'])}~{pp(bt['2026앞']['타이밍값']['95%'])}, "
      f"뒤 {pp(bt['2026뒤']['타이밍값']['5%'])}~{pp(bt['2026뒤']['타이밍값']['95%'])}, 2025 {pp(bt['2025']['타이밍값']['5%'])}~{pp(bt['2025']['타이밍값']['95%'])}. "
      "종목을 다시 뽑아도 앞은 +, 뒤는 −로 부호가 그대로라 몇 종목에 기댄 결과가 아니다(종목 재표본은 날짜 불확실성을 담지 않는다).")
    A(f"- 분해(설명용, 둘 다 사후 노출을 쓴다): 타이밍값 = 종목 안 타이밍(규칙 − P0) + 종목 배분(P0 − 같은 평균 노출). 앞 {pp(B['2026앞']['종목내타이밍'])} + {pp(B['2026앞']['종목배분_사후'])}, "
      f"뒤 {pp(B['2026뒤']['종목내타이밍'])} + {pp(B['2026뒤']['종목배분_사후'])}, 2025 {pp(B['2025']['종목내타이밍'])} + {pp(B['2025']['종목배분_사후'])}. "
      f"실력 0에서도 종목 안 타이밍은 −(앞 중앙 {pp(NU['2026앞']['종목내타이밍']['귀무_규칙값_중앙'])}), 종목 배분은 +(앞 중앙 {pp(NU['2026앞']['종목배분_사후']['귀무_규칙값_중앙'])})로 기계적으로 갈린다. "
      f"그 기계적 몫을 빼고 보면 앞은 둘 다 가짜 경로보다 컸고(종목 안 {pu(NU['2026앞']['종목내타이밍']['귀무_규칙값이_실제보다_낮은_비율'])}, 배분 {pu(NU['2026앞']['종목배분_사후']['귀무_규칙값이_실제보다_낮은_비율'])}보다 나음), "
      f"뒤의 종목 안 타이밍 {pp(B['2026뒤']['종목내타이밍'])}는 가짜 경로 {P['grid']['N_NULL']}개 중 실제보다 나쁜 것이 {pu(NU['2026뒤']['종목내타이밍']['귀무_규칙값이_실제보다_낮은_비율'])}였다.\n")

    # ---------------------------------------------------------------- 7. 장세
    A("## 7. 장세별 — 효율선초과·타이밍값·주기 승률·손익비·기대값\n")
    A("주기 = 선 하나의 25% 몫을 판 뒤 되살 때까지. 주기 손익 = 판 값/되산 값 − 1(+면 싸게 되산 것). 손익분기 손익비 = (1 − 승률)/승률.\n")
    A("| 창 | 매수·보유 | 후보 수익 / 낙폭 | 평균 노출 | 효율선초과 | 타이밍값 (실력 0 대비 하위 %) | 주기 n (종목당) | 승률 | 손익비 | 손익분기 손익비 | 주기 기대값 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for w in ("2026앞", "2026뒤", "2025", "2022", "2018"):
        d = B[w]
        be = (1 - d["승률"]) / d["승률"] if d["승률"] and np.isfinite(d["승률"]) else float("nan")
        A(f"| {wr(w)} | {pc(d['포트BH_수익'])} / {pc(d['포트BH_낙폭'])} | {pc(d['포트_수익'])} / {pc(d['포트_낙폭'])} | {d['평균노출']:.2f} | {pp(d['효율선초과'])} | "
          f"{pp(d['타이밍값'])} ({pu(tv(w)['귀무_규칙값이_실제보다_낮은_비율'])}) | {d['손절주기']} ({d['주기_종목당']:.1f}) | {pu(d['승률'])} | {f2(d['손익비'])} | {f2(be)} | {pc(d['기대값'])} |")
    A("")
    ctx = [y for y in ("2019", "2020", "2021", "2023", "2024")]
    A("- 맥락(장세 전환 위험): 타이밍값 " + ", ".join(f"{y} {pp(B[y]['타이밍값'])}" for y in ctx)
      + "; 효율선초과 " + ", ".join(f"{y} {pp(B[y]['효율선초과'])}" for y in ctx)
      + f". 다른 다섯 해 중 같은 평균 노출 고정보다 나은 해 {sum(1 for y in ctx if B[y]['타이밍값'] > 0)}개(2019는 QUBT 제외).")
    A(f"- 승률·손익비와 포트 결과가 엇갈린다. 2022는 주기 승률 {pu(B['2022']['승률'])}·손익비 {f2(B['2022']['손익비'])}·기대값 {pc(B['2022']['기대값'])}로 '승률은 낮아도 손익비로 이기는' 모양이었지만 타이밍값은 {pp(B['2022']['타이밍값'])}다. "
      f"앞은 반대로 기대값 {pc(B['2026앞']['기대값'])}인데 타이밍값 {pp(B['2026앞']['타이밍값'])}다. 주기 통계는 '나가 있던 동안'만 재고, 다시 다 들고 있던 동안(약세장의 반등 고점, 강세장의 상승)을 재지 않는다. "
      "일부 매도 규칙의 값은 주기 승률·손익비가 아니라 같은 평균 노출 대비 포트 수익으로 판단해야 한다.\n")

    # ---------------------------------------------------------------- 8. 판정
    A("## 8. 판정: refuted(우위 없음)\n")
    A("| 질문 | 답 |\n|---|---|")
    A(f"| 주장 숫자 재현 | 예. {len(rep)}/{len(rep)} 허용 오차 안, 갈래 파일과 차 {vb.get('최대_절대차', float('nan')):.0e} |")
    A("| 시점 누수 | 없음. 하루 늦춰도 결과가 거의 같다 |")
    A(f"| 확인 구간(뒤) | 실패. 효율선초과 {pp(B['2026뒤']['효율선초과'])}, 타이밍값 {pp(B['2026뒤']['타이밍값'])}, 미끄러짐 0.5% {pc(S['2026뒤']['포트_수익'])} |")
    A(f"| 이웃 | 앞·뒤·2025 효율선초과 모두 + {len(NS['효율선초과 앞·뒤·2025 모두>0'])}/{len(names)}, 뒤 + {NS['2026뒤 효율선초과>0']}/{len(names)} |")
    A(f"| 위약(실력 0 보정) | 앞만 실력 0보다 나음. 뒤 하위 {pu(tv('2026뒤')['귀무_규칙값이_실제보다_낮은_비율'])}, 2022 하위 {pu(tv('2022')['귀무_규칙값이_실제보다_낮은_비율'])}, 2025·2018 구별 안 됨 |")
    A(f"| 2025(보조) | 수익은 같은 평균 노출 고정과 비슷(타이밍값 {pp(B['2025']['타이밍값'])})하고 낙폭은 작았다({pc(B['2025']['포트_낙폭'])} vs 고정 {pc(B['2025']['고정노출_낙폭'])}) → 효율선초과 {pp(B['2025']['효율선초과'])}. "
      f"실력 0 가짜 경로 20개 중 {pu(NU['2025']['효율선초과']['귀무_규칙값이_실제보다_낮은_비율'])}보다 높아 95% 기준에 못 미치고, 미끄러짐 0.5%면 타이밍값 {pp(S['2025']['타이밍값'])}·효율선초과 {pp(S['2025']['효율선초과'])} |\n")
    A("앞 구간의 큰 우위(효율선초과 +20.4%p, 타이밍값 +14.6%p)는 계산이 맞고 실력 0 범위도 벗어난다. 그러나 그것은 이동평균 사다리 전체가 공유한 앞 장세의 성질이었다. "
      "같은 규칙 가족 38개가 뒤에서 모두 졌고, 2022에서도 실력 0보다 나빴다. 그래서 '고른 구간에서 이겼다'는 사실은 다음 구간에 대한 정보가 되지 못했다. 갈래의 '기각' 결론이 맞다.\n")

    # ---------------------------------------------------------------- 9. 선택편향·한계
    A("## 9. 선택편향·한계\n")
    A("- 후보는 136개 중 앞 효율선초과 1등이다. 앞 수치는 위로 치우쳐 있다. 이 검증의 이웃·위약·귀무 설정은 스크립트 맨 위에 먼저 적었고, 결과를 본 뒤 바꾸지 않았다.")
    A("- 원형 이동 위약(P1·P2)은 지시대로 했지만 미래를 본 노출이 생기는 구조적 치우침이 있다. 그래서 결론은 iid 가짜 경로로 보정한 값과 타이밍값(같은 평균 노출 고정 대비)에 기댄다. "
      "iid 가짜 경로는 변동성 뭉침도 지우므로, 귀무 범위가 실제 장세보다 좁거나 넓을 수 있다(가짜 경로 20개라 꼬리 비율은 5%p 단위).")
    A("- '같은 평균 노출'과 '같은 낙폭의 고정 노출'은 창이 끝난 뒤에야 아는 값이다. 비교 기준으로만 썼다.")
    A("- 2026 뒤는 매수·보유가 마이너스라 효율선의 최선이 현금(노출 0)이다. 그래서 뒤 효율선초과는 사실상 절대 수익이다. 타이밍값은 이 문제가 없다.")
    A("- 생존편향: 종목 목록이 2026-10 기준(+TIME 보유 이력)이다. 사라진 종목에서 선 이탈 매도가 막아 줬을 손실이 빠져 있어 이 규칙에 불리한 쪽이다. 2018·2019는 QUBT를 뺐다.")
    A("- 동일가중 종목 포트폴리오(매일 재조정), 수수료·현금 이자 0, 배당 수정 일봉. 데스크 7종목 규칙 위 결과가 아니다. 매수 추천·주문 수량이 아니다.")
    L.append("")
    (RESULTS / "p3_V_LADDER.md").write_text("\n".join(L), encoding="utf-8")
    print("md 저장", RESULTS / "p3_V_LADDER.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "md":
        write_md(json.loads((RESULTS / "p3_V_LADDER.json").read_text(encoding="utf-8")))
    else:
        main()
