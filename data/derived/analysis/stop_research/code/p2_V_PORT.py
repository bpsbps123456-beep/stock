"""p2_V_PORT: 갈래 PORT 대표 후보 반박 검증 (2026-10-10, claude).

후보(G1=G3): 7종목 그림자(모델 종가 평가) < SMA20(t 포함) → t+1 시가 전량 현금, 현금일 때 그림자 > SMA20 → t+1 시가 전량 복귀.
  장전: 노출 1이고 장전일 때만 이탈. 노출 1인 동안 종가 ≥ 이탈선이면 장전. 복귀일 종가 ≥ 이탈선이면 바로 장전.
  2026은 보유 기간 경로 끝 120일을 1/2 종가에 맞춰 붙여 예열. 2025는 2023-05-22부터 이어 돌린 경로를 자름.

독립 재계산: harness·stock_lab·p2_PORT를 import하지 않는다.
  cache/base_*.json의 일별 비중(weights)·종목(ticker)과 cache/ohlc.parquet의 시가·종가만으로 7종목 시가 경로와 종가 평가(그림자)를
  다시 만들고, 규칙·창 지표·손절 주기·효율선을 새 코드로 잰다.
장세 대리(7종목 경로가 없는 2018·2022 등): cache/p2_long.parquet(배당 수정)로 만든 SMH·QQQ·AI 동일가중·AI 모멘텀 7종목에 같은 규칙.
실행: cd code && PYTHONIOENCODING=utf-8 py p2_V_PORT.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p2_V_PORT"

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않는다)
CAND = {"n_exit": 20, "n_re": 20, "out_e": 0.0, "lag": 1, "band": 0.0}
WARM_N = 120                       # 2026 그림자 예열 일수(후보 정의 그대로)
SLIP = 0.005                       # 미끄러짐 민감도(노출 1단위 회전당)
# 이웃: 각 모수의 앞뒤 2칸. 이탈 SMA × 복귀 SMA 5×5(가운데 = 후보) + 이탈 노출 2 + 체결 지연 2 + 이탈·복귀 띠 2 = 31
NB_EXIT_N = [10, 15, 20, 25, 30]
NB_RE_N = [10, 15, 20, 25, 30]
NB_OUT_E = [0.25, 0.5]             # 0은 아래쪽 경계라 위쪽 2칸
NB_LAG = [2, 3]                    # 0일(같은 종가 정보로 같은 날 거래)은 불가능 → 위쪽 2칸. 0일은 진단용으로만 따로 잰다
NB_BAND = [0.005, 0.01]            # 이탈: 종가 < SMA×(1−h), 복귀: 종가 > SMA×(1+h). 0은 아래쪽 경계
PLACEBO_SEEDS = 20                 # 위약 주 판정
PLACEBO_EXTRA = 500                # 보조(순위 안정성 확인)
PROXIES = ["SMH", "QQQ", "EW_AI", "MOM7_AI"]
PROXY_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
MOM_LOOKBACK, MOM_N, MOM_BUFFER, MOM_MIN_HIST = 120, 7, 5, 252
RATIO_CLIP = (0.2, 5.0)            # 대리 동일가중의 하루 가격비 이상치 자르기(자료 오류 대비, 건수 보고)
ETFS = {"QQQ", "SPY", "SOXX", "SMH", "IWM", "RSP", "^VIX", "^VIX3M"}

W26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
WH = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"), "2023(5/22~)": ("2023-05-22", "2023-12-31"),
      "관세": ("2025-02-18", "2025-04-04"), "24여름": ("2024-06-18", "2024-08-05")}

# PORT가 주장한 숫자(p2_PORT.md·json, 소수)
CLAIMS = {
    ("2026", "전체", "수익"): 1.380, ("2026", "전체", "낙폭"): -0.235, ("2026", "전체", "기본_수익"): 2.237,
    ("2026", "전체", "기본_낙폭"): -0.203,
    ("2026", "앞", "수익"): 1.733, ("2026", "앞", "낙폭"): -0.142, ("2026", "앞", "효율선초과"): 0.371,
    ("2026", "뒤", "수익"): -0.129, ("2026", "뒤", "낙폭"): -0.235, ("2026", "뒤", "효율선초과"): -0.175,
    ("2026", "뒤", "기본_수익"): 0.046,
    ("2026", "AI쇼크", "수익"): -0.152, ("2026", "AI쇼크", "기본_수익"): -0.187,
    ("hold", "2025", "효율선초과"): -0.356, ("hold", "2025", "수익"): 0.004, ("hold", "2025", "낙폭"): -0.350,
    ("hold", "2025", "기본_수익"): 0.422, ("hold", "2025", "기본_낙폭"): -0.471,
    ("2026", "뒤", "주기"): 10, ("2026", "뒤", "승률"): 0.10, ("2026", "뒤", "평균이익"): 0.001,
    ("2026", "뒤", "평균손실"): -0.021, ("2026", "뒤", "손익비"): 0.07, ("2026", "뒤", "기대값"): -0.0186,
    ("hold", "2025", "주기"): 15, ("hold", "2025", "승률"): 0.27, ("hold", "2025", "손익비"): 1.85,
    ("hold", "2025", "기대값"): -0.0125,
}
CLAIMS_SLIP = {("2026", "전체", "수익"): 1.090, ("2026", "뒤", "수익"): -0.204}
CLAIMS_LAG2 = {("2026", "앞", "효율선초과"): -0.250}
TOL = 0.005                        # 0.5%p


# ============================================================================ 자료: 7종목 경로를 비중·시가·종가로 다시 만든다
def ohlc_wide():
    o = pd.read_parquet(CACHE / "ohlc.parquet")
    o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
    o = o.drop_duplicates(["date", "sym"], keep="last")
    return o.pivot(index="date", columns="sym", values="Open"), o.pivot(index="date", columns="sym", values="Close")


def rebuild(name, O, C, bump_after: int | None = None, rng=None):
    """base json의 일별 비중(그날 시가 거래 뒤)으로 시가 경로 vo[k]와 종가 평가 vc[k]를 계산한다.
    bump_after: 그 날짜 index 뒤 가격을 흔든다(미래 교란 점검용)."""
    b = json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
    days, W, tk = b["days"], b["weights"], b["ticker"]
    syms = sorted(set(tk.values()))
    Od = O.reindex(days)[syms].to_numpy(float).copy()
    Cd = C.reindex(days)[syms].to_numpy(float).copy()
    if bump_after is not None:
        n = len(days) - bump_after - 1
        Od[bump_after + 1:] *= 1 + rng.normal(0, 0.05, (n, len(syms)))
        Cd[bump_after + 1:] *= 1 + rng.normal(0, 0.05, (n, len(syms)))
    col = {s: i for i, s in enumerate(syms)}
    N = len(days)
    vo, vc = np.ones(N), np.ones(N)
    for k in range(N):
        w = W[k]
        cash = 1.0 - sum(w.values())
        sc, so = cash, cash
        for key, wt in w.items():
            j = col[tk[key]]
            o0, c0 = Od[k, j], Cd[k, j]
            sc += wt * (c0 / o0 if np.isfinite(o0) and np.isfinite(c0) and o0 > 0 else 1.0)
            if k + 1 < N:
                o1 = Od[k + 1, j]
                so += wt * (o1 / o0 if np.isfinite(o0) and np.isfinite(o1) and o0 > 0 else 1.0)
        vc[k] = vo[k] * sc
        if k + 1 < N:
            vo[k + 1] = vo[k] * so
    return {"days": days, "vo": vo, "vc": vc, "json_open": np.array(b["open"]), "json_close": np.array(b["close"])}


# ============================================================================ 규칙(새로 짠 상태 기계)
def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def rule_close(vc_ext, n_pre, n_exit, n_re, out_e, band):
    """종가 t의 상태(노출) 배열. vc_ext = 예열 n_pre일 + 본 구간. 지표는 t 포함 trailing."""
    sx, sr = sma(vc_ext, n_exit)[n_pre:], sma(vc_ext, n_re)[n_pre:]
    c = vc_ext[n_pre:]
    inn, armed = True, True
    ec = np.ones(len(c))
    for i in range(len(c)):
        if inn:
            if np.isfinite(sx[i]):
                if c[i] >= sx[i]:
                    armed = True
                if armed and c[i] < sx[i] * (1 - band):
                    inn, armed = False, False
        else:
            if np.isfinite(sr[i]) and c[i] > sr[i] * (1 + band):
                inn = True
                armed = bool(np.isfinite(sx[i]) and c[i] >= sx[i])
        ec[i] = 1.0 if inn else out_e
    return ec


def to_open(ec, lag):
    """종가 t 상태 → 시가 t+lag 노출. 시작일 시가는 1(보유로 시작)."""
    N = len(ec)
    e = np.ones(N)
    if lag == 0:                      # 진단 전용(불가능한 체결: 같은 날 종가 정보로 같은 날 시가)
        return ec.copy()
    e[lag:] = ec[:N - lag]
    return e


def returns(vo, e, slip_x=0.0, slip_r=0.0):
    """r[k] = 시가 k → 시가 k+1 수익(노출 e[k]). 회전 비용은 k일 시가에 낸다. 이탈 미끄러짐 slip_x, 복귀 slip_r(노출 1단위당)."""
    rb = vo[1:] / vo[:-1] - 1
    prev = np.concatenate([[1.0], e[:-1]])
    dn = np.maximum(prev - e, 0.0)[:-1]
    up = np.maximum(e - prev, 0.0)[:-1]
    return e[:-1] * rb - slip_x * dn - slip_r * up, rb


def cycles(days, vo, vc, e, slip_x=0.0, slip_r=0.0):
    out, xi = [], None
    for k in range(1, len(e)):
        if xi is None and e[k - 1] >= 0.999 and e[k] < 0.999:
            xi = k
        elif xi is not None and e[k] >= 0.999 and e[k - 1] < 0.999:
            ex, re_ = vo[xi] * (1 - slip_x), vo[k] * (1 + slip_r)
            out.append({"exit_day": days[xi], "re_day": days[k], "gain": float(ex / re_ - 1), "days_out": k - xi})
            xi = None
    if xi is not None:
        ex = vo[xi] * (1 - slip_x)
        out.append({"exit_day": days[xi], "re_day": None, "gain": float(ex / vc[-1] - 1), "days_out": len(e) - 1 - xi})
    return out


def mdd(cur):
    return float((cur / np.maximum.accumulate(cur) - 1).min())


EGRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)


def eff_excess(rule_cur, rb):
    """같은 창 고정 노출(0~1, 0.005 간격) 경로 가운데 낙폭이 규칙보다 깊지 않은 것의 최대 수익을 뺀다."""
    tgt = mdd(rule_cur)
    best, be = -1.0, 0.0
    for e in EGRID:
        cur = np.concatenate([[1.0], np.cumprod(1 + e * rb)])
        if mdd(cur) >= tgt - 1e-12 and cur[-1] - 1 > best:
            best, be = cur[-1] - 1, float(e)
    return float(rule_cur[-1] - 1 - best), be


def cyc_stats(g):
    g = np.asarray(g, float)
    win, loss = g[g > 0], g[g <= 0]
    return {"주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
            "평균이익": float(win.mean()) if len(win) else float("nan"),
            "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "기대값": float(g.mean()) if len(g) else float("nan")}


def wstats(days, r, rb, e, cyc, lo, hi):
    d = np.asarray(days[:-1])
    m = (d >= lo) & (d <= hi)
    x, b = r[m], rb[m]
    cur, bcur = np.concatenate([[1.0], np.cumprod(1 + x)]), np.concatenate([[1.0], np.cumprod(1 + b)])
    exc, e_eq = eff_excess(cur, b)
    out = {"수익": float(cur[-1] - 1), "낙폭": mdd(cur), "기본_수익": float(bcur[-1] - 1), "기본_낙폭": mdd(bcur),
           "효율선초과": exc, "효율선_e": e_eq, "평균노출": float(e[:-1][m].mean()) if m.any() else float("nan"),
           "칼마": float((cur[-1] - 1) / abs(mdd(cur))) if mdd(cur) < -1e-9 else float("inf")}
    out.update(cyc_stats([c["gain"] for c in cyc if lo <= c["exit_day"] <= hi]))
    return out


def run_path(P, sp, n_pre=0, vc_ext=None, slip_x=0.0, slip_r=0.0):
    days, vo, vc = P["days"], P["vo"], P["vc"]
    vc_ext = vc if vc_ext is None else vc_ext
    ec = rule_close(vc_ext, n_pre, sp["n_exit"], sp["n_re"], sp["out_e"], sp["band"])
    e = to_open(ec, sp["lag"])
    r, rb = returns(vo, e, slip_x, slip_r)
    cyc = cycles(days, vo, vc, e, slip_x, slip_r)
    return {"e": e, "ec": ec, "r": r, "rb": rb, "cycles": cyc}


def measure7(P26, PH, ext26, sp, slip_x=0.0, slip_r=0.0):
    a = run_path(P26, sp, WARM_N, ext26, slip_x, slip_r)
    h = run_path(PH, sp, 0, None, slip_x, slip_r)
    res = {"2026": {w: wstats(P26["days"], a["r"], a["rb"], a["e"], a["cycles"], lo, hi) for w, (lo, hi) in W26.items()},
           "hold": {w: wstats(PH["days"], h["r"], h["rb"], h["e"], h["cycles"], lo, hi) for w, (lo, hi) in WH.items()}}
    return res, a, h


# ============================================================================ 위약
def spells(ew):
    out, s = [], None
    for i, v in enumerate(ew):
        if v < 0.999 and s is None:
            s = i
        elif v >= 0.999 and s is not None:
            out.append((s, i - s))
            s = None
    if s is not None:
        out.append((s, len(ew) - s))
    return out


def window_measure(idx, ew, vo, rb):
    """창 안 노출 ew(창 index 기준)로 수익·낙폭·효율선초과·주기 기대값. 주기 = 시가 이탈 → 시가 복귀(창 끝에 걸리면 창 끝 다음 시가)."""
    x = ew * rb[idx]
    cur = np.concatenate([[1.0], np.cumprod(1 + x)])
    exc, _ = eff_excess(cur, rb[idx])
    g = []
    for s, d in spells(ew):
        k0 = idx[s]
        k1 = idx[s] + d if s + d < len(idx) else idx[-1] + 1
        g.append(vo[k0] / vo[k1] - 1)
    cs = cyc_stats(g)
    return {"수익": float(cur[-1] - 1), "낙폭": mdd(cur), "효율선초과": exc, "기대값": cs["기대값"], "주기": cs["주기"]}


def place_random(n, durs, rng):
    """길이 n 창에 길이 durs인 빠짐 구간을 겹치지 않게(사이에 보유 ≥ 1일) 무작위 배치(별·막대 균등)."""
    m = len(durs)
    if m == 0:
        return np.ones(n)
    free = n - sum(durs) - (m - 1)
    if free < 0:
        raise ValueError("배치 불가")
    pos = np.sort(rng.choice(free + m, size=m, replace=False))
    gaps = np.diff(np.concatenate([[-1], pos])) - 1          # 각 구간 앞 간격(0 이상)
    ew = np.ones(n)
    cur = 0
    for j, (gp, d) in enumerate(zip(gaps, durs)):
        cur += gp + (1 if j > 0 else 0)
        ew[cur:cur + d] = 0.0
        cur += d
    return ew


def placebo(P, e, lo, hi, pool, seeds, base_seed=0):
    days, vo = P["days"], P["vo"]
    rb = vo[1:] / vo[:-1] - 1
    d = np.asarray(days[:-1])
    idx = np.where((d >= lo) & (d <= hi))[0]
    ew = (e[:-1][idx] >= 0.999).astype(float)
    act = window_measure(idx, ew, vo, rb)
    sp = spells(ew)
    durs = [x[1] for x in sp]
    A, B = [], []
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        dd = [int(x) for x in rng.permutation(durs)] if durs else []
        A.append(window_measure(idx, place_random(len(idx), dd, rng), vo, rb))
        # B: 실제 이탈일 고정, 빠져 있던 기간은 후보 전체 빠짐 기간 분포에서 복원 추출(다음 이탈 전 1일은 보유)
        ewb = np.ones(len(idx))
        starts = [x[0] for x in sp]
        for j, st in enumerate(starts):
            dur = int(rng.choice(pool))
            nxt = starts[j + 1] - 1 if j + 1 < len(starts) else len(idx)
            ewb[st:min(st + dur, nxt)] = 0.0
        B.append(window_measure(idx, ewb, vo, rb))

    def rank(L, key):
        v = np.array([x[key] for x in L if np.isfinite(x[key])])
        a = act[key]
        if not len(v) or not np.isfinite(a):
            return None
        return {"실제": a, "위약_중앙": float(np.median(v)), "위약_이상비율": float((v >= a - 1e-12).mean()),
                "p": float((1 + (v >= a - 1e-12).sum()) / (1 + len(v)))}
    return {"빠짐구간": len(sp), "빠진날": int(sum(durs)), "창일수": int(len(idx)), "실제": act,
            "A": {k: rank(A, k) for k in ("효율선초과", "수익", "낙폭", "기대값")},
            "B": {k: rank(B, k) for k in ("효율선초과", "수익", "낙폭", "기대값")}}


# ============================================================================ 장세 대리(p2_long)
def load_long():
    df = pd.read_parquet(CACHE / "p2_long.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df.Close > 0) & (df.Open > 0) & (df.Low > 0)].drop_duplicates(["date", "sym"], keep="last")
    f = (df["Adj Close"] / df["Close"]).fillna(1.0)
    df = df.assign(O=df.Open * f, C=df.Close * f)
    cal = sorted(df[df.sym == "SPY"].date.unique())
    O = df.pivot(index="date", columns="sym", values="O").reindex(cal)
    C = df.pivot(index="date", columns="sym", values="C").reindex(cal)
    return cal, O, C


def proxy_paths():
    cal, O, C = load_long()
    out = {}
    for s in ("SMH", "QQQ"):
        o, c = O[s].to_numpy(float), C[s].to_numpy(float)
        out[s] = {"days": cal, "vo": o / o[0], "vc": c / o[0]}
    stocks = [s for s in O.columns if s not in ETFS]
    Oa, Ca = O[stocks].to_numpy(float), C[stocks].to_numpy(float)
    hist = np.cumsum(np.isfinite(Ca), axis=0) - np.isfinite(Ca)      # k일 전까지 유효 종가 수
    N = len(cal)
    clips = 0
    # 동일가중(매일 시가 재조정)
    vo, vc = np.ones(N), np.ones(N)
    for k in range(N):
        el = (hist[k] >= MOM_MIN_HIST) & np.isfinite(Oa[k])
        if not el.any():
            vc[k] = vo[k]
            if k + 1 < N:
                vo[k + 1] = vo[k]
            continue
        rc = np.where(np.isfinite(Ca[k][el]), Ca[k][el] / Oa[k][el], 1.0)
        rc2 = np.clip(rc, *RATIO_CLIP)
        clips += int((rc2 != rc).sum())
        vc[k] = vo[k] * rc2.mean()
        if k + 1 < N:
            ro = np.where(np.isfinite(Oa[k + 1][el]), Oa[k + 1][el] / Oa[k][el], rc)
            ro2 = np.clip(ro, *RATIO_CLIP)
            clips += int((ro2 != ro).sum())
            vo[k + 1] = vo[k] * ro2.mean()
    out["EW_AI"] = {"days": cal, "vo": vo, "vc": vc}
    # 모멘텀 7종목(120일 수익 상위, 주 첫 거래일 시가 재조정, 보유 종목은 12위 안이면 유지, 동일가중)
    weeks = [date.fromisoformat(d).isocalendar()[:2] for d in cal]
    vo, vc = np.ones(N), np.ones(N)
    held: dict = {}                    # 열 index → 비중
    for k in range(N):
        if k > MOM_LOOKBACK and weeks[k] != weeks[k - 1]:
            j = k - 1
            base_px = Ca[j - MOM_LOOKBACK]
            el = (hist[k] >= MOM_MIN_HIST) & np.isfinite(Ca[j]) & np.isfinite(base_px) & np.isfinite(Oa[k])
            sc = np.full(len(stocks), -np.inf)
            sc[el] = Ca[j][el] / base_px[el] - 1
            order = [int(i) for i in np.argsort(-sc) if np.isfinite(sc[i])]
            rank = {i: r for r, i in enumerate(order)}
            new = [i for i in held if rank.get(i, 10 ** 9) < MOM_N + MOM_BUFFER]
            for i in order:
                if len(new) >= MOM_N:
                    break
                if i not in new:
                    new.append(i)
            held = {i: 1.0 / len(new) for i in new} if new else {}
        cash = 1.0 - sum(held.values())
        sc_, so_ = cash, cash
        nw = {}
        for i, wt in held.items():
            o0, c0 = Oa[k, i], Ca[k, i]
            rc = c0 / o0 if np.isfinite(o0) and np.isfinite(c0) and o0 > 0 else 1.0
            sc_ += wt * float(np.clip(rc, *RATIO_CLIP))
            if k + 1 < N:
                o1 = Oa[k + 1, i]
                ro = o1 / o0 if np.isfinite(o0) and np.isfinite(o1) and o0 > 0 else rc
                ro = float(np.clip(ro, *RATIO_CLIP))
                so_ += wt * ro
                nw[i] = wt * ro
        vc[k] = vo[k] * sc_
        if k + 1 < N:
            vo[k + 1] = vo[k] * so_
            held = {i: v / so_ for i, v in nw.items()}
    out["MOM7_AI"] = {"days": cal, "vo": vo, "vc": vc}
    return out, clips


def run_proxy_year(P, Y, sp):
    days = P["days"]
    a = next(i for i, d in enumerate(days) if d >= f"{Y}-01-01")
    z = max(i for i, d in enumerate(days) if d <= f"{Y}-12-31")
    end = min(z + 1, len(days) - 1)
    seg = {"days": days[a:end + 1], "vo": P["vo"][a:end + 1], "vc": P["vc"][a:end + 1]}
    pre = min(a, WARM_N)
    ext = P["vc"][a - pre:end + 1]
    out = run_path(seg, sp, pre, ext)
    wins = {str(Y): (f"{Y}-01-01", f"{Y}-12-31")}
    if Y == 2026:
        wins = {"2026": W26["전체"], "2026앞": W26["앞"], "2026뒤": W26["뒤"]}
    return {w: wstats(seg["days"], out["r"], out["rb"], out["e"], out["cycles"], lo, hi) for w, (lo, hi) in wins.items()}


# ============================================================================ 본 실행
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def main():
    O, C = ohlc_wide()
    P26, PH = rebuild("base_2026", O, C), rebuild("base_hold", O, C)
    P774 = rebuild("base_2026_774", O, C)
    chk = {}
    for nm, P in (("2026", P26), ("hold", PH), ("774", P774)):
        chk[f"재구성_시가경로_최대상대차_{nm}"] = float(np.max(np.abs(P["vo"] / P["json_open"] - 1)))
        chk[f"재구성_종가평가_최대상대차_{nm}"] = float(np.max(np.abs(P["vc"] / P["json_close"] - 1)))
    # 2026 예열: 보유 기간 경로 끝 120일(2026-01-02 전)을 1/2 종가에 맞춰 붙인다
    hd = np.asarray(PH["days"])
    pre = PH["vc"][hd < P26["days"][0]][-WARM_N:]
    ext26 = np.concatenate([pre * (P26["vc"][0] / pre[-1]), P26["vc"]])
    ext774 = np.concatenate([pre * (P774["vc"][0] / pre[-1]), P774["vc"]])

    # ---- 1) 재현
    res, a26, ah = measure7(P26, PH, ext26, CAND)
    rep = []
    for (bn, w, k), cl in CLAIMS.items():
        mine = res[bn][w][k]
        tol = 0.5 if k == "주기" else (0.05 if k == "손익비" else (0.006 if k == "승률" else TOL))
        rep.append({"항목": f"{bn}/{w}/{k}", "주장": cl, "재계산": mine, "차": mine - cl, "일치": bool(abs(mine - cl) <= tol)})
    res_slip, _, _ = measure7(P26, PH, ext26, CAND, SLIP, SLIP)
    for (bn, w, k), cl in CLAIMS_SLIP.items():
        mine = res_slip[bn][w][k]
        rep.append({"항목": f"미끄러짐0.5%/{bn}/{w}/{k}", "주장": cl, "재계산": mine, "차": mine - cl, "일치": bool(abs(mine - cl) <= TOL)})
    res_lag2, _, _ = measure7(P26, PH, ext26, dict(CAND, lag=2))
    for (bn, w, k), cl in CLAIMS_LAG2.items():
        mine = res_lag2[bn][w][k]
        rep.append({"항목": f"하루늦춤/{bn}/{w}/{k}", "주장": cl, "재계산": mine, "차": mine - cl, "일치": bool(abs(mine - cl) <= TOL)})
    print("재현")
    for x in rep:
        print(f"  {x['항목']:<34} 주장 {x['주장']:+.4f} 재계산 {x['재계산']:+.4f} 차 {x['차']:+.4f} {'OK' if x['일치'] else '불일치'}")
    cyc26 = [dict(c, gain=round(c["gain"], 4)) for c in a26["cycles"]]
    print("2026 주기", [(c["exit_day"], c["re_day"], c["gain"]) for c in cyc26])

    # ---- 2) 시점 감사
    T = P26["days"].index("2026-05-01")
    rng = np.random.default_rng(7)
    Pb = rebuild("base_2026", O, C, bump_after=T, rng=rng)
    extb = np.concatenate([pre * (Pb["vc"][0] / pre[-1]), Pb["vc"]])
    eb = run_path(Pb, CAND, WARM_N, extb)
    chk["미래교란_노출차(T+1 시가까지)"] = float(np.max(np.abs(eb["e"][:T + 2] - a26["e"][:T + 2])))
    # 가치는 T일 시가까지(r[:T]) 같아야 한다. T+1 시가 가치는 흔든 T+1 시가를 쓰므로 달라지는 게 맞다(첫 실행에서 r[:T+1]로 잘못 잼 → 고침)
    chk["미래교란_가치차(T 시가까지)"] = float(np.max(np.abs(np.cumprod(1 + eb["r"][:T]) - np.cumprod(1 + a26["r"][:T]))))
    chk["미래교란_종가평가차(T 종가까지)"] = float(np.max(np.abs(Pb["vc"][:T + 1] - P26["vc"][:T + 1])))
    chk["신호_종가t_체결_시가t+1_확인"] = bool(np.all(a26["e"][1:] == a26["ec"][:-1]))
    res_lag0, _, _ = measure7(P26, PH, ext26, dict(CAND, lag=0))      # 불가능한 같은 날 체결(진단)
    res_lag3, _, _ = measure7(P26, PH, ext26, dict(CAND, lag=3))
    res_slipx, _, _ = measure7(P26, PH, ext26, CAND, SLIP, 0.0)       # 손절 체결만 −0.5%
    # 예열 민감도: 예열 없음(SMA20은 2026 20번째 날부터)
    a_nowarm = run_path(P26, CAND, 0, None)
    res_nowarm = {w: wstats(P26["days"], a_nowarm["r"], a_nowarm["rb"], a_nowarm["e"], a_nowarm["cycles"], lo, hi)
                  for w, (lo, hi) in W26.items()}
    # 2025를 2025-01-02에 보유로 새로 시작(이어 돌리기 대신)
    i25 = next(i for i, d in enumerate(PH["days"]) if d >= "2025-01-01")
    P25 = {"days": PH["days"][i25:], "vo": PH["vo"][i25:], "vc": PH["vc"][i25:]}
    a25 = run_path(P25, CAND, WARM_N, PH["vc"][i25 - WARM_N:])
    res_25fresh = wstats(P25["days"], a25["r"], a25["rb"], a25["e"], a25["cycles"], *WH["2025"])
    # #774(반도체 60%) 경로에 같은 규칙
    a774 = run_path(P774, CAND, WARM_N, ext774)
    res_774 = {w: wstats(P774["days"], a774["r"], a774["rb"], a774["e"], a774["cycles"], lo, hi) for w, (lo, hi) in W26.items()}

    # ---- 3) 이웃
    nbs = []
    for nx in NB_EXIT_N:
        for nr in NB_RE_N:
            nbs.append(dict(CAND, n_exit=nx, n_re=nr))
    nbs += [dict(CAND, out_e=v) for v in NB_OUT_E] + [dict(CAND, lag=v) for v in NB_LAG] + [dict(CAND, band=v) for v in NB_BAND]
    nb_rows = []
    for sp in nbs:
        r, _, _ = measure7(P26, PH, ext26, sp)
        nb_rows.append({"spec": sp, "앞": r["2026"]["앞"]["효율선초과"], "뒤": r["2026"]["뒤"]["효율선초과"],
                        "2025": r["hold"]["2025"]["효율선초과"], "2024": r["hold"]["2024"]["효율선초과"],
                        "2026수익": r["2026"]["전체"]["수익"], "앞수익": r["2026"]["앞"]["수익"], "뒤수익": r["2026"]["뒤"]["수익"],
                        "2025수익": r["hold"]["2025"]["수익"], "AI쇼크": r["2026"]["AI쇼크"]["수익"],
                        "앞기대값": r["2026"]["앞"]["기대값"], "뒤기대값": r["2026"]["뒤"]["기대값"],
                        "2025기대값": r["hold"]["2025"]["기대값"]})
    nb_only = [x for x in nb_rows if x["spec"] != CAND]
    nf = sum(x["앞"] > 0 for x in nb_only)
    nb_sum = {"이웃수": len(nb_only),
              "앞>0": nf, "뒤>0": sum(x["뒤"] > 0 for x in nb_only), "2025>0": sum(x["2025"] > 0 for x in nb_only),
              "셋다>0": sum(x["앞"] > 0 and x["뒤"] > 0 and x["2025"] > 0 for x in nb_only),
              "앞>0중_뒤>0": [sum(x["앞"] > 0 and x["뒤"] > 0 for x in nb_only), nf],
              "앞>0중_2025>0": [sum(x["앞"] > 0 and x["2025"] > 0 for x in nb_only), nf],
              "앞_중앙": float(np.median([x["앞"] for x in nb_only])), "뒤_중앙": float(np.median([x["뒤"] for x in nb_only])),
              "2025_중앙": float(np.median([x["2025"] for x in nb_only]))}
    print("이웃", nb_sum)

    # ---- 4) 위약
    pool = [c["days_out"] for c in a26["cycles"] + ah["cycles"] if c["re_day"] is not None]
    plc = {}
    for nm, P, e, (lo, hi) in (("2026앞", P26, a26["e"], W26["앞"]), ("2026뒤", P26, a26["e"], W26["뒤"]),
                               ("2026전체", P26, a26["e"], W26["전체"]), ("2025", PH, ah["e"], WH["2025"])):
        plc[nm] = placebo(P, e, lo, hi, pool, PLACEBO_SEEDS, 0)
        plc[nm]["보조500"] = placebo(P, e, lo, hi, pool, PLACEBO_EXTRA, 10_000)
        a_ = plc[nm]
        print("위약", nm, a_["빠짐구간"], a_["빠진날"], "실제", {k: round(v, 4) for k, v in a_["실제"].items()},
              "A 효율 p", a_["A"]["효율선초과"]["p"], "B 효율 p", a_["B"]["효율선초과"]["p"],
              "보조 A p", a_["보조500"]["A"]["효율선초과"]["p"], "B p", a_["보조500"]["B"]["효율선초과"]["p"])

    # ---- 5) 장세(대리)
    PX, clips = proxy_paths()
    reg = {}
    for nm in PROXIES:
        reg[nm] = {}
        for Y in PROXY_YEARS:
            reg[nm].update(run_proxy_year(PX[nm], Y, CAND))
    reg["7종목"] = {"2026앞": res["2026"]["앞"], "2026뒤": res["2026"]["뒤"], "2026": res["2026"]["전체"],
                  "2025": res["hold"]["2025"], "2024": res["hold"]["2024"], "2023(5/22~)": res["hold"]["2023(5/22~)"]}
    for nm in reg:
        print("장세", nm, {w: (round(v["수익"], 3), round(v["기본_수익"], 3), round(v["효율선초과"], 3), v["주기"],
                             round(v["승률"], 2) if np.isfinite(v["승률"]) else None,
                             round(v["손익비"], 2) if np.isfinite(v["손익비"]) else None,
                             round(v["기대값"], 4) if np.isfinite(v["기대값"]) else None) for w, v in reg[nm].items()})
    print("대리 가격비 자르기 건수", clips)

    payload = {"id": ID, "candidate": CAND, "checks": chk, "reproduction": rep, "result": res,
               "cycles_2026": cyc26, "cycles_hold_2025": [c for c in ah["cycles"] if "2025-01-01" <= c["exit_day"] <= "2025-12-31"],
               "slip_both": res_slip, "slip_exit_only": res_slipx, "lag2": res_lag2, "lag3": res_lag3, "lag0_diagnostic": res_lag0,
               "no_warmup_2026": res_nowarm, "fresh_2025": res_25fresh, "path_774": res_774,
               "neighbors": nb_rows, "neighbors_summary": nb_sum, "placebo": plc, "placebo_pool_days_out": pool,
               "regimes": reg, "proxy_ratio_clips": clips,
               "grid": {"NB_EXIT_N": NB_EXIT_N, "NB_RE_N": NB_RE_N, "NB_OUT_E": NB_OUT_E, "NB_LAG": NB_LAG, "NB_BAND": NB_BAND,
                        "PLACEBO_SEEDS": PLACEBO_SEEDS, "PLACEBO_EXTRA": PLACEBO_EXTRA, "PROXIES": PROXIES,
                        "PROXY_YEARS": PROXY_YEARS, "MOM": [MOM_LOOKBACK, MOM_N, MOM_BUFFER, MOM_MIN_HIST], "SLIP": SLIP}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(clean(payload), ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장", RESULTS / f"{ID}.json")
    return payload


if __name__ == "__main__":
    main()
