"""반박 검증 V_LINE(2026-10-10): 갈래 LINE의 대표 후보 G1 s3·r1(SMA100 이탈 매도 / SMA100 회복 재매수).

목적: '진짜 우위이고 이웃·다른 해에서도 버틴다'는 주장을 깨 보기. stock_lab.py를 import하지 않고
cache/p2_long.parquet만으로 엔진을 새로 짠다(배당 수정 = Adj Close/Close 비율을 시가·종가에 곱함).

규칙(후보): 종목마다 따로. 창 시작 직전 종가에 전량 보유.
  보유 중 c[t-1] < SMA100[t-1] → t일 시가 전량 매도. 현금 중 c[t-1] > SMA100[t-1] → t일 시가 전량 재매수.
  판단은 시가 직전 한 번(같은 날 되사기 없음). 포트폴리오 = 대상 종목 동일가중 매일 재조정.
  SMA100[t-1] = t-1일까지 배당 수정 종가 100개 평균. 장중 주문 없음(장중 손절가 감사는 해당 없음).
주기 = 판 체결가/다시 산 체결가 − 1, 창 끝까지 못 되산 주기는 창 끝 종가로 평가.

할 일 1) 독립 재계산 2) 시점 감사·하루 늦춘 판·미끄러짐 3) 이웃 4) 위약 5) 장세 → results/p2_V_LINE.md·json
위약: 사전 선언 P1(종목 안 날짜 섞기)·P2(무작위 재진입). P1b(첫 구간 고정)·P1x(종목끼리 경로 교환)는 P1 결과를 본 뒤
P1을 해석하려고 더한 보조 위약이다. P1이 사후 종목별 노출 비율을 써서 추세 규칙에 불리하게 치우친 것을 보고,
날짜 순서를 섞은 가짜 경로(실력 0)로 그 치우침을 재는 귀무 보정(null_gap)도 더했다(격자·후보는 바꾸지 않음).

실행: cd code && PYTHONIOENCODING=utf-8 py p2_V_LINE.py        (전체, json+md)
      cd code && PYTHONIOENCODING=utf-8 py p2_V_LINE.py md     (json으로 md만)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않음)
CAND = {"sell_n": 100, "buy_n": 100, "band": 0.0, "confirm": 1, "lag": 1}
NEIGH_LENS = [60, 80, 100, 120, 140]     # 매도선·재매수선 SMA 길이 각각 후보 ±2칸(20 간격) → 5 × 5 = 25(후보 포함)
NEIGH_BAND = [0.01, 0.02]                # 이력 폭 b: 매도 c<선×(1−b), 재매수 c>선×(1+b). 후보 0의 한쪽 이웃 2개
NEIGH_CONFIRM = [2, 3]                   # 연속 종가 확인 일수. 후보 1의 한쪽 이웃 2개
N_NEIGH = len(NEIGH_LENS) ** 2 + len(NEIGH_BAND) + len(NEIGH_CONFIRM)   # 29
SLIP = 0.005                             # 미끄러짐 민감도
N_SEEDS = 20                             # 위약 시드
MIN_HIST = 252                           # 창 시작 전 필요한 거래일
EFF_STEP = 0.005                         # 효율선 노출 격자 간격
WIN = {
    "2026": ("2026-01-02", "2026-10-09"), "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2022": ("2022-01-03", "2022-12-30"), "2018": ("2018-01-02", "2018-12-31"),
}
RUNS = {"2026": WIN["2026"], "2025": WIN["2025"], "2022": WIN["2022"], "2018": WIN["2018"]}
SLICES = {"2026": ["2026", "2026앞", "2026뒤"]}
SHOCK = ("2026-06-22", "2026-07-29")
# 갈래 LINE이 주장한 숫자(results/p2_LINE.md 와 과제 문)
CLAIM = {
    ("base", "2026", "포트_수익"): 0.367, ("base", "2026", "포트_낙폭"): -0.183, ("base", "2026", "포트BH_수익"): 0.586,
    ("base", "2026", "포트BH_낙폭"): -0.255, ("base", "2026", "효율선초과"): -0.030, ("base", "2026", "평균노출"): 0.56,
    ("base", "2026앞", "포트_수익"): 0.524, ("base", "2026앞", "포트_낙폭"): -0.063, ("base", "2026앞", "포트BH_수익"): 0.654,
    ("base", "2026앞", "포트BH_낙폭"): -0.124, ("base", "2026앞", "효율선초과"): 0.219,
    ("base", "2026뒤", "포트_수익"): -0.103, ("base", "2026뒤", "포트_낙폭"): -0.183, ("base", "2026뒤", "포트BH_수익"): -0.041,
    ("base", "2026뒤", "포트BH_낙폭"): -0.255, ("base", "2026뒤", "효율선초과"): -0.103,
    ("base", "2026뒤", "손절주기"): 410, ("base", "2026뒤", "승률"): 0.290, ("base", "2026뒤", "손익비"): 2.36,
    ("base", "2026뒤", "기대값"): -0.0015,
    ("base", "2026앞", "손절주기"): 310, ("base", "2026앞", "승률"): 0.223, ("base", "2026앞", "손익비"): 2.27,
    ("base", "2026앞", "기대값"): -0.017,
    ("base", "2025", "포트_수익"): 0.471, ("base", "2025", "포트_낙폭"): -0.152, ("base", "2025", "포트BH_수익"): 0.743,
    ("base", "2025", "포트BH_낙폭"): -0.336, ("base", "2025", "효율선초과"): 0.188, ("base", "2025", "손절주기"): 765,
    ("base", "2025", "승률"): 0.237, ("base", "2025", "손익비"): 2.18, ("base", "2025", "기대값"): -0.014,
    ("base", "2022", "포트_수익"): -0.192, ("base", "2022", "포트_낙폭"): -0.206, ("base", "2022", "포트BH_수익"): -0.306,
    ("base", "2022", "포트BH_낙폭"): -0.363,
    ("slip", "2026", "포트_수익"): 0.301, ("slip", "2026뒤", "포트_수익"): -0.127, ("slip", "2025", "포트_수익"): 0.392,
    ("shock", "shock", "포트_수익"): -0.153, ("shock", "shock", "포트BH_수익"): -0.218,
}


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
        ok = np.isfinite(g[["Open", "High", "Low", "Close"]].to_numpy()).all(axis=1)
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
        m[n - 1:] = (cs[n:] - cs[:-n]) / n
        s["_sma"][n] = m
    return s["_sma"][n]


def bounds(s, d0, d1):
    a = int(np.searchsorted(s["d"], d0, "left"))
    z = int(np.searchsorted(s["d"], d1, "right")) - 1
    return a, z


# ============================================================================ 규칙 경로 · 평가
def rule_path(s, a, z, sell_n, buy_n, band=0.0, confirm=1, lag=1):
    """p[i] = t=a+i일 시가 거래 뒤 보유(1)/현금(0). 판단은 c[t-lag]까지(lag=1 기본, 2 = 하루 늦춘 판)."""
    c, ls, lb = s["c"], sma(s, sell_n), sma(s, buy_n)
    p = np.zeros(z - a + 1, dtype=np.int8)
    held = 1
    for t in range(a, z + 1):
        j = t - lag
        if held:
            if all(c[j - k] < ls[j - k] * (1 - band) for k in range(confirm)):
                held = 0
        else:
            if all(c[j - k] > lb[j - k] * (1 + band) for k in range(confirm)):
                held = 1
        p[t - a] = held
    return p


def eval_path(s, a, z, p, ss=0.0, sb=0.0):
    """경로 → 일수익, 주기[(판 t, 판 체결가, 산 t 또는 None, 산 체결가)]. a-1 종가 보유로 시작."""
    o, c = s["o"], s["c"]
    r = np.empty(z - a + 1)
    cyc, ex, prev = [], None, 1
    for i in range(z - a + 1):
        t, cur = a + i, int(p[i])
        if prev and cur:
            r[i] = c[t] / c[t - 1] - 1
        elif prev and not cur:
            f = o[t] * (1 - ss)
            r[i] = f / c[t - 1] - 1
            ex = (t, f)
        elif cur:
            f = o[t] * (1 + sb)
            r[i] = c[t] / f - 1
            cyc.append((ex[0], ex[1], t, f))
            ex = None
        else:
            r[i] = 0.0
        prev = cur
    if ex is not None:
        cyc.append((ex[0], ex[1], None, None))
    return r, cyc


def curve(daily):
    return np.concatenate([[1.0], np.cumprod(1 + np.asarray(daily, float))])


def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def eff_excess(pr, bh_daily):
    tgt, best, be = mdd(pr), -np.inf, 0.0
    for e in np.arange(0.0, 1.0 + 1e-9, EFF_STEP):
        cv = curve(e * bh_daily)
        if mdd(cv) >= tgt - 1e-12 and cv[-1] - 1 > best:
            best, be = cv[-1] - 1, e
    return float(pr[-1] - 1 - best), float(be)


def cyc_stats(gains):
    g = np.asarray(gains, float)
    if not len(g):
        return {"손절주기": 0, "승률": np.nan, "평균이익": np.nan, "평균손실": np.nan, "손익비": np.nan, "기대값": np.nan}
    w, l_ = g[g > 0], g[g <= 0]
    return {"손절주기": int(len(g)), "승률": float(len(w) / len(g)),
            "평균이익": float(w.mean()) if len(w) else np.nan, "평균손실": float(l_.mean()) if len(l_) else np.nan,
            "손익비": float(w.mean() / -l_.mean()) if len(w) and len(l_) and l_.mean() < 0 else np.nan,
            "기대값": float(g.mean())}


def universe(data, d0):
    return [k for k, s in data.items() if int(np.searchsorted(s["d"], d0, "left")) >= MIN_HIST]


def run(data, run_key, path_fn, ss=0.0, sb=0.0):
    """path_fn(sym, s, a, z) → p. 반환: 일수익 R, B&H B, 노출 E(DataFrame), 주기 목록(sym 포함), 경로 dict."""
    d0, d1 = RUNS[run_key]
    R, B, E, cyc, paths = {}, {}, {}, [], {}
    for k in universe(data, d0):
        s = data[k]
        a, z = bounds(s, d0, d1)
        if a < MIN_HIST or z <= a:
            continue
        p = path_fn(k, s, a, z)
        r, cy = eval_path(s, a, z, p, ss, sb)
        idx = s["d"][a:z + 1]
        R[k] = pd.Series(r, idx)
        B[k] = pd.Series(s["c"][a:z + 1] / s["c"][a - 1:z] - 1, idx)
        E[k] = pd.Series(p.astype(float), idx)
        cyc += [(k,) + x for x in cy]
        paths[k] = (a, z, p)
    return {"R": pd.DataFrame(R).sort_index(), "B": pd.DataFrame(B).sort_index(), "E": pd.DataFrame(E).sort_index(),
            "cyc": cyc, "paths": paths}


def wstats(data, res, w):
    d0, d1 = WIN[w]
    m = (res["R"].index >= d0) & (res["R"].index <= d1)
    R, B, E = res["R"][m], res["B"][m], res["E"][m]
    pr, pb = curve(R.mean(axis=1).fillna(0)), curve(B.mean(axis=1).fillna(0))
    exc, e_eq = eff_excess(pr, B.mean(axis=1).fillna(0).to_numpy())
    sr, sbh = (1 + R.fillna(0)).prod() - 1, (1 + B.fillna(0)).prod() - 1
    gains = []
    for (k, et, ef, rt, rf) in res["cyc"]:
        s = data[k]
        if not (d0 <= s["d"][et] <= d1):
            continue
        _, iz = bounds(s, d0, d1)
        gains.append(ef / rf - 1 if rt is not None and rt <= iz else ef / s["c"][iz] - 1)
    out = {"종목수": int(R.shape[1]), "포트_수익": float(pr[-1] - 1), "포트_낙폭": mdd(pr),
           "포트BH_수익": float(pb[-1] - 1), "포트BH_낙폭": mdd(pb), "효율선초과": exc, "효율선_e": e_eq,
           "평균노출": float(E.mean().mean()), "수익_나은종목비율": float((sr > sbh + 1e-12).mean()),
           "최악5%_종목수익": float(sr.quantile(0.05)), "최악5%_BH": float(sbh.quantile(0.05))}
    out.update(cyc_stats(gains))
    return out


def all_stats(data, res_by_run):
    out = {}
    for rk, res in res_by_run.items():
        for w in SLICES.get(rk, [rk]):
            out[w] = wstats(data, res, w)
    return out


def shock_stats(res):
    m = (res["R"].index >= SHOCK[0]) & (res["R"].index <= SHOCK[1])
    pr, pb = curve(res["R"][m].mean(axis=1).fillna(0)), curve(res["B"][m].mean(axis=1).fillna(0))
    return {"포트_수익": float(pr[-1] - 1), "포트_낙폭": mdd(pr), "포트BH_수익": float(pb[-1] - 1), "포트BH_낙폭": mdd(pb)}


def rule_fn(sell_n=100, buy_n=100, band=0.0, confirm=1, lag=1):
    return lambda k, s, a, z: rule_path(s, a, z, sell_n, buy_n, band, confirm, lag)


# ============================================================================ 위약
def rle(p):
    vals, lens = [], []
    for x in p:
        if vals and vals[-1] == x:
            lens[-1] += 1
        else:
            vals.append(int(x))
            lens.append(1)
    return vals, lens


def shuffle_segments(p, rng):
    """같은 형(보유/현금) 순서는 그대로, 보유 길이끼리·현금 길이끼리 순서를 섞는다 → 노출·나가기 횟수·나가 있던 기간 분포 동일."""
    vals, lens = rle(p)
    ins = [L for v, L in zip(vals, lens) if v == 1]
    outs = [L for v, L in zip(vals, lens) if v == 0]
    rng.shuffle(ins)
    rng.shuffle(outs)
    it_in, it_out, q = iter(ins), iter(outs), []
    for v in vals:
        q += [v] * (next(it_in) if v == 1 else next(it_out))
    return np.array(q, dtype=np.int8)


def shuffle_keep_first(p, rng):
    """첫 구간(창 시작에 들고 있다가 첫 신호까지)은 그대로 두고 나머지만 섞는다."""
    vals, lens = rle(p)
    if len(vals) <= 2:
        return p.copy()
    return np.concatenate([p[:lens[0]], shuffle_segments(p[lens[0]:], rng)])


def p1_fn(base_paths, data, run_key, rng, keep_first=False):
    """위약 P1: 실제 경로를 창(2026은 앞·뒤 따로) 안에서 구간 순서만 섞는다. keep_first = 위약 P1b(첫 구간 고정)."""
    sh = shuffle_keep_first if keep_first else shuffle_segments

    def f(k, s, a, z):
        p = base_paths[k][2].copy()
        if run_key == "2026":
            cut = int(np.searchsorted(s["d"][a:z + 1], WIN["2026앞"][1], "right"))
            return np.concatenate([sh(p[:cut], rng), sh(p[cut:], rng)])
        return sh(p, rng)
    return f


def p1x_fn(base_paths, rng):
    """보조 위약 P1x(P1 결과를 본 뒤 해석용으로 추가): 실제 경로를 종목끼리 바꿔 단다(길이가 같은 종목끼리).
    날짜별 포트 노출(시장 타이밍)은 그대로, '어느 종목을 나갔나'(종목 고르기)만 무작위."""
    groups = {}
    for k, (a, z, p) in base_paths.items():
        groups.setdefault(len(p), []).append(k)
    assign = {}
    for L, ks in groups.items():
        perm = list(ks)
        rng.shuffle(perm)
        assign.update({k: base_paths[j][2] for k, j in zip(ks, perm)})

    def f(k, s, a, z):
        return assign[k].copy()
    return f


N_NULL, N_NULL_P1 = 10, 3      # 귀무 보정(P1 결과를 본 뒤 추가): 날짜 순서 섞은 가짜 경로 10개 × P1 시드 3


def null_data(data, run_key, rng):
    """귀무 자료: 그 실행의 대상 종목을 같은 달력(창 + 앞 252거래일)에 맞추고, (밤사이, 장중) 로그 수익 쌍의 날짜 순서를
    모든 종목에 같은 순열로 섞는다. 종목 간 동시 움직임·종목별 총수익은 남고, 시계열 예측력(추세·되돌림)은 사라진다.
    이 자료에서 규칙은 날짜 선택 실력이 0이다 → '실제 − P1' 차이가 여기서 얼마나 나오나 = P1의 구조적 치우침."""
    d0, d1 = RUNS[run_key]
    ref = max(data, key=lambda k: len(data[k]["d"]))
    rd = data[ref]["d"]
    i0 = int(np.searchsorted(rd, d0, "left")) - MIN_HIST
    i1 = int(np.searchsorted(rd, d1, "right")) - 1
    cal = rd[i0:i1 + 1]
    perm = np.concatenate([[0], 1 + rng.permutation(len(cal) - 1)])
    out = {}
    for k in universe(data, d0):
        s = data[k]
        o = pd.Series(s["o"], s["d"]).reindex(cal)
        c = pd.Series(s["c"], s["d"]).reindex(cal)
        c = c.ffill().bfill().to_numpy()
        o = np.where(np.isfinite(o.to_numpy()), o.to_numpy(), c)
        on = np.concatenate([[0.0], np.log(o[1:] / c[:-1])])[perm]
        intra = np.log(c / o)[perm]
        cs, os_ = np.empty(len(cal)), np.empty(len(cal))
        cs[0], os_[0] = c[0], o[0]
        for t in range(1, len(cal)):
            os_[t] = cs[t - 1] * np.exp(on[t])
            cs[t] = os_[t] * np.exp(intra[t])
        out[k] = {"d": cal, "o": os_, "c": cs, "_sma": {}}
    return out


def null_gap(data, keys=("포트_수익", "효율선초과", "기대값")):
    """귀무 자료에서 (규칙 − P1 중앙) 차이의 분포. 창별."""
    gaps = {w: {k: [] for k in keys} for w in ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")}
    rng = np.random.default_rng(777)
    for _ in range(N_NULL):
        for rk in RUNS:
            nd = null_data(data, rk, rng)
            br = run(nd, rk, rule_fn())
            bs = all_stats(nd, {rk: br})
            ps = [all_stats(nd, {rk: run(nd, rk, p1_fn(br["paths"], nd, rk, rng))}) for _ in range(N_NULL_P1)]
            for w in SLICES.get(rk, [rk]):
                for k in keys:
                    gaps[w][k].append(bs[w][k] - float(np.nanmedian([p[w][k] for p in ps])))
    return gaps


def diag(data, res, w):
    """진단(근거 아님): (1) 규칙이 들고 있던 날 vs 나가 있던 날 종목 일수익 평균, (2) 종목별 노출 비율 f로 가중한
    '무작위 시점 기대' 포트(위약 P1의 중심을 해석식으로)."""
    d0, d1 = WIN[w]
    m = (res["R"].index >= d0) & (res["R"].index <= d1)
    B, E = res["B"][m], res["E"][m]
    bi, bo = B.to_numpy()[E.to_numpy() == 1], B.to_numpy()[E.to_numpy() == 0]
    f = E.mean()
    fw = curve((B.fillna(0) * f).mean(axis=1).where(B.notna().any(axis=1), 0).to_numpy())
    return {"보유일_일평균": float(np.nanmean(bi)), "현금일_일평균": float(np.nanmean(bo)),
            "보유일수": int(np.isfinite(bi).sum()), "현금일수": int(np.isfinite(bo).sum()),
            "f가중_기대수익": float(fw[-1] - 1), "f가중_낙폭": mdd(fw),
            "f_승자상관": float(np.corrcoef(f.to_numpy(), ((1 + B.fillna(0)).prod() - 1).to_numpy())[0, 1])}


def p2_fn(pool, rng, n=100):
    """위약 P2: SMA100 이탈 매도는 그대로, 재매수는 실제 '나가 있던 거래일 수'(닫힌 주기) 분포에서 무작위로 뽑은 날 시가.
    무작위 재매수 뒤 다음 매도는 새 하향 이탈(종가 ≥ 선을 한 번 본 뒤 종가 < 선)일 때만(곧바로 되팔리는 왕복 방지)."""
    def f(k, s, a, z):
        c, ln = s["c"], sma(s, n)
        p = np.zeros(z - a + 1, dtype=np.int8)
        held, armed, out_t, D = 1, True, None, None
        for t in range(a, z + 1):
            j = t - 1
            if held:
                if not armed and c[j] >= ln[j]:
                    armed = True
                elif armed and c[j] < ln[j]:
                    held, out_t, D = 0, t, int(rng.choice(pool))
            elif t - out_t >= D:
                held, armed = 1, bool(c[j] >= ln[j])
            p[t - a] = held
        return p
    return f


def rank_vs(actual, plac):
    """실제가 위약 몇 %를 이겼나(동률 반)."""
    a = np.asarray(plac, float)
    a = a[np.isfinite(a)]
    if not len(a) or not np.isfinite(actual):
        return np.nan
    return float(((actual > a).sum() + 0.5 * (actual == a).sum()) / len(a))


# ============================================================================ 시점 감사
def timing_audit(data):
    """(1) t일 종가를 바꿔도 t일 보유(시가 결정)는 같아야 한다. (2) t 이후 가격을 흔들어도 t까지 일수익이 같아야 한다."""
    rng = np.random.default_rng(7)
    syms = universe(data, WIN["2026"][0])[:40]
    worst_pos, worst_val = 0, 0.0
    for k in syms:
        s = data[k]
        a, z = bounds(s, *WIN["2026"])
        p1 = rule_path(s, a, z, 100, 100)
        r1, _ = eval_path(s, a, z, p1)
        for T in (a + 20, a + 80, a + 150):
            if T > z:
                continue
            s2 = {"d": s["d"], "o": s["o"].copy(), "c": s["c"].copy(), "_sma": {}}
            s2["c"][T:] *= 1 + rng.normal(0, 0.08, len(s2["c"]) - T)
            s2["o"][T + 1:] *= 1 + rng.normal(0, 0.08, len(s2["o"]) - T - 1)
            p2 = rule_path(s2, a, z, 100, 100)
            r2, _ = eval_path(s2, a, z, p2)
            worst_pos = max(worst_pos, int(np.abs(p1[:T - a + 1] - p2[:T - a + 1]).max()))
            worst_val = max(worst_val, float(np.abs(r1[:T - a] - r2[:T - a]).max()) if T > a else 0.0)
    return {"t일_종가교란_뒤_t일까지_보유차": worst_pos, "t일_이후교란_뒤_t-1일까지_일수익차": worst_val, "점검종목": len(syms)}


# ============================================================================ 본 실행
def main():
    t0 = time.time()
    data = load()
    P = {"grid": {"CAND": CAND, "NEIGH_LENS": NEIGH_LENS, "NEIGH_BAND": NEIGH_BAND, "NEIGH_CONFIRM": NEIGH_CONFIRM,
                  "N_NEIGH": N_NEIGH, "SLIP": SLIP, "N_SEEDS": N_SEEDS, "EFF_STEP": EFF_STEP}}

    # 1) 독립 재계산
    base_res = {rk: run(data, rk, rule_fn()) for rk in RUNS}
    base = all_stats(data, base_res)
    shock = shock_stats(base_res["2026"])
    slip_res = {rk: run(data, rk, rule_fn(), SLIP, SLIP) for rk in ("2026", "2025")}
    slip = all_stats(data, slip_res)
    P.update(base=base, shock=shock, slip=slip)
    rep = []
    for (kind, w, key), v in CLAIM.items():
        src = {"base": base, "slip": slip}.get(kind)
        mine = shock[key] if kind == "shock" else src[w][key]
        tol = {"손절주기": 0, "손익비": 0.05, "평균노출": 0.01}.get(key, 0.005)   # 기본 0.5%p
        rep.append({"구분": kind, "창": w, "지표": key, "주장": v, "재계산": mine, "차": mine - v,
                    "일치": bool(abs(mine - v) <= tol + 1e-9)})
    P["reproduce"] = rep
    print(f"[1] 재계산 {time.time() - t0:.0f}s")

    # 2) 시점 감사 · 하루 늦춘 판 · 매도만 미끄러짐 · 양쪽 미끄러짐
    P["timing"] = timing_audit(data)
    lag_res = {rk: run(data, rk, rule_fn(lag=2)) for rk in ("2026", "2025", "2022", "2018")}
    P["lag2"] = all_stats(data, lag_res)
    sell_slip = {rk: run(data, rk, rule_fn(), SLIP, 0.0) for rk in ("2026", "2025")}
    P["sell_slip"] = all_stats(data, sell_slip)
    lag_slip = {rk: run(data, rk, rule_fn(lag=2), SLIP, SLIP) for rk in ("2026", "2025")}
    P["lag2_slip"] = all_stats(data, lag_slip)
    print(f"[2] 시점 {time.time() - t0:.0f}s")

    # 3) 이웃
    neigh = {}
    specs = [(f"S{a}/B{b}", dict(sell_n=a, buy_n=b)) for a in NEIGH_LENS for b in NEIGH_LENS]
    specs += [(f"S100/B100 폭{int(b * 100)}%", dict(band=b)) for b in NEIGH_BAND]
    specs += [(f"S100/B100 확인{k}일", dict(confirm=k)) for k in NEIGH_CONFIRM]
    assert len(specs) == N_NEIGH
    for name, kw in specs:
        rr = {rk: run(data, rk, rule_fn(**kw)) for rk in ("2026", "2025")}
        st = all_stats(data, rr)
        neigh[name] = {w: {k: st[w][k] for k in ("포트_수익", "포트_낙폭", "효율선초과", "평균노출", "손절주기", "승률", "손익비", "기대값")}
                       for w in ("2026", "2026앞", "2026뒤", "2025")}
    P["neigh"] = neigh
    names = list(neigh)
    ok3 = [n for n in names if all(neigh[n][w]["효율선초과"] > 0 for w in ("2026앞", "2026뒤", "2025"))]
    okf = [n for n in names if neigh[n]["2026앞"]["효율선초과"] > 0]
    P["neigh_summary"] = {
        "개수": len(names), "앞·뒤·2025 모두>0": len(ok3), "앞>0": len(okf),
        "앞>0 중 2025>0": sum(1 for n in okf if neigh[n]["2025"]["효율선초과"] > 0),
        "뒤>0": sum(1 for n in names if neigh[n]["2026뒤"]["효율선초과"] > 0),
        "뒤 BH수익 이김": sum(1 for n in names if neigh[n]["2026뒤"]["포트_수익"] > base["2026뒤"]["포트BH_수익"]),
        "2026 BH수익 이김": sum(1 for n in names if neigh[n]["2026"]["포트_수익"] > base["2026"]["포트BH_수익"]),
        "앞·2025 둘다>0": sum(1 for n in names if neigh[n]["2026앞"]["효율선초과"] > 0 and neigh[n]["2025"]["효율선초과"] > 0),
        "뒤 주기기대값>0": sum(1 for n in names if np.isfinite(neigh[n]["2026뒤"]["기대값"]) and neigh[n]["2026뒤"]["기대값"] > 0),
        "앞 주기기대값>0": sum(1 for n in names if np.isfinite(neigh[n]["2026앞"]["기대값"]) and neigh[n]["2026앞"]["기대값"] > 0),
    }
    print(f"[3] 이웃 {time.time() - t0:.0f}s")

    # 4) 위약
    keys = ("포트_수익", "포트_낙폭", "효율선초과", "기대값", "승률", "손익비", "평균노출", "손절주기")
    pl1 = {w: {k: [] for k in keys} for w in ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")}
    pl1b = {w: {k: [] for k in keys} for w in ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")}
    pl1x = {w: {k: [] for k in keys} for w in ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")}
    pl2 = {w: {k: [] for k in keys} for w in ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")}
    pools = {}
    for rk in RUNS:
        pools[rk] = np.array([rt - et for (_, et, _, rt, _) in base_res[rk]["cyc"] if rt is not None])
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        r1 = {rk: run(data, rk, p1_fn(base_res[rk]["paths"], data, rk, rng)) for rk in RUNS}
        s1 = all_stats(data, r1)
        r2 = {rk: run(data, rk, p2_fn(pools[rk], rng)) for rk in RUNS}
        s2 = all_stats(data, r2)
        r1b = {rk: run(data, rk, p1_fn(base_res[rk]["paths"], data, rk, rng, keep_first=True)) for rk in RUNS}
        s1b = all_stats(data, r1b)
        r1x = {rk: run(data, rk, p1x_fn(base_res[rk]["paths"], rng)) for rk in RUNS}
        s1x = all_stats(data, r1x)
        for w in pl1:
            for k in keys:
                pl1[w][k].append(s1[w][k])
                pl1b[w][k].append(s1b[w][k])
                pl1x[w][k].append(s1x[w][k])
                pl2[w][k].append(s2[w][k])

    def summ(pl):
        return {w: {k: {"실제": base[w][k], "위약_중앙": float(np.nanmedian(v)), "위약_5%": float(np.nanpercentile(v, 5)),
                        "위약_95%": float(np.nanpercentile(v, 95)), "실제가_이긴_비율": rank_vs(base[w][k], v)}
                    for k, v in d.items()} for w, d in pl.items()}
    P["placebo1"], P["placebo1b"], P["placebo2"], P["placebo1x"] = summ(pl1), summ(pl1b), summ(pl2), summ(pl1x)
    P["diag"] = {w: diag(data, base_res[rk], w) for rk in RUNS for w in SLICES.get(rk, [rk])}
    P["pools"] = {rk: {"n": int(len(v)), "중앙": float(np.median(v)), "평균": float(v.mean())} for rk, v in pools.items()}
    print(f"[4] 위약 {time.time() - t0:.0f}s")

    # 4b) P1 귀무 보정(P1 결과를 본 뒤 추가): 날짜 선택 실력 0인 가짜 경로에서도 '규칙 − P1'이 얼마나 마이너스인가
    ng = null_gap(data)
    P["null_gap"] = {}
    for w, d in ng.items():
        P["null_gap"][w] = {}
        for k, v in d.items():
            real = base[w][k] - P["placebo1"][w][k]["위약_중앙"]
            v = np.asarray(v, float)
            P["null_gap"][w][k] = {"실제차": float(real), "귀무_중앙": float(np.median(v)), "귀무_5%": float(np.percentile(v, 5)),
                                   "귀무_95%": float(np.percentile(v, 95)), "귀무가_실제보다_낮은_비율": float((v < real).mean())}
    print(f"[4b] 귀무 보정 {time.time() - t0:.0f}s")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "p2_V_LINE.json").write_text(json.dumps(P, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("저장", RESULTS / "p2_V_LINE.json", f"{time.time() - t0:.0f}s")
    write_md(json.loads((RESULTS / "p2_V_LINE.json").read_text(encoding="utf-8")))


# ============================================================================ 보고서
def pc(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp_(x, d=1):
    return pc(x, d).replace("%", "%p")


def pu(x, d=0):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def write_md(P):
    b, nb, ns = P["base"], P["neigh"], P["neigh_summary"]
    p1, p1b, p1x, p2, dg = P["placebo1"], P["placebo1b"], P["placebo1x"], P["placebo2"], P["diag"]
    ng = P["null_gap"]
    W6 = ("2026", "2026앞", "2026뒤", "2025", "2022", "2018")
    lab = {"2026": "2026", "2026앞": "앞", "2026뒤": "뒤", "2025": "2025", "2022": "2022(맥락)", "2018": "2018(맥락)"}
    back_exc = [nb[n]["2026뒤"]["효율선초과"] for n in nb]
    rep = P["reproduce"]
    nbad = sum(1 for r in rep if not r["일치"])
    maxd = max(abs(r["차"]) for r in rep if r["지표"] not in ("손절주기", "손익비"))
    L = ["# 반박 검증 V_LINE: G1 s3·r1 SMA100 이탈 매도 / SMA100 회복 재매수 (2026-10-10)\n",
         "스크립트 `code/p2_V_LINE.py`(stock_lab 미사용 독립 엔진), 전 결과 `results/p2_V_LINE.json`. "
         f"대상 2026 {b['2026']['종목수']}종목 · 2025 {b['2025']['종목수']} · 2022 {b['2022']['종목수']} · 2018 {b['2018']['종목수']}. "
         "**판정: refuted(우위 없음)**\n",
         "## 결론 3줄\n",
         f"1. **숫자는 맞다.** 주장 {len(rep)}개 중 허용 오차(0.5%p) 밖 {nbad}개, 최대 차 {maxd * 100:.2f}%p(반올림). 시점 누수도 없다.",
         f"2. **확인 구간에서 무너지고 이웃도 같다.** 뒤 구간 효율선초과는 후보 {pc(b['2026뒤']['효율선초과'])}이다. 이웃 {ns['개수']}개도 모두 마이너스였다"
         f"({pc(min(back_exc))}~{pc(max(back_exc))}). 앞·뒤·2025 모두 플러스인 이웃은 {ns['앞·뒤·2025 모두>0']}/{ns['개수']}다. "
         f"하루 늦춰도({pc(P['lag2']['2026뒤']['효율선초과'])}), 미끄러짐 0.5%를 넣어도({pc(P['slip']['2026뒤']['효율선초과'])}) 뒤는 마이너스다.",
         "3. **날짜 선택은 확인 구간에서 무작위보다 나빴고, 앞·2025의 우위는 약한 종목 빼기에서 왔다.** 종목별 노출·손절 횟수를 같게 두고 날짜만 "
         f"무작위로 바꾼 위약 P1은 6개 창 모두 수익이 더 높았다(0/20). 다만 P1은 사후 노출 비율을 써서 추세 규칙에 불리하게 치우친다. "
         f"실력 0인 가짜 경로로 이 치우침을 빼면 앞·2025·2022·2018은 무작위와 구별되지 않는다. 뒤만 그보다 나빴다(실제 차 "
         f"{pp_(ng['2026뒤']['포트_수익']['실제차'])}, 실력 0 범위 {pp_(ng['2026뒤']['포트_수익']['귀무_5%'])}~{pp_(ng['2026뒤']['포트_수익']['귀무_95%'])}). "
         "종목끼리 경로를 바꾼 위약 P1x보다 효율선초과가 높은 비율은 "
         f"앞 {pu(p1x['2026앞']['효율선초과']['실제가_이긴_비율'])}, 2025 {pu(p1x['2025']['효율선초과']['실제가_이긴_비율'])}였지만 "
         f"뒤는 {pu(p1x['2026뒤']['효율선초과']['실제가_이긴_비율'])}였다.\n"]

    L.append("## 1. 재현 (주장 → 독립 재계산)\n")
    L.append("| 창 | 수익 / 낙폭 | 매수·보유 | 효율선초과 | 주기 n · 승률 · 손익비 · 기대값 |\n|---|---|---|---|---|")
    cl = {(r["구분"], r["창"], r["지표"]): r for r in rep}

    def cv(kind, w, k, fmt=pc):
        r = cl.get((kind, w, k))
        return "" if r is None else f"{fmt(r['주장'])} → {fmt(r['재계산'])}"
    for w in ("2026", "2026앞", "2026뒤", "2025", "2022"):
        cyc = ""
        if ("base", w, "손절주기") in cl:
            cyc = (f"{cl[('base', w, '손절주기')]['주장']} → {cl[('base', w, '손절주기')]['재계산']} · {cv('base', w, '승률', lambda x: pu(x, 1))} · "
                   f"{cl[('base', w, '손익비')]['주장']:.2f} → {cl[('base', w, '손익비')]['재계산']:.2f} · {cv('base', w, '기대값', lambda x: pc(x, 2))}")
        L.append(f"| {lab[w]} | {cv('base', w, '포트_수익')} / {cv('base', w, '포트_낙폭')} | {cv('base', w, '포트BH_수익')} / {cv('base', w, '포트BH_낙폭')} "
                 f"| {cv('base', w, '효율선초과')} | {cyc} |")
    L.append(f"| 미끄러짐 0.5% | 2026 {cv('slip', '2026', '포트_수익')}, 뒤 {cv('slip', '2026뒤', '포트_수익')}, 2025 {cv('slip', '2025', '포트_수익')} | | | |")
    L.append(f"| 7월 쇼크 6/22~7/29 | {cv('shock', 'shock', '포트_수익')} | {cv('shock', 'shock', '포트BH_수익')} | | |")
    L.append(f"\n- 2026 평균 노출 {b['2026']['평균노출']:.2f}(주장 0.56). 앞·뒤 주기 통계는 창 끝까지 못 되산 주기를 그 창 끝 종가로 평가했다"
             "(앞 고르기에 뒤 가격이 섞이지 않음). 이렇게 해도 갈래 수치와 같다.\n")

    L.append("## 2. 격자 (스크립트 맨 위 상수, 결과를 본 뒤 바꾸지 않음)\n")
    L.append("| 부 | 내용 | 개수 |\n|---|---|---|")
    L.append(f"| 이웃: SMA 길이 | 매도선 {NEIGH_LENS} × 재매수선 {NEIGH_LENS}(20 간격, 후보 100의 ±2칸) | {len(NEIGH_LENS) ** 2} |")
    L.append(f"| 이웃: 이력 폭 | 매도 c<선×(1−b), 재매수 c>선×(1+b), b = {', '.join(pu(x) for x in NEIGH_BAND)}(후보 0의 한쪽) | {len(NEIGH_BAND)} |")
    L.append(f"| 이웃: 확인 일수 | 연속 {', '.join(map(str, NEIGH_CONFIRM))}일 종가(후보 1의 한쪽) | {len(NEIGH_CONFIRM)} |")
    L.append("| 시점 | 하루 늦춘 판(c[t−2]·SMA[t−2] → t 시가) · 매도만 0.5% · 양쪽 0.5% · 늦춤+양쪽 0.5% | 4 |")
    L.append("| 위약(사전) | P1 종목 안에서 보유·현금 구간 순서 섞기(노출·손절 횟수·나가 있던 기간 분포 동일, 2026은 앞·뒤 따로) · "
             f"P2 SMA100 매도는 그대로, 재매수만 실제 '나가 있던 거래일 수' 분포에서 무작위 | 2 × 시드 {N_SEEDS} |")
    L.append(f"| 위약(P1을 본 뒤 해석용 추가) | P1b 첫 구간(창 시작~첫 신호) 고정 · P1x 같은 길이 종목끼리 실제 경로 교환(날짜별 포트 노출 동일) | 2 × 시드 {N_SEEDS} |")
    L.append(f"| P1 귀무 보정(P1을 본 뒤 추가) | (밤사이·장중) 수익 쌍의 날짜 순서를 모든 종목에 같은 순열로 섞은 가짜 경로(창 + 앞 252일) → 규칙·P1 다시 | "
             f"{N_NULL} × P1 시드 {N_NULL_P1} |")
    L.append(f"\n이웃 합계 {N_NEIGH}개(후보 포함). 창: 2026 앞(1/2~5/29)·뒤(6/1~10/9)는 2026을 한 번 돌린 곡선을 자른다. 2025는 보조, 2022·2018은 맥락이다.\n")

    L.append("## 3. 변수 시점\n")
    L.append("| 변수 | 언제 아는가 | 쓰임 |\n|---|---|---|")
    L.append("| 종가 c[t−1], SMA100[t−1](t−1까지 배당 수정 종가 100개) | t−1 종가 | t일 시가 전량 매도·재매수 |")
    L.append("| 장중 손절가 | 없음(장중 주문 없는 규칙) | 감사 해당 없음 |")
    L.append("| 하루 늦춘 판 c[t−2], SMA100[t−2] | t−2 종가 | t일 시가 |")
    L.append("| 효율선 노출 e(같은 낙폭의 고정 노출) | 사후(창 전체) | 비교 기준만, 매매 근거 아님 |")
    L.append("| P1·P1b 종목별 노출 비율 | 사후(창 전체 규칙 결과) | 위약 쪽에 유리한 정보(아래 한계) |")
    t = P["timing"]
    L.append(f"\n감사: {t['점검종목']}종목에서 t일 종가와 그 뒤 가격을 흔들어도 t일 보유 차 {t['t일_종가교란_뒤_t일까지_보유차']}, "
             f"t−1일까지 일수익 차 {t['t일_이후교란_뒤_t-1일까지_일수익차']:.1e}. 판단은 시가 직전 한 번이라 같은 날 판 뒤 되사기가 없다.\n")

    L.append("## 4. 후보와 시점 변형 (동일가중 포트 수익 / 낙폭 (효율선초과))\n")
    L.append("| 판 | " + " | ".join(lab[w] for w in W6) + " |\n|---|" + "---|" * 6)
    L.append("| 매수·보유 | " + " | ".join(f"{pc(b[w]['포트BH_수익'])} / {pc(b[w]['포트BH_낙폭'])}" for w in W6) + " |")
    for key, name in (("base", "후보 s3·r1"), ("lag2", "하루 늦춤"), ("sell_slip", "매도만 0.5%"), ("slip", "양쪽 0.5%"),
                      ("lag2_slip", "늦춤+양쪽 0.5%")):
        S = P[key]
        L.append(f"| {name} | " + " | ".join(
            (f"{pc(S[w]['포트_수익'])} / {pc(S[w]['포트_낙폭'])} ({pc(S[w]['효율선초과'])})" if w in S else "—") for w in W6) + " |")
    L.append(f"\n- 하루 늦추면 오히려 나아진다(2026 {pc(P['lag2']['2026']['포트_수익'])}, 뒤 {pc(P['lag2']['2026뒤']['포트_수익'])}). "
             f"뒤 주기 승률도 {pu(b['2026뒤']['승률'])} → {pu(P['lag2']['2026뒤']['승률'])}로 오른다. 선을 깬 다음 날 시가가 단기 바닥이나 꼭지인 경우가 많다는 뜻이다. "
             "체결일 하루 차이로 2026 수익이 최대 약 4%p 달라진다.\n")

    L.append("## 5. 주기 승률·손익비 (장세별, 미끄러짐 0)\n")
    L.append("| 창 | 주기 n | 승률 | 평균이익 | 평균손실 | 손익비 | 손익분기 손익비 | 기대값 | 효율선초과 | 하루 늦춤 승률·손익비·기대값 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for w in ("2026앞", "2026뒤", "2025", "2022", "2018"):
        s, lg = b[w], P["lag2"][w]
        be = (1 - s["승률"]) / s["승률"] if s["승률"] else np.nan
        L.append(f"| {lab[w]} | {s['손절주기']} | {pu(s['승률'], 1)} | {pc(s['평균이익'])} | {pc(s['평균손실'])} | {s['손익비']:.2f} | {be:.2f} "
                 f"| {pc(s['기대값'], 2)} | {pc(s['효율선초과'])} | {pu(lg['승률'])} · {lg['손익비']:.2f} · {pc(lg['기대값'], 2)} |")
    L.append("\n- 2026·2025는 손익비 2.2~2.4가 손익분기(2.4~3.5)에 못 미쳐 기대값이 마이너스다. 꺾인 해 2022와 2018은 손익비 5.6~6.8로 기대값이 플러스였다. "
             "그래도 포트 효율선초과는 마이너스다. 주기 기대값은 나가 있던 구간만 재므로, 되산 뒤 들고 있던 날의 손실(아래 진단)은 들어가지 않는다.\n")

    L.append("## 6. 이웃 (효율선초과 앞 / 뒤 / 2025)\n")
    L.append("| 매도선 \\ 재매수선 | " + " | ".join(f"B{x}" for x in NEIGH_LENS) + " |\n|---|" + "---|" * len(NEIGH_LENS))
    for a in NEIGH_LENS:
        L.append(f"| S{a} | " + " | ".join(
            f"{pc(nb[f'S{a}/B{x}']['2026앞']['효율선초과'], 0)} / {pc(nb[f'S{a}/B{x}']['2026뒤']['효율선초과'], 0)} / "
            f"{pc(nb[f'S{a}/B{x}']['2025']['효율선초과'], 0)}" for x in NEIGH_LENS) + " |")
    extra = [n for n in nb if "폭" in n or "확인" in n]
    L.append("\n" + " · ".join(f"{n}: {pc(nb[n]['2026앞']['효율선초과'], 0)} / {pc(nb[n]['2026뒤']['효율선초과'], 0)} / "
                               f"{pc(nb[n]['2025']['효율선초과'], 0)}" for n in extra))
    L.append(f"\n- 앞 > 0 {ns['앞>0']}/{ns['개수']}, 앞·2025 둘 다 > 0 {ns['앞·2025 둘다>0']}/{ns['개수']}, 뒤 > 0 {ns['뒤>0']}/{ns['개수']}, "
             f"**앞·뒤·2025 모두 > 0 {ns['앞·뒤·2025 모두>0']}/{ns['개수']}**. 뒤 매수·보유 수익({pc(b['2026뒤']['포트BH_수익'])})을 이긴 이웃은 "
             f"{ns['뒤 BH수익 이김']}개, 2026 전체 매수·보유를 이긴 이웃은 {ns['2026 BH수익 이김']}개다. 후보는 우연한 한 점이 아니다. "
             "앞에서 좋고 뒤에서 나쁜 모양이 이웃 전체에 고르게 나타난다.\n")

    L.append(f"## 7. 위약 (실제가 위약 {N_SEEDS}개 중 몇 %를 이겼나, 50% = 무작위와 같음)\n")
    L.append("| 위약 · 지표 | " + " | ".join(lab[w] for w in W6) + " |\n|---|" + "---|" * 6)
    for nm, PL, ks in (("P1 날짜 무작위", p1, ("포트_수익", "효율선초과", "포트_낙폭", "기대값")),
                       ("P1b 첫 구간 고정", p1b, ("포트_수익", "효율선초과")),
                       ("P2 재진입 무작위", p2, ("효율선초과", "기대값")),
                       ("P1x 종목 교환", p1x, ("포트_수익", "효율선초과"))):
        for k in ks:
            kk = {"포트_수익": "수익", "효율선초과": "효율선초과", "포트_낙폭": "낙폭(얕을수록 이김)", "기대값": "주기 기대값"}[k]
            L.append(f"| {nm} · {kk} | " + " | ".join(pu(PL[w][k]["실제가_이긴_비율"]) for w in W6) + " |")
    L.append("\n| 수익 / 효율선초과 | " + " | ".join(lab[w] for w in W6) + " |\n|---|" + "---|" * 6)
    L.append("| 실제 | " + " | ".join(f"{pc(b[w]['포트_수익'])} / {pc(b[w]['효율선초과'])}" for w in W6) + " |")
    for nm, PL in (("P1 중앙", p1), ("P1b 중앙", p1b), ("P2 중앙", p2), ("P1x 중앙", p1x)):
        L.append(f"| {nm} | " + " | ".join(f"{pc(PL[w]['포트_수익']['위약_중앙'])} / {pc(PL[w]['효율선초과']['위약_중앙'])}" for w in W6) + " |")
    L.append(f"\n- P2 평균 노출은 {min(p2[w]['평균노출']['위약_중앙'] for w in W6):.2f}~{max(p2[w]['평균노출']['위약_중앙'] for w in W6):.2f}로 실제("
             f"{min(b[w]['평균노출'] for w in W6):.2f}~{max(b[w]['평균노출'] for w in W6):.2f})보다 높다. 그래서 수익보다 효율선초과로 본다. "
             f"P2 대기 기간 분포는 2026 닫힌 주기 {P['pools']['2026']['n']}개, 중앙 {P['pools']['2026']['중앙']:.0f}거래일이다.")
    L.append("- P1·P1b의 치우침: 위약은 종목별 노출 비율(창 전체 결과)을 받아 날짜에 고르게 편다. 추세 규칙은 오른 종목일수록 오래 들고 있어서, "
             "날짜 선택 실력이 0이어도 P1보다 낮게 나온다(평균 차 = −공분산(노출 비율, 종목 수익)). 그래서 아래처럼 실력 0인 가짜 경로에서 같은 차이를 재서 뺐다.\n")
    L.append("| 창 | 실제 − P1 중앙 (수익) | 실력 0 가짜 경로의 같은 차 중앙 [5~95%] | 가짜 경로가 실제보다 낮은 비율 | 효율선초과로 같은 비교 |")
    L.append("|---|---|---|---|---|")
    for w in W6:
        g, ge = ng[w]["포트_수익"], ng[w]["효율선초과"]
        L.append(f"| {lab[w]} | {pp_(g['실제차'])} | {pp_(g['귀무_중앙'])} [{pp_(g['귀무_5%'])}~{pp_(g['귀무_95%'])}] | {pu(g['귀무가_실제보다_낮은_비율'])} "
                 f"| {pp_(ge['실제차'])} vs {pp_(ge['귀무_중앙'])} ({pu(ge['귀무가_실제보다_낮은_비율'])}) |")
    L.append("\n- 낮은 비율 50% 근처면 무작위와 같고, 0%면 실력 0보다도 나쁘다. 뒤 구간(수익 0%)과 2026 전체(0%)만 실력 0보다 나빴다. "
             "앞·2025·2022·2018은 무작위와 구별되지 않는다(50~80%).")
    L.append("- P1x를 이기면 같은 날짜별 노출에서 어느 종목을 뺄지는 맞혔다는 뜻이다. 후보의 앞·2025 효율선 우위는 여기서 왔다. "
             "뒤에서는 그 효과도 사라졌다(5%). P2(재진입만 무작위) 대비도 앞 100%·2025 75%였으나 뒤는 0%였다.\n")

    L.append("## 보조·맥락\n")
    L.append("| 창 | 들고 있던 날 종목 일평균 | 나가 있던 날 종목 일평균 | 종목별 노출 비율과 그 종목 창 수익의 상관 |\n|---|---|---|---|")
    for w in W6:
        d = dg[w]
        L.append(f"| {lab[w]} | {pc(d['보유일_일평균'], 2)} | {pc(d['현금일_일평균'], 2)} | {d['f_승자상관']:.2f} |")
    L.append("\n- 뒤 구간·2022·2018에서는 나가 있던 날의 종목 수익이 들고 있던 날보다 높았다. 선을 깨고 판 뒤에 오르고, 되산 뒤에 내렸다는 뜻이다. "
             f"7월 쇼크(6/22~7/29) 손실은 {pc(P['shock']['포트_수익'])}로 매수·보유 {pc(P['shock']['포트BH_수익'])}보다 작았다. "
             "그러나 반등을 놓쳐 뒤 구간 전체로는 더 잃었다.")
    L.append("- 사용자 질문(승률 30%라도 손익비로 이기나)에 대한 이 후보의 답: 승률 22~34%, 손익비 2.2~6.8로 모양은 맞다. "
             "그러나 2026·2025에서는 손익비가 손익분기에 못 미쳤다. 손익비가 넉넉했던 해(2022·2018)에도 포트 효율선초과는 마이너스였다"
             f"(2022 {pc(b['2022']['효율선초과'])}, 2018 {pc(b['2018']['효율선초과'])}). 낙폭은 줄였지만 같은 낙폭의 덜 들고 있기보다 못했다.\n")

    L.append("## 선택편향·한계\n")
    L.append("- G1은 갈래 LINE 60개(서로 다른 58개) 중 앞 1등이다. 이웃 29개가 모두 앞 > 0이므로 앞 성적은 운 좋은 한 점이 아니라 앞 장세의 성질이다. 뒤 실패도 이웃 전체에 고르다.")
    L.append(f"- 생존편향: 종목 목록은 2026-10 기준이다. 2026 대상 {b['2026']['종목수']}종목 중 28개는 AI_UNIVERSE 밖이다(TIME·7종목 보유 이력: ADBE·MELI·PDD·RKLB 등). "
             "GOOG·GOOGL이 둘 다 들어가 알파벳 비중이 두 배다. 망한 종목이 빠져 있어 손절에는 불리한 쪽으로 기운다.")
    L.append("- 사전에 정한 위약 P1은 추세 규칙에 불리하게 치우쳐 있었다. 종목별 노출 비율과 그 종목 창 수익의 상관이 0.50~0.80이라 위약 수준이 부풀려진다"
             "(예: 2026 P1 중앙 +67.9%는 따라 할 수 없다). 그래서 'P1에 0/20'만으로 날짜 선택이 나쁘다고 결론 내지 않았다. "
             f"P1b·P1x·귀무 보정은 P1 결과를 본 뒤 해석용으로 더한 것이다. 후보·격자는 바꾸지 않았다. 귀무 보정은 가짜 경로 {N_NULL}개뿐이라 5~95% 구간이 거칠다.")
    L.append("- 수수료·현금 이자는 0이고, 미끄러짐은 민감도로만 넣었다. 동일가중 매일 재조정 포트 기준이며, 데스크 7종목 규칙 위 결과가 아니다.")
    (RESULTS / "p2_V_LINE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("md", RESULTS / "p2_V_LINE.md")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "md":
        write_md(json.loads((RESULTS / "p2_V_LINE.json").read_text(encoding="utf-8")))
    else:
        main()
