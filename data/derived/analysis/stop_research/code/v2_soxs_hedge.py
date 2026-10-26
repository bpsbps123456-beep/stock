"""v2 반박 검증: soxs_hedge (2026-10-10).

후보(F-헤지 앞 고르기): SOXX 종가 < SMA50인 동안(t일 종가 판정 → t+1 시가) 주식 80% + SOXS 20%, 그 밖에는 주식 100%, 매일 그 비중으로 맞춤.
harness.py를 import하지 않고 base_*.json·ohlc.parquet만으로 처음부터 다시 짠다.

할 일
  0) 자료 감사: SOXS·SQQQ Open/Close 가짜 점프(하루 +100% 이상·−60% 이하), 기초지수 대비 −3배 잔차.
  1) 후보 재계산 → 주장 숫자와 비교.
  2) 시점 감사: 신호 미래 누설 시험(잘라 낸 자료로 다시 계산), 적용일 확인, t+2 시가 지연판.
  3) 이웃 48개(SMA 40·50·60 × q 0.10·0.15·0.20·0.25 × SOXS·SQQQ × 매일·켜질 때 한 번 사서 보유)
     + 같은 계기 '주식만 (1−q), 나머지 현금' 24개(SMA × q × 재조정 방식).
  4) 같은 평균 노출의 고정 노출(실제로 그 노출로 재계산), 같은 낙폭의 고정 노출.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE, RES = ROOT / "cache", ROOT / "results"

P26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
PHOLD = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"), "관세": ("2025-02-18", "2025-04-04"),
         "24여름": ("2024-06-18", "2024-08-05"), "25가을": ("2025-10-13", "2025-11-21")}
PER = {"2026": P26, "hold": PHOLD}

CLAIM = {"2026_전체": (2.032, -0.157), "2026_앞": (1.944, -0.157), "2026_뒤": (0.030, -0.152), "2026_AI쇼크": (-0.126, None),
         "hold_2025": (0.235, -0.380), "hold_2024": (0.205, -0.250), "hold_관세": (-0.276, None), "hold_24여름": (-0.168, None)}
BASE_CLAIM = {"2026_전체": (2.237, -0.203), "2026_앞": (2.094, -0.187), "2026_뒤": (0.046, -0.203), "2026_AI쇼크": (-0.187, None),
              "2026_7~9월": (0.041, None), "hold_2025": (0.422, -0.471), "hold_2024": (0.334, -0.382)}
FIXED_CLAIM = {0.9: {"2026_전체": (1.923, -0.183), "2026_AI쇼크": (-0.168, None), "hold_2025": (0.410, -0.433)},
               0.8: {"2026_전체": (1.630, -0.163), "2026_AI쇼크": (-0.150, None), "hold_2025": (0.389, -0.393)},
               0.7: {"2026_전체": (1.358, -0.143), "hold_2025": (0.361, -0.351)}}


# ----------------------------------------------------------------------------- 자료
def load_base(name):
    b = json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
    days = list(b["days"])
    op = np.asarray(b["open"], float)
    ret = pd.Series(op[1:] / op[:-1] - 1, index=days[:-1])    # k일 시가 → k+1일 시가
    return days, ret


OH = pd.read_parquet(CACHE / "ohlc.parquet")
OH["date"] = pd.to_datetime(OH["date"]).dt.strftime("%Y-%m-%d")


def ser(sym, field):
    x = OH[(OH.sym == sym) & OH[field].notna()].set_index("date")[field].astype(float).sort_index()
    return x[~x.index.duplicated()]


BASES = {bn: load_base(f) for bn, f in (("2026", "base_2026"), ("hold", "base_hold"))}


def inv_ret(sym, bn, div=False):
    days, ret = BASES[bn]
    o = ser(sym, "Open")
    if div:   # 배당 반영(Adj Close/Close 비로 시가 보정) — 민감도
        f = (ser(sym, "Adj Close") / ser(sym, "Close")).reindex(o.index)
        o = o * f
    o = o.reindex(days)
    if o.isna().any():
        raise ValueError(f"{sym} 시가 빈 날 {int(o.isna().sum())}")
    a = o.to_numpy()
    return pd.Series(a[1:] / a[:-1] - 1, index=days[:-1])


# ----------------------------------------------------------------------------- 자료 감사
def data_audit():
    out = {}
    pairs = {"SOXS": "SOXX", "SQQQ": "QQQ"}
    for inv, und in pairs.items():
        d = {}
        for f in ("Open", "Close"):
            s = ser(inv, f)
            r = s.pct_change().dropna()
            d[f"{f} 하루 +100% 이상"] = int((r >= 1.0).sum())
            d[f"{f} 하루 −60% 이하"] = int((r <= -0.6).sum())
            d[f"{f} 하루 최대·최소"] = [round(float(r.max()), 4), str(r.idxmax()), round(float(r.min()), 4), str(r.idxmin())]
        c, cu = ser(inv, "Close"), ser(und, "Close")
        idx = c.index.intersection(cu.index)
        res = (c.reindex(idx).pct_change() + 3 * cu.reindex(idx).pct_change()).dropna()
        d["종가 −3배 잔차 최대"] = [round(float(res.abs().max()), 4), str(res.abs().idxmax())]
        o, ou = ser(inv, "Open"), ser(und, "Open")
        gap = (o.reindex(idx) / c.reindex(idx).shift(1) - 1) + 3 * (ou.reindex(idx) / cu.reindex(idx).shift(1) - 1)
        d["밤사이 갭 −3배 잔차 최대"] = [round(float(gap.abs().max()), 4), str(gap.abs().idxmax())]
        d["종가 범위(분할 수정 확인)"] = [round(float(c.min()), 2), round(float(c.max()), 2)]
        out[inv] = d
    return out


# ----------------------------------------------------------------------------- 신호·시점
def soxx_signal(n, upto=None):
    c = ser("SOXX", "Close")
    if upto is not None:
        c = c[c.index <= upto]
    sma = c.rolling(n).mean()
    return (c < sma).astype(float)[sma.notna()]


def apply_open(sig, days, lag=1):
    """lag=1: 거래일 k에 'k보다 엄격히 앞선 마지막 신호일' 값(t 종가 → t+1 시가). lag=2: 그보다 하루 앞 신호일 값(t+2 시가)."""
    idx = sig.index.to_numpy()
    vals = sig.to_numpy()
    pos = np.searchsorted(idx, np.asarray(days), side="left") - lag
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], 0.0)
    return pd.Series(out, index=days)


def leak_test(n=50, samples=40, seed=7):
    """임의의 t를 골라 t까지 잘라 낸 자료로 신호를 다시 계산해 전체 자료 신호의 t일 값과 같은지 본다."""
    full = soxx_signal(n)
    rng = np.random.default_rng(seed)
    pts = rng.choice(full.index[full.index >= "2023-05-01"], samples, replace=False)
    bad = [t for t in pts if soxx_signal(n, upto=t).iloc[-1] != full[t]]
    return {"검사 날짜 수": samples, "불일치": len(bad)}


# ----------------------------------------------------------------------------- 수익 경로
def path_daily(bn, on, q, prod):
    """매일 (1−q·on, q·on) 비중으로 맞춤. prod=None이면 인버스 없이 현금."""
    days, ret = BASES[bn]
    o = on.reindex(ret.index).astype(float)
    e = 1 - q * o
    r = e * ret
    if prod:
        r = r + q * o * inv_ret(prod, bn)
    return r, e


def path_hold(bn, on, q, prod):
    """켜지는 날 시가에 주식 q를 팔아 인버스(또는 현금) q를 한 번 사고, 꺼질 때까지 다시 맞추지 않음. 꺼지는 날 시가에 전부 주식."""
    days, ret = BASES[bn]
    o = on.reindex(ret.index).astype(float).to_numpy()
    ir = inv_ret(prod, bn).to_numpy() if prod else np.zeros(len(ret))
    rr = ret.to_numpy()
    sv, iv, prev = 1.0, 0.0, 0.0
    out, ex = [], []
    for k in range(len(rr)):
        tot = sv + iv
        if o[k] > 0.5 and prev <= 0.5:
            sv, iv = (1 - q) * tot, q * tot
        elif o[k] <= 0.5 and prev > 0.5:
            sv, iv = tot, 0.0
        ex.append(sv / tot)
        sv2, iv2 = sv * (1 + rr[k]), iv * (1 + ir[k])
        out.append((sv2 + iv2) / tot - 1)
        sv, iv, prev = sv2, iv2, o[k]
    return pd.Series(out, index=ret.index), pd.Series(ex, index=ret.index)


def path_fixed(bn, e):
    days, ret = BASES[bn]
    return e * ret, pd.Series(e, index=ret.index)


def stats(r, lo, hi):
    x = r[(r.index >= lo) & (r.index <= hi)].to_numpy()
    c = np.concatenate([[1.0], np.cumprod(1 + x)])
    return float(c[-1] - 1), float((c / np.maximum.accumulate(c) - 1).min())


def mean_exp(e, lo, hi):
    return float(e[(e.index >= lo) & (e.index <= hi)].mean())


def metr(paths):
    """paths = {bn: (r, e)} → {'2026_전체': (수익, 낙폭, 평균노출), ...}"""
    out = {}
    for bn, (r, e) in paths.items():
        for k, (lo, hi) in PER[bn].items():
            s = stats(r, lo, hi)
            out[f"{bn}_{k}"] = (round(s[0], 4), round(s[1], 4), round(mean_exp(e, lo, hi), 4))
    return out


def run(n, q, prod, mode, lag=1):
    res = {}
    for bn in ("2026", "hold"):
        days, _ = BASES[bn]
        on = apply_open(soxx_signal(n), days, lag)
        res[bn] = (path_daily if mode == "매일" else path_hold)(bn, on, q, prod)
    return metr(res), res


def fixed(e):
    return metr({bn: path_fixed(bn, e) for bn in ("2026", "hold")})


def fixed_at(key, e):
    bn = key.split("_")[0]
    r, _ = path_fixed(bn, e)
    return stats(r, *PER[bn][key.split("_", 1)[1]])


def fixed_same_mdd(key, mdd):
    """그 구간 낙폭이 mdd와 같아지는 고정 노출(이분법) → (e, 수익)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if fixed_at(key, mid)[1] > mdd:   # 낙폭이 더 얕음 → 노출 늘림
            lo = mid
        else:
            hi = mid
    e = (lo + hi) / 2
    return round(e, 4), round(fixed_at(key, e)[0], 4)


def true_turnover(n, q, prod, bn="2026", lo="2026-01-02", hi="2026-10-09"):
    """매일 비중 맞춤의 실제 회전율(목표 변경 + 하루 동안 벌어진 비중을 되돌리는 매매)."""
    days, ret = BASES[bn]
    on = apply_open(soxx_signal(n), days).reindex(ret.index)
    ir = inv_ret(prod, bn)
    t, w_prev = 0.0, (1.0, 0.0)
    for k in ret.index:
        tgt = (1 - q * on[k], q * on[k])
        if lo <= k <= hi:
            t += abs(tgt[0] - w_prev[0]) + abs(tgt[1] - w_prev[1])
        g = tgt[0] * (1 + ret[k]) + tgt[1] * (1 + ir[k]) + (1 - tgt[0] - tgt[1])
        w_prev = (tgt[0] * (1 + ret[k]) / g, tgt[1] * (1 + ir[k]) / g)
    return t / 2


def episodes(on, lo, hi):
    o = on[(on.index >= lo) & (on.index <= hi)]
    eps, s, prev = [], None, None
    for d, v in o.items():
        if v > 0.5 and s is None:
            s = d
        if v <= 0.5 and s is not None:
            eps.append((s, prev))
            s = None
        prev = d
    if s is not None:
        eps.append((s, prev))
    return eps


# ----------------------------------------------------------------------------- 실행
def main():
    out = {"후보": "SOXX<SMA50 → 주식 80% + SOXS 20%, 매일 맞춤", "자료감사": data_audit()}

    # 기본·고정 노출 재현
    basem = fixed(1.0)
    out["기본 재계산"] = basem
    out["기본 주장 차"] = {k: round(basem[k][0] - v[0], 4) for k, v in BASE_CLAIM.items()}
    fx = {e: fixed(e) for e in (0.9, 0.8, 0.7)}
    out["고정 노출 주장 차"] = {str(e): {k: round(fx[e][k][0] - v[0], 4) for k, v in FIXED_CLAIM[e].items()} for e in fx}

    # 1) 후보 재계산
    cand, cand_paths = run(50, 0.20, "SOXS", "매일")
    out["후보 재계산"] = cand
    out["후보 주장 차(수익, 낙폭)"] = {k: (round(cand[k][0] - v[0], 4), None if v[1] is None else round(cand[k][1] - v[1], 4))
                                for k, v in CLAIM.items()}
    days26, _ = BASES["2026"]
    on26 = apply_open(soxx_signal(50), days26)
    out["후보 2026 켜진 구간"] = episodes(on26, "2026-01-02", "2026-10-09")
    out["후보 켜진 날 비율"] = {"2026": round(float(on26.iloc[:-1].mean()), 3),
                          "hold": round(float(apply_open(soxx_signal(50), BASES['hold'][0]).iloc[:-1].mean()), 3)}
    # 배당 반영 민감도·실제 회전율
    days, ret = BASES["2026"]
    r_div = (1 - 0.2 * on26.reindex(ret.index)) * ret + 0.2 * on26.reindex(ret.index) * inv_ret("SOXS", "2026", div=True)
    out["후보 SOXS 배당 반영 2026 전체"] = [round(x, 4) for x in stats(r_div, *P26["전체"])]
    tt = true_turnover(50, 0.20, "SOXS")
    r_c = cand_paths["2026"][0]
    out["후보 2026 실제 회전율(편도)"] = round(tt, 3)
    out["후보 2026 전체 스프레드 0.1%·0.3% 근사"] = [round((1 + stats(r_c, *P26['전체'])[0]) * (1 - 0.001) ** (2 * tt) - 1, 4),
                                           round((1 + stats(r_c, *P26['전체'])[0]) * (1 - 0.003) ** (2 * tt) - 1, 4)]

    # 2) 시점 감사
    out["누설 시험"] = {f"SMA{n}": leak_test(n) for n in (40, 50, 60)}
    # 적용일 예: 7월 켜짐 날짜의 직전 거래일 SOXX 종가 < SMA50 인지
    c = ser("SOXX", "Close")
    sma = c.rolling(50).mean()
    chk = []
    for s, e in out["후보 2026 켜진 구간"]:
        prev = c.index[c.index < s][-1]
        chk.append({"켜진 첫 시가": s, "판정 종가일": prev, "SOXX": round(float(c[prev]), 2), "SMA50": round(float(sma[prev]), 2),
                    "그날(켜진 날) 종가<SMA": bool(c[s] < sma[s])})
    out["적용일 확인"] = chk
    lag2, _ = run(50, 0.20, "SOXS", "매일", lag=2)
    out["후보 t+2 시가"] = lag2

    # 3) 이웃 격자
    bw_front = 1 + basem["2026_앞"][0]
    rows = []
    for n in (40, 50, 60):
        for q in (0.10, 0.15, 0.20, 0.25):
            for mode in ("매일", "보유"):
                cash, _ = run(n, q, None, mode)
                cash2, _ = run(n, q, None, mode, lag=2)
                for prod in ("SOXS", "SQQQ"):
                    m, _ = run(n, q, prod, mode)
                    m2, _ = run(n, q, prod, mode, lag=2)
                    row = {"SMA": n, "q": q, "상품": prod, "재조정": mode}
                    for k in ("2026_전체", "2026_앞", "2026_뒤", "2026_AI쇼크", "2026_7~9월", "hold_2025", "hold_2024",
                              "hold_관세", "hold_24여름"):
                        row[k] = m[k][:2]
                        row[k + "_현금"] = cash[k][:2]
                    row["2026_평균노출"] = m["2026_전체"][2]
                    a = 1 + m["2026_앞"][0] >= 0.9 * bw_front
                    b2 = m["2026_뒤"][0] > basem["2026_뒤"][0] and m["2026_AI쇼크"][0] > basem["2026_AI쇼크"][0]
                    row["앞90%"], row["뒤·AI쇼크"] = bool(a), bool(b2)
                    row["2026 통과"] = bool(a and b2)
                    for y in ("hold_2025", "hold_2024"):
                        ey = m[y][2]
                        fr = fixed_at(y, ey)
                        row[f"{y}_고정같은노출"] = (round(ey, 4), round(fr[0], 4), round(fr[1], 4))
                        row[f"{y}>고정"] = bool(m[y][0] > fr[0])
                        e_m, r_m = fixed_same_mdd(y, m[y][1])
                        row[f"{y}_같은낙폭고정"] = (e_m, r_m)
                        row[f"{y}>같은낙폭고정"] = bool(m[y][0] > r_m)
                    e_m, r_m = fixed_same_mdd("2026_전체", m["2026_전체"][1])
                    row["2026_전체_같은낙폭고정"] = (e_m, r_m)
                    row["2026_전체>같은낙폭고정"] = bool(m["2026_전체"][0] > r_m)
                    row["인버스다리몫_2026전체"] = round(m["2026_전체"][0] - cash["2026_전체"][0], 4)
                    row["인버스다리몫_2026뒤"] = round(m["2026_뒤"][0] - cash["2026_뒤"][0], 4)
                    row["인버스다리몫_AI쇼크"] = round(m["2026_AI쇼크"][0] - cash["2026_AI쇼크"][0], 4)
                    row["인버스다리몫_2025"] = round(m["hold_2025"][0] - cash["hold_2025"][0], 4)
                    row["인버스다리몫_2024"] = round(m["hold_2024"][0] - cash["hold_2024"][0], 4)
                    row["낙폭차_2026전체(헤지-현금)"] = round(m["2026_전체"][1] - cash["2026_전체"][1], 4)
                    a2 = 1 + m2["2026_앞"][0] >= 0.9 * bw_front
                    b22 = m2["2026_뒤"][0] > basem["2026_뒤"][0] and m2["2026_AI쇼크"][0] > basem["2026_AI쇼크"][0]
                    row["t+2"] = {k: m2[k][:2] for k in ("2026_전체", "2026_앞", "2026_뒤", "2026_AI쇼크", "hold_2025",
                                                         "hold_2024")}
                    row["t+2 2026 통과"] = bool(a2 and b22)
                    rows.append(row)
    out["이웃"] = rows
    cash_rows = []
    for n in (40, 50, 60):
        for q in (0.10, 0.15, 0.20, 0.25):
            for mode in ("매일", "보유"):
                cm, _ = run(n, q, None, mode)
                a = 1 + cm["2026_앞"][0] >= 0.9 * bw_front
                b2 = cm["2026_뒤"][0] > basem["2026_뒤"][0] and cm["2026_AI쇼크"][0] > basem["2026_AI쇼크"][0]
                cash_rows.append({"SMA": n, "q": q, "재조정": mode, **{k: v[:2] for k, v in cm.items()},
                                  "2026_평균노출": cm["2026_전체"][2], "2026 통과": bool(a and b2)})
    out["현금 짝"] = cash_rows

    N = len(rows)
    cnt = lambda f: sum(1 for r in rows if f(r))
    summ = {
        "이웃 수": N,
        "앞90%": cnt(lambda r: r["앞90%"]),
        "뒤>기본·AI쇼크<기본": cnt(lambda r: r["뒤·AI쇼크"]),
        "2026 통과(둘 다)": cnt(lambda r: r["2026 통과"]),
        "2026 통과 t+2": cnt(lambda r: r["t+2 2026 통과"]),
        "AI쇼크<기본": cnt(lambda r: r["2026_AI쇼크"][0] > basem["2026_AI쇼크"][0]),
        "뒤>기본": cnt(lambda r: r["2026_뒤"][0] > basem["2026_뒤"][0]),
        "2025>같은노출고정": cnt(lambda r: r["hold_2025>고정"]),
        "2024>같은노출고정": cnt(lambda r: r["hold_2024>고정"]),
        "2025·2024 둘 다>같은노출고정": cnt(lambda r: r["hold_2025>고정"] and r["hold_2024>고정"]),
        "2025>같은낙폭고정": cnt(lambda r: r["hold_2025>같은낙폭고정"]),
        "2024>같은낙폭고정": cnt(lambda r: r["hold_2024>같은낙폭고정"]),
        "2026전체>같은낙폭고정": cnt(lambda r: r["2026_전체>같은낙폭고정"]),
        "인버스다리몫>0 (2026전체)": cnt(lambda r: r["인버스다리몫_2026전체"] > 0),
        "인버스다리몫>0 (2026뒤)": cnt(lambda r: r["인버스다리몫_2026뒤"] > 0),
        "인버스다리몫>0 (AI쇼크)": cnt(lambda r: r["인버스다리몫_AI쇼크"] > 0),
        "인버스다리몫>0 (2025)": cnt(lambda r: r["인버스다리몫_2025"] > 0),
        "인버스다리몫>0 (2024)": cnt(lambda r: r["인버스다리몫_2024"] > 0),
        "헤지 낙폭이 현금보다 얕음(2026전체)": cnt(lambda r: r["낙폭차_2026전체(헤지-현금)"] > 0),
        "현금 짝 2026 통과": sum(1 for r in cash_rows if r["2026 통과"]),
    }
    for prod in ("SOXS", "SQQQ"):
        for mode in ("매일", "보유"):
            sub = [r for r in rows if r["상품"] == prod and r["재조정"] == mode]
            summ[f"{prod}·{mode}: 2026 통과/12"] = sum(r["2026 통과"] for r in sub)
            summ[f"{prod}·{mode}: 앞90%/12"] = sum(r["앞90%"] for r in sub)
            summ[f"{prod}·{mode}: 뒤·AI쇼크/12"] = sum(r["뒤·AI쇼크"] for r in sub)
            summ[f"{prod}·{mode}: 2026 전체 수익 중앙"] = round(float(np.median([r["2026_전체"][0] for r in sub])), 4)
            summ[f"{prod}·{mode}: 2026 뒤 수익 중앙"] = round(float(np.median([r["2026_뒤"][0] for r in sub])), 4)
            summ[f"{prod}·{mode}: AI쇼크 중앙"] = round(float(np.median([r["2026_AI쇼크"][0] for r in sub])), 4)
    out["이웃 요약"] = summ
    # 참고: 같은 앞 고르기 규칙(앞 자산 ≥ 기본 90%, 앞 낙폭 1%p 이상 개선 → 앞 낙폭 최소 → 앞 수익 최대)을 이웃 48개에 적용
    bd_front = basem["2026_앞"][1]
    ok = [r for r in rows if 1 + r["2026_앞"][0] >= 0.9 * bw_front and r["2026_앞"][1] >= bd_front + 0.01]
    fp = max(ok, key=lambda r: (round(r["2026_앞"][1], 6), r["2026_앞"][0])) if ok else None
    out["이웃 48개 앞 고르기(사후 격자라 참고용)"] = ({k: fp[k] for k in ("SMA", "q", "상품", "재조정", "2026_전체", "2026_앞",
                                                     "2026_뒤", "2026_AI쇼크", "hold_2025", "hold_2024")} if fp else None)
    out["SMA별 2026 켜진 구간"] = {n: episodes(apply_open(soxx_signal(n), days26), "2026-01-02", "2026-10-09")
                              for n in (40, 50, 60)}
    cand_hold, _ = run(50, 0.20, "SOXS", "보유")
    out["후보 켜질 때 한 번 사서 보유"] = cand_hold

    # 4) 후보 vs 같은 평균 노출 고정
    e26 = cand["2026_전체"][2]
    fxc = fixed(round(e26, 4))
    keys = ("2026_전체", "2026_앞", "2026_뒤", "2026_AI쇼크", "2026_7~9월", "hold_2025", "hold_2024", "hold_관세", "hold_24여름")
    vs = {"2026 평균 노출(주식 비중)": round(e26, 4)}
    for k in keys:
        vs[k] = {"후보": cand[k][:2], "고정": fxc[k][:2], "수익차": round(cand[k][0] - fxc[k][0], 4),
                 "낙폭차(+면 후보가 얕음)": round(cand[k][1] - fxc[k][1], 4)}
    vs["같은 낙폭 고정"] = {}
    for k in ("2026_전체", "2026_뒤", "hold_2025", "hold_2024"):
        e_m, r_m = fixed_same_mdd(k, cand[k][1])
        vs["같은 낙폭 고정"][k] = {"후보 낙폭": cand[k][1], "고정 노출": e_m, "고정 수익": r_m, "후보 수익": cand[k][0],
                               "수익차": round(cand[k][0] - r_m, 4)}
    out["후보 vs 고정 노출"] = vs
    # 후보와 같은 계기의 현금판(인버스 없이 20% 현금)
    cash_c, _ = run(50, 0.20, None, "매일")
    out["후보 vs 현금판(SMA50 q0.2 매일)"] = {k: {"후보": cand[k][:2], "현금": cash_c[k][:2],
                                              "인버스다리몫": round(cand[k][0] - cash_c[k][0], 4)} for k in keys}

    p = RES / "v2_soxs_hedge.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # 출력
    print("[자료감사]", json.dumps(out["자료감사"], ensure_ascii=False))
    print("[기본 주장 차]", out["기본 주장 차"])
    print("[고정 주장 차]", out["고정 노출 주장 차"])
    print("[후보]", {k: v for k, v in cand.items()})
    print("[후보 주장 차]", out["후보 주장 차(수익, 낙폭)"])
    print("[켜진 구간 2026]", out["후보 2026 켜진 구간"])
    print("[켜진 비율]", out["후보 켜진 날 비율"])
    print("[배당 반영]", out["후보 SOXS 배당 반영 2026 전체"], "[회전율]", tt, out["후보 2026 전체 스프레드 0.1%·0.3% 근사"])
    print("[누설]", out["누설 시험"])
    for x in chk:
        print("  ", x)
    print("[t+2]", {k: lag2[k][:2] for k in keys})
    print("[이웃 요약]")
    for k, v in summ.items():
        print("  ", k, v)
    print("[이웃 앞 고르기]", out["이웃 48개 앞 고르기(사후 격자라 참고용)"])
    print("[SMA별 켜진 구간]", out["SMA별 2026 켜진 구간"])
    print("[후보 보유판]", {k: cand_hold[k][:2] for k in keys})
    print("[후보 vs 고정]")
    for k, v in vs.items():
        print("  ", k, v)
    print("[후보 vs 현금판]")
    for k, v in out["후보 vs 현금판(SMA50 q0.2 매일)"].items():
        print("  ", k, v)
    print("[이웃 표]")
    for r in rows:
        print(f"  SMA{r['SMA']} q{r['q']:.2f} {r['상품']} {r['재조정']}: 전체 {r['2026_전체'][0]:+.3f}/{r['2026_전체'][1]:.3f} "
              f"앞 {r['2026_앞'][0]:+.3f}/{r['2026_앞'][1]:.3f} 뒤 {r['2026_뒤'][0]:+.3f} AI {r['2026_AI쇼크'][0]:+.3f} "
              f"25 {r['hold_2025'][0]:+.3f}(고정 {r['hold_2025_고정같은노출'][1]:+.3f}) 24 {r['hold_2024'][0]:+.3f}"
              f"(고정 {r['hold_2024_고정같은노출'][1]:+.3f}) 노출 {r['2026_평균노출']:.3f} 통과 {r['2026 통과']} "
              f"t+2통과 {r['t+2 2026 통과']} 다리 {r['인버스다리몫_2026전체']:+.3f}/{r['인버스다리몫_AI쇼크']:+.3f}")
    print("[현금 짝]")
    for r in cash_rows:
        print(f"  SMA{r['SMA']} q{r['q']:.2f} {r['재조정']}: 전체 {r['2026_전체'][0]:+.3f}/{r['2026_전체'][1]:.3f} "
              f"앞 {r['2026_앞'][0]:+.3f}/{r['2026_앞'][1]:.3f} 뒤 {r['2026_뒤'][0]:+.3f} AI {r['2026_AI쇼크'][0]:+.3f} "
              f"25 {r['hold_2025'][0]:+.3f} 24 {r['hold_2024'][0]:+.3f} 통과 {r['2026 통과']}")
    print("저장:", p)


if __name__ == "__main__":
    main()
