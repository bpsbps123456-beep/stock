"""구현 B — SOXL·SOXS 장초 1분봉 RSI 70 반대 매수 백테스트(BRIEF.md 사전 등록, 구현 A와 독립 재구현).

RSI·교차·체결·보유·겹침을 날짜·봉 단위 반복문으로 직접 계산한다(라이브러리 지표 함수 없음).
32칸(신호 4 × RSI 기준 2 × 체결 2 × 창 2) × 반대·추종·B0, 주 칸 판정(날 묶음 부트스트랩·앞/뒤 반·날 성격),
보조 5분봉 확인.
입력: cache/1m/{SOXL,SOXS}.parquet, cache/5m/{SOXL,SOXS}.parquet (봉 시작 시각 ET)
출력: results/impl_b.json, results/events_b.csv
사용: py code/impl_b.py
"""
import csv
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results")

TICKERS = ("SOXL", "SOXS")
OPP = {"SOXL": "SOXS", "SOXS": "SOXL"}
RSI_N = 14
LEVEL = 70.0
TICK = 0.01            # 진입 +0.01, 목표 관통 +0.01, 시간 청산 −0.01
TARGET = 0.01          # +1%
HOLD_BARS = 30         # 1분봉 30개 = 30분
REG = (9 * 60 + 30, 15 * 60 + 59)      # 정규장 봉 시작
EXT = (4 * 60, 19 * 60 + 59)           # 연장시간 포함 봉 시작
WINDOWS = {"W60": (570, 629), "W30": (570, 599)}
LAGS = {"E1": 1, "E2": 2}
SIGNALS = ("T1", "T1F", "T2", "T3")
BASES = ("R", "X")
PRIMARY = "T1|R|E2|W60"
SEED = 20261029
N_BOOT = 10000
HALF1_LAST = "2026-10-13"   # 앞 반 9/30~10/13, 뒤 반 10/14~10/28
EFF_ONEWAY = 0.6


def hhmm(m):
    return "%02d:%02d" % (m // 60, m % 60)


# ---------------------------------------------------------------- 자료
def load(tk, iv):
    d = pd.read_parquet(os.path.join(ROOT, "cache", iv, f"{tk}.parquet"))
    idx = d.index
    days = [x.isoformat() for x in idx.date]
    mins = list(idx.hour * 60 + idx.minute)
    o, h, l, c = (d[k].astype(float).tolist() for k in ("Open", "High", "Low", "Close"))
    bars = []
    for i in range(len(d)):
        bars.append((days[i], int(mins[i]), o[i], h[i], l[i], c[i]))
    return bars


def table(bars):
    tab = {}
    for day, m, o, h, l, c in bars:
        tab[(day, m)] = (o, h, l, c)
    return tab


# ---------------------------------------------------------------- RSI (반복문)
def wilder_rsi(closes, n=RSI_N):
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    g = 0.0
    lo = 0.0
    for i in range(1, n + 1):
        ch = closes[i] - closes[i - 1]
        if ch > 0:
            g += ch
        elif ch < 0:
            lo += -ch
    ag = g / n
    al = lo / n
    out[n] = _rsi(ag, al)
    for i in range(n + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gi = ch if ch > 0 else 0.0
        li = -ch if ch < 0 else 0.0
        ag = (ag * (n - 1) + gi) / n
        al = (al * (n - 1) + li) / n
        out[i] = _rsi(ag, al)
    return out


def _rsi(ag, al):
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def crossings(bars, lo_min, hi_min, win_lo, win_hi):
    """기준 계열(봉 시작 lo_min~hi_min, 날을 넘어 이어 붙임)의 RSI로 창 안 T1(상향)·T3(하향) 교차를 찾는다.
    반환: {day: {"T1": [s,...], "T3": [s,...]}} (s = 신호 봉 시작 분)."""
    ser = [b for b in bars if lo_min <= b[1] <= hi_min]
    rsi = wilder_rsi([b[5] for b in ser])
    out = {}
    for i in range(1, len(ser)):
        day, m = ser[i][0], ser[i][1]
        if not (win_lo <= m <= win_hi):
            continue
        r0, r1 = rsi[i - 1], rsi[i]
        if r0 is None or r1 is None:
            continue
        rec = out.setdefault(day, {"T1": [], "T3": []})
        if r0 <= LEVEL < r1:
            rec["T1"].append(m)
        if r0 > LEVEL >= r1:
            rec["T3"].append(m)
    return out


def signal_minutes(cross_day, sig, win):
    """그날 그 티커의 창 안 신호 봉 시작 분 목록."""
    lo, hi = win
    t1 = [m for m in cross_day["T1"] if lo <= m <= hi]
    if sig == "T1":
        return t1
    if sig == "T1F":
        return t1[:1]
    if sig == "T2":
        return t1[1:2]
    if sig == "T3":
        return [m for m in cross_day["T3"] if lo <= m <= hi]
    raise ValueError(sig)


# ---------------------------------------------------------------- 체결·보유 (반복문)
def trade(tab, day, e, step, nbars):
    """Y의 봉 시작 e 시가+0.01 진입, 봉 e, e+step, ... (nbars개) 보유. 진입 봉 없으면 None."""
    b = tab.get((day, e))
    if b is None:
        return None
    P = b[0] + TICK
    up = P * (1.0 + TARGET) + TICK
    dn = P * (1.0 - TARGET) - TICK
    hit = False
    exit_m = None
    low = math.inf
    race = None
    last_m, last_c = None, None
    for j in range(nbars):
        m = e + j * step
        bb = tab.get((day, m))
        if bb is None:
            continue
        o, h, l, c = bb
        if race is None:
            w = h >= up
            ls = l <= dn
            if w and ls:
                race = "ambig"
            elif w:
                race = "win"
            elif ls:
                race = "loss"
        if not hit:
            if l < low:
                low = l
            last_m, last_c = m, c
            if h >= up:
                hit = True
                exit_m = m
    if hit:
        pnl = TARGET * 100.0
    else:
        exit_m = last_m
        pnl = ((last_c - TICK) / P - 1.0) * 100.0
    return {"entry": e, "P": P, "hit": hit, "exit": exit_m, "pnl": pnl,
            "mae": (low / P - 1.0) * 100.0, "race": race or "none",
            "short": (not hit) and last_m != e + (nbars - 1) * step}


def simulate(sigs, tabs, lag, step, nbars, overlap):
    """sigs: [(day, X, s, Y)] → 매매 목록과 건너뛴 수."""
    sigs = sorted(sigs, key=lambda r: (r[0], r[2], r[1]))
    busy = {}
    trades = []
    sk_ov = sk_nb = 0
    for day, X, s, Y in sigs:
        e = s + lag
        if overlap:
            last_exit = busy.get((day, Y))
            if last_exit is not None and e < last_exit + step:
                sk_ov += 1
                continue
        r = trade(tabs[Y], day, e, step, nbars)
        if r is None:
            sk_nb += 1
            continue
        prev = busy.get((day, Y))
        busy[(day, Y)] = r["exit"] if prev is None else max(prev, r["exit"])
        r.update(day=day, X=X, s=s, Y=Y)
        trades.append(r)
    return trades, sk_ov, sk_nb


def b0_trades(tabs, days, win, lag, step, nbars):
    lo, hi = win
    out = []
    nb = 0
    for day in days:
        for s in range(lo, hi + 1, step):
            for Y in TICKERS:
                r = trade(tabs[Y], day, s + lag, step, nbars)
                if r is None:
                    nb += 1
                    continue
                r.update(day=day, X=None, s=s, Y=Y)
                out.append(r)
    return out, nb


# ---------------------------------------------------------------- 집계
def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    if not xs:
        return None
    v = sorted(xs)
    k = len(v)
    return v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2.0


def rnd(x, k=6):
    return None if x is None else round(float(x), k)


def stats(trades, sk_ov, sk_nb):
    n = len(trades)
    pn = [t["pnl"] for t in trades]
    miss = [t["pnl"] for t in trades if not t["hit"]]
    rw = sum(1 for t in trades if t["race"] == "win")
    rl = sum(1 for t in trades if t["race"] in ("loss", "ambig"))
    ra = sum(1 for t in trades if t["race"] == "ambig")
    rn = sum(1 for t in trades if t["race"] == "none")
    return {
        "n": n,
        "n_days": len({t["day"] for t in trades}),
        "skipped_overlap": sk_ov,
        "skipped_nobar": sk_nb,
        "hit_rate": rnd(sum(1 for t in trades if t["hit"]) / n) if n else None,
        "ev_mean_pct": rnd(_mean(pn)),
        "ev_median_pct": rnd(_median(pn)),
        "miss_mean_pct": rnd(_mean(miss)),
        "mae_median_pct": rnd(_median([t["mae"] for t in trades])),
        "race_win": rnd(rw / n) if n else None,
        "race_loss": rnd(rl / n) if n else None,
        "race_ambig_n": ra,
        "race_none": rnd(rn / n) if n else None,
    }


def b0_stats(trades):
    s = stats(trades, 0, 0)
    return {k: s[k] for k in ("n", "hit_rate", "ev_mean_pct", "race_win", "race_loss")}


def per_day(trades, days):
    """날별 (매매 수, 손익 합, 도달 수) 배열."""
    pos = {d: i for i, d in enumerate(days)}
    cnt = np.zeros(len(days))
    sm = np.zeros(len(days))
    hits = np.zeros(len(days))
    for t in trades:
        i = pos[t["day"]]
        cnt[i] += 1
        sm[i] += t["pnl"]
        hits[i] += 1 if t["hit"] else 0
    return cnt, sm, hits


def boot_weights(n_days, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(n_boot, n_days))
    W = np.zeros((n_boot, n_days))
    for b in range(n_boot):
        for j in idx[b]:
            W[b, j] += 1
    return W


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [None, None]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [rnd(lo), rnd(hi)]


def boot_cell(W, fade, follow, b0, days):
    fc, fs, fh = per_day(fade, days)
    gc, gs, gh = per_day(follow, days)
    bc, bs, bh = per_day(b0, days)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_ev = (W @ fs) / (W @ fc)
        f_hr = (W @ fh) / (W @ fc)
        g_ev = (W @ gs) / (W @ gc)
        g_hr = (W @ gh) / (W @ gc)
        b_ev = (W @ bs) / (W @ bc)
        b_hr = (W @ bh) / (W @ bc)
    return {
        "fade_ev_ci": ci(f_ev), "fade_hit_ci": ci(f_hr),
        "follow_ev_ci": ci(g_ev), "follow_hit_ci": ci(g_hr),
        "b0_ev_ci": ci(b_ev), "b0_hit_ci": ci(b_hr),
        "fade_minus_b0_hit_ci": ci(f_hr - b_hr), "fade_minus_b0_ev_ci": ci(f_ev - b_ev),
        "fade_minus_follow_hit_ci": ci(f_hr - g_hr), "fade_minus_follow_ev_ci": ci(f_ev - g_ev),
        "fade_empty_draws": int(np.sum((W @ fc) == 0)),
        "follow_empty_draws": int(np.sum((W @ gc) == 0)),
    }


def halves(trades):
    a = [t["pnl"] for t in trades if t["day"] <= HALF1_LAST]
    b = [t["pnl"] for t in trades if t["day"] > HALF1_LAST]
    return {"first": {"n": len(a), "ev_mean_pct": rnd(_mean(a))},
            "second": {"n": len(b), "ev_mean_pct": rnd(_mean(b))}}


def judge(cell, bt, hv):
    f, b0 = cell["fade"], cell["b0"]
    c1 = (f["ev_mean_pct"] is not None and f["ev_mean_pct"] > 0
          and bt["fade_ev_ci"][0] is not None and bt["fade_ev_ci"][0] > 0)
    c2 = (f["hit_rate"] is not None and b0["hit_rate"] is not None
          and f["hit_rate"] - b0["hit_rate"] > 0
          and bt["fade_minus_b0_hit_ci"][0] is not None and bt["fade_minus_b0_hit_ci"][0] > 0)
    h1, h2 = hv["first"]["ev_mean_pct"], hv["second"]["ev_mean_pct"]
    c3 = h1 is not None and h2 is not None and h1 > 0 and h2 > 0
    verdict = "후보" if (c1 and c2 and c3) else ("변동성 장세 덕" if c1 else "불가")
    return {"c1_ev_pos_ci": bool(c1), "c2_hit_above_b0_ci": bool(c2), "c3_both_halves_pos": bool(c3),
            "verdict": verdict}


# ---------------------------------------------------------------- 날 성격
def day_character(tab, days):
    out = {}
    for day in days:
        o930 = tab[(day, 570)][0]
        o1100 = tab[(day, 660)][0]
        hi = -math.inf
        lo = math.inf
        for m in range(570, 660):
            b = tab.get((day, m))
            if b is None:
                continue
            hi = max(hi, b[1])
            lo = min(lo, b[2])
        eff = (o1100 - o930) / (hi - lo) if hi > lo else 0.0
        out[day] = {"eff": rnd(eff, 4), "type": "한 방향" if abs(eff) >= EFF_ONEWAY else "오락가락"}
    return out


def by_day_type(trades, dchar):
    res = {}
    for typ in ("한 방향", "오락가락"):
        tt = [t for t in trades if dchar[t["day"]]["type"] == typ]
        res[typ] = {"n_days": sum(1 for d in dchar.values() if d["type"] == typ),
                    "n": len(tt),
                    "hit_rate": rnd(sum(1 for t in tt if t["hit"]) / len(tt)) if tt else None,
                    "ev_mean_pct": rnd(_mean([t["pnl"] for t in tt]))}
    return res


# ---------------------------------------------------------------- 보조 5분봉
def aux_5m():
    bars = {tk: load(tk, "5m") for tk in TICKERS}
    tabs = {tk: table(bars[tk]) for tk in TICKERS}
    days = sorted({b[0] for b in bars["SOXL"] if REG[0] <= b[1] <= REG[1]})
    win = (570, 625)
    cross = {tk: crossings(bars[tk], REG[0], 15 * 60 + 55, win[0], win[1]) for tk in TICKERS}
    fade_s, foll_s = [], []
    for tk in TICKERS:
        for day, rec in cross[tk].items():
            for s in signal_minutes(rec, "T1", win):
                fade_s.append((day, tk, s, OPP[tk]))
                foll_s.append((day, tk, s, tk))
    fade, fo1, fn1 = simulate(fade_s, tabs, 5, 5, 6, True)
    foll, go1, gn1 = simulate(foll_s, tabs, 5, 5, 6, True)
    b0, _ = b0_trades(tabs, days, win, 5, 5, 6)
    periods = {"2026-08-31~09-29": ("2026-08-31", "2026-09-29"),
               "2026-09-30~10-28": ("2026-09-30", "2026-10-28")}
    out = {"rule": "5분 RSI(14, 기준 R) T1, 창 09:30~10:25 봉 시작, 다음 5분봉 시가+0.01, 6봉 보유, 겹침 금지(청산 봉 시작+5분)",
           "periods": {}}
    for name, (a, z) in periods.items():
        pd_days = [d for d in days if a <= d <= z]
        sel = lambda ts: [t for t in ts if a <= t["day"] <= z]
        f, g, b = sel(fade), sel(foll), sel(b0)   # 건너뛴 수는 두 기간 합으로만 아래에 보고
        W = boot_weights(len(pd_days))
        bt = boot_cell(W, f, g, b, pd_days)
        out["periods"][name] = {
            "n_days": len(pd_days),
            "fade": stats(f, None, None), "follow": stats(g, None, None), "b0": b0_stats(b),
            "boot": {k: bt[k] for k in ("fade_ev_ci", "fade_minus_b0_hit_ci", "fade_minus_follow_ev_ci")},
        }
    out["skipped_overlap"] = {"fade": fo1, "follow": go1}
    out["skipped_nobar"] = {"fade": fn1, "follow": gn1}
    return out


# ---------------------------------------------------------------- 본 실행
def main():
    os.makedirs(OUT, exist_ok=True)
    bars = {tk: load(tk, "1m") for tk in TICKERS}
    tabs = {tk: table(bars[tk]) for tk in TICKERS}
    days = sorted({b[0] for b in bars["SOXL"] if REG[0] <= b[1] <= REG[1]})
    days_s = sorted({b[0] for b in bars["SOXS"] if REG[0] <= b[1] <= REG[1]})
    assert days == days_s, "SOXL·SOXS 거래일 불일치"
    wide = WINDOWS["W60"]
    cross = {}
    for tk in TICKERS:
        cross[(tk, "R")] = crossings(bars[tk], REG[0], REG[1], wide[0], wide[1])
        cross[(tk, "X")] = crossings(bars[tk], EXT[0], EXT[1], wide[0], wide[1])

    W = boot_weights(len(days))
    dchar = day_character(tabs["SOXL"], days)
    cells, boots, judges, short_n = {}, {}, {}, 0
    b0_cache = {}
    primary_events = None
    primary_trades = None
    for sig in SIGNALS:
        for base in BASES:
            for ek, lag in LAGS.items():
                for wk, win in WINDOWS.items():
                    key = f"{sig}|{base}|{ek}|{wk}"
                    fade_s, foll_s = [], []
                    for tk in TICKERS:
                        for day, rec in cross[(tk, base)].items():
                            for s in signal_minutes(rec, sig, win):
                                fade_s.append((day, tk, s, OPP[tk]))
                                foll_s.append((day, tk, s, tk))
                    ov = sig in ("T1", "T3")
                    fade, fo, fn = simulate(fade_s, tabs, lag, 1, HOLD_BARS, ov)
                    foll, go, gn = simulate(foll_s, tabs, lag, 1, HOLD_BARS, ov)
                    if (ek, wk) not in b0_cache:
                        b0_cache[(ek, wk)] = b0_trades(tabs, days, win, lag, 1, HOLD_BARS)
                    b0, _ = b0_cache[(ek, wk)]
                    short_n += sum(1 for t in fade + foll + b0 if t["short"])
                    cell = {"fade": stats(fade, fo, fn), "follow": stats(foll, go, gn), "b0": b0_stats(b0)}
                    cells[key] = cell
                    bt = boot_cell(W, fade, foll, b0, days)
                    hv = halves(fade)
                    boots[key] = dict(bt, fade_halves=hv)
                    judges[key] = judge(cell, bt, hv)
                    if key == PRIMARY:
                        primary_trades = (fade, foll)

    # 주 칸 매매 목록
    fade, foll = primary_trades
    ev_path = os.path.join(OUT, "events_b.csv")
    with open(ev_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "signal_ticker", "signal_time", "buy", "entry_time", "entry_px",
                    "hit", "exit_time", "pnl_pct", "mae_pct"])
        for t in sorted(fade, key=lambda r: (r["day"], r["entry"], r["Y"])):
            w.writerow([t["day"], t["X"], hhmm(t["s"]), t["Y"], hhmm(t["entry"]), "%.4f" % t["P"],
                        1 if t["hit"] else 0, hhmm(t["exit"]), "%.6f" % t["pnl"], "%.6f" % t["mae"]])

    pj = judges[PRIMARY]
    pb = boots[PRIMARY]
    pc = cells[PRIMARY]
    judgement = {
        "primary": PRIMARY,
        "verdict": pj["verdict"],
        "criteria": {
            "1_fade_ev_mean_pct": pc["fade"]["ev_mean_pct"], "1_fade_ev_ci": pb["fade_ev_ci"],
            "1_pass": pj["c1_ev_pos_ci"],
            "2_fade_minus_b0_hit": rnd(pc["fade"]["hit_rate"] - pc["b0"]["hit_rate"]) if pc["fade"]["n"] else None,
            "2_ci": pb["fade_minus_b0_hit_ci"], "2_pass": pj["c2_hit_above_b0_ci"],
            "3_halves": pb["fade_halves"], "3_pass": pj["c3_both_halves_pos"],
        },
        "signal_effect": {
            "fade_minus_b0_ev": rnd(pc["fade"]["ev_mean_pct"] - pc["b0"]["ev_mean_pct"]) if pc["fade"]["n"] else None,
            "fade_minus_b0_ev_ci": pb["fade_minus_b0_ev_ci"],
            "fade_minus_follow_ev": rnd(pc["fade"]["ev_mean_pct"] - pc["follow"]["ev_mean_pct"])
            if pc["fade"]["n"] and pc["follow"]["n"] else None,
            "fade_minus_follow_ev_ci": pb["fade_minus_follow_ev_ci"],
            "fade_minus_follow_hit_ci": pb["fade_minus_follow_hit_ci"],
        },
        "day_type_fade": by_day_type(fade, dchar),
        "day_type_follow": by_day_type(foll, dchar),
        "day_character": dchar,
        "range_other_cells": {
            "n_cells": len(cells),
            "verdict_counts": {v: sum(1 for j in judges.values() if j["verdict"] == v)
                               for v in ("후보", "변동성 장세 덕", "불가")},
            "fade_ev_mean_min": min(c["fade"]["ev_mean_pct"] for c in cells.values() if c["fade"]["n"]),
            "fade_ev_mean_max": max(c["fade"]["ev_mean_pct"] for c in cells.values() if c["fade"]["n"]),
            "fade_hit_min": min(c["fade"]["hit_rate"] for c in cells.values() if c["fade"]["n"]),
            "fade_hit_max": max(c["fade"]["hit_rate"] for c in cells.values() if c["fade"]["n"]),
        },
    }
    result = {
        "cells": cells,
        "events": os.path.relpath(ev_path, ROOT).replace("\\", "/"),
        "judgement": judgement,
        "boot": boots,
        "verdicts_all_cells": judges,
        "aux_5m": aux_5m(),
        "meta": {
            "impl": "B (반복문 독립 재구현)",
            "days": days, "n_days": len(days),
            "half1_days": sum(1 for d in days if d <= HALF1_LAST),
            "half2_days": sum(1 for d in days if d > HALF1_LAST),
            "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.default_rng(seed).integers(0, 21, (10000, 21)), 모든 칸 같은 추출"},
            "trades_with_missing_last_bar": short_n,
            "notes": [
                "entry_px = 진입 봉 시가 + 0.01 (비용 포함 P)",
                "exit_time = 도달 봉 또는 마지막 봉의 봉 시작 시각(ET)",
                "race_loss 는 같은 봉 모호(ambig)를 포함",
                "겹침 금지는 T1·T3에만 적용(T1F·T2는 하루 티커당 신호 1개라 같은 Y 겹침이 생기지 않음)",
            ],
        },
    }
    with open(os.path.join(OUT, "impl_b.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    print("주 칸", PRIMARY, json.dumps(pc, ensure_ascii=False))
    print("판정", json.dumps(judgement["criteria"], ensure_ascii=False), pj["verdict"])
    print("신호 효과", json.dumps(judgement["signal_effect"], ensure_ascii=False))
    print("날 성격", json.dumps(judgement["day_type_fade"], ensure_ascii=False))
    print("32칸 범위", json.dumps(judgement["range_other_cells"], ensure_ascii=False))
    print("마지막 봉 결측 매매", short_n)


if __name__ == "__main__":
    main()
