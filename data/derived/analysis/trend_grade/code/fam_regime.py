"""계열 'regime' — '지금 추세 등급이 통하는 장세인가'를 미리 알 수 있는지 (2026-10-12).

기준 점수 2개
    L  = C/SMA200 - 1 (SMA200 = rolling(200, min_periods=120))
    E2 = {C/SMA200-1, C/SMA50-1, C/SMA20-1, C/252일 최고 고가-1} 각각을 H.pct_rank(x, universe_mask & x.notna())로
         단면 백분위로 바꾼 뒤 4개 평균(하나라도 비면 NaN; universe는 봉 130개 이상이라 사실상 항상 4개 다 있음).
         252일 최고 고가 = H.rolling(252, min_periods=120).max() (t일 고가 포함).

장세 지표 10개 (사전 등록, 모두 t일 종가까지 아는 값). 값은 1(on)·0(off)·NaN(정의 안 됨).
    G1  h=10 신호 중 매도일(s+11 시가) <= t 인 최근 20개 신호일의 IC10 평균 > 0  → ic10.shift(11).rolling(20, mp=10)
    G2  h=20, 매도일(s+21 시가) <= t, 최근 20개                                      → ic20.shift(21).rolling(20, mp=10)
        (IC는 기준 점수 자신, 채점기와 같은 방식: universe_mask & 점수 있음 & 다음 날 시가 있음, t+1 시가→t+1+h 시가)
    G3  universe_mask 종목 중 C > SMA50 비율 >= 50%
    G4  그 비율의 10거래일 변화 > 0
    G5  QQQ C > QQQ SMA50
    G6  QQQ C > QQQ SMA20
    G7  20일 수익(C/C[-20]-1)의 universe 단면 표준편차 >= 직전 252일(오늘 제외) 중앙값 (mp=120)
    G8  기준 점수 상위 10%(t일 백분위 > 0.9)의 20일 수익 평균 - universe 평균 > 0
    G9  SOXX C > SOXX SMA20 (stop_research/cache/ohlc.parquet, 분할 반영 Close)
    G10 G1 그리고 G5
격자 밖(extra, 결과 보기 전 등록, 4개):
    extra_G8lag  t-20일 기준 점수 상위 10%의 t-20→t 종가 수익 평균 - t-20일 universe 평균 > 0 (지연 0의 실현 성과)
    extra_QQQ200 QQQ C > QQQ SMA200

잴 것: 기준 점수 × 지표마다 front/back/y2025/y2024(+late는 h5·h10만)에서 지표 on/off별 IC10·IC20 평균·일수·NW t,
상위20% 초과수익(h20), 그리고 '주 첫 거래일 직전 종가에 지표 on → S(백분위>0.9) 같은 비중, off → 보유 종목 전체
같은 비중'의 주간 보유 수익·MDD(H.weekly_portfolio 방식 복제, 항상 S·항상 전체와 비교).
미래 정보 점검: (1) 지표마다 자르기 시험(17개 시점에서 t 이후 자료를 지우고 다시 계산해 t까지 전체 이력이 같은지),
(2) 일부러 새게 만든 G1(shift 10)이 자르기 시험에서 떨어지는지(시험의 검출력), (3) 변형마다 H.no_lookahead(기준 점수 + 10×on).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import harness as H

SOXX_PATH = os.path.join(H.ROOT, "data", "derived", "analysis", "stop_research", "cache", "ohlc.parquet")
WINDOWS = ("front", "back", "y2025", "y2024", "late")
GRID = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10")
EXTRA = ("extra_G8lag", "extra_QQQ200")
SPEC = {
    "G1": "최근 실현 IC10>0: ic10.shift(11).rolling(20,min_periods=10).mean()>0 (매도일 s+11<=t)",
    "G2": "최근 실현 IC20>0: ic20.shift(21).rolling(20,min_periods=10).mean()>0 (매도일 s+21<=t)",
    "G3": "universe 중 C>SMA50 비율>=0.5",
    "G4": "C>SMA50 비율의 10일 변화>0",
    "G5": "QQQ C>SMA50",
    "G6": "QQQ C>SMA20",
    "G7": "20일 수익 단면 표준편차 >= 직전 252일 중앙값(shift1, min_periods=120)",
    "G8": "t일 기준 점수 상위10%의 20일 수익 평균 - universe 평균 > 0",
    "G9": "SOXX C>SMA20",
    "G10": "G1 and G5 (어느 하나 NaN이면 NaN)",
    "extra_G8lag": "t-20일 상위10%의 t-20→t 종가 수익 평균 - t-20일 universe 평균 > 0",
    "extra_QQQ200": "QQQ C>SMA200(min_periods=120)",
}


def sma(x, n):
    return x.rolling(n, min_periods=min(n, 120)).mean()


# ---------------------------------------------------------------- 기준 점수

def score_L(P):
    C = P["C"]
    return C / sma(C, 200) - 1


def score_E2(P):
    C, Hh = P["C"], P["H"]
    um = H.universe_mask(P)
    comps = [C / sma(C, 200) - 1, C / sma(C, 50) - 1, C / sma(C, 20) - 1,
             C / Hh.rolling(252, min_periods=120).max() - 1]
    prs = [H.pct_rank(x, um & x.notna()) for x in comps]
    return (prs[0] + prs[1] + prs[2] + prs[3]) / 4.0


BASES = {
    "L": (score_L, "C/SMA200-1, SMA200 rolling(200,min_periods=120)"),
    "E2": (score_E2, "mean of pct_rank(universe_mask) of C/SMA200-1, C/SMA50-1, C/SMA20-1, C/max(H,252,mp120)-1; any NaN→NaN"),
}


# ---------------------------------------------------------------- 지표

_SOXX = None


def soxx_close():
    global _SOXX
    if _SOXX is None:
        df = pd.read_parquet(SOXX_PATH, columns=["date", "sym", "Close"])
        s = df[df["sym"] == "SOXX"].set_index("date")["Close"].sort_index()
        s.index = pd.to_datetime(s.index)
        _SOXX = s
    return _SOXX


def realized_ic(P, score, h):
    """신호일 s의 IC_h (s+1 시가 매수 → s+1+h 시가 매도). P 안의 자료만 쓴다."""
    O = P["O"]
    m = H.universe_mask(P) & score.notna() & O.shift(-1).notna()
    fr = (O.shift(-(1 + h)) / O.shift(-1) - 1).where(m)
    return H._row_spearman(score.where(m), fr)


def tri(v, cond):
    """v가 NaN이면 NaN, 아니면 cond를 1/0."""
    return cond.astype(float).where(v.notna())


def indicators(P, base_name, leaky=False):
    """{지표: (연속값 Series, on Series[1/0/NaN])}."""
    fn = BASES[base_name][0]
    score = fn(P)
    C = P["C"]
    cal = C.index
    um = H.universe_mask(P)
    out = {}
    ic10 = realized_ic(P, score, 10)
    ic20 = realized_ic(P, score, 20)
    v1 = ic10.shift(10 if leaky else 11).rolling(20, min_periods=10).mean()
    out["G1"] = (v1, tri(v1, v1 > 0))
    if leaky:
        return out
    v2 = ic20.shift(21).rolling(20, min_periods=10).mean()
    out["G2"] = (v2, tri(v2, v2 > 0))
    s50 = sma(C, 50)
    above = ((C > s50) & um & s50.notna()).sum(axis=1)
    base_n = (um & s50.notna()).sum(axis=1)
    ratio = (above / base_n).where(base_n > 0)
    out["G3"] = (ratio, tri(ratio, ratio >= 0.5))
    d10 = ratio.diff(10)
    out["G4"] = (d10, tri(d10, d10 > 0))
    qc = P["bench"]["C"]
    q50 = qc / sma(qc, 50) - 1
    out["G5"] = (q50, tri(q50, q50 > 0))
    q20 = qc / sma(qc, 20) - 1
    out["G6"] = (q20, tri(q20, q20 > 0))
    r20 = C / C.shift(20) - 1
    sd = r20.where(um).std(axis=1)
    med = sd.shift(1).rolling(252, min_periods=120).median()
    v7 = sd / med - 1
    out["G7"] = (v7, tri(v7, v7 >= 0))
    pr = H.pct_rank(score, um & score.notna())
    univ_r = r20.where(um).mean(axis=1)
    top_r = r20.where((pr > 0.9) & um).mean(axis=1)
    v8 = top_r - univ_r
    out["G8"] = (v8, tri(v8, v8 > 0))
    sx = soxx_close().reindex(cal)
    v9 = sx / sma(sx, 20) - 1
    out["G9"] = (v9, tri(v9, v9 > 0))
    g1, g5 = out["G1"][1], out["G5"][1]
    v10 = (g1 * g5).where(g1.notna() & g5.notna())
    out["G10"] = (v10, v10)
    # extra
    top_lag = (pr > 0.9).shift(20, fill_value=False)
    um_lag = um.shift(20, fill_value=False)
    vlag = r20.where(top_lag).mean(axis=1) - r20.where(um_lag).mean(axis=1)
    out["extra_G8lag"] = (vlag, tri(vlag, vlag > 0))
    q200 = qc / sma(qc, 200) - 1
    out["extra_QQQ200"] = (q200, tri(q200, q200 > 0))
    return out


def truncate(P, t):
    cut = {k: v.loc[:t] for k, v in P.items() if k != "bench"}
    cut["bench"] = {k: v.loc[:t] for k, v in P["bench"].items()}
    return cut


def _same(a, b):
    a, b = a.astype(float), b.reindex(a.index).astype(float)
    ok = (a.isna() & b.isna()) | np.isclose(a.fillna(1e300), b.fillna(-1e300), rtol=1e-9, atol=1e-12)
    return bool(ok.all()), int((~ok).sum())


def truncation_test(P, base_name, cutoffs, leaky=False):
    full = indicators(P, base_name, leaky=leaky)
    bad = {k: 0 for k in full}
    for c in cutoffs:
        part = indicators(truncate(P, c), base_name, leaky=leaky)
        for k in full:
            for j in (0, 1):
                ok, n = _same(full[k][j].loc[:c], part[k][j])
                if not ok:
                    bad[k] += n
    return {k: (bad[k] == 0) for k in bad}, bad


# ---------------------------------------------------------------- 채점

def split_ic(score, ind, wname):
    """H.evaluate와 같은 방식의 IC·상위20%를 지표 on/off로 나눈다."""
    P = H.load_panel()
    score = score.reindex(index=P["C"].index, columns=P["C"].columns)
    mask = H.eligible() & score.notna()
    pr = H.pct_rank(score, mask)
    excl = H.EXCLUDE.get(wname, set())
    cols = [c for c in P["C"].columns if c not in excl]
    out = {}
    for h in H.HORIZONS:
        dates = H.window_dates(wname, h)
        if len(dates) < 10:
            continue
        fr = H.forward(h).loc[dates, cols].where(mask.loc[dates, cols])
        sc = pr.loc[dates, cols]
        ic = H._row_spearman(sc, fr)
        mean_all = fr.mean(axis=1)
        top = fr.where(sc > 0.8).mean(axis=1) - mean_all
        g = ind.reindex(dates)
        d = {"all_ic": round(float(ic.dropna().mean()), 4), "nan_days": int(g.isna().sum())}
        for state, sel in (("on", g == 1), ("off", g == 0)):
            x = ic[sel].dropna()
            d[f"{state}_ic"] = round(float(x.mean()), 4) if len(x) else None
            d[f"{state}_t"] = round(H.newey_west_t(x, h), 2) if len(x) >= 5 else None
            d[f"{state}_days"] = int(len(x))
            tp = top[sel].dropna()
            d[f"{state}_top20"] = round(float(tp.mean()) * 100, 3) if len(tp) else None
        on = (g == 1).astype(int)
        d["on_spells"] = int(((on == 1) & (on.shift(1, fill_value=0) == 0)).sum())
        d["on_ratio"] = round(float((g == 1).sum() / max(g.notna().sum(), 1)) * 100, 1)
        out[f"h{h}"] = d
    return out


def weekly_switch(pr, ind, start, end, mask, umask, lo=0.90):
    """H.weekly_portfolio 복제. 주 첫 거래일 직전 종가에 ind==1이면 pr>lo & mask, 아니면(0·NaN) umask 전체."""
    P = H.load_panel()
    O, C = P["O"], P["C"]
    cal = C.index
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(days) < 3:
        return {}
    week = pd.Series(days.to_period("W-SUN"), index=days)
    firsts = [d for d, w in zip(days, week) if d == week[week == w].index[0]]
    eq, curve, picks, ons = 1.0, [], [], 0
    for i, w0 in enumerate(firsts):
        w1 = firsts[i + 1] if i + 1 < len(firsts) else None
        prev = cal[cal.get_loc(w0) - 1]
        on = ind.get(prev, np.nan) == 1
        ons += int(on)
        if on:
            row = pr.loc[prev]
            names = row[(row > lo) & mask.loc[prev]].index
        else:
            names = umask.columns[umask.loc[prev].to_numpy(bool)]
        names = [n for n in names if np.isfinite(O.at[w0, n])]
        picks.append(len(names))
        span = days[(days >= w0) & ((days < w1) if w1 is not None else (days <= days[-1]))]
        if not names:
            curve += [eq] * len(span)
            continue
        entry = O.loc[w0, names]
        for d in span:
            curve.append(eq * float((C.loc[d, names] / entry).mean(skipna=True)))
        if w1 is not None:
            ex = O.loc[w1, names].fillna(C.loc[span[-1], names])
            eq *= float((ex / entry).mean(skipna=True))
        else:
            eq = curve[-1]
    curve = pd.Series(curve)
    mdd = float((curve / curve.cummax() - 1).min()) if len(curve) else float("nan")
    return {"ret": round((eq - 1) * 100, 2), "mdd": round(mdd * 100, 2), "avg_n": round(float(np.mean(picks)), 1),
            "weeks_on": ons, "weeks": len(firsts)}


def window_bounds(wname):
    a, b = H.WINDOWS[wname]
    last = H.load_panel()["C"].index[-1].strftime("%Y-%m-%d")
    return a, min(b, last)


def portfolios(score, ind, wname, drop=()):
    P = H.load_panel()
    score = score.reindex(index=P["C"].index, columns=P["C"].columns)
    elig = H.eligible().copy()
    for c in drop:
        elig[c] = False
    mask = elig & score.notna()
    pr = H.pct_rank(score, mask)
    a, b = window_bounds(wname)
    return weekly_switch(pr, ind, a, b, mask, elig)


def monthly_on(ind, start="2025-01-01"):
    s = ind.loc[start:].dropna()
    m = s.groupby(s.index.to_period("M")).mean()
    return {str(k): round(float(v) * 100) for k, v in m.items()}


# ---------------------------------------------------------------- 실행

def main():
    t0 = time.time()
    P = H.load_panel()
    cal = P["C"].index
    rows, table = [], []
    meta = {"checks": {}}

    # 채점기 복제 확인: 항상 on = H.weekly_portfolio(S), 항상 off = H.bench_weekly(universe)
    sL = score_L(P)
    mask = H.eligible() & sL.notna()
    pr = H.pct_rank(sL, mask)
    for w in ("front", "back"):
        a, b = window_bounds(w)
        on_all = pd.Series(1.0, index=cal)
        off_all = pd.Series(0.0, index=cal)
        r1 = weekly_switch(pr, on_all, a, b, mask, H.eligible())
        r2 = H.weekly_portfolio(pr, 0.90, a, b, mask)
        r3 = weekly_switch(pr, off_all, a, b, mask, H.eligible())
        r4 = H.bench_weekly(a, b, "universe")
        ok = (r1["ret"], r1["mdd"]) == (r2["ret"], r2["mdd"]) and (r3["ret"], r3["mdd"]) == (r4["ret"], r4["mdd"])
        meta["checks"][f"weekly_replica_{w}"] = bool(ok)
        print("weekly replica", w, ok, r1, r2, r3, r4)

    cutoffs = [d for d in pd.date_range("2024-02-01", "2026-10-01", freq="2MS")]
    cutoffs = [cal[cal <= d][-1] for d in cutoffs] + [pd.Timestamp("2026-07-22"), pd.Timestamp("2026-09-30")]
    meta["trunc_cutoffs"] = [c.strftime("%Y-%m-%d") for c in cutoffs]

    for bname, (fn, bspec) in BASES.items():
        nl = H.no_lookahead(fn)
        score = fn(P)
        res = H.evaluate(score)
        print(H.summary_line(f"base_{bname}", res), "no_lookahead", nl)
        rows.append({"name": f"base_{bname}", "spec": bspec, "result": res, "no_lookahead": nl})

        tpass, tbad = truncation_test(P, bname, cutoffs)
        leak_pass, leak_bad = truncation_test(P, bname, cutoffs, leaky=True)
        meta["checks"][f"trunc_{bname}"] = tpass
        meta["checks"][f"trunc_{bname}_leaky_G1_detected"] = (not leak_pass["G1"], leak_bad["G1"])
        print("truncation", bname, tpass, "| leaky G1 detected:", not leak_pass["G1"], leak_bad["G1"])

        ind = indicators(P, bname)
        # 비교 기준: 항상 S, 항상 전체 (그리고 y2024 DRUG 제외 판)
        always = {}
        for w in ("front", "back", "y2025", "y2024", "late"):
            always[w] = {"S": portfolios(score, pd.Series(1.0, index=cal), w),
                         "universe": portfolios(score, pd.Series(0.0, index=cal), w)}
        always["y2024_exDRUG"] = {"S": portfolios(score, pd.Series(1.0, index=cal), "y2024", drop=("DRUG",)),
                                  "universe": portfolios(score, pd.Series(0.0, index=cal), "y2024", drop=("DRUG",))}
        meta[f"always_{bname}"] = always

        for gname in GRID + EXTRA:
            v, on = ind[gname]
            r = {"spec": SPEC[gname], "monthly_on_pct": monthly_on(on)}
            for w in WINDOWS:
                r[w] = split_ic(score, on, w)
                r[w]["port_switch"] = portfolios(score, on, w)
                r[w]["port_always_S"] = always[w]["S"]
                r[w]["port_always_universe"] = always[w]["universe"]
            r["y2024_exDRUG_port_switch"] = portfolios(score, on, "y2024", drop=("DRUG",))
            gated = (lambda f, b_, g_: (lambda PP: f(PP).add(10.0 * indicators(PP, b_)[g_][1], axis=0)))(fn, bname, gname)
            nlg = bool(H.no_lookahead(gated)) and bool(tpass[gname]) and bool(nl)
            name = f"{bname}×{gname}"
            rows.append({"name": name, "spec": f"base {bname} ({bspec}) | gate {gname}: {SPEC[gname]}",
                         "result": r, "no_lookahead": nlg})
            f20, b20 = r["front"]["h20"], r["back"]["h20"]
            line = {
                "name": name,
                "front_on_ic20": f20["on_ic"], "front_off_ic20": f20["off_ic"], "front_on_ratio": f20["on_ratio"],
                "back_on_ic20": b20["on_ic"], "back_off_ic20": b20["off_ic"], "back_on_ratio": b20["on_ratio"],
                "front_on_ic10": r["front"]["h10"]["on_ic"], "front_off_ic10": r["front"]["h10"]["off_ic"],
                "back_on_ic10": r["back"]["h10"]["on_ic"], "back_off_ic10": r["back"]["h10"]["off_ic"],
                "y2025_on_ic20": r["y2025"]["h20"]["on_ic"], "y2025_off_ic20": r["y2025"]["h20"]["off_ic"],
                "y2024_on_ic20": r["y2024"]["h20"]["on_ic"], "y2024_off_ic20": r["y2024"]["h20"]["off_ic"],
                "late_on_ic10": r["late"]["h10"]["on_ic"], "late_off_ic10": r["late"]["h10"]["off_ic"],
                "late_on_ratio": r["late"]["h10"]["on_ratio"],
                "front_on_top20": f20["on_top20"], "front_off_top20": f20["off_top20"],
                "back_on_top20": b20["on_top20"], "back_off_top20": b20["off_top20"],
                "front_on_t20": f20["on_t"], "back_on_t20": b20["on_t"],
                "front_spells": f20["on_spells"], "back_spells": b20["on_spells"],
                "sw_front": r["front"]["port_switch"].get("ret"), "sw_front_mdd": r["front"]["port_switch"].get("mdd"),
                "sw_back": r["back"]["port_switch"].get("ret"), "sw_back_mdd": r["back"]["port_switch"].get("mdd"),
                "sw_y2025": r["y2025"]["port_switch"].get("ret"), "sw_y2024": r["y2024"]["port_switch"].get("ret"),
                "sw_y2024_exDRUG": r["y2024_exDRUG_port_switch"].get("ret"),
                "sw_late": r["late"]["port_switch"].get("ret"),
                "weeks_on_front": r["front"]["port_switch"].get("weeks_on"),
                "weeks_on_back": r["back"]["port_switch"].get("weeks_on"),
                "no_lookahead": nlg,
            }
            table.append(line)
            print(f"{name:18s} F on {f20['on_ic']} ({f20['on_days']}d) off {f20['off_ic']} ({f20['off_days']}d) | "
                  f"B on {b20['on_ic']} ({b20['on_days']}d) off {b20['off_ic']} ({b20['off_days']}d) | "
                  f"25 on {line['y2025_on_ic20']} off {line['y2025_off_ic20']} | 24 on {line['y2024_on_ic20']} off {line['y2024_off_ic20']} | "
                  f"sw F {line['sw_front']} B {line['sw_back']} 25 {line['sw_y2025']} 24 {line['sw_y2024']} | nl {nlg}")
        print("always", bname, json.dumps({w: {k: v.get("ret") for k, v in d.items()} for w, d in always.items()}))

    path = H.save_results("regime", rows)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["table"] = table
    doc["meta"] = meta
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print("saved", path, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
