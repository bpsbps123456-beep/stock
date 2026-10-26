"""V1 독립 재계산: FINAL_E14s10_HYB를 원자료(가격 parquet, 보유 PDF JSON)에서 처음부터 다시 구현한다.

harness.py / fam_*.py / final.py는 import하지 않는다. 사양(과제 문장)대로 구현:
- 원점수(t) = 그날 eligible 단면 백분위 평균{C/SMA200-1, C/SMA50-1, C/252일 최고 고가-1, 클레노우126}
  클레노우126 = 최근 126종가 ln가격 회귀 기울기 b로 (exp(252b)-1)*R^2.
  SMA = rolling(n, min_periods=min(n,120)), 252일 고가 min_periods 120.
- 점수 = 원점수의 최근 10행 평균(min_periods 1), 그날 원점수 있을 때만.
- 등급 = 점수의 그날 백분위 >0.9 S, >0.7 A, >0.3 B, >0.1 C, 나머지 D.
  관문: S/A는 C>SMA50 & SMA50>SMA200 아니면 B, 이어서 C<SMA200 & SMA50<SMA200이면 D.
- eligible(t) = 보유(t) ∩ 봉 130개 이상 ∩ t+1 시가 있음.
  보유(t) = 펀드별 t 이하 최신 PDF의 'XXX US EQUITY' 합집합(별칭 적용).
  첫 스냅숏(2026-01-02) 이전: ALL150(가격 캐시 150종목 전체, harness 관례) / NONE(보유 없음, 사양 엄격) 두 변형.
- 채점: t 종가 신호 -> t+1 시가 매수 -> t+1+h 시가 매도. 거래일 달력 = QQQ 봉 날짜.
- 구간 판정 A(신호일>=시작 & 매도일<=끝, 기본), B(진입일>=시작), C(매도일만 구간 안)를 함께 계산.
- 주간 보유: 관례 H(구간 안 날짜를 달력 주로 묶어 각 주 첫 날 시가 진입, 마지막 묶음은 구간 끝 종가 청산,
  MDD는 일별 종가 평가) / 엄격 F(달력 주 첫 거래일 >= 시작이고 다음 주 첫 시가 <= 끝인 완주 주만).
실행: PYTHONIOENCODING=utf-8 py verify_v1_recompute.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
PX = ROOT / "data/derived/analysis/rsi_breakout/cache/1d_long.parquet"
HOLD = ROOT / "data/normalized/etf_holdings"
RES = ROOT / "data/derived/analysis/trend_grade/results"
OUT_JSON = RES / "verify_v1_recompute.json"
OUT_TXT = RES / "verify_v1_recompute.txt"

ALIAS = {"GOOGL": "GOOG", "SATS": "ECHO", "SKHYV": "SKHY", "MOG/A": "MOG-A"}
WINDOWS = {
    "front": ("2026-01-02", "2026-05-29"),
    "back": ("2026-06-01", "2026-10-09"),
    "y2025": ("2025-01-01", "2025-12-31"),
    "late": ("2026-09-01", "2026-10-09"),
}
TARGET = {
    "IC": {"front_IC20": (0.247, 4.68), "back_IC20": (-0.159, -1.5), "late_IC10": (0.167, None),
           "y2025_IC20": (-0.010, None)},
    "excess": {"front": {"S": 7.14, "A": 4.19, "B": 1.71, "C": -6.31, "D": -8.73},
               "back": {"S": -3.34, "A": -1.53, "B": -1.08, "C": -1.08, "D": 5.89}},
    "weekly": {"front": {"S": 76.1, "S_MDD": -21.3, "ALL": 61.5, "QQQ": 19.1},
               "back": {"S": -4.4, "S_MDD": -36.0, "ALL": 4.3, "QQQ": 1.9}},
    "shock": {"S": -42.3, "A": -31.9, "B": -16.4, "C": -14.3, "D": 9.1, "ALL": -16.8, "QQQ": -10.3, "S_n": 7},
    "monthly_IC20": {"2026-01": 0.36, "2026-02": 0.01, "2026-03": 0.34, "2026-04": 0.25, "2026-05": 0.35,
                     "2026-06": -0.23, "2026-07": -0.34, "2026-08": -0.03, "2026-09": 0.24,
                     "2025-02": -0.25, "2025-04": -0.33, "2025-05": -0.12},
}

# ---------------------------------------------------------------- data
px = pd.read_parquet(PX)
px["ts"] = pd.to_datetime(px["ts"]).dt.normalize()
px = px.drop_duplicates(["ts", "sym"], keep="last")
O = px.pivot(index="ts", columns="sym", values="Open").sort_index()
H = px.pivot(index="ts", columns="sym", values="High").sort_index()
C = px.pivot(index="ts", columns="sym", values="Close").sort_index()
cal = O.index[O["QQQ"].notna()]
extra_dates = O.index.difference(cal)
O, H, C = O.reindex(cal), H.reindex(cal), C.reindex(cal)
STOCKS = [s for s in O.columns if s != "QQQ"]
pos = {d: i for i, d in enumerate(cal)}


def per_symbol_features():
    sma50, sma200, hi252, clen, nbar = {}, {}, {}, {}, {}
    x = np.arange(126, dtype=float)
    xm = x - x.mean()
    sxx = (xm ** 2).sum()
    for s in O.columns:
        c = C[s].dropna()
        h = H[s].reindex(c.index)
        sma50[s] = c.rolling(50, min_periods=50).mean()
        sma200[s] = c.rolling(200, min_periods=120).mean()
        hi252[s] = h.rolling(252, min_periods=120).max()
        nbar[s] = pd.Series(np.arange(1, len(c) + 1), index=c.index)
        y = np.log(c.values)
        out = np.full(len(c), np.nan)
        if len(c) >= 126:
            w = np.lib.stride_tricks.sliding_window_view(y, 126)
            ym = w - w.mean(axis=1, keepdims=True)
            b = (ym * xm).sum(axis=1) / sxx
            ssy = (ym ** 2).sum(axis=1)
            r2 = np.where(ssy > 0, (b ** 2) * sxx / ssy, 0.0)
            out[125:] = (np.exp(252 * b) - 1) * r2
        clen[s] = pd.Series(out, index=c.index)
    f = lambda d: pd.DataFrame(d).reindex(cal)
    return f(sma50), f(sma200), f(hi252), f(clen), f(nbar)


SMA50, SMA200, HI252, CLEN, NBAR = per_symbol_features()


def load_holdings():
    snaps = {"456600": {}, "2ETFQ1": {}}
    for fp in sorted(glob.glob(str(HOLD / "*.json"))):
        name = os.path.basename(fp)
        fund = name[11:-5]
        if fund not in snaps:
            continue
        d = json.load(open(fp, encoding="utf-8"))
        syms = set()
        for h in d.get("holdings", []):
            code = (h.get("code") or "").strip()
            if code.endswith(" US EQUITY"):
                t = code[: -len(" US EQUITY")].strip()
                syms.add(ALIAS.get(t, t))
        snaps[fund][pd.Timestamp(name[:10])] = syms
    return snaps


SNAPS = load_holdings()
FIRST_SNAP = min(min(v) for v in SNAPS.values())
UNMAPPED = sorted({t for v in SNAPS.values() for ss in v.values() for t in ss} - set(STOCKS))


def universe(t, pre_mode):
    out, found = set(), False
    for d in SNAPS.values():
        ks = [k for k in d if k <= t]
        if ks:
            found = True
            out |= d[max(ks)]
    if not found:
        return set(STOCKS) if pre_mode == "ALL150" else set()
    return out & set(STOCKS)


def build(pre_mode):
    nxt_open = O.shift(-1)
    elig = pd.DataFrame(False, index=cal, columns=STOCKS)
    for t in cal:
        u = universe(t, pre_mode)
        if u:
            cols = sorted(u)
            ok = (NBAR.loc[t, cols] >= 130) & nxt_open.loc[t, cols].notna()
            elig.loc[t, cols] = ok.values
    feats = [C[STOCKS] / SMA200[STOCKS] - 1, C[STOCKS] / SMA50[STOCKS] - 1,
             C[STOCKS] / HI252[STOCKS] - 1, CLEN[STOCKS]]
    raw = (sum(f.where(elig).rank(axis=1, pct=True) for f in feats) / 4.0).where(elig)
    smooth = raw.rolling(10, min_periods=1).mean().where(raw.notna())
    pct = smooth.rank(axis=1, pct=True)
    g0 = pd.DataFrame(np.select([pct > 0.9, pct > 0.7, pct > 0.3, pct > 0.1, pct.notna()],
                                ["S", "A", "B", "C", "D"], default=""), index=cal, columns=STOCKS)
    c, s50, s200 = C[STOCKS], SMA50[STOCKS], SMA200[STOCKS]
    up = (c > s50) & (s50 > s200)
    down = (c < s200) & (s50 < s200)
    g = g0.mask(g0.isin(["S", "A"]) & ~up, "B")
    g = g.mask((g != "") & down, "D")
    # 관문 순서 반대(D 먼저, 다음 S/A 강등)와 비교
    g_alt = g0.mask((g0 != "") & down, "D")
    g_alt = g_alt.mask(g_alt.isin(["S", "A"]) & ~up, "B")
    order_diff = int(((g != g_alt) & (g != "")).sum().sum())
    gate_stats = {"S/A->B 강등": int((g0.isin(["S", "A"]) & ~up & ~down).sum().sum()),
                  "->D 강제(전 등급)": int(((g0 != "") & (g0 != "D") & down).sum().sum()),
                  "관문 순서 바꿨을 때 달라지는 칸": order_diff}
    return {"elig": elig, "raw": raw, "smooth": smooth, "g": g, "gate": gate_stats}


FWD = {h: (O.shift(-(1 + h)) / O.shift(-1) - 1)[STOCKS] for h in (10, 20)}


def in_window(t, h, w, mode):
    a, b = pd.Timestamp(WINDOWS[w][0]), pd.Timestamp(WINDOWS[w][1])
    i = pos[t]
    if i + 1 + h >= len(cal):
        return False
    entry, sell = cal[i + 1], cal[i + 1 + h]
    if mode == "A":
        return t >= a and sell <= b
    if mode == "B":
        return entry >= a and sell <= b
    return a <= sell <= b


def nw_t(x, lag):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    v = (e @ e) / n
    for l in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def daily_ic(smooth, h):
    ics = {}
    for t in cal:
        s, r = smooth.loc[t], FWD[h].loc[t]
        m = s.notna() & r.notna()
        if m.sum() >= 5:
            ics[t] = s[m].rank().corr(r[m].rank())
    return pd.Series(ics)


def ic_stats(ics, h, w, mode):
    sel = ics[[in_window(t, h, w, mode) for t in ics.index]]
    if not len(sel):
        return {"IC": None, "NW_t": None, "n_dates": 0}
    return {"IC": round(float(sel.mean()), 4), "NW_t": round(nw_t(sel.values, h), 2), "n_dates": int(len(sel)),
            "first_signal": str(sel.index.min().date()), "last_signal": str(sel.index.max().date())}


def grade_excess(g, h, w, mode):
    rows = []
    for t in cal:
        if not in_window(t, h, w, mode):
            continue
        gt, r = g.loc[t], FWD[h].loc[t]
        m = (gt != "") & r.notna()
        if m.sum() >= 5:
            rows.append((r[m] - r[m].mean()).groupby(gt[m]).mean())
    df = pd.DataFrame(rows)
    return {k: round(float(df[k].mean() * 100), 2) for k in "SABCD" if k in df}


def _mdd(curve):
    c = np.asarray(curve, float)
    return round(float((c / np.maximum.accumulate(c) - 1).min()) * 100, 2)


def weekly(B, w, conv):
    """conv H: harness 관례 재현, F: 사양 엄격(완주 주만)."""
    g, elig = B["g"], B["elig"]
    a, b = pd.Timestamp(WINDOWS[w][0]), pd.Timestamp(WINDOWS[w][1])
    segs = []
    if conv == "H":
        days = cal[(cal >= a) & (cal <= b)]
        reb = pd.Series(days, index=days).groupby(days.to_period("W-SUN")).min().tolist()
        for k, e in enumerate(reb):
            x = reb[k + 1] if k + 1 < len(reb) else None
            segs.append((cal[pos[e] - 1], e, x, days[(days >= e) & ((days < x) if x is not None else (days <= b))]))
    else:
        starts = pd.Series(cal, index=cal).groupby(cal.to_period("W-SUN")).min().tolist()
        for k in range(1, len(starts) - 1):
            e, x = starts[k], starts[k + 1]
            t = cal[pos[e] - 1]
            if t >= a and x <= b:  # 신호일 >= 시작, 매도(다음 주 첫 시가) <= 끝
                segs.append((t, e, x, cal[(cal >= e) & (cal < x)]))
    out, seg_log = {}, []
    for who in ("S", "ALL", "QQQ"):
        eq, closes, pts = 1.0, [1.0], [1.0]
        for t, e, x, sd in segs:
            gs = g.loc[t]
            basket = (list(gs.index[gs == "S"]) if who == "S" else
                      list(gs.index[elig.loc[t]]) if who == "ALL" else ["QQQ"])
            if not basket:
                closes += [eq] * len(sd)
                pts.append(eq)
                if who == "S":
                    seg_log.append((str(t.date()), str(e.date()), str(x.date()) if x is not None else "close", 0, 0.0))
                continue
            o0 = O.loc[e, basket]
            for d in sd:
                closes.append(eq * float((C.loc[d, basket] / o0).mean()))
            end = O.loc[x, basket] if x is not None else C.loc[sd[-1], basket]
            r = float((end / o0).mean())
            eq *= r
            pts.append(eq)
            if who == "S":
                seg_log.append((str(t.date()), str(e.date()), str(x.date()) if x is not None else f"{sd[-1].date()} close",
                                len(basket), round((r - 1) * 100, 2)))
        out[who] = {"total_pct": round((eq - 1) * 100, 2), "MDD_daily_close_pct": _mdd(closes),
                    "MDD_week_points_pct": _mdd(pts)}
    out["n_segments"] = len(segs)
    out["segments_S"] = seg_log
    return out


def shock(B, t0="2026-06-22", t1="2026-07-29"):
    g, elig = B["g"], B["elig"]
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    gs = g.loc[t0]
    r = C.loc[t1] / C.loc[t0] - 1
    row = {k: {"n": int((gs == k).sum()), "ret_pct": round(float(r[gs.index[gs == k]].mean()) * 100, 2)} for k in "SABCD"}
    row["S"]["syms"] = sorted(gs.index[gs == "S"])
    al = gs.index[elig.loc[t0]]
    row["ALL"] = {"n": int(len(al)), "ret_pct": round(float(r[al].mean()) * 100, 2)}
    row["QQQ"] = round(float(r["QQQ"]) * 100, 2)
    return row


def main():
    rep = {"meta": {"calendar": "QQQ 거래일", "n_cal": len(cal), "dates_not_in_QQQ": len(extra_dates),
                    "n_stocks": len(STOCKS), "first_snapshot": str(FIRST_SNAP.date()),
                    "holdings_US_codes_not_in_price_cache": UNMAPPED}, "target": TARGET}
    builds = {p: build(p) for p in ("ALL150", "NONE")}
    for p, B in builds.items():
        ic20, ic10 = daily_ic(B["smooth"], 20), daily_ic(B["smooth"], 10)
        r = {"gate": B["gate"],
             "n_eligible": {d: int(B["elig"].loc[pd.Timestamp(d)].sum()) for d in ("2025-12-31", "2026-01-02")}}
        for mode in ("A", "B", "C"):
            r[f"IC_{mode}"] = {f"{w}_IC{10 if w == 'late' else 20}":
                               ic_stats(ic10 if w == "late" else ic20, 10 if w == "late" else 20, w, mode)
                               for w in WINDOWS}
            r[f"excess_{mode}"] = {w: grade_excess(B["g"], 20, w, mode) for w in ("front", "back")}
        r["weekly_H"] = {w: weekly(B, w, "H") for w in ("front", "back")}
        r["weekly_F"] = {w: weekly(B, w, "F") for w in ("front", "back")}
        r["shock"] = shock(B)
        r["monthly_IC20"] = {str(k): round(float(v), 3) for k, v in
                             ic20.groupby(ic20.index.to_period("M")).mean().items() if str(k) >= "2025-01"}
        rep[p] = r
    t = pd.Timestamp("2025-12-31")
    gs = builds["ALL150"]["g"].loc[t]
    s1231 = sorted(gs.index[gs == "S"])
    u0102 = universe(pd.Timestamp("2026-01-02"), "NONE")
    rep["diag_2025-12-31"] = {"S_ALL150": s1231, "S_not_in_2026-01-02_holdings": [s for s in s1231 if s not in u0102]}
    OUT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # ------------------------------------------------------------ 표
    A, N = rep["ALL150"], rep["NONE"]
    L = ["V1 독립 재계산 (FINAL_E14s10_HYB) — 목표값 vs 재계산", ""]
    L.append("[1] IC (구간 A: 신호일>=시작 & 매도일<=끝)   목표 | ALL150 | NONE(사양 엄격) | 구간B(ALL150) | 구간C(ALL150)")
    for k, (tv, tt) in TARGET["IC"].items():
        a, n, b, c = A["IC_A"][k], N["IC_A"][k], A["IC_B"][k], A["IC_C"][k]
        L.append(f"  {k:11s} {tv:+.3f} (t {tt if tt is not None else '-'}) | {a['IC']:+.4f} (t {a['NW_t']}, n {a['n_dates']}) | "
                 f"{(n['IC'] if n['IC'] is not None else float('nan')):+.4f} (t {n['NW_t']}) | {b['IC']:+.4f} (t {b['NW_t']}) | {c['IC']:+.4f} (t {c['NW_t']})")
    L.append("")
    L.append("[2] 등급별 h20 초과수익 %p (구간 A)   목표 | ALL150 | NONE")
    for w in ("front", "back"):
        for k in "SABCD":
            L.append(f"  {w:5s} {k}  {TARGET['excess'][w][k]:+6.2f} | {A['excess_A'][w][k]:+6.2f} | {N['excess_A'][w][k]:+6.2f}")
    L.append("")
    L.append("[3] 주간 보유 총수익 % (MDD는 S, 일별 종가 평가)")
    L.append("  관례H = 구간 안 날짜를 달력 주로 묶음(front 첫 묶음 1/2 금 1일, 마지막 묶음 5/29 종가 청산)")
    L.append("  엄격F = 달력 주 첫 거래일 진입, 신호일>=시작, 다음 주 첫 시가<=끝(완주 주만)")
    L.append("  구간  항목   목표 | H·ALL150 | H·NONE | F·ALL150 | F·NONE")
    for w in ("front", "back"):
        tg = TARGET["weekly"][w]
        for who, key in (("S", "total_pct"), ("S_MDD", "MDD_daily_close_pct"), ("ALL", "total_pct"), ("QQQ", "total_pct")):
            src = "S" if who == "S_MDD" else who
            vals = [X[f"weekly_{cv}"][w][src][key] for cv, X in (("H", A), ("H", N), ("F", A), ("F", N))]
            L.append(f"  {w:5s} {who:5s} {tg[who]:+6.1f} | " + " | ".join(f"{v:+6.2f}" for v in vals))
    L.append("  front S 묶음(관례H·ALL150) 첫 2개·마지막: " + json.dumps(A["weekly_H"]["front"]["segments_S"][:2] +
                                                       A["weekly_H"]["front"]["segments_S"][-1:], ensure_ascii=False))
    L.append("  front S 묶음(관례H·NONE) 첫 2개: " + json.dumps(N["weekly_H"]["front"]["segments_S"][:2], ensure_ascii=False))
    L.append("")
    L.append("[4] 7월 충격 (6/22 종가 등급 -> 7/29 종가)   목표 | ALL150 | NONE")
    for k in ("S", "A", "B", "C", "D", "ALL"):
        L.append(f"  {k:3s} {TARGET['shock'][k]:+6.1f} | {A['shock'][k]['ret_pct']:+6.2f} (n {A['shock'][k]['n']}) | {N['shock'][k]['ret_pct']:+6.2f}")
    L.append(f"  QQQ {TARGET['shock']['QQQ']:+6.1f} | {A['shock']['QQQ']:+6.2f} | {N['shock']['QQQ']:+6.2f}   S: {A['shock']['S']['syms']}")
    L.append("")
    L.append("[5] 월별 IC20(신호월)   목표 | ALL150 | NONE")
    for m, tv in TARGET["monthly_IC20"].items():
        L.append(f"  {m} {tv:+.2f} | {A['monthly_IC20'].get(m, float('nan')):+.3f} | {N['monthly_IC20'].get(m, float('nan')):+.3f}")
    L.append("")
    L.append(f"[6] 관문: {A['gate']}")
    L.append(f"[7] 2025-12-31 신호: eligible ALL150 {A['n_eligible']['2025-12-31']} / NONE {N['n_eligible']['2025-12-31']}, "
             f"2026-01-02 eligible {A['n_eligible']['2026-01-02']}; 12-31 S {len(s1231)}종목 중 1/2 보유에 없는 것 "
             f"{rep['diag_2025-12-31']['S_not_in_2026-01-02_holdings']}")
    OUT_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
