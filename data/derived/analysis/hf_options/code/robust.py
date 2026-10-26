"""갈래 robust(BRIEF E2·E3): 격자 확인 칸 견고성 + T1·T2 순위 기반 민감도(사후 변형) + 2026 기본 비율.

(a) grid.json 확인 칸 4개(고점 3: eq_pcr5+&shvr_ai5-, skew+&shvr_ai5-, skew+&nq_am_chg4- h21 M / 바닥 1: es_lev_net_oi+ h21 R)
    - front·back 신호일 날짜, 5거래일 이내 간격을 한 사건으로 묶은 사건별 결과, 사건 하나 빼기
    - 문턱 ±0.05(쌍 0.65/0.75, 단독 0.75/0.85), h 5·10, 반대 결과(R↔M), SOXX, 2025(shift_p)
    - 신호 묶음 쏠림: 확인 칸끼리 신호일 겹침, 고른 상위 20칸의 front 신호일 집합 고유 개수
(b) T1'·T2'(사후 변형, 판정에 쓰지 않음): grid.composite 두 합성 → 각 합성의 250개 관측 이동 백분위(자기 포함, 과거만)
    T1' = 둘 다 ≥ 0.7, T2' = 둘 다 ≤ 0.3. 사전 등록 T2의 2025 신호일과 그 뒤 결과.
(c) 기간별 기본 비율(평균 R21·M21, P(M21 ≤ −5%)).
결과: results/robust.json. 표 원문은 표준 출력(마크다운) → results/robust.md 작성에 그대로 옮긴다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import grid as G
import harness as H

P = H.panel()
IDX = P.index
N = len(IDX)
Y = {"QQQ": H.outcomes("QQQ").reindex(IDX), "SOXX": H.outcomes("SOXX").reindex(IDX)}
PX = H.prices().reindex(IDX)
PERS = ("front", "back", "y2025")
EP_GAP = 5

CELLS = [
    {"cell": "eq_pcr5+&shvr_ai5-", "side": "top", "y": "M", "h": 21},
    {"cell": "skew+&shvr_ai5-", "side": "top", "y": "M", "h": 21},
    {"cell": "skew+&nq_am_chg4-", "side": "top", "y": "M", "h": 21},
    {"cell": "es_lev_net_oi+", "side": "bottom", "y": "R", "h": 21},
]
KNOWN = {
    "eq_pcr5": "CBOE 주식 옵션 풋/콜 거래량 5일 합 비율, 장 마감 뒤 공개 → 측정일 다음 거래일 시가부터(일별, 3거래일까지 이어 씀)",
    "shvr_ai5": "FINRA 장외 공매도 거래량 비율(AI 115종목 단순 평균) 5일 평균, 측정일 17시대 ET 게시 → 다음 거래일 시가부터(개장 전 게시 아닌 4일은 하루 늦춤)",
    "skew": "CBOE SKEW 지수 종가(Yahoo) → 다음 거래일 시가부터",
    "nq_am_chg4": "CFTC TFF 나스닥100 선물 자산운용 순포지션÷OI의 4주 변화, 화요일 기준·금요일 15:30 ET 공개 → 공개 다음 거래일 시가부터(15거래일까지)",
    "es_lev_net_oi": "CFTC TFF S&P500 선물 레버리지 펀드 순포지션÷OI, 같은 공개 시각 → 공개 다음 거래일 시가부터(15거래일까지)",
}


def parse(cell: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in (re.fullmatch(r"(.+)([+-])", p) for p in cell.split("&"))]


def make_sig(cell: str, th: float | None = None) -> np.ndarray:
    parts = parse(cell)
    if th is None:
        th = G.TH1 if len(parts) == 1 else G.TH2
    nan, conds = np.zeros(N, bool), []
    for s, sg in parts:
        x = P[s].to_numpy()
        nan |= np.isnan(x)
        with np.errstate(invalid="ignore"):
            conds.append(x >= th if sg == "+" else x <= 1 - th)
    return np.where(nan, np.nan, np.logical_and.reduce(conds).astype(float))


def n_shifts(sig: np.ndarray, y: np.ndarray) -> int:
    m = ~np.isnan(sig) & ~np.isnan(y)
    return max(int(m.sum()) - 2 * H.MIN_SHIFT + 1, 0)


def stat(sig: np.ndarray, asset: str, col: str, per: str, side: str) -> dict:
    m = H.period_mask(IDX, per)
    s, y = np.where(m, sig, np.nan), np.where(m, Y[asset][col].to_numpy(), np.nan)
    d, t, n = H.welch(s, y)
    ok = ~np.isnan(s) & ~np.isnan(y)
    rest = ok & (s == 0)
    p = H.shift_p(s, y, "less" if side == "top" else "greater") if n >= 3 else np.nan
    return {"d": d, "t": t, "n": n, "n_rest": int(rest.sum()), "mean_sig": float(np.nanmean(y[ok & (s == 1)])) if n else np.nan,
            "mean_rest": float(y[rest].mean()) if rest.any() else np.nan, "p": p, "n_shift": n_shifts(s, y),
            "p_floor": 1 / (n_shifts(s, y) + 1) if n_shifts(s, y) else np.nan,
            "le5_sig": float((y[ok & (s == 1)] <= -5).mean()) if (col.startswith("M") and n) else np.nan,
            "le5_rest": float((y[rest] <= -5).mean()) if (col.startswith("M") and rest.any()) else np.nan}


def good(side: str, d: float) -> bool:
    return bool(d == d and ((d < 0) if side == "top" else (d > 0)))


def episodes(sig: np.ndarray, per: str, gap: int = EP_GAP) -> list[np.ndarray]:
    on = np.where(H.period_mask(IDX, per) & (np.nan_to_num(sig) == 1))[0]
    if not len(on):
        return []
    cut = np.where(np.diff(on) > gap)[0] + 1
    return np.split(on, cut)


def ep_table(sig: np.ndarray, per: str, asset: str, side: str, h: int, yname: str) -> list[dict]:
    rows = []
    col = f"{yname}{h}"
    y = Y[asset][col].to_numpy()
    for e in episodes(sig, per):
        ye = y[e]
        rows.append({"start": IDX[e[0]], "end": IDX[e[-1]], "n_days": len(e), "n_outcome": int((~np.isnan(ye)).sum()),
                     "dates": [IDX[i] for i in e],
                     f"mean_{col}": float(np.nanmean(ye)) if (~np.isnan(ye)).any() else np.nan,
                     f"worst_{col}": float(np.nanmin(ye) if side == "top" else np.nanmax(ye)) if (~np.isnan(ye)).any() else np.nan,
                     "first_R21": float(Y[asset]["R21"].iloc[e[0]]), "first_M21": float(Y[asset]["M21"].iloc[e[0]]),
                     "mean_R21": float(np.nanmean(Y[asset]["R21"].to_numpy()[e])) if (~np.isnan(Y[asset]["R21"].to_numpy()[e])).any() else np.nan,
                     "mean_M21": float(np.nanmean(Y[asset]["M21"].to_numpy()[e])) if (~np.isnan(Y[asset]["M21"].to_numpy()[e])).any() else np.nan})
    return rows


def leave_one_out(sig: np.ndarray, per: str, asset: str, side: str, col: str) -> list[dict]:
    """사건 하나의 날을 표본에서 통째로 빼고(신호·비교군 모두에서) 다시 잰다."""
    out = []
    for e in episodes(sig, per):
        s = sig.copy()
        s[e] = np.nan
        r = stat(s, asset, col, per, side)
        out.append({"drop": f"{IDX[e[0]]}~{IDX[e[-1]]}", "d": r["d"], "n": r["n"], "p": r["p"], "same_dir": good(side, r["d"])})
    return out


def windows(sig: np.ndarray, per: str, h: int) -> int:
    """결과 창이 겹치지 않는 묶음 수(간격 ≥ h거래일이면 새 묶음)."""
    return len(episodes(sig, per, gap=h - 1))


def fmt(v, nd=2, sign=True):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fp(p):
    return "–" if p is None or p != p else f"{p:.3f}"


# ---------------- (a) 확인 칸 ----------------
def part_a() -> dict:
    res = {}
    for c in CELLS:
        cell, side, yname, h = c["cell"], c["side"], c["y"], c["h"]
        col = f"{yname}{h}"
        base_th = G.TH1 if len(parse(cell)) == 1 else G.TH2
        sig = make_sig(cell)
        r = {"meta": c, "known": {s: KNOWN[s] for s, _ in parse(cell)}, "base_th": base_th}
        # 날짜·사건
        r["signal_days"] = {per: [IDX[i] for i in np.where(H.period_mask(IDX, per) & (np.nan_to_num(sig) == 1))[0]] for per in PERS}
        r["episodes"] = {per: ep_table(sig, per, "QQQ", side, h, yname) for per in PERS}
        r["windows_nonoverlap"] = {per: windows(sig, per, h) for per in PERS}
        r["main"] = {per: stat(sig, "QQQ", col, per, side) for per in PERS}
        r["loo"] = {per: leave_one_out(sig, per, "QQQ", side, col) for per in ("front", "back")}
        # 문턱
        r["threshold"] = {}
        for th in (round(base_th - 0.05, 2), base_th, round(base_th + 0.05, 2)):
            s = make_sig(cell, th)
            r["threshold"][str(th)] = {per: {**stat(s, "QQQ", col, per, side), "episodes": len(episodes(s, per))} for per in PERS}
        # h·결과 종류·SOXX
        r["horizon"] = {}
        for asset in ("QQQ", "SOXX"):
            for yy in ("R", "M"):
                for hh in H.HORIZONS:
                    r["horizon"][f"{asset}_{yy}{hh}"] = {per: stat(sig, asset, f"{yy}{hh}", per, side) for per in PERS}
        res[cell] = r
    # 묶음 쏠림: 확인 칸끼리 신호일 겹침
    tops = [c["cell"] for c in CELLS if c["side"] == "top"]
    ov = {}
    for per in ("front", "back"):
        sets = {c: set(res[c]["signal_days"][per]) for c in tops}
        ov[per] = {f"{a} ∩ {b}": [len(sets[a] & sets[b]), len(sets[a] | sets[b])] for i, a in enumerate(tops) for b in tops[i + 1:]}
        ov[per]["union_days"] = sorted(set().union(*sets.values()))
    res["_overlap_top"] = ov
    # 고른 상위 20칸의 front 신호일 집합 고유 수
    import json
    g = json.loads((H.RES / "grid.json").read_text(encoding="utf-8"))
    conc = {}
    fr = H.period_mask(IDX, "front")
    for side in ("top", "bottom"):
        keys, members = {}, {}
        for p in g["picks"][side]:
            s = make_sig(p["cell"])
            yv = Y["QQQ"][f"{p['y']}{p['h']}"].to_numpy()
            k = tuple(IDX[np.where(fr & (np.nan_to_num(s) == 1) & ~np.isnan(yv))[0]])
            keys.setdefault(k, []).append(p["cell"])
        import collections
        cnt = collections.Counter(x for p in g["picks"][side] for x in p["cell"].split("&"))
        grp = collections.Counter(" x ".join(sorted(H.GROUP[x[:-1]] for x in p["cell"].split("&"))) for p in g["picks"][side])
        conc[side] = {"n_picks": len(g["picks"][side]), "unique_front_sets": len(keys), "signal_count": dict(cnt.most_common()),
                      "group_count": dict(grp),
                      "sets": [{"n_days": len(k), "span": f"{k[0]}~{k[-1]}" if k else "", "n_cells": len(v), "cells": v} for k, v in
                               sorted(keys.items(), key=lambda kv: -len(kv[1]))]}
    res["_pick_concentration"] = conc
    return res


# ---------------- (b) T1'·T2' ----------------
def part_b() -> dict:
    re_, sd = G.composite(P, G.RETAIL_EUPH), G.composite(P, G.SMART_DEF)
    re_p, sd_p = H.rolling_pct(re_, 250).reindex(IDX), H.rolling_pct(sd, 250).reindex(IDX)
    both = re_p.notna() & sd_p.notna()
    defs = {
        "T1_pre": (((re_ >= G.TH2) & (sd >= G.TH2)).astype(float).where(re_.notna() & sd.notna()), "top"),
        "T2_pre": (((re_ <= 1 - G.TH2) & (sd <= 1 - G.TH2)).astype(float).where(re_.notna() & sd.notna()), "bottom"),
        "T1_rank": (((re_p >= 0.7) & (sd_p >= 0.7)).astype(float).where(both), "top"),
        "T2_rank": (((re_p <= 0.3) & (sd_p <= 0.3)).astype(float).where(both), "bottom"),
    }
    out = {"composite_range": {}, "corr": {}}
    for per in PERS:
        m = H.period_mask(IDX, per)
        out["composite_range"][per] = {
            "RETAIL_EUPH": [float(re_[m].min()), float(re_[m].median()), float(re_[m].max())],
            "SMART_DEF": [float(sd[m].min()), float(sd[m].median()), float(sd[m].max())],
            "RETAIL_EUPH_rankpct": [float(re_p[m].min()), float(re_p[m].max())], "SMART_DEF_rankpct": [float(sd_p[m].min()), float(sd_p[m].max())],
            "n_valid": int((m & both.to_numpy()).sum())}
        out["corr"][per] = float(pd.concat([re_[m], sd[m]], axis=1).corr().iloc[0, 1])
    first_valid = IDX[np.where(both.to_numpy())[0][0]]
    out["rank_first_valid"] = first_valid
    for nm, (s, side) in defs.items():
        sig = s.to_numpy()
        rec = {"side": side, "n_days": {per: int(np.nansum(np.where(H.period_mask(IDX, per), sig, np.nan))) for per in PERS},
               "episodes": {per: [{"start": IDX[e[0]], "end": IDX[e[-1]], "n": len(e),
                                   "mean_R21": float(np.nanmean(Y["QQQ"]["R21"].to_numpy()[e])) if Y["QQQ"]["R21"].iloc[e].notna().any() else np.nan,
                                   "mean_M21": float(np.nanmean(Y["QQQ"]["M21"].to_numpy()[e])) if Y["QQQ"]["M21"].iloc[e].notna().any() else np.nan,
                                   "mean_R5": float(np.nanmean(Y["QQQ"]["R5"].to_numpy()[e])) if Y["QQQ"]["R5"].iloc[e].notna().any() else np.nan}
                                  for e in episodes(sig, per)] for per in PERS},
               "stats": {}}
        for asset in ("QQQ", "SOXX"):
            for yy in ("R", "M"):
                for hh in H.HORIZONS:
                    rec["stats"][f"{asset}_{yy}{hh}"] = {per: stat(sig, asset, f"{yy}{hh}", per, side) for per in PERS}
        out[nm] = rec
    # 사전 등록 T2의 2025 신호일 상세
    t2 = defs["T2_pre"][0].to_numpy()
    m25 = H.period_mask(IDX, "y2025")
    days25 = np.where(m25 & (np.nan_to_num(t2) == 1))[0]
    c25 = PX["QQQ_c"][m25]
    low_day = c25.idxmin()
    low_pos = IDX.get_loc(low_day)
    det = []
    for i in days25:
        q, s_ = Y["QQQ"].iloc[i], Y["SOXX"].iloc[i]
        det.append({"date": IDX[i], "RETAIL_EUPH": float(re_.iloc[i]), "SMART_DEF": float(sd.iloc[i]),
                    "RE_rankpct": float(re_p.iloc[i]), "SD_rankpct": float(sd_p.iloc[i]),
                    "qqq_open": float(PX["QQQ_o"].iloc[i]), "R5": float(q["R5"]), "R10": float(q["R10"]), "R21": float(q["R21"]), "M21": float(q["M21"]),
                    "soxx_R21": float(s_["R21"]), "tdays_from_2025_low": int(i - low_pos),
                    "components": {k: (None if np.isnan(P[k].iloc[i]) else round(float(P[k].iloc[i]), 3)) for k in list(G.RETAIL_EUPH) + list(G.SMART_DEF)}})
    out["T2_pre_2025_days"] = det
    out["qqq_2025_low"] = {"date": low_day, "close": float(c25.min()),
                           "close_2025_first": float(c25.iloc[0]), "close_2025_02_19_peak": float(c25[c25.index <= "2025-04-30"].max()),
                           "peak_date": c25[c25.index <= "2025-04-30"].idxmax()}
    # T2_rank 2025 날짜도
    t2r = defs["T2_rank"][0].to_numpy()
    out["T2_rank_2025_days"] = [{"date": IDX[i], "R21": float(Y["QQQ"]["R21"].iloc[i]), "M21": float(Y["QQQ"]["M21"].iloc[i]),
                                 "tdays_from_2025_low": int(i - low_pos)} for i in np.where(m25 & (np.nan_to_num(t2r) == 1))[0]]
    out["rank_days_2026"] = {nm: [IDX[i] for i in np.where(np.asarray(IDX >= "2026-01-02") & (np.nan_to_num(defs[nm][0].to_numpy()) == 1))[0]]
                             for nm in ("T1_rank", "T2_rank")}
    return out


# ---------------- (c) 기본 비율 ----------------
def part_c() -> dict:
    out = {}
    for asset in ("QQQ", "SOXX"):
        for per in PERS:
            m = H.period_mask(IDX, per)
            y = Y[asset][m]
            ok = y["M21"].notna()
            out[f"{asset}_{per}"] = {"n_R21": int(ok.sum()), "last_entry_R21": y.index[ok.to_numpy()][-1] if ok.any() else None,
                                     "mean_R21": float(y["R21"].mean()), "mean_M21": float(y["M21"].mean()),
                                     "P_M21_le_-5": float((y.loc[ok, "M21"] <= -5).mean()), "P_M21_le_-10": float((y.loc[ok, "M21"] <= -10).mean()),
                                     "mean_R10": float(y["R10"].mean()), "mean_M10": float(y["M10"].mean()),
                                     "mean_R5": float(y["R5"].mean()), "mean_M5": float(y["M5"].mean())}
    return out


def main() -> None:
    A, B, C = part_a(), part_b(), part_c()
    H.save("robust", {"a_cells": A, "b_T_rank": B, "c_base": C,
                      "notes": {"episode_gap": EP_GAP, "p_side": "top: shift_p 'less'(신호 뒤 더 나쁨), bottom: 'greater'",
                                "loo": "사건의 날을 신호·비교군 모두에서 빼고 다시 잼", "T_rank": "사후 변형, 판정에 쓰지 않음"}})
    # ---------- 출력(마크다운) ----------
    print("## (c) 기본 비율")
    print("| 자산·기간 | 결과 있는 날 | 평균 R21 | 평균 M21 | P(M21≤−5%) | P(M21≤−10%) |")
    print("|---|---|---|---|---|---|")
    for k, v in C.items():
        print(f"| {k} | {v['n_R21']} (~{v['last_entry_R21']}) | {fmt(v['mean_R21'])} | {fmt(v['mean_M21'])} | {v['P_M21_le_-5']:.2f} | {v['P_M21_le_-10']:.2f} |")
    print("\n## (a) 확인 칸")
    for c in CELLS:
        cell = c["cell"]
        r = A[cell]
        col = f"{c['y']}{c['h']}"
        print(f"\n### {cell} {col} ({c['side']})")
        for per in PERS:
            mm = r["main"][per]
            print(f"- {per}: 신호 {mm['n']}일/비교 {mm['n_rest']}일, 신호 평균 {fmt(mm['mean_sig'])} vs 나머지 {fmt(mm['mean_rest'])}, 차 {fmt(mm['d'])}, "
                  f"p {fp(mm['p'])} (이동 {mm['n_shift']}개, 최소 p {fp(mm['p_floor'])}), 사건 {len(r['episodes'][per])}개, 창 안 겹침 묶음 {r['windows_nonoverlap'][per]}개, "
                  f"M≤−5% 비율 신호 {fmt(mm['le5_sig'], 2, False)} / 나머지 {fmt(mm['le5_rest'], 2, False)}")
            for e in r["episodes"][per]:
                print(f"    - {e['start']}~{e['end']} ({e['n_days']}일, 결과 {e['n_outcome']}일): 평균 {col} {fmt(e[f'mean_{col}'])}, "
                      f"평균 R21 {fmt(e['mean_R21'])}, 평균 M21 {fmt(e['mean_M21'])}, 첫날 R21 {fmt(e['first_R21'])} M21 {fmt(e['first_M21'])}")
                print(f"      날짜: {', '.join(d[5:] for d in e['dates'])}")
        for per in ("front", "back"):
            for l in r["loo"][per]:
                print(f"  - LOO {per} 빼기 {l['drop']}: 차 {fmt(l['d'])}, n {l['n']}, p {fp(l['p'])}, 같은 방향 {l['same_dir']}")
        print("  | 문턱 | front 차(n,사건) | back 차(n,사건) p | 2025 차(n,사건) p |")
        print("  |---|---|---|---|")
        for th, v in r["threshold"].items():
            print(f"  | {th} | {fmt(v['front']['d'])} ({v['front']['n']},{v['front']['episodes']}) | {fmt(v['back']['d'])} ({v['back']['n']},{v['back']['episodes']}) {fp(v['back']['p'])} "
                  f"| {fmt(v['y2025']['d'])} ({v['y2025']['n']},{v['y2025']['episodes']}) {fp(v['y2025']['p'])} |")
        print("  | 결과 | front 차 (n) p | back 차 (n) p | 2025 차 (n) p |")
        print("  |---|---|---|---|")
        for k, v in r["horizon"].items():
            print(f"  | {k} | {fmt(v['front']['d'])} ({v['front']['n']}) {fp(v['front']['p'])} | {fmt(v['back']['d'])} ({v['back']['n']}) {fp(v['back']['p'])} "
                  f"| {fmt(v['y2025']['d'])} ({v['y2025']['n']}) {fp(v['y2025']['p'])} |")
    print("\n### 겹침", A["_overlap_top"])
    for side, v in A["_pick_concentration"].items():
        print(f"### 고른 20칸 쏠림 {side}: 고유 front 신호일 집합 {v['unique_front_sets']}개")
        for s in v["sets"]:
            print(f"  - {s['n_days']}일 {s['span']}: {s['n_cells']}칸 {s['cells'][:4]}")
    print("\n## (b) T1'·T2'")
    print("범위", B["composite_range"], "\n상관", B["corr"], "\n순위 첫 유효일", B["rank_first_valid"])
    for nm in ("T1_pre", "T2_pre", "T1_rank", "T2_rank"):
        v = B[nm]
        print(f"\n### {nm} ({v['side']}) 신호일 {v['n_days']}")
        for per in PERS:
            print(f"  {per} 사건: " + "; ".join(f"{e['start']}~{e['end']}({e['n']}일, R5 {fmt(e['mean_R5'],1)}, R21 {fmt(e['mean_R21'],1)}, M21 {fmt(e['mean_M21'],1)})" for e in v['episodes'][per]))
        print("  | 결과 | front 차 (n) p | back 차 (n) p | 2025 차 (n) p |")
        print("  |---|---|---|---|")
        for k, s in v["stats"].items():
            print(f"  | {k} | {fmt(s['front']['d'])} ({s['front']['n']}) {fp(s['front']['p'])} | {fmt(s['back']['d'])} ({s['back']['n']}) {fp(s['back']['p'])} "
                  f"| {fmt(s['y2025']['d'])} ({s['y2025']['n']}) {fp(s['y2025']['p'])} |")
    print("\n### 사전 등록 T2 2025 신호일", B["qqq_2025_low"])
    print("| 날짜 | 개인환호 | 큰손방어 | QQQ 시가 | R5 | R10 | R21 | M21 | SOXX R21 | 저점까지(거래일) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in B["T2_pre_2025_days"]:
        print(f"| {d['date']} | {d['RETAIL_EUPH']:.2f} | {d['SMART_DEF']:.2f} | {d['qqq_open']:.2f} | {fmt(d['R5'],1)} | {fmt(d['R10'],1)} | {fmt(d['R21'],1)} | {fmt(d['M21'],1)} | {fmt(d['soxx_R21'],1)} | {d['tdays_from_2025_low']} |")
    for d in B["T2_pre_2025_days"]:
        print(d["date"], d["components"])
    print("T2_rank 2025", B["T2_rank_2025_days"])
    print("rank 2026", B["rank_days_2026"])


if __name__ == "__main__":
    main()
