"""rs_pullback 갈래 entry_timing (2026-10-13): 데스크 7종목 규칙의 '새로 사는 종목'을 월요일 시가 대신
주중 상대강도 눌림을 기다려 더 싸게 살 수 있는가.

규칙(사전 등록 8개, BRIEF.md 공통 정의):
  T0  월요일(그 주 첫 거래일) 시가 매수(기본)
  T1  월~목 종가 중 처음 ex(1) < 0 인 날 → 다음 시가 매수, 없으면 금요일(그 주 마지막 거래일) 시가
  T2  같은 방식, 조건 ex(4) <= -3
  T3  같은 방식, 조건 P3(5) >= 3%  (상대강도선 R=C/QQQ의 5봉 최고 대비 낙폭)
  T4a/b/c  월요일 시가×(1-x) 지정가, x=1/2/3%, 월~목 유효. 그날 저가 <= 지정가면 체결(시가가 그 아래로
       갭 하락하면 시가 체결 = min(시가, 지정가)), 없으면 금요일 시가
  T5  결정 종가(월요일 직전 종가)에 ex(4)가 universe_mask 안 상위 20%('추격')면 T1 조건을 기다리고, 아니면 월요일 시가
대조(후보 아님, 선택에 쓰지 않음):
  Tfri  조건 없이 그 주 마지막 거래일 시가(순수 지연)
  Trand 그 주 각 거래일 시가 중 무작위 하루(기대값, 진입별 개선만)
  T4*_strict  지정가를 0.1% 이상 뚫어야 체결(줄 서기 낙관 보정, 민감도)
사후 진단(결과를 본 뒤 추가, 채택 후보 아님):
  T4*_mon  지정가를 월요일 하루만 걸고 못 사면 화요일 시가(금요일까지 기다리는 비용을 떼어 보기)

계좌: 기존 보유 유지·편출은 기본과 같이 월요일 시가. 새 종목 몫(월요일 시가 기준 목표 비중×계좌)은 체결까지 현금.
시점: 모든 조건은 그날 종가까지 아는 값, 체결은 다음 시가(T4는 걸어 둔 지정가). 미래 정보 없음(no_lookahead 확인).
가격: 계좌는 deskx.Engine의 시가(데스크와 같은 자료), 저가·QQQ·조건은 harness 패널(분할 반영). 두 자료의 시가가
같음을 확인한다(T4 체결가는 패널 비율을 엔진 시가에 곱한다).
일반 집합: harness universe_mask 안 S_120=ex(120) 상위 7에 매주 새로 들어온 종목(완충 없음)에 같은 규칙.

실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rs_pullback/code/entry_timing.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(HERE, "..", "results")

_spec = importlib.util.spec_from_file_location("tg_harness", os.path.join(ANA, "trend_grade", "code", "harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
sys.path.insert(0, os.path.join(ANA, "rsi_breakout", "code"))
from deskx import Engine, Params, START, PERIODS  # noqa: E402

FRONT = ("2026-01-02", "2026-05-29")
BACK = ("2026-06-01", "2026-10-09")
SEPOCT = ("2026-09-01", "2026-10-09")
SPREAD = 0.001
P_DESK = Params(semis_cap=0.5)

RULES = ["T0", "T1", "T2", "T3", "T4a", "T4b", "T4c", "T5"]
CONTROLS = ["Tfri", "T4a_strict", "T4b_strict", "T4c_strict"]
POSTHOC = ["T4a_mon", "T4b_mon", "T4c_mon"]
LIMITS = {"T4a": 0.01, "T4b": 0.02, "T4c": 0.03}


def iso_week(day: str):
    return date.fromisoformat(day).isocalendar()[:2]


# ------------------------------------------------------------------ 신호(t 종가까지)
def ex(P, n):
    q = P["bench"]["C"]
    return 100 * (P["C"] / P["C"].shift(n)).sub(q / q.shift(n), axis=0)


def p3(P, k):
    R = P["C"].div(P["bench"]["C"], axis=0)
    return 1 - R / R.rolling(k, min_periods=k).max()


def signals(P):
    return {"ex1": ex(P, 1), "ex4": ex(P, 4), "p3_5": p3(P, 5), "ex120": ex(P, 120)}


# ------------------------------------------------------------------ 체결 규칙
class Filler:
    """한 종목의 한 주 진입을 규칙별로 체결한다. 날짜는 문자열 'YYYY-MM-DD'."""

    def __init__(self, P, sig, umask):
        self.P, self.sig, self.umask = P, sig, umask
        self.idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(P["C"].index)}
        self.O = P["O"].to_numpy(float)
        self.L = P["L"].to_numpy(float)
        self.cols = {c: j for j, c in enumerate(P["C"].columns)}
        self.s = {k: v.to_numpy(float) for k, v in sig.items()}
        self.um = umask.to_numpy(bool)

    def chasing(self, sym, day0):
        """결정 종가(day0 직전 거래일)의 ex(4) 백분위 >= 0.80 (universe_mask 안)."""
        i = self.idx[day0] - 1
        j = self.cols[sym]
        x = self.s["ex4"][i, j]
        uni = self.s["ex4"][i][self.um[i] & np.isfinite(self.s["ex4"][i])]
        if not np.isfinite(x) or len(uni) < 10:
            return False, np.nan
        pct = float((uni <= x).mean())
        return pct >= 0.80, pct

    def fill(self, rule, sym, week):
        """week: 그 주 거래일 목록. 반환 (체결일 위치, 체결가/월요일 시가(패널), 대체 체결 여부)."""
        j = self.cols[sym]
        ii = [self.idx[d] for d in week]
        o0 = self.O[ii[0], j]
        last = len(week) - 1
        if rule == "T0" or last == 0:
            return 0, 1.0, False
        if rule == "Tfri":
            return last, self.O[ii[last], j] / o0, False
        if rule == "T5":
            ch, _ = self.chasing(sym, week[0])
            if not ch:
                return 0, 1.0, False
            rule = "T1"
        if rule in ("T1", "T2", "T3"):
            for pos in range(last):           # 월~목 종가
                i = ii[pos]
                if rule == "T1":
                    ok = self.s["ex1"][i, j] < 0
                elif rule == "T2":
                    ok = self.s["ex4"][i, j] <= -3
                else:
                    ok = self.s["p3_5"][i, j] >= 0.03
                if ok:
                    return pos + 1, self.O[ii[pos + 1], j] / o0, False
            return last, self.O[ii[last], j] / o0, True
        if rule.endswith("_mon"):
            lim = o0 * (1 - LIMITS[rule[:3]])
            if np.isfinite(self.L[ii[0], j]) and self.L[ii[0], j] <= lim:
                return 0, lim / o0, False
            return 1, self.O[ii[1], j] / o0, True
        if rule.startswith("T4"):
            x = LIMITS[rule[:3]]
            strict = rule.endswith("_strict")
            lim = o0 * (1 - x)
            for pos in range(last):           # 월~목 유효
                i = ii[pos]
                lo, op = self.L[i, j], self.O[i, j]
                if pos > 0 and np.isfinite(op) and op <= lim:   # 갭 하락은 시가 체결
                    return pos, op / o0, False
                touch = lo <= lim * (0.999 if strict else 1.0)
                if np.isfinite(lo) and touch:
                    return pos, lim / o0, False
            return last, self.O[ii[last], j] / o0, True
        raise ValueError(rule)

    def rand_ratio(self, sym, week):
        """그 주 거래일 시가 중 무작위 하루에 산 경우 월요일 시가 대비 비율의 기대값(개선 = E[o0/o]-1)."""
        j = self.cols[sym]
        ii = [self.idx[d] for d in week]
        o = self.O[ii, j]
        return float(np.nanmean(o[0] / o) - 1)


# ------------------------------------------------------------------ 데스크 계좌(주식 수 기준)
class DeskSim:
    def __init__(self, E: Engine, filler: Filler):
        self.E, self.F = E, filler
        self.days = E.days
        self.k0 = self.days.index([d for d in self.days if d >= START][0])
        self._o = {}

    def opn(self, key, k):
        if key not in self._o:
            o = self.E.col("opn", key)
            self._o[key] = pd.Series(o).ffill().to_numpy(float) if o is not None else None
        o = self._o[key]
        return float(o[k]) if o is not None and np.isfinite(o[k]) else np.nan

    def raw_opn(self, key, k):
        o = self.E.col("opn", key)
        return float(o[k]) if o is not None and np.isfinite(o[k]) else np.nan

    def week_of(self, k):
        wk = iso_week(self.days[k])
        out = [k]
        while out[-1] + 1 < len(self.days) and iso_week(self.days[out[-1] + 1]) == wk:
            out.append(out[-1] + 1)
        return out

    def run(self, rule: str, spread: float = 0.0, initial_t0: bool = False):
        """initial_t0: 첫 주(빈 계좌에서 7종목을 한꺼번에 사는 주)는 규칙과 무관하게 월요일 시가.
        이미 투자 중인 데스크에 맞는 비교(첫 주 개선이 3배 넘게 복리로 불어나는 왜곡을 뺀다)."""
        E, days = self.E, self.days
        cash, shares = 1.0, {}
        selected, started = set(), False
        pending = []          # dict(key, amount, k_fill, px)
        values, entries, turnover = [], [], 0.0
        for k in range(self.k0, len(days)):
            day = days[k]
            V = cash + sum(sh * self.opn(key, k) for key, sh in shares.items())
            values.append(V)
            reb = (not started) or iso_week(day) != iso_week(days[k - 1])
            if reb:
                keep, tgt, ranked = E.target(P_DESK, k, day, selected)
                if ranked:
                    started = True
                    assert not pending, "주가 끝나기 전에 체결 안 된 주문"
                    prev = set(shares)
                    for key in prev - set(tgt):                       # 편출: 월요일 시가
                        n = shares.pop(key) * self.opn(key, k)
                        cash += n * (1 - spread)
                        turnover += n / V
                    for key in prev & set(tgt):                       # 유지: 목표 비중으로 맞춤
                        o = self.opn(key, k)
                        want = tgt[key] * V / o
                        d = want - shares[key]
                        cash -= d * o + abs(d) * o * spread
                        turnover += abs(d) * o / V
                        shares[key] = want
                    week = self.week_of(k)
                    wdays = [days[i] for i in week]
                    for key in sorted(set(tgt) - prev):               # 신규 편입
                        amount = tgt[key] * V
                        sym = E.ticker[key]
                        pos, ratio, fb = self.F.fill("T0" if (initial_t0 and not prev) else rule, sym, wdays)
                        kf = week[pos]
                        o_mon = self.raw_opn(key, k)
                        if pos == 0 and ratio == 1.0:
                            px = o_mon
                        elif rule.startswith("T4") and not fb:  # _mon 대체는 fb=True라 엔진 시가
                            px = o_mon * ratio                       # 지정가·갭 시가: 패널 비율 × 엔진 월요일 시가
                        else:
                            px = self.raw_opn(key, kf)
                        rec = {"key": key, "sym": sym, "week": day, "fill_day": days[kf], "pos": pos,
                               "fallback": fb, "w": float(tgt[key]), "o_mon": o_mon, "px": px,
                               "imp": o_mon / px - 1, "single_day_week": len(week) == 1, "initial": not prev}
                        if rule == "T5":
                            rec["chasing_pct"] = self.F.chasing(sym, day)[1]
                        entries.append(rec)
                        pending.append({"key": key, "amount": amount, "k": kf, "px": px})
                    selected = set(keep)
            # 체결
            still = []
            for od in pending:
                if od["k"] == k:                                      # 현금은 체결 때 나간다
                    cash -= od["amount"]
                    shares[od["key"]] = od["amount"] * (1 - spread) / od["px"]
                    turnover += od["amount"] / V
                else:
                    still.append(od)
            pending = still
        v = pd.Series(values, index=days[self.k0:])
        return v, entries, turnover


def period_stats(E, v):
    res = E.evaluate("x", v)
    out = {}
    for lab, a, b in PERIODS:
        out[lab] = {"ret": round(res[lab]["수익"] * 100, 2), "mdd": round(res[lab]["낙폭"] * 100, 2)}
    out["monthly"] = {m: round(x * 100, 2) for m, x in res["월별"].items()}
    return out


# ------------------------------------------------------------------ 진입별 요약
def in_win(day, win):
    return win[0] <= day <= win[1]


def entry_summary(recs, key_imp="imp"):
    recs = [r for r in recs if not r.get("single_day_week")]
    if not recs:
        return {"n": 0}
    x = np.array([r[key_imp] for r in recs], float) * 100
    wk = defaultdict(list)
    for r, xi in zip(recs, x):
        wk[r["week"]].append(xi)
    wm = np.array([np.mean(v) for v in wk.values()])
    t_w = float(wm.mean() / (wm.std(ddof=1) / np.sqrt(len(wm)))) if len(wm) > 2 and wm.std(ddof=1) > 0 else float("nan")
    out = {"n": int(len(x)), "weeks": int(len(wm)), "mean": round(float(x.mean()), 3),
           "median": round(float(np.median(x)), 3), "win": round(float((x > 0).mean()) * 100, 1),
           "lose": round(float((x < 0).mean()) * 100, 1), "t_week": round(t_w, 2)}
    if "fallback" in recs[0]:
        fb = np.array([r["fallback"] for r in recs])
        waited = np.array([r["pos"] > 0 for r in recs])
        out["fill_share"] = round(float((~fb & waited).mean()) * 100, 1)
        out["fallback_share"] = round(float(fb.mean()) * 100, 1)
        out["monday_share"] = round(float((~waited).mean()) * 100, 1)
        if fb.any():
            out["fallback_mean"] = round(float(x[fb].mean()), 3)
        if (~fb & waited).any():
            out["filled_mean"] = round(float(x[~fb & waited].mean()), 3)
    if "w" in recs[0]:
        out["w_imp_sum_pp"] = round(float(sum(r["w"] * r[key_imp] for r in recs)) * 100, 3)
    return out


def concentration(recs, key_imp="imp"):
    recs = [r for r in recs if not r.get("single_day_week")]
    if not recs:
        return {}
    by_sym, by_week, by_month = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in recs:
        by_sym[r["sym"]].append(r[key_imp] * 100)
        by_week[r["week"]].append(r[key_imp] * 100)
        by_month[r["week"][:7]].append(r[key_imp] * 100)
    tot = sum(sum(v) for v in by_sym.values())
    syms = sorted(by_sym.items(), key=lambda kv: -abs(sum(kv[1])))
    weeks = sorted(by_week.items(), key=lambda kv: -abs(sum(kv[1])))
    x = np.array([r[key_imp] for r in recs]) * 100
    mean_all = float(x.mean())

    def mean_wo(pred):
        y = [r[key_imp] * 100 for r in recs if not pred(r)]
        return round(float(np.mean(y)), 3) if y else None

    top_sym = syms[0][0]
    top_week = weeks[0][0]
    return {
        "sum_pct": round(tot, 2),
        "top_syms": [(s, len(v), round(sum(v), 2)) for s, v in syms[:5]],
        "top_weeks": [(w, len(v), round(sum(v), 2)) for w, v in weeks[:5]],
        "by_month": {m: (len(v), round(float(np.mean(v)), 2)) for m, v in sorted(by_month.items())},
        "mean": round(mean_all, 3),
        "mean_wo_top_sym": [top_sym, mean_wo(lambda r: r["sym"] == top_sym)],
        "mean_wo_top_week": [top_week, mean_wo(lambda r: r["week"] == top_week)],
        "mean_wo_SNDK": mean_wo(lambda r: r["sym"] == "SNDK"),
        "mean_wo_july": mean_wo(lambda r: r["week"].startswith("2026-07")),
    }


# ------------------------------------------------------------------ 일반 집합: S_120 상위 7 신규 편입
def generic_entries(P, sig, umask, filler, rule, start, end, top=7, pct=None):
    cal = [d.strftime("%Y-%m-%d") for d in P["C"].index]
    ex120 = sig["ex120"].to_numpy(float)
    um = umask.to_numpy(bool)
    cols = list(P["C"].columns)
    O = P["O"].to_numpy(float)
    firsts = [i for i in range(1, len(cal)) if iso_week(cal[i]) != iso_week(cal[i - 1])]
    prev_top, out = None, []
    for i in firsts:
        if cal[i] > end:
            break
        row = ex120[i - 1]
        ok = um[i - 1] & np.isfinite(row)
        order = [cols[j] for j in np.argsort(-np.where(ok, row, -np.inf)) if ok[j]]
        if pct is None:
            cur = set(order[:top])
        else:                                   # 백분위 >= pct (BRIEF '강함' 0.70)
            n_ok = len(order)
            cur = set(order[:int(np.floor(n_ok * (1 - pct)))])
        if prev_top is not None and cal[i] >= start:
            week = [cal[i]]
            while len(cal) > i + len(week) and iso_week(cal[i + len(week)]) == iso_week(cal[i]):
                week.append(cal[i + len(week)])
            for sym in sorted(cur - prev_top):
                j = cols.index(sym)
                if not (np.isfinite(O[i, j]) and all(np.isfinite(O[filler.idx[d], j]) for d in week)):
                    continue
                if rule == "Trand":
                    out.append({"sym": sym, "week": cal[i], "imp": filler.rand_ratio(sym, week),
                                "single_day_week": len(week) == 1})
                    continue
                pos, ratio, fb = filler.fill(rule, sym, week)
                out.append({"sym": sym, "week": cal[i], "pos": pos, "fallback": fb, "imp": 1 / ratio - 1,
                            "single_day_week": len(week) == 1})
        prev_top = cur
    return out


def main():
    P = H.load_panel()
    sig = signals(P)
    for name, fn in (("ex1", lambda Q: ex(Q, 1)), ("ex4", lambda Q: ex(Q, 4)), ("p3_5", lambda Q: p3(Q, 5)),
                     ("ex120", lambda Q: ex(Q, 120))):
        assert H.no_lookahead(fn), name
    umask = H.universe_mask(P)
    F = Filler(P, sig, umask)
    E = Engine()
    S = DeskSim(E, F)

    # 엔진 시가 = 패널 시가 확인(2026, 데스크가 거친 종목)
    v_ref, rows_ref = E.run(P_DESK)
    keys = sorted({k for r in rows_ref for k in r["w"]})
    ratio_dev = 0.0
    for key in keys:
        o = pd.Series(E.col("opn", key), index=E.days).loc[START:]
        po = P["O"][E.ticker[key]]
        po.index = po.index.strftime("%Y-%m-%d")
        rr = (o / po.reindex(o.index)).dropna()
        ratio_dev = max(ratio_dev, float((rr - 1).abs().max()))

    out = {"family": "entry_timing", "params": "deskx Params(semis_cap=0.5), 기간 2026-01-02~2026-10-09",
           "price_check_max_abs(engine_open/panel_open-1)": ratio_dev,
           "rules": RULES, "controls": CONTROLS + ["Trand"], "desk": {}, "generic": {}}

    # 기본 재현
    v0, e0, _ = S.run("T0")
    out["base_reproduction_max_diff"] = float(np.max(np.abs(v0.to_numpy() - v_ref.to_numpy())))
    print("T0 vs Engine.run max diff", out["base_reproduction_max_diff"], "| open check", ratio_dev)

    for rule in RULES + CONTROLS + POSTHOC:
        v, ent, to = S.run(rule)
        vs, _, _ = S.run(rule, spread=SPREAD)
        vi, _, _ = S.run(rule, initial_t0=True)
        vis, _, _ = S.run(rule, spread=SPREAD, initial_t0=True)
        acc = period_stats(E, v)
        acc_s = period_stats(E, vs)
        acc_i = period_stats(E, vi)
        acc_is = period_stats(E, vis)
        d = {"account": acc, "account_spread0.1": {k: acc_s[k] for k in acc_s if k != "monthly"},
             "account_initial_t0": acc_i,
             "account_initial_t0_spread0.1": {k: acc_is[k] for k in acc_is if k != "monthly"},
             "turnover_sum": round(to, 2),
             "entries_all": entry_summary(ent),
             "entries_front": entry_summary([r for r in ent if in_win(r["week"], FRONT)]),
             "entries_back": entry_summary([r for r in ent if in_win(r["week"], BACK)]),
             "entries_front_ex_initial": entry_summary([r for r in ent if in_win(r["week"], FRONT) and not r["initial"]]),
             "entries_all_ex_initial": entry_summary([r for r in ent if not r["initial"]]),
             "entries_sepoct": entry_summary([r for r in ent if in_win(r["week"], SEPOCT)]),
             "concentration": concentration(ent),
             "concentration_ex_initial": concentration([r for r in ent if not r["initial"]])}
        if rule == "T5":
            d["chasing_share"] = round(float(np.mean([r["chasing_pct"] >= 0.80 for r in ent
                                                      if not r["single_day_week"]])) * 100, 1)
        d["entry_list"] = [{k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items() if k != "key"}
                           for r in ent]
        out["desk"][rule] = d
        a = acc
        ai = acc_i
        print(f"   init-T0: full {ai['전체']['ret']:8.2f} front {ai['앞 1/2~5/29']['ret']:8.2f} back {ai['뒤 6/1~10/9']['ret']:7.2f} "
              f"| entry ex-init F {d['entries_front_ex_initial'].get('mean')} (n{d['entries_front_ex_initial'].get('n')})")
        print(f"{rule:12s} full {a['전체']['ret']:8.2f} front {a['앞 1/2~5/29']['ret']:8.2f} back {a['뒤 6/1~10/9']['ret']:7.2f} "
              f"9/1~ {a['9/1~10/9']['ret']:6.2f} mdd {a['전체']['mdd']:7.2f} | spread full {acc_s['전체']['ret']:8.2f} | "
              f"entry F {d['entries_front'].get('mean')} (n{d['entries_front'].get('n')}) "
              f"B {d['entries_back'].get('mean')} (n{d['entries_back'].get('n')}) fb {d['entries_all'].get('fallback_share')}")

    # 일반 집합
    wins = (("front", FRONT), ("back", BACK), ("sepoct", SEPOCT), ("y2025", ("2025-01-02", "2025-12-31")),
            ("y2024", ("2024-01-02", "2024-12-31")))
    for setname, kw in (("generic", {"top": 7}), ("generic_strong30", {"pct": 0.70})):
        out.setdefault(setname, {})
        for wname, (a, b) in wins:
            res = {}
            for rule in RULES + ["Tfri", "Trand", "T4b_strict"] + POSTHOC:
                ent = generic_entries(P, sig, umask, F, rule, a, b, **kw)
                res[rule] = entry_summary(ent)
                if wname in ("front", "back"):
                    res[rule]["concentration"] = concentration(ent)
            out[setname][wname] = res
            print(setname, wname, {r: (res[r].get("mean"), res[r].get("t_week"), res[r].get("n")) for r in res})

    # 선택: 앞 구간 계좌 수익 최고(T0 제외) → 뒤 확인
    alts = [r for r in RULES if r != "T0"]
    fr = lambda r: out["desk"][r]["account"]["앞 1/2~5/29"]["ret"]
    bk = lambda r: out["desk"][r]["account"]["뒤 6/1~10/9"]["ret"]
    best = max(alts, key=fr)
    fri = lambda r: out["desk"][r]["account_initial_t0"]["앞 1/2~5/29"]["ret"]
    bki = lambda r: out["desk"][r]["account_initial_t0"]["뒤 6/1~10/9"]["ret"]
    best_i = max(alts, key=fri)
    out["selection_initial_t0"] = {"best_front": best_i, "best_front_ret": fri(best_i), "T0_front": fri("T0"),
                                   "best_back": bki(best_i), "T0_back": bki("T0"),
                                   "beats_T0_both": [r for r in alts if fri(r) > fri("T0") and bki(r) > bki("T0")],
                                   "posthoc_beats_T0_both": [r for r in POSTHOC if fri(r) > fri("T0") and bki(r) > bki("T0")]}
    print(out["selection_initial_t0"])
    out["selection"] = {
        "by": "앞 구간 계좌 수익(T0 제외 7개)",
        "best_front": best, "best_front_ret": fr(best), "T0_front": fr("T0"),
        "best_back": bk(best), "T0_back": bk("T0"),
        "beats_T0_both": [r for r in alts if fr(r) > fr("T0") and bk(r) > bk("T0")],
        "best_entry_front": max(alts, key=lambda r: out["desk"][r]["entries_front"]["mean"]),
    }
    print(out["selection"])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "entry_timing.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("saved", os.path.abspath(path))


if __name__ == "__main__":
    main()
