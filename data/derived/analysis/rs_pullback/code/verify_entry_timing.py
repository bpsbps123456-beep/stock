"""entry_timing 독립 반박 검증 (2026-10-13). entry_timing.py는 가져오지 않는다.

다시 재는 것
  1) 데스크 7종목(Params(semis_cap=0.5)) 계좌: T0·T1·T2·T3·T4a/b/c·T5·Tfri·T4a_mon 앞/뒤/전체 수익(첫 주 월요일 시가 고정).
     가격은 harness 패널(O·L·C, QQQ)만 쓰고 엔진은 주간 목표(E.target)만 받는다. T0는 deskx.Engine.run과 비교한다.
  2) 데스크 진입별 개선(월요일 시가/체결가 − 1), 첫 매수 제외 앞 구간.
  3) 일반 집합(universe_mask 안 ex(120) 상위 7 주간 신규) 앞/뒤/9월~/2025/2024 평균 개선.
반박 시도
  a) 무작위 체결일 귀무분포(신규 편입마다 그 주 거래일 하나를 무작위로 골라 시가 체결, 2000회): 앞·뒤 모두 T0를 넘는 비율.
  b) 쏠림: DELL만, 5/26주 전체를 T0로 고정했을 때 계좌 차이.
  c) 호가 편도 0.1%, 지정가 체결은 호가 없음(메이커) 변형. T4a_mon strict(0.1% 더 뚫어야 체결).
  d) 7월 제외 일반 집합 뒤 구간, 2025 부호.
  e) 시점: 신호 4개 harness.no_lookahead, 엔진 시가 = 패널 시가, 데스크 run_model(현재 50%)과 T0 경로 비교.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rs_pullback/code/verify_entry_timing.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "results", "verify_entry_timing.json")
sys.path.insert(0, os.path.join(ANA, "trend_grade", "code"))
import harness as H  # noqa: E402
sys.path.insert(0, os.path.join(ANA, "rsi_breakout", "code"))
from deskx import Engine, Params, START  # noqa: E402

PD = Params(semis_cap=0.5)
WIN = {"front": ("2026-01-02", "2026-05-29"), "back": ("2026-06-01", "2026-10-09"),
       "sepoct": ("2026-09-01", "2026-10-09"), "full": ("2026-01-02", "2026-10-09")}
LIM = {"T4a": 0.01, "T4b": 0.02, "T4c": 0.03}


def wk(day: str):
    return date.fromisoformat(day).isocalendar()[:2]


# ------------------------------------------------------------------ 패널·신호
P = H.load_panel()
CAL = [d.strftime("%Y-%m-%d") for d in P["C"].index]
ROW = {d: i for i, d in enumerate(CAL)}
COLS = list(P["C"].columns)
CI = {c: j for j, c in enumerate(COLS)}
Oa, La = P["O"].to_numpy(float), P["L"].to_numpy(float)
Oc = P["O"].ffill().to_numpy(float)          # 평가용(거래 없는 날은 직전 시가)


def f_ex(n):
    return lambda Q: 100 * (Q["C"] / Q["C"].shift(n)).sub(Q["bench"]["C"] / Q["bench"]["C"].shift(n), axis=0)


def f_p3(k):
    def g(Q):
        R = Q["C"].div(Q["bench"]["C"], axis=0)
        return 1 - R / R.rolling(k, min_periods=k).max()
    return g


SIGF = {"ex1": f_ex(1), "ex4": f_ex(4), "ex120": f_ex(120), "p3_5": f_p3(5)}
SIG = {k: f(P).to_numpy(float) for k, f in SIGF.items()}
UM = H.universe_mask(P).to_numpy(bool)


def chasing(i_dec, j):
    x = SIG["ex4"][i_dec, j]
    uni = SIG["ex4"][i_dec][UM[i_dec] & np.isfinite(SIG["ex4"][i_dec])]
    if not np.isfinite(x) or len(uni) < 10:
        return False
    return float((uni <= x).mean()) >= 0.80


def fill(rule, sym, days, rng=None, strict=0.0):
    """days: 그 주 거래일 문자열(첫날 = 결정 다음 시가). 반환 (주 안 위치, 체결가, 월요일 시가)."""
    j = CI[sym]
    r = [ROW[d] for d in days]
    o0 = Oa[r[0], j]
    last = len(days) - 1
    if rule == "T0" or last == 0:
        return 0, o0, o0
    if rule == "RAND":
        pos = int(rng.integers(0, last + 1))
        return pos, Oa[r[pos], j], o0
    if rule == "Tfri":
        return last, Oa[r[last], j], o0
    if rule == "T5":
        if not chasing(r[0] - 1, j):
            return 0, o0, o0
        rule = "T1"
    if rule in ("T1", "T2", "T3"):
        for pos in range(last):
            i = r[pos]
            ok = {"T1": SIG["ex1"][i, j] < 0, "T2": SIG["ex4"][i, j] <= -3, "T3": SIG["p3_5"][i, j] >= 0.03}[rule]
            if ok:
                return pos + 1, Oa[r[pos + 1], j], o0
        return last, Oa[r[last], j], o0
    if rule.endswith("_mon"):
        lim = o0 * (1 - LIM[rule[:3]])
        if La[r[0], j] <= lim * (1 - strict):
            return 0, lim, o0
        return 1, Oa[r[1], j], o0
    if rule[:3] in LIM:
        lim = o0 * (1 - LIM[rule[:3]])
        for pos in range(last):
            o, lo = Oa[r[pos], j], La[r[pos], j]
            if pos > 0 and o <= lim:
                return pos, o, o0
            if lo <= lim * (1 - strict):
                return pos, lim, o0
        return last, Oa[r[last], j], o0
    raise ValueError(rule)


# ------------------------------------------------------------------ 데스크: 주간 목표는 엔진, 계좌는 내 계산
E = Engine()
DAYS = E.days
K0 = DAYS.index([d for d in DAYS if d >= START][0])
assert all(d in ROW for d in DAYS[K0:]), "엔진 날짜가 패널에 없음"
SYM = lambda key: E.ticker[key]


def weekly_targets():
    out, selected, started = {}, set(), False
    for k in range(K0, len(DAYS)):
        if started and wk(DAYS[k]) == wk(DAYS[k - 1]):
            continue
        keep, tgt, ranked = E.target(PD, k, DAYS[k], selected)
        if ranked:
            started = True
            out[k] = dict(tgt)
            selected = set(keep)
    return out


TGT = weekly_targets()


def week_days(k):
    out = [k]
    while out[-1] + 1 < len(DAYS) and wk(DAYS[out[-1] + 1]) == wk(DAYS[k]):
        out.append(out[-1] + 1)
    return out


def run(rule, spread=0.0, maker_limit=False, force_t0=lambda sym, day: False, rng=None, strict=0.0,
        initial_t0=True, log=None):
    cash, sh, pend, vals = 1.0, {}, [], []
    first = min(TGT)
    for k in range(K0, len(DAYS)):
        i = ROW[DAYS[k]]
        V = cash + sum(n * Oc[i, CI[SYM(key)]] for key, n in sh.items())
        vals.append(V)
        if k in TGT:
            tgt = TGT[k]
            prev = set(sh)
            for key in prev - set(tgt):
                cash += sh.pop(key) * Oc[i, CI[SYM(key)]] * (1 - spread)
            for key in prev & set(tgt):
                o = Oc[i, CI[SYM(key)]]
                want = tgt[key] * V / o
                d = want - sh[key]
                cash -= d * o + abs(d) * o * spread
                sh[key] = want
            wds = week_days(k)
            for key in sorted(set(tgt) - prev):
                sym = SYM(key)
                use = rule
                if (initial_t0 and k == first) or force_t0(sym, DAYS[k]):
                    use = "T0"
                pos, px, o0 = fill(use, sym, [DAYS[x] for x in wds], rng=rng, strict=strict)
                limit_fill = use[:3] in LIM and pos < len(wds) and not np.isclose(px, Oa[ROW[DAYS[wds[pos]]], CI[sym]])
                sp = 0.0 if (maker_limit and limit_fill) else spread
                pend.append((key, tgt[key] * V, wds[pos], px, sp))
                if log is not None:
                    log.append({"sym": sym, "week": DAYS[k], "fill": DAYS[wds[pos]], "w": round(tgt[key], 4),
                                "imp": o0 / px - 1, "initial": k == first})
        keep = []
        for key, amt, kf, px, sp in pend:
            if kf == k:
                cash -= amt
                sh[key] = amt * (1 - sp) / px
            else:
                keep.append((key, amt, kf, px, sp))
        pend = keep
    return pd.Series(vals, index=DAYS[K0:])


def periods(v):
    r = (v.shift(-1) / v - 1).dropna()
    out = {}
    for name, (a, b) in WIN.items():
        x = r[(r.index >= a) & (r.index <= b)]
        c = (1 + x).cumprod()
        out[name] = {"ret": round(float(c.iloc[-1] - 1) * 100, 2),
                     "mdd": round(float((c / c.cummax() - 1).min()) * 100, 2)}
    out["monthly"] = {m: round(float((1 + r[r.index.str.startswith(m)]).prod() - 1) * 100, 2)
                      for m in [f"2026-{x:02d}" for x in range(1, 11)]}
    return out


# ------------------------------------------------------------------ 일반 집합
def generic(rule, a, b, top=7, skip_month=None):
    ex120 = SIG["ex120"]
    firsts = [i for i in range(1, len(CAL)) if wk(CAL[i]) != wk(CAL[i - 1])]
    prev, recs = None, []
    for i in firsts:
        if CAL[i] > b:
            break
        row = ex120[i - 1]
        ok = UM[i - 1] & np.isfinite(row)
        idx = np.where(ok)[0]
        cur = {COLS[j] for j in idx[np.argsort(-row[idx], kind="stable")][:top]}
        if prev is not None and CAL[i] >= a and not (skip_month and CAL[i][:7] == skip_month):
            days = [CAL[i]]
            while i + len(days) < len(CAL) and wk(CAL[i + len(days)]) == wk(CAL[i]):
                days.append(CAL[i + len(days)])
            if len(days) > 1:
                for sym in sorted(cur - prev):
                    j = CI[sym]
                    if not all(np.isfinite(Oa[ROW[d], j]) for d in days):
                        continue
                    pos, px, o0 = fill(rule, sym, days)
                    recs.append((CAL[i], sym, o0 / px - 1))
        prev = cur
    if not recs:
        return {"n": 0}
    x = np.array([r[2] for r in recs]) * 100
    wm = pd.Series(x).groupby([r[0] for r in recs]).mean()
    t = float(wm.mean() / (wm.std(ddof=1) / np.sqrt(len(wm)))) if len(wm) > 2 else float("nan")
    return {"n": len(x), "weeks": int(len(wm)), "mean": round(float(x.mean()), 3), "t_week": round(t, 2)}


def main():
    res = {"family": "verify_entry_timing", "checks": {}}
    ck = res["checks"]
    # e) 시점·자료
    ck["no_lookahead"] = {k: bool(H.no_lookahead(f)) for k, f in SIGF.items()}
    v_eng, rows = E.run(PD)
    keys = sorted({k for r in rows for k in r["w"]})
    dev = 0.0
    for key in keys:
        o = pd.Series(E.col("opn", key), index=DAYS).loc[START:]
        po = pd.Series(Oa[:, CI[SYM(key)]], index=CAL).reindex(o.index)
        dev = max(dev, float((o / po - 1).abs().max()))
    ck["engine_vs_panel_open_maxdev"] = dev
    v0 = run("T0")
    ck["T0_vs_engine_run_maxdiff"] = float(np.max(np.abs(v0.to_numpy() - v_eng.to_numpy())))
    try:
        from desk.strategies import run_model
        ref = run_model(E.book, start=START)
        ck["T0_vs_desk_run_model_current_maxdiff"] = float(np.max(np.abs(v_eng.to_numpy() - np.asarray(ref.value))))
    except Exception as e:  # noqa: BLE001
        ck["T0_vs_desk_run_model_current_maxdiff"] = f"실패: {e}"
    print("checks", ck)

    # 1·2) 데스크 계좌와 진입
    rules = ["T0", "T1", "T2", "T3", "T4a", "T4b", "T4c", "T5", "Tfri", "T4a_mon"]
    desk = {}
    for rl in rules:
        log = []
        v = run(rl, log=log)
        pr = periods(v)
        fe = [e["imp"] * 100 for e in log if not e["initial"] and WIN["front"][0] <= e["week"] <= WIN["front"][1]]
        be = [e["imp"] * 100 for e in log if not e["initial"] and e["week"] >= WIN["back"][0]]
        d = {"front": pr["front"]["ret"], "back": pr["back"]["ret"], "sepoct": pr["sepoct"]["ret"],
             "full": pr["full"]["ret"], "mdd": pr["full"]["mdd"],
             "rel_front_vs_T0_pct": None,
             "entry_front_n": len(fe), "entry_front_mean": round(float(np.mean(fe)), 3),
             "entry_front_median": round(float(np.median(fe)), 3), "entry_back": [round(x, 2) for x in be]}
        # 쏠림: DELL만 / 5/26주 전체 T0 고정
        for tag, ft in (("noDELL", lambda s, dd: s == "DELL"), ("no0526wk", lambda s, dd: dd == "2026-05-26"),
                        ("noDELL_AXTI", lambda s, dd: s in ("DELL", "AXTI"))):
            pv = periods(run(rl, force_t0=ft))
            d[f"front_{tag}"] = pv["front"]["ret"]
            d[f"full_{tag}"] = pv["full"]["ret"]
        ps = periods(run(rl, spread=0.001))
        pm = periods(run(rl, spread=0.001, maker_limit=True))
        d["full_spread"] = ps["full"]["ret"]
        d["front_spread"], d["back_spread"] = ps["front"]["ret"], ps["back"]["ret"]
        d["full_spread_maker_limit"] = pm["full"]["ret"]
        d["front_spread_maker_limit"], d["back_spread_maker_limit"] = pm["front"]["ret"], pm["back"]["ret"]
        d["full_incl_first_week"] = periods(run(rl, initial_t0=False))["full"]["ret"]
        d["july"] = pr["monthly"]["2026-07"]
        if rl == "T1":
            d["entries"] = [{k: (round(x, 4) if isinstance(x, float) else x) for k, x in e.items()} for e in log]
        desk[rl] = d
    for rl in rules:
        desk[rl]["rel_front_vs_T0_pct"] = round(((1 + desk[rl]["front"] / 100) / (1 + desk["T0"]["front"] / 100) - 1) * 100, 2)
    ms = periods(run("T4a_mon", strict=0.001))
    desk["T4a_mon_strict"] = {"front": ms["front"]["ret"], "back": ms["back"]["ret"], "full": ms["full"]["ret"]}
    res["desk"] = desk
    for rl in rules:
        d = desk[rl]
        print(f"{rl:8s} F {d['front']:7.2f} B {d['back']:6.2f} S {d['sepoct']:6.2f} full {d['full']:7.2f} mdd {d['mdd']:6.2f} "
              f"| rel {d['rel_front_vs_T0_pct']:6.2f} | ent F n{d['entry_front_n']} {d['entry_front_mean']:6.2f} med {d['entry_front_median']:5.2f} "
              f"| noDELL F {d['front_noDELL']:7.2f} no0526 F {d['front_no0526wk']:7.2f} | spr {d['full_spread']:7.2f} maker {d['full_spread_maker_limit']:7.2f}")
    print("T4a_mon strict", desk["T4a_mon_strict"])

    # a) 무작위 체결일 귀무분포
    rng = np.random.default_rng(20261013)
    t0f, t0b = desk["T0"]["front"], desk["T0"]["back"]
    sims = []
    for _ in range(2000):
        p = periods(run("RAND", rng=rng))
        sims.append((p["front"]["ret"] - t0f, p["back"]["ret"] - t0b))
    s = np.array(sims)
    res["random_null"] = {
        "n_sims": 2000,
        "front_diff_pct": {q: round(float(np.percentile(s[:, 0], q)), 2) for q in (5, 25, 50, 75, 95)},
        "back_diff_pct": {q: round(float(np.percentile(s[:, 1], q)), 2) for q in (5, 25, 50, 75, 95)},
        "share_beat_T0_both": round(float(((s[:, 0] > 0) & (s[:, 1] > 0)).mean()) * 100, 1),
        "share_front_ge_T4a_mon": round(float((s[:, 0] >= desk["T4a_mon"]["front"] - t0f).mean()) * 100, 1),
    }
    print("random null", res["random_null"])

    # 3) 일반 집합
    gw = {"front": WIN["front"], "back": WIN["back"], "sepoct": WIN["sepoct"],
          "y2025": ("2025-01-02", "2025-12-31"), "y2024": ("2024-01-02", "2024-12-31")}
    gen = {}
    for rl in ["T1", "T2", "T3", "T4a", "T4b", "T4c", "T5", "Tfri", "T4a_mon"]:
        gen[rl] = {w: generic(rl, a, b) for w, (a, b) in gw.items()}
        gen[rl]["back_exJuly"] = generic(rl, *WIN["back"], skip_month="2026-07")
        print(f"generic {rl:8s}", {w: (x.get("mean"), x.get("t_week"), x.get("n")) for w, x in gen[rl].items()})
    res["generic_top7"] = gen
    neg25 = sum(gen[r]["y2025"]["mean"] < 0 for r in ["T1", "T2", "T3", "T4a", "T4b", "T4c", "T5"])
    res["generic_2025_negative_of_7"] = int(neg25)
    # 앞 구간 계좌 격차 중 DELL·AXTI 두 건이 설명하는 몫(%p)
    conc = {}
    for rl in rules[1:]:
        gap = desk[rl]["front"] - t0f
        gap2 = desk[rl]["front_noDELL_AXTI"] - t0f
        conc[rl] = {"front_gap_pp": round(gap, 2), "gap_wo_DELL_pp": round(desk[rl]["front_noDELL"] - t0f, 2),
                    "gap_wo_DELL_AXTI_pp": round(gap2, 2),
                    "share_explained_pct": round((gap - gap2) / gap * 100, 1) if gap < 0 else None,
                    "beats_T0_both_wo_DELL": bool(desk[rl]["front_noDELL"] > t0f and desk[rl]["back"] > t0b)}
    res["front_gap_concentration"] = conc
    # 일반 집합 54칸 중 주 t >= 2 / <= -2
    cells = [(rl, w, x["t_week"]) for rl, g in gen.items() for w, x in g.items() if isinstance(x.get("t_week"), float)]
    res["generic_grid"] = {"cells": len(cells),
                           "t_ge_2": [(a, b, t) for a, b, t in cells if t >= 2],
                           "t_le_-2": [(a, b, t) for a, b, t in cells if t <= -2],
                           "front_and_back_positive": [rl for rl, g in gen.items()
                                                       if g["front"]["mean"] > 0 and g["back"]["mean"] > 0]}
    print("concentration", conc)
    print("generic grid", res["generic_grid"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
