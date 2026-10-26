"""갈래 G: 재진입 시점과 '그냥 들고 가기(keep)' 전환점 (2026-10-10).

질문: "어느 시점부터는 변동성을 떠나서 그냥 매수해놓고 keep하는 것이 낫다", "다시 언제 사나"를 숫자로.
1부 조건부 전방 수익: 252일 고점 대비 낙폭이 −D를 처음 넘은 날(같은 하락 국면 = 252일 신고가 사이 한 번) 이후 20·60·120거래일.
2부 '깊으면 안 판다·깊으면 산다': 표준 손절(그림자 트레일링 −X → 현금, 그림자 > SMA20 복귀) 위에
    (i) 들고 있는데 그림자 252일 낙폭 ≤ −D면 손절을 끄고 keep(252일 신고가까지),
    (ii) 나가 있는 동안 그림자 낙폭 ≤ −D에 닿으면 무조건 다시 사서 keep(252일 신고가까지), (iii) 둘 다.
3부 재진입 규칙 비교: 고정 이탈(그림자 트레일링 −12% / SOXX 종가 < SMA50 하향) × 재진입 15종.

시점: 모든 신호는 t일 종가(그림자 = 기본 경로 종가 평가, VIX·SOXX 종가)로 정하고 H.at_open으로 t+1 시가에 체결.
문턱(SMA·변동성·낙폭)은 모두 t까지의 이동·확장 창. 전체 기간 통계로 정한 문턱 없음.
그림자 낙폭은 각 기본 경로 자체(2026은 1/2부터 확장 고점, 보유 기간은 252일 이동 고점).
2026 그림자의 SMA·변동성 예열에는 보유 기간 기본 경로(같은 규칙, 주간 PDF) 끝부분을 이어 붙인다(1/2 종가에 맞춰 배율).
실행: cd code && PYTHONIOENCODING=utf-8 py fam_G_hold_reentry.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import harness as H

FAMILY = "G_hold_reentry"

# ============================================================================ 격자(사전 선언, 사후 수정 금지)
P1_DEPTHS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
P1_HORIZONS = [20, 60, 120]
P1_FURTHER = 0.10                      # 진입 뒤 그 기간 안에 진입가 대비 −10% 더 빠질 확률
P1_ROLL = 252

P2_X = [0.08, 0.10, 0.12]              # 표준 손절 트레일링 문턱. 0.10이 지시된 표준, 0.08·0.12는 민감도
P2_REENTRY_SMA = 20
P2_D = [0.15, 0.20, 0.25, 0.30]        # (i) 안 판다 / (ii) 산다 깊이
# 변형 = X × {표준, (i)D 4, (ii)D 4, (iii)Di×Dii 16} = 3 × 25 = 75

P3_EXITS = [("trail", 0.12), ("soxx_sma50", None)]
P3_REENTRY = [("time", 5), ("time", 10), ("time", 20), ("time", 40),
              ("bounce", 0.05), ("bounce", 0.10), ("bounce", 0.15),
              ("vix_ma20", None), ("vix_lt20", 20.0), ("vol10_40", None),
              ("sma", 10), ("sma", 20), ("sma", 50),
              ("stage_sma", (10, 20, 50)), ("stage_bounce", (0.05, 0.10, 0.15))]
# 변형 = 2 × 15 = 30.  합계 2·3부 = 105

GRID_DESC = ("2부 75 = 트레일링 손절 X {8,10,12}% × [표준 1 + (i)안 판다 D{15,20,25,30} 4 + (ii)깊으면 산다 D 4 + "
             "(iii) Di×Dii 16], 재진입 그림자>SMA20, keep은 252일 신고가까지. "
             "3부 30 = 이탈 {그림자 트레일링 −12%, SOXX<SMA50 하향} × 재진입 15 {시간 5·10·20·40일, 저점+5·10·15%, "
             "VIX<20일평균, VIX<20, 10일변동성<40일변동성, 그림자>SMA10·20·50, 단계(SMA10/20/50 각 1/3), "
             "단계(저점+5/10/15% 각 1/3)}. 합계 105. 1부(조건부 전방 수익)는 깊이 6 × 기간 3, 시험 변형 아님.")


# ============================================================================ 신호
def ffill_to(s: pd.Series, days) -> pd.Series:
    s = s.dropna().sort_index()
    u = s.reindex(sorted(set(s.index) | set(days))).ffill()
    return u.reindex(days)


_SIG: dict = {}


def signals(bn: str) -> pd.DataFrame:
    """t일 종가에 아는 값만(index = t). 체결은 at_open으로 t+1 시가."""
    if bn in _SIG:
        return _SIG[bn]
    b = H.base(bn)
    c = b.close_s.copy()
    if bn == "2026":                      # 지표 예열: 보유 기간 경로 끝을 1/2 종가에 맞춰 붙인다
        hc = H.base("hold").close_s
        pre = hc[hc.index < b.days[0]].iloc[-80:]
        ext = pd.concat([pre * (c.iloc[0] / pre.iloc[-1]), c])
    else:
        ext = c
    s = pd.DataFrame(index=b.days)
    s["c"] = c
    s["dd"] = c / c.rolling(P1_ROLL, min_periods=1).max() - 1
    for n in (10, 20, 50):
        s[f"sma{n}"] = ext.rolling(n, min_periods=1).mean().reindex(b.days)
    lr = np.log(ext).diff()
    s["vol10"] = lr.rolling(10, min_periods=5).std().reindex(b.days)
    s["vol40"] = lr.rolling(40, min_periods=20).std().reindex(b.days)
    vix = H.px("^VIX")
    s["vix"] = ffill_to(vix, b.days)
    s["vix_ma20"] = ffill_to(vix.rolling(20, min_periods=20).mean(), b.days)
    sx = H.px("SOXX")
    s["soxx"] = ffill_to(sx, b.days)
    s["soxx_sma50"] = ffill_to(sx.rolling(50, min_periods=50).mean(), b.days)
    _SIG[bn] = s
    return s


# ============================================================================ 상태 기계
def simulate(sig: pd.DataFrame, exit_kind: str, X: float | None, reentry: tuple,
             no_sell_D: float | None = None, deep_buy_D: float | None = None):
    """반환: (종가 뒤 노출 시계열 index=t, 사건 목록). 노출은 t+1 시가부터 적용(at_open)."""
    c, dd = sig.c.to_numpy(), sig.dd.to_numpy()
    sma = {n: sig[f"sma{n}"].to_numpy() for n in (10, 20, 50)}
    v10, v40 = sig.vol10.to_numpy(), sig.vol40.to_numpy()
    vix, vixm = sig.vix.to_numpy(), sig.vix_ma20.to_numpy()
    sx, sxm = sig.soxx.to_numpy(), sig.soxx_sma50.to_numpy()
    n = len(c)
    mode, peak, low, xi, stage, armed = "in", c[0], None, None, 0, True
    expo = np.ones(n)
    eps = []
    rk, rv = reentry
    for i in range(n):
        if mode == "in":
            peak = max(peak, c[i])
            if no_sell_D is not None and dd[i] <= -no_sell_D:
                mode = "keep"
                eps.append({"type": "keep_hold", "i": i})
            else:
                if exit_kind == "trail":
                    trig = c[i] / peak - 1 <= -X
                else:
                    ok = np.isfinite(sx[i]) and np.isfinite(sxm[i])
                    if ok and sx[i] >= sxm[i]:
                        armed = True
                    trig = ok and armed and sx[i] < sxm[i]
                if trig:
                    mode, xi, low, stage, armed = "out", i, c[i], 0, False
        elif mode == "out":
            low = min(low, c[i])
            back = False
            if deep_buy_D is not None and dd[i] <= -deep_buy_D:
                mode = "keep"
                eps.append({"type": "out", "xi": xi, "ri": i, "why": "deep"})
            else:
                if rk == "time":
                    back = i - xi >= rv
                elif rk == "bounce":
                    back = c[i] >= low * (1 + rv)
                elif rk == "vix_ma20":
                    back = np.isfinite(vixm[i]) and vix[i] < vixm[i]
                elif rk == "vix_lt20":
                    back = vix[i] < rv
                elif rk == "vol10_40":
                    back = np.isfinite(v10[i]) and np.isfinite(v40[i]) and v10[i] < v40[i]
                elif rk == "sma":
                    back = c[i] > sma[rv][i]
                elif rk == "stage_sma":
                    stage = max(stage, sum(c[i] > sma[m][i] for m in rv))
                    back = stage >= 3
                elif rk == "stage_bounce":
                    stage = max(stage, sum(c[i] >= low * (1 + q) for q in rv))
                    back = stage >= 3
                if back:
                    mode, peak = "in", c[i]
                    eps.append({"type": "out", "xi": xi, "ri": i, "why": rk})
        elif mode == "keep":
            if dd[i] >= -1e-9:
                mode, peak = "in", c[i]
        if mode in ("in", "keep"):
            expo[i] = 1.0
        else:
            expo[i] = stage / 3 if rk in ("stage_sma", "stage_bounce") else 0.0
    if mode == "out":
        eps.append({"type": "out", "xi": xi, "ri": None, "why": "end"})
    return pd.Series(expo, index=sig.index), eps


def episode_table(bn: str, eps: list, e_open: pd.Series) -> list:
    """나간 구간별: 이탈 체결(t+1 시가) → 저점(그림자 종가 최저) → 재진입 체결(t+1 시가)."""
    b = H.base(bn)
    o, c, days = np.asarray(b.open), np.asarray(b.close), b.days
    r = b.ret
    out = []
    for ep in eps:
        if ep["type"] != "out":
            continue
        xi, ri = ep["xi"], ep["ri"]
        xe = xi + 1
        if xe >= len(days):
            continue
        re_ = (ri + 1) if ri is not None and ri + 1 < len(days) else len(days) - 1
        lo_seg = c[xe:(ri + 1 if ri is not None else len(days))]
        low = float(np.min(lo_seg)) if len(lo_seg) else float(c[xe])
        rr = r.iloc[xe:re_]
        ee = e_open.reindex(rr.index)
        rel = float(np.prod(1 + ee * rr) / np.prod(1 + rr) - 1) if len(rr) else 0.0
        out.append({"exit_sig": days[xi], "exit_exec": days[xe], "re_exec": days[re_] if ri is not None else "끝(미복귀)",
                    "why": ep["why"], "days_out": int(re_ - xe), "shadow_while_out": round(float(o[re_] / o[xe] - 1), 4),
                    "saved_to_low": round(low / float(o[xe]) - 1, 4), "missed_from_low": round(float(o[re_]) / low - 1, 4),
                    "variant_vs_base": round(rel, 4)})
    return out


# ============================================================================ 1부: 조건부 전방 수익
def cond_events(op: pd.Series, cl: pd.Series, start: str | None = None, end: str | None = None, warm: int = 0):
    """op/cl 같은 index. 낙폭 = cl/252일 최고 cl − 1. 국면 = 252일 신고가(낙폭 0) 사이. 진입 = 다음날 시가."""
    cl = cl.dropna()
    op = op.reindex(cl.index)
    op = op.where(op > 0, cl.shift(1))
    dd = cl / cl.rolling(P1_ROLL, min_periods=1).max() - 1
    C, O, idx = cl.to_numpy(), op.to_numpy(), cl.index
    hit: set = set()
    ev = {D: [] for D in P1_DEPTHS}
    for i in range(len(C)):
        if dd.iloc[i] >= -1e-12:
            hit = set()
        for D in P1_DEPTHS:
            if D not in hit and dd.iloc[i] <= -D:
                hit.add(D)
                d = idx[i]
                if i < warm or (start and d < start) or (end and d > end):
                    continue
                row = {"date": d, "dd": round(float(dd.iloc[i]), 4)}
                if i + 1 < len(C):
                    ent = O[i + 1] if np.isfinite(O[i + 1]) else C[i]
                    for h in P1_HORIZONS:
                        if i + h < len(C):
                            row[f"r{h}"] = float(C[i + h] / ent - 1)
                            row[f"f{h}"] = bool(np.min(C[i + 1:i + h + 1]) / ent - 1 <= -P1_FURTHER)
                ev[D].append(row)
    return ev


def uncond(op: pd.Series, cl: pd.Series, start=None, end=None, warm=0):
    cl = cl.dropna()
    op = op.reindex(cl.index)
    op = op.where(op > 0, cl.shift(1))
    C, O, idx = cl.to_numpy(), op.to_numpy(), cl.index
    out = {}
    for h in P1_HORIZONS:
        rs, fs = [], []
        for i in range(warm, len(C) - h):
            d = idx[i]
            if (start and d < start) or (end and d > end):
                continue
            ent = O[i + 1] if np.isfinite(O[i + 1]) else C[i]
            rs.append(C[i + h] / ent - 1)
            fs.append(np.min(C[i + 1:i + h + 1]) / ent - 1 <= -P1_FURTHER)
        out[h] = {"n": len(rs), "mean": float(np.mean(rs)) if rs else np.nan,
                  "med": float(np.median(rs)) if rs else np.nan, "win": float(np.mean(np.array(rs) > 0)) if rs else np.nan,
                  "p_further": float(np.mean(fs)) if fs else np.nan}
    return out


def summarize_ev(ev):
    out = {}
    for D, rows in ev.items():
        s = {"n": len(rows)}
        for h in P1_HORIZONS:
            x = [r[f"r{h}"] for r in rows if f"r{h}" in r]
            f = [r[f"f{h}"] for r in rows if f"f{h}" in r]
            s[h] = {"n": len(x), "mean": float(np.mean(x)) if x else None, "med": float(np.median(x)) if x else None,
                    "win": float(np.mean(np.array(x) > 0)) if x else None, "p_further": float(np.mean(f)) if f else None}
        out[D] = s
    return out


def part1():
    series = {}
    for bn, lab in (("2026", "7종목 그림자 2026"), ("hold", "7종목 그림자 2023-05~2025")):
        b = H.base(bn)
        series[lab] = (pd.Series(b.open, index=b.days), b.close_s, None, None, 0)
    bh = H.base("hold")                   # 같은 보유 기간 경로에서 2025년에 생긴 사건만(국면 상태는 2023부터 이어짐)
    series["7종목 그림자 2025만"] = (pd.Series(bh.open, index=bh.days), bh.close_s, "2025-01-01", "2025-12-31", 0)
    for sym in ("SOXX", "QQQ"):           # ohlc 2022-06~ → 252일 창이 찬 뒤(2023-06~)만 사건으로 센다
        series[f"{sym} 2023-06~2026-10"] = (H.px(sym, "Open"), H.px(sym, "Close"), "2023-06-01", None, 0)
    lg = pd.read_parquet(H.CACHE / "G_long.parquet")
    for sym, lab in (("^SOX", "^SOX 1995~(맥락)"), ("^NDX", "^NDX 1986~(맥락)"), ("QQQ", "QQQ 2000~(맥락)")):
        g = lg[lg.sym == sym].set_index("date").sort_index()
        series[lab] = (g["Open"].astype(float), g["Close"].astype(float), None, None, P1_ROLL)
    res, events, base_u = {}, {}, {}
    for lab, (op, cl, st, en, warm) in series.items():
        ev = cond_events(op, cl, st, en, warm)
        res[lab] = summarize_ev(ev)
        base_u[lab] = uncond(op, cl, st, en, warm)
        events[lab] = {D: [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
                       for D, rows in ev.items()}
    return res, base_u, events


def p1_markdown(res, base_u):
    lines = ["| 대상 | 낙폭 | n | 20일 평균/중앙/플러스 | 60일 평균/중앙/플러스 | 120일 평균/중앙/플러스 | 60일 안 추가 −10% |",
             "|---|---|---|---|---|---|---|"]
    f = lambda s: "-" if s["n"] == 0 or s["mean"] is None else f"{s['mean']:+.1%}/{s['med']:+.1%}/{s['win']:.0%} (n{s['n']})"
    for lab, by in res.items():
        u = base_u[lab]
        lines.append(f"| {lab} | 무조건(모든 날) | - | {u[20]['mean']:+.1%}/{u[20]['med']:+.1%}/{u[20]['win']:.0%} | "
                     f"{u[60]['mean']:+.1%}/{u[60]['med']:+.1%}/{u[60]['win']:.0%} | "
                     f"{u[120]['mean']:+.1%}/{u[120]['med']:+.1%}/{u[120]['win']:.0%} | {u[60]['p_further']:.0%} |")
        for D, s in by.items():
            if s["n"] == 0:
                continue
            pf = s[60]["p_further"]
            lines.append(f"| {lab} | −{D:.0%} | {s['n']} | {f(s[20])} | {f(s[60])} | {f(s[120])} | "
                         f"{'-' if pf is None else f'{pf:.0%}'} |")
    return "\n".join(lines)


# ============================================================================ 2·3부 변형
def variants():
    out = []
    for X in P2_X:
        re_ = ("sma", P2_REENTRY_SMA)
        out.append((f"P2 X{X:.0%} 표준", "trail", X, re_, None, None))
        for D in P2_D:
            out.append((f"P2 X{X:.0%} (i)안판다 D{D:.0%}", "trail", X, re_, D, None))
        for D in P2_D:
            out.append((f"P2 X{X:.0%} (ii)산다 D{D:.0%}", "trail", X, re_, None, D))
        for Di in P2_D:
            for Dii in P2_D:
                out.append((f"P2 X{X:.0%} (iii) Di{Di:.0%} Dii{Dii:.0%}", "trail", X, re_, Di, Dii))
    for ek, ev in P3_EXITS:
        el = "그림자−12%" if ek == "trail" else "SOXX<SMA50"
        for rk, rv in P3_REENTRY:
            out.append((f"P3 {el} → {rk}{'' if rv is None else rv}", ek, ev, (rk, rv), None, None))
    return out


def run_variant(spec):
    name, ek, X, re_, Di, Dii = spec
    ex, eps_by = {}, {}
    for bn in ("2026", "hold"):
        b = H.base(bn)
        e_close, eps = simulate(signals(bn), ek, X, re_, Di, Dii)
        ex[bn] = H.at_open(e_close, b, fill=1.0)
        eps_by[bn] = eps
    params = {"exit": ek, "X": X, "reentry": list(re_) if not isinstance(re_[1], tuple) else [re_[0], list(re_[1])],
              "no_sell_D": Di, "deep_buy_D": Dii}
    row = H.evaluate(name, ex, None, params)
    return row, ex, eps_by


def win_effect(bn, e_open, lo, hi):
    b = H.base(bn)
    r = b.ret[(b.ret.index >= lo) & (b.ret.index <= hi)]
    ee = e_open.reindex(r.index)
    return float(np.prod(1 + ee * r) - 1), float(np.prod(1 + r) - 1)


def main():
    res1, u1, ev1 = part1()
    specs = variants()
    print("2·3부 변형 수", len(specs))
    rows, cache = [], {}
    for sp in specs:
        row, ex, eps = run_variant(sp)
        rows.append(row)
        cache[sp[0]] = (ex, eps, sp)

    # 사건 표(아낀 손실·놓친 반등): 3부 전 규칙 + 2부 X10 주요
    ev_tab = {}
    for name, (ex, eps, sp) in cache.items():
        if name.startswith("P3") or (name.startswith("P2 X10%") and ("표준" in name or "(i)" in name or "(ii)" in name)):
            ev_tab[name] = {bn: episode_table(bn, eps[bn], ex[bn]) for bn in ("2026", "hold")}

    # 규칙별 요약(2026·2025): 나간 횟수, 평균 아낀 손실, 평균 놓친 반등, 기본 대비 합산 효과
    def ev_sum(tab, lo, hi):
        t = [x for x in tab if lo <= x["exit_exec"] <= hi]
        if not t:
            return {"n": 0}
        rel = float(np.prod([1 + x["variant_vs_base"] for x in t]) - 1)
        return {"n": len(t), "saved_avg": float(np.mean([x["saved_to_low"] for x in t])),
                "missed_avg": float(np.mean([x["missed_from_low"] for x in t])),
                "days_out_avg": float(np.mean([x["days_out"] for x in t])), "rel_total": rel}

    ev_summary = {name: {"2026": ev_sum(t["2026"], "2026-01-01", "2026-12-31"),
                         "2025": ev_sum(t["hold"], "2025-01-01", "2025-12-31"),
                         "2024": ev_sum(t["hold"], "2024-01-01", "2024-12-31")} for name, t in ev_tab.items()}

    br = H.base_result()
    picks = {k: H.select_front(rows, br, kv) for k, kv in (("keep90", 0.9), ("keep80", 0.8), ("keep70", 0.7))}
    par = sorted(H.pareto_2026(rows, br, 0.90), key=lambda r: -r["2026"]["전체"]["수익"])
    focus = picks["keep90"] or picks["keep80"] or picks["keep70"] or (par[0] if par else None)
    focus_label = ("keep90" if picks["keep90"] else "keep80" if picks["keep80"] else "keep70" if picks["keep70"]
                   else "사후 파레토 최선")

    # 7월 쇼크·2~3월 행동
    def behavior(name):
        ex, eps, sp = cache[name]
        e26 = ex["2026"]
        tr = H.trace(e26)
        tab = episode_table("2026", eps["2026"], e26)
        feb = [x for x in tab if "2026-01-26" <= x["exit_exec"] <= "2026-04-10"]
        jul = [x for x in tab if "2026-06-15" <= x["exit_exec"] <= "2026-09-30"]
        w = {}
        for lab, (lo, hi) in {"2~3월(1/26~4/10)": ("2026-01-26", "2026-04-10"), "AI쇼크(6/22~7/29)": ("2026-06-22", "2026-07-29"),
                              "7~9월": ("2026-07-01", "2026-09-30"), "6/15~9/30": ("2026-06-15", "2026-09-30")}.items():
            v, bse = win_effect("2026", e26, lo, hi)
            w[lab] = {"variant": round(v, 4), "base": round(bse, 4)}
        keeps = [H.base("2026").days[ep["i"]] for ep in eps["2026"] if ep["type"] == "keep_hold"]
        return {"trace_6_15_9_30": tr, "feb_mar_episodes": feb, "jul_sep_episodes": jul, "windows": w,
                "keep_hold_days": keeps}

    beh = {}
    names_for_beh = []
    for k, v in picks.items():
        if v:
            names_for_beh.append(v["name"])
    for r in par[:3]:
        names_for_beh.append(r["name"])
    for nm in ("P2 X10% 표준", "P2 X10% (i)안판다 D15%", "P2 X10% (ii)산다 D15%", "P2 X10% (ii)산다 D20%"):
        names_for_beh.append(nm)
    for nm in dict.fromkeys(names_for_beh):
        beh[nm] = behavior(nm)

    # keep 전환점 표: X10 표준 vs (i)D vs (ii)D vs (iii)D=D, 2026 앞·뒤·전체, 2025, 2024
    def g(r, bn, k):
        return r[bn][k]["수익"], r[bn][k]["낙폭"]
    byname = {r["name"]: r for r in rows}
    keep_tab = []
    for X in P2_X:
        for lab in ["표준"] + [f"(i)안판다 D{D:.0%}" for D in P2_D] + [f"(ii)산다 D{D:.0%}" for D in P2_D] + \
                   [f"(iii) Di{D:.0%} Dii{D:.0%}" for D in P2_D]:
            nm = f"P2 X{X:.0%} {lab}"
            r = byname[nm]
            keep_tab.append({"name": nm, **{f"{bn}_{k}": [round(v, 4) for v in g(r, bn, k)]
                                             for bn, k in (("2026", "전체"), ("2026", "앞"), ("2026", "뒤"),
                                                           ("2026", "AI쇼크"), ("hold", "2025"), ("hold", "2024"),
                                                           ("hold", "관세"), ("hold", "25가을"), ("hold", "24여름"))},
                             "노출": round(r["2026"]["평균노출"], 3), "빠짐": r["2026"]["빠짐횟수"]})

    extra = {"part1_summary": {lab: {str(D): v for D, v in by.items()} for lab, by in res1.items()},
             "part1_unconditional": {lab: {str(h): v for h, v in by.items()} for lab, by in u1.items()},
             "part1_events_shadow": {lab: {str(D): v for D, v in ev1[lab].items()}
                                     for lab in ev1 if lab.startswith("7종목") or lab.startswith("SOXX") or lab.startswith("QQQ 2023")},
             "part1_md": p1_markdown(res1, u1),
             "episode_summary": ev_summary, "episodes": ev_tab, "behavior": beh, "keep_table": keep_tab,
             "focus": {"label": focus_label, "name": focus["name"] if focus else None}}
    payload = H.summarize_family(FAMILY, rows, GRID_DESC, extra)
    print(p1_markdown(res1, u1))
    return payload


if __name__ == "__main__":
    main()
