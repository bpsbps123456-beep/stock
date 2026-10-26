"""v2 반박 검증: 후보 dd10_buy15 (2026-10-10).

후보(갈래 G 사후 최선 'P2 X10% (ii)산다 D15%'):
  그림자(기본 경로 종가 평가) 트레일링 고점 대비 −X(10%) → 다음 시가 전량 현금 → 그림자 > SMA20이면 복귀(고점 다시 잡음).
  나가 있는 동안 그림자 252일 고점 대비 낙폭 ≤ −D(15%) → 다음 시가 되사기 → 252일 신고가까지 손절 끔(keep).
harness.py를 import하지 않는다. base_*.json만으로 신호·노출·수익·구간 지표를 처음부터 다시 계산한다.
실행: cd code && PYTHONIOENCODING=utf-8 py v2_dd10_buy15.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "v2_dd10_buy15"

P26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
PH = {"전체": ("2023-05-22", "2025-12-31"), "2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"),
      "관세": ("2025-02-18", "2025-04-04"), "24여름": ("2024-06-18", "2024-08-05"), "25가을": ("2025-10-13", "2025-11-21")}

CLAIM = {"y26_all_ret": 2.593, "y26_all_mdd": -0.166, "y26_back_ret": 0.161, "y26_back_mdd": -0.126,
         "ai_shock_ret": -0.084, "jul_sep_ret": 0.172, "y25_ret": 0.540, "y25_mdd": -0.442, "y24_ret": 0.547,
         "avg_exposure_2026": 0.92, "exits_2026": 6, "d20_y26": 1.679, "d20_y25": 0.388}

# 이웃 격자(과제 지정): X {8..12}% × D {12,13,14,15,16,17,18,20}% × 해제 3종, X<D만
NB_X = [0.08, 0.09, 0.10, 0.11, 0.12]
NB_D = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]
NB_REL = ["nh252", "peak", "fix60"]
REL_KO = {"nh252": "252일 신고가", "peak": "손절 기준 고점 회복", "fix60": "60거래일 고정", "none": "keep 없음(즉시 손절 재가동)"}


# ============================================================================ 자료
class B:
    def __init__(self, name):
        d = json.loads((CACHE / f"base_{name}.json").read_text(encoding="utf-8"))
        self.name = name
        self.days = list(d["days"])
        self.open = np.asarray(d["open"], float)
        self.close = np.asarray(d["close"], float)
        self.n = len(self.days)
        self.ret = self.open[1:] / self.open[:-1] - 1          # ret[k]: k일 시가 → k+1일 시가, k = 0..n-2
        self.rdays = np.asarray(self.days[:-1])


BASES = {"2026": B("2026"), "hold": B("hold")}
PER = {"2026": P26, "hold": PH}


def signals(bn: str, warm: bool = True, chain_peak: bool = False):
    """t일 종가에 아는 값: 그림자 종가 c, SMA20(t 포함), 252일 고점 대비 낙폭(t 포함)."""
    b = BASES[bn]
    c = pd.Series(b.close, index=b.days)
    if bn == "2026" and (warm or chain_peak):
        hc = pd.Series(BASES["hold"].close, index=BASES["hold"].days)
        scale = c.iloc[0] / hc.iloc[-1]
    if bn == "2026" and warm:
        pre = hc.iloc[-80:]
        ext = pd.concat([pre * (c.iloc[0] / pre.iloc[-1]), c])
    else:
        ext = c
    sma20 = ext.rolling(20, min_periods=1).mean().reindex(b.days).to_numpy()
    if bn == "2026" and chain_peak:                 # 민감도: 2025년 고점을 이어 붙인 252일 고점
        ext2 = pd.concat([hc * scale, c])
        ext2 = ext2[~ext2.index.duplicated(keep="last")]
        dd = (ext2 / ext2.rolling(252, min_periods=1).max() - 1).reindex(b.days).to_numpy()
    else:
        dd = (c / c.rolling(252, min_periods=1).max() - 1).to_numpy()
    return c.to_numpy(), sma20, dd


# ============================================================================ 상태 기계(독립 구현)
def simulate(c, sma, dd, X, D, release="nh252", keep_n=60):
    """반환: expo_close[i](i일 종가 뒤 정한 노출, i+1 시가부터), 사건 목록."""
    n = len(c)
    mode, peak = "in", c[0]
    xi = ki = None
    exit_peak = None
    expo = np.ones(n)
    eps = []
    for i in range(n):
        if mode == "in":
            peak = max(peak, c[i])
            if c[i] / peak - 1 <= -X:
                mode, xi, exit_peak = "out", i, peak
        elif mode == "out":
            if D is not None and dd[i] <= -D:
                eps.append({"why": "deep", "xi": xi, "ri": i})
                if release == "none":
                    mode, peak = "in", c[i]
                else:
                    mode, ki = "keep", i
            elif c[i] > sma[i]:
                eps.append({"why": "sma", "xi": xi, "ri": i})
                mode, peak = "in", c[i]
        elif mode == "keep":
            if release == "nh252":
                rel = dd[i] >= -1e-9
            elif release == "peak":
                rel = c[i] >= exit_peak
            elif release == "fix60":
                rel = i - ki >= keep_n
            else:
                rel = True
            if rel:
                mode, peak = "in", c[i]
        expo[i] = 0.0 if mode == "out" else 1.0
    if mode == "out":
        eps.append({"why": "end", "xi": xi, "ri": None})
    return expo, eps


def to_open(expo_close, lag=1):
    """t일 종가 노출 → t+lag 시가 적용. 앞쪽 빈 날은 1.0(기본 경로 그대로)."""
    e = np.ones(len(expo_close))
    e[lag:] = expo_close[:-lag]
    return e


def path_stats(r, rdays, lo, hi):
    m = (rdays >= lo) & (rdays <= hi)
    x = r[m]
    c = np.concatenate([[1.0], np.cumprod(1 + x)])
    return float(c[-1] - 1), float((c / np.maximum.accumulate(c) - 1).min())


def evaluate(bn, e_open, cost=0.0):
    b = BASES[bn]
    e = e_open[:-1]
    r = e * b.ret
    if cost:
        turn = np.abs(np.diff(np.concatenate([[1.0], e])))
        r = r - cost * turn
    out = {k: dict(zip(("ret", "mdd"), path_stats(r, b.rdays, *v))) for k, v in PER[bn].items()}
    prev = np.concatenate([[1.0], e[:-1]])
    out["avg_e"] = float(e.mean())
    out["exits"] = int(((prev >= 0.999) & (e < 0.999)).sum())
    for k, (lo, hi) in PER[bn].items():
        m = (b.rdays >= lo) & (b.rdays <= hi)
        out[f"avg_e_{k}"] = float(e[m].mean())
        out[f"exits_{k}"] = int(((prev >= 0.999) & (e < 0.999) & m).sum())
    return out


def fixed(bn, e_const):
    return evaluate(bn, np.full(BASES[bn].n, float(e_const)))


def run(X, D, release="nh252", lag=1, warm=True, chain_peak=False, cost=0.0, keep_n=60):
    res, ex, eps = {}, {}, {}
    for bn in ("2026", "hold"):
        c, s, dd = signals(bn, warm=warm, chain_peak=chain_peak)
        ec, ep = simulate(c, s, dd, X, D, release, keep_n)
        eo = to_open(ec, lag)
        res[bn] = evaluate(bn, eo, cost)
        ex[bn], eps[bn] = eo, ep
    return res, ex, eps


def episodes(bn, eps, lag=1):
    """나간 구간: 체결일, 사유, 현금 일수, 그 사이 기본 경로 수익, 기본 대비 효과(= 1/(1+기본) − 1)."""
    b = BASES[bn]
    out = []
    for ep in eps:
        xe = ep["xi"] + lag
        re_ = (ep["ri"] + lag) if ep["ri"] is not None else b.n - 1
        if xe >= b.n:
            continue
        re_ = min(re_, b.n - 1)
        base_ret = float(b.open[re_] / b.open[xe] - 1)
        out.append({"exit_exec": b.days[xe], "re_exec": b.days[re_] if ep["ri"] is not None else "끝",
                    "why": ep["why"], "days_out": int(re_ - xe), "base_while_out": round(base_ret, 4),
                    "rel_vs_base": round(1 / (1 + base_ret) - 1, 4)})
    return out


def ep_sum(tab, lo, hi, why=None):
    t = [x for x in tab if lo <= x["exit_exec"] <= hi and (why is None or x["why"] == why)]
    if not t:
        return {"n": 0, "rel": 0.0, "days": 0}
    return {"n": len(t), "rel": float(np.prod([1 + x["rel_vs_base"] for x in t]) - 1),
            "days": int(sum(x["days_out"] for x in t))}


# ============================================================================ 본문
def main():
    base = {bn: evaluate(bn, np.ones(BASES[bn].n)) for bn in ("2026", "hold")}
    b26, bh = base["2026"], base["hold"]
    print("기본 2026", {k: (round(v['ret'], 4), round(v['mdd'], 4)) for k, v in b26.items() if isinstance(v, dict)})
    print("기본 hold", {k: (round(v['ret'], 4), round(v['mdd'], 4)) for k, v in bh.items() if isinstance(v, dict)})

    # ---- 1) 독립 재계산
    cand, cex, ceps = run(0.10, 0.15, "nh252")
    c26, ch = cand["2026"], cand["hold"]
    rec = {"y26_all_ret": c26["전체"]["ret"], "y26_all_mdd": c26["전체"]["mdd"], "y26_front_ret": c26["앞"]["ret"],
           "y26_front_mdd": c26["앞"]["mdd"], "y26_back_ret": c26["뒤"]["ret"], "y26_back_mdd": c26["뒤"]["mdd"],
           "ai_shock_ret": c26["AI쇼크"]["ret"], "jul_sep_ret": c26["7~9월"]["ret"], "y25_ret": ch["2025"]["ret"],
           "y25_mdd": ch["2025"]["mdd"], "y24_ret": ch["2024"]["ret"], "y24_mdd": ch["2024"]["mdd"],
           "avg_exposure_2026": c26["avg_e"], "exits_2026": c26["exits"], "hold_all_ret": ch["전체"]["ret"]}
    d20, _, _ = run(0.10, 0.20, "nh252")
    rec["d20_y26"], rec["d20_y25"] = d20["2026"]["전체"]["ret"], d20["hold"]["2025"]["ret"]
    diffs = {k: rec[k] - v for k, v in CLAIM.items() if k in rec}
    print("재계산", {k: round(v, 4) for k, v in rec.items()})
    print("차(재계산−주장)", {k: round(v, 4) for k, v in diffs.items()})

    # ---- 2) 시점 감사: t+2 시가, SMA 예열 없음, 2025 고점 이어 붙이기, 스프레드
    lag2, _, _ = run(0.10, 0.15, "nh252", lag=2)
    nowarm, _, _ = run(0.10, 0.15, "nh252", warm=False)
    chain, chex, cheps = run(0.10, 0.15, "nh252", chain_peak=True)
    cost, _, _ = run(0.10, 0.15, "nh252", cost=0.001)
    std, sex, seps = run(0.10, None)                      # 손절만
    std_lag2, _, _ = run(0.10, None, lag=2)
    none_keep, _, _ = run(0.10, 0.15, "none")             # 되사기만 하고 keep 없음

    # ---- 4) 고정 노출(실제로 그 노출로 재계산)
    e26 = round(c26["avg_e"], 3)
    fx26 = {"2026": fixed("2026", e26), "hold": fixed("hold", e26)}
    fx_own = {"2025": fixed("hold", ch["avg_e_2025"]), "2024": fixed("hold", ch["avg_e_2024"]),
              "hold_all": fixed("hold", ch["avg_e_전체"])}

    # ---- 3) 이웃 격자
    nb = []
    front_need = 0.9 * (1 + b26["앞"]["ret"]) - 1
    for rel in NB_REL:
        for X in NB_X:
            for D in NB_D:
                if not X < D:
                    continue
                r, _, _ = run(X, D, rel)
                r2, _, _ = run(X, D, rel, lag=2)
                a, h = r["2026"], r["hold"]
                f25 = fixed("hold", h["avg_e_2025"])["2025"]["ret"]
                f24 = fixed("hold", h["avg_e_2024"])["2024"]["ret"]
                f26 = fixed("2026", a["avg_e"])["전체"]["ret"]
                row = {"X": X, "D": D, "release": rel,
                       "y26": a["전체"]["ret"], "y26_mdd": a["전체"]["mdd"], "front": a["앞"]["ret"], "back": a["뒤"]["ret"],
                       "ai": a["AI쇼크"]["ret"], "y25": h["2025"]["ret"], "y24": h["2024"]["ret"], "hold": h["전체"]["ret"],
                       "avg_e26": a["avg_e"], "avg_e25": h["avg_e_2025"], "avg_e24": h["avg_e_2024"],
                       "fix26": f26, "fix25": f25, "fix24": f24, "exits26": a["exits"],
                       "lag2_y26": r2["2026"]["전체"]["ret"], "lag2_back": r2["2026"]["뒤"]["ret"],
                       "lag2_ai": r2["2026"]["AI쇼크"]["ret"], "lag2_y25": r2["hold"]["2025"]["ret"]}
                row["pass26"] = bool(row["front"] >= front_need and row["back"] > b26["뒤"]["ret"] and row["ai"] > b26["AI쇼크"]["ret"])
                row["pass26_lag2"] = bool(r2["2026"]["앞"]["ret"] >= front_need and row["lag2_back"] > b26["뒤"]["ret"]
                                          and row["lag2_ai"] > b26["AI쇼크"]["ret"])
                row["beat_fix25"] = bool(row["y25"] > f25)
                row["beat_fix24"] = bool(row["y24"] > f24)
                row["beat_fix26"] = bool(row["y26"] > f26)
                row["beat_base26"] = bool(row["y26"] > b26["전체"]["ret"])
                row["beat_base25"] = bool(row["y25"] > bh["2025"]["ret"])
                row["beat_base24"] = bool(row["y24"] > bh["2024"]["ret"])
                row["beat_base_hold"] = bool(row["hold"] > bh["전체"]["ret"])
                nb.append(row)
    N = len(nb)

    def cnt(key, sub=None):
        s = nb if sub is None else [r for r in nb if sub(r)]
        return sum(r[key] for r in s), len(s)

    agg = {k: cnt(k) for k in ("pass26", "pass26_lag2", "beat_fix25", "beat_fix24", "beat_fix26", "beat_base26",
                               "beat_base25", "beat_base24", "beat_base_hold")}
    agg["all_three"] = (sum(r["pass26"] and r["beat_fix25"] and r["beat_fix24"] for r in nb), N)
    by_rel = {rel: {k: cnt(k, lambda r, rel=rel: r["release"] == rel) for k in
                    ("pass26", "beat_fix25", "beat_fix24", "beat_base26", "beat_base25", "beat_base24", "beat_base_hold")}
              for rel in NB_REL}
    by_D = {f"{D:.2f}": {k: cnt(k, lambda r, D=D: r["D"] == D) for k in ("pass26", "beat_base26", "beat_base25", "beat_fix25")}
            for D in NB_D}
    by_X = {f"{X:.2f}": {k: cnt(k, lambda r, X=X: r["X"] == X) for k in ("pass26", "beat_base26", "beat_base25", "beat_fix25")}
            for X in NB_X}
    # nh252 X10 D 단면
    slice_x10 = [{k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in
                  ("D", "y26", "front", "back", "ai", "y25", "y24", "avg_e26", "pass26")}
                 for r in nb if r["release"] == "nh252" and abs(r["X"] - 0.10) < 1e-9]
    slice_d15 = [{k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in
                  ("X", "release", "y26", "back", "ai", "y25", "y24", "pass26")}
                 for r in nb if abs(r["D"] - 0.15) < 1e-9]

    # ---- 분해: 손절만 / 되사기 증분 / 둘 다, 사건 사유별
    tab26 = episodes("2026", ceps["2026"])
    tabh = episodes("hold", ceps["hold"])
    dec = {}
    for lab, bn, (lo, hi) in (("2026", "2026", P26["전체"]), ("2026뒤", "2026", P26["뒤"]), ("2025", "hold", PH["2025"]),
                              ("2024", "hold", PH["2024"])):
        k = {"2026": "전체", "2026뒤": "뒤", "2025": "2025", "2024": "2024"}[lab]
        bsr = base[bn][k]["ret"]
        stdr, cr, nk = std[bn][k]["ret"], cand[bn][k]["ret"], none_keep[bn][k]["ret"]
        tab = tab26 if bn == "2026" else tabh
        dec[lab] = {"base": bsr, "stop_only": stdr, "stop_only_vs_base": (1 + stdr) / (1 + bsr) - 1,
                    "both": cr, "both_vs_base": (1 + cr) / (1 + bsr) - 1,
                    "deepbuy_increment_vs_stop": (1 + cr) / (1 + stdr) - 1,
                    "deepbuy_no_keep": nk, "keep_increment": (1 + cr) / (1 + nk) - 1,
                    "ep_deep": ep_sum(tab, lo, hi, "deep"), "ep_sma": ep_sum(tab, lo, hi, "sma"),
                    "ep_end": ep_sum(tab, lo, hi, "end")}

    # 2026 1~2일 현금 사건 제외 시(잡음 크기)
    short = [x for x in tab26 if x["days_out"] <= 2]
    long_ = [x for x in tab26 if x["days_out"] > 2]
    noise = {"short_n": len(short), "short_rel": float(np.prod([1 + x["rel_vs_base"] for x in short]) - 1),
             "long_n": len(long_), "long_rel": float(np.prod([1 + x["rel_vs_base"] for x in long_]) - 1)}

    # ---- 2026 노출 경로 확인(종가 신호 → 다음 시가)
    b = BASES["2026"]
    c, s, dd = signals("2026")
    ec, _ = simulate(c, s, dd, 0.10, 0.15)
    audit = []
    for i in range(1, b.n):
        if cex["2026"][i] != cex["2026"][i - 1]:
            audit.append({"exec_open": b.days[i], "sig_close": b.days[i - 1], "expo": cex["2026"][i],
                          "sig_trail_or_dd": round(float(dd[i - 1]), 4), "c_vs_sma20": round(float(c[i - 1] / s[i - 1] - 1), 4)})

    payload = {"id": ID, "base": {"2026": b26, "hold": bh}, "claim": CLAIM, "recomputed": rec, "diff": diffs,
               "candidate": cand, "lag2": lag2, "no_warm_sma": nowarm, "chain_2025_peak": chain, "spread_0.1%": cost,
               "stop_only": std, "stop_only_lag2": std_lag2, "deepbuy_no_keep": none_keep,
               "fixed_at_avg_e26": {"e": e26, **fx26}, "fixed_at_own_year_e": fx_own,
               "neighbors_n": N, "neighbors_agg": agg, "neighbors_by_release": by_rel, "neighbors_by_D": by_D,
               "neighbors_by_X": by_X, "slice_X10_nh252": slice_x10, "slice_D15": slice_d15, "neighbors": nb,
               "decomposition": dec, "episodes_2026": tab26, "episodes_hold": tabh, "noise_2026": noise,
               "exposure_changes_2026": audit, "chain_episodes_2026": episodes("2026", cheps["2026"])}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    P = lambda d, k: f"{d[k]['ret']:+.1%}/{d[k]['mdd']:.1%}"
    print("\n후보", P(c26, "전체"), "앞", P(c26, "앞"), "뒤", P(c26, "뒤"), "AI", f"{c26['AI쇼크']['ret']:+.1%}",
          "7~9", f"{c26['7~9월']['ret']:+.1%}", "| 2025", P(ch, "2025"), "2024", P(ch, "2024"), "보유", P(ch, "전체"),
          "노출", round(c26["avg_e"], 3), "나감", c26["exits"])
    for lab, r in (("t+2", lag2), ("SMA예열없음", nowarm), ("2025고점연결", chain), ("스프레드0.1%", cost),
                   ("손절만", std), ("손절만 t+2", std_lag2), ("되사기 keep없음", none_keep)):
        a, h = r["2026"], r["hold"]
        print(f"{lab:<14}", P(a, "전체"), "앞", P(a, "앞"), "뒤", P(a, "뒤"), "AI", f"{a['AI쇼크']['ret']:+.1%}",
              "| 2025", P(h, "2025"), "2024", P(h, "2024"), "보유", P(h, "전체"), "노출", round(a["avg_e"], 3),
              "나감", a["exits"])
    print("고정", e26, "2026", P(fx26["2026"], "전체"), "뒤", P(fx26["2026"], "뒤"), "AI", f"{fx26['2026']['AI쇼크']['ret']:+.1%}",
          "2025", P(fx26["hold"], "2025"), "2024", P(fx26["hold"], "2024"), "보유", P(fx26["hold"], "전체"))
    print("고정(같은 해 노출)", {k: (round(v[k if k != 'hold_all' else '전체']['ret'], 4)) for k, v in fx_own.items()},
          "노출", round(ch["avg_e_2025"], 3), round(ch["avg_e_2024"], 3), round(ch["avg_e_전체"], 3))
    print("\n이웃", N, {k: f"{v[0]}/{v[1]}" for k, v in agg.items()})
    for rel, v in by_rel.items():
        print("  ", REL_KO[rel], {k: f"{a}/{b_}" for k, (a, b_) in v.items()})
    for D, v in by_D.items():
        print("   D", D, {k: f"{a}/{b_}" for k, (a, b_) in v.items()})
    for X, v in by_X.items():
        print("   X", X, {k: f"{a}/{b_}" for k, (a, b_) in v.items()})
    print("\nX10 252일 단면")
    for r in slice_x10:
        print("  ", r)
    print("\nD15 단면")
    for r in slice_d15:
        print("  ", r)
    print("\n분해")
    for k, v in dec.items():
        print(" ", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("잡음", noise)
    print("2026 사건", tab26)
    print("2026 노출 변경", audit)
    print("2025고점연결 사건", payload["chain_episodes_2026"])
    return payload


if __name__ == "__main__":
    main()
