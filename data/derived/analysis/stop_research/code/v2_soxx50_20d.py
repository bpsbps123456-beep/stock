"""v2 반박 검증: soxx50_20d (갈래 G 앞 고르기) — 2026-10-10.

후보: SOXX 종가가 SMA50 아래로 하향 돌파(전날 위, 오늘 아래)한 날 t → t+1 시가에 전량 현금 → 나간 신호일부터
20거래일째 종가에 복귀 결정 → 그다음 시가에 복귀(그동안 신호 무시).

harness.py·fam_G_hold_reentry.py를 import하지 않는다. base_*.json과 ohlc.parquet만으로 처음부터 다시 짠다.
  ret[k] = open[k+1]/open[k] - 1 (k일 시가 거래 뒤 보유 수익), 노출 e_open[k] = k보다 엄격히 앞선 SOXX 거래일 종가에 정한 값.
  구간 수익 = 구간에 속한 ret[k]의 (1 + e·ret) 곱, 낙폭 = 그 곱 경로(시작 1.0 포함)의 최대 낙폭.

상태 시작 두 가지
  fresh      : 각 기본 경로 첫날에 '보유 중'으로 새로 시작(갈래 G와 같은 관행). 첫날 노출 1.0.
  continuous : 2022-08 SMA가 찬 날부터 SOXX 달력에서 끊김 없이 이어 계산(실제 운용과 같음).

실행: cd code && PYTHONIOENCODING=utf-8 py v2_soxx50_20d.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "v2_soxx50_20d"

P26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
PH = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"), "관세": ("2025-02-18", "2025-04-04"),
      "24여름": ("2024-06-18", "2024-08-05"), "25가을": ("2025-10-13", "2025-11-21")}

CLAIM = {"2026 전체": (2.302, -0.142), "2026 앞": (1.891, -0.142), "2026 뒤": (0.142, -0.122), "AI쇼크": (-0.073, None),
         "7~9월": (0.137, None), "2025": (-0.024, -0.423), "2024": (0.172, None), "평균노출": 0.79}

SMAS = [40, 45, 50, 55, 60]
HOLDS = [10, 15, 20, 25, 30]
LOWS = [0.0, 0.5]


# ============================================================================ 자료
def load_base(name):
    b = json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
    days = list(b["days"])
    op = np.asarray(b["open"], float)
    ret = pd.Series(op[1:] / op[:-1] - 1, index=days[:-1])
    return {"days": days, "open": op, "close": np.asarray(b["close"], float), "ret": ret}


def soxx(field="Close"):
    o = pd.read_parquet(CACHE / "ohlc.parquet", columns=["date", "sym", field])
    o = o[o.sym == "SOXX"].copy()
    o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
    s = o.set_index("date")[field].astype(float).sort_index()
    return s[~s.index.duplicated()]


# ============================================================================ 상태 기계 (SOXX 달력, 종가 결정)
def run_rule(c: pd.Series, n: int, hold: int, low: float, kind: str = "cross", start: str | None = None):
    """반환: (e_close: index=SOXX 날짜, 그날 종가 뒤 정한 노출, 사건 [(나감 신호일, 복귀 신호일)]).
    kind
      cross        : 하향 돌파(전날 c≥SMA, 오늘 c<SMA)에서 나감, 정확히 hold일 뒤 복귀. 나가 있는 동안 신호 무시.
      level_reexit : 보유 중 c<SMA면 나감(돌파 불필요). hold일 뒤 일단 복귀, 다음 종가에도 아래면 다시 나감.
      level_extend : 보유 중 c<SMA면 나감. hold일이 지나고 c≥SMA인 날 복귀(아래 있는 동안 계속 머묾). hold=0이면 갈래 D.
    start: 이 날짜부터 '보유 중'으로 새로 시작(fresh). None이면 SMA가 찬 첫날부터 이어 계산(continuous)."""
    sma = c.rolling(n, min_periods=n).mean()
    below = (c < sma).to_numpy()
    ok = sma.notna().to_numpy()
    idx = c.index
    i0 = int(np.argmax(ok)) if start is None else int(np.searchsorted(idx.to_numpy(), start))
    e = np.full(len(c), np.nan)
    state, xi = "in", None
    eps = []
    for i in range(i0, len(c)):
        if not ok[i]:
            e[i] = 1.0
            continue
        if state == "in":
            if kind == "cross":
                prev_above = ok[i - 1] and not below[i - 1] if i >= 1 else False
                trig = below[i] and prev_above
            else:
                trig = below[i]
            if trig:
                state, xi = "out", i
        else:
            if kind == "level_extend":
                back = (i - xi >= hold) and not below[i]
            else:
                back = i - xi >= hold
            if back:
                state = "in"
                eps.append((idx[xi], idx[i]))
                xi = None
        e[i] = 1.0 if state == "in" else low
    if state == "out":
        eps.append((idx[xi], None))
    return pd.Series(e, index=idx).iloc[i0:], eps


def to_open(e_close: pd.Series, days, delay: int = 0, fill=1.0) -> pd.Series:
    """e_open[k] = k보다 엄격히 앞선 날짜의 최신 결정값. delay=1이면 SOXX 달력에서 하루 더 늦춘다(t+2 시가)."""
    s = e_close.dropna()
    if delay:
        s = s.shift(delay).dropna()
    pos = np.searchsorted(s.index.to_numpy(), np.asarray(days), side="left") - 1
    v = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(v, index=days).fillna(fill)


# ============================================================================ 지표
def seg(ret: pd.Series, e: pd.Series, lo, hi):
    m = (ret.index >= lo) & (ret.index <= hi)
    x = (e.reindex(ret.index).to_numpy() * ret.to_numpy())[m]
    c = np.concatenate([[1.0], np.cumprod(1 + x)])
    return float(c[-1] - 1), float((c / np.maximum.accumulate(c) - 1).min()), float(e.reindex(ret.index)[m].mean())


def evaluate(e26: pd.Series, eh: pd.Series, B26, BH):
    out = {}
    for k, (lo, hi) in P26.items():
        r, d, x = seg(B26["ret"], e26, lo, hi)
        out[f"2026_{k}"] = {"ret": r, "mdd": d, "expo": x}
    for k, (lo, hi) in PH.items():
        r, d, x = seg(BH["ret"], eh, lo, hi)
        out[k] = {"ret": r, "mdd": d, "expo": x}
    ee = e26.reindex(B26["ret"].index)
    out["avg_expo_2026"] = float(ee.mean())
    out["exits_2026"] = int(((ee.shift(1).fillna(1.0) >= 0.999) & (ee < 0.999)).sum())
    return out


def fixed(e, B26, BH):
    return evaluate(pd.Series(e, index=B26["ret"].index), pd.Series(e, index=BH["ret"].index), B26, BH)


def fixed_seg(B, lo, hi, e):
    return seg(B["ret"], pd.Series(e, index=B["ret"].index), lo, hi)


def episodes(B, e_open: pd.Series, eps, lo, hi):
    """나간 구간: 나감 체결(신호 다음 거래일 시가) → 복귀 체결. 그동안 기본 경로 시가 수익과 규칙의 상대 효과."""
    days, op, cl = B["days"], B["open"], B["close"]
    pos = {d: i for i, d in enumerate(days)}
    rows = []
    for xs, rs in eps:
        if rs is not None and rs < days[0]:      # 기본 경로 시작 전에 끝난 사건(이어 계산 모드)은 건너뛴다
            continue
        nxt = [d for d in days if d > xs]
        if not nxt or not (lo <= nxt[0] <= hi):
            continue
        a = pos[nxt[0]]
        if rs is None:
            b_ = len(days) - 1
        else:
            nr = [d for d in days if d > rs]
            b_ = pos[nr[0]] if nr else len(days) - 1
        r = B["ret"].iloc[a:b_]
        ee = e_open.reindex(r.index)
        low = float(np.min(cl[a:b_ + 1])) if b_ > a else float(cl[a])
        rows.append({"exit_sig": xs, "exit_exec": days[a], "re_exec": days[b_] if rs else "끝",
                     "days_out": b_ - a, "base_while_out": round(float(op[b_] / op[a] - 1), 4),
                     "base_to_low": round(low / op[a] - 1, 4), "low_to_reentry": round(float(op[b_]) / low - 1, 4),
                     "rule_vs_base": round(float(np.prod(1 + ee * r) / np.prod(1 + r) - 1), 4)})
    return rows


def fmt(r, d=None):
    return f"{r:+.1%}" + (f"/{d:.1%}" if d is not None else "")


# ============================================================================ 본문
def main():
    B26, BH = load_base("base_2026"), load_base("base_hold")
    c = soxx("Close")
    base = fixed(1.0, B26, BH)

    def variant(n, hold, low, kind="cross", mode="fresh", delay=0, px=None):
        cc = c if px is None else px
        e26c, ep26 = run_rule(cc, n, hold, low, kind, None if mode == "continuous" else B26["days"][0])
        ehc, eph = run_rule(cc, n, hold, low, kind, None if mode == "continuous" else BH["days"][0])
        e26 = to_open(e26c, B26["days"], delay)
        eh = to_open(ehc, BH["days"], delay)
        return evaluate(e26, eh, B26, BH), e26, eh, ep26, eph

    # ---- 1) 독립 재계산
    cand, e26, eh, ep26, eph = variant(50, 20, 0.0)
    rec = {"y26_all_ret": cand["2026_전체"]["ret"], "y26_all_mdd": cand["2026_전체"]["mdd"],
           "y26_front_ret": cand["2026_앞"]["ret"], "y26_front_mdd": cand["2026_앞"]["mdd"],
           "y26_back_ret": cand["2026_뒤"]["ret"], "y26_back_mdd": cand["2026_뒤"]["mdd"],
           "ai_shock_ret": cand["2026_AI쇼크"]["ret"], "jul_sep_ret": cand["2026_7~9월"]["ret"],
           "y25_ret": cand["2025"]["ret"], "y25_mdd": cand["2025"]["mdd"], "y24_ret": cand["2024"]["ret"],
           "y24_mdd": cand["2024"]["mdd"], "avg_exposure_2026": cand["avg_expo_2026"]}
    pairs = [("2026 전체", rec["y26_all_ret"], rec["y26_all_mdd"]), ("2026 앞", rec["y26_front_ret"], rec["y26_front_mdd"]),
             ("2026 뒤", rec["y26_back_ret"], rec["y26_back_mdd"]), ("AI쇼크", rec["ai_shock_ret"], None),
             ("7~9월", rec["jul_sep_ret"], None), ("2025", rec["y25_ret"], rec["y25_mdd"]), ("2024", rec["y24_ret"], None)]
    disc = []
    for k, r, d in pairs:
        cr, cd = CLAIM[k]
        if abs(r - cr) > 0.003:
            disc.append(f"{k} 수익 주장 {cr:+.1%} vs 재계산 {r:+.1%}")
        if cd is not None and d is not None and abs(d - cd) > 0.003:
            disc.append(f"{k} 낙폭 주장 {cd:.1%} vs 재계산 {d:.1%}")
    if abs(rec["avg_exposure_2026"] - CLAIM["평균노출"]) > 0.006:
        disc.append(f"평균 노출 주장 0.79 vs 재계산 {rec['avg_exposure_2026']:.3f}")
    ep26_rows = episodes(B26, e26, ep26, "2026-01-01", "2026-12-31")
    eph_rows = episodes(BH, eh, eph, "2023-05-22", "2025-12-31")
    print("후보 2026", fmt(rec["y26_all_ret"], rec["y26_all_mdd"]), "앞", fmt(rec["y26_front_ret"], rec["y26_front_mdd"]),
          "뒤", fmt(rec["y26_back_ret"], rec["y26_back_mdd"]), "AI", fmt(rec["ai_shock_ret"]), "7~9", fmt(rec["jul_sep_ret"]),
          "| 2025", fmt(rec["y25_ret"], rec["y25_mdd"]), "2024", fmt(rec["y24_ret"], rec["y24_mdd"]),
          "| 노출", round(rec["avg_exposure_2026"], 4), "나감", cand["exits_2026"])
    print("불일치:", disc or "없음")
    for x in ep26_rows:
        print("  2026 사건", x)

    # ---- 2) 시점 감사
    timing = {}
    # (a) G의 armed 방식과 순수 돌파의 차이가 날 수 있는 날: 복귀 결정일에 위, 다음 날 아래
    sma50 = c.rolling(50, min_periods=50).mean()
    bl = (c < sma50)
    edge = []
    for xs, rs in ep26 + eph:
        if rs is None:
            continue
        i = c.index.get_loc(rs)
        if i + 1 < len(c) and not bl.iloc[i] and bl.iloc[i + 1]:
            edge.append(rs)
    timing["armed_edge_days"] = edge
    # (b) 하루 늦춤(t+2 시가)
    d1, *_ = variant(50, 20, 0.0, delay=1)
    timing["delay_plus1"] = d1
    # (c) 상태 이어 계산(continuous)
    cont, e26c_, ehc_, ep26c, ephc = variant(50, 20, 0.0, mode="continuous")
    timing["continuous"] = cont
    timing["continuous_first_day_expo_2026"] = float(e26c_.iloc[0])
    timing["continuous_first_day_expo_hold"] = float(ehc_.iloc[0])
    # (d) Adj Close로 SMA·신호
    adj, *_ = variant(50, 20, 0.0, px=soxx("Adj Close"))
    timing["adj_close"] = adj
    # (e) 노출 바뀐 날 검사: e_open[k]가 k 이전 종가 결정과 같아야 한다(미래 정보 없음)
    e26_close, _ = run_rule(c, 50, 20, 0.0, "cross", B26["days"][0])
    look = []
    for k, d in enumerate(B26["days"][1:], start=1):
        prev = B26["days"][k - 1]
        if abs(e26.loc[d] - e26_close.loc[prev]) > 1e-12:
            look.append(d)
    timing["lookahead_mismatch_days"] = look
    print("돌파 경계 사례", edge, "| 시점 불일치", look)
    print("하루 늦춤 2026", fmt(d1["2026_전체"]["ret"], d1["2026_전체"]["mdd"]), "뒤", fmt(d1["2026_뒤"]["ret"]),
          "AI", fmt(d1["2026_AI쇼크"]["ret"]), "2025", fmt(d1["2025"]["ret"]), "2024", fmt(d1["2024"]["ret"]))
    print("이어 계산 2026", fmt(cont["2026_전체"]["ret"], cont["2026_전체"]["mdd"]), "앞", fmt(cont["2026_앞"]["ret"]),
          "뒤", fmt(cont["2026_뒤"]["ret"]), "AI", fmt(cont["2026_AI쇼크"]["ret"]), "2025", fmt(cont["2025"]["ret"]),
          "2024", fmt(cont["2024"]["ret"]), "첫날노출", timing["continuous_first_day_expo_2026"],
          timing["continuous_first_day_expo_hold"])
    print("Adj Close 2026", fmt(adj["2026_전체"]["ret"]), "뒤", fmt(adj["2026_뒤"]["ret"]), "2025", fmt(adj["2025"]["ret"]))

    # ---- 3) 이웃 격자
    b_front_w = 1 + base["2026_앞"]["ret"]
    rows = []
    for kind in ("cross", "level_reexit", "level_extend"):
        for n in SMAS:
            for hold in HOLDS:
                for low in LOWS:
                    for mode in ("fresh", "continuous"):
                        m, *_ = variant(n, hold, low, kind, mode)
                        f_ok = 1 + m["2026_앞"]["ret"] >= 0.9 * b_front_w
                        b_ok = m["2026_뒤"]["ret"] > base["2026_뒤"]["ret"] and m["2026_AI쇼크"]["ret"] > base["2026_AI쇼크"]["ret"]
                        fx = {}
                        for y in ("2025", "2024"):
                            ex = m[y]["expo"]
                            fx[y] = fixed_seg(BH, *PH[y], ex)[0]
                        fx26 = fixed(m["avg_expo_2026"], B26, BH)
                        rows.append({"kind": kind, "sma": n, "hold": hold, "low": low, "mode": mode,
                                     "y26": m["2026_전체"]["ret"], "y26_mdd": m["2026_전체"]["mdd"],
                                     "front": m["2026_앞"]["ret"], "back": m["2026_뒤"]["ret"],
                                     "back_mdd": m["2026_뒤"]["mdd"], "ai": m["2026_AI쇼크"]["ret"],
                                     "y25": m["2025"]["ret"], "y24": m["2024"]["ret"], "expo26": m["avg_expo_2026"],
                                     "expo25": m["2025"]["expo"], "expo24": m["2024"]["expo"],
                                     "fixed25_same_expo": fx["2025"], "fixed24_same_expo": fx["2024"],
                                     "fixed26_same_expo": fx26["2026_전체"]["ret"],
                                     "fixed25_at_expo26": fx26["2025"]["ret"], "fixed24_at_expo26": fx26["2024"]["ret"],
                                     "exits26": m["exits_2026"], "pass_front": bool(f_ok), "pass_back": bool(b_ok),
                                     "pass_2026": bool(f_ok and b_ok),
                                     "beat_fixed_2025": bool(m["2025"]["ret"] > fx["2025"]),
                                     "beat_fixed_2024": bool(m["2024"]["ret"] > fx["2024"]),
                                     "beat_fixed_2026": bool(m["2026_전체"]["ret"] > fx26["2026_전체"]["ret"])})
    df = pd.DataFrame(rows)

    def tally(sub):
        return {"n": int(len(sub)), "pass_front": int(sub.pass_front.sum()), "pass_back": int(sub.pass_back.sum()),
                "pass_2026": int(sub.pass_2026.sum()), "beat_fixed_2025": int(sub.beat_fixed_2025.sum()),
                "beat_fixed_2024": int(sub.beat_fixed_2024.sum()), "beat_fixed_2026": int(sub.beat_fixed_2026.sum()),
                "beat_base_2025": int((sub.y25 > base["2025"]["ret"]).sum()),
                "y26_med": float(sub.y26.median()), "back_med": float(sub.back.median()), "ai_med": float(sub.ai.median()),
                "y25_med": float(sub.y25.median()), "y24_med": float(sub.y24.median())}

    tallies = {}
    for kind in ("cross", "level_reexit", "level_extend"):
        for mode in ("fresh", "continuous"):
            tallies[f"{kind}_{mode}"] = tally(df[(df.kind == kind) & (df["mode"] == mode)])
            for low in LOWS:
                tallies[f"{kind}_{mode}_low{low}"] = tally(df[(df.kind == kind) & (df["mode"] == mode) & (df.low == low)])
    for k, v in tallies.items():
        if k.count("_") <= 2 or "low" in k:
            print(k, v)

    # SMA×hold 표(cross, fresh, low 0): 2026 뒤 / AI쇼크 / 2025
    piv = {}
    for col in ("y26", "front", "back", "ai", "y25", "y24", "expo26"):
        p = df[(df.kind == "cross") & (df["mode"] == "fresh") & (df.low == 0.0)].pivot(index="sma", columns="hold", values=col)
        piv[col] = {str(i): {str(j): round(float(p.loc[i, j]), 4) for j in p.columns} for i in p.index}
        print(col)
        print(p.round(3).to_string())

    # ---- 4) 고정 노출 대비 (후보 2026 평균 노출 그대로 다시 잼)
    ex = cand["avg_expo_2026"]
    fx = fixed(ex, B26, BH)
    vs = {}
    for k in ("2026_전체", "2026_앞", "2026_뒤", "2026_AI쇼크", "2026_7~9월", "2025", "2024", "관세", "24여름", "25가을"):
        vs[k] = {"cand_ret": cand[k]["ret"], "fixed_ret": fx[k]["ret"], "d_ret": cand[k]["ret"] - fx[k]["ret"],
                 "cand_mdd": cand[k]["mdd"], "fixed_mdd": fx[k]["mdd"], "d_mdd": cand[k]["mdd"] - fx[k]["mdd"]}
    # 연도별 같은 노출(그 해 평균 노출)
    vs_year = {}
    for y in ("2025", "2024"):
        e_y = cand[y]["expo"]
        r_, d_, _ = fixed_seg(BH, *PH[y], e_y)
        vs_year[y] = {"expo": e_y, "fixed_ret": r_, "fixed_mdd": d_, "cand_ret": cand[y]["ret"], "cand_mdd": cand[y]["mdd"]}
    print("고정 노출", round(ex, 4))
    for k, v in vs.items():
        print(f"  {k}: 후보 {fmt(v['cand_ret'], v['cand_mdd'])} 고정 {fmt(v['fixed_ret'], v['fixed_mdd'])} "
              f"차 {v['d_ret']:+.1%} / 낙폭차 {v['d_mdd']:+.1%}")
    print("  같은 해 평균 노출", {y: (round(v['expo'], 3), fmt(v['fixed_ret'], v['fixed_mdd'])) for y, v in vs_year.items()})

    # ---- 갈래 D(아래 있는 동안 현금, 위로 회복하면 복귀)와 사건 비교
    dlike, e26d, ehd, ep26d, ephd = variant(50, 0, 0.0, "level_extend", "continuous")
    d_rows = episodes(B26, e26d, ep26d, "2026-01-01", "2026-12-31")
    print("D형 SOXX<SMA50→0 2026", fmt(dlike["2026_전체"]["ret"], dlike["2026_전체"]["mdd"]), "뒤", fmt(dlike["2026_뒤"]["ret"]),
          "AI", fmt(dlike["2026_AI쇼크"]["ret"]), "나감", dlike["exits_2026"])
    for x in d_rows:
        print("  D 사건", x)
    # 갈래 D 결과 파일 숫자 확인
    d_file = None
    try:
        dj = json.loads((RESULTS / "D_trend_breadth.json").read_text(encoding="utf-8"))
        for r in dj["rows"]:
            if r["name"] == "A SOXX<SMA50 →0.0":
                d_file = {k: r[k] for k in ("2026_전체_수익", "2026_뒤_수익", "2026_AI쇼크_수익", "2026_평균노출", "2026_빠짐횟수")}
    except Exception as exc:  # noqa: BLE001
        d_file = {"error": str(exc)}
    print("D 결과 파일", d_file)

    # 사건 구간별 비교: 같은 기간(나감 체결→복귀 체결)마다 후보·D·기본의 수익
    def window(e, lo, hi):
        return seg(B26["ret"], e, lo, hi)[0]
    cmp_rows = []
    marks = [("2026-01-02", "2026-03-03"), ("2026-03-04", "2026-03-31"), ("2026-04-01", "2026-05-29"),
             ("2026-06-01", "2026-07-13"), ("2026-07-14", "2026-08-10"), ("2026-08-11", "2026-10-09")]
    for lo, hi in marks:
        cmp_rows.append({"lo": lo, "hi": hi, "base": window(pd.Series(1.0, index=B26["ret"].index), lo, hi),
                         "cand": window(e26, lo, hi), "D": window(e26d, lo, hi),
                         "D_days_out": int((e26d.reindex(B26["ret"].index)[(B26["ret"].index >= lo) & (B26["ret"].index <= hi)] < 0.5).sum()),
                         "cand_days_out": int((e26.reindex(B26["ret"].index)[(B26["ret"].index >= lo) & (B26["ret"].index <= hi)] < 0.5).sum())})
    for x in cmp_rows:
        print("  구간", x)

    # 보유 기간 사건(2025·2024)
    for x in eph_rows:
        if x["exit_exec"] >= "2024-01-01":
            print("  보유 사건", x)

    # 이웃 통과 목록·SMA별 2026 나감 날짜·복귀 결정일 SOXX 위치
    sub = df[(df.kind == "cross") & (df["mode"] == "fresh")]
    passing = sub[sub.pass_2026][["sma", "hold", "low", "y26", "front", "back", "ai", "y25", "y24", "expo26",
                                  "fixed25_same_expo", "fixed24_same_expo"]].round(4).to_dict("records")
    exits_by_sma = {n: run_rule(c, n, 20, 0.0, "cross", B26["days"][0])[1] for n in SMAS}
    gap = {d: round(float(c[d] / sma50[d] - 1), 4) for d in ("2026-03-03", "2026-03-31", "2026-04-06", "2026-07-13",
                                                             "2026-07-14", "2026-07-15", "2026-08-10", "2026-09-18")}
    print("통과 이웃", passing)
    print("SMA별 나감(보유 20일)", exits_by_sma)
    print("SOXX/SMA50-1", gap)
    extra = {"passing_neighbors_cross_fresh": passing, "exits_2026_by_sma_hold20": exits_by_sma, "soxx_vs_sma50": gap,
             "beat_fixed_at_expo26": {"2025": int((sub.y25 > sub.fixed25_at_expo26).sum()),
                                      "2024": int((sub.y24 > sub.fixed24_at_expo26).sum())}}

    payload = {"id": ID, "extra": extra, "recomputed": rec, "claim": CLAIM, "discrepancies": disc, "base": base,
               "episodes_2026": ep26_rows, "episodes_hold": eph_rows,
               "timing": {k: v for k, v in timing.items()}, "neighbors": rows, "tallies": tallies, "pivots": piv,
               "fixed_same_expo": {"expo": ex, "vs": vs, "by_year_expo": vs_year},
               "D_like": {"metrics": dlike, "episodes_2026": d_rows, "D_file": d_file, "windows": cmp_rows}}
    (RESULTS / f"{ID}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("저장", RESULTS / f"{ID}.json")


if __name__ == "__main__":
    main()
