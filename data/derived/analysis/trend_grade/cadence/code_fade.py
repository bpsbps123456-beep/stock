"""과제 E3 — S가 끝나는 모양: 급락형 대 서서히 빠지는 형 (2026-10-12).

사용자 질문(2026-10-12): 큰 펀드는 일·주·월·분기로 리밸런싱한다. 3월까지 S였던 회사가 4월부터 아닌 경우,
꼭 급락이 아니라 서서히 빠지기도 한다. 등급을 나누는 최소 시간 단위를 생각해 보자.

등급: 연구 최종식(FINAL, ../code/final.py)과 같은 식을 공통 채점기(../code/harness.py)로 만든다.
    점수 = 미국 보유 종목 안 백분위 평균{C/SMA200−1, C/SMA50−1, C/252일 최고 고가−1, Clenow126}의 10일 평균
    등급 = 그 점수의 그날 백분위 → 상위 10% S / 20% A / 40% B / 20% C / 10% D, 관문(S·A는 C>SMA50>SMA200,
    C<SMA200 그리고 SMA50<SMA200이면 D). 순위 모집단 = H.universe_mask(그날 보유 + 봉 130개, t일까지 아는 값).
    final.py는 채점용 eligible()(다음 날 시가 있음 포함)로 순위를 냈다. 마지막 날을 빼면 같다(아래 parity로 확인).
    당일 등급 = 10일 평균하지 않은 원점수로 같은 사다리·관문(데스크 '당일 X'와 같은 식).
    모집단: 2026-01-02 전에는 2026 보유 이력 150종목 전체, 그 뒤에는 그날 최신 PDF 보유 종목(harness 관례).

사전 정의(측정 전에 고정, 과제 문구 그대로 옮김)
(1) S 구간 = 평활 등급 S가 연속인 거래일 묶음. 첫날 s, 마지막 날 L, 끝난 날 X = L 다음 거래일(등급이 S 아님).
    끝난 원인: rank = X의 평활 백분위 ≤ 0.90 / gate = 백분위는 상위 10%인데 관문(C>SMA50>SMA200)이 깨짐 /
    left = X에 보유 종목에서 빠짐(또는 시세 없음) / ongoing = 자료 끝(2026-10-09)까지 S.
    급락형: [L−10, L+10] 거래일 중 어느 날이든 종가가 그날까지 20거래일 최고 종가 대비 −15% 이하. (먼저 검사)
    서서히형: 급락형 아님, [L−19, L] 20거래일 안 종가의 고점(구간 안 누적 최고) 대비 낙폭이 끝까지 −10%보다 얕음,
             그리고 끝난 원인이 rank(점수 순위가 내려가서 S를 잃음).
    중간형: 나머지(left·ongoing은 분류하지 않고 따로 센다).
    이후 성과: X 종가 신호 → X+1 시가 매수 → X+1+h 시가(h = 20·60). 초과 = 같은 날 eligible 종목 평균 대비.
    최대 낙폭 = X+1 시가 대비 [X+1, X+h] 최저 저가(h일 모두 있어야). 다시 S = [X+1, X+k]에 평활 S가 있는지(k=10·20·60).
(2) 먼저 줄이기 신호(S인 날 d 종가에 알 수 있는 값만): 점수 기울기(평활 점수 20일 OLS 기울기), 50일선 거리 감소
    (C/SMA50−1의 지난 20일 최고 대비 변화), 52주 고점 뒤 경과일(252일 최고 고가가 찍힌 뒤 거래일 수), 당일 등급.
    연속 신호 3개 × 문턱 3단(1~5월 S인 날 분포의 나쁜 쪽 10·20·30%) + 당일 등급 3개(S 아님 / A 아래 / 2일 연속 S 아님)
    = 12개. 구간에 들어간 뒤(s+1일 이후) 처음 켜진 날 f에 f+1 시가로 팔고, 기본(S를 잃으면 팖: X+1 시가)과 비교.
    고르기: 1~5월(끝난 날 X 기준) 보유 종목 평균 대비 거래 수익 개선 평균이 가장 큰 1개 → 6~10월에서 확인.
(3) 매도 규칙(진입은 모두 같다: S 구간 첫날 종가 → 다음 날 시가 매수, 이미 들고 있으면 그대로; 같은 비중):
    R1 S를 잃으면 팖 / R2 A 아래(B 이하)면 팖 / R3 10거래일 연속 S 아님이면 팖 / R4 주 마지막 거래일 등급이 S 아니면 팖 /
    R5 당일(평활 없는) 등급이 C 이하면 팖 / R6(격자 밖 1개, 과제에 없던 월간) 월 마지막 거래일 등급이 S 아니면 팖.
    모두 신호 다음 거래일 시가 체결. 보유 종목에서 빠지면 다음 시가에 강제 매도. 휘둘림 = 판 신호일 뒤 10거래일 안에
    평활 등급이 다시 S가 된 비율. 포트폴리오는 매일 같은 비중(들고 있는 종목 수로 나눔), 빈 날은 현금 0%.
DRUG는 2024년 구간에서 뺀다(harness EXCLUDE, 2024-10-15 하루 +1446%).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import harness as H  # noqa: E402

START = "2024-07-01"
PERIODS = {
    "y2024h2": ("2024-07-01", "2024-12-31"),
    "y2025": ("2025-01-02", "2025-12-31"),
    "front": ("2026-01-02", "2026-05-29"),
    "back": ("2026-06-01", "2026-12-31"),
}
PERIOD_KO = {"y2024h2": "2024 하반기(보조)", "y2025": "2025(보조)", "front": "2026 1~5월(고름)",
             "back": "2026 6~10월(확인)", "all": "전체 2024-07~2026-10"}
DRUG_CUT = pd.Timestamp("2025-01-01")
CRASH_DD, CRASH_HALF = -0.15, 10
SLOW_DD, SLOW_WIN = -0.10, 20
ROLL_PEAK = 20
QUANTS = (0.10, 0.20, 0.30)
WHIP = 10


def r_(x, k=2):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, k) if np.isfinite(x) else None


def pct(x, k=2):
    return r_(None if x is None else float(x) * 100, k)


# ------------------------------------------------------------------ 등급과 신호
def build():
    P = H.load_panel()
    C, O, Hi, Lo = P["C"], P["O"], P["H"], P["L"]
    um = H.universe_mask(P)
    sc_s = FC.make(FC.GRID["E14"], 10)(P)
    sc_r = FC.make(FC.GRID["E14"], 0)(P)
    G = FL.hyb_grades(sc_s, P, um)
    GR = FL.hyb_grades(sc_r, P, um)
    PR = H.pct_rank(sc_s, um & sc_s.notna())
    E = H.eligible().copy()
    E.loc[E.index < DRUG_CUT, "DRUG"] = False
    return P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E


def slope20(x: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    a = x.to_numpy(float)
    out = np.full(a.shape, np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n, axis=0)  # (T-n+1, N, n)
        t = np.arange(n, dtype=float) - (n - 1) / 2.0
        out[n - 1:] = (w * t).sum(axis=2) / float(t @ t)  # NaN이 하나라도 있으면 NaN
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def high_age(Hi: pd.DataFrame, n: int = 252, minp: int = 120) -> pd.DataFrame:
    """252일 최고 고가(FC와 같은 창·min_periods)가 찍힌 뒤 경과 거래일(같은 값이면 마지막 날 기준)."""
    a = Hi.to_numpy(float)
    T, N = a.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        col = a[:, j]
        for t in range(T):
            if not np.isfinite(col[t]):
                continue
            w = col[max(0, t - n + 1): t + 1]
            ok = np.isfinite(w)
            if ok.sum() < minp:
                continue
            m = np.nanmax(w)
            last = np.where(w == m)[0][-1]
            out[t, j] = len(w) - 1 - last
    return pd.DataFrame(out, index=Hi.index, columns=Hi.columns)


def signals(C, Hi, sc_s):
    m = C / FC.sma(C, 50) - 1.0
    return {
        "slope": slope20(sc_s),                     # 낮을수록 나쁨
        "ma50drop": m - m.rolling(20, min_periods=20).max(),  # 낮을수록 나쁨(0 = 20일 최고 거리 유지)
        "hiage": high_age(Hi),                      # 높을수록 나쁨
    }


def period_of(ts: pd.Timestamp) -> str | None:
    for k, (a, b) in PERIODS.items():
        if pd.Timestamp(a) <= ts <= pd.Timestamp(b):
            return k
    return None


# ------------------------------------------------------------------ (1) S 구간
def spells(ctx) -> list[dict]:
    P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E = ctx
    cal = C.index
    T = len(cal)
    Ga, PRa, uma = G.to_numpy(object), PR.to_numpy(float), um.to_numpy(bool)
    Ca, Oa, La = C.to_numpy(float), O.to_numpy(float), Lo.to_numpy(float)
    isS = Ga == "S"
    roll_peak = C.rolling(ROLL_PEAK, min_periods=1).max().to_numpy(float)
    dd20 = Ca / roll_peak - 1.0
    first_graded = np.array([np.argmax(pd.notna(Ga[:, j])) if pd.notna(Ga[:, j]).any() else -1
                             for j in range(Ga.shape[1])])
    # 이후 성과 표
    Ea = E.to_numpy(bool)
    fw, umean, mdd, umdd = {}, {}, {}, {}
    for h in (20, 60):
        f = (O.shift(-(1 + h)) / O.shift(-1) - 1.0)
        fw[h] = f.to_numpy(float)
        umean[h] = f.where(E).mean(axis=1).to_numpy(float)
        lowmin = Lo[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
        md = (lowmin / O.shift(-1) - 1.0)
        mdd[h] = md.to_numpy(float)
        umdd[h] = md.where(E).mean(axis=1).to_numpy(float)
    start = cal.searchsorted(pd.Timestamp(START))
    out = []
    for j, sym in enumerate(C.columns):
        col = isS[:, j].astype(int)
        d = np.diff(np.r_[0, col, 0])
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
        for s, L in zip(starts, ends):
            X = L + 1
            if X < start and X < T:
                continue  # 2024-07 전에 끝난 구간
            if sym == "DRUG" and cal[min(X, T - 1)] < DRUG_CUT:
                continue
            rec = {"sym": sym, "s": int(s), "L": int(L), "X": int(X) if X < T else None,
                   "start": cal[s].strftime("%Y-%m-%d"), "last": cal[L].strftime("%Y-%m-%d"),
                   "len": int(L - s + 1), "left_censored": bool(s == first_graded[j])}
            if X >= T:
                rec["cause"] = "ongoing"
            elif pd.isna(Ga[X, j]):
                rec["cause"] = "left" if not uma[X, j] else "nograde"
            elif PRa[X, j] > 0.90:
                rec["cause"] = "gate"
            else:
                rec["cause"] = "rank"
            rec["end"] = cal[X].strftime("%Y-%m-%d") if X < T else None
            rec["period"] = period_of(cal[X]) if X < T else None
            # 모양 분류
            a, b = max(0, L - CRASH_HALF), min(T - 1, L + CRASH_HALF)
            w = dd20[a:b + 1, j]
            hit = np.where(w <= CRASH_DD)[0]
            rec["crash"] = bool(len(hit))
            rec["crash_win_partial"] = bool(L + CRASH_HALF > T - 1)
            rec["crash_when"] = (None if not len(hit) else ("by_L" if a + hit[0] <= L else "after_L"))
            rec["crash_day_vs_L"] = None if not len(hit) else int(a + hit[0] - L)
            c = Ca[max(0, L - SLOW_WIN + 1): L + 1, j]
            c = c[np.isfinite(c)]
            dds = float((c / np.maximum.accumulate(c) - 1.0).min()) if len(c) else np.nan
            rec["dd_last20"] = dds
            rec["min_dd20_window"] = float(np.nanmin(w)) if np.isfinite(w).any() else None
            if rec["cause"] in ("rank", "gate"):
                if rec["crash"]:
                    rec["shape"] = "crash"
                elif dds > SLOW_DD and rec["cause"] == "rank":
                    rec["shape"] = "slow"
                else:
                    rec["shape"] = "mid"
            else:
                rec["shape"] = None
            # 구간 안 성과·되돌림
            seg = Ca[s:L + 1, j]
            pk = int(np.nanargmax(seg))
            rec["peak_to_L_days"] = int(L - (s + pk))
            prseg = PRa[s:L + 1, j]
            rec["prpeak_to_L_days"] = int(L - (s + int(np.nanargmax(prseg))))
            rec["pr_X"] = r_(PRa[X, j], 4) if X < T else None
            if X < T - 1 and s + 1 < T and np.isfinite(Oa[s + 1, j]) and np.isfinite(Oa[X + 1, j]):
                rec["spell_ret"] = float(Oa[X + 1, j] / Oa[s + 1, j] - 1)
                rec["giveback"] = float(Oa[X + 1, j] / np.nanmax(seg) - 1)
            else:
                rec["spell_ret"] = rec["giveback"] = None
            if X < T:
                for h in (20, 60):
                    v = fw[h][X, j]
                    rec[f"f{h}"] = float(v) if np.isfinite(v) else None
                    rec[f"f{h}_ex"] = float(v - umean[h][X]) if np.isfinite(v) and np.isfinite(umean[h][X]) else None
                    m_ = mdd[h][X, j]
                    rec[f"mdd{h}"] = float(m_) if np.isfinite(m_) else None
                    rec[f"umdd{h}"] = float(umdd[h][X]) if np.isfinite(m_) else None
                for k in (10, 20, 60):
                    seg2 = isS[X + 1: X + 1 + k, j]
                    if seg2.any():
                        rec[f"backS{k}"] = True
                    elif X + k <= T - 1:
                        rec[f"backS{k}"] = False
                    else:
                        rec[f"backS{k}"] = None
            out.append(rec)
    return out


def _stats(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": pct(v.mean()), "median": pct(np.median(v)),
            "q25": pct(np.quantile(v, 0.25)), "q75": pct(np.quantile(v, 0.75)), "pos": r_((v > 0).mean() * 100, 1)}


def _rate(vals):
    v = [x for x in vals if x is not None]
    return {"n": len(v), "pct": r_(100 * np.mean(v), 1) if v else None}


def spell_tables(sp: list[dict]) -> dict:
    out = {}
    for per in list(PERIODS) + ["all"]:
        rows = [r for r in sp if r.get("X") is not None and (per == "all" or r["period"] == per)]
        done = [r for r in rows if r["shape"]]
        res = {"spells_ended": len(rows),
               "cause": {c: sum(r["cause"] == c for r in rows) for c in ("rank", "gate", "left", "nograde")},
               "classified": len(done)}
        for shp in ("crash", "slow", "mid", "ALL"):
            rr = done if shp == "ALL" else [r for r in done if r["shape"] == shp]
            lens = np.array([r["len"] for r in rr], float)
            d = {"n": len(rr), "share": r_(100 * len(rr) / max(len(done), 1), 1)}
            if len(rr):
                d["len_mean"] = r_(lens.mean(), 1)
                d["len_median"] = r_(np.median(lens), 1)
                d["len_q25_q75"] = [r_(np.quantile(lens, .25), 1), r_(np.quantile(lens, .75), 1)]
                d["len_1day_pct"] = r_(100 * np.mean(lens == 1), 1)
                d["f20_ex"] = _stats([r.get("f20_ex") for r in rr])
                d["f60_ex"] = _stats([r.get("f60_ex") for r in rr])
                d["f20_raw"] = _stats([r.get("f20") for r in rr])
                d["mdd20"] = _stats([r.get("mdd20") for r in rr])
                d["mdd20_universe"] = _stats([r.get("umdd20") for r in rr])
                d["mdd60"] = _stats([r.get("mdd60") for r in rr])
                d["mdd60_universe"] = _stats([r.get("umdd60") for r in rr])
                for k in (10, 20, 60):
                    d[f"backS{k}"] = _rate([r.get(f"backS{k}") for r in rr])
                d["spell_ret"] = _stats([r.get("spell_ret") for r in rr])
                d["giveback_from_peak"] = _stats([r.get("giveback") for r in rr])
                d["peak_to_L_days_median"] = r_(np.median([r["peak_to_L_days"] for r in rr]), 1)
                d["prpeak_to_L_days_median"] = r_(np.median([r["prpeak_to_L_days"] for r in rr]), 1)
                if shp == "crash":
                    d["crash_by_L"] = sum(r["crash_when"] == "by_L" for r in rr)
                    d["crash_after_L"] = sum(r["crash_when"] == "after_L" for r in rr)
                    d["crash_from_gate_exit"] = sum(r["cause"] == "gate" for r in rr)
                    # 하위 분할: L까지 −15%가 난 것(분류에 X 뒤 정보 없음) / L 뒤에 난 것(분류가 X 뒤 10일을 봄)
                    for when in ("by_L", "after_L"):
                        sub = [r for r in rr if r["crash_when"] == when]
                        d[f"sub_{when}"] = {"n": len(sub), "f20_ex": _stats([r.get("f20_ex") for r in sub]),
                                            "f60_ex": _stats([r.get("f60_ex") for r in sub]),
                                            "mdd20": _stats([r.get("mdd20") for r in sub]),
                                            "mdd20_universe": _stats([r.get("umdd20") for r in sub]),
                                            "backS20": _rate([r.get("backS20") for r in sub]),
                                            "len_median": r_(np.median([r["len"] for r in sub]), 1) if sub else None}
                if shp == "mid":
                    d["mid_from_gate"] = sum(r["cause"] == "gate" for r in rr)
                    d["mid_dd_10to15"] = sum(r["cause"] == "rank" for r in rr)
                # 길이 5일 이상만(짧은 깜빡임 제외)
                long_ = [r for r in rr if r["len"] >= 5]
                d["len5plus"] = {"n": len(long_), "f20_ex": _stats([r.get("f20_ex") for r in long_]),
                                 "backS20": _rate([r.get("backS20") for r in long_])}
            res[shp] = d
        out[per] = res
    ongoing = [r for r in sp if r["cause"] == "ongoing"]
    out["ongoing_at_end"] = [{"sym": r["sym"], "start": r["start"], "len": r["len"]} for r in ongoing]
    out["left_censored_spells"] = sum(r["left_censored"] for r in sp)
    return out


# ------------------------------------------------------------------ (2) 먼저 줄이기
def early_cut(ctx, sp, sig) -> dict:
    P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E = ctx
    cal = C.index
    T = len(cal)
    Ga, GRa = G.to_numpy(object), GR.to_numpy(object)
    Oa, Ea = O.to_numpy(float), E.to_numpy(bool)
    isS = Ga == "S"
    fa, fb = pd.Timestamp(PERIODS["front"][0]), pd.Timestamp(PERIODS["front"][1])
    fmask = ((cal >= fa) & (cal <= fb))[:, None] & isS
    thr, fires = {}, {}
    for name, df in sig.items():
        a = df.to_numpy(float)
        v = a[fmask & np.isfinite(a)]
        for q in QUANTS:
            if name == "hiage":
                t = float(np.quantile(v, 1 - q))
                fires[f"{name}_q{int(q * 100)}"] = a >= t
            else:
                t = float(np.quantile(v, q))
                fires[f"{name}_q{int(q * 100)}"] = a <= t
            thr[f"{name}_q{int(q * 100)}"] = r_(t, 5)
    rawS = GRa == "S"
    rawSA = (GRa == "S") | (GRa == "A")
    graded = pd.notna(GRa)
    fires["raw_notS"] = graded & ~rawS
    fires["raw_belowA"] = graded & ~rawSA
    prev = np.vstack([np.zeros((1, rawS.shape[1]), bool), (graded & ~rawS)[:-1]])
    fires["raw_notS_2d"] = graded & ~rawS & prev
    thr.update({"raw_notS": "당일 등급 ≠ S", "raw_belowA": "당일 등급 B 이하", "raw_notS_2d": "당일 등급 2일 연속 ≠ S"})

    # 기본 거래와 같은 기간 보유 종목 평균(시가→시가)
    def u_ret(i_sig, i_a, i_b):
        sel = Ea[i_sig]
        ra = Oa[i_b, sel] / Oa[i_a, sel] - 1.0
        return float(np.nanmean(ra)) if np.isfinite(ra).any() else np.nan

    base = []
    for r in sp:
        X = r.get("X")
        if X is None or X + 1 >= T:
            continue
        j = C.columns.get_loc(r["sym"])
        s, L = r["s"], r["L"]
        if not (np.isfinite(Oa[s + 1, j]) and np.isfinite(Oa[X + 1, j])):
            continue
        b_ret = Oa[X + 1, j] / Oa[s + 1, j] - 1
        b_ex = b_ret - u_ret(s, s + 1, X + 1)
        base.append((r, j, s, L, X, b_ret, b_ex))
    res = {"thresholds_from_front_S_days": thr, "n_front_S_days": int(fmask.sum()), "variants": {}}
    per_spell = {}
    for name, F in fires.items():
        rows = []
        for (r, j, s, L, X, b_ret, b_ex) in base:
            f = None
            if L >= s + 1:
                hit = np.where(F[s + 1:L + 1, j])[0]
                if len(hit):
                    f = s + 1 + int(hit[0])
            if f is None or not np.isfinite(Oa[f + 1, j]):
                rows.append({"period": r["period"], "shape": r["shape"], "fired": False, "gain": 0.0, "gain_ex": 0.0,
                             "lead": 0, "len": r["len"]})
                continue
            e_ret = Oa[f + 1, j] / Oa[s + 1, j] - 1
            e_ex = e_ret - u_ret(s, s + 1, f + 1)
            rows.append({"period": r["period"], "shape": r["shape"], "fired": True, "gain": e_ret - b_ret,
                         "gain_ex": e_ex - b_ex, "lead": int(X - f), "len": r["len"],
                         "after_drop": float(Oa[X + 1, j] / Oa[f + 1, j] - 1)})
        per_spell[name] = rows
        v = {}
        for per in list(PERIODS) + ["all"]:
            rr = [x for x in rows if per == "all" or x["period"] == per]
            fr = [x for x in rr if x["fired"]]
            g = np.array([x["gain"] for x in rr], float)
            gx = np.array([x["gain_ex"] for x in rr], float)
            gf = np.array([x["gain"] for x in fr], float)
            d = {"spells": len(rr), "fired": len(fr), "fired_pct": r_(100 * len(fr) / max(len(rr), 1), 1),
                 "net_gain_per_spell": pct(g.mean()) if len(g) else None,
                 "net_gain_ex_per_spell": pct(gx.mean()) if len(gx) else None,
                 "lead_days_median": r_(np.median([x["lead"] for x in fr]), 1) if fr else None,
                 "fired_better_pct": r_(100 * np.mean(gf > 0), 1) if len(gf) else None,
                 "avoided_loss_sum": pct(gf[gf > 0].sum()) if len(gf) else None,
                 "missed_gain_sum": pct(gf[gf < 0].sum()) if len(gf) else None,
                 "avoided_loss_mean_when_better": pct(gf[gf > 0].mean()) if (gf > 0).any() else None,
                 "missed_gain_mean_when_worse": pct(gf[gf < 0].mean()) if (gf < 0).any() else None}
            for shp in ("slow", "crash", "mid"):
                sr = [x for x in rr if x["shape"] == shp]
                sf = [x for x in sr if x["fired"]]
                d[f"{shp}_caught_pct"] = r_(100 * len(sf) / len(sr), 1) if sr else None
                d[f"{shp}_lead_median"] = r_(np.median([x["lead"] for x in sf]), 1) if sf else None
                d[f"{shp}_gain_mean"] = pct(np.mean([x["gain"] for x in sr])) if sr else None
            v[per] = d
        res["variants"][name] = v
    # 고르기: front 순이익(보유 종목 평균 대비) 최대
    names = list(fires)
    fr_score = {n: res["variants"][n]["front"]["net_gain_ex_per_spell"] for n in names}
    bk_score = {n: res["variants"][n]["back"]["net_gain_ex_per_spell"] for n in names}
    pick = max(names, key=lambda n: fr_score[n] if fr_score[n] is not None else -1e9)
    fs = pd.Series(fr_score, dtype=float)
    bs = pd.Series(bk_score, dtype=float)
    res["selection"] = {
        "rule": "1~5월(끝난 날 기준) 보유 종목 평균 대비 거래 수익 개선(구간당 평균) 최대",
        "pick": pick, "front": fr_score[pick], "back": bk_score[pick],
        "y2025": res["variants"][pick]["y2025"]["net_gain_ex_per_spell"],
        "y2024h2": res["variants"][pick]["y2024h2"]["net_gain_ex_per_spell"],
        "front_positive": int((fs > 0).sum()), "back_positive": int((bs > 0).sum()), "n_variants": len(names),
        "front_back_rank_corr": r_(fs.rank().corr(bs.rank()), 3),
        "back_rank_of_pick": int(bs.rank(ascending=False)[pick]),
    }
    return res, fires


def auc_table(ctx, sp, sig, fires) -> dict:
    """진단: 서서히형 끝나기 0~9일 전 S인 날(양성) 대 S가 20일 이상 더 이어진 S인 날(음성)."""
    P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E = ctx
    cal = C.index
    T = len(cal)
    pos_idx, neg_idx = [], []
    for r in sp:
        j = C.columns.get_loc(r["sym"])
        s, L = r["s"], r["L"]
        end_known = r["X"] is not None
        for d in range(s, L + 1):
            if cal[d] < pd.Timestamp(START):
                continue
            left = L - d
            if r["shape"] == "slow" and 0 <= left <= 9:
                pos_idx.append((d, j))
            elif left >= 20 or (not end_known and (T - 1 - d) >= 20):
                neg_idx.append((d, j))
    out = {}
    for per in list(PERIODS) + ["all"]:
        def keep(lst):
            return [(d, j) for d, j in lst if per == "all" or period_of(cal[d]) == per]
        pi, ni = keep(pos_idx), keep(neg_idx)
        d = {"pos_days": len(pi), "neg_days": len(ni)}
        for name, df in sig.items():
            a = df.to_numpy(float)
            sgn = 1.0 if name == "hiage" else -1.0  # 클수록 나쁨으로 맞춤
            pv = np.array([sgn * a[x] for x in pi], float)
            nv = np.array([sgn * a[x] for x in ni], float)
            pv, nv = pv[np.isfinite(pv)], nv[np.isfinite(nv)]
            if len(pv) and len(nv):
                allv = np.r_[pv, nv]
                rk = pd.Series(allv).rank().to_numpy()
                auc = (rk[:len(pv)].sum() - len(pv) * (len(pv) + 1) / 2) / (len(pv) * len(nv))
                d[f"auc_{name}"] = r_(auc, 3)
                d[f"median_pos_{name}"] = r_(sgn * np.median(pv), 4)
                d[f"median_neg_{name}"] = r_(sgn * np.median(nv), 4)
        for name in ("raw_notS", "raw_belowA", "raw_notS_2d"):
            F = fires[name]
            d[f"hit_{name}"] = r_(100 * np.mean([F[x] for x in pi]), 1) if pi else None
            d[f"false_{name}"] = r_(100 * np.mean([F[x] for x in ni]), 1) if ni else None
        out[per] = d
    return out


# ------------------------------------------------------------------ (3) 매도 규칙
RULES = {
    "R1_S_lost": "평활 등급이 S 아니면 다음 날 시가에 팖",
    "R2_below_A": "평활 등급이 B 이하면 팖(A까지는 보유)",
    "R3_nonS_10d": "평활 등급이 10거래일 연속 S 아니면 팖(사이에 S면 다시 셈)",
    "R4_weekly": "주 마지막 거래일 평활 등급이 S 아니면 다음 주 첫 시가에 팖",
    "R5_raw_C": "당일(평활 없는) 등급이 C 이하면 팖",
    "R6_monthly_extra": "(격자 밖) 월 마지막 거래일 평활 등급이 S 아니면 팖",
}


def simulate(ctx) -> dict:
    P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E = ctx
    cal = C.index
    T = len(cal)
    Ga, GRa = G.to_numpy(object), GR.to_numpy(object)
    Oa, Ea, uma = O.to_numpy(float), E.to_numpy(bool), um.to_numpy(bool)
    Cf = C.ffill().to_numpy(float)
    Ca = C.to_numpy(float)
    isS = Ga == "S"
    wk = pd.Series(cal.to_period("W-SUN"), index=cal)
    week_last = (wk != wk.shift(-1)).to_numpy().copy()
    week_last[-1] = False  # 마지막 날은 주가 안 끝났을 수 있음(10-09 금요일이지만 다음 시가 없음)
    mo = pd.Series(cal.to_period("M"), index=cal)
    month_last = (mo != mo.shift(-1)).to_numpy().copy()
    month_last[-1] = False
    start = cal.searchsorted(pd.Timestamp(START))
    starts = isS & ~np.vstack([np.zeros((1, isS.shape[1]), bool), isS[:-1]])

    def u_ret(i_sig, i_a, i_b, use_close_end=False):
        sel = Ea[i_sig]
        end = Ca[i_b, sel] if use_close_end else Oa[i_b, sel]
        ra = end / Oa[i_a, sel] - 1.0
        return float(np.nanmean(ra)) if np.isfinite(ra).any() else np.nan

    bench_u = []
    Cpos = C.to_numpy(float)
    for t in range(T):
        if t == 0:
            bench_u.append(0.0)
            continue
        sel = uma[t - 1] & np.isfinite(Cpos[t]) & np.isfinite(Cpos[t - 1])
        if cal[t] < DRUG_CUT:
            sel = sel & (C.columns != "DRUG")
        bench_u.append(float(np.mean(Cpos[t, sel] / Cpos[t - 1, sel] - 1)) if sel.any() else 0.0)
    bench_u = np.array(bench_u)
    qc = P["bench"]["C"].reindex(cal).to_numpy(float)
    bench_q = np.r_[0.0, qc[1:] / qc[:-1] - 1]

    out = {}
    trades_all = {}
    ports = {}
    for rule in RULES:
        trades = []
        csum, ccnt = np.zeros(T), np.zeros(T)
        for j, sym in enumerate(C.columns):
            t = start
            while t < T - 1:
                if not starts[t, j] or (sym == "DRUG" and cal[t] < DRUG_CUT) or not np.isfinite(Oa[t + 1, j]):
                    t += 1
                    continue
                s = t
                e = s + 1
                nons = 0
                d = e
                sig_day, cause = None, None
                while d < T:
                    if not uma[d, j]:
                        sig_day, cause = d, "forced"
                        break
                    g = Ga[d, j]
                    nons = 0 if g == "S" else nons + 1
                    hit = False
                    if rule == "R1_S_lost":
                        hit = g != "S"
                    elif rule == "R2_below_A":
                        hit = g not in ("S", "A")
                    elif rule == "R3_nonS_10d":
                        hit = nons >= 10
                    elif rule == "R4_weekly":
                        hit = week_last[d] and g != "S"
                    elif rule == "R5_raw_C":
                        hit = GRa[d, j] in ("C", "D")
                    elif rule == "R6_monthly_extra":
                        hit = month_last[d] and g != "S"
                    if hit:
                        sig_day, cause = d, "rule"
                        break
                    d += 1
                # 체결
                if sig_day is None or sig_day + 1 >= T:
                    x, px, kind = T - 1, Cf[T - 1, j], "open"
                    sig_day = None if sig_day is None else sig_day
                    if cause is None:
                        cause = "open"
                else:
                    k = sig_day + 1
                    while k < T and not np.isfinite(Oa[k, j]):
                        k += 1
                    if k < T:
                        x, px, kind = k, Oa[k, j], "open_px"
                    else:
                        x, px, kind = T - 1, Cf[T - 1, j], "close_px"
                ret = px / Oa[e, j] - 1
                um_ret = u_ret(s, e, x, use_close_end=(kind != "open_px"))
                tr = {"sym": sym, "sig_in": cal[s].strftime("%Y-%m-%d"), "entry": cal[e].strftime("%Y-%m-%d"),
                      "exit": cal[x].strftime("%Y-%m-%d"), "hold": int(x - e), "ret": float(ret),
                      "ex": float(ret - um_ret) if np.isfinite(um_ret) else None, "cause": cause,
                      "period": period_of(cal[e])}
                if cause == "rule":
                    tr["grade_at_sell"] = Ga[sig_day, j]
                    w = isS[sig_day + 1: sig_day + 1 + WHIP, j]
                    tr["whip"] = bool(w.any()) if (w.any() or sig_day + WHIP <= T - 1) else None
                    st = starts[sig_day + 1: sig_day + 1 + WHIP, j]
                    tr["rebuy10"] = bool(st.any()) if (st.any() or sig_day + WHIP <= T - 1) else None
                    if x + WHIP < T and np.isfinite(Oa[x + WHIP, j]) and kind == "open_px":
                        tr["post10"] = float(Oa[x + WHIP, j] / px - 1)
                        u10 = u_ret(sig_day, x, x + WHIP)
                        tr["post10_ex"] = float(tr["post10"] - u10) if np.isfinite(u10) else None
                trades.append(tr)
                # 일별 기여(같은 비중)
                csum[e] += Ca[e, j] / Oa[e, j] - 1 if np.isfinite(Ca[e, j]) else 0.0
                ccnt[e] += 1
                for dd in range(e + 1, x):
                    csum[dd] += Cf[dd, j] / Cf[dd - 1, j] - 1
                    ccnt[dd] += 1
                if kind == "open_px" and x > e:
                    csum[x] += px / Cf[x - 1, j] - 1
                    ccnt[x] += 1
                elif x > e:
                    csum[x] += Cf[x, j] / Cf[x - 1, j] - 1
                    ccnt[x] += 1
                t = max(x, s + 1)  # 판 날(시가)부터 다음 S 진입을 찾는다
        port = np.where(ccnt > 0, csum / np.maximum(ccnt, 1), 0.0)
        res = {}
        for per in list(PERIODS) + ["all"]:
            a, b = (START, "2026-12-31") if per == "all" else PERIODS[per]
            m = (cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))
            tt = [x for x in trades if per == "all" or x["period"] == per]
            rr = np.array([x["ret"] for x in tt], float)
            ex = np.array([x["ex"] for x in tt if x["ex"] is not None], float)
            hold = np.array([x["hold"] for x in tt], float)
            ruled = [x for x in tt if x["cause"] == "rule"]
            wh = [x["whip"] for x in ruled if x.get("whip") is not None]
            rb = [x["rebuy10"] for x in ruled if x.get("rebuy10") is not None]
            p10 = [x["post10"] for x in ruled if x.get("post10") is not None]
            p10x = [x["post10_ex"] for x in ruled if x.get("post10_ex") is not None]
            srt = np.sort(rr)[::-1]
            top3_out = srt[3:] if len(srt) > 3 else np.array([])
            pos_sum = srt[srt > 0].sum() if len(srt) else 0.0
            k50 = int(np.searchsorted(np.cumsum(srt[srt > 0]), 0.5 * pos_sum) + 1) if pos_sum > 0 else None
            eq = np.cumprod(1 + port[m])
            eu = np.cumprod(1 + bench_u[m])
            eqq = np.cumprod(1 + bench_q[m])
            d = {"trades": len(tt), "rule_exits": len(ruled),
                 "forced_exits": sum(x["cause"] == "forced" for x in tt),
                 "open_at_end": sum(x["cause"] == "open" for x in tt),
                 "hold_mean": r_(hold.mean(), 1) if len(hold) else None,
                 "hold_median": r_(np.median(hold), 1) if len(hold) else None,
                 "ret_mean": pct(rr.mean()) if len(rr) else None,
                 "ret_median": pct(np.median(rr)) if len(rr) else None,
                 "win_pct": r_(100 * np.mean(rr > 0), 1) if len(rr) else None,
                 "ex_mean": pct(ex.mean()) if len(ex) else None,
                 "ex_median": pct(np.median(ex)) if len(ex) else None,
                 "whipsaw_pct": r_(100 * np.mean(wh), 1) if wh else None, "whipsaw_n": len(wh),
                 "rebuy10_pct": r_(100 * np.mean(rb), 1) if rb else None,
                 "post_sell_10d_mean": pct(np.mean(p10)) if p10 else None,
                 "post_sell_10d_ex_mean": pct(np.mean(p10x)) if p10x else None,
                 "ret_mean_without_top3": pct(top3_out.mean()) if len(top3_out) else None,
                 "trades_for_half_of_gains": k50,
                 "port_ret": pct(eq[-1] - 1) if m.any() else None,
                 "port_mdd": pct((eq / np.maximum.accumulate(eq) - 1).min()) if m.any() else None,
                 "avg_positions": r_(ccnt[m].mean(), 1), "invested_days_pct": r_(100 * np.mean(ccnt[m] > 0), 1),
                 "universe_ew_ret": pct(eu[-1] - 1) if m.any() else None,
                 "universe_ew_mdd": pct((eu / np.maximum.accumulate(eu) - 1).min()) if m.any() else None,
                 "qqq_ret": pct(eqq[-1] - 1) if m.any() else None,
                 "days": int(m.sum())}
            if rule == "R5_raw_C":
                gs = [x.get("grade_at_sell") for x in ruled]
                d["smoothed_grade_at_sell"] = {k: sum(g == k for g in gs) for k in ("S", "A", "B", "C", "D")}
            res[per] = d
        out[rule] = res
        trades_all[rule] = trades
        ports[rule] = port
        print(rule, {p: (res[p]["trades"], res[p]["hold_mean"], res[p]["ex_mean"], res[p]["port_ret"],
                         res[p]["whipsaw_pct"]) for p in ("front", "back", "y2025")}, flush=True)
    # 규칙끼리 일별 포트폴리오 수익 차이(R1 대비), Newey-West t(시차 10)
    diff = {}
    for rule in RULES:
        if rule == "R1_S_lost":
            continue
        dd = {}
        for per in list(PERIODS) + ["all"]:
            a, b = (START, "2026-12-31") if per == "all" else PERIODS[per]
            m = (cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))
            x = pd.Series(ports[rule][m] - ports["R1_S_lost"][m])
            dd[per] = {"mean_bp_per_day": r_(x.mean() * 1e4, 2), "nw_t": r_(H.newey_west_t(x, 10), 2), "days": int(m.sum())}
        diff[rule] = dd
    out["_diff_vs_R1"] = diff
    m = cal >= pd.Timestamp(START)
    pf = pd.DataFrame({k: v[m] for k, v in ports.items()}, index=cal[m])
    mon = (1 + pf).groupby(pf.index.to_period("M")).prod() - 1
    out["_monthly_port_ret_pct"] = {str(p): {k: r_(v * 100, 1) for k, v in row.items()} for p, row in mon.iterrows()}
    return out, trades_all


# ------------------------------------------------------------------ 진단(선택에 쓰지 않음): 시간 단위와 급락 기준선
def cadence_diag(ctx, sp) -> dict:
    """S 구간이 주·월·분기 점검일에 보이는지, 짧은 끊김을 이으면 얼마나 긴지, 급락 조건의 평소 빈도."""
    P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E = ctx
    cal = C.index
    T = len(cal)
    Ga = G.to_numpy(object)
    isS = Ga == "S"
    checks_ = {}
    for name, per in (("week", "W-SUN"), ("month", "M"), ("quarter", "Q")):
        p = pd.Series(cal.to_period(per), index=cal)
        checks_[name] = (p != p.shift(-1)).to_numpy().copy()
    done = [r for r in sp if r.get("X") is not None]
    out = {}
    for per in list(PERIODS) + ["all"]:
        rr = [r for r in done if per == "all" or r["period"] == per]
        d = {"spells": len(rr), "S_days": int(sum(r["len"] for r in rr))}
        for name, mk in checks_.items():
            vis, lag = [], []
            for r in rr:
                idx = np.where(mk[r["s"]:r["L"] + 1])[0]
                vis.append(len(idx) > 0)
                if len(idx):
                    lag.append(int(idx[0]))
            v = np.array(vis)
            lens = np.array([r["len"] for r in rr])
            d[name] = {"visible_spells_pct": r_(100 * v.mean(), 1) if len(v) else None,
                       "visible_S_days_pct": r_(100 * lens[v].sum() / max(lens.sum(), 1), 1) if len(v) else None,
                       "lag_to_first_check_median": r_(np.median(lag), 1) if lag else None}
        out[per] = d
    # 짧은 끊김 잇기: 같은 종목에서 S 아닌 날이 g일 이하로 끊겼다 다시 S면 한 묶음
    epi = {}
    for g in (0, 5, 10, 20):
        lens = []
        for sym in C.columns:
            rs = sorted([r for r in sp if r["sym"] == sym], key=lambda r: r["s"])
            cur = None
            for r in rs:
                if cur and r["s"] - cur[1] - 1 <= g:
                    cur[1] = r["L"]
                else:
                    if cur:
                        lens.append(cur)
                    cur = [r["s"], r["L"]]
            if cur:
                lens.append(cur)
        L_ = np.array([b - a + 1 for a, b in lens], float)
        epi[f"gap<={g}"] = {"episodes": int(len(L_)), "len_median": r_(np.median(L_), 1), "len_mean": r_(L_.mean(), 1),
                            "q25_q75": [r_(np.quantile(L_, .25), 1), r_(np.quantile(L_, .75), 1)],
                            "ge20_pct": r_(100 * np.mean(L_ >= 20), 1), "ge60_pct": r_(100 * np.mean(L_ >= 60), 1)}
    # 급락 조건(±10일 안 20일 최고 종가 대비 −15%)이 '끝나지 않은' S인 날에도 얼마나 자주 맞는가
    Ca = C.to_numpy(float)
    dd20 = Ca / C.rolling(ROLL_PEAK, min_periods=1).max().to_numpy(float) - 1.0
    hit = np.zeros(Ca.shape, bool)
    for k in range(-CRASH_HALF, CRASH_HALF + 1):
        sh = np.full(Ca.shape, np.nan)
        if k >= 0:
            sh[:T - k] = dd20[k:]
        else:
            sh[-k:] = dd20[:T + k]
        hit |= np.nan_to_num(sh, nan=0.0) <= CRASH_DD
    mid_days, all_days = [], []
    start = cal.searchsorted(pd.Timestamp(START))
    for r in sp:
        j = C.columns.get_loc(r["sym"])
        for d_ in range(max(r["s"], start), r["L"] + 1):
            left = r["L"] - d_
            if (left >= 20 or r["X"] is None) and d_ + CRASH_HALF <= T - 1:
                mid_days.append(hit[d_, j])
    umask = um.to_numpy(bool)
    for t in range(start, T - CRASH_HALF):
        all_days.extend(hit[t, umask[t]].tolist())
    base = {"S_days_20plus_before_end": {"n": len(mid_days), "pct": r_(100 * np.mean(mid_days), 1)},
            "all_universe_days": {"n": len(all_days), "pct": r_(100 * np.mean(all_days), 1)}}
    return {"visibility": out, "episodes": epi, "crash_condition_base_rate": base}


# ------------------------------------------------------------------ 점검
def checks(ctx, sig) -> dict:
    P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E = ctx
    out = {}
    # final.py 방식(eligible로 순위)과 같은지: 2026 마지막 날 제외
    Gf = FL.hyb_grades(sc_s, P, H.eligible())
    cal = C.index
    m = (cal >= pd.Timestamp("2024-07-01")) & (cal < cal[-1])
    a, b = G.loc[m], Gf.loc[m]
    both = a.notna() & b.notna()
    out["parity_vs_final_eligible"] = {"cells": int(both.sum().sum()),
                                      "agree_pct": r_(100 * float((a[both] == b[both]).sum().sum() / both.sum().sum()), 3),
                                      "nan_mismatch": int((a.notna() != b.notna()).sum().sum())}
    # 미래 정보 시험: 자료를 잘라 다시 계산한 등급·신호의 마지막 행
    ok = True
    detail = {}
    for c in ("2025-03-14", "2026-03-13", "2026-07-22"):
        t = pd.Timestamp(c)
        cut = {k: (v.loc[:t] if isinstance(v, pd.DataFrame) else v) for k, v in P.items() if k != "bench"}
        cut["bench"] = {k: v.loc[:t] for k, v in P["bench"].items()}
        umc = H.universe_mask(cut)
        s2 = FC.make(FC.GRID["E14"], 10)(cut)
        r2 = FC.make(FC.GRID["E14"], 0)(cut)
        g2 = FL.hyb_grades(s2, cut, umc).loc[t]
        gr2 = FL.hyb_grades(r2, cut, umc).loc[t]
        sg2 = signals(cut["C"], cut["H"], s2)
        same_g = bool(((g2 == G.loc[t]) | (g2.isna() & G.loc[t].isna())).all())
        same_gr = bool(((gr2 == GR.loc[t]) | (gr2.isna() & GR.loc[t].isna())).all())
        same_s = True
        for k in sig:
            x, y = sig[k].loc[t], sg2[k].loc[t]
            same_s &= bool(((x.isna() & y.isna()) | np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12)).all())
        detail[c] = {"grade": same_g, "raw_grade": same_gr, "signals": same_s}
        ok &= same_g and same_gr and same_s
    out["no_lookahead"] = ok
    out["no_lookahead_detail"] = detail
    return out


def main():
    t0 = time.time()
    ctx = build()
    P, C, O, Hi, Lo, um, sc_s, G, GR, PR, E = ctx
    print("grades built", f"{time.time() - t0:.0f}s", flush=True)
    sig = signals(C, Hi, sc_s)
    print("signals built", f"{time.time() - t0:.0f}s", flush=True)
    sp = spells(ctx)
    tab = spell_tables(sp)
    print("spells", len(sp), f"{time.time() - t0:.0f}s", flush=True)
    early, fires = early_cut(ctx, sp, sig)
    auc = auc_table(ctx, sp, sig, fires)
    print("early done", early["selection"], f"{time.time() - t0:.0f}s", flush=True)
    sim, trades = simulate(ctx)
    cad = cadence_diag(ctx, sp)
    print("cadence", json.dumps(cad, ensure_ascii=False)[:2500], flush=True)
    chk = checks(ctx, sig)
    print("checks", chk, f"{time.time() - t0:.0f}s", flush=True)
    cal = C.index
    S_counts = (G == "S").sum(axis=1)
    m = cal >= pd.Timestamp(START)
    meta = {
        "panel": [cal[0].strftime("%Y-%m-%d"), cal[-1].strftime("%Y-%m-%d")],
        "symbols": int(C.shape[1]), "start": START, "periods": PERIODS,
        "S_per_day": {"mean": r_(S_counts[m].mean(), 1), "min": int(S_counts[m].min()), "max": int(S_counts[m].max())},
        "graded_per_day_mean": r_(G.loc[m].notna().sum(axis=1).mean(), 1),
        "variant_count": {"(1) 분류 사양": 1, "(2) 먼저 줄이기 신호": len(fires), "(3) 매도 규칙": len(RULES),
                          "그중 격자 밖": 1, "합계": 1 + len(fires) + len(RULES)},
        "definitions": __doc__,
    }
    res = {"meta": meta, "checks": chk, "spells": tab, "cadence_diag": cad, "early_cut": early, "early_auc": auc,
           "sell_rules": sim, "rule_desc": RULES,
           "spell_list": [{k: (r_(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in sp]}
    path = os.path.join(HERE, "results_fade.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: None)
    print("saved", path, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
