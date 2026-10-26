"""과제 E2 — 추세 등급(최종식) 전환이 언제 일어나는가: 달력 경계·실적 발표·펀드 매매 (2026-10-12).

질문(사용자 2026-10-12): 큰 펀드는 일·주·월·분기 단위로 리밸런싱한다. '3월까지 S, 4월부터 아님' 같은
분기 경계 전환이 우연보다 많은가? 등급을 나누는 최소 시간 단위를 정하는 근거를 숫자로 만든다.

등급: 연구 최종식 그대로(code/final.py FINAL_E14s10_HYB).
  점수 = fam_composite.make(GRID['E14'], 10)  (L·M·H·Q 단면 백분위 평균의 10일 평균)
  등급 = fam_ladder.hyb_grades(점수, P, harness.universe_mask(P))  (피라미드 10/20/40/20/10 + 관문)
  종목군: 2026-01-02부터 그날 보유 미국 종목(PDF D → 미국 D), 그 전은 패널 150종목 전체(봉 130개 이상).
분석 구간: 전환일 2024-07-01 ~ 2026-10-09.
제외: 2026-01-02~01-15 전환(종목군이 145→약 60으로 바뀌고 10일 평균이 섞이는 구간).

전환 정의(이틀 모두 등급이 있는 종목만)
  S 이탈 = t−1 S, t는 A 이하.  위험 집합 = t−1 S이고 t에 등급 있음.
  S 진입 = t−1 A 이하, t는 S. 위험 집합 = t−1 A 이하이고 t에 등급 있음.
  t−1 S였는데 t에 종목군 밖(펀드 편출 등)은 이탈로 세지 않고 따로 센다.

검정
  달력: 이탈·진입 수의 구간별 비율을 (a) 무작위 날짜 비율(그 구간 날 수 비율) (b) 위험 집합 비율과 비교.
        카이제곱(독립 가정, 같은 날 여러 종목이 함께 바뀌면 과대)과 원형 이동 순열(날짜별 시계열을 달력
        표지에 대해 1..n−1칸 모두 밀기, 같은 날 몰림·자기상관 보존)을 같이 적는다.
  실적: 종목별 발표일 전체를 같은 무작위 거래일(±16~31)만큼 밀어 1000번 다시 잰다.
  펀드: 종목·펀드별 매매일 전체를 2026 거래일 안에서 같은 무작위 칸만큼 원형 이동해 1000번 다시 잰다.

실행: cd data/derived/analysis/trend_grade/cadence && PYTHONIOENCODING=utf-8 py code_transitions.py
산출: results_transitions.json, results_transitions_earnings_dates.json(yfinance 캐시), results_transitions_fund_trades.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import harness as H  # noqa: E402

sys.path.insert(0, H.ROOT)
from gbrc.etf import market  # noqa: E402
from gbrc.etf.flows import BUY_KINDS, SELL_KINDS, flows_between  # noqa: E402
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402

START, END = "2024-07-01", "2026-10-09"
SWITCH = ("2026-01-02", "2026-01-15")
FUNDS = {"TIME": "456600", "KoAct": "2ETFQ1"}
CODES = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
NAMES = {v: k for k, v in CODES.items()}
QB_LO, QB_HI = -5, 4  # 분기 경계 창: 앞 분기 마지막 5거래일 + 새 분기 첫 5거래일 (10일)
EW_LO, EW_HI = -5, 10  # 실적 창: 반응일 −5 ~ +10 거래일
LEAD = 20
NPERM = 1000
SEED = 20261012
OUT = os.path.join(HERE, "results_transitions.json")
EARN_CACHE = os.path.join(HERE, "results_transitions_earnings_dates.json")
TRADES_OUT = os.path.join(HERE, "results_transitions_fund_trades.json")


def r(x, k=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, k) if np.isfinite(x) else None


# ------------------------------------------------------------------ 등급
def grade_codes(smooth: int) -> pd.DataFrame:
    P = H.load_panel()
    um = H.universe_mask(P)
    sc = FC.make(FC.GRID["E14"], smooth)(P)
    G = FL.hyb_grades(sc, P, um).to_numpy(object)
    out = np.full(G.shape, np.nan)
    for k, v in CODES.items():
        out[G == k] = v
    return pd.DataFrame(out, index=P["C"].index, columns=P["C"].columns)


# ------------------------------------------------------------------ 달력 표지
def calendar_labels(cal: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.Series(np.arange(len(cal)), index=cal)
    ym = cal.to_period("M")
    tdom = s.groupby(ym).cumcount().to_numpy() + 1
    n_in = s.groupby(ym).transform("count").to_numpy()
    last_month = ym[-1]
    tdom_end = np.where(ym == last_month, np.nan, n_in - tdom + 1)  # 마지막 달(10월)은 미완
    qstart = np.array([(d.month in (1, 4, 7, 10)) and t == 1 for d, t in zip(cal, tdom)])
    qs_idx = np.flatnonzero(qstart)
    idx = np.arange(len(cal))
    # 가장 가까운 분기 첫 거래일까지 거리(t − qs). 동률이면 미래 쪽(음수)을 쓴다.
    dist = idx[:, None] - qs_idx[None, :]
    best = np.argmin(np.abs(dist) + (dist > 0) * 0.1, axis=1)
    qk = dist[idx, best]
    miq = np.array([(d.month - 1) % 3 + 1 for d in cal])
    return pd.DataFrame({"tdom": tdom, "tdom_end": tdom_end, "qk": qk, "miq": miq,
                         "month": [d.month for d in cal]}, index=cal)


# ------------------------------------------------------------------ 전환
def transitions(G: pd.DataFrame, cal_sel: pd.DatetimeIndex) -> dict:
    g = G.to_numpy(float)
    pos = G.index.get_indexer(cal_sel)
    g0, g1 = g[pos - 1], g[pos]
    both = np.isfinite(g0) & np.isfinite(g1)
    return {
        "exit": both & (g0 == 4) & (g1 < 4),
        "risk_exit": both & (g0 == 4),
        "entry": both & (g0 < 4) & (g1 == 4),
        "risk_entry": both & (g0 < 4),
        "leave_S": (g0 == 4) & ~np.isfinite(g1),
        "g0": g0, "g1": g1, "pos": pos,
    }


def hr_stats(ev: np.ndarray, ar: np.ndarray, mask: np.ndarray) -> dict:
    """ev·ar: 날짜별 수(1차원). mask: 날짜별 bool."""
    e_in, e_out = ev[mask].sum(), ev[~mask].sum()
    a_in, a_out = ar[mask].sum(), ar[~mask].sum()
    rate_in = e_in / a_in if a_in else np.nan
    rate_out = e_out / a_out if a_out else np.nan
    return {"events_in": int(e_in), "events_total": int(e_in + e_out),
            "event_share_in": r(e_in / max(e_in + e_out, 1)),
            "day_share_in": r(mask.mean()), "risk_share_in": r(a_in / max(a_in + a_out, 1)),
            "hazard_in_pct": r(rate_in * 100, 3), "hazard_out_pct": r(rate_out * 100, 3),
            "hazard_ratio": r(rate_in / rate_out if rate_out else np.nan, 3)}


def circ_perm_hr(ev, ar, mask) -> dict:
    """달력 표지를 1..n−1칸 원형 이동(전수)해 위험비를 다시 잰다. 단측 p(더 많음)·양측 p(|log|)."""
    obs = hr_stats(ev, ar, mask)["hazard_ratio"]
    n = len(ev)
    vals = []
    for s in range(1, n):
        m = np.roll(mask, s)
        a_in, a_out = ar[m].sum(), ar[~m].sum()
        if not a_in or not a_out or not ev[~m].sum():
            continue
        vals.append((ev[m].sum() / a_in) / (ev[~m].sum() / a_out))
    vals = np.array(vals)
    lo = np.log(np.clip(vals, 1e-9, None))
    lobs = np.log(max(obs, 1e-9)) if obs else 0.0
    return {"observed": obs, "n_shifts": int(len(vals)),
            "p_more": r((1 + (vals >= obs).sum()) / (1 + len(vals))),
            "p_two_sided": r((1 + (np.abs(lo) >= abs(lobs)).sum()) / (1 + len(vals))),
            "null_p05_p50_p95": [r(np.percentile(vals, q), 3) for q in (5, 50, 95)]}


def chi_bins(ev, ar, labels, bins) -> dict:
    """구간별 관측 수 vs 기대(위험 집합 비율, 무작위 날짜 비율). 카이제곱 + 원형 이동 순열."""
    def stat(lab):
        O = np.array([ev[lab == b].sum() for b in bins], float)
        A = np.array([ar[lab == b].sum() for b in bins], float)
        E = O.sum() * A / A.sum()
        ok = E > 0
        return float(((O[ok] - E[ok]) ** 2 / E[ok]).sum()), O, E
    chi, O, E = stat(labels)
    D = np.array([(labels == b).sum() for b in bins], float)
    Ed = O.sum() * D / D.sum()
    chi_d = float(((O - Ed) ** 2 / np.where(Ed > 0, Ed, np.nan))[Ed > 0].sum())
    dfree = int((E > 0).sum() - 1)
    n = len(ev)
    null = []
    for s in range(1, n):
        null.append(stat(np.roll(labels, s))[0])
    null = np.array(null)
    return {
        "bins": [str(b) for b in bins], "observed": O.astype(int).tolist(),
        "expected_by_risk": [r(x, 1) for x in E], "expected_by_days": [r(x, 1) for x in Ed],
        "ratio_obs_over_risk": [r(o / e, 2) if e else None for o, e in zip(O, E)],
        "chi2_vs_risk": r(chi, 2), "df": dfree, "p_chi2_naive": r(stats.chi2.sf(chi, dfree)),
        "chi2_vs_days": r(chi_d, 2), "p_chi2_days_naive": r(stats.chi2.sf(chi_d, dfree)),
        "p_circular_perm": r((1 + (null >= chi).sum()) / (1 + len(null))),
    }


def calendar_tests(T: dict, lab: pd.DataFrame, keep: np.ndarray) -> dict:
    out = {}
    L = lab.iloc[T["pos"]]
    for kind, risk in (("exit", "risk_exit"), ("entry", "risk_entry")):
        ev = T[kind].sum(axis=1).astype(float)[keep]
        ar = T[risk].sum(axis=1).astype(float)[keep]
        Lk = L[keep]
        qk = Lk["qk"].to_numpy()
        res = {}
        for name, lo, hi in (("q_boundary_pm5", QB_LO, QB_HI), ("q_boundary_pm10", -10, 9),
                             ("q_end_last5", -5, -1), ("q_start_first5", 0, 4)):
            m = (qk >= lo) & (qk <= hi)
            res[name] = {**hr_stats(ev, ar, m), "perm": circ_perm_hr(ev, ar, m)}
            # 단순 이항 검정(독립 가정): 이벤트가 창에 들어갈 확률 = 위험 집합 비율
            k, nn = int(ev[m].sum()), int(ev.sum())
            p0 = ar[m].sum() / ar.sum()
            res[name]["p_binom_naive_vs_risk"] = r(stats.binomtest(k, nn, p0, alternative="greater").pvalue)
            res[name]["p_binom_naive_vs_days"] = r(stats.binomtest(k, nn, float(m.mean()), alternative="greater").pvalue)
        # 월 안 위치(1~23) 원표
        tdom = Lk["tdom"].to_numpy()
        res["tdom_1_23"] = {"observed": [int(ev[tdom == i].sum()) for i in range(1, 24)],
                            "risk": [int(ar[tdom == i].sum()) for i in range(1, 24)],
                            "days": [int((tdom == i).sum()) for i in range(1, 24)]}
        res["tdom_1_23"]["hazard_pct"] = [r(o / a * 100, 3) if a else None for o, a in
                                          zip(res["tdom_1_23"]["observed"], res["tdom_1_23"]["risk"])]
        # 월 안 위치 묶음: 첫 5거래일 / 마지막 5거래일 / 가운데 (10월 미완 달은 마지막 5일 판단 불가 → 뺀다)
        te = Lk["tdom_end"].to_numpy()
        okm = np.isfinite(te)
        grp = np.where(tdom <= 5, "first5", np.where(te <= 5, "last5", "middle"))
        res["month_position_3bins"] = chi_bins(ev[okm], ar[okm], grp[okm], ["first5", "middle", "last5"])
        res["tdom_23bins"] = chi_bins(ev, ar, np.minimum(tdom, 23), list(range(1, 24)))
        res["month_in_quarter"] = chi_bins(ev, ar, Lk["miq"].to_numpy(), [1, 2, 3])
        out[kind] = res
    return out


# ------------------------------------------------------------------ 이탈 모양·지속성
def exit_shape(T, G, keep, cal_sel) -> dict:
    P = H.load_panel()
    C = P["C"].to_numpy(float)
    um = H.universe_mask(P).to_numpy(bool)
    g = G.to_numpy(float)
    rows = []
    ex = T["exit"] & keep[:, None]
    for i, j in zip(*np.nonzero(ex)):
        t = T["pos"][i]
        rec = {"to": NAMES[int(g[t, j])]}
        for h in (5, 10, 20):
            if t - h >= 0 and np.isfinite(C[t - h, j]):
                rr = C[t, j] / C[t - h, j] - 1
                uni = um[t] & np.isfinite(C[t - h]) & np.isfinite(C[t])
                mu = np.nanmean(C[t, uni] / C[t - h, uni] - 1)
                rec[f"ret{h}"] = rr
                rec[f"ex{h}"] = rr - mu
        k = 0
        while t - 1 - k >= 0 and g[t - 1 - k, j] == 4:
            k += 1
        rec["s_streak_before"] = k
        for h in (5, 10, 20):
            fut = g[t + 1:t + 1 + h, j]
            rec[f"reenter{h}"] = bool((fut == 4).any()) if len(fut) == h else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = {"n": int(len(df)), "to_grade": dict(Counter(df["to"])) if len(df) else {}}
    for c in ("ret5", "ret10", "ret20", "ex5", "ex10", "ex20"):
        x = df[c].dropna()
        out[c] = {"q25": r(x.quantile(.25) * 100, 2), "median": r(x.median() * 100, 2),
                  "q75": r(x.quantile(.75) * 100, 2), "mean": r(x.mean() * 100, 2)}
    x = df["ret10"].dropna()
    out["share_ret10_le_-15"] = r((x <= -0.15).mean())
    out["share_ret10_le_-10"] = r((x <= -0.10).mean())
    out["share_ret10_gt_0"] = r((x > 0).mean())
    out["share_ex10_le_-10"] = r((df["ex10"].dropna() <= -0.10).mean())
    s = df["s_streak_before"]
    out["s_streak_before"] = {"median": r(s.median(), 1), "mean": r(s.mean(), 1),
                              "share_le5": r((s <= 5).mean()), "share_le21": r((s <= 21).mean())}
    for h in (5, 10, 20):
        x = df[f"reenter{h}"].dropna().astype(bool)
        out[f"reenter_within_{h}d"] = r(x.mean())
    return out


def persistence(G: pd.DataFrame, cal_sel: pd.DatetimeIndex) -> dict:
    """S였던 종목이 h거래일 뒤에도 S일 확률(전환 연구 구간 안, 종목군 전환 경계를 건너는 쌍 제외)."""
    g = G.to_numpy(float)
    pos = G.index.get_indexer(cal_sel)
    sw = G.index.get_loc(pd.Timestamp(SWITCH[1]))
    sw0 = G.index.get_loc(pd.Timestamp(SWITCH[0]))
    out = {}
    for label, sub in (("all", None), ("pre2026", "pre"), ("2026", "post")):
        res = {}
        for h in (1, 5, 10, 21, 42, 63):
            t0 = pos[(pos + h) <= pos[-1]]
            if sub == "pre":
                t0 = t0[t0 + h < sw0]
            elif sub == "post":
                t0 = t0[t0 >= sw]
            else:
                t0 = t0[~((t0 < sw) & (t0 + h >= sw0))]
            a, b = g[t0], g[t0 + h]
            ok = (a == 4) & np.isfinite(b)
            base = np.isfinite(a) & np.isfinite(b)
            res[f"h{h}"] = {"p_still_S": r((b[ok] == 4).mean()), "n": int(ok.sum()),
                            "p_S_any": r((b[base] == 4).mean())}
        out[label] = res
    # S 연속 구간 길이(종목군 밖·연구 구간 끝은 끊김으로 처리, 양 끝에 걸린 구간은 censored)
    spells, cens = [], 0
    sub = g[pos]
    for j in range(sub.shape[1]):
        col = sub[:, j] == 4
        run = 0
        for i, v in enumerate(col):
            brk = cal_sel[i] == pd.Timestamp(SWITCH[0])
            if v and not brk:
                run += 1
            else:
                if run:
                    spells.append(run)
                run = 1 if (v and brk) else 0
        if run:
            cens += 1
    sp = np.array(spells)
    out["s_spells"] = {"n": int(len(sp)), "median": r(np.median(sp), 1), "mean": r(sp.mean(), 1),
                       "share_le5": r((sp <= 5).mean()), "share_le10": r((sp <= 10).mean()),
                       "share_le21": r((sp <= 21).mean()), "share_le63": r((sp <= 63).mean()),
                       "stockday_share_in_spells_ge21": r(sp[sp >= 21].sum() / sp.sum()),
                       "stockday_share_in_spells_ge63": r(sp[sp >= 63].sum() / sp.sum()),
                       "open_at_end": int(cens)}
    return out


# ------------------------------------------------------------------ 월 단위 (4)
def monthly(G: pd.DataFrame) -> dict:
    cal = G.index
    sel = cal[(cal >= pd.Timestamp("2024-07-01")) & (cal <= pd.Timestamp(END))]
    months = sorted(set(sel.to_period("M")))
    g = G.loc[sel]
    ym = sel.to_period("M")
    end_grade, s_share, graded_share = {}, {}, {}
    for m in months:
        rows = g[ym == m]
        end_grade[m] = rows.iloc[-1]
        s_share[m] = (rows == 4).sum() / rows.notna().sum().replace(0, np.nan)
        graded_share[m] = rows.notna().mean()
    pairs = []
    for m0, m1 in zip(months, months[1:]):
        if str(m0) == "2025-12":
            continue  # 종목군 전환
        e0, e1 = end_grade[m0], end_grade[m1]
        okE = (e0 == 4) & e1.notna()
        okN = (e0 < 4) & e1.notna()
        sm0 = (s_share[m0] >= 0.5) & (graded_share[m1] >= 0.5)
        pairs.append({
            "pair": f"{m0}->{m1}", "boundary": m0.month in (3, 6, 9, 12), "partial_next": str(m1) == "2026-10",
            "pre2026": m1.year < 2026,
            "end_S": int(okE.sum()), "end_exit": int(((e1 < 4) & okE).sum()),
            "end_nonS": int(okN.sum()), "end_entry": int(((e1 == 4) & okN).sum()),
            "Smonth": int(sm0.sum()), "Smonth_exit": int((sm0 & (s_share[m1] < 0.5)).sum()),
            "nonSmonth": int(((s_share[m0] < 0.5) & (graded_share[m1] >= 0.5) & s_share[m0].notna()).sum()),
            "nonSmonth_entry": int(((s_share[m0] < 0.5) & (s_share[m1] >= 0.5) & (graded_share[m1] >= 0.5)).sum()),
        })
    df = pd.DataFrame(pairs)
    rng = np.random.default_rng(SEED)
    out = {"pairs": pairs, "tests": {}}
    for subset_name, sub in (("all", df), ("all_no_partial", df[~df.partial_next]),
                             ("pre2026", df[df.pre2026]), ("2026", df[~df.pre2026])):
        res = {}
        for ev, risk in (("end_exit", "end_S"), ("end_entry", "end_nonS"),
                         ("Smonth_exit", "Smonth"), ("nonSmonth_entry", "nonSmonth")):
            b = sub[sub.boundary]
            nb = sub[~sub.boundary]
            rb = b[ev].sum() / max(b[risk].sum(), 1)
            rn = nb[ev].sum() / max(nb[risk].sum(), 1)
            table = [[int(b[ev].sum()), int(b[risk].sum() - b[ev].sum())],
                     [int(nb[ev].sum()), int(nb[risk].sum() - nb[ev].sum())]]
            p_f = stats.fisher_exact(table, alternative="greater").pvalue if b[risk].sum() and nb[risk].sum() else None
            # 순열: 어떤 달 쌍이 '분기 경계'인지 표지를 무작위로 다시 붙인다(같은 개수).
            k = int(sub.boundary.sum())
            evv, rk = sub[ev].to_numpy(float), sub[risk].to_numpy(float)
            null = []
            for _ in range(20000):
                pick = np.zeros(len(sub), bool)
                pick[rng.choice(len(sub), k, replace=False)] = True
                a1, a2 = rk[pick].sum(), rk[~pick].sum()
                if a1 and a2:
                    null.append(evv[pick].sum() / a1 - evv[~pick].sum() / a2)
            null = np.array(null)
            diff = rb - rn
            per_pair = (sub[ev] / sub[risk].replace(0, np.nan))
            res[ev] = {"boundary_rate": r(rb), "other_rate": r(rn), "diff": r(diff), "ratio": r(rb / rn if rn else np.nan, 3),
                       "boundary_events": int(b[ev].sum()), "boundary_risk": int(b[risk].sum()),
                       "other_events": int(nb[ev].sum()), "other_risk": int(nb[risk].sum()),
                       "n_boundary_pairs": k, "n_pairs": int(len(sub)),
                       "p_fisher_naive": r(p_f), "p_perm_labels": r((1 + (null >= diff).sum()) / (1 + len(null))),
                       "boundary_pair_rates": {p: r(v) for p, v in zip(b["pair"], per_pair[sub.boundary])},
                       "other_pair_rate_median": r(per_pair[~sub.boundary].median())}
        out["tests"][subset_name] = res
    return out


# ------------------------------------------------------------------ 실적 (2)
def fetch_earnings(syms: list[str]) -> dict:
    cache = {}
    if os.path.exists(EARN_CACHE):
        cache = json.load(open(EARN_CACHE, encoding="utf-8"))
    import yfinance as yf
    todo = [s for s in syms if s not in cache.get("dates", {}) and s not in cache.get("failed", {})]
    dates, failed = cache.get("dates", {}), cache.get("failed", {})
    for s in todo:
        try:
            df = yf.Ticker(s).get_earnings_dates(limit=16)
            if df is None or len(df) == 0:
                failed[s] = "빈 결과"
            else:
                dates[s] = sorted({str(ts) for ts in df.index})
        except Exception as e:  # noqa: BLE001
            failed[s] = repr(e)[:200]
        time.sleep(0.3)
    cache = {"source": "yfinance Ticker(sym).get_earnings_dates(limit=16)", "fetched": time.strftime("%Y-%m-%d %H:%M"),
             "dates": dates, "failed": failed}
    with open(EARN_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=1)
    return cache


def reaction_index(cal: pd.DatetimeIndex, stamps: list[str]) -> list[int]:
    """발표 시각 16:00 이후면 다음 거래일, 그 전(장 전·장중)이면 그날(휴장이면 다음 거래일)."""
    out = set()
    for s in stamps:
        ts = pd.Timestamp(s)
        day = pd.Timestamp(ts.date())
        after = ts.hour >= 16
        i = cal.searchsorted(day, side="right" if after else "left")
        if i < len(cal):
            out.add(int(i))
    return sorted(out)


def earnings_analysis(G: pd.DataFrame, T: dict, keep: np.ndarray, cal_sel, lab) -> dict:
    cal = G.index
    syms = list(G.columns)
    cache = fetch_earnings(syms)
    react = {s: reaction_index(cal, cache["dates"][s]) for s in syms if s in cache["dates"]}
    p0, p1 = cal.get_loc(cal_sel[0]), cal.get_loc(cal_sel[-1])
    cover = {s: sum(1 for i in v if p0 <= i <= p1) for s, v in react.items()}
    # 2024-07~2026-10 구간에 등급이 있었던 종목 중 발표일이 6개 미만인 종목(부분 커버)
    graded_any = G.iloc[p0:p1 + 1].notna().any()
    partial = {s: c for s, c in cover.items() if c < 6 and graded_any.get(s, False)}
    missing = [s for s in syms if s not in react]
    g = G.to_numpy(float)
    n_t, n_s = len(cal), len(syms)
    j_of = {s: j for j, s in enumerate(syms)}

    def window_mask(rx: dict, lo=EW_LO, hi=EW_HI) -> tuple[np.ndarray, np.ndarray]:
        W = np.zeros((n_t, n_s), bool)
        K = np.full((n_t, n_s), 99)
        for s, idx in rx.items():
            j = j_of[s]
            for i in idx:
                a, b = max(i + lo, 0), min(i + hi, n_t - 1)
                W[a:b + 1, j] = True
                ks = np.arange(a, b + 1) - i
                cur = K[a:b + 1, j]
                K[a:b + 1, j] = np.where(np.abs(ks) < np.abs(cur), ks, cur)
        return W, K

    has_e = np.array([s in react for s in syms])
    pos = T["pos"]
    kk = keep[:, None] & has_e[None, :]

    def hr(W):
        Wp = W[pos]
        out = {}
        for kind, risk in (("exit", "risk_exit"), ("entry", "risk_entry")):
            ev, ar = T[kind] & kk, T[risk] & kk
            e_in, a_in = (ev & Wp).sum(), (ar & Wp).sum()
            e_out, a_out = (ev & ~Wp).sum(), (ar & ~Wp).sum()
            out[kind] = {"events_in": int(e_in), "events_total": int(e_in + e_out),
                         "event_share_in": r(e_in / max(e_in + e_out, 1)),
                         "risk_share_in": r(a_in / max(a_in + a_out, 1)),
                         "hazard_in_pct": r(e_in / a_in * 100, 3) if a_in else None,
                         "hazard_out_pct": r(e_out / a_out * 100, 3) if a_out else None,
                         "hazard_ratio": r((e_in / a_in) / (e_out / a_out), 3) if a_in and a_out and e_out else None}
        return out

    W, K = window_mask(react)
    obs = hr(W)
    rng = np.random.default_rng(SEED)
    offs = np.r_[np.arange(-31, -15), np.arange(16, 32)]
    null = {"exit": [], "entry": []}
    for _ in range(NPERM):
        rx = {s: [i + int(rng.choice(offs)) for i in v] for s, v in react.items()}
        rx = {s: [i for i in v if 0 <= i < n_t] for s, v in rx.items()}
        Wn, _ = window_mask(rx)
        h = hr(Wn)
        for k in null:
            null[k].append(h[k]["hazard_ratio"] or np.nan)
    for k in null:
        arr = np.array(null[k], float)
        o = obs[k]["hazard_ratio"]
        obs[k]["p_perm_shifted_calendar"] = r((1 + (arr >= o).sum()) / (1 + np.isfinite(arr).sum()))
        obs[k]["null_p05_p50_p95"] = [r(np.nanpercentile(arr, q), 3) for q in (5, 50, 95)]
    # 사건 시각 분포: 반응일 기준 k = −5..+10
    Kp = K[pos]
    prof = {}
    for kind, risk in (("exit", "risk_exit"), ("entry", "risk_entry")):
        ev, ar = T[kind] & kk, T[risk] & kk
        rows = {}
        for k in range(EW_LO, EW_HI + 1):
            m = Kp == k
            rows[str(k)] = {"events": int((ev & m).sum()), "risk": int((ar & m).sum()),
                            "hazard_pct": r((ev & m).sum() / max((ar & m).sum(), 1) * 100, 3)}
        prof[kind] = rows
    # 발표 뒤 S 유지 확률과 등급 변화 분포
    sw0, sw1 = cal.get_loc(pd.Timestamp(SWITCH[0])), cal.get_loc(pd.Timestamp(SWITCH[1]))

    def crosses(a, b):
        return a < sw0 <= b or (sw0 <= a <= sw1) or (sw0 <= b <= sw1)

    react_any = np.zeros((n_t, n_s), bool)
    for s, idx in react.items():
        for i in idx:
            react_any[i, j_of[s]] = True
    cum = np.cumsum(react_any, axis=0)

    def has_react(a, b, j):  # (a, b] 안 반응일 존재
        return cum[min(b, n_t - 1), j] - cum[a, j] > 0

    retention, matrix_e, matrix_b = {}, np.zeros((5, 5), int), np.zeros((5, 5), int)
    for h in (6, 11, 21):  # r−1 → r+5, r+10, r+20
        kept_e, n_e, kept_b, n_b = 0, 0, 0, 0
        for s, idx in react.items():
            j = j_of[s]
            for i in idx:
                t0 = i - 1
                if t0 < p0 or t0 + h > p1 or crosses(t0, t0 + h):
                    continue
                a, b = g[t0, j], g[t0 + h, j]
                if a == 4 and np.isfinite(b):
                    n_e += 1
                    kept_e += b == 4
                if h == 11 and np.isfinite(a) and np.isfinite(b):
                    matrix_e[4 - int(a), 4 - int(b)] += 1
        for t0 in range(p0, p1 - h + 1):
            if crosses(t0, t0 + h):
                continue
            a, b = g[t0], g[t0 + h]
            for j in range(n_s):
                if not has_e[j] or has_react(t0, t0 + h, j):
                    continue
                if a[j] == 4 and np.isfinite(b[j]):
                    n_b += 1
                    kept_b += b[j] == 4
                if h == 11 and np.isfinite(a[j]) and np.isfinite(b[j]):
                    matrix_b[4 - int(a[j]), 4 - int(b[j])] += 1
        retention[f"r-1_to_r+{h - 1}"] = {"earnings_p_still_S": r(kept_e / max(n_e, 1)), "earnings_n": int(n_e),
                                          "no_earnings_p_still_S": r(kept_b / max(n_b, 1)), "no_earnings_n": int(n_b)}

    def summarize(M):
        lab5 = ["S", "A", "B", "C", "D"]
        rows = {}
        for i, a in enumerate(lab5):
            tot = M[i].sum()
            up = M[i, :i].sum()
            down = M[i, i + 1:].sum()
            rows[a] = {"n": int(tot), "up": r(up / max(tot, 1)), "same": r(M[i, i] / max(tot, 1)),
                       "down": r(down / max(tot, 1)), "to": {lab5[k]: int(M[i, k]) for k in range(5)}}
        allm = M.sum()
        idx = np.arange(5)
        mad = (np.abs(idx[:, None] - idx[None, :]) * M).sum() / max(allm, 1)
        return {"by_start": rows, "mean_abs_grade_change": r(mad, 3),
                "share_changed": r(1 - np.trace(M) / max(allm, 1)),
                "share_moved_2plus": r((M * (np.abs(idx[:, None] - idx[None, :]) >= 2)).sum() / max(allm, 1))}

    # 실적 반응일이 분기 경계 창에 들어가는 비율(달력 경계와 실적이 겹치는 정도)
    qk_all = lab["qk"].to_numpy()
    rd = [i for v in react.values() for i in v if p0 <= i <= p1]
    qk_r = qk_all[rd]
    days_share = float(((qk_all[p0:p1 + 1] >= QB_LO) & (qk_all[p0:p1 + 1] <= QB_HI)).mean())
    miq_r = Counter(int(x) for x in lab["miq"].to_numpy()[rd])
    tdom_r = lab["tdom"].to_numpy()[rd]
    return {
        "source": cache["source"], "fetched": cache["fetched"],
        "n_symbols_with_dates": len(react), "failed": cache["failed"], "missing": missing,
        "partial_coverage_lt6_in_window": partial,
        "n_reaction_days_in_window": len(rd),
        "hazard": obs, "event_time_profile": prof, "retention": retention,
        "grade_change_r-1_to_r+10": {"earnings": summarize(matrix_e), "no_earnings_baseline": summarize(matrix_b)},
        "earnings_vs_quarter_boundary": {
            "share_reaction_days_in_q_boundary_pm5": r(((qk_r >= QB_LO) & (qk_r <= QB_HI)).mean()),
            "share_trading_days_in_q_boundary_pm5": r(days_share),
            "reaction_qk_median": r(np.median(qk_r), 1),
            "reaction_qk_q25_q75": [r(np.percentile(qk_r, 25), 1), r(np.percentile(qk_r, 75), 1)],
            "month_in_quarter_counts": {str(k): v for k, v in sorted(miq_r.items())},
            "tdom_median": r(np.median(tdom_r), 1),
        },
        "_W": W, "_react": react,
    }


# ------------------------------------------------------------------ 펀드 매매 (3)
def fund_trades(cal: pd.DatetimeIndex) -> tuple[list[dict], dict, list[str]]:
    snaps = {name: load_fund_snapshots(fid, "2026-12-31") for name, fid in FUNDS.items()}
    sessions = sorted(set().union(*[set(v) for v in snaps.values()]))
    cols = set(H.load_panel()["C"].columns)
    events, unmatched = [], Counter()
    held = {}
    for name, sn in snaps.items():
        dates = sorted(sn)
        held[name] = {}
        for d in dates:
            hs = set()
            for row in sn[d].get("holdings", []):
                tk = market.us_ticker(market.holding_key(row.get("code")))
                if tk:
                    tk = H.ALIASES.get(tk, tk)
                    if tk in cols:
                        hs.add(tk)
            held[name][d] = hs
        for prev, cur in zip(dates, dates[1:]):
            res = flows_between(sn, prev, cur)
            if res is None:
                continue
            gap = sessions.index(cur) - sessions.index(prev)
            for fl in res.flows.values():
                if fl.kind not in BUY_KINDS | SELL_KINDS:
                    continue
                if not market.is_us_listed(fl.key):
                    continue
                tk = market.us_ticker(fl.key)
                tk = H.ALIASES.get(tk, tk)
                if tk not in cols:
                    unmatched[tk] += 1
                    continue
                ref = market.price_reference_date(fl.key, cur, sessions)
                mgr = cal[cal <= pd.Timestamp(ref)][-1]
                pubs = cal[cal >= pd.Timestamp(cur)]
                krx = market.previous_session(cur, 1, sessions)
                events.append({
                    "fund": name, "pdf_date": cur, "krx_session": krx, "mgr_us_day": mgr.strftime("%Y-%m-%d"),
                    "pub_us_day": pubs[0].strftime("%Y-%m-%d") if len(pubs) else None,
                    "sym": tk, "kind": fl.kind, "side": "buy" if fl.kind in BUY_KINDS else "sell",
                    "selective_pct": r(fl.selective_pct, 2), "trade_pp": r(fl.trade_pp, 3),
                    "weight": r(fl.weight, 3), "gap_sessions": gap,
                })
    return events, held, sessions, dict(unmatched)


def fund_analysis(G: pd.DataFrame, T: dict, keep: np.ndarray, cal_sel, lab) -> dict:
    cal = G.index
    events, held, sessions, unmatched = fund_trades(cal)
    with open(TRADES_OUT, "w", encoding="utf-8") as fh:
        json.dump({"note": "flows_between(연속 스냅샷) 미국 상장 실매매. mgr_us_day = PDF 평가 종가일(매니저 매매 무렵), "
                           "pub_us_day = PDF 날짜 이후 첫 미국 거래일(공개 뒤 따라 살 수 있는 날)",
                   "events": events, "unmatched": unmatched}, fh, ensure_ascii=False, indent=1)
    g = G.to_numpy(float)
    syms = list(G.columns)
    j_of = {s: j for j, s in enumerate(syms)}
    n_t = len(cal)
    pos = T["pos"]
    first_cov = min(cal.get_loc(pd.Timestamp(e["mgr_us_day"])) for e in events)
    last_cov = cal.get_loc(cal[cal <= pd.Timestamp(max(e["mgr_us_day"] for e in events))][-1])
    # 펀드별 보유 여부(날짜 × 종목): 그날 이전 최신 PDF
    hold = {}
    for name, hm in held.items():
        ds = sorted(hm)
        M = np.zeros((n_t, len(syms)), bool)
        k = -1
        for i, t in enumerate(cal):
            ts = t.strftime("%Y-%m-%d")
            while k + 1 < len(ds) and ds[k + 1] <= ts:
                k += 1
            if k >= 0 and t >= pd.Timestamp(ds[0]):
                for s in hm[ds[k]]:
                    M[i, j_of[s]] = True
        hold[name] = M
    hold["either"] = hold["TIME"] | hold["KoAct"]

    def trade_matrix(side, fund, basis="mgr_us_day", shift=None):
        M = np.zeros((n_t, len(syms)), bool)
        for e in events:
            if e["side"] != side or (fund != "either" and e["fund"] != fund):
                continue
            i = cal.get_loc(pd.Timestamp(e[basis]))
            j = j_of[e["sym"]]
            if shift is not None:
                span = last_cov - first_cov + 1
                i = first_cov + (i - first_cov + shift[(e["fund"], e["sym"])]) % span
            M[i, j] = True
        return M

    def lead_stats(M, kind, risk, stay, fund, extra=None):
        """이벤트(t) 앞 0~LEAD거래일 안 매매가 있었던 비율, 대조군과 비교."""
        cum = np.cumsum(M, axis=0)
        lo = pos - LEAD - 1
        win = cum[pos] - np.where(lo[:, None] >= 0, cum[np.clip(lo, 0, None)], 0)
        has = win > 0
        okt = keep & (pos - LEAD >= first_cov) & (pos <= last_cov + 0)
        # 그 펀드가 창(t−LEAD..t) 안 어느 날이든 들고 있던 종목만(편출 뒤 사라진 종목도 포함되게)
        hc = np.cumsum(hold[fund], axis=0)
        H_ = (hc[pos] - np.where(lo[:, None] >= 0, hc[np.clip(lo, 0, None)], 0)) > 0
        ev = T[kind] & okt[:, None] & H_
        ct = stay & okt[:, None] & H_
        if extra is not None:
            ev, ct = ev & extra, ct & extra
        a = has[ev].mean() if ev.sum() else np.nan
        b = has[ct].mean() if ct.sum() else np.nan
        return a, b, int(ev.sum()), int(ct.sum()), has, ev

    out = {"n_events": len(events), "unmatched_tickers": unmatched,
           "coverage_us_days": [cal[first_cov].strftime("%Y-%m-%d"), cal[last_cov].strftime("%Y-%m-%d")],
           "events_by_fund_kind": {f"{f}:{k}": c for (f, k), c in Counter((e["fund"], e["kind"]) for e in events).items()},
           "multi_session_gaps": sum(1 for e in events if e["gap_sessions"] > 1)}
    stay_S = (T["g0"] == 4) & (T["g1"] == 4)
    stay_N = (T["g0"] < 4) & (T["g1"] < 4)
    rng = np.random.default_rng(SEED)
    keys = sorted({(e["fund"], e["sym"]) for e in events})
    lead = {}
    for label, side, kind, risk, stay in (("sell_before_S_exit", "sell", "exit", "risk_exit", stay_S),
                                          ("buy_before_S_entry", "buy", "entry", "risk_entry", stay_N)):
        res = {}
        for fund in ("either", "TIME", "KoAct"):
            for basis in ("mgr_us_day", "pub_us_day"):
                M = trade_matrix(side, fund, basis)
                a, b, ne, nc, has, ev = lead_stats(M, kind, risk, stay, fund)
                item = {"p_trade_in_lead_given_event": r(a), "p_trade_in_lead_given_stay": r(b),
                        "ratio": r(a / b if b else np.nan, 3), "n_events": ne, "n_stay": nc}
                if basis == "mgr_us_day":
                    null = []
                    for _ in range(NPERM):
                        sh = {k: int(rng.integers(0, last_cov - first_cov + 1)) for k in keys}
                        Mn = trade_matrix(side, fund, basis, sh)
                        an, bn, *_ = lead_stats(Mn, kind, risk, stay, fund)
                        null.append(an - bn)
                    null = np.array(null, float)
                    item["p_perm_shift"] = r((1 + (null >= (a - b)).sum()) / (1 + np.isfinite(null).sum()))
                    item["null_diff_p05_p50_p95"] = [r(np.nanpercentile(null, q), 4) for q in (5, 50, 95)]
                    # 선행 일수 분포(가장 가까운 앞선 매매 → 이벤트)
                    lags = []
                    for i, j in zip(*np.nonzero(ev & has)):
                        t = pos[i]
                        prior = np.flatnonzero(M[max(t - LEAD, 0):t + 1, j])
                        lags.append(int(t - (max(t - LEAD, 0) + prior[-1])))
                    item["lag_days_hist_0_20"] = [lags.count(k) for k in range(LEAD + 1)]
                res[f"{fund}|{basis}"] = item
            if side == "buy":
                # 층화: 전날 A였던 종목끼리만(진입 대부분이 A에서 온다)
                M = trade_matrix(side, fund, "mgr_us_day")
                a, b, ne, nc, *_ = lead_stats(M, kind, risk, stay, fund, extra=(T["g0"] == 3))
                res[f"{fund}|mgr_us_day|from_A_only"] = {"p_trade_in_lead_given_event": r(a),
                                                        "p_trade_in_lead_given_stay": r(b),
                                                        "ratio": r(a / b if b else np.nan, 3),
                                                        "n_events": ne, "n_stay": nc}
        lead[label] = res
    out["lead"] = lead
    # 반대 방향: 매매 뒤 20거래일 안 S 이탈/진입 확률
    after = {}
    for label, side, want in (("after_sell_S_exit20", "sell", "exit"), ("after_buy_S_entry20", "buy", "entry")):
        res = {}
        for fund in ("either", "TIME", "KoAct"):
            M = trade_matrix(side, fund, "mgr_us_day")
            H_ = hold["either"] if fund == "either" else hold[fund]
            ev_hits, ev_n, bs_hits, bs_n = 0, 0, 0, 0
            strat = defaultdict(lambda: [0, 0, 0, 0])  # 매매 시점 등급별 [매매 적중, 매매 수, 대조 적중, 대조 수]
            cum = np.cumsum(M, axis=0)
            for t in range(max(first_cov, cal.get_loc(pd.Timestamp(SWITCH[1])) + 1), min(last_cov, n_t - 1 - LEAD) + 1):
                a = g[t]
                fut = g[t + 1:t + 1 + LEAD]
                for j in range(len(syms)):
                    if not np.isfinite(a[j]) or not H_[t, j]:
                        continue
                    if want == "exit":
                        if a[j] != 4:
                            continue
                        f = fut[:, j]
                        hit = bool(((f < 4) & np.isfinite(f)).any())
                    else:
                        if a[j] == 4:
                            continue
                        hit = bool((fut[:, j] == 4).any())
                    recent = cum[t, j] - (cum[t - LEAD - 1, j] if t - LEAD - 1 >= 0 else 0) > 0
                    st = strat[NAMES[int(a[j])]]
                    if M[t, j]:
                        ev_n += 1
                        ev_hits += hit
                        st[0] += hit
                        st[1] += 1
                    elif not recent:
                        bs_n += 1
                        bs_hits += hit
                        st[2] += hit
                        st[3] += 1
            res[fund] = {"p_after_trade": r(ev_hits / max(ev_n, 1)), "n_trade_stockdays": ev_n,
                         "p_baseline_no_trade_prev20": r(bs_hits / max(bs_n, 1)), "n_baseline": bs_n,
                         "by_grade_at_trade": {k: {"p_after_trade": r(v[0] / v[1]) if v[1] else None, "n_trade": v[1],
                                                   "p_baseline": r(v[2] / v[3]) if v[3] else None, "n_baseline": v[3]}
                                               for k, v in sorted(strat.items())}}
        after[label] = res
    out["after_trade"] = after
    # 매매 시점 등급 분포(매니저 평가일 기준)
    gd = {}
    for side in ("buy", "sell"):
        for fund in ("TIME", "KoAct"):
            c = Counter()
            for e in events:
                if e["side"] != side or e["fund"] != fund:
                    continue
                v = g[cal.get_loc(pd.Timestamp(e["mgr_us_day"])), j_of[e["sym"]]]
                c[NAMES[int(v)] if np.isfinite(v) else "등급없음"] += 1
            gd[f"{fund}:{side}"] = dict(c)
    out["grade_at_trade"] = gd
    # 펀드 매매의 달력 몰림(KRX 세션 기준). 세션 = 두 펀드 스냅샷 날짜 합집합(첫 날은 비교 대상 없음)
    ses = pd.DatetimeIndex(sorted({pd.Timestamp(e["krx_session"]) for e in events} |
                                  {pd.Timestamp(s) for s in sessions}))
    ses = ses[(ses >= pd.Timestamp(min(e["krx_session"] for e in events)))]
    ses = ses[ses <= pd.Timestamp(max(e["krx_session"] for e in events))]
    sl = calendar_labels(ses)
    cal_res = {}
    for fund in ("either", "TIME", "KoAct"):
        cnt = np.zeros(len(ses))
        pp = np.zeros(len(ses))
        for e in events:
            if fund != "either" and e["fund"] != fund:
                continue
            i = ses.get_loc(pd.Timestamp(e["krx_session"]))
            cnt[i] += 1
            pp[i] += abs(e["trade_pp"] or 0)
        ones = np.ones(len(ses))
        res = {}
        te = sl["tdom_end"].to_numpy()
        td = sl["tdom"].to_numpy()
        qk = sl["qk"].to_numpy()
        for name, m in (("month_last5", np.isfinite(te) & (te <= 5)), ("month_last3", np.isfinite(te) & (te <= 3)),
                        ("month_first5", td <= 5), ("q_boundary_pm5", (qk >= QB_LO) & (qk <= QB_HI)),
                        ("q_end_last5", (qk >= -5) & (qk <= -1))):
            st = hr_stats(cnt, ones, m)
            st_pp = hr_stats(pp, ones, m)
            res[name] = {"trade_share": st["event_share_in"], "session_share": st["day_share_in"],
                         "trades_per_session_ratio": st["hazard_ratio"],
                         "abs_pp_share": st_pp["event_share_in"],
                         "p_circular_perm_count": circ_perm_hr(cnt, ones, m)["p_more"],
                         "p_circular_perm_abs_pp": circ_perm_hr(pp, ones, m)["p_more"]}
        res["tdom_1_23_trades"] = [int(cnt[td == i].sum()) for i in range(1, 24)]
        res["tdom_1_23_sessions"] = [int((td == i).sum()) for i in range(1, 24)]
        top = np.argsort(-cnt)[:8]
        res["busiest_sessions"] = [{"krx_session": ses[i].strftime("%Y-%m-%d"), "trades": int(cnt[i]),
                                    "abs_pp": r(pp[i], 2)} for i in top]
        res["n_sessions"] = int(len(ses))
        res["n_trades"] = int(cnt.sum())
        cal_res[fund] = res
    out["calendar"] = cal_res
    return out


# ------------------------------------------------------------------ 분기 경계 사례
def boundary_cases(T, cal_sel, keep, G) -> list[dict]:
    """분기 경계마다 창(−5..+4) 안 S 이탈·진입 수와 그 밖 같은 길이 평균."""
    lab = calendar_labels(G.index).iloc[T["pos"]]
    qk = lab["qk"].to_numpy()
    ex = (T["exit"].sum(axis=1)).astype(float)
    en = (T["entry"].sum(axis=1)).astype(float)
    re = (T["risk_exit"].sum(axis=1)).astype(float)
    out = []
    starts = [i for i in range(len(cal_sel)) if qk[i] == 0]
    for i0 in starts:
        a, b = max(i0 + QB_LO, 0), min(i0 + QB_HI, len(cal_sel) - 1)
        m = np.zeros(len(cal_sel), bool)
        m[a:b + 1] = True
        m &= keep
        out.append({"quarter_start": cal_sel[i0].strftime("%Y-%m-%d"), "days": int(m.sum()),
                    "S_exits": int(ex[m].sum()), "S_entries": int(en[m].sum()),
                    "S_risk": int(re[m].sum()),
                    "exit_hazard_pct": r(ex[m].sum() / max(re[m].sum(), 1) * 100, 2)})
    return out


def leave_one_out(T, keep, lab_sel) -> dict:
    qk = lab_sel["qk"].to_numpy()
    ex = T["exit"].sum(axis=1).astype(float)
    ar = T["risk_exit"].sum(axis=1).astype(float)
    en = T["entry"].sum(axis=1).astype(float)
    an = T["risk_entry"].sum(axis=1).astype(float)
    starts = np.flatnonzero(qk == 0)
    m_all = (qk >= QB_LO) & (qk <= QB_HI)
    res = {}
    for i0 in starts:
        drop = np.zeros(len(qk), bool)
        drop[max(i0 + QB_LO, 0):i0 + QB_HI + 1] = True
        k = keep & ~drop
        res[lab_sel.index[i0].strftime("%Y-%m-%d")] = {
            "exit_hr_without": hr_stats(ex[k], ar[k], m_all[k])["hazard_ratio"],
            "entry_hr_without": hr_stats(en[k], an[k], m_all[k])["hazard_ratio"]}
    return res


def main():
    t0 = time.time()
    out = {"spec": {
        "grade": "FINAL = fam_composite.make(GRID['E14'],10) → fam_ladder.hyb_grades(·, P, harness.universe_mask(P))",
        "sample": f"전환일 {START}~{END}, 2026-01-02~01-15 제외(종목군 145→약 60 전환)",
        "universe": "2026-01-02부터 그날 보유 미국 종목(두 펀드 합집합), 그 전은 패널 150종목(봉 130개 이상)",
        "q_boundary_window": f"분기 첫 거래일 기준 {QB_LO}..+{QB_HI} 거래일(앞 분기 마지막 5일 + 새 분기 첫 5일)",
        "earnings_window": f"반응일 기준 {EW_LO}..+{EW_HI} 거래일. 반응일 = 16:00 이후 발표면 다음 거래일, 그 전이면 그날",
        "fund_trade_day": "mgr_us_day = gbrc.etf.market.price_reference_date(PDF D) 이하 마지막 미국 거래일(대개 미국 D−2); pub_us_day = PDF D 이후 첫 미국 거래일",
        "permutations": NPERM, "seed": SEED}}
    results = {}
    for label, smooth in (("final_s10", 10), ("raw_s0", 0)):
        G = grade_codes(smooth)
        cal = G.index
        cal_sel = cal[(cal >= pd.Timestamp(START)) & (cal <= pd.Timestamp(END))]
        T = transitions(G, cal_sel)
        keep = ~((cal_sel >= pd.Timestamp(SWITCH[0])) & (cal_sel <= pd.Timestamp(SWITCH[1])))
        lab_all = calendar_labels(cal)
        lab_sel = lab_all.iloc[T["pos"]]
        pre = keep & (cal_sel < pd.Timestamp("2026-01-01"))
        post = keep & (cal_sel >= pd.Timestamp("2026-01-01"))
        res = {"counts": {
            "days": int(keep.sum()), "S_exits": int(T["exit"][keep].sum()), "S_entries": int(T["entry"][keep].sum()),
            "S_risk_stockdays": int(T["risk_exit"][keep].sum()),
            "S_left_universe": int(T["leave_S"][keep].sum()),
            "S_per_day_mean_pre2026": r(T["risk_exit"][pre].sum(axis=1).mean(), 1),
            "S_per_day_mean_2026": r(T["risk_exit"][post].sum(axis=1).mean(), 1),
            "exit_hazard_pct_all": r(T["exit"][keep].sum() / T["risk_exit"][keep].sum() * 100, 3)}}
        res["calendar"] = {"all": calendar_tests(T, lab_all, keep)}
        if label == "final_s10":
            res["calendar"]["pre2026"] = calendar_tests(T, lab_all, pre)
            res["calendar"]["2026"] = calendar_tests(T, lab_all, post)
            res["boundary_cases"] = boundary_cases(T, cal_sel, keep, G)
            res["leave_one_boundary_out"] = leave_one_out(T, keep, lab_sel)
            res["exit_shape"] = exit_shape(T, G, keep, cal_sel)
            res["persistence"] = persistence(G, cal_sel)
            res["monthly"] = monthly(G)
            print("calendar/monthly done", round(time.time() - t0, 1), flush=True)
            E = earnings_analysis(G, T, keep, cal_sel, lab_all)
            W = E.pop("_W")[T["pos"]]
            E.pop("_react")
            # 민감도: 실적 창(−5..+10) 밖 종목-날만으로 분기 경계 위험비
            qk = lab_sel["qk"].to_numpy()[keep]
            mq = (qk >= QB_LO) & (qk <= QB_HI)
            sens = {}
            for kind, risk in (("exit", "risk_exit"), ("entry", "risk_entry")):
                ev = (T[kind] & ~W).sum(axis=1).astype(float)[keep]
                ar = (T[risk] & ~W).sum(axis=1).astype(float)[keep]
                sens[kind] = {**hr_stats(ev, ar, mq), "perm": circ_perm_hr(ev, ar, mq)}
            res["calendar_outside_earnings_windows"] = sens
            res["earnings"] = E
            print("earnings done", round(time.time() - t0, 1), flush=True)
            res["fund"] = fund_analysis(G, T, keep, cal_sel, lab_all)
            print("fund done", round(time.time() - t0, 1), flush=True)
        else:
            res["persistence"] = persistence(G, cal_sel)
            res["monthly"] = monthly(G)
        results[label] = res
    out["results"] = results
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("saved", OUT, round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
