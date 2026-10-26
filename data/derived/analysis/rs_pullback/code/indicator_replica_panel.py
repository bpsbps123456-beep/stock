"""indicator_replica B: 지표 근사(RS Main·Latent_H·FadeGap)로 만든 사전 등록 순위 8개를 공통 채점기로 잰다.

정의 (t일 종가까지 아는 값만)
    ex4      = 100×(C/C[4] − QQQ/QQQ[4])                  (RS Main)
    Latent_H = 최근 5봉 최대 100×(H/C[4] − QQQ_H/QQQ[4])   (고가 기준 4일 초과수익의 최근 고점)
    FadeGap  = max(0, Latent_H − ex4)
순위 8개(사전 등록): ex4, −ex4, Latent_H, −FadeGap, Latent_H − FadeGap, ex4|FadeGap<=1,
    Watch(Latent_H 상위 30% & FadeGap 상위 1/3, 점수 Latent_H), Strong(ex4 상위 30% & FadeGap<=1, 점수 ex4).
백분위는 harness.universe_mask(P) 안 단면 순위(t일까지 아는 종목군).

보조(선택에 쓰지 않음, 설명용): ex4의 보유기간 1·2일 IC, 2026 월별 IC, 강한 종목(ex60 상위 30%) 안 ex4 IC,
    부분집합 순위(6~8)의 '집합 초과'(집합 평균 − 그날 eligible 평균), 쏠림 점검(SNDK 빼기, 7월 빼기, 극단 5일 빼기).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "trend_grade", "code"))
import harness  # noqa: E402

RES = os.path.join(HERE, "..", "results")
TOP = 0.70
WINS = ("front", "back", "y2025", "y2024", "late")


def parts(P: dict) -> dict:
    C, H = P["C"], P["H"]
    Q, QH = P["bench"]["C"], P["bench"]["H"]
    ex4 = 100 * (C / C.shift(4)).sub(Q / Q.shift(4), axis=0)
    lat_raw = 100 * (H / C.shift(4)).sub(QH / Q.shift(4), axis=0)
    lat = lat_raw.rolling(5, min_periods=5).max()
    fg = (lat - ex4).clip(lower=0)
    um = harness.universe_mask(P)
    pr = lambda x: x.where(um).rank(axis=1, pct=True, method="average")
    return {"ex4": ex4, "lat": lat, "fg": fg, "um": um, "pr": pr}


def r_ex4(P):
    return parts(P)["ex4"]


def r_neg_ex4(P):
    return -parts(P)["ex4"]


def r_lat(P):
    return parts(P)["lat"]


def r_neg_fg(P):
    return -parts(P)["fg"]


def r_lat_minus_fg(P):
    d = parts(P)
    return d["lat"] - d["fg"]


def r_ex4_nofade(P):
    d = parts(P)
    return d["ex4"].where(d["fg"] <= 1)


def r_watch(P):
    d = parts(P)
    ok = (d["pr"](d["lat"]) >= TOP) & (d["pr"](d["fg"]) >= 2 / 3)
    return d["lat"].where(ok)


def r_strong(P):
    d = parts(P)
    ok = (d["pr"](d["ex4"]) >= TOP) & (d["fg"] <= 1)
    return d["ex4"].where(ok)


RANKERS = [
    ("R1_ex4", "RS Main = ex(4), 클수록 강함", r_ex4),
    ("R2_neg_ex4", "−ex(4) (4일 단기 반전)", r_neg_ex4),
    ("R3_latent_H", "Latent_H = 5봉 최대 고가 기준 4일 초과", r_lat),
    ("R4_neg_fadegap", "−FadeGap (덜 꺾인 종목)", r_neg_fg),
    ("R5_latent_minus_fade", "Latent_H − FadeGap", r_lat_minus_fg),
    ("R6_ex4_if_fade_le1", "ex(4), FadeGap<=1인 종목만(그 외 NaN)", r_ex4_nofade),
    ("R7_watch", "Watch: Latent_H 상위30% & FadeGap 상위 1/3, 점수 Latent_H", r_watch),
    ("R8_strong", "Strong: ex(4) 상위30% & FadeGap<=1, 점수 ex4", r_strong),
]


def daily_ic(score: pd.DataFrame, h: int, dates, cols=None, extra_mask=None) -> pd.Series:
    P = harness.load_panel()
    score = score.reindex(index=P["C"].index, columns=P["C"].columns)
    mask = harness.eligible() & score.notna()
    if extra_mask is not None:
        mask &= extra_mask.reindex_like(mask).fillna(False).astype(bool)
    cols = cols or list(P["C"].columns)
    fr = harness.forward(h).loc[dates, cols].where(mask.loc[dates, cols])
    sc = score.loc[dates, cols].where(mask.loc[dates, cols])
    return harness._row_spearman(sc, fr).dropna()


def set_excess(flag: pd.DataFrame, h: int, wname: str, cols=None) -> dict:
    """flag 집합(eligible 안) 평균 수익 − 그날 eligible 전체 평균, 날짜 평균(%p)·NW t."""
    P = harness.load_panel()
    excl = harness.EXCLUDE.get(wname, set())
    cols = cols or [c for c in P["C"].columns if c not in excl]
    dates = harness.window_dates(wname, h)
    el = harness.eligible().loc[dates, cols]
    fr = harness.forward(h).loc[dates, cols].where(el)
    f = flag.reindex(index=P["C"].index, columns=P["C"].columns).fillna(False).astype(bool).loc[dates, cols] & el
    d = (fr.where(f).mean(axis=1) - fr.mean(axis=1)).dropna()
    return {"x": round(float(d.mean()) * 100, 3), "t": round(harness.newey_west_t(d, h), 2),
            "hit": round(float((d > 0).mean()) * 100, 1), "days": int(len(d)),
            "avg_n": round(float(f.sum(axis=1).loc[d.index].mean()), 1)}


def fidelity() -> dict:
    """스크린샷 프리장 촬영(완성 일봉) 행과 근사값 비교: 평균 절대 오차."""
    P = harness.load_panel()
    d = parts(P)
    t = pd.read_csv(os.path.join(RES, "indicator_shots_2026-07.csv"))
    bars = {"S5": "2026-07-06", "S1": "2026-07-08", "S2": "2026-07-09", "S4": "2026-07-13"}
    rows = []
    for _, r in t[t.shot.isin(list(bars))].iterrows():
        if r.ticker not in P["C"].columns:
            continue
        b = pd.Timestamp(bars[r.shot])
        rows.append({"rs": d["ex4"].at[b, r.ticker] - r.rs_main, "lat": d["lat"].at[b, r.ticker] - r.latent,
                     "fg": d["fg"].at[b, r.ticker] - r.fade_gap if pd.notna(r.fade_gap) else np.nan})
    e = pd.DataFrame(rows)
    return {"n": int(len(e)), "mae_rs_main": round(float(e.rs.abs().mean()), 3),
            "mae_latent": round(float(e.lat.abs().mean()), 3),
            "latent_within_0.25": round(float((e.lat.abs() <= 0.25).mean()) * 100, 1),
            "mae_fade_gap": round(float(e.fg.abs().mean()), 3)}


def main() -> None:
    P = harness.load_panel()
    out = {"family": "indicator_replica", "definitions": __doc__, "fidelity_vs_screenshots": fidelity(),
           "benchmarks": harness.benchmarks(), "rows": []}
    d = parts(P)
    scores = {}
    for name, spec, fn in RANKERS:
        sc = fn(P)
        scores[name] = sc
        nl = harness.no_lookahead(fn)
        res = harness.evaluate(sc, windows=WINS)
        row = {"name": name, "spec": spec, "no_lookahead": nl, "result": res}
        if name in ("R6_ex4_if_fade_le1", "R7_watch", "R8_strong"):
            flag = sc.notna()
            row["set_excess"] = {w: {f"h{h}": set_excess(flag, h, w) for h in (5, 10, 20)} for w in WINS}
        out["rows"].append(row)
        print(harness.summary_line(name, res), "NL", nl)
        if "set_excess" in row:
            print("   set", {w: (row["set_excess"][w]["h5"]["x"], row["set_excess"][w]["h5"]["t"],
                              row["set_excess"][w]["h20"]["x"], row["set_excess"][w]["h20"]["avg_n"]) for w in WINS})

    # ── 보조: ex4 반전/추세 진단 ─────────────────────────
    ex4 = d["ex4"]
    sup = {}
    # 1) 보유기간 프로필(IC, 날짜별 NW t)
    prof = {}
    for w in WINS:
        prof[w] = {}
        for h in (1, 2, 5, 10, 20):
            dates = harness.window_dates(w, h)
            if len(dates) < 10:
                continue
            cols = [c for c in P["C"].columns if c not in harness.EXCLUDE.get(w, set())]
            ic = daily_ic(ex4, h, dates, cols)
            prof[w][f"h{h}"] = {"ic": round(float(ic.mean()), 4), "t": round(harness.newey_west_t(ic, h), 2),
                                "pos": round(float((ic > 0).mean()) * 100, 1), "days": int(len(ic))}
    sup["ex4_ic_profile"] = prof
    # 상위/하위 30% 집합 초과(5일)
    pr4 = d["pr"](ex4)
    sup["ex4_top30_vs_bot30"] = {w: {f"h{h}": {"top30": set_excess(pr4 >= TOP, h, w),
                                               "bot30": set_excess(pr4 <= 1 - TOP, h, w)} for h in (1, 5, 20)}
                                 for w in ("front", "back", "y2025", "y2024")}
    # 2) 2026 월별 IC (h5, 신호일 기준 월)
    dates5 = harness.window_dates("front", 5).append(harness.window_dates("back", 5))
    ic5 = daily_ic(ex4, 5, dates5)
    ic1 = daily_ic(ex4, 1, harness.window_dates("front", 1).append(harness.window_dates("back", 1)))
    sup["ex4_ic_by_month_2026"] = {str(k): {"ic5": round(float(v5.mean()), 3), "n": int(len(v5)),
                                            "ic1": round(float(ic1[ic1.index.to_period('M') == k].mean()), 3)}
                                   for k, v5 in ic5.groupby(ic5.index.to_period("M"))}
    # 3) 강한 종목(ex60 상위 30%) 안에서 ex4 IC — 개념('강한 종목 중 눌린 것')의 직접 확인
    C, Q = P["C"], P["bench"]["C"]
    ex60 = 100 * (C / C.shift(60)).sub(Q / Q.shift(60), axis=0)
    strong60 = d["pr"](ex60) >= TOP
    cond = {}
    for w in ("front", "back", "y2025", "y2024"):
        cond[w] = {}
        for h in (5, 10, 20):
            dates = harness.window_dates(w, h)
            ic = daily_ic(ex4, h, dates, extra_mask=strong60)
            cond[w][f"h{h}"] = {"ic": round(float(ic.mean()), 4), "t": round(harness.newey_west_t(ic, h), 2),
                                "days": int(len(ic))}
    sup["ex4_ic_within_ex60_top30"] = cond
    # 4) 쏠림 점검: ex4 IC(h5·h20) front/back에서 SNDK 빼기, 2026-06-22~07-31 신호일 빼기, |IC| 극단 5일 빼기
    conc = {}
    for w in ("front", "back"):
        for h in (5, 20):
            dates = harness.window_dates(w, h)
            base = daily_ic(ex4, h, dates)
            nos = daily_ic(ex4, h, dates, [c for c in P["C"].columns if c != "SNDK"])
            jul = base[(base.index < pd.Timestamp("2026-06-22")) | (base.index > pd.Timestamp("2026-07-31"))]
            trim = base.drop(base.abs().sort_values().index[-5:])
            conc[f"{w}_h{h}"] = {"base": round(float(base.mean()), 4), "base_t": round(harness.newey_west_t(base, h), 2),
                                 "no_SNDK": round(float(nos.mean()), 4),
                                 "no_jun22_jul31": round(float(jul.mean()), 4) if len(jul) else None,
                                 "no_jun22_jul31_t": round(harness.newey_west_t(jul, h), 2) if len(jul) > 5 else None,
                                 "drop_5_extreme_days": round(float(trim.mean()), 4)}
    # 상위 20% 초과에서 SNDK 빼기(R2 = −ex4)
    for w in ("front", "back"):
        for h in (5, 20):
            dates = harness.window_dates(w, h)
            for drop in (None, "SNDK"):
                cols = [c for c in P["C"].columns if c != drop]
                el = harness.eligible().loc[dates, cols]
                sc = (-ex4).loc[dates, cols].where(el)
                prr = sc.rank(axis=1, pct=True)
                fr = harness.forward(h).loc[dates, cols].where(el & sc.notna())
                top = (fr.where(prr > 0.8).mean(axis=1) - fr.mean(axis=1)).mean() * 100
                conc[f"{w}_h{h}"]["negex4_top20" + ("" if drop is None else "_noSNDK")] = round(float(top), 3)
    sup["concentration_ex4"] = conc
    # 5) 비용 민감도: −ex4 상위 20%를 5일마다 전부 바꾼다고 보면 편도 0.1% → 회당 0.2%p
    sup["cost_note"] = "편도 0.1%(왕복 0.2%p)를 5일 보유마다 전량 교체로 빼면 h5 top20 초과에서 0.2%p를 뺀다."
    # 6) ex4 잔차 IC: 그날 순위에서 ex20·ex60 순위를 뺀(OLS) 나머지 — 긴 모멘텀 반전과 별개인지
    ex20 = 100 * (C / C.shift(20)).sub(Q / Q.shift(20), axis=0)
    el = harness.eligible()
    ok = el & ex4.notna() & ex20.notna() & ex60.notna()
    r4, r20, r60 = (x.where(ok).rank(axis=1, pct=True) for x in (ex4, ex20, ex60))
    resid = pd.DataFrame(np.nan, index=ex4.index, columns=ex4.columns)
    for t in ex4.index[ok.sum(axis=1) >= 10]:
        m = ok.loc[t].to_numpy()
        X = np.column_stack([np.ones(m.sum()), r20.loc[t].to_numpy()[m], r60.loc[t].to_numpy()[m]])
        y = r4.loc[t].to_numpy()[m]
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        row = np.full(len(m), np.nan)
        row[m] = y - X @ b
        resid.loc[t] = row
    rr = {}
    for w in ("front", "back", "y2025", "y2024"):
        rr[w] = {}
        for h in (1, 5, 20):
            ic = daily_ic(resid, h, harness.window_dates(w, h))
            rr[w][f"h{h}"] = {"ic": round(float(ic.mean()), 4), "t": round(harness.newey_west_t(ic, h), 2)}
    sup["ex4_resid_after_ex20_ex60_ic"] = rr
    # 7) Watch 분해: 최근 강했던 종목(Latent_H 상위30%) 안에서 꺾인 쪽(FadeGap 상위1/3) vs 안 꺾인 쪽
    L30 = d["pr"](d["lat"]) >= TOP
    F33 = d["pr"](d["fg"]) >= 2 / 3
    sets = {"L30": L30, "F33": F33, "watch=L30&F33": L30 & F33, "L30&notF33": L30 & ~F33,
            "strong=ex4top30&fg<=1": (pr4 >= TOP) & (d["fg"] <= 1)}
    dec = {}
    for w in ("front", "back", "y2025", "y2024"):
        dec[w] = {}
        for h in (1, 5, 10, 20):
            cell = {k: set_excess(v, h, w) for k, v in sets.items()}
            # 짝 차이: watch − L30&notF33 (같은 날 둘 다 있을 때)
            excl = harness.EXCLUDE.get(w, set())
            cols = [c for c in P["C"].columns if c not in excl]
            dates = harness.window_dates(w, h)
            e_ = el.loc[dates, cols]
            fr = harness.forward(h).loc[dates, cols].where(e_)
            a = fr.where((L30 & F33).loc[dates, cols]).mean(axis=1)
            b = fr.where((L30 & ~F33).loc[dates, cols]).mean(axis=1)
            dd = (a - b).dropna()
            cell["watch_minus_L30notF33"] = {"x": round(float(dd.mean()) * 100, 3),
                                             "t": round(harness.newey_west_t(dd, h), 2), "days": int(len(dd))}
            dec[w][f"h{h}"] = cell
    sup["watch_decomposition"] = dec
    # 8) 변동성 교란 점검: FadeGap<=1은 변동이 작은 종목(KO·WMT 등)에 몰릴 수 있다.
    #    20일 일간 수익 표준편차의 그날 3분위(universe_mask 안) 안에서 뺀 수익으로 집합 초과를 다시 잰다.
    vol = C.pct_change().rolling(20, min_periods=20).std()
    vpr = d["pr"](vol)
    sets2 = dict(sets)
    sets2["fg<=1"] = d["fg"] <= 1
    sets2["ex4top30"] = pr4 >= TOP
    sets2["ex4bot30"] = pr4 <= 1 - TOP
    vn_all = {}
    for nq in (3, 5):
        vt = np.ceil(vpr * nq).clip(1, nq)
        vn = {}
        for w in ("front", "back", "y2025", "y2024"):
            vn[w] = {}
            excl = harness.EXCLUDE.get(w, set())
            cols = [c for c in P["C"].columns if c not in excl]
            for h in (5, 20):
                dates = harness.window_dates(w, h)
                e_ = el.loc[dates, cols] & vt.loc[dates, cols].notna()
                fr = harness.forward(h).loc[dates, cols].where(e_)
                vtt = vt.loc[dates, cols].where(e_)
                adj = fr.copy()
                for k in range(1, nq + 1):
                    m = vtt == k
                    adj = adj.where(~m, fr.sub(fr.where(m).mean(axis=1), axis=0))
                cell = {}
                for k, v in sets2.items():
                    f = v.reindex_like(fr).fillna(False).astype(bool).loc[dates, cols] & e_
                    dd = adj.where(f).mean(axis=1).dropna()
                    cell[k] = {"x": round(float(dd.mean()) * 100, 3), "t": round(harness.newey_west_t(dd, h), 2),
                               "vol_pct": round(float(vpr.loc[dates, cols].where(f).stack().mean()), 2)}
                vn[w][f"h{h}"] = cell
        vn_all[f"q{nq}"] = vn
    vn = vn_all
    sup["vol_neutral_set_excess"] = vn
    # 9) 쏠림: Strong 집합·(Watch − 안 꺾인 강세) 차이의 back 값에서 SNDK 빼기, 6/22~7/31 신호일 빼기, 기여 큰 5일 빼기
    cc = {}
    for w in ("front", "back"):
        for h in (5, 20):
            dates = harness.window_dates(w, h)
            for name, fa, fb in (("strong", sets["strong=ex4top30&fg<=1"], None),
                                 ("watch_minus_L30notF33", L30 & F33, L30 & ~F33)):
                res_ = {}
                for drop in (None, "SNDK"):
                    cols = [c for c in P["C"].columns if c != drop]
                    e_ = el.loc[dates, cols]
                    fr = harness.forward(h).loc[dates, cols].where(e_)
                    a = fr.where(fa.loc[dates, cols] & e_).mean(axis=1)
                    b = fr.mean(axis=1) if fb is None else fr.where(fb.loc[dates, cols] & e_).mean(axis=1)
                    dd = (a - b).dropna()
                    key = "all" if drop is None else "no_SNDK"
                    res_[key] = round(float(dd.mean()) * 100, 3)
                    if drop is None:
                        res_["t"] = round(harness.newey_west_t(dd, h), 2)
                        nj = dd[(dd.index < pd.Timestamp("2026-06-22")) | (dd.index > pd.Timestamp("2026-07-31"))]
                        res_["no_jun22_jul31"] = round(float(nj.mean()) * 100, 3) if len(nj) else None
                        res_["no_jun22_jul31_t"] = round(harness.newey_west_t(nj, h), 2) if len(nj) > 5 else None
                        top5 = dd.abs().sort_values().index[-5:]
                        res_["drop_5_largest_abs_days"] = round(float(dd.drop(top5).mean()) * 100, 3)
                        res_["median"] = round(float(dd.median()) * 100, 3)
                cc[f"{name}_{w}_h{h}"] = res_
    sup["concentration_sets"] = cc
    print(json.dumps(cc, ensure_ascii=False))
    for q in vn:
        for w in vn[q]:
            for h in vn[q][w]:
                print("VN", q, w, h, {k: (v["x"], v["t"]) for k, v in vn[q][w][h].items()
                                      if k in ("strong=ex4top30&fg<=1", "fg<=1", "watch=L30&F33", "L30&notF33")})
    out["supplementary"] = sup
    print(json.dumps(rr, ensure_ascii=False))
    for w in dec:
        for h in dec[w]:
            print(w, h, {k: (v["x"], v["t"]) for k, v in dec[w][h].items()})
    print(json.dumps(sup["ex4_ic_profile"], ensure_ascii=False))
    print(json.dumps(sup["ex4_ic_by_month_2026"], ensure_ascii=False))
    print(json.dumps(sup["ex4_ic_within_ex60_top30"], ensure_ascii=False))
    print(json.dumps(sup["concentration_ex4"], ensure_ascii=False))
    print(json.dumps(sup["ex4_top30_vs_bot30"], ensure_ascii=False))

    shots = os.path.join(RES, "indicator_replica_shots.json")
    if os.path.exists(shots):
        a = json.load(open(shots, encoding="utf-8"))
        out["part_A_shots"] = {k: a[k] for k in a if k != "rows"}
    path = os.path.join(RES, "indicator_replica.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(path, json.dumps(out["fidelity_vs_screenshots"]))


if __name__ == "__main__":
    main()
