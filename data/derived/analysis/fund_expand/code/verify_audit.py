"""적대적 버그 점검 (2026-10-12): compare / lag_check / lag_check2의 월말·매일 비교가 코드 결함 때문에 틀렸는가.

점검 항목
  1 monkeypatch: FC·FL이 같은 harness 모듈을 보는가, 점수가 바뀐 범위(um) 안에서만 값이 있는가
  2 매일 보유(H.membership)를 스냅샷에서 독립 재계산해 같은가, 패널에 없어 빠진 보유 종목
  3 월말·주간 앵커: 앵커가 미국 거래일인가, 행이 앵커 날의 매일 보유와 같은가, 미래 스냅샷을 쓰는가
  4 결정일(직전 거래일) 보유가 매일 기준과 같은 비율(주간·격주·월말)
  5 U1(로봇)이 2026-04-30 전에 끼어드는가 (compare 방식 vs lag_check2 방식)
  6 2025 보조 처리(aux 합집합)가 2026 등급으로 새는가, U4 결과 크기
  7 빠진 열(가격 없음·봉 부족)
  8 기준 숫자 재현(U0 매일 58.4, U4 77.9, 월말 U0 83.6·U4 81.7·U1 105.4)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as X  # noqa: E402
import lag_check2 as L2  # noqa: E402

H, FC, FL = X.H, X.FC, X.FL
RULES = ("S", "SA", "BUF", "TOP7", "ALL")
T0 = time.time()

# 같은 가격에서 재료 계산은 범위와 무관하므로 한 번만 계산(결과 동일, 시간·메모리 절약)
_raw_orig = FC.raw_ingredients
_raw_cache = {}


def _raw_cached(P, need):
    key = (id(P["C"]), frozenset(need))
    if key not in _raw_cache:
        _raw_cache[key] = _raw_orig(P, need)
    return _raw_cache[key]


FC.raw_ingredients = _raw_cached


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def grades_sc(P, mask):
    um = mask & (P["C"].notna().cumsum() >= H.MIN_BARS) & P["C"].notna()
    orig = H.universe_mask
    H.universe_mask = lambda _P: um
    try:
        sc = FC.make(FC.GRID["E14"], 10)(P)
    finally:
        H.universe_mask = orig
    return sc, FL.hyb_grades(sc, P, um), um


def y26(P, G, um, wins=("front", "back", "y2026")):
    out = {}
    for w in wins:
        a, b = X.WINDOWS[w]
        out[w] = {}
        for r in RULES:
            x = X.weekly(P, G, um, a, b, r)
            out[w][r] = x["ret"]
            if w == "y2026":
                out[w][r + "_picks"] = x["picks"]
    return out


def decision_dates(cal, a, b):
    d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))]
    wk = pd.Series(d.to_period("W-SUN"), index=d)
    firsts = [x for x, w in zip(d, wk) if x == wk[wk == w].index[0]]
    out = []
    for i in range(len(firsts) - 1):
        prev = cal[cal.get_loc(firsts[i]) - 1]
        if prev >= pd.Timestamp(a):
            out.append(prev)
    return out


def jacc(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    res = {}
    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    cal, cols = P["C"].index, P["C"].columns
    rows26 = cal >= pd.Timestamp("2026-01-02")
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    log("panel", P["C"].shape)

    # ---- 1 모듈 동일성
    res["1_module_identity"] = {"FC.H_is_H": FC.H is H, "FL.H_is_H": FL.H is H,
                                "composite_calls_H.universe_mask": "H.universe_mask(P)" in open(FC.__file__, encoding="utf-8").read()}

    # ---- 2 매일 보유 독립 재계산
    snaps = H._snapshot_members()
    sdays = sorted(snaps)
    mism, dropped = 0, {}
    base_panel_cols = set(H.load_panel()["C"].columns)
    for d in cal[rows26]:
        k = max(s for s in sdays if pd.Timestamp(s) <= d)
        exp = {s for s in snaps[k] if s in base_panel_cols}
        got = set(cols[base.loc[d].to_numpy()])
        if exp != got:
            mism += 1
    for k in sdays:
        for s in snaps[k] - base_panel_cols:
            dropped[s] = dropped.get(s, 0) + 1
    in_ext = {s: n for s, n in dropped.items() if s in cols}
    res["2_base_membership"] = {"days_mismatch_vs_independent": mism, "snapshot_days": len(sdays),
                                "held_but_not_in_150_panel(snap-days)": dict(sorted(dropped.items(), key=lambda x: -x[1])),
                                "of_which_in_extended_panel(U0 excludes, U4 can add)": in_ext}
    log("2", res["2_base_membership"]["days_mismatch_vs_independent"], "dropped", len(dropped), "in_ext", in_ext)

    # ---- 3 앵커
    month_snaps = sorted(extra["bio"])
    anchors_in_cal = {s: pd.Timestamp(s) in cal for s in month_snaps}
    anchors_have_base_snap = {s: s in snaps for s in month_snaps}
    mon = L2.snapped(base, cal, month_snaps)
    ms = [pd.Timestamp(s) for s in month_snaps]
    bad_row, future = 0, 0
    first_seen = {}
    for d in cal[rows26]:
        for s in cols[base.loc[d].to_numpy()]:
            first_seen.setdefault(s, d)
    for d in cal[rows26]:
        a = max(x for x in ms if x <= d)
        if not np.array_equal(mon.loc[d].to_numpy(), base.loc[a].to_numpy()):
            bad_row += 1
        for s in cols[mon.loc[d].to_numpy()]:
            if s in first_seen and first_seen[s] > d:
                future += 1
    # 추가 펀드 쪽도 같은 날부터인가(compare.membership·with_extras)
    m4 = L2.with_extras(mon, ["rob", "bio", "dem"], extra, cols, cal)
    early_extra = 0
    for f in ("rob", "bio", "dem"):
        f0 = pd.Timestamp(min(extra[f]))
        only = sorted(set().union(*extra[f].values()) & set(cols))
        pre = m4.loc[(cal < f0) & rows26, only] & ~mon.loc[(cal < f0) & rows26, only]
        early_extra += int(pre.to_numpy().sum())
    res["3_anchoring"] = {"month_anchors": month_snaps, "all_anchors_US_trading_days": all(anchors_in_cal.values()),
                          "anchors_with_base_snapshot": anchors_have_base_snap,
                          "rows_not_equal_base_at_anchor": bad_row, "member_cells_seen_before_first_daily_appearance": future,
                          "extra_cells_set_before_fund_first_pdf": early_extra}
    log("3", res["3_anchoring"])

    # ---- 4 결정일 보유 일치율
    dd = decision_dates(cal, *X.WINDOWS["y2026"][:2])
    d26 = cal[rows26]
    wk = pd.Series(d26.to_period("W-SUN"), index=d26)
    week_last = [g.index[-1] for _, g in wk.groupby(wk)]
    cad = {"weekly": L2.snapped(base, cal, [d26[0]] + week_last),
           "biweekly": L2.snapped(base, cal, [d26[0]] + week_last[1::2]), "monthly": mon}
    res["4_decision_day_match"] = {}
    for k, m in cad.items():
        same = sum(bool(np.array_equal(m.loc[p].to_numpy(), base.loc[p].to_numpy())) for p in dd)
        jj = [jacc(set(cols[m.loc[p].to_numpy()]), set(cols[base.loc[p].to_numpy()])) for p in dd]
        res["4_decision_day_match"][k] = {"decision_days": len(dd), "identical_to_daily": same,
                                          "mean_jaccard": round(float(np.mean(jj)), 3), "min_jaccard": round(float(np.min(jj)), 3)}
    log("4", res["4_decision_day_match"])

    # ---- 7 빠진 열
    missing_px = sorted(allx - set(cols))
    held_days_missing = {}
    for f, v in extra.items():
        for d, syms in v.items():
            for s in syms - set(cols):
                held_days_missing[s] = held_days_missing.get(s, 0) + 1
    bars = P["C"].notna().cumsum()
    never_um = {}
    for s in sorted(allx & set(cols)):
        b = bars.loc[rows26, s]
        if b.max() < H.MIN_BARS:
            never_um[s] = int(b.max())
    res["7_missing_columns"] = {"no_price": missing_px, "no_price_pdf_count": held_days_missing,
                                "never_reach_MIN_BARS_in_2026": never_um}
    log("7", res["7_missing_columns"])

    # ---- 8/1/5/6: 등급 실행(순서대로, 하나씩)
    runs = {}
    aux_base = X.membership(P, [], extra, aux_all=True)

    def run(name, mask, keep_sc=False):
        sc, G, um = grades_sc(P, mask)
        r = y26(P, G, um)
        info = {"y2026": {k: r["y2026"][k] for k in RULES}, "front": r["front"], "back": r["back"]}
        runs[name] = (G, um, r)
        log(name, info["y2026"], "front S", r["front"]["S"], "back S", r["back"]["S"])
        if keep_sc:
            return sc, G, um, info
        del sc
        return None, G, um, info

    out8 = {}
    # compare 방식 U0·U1·U4 (매일 기준 + 월말 추가, aux 합집합에 추가 펀드 포함)
    sc4, G4, um4, out8["cmp_U4"] = run("cmp_U4", X.membership(P, X.VARIANTS["U4"], extra, aux_all=True), keep_sc=True)
    # 1 monkeypatch 효과: 점수 값이 um 밖에 없어야 하고, 추가 펀드 전용 종목에도 점수가 있어야 한다
    hmask = H.universe_mask(P)  # 원래 harness 범위(150 패널 보유)
    hmask = hmask.reindex(index=cal, columns=cols).fillna(False).astype(bool)
    only_extra = [c for c in cols if c not in base_panel_cols]
    res["1_monkeypatch_effect"] = {
        "score_cells_outside_um": int((sc4.notna() & ~um4).to_numpy().sum()),
        "score_cells_outside_original_harness_mask_2026": int((sc4.notna() & ~hmask).loc[rows26].to_numpy().sum()),
        "extra_only_names_with_score_2026": int(sc4.loc[rows26, only_extra].notna().any().sum()),
        "grade_cells_outside_um": int((G4.notna() & ~um4).to_numpy().sum()),
    }
    log("1", res["1_monkeypatch_effect"])
    del sc4
    _, G0, um0, out8["cmp_U0"] = run("cmp_U0", X.membership(P, [], extra, aux_all=True))
    _, G1, um1, out8["cmp_U1"] = run("cmp_U1", X.membership(P, X.VARIANTS["U1"], extra, aux_all=True))

    def first_diff(Ga, Gb, uma, umb, lo="2026-01-02", hi="2026-04-29"):
        sl = slice(lo, hi)
        a = Ga.loc[sl].where(Ga.loc[sl].notna(), "-").to_numpy()
        b = Gb.loc[sl].where(Gb.loc[sl].notna(), "-").to_numpy()
        diff = a != b
        days = Ga.loc[sl].index[diff.any(axis=1)]
        mdiff = (uma.loc[sl] != umb.loc[sl]).any(axis=1)
        return {"grade_cells_diff": int(diff.sum()), "days_with_diff": len(days),
                "first": str(days[0].date()) if len(days) else None, "last": str(days[-1].date()) if len(days) else None,
                "mask_days_diff": int(mdiff.sum())}

    def picks_diff(ra, rb, before="2026-04-30"):
        # y2026 주별 picks 비교(결정이 before 이전인 주)
        dd_ = decision_dates(cal, *X.WINDOWS["y2026"][:2])
        out = {}
        for r in RULES:
            pa, pb = ra["y2026"][r + "_picks"], rb["y2026"][r + "_picks"]
            n = sum(1 for i, p in enumerate(dd_) if p < pd.Timestamp(before) and i < len(pa) and i < len(pb)
                    and set(pa[i]) != set(pb[i]))
            out[r] = n
        return out

    res["5_U1_start"] = {"compare_style_U0_vs_U1_before_0430": first_diff(G0, G1, um0, um1),
                         "compare_style_weeks_with_diff_picks_before_0430": picks_diff(runs["cmp_U0"][2], runs["cmp_U1"][2])}
    log("5a", res["5_U1_start"])
    del G1, um1
    runs.pop("cmp_U1")

    # 6 aux 누수 크기: U4 매일 기준, 2025 범위를 U0와 같게(추가 펀드 없음)
    m = aux_base.copy()
    m.loc[rows26] = X.membership(P, X.VARIANTS["U4"], extra, aux_all=True).loc[rows26]
    _, G4b, um4b, out8["U4_daily_aux_base"] = run("U4_daily_aux_base", m)
    res["6_aux_leak_U4"] = {"cmp_U4(aux with extras)": out8["cmp_U4"], "U4(aux = base only)": out8["U4_daily_aux_base"],
                            "grade_diff_2026": first_diff(G4, G4b, um4, um4b, hi="2026-10-09"),
                            "weeks_diff_picks_2026": picks_diff(runs["cmp_U4"][2], runs["U4_daily_aux_base"][2], before="2027-01-01")}
    log("6", res["6_aux_leak_U4"]["grade_diff_2026"], res["6_aux_leak_U4"]["weeks_diff_picks_2026"])
    for k in ("cmp_U4", "U4_daily_aux_base"):
        runs.pop(k)
    del G4, um4, G4b, um4b

    # lag_check2 방식 월말 U0·U1·U4
    for name, funds in (("mon_U0", []), ("mon_U1", ["rob"]), ("mon_U4", ["rob", "bio", "dem"])):
        mm = aux_base.copy()
        mm.loc[rows26] = L2.with_extras(mon, funds, extra, cols, cal).loc[rows26]
        _, Gx, umx, out8[name] = run(name, mm)
    res["5_U1_start"]["lag_check2_style_U0_vs_U1_before_0430"] = first_diff(runs["mon_U0"][0], runs["mon_U1"][0],
                                                                            runs["mon_U0"][1], runs["mon_U1"][1])
    res["5_U1_start"]["lag_check2_style_weeks_with_diff_picks_before_0430"] = picks_diff(runs["mon_U0"][2], runs["mon_U1"][2])
    log("5b", res["5_U1_start"])
    res["8_reproduction"] = out8
    (HERE.parent / "results").mkdir(exist_ok=True)
    (HERE.parent / "results" / "verify_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                               encoding="utf-8")
    log("saved")


if __name__ == "__main__":
    main()
