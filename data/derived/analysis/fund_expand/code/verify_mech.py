"""메커니즘 검증 (2026-10-12): 왜 U0 S가 매일 보유 58.4 → 월말 보유 83.6으로 오르고, 왜 월말 조건에서 U1(+로봇)이
U0보다 높은가(105.4 vs 83.6). 2026 전체(y2026) S 규칙, 매주 선택을 다시 뽑아 종목별 기여를 나눈다.

기여 분해(주마다 정확):
  rA - rB = (1/nA) * [ sum_{A only} (r_s - rB) - sum_{B only} (r_s - rB) ]
  -> A에만 있는 종목은 B 포트폴리오보다 잘 오른 만큼, B에만 있는 종목은 B 포트폴리오보다 덜 오른 만큼 A를 돕는다.
  최종 수익률 차(%p)로 바꿀 때는 'A 경로에서 그 주만 B로 바꿨다면' 분해(정확, 잔차 0):
  eqA_T - eqB_T = sum_t eqB_t * (rA_t - rB_t) * prod_{u>t}(1 + rA_u)
  단순판(과제 정의): A에만 있는 종목 r_s/nA, B에만 있는 종목 -r_s/nB, 공통 종목 비중 차 r_s(1/nA-1/nB)는 따로.
분류: 월말에만 든 종목이 결정일에 매일 보유에 없으면 '스냅샷 뒤 펀드가 판 종목(낡은 보유)', 매일에만 든 종목이
결정일에 월말 범위에 없으면 '스냅샷 뒤 새로 산 종목(아직 안 들어옴)', 둘 다 범위에 있으면 '등급 차(순위 모집단·점수 이력)'.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as X  # noqa: E402
import lag_check2 as L2  # noqa: E402

H = X.H
WIN = "y2026"


def weekly_detail(P, G, um, a, b, rule="S"):
    """compare.weekly의 S 경로를 그대로 따라가며 주별 (w0, w1, prev, names)를 남긴다."""
    O, C = P["O"], P["C"]
    cal = C.index
    d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))]
    wk = pd.Series(d.to_period("W-SUN"), index=d)
    firsts = [x for x, w in zip(d, wk) if x == wk[wk == w].index[0]]
    eq, rows = 1.0, []
    for i in range(len(firsts) - 1):
        w0, w1 = firsts[i], firsts[i + 1]
        prev = cal[cal.get_loc(w0) - 1]
        if prev < pd.Timestamp(a):
            continue
        g, ok = G.loc[prev], um.loc[prev]
        new = {s for s in g.index if ok[s] and g[s] == "S"}
        names = sorted(s for s in new if np.isfinite(O.at[w0, s]) and np.isfinite(O.at[w1, s]))
        r = {s: float(O.at[w1, s] / O.at[w0, s] - 1) for s in names}
        rp = float(np.mean(list(r.values()))) if names else 0.0
        rows.append({"w0": w0, "w1": w1, "prev": prev, "names": names, "r": r, "rp": rp})
        if names:
            eq *= 1 + rp
    return rows, round((eq - 1) * 100, 1)


def attribute(A, B, tagA, tagB, info):
    """A 대비 B. info(side, s, prev) -> 분류 문자열."""
    wa = {x["w0"]: x for x in A}
    wb = {x["w0"]: x for x in B}
    weeks = sorted(set(wa) | set(wb))
    # 주별 수익
    rA = [wa[w]["rp"] if w in wa and wa[w]["names"] else 0.0 for w in weeks]
    rB = [wb[w]["rp"] if w in wb and wb[w]["names"] else 0.0 for w in weeks]
    eqB = np.cumprod([1.0] + [1 + x for x in rB])[:-1]
    growA_after = np.cumprod(([1 + x for x in rA][::-1]))[::-1]
    growA_after = np.append(growA_after[1:], 1.0)
    scale = eqB * growA_after  # 그 주 1의 차이가 최종 자산에서 갖는 크기
    per = defaultdict(lambda: {"exact_pp": 0.0, "weekly_sum_pp": 0.0, "simple_pp": 0.0, "weeks": [], "ret": [], "why": defaultdict(int)})
    dilution = 0.0
    wk_gap = []
    for k, w in enumerate(weeks):
        a = wa.get(w, {"names": [], "r": {}, "prev": None})
        b = wb.get(w, {"names": [], "r": {}, "prev": None})
        sa, sb = set(a["names"]), set(b["names"])
        na, nb = len(sa), len(sb)
        prev = a["prev"] or b["prev"]
        ra_, rb_ = rA[k], rB[k]
        wk_gap.append((w, round((ra_ - rb_) * 100, 2), round(float(scale[k] * (ra_ - rb_) * 100), 2),
                       sorted(sa - sb), sorted(sb - sa)))
        for s in sa - sb:
            r = a["r"][s]
            c = (r - rb_) / na if nb else r / na
            p = per[(tagA, s)]
            p["exact_pp"] += scale[k] * c * 100
            p["weekly_sum_pp"] += c * 100
            p["simple_pp"] += r / na * 100
            p["weeks"].append(w.date().isoformat())
            p["ret"].append(round(r * 100, 1))
            p["why"][info(tagA, s, prev)] += 1
        for s in sb - sa:
            r = b["r"][s]
            c = -(r - rb_) / na if na else -r / nb
            p = per[(tagB, s)]
            p["exact_pp"] += scale[k] * c * 100
            p["weekly_sum_pp"] += c * 100
            p["simple_pp"] += -r / nb * 100
            p["weeks"].append(w.date().isoformat())
            p["ret"].append(round(r * 100, 1))
            p["why"][info(tagB, s, prev)] += 1
        for s in sa & sb:
            if na and nb:
                dilution += a["r"][s] * (1 / na - 1 / nb) * 100
    total = sum(v["exact_pp"] for v in per.values())
    retA = (np.prod([1 + x for x in rA]) - 1) * 100
    retB = (np.prod([1 + x for x in rB]) - 1) * 100
    rows = []
    for (side, s), v in per.items():
        rows.append({"side": f"{side}_only", "sym": s, "exact_pp": round(v["exact_pp"], 1),
                     "weekly_sum_pp": round(v["weekly_sum_pp"], 1), "simple_pp": round(v["simple_pp"], 1),
                     "n_weeks": len(v["weeks"]), "weeks": v["weeks"], "wk_ret_pct": v["ret"], "why": dict(v["why"])})
    rows.sort(key=lambda x: -x["exact_pp"])
    helps = [x for x in rows if x["exact_pp"] > 0]
    hurts = sorted([x for x in rows if x["exact_pp"] < 0], key=lambda x: x["exact_pp"])
    gap = retA - retB
    top = lambda k: round(sum(x["exact_pp"] for x in rows[:k]), 1)  # noqa: E731
    simple_total = sum(v["simple_pp"] for v in per.values())
    wk_gap.sort(key=lambda x: -abs(x[2]))
    return {
        "retA": round(retA, 1), "retB": round(retB, 1), "gap_pp": round(gap, 1),
        "sum_exact_pp": round(total, 1), "sum_weekly_arith_pp": round(sum(v["weekly_sum_pp"] for v in per.values()), 1),
        "simple": {"held_only_sum_pp": round(simple_total, 1), "common_weight_diff_pp": round(dilution, 1)},
        "n_names_differing": len(rows),
        "top1_pp": top(1), "top3_pp": top(3), "top5_pp": top(5),
        "top3_share_of_gap": round(top(3) / gap, 2) if gap else None,
        "gross_positive_pp": round(sum(x["exact_pp"] for x in helps), 1),
        "gross_negative_pp": round(sum(x["exact_pp"] for x in hurts), 1),
        "top5_helping": helps[:5], "top5_hurting": hurts[:5],
        "top_weeks": [{"w0": w.date().isoformat(), "wk_gap_pp": g, "final_pp": f, "A_only": ao, "B_only": bo}
                      for w, g, f, ao, bo in wk_gap[:8]],
        "all": rows,
    }


def main():
    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    cal, cols = P["C"].index, P["C"].columns
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    rows26 = cal >= pd.Timestamp("2026-01-02")
    aux = X.membership(P, [], extra, aux_all=True)
    snaps = sorted(extra["bio"])
    snap_ts = [pd.Timestamp(s) for s in snaps]
    mon = L2.snapped(base, cal, snaps)
    a, b = X.WINDOWS[WIN]

    masks = {}
    masks["U0_daily"] = aux.copy()
    m = aux.copy(); m.loc[rows26] = mon.loc[rows26]; masks["U0_monthly"] = m
    m = aux.copy(); m.loc[rows26] = L2.with_extras(mon, ["rob"], extra, cols, cal).loc[rows26]; masks["U1_monthly"] = m

    res, det, GG, UM = {"check": {}}, {}, {}, {}
    for name, mk in masks.items():  # 하나씩
        G, um = X.grades(P, mk)
        rows, ret = weekly_detail(P, G, um, a, b)
        ref = X.weekly(P, G, um, a, b, "S")
        same = ref["picks"] == [r["names"] for r in rows if r["names"]] or \
            [sorted(x) for x in ref["picks"]] == [r["names"] for r in rows if r["names"]]
        res["check"][name] = {"ret_mine": ret, "ret_compare_weekly": ref["ret"], "picks_identical": bool(same),
                              "weeks": ref["weeks"], "avg_n": ref["avg_n"]}
        print(name, res["check"][name], flush=True)
        det[name], GG[name], UM[name] = rows, G, um

    def first_after(sym, start, val):
        s = base.loc[base.index > start, sym]
        hit = s[s == val]
        return hit.index[0].date().isoformat() if len(hit) else None

    def anchor_of(d):
        return max(x for x in snap_ts if x <= d)

    # (1) 매일 → 월말
    def info1(side, s, prev):
        anc = anchor_of(prev)
        if side == "monthly":  # 월말에만 S
            if not base.at[prev, s]:
                return f"stale_sold(snap {anc.date()} held, sold {first_after(s, anc, False)})"
            if not UM["U0_daily"].at[prev, s]:
                return "in_daily_base_but_not_universe"
            return f"grade_diff(daily={GG['U0_daily'].at[prev, s]})"
        # 매일에만 S
        if not mon.at[prev, s]:
            return f"new_buy(bought {first_after(s, anc, True)}, after snap {anc.date()})"
        if not UM["U0_monthly"].at[prev, s]:
            return "in_monthly_base_but_not_universe"
        return f"grade_diff(monthly={GG['U0_monthly'].at[prev, s]})"

    res["daily_to_monthly"] = attribute(det["U0_monthly"], det["U0_daily"], "monthly", "daily", info1)

    # (2) 월말 U0 → 월말 U1
    rob_all = set().union(*extra["rob"].values())

    def info2(side, s, prev):
        if side == "U1":
            if not UM["U0_monthly"].at[prev, s]:
                return "robot_fund_name" if s in rob_all else "not_in_U0_universe"
            return f"grade_diff(U0={GG['U0_monthly'].at[prev, s]})"
        return f"displaced(U1={GG['U1_monthly'].at[prev, s]})"

    res["U0m_to_U1m"] = attribute(det["U1_monthly"], det["U0_monthly"], "U1", "U0", info2)

    # 요약: 사유 묶음별 기여
    for key in ("daily_to_monthly", "U0m_to_U1m"):
        agg = defaultdict(float)
        for x in res[key]["all"]:
            tot = sum(x["why"].values())
            for w, n in x["why"].items():
                tag = x["side"] + ":" + w.split("(")[0]
                agg[tag] += x["exact_pp"] * n / tot
        res[key]["by_reason_pp"] = {k: round(v, 1) for k, v in sorted(agg.items(), key=lambda kv: -abs(kv[1]))}
        res[key].pop("all")

    out = HERE.parent / "results" / "verify_mech.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for key in ("daily_to_monthly", "U0m_to_U1m"):
        r = res[key]
        print("\n==", key, {k: r[k] for k in ("retA", "retB", "gap_pp", "sum_exact_pp", "sum_weekly_arith_pp", "simple",
                                             "n_names_differing", "top1_pp", "top3_pp", "top5_pp", "top3_share_of_gap",
                                             "gross_positive_pp", "gross_negative_pp")})
        print("by_reason", r["by_reason_pp"])
        for lab in ("top5_helping", "top5_hurting"):
            print(" ", lab)
            for x in r[lab]:
                print("   ", x["side"], x["sym"], x["exact_pp"], "wkarith", x["weekly_sum_pp"], "simple", x["simple_pp"],
                      "n", x["n_weeks"], x["weeks"], x["wk_ret_pct"], x["why"])
        print("  top weeks")
        for x in r["top_weeks"]:
            print("   ", x)


if __name__ == "__main__":
    main()
