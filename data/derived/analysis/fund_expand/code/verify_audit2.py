"""적대적 점검 2 (2026-10-12): (a) weekly()가 다음 주 시가 있는 종목만 남기는 미래 정보 필터가 몇 번 작동하나
(b) S 수익이 스냅샷 날짜(위상)만 바꿔도 얼마나 흔들리나 — U0만, 월말 대신 월말 5·10·15거래일 전 앵커
(c) U0 매일 대 월말 S 차이가 몇 주에 몰려 있나.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_audit as V  # noqa: E402  (raw 재료 캐시·grades_sc 재사용)

X, L2, H = V.X, V.L2, V.H
RULES = V.RULES


def week_detail(P, G, um, a, b, rule):
    """X.weekly와 같은 규칙으로 주별 수익과, 시가 없음으로 빠진 종목 수를 낸다."""
    O, C = P["O"], P["C"]
    cal = C.index
    d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))]
    wk = pd.Series(d.to_period("W-SUN"), index=d)
    firsts = [x for x, w in zip(d, wk) if x == wk[wk == w].index[0]]
    mom = C / C.shift(120) - 1
    held, rows, dropped = set(), [], 0
    for i in range(len(firsts) - 1):
        w0, w1 = firsts[i], firsts[i + 1]
        prev = cal[cal.get_loc(w0) - 1]
        if prev < pd.Timestamp(a):
            continue
        g, ok = G.loc[prev], um.loc[prev]
        if rule == "S":
            new = {s for s in g.index if ok[s] and g[s] == "S"}
        elif rule == "TOP7":
            new = set(mom.loc[prev][ok].dropna().sort_values(ascending=False).index[:7])
        else:
            new = {s for s in g.index if ok[s]}
        held = new
        names = [s for s in held if np.isfinite(O.at[w0, s]) and np.isfinite(O.at[w1, s])]
        dropped += len(held) - len(names)
        if not names:
            continue
        r = float((O.loc[w1, names] / O.loc[w0, names]).mean())
        rows.append((prev, tuple(sorted(names)), r))
    return rows, dropped


def main():
    res = {}
    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    cal, cols = P["C"].index, P["C"].columns
    rows26 = cal >= pd.Timestamp("2026-01-02")
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    aux = X.membership(P, [], extra, aux_all=True)
    a, b = X.WINDOWS["y2026"]

    # (a) 미래 정보 필터
    out_a = {}
    for name, mask in (("U0_daily", aux), ("U4_daily", X.membership(P, X.VARIANTS["U4"], extra, aux_all=True))):
        _, G, um = V.grades_sc(P, mask)
        out_a[name] = {r: week_detail(P, G, um, a, b, r)[1] for r in ("S", "TOP7", "ALL")}
        if name == "U0_daily":
            G0d, um0d = G, um
    res["a_names_dropped_for_missing_open_w0_or_w1"] = out_a
    print("a", out_a, flush=True)

    # (c) U0 매일 대 월말 S: 주별 차이 분해
    mon = L2.snapped(base, cal, sorted(extra["bio"]))
    mm = aux.copy()
    mm.loc[rows26] = mon.loc[rows26]
    _, G0m, um0m = V.grades_sc(P, mm)
    rd, _ = week_detail(P, G0d, um0d, a, b, "S")
    rm, _ = week_detail(P, G0m, um0m, a, b, "S")
    same = sum(1 for x, y in zip(rd, rm) if x[1] == y[1])
    gaps = sorted(((np.log(y[2]) - np.log(x[2]), str(x[0].date()), x[1], y[1]) for x, y in zip(rd, rm) if x[1] != y[1]),
                  key=lambda t: -abs(t[0]))
    tot = sum(g[0] for g in gaps)
    res["c_U0_daily_vs_monthly_S"] = {
        "repro_ret_daily_monthly": [round((np.prod([x[2] for x in rd]) - 1) * 100, 1),
                                    round((np.prod([x[2] for x in rm]) - 1) * 100, 1)],
        "weeks": len(rd), "weeks_same_picks": same, "total_log_gap": round(tot, 3),
        "top3_weeks_log_gap": [(round(g[0], 3), g[1], list(g[2]), list(g[3])) for g in gaps[:3]],
        "share_of_gap_in_top3_weeks": round(sum(g[0] for g in gaps[:3]) / tot, 2) if tot else None}
    print("c", json.dumps(res["c_U0_daily_vs_monthly_S"], ensure_ascii=False), flush=True)
    del G0m, um0m

    # (b) 위상 민감도: 월 1회 스냅샷인데 날짜를 월말 k거래일 전으로
    ms = [pd.Timestamp(s) for s in sorted(extra["bio"])]
    out_b = {}
    for k in (5, 10, 15):
        anchors = [ms[0]]
        for s in ms[1:-1]:
            i = cal.get_loc(s) - k
            anchors.append(cal[i])
        anchors.append(ms[-1])
        mk = aux.copy()
        mk.loc[rows26] = L2.snapped(base, cal, sorted(set(anchors))).loc[rows26]
        _, G, um = V.grades_sc(P, mk)
        out_b[f"monthly_minus_{k}d"] = {r: X.weekly(P, G, um, a, b, r)["ret"] for r in RULES}
        print("b", k, out_b[f"monthly_minus_{k}d"], flush=True)
        del G, um
    res["b_U0_monthly_phase_sensitivity_y2026"] = out_b
    (HERE.parent / "results" / "verify_audit2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
