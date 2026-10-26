"""메커니즘 검증 2 (2026-10-12, verify_mech 결과를 본 뒤 추가 — 선택편향 있음, 변형 수는 아래 고정 6개뿐).

A. 통로 나누기(매일 → 월말): 지금 범위(TIME+2ETFQ1)를
   addlag  = 매일 보유 AND 직전 월말 보유  (새로 산 종목은 다음 월말까지 기다림, 판 종목은 바로 뺌)
   droplag = 매일 보유 OR  직전 월말 보유  (새로 산 종목은 바로 넣음, 판 종목은 월말까지 남김)
   으로 바꿔 S·TOP7·ALL을 다시 잰다. addlag가 월말(83.6)에 가까우면 '늦게 넣기', droplag면 '늦게 빼기'가 원인.
B. 새로 들어온 종목 효과: U0 매일 S 선택(종목-주)을 결정일까지 연속 보유 일수 <=10(점수 평활 창과 같은 10일) 대 >10로 나눠
   주간 수익과 그 주 포트폴리오 대비 초과를 잰다.
C. U1(+로봇) 월말이 왜 이기나: 같은 수 대조.
   C1 = U0 순위로 U1의 그 주 S 개수만큼 고르기(관문 같음), C2 = U1 순위로 U0의 S 개수만큼 고르기.
   C1이 U1(105.4)에 가까우면 'S 자리 수 증가', C2가 U1에 가까우면 '순위 자체가 나아짐'.
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
import control as CT  # noqa: E402
import lag_check as L  # noqa: E402
import lag_check2 as L2  # noqa: E402
import verify_mech as VM  # noqa: E402

H = X.H
RULES = ("S", "TOP7", "ALL")


def brief(r):
    return {w: {k: r[w][k]["ret"] for k in RULES} | {"S_n": r[w]["S"]["avg_n"]} for w in ("front", "back", "y2026")}


def entry_age(um):
    """결정일까지 연속으로 범위 안에 있던 거래일 수."""
    a = um.to_numpy(bool).astype(int)
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        out[i] = (out[i - 1] + 1) * a[i] if i else a[i]
    return pd.DataFrame(out, index=um.index, columns=um.columns)


def fresh_split(rows, age):
    g = defaultdict(lambda: {"n": 0, "ret": [], "exc": []})
    for x in rows:
        if not x["names"]:
            continue
        for s in x["names"]:
            k = "age<=10" if age.at[x["prev"], s] <= 10 else "age>10"
            g[k]["n"] += 1
            g[k]["ret"].append(x["r"][s])
            g[k]["exc"].append(x["r"][s] - x["rp"])
    return {k: {"name_weeks": v["n"], "mean_wk_ret_pct": round(100 * float(np.mean(v["ret"])), 2),
                "mean_excess_vs_port_pct": round(100 * float(np.mean(v["exc"])), 2)} for k, v in g.items()}


def weekly_k(P, sc, um, up, a, b, kmap):
    """주마다 k개(kmap[w0])를 점수 상위·관문 통과에서 고른다. compare.weekly와 같은 체결."""
    O, C = P["O"], P["C"]
    cal = C.index
    d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))]
    wk = pd.Series(d.to_period("W-SUN"), index=d)
    firsts = [x for x, w in zip(d, wk) if x == wk[wk == w].index[0]]
    eq, n = 1.0, []
    for i in range(len(firsts) - 1):
        w0, w1 = firsts[i], firsts[i + 1]
        prev = cal[cal.get_loc(w0) - 1]
        if prev < pd.Timestamp(a) or kmap.get(w0, 0) == 0:
            continue
        row = sc.loc[prev][um.loc[prev] & up.loc[prev].fillna(False)].dropna()
        names = [s for s in row.sort_values(ascending=False).index[:kmap[w0]] if np.isfinite(O.at[w0, s]) and np.isfinite(O.at[w1, s])]
        if not names:
            continue
        n.append(len(names))
        eq *= float((O.loc[w1, names] / O.loc[w0, names]).mean())
    return round((eq - 1) * 100, 1), round(float(np.mean(n)), 2)


def main():
    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    cal, cols = P["C"].index, P["C"].columns
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    rows26 = cal >= pd.Timestamp("2026-01-02")
    aux = X.membership(P, [], extra, aux_all=True)
    snaps = sorted(extra["bio"])
    mon = L2.snapped(base, cal, snaps)
    a, b = X.WINDOWS["y2026"]
    res = {"A_channels": {}, "B_fresh": {}, "C_slots": {}}

    # A
    for label, mm in (("addlag(base&monthly)", base & mon), ("droplag(base|monthly)", base | mon)):
        m = aux.copy()
        m.loc[rows26] = mm.loc[rows26]
        r = brief(L.run(P, m))
        n26 = float(m.loc[rows26].sum(axis=1).mean())
        res["A_channels"][label] = r | {"avg_members_2026": round(n26, 1)}
        print("A", label, r, "members", round(n26, 1), flush=True)
    res["A_channels"]["reference"] = {"daily_S": 58.4, "monthly_S": 83.6, "daily_TOP7": 113.5, "monthly_TOP7": 119.2,
                                      "daily_ALL": 63.4, "monthly_ALL": 52.7,
                                      "avg_members_daily": round(float(base.loc[rows26].sum(axis=1).mean()), 1),
                                      "avg_members_monthly": round(float(mon.loc[rows26].sum(axis=1).mean()), 1)}

    # B
    for name, m in (("U0_daily", aux), ("U0_monthly", None)):
        if m is None:
            m = aux.copy()
            m.loc[rows26] = mon.loc[rows26]
        G, um = X.grades(P, m)
        rows, ret = VM.weekly_detail(P, G, um, a, b)
        res["B_fresh"][name] = {"S_ret": ret, "split": fresh_split(rows, entry_age(um))}
        print("B", name, res["B_fresh"][name], flush=True)

    # C
    keep = {}
    for name, funds in (("U0_monthly", []), ("U1_monthly", ["rob"])):
        m = aux.copy()
        m.loc[rows26] = L2.with_extras(mon, funds, extra, cols, cal).loc[rows26]
        sc, um, up, G = CT.score_and_gate(P, m)
        rows, ret = VM.weekly_detail(P, G, um, a, b)
        keep[name] = {"sc": sc, "um": um, "up": up, "rows": rows, "ret": ret}
        u = um.loc[rows26].sum(axis=1)
        res["C_slots"][name] = {"S_ret": ret, "avg_universe_2026": round(float(u.mean()), 1),
                                "avg_S_per_week": round(float(np.mean([len(x["names"]) for x in rows])), 2)}
        print("C", name, res["C_slots"][name], flush=True)
    k0 = {x["w0"]: len(x["names"]) for x in keep["U0_monthly"]["rows"]}
    k1 = {x["w0"]: len(x["names"]) for x in keep["U1_monthly"]["rows"]}
    u0, u1 = keep["U0_monthly"], keep["U1_monthly"]
    res["C_slots"]["check_U0_rank_U0_count"] = weekly_k(P, u0["sc"], u0["um"], u0["up"], a, b, k0)
    res["C_slots"]["check_U1_rank_U1_count"] = weekly_k(P, u1["sc"], u1["um"], u1["up"], a, b, k1)
    res["C_slots"]["C1_U0_rank_U1_count"] = weekly_k(P, u0["sc"], u0["um"], u0["up"], a, b, k1)
    res["C_slots"]["C2_U1_rank_U0_count"] = weekly_k(P, u1["sc"], u1["um"], u1["up"], a, b, k0)
    # U1에만 S였던 종목의 U0 안 순위(점수 백분위)
    pr0 = H.pct_rank(u0["sc"], u0["um"] & u0["sc"].notna())
    w0 = {x["w0"]: x for x in u0["rows"]}
    pcts, rob_in_S = [], 0
    rob_all = set().union(*extra["rob"].values())
    for x in u1["rows"]:
        for s in set(x["names"]) - set(w0[x["w0"]]["names"]):
            v = pr0.at[x["prev"], s]
            if np.isfinite(v):
                pcts.append(float(v))
            elif s in rob_all:
                rob_in_S += 1
    res["C_slots"]["U1_only_U0_pct"] = {"n": len(pcts), "min": round(min(pcts), 3), "median": round(float(np.median(pcts)), 3),
                                        "max": round(max(pcts), 3), "robot_name_weeks_not_in_U0": rob_in_S}
    for k in ("check_U0_rank_U0_count", "check_U1_rank_U1_count", "C1_U0_rank_U1_count", "C2_U1_rank_U0_count", "U1_only_U0_pct"):
        print("C", k, res["C_slots"][k], flush=True)
    (HERE.parent / "results" / "verify_mech2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
