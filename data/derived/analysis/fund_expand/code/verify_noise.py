"""스냅샷 날짜 잡음 띠(2026-10-12, "더 수집할 가치가 없다는 거야?" 후속 검증).

지금 범위(U0 = TIME + 2ETFQ1)를 월 1회 스냅샷으로 낮추되 기준일을 바꿔 가며(달의 k번째 거래일, 월말),
또 주 1회 스냅샷으로 낮추되 요일을 바꿔 가며 같은 시험을 돌린다. 스냅샷은 그 날짜부터만 쓴다(앞날 정보 없음).
같은 정보량(월 1회)에서 날짜만 바꿨을 때 S·TOP7·ALL 2026 수익이 얼마나 흔들리는지가 '잡음 띠'다.
선택: --paired 면 같은 기준일로 U4(기본 범위 스냅샷 + 추가 3펀드 월말 PDF)도 돌려 U4-U0 짝 차이를 본다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as X  # noqa: E402
import lag_check as L  # noqa: E402
import lag_check2 as L2  # noqa: E402

H = X.H
RULES = ("S", "SA", "BUF", "TOP7", "ALL")
WINS = ("front", "back", "y2026")


def monthly_anchors(d26: pd.DatetimeIndex, k) -> list:
    """달마다 k번째 거래일(k='last'면 마지막). 그 달 거래일이 k보다 적으면 마지막 날. 첫 거래일은 늘 시작 앵커."""
    out = [d26[0]]
    per = pd.Series(d26, index=d26).groupby(d26.to_period("M"))
    for _, g in per:
        days = list(g.values)
        if k == "last":
            out.append(pd.Timestamp(days[-1]))
        else:
            out.append(pd.Timestamp(days[min(k, len(days)) - 1]))
    return sorted(set(out))


def weekly_anchors(d26: pd.DatetimeIndex, wd: int) -> list:
    """주마다 요일 wd(0=월) 이하인 마지막 거래일, 없으면(휴장) 그 주 첫 거래일."""
    out = [d26[0]]
    wk = pd.Series(d26, index=d26).groupby(d26.to_period("W-SUN"))
    for _, g in wk:
        days = [pd.Timestamp(x) for x in g.values]
        le = [x for x in days if x.weekday() <= wd]
        out.append(le[-1] if le else days[0])
    return sorted(set(out))


def slim(r):
    return {w: {k: {"ret": r[w][k]["ret"], "mdd": r[w][k]["mdd"], "avg_n": r[w][k]["avg_n"], "weeks": r[w][k]["weeks"]}
                for k in RULES} for w in WINS}


def band(vals: dict) -> dict:
    v = np.array(list(vals.values()), dtype=float)
    return {"n": len(v), "min": round(float(v.min()), 1), "max": round(float(v.max()), 1),
            "range": round(float(v.max() - v.min()), 1), "mean": round(float(v.mean()), 1),
            "median": round(float(np.median(v)), 1), "std": round(float(v.std(ddof=1)), 1) if len(v) > 1 else 0.0,
            "argmin": min(vals, key=vals.get), "argmax": max(vals, key=vals.get)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--ks", default="1,3,5,7,9,11,13,15,17,last")
    ap.add_argument("--out", default="verify_noise.json")
    args = ap.parse_args()

    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    cal, cols = P["C"].index, P["C"].columns
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    rows26 = cal >= pd.Timestamp("2026-01-02")
    d26 = cal[rows26]
    aux = X.membership(P, [], extra, aux_all=True)  # 2025 이전 처리는 compare와 같게

    def run_u0(snaps):
        m = aux.copy()
        m.loc[rows26] = L2.snapped(base, cal, snaps).loc[rows26]
        return slim(L.run(P, m))

    def run_u4(snaps):
        m = aux.copy()
        m.loc[rows26] = L2.with_extras(L2.snapped(base, cal, snaps), X.VARIANTS["U4"], extra, cols, cal).loc[rows26]
        return slim(L.run(P, m))

    res = {"schedules": {}, "u0": {}, "u4": {}}
    t0 = time.time()
    # 재현 점검: 매일(원래 U0), lag_check 월말(bio PDF 날짜), lag_check2 주간(그 주 마지막 거래일)
    m = aux.copy()
    res["u0"]["daily"] = slim(L.run(P, m))
    print("daily", {w: res["u0"]["daily"][w]["S"]["ret"] for w in WINS}, f"{time.time() - t0:.0f}s", flush=True)

    sched = {"M_pdfdates": [pd.Timestamp(s) for s in sorted(extra["bio"])]}
    ks = [k if k == "last" else int(k) for k in args.ks.split(",")]
    for k in ks:
        sched[f"M_td{k}" if k != "last" else "M_last"] = monthly_anchors(d26, k)
    for wd, nm in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri")):
        sched[f"W_{nm}"] = weekly_anchors(d26, wd)
    for name, snaps in sched.items():
        res["schedules"][name] = [s.date().isoformat() for s in snaps]
        res["u0"][name] = run_u0(snaps)
        r = res["u0"][name]
        line = " ".join(f"{k} {r['y2026'][k]['ret']:+.1f}" for k in ("S", "TOP7", "ALL"))
        if args.paired:
            res["u4"][name] = run_u4(snaps)
            q = res["u4"][name]
            line += " || U4 " + " ".join(f"{k} {q['y2026'][k]['ret']:+.1f}" for k in ("S", "TOP7", "ALL"))
        print(name, f"S front {r['front']['S']['ret']:+.1f} back {r['back']['S']['ret']:+.1f} |", line,
              f"{time.time() - t0:.0f}s", flush=True)

    # 띠: 월 1회(재현용 M_pdfdates는 M_last와 거의 같아 띠 계산에서 뺌), 주 1회
    groups = {"monthly": [n for n in sched if n.startswith("M_") and n != "M_pdfdates"],
              "weekly": [n for n in sched if n.startswith("W_")]}
    bands = {}
    for g, names in groups.items():
        bands[g] = {}
        for w in WINS:
            for k in RULES:
                bands[g][f"{w}.{k}"] = band({n: res["u0"][n][w][k]["ret"] for n in names})
        if args.paired:
            # monthly: 기본 범위도 월 1회 = 같은 조건 / weekly: 기본 주 1회 + 추가 월말 = 섞인 조건(원래 비교와 같은 꼴)
            pk, uk = f"paired_U4_minus_U0_{g}", f"U4_{g}"
            bands[pk], bands[uk] = {}, {}
            for w in WINS:
                for k in RULES:
                    bands[pk][f"{w}.{k}"] = band(
                        {n: round(res["u4"][n][w][k]["ret"] - res["u0"][n][w][k]["ret"], 1) for n in names})
                    bands[uk][f"{w}.{k}"] = band({n: res["u4"][n][w][k]["ret"] for n in names})
    res["bands"] = bands
    out = HERE.parent / "results" / args.out
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in bands:
        for key in ("y2026.S", "front.S", "back.S", "y2026.TOP7", "y2026.ALL", "y2026.SA", "y2026.BUF"):
            b = bands[g][key]
            print(g, key, f"min {b['min']} max {b['max']} range {b['range']} std {b['std']} mean {b['mean']} "
                          f"({b['argmin']}..{b['argmax']})")
    print("saved", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
