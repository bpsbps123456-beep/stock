"""N1 본 계산: 주가·금리 동반 하락 첫 발생(M)의 긴 이력 검증. SOXX(2001-07~)·QQQ(1999-03~)·SPY(1994~), ^TNX.
조건은 각 지수 자신의 5일 수익으로 정의한다. 사슬(첫 발생·5일 간격)은 상장일부터 끊지 않고 잇는다.
기간은 진입일 연도로 나눈다. 묶음 부트스트랩은 달력 월(같은 날·같은 달 신호를 함께 뽑음), 5,000회.
-> results/n1_main.csv(지수×기간×기간길이), n1_regime.csv, n1_yield_sens.csv, n1_split_c.csv,
   n1_robust.csv(배당 반영·분기 묶음), n1_signals.csv, n1_verdict.json"""
import json
import pathlib
import sys

import exchange_calendars as xcals
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
from n1_lib import HORIZONS, attach_controls, block_boot, conditions, daily_table, onsets, summarize  # noqa: E402

RES = HERE / "results"
SYMS = ["SOXX", "QQQ", "SPY"]
PERIODS = {"2003-07": (2003, 2007), "2008-09": (2008, 2009), "2010-19": (2010, 2019), "2020-23": (2020, 2023),
           "2024-26": (2024, 2026), "2003-23": (2003, 2023), "pre2003": (1990, 2002)}
B = 5000

cal = xcals.get_calendar("XNYS", start="1994-01-01", end="2026-12-31")
SES = cal.sessions_in_range("1994-01-03", "2026-10-23").tz_localize(None)


def load(kind):
    raw = pd.read_parquet(HERE / "cache" / f"n1_{kind}.parquet")
    op = raw.pivot(index="date", columns="sym", values="O").reindex(SES)
    cl = raw.pivot(index="date", columns="sym", values="C").reindex(SES)
    return op, cl


def build(op, cl, sym, thr=0.0):
    first = cl[sym].first_valid_index()
    o, c, t = op[sym].loc[first:], cl[sym].loc[first:], cl["^TNX"].loc[first:]
    assert o.notna().all() and c.notna().all(), sym
    d = daily_table(o, c, t)
    ev = {fam: attach_controls(d, onsets(cond)) for fam, cond in conditions(c, t, thr).items()}
    return d, ev


def in_period(df, y0, y1):
    return df[(df.year >= y0) & (df.year <= y1)]


def months_of(d, y0, y1, col="month"):
    return np.unique(in_period(d, y0, y1)[col].to_numpy())


def flat(prefix, res):
    out = {}
    for k, v in res.items():
        if isinstance(v, list):
            out[f"{prefix}{k}_lo"], out[f"{prefix}{k}_hi"] = v
        else:
            out[f"{prefix}{k}"] = v
    return out


def cell(d, ev, y0, y1, h, col="month"):
    g = {k: in_period(v, y0, y1) for k, v in ev.items()}
    sM, sC, sD = (summarize(g[k], h) for k in ["M", "C", "D"])
    bt = block_boot({"M": g["M"], "C": g["C"], "D": g["D"]}, h, months_of(d, y0, y1, col), B=B, col=col)
    row = dict(n_M=sM["n"], wins_M=sM["wins"], hit_M=sM["hit"], mean_M=sM["mean"],
               a_hit=sM["a_hit"], a_mean=sM["a_mean"], b_hit=sM["b_hit"], b_mean=sM["b_mean"],
               n_C=sC["n"], hit_C=sC["hit"], mean_C=sC["mean"], C_b_hit=sC["b_hit"], C_b_mean=sC["b_mean"],
               n_D=sD["n"], hit_D=sD["hit"], mean_D=sD["mean"], D_b_hit=sD["b_hit"], D_b_mean=sD["b_mean"])
    keep = ["mean_M_ci", "dhit_b", "dhit_b_ci", "dmean_b", "dmean_b_ci", "dhit_C", "dhit_C_ci", "dmean_C",
            "dmean_C_ci", "dhit_D", "dhit_D_ci", "dmean_D", "dmean_D_ci"]
    row.update(flat("", {k: bt[k] for k in keep}))
    return row


def main():
    op, cl = load("raw")
    built = {s: build(op, cl, s) for s in SYMS}
    # 신호 목록
    sig = []
    for s, (d, ev) in built.items():
        for fam, e in ev.items():
            x = e[["year", "bucket", "above200", "ret5_sig", "dy5_sig"] + [f"r{h}" for h in HORIZONS] + ["b_mean5", "b_hit5"]].copy()
            x.insert(0, "entry", x.index.date)
            x.insert(0, "family", fam)
            x.insert(0, "sym", s)
            sig.append(x)
    pd.concat(sig).to_csv(RES / "n1_signals.csv", index=False, encoding="utf-8-sig")

    rows = []
    for s, (d, ev) in built.items():
        for pname, (y0, y1) in PERIODS.items():
            if in_period(d, y0, y1).empty:
                continue
            for h in HORIZONS:
                rows.append(dict(sym=s, period=pname, h=h, **cell(d, ev, y0, y1, h)))
        print(s, "main done")
    main_df = pd.DataFrame(rows)
    main_df.to_csv(RES / "n1_main.csv", index=False, encoding="utf-8-sig")

    # 장세별(신호일 종가가 200일선 위·아래). (b)는 같은 해·낙폭·장세 날, (c)는 같은 장세 C 신호
    rrows = []
    for s, (d, _) in built.items():
        first = cl[s].first_valid_index()
        c, t = cl[s].loc[first:], cl["^TNX"].loc[first:]
        conds = conditions(c, t)
        evR = {fam: attach_controls(d, onsets(cond), regime_match=True) for fam, cond in conds.items()}
        for pname in ["2003-23", "2024-26", "2003-07", "2008-09", "2010-19", "2020-23"]:
            y0, y1 = PERIODS[pname]
            for reg in [1.0, 0.0]:
                g = {k: in_period(v[v.above200 == reg], y0, y1) for k, v in evR.items()}
                for h in [5, 20]:
                    sM, sC = summarize(g["M"], h), summarize(g["C"], h)
                    bt = block_boot({"M": g["M"], "C": g["C"]}, h, months_of(d, y0, y1), B=B)
                    rrows.append(dict(sym=s, period=pname, above200=int(reg), h=h, n_M=sM["n"], hit_M=sM["hit"], mean_M=sM["mean"],
                                      b_hit=sM["b_hit"], b_mean=sM["b_mean"], n_C=sC["n"], hit_C=sC["hit"], mean_C=sC["mean"],
                                      **flat("", {k: bt[k] for k in ["dhit_b", "dhit_b_ci", "dmean_b", "dmean_b_ci", "dhit_C", "dhit_C_ci", "dmean_C", "dmean_C_ci"]})))
    pd.DataFrame(rrows).to_csv(RES / "n1_regime.csv", index=False, encoding="utf-8-sig")
    print("regime done")

    # 금리 하락 폭 민감도(설명용): M_thr = 주가 5일 < 0 & 금리 5일 변화 < thr
    yrows = []
    for s in SYMS:
        d, ev0 = built[s]
        for thr in [0.0, -0.05, -0.10, -0.20]:
            _, evt = build(op, cl, s, thr)
            for pname in ["2003-23", "2024-26"]:
                y0, y1 = PERIODS[pname]
                for h in [5, 20]:
                    g = {"M": in_period(evt["M"], y0, y1), "C": in_period(ev0["C"], y0, y1)}
                    sM = summarize(g["M"], h)
                    bt = block_boot(g, h, months_of(d, y0, y1), B=B)
                    yrows.append(dict(sym=s, thr_bp=int(round(thr * 100)), period=pname, h=h, n_M=sM["n"], hit_M=sM["hit"],
                                      mean_M=sM["mean"], b_hit=sM["b_hit"], b_mean=sM["b_mean"],
                                      **flat("", {k: bt[k] for k in ["dhit_b", "dhit_b_ci", "dmean_b", "dmean_b_ci", "dhit_C", "dmean_C", "dmean_C_ci"]})))
    pd.DataFrame(yrows).to_csv(RES / "n1_yield_sens.csv", index=False, encoding="utf-8-sig")
    print("yield sens done")

    # 같은 날짜 집합 안에서 금리 방향의 몫(설명용): 주가 하락 첫 발생(C) 중 그날 금리 5일 하락 vs 상승
    crows = []
    for s, (d, ev) in built.items():
        C = ev["C"]
        for pname in ["2003-23", "2024-26", "2003-07", "2008-09", "2010-19", "2020-23"]:
            y0, y1 = PERIODS[pname]
            g = in_period(C, y0, y1)
            gf, gr = g[g.dy5_sig < 0], g[g.dy5_sig > 0]
            for h in [5, 20]:
                bt = block_boot({"M": gf, "X": gr}, h, months_of(d, y0, y1), B=B)
                sf, sr = summarize(gf, h), summarize(gr, h)
                crows.append(dict(sym=s, period=pname, h=h, n_fall=sf["n"], hit_fall=sf["hit"], mean_fall=sf["mean"],
                                  ex_b_fall=sf["mean"] - sf["b_mean"], n_rise=sr["n"], hit_rise=sr["hit"], mean_rise=sr["mean"],
                                  ex_b_rise=sr["mean"] - sr["b_mean"],
                                  **flat("", {"dmean": bt["dmean_X"], "dmean_ci": bt["dmean_X_ci"], "dhit": bt["dhit_X"], "dhit_ci": bt["dhit_X_ci"]})))
    pd.DataFrame(crows).to_csv(RES / "n1_split_c.csv", index=False, encoding="utf-8-sig")
    print("split C done")

    # 강건성: 배당 반영 가격, 분기 묶음
    rob = []
    opA, clA = load("adj")
    for s in SYMS:
        d, ev = built[s]
        dA, evA = build(opA, clA, s)
        for lab, (dd, ee, col) in {"raw_month": (d, ev, "month"), "raw_quarter": (d, ev, "quarter"),
                                   "divadj_month": (dA, evA, "month")}.items():
            for pname in ["2003-23", "2024-26"]:
                y0, y1 = PERIODS[pname]
                rob.append(dict(sym=s, version=lab, period=pname, h=5, **cell(dd, ee, y0, y1, 5, col)))
    rob = pd.DataFrame(rob)
    rob.to_csv(RES / "n1_robust.csv", index=False, encoding="utf-8-sig")
    print("robust done")

    # 판정(BRIEF N1): 2003~2023, 5일. M의 상승 비율·평균이 (b)·(c)를 모두 넘고, 네 차이의 95% 구간 하한 > 0
    ver = {}
    for s in SYMS:
        r = main_df[(main_df.sym == s) & (main_df.period == "2003-23") & (main_df.h == 5)].iloc[0]
        checks = {}
        for k in ["dhit_b", "dmean_b", "dhit_C", "dmean_C"]:
            checks[k] = dict(point=float(r[k]), lo=float(r[f"{k}_ci_lo"]), hi=float(r[f"{k}_ci_hi"]),
                             point_pos=bool(r[k] > 0), ci_above0=bool(r[f"{k}_ci_lo"] > 0))
        ver[s] = dict(n_M=int(r.n_M), hit_M=float(r.hit_M), mean_M=float(r.mean_M), b_hit=float(r.b_hit), b_mean=float(r.b_mean),
                      n_C=int(r.n_C), hit_C=float(r.hit_C), mean_C=float(r.mean_C), checks=checks,
                      pass_point=all(c["point_pos"] for c in checks.values()),
                      pass_full=all(c["point_pos"] and c["ci_above0"] for c in checks.values()))
    ver["primary"] = "SOXX"
    ver["rule"] = "2003~2023, 5일: M 상승비율·평균 > (b)와 (c), 네 차이의 월 묶음 부트스트랩 95% 하한 > 0"
    ver["grid"] = dict(main_cells="지수 3 × 조건 3(M·C·D) × 기간 5 = 45(기간길이 5개 보조, 2003-23 합계·pre2003 설명용 추가)",
                       regime="지수 3 × 기간 6 × 장세 2 × 기간길이 2", yield_sens="지수 3 × 문턱 4 × 기간 2 × 기간길이 2",
                       bootstrap=f"달력 월 묶음 {B}회(분기 묶음 강건성)")
    (RES / "n1_verdict.json").write_text(json.dumps(ver, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ver, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
