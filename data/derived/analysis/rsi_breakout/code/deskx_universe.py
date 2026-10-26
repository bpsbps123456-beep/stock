"""universe 계열(2026-10-10): 데스크 7종목(#774) 후보를 TIME 미국 보유 → TIME ∪ KoAct 미국 보유로 넓히면?

질문  MU가 왜 한 번도 안 뽑히나 → 데스크는 TIME 미국 보유만 순위를 매긴다. TIME은 MU를 3/17~3/25에만 들었고
      KoAct는 매일 약 4.4% 든다. KoAct 보유까지 후보에 넣으면 MU·PLTR이 언제 뽑히고 얼마를 벌었나.
격자  universe='union' × weight 4개 = 4개(사전 등록) + 기준(#774). 다른 값(120일·7종목·완충 5·상한·주간·수수료 0)은 그대로.
        time   두 펀드 다 들면 TIME 비중, KoAct에만 있으면 KoAct 비중
        avg    두 펀드 비중 평균(없는 쪽 0)
        max    두 비중 중 큰 값
        equal  같은 비중(상한은 그대로 적용)
고르기 앞 1/2~5/29 수익 최고 하나. 뒤 6/1~10/9·9/1~10/9·월별·낙폭은 확인용.
사후 1개(결과를 본 뒤 추가, 고르기 대상 아님): 'union·time 핵심만' = KoAct에만 있는 종목은 KoAct 비중 3% 이상일 때만 후보
      (격자 결과에서 손실이 KoAct 소형 종목(ORKA·PEPG·SIMO·VICR 등)에서 났고 MU는 KoAct 평균 4.6%라 MU만 남기는 꼴).
시점  k일 시가 결정 = k일보다 앞선 날짜 PDF(두 펀드 모두) + k-1일 종가까지(deskx와 같음).
기여  i일 보유 비중 w로 i일 시가→i+1일 시가 손익 = 가치_i × w × r. 구간 기여 = 구간 손익 합 ÷ 구간 시작 가치
      (종목별 합 = 구간 수익). 주간 기여 = 그 주 손익 ÷ 그 주 시작 가치(포트폴리오 주간 수익 중 몫).
실행  py data/derived/analysis/rsi_breakout/code/deskx_universe.py → results/deskx/universe.json
"""
import json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, PERIODS, MONTHS, START, END   # noqa: E402
from desk import config                                          # noqa: E402
from desk.strategies import iso_week, UNKNOWN                    # noqa: E402

OUT = os.path.join(HERE, "..", "results", "deskx", "universe.json")
FRONT, BACK, LATE, ALL = (p[0] for p in PERIODS)
WATCH = ("MU", "PLTR")
BASE = ("기준 #774 (TIME만)", Params())
GRID = [(f"union·{w}", Params(universe="union", weight=w)) for w in ("time", "avg", "max", "equal")]
POSTHOC = ("[사후] union·time 핵심만(KoAct전용≥3%)", Params(universe="union", weight="time"), 0.03)
REF = ("MU", "PLTR", "SNDK", "WDC", "DELL", "STX", "QQQ")


class CoreEngine(Engine):
    """KoAct에만 있는 종목은 KoAct 비중이 min_koact 이상일 때만 후보로 둔다(사후 변형용)."""
    min_koact = None

    def candidates(self, p, session):
        out = super().candidates(p, session)
        if self.min_koact is None or p.universe != "union":
            return out
        from desk.strategies import latest_before
        tpdf, kpdf = latest_before(self.book.pdfs, session), latest_before(self.kpdfs, session)
        tw = self.tw.get(tpdf, {}) if tpdf else {}
        kw = self.kw.get(kpdf, {}) if kpdf else {}
        return {k: v for k, v in out.items() if tw.get(k, 0.0) > 0 or kw.get(k, 0.0) >= self.min_koact}


def daily_pnl(E, v, rows):
    """종목별 일 손익(가치 단위). pnl[i][key] = value[i] * w_i[key] * r_key(i→i+1)."""
    days = list(v.index)
    val = v.to_numpy()
    k0 = E.days.index(days[0])
    out = []
    for i in range(len(days) - 1):
        k = k0 + i
        d = {}
        for key, wt in rows[i]["w"].items():
            o = E.col("opn", key)
            r = (o[k + 1] / o[k] - 1) if o is not None and np.isfinite(o[k + 1]) and np.isfinite(o[k]) and o[k] > 0 else 0.0
            d[key] = val[i] * wt * r
        out.append(d)
    # 맞춤 확인: 손익 합 = 가치 변화
    err = max(abs(sum(out[i].values()) - (val[i + 1] - val[i])) for i in range(len(out)))
    assert err < 1e-9, err
    return out


def contributions(E, v, pnl, a, b):
    """[a, b] 구간 종목별 기여(구간 시작 가치 대비)."""
    days = list(v.index)
    idx = [i for i, d in enumerate(days[:-1]) if a <= d <= b]
    if not idx:
        return {}
    base = v.iloc[idx[0]]
    acc = defaultdict(float)
    for i in idx:
        for key, x in pnl[i].items():
            acc[E.ticker[key]] += float(x / base)
    return dict(acc)


def rebalance_log(E, p, v, rows, pnl):
    """주간 재조정마다 MU·PLTR의 순위·모멘텀·비중·그 주 기여."""
    days = list(v.index)
    k0 = E.days.index(days[0])
    reb = [0] + [i for i in range(1, len(days)) if iso_week(days[i]) != iso_week(days[i - 1])]
    bounds = reb + [len(days) - 1]
    watch_keys = {E.ticker[k]: k for k in E.ticker if E.ticker[k] in WATCH}
    log = {t: [] for t in WATCH}
    for j, i0 in enumerate(reb):
        i1 = bounds[j + 1]                      # 주 끝(다음 재조정일, 손익 구간 [i0, i1))
        day = days[i0]
        ranked = E.ranking(p, k0 + i0, day)
        pos = {r["key"]: (n + 1, r["mom"]) for n, r in enumerate(ranked)}
        wk_val = v.iloc[i0]
        for t, key in watch_keys.items():
            w0 = rows[i0]["w"].get(key, 0.0)
            if w0 <= 0:
                continue
            c = float(sum(pnl[i].get(key, 0.0) for i in range(i0, i1)) / wk_val)
            rk, mom = pos.get(key, (None, None))
            log[t].append({"재조정일": day, "순위": rk, "모멘텀120": None if mom is None else round(mom, 4),
                           "비중": round(w0, 4), "주간기여": round(c, 4),
                           "주간포트수익": round(float(v.iloc[i1] / wk_val - 1), 4)})
    return log


def time_held(E, key, day):
    from desk.strategies import latest_before
    pdf = latest_before(E.book.pdfs, day)
    return pdf is not None and E.tw.get(pdf, {}).get(key, 0.0) > 0


def koact_only_stats(E, v, rows, pnl):
    """그날 TIME PDF에 없는(KoAct가 들여온) 종목의 평균 비중과 구간 기여."""
    days = list(v.index)
    flag = []
    for i, row in enumerate(rows):
        flag.append({key for key in row["w"] if not time_held(E, key, days[i])})
    out = {}
    for lab, a, b in PERIODS:
        idx = [i for i, d in enumerate(days[:-1]) if a <= d <= b]
        base = v.iloc[idx[0]]
        c = float(sum(pnl[i].get(key, 0.0) for i in idx for key in flag[i]) / base)
        avg_w = float(np.mean([sum(rows[i]["w"][key] for key in flag[i]) for i in idx]))
        out[lab] = {"기여": round(c, 4), "평균비중": round(avg_w, 4)}
    names = defaultdict(int)
    for i in range(len(rows)):
        for key in flag[i]:
            names[E.ticker[key]] += 1
    out["종목(보유일수)"] = dict(sorted(names.items(), key=lambda x: -x[1]))
    return out


def invested(rows):
    return float(np.mean([sum(r["w"].values()) for r in rows]))


def run_one(E, name, p):
    v, rows = E.run(p)
    res = E.evaluate(name, v, rows)
    pnl = daily_pnl(E, v, rows)
    res["평균투자비중"] = invested(rows)
    res["종목기여"] = {lab: {t: round(x, 4) for t, x in sorted(contributions(E, v, pnl, a, b).items(), key=lambda kv: -abs(kv[1]))[:12]}
                     for lab, a, b in PERIODS}
    res["MU·PLTR"] = rebalance_log(E, p, v, rows, pnl)
    res["MU·PLTR 구간기여"] = {lab: {t: round(contributions(E, v, pnl, a, b).get(t, 0.0), 4) for t in WATCH}
                           for lab, a, b in PERIODS}
    if p.universe == "union":
        res["KoAct가 들여온 종목"] = koact_only_stats(E, v, rows, pnl)
    res["params"] = {k: getattr(p, k) for k in ("universe", "weight")}
    return res, v, rows, pnl


def hold_returns(E):
    """참고: 종목을 그냥 들고 있었다면(시가→시가, 데스크와 같은 잣대) 구간 수익과 낙폭."""
    days = [d for d in E.days if d >= START]
    k0 = E.days.index(days[0])
    out = {}
    for t in REF:
        o = E.book.opn[t].to_numpy(float) if t in E.book.opn.columns else None
        if o is None:
            continue
        o = o[k0:k0 + len(days)]
        row = {}
        for lab, a, b in PERIODS:
            idx = [i for i, d in enumerate(days[:-1]) if a <= d <= b]
            seg = o[idx[0]: idx[-1] + 2]
            seg = seg[np.isfinite(seg)]
            row[lab] = {"수익": round(float(seg[-1] / seg[0] - 1), 4),
                        "낙폭": round(float((seg / np.maximum.accumulate(seg) - 1).min()), 4)}
        out[t] = row
    return out


def pct(x):
    return f"{x * 100:+6.1f}" if x is not None and np.isfinite(x) else "   n/a"


def main():
    E = CoreEngine()
    results, extra = [], {}
    for name, p in [BASE] + GRID:
        res, v, rows, pnl = run_one(E, name, p)
        results.append(res)
        extra[name] = (v, rows, pnl)
    E.min_koact = POSTHOC[2]
    post, *_ = run_one(E, POSTHOC[0], POSTHOC[1])
    post["사후"] = True
    E.min_koact = None
    refs = hold_returns(E)
    base = results[0]
    grid = results[1:]
    best = max(grid, key=lambda r: r[FRONT]["수익"])
    both = [r["name"] for r in grid if r[FRONT]["수익"] > base[FRONT]["수익"] and r[BACK]["수익"] > base[BACK]["수익"]]
    mdd_ok = [r["name"] for r in grid if r[ALL]["낙폭"] > base[ALL]["낙폭"] and r[ALL]["수익"] >= 0.9 * base[ALL]["수익"]]

    # 기준 대비 종목 기여 차(뒤·앞): 어떤 종목이 들어오고 빠져 차이가 났나
    diff = {}
    for r in grid:
        d = {}
        for lab in (FRONT, BACK):
            vb, rb, pb = extra[base["name"]]
            vv, rv, pv = extra[r["name"]]
            a, b = next((x[1], x[2]) for x in PERIODS if x[0] == lab)
            cb, cv = contributions(E, vb, pb, a, b), contributions(E, vv, pv, a, b)
            keys = set(cb) | set(cv)
            dd = {t: cv.get(t, 0.0) - cb.get(t, 0.0) for t in keys}
            d[lab] = {t: round(x, 4) for t, x in sorted(dd.items(), key=lambda kv: -abs(kv[1]))[:8]}
        diff[r["name"]] = d

    # ------------------------------------------------------------ 표
    hdr = f"{'변형':<20}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'MDD전체':>8}{'MDD뒤':>8}" + \
          "".join(f"{m[5:]:>7}" for m in MONTHS[5:]) + f"{'반도체':>7}{'투자':>6}"
    print("[universe 계열] 격자 4개 + 기준. 고르기 = 앞 1/2~5/29 수익 최고")
    print(hdr)
    for r in results + [post]:
        print(f"{r['name']:<20}{pct(r[FRONT]['수익'])}  {pct(r[BACK]['수익'])}{pct(r[LATE]['수익'])}  "
              f"{pct(r[ALL]['수익'])}  {pct(r[ALL]['낙폭'])}  {pct(r[BACK]['낙폭'])}"
              + "".join(f"{pct(r['월별'][m])}" for m in MONTHS[5:])
              + f"{r['반도체평균'] * 100:6.1f}{r['평균투자비중'] * 100:6.1f}")
    print(f"\n앞 기준 최고: {best['name']}  |  앞·뒤 모두 기준 초과: {both or '없음'}  |  "
          f"낙폭 개선·전체 수익 90% 이상 유지: {mdd_ok or '없음'}")
    print("\n참고 그냥 들고 있기(시가→시가) 앞/뒤/9/1~/전체 수익, 전체 낙폭:")
    for t, row in refs.items():
        print(f"  {t:<5}" + "".join(pct(row[lab]["수익"]) for lab in (FRONT, BACK, LATE, ALL)) + f"  MDD{pct(row[ALL]['낙폭'])}")
    for r in results + [post]:
        print(f"\n== {r['name']}  MU·PLTR 구간기여 {r['MU·PLTR 구간기여'][ALL]}")
        for t in WATCH:
            lg = r["MU·PLTR"][t]
            if lg:
                print(f"  {t}: " + ", ".join(f"{x['재조정일'][5:]}(#{x['순위']},{x['비중']:.2f},{x['주간기여'] * 100:+.1f}p)" for x in lg))
            else:
                print(f"  {t}: 선택 없음")
        if "KoAct가 들여온 종목" in r:
            ko = r["KoAct가 들여온 종목"]
            print("  KoAct 전용 종목 기여/평균비중:", {lab: ko[lab] for lab in (FRONT, BACK, ALL)})
            print("  KoAct 전용 종목(보유일수):", dict(list(ko["종목(보유일수)"].items())[:12]))
            if r["name"] in diff:
                print("  기준 대비 종목 기여 차:", diff[r["name"]])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"family": "universe", "grid": "union × weight{time,avg,max,equal} = 4 (+기준 #774)",
                   "start": START, "end": END, "base": base, "variants": grid, "best_by_front": best["name"],
                   "beats_base_both_halves": both, "lower_mdd_small_cost": mdd_ok, "contrib_diff_vs_base": diff,
                   "posthoc": post, "hold_reference": refs},
                  f, ensure_ascii=False, indent=1)
    print("\n저장:", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
