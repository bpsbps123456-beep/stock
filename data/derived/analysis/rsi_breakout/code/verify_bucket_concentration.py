"""감사(robustness / concentration): 버킷 분산 모멘텀 앞 기준 최선(T-sub-N5-eq) 대 live 데스크 규칙. 2026-10-10.

bucket_main.py의 BucketEngine을 그대로 가져다 쓴다(고치지 않음). 결과 JSON은 스크래치패드에만 쓴다.
점검
  1) 2026 월별(1~10월): 최선과 이웃(N4·N6 격자 밖, N3·N7, fund, sector, TK) 대 live
  2) 집중: 이름별 기여, 최대 기여 종목(들) 후보에서 빼고 다시 돌리기(최선·live 모두),
     최선이 live를 가장 크게 이긴 날 2·5일을 live 수익으로 바꾸기, 재조정 요일 바꾸기(월~금), 시작일 바꾸기, 6/1 새 출발
  3) 분산이 낙폭을 줄였나: 7월, MDD 구간(고점·저점 날짜), 서로의 MDD 구간 수익, 일간 변동성,
     재조정 때 든 종목끼리 직전 60거래일 평균 상관(사전 값)
  4) 실제로 든 버킷·테마(코인·전력/냉각·SW·광 등)와 기여
  5) 이웃: 모멘텀 90/150, 버퍼 N+3/N+8 (live도 같은 변화로 대조)
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py verify_bucket_concentration.py
"""
import json, os, sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deskx  # noqa: E402
from bucket_main import BucketEngine, BParams, contributions, open_ret, MERGE  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import in_semis_budget, UNKNOWN, run_model, iso_week as _iso_week  # noqa: E402
import watchlist  # noqa: E402

SCRATCH = r"C:\Users\geunt\AppData\Local\Temp\claude\C--Users-geunt-OneDrive-Desktop---\a98bfc76-8181-4d46-afbf-2585bc00e0fc\scratchpad"
OUT = os.path.join(SCRATCH, "verify_bucket_concentration.json")
MONTHS = [f"2026-{m:02d}" for m in range(1, 11)]
PER = (("front", "2026-01-02", "2026-05-29"), ("back", "2026-06-01", "2026-10-09"),
       ("sepoct", "2026-09-01", "2026-10-09"), ("full", "2026-01-02", "2026-10-09"))

PICK = BParams(universe="time", gran="sub", n=5, wmode="eq")
LIVE = BParams(gran="name", wmode="fund", n=7, semis_cap=0.50)

SUB2GROUP = {s: g for g, subs in watchlist.AI_UNIVERSE.items() for s in subs}
SEMIS_GROUPS = {"칩 설계", "파운드리·후공정", "메모리·스토리지", "장비·소재·테스트", "기판·부품"}
SECTOR2THEME = {"AI 반도체": "반도체·메모리", "반도체(적자)": "반도체·메모리", "메모리": "반도체·메모리",
                "반도체 장비": "반도체·메모리", "반도체 장비(적자)": "반도체·메모리", "반도체 소재": "반도체·메모리",
                "핀테크/디지털자산": "코인·디지털자산", "원자력/에너지": "전력·냉각·에너지", "엔터프라이즈 SW": "SW·AI 서비스",
                "빅테크": "빅테크·AI 모델", "중국 AI/플랫폼": "빅테크·AI 모델", "네트워크/광통신": "네트워크·광",
                "AI 인프라": "서버·데이터센터", "인터넷 인프라/CDN": "SW·AI 서비스"}
GROUP2THEME = {"네트워크·광통신": "네트워크·광", "서버·시스템": "서버·데이터센터", "전력·에너지": "전력·냉각·에너지",
               "데이터센터": "서버·데이터센터", "빅테크·AI 모델": "빅테크·AI 모델", "소프트웨어·보안": "SW·AI 서비스",
               "AI 응용": "SW·AI 서비스"}


class XEngine(BucketEngine):
    """후보에서 특정 티커를 빼는 감사용 엔진."""
    exclude: frozenset = frozenset()

    def candidates(self, p, session):
        c = super().candidates(p, session)
        return {k: v for k, v in c.items() if self.ticker[k] not in self.exclude} if self.exclude else c


def theme(E, key):
    sub = E.bucket(key, "sub")
    if sub.startswith("미분류:"):
        return "미분류"
    if sub in SUB2GROUP:
        g = SUB2GROUP[sub]
        if sub == "채굴→AI 전환":
            return "코인 채굴→AI"
        return "반도체·메모리" if g in SEMIS_GROUPS else GROUP2THEME.get(g, g)
    return SECTOR2THEME.get(sub, "기타:" + sub)


def daily(v):
    return (v.shift(-1) / v - 1).dropna()


def stats_from_r(r):
    out = {}
    for lab, a, b in PER:
        x = r[(r.index >= a) & (r.index <= b)]
        out[lab] = float((1 + x).prod() - 1) if len(x) else float("nan")
    curve = pd.concat([pd.Series([1.0], index=["start"]), (1 + r).cumprod()])
    dd = curve / np.maximum.accumulate(curve) - 1
    t = dd.idxmin()
    pk = curve.loc[:t].idxmax()
    out["mdd"] = float(dd.min())
    out["mdd_peak"], out["mdd_trough"] = pk, t
    out["months"] = {m: float((1 + r[r.index.str.startswith(m)]).prod() - 1) for m in MONTHS}
    out["vol_ann"] = float(r.std() * np.sqrt(252))
    return out


def run(E, p, exclude=()):
    E.exclude = frozenset(exclude)
    try:
        v, rows, log = E.run_logged(p)
    finally:
        E.exclude = frozenset()
    return v, rows, log


def window_ret(r, a, b):
    """a일 시가 → b일 시가(b일 전까지의 수익)."""
    x = r[(r.index >= a) & (r.index < b)]
    return float((1 + x).prod() - 1)


def exante_corr(E, rows, log, lb=60):
    k0 = E.days.index(rows[0]["day"])
    day_i = {r["day"]: i for i, r in enumerate(rows)}
    out = []
    for e in log:
        i = day_i.get(e["session"])
        if i is None:
            continue
        k = k0 + i
        names = [x for x, wt in rows[i]["w"].items() if wt > 1e-9]
        if len(names) < 2:
            continue
        R = []
        for x in names:
            c = E.col("adj", x)
            seg = c[k - 1 - lb:k]
            R.append(np.diff(seg) / seg[:-1])
        R = np.array(R)
        ok = np.isfinite(R).all(axis=0)
        C = np.corrcoef(R[:, ok])
        iu = np.triu_indices(len(names), 1)
        out.append(float(np.nanmean(C[iu])))
    return float(np.mean(out)) if out else float("nan")


def theme_table(E, v, rows, log):
    contrib = contributions(E, v, rows, lambda x: theme(E, x))["bucket_contrib"]
    w = defaultdict(float)
    for row in rows:
        for x, wt in row["w"].items():
            w[theme(E, x)] += wt / len(rows)
    weeks = Counter(theme(E, x) for e in log for x in e["keep"])
    names = defaultdict(set)
    for e in log:
        for x in e["keep"]:
            names[theme(E, x)].add(E.ticker[x])
    ths = sorted(set(contrib) | set(w), key=lambda t: -w[t])
    return {t: {"avg_weight": round(w[t], 3), "name_weeks": weeks.get(t, 0), "contrib": contrib.get(t, 0.0),
                "names": sorted(names[t])} for t in ths}


def main():
    E = XEngine()
    res = {}

    # 0) 재현: live = run_model, 최선 = builder main.json
    v_live, rows_live, log_live = run(E, LIVE)
    ref = np.asarray(run_model(E.book, start=deskx.START).value)
    d0 = float(np.max(np.abs(v_live.to_numpy() - ref)))
    v_pick, rows_pick, log_pick = run(E, PICK)
    with open(os.path.join(HERE, "../results/bucket/main.json"), encoding="utf-8") as f:
        mj = json.load(f)
    bp = next(x for x in mj["variants"] if x["name"] == "T-sub-N5-eq")
    sp, sl = stats_from_r(daily(v_pick)), stats_from_r(daily(v_live))
    print(f"[0] live vs run_model 최대차 {d0:.2e}; 최선 재현 앞 {sp['front']:.4f}(builder {bp['front']:.4f}) "
          f"뒤 {sp['back']:.4f}({bp['back']:.4f}) 전체 {sp['full']:.4f}({bp['full']:.4f})")
    res["repro"] = {"live_vs_run_model": d0, "pick": {k: sp[k] for k in ("front", "back", "full", "mdd")},
                    "builder": {k: bp[k] for k in ("front", "back", "full", "mdd")}}

    # 1) 월별: 최선과 이웃
    neigh = [("live", LIVE), ("#774", replace(LIVE, semis_cap=0.60)), ("PICK T-sub-N5-eq", PICK),
             ("T-sub-N4-eq(격자 밖)", replace(PICK, n=4)), ("T-sub-N6-eq(격자 밖)", replace(PICK, n=6)),
             ("T-sub-N3-eq", replace(PICK, n=3)), ("T-sub-N7-eq", replace(PICK, n=7)),
             ("T-sub-N5-fund", replace(PICK, wmode="fund")), ("T-sector-N5-eq", replace(PICK, gran="sector")),
             ("TK-sub-N5-eq", replace(PICK, universe="union")),
             ("T-sector-N10-eq", replace(PICK, gran="sector", n=10)), ("TK-sub-N10-eq", replace(PICK, universe="union", n=10))]
    runs = {}
    for nm, p in neigh:
        v, rows, log = (v_live, rows_live, log_live) if nm == "live" else (v_pick, rows_pick, log_pick) \
            if nm.startswith("PICK") else run(E, p)
        runs[nm] = (v, rows, log, p)
    print("\n[1] 2026 월별 수익(시가 경로, 소수)")
    print(f"{'변형':<24}" + "".join(f"{m[-2:]:>7}" for m in MONTHS) + f"{'앞':>7}{'뒤':>7}{'전체':>7}{'MDD':>7}{'live초과월':>9}")
    rl = daily(v_live)
    sl_m = stats_from_r(rl)["months"]
    res["monthly"] = {}
    for nm, (v, rows, log, p) in runs.items():
        s = stats_from_r(daily(v))
        beat = sum(1 for m in MONTHS if s["months"][m] > sl_m[m])
        res["monthly"][nm] = {"months": s["months"], **{k: s[k] for k in ("front", "back", "sepoct", "full", "mdd")},
                              "months_beating_live": beat}
        print(f"{nm:<24}" + "".join(f"{s['months'][m]:>7.3f}" for m in MONTHS)
              + f"{s['front']:>7.3f}{s['back']:>7.3f}{s['full']:>7.3f}{s['mdd']:>7.3f}{beat:>6}/10")

    # 2a) 기여 집중
    cp = contributions(E, v_pick, rows_pick, lambda x: E.bucket(x, "sub"))
    cl = contributions(E, v_live, rows_live, lambda x: x)
    print("\n[2a] 이름별 기여 상위5 — 최선:", cp["top5_contrib"], "상위3 몫", round(cp["top3_share"], 3))
    print("      live:", cl["top5_contrib"], "상위3 몫", round(cl["top3_share"], 3))
    res["contrib"] = {"pick": cp, "live": cl}

    # 2b) 최대 기여 종목 빼기
    print("\n[2b] 후보에서 종목 빼고 다시 돌림(최선 / live): 앞, 뒤, 전체, MDD")
    res["drop"] = {}
    top_pick = cp["top3"]
    for ex in ([top_pick[0]], top_pick[:2], top_pick[:3], ["SNDK"], ["DELL"], ["SNDK", "WDC", "STX"]):
        ex = list(dict.fromkeys(ex))
        a = stats_from_r(daily(run(E, PICK, ex)[0]))
        b = stats_from_r(daily(run(E, LIVE, ex)[0]))
        key = "-".join(ex)
        res["drop"][key] = {"pick": {k: a[k] for k in ("front", "back", "full", "mdd")},
                            "live": {k: b[k] for k in ("front", "back", "full", "mdd")}}
        print(f"  빼기 {key:<16} 최선 {a['front']:.3f} {a['back']:.3f} {a['full']:.3f} {a['mdd']:.3f}  |  live "
              f"{b['front']:.3f} {b['back']:.3f} {b['full']:.3f} {b['mdd']:.3f}  | 전체 차 {a['full'] - b['full']:+.3f}")

    # 2c) 최선이 live를 가장 크게 이긴 날을 live 수익으로 바꾸기
    rp = daily(v_pick)
    ex = (rp - rl)
    lex = np.log1p(rp) - np.log1p(rl)
    res["top_days"] = {}
    print("\n[2c] 최선-live 일간 초과 상위 날을 live 수익으로 교체")
    for kk in (1, 2, 5):
        for lab, a_, b_ in (("full", "2026-01-02", "2026-10-09"), ("front", "2026-01-02", "2026-05-29"),
                            ("back", "2026-06-01", "2026-10-09")):
            sub_ex = ex[(ex.index >= a_) & (ex.index <= b_)]
            days = list(sub_ex.sort_values(ascending=False).index[:kk])
            r2 = rp.copy()
            r2.loc[days] = rl.loc[days]
            s2 = stats_from_r(r2)
            share = float(lex.loc[days].sum() / lex[(lex.index >= a_) & (lex.index <= b_)].sum())
            res["top_days"][f"{lab}_top{kk}"] = {"days": {d: round(float(ex[d]), 4) for d in days}, lab: s2[lab],
                                                 "live": stats_from_r(rl)[lab], "log_excess_share": share}
            print(f"  {lab:<5} 상위 {kk}일 {[(d, round(float(ex[d]), 3)) for d in days]} → 최선 {s2[lab]:.3f} "
                  f"(원래 {stats_from_r(rp)[lab]:.3f}, live {stats_from_r(rl)[lab]:.3f}), 로그초과 중 몫 {share:.2f}")

    # 2d) 재조정 요일 바꾸기(주의 경계를 s일 미룸: s=0 월요일 = 원래 규칙)
    print("\n[2d] 재조정 요일(주 경계를 s일 미룸; 같은 PDF·전일 종가 규칙): 최선 / live  앞, 뒤, 전체, MDD")
    res["weekday"] = {}
    try:
        for s in range(5):
            deskx.iso_week = (lambda s_: (lambda day: (date.fromisoformat(day) - timedelta(days=s_)).isocalendar()[:2]))(s)
            a = stats_from_r(daily(run(E, PICK)[0]))
            b = stats_from_r(daily(run(E, LIVE)[0]))
            res["weekday"][s] = {"pick": {k: a[k] for k in ("front", "back", "full", "mdd")},
                                 "live": {k: b[k] for k in ("front", "back", "full", "mdd")}}
            print(f"  s={s} {['월', '화', '수', '목', '금'][s]}  최선 {a['front']:.3f} {a['back']:.3f} {a['full']:.3f} {a['mdd']:.3f}"
                  f"  | live {b['front']:.3f} {b['back']:.3f} {b['full']:.3f} {b['mdd']:.3f}"
                  f"  | 앞차 {a['front'] - b['front']:+.3f} 뒤차 {a['back'] - b['back']:+.3f}")
    finally:
        deskx.iso_week = _iso_week

    # 2e) 시작일 바꾸기 / 6/1 새 출발
    print("\n[2e] 시작일 바꾸기(그날 현금에서 시작): 최선 / live — 시작일~5/29 또는 6/1~10/9")
    res["start"] = {}
    saved = deskx.START
    try:
        for st, lab, a_, b_ in (("2026-01-12", "1/12~5/29", "2026-01-12", "2026-05-29"),
                                ("2026-02-02", "2/2~5/29", "2026-02-02", "2026-05-29"),
                                ("2026-03-02", "3/2~5/29", "2026-03-02", "2026-05-29"),
                                ("2026-06-01", "6/1~10/9 새 출발", "2026-06-01", "2026-10-09"),
                                ("2026-07-06", "7/6~10/9 새 출발", "2026-07-06", "2026-10-09")):
            deskx.START = st
            ra = daily(run(E, PICK)[0])
            rb = daily(run(E, LIVE)[0])
            a = window_ret(ra, a_, "2026-12-31" if b_ == "2026-10-09" else "2026-06-01")
            b = window_ret(rb, a_, "2026-12-31" if b_ == "2026-10-09" else "2026-06-01")
            res["start"][lab] = {"pick": a, "live": b}
            print(f"  {lab:<18} 최선 {a:.3f}  live {b:.3f}  차 {a - b:+.3f}")
    finally:
        deskx.START = saved

    # 3) 분산과 낙폭
    print("\n[3] 분산이 낙폭을 줄였나")
    sl_full = stats_from_r(rl)
    res["dd"] = {}
    for nm in ("live", "#774", "PICK T-sub-N5-eq", "T-sub-N3-eq", "T-sub-N7-eq", "T-sector-N5-eq",
               "T-sector-N10-eq", "TK-sub-N10-eq"):
        v, rows, log, p = runs[nm]
        r = daily(v)
        s = stats_from_r(r)
        in_live_dd = window_ret(r, sl_full["mdd_peak"] if sl_full["mdd_peak"] != "start" else "2026-01-02",
                                sl_full["mdd_trough"])
        in_pick_dd = window_ret(r, sp["mdd_peak"] if sp["mdd_peak"] != "start" else "2026-01-02", sp["mdd_trough"])
        corr = exante_corr(E, rows, log)
        semis = float(np.mean([sum(wt for x, wt in row["w"].items() if in_semis_budget(E.sector.get(x, UNKNOWN)))
                               for row in rows]))
        nb = float(np.mean([len({E.bucket(x, "sub") for x, wt in row["w"].items() if wt > 1e-9}) for row in rows]))
        res["dd"][nm] = {"mdd": s["mdd"], "peak": s["mdd_peak"], "trough": s["mdd_trough"], "jul": s["months"]["2026-07"],
                         "in_live_mdd_window": in_live_dd, "in_pick_mdd_window": in_pick_dd, "vol_ann": s["vol_ann"],
                         "full": s["full"], "full_over_mdd": s["full"] / -s["mdd"], "exante_corr60": corr,
                         "semis_like_avg": semis, "sub_buckets_avg": nb}
        print(f"  {nm:<18} MDD {s['mdd']:.3f} ({s['mdd_peak']}→{s['mdd_trough']}) 7월 {s['months']['2026-07']:.3f} "
              f"live MDD구간 {in_live_dd:.3f} 최선 MDD구간 {in_pick_dd:.3f} 변동성 {s['vol_ann']:.2f} "
              f"전체/|MDD| {s['full'] / -s['mdd']:.1f} 사전상관 {corr:.2f} 반도체류 {semis:.2f} 세부버킷 {nb:.1f}")
    # 격자 전체: 버킷 수(N)와 MDD·수익
    print("  격자 32개(main.json): 같은 U·G·W 안에서 N이 커질 때 MDD·전체")
    fam = defaultdict(list)
    for x in mj["variants"]:
        u, g, n, w = x["name"].split("-")
        fam[(u, g, w)].append((int(n[1:]), x["mdd"], x["full"], x["semis_avg"]))
    better_mdd_than_live = [x["name"] for x in mj["variants"] if x["mdd"] > sl_full["mdd"]]
    res["grid_family"] = {"-".join(k): sorted(vv) for k, vv in fam.items()}
    res["grid_mdd_better_than_live"] = better_mdd_than_live
    for k, vv in fam.items():
        vv.sort()
        print(f"   {'-'.join(k):<12} " + "  ".join(f"N{n}: MDD {m:.3f} 전체 {f:.2f} 반도체류 {s_:.2f}" for n, m, f, s_ in vv))
    print(f"  live보다 MDD 얕은 격자 변형: {better_mdd_than_live or '없음'} (live {sl_full['mdd']:.3f})")

    # 4) 실제로 든 버킷·테마
    print("\n[4] 테마별 평균 비중 / 보유 종목·주 / 기여(시작 1 기준 금액)")
    res["themes"] = {}
    for nm in ("live", "PICK T-sub-N5-eq", "T-sector-N10-eq", "TK-sub-N10-eq", "TK-sub-N5-eq"):
        v, rows, log, p = runs[nm]
        tt = theme_table(E, v, rows, log)
        res["themes"][nm] = tt
        print(f"  {nm}:")
        for t, d in tt.items():
            print(f"     {t:<16} 비중 {d['avg_weight']:.3f}  종목·주 {d['name_weeks']:>3}  기여 {d['contrib']:+.3f}  {d['names']}")
    # 최선의 세부 버킷 기여
    print("  최선 세부 버킷 기여:", {b: c for b, c in cp["bucket_contrib"].items()})
    # 후보에 테마가 있었나(T vs TK, 매주 평균 후보 수)
    print("  후보 안 테마별 평균 종목 수(재조정 주 평균):")
    res["candidate_themes"] = {}
    for uni in ("time", "union"):
        cnt = defaultdict(list)
        p = replace(PICK, universe=uni)
        for e in log_pick:
            k = E.days.index(e["session"])
            ranked = E.ranking(replace(p, weight="time"), k, e["session"])
            c = Counter(theme(E, r["key"]) for r in ranked)
            for t in set(c) | set(cnt):
                cnt[t].append(c.get(t, 0))
        avg = {t: round(float(np.mean(x)), 1) for t, x in sorted(cnt.items(), key=lambda z: -np.mean(z[1]))}
        res["candidate_themes"][uni] = avg
        print(f"   {uni}: {avg}")

    # 5) 이웃: lookback·buffer, live도 같은 변화로 대조
    print("\n[5] 이웃(감사용, 선택 근거 아님): 최선 / live 같은 변화")
    res["neighbours"] = {}
    for lab, kw in (("lookback 90", dict(lookback=90)), ("lookback 150", dict(lookback=150)),
                    ("buffer N+3", dict(buffer=3)), ("buffer N+8", dict(buffer=8)), ("기본", {})):
        a = stats_from_r(daily(run(E, replace(PICK, **kw))[0]))
        b = stats_from_r(daily(run(E, replace(LIVE, **kw))[0])) if kw else sl_full
        res["neighbours"][lab] = {"pick": {k: a[k] for k in ("front", "back", "sepoct", "full", "mdd")},
                                  "live": {k: b[k] for k in ("front", "back", "sepoct", "full", "mdd")}}
        print(f"  {lab:<12} 최선 앞 {a['front']:.3f} 뒤 {a['back']:.3f} 9/1~ {a['sepoct']:.3f} 전체 {a['full']:.3f} MDD {a['mdd']:.3f}"
              f"  | live 앞 {b['front']:.3f} 뒤 {b['back']:.3f} 전체 {b['full']:.3f} MDD {b['mdd']:.3f}"
              f"  | 앞차 {a['front'] - b['front']:+.3f} 뒤차 {a['back'] - b['back']:+.3f}")

    os.makedirs(SCRATCH, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("saved", OUT)


if __name__ == "__main__":
    main()
