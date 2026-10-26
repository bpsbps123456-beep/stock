"""deskx_expand.py 감사(렌즈: 대리 매핑·시점·자료), 2026-10-10 연구용 스크래치. 데스크·공유 연구 파일은 고치지 않는다.

1) T/off/없음 = 데스크 run_model(config.RULE 반도체 50%) 재현. 비교 기준은 대리 열을 붙이지 않은 새 deskx.Engine의 책.
   TK/off/없음 = deskx Engine(universe='union', semis 0.50)도 다시 잰다.
2) 세션 여러 개에서 PDF 원본 행으로 후보군·대리 매핑·기본 비중을 따로 다시 만들어 ExpandEngine.candidates와 맞춘다.
   PDF는 '세션보다 엄격히 앞선 날짜'만(독립 구현: max(d < session)).
   매핑은 AI_UNIVERSE를 직접 뒤집어 만든다(세부 섹터의 미국 상장 → 없으면 대분류의 미국 상장 → 없으면 버림).
   명세에 없는 9988.HK→BABA 연결(CROSS_LISTING)은 따로 표시한다.
3) 대리 가격: 새로 받은 6종목의 날짜 정렬·결측·분할 의심 점프, 모든 대리 종목의 120일 창 결측,
   보유 중 시가 결측(조용히 수익 0 처리) 여부를 13개 변형 경로 전부에서 센다.
4) 사후 위험: 실제로 산 대리 종목이 산 날 직전 PDF에서 TIME·KoAct에 있었는지, 대리 가산(이미 든 종목에 얹은 비중) 크기.
5) 민감도(감사용, 선택 후보 아님): 9988.HK 연결 제거, 대분류 대체(MLCC→TTMI·APH) 제거.
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py verify_expand_mapping.py
"""
import json, os, sys
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deskx_expand as X  # noqa: E402
from deskx import Engine, Params, START, PERIODS  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import run_model, research_key, iso_week  # noqa: E402
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402
from gbrc.etf.market import is_cash_holding, listing_market, yahoo_ticker  # noqa: E402
from gbrc.etf.research_data import exclusion_reason  # noqa: E402
from watchlist import AI_UNIVERSE  # noqa: E402

OUT = os.path.join(HERE, "../results/deskx/verify_expand_mapping.json")
FRONT, BACK, SEP, FULL = (p[0] for p in PERIODS)
SESSIONS = ["2026-01-05", "2026-02-17", "2026-03-30", "2026-04-20", "2026-05-18", "2026-05-26", "2026-06-08",
            "2026-08-17", "2026-10-12"]
report = {}


def pdf_before(dates, session):
    prior = [d for d in dates if d < session]
    return max(prior) if prior else None


def rows_of(snap):
    """(yahoo, market, weight) for priced, non-cash, non-excluded rows; independent of us_weights/non_us_weights."""
    out = defaultdict(float)
    mk = {}
    for r in snap["holdings"]:
        if is_cash_holding(r) or exclusion_reason(r):
            continue
        key = research_key(r.get("code"))
        m = listing_market(key)
        if m is None:
            raise AssertionError(f"unclassified row {r}")
        out[key] += float(r["weight"]) / 100.0
        mk[key] = m
    return out, mk


# AI_UNIVERSE 독립 역색인
SYM2 = {}
for top, g in AI_UNIVERSE.items():
    for sub, syms in g.items():
        for s in syms:
            SYM2[s] = (top, sub)


def is_us_symbol(s):
    return "." not in s and not s[:1].isdigit()


def my_proxies(key, cross=True, top_fallback=True):
    yt = yahoo_ticker(key)
    tries = [yt]
    if yt.endswith(".KS"):
        tries.append(yt[:-3] + ".KQ")
    if cross and yt == "9988.HK":
        tries = ["BABA"]
    hit = next((s for s in tries if s in SYM2), None)
    if hit is None:
        return [], "dropped"
    top, sub = SYM2[hit]
    us = [s for s in AI_UNIVERSE[top][sub] if is_us_symbol(s)]
    if us:
        return us, "sub"
    if not top_fallback:
        return [], "dropped"
    us = [s for ss in AI_UNIVERSE[top].values() for s in ss if is_us_symbol(s)]
    return (us, "top") if us else ([], "dropped")


def my_candidates(tsn, ksn, tdates, kdates, session, univ, proxies, cross=True, top_fallback=True):
    """yahoo ticker -> base weight."""
    tp, kp = pdf_before(tdates, session), pdf_before(kdates, session)
    tw, tm = rows_of(tsn[tp])
    base = {yahoo_ticker(k): w for k, w in tw.items() if tm[k] == "US" and w > 0}
    if univ == "TK":
        kw, km = rows_of(ksn[kp])
        assert all(m == "US" for m in km.values()), "KoAct non-US row"
        for k, w in kw.items():
            t = yahoo_ticker(k)
            if w > 0 and t not in base:
                base[t] = w
    if proxies:
        for k, w in tw.items():
            if tm[k] == "US":
                continue
            prox, _ = my_proxies(k, cross, top_fallback)
            for s in prox:
                base[s] = base.get(s, 0.0) + w / len(prox)
    return base, tp, kp


def main():
    E = X.ExpandEngine()
    plain = Engine()
    tsn = load_fund_snapshots("456600", "9999-12-31")
    ksn = load_fund_snapshots("2ETFQ1", "9999-12-31")
    tdates, kdates = sorted(tsn), sorted(ksn)
    G = dict(X.GRID)

    # ---------------------------------------------------------------- 1) 재현
    assert config.RULE["semis_cap"] == 0.50, config.RULE
    ref = np.asarray(run_model(plain.book, start=START).value)
    v_live, _ = E.run(G["T/off/m=none"])
    d1 = float(np.max(np.abs(v_live.to_numpy() - ref)))
    v_u, _ = plain.run(Params(universe="union", semis_cap=0.50))
    v_tk, _ = E.run(G["TK/off/m=none"])
    d2 = float(np.max(np.abs(v_u.to_numpy() - v_tk.to_numpy())))
    ev = plain.evaluate("live", pd.Series(ref, index=v_live.index))
    # 동결 가격판 vs 지금 데스크 캐시(읽기만): 동결 뒤 데스크 캐시가 바뀌었는지
    drift = {}
    for f in ("open", "adj_close"):
        a = pd.read_csv(os.path.join(X.HERE, f"../cache/deskx_prices/prices_{f}.csv"), index_col=0)
        b = pd.read_csv(config.CACHE / f"prices_{f}.csv", index_col=0)
        cols = [c for c in a.columns if c in b.columns]
        idx = [i for i in a.index if i in b.index and i >= "2025-06-01"]
        r = (a.loc[idx, cols] / b.loc[idx, cols] - 1).abs()
        drift[f] = {"max_rel": float(np.nanmax(r.to_numpy())), "rows_only_in_desk": len(set(b.index) - set(a.index)),
                    "frozen_last": a.index[-1], "desk_last": b.index[-1]}
    report["1_reproduction"] = {
        "T_off_none_vs_run_model(fresh plain book, RULE semis .50)": d1,
        "TK_off_none_vs_deskx_union": d2,
        "live_metrics": {k: round(ev[k]["수익"], 4) for k in (FRONT, BACK, SEP, FULL)} | {"MDD": round(ev[FULL]["낙폭"], 4)},
        "frozen_vs_current_desk_cache": drift}
    assert d1 < 1e-12 and d2 < 1e-12, (d1, d2)
    print("1) 재현", report["1_reproduction"])

    # ---------------------------------------------------------------- 2) 세션별 후보군 독립 재구성
    by_t = defaultdict(list)
    for k, t in E.ticker.items():
        by_t[t].append(k)
    dup = {t: ks for t, ks in by_t.items() if len(ks) > 1}
    sess_rep = {}
    for s in SESSIONS:
        rec = {}
        for name in ("T/off/m=none", "T/on/m=none", "TK/off/m=none", "TK/on/m=none"):
            p = G[name]
            mine, tp, kp = my_candidates(tsn, ksn, tdates, kdates, s, p.univ, p.proxies)
            theirs = {E.ticker[k]: v for k, v in E.candidates(p, s).items()}
            keys_eq = set(mine) == set(theirs)
            maxdiff = max(abs(mine.get(t, 0) - theirs.get(t, 0)) for t in set(mine) | set(theirs))
            assert keys_eq and maxdiff < 1e-12, (s, name, set(mine) ^ set(theirs), maxdiff)
            assert tp < s and (kp is None or kp < s)
            rec[name] = {"n": len(mine), "max_abs_diff": maxdiff}
        tw, tm = rows_of(tsn[tp])
        nonus = {yahoo_ticker(k): round(w, 4) for k, w in tw.items() if tm[k] != "US"}
        prox = {yahoo_ticker(k): my_proxies(k) for k, w in tw.items() if tm[k] != "US"}
        base_off, _, _ = my_candidates(tsn, ksn, tdates, kdates, s, "T", False)
        base_on, _, _ = my_candidates(tsn, ksn, tdates, kdates, s, "T", True)
        added = {t: round(base_on[t] - base_off.get(t, 0.0), 4) for t in base_on if base_on[t] - base_off.get(t, 0.0) > 1e-12}
        dropped_w = sum(w for k, w in tw.items() if tm[k] != "US" and not my_proxies(k)[0])
        sess_rep[s] = {"pdf_time": tp, "pdf_koact": kp, "pools": rec, "time_nonus": nonus,
                       "nonus_total": round(sum(nonus.values()), 4), "nonus_dropped_w": round(dropped_w, 4),
                       "proxy_added_T": added,
                       "proxy_added_to_already_held": sorted(t for t in added if t in base_off),
                       "cross_listing_used": "9988.HK" in nonus}
        print(f"2) {s} PDF T {tp} K {kp} 비미국 {sess_rep[s]['nonus_total']:.3f} 버림 {dropped_w:.3f} 대리 가산 {added}")
    # 매핑 표 전체: 독립 매핑 == 빌더 매핑
    mism = {}
    for key, info in E.proxy_info.items():
        mine, how = my_proxies(key)
        if mine != info["proxies"] or how != info["via"]:
            mism[key] = (mine, how, info)
    assert not mism, mism
    report["2_pools"] = {"sessions": sess_rep, "mapping_mismatch": mism, "duplicate_ticker_keys": dup,
                         "koact_non_us_rows": sum(1 for d in kdates for k, m in rows_of(ksn[d])[1].items() if m != "US"),
                         "spec_deviation": "9988.HK -> BABA via manual CROSS_LISTING (spec: no AI_UNIVERSE match -> drop)"}

    # ---------------------------------------------------------------- 3) 가격
    days = E.days
    k_first = days.index([d for d in days if d >= START][0]) - 1 - 120   # 첫 순위에 필요한 가장 이른 날
    proxies_all = sorted({s for pr in E.proxy_map.values() for s in pr})
    price = {}
    for s in proxies_all:
        key = E.proxy_key[s]
        o, a, c = E.col("opn", key), E.col("adj", key), E.col("cls", key)
        win = slice(k_first, len(days))
        nan_o = int(np.sum(~np.isfinite(o[win]))) if o is not None else None
        nan_a = int(np.sum(~np.isfinite(a[win]))) if a is not None else None
        jump = float(np.nanmax(np.abs(np.diff(np.log(o[win]))))) if o is not None else None
        # 종가/수정종가 비율은 배당 때만 조금 움직여야 한다(분할이면 크게 튄다)
        ratio = c[win] / a[win]
        rj = float(np.nanmax(np.abs(np.diff(np.log(ratio))))) if a is not None else None
        price[s] = {"key": key, "fetched": s in E.fetched, "nan_open": nan_o, "nan_adj": nan_a,
                    "max_abs_log_open_jump": round(jump, 3) if jump is not None else None,
                    "max_close_adj_ratio_step": round(rj, 4) if rj is not None else None}
    csv_dates = pd.read_csv(os.path.join(X.PROXY_CACHE, "prices_open.csv"), index_col=0).index.tolist()
    lo = days[k_first]
    align = {"book_days_missing_in_csv": [d for d in days if d >= lo and d not in csv_dates],
             "csv_dates_not_in_book": [d for d in csv_dates if d not in days],
             "book_last": days[-1], "csv_last": csv_dates[-1], "window_start": lo}
    print("3) 가격 정렬", align)
    for s, d in price.items():
        flag = (d["nan_open"] or d["nan_adj"] or (d["max_abs_log_open_jump"] or 0) > 0.35
                or (d["max_close_adj_ratio_step"] or 0) > 0.03)
        if flag or d["fetched"]:
            print("   ", s, d)

    # 모든 재조정 세션: 후보인데 가격이 없어 순위에서 빠진 종목(가장 넓은 TK/on 후보군)
    first = [d for d in days if d >= START][0]
    rebs = [d for i, d in enumerate(days) if d >= START and (d == first or iso_week(d) != iso_week(days[i - 1]))]
    unranked = defaultdict(lambda: {"weeks": 0, "max_base_w": 0.0, "src": None})
    for s in rebs:
        k = days.index(s)
        cand = E.candidates(G["TK/on/m=none"], s)
        src = dict(E._src)
        ranked = {r["key"] for r in E.ranking(G["TK/on/m=none"], k, s)}
        for key, w in cand.items():
            if key not in ranked:
                u = unranked[E.ticker[key]]
                u["weeks"] += 1
                u["max_base_w"] = max(u["max_base_w"], round(w, 4))
                u["src"] = src.get(key)
    report["3_prices"] = {"proxies": price, "alignment": align, "unranked_candidates_TK_on": dict(unranked)}
    print("   순위에서 빠진 후보(가격 이력 부족):", {t: (u["weeks"], u["max_base_w"], u["src"]) for t, u in unranked.items()})

    # 13개 경로: 보유 중 시가 결측(수익 0으로 조용히 처리)
    held_nan = {}
    runs = {}
    for name, p in list(X.GRID) + [X.POSTHOC]:
        E.log = []
        v, rows = E.run(p)
        runs[name] = (v, rows, E.log)
        E.log = None
        k0 = days.index(rows[0]["day"])
        bad = []
        for i in range(len(rows) - 1):
            k = k0 + i + 1
            for key, wt in rows[i]["w"].items():
                o = E.col("opn", key)
                if wt > 1e-12 and (o is None or not (np.isfinite(o[k]) and np.isfinite(o[k - 1]))):
                    bad.append((days[k], E.ticker[key], round(wt, 4)))
        held_nan[name] = bad
    report["3_prices"]["held_with_missing_open"] = held_nan
    print("   보유 중 시가 결측:", {n: len(b) for n, b in held_nan.items()})

    # ---------------------------------------------------------------- 4) 사후 위험
    def fund_w(sn, dates, session, t):
        p = pdf_before(dates, session)
        if p is None:
            return 0.0
        w, m = rows_of(sn[p])
        return round(sum(v for k, v in w.items() if yahoo_ticker(k) == t), 4)

    def ever_before(sn, dates, session, t):
        return any(sum(v for k, v in rows_of(sn[d])[0].items() if yahoo_ticker(k) == t) > 0 for d in dates if d < session)

    hind = {}
    for name, (v, rows, log) in runs.items():
        prev, ent = set(), []
        boost_w = []
        for e in log:
            held = {x for x, w in e["tgt"].items() if w > 1e-12}
            for x in held - prev:
                if e["src"].get(x) == "proxy":
                    t = E.ticker[x]
                    ent.append({"session": e["session"], "ticker": t, "weight": round(e["tgt"][x], 4),
                                "rank": e["rank"][x], "proxy_for": e["from"][x],
                                "TIME_w_prior_pdf": fund_w(tsn, tdates, e["session"], t),
                                "KoAct_w_prior_pdf": fund_w(ksn, kdates, e["session"], t),
                                "in_any_fund_2026_before": ever_before(tsn, tdates, e["session"], t)
                                or ever_before(ksn, kdates, e["session"], t)})
            # 대리 가산: 이미 펀드 몫으로 든 종목에 얹은 기본 비중 몫(그 주 목표 비중 중 대리 몫 비율 근사)
            for x in held:
                if e["src"].get(x) in ("time", "koact") and e["from"].get(x):
                    boost_w.append((e["session"], E.ticker[x]))
            prev = held
        hind[name] = {"proxy_entries": ent, "boosted_weeks": len(boost_w)}
    report["4_hindsight"] = hind
    for name, h in hind.items():
        if h["proxy_entries"]:
            print("4)", name, [(x["session"], x["ticker"], x["rank"], "T%.3f" % x["TIME_w_prior_pdf"],
                                "K%.3f" % x["KoAct_w_prior_pdf"]) for x in h["proxy_entries"]])

    # ---------------------------------------------------------------- 5) 민감도(감사용)
    sens = {}
    saved = {k: list(v) for k, v in E.proxy_map.items()}
    def metrics(v):
        r = E.evaluate("x", v)
        return {k: round(r[k]["수익"], 4) for k in (FRONT, BACK, SEP, FULL)} | {"MDD": round(r[FULL]["낙폭"], 4)}
    try:
        for label, drop in (("no_9988_BABA", ["9988 HK EQUITY"]), ("no_top_fallback", ["009150", "6981 JP EQUITY"])):
            for k in drop:
                E.proxy_map[k] = []
            for m in (1, 2, 7):
                name = f"T/on/m={'none' if m == 7 else m}"
                v, _ = E.run(G[name])
                sens[f"{name} [{label}]"] = metrics(v)
            E.proxy_map.update({k: list(saved[k]) for k in drop})
    finally:
        E.proxy_map.update(saved)
    for name in ("T/on/m=1", "T/on/m=2", "T/on/m=none", "T/off/m=none"):
        sens[name] = metrics(runs[name][0])
    report["5_sensitivity"] = sens
    for k, v in sens.items():
        print("5)", f"{k:<36}", v)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
