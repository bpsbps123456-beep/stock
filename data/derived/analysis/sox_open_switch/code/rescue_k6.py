"""K6 되살리기(BRIEF7.md) — F1-A(BRIEF2)를 안 본 2016~2022 Alpaca 1분봉에서 한 번 잰다. 구현 A.

후보(얼린 그대로): fam_F1.make({"N":1,"entry":"confirm","tgt":2,"stop":None,"rev":False,"filt":None}), 마감 actual.
자료: Data('alp1m_2016') 2016-01-04~2022-12-30 모든 날(잠금 구간 아님, days= 전체), 비용표 cost_table_2016.json.
판정 칸(BRIEF7): ① 평균 > 0 ② 날 묶음 부트스트랩 단측 p(10,000회, seed 20261104; Holm m = 6은 묶음 전체에서)
  ③ 상위 2일 빼도 평균 > 0 ④ 비용 2배(cost_mult=2) 평균 > 0 ⑤ 해 절반 이상(7해 중 4해 이상) 양.
추가(설명·감도): 표준시 날(10:00 마감)·서머타임 날(11:00) 나눔, 체결·청산 봉이 앞 종가로 채운 봉(has=False)인
  매매 수와 그 매매를 뺀 감도(그날 빼기 / 그 매매만 0으로), 대조 늘 SOXL·늘 SOXS·거울(설명용).
lib.py·fam_F1.py 계산은 고치지 않는다(불러 쓰기만).
실행: py rescue_k6.py → results/rescue_k6.json, results/rescue_k6_daily.csv, results/rescue_k6_trades.csv
"""
import hashlib, json, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fam_F1  # noqa: E402
from lib import Data, Engine, cutoff_idx, idx  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
SRC = "alp1m_2016"
COST = os.path.join(ROOT, "cost_table_2016.json")
PARAMS = {"N": 1, "entry": "confirm", "tgt": 2, "stop": None, "rev": False, "filt": None}
SEED, NBOOT = 20261104, 10000


def boot_p(r, seed=SEED, nboot=NBOOT):
    """날 묶음 부트스트랩: 날을 복원 추출한 평균이 0 이하인 비율(단측 p)과 95% 구간. r은 %."""
    rng = np.random.default_rng(seed)
    n = len(r)
    bs = np.empty(nboot)
    for a in range(0, nboot, 1000):
        b = min(nboot, a + 1000)
        bs[a:b] = r[rng.integers(0, n, (b - a, n))].mean(1)
    return float((bs <= 0).mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def stats(daily, seed=SEED, with_boot=True):
    r = daily["ret"].values * 100
    act = daily["n"].values > 0
    out = {"days": int(len(r)), "active_days": int(act.sum()), "trades": int(daily["n"].sum()),
           "mean_pct": float(r.mean()) if len(r) else None,
           "median_pct": float(np.median(r)) if len(r) else None,
           "median_active_pct": float(np.median(r[act])) if act.any() else None,
           "sd_pct": float(r.std(ddof=1)) if len(r) > 1 else None,
           "hit_active": float((r[act] > 0).mean()) if act.any() else None,
           "worst_pct": float(r.min()) if len(r) else None, "best_pct": float(r.max()) if len(r) else None,
           "sum_pct": float(r.sum()), "compound_pct": float((np.prod(1 + r / 100) - 1) * 100)}
    if len(r) > 2:
        srt = np.sort(r)
        out["drop_top2_mean_pct"] = float(srt[:-2].mean())
        out["top2_days"] = [{"day": str(daily["day"].iloc[k]), "ret_pct": float(r[k])} for k in np.argsort(r)[-2:][::-1]]
        out["t"] = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if r.std() > 0 else 0.0
        eq = np.cumprod(1 + r / 100)
        out["max_dd_pct"] = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
        if with_boot:
            out["p_one_sided"], out["ci95"] = boot_p(r, seed)
    return out


def by_year(daily):
    yr = pd.to_datetime(daily["day"]).dt.year.values
    out = {}
    for y in sorted(set(yr)):
        m = yr == y
        r = daily["ret"].values[m] * 100
        act = daily["n"].values[m] > 0
        out[str(y)] = {"days": int(m.sum()), "active_days": int(act.sum()), "mean_pct": float(r.mean()),
                       "hit_active": float((r[act] > 0).mean()) if act.any() else None,
                       "compound_pct": float((np.prod(1 + r / 100) - 1) * 100), "worst_pct": float(r.min())}
    pos = sum(1 for v in out.values() if v["active_days"] > 0 and v["mean_pct"] > 0)
    have = sum(1 for v in out.values() if v["active_days"] > 0)
    return out, pos, have


def flag_trades(D, tr):
    """체결 봉·청산 봉이 그 종목의 실제 봉(has)인지. 매매 기록 시각 = 체결 봉 시작 시각(lib)."""
    if not len(tr):
        return tr.assign(entry_filled=[], exit_filled=[])
    ef, xf = [], []
    for day, sym, en, ex in zip(tr["day"], tr["sym"], tr["entry"], tr["exit"]):
        d = D.pos[day]
        ef.append(not bool(D.has[sym][d, idx(en)]))
        xf.append(not bool(D.has[sym][d, idx(ex)]))
    return tr.assign(entry_filled=ef, exit_filled=xf)


def drop_filled(daily, tr):
    """채운 봉 매매를 뺀 두 감도: (a) 그런 매매가 있는 날을 표본에서 뺀다 (b) 그 매매만 0으로 둔다(남은 매매의 곱)."""
    bad = tr[tr.entry_filled | tr.exit_filled]
    bad_days = set(bad["day"])
    a = daily[~daily["day"].isin(bad_days)].reset_index(drop=True)
    good = tr[~(tr.entry_filled | tr.exit_filled)]
    g = good.groupby("day")["ret"].apply(lambda x: float(np.prod(1 + x.values) - 1))
    gn = good.groupby("day").size()
    b = daily[["day"]].copy()
    b["ret"] = [g.get(dd, 0.0) for dd in b["day"]]
    b["n"] = [int(gn.get(dd, 0)) for dd in b["day"]]
    return a, b, len(bad), len(bad_days)


def season_split(daily):
    summer = np.array([cutoff_idx(d) == idx("11:00") for d in daily["day"]])
    out = {}
    for nm, m in (("summer_11h", summer), ("winter_10h", ~summer)):
        sub = daily[m].reset_index(drop=True)
        s = stats(sub)
        y, pos, have = by_year(sub)
        out[nm] = {**s, "years_positive": f"{pos}/{have}", "by_year_mean_pct": {k: round(v["mean_pct"], 4) for k, v in y.items()}}
    return out


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    t0 = time.time()
    D = Data(SRC)
    costs = json.load(open(COST, encoding="utf-8"))
    print(f"Data {time.time() - t0:.0f}s, 날 {len(D.days)} {D.days[0]}~{D.days[-1]}", flush=True)
    all_days = list(range(len(D.days)))
    half_days = int((D.rth_end != idx("16:00")).sum())

    runs = {}
    for nm, kw, mk in (("base", {}, fam_F1.make(PARAMS)),
                       ("cost2x", {"cost_mult": 2.0}, fam_F1.make(PARAMS)),
                       ("mirror", {}, fam_F1.make({**PARAMS, "mirror": True})),
                       ("always_SOXL", {}, lambda: fam_F1.Always(1)),
                       ("always_SOXS", {}, lambda: fam_F1.Always(-1))):
        E = Engine(D, costs=costs, scheme="actual", **kw)
        daily, tr = E.run(mk, days=all_days)
        runs[nm] = (daily, flag_trades(D, tr))
        print(nm, f"{time.time() - t0:.0f}s", round(daily["ret"].mean() * 100, 4), flush=True)

    daily, tr = runs["base"]
    d2, tr2 = runs["cost2x"]
    nan_days = int(daily["ret"].isna().sum())
    assert nan_days == 0 and int(d2["ret"].isna().sum()) == 0, "NaN 수익 날"
    assert np.allclose(daily.groupby("day")["n"].sum().values, tr.groupby("day").size().reindex(daily["day"]).fillna(0).values)

    s = stats(daily)
    s2 = stats(d2)
    yr, pos, have = by_year(daily)
    yr2, pos2, have2 = by_year(d2)
    s["p_one_sided_seed20261029"] = boot_p(daily["ret"].values * 100, seed=20261029)[0]

    # 채운 봉 매매
    fa, fb, nbad, nbad_days = drop_filled(daily, tr)
    fa2, fb2, nbad2, _ = drop_filled(d2, tr2)
    ya, posa, havea = by_year(fa)
    filled = {
        "trades_total": int(len(tr)),
        "trades_entry_filled": int(tr.entry_filled.sum()), "trades_exit_filled": int(tr.exit_filled.sum()),
        "trades_any_filled": nbad, "days_with_filled_trade": nbad_days,
        "any_filled_by_year": {str(k): int(v) for k, v in
                               tr[tr.entry_filled | tr.exit_filled].groupby(pd.to_datetime(tr["day"]).dt.year).size().items()},
        "any_filled_by_sym": {k: int(v) for k, v in tr[tr.entry_filled | tr.exit_filled]["sym"].value_counts().items()},
        "filled_trade_mean_pct": float(tr[tr.entry_filled | tr.exit_filled]["ret"].mean() * 100) if nbad else None,
        "unfilled_trade_mean_pct": float(tr[~(tr.entry_filled | tr.exit_filled)]["ret"].mean() * 100),
        "drop_days": {**stats(fa), "years_positive": f"{posa}/{havea}",
                      "by_year_mean_pct": {k: round(v["mean_pct"], 4) for k, v in ya.items()},
                      "cost2x_mean_pct": stats(fa2, with_boot=False)["mean_pct"]},
        "zero_trade": {**stats(fb), "cost2x_mean_pct": stats(fb2, with_boot=False)["mean_pct"]},
        "cost2x_trades_any_filled": nbad2}
    # SOXL 결정 봉(09:30 봉) 채움 여부 — 설명용
    soxl_open_missing = int((~D.has["SOXL"][:, idx("09:30")]).sum())
    soxs_open_missing = int((~D.has["SOXS"][:, idx("09:30")]).sum())

    exits = tr["why"].value_counts().to_dict()
    syms = tr["sym"].value_counts().to_dict()
    per_why = {k: float(g["ret"].mean() * 100) for k, g in tr.groupby("why")}
    per_sym = {k: float(g["ret"].mean() * 100) for k, g in tr.groupby("sym")}

    judge = {"1_mean_gt0": s["mean_pct"] > 0,
             "2_p_raw": s["p_one_sided"], "2_p_le_0.05_over_6(Holm 첫 단계)": s["p_one_sided"] <= 0.05 / 6,
             "2_p_le_0.05(Holm 마지막 단계, 묶음 전체 보정은 상위에서)": s["p_one_sided"] <= 0.05,
             "3_drop_top2_gt0": s["drop_top2_mean_pct"] > 0,
             "4_cost2x_gt0": s2["mean_pct"] > 0,
             "5_years_half_positive": pos >= have / 2, "years_positive": f"{pos}/{have}"}
    judge["all_but_holm_pass"] = bool(judge["1_mean_gt0"] and judge["3_drop_top2_gt0"] and judge["4_cost2x_gt0"]
                                      and judge["5_years_half_positive"])

    out = {"bundle": "K6", "impl": "A", "brief": "BRIEF7.md", "candidate": "F1-A (BRIEF2)", "params": PARAMS,
           "factory": 'fam_F1.make({"N":1,"entry":"confirm","tgt":2,"stop":None,"rev":False,"filt":None})',
           "engine": "lib.Engine fill=close, scheme=actual, costs=cost_table_2016.json, 1분 지연 다음 봉 종가",
           "data": {"src": SRC, "days": len(D.days), "first": str(D.days[0]), "last": str(D.days[-1]),
                    "half_days_13h": half_days, "soxl_0930_bar_missing_days": soxl_open_missing,
                    "soxs_0930_bar_missing_days": soxs_open_missing},
           "costs": costs, "seed": SEED, "nboot": NBOOT,
           "base": {**s, "years_positive": f"{pos}/{have}"}, "by_year": yr,
           "cost2x": {**s2, "years_positive": f"{pos2}/{have2}",
                      "by_year_mean_pct": {k: round(v["mean_pct"], 4) for k, v in yr2.items()}},
           "season": season_split(daily), "season_cost2x": {k: {"days": v["days"], "mean_pct": v["mean_pct"]}
                                                            for k, v in season_split(d2).items()},
           "exit_mix": {"why": {k: int(v) for k, v in exits.items()}, "by_sym": {k: int(v) for k, v in syms.items()},
                        "trade_mean_by_why_pct": per_why, "trade_mean_by_sym_pct": per_sym},
           "filled_bars": filled,
           "controls_explain_only": {nm: {"mean_pct": float(runs[nm][0]["ret"].mean() * 100),   # NaN(그날 09:31 전 봉 없음) 날은 뺀 평균
                                          "nan_days": int(runs[nm][0]["ret"].isna().sum()),
                                          "by_year_mean_pct": {str(y): round(float(g.mean() * 100), 4) for y, g in
                                                               runs[nm][0]["ret"].groupby(pd.to_datetime(runs[nm][0]["day"]).dt.year)}}
                                     for nm in ("mirror", "always_SOXL", "always_SOXS")},
           "judge": judge,
           "code_sha256_16": {f: sha16(os.path.join(HERE, f)) for f in ("lib.py", "fam_F1.py", "rescue_k6.py")},
           "cost_table_sha256_16": sha16(COST), "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "rescue_k6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    dd = daily.rename(columns={"ret": "ret", "n": "n"}).copy()
    dd["ret_cost2x"] = d2["ret"].values
    dd["season"] = ["summer_11h" if cutoff_idx(x) == idx("11:00") else "winter_10h" for x in dd["day"]]
    bad_days = set(tr[tr.entry_filled | tr.exit_filled]["day"])
    dd["filled_trade"] = dd["day"].isin(bad_days)
    dd.to_csv(os.path.join(RES, "rescue_k6_daily.csv"), index=False, encoding="utf-8-sig")
    tr.to_csv(os.path.join(RES, "rescue_k6_trades.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps({"base": {k: s[k] for k in ("days", "active_days", "mean_pct", "p_one_sided", "drop_top2_mean_pct", "hit_active")},
                      "cost2x": s2["mean_pct"], "years": {k: round(v["mean_pct"], 3) for k, v in yr.items()},
                      "season": {k: (v["days"], round(v["mean_pct"], 3), v["p_one_sided"]) for k, v in out["season"].items()},
                      "filled": {k: filled[k] for k in ("trades_any_filled", "days_with_filled_trade")},
                      "filled_drop_days_mean": filled["drop_days"]["mean_pct"], "judge": judge}, ensure_ascii=False, indent=1,
                     default=str))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
