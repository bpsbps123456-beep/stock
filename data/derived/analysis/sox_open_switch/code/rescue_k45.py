"""BRIEF7 K4·K5 (구현 A) — 안 본 자료 2016-01-04~2022-12-30 Alpaca 1분봉에서 사용자 방향 x = 3% 사슬을 다시 만들어 판정 칸을 잰다.

정본: BRIEF7.md(후보·판정), BRIEF3.md(H0 사용자 방향), BRIEF4.md C3 H4(ev4_build.py 사슬·c3_hyp.py 조건).
사용: cd code && py rescue_k45.py → results/ev4_events_2016.parquet(+ _meta.json), results/rescue_k45.json

결과 보기 전에 정한 것(BRIEF7·오케스트레이터 지시 그대로)
- 자료: Data('alp1m_2016')를 한 번만 읽고, 비용표는 cost_table_2016.json을 엔진과 h0.sim_trade에 함께 넘긴다
  (ev4_build.build(costs=...)). 사건 표 = ev4_build 사슬 정의·조건 계산 그대로, 실행 날 = 2016-01-04~2022-12-30.
- K5: side = user(SOXL 오름 → SOXS 매수), x = 3%, r_w90(+1% 지정가, 10:59 봉 종가, 손절 없음) = h0.H0 단방향 단독 실행.
  하루 손익 = 그날 매매 곱 − 1, 사건 없는 날 0, 2016~2022 모든 거래일 평균.
  p = 날 묶음 부트스트랩(그 거래일 복원 추출 10,000회, seed 20261104) 평균 ≤ 0 비율.
- K4: 같은 사슬 사건 중 c10_wick ≥ 0.5 − 1e-9(c3_hyp H4와 같은 식). 하루 평균은 K5와 같은 방식.
  p = 같은 해의 조건 없는 사용자 x3 사건에서 그해 K4 사건 수만큼 비복원으로 뽑기를 10,000번(seed 20261104, 해 순서대로
  같은 rng) → 같은 하루 평균 → 관측 이상 비율. 같은 날 뽑힌 사건은 곱으로 묶는다(관측과 같은 식). 날 부트스트랩 p는 참고.
- 둘 다: 상위 2일(하루 손익 큰 두 날)을 표본에서 빼고 평균, 비용 2배(r_w90_c2 = 같은 결정 봉 반 호가 2배) 하루 평균,
  해마다 하루 평균(사건이 있는 해 중 양(+)인 해 수). Holm(m = 6)은 여섯 후보를 모은 쪽에서 한다(여기는 날 p).
- 채운 봉: 2016~2018 SOXS 장초 빈 분은 엔진이 앞 종가로 채운다. 산 종목(SOXS)의 체결 봉 또는 청산 봉이 Data.has = False인
  매매를 세고, 그 매매를 뺀 감도(K4 대조 모집단도 같이 뺌)를 낸다. 신호 봉(SOXL)이 채운 봉인 사건 수도 적는다.
점검: ev4_build 내부 점검(엔진 = sim_trade, h0.H0 비트 일치, 미래 참조), 엔진 날별 = 사건 표 날별 곱, 체결가에 2016 비용표가
들어갔는지, r_w90·r_w90_c2를 이 파일의 단순 재계산으로 다시 냄, 실행 뒤 h0.COSTS가 기본 표로 돌아왔는지.
"""
import hashlib
import json
import os
import sys
import time

os.environ["SOS_UNLOCK"] = "frozen"          # 2016~2022는 잠금 구간(2025·2026)과 겹치지 않는다
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
from lib import Data, ROOT, TOUCH, bucket, idx, load_costs  # noqa: E402
import ev4_build as EB        # noqa: E402
import h0 as H0M              # noqa: E402
from c3_hyp import day_arrays  # noqa: E402

RES = os.path.join(ROOT, "results")
SEED, NB, NR, EPS = 20261104, 10000, 10000, 1e-9
X, TGT = 0.03, 0.01
START, END = "2016-01-04", "2022-12-30"
COSTS_PATH = os.path.join(ROOT, "cost_table_2016.json")
OUT_EV = os.path.join(RES, "ev4_events_2016.parquet")
OUT_META = os.path.join(RES, "ev4_events_2016_meta.json")
OUT = os.path.join(RES, "rescue_k45.json")
CUT = idx("11:00")
T0 = time.time()


def fnum(v):
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def log(msg):
    print(f"{msg} ({time.time() - T0:.0f}s)", flush=True)


# ------------------------------------------------------------------ 통계

def boot(dly, ix):
    bd = dly[ix].mean(1)
    return {"p_one_sided": fnum((bd <= 0).mean()), "ci95": [fnum(np.percentile(bd, 2.5)), fnum(np.percentile(bd, 97.5))]}


def drop_top2(dly, days):
    top = np.argsort(-dly, kind="stable")[:2]
    keep = np.ones(len(dly), bool)
    keep[top] = False
    return {"days_dropped": [str(days[i].date()) for i in top], "values": [fnum(dly[i]) for i in top],
            "daily_mean": fnum(dly[keep].mean()), "days": int(keep.sum())}


def null_subset(pool, mask, ndays, rng):
    """해마다 pool(그해 조건 없는 사건)에서 그해 mask 사건 수만큼 비복원 NR번 → 하루 평균(NR,). 관측값도 같은 식으로 낸다."""
    tot = np.zeros(NR)
    obs = 0.0
    for y in sorted(pool["year"].unique()):
        sel = (pool["year"] == y).values
        P = pool[sel]
        n = int(mask[sel].sum())
        lr = np.log1p(P["r"].values / 100)
        dcode = pd.factorize(P["day"])[0]
        W = np.zeros((len(P), dcode.max() + 1))
        W[np.arange(len(P)), dcode] = lr
        obs += np.expm1(mask[sel].astype(float) @ W).sum() * 100
        if n == 0:
            continue
        if n < len(P):
            keys = rng.random((NR, len(P)))
            pick = np.argpartition(keys, n - 1, axis=1)[:, :n]
            M = np.zeros((NR, len(P)))
            M[np.arange(NR)[:, None], pick] = 1.0
        else:
            M = np.ones((NR, len(P)))
        tot += np.expm1(M @ W).sum(1) * 100
    return tot / ndays, obs / ndays


def summarize(sub, alld, ix, pool=None, pool_mask=None, label=""):
    """sub = 판정 칸 사건, alld = 2016~2022 모든 거래일(Timestamp). pool·pool_mask를 주면 K4식 같은 해 무작위 고르기 대조."""
    dly, S, N = day_arrays(sub, alld, "r_w90")
    c2, _, _ = day_arrays(sub, alld, "r_w90_c2")
    n = int(N.sum())
    act = N > 0
    out = {"label": label, "days": len(alld), "trades": n, "active_days": int(act.sum()),
           "daily_mean": fnum(dly.mean()), "trade_mean": fnum(S.sum() / n) if n else None,
           "trade_median": fnum(sub["r_w90"].median()) if n else None,
           "win_rate": fnum((sub["r_w90"] > 0).mean()) if n else None,
           "target_hit_rate": fnum(sub["hit"].mean()) if n else None,
           "worst_trade": fnum(sub["r_w90"].min()) if n else None, "best_trade": fnum(sub["r_w90"].max()) if n else None,
           "worst_day": fnum(dly.min()), "best_day": fnum(dly.max()),
           "boot_day": boot(dly, ix) if n else None,
           "drop_top2_days": drop_top2(dly, alld),
           "cost2x": {"daily_mean": fnum(c2.mean()), "trade_mean": fnum(sub["r_w90_c2"].mean()) if n else None,
                      "boot_day": boot(c2, ix) if n else None}}
    if pool is not None:
        nm, obs_log = null_subset(pool, pool_mask, len(alld), np.random.default_rng(SEED))
        assert abs(obs_log - dly.mean()) < 1e-9, (label, obs_log, dly.mean())
        k = int((nm >= dly.mean() - 1e-12).sum())
        out["ctrl_same_year_subset"] = {"p_one_sided": k / NR, "p_plus1": (k + 1) / (NR + 1), "null_mean": fnum(nm.mean()),
                                        "null_q95": fnum(np.percentile(nm, 95)), "pool_trades": int(len(pool)),
                                        "pool_daily_mean": fnum(day_arrays(pool.rename(columns={"r": "r_w90"}), alld, "r_w90")[0].mean())}
    by = {}
    for y in sorted({d.year for d in alld}):
        dy = [d for d in alld if d.year == y]
        sy = sub[sub["year"] == y]
        a, Sy, Ny = day_arrays(sy, dy, "r_w90")
        b, _, _ = day_arrays(sy, dy, "r_w90_c2")
        ny = int(Ny.sum())
        by[str(y)] = {"days": len(dy), "trades": ny, "active_days": int((Ny > 0).sum()), "daily_mean": fnum(a.mean()),
                      "trade_mean": fnum(Sy.sum() / ny) if ny else None, "win_rate": fnum((sy["r_w90"] > 0).mean()) if ny else None,
                      "target_hit_rate": fnum(sy["hit"].mean()) if ny else None, "cost2x_daily_mean": fnum(b.mean())}
    with_ev = [y for y in by if by[y]["trades"] > 0]
    pos = [y for y in with_ev if (by[y]["daily_mean"] or 0) > 0]
    out["by_year"] = by
    out["years_positive"] = f"{len(pos)}/{len(with_ev)}"
    p_main = out["ctrl_same_year_subset"]["p_one_sided"] if pool is not None else (out["boot_day"] or {}).get("p_one_sided")
    out["p_main"] = p_main
    out["criteria"] = {"1_mean_gt0": bool((out["daily_mean"] or 0) > 0),
                       "2_p_raw_le_0.05": bool(p_main is not None and p_main <= 0.05),
                       "2_p_bonferroni6_le_0.05": bool(p_main is not None and 6 * p_main <= 0.05),
                       "3_drop_top2_gt0": bool((out["drop_top2_days"]["daily_mean"] or 0) > 0),
                       "4_cost2x_gt0": bool((out["cost2x"]["daily_mean"] or 0) > 0),
                       "5_half_years_positive": bool(len(with_ev) > 0 and 2 * len(pos) >= len(with_ev))}
    return out


# ------------------------------------------------------------------ 점검용 단순 재계산

def indep_ret(D, costs, d, f, cm):
    """체결 봉 f 종가 + 반 호가 × cm, +1% 지정가(고가 ≥ 목표 × 1.0002, 체결 다음 봉부터), 못 닿으면 10:59 봉 종가 − 반 호가 × cm."""
    A = D.A["SOXS"]
    hs = costs["SOXS"]
    px = A["Close"][d, f] * (1 + hs[bucket(f)] * cm)
    tg = px * (1 + TGT)
    hh = A["High"][d, f + 1:CUT]
    k = np.nonzero(hh >= tg * TOUCH)[0]
    if len(k):
        return (tg / px - 1) * 100, f + 1 + int(k[0])
    return (A["Close"][d, CUT - 1] * (1 - hs[bucket(CUT - 1)] * cm) / px - 1) * 100, CUT - 1


def main():
    costs = json.load(open(COSTS_PATH, encoding="utf-8"))
    default_costs = load_costs()
    assert costs != default_costs, "2016 비용표가 기본 표와 같다"
    D = Data("alp1m_2016")
    days = EB.run_days(D, START, END)
    log(f"자료 {D.days[0]}~{D.days[-1]} {len(D.days)}일, 실행 {D.days[days[0]]}~{D.days[days[-1]]} {len(days)}일")
    if "--reuse" in sys.argv:             # 사건 표를 이미 만들었으면 다시 안 만든다(엔진 날별·점검은 같이 저장한 것을 읽음)
        ev = pd.read_parquet(OUT_EV)
        bchecks = json.load(open(OUT_META, encoding="utf-8"))["checks"]
        dailies = {"user_0.03": pd.read_csv(OUT_EV.replace(".parquet", "_engine_daily_user_x3.csv"), parse_dates=["day"])}
    else:
        ev, bchecks, dailies = EB.build(D, days, costs=costs, out=OUT_EV, meta_path=OUT_META, src="alp1m_2016", h0_ref=False,
                                        extra_meta={"costs": "cost_table_2016.json", "run_days": [START, END, len(days)],
                                                    "brief": "BRIEF4.md '사건과 결과'·'조건 목록' 사슬 그대로, BRIEF7 K4·K5 새 자료 재실행"})
        dailies["user_0.03"].to_csv(OUT_EV.replace(".parquet", "_engine_daily_user_x3.csv"), index=False, float_format="%.17g")
    assert H0M.COSTS == default_costs, "h0.COSTS가 기본 표로 돌아오지 않았다"
    log(f"사건 표 {len(ev)}행")

    alld = [pd.Timestamp(D.days[k]) for k in days]
    u3 = ev[(ev["side"] == "user") & np.isclose(ev["x"], X)].copy().reset_index(drop=True)
    checks = {}
    # 엔진 날별 = 사건 표 날별 곱
    eng = dailies["user_0.03"]
    dly5, _, N5 = day_arrays(u3, alld, "r_w90")
    assert len(eng) == len(alld) and (pd.to_datetime(eng["day"]).values == np.array(alld, dtype="datetime64[ns]")).all()
    checks["engine_daily_vs_events_max_abs_diff"] = float(np.max(np.abs(eng["ret"].values * 100 - dly5)))
    checks["engine_trades_vs_events"] = [int(eng["n"].sum()), int(N5.sum())]
    checks["h0_bitmatch_user_0.03"] = bchecks["h0_bitmatch"]["user_0.03"]
    checks["engine_vs_sim_max_abs_diff_user_0.03"] = bchecks["engine_vs_sim_max_abs_diff"]["user_0.03"]
    checks["leak_test"] = {k: bchecks["leak_test"][k] for k in ("n", "mismatch_events")}
    checks["nan_r_w90"] = int(u3["r_w90"].isna().sum())
    # 2016 비용표가 체결가에 들어갔는지
    cl = D.A["SOXS"]["Close"]
    want = np.array([cl[d, f] * (1 + costs["SOXS"][bucket(f)]) for d, f in zip(u3["d"], u3["fill_i"])])
    checks["entry_px_uses_2016_costs_max_rel_diff"] = float(np.max(np.abs(u3["entry_px"].values / want - 1)))
    old = np.array([cl[d, f] * (1 + default_costs["SOXS"][bucket(f)]) for d, f in zip(u3["d"], u3["fill_i"])])
    checks["entry_px_vs_default_costs_max_rel_diff"] = float(np.max(np.abs(u3["entry_px"].values / old - 1)))
    # 단순 재계산
    r1 = np.array([indep_ret(D, costs, d, f, 1.0) for d, f in zip(u3["d"], u3["fill_i"])], dtype=object)
    r2 = np.array([indep_ret(D, costs, d, f, 2.0)[0] for d, f in zip(u3["d"], u3["fill_i"])])
    checks["indep_r_w90_max_abs_diff"] = float(np.max(np.abs(np.array([a for a, _ in r1], float) - u3["r_w90"].values)))
    checks["indep_exit_bar_mismatch"] = int((np.array([b for _, b in r1]) != u3["exit_i"].values).sum())
    checks["indep_r_w90_c2_max_abs_diff"] = float(np.max(np.abs(r2 - u3["r_w90_c2"].values)))
    log(json.dumps(checks, ensure_ascii=False, default=str))
    assert checks["engine_daily_vs_events_max_abs_diff"] < 1e-9 and checks["indep_r_w90_max_abs_diff"] < 1e-9
    assert checks["indep_r_w90_c2_max_abs_diff"] < 1e-9 and checks["entry_px_uses_2016_costs_max_rel_diff"] < 1e-12
    assert checks["h0_bitmatch_user_0.03"]["match"] and checks["leak_test"]["mismatch_events"] == 0

    # 채운 봉 표시
    hs_ = D.has["SOXS"]
    u3["fill_filled"] = ~hs_[u3["d"].values, u3["fill_i"].values]
    u3["exit_filled"] = ~hs_[u3["d"].values, u3["exit_i"].values]
    u3["sig_filled"] = ~D.has["SOXL"][u3["d"].values, u3["dec_i"].values]
    u3["any_filled"] = u3["fill_filled"] | u3["exit_filled"]
    k4m = (u3["c10_wick"] >= 0.5 - EPS).values

    def filled_counts(s):
        return {"trades": int(len(s)), "fill_bar_filled": int(s["fill_filled"].sum()), "exit_bar_filled": int(s["exit_filled"].sum()),
                "fill_or_exit_filled": int(s["any_filled"].sum()), "signal_bar_filled_SOXL": int(s["sig_filled"].sum()),
                "by_year_fill_or_exit": {str(y): int(g["any_filled"].sum()) for y, g in s.groupby("year")},
                "by_year_trades": {str(y): int(len(g)) for y, g in s.groupby("year")},
                "filled_exit_why": {str(k): int(v) for k, v in s.loc[s["exit_filled"], "why"].value_counts().items()}}

    ix = np.random.default_rng(SEED).integers(0, len(alld), (NB, len(alld)))
    pool = u3[["day", "year"]].assign(r=u3["r_w90"].values)
    K5 = summarize(u3, alld, ix, label="K5 사용자 방향 x = 3% 단독")
    K4 = summarize(u3[k4m], alld, ix, pool=pool, pool_mask=k4m, label="K4 H4 윗꼬리 ≥ 0.5")
    log(f"K5 하루 {K5['daily_mean']:.4f} p {K5['p_main']}  K4 하루 {K4['daily_mean']:.4f} n {K4['trades']} p {K4['p_main']}")
    # 감도: 체결·청산 봉이 채운 봉인 매매 빼기(K4 대조 모집단도 같이 뺌)
    keep = ~u3["any_filled"].values
    u3k = u3[keep]
    poolk = pool[keep]
    K5s = summarize(u3k, alld, ix, label="K5 채운 봉 매매 뺌")
    K4s = summarize(u3k[k4m[keep]], alld, ix, pool=poolk, pool_mask=k4m[keep], label="K4 채운 봉 매매 뺌")
    K5["filled_bars"], K4["filled_bars"] = filled_counts(u3), filled_counts(u3[k4m])
    K5["excl_filled"], K4["excl_filled"] = K5s, K4s
    # 2026 근사 결과(c3.json, 같은 정의·2025~26 비용표) — 결론 합치기용 참고
    try:
        c3 = json.load(open(os.path.join(RES, "c3.json"), encoding="utf-8"))
        h4, b = c3["hypotheses"]["H4"], c3["baseline"]["user_all_2026"]
        K4["ref_2026"] = {k: h4["stats_2026"].get(k) for k in ("trades", "daily_mean", "trade_mean", "daily_p_one_sided")} | \
            {"holm10_p": h4["holm_p"], "random_subset_p_posthoc": (h4.get("posthoc_robustness") or {}).get("null_random_subset_p")}
        K5["ref_2026"] = {k: b.get(k) for k in ("trades", "daily_mean", "trade_mean", "daily_p_one_sided")}
    except (OSError, KeyError) as e:
        K4["ref_2026"] = K5["ref_2026"] = f"읽기 실패 {e}"
    tl = u3.loc[k4m, ["day", "dec_close", "entry", "exit", "c10_wick", "r_w90", "r_w90_c2", "why", "any_filled"]].copy()
    tl["day"] = tl["day"].dt.strftime("%Y-%m-%d")
    K4["trades_list"] = json.loads(tl.to_json(orient="records", force_ascii=False))

    out = {"brief": "BRIEF7.md K4·K5 (구현 A)",
           "data": f"Data('alp1m_2016') {D.days[0]}~{D.days[-1]}, 실행 {START}~{END} {len(alld)}거래일, 비용표 cost_table_2016.json",
           "costs_2016": costs, "units": "% (1.0 = 1%)",
           "settings": {"seed": SEED, "boot": NB, "null_draws": NR, "x": X, "target": TGT, "window": "w90(10:59 봉 종가 청산)",
                        "K4_condition": "c10_wick >= 0.5 - 1e-9", "drop_top2": "하루 손익 큰 두 날을 표본에서 뺌(stage5 drop_top5와 같은 방식)",
                        "K4_null": "해마다 그해 조건 없는 사용자 x3 사건에서 그해 K4 수만큼 비복원, 같은 날은 곱, 하루 평균 ≥ 관측 비율",
                        "K5_p": "2016~2022 거래일 복원 추출 하루 평균 ≤ 0 비율"},
           "code_sha256_16": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16],
           "ev4_build_sha256_16": hashlib.sha256(open(EB.__file__, "rb").read()).hexdigest()[:16],
           "events_sha256_16": hashlib.sha256(open(OUT_EV, "rb").read()).hexdigest()[:16],
           "n_events_user_x3_by_year": {str(y): int(v) for y, v in u3.groupby("year").size().items()},
           "checks": checks, "K5": K5, "K4": K4,
           "notes": ["사건 표는 조건 없는 사슬(한 번에 하나, 청산 뒤 L 다시 잼)의 사건이다. K4는 그 사건을 거른 것이라 조건 단독 실행과 다를 수 있다(c3_hyp 한계와 같음).",
                     "Holm m = 6은 여섯 후보를 모은 쪽에서 계산한다. criteria의 2_p_raw는 보정 전, 2_p_bonferroni6은 가장 엄격한 경우다.",
                     "r_w90_c2는 같은 결정 봉에 반 호가만 2배로 다시 잰 값(사슬은 기본 비용 그대로)."]}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    log(f"저장 {OUT}")


if __name__ == "__main__":
    main()
