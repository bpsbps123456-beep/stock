"""BRIEF4 C2 — 모형 전방 예측(고정 모형 2 × 학습 창 3, Holm m = 6).

무엇을 재나 (BRIEF4 'C2' 절 그대로, 결과 보기 전에 고정)
- 사건: results/ev4_events.parquet 5,194건(면 2 × 문턱 4, 2023-01-03~2026-10-28).
- 특징: 조건 28개 + 면 + 문턱.
  수치 조건 24개는 c##_* 값 그대로. 범주 조건은 지시 변수: c09 RSI 다이버전스(0/1), c14 최고 종가(0/1),
  c25 요일(월~금 5개 지시 변수), c28 순번(≥2 지시 변수, 수준 '1'/'≥2'). 면 = 거울 지시 변수, 문턱 = x(%).
- 목표: r_w90(%, 1.0 = 1%).
- 모형(고정): (a) ridge = 결측 대치 → StandardScaler → Ridge(alpha=10)
              (b) hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                  min_samples_leaf=50, l2_regularization=1.0, random_state=0), 결측은 그대로(HGB 내장 처리).
- 학습 창(고정): w2023 = 2023-01부터, w2025 = 2025-01부터, w2026 = 2026-01부터(2026-03부터 예측). 모두 늘려 감.
- 매달 m에 (창 시작 ≤ 사건 달 < m)인 사건으로 학습 → m 사건을 예측. 예측 > 0.05(%)인 사건을 산다.
- 겹침: 한 번에 한 종목(면·문턱 가리지 않음). 날마다 고른 사건을 결정 봉 순(같은 봉이면 예측 큰 것, 그다음 x 작은 것)으로
  보고, 앞 매매 청산 봉 이하에서 결정한 사건은 건너뛴다(엔진 '청산 봉에서는 새로 안 들어감'과 같다: dec_i > 앞 exit_i).
- 하루 수익 = 그날 매매 (1 + r/100)의 곱 − 1, 매매 없는 날 0. 날 = Data('alp1m_long').days 중 예측한 달의 거래일 전부.
- 통계(lib.summarize, 날 부트스트랩 10,000회 seed 20261102): 하루 평균, 95% 구간, 단측 p(부트스트랩 평균 ≤ 0 비율).
  두 반(1~6월·7~10월; w2026은 3~6월·7~10월), 가장 좋은 달(날 수익 합 최대) 뺀 평균, 비용 2배(r_w90_c2, 같은 고르기).
- 대조: (1) 사건 전부(같은 겹침 규칙, 같은 봉이면 x 작은 것) (2) 무작위 1,000회(seed 20261102): 날마다 모형이 고른 수
  k_d(겹침 거르기 전)만큼 그날 사건에서 비복원 무작위로 골라 같은 겹침 규칙을 쓴다. 모형 평균이 무작위 평균 분포의 몇 분위인지.
- Holm: 2026 6설정의 단측 부트스트랩 p를 step-down 보정(m = 6).
- 판정(BRIEF4 C2 + 세 갈래 공통 추가 조건):
  통과 = 95% 하한 > 0 & 평균 > 무작위 95% 분위 & Holm p ≤ 0.05
         & 두 반 모두 > 0 & 가장 좋은 달 빼도 > 0 & 비용 2배 > 0.
  앞 세 개만 맞으면 'C2 기준만 통과'로 따로 적는다.
- 보조(판정 밖): 2025 각 달(w2023 = 2025-01~12, w2025 = 2025-01부터 늘려 감이라 2025-03~12 예측, w2026은 해당 없음),
  2024 각 달(w2023만, 2023 사건으로 시작). 같은 절차·같은 대조.
- 특징 중요도(hgb만): 2026 매달 모형을 그달 사건(표본 밖)에서 조건 묶음(요일 5개 등은 함께)을 섞어 MSE 증가를 잰다
  (10회, seed 0). 달별 증가 × 사건 수를 더해 사건 수로 나눈다. ridge는 마지막 달(2026-10) 모형의 표준화 계수를 적는다.

결과 보기 전에 정한 구현 선택(BRIEF4에 적혀 있지 않은 부분)
1. ridge 결측: 학습 구간 중앙값으로 채우고, 표 전체에서 결측이 있는 조건마다 결측 지시 변수를 더한다(c06 첫 5분은 결정 봉
   < 09:34면 구조적 결측 281건, c21·c27은 첫 60거래일). 학습 구간에서 열 전체가 결측이면 0으로 채운다.
2. 문턱은 수치 x(2.5·3·3.5·4) 하나로 넣는다.
3. 무작위의 '같은 수'는 겹침 거르기 전 고른 수다(같은 절차를 무작위 고르기에 그대로 적용).
4. 같은 봉 동률은 같은 면·같은 진입이라 손익이 같다(사건 표에서 면이 다른 같은 봉 동률 0건, 같은 봉 동률의 r_w90 차 0건).
5. 사건 정의의 한계: 면·문턱마다 앞 사건 청산 뒤 L을 다시 잰다. 모형이 앞 사건을 건너뛰어도 뒤 사건은 그 사슬대로다.

사용: cd code && py c2_wf.py → results/c2.json, results/c2_trades_<ridge|hgb>_<w2023|w2025|w2026>.csv, results/c2_preds.csv
"""
import hashlib
import json
import os
import sys
import time

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF4는 2023~2026 전체를 쓴다(허용)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402
from sklearn.linear_model import Ridge        # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402

import lib                                    # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
EV = os.path.join(RES, "ev4_events.parquet")
NBOOT, SEED, NRAND = 10000, 20261102, 1000
THR = 0.05
ALPHA = 0.05
NUM = ["c01_time", "c02_speed", "c03_vs_open", "c04_vs_prevclose", "c05_gap", "c06_first5", "c07_vwap", "c08_rsi",
       "c10_wick", "c11_vol", "c12_risevol", "c13_newhighs", "c15_vs_prevhigh", "c16_vs_prehigh", "c17_qqq",
       "c18_nvda", "c19_soxx", "c20_prevret", "c21_prevrange", "c22_preveff", "c23_ret5", "c24_pre", "c26_ma20",
       "c27_e90range"]
MODELS = ["ridge", "hgb"]
WINDOWS = {"w2023": "2023-01", "w2025": "2025-01", "w2026": "2026-01"}
# 예측 연도별 첫 예측 달(창 시작이 그해면 3월부터 — BRIEF4 (iii) '3월부터 예측'을 같은 꼴의 2025 w2025에도 적용)
TEST_SETS = {
    "2026": {"w2023": ("2026-01", "2026-10"), "w2025": ("2026-01", "2026-10"), "w2026": ("2026-03", "2026-10")},
    "2025": {"w2023": ("2025-01", "2025-12"), "w2025": ("2025-03", "2025-12")},
    "2024": {"w2023": ("2024-01", "2024-12")},
}
HALVES_2026 = [("2026-01", "2026-06"), ("2026-07", "2026-10")]


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def build_features(E):
    X = pd.DataFrame(index=E.index)
    groups = {}
    for c in NUM:
        X[c] = E[c].astype(float)
        groups[c] = [c]
    X["c09_rsidiv"] = E["c09_rsidiv"].astype(float)
    groups["c09_rsidiv"] = ["c09_rsidiv"]
    X["c14_topclose"] = E["c14_topclose"].astype(float)
    groups["c14_topclose"] = ["c14_topclose"]
    wd = E["c25_weekday"]
    cols = []
    for k, nm in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        X[f"c25_wd_{nm}"] = np.where(wd.isna(), np.nan, (wd == k).astype(float))
        cols.append(f"c25_wd_{nm}")
    groups["c25_weekday"] = cols
    X["c28_seq_ge2"] = np.where(E["c28_seq"].isna(), np.nan, (E["c28_seq"] >= 2).astype(float))
    groups["c28_seq"] = ["c28_seq_ge2"]
    X["side_mirror"] = (E["side"].astype(str) == "mirror").astype(float)
    groups["side"] = ["side_mirror"]
    X["x_pct"] = E["x"].astype(float) * 100
    groups["x"] = ["x_pct"]
    order = ([f"c{n:02d}" for n in range(1, 29)])
    gkeys = sorted([g for g in groups if g.startswith("c")], key=lambda g: int(g[1:3])) + ["side", "x"]
    assert len(gkeys) == 30 and [g[:3] for g in gkeys[:28]] == order
    groups = {g: groups[g] for g in gkeys}
    return X, groups


class RidgeModel:
    def __init__(self, nan_cols):
        self.nan_cols = nan_cols

    def _prep(self, X):
        Z = X.copy()
        for c in self.nan_cols:
            Z[c + "_na"] = X[c].isna().astype(float)
        Z[list(X.columns)] = X.fillna(self.med)
        return Z.values

    def fit(self, X, y):
        med = X.median()
        self.med = med.fillna(0.0)
        Z = self._prep(X)
        self.sc = StandardScaler().fit(Z)
        self.m = Ridge(alpha=10.0).fit(self.sc.transform(Z), y)
        self.names = list(X.columns) + [c + "_na" for c in self.nan_cols]
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(self._prep(X)))


def make_model(name, nan_cols):
    if name == "ridge":
        return RidgeModel(nan_cols)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=0)


def month_range(a, b):
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def take(sub, key):
    """sub: 그날 고른 사건(DataFrame). key: 정렬 열 목록. 반환: 실제 산 사건 색인 목록."""
    out, last = [], -1
    for ix, r in sub.sort_values(key[0], ascending=key[1], kind="mergesort").iterrows():
        if r["dec_i"] > last:
            out.append(ix)
            last = r["exit_i"]
    return out


def day_returns(E, taken_idx, days, col="r_w90"):
    t = E.loc[taken_idx]
    g = t.groupby("d")[col].apply(lambda s: float(np.prod(1 + s.values / 100) - 1))
    n = t.groupby("d").size()
    daily = pd.DataFrame({"d": days})
    daily["day"] = daily["d"].map(lambda k: DAYS[k])
    daily["ret"] = daily["d"].map(g).fillna(0.0)
    daily["n"] = daily["d"].map(n).fillna(0).astype(int)
    return daily


def stats_block(E, taken, days, months_halves=None):
    daily = day_returns(E, taken, days)
    s = lib.summarize(daily, nboot=NBOOT, seed=SEED)
    r = daily["ret"].values * 100
    daily["month"] = [str(pd.Timestamp(x).to_period("M")) for x in daily["day"]]
    bym = daily.groupby("month")["ret"].agg(["sum", "mean", "size"])
    s["by_month_mean_pct"] = {m: float(v * 100) for m, v in bym["mean"].items()}
    s["by_month_trades"] = {m: int(v) for m, v in daily.groupby("month")["n"].sum().items()}
    best = bym["sum"].idxmax()
    keep = daily["month"] != best
    s["best_month"] = best
    s["ex_best_month_mean_pct"] = float(r[keep.values].mean())
    if months_halves:
        hv = []
        for a, b in months_halves:
            m = (daily["month"] >= a) & (daily["month"] <= b)
            hv.append(float(r[m.values].mean()) if m.any() else None)
        s["halves_mean_pct"] = hv
    s["cost2x_mean_pct"] = float(day_returns(E, taken, days, col="r_w90_c2")["ret"].mean() * 100)
    t = E.loc[taken]
    s["per_trade_mean_pct"] = float(t["r_w90"].mean()) if len(t) else None
    s["hit_rate"] = float(t["hit"].mean()) if len(t) else None
    s["miss_mean_pct"] = float(t.loc[t["hit"] == 0, "r_w90"].mean()) if (t["hit"] == 0).any() else None
    s["trades_by_side"] = {k: int(v) for k, v in t["side"].astype(str).value_counts().items()}
    s["trades_by_x"] = {str(k): int(v) for k, v in t["x"].value_counts().sort_index().items()}
    worst = np.sort(r)[:3]
    s["worst3_days_pct"] = [float(v) for v in worst]
    return s, daily


def random_control(E, sel_mask, days, rng):
    """날마다 모형이 고른 수만큼 그날 사건에서 무작위로 골라 같은 겹침 규칙. 반환: 1,000회 하루 평균(%)."""
    by_day = {}
    for d, g in E.groupby("d"):
        k = int(sel_mask.loc[g.index].sum())
        if k == 0:
            continue
        by_day[d] = (g["dec_i"].values, g["exit_i"].values, g["x"].values, g["r_w90"].values, k)
    nd = len(days)
    out = np.empty(NRAND)
    for rep in range(NRAND):
        tot = 0.0
        for d, (dec, ex, x, r, k) in by_day.items():
            ch = rng.choice(len(dec), k, replace=False)
            o = ch[np.lexsort((x[ch], dec[ch]))]
            last, prod = -1, 1.0
            for j in o:
                if dec[j] > last:
                    prod *= 1 + r[j] / 100
                    last = ex[j]
            tot += prod - 1
        out[rep] = tot / nd * 100
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, j in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[j]))
        adj[j] = run
    return adj


def main():
    global DAYS
    t0 = time.time()
    E = pd.read_parquet(EV).reset_index(drop=True)
    D = lib.Data("alp1m_long")
    DAYS = list(D.days)
    del D
    assert all(pd.Timestamp(DAYS[k]) == pd.Timestamp(v) for k, v in zip(E["d"], E["day"])), "사건 d ↔ Data 날 불일치"
    month_of_day = {k: str(pd.Timestamp(v).to_period("M")) for k, v in enumerate(DAYS)}
    X, groups = build_features(E)
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    y = E["r_w90"].values
    mon = E["month"].astype(str).values

    preds = {}            # (model, window) -> Series(NaN 기본), 연도별 시험 달만 채움
    train_n = {}
    perm = {}             # (window) -> {group: [sum ΔMSE × n, n]}
    ridge_coef = {}
    for model in MODELS:
        for w, wstart in WINDOWS.items():
            p = pd.Series(np.nan, index=E.index)
            tn = pd.Series(np.nan, index=E.index)
            for yr, spec in TEST_SETS.items():
                if w not in spec:
                    continue
                for m in month_range(*spec[w]):
                    tr = (mon >= wstart) & (mon < m)
                    te = mon == m
                    if not te.any() or tr.sum() == 0:
                        continue
                    mdl = make_model(model, nan_cols).fit(X[tr], y[tr])
                    p[te] = mdl.predict(X[te])
                    tn[te] = int(tr.sum())
                    if yr == "2026" and model == "hgb":
                        acc = perm.setdefault(w, {g: [0.0, 0] for g in groups})
                        Xt, yt = X[te].copy(), y[te]
                        base = float(np.mean((mdl.predict(Xt) - yt) ** 2))
                        prng = np.random.default_rng(0)
                        for g, cols in groups.items():
                            inc = []
                            for _ in range(10):
                                Xp = Xt.copy()
                                sh = prng.permutation(len(Xt))
                                Xp[cols] = Xt[cols].values[sh]
                                inc.append(float(np.mean((mdl.predict(Xp) - yt) ** 2)) - base)
                            acc[g][0] += float(np.mean(inc)) * int(te.sum())
                            acc[g][1] += int(te.sum())
                    if yr == "2026" and model == "ridge" and m == "2026-10":
                        ridge_coef[w] = sorted(zip(mdl.names, map(float, mdl.m.coef_)), key=lambda z: -abs(z[1]))
            preds[(model, w)] = p
            train_n[(model, w)] = tn

    # 시뮬레이션
    results = {"2026": {}, "2025": {}, "2024": {}}
    trades_rows = {}
    for yr, spec in TEST_SETS.items():
        for model in MODELS:
            for w in WINDOWS:
                if w not in spec:
                    continue
                a, b = spec[w]
                months = month_range(a, b)
                days = [k for k in range(len(DAYS)) if month_of_day[k] in months]
                inm = np.isin(mon, months)
                Ei = E[inm].copy()
                Ei["pred"] = preds[(model, w)][inm]
                assert Ei["pred"].notna().all(), (yr, model, w)
                sel = Ei["pred"] > THR
                taken = []
                for d, g in Ei[sel].groupby("d"):
                    taken += take(g, (["dec_i", "pred", "x"], [True, False, True]))
                halves = None
                if yr == "2026":
                    halves = [(max(a, h0), h1) for h0, h1 in HALVES_2026]
                s, daily = stats_block(Ei, taken, days, halves)
                allt = []
                for d, g in Ei.groupby("d"):
                    allt += take(g, (["dec_i", "x"], [True, True]))
                s_all, _ = stats_block(Ei, allt, days, halves)
                rnd = random_control(Ei, sel, days, np.random.default_rng(SEED))
                mean = s["mean_pct"]
                ic = float(Ei["pred"].corr(Ei["r_w90"], method="spearman"))
                res = {
                    "model": model, "window": w, "train_start": WINDOWS[w], "test_months": [a, b],
                    "days": len(days), "events_in_test": int(len(Ei)), "events_selected": int(sel.sum()),
                    "selected_event_mean_pct": float(Ei.loc[sel, "r_w90"].mean()) if sel.any() else None,
                    "unselected_event_mean_pct": float(Ei.loc[~sel, "r_w90"].mean()) if (~sel).any() else None,
                    "spearman_pred_vs_r": ic,
                    "model_stats": s,
                    "all_events": {k: s_all[k] for k in ["mean_pct", "ci95", "p_one_sided", "trades", "active_days",
                                                         "per_trade_mean_pct", "hit_rate", "halves_mean_pct",
                                                         "cost2x_mean_pct", "ex_best_month_mean_pct"] if k in s_all},
                    "random": {"mean_of_means": float(rnd.mean()), "q05": float(np.percentile(rnd, 5)),
                               "q50": float(np.percentile(rnd, 50)), "q95": float(np.percentile(rnd, 95)),
                               "model_percentile": float((rnd < mean).mean()),
                               "p_ge": float((1 + (rnd >= mean).sum()) / (1 + NRAND))},
                }
                if yr == "2026":
                    # 공통 비교용: 2026-03~10만
                    if w != "w2026":
                        months_c = month_range("2026-03", "2026-10")
                        days_c = [k for k in days if month_of_day[k] in months_c]
                        tc = [i for i in taken if month_of_day[E.loc[i, "d"]] in months_c]
                        sc, _ = stats_block(Ei, tc, days_c, [("2026-03", "2026-06"), ("2026-07", "2026-10")])
                        res["mar_oct_view"] = {k: sc[k] for k in ["mean_pct", "ci95", "p_one_sided", "trades",
                                                                  "halves_mean_pct", "cost2x_mean_pct"]}
                results[yr][f"{model}_{w}"] = res
                t = Ei.loc[taken, ["day", "month", "side", "rise", "buy", "x", "dec_close", "entry", "exit", "pred",
                                   "r_w90", "r_w90_c2", "why", "hit", "mae", "mfe"]].copy()
                t.insert(0, "set", yr)
                t["train_n"] = train_n[(model, w)][t.index].astype(int).values
                trades_rows.setdefault((model, w), []).append(t)
                print(f"{yr} {model}_{w}: mean {mean:+.3f} ci {s['ci95'][0]:+.3f}~{s['ci95'][1]:+.3f} "
                      f"p {s['p_one_sided']:.4f} n {s['trades']} rand q95 {res['random']['q95']:+.3f} "
                      f"all {s_all['mean_pct']:+.3f} ic {ic:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    # Holm (2026, m = 6) 과 판정
    keys = [f"{m}_{w}" for m in MODELS for w in WINDOWS]
    ps = [results["2026"][k]["model_stats"]["p_one_sided"] for k in keys]
    adj = holm(ps)
    verdict = {}
    for k, pa in zip(keys, adj):
        r = results["2026"][k]
        s = r["model_stats"]
        c_lo = s["ci95"][0] > 0
        c_rand = s["mean_pct"] > r["random"]["q95"]
        c_holm = pa <= ALPHA
        c_h = all(h is not None and h > 0 for h in s["halves_mean_pct"])
        c_best = s["ex_best_month_mean_pct"] > 0
        c_cost = s["cost2x_mean_pct"] > 0
        r["holm_p"] = float(pa)
        r["checks"] = {"ci_low_gt0": bool(c_lo), "gt_random_q95": bool(c_rand), "holm_le_0.05": bool(c_holm),
                       "halves_gt0": bool(c_h), "ex_best_month_gt0": bool(c_best), "cost2x_gt0": bool(c_cost)}
        core = c_lo and c_rand and c_holm
        verdict[k] = "통과" if core and c_h and c_best and c_cost else ("C2 기준만 통과" if core else "불통과")
        r["verdict"] = verdict[k]

    imp = {}
    for w, acc in perm.items():
        imp[w] = sorted([(g, v[0] / v[1]) for g, v in acc.items()], key=lambda z: -z[1])
    out = {
        "brief": "BRIEF4.md 'C2 모형 전방 예측'",
        "data": "results/ev4_events.parquet (5,194 사건) + Data('alp1m_long').days",
        "code_sha256_16": sha16(os.path.abspath(__file__)),
        "lib_sha256_16": sha16(os.path.join(HERE, "lib.py")),
        "events_sha256_16": sha16(EV),
        "units": "% (1.0 = 1%)",
        "settings": {"models": MODELS, "windows": WINDOWS, "threshold_pred_pct": THR, "test_sets": TEST_SETS,
                     "nboot": NBOOT, "seed": SEED, "nrand": NRAND, "features": list(X.columns),
                     "ridge_nan_indicator_cols": nan_cols, "n_feature_groups": len(groups)},
        "results": results,
        "holm_2026": dict(zip(keys, map(float, adj))),
        "verdict_2026": verdict,
        "hgb_permutation_importance_2026_oos_dmse": {w: [[g, float(v)] for g, v in lst] for w, lst in imp.items()},
        "ridge_coef_2026_10_std": {w: [[n, c] for n, c in lst[:12]] for w, lst in ridge_coef.items()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(RES, "c2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    for (model, w), lst in trades_rows.items():
        pd.concat(lst).to_csv(os.path.join(RES, f"c2_trades_{model}_{w}.csv"), index=False, encoding="utf-8-sig")
    pr = E[["day", "side", "x", "dec_close", "r_w90"]].copy()
    for (model, w), p in preds.items():
        pr[f"pred_{model}_{w}"] = p.values
    pr.to_csv(os.path.join(RES, "c2_preds.csv"), index=False, encoding="utf-8-sig")
    print("verdict", verdict)
    print("holm", out["holm_2026"])
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
