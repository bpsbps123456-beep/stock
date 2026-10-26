"""반박 검증(재계산 렌즈): b4_flow 칸 'N20|gvip_rel20_p≥0.5382(상위3분위)'를 갈래 코드 없이 다시 만든다.

입력
- 하네스 사건표 results/events_N20.parquet (사건·short_ret_h10만 사용)
- 원자료 ../leverage_flood/results/us_daily_panel.csv 의 GVIP_SPY (교차 확인: ../hf_options/cache/hf_proxies_by_symbol.parquet
  GVIP 종가 ÷ cache/daily.parquet SPY 종가)
- 일봉 cache/daily.parquet (공매도 손익 표본 재계산 확인)

정의(BRIEF + 후보 정의 그대로)
- x(D) = GVIP/SPY 비율(휴장 빈 줄 뺀 뒤)의 20거래일 변화율, p(D) = x의 250일 백분위(D 포함 창).
- 이탈일 t의 p(t)를 쓴다(종가 D 16시에 알 수 있음 → 진입 t+1 시가).
- 문턱 = 앞 구간(2026-01-02~05-29) N=20 사건(종목+ETF, short_ret_h10 있는 것) p 값의 2/3 분위. 켜짐 = p ≥ 문턱.
- 차이 = 켜짐 평균 − 꺼짐 평균(short_ret_h10, 비용 포함). 날짜 묶음 t = 이탈일로 묶은 CR1(직접 구현).
- 순열 p: 날짜 고정 특징이므로 뒤 구간 이탈 날짜끼리 표지를 통째 섞는다(200번, 한쪽).
백분위 정의가 갈래마다 다를 수 있어 여러 변형(창 포함/제외, ≤ / <, 최소 기간)과 다른 원천으로도 다시 잰다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT.parent
FRONT = ("2026-01-02", "2026-05-29")
BACK = ("2026-06-01", "2026-09-29")
AUX = ("2025-01-01", "2025-12-31")
COST = 0.001
SEED = 20261014


# ───────── 특징 ─────────
def ratio_panel():
    d = pd.read_csv(ANA / "leverage_flood/results/us_daily_panel.csv", parse_dates=["date"]).set_index("date")
    return d["GVIP_SPY"].dropna()


def ratio_raw():
    g = pd.read_parquet(ANA / "hf_options/cache/hf_proxies_by_symbol.parquet")
    g = g[g["symbol"] == "GVIP"].copy()
    g["known"] = pd.to_datetime(g["known"])
    gv = g.set_index("known")["close"]
    D = pd.read_parquet(ROOT / "cache/daily.parquet")
    spy = D[D["sym"] == "SPY"].set_index("date")["Close"]
    r = (gv / spy).dropna()
    return r


def pct_rank(x, win=250, mode="incl_le", minp=None):
    minp = win if minp is None else minp
    if mode == "incl_le":        # 창(현재 포함) 안 현재 이하 비율
        f = lambda a: np.mean(a <= a[-1])
    elif mode == "incl_lt":      # 현재 미만 비율
        f = lambda a: np.mean(a < a[-1])
    elif mode == "prior_le":     # 직전 win−1 값 중 현재 이하 비율
        f = lambda a: np.mean(a[:-1] <= a[-1])
    elif mode == "avg":          # pandas rank(pct) 평균 순위
        f = lambda a: (np.sum(a < a[-1]) + 0.5 * (np.sum(a == a[-1]) + 1)) / len(a)
    else:
        raise ValueError(mode)
    return x.rolling(win, min_periods=minp).apply(f, raw=True)


def feature(ratio, mode="incl_le", minp=None):
    x = ratio.pct_change(20, fill_method=None)
    return pct_rank(x, 250, mode, minp)


# ───────── 통계 ─────────
def cr1_diff(on, y, dates):
    """켜짐 평균 − 꺼짐 평균과 날짜 묶음(CR1) t. 회귀 y = a + b·on, 이탈일 군집."""
    on = np.asarray(on, float)
    y = np.asarray(y, float)
    X = np.c_[np.ones_like(on), on]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    codes = pd.factorize(pd.Series(dates))[0]
    G = codes.max() + 1
    S = np.zeros((G, 2))
    np.add.at(S, codes, X * u[:, None])
    meat = S.T @ S
    n, k = len(y), 2
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return float(b[1]), float(b[1] / np.sqrt(V[1, 1])), int(G)


def perm_p_dates(on, y, dates, k=200, seed=SEED):
    """날짜 고정 표지를 이탈 날짜끼리 통째 섞은 한쪽 p."""
    rng = np.random.default_rng(seed)
    dts = pd.Series(dates)
    ud = np.array(sorted(dts.unique()))
    lab = pd.Series(on, index=dts.values).groupby(level=0).first().reindex(ud).to_numpy()
    idx = np.searchsorted(ud, dts.values)
    obs = y[on == 1].mean() - y[on == 0].mean()
    cnt = fin = 0
    for _ in range(k):
        pl = lab[rng.permutation(len(ud))][idx]
        if pl.sum() == 0 or (pl == 0).sum() == 0:
            continue
        d = y[pl == 1].mean() - y[pl == 0].mean()
        fin += 1
        cnt += d >= obs - 1e-15
    return (1 + cnt) / (1 + fin)


def shift_p_dates(on, y, dates):
    """보조: 이탈 날짜를 시간순으로 놓고 표지를 1..D−1칸 원형 이동(국면 지속성 보존)한 한쪽 p."""
    dts = pd.Series(dates)
    ud = np.array(sorted(dts.unique()))
    lab = pd.Series(on, index=dts.values).groupby(level=0).first().reindex(ud).to_numpy()
    idx = np.searchsorted(ud, dts.values)
    obs = y[on == 1].mean() - y[on == 0].mean()
    cnt = fin = 0
    for s in range(1, len(ud)):
        pl = np.roll(lab, s)[idx]
        if pl.sum() == 0 or (pl == 0).sum() == 0:
            continue
        fin += 1
        cnt += (y[pl == 1].mean() - y[pl == 0].mean()) >= obs - 1e-15
    return (1 + cnt) / (1 + fin)


def episode_share(sub):
    on = sub[sub["on"] == 1]
    tot = on["y"].sum()
    if tot <= 0:
        return np.nan, None
    s = on.groupby("episode")["y"].sum()
    s = s[s.index >= 0]
    return float(s.max() / tot), int(s.idxmax())


def evaluate(ev, feat, label):
    e = ev.copy()
    e["p"] = feat.reindex(pd.DatetimeIndex(e["t"])).to_numpy()
    e = e[e["short_ret_h10"].notna()]
    fr = e[(e["t"] >= FRONT[0]) & (e["t"] <= FRONT[1])]
    thr = float(np.quantile(fr["p"].dropna(), 2 / 3))
    e["on"] = np.where(e["p"].isna(), np.nan, (e["p"] >= thr).astype(float))
    e["y"] = e["short_ret_h10"]
    out = {"variant": label, "threshold": thr, "missing_p_front": int(fr["p"].isna().sum())}
    for nm, (a, b) in (("f", FRONT), ("b", BACK), ("a", AUX)):
        s = e[(e["t"] >= a) & (e["t"] <= b) & e["on"].notna()]
        on = s["on"].to_numpy()
        y = s["y"].to_numpy()
        diff, t, G = cr1_diff(on, y, s["t"].to_numpy())
        out[f"{nm}_n_on"] = int(on.sum())
        out[f"{nm}_n_off"] = int((on == 0).sum())
        out[f"{nm}_dates_on"] = int(s.loc[s["on"] == 1, "t"].nunique())
        out[f"{nm}_dates_all"] = G
        out[f"{nm}_mean_on"] = float(y[on == 1].mean() * 100)
        out[f"{nm}_mean_off"] = float(y[on == 0].mean() * 100)
        out[f"{nm}_diff"] = diff * 100
        out[f"{nm}_t"] = t
        out[f"{nm}_n_missing"] = int(((e["t"] >= a) & (e["t"] <= b) & e["on"].isna()).sum())
        if nm == "b":
            out["b_perm_p"] = perm_p_dates(on, y, s["t"].to_numpy())
            out["b_shift_p"] = shift_p_dates(on, y, s["t"].to_numpy())
            wk = s["t"].dt.to_period("W").astype(str).to_numpy()
            out["b_t_week"] = cr1_diff(on, y, wk)[1]
            out["b_n_weeks"] = int(len(np.unique(wk)))
            sh, ep = episode_share(s)
            out["b_episode_top_share"], out["b_episode_top"] = sh, ep
            # 켜진 사건 평균 공매도 손익 자체의 날짜 묶음 t
            so = s[s["on"] == 1]
            m = so["y"].mean()
            dm = so.groupby("t")["y"].apply(lambda v: (v - m).sum()).to_numpy() / len(so)
            Gn = len(dm)
            out["b_on_mean_t"] = float(m / np.sqrt((dm ** 2).sum() * Gn / (Gn - 1)))
            # 날짜 가중(날짜마다 한 표) 켜짐·꺼짐 평균
            dmean = s.groupby(["t", "on"])["y"].mean().reset_index()
            out["b_datew_diff"] = float((dmean[dmean.on == 1].y.mean() - dmean[dmean.on == 0].y.mean()) * 100)
            # 켜짐 날짜 목록(날짜·사건 수·평균)
            out["b_on_dates"] = {str(k.date()): [int(len(v)), round(float(v.mean() * 100), 2)]
                                 for k, v in so.groupby("t")["y"]}
            # 종목/ETF 나눠 보기
            for kd in ("stock", "etf"):
                sk = s[s["kind"] == kd]
                if (sk["on"] == 1).sum() and (sk["on"] == 0).sum():
                    out[f"b_{kd}_diff"] = float((sk[sk.on == 1].y.mean() - sk[sk.on == 0].y.mean()) * 100)
                    out[f"b_{kd}_n_on"] = int((sk["on"] == 1).sum())
    return out, e


# ───────── 공매도 손익 원자료 재계산(사건표 값 확인) ─────────
def recheck_short(ev):
    D = pd.read_parquet(ROOT / "cache/daily.parquet")
    q = D[D["sym"] == "QQQ"]["date"].sort_values().unique()
    cal = pd.DatetimeIndex(q)
    piv = {f: D.pivot(index="date", columns="sym", values=f).reindex(cal) for f in ("Open", "Low", "Close")}
    diffs = []
    for r in ev.itertuples():
        o = piv["Open"][r.sym].to_numpy()
        lo = piv["Low"][r.sym].to_numpy()
        c = piv["Close"][r.sym].to_numpy()
        i = cal.get_loc(r.t)
        Lv = np.nanmin(lo[i - 20:i]) if not np.isnan(lo[i - 20:i]).any() else np.nan
        e = o[i + 1]
        res = np.nan
        for k in range(i + 1, i + 10):
            if c[k] > Lv:
                res = (e - o[k + 1]) / e - COST
                break
        else:
            res = (e - c[i + 10]) / e - COST
        diffs.append(abs(res - r.short_ret_h10))
    return float(np.nanmax(diffs)), int(np.sum(np.array(diffs) > 1e-9)), len(diffs)


def main():
    ev = pd.read_parquet(ROOT / "results/events_N20.parquet")
    ev["t"] = pd.to_datetime(ev["t"])
    rp = ratio_panel()
    rr = ratio_raw()
    variants = [
        ("panel|incl_le|minp250", feature(rp, "incl_le")),
        ("panel|incl_lt|minp250", feature(rp, "incl_lt")),
        ("panel|prior_le|minp250", feature(rp, "prior_le")),
        ("panel|avg|minp250", feature(rp, "avg")),
        ("panel|incl_le|minp150", feature(rp, "incl_le", 150)),
        ("raw_hf+daily|incl_le|minp250", feature(rr, "incl_le")),
        ("raw_hf+daily|prior_le|minp250", feature(rr, "prior_le")),
    ]
    res = []
    main_e = None
    for lab, f in variants:
        o, e = evaluate(ev, f, lab)
        res.append(o)
        if main_e is None:
            main_e = e
    # 사건표 공매도 손익 원자료 재계산(뒤 구간 N=20 전체)
    b = main_e[(main_e["t"] >= BACK[0]) & (main_e["t"] <= BACK[1])]
    mx, nbad, nchk = recheck_short(b)
    out = {"variants": res, "short_recheck_back": {"max_abs_diff": mx, "n_mismatch": nbad, "n_checked": nchk}}
    (ROOT / "results/verify_b4_flow_recompute.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    cols = ["variant", "threshold", "f_n_on", "f_diff", "f_t", "b_n_on", "b_dates_on", "b_mean_on", "b_mean_off",
            "b_diff", "b_t", "b_perm_p", "b_shift_p", "b_t_week", "b_n_weeks", "b_on_mean_t", "b_datew_diff", "b_episode_top_share", "a_diff", "a_n_on",
            "b_n_missing", "f_n_missing", "a_n_missing"]
    print(pd.DataFrame(res)[cols].to_string())
    print(json.dumps({k: res[0].get(k) for k in ("b_on_dates", "b_stock_diff", "b_stock_n_on", "b_etf_diff",
                                                  "b_etf_n_on", "b_episode_top", "f_dates_on", "b_dates_all")},
                     ensure_ascii=False, default=str))
    print("short recheck", out["short_recheck_back"])


if __name__ == "__main__":
    main()
