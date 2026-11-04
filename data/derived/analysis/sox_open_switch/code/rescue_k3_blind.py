"""K3(+ K2 보조) 눈가림 재구현 B — BRIEF7.

원래 연구 코드(lib.py·stage5.py·redfirst.py 등)와 rescue_k3.py를 import하지 않는다.
쓰는 것: BRIEF5/6/7 정의, cache/alp1m_2016/SOXS.parquet, cost_table_2016.json, yfinance SOXX 일봉.

K3: SOXX 추세 5단계(trend_5zone.pine) D-1 = 극상방(+2)인 날 D, SOXS 09:31 봉 종가 x (1 + open5 반 호가) 매수
    -> 09:59 봉 종가 x (1 - early 반 호가) 매도(std, 10:00 마감 1분 전 봉).
K2 보조: 빨강 첫날 R1(D-1 in {-1,-2}, D-2 not in {-1,-2}), SOXS 09:31 -> 10:59 봉 종가 x (1 - late 반 호가)(w90).
빈 1분봉은 직전 종가로 채운다(asof). 체결·청산 봉이 채운 봉인 매매를 세고, 뺀 감도도 낸다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "rescue_k3_blind.json"
SEED = 20261104
B = 10_000
START, END = pd.Timestamp("2016-01-04"), pd.Timestamp("2022-12-30")


# ---------------- 추세 5단계 (trend_5zone.pine zoneCalc) ----------------
def pine_rma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.rma: 첫 값은 앞 n개 단순평균(na가 끼면 na), 그 뒤 alpha = 1/n 재귀."""
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i in range(len(x)):
        if np.isnan(prev):
            if i >= n - 1:
                w = x[i - n + 1:i + 1]
                if not np.isnan(w).any():
                    prev = w.mean()
                    out[i] = prev
        else:
            prev = (x[i] - prev) / n + prev if not np.isnan(x[i]) else prev
            out[i] = prev
    return out


def stage_series(close: pd.Series, N=20, th=0.5, rsi_len=14, hi=70.0, lo=30.0) -> pd.Series:
    c = close.to_numpy(float)
    lc = np.log(c)
    lr = np.r_[np.nan, np.diff(lc)]
    sd = pd.Series(lr).rolling(N).std(ddof=1).to_numpy()          # ta.stdev(lr, N, biased=false)
    k = np.sqrt((N - 1.0) * (2.0 * N - 1.0) / (6.0 * N))
    sma = pd.Series(lc).rolling(N).mean().to_numpy()
    z = (lc - sma) / (sd * k)
    ch = np.r_[np.nan, np.diff(c)]
    up = np.where(np.isnan(ch), np.nan, np.maximum(ch, 0.0))
    dn = np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0.0))
    ru, rd = pine_rma(up, rsi_len), pine_rma(dn, rsi_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(rd == 0, 100.0, np.where(ru == 0, 0.0, 100.0 - 100.0 / (1.0 + ru / rd)))
    rsi = np.where(np.isnan(ru) | np.isnan(rd), np.nan, rsi)
    zn = np.full(len(c), np.nan)
    for i in range(len(c)):
        if np.isnan(z[i]) or np.isnan(rsi[i]) or i < 129:
            continue
        r = rsi[i]
        zn[i] = 2 if r >= hi else -2 if r <= lo else 1 if z[i] >= th else -1 if z[i] <= -th else 0
    return pd.DataFrame({"close": c, "rsi": rsi, "z": z, "stage": zn}, index=close.index)


def load_soxx_daily() -> pd.Series:
    df = yf.download("SOXX", start="2001-01-01", end="2023-01-11", auto_adjust=False,
                     progress=False, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


# ---------------- 1분봉 가격 ----------------
def minute_prices(days: list, soxs: pd.DataFrame, hhmm: list) -> pd.DataFrame:
    """날 x 시각마다 (그 분 시작 봉 종가, 채움 여부, 채운 출처). 빈 분은 직전 봉 종가(asof)."""
    idx = soxs.index
    close = soxs["Close"].to_numpy()
    rows = []
    for d in days:
        rec = {"date": d}
        for h in hhmm:
            t = pd.Timestamp(f"{d.date()} {h}", tz="America/New_York")
            pos = idx.searchsorted(t, side="right") - 1
            px = close[pos]
            filled = idx[pos] != t
            rec[f"px_{h}"] = px
            rec[f"fill_{h}"] = bool(filled)
            if filled:
                src = idx[pos]
                rec[f"src_{h}"] = ("prevday" if src.date() != d.date() else
                                   "premarket" if src.strftime("%H:%M") < "09:30" else "regular")
            else:
                rec[f"src_{h}"] = ""
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


def trade_returns(px: pd.DataFrame, cost: dict, exit_h: str, exit_zone: str, mult=1.0) -> pd.Series:
    entry = px["px_09:31"] * (1 + mult * cost["open5"])   # 09:31 봉 시작 -> open5
    exitp = px[f"px_{exit_h}"] * (1 - mult * cost[exit_zone])
    return (exitp / entry - 1.0) * 100.0


# ---------------- 대조 ----------------
def random_day_p(r_all: pd.Series, ev_mask: pd.Series, B: int, seed: int) -> dict:
    """같은 해의 아무 거래일(사건일 포함)에서 그 해 사건 수만큼 비복원 추출, 풀 평균 >= 관측 비율."""
    rng = np.random.default_rng(seed)
    ev_mask = ev_mask.reindex(r_all.index, fill_value=False).astype(bool)
    obs = r_all[ev_mask].mean()
    years = r_all.index.year
    tot = np.zeros(B)
    ntot = 0
    for y in sorted(set(years)):
        ry = r_all[years == y].to_numpy()
        n = int(ev_mask[years == y].sum())
        if n == 0:
            continue
        keys = rng.random((B, len(ry)))
        pick = np.argpartition(keys, n - 1, axis=1)[:, :n] if n < len(ry) else np.tile(np.arange(len(ry)), (B, 1))
        tot += ry[pick].sum(axis=1)
        ntot += n
    means = tot / ntot
    return {"p_one_sided": float((means >= obs - 1e-12).mean()), "null_mean": float(means.mean()),
            "null_p95": float(np.quantile(means, 0.95)), "B": B, "seed": seed}


def summarize(r: pd.Series, r2: pd.Series, r_all: pd.Series, mask: pd.Series, seed: int) -> dict:
    x = r[mask]
    if len(x) == 0:
        return {"n": 0}
    xs = x.sort_values(ascending=False)
    by_year = {}
    for y, g in x.groupby(x.index.year):
        by_year[str(y)] = {"n": int(len(g)), "mean": round(float(g.mean()), 4)}
    ypos = sum(1 for v in by_year.values() if v["mean"] > 0)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    out = {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "hit": float((x > 0).mean()),
        "worst": float(x.min()),
        "best": float(x.max()),
        "t": float(x.mean() / se) if se and not np.isnan(se) else None,
        "drop_top2": float(xs.iloc[2:].mean()) if len(xs) > 2 else None,
        "cost2x": float(r2[mask].mean()),
        "years_positive": f"{ypos}/{len(by_year)}",
        "by_year": by_year,
    }
    out.update(random_day_p(r_all, mask, B, seed))
    return out


def main():
    cost = json.load(open(BASE / "cost_table_2016.json", encoding="utf-8"))["SOXS"]

    # 일봉 단계
    soxx = load_soxx_daily()
    st = stage_series(soxx)
    st["prev"] = st["stage"].shift(1)    # D 장초에 쓰는 값 = D-1 종가 단계
    st["prev2"] = st["stage"].shift(2)

    # 1분봉 (한 번만 읽음)
    soxs = pd.read_parquet(BASE / "cache" / "alp1m_2016" / "SOXS.parquet", columns=["Close"]).sort_index()
    et = soxs.index
    hm = et.strftime("%H:%M")
    reg = (hm >= "09:30") & (hm < "16:00")
    days_1m = pd.DatetimeIndex(sorted(set(et[reg].tz_localize(None).normalize())))
    days_1m = days_1m[(days_1m >= START) & (days_1m <= END)]
    days_yf = st.index[(st.index >= START) & (st.index <= END)]
    only_1m = sorted(set(days_1m) - set(days_yf))
    only_yf = sorted(set(days_yf) - set(days_1m))
    days = pd.DatetimeIndex(sorted(set(days_1m) & set(days_yf)))

    px = minute_prices(list(days), soxs, ["09:31", "09:59", "10:59"])

    r_std = trade_returns(px, cost, "09:59", "early")
    r_std2 = trade_returns(px, cost, "09:59", "early", mult=2.0)
    r_w90 = trade_returns(px, cost, "10:59", "late")
    r_w902 = trade_returns(px, cost, "10:59", "late", mult=2.0)

    prev = st["prev"].reindex(days)
    prev2 = st["prev2"].reindex(days)
    red = {-1.0, -2.0}
    k3 = prev == 2
    r1 = prev.isin(red) & ~prev2.isin(red)

    fill_std = px["fill_09:31"] | px["fill_09:59"]
    fill_w90 = px["fill_09:31"] | px["fill_10:59"]

    res = {
        "meta": {
            "impl": "blind B (rescue_k3_blind.py), lib/stage5/redfirst/rescue_k3 미사용",
            "data_1m": "cache/alp1m_2016/SOXS.parquet",
            "costs": "cost_table_2016.json SOXS " + json.dumps(cost),
            "soxx_daily": f"yfinance SOXX auto_adjust=False Close {soxx.index[0].date()}~{soxx.index[-1].date()} ({len(soxx)}봉)",
            "days": int(len(days)), "first": str(days[0].date()), "last": str(days[-1].date()),
            "days_only_in_1m": [str(d.date()) for d in only_1m],
            "days_only_in_yf": [str(d.date()) for d in only_yf],
            "fill_rule": "빈 분은 직전 봉 종가(asof, 같은 날 프리장·정규장 또는 전날)",
            "random_control": "같은 해 거래일(사건일 포함)에서 그 해 사건 수 비복원 추출, 풀 평균 >= 관측 비율",
            "stage_counts_2016_2022": {str(int(k)): int(v) for k, v in
                                       st.loc[(st.index >= START) & (st.index <= END), "stage"].value_counts().sort_index().items()},
            "fills_all_days": {h: int(px[f"fill_{h}"].sum()) for h in ["09:31", "09:59", "10:59"]},
        },
    }

    # K3 주
    k3_all = summarize(r_std, r_std2, r_std, k3, SEED)
    nf = k3 & ~fill_std
    k3_nofill = summarize(r_std, r_std2, r_std[~fill_std], nf, SEED)
    k3_fill_info = {
        "n_filled_trades": int((k3 & fill_std).sum()),
        "entry_filled": int((k3 & px["fill_09:31"]).sum()),
        "exit_filled": int((k3 & px["fill_09:59"]).sum()),
        "filled_by_year": {str(y): int(v) for y, v in (k3 & fill_std)[k3].groupby(days[k3.to_numpy()].year).sum().items()},
        "filled_sources": {h: px.loc[k3 & px[f"fill_{h}"], f"src_{h}"].value_counts().to_dict() for h in ["09:31", "09:59"]},
    }
    # 정규장 직전 SOXS 분 단위 연속 봉 여부와 무관하게, 평균의 차이
    k3_events = pd.DataFrame({
        "date": [str(d.date()) for d in days[k3.to_numpy()]],
        "stage_prev": prev[k3].astype(int).to_numpy(),
        "rsi_prev": st["rsi"].shift(1).reindex(days)[k3].round(2).to_numpy(),
        "px_0931": px.loc[k3, "px_09:31"].to_numpy(),
        "px_0959": px.loc[k3, "px_09:59"].to_numpy(),
        "fill_0931": px.loc[k3, "fill_09:31"].to_numpy(),
        "fill_0959": px.loc[k3, "fill_09:59"].to_numpy(),
        "r_pct": r_std[k3].round(4).to_numpy(),
    })
    res["K3"] = {
        "rule": "SOXX 단계(D-1)=+2, SOXS 09:31 종가+open5 -> 09:59 종가-early (std)",
        "all": k3_all, "excl_filled": k3_nofill, "fill": k3_fill_info,
        "events": k3_events.to_dict(orient="records"),
        "all_days_mean_std": float(r_std.mean()),
    }

    # 판정 (Holm m = 6은 오케스트레이터가 다른 후보 p와 묶는다; 여기서는 원 p와 단독 판정 5항목)
    a = k3_all
    res["K3"]["checks"] = {
        "1_mean_gt0": a["mean"] > 0,
        "2_p_raw": a["p_one_sided"],
        "2_holm_m6_if_smallest_bound": min(1.0, 6 * a["p_one_sided"]),
        "3_drop_top2_gt0": (a["drop_top2"] or -1) > 0,
        "4_cost2x_gt0": a["cost2x"] > 0,
        "5_half_years_positive": int(a["years_positive"].split("/")[0]) * 2 >= int(a["years_positive"].split("/")[1]),
    }

    # K2 보조 (판정 밖): R1, SOXS 09:31 -> 10:59 (w90)
    k2_all = summarize(r_w90, r_w902, r_w90, r1, SEED)
    nf2 = r1 & ~fill_w90
    k2_nofill = summarize(r_w90, r_w902, r_w90[~fill_w90], nf2, SEED)
    res["K2_aux"] = {
        "rule": "R1 빨강 첫날(D-1 in {-1,-2}, D-2 not), SOXS 09:31 종가+open5 -> 10:59 종가-late (w90)",
        "all": k2_all, "excl_filled": k2_nofill,
        "fill": {"n_filled_trades": int((r1 & fill_w90).sum()),
                 "entry_filled": int((r1 & px["fill_09:31"]).sum()),
                 "exit_filled": int((r1 & px["fill_10:59"]).sum())},
        "events": [{"date": str(d.date()), "stage_prev": int(prev[d]), "stage_prev2": int(prev2[d]),
                    "r_pct": round(float(r_w90[d]), 4), "filled": bool(fill_w90[d])} for d in days[r1.to_numpy()]],
        "all_days_mean_w90": float(r_w90.mean()),
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")

    def line(name, s):
        print(f"{name}: n={s['n']} mean={s['mean']:+.4f} med={s['median']:+.4f} hit={s['hit']:.3f} "
              f"drop2={s['drop_top2']:+.4f} cost2x={s['cost2x']:+.4f} p={s['p_one_sided']:.4f} "
              f"yrs+={s['years_positive']} worst={s['worst']:+.3f}")
        print("   by_year", s["by_year"])

    print(res["meta"]["days"], "days", res["meta"]["days_only_in_1m"][:5], res["meta"]["days_only_in_yf"][:5])
    print("fills all days", res["meta"]["fills_all_days"], "stage counts", res["meta"]["stage_counts_2016_2022"])
    line("K3 all", k3_all)
    line("K3 nofill", k3_nofill)
    print("K3 fill", k3_fill_info)
    print("K3 checks", res["K3"]["checks"])
    line("K2aux all", k2_all)
    line("K2aux nofill", k2_nofill)
    print("K2 fill", res["K2_aux"]["fill"])
    print("all-day mean std", r_std.mean(), "w90", r_w90.mean())


if __name__ == "__main__":
    main()
