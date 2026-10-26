"""BRIEF.md 시험을 그대로 돌린다 → results/eval.json, results/eval.md.

읽는 것: results/daily.parquet(build.py), cache/leaps(헤지 예시).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

RES = HERE.parent / "results"
SIGNALS = list(build.DIRECTION) + ["CALLHOT"]
MAIN_TH, SENS_TH = 0.80, (0.70, 0.90)
LOOKBACK_TOP = 15
DD_TOP = -7.0


def periods(df: pd.DataFrame, asset: str) -> dict[str, pd.Index]:
    ok = df.index[df[f"{asset}_mdd"].notna()]
    return {"front": ok[(ok >= "2026-01-02") & (ok <= "2026-05-29")],
            "back": ok[ok >= "2026-06-01"],
            "y2026": ok[ok >= "2026-01-02"],
            "y2025": ok[(ok >= "2025-01-02") & (ok <= "2025-12-31")]}


def diff_stats(sig: pd.Series, y: pd.Series) -> dict:
    m = sig.notna() & y.notna()
    s, v = sig[m].astype(bool), y[m]
    if s.sum() == 0 or (~s).sum() == 0:
        return {"n": int(m.sum()), "n_sig": int(s.sum()), "sig": None, "rest": None, "diff": None}
    return {"n": int(m.sum()), "n_sig": int(s.sum()), "sig": round(v[s].mean(), 2), "rest": round(v[~s].mean(), 2),
            "diff": round(v[s].mean() - v[~s].mean(), 2)}


def shift_p(sig: pd.Series, y: pd.Series, min_shift: int = 20) -> float | None:
    """신호열을 원형 이동해 만든 귀무분포에서 '신호일이 더 나쁨'(차이 ≤ 관측) 비율. 자기상관 보존."""
    m = sig.notna() & y.notna()
    s, v = sig[m].astype(bool).to_numpy(), y[m].to_numpy()
    n = len(s)
    if s.sum() == 0 or s.all() or n < 3 * min_shift:
        return None
    obs = v[s].mean() - v[~s].mean()
    null = []
    for k in range(min_shift, n - min_shift + 1):
        r = np.roll(s, k)
        if r.any() and not r.all():
            null.append(v[r].mean() - v[~r].mean())
    null = np.array(null)
    return round(float(((null <= obs).sum() + 1) / (len(null) + 1)), 3)


def tops(c: pd.Series, dd: float = DD_TOP) -> list[dict]:
    out, peak, pdate, tr, tdate = [], c.iloc[0], c.index[0], c.iloc[0], c.index[0]
    for d, v in c.items():
        if v > peak:
            if (tr / peak - 1) * 100 <= dd:
                out.append({"peak": pdate, "trough": tdate, "dd": round((tr / peak - 1) * 100, 1)})
            peak, pdate, tr, tdate = v, d, v, d
        elif v < tr:
            tr, tdate = v, d
    if (tr / peak - 1) * 100 <= dd:
        out.append({"peak": pdate, "trough": tdate, "dd": round((tr / peak - 1) * 100, 1), "ongoing": True})
    return out


def episodes(sig: pd.Series, gap: int = 5) -> list[str]:
    on = list(sig.index[sig.fillna(False).astype(bool)])
    pos = {d: i for i, d in enumerate(sig.index)}
    starts, last = [], None
    for d in on:
        if last is None or pos[d] - pos[last] > gap:
            starts.append(d)
        last = d
    return starts


def hedge_examples(df: pd.DataFrame) -> dict:
    """QQQ 1년 만기 10% 외가격 풋: 날짜별 비용(현물 대비 %)과, 고점에 샀다면 저점에서의 가치."""
    S = df["QQQ"]
    legs = []
    for f in sorted((build.CACHE / "leaps").glob("QQQ_*.parquet")):
        d = pd.read_parquet(f)
        d = d[(d["kind"] == "P") & (d["volume"] > 0)]
        d["expiry"] = f.stem.split("_")[1]
        legs.append(d)
    p = pd.concat(legs)
    p["T"] = (pd.to_datetime(p["expiry"]) - pd.to_datetime(p["date"])).dt.days / 365
    cost = {}
    for day, g in p[p["date"].isin(S.index)].groupby("date"):
        g = g[(g["T"] > 0.75) & (g["T"] < 1.5) & ((g["strike"] / (0.9 * S[day]) - 1).abs() <= 0.02)]
        if g.empty:  # 그날 체결된 1년 안팎·행사가 0.9±2% 풋이 없으면 건너뛴다
            continue
        row = g.loc[((g["strike"] / (0.9 * S[day]) - 1).abs() + (g["T"] - 1).abs() / 10).idxmin()]
        k = row["strike"]
        cost[day] = {"expiry": row["expiry"], "strike": k, "price": row["close"], "cost_pct": row["close"] / S[day] * 100}
    cost = pd.DataFrame(cost).T
    cost["cost_pct"] = cost["cost_pct"].astype(float)
    out = {"cost_2026_median": round(cost.loc[cost.index >= "2026-01-02", "cost_pct"].median(), 2),
           "cost_2026_p10": round(cost.loc[cost.index >= "2026-01-02", "cost_pct"].quantile(0.1), 2),
           "cost_2026_p90": round(cost.loc[cost.index >= "2026-01-02", "cost_pct"].quantile(0.9), 2), "cases": []}
    for t in tops(S[S.index >= "2025-06-01"]):
        pk, tr = t["peak"], t["trough"]
        if pk not in cost.index:
            continue
        c = cost.loc[pk]
        same = p[(p["expiry"] == c["expiry"]) & (p["strike"] == c["strike"])].sort_values("date")
        q = same[same["date"] <= tr]
        if q.empty:
            continue
        v, last = q.iloc[-1], same[same["date"] <= S.index[-1]].iloc[-1]
        rank = (cost.loc[cost.index >= "2025-07-01", "cost_pct"] <= c["cost_pct"]).mean()
        out["cases"].append({"peak": pk, "trough": tr, "qqq_dd": t["dd"], "contract": f"QQQ {c['expiry']} {c['strike']:.0f}P",
                             "cost_pct": round(c["cost_pct"], 2), "cost_rank_since_2025_07": round(float(rank), 2),
                             "value_at_trough_pct": round(v["close"] / S[pk] * 100, 2), "value_date": v["date"],
                             "gain_pct_of_spot": round((v["close"] - c["price"]) / S[pk] * 100, 2),
                             "held_to": last["date"], "held_gain_pct_of_spot": round((last["close"] - c["price"]) / S[pk] * 100, 2),
                             "qqq_peak_to_last": round((S.iloc[-1] / S[pk] - 1) * 100, 1)})
    cost.to_csv(RES / "hedge_cost_daily.csv")
    return out


def main() -> None:
    df = pd.read_parquet(RES / "daily.parquet")
    res = {"main": {}, "sens": {}, "tops": {}, "false_alarm": {}, "base": {}}
    for asset in ("QQQ", "SOXX"):
        P = periods(df, asset)
        for y in ("mdd", "ret"):
            Y = df[f"{asset}_{y}"]
            res["base"][f"{asset}_{y}"] = {k: round(Y[v].mean(), 2) for k, v in P.items()}
            for s in SIGNALS:
                w = df[f"{s}_w"]
                for th in (MAIN_TH,) + SENS_TH:
                    sig = (w >= th).where(w.notna())
                    row = {k: diff_stats(sig[v], Y[v]) for k, v in P.items()}
                    row["p2026"] = shift_p(sig[P["y2026"]], Y[P["y2026"]])
                    row["p2025"] = shift_p(sig[P["y2025"]], Y[P["y2025"]])
                    (res["main"] if th == MAIN_TH else res["sens"])[f"{asset}|{y}|{s}|{th}"] = row
        c = df[asset].dropna()
        tl = tops(c[c.index >= "2025-01-02"])
        pos = {d: i for i, d in enumerate(df.index)}
        for t in tl:
            i = pos[t["peak"]]
            win = df.index[max(0, i - LOOKBACK_TOP + 1):i + 1]
            t["signals"] = {s: {"any": bool((df.loc[win, f"{s}_w"] >= MAIN_TH).any()) if df.loc[win, f"{s}_w"].notna().any() else None,
                                "at_peak": None if pd.isna(df.at[t["peak"], f"{s}_w"]) else round(float(df.at[t["peak"], f"{s}_w"]), 2)}
                            for s in SIGNALS}
        res["tops"][asset] = tl
    q = df["QQQ_mdd"]
    for s in SIGNALS:
        w = df[f"{s}_w"]
        sig = (w >= MAIN_TH).where(w.notna())
        for name, lo, hi in (("y2026", "2026-01-02", "2026-12-31"), ("y2025", "2025-01-02", "2025-12-31")):
            st = [d for d in episodes(sig[(sig.index >= lo) & (sig.index <= hi)]) if pd.notna(q.get(d))]
            hits = [d for d in st if q[d] <= -5]
            res["false_alarm"][f"{s}|{name}"] = {"episodes": len(st), "followed_by_qqq_dd5": len(hits),
                                                "starts": st}
    for name, lo, hi in (("y2026", "2026-01-02", "2026-12-31"), ("y2025", "2025-01-02", "2025-12-31")):
        v = q[(q.index >= lo) & (q.index <= hi)].dropna()
        res["base"][f"P(QQQ 20d dd<=-5)|{name}"] = round(float((v <= -5).mean()), 3)
    res["hedge"] = hedge_examples(df)
    (RES / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("base", "hedge")}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
