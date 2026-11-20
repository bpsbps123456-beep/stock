"""한국인 매수 비중(SEIBRO) → 다음 20·60거래일 QQQ 대비 수익. 사전 등록 PREREG.md.

가격: ../post_earnings_risers_20261120/cache/w_*.parquet (fetch_wide2.py로 다시 만든다, 2026-11-18까지 사용)
결과: cache/panel.csv(결정일 × 종목, 12MB라 Git 제외), results/summary.json, results/weekly_ic.csv
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
PX = ROOT / "data/derived/analysis/post_earnings_risers_20261120/cache"
SEI = ROOT / "data/derived/analysis/seibro_flows"
LASTDAY = "2026-11-18"
BENCH = {"QQQ", "SPY", "SOXX", "XLY", "XLF", "XLI", "XLK", "XLV", "XLRE", "XLU", "XLE", "XLP", "XLB", "XLC", "^VIX",
         "IGV", "URA", "IWM", "ARKK"}


def seibro_daily():
    m = pd.read_csv(SEI / "derived/isin_master.csv", encoding="utf-8-sig")
    m = m[m.figi_type.isin(["Common Stock", "ADR", "REIT", "NY Reg Shrs"]) & m.figi_ticker.notna()]
    i2t = {i: str(t).replace("/", "-").replace(" ", "-") for i, t in zip(m["isin"], m["figi_ticker"])}
    rec = {}
    for f in glob.glob(str(SEI / "raw/daily/d*/2025*.json")) + glob.glob(str(SEI / "raw/daily/d*/2026*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for r in d["rows"]:
            t = i2t.get(r["isin"])
            if t:
                rec[(d["start"], t)] = (r["buy"], r["net"])
    df = pd.DataFrame([(pd.Timestamp(k[0]), k[1], *v) for k, v in rec.items()], columns=["sdate", "ticker", "buy", "net"])
    buy = df.pivot_table(index="sdate", columns="ticker", values="buy", aggfunc="sum")
    net = df.pivot_table(index="sdate", columns="ticker", values="net", aggfunc="sum")
    return buy, net, set(i2t.values())


def build():
    px = {k: pd.read_parquet(PX / f"w_{k}.parquet") for k in ("open", "close", "adjclose", "volume")}
    idx = px["close"]["QQQ"].dropna().index
    idx = idx[idx <= LASTDAY]
    px = {k: v.reindex(idx) for k, v in px.items()}
    ac, cl, vo = px["adjclose"], px["close"], px["volume"]
    ao = px["open"] * ac / cl
    dv = cl * vo
    lr = np.log(ac / ac.shift(1))
    buy, net, _ = seibro_daily()
    sd = buy.index
    cols = [c for c in ac.columns if c not in BENCH]
    fund = set(pd.read_csv(ROOT / "data/derived/analysis/post_earnings_risers_20261120/results/universe.csv").ticker)
    s = pd.Series(idx, index=idx)
    fridays = s.groupby(s.dt.strftime("%G-%V")).max()
    fridays = [d for d in fridays if pd.Timestamp("2025-03-07") <= d <= pd.Timestamp("2026-10-16")]
    pos = {d: i for i, d in enumerate(idx)}
    n = len(idx)
    rows = []
    q = ac["QQQ"]
    for t in fridays:
        i = pos[t]
        if i < 65 or i + 20 >= n:
            continue
        adv = dv[cols].iloc[i - 19:i + 1].mean()
        ok = adv[adv >= 5e7].index
        dvw = dv[ok].iloc[i - 21:i - 1].sum()  # 2세션 앞당긴 미국 거래대금
        win = (sd > t - pd.Timedelta(days=28)) & (sd <= t)
        kb = buy.loc[win].reindex(columns=ok).sum()
        kn = net.loc[win].reindex(columns=ok).sum()
        k3 = buy.loc[win].reindex(columns=ok).notna().sum()
        mom = (ac[ok].iloc[i] / ac[ok].iloc[i - 20] - 1) - (q.iloc[i] / q.iloc[i - 20] - 1)
        vol60 = lr[ok].iloc[i - 59:i + 1].std()
        y20 = (ac[ok].iloc[i + 20] / ao[ok].iloc[i + 1] - 1) - (q.iloc[i + 20] / ao["QQQ"].iloc[i + 1] - 1)
        y60 = ((ac[ok].iloc[i + 60] / ao[ok].iloc[i + 1] - 1) - (q.iloc[i + 60] / ao["QQQ"].iloc[i + 1] - 1)) if i + 60 < n else pd.Series(np.nan, index=ok)
        d = pd.DataFrame({"date": t, "ticker": ok, "K1": 100 * kb / dvw, "K2": 100 * kn / dvw, "K3": k3,
                          "mom20": 100 * mom, "logdv": np.log10(adv[ok]), "vol60": 100 * vol60,
                          "y20": 100 * y20, "y60": 100 * y60, "fund": [x in fund for x in ok]})
        rows.append(d.dropna(subset=["y20", "mom20"]))
    p = pd.concat(rows, ignore_index=True)
    p["half"] = np.where(p.date < "2026-01-01", "2025", np.where(p.date < "2026-07-01", "H1", "H2"))
    return p


def ic_by_week(p, x, y):
    out = []
    for d, g in p.groupby("date"):
        g = g[[x, y]].dropna()
        if len(g) >= 20 and g[x].nunique() > 3:
            out.append((d, g[x].rank().corr(g[y].rank()), len(g)))
    return pd.DataFrame(out, columns=["date", "ic", "n"])


def summarize(w):
    """평균 IC와 겹치지 않는 4주 간격 표본 t(시작 주 4가지 중 중앙값)."""
    if w.empty:
        return {"ic": np.nan, "t": np.nan, "weeks": 0}
    ts = []
    for k in range(4):
        s = w.ic.iloc[k::4]
        ts.append(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan)
    return {"ic": round(float(w.ic.mean()), 4), "t": round(float(np.nanmedian(ts)), 2), "weeks": int(len(w)),
            "n_avg": int(w.n.mean())}


def controlled_ic(p, x, y):
    """거래대금 3분위 × 모멘텀 3분위 칸 안 IC의 건수 가중 평균, 주별."""
    out = []
    for d, g in p.groupby("date"):
        g = g[[x, y, "logdv", "mom20"]].dropna()
        if len(g) < 60:
            continue
        g = g.assign(a=pd.qcut(g.logdv.rank(method="first"), 3, labels=False), b=pd.qcut(g.mom20.rank(method="first"), 3, labels=False))
        ics, ns = [], []
        for _, c in g.groupby(["a", "b"]):
            if len(c) >= 10 and c[x].nunique() > 2:
                ics.append(c[x].rank().corr(c[y].rank()))
                ns.append(len(c))
        if ns:
            out.append((d, float(np.average(ics, weights=ns)), sum(ns)))
    return pd.DataFrame(out, columns=["date", "ic", "n"])


def user_type(p, y, by_size=False):
    """모멘텀 상위 1/3 안: (등장 없음 또는 K1 하위 1/3) − (K1 상위 1/3) 주별 평균 초과수익 차.
    0이 많아 분위가 0으로 뭉치면 '상위 1/3'이 전체가 되므로, 상위 1/3은 K1 > 0 중 큰 순서로 최대 1/3개를 쓴다
    (사전 등록 문구의 구현 보정, 결과 보기 전 정의 그대로의 뜻). by_size: 거래대금 3분위 안에서 같은 비교 후 평균."""
    out = []
    for d, g in p.groupby("date"):
        g = g.dropna(subset=[y])
        top = g[g.mom20 >= g.mom20.quantile(2 / 3)]
        if len(top) < 30:
            continue
        cells = [top] if not by_size else [c for _, c in top.groupby(pd.qcut(top.logdv.rank(method="first"), 3, labels=False))]
        diffs, wl, wh, nl, nh = [], [], [], [], []
        for c in cells:
            k = int(np.ceil(len(c) / 3))
            pos_ = c[c.K1 > 0].sort_values("K1", ascending=False)
            hi = pos_.head(k)
            lo = c[(c.K3 == 0) | (c.K1.rank(method="first") <= k)]
            lo = lo[~lo.index.isin(hi.index)]
            if len(hi) >= 3 and len(lo) >= 3:
                diffs.append(lo[y].mean() - hi[y].mean()); wl.append((lo[y] > 0).mean()); wh.append((hi[y] > 0).mean())
                nl.append(len(lo)); nh.append(len(hi))
        if diffs:
            out.append((d, np.mean(diffs), np.mean(wl), np.mean(wh), sum(nl), sum(nh)))
    return pd.DataFrame(out, columns=["date", "diff", "win_lo", "win_hi", "n_lo", "n_hi"])


def tstat4(series):
    ts = []
    for k in range(4):
        s = series.iloc[k::4]
        ts.append(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan)
    return round(float(np.nanmedian(ts)), 2) if not all(np.isnan(ts)) else np.nan


def main() -> None:
    p = build()
    (BASE / "results").mkdir(exist_ok=True)
    (BASE / "cache").mkdir(exist_ok=True)
    p.to_csv(BASE / "cache/panel.csv", index=False)
    res, weekly = {}, []
    for uname, u in (("넓은 표본", p), ("펀드 161", p[p.fund])):
        for half in ("2025", "H1", "H2"):
            g = u[u.half == half]
            r = {}
            for x in ("K1", "K2", "K3"):
                for y in ("y20", "y60"):
                    w = ic_by_week(g, x, y)
                    r[f"{x}_{y}"] = summarize(w)
                    w = w.assign(universe=uname, half=half, x=x, y=y)
                    weekly.append(w)
            r["K1_same_window(mom20)"] = summarize(ic_by_week(g, "K1", "mom20"))
            r["K1_y20_controlled"] = summarize(controlled_ic(g, "K1", "y20"))
            for nm_, bs in (("user_type_y20", False), ("user_type_y20_size", True)):
                ut = user_type(g, "y20", bs)
                r[nm_] = {"diff_mean": round(float(ut["diff"].mean()), 3) if len(ut) else np.nan, "t4": tstat4(ut["diff"]) if len(ut) else np.nan,
                                  "weeks_pos": f"{int((ut['diff'] > 0).sum())}/{len(ut)}",
                                  "win_lo": round(float(ut.win_lo.mean()), 3) if len(ut) else np.nan,
                                  "win_hi": round(float(ut.win_hi.mean()), 3) if len(ut) else np.nan,
                                  "n_lo": round(float(ut.n_lo.mean()), 1) if len(ut) else np.nan,
                                  "n_hi": round(float(ut.n_hi.mean()), 1) if len(ut) else np.nan}
            r["share_zero_K3"] = round(float((g.K3 == 0).mean()), 3)
            r["stock_weeks"] = int(len(g))
            res[f"{uname}|{half}"] = r
    (BASE / "results/summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.concat(weekly).to_csv(BASE / "results/weekly_ic.csv", index=False)
    for k, r in res.items():
        print(k, "| 주식-주", r["stock_weeks"], "등장없음", r["share_zero_K3"])
        for kk in ("K1_y20", "K1_y20_controlled", "K1_y60", "K2_y20", "K3_y20", "K1_same_window(mom20)"):
            print("   ", kk, r[kk])
        print("    user_type", r["user_type_y20"])
        print("    user_type_size", r["user_type_y20_size"])


if __name__ == "__main__":
    main()
