"""PREREG3 넓은 표본 요인·결과 표 → results/panel_wide.csv

사건: 넓은 연구 사건표 2026(src=wide) + 펀드 사건표 2026(src=fund). 같은 함수로 같은 가격(cache/w_*)에서 계산한다.
가격 마지막 날은 2026-11-18(yfinance 11-19 행 일부 결측).
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
CACHE = BASE / "cache"
LASTDAY = "2026-11-18"
SEC_ETF = {"Technology": "XLK"}


def load_px():
    px = {k: pd.read_parquet(CACHE / f"w_{k}.parquet") for k in ("open", "high", "low", "close", "adjclose", "volume")}
    idx = px["close"]["QQQ"].dropna().index
    idx = idx[idx <= LASTDAY]
    px = {k: v.reindex(idx) for k, v in px.items()}
    f = px["adjclose"] / px["close"]
    px["aopen"] = px["open"] * f
    px["ahigh"] = px["high"] * f
    px["alow"] = px["low"] * f
    return px, idx


def occ_panel(tickers):
    parts = []
    for f in sorted(glob.glob(str(ROOT / "data/derived/analysis/options_sentiment/cache/occ/202[56]-*.parquet"))):
        d = pd.read_parquet(f)
        d = d[d.underlying.isin(tickers)]
        d.insert(0, "date", pd.Timestamp(Path(f).stem))
        parts.append(d)
    o = pd.concat(parts, ignore_index=True)
    o["tot"] = o[["C_C", "C_P", "F_C", "F_P", "M_C", "M_P"]].sum(axis=1)
    return {t: g.set_index("date").sort_index() for t, g in o.groupby("underlying")}


def finra():
    """결제일별 전 종목 파일 + 공표일(Astra 수집표의 결제일→공표일 대응)."""
    pub = pd.read_parquet(BASE / "astra/data/short_interest.parquet")[["settlement_date", "publish_date"]].drop_duplicates()
    pub = dict(zip(pub.settlement_date, pub.publish_date))
    rows = []
    for f in sorted(glob.glob(str(BASE / "astra/cache/finra/*.csv"))):
        d = pd.read_csv(f, sep="|", dtype={"symbolCode": str})
        sd = Path(f).stem
        p = pub.get(sd)
        if p is None:  # 대응 없으면 결제일 + 8영업일로 보수적으로
            p = str((pd.Timestamp(sd) + pd.offsets.BDay(8)).date())
        d = d[d.stockSplitFlag.isna()]
        rows.append(pd.DataFrame({"ticker": d.symbolCode.str.replace(".", "-", regex=False), "pub": pd.Timestamp(p),
                                  "dtc": d.daysToCoverQuantity, "cur": d.currentShortPositionQuantity,
                                  "prev": d.previousShortPositionQuantity}))
    s = pd.concat(rows)
    return {t: g.sort_values("pub") for t, g in s.groupby("ticker")}


def seibro(tickers):
    S = ROOT / "data/derived/analysis/seibro_flows"
    m = pd.read_csv(S / "derived/isin_master.csv", encoding="utf-8-sig")
    i2t = {i: str(t).replace("/", "-") for i, t in zip(m["isin"], m["figi_ticker"]) if isinstance(t, str)}
    rec = {}
    for f in glob.glob(str(S / "raw/daily/d*/2025*.json")) + glob.glob(str(S / "raw/daily/d*/2026*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for r in d["rows"]:
            t = i2t.get(r["isin"])
            if t in tickers:
                rec[(d["start"], t)] = r["buy"]
    df = pd.DataFrame([(pd.Timestamp(k[0]), k[1], v) for k, v in rec.items()], columns=["sdate", "ticker", "buy"])
    return {t: g.set_index("sdate").sort_index() for t, g in df.groupby("ticker")}


def main() -> None:
    px, idx = load_px()
    n = len(idx)
    ac, ao, ah, al, vo, cl = px["adjclose"], px["aopen"], px["ahigh"], px["alow"], px["volume"], px["close"]
    lr = np.log(ac / ac.shift(1))
    intra = np.log(ac / ao)
    over = np.log(ao / ac.shift(1))
    wide = pd.read_csv(BASE.parent / "earnings_trend/results/events_wide.csv")
    wide = wide[(wide.reaction_date >= "2026-01-02")]
    fund = pd.read_csv(BASE / "results/events.csv")
    fsec = wide.drop_duplicates("ticker").set_index("ticker")[["sector", "etf"]]
    ffeat = pd.read_csv(BASE / "results/features.csv").set_index(["ticker", "reaction_date"])
    evs = [pd.DataFrame({"src": "wide", "ticker": wide.ticker, "reaction_date": wide.reaction_date, "sector": wide.sector,
                         "etf": wide.etf, "mcap_b": wide.mcap_b, "eps_surprise": wide.eps_surprise,
                         "rev_yoy": wide.rev_yoy, "rev_accel": wide.rev_accel}),
           pd.DataFrame({"src": "fund", "ticker": fund.ticker, "reaction_date": fund.reaction_date,
                         "sector": fund.ticker.map(fsec.sector), "etf": fund.ticker.map(fsec.etf),
                         "mcap_b": [ffeat.px_mcap_b.get((t, r), np.nan) for t, r in zip(fund.ticker, fund.reaction_date)],
                         "eps_surprise": fund.eps_surprise, "rev_yoy": np.nan, "rev_accel": np.nan})]
    ev = pd.concat(evs, ignore_index=True)
    ev = ev[ev.ticker.isin(ac.columns)]
    ev["etf"] = ev.etf.fillna("QQQ").replace({"SPY": "QQQ"})
    ev.loc[~ev.etf.isin(ac.columns), "etf"] = "QQQ"
    same_day = wide.groupby("reaction_date").size()
    tickers = set(ev.ticker)
    occ = occ_panel(tickers)
    fin = finra()
    kr = seibro(tickers)
    q, vix, sox = ac["QQQ"], ac["^VIX"] if "^VIX" in ac else cl["^VIX"], ac["SOXX"]
    q50 = q.rolling(50).mean()
    pos = {d: i for i, d in enumerate(idx)}
    rows = []
    for e in ev.itertuples():
        k = pos.get(pd.Timestamp(e.reaction_date))
        if k is None or k < 253:
            continue
        t = e.ticker
        c, v = ac[t], vo[t]
        if np.isnan(c.iloc[k]) or np.isnan(c.iloc[k - 1]):
            continue
        r = {"src": e.src, "ticker": t, "reaction_date": e.reaction_date, "k": k, "sector": e.sector, "etf": e.etf,
             "mcap_b": e.mcap_b, "eps_surprise": e.eps_surprise, "rev_yoy": e.rev_yoy, "rev_accel": e.rev_accel}
        qq = lambda a, b: q.iloc[b] / q.iloc[a] - 1  # noqa: E731
        for h in (20, 60):
            r[f"pre{h}_x"] = 100 * (c.iloc[k - 1] / c.iloc[k - 1 - h] - 1 - qq(k - 1 - h, k - 1))
        r["dist_hi"] = 100 * (c.iloc[k - 1] / ah[t].iloc[k - 252:k].max() - 1)
        r["vol60"] = 100 * lr[t].iloc[k - 60:k].std()
        base = v.iloc[k - 65:k - 5].median()
        r["vsurge5"] = np.log(v.iloc[k - 5:k].mean() / base) if base and base > 0 else np.nan
        r["dvol20"] = np.log10(max((cl[t] * v).iloc[k - 20:k].mean(), 1))
        r["intra20"] = 100 * float((intra[t].iloc[k - 20:k] - intra["QQQ"].iloc[k - 20:k]).sum())
        r["over20"] = 100 * float((over[t].iloc[k - 20:k] - over["QQQ"].iloc[k - 20:k]).sum())
        lw = lr[t].iloc[k - 20:k]
        r["eff20"] = abs(lw.sum()) / lw.abs().sum() if lw.abs().sum() > 0 else np.nan
        r["r0_x"] = 100 * (c.iloc[k] / c.iloc[k - 1] - 1 - qq(k - 1, k))
        r["gap"] = 100 * (ao[t].iloc[k] / c.iloc[k - 1] - 1)
        r["o2c"] = 100 * (c.iloc[k] / ao[t].iloc[k] - 1)
        rng = ah[t].iloc[k] - al[t].iloc[k]
        r["clv"] = (c.iloc[k] - al[t].iloc[k]) / rng if rng > 0 else np.nan
        m50 = v.iloc[k - 50:k].median()
        r["vol0"] = v.iloc[k] / m50 if m50 and m50 > 0 else np.nan
        r["r0_sig"] = r["r0_x"] / r["vol60"] if r["vol60"] else np.nan
        r["qqq_pre20"] = 100 * qq(k - 21, k - 1)
        r["qqq_above50"] = bool(q.iloc[k - 1] > q50.iloc[k - 1])
        r["vix"] = vix.iloc[k - 1]
        r["soxx_rel20"] = 100 * (sox.iloc[k - 1] / sox.iloc[k - 21] - 1 - qq(k - 21, k - 1))
        r["same_day_n"] = int(same_day.get(e.reaction_date, 0))
        r["react_mon"] = pd.Timestamp(e.reaction_date).weekday() == 0  # 금요일 장 후·주말 발표 → 월요일 반응
        ob = occ.get(t)
        if ob is not None:
            w5 = ob.loc[(ob.index > idx[k - 6]) & (ob.index <= idx[k - 1])]
            pr = ob.loc[(ob.index > idx[k - 66]) & (ob.index <= idx[k - 6])]
            if len(w5) >= 3 and len(pr) >= 30:
                r["occ_tot_ab"] = np.log((w5.tot.mean() + 1) / (pr.tot.median() + 1))
                r["occ_call_ab"] = np.log((w5.C_C.mean() + 1) / (pr.C_C.median() + 1))
                r["occ_putshare"] = (w5.C_P.sum() + w5.F_P.sum() + w5.M_P.sum()) / max(w5.tot.sum(), 1)
        fb = fin.get(t)
        if fb is not None:
            p = fb[fb.pub <= idx[k - 1]]
            if len(p):
                r["dtc"] = p.dtc.iloc[-1]
                r["si_ch"] = p.cur.iloc[-1] / p.prev.iloc[-1] - 1 if p.prev.iloc[-1] > 0 else np.nan
        kb = kr.get(t)
        win = kb.loc[(kb.index > idx[k - 22]) & (kb.index <= idx[k - 1])] if kb is not None else None
        r["kr_days_pre"] = 0 if win is None else len(win)
        dv = (cl[t] * v).iloc[k - 21:k - 1].sum()
        r["kr_buy_share"] = 0.0 if win is None or not dv else 100 * win.buy.sum() / dv
        # E6 전용(+5 종가에 앎)
        if k + 5 < n:
            r["d5_x"] = 100 * (c.iloc[k + 5] / ao[t].iloc[k + 1] - 1 - (q.iloc[k + 5] / ao["QQQ"].iloc[k + 1] - 1))
            lo5, hi5 = al[t].iloc[k + 1:k + 6].min(), ah[t].iloc[k + 1:k + 6].max()
            r["str5"] = (c.iloc[k + 5] - lo5) / (hi5 - lo5) if hi5 > lo5 else np.nan
        # 결과
        se = e.etf
        for entry, s0, off in (("e1", 1, 0), ("e6", 6, 5)):
            for h in (5, 10, 20):
                end = k + off + h
                if end < n and k + s0 < n:
                    a0 = ao[t].iloc[k + s0]
                    rr = c.iloc[end] / a0 - 1
                    r[f"{entry}_h{h}"] = 100 * (rr - (q.iloc[end] / ao["QQQ"].iloc[k + s0] - 1))
                    r[f"{entry}_h{h}_s"] = 100 * (rr - (ac[se].iloc[end] / ao[se].iloc[k + s0] - 1))
        rows.append(r)
    p = pd.DataFrame(rows)
    p["fund_ticker"] = p.ticker.isin(set(fund.ticker))
    p.to_csv(BASE / "results/panel_wide.csv", index=False)
    print(p.shape, p.src.value_counts().to_dict())
    print(p.groupby("src").reaction_date.agg(["min", "max"]))
    print(p.notna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
