"""2차: 넓힌 400종목에서 v3 원안 ①·② 신호(겹침 빼기 전)와 '모양 실패' 맥락 행을 모으고, BRIEF2의 조건 변수 18개를 붙인다.
모든 변수는 신호 봉 마감 때 안다(일봉은 전일까지, 분봉은 신호 봉까지).
사용: py cond_build.py 2026   → cache/cond_2026.parquet
      py cond_build.py 2025   → cache/cond_2025.parquet"""
import os, sys, json, time
import numpy as np
import pandas as pd
import requests
import engine as E
import evaluate as V
from fetch_alpaca import keys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
YEAR = {"2026": dict(src="alp1m", lo=20260701, hi=20261028, daily="daily_2026.parquet"),
        "2025": dict(src="alp1m_2025", lo=20250701, hi=20251031, daily="daily_2025.parquet")}
SPEC = {"A": V.ORIG["A"], "B": V.ORIG["B"]}


def etf_daily(year):
    """QQQ 공식 일봉 종가(전일 종가용)."""
    end_ts = min(pd.Timestamp(f"{year}-11-05", tz="UTC"), pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20))
    start, end = f"{year}-05-01T00:00:00Z", end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")      # 무료 SIP는 15분 지난 끝만 받는다
    j = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=keys(),
                     params={"symbols": "QQQ", "timeframe": "1Day", "start": start, "end": end, "limit": 10000,
                             "adjustment": "split", "feed": "sip"}, timeout=60).json()
    b = j["bars"]["QQQ"]
    return pd.Series([x["c"] for x in b], index=pd.to_datetime([x["t"][:10] for x in b]))


def daily_feats(dd):
    """일봉 → 날짜별(그날 '전일까지') 전일 종가·20일 거래대금 중앙값·5일 수익률·ATR14%."""
    dd = dd.sort_values("date").copy()
    dv = dd.v * dd.vw.fillna(dd.c)
    pc = dd.c.shift(1)
    tr = np.maximum(dd.h - dd.l, np.maximum((dd.h - pc).abs(), (dd.l - pc).abs()))
    tr.iloc[0] = dd.h.iloc[0] - dd.l.iloc[0]
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f = pd.DataFrame({"prevclose": dd.c.shift(1).to_numpy(),
                      "adv20": dv.shift(1).rolling(20, min_periods=15).median().to_numpy(),
                      "ret5d": (dd.c.shift(1) / dd.c.shift(6) - 1).to_numpy(),
                      "atrd": (atr.shift(1) / dd.c.shift(1)).to_numpy()}, index=dd.date.to_numpy())
    return f


def minute_close(q, idx_ts):
    """QQQ 분봉 종가에서 각 시각(유닉스 분) 이하 마지막 값."""
    pos = np.searchsorted(q[0], idx_ts, side="right") - 1
    out = np.where(pos >= 0, q[1][np.clip(pos, 0, None)], np.nan)
    return out


def main(year):
    cfg = YEAR[year]
    uni = json.load(open(os.path.join(RES, "universe2.json"), encoding="utf-8"))
    syms = sorted(uni)
    if os.environ.get("COND_SYMS"):                       # 점검용: 일부 종목만
        syms = [x for x in syms if x in os.environ["COND_SYMS"].split(",")]
    daily = pd.read_parquet(os.path.join(E.CACHE, cfg["daily"]))
    dfe = {s: daily_feats(g) for s, g in daily.groupby("sym")}
    qd = etf_daily(year)
    qm = E.load("QQQ", cfg["src"])
    q_ts = ((qm.index.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy()
    q = (q_ts, qm.Close.to_numpy())
    ev = pd.read_parquet(os.path.join(ROOT, "data/derived/analysis/earnings_support/cache/ev_meta.parquet"))
    ev_tickers = set(ev.ticker)
    ev_days = set(zip(ev.ticker, ev.reaction_date.astype(str).str.replace("-", "").astype(int)))
    rows, minute_parts = [], []
    t0 = time.time()
    for sid, sym in enumerate(syms):
        raw = pd.read_parquet(os.path.join(E.CACHE, cfg["src"], f"{sym}.parquet")) if os.path.exists(
            os.path.join(E.CACHE, cfg["src"], f"{sym}.parquet")) else None
        if raw is None or len(raw) < 2000 or sym not in dfe:
            continue
        d = E.load(sym, cfg["src"])
        a = E.prep(d)
        out = E.bar_outcomes(a)
        inper = (a["day"] >= cfg["lo"]) & (a["day"] <= cfg["hi"])
        v = out["valid_e2"] & out["valid_e0"] & inper
        minute_parts.append(pd.DataFrame({"ts": a["ts"][v], "r1": out["r1_e2"][v]}))
        A, B = E.candidates(sym, a, out, 120, 0.0005, mode="newlow")
        # 당일 VWAP(04:00부터 누적), 시간대 거래량 표
        rr = raw.reindex(d.index)
        pv = (rr.VWAP.fillna((rr.High + rr.Low + rr.Close) / 3) * rr.Volume).to_numpy()
        vol = a["v"]
        day = a["day"]
        newday = np.r_[True, day[1:] != day[:-1]]
        grp = np.cumsum(newday)
        cpv = pd.Series(pv).groupby(grp).cumsum().to_numpy()
        cv = pd.Series(vol).groupby(grp).cumsum().to_numpy()
        vwap = np.where(cv > 0, cpv / np.where(cv > 0, cv, 1), np.nan)
        days = np.unique(day)
        vt = pd.DataFrame({"day": day, "m": a["mins"], "v": vol}).pivot_table(index="day", columns="m", values="v",
                                                                               aggfunc="sum").reindex(days).fillna(0.0)
        vt_med = vt.shift(1).rolling(10, min_periods=10).median()
        f = dfe[sym]
        for pat, X in (("A", A), ("B", B)):
            if X is None or len(X) == 0:
                continue
            sm, cm = V.masks(X, pat, SPEC[pat])
            sane = X["sane"].to_numpy()
            for arm, mk in (("sig", sm & sane), ("ctx", cm & sane)):
                Y = X[mk].copy()
                if len(Y) == 0:
                    continue
                # 연쇄: 그날 같은 종목·같은 묶음의 앞선 신호 수(겹침 빼기 전, 시간대 무관)
                Y = Y.sort_values("ts")
                Y["chain"] = Y.groupby("day").cumcount()
                Y = Y[Y.valid & Y.tradable & (Y.day >= cfg["lo"]) & (Y.day <= cfg["hi"])]
                if len(Y) == 0:
                    continue
                t = Y.t.to_numpy().astype(int)
                dts = pd.to_datetime(Y.day.astype(str), format="%Y%m%d")
                fi = f.reindex(dts)
                Y["arm"] = arm
                Y["pat"] = pat
                Y["sym"] = sym
                Y["sid"] = sid
                Y["ai"] = uni[sym]["ai"]
                Y["prevclose"] = fi.prevclose.to_numpy()
                Y["cap_b"] = uni[sym]["shares"] * Y.prevclose / 1e9
                Y["adv20"] = fi.adv20.to_numpy()
                Y["ret5d"] = fi.ret5d.to_numpy()
                Y["atrd"] = fi.atrd.to_numpy()
                Y["mins"] = a["mins"][t]
                Y["dayret"] = a["l"][t] / Y.prevclose - 1
                qpc = qd.shift(1).reindex(dts).to_numpy()
                tsx = a["ts"][t]
                qnow = minute_close(q, tsx)
                q30 = minute_close(q, tsx - 30)
                Y["qqq_day"] = qnow / qpc - 1
                Y["qqq_30"] = qnow / q30 - 1
                Y["rel"] = (a["c"][t] / Y.prevclose - 1) - Y.qqq_day
                med = np.array([vt_med.at[dd_, mm] if (mm in vt_med.columns) else np.nan
                                for dd_, mm in zip(Y.day.to_numpy(), Y.mins.to_numpy())])
                Y["vol_tod"] = np.where(med > 0, vol[t] / np.where(med > 0, med, 1), np.inf)
                Y["vwap_gap"] = a["c"][t] / vwap[t] - 1
                Y["recsize"] = ((np.minimum(a["o"][t], a["c"][t]) - a["l"][t]) / a["c"][t]) if pat == "A" else (
                    (a["c"][t] - a["o"][t]) / a["c"][t])
                Y["kcnt"] = Y["level2"] if pat == "A" else Y["touch2"]
                Y["earn"] = [1 if (sym, int(x)) in ev_days else (0 if sym in ev_tickers else -1) for x in Y.day]
                Y["sess_in"] = a["sess"][t + 1]
                Y["sess_out"] = a["sess"][out["zz"][t]]
                rows.append(Y)
        if sid % 25 == 0:
            print(year, sid, sym, f"{time.time() - t0:.0f}s", flush=True)
    R = pd.concat(rows, ignore_index=True)
    mm = pd.concat(minute_parts, ignore_index=True)
    g = mm.groupby("ts").r1.agg(["sum", "count"])
    j = g.reindex(R.ts.to_numpy())
    own = R.r1_e2.to_numpy()
    R["bt2_r1"] = (j["sum"].to_numpy() - own) / (j["count"].to_numpy() - 1)
    keep = ["arm", "pat", "sym", "sid", "ts", "day", "t", "k", "sess", "mins", "ai", "prevclose", "cap_b", "adv20",
            "ret5d", "atrd", "dayret", "qqq_day", "qqq_30", "rel", "vol_tod", "vwap_gap", "dd", "chain", "kcnt", "recsize",
            "earn", "r1_e2", "r1net_e2", "hit_e2", "hits_e2", "race_e2", "mfe_e2", "mae_e2", "r2_e2", "r1_e0", "hit_e0",
            "sess_in", "sess_out", "bt2_r1"]
    R[keep].to_parquet(os.path.join(E.CACHE, f"cond_{year}{os.environ.get('COND_TAG', '')}.parquet"))
    print("done", year, len(R), R.groupby(["pat", "arm"]).size().to_dict(), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
