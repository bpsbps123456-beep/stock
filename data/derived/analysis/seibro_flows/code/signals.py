"""SEIBRO → 신호 표(BRIEF v2 §2·§4·§6). 앞날 가격을 쓰지 않는다(보관 좌수의 평가 종가만 과거 종가를 쓴다).

출력
- derived/sig_daily.parquet: 미국 세션 e(진입)마다 그 시각에 알 수 있는 가장 늦은 신호값(일별 5거래일 합).
- derived/sig_weekly.parquet: 미국 거래 주 창마다 신호값과 진입(설명용).
- derived/sig_meta.json: 절단 비율, 고정 구성(C3) 기록, 진입 대응표 요약.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
NONEQ = {"BOND", "CASH", "CMDTY", "VIX", "CRYPTO", "FX"}
DUMMY = {"US1234567824"}


def nyse():
    import exchange_calendars as xc

    return xc.get_calendar("XNYS", start="2022-12-01", end="2027-03-01").sessions


def ksd():
    import exchange_calendars as xc

    s = xc.get_calendar("XKRX", start="2022-12-01", end="2027-03-01").sessions
    return s[s != pd.Timestamp("2026-06-03")]


def known(d, ksd_days, n=1) -> pd.Timestamp:
    i = ksd_days.searchsorted(pd.Timestamp(d), side="right")
    return ksd_days[i + n - 1]


def entry_of(k: pd.Timestamp, sessions) -> pd.Timestamp:
    return sessions[sessions.searchsorted(k)]


def load_classes() -> pd.DataFrame:
    c = pd.read_csv(ROOT / "classes.csv").set_index("isin")
    for k in ["semi", "single"]:
        c[k] = c[k].astype(str).str.lower().eq("true")
    c["equity"] = ~c["utype"].isin(NONEQ)
    c["lev_long"] = c["equity"] & (c["sign"] == 1) & (c["L"] >= 1.25)
    c["EQ3L"] = c["lev_long"] & (c["L"] >= 3)
    c["IDX"] = c["utype"].eq("EQ_INDEX")
    c["EQ_INV"] = c["equity"] & (c["sign"] == -1)
    c["spot"] = c["equity"] & (c["sign"] == 0)
    m = pd.read_csv(DER / "isin_master.csv").set_index("isin")
    c["first_seen"] = pd.to_datetime(m["first_seen"].reindex(c.index))
    c["product_first"] = c.groupby("product")["first_seen"].transform("min").fillna(c["first_seen"])
    return c


def flags(isins: pd.Series, cls: pd.DataFrame) -> pd.DataFrame:
    f = cls.reindex(isins.unique())
    f["product"] = f["product"].fillna("")
    f[["equity", "spot"]] = f[["equity", "spot"]].fillna(True)  # 분류 밖 = 보통주(주식, 현물)
    f["product_first"] = f["product_first"].fillna(pd.Timestamp("2000-01-01"))
    for k in ["lev_long", "EQ3L", "IDX", "EQ_INV", "semi", "single"]:
        f[k] = f[k].fillna(False).astype(bool)
    return f


IPO = {k: pd.Timestamp(v["first_trade"]) for k, v in json.loads((DER / "ipo.json").read_text(encoding="utf-8")).items()}


def ipo_excluded(when: pd.Timestamp) -> set:
    """창의 마지막 거래일이 첫 거래 뒤 28일 안인 신규 상장 ISIN(분모에서만 뺀다)."""
    return {i for i, d in IPO.items() if d <= when <= d + pd.Timedelta(days=28)}


def window_values(p: pd.DataFrame, fl: pd.DataFrame, net_basis: str, fixed_before=None, excl=frozenset()) -> dict:
    """한 창(또는 5일 묶음)의 ISIN별 합 p[isin, buy, sell, net, in_d1, in_d4]로 신호 원재료를 만든다."""
    p = p.join(fl, on="isin")
    eq = p[p["equity"] & ~p["isin"].isin(DUMMY)]
    den = eq[~eq["isin"].isin(excl)]
    prod = eq.groupby("product")[["buy", "sell", "net"]].sum()

    def g(name, col):
        return float(prod.loc[name, col]) if name in prod.index else np.nan

    v = {}
    v["N"] = float(den["net"].clip(lower=0).sum()) if net_basis == "union" else float(den.loc[den["in_d4"], "net"].sum())
    v["B"] = float(den["buy_d1"].sum())
    for name in ["SOXL", "SOXS", "KORU", "TQQQ", "NVDA", "NVDL"]:
        for col in ["buy", "sell", "net"]:
            v[f"{name}_{col}"] = g(name, col)
    li = eq[eq["lev_long"] & eq["IDX"]]
    v["EQ3L_net"], v["EQ3L_buy"] = float(eq.loc[eq["EQ3L"], "net"].sum()), float(eq.loc[eq["EQ3L"], "buy"].sum())
    v["LIDX_net"], v["LIDX_buy"] = float(li["net"].sum()), float(li["buy"].sum())
    v["INVIDX_buy"] = float(eq.loc[eq["EQ_INV"] & eq["IDX"], "buy"].sum())
    # 고정 품목판(강건 ⑥): 기준 창(약 1년) 시작 전에 처음 보인 상품만
    if fixed_before is not None:
        old = eq["product_first"] < fixed_before
        v["EQ3L_net_fx"] = float(eq.loc[eq["EQ3L"] & old, "net"].sum())
        v["EQ3L_buy_fx"] = float(eq.loc[eq["EQ3L"] & old, "buy"].sum())
        v["LIDX_net_fx"] = float(li.loc[li["product_first"] < fixed_before, "net"].sum())
        v["LIDX_buy_fx"] = float(li.loc[li["product_first"] < fixed_before, "buy"].sum())
        v["INVIDX_buy_fx"] = float(eq.loc[eq["EQ_INV"] & eq["IDX"] & old, "buy"].sum())
    v["SPOT_buy"] = float(den.loc[den["spot"], "buy"].sum())
    nets = eq.groupby("product")["net"].sum()
    nets = nets[nets.index != ""]
    v["K1"] = float(len(nets) > 0 and nets.idxmax() == "SOXL")
    return v


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=df.index)
    N, B = df["N"], df["B"]
    s["S1"] = df["SOXL_net"].fillna(0) / N
    s["S2"] = df["SOXL_buy"].fillna(0) / B
    s["S3"] = df["EQ3L_net"] / N
    s["S4"] = df["EQ3L_buy"] / B
    s["S5n"] = df["LIDX_net"] / N
    s["S5b"] = df["LIDX_buy"] / B
    s["S7"] = df["SOXL_buy"] / (df["SOXL_buy"] + df["SOXS_buy"].fillna(0))
    s["S8"] = df["LIDX_buy"] / (df["LIDX_buy"] + df["INVIDX_buy"])
    s["S9n"] = df["KORU_net"].fillna(0) / N
    s["S9b"] = df["KORU_buy"].fillna(0) / B
    s["S10n"] = df["TQQQ_net"].fillna(0) / N
    s["S10b"] = df["TQQQ_buy"].fillna(0) / B
    s["L1"] = df["SOXL_buy"] / df["NVDA_buy"]
    s["L2"] = df["LIDX_buy"] / df["SPOT_buy"]
    s["Ra"] = (df["SOXL_buy"] + df["SOXL_sell"]) / (df["SOXL_buy"] + df["SOXL_sell"]
                                                    + (df["SOXS_buy"] + df["SOXS_sell"]).fillna(0))
    s["Rb_raw"] = df["NVDL_net"].fillna(0)
    for name in ["KORU", "SOXS", "NVDL"]:
        s[f"cens_{name}"] = df[f"{name}_buy"].isna().astype(float)
    return s


def daily_panel(cls: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cal = pd.read_csv(DER / "seibro_calendar.csv", parse_dates=["seibro_date"])
    cal = cal[cal["n_pred"] > 0].copy()
    cal["last_trade"] = pd.to_datetime(cal["trades"].str.split(";").str[-1])
    p = pd.read_parquet(DER / "settle_daily.parquet")
    p["date"] = pd.to_datetime(p["start"])
    tot = pd.read_parquet(DER / "totals_daily.parquet")
    # B는 그날 매수 Top50 목록 안 주식의 매수 합. ISIN별로 그 날 매수 목록에 들었을 때만 매수를 센다.
    p["buy_d1"] = p["buy"].where(p["in_d1"], 0.0)
    fl = flags(p["isin"], cls)
    trades = sorted(cal["last_trade"].unique())
    by_date = {d: g for d, g in p.groupby("date")}
    rows = []
    for _, r in cal.iterrows():
        t_end = r["last_trade"]
        i = trades.index(t_end)
        t_start = trades[max(0, i - 4)]
        dates = cal.loc[(cal["last_trade"] >= t_start) & (cal["last_trade"] <= t_end), "seibro_date"]
        if i < 4:
            continue
        chunk = pd.concat([by_date[d] for d in dates if d in by_date])
        agg = chunk.groupby("isin").agg(buy=("buy", "sum"), sell=("sell", "sum"), net=("net", "sum"),
                                        buy_d1=("buy_d1", "sum"), in_d4=("in_d4", "any")).reset_index()
        v = window_values(agg, fl, "union", fixed_before=t_start - pd.Timedelta(days=365), excl=ipo_excluded(t_end))
        v.update({"L": r["seibro_date"], "t_start": t_start, "t_end": t_end, "n_dates": len(dates)})
        rows.append(v)
    df = pd.DataFrame(rows)
    df = pd.concat([df, ratios(df)], axis=1)
    return df, {"daily_windows": len(df)}


def weekly_panel(cls: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(DER / "settle_weekly.parquet")
    p["buy_d1"] = p["buy"].where(p["in_d1"], 0.0)
    fl = flags(p["isin"], cls)
    t = pd.read_parquet(DER / "totals_weekly.parquet")
    rows = []
    for (s, e), g in p.groupby(["start", "end"]):
        meta = t[(t["start"] == s) & (t["end"] == e)].iloc[0]
        v = window_values(g, fl, "list", excl=ipo_excluded(pd.Timestamp(meta.get("meta_trade_last"))))
        v.update({"start": s, "end": e, "week": meta.get("meta_week"), "trade_first": meta.get("meta_trade_first"),
                  "trade_last": meta.get("meta_trade_last")})
        rows.append(v)
    df = pd.DataFrame(rows)
    return pd.concat([df, ratios(df)], axis=1)


def custody_panel(cls: pd.DataFrame, sessions, ksd_days) -> pd.DataFrame:
    c = pd.read_parquet(DER / "custody.parquet")
    c["date"] = pd.to_datetime(c["date"])
    fl = flags(c["isin"], cls)
    c = c.join(fl, on="isin")
    c = c[~c["isin"].isin(DUMMY)]
    px = pd.read_csv(ROOT / "raw" / "prices.csv", parse_dates=["date"])
    close = px[px["ticker"] == "SOXL"].set_index("date")["Close"]
    lag = json.loads((DER / "custody_alignment.json").read_text(encoding="utf-8"))["chosen_lag"]
    dates = sorted(c["date"].unique())
    eqsum = c[c["equity"]].groupby("date")["amt"].sum()
    soxl = c[c["product"] == "SOXL"].groupby("date")["amt"].sum()
    li = c[c["lev_long"] & c["IDX"]]
    li_by = li.pivot_table(index="date", columns="product", values="amt", aggfunc="sum")
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df["eq_sum"] = eqsum
    df["SOXL_amt"] = soxl
    df["C2"] = df["SOXL_amt"] / df["eq_sum"]
    # C3 고정 구성: 지난 250 보관일 중 95% 이상 Top50에 든 지수형 레버리지 롱 상품(그날까지의 자료만)
    present = li_by.notna().astype(float)
    share = present.rolling(250, min_periods=120).mean().shift(0)
    c3, comp = [], {}
    for d in df.index:
        members = share.columns[(share.loc[d] >= 0.95).values] if d in share.index else []
        comp[d.date().isoformat()] = list(members)
        c3.append(li_by.loc[d, members].sum() / df.loc[d, "eq_sum"] if len(members) else np.nan)
    df["C3"] = c3
    df["C4"] = df["C3"] - df["C3"].shift(20)
    k = [known(d, ksd_days, 2) for d in df.index]
    vd = [sessions[sessions.searchsorted(kk) - 1 + lag] for kk in k]
    df["val_date"] = vd
    df["SOXL_shares"] = df["SOXL_amt"].values / close.reindex(vd).values
    df["C1"] = np.log(df["SOXL_shares"]).diff(20)
    df["known"] = k
    df["entry"] = [entry_of(kk, sessions) for kk in k]
    df.index.name = "custody_date"
    comp_changes = {}
    prev = None
    for d, m in comp.items():
        if m != prev:
            comp_changes[d] = m
            prev = m
    df.attrs["c3_composition_changes"] = comp_changes
    return df.reset_index()


def to_entry_index(sig: pd.DataFrame, sessions, cols, date_col="entry") -> pd.DataFrame:
    """진입 세션마다 그때 알 수 있는 가장 늦은 값(앞 값 이어 쓰기)."""
    s = sig.sort_values(date_col).groupby(date_col)[cols].last()
    idx = sessions[(sessions >= s.index.min()) & (sessions <= pd.Timestamp("2026-10-23"))]
    return s.reindex(idx).ffill()


def main() -> None:
    sessions, ksd_days = nyse(), ksd()
    cls = load_classes()
    d, meta = daily_panel(cls)
    d["known"] = [known(x, ksd_days, 1) for x in d["L"]]
    d["entry"] = [entry_of(k, sessions) for k in d["known"]]
    w = weekly_panel(cls)
    w["known"] = [known(pd.Timestamp(x), ksd_days, 1) for x in w["end"]]
    w["entry"] = [entry_of(k, sessions) for k in w["known"]]
    cu = custody_panel(cls, sessions, ksd_days)
    scols = ["S1", "S2", "S3", "S4", "S5n", "S5b", "S7", "S8", "S9n", "S9b", "S10n", "S10b", "L1", "L2", "K1", "Ra",
             "Rb_raw", "cens_KORU", "cens_SOXS", "cens_NVDL", "L", "t_start", "t_end", "n_dates",
             "N", "B", "SOXL_net", "SOXL_buy", "SOXS_buy", "EQ3L_net", "EQ3L_buy", "LIDX_net", "LIDX_buy",
             "INVIDX_buy", "SPOT_buy", "KORU_net", "KORU_buy", "TQQQ_net", "TQQQ_buy", "NVDA_buy",
             "EQ3L_net_fx", "EQ3L_buy_fx", "LIDX_net_fx", "LIDX_buy_fx", "INVIDX_buy_fx"]
    de = to_entry_index(d, sessions, scols)
    ce = to_entry_index(cu.rename(columns={"custody_date": "cdate"}), sessions, ["C1", "C2", "C3", "C4", "cdate"])
    panel = de.join(ce, how="left")
    panel["Rb"] = panel["Rb_raw"].rolling(20, min_periods=20).sum()  # NVDL 20결제일 순매수 합(진입 세션 기준 근사)
    panel.index.name = "entry"
    panel.reset_index().to_parquet(DER / "sig_daily.parquet", index=False)
    w.to_parquet(DER / "sig_weekly.parquet", index=False)
    cu.to_parquet(DER / "sig_custody.parquet", index=False)
    meta.update({"entries": int(len(panel)), "first_entry": str(panel.index.min().date()),
                 "c3_composition_changes": cu.attrs.get("c3_composition_changes", {}),
                 "censor_rate_2026": {k: float(panel.loc["2026", f"cens_{k}"].mean()) for k in ["KORU", "SOXS", "NVDL"]}})
    (DER / "sig_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "c3_composition_changes"}, ensure_ascii=False, default=str))
    print("C3 구성 변화", len(meta["c3_composition_changes"]))


if __name__ == "__main__":
    main()
