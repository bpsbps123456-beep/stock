"""설명용 표(BRIEF v2 §9, 판정 아님). 본판정 뒤에 돌린다. 정의를 바꾸지 않는다.

1) 주간 화면표: 미국 거래 주마다 SOXL 순매수 순위·금액·비중, 3배 롱 비중, SOXL 강세 몫, KORU, SOXL 보관, 진입 뒤 1~4주 SOXX
2) 사건 패널: SOXX·EWY의 A·T(2024-07 ~ 2026-10)마다 앞 40·뒤 20세션 백분위 경로, 앞 2주 안 ≥ 80 켜짐(재현율)
3) 2026 헛경보: 켜짐 덩어리 중 뒤 10세션 안 A가 없는 것
4) 지금 vs 7월 사건 직전
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import harness as hz
from events import drawdown_starts, zigzag_tops

ROOT = Path(__file__).resolve().parents[1]
DER, RES = ROOT / "derived", ROOT / "results" / "describe"
SIGS = ["S1", "S3", "S7", "S9n", "C1", "C2"]


def closes(px, t):
    s = px[px["ticker"] == t].set_index("date")
    return s[~s.index.duplicated()]


def weekly_table(px) -> pd.DataFrame:
    w = pd.read_parquet(DER / "sig_weekly.parquet")
    p = pd.read_parquet(DER / "settle_weekly.parquet")
    cls = pd.read_csv(ROOT / "classes.csv").set_index("isin")
    p = p.join(cls[["product"]], on="isin")
    soxl = p[p["product"] == "SOXL"].set_index(["start", "end"])[["rank_d4", "rank_d1", "net", "buy"]]
    w = w.join(soxl, on=["start", "end"])
    cu = pd.read_parquet(DER / "sig_custody.parquet")
    cu["custody_date"] = pd.to_datetime(cu["custody_date"])
    cu["known"] = pd.to_datetime(cu["known"])
    cal = pd.read_csv(DER / "seibro_calendar.csv", parse_dates=["seibro_date"])
    soxx = closes(px, "SOXX")
    sess = soxx.index
    out = []
    for _, r in w.iterrows():
        e = pd.Timestamp(r["entry"])
        if e not in sess:
            continue
        i = sess.get_loc(e)
        o = soxx["Open"].iloc[i]
        fw = {f"SOXX_{k}w": (soxx["Close"].iloc[i + 5 * k - 1] / o - 1) if i + 5 * k - 1 < len(sess) else np.nan
              for k in (1, 2, 4)}
        tl, tf = pd.Timestamp(r["trade_last"]), pd.Timestamp(r["trade_first"])
        prev = soxx["Close"][soxx.index < tf]
        wk = soxx["Close"].get(tl, np.nan) / prev.iloc[-1] - 1 if len(prev) else np.nan
        c = cu[cu["known"] <= e].sort_values("custody_date").tail(1)  # 진입 때 알 수 있는 마지막 보관
        inwin = cal[(cal["seibro_date"] >= pd.Timestamp(r["start"])) & (cal["seibro_date"] <= pd.Timestamp(r["end"]))]
        tds = sorted({t for s in inwin["trades"].dropna() for t in s.split(";")})
        out.append({"week": r["week"], "trades": f"{tf.date()}~{tl.date()}", "entry": e.date(),
                    "trade_days_in_window": len(tds), "trades_actual": f"{tds[0]}~{tds[-1]}" if tds else "",
                    "SOXL_rank_net": r["rank_d4"], "SOXL_net_M": r["net"] / 1e6, "S1_w": r["S1"], "S3_w": r["S3"],
                    "S7_w": r["S7"], "KORU_net_M": (r["KORU_net"] or 0) / 1e6 if pd.notna(r["KORU_net"]) else np.nan,
                    "SOXL_custody_B": float(c["SOXL_amt"].iloc[0]) / 1e9 if len(c) else np.nan,
                    "SOXX_same_week": wk, **fw})
    t = pd.DataFrame(out)
    for s in ["S1_w", "S3_w", "S7_w"]:
        t[s + "_pct52"] = hz.rolling_pct(t[s], 52, 26)
    return t


def event_panels(px, pct: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, paths = [], {}
    for tk in ["SOXX", "EWY"]:
        c = closes(px, tk)["Close"]
        a = [pd.Timestamp(x["A"]) for x in drawdown_starts(c, "2026-10-23")]
        tops, _, _ = zigzag_tops(c, "2026-10-23")
        t = [pd.Timestamp(x["T"]) for x in tops]
        cl = closes(px, tk)["Close"].reindex(pct.index).shift(1)
        plac = hz.rolling_pct(np.log(cl / cl.shift(20)))
        for kind, dates in [("A", a), ("T", t)]:
            for d in dates:
                if d < pd.Timestamp("2024-07-01") or d not in pct.index:
                    continue
                i = pct.index.get_loc(d)
                pre = pct.iloc[max(0, i - 10):i]
                rec = {"target": tk, "kind": kind, "date": d.date().isoformat(),
                       "B2_overlap": bool(pd.Timestamp("2026-07-07") <= d <= pd.Timestamp("2026-08-10"))}
                for s in SIGS:
                    rec[f"{s}_on80_pre2w"] = bool((pre[s] >= 80).any()) if pre[s].notna().any() else None
                    rec[f"{s}_max_pre2w"] = float(pre[s].max()) if pre[s].notna().any() else None
                rec["placebo_on80_pre2w"] = bool((plac.iloc[max(0, i - 10):i] >= 80).any())
                rows.append(rec)
                win = pct.iloc[max(0, i - 40):i + 21][SIGS].copy()
                win["placebo"] = plac.iloc[max(0, i - 40):i + 21]
                win["rel"] = np.arange(len(win)) - min(40, i)
                paths[f"{tk}_{kind}_{d.date()}"] = win.round(1).reset_index().astype(str).to_dict("list")
    return pd.DataFrame(rows), paths


def false_alarms(px, pct: pd.DataFrame) -> pd.DataFrame:
    c = closes(px, "SOXX")["Close"]
    a = [pd.Timestamp(x["A"]) for x in drawdown_starts(c, "2026-10-23")]
    yr = pct.loc["2026-01-02":"2026-10-23"]
    rows = []
    for s in ["S1", "S3", "S7", "C1", "C2"]:
        on = (yr[s] >= 80).values
        idx = np.flatnonzero(on)
        if not len(idx):
            continue
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 5) + 1)
        for g in groups:
            st, en = yr.index[g[0]], yr.index[g[-1]]
            j = pct.index.get_loc(en)
            hit = [d for d in a if st <= d <= pct.index[min(j + 10, len(pct.index) - 1)]]
            rows.append({"signal": s, "start": st.date(), "end": en.date(), "days": len(g),
                         "A_within_10_after": ";".join(str(x.date()) for x in hit)})
    return pd.DataFrame(rows)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    px = hz.load_prices()
    pct = pd.read_parquet(ROOT / "results" / "real" / "pct_daily.parquet")
    pct.index = pd.DatetimeIndex(pct.index)
    wt = weekly_table(px)
    wt.to_csv(RES / "weekly_screen.csv", index=False)
    ev, paths = event_panels(px, pct)
    ev.to_csv(RES / "event_recall.csv", index=False)
    (RES / "event_paths.json").write_text(json.dumps(paths, ensure_ascii=False), encoding="utf-8")
    fa = false_alarms(px, pct)
    fa.to_csv(RES / "false_alarms_2026.csv", index=False)
    raw = pd.read_parquet(DER / "sig_daily.parquet").set_index("entry")
    now = {"entry": "2026-10-23", **{s: float(pct[s].iloc[-1]) for s in SIGS + ["S2", "L1", "C3"]},
           **{f"{s}_raw": float(raw[s].iloc[-1]) for s in ["S1", "S3", "S7", "C2"]}}
    jul = pct.loc[:"2026-07-09"].iloc[-1]
    rawj = raw.loc[:"2026-07-09"].iloc[-1]
    before = {"entry": str(pct.loc[:"2026-07-09"].index[-1].date()), **{s: float(jul[s]) for s in SIGS + ["S2", "L1", "C3"]},
              **{f"{s}_raw": float(rawj[s]) for s in ["S1", "S3", "S7", "C2"]}}
    (RES / "now_vs_july.json").write_text(json.dumps({"now": now, "before_july_A": before}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(ev.groupby(["target", "kind"])[[c for c in ev.columns if c.endswith("_on80_pre2w")]].mean().round(2).to_string())
    print(fa.to_string())
    print(json.dumps({"now": now, "before_july_A": before}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
