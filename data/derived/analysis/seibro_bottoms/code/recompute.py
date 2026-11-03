"""B15 바닥 자료판 독립 재계산과 derived/ 대조(results/recompute/report.md).

compute: build.py·derived/를 읽지 않고 원자료(../seibro_flows/raw)·classes.csv·ipo.json·yfinance 가격만으로 다시 센다.
  - 결제 SEIBRO 날짜 <= 2026-11-03, 보관 날짜 <= 2026-11-02, 가격 <= 2026-11-03, 지그재그 <= 2026-11-02(BRIEF §3).
  - 사건(B15·B10·F), 사건 k=-5..+5 날짜 대응·일별 값, W2·W3의 X4·H2와 P, 보관 좌수·지연 추정.
  - 묶음(BRIEF §2·§10): 예측 이틀치 SEIBRO 날짜는 거래일 수로 나눈다. 빈 날이 낀 어긋남만 이웃과 합친다(이번 자료 0건).
    관측 double·예측 normal(거래 급증)은 합치지 않는다. 예측 거래일 없는 날짜의 금액은 버린다(양을 보고).
  - N+: 상품(사슬) 단위로 순매수를 합친 뒤 양수만 더한다. 창 N+ = 일별 N+의 합.
  - 결과: results/recompute/mine.json (+ 대조용 중간표 _me.parquet·_cu.parquet·_tmap.parquet, 가격 prices_yf.csv)
compare: mine.json과 derived/(events.json, units.parquet, measures_daily.parquet, measures_win.parquet,
  custody_daily.parquet)를 맞대 results/recompute/compare.csv를 만든다.
실행: py -B code/recompute.py all   (compute → compare)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PRIOR = HERE.parent / "seibro_flows"
RAW = PRIOR / "raw"
OUT = HERE / "results" / "recompute"
sys.path.insert(0, str(PRIOR / "code"))
from calendar_map import calendars, seibro_of  # noqa: E402  날짜 규약은 앞선 연구 그대로
from events import zigzag_tops  # noqa: E402

SETTLE_MAX = "2026-11-03"  # 결제 SEIBRO 날짜 끝(미국 거래 10/30)
CUST_MAX = "2026-11-02"  # 보관 날짜 끝
PX_END = "2026-11-03"  # 가격 끝
ZZ_END = "2026-11-02"  # 지그재그 끝(BRIEF §3)
SOXL = "US25459W4583"
NONEQ = {"BOND", "CASH", "CMDTY", "VIX", "CRYPTO", "FX"}
DUMMY = {"US1234567824"}


# ---------------- 가격 ----------------
def prices() -> pd.DataFrame:
    f = OUT / "prices_yf.csv"
    if not f.exists():
        import yfinance as yf

        out = []
        for tk in ["SOXX", "SOXL", "SOXS"]:
            d = yf.download(tk, start="2022-01-01", end="2026-11-04", auto_adjust=False, progress=False,
                            actions=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.reset_index().rename(columns={"Date": "date", "Adj Close": "AdjClose"})
            d["ticker"] = tk
            out.append(d[["date", "ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]])
        pd.concat(out).to_csv(f, index=False)
    px = pd.read_csv(f, parse_dates=["date"])
    return px[px["date"] <= PX_END]


def closes(px: pd.DataFrame, tk: str) -> pd.Series:
    return px[px["ticker"] == tk].set_index("date")["Close"].sort_index()


# ---------------- 사건 ----------------
def events(soxx: pd.Series) -> dict:
    c = soxx[(soxx.index >= "2022-10-01") & (soxx.index <= ZZ_END)]
    pos = {d: i for i, d in enumerate(soxx.index)}
    t15, b15, pend15 = zigzag_tops(c, pct=0.15)
    t10, b10, pend10 = zigzag_tops(c, pct=0.10)
    tops15 = [pd.Timestamp(x["T"]) for x in t15]
    tops10 = [pd.Timestamp(x["T"]) for x in t10]

    def prev_top(b, tops):
        p = [t for t in tops if t < b]
        return p[-1] if p else None

    first_trade = pd.Timestamp("2022-12-28")  # SEIBRO 첫 날짜(2023-01-02)에 담긴 거래일
    b15_all = []
    for x in b15:
        b = pd.Timestamp(x["B"])
        tp = prev_top(b, tops15)
        b15_all.append({"date": x["B"], "confirmed": x["confirmed"], "close": float(soxx[b]),
                        "top": tp.date().isoformat() if tp is not None else None,
                        "drop": float(soxx[b] / soxx[tp] - 1) if tp is not None else None})
    # 10% 바닥: 15세션 안에 이어지면 더 낮은 하나
    raw10 = [pd.Timestamp(x["B"]) for x in b10]
    merged, absorbed = [], []
    for b in raw10:
        if merged and pos[b] - pos[merged[-1]] <= 15:
            keep, drop = (b, merged[-1]) if soxx[b] < soxx[merged[-1]] else (merged[-1], b)
            absorbed.append({"kept": keep.date().isoformat(), "dropped": drop.date().isoformat(),
                             "gap_sessions": int(abs(pos[b] - pos[merged[-1]]))})
            merged[-1] = keep
        else:
            merged.append(b)
    b10_in = [b for b in merged if b > first_trade]
    b15_in = [x for x in b15_all if pd.Timestamp(x["date"]) > first_trade]

    # F 가짜 바닥(①~④)
    v, idx = soxx.values, soxx.index
    bpos = [pos[b] for b in b10_in]
    F, undet = [], []
    for i in range(len(v)):
        t = idx[i]
        if t <= first_trade:
            continue
        c1 = v[i] == v[max(0, i - 9):i + 1].min()
        c2 = v[i] <= 0.9 * v[max(0, i - 59):i + 1].max()  # 그날 포함 60세션
        c2x = v[i] <= 0.9 * v[max(0, i - 60):i].max()  # 그날 뺀 60세션(민감도)
        fut = v[i + 1:i + 21]
        c3 = len(fut) > 0 and fut.min() < 0.97 * v[i]
        c4 = all(abs(i - p) > 5 for p in bpos)
        if c1 and c2 and c4 and not c3 and len(fut) < 20:
            undet.append(t.date().isoformat())
        if c1 and c2 and c3 and c4:
            tp = prev_top(t, tops10)
            nb = [b for b in b10_in if b >= t]
            nb = nb[0] if nb else None
            ph_ok = nb is not None and prev_top(nb, tops10) == tp
            cur = nb is None and tp is not None and tp == tops10[-1]  # 아직 바닥이 안 난 지금 하락
            F.append({"date": t.date().isoformat(), "close": float(v[i]), "fwd_sessions": int(len(fut)),
                      "c2_excl_same": bool(c2x), "top10": tp.date().isoformat() if tp is not None else None,
                      "phase_bottom": nb.date().isoformat() if (nb is not None and ph_ok) else ("current" if cur else None)})
    return {"b15": b15_in, "b15_out_of_data": [x["date"] for x in b15_all if x not in b15_in],
            "b10": [b.date().isoformat() for b in b10_in],
            "b10_out_of_data": [b.date().isoformat() for b in merged if b <= first_trade],
            "b10_absorbed": absorbed, "b10_raw": [b.date().isoformat() for b in raw10],
            "tops10": [t.date().isoformat() for t in tops10], "tops15": [t.date().isoformat() for t in tops15],
            "F": F, "F_undetermined": undet, "pending15": pend15, "pending10": pend10}


# ---------------- 결제 원자료 ----------------
def load_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    recs, empty = [], []
    for d in "1234":
        for p in sorted((RAW / "daily" / f"d{d}").glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["start"] > SETTLE_MAX:
                continue
            assert r["start"] == r["end"]
            if not r["rows"]:
                empty.append((r["start"], d))
            for row in r["rows"]:
                recs.append({"date": r["start"], "dtype": d, **row})
    rows = pd.DataFrame(recs)
    rows["date"] = pd.to_datetime(rows["date"])
    dates = sorted(set(rows["date"]) | {pd.Timestamp(x[0]) for x in empty})
    tot = pd.DataFrame(index=pd.DatetimeIndex(dates))
    tot["A1"] = rows[rows["dtype"] == "3"].groupby("date")["tot"].sum()
    tot["A2"] = rows[rows["dtype"] == "1"].groupby("date")["buy"].sum()
    tot["A3"] = rows[rows["dtype"] == "2"].groupby("date")["sell"].sum()
    return rows, tot.fillna(0.0)


def seibro_calendar(tot: pd.DataFrame) -> pd.DataFrame:
    nyse, settle = calendars()
    last = tot.index.max().date()
    trades = [t for t in nyse if dt.date(2022, 12, 26) <= t <= last]
    m: dict = {}
    for t in trades:
        m.setdefault(seibro_of(t, settle), []).append(t)
    cal = pd.DataFrame({"seibro": tot.index})
    cal["trades"] = cal["seibro"].map(lambda d: [pd.Timestamp(x) for x in m.get(d.date(), [])])
    cal["n_pred"] = cal["trades"].map(len)
    cal["A1"] = tot["A1"].values
    single = cal["A1"].where(cal["n_pred"] == 1)
    cal["ratio"] = cal["A1"] / single.rolling(41, center=True, min_periods=10).median()
    cal["prev_ratio"] = cal["ratio"].shift(1)
    cal["next_ratio"] = cal["ratio"].shift(-1)
    cal["prev_n"] = cal["n_pred"].shift(1)
    cal["next_n"] = cal["n_pred"].shift(-1)
    cal["obs"] = pd.cut(cal["ratio"], [-1, 0.3, 1.6, 99], labels=["empty", "normal", "double"]).astype(str)
    cal["pred"] = cal["n_pred"].map(lambda n: "empty" if n == 0 else "double" if n >= 2 else "normal")
    # 묶음: 기본은 날짜 하나. 빈 날이 낀 어긋남(obs != pred, 한쪽이 empty)만 더 어긋난 이웃 날짜와 합친다
    unit = list(range(len(cal)))
    mis = (cal["obs"] != cal["pred"]) & ((cal["obs"] == "empty") | (cal["pred"] == "empty"))
    for i in np.where(mis.values)[0]:
        cand = [j for j in (i - 1, i + 1) if 0 <= j < len(cal)]
        j = max(cand, key=lambda j: abs(np.log(max(cal["ratio"].iloc[j], 1e-3)) - np.log(max(cal["n_pred"].iloc[j], 1e-3))))
        a, b = unit[i], unit[j]
        unit = [min(a, b) if u in (a, b) else u for u in unit]
    cal["unit"] = [cal["seibro"].iloc[u] for u in unit]
    cal["empty_mismatch"] = mis.values
    return cal


def calendar_check(cal: pd.DataFrame) -> dict:
    e = cal[cal["n_pred"] == 0]
    d2 = cal[cal["n_pred"] >= 2]
    s1 = cal[cal["n_pred"] == 1]
    hi = s1[s1["ratio"] > 1.6]
    low_nb = ((hi["prev_ratio"] < 0.5) & (hi["prev_n"] > 0)) | ((hi["next_ratio"] < 0.5) & (hi["next_n"] > 0))
    return {"dates": int(len(cal)), "pred_empty": int(len(e)), "pred_empty_max_ratio": float(e["ratio"].max()),
            "pred_double": int(len(d2)), "pred_double_min_ratio": float(d2["ratio"].min()),
            "pred_double_dates": [d.date().isoformat() for d in d2["seibro"]],
            "pred_single": int(len(s1)), "single_ratio_lt_0.3": int((s1["ratio"] < 0.3).sum()),
            "single_ratio_gt_1.6": int(len(hi)), "single_hi_with_low_neighbor": int(low_nb.sum()),
            "single_hi": [{"seibro": r["seibro"].date().isoformat(), "trade": r["trades"][0].date().isoformat(),
                           "ratio": round(float(r["ratio"]), 3), "prev": round(float(r["prev_ratio"]), 3),
                           "next": round(float(r["next_ratio"]), 3)} for _, r in hi.iterrows()]}


# ---------------- 분류 ----------------
def load_classes() -> tuple[pd.DataFrame, dict]:
    c = pd.read_csv(PRIOR / "classes.csv").set_index("isin")
    c["equity"] = ~c["utype"].isin(NONEQ)
    ipo = {k: pd.Timestamp(v["first_trade"])
           for k, v in json.loads((PRIOR / "derived" / "ipo.json").read_text(encoding="utf-8")).items()}
    return c, ipo


# ---------------- 거래일 자료판 ----------------
def trade_panel(rows: pd.DataFrame, cal: pd.DataFrame, cls: pd.DataFrame):
    """거래일 × ISIN 금액. 합쳐진 SEIBRO 날짜는 거래일 수로 나눈다."""
    u = rows.sort_values("dtype").drop_duplicates(["date", "isin"])[["date", "isin", "name", "buy", "sell", "tot", "net"]]
    umap = dict(zip(cal["seibro"], cal["unit"]))
    names0 = u.sort_values("date").groupby("isin")["name"].last()
    u = u.assign(date=u["date"].map(umap)).groupby(["date", "isin"], as_index=False)[["buy", "sell", "tot", "net"]].sum()
    u["name"] = u["isin"].map(names0)
    ug = cal.groupby("unit")["trades"].agg(lambda x: sum(x, []))  # 묶음에 담긴 거래일
    isins = sorted(u["isin"].unique())
    col = {k: i for i, k in enumerate(isins)}
    tm = [(t, s, len(ts)) for s, ts in ug.items() for t in ts]
    tmap = pd.DataFrame(tm, columns=["trade", "seibro", "n_in_date"]).sort_values("trade").reset_index(drop=True)
    T = len(tmap)
    mats = {k: np.zeros((T, len(isins))) for k in ["buy", "sell", "net"]}
    present = np.zeros((T, len(isins)), dtype=bool)
    by = {d: g for d, g in u.groupby("date")}
    for i, r in tmap.iterrows():
        g = by.get(r["seibro"])
        if g is None:
            continue
        j = g["isin"].map(col).values
        present[i, j] = True
        for k in mats:
            mats[k][i, j] = g[k].values / r["n_in_date"]
    cl = cls.reindex(isins)
    unclassified = [i for i in isins if i not in cls.index]
    equity = cl["equity"].fillna(True).astype(bool).to_numpy(copy=True)  # 분류 밖 = 보통주
    equity &= ~np.isin(np.array(isins), list(DUMMY))
    product = cl["product"].fillna("").to_numpy(copy=True)
    names0 = u.sort_values("date").groupby("isin")["name"].last()
    # 상품 키(사슬): classes product, 없으면 이름 끝 'SPLR|MRGR|EXOF <번호> <ISIN>'의 ISIN 상품, 없으면 ISIN
    key = []
    for i, pr in zip(isins, product):
        if pr:
            key.append(pr)
            continue
        tail = str(names0.get(i, "")).split()
        k2 = tail[-1] if len(tail) >= 3 and tail[-3] in ("SPLR", "MRGR", "EXOF") else None
        pk = cls["product"].get(k2) if k2 is not None and k2 in cls.index else None
        key.append(pk if isinstance(pk, str) and pk else (k2 or i))
    key = np.array(key)
    names = u.sort_values("date").groupby("isin")["name"].last()
    return tmap, isins, mats, present, equity, product, unclassified, names, key


def rolling_sum(a: np.ndarray, n: int) -> np.ndarray:
    cs = np.cumsum(np.vstack([np.zeros((1,) + a.shape[1:]), a]), axis=0)
    out = np.full(a.shape, np.nan)
    out[n - 1:] = cs[n:] - cs[:-n]
    return out


def ipo_mask(tmap: pd.DataFrame, isins: list, ipo: dict) -> np.ndarray:
    """창 끝 거래일 기준 첫 거래 뒤 28일 안인 ISIN(N⁺에서만 뺀다)."""
    m = np.zeros((len(tmap), len(isins)), dtype=bool)
    tr = pd.DatetimeIndex(tmap["trade"])
    for k, d in ipo.items():
        if k in isins:
            m[(tr >= d) & (tr <= d + pd.Timedelta(days=28)), isins.index(k)] = True
    return m


def measures(tmap, isins, mats, present, equity, product, ipo, tot_by_seibro, key=None) -> pd.DataFrame:
    """거래일 t를 끝으로 하는 1·5세션 창 값(X4·H2와 재료)."""
    psoxl, psoxs = product == "SOXL", product == "SOXS"
    ex = ipo_mask(tmap, isins, ipo)
    out = {"trade": pd.DatetimeIndex(tmap["trade"]), "seibro": pd.DatetimeIndex(tmap["seibro"]),
           "n_in_date": tmap["n_in_date"].values}
    for k in ["buy", "sell", "net"]:
        out[f"SOXL_{k}"] = mats[k][:, psoxl].sum(axis=1)
        out[f"SOXS_{k}"] = mats[k][:, psoxs].sum(axis=1)
    out["SOXS_present"] = present[:, psoxs].any(axis=1)
    out["SOXL_present"] = present[:, psoxl].any(axis=1)
    out["A1"] = tot_by_seibro.reindex(tmap["seibro"])["A1"].values / tmap["n_in_date"].values
    # N⁺는 상품(사슬) 단위로 순매수를 합친 뒤 양수만 더한다. key 없으면 ISIN 단위(대조용)
    ks = sorted(set(key)) if key is not None else list(isins)
    kpos = {k: j for j, k in enumerate(ks)}
    M = np.zeros((len(isins), len(ks)))
    M[np.arange(len(isins)), [kpos[k] for k in (key if key is not None else isins)]] = 1.0
    net_p = mats["net"] @ M
    eq_p = (equity.astype(float) @ M) > 0  # 사슬 안 utype은 같다(자료판 점검)
    ex_p = ((ex.astype(float) @ M) > 0)
    pos1 = np.clip(net_p, 0, None)
    pos1[:, ~eq_p] = 0.0
    d1 = (pos1 * ~ex_p).sum(axis=1)  # 일별 N⁺(그날 끝 기준 제외)
    pos_isin = np.clip(mats["net"], 0, None)
    pos_isin[:, ~equity] = 0.0
    out["Npos_d_isin"] = (pos_isin * ~ex).sum(axis=1)  # ISIN 단위(첫 판, 대조용)
    out["Npos_d"] = d1
    for n in [1, 5]:
        posw = np.clip(rolling_sum(net_p, n), 0, None)
        posw[:, ~eq_p] = 0.0
        npos = np.nansum(posw * ~ex_p, axis=1)
        npos[:n - 1] = np.nan
        ndsum = rolling_sum(d1[:, None], n)[:, 0]
        sb = rolling_sum(out["SOXL_buy"][:, None], n)[:, 0]
        sn = rolling_sum(out["SOXL_net"][:, None], n)[:, 0]
        xb = rolling_sum(out["SOXS_buy"][:, None], n)[:, 0]
        out[f"Npos_w{n}"] = npos
        out[f"Npos_daysum_w{n}"] = ndsum
        out[f"SOXL_net_w{n}"] = sn
        out[f"SOXL_buy_w{n}"] = sb
        out[f"SOXS_buy_w{n}"] = xb
        out[f"SOXS_cens_days_w{n}"] = rolling_sum((~out["SOXS_present"]).astype(float)[:, None], n)[:, 0]
        out[f"X4_w{n}"] = sn / ndsum  # 주: 일별 N+의 창 합(자료판 정의)
        out[f"X4pos_w{n}"] = sn / npos  # 대안: 창 합 순매수의 양수 합
        out[f"H2_w{n}"] = sb / (sb + xb)
    return pd.DataFrame(out)


def pct_of_last(vals: np.ndarray, kind: str) -> float:
    x = vals[-1]
    a = vals[~np.isnan(vals)]
    n = len(a)
    left, right = int((a < x).sum()), int((a <= x).sum())
    if kind == "rank":  # 평균 순위 / n (scipy percentileofscore 'rank')
        return 100.0 * (left + right + (1 if right > left else 0)) / 2 / n
    if kind == "weak":
        return 100.0 * right / n
    if kind == "strict":
        return 100.0 * left / n
    return 100.0 * (left + right) / 2 / n  # mean


def percentile(series: pd.Series, i: int, look=250, minp=120) -> dict:
    w = series.values[max(0, i - look + 1):i + 1]
    nn = int((~np.isnan(w)).sum())
    if nn < minp or np.isnan(w[-1]):
        return {"n": nn}
    return {"n": nn, **{k: pct_of_last(w, k) for k in ["rank", "weak", "strict", "mean"]}}


# ---------------- 보관 ----------------
def custody(px: pd.DataFrame, cal: pd.DataFrame, tmap: pd.DataFrame, sx_net_seibro: pd.Series) -> dict:
    recs = []
    for p in sorted((RAW / "custody" / "amt").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["start"] > CUST_MAX:
            continue
        a = [x["amt"] for x in r["rows"] if x["isin"] == SOXL]
        recs.append({"date": pd.Timestamp(r["start"]), "top50": sum(x["amt"] for x in r["rows"]),
                     "soxl": a[0] if a else np.nan, "n": r["n"]})
    cu = pd.DataFrame(recs).set_index("date").sort_index()
    cl = closes(px, "SOXL")
    sess = cl.index
    upos = np.array([sess.searchsorted(D, side="right") - 1 for D in cu.index])  # D 이하 마지막 미국 세션
    # 평가 종가: Δlog(보관)를 가장 잘 따르는 종가(D 이하 마지막 세션에서 off만큼)
    y = np.log(cu["soxl"]).diff()
    val = {}
    for off in [-2, -1, 0, 1]:
        x = pd.Series(np.log(cl.values[upos + off]), index=cu.index).diff()
        ok = y.notna() & x.notna() & (x.abs() > 0)
        val[off] = {yr: round(float(np.corrcoef(x[ok & (cu.index.year == yr)], y[ok & (cu.index.year == yr)])[0, 1] ** 2), 4)
                    for yr in [2023, 2024, 2025, 2026]}
    best_off = max(val, key=lambda o: np.mean(list(val[o].values())))
    cu["val_date"] = sess[upos + best_off]
    cu["val_close"] = cl.values[upos + best_off]
    cu["shares"] = cu["soxl"] / cu["val_close"]
    cu["dsh_usd"] = cu["shares"].diff() * cu["val_close"]
    # 지연 1: 보관 D가 담는 마지막 SEIBRO 결제 날짜 = D 순번 + off
    sd = pd.DatetimeIndex(cal["seibro"])
    net_s = sx_net_seibro.reindex(sd).fillna(0.0)
    spos = pd.Series(np.arange(len(sd)), index=sd)
    periods = [("T+2", cu.index < "2024-05-30"), ("T+1", cu.index >= "2024-06-01")]
    lag_s, fit0 = {}, None
    for off in [-2, -1, 0, 1, 2]:
        f, prevp = [], None
        for D in cu.index:
            p = spos[D] + off
            f.append(np.nan if (prevp is None or p < 0 or p >= len(sd)) else net_s.iloc[prevp + 1:p + 1].sum())
            prevp = p
        f = pd.Series(f, index=cu.index)
        ok = f.notna() & cu["dsh_usd"].notna()
        lag_s[off] = {k: round(float(np.corrcoef(cu["dsh_usd"][ok & m], f[ok & m])[0, 1]), 4) for k, m in periods}
        if off == 0:
            b = np.polyfit(f[ok], cu["dsh_usd"][ok], 1)
            fit0 = {"slope": round(float(b[0]), 4), "n": int(ok.sum()),
                    "median_abs_resid": float((cu["dsh_usd"] - f)[ok].abs().median())}
    # 지연 2: 미국 세션 단위. 보관 D가 담는 마지막 거래일 = 세션[u(D) - L]
    tr_net = pd.Series([sx_net_seibro.get(s, 0.0) / n for s, n in zip(tmap["seibro"], tmap["n_in_date"])],
                       index=pd.DatetimeIndex(tmap["trade"]))
    lag_t = {}
    for L in range(0, 5):
        last = sess[upos - L]
        f, prev = [], None
        for lt in last:
            f.append(np.nan if prev is None else tr_net[(tr_net.index > prev) & (tr_net.index <= lt)].sum())
            prev = lt
        f = pd.Series(f, index=cu.index)
        ok = f.notna() & cu["dsh_usd"].notna()
        lag_t[L] = {k: round(float(np.corrcoef(cu["dsh_usd"][ok & m], f[ok & m])[0, 1]), 4) for k, m in periods}
    return {"cu": cu, "valuation_r2": val, "best_val_offset": int(best_off), "lag_seibro_corr": lag_s,
            "lag_seibro_fit0": fit0, "lag_session_corr": lag_t}


# ---------------- 사건 창 ----------------
def event_rows(ev_date: str, me: pd.DataFrame, soxx: pd.Series, kmin=-5, kmax=5) -> list[dict]:
    trades = list(me["trade"])
    sess = soxx.index
    i0 = sess.get_loc(pd.Timestamp(ev_date))
    out = []
    for k in range(kmin, kmax + 1):
        t = sess[i0 + k]
        if t not in trades:
            out.append({"k": k, "trade": t.date().isoformat(), "seibro": None})
            continue
        r = me.iloc[trades.index(t)]
        out.append({"k": k, "trade": t.date().isoformat(), "seibro": r["seibro"].date().isoformat(),
                    "merged": bool(r["n_in_date"] > 1), "soxx_close": float(soxx[t]),
                    "SOXL_buy": float(r["SOXL_buy"]), "SOXL_sell": float(r["SOXL_sell"]),
                    "SOXL_net": float(r["SOXL_net"]),
                    "SOXS_net": float(r["SOXS_net"]) if r["SOXS_present"] else None,
                    "SOXS_censored": bool(not r["SOXS_present"]), "A1": float(r["A1"]),
                    "Npos_d": float(r["Npos_d"])})
    return out


def window_vals(ev_date: str, me: pd.DataFrame) -> dict:
    i0 = list(me["trade"]).index(pd.Timestamp(ev_date))
    res = {}
    for wname, end, n in [("W2", i0 - 1, 5), ("W3", i0, 1)]:
        r = me.iloc[end]
        rec = {"end": r["trade"].date().isoformat(), "n_sessions": n, "SOXL_net": float(r[f"SOXL_net_w{n}"]),
               "Npos": float(r[f"Npos_w{n}"]), "Npos_daysum": float(r[f"Npos_daysum_w{n}"]),
               "SOXL_buy": float(r[f"SOXL_buy_w{n}"]), "SOXS_buy": float(r[f"SOXS_buy_w{n}"]),
               "SOXS_cens_days": int(r[f"SOXS_cens_days_w{n}"])}
        for m in ["X4", "X4pos", "H2"]:
            rec[m] = float(r[f"{m}_w{n}"])
            rec[f"P_{m}"] = percentile(me[f"{m}_w{n}"], end)
        res[wname] = rec
    return res


def custody_event(ev_date: str, cu: pd.DataFrame, tmap: pd.DataFrame, soxx: pd.Series, kmin=-5, kmax=5) -> list[dict]:
    sess = soxx.index
    i0 = sess.get_loc(pd.Timestamp(ev_date))
    tm = tmap.set_index("trade")
    out = []
    for k in range(kmin, kmax + 1):
        t = sess[i0 + k]
        s, n = tm.loc[t, "seibro"], int(tm.loc[t, "n_in_date"])
        D = s if s in cu.index else None  # 거래일 t의 결제가 처음 담기는 보관 날짜 = t의 SEIBRO 날짜
        # 세션 지연 L(실측: T+2 3, T+1 2): 보관 D는 세션[u(D) − L]까지 담는다 → t를 처음 담는 D
        L = 3 if t < pd.Timestamp("2024-05-28") else 2
        upos = np.array([sess.searchsorted(x, side="right") - 1 for x in cu.index])
        okD = [x for x, up in zip(cu.index, upos) if up - L >= i0 + k]
        DL = okD[0] if okD else None
        rec = {"k": k, "trade": t.date().isoformat(), "custody_date": D.date().isoformat() if D is not None else None,
               "custody_date_lag": DL.date().isoformat() if DL is not None else None,
               "shared_with_other_trade": n > 1}
        if D is not None:
            r = cu.loc[D]
            rec.update({"amt": float(r["soxl"]), "val_date": r["val_date"].date().isoformat(),
                        "val_close": float(r["val_close"]), "shares": float(r["shares"])})
        out.append(rec)
    return out


def compute() -> dict:
    px = prices()
    soxx = closes(px, "SOXX")
    ev = events(soxx)
    rows, tot = load_daily()
    cal = seibro_calendar(tot)
    cls, ipo = load_classes()
    tmap, isins, mats, present, equity, product, unclassified, names, key = trade_panel(rows, cal, cls)
    tot_u = tot.groupby(cal.set_index("seibro")["unit"].reindex(tot.index).values).sum()  # 묶음 합
    me = measures(tmap, isins, mats, present, equity, product, ipo, tot_u, key)
    e = cal[cal["n_pred"] == 0]
    dropped = {"dates": int(len(e)), "A1_sum": float(e["A1"].sum()),
               "A1_max": {"date": e.loc[e["A1"].idxmax(), "seibro"].date().isoformat(), "A1": float(e["A1"].max())}}
    u = rows.sort_values("dtype").drop_duplicates(["date", "isin"])
    sx_net_seibro = u[u["isin"] == SOXL].set_index("date")["net"]
    cus = custody(px, cal, tmap, sx_net_seibro)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "events": ev,
           "calendar_check": calendar_check(cal), "dropped_pred_empty": dropped,
           "unclassified_isins": {i: names.get(i, "") for i in unclassified},
           "trade_dates": int(len(tmap)), "trade_first": tmap["trade"].min().date().isoformat(),
           "trade_last": tmap["trade"].max().date().isoformat(), "b15": {}}
    for x in ev["b15"]:
        res["b15"][x["date"]] = {"days": event_rows(x["date"], me, soxx), "win": window_vals(x["date"], me)}
    res["custody"] = {k: cus[k] for k in ["valuation_r2", "best_val_offset", "lag_seibro_corr", "lag_seibro_fit0",
                                          "lag_session_corr"]}
    res["custody"]["events"] = {d: custody_event(d, cus["cu"], tmap, soxx) for d in ["2025-04-08", "2026-07-29"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mine.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    me.to_parquet(OUT / "_me.parquet", index=False)
    cus["cu"].reset_index().to_parquet(OUT / "_cu.parquet", index=False)
    tmap.to_parquet(OUT / "_tmap.parquet", index=False)
    return res


# ---------------- derived/ 대조 ----------------
DER = HERE / "derived"
PKIND = "mean"  # 백분위 = (작은 수 + 같은 수 ÷ 2) ÷ n × 100. 자료판과 같은 정의(평균 순위식 rank와 0.5/n 차)


def rel(a, b) -> float:
    """상대 차 |a − b| ÷ |b|(b = 자료판). 둘 다 0이면 0."""
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return np.nan if not (a is None and b is None) else 0.0
    return 0.0 if a == b else abs(a - b) / max(abs(b), 1e-12)


def compare() -> pd.DataFrame:
    mine = json.loads((OUT / "mine.json").read_text(encoding="utf-8"))
    me = pd.read_parquet(OUT / "_me.parquet")
    tmap = pd.read_parquet(OUT / "_tmap.parquet")
    cu = pd.read_parquet(OUT / "_cu.parquet").set_index("date")
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date")
    mw = pd.read_parquet(DER / "measures_win.parquet")
    un = pd.read_parquet(DER / "units.parquet")
    cd = pd.read_parquet(DER / "custody_daily.parquet")
    rows = []

    def add(sec, item, mv, dv, kind="num"):
        if kind == "num":
            r = rel(mv, dv)
            ok = (r <= 0.01) if not np.isnan(r) else False
        else:
            r, ok = (0.0 if mv == dv else 1.0), mv == dv
        rows.append({"section": sec, "item": item, "mine": mv, "derived": dv, "rel_diff": r, "ok": bool(ok)})

    # (1) 사건
    e = mine["events"]
    add("events", "B15", [x["date"] for x in e["b15"]], ev["B15"], "eq")
    add("events", "B10", e["b10"], ev["B10"], "eq")
    dF = {f["date"]: f["episode"] for f in ev["F"]}
    mF = {f["date"]: f["phase_bottom"] for f in e["F"]}
    add("events", "F 날짜", sorted(mF), sorted(dF), "eq")
    add("events", "F 국면", [mF[k] for k in sorted(mF)], [dF.get(k) for k in sorted(mF)], "eq")
    add("events", "F 미정", e["F_undetermined"], [x["date"] for x in ev["F_undetermined"]], "eq")
    dB = {x["bottom"]: x for x in ev["B10_events"]}
    for x in e["b15"]:
        b = dB.get(x["date"], {})
        add("events", f"{x['date']} 15% 고점", x["top"], b.get("top15"), "eq")
        add("events", f"{x['date']} 낙폭", x["drop"], b.get("dd15"))
    # 날짜표 전체(1002날짜): 예측 거래일·묶음
    un["sd"] = pd.to_datetime(un["seibro_date"])
    mt = tmap.groupby("seibro")["trade"].apply(lambda s: ";".join(sorted(t.date().isoformat() for t in s)))
    dt_ = un.set_index("sd")["unit_trades"].astype(str)
    common = mt.index.intersection(dt_.index)
    bad = [d.date().isoformat() for d in common if mt[d] != dt_[d]]
    add("calendar", f"묶음 거래일 일치(날짜 {len(common)}개)", len(bad), 0)
    add("calendar", "합친 묶음 수", int((tmap["n_in_date"] > 1).sum()) // 2, int(un.loc[un["merged"], "unit_id"].nunique()))
    # (2) 일별 값: 전 964거래일과 B15 k=-5..+5
    me = me.set_index("trade")
    cols = {"SOXL_buy": "SOXL_buy", "SOXL_sell": "SOXL_sell", "SOXL_net": "SOXL_net", "SOXS_net": "SOXS_net", "A1": "A1"}
    idx = me.index.intersection(md.index)
    add("daily_all", "거래일 수", len(me), len(md))
    for mc, dc in cols.items():
        d = (me.loc[idx, mc] - md.loc[idx, dc]).abs()
        rd = (d / md.loc[idx, dc].abs().clip(lower=1.0)).fillna(0.0)
        rows.append({"section": "daily_all", "item": f"{mc} 최대 상대차({len(idx)}일, 다른 날 {int((d > 0).sum())}, 최대 절대차 {d.max():,.0f} USD)",
                     "mine": float(rd.max()), "derived": 0.0, "rel_diff": float(rd.max()), "ok": bool(rd.max() <= 0.01)})
    for b, v in mine["b15"].items():
        for x in v["days"]:
            t = pd.Timestamp(x["trade"])
            r = md.loc[t]
            add("map", f"{b} k{x['k']:+d} {x['trade']} SEIBRO", x["seibro"], pd.Timestamp(r["seibro_date"]).date().isoformat(), "eq")
            add("map", f"{b} k{x['k']:+d} 묶음", x["merged"], bool(r["merged"]), "eq")
            for mc in ["SOXL_buy", "SOXL_sell", "SOXL_net", "A1"]:
                add("daily", f"{b} k{x['k']:+d} {mc}", x[mc], float(r[mc]))
            add("daily", f"{b} k{x['k']:+d} SOXS_net", x["SOXS_net"], float(r["SOXS_net"]) if not r["cens_SOXS"] else None)
    # (3) 창 X4·H2와 P
    W = mw.set_index(["date", "L", "measure"])
    for b, v in mine["b15"].items():
        for wn, y in v["win"].items():
            L = y["n_sessions"]
            for m in ["X4", "H2"]:
                rr = W.loc[(pd.Timestamp(y["end"]), L, m)]
                add("window", f"{b} {wn} {m}", y[m], float(rr["value"]))
                add("window", f"{b} {wn} P_{m}", y[f"P_{m}"].get(PKIND), float(rr["P"]))
                add("window", f"{b} {wn} nP_{m}", y[f"P_{m}"]["n"], int(rr["nP"]), "eq")
    # 창 값 전체 시계열(L 1·5)
    for L in [1, 5]:
        for m in ["X4", "H2"]:
            d = mw[(mw["L"] == L) & (mw["measure"] == m)].set_index("date")["value"]
            x = me[f"{m}_w{L}"].reindex(d.index)
            ok = d.notna() & x.notna()
            rd = ((x - d).abs() / d.abs().clip(lower=1e-12))[ok]
            add("window_all", f"L{L} {m} 최대 상대차(n {int(ok.sum())})", float(rd.max()), 0.0, "eq" if rd.max() <= 1e-9 else "num")
    # P 전체 시계열(L 1·5, X4·H2): 지난 250세션 평균 순위 백분위
    for L in [1, 5]:
        for m in ["X4", "H2"]:
            d = mw[(mw["L"] == L) & (mw["measure"] == m)].set_index("date")
            ser = me[f"{m}_w{L}"]
            mp = pd.Series([percentile(ser, i).get(PKIND, np.nan) for i in range(len(ser))], index=ser.index)
            x = mp.reindex(d.index)
            ok = d["P"].notna() & x.notna()
            nb = int((d["P"].notna() != x.notna()).sum())
            add("window_all", f"L{L} {m} P 최대 절대차(점, n {int(ok.sum())}, 결측 어긋남 {nb})", float((x - d["P"]).abs()[ok].max()), 0.0,
                "eq" if (x - d["P"]).abs()[ok].max() <= 1e-9 and nb == 0 else "num")
    # (4) 보관 SOXL
    cs = cd[cd["product"] == "SOXL"].copy()
    cs["trade_date"] = pd.to_datetime(cs["trade_date"])
    cs["custody_date"] = pd.to_datetime(cs["custody_date"])
    byD = cs.set_index("custody_date")
    for b, lst in mine["custody"]["events"].items():
        for x in lst:
            t = pd.Timestamp(x["trade"])
            hit = cs[cs["trade_date"] == t]
            Dd = hit["custody_date"].iloc[0].date().isoformat() if len(hit) else None
            add("custody", f"{b} k{x['k']:+d} {x['trade']} 보관 날짜", x["custody_date_lag"], Dd, "eq")
            D = pd.Timestamp(x["custody_date_lag"])
            add("custody", f"{b} k{x['k']:+d} 금액", float(cu.loc[D, "soxl"]), float(byD.loc[D, "amt"]))
            add("custody", f"{b} k{x['k']:+d} 좌수", float(cu.loc[D, "shares"]), float(byD.loc[D, "SOXL_shares"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "compare.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "compute"
    if what == "compute":
        r = compute()
        print(json.dumps({k: r[k] for k in ["calendar_check", "dropped_pred_empty", "trade_dates", "trade_first",
                                             "trade_last"]}, ensure_ascii=False, default=str)[:600])
    if what in ("compare", "all"):
        if what == "all":
            compute()
        df = compare()
        print(df.groupby("section")["ok"].agg(["size", "sum"]).to_string())
        print(df[~df["ok"]].to_string(max_colwidth=60))
