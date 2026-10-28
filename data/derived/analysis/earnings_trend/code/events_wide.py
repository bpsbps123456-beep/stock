"""실적 후 흐름 조건 규칙 연구 — 넓은 표본 이벤트 표(진입 시점에 알 수 있는 특징 + 이후 수익).

실행: py data/derived/analysis/earnings_trend/code/events_wide.py  → results/events_wide.csv
      py data/derived/analysis/earnings_trend/code/events_wide.py --placebo → cache/wide/placebo_wide.parquet
      (가짜 사건 = 실적과 15거래일 넘게 떨어진 보통 날. 규칙이 실적 때문인지 2026 업종·스타일 흐름 때문인지 가른다)

진입 두 가지:
  E0 = 반응일 종가(장후 발표면 다음 날, 장전 발표면 그날). 특징은 반응일 종가까지.
  E5 = 반응일 5거래일 뒤 종가. 특징에 반응 뒤 5일의 움직임(d5_x 등)을 더한다.
결과(SPY 대비 초과수익, %): h20·h40(진입 뒤 20·40거래일), nx(다음 실적 반응일 전날까지). 업종 ETF 대비는 *_s.
발표 시각: SEC 8-K 2.02 접수(±3일) 1순위, 없으면 yfinance 시각(16시 이후=장후, 9시30분 전=장전, 그 밖은 버림).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sec import RESULTS  # noqa: E402
from fetch_wide import W, load_ohlcv  # noqa: E402

SECTOR_ETF = {"Technology": "XLK", "Finance": "XLF", "Health Care": "XLV", "Energy": "XLE", "Industrials": "XLI",
              "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Utilities": "XLU",
              "Basic Materials": "XLB", "Real Estate": "XLRE", "Telecommunications": "XLC"}
CHIP_TOPS = ("칩 설계", "파운드리·후공정", "메모리·스토리지", "장비·소재·테스트")
MIN_CAP_EVENT = 2e9


def etf_for(row: pd.Series, chip_subs: set[str]) -> str:
    if row["ai_sub"] in chip_subs:
        return "SOXX"
    return SECTOR_ETF.get(row["sector"], "SPY")


def reaction_index(ts: pd.Timestamp, days: pd.DatetimeIndex, after_close: bool) -> int:
    d = ts.normalize()
    return int(days.searchsorted(d, side="right" if after_close else "left"))


def main() -> None:
    placebo = "--placebo" in sys.argv
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    import watchlist  # noqa: E402
    chip_subs = {sub for top, subs in watchlist.AI_UNIVERSE.items() if top in CHIP_TOPS for sub in subs}
    uni = pd.read_csv(RESULTS / "wide_universe.csv").fillna({"sector": "", "ai_sub": ""}).set_index("ticker")
    eps = pd.read_csv(RESULTS / "wide_eps.csv")
    sec = pd.read_csv(W / "sec_202.csv")
    sec["acc"] = pd.to_datetime(sec["acceptance_et"])
    ohlcv = load_ohlcv()
    C, O, H, L, V = (ohlcv[k] for k in ["Close", "Open", "High", "Low", "Volume"])
    days = C.index
    last_k = len(days) - 1
    spy = C["SPY"].values
    lr = np.log(C).diff()
    # 동일가중 '평균 종목' 지수(ETF 제외, 하루 ±50% 밖은 자료 오류로 보고 뺀다) — 넓은 표본의 기준 승률을 50% 가까이로
    stocks = [x for x in C.columns if x not in ("SPY", "QQQ", "SOXX", "IGV", "XLK", "XLF", "XLV", "XLE", "XLI",
                                                   "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC")]
    dr = C[stocks].pct_change()
    ew = (1 + dr.where(dr.abs() < 0.5).mean(axis=1).fillna(0)).cumprod().values
    # 분기 매출(frames) → (cik, frame) 표
    ciks = json.loads((W / "ciks.json").read_text(encoding="utf-8"))
    fr = pd.read_csv(W / "frames_revenue.csv")
    fr["end"] = pd.to_datetime(fr["end"])
    fr_by = {c: g.set_index("frame")[["end", "val"]] for c, g in fr.groupby("cik")}

    def prev_frame(f: str, n: int) -> str:
        y, q = int(f[:4]), int(f[-1])
        idx = y * 4 + (q - 1) - n
        return f"{idx // 4}Q{idx % 4 + 1}"

    eps["d"] = pd.to_datetime(eps["date"])
    eps = eps[(eps["d"] >= "2024-09-01") & (eps["d"] <= days[-1]) & eps["eps_reported"].notna()]
    rows = []
    for t, g in eps.sort_values("d").groupby("ticker"):
        if t not in C or t not in uni.index:
            continue
        info = uni.loc[t]
        c, o, h, lo, v = C[t].values, O[t].values, H[t].values, L[t].values, V[t].values
        r = lr[t].values
        st = sec[sec["ticker"] == t]
        evs = []
        for _, e in g.iterrows():
            s = st[(st["acc"].dt.normalize() - e["d"]).abs().dt.days <= 3]
            if len(s):
                acc = s.iloc[(s["acc"].dt.normalize() - e["d"]).abs().argsort().iloc[0]]["acc"]
                after = acc.hour * 60 + acc.minute >= 16 * 60
                when, src = acc, "sec"
            else:
                hh, mm = map(int, e["time_et"].split(":"))
                if hh * 60 + mm >= 16 * 60:
                    after = True
                elif hh * 60 + mm < 9 * 60 + 30 and hh >= 4:
                    after = False
                else:
                    continue  # 시각 미상
                when, src = e["d"], "yf"
            k = reaction_index(when, days, after)
            if 61 <= k <= last_k:
                evs.append((k, e, src))
        # 같은 반응일 중복 제거, 45일(약 31거래일) 안 재발표는 뺀다
        clean, lastk = [], -999
        for k, e, src in sorted(evs, key=lambda x: x[0]):
            if k - lastk >= 31:
                clean.append((k, e, src))
                lastk = k
        cap_now, px_now = info.get("cap_now"), c[last_k] if not np.isnan(c[last_k]) else np.nan
        cik = ciks.get(t.replace("-", ".")) or ciks.get(t)
        anchors = [(k, e, src, (clean[i + 1][0] if i + 1 < len(clean) else None)) for i, (k, e, src) in enumerate(clean)]
        if placebo:
            # 가짜 사건: 실적 반응일에서 15거래일 넘게 떨어진 날을 10거래일 간격으로. 실적 숫자는 직전 발표 값(묵은 값)
            ks = [a[0] for a in anchors]
            fake = []
            for kk in range(61, last_k - 5, 10):
                # 직전 실적에서 15거래일 넘게 지났고, 뒤 25거래일 안에 다음 실적이 없어야 한다
                # (진짜 실적 사건처럼 이후 20일 창에 실적 급변이 섞이지 않게)
                if not ks or min(abs(kk - x) for x in ks) <= 15 or any(kk < x <= kk + 25 for x in ks):
                    continue
                prev = [a for a in anchors if a[0] < kk]
                if not prev:
                    continue
                nxt = [x for x in ks if x > kk]
                fake.append((kk, prev[-1][1], "placebo", nxt[0] if nxt else None))
            anchors = fake
        for k, e, src, nk in anchors:
            end = (nk - 1) if nk is not None else last_k
            p0 = c[k - 1]
            if not (p0 > 0) or np.isnan(c[k]):
                continue
            mcap = cap_now * p0 / px_now if cap_now and px_now and px_now > 0 else np.nan
            if pd.notna(mcap) and mcap < MIN_CAP_EVENT:
                continue
            vol60 = np.nanstd(r[k - 60:k]) * 100
            avgv = np.nanmean(v[k - 51:k - 1])
            hi252 = np.nanmax(h[max(0, k - 253):k - 1]) if k > 1 else np.nan
            ma50, ma200 = np.nanmean(c[k - 51:k - 1]), np.nanmean(c[max(0, k - 201):k - 1])

            def ret(a: int, b: int, series: np.ndarray = c) -> float:
                if a < 0 or b > last_k or b < a:
                    return np.nan
                x, y = series[a], series[b]
                return (y / x - 1) * 100 if x > 0 and not np.isnan(y) else np.nan

            row = {"ticker": t, "date": e["date"] if not placebo else days[k].strftime("%Y-%m-%d"), "time_src": src,
                   "reaction_date": days[k].strftime("%Y-%m-%d"),
                   "k": k, "end_k": end, "partial": nk is None, "sector": info.get("sector", ""),
                   "ai_sub": info.get("ai_sub", ""), "mcap_b": mcap / 1e9 if pd.notna(mcap) else np.nan,
                   "eps_surprise": e["eps_surprise_pct"]}
            row["etf"] = etf_for(pd.Series({"ai_sub": row["ai_sub"], "sector": row["sector"]}), chip_subs)
            sb = C[row["etf"]].values
            # 반응일
            row["r0"] = ret(k - 1, k)
            row["r0_x"] = row["r0"] - ret(k - 1, k, spy)
            row["r0_sig"] = row["r0"] / vol60 if vol60 > 0 else np.nan
            row["gap"] = (o[k] / p0 - 1) * 100 if o[k] > 0 else np.nan
            row["o2c"] = (c[k] / o[k] - 1) * 100 if o[k] > 0 else np.nan
            row["clv"] = (c[k] - lo[k]) / (h[k] - lo[k]) if h[k] > lo[k] else np.nan
            row["vol_ratio"] = v[k] / avgv if avgv and avgv > 0 else np.nan
            # 발표 전(전날 종가까지)
            row["pre20_x"] = ret(k - 21, k - 1) - ret(k - 21, k - 1, spy)
            row["pre60_x"] = ret(k - 61, k - 1) - ret(k - 61, k - 1, spy)
            row["above_ma"] = int(p0 > ma50 and p0 > ma200)
            row["below_ma"] = int(p0 < ma50 and p0 < ma200)
            row["dist_hi"] = (p0 / hi252 - 1) * 100 if hi252 and hi252 > 0 else np.nan
            row["vol60"] = vol60
            # 분기 매출 YoY·가속(frames): 발표일 전 100일 안에 끝난 가장 늦은 분기
            row["rev_yoy"] = row["rev_accel"] = np.nan
            if cik in fr_by:
                f_ = fr_by[cik]
                ref_d = e["d"] if not placebo else days[k]
                cand = f_[(f_["end"] < ref_d) & ((ref_d - f_["end"]).dt.days <= 190 if placebo else (ref_d - f_["end"]).dt.days <= 100)]
                if len(cand):
                    fl = cand["end"].idxmax()

                    def yoy(lbl: str) -> float:
                        p = prev_frame(lbl, 4)
                        return (f_.at[lbl, "val"] / f_.at[p, "val"] - 1) * 100 if lbl in f_.index and p in f_.index and f_.at[p, "val"] > 0 else np.nan
                    row["rev_yoy"] = yoy(fl)
                    row["rev_accel"] = row["rev_yoy"] - yoy(prev_frame(fl, 1))
            # 결과: E0(반응일 종가 진입)
            for hz in (5, 20, 40):
                b = k + hz
                row[f"h{hz}_x"] = ret(k, b) - ret(k, b, spy) if b <= last_k else np.nan  # 고정 기간(다음 실적을 넘어도 그대로)
                row[f"h{hz}_s"] = ret(k, b) - ret(k, b, sb) if pd.notna(row[f"h{hz}_x"]) else np.nan
            for hz in (20,):
                b = k + hz
                row[f"h{hz}_e"] = ret(k, b) - ret(k, b, ew) if b <= last_k else np.nan
            row["nx_e"] = ret(k, end) - ret(k, end, ew) if end > k else np.nan
            row["nx_x"] = ret(k, end) - ret(k, end, spy) if end > k else np.nan
            row["nx_s"] = ret(k, end) - ret(k, end, sb) if end > k else np.nan
            row["nx_days"] = end - k
            # E5(반응 5일 뒤 진입): 반응 뒤 5일 움직임과 이후 결과
            k5 = k + 5
            if k5 < end:
                row["d5_x"] = ret(k, k5) - ret(k, k5, spy)
                row["held_reaction"] = int(c[k5] >= c[k])
                row["back_to_pre"] = int((c[k5] - p0) * np.sign(row["r0"]) <= 0)  # 반응을 모두 되돌림
                row["e5_h20_x"] = ret(k5, k5 + 20) - ret(k5, k5 + 20, spy) if k5 + 20 <= last_k else np.nan
                row["e5_nx_x"] = ret(k5, end) - ret(k5, end, spy)
                row["e5_h20_e"] = ret(k5, k5 + 20) - ret(k5, k5 + 20, ew) if k5 + 20 <= last_k else np.nan
                row["e5_nx_e"] = ret(k5, end) - ret(k5, end, ew)
                row["e5_h20_s"] = ret(k5, k5 + 20) - ret(k5, k5 + 20, sb) if k5 + 20 <= last_k else np.nan
                row["e5_nx_s"] = ret(k5, end) - ret(k5, end, sb)
            rows.append(row)
    out = pd.DataFrame(rows)
    rd = pd.to_datetime(out["reaction_date"])
    out["season"] = rd.dt.year.astype(str) + "Q" + rd.dt.quarter.astype(str)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    if placebo:
        out.to_parquet(W / "placebo_wide.parquet")  # 크다(수만 행) — Git 제외 캐시
        print(f"가짜 사건 {len(out)}건")
        return
    out.to_csv(RESULTS / "events_wide.csv", index=False)
    print(f"이벤트 {len(out)}건, 종목 {out['ticker'].nunique()}개, 시각 출처 {out['time_src'].value_counts().to_dict()}")
    print(out.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
