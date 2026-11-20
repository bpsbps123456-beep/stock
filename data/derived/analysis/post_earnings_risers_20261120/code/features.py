"""사건마다 실적 숫자 밖의 요인을 붙인다. 모든 '사전' 요인은 반응일 전날 종가(또는 그 전 공개분)까지의 값.

묶음
- px_*  가격·거래량(발표 전날까지)과 반응일 모양(반응일 종가에 앎)
- fm_*  두 펀드매니저 수량 변화(펀드 전체 설정·환매를 뺀 '능동' 변화)
- occ_* OCC 고객(개인+기관 고객) 옵션 거래량, 발표 전 5거래일 이상 증가
- kr_*  SEIBRO 한국인 미국 주식 결제 Top50(개미 관심의 직접 대리값). 결제일 D는 미국 거래 약 2세션 뒤, D 다음 평일 공개
- post_* 반응일 뒤에야 아는 값(설명용, 매매 근거 아님)
결과: results/features.csv
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
CACHE = BASE / "cache"
OCC = ROOT / "data/derived/analysis/options_sentiment/cache/occ"
SEIBRO = ROOT / "data/derived/analysis/seibro_flows"


def seibro_daily(tickers: set[str]) -> pd.DataFrame:
    """SEIBRO 날짜 × 종목: 네 Top50 목록 중 하나라도 든 날의 매수·매도·순매수(달러)."""
    master = pd.read_csv(SEIBRO / "derived/isin_master.csv", encoding="utf-8-sig")
    isin2tk = dict(zip(master.isin, master.figi_ticker))
    recs = {}
    for f in glob.glob(str(SEIBRO / "raw/daily/d*/2025*.json")) + glob.glob(str(SEIBRO / "raw/daily/d*/2026*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for r in d["rows"]:
            tk = isin2tk.get(r["isin"])
            if tk is None or (isinstance(tk, float)):
                continue
            tk = str(tk).replace("/", "-").replace(" ", "-")
            if tk not in tickers:
                continue
            recs[(d["start"], tk)] = (r["buy"], r["sell"], r["net"])
    df = pd.DataFrame([(k[0], k[1], *v) for k, v in recs.items()], columns=["sdate", "ticker", "buy", "sell", "net"])
    df["sdate"] = pd.to_datetime(df.sdate)
    return df


def seibro_custody(tickers: set[str]) -> pd.DataFrame:
    master = pd.read_csv(SEIBRO / "derived/isin_master.csv", encoding="utf-8-sig")
    isin2tk = dict(zip(master.isin, master.figi_ticker))
    rows = []
    for f in glob.glob(str(SEIBRO / "raw/custody/amt/2025*.json")) + glob.glob(str(SEIBRO / "raw/custody/amt/2026*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for r in d["rows"]:
            tk = isin2tk.get(r["isin"])
            if isinstance(tk, str) and tk.replace("/", "-") in tickers:
                rows.append((d["start"], tk.replace("/", "-"), r["rank"], r["amt"]))
    df = pd.DataFrame(rows, columns=["cdate", "ticker", "rank", "amt"])
    df["cdate"] = pd.to_datetime(df.cdate)
    return df


def occ_panel(tickers: list[str]) -> pd.DataFrame:
    parts = []
    for f in sorted(OCC.glob("2025-*.parquet")) + sorted(OCC.glob("2026-*.parquet")):
        d = pd.read_parquet(f)
        d = d[d.underlying.isin(tickers)]
        d.insert(0, "date", pd.Timestamp(f.stem))
        parts.append(d)
    o = pd.concat(parts, ignore_index=True)
    o["tot"] = o[["C_C", "C_P", "F_C", "F_P", "M_C", "M_P"]].sum(axis=1)
    return o


def fund_active_change(held: pd.DataFrame, dates: list[str]):
    """펀드별 날짜 × 종목 수량표와, 두 날짜 사이 '능동' 로그 변화(그 펀드 보유 종목 중앙값을 뺀 값)."""
    tabs = {f: g.pivot_table(index="date", columns="ticker", values="qty", aggfunc="last") for f, g in held.groupby("fund")}
    return tabs


def active(tab: pd.DataFrame, d0: str, d1: str, tk: str):
    """d0 → d1 수량 변화. (능동 로그 변화, 신규 편입, 전량 매도)"""
    s0 = tab.loc[tab.index[tab.index <= d0][-1]] if (tab.index <= d0).any() else None
    s1 = tab.loc[tab.index[tab.index <= d1][-1]] if (tab.index <= d1).any() else None
    if s0 is None or s1 is None:
        return np.nan, False, False
    q0, q1 = s0.get(tk, np.nan), s1.get(tk, np.nan)
    both = s0.notna() & s1.notna() & (s0 > 0) & (s1 > 0)
    med = np.log(s1[both] / s0[both]).median() if both.sum() >= 5 else 0.0
    if pd.notna(q0) and pd.notna(q1) and q0 > 0 and q1 > 0:
        return float(np.log(q1 / q0) - med), False, False
    if (pd.isna(q0) or q0 == 0) and pd.notna(q1) and q1 > 0:
        return np.nan, True, False
    if pd.notna(q0) and q0 > 0 and (pd.isna(q1) or q1 == 0):
        return np.nan, False, True
    return np.nan, False, False


def sector_map(tickers):
    """watchlist AI 하위 업종, 없으면 yfinance 업종(캐시)."""
    import sys
    sys.path.insert(0, str(ROOT))
    import watchlist as w
    grp = {}
    for g, subs in w.AI_UNIVERSE.items():
        for s, syms in subs.items():
            for x in syms:
                grp.setdefault(x, g)
    cache = CACHE / "yf_sector.csv"
    if cache.exists():
        yfs = pd.read_csv(cache).set_index("ticker")["sector"].to_dict()
    else:
        import yfinance as yf
        yfs = {}
        for t in tickers:
            try:
                yfs[t] = yf.Ticker(t).info.get("sector")
            except Exception:
                yfs[t] = None
        pd.Series(yfs, name="sector").rename_axis("ticker").reset_index().to_csv(cache, index=False)
    return {t: (grp.get(t) or grp.get({"GOOG": "GOOGL"}.get(t, t)) or f"비AI:{yfs.get(t) or '?'}") for t in tickers}


def main() -> None:
    ev = pd.read_csv(BASE / "results/events.csv")
    px = {k: pd.read_parquet(CACHE / f"px_{k}.parquet") for k in ("open", "high", "low", "close", "adjclose", "volume")}
    idx = px["close"]["QQQ"].dropna().index
    px = {k: v.reindex(idx) for k, v in px.items()}
    ac, vol, hi, lo, cl, op = px["adjclose"], px["volume"], px["high"], px["low"], px["close"], px["open"]
    info = pd.read_csv(CACHE / "info.csv").set_index("ticker")
    held = pd.read_csv(BASE / "results/held_days.csv")
    tabs = fund_active_change(held, sorted(held.date.unique()))
    tickers = sorted(ev.ticker.unique())
    sec_of = sector_map(tickers)
    kr = seibro_daily(set(tickers))
    krc = seibro_custody(set(tickers))
    kr_by = {t: g.set_index("sdate").sort_index() for t, g in kr.groupby("ticker")}
    krc_by = {t: g.set_index("cdate").sort_index() for t, g in krc.groupby("ticker")}
    o = occ_panel(tickers)
    occ_by = {t: g.set_index("date").sort_index() for t, g in o.groupby("underlying")}
    occ_dates = pd.DatetimeIndex(sorted(o.date.unique()))
    n = len(idx)

    rows = []
    for e in ev.itertuples():
        k, tk = int(e.k), e.ticker
        f = {"ticker": tk, "reaction_date": e.reaction_date}
        c, q = ac[tk], ac["QQQ"]
        r = c.pct_change()
        f["group"] = sec_of.get(tk)
        # ── 가격·거래량(발표 전날 종가까지) ──
        for h in (20, 60):
            f[f"px_pre{h}_x"] = 100 * ((c.iloc[k - 1] / c.iloc[k - 1 - h] - 1) - (q.iloc[k - 1] / q.iloc[k - 1 - h] - 1))
        f["px_dist_hi"] = 100 * (cl[tk].iloc[k - 1] / hi[tk].iloc[max(0, k - 252):k].max() - 1)
        f["px_vol60"] = 100 * r.iloc[k - 60:k].std()
        v = vol[tk]
        base = v.iloc[k - 65:k - 5].median()
        f["px_vsurge5"] = np.log(v.iloc[k - 5:k].mean() / base) if base and base > 0 else np.nan
        dv = (cl[tk] * v).iloc[k - 20:k].mean()
        f["px_dvol_m"] = dv / 1e6
        last_close = cl[tk].dropna().iloc[-1]
        mc_now = info["mcap_now"].get(tk, np.nan)
        f["px_mcap_b"] = mc_now * cl[tk].iloc[k - 1] / last_close / 1e9 if pd.notna(mc_now) else np.nan
        f["px_turn"] = dv / (f["px_mcap_b"] * 1e9) * 100 if f["px_mcap_b"] else np.nan  # 하루 회전율 %
        # ── 반응일 모양(반응일 종가에 앎) ──
        rng = hi[tk].iloc[k] - lo[tk].iloc[k]
        f["px_clv0"] = (cl[tk].iloc[k] - lo[tk].iloc[k]) / rng if rng > 0 else np.nan
        f["px_o2c0"] = 100 * (cl[tk].iloc[k] / op[tk].iloc[k] - 1)
        f["px_vol0"] = v.iloc[k] / v.iloc[k - 50:k].median() if v.iloc[k - 50:k].median() > 0 else np.nan
        f["px_r0_sig"] = e.r0_x / f["px_vol60"] if f["px_vol60"] else np.nan
        # ── 실적 숫자(뻔한 이유) ──
        f["obv_eps_surprise"] = e.eps_surprise
        # ── 펀드매니저(발표 전 20거래일, 반응 뒤 5거래일) ──
        d_pre20, d_pre, d_post5 = str(idx[k - 21].date()), str(idx[k - 1].date()), str(idx[min(k + 5, n - 1)].date())
        pre_a, post_a, new_pre, out_post, held_n = [], [], False, False, 0
        for fund, tab in tabs.items():
            a, new, _ = active(tab, d_pre20, d_pre, tk)
            if not np.isnan(a):
                pre_a.append(a)
            new_pre |= new
            if getattr(e, f"held_{fund}"):
                held_n += 1
                if k + 5 < n:
                    a2, _, gone = active(tab, d_pre, d_post5, tk)
                    if not np.isnan(a2):
                        post_a.append(a2)
                    out_post |= gone
        f["fm_pre_active"] = np.mean(pre_a) if pre_a else np.nan
        f["fm_new_pre"] = new_pre
        f["fm_both"] = held_n == 2
        f["fm_weight"] = e.w_TIME + e.w_KoAct
        f["post_fm_active5"] = np.mean(post_a) if post_a else np.nan
        f["post_fm_exit5"] = out_post
        # ── OCC 고객 옵션(발표 전 5거래일 vs 그 전 60거래일) ──
        ob = occ_by.get(tk)
        if ob is not None and len(ob) > 40:
            d_end = idx[k - 1]
            w5 = ob.loc[(ob.index > idx[k - 6]) & (ob.index <= d_end)]
            prior = ob.loc[(ob.index > idx[k - 66]) & (ob.index <= idx[k - 6])]
            if len(w5) >= 3 and len(prior) >= 30:
                for col, nm in (("C_C", "call"), ("tot", "tot")):
                    b = prior[col].median()
                    f[f"occ_{nm}_ab"] = np.log((w5[col].mean() + 1) / (b + 1))
                cc = w5.C_C.sum() / max(w5.C_C.sum() + w5.C_P.sum(), 1)
                pc = (prior.C_C / (prior.C_C + prior.C_P).replace(0, np.nan)).median()
                f["occ_cust_callshare_ch"] = cc - pc
                f["occ_cust_share"] = (w5.C_C.sum() + w5.C_P.sum()) / max(w5.tot.sum(), 1)
                f["occ_opt_to_stock"] = 100 * prior.tot.median() / max(v.iloc[k - 66:k - 6].median(), 1)
        # ── SEIBRO 한국 개인(공개된 결제일만: 반응일 전날까지의 SEIBRO 날짜) ──
        kb = kr_by.get(tk)
        d_lo, d_hi = idx[k - 22], idx[k - 1]
        if kb is not None:
            w = kb.loc[(kb.index > d_lo) & (kb.index <= d_hi)]
            f["kr_days_pre"] = len(w)
            f["kr_buy_pre_m"] = w.buy.sum() / 1e6
            f["kr_net_pre_m"] = w.net.sum() / 1e6
            f["kr_ever_before"] = bool((kb.index <= d_hi).any())
            f["kr_days_250"] = int(((kb.index > idx[max(0, k - 250)]) & (kb.index <= d_hi)).sum())
            wp = kb.loc[(kb.index > idx[min(k + 1, n - 1)]) & (kb.index <= idx[min(k + 22, n - 1)])]
            f["post_kr_buy_m"] = wp.buy.sum() / 1e6 if k + 22 < n else np.nan
            f["post_kr_net_m"] = wp.net.sum() / 1e6 if k + 22 < n else np.nan
        else:
            f.update(kr_days_pre=0, kr_buy_pre_m=0.0, kr_net_pre_m=0.0, kr_ever_before=False, kr_days_250=0,
                     post_kr_buy_m=0.0 if k + 22 < n else np.nan, post_kr_net_m=0.0 if k + 22 < n else np.nan)
        dvm = f["px_dvol_m"] * 20 if f["px_dvol_m"] else np.nan
        f["kr_buy_share_pre"] = 100 * f["kr_buy_pre_m"] / dvm if dvm else np.nan  # 20일 미국 거래대금 중 한국인 매수 %
        cb = krc_by.get(tk)
        if cb is not None:
            pub = idx[k - 3]  # 보관은 2평일 뒤 공개
            cc = cb.loc[cb.index <= pub]
            f["kr_custody_top50"] = bool(len(cc) and (pub - cc.index[-1]).days <= 7)
            f["kr_custody_rank"] = int(cc["rank"].iloc[-1]) if f["kr_custody_top50"] else np.nan
        else:
            f["kr_custody_top50"], f["kr_custody_rank"] = False, np.nan
        rows.append(f)
    feat = pd.DataFrame(rows)
    feat.to_csv(BASE / "results/features.csv", index=False)
    print(feat.shape)
    print(feat.notna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
