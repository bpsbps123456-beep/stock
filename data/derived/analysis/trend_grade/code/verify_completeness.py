"""추세 등급 탭 이식 완결성 점검 (V4, 2026-10-12). 저장소 파일은 읽기만 하고 결과는 results/verify_completeness.json.

연구(harness·final.py FINAL_E14s10_HYB)는 미국 보유 종목·그날 보유(시점 일치)·연구 패널 가격으로 쟀다.
탭(desk/trend.py 초안)은 오늘 보유 전체(미국+해외)·yfinance 2년 일봉·미국 달력에 해외 값을 최대 5일 이어 쓴다.
그 차이가 등급을 얼마나 바꾸는지 잰다.

 A 해외 9종목 섞기: 휴장 공백·5일 한도 초과로 빈 칸, 통화(현지 vs 달러), 미국 종목 등급 바뀜, S·A 칸 점유
 B 상장 130~199봉: SMA200 확장 평균·252일 고가 단축 효과(긴 이력 종목을 k봉으로 잘라 재현) + 실제 신규 상장
 C 보유 기준: 그날 보유(연구) vs 오늘 보유 고정(탭)의 과거 등급 차이와 생존 색
 D yfinance: 종가 vs 수정종가(배당), 연구 패널과 차이, 분할·가격제한 위반 점프, 거래량 0 봉, 못 받은 기호
 E 관문 D 부풀림: 2026-07, 2025-04 등급 비율과 관문 D의 이후 수익
 F 10일 평활 지연: 7월 충격 S 종목이 S에서 내려가기까지 거래일(평활 vs 원점수), 2026 전체 S 이탈·진입 지연
 G 화면 경고 후보: 그 시점에 아는 지난 완료 IC가 다음 IC를 맞히는가

시험 범위: A 해외 9종목 × 통화 2가지 × 2구간, B k = 130·150·175·199 4가지 × 2026 거래일, F 평활 0·10일 2가지,
G 지난 완료 신호 20개 평균 1가지(h 10·20). 결과를 보고 고른 변형 없음(측정만).
yfinance 2년 일봉은 임시 폴더(trend_grade_verify_yf2y.parquet)에 한 번 받아 둔다.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import harness as H  # noqa: E402

sys.path.insert(0, H.ROOT)
from gbrc.etf.market import company_key, is_cash_holding, listing_market, yahoo_ticker  # noqa: E402
from gbrc.etf.storage import load_normalized_snapshot  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "verify_completeness.json")
CACHE = os.path.join(tempfile.gettempdir(), "trend_grade_verify_yf2y.parquet")
FOREIGN = ["009150.KS", "0043Y0.KS", "285A.T", "6981.T", "5803.T", "2513.HK", "0100.HK", "300502.SZ", "300308.SZ"]
FX = {"KS": "KRW=X", "T": "JPY=X", "HK": "HKD=X", "SZ": "CNY=X"}
LIMITS = {"KS": 0.30, "SZ": 0.20}  # 일간 가격제한(코스피·코스닥 30%, 창업판 20%). 넘으면 미반영 분할·권리락 의심
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
PARTS = ("L", "M", "H", "Q")
GR = ("S", "A", "B", "C", "D")
ORDER = {g: i for i, g in enumerate(GR)}
TAB_DAYS = 165  # desk/trend.py cal = 최근 max(160, 30) + 5 미국 거래일
CHART_DAYS = 160


def r_(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if np.isfinite(x) else None


def gof(p: float) -> str:
    return next((g for g, fl in H.PYRAMID if p > fl), "D")


# ----------------------------------------------------------------------------- 탭과 같은 보유·가격
def tab_holdings() -> tuple[list[dict], dict]:
    """desk/trend.py holdings()와 같은 규칙(두 펀드 최신 스냅샷, 현금·선물 제외, 회사 하나에 한 줄)."""
    rows, dates = {}, {}
    for fid, label in (("456600", "TIME"), ("2ETFQ1", "KoAct")):
        p = sorted(Path(H.SNAPS).glob(f"*_{fid}.json"))[-1]
        as_of = p.name[:10]
        dates[label] = as_of
        for h in load_normalized_snapshot(p)["holdings"]:
            code = h.get("code")
            m = listing_market(code)
            if is_cash_holding(h) or m not in ("US", "KR", "JP", "HK", "CN"):
                continue
            sym = yahoo_ticker(code, as_of)
            if not sym:
                continue
            r = rows.setdefault(company_key(code), {"symbol": sym, "market": m, "funds": {}})
            r["funds"][label] = round(r["funds"].get(label, 0.0) + float(h["weight"]), 2)
    return list(rows.values()), dates


def download(symbols: list[str]) -> dict:
    """{field: 날짜 × 기호 넓은 표}. 임시 폴더에 긴 표로 저장해 다시 쓴다."""
    want = sorted(set(symbols))
    if os.path.exists(CACHE):
        long = pd.read_parquet(CACHE)
        if set(want) <= set(long["sym"].unique()):
            return {f: long.pivot(index="date", columns="sym", values=f) for f in FIELDS}
    import yfinance as yf

    fr = yf.download(want, period="2y", auto_adjust=False, progress=False, threads=True, group_by="column")
    parts = []
    for s in want:
        try:
            d = pd.DataFrame({f: fr[f][s] for f in FIELDS})
        except KeyError:
            continue
        d["sym"] = s
        d["date"] = pd.DatetimeIndex(d.index)
        parts.append(d.reset_index(drop=True))
    long = pd.concat(parts, ignore_index=True)
    long.to_parquet(CACHE)
    return {f: long.pivot(index="date", columns="sym", values=f) for f in FIELDS}


def bars_of(frame: dict, sym: str) -> pd.DataFrame | None:
    if sym not in frame["Close"].columns:
        return None
    b = pd.DataFrame({f: frame[f][sym] for f in FIELDS})
    b = b[b["Close"].notna() & (b["High"] > 0)]
    idx = pd.DatetimeIndex(b.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    b.index = idx.normalize()
    b = b[~b.index.duplicated(keep="last")]
    return b if len(b) else None


def clenow1(c: pd.Series, n: int = 126) -> pd.Series:
    return FC.clenow(c.to_frame("x"), n)["x"]


def ind_of(b: pd.DataFrame, price: str = "Close", fx: pd.Series | None = None) -> pd.DataFrame:
    """종목 하나의 재료(자기 거래일). price='Adj Close'면 종가·고가를 같은 비율로 수정. fx(현지/달러)면 달러로."""
    k = b[price] / b["Close"] if price != "Close" else 1.0
    c, hi = b["Close"] * k, b["High"] * k
    if fx is not None:
        f = fx.reindex(b.index, method="ffill").bfill()
        c, hi = c / f, hi / f
    s50 = c.rolling(50, min_periods=50).mean()
    s200 = c.rolling(200, min_periods=120).mean()
    h252 = hi.rolling(252, min_periods=120).max()
    out = pd.DataFrame({"L": c / s200 - 1, "M": c / s50 - 1, "H": c / h252 - 1, "Q": clenow1(c),
                        "C": c, "s50": s50, "s200": s200, "bars": np.arange(1, len(c) + 1, dtype=float)},
                       index=b.index)
    out["src"] = np.arange(len(c), dtype=float)  # 자기 봉 번호(이어 쓴 날 판별용)
    return out


def wide(inds: dict, cal: pd.DatetimeIndex, col: str, limit: int | None = 5) -> pd.DataFrame:
    """desk/trend.py score_table.wide와 같다: 미국 날짜에 그날 또는 직전 현지 값(최대 limit 칸 이어 씀)."""
    return pd.DataFrame({s: d[col].reindex(cal, method="ffill", limit=limit) for s, d in inds.items()}, index=cal)


def gate(g: pd.DataFrame, C: pd.DataFrame, s50: pd.DataFrame, s200: pd.DataFrame) -> pd.DataFrame:
    a = g.to_numpy(object).copy()
    ok = ((C > s50) & (s50 > s200)).to_numpy()
    dc = ((C < s200) & (s50 < s200)).to_numpy()
    graded = pd.notna(a)
    top = graded & ((a == "S") | (a == "A"))
    a[top & ~ok] = "B"
    a[graded & dc] = "D"
    return pd.DataFrame(a, index=g.index, columns=g.columns)


def tab_grades(inds: dict, cal: pd.DatetimeIndex, smooth: int = 10, limit: int | None = 5) -> tuple:
    """탭 순서 그대로: 재료별 단면 백분위(봉 130 이상) → 평균 → smooth일 평균 → 다시 백분위 → 피라미드 → 관문."""
    ok = wide(inds, cal, "bars", limit) >= H.MIN_BARS
    num, den = 0.0, 0.0
    for p in PARTS:
        x = wide(inds, cal, p, limit).replace([np.inf, -np.inf], np.nan).where(ok)
        pr = x.rank(axis=1, pct=True)
        num = num + pr.fillna(0.0)
        den = den + pr.notna().astype(float)
    raw = (num / den).where(den > 0)
    if smooth > 1:
        raw = raw.rolling(smooth, min_periods=1).mean().where(raw.notna())
    pct = raw.rank(axis=1, pct=True)
    g = H.grade_of(pct, H.PYRAMID)
    g = gate(g, wide(inds, cal, "C", limit), wide(inds, cal, "s50", limit), wide(inds, cal, "s200", limit))
    return g, pct


def diff_share(a: pd.DataFrame, b: pd.DataFrame, cols=None) -> dict:
    cols = list(a.columns) if cols is None else [c for c in cols if c in a.columns]
    x, y = a[cols], b.reindex(index=a.index, columns=cols)
    both = x.notna() & y.notna()
    ne = (x != y) & both
    only = x.notna() ^ y.notna()
    return {"cells": int(both.sum().sum()), "diff_pct": r_(100 * ne.sum().sum() / max(both.sum().sum(), 1), 1),
            "one_side_blank_cells": int(only.sum().sum())}


def counts(g: pd.DataFrame, cols=None) -> dict:
    g = g if cols is None else g[cols]
    n = g.notna().sum(axis=1).replace(0, np.nan)
    return {k: r_(((g == k).sum(axis=1) / n).mean() * 100, 1) for k in GR}


def panel_score(P: dict, um: pd.DataFrame, smooth: int = 10) -> pd.DataFrame:
    """연구 FC.composite + 평활과 같은 식, 보유 마스크만 바꿀 수 있게."""
    raw = FC.raw_ingredients(P, set(PARTS))
    num, den = None, None
    for k in PARTS:
        x = raw[k].replace([np.inf, -np.inf], np.nan)
        pr = H.pct_rank(x, um & x.notna())
        has = pr.notna().astype(float)
        num = pr.fillna(0.0) if num is None else num + pr.fillna(0.0)
        den = has if den is None else den + has
    sc = (num / den).where(den > 0)
    if smooth:
        sc = sc.rolling(smooth, min_periods=1).mean().where(sc.notna())
    return sc


# ============================================================================= A 해외 섞기
def part_A(rows, frame):
    us_rows = [r for r in rows if r["market"] == "US"]
    fx_rows = [r for r in rows if r["market"] != "US"]
    cal = bars_of(frame, "QQQ").index
    inds, missing = {}, []
    for r in rows:
        b = bars_of(frame, r["symbol"])
        if b is None:
            missing.append(r["symbol"])
            continue
        inds[r["symbol"]] = ind_of(b)
    fxs = {k: bars_of(frame, v)["Close"] for k, v in FX.items()}
    inds_usd = dict(inds)
    for r in fx_rows:
        s = r["symbol"]
        if s in inds:
            inds_usd[s] = ind_of(bars_of(frame, s), fx=fxs[s.split(".")[-1]])
    us = [r["symbol"] for r in us_rows if r["symbol"] in inds]
    fo = [r["symbol"] for r in fx_rows if r["symbol"] in inds]
    out = {"n_rows": len(rows), "n_us": len(us_rows), "n_foreign": len(fx_rows), "missing_yf": missing,
           "foreign_symbols": fo,
           "under_130_bars": [[r["symbol"], int(inds[r["symbol"]]["bars"].iloc[-1]), r["funds"]] for r in rows
                              if r["symbol"] in inds and inds[r["symbol"]]["bars"].iloc[-1] < H.MIN_BARS],
           "bars_130_199": [[r["symbol"], int(inds[r["symbol"]]["bars"].iloc[-1]), r["funds"]] for r in rows
                            if r["symbol"] in inds and H.MIN_BARS <= inds[r["symbol"]]["bars"].iloc[-1] < 200]}

    win = cal[cal >= pd.Timestamp("2026-01-02")]
    st = {}
    for s in fo:
        d = inds[s]
        src = d["src"].reindex(cal, method="ffill")  # 그 미국 날짜에 쓰는 현지 봉 번호
        same_day = pd.Series(cal.isin(d.index), index=cal)
        run, runs = 0, []
        for t in win:
            run = 0 if same_day[t] else run + 1
            runs.append(run)
        runs = pd.Series(runs, index=win)
        run2, n_over5, worst = 0, 0, (0, None)
        for t in cal[cal >= d.index[0]]:
            run2 = 0 if same_day[t] else run2 + 1
            n_over5 += run2 > 5
            if run2 > worst[0]:
                worst = (run2, str(t.date()))
        st[s] = {"max_carried_2y": worst[0], "max_carried_2y_at": worst[1], "blank_over5_2y": int(n_over5),
                 "bars_now": int(d["bars"].iloc[-1]), "first": str(d.index[0].date()), "last": str(d.index[-1].date()),
                 "us_days_2026": len(win), "us_days_without_local_bar_2026": int((~same_day.loc[win]).sum()),
                 "max_consecutive_carried": int(runs.max()), "blank_over5_2026": int((runs > 5).sum()),
                 "local_days_not_us_2026": int((~d.index[d.index >= win[0]].isin(cal)).sum()),
                 "bars_on_first_2026": r_(d["bars"].reindex(cal, method="ffill").get(win[0]), 0)}
    out["foreign_status"] = st
    out["us_last_bar"] = sorted({str(inds[s].index[-1].date()) for s in us})

    g_us, _ = tab_grades({s: inds[s] for s in us}, cal)
    g_all, p_all = tab_grades(inds, cal)
    g_usd, p_usd = tab_grades(inds_usd, cal)
    g_nolim, _ = tab_grades(inds, cal, limit=None)
    res = {}
    for wname, days in (("y2026", win), ("last160", cal[-CHART_DAYS:])):
        gu, ga, gd = g_us.loc[days], g_all.loc[days], g_usd.loc[days]
        fo_g = [s for s in fo if ga[s].notna().any()]
        nS, nA = (ga == "S").sum(axis=1), (ga == "A").sum(axis=1)
        fS, fA = (ga[fo] == "S").sum(axis=1), (ga[fo] == "A").sum(axis=1)
        sa_us = ((gu[us] == "S") | (gu[us] == "A"))
        sa_mx = ((ga[us] == "S") | (ga[us] == "A"))
        res[wname] = {
            "us_grade_changed_by_adding_foreign": diff_share(gu, ga, us),
            "us_SA_lost_cells": int((sa_us & ~sa_mx).sum().sum()), "us_SA_cells_us_only": int(sa_us.sum().sum()),
            "graded_total_mean": r_(ga.notna().sum(axis=1).mean(), 1),
            "foreign_graded_mean": r_(ga[fo].notna().sum(axis=1).mean(), 1),
            "foreign_share_of_graded_pct": r_(100 * (ga[fo].notna().sum(axis=1) / ga.notna().sum(axis=1)).mean(), 1),
            "foreign_share_of_S_pct": r_(100 * fS.sum() / max(nS.sum(), 1), 1),
            "foreign_share_of_A_pct": r_(100 * fA.sum() / max(nA.sum(), 1), 1),
            "days_with_foreign_in_S_pct": r_(100 * (fS > 0).mean(), 1),
            "foreign_grade_share": counts(ga, fo_g) if fo_g else None,
            "us_grade_share_mixed": counts(ga, us),
            "currency_usd_vs_local_foreign_diff": diff_share(ga, gd, fo),
            "currency_usd_vs_local_us_diff": diff_share(ga, gd, us),
            "currency_pct_shift_foreign_mean_abs_pp": r_(100 * (p_all[fo] - p_usd[fo]).loc[days].abs().stack().mean(), 2),
            "blank_from_limit5_cells": int((g_nolim.loc[days, fo].notna() & ga[fo].isna()).sum().sum()),
        }
    out["grades"] = res
    t = cal[-1]
    out["today"] = {"date": str(t.date()), "local": {s: g_all.at[t, s] for s in fo}, "usd": {s: g_usd.at[t, s] for s in fo},
                    "S_now": [s for s in g_all.columns if g_all.at[t, s] == "S"],
                    "A_now": [s for s in g_all.columns if g_all.at[t, s] == "A"],
                    "dist_now": {k: int((g_all.loc[t] == k).sum()) for k in GR},
                    "ungraded_now": [s for s in g_all.columns if pd.isna(g_all.at[t, s])]}
    cal_tab = cal[-TAB_DAYS:]
    g_tab, _ = tab_grades(inds, cal_tab)
    out["tab_165day_warmup"] = {"first10_diff": diff_share(g_all.loc[cal_tab[:10]], g_tab.loc[cal_tab[:10]]),
                                "rest_diff": diff_share(g_all.loc[cal_tab[10:]], g_tab.loc[cal_tab[10:]])}
    # 해외 등급 뒤 20 현지 거래일 수익(설명용, n 작음)
    rec = []
    for s in fo:
        c = inds[s]["C"]
        f20 = (c.shift(-20) / c - 1).reindex(cal, method="ffill", limit=5)
        for d0 in win:
            g = g_all.at[d0, s]
            if g is not None and pd.notna(f20.get(d0)):
                rec.append((d0, s, g, f20[d0]))
    fw = pd.DataFrame(rec, columns=["d", "s", "g", "r"])
    if len(fw):
        out["foreign_fwd20_local_by_grade_2026"] = {k: {"cells": int((fw.g == k).sum()), "names": int(fw.s[fw.g == k].nunique()),
                                                        "ret_mean_pct": r_(100 * fw.r[fw.g == k].mean(), 2)} for k in GR}
    return out, inds, cal


# ============================================================================= B 상장 130~199봉
def part_B(P, E) -> dict:
    C, Hi = P["C"], P["H"]
    um = H.universe_mask(P)
    raw = FC.raw_ingredients(P, set(PARTS))
    pr = {k: H.pct_rank(raw[k].replace([np.inf, -np.inf], np.nan), um & raw[k].notna()) for k in PARTS}
    comp = (sum(pr[k].fillna(0) for k in PARTS) / sum(pr[k].notna().astype(float) for k in PARTS)).where(um)
    s50, s200 = FC.sma(C, 50), FC.sma(C, 200)
    nbars = C.notna().cumsum()
    days = C.index[(C.index >= pd.Timestamp("2026-01-02")) & (C.index <= pd.Timestamp("2026-10-09"))]
    out = {"note": "긴 이력(봉 300 이상) 종목의 같은 가격 경로를 상장 k봉째로 가정: SMA200→k봉 평균, 252일 고가→k봉 고가. "
                   "M·Q는 그대로. 원점수(평활 전) 단면 백분위·피라미드+관문 등급 변화."}

    def sub(o: np.ndarray, x: float) -> float:
        return (1 + (o < x).sum() + 0.5 * (o == x).sum()) / (len(o) + 1)

    for k in (130, 150, 175, 199):
        smak = C.rolling(k, min_periods=k).mean()
        Lk = C / smak - 1
        Hk = C / Hi.rolling(k, min_periods=k).max() - 1
        rec = []
        for t in days:
            m = um.loc[t]
            names = list(m[m].index)
            if len(names) < 20:
                continue
            Lf, Hf, Cf = raw["L"].loc[t, names], raw["H"].loc[t, names], comp.loc[t, names]
            for n in names:
                if nbars.at[t, n] < 300:
                    continue
                pl = sub(Lf.drop(n).dropna().to_numpy(), Lk.at[t, n])
                ph = sub(Hf.drop(n).dropna().to_numpy(), Hk.at[t, n])
                vals = [v for v in (pl, pr["M"].at[t, n], ph, pr["Q"].at[t, n]) if np.isfinite(v)]
                cn = float(np.mean(vals))
                oth = Cf.drop(n).dropna().to_numpy()
                po, pn = sub(oth, Cf[n]), sub(oth, cn)
                c_, a50, a200, ak = C.at[t, n], s50.at[t, n], s200.at[t, n], smak.at[t, n]
                gf, gk = (c_ > a50) and (a50 > a200), (c_ > a50) and (a50 > ak)
                df_, dk = (c_ < a200) and (a50 < a200), (c_ < ak) and (a50 < ak)

                def hyb(p, gt, dd):
                    g = gof(p)
                    g = "B" if g in ("S", "A") and not gt else g
                    return "D" if dd else g
                rec.append((pl - pr["L"].at[t, n], ph - pr["H"].at[t, n], pn - po, hyb(po, gf, df_), hyb(pn, gk, dk),
                            gf != gk, df_ != dk, raw["L"].at[t, n]))
        df = pd.DataFrame(rec, columns=["dL", "dH", "dpct", "g0", "g1", "gate_flip", "d_flip", "Lfull"])
        up = df.g1.map(ORDER) < df.g0.map(ORDER)
        dn = df.g1.map(ORDER) > df.g0.map(ORDER)
        out[f"k{k}"] = {"cells": len(df), "dL_pp_mean": r_(100 * df.dL.mean(), 1), "dL_pp_mean_abs": r_(100 * df.dL.abs().mean(), 1),
                        "dH_pp_mean": r_(100 * df.dH.mean(), 1), "dpct_pp_mean": r_(100 * df.dpct.mean(), 1),
                        "dpct_pp_mean_abs": r_(100 * df.dpct.abs().mean(), 1),
                        "dpct_pp_p90_abs": r_(100 * df.dpct.abs().quantile(0.9), 1),
                        "grade_changed_pct": r_(100 * (df.g0 != df.g1).mean(), 1),
                        "upgraded_pct": r_(100 * up.mean(), 1), "downgraded_pct": r_(100 * dn.mean(), 1),
                        "gate_flip_pct": r_(100 * df.gate_flip.mean(), 1), "dcond_flip_pct": r_(100 * df.d_flip.mean(), 1),
                        "dpct_pp_when_Lfull_gt_0.3": r_(100 * df.dpct[df.Lfull > 0.3].mean(), 1),
                        "dpct_pp_when_Lfull_lt_0": r_(100 * df.dpct[df.Lfull < 0].mean(), 1),
                        "S_full_cells": int((df.g0 == "S").sum()), "S_kept_pct": r_(100 * ((df.g0 == "S") & (df.g1 == "S")).sum() / max((df.g0 == "S").sum(), 1), 1),
                        "D_full_cells": int((df.g0 == "D").sum()), "D_kept_pct": r_(100 * ((df.g0 == "D") & (df.g1 == "D")).sum() / max((df.g0 == "D").sum(), 1), 1)}
    # 실제 신규 상장: 봉 130~199인 날의 등급과 h20 초과수익(그날 eligible 평균 대비)
    sc = FC.make(FC.GRID["E14"], 10)(P)
    G = FL.hyb_grades(sc, P, E).where(E)
    fr = H.forward(20).where(E)
    ex = fr.sub(fr.mean(axis=1), axis=0)
    first = C.apply(lambda x: x.first_valid_index())
    late = [c for c in C.columns if first[c] > C.index[0]]
    young = (nbars >= 130) & (nbars < 200) & E
    young.loc[:, [c for c in C.columns if c not in late]] = False
    ys = {}
    for n in [c for c in C.columns if young[c].any()]:
        d = young.index[young[n]]
        g = G.loc[d, n]
        ys[n] = {"from": str(d[0].date()), "to": str(d[-1].date()), "days": len(d),
                 "grades": {k: int((g == k).sum()) for k in GR}, "ex20_pct_mean": r_(100 * ex.loc[d, n].mean(), 2)}
    out["actual_young_listings"] = ys
    yw = G.where(young).loc["2025-06-01":"2026-10-09"]
    ow = G.where(E & (nbars >= 200)).loc["2025-06-01":"2026-10-09"]
    out["share_2025_06_on"] = {"young": counts(yw.dropna(axis=1, how="all")),
                               "young_cells": int(yw.notna().sum().sum()), "mature": counts(ow)}
    return out


# ============================================================================= C 보유 기준
def part_C(P, rows) -> dict:
    C = P["C"]
    today_us = sorted({H.ALIASES.get(r["symbol"], r["symbol"]) for r in rows if r["market"] == "US"})
    in_panel = [s for s in today_us if s in C.columns]
    base = (C.notna().cumsum() >= H.MIN_BARS) & C.notna()
    um_pit = H.universe_mask(P)
    member = pd.DataFrame(np.zeros(C.shape, bool), index=C.index, columns=C.columns)
    member[in_panel] = True
    um_today = base & member
    sc_pit, sc_tod = panel_score(P, um_pit), panel_score(P, um_today)
    G_pit, G_tod = FL.hyb_grades(sc_pit, P, um_pit), FL.hyb_grades(sc_tod, P, um_today)
    fin = FC.make(FC.GRID["E14"], 10)(P)
    same = bool(np.allclose(fin.fillna(-9).to_numpy(), sc_pit.fillna(-9).to_numpy()))
    cal = C.index
    out = {"today_us_n": len(today_us), "today_us_in_panel": len(in_panel),
           "not_in_panel": [s for s in today_us if s not in C.columns], "final_score_reproduced": same}
    E = base & P["O"].shift(-1).notna()  # 보유 여부와 무관한 채점 자격(마스크 M과 곱해 쓴다)
    fr = H.forward(20)
    for wname, days in (("last160", cal[-CHART_DAYS:]), ("y2026", cal[cal >= pd.Timestamp("2026-01-02")])):
        gp, gt = G_pit.loc[days, in_panel], G_tod.loc[days, in_panel]
        held = um_pit.loc[days, in_panel]
        shown = gt.notna()
        o = {"cells_shown": int(shown.sum().sum()),
             "shown_but_not_held_then_pct": r_(100 * (shown & ~held).sum().sum() / max(shown.sum().sum(), 1), 1),
             "grade_diff_when_held_then": diff_share(gp.where(held), gt.where(held)),
             "S_cells_today_set": int((gt == "S").sum().sum()),
             "S_today_set_also_S_pit_pct": r_(100 * ((gt == "S") & (gp == "S")).sum().sum() / max((gt == "S").sum().sum(), 1), 1)}
        d0, d1 = days[0], days[-1]
        h0 = um_pit.loc[d0]
        h0 = list(h0[h0].index)
        stay = [s for s in h0 if s in in_panel]
        gone = [s for s in h0 if s not in in_panel]
        newin = [s for s in in_panel if s not in h0 and pd.notna(C.at[d0, s])]
        ret = C.loc[d1] / C.loc[d0] - 1
        o["survivor"] = {"start": str(d0.date()), "end": str(d1.date()),
                         "held_start_and_today": len(stay), "ret_pct": r_(100 * ret[stay].mean(), 1),
                         "held_start_gone_today": len(gone), "gone_ret_pct": r_(100 * ret[gone].mean(), 1),
                         "added_later": len(newin), "added_ret_pct": r_(100 * ret[newin].mean(), 1)}
        dd = days[days.isin(H.window_dates("back", 20).append(H.window_dates("front", 20)))]
        if len(dd):
            frw = fr.loc[dd]

            def s_ex(G, M):
                f = frw.where(M.loc[dd] & E.loc[dd])
                g = G.loc[dd]
                return {k: r_(100 * (f.where(g == k).mean(axis=1) - f.mean(axis=1)).mean(), 2) for k in GR}
            o["h20_ex_by_grade_pit"] = s_ex(G_pit, um_pit)
            o["h20_ex_by_grade_today_set"] = s_ex(G_tod, um_today)
            o["h20_days"] = len(dd)
        out[wname] = o
    # 탭 '최근 성적'은 오늘 보유 종목만 본다 → 지난 20일 사이 판 종목이 빠진다. t−20 보유 중 t에 남은 것 vs 판 것의 20일 수익.
    rec = []
    days = cal[(cal >= pd.Timestamp("2026-02-02")) & (cal <= pd.Timestamp("2026-10-09"))]
    for t in days:
        t0 = cal[cal.get_loc(t) - 20]
        h0, h1 = um_pit.loc[t0], um_pit.loc[t]
        r = C.loc[t] / C.loc[t0] - 1
        stay = h0 & h1 & r.notna()
        gone = h0 & ~h1 & r.notna()
        if gone.sum():
            rec.append((r[stay].mean(), r[gone].mean(), int(gone.sum()), int(stay.sum())))
    df = pd.DataFrame(rec, columns=["stay", "gone", "n_gone", "n_stay"])
    out["realized_exit_bias_2026"] = {"days": len(df), "stay_ret20_pct": r_(100 * df.stay.mean(), 2),
                                      "gone_ret20_pct": r_(100 * df.gone.mean(), 2),
                                      "gone_per_day": r_(df.n_gone.mean(), 1), "stay_per_day": r_(df.n_stay.mean(), 1)}
    return out


# ============================================================================= H 탭 초안의 현재 설계
def rank_vs(x: pd.DataFrame, ref: list) -> pd.DataFrame:
    """desk/trend.py rank_vs와 같다: ref끼리 보통 백분위, ref 밖은 ref 분포 안 위치."""
    cols = [c for c in x.columns if c in ref]
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    out[cols] = x[cols].rank(axis=1, pct=True)
    R = x[cols].to_numpy(float)
    n = np.isfinite(R).sum(axis=1)
    for c in x.columns:
        if c in cols:
            continue
        v = x[c].to_numpy(float)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            p = ((R < v).sum(axis=1) + 0.5 * (R == v).sum(axis=1) + 0.5) / n
        p[~np.isfinite(v[:, 0]) | (n == 0)] = np.nan
        out[c] = np.minimum(p, 1.0)
    return out


def tab_grades_ref(inds: dict, cal: pd.DatetimeIndex, ref: list, smooth: int = 10, limit: int | None = 5) -> tuple:
    ok = wide(inds, cal, "bars", limit) >= H.MIN_BARS
    num, den = 0.0, 0.0
    for p in PARTS:
        x = wide(inds, cal, p, limit).replace([np.inf, -np.inf], np.nan).where(ok)
        pr = rank_vs(x, ref)
        num = num + pr.fillna(0.0)
        den = den + pr.notna().astype(float)
    raw = (num / den).where(den > 0)
    sm = raw.rolling(smooth, min_periods=1).mean().where(raw.notna()) if smooth > 1 else raw
    pct = rank_vs(sm, ref)
    g = H.grade_of(pct, H.PYRAMID)
    g = gate(g, wide(inds, cal, "C", limit), wide(inds, cal, "s50", limit), wide(inds, cal, "s200", limit))
    return g, pct


def part_H(rows, frame, inds, cal) -> dict:
    """초안(01:02 기준): 순위는 미국 보유끼리(ref), 해외는 그 분포에 견줌. SKHY는 봉 130 미만이면 000660.KS 이력."""
    us = [r["symbol"] for r in rows if r["market"] == "US" and r["symbol"] in inds]
    fo = [r["symbol"] for r in rows if r["market"] != "US" and r["symbol"] in inds]
    out = {}
    hx = bars_of(frame, "000660.KS")
    ad = bars_of(frame, "SKHY")
    fxk = bars_of(frame, "KRW=X")["Close"]
    inds2 = dict(inds)
    inds2["SKHY"] = ind_of(hx)  # 초안처럼 원화 000660.KS 이력을 SKHY 자리에
    ref = list(us)  # SKHY는 미국 상장이라 ref 안(원화 이력이 미국 분포에 들어감)
    g, pct = tab_grades_ref(inds2, cal, ref)
    g_us_only, _ = tab_grades({s: inds2[s] for s in us}, cal)
    win = cal[cal >= pd.Timestamp("2026-01-02")]
    gw = g.loc[win]
    nS = (gw == "S").sum(axis=1)
    nA = (gw == "A").sum(axis=1)
    out["ref_ranking_2026"] = {
        "us_grades_same_as_us_only_pct": r_(100 - diff_share(g_us_only.loc[win], gw, us)["diff_pct"], 1),
        "S_total_per_day_mean": r_(nS.mean(), 2), "S_total_per_day_max": int(nS.max()),
        "S_foreign_per_day_mean": r_((gw[fo] == "S").sum(axis=1).mean(), 2),
        "A_total_per_day_mean": r_(nA.mean(), 2), "A_foreign_per_day_mean": r_((gw[fo] == "A").sum(axis=1).mean(), 2),
        "days_S_over_quota_pct": r_(100 * (nS > (gw[us].notna().sum(axis=1) * 0.1).round() + 0).mean(), 1),
        "foreign_grade_share": counts(gw, [s for s in fo if gw[s].notna().any()]),
        "today": {s: g.at[cal[-1], s] for s in fo + ["SKHY"]}}
    fxs = {k: bars_of(frame, v)["Close"] for k, v in FX.items()}
    inds_usd = dict(inds2)
    for s_ in fo:
        inds_usd[s_] = ind_of(bars_of(frame, s_), fx=fxs[s_.split(".")[-1]])
    gu, pu = tab_grades_ref(inds_usd, cal, ref)
    out["ref_ranking_2026"]["currency_usd_vs_local_foreign_diff"] = diff_share(gw, gu.loc[win], fo)
    out["ref_ranking_2026"]["currency_pct_shift_foreign_mean_abs_pp"] = r_(100 * (pct[fo] - pu[fo]).loc[win].abs().stack().mean(), 2)
    out["ref_ranking_2026"]["SA_cells_local"] = int(gw[fo].isin(["S", "A"]).sum().sum())
    out["ref_ranking_2026"]["SA_cells_usd"] = int(gu.loc[win, fo].isin(["S", "A"]).sum().sum())
    # SKHY 대용: ADR 상장 뒤 ADR(달러)과 000660.KS(원화·달러 환산) 수익 비교
    j0 = ad.index[0]
    h_usd = hx["Close"] / fxk.reindex(hx.index, method="ffill")
    a_ret = ad["Close"].pct_change().dropna()
    k_ret = h_usd.reindex(ad.index, method="ffill").pct_change().dropna()
    k_ret_next = h_usd.pct_change().shift(-1).reindex(ad.index, method="ffill").dropna()  # 서울 다음 날(시차)
    jj = a_ret.index.intersection(k_ret.index)
    jn = a_ret.index.intersection(k_ret_next.index)
    out["skhy_proxy"] = {
        "adr_first": str(j0.date()), "adr_bars": len(ad), "kr_bars": len(hx),
        "corr_same_date": r_(np.corrcoef(a_ret.loc[jj], k_ret.loc[jj])[0, 1], 3),
        "corr_seoul_next_day": r_(np.corrcoef(a_ret.loc[jn], k_ret_next.loc[jn])[0, 1], 3),
        "adr_ret_since_listing_pct": r_(100 * (ad["Close"].iloc[-1] / ad["Close"].iloc[0] - 1), 1),
        "kr_usd_ret_same_span_pct": r_(100 * (h_usd.loc[:ad.index[-1]].iloc[-1] / h_usd.loc[j0:].iloc[0] - 1), 1),
        "kr_krw_ret_same_span_pct": r_(100 * (hx["Close"].loc[:ad.index[-1]].iloc[-1] / hx["Close"].loc[j0:].iloc[0] - 1), 1),
        "grade_krw_vs_usd_2026_diff": None}
    others = [u for u in us if u != "SKHY"]
    g_wo, _ = tab_grades_ref({k: v for k, v in inds2.items() if k != "SKHY"}, cal, others)
    out["skhy_proxy"]["other_us_grade_diff_with_vs_without_proxy_2026"] = diff_share(gw, g_wo.loc[win], others)
    inds3 = dict(inds2)
    inds3["SKHY"] = ind_of(hx, fx=fxk)
    g3, p3 = tab_grades_ref(inds3, cal, ref)
    out["skhy_proxy"]["grade_krw_vs_usd_2026_diff"] = diff_share(g.loc[win], g3.loc[win], ["SKHY"])
    out["skhy_proxy"]["pct_krw_vs_usd_mean_abs_pp"] = r_(100 * (p3["SKHY"] - pct["SKHY"]).loc[win].abs().mean(), 2)
    # 봉 130 도달 때 ADR 자기 이력으로 바뀌는 순간의 점프 추정: 000660(달러) 전체 이력 vs 마지막 130봉만
    full = ind_of(hx, fx=fxk)
    tail = ind_of(hx.iloc[-130:], fx=fxk)
    last = full.index[-1]
    out["skhy_proxy"]["switch_jump_estimate_now"] = {
        "L_full": r_(100 * full.at[last, "L"], 1), "L_130only": r_(100 * tail.at[last, "L"], 1),
        "H_full": r_(100 * full.at[last, "H"], 1), "H_130only": r_(100 * tail.at[last, "H"], 1),
        "Q_full": r_(full.at[last, "Q"], 2), "Q_130only": r_(tail.at[last, "Q"], 2)}
    inds4 = dict(inds3)
    inds4["SKHY"] = tail
    g4, p4 = tab_grades_ref(inds4, cal[-20:], ref, smooth=1)
    g3b, p3b = tab_grades_ref(inds3, cal[-20:], ref, smooth=1)
    out["skhy_proxy"]["switch_jump_estimate_now"].update(
        {"raw_pct_full": r_(100 * p3b.at[cal[-1], "SKHY"], 1), "raw_pct_130only": r_(100 * p4.at[cal[-1], "SKHY"], 1),
         "grade_full": g3b.at[cal[-1], "SKHY"], "grade_130only": g4.at[cal[-1], "SKHY"]})
    return out


# ============================================================================= D yfinance 자료
def part_D(P, rows, frame, inds, cal) -> dict:
    C = P["C"]
    out = {}
    us = [r["symbol"] for r in rows if r["market"] == "US"]
    dc = []
    for s in us:
        b = bars_of(frame, s)
        ps = H.ALIASES.get(s, s)
        if b is None or ps not in C.columns:
            continue
        j = b.index.intersection(C.index[C.index >= pd.Timestamp("2026-01-02")])
        pc = C.loc[j, ps]
        e1 = (b.loc[j, "Close"] / pc - 1).abs()
        e2 = (b.loc[j, "Adj Close"] / pc - 1).abs()
        dc.append((s, float(e1.median()), float(e1.max()), str(e1.idxmax().date()), float(e2.median())))
    dcf = pd.DataFrame(dc, columns=["s", "med", "max", "at", "adjmed"])
    out["panel_vs_yf_close_2026"] = {"n": len(dcf), "median_abs_pct": r_(100 * dcf["med"].median(), 3),
                                     "worst5": [[s, r_(100 * m, 2), a] for s, m, a in dcf.sort_values("max").tail(5)[["s", "max", "at"]].values]}
    out["panel_vs_yf_adjclose_2026_median_abs_pct"] = r_(100 * dcf["adjmed"].median(), 3)
    divy = {}
    for r in rows:
        b = bars_of(frame, r["symbol"])
        if b is None or len(b) < 200:
            continue
        a = b["Adj Close"] / b["Close"]
        divy[r["symbol"]] = float(a.iloc[-1] / a.iloc[-200] - 1)
    dv = pd.Series(divy).sort_values()
    out["dividend_effect_200bar_pct"] = {"median": r_(100 * dv.median(), 3), "top5": [[k, r_(100 * v, 2)] for k, v in dv.tail(5).items()]}
    inds_adj = {r["symbol"]: ind_of(bars_of(frame, r["symbol"]), price="Adj Close") for r in rows if bars_of(frame, r["symbol"]) is not None}
    g0, _ = tab_grades(inds, cal)
    g1, _ = tab_grades(inds_adj, cal)
    out["close_vs_adjclose_grade_diff_last160"] = diff_share(g0.loc[cal[-CHART_DAYS:]], g1.loc[cal[-CHART_DAYS:]])
    usp = [s for s in us if H.ALIASES.get(s, s) in C.columns and s in inds]
    pan = {}
    for s in usp:
        ps = H.ALIASES.get(s, s)
        b = pd.DataFrame({"Close": C[ps], "High": P["H"][ps]}).dropna()
        b["Adj Close"] = b["Close"]
        pan[s] = ind_of(b)
    c2 = cal[cal <= C.index[-1]]
    gp, _ = tab_grades(pan, c2)
    gy, _ = tab_grades({s: inds[s] for s in usp}, c2)
    out["panel_vs_yf_grade_diff_last160_us"] = diff_share(gp.loc[c2[-CHART_DAYS:]], gy.loc[c2[-CHART_DAYS:]])
    jumps, zero, flat = [], {}, {}
    for r in rows:
        s = r["symbol"]
        b = bars_of(frame, s)
        if b is None:
            continue
        ret = b["Close"].pct_change()
        lim = LIMITS.get(s.split(".")[-1]) if "." in s else None
        thr = (lim + 0.005) if lim else 0.45
        for d, v in ret[ret.abs() > thr].items():
            jumps.append([s, str(d.date()), r_(100 * v, 1), f">{int(100 * thr)}%"])
        z = b[b["Volume"].fillna(0) == 0]
        if len(z):
            zero[s] = {"n": len(z), "last": str(z.index[-1].date())}
        fl = b[(b["High"] == b["Low"]) & (b["Close"] == b["Close"].shift(1))]
        if len(fl):
            flat[s] = len(fl)
    out["jumps"] = jumps
    out["zero_volume_bars"] = zero
    out["flat_repeat_bars"] = flat
    return out


# ============================================================================= E 관문 D 부풀림
def part_E(P, E) -> dict:
    C = P["C"]
    sc = FC.make(FC.GRID["E14"], 10)(P)
    pr = H.pct_rank(sc, E & sc.notna())
    G0 = H.grade_of(pr, H.PYRAMID).where(E)
    G = FL.hyb_grades(sc, P, E).where(E)
    fr = H.forward(20).where(E)
    ex = fr.sub(fr.mean(axis=1), axis=0)
    wins = {"2026_07": ("2026-07-01", "2026-07-31"), "2026_06_22_07_29": ("2026-06-22", "2026-07-29"),
            "2025_04": ("2025-04-01", "2025-04-30"), "front": ("2026-01-02", "2026-05-29"),
            "back": ("2026-06-01", "2026-10-09"), "y2025": ("2025-01-02", "2025-12-31")}
    out = {}
    for name, (a, b) in wins.items():
        d = C.index[(C.index >= pd.Timestamp(a)) & (C.index <= pd.Timestamp(b))]
        g, g0 = G.loc[d], G0.loc[d]
        n = g.notna().sum(axis=1).replace(0, np.nan)
        gate_D = (g == "D") & (g0 != "D")
        dem = (g0 == "S") | (g0 == "A")
        dem = dem & (g == "B")
        eD = ex.loc[d]
        dsh = (g == "D").sum(axis=1) / n
        out[name] = {"days": len(d), "share_pct": counts(g), "D_share_max_pct": r_(100 * dsh.max(), 1),
                     "D_share_max_day": str(dsh.idxmax().date()) if dsh.notna().any() else None,
                     "S_count_min": int((g == "S").sum(axis=1).min()), "S_count_mean": r_((g == "S").sum(axis=1).mean(), 1),
                     "gate_only_D_share_pct": r_(100 * (gate_D.sum(axis=1) / n).mean(), 1),
                     "SA_demoted_to_B_per_day": r_(dem.sum(axis=1).mean(), 1),
                     "h20_ex_pct": {"rank_D": r_(100 * eD.where((g == "D") & (g0 == "D")).stack().mean(), 2),
                                    "gate_only_D": r_(100 * eD.where(gate_D).stack().mean(), 2),
                                    "SA_demoted_to_B": r_(100 * eD.where(dem).stack().mean(), 2),
                                    "S": r_(100 * eD.where(g == "S").stack().mean(), 2)}}
    return out


# ============================================================================= F 평활 지연
def part_F(P, E) -> dict:
    C, O = P["C"], P["O"]
    G10 = FL.hyb_grades(FC.make(FC.GRID["E14"], 10)(P), P, E).where(E)
    G0 = FL.hyb_grades(FC.make(FC.GRID["E14"], 0)(P), P, E).where(E)
    cal = C.index
    t0 = pd.Timestamp("2026-06-22")
    i0 = cal.get_loc(t0)
    names = [n for n in G10.columns if G10.at[t0, n] == "S"]

    def first(G, n, test):
        col = G.columns.get_loc(n)
        for j in range(i0 + 1, min(i0 + 60, len(cal))):
            v = G.iat[j, col]
            if v is not None and test(v):
                return j - i0
        return None

    def ret(n, k):
        return r_(100 * (C.iat[i0 + k, C.columns.get_loc(n)] / C.at[t0, n] - 1), 1) if k is not None else None
    rows = {}
    for n in names:
        a10 = first(G10, n, lambda v: v != "S")
        a0 = 0 if G0.at[t0, n] != "S" else first(G0, n, lambda v: v != "S")
        rows[n] = {"s10_leave_S_days": a10, "s0_leave_S_days": a0,
                   "s10_to_CD_days": first(G10, n, lambda v: v in ("C", "D")), "s0_to_CD_days": first(G0, n, lambda v: v in ("C", "D")),
                   "ret_at_s10_leave_pct": ret(n, a10), "ret_at_s0_leave_pct": ret(n, a0),
                   "ret_6_22_to_7_29_pct": r_(100 * (C.at[pd.Timestamp("2026-07-29"), n] / C.at[t0, n] - 1), 1),
                   "path_s10": "".join((G10.iat[i0 + j, G10.columns.get_loc(n)] or "-") for j in range(0, 27)),
                   "path_s0": "".join((G0.iat[i0 + j, G0.columns.get_loc(n)] or "-") for j in range(0, 27))}
    out = {"july_S_on_2026_06_22": rows}
    v = list(rows.values())
    pick = lambda k: [x[k] for x in v if x[k] is not None]  # noqa: E731
    out["july_summary"] = {"n": len(v), "s10_leave_median": r_(np.median(pick("s10_leave_S_days")), 1),
                           "s0_leave_median": r_(np.median(pick("s0_leave_S_days")), 1),
                           "s10_ret_at_leave_mean": r_(np.mean(pick("ret_at_s10_leave_pct")), 1),
                           "s0_ret_at_leave_mean": r_(np.mean(pick("ret_at_s0_leave_pct")), 1)}

    a10, a0 = G10.to_numpy(object), G0.to_numpy(object)
    Oa = O.to_numpy(float)
    jdays = np.where((cal >= pd.Timestamp("2026-01-02")) & (cal <= pd.Timestamp("2026-09-10")))[0]

    def lags(enter: bool) -> dict:
        rec = []
        for j in jdays:
            for k in range(a0.shape[1]):
                if enter:
                    trig = a0[j, k] == "S" and a0[j - 1, k] is not None and a0[j - 1, k] != "S" and a10[j - 1, k] is not None and a10[j - 1, k] != "S" and a10[j, k] != "S"
                else:
                    trig = a0[j - 1, k] == "S" and a10[j - 1, k] == "S" and a0[j, k] is not None and a0[j, k] != "S" and a10[j, k] == "S"
                if not trig:
                    continue
                lag = None
                for m in range(j + 1, min(j + 30, len(cal))):
                    x = a10[m, k]
                    if x is None:
                        break
                    if (enter and x == "S") or (not enter and x != "S"):
                        lag = m - j
                        break
                rr = None
                if lag is not None and j + lag + 1 < len(cal) and np.isfinite(Oa[j + 1, k]) and np.isfinite(Oa[j + lag + 1, k]):
                    rr = Oa[j + lag + 1, k] / Oa[j + 1, k] - 1
                rec.append((lag, rr))
        df = pd.DataFrame(rec, columns=["lag", "ret"], dtype=float)
        return {"events": len(df), "never_within_30_pct": r_(100 * df["lag"].isna().mean(), 1),
                "lag_median": r_(df["lag"].median(), 1), "lag_mean": r_(df["lag"].mean(), 1),
                "ret_during_lag_mean_pct": r_(100 * df["ret"].mean(), 2), "ret_during_lag_median_pct": r_(100 * df["ret"].median(), 2)}
    out["exit_lag_2026"] = lags(False)
    out["entry_lag_2026"] = lags(True)
    d = cal[(cal >= pd.Timestamp("2026-01-02")) & (cal <= pd.Timestamp("2026-10-09"))]
    for nm, G in (("s10", G10), ("s0", G0)):
        g, p = G.loc[d], G.shift(1).loc[d]
        both = g.notna() & p.notna()
        out[f"daily_change_pct_{nm}"] = r_(100 * ((g != p) & both).sum().sum() / both.sum().sum(), 1)
    return out


# ============================================================================= G 지난 완료 IC
def part_G(P, E) -> dict:
    sc = FC.make(FC.GRID["E14"], 10)(P)
    pr = H.pct_rank(sc, E & sc.notna())
    out = {"note": "t일 장 마감에 아는 IC = 매도 시가(t+1+h)가 t 이하인 신호(신호일 ≤ t−(h+1))의 최근 20개 평균. "
                   "그 값의 부호로 t일 신호의 IC 부호를 맞히는지."}
    for h in (20, 10):
        fr = H.forward(h).where(E)
        ic = H._row_spearman(pr, fr)
        ex = H.exit_date(h).reindex(ic.index)
        known = ic.shift(h + 1).rolling(20, min_periods=10).mean()
        cal = ic.index
        d = cal[(cal >= pd.Timestamp("2025-03-01")) & ex.notna().to_numpy()]
        x, y = known.loc[d], ic.loc[d]
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        d26 = x.index >= pd.Timestamp("2026-01-01")
        aft = (x.index >= pd.Timestamp("2026-06-01"))
        out[f"h{h}"] = {"days": int(ok.sum()), "corr": r_(np.corrcoef(x, y)[0, 1], 3),
                        "corr_2026": r_(np.corrcoef(x[d26], y[d26])[0, 1], 3),
                        "sign_hit_pct": r_(100 * ((x > 0) == (y > 0)).mean(), 1), "base_pos_pct": r_(100 * (y > 0).mean(), 1),
                        "ic_mean_when_known_neg": r_(y[x < 0].mean(), 3), "ic_mean_when_known_pos": r_(y[x > 0].mean(), 3),
                        "share_known_neg_pct": r_(100 * (x < 0).mean(), 1),
                        "first_known_neg_after_2026_06": str(x[aft & (x < 0)].index.min().date()) if (aft & (x < 0)).any() else None,
                        "first_known_pos_after_2026_08": str(x[(x.index >= pd.Timestamp("2026-08-01")) & (x > 0)].index.min().date())
                        if ((x.index >= pd.Timestamp("2026-08-01")) & (x > 0)).any() else None}
    return out


def main():
    P = H.load_panel()
    E = H.eligible()
    rows, dates = tab_holdings()
    frame = download([r["symbol"] for r in rows] + ["QQQ", "000660.KS"] + list(FX.values()))
    res = {"fund_dates": dates, "computed": pd.Timestamp.now().isoformat(timespec="seconds"),
           "panel_last": str(P["C"].index[-1].date()), "yf_last": str(frame["Close"].index.max())}
    a, inds, cal = part_A(rows, frame)
    res["A_foreign_mix"] = a
    print("A", flush=True)
    res["B_young_listing"] = part_B(P, E)
    print("B", flush=True)
    res["C_membership"] = part_C(P, rows)
    print("C", flush=True)
    res["D_yfinance"] = part_D(P, rows, frame, inds, cal)
    print("D", flush=True)
    res["E_gate_D"] = part_E(P, E)
    print("E", flush=True)
    res["F_smoothing_lag"] = part_F(P, E)
    print("F", flush=True)
    res["G_known_ic"] = part_G(P, E)
    print("G", flush=True)
    res["H_draft_design"] = part_H(rows, frame, inds, cal)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
