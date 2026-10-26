"""버킷 분산 모멘텀 — 미사용 구간(2023-05-22 ~ 2025-12-31, TIME만) 검증 (2026-10-10).

사전 등록 규칙(결과를 본 뒤 바꾸지 않음):
  매주 ISO 주 첫 미국 세션 시가에 결정. 세션 날짜보다 '앞선' 날짜의 TIME PDF와 전 세션 종가까지만 쓴다.
  후보 U='T': TIME 미국 보유(그 PDF에서 비중 > 0, 120세션 가격이 있는 종목).
  버킷 G: 'sector' = desk.strategies.sector_of에 반도체 병합
            {AI 반도체, 반도체(적자)} -> '반도체 칩', {반도체 장비, 반도체 장비(적자), 반도체 소재} -> '반도체 장비·소재'
          'sub'    = desk.strategies.subsector_of(AI_UNIVERSE 소분류, 없으면 섹터)
          두 방식 모두 '미분류' 종목은 종목마다 각자 버킷.
  모멘텀 = 전 세션 수정종가 기준 120세션 수익률. 버킷 대장 = 버킷 안 모멘텀 1위. 대장끼리 모멘텀 순위.
  선택(N개, 버킷당 1개): 지난주 종목 중 '버킷 대장 순위 <= N+5' 이고 '버킷 안 모멘텀 2위 이내'면 유지
          (같은 버킷 두 개면 높은 쪽), 나머지는 대장 순위 위에서부터 이미 든 버킷을 건너뛰며 N개까지 채움.
  비중 W: 'eq' = 같은 비중, 'fund' = TIME NAV 비중 비례.
  상한: 종목 <= max(0.25, 1/N), watchlist 섹터(병합 안 함, 데스크처럼 '미분류'도 한 묶음) <= 60%,
        반도체 예산 없음. 상한에 막혀 못 넣은 비중은 현금(desk.strategies.project_caps 그대로 사용).
  N {3,5,7,10} x G {sector, sub} x W {eq, fund} = 16개. 손절 없음.
기준: 데스크 규칙(TIME 미국, 120일 상위 7, 버퍼 5, TIME 비중, 25/60, 반도체+미분류 50%) 과 #774(같은데 60%).

자료: strategy_lab/holdout (TIME 운용사 원천 주간 PDF 137장, 가격 2022-09-01 ~ 2026-01-15, yfinance).
수익: USD 시가→시가(분할만 반영된 원시 시가, 데스크 run_model과 같음, 배당 미반영). 수수료 0.
원화 환산: KRW=X로 따로 환산해 TIME ETF(원화)와의 배수를 낸다.
엔진 확인: 같은 코드로 2026 데스크 가격 사본(rsi_breakout/cache/deskx_prices)과 정본 TIME 스냅샷을 돌려
          #774 +222.1%·데스크 규칙 +223.7%가 재현되는지 본다(deskx base와 비교).

실행: py data/derived/analysis/rsi_breakout/code/bucket_holdout.py
결과: data/derived/analysis/rsi_breakout/results/bucket/holdout.json
"""
from __future__ import annotations

import glob
import json
from bisect import bisect_left, bisect_right
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))
sys.path.insert(0, ROOT)
from desk import config  # noqa: E402
from desk.strategies import (UNKNOWN, iso_week, latest_before, project_caps, sector_of,  # noqa: E402
                             subsector_of, us_weights)
from gbrc.etf.market import yahoo_ticker  # noqa: E402

HOLD = os.path.join(ROOT, "data", "derived", "analysis", "strategy_lab", "holdout")
FROZEN_2026 = os.path.join(ROOT, "data", "derived", "analysis", "rsi_breakout", "cache", "deskx_prices")
OUT = os.path.join(ROOT, "data", "derived", "analysis", "rsi_breakout", "results", "bucket", "holdout.json")

START, END = "2023-05-22", "2026-01-02"   # 값 1.0 = 5/22 시가, 끝 = 2026-01-02 시가(2025-12-31 종가 다음)
YEARS = (("2023(5/22~)", "2023-05-22", "2024-01-02"), ("2024", "2024-01-02", "2025-01-02"),
         ("2025", "2025-01-02", "2026-01-02"))
STRESS = (("23년 가을 조정 7/31~10/27", "2023-07-31", "2023-10-27"),
          ("24년 7~8월 급락 7/10~8/7", "2024-07-10", "2024-08-07"),
          ("25년 관세 폭락 2/19~4/8", "2025-02-19", "2025-04-08"),
          ("25년 반등 4/8~6/30", "2025-04-08", "2025-06-30"))
LOOKBACK = 120
MERGE = {"AI 반도체": "반도체 칩", "반도체(적자)": "반도체 칩",
         "반도체 장비": "반도체 장비·소재", "반도체 장비(적자)": "반도체 장비·소재", "반도체 소재": "반도체 장비·소재"}
NO_SEMIS_BUDGET = 10.0


# ---------------------------------------------------------------------------- data
class Book:
    """TIME 스냅샷 + 가격(QQQ 거래일 달력)."""

    def __init__(self, snaps: dict, prices: dict):
        adj = prices["Adj Close"]
        self.days = [d.strftime("%Y-%m-%d") for d in adj["QQQ"].dropna().index]
        idx = pd.DatetimeIndex(pd.to_datetime(self.days))

        def on_days(df):
            return df.reindex(idx)

        self.adj_df, self.opn_df, self.cls_df = (on_days(prices[f]) for f in ("Adj Close", "Open", "Close"))
        self.fx = on_days(adj)["KRW=X"].ffill().to_numpy(float)
        self.pdfs = sorted(snaps)
        self.w = {p: us_weights(snaps[p]) for p in self.pdfs}
        keys = sorted({k for w in self.w.values() for k in w})
        self.ticker = {k: yahoo_ticker(k) for k in keys}
        self.keys = keys
        self.sector = {k: sector_of(k, self.ticker[k]) for k in keys}
        self.sub = {k: subsector_of(k, self.ticker[k]) for k in keys}
        self._cache = {}

    def col(self, frame: str, key: str):
        ck = (frame, key)
        if ck not in self._cache:
            df = {"adj": self.adj_df, "opn": self.opn_df, "cls": self.cls_df}[frame]
            tk = self.ticker.get(key)
            self._cache[ck] = df[tk].to_numpy(float) if tk in df.columns else None
        return self._cache[ck]

    def bucket(self, key: str, G: str) -> str:
        lab = self.sector[key] if G == "sector" else self.sub[key]
        if lab == UNKNOWN:
            return f"{UNKNOWN}:{key}"
        return MERGE.get(lab, lab) if G == "sector" else lab


def load_holdout() -> Book:
    snaps = {}
    for p in sorted(glob.glob(os.path.join(HOLD, "time_weekly", "*.json"))):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        snaps[s["actual_date"]] = s
    prices = {f: pd.read_csv(os.path.join(HOLD, f"prices_{f.replace(' ', '_').lower()}.csv"),
                             index_col=0, parse_dates=True) for f in ("Adj Close", "Open", "Close")}
    return Book(snaps, prices)


def load_2026() -> Book:
    """엔진 확인용: 데스크 가격 사본(연구 폴더) + 정본 TIME 스냅샷. 데스크 캐시는 읽지도 쓰지도 않는다."""
    from gbrc.etf.trades import load_fund_snapshots
    snaps = load_fund_snapshots("456600", "9999-12-31")
    prices = {f: pd.read_csv(os.path.join(FROZEN_2026, f"prices_{f.replace(' ', '_').lower()}.csv"),
                             index_col=0, parse_dates=True) for f in ("Adj Close", "Open", "Close")}
    return Book(snaps, prices)


# ---------------------------------------------------------------------------- rules
def caps(raw: dict, sector: dict, stock: float, sect: float, semis: float) -> dict:
    saved = dict(config.RULE)
    try:
        config.RULE.update(stock_cap=stock, sector_cap=sect, semis_cap=semis)
        return project_caps(raw, sector)
    finally:
        config.RULE.clear()
        config.RULE.update(saved)


def ranking(book: Book, k: int, session: str, same_day_pdf: bool = False):
    """desk.strategies.momentum_ranking과 같은 순위(키 정렬 후 모멘텀 내림차순, 안정 정렬)."""
    if same_day_pdf:
        i = bisect_right(book.pdfs, session) - 1
        pdf = book.pdfs[i] if i >= 0 else None
    else:
        pdf = latest_before(book.pdfs, session)
    if pdf is None or k - 1 - LOOKBACK < 0:
        return pdf, []
    w = book.w[pdf]
    rows = []
    for key in sorted(w):
        if w[key] <= 0:
            continue
        px = book.col("adj", key)
        if px is None:
            continue
        p1, p0 = px[k - 1], px[k - 1 - LOOKBACK]
        if not (np.isfinite(p1) and np.isfinite(p0) and p0 > 0):
            continue
        rows.append({"key": key, "mom": p1 / p0 - 1, "w": w[key]})
    rows.sort(key=lambda r: -r["mom"])
    return pdf, rows


def desk_picker(semis_cap: float, n: int = 7, buffer: int = 5, same_day_pdf=False, sector_override=None):
    def pick(book: Book, k: int, session: str, current: list):
        pdf, ranked = ranking(book, k, session, same_day_pdf)
        if not ranked:
            return None
        order = [r["key"] for r in ranked]
        keep = [x for x in order[:n + buffer] if x in current][:n]
        for x in order:
            if len(keep) >= n:
                break
            if x not in keep:
                keep.append(x)
        tw = {r["key"]: r["w"] for r in ranked}
        tot = sum(tw[x] for x in keep)
        raw = {x: tw[x] / tot for x in keep}
        sec = sector_override(book) if sector_override else book.sector
        return {"pdf": pdf, "selected": keep, "raw": raw, "target": caps(raw, sec, 0.25, 0.60, semis_cap),
                "n_buckets": None}
    return pick


def bucket_picker(G: str, N: int, W: str, same_day_pdf=False, sector_override=None):
    stock_cap = max(0.25, 1.0 / N)

    def pick(book: Book, k: int, session: str, current: list):
        pdf, ranked = ranking(book, k, session, same_day_pdf)
        if not ranked:
            return None
        members: dict[str, list] = {}
        for r in ranked:                                   # ranked는 모멘텀 내림차순
            members.setdefault(book.bucket(r["key"], G), []).append(r)
        leaders = sorted(members, key=lambda b: (-members[b][0]["mom"], ranked.index(members[b][0])))
        lrank = {b: i + 1 for i, b in enumerate(leaders)}
        inrank = {r["key"]: i + 1 for b in members for i, r in enumerate(members[b])}
        mom = {r["key"]: r["mom"] for r in ranked}
        kept: dict[str, str] = {}
        for key in current:
            if key not in mom:
                continue
            b = book.bucket(key, G)
            if lrank[b] <= N + 5 and inrank[key] <= 2:
                if b not in kept or mom[key] > mom[kept[b]]:
                    kept[b] = key
        sel = [kept[b] for b in sorted(kept, key=lambda b: lrank[b])][:N]
        held = {book.bucket(x, G) for x in sel}
        for b in leaders:
            if len(sel) >= N:
                break
            if b not in held:
                sel.append(members[b][0]["key"])
                held.add(b)
        if W == "eq":
            raw = {x: 1.0 / len(sel) for x in sel}
        else:
            tw = {r["key"]: r["w"] for r in ranked}
            tot = sum(tw[x] for x in sel)
            raw = {x: tw[x] / tot for x in sel}
        sec = sector_override(book) if sector_override else book.sector
        return {"pdf": pdf, "selected": sel, "raw": raw,
                "target": caps(raw, sec, stock_cap, 0.60, NO_SEMIS_BUDGET), "n_buckets": len(leaders)}
    return pick


def unc_own_group(book: Book):
    """보조 민감도: 섹터 상한에서 '미분류'를 종목마다 각자 묶음으로 본다(사전 등록 규칙 아님)."""
    return {k: (f"{UNKNOWN}:{k}" if v == UNKNOWN else v) for k, v in book.sector.items()}


# ---------------------------------------------------------------------------- path
def run(book: Book, pick, start: str, end: str):
    """desk.strategies.run_model과 같은 시가→시가 경로(원시 시가, 드리프트, 남는 비중은 현금)."""
    k0 = next(i for i, d in enumerate(book.days) if d >= start)
    k1 = max(i for i, d in enumerate(book.days) if d <= end)
    value, rows, decisions = [1.0], [], []
    w: dict[str, float] = {}
    current: list[str] = []
    started = False
    for k in range(k0, k1 + 1):
        day = book.days[k]
        if k > k0:
            rets = {}
            for key in w:
                o = book.col("opn", key)
                ok = o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0
                rets[key] = o[k] / o[k - 1] - 1 if ok else 0.0
            gross = sum(wt * rets[key] for key, wt in w.items())
            total = sum(wt * (1 + rets[key]) for key, wt in w.items()) + (1 - sum(w.values()))
            pre = {key: wt * (1 + rets[key]) / total for key, wt in w.items()}
            value.append(value[-1] * (1 + gross))
        else:
            rets, pre = {}, {}
        reb = (not started) or iso_week(day) != iso_week(book.days[k - 1])
        traded = 0.0
        w = pre
        if reb:
            dec = pick(book, k, day, current)
            if dec is not None:
                started = True
                tgt = dec["target"]
                traded = 0.5 * sum(abs(tgt.get(x, 0.0) - w.get(x, 0.0)) for x in set(tgt) | set(w))
                w, current = dict(tgt), list(dec["selected"])
                decisions.append({"day": day, "pdf": dec["pdf"], "selected": list(dec["selected"]),
                                  "invested": sum(tgt.values()), "n_buckets": dec["n_buckets"],
                                  "raw": dict(dec["raw"]), "target": dict(tgt)})
        rows.append({"day": day, "w": dict(w), "rets": rets, "traded": traded})
    return pd.Series(value, index=book.days[k0:k1 + 1]), rows, decisions


# ---------------------------------------------------------------------------- stats
def seg(v: pd.Series, a: str, b: str) -> float:
    """시가 a(이상 첫 세션)부터 시가 b(이하 마지막 세션)까지 수익."""
    x = v[(v.index >= a) & (v.index <= b)]
    return float(x.iloc[-1] / x.iloc[0] - 1)


def mdd(v: pd.Series) -> float:
    a = v.to_numpy()
    return float((a / np.maximum.accumulate(a) - 1).min())


def fx_before(book: Book, day: str) -> float:
    i = bisect_left(book.days, day) - 1
    return float(book.fx[i])


def evaluate(book: Book, name: str, v: pd.Series, rows, decisions, G: str, etf: dict):
    years = (date.fromisoformat(v.index[-1]) - date.fromisoformat(v.index[0])).days / 365.25
    tot = float(v.iloc[-1] - 1)
    fxr = fx_before(book, v.index[-1]) / fx_before(book, v.index[0])
    krw = (1 + tot) * fxr - 1
    live = [r for r in rows if r["day"] >= decisions[0]["day"]]
    # 기여: 시작 가치 1 기준 금액(deskx_caps.contributions와 같은 정의), 합 = 전체 이익
    vals = v.to_numpy()
    contrib: dict[str, float] = {}
    for i in range(1, len(rows)):
        for key, wt in rows[i - 1]["w"].items():
            tk = book.ticker[key]
            contrib[tk] = contrib.get(tk, 0.0) + vals[i - 1] * wt * rows[i]["rets"].get(key, 0.0)
    assert abs(sum(contrib.values()) - tot) < 1e-8 * max(1.0, abs(tot)), (sum(contrib.values()), tot)
    top = sorted(contrib.items(), key=lambda x: -x[1])
    semis = [sum(wt for key, wt in r["w"].items() if book.sector[key] in config.SEMIS_SECTORS) for r in live]
    unc = [sum(wt for key, wt in r["w"].items() if book.sector[key] == UNKNOWN) for r in live]
    nb = [len({book.bucket(key, G) for key in r["w"]}) for r in live]
    nn = [len(r["w"]) for r in live]
    inv = [sum(r["w"].values()) for r in live]
    turn_ex_first = sum(r["traded"] for r in rows if r["day"] > decisions[0]["day"])
    reb_weeks = len(decisions)
    names = sorted({book.ticker[k] for r in live for k in r["w"]})
    out = {
        "name": name, "full": tot, "cagr": (1 + tot) ** (1 / years) - 1, "mdd": mdd(v),
        "full_krw": krw, "ratio_etf_krw": krw / etf["krw"], "ratio_etf_usd": tot / etf["usd"],
        "years": {lab: seg(v, a, b) for lab, a, b in YEARS},
        "stress": {lab: seg(v, a, b) for lab, a, b in STRESS},
        "buckets_avg": float(np.mean(nb)), "names_avg": float(np.mean(nn)),
        "semis_avg": float(np.mean(semis)), "unclassified_avg": float(np.mean(unc)),
        "semis_plus_unc_avg": float(np.mean(np.add(semis, unc))), "invested_avg": float(np.mean(inv)),
        "cash_weeks_share": float(np.mean([d["invested"] < 0.99 for d in decisions])),
        "top3": [t for t, _ in top[:3]], "top3_share": sum(c for _, c in top[:3]) / tot if tot else float("nan"),
        "top5_contrib": {t: round(c, 4) for t, c in top[:5]},
        "bottom3_contrib": {t: round(c, 4) for t, c in top[-3:]},
        "turnover_annual": turn_ex_first / years, "turnover_per_rebalance": turn_ex_first / max(1, reb_weeks - 1),
        "rebalances": reb_weeks, "unique_names": len(names),
        "first_decision": decisions[0]["day"], "first_pdf": decisions[0]["pdf"],
        "last_holdings": {book.ticker[k]: round(wt, 3) for k, wt in rows[-1]["w"].items()},
    }
    def share(d, which, test):
        return sum(wt for key, wt in d[which].items() if test(book.sector[key]))

    su = lambda sec: sec in config.SEMIS_SECTORS or sec == UNKNOWN  # noqa: E731
    un = lambda sec: sec == UNKNOWN  # noqa: E731
    out["caps_at_rebalance"] = {
        "any_cap_binds_share": float(np.mean([max(abs(d["target"].get(x, 0.0) - d["raw"][x]) for x in d["raw"]) > 1e-6
                                              for d in decisions])),
        "semis_unc_raw_avg": float(np.mean([share(d, "raw", su) for d in decisions])),
        "semis_unc_target_avg": float(np.mean([share(d, "target", su) for d in decisions])),
        "semis_unc_budget_binds_share": float(np.mean([share(d, "raw", su) - share(d, "target", su) > 1e-6
                                                       for d in decisions])),
        "unc_raw_over_60_share": float(np.mean([share(d, "raw", un) > 0.60 + 1e-9 for d in decisions])),
        "unc_raw_avg": float(np.mean([share(d, "raw", un) for d in decisions])),
        "unc_selected_avg": float(np.mean([sum(book.sector[x] == UNKNOWN for x in d["selected"]) for d in decisions])),
    }
    if decisions[0]["n_buckets"] is not None:
        nbk = [d["n_buckets"] for d in decisions]
        out["available_buckets"] = {"min": int(min(nbk)), "avg": float(np.mean(nbk)), "max": int(max(nbk))}
    return out


def etf_levels(book: Book):
    s = book.adj_df["456600.KS"].dropna()
    s.index = s.index.strftime("%Y-%m-%d")

    def lvl(d):
        x = s[s.index <= d]
        return float(x.iloc[-1]), x.index[-1]

    return lvl


def benchmarks(book: Book):
    lvl = etf_levels(book)
    a, da = lvl("2023-05-23")
    b, db = lvl(END)
    etf_krw = b / a - 1
    fxr = fx_before(book, db) / fx_before(book, da)
    etf_usd = (1 + etf_krw) / fxr - 1
    s = book.adj_df["456600.KS"].dropna()
    s = s[(s.index >= "2023-05-23") & (s.index <= END)]
    etf = {"krw": etf_krw, "usd": etf_usd, "mdd_krw": float((s / s.cummax() - 1).min()),
           "dates": [da, db],
           "years_krw": {lab: lvl(y1)[0] / lvl("2023-05-23" if lab.startswith("2023") else y0)[0] - 1
                         for lab, y0, y1 in YEARS}}
    # QQQ: 배당 반영 시가(시가 x 수정종가/종가)로 같은 시가→시가 구간
    q = (book.opn_df["QQQ"] * book.adj_df["QQQ"] / book.cls_df["QQQ"])
    q.index = q.index.strftime("%Y-%m-%d")
    qv = q[(q.index >= START) & (q.index <= END)]
    qqq = {"usd": float(qv.iloc[-1] / qv.iloc[0] - 1), "mdd_usd_open": mdd(qv / qv.iloc[0]),
           "krw": float((qv.iloc[-1] / qv.iloc[0]) * fx_before(book, qv.index[-1]) / fx_before(book, qv.index[0]) - 1),
           "years_usd": {lab: seg(qv, a0, b0) for lab, a0, b0 in YEARS}}
    return etf, qqq


# ---------------------------------------------------------------------------- 2026 engine check
def engine_check():
    book = load_2026()
    out = {}
    for lab, cap in (("#774 (semis 60%)", 0.60), ("desk rule (semis 50%)", 0.50)):
        v, rows, dec = run(book, desk_picker(cap), "2026-01-02", "2026-10-09")
        r = (v.shift(-1) / v - 1).dropna()
        out[lab] = {"front_1/2~5/29": float((1 + r[(r.index >= "2026-01-02") & (r.index <= "2026-05-29")]).prod() - 1),
                    "back_6/1~10/9": float((1 + r[(r.index >= "2026-06-01") & (r.index <= "2026-10-09")]).prod() - 1),
                    "full": float((1 + r).prod() - 1), "mdd": mdd(v), "last_day": v.index[-1]}
    ref = os.path.join(ROOT, "data", "derived", "analysis", "rsi_breakout", "results", "deskx", "universe.json")
    if os.path.exists(ref):
        with open(ref, encoding="utf-8") as f:
            base = json.load(f)["base"]
        out["deskx_base_#774_full"] = base["전체"]["수익"]
        out["abs_diff_#774_full"] = abs(out["#774 (semis 60%)"]["full"] - base["전체"]["수익"])
    return out


# ---------------------------------------------------------------------------- main
def main():
    check = engine_check()
    print("엔진 확인(2026):", json.dumps(check, ensure_ascii=False))
    book = load_holdout()
    etf, qqq = benchmarks(book)
    unc = sorted(book.ticker[k] for k in book.keys if book.sector[k] == UNKNOWN)
    label_stats = {"companies": len(book.keys), "unclassified_sector": len(unc),
                   "unclassified_sub": sum(book.sub[k] == UNKNOWN for k in book.keys),
                   "unclassified_tickers": unc,
                   "known_semis_names": sorted(book.ticker[k] for k in book.keys if book.sector[k] in config.SEMIS_SECTORS),
                   "pdfs": [book.pdfs[0], book.pdfs[-1], len(book.pdfs)]}
    print(f"TIME ETF KRW {etf['krw']:+.4f} USD {etf['usd']:+.4f} MDD {etf['mdd_krw']:.4f} | QQQ USD {qqq['usd']:+.4f}")
    print(f"종목 {label_stats['companies']}, 섹터 미분류 {label_stats['unclassified_sector']}, 소분류 미분류 {label_stats['unclassified_sub']}")

    specs = [("데스크 규칙 (반도체+미분류 50%)", "desk", desk_picker(0.50), "sector"),
             ("#774 (반도체+미분류 60%)", "desk", desk_picker(0.60), "sector")]
    for G in ("sector", "sub"):
        for N in (3, 5, 7, 10):
            for W in ("eq", "fund"):
                specs.append((f"bucket {G} N{N} {W}", "bucket", bucket_picker(G, N, W), G))
    results = []
    for name, fam, pick, G in specs:
        v, rows, dec = run(book, pick, START, END)
        res = evaluate(book, name, v, rows, dec, G, etf)
        res["family"] = fam
        if fam == "bucket":       # 버킷 변형에는 반도체 예산이 없다(semis+unc 변화는 종목 상한 때문)
            res["caps_at_rebalance"]["semis_unc_budget_binds_share"] = None
        res["sub_buckets_avg"] = float(np.mean([len({book.bucket(k, "sub") for k in r["w"]})
                                                for r in rows if r["day"] >= dec[0]["day"]]))
        results.append(res)
        y = res["years"]
        print(f"{name:34s} {res['full']:+8.3f} CAGR {res['cagr']:+.3f} MDD {res['mdd']:+.3f} "
              f"xETF(KRW) {res['ratio_etf_krw']:.2f} | {y['2023(5/22~)']:+.3f} {y['2024']:+.3f} {y['2025']:+.3f} | "
              f"bk {res['buckets_avg']:.1f} semi {res['semis_avg']:.2f} unc {res['unclassified_avg']:.2f} "
              f"inv {res['invested_avg']:.2f} top3 {res['top3_share']:.2f} {','.join(res['top3'])} "
              f"turn/yr {res['turnover_annual']:.1f}")
        c = res["caps_at_rebalance"]
        print(f"{'':34s} caps bind {c['any_cap_binds_share']:.2f} | semis+unc raw {c['semis_unc_raw_avg']:.2f} -> "
              f"{c['semis_unc_target_avg']:.2f} (budget binds {c['semis_unc_budget_binds_share'] if c['semis_unc_budget_binds_share'] is None else round(c['semis_unc_budget_binds_share'], 2)}) | "
              f"unc raw {c['unc_raw_avg']:.2f}, >60% {c['unc_raw_over_60_share']:.2f}, unc names {c['unc_selected_avg']:.1f}")

    # 보조(사전 등록 아님, 선택에 쓰지 않음): ① PDF 당일 사용(<=) ② 미분류를 종목별 섹터로
    #   (project_caps는 sector == '미분류'만 반도체 예산에 넣으므로 ②에서는 미분류가 60% 섹터 한 묶음에서도,
    #    데스크 기준의 반도체+미분류 예산에서도 빠진다. 버킷 변형은 반도체 예산이 없어 60% 묶음만 바뀐다.)
    aux = {"pdf_same_day": {}, "unclassified_own_cap_group": {}}
    for name, fam, _, G in specs:
        if fam == "desk":
            cap = 0.50 if "50%" in name else 0.60
            p1, p2 = desk_picker(cap, same_day_pdf=True), desk_picker(cap, sector_override=unc_own_group)
        else:
            _, g, n, w = name.split()
            p1 = bucket_picker(g, int(n[1:]), w, same_day_pdf=True)
            p2 = bucket_picker(g, int(n[1:]), w, sector_override=unc_own_group)
        for tag, p in (("pdf_same_day", p1), ("unclassified_own_cap_group", p2)):
            v, rows, dec = run(book, p, START, END)
            aux[tag][name] = {"full": float(v.iloc[-1] - 1), "mdd": mdd(v),
                              "years": {lab: seg(v, a, b) for lab, a, b in YEARS},
                              "invested_avg": float(np.mean([sum(r["w"].values()) for r in rows if r["day"] >= dec[0]["day"]]))}
    for tag, d in aux.items():
        print(f"\n보조 {tag}")
        for name, r in d.items():
            print(f"  {name:34s} {r['full']:+8.3f} MDD {r['mdd']:+.3f} inv {r['invested_avg']:.2f}")

    desk = results[0]
    out = {
        "meta": {
            "strategy": "버킷 분산 모멘텀 (사전 등록, 결과 본 뒤 변경 없음)",
            "period": {"start_open": START, "end_open": END, "note": "값 1.0 = 2023-05-22 시가(현금), 첫 결정 2023-05-23 시가"},
            "timing": "결정: ISO 주 첫 미국 세션 시가, 세션 날짜보다 앞선 TIME PDF(주간 PDF라 보통 지난주 월요일 PDF) + 전 세션 수정종가",
            "returns": "USD 시가→시가(원시 시가=분할 반영·배당 미반영, 데스크 run_model과 같음), 수수료 0. KRW는 KRW=X(경계 세션 전날 값)로 환산",
            "mdd": "USD 시가 경로",
            "turnover": "편도 회전율 합(첫 매수 제외)/년. turnover_per_rebalance = 재조정 1회 평균",
            "semis_avg": "알려진 반도체 섹터(desk config.SEMIS_SECTORS) 평균 비중, 미분류 제외. semis_plus_unc_avg = 데스크 예산 정의",
            "top3_share": "시작 가치 1 기준 금액 기여 상위 3 / 전체 이익 (deskx_caps.contributions 정의)",
            "grid": "16 버킷 변형(G 2 x N 4 x W 2) + 기준 2(데스크 50%, #774 60%). 보조 민감도 2종 x 18 = 36(선택에 안 씀)",
            "labels": "desk.strategies.sector_of / subsector_of (desk 키: research_key, us_weights)",
        },
        "engine_check_2026": check,
        "label_stats": label_stats,
        "benchmarks": {"TIME_ETF": etf, "QQQ": qqq},
        "results": results,
        "aux": aux,
        "vs_desk": {
            "higher_total_lower_mdd": [r["name"] for r in results[2:] if r["full"] > desk["full"] and r["mdd"] > desk["mdd"]],
            "beats_desk_2024_and_2025": [r["name"] for r in results[2:]
                                         if r["years"]["2024"] > desk["years"]["2024"] and r["years"]["2025"] > desk["years"]["2025"]],
            "beats_etf_krw": [r["name"] for r in results if r["full_krw"] > etf["krw"]],
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n저장", OUT)


if __name__ == "__main__":
    main()
