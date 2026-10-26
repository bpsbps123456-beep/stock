"""적대적 검증 #2: gate_main.py의 2023-05-22~2025-12-31 보조 구간과 시점 감사 (2026-10-10, 연구용).

읽기만 한다. 쓰는 곳: results/gate/verify_holdout_timing.json 하나.
자료·종목 고르기만 verify_bucket_holdout(Data, desk_rule, bucket_rule, schedule)을 쓰고, 경로는 새로 짠다:
  주식 수 장부(시가로 사고 팔고 시가로 평가, 결측 시가는 직전 시가 유지), 자체 SMA(QQQ는 holdout prices_close.csv,
  SOXX는 stop_research ohlc.parquet 종가), 자체 연도·낙폭 계산.
  1. live50·c774·nocap·sub5eq × none·soxx_w·soxx_d·qqq_w × lag1·strict → gate_main main.json과 비교(허용 0.5%p)
  2. qqq_w lag1 vs strict 분해: none 경로 차 + 게이트 차, 혼합(고르기 시점 / 게이트 시점 교차), 결정 요일 0~4, same,
     주별 기여 상위 주
  3. PDF 날짜 의미: 평가액/수량/종가로 본 내재 환율 분산이 가장 작은 미국 종가일 = 평가 기준일.
     2026 복제: 일별 PDF(데스크) vs 주 1개 PDF로 줄인 strict·same·lag1 → 어느 모드가 데스크에 가까운가
  4. d5ce794 이전 watchlist(56종목 미분류)를 하위 프로세스로 돌려 보조 구간·2026 데스크 결과 차이
  5. C3: 보조 구간에서 결정 요일별 soxx_w - none
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/verify_gate_holdout.py
"""
import os
import sys

OLDWL = os.environ.get("GATE_OLD_WATCHLIST")
if OLDWL:                                   # d5ce794 이전 watchlist를 먼저 읽어 sys.modules에 고정
    sys.path.insert(0, OLDWL)
    import watchlist  # noqa: F401,E402
    assert os.path.dirname(os.path.abspath(watchlist.__file__)) == os.path.abspath(OLDWL), watchlist.__file__

import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
from bisect import bisect_left, bisect_right  # noqa: E402
from datetime import date  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_bucket_holdout as V  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import UNKNOWN, in_semis_budget, research_key  # noqa: E402

ROOT = V.ROOT
PARQ = os.path.join(ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet")
MAIN_JSON = os.path.join(HERE, "../results/gate/main.json")
OUT = os.path.join(HERE, "../results/gate/verify_holdout_timing.json")
SP = os.environ.get("GATE_SCRATCH", "")
NEW56 = ["AMKR", "AMBA", "MBLY", "AI", "ADBE", "PATH", "WDAY", "SOUN", "TWLO", "BBAI", "RELX", "CXM", "INOD",
         "SMCI", "P", "CORZ", "CEG", "VST", "CCJ", "LEU", "BWXT", "ETN", "POWL", "GNRC", "FLNC", "EOSE", "NXT", "RUN",
         "KMI", "LNG", "EQT", "BKR", "LMND", "SE", "CPNG", "APP", "RDDT", "PINS", "PERI", "NRDY", "PL", "SERV",
         "QBTS", "QUBT", "SDGR", "RXRX", "ISRG", "BIDU", "MP", "USAR", "CRML", "UAMY", "RBLX", "GETY", "SSTK", "BZFD"]
assert len(NEW56) == 56 and len(set(NEW56)) == 56
GSYM = {"soxx_w": "SOXX", "soxx_d": "SOXX", "qqq_w": "QQQ"}


# ----------------------------------------------------------------------------- 신호(자체)
class Sig:
    def __init__(self, D):
        q = D.raw["close"]["QQQ"].dropna()
        q.index = pd.to_datetime(q.index).strftime("%Y-%m-%d")
        o = pd.read_parquet(PARQ, columns=["date", "sym", "Close"])
        o = o[o.sym == "SOXX"]
        s = pd.Series(o["Close"].astype(float).to_numpy(), index=pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d"))
        s = s[~s.index.duplicated()].sort_index()
        self.src = {"QQQ": q.sort_index(), "SOXX": s}
        self.cache = {}

    def below(self, sym, day, n=20):
        if (sym, n) not in self.cache:
            c = self.src[sym].to_numpy(float)
            sma = np.full(len(c), np.nan)
            cs = np.cumsum(np.insert(c, 0, 0.0))
            sma[n - 1:] = (cs[n:] - cs[:-n]) / n
            self.cache[sym, n] = (list(self.src[sym].index), c, sma)
        idx, c, sma = self.cache[sym, n]
        i = bisect_right(idx, day) - 1
        if i < 0 or not np.isfinite(sma[i]):
            return False
        return bool(c[i] < sma[i])

    def detail(self, sym, day, n=20):
        self.below(sym, day, n)
        idx, c, sma = self.cache[sym, n]
        i = bisect_right(idx, day) - 1
        return idx[i], round(float(c[i]), 2), round(float(sma[i]), 2)


# ----------------------------------------------------------------------------- 일정
def wd_schedule(D, wd, k0, k1, same=False):
    """ISO 주마다 요일 >= wd인 첫 세션, 그 세션보다 앞선(same이면 같은 날 포함) 최신 PDF."""
    out, done = {}, set()
    for k in range(k0, k1 + 1):
        day = D.days[k]
        wk = V.iso(day)
        if wk in done or date.fromisoformat(day).weekday() < wd:
            continue
        i = (bisect_right if same else bisect_left)(D.pdfs, day) - 1
        if i < 0:
            continue
        out[k] = D.pdfs[i]
        done.add(wk)
    return out


# ----------------------------------------------------------------------------- 고르기(재사용) + 경로(자체)
def picks(D, rule, sched):
    """고르기는 게이트와 무관(gate_main도 cur를 게이트와 상관없이 갱신)하므로 일정마다 한 번만 센다."""
    cur, out = [], {}
    for k in sorted(sched):
        res = rule(D, k, sched[k], cur, ())
        if res is not None:
            cur = list(res[0])
            out[k] = (list(res[0]), dict(res[1]))
    return out


def gated(base, g, on, semis):
    if not on or g == "none":
        return dict(base)
    if g == "qqq_w":
        return {}
    return {x: v for x, v in base.items() if not semis(x)}


def path(D, pk, g, sig, k0, k1, gate_days=None, sector=None):
    """주식 수 장부. pk: {k: (sel, tgt)}. gate_days: 주간 게이트를 다시 볼 세션(없으면 고르는 날과 같음)."""
    sector = sector or D.sector
    semis = lambda x: in_semis_budget(sector.get(x, UNKNOWN))
    sym = GSYM.get(g)
    units, cash, base, on = {}, 1.0, {}, False
    last = {}
    vals, ons, inv = [], [], []
    nan_hold = 0
    for k in range(k0, k1 + 1):
        px = {}
        for x in list(units):
            o = D.O[k, D.j[x]]
            if np.isfinite(o) and o > 0:
                last[x] = o
            else:
                nan_hold += 1
            px[x] = last[x]
        value = cash + sum(units[x] * px[x] for x in units)
        sk = sym is not None and sig.below(sym, D.days[k - 1], 20)
        redo = False
        if k in pk:
            base = pk[k][1]
            if gate_days is None:
                on = sk and g != "none"
            redo = True
        if gate_days is not None and k in gate_days and base and g != "none" and sk != on:
            on = sk
            redo = True
        if redo:
            tgt = gated(base, g, on, semis)
            units, cash = {}, value
            for x, wt in tgt.items():
                o = D.O[k, D.j[x]]
                assert np.isfinite(o) and o > 0, (D.days[k], D.tk[x])
                last[x] = o
                units[x] = value * wt / o
                cash -= value * wt
        elif g == "soxx_d" and base and sk != on:
            if sk:
                for x in [x for x in units if semis(x)]:
                    cash += units.pop(x) * px[x]
            else:
                add = {x: v for x, v in base.items() if semis(x)}
                room = cash / value
                need = sum(add.values())
                sc = min(1.0, room / need) if need > 0 else 0.0
                for x, v in add.items():
                    o = D.O[k, D.j[x]]
                    assert np.isfinite(o) and o > 0, (D.days[k], D.tk[x])
                    last[x] = o
                    units[x] = units.get(x, 0.0) + value * v * sc / o
                    cash -= value * v * sc
            on = sk
        vals.append(value)
        ons.append(on)
        inv.append(1 - cash / value)
    v = pd.Series(np.array(vals) / vals[0], index=D.days[k0:k1 + 1])
    return v, np.array(ons), np.array(inv), nan_hold


def stats(v):
    a = v.to_numpy()
    dd = a / np.maximum.accumulate(a) - 1
    yrs = {}
    for y, s, e in V.YEARS:
        x = v[(v.index >= s) & (v.index <= e)]
        yrs[y] = float(x.iloc[-1] / x.iloc[0] - 1)
    return {"전체": float(a[-1] / a[0] - 1), "낙폭": float(dd.min()), "낙폭저점": v.index[int(dd.argmin())], "연도": yrs}


RULES = {"live50": lambda: V.desk_rule(semis=0.50), "c774": lambda: V.desk_rule(semis=0.60),
         "nocap": lambda: V.desk_rule(semis=10.0), "sub5eq": lambda: V.bucket_rule("sub", 5, "eq")}


def run_holdout(D, sig, bases, gates, modes):
    k0 = next(i for i, d in enumerate(D.days) if d >= V.START)
    k1 = max(i for i, d in enumerate(D.days) if d <= V.END)
    scheds = {}
    for m in modes:
        if m in ("lag1", "strict", "same"):
            scheds[m] = V.schedule(D, m, k0, k1)
        elif m.startswith("wd"):
            scheds[m] = wd_schedule(D, int(m[2]), k0, k1)
    out, P, paths = {}, {}, {}
    for b in bases:
        rule = RULES[b]()
        for m in modes:
            P[b, m] = picks(D, rule, scheds[m])
            for g in gates:
                v, ons, inv, nh = path(D, P[b, m], g, sig, k0, k1)
                s = stats(v)
                s.update({"게이트일비율": float(ons.mean()), "현금평균": float(1 - inv.mean()), "결측시가보유일": nh})
                out[f"{b}·{g}·{m}"] = s
                paths[b, g, m] = (v, ons, inv)
    return out, P, paths, scheds, (k0, k1)


# ----------------------------------------------------------------------------- 3. PDF 평가 기준일
def pdf_valuation_lag(D):
    close = D.raw["close"].copy()
    close.index = pd.to_datetime(close.index).strftime("%Y-%m-%d")
    close = close[~close.index.duplicated()]
    cidx = set(close.index)
    fx = D.raw["adj_close"]["KRW=X"].copy()
    fx.index = pd.to_datetime(fx.index).strftime("%Y-%m-%d")
    fx = fx[~fx.index.duplicated()]
    res, picks_ = {}, []
    files = sorted(glob.glob(os.path.join(V.HOLD, "time_weekly", "*.json")))
    for p in files:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        d = s["actual_date"]
        rows = []
        for h in s["holdings"]:
            code = V.ISIN.get(h.get("code"), h.get("code"))
            key = research_key(code)
            if key not in D.tk or not h.get("qty") or not h.get("value_krw"):
                continue
            rows.append((D.tk[key], float(h["value_krw"]) / float(h["qty"])))
        # 후보: PDF 날짜 포함 그 전 미국 세션 5개
        i = bisect_right(D.days, d) - 1
        best = None
        cand = {}
        for off in range(0, 5):
            day = D.days[i - off]
            if day not in cidx:
                continue
            lf = []
            for tk, per in rows:
                c = float(close.at[day, tk]) if tk in close.columns else np.nan
                if np.isfinite(c) and c > 0:
                    lf.append(math.log(per / c))
            if len(lf) < 5:
                continue
            lf = np.array(lf)
            mad = float(np.median(np.abs(lf - np.median(lf))))
            cand[day] = (mad, float(np.exp(np.median(lf))), len(lf))
            if best is None or mad < best[1]:
                best = (day, mad)
        if best is None:
            continue
        us_before = [x for x in D.days if x < d]
        n_sessions_back = len(us_before) - us_before.index(best[0]) if best[0] in us_before else 0
        res[d] = {"best_us_close": best[0], "sessions_back": n_sessions_back, "weekday_pdf": date.fromisoformat(d).weekday(),
                  "weekday_close": date.fromisoformat(best[0]).weekday(), "mad": best[1],
                  "implied_fx": cand[best[0]][1], "krw_x_that_day": float(fx.get(best[0], np.nan)),
                  "runner_up_mad": sorted(m[0] for m in cand.values())[1] if len(cand) > 1 else None}
    return res


# ----------------------------------------------------------------------------- 3b. 2026 복제
def replica_2026():
    from dataclasses import replace  # noqa: F401
    from bucket_main import BucketEngine, BParams
    from deskx import START
    E = BucketEngine()
    sig = SimpleNamespace()
    o = pd.read_parquet(PARQ, columns=["date", "sym", "Close"])
    src = {}
    for s in ("SOXX", "QQQ"):
        x = o[o.sym == s]
        ser = pd.Series(x["Close"].astype(float).to_numpy(), index=pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
        src[s] = ser[~ser.index.duplicated()].sort_index()

    def below(sym, day, n=20):
        s = src[sym]
        m = s.rolling(n, min_periods=n).mean()
        i = s.index.searchsorted(day, side="right") - 1
        return bool(i >= 0 and np.isfinite(m.iloc[i]) and s.iloc[i] < m.iloc[i])
    memo = {}

    def below_m(sym, day):
        if (sym, day) not in memo:
            memo[sym, day] = below(sym, day)
        return memo[sym, day]

    bases = {"live50": BParams(gran="name", wmode="fund", n=7, semis_cap=0.50),
             "nocap": BParams(gran="name", wmode="fund", n=7, semis_cap=1.0),
             "sub5eq": BParams(gran="sub", wmode="eq", n=5)}
    all_pdfs = list(E.book.pdfs)
    weekly, seen = [], set()
    for p in all_pdfs:
        wk = V.iso(p)
        if wk not in seen:
            weekly.append(p)
            seen.add(wk)
    k0 = E.days.index([d for d in E.days if d >= START][0])
    k1 = len(E.days) - 1
    semis = lambda x: in_semis_budget(E.sector.get(x, UNKNOWN))
    modes = {"daily_strict(데스크)": (all_pdfs, "strict"), "daily_lag1": (all_pdfs, "lag1"),
             "weekly_strict": (weekly, "strict"), "weekly_same": (weekly, "same"), "weekly_lag1": (weekly, "lag1")}
    res = {}
    paths = {}
    for mname, (pdfs, mode) in modes.items():
        sched = V.schedule(SimpleNamespace(days=E.days, pdfs=pdfs), mode, k0, k1)
        for b, p in bases.items():
            sel, pk = set(), {}
            for k in sorted(sched):
                keep, tgt, ranked = E.target(p, k, sched[k] + "~", sel)   # '~' → latest_before가 정확히 그 PDF를 고름
                if ranked:
                    sel = set(keep)
                    pk[k] = (list(keep), dict(tgt))
            for g in ("none", "soxx_w", "qqq_w"):
                units, cash, base, on, vals, lastp = {}, 1.0, {}, False, [], {}
                for k in range(k0, k1 + 1):
                    for x in units:
                        ox = E.col("opn", x)[k]
                        if np.isfinite(ox) and ox > 0:
                            lastp[x] = ox
                    value = cash + sum(units[x] * lastp[x] for x in units)
                    if k in pk:
                        base = pk[k][1]
                        on = g != "none" and below_m(GSYM[g], E.days[k - 1]) if g != "none" else False
                        tgt = gated(base, g, on, semis)
                        units, cash = {}, value
                        for x, wt in tgt.items():
                            ox = E.col("opn", x)[k]
                            assert np.isfinite(ox) and ox > 0, (E.days[k], x)
                            lastp[x] = ox
                            units[x] = value * wt / ox
                            cash -= value * wt
                    vals.append(value)
                v = pd.Series(np.array(vals) / vals[0], index=E.days[k0:k1 + 1])
                r = E.evaluate(f"{b}·{g}·{mname}", v)
                res[f"{b}·{g}·{mname}"] = {"앞": r["앞 1/2~5/29"]["수익"], "뒤": r["뒤 6/1~10/9"]["수익"],
                                           "전체": r["전체"]["수익"], "낙폭": r["전체"]["낙폭"],
                                           "결정요일": sorted({date.fromisoformat(E.days[k]).weekday() for k in pk}),
                                           "PDF나이평균": float(np.mean([(date.fromisoformat(E.days[k]) - date.fromisoformat(sched[k])).days
                                                                     for k in pk]))}
                paths[b, g, mname] = (v, {k: tuple(sorted(pk[k][0])) for k in pk})
    # 데스크(일별 strict)와의 거리: 일 수익 상관, 같은 주 보유 일치율
    dist = {}
    for b in bases:
        for g in ("none", "soxx_w", "qqq_w"):
            ref, refpk = paths[b, g, "daily_strict(데스크)"]
            rr = ref.pct_change().dropna()
            for mname in modes:
                if mname.startswith("daily_strict"):
                    continue
                v, pk = paths[b, g, mname]
                r = v.pct_change().dropna()
                ref_by_wk = {V.iso(E.days[k]): s for k, s in refpk.items()}
                my_by_wk = {V.iso(E.days[k]): s for k, s in pk.items()}
                common = [w for w in ref_by_wk if w in my_by_wk]
                same_sel = float(np.mean([ref_by_wk[w] == my_by_wk[w] for w in common])) if common else float("nan")
                dist[f"{b}·{g}·{mname}"] = {"일수익상관": float(np.corrcoef(rr, r)[0, 1]),
                                            "추적오차_연율": float((r - rr).std() * np.sqrt(252)),
                                            "전체차_%p": (float(v.iloc[-1]) - float(ref.iloc[-1])) * 100,
                                            "주별종목일치": same_sel}
    unk = sorted({E.ticker[k] for k, s in E.sector.items() if s == UNKNOWN and k in E.book.keys})
    return {"runs": res, "dist_to_desk": dist, "weekly_pdfs": len(weekly), "daily_pdfs": len(all_pdfs),
            "unclassified_2026_time_keys": unk}


# ----------------------------------------------------------------------------- 4. 이전 watchlist (하위 프로세스 본체)
def old_watchlist_child():
    D = V.Data(restore=True)
    sig = Sig(D)
    out, _, _, _, _ = run_holdout(D, sig, ["live50", "c774", "nocap", "sub5eq"], ["none", "soxx_w", "qqq_w"],
                                  ["lag1", "strict"])
    unk = sorted({D.tk[k] for k, s in D.sector.items() if s == UNKNOWN})
    sem = sorted({D.tk[k] for k, s in D.sector.items() if in_semis_budget(s)})
    # 2026 데스크(run_model) 이전 분류 값
    from deskx import Engine, START, PERIODS
    from desk.strategies import run_model
    from desk.validate import codex_caps
    E = Engine()
    r50 = run_model(E.book, start=START)
    with codex_caps():
        r60 = run_model(E.book, start=START)
    d26 = {}
    for nm, r in (("live50", r50), ("c774", r60)):
        v = pd.Series(np.asarray(r.value), index=[d for d in E.days if d >= START])
        ev = E.evaluate(nm, v)
        d26[nm] = {lab: ev[lab]["수익"] for lab, _, _ in PERIODS} | {"낙폭": ev["전체"]["낙폭"]}
    print(json.dumps({"runs": out, "unclassified": unk, "semis_budget_names": sem, "desk2026": d26},
                     ensure_ascii=False, default=str))


# ----------------------------------------------------------------------------- main
def pct(x):
    return f"{x * 100:+7.1f}"


def main():
    with open(MAIN_JSON, encoding="utf-8") as f:
        M = json.load(f)
    ref = M["holdout"]["runs"]
    D = V.Data(restore=True)
    sig = Sig(D)
    out = {}

    # 0. 남은 미분류와 56종목의 보조 구간 등장
    unk = sorted({D.tk[k] for k, s in D.sector.items() if s == UNKNOWN})
    in_hold = {tk: D.sector[k] for k, tk in D.tk.items() if tk in NEW56}
    print(f"[4] 보조 구간 TIME 미국 회사 {len(D.keys)}개, 남은 미분류 {unk}; 56종목 중 보조 구간 등장 {len(in_hold)}개")
    moved_out = sorted(tk for tk, s in in_hold.items() if not in_semis_budget(s))
    stayed_in = sorted(tk for tk, s in in_hold.items() if in_semis_budget(s))
    print(f"    반도체 예산 밖으로 나간 것 {len(moved_out)}, 안에 남은 것 {stayed_in}")
    out["unclassified_after"] = unk
    out["new56_in_holdout"] = in_hold

    # 1. 독립 재계산
    bases, gates, modes = ["live50", "c774", "nocap", "sub5eq"], ["none", "soxx_w", "soxx_d", "qqq_w"], \
        ["lag1", "strict", "same", "wd0", "wd1", "wd2", "wd3", "wd4"]
    mine, P, paths, scheds, (k0, k1) = run_holdout(D, sig, bases, gates, modes)
    print("\n[1] 독립 재계산 vs gate_main (전체, 낙폭, 연도) — 차이는 %p")
    cmp_ = {}
    for key, r in ref.items():
        m = mine[key]
        d = {"전체": (m["전체"] - r["전체"]) * 100, "낙폭": (m["낙폭"] - r["낙폭"]) * 100}
        for y in ("2023", "2024", "2025"):
            d[y] = (m["연도"][y] - r["연도"][y]) * 100
        d["게이트일"] = (m["게이트일비율"] - r["전환일비율"]) * 100
        cmp_[key] = d
        flag = "" if max(abs(x) for x in d.values()) <= 0.5 else "  <-- 0.5%p 초과"
        print(f"  {key:<22} 나 {pct(m['전체'])} {pct(m['낙폭'])} | ref {pct(r['전체'])} {pct(r['낙폭'])} | 최대차 "
              f"{max(abs(x) for x in d.values()):.2e}{flag}")
    out["recompute_vs_main"] = {"max_abs_diff_pp": max(max(abs(x) for x in d.values()) for d in cmp_.values()),
                                "rows": cmp_, "mine": {k: mine[k] for k in ref}}
    print(f"  32개 최대 |차| {out['recompute_vs_main']['max_abs_diff_pp']:.2e}%p; 결측 시가 보유일 최대 "
          f"{max(m['결측시가보유일'] for m in mine.values())}")
    assert scheds["wd0"] == scheds["strict"], "wd0 일정은 strict와 같아야 한다"

    # 2. qqq_w lag1 vs strict 분해 (live50)
    print("\n[2] live50·qqq_w lag1 vs strict 분해")
    b = "live50"
    ln = lambda x: math.log(1 + x)
    dec = {}
    for m in ("lag1", "strict", "same", "wd1"):
        none, q = mine[f"{b}·none·{m}"]["전체"], mine[f"{b}·qqq_w·{m}"]["전체"]
        dec[m] = {"none": none, "qqq_w": q, "현금주가뺀로그수익": ln(none) - ln(q)}
        print(f"  {m:<7} none {pct(none)}  qqq_w {pct(q)}  → 현금 주에 빠진 로그수익 {ln(none) - ln(q):+.3f} "
              f"(= 그 주들 합 {math.exp(ln(none) - ln(q)) - 1:+.1%})")
    # 혼합: 고르기 시점 × 게이트 시점
    hyb = {}
    gl1 = set(scheds["lag1"])
    gst = set(scheds["strict"])
    for pm, gm, gd in (("strict", "lag1", gl1), ("lag1", "strict", gst)):
        v, ons, inv, _ = path(D, P[b, pm], "qqq_w", sig, k0, k1, gate_days=gd)
        hyb[f"고르기 {pm} × 게이트 {gm}"] = stats(v) | {"게이트일비율": float(ons.mean())}
    hyb["고르기 strict × 게이트 strict"] = mine[f"{b}·qqq_w·strict"]
    hyb["고르기 lag1 × 게이트 lag1"] = mine[f"{b}·qqq_w·lag1"]
    for k, x in hyb.items():
        print(f"  {k:<28} 전체 {pct(x['전체'])} 낙폭 {pct(x['낙폭'])}")
    # 결정 요일 0~4(앞선 PDF) — qqq_w, none
    wdrow = {}
    for bb in ("live50", "nocap", "sub5eq"):
        for g in ("none", "soxx_w", "qqq_w"):
            wdrow[f"{bb}·{g}"] = [mine[f"{bb}·{g}·wd{i}"]["전체"] for i in range(5)]
            print(f"  {bb}·{g:<7} 결정 요일 월~금 전체 " + " ".join(pct(x) for x in wdrow[f'{bb}·{g}'])
                  + f" | same {pct(mine[f'{bb}·{g}·same']['전체'])} lag1 {pct(mine[f'{bb}·{g}·lag1']['전체'])}")
    # 주별 기여: qqq_w strict 경로와 lag1 경로의 일 로그수익 차를 ISO 주로 묶음
    vs, ons_s, _ = paths[b, "qqq_w", "strict"]
    vl, ons_l, _ = paths[b, "qqq_w", "lag1"]
    ns_, _, _ = paths[b, "none", "strict"]
    nl_, _, _ = paths[b, "none", "lag1"]
    days = list(vs.index)
    ls_ = np.diff(np.log(vs.to_numpy()))
    ll_ = np.diff(np.log(vl.to_numpy()))
    lns = np.diff(np.log(ns_.to_numpy()))
    lnl = np.diff(np.log(nl_.to_numpy()))
    wk = {}
    for i in range(len(ls_)):
        w = V.iso(days[i])
        e = wk.setdefault(w, {"start": days[i], "strict_qqq": 0.0, "lag1_qqq": 0.0, "strict_none": 0.0, "lag1_none": 0.0,
                              "strict_on": [], "lag1_on": []})
        e["strict_qqq"] += ls_[i]
        e["lag1_qqq"] += ll_[i]
        e["strict_none"] += lns[i]
        e["lag1_none"] += lnl[i]
        e["strict_on"].append(bool(ons_s[i]))
        e["lag1_on"].append(bool(ons_l[i]))
    rows = []
    for w, e in wk.items():
        gap = e["lag1_qqq"] - e["strict_qqq"]
        mon = e["start"]
        rows.append({"주": f"{w[0]}-W{w[1]:02d}", "첫날": mon, "lag1-strict 로그차": gap,
                     "strict 현금일": sum(e["strict_on"]), "lag1 현금일": sum(e["lag1_on"]),
                     "strict none 주수익": math.expm1(e["strict_none"]), "lag1 none 주수익": math.expm1(e["lag1_none"]),
                     "strict qqq 주수익": math.expm1(e["strict_qqq"]), "lag1 qqq 주수익": math.expm1(e["lag1_qqq"])})
    rows.sort(key=lambda r: r["lag1-strict 로그차"])
    tot_gap = sum(r["lag1-strict 로그차"] for r in rows)
    # 그 주의 신호 근거
    for r in rows[:8] + rows[-4:]:
        k = D.days.index(r["첫날"]) if r["첫날"] in D.days else None
        ks = [kk for kk in scheds["strict"] if V.iso(D.days[kk]) == V.iso(r["첫날"])]
        kl = [kk for kk in scheds["lag1"] if V.iso(D.days[kk]) == V.iso(r["첫날"])]
        r["strict 결정"] = (D.days[ks[0]], sig.detail("QQQ", D.days[ks[0] - 1])) if ks else None
        r["lag1 결정"] = (D.days[kl[0]], sig.detail("QQQ", D.days[kl[0] - 1])) if kl else None
    neg = sorted(rows, key=lambda r: r["lag1-strict 로그차"])
    top_neg_share = sum(r["lag1-strict 로그차"] for r in neg[:10]) / tot_gap
    print(f"  주별 합 로그차 {tot_gap:+.3f} (= lag1 {pct(mine[f'{b}·qqq_w·lag1']['전체'])} vs strict "
          f"{pct(mine[f'{b}·qqq_w·strict']['전체'])}); 나쁜 10주가 차이의 {top_neg_share:.0%}")
    for r in neg[:8]:
        print(f"   {r['주']} {r['첫날']} 차 {r['lag1-strict 로그차']:+.3f} | 현금일 s{r['strict 현금일']} l{r['lag1 현금일']} | "
              f"none 주수익 s{r['strict none 주수익']:+.1%} l{r['lag1 none 주수익']:+.1%} | qqq_w s{r['strict qqq 주수익']:+.1%} "
              f"l{r['lag1 qqq 주수익']:+.1%} | 결정 s{r['strict 결정']} l{r['lag1 결정']}")
    print("   (lag1이 더 좋은 주)")
    for r in neg[-4:]:
        print(f"   {r['주']} {r['첫날']} 차 {r['lag1-strict 로그차']:+.3f} | 현금일 s{r['strict 현금일']} l{r['lag1 현금일']} | "
              f"none 주수익 s{r['strict none 주수익']:+.1%} l{r['lag1 none 주수익']:+.1%} | 결정 s{r['strict 결정']} l{r['lag1 결정']}")
    # 게이트 상태가 다른 주 수와 그 주들의 기여
    diff_state = [r for r in rows if (r["strict 현금일"] > 0) != (r["lag1 현금일"] > 0)]
    same_state = [r for r in rows if (r["strict 현금일"] > 0) == (r["lag1 현금일"] > 0)]
    print(f"  게이트 상태가 다른 주 {len(diff_state)}/{len(rows)}: 로그차 합 {sum(r['lag1-strict 로그차'] for r in diff_state):+.3f};"
          f" 같은 주 {len(same_state)}: 합 {sum(r['lag1-strict 로그차'] for r in same_state):+.3f}")
    out["qqq_decomp"] = {"by_mode": dec, "hybrid": hyb, "weekday": wdrow, "total_log_gap": tot_gap,
                         "worst_weeks": neg[:12], "best_weeks": neg[-6:],
                         "diff_state_weeks": len(diff_state), "diff_state_gap": sum(r["lag1-strict 로그차"] for r in diff_state),
                         "same_state_gap": sum(r["lag1-strict 로그차"] for r in same_state)}

    # 3. PDF 날짜 의미 + 결정 때 PDF 나이
    val = pdf_valuation_lag(D)
    from collections import Counter
    sb = Counter(x["sessions_back"] for x in val.values())
    mon = [x for d, x in val.items() if x["weekday_pdf"] == 0]
    sbm = Counter(x["sessions_back"] for x in mon)
    print(f"\n[3] PDF 평가 기준일: 미국 세션 몇 개 전 종가인가 {dict(sb)} (월요일 PDF만 {dict(sbm)}),"
          f" 그 종가의 요일(월요일 PDF) {dict(Counter(x['weekday_close'] for x in mon))}")
    ages = {}
    for m in ("strict", "same", "lag1"):
        s = scheds[m]
        ages[m] = {"PDF나이평균": float(np.mean([(date.fromisoformat(D.days[k]) - date.fromisoformat(p)).days for k, p in s.items()])),
                   "결정요일": dict(Counter(date.fromisoformat(D.days[k]).weekday() for k in s)), "결정수": len(s)}
        print(f"  {m:<7} 결정 {len(s)}회, 요일 {ages[m]['결정요일']}, PDF 나이(달력일) 평균 {ages[m]['PDF나이평균']:.1f}")
    out["pdf_semantics"] = {"sessions_back_counts": dict(sb), "monday_pdf_sessions_back": dict(sbm),
                            "sample": {d: val[d] for d in list(val)[::15]}, "decision_age": ages}
    print("  2026 복제(일별 PDF 데스크 vs 주 1개 PDF 모드) 실행 중…")
    rep = replica_2026()
    for k, x in rep["runs"].items():
        print(f"  {k:<36} 앞 {pct(x['앞'])} 뒤 {pct(x['뒤'])} 전체 {pct(x['전체'])} 낙폭 {pct(x['낙폭'])} 요일 {x['결정요일']} "
              f"PDF나이 {x['PDF나이평균']:.1f}")
    for k, x in rep["dist_to_desk"].items():
        print(f"  거리 {k:<34} 상관 {x['일수익상관']:.3f} 추적오차 {x['추적오차_연율']:.3f} 전체차 {x['전체차_%p']:+7.1f}%p "
              f"주별 종목 일치 {x['주별종목일치']:.2f}")
    print(f"  2026 TIME 미국 보유 미분류: {rep['unclassified_2026_time_keys']}")
    out["replica_2026"] = rep

    # 4. 이전 watchlist(하위 프로세스)
    print("\n[4] d5ce794 이전 watchlist로 다시")
    oldwl = os.path.join(SP, "oldwl") if SP else ""
    old = None
    if oldwl and os.path.exists(os.path.join(oldwl, "watchlist.py")):
        env = dict(os.environ, GATE_OLD_WATCHLIST=oldwl, PYTHONIOENCODING="utf-8")
        cp = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env, capture_output=True,
                            text=True, encoding="utf-8")
        assert cp.returncode == 0, cp.stderr[-3000:]
        old = json.loads(cp.stdout.strip().splitlines()[-1])
        print(f"  이전 분류 미분류 수(보조 구간) {len(old['unclassified'])}")
        for key, x in old["runs"].items():
            n = mine[key]
            print(f"  {key:<22} 이전 {pct(x['전체'])} {pct(x['낙폭'])} | 지금 {pct(n['전체'])} {pct(n['낙폭'])} | 차 "
                  f"{(n['전체'] - x['전체']) * 100:+6.1f}%p")
        print("  2026 데스크(run_model) 이전 분류:", {k: {a: round(b * 100, 1) for a, b in v.items()} for k, v in old["desk2026"].items()})
        out["old_watchlist"] = {"runs": old["runs"], "unclassified": old["unclassified"], "desk2026": old["desk2026"],
                                "diff_pp": {k: (mine[k]["전체"] - x["전체"]) * 100 for k, x in old["runs"].items()}}
    # 56종목이 실제로 뽑힌 주(live50·strict / lag1)
    used = {}
    for m in ("strict", "lag1"):
        for bb in ("live50", "sub5eq"):
            cnt = {}
            for k, (sel, tgt) in P[bb, m].items():
                for x in sel:
                    if D.tk[x] in NEW56:
                        cnt[D.tk[x]] = cnt.get(D.tk[x], 0) + 1
            used[f"{bb}·{m}"] = dict(sorted(cnt.items(), key=lambda t: -t[1]))
    print("  56종목이 뽑힌 주 수:", used)
    out["new56_selected_weeks"] = used

    # 5. C3: 보조 구간 결정 요일별 soxx_w - none, 낙폭
    print("\n[5] 보조 구간 결정 요일별 soxx_w - none (전체 %p / 낙폭)")
    c3 = {}
    for bb in ("live50", "c774", "nocap", "sub5eq"):
        d = [(mine[f"{bb}·soxx_w·wd{i}"]["전체"] - mine[f"{bb}·none·wd{i}"]["전체"]) * 100 for i in range(5)]
        q = [(mine[f"{bb}·qqq_w·wd{i}"]["전체"] - mine[f"{bb}·none·wd{i}"]["전체"]) * 100 for i in range(5)]
        dm = [(mine[f"{bb}·soxx_w·wd{i}"]["낙폭"], mine[f"{bb}·none·wd{i}"]["낙폭"]) for i in range(5)]
        c3[bb] = {"soxx_w-none": d, "qqq_w-none": q, "soxx_w,none 낙폭": dm}
        print(f"  {bb:<7} soxx_w-none " + " ".join(f"{x:+6.1f}" for x in d) + " | qqq_w-none " + " ".join(f"{x:+6.1f}" for x in q)
              + " | 낙폭 s/n " + " ".join(f"{a * 100:.0f}/{b * 100:.0f}" for a, b in dm))
    out["c3_weekday_holdout"] = c3
    out["all_mine"] = mine

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        old_watchlist_child()
    else:
        main()
