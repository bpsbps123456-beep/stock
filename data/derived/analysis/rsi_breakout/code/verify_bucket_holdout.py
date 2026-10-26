"""적대적 감사: bucket_holdout.py(버킷 분산 모멘텀, 2023-05-22~2025-12-31 TIME 미사용 구간) 검증 (2026-10-10).

독립 재구현(선택·경로·시점 코드를 새로 씀, 상한만 desk.strategies.project_caps 정본 사용)으로
  A. 빌더 18개 결과 재현(같은 조건: 앞선 PDF, 원시 시가 USD)
  B. 시점·가격 정렬·분할 점검(3개 날짜 + 공휴일 주 + 알려진 분할일)
  C. 보고서 13장 재현(실험실 타이밍 '시가 +0/+1일', 배당 반영 시가, 원화) — 엔진·자료 일치 확인
  D. 입력 결함: TIME PDF의 ISIN 표기 3종목(SMCI·ANET·BZFD)이 desk us_weights에서 빠짐
     (strategy_comparison_audit/holdout_isin_audit.md가 2026-09-27에 이미 지적). 복원 민감도.
  E. 강건성: 시점 3종(앞선 PDF / 당일 PDF / 공시 +1세션) x 복원 여부 x PLTR 제외, 연도·급락 구간, 낙폭 구간.
쓰기: results/bucket/verify_holdout.json 만. desk/gbrc/docs/watchlist·기존 연구 파일은 읽기만.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/verify_bucket_holdout.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from bisect import bisect_left, bisect_right
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, ROOT)
from desk import config  # noqa: E402
from desk.strategies import UNKNOWN, project_caps, sector_of, subsector_of, us_weights  # noqa: E402
from gbrc.etf.market import yahoo_ticker  # noqa: E402

LAB = os.path.join(ROOT, "data", "derived", "analysis", "strategy_lab")
HOLD = os.path.join(LAB, "holdout")
AUDIT = os.path.join(ROOT, "data", "derived", "analysis", "strategy_comparison_audit")
BUILDER_JSON = os.path.join(HERE, "..", "results", "bucket", "holdout.json")
OUT = os.path.join(HERE, "..", "results", "bucket", "verify_holdout.json")

ISIN = {"US86800U1043": "SMCI US EQUITY", "US12430A1025": "BZFD US EQUITY", "US0404131064": "ANET US EQUITY"}
MERGE = {"AI 반도체": "반도체 칩", "반도체(적자)": "반도체 칩",
         "반도체 장비": "반도체 장비·소재", "반도체 장비(적자)": "반도체 장비·소재", "반도체 소재": "반도체 장비·소재"}
START, END = "2023-05-22", "2026-01-02"
YEARS = (("2023", "2023-05-22", "2024-01-02"), ("2024", "2024-01-02", "2025-01-02"), ("2025", "2025-01-02", "2026-01-02"))
STRESS = (("23가을 7/31~10/27", "2023-07-31", "2023-10-27"), ("24여름 7/10~8/7", "2024-07-10", "2024-08-07"),
          ("25봄 2/19~4/8", "2025-02-19", "2025-04-08"), ("25반등 4/8~6/30", "2025-04-08", "2025-06-30"))
LOOK = 120


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, restore: bool):
        snaps = {}
        for p in sorted(glob.glob(os.path.join(HOLD, "time_weekly", "*.json"))):
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            if restore:
                s = dict(s)
                s["holdings"] = [dict(h, code=ISIN.get(h.get("code"), h.get("code"))) for h in s["holdings"]]
            snaps[s["actual_date"]] = s
        self.pdfs = sorted(snaps)
        self.w = {p: us_weights(snaps[p]) for p in self.pdfs}
        frames = {}
        for f in ("adj_close", "open", "close"):
            a = pd.read_csv(os.path.join(HOLD, f"prices_{f}.csv"), index_col=0, parse_dates=True)
            b = pd.read_csv(os.path.join(AUDIT, f"added_prices_{f}.csv"), index_col=0, parse_dates=True)
            frames[f] = a.join(b, how="outer") if restore else a
        self.days = [d.strftime("%Y-%m-%d") for d in frames["adj_close"]["QQQ"].dropna().index]
        idx = pd.DatetimeIndex(pd.to_datetime(self.days))
        self.keys = sorted({k for w in self.w.values() for k in w})
        self.tk = {k: yahoo_ticker(k) for k in self.keys}
        cols = [self.tk[k] for k in self.keys]
        missing = [t for t in cols if t not in frames["adj_close"].columns]
        assert not missing, missing
        self.A = frames["adj_close"].reindex(idx)[cols].to_numpy(float)
        self.O = frames["open"].reindex(idx)[cols].to_numpy(float)
        self.C = frames["close"].reindex(idx)[cols].to_numpy(float)
        self.Odiv = self.O * self.A / self.C
        self.fx = frames["adj_close"].reindex(idx)["KRW=X"].ffill().to_numpy(float)
        self.qqq = (frames["open"]["QQQ"] * frames["adj_close"]["QQQ"] / frames["close"]["QQQ"]).reindex(idx).to_numpy(float)
        self.etf = frames["adj_close"]["456600.KS"].dropna()
        self.j = {k: i for i, k in enumerate(self.keys)}
        self.sector = {k: sector_of(k, self.tk[k]) for k in self.keys}
        self.sub = {k: subsector_of(k, self.tk[k]) for k in self.keys}
        self.raw = frames

    def bucket(self, key, G):
        lab = self.sector[key] if G == "sector" else self.sub[key]
        if lab == UNKNOWN:
            return UNKNOWN + ":" + key
        return MERGE.get(lab, lab) if G == "sector" else lab


# ----------------------------------------------------------------------------- timing
def iso(d):
    return date.fromisoformat(d).isocalendar()[:2]


def schedule(D: Data, mode: str, k0: int, k1: int):
    """{session index: pdf}. strict/same: ISO 주 첫 세션(PDF < / <= 세션). lagN: 실험실 방식(PDF 날짜 이후 첫 미국 세션 + N)."""
    out = {}
    if mode in ("strict", "same"):
        started, prev_wk = False, None
        for k in range(k0, k1 + 1):
            day = D.days[k]
            if started and iso(day) == prev_wk:
                continue
            i = (bisect_left if mode == "strict" else bisect_right)(D.pdfs, day) - 1
            if i < 0:
                continue
            out[k], started, prev_wk = D.pdfs[i], True, iso(day)
        return out
    shift = int(mode[3:])
    keyed = {}
    for p in D.pdfs:
        i = bisect_left(D.days, p) + shift
        keyed[min(i, len(D.days) - 1)] = p          # 늦은 PDF가 이긴다(정렬 순)
    prev_wk = None
    for k in sorted(keyed):
        if k < k0 or k > k1:
            continue
        if iso(D.days[k]) != prev_wk:
            out[k], prev_wk = keyed[k], iso(D.days[k])
    return out


# ----------------------------------------------------------------------------- rules
def ranking(D: Data, k: int, pdf: str, exclude=(), need_mom=True):
    rows = []
    for key, wt in sorted(D.w[pdf].items()):
        if wt <= 0 or key in exclude:
            continue
        j = D.j[key]
        p1 = D.A[k - 1, j]
        p0 = D.A[k - 1 - LOOK, j] if k - 1 - LOOK >= 0 else np.nan
        if need_mom:
            if not (np.isfinite(p1) and np.isfinite(p0) and p0 > 0):
                continue
            mom = p1 / p0 - 1
        else:
            if not np.isfinite(p1):
                continue
            mom = p1 / p0 - 1 if np.isfinite(p0) and p0 > 0 else -np.inf
        rows.append((key, mom, wt))
    rows.sort(key=lambda r: -r[1])
    return rows


def with_caps(raw, sec, stock, sect, semis):
    saved = dict(config.RULE)
    try:
        config.RULE.update(stock_cap=stock, sector_cap=sect, semis_cap=semis)
        t = project_caps(raw, sec)
    finally:
        config.RULE.clear()
        config.RULE.update(saved)
    # 상한 정합성(감사): 합 <= 1, 종목 <= 상한, 섹터 <= 상한
    assert sum(t.values()) <= 1 + 1e-9 and all(v <= stock + 1e-9 for v in t.values())
    g = {}
    for x, v in t.items():
        g[sec.get(x, UNKNOWN)] = g.get(sec.get(x, UNKNOWN), 0) + v
    assert all(v <= sect + 1e-9 for v in g.values()), g
    return t


def desk_rule(n=7, buf=5, semis=0.50, stock=0.25, sect=0.60, weight="fund", rank_by="mom", caps=True):
    def f(D, k, pdf, cur, exclude):
        rows = ranking(D, k, pdf, exclude, need_mom=(rank_by == "mom"))
        if rank_by == "w":
            rows.sort(key=lambda r: -r[2])
        if not rows:
            return None
        order = [r[0] for r in rows]
        keep = [x for x in order[:n + buf] if x in cur][:n]
        for x in order:
            if len(keep) >= n:
                break
            if x not in keep:
                keep.append(x)
        tw = {r[0]: r[2] for r in rows}
        raw = ({x: tw[x] / sum(tw[y] for y in keep) for x in keep} if weight == "fund"
               else {x: 1 / len(keep) for x in keep})
        tgt = with_caps(raw, D.sector, stock, sect, semis) if caps else raw
        return keep, tgt
    return f


def bucket_rule(G, N, W):
    def f(D, k, pdf, cur, exclude):
        rows = ranking(D, k, pdf, exclude)
        if not rows:
            return None
        pos = {r[0]: i for i, r in enumerate(rows)}
        mom = {r[0]: r[1] for r in rows}
        groups = {}
        for r in rows:
            groups.setdefault(D.bucket(r[0], G), []).append(r[0])
        leaders = sorted(groups, key=lambda b: pos[groups[b][0]])
        lr = {b: i + 1 for i, b in enumerate(leaders)}
        best = {}
        for x in cur:
            if x not in pos:
                continue
            b = D.bucket(x, G)
            if lr[b] <= N + 5 and groups[b].index(x) < 2 and (b not in best or mom[x] > mom[best[b]]):
                best[b] = x
        sel = [best[b] for b in sorted(best, key=lr.get)][:N]
        have = {D.bucket(x, G) for x in sel}
        for b in leaders:
            if len(sel) >= N:
                break
            if b not in have:
                sel.append(groups[b][0])
                have.add(b)
        assert len({D.bucket(x, G) for x in sel}) == len(sel)
        if W == "eq":
            raw = {x: 1 / len(sel) for x in sel}
        else:
            tw = {r[0]: r[2] for r in rows}
            raw = {x: tw[x] / sum(tw[y] for y in sel) for x in sel}
        return sel, with_caps(raw, D.sector, max(0.25, 1 / N), 0.60, 10.0)
    return f


# ----------------------------------------------------------------------------- path
def simulate(D: Data, rule, mode="strict", start=START, end=END, div=False, exclude=()):
    k0 = next(i for i, d in enumerate(D.days) if d >= start)
    k1 = max(i for i, d in enumerate(D.days) if d <= end)
    sched = schedule(D, mode, k0, k1)
    O = D.Odiv if div else D.O
    val = np.ones(k1 - k0 + 1)
    w = {}
    cur = []
    log = []
    started_k = None
    for k in range(k0, k1 + 1):
        t = k - k0
        if t > 0:
            g, tot, new = 0.0, 0.0, {}
            for key, wt in w.items():
                j = D.j[key]
                a, b = O[k - 1, j], O[k, j]
                r = b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
                g += wt * r
                new[key] = wt * (1 + r)
            tot = sum(new.values()) + 1 - sum(w.values())
            w = {key: v / tot for key, v in new.items()}
            val[t] = val[t - 1] * (1 + g)
        if k in sched:
            res = rule(D, k, sched[k], cur, exclude)
            if res is not None:
                cur, w = list(res[0]), dict(res[1])
                started_k = started_k if started_k is not None else k
                log.append({"day": D.days[k], "pdf": sched[k], "sel": list(cur), "tgt": w})
    return pd.Series(val, index=D.days[k0:k1 + 1]), log


def mdd_info(v: pd.Series):
    a = v.to_numpy()
    peak = np.maximum.accumulate(a)
    dd = a / peak - 1
    i = int(dd.argmin())
    p = int(np.argmax(a[:i + 1] >= peak[i] - 1e-15))
    return float(dd[i]), v.index[p], v.index[i]


def seg(v, a, b):
    x = v[(v.index >= a) & (v.index <= b)]
    return float(x.iloc[-1] / x.iloc[0] - 1)


def stats(D, v, log):
    m, pk, tr = mdd_info(v)
    k0, k1 = D.days.index(v.index[0]), D.days.index(v.index[-1])
    usd = float(v.iloc[-1] - 1)
    krw = (1 + usd) * D.fx[k1 - 1] / D.fx[k0 - 1] - 1
    return {"full": usd, "krw": krw, "mdd": m, "mdd_peak": pk, "mdd_trough": tr,
            "years": {a: seg(v, b, c) for a, b, c in YEARS}, "stress": {a: seg(v, b, c) for a, b, c in STRESS},
            "first": log[0]["day"] if log else None, "decisions": len(log)}


def specs():
    out = [("desk50", desk_rule(semis=0.50)), ("c774_60", desk_rule(semis=0.60))]
    for G in ("sector", "sub"):
        for N in (3, 5, 7, 10):
            for W in ("eq", "fund"):
                out.append((f"{G}-N{N}-{W}", bucket_rule(G, N, W)))
    return out


BUILDER_NAME = {"desk50": "데스크 규칙 (반도체+미분류 50%)", "c774_60": "#774 (반도체+미분류 60%)"}


def bname(n):
    if n in BUILDER_NAME:
        return BUILDER_NAME[n]
    g, nn, w = n.split("-")
    return f"bucket {g} {nn} {w}"


# ----------------------------------------------------------------------------- main
def main():
    out = {}
    D0 = Data(restore=False)
    D1 = Data(restore=True)
    print(f"PDF {len(D0.pdfs)} ({D0.pdfs[0]}..{D0.pdfs[-1]}), 회사 {len(D0.keys)} -> 복원 {len(D1.keys)}; "
          f"섹터 미분류 {sum(D0.sector[k] == UNKNOWN for k in D0.keys)}, 소분류 미분류 {sum(D0.sub[k] == UNKNOWN for k in D0.keys)}")
    smci = [p for p in D1.pdfs if "SMCI US EQUITY" in D1.w[p]]
    out["isin_gap"] = {"companies_before": len(D0.keys), "companies_after": len(D1.keys),
                       "smci_pdfs": len(smci), "smci_max_w": max(D1.w[p]["SMCI US EQUITY"] for p in smci),
                       "unclassified_after": sum(D1.sector[k] == UNKNOWN for k in D1.keys)}

    # A. 빌더 재현 ------------------------------------------------------------------
    with open(BUILDER_JSON, encoding="utf-8") as f:
        bj = json.load(f)
    bres = {r["name"]: r for r in bj["results"]}
    A = {}
    runs = {}
    for name, rule in specs():
        v, log = simulate(D0, rule, "strict")
        s = stats(D0, v, log)
        runs[("base", "strict", name)] = (v, log, s)
        b = bres[bname(name)]
        A[name] = {"mine": s["full"], "builder": b["full"], "diff": s["full"] - b["full"],
                   "mdd_mine": s["mdd"], "mdd_builder": b["mdd"],
                   "ydiff_max": max(abs(s["years"][y] - b["years"][yy]) for y, yy in zip(("2023", "2024", "2025"), ("2023(5/22~)", "2024", "2025")))}
    mx = max(abs(a["diff"]) for a in A.values())
    print(f"\nA. 빌더 재현(앞선 PDF, 원시 시가 USD): 18개 최대 |차이| {mx:.2e}, MDD 최대 |차이| "
          f"{max(abs(a['mdd_mine'] - a['mdd_builder']) for a in A.values()):.2e}, 연도 {max(a['ydiff_max'] for a in A.values()):.2e}")
    out["A_reproduce"] = A

    # B. 시점·가격·분할 -------------------------------------------------------------
    print("\nB1. 시점(세션 -> 쓰는 PDF, 묵은 일수)")
    B1 = []
    for day in ("2023-10-02", "2024-01-16", "2025-02-03", "2024-09-16", "2024-09-23"):
        k = D0.days.index(day)
        row = {"session": day}
        for mode in ("strict", "same"):
            i = (bisect_left if mode == "strict" else bisect_right)(D0.pdfs, day) - 1
            row[mode] = D0.pdfs[i]
            row[mode + "_age_days"] = (date.fromisoformat(day) - date.fromisoformat(D0.pdfs[i])).days
        lag1 = schedule(D0, "lag1", D0.days.index("2023-05-23"), len(D0.days) - 1)
        nxt = [(kk, p) for kk, p in lag1.items() if kk >= k][:1]
        row["lag1_next"] = (D0.days[nxt[0][0]], nxt[0][1]) if nxt else None
        bl = [d for d in runs[("base", "strict", "desk50")][1] if d["day"] == day]
        row["builder_desk_pdf"] = bl[0]["pdf"] if bl else None
        B1.append(row)
        print("  ", row)
    ages = []
    for d in runs[("base", "strict", "desk50")][1]:
        ages.append((date.fromisoformat(d["day"]) - date.fromisoformat(d["pdf"])).days)
    print(f"   앞선 PDF 나이(달력일): 평균 {np.mean(ages):.1f}, 최소 {min(ages)}, 최대 {max(ages)}; 1주 이상 {np.mean(np.array(ages) >= 7):.2f}")
    out["B1_timing"] = {"rows": B1, "age_mean": float(np.mean(ages)), "age_min": int(min(ages)), "age_max": int(max(ages)),
                        "age_ge7_share": float(np.mean(np.array(ages) >= 7))}

    print("\nB2. 가격 정렬(CSV 날짜 직접 조회 vs 엔진 배열)")
    B2 = []
    for day, tk in (("2023-10-02", "PLTR"), ("2024-08-05", "NVDA"), ("2025-04-07", "OKLO")):
        k = D0.days.index(day)
        key = next(x for x in D0.keys if D0.tk[x] == tk)
        j = D0.j[key]
        raw_o = D0.raw["open"][tk]
        raw_a = D0.raw["adj_close"][tk]
        r = {"day": day, "ticker": tk, "open_k": D0.O[k, j], "csv_open_k": float(raw_o.loc[day]),
             "open_k-1": D0.O[k - 1, j], "csv_open_prev": float(raw_o.loc[D0.days[k - 1]]), "prev_day": D0.days[k - 1],
             "adj_k-1": D0.A[k - 1, j], "adj_k-121": D0.A[k - 121, j], "day_k-121": D0.days[k - 121],
             "csv_adj_k-121": float(raw_a.loc[D0.days[k - 121]])}
        assert abs(r["open_k"] - r["csv_open_k"]) < 1e-9 and abs(r["adj_k-121"] - r["csv_adj_k-121"]) < 1e-9
        B2.append(r)
        print("  ", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()})
    out["B2_alignment"] = B2

    print("\nB3. 분할일 시가 연속성(원시 시가는 분할 반영이어야 함)")
    B3 = []
    for tk, day, ratio in (("NVDA", "2024-06-10", 10), ("AVGO", "2024-07-15", 10), ("ANET", "2024-12-04", 4),
                           ("SMCI", "2024-10-01", 10)):
        DD = D1 if tk == "SMCI" else D0
        key = next((x for x in DD.keys if DD.tk[x] == tk), None)
        if key is None:
            continue
        k = DD.days.index(day)
        j = DD.j[key]
        r = {"ticker": tk, "split_day": day, "ratio": ratio, "open_prev": round(DD.O[k - 1, j], 3),
             "open_day": round(DD.O[k, j], 3), "o2o": round(DD.O[k, j] / DD.O[k - 1, j] - 1, 4)}
        B3.append(r)
        print("  ", r)
    # 보유 중 큰 시가 수익 전부: 같은 날 종가 수익과 비교
    big = []
    for (_, _, name), (v, log, s) in runs.items():
        pass
    held = set()
    for (_, _, name), (v, log, s) in runs.items():
        for d in log:
            held |= set(d["tgt"])
    for key in sorted(held):
        j = D0.j[key]
        o = D0.O[:, j]
        c = D0.C[:, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            ro = o[1:] / o[:-1] - 1
            rc = c[1:] / c[:-1] - 1
        for i in np.where(np.abs(np.nan_to_num(ro)) > 0.45)[0]:
            if D0.days[i + 1] < "2023-05-22":
                continue
            big.append({"ticker": D0.tk[key], "day": D0.days[i + 1], "o2o": round(float(ro[i]), 3),
                        "c2c_prev": round(float(rc[i - 1]), 3), "c2c_day": round(float(rc[i]), 3)})
    nan_open = 0
    for (_, _, name), (v, log, s) in runs.items():
        pass
    print(f"   보유 종목의 |시가→시가| > 45% 일: {len(big)}건 ->", big[:12])
    out["B3_splits"] = {"known": B3, "big_open_moves": big}

    # 보유 중 시가 결측(수익 0 처리) 일수
    miss = {}
    for name in ("desk50", "sub-N5-eq", "sub-N5-fund", "sector-N5-fund"):
        v, log, s = runs[("base", "strict", name)]
        days = {d["day"]: d for d in log}
        cnt = 0
        k0 = D0.days.index(v.index[0])
        cur = {}
        for k in range(k0, k0 + len(v)):
            if D0.days[k] in days:
                cur = days[D0.days[k]]["tgt"]
            for key in cur:
                j = D0.j[key]
                if k + 1 < len(D0.days) and not (np.isfinite(D0.O[k, j]) and np.isfinite(D0.O[k + 1, j])):
                    cnt += 1
        miss[name] = cnt
    print("   보유 중 시가 결측(수익 0 처리) 종목-일:", miss)
    out["B3_missing_open_holding_days"] = miss

    # C. 보고서 13장 재현 ------------------------------------------------------------
    print("\nC. 보고서 13장 재현(실험실 타이밍, 배당 반영 시가, 원화, 끝 2025-12-31 시가)")
    lab_rules = {"TIME 복제": desk_rule(n=500, buf=0, weight="fund", rank_by="w", caps=False),
                 "비중 Top5": desk_rule(n=5, buf=5, weight="fund", rank_by="w", caps=False),
                 "H6e": desk_rule(n=5, buf=5, weight="eq", caps=False),
                 "H6b": desk_rule(n=5, buf=5, weight="fund", caps=False),
                 "H6f": desk_rule(n=10, buf=10, weight="eq", caps=False)}
    rep13 = {"lag0": {"TIME 복제": 2.24, "비중 Top5": 3.36, "H6e": 1.86, "H6b": 2.91, "H6f": 2.16},
             "lag1": {"TIME 복제": 2.21, "비중 Top5": 3.36, "H6e": 1.93, "H6b": 2.85, "H6f": 1.28}}
    C = {}
    for mode in ("lag0", "lag1"):
        k_first = None
        for name, rule in lab_rules.items():
            sched_start = "2023-05-22" if mode == "lag0" else "2023-05-23"
            v, log = simulate(D0, rule, mode, start=sched_start, end="2025-12-31", div=True)
            k0, k1 = D0.days.index(v.index[0]), D0.days.index(v.index[-1])
            krw = (1 + v.iloc[-1] - 1) * D0.fx[k1] / D0.fx[k0] - 1
            C[f"{mode}:{name}"] = {"krw": float(krw), "report13": rep13[mode][name], "mdd_usd": mdd_info(v)[0]}
        print("  ", mode, {n: (round(C[f'{mode}:{n}']['krw'], 3), rep13[mode][n]) for n in lab_rules})
    # 감사(holdout_isin_audit) H6e 시가+1: 원본 +1.9246, 복원 +2.1227
    v, log = simulate(D1, lab_rules["H6e"], "lag1", start="2023-05-23", end="2025-12-31", div=True)
    k0, k1 = D1.days.index(v.index[0]), D1.days.index(v.index[-1])
    C["lag1:H6e_restored"] = {"krw": float(v.iloc[-1] * D1.fx[k1] / D1.fx[k0] - 1), "isin_audit": 2.122732}
    print("   H6e 시가+1 복원:", round(C["lag1:H6e_restored"]["krw"], 4), "(감사 +2.1227)")
    out["C_report13"] = C

    # D/E. 강건성 행렬 ---------------------------------------------------------------
    print("\nD/E. 시점 x ISIN 복원 x PLTR 제외 (USD 시가 경로, 끝 2026-01-02 시가)")
    M = {}
    for dtag, DD in (("base", D0), ("restored", D1)):
        for mode in ("strict", "same", "lag1"):
            for excl in ((), ("PLTR US EQUITY",)):
                tag = f"{dtag}|{mode}|{'noPLTR' if excl else 'all'}"
                st = START if mode != "lag1" else START
                for name, rule in specs():
                    if (dtag, mode, name) in [(a, b, c) for a, b, c in runs] and not excl:
                        v, log, s = runs[(dtag, mode, name)]
                    else:
                        v, log = simulate(DD, rule, mode, start=st, exclude=excl)
                        s = stats(DD, v, log)
                        if not excl:
                            runs[(dtag, mode, name)] = (v, log, s)
                    M.setdefault(tag, {})[name] = s
    out["matrix"] = M

    def line(tag):
        d = M[tag]
        desk = d["desk50"]["full"]
        names = [n for n in d if n not in ("desk50", "c774_60")]
        beat = [n for n in names if d[n]["full"] > desk]
        beat_both = [n for n in names if d[n]["full"] > desk and d[n]["mdd"] > d["desk50"]["mdd"]]
        return desk, d["c774_60"]["full"], d["sub-N5-eq"]["full"], len(beat), len(beat_both), \
            sorted(names, key=lambda n: -d[n]["full"])[:3], d["desk50"]["mdd"], d["sub-N5-eq"]["mdd"]

    summ = {}
    for tag in M:
        desk, c774, fp, nb, nbb, top, dm, fm = line(tag)
        summ[tag] = {"desk50": desk, "c774": c774, "front_pick_sub_N5_eq": fp, "beat_desk": nb,
                     "beat_desk_and_mdd": nbb, "top3": top, "desk_mdd": dm, "front_pick_mdd": fm}
        print(f"  {tag:28s} desk50 {desk:+.3f} ({dm:+.3f}) #774 {c774:+.3f} | 2026앞선택 sub-N5-eq {fp:+.3f} ({fm:+.3f}) "
              f"| 데스크 이긴 수 {nb}/16, 수익·낙폭 둘다 {nbb}/16 | 상위 {top}")
    out["matrix_summary"] = summ

    # 변형별 시점 간 순위 안정성 (base/all)
    print("\nE1. 16개 버킷 변형 순위의 시점 간 상관(스피어만, base·전체)")
    names = [n for n, _ in specs()][2:]
    rk = {}
    for mode in ("strict", "same", "lag1"):
        vals = [M[f"base|{mode}|all"][n]["full"] for n in names]
        rk[mode] = pd.Series(vals, index=names).rank()
    cors = {f"{a}~{b}": float(rk[a].corr(rk[b])) for a, b in (("strict", "same"), ("strict", "lag1"), ("same", "lag1"))}
    print("  ", {k: round(v, 2) for k, v in cors.items()})
    out["rank_corr"] = cors
    rkr = {}
    for mode in ("strict", "same", "lag1"):
        rkr[mode] = pd.Series([M[f"restored|{mode}|all"][n]["full"] for n in names], index=names).rank()
    out["rank_corr_base_vs_restored"] = {m: float(rk[m].corr(rkr[m])) for m in rk}
    print("   복원 전후 순위 상관:", {m: round(v, 2) for m, v in out["rank_corr_base_vs_restored"].items()})

    # E2. 연도·급락·낙폭 구간: 데스크 vs 앞 선택 vs N5 넷 (복원·lag1 포함)
    print("\nE2. 연도·급락 구간 (USD 시가 경로)")
    E2 = {}
    for tag in ("base|strict|all", "base|same|all", "base|lag1|all", "restored|strict|all", "restored|lag1|all"):
        for n in ("desk50", "c774_60", "sub-N5-eq", "sub-N5-fund", "sector-N5-fund", "sector-N5-eq"):
            s = M[tag][n]
            E2[f"{tag}:{n}"] = s
            print(f"  {tag:22s} {n:15s} 전체 {s['full']:+.3f} | " + " ".join(f"{y} {s['years'][y]:+.3f}" for y in s["years"]) +
                  " | " + " ".join(f"{a} {b:+.3f}" for a, b in s["stress"].items()) +
                  f" | MDD {s['mdd']:+.3f} {s['mdd_peak']}~{s['mdd_trough']}")
    # 원화 TIME ETF
    e = D0.etf
    e.index = e.index.strftime("%Y-%m-%d")
    a = e[e.index <= "2023-05-23"].iloc[-1]
    b = e[e.index <= "2026-01-02"].iloc[-1]
    out["etf_krw"] = float(b / a - 1)
    print(f"\nTIME ETF 원화 {out['etf_krw']:+.4f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("저장", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
