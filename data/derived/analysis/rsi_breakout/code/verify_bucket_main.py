"""bucket_main.py 독립 검증(감사, 2026-10-10). desk/·gbrc/·기존 연구 파일은 고치지 않고 읽기만 한다.

bucket_main.py·deskx.Engine의 선택·비중·경로 코드를 쓰지 않고 처음부터 다시 짠다.
  가격: rsi_breakout/cache/deskx_prices/*.csv를 직접 읽음(desk.marketdata.history 호출 안 함)
  PDF: gbrc.etf.trades.load_fund_snapshots + desk.strategies.us_weights(미국 보유 정의, 데스크와 같은 정의)
  분류: desk.strategies.sector_of / subsector_of(사전 등록 규칙이 이 함수를 정의로 지정)
  상한: 데스크 project_caps를 쓰지 않고 '동결 시각' 닫힌 식으로 다시 계산
        (종목 i는 비율 raw_i로 자라다 min(종목 상한 시각, 섹터 상한 시각, 반도체 예산 시각)에 멈춘다)
점검
  1) 32개 + live + #774 + 사후 a/b/c 수익·MDD·구조 지표를 main.json과 대조
  2) 모든 변형·모든 주의 선택 목록·목표 비중을 bucket_main의 주간 로그와 대조
  3) 매주 불변식: 버킷당 1종목, N개 이하, 유지 규칙(N+5·버킷 안 2위), 채우기 순서, 상한, 현금
  4) 미리보기 결정(2026-10-12) 재계산
  5) 앞보기 교란 시험: 결정일 이후 가격·PDF·SOXX를 망가뜨려도 bucket_main 결정이 같아야 함
  6) 견고성(감사자 사후, 선택 근거 아님): 재조정 요일 이동, 상위 기여 종목 제외, 주간 초과수익 t값
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py verify_bucket_main.py
"""
import copy, json, os, sys
from bisect import bisect_left
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, ROOT)
from desk.strategies import us_weights, sector_of, subsector_of, UNKNOWN  # noqa: E402
from desk import config  # noqa: E402
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402
from gbrc.etf.market import yahoo_ticker  # noqa: E402

FROZEN = os.path.join(HERE, "../cache/deskx_prices")
MAIN = os.path.join(HERE, "../results/bucket/main.json")
SOXX_SRC = os.path.join(ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet")
OUT = os.environ.get("VERIFY_BUCKET_OUT")   # 지정하면 그 경로에 JSON 저장(결과 폴더에는 쓰지 않음)
START = "2026-01-02"
SEMIS = {"AI 반도체", "메모리", "반도체 장비", "반도체 장비(적자)", "반도체(적자)", "반도체 소재"}
MERGE = {"AI 반도체": "반도체 칩", "반도체(적자)": "반도체 칩",
         "반도체 장비": "반도체 장비·소재", "반도체 장비(적자)": "반도체 장비·소재", "반도체 소재": "반도체 장비·소재"}
assert SEMIS == set(config.SEMIS_SECTORS)


def semis_like(sec):
    return sec in SEMIS or sec == UNKNOWN


# ------------------------------------------------------------------ 자료
class Data:
    def __init__(self):
        rd = lambda n: pd.read_csv(os.path.join(FROZEN, n), index_col=0, parse_dates=True)
        adj, opn, cls = rd("prices_adj_close.csv"), rd("prices_open.csv"), rd("prices_close.csv")
        self.days = [d.strftime("%Y-%m-%d") for d in adj["QQQ"].dropna().index]
        fix = lambda df: df.set_axis(df.index.strftime("%Y-%m-%d")).reindex(self.days)
        self.adj, self.opn, self.cls = fix(adj), fix(opn), fix(cls)
        t, kq = load_fund_snapshots("456600", "9999-12-31"), load_fund_snapshots("2ETFQ1", "9999-12-31")
        self.tw = {d: us_weights(s) for d, s in t.items()}
        self.kw = {d: us_weights(s) for d, s in kq.items()}
        self.tp, self.kp = sorted(self.tw), sorted(self.kw)
        keys = {k for w in list(self.tw.values()) + list(self.kw.values()) for k in w}
        self.tk = {k: yahoo_ticker(k) for k in keys}
        self.sec = {k: sector_of(k, self.tk[k]) for k in keys}
        self.sub = {k: subsector_of(k, self.tk[k]) for k in keys}
        o = pd.read_parquet(SOXX_SRC, columns=["date", "sym", "Close"])
        o = o[o.sym == "SOXX"]
        s = pd.Series(o["Close"].astype(float).to_numpy(), index=pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d"))
        self.soxx = s[~s.index.duplicated()].sort_index()
        self.arr = {}

    def a(self, frame, key):
        tk = self.tk[key]
        if (frame, tk) not in self.arr:
            df = getattr(self, frame)
            self.arr[(frame, tk)] = df[tk].to_numpy(float) if tk in df.columns else None
        return self.arr[(frame, tk)]

    def pdf_before(self, dates, session):
        i = bisect_left(dates, session)
        return dates[i - 1] if i > 0 else None

    def bucket(self, key, gran):
        if gran == "name":
            return "N:" + key
        s = MERGE.get(self.sec[key], self.sec[key]) if gran == "sector" else self.sub[key]
        return "미분류:" + self.tk[key] if s == UNKNOWN else s

    def soxx_gate(self, k):
        """k일 시가 결정: k-1일(마지막 결정 종가) SOXX 종가 < 직전 20개 종가 평균."""
        s = self.soxx[self.soxx.index <= self.days[k - 1]]
        if len(s) < 20:
            return False
        assert s.index[-1] == self.days[k - 1], ("SOXX 날짜 어긋남", s.index[-1], self.days[k - 1])
        return bool(s.iloc[-1] < s.iloc[-20:].mean())


# ------------------------------------------------------------------ 규칙(독립 구현)
def candidates(D, U, session):
    tp = D.pdf_before(D.tp, session)
    tw = D.tw[tp] if tp else {}
    if U == "T":
        return {k: v for k, v in tw.items() if v > 0}
    kp = D.pdf_before(D.kp, session)
    kw = D.kw[kp] if kp else {}
    out = {}
    for k in set(tw) | set(kw):
        v = tw.get(k, 0.0) if tw.get(k, 0.0) > 0 else kw.get(k, 0.0)
        if v > 0:
            out[k] = v
    return out


def rank(D, U, k, session, L):
    if k - 1 - L < 0:
        return []
    out = []
    for key, w in sorted(candidates(D, U, session).items()):
        px = D.a("adj", key)
        if px is None:
            continue
        p1, p0 = px[k - 1], px[k - 1 - L]
        if np.isfinite(p1) and np.isfinite(p0) and p0 > 0:
            out.append((key, p1 / p0 - 1, w))
    return sorted(out, key=lambda r: -r[1])


def choose(D, ranked, held, N, gran, buf=5):
    mom = {k: m for k, m, _ in ranked}
    pos = {k: i for i, (k, _, _) in enumerate(ranked)}
    groups = {}
    for k, _, _ in ranked:
        groups.setdefault(D.bucket(k, gran), []).append(k)
    for b in groups:
        groups[b].sort(key=lambda k: pos[k])
    leaders = sorted(groups, key=lambda b: pos[groups[b][0]])
    lrank = {b: i + 1 for i, b in enumerate(leaders)}
    ok = [k for k in held if k in mom and lrank[D.bucket(k, gran)] <= N + buf
          and groups[D.bucket(k, gran)].index(k) <= 1]
    ok.sort(key=lambda k: pos[k])
    kept, used = [], set()
    for k in ok:
        b = D.bucket(k, gran)
        if b not in used and len(kept) < N:
            kept.append(k)
            used.add(b)
    sel = list(kept)
    for b in leaders:
        if len(sel) == N:
            break
        if b not in used:
            sel.append(groups[b][0])
            used.add(b)
    return sel, kept, lrank, groups


def freeze_time(rates, caps_before, cap):
    """Σ rate_i·min(t, f_i) = cap를 푸는 t(없으면 inf). 구간 선형이라 꺾이는 점을 따라 정확히 푼다."""
    if cap <= 0:
        return 0.0
    pts = sorted(set([0.0] + [f for f in caps_before if np.isfinite(f)]))
    g = lambda t: sum(r * min(t, f) for r, f in zip(rates, caps_before))
    for lo, hi in zip(pts, pts[1:] + [np.inf]):
        slope = sum(r for r, f in zip(rates, caps_before) if f > lo)
        if slope <= 0:
            continue
        t = lo + (cap - g(lo)) / slope
        if t <= hi + 1e-15:
            return t
    return np.inf


def caps(D, raw, stock_cap, sector_cap=0.60, semis_cap=None):
    names = [k for k in raw if raw[k] > 0]
    if not names:
        return {}
    f = {k: stock_cap / raw[k] for k in names}
    for g in {D.sec[k] for k in names}:
        mem = [k for k in names if D.sec[k] == g]
        tg = freeze_time([raw[k] for k in mem], [f[k] for k in mem], sector_cap)
        for k in mem:
            f[k] = min(f[k], tg)
    if semis_cap is not None:
        mem = [k for k in names if semis_like(D.sec[k])]
        if mem:
            ts = freeze_time([raw[k] for k in mem], [f[k] for k in mem], semis_cap)
            for k in mem:
                f[k] = min(f[k], ts)
    t = freeze_time([raw[k] for k in names], [f[k] for k in names], sum(raw[k] for k in names))
    out = {k: raw[k] * min(t, f[k]) for k in names}
    return {k: v for k, v in out.items() if v > 1e-12}


def decide(D, v, k, session, held):
    ranked = rank(D, v["U"], k, session, v["L"])
    if not ranked:
        return None
    sel, kept, lrank, groups = choose(D, ranked, held, v["N"], v["G"])
    if v["W"] == "eq":
        raw = {k_: 1.0 / v["N"] for k_ in sel}
    else:
        wm = {k_: w for k_, _, w in ranked}
        tot = sum(wm[k_] for k_ in sel)
        raw = {k_: wm[k_] / tot for k_ in sel}
    tgt = caps(D, raw, v.get("stock", max(0.25, 1.0 / v["N"])), 0.60, v.get("semis"))
    gate = False
    if v.get("gate"):
        gate = D.soxx_gate(k)
        if gate:
            tgt = {k_: w for k_, w in tgt.items() if not semis_like(D.sec[k_])}
    return dict(session=session, sel=sel, kept=kept, tgt=tgt, gate=gate, ranked=ranked, lrank=lrank, groups=groups)


def iso(d):
    return date.fromisoformat(d).isocalendar()[:2]


def simulate(D, v, offset=0, exclude=()):
    """offset=0: ISO 주 첫 거래일 결정. offset=j: 그 주 j+1번째 거래일(없으면 그 주 마지막 거래일)."""
    k0 = next(i for i, d in enumerate(D.days) if d >= START)
    weeks = {}
    for i in range(k0, len(D.days)):
        weeks.setdefault(iso(D.days[i]), []).append(i)
    reb_days = {w[min(offset, len(w) - 1)] for w in weeks.values()}
    V, W, logs, w, held, started = [1.0], [], [], {}, [], False
    for k in range(k0, len(D.days)):
        if k > k0:
            r = {}
            for key in w:
                o = D.a("opn", key)
                r[key] = o[k] / o[k - 1] - 1 if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                    and o[k - 1] > 0 else 0.0
            g = sum(x * r[key] for key, x in w.items())
            tot = 1 + g
            w = {key: x * (1 + r[key]) / tot for key, x in w.items()}
            V.append(V[-1] * tot)
        if (not started) or k in reb_days:
            dec = decide_excl(D, v, k, D.days[k], held, exclude) if exclude else decide(D, v, k, D.days[k], held)
            if dec is not None:
                started = True
                w, held = dict(dec["tgt"]), list(dec["sel"])
                logs.append(dec)
        W.append(dict(w))
    return pd.Series(V, index=D.days[k0:]), W, logs


def decide_excl(D, v, k, session, held, exclude):
    saved = D.tw, D.kw
    try:
        D.tw = {d: {x: y for x, y in w.items() if D.tk[x] not in exclude} for d, w in saved[0].items()}
        D.kw = {d: {x: y for x, y in w.items() if D.tk[x] not in exclude} for d, w in saved[1].items()}
        return decide(D, v, k, session, held)
    finally:
        D.tw, D.kw = saved


def period(V, a, b):
    """deskx.evaluate와 같은 약속: a~b 날짜 시가에서 시작하는 하루 수익을 곱한다(마지막 날 다음 시가가 없으면 그날 제외)."""
    idx = list(V.index)
    i = next(j for j, d in enumerate(idx) if d >= a)
    j = max(j for j, d in enumerate(idx) if d <= b)
    j = min(j + 1, len(idx) - 1)
    seg = V.iloc[i:j + 1].to_numpy()
    curve = seg / seg[0]
    dd_desk = (curve[1:] / np.maximum.accumulate(curve[1:]) - 1).min()       # deskx 약속(첫 수익 뒤부터)
    dd_std = (curve / np.maximum.accumulate(curve) - 1).min()                 # 시작점 포함
    return float(curve[-1] - 1), float(dd_desk), float(dd_std)


def metrics(D, V, W, logs, gran):
    f, b, s, a = period(V, START, "2026-05-29"), period(V, "2026-06-01", "2026-10-09"), \
        period(V, "2026-09-01", "2026-10-09"), period(V, START, "2026-10-09")
    semis = np.mean([sum(x for k, x in w.items() if semis_like(D.sec[k])) for w in W])
    nb = np.mean([len({D.bucket(k, gran) for k, x in w.items() if x > 1e-12}) for w in W])
    turn = np.mean([len(set(y["sel"]) - set(x["sel"])) for x, y in zip(logs, logs[1:])])
    months = {}
    for m in range(6, 11):
        a_, b_ = f"2026-{m:02d}-01", f"2026-{m:02d}-31"
        months[f"2026-{m:02d}"] = period(V, a_, b_)[0]
    return dict(front=f[0], back=b[0], sepoct=s[0], full=a[0], mdd=a[1], mdd_std=a[2], semis_avg=float(semis),
                buckets_avg=float(nb), turnover=float(turn), months=months)


def top_share(D, V, W):
    vals, by = V.to_numpy(), {}
    k0 = D.days.index(V.index[0])
    for i in range(len(W) - 1):
        k = k0 + i + 1
        for key, x in W[i].items():
            o = D.a("opn", key)
            r = o[k] / o[k - 1] - 1 if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0 \
                else 0.0
            by[D.tk[key]] = by.get(D.tk[key], 0.0) + vals[i] * x * r
    gain = vals[-1] - 1
    top = sorted(by.items(), key=lambda z: -z[1])
    return sum(c for _, c in top[:3]) / gain, [t for t, _ in top[:3]], top


# ------------------------------------------------------------------ 실행
def grid():
    out = []
    for U in ("T", "TK"):
        for G in ("sector", "sub"):
            for N in (3, 5, 7, 10):
                for Wm in ("eq", "fund"):
                    out.append((f"{U}-{G}-N{N}-{Wm}", dict(U=U, G=G, N=N, W=Wm, L=120)))
    return out


def main():
    D = Data()
    M = json.load(open(MAIN, encoding="utf-8"))
    report = {"diffs": {}, "decision_mismatch": {}, "invariant_fail": [], "dumps": {}, "robust": {}}

    # 1) live·#774(종목 하나가 한 버킷 = 데스크 select) 와 QQQ
    live_v = dict(U="T", G="name", N=7, W="fund", L=120, stock=0.25, semis=0.50)
    c774_v = dict(live_v, semis=0.60)
    from desk.strategies import load_book, run_model  # 데스크 경로와도 맞춰 본다(가격은 고정본으로 바꿔 끼움)
    from desk import marketdata
    files = {f: f"prices_{f.replace(' ', '_').lower()}.csv" for f in marketdata.FIELDS}
    marketdata.history = lambda tickers, force=False: {
        f: pd.read_csv(os.path.join(FROZEN, n), index_col=0, parse_dates=True) for f, n in files.items()}
    book = load_book()
    ref = {}
    ref["live"] = np.asarray(run_model(book, start=START).value)
    from desk.validate import codex_caps
    with codex_caps():
        ref["c774"] = np.asarray(run_model(book, start=START).value)
    bench = {}
    for nm, v, js in (("live", live_v, "live"), ("c774", c774_v, "c774")):
        V, W, logs = simulate(D, v)
        bench[nm] = (V, W, logs)
        mt = metrics(D, V, W, logs, "sector")
        report["diffs"][f"{nm}_vs_run_model"] = float(np.max(np.abs(V.to_numpy() - ref[nm])))
        for f_ in ("front", "back", "sepoct", "full", "mdd"):
            report["diffs"][f"{nm}.{f_}"] = abs(mt[f_] - M["benchmark"][js][f_])
        report[nm] = mt
    q = D.opn["QQQ"].to_numpy(float)
    k0 = D.days.index(START)
    Vq = pd.Series(q[k0:] / q[k0], index=D.days[k0:])
    report["qqq"] = {lab: period(Vq, a, b)[0] for lab, a, b in (("front", START, "2026-05-29"),
                                                               ("back", "2026-06-01", "2026-10-09"),
                                                               ("full", START, "2026-10-09"))}
    report["qqq"]["mdd"] = period(Vq, START, "2026-10-09")[1]

    # 2) 32개
    js = {r["name"]: r for r in M["variants"]}
    sys.path.insert(0, HERE)
    from bucket_main import BucketEngine, GRID  # 결정 대조용으로만 읽음
    E = BucketEngine()
    bgrid = dict(GRID)
    res = {}
    for name, v in grid():
        V, W, logs = simulate(D, v)
        mt = metrics(D, V, W, logs, v["G"])
        mt["top3_share"], mt["top3"], _ = top_share(D, V, W)
        res[name] = (v, V, W, logs, mt)
        for f_ in ("front", "back", "sepoct", "full", "mdd", "semis_avg", "buckets_avg", "turnover", "top3_share"):
            report["diffs"][f"{name}.{f_}"] = abs(mt[f_] - js[name][f_])
        # 주간 결정 대조
        _, _, blog = E.run_logged(bgrid[name])
        bad = []
        if len(blog) != len(logs):
            bad.append(("주 수", len(blog), len(logs)))
        for x, y in zip(blog, logs):
            if x["session"] != y["session"] or list(x["keep"]) != y["sel"]:
                bad.append((y["session"], [D.tk[k] for k in x["keep"]], [D.tk[k] for k in y["sel"]]))
        report["decision_mismatch"][name] = bad[:5]
        # 불변식
        for y in logs:
            N, b = v["N"], [D.bucket(k, v["G"]) for k in y["sel"]]
            pos = {k: i for i, (k, _, _) in enumerate(y["ranked"])}
            t = y["tgt"]
            cond = {
                "one_per_bucket": len(set(b)) == len(b),
                "<=N": len(y["sel"]) <= N,
                "fill_when_possible": len(y["sel"]) == min(N, len(y["lrank"])),
                "kept_rule": all(y["lrank"][D.bucket(k, v["G"])] <= N + 5 and y["groups"][D.bucket(k, v["G"])].index(k) <= 1
                                 for k in y["kept"]),
                "fill_are_leaders": all(y["groups"][D.bucket(k, v["G"])][0] == k for k in y["sel"][len(y["kept"]):]),
                "fill_order": [y["lrank"][D.bucket(k, v["G"])] for k in y["sel"][len(y["kept"]):]]
                == sorted(y["lrank"][D.bucket(k, v["G"])] for k in y["sel"][len(y["kept"]):]),
                "stock_cap": all(x <= max(0.25, 1 / N) + 1e-9 for x in t.values()),
                "sector_cap": all(sum(x for k, x in t.items() if D.sec[k] == s) <= 0.60 + 1e-9
                                  for s in {D.sec[k] for k in t}),
                "sum<=1": sum(t.values()) <= 1 + 1e-9,
            }
            if not all(cond.values()):
                report["invariant_fail"].append((name, y["session"], [c for c, ok in cond.items() if not ok]))
    # 앞 최선 다시 고르기
    best = max(res, key=lambda n: res[n][4]["front"])
    live = report["live"]
    report["best_by_front"] = best
    report["beats_live_both"] = [n for n in res if res[n][4]["front"] > live["front"] and res[n][4]["back"] > live["back"]]
    report["higher_total_lower_mdd"] = [n for n in res if res[n][4]["full"] > live["full"] and res[n][4]["mdd"] > live["mdd"]]

    # 사후 a/b/c(앞 최선) 대조
    bv = res[best][0]
    post = {"a": dict(bv, semis=0.50), "b": dict(bv, gate=True), "c90": dict(bv, L=90), "c150": dict(bv, L=150),
            "ctrl_live_gate": dict(live_v, gate=True)}
    for (lab, v), pj in zip(post.items(), M["posthoc"]):
        V, W, logs = simulate(D, v)
        mt = metrics(D, V, W, logs, v["G"] if v["G"] != "name" else "sector")
        mt["gate_weeks"] = sum(1 for y in logs if y["gate"])
        report[f"post_{lab}"] = mt
        for f_ in ("front", "back", "sepoct", "full", "mdd"):
            report["diffs"][f"post_{lab}.{f_}"] = abs(mt[f_] - pj[f_])

    # 3) 결정 덤프: 앞 최선과 TK-sub-N7-fund, 5개 날짜
    for name in (best, "TK-sub-N7-fund"):
        v, V, W, logs, _ = res[name]
        idx = [0, 9, 21, 30, len(logs) - 1]
        dump = []
        for i in idx:
            y = logs[i]
            prev = logs[i - 1]["sel"] if i else []
            dump.append({"session": y["session"],
                         "pdf_time": D.pdf_before(D.tp, y["session"]), "pdf_koact": D.pdf_before(D.kp, y["session"]),
                         "decision_close": D.days[D.days.index(y["session"]) - 1],
                         "leaders_top10": [(D.tk[y["groups"][b][0]], b, round(next(m for k, m, _ in y["ranked"]
                                                                              if k == y["groups"][b][0]), 3))
                                           for b in sorted(y["lrank"], key=y["lrank"].get)[:10]],
                         "held_last_week": [(D.tk[k], D.bucket(k, v["G"]),
                                             y["lrank"].get(D.bucket(k, v["G"])),
                                             (y["groups"].get(D.bucket(k, v["G"]), []).index(k) + 1)
                                             if k in y["groups"].get(D.bucket(k, v["G"]), []) else None) for k in prev],
                         "kept": [D.tk[k] for k in y["kept"]], "filled": [D.tk[k] for k in y["sel"][len(y["kept"]):]],
                         "weights": {D.tk[k]: round(x, 4) for k, x in sorted(y["tgt"].items(), key=lambda z: -z[1])},
                         "cash": round(1 - sum(y["tgt"].values()), 4),
                         "sectors": {D.tk[k]: D.sec[k] for k in y["tgt"]}})
        report["dumps"][name] = dump

    # 4) 2026-10-12 미리보기 재계산(앞 최선)
    v, V, W, logs, _ = res[best]
    nxt = decide(D, v, len(D.days), "2026-10-12", logs[-1]["sel"])
    report["preview"] = {D.tk[k]: round(x, 4) for k, x in nxt["tgt"].items()}
    report["preview_buckets"] = {D.tk[k]: D.bucket(k, v["G"]) for k in nxt["sel"]}
    report["preview_vs_main"] = {h["name"]: h["weight"] for h in M["next_rebalance_preview"]["holdings"]}

    # 5) 앞보기 교란 시험(bucket_main 엔진 자체에 대해)
    rng = np.random.default_rng(7)
    la = []
    for name in (best, "TK-sub-N7-fund"):
        p = bgrid[name]
        _, _, blog = E.run_logged(p)
        for i in (3, 12, 25, 36):
            sess = blog[i]["session"]
            k = E.days.index(sess)
            cur = set(blog[i - 1]["keep"])
            base = E.target(p, k, sess, cur)[:2]
            E2 = copy.copy(E)
            E2.book = copy.copy(E.book)
            for fr in ("adj", "opn", "cls"):
                df = getattr(E.book, fr).copy()
                df.iloc[k:] = df.iloc[k:].to_numpy() * rng.uniform(0.2, 5.0, size=df.iloc[k:].shape)
                setattr(E2.book, fr, df)
            E2.tw = {d: ({x: rng.uniform(0.001, 0.2) for x in w} if d >= sess else w) for d, w in E.tw.items()}
            E2.kw = {d: ({x: rng.uniform(0.001, 0.2) for x in w} if d >= sess else w) for d, w in E.kw.items()}
            E2.book._w = {}
            E2.soxx = E.soxx.copy()
            E2.soxx[E2.soxx.index >= sess] *= 0.3
            E2.soxx_sma20 = E2.soxx.rolling(20, min_periods=20).mean()
            pg = p.__class__(**{**p.__dict__, "soxx_gate": True})
            base_g = E.target(pg, k, sess, cur)[:2]
            pert = E2.target(p, k, sess, cur)[:2]
            pert_g = E2.target(pg, k, sess, cur)[:2]
            la.append((name, sess, base[0] == pert[0] and base[1] == pert[1],
                       base_g[0] == pert_g[0] and base_g[1] == pert_g[1]))
    report["lookahead_perturbation"] = la

    # 6) 견고성(감사자 사후 — 선택 근거 아님)
    rob = {}
    for off in range(5):
        row = {}
        for nm, v in (("live", live_v), ("pick", bv)):
            V, W, logs = simulate(D, v, offset=off)
            mt = metrics(D, V, W, logs, "sector")
            row[nm] = {f_: round(mt[f_], 4) for f_ in ("front", "back", "full", "mdd")}
        rob[f"rebalance_day_{off + 1}"] = row
    for ex in (("SNDK",), ("DELL",), ("SNDK", "DELL")):
        row = {}
        for nm, v in (("live", live_v), ("pick", bv)):
            V, W, logs = simulate(D, v, exclude=ex)
            mt = metrics(D, V, W, logs, "sector")
            row[nm] = {f_: round(mt[f_], 4) for f_ in ("front", "back", "full", "mdd")}
        rob["exclude_" + "+".join(ex)] = row
    Vl, Vp = bench["live"][0], res[best][1]
    rl, rp = Vl.pct_change().dropna(), Vp.pct_change().dropna()
    wk = pd.Series([iso(d) for d in rl.index], index=rl.index)
    ex_w = ((1 + rp).groupby(wk.values).prod() - (1 + rl).groupby(wk.values).prod())
    back_w = ex_w[[w >= iso("2026-06-01") for w in ex_w.index]]
    front_w = ex_w[[w < iso("2026-06-01") for w in ex_w.index]]
    rob["weekly_excess_pick_minus_live"] = {
        "front_mean": float(front_w.mean()), "front_t": float(front_w.mean() / front_w.std() * np.sqrt(len(front_w))),
        "back_mean": float(back_w.mean()), "back_t": float(back_w.mean() / back_w.std() * np.sqrt(len(back_w))),
        "weeks_front": len(front_w), "weeks_back": len(back_w),
        "share_weeks_pick_ahead_back": float((back_w > 0).mean())}
    # 앞 최선이 앞 구간에서 '우연히 1등'일 정도: 32개의 앞 수익 분포
    fr = sorted((res[n][4]["front"] for n in res), reverse=True)
    rob["front_top5"] = [round(x, 4) for x in fr[:5]]
    report["robust"] = rob

    # 표
    mx = max(report["diffs"].values())
    print("main.json·run_model 대비 최대 절대차:", mx)
    print("  가장 큰 5개:", sorted(report["diffs"].items(), key=lambda z: -z[1])[:5])
    print("주간 결정 불일치(변형별 처음 5개):", {k: v for k, v in report["decision_mismatch"].items() if v} or "없음")
    print("불변식 위반:", report["invariant_fail"][:10] or "없음")
    print("앞 최선:", best, "| 앞·뒤 모두 live 초과:", report["beats_live_both"],
          "| 전체↑·MDD↓:", report["higher_total_lower_mdd"] or "없음")
    print("live:", {k: round(x, 4) for k, x in report["live"].items() if k != "months"}, "QQQ:", report["qqq"])
    print("앞보기 교란(이름, 날짜, 기본 같음, 게이트 같음):", la)
    print("미리보기(독립):", report["preview"], report["preview_buckets"])
    print("미리보기(main.json):", report["preview_vs_main"])
    for name, dump in report["dumps"].items():
        print(f"\n== {name} 결정 덤프 ==")
        for d in dump:
            print(json.dumps(d, ensure_ascii=False))
    print("\n== 견고성(감사자 사후) ==")
    for k, v in rob.items():
        print(k, v)
    if OUT:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=str)
        print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
