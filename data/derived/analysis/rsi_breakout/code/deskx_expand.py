"""데스크 7종목 규칙의 후보군 넓히기: KoAct 미국 보유 + 비미국 보유의 미국 대리 종목(2026-10-10, 연구용 스크래치).

질문(사용자): 세부 섹터로 나눠 담는다면 KoAct 종목도 필요하면 넣고, 삼성전기 같은 국장·해외 종목은
            같은 세부 섹터의 미국 대장으로 바꿔 사면 어떤가(AI칩 패키징, 데이터센터 등).
후보군 U(재조정일 t, t보다 앞선 날짜의 최신 PDF만 사용):
  'T'  = TIME 미국 보유(지금 데스크),  'TK' = TIME ∪ KoAct 미국 보유.
  대리 'off' | 'on': on이면 그 PDF들의 비미국 보유마다 AI_UNIVERSE 세부 섹터의 미국 상장 종목을 대리로 넣는다
      (세부 섹터에 미국 종목이 없으면 그 세부 섹터가 속한 AI_UNIVERSE 대분류의 미국 종목, 그래도 없으면 버림).
      대리 종목은 펀드가 들고 있지 않아도 후보가 된다.
기본 비중(상한 전 재료): 미국 보유는 TIME에 있으면 TIME NAV 비중, 없으면 KoAct NAV 비중.
      비미국 보유의 NAV 비중은 그 대리 종목들에 똑같이 나눠 더한다.
선택: 120거래일 수정종가 수익 순위, 지난주 종목은 상위 12 안이면 유지, 세부 섹터 그룹당 최대 m종목
      (m ∈ {1, 2, 없음}; 그룹 = desk.strategies.subsector_of, '미분류'는 종목마다 따로), 위에서부터 7개 채움.
      선택 코드는 deskx_subcount.SubEngine.limited_select를 그대로 쓴다.
상한: 종목 25%, 관심 섹터 60%, 반도체+미분류 예산 50%(config.RULE 그대로). 매주 재조정, 시가→시가 USD, 수수료 0.
격자(사전 등록 12개): {T, TK} × {대리 off, on} × m {1, 2, 없음}. T/off/없음은 지금 데스크 경로와 1e-12 안에서 같아야 한다.
선택 규칙: 앞 1/2~5/29 수익만으로 최선을 고르고 그 뒤 6/1~10/9를 보고한다.
대리 종목 가격: 동결 가격판(deskx_prices)에 없는 대리 종목만 yfinance(auto_adjust=False, 2025-06-01~2026-10-10)로
      받아 ../cache/proxy_prices/에 저장하고 E.book.days에 맞춘다. 데스크 캐시는 건드리지 않는다.
비미국 보유 → AI_UNIVERSE 심볼: yahoo_ticker(key). 국내 종목은 .KS로 나오므로 .KQ도 시도한다.
      같은 회사의 다른 상장은 watchlist 주석의 '이력이 긴 쪽' 원칙대로 미국 상장으로 잇는다(CROSS_LISTING, 알리바바 9988→BABA).
사후 위험: AI_UNIVERSE(곧 대리 명단)는 2026년 10월에 만든 목록이다. 대리 'on' 결과에는 뒤늦게 고른 명단의 이점이 섞인다.
      관심 섹터 라벨도 지금 watchlist.py 값을 전 기간에 쓴다(데스크와 같은 방식).
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py deskx_expand.py
"""
import json, os, sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, PERIODS, START  # noqa: E402
from deskx_subcount import SubEngine, diversification  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import (project_caps, run_model, latest_before, research_key, sector_of, in_semis_budget,  # noqa: E402
                             UNKNOWN)
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402
from gbrc.etf.market import is_cash_holding, listing_market, yahoo_ticker  # noqa: E402
from gbrc.etf.research_data import exclusion_reason  # noqa: E402
from watchlist import AI_UNIVERSE, resolve_subsector  # noqa: E402

OUT = os.path.join(HERE, "../results/deskx/expand.json")
PROXY_CACHE = os.path.join(HERE, "../cache/proxy_prices")
FETCH_START, FETCH_END = "2025-06-01", "2026-10-10"
FIELDS = ("Adj Close", "Close", "Open")
FRAME = {"Adj Close": "adj", "Close": "cls", "Open": "opn"}
FRONT, BACK, SEP, FULL = (p[0] for p in PERIODS)
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
NEXT_SESSION = "2026-10-12"
TIME_CODE, KOACT_CODE = "456600", "2ETFQ1"
CROSS_LISTING = {"9988.HK": "BABA"}       # 같은 회사의 홍콩 상장 → AI_UNIVERSE에 있는 미국 상장

US_SYMBOLS = {s for g in AI_UNIVERSE.values() for ss in g.values() for s in ss if "." not in s}
SUB_US = {sub: [s for s in syms if s in US_SYMBOLS] for g in AI_UNIVERSE.values() for sub, syms in g.items()}
TOP_OF = {sub: top for top, g in AI_UNIVERSE.items() for sub in g}
TOP_US = {top: [s for ss in g.values() for s in ss if s in US_SYMBOLS] for top, g in AI_UNIVERSE.items()}


@dataclass(frozen=True)
class ExpParams(Params):
    univ: str = "T"          # 'T' | 'TK'
    proxies: bool | str = False   # False | True(대리 전부에 똑같이 나눔) | 'leader'(사후: 120일 수익 1위 대리 하나에 전부)
    m: int = 7               # 7 = 한도 없음


GRID = [(f"{u}/{'on' if px else 'off'}/m={'none' if m == 7 else m}", ExpParams(univ=u, proxies=px, m=m, semis_cap=0.50))
        for u in ("T", "TK") for px in (False, True) for m in (1, 2, 7)]
LIVE_NAME = "T/off/m=none"
# 사후 아이디어 1개(결과를 본 뒤 추가, 선택 근거 아님): 사용자의 '대장을 산다'를 그대로 — 비미국 보유의 비중을
# 대리 종목 중 k-1일 120일 수익 1위 하나에 전부 준다. 격자에서 앞 기준 최선이던 m=1, TIME 후보군 위에 얹는다.
POSTHOC = ("[사후] T/leader/m=1", ExpParams(univ="T", proxies="leader", m=1, semis_cap=0.50))


def resolve_symbol(key):
    """비미국 보유 키 → (AI_UNIVERSE 심볼 또는 원래 Yahoo 심볼, 세부 섹터 또는 None)."""
    yt = yahoo_ticker(key)
    if yt is None:
        return None, None
    tries = [CROSS_LISTING.get(yt, yt)]
    if yt.endswith(".KS"):
        tries.append(yt[:-3] + ".KQ")
    for s in tries:
        sub = resolve_subsector(s)
        if sub:
            return s, sub
    return yt, None


def proxies_for(sub):
    """세부 섹터의 미국 상장 종목, 없으면 그 대분류의 미국 상장 종목, 그래도 없으면 []."""
    if not sub:
        return [], None
    if SUB_US.get(sub):
        return list(SUB_US[sub]), "sub"
    us = TOP_US.get(TOP_OF.get(sub), [])
    return (list(us), "top") if us else ([], None)


def non_us_weights(snapshot):
    out = {}
    for row in snapshot["holdings"]:
        if is_cash_holding(row) or exclusion_reason(row):
            continue
        key = research_key(row.get("code"))
        if listing_market(key) in (None, "US"):
            continue
        out[key] = out.get(key, 0.0) + float(row["weight"]) / 100.0
    return out


def fetch_proxy_prices(tickers, days):
    """대리 종목 일봉(Open/Close/Adj Close, 분할 반영·배당 미반영 시가) → {field: DataFrame(days × tickers)}."""
    os.makedirs(PROXY_CACHE, exist_ok=True)
    meta_path = os.path.join(PROXY_CACHE, "meta.json")
    paths = {f: os.path.join(PROXY_CACHE, f"prices_{f.replace(' ', '_').lower()}.csv") for f in FIELDS}
    have = []
    if os.path.exists(meta_path) and all(os.path.exists(p) for p in paths.values()):
        with open(meta_path, encoding="utf-8") as fh:
            have = json.load(fh).get("tickers", [])
    if not set(tickers) <= set(have):
        import yfinance as yf
        frame = yf.download(sorted(tickers), start=FETCH_START, end=FETCH_END, auto_adjust=False,
                            progress=False, threads=True)
        for f in FIELDS:
            df = frame[f]
            if isinstance(df, pd.Series):
                df = df.to_frame(sorted(tickers)[0])
            df.to_csv(paths[f])
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"tickers": sorted(tickers), "start": FETCH_START, "end": FETCH_END,
                       "source": "yfinance auto_adjust=False"}, fh)
    out = {}
    for f in FIELDS:
        df = pd.read_csv(paths[f], index_col=0, parse_dates=True)
        df.index = df.index.strftime("%Y-%m-%d")
        out[f] = df.reindex(days)[[t for t in tickers if t in df.columns]]
    return out


class ExpandEngine(SubEngine):
    def __init__(self):
        super().__init__()
        self.base_ticker = dict(self.ticker)
        ksnaps = load_fund_snapshots(KOACT_CODE, "9999-12-31")
        self.nonus = {"T": {d: non_us_weights(s) for d, s in self.book.snaps.items()},
                      "K": {d: non_us_weights(s) for d, s in ksnaps.items()}}
        # 비미국 보유 → 대리 미국 종목
        self.proxy_map, self.proxy_info = {}, {}
        for fund in self.nonus.values():
            for w in fund.values():
                for key in w:
                    if key in self.proxy_map:
                        continue
                    sym, sub = resolve_symbol(key)
                    prox, how = proxies_for(sub)
                    self.proxy_map[key] = prox
                    self.proxy_info[key] = {"symbol": sym, "subsector": sub, "proxies": prox,
                                            "via": how or "dropped"}
        by_ticker = {t: k for k, t in self.ticker.items()}
        self.proxy_key = {}
        new = []
        for prox in self.proxy_map.values():
            for s in prox:
                if s in self.proxy_key:
                    continue
                k = by_ticker.get(s, f"{s} US EQUITY")
                self.proxy_key[s] = k
                if k not in self.ticker:
                    self.ticker[k] = s
                    self.sector[k] = sector_of(k, s)
                    new.append(s)
        need = sorted({s for s in self.proxy_key if s not in self.book.adj.columns})
        self.fetched = need
        if need:
            px = fetch_proxy_prices(need, self.days)
            for f, attr in FRAME.items():
                df = getattr(self.book, attr)
                add = px[f][[c for c in px[f].columns if c not in df.columns]]
                setattr(self.book, attr, pd.concat([df, add], axis=1))
        self._src = {}
        self._from = {}

    # ------------------------------------------------------------ 후보
    def candidates(self, p, session):
        tpdf = latest_before(self.book.pdfs, session)
        tw = self.tw.get(tpdf, {}) if tpdf else {}
        base = {k: v for k, v in tw.items() if v > 0}
        src = dict.fromkeys(base, "time")
        kpdf = None
        if getattr(p, "univ", "T") == "TK":
            kpdf = latest_before(self.kpdfs, session)
            for k, v in (self.kw.get(kpdf, {}) if kpdf else {}).items():
                if v > 0 and k not in base:
                    base[k] = v
                    src[k] = "koact"
        frm = defaultdict(list)
        if getattr(p, "proxies", False):
            nonus = dict(self.nonus["T"].get(tpdf, {}) if tpdf else {})
            if kpdf:
                for k, v in self.nonus["K"].get(kpdf, {}).items():
                    nonus.setdefault(k, v)          # TIME에 있으면 TIME 비중, 없으면 KoAct 비중
            leader = p.proxies == "leader"
            kk = bisect_left(self.days, session)       # session이 days에 없으면(미리보기) len(days)
            for k, v in nonus.items():
                prox = self.proxy_map.get(k, [])
                if leader and prox:
                    mom = {s: self.momentum(self.proxy_key[s], kk, p.lookback) for s in prox}
                    mom = {s: x for s, x in mom.items() if np.isfinite(x)}
                    prox = [max(mom, key=mom.get)] if mom else []
                for s in prox:
                    pk = self.proxy_key[s]
                    base[pk] = base.get(pk, 0.0) + v / len(prox)
                    src.setdefault(pk, "proxy")
                    frm[pk].append(self.proxy_info[k]["symbol"])
        self._src, self._from = src, dict(frm)
        return base

    def momentum(self, key, k, lookback):
        px = self.col("adj", key)
        if px is None or k - 1 - lookback < 0:
            return np.nan
        p1, p0 = px[k - 1], px[k - 1 - lookback]
        return p1 / p0 - 1 if np.isfinite(p1) and np.isfinite(p0) and p0 > 0 else np.nan

    def target(self, p, k, session, current):
        ranked = self.ranking(p, k, session)
        if not ranked:
            return [], {}, ranked
        order = [r["key"] for r in ranked]
        keep = self.limited_select(order, current, p.n, p.buffer, p.m)
        wmap = {r["key"]: r["w"] for r in ranked}
        tot = sum(wmap[x] for x in keep)
        raw = {x: wmap[x] / tot for x in keep}
        saved = dict(config.RULE)
        try:
            config.RULE.update(stock_cap=p.stock_cap, sector_cap=p.sector_cap, semis_cap=p.semis_cap)
            tgt = project_caps(raw, self.sector)
        finally:
            config.RULE.clear()
            config.RULE.update(saved)
        if self.log is not None:
            rank = {x: i + 1 for i, x in enumerate(order)}
            self.log.append({"session": session, "keep": list(keep), "tgt": dict(tgt),
                             "src": {x: self._src.get(x, "?") for x in keep},
                             "from": {x: self._from.get(x, []) for x in keep},
                             "rank": {x: rank[x] for x in keep}})
        return keep, tgt, ranked


def contrib_by_name(E, v, rows):
    k0 = E.days.index(rows[0]["day"])
    vals = v.to_numpy()
    contrib = defaultdict(float)
    for i in range(len(rows) - 1):
        k = k0 + i + 1
        for key, wt in rows[i]["w"].items():
            o = E.col("opn", key)
            ret = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                and o[k - 1] > 0 else 0.0
            contrib[E.ticker[key]] += vals[i] * wt * ret
    gain = float(vals[-1] - 1)
    assert abs(sum(contrib.values()) - gain) < 1e-9 * max(1.0, abs(gain))
    return dict(contrib), gain


def one(E, name, p):
    E.log = []
    try:
        v, rows = E.run(p)
        log = E.log
    finally:
        E.log = None
    res = E.evaluate(name, v, rows)
    res["params"] = asdict(p)
    res.update(diversification(E, rows))
    contrib, gain = contrib_by_name(E, v, rows)
    top = sorted(contrib.items(), key=lambda x: -x[1])
    res["상위3"] = [t for t, _ in top[:3]]
    res["상위3비중"] = sum(c for _, c in top[:3]) / gain
    res["상위5기여"] = {t: round(c, 4) for t, c in top[:5]}
    # 출처별 비중: 각 날의 보유를 그 직전 재조정 때의 출처(time/koact/proxy)로 나눈다
    by_session = {e["session"]: e for e in log}
    cur, share = None, {"time": [], "koact": [], "proxy": []}
    for row in rows:
        if row["day"] in by_session:
            cur = by_session[row["day"]]
        for s in share:
            share[s].append(sum(wt for key, wt in row["w"].items() if cur and cur["src"].get(key) == s))
    res["KoAct전용비중"] = float(np.mean(share["koact"]))
    res["대리비중"] = float(np.mean(share["proxy"]))
    # 실제로 산 KoAct 전용·대리 종목(진입일, 보유 주, 기여)
    bought, prev = {}, set()
    for e in log:
        held = {x for x, w in e["tgt"].items() if w > 1e-12}
        for x in held:
            s = e["src"].get(x)
            if s in ("koact", "proxy"):
                tk = E.ticker[x]
                # entries = 새로 산 재조정일, first = 이 출처로 처음 들고 있던 주(이미 TIME 몫으로 들고 있다가
                # TIME이 판 뒤 KoAct·대리 몫으로 계속 든 경우 entries가 비어 있을 수 있다)
                d = bought.setdefault(tk, {"source": s, "first": e["session"], "entries": [], "weeks": 0,
                                           "from": set(), "subsector": E.group(x), "sector": E.sector.get(x, UNKNOWN)})
                d["weeks"] += 1
                d["from"].update(e["from"].get(x, []))
                if x not in prev:
                    d["entries"].append(e["session"])
        prev = held
    for tk, d in bought.items():
        d["from"] = sorted(d["from"])
        d["contribution"] = round(float(contrib.get(tk, 0.0)), 4)
    res["산종목"] = dict(sorted(bought.items(), key=lambda x: x[1]["first"]))
    # 대리 비중이 더해진 TIME/KoAct 보유 종목(대리 가산으로 비중이 바뀐 주 수)
    boost = Counter(E.ticker[x] for e in log for x in e["keep"] if e["src"].get(x) in ("time", "koact")
                    and e["from"].get(x))
    res["대리가산종목"] = dict(boost.most_common())
    res["재조정수"] = len(log)
    res["_log"], res["_v"] = log, v
    return res


def preview(E, p, log):
    current = set(log[-1]["keep"])
    k = len(E.days)
    keep, tgt, ranked = E.target(p, k, NEXT_SESSION, current)
    src, frm = dict(E._src), dict(E._from)
    rank = {r["key"]: (i + 1, r["mom"], r["w"]) for i, r in enumerate(ranked)}
    hold = [{"name": E.ticker[x], "weight": round(tgt.get(x, 0.0), 4), "source": src.get(x),
             "proxy_for": frm.get(x, []), "subsector": E.group(x), "sector": E.sector.get(x, UNKNOWN),
             "rank": rank[x][0], "mom120": round(rank[x][1], 4), "base_w": round(rank[x][2], 4)} for x in keep]
    hold.sort(key=lambda h: -h["weight"])
    semis = sum(tgt.get(x, 0.0) for x in keep if in_semis_budget(E.sector.get(x, UNKNOWN)))
    return {"session": NEXT_SESSION, "data_through": E.days[-1],
            "pdf_time": latest_before(E.book.pdfs, NEXT_SESSION), "pdf_koact": latest_before(E.kpdfs, NEXT_SESSION),
            "holdings": hold, "held_last_week": sorted(E.ticker[x] for x in current),
            "cash": round(1 - sum(tgt.values()), 4), "semis_budget_weight": round(semis, 4),
            "top12": [f"{E.ticker[r['key']]}({src.get(r['key'])},{E.group(r['key'])})" for r in ranked[:12]]}


def main():
    E = ExpandEngine()
    # 1) 재현 확인: T/off/없음 = 데스크 run_model(config.RULE 반도체 50%)
    ref_live = np.asarray(run_model(E.book, start=START).value)
    res = {name: one(E, name, p) for name, p in GRID}
    live = res[LIVE_NAME]
    d_live = float(np.max(np.abs(live["_v"].to_numpy() - ref_live)))
    plain = Engine()          # 고치지 않은 deskx 엔진: TK/off/없음 = universe 'union', weight 'time'(KoAct 전용은 KoAct 비중)
    d_tk = float(np.max(np.abs(plain.run(Params(universe="union", semis_cap=0.50))[0].to_numpy()
                               - res["TK/off/m=none"]["_v"].to_numpy())))
    checks = {"T_off_none_vs_run_model_live50": d_live, "TK_off_none_vs_deskx_union_time_weight": d_tk}
    print("재현 확인(최대 절대차):", checks)
    assert max(checks.values()) < 1e-12, checks
    post = one(E, *POSTHOC)

    out = [res[name] for name, _ in GRID]
    cand = [r for r in out if r["name"] != LIVE_NAME]
    best = max(out, key=lambda r: r[FRONT]["수익"])
    both = [r["name"] for r in cand if r[FRONT]["수익"] > live[FRONT]["수익"] and r[BACK]["수익"] > live[BACK]["수익"]]

    print("\n비미국 보유 → 대리:")
    for key, d in sorted(E.proxy_info.items(), key=lambda x: x[1]["via"]):
        print(f"  {key:<20} {d['symbol']:<11} {str(d['subsector']):<14} {d['via']:<8} {','.join(d['proxies'])}")
    print("가격을 새로 받은 대리 종목:", E.fetched)

    head = f"{'변형':<16}{'앞':>8}{'뒤':>8}{'9/1~':>8}{'전체':>8}{'MDD':>8}" \
           + "".join(f"{m[-2:] + '월':>7}" for m in MONTHS) \
           + f"{'반도체':>7}{'세부수':>6}{'상위3':>6}{'KoAct':>7}{'대리':>6}  상위3종목"
    print("\n" + head)
    for r in out + [post]:
        print(f"{r['name']:<16}{r[FRONT]['수익']:>8.3f}{r[BACK]['수익']:>8.3f}{r[SEP]['수익']:>8.3f}"
              f"{r[FULL]['수익']:>8.3f}{r[FULL]['낙폭']:>8.3f}"
              + "".join(f"{r['월별'][m]:>7.3f}" for m in MONTHS)
              + f"{r['반도체평균']:>7.3f}{r['세부섹터수평균']:>6.2f}{r['상위3비중']:>6.2f}"
              + f"{r['KoAct전용비중']:>7.3f}{r['대리비중']:>6.3f}  {','.join(r['상위3'])}")
    print(f"\n앞 기준 최선: {best['name']} (앞 {best[FRONT]['수익']:.3f}, 뒤 {best[BACK]['수익']:.3f})"
          f" | 앞·뒤 모두 live 초과: {both or '없음'}")
    for r in out + [post]:
        if r["산종목"]:
            print(f"{r['name']} 산 KoAct전용·대리 종목:",
                  {t: (d['source'][0], d['entries'] or ['보유중:' + d['first']], d['weeks'], d['contribution'])
                   for t, d in r["산종목"].items()})

    best_p = dict(GRID)[best["name"]]
    nxt = preview(E, best_p, best["_log"])
    nxt_live = preview(E, dict(GRID)[LIVE_NAME], live["_log"])
    print(f"\n{NEXT_SESSION} 재조정 미리보기({best['name']}, 자료 {nxt['data_through']}까지, PDF {nxt['pdf_time']}):")
    for h in nxt["holdings"]:
        print(f"  {h['name']:<6}{h['weight']:>7.3f}  {h['source']:<6}{h['subsector']}  (순위 {h['rank']})"
              + (f"  대리: {','.join(h['proxy_for'])}" if h["proxy_for"] else ""))
    print(f"  현금 {nxt['cash']:.3f}, 반도체 예산 비중 {nxt['semis_budget_weight']:.3f}")
    print("  live 규칙이면:", [(h["name"], h["weight"]) for h in nxt_live["holdings"]], "현금", nxt_live["cash"])
    best_on = max([r for r in out if dict(GRID)[r["name"]].proxies], key=lambda r: r[FRONT]["수익"])
    nxt_on = preview(E, dict(GRID)[best_on["name"]], best_on["_log"])
    print(f"  대리 on 중 앞 최선({best_on['name']})이면:",
          [(h["name"], h["weight"], h["source"]) for h in nxt_on["holdings"]], "현금", nxt_on["cash"])

    def clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"family": "expand", "grid": len(GRID),
                   "grid_spec": "univ {T, TK} x proxies {off, on} x m {1, 2, none(7)}; semis budget 0.50, sector 0.60, stock 0.25",
                   "group_rule": "subsector_of(key,ticker); '미분류' = 종목마다 별도 그룹",
                   "proxy_rule": "non-US holding -> US AI_UNIVERSE names of its sub-sector; else US names of its top group; else dropped; NAV weight split equally",
                   "hindsight": "AI_UNIVERSE (proxy list) assembled Oct 2026; watchlist sector labels are current values",
                   "checks": checks,
                   "proxy_map": E.proxy_info, "fetched_proxy_prices": E.fetched,
                   "variants": [clean(r) for r in out], "best_by_front": best["name"], "beats_live_both_halves": both,
                   "posthoc": clean(post), "posthoc_note": "added after seeing the grid; not a selection candidate",
                   "next_rebalance_preview": nxt, "next_rebalance_live": nxt_live,
                   "next_rebalance_best_proxy_on": {"variant": best_on["name"], **nxt_on},
                   "weekly_log": {r["name"]: [{"session": e["session"],
                                               "tgt": {E.ticker[x]: round(w, 4) for x, w in e["tgt"].items()},
                                               "src": {E.ticker[x]: s for x, s in e["src"].items()}}
                                              for e in r["_log"]] for r in out + [post]}},
                  f, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
