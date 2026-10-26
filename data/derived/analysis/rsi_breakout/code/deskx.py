"""데스크 7종목 규칙(#774) 위에 오늘 나온 아이디어를 얹어 보는 연구용 엔진(2026-10-10). desk/ 코드는 고치지 않는다.

기본값으로 돌리면 desk.strategies.run_model과 같은 시가 경로가 나와야 한다(check()로 확인).
얹을 수 있는 것(Params):
  universe     'time'(기본: TIME 미국 보유) | 'union'(TIME ∪ KoAct 미국 보유)
  weight       'time'(기본: TIME NAV 비중) | 'time_else_koact'(TIME에 없으면 KoAct 비중) | 'avg'(두 펀드 비중 평균, 없는 쪽 0)
               | 'max'(두 비중 중 큰 값) | 'equal'(같은 비중)
  lookback n buffer stock_cap sector_cap semis_cap   (desk/config.py RULE와 같은 뜻)
  rebalance    'weekly'(기본: ISO 주 첫 거래일) | 'daily'
  trail        None | x: 들고 있는 종목의 종가가 산 뒤 최고 종가의 (1-x) 아래로 내려가면 다음 시가에 판다(손절 주문 대용).
  trail_refill False(판 자리는 다음 재조정까지 현금) | True(그날 순위에서 다음 종목으로 바로 채움)
               알려진 버그(2026-10-10 검증): target()이 keep을 n개로 잘라 방금 판 종목이 1순위면 아무것도 못 채우고,
               같은 주에 앞서 판 종목을 다시 산다. 고친 계산은 verify_deskx_stops.py에 있다(고친 값도 기본보다 나빴다).
               재조정일에는 손절 판정을 하지 않는다(문턱을 이미 깬 종목을 다시 담고 다음 날 판다).
  min_near_high None | y: 재조정 때 종가/252일 최고 '종가'가 y 미만인 종목은 후보에서 뺀다(꺾인 주도주 제외)
시점: k일 시가 결정은 k일보다 앞선 날짜의 PDF와 k-1일 종가까지만 쓴다(데스크 §3과 같음).
"""
import os, sys
from dataclasses import dataclass, replace
from datetime import date
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))
sys.path.insert(0, ROOT)
from desk import config
from desk.strategies import (load_book, run_model, project_caps, sector_of, us_weights, research_key, latest_before,
                             iso_week, UNKNOWN)
from gbrc.etf.trades import load_fund_snapshots
from gbrc.etf.market import yahoo_ticker

START, END = "2026-01-02", "2026-10-09"
PERIODS = (("앞 1/2~5/29", "2026-01-02", "2026-05-29"), ("뒤 6/1~10/9", "2026-06-01", "2026-10-09"),
           ("9/1~10/9", "2026-09-01", "2026-10-09"), ("전체", START, END))
MONTHS = [f"2026-{m:02d}" for m in range(1, 11)]


@dataclass(frozen=True)
class Params:
    universe: str = "time"
    weight: str = "time"
    lookback: int = 120
    n: int = 7
    buffer: int = 5
    stock_cap: float = 0.25
    sector_cap: float = 0.60
    semis_cap: float = 0.60
    rebalance: str = "weekly"
    trail: float | None = None
    trail_refill: bool = False
    min_near_high: float | None = None


FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cache/deskx_prices")


def _use_frozen_prices():
    """데스크 가격 캐시(prices_*.csv)를 연구 폴더에 한 번 복사해 두고 그 복사본만 읽는다.
    여러 실험이 동시에 돌 때 데스크 캐시를 다시 받거나 덮어쓰지 않게 한다."""
    import shutil
    from desk import marketdata
    files = {f: f"prices_{f.replace(' ', '_').lower()}.csv" for f in marketdata.FIELDS}
    if not all(os.path.exists(os.path.join(FROZEN, n)) for n in files.values()):
        return False
    marketdata.history = lambda tickers, force=False: {
        f: pd.read_csv(os.path.join(FROZEN, n), index_col=0, parse_dates=True) for f, n in files.items()}
    return True


def _freeze_prices():
    import shutil
    from desk import marketdata
    os.makedirs(FROZEN, exist_ok=True)
    for f in marketdata.FIELDS:
        n = f"prices_{f.replace(' ', '_').lower()}.csv"
        shutil.copy(config.CACHE / n, os.path.join(FROZEN, n))


class Engine:
    def __init__(self):
        kact = load_fund_snapshots("2ETFQ1", "9999-12-31")
        self.kpdfs = sorted(kact)
        self.kw = {d: us_weights(s) for d, s in kact.items()}
        kkeys = {k for w in self.kw.values() for k in w}
        frozen = _use_frozen_prices()
        self.book = load_book(extra=sorted({yahoo_ticker(k) for k in kkeys}))
        if not frozen:
            _freeze_prices()
        b = self.book
        self.ticker = dict(b.ticker)
        for k in kkeys:
            self.ticker.setdefault(k, yahoo_ticker(k))
        self.sector = dict(b.sector)
        for k in kkeys:
            if k not in self.sector:
                self.sector[k] = sector_of(k, self.ticker[k])
        self.tw = {d: b.weights(d) for d in b.pdfs}
        self.days = b.days
        self._col = {}
        self.high = None

    def col(self, frame, key):
        tk = self.ticker[key]
        df = getattr(self.book, frame)
        return df[tk].to_numpy(float) if tk in df.columns else None

    def hi252(self, key, k):
        """k-1일까지 252거래일 최고 '종가'(고가 자료가 책에 없어 종가로 잰다)."""
        c = self.col("cls", key)
        if c is None or k - 1 < 0:
            return np.nan
        w = c[max(0, k - 252):k]
        return np.nanmax(w) if np.isfinite(w).any() else np.nan

    # ------------------------------------------------------------ 순위·비중
    def candidates(self, p: Params, session: str) -> dict[str, float]:
        """후보 종목 → 기본 비중 재료(펀드 NAV 비중)."""
        tpdf = latest_before(self.book.pdfs, session)
        tw = self.tw.get(tpdf, {}) if tpdf else {}
        if p.universe == "time":
            return {k: v for k, v in tw.items() if v > 0}
        kpdf = latest_before(self.kpdfs, session)
        kw = self.kw.get(kpdf, {}) if kpdf else {}
        out = {}
        for k in set(tw) | set(kw):
            a, c = tw.get(k, 0.0), kw.get(k, 0.0)
            v = {"time": a, "time_else_koact": a if a > 0 else c, "avg": (a + c) / 2, "max": max(a, c),
                 "equal": 1.0}[p.weight]
            if p.weight == "time" and a <= 0:
                v = c                      # union인데 weight='time'이면 KoAct 전용 종목은 KoAct 비중
            if v > 0:
                out[k] = v
        return out

    def ranking(self, p: Params, k: int, session: str) -> list[dict]:
        cand = self.candidates(p, session)
        rows = []
        for key, w in sorted(cand.items()):
            px = self.col("adj", key)
            if px is None or k - 1 - p.lookback < 0:
                continue
            p1, p0 = px[k - 1], px[k - 1 - p.lookback]
            if not (np.isfinite(p1) and np.isfinite(p0) and p0 > 0):
                continue
            if p.min_near_high is not None:
                c = self.col("cls", key)
                h = self.hi252(key, k)
                if not (np.isfinite(h) and h > 0 and c[k - 1] / h >= p.min_near_high):
                    continue
            rows.append({"key": key, "mom": p1 / p0 - 1, "w": w if p.weight != "equal" else 1.0})
        rows.sort(key=lambda r: -r["mom"])
        return rows

    def target(self, p: Params, k: int, session: str, current: set[str]):
        ranked = self.ranking(p, k, session)
        if not ranked:
            return [], {}, ranked
        order = [r["key"] for r in ranked]
        keep = [x for x in order[:p.n + p.buffer] if x in current][:p.n]
        for x in order:
            if len(keep) >= p.n:
                break
            if x not in keep:
                keep.append(x)
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
        return keep, tgt, ranked

    # ------------------------------------------------------------ 경로
    def run(self, p: Params = Params()):
        days = [d for d in self.days if d >= START]
        k0 = self.days.index(days[0])
        value, rows = [1.0], []
        w: dict[str, float] = {}
        selected: set[str] = set()
        peak: dict[str, float] = {}
        started = False
        for k in range(k0, len(self.days)):
            day = self.days[k]
            if k > k0:
                rets = {}
                for key in w:
                    o = self.col("opn", key)
                    rets[key] = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                        and o[k - 1] > 0 else 0.0
                gross = sum(wt * rets[key] for key, wt in w.items())
                total = sum(wt * (1 + rets[key]) for key, wt in w.items()) + (1 - sum(w.values()))
                w = {key: wt * (1 + rets[key]) / total for key, wt in w.items()}
                value.append(value[-1] * (1 + gross))
            # 산 뒤 최고 종가 갱신(k-1일 종가까지)
            for key in list(peak):
                c = self.col("cls", key)
                if c is not None and np.isfinite(c[k - 1]):
                    peak[key] = max(peak[key], c[k - 1])
            reb = (not started) or (p.rebalance == "daily") or iso_week(day) != iso_week(self.days[k - 1])
            sold_today = []
            if p.trail is not None and not reb:
                for key in list(w):
                    c = self.col("cls", key)
                    if key in peak and c is not None and np.isfinite(c[k - 1]) and c[k - 1] <= peak[key] * (1 - p.trail):
                        sold_today.append(key)
                for key in sold_today:
                    w.pop(key)
                    peak.pop(key, None)
                    selected.discard(key)
                if sold_today and p.trail_refill:
                    keep, tgt, _ = self.target(p, k, day, selected)
                    for key in keep:
                        if key not in w and key not in sold_today and len(w) < p.n:
                            room = 1 - sum(w.values())
                            add = min(tgt.get(key, 0.0), room)
                            if add > 1e-9:
                                w[key] = add
                                selected.add(key)
                                o = self.col("opn", key)
                                peak[key] = o[k] if o is not None and np.isfinite(o[k]) else np.nan
            if reb:
                keep, tgt, ranked = self.target(p, k, day, selected)
                if ranked:
                    started = True
                    for key in set(tgt) - set(w):
                        o = self.col("opn", key)
                        peak[key] = o[k] if o is not None and np.isfinite(o[k]) else np.nan
                    for key in set(w) - set(tgt):
                        peak.pop(key, None)
                    w, selected = dict(tgt), set(keep)
            rows.append(dict(day=day, w=dict(w)))
        v = pd.Series(value, index=days)
        return v, rows

    def evaluate(self, name, v, rows=None):
        r = (v.shift(-1) / v - 1).dropna()
        out = {"name": name}
        for lab, a, b in PERIODS:
            x = r[(r.index >= a) & (r.index <= b)]
            curve = (1 + x).cumprod()
            out[lab] = {"수익": float(curve.iloc[-1] - 1) if len(x) else float("nan"),
                        "낙폭": float((curve / np.maximum.accumulate(curve) - 1).min()) if len(x) else float("nan")}
        out["월별"] = {m: float((1 + r[r.index.str.startswith(m)]).prod() - 1) for m in MONTHS}
        if rows:
            semis = []
            for row in rows:
                semis.append(sum(wt for key, wt in row["w"].items()
                                 if self.sector.get(key, UNKNOWN) in config.SEMIS_SECTORS or self.sector.get(key) == UNKNOWN))
            out["반도체평균"] = float(np.mean(semis))
            out["마지막보유"] = {self.ticker[k]: round(wt, 3) for k, wt in rows[-1]["w"].items()}
            names = set()
            for row in rows:
                names |= {self.ticker[k] for k in row["w"]}
            out["거친종목수"] = len(names)
        return out


def check():
    from desk.validate import codex_caps      # 2026-10-10부터 데스크는 반도체 50%, 이 엔진 기본값은 #774(60%)
    E = Engine()
    v, rows = E.run()
    with codex_caps():
        ref = run_model(E.book, start=START)
    diff = np.max(np.abs(v.to_numpy() - np.asarray(ref.value)))
    res = E.evaluate("기본(#774 재현)", v, rows)
    print("데스크 run_model과의 최대 차", diff)
    print({k: res[k] for k in ("앞 1/2~5/29", "뒤 6/1~10/9", "9/1~10/9", "전체")}, res["반도체평균"], res["마지막보유"])
    return diff


if __name__ == "__main__" and sys.argv[1:] == ["check"]:
    check()
