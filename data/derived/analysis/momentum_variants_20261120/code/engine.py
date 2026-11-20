"""모멘텀 변형 공통 엔진 (2026-11-20). 모든 갈래가 이 파일을 그대로 쓴다(수정 금지, 버그는 보고).

시점 규칙
- 세션 k의 결정은 k-1 종가까지의 값만 쓴다. 체결은 k(+delay) 개장가(수정 시가 = 시가 × 수정종가/종가).
- 펀드 보유 종목군은 k 날짜보다 앞선 최신 스냅숏(`held_days.csv`, 2026-01-02~)만 쓴다. 2025에는 펀드 종목군이 없다.
- 거래대금 문턱은 k-20 ~ k-1 평균 종가×거래량.

종목군(universe)
  TIME, KOACT, FUNDS(=둘 합)   : 그때 보유 미국 종목(2026만)
  AI                            : watchlist.AI_UNIVERSE 미국 종목(지금 목록 → 사후 선택이 섞임)
  WIDE50 / WIDE200              : 넓은 표본(앞선 넓은 연구 2025-06 이후 실적 종목, 지금 시총 기준 목록 → 생존 편향) 중
                                  20일 평균 거래대금 5천만 / 2억 달러 이상
  WIDETECH50                    : WIDE50 중 기술 업종(넓은 표 sector Technology) 또는 SOXX 계열
  ALL50                         : WIDE ∪ 펀드 2026 보유 ∪ AI, 거래대금 5천만 달러 이상
상한(caps=True): 한 종목 25%, 같은 섹터 60%, 반도체 50%(desk.strategies.project_caps). 섹터 이름은 watchlist가 아는 종목은
그 이름, 모르는 종목은 넓은 표 sector(반도체 ETF가 SOXX면 'AI 반도체'로 묶음). 데스크와 달리 미분류를 반도체로 세지 않는다.

평가: run() → Result(세션별 개장·종가 가치, 노출, 회전율). metrics()가 2026 전체·상반기(H1, 1/2 개장→6/30 종가)·
하반기(H2, 7/1 개장→마지막 종가)·최근 분기(9/1 개장→마지막 종가)·최대 낙폭·2025 하반기(보조, 종목군이 있을 때)를 낸다.
마지막 종가는 LAST(2026-11-18, yfinance 11/19 행 일부 결측).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
for p in (str(ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)
import watchlist as WL  # noqa: E402
from desk import config as DC  # noqa: E402
from desk import strategies as DS  # noqa: E402

LAST = "2026-11-18"
Y26 = "2026-01-02"
PERIODS = {"Y2026": ("2026-01-02", LAST), "H1": ("2026-01-02", "2026-06-30"), "H2": ("2026-07-01", LAST),
           "RECENT": ("2026-09-01", LAST), "Y2025H2": ("2025-07-01", "2025-12-31")}
BENCH = ["QQQ", "SPY", "SOXX", "IWM", "XLK", "^VIX"]


@dataclass
class Result:
    days: list
    v_open: np.ndarray            # value at each session open (before that open's trade cost), start 1.0
    v_close: np.ndarray           # value at each close
    expo: np.ndarray              # invested fraction held from each open
    r_oo: np.ndarray              # strategy open→next open return per session (fully invested basket)
    r_oc: np.ndarray              # strategy open→close return per session
    turnover: float               # mean one-way traded weight per rebalance
    holdings: dict = field(default_factory=dict)   # rebalance day -> {ticker: weight}


class Engine:
    def __init__(self):
        c = BASE / "cache"
        ac = pd.read_parquet(c / "px_adjclose.parquet")
        cl = pd.read_parquet(c / "px_close.parquet")
        op = pd.read_parquet(c / "px_open.parquet")
        vo = pd.read_parquet(c / "px_volume.parquet")
        days = ac["QQQ"].dropna().index
        days = days[days <= LAST]
        ac, cl, op, vo = (x.reindex(days) for x in (ac, cl, op, vo))
        self.days = [d.strftime("%Y-%m-%d") for d in days]
        self.pos = {d: i for i, d in enumerate(self.days)}
        self.tickers = [t for t in ac.columns if t not in BENCH]
        self.bench_ac = ac[[b for b in BENCH if b in ac.columns]]
        self.bench_ao = (op * ac / cl)[[b for b in BENCH if b in ac.columns]]
        self.AC = ac[self.tickers].to_numpy(float)
        self.AO = (op * ac / cl)[self.tickers].to_numpy(float)
        self.CL = cl[self.tickers].to_numpy(float)
        self.DV = (cl * vo)[self.tickers].to_numpy(float)
        self.col = {t: i for i, t in enumerate(self.tickers)}
        meta = pd.read_csv(c / "meta.csv", index_col=0)
        self.meta = meta.reindex(self.tickers)
        self.sector = self._sectors()
        self.adv20 = pd.DataFrame(self.DV).rolling(20, min_periods=15).mean().shift(1).to_numpy()  # through k-1
        held = pd.read_csv(ROOT / "data/derived/analysis/post_earnings_risers_20261120/results/held_days.csv")
        held = held[held.ticker.isin(self.col)]
        self.snap_dates = sorted(held.date.unique())
        self.held = {(d, f): g.set_index("ticker").weight.to_dict() for (d, f), g in held.groupby(["date", "fund"])}
        ai = self.meta.index[self.meta.in_ai.fillna(False).astype(bool)]
        wide = self.meta.index[self.meta.in_wide.fillna(False).astype(bool)]
        tech = self.meta.index[(self.meta.sector == "Technology") | (self.meta.etf == "SOXX")]
        self.static = {"AI": self._mask(ai), "WIDE": self._mask(wide), "TECH": self._mask(tech),
                       "ALLBASE": self._mask(set(ai) | set(wide) | set(self.meta.index[self.meta.in_fund.fillna(False).astype(bool)]))}
        self._u_cache = {}

    # ------------------------------------------------------------------ setup
    def _mask(self, names) -> np.ndarray:
        m = np.zeros(len(self.tickers), bool)
        for t in names:
            if t in self.col:
                m[self.col[t]] = True
        return m

    def _sectors(self) -> dict:
        out = {}
        for t in self.tickers:
            s = WL.resolve_sector(t, default=DS.UNKNOWN)
            if s == DS.UNKNOWN:
                etf = self.meta.etf.get(t)
                sec = self.meta.sector.get(t)
                s = "AI 반도체" if etf == "SOXX" else (sec if isinstance(sec, str) else "기타")
            out[t] = s
        return out

    def snapshot_before(self, day: str):
        i = int(np.searchsorted(self.snap_dates, day)) - 1
        return self.snap_dates[i] if i >= 0 else None

    def universe(self, name: str, k: int) -> np.ndarray:
        """Bool mask over tickers eligible for the decision at session k."""
        key = (name, k)
        if key in self._u_cache:
            return self._u_cache[key]
        day = self.days[k]
        has_px = np.isfinite(self.AC[k - 1]) & np.isfinite(self.AO[min(k, len(self.days) - 1)])
        if name in ("TIME", "KOACT", "FUNDS"):
            snap = self.snapshot_before(day)
            m = np.zeros(len(self.tickers), bool)
            if snap is not None:
                funds = {"TIME": ["TIME"], "KOACT": ["KoAct"], "FUNDS": ["TIME", "KoAct"]}[name]
                names = set().union(*(self.held.get((snap, f), {}).keys() for f in funds))
                m = self._mask(names)
        elif name == "AI":
            m = self.static["AI"].copy()
        elif name.startswith("WIDETECH"):
            m = self.static["WIDE"] & self.static["TECH"] & (self.adv20[k] >= float(name[8:]) * 1e6)
        elif name.startswith("WIDE"):
            m = self.static["WIDE"] & (self.adv20[k] >= float(name[4:]) * 1e6)
        elif name.startswith("ALL"):
            m = self.static["ALLBASE"] & (self.adv20[k] >= float(name[3:]) * 1e6)
        else:
            raise ValueError(name)
        m = m & has_px
        self._u_cache[key] = m
        return m

    def time_weights(self, k: int) -> np.ndarray:
        snap = self.snapshot_before(self.days[k])
        w = np.zeros(len(self.tickers))
        for t, v in (self.held.get((snap, "TIME"), {}) if snap else {}).items():
            w[self.col[t]] = v
        return w

    # ------------------------------------------------------------------ scores (rows = decision session k)
    def ret(self, lb: int, skip: int = 0) -> np.ndarray:
        """S[k] = AC[k-1-skip] / AC[k-1-lb] - 1 (closes through k-1)."""
        A = self.AC
        S = np.full_like(A, np.nan)
        S[lb + 1:] = A[lb - skip: len(A) - 1 - skip] / A[: len(A) - lb - 1] - 1 if skip < lb else np.nan
        return S

    def vol(self, lb: int) -> np.ndarray:
        """Daily log-return std over the lb sessions through k-1."""
        lr = np.log(self.AC[1:] / self.AC[:-1])
        s = pd.DataFrame(lr).rolling(lb, min_periods=max(5, lb // 2)).std().to_numpy()
        out = np.full_like(self.AC, np.nan)
        out[2:] = s[:-1]
        return out

    def bench_ret(self, sym: str, lb: int) -> np.ndarray:
        a = self.bench_ac[sym].to_numpy(float)
        out = np.full(len(a), np.nan)
        out[lb + 1:] = a[lb:-1] / a[:-lb - 1] - 1
        return out

    # ------------------------------------------------------------------ backtest
    def rebalance_days(self, start_k: int, end_k: int, freq: str) -> list[int]:
        ks = []
        for k in range(start_k, end_k + 1):
            d0 = pd.Timestamp(self.days[k])
            if k == start_k:
                ks.append(k)
                continue
            dp = pd.Timestamp(self.days[k - 1])
            if freq == "D":
                ks.append(k)
            elif freq == "W" and d0.isocalendar()[:2] != dp.isocalendar()[:2]:
                ks.append(k)
            elif freq == "2W" and d0.isocalendar()[:2] != dp.isocalendar()[:2] and d0.isocalendar()[1] % 2 == 0:
                ks.append(k)
            elif freq == "M" and d0.month != dp.month:
                ks.append(k)
        return ks

    def run(self, score: np.ndarray, universe: str = "TIME", n: int = 7, buffer: int = 5, weight: str = "equal",
            caps: bool = True, freq: str = "W", delay: int = 0, start: str = Y26, end: str = LAST,
            cost_bps: float = 0.0, exposure=None, cap_rule: dict | None = None, ivol_lb: int = 60,
            pick=None) -> Result:
        """Weekly top-n momentum. score[k] ranks names for the decision at k (higher = better).

        pick(k, cand_idx, score_k, prev_set) -> list of chosen column indices may replace the default
        buffer selection. exposure: array over [start..end] sessions (fraction held from that open,
        decided with info through the previous close) or None for 1.0.
        """
        k0, k1 = self.pos[start], self.pos[end]
        rebs = set(self.rebalance_days(k0, k1, freq))
        vol = self.vol(ivol_lb) if weight == "ivol" else None
        rule = cap_rule or DC.RULE
        T = k1 - k0 + 1
        w = np.zeros(len(self.tickers))
        prev: list[int] = []
        pending = {}  # k_exec -> target
        r_oo, r_oc, v_open, v_close, holds, turns = np.zeros(T), np.zeros(T), np.ones(T), np.ones(T), {}, []
        cf = np.ones(T)  # trade-cost factor applied at that open
        for j in range(T):
            k = k0 + j
            if k in rebs:
                cand = np.where(self.universe(universe, k) & np.isfinite(score[k]))[0]
                if len(cand):
                    order = cand[np.argsort(-score[k][cand], kind="stable")]
                    if pick is not None:
                        chosen = list(pick(k, order, score[k], set(prev)))
                    else:
                        top = list(order[: n + buffer])
                        chosen = [c for c in top if c in prev][:n]
                        for c in order:
                            if len(chosen) >= n:
                                break
                            if c not in chosen:
                                chosen.append(c)
                    if weight == "time":
                        tw = self.time_weights(k)
                        raw = {self.tickers[c]: tw[c] for c in chosen if tw[c] > 0}
                    elif weight == "ivol":
                        iv = {self.tickers[c]: 1 / vol[k][c] for c in chosen if np.isfinite(vol[k][c]) and vol[k][c] > 0}
                        raw = iv
                    else:
                        raw = {self.tickers[c]: 1.0 for c in chosen}
                    tot = sum(raw.values())
                    raw = {t: v / tot for t, v in raw.items()} if tot > 0 else {}
                    if caps and raw:
                        saved = dict(DC.RULE)
                        DC.RULE.update(rule)
                        try:
                            tgt = DS.project_caps(raw, self.sector)
                        finally:
                            DC.RULE.clear()
                            DC.RULE.update(saved)
                    else:
                        tgt = raw
                    vec = np.zeros(len(self.tickers))
                    for t, v in tgt.items():
                        vec[self.col[t]] = v
                    pending[min(k + delay, k1)] = (vec, k)
                    prev = chosen
            if k in pending:
                vec, kd = pending.pop(k)
                turn = np.abs(vec - w).sum() / 2
                turns.append(turn)
                if j > 0:
                    cf[j] = 1 - cost_bps / 1e4 * 2 * turn
                    v_open[j] *= cf[j]
                w = vec
                holds[self.days[kd]] = {self.tickers[i]: float(w[i]) for i in np.where(w > 0)[0]}
            # intraday
            oc = self.AC[k] / self.AO[k] - 1
            oc = np.where(np.isfinite(oc), oc, 0.0)
            r_oc[j] = float((w * oc).sum())
            v_close[j] = v_open[j] * (1 + r_oc[j])
            if j + 1 < T:
                oo = self.AO[k + 1] / self.AO[k] - 1
                held_bad = (w > 0) & ~np.isfinite(oo)
                oo = np.where(np.isfinite(oo), oo, np.where(held_bad, oc, 0.0))  # 다음 시가가 없으면 그날 종가로
                r_oo[j] = float((w * oo).sum())
                v_open[j + 1] = v_open[j] * (1 + r_oo[j])
                grown = w * (1 + oo)
                cash = 1 - w.sum()
                w = grown / (grown.sum() + cash) if grown.sum() + cash > 0 else w
        expo = np.ones(T) if exposure is None else np.asarray(exposure, float)
        if exposure is not None:
            vo, vc = np.ones(T), np.ones(T)
            for j in range(T):
                vo[j] *= 1 - expo[j] * (1 - cf[j])  # 비중만큼만 거래 비용
                vc[j] = vo[j] * (1 + expo[j] * r_oc[j])
                if j + 1 < T:
                    vo[j + 1] = vo[j] * (1 + expo[j] * r_oo[j])
            v_open, v_close = vo, vc
        return Result(self.days[k0:k1 + 1], v_open, v_close, expo, r_oo, r_oc,
                      float(np.mean(turns)) if turns else 0.0, holds)

    # ------------------------------------------------------------------ evaluation
    @staticmethod
    def span(res: Result, a: str, b: str) -> float:
        d = res.days
        if a < d[0] or b > d[-1]:
            return float("nan")
        i = next(j for j, x in enumerate(d) if x >= a)
        jb = max(j for j, x in enumerate(d) if x <= b)
        return float(res.v_close[jb] / res.v_open[i] - 1)

    @staticmethod
    def mdd(res: Result, a: str | None = None, b: str | None = None) -> float:
        d = np.array(res.days)
        m = np.ones(len(d), bool)
        if a:
            m &= d >= a
        if b:
            m &= d <= b
        v = np.r_[res.v_open[m][0], res.v_close[m]]
        return float((v / np.maximum.accumulate(v) - 1).min())

    def metrics(self, res: Result) -> dict:
        out = {}
        for nm, (a, b) in PERIODS.items():
            if nm == "Y2025H2":
                continue
            out[nm] = self.span(res, a, b)
        out["MDD2026"] = self.mdd(res, "2026-01-02", LAST)
        out["MDD_H2"] = self.mdd(res, "2026-07-01", LAST)
        out["avg_expo"] = float(np.mean(res.expo))
        out["turnover"] = res.turnover
        return out

    def metrics_2025(self, res: Result) -> dict:
        return {"Y2025H2": self.span(res, *PERIODS["Y2025H2"]), "MDD2025H2": self.mdd(res, *PERIODS["Y2025H2"])}

    def bench(self, sym: str, start: str = Y26, end: str = LAST) -> Result:
        k0, k1 = self.pos[start], self.pos[end]
        a = self.bench_ac[sym].to_numpy(float)[k0:k1 + 1]
        o = self.bench_ao[sym].to_numpy(float)[k0:k1 + 1]
        T = k1 - k0 + 1
        r_oo = np.r_[o[1:] / o[:-1] - 1, 0.0]
        r_oc = a / o - 1
        return Result(self.days[k0:k1 + 1], o / o[0], a / o[0], np.ones(T), r_oo, r_oc, 0.0, {})

    def exposure_control(self, base: Result, expo_mean: float) -> Result:
        """같은 평균 비중 대조: base 바구니를 늘 expo_mean만큼 든 경로."""
        T = len(base.days)
        vo, vc = np.ones(T), np.ones(T)
        for j in range(T):
            vc[j] = vo[j] * (1 + expo_mean * base.r_oc[j])
            if j + 1 < T:
                vo[j + 1] = vo[j] * (1 + expo_mean * base.r_oo[j])
        return Result(base.days, vo, vc, np.full(T, expo_mean), base.r_oo, base.r_oc, base.turnover, {})

    @staticmethod
    def walk_forward(results: dict, lookback: int = 63, start: str = "2026-03-02") -> dict:
        """이동 전방 검증: 매달 첫 세션에 직전 lookback 세션 수익이 가장 좋은 설정을 골라 그달 보유.

        results: {name: Result} (같은 날짜). 결정은 그 달 첫 세션 전날 종가까지의 가치(v_close)로."""
        names = list(results)
        d = results[names[0]].days
        V = np.column_stack([results[n].v_close for n in names])
        OO = np.column_stack([results[n].r_oo for n in names])
        OC = np.column_stack([results[n].r_oc for n in names])
        choice, cur = [], None
        for j, day in enumerate(d):
            if day < start:
                choice.append(None)
                continue
            if cur is None or day[:7] != d[j - 1][:7]:
                lo = max(0, j - 1 - lookback)
                perf = V[j - 1] / V[lo]
                cur = int(np.nanargmax(perf))
            choice.append(cur)
        js = [j for j, c in enumerate(choice) if c is not None]
        vo, vc = [1.0], []
        for i, j in enumerate(js):
            c = choice[j]
            vc.append(vo[-1] * (1 + OC[j, c]))
            if i + 1 < len(js):
                vo.append(vo[-1] * (1 + OO[j, c]))
        picks = [names[choice[j]] for j in js]
        res = Result([d[j] for j in js], np.array(vo), np.array(vc), np.ones(len(js)),
                     np.zeros(len(js)), np.zeros(len(js)), 0.0, {})
        return {"result": res, "picks_by_month": {d[j][:7]: names[choice[j]] for j in js if j == js[0] or d[j][:7] != d[j - 1][:7]},
                "n_switch": int(sum(1 for a, b in zip(picks, picks[1:]) if a != b))}


def baseline_desk() -> dict:
    """데스크 현행 규칙(§3, 반도체 50%)의 USD 경로(desk.strategies.run_model) — 비교 기준."""
    book = DS.load_book(through=LAST, extra=["SOXX"])
    path = DS.run_model(book)
    return {"days": path.days, "v_open": path.value, "v_close": path.close}
