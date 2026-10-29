"""SOXL·SOXS 장초 바꿔 사기 — 공통 자료·체결 엔진(모든 전략이 같은 체결·비용을 쓴다).

자료: cache/alp1m/<SYM>.parquet(Alpaca SIP 1분봉, 분할 반영). 하루 = 04:00~19:59 ET 960분 격자(빈 분은 앞 종가로 채우고 거래량 0).
시각 색인 i = 04:00부터 지난 분(09:30 = 330). 봉 i는 i분에 시작해 i+1분에 끝난다.

체결(BRIEF2.md '체결' 절):
- 전략은 봉 i 마감 뒤 on_bar(ctx, i)에서 Order를 낸다. 시장가는 봉 i+1 종가에 체결(사람 1분 지연, fill='close').
  감도용 fill='open'은 봉 i+1 시가.
- 걸어 둔 손절·목표(체결가 대비 %)는 체결 다음 봉부터 봉 안에서 확인한다. 손절: 저가 ≤ L이면 min(시가, L)에 판다(− 반 호가).
  목표: 고가 ≥ T × 1.0002면 T에 판다(지정가, 호가 비용 없음). 한 봉에 둘 다면 손절.
- 시장가 매수 = 가격 × (1 + 반 호가), 시장가 매도 = 가격 × (1 − 반 호가). 반 호가는 cost_table.json(실측) 시간대별 값.
- 주문 마감: 그날 밤 24:00 KST를 ET로 바꾼 시각(서머타임 11:00, 표준시 10:00). 마지막 결정은 마감 2분 전 봉,
  남은 보유는 마감 1분 전 봉 종가에 판다. scheme='std'는 모든 날을 10:00 마감으로 잰다(11/2~3월 운용판).
- 하루 수익 = 매매 수익의 곱 − 1(한 번에 한 종목, 자금 전부). 매매 기록의 시각은 체결이 일어난 봉의 시작 시각이다.
"""
import json, os
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "cache")
SYMS = ["SOXL", "SOXS", "SOXX", "QQQ", "NVDA"]
ET, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")
NMIN = 960                      # 04:00~19:59
RTH0, RTH1 = 330, 720           # 09:30, 16:00 (색인)
FIELDS = ["Open", "High", "Low", "Close", "Volume", "VWAP"]
PERIODS = {"sel": ("2026-01-02", "2026-06-30"),      # 고르기
           "conf": ("2026-07-01", "2026-10-28"),     # 확인(고르기 뒤에만)
           "aux": ("2025-01-02", "2025-12-31")}      # 보조(고르기 뒤에만)
LOCKED = ("conf", "aux")
TOUCH = 1.0002                  # 목표 지정가 관통 조건


def hm(i):
    m = i + 240
    return f"{m // 60:02d}:{m % 60:02d}"


def idx(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - 240


def cutoff_idx(day, scheme="actual"):
    """그날 밤 24:00 KST의 ET 시각 색인(서머타임 420 = 11:00, 표준시 360 = 10:00)."""
    if scheme == "std":
        return idx("10:00")
    k = datetime(day.year, day.month, day.day, tzinfo=KST) + timedelta(days=1)
    e = k.astimezone(ET)
    return e.hour * 60 + e.minute - 240


class Data:
    """모든 날의 960분 격자. A[sym][field] = (날 수, 960) 배열."""
    def __init__(self):
        frames = {s: pd.read_parquet(os.path.join(CACHE, "alp1m", f"{s}.parquet")) for s in SYMS}
        rth = frames["SOXL"]
        m = rth.index.hour * 60 + rth.index.minute
        days = sorted({d for d, mm in zip(rth.index.date, m) if 570 <= mm < 960})
        self.days = days
        self.pos = {d: k for k, d in enumerate(days)}
        self.A, self.has = {}, {}
        for s, f in frames.items():
            mins = f.index.hour * 60 + f.index.minute - 240
            di = np.array([self.pos.get(d, -1) for d in f.index.date])
            ok = (di >= 0) & (mins >= 0) & (mins < NMIN)
            arrs = {}
            for fld in FIELDS:
                a = np.full((len(days), NMIN), np.nan)
                a[di[ok], mins[ok]] = f[fld].values[ok]
                arrs[fld] = a
            has = ~np.isnan(arrs["Close"])
            c = pd.DataFrame(arrs["Close"]).T.ffill().T.values          # 같은 날 앞 종가로 채움
            for fld in ["Open", "High", "Low", "VWAP"]:
                arrs[fld] = np.where(has, arrs[fld], c)
            arrs["Close"] = c
            arrs["Volume"] = np.where(has, arrs["Volume"], 0.0)
            self.A[s], self.has[s] = arrs, has
        rows = []
        for s in SYMS:
            o, h, l, c = (self.A[s][k][:, RTH0:RTH1] for k in ["Open", "High", "Low", "Close"])
            rows.append(pd.DataFrame({"day": days, "sym": s, "open": o[:, 0], "high": np.nanmax(h, 1),
                                      "low": np.nanmin(l, 1), "close": c[:, -1]}))
        self.daily = pd.concat(rows).set_index(["sym", "day"]).sort_index()

    def days_in(self, period):
        a, b = PERIODS[period]
        a, b = pd.Timestamp(a).date(), pd.Timestamp(b).date()
        return [k for k, d in enumerate(self.days) if a <= d <= b]


class Ctx:
    """전략이 보는 창. 오늘 배열은 지금 봉 i까지만 준다(미래 참조 차단)."""
    def __init__(self, data, d):
        self._D, self.d, self.day = data, d, data.days[d]
        self.i = None
        self.pos, self.entry_px, self.entry_i = 0, None, None
        self.cut = None
        self.state = {}

    def bars(self, sym, field="Close"):
        return self._D.A[sym][field][self.d, : self.i + 1]

    def c(self, sym="SOXL"):
        return self.bars(sym, "Close")

    def prev_day(self, sym="SOXL", k=1):
        """k거래일 전 정규장 시가·고가·저가·종가(dict). 없으면 None."""
        if self.d - k < 0:
            return None
        return self._D.daily.loc[(sym, self._D.days[self.d - k])].to_dict()

    def prev_bars(self, sym="SOXL", field="Close", k=1):
        """k거래일 전 하루 960분 배열 전체(과거라 다 볼 수 있다)."""
        return None if self.d - k < 0 else self._D.A[sym][field][self.d - k]

    @property
    def now(self):
        return hm(self.i + 1)          # 지금 시각 = 봉 i 마감


@dataclass
class Order:
    pos: int                 # +1 SOXL, -1 SOXS, 0 현금
    stop: float = None       # 체결가 대비 손절 비율(0.01 = −1%)
    target: float = None     # 체결가 대비 목표 비율
    keep: bool = False       # True면 같은 보유에 손절·목표만 바꾼다(1분 뒤부터)


INST = {1: "SOXL", -1: "SOXS"}


def load_costs():
    p = os.path.join(ROOT, "cost_table.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {"SOXL": {"pre": 0.0010, "open5": 0.0006, "early": 0.0002, "late": 0.0002},
            "SOXS": {"pre": 0.0015, "open5": 0.0010, "early": 0.0004, "late": 0.0003}}


def bucket(i):
    if i < RTH0:
        return "pre"
    if i < RTH0 + 5:
        return "open5"
    if i < idx("10:00"):
        return "early"
    return "late"


class Engine:
    def __init__(self, data, costs=None, fill="close", scheme="actual", cost_mult=1.0):
        self.D, self.fill, self.scheme = data, fill, scheme
        self.costs, self.cm = costs or load_costs(), cost_mult

    def hs(self, sym, i):
        return self.costs[sym][bucket(i)] * self.cm

    def run_day(self, strat, d):
        D, ctx = self.D, Ctx(self.D, d)
        cut = cutoff_idx(D.days[d], self.scheme)
        ctx.cut = cut
        start = idx(getattr(strat, "start", "09:30"))
        trades, pend = [], None
        stop_px = tgt_px = None
        active = None
        ret = 1.0

        def close_pos(i, px, why, limit=False):
            nonlocal ret, stop_px, tgt_px, active
            sym = INST[ctx.pos]
            out = px if limit else px * (1 - self.hs(sym, i))
            r = out / ctx.entry_px - 1
            trades.append((D.days[d], sym, hm(ctx.entry_i), hm(i), ctx.entry_px, out, r, why))
            ret *= 1 + r
            ctx.pos, ctx.entry_px, ctx.entry_i = 0, None, None
            stop_px = tgt_px = active = None

        if hasattr(strat, "on_day"):
            ctx.i = start - 1
            strat.on_day(ctx)
        for i in range(start, cut):
            ctx.i = i
            # 1) 걸어 둔 손절·목표
            if ctx.pos and active is not None and i >= active:
                A = D.A[INST[ctx.pos]]
                o, h, l = A["Open"][d, i], A["High"][d, i], A["Low"][d, i]
                if stop_px is not None and l <= stop_px:
                    close_pos(i, min(o, stop_px), "stop")
                elif tgt_px is not None and h >= tgt_px * TOUCH:
                    close_pos(i, tgt_px, "target", limit=True)
            # 2) 앞 봉에서 낸 주문 체결
            if pend is not None:
                o_ = pend
                pend = None
                if o_.keep:
                    if ctx.pos and o_.pos == ctx.pos:     # 이미 손절·목표로 나갔으면 버린다
                        stop_px = ctx.entry_px * (1 - o_.stop) if o_.stop else None
                        tgt_px = ctx.entry_px * (1 + o_.target) if o_.target else None
                        active = i
                elif o_.pos != ctx.pos:
                    fld = "Close" if self.fill == "close" else "Open"
                    if ctx.pos and o_.pos != ctx.pos:
                        close_pos(i, D.A[INST[ctx.pos]][fld][d, i], "signal")
                    if o_.pos and ctx.pos == 0:
                        sym = INST[o_.pos]
                        px = D.A[sym][fld][d, i] * (1 + self.hs(sym, i))
                        ctx.pos, ctx.entry_px, ctx.entry_i = o_.pos, px, i
                        stop_px = px * (1 - o_.stop) if o_.stop else None
                        tgt_px = px * (1 + o_.target) if o_.target else None
                        active = i + 1 if self.fill == "close" else i
                        if self.fill == "open":          # 시가 체결이면 같은 봉 안 손절·목표도 본다
                            A = D.A[sym]
                            if stop_px is not None and A["Low"][d, i] <= stop_px:
                                close_pos(i, min(A["Open"][d, i], stop_px), "stop")
                            elif tgt_px is not None and A["High"][d, i] >= tgt_px * TOUCH:
                                close_pos(i, tgt_px, "target", limit=True)
            # 3) 전략 결정(마감 2분 전 봉까지)
            if i <= cut - 2:
                o_ = strat.on_bar(ctx, i)
                if o_ is not None and (o_.pos != ctx.pos or o_.keep):
                    pend = o_
        if ctx.pos:
            close_pos(cut - 1, D.A[INST[ctx.pos]]["Close"][d, cut - 1], "cutoff")
        return ret - 1, trades

    def run(self, make_strat, period=None, days=None):
        """make_strat: 날마다 새 전략 객체를 만드는 함수. 반환: (날별 DataFrame, 매매 DataFrame)."""
        if days is None:
            if period in LOCKED and os.environ.get("SOS_UNLOCK") != "frozen":
                raise PermissionError(f"{period} 구간은 후보를 얼린 뒤 확인 단계에서만 연다")
            days = self.D.days_in(period)
        rows, allt = [], []
        for d in days:
            r, t = self.run_day(make_strat(), d)
            rows.append((self.D.days[d], r, len(t)))
            allt += t
        daily = pd.DataFrame(rows, columns=["day", "ret", "n"])
        tr = pd.DataFrame(allt, columns=["day", "sym", "entry", "exit", "entry_px", "exit_px", "ret", "why"])
        return daily, tr


def summarize(daily, nboot=10000, seed=20261029):
    r = daily["ret"].values * 100
    act = daily["n"].values > 0
    out = {"days": int(len(r)), "active_days": int(act.sum()), "trades": int(daily["n"].sum()),
           "mean_pct": float(r.mean()) if len(r) else None,
           "sd_pct": float(r.std(ddof=1)) if len(r) > 1 else None,
           "hit_days": float((r[act] > 0).mean()) if act.any() else None,
           "sum_pct": float(r.sum()), "compound_pct": float((np.prod(1 + r / 100) - 1) * 100)}
    if len(r) > 1:
        out["t"] = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if r.std() > 0 else 0.0
        rng = np.random.default_rng(seed)
        bs = r[rng.integers(0, len(r), (nboot, len(r)))].mean(1)
        out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["p_one_sided"] = float((bs <= 0).mean())
        eq = np.cumprod(1 + r / 100)
        out["max_dd_pct"] = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return out
