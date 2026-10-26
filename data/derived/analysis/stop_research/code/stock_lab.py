"""2단계 종목 단위 엔진(2026-10-10): 한 종목을 들고 있다가 손절·재진입·사다리 매수 규칙을 따르면 어떻게 되나.

자료: cache/p2_long.parquet(p2_fetch_long.py). 가격은 배당 수정(Adj Close/Close 비율을 시가·고가·저가에도 곱함).

시점(어기면 안 됨)
  - policy.on_open(t, st)   : t일 시가 직전 판단. t-1일 종가까지의 정보만 쓴다(지표 배열은 [t-1]로 읽는다).
                              반환 = 목표 보유 비율(0~1) 또는 None(그대로). t일 시가에 체결.
  - policy.stop(t, st)      : t일 장중 손절 주문 가격(t-1까지 정보). 시가 ≤ 가격이면 시가, 저가 ≤ 가격이면 그 가격에 전량 매도.
  - policy.limit_buys(t, st): t일 장중 지정가 매수 [(가격, 목표 비율)]. 저가 ≤ 가격이면 min(시가, 가격)에 목표 비율까지 산다.
  - 같은 날 순서: 시가 거래 → 손절 → 지정가 매수(같은 날 손절과 매수가 겹치면 손절 먼저, 보수적).
  - policy.after_close(t, st): t일 종가 뒤 상태 갱신(원하면). 엔진이 peak·trough·days_out은 알아서 갱신한다.

상태 st(dict): pos(보유 비율), entry_px, entry_t, peak(산 뒤 최고 종가, t-1까지), high_peak(산 뒤 최고 고가),
  exit_px, exit_t, trough(판 뒤 최저 종가, t-1까지), days_out, n_exits, last_reason.

평가: 창 [a, b] 시작 직전 종가에 그 종목을 다 들고 있다(가치 1). 매수·보유(B&H)는 c[t]/c[a-1].
손절 주기(cycle): 한 번 판 뒤 다시 살 때까지. 주기 손익 = 판 가격/다시 산 가격 − 1(+면 싸게 되산 것 = 손절 성공).
  창 끝까지 못 되샀으면 판 가격/창 끝 종가 − 1. 이것으로 승률·손익비(평균 이익/평균 손실)·기대값을 낸다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ETFS = {"QQQ", "SPY", "SOXX", "SMH", "IWM", "RSP", "^VIX", "^VIX3M"}

# 평가 창(사전 고정). 2026 앞으로 고르고 뒤로 확인, 2025 보조, 나머지 해는 맥락(장세 전환 위험)으로만.
WINDOWS = {
    "2026": ("2026-01-02", "2026-10-09"), "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2024": ("2024-01-02", "2024-12-31"), "2023": ("2023-01-03", "2023-12-29"),
    "2022": ("2022-01-03", "2022-12-30"), "2021": ("2021-01-04", "2021-12-31"), "2020": ("2020-01-02", "2020-12-31"),
    "2019": ("2019-01-02", "2019-12-31"), "2018": ("2018-01-02", "2018-12-31"),
}
# 2026 앞·뒤는 2026 전체를 이어 돌린 곡선을 잘라 잰다(상태 이어짐). 나머지 해는 그해 첫날 들고 시작.
RUNS = {"2026": ("2026-01-02", "2026-10-09"), "2025": WINDOWS["2025"], "2024": WINDOWS["2024"], "2023": WINDOWS["2023"],
        "2022": WINDOWS["2022"], "2021": WINDOWS["2021"], "2020": WINDOWS["2020"], "2019": WINDOWS["2019"],
        "2018": WINDOWS["2018"]}
SLICES = {"2026": ["2026", "2026앞", "2026뒤"]}


# ----------------------------------------------------------------------------- 자료
@dataclass
class Bars:
    sym: str
    days: np.ndarray   # 'YYYY-MM-DD'
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    _cache: dict = field(default_factory=dict)

    def idx(self, day: str, side="left") -> int:
        return int(np.searchsorted(self.days, day, side=side))

    # ---- 지표(모두 t까지의 값이 [t]에 있다. 정책은 [t-1]을 읽어야 한다)
    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def sma(self, n):
        return self._memo(("sma", n), lambda: pd.Series(self.c).rolling(n, min_periods=n).mean().to_numpy())

    def ema(self, n):
        return self._memo(("ema", n), lambda: pd.Series(self.c).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy())

    def tr(self):
        def f():
            pc = np.concatenate([[np.nan], self.c[:-1]])
            return np.nanmax(np.vstack([self.h - self.l, np.abs(self.h - pc), np.abs(self.l - pc)]), axis=0)
        return self._memo(("tr",), f)

    def atr(self, n):
        return self._memo(("atr", n), lambda: pd.Series(self.tr()).rolling(n, min_periods=n).mean().to_numpy())

    def low_n(self, n):            # n일 최저 저가(t 포함)
        return self._memo(("lown", n), lambda: pd.Series(self.l).rolling(n, min_periods=n).min().to_numpy())

    def high_n(self, n):           # n일 최고 고가(t 포함)
        return self._memo(("highn", n), lambda: pd.Series(self.h).rolling(n, min_periods=n).max().to_numpy())

    def close_high(self, n):       # n일 최고 종가(t 포함)
        return self._memo(("chigh", n), lambda: pd.Series(self.c).rolling(n, min_periods=1).max().to_numpy())

    def close_low(self, n):
        return self._memo(("clow", n), lambda: pd.Series(self.c).rolling(n, min_periods=n).min().to_numpy())

    def ret(self):
        return self._memo(("ret",), lambda: np.concatenate([[np.nan], self.c[1:] / self.c[:-1] - 1]))

    def rvol(self, n):             # 연율 실현 변동성
        return self._memo(("rvol", n), lambda: pd.Series(self.ret()).rolling(n, min_periods=n).std().to_numpy() * np.sqrt(252))

    def range_pct(self):           # (고가-저가)/전일 종가
        def f():
            pc = np.concatenate([[np.nan], self.c[:-1]])
            return (self.h - self.l) / pc
        return self._memo(("rangep",), f)

    def bb_width(self, n=20, k=2.0):
        def f():
            s = pd.Series(self.c)
            return (2 * k * s.rolling(n, min_periods=n).std() / s.rolling(n, min_periods=n).mean()).to_numpy()
        return self._memo(("bbw", n, k), f)

    def dd252(self):               # 252일 최고 종가 대비 낙폭
        return self._memo(("dd252",), lambda: self.c / self.close_high(252) - 1)


_LONG = None


def load() -> dict[str, Bars]:
    global _LONG
    if _LONG is None:
        df = pd.read_parquet(CACHE / "p2_long.parquet").sort_values(["sym", "date"])
        out = {}
        for sym, g in df.groupby("sym"):
            g = g.dropna(subset=["Open", "High", "Low", "Close"])
            g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0)]
            f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
            out[sym] = Bars(sym, g.date.to_numpy(), g.Open.to_numpy() * f, g.High.to_numpy() * f, g.Low.to_numpy() * f,
                            g.Close.to_numpy() * f, g.Volume.to_numpy(float))
        _LONG = out
    return _LONG


def universe(min_days_before: int = 252, start: str = "2026-01-02") -> list[str]:
    """그 창 시작 전에 min_days_before 거래일 이상 자료가 있는 주식(ETF 제외)."""
    out = []
    for s, b in load().items():
        if s in ETFS:
            continue
        if b.idx(start) >= min_days_before:
            out.append(s)
    return sorted(out)


def sources() -> dict:
    return json.loads((CACHE / "p2_universe.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- 정책
class Policy:
    """기본 = 매수·보유. 하위 클래스가 on_open/stop/limit_buys를 덮어쓴다. b = Bars, t = 오늘 인덱스."""
    name = "매수·보유"

    def on_open(self, b: Bars, t: int, st: dict):
        return None

    def stop(self, b: Bars, t: int, st: dict):
        return None

    def limit_buys(self, b: Bars, t: int, st: dict):
        return ()


# ----------------------------------------------------------------------------- 엔진
def simulate(b: Bars, a: int, z: int, policy: Policy, slip: float = 0.0):
    """a..z(포함) 거래일. a-1 종가에 전량 보유(가치 1)로 시작. 반환: 종가 가치 배열(길이 z-a+2, 첫 값 1), 주기 목록, 이벤트."""
    c, o, h, l = b.c, b.o, b.h, b.l
    cash, sh = 0.0, 1.0 / c[a - 1]
    st = {"pos": 1.0, "entry_px": c[a - 1], "entry_t": a - 1, "peak": c[a - 1], "high_peak": h[a - 1],
          "exit_px": None, "exit_t": None, "trough": None, "days_out": 0, "n_exits": 0, "last_reason": None,
          "start_t": a}
    vals = [1.0]
    pos_close = []
    cycles, events = [], []

    def equity(px):
        return cash + sh * px

    def set_pos(target, px, t, reason):
        nonlocal cash, sh
        eq = equity(px)
        cur = sh * px / eq if eq > 0 else 0.0
        if abs(target - cur) < 1e-9:
            return
        if target < cur:                                   # 매도
            fill = px * (1 - slip)
            sell_sh = sh - target * eq / px
            cash += sell_sh * fill
            sh -= sell_sh
            if target <= 1e-9:
                sh = 0.0
                st.update(exit_px=fill, exit_t=t, trough=fill, days_out=0, n_exits=st["n_exits"] + 1, last_reason=reason)
                events.append((b.days[t], "sell", reason, fill))
        else:                                              # 매수
            fill = px * (1 + slip)
            buy_val = (target - cur) * eq
            buy_val = min(buy_val, cash)
            sh += buy_val / fill
            cash -= buy_val
            if cur <= 1e-9:
                if st["exit_px"] is not None:
                    cycles.append({"exit_day": b.days[st["exit_t"]], "exit_px": st["exit_px"], "re_day": b.days[t],
                                   "re_px": fill, "gain": st["exit_px"] / fill - 1, "days_out": t - st["exit_t"],
                                   "reason": st["last_reason"], "open": False})
                st.update(entry_px=fill, entry_t=t, peak=fill, high_peak=fill, exit_px=None, exit_t=None, trough=None)
            events.append((b.days[t], "buy", reason, fill))
        eqn = equity(px)
        st["pos"] = sh * px / eqn if eqn > 0 else 0.0

    for t in range(a, z + 1):
        tgt = policy.on_open(b, t, st)
        if tgt is not None and np.isfinite(o[t]):
            set_pos(float(np.clip(tgt, 0.0, 1.0)), o[t], t, "open")
        if st["pos"] > 1e-9:
            L = policy.stop(b, t, st)
            if L is not None and np.isfinite(L):
                if o[t] <= L:
                    set_pos(0.0, o[t], t, "stop_gap")
                elif l[t] <= L:
                    set_pos(0.0, L, t, "stop")
        for lvl, tgt in sorted(policy.limit_buys(b, t, st), key=lambda x: -x[0]):
            if lvl is None or not np.isfinite(lvl):
                continue
            if l[t] <= lvl and st["pos"] < tgt - 1e-9:
                set_pos(tgt, min(o[t], lvl), t, "limit")
        vals.append(equity(c[t]))
        pos_close.append(st["pos"])
        # 종가 뒤 상태(다음 날 판단에 쓰임)
        if st["pos"] > 1e-9:
            st["peak"] = max(st["peak"], c[t])
            st["high_peak"] = max(st["high_peak"], h[t])
        else:
            st["days_out"] += 1
            st["trough"] = min(st["trough"], c[t]) if st["trough"] is not None else c[t]
        if hasattr(policy, "after_close"):
            policy.after_close(b, t, st)
    if st["pos"] <= 1e-9 and st["exit_px"] is not None:
        cycles.append({"exit_day": b.days[st["exit_t"]], "exit_px": st["exit_px"], "re_day": None, "re_px": c[z],
                       "gain": st["exit_px"] / c[z] - 1, "days_out": z - st["exit_t"], "reason": st["last_reason"],
                       "open": True})
    simulate.last_pos = np.array(pos_close)        # 그날 종가 시점 보유 비율(노출 계산용)
    return np.array(vals), cycles, events


def _mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def run_universe(policy_factory, syms: list[str] | None = None, runs=None, slip: float = 0.0):
    """정책을 종목마다 새로 만들어(policy_factory()) 각 실행 창에서 돌린다.
    반환: {창: {"per": DataFrame(종목별 수익·낙폭·B&H·빠짐 수·노출), "port": 동일가중 일별 포트폴리오(규칙·B&H), "cycles": 주기 목록}}."""
    data = load()
    runs = runs or RUNS
    out = {}
    for rname, (d0, d1) in runs.items():
        rows, cyc, daily_r, daily_bh, expo = [], [], {}, {}, {}
        for s in (syms or universe(252, d0)):
            b = data[s]
            a, z = b.idx(d0), b.idx(d1, "right") - 1
            if a < 252 or z <= a:
                continue
            v, cycles, ev = simulate(b, a, z, policy_factory(), slip)
            bh = np.concatenate([[1.0], b.c[a:z + 1] / b.c[a - 1]])
            dd = b.days[a:z + 1]
            daily_r[s] = pd.Series(v[1:] / v[:-1] - 1, index=dd)
            daily_bh[s] = pd.Series(bh[1:] / bh[:-1] - 1, index=dd)
            expo[s] = pd.Series(simulate.last_pos, index=dd)
            rows.append({"sym": s, "ret": v[-1] - 1, "mdd": _mdd(v), "bh_ret": bh[-1] - 1, "bh_mdd": _mdd(bh),
                         "exits": sum(1 for e in ev if e[1] == "sell"), "days": z - a + 1})
            for cy in cycles:
                cy["sym"] = s
            cyc += cycles
        R, B = pd.DataFrame(daily_r).sort_index(), pd.DataFrame(daily_bh).sort_index()
        out[rname] = {"per": pd.DataFrame(rows), "R": R, "B": B, "E": pd.DataFrame(expo).sort_index(), "cycles": cyc}
    return out


def same_mdd_excess(rule_curve: np.ndarray, bh_daily: np.ndarray) -> tuple[float, float]:
    """같은 최대 낙폭 고정 노출 효율선 대비 초과 수익: 매수·보유 일수익에 노출 e(0~1, 0.005 간격)를 곱한 경로 가운데
    낙폭이 규칙보다 깊지 않은 것의 최대 수익을 빼 준다. 반환 (초과 수익, 그 e)."""
    target = _mdd(rule_curve)
    best_ret, best_e = -1.0, 0.0
    for e in np.arange(0.0, 1.0001, 0.005):
        cur = np.concatenate([[1.0], np.cumprod(1 + e * bh_daily)])
        if _mdd(cur) >= target - 1e-12 and cur[-1] - 1 > best_ret:
            best_ret, best_e = cur[-1] - 1, e
    return float(rule_curve[-1] - 1 - best_ret), float(best_e)


def window_stats(res: dict) -> dict:
    """run_universe 결과 → 창별 요약(2026은 앞·뒤로 잘라서도). 동일가중 포트폴리오(매일 재조정) 기준과 종목별 기준을 함께."""
    out = {}
    for rname, r in res.items():
        for w in SLICES.get(rname, [rname]):
            d0, d1 = WINDOWS[w]
            R = r["R"][(r["R"].index >= d0) & (r["R"].index <= d1)]
            B = r["B"][(r["B"].index >= d0) & (r["B"].index <= d1)]
            if not len(R):
                continue
            pr = np.concatenate([[1.0], (1 + R.mean(axis=1).fillna(0)).cumprod().to_numpy()])
            pb = np.concatenate([[1.0], (1 + B.mean(axis=1).fillna(0)).cumprod().to_numpy()])
            sr = (1 + R.fillna(0)).prod() - 1
            sb = (1 + B.fillna(0)).prod() - 1
            smdd = R.fillna(0).apply(lambda x: _mdd(np.concatenate([[1.0], (1 + x).cumprod().to_numpy()])))
            bmdd = B.fillna(0).apply(lambda x: _mdd(np.concatenate([[1.0], (1 + x).cumprod().to_numpy()])))
            cyc = [cy for cy in r["cycles"] if d0 <= cy["exit_day"] <= d1]
            g = np.array([cy["gain"] for cy in cyc]) if cyc else np.array([])
            win, loss = g[g > 0], g[g <= 0]
            exc, e_eq = same_mdd_excess(pr, B.mean(axis=1).fillna(0).to_numpy())
            out[w] = {
                "종목수": int(R.shape[1]),
                "포트_수익": float(pr[-1] - 1), "포트_낙폭": _mdd(pr), "포트BH_수익": float(pb[-1] - 1), "포트BH_낙폭": _mdd(pb),
                "효율선초과": exc, "효율선_e": e_eq,
                "평균노출": float(r["E"][(r["E"].index >= d0) & (r["E"].index <= d1)].mean().mean()) if "E" in r else float("nan"),
                "종목_수익차_중앙": float((sr - sb).median()), "종목_낙폭개선_중앙": float((smdd - bmdd).median()),
                "수익_나은종목비율": float((sr > sb + 1e-12).mean()), "최악5%_종목수익": float(sr.quantile(0.05)),
                "최악5%_BH": float(sb.quantile(0.05)),
                "손절주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
                "평균이익": float(win.mean()) if len(win) else float("nan"),
                "평균손실": float(loss.mean()) if len(loss) else float("nan"),
                "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
                "기대값": float(g.mean()) if len(g) else float("nan"),
            }
    return out


def fmt(ws: dict, keys=("2026", "2026앞", "2026뒤", "2025", "2022")) -> str:
    parts = []
    for k in keys:
        if k in ws:
            s = ws[k]
            parts.append(f"{k} 포트 {s['포트_수익']:+.1%}/{s['포트_낙폭']:.1%}(BH {s['포트BH_수익']:+.1%}/{s['포트BH_낙폭']:.1%}) "
                         f"효율선초과 {s['효율선초과']:+.1%} 노출 {s['평균노출']:.2f} "
                         f"승률 {s['승률']:.0%} 손익비 {s['손익비']:.2f} 기대 {s['기대값']:+.1%} n{s['손절주기']}")
    return " | ".join(parts)


def save(name: str, payload: dict):
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- 자체 점검
class _Trail(Policy):
    def __init__(self, x): self.x = x
    def stop(self, b, t, st): return st["peak"] * (1 - self.x)
    def on_open(self, b, t, st):          # 판 뒤 20일 최고 종가를 넘으면 다시 산다
        if st["pos"] <= 1e-9 and t - 1 >= 0 and b.c[t - 1] >= b.close_high(20)[t - 1] and st["days_out"] >= 1:
            return 1.0
        return None


def check():
    data = load()
    syms = universe(252, "2026-01-02")
    print("2026 종목 수", len(syms), "· 2022 종목 수", len(universe(252, "2022-01-03")), "· 2018", len(universe(252, "2018-01-02")))
    res = run_universe(Policy, syms[:40], runs={"2026": RUNS["2026"]})
    per = res["2026"]["per"]
    d = float((per.ret - per.bh_ret).abs().max())
    print("매수·보유 정책 vs B&H 최대 차(0이어야 함)", d)
    ws = window_stats(run_universe(lambda: _Trail(0.15), syms, runs={"2026": RUNS["2026"], "2022": RUNS["2022"]}))
    print("예시 추적 15% + 20일 신고가 재진입:", fmt(ws))
    # 시점 점검: t 이후 가격을 바꿔도 t까지 가치가 같아야 한다
    b = data[syms[0]]
    a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
    v1, _, _ = simulate(b, a, z, _Trail(0.15))
    T = a + 60
    b2 = Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
    rng = np.random.default_rng(0)
    for arr in (b2.o, b2.h, b2.l, b2.c):
        arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
    b2.h[T + 1:] = np.maximum.reduce([b2.h[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
    b2.l[T + 1:] = np.minimum.reduce([b2.l[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
    v2, _, _ = simulate(b2, a, z, _Trail(0.15))
    print("미래 교란 뒤 t까지 가치 최대 차(0이어야 함)", float(np.max(np.abs(v1[:T - a + 2] - v2[:T - a + 2]))))


if __name__ == "__main__":
    check()
