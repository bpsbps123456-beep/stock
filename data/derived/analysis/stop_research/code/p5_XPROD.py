"""5단계 XPROD(2026-10-11): 같은 손절·비중 규칙 묶음이 다른 3배 상품 9개에서도 통하나(상품별 우연 걸러내기).

사용자: "SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?" 이 갈래는 SOXL·KORU 결과를 보지 않고(독립) 같은 사전 고정 규칙
22개(+덜 담기 3개)를 TQQQ·UPRO·TNA·TECL·YINN·EDC·FAS·LABU·NUGT의 기초로 합성한 3배 상품에 건다.

합성(사전 고정): lev5.synth(기초, 3, 보수 0.95%, 추가 비용). 추가 비용 = 실제 상품과 겹치는 기간의 연 로그 비용(lev5.validate 방식,
  추가 비용 0으로 합성한 것과 비교). TQQQ 기초는 ^NDX에 1999-03-11부터 QQQ, UPRO 기초는 ^GSPC에 1993-02-01부터 SPY를 이었다
  (배당 포함·실제 시가; 그 전 지수 구간은 시가 = 전일 종가라 시가 체결이 사실상 종가 체결). NUGT는 2011-11 전·2020-03 뒤가 2배라
  3배 구간(2011-12-05~2020-03-20)에서만 비용을 쟀고, 합성은 전 기간 3배다.
엔진(직접 짠 일별 루프): t일 종가까지의 기초 지표로 목표 비율 w를 정하고 t+1 시가에 맞춘다. 나머지 현금 0%(민감도 연 4%).
  맞춤 방식: 매일(F, V 매일) / 주 첫 거래일 시가만(V 주 1회) / 목표가 바뀔 때만(T·VS·TV·S; 사이엔 비율이 흘러감).
  S(넓은 추적 손절): 산 뒤 상품 최고 종가 × (1 − x)를 t−1까지 정보로 장중 스탑(시가가 이미 아래면 시가). 판 날 종가부터
  기초 > SMA50이면 다음 시가에 전부 재진입. 미끄러짐 s는 거래한 비율 |Δw| × s만큼 가치에서 뺀다(사이 흐름 포함).
  상품마다 합성 첫날부터 2026-10-09까지 한 번 이어 돌린 곡선을 창마다 잘라 잰다(창 시작 때 규칙 상태가 이어짐).
  기초 첫 252거래일은 지표 준비 구간이라 평가하지 않는다. '긴 역사' = 그다음 날 ~ 2025-12-31.
평가: 효율선초과 = 규칙 수익 − (낙폭이 규칙보다 깊지 않은 '덜 담기' w∈{0, 0.01, …, 1} 매일 맞춤 가운데 최고 수익).
  긴 역사는 연율(CAGR) 차. 같은 노출 = 규칙의 평균 보유 비율과 같은 고정 w 대비.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lev5  # noqa: E402
import stock_lab as S  # noqa: E402

RESULTS = HERE.parent / "results"
LEV, FEE, WARM = 3.0, 0.0095, 252
END, LONG_END = "2026-10-09", "2025-12-31"
SLIP = 0.005
CASH4 = 0.04
WIPE = -0.90

PRODUCTS = {  # 상품: (기초 설명, 기초 Bars, 비용 대조 구간 또는 None = 겹치는 전 기간)
    "TQQQ": ("^NDX→QQQ(1999-03-11~)", lambda: lev5.splice("^NDX", "QQQ", "1999-03-11"), None),
    "UPRO": ("^GSPC→SPY(1993-02-01~)", lambda: lev5.splice("^GSPC", "SPY", "1993-02-01"), None),
    "TNA": ("IWM", lambda: lev5.bars("IWM"), None),
    "TECL": ("XLK", lambda: lev5.bars("XLK"), None),
    "YINN": ("FXI", lambda: lev5.bars("FXI"), None),
    "EDC": ("EEM", lambda: lev5.bars("EEM"), None),
    "FAS": ("XLF", lambda: lev5.bars("XLF"), None),
    "LABU": ("XBI", lambda: lev5.bars("XBI"), None),
    "NUGT": ("GDX", lambda: lev5.bars("GDX"), ("2011-12-05", "2020-03-20")),
}

WINDOWS = {  # 사전 고정
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31"),
    "2022": ("2022-01-03", "2022-12-30"), "2020": ("2020-01-02", "2020-12-31"), "2008": ("2008-01-02", "2008-12-31"),
    "2000~02": ("2000-03-01", "2002-10-31"),
}
CRASH = {  # 폭락 구간(맥락, 고르기에 쓰지 않음)
    "2000-03~2002-10": ("2000-03-01", "2002-10-31"), "2007-10~2009-03": ("2007-10-01", "2009-03-31"),
    "2011(5~9월)": ("2011-05-02", "2011-09-30"), "2015-06~2016-02": ("2015-06-01", "2016-02-29"),
    "2018 4분기": ("2018-10-01", "2018-12-31"), "2020-02~03": ("2020-02-03", "2020-03-31"), "2022": ("2022-01-03", "2022-12-30"),
    "2025 관세(2~4월)": ("2025-02-03", "2025-04-30"), "2026 7월": ("2026-07-01", "2026-07-31"),
}
EXC_WINDOWS = ("2026앞", "2026뒤", "2025", "긴역사")


# ----------------------------------------------------------------------------- 합성·대조
def validate_range(product: str, syn: S.Bars, rng) -> dict:
    """lev5.validate와 같은 식을 rng(시작, 끝) 안 겹치는 날에서만."""
    if rng is None:
        return lev5.validate(product, syn)
    a = lev5.bars(product)
    common = np.intersect1d(a.days, syn.days)
    common = common[(common >= rng[0]) & (common <= rng[1])]
    ia, isy = np.searchsorted(a.days, common), np.searchsorted(syn.days, common)
    ra, rs = a.c[ia][1:] / a.c[ia][:-1] - 1, syn.c[isy][1:] / syn.c[isy][:-1] - 1
    yrs = len(common) / 252
    ca, cs = a.c[ia][-1] / a.c[ia][0], syn.c[isy][-1] / syn.c[isy][0]
    mdd = lambda v: float((v / np.maximum.accumulate(v) - 1).min())
    return {"product": product, "시작": str(common[0]), "일수": int(len(common)), "일수익상관": float(np.corrcoef(ra, rs)[0, 1]),
            "연비용_로그": float(-np.log(ca / cs) / yrs), "실제_낙폭": mdd(a.c[ia]), "합성_낙폭": mdd(syn.c[isy])}


def build():
    """상품별 (기초, 합성, 대조) — 추가 비용은 대조로 얻어 넣는다."""
    out = {}
    for p, (desc, uf, rng) in PRODUCTS.items():
        u = uf()
        u = S.Bars(u.sym, u.days[u.days <= END], *(x[u.days <= END] for x in (u.o, u.h, u.l, u.c, u.v)))
        s0 = lev5.synth(u, LEV, FEE, 0.0)
        v0 = validate_range(p, s0, rng)
        extra = v0["연비용_로그"]
        s1 = lev5.synth(u, LEV, FEE, extra)
        v1 = validate_range(p, s1, rng)
        out[p] = {"desc": desc, "u": u, "P": s1, "extra": extra, "val0": v0, "val1": v1}
    return out


# ----------------------------------------------------------------------------- 규칙
def indicators(u: S.Bars) -> dict:
    s = pd.Series(u.c)
    sig = s.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    med = sig.expanding(min_periods=20).median().shift(1)          # t−1까지의 확장 중앙값
    return {"c": u.c, "sma": {n: s.rolling(n, min_periods=n).mean().to_numpy() for n in (50, 100, 150, 200)},
            "sig": sig.to_numpy(), "med": med.to_numpy()}


def _trend(c, m, band):
    out = np.ones(len(c))
    st = 1.0
    for t in range(len(c)):
        if np.isfinite(m[t]):
            if st == 1.0 and c[t] < m[t] * (1 - band):
                st = 0.0
            elif st == 0.0 and c[t] > m[t] * (1 + band):
                st = 1.0
        out[t] = st
    return out


def _fin(x, default=1.0):
    return np.where(np.isfinite(x), x, default)


def rules(ind: dict) -> list[dict]:
    """사전 고정 규칙 묶음. tgt[t] = t일 종가에 정한 목표(t+1 시가 체결)."""
    c, sma, sig, med = ind["c"], ind["sma"], ind["sig"], ind["med"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = _fin(med / sig, 1.0)
        tv3 = _fin(3 * sig, np.nan)
    R = [{"id": "R0", "g": "R0", "name": "그냥 보유(w=1)", "mode": "event", "tgt": np.ones(len(c))}]
    for w in (0.33, 0.5, 0.67):
        R.append({"id": f"F{w:g}", "g": "F", "name": f"덜 담기 w={w:g} 매일 맞춤", "mode": "daily", "tgt": np.full(len(c), w)})
    for n in (50, 100, 150, 200):
        for b in (0.0, 0.03):
            R.append({"id": f"T{n}" + ("b3" if b else ""), "g": "T", "mode": "event", "tgt": _trend(c, sma[n], b),
                      "name": f"기초<SMA{n}" + (" (띠 3%)" if b else "") + " → 0"})
    with np.errstate(invalid="ignore"):
        R.append({"id": "T200h", "g": "T", "mode": "event", "name": "기초<SMA200 → 0.5",
                  "tgt": np.where(np.isfinite(sma[200]) & (c < sma[200]), 0.5, 1.0)})
        R += [
            {"id": "V1", "g": "V", "mode": "daily", "name": "w=m/σ20 매일", "tgt": np.clip(ratio, 0, 1)},
            {"id": "V075", "g": "V", "mode": "daily", "name": "w=0.75·m/σ20 매일", "tgt": np.clip(0.75 * ratio, 0, 1)},
            {"id": "V45", "g": "V", "mode": "daily", "name": "w=0.45/(3σ20) 매일", "tgt": np.clip(_fin(0.45 / tv3, 1.0), 0, 1)},
            {"id": "V60", "g": "V", "mode": "daily", "name": "w=0.60/(3σ20) 매일", "tgt": np.clip(_fin(0.60 / tv3, 1.0), 0, 1)},
            {"id": "V1w", "g": "V", "mode": "weekly", "name": "w=m/σ20 주 1회", "tgt": np.clip(ratio, 0, 1)},
            {"id": "V60w", "g": "V", "mode": "weekly", "name": "w=0.60/(3σ20) 주 1회", "tgt": np.clip(_fin(0.60 / tv3, 1.0), 0, 1)},
        ]
        for k in (1.5, 2.0):
            R.append({"id": f"VS{k:g}", "g": "VS", "mode": "event", "name": f"σ20>{k:g}·m → 0",
                      "tgt": np.where(np.isfinite(sig) & np.isfinite(med) & (sig > k * med), 0.0, 1.0)})
        up = ~(np.isfinite(sma[200]) & (c < sma[200]))
        calm = ~(np.isfinite(sig) & np.isfinite(med) & (sig >= 1.5 * med))
        R.append({"id": "TV", "g": "TV", "mode": "event", "name": "SMA200 위·σ20<1.5m 둘 다 1, 하나 0.5, 없음 0",
                  "tgt": np.where(up & calm, 1.0, np.where(up | calm, 0.5, 0.0))})
    re50 = np.isfinite(sma[50]) & (c > np.where(np.isfinite(sma[50]), sma[50], np.inf))
    for x in (0.30, 0.40, 0.50):
        R.append({"id": f"S{int(x * 100)}", "g": "S", "mode": "stop", "stop": x, "re": re50, "tgt": None,
                  "name": f"상품 추적 −{int(x * 100)}% 장중 → 기초>SMA50 재진입"})
    for x in (0.30, 0.40, 0.50):   # 사후 민감도(고르기 대상 아님): 판 날 기초가 아직 SMA50 위면 바로 되사는 문제를 거른 판
        R.append({"id": f"S{int(x * 100)}x", "g": "Sx", "mode": "stop", "stop": x, "re": re50, "tgt": None, "cross": True,
                  "name": f"(민감도) 추적 −{int(x * 100)}% → 기초가 SMA50 아래로 갔다가 위로 오면 재진입"})
    return R


# ----------------------------------------------------------------------------- 엔진
def week_ids(days):
    d = pd.to_datetime(pd.Series(days))
    iso = d.dt.isocalendar()
    return (iso.year.astype(int) * 100 + iso.week.astype(int)).to_numpy()


def run(P: S.Bars, rule: dict, a: int, z: int, slip=0.0, cash_r=0.0, wk=None):
    """a−1 종가에 가치 1, 규칙의 그 시점 비율로 시작. 반환 vals(a−1..z 종가), pos(a..z 종가 비율), trd(a..z 거래 |Δw|)."""
    o, l, c = P.o.tolist(), P.l.tolist(), P.c.tolist()
    n = z - a + 1
    vals = np.empty(n + 1); pos = np.empty(n); trd = np.zeros(n)
    mode = rule["mode"]
    g = cash_r / 252
    if mode == "stop":
        x, re = rule["stop"], rule["re"].tolist()
        cross = rule.get("cross", False)      # 민감도: 판 뒤 기초가 SMA50 아래로 한 번 내려간 다음 위로 올라와야 재진입
        below = False
        w = 1.0
    else:
        tgt = rule["tgt"].tolist()
        w = float(tgt[a - 1])
    sh = w / c[a - 1]; cash = 1.0 - w
    last = w; in_pos = True; peak = c[a - 1]; reenter = False
    vals[0] = 1.0
    wkl = wk.tolist() if wk is not None else None
    for k in range(n):
        t = a + k
        if g:
            cash *= 1 + g
        po = o[t]
        eq = cash + sh * po
        if mode == "stop":
            if reenter:
                eq -= slip * eq
                sh = eq / po; cash = 0.0; in_pos = True; peak = po; reenter = False; trd[k] += 1.0
            if in_pos:
                Lp = peak * (1 - x)
                px = po if po <= Lp else (Lp if l[t] <= Lp else None)
                if px is not None:
                    val = sh * px
                    eqx = cash + val
                    trd[k] += val / eqx if eqx > 0 else 0.0
                    cash += val * (1 - slip); sh = 0.0; in_pos = False; below = False
        else:
            tg = tgt[t - 1]
            if mode == "daily" or (mode == "weekly" and wkl[t] != wkl[t - 1]) or (mode == "event" and tg != last):
                wd = sh * po / eq if eq > 0 else 0.0
                d = abs(tg - wd)
                if d > 1e-12:
                    eq -= slip * d * eq
                    sh = tg * eq / po; cash = eq - sh * po
                    trd[k] += d
                last = tg
        vc = cash + sh * c[t]
        vals[k + 1] = vc
        pos[k] = sh * c[t] / vc if vc > 0 else 0.0
        if mode == "stop":
            if in_pos:
                if c[t] > peak:
                    peak = c[t]
            elif not cross and re[t]:
                reenter = True
            elif cross:
                if not re[t]:
                    below = True
                elif below:
                    reenter = True
    return vals, pos, trd


W_GRID = np.round(np.arange(0, 101) / 100, 2)


def const_paths(P: S.Bars, a: int, z: int, slip=0.0, cash_r=0.0):
    """덜 담기 w∈W_GRID를 매일 시가에 맞춘 종가 가치(행 = a−1..z, 열 = w). run(daily, 상수 목표)과 같은 식."""
    W = W_GRID
    o, c = P.o, P.c
    n = z - a + 1
    out = np.empty((n + 1, len(W))); out[0] = 1.0
    sh = W / c[a - 1]; cash = 1.0 - W
    g = cash_r / 252
    for k in range(n):
        t = a + k
        if g:
            cash = cash * (1 + g)
        eq = cash + sh * o[t]
        wd = sh * o[t] / eq
        eq = eq - slip * np.abs(W - wd) * eq
        sh = W * eq / o[t]; cash = eq - sh * o[t]
        out[k + 1] = cash + sh * c[t]
    return out


# ----------------------------------------------------------------------------- 지표
def mdd(v):
    v = np.asarray(v)
    return float((v / np.maximum.accumulate(v) - 1).min())


def mdd_cols(M):
    return (M / np.maximum.accumulate(M, axis=0) - 1).min(axis=0)


def cagr(v):
    return float(v[-1] ** (252 / (len(v) - 1)) - 1) if len(v) > 1 and v[-1] > 0 else -1.0


def underwater_years(v):
    peak, start, longest, cur = v[0], 0, 0, 0
    for i, x in enumerate(v):
        if x >= peak:
            peak = x; cur = 0
        else:
            cur += 1
            longest = max(longest, cur)
    return longest / 252, bool(v[-1] < peak)


def window_idx(P, d0, d1, a, z):
    i0, i1 = P.idx(d0), P.idx(d1, "right") - 1
    if i0 < a or i1 > z or i1 <= i0:
        return None
    return i0, i1


def evaluate(prod: dict, rs: list[dict], slip=0.0, cash_r=0.0, detail=True):
    """한 상품의 모든 규칙 × 창 결과."""
    P = prod["P"]
    a = WARM
    z = P.idx(END, "right") - 1
    wk = week_ids(P.days)
    F = const_paths(P, a, z, slip, cash_r)
    wins = dict(WINDOWS)
    wins["긴역사"] = (P.days[a], LONG_END)
    if detail:
        wins.update({f"폭락:{k}": v for k, v in CRASH.items()})
    # 창별 효율선 재료
    fz = {}
    for wn, (d0, d1) in wins.items():
        ix = window_idx(P, d0, d1, a, z) if wn != "긴역사" else (a, P.idx(LONG_END, "right") - 1)
        if ix is None:
            continue
        i0, i1 = ix
        Fs = F[i0 - a: i1 - a + 2] / F[i0 - a]
        fz[wn] = {"ix": ix, "ret": Fs[-1] - 1, "mdd": mdd_cols(Fs),
                  "cagr": Fs[-1] ** (252 / (len(Fs) - 1)) - 1 if wn == "긴역사" else None}
    out = {}
    for r in rs:
        vals, pos, trd = run(P, r, a, z, slip, cash_r, wk)
        res = {}
        for wn, fw in fz.items():
            i0, i1 = fw["ix"]
            v = vals[i0 - a: i1 - a + 2] / vals[i0 - a]
            ps, td = pos[i0 - a: i1 - a + 1], trd[i0 - a: i1 - a + 1]
            yrs = len(td) / 252
            m = mdd(v)
            ok = fw["mdd"] >= m - 1e-12
            long = wn == "긴역사"
            met = cagr(v) if long else float(v[-1] - 1)
            fm = fw["cagr"] if long else fw["ret"]
            kb = int(np.argmax(np.where(ok, fm, -np.inf)))
            ke = int(np.clip(round(float(ps.mean()) * 100), 0, 100))
            d = {"ret": float(v[-1] - 1), "mdd": m, "expo": float(ps.mean()),
                 "eff": float(met - fm[kb]), "eff_w": float(W_GRID[kb]), "eff_ret": float(fm[kb]),
                 "same_exp": float(met - fm[ke]), "same_exp_w": float(W_GRID[ke]), "same_exp_mdd": float(fw["mdd"][ke]),
                 "trades_y": float((td >= 0.01).sum() / yrs), "turn_y": float(td.sum() / yrs)}
            if long:
                d["cagr"] = met
                d["same_mdd_F_mdd"] = float(fw["mdd"][kb])
                if detail:
                    rl = v[252:] / v[:-252] - 1
                    d["worst1y"] = float(rl.min()) if len(rl) else float("nan")
                    uw, still = underwater_years(v)
                    d["underwater_y"] = uw; d["underwater_now"] = still
                    d["start"] = str(P.days[i0])
            res[wn] = d
        if detail:  # 해별 수익(긴 역사 시작 다음 해 ~ 2026)
            yr = {}
            for y in range(int(P.days[a][:4]) + 1, 2027):
                ix = window_idx(P, f"{y}-01-01", f"{y}-12-31", a, z)
                if ix:
                    i0, i1 = ix
                    yr[str(y)] = float(vals[i1 - a + 1] / vals[i0 - a] - 1)
            res["해별"] = yr
        out[r["id"]] = res
    return out


# ----------------------------------------------------------------------------- 점검
def check_future_leak(prod: dict) -> float:
    """t 이후 기초·상품 가격을 흔들어도 t 종가까지 가치가 같아야 한다(모든 규칙 최대 차)."""
    P, u = prod["P"], prod["u"]
    a, z = WARM, P.idx(END, "right") - 1
    T = P.idx("2020-03-02")
    rng = np.random.default_rng(1)
    def shake(b):
        arrs = [x.copy() for x in (b.o, b.h, b.l, b.c)]
        for x in arrs:
            x[T + 1:] *= 1 + rng.normal(0, 0.05, len(x) - T - 1)
        o, h, l, c = arrs
        h[T + 1:] = np.maximum.reduce([h[T + 1:], o[T + 1:], c[T + 1:]])
        l[T + 1:] = np.minimum.reduce([l[T + 1:], o[T + 1:], c[T + 1:]])
        return S.Bars(b.sym, b.days, o, h, l, c, b.v)
    P2, u2 = shake(P), shake(u)
    r1, r2 = rules(indicators(u)), rules(indicators(u2))
    wk = week_ids(P.days)
    worst = 0.0
    for x, y in zip(r1, r2):
        v1 = run(P, x, a, z, 0.005, 0.0, wk)[0]
        v2 = run(P2, y, a, z, 0.005, 0.0, wk)[0]
        worst = max(worst, float(np.max(np.abs(v1[: T - a + 2] - v2[: T - a + 2]))))
    return worst


class _SPol(S.Policy):
    def __init__(self, x, re): self.x, self.re = x, re
    def stop(self, b, t, st): return st["peak"] * (1 - self.x)
    def on_open(self, b, t, st):
        if st["pos"] <= 1e-9 and self.re[t - 1]:
            return 1.0
        return None


class _TPol(S.Policy):
    def __init__(self, tgt): self.tgt = tgt
    def on_open(self, b, t, st):
        return float(self.tgt[t - 1]) if abs(self.tgt[t - 1] - st["pos"]) > 1e-9 and (self.tgt[t - 1] in (0.0, 1.0)) else None


def check_vs_stock_lab(prod: dict) -> dict:
    """같은 규칙을 stock_lab.simulate로 새로 돌린 것과 비교(창 시작에 보유 1)."""
    P, u = prod["P"], prod["u"]
    rs = {r["id"]: r for r in rules(indicators(u))}
    a, z = P.idx("2018-01-02"), P.idx("2022-12-30", "right") - 1
    wk = week_ids(P.days)
    out = {}
    r = rs["S30"]
    v1 = run(P, r, a, z, 0.0, 0.0, wk)[0]
    v2, _, _ = S.simulate(P, a, z, _SPol(0.30, r["re"]))
    out["S30"] = float(np.max(np.abs(v1 / v2 - 1)))
    r = rs["T200"]
    if r["tgt"][a - 1] == 1.0:
        v1 = run(P, r, a, z, 0.0, 0.0, wk)[0]
        v2, _, _ = S.simulate(P, a, z, _TPol(r["tgt"]))
        out["T200"] = float(np.max(np.abs(v1 / v2 - 1)))
    return out


def check_F_equals_frontier(prod: dict) -> float:
    """덜 담기 규칙(run, daily)과 효율선 재료(const_paths)의 같은 w 곡선이 같아야 한다(미끄러짐 0.5%로)."""
    P = prod["P"]
    a, z = WARM, P.idx(END, "right") - 1
    F = const_paths(P, a, z, SLIP)
    wk = week_ids(P.days)
    worst = 0.0
    for r in rules(indicators(prod["u"])):
        if r["g"] == "F":
            v = run(P, r, a, z, SLIP, 0.0, wk)[0]
            k = int(round(float(r["tgt"][0]) * 100))
            worst = max(worst, float(np.max(np.abs(v / F[:, k] - 1))))
    return worst


def real_check(prods: dict, ids: list[str]) -> dict:
    """실제 상품 가격에 같은 규칙(기초 신호)을 걸어 긴 역사 효율선초과(연)를 합성과 비교. 기간 = 실제 상장 252일 뒤 ~ 2025-12-31."""
    out = {}
    for p, d in prods.items():
        if p == "NUGT":
            continue
        real = lev5.bars(p)
        P, u = d["P"], d["u"]
        common = np.intersect1d(real.days, P.days)
        common = common[common <= LONG_END]
        ir, ip = np.searchsorted(real.days, common), np.searchsorted(P.days, common)
        R = S.Bars(p, common, real.o[ir], real.h[ir], real.l[ir], real.c[ir], real.v[ir])
        Psyn = S.Bars(p + "s", common, P.o[ip], P.h[ip], P.l[ip], P.c[ip], P.v[ip])
        ind = indicators(u)
        rs = {r["id"]: r for r in rules(ind)}
        a0 = max(1, int(np.searchsorted(common, real.days[min(252, len(real.days) - 1)])))
        z0 = len(common) - 1
        wk = week_ids(common)
        res = {}
        for B, tag in ((R, "실제"), (Psyn, "합성")):
            F = const_paths(B, a0, z0)
            Fc = F[-1] ** (252 / (len(F) - 1)) - 1
            Fm = mdd_cols(F)
            for i in ids:
                r = dict(rs[i])
                if r["tgt"] is not None:
                    r["tgt"] = r["tgt"][ip]
                else:
                    r["re"] = r["re"][ip]
                v = run(B, r, a0, z0, 0.0, 0.0, wk)[0]
                m = mdd(v)
                ok = Fm >= m - 1e-12
                res.setdefault(i, {})[tag] = {"cagr": cagr(v), "mdd": m, "eff": float(cagr(v) - np.max(np.where(ok, Fc, -np.inf)))}
            res.setdefault("_R0", {})[tag] = {"cagr": float(Fc[-1]), "mdd": float(Fm[-1])}
        res["_기간"] = f"{common[a0]}~{common[z0]}"
        out[p] = res
    return out


def sizing(prods: dict) -> dict:
    """긴 역사(설명용, 사후): 고정 w별 연율·최대 낙폭 → 연율 최대 w, 낙폭 −90%·−50%를 넘지 않는 최대 w."""
    out = {}
    for p, d in prods.items():
        P = d["P"]
        a, z = WARM, P.idx(LONG_END, "right") - 1
        F = const_paths(P, a, z)
        cg = F[-1] ** (252 / (len(F) - 1)) - 1
        md = mdd_cols(F)
        kb = int(np.argmax(cg))
        k90 = int(np.max(np.where(md > WIPE)[0]))
        k50 = int(np.max(np.where(md >= -0.5)[0]))
        out[p] = {"연율최대_w": float(W_GRID[kb]), "연율최대_연율": float(cg[kb]), "연율최대_낙폭": float(md[kb]),
                  "전멸피함_최대w": float(W_GRID[k90]), "전멸피함_연율": float(cg[k90]),
                  "낙폭50_최대w": float(W_GRID[k50]), "낙폭50_연율": float(cg[k50]),
                  "w0.33_연율": float(cg[33]), "w0.33_낙폭": float(md[33]), "보유_연율": float(cg[100]), "보유_낙폭": float(md[100])}
    return out


# ----------------------------------------------------------------------------- 집계
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p"


def aggregate(E: dict, ids: list[str]) -> dict:
    """규칙별 상품 9개 집계. E[p][rid][창]."""
    prods = list(E)
    agg = {}
    for i in ids:
        a = {}
        for wn in EXC_WINDOWS:
            xs = [E[p][i][wn]["eff"] for p in prods if wn in E[p][i]]
            se = [E[p][i][wn]["same_exp"] for p in prods if wn in E[p][i]]
            a[wn] = {"n": len(xs), "pos": int(sum(x > 1e-9 for x in xs)), "med": float(np.median(xs)),
                     "med_same_exp": float(np.median(se)), "pos_same_exp": int(sum(x > 1e-9 for x in se))}
        L = {p: E[p][i]["긴역사"] for p in prods}
        a["wipe_rule"] = int(sum(L[p]["mdd"] <= WIPE for p in prods))
        a["wipe_hold"] = int(sum(E[p]["R0"]["긴역사"]["mdd"] <= WIPE for p in prods))
        a["wipe_same_exp_F"] = int(sum(L[p]["same_exp_mdd"] <= WIPE for p in prods))
        a["wipe_same_mdd_F"] = int(sum(L[p]["same_mdd_F_mdd"] <= WIPE for p in prods))
        a["med_cagr"] = float(np.median([L[p]["cagr"] for p in prods]))
        a["med_mdd"] = float(np.median([L[p]["mdd"] for p in prods]))
        a["med_trades"] = float(np.median([L[p]["trades_y"] for p in prods]))
        a["med_turn"] = float(np.median([L[p]["turn_y"] for p in prods]))
        a["med_expo"] = float(np.median([L[p]["expo"] for p in prods]))
        a["med_worst1y"] = float(np.median([L[p]["worst1y"] for p in prods]))
        a["med_underwater"] = float(np.median([L[p]["underwater_y"] for p in prods]))
        for wn in WINDOWS:
            xs = [E[p][i][wn]["ret"] for p in prods if wn in E[p][i]]
            ms = [E[p][i][wn]["mdd"] for p in prods if wn in E[p][i]]
            a["ret_" + wn] = {"n": len(xs), "med": float(np.median(xs)) if xs else float("nan"),
                              "med_mdd": float(np.median(ms)) if ms else float("nan")}
        for cn in CRASH:
            k = "폭락:" + cn
            xs = [E[p][i][k]["ret"] for p in prods if k in E[p][i]]
            a["crash_" + cn] = {"n": len(xs), "med": float(np.median(xs)) if xs else float("nan")}
        front = []
        for p in prods:
            f = E[p][i]["2026앞"]
            n_days = WINDOWS_DAYS[p]
            ann = (1 + f["ret"]) ** (252 / n_days) - 1
            front.append(ann / max(abs(f["mdd"]), 0.01))
        a["med_calmar_front"] = float(np.median(front))
        agg[i] = a
    return agg


WINDOWS_DAYS: dict = {}


def main():
    prods = build()
    print("합성 대조:")
    for p, d in prods.items():
        v0, v1 = d["val0"], d["val1"]
        print(f"  {p:5s} ← {d['desc']:24s} 시작 {d['P'].days[0]} 대조 {v0['시작']}~ {v0['일수']}일 상관 {v0['일수익상관']:.3f} "
              f"추가비용 {d['extra'] * 100:+.2f}% 낙폭 실제 {v1['실제_낙폭']:.1%} 합성 {v1['합성_낙폭']:.1%}")
    R_all = None
    E, E_slip, E_cash = {}, {}, {}
    for p, d in prods.items():
        rs = rules(indicators(d["u"]))
        R_all = R_all or [(r["id"], r["g"], r["name"]) for r in rs]
        WINDOWS_DAYS[p] = (lambda P: (P.idx(WINDOWS["2026앞"][1], "right") - P.idx(WINDOWS["2026앞"][0])))(d["P"])
        E[p] = evaluate(d, rs)
        E_slip[p] = evaluate(d, rs, slip=SLIP, detail=False)
        E_cash[p] = evaluate(d, rs, cash_r=CASH4, detail=False)
        print(p, "평가 끝")
    ids = [x[0] for x in R_all]
    names = {x[0]: x[2] for x in R_all}
    groups = {x[0]: x[1] for x in R_all}
    agg = aggregate(E, ids)

    # 비교용(F)·보유를 뺀 규칙 = 고르기 대상. R0 포함 여부: 보유는 효율선초과 ≤ 0이라 넣어도 결과에 영향 없음 → 넣는다.
    cand = [i for i in ids if groups[i] not in ("F", "Sx")]
    # 대표 후보: 긴 역사 효율선초과 > 0 상품 수 최대(동률이면 중앙 초과 큰 것)
    rep = max(cand, key=lambda i: (agg[i]["긴역사"]["pos"], agg[i]["긴역사"]["med"]))
    G1 = max(cand, key=lambda i: agg[i]["2026앞"]["med"])
    G2 = max(cand, key=lambda i: agg[i]["긴역사"]["med"])
    G3 = max(cand, key=lambda i: agg[i]["med_calmar_front"])

    # '앞 > 0' 조건부 성공(상품×규칙 쌍)
    pairs = [(p, i) for p in E for i in cand]
    def rate(cond, wn):
        sel = [(p, i) for p, i in pairs if cond(p, i)]
        if not sel:
            return {"n": 0, "rate": float("nan")}
        return {"n": len(sel), "rate": float(np.mean([E[p][i][wn]["eff"] > 1e-9 for p, i in sel]))}
    cond_front = lambda p, i: E[p][i]["2026앞"]["eff"] > 1e-9
    succ = {wn: {"앞>0": rate(cond_front, wn), "무조건": rate(lambda p, i: True, wn)} for wn in ("2026뒤", "2025", "긴역사")}

    # 미끄러짐·현금 이자 민감도 집계
    agg_slip = {i: {wn: {"pos": int(sum(E_slip[p][i][wn]["eff"] > 1e-9 for p in E_slip if wn in E_slip[p][i])),
                         "med": float(np.median([E_slip[p][i][wn]["eff"] for p in E_slip if wn in E_slip[p][i]]))}
                    for wn in EXC_WINDOWS} for i in ids}
    for i in ids:
        agg_slip[i]["med_cagr"] = float(np.median([E_slip[p][i]["긴역사"]["cagr"] for p in E_slip]))
    agg_cash = {i: {wn: {"pos": int(sum(E_cash[p][i][wn]["eff"] > 1e-9 for p in E_cash if wn in E_cash[p][i])),
                         "med": float(np.median([E_cash[p][i][wn]["eff"] for p in E_cash if wn in E_cash[p][i]]))}
                    for wn in EXC_WINDOWS} for i in ids}
    for i in ids:
        agg_cash[i]["med_cagr"] = float(np.median([E_cash[p][i]["긴역사"]["cagr"] for p in E_cash]))

    # 점검
    leak = check_future_leak(prods["TQQQ"])
    xchk = {p: check_vs_stock_lab(prods[p]) for p in ("TQQQ", "LABU")}
    fchk = check_F_equals_frontier(prods["TQQQ"])
    print("미래 교란 최대 차", leak, "| stock_lab 대조", xchk, "| 덜 담기 곡선 = 효율선 재료 최대 차(0이어야 함)", fchk)
    picks = list(dict.fromkeys([rep, G1, G2, G3, "T200", "S30", "V1", "V1w"]))
    real = real_check(prods, picks)
    size = sizing(prods)

    payload = {
        "설명": __doc__, "규칙": {i: {"묶음": groups[i], "이름": names[i]} for i in ids}, "규칙수": len(cand), "덜담기수": 3,
        "상품수": len(E), "칸수": len(E) * len(ids),
        "합성대조": {p: {"기초": d["desc"], "합성시작": str(d["P"].days[0]), "긴역사시작": str(d["P"].days[WARM]),
                       "추가비용_연로그": d["extra"], "대조_추가비용0": d["val0"], "대조_추가비용반영": d["val1"]} for p, d in prods.items()},
        "집계": agg, "집계_미끄러짐0.5%": agg_slip, "집계_현금4%": agg_cash,
        "고르기": {"대표(긴역사 >0 상품 수 최대)": rep, "G1(2026앞 효율선초과 중앙 최대)": G1,
                 "G2(긴역사 효율선초과 중앙 최대)": G2, "G3(2026앞 칼마 중앙 최대)": G3},
        "앞>0 조건부 성공": succ, "실제상품 대조": real, "덜담기_크기(긴역사, 설명용)": size,
        "점검": {"미래교란_최대차": leak, "stock_lab_대조_최대상대차": xchk, "덜담기곡선_효율선재료_최대상대차": fchk},
        "상품별": E,
        "상품별_미끄러짐0.5%": {p: {i: {wn: {k: E_slip[p][i][wn][k] for k in ("ret", "mdd", "eff")} for wn in EXC_WINDOWS
                                      if wn in E_slip[p][i]} for i in ids} for p in E_slip},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "p5_XPROD.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_md(prods, E, agg, agg_slip, agg_cash, ids, names, groups, cand, rep, G1, G2, G3, succ, real, leak, xchk, fchk, size)
    print("대표", rep, "G1", G1, "G2", G2, "G3", G3)


# ----------------------------------------------------------------------------- 보고서
def write_md(prods, E, agg, agg_slip, agg_cash, ids, names, groups, cand, rep, G1, G2, G3, succ, real, leak, xchk, fchk, size):
    P9 = list(E)
    nm = lambda i: f"{names[i]} ({i})"
    L = []
    a_rep = agg[rep]
    L.append("# 5단계 XPROD: 같은 손절·비중 규칙이 다른 3배 상품 9개에서도 통하나 (2026-10-11)\n")
    L.append("> 사용자: \"SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?\" — 이 갈래는 SOXL·KORU를 보지 않고, 같은 사전 고정 규칙 "
             f"{len(cand)}개(+덜 담기 3개)를 다른 3배 상품 9개({', '.join(P9)})의 긴 합성 역사에 걸어 '상품별 우연'을 거른다.")
    L.append("> 스크립트 `code/p5_XPROD.py`, 숫자 `results/p5_XPROD.json`. 효율선초과 = 규칙 수익 − 낙폭이 규칙보다 깊지 않은 "
             "'덜 담기'(같은 상품을 w만 들고 나머지 현금, 매일 맞춤, w 0~1을 0.01 간격) 가운데 최고 수익. 긴 역사는 연율 차.\n")
    L.append("## 결론\n")
    L.append("@@CONCLUSION@@\n")

    # 한눈 표
    L.append("## 한눈 표 (상품 9개, 효율선초과 > 0 상품 수 / 전멸 상품 수 / 중앙값)\n")
    L.append("| 규칙 | 긴 역사 효율선 >0 | 2026 앞·뒤·2025 >0 | 전멸(−90%) | 긴 역사 연율·낙폭(중앙) | 미끄러짐 0.5% 긴 역사 >0 | 연 매매 |")
    L.append("|---|---|---|---|---|---|---|")
    for i in list(dict.fromkeys(["R0", "F0.33", "F0.5", rep, G2, G1, "TV", "S30", "S50"])):
        a = agg[i]
        L.append(f"| {nm(i)} | {a['긴역사']['pos']}/9 ({pp(a['긴역사']['med'])}) | {a['2026앞']['pos']}·{a['2026뒤']['pos']}·{a['2025']['pos']} | "
                 f"{a['wipe_rule']}/9 | {pct(a['med_cagr'])} · {a['med_mdd'] * 100:.0f}% | {agg_slip[i]['긴역사']['pos']}/9 | {a['med_trades']:.0f} |")
    L.append("")
    L.append("덜 담기 크기(긴 역사, 사후 설명용 — 고르기에 쓰지 않음): 고정 w별로 연율이 가장 높은 w와 낙폭 한도별 최대 w.\n")
    L.append("| 상품 | 보유 연율·낙폭 | 연율 최대 w · 연율 · 낙폭 | 낙폭 −90% 안 최대 w · 연율 | 낙폭 −50% 안 최대 w · 연율 | w=0.33 연율·낙폭 |")
    L.append("|---|---|---|---|---|---|")
    for p, zz in size.items():
        L.append(f"| {p} | {pct(zz['보유_연율'])} · {zz['보유_낙폭'] * 100:.0f}% | {zz['연율최대_w']:.2f} · {pct(zz['연율최대_연율'])} · {zz['연율최대_낙폭'] * 100:.0f}% | "
                 f"{zz['전멸피함_최대w']:.2f} · {pct(zz['전멸피함_연율'])} | {zz['낙폭50_최대w']:.2f} · {pct(zz['낙폭50_연율'])} | {pct(zz['w0.33_연율'])} · {zz['w0.33_낙폭'] * 100:.0f}% |")
    mz = lambda k: float(np.median([zz[k] for zz in size.values()]))
    L.append(f"| 중앙 | {pct(mz('보유_연율'))} | {mz('연율최대_w'):.2f} · {pct(mz('연율최대_연율'))} | {mz('전멸피함_최대w'):.2f} | {mz('낙폭50_최대w'):.2f} | {pct(mz('w0.33_연율'))} |")
    L.append("")

    # 사용자 표
    L.append("## 사용자 표: 규칙 → 창별 수익(상품 9개 중앙값, 수익/최대 낙폭)\n")
    L.append("수수료 0, 미끄러짐 0, 현금 이자 0. 2000~02는 그 구간 합성이 있는 상품만(괄호 = 상품 수). 전멸 = 긴 역사 최대 낙폭 −90% 이하인 상품 수/9.\n")
    hdr = "| 규칙 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 2020 | 2008 | 2000~02 | 긴 역사 연율 | 긴 역사 최대 낙폭 | 전멸 | 연 매매 |"
    L.append(hdr); L.append("|" + "---|" * 12)
    for i in ids:
        a = agg[i]
        cells = []
        for wn in WINDOWS:
            r = a["ret_" + wn]
            s = f"{pct(r['med'])}/{r['med_mdd'] * 100:.0f}%" if r["n"] else "—"
            if wn == "2000~02":
                s += f" ({r['n']})"
            cells.append(s)
        tag = " **대표**" if i == rep else ""
        L.append(f"| {nm(i)}{tag} | " + " | ".join(cells) +
                 f" | {pct(a['med_cagr'])} | {a['med_mdd'] * 100:.0f}% | {a['wipe_rule']}/9 | {a['med_trades']:.1f} |")
    L.append("")

    # 일반화 표
    L.append("## 일반화 표: 효율선초과 > 0 상품 수(/9)와 중앙값\n")
    L.append("긴 역사 = 합성 시작 + 252거래일 ~ 2025-12-31(연율 %p). 같은 노출 = 규칙의 평균 보유 비율과 같은 고정 w 대비. "
             "전멸 상품 수: 규칙 / 보유 / 같은 평균 노출 덜 담기 / 같은 낙폭 덜 담기(정의상 규칙 이하).\n")
    L.append("| 규칙 | 긴 역사 >0 | 긴 역사 중앙 | 같은 노출 대비(긴) | 2026 앞 >0 · 중앙 | 2026 뒤 >0 · 중앙 | 2025 >0 · 중앙 | 전멸 규칙/보유/같은노출/같은낙폭 | 미끄러짐 0.5% 긴 역사 >0 · 중앙 |")
    L.append("|" + "---|" * 9)
    for i in ids:
        a = agg[i]; s = agg_slip[i]
        L.append(f"| {nm(i)} | {a['긴역사']['pos']}/9 | {pp(a['긴역사']['med'])} | {pp(a['긴역사']['med_same_exp'])} ({a['긴역사']['pos_same_exp']}/9) | "
                 f"{a['2026앞']['pos']} · {pp(a['2026앞']['med'])} | {a['2026뒤']['pos']} · {pp(a['2026뒤']['med'])} | "
                 f"{a['2025']['pos']} · {pp(a['2025']['med'])} | {a['wipe_rule']}/{a['wipe_hold']}/{a['wipe_same_exp_F']}/{a['wipe_same_mdd_F']} | "
                 f"{s['긴역사']['pos']} · {pp(s['긴역사']['med'])} |")
    L.append("")

    # 대표 후보 상품별
    for rid, title in ((rep, "대표 후보"),):
        L.append(f"## {title} 상품별: {nm(rid)}\n")
        L.append("| 상품(기초) | 긴 역사 시작 | 규칙 연율/낙폭 | 보유 연율/낙폭 | 같은 낙폭 덜 담기 w · 연율 | 효율선초과 긴·앞·뒤·2025 | 최악 1년 규칙/보유 | 전고점 회복 최장(년) 규칙/보유 | 연 매매 |")
        L.append("|" + "---|" * 9)
        for p in P9:
            e, h = E[p][rid], E[p]["R0"]
            g = e["긴역사"]; gh = h["긴역사"]
            L.append(f"| {p}({prods[p]['desc']}) | {g['start']} | {pct(g['cagr'])}/{g['mdd'] * 100:.0f}% | {pct(gh['cagr'])}/{gh['mdd'] * 100:.0f}% | "
                     f"{g['eff_w']:.2f} · {pct(g['eff_ret'])} | {pp(g['eff'])} · {pp(e['2026앞']['eff'])} · {pp(e['2026뒤']['eff'])} · {pp(e['2025']['eff'])} | "
                     f"{pct(g['worst1y'], 0)}/{pct(gh['worst1y'], 0)} | {g['underwater_y']:.1f}{'(미회복)' if g['underwater_now'] else ''}/"
                     f"{gh['underwater_y']:.1f}{'(미회복)' if gh['underwater_now'] else ''} | {g['trades_y']:.1f} |")
        L.append("")

    # 폭락 구간
    show = list(dict.fromkeys(["R0", "F0.5", "F0.67", rep, G1, G2, G3, "T200", "V1", "S30"]))
    L.append("## 폭락 구간: 상품 중앙 수익(괄호 = 그 구간 합성이 있는 상품 수)\n")
    L.append("| 규칙 | " + " | ".join(CRASH) + " |"); L.append("|" + "---|" * (len(CRASH) + 1))
    for i in show:
        L.append(f"| {nm(i)} | " + " | ".join(f"{pct(agg[i]['crash_' + c]['med'], 0)} ({agg[i]['crash_' + c]['n']})" for c in CRASH) + " |")
    L.append("")

    # 효율선 표
    L.append("## 효율선 표: 창별 효율선초과(상품 중앙)와 그 덜 담기 w(중앙)\n")
    L.append("| 규칙 | 2026 앞 | 2026 뒤 | 2025 | 긴 역사(연율) | 긴 역사 평균 노출(중앙) | 같은 노출 대비 앞·뒤·2025·긴 |")
    L.append("|---|---|---|---|---|---|---|")
    for i in show:
        a = agg[i]
        ws = {wn: float(np.median([E[p][i][wn]["eff_w"] for p in P9])) for wn in EXC_WINDOWS}
        L.append(f"| {nm(i)} | " + " | ".join(f"{pp(a[wn]['med'])} (w {ws[wn]:.2f})" for wn in EXC_WINDOWS) +
                 f" | {a['med_expo']:.2f} | " + " · ".join(pp(a[wn]['med_same_exp']) for wn in EXC_WINDOWS) + " |")
    L.append("")

    # 고르기
    L.append("## 고르기 표 (앞으로 고르고 뒤·2025·긴 역사 확인, 대상 = 덜 담기 3개를 뺀 규칙 22개)\n")
    L.append("| 방법 | 고른 것 | 고른 근거 값 | 2026 뒤 >0 · 중앙 | 2025 >0 · 중앙 | 긴 역사 >0 · 중앙(연) | 전멸 규칙/보유 |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, i, basis in (("대표: 긴 역사 >0 상품 수 최대", rep, f"{agg[rep]['긴역사']['pos']}/9 · {pp(agg[rep]['긴역사']['med'])}"),
                          ("G1: 2026 앞 효율선초과 중앙 최대", G1, f"앞 {agg[G1]['2026앞']['pos']}/9 · {pp(agg[G1]['2026앞']['med'])}"),
                          ("G2: 긴 역사 효율선초과 중앙 최대", G2, f"긴 {agg[G2]['긴역사']['pos']}/9 · {pp(agg[G2]['긴역사']['med'])}"),
                          ("G3: 2026 앞 칼마 중앙 최대", G3, f"칼마 {agg[G3]['med_calmar_front']:.1f} (보유 {agg['R0']['med_calmar_front']:.1f})")):
        a = agg[i]
        L.append(f"| {lab} | {nm(i)} | {basis} | {a['2026뒤']['pos']} · {pp(a['2026뒤']['med'])} | {a['2025']['pos']} · {pp(a['2025']['med'])} | "
                 f"{a['긴역사']['pos']} · {pp(a['긴역사']['med'])} | {a['wipe_rule']}/{a['wipe_hold']} |")
    L.append("")
    L.append("'2026 앞 > 0' 쌍(상품×규칙)이 다른 창에서도 > 0인 비율(괄호 = 고르지 않은 전체 비율):\n")
    for wn, s in succ.items():
        L.append(f"- {wn}: {s['앞>0']['rate'] * 100:.0f}% (n {s['앞>0']['n']}) vs 무조건 {s['무조건']['rate'] * 100:.0f}% (n {s['무조건']['n']})")
    L.append("- 주의: '앞 > 0' 쌍 27개 중 20개가 TQQQ·TECL·UPRO의 추세 규칙이다. 2025·긴 역사 성공률이 높은 것은 규칙보다 '미국 대형 기술주 지수에서는 추세 필터가 통했다'는 상품 묶음 효과다.")
    L.append("")

    # 실제 상품 대조
    L.append("## 실제 상품 가격으로 같은 규칙(기초 신호) — 상장 1년 뒤 ~ 2025, 효율선초과(연 %p) 실제 / 합성\n")
    rid_list = [k for k in next(iter(real.values())) if not k.startswith("_")]
    L.append("| 상품(기간) | " + " | ".join(rid_list) + " | 보유 연율 실제/합성 |"); L.append("|" + "---|" * (len(rid_list) + 2))
    for p, r in real.items():
        L.append(f"| {p}({r['_기간']}) | " + " | ".join(f"{pp(r[i]['실제']['eff'])} / {pp(r[i]['합성']['eff'])}" for i in rid_list) +
                 f" | {pct(r['_R0']['실제']['cagr'])} / {pct(r['_R0']['합성']['cagr'])} |")
    L.append("| 실제 > 0 상품 수 | " + " | ".join(f"{sum(real[p][i]['실제']['eff'] > 1e-9 for p in real)}/{len(real)}" for i in rid_list) + " | |")
    L.append("")
    L.append("- 실제 상품 구간(2009~2025)에서도 변동성 계열(V45·V1)은 TQQQ·UPRO·TECL·FAS(미국 대형·기술·금융)에서만 효율선을 이기고 TNA·YINN·EDC·LABU(소형·중국·신흥·바이오)에서는 졌다. 규칙보다 '어느 상품이냐'가 결과를 갈랐다. 합성과 실제의 차는 대부분 1~2%p 안이다(NUGT는 2배 구간이 있어 뺐다).")
    L.append("")

    # 합성 대조·한계
    L.append("## 합성 대조(추가 비용은 이 값으로 넣음)\n")
    L.append("| 상품 | 기초 | 합성 시작 | 대조 구간 시작·일수 | 일수익 상관 | 추가 비용(연 로그) | 최대 낙폭 실제 / 합성(반영 후) |")
    L.append("|---|---|---|---|---|---|---|")
    for p, d in prods.items():
        v0, v1 = d["val0"], d["val1"]
        L.append(f"| {p} | {d['desc']} | {d['P'].days[0]} | {v0['시작']} · {v0['일수']} | {v0['일수익상관']:.3f} | {d['extra'] * 100:+.2f}% | "
                 f"{v1['실제_낙폭'] * 100:.1f}% / {v1['합성_낙폭'] * 100:.1f}% |")
    L.append("")
    L.append("## 시험 범위·한계\n")
    L.append(f"- 시험 범위: 규칙 {len(cand)}개(R0 1 + T 9 + V 6 + VS 2 + TV 1 + S 3) + 덜 담기 3 + 사후 민감도 3(S 재진입을 'SMA50 아래로 갔다가 위'로, 고르기 제외) = {len(ids)}개 × 상품 9 = {len(ids) * 9}칸, "
             f"창 4개(2026 앞·뒤, 2025, 긴 역사) + 맥락 창 6 + 폭락 9, 효율선 w 101개, 미끄러짐 0·0.5%, 현금 0·4%. 규칙·창·고르기 방법은 결과 전에 고정했다. "
             "고르기 3종 × 22규칙이라 '앞'으로 고른 값에는 선택편향이 있다.")
    L.append(f"- 점검: 미래 가격 교란 뒤 그 전까지 가치 최대 차 {leak:.1e}(0이어야 함). stock_lab.simulate와 같은 규칙 대조 최대 상대차 "
             + ", ".join(f"{p} {', '.join(f'{k} {v:.1e}' for k, v in x.items())}" for p, x in xchk.items())
             + f". 덜 담기 규칙 곡선과 효율선 재료의 같은 w 곡선 최대 상대차 {fchk:.1e}(0이어야 함). 덜 담기 규칙의 효율선초과는 0 이하다(하락 창에서는 더 낮은 w나 현금이 더 낫다).")
    L.append("- 합성 한계: 추가 비용을 상수로 넣었다(실제는 금리·차입 비용에 따라 변함). YINN(FXI)·FAS(XLF)는 실제 상품 지수와 다른 ETF를 기초로 써서 일수익 상관이 낮다. "
             "NUGT는 실제로 2011-11 전·2020-03 뒤 2배였는데 합성은 전 기간 3배다(실제 대조는 3배 구간만). "
             "^NDX(1999-03 전)·^GSPC(1993-02 전)는 시가가 전일 종가라 그 구간의 시가 체결은 사실상 전일 종가 체결이다(약간 유리).")
    L.append("- 생존편향: 9개 모두 지금 살아 있는 3배 상품이다. 기초 지수는 망하지 않았지만 상품 자체가 청산된 3배 ETF(예: 일부 원자재·변동성 상품)는 없다.")
    L.append("- 상품·주문: 미국 상장 3배 ETF, 수수료 0(§1). 장중 스탑은 t−1까지 정보로 정한 가격에 체결된다고 봤고, 프리·애프터장 먼저 닿는 체결과 갭 체결 미끄러짐은 재지 않았다(갭은 시가 체결로 반영).")
    L.append("- 다른 해의 실패는 장세 전환 위험으로만 적는다(연구 보고 규칙). 매수 추천·주문 수량이 아니다.")
    text = "\n".join(L)
    text = text.replace("@@CONCLUSION@@", conclusion(agg, agg_slip, agg_cash, rep, G1, G2, G3, succ, names, cand))
    (RESULTS / "p5_XPROD.md").write_text(text, encoding="utf-8")


def conclusion(agg, agg_slip, agg_cash, rep, G1, G2, G3, succ, names, cand) -> str:
    a, s = agg[rep], agg_slip[rep]
    g2 = agg[G2]
    best = {wn: max(agg[i][wn]["pos"] for i in cand if i != "R0") for wn in ("2026앞", "2026뒤", "2025")}
    n5 = sum(agg[i]["긴역사"]["pos"] >= 5 for i in cand)
    S3 = [agg[f"S{x}"]["wipe_rule"] for x in (30, 40, 50)]
    lines = [
        f"1. **다른 3배 상품 9개에서 '덜 담기'를 꾸준히 이긴 손절·추세·변동성 규칙은 없다.** 규칙 22개 가운데 2026 앞·뒤·2025에 효율선을 이긴 상품 수는 많아야 "
        f"{best['2026앞']}·{best['2026뒤']}·{best['2025']}/9이고, 2026 앞에서 이긴 상품-규칙 쌍이 뒤에서도 이긴 비율은 {succ['2026뒤']['앞>0']['rate'] * 100:.0f}%로 "
        f"고르지 않은 {succ['2026뒤']['무조건']['rate'] * 100:.0f}%와 같다. 긴 역사에서 절반(5/9) 이상 이긴 규칙은 {n5}개뿐이다.",
        f"2. **전멸(최대 낙폭 −90% 이하)은 손절이 아니라 크기로 막혔다.** 긴 역사에서 그냥 보유는 {agg['R0']['wipe_rule']}/9 상품이 전멸, 덜 담기 1/3은 {agg['F0.33']['wipe_rule']}/9, 1/2은 {agg['F0.5']['wipe_rule']}/9. "
        f"넓은 추적 손절(−30·−40·−50% → 기초>SMA50 재진입)은 {S3[0]}·{S3[1]}·{S3[2]}/9로 막지 못했다(약세장 반등마다 다시 사서 또 잘림). 재진입을 'SMA50 아래로 갔다가 다시 위'로 바꾼 사후 민감도도 {agg['S30x']['wipe_rule']}·{agg['S40x']['wipe_rule']}·{agg['S50x']['wipe_rule']}/9.",
        f"3. 긴 역사로 고른 대표 후보 **{names[rep]}**(변동성이 오르면 줄이기)는 {a['긴역사']['pos']}/9 상품·중앙 {pp(a['긴역사']['med'])}/연으로 이겼고 같은 평균 노출의 고정 비율보다는 "
        f"{a['긴역사']['pos_same_exp']}/9에서 나았다(이론과 같은 방향). 그러나 2026 앞·뒤·2025는 {a['2026앞']['pos']}·{a['2026뒤']['pos']}·{a['2025']['pos']}/9, 미끄러짐 0.5%면 {s['긴역사']['pos']}/9·{pp(s['긴역사']['med'])}, "
        f"매일 조정(연 {a['med_trades']:.0f}회)이라 2026 기준으로는 합격하지 못했다(긴 역사 관찰로만 남김). 추세 {names[G2]}는 긴 역사 {g2['긴역사']['pos']}/9·{pp(g2['긴역사']['med'])}(미끄러짐에도 {agg_slip[G2]['긴역사']['pos']}/9)였지만 2026 뒤 {g2['2026뒤']['pos']}/9.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
