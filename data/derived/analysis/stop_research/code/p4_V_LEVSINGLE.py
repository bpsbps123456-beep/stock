"""반박 검증 V_LEVSINGLE(2026-10-11): 4단계 LEVSINGLE 대표 후보 G1 'QQQ가 50일선 아래면 2배 단일 종목 상품 → 기초 1배'.

엔진(trade_lab·lev_lab·stock_lab·tranche_lab·p4_LEVSINGLE)을 import하지 않는다. cache/p4_lev.parquet·p4_meta.json만 읽어 새로 짠다.
  1) 독립 재계산: 자료 손질(배당 수정·자료 오류 의심·SPAC 자르기), 보정·기본 합성, 추가 비용, 신호, 갈아타기, 효율선, 실제 상품.
  2) 시점: 미래 가격 교란 감사, 하루 늦춘 판, 미끄러짐 0.5%, 배당 수정 안 한 QQQ 신호.
  3) 이웃(사전 고정): SMA 길이 · 꺼질 때 배수 · 띠 각각 후보 ±2칸(5×5×5 = 125개, 후보 포함) + 신호원 3개.
  4) 위약: 같은 창 안 '나가 있기' 구간 길이·개수를 그대로 두고 날짜만 무작위로 옮긴 판(모든 상품 같은 날짜, 시드 20).
  5) 장세: 2026 앞·뒤, 2025, 2022, 2020, 2018, 2018 4분기, 2020 2~3월, 2015~2026 이어 돌리기 연율·최대 낙폭·−80% 전멸.
실행: cd code && PYTHONIOENCODING=utf-8 py p4_V_LEVSINGLE.py  → results/p4_V_LEVSINGLE.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않는다)
L = 2
FEE = 0.0105
MIN_HIST = 126
TRACK_DAYS, TRACK_MIN, EXTRA_FALLBACK = 252, 60, 0.10
SPAC_START = {"IONQ": "2021-10-01", "RGTI": "2022-03-02", "OKLO": "2024-05-10"}
NOT_SINGLE = {"QQQ", "SPY", "SOXX"}
K_GRID = np.round(np.arange(0, L + 1e-9, 0.05), 2)          # 41개
DEATH = -0.80
SLIP = 0.005
CAND = {"src": "QQQ", "n": 50, "m": 1.0, "band": 0.0}      # m = 꺼질 때 배수(1 = 기초 1배), band: 종가 < SMA×(1+band)면 꺼짐
NB_N = (30, 40, 50, 60, 70)                                 # SMA 길이 ±2칸(10일 간격)
NB_M = (0.5, 0.75, 1.0, 1.25, 1.5)                          # 꺼질 때 배수 ±2칸(0.25 간격). ≤1: 기초 m + 현금, >1: 상품 (m−1) + 기초 (2−m)
NB_B = (-0.02, -0.01, 0.0, 0.01, 0.02)                      # 띠 ±2칸(1%p 간격)
NB_SRC = ("SPY", "XLK", "SOXX")                             # 신호원만 바꾼 판(나머지 후보 그대로)
EXTRA_RULES = {"T QQQ SMA50(현금)": {"src": "QQQ", "n": 50, "m": 0.0, "band": 0.0},
               "G2 D 자기 SMA200": {"src": "own", "n": 200, "m": 1.0, "band": 0.0},
               "G3 T 자기 SMA100(현금)": {"src": "own", "n": 100, "m": 0.0, "band": 0.0}}
N_SEED = 20
TOL = 0.005                                                 # 재현 허용 차 0.5%p

RUNS = {"2026": ("2026-01-02", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31")}
for _y in range(2024, 2014, -1):
    RUNS[str(_y)] = (f"{_y}-01-01", f"{_y}-12-31")
WIN = {"2026앞": ("2026", "2026-01-02", "2026-05-29"), "2026뒤": ("2026", "2026-06-01", "2026-10-09"),
       "2026": ("2026", "2026-01-02", "2026-10-09")}
for _r, (_a, _b) in RUNS.items():
    if _r != "2026":
        WIN[_r] = (_r, _a, _b)
WIN["2018Q4"] = ("2018", "2018-10-01", "2018-12-31")
WIN["2020급락"] = ("2020", "2020-02-19", "2020-03-31")
FULL = ("2015-01-02", "2026-10-09")
YEAR_WINS = [str(y) for y in range(2015, 2027)]             # 상품-해(2026 = 1/2~10/9)
MAIN_WINS = ("2026앞", "2026뒤", "2025")

# 갈래 주장(results/p4_LEVSINGLE.md 대표 G1, 보정 합성) — 비교용
CLAIM = {
    "효율선초과_평균": {"2026앞": 0.68, "2026뒤": -0.41, "2025": -0.01},
    "효율선초과_중앙": {"2026앞": -0.07, "2026뒤": -0.40, "2025": -0.19},
    "보유대비_평균": {"2026앞": 0.84, "2026뒤": -0.08, "2025": 0.37},
    "보유대비_이긴수": {"2026앞": "18/23", "2026뒤": "5/23", "2025": "15/21"},
    "수익_중앙": {"2026앞": 0.35, "2026뒤": -0.30, "2025": 0.11},
    "낙폭_중앙": {"2026앞": -0.39, "2026뒤": -0.48, "2025": -0.57},
    "보유수익_중앙": {"2026앞": 0.25, "2026뒤": -0.30, "2025": -0.01},
    "보유낙폭_중앙": {"2026앞": -0.50, "2026뒤": -0.56, "2025": -0.69},
    "미끄러짐_효율선초과_평균": {"2026앞": 0.36, "2026뒤": -0.50, "2025": -0.25},
    "실제상품_효율선초과_평균": {"2026앞": 0.099, "2026뒤": -0.406},
    "상품해_전멸비율": 0.107, "보유_상품해_전멸비율": 0.189,
    "2022_전멸": "3/16", "보유_2022_전멸": "11/16",
    "장기_연율_중앙": 0.327, "장기_낙폭_중앙": -0.84, "보유_장기_연율_중앙": 0.321, "보유_장기_낙폭_중앙": -0.93,
    "1배_장기_연율_중앙": 0.264, "1배_장기_낙폭_중앙": -0.66,
    "기본합성_2025_효율선초과_평균": 0.09,
    "SNDK_보유_2026앞": 28.79, "SNDK_효율선초과_2026앞": 13.36,
    "T QQQ SMA50(현금)_2026뒤": -0.50,
    "G2 D 자기 SMA200": {"2026앞": -0.17, "2026뒤": -0.27, "2025": -0.13},
    "G3 T 자기 SMA100(현금)": {"2026앞": -0.24, "2026뒤": -0.35, "2025": -0.44},
    "주기": {"2026앞": (92, 0.52, 0.031), "2026뒤": (184, 0.33, -0.012), "2025": (147, 0.37, 0.030)},
}


# ============================================================================ 자료
def ss(days, d, side="left"):
    return int(np.searchsorted(days, d, side=side))


def load_raw():
    df = pd.read_parquet(CACHE / "p4_lev.parquet")
    irx = df[df.sym == "^IRX"].set_index("date")["Close"].astype(float).sort_index()
    irx = irx[~irx.index.duplicated()] / 100.0
    bars, raw_close = {}, {}
    for sym, g in df[df.sym != "^IRX"].sort_values(["sym", "date"]).groupby("sym"):
        g = g.dropna(subset=["Open", "High", "Low", "Close"])
        g = g[(g.Close > 0) & (g.Open > 0) & (g.Low > 0)]
        if len(g) < 30:
            continue
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        bars[sym] = {"days": g.date.to_numpy().astype(str), "o": g.Open.to_numpy() * f, "h": g.High.to_numpy() * f,
                     "l": g.Low.to_numpy() * f, "c": g.Close.to_numpy() * f}
        raw_close[sym] = g.Close.to_numpy().astype(float)
    meta = json.loads((CACHE / "p4_meta.json").read_text(encoding="utf-8"))["lev"]
    return bars, irx, meta, raw_close


def suspicious(c):
    r = c[1:] / c[:-1] - 1
    return bool(np.any(r > 1.5) or np.any(r < -0.7))


def synth(u, rf, fee):
    """일일 재설정 2배: 종가 수익 = 2×기초 − 금리 − 비용, 시가 = 전일 합성 종가 × (1 + 2×기초 시가 변화)."""
    cu, ou = u["c"], u["o"]
    r = np.zeros(len(cu))
    r[1:] = L * (cu[1:] / cu[:-1] - 1) - (L - 1) * rf[1:] / 252 - fee / 252
    g = np.maximum(1 + r, 0.01)
    g[0] = 1.0
    c = 100.0 * np.cumprod(g)
    o = np.empty_like(c)
    o[0] = 100.0
    o[1:] = c[:-1] * np.maximum(1 + L * (ou[1:] / cu[:-1] - 1), 0.01)
    return {"days": u["days"], "o": o, "c": c}


def track_cost(u_sym, bars, meta, rf):
    s = synth(bars[u_sym], rf, FEE)
    out = []
    for p, info in meta.items():
        if info["under"] != u_sym or info["L"] != L or p not in bars:
            continue
        a = bars[p]
        com = np.intersect1d(a["days"], s["days"])
        if len(com) < TRACK_MIN:
            out.append((p, None, len(com)))
            continue
        com = com[-(TRACK_DAYS + 1):]
        ia, isy = np.searchsorted(a["days"], com), np.searchsorted(s["days"], com)
        yrs = (len(com) - 1) / 252
        drag = -(np.log(a["c"][ia[-1]] / a["c"][ia[0]]) - np.log(s["c"][isy[-1]] / s["c"][isy[0]])) / yrs
        out.append((p, float(drag), len(np.intersect1d(a["days"], s["days"]))))
    vals = [d for _, d, _ in out if d is not None]
    return (max(0.0, float(np.median(vals))) if vals else EXTRA_FALLBACK), out


def cut(b, i0):
    return {k: (v[i0:] if isinstance(v, np.ndarray) else v) for k, v in b.items()}


def sma(c, n):
    return pd.Series(c).rolling(n, min_periods=n).mean().to_numpy()


def build():
    bars, irx, meta, raw_close = load_raw()
    unders = sorted({v["under"] for v in meta.values() if v["L"] == L} - NOT_SINGLE)
    excluded = [u for u in unders if suspicious(bars[u]["c"])]
    U = {}
    for u in unders:
        if u in excluded:
            continue
        full = bars[u]
        rf_full = irx.reindex(full["days"]).ffill().fillna(0.0).to_numpy()
        extra, track = track_cost(u, bars, meta, rf_full)
        i0 = ss(full["days"], SPAC_START[u]) if u in SPAC_START else 0
        ub = cut(full, i0)
        U[u] = {"und": ub, "rf": rf_full[i0:], "extra": extra, "track": track,
                "cal": cut(synth(full, rf_full, FEE + extra), i0), "basic": cut(synth(full, rf_full, FEE), i0)}
    return U, excluded, bars, meta, raw_close


_SIG_CACHE = {}


def off_signal(U, bars, u, src, n, band, delay=1, raw_close=None):
    """state[t] = True면 t일 시가부터 그날 '꺼짐'. 판단은 t−delay일 종가(기본 delay=1)."""
    key = (u, src, n, band, delay, raw_close is not None)
    if key in _SIG_CACHE:
        return _SIG_CACHE[key]
    ub = U[u]["und"]
    if src == "own":
        c = ub["c"]
        s = sma(c, n)
        off = np.isfinite(s) & (c < s * (1 + band))
    else:
        q = bars[src]
        qc = raw_close[src] if raw_close is not None else q["c"]
        s = sma(qc, n)
        o = np.isfinite(s) & (qc < s * (1 + band))
        qi = np.searchsorted(q["days"], ub["days"])
        assert np.all(q["days"][qi] == ub["days"]), (u, src)
        off = o[qi]
    st = np.zeros(len(off), bool)
    st[delay:] = off[:-delay]
    _SIG_CACHE[key] = st
    return st


# ============================================================================ 엔진(갈아타기)
def run_switch(p, u, a, z, state, m=1.0, slip=0.0, want_cycles=False):
    """a−1 종가에 상품 100%(가치 1). state[t]가 켜지면 t 시가에 '꺼짐 묶음'(m배)으로, 꺼지면 t 시가에 상품 100%로.
    m ≤ 1: 기초 m + 현금 1−m. m > 1: 상품 (m−1) + 기초 (2−m). 묶음은 바꾼 날 비중으로 산 뒤 그대로 둔다."""
    po, pc, uo, uc = p["o"], p["c"], u["o"], u["c"]
    sh_p, sh_u, cash, off = 1.0 / pc[a - 1], 0.0, 0.0, False
    wp = max(m - 1.0, 0.0)
    wu = m if m <= 1 else 2.0 - m
    vals = np.empty(z - a + 2)
    vals[0] = 1.0
    mult = np.empty(z - a + 1)
    cycles, cur = [], None
    for t in range(a, z + 1):
        s = bool(state[t])
        if s and not off:
            sell_sh = sh_p * (1 - wp)
            proceeds = sell_sh * po[t] * (1 - slip)
            frac_u = wu / (1 - wp) if (1 - wp) > 0 else 0.0
            sh_u = proceeds * frac_u / (uo[t] * (1 + slip))
            cash = proceeds * (1 - frac_u)
            if want_cycles:
                cur = {"exit_t": t, "sh_p0": sh_p}
            sh_p = sh_p - sell_sh
            off = True
        elif off and not s:
            proceeds = sh_u * uo[t] * (1 - slip) + cash
            sh_p = sh_p + proceeds / (po[t] * (1 + slip))
            sh_u, cash, off = 0.0, 0.0, False
            if want_cycles:
                cur.update(re_t=t, gain=sh_p / cur["sh_p0"] - 1)
                cycles.append(cur)
                cur = None
        vals[t - a + 1] = sh_p * pc[t] + sh_u * uc[t] + cash
        mult[t - a] = (L * (sh_p * pc[t]) + 1.0 * sh_u * uc[t]) / vals[t - a + 1]
    if want_cycles and cur is not None:
        cur.update(re_t=None, sh_u=sh_u, cash=cash, sh_p_left=sh_p)
        cycles.append(cur)
    return vals, mult, cycles


def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def kpath(ru, rfd, k, cost, rp=None):
    if k <= 1:
        return k * ru
    if rp is not None:
        return (k - 1) * rp + (2 - k) * ru
    return np.maximum(k * ru - (k - 1) * rfd - cost * (k - 1) / 252, -0.99)


_FR = {}


def frontier(key, ru, rfd, cost, rp=None):
    if key in _FR:
        return _FR[key]
    out = np.empty((len(K_GRID), 2))
    for j, k in enumerate(K_GRID):
        cur = np.concatenate([[1.0], np.cumprod(1 + kpath(ru, rfd, k, cost, rp))])
        out[j] = (cur[-1] - 1, mdd(cur))
    _FR[key] = out
    return out


def excess(ret, dd, fr):
    ok = fr[:, 1] >= dd - 1e-12
    j = int(np.argmax(np.where(ok, fr[:, 0], -np.inf)))
    return float(ret - fr[j, 0]), float(K_GRID[j])


def win_slice(days, a, z, w0, w1):
    return max(ss(days, w0), a), min(ss(days, w1, "right") - 1, z)


# ============================================================================ 평가
def evaluate(U, bars, rule, var="cal", slip=0.0, wins=MAIN_WINS, delay=1, state_fn=None, raw_close=None, cycles=False):
    """rule = {'src','n','m','band'} 또는 'hold'. 반환 행 목록(상품 × 창)."""
    rows = []
    need_runs = sorted({WIN[w][0] for w in wins})
    for u, D in U.items():
        ub, p = D["und"], D[var]
        cost = FEE + (D["extra"] if var == "cal" else 0.0)
        days = ub["days"]
        for rn in need_runs:
            d0, d1 = RUNS[rn]
            a, z = ss(days, d0), ss(days, d1, "right") - 1
            if a < MIN_HIST or z <= a:
                continue
            if rule == "hold":
                v, mult, cyc = p["c"][a - 1:z + 1] / p["c"][a - 1], np.full(z - a + 1, float(L)), []
            else:
                if state_fn is not None:
                    st = state_fn(u, days)
                else:
                    st = off_signal(U, bars, u, rule["src"], rule["n"], rule["band"], delay, raw_close)
                v, mult, cyc = run_switch(p, ub, a, z, st, rule["m"], slip, want_cycles=cycles)
            for w in wins:
                wr, w0, w1 = WIN[w]
                if wr != rn:
                    continue
                i0, i1 = win_slice(days, a, z, w0, w1)
                if i1 <= i0:
                    continue
                ru = ub["c"][i0:i1 + 1] / ub["c"][i0 - 1:i1] - 1
                rfd = D["rf"][i0:i1 + 1] / 252
                fr = frontier((u, var, w), ru, rfd, cost)
                vv = v[i0 - a:i1 - a + 2]
                vv = vv / vv[0]
                ret, dd = float(vv[-1] - 1), mdd(vv)
                exc, keq = excess(ret, dd, fr)
                hv = p["c"][i0 - 1:i1 + 1] / p["c"][i0 - 1]
                hexc, _ = excess(float(hv[-1] - 1), mdd(hv), fr)
                kbar = float(mult[i0 - a:i1 - a + 1].mean())
                rk = np.concatenate([[1.0], np.cumprod(1 + kpath(ru, rfd, kbar, cost))])
                row = {"under": u, "win": w, "ret": ret, "mdd": dd, "exc": exc, "k_eq": keq, "hold_ret": float(hv[-1] - 1),
                       "hold_mdd": mdd(hv), "hold_exc": hexc, "kbar": kbar, "exp_exc": float(ret - (rk[-1] - 1)),
                       "u_ret": float(ub["c"][i1] / ub["c"][i0 - 1] - 1), "u_mdd": mdd(ub["c"][i0 - 1:i1 + 1] / ub["c"][i0 - 1])}
                if cycles:
                    gains = []
                    for cy in cyc:
                        if not (i0 <= cy["exit_t"] <= i1):
                            continue
                        if cy.get("re_t") is not None and cy["re_t"] <= i1:
                            gains.append(cy["gain"])
                        else:   # 창 끝까지 못 되샀으면 창 끝 종가로 값을 매김(미래 정보 차단)
                            # 판 뒤 기초로 들고 있던 몫을 창 끝 종가 상품 주식으로 환산
                            sh_u_at = None
                            # 다시 계산: exit_t 시가의 교환 비율
                            t = cy["exit_t"]
                            sh_u_at = cy["sh_p0"] * p["o"][t] * (1 - slip) / (ub["o"][t] * (1 + slip))
                            gains.append(sh_u_at * ub["c"][i1] / p["c"][i1] / cy["sh_p0"] - 1)
                    row["gains"] = gains
                rows.append(row)
    return pd.DataFrame(rows)


def full_run(U, rule, var="cal"):
    rows = []
    for u, D in U.items():
        ub, p = D["und"], D[var]
        days = ub["days"]
        a = max(ss(days, FULL[0]), MIN_HIST)
        z = ss(days, FULL[1], "right") - 1
        yrs = (z - a + 1) / 252
        if rule == "hold":
            v = p["c"][a - 1:z + 1] / p["c"][a - 1]
        elif rule == "1x":
            v = ub["c"][a - 1:z + 1] / ub["c"][a - 1]
        else:
            st = off_signal(U, BARS, u, rule["src"], rule["n"], rule["band"])
            v, _, _ = run_switch(p, ub, a, z, st, rule["m"])
        rows.append({"under": u, "start": str(days[a]), "cagr": float(max(v[-1], 1e-9) ** (1 / yrs) - 1), "mdd": mdd(v),
                     "ret": float(v[-1] - 1)})
    return pd.DataFrame(rows)


def summ(df, hold=None):
    out = {}
    for w, g in df.groupby("win"):
        r = {"n": int(len(g)), "exc_mean": float(g.exc.mean()), "exc_med": float(g.exc.median()),
             "ret_med": float(g.ret.median()), "mdd_med": float(g.mdd.median()), "hold_ret_med": float(g.hold_ret.median()),
             "hold_mdd_med": float(g.hold_mdd.median()), "vs_hold_mean": float((g.exc - g.hold_exc).mean()),
             "vs_hold_win": int(((g.exc - g.hold_exc) > 1e-12).sum()), "exp_exc_mean": float(g.exp_exc.mean()),
             "kbar_mean": float(g.kbar.mean()), "dead": int((g.mdd <= DEATH).sum()), "hold_dead": int((g.hold_mdd <= DEATH).sum()),
             "u_dead": int((g.u_mdd <= DEATH).sum()), "exc_pos": int((g.exc > 1e-12).sum()),
             "u_ret_med": float(g.u_ret.median()), "u_mdd_med": float(g.u_mdd.median()),
             "hold_exc_mean": float(g.hold_exc.mean()),
             "exc_mean_exSNDK": float(g[g.under != "SNDK"].exc.mean())}
        if "gains" in g:
            x = np.array([y for gg in g.gains for y in gg])
            if len(x):
                win, loss = x[x > 0], x[x <= 0]
                r.update({"cyc_n": int(len(x)), "cyc_win": float(len(win) / len(x)), "cyc_exp": float(x.mean()),
                          "cyc_pf": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else None})
        out[w] = r
    return out


# ============================================================================ 실제 상품(2026)
def actual(U, bars, meta, rule, slip=0.0, delay=1):
    rows = []
    for u, D in U.items():
        cands = sorted((bars[p]["days"][0], p) for p, i in meta.items()
                       if i["under"] == u and i["L"] == L and p in bars and bars[p]["days"][0] < "2025-12-31")
        if not cands:
            continue
        pn = cands[0][1]
        pb = bars[pn]
        ub = D["und"]
        ui = np.searchsorted(ub["days"], pb["days"])
        ok = ui < len(ub["days"])
        ok[ok] = ub["days"][ui[ok]] == pb["days"][ok]
        keep = np.flatnonzero(ok)
        p = {k: pb[k][keep] for k in ("days", "o", "c")}
        ui = ui[keep]
        u2 = {k: ub[k][ui] for k in ("days", "o", "c")}
        st_full = off_signal(U, bars, u, rule["src"], rule["n"], rule["band"], 1)
        # 상품 날짜 기준 상태: 전날(상품 날짜) 신호. delay=1이면 갈래 엔진과 같다(off[ui][t−1]).
        off_raw = np.zeros(len(st_full), bool)
        off_raw[:-1] = st_full[1:]                                # off_raw[i] = i일 종가 신호
        o2 = off_raw[ui]
        st = np.zeros(len(o2), bool)
        st[delay:] = o2[:-delay]
        a, z = ss(p["days"], RUNS["2026"][0]), ss(p["days"], RUNS["2026"][1], "right") - 1
        if a < 1 or z <= a:
            continue
        v, _, _ = run_switch(p, u2, a, z, st, rule["m"], slip)
        for w in ("2026앞", "2026뒤"):
            _, w0, w1 = WIN[w]
            i0, i1 = win_slice(p["days"], a, z, w0, w1)
            ru = u2["c"][i0:i1 + 1] / u2["c"][i0 - 1:i1] - 1
            rp = p["c"][i0:i1 + 1] / p["c"][i0 - 1:i1] - 1
            fr = frontier((pn, "actual", w), ru, None, 0.0, rp)
            vv = v[i0 - a:i1 - a + 2]
            vv = vv / vv[0]
            exc, _ = excess(float(vv[-1] - 1), mdd(vv), fr)
            hv = p["c"][i0 - 1:i1 + 1] / p["c"][i0 - 1]
            hexc, _ = excess(float(hv[-1] - 1), mdd(hv), fr)
            rows.append({"product": pn, "under": u, "win": w, "ret": float(vv[-1] - 1), "mdd": mdd(vv), "exc": exc,
                         "hold_ret": float(hv[-1] - 1), "hold_exc": hexc})
    return pd.DataFrame(rows)


# ============================================================================ 시점 감사(미래 교란)
def audit(U, bars, n_case=6):
    """기초·QQQ의 T일 이후 가격을 흔들고 합성·신호·가치를 다시 만든다. T일까지 가치·상태가 같아야 한다."""
    rng = np.random.default_rng(11)
    worst_v, worst_s, n = 0.0, 0, 0
    us = list(U)[:n_case]
    for u in us:
        D = U[u]
        days = D["und"]["days"]
        a, z = ss(days, "2026-01-02"), ss(days, "2026-10-09", "right") - 1
        for T in (a + 30, a + 95, a + 150):
            ub2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in D["und"].items()}
            for k in ("o", "h", "l", "c"):
                ub2[k][T + 1:] *= 1 + rng.normal(0, 0.06, len(ub2[k]) - T - 1)
            q = bars["QQQ"]
            q2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in q.items()}
            qT = ss(q["days"], days[T], "right")
            q2["c"][qT:] *= 1 + rng.normal(0, 0.04, len(q2["c"]) - qT)
            p2 = synth(ub2, D["rf"], FEE + D["extra"])
            # 신호 다시
            s1 = sma(q["c"], 50); o1 = np.isfinite(s1) & (q["c"] < s1)
            s2 = sma(q2["c"], 50); o2 = np.isfinite(s2) & (q2["c"] < s2)
            qi = np.searchsorted(q["days"], days)
            st1 = np.zeros(len(days), bool); st1[1:] = o1[qi][:-1]
            st2 = np.zeros(len(days), bool); st2[1:] = o2[qi][:-1]
            v1, _, _ = run_switch(D["cal"], D["und"], a, z, st1)
            # 원래 합성은 SPAC 자르기 전부터 만들어 수준이 다르다 → 수익률로 비교
            v2, _, _ = run_switch(p2, ub2, a, z, st2)
            r1 = v1[1:T - a + 2] / v1[:T - a + 1]
            r2 = v2[1:T - a + 2] / v2[:T - a + 1]
            worst_v = max(worst_v, float(np.abs(r1 - r2).max()))
            worst_s += int((st1[:T + 1] != st2[:T + 1]).sum())
            n += 1
    return {"cases": n, "T까지_일수익_최대차": worst_v, "T까지_상태_다른날": worst_s}


# ============================================================================ 위약
def runs_of(x):
    """불리언 배열의 True 구간 길이 목록."""
    out, cur = [], 0
    for b in x:
        if b:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def place(n, lengths, rng):
    m = len(lengths)
    st = np.zeros(n, bool)
    if m == 0:
        return st
    lens = list(rng.permutation(lengths))
    free = n - sum(lens) - (m - 1)
    if free < 0:
        return None
    cuts = np.sort(rng.choice(free + m, m, replace=False))
    gaps = np.diff(np.concatenate([[-1], cuts])) - 1          # m개 비음수 간격(앞쪽), 나머지는 끝
    pos = 0
    for i, ln in enumerate(lens):
        pos += gaps[i] + (1 if i > 0 else 0)
        st[pos:pos + ln] = True
        pos += ln
    return st


def placebo_states(bars, rng):
    """실제 QQQ SMA50 상태(t일 시가부터 꺼짐)를 날짜 기준으로 만들고, 창(2026앞·2026뒤·2025)마다 꺼짐 구간 길이·개수를 그대로 둔 채
    날짜만 무작위로 옮긴다. 모든 상품에 같은 날짜."""
    q = bars["QQQ"]
    s = sma(q["c"], CAND["n"])
    o = np.isfinite(s) & (q["c"] < s)
    st = np.zeros(len(o), bool)
    st[1:] = o[:-1]
    days = q["days"]
    new = st.copy()
    info = {}
    for w in MAIN_WINS:
        _, w0, w1 = WIN[w]
        i0, i1 = ss(days, w0), ss(days, w1, "right") - 1
        seg = st[i0:i1 + 1]
        lens = runs_of(seg)
        ps = place(len(seg), lens, rng)
        new[i0:i1 + 1] = ps
        info[w] = {"days": int(len(seg)), "off_days": int(seg.sum()), "episodes": len(lens)}
    return pd.Series(new, index=days), info


# ============================================================================ 실행
BARS = None


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def main():
    global BARS
    t0 = time.time()
    U, excluded, bars, meta, raw_close = build()
    BARS = bars
    out = {"config": {"CAND": CAND, "NB_N": NB_N, "NB_M": NB_M, "NB_B": NB_B, "NB_SRC": NB_SRC, "K_GRID_n": len(K_GRID),
                      "SLIP": SLIP, "N_SEED": N_SEED, "excluded": excluded, "unders": list(U)},
           "extra": {u: {"extra": D["extra"], "track": D["track"], "start": str(D["und"]["days"][0])} for u, D in U.items()}}
    print(f"기초 {len(U)}개, 제외 {excluded}")

    # ---- 1) 재현
    rep = {}
    for var in ("cal", "basic"):
        df = evaluate(U, bars, CAND, var=var, wins=tuple(WIN), cycles=True)
        rep[var] = {"rule": summ(df), "rows": df.drop(columns=["gains"]).to_dict("records")}
        hd = evaluate(U, bars, "hold", var=var, wins=tuple(WIN))
        rep[var]["hold"] = summ(hd)
        rep[var]["hold_rows"] = hd.to_dict("records")
    out["reproduce"] = {k: {"rule": v["rule"], "hold": v["hold"]} for k, v in rep.items()}
    cal_rows = pd.DataFrame(rep["cal"]["rows"])
    hold_rows = pd.DataFrame(rep["cal"]["hold_rows"])
    # 갈래 결과 파일(results/p4_LEVSINGLE.json)과 행 단위 대조(엔진 import 없이 숫자만 읽음)
    try:
        orig = pd.DataFrame(json.loads((RESULTS / "p4_LEVSINGLE.json").read_text(encoding="utf-8"))["rows"])
        cmp_ = {}
        for var, rn in (("cal", "D 1배로 [QQQ SMA50]"), ("cal", "보유(2배)"), ("basic", "D 1배로 [QQQ SMA50]")):
            o = orig[(orig["var"] == var) & (orig.rule == rn) & (orig.win != "전체")].set_index(["under", "win"])
            mine = pd.DataFrame(rep[var]["rows" if rn != "보유(2배)" else "hold_rows"]).set_index(["under", "win"])
            j = o.join(mine, rsuffix="_v", how="inner")
            cmp_[f"{var}|{rn}"] = {"n": int(len(j)), "n_orig": int(len(o)), "n_mine": int(len(mine)),
                                   **{f"max|{c}|": float((j[c] - j[c + "_v"]).abs().max()) for c in ("ret", "mdd", "exc", "kbar")}}
        out["row_compare"] = cmp_
    except Exception as e:      # noqa: BLE001
        out["row_compare"] = {"error": repr(e)}
    # 상품-해 전멸
    yr = cal_rows[cal_rows.win.isin(YEAR_WINS)]
    hy = hold_rows[hold_rows.win.isin(YEAR_WINS)]
    out["product_year"] = {"n": int(len(yr)), "dead": int((yr.mdd <= DEATH).sum()), "hold_dead": int((hy.mdd <= DEATH).sum()),
                           "u_dead": int((yr.u_mdd <= DEATH).sum())}
    # 장기
    fr_rule, fr_hold, fr_1x = full_run(U, CAND), full_run(U, "hold"), full_run(U, "1x")
    long14 = fr_rule.start <= FULL[0]
    out["long"] = {"n14": int(long14.sum()),
                   "rule": {"cagr_med": float(fr_rule[long14].cagr.median()), "mdd_med": float(fr_rule[long14].mdd.median()),
                            "dead14": int((fr_rule[long14].mdd <= DEATH).sum()), "dead_all": int((fr_rule.mdd <= DEATH).sum())},
                   "hold": {"cagr_med": float(fr_hold[long14].cagr.median()), "mdd_med": float(fr_hold[long14].mdd.median()),
                            "dead14": int((fr_hold[long14].mdd <= DEATH).sum()), "dead_all": int((fr_hold.mdd <= DEATH).sum())},
                   "1x": {"cagr_med": float(fr_1x[long14].cagr.median()), "mdd_med": float(fr_1x[long14].mdd.median()),
                          "dead14": int((fr_1x[long14].mdd <= DEATH).sum()), "dead_all": int((fr_1x.mdd <= DEATH).sum())},
                   "rows": pd.concat([fr_rule.assign(rule="G1"), fr_hold.assign(rule="hold"), fr_1x.assign(rule="1x")]).to_dict("records")}
    # SNDK
    sn = cal_rows[(cal_rows.under == "SNDK") & (cal_rows.win == "2026앞")].iloc[0]
    front = cal_rows[cal_rows.win == "2026앞"]
    out["sndk"] = {"hold_ret": float(sn.hold_ret), "exc": float(sn.exc), "rule_ret": float(sn.ret),
                   "exc_mean_exSNDK": float(front[front.under != "SNDK"].exc.mean()),
                   "exc_trim1": float(np.sort(front.exc.to_numpy())[1:-1].mean()),
                   "vs_hold_mean_exSNDK": float((front[front.under != "SNDK"].exc - front[front.under != "SNDK"].hold_exc).mean())}
    # 미끄러짐
    sl = evaluate(U, bars, CAND, slip=SLIP)
    out["slip"] = summ(sl)
    # 실제 상품
    act = actual(U, bars, meta, CAND)
    out["actual"] = {w: {"n": int(len(g)), "exc_mean": float(g.exc.mean()), "exc_med": float(g.exc.median()),
                         "vs_hold_win": int(((g.exc - g.hold_exc) > 1e-12).sum()), "hold_exc_mean": float(g.hold_exc.mean()),
                         "ret_med": float(g.ret.median()), "hold_ret_med": float(g.hold_ret.median())}
                     for w, g in act.groupby("win")}
    out["actual_rows"] = act.to_dict("records")
    act_s = actual(U, bars, meta, CAND, slip=SLIP)
    out["actual_slip"] = {w: float(g.exc.mean()) for w, g in act_s.groupby("win")}
    act_d = actual(U, bars, meta, CAND, delay=2)
    out["actual_delay"] = {w: float(g.exc.mean()) for w, g in act_d.groupby("win")}
    # 다른 규칙 교차 확인
    out["other_rules"] = {}
    for name, r in EXTRA_RULES.items():
        out["other_rules"][name] = summ(evaluate(U, bars, r))
    print(f"재현 끝 {time.time() - t0:.0f}s")

    # ---- 2) 시점
    out["audit"] = audit(U, bars)
    dl = evaluate(U, bars, CAND, delay=2)
    out["delay"] = summ(dl)
    base = cal_rows[cal_rows.win.isin(MAIN_WINS)].set_index(["under", "win"])
    dj = dl.set_index(["under", "win"]).join(base[["exc", "ret"]], rsuffix="_base")
    out["delay_rows"] = dj.reset_index()[["under", "win", "exc", "exc_base", "ret", "ret_base"]].to_dict("records")
    out["delay_slip"] = summ(evaluate(U, bars, CAND, delay=2, slip=SLIP))
    out["raw_qqq"] = summ(evaluate(U, bars, CAND, raw_close=raw_close))
    q = bars["QQQ"]
    s_adj = sma(q["c"], 50); s_raw = sma(raw_close["QQQ"], 50)
    i26 = ss(q["days"], "2025-01-02")
    d_adj = (q["c"] < s_adj)[i26:]
    d_raw = (raw_close["QQQ"] < s_raw)[i26:]
    out["raw_qqq_diff_days_2025_2026"] = int((d_adj != d_raw).sum())
    # QQQ 넘나듦 수
    st = np.zeros(len(q["c"]), bool); st[1:] = (np.isfinite(s_adj) & (q["c"] < s_adj))[:-1]
    cross = {}
    for w in MAIN_WINS:
        _, w0, w1 = WIN[w]
        i0, i1 = ss(q["days"], w0), ss(q["days"], w1, "right") - 1
        seg = st[i0 - 1:i1 + 1]
        cross[w] = {"switches": int((seg[1:] != seg[:-1]).sum()), "exits": int(((~seg[:-1]) & seg[1:]).sum()),
                    "off_days": int(st[i0:i1 + 1].sum()), "days": int(i1 - i0 + 1)}
    out["qqq_switches"] = cross
    print(f"시점 끝 {time.time() - t0:.0f}s")

    # ---- 3) 이웃
    nb = []
    for n in NB_N:
        for m in NB_M:
            for b in NB_B:
                r = {"src": "QQQ", "n": n, "m": m, "band": b}
                s = summ(evaluate(U, bars, r))
                nb.append({"src": "QQQ", "n": n, "m": m, "band": b, **{f"{k}_{w}": s[w][k] for w in MAIN_WINS
                                                                         for k in ("exc_mean", "exc_med", "vs_hold_mean", "exc_mean_exSNDK")}})
    for src in NB_SRC:
        r = {"src": src, "n": 50, "m": 1.0, "band": 0.0}
        s = summ(evaluate(U, bars, r))
        nb.append({"src": src, "n": 50, "m": 1.0, "band": 0.0, **{f"{k}_{w}": s[w][k] for w in MAIN_WINS
                                                                    for k in ("exc_mean", "exc_med", "vs_hold_mean", "exc_mean_exSNDK")}})
    nb = pd.DataFrame(nb)
    out["neighbors"] = nb.to_dict("records")
    allpos = {}
    for k in ("exc_mean", "exc_med", "vs_hold_mean", "exc_mean_exSNDK"):
        cols = [f"{k}_{w}" for w in MAIN_WINS]
        allpos[k] = {"모두+": int((nb[cols] > 0).all(axis=1).sum()), "앞+": int((nb[cols[0]] > 0).sum()),
                     "앞+중뒤+": int(((nb[cols[0]] > 0) & (nb[cols[1]] > 0)).sum()),
                     "앞+중2025+": int(((nb[cols[0]] > 0) & (nb[cols[2]] > 0)).sum()),
                     "뒤+": int((nb[cols[1]] > 0).sum()), "2025+": int((nb[cols[2]] > 0).sum()), "n": int(len(nb))}
    out["neighbors_allpos"] = allpos
    print(f"이웃 끝 {time.time() - t0:.0f}s")

    # ---- 4) 위약
    rng = np.random.default_rng(20261011)
    pl = []
    for seed in range(N_SEED):
        ser, info = placebo_states(bars, rng)

        def fn(u, days, ser=ser):
            return ser.reindex(days).fillna(False).to_numpy(bool)
        s = summ(evaluate(U, bars, {"src": "QQQ", "n": 50, "m": 1.0, "band": 0.0}, state_fn=fn))
        pl.append({"seed": seed, **{f"{k}_{w}": s[w][k] for w in MAIN_WINS for k in ("exc_mean", "exc_med", "vs_hold_mean", "exp_exc_mean", "exc_mean_exSNDK")}})
    pl = pd.DataFrame(pl)
    real = out["reproduce"]["cal"]["rule"]
    rank = {}
    for w in MAIN_WINS:
        for k in ("exc_mean", "exc_med", "vs_hold_mean", "exp_exc_mean", "exc_mean_exSNDK"):
            x = pl[f"{k}_{w}"].to_numpy()
            rv = real[w][k]
            rank[f"{k}_{w}"] = {"real": rv, "placebo_med": float(np.median(x)), "p05": float(np.quantile(x, 0.05)),
                                "p95": float(np.quantile(x, 0.95)), "real_beats": int((rv > x).sum()), "n": int(len(x))}
    out["placebo"] = {"rank": rank, "info": info, "rows": pl.to_dict("records")}
    print(f"위약 끝 {time.time() - t0:.0f}s")

    (RESULTS / "p4_V_LEVSINGLE.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("저장", RESULTS / "p4_V_LEVSINGLE.json")

    # ---- 화면 요약
    R = out["reproduce"]["cal"]["rule"]
    H = out["reproduce"]["cal"]["hold"]
    for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018", "2018Q4", "2020급락", "2026"):
        r = R[w]
        print(f"{w:8s} n{r['n']:2d} 효율 평균 {pct(r['exc_mean'])} 중앙 {pct(r['exc_med'])} | 보유대비 {pct(r['vs_hold_mean'])} "
              f"({r['vs_hold_win']}/{r['n']}) | 수익 중앙 {pct(r['ret_med'])} 낙폭 {pct(r['mdd_med'])} | 보유 {pct(r['hold_ret_med'])} "
              f"{pct(r['hold_mdd_med'])} | 1배 {pct(r['u_ret_med'])} {pct(r['u_mdd_med'])} | 전멸 {r['dead']}·보유 {r['hold_dead']}·1배 {r['u_dead']} "
              f"| 주기 {r.get('cyc_n')} 승률 {r.get('cyc_win')} 기대 {r.get('cyc_exp')} | 노출대비 {pct(r['exp_exc_mean'])} 평균배수 {r['kbar_mean']:.2f}")
    print("기본 합성 2025", pct(out["reproduce"]["basic"]["rule"]["2025"]["exc_mean"]), "앞", pct(out["reproduce"]["basic"]["rule"]["2026앞"]["exc_mean"]),
          "뒤", pct(out["reproduce"]["basic"]["rule"]["2026뒤"]["exc_mean"]))
    print("미끄러짐", {w: pct(out["slip"][w]["exc_mean"]) for w in MAIN_WINS})
    print("하루 늦춤", {w: (pct(out["delay"][w]["exc_mean"]), pct(out["delay"][w]["vs_hold_mean"])) for w in MAIN_WINS})
    print("늦춤+미끄러짐", {w: pct(out["delay_slip"][w]["exc_mean"]) for w in MAIN_WINS})
    print("배당 수정 안 한 QQQ", {w: pct(out["raw_qqq"][w]["exc_mean"]) for w in MAIN_WINS}, "다른 날", out["raw_qqq_diff_days_2025_2026"])
    print("실제", out["actual"], "미끄러짐", out["actual_slip"], "늦춤", out["actual_delay"])
    print("행 대조", out["row_compare"])
    print("상품-해", out["product_year"], "장기", {k: v for k, v in out["long"].items() if k != "rows"})
    print("SNDK", out["sndk"])
    print("다른 규칙", {k: {w: pct(v[w]["exc_mean"]) for w in MAIN_WINS} for k, v in out["other_rules"].items()})
    print("감사", out["audit"], "넘나듦", out["qqq_switches"])
    print("이웃 모두+", allpos)
    print("위약", json.dumps(rank, ensure_ascii=False, default=float))
    print(f"끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
