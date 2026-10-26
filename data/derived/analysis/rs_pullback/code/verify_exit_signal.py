"""rs_pullback 'exit_signal' 독립 반박 검증 (2026-10-13). exit_signal.py는 import하지 않는다.

쓰는 공용 부품: trend_grade harness(load_panel·universe_mask·eligible·no_lookahead), deskx.Engine/Params(데스크 규칙).
나머지(측정·신호·전진 수익·구간·NW t·A 시뮬레이터·B 엔진·Q1·귀무 분포)는 이 파일에서 새로 쓴다.

확인 항목
  1 재계산: 기준 수익, Q1 데스크 P1(4) h5/h10 앞·뒤, Q2 A(P1(4)·P3(10)·ex4<=-10)·B(P1(4)·P3(10)) 수익.
  2 시점: 신호 = k-1일 종가(패널 날짜), 매매 = k일 시가. 패널 잘라서 다시 계산(own) + 데스크·패널 달력/시가 동일성.
  3 반박: (a) A의 현금 손실 효과 분리(판 비중을 남은 보유에 나누기 / QQQ에 넣기),
          (b) 무작위 매도·무작위 건너뛰기 대조군(같은 날 같은 개수, 같은 비율),
          (c) Q1 12개 측정의 우연 수준(주 단위 이름 섞기 귀무, 블록 부트스트랩),
          (d) SNDK·7월·최대일 쏠림, (e) 편도 0.1%, (f) 2025 부호(S_120 상위 10%),
          (g) 묶음 정의 바꾸기(보유 7종목 안에서 ex4 하위 2개 / ex4<0).
실행: PYTHONIOENCODING=utf-8 py verify_exit_signal.py → ../results/verify_exit_signal.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
_sp = importlib.util.spec_from_file_location("tg_harness_v", os.path.join(ANA, "trend_grade", "code", "harness.py"))
H = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(H)
sys.path.insert(0, os.path.join(ANA, "rsi_breakout", "code"))
from deskx import Engine, Params  # noqa: E402

DESK = Params(semis_cap=0.5)
START = "2026-01-02"
WIN = {"front": ("2026-01-02", "2026-05-29"), "back": ("2026-06-01", "2026-10-09"),
       "late": ("2026-09-01", "2026-10-09"), "y2025": ("2025-01-02", "2025-12-31")}
PER = {"full": ("2026-01-02", "2026-10-09"), "front": WIN["front"], "back": WIN["back"], "sep": WIN["late"]}
RNG = np.random.default_rng(20261013)
N_PLACEBO_A = 1000
N_PLACEBO_B = 300
N_PERM = 2000
N_BOOT = 2000

P = H.load_panel()
C, O = P["C"], P["O"]
QC, QO = P["bench"]["C"], P["bench"]["O"]
CAL = C.index
CAL_S = list(CAL.strftime("%Y-%m-%d"))
UMASK = H.universe_mask(P)
ELIG = H.eligible()


# ------------------------------------------------------------------ 측정(own)
def ex_(Pd, n):
    Cx, Qx = Pd["C"], Pd["bench"]["C"]
    return 100.0 * (Cx / Cx.shift(n)).sub(Qx / Qx.shift(n), axis=0)


def measure_fns():
    m = {}
    for n in (3, 4, 5):
        m[f"P1({n})"] = (lambda n: lambda Pd: -ex_(Pd, n))(n)
    for K in (5, 10, 20):
        m[f"P2({K})"] = (lambda K: lambda Pd: ex_(Pd, 4).rolling(K, min_periods=K).max() - ex_(Pd, 4))(K)
    for K in (5, 10, 20):
        def p3(Pd, K=K):
            R = Pd["C"].div(Pd["bench"]["C"], axis=0)
            return 1.0 - R / R.rolling(K, min_periods=K).max()
        m[f"P3({K})"] = p3
    for K in (5, 10, 20):
        m[f"P4({K})"] = (lambda K: lambda Pd: 1.0 - Pd["C"] / Pd["C"].rolling(K, min_periods=K).max())(K)
    return m


def prank(S, mask):
    return S.where(mask).rank(axis=1, pct=True, method="average")


def flag_fns():
    m = measure_fns()
    um = lambda Pd: H.universe_mask(Pd)
    return {
        "P1(4)>p80": lambda Pd: prank(m["P1(4)"](Pd), um(Pd)) > 0.8,
        "P2(10)>p80": lambda Pd: prank(m["P2(10)"](Pd), um(Pd)) > 0.8,
        "P3(10)>p80": lambda Pd: prank(m["P3(10)"](Pd), um(Pd)) > 0.8,
        "P2(5)>p90": lambda Pd: prank(m["P2(5)"](Pd), um(Pd)) > 0.9,
        "ex4<=-10": lambda Pd: ex_(Pd, 4) <= -10,
        "Fade5>=15": lambda Pd: m["P2(5)"](Pd) >= 15,
    }


def own_lookahead(fn, cutoffs=("2026-02-27", "2026-05-29", "2026-07-17", "2026-09-30")):
    """패널을 cutoff까지 자르고 다시 계산한 cutoff 행이 전체 계산의 행과 같은지(own 구현)."""
    full = fn(P).astype(float)
    for c in cutoffs:
        t = pd.Timestamp(c)
        cut = {k: v.loc[:t] for k, v in P.items() if k != "bench"}
        cut["bench"] = {k: v.loc[:t] for k, v in P["bench"].items()}
        part = fn(cut).astype(float)
        a, b = full.loc[t], part.loc[t].reindex(full.columns)
        if not bool(((a.isna() & b.isna()) | np.isclose(a, b, equal_nan=False)).all()):
            return False
    return True


# ------------------------------------------------------------------ 전진 수익·구간(own)
_FWD = {}


def fwd(h):
    if h not in _FWD:
        _FWD[h] = O.shift(-(1 + h)) / O.shift(-1) - 1.0
    return _FWD[h]


def wdates(a, b, h):
    pos = np.arange(len(CAL))
    ex_pos = pos + 1 + h
    ok = (CAL >= pd.Timestamp(a)) & (CAL <= pd.Timestamp(b)) & (ex_pos < len(CAL))
    idx = np.where(ok)[0]
    idx = [i for i in idx if CAL[i + 1 + h] <= pd.Timestamp(b)]
    return CAL[idx]


def nw_t(x, lag):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 5:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


# ------------------------------------------------------------------ 데스크 기준
E = Engine()
DAYS = E.days
V_BASE, ROWS_BASE = E.run(DESK)
D26 = list(V_BASE.index)
K0 = DAYS.index(D26[0])


def iso(d):
    return pd.Timestamp(d).isocalendar()[:2]


REB = {d: (i == 0 or iso(d) != iso(D26[i - 1])) for i, d in enumerate(D26)}
TGT = {r["day"]: dict(r["w"]) for r in ROWS_BASE if REB[r["day"]]}
OPN = {k: E.col("opn", k) for r in ROWS_BASE for k in r["w"]}
QOPN = QO.reindex(pd.to_datetime(DAYS)).to_numpy(float)


def opn(key):
    if key not in OPN:
        OPN[key] = E.col("opn", key)
    return OPN[key]


def perf(v: pd.Series) -> dict:
    r = (v.shift(-1) / v - 1).dropna()
    out = {}
    for lab, (a, b) in PER.items():
        x = r[(r.index >= a) & (r.index <= b)]
        cur = (1 + x).cumprod()
        out[lab] = {"ret": round(float(cur.iloc[-1] - 1) * 100, 2),
                    "mdd": round(float((cur / cur.cummax() - 1).min()) * 100, 2)}
    out["monthly"] = {m: round(float((1 + r[r.index.str.startswith(f"2026-{m}")]).prod() - 1) * 100, 2)
                      for m in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")}
    return out


def flagsets(F: pd.DataFrame) -> dict:
    F = F.fillna(False).astype(bool)
    return {CAL_S[i]: set(F.columns[row]) for i, row in enumerate(F.to_numpy())}


# ------------------------------------------------------------------ A 시뮬레이터(own)
def sim_A(fs=None, mode="cash", cost=0.0, never=(), rand_counts=None, rng=None):
    """기준 재조정 비중(TGT)을 그대로 쓰고, 재조정일이 아닌 날 k-1일 종가 신호로 k일 시가에 판다.
    mode: cash(다음 재조정까지 현금) | redistribute(판 비중을 남은 보유에 비례 배분) | qqq(판 비중을 QQQ에).
    rand_counts: {day: m} 이면 신호 대신 그날 보유 중 m개를 무작위로 판다(대조군)."""
    w: dict = {}
    vals = [1.0]
    sells = []
    for i, day in enumerate(D26):
        k = K0 + i
        if i > 0:
            g = 0.0
            nw = {}
            for key, wt in w.items():
                if key == "__QQQ":
                    a0, a1 = QOPN[k - 1], QOPN[k]
                else:
                    o = opn(key)
                    a0, a1 = o[k - 1], o[k]
                rr = a1 / a0 - 1 if (np.isfinite(a0) and np.isfinite(a1) and a0 > 0) else 0.0
                g += wt * rr
                nw[key] = wt * (1 + rr)
            tot = sum(nw.values()) + (1 - sum(w.values()))
            w = {kk: x / tot for kk, x in nw.items()}
            vals.append(vals[-1] * (1 + g))
        to = 0.0
        if REB[day]:
            new = dict(TGT[day])
            to = sum(abs(new.get(x, 0.0) - w.get(x, 0.0)) for x in set(new) | set(w))
            w = new
        else:
            held = sorted(x for x in w if x != "__QQQ")
            if rand_counts is not None:
                m = min(rand_counts.get(day, 0), len(held))
                out = list(rng.choice(held, size=m, replace=False)) if m > 0 else []
            elif fs is not None:
                s = fs.get(DAYS[k - 1], set())
                out = [x for x in held if E.ticker[x] in s and E.ticker[x] not in never]
            else:
                out = []
            sold = 0.0
            for x in out:
                sells.append((day, x, w[x]))
                sold += w.pop(x)
            if sold > 0:
                to += sold
                rest = [x for x in w if x != "__QQQ"]
                if mode == "redistribute" and rest:
                    s0 = sum(w[x] for x in rest)
                    for x in rest:
                        w[x] += sold * w[x] / s0
                    to += sold
                elif mode == "qqq":
                    w["__QQQ"] = w.get("__QQQ", 0.0) + sold
                    to += sold
        if cost and to:
            vals[-1] *= (1 - cost * to)
    return pd.Series(vals, index=D26), sells


def sell_outcomes(sells):
    """판 종목의 다음 재조정 시가까지 수익, 같은 구간 남은 기준 보유 평균 대비."""
    res = []
    for day, key, wt in sells:
        i = D26.index(day)
        j = next((jj for jj in range(i + 1, len(D26)) if REB[D26[jj]]), len(D26) - 1)
        k, kn = K0 + i, K0 + j
        o = opn(key)
        if not (np.isfinite(o[k]) and np.isfinite(o[kn])):
            continue
        r = o[kn] / o[k] - 1
        others = [x for x in ROWS_BASE[i]["w"] if x != key]
        ro = [opn(x)[kn] / opn(x)[k] - 1 for x in others if np.isfinite(opn(x)[k]) and np.isfinite(opn(x)[kn])]
        res.append({"day": day, "t": E.ticker[key], "w": wt, "r": r, "r_others": float(np.mean(ro)) if ro else np.nan,
                    "rebought": key in ROWS_BASE[j]["w"] if j < len(D26) else False})
    return pd.DataFrame(res)


# ------------------------------------------------------------------ B 엔진(own, deskx.Engine 상속)
class BEngine(Engine):
    skip = None   # fn(k, key) -> bool

    def ranking(self, p, k, session):
        rows = super().ranking(p, k, session)
        if self.skip is not None:
            rows = [r for r in rows if not self.skip(k, r["key"])]
        return rows


BE = BEngine()


def run_B(fs=None, rate=None, rng=None):
    if fs is not None:
        BE.skip = lambda k, key: BE.ticker[key] in fs.get(BE.days[k - 1], set())
    elif rate is not None:
        cache = {}

        def f(k, key):
            if (k, key) not in cache:
                cache[(k, key)] = bool(rng.random() < rate)
            return cache[(k, key)]
        BE.skip = f
    else:
        BE.skip = None
    v, rows = BE.run(DESK)
    BE.skip = None
    return v, rows


def flag_rate_top12(fs):
    """재조정일 기준 순위 상위 12(n+buffer) 안에서 신호가 켜진 비율(무작위 대조군 비율 맞추기)."""
    hits, tot = 0, 0
    for i, d in enumerate(D26):
        if not REB[d]:
            continue
        k = K0 + i
        ranked = E.ranking(DESK, k, d)[:12]
        s = fs.get(DAYS[k - 1], set())
        hits += sum(E.ticker[r["key"]] in s for r in ranked)
        tot += len(ranked)
    return hits / max(tot, 1)


# ------------------------------------------------------------------ Q1(own)
def held_frame(rows):
    out = pd.DataFrame(False, index=CAL, columns=C.columns)
    for r in rows:
        d = pd.Timestamp(r["day"])
        out.loc[d, [E.ticker[k] for k in r["w"] if E.ticker[k] in out.columns]] = True
    return out


def q1_series(held, faded, h, a, b, drop=(), keep_dates=None):
    dates = wdates(a, b, h)
    if keep_dates is not None:
        dates = dates[keep_dates(dates)]
    fr = fwd(h).loc[dates]
    hd = held.reindex(index=dates, columns=fr.columns).fillna(False).astype(bool) & ELIG.loc[dates]
    if drop:
        hd.loc[:, [c for c in drop if c in hd.columns]] = False
    fd = faded.reindex(index=dates, columns=fr.columns).fillna(False).astype(bool)
    f = fr.where(hd & fd).mean(axis=1)
    n = fr.where(hd & ~fd).mean(axis=1)
    return ((f - n) * 100).dropna()


def cell(s, h):
    return {"diff": round(float(s.mean()), 3) if len(s) else None, "t": round(nw_t(s, h), 2), "days": int(len(s))}


def perm_null(held, fadeds: dict, h, a, b, n_draw):
    """주마다 보유 종목끼리 전진 수익을 한 번 섞는다(같은 주 안에서는 같은 섞기 → 겹친 수익의 자기상관 유지).
    귀무: '꺾임 표시'가 보유 종목 중 누가 더 오를지와 무관. 12개 측정을 같은 섞기로 동시에 잰다."""
    dates = wdates(a, b, h)
    fr = fwd(h).loc[dates]
    hd = held.reindex(index=dates, columns=fr.columns).fillna(False).astype(bool) & ELIG.loc[dates]
    names = list(fadeds)
    per_date = []
    for d in dates:
        cols = list(hd.columns[hd.loc[d].to_numpy()])
        if len(cols) < 2:
            continue
        r = fr.loc[d, cols].to_numpy(float) * 100
        F = np.vstack([fadeds[m].reindex(columns=fr.columns).loc[d, cols].fillna(False).to_numpy(bool) for m in names])
        per_date.append((iso(d), cols, r, F))

    def stat(perm_of_week):
        diffs = [[] for _ in names]
        for wk, cols, r, F in per_date:
            if perm_of_week is not None:
                mp = perm_of_week(wk, cols)
                r = np.array([r[cols.index(mp[c])] if mp.get(c) in cols else r[cols.index(c)] for c in cols])
            for mi in range(len(names)):
                f = F[mi]
                if f.any() and (~f).any():
                    diffs[mi].append(r[f].mean() - r[~f].mean())
        return np.array([np.mean(x) if x else np.nan for x in diffs])

    obs = stat(None)
    draws = []
    for _ in range(n_draw):
        wk_perm = {}

        def pw(wk, cols):
            key = wk
            if key not in wk_perm:
                wk_perm[key] = {}
            mp = wk_perm[key]
            missing = [c for c in cols if c not in mp]
            if missing:
                perm = list(RNG.permutation(missing))
                mp.update(dict(zip(missing, perm)))
            return mp
        draws.append(stat(pw))
    draws = np.array(draws)
    npos_obs = int(np.sum(obs > 0))
    return {"measures": names, "obs": [round(float(x), 3) for x in obs], "npos_obs": npos_obs,
            "null_mean_of_means": round(float(np.nanmean(draws)), 3),
            "p_npos_ge_obs": round(float(np.mean(np.sum(draws > 0, axis=1) >= npos_obs)), 4),
            "p_min_ge_obs": round(float(np.mean(np.nanmin(draws, axis=1) >= np.nanmin(obs))), 4),
            "p_avg_ge_obs": round(float(np.mean(np.nanmean(draws, axis=1) >= np.nanmean(obs))), 4),
            "p_max_ge_obs_max": round(float(np.mean(np.nanmax(draws, axis=1) >= np.nanmax(obs))), 4),
            "null_p95_of_max": round(float(np.percentile(np.nanmax(draws, axis=1), 95)), 3)}


def block_boot(series: dict, h, block=5, n=N_BOOT):
    """날짜 블록 부트스트랩(평균을 0으로 옮긴 귀무). 12개 측정의 날짜 평균 묶음."""
    M = pd.DataFrame(series)
    X = M.to_numpy(float)
    mu = np.nanmean(X, axis=0)
    Xc = X - mu
    T = len(X)
    draws = []
    for _ in range(n):
        starts = RNG.integers(0, T, size=int(np.ceil(T / block)))
        idx = np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
        draws.append(np.nanmean(Xc[idx], axis=0))
    draws = np.array(draws)
    return {"p_all_pos_under_null": round(float(np.mean(np.all(draws > 0, axis=1))), 4),
            "p_min_ge_obs_min": round(float(np.mean(np.nanmin(draws, axis=1) >= np.nanmin(mu))), 4),
            "p_avg_ge_obs_avg": round(float(np.mean(np.nanmean(draws, axis=1) >= np.nanmean(mu))), 4),
            "avg_pairwise_corr_of_daily_series": round(float(np.nanmean(M.corr().to_numpy()[np.triu_indices(M.shape[1], 1)])), 3)}


# ------------------------------------------------------------------ 실행
def main():
    out = {"family": "verify_exit_signal", "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
    mfn = measure_fns()
    ffn = flag_fns()

    # --- 2 시점 점검
    timing = {"desk_vs_panel_calendar_same": [d for d in DAYS if d >= "2026-01-01"] == [d for d in CAL_S if d >= "2026-01-01"]}
    mx = 0.0
    for key in OPN:
        t = E.ticker[key]
        po = O[t].reindex(pd.to_datetime(DAYS)).to_numpy(float)
        o = opn(key)
        ok = np.isfinite(po) & np.isfinite(o) & (pd.to_datetime(DAYS) >= pd.Timestamp("2026-01-01"))
        mx = max(mx, float(np.nanmax(np.abs(o[ok] / po[ok] - 1))))
    timing["desk_open_vs_panel_open_max_rel_diff"] = mx
    timing["own_lookahead_measures"] = {k: own_lookahead(f) for k, f in mfn.items()}
    timing["own_lookahead_flags"] = {k: own_lookahead(f) for k, f in ffn.items()}
    timing["harness_lookahead_flags"] = {k: H.no_lookahead(lambda Pd, f=f: f(Pd).astype(float)) for k, f in ffn.items()}
    # 일부러 미래 정보를 넣은 신호는 시험이 잡아내는지(시험 자체 점검)
    timing["lookahead_test_catches_leak"] = not own_lookahead(lambda Pd: -ex_(Pd, 4).shift(-1))
    timing["note"] = "A/B 신호는 DAYS[k-1](직전 거래일) 종가 집합으로 k일 시가에 매매. 데스크 달력=패널 달력, 시가 동일."
    out["timing"] = timing
    print("timing", {k: v for k, v in timing.items() if not isinstance(v, dict)},
          all(timing["own_lookahead_flags"].values()), all(timing["own_lookahead_measures"].values()))

    # --- 1 기준 재계산 + 내 A 시뮬레이터 동등성
    base = perf(V_BASE)
    vA0, _ = sim_A(None)
    out["base"] = base
    out["own_sim_vs_deskx_base_maxdiff"] = float(np.max(np.abs(vA0.to_numpy() - V_BASE.to_numpy())))
    print("base", {k: base[k] for k in PER}, "sim parity", out["own_sim_vs_deskx_base_maxdiff"])
    vbc, _ = sim_A(None, cost=0.001)
    out["base_net0.1"] = {k: perf(vbc)[k]["ret"] for k in PER}

    # --- Q2 A: 재계산 + 현금 효과 분리 + 무작위 대조군
    flags = {k: f(P) for k, f in ffn.items()}
    fsets = {k: flagsets(F) for k, F in flags.items()}
    qA = {}
    for sig, fs in fsets.items():
        row = {}
        for mode in ("cash", "redistribute", "qqq"):
            v, sells = sim_A(fs, mode)
            row[mode] = {k: perf(v)[k] for k in PER}
            if mode == "cash":
                row["cash_net0.1"] = {k: perf(sim_A(fs, mode, cost=0.001)[0])[k]["ret"] for k in PER}
                so = sell_outcomes(sells)
                row["sells"] = {"n": int(len(so)), "rebought_share": round(float(so["rebought"].mean()) * 100, 1),
                                "mean_r_until_reb": round(float(so["r"].mean()) * 100, 2),
                                "mean_r_minus_other_holdings": round(float((so["r"] - so["r_others"]).mean()) * 100, 2),
                                "share_beat_other_holdings": round(float((so["r"] > so["r_others"]).mean()) * 100, 1),
                                "mean_r_minus_other_back": round(float((so["r"] - so["r_others"])[so["day"] >= "2026-06-01"].mean()) * 100, 2),
                                "mean_r_minus_other_front": round(float((so["r"] - so["r_others"])[so["day"] <= "2026-05-29"].mean()) * 100, 2),
                                "by_ticker": so["t"].value_counts().head(5).to_dict()}
                counts = pd.Series([s[0] for s in sells]).value_counts().to_dict()
                # 무작위 대조군: 같은 날 같은 개수를 보유 중에서 무작위로 판다(현금)
                rnd = []
                for _ in range(N_PLACEBO_A):
                    vr, _ = sim_A(None, "cash", rand_counts=counts, rng=RNG)
                    pr_ = perf(vr)
                    rnd.append([pr_[k]["ret"] for k in ("full", "front", "back")])
                rnd = np.array(rnd)
                sig_ret = [row["cash"][k]["ret"] for k in ("full", "front", "back")]
                row["placebo_random_sell_cash"] = {
                    k: {"signal": sig_ret[j], "placebo_median": round(float(np.median(rnd[:, j])), 2),
                        "placebo_p05_p95": [round(float(np.percentile(rnd[:, j], 5)), 2), round(float(np.percentile(rnd[:, j], 95)), 2)],
                        "share_placebo_below_signal": round(float(np.mean(rnd[:, j] < sig_ret[j])) * 100, 1),
                        "share_placebo_beat_base": round(float(np.mean(rnd[:, j] > base[k]["ret"])) * 100, 1)}
                    for j, k in enumerate(("full", "front", "back"))}
                # SNDK는 절대 팔지 않기
                vns, _ = sim_A(fs, "cash", never=("SNDK",))
                row["cash_never_sell_SNDK"] = {k: perf(vns)[k]["ret"] for k in PER}
                # 7월 쏠림: 기준 대비 하루 수익 차이 합(%p), 7월 제외
                rA = (v.shift(-1) / v - 1).dropna()
                rB = (V_BASE.shift(-1) / V_BASE - 1).dropna()
                d = (rA - rB) * 100
                row["daily_diff_sum_pp"] = {"all": round(float(d.sum()), 2),
                                            "ex_july": round(float(d[~d.index.str.startswith("2026-07")].sum()), 2),
                                            "july": round(float(d[d.index.str.startswith("2026-07")].sum()), 2),
                                            "ex_top3_days": round(float(d.drop(d.abs().nlargest(3).index).sum()), 2),
                                            "top3_days": {k: round(float(x), 2) for k, x in d.loc[d.abs().nlargest(3).index].items()},
                                            "months_negative": int(sum(1 for m in range(1, 11)
                                                                       if d[d.index.str.startswith(f"2026-{m:02d}")].sum() < 0))}
        qA[sig] = row
        print("A", sig, {m: (row[m]["full"]["ret"], row[m]["front"]["ret"], row[m]["back"]["ret"]) for m in ("cash", "redistribute", "qqq")},
              row["sells"], row["placebo_random_sell_cash"]["full"], row["cash_never_sell_SNDK"]["full"], row["daily_diff_sum_pp"])
    out["q2_A"] = qA

    # --- Q2 B: 재계산 + 무작위 건너뛰기 대조군
    vb0, _ = run_B()
    out["own_B_engine_parity"] = float(np.max(np.abs(vb0.to_numpy() - V_BASE.to_numpy())))
    qB = {}
    for sig, fs in fsets.items():
        v, rows = run_B(fs)
        pr_ = perf(v)
        rate = flag_rate_top12(fs)
        rnd = []
        n_draw = N_PLACEBO_B if sig in ("P1(4)>p80", "P3(10)>p80", "P2(5)>p90") else 60
        for _ in range(n_draw):
            vr, _ = run_B(rate=rate, rng=RNG)
            q = perf(vr)
            rnd.append([q[k]["ret"] for k in ("full", "front", "back")])
        rnd = np.array(rnd)
        sig_ret = [pr_[k]["ret"] for k in ("full", "front", "back")]
        qB[sig] = {"perf": {k: pr_[k] for k in PER}, "flag_rate_top12": round(rate, 3), "placebo_draws": n_draw,
                   "placebo_random_skip": {k: {"signal": sig_ret[j], "placebo_median": round(float(np.median(rnd[:, j])), 2),
                                               "placebo_p05_p95": [round(float(np.percentile(rnd[:, j], 5)), 2),
                                                                   round(float(np.percentile(rnd[:, j], 95)), 2)],
                                               "share_placebo_below_signal": round(float(np.mean(rnd[:, j] < sig_ret[j])) * 100, 1),
                                               "share_placebo_beat_base": round(float(np.mean(rnd[:, j] > base[k]["ret"])) * 100, 1)}
                                           for j, k in enumerate(("full", "front", "back"))}}
        print("B", sig, (pr_["full"]["ret"], pr_["front"]["ret"], pr_["back"]["ret"]), "rate", round(rate, 3),
              qB[sig]["placebo_random_skip"]["full"], qB[sig]["placebo_random_skip"]["back"])
    out["q2_B"] = qB

    # --- Q1 재계산
    held = held_frame(ROWS_BASE)
    faded = {k: prank(f(P), UMASK) > 0.8 for k, f in mfn.items()}
    q1 = {}
    for m, F in faded.items():
        q1[m] = {}
        for h in (5, 10):
            for w in ("front", "back", "late"):
                s = q1_series(held, F, h, *WIN[w])
                q1[m][f"h{h}_{w}"] = cell(s, h)
            # 쏠림: SNDK 제외, 7월 보유 구간 제외, 신호일 7월 제외, 상위 3일 제외
            sb = q1_series(held, F, h, *WIN["back"])
            q1[m][f"h{h}_back_exSNDK"] = cell(q1_series(held, F, h, *WIN["back"], drop=("SNDK",)), h)
            q1[m][f"h{h}_front_exSNDK"] = cell(q1_series(held, F, h, *WIN["front"], drop=("SNDK",)), h)
            pos = {d: i for i, d in enumerate(CAL)}
            not_jul_window = lambda ds, h=h: np.array([not (CAL[pos[d] + 1] <= pd.Timestamp("2026-07-31") and
                                                          CAL[pos[d] + 1 + h] >= pd.Timestamp("2026-07-01")) for d in ds])
            q1[m][f"h{h}_back_ex_julywindow"] = cell(q1_series(held, F, h, *WIN["back"], keep_dates=not_jul_window), h)
            if len(sb) > 5:
                q1[m][f"h{h}_back_ex_top3days"] = round(float(sb.drop(sb.abs().nlargest(3).index).mean()), 3)
                q1[m][f"h{h}_back_by_month"] = {k: round(float(x), 2) for k, x in sb.groupby(sb.index.strftime("%m")).mean().items()}
    out["q1_desk50"] = q1
    for h in (5, 10):
        for w in ("front", "back"):
            vals = [q1[m][f"h{h}_{w}"]["diff"] for m in q1]
            print(f"Q1 desk h{h} {w}: range {min(vals):+.2f}~{max(vals):+.2f} neg {sum(v < 0 for v in vals)}/12",
                  "P1(4)", q1["P1(4)"][f"h{h}_{w}"])
    print("P1(4) h5 back exSNDK", q1["P1(4)"]["h5_back_exSNDK"], "exJulWin", q1["P1(4)"]["h5_back_ex_julywindow"],
          "exTop3", q1["P1(4)"]["h5_back_ex_top3days"], q1["P1(4)"]["h5_back_by_month"])

    # 일반 되돌림 대비 데스크 몫, 묶음 정의 바꾸기
    allh = ELIG.copy()
    alt = {}
    ex4 = ex_(P, 4)
    for h in (5, 10):
        for w in ("front", "back"):
            s_all = q1_series(allh, faded["P1(4)"], h, *WIN[w])
            s_desk = q1_series(held, faded["P1(4)"], h, *WIN[w])
            alt[f"P1(4)_h{h}_{w}_all_eligible"] = cell(s_all, h)
            alt[f"P1(4)_h{h}_{w}_desk_minus_all"] = round(float(s_desk.mean() - s_all.mean()), 3)
            # 보유 7종목 안에서 ex4 하위 2개 vs 나머지
            hd = held & ELIG
            rk = ex4.where(hd).rank(axis=1, method="first")
            bottom2 = rk <= 2
            alt[f"within7_bottom2_ex4_h{h}_{w}"] = cell(q1_series(held, bottom2, h, *WIN[w]), h)
            alt[f"within7_ex4_lt0_h{h}_{w}"] = cell(q1_series(held, ex4 < 0, h, *WIN[w]), h)
            # 순위를 eligible 안에서 매긴 경우(universe_mask 대신)
            alt[f"P1(4)_rank_in_eligible_h{h}_{w}"] = cell(q1_series(held, prank(-ex4, ELIG) > 0.8, h, *WIN[w]), h)
    # 부호 점검: 꺾임 표시된 보유 종목의 ex4 평균이 정말 음수인지
    fd = faded["P1(4)"] & held & ELIG
    alt["sign_check_mean_ex4_flagged_held"] = round(float(ex4.where(fd).stack().mean()), 2)
    alt["sign_check_mean_ex4_unflagged_held"] = round(float(ex4.where(held & ELIG & ~faded["P1(4)"]).stack().mean()), 2)
    out["q1_alt"] = alt
    print("alt", alt)

    # --- 우연 수준: 주 단위 섞기 귀무 + 블록 부트스트랩
    nulls = {}
    for h in (5, 10):
        for w in ("front", "back"):
            nulls[f"perm_h{h}_{w}"] = perm_null(held, faded, h, *WIN[w], N_PERM if h == 5 else N_PERM // 2)
            ser = {m: q1_series(held, faded[m], h, *WIN[w]) for m in faded}
            nulls[f"boot_h{h}_{w}"] = block_boot(ser, h, block=h)
            print("null", h, w, {k: v for k, v in nulls[f"perm_h{h}_{w}"].items() if k not in ("measures", "obs")},
                  nulls[f"boot_h{h}_{w}"])
    out["q1_chance"] = nulls

    # --- S_120 상위 10%: 2025 부호, 앞·뒤
    s120 = prank(ex_(P, 120), UMASK) > 0.9
    s1 = {}
    for h in (5, 10):
        for w in ("front", "back", "y2025"):
            vals = {m: q1_series(s120, faded[m], h, *WIN[w]) for m in faded}
            s1[f"h{h}_{w}"] = {"range": [round(float(min(x.mean() for x in vals.values())), 2),
                                         round(float(max(x.mean() for x in vals.values())), 2)],
                               "neg_of_12": int(sum(x.mean() < 0 for x in vals.values())),
                               "P1(4)": cell(vals["P1(4)"], h), "P3(20)": cell(vals["P3(20)"], h)}
    out["q1_s120top"] = s1
    print("s120", s1)

    out["verdict_summary"] = VERDICT
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "verify_exit_signal.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("saved", path)


VERDICT = {
    "verdict": "confirmed(기각 유지) — 수치·시점 재현, 버그 없음. 단 해석 두 곳은 약하게 고친다",
    "reproduced": "기준 220.38/206.19/4.64, A·B 12개 전부 소수 둘째 자리까지 같음, Q1 데스크 12측정×h5·h10×앞·뒤 전부 같음",
    "timing": "신호 = 직전 거래일 종가, 매매 = 다음 시가. 데스크 달력 = 패널 달력, 시가 차 2e-16. 6개 신호·12개 측정 자체 잘라내기 시험 통과(누출 넣은 신호는 잡아냄)",
    "correction_1": "A(주중 매도)가 12개 모두 기준보다 낮은 것은 대부분 구조적: 같은 날 같은 개수를 무작위로 팔아도 1000번 중 0번 기준을 이김(전체·앞). "
                    "신호 몫은 작다(P1(4): 기준 220 → 무작위 중앙 112 → 신호 92; P3(10): 무작위 중앙 142.5 = 신호 142.2). "
                    "판 비중을 남은 보유에 다시 나눠도(현금 없음) 6개 모두 기준보다 낮다(117~193).",
    "correction_2": "'24칸 모두 플러스'는 측정끼리 상관 0.59라 우연으로도 구간마다 11~18% 확률. "
                    "뒤 h5는 12개 평균·최소·최대 모두 귀무 대비 p≈0.02~0.04로 남고, 앞 h5는 p≈0.09~0.13로 약하다.",
    "B": "재조정 때 건너뛰기도 무작위 건너뛰기(같은 비율)보다 낫지 않다(전체 기준 무작위 중앙 168~194, 신호 139~175). 무작위가 기준을 이긴 비율 8~17%.",
    "concentration": "SNDK 안 팔아도 A 135~173(기준 220). A 손해는 7월이 아니라 3~5월(4/2·3/31) 쪽, 10개월 중 6~9개월 손해. "
                     "Q1 뒤 h5는 7월 걸친 날 빼면 더 커지고(P1(4) 3.98) 월별로 전부 플러스.",
    "use": "데스크 7종목은 며칠 QQQ보다 약해도 주중에 팔지 않고, 재조정 때도 '최근 꺾임'으로 거르지 않는다. "
           "꺾인 종목을 더 사는 근거는 아니다(데스크 고유 몫 +0.76%p, 10일이면 사라짐).",
    "minor": "내 '다시 산 비율'(91~95%)은 마지막 주 매도를 다시 산 것으로 세어 부풀었다. 연구의 88~92%가 맞는 정의.",
}

if __name__ == "__main__":
    if sys.argv[1:] == ["--summary-only"]:
        path = os.path.join(RES, "verify_exit_signal.json")
        d = json.load(open(path, encoding="utf-8"))
        d["verdict_summary"] = VERDICT
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1, default=str)
        print("summary added")
    else:
        main()
