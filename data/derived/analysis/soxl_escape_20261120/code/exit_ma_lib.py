"""갈래 exit_ma — 이평선·추적 손절 긴급탈출 신호와 채점 도우미 (2026-11-20).

harness.py의 자료·지표·체결·채점을 그대로 쓴다. 이 파일은 신호 정의와
'진입 기준 추적'(hold_sim과 같은 진입·무작위 대조 재현)만 더한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import harness as H


# ───────────── 지표(파인 정의) ─────────────
def supertrend_dir(b: pd.DataFrame, factor: float, n: int = 10) -> pd.Series:
    """ta.supertrend(factor, n)의 direction. 1 = 하락 추세(선이 위), -1 = 상승 추세."""
    hl2 = ((b["High"] + b["Low"]) / 2).to_numpy(float)
    a = H.atr(b, n).to_numpy(float)
    c = b["Close"].to_numpy(float)
    m = len(c)
    lower = np.full(m, np.nan)
    upper = np.full(m, np.nan)
    st = np.full(m, np.nan)
    dr = np.full(m, np.nan)
    for i in range(m):
        if np.isnan(a[i]):
            continue
        lo = hl2[i] - factor * a[i]
        up = hl2[i] + factor * a[i]
        pl = lower[i - 1] if i > 0 and not np.isnan(lower[i - 1]) else lo
        pu = upper[i - 1] if i > 0 and not np.isnan(upper[i - 1]) else up
        pc = c[i - 1] if i > 0 else c[i]
        lower[i] = lo if (lo > pl or pc < pl) else pl
        upper[i] = up if (up < pu or pc > pu) else pu
        if i == 0 or np.isnan(a[i - 1]):
            dr[i] = 1
        elif st[i - 1] == upper[i - 1]:
            dr[i] = -1 if c[i] > upper[i] else 1
        else:
            dr[i] = 1 if c[i] < lower[i] else -1
        st[i] = lower[i] if dr[i] == -1 else upper[i]
    return pd.Series(dr, index=b.index)


def runlen(s: pd.Series) -> pd.Series:
    """연속 True 길이."""
    v = s.fillna(False).to_numpy(bool)
    out = np.zeros(len(v), dtype=int)
    for i in range(len(v)):
        out[i] = out[i - 1] + 1 if v[i] and i > 0 else (1 if v[i] else 0)
    return pd.Series(out, index=s.index)


def reg_cummax_high(b: pd.DataFrame) -> pd.Series:
    """정규장(09:30~) 시작 뒤 지금까지 최고가. 정규장 밖은 NaN."""
    hh = b["High"].where(b["reg"])
    return hh.groupby(b["date"]).cummax()


# ───────────── 신호 → 체결 위치 ─────────────
def state_and_event(sig: pd.Series, b: pd.DataFrame):
    """상태 bool → (상태 체결 위치 전부, 사건=켜지는 순간 체결 위치)."""
    sig = sig.fillna(False).astype(bool)
    onset = sig & ~sig.shift(1, fill_value=False)
    sp = H.exec_index(sig, b)
    ep = H.exec_index(onset, b)
    return sp[sp >= 0], ep[ep >= 0]


# ───────────── 진입 기준 추적(hold_sim 재현) ─────────────
def _entries(rng, entry_mask=None, step=5):
    d = H.load_1m()
    f = H._fwd_tables()
    ent = H.baseline_positions(rng, entry_mask)
    ent = ent[(d["mod"].values[ent] - H.REG_OPEN) % step == 0]
    ent = ent[ent < f["last_reg"][ent]]
    if (d.index[-1].hour * 60 + d.index[-1].minute) < H.REG_CLOSE:
        ent = ent[d["date"].values[ent] != d["date"].values[-1]]
    return ent


def entry_trail_exits(ent: np.ndarray, x: float, tf: int) -> np.ndarray:
    """진입(1분봉 시가) 뒤 최고가(1분봉 고가 누적, 진입 봉 포함) 대비 종가가 −x 이하인 첫 tf분봉 완성 → 다음 1분봉 시가.
    tf=3이면 3분 칸의 마지막 1분봉(자정 기준 칸 끝)에서만 검사. 반환: 진입별 탈출 체결 위치(없으면 10**12)."""
    d = H.load_1m()
    f = H._fwd_tables()
    Hh, C = d["High"].to_numpy(float), d["Close"].to_numpy(float)
    mod = d["mod"].to_numpy()
    lr = f["last_reg"]
    chk = np.ones(len(d), bool) if tf == 1 else ((mod + 1) % tf == 0)
    out = np.full(len(ent), 10**12, dtype=np.int64)
    for k, e in enumerate(ent):
        end = lr[e]
        hh = np.maximum.accumulate(Hh[e:end + 1])
        hit = np.nonzero((C[e:end + 1] <= hh * (1 - x)) & chk[e:end + 1])[0]
        if len(hit):
            out[k] = e + hit[0] + 1
    return out


def hold_sim_from_exits(ent: np.ndarray, ex: np.ndarray, n_perm: int = 20, seed: int = 7) -> dict:
    """harness.hold_sim과 같은 손익·무작위 대조(같은 날 보유시간 섞기) 계산, 진입별 탈출 위치를 받는다."""
    d = H.load_1m()
    O, C = d["Open"].to_numpy(float), d["Close"].to_numpy(float)
    f = H._fwd_tables()
    lr = f["last_reg"][ent]
    use = ex <= lr
    exit_px = np.where(use, O[np.minimum(ex, len(O) - 1)], C[lr])
    pnl_rule = exit_px / O[ent] - 1
    pnl_hold = C[lr] / O[ent] - 1
    dur = np.where(use, ex - ent, lr - ent + 1)
    rng_ = np.random.default_rng(seed)
    dates = d["date"].values[ent]
    pr = []
    for _ in range(n_perm):
        dd = dur.copy()
        for dt in np.unique(dates):
            m = np.nonzero(dates == dt)[0]
            dd[m] = rng_.permutation(dd[m])
        e2 = ent + dd
        px2 = np.where(e2 <= lr, O[np.minimum(e2, len(O) - 1)], C[lr])
        pr.append(px2 / O[ent] - 1)
    pnl_rand = np.mean(pr, axis=0)

    def tail(x, q=0.05):
        c = np.nanquantile(x, q)
        return float(np.nanmean(x[x <= c]))

    g = pd.DataFrame({"d": dates, "x": pnl_rule - pnl_rand}).groupby("d")["x"].mean()
    return {
        "entries": int(len(ent)), "exit_rate%": round(100 * use.mean(), 1),
        "avg_hold_min": round(float(dur.mean()), 1),
        "rule_mean%": round(100 * np.nanmean(pnl_rule), 3), "hold_mean%": round(100 * np.nanmean(pnl_hold), 3),
        "rand_mean%": round(100 * np.nanmean(pnl_rand), 3),
        "rule_cvar5%": round(100 * tail(pnl_rule), 3), "hold_cvar5%": round(100 * tail(pnl_hold), 3),
        "rand_cvar5%": round(100 * tail(pnl_rand), 3),
        "x_rand_mean%": round(100 * float(np.nanmean(pnl_rule - pnl_rand)), 3),
        "x_rand_t_days": round(float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), 2) if len(g) > 2 and g.std(ddof=1) > 0 else None,
        "rule_worst%": round(100 * float(np.nanmin(pnl_rule)), 2), "hold_worst%": round(100 * float(np.nanmin(pnl_hold)), 2),
    }


def entry_trail_sim(rng, x, tf, entry_mask=None):
    ent = _entries(rng, entry_mask)
    ex = entry_trail_exits(ent, x, tf)
    return hold_sim_from_exits(ent, ex), ex[ex < 10**12]
