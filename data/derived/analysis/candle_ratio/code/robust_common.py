"""robust 갈래 공통: 시작 오프셋을 넣은 봉 묶음, 봉별 보유·수익, 구간 지표(전략·보유·같은 비중 그냥 보유).

engine.py는 고치지 않는다. tf_bars_off는 engine.tf_bars를 복사해 시작 오프셋(off분)과
기본봉→묶음봉 번호(gid)를 더했다. off=0이면 engine.tf_bars와 같다(robust_run.py가 확인).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from engine import ET, SESS, WIN, CACHE, HERE, awake_mask, bh, load, positions, ratio, segments, sim  # noqa: F401

ORIG = dict(tf=40, sess="eth", N=10, kind="count", b=0.5, s=0.5, follow=True)
COST_REAL = {"eth": 0.001, "rth": 0.0005}
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
RES = HERE / "results"


def tf_bars_off(base: pd.DataFrame, minutes: int, sess: str, off: int = 0):
    """engine.tf_bars + 시작 오프셋. 묶음 경계 = 세션 시작 + off + minutes·k.
    off>0이면 세션 시작~시작+off가 짧은 첫 봉(k=-1)이 된다(트레이딩뷰 정렬이 04:00이 아닐 때의 모양).
    돌려주는 것: (묶음봉 표, 세션 안 기본봉 표(gid = 묶음봉 번호))."""
    start, end = SESS[sess]
    t = base.index
    mins = np.asarray(t.hour * 60 + t.minute)
    keep = (mins >= start) & (mins < end)
    b = base[keep]
    mins = mins[keep]
    dates = np.asarray(b.index.date)
    k = (mins - start - off) // minutes
    frame = pd.DataFrame({"O": b.Open.values, "H": b.High.values, "L": b.Low.values, "C": b.Close.values,
                          "V": b.Volume.values, "d": dates, "k": k, "t": b.index})
    g = frame.groupby(["d", "k"], sort=True)
    out = pd.DataFrame({"O": g.O.first(), "H": g.H.max(), "L": g.L.min(), "C": g.C.last(), "V": g.V.sum(),
                        "t": g.t.first()})
    idx = out.index.to_frame(index=False)
    out = out.reset_index(drop=True)
    out["day"] = idx.d.values
    st = np.maximum(start + off + idx.k.values * minutes, start)
    out["start"] = pd.DatetimeIndex(pd.to_datetime(idx.d.values)).tz_localize(ET) + pd.to_timedelta(st, unit="m")
    out["lod"] = np.r_[out.day.values[1:] != out.day.values[:-1], True]
    rows = pd.DataFrame({"O": b.Open.values, "C": b.Close.values, "t": b.index, "d": dates, "m": mins,
                         "gid": g.ngroup().values})
    return out, rows


def bars_for(sym: str, base: str, tf: int, sess: str = "eth", off: int = 0, d: pd.DataFrame | None = None):
    d = load(sym, base) if d is None else d
    tb, rows = tf_bars_off(d, tf, sess, off)
    w = WIN[base]
    keep = tb.day.values >= w["warm"]
    first = int(np.argmax(keep)) if keep.any() else len(tb)
    tb = tb.iloc[first:].reset_index(drop=True)
    rows = rows[rows.gid >= first].reset_index(drop=True)
    rows["gid"] -= first
    O, C, V = (tb[k].values.astype(np.float64) for k in ("O", "C", "V"))
    seg = segments(tb.day.values, w)
    bpd = int(pd.Series(tb.day.values).value_counts().median()) if len(tb) else 0
    return dict(tb=tb, rows=rows, O=O, C=C, V=V, lod=tb.lod.values, seg=seg, awake=awake_mask(tb.start),
                ones=np.ones(len(tb), np.bool_), bpd=bpd, day=tb.day.values)


def pos_ret(O, C, p, awake, b, s, follow, cost, lod=None):
    """봉별 보유(시가→다음 시가)와 봉별 로그수익(비용 포함). 밤 보유(dayflat 아님)만."""
    lod = np.zeros(len(O), np.bool_) if lod is None else lod
    pos, _ = positions(O, C, p, lod, awake, b, s, follow, False)
    rb = np.r_[np.log(O[1:] / O[:-1]), np.log(C[-1] / O[-1])]
    chg = np.r_[pos[0] != 0, pos[1:] != pos[:-1]]
    r = pos * rb + np.log(1.0 - cost) * chg
    return pos, r, rb


@njit(cache=True)
def wmetrics(r, rb, pos, mask, out):
    """mask 구간 지표. out: [0] 전략 로그수익 [1] 전략 최대 낙폭(로그) [2] 보유 로그수익 [3] 보유 최대 낙폭(로그)
    [4] 평균 보유 비율 x [5] 같은 비중 그냥 보유 단순수익 [6] 그 최대 낙폭(단순) [7] 봉 수."""
    eq = 0.0; pk = 0.0; mdd = 0.0; beq = 0.0; bpk = 0.0; bmdd = 0.0; n = 0; h = 0
    for j in range(len(r)):
        if mask[j]:
            eq += r[j]; pk = max(pk, eq); mdd = min(mdd, eq - pk)
            beq += rb[j]; bpk = max(bpk, beq); bmdd = min(bmdd, beq - bpk)
            n += 1; h += pos[j]
    x = h / n if n > 0 else 0.0
    e = 1.0; epk = 1.0; emdd = 0.0; cum = 0.0
    for j in range(len(r)):
        if mask[j]:
            cum += rb[j]
            e = 1.0 + x * (np.exp(cum) - 1.0)
            epk = max(epk, e)
            emdd = min(emdd, e / epk - 1.0)
    out[0] = eq; out[1] = mdd; out[2] = beq; out[3] = bmdd; out[4] = x; out[5] = e - 1.0; out[6] = emdd; out[7] = n


def window_masks(seg):
    return (seg >= 0, seg == 0, seg == 1)  # 전체(고르기+확인, 9/5 제외), 고르기, 확인


def n_days(base: str, sym: str = "MU") -> int:
    tb, _ = tf_bars_off(load(sym, base), 60, "eth")
    return int(len(np.unique(tb.day.values[tb.day.values >= WIN[base]["warm"]])))


def null_shifts(base: str):
    D = n_days(base)
    return D, list(range(15, D - 15 + 1))


def pct(x):
    return float(np.expm1(x) * 100.0)
