"""양봉 비율 3단계 — 시간봉 조합판(BRIEF3.md). 신호 시간봉 b × 구간 판정 시간봉 z.

- 신호 s: b분 연장 봉, N10 개수 따라가기 0.5/0.5(원안 규칙, 시간봉만 바꿈).
- 판정(z == b): ER20 ≥ 0.3, ADX14 ≥ 20, |p20 − 0.5| ≥ 0.1.  (z > b): 위 셋을 z봉에서 계산 + AGREE(z봉 원안 상태 == s).
  z ∈ {b, 2b, 4b, 8b (≤ 960분)} ∪ {일봉(정규장, 16:00 ET 마감 뒤 반영)}.
- 예열: 7/17 전은 yfinance 1시간봉(5/1~)으로 묶은 봉을 앞에 붙인다. 평가는 7/17부터의 5분봉 봉만(seg ≥ 0) 쓴다.
- 구간·행동·모의매매는 zones3와 같다(A0 현금, A2 동결, A3 원안). 보유가 0·1뿐이라 절반 재조정 회계 문제는 없다.
- 시각은 모두 ns 정수로 맞춘다.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from engine import CACHE, ET, WIN, awake_mask, load, ratio, segments, tf_bars
from zones3 import adx, daily_state, efficiency, signal_state, sim_frac, static_path, targets

N, B, S = 10, 0.5, 0.5
BASE_TFS = [10, 20, 30, 40, 60, 90, 120, 180, 240]
DEFAULT = (40, 160, "AGREE", "A0")
START5 = date(2026, 7, 17)
WARM1H = date(2026, 6, 1)


def zone_tfs(b):
    return [b] + [m * b for m in (2, 4, 8) if m * b <= 960] + ["D"]


def _ns(x):
    return pd.DatetimeIndex(x).as_unit("ns").asi8


def bars_with_warm(sym, tf):
    """7/17 전은 1시간봉 묶음(예열), 7/17부터는 5분봉 묶음. end = 실제 마지막 구성 봉 끝(ns)."""
    d5 = load(sym, "5m")
    h1 = load(sym, "1h")
    h1 = h1[(h1.index.date >= WARM1H) & (h1.index.date < START5)]
    parts = []
    for base, step in ((h1, None), (d5, 5)):
        if not len(base):
            continue
        tb = tf_bars(base, tf, "eth")
        # 구성 봉 끝: 5분봉은 시작+5분, 1시간봉은 시작+60분(09:00·15:30은 30분)
        t = base.index
        m = np.asarray(t.hour * 60 + t.minute)
        keep = (m >= 240) & (m < 1200)
        bt = t[keep]
        if step:
            ln = np.full(len(bt), 5)
        else:
            mm = m[keep]
            ln = np.where((mm == 540) | (mm == 930), 30, 60)
        bend = _ns(bt) + ln.astype(np.int64) * 60_000_000_000
        k = (m[keep] - 240) // tf
        g = pd.DataFrame({"d": np.asarray(bt.date), "k": k, "e": bend}).groupby(["d", "k"], sort=True).e.max()
        tb = tb.copy()
        tb["end"] = g.values
        tb["warm"] = step is None
        parts.append(tb)
    tb = pd.concat(parts, ignore_index=True)
    tb = tb[tb.day >= WARM1H].reset_index(drop=True)
    return tb


def z_indicators(tb):
    O, H, L, C, V = (tb[k].values.astype(np.float64) for k in ("O", "H", "L", "C", "V"))
    p10 = ratio(O, C, V, N, "count")
    p20 = ratio(O, C, V, 20, "count")
    return dict(state=signal_state(p10, B, S), er=efficiency(C, 20), adx=adx(H, L, C, 14), p20=p20, end=tb.end.values.astype(np.int64))


def map_last(z_end, z_val, base_end):
    idx = np.searchsorted(z_end, base_end, side="right") - 1
    v = z_val.astype(float)
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)


def daily_values(sym):
    d1 = pd.read_parquet(CACHE / f"{sym}_1d.parquet")
    d1 = d1[pd.DatetimeIndex(d1.index).date <= date(2026, 10, 22)]
    O, H, L, C = (d1[k].values.astype(np.float64) for k in ("Open", "High", "Low", "Close"))
    days = pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(d1.index).date)).tz_localize(ET)
    end = _ns(days + pd.Timedelta(minutes=960))
    return dict(state=signal_state(ratio(O, C, np.ones(len(C)), N, "count"), B, S).astype(float), er=efficiency(C, 20),
                adx=adx(H, L, C, 14), p20=ratio(O, C, np.ones(len(C)), 20, "count"), end=end)


def prepare(sym, b):
    tb = bars_with_warm(sym, b)
    O, H, L, C, V = (tb[k].values.astype(np.float64) for k in ("O", "H", "L", "C", "V"))
    base = z_indicators(tb)
    s = base["state"]
    flags = {}
    dly = None
    for z in zone_tfs(b):
        if z == b:
            flags[f"z{b}_ER"] = base["er"] >= 0.3
            flags[f"z{b}_ADX"] = base["adx"] >= 20
            flags[f"z{b}_EXT"] = np.abs(base["p20"] - 0.5) >= 0.1 - 1e-12
            continue
        if z == "D":
            zi = dly = daily_values(sym)
        else:
            zi = z_indicators(bars_with_warm(sym, z))
        st = map_last(zi["end"], zi["state"], base["end"])
        flags[f"z{z}_AGREE"] = st == s
        flags[f"z{z}_ER"] = map_last(zi["end"], zi["er"], base["end"]) >= 0.3
        flags[f"z{z}_ADX"] = map_last(zi["end"], zi["adx"], base["end"]) >= 20
        flags[f"z{z}_EXT"] = np.abs(map_last(zi["end"], zi["p20"], base["end"]) - 0.5) >= 0.1 - 1e-12
    for k in flags:
        flags[k] = np.nan_to_num(flags[k].astype(float), nan=0.0).astype(bool)
    seg = segments(tb.day.values, WIN["5m"])
    seg[tb.warm.values] = -1
    return dict(sym=sym, b=b, tb=tb, O=O, C=C, s=s, flags=flags, seg=seg, awake=awake_mask(tb.start))


def run(P, cost, ideal, flags=None):
    """전략: 원안(A3) + 판정마다 A0·A2. 같은 비중 그냥 보유와 보유를 함께 낸다."""
    flags = flags if flags is not None else P["flags"]
    O, C, seg = P["O"], P["C"], P["seg"]
    aw = np.ones(len(O), np.bool_) if ideal else P["awake"]
    pos = np.zeros(len(O))
    rows = []
    strategies = [("BASE", None, 3)] + [(nm, nm, ai) for nm in flags for ai in (0, 2)]
    for nm, fl, ai in strategies:
        flag = flags[fl] if fl else np.zeros(len(O), np.bool_)
        out = sim_frac(O, C, targets(P["s"], flag, ai), seg, aw, cost, pos)
        r = dict(b=P["b"], clf=nm, action={0: "A0", 2: "A2", 3: "A3"}[ai])
        for w, k in (("sel", 0), ("conf", 1)):
            x = out[2 + k] / out[4 + k]
            st, st_mdd, bh, bh_mdd = static_path(O, C, seg, x, k)
            r.update({f"ret_{w}": float(np.expm1(out[k])), f"mdd_{w}": float(np.expm1(out[6 + k])), f"x_{w}": x,
                      f"static_{w}": st, f"static_mdd_{w}": st_mdd, f"bh_{w}": bh, f"turn_{w}": out[9 + k]})
        rows.append(r)
    return rows


def diag(P, ideal, flags=None):
    """원안(비용 0) 봉별 보유로 판정마다 '추세 봉'·'애매 봉'의 노출 맞춘 초과(합, 로그 근사 아닌 단순 합)."""
    flags = flags if flags is not None else P["flags"]
    O, C, seg = P["O"], P["C"], P["seg"]
    aw = np.ones(len(O), np.bool_) if ideal else P["awake"]
    pos = np.zeros(len(O))
    sim_frac(O, C, targets(P["s"], np.zeros(len(O), np.bool_), 3), seg, aw, 0.0, pos)
    r = np.r_[O[1:] / O[:-1] - 1, C[-1] / O[-1] - 1]
    out = []
    for nm, fl in flags.items():
        z = np.r_[False, fl[:-1]]  # 그 봉에 걸린 판정 = 직전 봉 마감의 flag
        row = dict(b=P["b"], clf=nm)
        for w, k in (("sel", 0), ("conf", 1)):
            for zn, m in (("trend", z), ("amb", ~z)):
                mm = m & (seg == k)
                n = int(mm.sum())
                if n == 0:
                    row[f"{zn}_{w}"] = np.nan
                    row[f"{zn}_{w}_n"] = 0
                    continue
                xz = pos[mm].mean()
                row[f"{zn}_{w}"] = float((pos[mm] * r[mm]).sum() - xz * r[mm].sum())
                row[f"{zn}_{w}_n"] = n
        out.append(row)
    return out
