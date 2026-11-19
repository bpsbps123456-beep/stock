"""갈래 exit_ma 격자 실행: TRAIN만으로 상위 3개 선택 → TEST·PRE 그대로 기록.

실행: py exit_ma_run.py  → results/exit_ma.json, results/exit_ma_grid.csv
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import exit_ma_lib as L  # noqa: E402

t0 = time.time()
b3 = H.resample(3)
RALLY = H.to_1m_mask(H.ctx_rally(b3, 20, 0.015), b3)
DECLINE = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)
BARS = {1: H.resample(1), 3: b3}

_ind: dict = {}


def E(tf, n):
    k = ("ema", tf, n)
    if k not in _ind:
        _ind[k] = H.ema(BARS[tf]["Close"], n)
    return _ind[k]


def ATR(tf, n=14):
    k = ("atr", tf, n)
    if k not in _ind:
        _ind[k] = H.atr(BARS[tf], n)
    return _ind[k]


# ───────────── 격자 ─────────────
CONFIGS = []   # (name, family, tf, params, builder) — builder() -> ('state'|'event', sig Series)


def add(name, family, tf, params, kind, fn):
    CONFIGS.append(dict(name=name, family=family, tf=tf, params=params, kind=kind, fn=fn))


for tf in (1, 3):
    b = BARS[tf]
    c = b["Close"]
    # 1) 종가 < EMA n : 상태형 / 하향 교차 사건형   (5선 × 2형 × 2봉 = 20)
    for n in (10, 25, 60, 120, 200):
        add(f"ema{n}_below_state_{tf}m", "ema_below", tf, dict(n=n), "state",
            lambda c=c, tf=tf, n=n: c < E(tf, n))
        add(f"ema{n}_cross_event_{tf}m", "ema_cross", tf, dict(n=n), "event",
            lambda c=c, tf=tf, n=n: (c < E(tf, n)) & (c.shift(1) >= E(tf, n).shift(1)))
    # 2) 이탈 지속 N봉 (3선 × N2 × 2봉 = 12)
    for n in (10, 25, 60):
        for N in (3, 5):
            add(f"ema{n}_below{N}bars_{tf}m", "ema_persist", tf, dict(n=n, N=N), "state",
                lambda c=c, tf=tf, n=n, N=N: L.runlen(c < E(tf, n)) >= N)
    # 3) 여러 선 동시 이탈·역배열 (4 × 2봉 = 8)
    add(f"below_all3_{tf}m", "ema_multi", tf, {}, "state",
        lambda c=c, tf=tf: (c < E(tf, 10)) & (c < E(tf, 25)) & (c < E(tf, 60)))
    add(f"bear_order_state_{tf}m", "ema_order", tf, {}, "state",
        lambda tf=tf: (E(tf, 10) < E(tf, 25)) & (E(tf, 25) < E(tf, 60)))
    add(f"e10x25_down_event_{tf}m", "ema_order", tf, {}, "event",
        lambda tf=tf: (E(tf, 10) < E(tf, 25)) & (E(tf, 10).shift(1) >= E(tf, 25).shift(1)))
    add(f"e25x60_down_event_{tf}m", "ema_order", tf, {}, "event",
        lambda tf=tf: (E(tf, 25) < E(tf, 60)) & (E(tf, 25).shift(1) >= E(tf, 60).shift(1)))
    # 4) 기울기 꺾임 (2선 × 2봉 = 4): EMA가 직전 봉보다 낮아지는 순간
    for n in (25, 60):
        add(f"ema{n}_slope_down_event_{tf}m", "ema_slope", tf, dict(n=n), "event",
            lambda tf=tf, n=n: (E(tf, n) < E(tf, n).shift(1)) & (E(tf, n).shift(1) >= E(tf, n).shift(2)))
    # 5) 이탈 뒤 되돌림 실패 (2선 × 2봉 = 4): 최근 20봉 안 하향 교차가 있었고, 이번 봉 고가가 EMA에 닿았지만 종가는 아래
    for n in (25, 60):
        def _retest(c=c, b=b, tf=tf, n=n):
            e = E(tf, n)
            cross = (c < e) & (c.shift(1) >= e.shift(1))
            recent = cross.shift(1, fill_value=False).astype(int).rolling(20, min_periods=1).max() > 0
            return recent & (b["High"] >= e) & (c < e)
        add(f"ema{n}_retest_fail_event_{tf}m", "ema_retest", tf, dict(n=n, M=20), "event", _retest)
    # 6) ATR 샹들리에 (N2 × k3 × 2봉 = 12)
    for N in (22, 60):
        for k in (2.0, 3.0, 4.0):
            add(f"chand_N{N}_k{k:g}_{tf}m", "chandelier", tf, dict(N=N, k=k, atr=14), "state",
                lambda b=b, c=c, tf=tf, N=N, k=k: c < b["High"].rolling(N).max() - k * ATR(tf))
    # 7) 수퍼트렌드 하락 상태 (f3 × 2봉 = 6)
    for fct in (2.0, 3.0, 4.0):
        add(f"supertrend_f{fct:g}_{tf}m", "supertrend", tf, dict(factor=fct, atr=10), "state",
            lambda b=b, fct=fct: L.supertrend_dir(b, fct, 10) == 1)
    # 8) 장중(09:30~) 고점 대비 −x% (x4 × 2봉 = 8)
    for x in (0.02, 0.03, 0.05, 0.08):
        add(f"dayhigh_trail_{int(x*100)}pct_{tf}m", "dayhigh_trail", tf, dict(x=x), "state",
            lambda b=b, c=c, x=x: c <= L.reg_cummax_high(b) * (1 - x))
    # 10) 진단 뒤 추가(1봉 기울기 꺾임은 종가 교차와 수학적으로 같아서, 그리고 '계속 빠지는' 이탈을 가리려고)
    #     2선 × 3형 × 2봉 = 12: 5봉 기울기 꺾임 사건, RSI14>=45일 때만의 하향 교차, 선 아래이면서 선이 5봉 전보다 낮은 상태
    for n in (25, 60):
        add(f"ema{n}_slope5_down_event_{tf}m", "ema_slope5", tf, dict(n=n, k=5), "event",
            lambda tf=tf, n=n: (E(tf, n) < E(tf, n).shift(5)) & (E(tf, n).shift(1) >= E(tf, n).shift(6)))
        add(f"ema{n}_cross_rsi45_event_{tf}m", "ema_cross_rsi", tf, dict(n=n, rsi_min=45), "event",
            lambda c=c, tf=tf, n=n: (c < E(tf, n)) & (c.shift(1) >= E(tf, n).shift(1)) & (H.rsi(c, 14) >= 45))
        add(f"ema{n}_below_falling_state_{tf}m", "ema_below_falling", tf, dict(n=n, k=5), "state",
            lambda c=c, tf=tf, n=n: (c < E(tf, n)) & (E(tf, n) < E(tf, n).shift(5)))
    # 9) 진입 뒤 최고가 대비 −x% (x4 × 2봉 = 8)
    for x in (0.01, 0.02, 0.03, 0.05):
        add(f"entry_trail_{int(x*100)}pct_{tf}m", "entry_trail", tf, dict(x=x), "entry", None)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def build(cfg):
    if cfg["kind"] == "entry":
        return None, None
    sig = cfg["fn"]().fillna(False).astype(bool)
    b = BARS[cfg["tf"]]
    if cfg["kind"] == "state":
        sp, ep = L.state_and_event(sig, b)
        return sp, ep
    ep = H.exec_index(sig, b)
    ep = ep[ep >= 0]
    return ep, ep


def evaluate(cfg, rng, cache):
    """한 설정의 한 구간 성적."""
    if cfg["kind"] == "entry":
        x, tf = cfg["params"]["x"], cfg["tf"]
        hs, exs = L.entry_trail_sim(rng, x, tf)
        hr, _ = L.entry_trail_sim(rng, x, tf, entry_mask=RALLY)
        ev = H.event_metrics(H.dedup(np.unique(exs), 30))
    else:
        if "pos" not in cache:
            cache["pos"] = build(cfg)
        hp, ep = cache["pos"]
        hs = H.hold_sim(hp, rng)
        hr = H.hold_sim(hp, rng, entry_mask=RALLY)
        if "ev" not in cache:
            cache["ev"] = H.event_metrics(H.dedup(ep, 30))
        ev = cache["ev"]
    sm = H.summarize(ev, rng, "exit", RALLY)
    return dict(hold=hs, hold_rally=hr, summ=sm)


def row(cfg, r):
    h, hr, s = r["hold"], r["hold_rally"], r["summ"]
    return dict(name=cfg["name"], family=cfg["family"], tf=cfg["tf"], kind=cfg["kind"],
                x_rand=h["x_rand_mean%"], x_rand_t=h["x_rand_t_days"], exit_rate=h["exit_rate%"],
                hold_min=h["avg_hold_min"], rule_cvar5=h["rule_cvar5%"], rand_cvar5=h["rand_cvar5%"],
                x_rand_rally=hr["x_rand_mean%"], x_rand_rally_t=hr["x_rand_t_days"],
                n=s.get("n"), per_day=s.get("per_day"), r60_x_dm=s.get("r60_x_dm%"),
                rclose_x_dm=s.get("rclose_x_dm%"), r30_x_dm=s.get("r30_x_dm%"),
                fp_good=s.get("fp_good%"), fp_base=s.get("fp_good_base%"))


if __name__ == "__main__":
    # 재현 확인: 진입 기준 시뮬레이터가 harness.hold_sim과 같은 값을 내는지(상태형 EMA60 3분)
    sp, _ = build(next(c for c in CONFIGS if c["name"] == "ema60_below_state_3m"))
    ent = L._entries(H.TRAIN)
    xs = np.sort(sp)
    k = np.searchsorted(xs, ent + 1)
    ex = np.where(k < len(xs), xs[np.minimum(k, len(xs) - 1)], 10**12)
    a = L.hold_sim_from_exits(ent, ex)
    b_ = H.hold_sim(sp, H.TRAIN)
    assert a == b_, (a, b_)
    print("hold_sim 재현 일치", a["x_rand_mean%"])

    caches = {c["name"]: {} for c in CONFIGS}
    rows = []
    res_train = {}
    for cfg in CONFIGS:
        r = evaluate(cfg, H.TRAIN, caches[cfg["name"]])
        res_train[cfg["name"]] = r
        rows.append(row(cfg, r))
        print(f"{cfg['name']:34s} x_rand {r['hold']['x_rand_mean%']:+.3f} t {r['hold']['x_rand_t_days']}  "
              f"rally {r['hold_rally']['x_rand_mean%']:+.3f}  /day {r['summ'].get('per_day')}", flush=True)
    grid = pd.DataFrame(rows)
    out_dir = os.path.join(H.ROOT, "results")
    grid.to_csv(os.path.join(out_dir, "exit_ma_grid.csv"), index=False, encoding="utf-8-sig")

    # 선택 규칙(사전 고정): TRAIN 사건 빈도 0.3~6회/일(진입 기준 추적은 진입마다 탈출이라 빈도 조건 면제),
    # TRAIN x_rand_mean% 내림차순 상위 3.
    elig = grid[(grid.kind == "entry") | grid.per_day.between(0.3, 6.0)]
    top = elig.sort_values("x_rand", ascending=False).head(3)
    print(top[["name", "x_rand", "x_rand_t", "x_rand_rally", "per_day"]])

    result = dict(branch="exit_ma", configs_tested=len(CONFIGS), selection="TRAIN per_day 0.3~6 (entry 면제) 중 x_rand_mean% 상위 3",
                  grid_rows=clean(rows), top=[])
    for nm in list(top.name) + ["ema60_cross_event_3m", "ema60_below_state_3m"]:
        cfg = next(c for c in CONFIGS if c["name"] == nm)
        ent = dict(name=nm, family=cfg["family"], tf=cfg["tf"], kind=cfg["kind"], params=cfg["params"],
                   selected=nm in list(top.name), TRAIN=res_train[nm])
        for rng, lab in ((H.TEST, "TEST"), (H.PRE, "PRE")):
            ent[lab] = evaluate(cfg, rng, caches[nm])
        result["top"].append(ent)
        print(nm, "TEST", ent["TEST"]["hold"]["x_rand_mean%"], ent["TEST"]["hold"]["x_rand_t_days"],
              "PRE", ent["PRE"]["hold"]["x_rand_mean%"])
    with open(os.path.join(out_dir, "exit_ma.json"), "w", encoding="utf-8") as fh:
        json.dump(clean(result), fh, ensure_ascii=False, indent=1)
    print(round(time.time() - t0, 1), "s")
