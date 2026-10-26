"""갈래 I (2차): 조건부 반도체 한도(2026-10-10).

7종목 규칙(데스크 run_model)은 그대로 두고, 계기가 켜졌을 때만 '반도체+미분류 합계 한도'(현행 50%)를 더 낮춘다.
7월 쇼크는 반도체 안의 쇼크(SOXX −29%, 소프트웨어 상승)였으므로, 비반도체 종목은 건드리지 않고 반도체 몫만 줄이는 손잡이다.

엔진: build_base.book_2026()/book_hold() + desk.strategies(weekly_target·project_caps·_drift)로 run_model과 같은 루프.
      desk/ 는 import만 한다. config.RULE['semis_cap']은 project_caps 호출 동안만 바꾸고 try/finally로 되돌린다.
      계기가 꺼져 있으면 desk run_model과 1e-9로 같아야 한다(check_engine, 실행 때마다 먼저 확인).
시점: 계기는 t일 종가 → t+1 시가(H.at_open). 주간 모드는 재조정일(ISO 주 첫 거래일) 시가의 신호 = 직전 거래일 종가 신호.

격자(실행 전 고정, 4×4×2×2 = 64)
  계기   S1 SOXX 종가 < SMA50 / S2 SOXX 종가 < SMA20 / S3 SOXX 종가 ≤ 20일 최고 종가 × 0.90 / S4 HYG/IEF(Adj Close) < 그 SMA50
  한도   켜졌을 때 반도체+미분류 합계 {0, 0.15, 0.25, 0.35}
  넘친 몫 재배분 = project_caps 그대로(semis_cap만 바꿈: 비반도체 고른 종목이 raw 비례로 25% 종목·60% 섹터 한도까지 받고 남으면 현금)
          현금   = 현행 목표(50%)에서 반도체 종목만 비례 축소, 비반도체 비중은 그대로
  적용   주간 = 재조정일에만(그 주 신호). 주중에는 기본처럼 비중이 흘러간다.
         즉시 = 재조정일은 주간과 같고, 주중에 신호가 꺼짐→켜짐으로 바뀐 첫 시가에 반도체 합계를 한도로 깎는다
                (재배분이면 깎은 몫을 그 주 비반도체 고른 종목에 raw 비례로 현재 비중에서 25%·60% 한도까지 더하고 나머지 현금).
                켜진 동안 다시 깎지 않고(주간 규칙처럼 흘러가게 둔다), 꺼지면 다음 재조정일까지 그대로 둔다.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

import harness as H
import build_base as BB                    # ROOT를 sys.path에 넣고 desk를 불러온다
from desk import config, strategies as S   # noqa: E402  (import만, 수정 금지)

FAMILY = "I_semicap"
SIGNALS = {"S1": "SOXX<SMA50", "S2": "SOXX<SMA20", "S3": "SOXX≤20일고점×0.90", "S4": "HYG/IEF<SMA50"}
CAPS = [0.0, 0.15, 0.25, 0.35]
OVERFLOWS = ["재배분", "현금"]
TIMINGS = ["주간", "즉시"]
GRID_DESC = ("계기 4(S1 SOXX<SMA50, S2 SOXX<SMA20, S3 SOXX 20일 최고 종가 대비 ≤−10%, S4 HYG/IEF Adj<SMA50) × "
             "켜졌을 때 반도체+미분류 한도 4{0,0.15,0.25,0.35} × 넘친 몫 2{project_caps 재배분, 현금} × "
             "적용 2{주간 재조정일만, 켜진 첫 시가 즉시 깎고 다음 재조정일까지 유지} = 64. 실행 전 고정, 결과 본 뒤 추가 없음.")
AISHOCK = ("2026-06-22", "2026-07-29")
POST_REFS = ["S2 SOXX<SMA20 한도0% 현금 주간", "S2 SOXX<SMA20 한도0% 재배분 주간", "S2 SOXX<SMA20 한도0% 현금 즉시",
             "S2 SOXX<SMA20 한도15% 현금 주간", "S4 HYG/IEF<SMA50 한도15% 현금 즉시", "S4 HYG/IEF<SMA50 한도15% 재배분 즉시"]
FEBMAR = [("2026-02-03", "2026-02-05"), ("2026-02-06", "2026-02-24"), ("2026-02-25", "2026-03-06"),
          ("2026-03-09", "2026-03-18"), ("2026-03-19", "2026-03-30"), ("2026-03-31", "2026-04-14"),
          ("2026-01-26", "2026-04-30")]
SNAP_DAYS = ["2026-06-18", "2026-06-22", "2026-06-23", "2026-06-29", "2026-07-06", "2026-07-08", "2026-07-09",
             "2026-07-13", "2026-07-15", "2026-07-17", "2026-07-20", "2026-07-27", "2026-07-29", "2026-08-03",
             "2026-08-10"]
SNAP_FEB = ["2026-02-02", "2026-02-03", "2026-02-06", "2026-02-09", "2026-02-23", "2026-02-25", "2026-03-02",
            "2026-03-06", "2026-03-09", "2026-03-16", "2026-03-19", "2026-03-23", "2026-03-30", "2026-04-06"]


@contextmanager
def semis_cap(c: float):
    saved = dict(config.RULE)
    try:
        config.RULE["semis_cap"] = c
        yield
    finally:
        config.RULE.clear()
        config.RULE.update(saved)


# ----------------------------------------------------------------------------- 계기(t일 종가)
def raw_signals() -> dict:
    c = H.px("SOXX", "Close")
    r = (H.px("HYG", "Adj Close") / H.px("IEF", "Adj Close")).dropna()
    s = {"S1": (c < c.rolling(50).mean()) & c.rolling(50).mean().notna(),
         "S2": (c < c.rolling(20).mean()) & c.rolling(20).mean().notna(),
         "S3": (c / c.rolling(20).max() - 1) <= -0.10,
         "S4": (r < r.rolling(50).mean()) & r.rolling(50).mean().notna()}
    return {k: v.astype(float) for k, v in s.items()}


def signals_at_open(bn: str) -> dict:
    b = H.base(bn)
    return {k: (H.at_open(v, b, fill=0.0) > 0.5).to_numpy(bool) for k, v in raw_signals().items()}


# ----------------------------------------------------------------------------- 엔진
class Engine:
    def __init__(self, bn: str):
        self.bn = bn
        self.b = H.base(bn)
        self.book = BB.book_2026() if bn == "2026" else BB.book_hold()
        bk, days = self.book, self.b.days
        self.k0 = bk.days.index(days[0])
        if bk.days[self.k0:self.k0 + len(days)] != days:
            raise ValueError("book 거래일과 기본 경로 거래일이 다르다")
        self.rule = dict(config.RULE)
        # 재조정 결정은 비중과 무관(선택은 순위·버퍼만 본다) → 한 번만 계산(run_model과 같은 조건)
        self.dec = {}
        last, selected = {}, set()
        for i, day in enumerate(days):
            k = self.k0 + i
            if not last or S.iso_week(day) != S.iso_week(bk.days[k - 1]):
                d = S.weekly_target(bk, k, day, selected)
                if d["ranked"]:
                    selected, last = set(d["selected"]), d
                    self.dec[i] = {"raw": d["raw"], "target": d["target"], "selected": list(d["selected"])}
        keys = {key for d in self.dec.values() for key in d["raw"]}
        self.O = {key: S._col(bk, bk.opn, key) for key in keys}
        self.semis = {key for key in keys if S.in_semis_budget(bk.sector[key])}
        self.sector = bk.sector

    def rets(self, w: dict, k: int) -> dict:
        out = {}
        for key in w:
            o = self.O[key]
            out[key] = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                and o[k - 1] > 0 else 0.0
        return out

    def ssum(self, w: dict) -> float:
        return sum(v for key, v in w.items() if key in self.semis)

    def scale_semis(self, w: dict, c: float) -> dict:
        s = self.ssum(w)
        if s <= c + 1e-12:
            return dict(w)
        f = c / s
        out = {}
        for key, v in w.items():
            nv = v * f if key in self.semis else v
            if nv > 1e-12:
                out[key] = nv
        return out

    def refill(self, w: dict, raw: dict, add: float) -> dict:
        """주중 즉시 재배분: 그 주 비반도체 고른 종목이 현재 비중에서 raw 비례로 25% 종목·60% 섹터 한도까지 받는다."""
        stock, detail = self.rule["stock_cap"], self.rule["sector_cap"]
        names = [key for key, v in raw.items() if v > 0 and key not in self.semis]
        alloc = dict(w)
        left = add
        for _ in range(4 * len(names) + 8):
            if left <= 1e-12:
                break
            grp: dict = {}
            for key, v in alloc.items():
                grp[self.sector[key]] = grp.get(self.sector[key], 0.0) + v
            active = [key for key in names if stock - alloc.get(key, 0.0) > 1e-12
                      and detail - grp.get(self.sector[key], 0.0) > 1e-12]
            if not active:
                break
            rate = sum(raw[key] for key in active)
            steps = [left / rate] + [(stock - alloc.get(key, 0.0)) / raw[key] for key in active]
            for g in {self.sector[key] for key in active}:
                steps.append((detail - grp.get(g, 0.0)) / sum(raw[key] for key in active if self.sector[key] == g))
            step = max(0.0, min(steps))
            for key in active:
                alloc[key] = alloc.get(key, 0.0) + step * raw[key]
            left -= step * rate
        return alloc

    def run(self, sig=None, cap=None, overflow=None, timing=None):
        """sig = 거래일별 bool 배열(그날 시가에 쓸 계기). None이면 현행 규칙 그대로."""
        days = self.b.days
        w: dict = {}
        raw: dict = {}
        value, weights, cuts, turn = [1.0], [], [], []
        for i, day in enumerate(days):
            k = self.k0 + i
            if i > 0:
                w, gross = S._drift(w, self.rets(w, k))
                value.append(value[-1] * (1 + gross))
            w_pre = dict(w)
            on = sig is not None and bool(sig[i])
            if i in self.dec:
                d = self.dec[i]
                raw = d["raw"]
                if on and overflow == "재배분":
                    with semis_cap(cap):
                        w = S.project_caps(raw, self.sector)
                elif on:
                    w = self.scale_semis(d["target"], cap)
                else:
                    w = dict(d["target"])
            elif on and timing == "즉시" and not bool(sig[i - 1]) and self.ssum(w) > cap + 1e-9:
                before = sum(w.values())
                w2 = self.scale_semis(w, cap)
                if overflow == "재배분":
                    w2 = self.refill(w2, raw, before - sum(w2.values()))
                cuts.append((day, round(self.ssum(w), 4), round(sum(w2.values()), 4)))
                w = w2
            weights.append(dict(w))
            turn.append(sum(abs(w.get(key, 0.0) - w_pre.get(key, 0.0)) for key in set(w) | set(w_pre)))
        v = pd.Series(value, index=days)
        r = (v.shift(-1) / v - 1).dropna()
        self.turn = pd.Series(turn, index=days)      # 그날 시가 거래의 회전율(편도 합)
        return r, weights, cuts


def check_engine(eng: Engine) -> dict:
    """계기 끈 엔진 = desk run_model(현행 RULE) = 기본 경로 json. 최대 차를 돌려준다."""
    r, wts, _ = eng.run()
    path = S.run_model(eng.book, start=eng.b.days[0])
    n = len(eng.b.days)
    v_mine = np.concatenate([[1.0], (1 + r.to_numpy()).cumprod()])
    d_model = float(np.max(np.abs(v_mine - np.asarray(path.value[:n]))))
    d_base = float(np.max(np.abs(r.to_numpy() - eng.b.ret.to_numpy())))
    d_w = max(abs(wts[i].get(key, 0.0) - path.weights[i].get(key, 0.0))
              for i in range(n) for key in set(wts[i]) | set(path.weights[i]))
    return {"value_vs_run_model": d_model, "ret_vs_base_json": d_base, "weights_vs_run_model": d_w}


# ----------------------------------------------------------------------------- 보고용
def comp(eng: Engine, w: dict) -> dict:
    s = float(eng.ssum(w))
    inv = float(sum(w.values()))
    return {"반도체": round(s, 4), "비반도체": round(inv - s, 4), "현금": round(1 - inv, 4)}


def with_spread(r: pd.Series, turn: pd.Series, cost: float = 0.001) -> pd.Series:
    """k일 시가 거래 회전율 × cost를 그날 시가 자산에서 뺀다(재조정·깎음 모두 포함, 기본 경로도 같은 방식)."""
    t = turn.reindex(r.index).fillna(0.0)
    return (1 + r) * (1 - cost * t) - 1


def rel_exposure(wv: list, wb: list, days: list) -> pd.Series:
    inv_v = np.array([sum(x.values()) for x in wv])
    inv_b = np.array([sum(x.values()) for x in wb])
    e = np.where(inv_b > 1e-9, inv_v / np.where(inv_b > 1e-9, inv_b, 1.0), 1.0)
    return pd.Series(e, index=days)


def spells(sig: np.ndarray, days: list, lo: str, hi: str) -> list:
    out, start = [], None
    for i, d in enumerate(days):
        if sig[i] and start is None:
            start = d
        if start is not None and (not sig[i] or i == len(days) - 1):
            end = days[i - 1] if not sig[i] else d
            if end >= lo and start <= hi:
                out.append(f"{start}~{end}")
            start = None
    return out


def window_ret(r: pd.Series, lo: str, hi: str) -> float:
    return H.stats(r, lo, hi)["수익"]


def trough(r: pd.Series, lo: str, hi: str) -> dict:
    """구간 안 최대 낙폭의 고점·저점(시가 날짜: 그 날 시가의 자산)."""
    x = r[(r.index >= lo) & (r.index <= hi)]
    c = np.concatenate([[1.0], (1 + x.to_numpy()).cumprod()])
    lab = list(x.index) + ["(구간 끝 다음 시가)"]          # c[j] = lab[j] 시가의 자산
    dd = c / np.maximum.accumulate(c) - 1
    j = int(dd.argmin())
    p = int(np.argmax(c[:j + 1]))
    return {"고점": lab[p], "저점": lab[j], "낙폭": float(dd[j])}


def fixed_cmp(bn: str, e: float) -> dict:
    b = H.base(bn)
    r = H.overlay(b, pd.Series(e, index=b.days))
    return {k: H.stats(r, *v) for k, v in b.periods.items()}


def main():
    saved_rule = dict(config.RULE)
    engines = {bn: Engine(bn) for bn in ("2026", "hold")}
    for bn, eng in engines.items():
        if eng.rule != json.loads((H.CACHE / f"{H.FILES[bn]}.json").read_text(encoding="utf-8"))["rule"]:
            raise SystemExit(f"{bn}: config.RULE이 기본 경로 규칙과 다르다")
    chk = {bn: check_engine(eng) for bn, eng in engines.items()}
    print("엔진 확인(계기 끔 vs desk run_model):", json.dumps(chk, ensure_ascii=False))
    for bn, c in chk.items():
        if max(c.values()) > 1e-9:
            raise SystemExit(f"{bn}: 엔진이 run_model과 다르다 {c}")
    sigs = {bn: signals_at_open(bn) for bn in engines}
    base_runs, base_turn = {}, {}
    for bn, eng in engines.items():
        base_runs[bn] = eng.run()
        base_turn[bn] = eng.turn

    rows, store = [], {}
    for s in SIGNALS:
        for cap in CAPS:
            for ov in OVERFLOWS:
                for tm in TIMINGS:
                    name = f"{s} {SIGNALS[s]} 한도{cap:.0%} {ov} {tm}"
                    params = {"signal": s, "cap": cap, "overflow": ov, "timing": tm}
                    rr, ex, out = {}, {}, {}
                    for bn, eng in engines.items():
                        r, wts, cuts = eng.run(sigs[bn][s], cap, ov, tm)
                        rr[bn] = r
                        ex[bn] = rel_exposure(wts, base_runs[bn][1], eng.b.days)
                        out[bn] = (r, wts, cuts, eng.turn)
                    row = H.evaluate_returns(name, rr, params, ex)
                    for bn in engines:
                        b_ = engines[bn].b
                        row[bn]["전체_스프레드0.1%(회전율)"] = H.stats(with_spread(out[bn][0], out[bn][3]),
                                                                    *b_.periods["전체"])["수익"]
                        row[bn]["계기켠날"] = int(sigs[bn][s][:-1].sum())
                        row[bn]["주중깎음"] = len(out[bn][2])
                        sem = np.array([engines[bn].ssum(x) for x in out[bn][1]])[:-1]
                        row[bn]["평균반도체"] = float(sem.mean())
                    rows.append(row)
                    store[name] = out

    # 기본 경로 참고 통계
    b26 = engines["2026"]
    sem_b = np.array([b26.ssum(x) for x in base_runs["2026"][1]])[:-1]
    print(f"기본 2026 평균 반도체 {sem_b.mean():.3f}, 평균 투자 {np.mean([sum(x.values()) for x in base_runs['2026'][1]][:-1]):.3f}")

    # 계기 켠 구간(2026)
    sig_spells = {s: spells(sigs["2026"][s], b26.b.days, "2026-01-01", "2026-12-31") for s in SIGNALS}
    sig_days = {s: {"2026": int(sigs["2026"][s][:-1].sum()), "hold": int(sigs["hold"][s][:-1].sum())} for s in SIGNALS}
    for s in SIGNALS:
        print(s, SIGNALS[s], "켠 날", sig_days[s], "2026 구간", sig_spells[s])

    # 고르기 전 미리 저장할 참고: 앞 고르기 후보와 사후 관심 변형
    br = H.base_result()
    picks = {k: H.select_front(rows, br, kk) for k, kk in (("keep90", 0.9), ("keep80", 0.8), ("keep70", 0.7))}
    byname = {r["name"]: r for r in rows}
    focus = [v["name"] for v in picks.values() if v]
    # 사후 참고(1차 실행 결과를 본 뒤 정함, 선택편향 있음): 앞 낙폭 동률(-14.15%) 6개 중 다른 계기 대표와 그 변형들
    focus += POST_REFS
    focus = list(dict.fromkeys(focus))
    bw, bd = 1 + br["2026"]["앞"]["수익"], br["2026"]["앞"]["낙폭"]
    front_pass = [r for r in rows if 1 + r["2026"]["앞"]["수익"] >= 0.9 * bw and r["2026"]["앞"]["낙폭"] >= bd + 0.01]
    front_pass.sort(key=lambda r: (-round(r["2026"]["앞"]["낙폭"], 6), -r["2026"]["앞"]["수익"]))
    bb = br["2026"]["뒤"]
    fp_table = [{"name": r["name"], "앞": r["2026"]["앞"], "뒤": r["2026"]["뒤"], "AI쇼크": r["2026"]["AI쇼크"]["수익"],
                 "2025": r["hold"]["2025"],
                 "뒤_둘다개선": bool(r["2026"]["뒤"]["수익"] > bb["수익"] and r["2026"]["뒤"]["낙폭"] > bb["낙폭"])}
                for r in front_pass]
    print(f"앞 keep90 통과 {len(front_pass)}개, 그중 뒤 구간 수익·낙폭 모두 개선 {sum(x['뒤_둘다개선'] for x in fp_table)}개")
    cnt = {"전체64_뒤둘다개선": sum(1 for r in rows if r["2026"]["뒤"]["수익"] > bb["수익"] and r["2026"]["뒤"]["낙폭"] > bb["낙폭"]),
           "전체64_AI쇼크개선": sum(1 for r in rows if r["2026"]["AI쇼크"]["수익"] > br["2026"]["AI쇼크"]["수익"]),
           "전체64_2025둘다개선": sum(1 for r in rows if r["hold"]["2025"]["수익"] > br["hold"]["2025"]["수익"]
                                  and r["hold"]["2025"]["낙폭"] > br["hold"]["2025"]["낙폭"])}
    print(cnt)
    rb_cost = {bn: H.stats(with_spread(base_runs[bn][0], base_turn[bn]), *engines[bn].b.periods["전체"])["수익"]
               for bn in engines}
    print("기본 스프레드0.1%(회전율) 전체:", rb_cost)

    rb26 = base_runs["2026"][0]
    detail = {}
    for nm in focus:
        r26, wts, cuts, _ = store[nm]["2026"]
        rh, wh, cuth, _ = store[nm]["hold"]
        days = b26.b.days
        e26 = byname[nm]["2026"]["평균노출"]
        eh = byname[nm]["hold"]["평균노출"]
        snap = {d: {"변형": comp(b26, wts[days.index(d)]), "기본": comp(b26, base_runs["2026"][1][days.index(d)]),
                    "계기": bool(sigs["2026"][byname[nm]["params"]["signal"]][days.index(d)])} for d in SNAP_DAYS}
        snapf = {d: {"변형": comp(b26, wts[days.index(d)]), "기본": comp(b26, base_runs["2026"][1][days.index(d)]),
                     "계기": bool(sigs["2026"][byname[nm]["params"]["signal"]][days.index(d)])} for d in SNAP_FEB}
        lo, hi = AISHOCK
        i0, i1 = days.index(lo), days.index(hi)
        shock_avg = {"변형": {k: float(np.mean([comp(b26, wts[i])[k] for i in range(i0, i1 + 1)]))
                             for k in ("반도체", "비반도체", "현금")},
                     "기본": {k: float(np.mean([comp(b26, base_runs["2026"][1][i])[k] for i in range(i0, i1 + 1)]))
                             for k in ("반도체", "비반도체", "현금")}}
        febmar = {f"{a}~{c}": {"변형": window_ret(r26, a, c), "기본": window_ret(rb26, a, c)} for a, c in FEBMAR}
        detail[nm] = {"params": byname[nm]["params"], "2026_평균노출(기본 대비 투자 비율)": e26, "hold_평균노출": eh,
                      "fixed_same_exposure_2026": {k: fixed_cmp("2026", e26)[k] for k in ("전체", "앞", "뒤", "AI쇼크", "7~9월")},
                      "fixed_same_exposure_hold": {k: fixed_cmp("hold", eh)[k] for k in ("2025", "2024", "관세", "24여름", "25가을")},
                      "aishock_snap": snap, "aishock_avg_comp": shock_avg, "febmar_snap": snapf, "febmar_windows": febmar,
                      "cuts_2026": cuts, "cuts_hold_n": len(cuth),
                      "front_trough": trough(r26, "2026-01-02", "2026-05-29"),
                      "back_trough": trough(r26, "2026-06-01", "2026-10-09"),
                      "spread0.1%_2026전체": byname[nm]["2026"]["전체_스프레드0.1%(회전율)"],
                      "spread0.1%_hold전체": byname[nm]["hold"]["전체_스프레드0.1%(회전율)"]}
        print("\n==", nm)
        print(" 평균노출(기본 대비) 2026", round(e26, 3), "hold", round(eh, 3), "주중깎음", len(cuts))
        print(" 쇼크 평균 구성", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in shock_avg.items()})
        for d, x in snap.items():
            print("  ", d, "계기" if x["계기"] else "    ", "변형", x["변형"], "기본", x["기본"])
        for wnd, x in febmar.items():
            print("   2~3월", wnd, f"변형 {x['변형']:+.1%} 기본 {x['기본']:+.1%}")
        fx = detail[nm]["fixed_same_exposure_2026"]
        print(" 같은 노출 고정", {k: f"{v['수익']:+.1%}/{v['낙폭']:.1%}" for k, v in fx.items()})

    # 기본 경로 7월 그룹 기여(반도체 묶음 vs 비반도체)
    days = b26.b.days
    lo, hi = AISHOCK
    i0, i1 = days.index(lo), days.index(hi) + 1
    gs, gn = 1.0, 1.0
    cs, cn = 0.0, 0.0
    w_prev = base_runs["2026"][1]
    for i in range(i0, i1):
        w = w_prev[i]
        rets = b26.rets(w, b26.k0 + i + 1)
        ws = sum(v for k, v in w.items() if k in b26.semis)
        wn = sum(v for k, v in w.items() if k not in b26.semis)
        rs = sum(v * rets[k] for k, v in w.items() if k in b26.semis)
        rn = sum(v * rets[k] for k, v in w.items() if k not in b26.semis)
        gs *= 1 + (rs / ws if ws > 0 else 0)
        gn *= 1 + (rn / wn if wn > 0 else 0)
        cs += rs
        cn += rn
    mkt = {sym: float(H.px(sym, "Open").loc[days[i1]] / H.px(sym, "Open").loc[lo] - 1) for sym in ("SOXX", "IGV", "QQQ", "DELL")}
    group = {"반도체 묶음 수익": gs - 1, "비반도체 묶음 수익": gn - 1, "반도체 기여 합(단순합)": cs, "비반도체 기여 합": cn,
             "시가 6/22→7/30": mkt}
    print("\n기본 7월 그룹:", json.dumps(group, ensure_ascii=False))

    hold_spells = {s: spells(sigs["hold"][s], engines["hold"].b.days, "2024-01-01", "2025-12-31") for s in SIGNALS}
    sem_h = np.array([engines["hold"].ssum(x) for x in base_runs["hold"][1]])[:-1]
    print(f"기본 hold 평균 반도체+미분류 {sem_h.mean():.3f}")
    print("기본 앞·뒤 바닥:", trough(rb26, "2026-01-02", "2026-05-29"), trough(rb26, "2026-06-01", "2026-10-09"))
    extra = {"engine_check": chk, "signal_days": sig_days, "signal_spells_2026_open": sig_spells,
             "signal_spells_hold_open_2024_25": hold_spells, "base_avg_semis_hold": float(sem_h.mean()),
             "front_pass_keep90": fp_table, "counts": cnt, "base_spread0.1%_전체": rb_cost,
             "base_front_trough": trough(rb26, "2026-01-02", "2026-05-29"),
             "base_back_trough": trough(rb26, "2026-06-01", "2026-10-09"),
             "base_avg_semis_2026": float(sem_b.mean()), "aishock_groups_base": group, "detail": detail,
             "row_extra": {r["name"]: {bn: {k: r[bn][k] for k in ("계기켠날", "주중깎음", "평균반도체")} for bn in ("2026", "hold")}
                           for r in rows}}
    payload = H.summarize_family(FAMILY, rows, GRID_DESC, extra)

    # 전 변형 짧은 표(정렬: 계기·한도·넘친 몫·적용)
    print("\n전 변형: 이름 | 2026 전체 | 앞 | 뒤 | AI쇼크 | 7~9월 | 2025 | 2024 | 노출 | 평균반도체")
    for r in rows:
        a, h = r["2026"], r["hold"]
        print(f"{r['name']:<44} {a['전체']['수익']:+.1%}/{a['전체']['낙폭']:.1%} {a['앞']['수익']:+.1%}/{a['앞']['낙폭']:.1%} "
              f"{a['뒤']['수익']:+.1%}/{a['뒤']['낙폭']:.1%} {a['AI쇼크']['수익']:+.1%} {a['7~9월']['수익']:+.1%} "
              f"{h['2025']['수익']:+.1%}/{h['2025']['낙폭']:.1%} {h['2024']['수익']:+.1%} {a['평균노출']:.2f} {a['평균반도체']:.2f}")
    if config.RULE != saved_rule:
        raise SystemExit("config.RULE이 되돌려지지 않았다")
    print("config.RULE 복구 확인:", config.RULE)
    return payload


if __name__ == "__main__":
    main()
