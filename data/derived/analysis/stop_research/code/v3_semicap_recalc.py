"""v3 관점 1: 갈래 I 후보(S2 0% 현금 주간) 재계산·이웃·시점 반박 검토(2026-10-10).

후보: 재조정일(ISO 주 첫 거래일) 시가에 직전 거래일 종가 기준 SOXX 종가 < SMA20(종가 20일, t 포함)이면
      그 주 반도체+미분류 합계 한도 0(넘친 몫 현금, 비반도체는 현행 목표 그대로), 아니면 현행 50%.

엔진은 새로 짰다(v2_I_semicap.py는 읽기만). desk.strategies의 weekly_target·project_caps·_drift·in_semis_budget와
build_base.book_2026()/book_hold()만 쓴다. desk/는 import만 하고 config.RULE['semis_cap']은 project_caps 호출 동안만
바꿨다가 try/finally로 되돌린다. 계기를 끄면 desk run_model과 1e-9로 같아야 한다(실행 때마다 먼저 확인).
harness(H)는 자료(px·base)와 구간 지표(stats·periods)에만 쓴다.

판정 기준(반박 검토 K의 권고): '같은 최대 낙폭을 내는 고정 노출'보다 수익이 높은가.
  고정 노출 = 기본 경로 일별 수익 × c (c는 0~1, 0.005 간격 201개, 구간마다 실제로 다시 계산).
  효율선 수익 = 낙폭이 변형 이하(같거나 얕음)인 c 중 최대 수익. 차 = 변형 수익 − 효율선 수익.

시험 범위(실행 전 고정, 결과를 본 뒤 추가하지 않음)
  (1) 재현: 후보와 기본의 주장 숫자(0.3%p 넘게 다르면 원인 규명).
  (2) 이웃 48개: SMA {10,15,20,25,30,40} × 켜졌을 때 한도 {0, 0.10, 0.15, 0.25} × 넘친 몫 {현금, 재배분}. 모두 주간·신호 (a).
  (3) 신호 날짜 민감도(후보 SMA20·0%·현금 고정):
      (a) 직전 거래일 종가(후보) (b) 그 전 거래일 종가(하루 더 오래된 정보)
      (c) 직전 5거래일 중 3일 이상 종가 < 그날 SMA20 (d) 직전 주 평균 종가 < 직전 주 마지막 거래일 SMA20
      (e) 체결 하루 지연: 7종목 교체는 월요일 그대로(그날은 지난주 한도 유지), 이번 주 한도(금요일 종가 신호)는 화요일 시가에 적용
      (e') 화요일 시가에 월요일 종가 신호로 적용(하루 늦은 결정, 최신 정보)
  (4) 추가 반박(사전 고정): 결정 요일 이동 — 7종목 교체는 월요일 그대로, 반도체 한도 결정·적용만 화·수·목·금 시가
      (그 전 거래일 종가 신호, 다음 주 같은 요일까지 유지).
  (5) 위약(placebo): (P1) 계기 없는 상시 반도체 한도 {0,0.05,…,0.45} 현금 — 같은 낙폭의 상시 한도보다 나은가
      (P2) 후보의 주간 신호 열을 주 단위로 원형 이동(1~N−1주) — 실제 신호의 순위.
  (6) 결과 본 뒤 추가한 점검(선택편향 있음, 판정의 안정성 점검용):
      (6a) 요일 이동 코드 확인: 월요일(wd=0)이 후보와 같은지
      (6b) 현금 이웃 24개(SMA 6 × 한도 4) × 결정 요일 5(월~금): 요일별 통과 수와 5요일 평균
      (6c) 요일 없는 매일 상태판: 매 시가에 신호가 켜져 있으면 한도(현금으로 깎음), 꺼지면 그 주 목표 반도체까지 되삼.
           SMA 6 × 한도 {0, 0.15}
      (6d) 후보의 주별 기여: 결정 주마다 기본 대비 로그 수익 차 — 우위가 몇 주에 몰렸는지
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import date

import numpy as np
import pandas as pd

import harness as H
import build_base as BB                    # ROOT를 sys.path에 넣는다
from desk import config, strategies as S   # noqa: E402  (import만)

OUT = "v3_semicap_recalc"
SMAS = [10, 15, 20, 25, 30, 40]
CAPS = [0.0, 0.10, 0.15, 0.25]
OVERFLOWS = ["현금", "재배분"]
CAND = {"sma": 20, "cap": 0.0, "overflow": "현금"}
EXTRA26 = {"6/22~9/30": ("2026-06-22", "2026-09-30")}
CLAIM = {"후보": {"2026 전체": (2.172, -0.142), "2026 앞": (1.903, -0.142), "2026 뒤": (0.093, -0.109),
                  "2026 AI쇼크": (-0.082, None), "2026 7~9월": (0.186, None), "hold 2025": (0.484, -0.351),
                  "hold 2024": (0.269, -0.265), "평균노출": 0.78},
         "기본": {"2026 전체": (2.237, -0.203), "2026 뒤": (0.046, -0.203), "2026 AI쇼크": (-0.187, None),
                  "hold 2025": (0.422, -0.471), "hold 2024": (0.334, -0.382)}}
FRONT_KEYS = [("2026", "전체"), ("2026", "뒤"), ("hold", "2025"), ("hold", "2024")]
CGRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)


@contextmanager
def semis_cap(c: float):
    saved = dict(config.RULE)
    try:
        config.RULE["semis_cap"] = c
        yield
    finally:
        config.RULE.clear()
        config.RULE.update(saved)


# ============================================================================ 엔진
class Engine:
    """desk run_model과 같은 루프. cap_t[i] = i일 시가에 적용 중인 반도체+미분류 한도(None = 현행)."""

    def __init__(self, bn: str):
        self.bn = bn
        self.b = H.base(bn)
        self.book = BB.book_2026() if bn == "2026" else BB.book_hold()
        bk, days = self.book, self.b.days
        self.days = days
        self.k0 = bk.days.index(days[0])
        if bk.days[self.k0:self.k0 + len(days)] != days:
            raise ValueError("book 거래일과 기본 경로 거래일이 다르다")
        # run_model과 같은 결정 조건: 첫 결정 전에는 매일, 그 뒤에는 ISO 주가 바뀐 날만 weekly_target
        self.dec = {}
        last, selected = False, set()
        for i, day in enumerate(days):
            k = self.k0 + i
            if not last or S.iso_week(day) != S.iso_week(bk.days[k - 1]):
                d = S.weekly_target(bk, k, day, selected)
                if d["ranked"]:
                    selected, last = set(d["selected"]), True
                    self.dec[i] = {"raw": d["raw"], "target": d["target"]}
        keys = {key for d in self.dec.values() for key in d["raw"]}
        self.O = {key: S._col(bk, bk.opn, key) for key in keys}
        self.semis = {key for key in keys if S.in_semis_budget(bk.sector[key])}
        self.sector = bk.sector
        self._pc = {}
        self.dec_of = np.full(len(days), -1)      # i일에 효력 있는 마지막 결정일
        cur = -1
        for i in range(len(days)):
            if i in self.dec:
                cur = i
            self.dec_of[i] = cur

    def rets(self, w: dict, k: int) -> dict:
        out = {}
        for key in w:
            o = self.O[key]
            ok = o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0
            out[key] = (o[k] / o[k - 1] - 1) if ok else 0.0
        return out

    def ssum(self, w: dict) -> float:
        return float(sum(v for key, v in w.items() if key in self.semis))

    def scale_semis(self, w: dict, c: float) -> dict:
        s = self.ssum(w)
        if s <= c + 1e-12:
            return dict(w)
        f = c / s
        return {key: (v * f if key in self.semis else v) for key, v in w.items()
                if (v * f if key in self.semis else v) > 1e-12}

    def target(self, i: int, cap, overflow: str) -> dict:
        d = self.dec[i]
        if cap is None:
            return dict(d["target"])
        if overflow == "현금":
            return self.scale_semis(d["target"], cap)
        if (i, cap) not in self._pc:
            with semis_cap(cap):
                self._pc[(i, cap)] = S.project_caps(d["raw"], self.sector)
        return dict(self._pc[(i, cap)])

    def midweek(self, w: dict, i_dec: int, cap, prev_cap) -> dict:
        """주중 한도 변경(현금 판만): 내리면 흘러간 반도체 비중을 비례 축소, 올리면 그 주 목표의 반도체 비중까지(현금 안에서) 되산다."""
        s = self.ssum(w)
        if cap is not None and s > cap + 1e-12:
            return self.scale_semis(w, cap)
        lo_prev = 0.5 if prev_cap is None else prev_cap
        lo_new = 0.5 if cap is None else cap
        if lo_new <= lo_prev + 1e-12:
            return dict(w)
        tgt = self.target(i_dec, cap, "현금")
        ts = {key: v for key, v in tgt.items() if key in self.semis}
        non = {key: v for key, v in w.items() if key not in self.semis}
        room = 1.0 - sum(non.values())
        tot = sum(ts.values())
        f = min(1.0, room / tot) if tot > 1e-12 else 0.0
        out = dict(non)
        for key, v in ts.items():
            if v * f > 1e-12:
                out[key] = v * f
        return out

    def run(self, cap_t=None, overflow: str = "현금"):
        days = self.days
        w: dict = {}
        value, weights = [1.0], []
        for i in range(len(days)):
            k = self.k0 + i
            if i > 0:
                w, gross = S._drift(w, self.rets(w, k))
                value.append(value[-1] * (1 + gross))
            cap = None if cap_t is None else cap_t[i]
            if i in self.dec:
                w = self.target(i, cap, overflow)
            elif cap_t is not None and i > 0 and self.dec_of[i] >= 0 and cap != cap_t[i - 1]:
                if overflow != "현금":
                    raise ValueError("주중 한도 변경은 현금 판만 정의했다")
                w = self.midweek(w, int(self.dec_of[i]), cap, cap_t[i - 1])
            weights.append(dict(w))
        v = pd.Series(value, index=days)
        return (v.shift(-1) / v - 1).dropna(), weights


def check_engine(eng: Engine) -> dict:
    r, wts = eng.run()
    path = S.run_model(eng.book, start=eng.days[0])
    n = len(eng.days)
    v_mine = np.concatenate([[1.0], (1 + r.to_numpy()).cumprod()])
    return {"자산_vs_run_model": float(np.max(np.abs(v_mine - np.asarray(path.value[:n])))),
            "수익_vs_기본json": float(np.max(np.abs(r.to_numpy() - eng.b.ret.to_numpy()))),
            "비중_vs_run_model": float(max(abs(wts[i].get(key, 0.0) - path.weights[i].get(key, 0.0))
                                          for i in range(n) for key in set(wts[i]) | set(path.weights[i])))}


# ============================================================================ 신호(i일 시가에 쓸 값)
SOXX = H.px("SOXX", "Close")
SOX_DATES = SOXX.index.to_numpy()


def aligned(sig: pd.Series, days: list, lag_obs: int = 0) -> np.ndarray:
    """sig(index=SOXX 거래일, t일 종가로 아는 값) → 각 거래일 시가에 쓸 값: 그 날보다 엄격히 앞선 마지막 관측에서 lag_obs개 더 앞."""
    s = sig.reindex(SOXX.index)
    idx = np.searchsorted(SOX_DATES, np.asarray(days), side="left") - 1 - lag_obs
    vals = s.to_numpy()
    out = np.array([bool(vals[j]) if j >= 0 and np.isfinite(vals[j]) else False for j in idx])
    return out


def below_sma(n: int) -> pd.Series:
    m = SOXX.rolling(n).mean()
    return ((SOXX < m) & m.notna()).astype(float).where(m.notna())


def sig_c() -> pd.Series:
    b = below_sma(20)
    return (b.rolling(5).sum() >= 3).astype(float).where(b.rolling(5).count() == 5)


def sig_d() -> pd.Series:
    m = SOXX.rolling(20).mean()
    wk = np.array(["%d-%02d" % date.fromisoformat(d).isocalendar()[:2] for d in SOXX.index])
    avg = SOXX.groupby(wk).transform(lambda x: x.expanding().mean())   # 같은 ISO 주, t까지 평균
    return ((avg < m) & m.notna()).astype(float).where(m.notna())


def weekly_caps(eng: Engine, sig: np.ndarray, cap: float) -> list:
    """주간: 결정일 시가 신호로 그 주 한도를 정한다(다음 결정일까지 유지)."""
    return [None if eng.dec_of[i] < 0 or not sig[eng.dec_of[i]] else cap for i in range(len(eng.days))]


def delayed_caps(eng: Engine, sig_week: np.ndarray, cap: float, fresh: np.ndarray | None = None) -> list:
    """체결 하루 지연: 결정일(월) 시가는 지난주 한도 유지, 다음 거래일(화) 시가부터 이번 주 한도.
    fresh가 주어지면 화요일 한도를 화요일 시가 신호(=월요일 종가)로 정한다."""
    n = len(eng.days)
    decs = sorted(eng.dec)
    out = [None] * n
    prev = None
    for j, d in enumerate(decs):
        nxt = decs[j + 1] if j + 1 < n and j + 1 < len(decs) else n
        src = (fresh[d + 1] if fresh is not None and d + 1 < n else sig_week[d])
        new = cap if src else None
        for i in range(d, nxt):
            out[i] = prev if i == d else new
        prev = new
    return out


def weekday_caps(eng: Engine, sig: np.ndarray, cap: float, wd: int) -> list:
    """반도체 한도 결정 요일 이동: 각 ISO 주에서 요일 ≥ wd인 첫 거래일 시가에 그 시가 신호로 정하고 다음 결정까지 유지."""
    out, cur, seen = [], None, set()
    for i, d in enumerate(eng.days):
        dd = date.fromisoformat(d)
        wk = dd.isocalendar()[:2]
        if wk not in seen and dd.weekday() >= wd:
            seen.add(wk)
            cur = cap if sig[i] else None
        out.append(cur if eng.dec_of[i] >= 0 else None)
    return out


# ============================================================================ 지표
def periods(bn: str) -> dict:
    p = dict(H.base(bn).periods)
    if bn == "2026":
        p.update(EXTRA26)
    return p


_FIX = {}


def fixed_grid(bn: str, lo: str, hi: str):
    if (bn, lo, hi) not in _FIX:
        r = H.base(bn).ret
        _FIX[(bn, lo, hi)] = [(float(c), H.stats(c * r, lo, hi)) for c in CGRID]
    return _FIX[(bn, lo, hi)]


def frontier(bn: str, lo: str, hi: str, v: dict) -> dict:
    ok = [(c, g) for c, g in fixed_grid(bn, lo, hi) if g["낙폭"] >= v["낙폭"] - 1e-9]
    if not ok:
        return {"c": None, "고정수익": None, "차": None}
    c, g = max(ok, key=lambda x: (x[1]["수익"], -x[0]))
    return {"c": c, "고정수익": g["수익"], "고정낙폭": g["낙폭"], "차": v["수익"] - g["수익"]}


def rel_exposure(wv: list, wb: list) -> np.ndarray:
    iv = np.array([sum(x.values()) for x in wv])
    ib = np.array([sum(x.values()) for x in wb])
    return np.where(ib > 1e-9, iv / np.where(ib > 1e-9, ib, 1.0), 1.0)


def measure(name: str, runs: dict, base_w: dict, eng: dict, params=None) -> dict:
    row = {"name": name, "params": params or {}}
    for bn, (r, wts) in runs.items():
        m = {k: H.stats(r, lo, hi) for k, (lo, hi) in periods(bn).items()}
        e = rel_exposure(wts, base_w[bn])[:-1]
        m["평균노출"] = float(e.mean())
        m["평균반도체"] = float(np.mean([eng[bn].ssum(x) for x in wts[:-1]]))
        m["평균투자"] = float(np.mean([sum(x.values()) for x in wts[:-1]]))
        row[bn] = m
    row["효율선"] = {f"{bn} {k}": frontier(bn, *periods(bn)[k], row[bn][k]) for bn, k in FRONT_KEYS if bn in row}
    return row


def fmt(d: dict) -> str:
    return f"{d['수익']:+.1%}/{d['낙폭']:.1%}"


def passes(r: dict, bfa: float) -> dict:
    p = {"앞90": (1 + r["2026"]["앞"]["수익"]) / bfa >= 0.90, "뒤>고정": (r["효율선"]["2026 뒤"]["차"] or -1) > 1e-4,
         "2025>고정": (r["효율선"]["hold 2025"]["차"] or -1) > 1e-4}
    p["셋"] = all(p.values())
    return p


def posthoc(eng, base_w, sig_open, base_row, cand, bfa) -> dict:
    """(6) 결과 본 뒤 추가한 점검."""
    out = {}
    # (6a) 요일 이동 코드 확인
    r0 = {bn: e.run(weekday_caps(e, sig_open[bn][20], 0.0, 0), "현금") for bn, e in eng.items()}
    out["wd0_same"] = bool(all(np.allclose(r0[bn][0].to_numpy(), e.run(weekly_caps(e, sig_open[bn][20], 0.0), "현금")[0]
                                           .to_numpy(), atol=1e-12) for bn, e in eng.items()))
    print("\n(6a) 요일 이동(월) = 후보:", out["wd0_same"])
    # (6b) 현금 이웃 × 결정 요일 5
    tab, cnt = [], {}
    names = ["월", "화", "수", "목", "금"]
    for n in SMAS:
        for cap in CAPS:
            per = []
            for wd in range(5):
                runs = {bn: e.run(weekday_caps(e, sig_open[bn][n], cap, wd), "현금") for bn, e in eng.items()}
                m = measure(f"SMA{n} 한도{cap:.0%} 현금 {names[wd]}", runs, base_w, eng)
                p = passes(m, bfa)
                per.append({"요일": names[wd], "2026전체": m["2026"]["전체"], "앞": m["2026"]["앞"], "뒤": m["2026"]["뒤"],
                            "6/22~9/30": m["2026"]["6/22~9/30"]["수익"], "2025": m["hold"]["2025"],
                            "2024": m["hold"]["2024"],
                            "효율선차": {k: m["효율선"][k]["차"] for k in m["효율선"]}, "통과": p})
            avg = {k: float(np.mean([x["효율선차"][k] for x in per])) for k in per[0]["효율선차"]}
            avg_back = float(np.mean([x["뒤"]["수익"] for x in per]))
            tab.append({"name": f"SMA{n} 한도{cap:.0%} 현금", "요일별": per, "5요일평균_효율선차": avg,
                        "5요일평균_뒤수익": avg_back,
                        "통과요일수": {k: int(sum(x["통과"][k] for x in per)) for k in ("앞90", "뒤>고정", "2025>고정", "셋")}})
    for k in ("앞90", "뒤>고정", "2025>고정", "셋"):
        cnt[k] = {names[wd]: int(sum(t["요일별"][wd]["통과"][k] for t in tab)) for wd in range(5)}
    cnt["5요일평균 뒤>고정"] = int(sum(t["5요일평균_효율선차"]["2026 뒤"] > 1e-4 for t in tab))
    cnt["5요일평균 2025>고정"] = int(sum(t["5요일평균_효율선차"]["hold 2025"] > 1e-4 for t in tab))
    cnt["5요일 모두 셋 통과"] = int(sum(t["통과요일수"]["셋"] == 5 for t in tab))
    print("(6b) 현금 이웃 24 × 요일 5 통과 수(요일별):", json.dumps(cnt, ensure_ascii=False))
    for t in tab:
        a = t["5요일평균_효율선차"]
        print(f"   {t['name']:<16} 뒤 수익(월~금) " + " ".join(f"{x['뒤']['수익']:+.1%}" for x in t["요일별"]) +
              f" | 5요일평균 효율선차 전체 {a['2026 전체']:+.3f} 뒤 {a['2026 뒤']:+.3f} 2025 {a['hold 2025']:+.3f} "
              f"2024 {a['hold 2024']:+.3f} | 셋통과 요일 {t['통과요일수']['셋']}")
    out["phase_tab"], out["phase_cnt"] = tab, cnt
    # 후보(SMA20 0% 현금) 요일별 뒤 구간 분해
    wins = {"6/1~6/19": ("2026-06-01", "2026-06-19"), "6/22~7/29": ("2026-06-22", "2026-07-29"),
            "7/30~9/11": ("2026-07-30", "2026-09-11"), "9/14~10/9": ("2026-09-14", "2026-10-09"),
            "2/23~4/10": ("2026-02-23", "2026-04-10")}
    e26 = eng["2026"]
    dec_tab = {"기본": {k: H.stats(e26.b.ret, *v)["수익"] for k, v in wins.items()}}
    for wd in range(5):
        r, _ = e26.run(weekday_caps(e26, sig_open["2026"][20], 0.0, wd), "현금")
        dec_tab[names[wd]] = {k: H.stats(r, *v)["수익"] for k, v in wins.items()}
    out["phase_windows"] = dec_tab
    print("   SMA20 0% 현금 요일별 구간 수익:")
    for k, v in dec_tab.items():
        print("     ", k, " ".join(f"{w} {x:+.1%}" for w, x in v.items()))
    # (6c) 매일 상태판
    daily = []
    for n in SMAS:
        for cap in (0.0, 0.15):
            runs = {}
            for bn, e in eng.items():
                s = sig_open[bn][n]
                ct = [(cap if s[i] else None) if e.dec_of[i] >= 0 else None for i in range(len(e.days))]
                runs[bn] = e.run(ct, "현금")
            m = measure(f"매일 SMA{n} 한도{cap:.0%} 현금", runs, base_w, eng, {"sma": n, "cap": cap, "mode": "매일"})
            m["통과"] = passes(m, bfa)
            daily.append(m)
    print("(6c) 매일 상태판")
    for r in daily:
        a, h, f = r["2026"], r["hold"], r["효율선"]
        fd = "/".join(f"{(f[k]['차'] or 0) * 100:+.1f}" for k in ("2026 전체", "2026 뒤", "hold 2025", "hold 2024"))
        print(f"   {r['name']:<20} {fmt(a['전체'])} {fmt(a['앞'])} {fmt(a['뒤'])} {a['AI쇼크']['수익']:+.1%} "
              f"{a['6/22~9/30']['수익']:+.1%} {fmt(h['2025'])} {fmt(h['2024'])} {fd} 노출 {a['평균노출']:.2f} "
              f"{'셋통과' if r['통과']['셋'] else ''}")
    out["daily"] = daily
    # (6d) 후보 주별 기여(2026)
    e = eng["2026"]
    rb = e.b.ret
    rc, _ = e.run(weekly_caps(e, sig_open["2026"][20], 0.0), "현금")
    decs = sorted(e.dec)
    wk = []
    for j, d in enumerate(decs):
        lo = e.days[d]
        hi = e.days[decs[j + 1] - 1] if j + 1 < len(decs) else e.days[-2]
        xb = float(np.prod(1 + rb[(rb.index >= lo) & (rb.index <= hi)].to_numpy()) - 1)
        xc = float(np.prod(1 + rc[(rc.index >= lo) & (rc.index <= hi)].to_numpy()) - 1)
        wk.append({"주": lo, "신호": bool(sig_open["2026"][20][d]), "기본": xb, "후보": xc,
                   "로그차": float(np.log1p(xc) - np.log1p(xb))})
    tot = sum(x["로그차"] for x in wk)
    on = [x for x in wk if x["신호"]]
    srt = sorted(on, key=lambda x: -x["로그차"])
    out["weekly"] = {"전체 로그차": tot, "켜진 주": len(on), "주": wk}
    print(f"(6d) 후보 2026 주별 기여: 전체 로그차 {tot:+.3f}, 켜진 주 {len(on)}개")
    for x in srt:
        print(f"   {x['주']} 기본 {x['기본']:+.1%} 후보 {x['후보']:+.1%} 로그차 {x['로그차']:+.3f}")
    return out


# ============================================================================ 실행
def main():
    saved_rule = dict(config.RULE)
    eng = {bn: Engine(bn) for bn in ("2026", "hold")}
    for bn, e in eng.items():
        rule = json.loads((H.CACHE / f"{H.FILES[bn]}.json").read_text(encoding="utf-8"))["rule"]
        if dict(config.RULE) != rule:
            raise SystemExit(f"{bn}: config.RULE이 기본 경로 규칙과 다르다")
    chk = {bn: check_engine(e) for bn, e in eng.items()}
    print("엔진 확인(계기 끔 vs desk run_model·기본 json):", json.dumps(chk, ensure_ascii=False))
    for bn, c in chk.items():
        if max(c.values()) > 1e-9:
            raise SystemExit(f"{bn}: 엔진이 run_model과 다르다 {c}")
    base_runs = {bn: e.run() for bn, e in eng.items()}
    base_w = {bn: base_runs[bn][1] for bn in eng}
    base_row = measure("기본", base_runs, base_w, eng)

    # 신호(SOXX 관측 기준) → 거래일 시가 정렬
    sig_open = {bn: {n: aligned(below_sma(n), e.days) for n in SMAS} for bn, e in eng.items()}
    # harness.at_open과 같은지 교차 확인(SMA20)
    for bn, e in eng.items():
        h = (H.at_open(below_sma(20).dropna(), e.b, fill=0.0) > 0.5).to_numpy(bool)
        if not np.array_equal(h, sig_open[bn][20]):
            raise SystemExit(f"{bn}: 신호 정렬이 harness.at_open과 다르다")

    # ---- (1) 후보 재현
    cand_runs = {bn: e.run(weekly_caps(e, sig_open[bn][20], 0.0), "현금") for bn, e in eng.items()}
    cand = measure("후보 S2 SMA20 0% 현금 주간", cand_runs, base_w, eng, CAND)
    repro = {}
    for who, row in (("후보", cand), ("기본", base_row)):
        for lab, val in CLAIM[who].items():
            if lab == "평균노출":
                got = row["2026"]["평균노출"]
                repro[f"{who} {lab}"] = {"주장": val, "재계산": round(got, 4), "차(%p)": round((got - val) * 100, 2)}
                continue
            bn, k = lab.split(" ")
            got = row[bn][k]
            d = {"주장": val[0], "재계산": round(got["수익"], 4), "차(%p)": round((got["수익"] - val[0]) * 100, 2)}
            if val[1] is not None:
                d.update({"주장낙폭": val[1], "재계산낙폭": round(got["낙폭"], 4),
                          "낙폭차(%p)": round((got["낙폭"] - val[1]) * 100, 2)})
            repro[f"{who} {lab}"] = d
    print("\n(1) 재현")
    for k, v in repro.items():
        print("  ", k, v)
    bad = {k: v for k, v in repro.items() if abs(v["차(%p)"]) > 0.3 or abs(v.get("낙폭차(%p)", 0)) > 0.3}
    print("   0.3%p 넘는 차:", bad or "없음 (주장 숫자는 소수 1자리 반올림)")

    # ---- (2) 이웃 48개
    rows = []
    for n in SMAS:
        for cap in CAPS:
            for ov in OVERFLOWS:
                runs = {bn: e.run(weekly_caps(e, sig_open[bn][n], cap), ov) for bn, e in eng.items()}
                rows.append(measure(f"SMA{n} 한도{cap:.0%} {ov}", runs, base_w, eng,
                                    {"sma": n, "cap": cap, "overflow": ov}))
    bfa = 1 + base_row["2026"]["앞"]["수익"]
    for r in rows + [cand, base_row]:
        r["앞자산비"] = (1 + r["2026"]["앞"]["수익"]) / bfa
        r["통과"] = {"앞90": r["앞자산비"] >= 0.90, "뒤>고정": (r["효율선"]["2026 뒤"]["차"] or -1) > 1e-4,
                   "2025>고정": (r["효율선"]["hold 2025"]["차"] or -1) > 1e-4}
        r["통과"]["셋"] = all(r["통과"].values())
    cnt = {k: int(sum(r["통과"][k] for r in rows)) for k in ("앞90", "뒤>고정", "2025>고정", "셋")}
    cnt["2026전체>고정"] = int(sum((r["효율선"]["2026 전체"]["차"] or -1) > 1e-4 for r in rows))
    cnt["2024>고정"] = int(sum((r["효율선"]["hold 2024"]["차"] or -1) > 1e-4 for r in rows))
    cnt["뒤 수익·낙폭 모두 기본보다 나음"] = int(sum(r["2026"]["뒤"]["수익"] > base_row["2026"]["뒤"]["수익"]
                                          and r["2026"]["뒤"]["낙폭"] > base_row["2026"]["뒤"]["낙폭"] for r in rows))
    cnt["넷(셋+2024>고정)"] = int(sum(r["통과"]["셋"] and (r["효율선"]["hold 2024"]["차"] or -1) > 1e-4 for r in rows))
    print("\n(2) 이웃 48개 통과 수:", cnt)
    print("이름 | 2026 전체 | 앞 | 뒤 | AI쇼크 | 6/22~9/30 | 2025 | 2024 | 앞자산비 | 효율선 차(전체/뒤/2025/2024) | 노출")
    for r in [base_row, cand] + rows:
        a, h, f = r["2026"], r["hold"], r["효율선"]
        fd = "/".join(f"{(f[k]['차'] or 0) * 100:+.1f}" for k in ("2026 전체", "2026 뒤", "hold 2025", "hold 2024"))
        print(f"{r['name']:<26} {fmt(a['전체'])} {fmt(a['앞'])} {fmt(a['뒤'])} {a['AI쇼크']['수익']:+.1%} "
              f"{a['6/22~9/30']['수익']:+.1%} {fmt(h['2025'])} {fmt(h['2024'])} {r['앞자산비']:.3f} {fd} "
              f"{a['평균노출']:.2f} {'셋통과' if r['통과']['셋'] else ''}")

    # ---- (3) 신호 날짜 민감도(후보 SMA20·0%·현금)
    sens_sig = {}
    for bn, e in eng.items():
        sens_sig[bn] = {"(a) 직전 거래일 종가(후보)": sig_open[bn][20],
                        "(b) 그 전 거래일 종가": aligned(below_sma(20), e.days, lag_obs=1),
                        "(c) 직전 5일 중 3일 이상 아래": aligned(sig_c(), e.days),
                        "(d) 직전 주 평균 < 마지막 SMA20": aligned(sig_d(), e.days)}
    sens = []
    for lab in sens_sig["2026"]:
        runs = {bn: e.run(weekly_caps(e, sens_sig[bn][lab], 0.0), "현금") for bn, e in eng.items()}
        sens.append(measure(lab, runs, base_w, eng))
    runs = {bn: e.run(delayed_caps(e, sig_open[bn][20], 0.0), "현금") for bn, e in eng.items()}
    sens.append(measure("(e) 체결 하루 지연(금 신호→화 시가)", runs, base_w, eng))
    runs = {bn: e.run(delayed_caps(e, sig_open[bn][20], 0.0, fresh=sig_open[bn][20]), "현금") for bn, e in eng.items()}
    sens.append(measure("(e') 화 시가에 월 종가 신호", runs, base_w, eng))
    for wd, nm in ((1, "화"), (2, "수"), (3, "목"), (4, "금")):
        runs = {bn: e.run(weekday_caps(e, sig_open[bn][20], 0.0, wd), "현금") for bn, e in eng.items()}
        sens.append(measure(f"(f) 한도 결정 {nm}요일 시가", runs, base_w, eng))
    for r in sens:
        r["앞자산비"] = (1 + r["2026"]["앞"]["수익"]) / bfa
        r["통과"] = {"앞90": r["앞자산비"] >= 0.90, "뒤>고정": (r["효율선"]["2026 뒤"]["차"] or -1) > 1e-4,
                   "2025>고정": (r["효율선"]["hold 2025"]["차"] or -1) > 1e-4}
        r["통과"]["셋"] = all(r["통과"].values())
    print("\n(3) 신호 날짜·체결 민감도")
    for r in sens:
        a, h, f = r["2026"], r["hold"], r["효율선"]
        fd = "/".join(f"{(f[k]['차'] or 0) * 100:+.1f}" for k in ("2026 전체", "2026 뒤", "hold 2025", "hold 2024"))
        print(f"{r['name']:<30} {fmt(a['전체'])} {fmt(a['앞'])} {fmt(a['뒤'])} {a['AI쇼크']['수익']:+.1%} "
              f"{a['6/22~9/30']['수익']:+.1%} {a['7~9월']['수익']:+.1%} {fmt(h['2025'])} {fmt(h['2024'])} "
              f"{r['앞자산비']:.3f} {fd} {a['평균노출']:.2f}")
    # 2026 신호가 켜진 주(결정일)
    weeks_on = {lab: [eng["2026"].days[i] for i in sorted(eng["2026"].dec) if s[i]]
                for lab, s in sens_sig["2026"].items()}

    # ---- (5) 위약
    plac = []
    for c in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        runs = {bn: e.run([c] * len(e.days), "현금") for bn, e in eng.items()}
        plac.append(measure(f"상시 한도{c:.0%} 현금", runs, base_w, eng, {"cap": c}))

    def best_static(row, bn, k):
        v = row[bn][k]
        ok = [p[bn][k]["수익"] for p in plac + [base_row] if p[bn][k]["낙폭"] >= v["낙폭"] - 1e-9]
        return (max(ok) if ok else None)

    static_cmp = {}
    for r in [cand] + sens[1:4]:
        static_cmp[r["name"]] = {f"{bn} {k}": {"변형": r[bn][k]["수익"], "같은낙폭 상시한도 최대": best_static(r, bn, k)}
                                 for bn, k in [("2026", "전체"), ("2026", "앞"), ("2026", "뒤"), ("hold", "2025"),
                                               ("hold", "2024")]}
    print("\n(P1) 상시 반도체 한도(현금)")
    for p in plac:
        a, h = p["2026"], p["hold"]
        print(f"  {p['name']:<16} {fmt(a['전체'])} {fmt(a['앞'])} {fmt(a['뒤'])} {a['AI쇼크']['수익']:+.1%} "
              f"{fmt(h['2025'])} {fmt(h['2024'])} 노출 {a['평균노출']:.2f}")
    print("  후보 대비:", json.dumps(static_cmp[cand["name"]], ensure_ascii=False, default=lambda x: round(x, 4)))

    # P2 원형 이동: 결정일 신호 열을 주 단위로 돌린다(켜진 비율·연속 구조 유지)
    shift_res = {}
    for bn, e in eng.items():
        decs = sorted(e.dec)
        s_dec = np.array([sig_open[bn][20][i] for i in decs])
        out = []
        for sh in range(len(decs)):
            rolled = np.roll(s_dec, sh)
            sig = np.zeros(len(e.days), bool)
            for j, i in enumerate(decs):
                sig[i] = rolled[j]
            r, wts = e.run(weekly_caps(e, sig, 0.0), "현금")
            keys = ["전체", "앞", "뒤", "6/22~9/30"] if bn == "2026" else ["2025", "2024", "전체"]
            m = {k: H.stats(r, *periods(bn)[k]) for k in keys}
            fr = {k: frontier(bn, *periods(bn)[k], m[k])["차"] for k in keys}
            out.append({"shift": sh, **{k: m[k] for k in keys}, "효율선차": fr})
        act = out[0]
        rank = {}
        for k in act["효율선차"]:
            vals = np.array([o["효율선차"][k] for o in out])
            rank[k] = {"실제": act["효율선차"][k], "이동 중 실제보다 큰 비율": float(np.mean(vals[1:] > act["효율선차"][k])),
                       "이동 중앙값": float(np.median(vals[1:])), "이동 양수 비율": float(np.mean(vals[1:] > 0)),
                       "주수": len(decs) - 1}
        shift_res[bn] = {"rank": rank, "켜진 주 비율": float(s_dec.mean())}
        print(f"\n(P2) {bn} 원형 이동 {len(decs) - 1}개, 켜진 주 비율 {s_dec.mean():.2f}")
        for k, v in rank.items():
            print("   ", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})

    post = posthoc(eng, base_w, sig_open, base_row, cand, bfa)

    # 저장
    def slim(r):
        out = {"name": r["name"], "params": r.get("params", {})}
        for bn in ("2026", "hold"):
            out[bn] = {k: ({kk: round(vv, 5) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 5))
                       for k, v in r[bn].items()}
        out["효율선"] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in r["효율선"].items()}
        for k in ("앞자산비", "통과"):
            if k in r:
                out[k] = r[k]
        return out

    payload = {"엔진확인": chk, "재현": repro, "기본": slim(base_row), "후보": slim(cand),
               "이웃48_통과수": cnt, "이웃48": [slim(r) for r in rows], "신호민감도": [slim(r) for r in sens],
               "신호켜진주_2026": weeks_on, "위약_상시한도": [slim(p) for p in plac],
               "위약_상시한도_비교": static_cmp, "위약_원형이동": shift_res,
               "시험범위": {"이웃": len(rows), "신호민감도": len(sens), "상시한도": len(plac),
                        "원형이동": {bn: len(eng[bn].dec) - 1 for bn in eng}, "고정노출 c": len(CGRID),
                        "결과본뒤_요일x현금이웃": 24 * 5, "결과본뒤_매일상태": 12},
               "결과본뒤추가": {"요일0_후보와같음": post["wd0_same"],
                           "요일별_현금이웃": post["phase_tab"], "요일별_현금이웃_통과": post["phase_cnt"],
                           "후보_요일별_구간": post["phase_windows"],
                           "매일상태": [slim(r) for r in post["daily"]], "후보_주별기여": post["weekly"]}}
    p = H.RESULTS / f"{OUT}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if dict(config.RULE) != saved_rule:
        raise SystemExit("config.RULE이 되돌려지지 않았다")
    print("\nconfig.RULE 복구 확인:", config.RULE, "\n저장:", p)


if __name__ == "__main__":
    main()
