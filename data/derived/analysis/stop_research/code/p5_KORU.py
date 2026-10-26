"""5단계 KORU 갈래(2026-10-11): KORU(한국 3배)에 손절·추세·변동성 규칙을 걸면 '덜 담기'를 넘나.

사용자 질문: "SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?"
기초 = EWY(iShares MSCI 한국, 2000-05-12~, 미국 장 시간, 달러 기준). 상품 = EWY 3배 합성(보수 0.95% + 대조로 얻은 추가
연 로그 비용 2.6%, 금융비용 ^IRX) 2000-05~ 와 실제 KORU(2013-04-10~). 신호는 모두 EWY 종가로 낸다.

시점(사전 고정)
  - 비율 w는 t일 종가까지의 EWY 정보로 정해 t+1일 시가에 맞춘다(des[t] → t+1 시가).
  - 추적 손절가는 t−1일까지의 상품 최고 종가로 정한다. t일 시가가 손절가 이하면 시가, 저가가 닿으면 그 가격에 전량 매도.
  - 나머지는 현금 0%(연 4% 판은 한 줄 민감도). 수수료 0, 미끄러짐 0.5%는 |Δw|에 곱하는 민감도.
  - 지표가 아직 없으면(앞쪽 준비 기간) w=1(그냥 보유).

규칙 묶음(결과를 보기 전에 고정, 21개 + 그냥 보유 + 덜 담기 3개)
  R0 그냥 보유 · F 덜 담기 w∈{0.33,0.5,0.67} 매일 맞춤
  T  EWY 종가 < SMA n → 0, 위 → 1, n∈{50,100,150,200} × 띠 {0, 3%} = 8, SMA200 아래 → 0.5 = 1
  V  σ20(EWY 20일 실현변동성), m(σ20의 t−1까지 확장 중앙값, 60개 이상): w=clip(m/σ20), clip(0.75m/σ20),
     clip(σ*/(3σ20)) σ*∈{0.45,0.60} 매일 4 + 주 1회(주 첫 거래일) m/σ20·σ*0.60 = 6
  VS σ20 > k·m → 0, k∈{1.5, 2.0} = 2
  TV EWY > SMA200 그리고 σ20 < 1.5m → 1, 하나만 → 0.5, 둘 다 아님 → 0 = 1
  S  상품 산 뒤 최고 종가 대비 −{30,40,50}% 장중 → EWY > SMA50 종가 다음 시가 재진입 = 3

평가: 효율선초과 = 규칙 수익 − (규칙 최대 낙폭보다 깊지 않은 덜 담기 w(0~1, 0.01 간격) 가운데 최고 수익). 긴 창은 연율(CAGR) 차.
고르기: G1 2026 앞 효율선초과 최대(실제 KORU), G2 긴 역사(2000-05~2025-12, 합성) 효율선초과 최대, G3 2026 앞 칼마 최대.
출력: results/p5_KORU.md, results/p5_KORU.json. 실행: py p5_KORU.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import lev5
import stock_lab as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FEE, EXTRA = 0.0095, 0.026          # KORU 대조: 보수 0.95% + 추가 연 로그 비용 2.6%
SLIP = 0.005
CASH4 = 0.04
WGRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
MEDIAN_MIN = 60                     # 확장 중앙값을 쓰기 시작하는 σ20 개수

PICK, CHECK, AUX = "2026앞", "2026뒤", "2025"
WIN = {
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2026": ("2026-01-02", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"),
    # 폭락·반등 구간(맥락; 날짜는 사후에 정했으므로 고르기에 쓰지 않음. LEVIDX와 같은 날짜 + 한국 구간)
    "2000~02": ("2000-05-12", "2002-10-31"), "2008위기": ("2007-10-01", "2009-03-31"), "2011": ("2011-01-03", "2011-12-30"),
    "2015~16": ("2015-05-01", "2016-02-29"), "2018Q4": ("2018-10-01", "2018-12-31"), "2020급락": ("2020-02-20", "2020-03-23"),
    "2020반등": ("2020-03-24", "2020-08-31"), "2022": ("2022-01-03", "2022-12-30"), "2024하반기": ("2024-07-01", "2024-12-31"),
    "2025관세": ("2025-02-20", "2025-04-08"), "2025반등": ("2025-04-09", "2025-07-31"), "AI쇼크": ("2026-06-23", "2026-07-29"),
    # 긴 역사
    "긴역사": ("2000-05-12", "2026-10-09"), "긴역사_앞이전": ("2000-05-12", "2025-12-31"),
    "실제기간": ("2013-04-11", "2026-10-09"),
    # 모든 지표가 준비된 뒤(200일선은 2001-02 말부터). 2000년에는 긴 지표 규칙이 켜질 수 없어 그 영향을 뺀 비교
    "준비후": ("2001-03-01", "2026-10-09"), "준비후_앞이전": ("2001-03-01", "2025-12-31"),
}
WIN["2000"] = ("2000-05-12", "2000-12-31")
for _y in range(2001, 2027):
    WIN.setdefault(str(_y), (f"{_y}-01-01", f"{_y}-12-31"))
CRASH = ["2000~02", "2008위기", "2011", "2015~16", "2018Q4", "2020급락", "2020반등", "2022", "2024하반기", "2025관세",
         "2025반등", "AI쇼크"]
CRASH_NAME = {"2000~02": "2000~02(5/12~)", "2008위기": "2007-10~2009-03", "2011": "2011", "2015~16": "2015-05~2016-02",
              "2018Q4": "2018 4분기", "2020급락": "2020 급락", "2020반등": "2020 반등", "2022": "2022",
              "2024하반기": "2024 하반기", "2025관세": "2025 관세", "2025반등": "2025 반등", "AI쇼크": "2026 7월 쇼크"}
LONG_WINS = {"긴역사", "긴역사_앞이전", "실제기간", "준비후", "준비후_앞이전"}


# ----------------------------------------------------------------------------- 규칙
def rule_book():
    R = [("R0", "그냥 보유", "R", {})]
    for w in (0.33, 0.5, 0.67):
        R.append((f"F{int(round(w * 100))}", f"덜 담기 {w:g}(매일 맞춤)", "F", {"w": w}))
    for n in (50, 100, 150, 200):
        R.append((f"T{n}", f"EWY<{n}일선 → 현금", "T", {"n": n, "band": 0.0}))
        R.append((f"T{n}b", f"EWY {n}일선 띠 3% → 현금", "T", {"n": n, "band": 0.03}))
    R.append(("T200h", "EWY<200일선 → 절반", "T", {"n": 200, "half": True}))
    R += [("V1", "변동성 맞춤 m/σ20", "V", {"k": 1.0}), ("V075", "변동성 맞춤 0.75·m/σ20", "V", {"k": 0.75}),
          ("V45", "목표 변동성 45%", "V", {"tgt": 0.45}), ("V60", "목표 변동성 60%", "V", {"tgt": 0.60}),
          ("V1w", "변동성 맞춤 m/σ20 주 1회", "V", {"k": 1.0, "weekly": True}),
          ("V60w", "목표 변동성 60% 주 1회", "V", {"tgt": 0.60, "weekly": True})]
    R += [("VS15", "σ20>1.5×중앙 → 현금", "VS", {"k": 1.5}), ("VS20", "σ20>2×중앙 → 현금", "VS", {"k": 2.0})]
    R.append(("TV", "200일선+변동성 3단(1/0.5/0)", "TV", {}))
    for x in (0.30, 0.40, 0.50):
        R.append((f"S{int(x * 100)}", f"추적 −{int(x * 100)}% → EWY>50일선 재진입", "S", {"x": x}))
    return R


def indicators(u: S.Bars) -> dict:
    sig = u.rvol(20)
    m = pd.Series(sig).expanding(min_periods=MEDIAN_MIN).median().shift(1).to_numpy()   # t−1까지
    wk = pd.to_datetime(pd.Series(u.days)).dt.isocalendar()
    key = (wk.year.astype(int) * 100 + wk.week.astype(int)).to_numpy()
    first_of_week = np.concatenate([[True], key[1:] != key[:-1]])
    return {"c": u.c, "sma": {n: u.sma(n) for n in (50, 100, 150, 200)}, "sig": sig, "m": m, "fow": first_of_week}


def desired(fam: str, p: dict, ind: dict) -> np.ndarray:
    """des[t] = t일 종가 뒤 정한 목표 비율(t+1 시가 체결). NaN = 그날은 맞추지 않음(주 1회 규칙)."""
    c = ind["c"]
    n = len(c)
    if fam == "R":
        return np.ones(n)
    if fam == "F":
        return np.full(n, p["w"])
    if fam == "T":
        s = ind["sma"][p["n"]]
        if p.get("half"):
            return np.where(np.isfinite(s) & (c < s), 0.5, 1.0)
        b = p["band"]
        if b == 0:
            return np.where(np.isfinite(s) & (c < s), 0.0, 1.0)
        out = np.ones(n)
        state = 1.0
        for t in range(n):
            if np.isfinite(s[t]):
                if state == 1.0 and c[t] < s[t] * (1 - b):
                    state = 0.0
                elif state == 0.0 and c[t] > s[t] * (1 + b):
                    state = 1.0
            out[t] = state
        return out
    sig, m = ind["sig"], ind["m"]
    if fam == "V":
        if "tgt" in p:
            w = np.where(np.isfinite(sig), np.clip(p["tgt"] / (3 * sig), 0, 1), 1.0)
        else:
            ok = np.isfinite(sig) & np.isfinite(m)
            w = np.where(ok, np.clip(p["k"] * m / np.where(ok, sig, 1.0), 0, 1), 1.0)
        if p.get("weekly"):
            # 주 첫 거래일 시가에만 맞춘다: 그 시가에 쓰는 신호는 전날 종가(des[t−1]) → des[t−1]을 남기고 나머지는 NaN
            fow_next = np.concatenate([ind["fow"][1:], [False]])
            w = np.where(fow_next, w, np.nan)
        return w
    if fam == "VS":
        ok = np.isfinite(sig) & np.isfinite(m)
        return np.where(ok & (sig > p["k"] * np.where(ok, m, 0)), 0.0, 1.0)
    if fam == "TV":
        s = ind["sma"][200]
        up = ~(np.isfinite(s) & (c < s))
        ok = np.isfinite(sig) & np.isfinite(m)
        calm = ~(ok & (sig >= 1.5 * np.where(ok, m, 0)))
        return np.where(up & calm, 1.0, np.where(up | calm, 0.5, 0.0))
    raise ValueError(fam)


# ----------------------------------------------------------------------------- 엔진
def run(P: S.Bars, des=None, stop_x=None, reent=None, slip=0.0, cash_rate=0.0):
    """P = 상품 일봉. 0일 종가에 가치 1(처음 비율 = des[0] 또는 1)로 시작.
    반환 Vc(종가 가치), Wc(종가 비율), Wo(시가 체결 뒤 비율), turn(그날 |Δw| 합)."""
    o, l, c = P.o, P.l, P.c
    n = len(c)
    Vc, Wc, Wo, turn = np.empty(n), np.empty(n), np.empty(n), np.zeros(n)
    w0 = 1.0 if (stop_x is not None or not np.isfinite(des[0])) else float(des[0])
    cash, sh = 1.0 - w0, w0 / c[0]
    in_pos, peak = True, c[0]
    Vc[0], Wc[0], Wo[0] = 1.0, w0, w0
    for t in range(1, n):
        cash *= 1 + cash_rate / 252
        Vo = cash + sh * o[t]
        if stop_x is None:
            tgt = des[t - 1]
            if np.isfinite(tgt):
                cur = sh * o[t] / Vo if Vo > 0 else 0.0
                dw = abs(tgt - cur)
                if dw > 1e-12:
                    Vo -= slip * dw * Vo
                    sh = tgt * Vo / o[t]
                    cash = Vo - sh * o[t]
                    turn[t] += dw
            Wo[t] = sh * o[t] / Vo if Vo > 0 else 0.0
        else:
            if not in_pos and reent[t - 1]:
                Vo -= slip * Vo
                sh, cash = Vo / o[t], 0.0
                in_pos, peak = True, o[t]
                turn[t] += 1.0
            Wo[t] = 1.0 if in_pos else 0.0
            if in_pos:
                L = peak * (1 - stop_x)
                px = o[t] if o[t] <= L else (L if l[t] <= L else None)
                if px is not None:
                    v = cash + sh * px
                    cash, sh = v * (1 - slip), 0.0
                    in_pos = False
                    turn[t] += 1.0
        Vc[t] = cash + sh * c[t]
        Wc[t] = sh * c[t] / Vc[t] if Vc[t] > 0 else 0.0
        if stop_x is not None and in_pos:
            peak = max(peak, c[t])
    return Vc, Wc, Wo, turn


def fixed_curves(P: S.Bars, W=WGRID, slip=0.0, cash_rate=0.0):
    """덜 담기 w(배열) 각각을 run()과 같은 방식(매일 시가에 w로 맞춤)으로. 반환 Vc[len(W), n]."""
    W = np.asarray(W, float)
    n = len(P.c)
    cash, sh = 1 - W, W / P.c[0]
    Vc = np.empty((len(W), n))
    Vc[:, 0] = 1.0
    for t in range(1, n):
        cash = cash * (1 + cash_rate / 252)
        Vo = cash + sh * P.o[t]
        cur = np.where(Vo > 0, sh * P.o[t] / np.where(Vo > 0, Vo, 1), 0.0)
        Vo = Vo - slip * np.abs(W - cur) * Vo
        sh = W * Vo / P.o[t]
        cash = Vo - sh * P.o[t]
        Vc[:, t] = cash + sh * P.c[t]
    return Vc


# ----------------------------------------------------------------------------- 지표
def _mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def span(days, d0, d1, grace_days=30):
    """창 [d0, d1]의 (a, z). a−1일 종가가 기준. 상품 자료가 창 시작보다 grace_days 넘게 늦게 시작하면 None."""
    a, z = int(np.searchsorted(days, d0)), int(np.searchsorted(days, d1, side="right")) - 1
    a = max(a, 1)
    if z < a:
        return None
    if pd.Timestamp(days[a - 1]) - pd.Timestamp(d0) > pd.Timedelta(days=grace_days):
        return None
    return a, z


def cagr(ret, ndays):
    return float((1 + ret) ** (252 / max(ndays, 1)) - 1) if ret > -1 else -1.0


def underwater(v):
    """가장 긴 전고점 아래 기간(거래일)과 그 기간이 창 안에서 회복으로 끝났는지."""
    under = v < np.maximum.accumulate(v) * (1 - 1e-12)
    best = cur = 0
    best_end_open = False
    for i, x in enumerate(under):
        cur = cur + 1 if x else 0
        if cur > best:
            best = cur
            best_end_open = (i == len(under) - 1)
        elif cur == best and cur > 0 and i == len(under) - 1:
            best_end_open = True
    return best, not best_end_open


_FC = {}


def frontier_stats(frontier, a, z):
    key = (id(frontier), a, z)
    if key not in _FC:
        F = frontier[:, a - 1:z + 1]
        F = F / F[:, :1]
        _FC[key] = (F[:, -1] - 1, (F / np.maximum.accumulate(F, axis=1) - 1).min(axis=1))
    return _FC[key]


def wstat(Vc, Wc, turn, days, d0, d1, frontier=None):
    sp = span(days, d0, d1)
    if sp is None:
        return None
    a, z = sp
    seg = Vc[a - 1:z + 1] / Vc[a - 1]
    nd = z - a + 1
    ret, mdd = float(seg[-1] - 1), _mdd(seg)
    yrs = nd / 252
    out = {"시작": str(days[a - 1]), "끝": str(days[z]), "일수": nd, "수익": ret, "낙폭": mdd, "연율": cagr(ret, nd),
           "노출": float(Wc[a:z + 1].mean()), "연매매": float((turn[a:z + 1] >= 0.05).sum() / yrs),
           "연회전": float(turn[a:z + 1].sum() / yrs)}
    out["칼마"] = float(out["연율"] / -mdd) if mdd < 0 else float("nan")
    if frontier is not None:
        fr, fm = frontier_stats(frontier, a, z)
        ok = fm >= mdd - 1e-12
        j = int(np.argmax(np.where(ok, fr, -np.inf)))
        long = nd > 300
        if long:
            out["효율선초과"] = out["연율"] - cagr(float(fr[j]), nd)
        else:
            out["효율선초과"] = ret - float(fr[j])
        out["효율선_w"] = float(WGRID[j])
        out["효율선_수익"] = float(fr[j])
        k = int(round(out["노출"] * 100))
        out["같은노출_w"] = float(WGRID[k])
        out["같은노출차"] = (out["연율"] - cagr(float(fr[k]), nd)) if long else ret - float(fr[k])
        if long:
            out["효율선_연율"] = cagr(float(fr[j]), nd)
    if nd > 300:
        r1 = Vc[a - 1 + 252:z + 1] / Vc[a - 1:z + 1 - 252] - 1
        out["최악1년"] = float(r1.min()) if len(r1) else float("nan")
        uw, rec = underwater(seg)
        out["최장회복_년"] = uw / 252
        out["끝에회복"] = rec
        out["전멸"] = bool(mdd <= -0.90)
    return out


# ----------------------------------------------------------------------------- 실행
def build():
    u = lev5.bars("EWY")
    syn = lev5.synth(u, 3, FEE, EXTRA)
    real = lev5.bars("KORU")
    return u, syn, real


def align(u: S.Bars, P: S.Bars, arr):
    idx = np.searchsorted(u.days, P.days)
    assert np.all(u.days[idx] == P.days)
    return arr[idx]


def run_rule(u, P, ind, rule, slip=0.0, cash_rate=0.0):
    rid, name, fam, p = rule
    if fam == "S":
        s50 = ind["sma"][50]
        re = ~(np.isfinite(s50) & (ind["c"] <= s50))       # 지표 없으면 재진입 허용(그냥 보유 기본)
        return run(P, stop_x=p["x"], reent=align(u, P, re), slip=slip, cash_rate=cash_rate)
    return run(P, des=align(u, P, desired(fam, p, ind)), slip=slip, cash_rate=cash_rate)


def evaluate(u, P, ind, rules, wins, slip=0.0, cash_rate=0.0):
    fr = fixed_curves(P, WGRID, slip, cash_rate)
    out = {}
    for rule in rules:
        Vc, Wc, Wo, turn = run_rule(u, P, ind, rule, slip, cash_rate)
        out[rule[0]] = {w: wstat(Vc, Wc, turn, P.days, *WIN[w], frontier=fr) for w in wins}
    return out, fr


def leak_check(u, rules):
    """t 이후 EWY 가격을 흔들어도 t까지의 가치와 t일 시가 체결 비율이 같아야 한다(종가 신호 → 다음 시가 체결 확인)."""
    rng = np.random.default_rng(7)
    T = u.idx("2020-03-02")
    u2 = S.Bars(u.sym, u.days, u.o.copy(), u.h.copy(), u.l.copy(), u.c.copy(), u.v)
    shock = 1 + rng.normal(0, 0.04, len(u.c) - T)
    u2.c[T:] *= shock
    u2.o[T + 1:] *= shock[1:]
    u2.h[T:] = np.maximum.reduce([u2.h[T:] * shock, u2.o[T:], u2.c[T:]])
    u2.l[T:] = np.minimum.reduce([u2.l[T:] * shock, u2.o[T:], u2.c[T:]])
    P1, P2 = lev5.synth(u, 3, FEE, EXTRA), lev5.synth(u2, 3, FEE, EXTRA)
    i1, i2 = indicators(u), indicators(u2)
    worst_v, worst_w = 0.0, 0.0
    for rule in rules:
        a = run_rule(u, P1, i1, rule)
        b = run_rule(u2, P2, i2, rule)
        worst_v = max(worst_v, float(np.max(np.abs(a[0][:T] - b[0][:T]))))
        worst_w = max(worst_w, float(np.max(np.abs(a[2][:T + 1] - b[2][:T + 1]))))
    return {"교란시작": str(u.days[T]), "T전_가치_최대차": worst_v, "T일시가비율_최대차": worst_w}


def engine_check(u, P, ind, rules):
    Vc = run(P, des=np.ones(len(P.c)))[0]
    d0 = float(np.max(np.abs(Vc - P.c / P.c[0])))
    fr = fixed_curves(P, [0.5])
    v5 = run(P, des=np.full(len(P.c), 0.5))[0]
    return {"보유=상품": d0, "덜담기0.5_엔진차": float(np.max(np.abs(fr[0] - v5) / v5))}


def cycle_stats(u, P, ind, rule):
    """추적 손절 주기: 판 날 종가 대비 다시 산 시가(+면 더 비싸게 되삼), 밖에 있던 거래일."""
    Vc, Wc, Wo, turn = run_rule(u, P, ind, rule)
    pos = Wc > 1e-9
    exits = [t for t in range(1, len(pos)) if pos[t - 1] and not pos[t]]
    entries = [t for t in range(1, len(pos)) if not pos[t - 1] and Wo[t] > 0.5]
    g, d = [], []
    for e in exits:
        nxt = [t for t in entries if t > e]
        if nxt:
            g.append(P.o[nxt[0]] / P.c[e] - 1)
            d.append(nxt[0] - e)
    g = np.array(g)
    return {"주기": int(len(g)), "더비싸게_비율": float((g > 0).mean()) if len(g) else float("nan"),
            "중앙_재진입차": float(np.median(g)) if len(g) else float("nan"), "중앙_대기일": float(np.median(d)) if d else float("nan")}


def main():
    u, syn, real = build()
    ind = indicators(u)
    rules = rule_book()
    val_raw = lev5.validate("KORU", lev5.synth(u, 3, FEE, 0.0))
    val_cal = lev5.validate("KORU", syn)
    wins_all = list(WIN.keys())
    print("대조(추가비용 0)", {k: round(v, 3) if isinstance(v, float) else v for k, v in val_raw.items()})
    print("대조(추가비용 2.6%)", {k: round(v, 3) if isinstance(v, float) else v for k, v in val_cal.items()})
    checks = {"엔진": engine_check(u, syn, ind, rules), "미래교란": leak_check(u, rules)}
    print("점검", checks)

    res = {}
    for prod, P in (("합성", syn), ("실제", real)):
        r0, fr0 = evaluate(u, P, ind, rules, wins_all)
        r5, _ = evaluate(u, P, ind, rules, [PICK, CHECK, AUX, "긴역사", "긴역사_앞이전", "준비후", "실제기간", "2022", "2020"],
                         slip=SLIP)
        r4, _ = evaluate(u, P, ind, rules, [PICK, CHECK, AUX, "긴역사", "실제기간"], cash_rate=CASH4)
        res[prod] = {"기본": r0, "미끄러짐0.5%": r5, "현금4%": r4}
        # 덜 담기 효율선 자체(연율 최대 w)
        if prod == "합성":
            sp = span(P.days, *WIN["긴역사"])
            a, z = sp
            nd = z - a + 1
            seg = fr0[:, a - 1:z + 1] / fr0[:, a - 1:a]
            cg = np.array([cagr(float(x - 1), nd) for x in seg[:, -1]])
            md = (seg / np.maximum.accumulate(seg, axis=1) - 1).min(axis=1)
            res["효율선_긴역사"] = {f"{w:.2f}": {"연율": float(cg[i]), "낙폭": float(md[i])} for i, w in enumerate(WGRID)}
    # EWY 1배(참고)
    ewy = {}
    Ve = u.c / u.c[0]
    for w in [PICK, CHECK, AUX, "2022", "2020", "2008위기", "2000~02", "긴역사", "AI쇼크", "2024하반기"]:
        ewy[w] = wstat(Ve, np.ones(len(Ve)), np.zeros(len(Ve)), u.days, *WIN[w])
    res["EWY1배"] = ewy

    sel = select(res, rules)
    res["손절주기"] = {prod: {r[0]: cycle_stats(u, P, ind, r) for r in rules if r[2] == "S"}
                     for prod, P in (("합성", syn), ("실제", real))}
    payload = {"규칙": [{"id": r[0], "이름": r[1], "계열": r[2], "값": r[3]} for r in rules], "창": WIN, "대조": {"추가0": val_raw,
               "추가2.6%": val_cal}, "점검": checks, "결과": res, "고르기": sel,
               "설정": {"보수": FEE, "추가비용": EXTRA, "미끄러짐": SLIP, "현금이자_민감도": CASH4, "w간격": 0.01,
                      "중앙값_최소개수": MEDIAN_MIN}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "p5_KORU.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_js), encoding="utf-8")
    md = report(res, rules, sel, checks, val_raw, val_cal)
    (RESULTS / "p5_KORU.md").write_text(md, encoding="utf-8")
    print(md)


def _js(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


ACTIVE = lambda rules: [r for r in rules if r[2] not in ("R", "F")]


def select(res, rules):
    act = ACTIVE(rules)
    R, Sy = res["실제"]["기본"], res["합성"]["기본"]
    exc = lambda src, rid, w: src[rid][w]["효율선초과"] if src[rid].get(w) else float("nan")
    g1 = max(act, key=lambda r: exc(R, r[0], PICK))[0]
    g2 = max(act, key=lambda r: exc(Sy, r[0], "긴역사_앞이전"))[0]
    g3 = max(act, key=lambda r: R[r[0]][PICK]["칼마"])[0]
    g2b = max(act, key=lambda r: exc(Sy, r[0], "준비후_앞이전"))[0]
    passed = [r[0] for r in act if exc(R, r[0], PICK) > 1e-9]
    allr = [r[0] for r in act]
    same = lambda src, rid, w: src[rid][w]["같은노출차"] if src[rid].get(w) else float("nan")
    passed_same = [r[0] for r in act if same(R, r[0], PICK) > 1e-9]

    def rate(ids, src, w, f=exc):
        v = [f(src, i, w) for i in ids]
        v = [x for x in v if np.isfinite(x)]
        return (sum(x > 1e-9 for x in v), len(v))
    out = {"G1": g1, "G2": g2, "G3": g3, "G2_준비후": g2b, "앞통과": passed, "앞통과_같은노출": passed_same,
           "성공": {w: {"통과집합": rate(passed, src, w), "전체": rate(allr, src, w)}
                  for w, src in ((CHECK, R), (AUX, R), ("긴역사", Sy), ("준비후", Sy), ("실제기간", R))},
           "성공_같은노출": {w: {"통과집합": rate(passed_same, src, w, same), "전체": rate(allr, src, w, same)}
                       for w, src in ((CHECK, R), (AUX, R), ("긴역사", Sy), ("준비후", Sy), ("실제기간", R))},
           "합성으로고른G1": max(act, key=lambda r: exc(Sy, r[0], PICK))[0]}
    return out


# ----------------------------------------------------------------------------- 보고서
def pct(x, d=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def rm(s):
    if not s:
        return "—"
    return f"{pct(s['수익'], 0 if abs(s['수익']) >= 1 else 1)}/{pct(s['낙폭'], 0, False)}"


def report(res, rules, sel, checks, val_raw, val_cal):
    R, Sy = res["실제"]["기본"], res["합성"]["기본"]
    R5, S5 = res["실제"]["미끄러짐0.5%"], res["합성"]["미끄러짐0.5%"]
    S4 = res["합성"]["현금4%"]
    name = {r[0]: r[1] for r in rules}
    L = []
    A = L.append
    g1, g2, g3, g2b = sel["G1"], sel["G2"], sel["G3"], sel["G2_준비후"]

    def ex(src, rid, w):
        s = src[rid].get(w)
        return s["효율선초과"] if s else float("nan")

    lg = lambda rid: Sy[rid]["긴역사"]
    rl = lambda rid: R[rid]["실제기간"]
    fx = res["효율선_긴역사"]
    best_w = max(fx.items(), key=lambda kv: kv[1]["연율"])
    sc, ss = sel["성공"], sel["성공_같은노출"]
    passed = sel["앞통과"]
    fam_of = {r[0]: r[2] for r in rules}
    n_vol_pass = sum(fam_of[i] in ("V", "VS", "TV") for i in passed)
    s_ids = [r[0] for r in rules if r[2] == "S"]
    alive = [r[0] for r in rules if not lg(r[0])["전멸"]]
    rng = lambda xs: f"{pct(min(xs))}~{pct(max(xs))}"
    rng0 = lambda xs: f"{min(xs) * 100:.0f}~{max(xs) * 100:.0f}%"

    A("# 5단계 KORU: 한국 3배에 손절·추세·변동성 규칙 (2026-10-11)")
    A("")
    A("> 사용자: \"SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?\"")
    A(f"> 기초 EWY(2000-05-12~), 상품 = EWY 3배 합성(보수 0.95% + 추가 비용 연 2.6%, 실제 KORU와 일수익 상관 {val_cal['일수익상관']:.2f}) 2000~ · 실제 KORU 2013-04~. "
      "신호는 EWY 종가 → 다음 시가 체결, 나머지 현금 0%, 수수료 0. 스크립트 `code/p5_KORU.py`, 숫자 `results/p5_KORU.json`.")
    A("> 2026 앞·뒤·2025·2022·2020·실제 2013~ = 실제 KORU, 2008·2000~02·긴 역사 = 합성. "
      "효율선초과 = 규칙 − '같은 최대 낙폭 이하로 KORU를 덜 담고(나머지 현금) 매일 맞추기' 중 최고.")
    A("")
    A("## 결론")
    A("")
    A(f"1. **2026 기준 기각.** 21개 중 2026 앞(1/2~5/29)에서 효율선을 이긴 {len(passed)}개(변동성 계열 {n_vol_pass}개) 가운데 뒤(6/1~10/9)에서도 이긴 것은 "
      f"{sc[CHECK]['통과집합'][0]}개다. 뒤는 실제 KORU {pct(R['R0'][CHECK]['수익'])}라 효율선이 현금이었고 플러스를 낸 규칙이 없었다. "
      f"G1 '{name[g1]}'은 앞 {pct(ex(R, g1, PICK))}p → 뒤 {pct(ex(R, g1, CHECK))}p. 같은 평균 노출의 덜 담기 대비로 봐도 앞 통과 "
      f"{len(sel['앞통과_같은노출'])}개 중 뒤 {ss[CHECK]['통과집합'][0]}개·긴 역사 {ss['긴역사']['통과집합'][0]}개가 이겼다"
      f"(고르지 않은 21개는 {ss[CHECK]['전체'][0]}·{ss['긴역사']['전체'][0]}개).")
    A(f"2. **추적 손절은 KORU에서 가장 나빴다.** 추적 −30·−40·−50%(EWY 50일선 위 재진입)는 2000~2026 연 {rng([lg(i)['연율'] for i in s_ids])}, "
      f"최대 낙폭 {pct(min(lg(i)['낙폭'] for i in s_ids), 1, False)}~{pct(max(lg(i)['낙폭'] for i in s_ids), 1, False)}로 모두 전멸했고, "
      f"효율선보다 연 {rng([ex(Sy, i, '긴역사') for i in s_ids])}p, 실제 KORU 2013~ 연 {rng([ex(R, i, '실제기간') for i in s_ids])}p 뒤졌다. "
      f"판 뒤 다시 산 값이 판 날 종가보다 비싼 주기가 {rng0([res['손절주기']['합성'][i]['더비싸게_비율'] for i in s_ids])}"
      f"(합성 {sum(res['손절주기']['합성'][i]['주기'] for i in s_ids)}주기, 중앙 {rng([res['손절주기']['합성'][i]['중앙_재진입차'] for i in s_ids])} 비쌈; "
      f"실제 {sum(res['손절주기']['실제'][i]['주기'] for i in s_ids)}주기 {rng0([res['손절주기']['실제'][i]['더비싸게_비율'] for i in s_ids])})였다.")
    A(f"3. **전멸을 막은 것은 '덜 담기'와 띠 있는 추세 필터뿐이다.** 그냥 보유는 연 {pct(lg('R0')['연율'])}/{pct(lg('R0')['낙폭'], 1, False)}"
      f"(실제 2013~ {pct(rl('R0')['연율'])}/{pct(rl('R0')['낙폭'], 0, False)}). 덜 담기는 w={best_w[0]}에서 연율이 가장 높았고"
      f"({pct(best_w[1]['연율'])}/{pct(best_w[1]['낙폭'], 0, False)}), w=0.33은 {pct(fx['0.33']['연율'])}/{pct(fx['0.33']['낙폭'], 0, False)}. "
      f"긴 역사로 고른 G2 '{name[g2]}'는 연 {pct(lg(g2)['연율'])}/{pct(lg(g2)['낙폭'], 0, False)}(효율선 대비 연 {pct(ex(Sy, g2, '긴역사'))}p, "
      f"실제 2013~ {pct(ex(R, g2, '실제기간'))}p)였지만 2026 앞 {pct(ex(R, g2, PICK))}p·뒤 {pct(ex(R, g2, CHECK))}p로 졌고, "
      f"시작을 지표 준비 후(2001-03)로 옮기면 1등이 '{name[g2b]}'로 바뀐다(그 규칙 2025 {pct(ex(R, g2b, AUX))}p, 실제 2013~ 연 {pct(ex(R, g2b, '실제기간'))}p). "
      f"7월 쇼크(실제 {pct(R['R0']['AI쇼크']['수익'])})에는 EWY가 200일선 위라 200일선 규칙이 켜지지 않았다.")
    A("")
    # ---------- 사용자 표
    A("## 사용자 표 (수익/최대 낙폭)")
    A("")
    A("| 규칙 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 2020 | 2008 위기 | 2000~02 | 실제 KORU 2013~ 연율/낙폭 | 긴 역사 연율/낙폭 | 전멸 | 최악 1년 | 연 매매 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rules:
        rid = r[0]
        s = lg(rid)
        tag = "".join(f" **({g})**" for g, x in (("G1", g1), ("G2", g2), ("G3", g3)) if x == rid)
        A(f"| {name[rid]}{tag} | {rm(R[rid][PICK])} | {rm(R[rid][CHECK])} | {rm(R[rid][AUX])} | {rm(R[rid]['2022'])} | "
          f"{rm(R[rid]['2020'])} | {rm(Sy[rid]['2008위기'])} | {rm(Sy[rid]['2000~02'])} | {pct(rl(rid)['연율'])}/{pct(rl(rid)['낙폭'], 0, False)} | "
          f"{pct(s['연율'])}/{pct(s['낙폭'], 0, False)} | {'예' if s['전멸'] else '아니오'} | {pct(s['최악1년'], 0)} | {s['연매매']:.1f} |")
    e = res["EWY1배"]
    A(f"| (참고) EWY 1배 보유 | {rm(e[PICK])} | {rm(e[CHECK])} | {rm(e[AUX])} | {rm(e['2022'])} | {rm(e['2020'])} | "
      f"{rm(e['2008위기'])} | {rm(e['2000~02'])} | — | {pct(e['긴역사']['연율'])}/{pct(e['긴역사']['낙폭'], 0, False)} | 아니오 | "
      f"{pct(e['긴역사']['최악1년'], 0)} | 0 |")
    A("")
    A("- 긴 역사 = 합성 2000-05-12~2026-10-09. 전멸 = 최대 낙폭 −90% 이하. 연 매매 = 비율을 5%p 이상 바꾼 날 수/년(긴 역사). "
      "매일 맞춤(덜 담기·변동성 맞춤)은 작은 조정이 매일 생기고 그 비용은 미끄러짐 민감도에 넣었다.")
    A("- 2000~02 칸: 150·200일선과 변동성 중앙값 규칙은 2000년에 지표가 아직 없어 그냥 보유였다(EWY 자료 2000-05-12 시작). 50일선은 2000-07부터 켜졌다.")
    A("")
    # ---------- 효율선 표
    A("## 효율선 표 (효율선초과; 짧은 창은 수익 차 %p, 긴 창은 연율 차 %p)")
    A("")
    A("| 규칙 | 2026 앞 | 2026 뒤 | 2025 | 긴 역사 2000-05~2025(연) | 지표 준비 후 2001-03~2025(연) | 긴 역사 전체(연) | 실제 KORU 2013~(연) | "
      "같은 평균 노출 대비 앞·뒤·긴(연) | 미끄러짐 0.5%: 앞·뒤·긴(연) | 긴 역사 평균 노출 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rules:
        rid = r[0]
        if r[2] == "R":
            continue
        A(f"| {name[rid]} | {pct(ex(R, rid, PICK))} | {pct(ex(R, rid, CHECK))} | {pct(ex(R, rid, AUX))} | "
          f"{pct(ex(Sy, rid, '긴역사_앞이전'))} | {pct(ex(Sy, rid, '준비후_앞이전'))} | {pct(ex(Sy, rid, '긴역사'))} | {pct(ex(R, rid, '실제기간'))} | "
          f"{pct(R[rid][PICK]['같은노출차'])} · {pct(R[rid][CHECK]['같은노출차'])} · {pct(Sy[rid]['긴역사']['같은노출차'])} | "
          f"{pct(ex(R5, rid, PICK))} · {pct(ex(R5, rid, CHECK))} · {pct(ex(S5, rid, '긴역사'))} | {lg(rid)['노출']:.2f} |")
    A("")
    A("덜 담기 효율선(합성, 2000-05~2026-10, 매일 맞춤): " + ", ".join(
        f"w={w} 연 {pct(fx[w]['연율'])}/{pct(fx[w]['낙폭'], 0, False)}" for w in ("0.10", "0.20", "0.33", "0.40", "0.50", "0.67", "1.00")) +
        f". 연율 최대 w={best_w[0]}. 2026 뒤는 모든 w가 손실이라 효율선 = 현금(w=0)이다.")
    A("")
    # ---------- 고르기 표
    A("## 고르기 (앞으로 골라 뒤·2025·긴 역사 확인)")
    A("")
    A("| 방법 | 고른 것 | 고른 값 | 2026 뒤 효율선초과 (수익/낙폭) | 2025 효율선초과 | 긴 역사 전체 연율/낙폭 (효율선초과 연) | 실제 KORU 2013~ 효율선초과(연) |")
    A("|---|---|---|---|---|---|---|")
    rows = (("G1 2026 앞 효율선초과 최대", g1, f"앞 {pct(ex(R, g1, PICK))}p"),
            ("G2 긴 역사(2000-05~2025-12) 효율선초과 최대", g2, f"연 {pct(ex(Sy, g2, '긴역사_앞이전'))}p"),
            ("G2' 지표 준비 후(2001-03~2025-12) 최대", g2b, f"연 {pct(ex(Sy, g2b, '준비후_앞이전'))}p"),
            ("G3 2026 앞 칼마 최대", g3, f"앞 칼마 {R[g3][PICK]['칼마']:.0f}(보유 {R['R0'][PICK]['칼마']:.0f})"))
    for gname, rid, how in rows:
        A(f"| {gname} | {name[rid]} | {how} | {pct(ex(R, rid, CHECK))}p ({rm(R[rid][CHECK])}) | {pct(ex(R, rid, AUX))}p | "
          f"{pct(lg(rid)['연율'])}/{pct(lg(rid)['낙폭'], 0, False)} ({pct(ex(Sy, rid, '긴역사'))}p) | {pct(ex(R, rid, '실제기간'))}p |")
    A("")
    c0 = R["R0"][PICK]["칼마"]
    ties = [r[0] for r in rules if r[2] not in ("R", "F") and abs(R[r[0]][PICK]["칼마"] - c0) < 1e-9 * abs(c0)]
    A(f"- G3: 2026 앞에 한 번도 켜지지 않은 {len(ties)}개({', '.join(name[i] for i in ties)})가 그냥 보유와 칼마 동률이고, "
      f"부동소수 차로 '{name[g3]}'가 뽑혔다. 사실상 '그냥 보유'를 고른 것이다. 합성으로 2026 앞을 고르면 G1 = {name[sel['합성으로고른G1']]}.")
    A("")
    A("'앞 > 0' 집합의 성공 비율 vs 고르지 않은 21개 전체:")
    A("")
    A("| 확인 창 | 효율선초과: 앞 통과 집합 | 효율선초과: 전체 | 같은 평균 노출 대비: 앞 통과 집합 | 같은 평균 노출 대비: 전체 |")
    A("|---|---|---|---|---|")
    for w, lab in ((CHECK, "2026 뒤(실제)"), (AUX, "2025(실제)"), ("긴역사", "긴 역사 전체(합성)"), ("준비후", "지표 준비 후 2001-03~(합성)"),
                   ("실제기간", "실제 KORU 2013~")):
        a, b, c, d = sc[w]["통과집합"], sc[w]["전체"], ss[w]["통과집합"], ss[w]["전체"]
        A(f"| {lab} | {a[0]}/{a[1]} | {b[0]}/{b[1]} | {c[0]}/{c[1]} | {d[0]}/{d[1]} |")
    A("")
    A(f"- 효율선초과 앞 통과: {', '.join(name[i] for i in passed) or '없음'}.")
    A(f"- 같은 평균 노출 대비 앞 통과: {', '.join(name[i] for i in sel['앞통과_같은노출']) or '없음'}.")
    A("")
    # ---------- 폭락 구간
    show = list(dict.fromkeys(["R0", "F33", "F50", "T50b", "T200", "T200b", "V45", "VS15", "TV", "S30", "S40", g1, g2, g3]))
    A("## 폭락·반등 구간 (수익/최대 낙폭; 2013-04 이후 실제 KORU, 이전 합성) — 날짜는 사후에 정해 고르기에 쓰지 않음")
    A("")
    A("| 구간 | " + " | ".join(name[i] for i in show) + " |")
    A("|---|" + "---|" * len(show))
    for w in CRASH:
        src = R if WIN[w][0] >= "2013-04-11" else Sy
        A(f"| {CRASH_NAME[w]} | " + " | ".join(rm(src[i][w]) for i in show) + " |")
    A("")
    # ---------- 해별
    A("## 해별 수익 (합성; 2014~ 괄호 = 실제 KORU)")
    A("")
    A("| 해 | " + " | ".join(name[i] for i in show) + " |")
    A("|---|" + "---|" * len(show))
    for y in range(2000, 2027):
        y = str(y)
        cells = []
        for i in show:
            s = Sy[i].get(y)
            rr = R[i].get(y) if int(y) >= 2014 else None
            cells.append((f"{pct(s['수익'], 0)}" + (f" ({pct(rr['수익'], 0)})" if rr else "")) if s else "—")
        A(f"| {y}{'(5/12~)' if y == '2000' else ''} | " + " | ".join(cells) + " |")
    A("")
    # ---------- 긴 역사 세부
    A("## 긴 역사 세부 (합성 2000-05-12~2026-10-09)")
    A("")
    A("| 규칙 | 연율 | 최대 낙폭 | 전멸(≤−90%) | 최악 1년 | 가장 긴 전고점 아래(년) | 지표 준비 후 2001-03~ 연율/낙폭 | 평균 노출 | 연 매매 | 연 회전 | "
      "미끄러짐 0.5% 연율 | 현금 4% 연율 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rules:
        rid = r[0]
        s = lg(rid)
        p2 = Sy[rid]["준비후"]
        A(f"| {name[rid]} | {pct(s['연율'])} | {pct(s['낙폭'], 1, False)} | {'예' if s['전멸'] else '아니오'} | {pct(s['최악1년'], 0)} | "
          f"{s['최장회복_년']:.1f}{'' if s['끝에회복'] else '(미회복)'} | {pct(p2['연율'])}/{pct(p2['낙폭'], 0, False)} | {s['노출']:.2f} | "
          f"{s['연매매']:.1f} | {s['연회전']:.1f} | {pct(S5[rid]['긴역사']['연율'])} | {pct(S4[rid]['긴역사']['연율'])} |")
    A("")
    A(f"- 전멸을 피한 것: {', '.join(name[i] for i in alive)}.")
    A("")
    # ---------- 한계
    A("## 시험 범위·한계")
    A("")
    A(f"- 시험 범위: 규칙 21개(T 9·V 6·VS 2·TV 1·S 3) + 그냥 보유 + 덜 담기 3, 상품 2(합성·실제), 효율선 w 101개(0.01 간격), "
      f"창 {len(WIN)}개(해별 27 포함). 고르기 3종(G1·G2·G3)과 확인용 G2'. 규칙·평가는 결과를 보기 전에 고정했다. "
      "21개 중 최대를 고르는 선택편향이 있다. '지표 준비 후' 창(2001-03~)은 첫 결과를 본 뒤 2000년 준비 기간 효과를 떼어 보려고 더했다.")
    A(f"- 합성 대조: 추가 비용 0이면 실제 KORU보다 연 {val_raw['연비용_로그'] * 100:.1f}% 덜 들고, 2.6%를 넣으면 차 {val_cal['연비용_로그'] * 100:+.1f}%, "
      f"일수익 상관 {val_cal['일수익상관']:.3f}, 최대 낙폭 실제 {val_cal['실제_낙폭'] * 100:.1f}% vs 합성 {val_cal['합성_낙폭'] * 100:.1f}%. "
      "상관이 0.98에 그치는 이유: KORU의 3배 재설정은 한국 장 마감 MSCI Korea 25/50 지수 기준인데 합성은 미국 장 EWY 종가 기준이다"
      "(2020-03-16 실제 −27% vs 합성 −47%, 다음 날 반대). 2013~2016 실제 KORU는 거래가 적어 가격이 멈춘 날이 있다. "
      f"2026년은 실제가 합성보다 뒤처졌다(2026 합성 {pct(Sy['R0']['2026']['수익'], 0)} vs 실제 {pct(R['R0']['2026']['수익'], 0)}).")
    A("- 환율: EWY·KORU는 달러 상품이라 원화 환율이 섞여 있다(2008 원화 급락이 EWY 낙폭을 키웠다). 원화로 사고파는 사람의 손익은 환율만큼 또 다르다.")
    A("- 2000~02는 EWY 자료가 2000-05-12부터라 KOSPI 2000년 1월 고점부터는 못 잰다. 지표 준비 기간(50~200일, 변동성 중앙값 σ20 60개)에는 그냥 보유로 두었다. "
      "2000년 일부 날(11%)은 시가=고가=저가=종가라 그날 장중 손절은 시가에 처리된다.")
    A("- 현금 이자 0(보수적). 현금 4% 판은 긴 역사 세부 표 끝 열. 프리장·주간거래의 손절 체결, 호가 스프레드는 미끄러짐 0.5%로만 대신했다. "
      "손절가 아래로 열린 날은 시가에 체결했다.")
    A(f"- 시점 점검: 그냥 보유=상품 차 {checks['엔진']['보유=상품']:.1e}, 덜 담기 0.5 두 엔진 차 {checks['엔진']['덜담기0.5_엔진차']:.1e}. "
      f"{checks['미래교란']['교란시작']}부터 EWY를 흔들어도 그 전 가치 최대 차 {checks['미래교란']['T전_가치_최대차']:.1e}, "
      f"그날 시가 체결 비율 최대 차 {checks['미래교란']['T일시가비율_최대차']:.1e}(둘 다 0이어야 함).")
    A("- 매수 추천·주문 수량이 아니다. 규칙과 비율 w까지만 잰다.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
