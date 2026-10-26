"""급등 후 눌림목 매수 연구(2026-10-10 사용자 질문, 키 surge_pullback).

질문(사용자): 급등 후 눌림목에 사는 전략은 어떤가.
자료: build_daily.py 정규장 일봉(현재 상장·시총 3억 달러 이상 3,564종목, 2023-11~2026-10-06). 장외 봉은 쓰지 않는다.
급등 사건(날 e): 정규장 종가 기준 전날 대비 +10% 이상(하위 묶음 +15%·+20%, 거래량 20일 평균의 2배 이상).
  거르기: 전날 종가 >= $3, 직전 20일 평균 거래대금 >= 200만 달러, 하루 +100% 이하(자료의 병합·분할 흔적 배제), 정규장 봉 4개 이상.
  한 종목에서 사건 뒤 10거래일 안의 새 급등은 같은 사건으로 본다(겹치지 않게).
눌림목 매수(사건 뒤 1~10거래일, 사건 하나에 한 번). B = 급등 전날 종가, H = 사건 날부터 '전날'까지의 최고가(그날 아는 값).
  전날까지 종가가 B 아래로 떨어진 적이 있으면 급등이 지워진 것으로 보고 더 사지 않는다(실패 취소).
  F382·F50·F618: 지정가 H - f x (H - B), 저가가 닿으면 min(시가, 지정가). F50이 기본(가장 흔한 '눌림목' 정의, 미리 정함).
  EMA10·EMA20: 전날 종가 기준 일봉 EMA에 지정가(전날 종가가 EMA 위일 때만), 닿으면 min(시가, EMA).
  REV: 사건 뒤 이틀 연속 종가 하락 다음 날 종가가 그 전날 고가를 넘으면(반전) 다음 날 시가.
  RSI50: 일봉 RSI14가 사건 뒤 70 이상을 찍은 뒤 50 이하로 마감하면 다음 날 시가.
비교 매수: M0 급등 날 종가(추격), M1 다음 날 시가, D1~D10 = 사건 뒤 d번째 날 시가(같은 사건 '아무 날' 잣대).
매도: H5·H10·H20 = 매수 날을 포함하지 않고 N거래일 뒤 종가.
  TGT = 전 고점(매수 때의 H, M0·M1은 사건 날 고가) 지정가 매도, 종가가 B 아래면 다음 날 시가 손절, 최대 20거래일 뒤 종가.
  자료 끝까지 안 팔렸으면 마지막 종가로 평가(미청산).
시점 원칙: 지정가는 전날까지의 값(H, EMA)으로 정하고, 신호(REV, RSI50)는 마감 확정 뒤 다음 날 시가에 산다.
사용: py study.py → <scratchpad>/sp_trades.parquet(거래표), results/study.txt(요약)
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
AN = os.path.normpath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(AN, "kalman_zones", "code"))
from osc import rsi_state

SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/6a57b230-3ac1-4ae6-9d34-b1d8a1c2ce1b/scratchpad"
X_MIN, W = 0.10, 10
ENTRY = ["M0", "M1", "F382", "F50", "F618", "EMA10", "EMA20", "REV", "RSI50"] + [f"D{d}" for d in range(1, W + 1)]
EXIT = ["H5", "H10", "H20", "TGT"]


@njit(cache=True)
def _exit(o, h, c, t, P, B, tgt, ex):
    """매수 봉 t·가격 P → (매도 봉, 매도가, 미청산). ex: 0 H5 / 1 H10 / 2 H20 / 3 TGT."""
    n = len(c)
    if ex < 3:
        N = (5, 10, 20)[ex]
        xi = t + N
        if xi >= n:
            return n - 1, c[n - 1], True
        return xi, c[xi], False
    tg = max(tgt, P * 1.0001)
    for j in range(t + 1, t + 21):
        if j >= n:
            return n - 1, c[n - 1], True
        if o[j] >= tg:
            return j, o[j], False
        if h[j] >= tg:
            return j, tg, False
        if c[j] < B:
            if j + 1 >= n:
                return n - 1, c[n - 1], True
            return j + 1, o[j + 1], False
        if j == t + 20:
            return j, c[j], False
    return n - 1, c[n - 1], True


@njit(cache=True)
def sim(o, h, l, c, e10, e20, r14, ev, W):
    """사건 봉 배열 ev → 행: 사건 순번, 매수 규칙 번호, 매도 규칙 번호, 매수 봉, 매수가, 매도 봉, 매도가, 미청산, 목표가."""
    n = len(c)
    cap = len(ev) * 19 * 4
    EV = np.empty(cap, np.int64); EN = np.empty(cap, np.int64); EX = np.empty(cap, np.int64)
    TI = np.empty(cap, np.int64); PI = np.empty(cap); XI = np.empty(cap, np.int64); XP = np.empty(cap)
    OP = np.empty(cap, np.bool_); TG = np.empty(cap)
    m = 0
    for k in range(len(ev)):
        e = ev[k]
        B = c[e - 1]
        ent_t = np.full(19, -1, np.int64)
        ent_p = np.full(19, np.nan)
        ent_h = np.full(19, np.nan)
        ent_t[0], ent_p[0], ent_h[0] = e, c[e], h[e]                       # M0 급등 날 종가
        if e + 1 < n:
            ent_t[1], ent_p[1], ent_h[1] = e + 1, o[e + 1], h[e]           # M1 다음 날 시가
        for d in range(1, W + 1):                                           # D1~D10 아무 날 시가
            t = e + d
            if t < n:
                hh = h[e]
                for j in range(e, t):
                    hh = max(hh, h[j])
                ent_t[8 + d], ent_p[8 + d], ent_h[8 + d] = t, o[t], hh
        # 눌림목 규칙: 사건 뒤 1~W일, 실패 취소, 한 번만
        hh = h[e]
        failed = False
        rsi_hot = r14[e] >= 70
        for t in range(e + 1, min(e + W + 1, n)):
            if c[t - 1] < B and t - 1 > e:
                failed = True
            if failed:
                break
            # 지정가(전날까지 값)
            for q, f in ((2, 0.382), (3, 0.5), (4, 0.618)):
                if ent_t[q] < 0:
                    L = hh - f * (hh - B)
                    if l[t] <= L:
                        ent_t[q], ent_p[q], ent_h[q] = t, min(o[t], L), hh
            for q, em in ((5, e10), (6, e20)):
                if ent_t[q] < 0 and c[t - 1] > em[t - 1] and l[t] <= em[t - 1]:
                    ent_t[q], ent_p[q], ent_h[q] = t, min(o[t], em[t - 1]), hh
            # 마감 확정 신호 → 이 날(t) 시가
            if ent_t[7] < 0 and t - 4 >= e and c[t - 2] < c[t - 3] and c[t - 3] < c[t - 4] and c[t - 1] > h[t - 2]:
                ent_t[7], ent_p[7], ent_h[7] = t, o[t], hh
            if r14[t - 1] >= 70:
                rsi_hot = True
            if ent_t[8] < 0 and rsi_hot and t - 1 > e and r14[t - 1] <= 50:
                ent_t[8], ent_p[8], ent_h[8] = t, o[t], hh
            hh = max(hh, h[t])
        for q in range(19):
            if ent_t[q] < 0:
                continue
            for ex in range(4):
                xi, xp, op = _exit(o, h, c, ent_t[q], ent_p[q], B, ent_h[q], ex)
                EV[m], EN[m], EX[m], TI[m], PI[m], XI[m], XP[m], OP[m], TG[m] = k, q, ex, ent_t[q], ent_p[q], xi, xp, op, ent_h[q]
                m += 1
    return EV[:m], EN[:m], EX[:m], TI[:m], PI[:m], XI[:m], XP[:m], OP[:m], TG[:m]


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def events(c, h, v, dv20, nbar, pc, r):
    out, last = [], -10**9
    for e in np.flatnonzero((r >= X_MIN) & (r <= 1.0) & (pc >= 3) & (dv20 >= 2e6) & (nbar >= 4)):
        if e - last <= W:
            continue
        out.append(e)
        last = e
    return np.array(out, np.int64)


def main():
    D = pd.read_parquet(os.path.join(SP, "sp_daily.parquet"))
    D["sym"] = D.sym.astype(str)
    spy = D[D.sym == "SPY"].set_index("day").c.astype(float)
    spy50 = spy.rolling(50).mean()
    rows = []
    for s, g in D.groupby("sym", sort=True):
        if s in ("SPY", "QQQ") or len(g) < 60:
            continue
        g = g.sort_values("day")
        o, h, l, c = (g[k].values.astype(float) for k in ("o", "h", "l", "c"))
        v, nbar = g.v.values.astype(float), g.nbar.values
        pc = np.r_[np.nan, c[:-1]]
        r = c / pc - 1
        dv = pd.Series(c * v)
        dv20 = dv.shift(1).rolling(20, min_periods=10).mean().values
        v20 = pd.Series(v).shift(1).rolling(20, min_periods=10).mean().values
        ev = events(c, h, v, dv20, nbar, pc, r)
        if not len(ev):
            continue
        r14 = rsi_state(c, 14)[0]
        EV, EN, EX, TI, PI, XI, XP, OP, TG = sim(o, h, l, c, ema(c, 10), ema(c, 20), r14, ev, W)
        days = g.day.values
        e = ev[EV]
        with np.errstate(divide="ignore", invalid="ignore"):
            vr = np.where(v20[e] > 0, v[e] / v20[e], np.nan)
        rows.append(pd.DataFrame(dict(                       # 메모리: 종목마다 배열로 만들어 붙인다(파이썬 튜플 100만 개 피함)
            sym=s, eday=days[e], entry=np.array(ENTRY)[EN], exit=np.array(EXIT)[EX], tday=days[TI], ent=PI,
            xday=days[XI], px=XP, open=OP, lag=TI - e, hold=XI - TI, r0=r[e], vr=vr, dv20=dv20[e], pc=pc[e], tgt=TG)))
    t = pd.concat(rows, ignore_index=True)
    t["ret"] = t.px / t.ent - 1
    # 같은 보유 구간 SPY(매수 전날 종가 → 매도 날 종가; M0는 사건 날 종가부터)
    sd = spy.index.values
    pos = {d: k for k, d in enumerate(sd)}
    sc = spy.values
    k0 = np.array([pos.get(d, -1) for d in t.tday]) - np.where(t.entry == "M0", 0, 1)
    k1 = np.array([pos.get(d, -1) for d in t.xday])
    t["spy"] = np.where((k0 >= 0) & (k1 >= 0), sc[np.maximum(k1, 0)] / sc[np.maximum(k0, 0)] - 1, np.nan)
    t["ex_spy"] = t.ret - t.spy
    t["spy_up"] = [bool(spy.get(d, np.nan) > spy50.get(d, np.nan)) for d in t.eday]
    t["yr"] = [("2024" if d.year == 2024 else "2025" if d.year == 2025 else "2026 상" if d < date(2026, 7, 1) else "2026 하")
               if d.year >= 2024 else None for d in t.eday]
    t = t[t.yr.notna()]
    for col in ("sym", "entry", "exit", "yr"):
        t[col] = t[col].astype("category")
    t.to_parquet(os.path.join(SP, "sp_trades.parquet"))
    summarize(t)


def summarize(t):
    L = ["급등 후 눌림목 매수 (2026-10-10, study.py)",
         "칸: 평균·중앙·승률·하위10 = 거래 수익, 아무날대비 = 같은 사건의 D1~D10 시가 매수(같은 매도 규칙) 평균을 뺀 값,",
         "    t = 사건 날짜 묶음 t, SPY초과 = 같은 보유 구간 SPY를 뺀 값, 지연 = 사건 뒤 며칠째 샀나(중앙), 체결률 = 사건 중 산 비율"]
    nev = t[(t.entry == "M0") & (t.exit == "H5")].groupby("yr", observed=True).size()
    base = (t[t.entry.astype(str).str.startswith("D")].groupby(["sym", "eday", "exit"], observed=True).ret.mean()
            .rename("dmean"))
    t = t.join(base, on=["sym", "eday", "exit"])
    t["vs_day"] = t.ret - t.dmean
    L.append("사건 수(+10%): " + ", ".join(f"{k} {v}" for k, v in nev.items()))
    names = ["M0", "M1", "F382", "F50", "F618", "EMA10", "EMA20", "REV", "RSI50", "D1", "D3", "D5", "D10"]
    for sub, f in (("+10% 전체", lambda x: x), ("+20% 이상", lambda x: x[x.r0 >= 0.2]),
                   ("+10% · 거래량 2배 이상", lambda x: x[x.vr >= 2])):
        x0 = f(t)
        for ex in EXIT:
            rows = []
            for nm in names:
                for yr in ("2024", "2025", "2026 상", "2026 하"):
                    y = x0[(x0.entry == nm) & (x0.exit == ex) & (x0.yr == yr)]
                    if not len(y):
                        continue
                    ne = len(f(t[(t.entry == "M0") & (t.exit == ex) & (t.yr == yr)]))
                    dm = y.groupby("eday").vs_day.mean()
                    se = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan
                    rows.append(dict(매수=nm, 기간=yr, 건수=len(y), 체결률=len(y) / ne if ne else np.nan, 평균=y.ret.mean(),
                                     중앙=y.ret.median(), 승률=(y.ret > 0).mean(), 하위10=y.ret.quantile(.1),
                                     아무날대비=y.vs_day.mean(), t=dm.mean() / se if se and se > 0 else np.nan,
                                     SPY초과=y.ex_spy.mean(), 지연=y.lag.median(), 미청산=y.open.mean()))
            df = pd.DataFrame(rows)
            for k, fm in (("체결률", "{:.0%}"), ("평균", "{:+.2%}"), ("중앙", "{:+.2%}"), ("승률", "{:.0%}"), ("하위10", "{:+.1%}"),
                          ("아무날대비", "{:+.2%}"), ("t", "{:+.2f}"), ("SPY초과", "{:+.2%}"), ("지연", "{:.0f}"), ("미청산", "{:.0%}")):
                df[k] = df[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
            L += ["", f"=== [{sub}] 매도 {ex}", df.to_string(index=False)]
    with open(os.path.join(ROOT, "results", "study.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
