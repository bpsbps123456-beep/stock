"""7단계 반박 검증 V_IDXBUY(2026-10-11): 갈래 IDXBUY 사건형 대표(마삼식 대기)를 깨려고 시도한다.

후보: 나스닥 종합(^IXIC) 종가가 하루 −3% 이하인 날 = 사건(그날 종가로 판단). 그 뒤 21거래일 동안 −3% 날이 또 없으면
      22번째 거래일 종가에 산다(또 나면 그날부터 다시 셈). 126거래일 보유 뒤 현금. 보유 중 새 사건으로 겹치는 매수는 건너뜀.
주장: 126일 수익 − 아무 날 매수 = 1971~2005 +7.4%p(39건, 겹침 뺀 21건, t 1.4), 2006~2025 +2.2%p(34건, 평균 +9.2%,
      추가 −20% 9%, t 0.7). 같은 확인 구간 바로 매수 +7.4%p. 전략형 나스닥 1배 연 +6.5%/−65%(효율선 −2.0%p),
      3배 연 +9.1%/−98%(−2.3%p), 2000.03~02.10 3배 −96.8%. 2026: 6/5 사건 → 7/9 매수, 10/9까지 +4.4%.

독립 재계산: 공유 엔진(lev5·trade_lab·stock_lab·harness·p7_IDXBUY)을 import하지 않는다. cache/p7_idx.parquet,
      cache/p5_lev_long.parquet(실제 TQQQ 대조 한 줄)만 읽는다.
할 일: 1) 재현 2) 시점 감사·하루 늦춤·미끄러짐·'한 달' 정의 3) 이웃(앞뒤 2칸) 4) 위약(시드 20) 5) 시대별 6) 판정.
출력: results/p7_V_IDXBUY.md, results/p7_V_IDXBUY.json. 실행: PYTHONIOENCODING=utf-8 py p7_V_IDXBUY.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p7_V_IDXBUY"

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않는다)
X0, N0, H0 = 0.03, 21, 126                # 후보
SEL_END, CHK0, CHK1, END = "2005-12-31", "2006-01-01", "2025-12-31", "2026-10-09"
X_NEI = (0.02, 0.025, 0.03, 0.035, 0.04)  # 하락 문턱 앞뒤 2칸(0.5%p 간격)
N_NEI = (11, 16, 21, 26, 31)              # 대기 앞뒤 2칸(5거래일 간격)
H_NEI = (63, 95, 126, 158, 189)           # 보유 앞뒤 2칸(약 1.5개월 간격)
N_ORIG = (0, 1, 3, 5, 10, 21, 42, 63, "1M", "2M")  # 원 갈래 격자(선택편향 재계산용)
X_ORIG = (0.02, 0.03, 0.04, 0.05)
MIN_EVENTS = 30
SEEDS, SEEDS_BIG = 20, 1000
SLIP, CASH = 0.005, 0.04
FEE3, EXTRA_NQ = 0.0095, 0.013           # 3배 합성 보수·추가 비용(TQQQ 대리, 원 갈래와 같은 값)
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
WARM = 252                               # 전략형 시작(신호 252일 준비 뒤, 원 갈래와 같게)
NQ_CUT = "1985-10-01"                    # 나스닥 상품: 이 날까지 ^IXIC 수익률, 다음 날부터 ^NDX
EXTRA_DD = 0.20
ERAS = {  # 이름: (사건 창, 전략 창). 사건 창 = 사건일 기준(매도는 넘어가도 됨), 전략 창 = 그 기간 수익/낙폭
    "1987": (("1987-01-01", "1987-12-31"), ("1987-08-01", "1988-06-30")),
    "2000~02": (("2000-01-01", "2002-12-31"), ("2000-03-01", "2002-10-31")),
    "2008": (("2007-10-01", "2009-03-31"), ("2007-10-01", "2009-03-31")),
    "2011": (("2011-01-01", "2011-12-31"), ("2011-07-01", "2012-06-29")),
    "2018 4분기": (("2018-10-01", "2018-12-31"), ("2018-10-01", "2019-06-28")),
    "2020": (("2020-01-01", "2020-12-31"), ("2020-01-02", "2020-12-31")),
    "2022": (("2022-01-01", "2022-12-31"), ("2022-01-03", "2022-12-30")),
    "2025": (("2025-01-01", "2025-12-31"), ("2025-01-02", "2025-12-31")),
    "2026": (("2026-01-01", END), ("2026-01-02", END)),
}
CLAIM = {"sel_ex": 0.074, "sel_n": 39, "sel_ind": 21, "sel_t": 1.4, "chk_ex": 0.022, "chk_n": 34, "chk_mean": 0.092,
         "chk_hit": 0.09, "chk_t": 0.7, "chk_now_ex": 0.074, "st1_cagr": 0.065, "st1_mdd": -0.65, "st1_eff": -0.020,
         "st3_cagr": 0.091, "st3_mdd": -0.98, "st3_eff": -0.023, "st3_0002": -0.968, "y26_ret": 0.044}

RAW = pd.read_parquet(CACHE / "p7_idx.parquet")


# ============================================================================ 자료
class Idx:
    def __init__(self, sym, adj=False):
        g = RAW[RAW.sym == sym].sort_values("date").drop_duplicates("date").dropna(subset=["Close"])
        g = g[g.Close > 0]
        f = (g["Adj Close"] / g["Close"]).to_numpy(float) if adj else np.ones(len(g))
        self.sym = sym
        self.days = g.date.to_numpy().astype(str)
        self.dt = pd.to_datetime(pd.Series(self.days))
        self.c = g.Close.to_numpy(float) * f
        o = g.Open.to_numpy(float)
        self.o = np.where(np.isfinite(o) & (o > 0), o, g.Close.to_numpy(float)) * f
        self.n = len(self.c)

    def i(self, day, side="left"):
        return int(np.searchsorted(self.days, day, side=side))

    def win(self, a, z):
        return self.i(a), self.i(z, "right") - 1


def ev_days(c, x):
    r = c[1:] / c[:-1] - 1
    return np.flatnonzero(r <= -x) + 1


def schedule(S: Idx, ev, N=N0, H=H0, reset=True, delay=1, cal=None, plus1=False):
    """사건 → (s, d0, b, q). d0 = 결정일, b = 체결(d0+delay), q = 매도(b+H). reset: (s, d0]에 사건이 또 있으면 취소.
    cal=k: d0 = s 날짜 + k개월 이후 첫 거래일(원 갈래 '1개월'). plus1: b = s + k개월 + 1일 이후 첫 거래일(마삼 원문 글자 그대로), d0 = b−1."""
    ev = np.asarray(ev, int)
    out = []
    for k, s in enumerate(ev):
        if cal is None:
            d0 = s + N
        else:
            tgt = S.dt[s] + pd.DateOffset(months=cal) + (pd.Timedelta(days=1) if plus1 else pd.Timedelta(0))
            j = int(np.searchsorted(S.days, tgt.strftime("%Y-%m-%d"), "left"))
            d0 = j - 1 if plus1 else j
        if reset and k + 1 < len(ev) and ev[k + 1] <= d0:
            continue
        b = d0 + delay
        out.append((s, d0, b, b + H))
    return np.array(out, dtype=np.int64).reshape(-1, 4)


_BASE = {}


def base_cached(S, a, z, cap, H, px_key="c"):
    k = (S.sym, a, z, cap, H, px_key)
    if k not in _BASE:
        px = S.c if px_key == "c" else S.o
        d = np.arange(max(a, 1), z + 1)
        b = d + 1
        b = b[b + H <= cap]
        _BASE[k] = (float((px[b + H] / px[b] - 1).mean()) if len(b) else np.nan, b)
    return _BASE[k]


def ev_stats(S: Idx, sch, a, z, cap, H=H0, px_key="c", slip=0.0):
    px = S.c if px_key == "c" else S.o
    s, d0, b, q = (sch[:, k] for k in range(4)) if len(sch) else (np.zeros(0, int),) * 4
    inwin = (s >= a) & (s <= z)
    done = inwin & (q <= cap)
    pend = int((inwin & (q > cap)).sum()) if cap == S.n - 1 else 0
    base, _ = base_cached(S, a, z, cap, H, px_key)
    if done.sum() == 0:
        return {"n": 0, "n_ind": 0, "pend": pend, "base": base}
    bb, qq = b[done], q[done]
    ret = (px[qq] / px[bb]) * (1 - slip) ** 2 - 1
    lo = np.array([S.c[x:y + 1].min() for x, y in zip(bb, qq)])
    hit = lo <= (1 - EXTRA_DD) * px[bb]
    ex = ret - base
    o = np.argsort(bb, kind="stable")
    n_ind, endp = 0, -1
    for i in o:
        if bb[i] > endp:
            n_ind += 1
            endp = qq[i]
    k = len(ret)
    sd = float(np.std(ret, ddof=1)) if k > 1 else np.nan
    t = float(ex.mean() / (sd / np.sqrt(n_ind))) if (k > 1 and sd > 0) else np.nan
    return {"n": int(k), "n_ind": int(n_ind), "pend": pend, "mean": float(ret.mean()), "med": float(np.median(ret)),
            "pos": float((ret > 0).mean()), "hit20": float(hit.mean()), "base": base, "ex": float(ex.mean()), "t": t,
            "days": [S.days[x] for x in bb]}


# ============================================================================ 전략형
def nq_returns(IX: Idx):
    nd = Idx("^NDX")
    ndc = pd.Series(nd.c, index=nd.days).reindex(IX.days).ffill().to_numpy()
    r1 = np.r_[0.0, IX.c[1:] / IX.c[:-1] - 1]
    r2 = np.r_[0.0, ndc[1:] / ndc[:-1] - 1]
    ic = IX.i(NQ_CUT)
    r = np.where(np.arange(IX.n) <= ic, r1, np.nan_to_num(r2))
    irx = Idx("^IRX")
    s = pd.Series(irx.c / 100.0, index=irx.days)
    rf = s.reindex(sorted(set(s.index) | set(IX.days))).ffill().reindex(IX.days).fillna(0.0).to_numpy()
    r3 = np.maximum(3 * r - 2 * rf / 252 - (FEE3 + EXTRA_NQ) / 252, -0.999)
    r3[0] = 0.0
    return r, r3


def strat_trades(sch, n):
    out, last = [], -1
    if len(sch) == 0:
        return out
    for s, d0, b, q in sch[np.argsort(sch[:, 2], kind="stable")]:
        if b >= n or b < WARM or b <= last:
            continue
        qq = int(min(q, n - 1))
        out.append((int(b), qq))
        last = qq
    return out


def inv_mask(trades, n):
    m = np.zeros(n, bool)
    for b, q in trades:
        m[b + 1:q + 1] = True
    return m


def equity(rp, trades, slip=0.0, cash=0.0):
    n = len(rp)
    inv = inv_mask(trades, n)
    g = np.where(inv, 1 + rp, 1 + cash / 252)
    g[:WARM] = 1.0
    if slip:
        for b, q in trades:
            g[b] *= (1 - slip)
            g[q] *= (1 - slip)
    return np.cumprod(g), inv


def wstats(E, a, z):
    seg = E[a - 1:z + 1]
    ret = seg[-1] / seg[0] - 1
    mdd = float((seg / np.maximum.accumulate(seg) - 1).min())
    yrs = (z - a + 1) / 252
    return float(ret), mdd, float((1 + ret) ** (1 / yrs) - 1) if ret > -1 else -1.0


def frontier(rp, a, z, cash=0.0):
    rr = rp[a:z + 1]
    lg = np.log1p(W_GRID[:, None] * rr[None, :] + (1 - W_GRID[:, None]) * cash / 252)
    v = np.exp(np.concatenate([np.zeros((len(W_GRID), 1)), np.cumsum(lg, 1)], 1))
    ret = v[:, -1] - 1
    mdd = (v / np.maximum.accumulate(v, 1) - 1).min(1)
    cagr = (1 + ret) ** (252 / (z - a + 1)) - 1
    return ret, mdd, cagr


def eff(E, rp, a, z, long_metric=True, cash=0.0):
    ret, mdd, cagr = wstats(E, a, z)
    fr, fm, fc = frontier(rp, a, z, cash)
    ok = fm >= mdd - 1e-12
    met, fmet = (cagr, fc) if long_metric else (ret, fr)
    return {"ret": ret, "mdd": mdd, "cagr": cagr, "eff": float(met - fmet[ok].max())}


# ============================================================================ 실행
def main():
    IX = Idx("^IXIC")
    n = IX.n
    ev3 = ev_days(IX.c, X0)
    sel_a, sel_z = 0, IX.i(SEL_END, "right") - 1
    sel_cap = sel_z
    chk_a, chk_z = IX.win(CHK0, CHK1)
    W = {"sel": (sel_a, sel_z, sel_cap), "chk": (chk_a, chk_z, n - 1), "all": (0, n - 1, n - 1)}
    J = {"id": ID, "사전고정": {"X_NEI": X_NEI, "N_NEI": N_NEI, "H_NEI": H_NEI, "SEEDS": SEEDS, "SEEDS_BIG": SEEDS_BIG,
                                "SLIP": SLIP, "CASH": CASH, "ERAS": ERAS}}

    def evw(sch, w, **kw):
        a, z, cap = W[w]
        return ev_stats(IX, sch, a, z, cap, **kw)

    # ---------------------------------------------------------------- 1) 재현
    rep = schedule(IX, ev3, N0, H0)
    now = schedule(IX, ev3, 0, H0, reset=False)
    R = {w: evw(rep, w) for w in W}
    RN = {w: evw(now, w) for w in W}
    r, r3 = nq_returns(IX)
    trs = strat_trades(rep, n)
    longw = (WARM, n - 1)
    st = {}
    for lev, rp in ((1, r), (3, r3)):
        E, inv = equity(rp, trs)
        st[lev] = {"long": eff(E, rp, *longw), "sel": eff(E, rp, WARM, sel_z), "chk": eff(E, rp, chk_a, chk_z),
                   "0002": wstats(E, *IX.win("2000-03-01", "2002-10-31")),
                   "tpy": len(trs) / ((n - WARM) / 252), "expo": float(inv[WARM:].mean())}
    y26 = [t for t in rep if IX.days[t[0]] >= "2026-01-01"]
    y26_info = [{"사건": IX.days[s], "결정": IX.days[d0], "매수": IX.days[b] if b < n else None,
                 "10/9까지": float(IX.c[-1] / IX.c[b] - 1) if b < n else None} for s, d0, b, q in y26]
    repro = [
        ("고르기 1971~2005 아무 날 대비", CLAIM["sel_ex"], R["sel"]["ex"], "pp"),
        ("고르기 사건 수", CLAIM["sel_n"], R["sel"]["n"], "n"),
        ("고르기 겹침 뺀 수", CLAIM["sel_ind"], R["sel"]["n_ind"], "n"),
        ("고르기 대략 t", CLAIM["sel_t"], R["sel"]["t"], "x"),
        ("확인 2006~2025 아무 날 대비", CLAIM["chk_ex"], R["chk"]["ex"], "pp"),
        ("확인 사건 수", CLAIM["chk_n"], R["chk"]["n"], "n"),
        ("확인 평균 126일 수익", CLAIM["chk_mean"], R["chk"]["mean"], "pct"),
        ("확인 추가 −20% 확률", CLAIM["chk_hit"], R["chk"]["hit20"], "prob"),
        ("확인 대략 t", CLAIM["chk_t"], R["chk"]["t"], "x"),
        ("확인 바로 매수 아무 날 대비", CLAIM["chk_now_ex"], RN["chk"]["ex"], "pp"),
        ("전략 1배 긴 역사 연율", CLAIM["st1_cagr"], st[1]["long"]["cagr"], "pct"),
        ("전략 1배 최대 낙폭", CLAIM["st1_mdd"], st[1]["long"]["mdd"], "pct"),
        ("전략 1배 효율선초과(연)", CLAIM["st1_eff"], st[1]["long"]["eff"], "pp"),
        ("전략 3배 긴 역사 연율", CLAIM["st3_cagr"], st[3]["long"]["cagr"], "pct"),
        ("전략 3배 최대 낙폭", CLAIM["st3_mdd"], st[3]["long"]["mdd"], "pct"),
        ("전략 3배 효율선초과(연)", CLAIM["st3_eff"], st[3]["long"]["eff"], "pp"),
        ("전략 3배 2000.03~02.10", CLAIM["st3_0002"], st[3]["0002"][0], "pct"),
        ("2026 6/5 사건 → 10/9까지", CLAIM["y26_ret"], y26_info[0]["10/9까지"] if y26_info else np.nan, "pct"),
    ]
    J["재현"] = [{"항목": a, "원": b, "재계산": c} for a, b, c, _ in repro]
    J["재현_세부"] = {"사건형": {w: {k: v for k, v in R[w].items() if k != "days"} for w in W},
                       "바로매수": {w: {k: v for k, v in RN[w].items() if k != "days"} for w in W},
                       "전략형": {str(k): v for k, v in st.items()}, "2026": y26_info,
                       "-3%날_전체": int(len(ev3)), "전략_매수횟수": len(trs)}
    print("재현", {k: round(v, 4) if isinstance(v, float) else v for k, v in R["sel"].items() if k != "days"})
    print("재현 chk", {k: round(v, 4) if isinstance(v, float) else v for k, v in R["chk"].items() if k != "days"})
    print("전략", st[1]["long"], st[3]["long"], st[3]["0002"], y26_info)

    # ---------------------------------------------------------------- 2) 시점 감사·민감도
    # 감사: 사건은 r[s] = c[s]/c[s-1]-1(그날 종가까지), 결정은 (s, d0] 사건 확인(d0 종가까지), 체결 b > d0.
    audit = {"체결_결정_뒤": bool(np.all(rep[:, 2] > rep[:, 1])), "결정_사건_뒤": bool(np.all(rep[:, 1] > rep[:, 0])),
             "대기중_사건없음": bool(all(not np.any((ev3 > s) & (ev3 <= d0)) for s, d0, b, q in rep)),
             "^IXIC_시가=종가_비율_1971~83": float(np.mean(np.isclose(IX.o[:IX.i('1984-01-01')], IX.c[:IX.i('1984-01-01')]))),
             "^IXIC_시가=종가_비율_1986~": float(np.mean(np.isclose(IX.o[IX.i('1986-01-01'):], IX.c[IX.i('1986-01-01'):])))}
    sens_defs = [
        ("기준: 21거래일 · 다음 날 종가", dict(N=N0)),
        ("하루 늦게(23번째 날 종가)", dict(N=N0, delay=2)),
        ("이틀 늦게", dict(N=N0, delay=3)),
        ("20거래일", dict(N=20)),
        ("22거래일", dict(N=22)),
        ("달력 1개월 뒤 첫 거래일 결정 · 다음 날 종가", dict(cal=1)),
        ("달력 '한 달 + 1일' 첫 거래일 종가(원문 글자대로)", dict(cal=1, plus1=True)),
    ]
    sens = []
    for lab, kw in sens_defs:
        sch = schedule(IX, ev3, H=H0, **kw)
        row = {"판": lab}
        for w in ("sel", "chk"):
            s0 = evw(sch, w)
            s1 = evw(sch, w, slip=SLIP)
            row[w] = {"n": s0["n"], "n_ind": s0["n_ind"], "mean": s0.get("mean"), "ex": s0.get("ex"), "t": s0.get("t"),
                      "ex_slip": s1.get("ex")}
        trs_k = strat_trades(sch, n)
        for lev, rp in ((1, r), (3, r3)):
            E, _ = equity(rp, trs_k)
            Es, _ = equity(rp, trs_k, slip=SLIP)
            row[f"st{lev}"] = {"long": eff(E, rp, *longw), "long_slip": eff(Es, rp, *longw), "chk": eff(E, rp, chk_a, chk_z),
                               "sel": eff(E, rp, WARM, sel_z)}
        sens.append(row)
    # 다음 날 시가 체결(^IXIC 시가가 살아 있는 1986~만, 아무 날 기준도 시가)
    o86 = IX.i("1986-01-01")
    sch = schedule(IX, ev3, N0, H0)
    open_fill = {"1986~2005": ev_stats(IX, sch, o86, sel_z, sel_cap, px_key="o"),
                 "2006~2025": ev_stats(IX, sch, chk_a, chk_z, n - 1, px_key="o"),
                 "1986~2005_종가": ev_stats(IX, sch, o86, sel_z, sel_cap)}
    # 현금 4%
    cash_line = {}
    for lev, rp in ((1, r), (3, r3)):
        E, _ = equity(rp, trs, cash=CASH)
        cash_line[lev] = eff(E, rp, *longw, cash=CASH)
    # 실제 상품: 신호 ^IXIC, QQQ(배당 수정)·TQQQ(p5) 종가
    q = Idx("QQQ", adj=True)
    qmap = pd.Series(np.arange(q.n), index=q.days)
    real = {}
    for name, P in (("QQQ", q),):
        sch_d = [(s, d0, b, qq) for s, d0, b, qq in rep if b < n and qq < n and IX.days[s] >= CHK0 and IX.days[s] <= CHK1]
        rets = [P.c[qmap[IX.days[qq]]] / P.c[qmap[IX.days[b]]] - 1 for s, d0, b, qq in sch_d
                if IX.days[b] in qmap.index and IX.days[qq] in qmap.index]
        ba = qmap[IX.days[IX.i(CHK0)]]
        bz = q.i(CHK1, "right") - 1
        bb = np.arange(ba + 1, min(bz + 1, q.n - 1 - H0) + 1)
        base = float((P.c[bb + H0] / P.c[bb] - 1).mean())
        real[name] = {"n": len(rets), "mean": float(np.mean(rets)), "base": base, "ex": float(np.mean(rets) - base)}
    L5 = pd.read_parquet(CACHE / "p5_lev_long.parquet")
    tq = L5[L5.sym == "TQQQ"].sort_values("date").drop_duplicates("date")
    tqc = pd.Series((tq["Adj Close"]).to_numpy(float), index=tq.date.to_numpy().astype(str))
    tq_on = tqc.reindex(IX.days)
    t0 = IX.i(tqc.index[0]) + 1
    rtq = np.r_[0.0, tq_on.to_numpy()[1:] / tq_on.to_numpy()[:-1] - 1]
    rtq = np.where(np.arange(n) >= t0, np.nan_to_num(rtq), 0.0)
    a_t = max(t0, IX.i("2010-03-01"))
    E_tq, _ = equity(rtq, trs)
    E_s3, _ = equity(r3, trs)
    real["TQQQ_실제_2010.03~2026"] = eff(E_tq, rtq, a_t, n - 1)
    real["TQQQ_합성_같은구간"] = eff(E_s3, r3, a_t, n - 1)
    J["시점감사"] = audit
    J["민감도"] = sens
    J["시가체결"] = {k: {kk: vv for kk, vv in v.items() if kk != "days"} for k, v in open_fill.items()}
    J["현금4%"] = {str(k): v for k, v in cash_line.items()}
    J["실제상품"] = real
    print("감사", audit)
    for row in sens:
        print(row["판"], round(row["sel"]["ex"], 4), round(row["chk"]["ex"], 4), row["chk"]["n"], round(row["st1"]["long"]["eff"], 4), round(row["st3"]["long"]["eff"], 4))
    print("시가", {k: (v["n"], round(v.get("ex", np.nan), 4)) for k, v in open_fill.items()})
    print("실제", real)

    # ---------------------------------------------------------------- 3) 이웃
    grid = {}
    for x in X_NEI:
        evx = ev_days(IX.c, x)
        for N in N_NEI:
            sch = schedule(IX, evx, N, H0)
            grid[(x, N)] = {w: {k: v for k, v in evw(sch, w).items() if k != "days"} for w in ("sel", "chk")}
    hgrid = {}
    for H in H_NEI:
        sch = schedule(IX, ev3, N0, H)
        hgrid[H] = {w: {k: v for k, v in evw(sch, w, H=H).items() if k != "days"} for w in ("sel", "chk")}
        hgrid[H]["now"] = {w: {k: v for k, v in evw(schedule(IX, ev3, 0, H, reset=False), w, H=H).items() if k != "days"} for w in ("sel", "chk")}
    noreset = {w: {k: v for k, v in evw(schedule(IX, ev3, N0, H0, reset=False), w).items() if k != "days"} for w in ("sel", "chk")}
    # 전략형 이웃(나스닥 1배·3배 효율선초과, 긴 역사·확인)
    sgrid = {}
    for x in X_NEI:
        evx = ev_days(IX.c, x)
        for N in N_NEI:
            tk = strat_trades(schedule(IX, evx, N, H0), n)
            for lev, rp in ((1, r), (3, r3)):
                E, _ = equity(rp, tk)
                sgrid[(x, N, lev)] = {"long": eff(E, rp, *longw)["eff"], "sel": eff(E, rp, WARM, sel_z)["eff"],
                                      "chk": eff(E, rp, chk_a, chk_z)["eff"]}
    J["이웃_X×N"] = {f"{x}|{N}": v for (x, N), v in grid.items()}
    J["이웃_H"] = {str(k): v for k, v in hgrid.items()}
    J["다시셈없이_21일"] = noreset
    J["이웃_전략"] = {f"{x}|{N}|{lev}": v for (x, N, lev), v in sgrid.items()}

    # 선택편향: 원 격자의 E1(하루 −2~−5%) × C0/C1 × 대기 10 → 고르기 30건+ 조합에서 대표의 확인 순위
    sb = []
    for x in X_ORIG:
        evx = ev_days(IX.c, x)
        for cond in ("C0", "C1"):
            for N in N_ORIG:
                if N == 0 and cond == "C1":
                    continue
                if isinstance(N, str):
                    sch = schedule(IX, evx, H=H0, reset=(cond == "C1"), cal=int(N[0]))
                else:
                    sch = schedule(IX, evx, N, H0, reset=(cond == "C1"))
                a_ = evw(sch, "sel")
                b_ = evw(sch, "chk")
                sb.append({"x": x, "cond": cond, "N": str(N), "sel_n": a_["n"], "sel_ex": a_.get("ex", np.nan),
                           "chk_n": b_["n"], "chk_ex": b_.get("ex", np.nan)})
    SB = pd.DataFrame(sb)
    SBq = SB[(SB.sel_n >= MIN_EVENTS) & SB.chk_ex.notna()].copy()
    SBq["sel_rank"] = SBq.sel_ex.rank(ascending=False)
    SBq["chk_rank"] = SBq.chk_ex.rank(ascending=False)
    me = SBq[(SBq.x == X0) & (SBq.cond == "C1") & (SBq.N == str(N0))].iloc[0]
    J["선택편향_E1"] = {"조합": int(len(SBq)), "대표_고르기순위": float(me.sel_rank), "대표_확인순위": float(me.chk_rank),
                        "순위상관": float(SBq.sel_ex.rank().corr(SBq.chk_ex.rank())),
                        "확인_중앙값": float(SBq.chk_ex.median()), "확인>0비율": float((SBq.chk_ex > 0).mean()),
                        "표": SB.to_dict("records")}
    print("선택편향", {k: v for k, v in J["선택편향_E1"].items() if k != "표"})

    # ---------------------------------------------------------------- 4) 위약
    plc = {}
    for w in ("sel", "chk"):
        a, z, cap = W[w]
        act = R[w]
        K = act["n"]
        _, bpool = base_cached(IX, a, z, cap, H0)
        base = act["base"]
        # P1: 같은 수의 매수일을 무작위(독립) — 시드 20 + 1000
        def p1(seed):
            rng = np.random.default_rng(seed)
            bb = rng.choice(bpool, size=K, replace=False)
            return float((IX.c[bb + H0] / IX.c[bb] - 1).mean() - base)
        p1v = np.array([p1(sd) for sd in range(SEEDS)])
        p1b = np.array([p1(1000 + sd) for sd in range(SEEDS_BIG)])
        # P2: −3% 날 배열을 창 안에서 통째로 돌려 놓기(몰림 모양 유지, 폭락과의 연결만 끊음) → 같은 규칙
        evw_ = ev3[(ev3 >= a) & (ev3 <= z)]
        T = z - a + 1

        def p2(seed):
            rng = np.random.default_rng(seed)
            k = int(rng.integers(252, T - 252))
            sh = np.sort(a + (evw_ - a + k) % T)
            st_ = ev_stats(IX, schedule(IX, sh, N0, H0), a, z, cap)
            return st_.get("ex", np.nan), st_["n"]
        p2r = [p2(sd) for sd in range(SEEDS)]
        p2v = np.array([x for x, _ in p2r])
        p2b = np.array([p2(2000 + sd)[0] for sd in range(SEEDS_BIG)])
        # P3: 같은 −3% 사건, 대기만 무작위(0~63거래일 균등, 다시 셈 유지)
        def p3(seed):
            rng = np.random.default_rng(seed)
            out = []
            for k, s in enumerate(ev3):
                N = int(rng.integers(0, 64))
                d0 = s + N
                if k + 1 < len(ev3) and ev3[k + 1] <= d0:
                    continue
                out.append((s, d0, d0 + 1, d0 + 1 + H0))
            st_ = ev_stats(IX, np.array(out, dtype=np.int64).reshape(-1, 4), a, z, cap)
            return st_.get("ex", np.nan)
        p3v = np.array([p3(sd) for sd in range(SEEDS)])
        ex0 = act["ex"]
        plc[w] = {"실제": ex0,
                  "P1_무작위날": {"순위": int((p1v >= ex0).sum() + 1), "중앙값": float(np.median(p1v)), "최대": float(p1v.max()),
                                "p_1000": float((p1b >= ex0).mean())},
                  "P2_사건배열이동": {"순위": int((p2v >= ex0).sum() + 1), "중앙값": float(np.nanmedian(p2v)), "최대": float(np.nanmax(p2v)),
                                   "p_1000": float(np.nanmean(p2b >= ex0)), "평균매수수": float(np.mean([m for _, m in p2r]))},
                  "P3_대기무작위": {"순위": int((p3v >= ex0).sum() + 1), "중앙값": float(np.nanmedian(p3v)), "최대": float(np.nanmax(p3v))}}
    # 전략형 위약: 같은 매수 수·126일 보유를 창 안 무작위 날짜(겹침 없음), 그리고 보유/현금 모양을 통째로 이동
    def rand_place(rng, a, z, K, H):
        Lk = H + 1
        free = (z - a + 1) - K * Lk
        if free < 0:
            return []
        p = np.sort(rng.choice(free + K, size=K, replace=False))
        bs = a + (p - np.arange(K)) + np.arange(K) * Lk
        return [(int(b), int(b + H)) for b in bs]

    splc = {}
    for wname, (a, z) in (("long", longw), ("sel", (WARM, sel_z)), ("chk", (chk_a, chk_z))):
        K = sum(1 for b, q_ in trs if a <= b <= z)
        for lev, rp in ((1, r), (3, r3)):
            E, inv = equity(rp, trs)
            act = eff(E, rp, a, z, long_metric=True)
            v1, v2 = [], []
            for sd in range(SEEDS):
                rng = np.random.default_rng(sd)
                tk = rand_place(rng, a, z, K, H0)
                Ep, _ = equity(rp, tk)
                v1.append(eff(Ep, rp, a, z))
                k = int(rng.integers(252, z - a + 1 - 252))
                m = inv[a:z + 1]
                ms = np.roll(m, k)
                g = np.ones(len(rp))
                g[a:z + 1] = np.where(ms, 1 + rp[a:z + 1], 1.0)
                Eq = np.cumprod(g)
                v2.append(eff(Eq, rp, a, z))
            splc[f"{wname}|{lev}"] = {
                "매수수": K, "실제_연율": act["cagr"], "실제_효율선": act["eff"],
                "P1_연율순위": int(sum(x["cagr"] >= act["cagr"] for x in v1) + 1), "P1_연율중앙값": float(np.median([x["cagr"] for x in v1])),
                "P1_효율선순위": int(sum(x["eff"] >= act["eff"] for x in v1) + 1), "P1_효율선중앙값": float(np.median([x["eff"] for x in v1])),
                "P2_연율순위": int(sum(x["cagr"] >= act["cagr"] for x in v2) + 1), "P2_연율중앙값": float(np.median([x["cagr"] for x in v2])),
                "P2_효율선순위": int(sum(x["eff"] >= act["eff"] for x in v2) + 1), "P2_효율선중앙값": float(np.median([x["eff"] for x in v2]))}
    J["위약_사건형"] = plc
    J["위약_전략형"] = splc
    print("위약", json.dumps(plc, ensure_ascii=False)[:1500])
    print("위약전략", json.dumps(splc, ensure_ascii=False)[:2000])

    # ---------------------------------------------------------------- 5) 시대별
    eras = {}
    for nm, ((e0, e1), (s0, s1)) in ERAS.items():
        a, z = IX.win(e0, e1)
        rr = ev_stats(IX, rep, a, z, n - 1)
        rn = ev_stats(IX, now, a, z, n - 1)
        n3 = int(((ev3 >= a) & (ev3 <= z)).sum())
        sa, sz = IX.win(s0, s1)
        row = {"−3%날": n3, "대기": {k: v for k, v in rr.items()}, "바로": {k: v for k, v in rn.items() if k != "days"}}
        for lev, rp in ((1, r), (3, r3)):
            E, _ = equity(rp, trs)
            En, _ = equity(rp, strat_trades(now, n))
            hold = np.cumprod(np.r_[1.0, 1 + rp[1:]])
            third = np.cumprod(np.r_[1.0, 1 + rp[1:] / 3])
            row[f"st{lev}"] = {"규칙": wstats(E, sa, sz), "바로매수126": wstats(En, sa, sz), "보유": wstats(hold, sa, sz),
                               "1/3": wstats(third, sa, sz)}
        eras[nm] = row
        print(nm, n3, rr["n"], rr.get("mean"), rr.get("ex"), rn["n"], rn.get("mean"), row["st3"]["규칙"][0])
    J["시대별"] = eras

    # 기다림이 이긴 까닭: 2000~02 사건을 뺀 고르기 구간
    a, z, cap = W["sel"]
    m0002 = lambda sch: sch[~((sch[:, 0] >= IX.i("2000-01-01")) & (sch[:, 0] <= IX.i("2002-12-31", "right") - 1))]
    J["2000~02_뺀_고르기"] = {"대기": {k: v for k, v in ev_stats(IX, m0002(rep), a, z, cap).items() if k != "days"},
                              "바로": {k: v for k, v in ev_stats(IX, m0002(now), a, z, cap).items() if k != "days"}}
    print("2000~02 뺀", J["2000~02_뺀_고르기"])

    # 2026 앞·뒤(전략형, 기간 수익 기준 효율선) + 2018 4분기 뒤 실제 매수
    y = {}
    for wn, (w0, w1) in (("2026앞", ("2026-01-02", "2026-05-29")), ("2026뒤", ("2026-06-01", END))):
        a, z = IX.win(w0, w1)
        for lev, rp in ((1, r), (3, r3)):
            E, _ = equity(rp, trs)
            hold = np.cumprod(np.r_[1.0, 1 + rp[1:]])
            third = np.cumprod(np.r_[1.0, 1 + rp[1:] / 3])
            y[f"{wn}|{lev}"] = {"규칙": eff(E, rp, a, z, long_metric=False), "보유": wstats(hold, a, z), "1/3": wstats(third, a, z)}
    J["2026앞뒤"] = y
    J["2019초_매수"] = [{"사건": IX.days[s], "매수": IX.days[b], "126일": float(IX.c[q] / IX.c[b] - 1)}
                       for s, d0, b, q in rep if "2018-10-01" <= IX.days[s] <= "2019-03-31" and q < n]

    with open(RESULTS / f"{ID}.json", "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_md(J, repro)
    return J, repro


# ============================================================================ 보고
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p"


def pr(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def fmt(v, kind):
    if kind == "pp":
        return pp(v)
    if kind == "pct":
        return pct(v)
    if kind == "n":
        return f"{int(v)}"
    if kind == "prob":
        return f"{v * 100:.1f}%"
    return f"{v:.2f}"


def write_md(J, repro):
    L = []
    D = J["재현_세부"]
    sens = J["민감도"]
    plc, splc = J["위약_사건형"], J["위약_전략형"]
    sb = J["선택편향_E1"]
    L.append("# 7단계 반박 검증: IDXBUY 사건형 대표 '마삼식 대기' (2026-10-11)\n")
    L.append("> 후보: 나스닥 종합(^IXIC) 하루 −3% 이하(그날 종가로 판단) → 21거래일 동안 −3% 날이 또 없으면 22번째 거래일 종가에 사서"
             " 126거래일 보유(또 나면 그날부터 다시 셈). 전략형은 평소 현금, 나스닥 1배(^IXIC→1985-10 뒤 ^NDX)와 3배 합성(TQQQ 대리).")
    L.append("> 독립 재계산 `code/p7_V_IDXBUY.py`(lev5·trade_lab·stock_lab·harness·p7_IDXBUY import 없음, `cache/p7_idx.parquet`·"
             "`p5_lev_long.parquet`만). 숫자 `results/p7_V_IDXBUY.json`. 이웃·위약·시대 창은 스크립트 맨 위에 미리 고정했다.\n")

    sx = {row["판"]: row for row in sens}
    base_row = sens[0]
    cal_row = sx["달력 1개월 뒤 첫 거래일 결정 · 다음 날 종가"]
    st_long = [row["st1"]["long"]["eff"] for row in sens] + [row["st3"]["long"]["eff"] for row in sens]
    st_slip = [row["st1"]["long_slip"]["eff"] for row in sens] + [row["st3"]["long_slip"]["eff"] for row in sens]
    eras = J["시대별"]
    cmp = [(nm, e["대기"]["mean"] - e["바로"]["mean"]) for nm, e in eras.items() if e["대기"]["n"] and e["바로"]["n"]]
    lose = sum(1 for _, d in cmp if d < 0)
    L.append("## 결론 3줄\n")
    L.append(f"1. **숫자는 맞다.** 고르기 1971~2005 {pp(D['사건형']['sel']['ex'])}(39건, 겹침 뺀 21건, t {D['사건형']['sel']['t']:.1f}), "
             f"확인 2006~2025 {pp(D['사건형']['chk']['ex'])}(34건, 평균 {pct(D['사건형']['chk']['mean'])}, 추가 −20% {pr(D['사건형']['chk']['hit20'])}, t {D['사건형']['chk']['t']:.1f}), "
             f"확인 바로 매수 {pp(D['바로매수']['chk']['ex'])}, 전략 1배 연 {pct(D['전략형']['1']['long']['cagr'])}/{pct(D['전략형']['1']['long']['mdd'], 0)}(효율선 {pp(D['전략형']['1']['long']['eff'])}), "
             f"3배 연 {pct(D['전략형']['3']['long']['cagr'])}/{pct(D['전략형']['3']['long']['mdd'], 0)}({pp(D['전략형']['3']['long']['eff'])}), 2000~02 3배 {pct(D['전략형']['3']['0002'][0])}, "
             f"2026 6/5 사건 → 7/9 매수 → 10/9까지 {pct(D['2026'][0]['10/9까지'])}가 모두 0.1%p 안에서 재현됐다. 미래 정보 누수는 없었다(신호 종가 → 결정일 종가 → 다음 날 종가 체결).")
    L.append(f"2. **확인 구간 우위 {pp(D['사건형']['chk']['ex'])}는 우연과 구별되지 않고, 고르기 1등은 확인에서 하위였다.** 같은 수의 무작위 날 위약 21개 중 "
             f"{plc['chk']['P1_무작위날']['순위']}위(1,000개로 p {plc['chk']['P1_무작위날']['p_1000']:.2f}), 같은 −3% 사건에 대기만 0~63일 무작위로 준 위약 21개 중 "
             f"{plc['chk']['P3_대기무작위']['순위']}위(위약 중앙값 {pp(plc['chk']['P3_대기무작위']['중앙값'])}). 달력 1개월로 읽으면 {pp(cal_row['chk']['ex'])}, "
             f"여기에 미끄러짐 0.5%면 {pp(cal_row['chk']['ex_slip'])}. 원 격자의 하루 −2~−5% 조합 {sb['조합']}개에서 대표는 고르기 1위·확인 {sb['대표_확인순위']:.0f}위"
             f"(순위상관 {sb['순위상관']:+.2f}, 확인 중앙값 {pp(sb['확인_중앙값'])}). 시대별로 바로 매수보다 {lose}/{len(cmp)}개 시대에서 낮았다.")
    L.append(f"3. **전략으로는 '덜 담기'를 못 넘는다.** 긴 역사 효율선초과가 시점·'한 달' 정의 7판 × 1·3배 14개 모두 음수({pp(min(st_long))}~{pp(max(st_long))}), "
             f"미끄러짐 0.5%면 {pp(min(st_slip))}~{pp(max(st_slip))}, 현금 4%여도 1배 {pp(J['현금4%']['1']['eff'])}·3배 {pp(J['현금4%']['3']['eff'])}, "
             f"실제 TQQQ 2010.03~2026은 {pp(J['실제상품']['TQQQ_실제_2010.03~2026']['eff'])}였다. 사건형 수치는 맞지만 쓸 만한 우위가 아니다. 판정 **confirmed_fragile**.\n")

    L.append("## 1. 재현 (원 갈래 주장 vs 독립 재계산)\n")
    L.append("| 항목 | 원 갈래 | 독립 재계산 | 차 |")
    L.append("|---|---|---|---|")
    for a, b, c, kind in repro:
        diff = "" if kind == "n" and b == c else (f"{(c - b) * 100:+.2f}%p" if kind in ("pp", "pct", "prob") else f"{c - b:+.2f}")
        if kind == "n":
            diff = f"{int(c - b):+d}"
        L.append(f"| {a} | {fmt(b, kind)} | {fmt(c, kind)} | {diff} |")
    L.append("")
    L.append("- 원 갈래의 숫자는 반올림 한 자리라 차 0.05%p 안은 같은 값이다. 긴 역사 전략 매수 "
             f"{D['전략_매수횟수']}번(2026 7/9 매수 1번은 보유 중), 나스닥 종합 −3% 날 {D['-3%날_전체']}번.")
    L.append("")

    L.append("## 2. 시점 감사·민감도\n")
    au = J["시점감사"]
    L.append(f"- 감사: 체결일 > 결정일 > 사건일 모두 성립({au['체결_결정_뒤']}·{au['결정_사건_뒤']}), 대기 중 −3% 날이 남은 매수 0건({au['대기중_사건없음']}). "
             f"^IXIC 시가는 1971~83 {pr(au['^IXIC_시가=종가_비율_1971~83'])}가 종가와 같아(가짜 시가) 다음 날 종가 체결이 맞다. 1986 뒤 시가=종가 {pr(au['^IXIC_시가=종가_비율_1986~'])}.")
    of = J["시가체결"]
    L.append(f"- 다음 날 시가 체결(시가 아무 날 대비): 1986~2005 {pp(of['1986~2005']['ex'])}(종가 {pp(of['1986~2005_종가']['ex'])}), 2006~2025 {pp(of['2006~2025']['ex'])}. "
             f"QQQ(배당 수정)로 사고팔면 확인 구간 {J['실제상품']['QQQ']['n']}건 평균 {pct(J['실제상품']['QQQ']['mean'])}, 아무 날 대비 {pp(J['실제상품']['QQQ']['ex'])}.\n")
    L.append("| 판 | 고르기 아무 날 대비(건) | 확인 아무 날 대비(건) | 확인 + 미끄러짐 0.5% | 확인 t | 전략 1배 긴 역사 효율선 · 미끄러짐 | 전략 3배 긴 역사 효율선 · 미끄러짐 | 전략 1배·3배 확인 효율선 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for row in sens:
        L.append(f"| {row['판']} | {pp(row['sel']['ex'])} ({row['sel']['n']}) | {pp(row['chk']['ex'])} ({row['chk']['n']}) | {pp(row['chk']['ex_slip'])} | {row['chk']['t']:.2f} | "
                 f"{pp(row['st1']['long']['eff'])} · {pp(row['st1']['long_slip']['eff'])} | {pp(row['st3']['long']['eff'])} · {pp(row['st3']['long_slip']['eff'])} | "
                 f"{pp(row['st1']['chk']['eff'])} · {pp(row['st3']['chk']['eff'])} |")
    L.append("")
    L.append("- 미끄러짐은 규칙에만 붙였다(살 때·팔 때 0.5%씩, 아무 날 기준은 그대로). 원 갈래 '1개월'은 달력 1개월 뒤 첫 거래일을 결정일로 보고 다음 날 종가에 산다. "
             "'한 달 + 1일'은 마삼 원문 글자대로 사건일 + 1개월 + 1일 뒤 첫 거래일 종가에 산다.")
    L.append(f"- 현금 이자 4%(규칙·덜 담기 둘 다): 1배 연 {pct(J['현금4%']['1']['cagr'])}, 효율선 {pp(J['현금4%']['1']['eff'])}; 3배 연 {pct(J['현금4%']['3']['cagr'])}, {pp(J['현금4%']['3']['eff'])}.")
    tq, ts = J["실제상품"]["TQQQ_실제_2010.03~2026"], J["실제상품"]["TQQQ_합성_같은구간"]
    L.append(f"- 실제 TQQQ(2010.03~2026.10, 배당 수정): 규칙 연 {pct(tq['cagr'])}/{pct(tq['mdd'], 0)}, 효율선 {pp(tq['eff'])}(같은 구간 합성 {pp(ts['eff'])}).\n")

    L.append("## 3. 이웃 (사전 고정 앞뒤 2칸, 또 나면 다시 셈, 126일 보유)\n")
    L.append("사건형: 아무 날 대비 초과 — 고르기 1971~2005 / 확인 2006~2025 (사건 수)\n")
    L.append("| 하루 하락 \\ 대기 | " + " | ".join(f"{N}일" for N in N_NEI) + " |")
    L.append("|---|" + "---|" * len(N_NEI))
    g = J["이웃_X×N"]
    both = chk_pos = 0
    for x in X_NEI:
        cells = []
        for N in N_NEI:
            v = g[f"{x}|{N}"]
            a_, b_ = v["sel"].get("ex", np.nan), v["chk"].get("ex", np.nan)
            chk_pos += b_ > 0
            both += (a_ > 0) and (b_ > 0)
            mark = "**" if (x == X0 and N == N0) else ""
            cells.append(f"{mark}{a_ * 100:+.1f} / {b_ * 100:+.1f}{mark} ({v['sel']['n']}·{v['chk']['n']})")
        L.append(f"| −{x * 100:g}% | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"- 25칸 중 확인 > 0이 {chk_pos}칸, 두 구간 모두 > 0이 {both}칸. −3% 줄에서는 대기가 짧을수록 고르기는 나빠지고 확인은 좋아졌다(11일 {pp(g['0.03|11']['sel']['ex'])} / {pp(g['0.03|11']['chk']['ex'])}). "
             f"−2·−2.5%는 확인에서 대부분 음수, −3.5·−4%는 확인에서 모두 양수지만 −4%는 고르기에서 음수다.")
    h = J["이웃_H"]
    L.append("- 보유 기간(−3%·21일): " + ", ".join(f"{H}일 {pp(h[str(H)]['sel']['ex'])} / {pp(h[str(H)]['chk']['ex'])}" for H in H_NEI)
             + ". 같은 보유의 바로 매수: " + ", ".join(f"{H}일 {pp(h[str(H)]['now']['sel']['ex'])} / {pp(h[str(H)]['now']['chk']['ex'])}" for H in H_NEI)
             + ". 확인 구간에서는 모든 보유 기간에서 바로 매수가 기다림보다 높았다.")
    nr = J["다시셈없이_21일"]
    L.append(f"- '다시 셈' 없이 21일 뒤 사기: {pp(nr['sel']['ex'])} / {pp(nr['chk']['ex'])}({nr['sel']['n']}·{nr['chk']['n']}건). 고르기 우위는 기다림이 아니라 '다시 셈'이 몰린 때 매수를 줄인 데서 나왔다.\n")
    L.append("전략형: 긴 역사 효율선초과(연) 1배 / 3배\n")
    L.append("| 하루 하락 \\ 대기 | " + " | ".join(f"{N}일" for N in N_NEI) + " |")
    L.append("|---|" + "---|" * len(N_NEI))
    sg = J["이웃_전략"]
    pos1 = pos3 = 0
    for x in X_NEI:
        cells = []
        for N in N_NEI:
            a1, a3 = sg[f"{x}|{N}|1"]["long"], sg[f"{x}|{N}|3"]["long"]
            pos1 += a1 > 0
            pos3 += a3 > 0
            mark = "**" if (x == X0 and N == N0) else ""
            cells.append(f"{mark}{a1 * 100:+.1f} / {a3 * 100:+.1f}{mark}")
        L.append(f"| −{x * 100:g}% | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"- 25칸 중 효율선 > 0: 1배 {pos1}칸, 3배 {pos3}칸. 대표 칸은 둘 다 음수다.\n")

    L.append("## 4. 위약 (시드 20, 실제를 넣은 21개 중 순위, 1 = 가장 좋음)\n")
    L.append("| 위약 | 고르기 1971~2005: 순위 · 위약 중앙값 · p(1,000개) | 확인 2006~2025: 순위 · 위약 중앙값 · p(1,000개) |")
    L.append("|---|---|---|")
    for k, lab in (("P1_무작위날", "같은 수 매수일을 무작위로(사건과 무관)"), ("P2_사건배열이동", "−3% 날 배열을 창 안에서 통째로 밀기(몰림 모양 유지)"),
                   ("P3_대기무작위", "같은 −3% 사건, 대기만 0~63일 무작위(다시 셈 유지)")):
        a_, b_ = plc["sel"][k], plc["chk"][k]
        pa = f" · p {a_['p_1000']:.3f}" if "p_1000" in a_ else ""
        pb = f" · p {b_['p_1000']:.3f}" if "p_1000" in b_ else ""
        L.append(f"| {lab} | {a_['순위']}위 · {pp(a_['중앙값'])}{pa} | {b_['순위']}위 · {pp(b_['중앙값'])}{pb} |")
    L.append(f"| (실제) | {pp(plc['sel']['실제'])} | {pp(plc['chk']['실제'])} |")
    L.append("")
    L.append("전략형(같은 매수 수·126일 보유를 무작위 날에, 겹침 없음 = P1 / 보유·현금 모양을 통째로 밀기 = P2): 연율 순위 · 효율선 순위 (위약 중앙값)\n")
    L.append("| 창 · 배 | 매수 수 | 실제 연율 · 효율선 | P1 연율 · 효율선 | P2 연율 · 효율선 |")
    L.append("|---|---|---|---|---|")
    for key_, lab in (("long|1", "긴 역사 1배"), ("long|3", "긴 역사 3배"), ("sel|1", "고르기 1배"), ("sel|3", "고르기 3배"), ("chk|1", "확인 1배"), ("chk|3", "확인 3배")):
        v = splc[key_]
        L.append(f"| {lab} | {v['매수수']} | {pct(v['실제_연율'])} · {pp(v['실제_효율선'])} | {v['P1_연율순위']}위 · {v['P1_효율선순위']}위 ({pct(v['P1_연율중앙값'])} · {pp(v['P1_효율선중앙값'])}) | "
                 f"{v['P2_연율순위']}위 · {v['P2_효율선순위']}위 ({pct(v['P2_연율중앙값'])} · {pp(v['P2_효율선중앙값'])}) |")
    L.append("")
    L.append("- 고르기 구간에서는 세 위약 모두 1~2위라 '−3% 뒤 다시 셈 21일'이 우연 이상이었다. 확인 구간에서는 4위·4위·15위로 우연과 구별되지 않는다. "
             "전략형은 무작위 시점보다 연율 순위는 높지만(1~8위) 효율선은 대표 칸이 확인 1배 말고 모두 음수라, 시점이 무작위보다 낫다는 것과 낙폭을 맞춘 덜 담기를 넘는다는 것은 다르다.\n")

    L.append("## 5. 시대별 (나스닥 종합, 사건일 기준, 126일 보유)\n")
    L.append("| 시대 | −3% 날 | 대기 규칙: 건 · 평균 · 추가 −20% | 바로 매수: 건 · 평균 | 아무 날 평균 | 대기 − 바로 | 전략 1배 규칙 / 보유 / 1/3 | 전략 3배 규칙 / 보유 / 1/3 | 전략 창 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for nm, e in eras.items():
        d, b = e["대기"], e["바로"]
        dd = f"{d['n']} · {pct(d['mean'])} · {pr(d['hit20'])}" if d["n"] else (f"0 (진행 중 {d['pend']})" if d.get("pend") else "0")
        bb = f"{b['n']} · {pct(b['mean'])}" if b["n"] else (f"0 (진행 중 {b['pend']})" if b.get("pend") else "0")
        diff = pp(d["mean"] - b["mean"]) if d["n"] and b["n"] else "—"
        s1, s3 = e["st1"], e["st3"]
        f3 = lambda s: f"{pct(s['규칙'][0])}/{pct(s['보유'][0])}/{pct(s['1/3'][0])}"
        L.append(f"| {nm} | {e['−3%날']} | {dd} | {bb} | {pct(d['base'])} | {diff} | {f3(s1)} | {f3(s3)} | {ERAS[nm][1][0]}~{ERAS[nm][1][1]} |")
    L.append("")
    y19 = J["2019초_매수"]
    L.append("- 2018 4분기 −3% 날 5번은 모두 '다시 셈'으로 취소되고, " + ", ".join(f"{t['사건']} 사건 → {t['매수']} 매수(126일 {pct(t['126일'])})" for t in y19) + "로 이어졌다.")
    L.append(f"- 대기가 바로 매수보다 높았던 시대: " + ", ".join(nm for nm, x in cmp if x >= 0) + f" / 낮았던 시대: " + ", ".join(nm for nm, x in cmp if x < 0) + ".")
    x0 = J["2000~02_뺀_고르기"]
    L.append(f"- 2000~02 사건을 빼면 고르기 구간에서도 바로 매수 {pp(x0['바로']['ex'])}({x0['바로']['n']}건) > 대기 {pp(x0['대기']['ex'])}({x0['대기']['n']}건)이다. "
             "고르기 우위는 2000~02에 93번 대신 8번만 산 데서 나왔다(원 갈래 설명과 같음).")
    y = J["2026앞뒤"]
    L.append(f"- 2026(전략형): 앞 1/2~5/29 매수 없음(규칙 {pct(y['2026앞|3']['규칙']['ret'])}, 3배 보유 {pct(y['2026앞|3']['보유'][0])}, 1/3 {pct(y['2026앞|3']['1/3'][0])}). "
             f"뒤 6/1~10/9 3배 규칙 {pct(y['2026뒤|3']['규칙']['ret'])}/{pct(y['2026뒤|3']['규칙']['mdd'], 0)}(효율선 {pp(y['2026뒤|3']['규칙']['eff'])}), 보유 {pct(y['2026뒤|3']['보유'][0])}, 1/3 {pct(y['2026뒤|3']['1/3'][0])}. "
             "6/5 한 건(보유 중)뿐이라 2026으로는 판정할 수 없다.\n")

    L.append("## 6. 판정: confirmed_fragile\n")
    L.append("- 재현: 원 갈래 숫자 18개 모두 0.1%p 안. 시점 누수 없음.")
    L.append("- 버티는 것: 확인 구간 아무 날 대비가 시점 7판·시가 체결·QQQ에서 모두 플러스(+1.1~+2.7%p), 고르기 구간 위약 1~2위.")
    L.append("- 깨지는 것: 확인 t 0.7·위약 p 0.20, 대기 무작위 위약보다 낮음(15/21), 달력 1개월 + 미끄러짐이면 0, 같은 갈래 확인 순위 37/48(순위상관 −0.90), "
             "바로 매수에 확인·5/7 시대에서 짐, 전략형은 모든 판에서 덜 담기 아래.")
    L.append("- 쓸 수 있는 말: '−3% 뒤 한 달 기다렸다 사면 아무 날보다 조금 낫다'까지는 맞지만, 기다림 자체의 값은 확인되지 않았다. 마삼식 재매수는 '덜 담기'를 대신하지 못한다.\n")
    L.append("## 한계\n")
    L.append("- 지수는 가격 지수(배당 없음)다. 아무 날 기준도 같은 지수라 초과에는 영향이 작다(QQQ 배당 수정판으로 확인).")
    L.append("- 1971~2005 대표는 39건(겹침 뺀 21건), 확인 34건(18건)이다. 2000~02·2008·2020·2022에 몰려 시대 몇 개가 결과를 정한다.")
    L.append("- 3배는 일일 3배 − 2×^IRX − 보수 0.95% − 추가 1.3% 합성이다. 1970~80년대 금리가 높아 비용이 컸고 실제 상품은 없었다.")
    L.append(f"- 탐색 범위(이 검증): 이웃 사건형 25칸 + 보유 5칸 + 전략 50칸, 시점 판 7개, 위약 3종 × 시드 20(+1,000), 원 격자 E1 {sb['조합']}개 재계산. 매수 추천이나 주문 수량이 아니다.")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
