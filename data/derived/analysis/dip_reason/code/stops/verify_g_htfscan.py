"""반박 검증 — g_htfscan 최선안 '본전+2%|깨어|80분RSI14dev>30' 독립 재구현(2026-10-08).

정의(연구자 보고 best_rule.definition만 보고 다시 씀, g_htfscan.py·fixedtrail.py 코드 복사·import 없음):
  들고 있는 봉 i(i >= ei+1) 시가에 판다, 네 조건 모두 참일 때:
   (1) 무장 max(c[ei..i-1]) >= ent*(1+arm)   (2) c[i-1] <= ent*lvl   (3) 봉 i 시작 시각 KST 08:00~23:59
   (4) dev[i] > g, dev[i] = tf분 큰 봉 RSI(n) 진행 중 값: 봉 i-1 시작 시각이 속한 큰 봉 B의 바로 앞 완성 큰 봉까지의
       Wilder RMA 상태에 c[i-1]을 넣은 값. 큰 봉 = 5분봉(프리·애프터 포함), ET 04:00 기준 tf분 구간의 마지막 종가.
       dev[i]가 nan(준비 전)이면 (1)~(3)만으로 판다.
  RSI는 htf.py·stop_lab.htf_rsi를 쓰지 않고 직접 만든다(첫 n개 변화 단순평균 시작). stop_lab.htf_rsi dev와 값이 같은지도 센다.
점검: 재현(compare), stop_audit(early·late), 조건 없음·반대 조건, 이웃(g 25·35, 무장 1.5·2.5%, 손절선 0.99·1.01, tf 60·120, n 7·21),
      하루 의존(최고 이득일·상위 2일 뺌), 부록: 연구자가 전방 확인 후보로 든 '종가-3%|40분RSI7dev>30'의 같은 점검.
사용: py stops/verify_g_htfscan.py   (결과: ../results/stops/verify_g_htfscan.txt)
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import stop_lab  # noqa: E402
from stop_lab import compare, stop_audit, load, base_trades, run, htf_rsi, UNIV, RAW  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_g_htfscan.txt"))
U78, A137 = UNIV["사용자78"], UNIV["전체137"]
_M5, _BIG, _DEV = {}, {}, {}


def big_state(sym, tf, n):
    """5분봉 → tf분 큰 봉(ET 04:00 기준) 시작 시각(ns), 종가, 그 봉까지의 RMA 평균 상승·하락."""
    k = (sym, tf, n)
    if k in _BIG:
        return _BIG[k]
    if sym not in _M5:
        _M5[sym] = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    m = _M5[sym]
    t = m.index.tz_convert("America/New_York")
    a4 = t.normalize() + pd.Timedelta(hours=4)
    st = a4 + ((t - a4) // pd.Timedelta(minutes=tf)) * pd.Timedelta(minutes=tf)
    s = pd.Series(m["Close"].values.astype(float), index=st).groupby(level=0).last().dropna()
    cl = s.values
    ag = np.full(len(cl), np.nan)
    al = np.full(len(cl), np.nan)
    if len(cl) > n:
        d = np.diff(cl)
        up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
        g_, l_ = up[:n].mean(), dn[:n].mean()          # 큰 봉 1..n의 변화 평균 = 큰 봉 n까지의 상태
        ag[n], al[n] = g_, l_
        for j in range(n + 1, len(cl)):
            g_ = (g_ * (n - 1) + up[j - 1]) / n
            l_ = (l_ * (n - 1) + dn[j - 1]) / n
            ag[j], al[j] = g_, l_
    _BIG[k] = (s.index.as_unit("ns").asi8, cl, ag, al)
    return _BIG[k]


def my_dev(A, tf, n):
    k = (A["sym"], A["mins"], len(A["tk"]), tf, n)
    if k in _DEV:
        return _DEV[k]
    starts, cl, ag, al = big_state(A["sym"], tf, n)
    idx = A["idx"]                                     # ET
    a4 = (idx.normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
    tk = A["tk"]
    step = tf * 60 * 10**9
    B = a4 + ((tk - a4) // step) * step                # 봉 k가 속한 큰 봉 시작
    out = np.full(len(tk), np.nan)
    # 봉 i 판단: 봉 i-1이 속한 큰 봉 B[i-1]의 바로 앞 완성 큰 봉 j (시작 < B[i-1]인 마지막 큰 봉)
    j = np.searchsorted(starts, B[:-1], side="left") - 1
    ok = j >= 0
    jj = np.maximum(j, 0)
    p = A["c"][:-1]
    ch = p - cl[jj]
    g2 = (ag[jj] * (n - 1) + np.maximum(ch, 0)) / n
    l2 = (al[jj] * (n - 1) + np.maximum(-ch, 0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(l2 == 0, 100.0, 100 - 100 / (1 + g2 / l2))
    r[~ok | np.isnan(ag[jj])] = np.nan
    out[1:] = r
    _DEV[k] = out
    return out


def mk_be(arm=0.02, lvl=1.0, tf=80, n=14, g=30.0, side="gt"):
    """본전 계열. side: 'gt' 조건 > g 일 때만, 'le' <= g 일 때만, None 조건 없음."""
    def make_rule(A):
        c = A["c"]
        awake = np.asarray(A["idx"].tz_convert("Asia/Seoul").hour >= 8)
        dev = my_dev(A, tf, n) if side else None
        st = {"ei": -1, "i": -1, "mx": -np.inf}

        def rule(i, ei, ent):
            if st["ei"] == ei and st["i"] == i - 1:
                mx = max(st["mx"], c[i - 1])
            else:
                mx = c[ei:i].max()
            st["ei"], st["i"], st["mx"] = ei, i, mx
            if not (mx >= ent * (1 + arm) and c[i - 1] <= ent * lvl and awake[i]):
                return False
            if side is None:
                return True
            x = dev[i]
            if np.isnan(x):
                return True
            return x > g if side == "gt" else x <= g
        return rule
    return make_rule


def mk_fx(p=0.03, tf=40, n=7, g=30.0, side="gt"):
    """고정 종가 손절 + 큰 봉 RSI dev 조건(부록)."""
    def make_rule(A):
        c = A["c"]
        dev = my_dev(A, tf, n) if side else None

        def rule(i, ei, ent):
            if c[i - 1] > ent * (1 - p):
                return False
            if side is None:
                return True
            x = dev[i]
            if np.isnan(x):
                return True
            return x > g if side == "gt" else x <= g
        return rule
    return make_rule


def daydiff(make_rule, period, reg, syms):
    t = run(load(period), make_rule, period, reg, syms=syms)
    b = base_trades(period, reg)
    b = b[b.sym.isin(syms)]
    days = sorted(set(b.day) | set(t.day))
    return (t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0))


def cells(df):
    out = {}
    for r in df.itertuples():
        out[(r.기간, r.범위, r.종목)] = r
    return out


def row4(df):
    c = cells(df)
    return {k: c[k] for k in [("early", "24시간", "사용자78"), ("early", "정규장만", "사용자78"),
                               ("late", "24시간", "사용자78"), ("late", "정규장만", "사용자78")]}


def fmt4(df):
    r = row4(df)
    return " | ".join(f"{k[0]} {k[1]} {v.차이:+.2f} (t {v.t:+.2f}, 손절 {v.손절})" for k, v in r.items())


def dev_match(period, tf, n, per=40, seed=3):
    """내 dev와 stop_lab.htf_rsi dev 비교(무작위 종목 per개, 기간 안 모든 봉)."""
    S = load(period)
    rng = np.random.default_rng(seed)
    syms = sorted(S)
    bad = tot = 0
    worst = 0.0
    for _ in range(per):
        A = S[syms[int(rng.integers(len(syms)))]]
        a = my_dev(A, tf, n)
        b, _c = htf_rsi(A, tf, n)
        both = ~(np.isnan(a) & np.isnan(b))
        diff = np.abs(np.nan_to_num(a, nan=-999) - np.nan_to_num(b, nan=-999))[both]
        tot += both.sum()
        bad += int((diff > 1e-6).sum())
        if len(diff):
            worst = max(worst, float(diff.max()))
    stop_lab._HTF.clear()
    return bad, tot, worst


if __name__ == "__main__":
    t0 = time.time()
    L = []
    w = L.append
    w("반박 검증 — g_htfscan 최선안 '본전+2%|깨어|80분RSI14dev>30' 독립 재구현 (2026-10-08)")
    w("기준(손절 없음, 사용자78): 24시간 early +27.12 · late +17.04, 정규장만 early +1.47 · late +2.28")
    w("연구자 보고: early24 +0.79(t 1.02) eReg +0.28 late24 -0.44(t -1.40) lReg -0.05, 손절 155/47/43/14")
    w("")
    for p in ("early", "late"):
        bad, tot, wv = dev_match(p, 80, 14)
        w(f"내 80분 RSI14 dev vs stop_lab.htf_rsi dev ({p}, 무작위 40종목 전체 봉): 불일치 {bad}/{tot}, 최대 차 {wv:.3g}")
    w("")
    rules = []
    best = ("최선안 본전+2%|깨어|80분RSI14dev>30", mk_be())
    rules.append(best)
    rules.append(("조건 없음 본전+2%|깨어", mk_be(side=None)))
    rules.append(("반대 조건 80분RSI14dev<=30", mk_be(side="le")))
    nb = [("이웃 g 25", mk_be(g=25)), ("이웃 g 35", mk_be(g=35)),
          ("이웃 무장 +1.5%", mk_be(arm=0.015)), ("이웃 무장 +2.5%", mk_be(arm=0.025)),
          ("이웃 손절선 매수가x0.99", mk_be(lvl=0.99)), ("이웃 손절선 매수가x1.01", mk_be(lvl=1.01)),
          ("이웃 큰 봉 60분", mk_be(tf=60)), ("이웃 큰 봉 120분", mk_be(tf=120)),
          ("이웃 RSI 7", mk_be(n=7)), ("이웃 RSI 21", mk_be(n=21))]
    rules += nb
    apx = [("부록 종가-3%|40분RSI7dev>30", mk_fx()), ("부록 조건 없음 종가-3%", mk_fx(side=None)),
           ("부록 반대 40분RSI7dev<=30", mk_fx(side="le")),
           ("부록 이웃 g 25", mk_fx(g=25)), ("부록 이웃 g 35", mk_fx(g=35)),
           ("부록 이웃 손절 -2.5%", mk_fx(p=0.025)), ("부록 이웃 손절 -3.5%", mk_fx(p=0.035)),
           ("부록 이웃 큰 봉 30분", mk_fx(tf=30)), ("부록 이웃 큰 봉 60분", mk_fx(tf=60)),
           ("부록 이웃 RSI 5", mk_fx(n=5)), ("부록 이웃 RSI 9", mk_fx(n=9))]
    rules += apx
    res = {}
    for nm, mr in rules:
        t1 = time.time()
        df = compare(mr, nm, show=False)
        res[nm] = df
        print(f"{nm}: {fmt4(df)}  [{time.time() - t1:.0f}s]", flush=True)
    w("== 재현: 최선안 전체 표 (compare)")
    bdf = res[best[0]]
    w(bdf[["기간", "범위", "종목", "건수", "합", "차이", "t", "나은날", "나쁜날", "손절", "손절평균", "최악", "차이_10월6일"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))
    w("")
    w("== stop_audit (사용자78)")
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(best[1], p, reg_only=reg)
            w(f"  {p} {'정규장만' if reg else '24시간'}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
    w("")
    w("== 대조·이웃 (사용자78 차이)")
    for nm, _ in rules[:3 + len(nb)]:
        w(f"  {nm}: {fmt4(res[nm])}")
    w("")
    w("== 하루 의존 (사용자78, 날짜별 합 차이)")
    for nm, mr in (best, apx[0]):
        for p in ("early", "late"):
            for reg in (False, True):
                d = daydiff(mr, p, reg, U78)
                top = d.sort_values(ascending=False)
                w(f"  {nm} {p} {'정규장만' if reg else '24시간'}: 합 {d.sum():+.3f}, 최고일 {top.index[0]} {top.iloc[0]:+.3f} -> 뺌 {d.sum() - top.iloc[0]:+.3f}, "
                  f"상위2일 뺌 {d.sum() - top.iloc[:2].sum():+.3f}, 최악일 {top.index[-1]} {top.iloc[-1]:+.3f}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}")
    w("")
    w("== 부록: 종가-3%|40분RSI7dev>30 (연구자가 전방 확인 후보로 든 규칙) 대조·이웃")
    for nm, _ in apx:
        w(f"  {nm}: {fmt4(res[nm])}")
    a_e = stop_audit(apx[0][1], "early")
    a_l = stop_audit(apx[0][1], "late")
    w("  stop_audit early 24시간: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a_e.items()))
    w("  stop_audit late 24시간: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a_l.items()))
    w("")
    w(f"시험한 규칙 수: {len(rules)} (최선안 1 + 대조 2 + 이웃 {len(nb)} + 부록 {len(apx)})")
    allpos = [nm for nm, _ in rules if all(v.차이 > 0 for v in row4(res[nm]).values())]
    w(f"사용자78 네 칸 모두 플러스: {len(allpos)}개 -> {', '.join(allpos) if allpos else '없음'}")
    w("")
    w("== 모든 규칙 전체 표")
    w(pd.concat(res.values(), ignore_index=True).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L).split("== 모든 규칙 전체 표")[0])
    print(f"{time.time() - t0:.0f}s")
