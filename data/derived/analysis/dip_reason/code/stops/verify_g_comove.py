"""반박 검증 — g_comove 최선안 '묶음중앙30분<=-2% [24시간] | dev>30' 독립 재구현(2026-10-08).
연구자 코드(comove.py, g_comove.py)는 불러오지 않는다. 보고서의 best_rule.definition 글만 보고 다시 만들었다.

규칙(봉 i 시가에 판다, i-1 봉까지 값만):
  (1) 묶음 30분: 같은 대분류 그룹(fetch.universe) 다른 종목(stop_lab load(period)에 자료 있는 것)의 30분 수익률 중앙값(NaN 제외) <= -X.
      종목별 30분 수익률 = [시작 <= tk[i-1]인 마지막 봉 종가] / [시작 <= tk[i-1]-W분인 마지막 봉 종가] - 1.
      그룹이 사전에 없거나 다른 종목 3개 미만이면 QQQ 하나(events.bars, early 5분봉·late 1분봉→3분봉).
  (2) 40분봉 RSI 조건: dev40[i] <= g이면 팔지 않음(NaN이면 판다). dev40은 여기서 따로 계산한다:
      5분봉(프리·애프터 포함)을 04:00 ET 기준 tf분 칸으로 묶어 칸 종가 = 마지막 5분 종가, Wilder RSI(첫 n개 변화 단순평균으로 시작).
      봉 i-1이 속한 칸보다 앞서 끝난 마지막 칸의 (평균 상승, 평균 하락, 종가) 상태에 c[i-1]을 넣은 진행 중 RSI.
      stop_lab.htf_rsi와 값이 같은지도 확인한다.
  세션 제한 없음(24시간 적용), 자기 하락 조건 없음, 종가 기준. 매도 순서·재매수 금지는 stop_lab 시험대 그대로.
점검: 재현(compare, stop_audit), 이웃(g 25·35, X 1.5·2.5%, 창 20·40분, 큰 봉 30·60분, i-1 정규장만), 하루 의존, 미래 참조(잘라내기).
사용: py stops/verify_g_comove.py (code 폴더에서) → ../results/stops/verify_g_comove.txt
"""
import gc, io, os, sys, time, warnings
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
import stop_lab as SL
from stop_lab import load, compare, stop_audit, run, base_trades, htf_rsi, UNIV, RAW
from events import bars
from fetch import universe

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "verify_g_comove.txt"))
NSM = 60 * 10**9
GRP = universe()
U78 = UNIV["사용자78"]
_GW = {}     # (period, win) -> {sym: 배열, 원소 j = i-1 = j 시점 묶음 중앙 W분 수익률}
_DEV = {}    # (sym, mins, ntk, tf, n) -> dev 배열(인덱스 i = 봉 i 시가 판단용)


def close_asof(ptk, pc, t):
    k = np.searchsorted(ptk, t, side="right") - 1
    out = np.full(len(t), np.nan)
    m = k >= 0
    out[m] = pc[k[m]]
    return out


def gw_for(s, S, qqq, win):
    g = GRP.get(s)
    peers = [p for p in S if g is not None and p != s and GRP.get(p) == g]
    srcs = [(S[p]["tk"], S[p]["c"]) for p in peers] if len(peers) >= 3 else [qqq]
    tk = S[s]["tk"]
    R = np.empty((len(tk), len(srcs)))
    for k, (ptk, pc) in enumerate(srcs):
        R[:, k] = close_asof(ptk, pc, tk) / close_asof(ptk, pc, tk - win * NSM) - 1
    return np.nanmedian(R, axis=1), len(peers) >= 3


def qqq_bars(period):
    q = bars(RAW, "QQQ", 5 if period == "early" else 3)
    return q.index.as_unit("ns").asi8, q.Close.values.astype(float)


NPEER = {}


def build_gw(period, win):
    key = (period, win)
    if key in _GW:
        return _GW[key]
    S = load(period)
    qqq = qqq_bars(period)
    res = {}
    for s in S:
        res[s], hasp = gw_for(s, S, qqq, win)
        NPEER[period, s] = hasp
    _GW[key] = res
    del S
    gc.collect()
    return res


def wilder(hc, n):
    m = len(hc)
    ag, al = np.full(m, np.nan), np.full(m, np.nan)
    if m <= n:
        return ag, al
    ch = np.diff(hc)                      # ch[k-1] = hc[k] - hc[k-1]
    a, b = np.maximum(ch[:n], 0).mean(), np.maximum(-ch[:n], 0).mean()
    ag[n], al[n] = a, b
    for k in range(n + 1, m):
        x = ch[k - 1]
        a = (a * (n - 1) + max(x, 0.0)) / n
        b = (b * (n - 1) + max(-x, 0.0)) / n
        ag[k], al[k] = a, b
    return ag, al


def bins_of(idx_et, tf):
    """04:00 ET 기준 tf분 칸 시작(ns)."""
    t = idx_et.as_unit("ns").asi8
    anc = (idx_et.normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
    st = tf * NSM
    return anc + ((t - anc) // st) * st


def my_dev(A, tf=40, n=14, m5=None, c=None):
    key = (A["sym"], A["mins"], len(A["tk"]), tf, n)
    use_cache = m5 is None and c is None
    if use_cache and key in _DEV:
        return _DEV[key]
    if m5 is None:
        m5 = pd.read_parquet(os.path.join(RAW, f"5m_{A['sym']}.parquet")).sort_index()
    if c is None:
        c = A["c"]
    b = bins_of(m5.index, tf)
    ser = pd.Series(m5.Close.values.astype(float), index=b).dropna()
    hc_s = ser.groupby(level=0).last()
    hs, hc = hc_s.index.values.astype(np.int64), hc_s.values
    ag, al = wilder(hc, n)
    cur = bins_of(A["idx"], tf)                       # 봉 k가 속한 칸
    dev = np.full(len(c), np.nan)
    # 봉 i 판단: 봉 i-1이 속한 칸 cur[i-1]보다 먼저 시작한(=이미 끝난) 마지막 칸
    j = np.searchsorted(hs, cur[:-1], side="left") - 1
    ok = j >= 0
    jj = np.maximum(j, 0)
    x = c[:-1] - hc[jj]
    g = (ag[jj] * (n - 1) + np.maximum(x, 0)) / n
    l = (al[jj] * (n - 1) + np.maximum(-x, 0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(l == 0, 100.0, 100 - 100 / (1 + g / l))
    r[~ok | np.isnan(ag[jj])] = np.nan
    dev[1:] = r
    if use_cache:
        _DEV[key] = dev
    return dev


def rule_maker(X=0.02, g=30, win=30, tf=40, n=14, mode="gate", regonly=False):
    """mode: gate(dev>g 또는 NaN일 때만 자름) none(조건 없음) rev(dev<=g일 때만 자름)."""
    def make_rule(A):
        period = "early" if A["mins"] == 5 else "late"
        gw = _GW[(period, win)][A["sym"]]
        sig = np.zeros(len(A["c"]), bool)
        sig[1:] = gw[:-1] <= -X                           # sig[i]는 gw[i-1] 사용
        if regonly:
            sig[1:] &= A["reg"][:-1]
        if mode != "none":
            d = my_dev(A, tf, n)
            low = d <= g                                   # NaN -> False
            sig &= (low if mode == "rev" else ~low)

        def rule(i, ei, ent):
            return bool(sig[i])
        return rule
    return make_rule


def u78(df, period, scope):
    return df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == "사용자78")].iloc[0]


def daydep(make_rule, period, reg_only=False):
    S = load(period)
    t = run(S, make_rule, period, reg_only, True, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    del S
    gc.collect()
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d, t


def la_check(P):
    """미래 참조 잘라내기 시험: 컷오프 뒤 정보를 지우거나 무작위로 바꿔도 컷오프 전 결정 신호가 같은지."""
    rng = np.random.default_rng(7)
    for period in ("early", "late"):
        S = load(period)
        mins = 5 if period == "early" else 3
        qqq = qqq_bars(period)
        gw0 = _GW[(period, 30)]
        tot = bad = 0
        totd = badd = 0
        for s in ("NVDA", "MU", "PLTR", "TEM", "COIN", "IONQ", "SNDK"):
            if s not in S:
                continue
            A = S[s]
            tk = A["tk"]
            for cut in rng.choice(tk[len(tk) // 5: -len(tk) // 5], 2, replace=False):
                # 묶음: 모든 종목·QQQ의 '종가 확정 시각(시작+봉길이) > cut' 봉 종가를 무작위로 바꿈
                S2 = {}
                for p, B in S.items():
                    c2 = B["c"].copy()
                    m = B["tk"] + mins * NSM > cut
                    c2[m] *= rng.uniform(0.5, 1.5, m.sum())
                    S2[p] = dict(tk=B["tk"], c=c2)
                qc2 = qqq[1].copy()
                m = qqq[0] + mins * NSM > cut
                qc2[m] *= rng.uniform(0.5, 1.5, m.sum())
                gw1 = gw_for(s, S2, (qqq[0], qc2), 30)[0]
                k = np.nonzero(tk[1:] <= cut)[0]          # 결정 봉 i = k+1 의 시가 <= cut → gw[i-1]=gw[k]
                tot += len(k)
                bad += int((~np.isclose(gw0[s][k], gw1[k], equal_nan=True)).sum())
                # 40분 RSI: 5분봉 중 끝나는 시각 > cut 지움, 자기 종가도 확정 > cut이면 무작위
                m5 = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet")).sort_index()
                m5 = m5[m5.index.as_unit("ns").asi8 + 5 * NSM <= cut]
                c2 = A["c"].copy()
                mm = tk + mins * NSM > cut
                c2[mm] *= rng.uniform(0.5, 1.5, mm.sum())
                d0 = my_dev(A)
                d1 = my_dev(A, m5=m5, c=c2)
                ki = np.nonzero(tk <= cut)[0]
                ki = ki[ki >= 1]
                totd += len(ki)
                badd += int((~np.isclose(d0[ki], d1[ki], equal_nan=True)).sum())
        P(f"  {period}: 묶음 30분 값 {tot:,}개 중 바뀐 것 {bad}, 40분 dev {totd:,}개 중 바뀐 것 {badd}")
        del S
        gc.collect()


def main():
    t0 = time.time()
    log = io.StringIO()

    def P(*a):
        print(*a, flush=True)
        print(*a, file=log)

    P("반박 검증 — g_comove 최선안 독립 재구현 (2026-10-08)")
    P("대상: 묶음중앙30분<=-2% [24시간] | dev>30 (40분봉 RSI14 진행 중 값이 30 이하면 손절 보류)")
    for period in ("early", "late"):
        for w in (30, 20, 40):
            build_gw(period, w)
    P(f"묶음 30분 신호 준비 끝 ({time.time() - t0:.0f}s)")

    # 40분 RSI 자체 구현 vs stop_lab.htf_rsi
    P("\n[1] 40분 RSI 자체 구현 vs stop_lab.htf_rsi(dev)")
    for period in ("early", "late"):
        S = load(period)
        tot = bad = nn = 0
        for s, A in S.items():
            d0 = htf_rsi(A, 40, 14)[0]
            d1 = my_dev(A, 40, 14)
            tot += len(d0)
            bad += int((~np.isclose(d0, d1, equal_nan=True, atol=1e-6)).sum())
            nn += int(np.isnan(d1).sum())
        P(f"  {period}: 값 {tot:,}개 중 다른 값 {bad}, NaN {nn:,}")
        SL._HTF.clear()
        del S
        gc.collect()

    P("\n[2] 미래 참조 잘라내기 시험(7종목 × 컷오프 2개)")
    la_check(P)

    specs = [
        ("최선안 X2% g30 W30 40분", dict()),
        ("조건 없음", dict(mode="none")),
        ("반대 dev<=30", dict(mode="rev")),
        ("이웃 g25", dict(g=25)),
        ("이웃 g35", dict(g=35)),
        ("이웃 X1.5%", dict(X=0.015)),
        ("이웃 X2.5%", dict(X=0.025)),
        ("이웃 창20분", dict(win=20)),
        ("이웃 창40분", dict(win=40)),
        ("이웃 30분봉 RSI", dict(tf=30)),
        ("이웃 60분봉 RSI", dict(tf=60)),
        ("이웃 i-1 정규장만", dict(regonly=True)),
    ]
    tabs, rows = {}, []
    for nm, kw in specs:
        df = compare(rule_maker(**kw), nm, show=False)
        tabs[nm] = df
        r = {k: u78(df, p, sc) for k, (p, sc) in
             dict(e24=("early", "24시간"), eR=("early", "정규장만"), l24=("late", "24시간"), lR=("late", "정규장만")).items()}
        rows.append(dict(규칙=nm, e24=r["e24"].차이, e24t=r["e24"].t, eR=r["eR"].차이, l24=r["l24"].차이, l24t=r["l24"].t,
                         lR=r["lR"].차이, e손절=r["e24"].손절, l손절=r["l24"].손절))
        print(f"  {nm}: e24 {r['e24'].차이:+.3f} eR {r['eR'].차이:+.3f} l24 {r['l24'].차이:+.3f} lR {r['lR'].차이:+.3f} ({time.time() - t0:.0f}s)", flush=True)
        SL._HTF.clear()
        gc.collect()
    R = pd.DataFrame(rows)

    P("\n[3] 재현: 최선안 compare 전체 표")
    best = tabs[specs[0][0]]
    f = best[["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일"]].copy()
    P(f.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    rep = {("early", "24시간", "사용자78"): 1.615, ("early", "24시간", "전체137"): 2.091, ("early", "정규장만", "사용자78"): 1.756,
           ("early", "정규장만", "전체137"): 1.653, ("late", "24시간", "사용자78"): -0.93, ("late", "24시간", "전체137"): -1.036,
           ("late", "정규장만", "사용자78"): -0.072, ("late", "정규장만", "전체137"): 0.029}
    P("  연구자 보고와 차이:")
    for (p, sc, un), v in rep.items():
        x = best[(best.기간 == p) & (best.범위 == sc) & (best.종목 == un)].iloc[0]
        P(f"    {p} {sc} {un}: 재현 {x.차이:+.3f} / 보고 {v:+.3f} / 차 {x.차이 - v:+.3f} (손절 {x.손절})")

    P("\n[4] stop_audit (사용자78)")
    mk = rule_maker()
    for p in ("early", "late"):
        for ro in (False, True):
            a = stop_audit(mk, p, reg_only=ro)
            P(f"  {p} {'정규장만' if ro else '24시간'}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
            gc.collect()

    P("\n[5] 규칙별 사용자78 차이(조건 효과 분리 + 이웃)")
    P(R.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    P("\n[6] 하루 의존(최선안, 사용자78 날짜별 합 차이)")
    for p in ("early", "late"):
        for ro in (False, True):
            d, _ = daydep(mk, p, ro)
            nz = d[d.abs() > 1e-12].sort_values(ascending=False)
            tot = nz.sum()
            P(f"  {p} {'정규장만' if ro else '24시간'}: 합 {tot:+.3f}, 다른 날 {len(nz)} (+{int((nz > 0).sum())}/-{int((nz < 0).sum())}), "
              f"최대 이득일 {nz.index[0]} {nz.iloc[0]:+.3f} → 빼면 {tot - nz.iloc[0]:+.3f}, 상위 2일 빼면 {tot - nz.head(2).sum():+.3f}, "
              f"최대 손해일 {nz.index[-1]} {nz.iloc[-1]:+.3f}")
    P(f"\n시험한 규칙 {len(specs)}개, 걸린 시간 {time.time() - t0:.0f}초")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(log.getvalue())
    print("저장:", OUT)


if __name__ == "__main__":
    main()
