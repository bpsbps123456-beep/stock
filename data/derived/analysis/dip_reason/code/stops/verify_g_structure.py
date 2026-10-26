"""반박 검증 — 구조 붕괴 + 40분봉 RSI 조건 최선안 LCR_below|c|0.0%|all|conf>30 을 정의 문장만 보고 다시 구현(2026-10-08).
정의: 매수 봉 ei에서 바닥 L 고정. 매수 봉 mod(ET 0시부터 분) >= 960 이고 그 ET 날짜에 정규장이 있으면 그날 정규장 최저 저가,
  아니면 매수 날짜보다 앞선, 정규장이 있던 마지막 ET 날짜의 정규장 최저 저가. L < 매수가일 때만 켜짐(아니면 그 거래 손절 없음).
  보유 봉 i(i-1 >= ei, 모든 세션)에서 c[i-1] < L*(1-buf) 이고 conf40[i] > g 이면 봉 i 시가에 판다(매 봉 다시 판단, NaN이면 안 팜).
  conf40[i] = 봉 i 시작 시각까지 '끝난'(시작+40분 <= 봉 i 시작) 마지막 40분봉의 RSI14(와일더, 첫 14개 변화 단순평균 시작).
  40분봉은 RAW 5m_<티커>.parquet 종가를 ET 04:00 기준 40분 묶음으로 직접 만든다(stop_lab.htf_rsi를 쓰지 않고 따로 계산, 일치 여부 보고).
이웃: g 25/35, buf -0.5%/+0.5%, 확인 저가(l), dev>30, 큰 봉 30·60분, 범위 reg. 대조: 조건 없음, 반대 조건(conf<=30).
하루 의존: 사용자78 날짜별 합 차이에서 가장 이득 큰 날 제외.
사용: py stops/verify_g_structure.py  (code 폴더에서) → ../results/stops/verify_g_structure.txt
"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import stop_lab
from stop_lab import run, compare, stop_audit, base_trades, htf_rsi, UNIV, RAW

_S, _load0 = {}, stop_lab.load


def load(period):          # 같은 값, 한 번만 읽기(메모리 절약). stop_lab.py 파일은 그대로.
    if period not in _S:
        _S[period] = _load0(period)
    return _S[period]


stop_lab.load = load
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_g_structure.txt"))
FH = open(OUT, "w", encoding="utf-8")
U78 = UNIV["사용자78"]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n"); FH.flush()


# ---------------------------------------------------------------- 바닥 L(봉마다 '그 봉에서 샀다면')
def floor_lcr(A):
    reg, day, l, mod = A["reg"].astype(bool), A["day"], A["l"], A["mod"]
    od = np.array([d.toordinal() for d in day], dtype=np.int64)
    rl = pd.Series(l[reg]).groupby(od[reg]).min()
    ro, rv = rl.index.values.astype(np.int64), rl.values.astype(float)
    kp = np.searchsorted(ro, od, side="left") - 1               # 그 날짜보다 앞선 마지막 정규장 날짜
    prev = np.where(kp >= 0, rv[np.clip(kp, 0, None)], np.nan)
    ks = np.searchsorted(ro, od, side="left")
    has = (ks < len(ro)) & (ro[np.clip(ks, 0, len(ro) - 1)] == od)
    same = np.where(has, rv[np.clip(ks, 0, len(ro) - 1)], np.nan)
    return np.where((mod >= 960) & has, same, prev)


# ---------------------------------------------------------------- 큰 봉 RSI(독립 계산)
_H = {}


def wilder(c, n):
    m = len(c)
    g, lo = np.full(m, np.nan), np.full(m, np.nan)
    if m <= n:
        return g, lo, np.full(m, np.nan)
    ch = np.diff(c)
    up, dn = np.clip(ch, 0, None), np.clip(-ch, 0, None)
    ag, al = up[:n].mean(), dn[:n].mean()
    g[n], lo[n] = ag, al
    for k in range(n + 1, m):
        ag = (ag * (n - 1) + up[k - 1]) / n
        al = (al * (n - 1) + dn[k - 1]) / n
        g[k], lo[k] = ag, al
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(lo == 0, 100.0, 100 - 100 / (1 + g / lo))
    r[np.isnan(g)] = np.nan
    return g, lo, r


def big_rsi(A, tf=40, n=14):
    """(dev, conf) — 인덱스 i는 봉 i 시가에 판단할 때 쓸 수 있는 값."""
    key = (A["sym"], A["mins"], len(A["tk"]), tf, n)
    if key in _H:
        return _H[key]
    m5 = pd.read_parquet(os.path.join(RAW, f"5m_{A['sym']}.parquet"))
    ix = m5.index.tz_convert("America/New_York")
    mod5 = ix.hour * 60 + ix.minute
    st = ix.normalize() + pd.to_timedelta(((mod5 - 240) // tf) * tf + 240, unit="min")   # 04:00 기준 tf분 묶음 시작
    cl = pd.Series(m5.Close.values.astype(float), index=st.as_unit("ns").asi8).dropna()
    b = cl.groupby(level=0).last()
    bs = b.index.values.astype(np.int64)
    be = bs + tf * 60 * 10**9                                       # 묶음 끝
    bc = b.values
    g, lo, r = wilder(bc, n)
    tk = A["tk"]
    jc = np.searchsorted(be, tk, side="right") - 1                  # 끝 <= 봉 i 시작인 마지막 묶음
    conf = np.where(jc >= 0, r[np.clip(jc, 0, None)], np.nan)
    # dev: 봉 i-1이 속한 묶음 이전의 마지막 완성 묶음 상태 + c[i-1]
    tkp = np.r_[tk[0], tk[:-1]]
    idx = A["idx"]
    mod_p = np.r_[0, (idx.hour * 60 + idx.minute).values[:-1]]
    nrm = (idx.normalize()).as_unit("ns").asi8
    nrm_p = np.r_[nrm[0], nrm[:-1]]
    cur_p = nrm_p + (((mod_p - 240) // tf) * tf + 240).astype(np.int64) * 60 * 10**9
    jd = np.searchsorted(be, cur_p, side="right") - 1
    jj = np.clip(jd, 0, None)
    cp = np.r_[np.nan, A["c"][:-1]]
    ch = cp - bc[jj]
    gg = (g[jj] * (n - 1) + np.clip(ch, 0, None)) / n
    ll = (lo[jj] * (n - 1) + np.clip(-ch, 0, None)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(ll == 0, 100.0, 100 - 100 / (1 + gg / ll))
    dev = np.where((jd >= 0) & ~np.isnan(g[jj]), dev, np.nan)
    dev[0] = conf[0] = np.nan
    _H[key] = (dev, conf)
    return dev, conf


def make(src="c", buf=0.0, gate="conf", op=">", g=30.0, tf=40, scope="all"):
    def make_rule(A):
        Lb = floor_lcr(A)
        x = A["c"] if src == "c" else A["l"]
        reg = A["reg"].astype(bool)
        if gate is not None:
            dv, cf = big_rsi(A, tf)
            H = cf if gate == "conf" else dv

        def rule(i, ei, ent):
            L = Lb[ei]
            if not (L < ent):
                return False
            if scope == "reg" and not (reg[i - 1] and reg[i]):
                return False
            if not (x[i - 1] < L * (1 - buf)):
                return False
            if gate is None:
                return True
            h = H[i]
            return bool(h > g) if op == ">" else bool(h <= g)
        return rule
    return make_rule


def cellv(df, p, s, u="사용자78"):
    return df[(df.기간 == p) & (df.범위 == s) & (df.종목 == u)].iloc[0]


RES = {}


def test(label, **kw):
    t0 = time.time()
    df = compare(make(**kw), label, show=False)
    RES[label] = df
    v = [cellv(df, p, s) for p in ("early", "late") for s in ("24시간", "정규장만")]
    a = [cellv(df, p, "24시간", "전체137") for p in ("early", "late")]
    say(f"{label:<34} E24 {v[0].차이:+6.2f}(t{v[0].t:+.2f},{v[0].손절:>3}) ERg {v[1].차이:+6.2f}(t{v[1].t:+.2f},{v[1].손절:>3})"
        f" | L24 {v[2].차이:+6.2f}(t{v[2].t:+.2f},{v[2].손절:>3}) LRg {v[3].차이:+6.2f}(t{v[3].t:+.2f},{v[3].손절:>3})"
        f" | 137 E24 {a[0].차이:+6.2f} L24 {a[1].차이:+6.2f} [{time.time() - t0:.0f}s]")
    return df


if __name__ == "__main__":
    T0 = time.time()
    say("반박 검증: LCR_below|c|0.0%|all|conf>30 (정의 문장만 보고 독립 재구현, 40분봉 RSI도 직접 계산)")
    say("연구자 보고 사용자78: E24 +0.73 ERg +1.17 L24 -0.58 LRg -0.24 / 137 E24 +1.17 L24 -1.22\n")

    # 1) 독립 40분봉 RSI vs stop_lab.htf_rsi 일치
    say("== 1. 독립 40분봉 RSI vs stop_lab.htf_rsi (모든 종목·모든 봉)")
    for p in ("early", "late"):
        S = load(p)
        nb = nbd = nt = ntd = 0
        for s, A in S.items():
            d1, c1 = big_rsi(A)
            d0, c0 = htf_rsi(A)
            m = ~(np.isnan(c1) & np.isnan(c0))
            nt += int(m.sum()); nb += int((~np.isclose(c1[m], c0[m], atol=1e-9, equal_nan=True)).sum())
            md = ~(np.isnan(d1) & np.isnan(d0))
            ntd += int(md.sum()); nbd += int((~np.isclose(d1[md], d0[md], atol=1e-9, equal_nan=True)).sum())
        say(f"  {p}: conf 불일치 {nb}/{nt}, dev 불일치 {nbd}/{ntd}")

    # 2) 미래 참조 점검: 봉 i 시작 뒤 5분봉을 망가뜨려도 conf[i] 같아야 함(표본)
    say("\n== 2. 미래 참조 점검: 봉 i 시작 이후 5분봉 종가를 무작위로 바꿔도 conf[i]·dev[i]가 같은가")
    rng = np.random.default_rng(11)
    bad = tot = 0
    for p in ("early", "late"):
        S = load(p)
        syms = sorted(S)
        for _ in range(60):
            s = syms[rng.integers(len(syms))]
            A = S[s]
            i = int(rng.integers(300, len(A["tk"])))
            d0, c0 = big_rsi(A)
            m5 = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet"))
            ts = m5.index.tz_convert("UTC").as_unit("ns").asi8
            fut = ts + 5 * 60 * 10**9 > A["tk"][i]
            m5x = m5.copy()
            m5x.loc[fut, "Close"] = m5x.loc[fut, "Close"].values * rng.uniform(0.5, 1.5, fut.sum())
            Ax = dict(A); Ax["sym"] = s
            Ax["c"] = A["c"].copy(); Ax["c"][i:] = Ax["c"][i:] * 1.37      # 봉 i 이후 기준봉 종가도 바꿈
            orig = pd.read_parquet
            pd.read_parquet = lambda path, *a, **k: m5x if path.endswith(f"5m_{s}.parquet") else orig(path, *a, **k)
            key = (s, A["mins"], len(A["tk"]), 40, 14)
            saved = _H.pop(key)
            try:
                d1, c1 = big_rsi(Ax)
            finally:
                pd.read_parquet = orig
                _H[key] = saved
            tot += 1
            bad += 0 if (np.isclose(c1[i], c0[i], equal_nan=True) and np.isclose(d1[i], d0[i], equal_nan=True)) else 1
    say(f"  {tot}개 표본 중 불일치 {bad}개")

    # 3) 재현 + 대조
    say("\n== 3. 재현(사용자78 차이 E24/ERg/L24/LRg, t, 손절 수)")
    best = test("재현 conf>30", gate="conf", op=">", g=30)
    test("대조 조건 없음", gate=None)
    test("대조 반대 conf<=30", gate="conf", op="<=", g=30)
    f = best.copy()
    say(f[["기간", "범위", "종목", "건수", "합", "차이", "t", "나은날", "나쁜날", "손절", "최악", "차이_10월6일"]].round(3).to_string(index=False))

    # 4) stop_audit
    say("\n== 4. stop_audit (사용자78)")
    mk = make(gate="conf", op=">", g=30)
    for p in ("early", "late"):
        for ro in (False, True):
            a = stop_audit(mk, p, reg_only=ro)
            say(f"  {p} {'정규장만' if ro else '24시간'}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    # 5) 하루 의존
    say("\n== 5. 하루 의존(사용자78 날짜별 합 차이)")
    for p in ("early", "late"):
        for ro in (False, True):
            b = base_trades(p, ro)
            b = b[b.sym.isin(U78)]
            t = run(load(p), mk, p, ro, syms=U78)
            days = sorted(set(b.day) | set(t.day))
            d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
            top = d.sort_values(ascending=False)
            say(f"  {p} {'정규장만' if ro else '24시간'}: 합 {d.sum():+.3f}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()},"
                f" 최대이득일 {top.index[0]} {top.iloc[0]:+.3f} → 빼면 {d.sum() - top.iloc[0]:+.3f};"
                f" 상위2일 빼면 {d.sum() - top.iloc[:2].sum():+.3f}; 최대손해일 {d.idxmin()} {d.min():+.3f}")

    # 6) 이웃
    say("\n== 6. 이웃(한 단계 위·아래)")
    test("g25 conf>25", gate="conf", op=">", g=25)
    test("g35 conf>35", gate="conf", op=">", g=35)
    test("buf -0.5% (L*1.005)", buf=-0.005, gate="conf", op=">", g=30)
    test("buf +0.5% (L*0.995)", buf=0.005, gate="conf", op=">", g=30)
    test("확인 저가 l", src="l", gate="conf", op=">", g=30)
    test("dev>30", gate="dev", op=">", g=30)
    test("큰 봉 30분 conf>30", gate="conf", op=">", g=30, tf=30)
    test("큰 봉 60분 conf>30", gate="conf", op=">", g=30, tf=60)
    test("범위 reg conf>30", gate="conf", op=">", g=30, scope="reg")

    say("\n== 요약(사용자78 E24 / ERg / L24 / LRg)")
    for k, df in RES.items():
        v = [cellv(df, p, s).차이 for p in ("early", "late") for s in ("24시간", "정규장만")]
        say(f"  {k:<34} " + " / ".join(f"{x:+.2f}" for x in v) + ("  네 칸 플러스" if all(x > 0 for x in v) else ""))
    say(f"\n걸린 시간 {time.time() - T0:.0f}s")
    FH.close()
