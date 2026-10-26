"""반박 검증 - g_waterfall 최선안 '#68 30분 -3% & 매수가 -0% 아래 & dev40 > 35'를 정의 문장만 보고 다시 만든다(2026-10-08).

정의(연구자 보고): 보유 내내, 모든 세션. 봉 i 시가에 판다 if
  (1) c[i-1] <= ref*(1-0.03), ref = c[i-1-m], m = 30//봉분(late 10봉, early 6봉), i-1-m < ei 이면 ref = 매수가
  (2) c[i-1] <= 매수가
  (3) 진행 중 40분봉 RSI14 > 35 (04:00 ET 기준 40분봉, 5분봉 parquet, 직전 완성 40분봉 RMA 상태에 c[i-1]을 넣은 값). NaN이면 손절 안 함.
40분봉 RSI는 stop_lab.htf_rsi를 쓰지 않고 여기서 따로 만든다(와일더 RMA, 첫 n개 변화 단순평균 시작).
  mode "A": 봉 i-1의 시작 시각이 속한 40분봉을 진행 중으로 보고, 그 앞 마지막 40분봉 상태 + c[i-1]  (정의 문장 그대로)
  mode "B": 봉 i 시작 시각(= 봉 i-1 종가 시각)까지 끝난 마지막 40분봉 상태 + c[i-1]  (3분봉이 40분 경계를 걸칠 때 다름)
  conf    : 봉 i 시작 시각까지 끝난 마지막 40분봉의 RSI
이웃: g 30·33·37·40, 하락폭 y 2·2.5·3.5·4%, 창 M 15·20·45·60분, 매수가 아래 z 1·2%, conf, mode B, 큰 봉 30·60분.
사용: py stops/verify_g_waterfall.py   (code 폴더에서)   출력: ../results/stops/verify_g_waterfall.txt"""
import os, sys, time
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, compare, stop_audit, run, base_trades, htf_rsi, UNIV, RAW

OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "verify_g_waterfall.txt"))
U78 = UNIV["사용자78"]
NS_MIN = 60 * 10**9
_M5, _BIG = {}, {}


def big_bars(sym, tf):
    """5분봉 -> tf분 큰 봉(04:00 ET 기준) 시작 시각(ns)·종가."""
    k = (sym, tf)
    if k in _BIG:
        return _BIG[k]
    if sym not in _M5:
        _M5[sym] = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    m5 = _M5[sym]
    t = m5.index
    t_ns = t.as_unit("ns").asi8
    a_ns = (t.normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
    st = a_ns + np.floor_divide(t_ns - a_ns, tf * NS_MIN) * tf * NS_MIN
    s = pd.Series(m5.Close.values.astype(float), index=st)
    s = s[~np.isnan(s.values)]
    last = s.groupby(level=0, sort=True).last()
    _BIG[k] = (last.index.values.astype(np.int64), last.values.astype(float))
    return _BIG[k]


def wilder(c, n):
    """평균 상승·하락 상태(봉 k 종가까지 반영). k < n 이면 NaN."""
    m = len(c)
    ag, al = np.full(m, np.nan), np.full(m, np.nan)
    if m <= n:
        return ag, al
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    a, b = up[:n].mean(), dn[:n].mean()
    ag[n], al[n] = a, b
    for k in range(n + 1, m):
        a = (a * (n - 1) + up[k - 1]) / n
        b = (b * (n - 1) + dn[k - 1]) / n
        ag[k], al[k] = a, b
    return ag, al


def rsi_of(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b == 0, 100.0, 100.0 - 100.0 / (1.0 + a / b))
    return np.where(np.isnan(a) | np.isnan(b), np.nan, r)


def my_htf(A, tf=40, n=14, kind="dev", mode="A"):
    hs, hc = big_bars(A["sym"], tf)
    ag, al = wilder(hc, n)
    tk = A["tk"].astype(np.int64)
    N = len(tk)
    out = np.full(N, np.nan)
    if N < 2:
        return out
    idx = A["idx"]
    anc = (idx.normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
    step = tf * NS_MIN
    tprev = tk[:-1]                                          # 봉 i-1 시작 (i = 1..N-1)
    tnow = tk[1:]                                            # 봉 i 시작
    if kind == "conf":
        j = np.searchsorted(hs + step, tnow, side="right") - 1      # 끝난 시각 <= 봉 i 시작
        ok = j >= 0
        jj = np.maximum(j, 0)
        out[1:] = np.where(ok, rsi_of(ag[jj], al[jj]), np.nan)
        return out
    if mode == "A":
        cur = anc[:-1] + np.floor_divide(tprev - anc[:-1], step) * step  # 봉 i-1이 속한 큰 봉 시작
        j = np.searchsorted(hs, cur, side="left") - 1                     # 그 앞 마지막 큰 봉
    else:
        j = np.searchsorted(hs + step, tnow, side="right") - 1
    ok = j >= 0
    jj = np.maximum(j, 0)
    p = A["c"][:-1]
    ch = p - hc[jj]
    a = (ag[jj] * (n - 1) + np.clip(ch, 0, None)) / n
    b = (al[jj] * (n - 1) + np.clip(-ch, 0, None)) / n
    out[1:] = np.where(ok, rsi_of(a, b), np.nan)
    return out


def rule_of(M=30, y=0.03, z=0.0, g=35, kind="dev", mode="A", tf=40, n=14, gate=">"):
    """gate: '>' 정방향, '<=' 반대, None 조건 없음."""
    def make_rule(A):
        c = A["c"]
        m = max(1, M // int(A["mins"]))
        h = my_htf(A, tf, n, kind, mode) if gate else None

        def rule(i, ei, ent):
            p = c[i - 1]
            if p > ent * (1.0 - z):
                return False
            b = i - 1 - m
            ref = c[b] if b >= ei else ent
            if p > ref * (1.0 - y):
                return False
            if gate is None:
                return True
            v = h[i]
            if v != v:
                return False
            return v > g if gate == ">" else v <= g
        return rule
    return make_rule


def say(f, s=""):
    print(s, flush=True)
    f.write(s + "\n")


def cells(df):
    q = lambda p, sc, un="사용자78": df[(df.기간 == p) & (df.범위 == sc) & (df.종목 == un)].iloc[0]
    return {f"{p}_{tag}{'' if un == '사용자78' else '_137'}": q(p, sc, un)
            for p in ("early", "late") for sc, tag in (("24시간", "24"), ("정규장만", "정규")) for un in ("사용자78", "전체137")}


def day_series(mk, period, reg_only=False):
    t = run(load(period), mk, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d[d.abs() > 1e-12].sort_values(ascending=False)


def main():
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as f:
        say(f, "반박 검증 g_waterfall #68 (30분 -3% & 매수가 -0% 아래 & dev40 > 35) - verify_g_waterfall.py (2026-10-08)")
        say(f, __doc__.split("\n", 1)[1].rsplit("사용:", 1)[0].rstrip())

        # 0) 내 40분봉 RSI vs stop_lab.htf_rsi
        say(f, "\n=== 0) 독립 40분봉 RSI와 stop_lab.htf_rsi 비교(모든 봉) ===")
        for p in ("early", "late"):
            S = load(p)
            dA = dB = dC = 0.0
            nA = nB = nC = tot = 0
            for s, A in S.items():
                dev, conf = htf_rsi(A, 40, 14)
                mA, mB, mC = my_htf(A, 40, 14, "dev", "A"), my_htf(A, 40, 14, "dev", "B"), my_htf(A, 40, 14, "conf")
                ok = ~np.isnan(dev)
                tot += int(ok.sum())
                for mine, ref, key in ((mA, dev, "A"), (mB, dev, "B"), (mC, conf, "C")):
                    both = ~np.isnan(mine) & ~np.isnan(ref)
                    nan_mis = int((np.isnan(mine) != np.isnan(ref)).sum())
                    dd = np.abs(mine[both] - ref[both])
                    mx = float(dd.max()) if len(dd) else 0.0
                    nbig = int((dd > 1e-6).sum()) + nan_mis
                    if key == "A":
                        dA, nA = max(dA, mx), nA + nbig
                    elif key == "B":
                        dB, nB = max(dB, mx), nB + nbig
                    else:
                        dC, nC = max(dC, mx), nC + nbig
            say(f, f"  {p}: 봉 {tot}개 | mode A vs dev 다른 봉 {nA}, 최대 차 {dA:.2e} | mode B vs dev 다른 봉 {nB}, 최대 차 {dB:.2f} "
                   f"| conf 다른 봉 {nC}, 최대 차 {dC:.2e}")

        # 1) 재현
        say(f, "\n=== 1) 재현: 정의대로 다시 만든 최선안 ===")
        best = rule_of()
        df = compare(best, "재현 #68", show=False)
        C = cells(df)
        for k, r in C.items():
            say(f, f"  {k}: 차이 {r.차이:+.3f} t {r.t:+.2f} 손절 {r.손절} 최악 {r.최악:+.3f} 10/6 {r.차이_10월6일:+.3f} 나은날 {r.나은날} 나쁜날 {r.나쁜날}")
        say(f, "  연구자 보고: early24 +0.775(t 1.47, 손절 107) early정규 +0.142 late24 +0.17(t 0.86, 손절 16) late정규 +0.074 | 137: +0.662 -0.149 -0.01 +0.13")
        for p in ("early", "late"):
            for reg in (False, True):
                a = stop_audit(best, p, reg)
                say(f, f"  stop_audit {p} {'정규장만' if reg else '24시간'}: " + ", ".join(
                    f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

        # 2) 하루 의존
        say(f, "\n=== 2) 하루 의존(사용자78 날짜별 합 차이) ===")
        dep = {}
        for p in ("early", "late"):
            for reg in (False, True):
                d = day_series(best, p, reg)
                tot = d.sum()
                top1 = d.iloc[0] if len(d) else 0.0
                top2 = d.iloc[:2].sum() if len(d) else 0.0
                dep[(p, reg)] = (tot, tot - top1, tot - top2)
                say(f, f"  {p} {'정규장만' if reg else '24시간'}: 합 {tot:+.3f}, 바뀐 날 {len(d)} (+{int((d > 0).sum())}/-{int((d < 0).sum())}), "
                       f"최대 이득일 {d.index[0] if len(d) else '-'} {top1:+.3f} -> 빼면 {tot - top1:+.3f}, 상위2 빼면 {tot - top2:+.3f}, "
                       f"상위5 {', '.join(f'{k}:{v:+.3f}' for k, v in d.head(5).items())}")

        # 3) 이웃
        say(f, "\n=== 3) 이웃 매개변수 + 비교(사용자78 차이, early24 | early정규 | late24 | late정규 | 137 early24 | 137 late24 | 손절 e/l) ===")
        NB = [
            ("g 30", dict(g=30)), ("g 33", dict(g=33)), ("g 37", dict(g=37)), ("g 40", dict(g=40)),
            ("y 2%", dict(y=0.02)), ("y 2.5%", dict(y=0.025)), ("y 3.5%", dict(y=0.035)), ("y 4%", dict(y=0.04)),
            ("M 15", dict(M=15)), ("M 20", dict(M=20)), ("M 45", dict(M=45)), ("M 60", dict(M=60)),
            ("z 1%", dict(z=0.01)), ("z 2%", dict(z=0.02)),
            ("conf40 > 35", dict(kind="conf")), ("mode B(종가 시각 기준 완성봉)", dict(mode="B")),
            ("큰 봉 30분", dict(tf=30)), ("큰 봉 60분", dict(tf=60)),
            ("조건 없음(같은 손절)", dict(gate=None)), ("반대 dev40 <= 35", dict(gate="<=")),
        ]
        rows = [("기준안(재현)", C)]
        for name, kw in NB:
            d2 = compare(rule_of(**kw), name, show=False)
            rows.append((name, cells(d2)))
            say(f, f"  [{time.time() - t0:.0f}s] {name} 완료")
        tab = []
        for name, Cx in rows:
            tab.append(dict(이웃=name, e24=Cx["early_24"].차이, e정규=Cx["early_정규"].차이, l24=Cx["late_24"].차이, l정규=Cx["late_정규"].차이,
                            e24_t=Cx["early_24"].t, l24_t=Cx["late_24"].t, e24_137=Cx["early_24_137"].차이, l24_137=Cx["late_24_137"].차이,
                            손절e=Cx["early_24"].손절, 손절l=Cx["late_24"].손절))
        T = pd.DataFrame(tab)
        say(f, T.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        nb = T[~T.이웃.isin(["기준안(재현)", "조건 없음(같은 손절)", "반대 dev40 <= 35"])]
        four = nb[(nb.e24 > 0) & (nb.e정규 > 0) & (nb.l24 > 0) & (nb.l정규 > 0)]
        say(f, f"\n  이웃 {len(nb)}개 중 사용자78 네 칸 모두 플러스 {len(four)}: {list(four.이웃)}")
        say(f, f"  late24 음수 이웃: {list(nb[nb.l24 < 0].이웃)}; late정규 음수 이웃: {list(nb[nb.l정규 < 0].이웃)}; "
               f"early24 음수 이웃: {list(nb[nb.e24 < 0].이웃)}")
        say(f, f"\n총 실행 {time.time() - t0:.0f}s, compare 평가 {1 + len(NB)}회")
    print("저장:", OUT)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def gscan():
    """추가: RSI 기준 g를 1 단위로 촘촘히(31·32·34·36·38). 결과는 같은 출력 파일 끝에 덧붙인다.  사용: py stops/verify_g_waterfall.py gscan"""
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        say(f, "\n=== 4) g 촘촘히(사용자78 차이; 다른 값은 기준안과 같음) ===")
        for g in (31, 32, 34, 36, 38):
            Cx = cells(compare(rule_of(g=g), f"g {g}", show=False))
            say(f, f"  g {g}: e24 {Cx['early_24'].차이:+.3f} e정규 {Cx['early_정규'].차이:+.3f} l24 {Cx['late_24'].차이:+.3f} "
                   f"l정규 {Cx['late_정규'].차이:+.3f} | 손절 e {Cx['early_24'].손절} l {Cx['late_24'].손절} ({time.time() - t0:.0f}s)")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "gscan":
    gscan()
