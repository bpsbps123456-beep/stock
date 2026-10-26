"""반박 검증 — g_fixed 최선안 '고정종가-5%|conf>40|첫신호' 독립 재구현(2026-10-08).

정의(연구자 보고 best_rule.definition만 보고 다시 씀, g_fixed.py 코드 복사 없음):
  매수(ei, ent) 뒤 봉 i(i >= ei+1) 시가 판단. 그 거래에서 c[i-1] <= ent*(1-p)가 '처음' 성립한 봉 i에서만 본다.
  그때 conf[i] > g 이면 판다. conf[i] = 봉 i 시작 시각까지 끝난 마지막 tf분 큰 봉의 RSI(n). NaN·g 이하면 그 거래는 끝까지 손절 안 함.
  40분봉 RSI는 htf.py를 쓰지 않고 5분봉에서 직접 만든다(ET 04:00 기준 tf분 구간, 마지막 종가, Wilder RMA, 첫 n개 변화 단순평균 시작).
  '끝난 큰 봉' = 구간 끝(시작+tf) <= 봉 i 시작 시각. stop_lab.htf_rsi의 conf와 값이 같은지도 센다.
점검: 최선안 재현, stop_audit(early·late), 이웃(g 35·45, p 4·6%, tf 30·60분), 날짜 집중(최대 기여일·상위 2일 제외), late 얇은 체결 영향,
      무작위 조건 대조(-5% 첫 신호 사건 중 conf>40 부분집합 vs 같은 크기 무작위 부분집합).
사용: py stops/verify_g_fixed.py   (결과: ../results/stops/verify_g_fixed.txt)
"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import compare, stop_audit, load, base_trades, run, htf_rsi, UNIV, RAW  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_g_fixed.txt"))
U78 = UNIV["사용자78"]
_M5, _BIG, _CONF = {}, {}, {}


def big_rsi(sym, tf, n):
    """5분봉 → tf분 큰 봉(ET 04:00 기준) 끝 시각(ns, UTC)과 RSI 배열."""
    k = (sym, tf, n)
    if k in _BIG:
        return _BIG[k]
    if sym not in _M5:
        _M5[sym] = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    m = _M5[sym]
    t = m.index.tz_convert("America/New_York")
    a4 = t.normalize() + pd.Timedelta(hours=4)
    bin_start = a4 + ((t - a4) // pd.Timedelta(minutes=tf)) * pd.Timedelta(minutes=tf)
    s = pd.Series(m["Close"].values.astype(float), index=bin_start).groupby(level=0).last().dropna()
    cl = s.values
    rsi = np.full(len(cl), np.nan)
    if len(cl) > n:
        d = np.diff(cl)
        up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
        ag, al = up[:n].mean(), dn[:n].mean()
        for j in range(n, len(cl)):           # rsi[j] = j번째 큰 봉 종가까지
            if j > n:
                ag = (ag * (n - 1) + up[j - 1]) / n
                al = (al * (n - 1) + dn[j - 1]) / n
            rsi[j] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    ends = (s.index + pd.Timedelta(minutes=tf)).as_unit("ns").asi8
    _BIG[k] = (ends, rsi)
    return _BIG[k]


def my_conf(A, tf=40, n=14):
    k = (A["sym"], A["mins"], len(A["tk"]), tf, n)
    if k not in _CONF:
        ends, rsi = big_rsi(A["sym"], tf, n)
        j = np.searchsorted(ends, A["tk"], side="right") - 1          # 끝 <= 봉 i 시작인 마지막 큰 봉
        _CONF[k] = np.where(j >= 0, rsi[np.maximum(j, 0)], np.nan)
    return _CONF[k]


def mk_rule(p=0.05, g=40, tf=40, n=14):
    def make_rule(A):
        c = A["c"]
        cf = my_conf(A, tf, n)

        def rule(i, ei, ent):
            thr = ent * (1 - p)
            if not c[i - 1] <= thr:
                return False
            if i - 1 > ei and c[ei:i - 1].min() <= thr:   # 이미 앞 봉에서 첫 신호가 났음 -> 그때 안 팔았으면 끝까지 안 판다
                return False
            v = cf[i]
            return bool(v == v and v > g)
        return rule
    return make_rule


def cellmap(df):
    return {(r.기간, r.범위, r.종목): r for r in df.itertuples()}


def day_split(mk, period, syms=U78, reg_only=False):
    t = run(load(period), mk, period, reg_only, syms=syms)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(syms)]
    days = sorted(set(b.day) | set(t.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d.sort_values(ascending=False), t


if __name__ == "__main__":
    t0 = time.time()
    L = []
    w = lambda s="": (L.append(s), print(s, flush=True))
    w("반박 검증 — g_fixed 최선안 '고정종가-5%|conf>40|첫신호' 독립 재구현 (2026-10-08)")
    w()
    # 0) 내 40분봉 conf와 stop_lab.htf_rsi conf 일치 여부
    mism, tot = 0, 0
    for per in ("early", "late"):
        S = load(per)
        for s, A in S.items():
            a, b = my_conf(A), htf_rsi(A, 40, 14)[1]
            both_nan = np.isnan(a) & np.isnan(b)
            ok = both_nan | (np.abs(a - b) < 1e-6)
            ok[0] = True                                   # htf_rsi는 [0]을 NaN으로 둠
            mism += int((~ok).sum()); tot += len(a)
    w(f"[0] 독립 40분봉 conf vs stop_lab.htf_rsi conf: 불일치 {mism}/{tot}봉")
    w()
    # 1) 재현 + 이웃
    specs = [("최선 p5 g40 tf40", dict(p=.05, g=40, tf=40)),
             ("g35", dict(p=.05, g=35, tf=40)), ("g45", dict(p=.05, g=45, tf=40)),
             ("p4", dict(p=.04, g=40, tf=40)), ("p6", dict(p=.06, g=40, tf=40)),
             ("tf30", dict(p=.05, g=40, tf=30)), ("tf60", dict(p=.05, g=40, tf=60)),
             ("p4 g35", dict(p=.04, g=35, tf=40)), ("p6 g45", dict(p=.06, g=45, tf=40)),
             ("p4 g45", dict(p=.04, g=45, tf=40)), ("p6 g35", dict(p=.06, g=35, tf=40))]
    rows, full = [], {}
    for lab, sp in specs:
        df = compare(mk_rule(**sp), lab, show=False)
        full[lab] = df
        m = cellmap(df)
        r = dict(규칙=lab)
        for per in ("early", "late"):
            for sc, k in (("24시간", "24"), ("정규장만", "Reg")):
                x = m[(per, sc, "사용자78")]
                r[f"{per[0]}{k}"] = x.차이; r[f"{per[0]}{k}t"] = x.t; r[f"{per[0]}{k}손절"] = x.손절
            r[f"{per[0]}24_137"] = m[(per, "24시간", "전체137")].차이
            r[f"{per[0]}Reg_137"] = m[(per, "정규장만", "전체137")].차이
        rows.append(r)
        print(f"  {lab} {time.time() - t0:.0f}s", flush=True)
    sm = pd.DataFrame(rows)
    w("[1] 재현·이웃 (사용자78 차이, t, 손절 수; _137 = 전체137 차이)")
    w(sm.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    w()
    w("[1b] 최선안 전체 표")
    w(full["최선 p5 g40 tf40"].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    w()
    rep = {"e24": +0.6186, "eReg": +0.4324, "l24": +0.1723, "lReg": +0.0722}
    b0 = sm.iloc[0]
    w("연구자 보고 대비(내 값 - 보고 값): " + ", ".join(f"{k} {b0[k] - v:+.4f}" for k, v in rep.items()))
    w()
    # 2) stop_audit
    mk = mk_rule(p=.05, g=40, tf=40)
    w("[2] stop_audit 사용자78")
    for per in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk, per, reg_only=reg)
            w(f"  {per} {'정규장만' if reg else '24시간'}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
    w()
    # 3) 날짜 집중
    w("[3] 날짜 집중 (사용자78, 날짜별 '규칙 합 - 기준 합')")
    for per in ("early", "late"):
        for reg in (False, True):
            d, t = day_split(mk, per, reg_only=reg)
            w(f"  {per} {'정규장만' if reg else '24시간'}: 합 {d.sum():+.3f}, 상위 3일 {', '.join(f'{k} {v:+.3f}' for k, v in d.head(3).items())}; "
              f"최대 1일 빼면 {d.sum() - d.iloc[0]:+.3f}, 상위 2일 빼면 {d.sum() - d.iloc[:2].sum():+.3f}; 하위 1일 {d.index[-1]} {d.iloc[-1]:+.3f}; "
              f"나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}")
    w()
    # 4) late 손절 거래별 득실과 얇은 체결
    w("[4] late 24시간 사용자78 손절 거래별 (손절 수익 - 기준 수익)")
    S = load("late")
    _, t = day_split(mk, "late")
    b = base_trades("late").set_index(["sym", "ei"]).ret
    st = t[t.why == "손절"].copy()
    st["기준"] = [b.get((r.sym, r.ei), np.nan) for r in st.itertuples()]
    st["득실"] = st.ret - st.기준
    st["매도세션"] = [S[r.sym]["ses"][r.xi] for r in st.itertuples()]
    st["c_prev/ent"] = [S[r.sym]["c"][r.xi - 1] / r.ent - 1 for r in st.itertuples()]
    st["conf"] = [my_conf(S[r.sym])[r.xi] for r in st.itertuples()]
    w(st[["sym", "day", "ses", "매도세션", "ret", "기준", "득실", "c_prev/ent", "conf"]].sort_values("득실", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    thin = st.px > st.ent
    w(f"  손절 {len(st)}건, 득실 합 {st.득실.sum():+.3f}; 체결가>매수가(얇은 체결 흔적) {int(thin.sum())}건 득실 {st.득실[thin].sum():+.3f}; "
      f"정상 {int((~thin).sum())}건 득실 {st.득실[~thin].sum():+.3f}; 최대 1건 빼면 {st.득실.sum() - st.득실.max():+.3f}")
    w()
    # 5) 무작위 조건 대조: 기준 거래(손절 없음)에서 종가 -5% 첫 신호 사건마다 '그때 팔기 - 끝까지'를 구하고,
    #    conf>40 부분집합 합이 같은 크기 무작위 부분집합 합보다 큰지(조건이 '덜 자르기' 말고 고르는 힘이 있는지). 재매수 경로 효과는 뺌.
    w("[5] 무작위 조건 대조 (사용자78 24시간, 기준 거래의 -5% 첫 신호 사건, 1만 번)")
    rng = np.random.default_rng(0)
    for per in ("early", "late"):
        S = load(per)
        b = base_trades(per)
        b = b[b.sym.isin(U78)]
        val, gate = [], []
        for r in b.itertuples():
            A = S[r.sym]
            c, o = A["c"], A["o"]
            thr = r.ent * 0.95
            for i in range(r.ei + 1, r.xi):
                if c[i - 1] <= thr:
                    val.append(o[i] / r.ent - 1 - r.ret)
                    v = my_conf(A)[i]
                    gate.append(v == v and v > 40)
                    break
        val, gate = np.array(val), np.array(gate)
        k = int(gate.sum())
        sims = np.array([val[rng.choice(len(val), k, replace=False)].sum() for _ in range(10000)])
        w(f"  {per}: 사건 {len(val)}건(전체 합 {val.sum():+.3f}, 평균 {val.mean():+.4f}), conf>40 {k}건 합 {val[gate].sum():+.3f} "
          f"(평균 {val[gate].mean():+.4f}) / conf<=40·NaN {len(val) - k}건 합 {val[~gate].sum():+.3f}; "
          f"무작위 {k}건 합 평균 {sims.mean():+.3f}, 95% {np.quantile(sims, .95):+.3f}, 무작위가 같거나 큰 비율 p={np.mean(sims >= val[gate].sum()):.4f}")
    w()
    # 6) 실행 가능성: 사용자는 KST 0~8시(ET 11~19시)에 자므로 손절(아래 매도) 주문을 못 낸다. 첫 신호 봉 i 시가가 KST 8~24시일 때만 판다.
    #    그 밖에서 첫 신호가 나면 다음 아침 8시 정리까지 들고 간다(첫신호 방식 그대로).
    def mk_kst(A):
        base = mk_rule(p=.05, g=40, tf=40)(A)
        hk = ((A["tk"] // 10**9 // 3600) + 9) % 24
        return lambda i, ei, ent: base(i, ei, ent) and hk[i] >= 8
    w("[6] 실행 가능 시간 한정(손절 판단 봉 시가 KST 8~24시만) — 사용자78·전체137")
    dfk = compare(mk_kst, "최선+KST8~24", show=False)
    w(dfk[["기간", "범위", "종목", "차이", "t", "손절", "최악"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    w()
    w(f"소요 {time.time() - t0:.0f}s")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
