"""반박 검증 — 흘러내림 시간손절 + 40분봉 RSI 조건 최선안 'F5 M30 max | band 30<dev<=45 보류'를 정의 문장만 보고 다시 구현(2026-10-08).
정의: 프리·애프터 매수(ses[ei] != '정규')만. chk = ei+1 이상에서 처음 오는 '정규장이고 ET 분 >= 570+M' 봉.
  보유 봉 i >= chk, tk[i] < dead[ei], lo < dev[i] <= hi (dev = htf_rsi(A, tf, n)[0], 인덱스 i 그대로, NaN이면 안 팜),
  max(c[ei..i-1]) < 매수가 x (1+Y) 이면 봉 i 시가에 판다(보류형: chk 이후 매 봉 다시 봄). 1회형은 i == chk 한 번만.
  기본값 M=30, Y=+0.5%, lo=30, hi=45, tf=40, n=14, dev.
이웃: lo 25/35, hi 40/50, Y 0/+1.0%, M 15/45, conf, 큰 봉 30·60분, 1회형. 대조: 조건 없음, 반대 조건(dev<=30 보류).
하루 의존: 사용자78 24시간 날짜별 합 차이에서 가장 이득 큰 날 제외. 40분 RSI는 손절 봉마다 5분봉으로 따로 계산해 일치 확인.
사용: py stops/verify_g_grind.py  (code 폴더에서) → ../results/stops/verify_g_grind.txt
"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import stop_lab
from stop_lab import run, compare, stop_audit, base_trades, htf_rsi, UNIV, RAW

_S, _load0 = {}, stop_lab.load


def load(period):          # 같은 값을 한 번만 읽는다(속도). stop_lab.py 파일은 그대로.
    if period not in _S:
        _S[period] = _load0(period)
    return _S[period]


stop_lab.load = load
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_g_grind.txt"))
FH = open(OUT, "w", encoding="utf-8")
U78 = UNIV["사용자78"]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n"); FH.flush()


def factory(lo=30.0, hi=45.0, M=30, Y=0.005, src="dev", tf=40, n=14, defer=True, gate=True, reverse=False):
    """정의 문장 그대로. gate=False면 조건 없음, reverse=True면 x <= lo 일 때만 자름."""
    def make_rule(A):
        c, tk, dead, ses, reg, mod = A["c"], A["tk"], A["dead"], A["ses"], A["reg"], A["mod"]
        N = len(c)
        dev, conf = htf_rsi(A, tf, n)
        x = dev if src == "dev" else conf
        okb = reg & (mod >= 570 + M)
        cand = np.where(okb, np.arange(N), N)
        nxt = np.r_[np.minimum.accumulate(cand[::-1])[::-1], N]      # nxt[k] = k 이상 첫 검사 가능 봉(없으면 N)
        offh = ses != "정규"

        def rule(i, ei, ent):
            if not offh[ei]:
                return False
            ck = nxt[ei + 1]
            if ck >= N:
                return False
            if defer:
                if i < ck:
                    return False
            elif i != ck:
                return False
            if not (tk[i] < dead[ei]):
                return False
            if gate:
                v = x[i]
                if np.isnan(v):
                    return False
                if reverse:
                    if not (v <= lo):
                        return False
                elif not (lo < v <= hi):
                    return False
            hi_close = np.max(c[ei:i])                                 # 봉 ei..i-1 종가
            return hi_close < ent * (1.0 + Y)
        return rule
    return make_rule


def cells(df):
    g = df.set_index(["기간", "범위", "종목"])
    return g


def short(df):
    g = cells(df)
    u = lambda p, r: g.loc[(p, r, "사용자78")]
    a = lambda p, r: g.loc[(p, r, "전체137")]
    e24, er, l24, lr = u("early", "24시간"), u("early", "정규장만"), u("late", "24시간"), u("late", "정규장만")
    return (f"e24 {e24.차이:+.2f}(t {e24.t:+.2f}, 손절 {e24.손절}) eR {er.차이:+.2f} | l24 {l24.차이:+.2f}(t {l24.t:+.2f}, 손절 {l24.손절}) "
            f"lR {lr.차이:+.2f} | 137 e24 {a('early', '24시간').차이:+.2f} l24 {a('late', '24시간').차이:+.2f}")


def day_dep(mk, period):
    t = run(load(period), mk, period, False, syms=U78)
    b = base_trades(period, False)
    b = b[b.sym.isin(U78)]
    days = sorted(set(b.day) | set(t.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d.sort_values(ascending=False), t


def my_rsi40(sym, t_prev_start, c_prev, tf=40, n=14):
    """독립 계산: 봉 i-1 시작 시각이 속한 tf분 묶음(ET 04:00 기준) 이전에 끝난 묶음 종가로 와일더 RSI 상태를 만들고 c[i-1]을 넣는다."""
    if sym not in _M5:
        _M5[sym] = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    m5 = _M5[sym]
    ap = t_prev_start.normalize() + pd.Timedelta(hours=4)
    step = pd.Timedelta(minutes=tf)
    bucket = ap + ((t_prev_start - ap) // step) * step
    x = m5[m5.index < bucket]
    if not len(x):
        return np.nan
    ax = x.index.normalize() + pd.Timedelta(hours=4)
    key = ax + ((x.index - ax) // step) * step
    closes = x.Close.groupby(key).last().values.astype(float)
    if len(closes) <= n:
        return np.nan
    ch = np.diff(closes)
    ag, al = np.maximum(ch[:n], 0).mean(), np.maximum(-ch[:n], 0).mean()
    for d in ch[n:]:
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
    d = c_prev - closes[-1]
    ag = (ag * (n - 1) + max(d, 0)) / n
    al = (al * (n - 1) + max(-d, 0)) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


_M5 = {}


def main():
    t0 = time.time()
    say(__doc__)
    best = factory()
    specs = [
        ("최선안 재현 band 30<dev<=45 보류 M30 Y+0.5%", best),
        ("대조: 조건 없음(같은 손절)", factory(gate=False)),
        ("대조: 반대 조건 dev<=30 보류", factory(reverse=True)),
        ("이웃 lo 25 (25<dev<=45)", factory(lo=25)),
        ("이웃 lo 35 (35<dev<=45)", factory(lo=35)),
        ("이웃 hi 40 (30<dev<=40)", factory(hi=40)),
        ("이웃 hi 50 (30<dev<=50)", factory(hi=50)),
        ("이웃 Y 0% (반등 기준 매수가)", factory(Y=0.0)),
        ("이웃 Y +1.0%", factory(Y=0.01)),
        ("이웃 M15 (9:45 ET)", factory(M=15)),
        ("이웃 M45 (10:15 ET)", factory(M=45)),
        ("이웃 conf (30<conf<=45)", factory(src="conf")),
        ("이웃 큰 봉 30분", factory(tf=30)),
        ("이웃 큰 봉 60분", factory(tf=60)),
        ("이웃 1회형(10:00 ET 한 번)", factory(defer=False)),
    ]
    res = {}
    for nm, mk in specs:
        df = compare(mk, nm, show=False)
        res[nm] = df
        say(f"{nm:<42} | {short(df)}  [{time.time() - t0:.0f}s]")

    say("\n=== 최선안 재현 전체 표")
    f = res[specs[0][0]].copy()
    for k, fm in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                  ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    say(f.drop(columns=["규칙"]).to_string(index=False))

    say("\n=== stop_audit 사용자78 (최선안 / 조건 없음)")
    for p in ("early", "late"):
        for lab, mk in (("최선안", best), ("조건없음", specs[1][1])):
            a = stop_audit(mk, p)
            say(f"{p} 24시간 {lab}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    say("\n=== 하루 의존(사용자78 24시간, 최선안 - 기준)")
    for p in ("early", "late"):
        d, t = day_dep(best, p)
        say(f"{p}: 합 {d.sum():+.3f}, 나은 날 {(d > 1e-12).sum()} 나쁜 날 {(d < -1e-12).sum()}; 상위 3일 "
            + ", ".join(f"{k} {v:+.3f}" for k, v in d.head(3).items())
            + f"; 최고일 빼면 {d.sum() - d.iloc[0]:+.3f}; 상위 2일 빼면 {d.sum() - d.iloc[:2].sum():+.3f}")

    say("\n=== 40분 RSI 독립 계산 일치(최선안 손절 봉, 5분봉에서 직접)")
    for p in ("early", "late"):
        S = load(p)
        t = run(S, best, p, False, syms=U78)
        st = t[t.why == "손절"]
        diffs, inband = [], 0
        for x in st.itertuples():
            A = S[x.sym]; i = x.xi
            v = my_rsi40(x.sym, A["idx"][i - 1], A["c"][i - 1])
            dv = htf_rsi(A, 40, 14)[0][i]
            diffs.append(abs(v - dv))
            inband += int(30 < v <= 45)
        say(f"{p}: 손절 {len(st)}건, |내 계산 - htf_rsi dev| 최대 {np.nanmax(diffs) if diffs else 0:.2e}, 내 계산으로 30~45 안 {inband}건")
    say(f"\n끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
