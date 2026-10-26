"""반박 검증 — 기준봉 RSI + 40분봉 RSI 결합 최선안 'A_rsi<30_15분&d40>30'을 정의 문장만 보고 다시 구현(2026-10-08).
정의: 매수 봉 ei, k = max(1, m // 봉 길이)(early 5분 3봉, late 3분 5봉). 봉 i = ei + k 시가에서 한 번만 본다.
  (1) 기준봉 RSI r[ei..i-1] 최고 < T(=30)  (2) 40분봉 RSI14 진행 중 값 dev[i] > g(=30), dev = htf_rsi(A, 40, 14)[0], NaN이면 안 팜.
  모든 세션. 매도 순서·재매수 금지는 시험대(stop_lab.sim) 그대로.
이웃: T 25/35, m 10/20, g 25/35, conf, 큰 봉 30·60분, 매 봉 확인형. 대조: 조건 없음, 반대 조건(dev <= g).
하루 의존: 사용자78 24시간·정규장만 날짜별 합 차이에서 가장 이득 큰 날 제외.
40분 RSI 독립 점검: 손절 봉의 dev[i]를 5분봉에서 직접(내 Wilder 재귀) 다시 계산해 비교.
사용: py stops/verify_g_rsiboth.py  (code 폴더에서) → ../results/stops/verify_g_rsiboth.txt
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
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_g_rsiboth.txt"))
FH = open(OUT, "w", encoding="utf-8")
U78 = UNIV["사용자78"]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    FH.write(s + "\n"); FH.flush()


def factory(T=30.0, m=15, g=30.0, src="dev", tf=40, n=14, mode="gate", once=True):
    """mode: gate = 40분 값 > g 일 때만 자름, none = 40분 조건 없음, rev = 40분 값 <= g 일 때만 자름.
    once=True면 i == ei+k 한 번만, False면 i >= ei+k 매 봉(그때까지 r 최고 < T 유지 시)."""
    def make_rule(A):
        r = A["r"]
        k = max(1, m // A["mins"])
        dev, conf = htf_rsi(A, tf, n)
        x = dev if src == "dev" else conf

        def rule(i, ei, ent):
            if once:
                if i != ei + k:
                    return False
            elif i < ei + k:
                return False
            hi = -np.inf
            for j in range(ei, i):          # 봉 ei..i-1
                if r[j] > hi:
                    hi = r[j]
            if not (hi < T):
                return False
            if mode == "none":
                return True
            v = x[i]
            if np.isnan(v):
                return False
            return (v > g) if mode == "gate" else (v <= g)
        return rule
    return make_rule


def cells(df):
    out = {}
    for q in df.itertuples():
        out[(q.기간, q.범위, q.종목)] = q
    return out


def brief(label, df):
    c = cells(df)
    parts = []
    for p in ("early", "late"):
        for sc in ("24시간", "정규장만"):
            q = c[(p, sc, "사용자78")]
            parts.append(f"{p[0]}{'24' if sc == '24시간' else '정규'} {q.차이:+.2f}(t {q.t:+.2f}, {q.손절})")
    a1, a2 = c[("early", "24시간", "전체137")], c[("late", "24시간", "전체137")]
    say(f"  {label:34s} " + " | ".join(parts) + f" | 전체 e24 {a1.차이:+.2f} l24 {a2.차이:+.2f}")
    return c


def daydiff(mk, period, reg_only=False):
    t = run(load(period), mk, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    return t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)


# ---------- 40분 RSI 독립 계산(내 구현) ----------
_M5 = {}


def my_dev(sym, t_prev, p, tf=40, n=14):
    """봉 i-1 시작 시각 t_prev(ET), 가격 p = c[i-1]. t_prev가 속한 tf분 봉(04:00 ET 기준) 시작 전에 끝난 큰 봉으로 Wilder RSI 상태를 만들고 p를 넣는다."""
    if sym not in _M5:
        m5 = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
        idx = m5.index if m5.index.tz is not None else m5.index.tz_localize("UTC")
        idx = idx.tz_convert("America/New_York")
        cl = m5["Close"].values.astype(float)
        ok = ~np.isnan(cl)
        idx, cl = idx[ok], cl[ok]
        a4 = idx.normalize() + pd.Timedelta(hours=4)
        # 04:00 이전 봉은 전날 04:00 기준으로 같은 격자(1440 = 36 x 40)라 날짜 기준만 맞으면 된다.
        bin_start = a4 + ((idx - a4) // pd.Timedelta(minutes=tf)) * pd.Timedelta(minutes=tf)
        s = pd.Series(cl, index=bin_start)
        last = s.groupby(level=0).last()
        _M5[sym] = (last.index.as_unit("ns").asi8, last.values)
    bs, bc = _M5[sym]
    a4 = t_prev.normalize() + pd.Timedelta(hours=4)
    cur = a4 + ((t_prev - a4) // pd.Timedelta(minutes=tf)) * pd.Timedelta(minutes=tf)
    k = np.searchsorted(bs, cur.value, side="left")      # 완성 큰 봉 = bs < cur
    c = bc[:k]
    if len(c) <= n:
        return np.nan
    ch = np.diff(c)
    up, dn = np.maximum(ch, 0), np.maximum(-ch, 0)
    ag, al = up[:n].mean(), dn[:n].mean()
    for u, d in zip(up[n:], dn[n:]):
        ag = (ag * (n - 1) + u) / n
        al = (al * (n - 1) + d) / n
    chp = p - c[-1]
    ag = (ag * (n - 1) + max(chp, 0)) / n
    al = (al * (n - 1) + max(-chp, 0)) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def main():
    t0 = time.time()
    say("반박 검증 — A_rsi<30_15분&d40>30 (verify_g_rsiboth.py, 2026-10-08)")
    say("정의만 보고 다시 구현. 숫자 = 사용자78 '차이'(규칙 합 - 기준 합), 괄호 = (날짜별 짝 t, 손절 수)")
    say("")
    best = factory()
    df = compare(best, "재현 A_rsi<30_15분&d40>30", show=False)
    say("[1] 재현 전체 표")
    f = df.drop(columns=["규칙"]).copy()
    for k, fmt in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                   ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    for ln in f.to_string(index=False).splitlines():
        say("  " + ln)
    rep = {"early24u": 0.98, "early정규u": 1.31, "late24u": -0.71, "late정규u": -0.02}
    c = cells(df)
    for key, (p, sc) in (("early24u", ("early", "24시간")), ("early정규u", ("early", "정규장만")),
                         ("late24u", ("late", "24시간")), ("late정규u", ("late", "정규장만"))):
        q = c[(p, sc, "사용자78")]
        say(f"  {key}: 재현 {q.차이:+.3f} vs 보고 {rep[key]:+.2f} → 차 {q.차이 - rep[key]:+.3f}")
    say("")
    say("[2] stop_audit(사용자78)")
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(best, p, reg_only=reg)
            say(f"  {p} {'정규장만' if reg else '24시간'}: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    say("")
    say("[3] 하루 의존(사용자78 날짜별 합 차이)")
    for p in ("early", "late"):
        for reg in (False, True):
            d = daydiff(best, p, reg)
            top = d.sort_values(ascending=False)
            say(f"  {p} {'정규장만' if reg else '24시간'}: 합 {d.sum():+.3f}, 나은 날 {int((d > 1e-12).sum())} 나쁜 날 {int((d < -1e-12).sum())}, "
                f"상위 3일 {', '.join(f'{k} {v:+.3f}' for k, v in top.head(3).items())}, 최상위 1일 빼면 {d.sum() - top.iloc[0]:+.3f}, "
                f"2일 빼면 {d.sum() - top.iloc[:2].sum():+.3f}, 최하위 1일 {top.index[-1]} {top.iloc[-1]:+.3f}")
    say("")
    say("[4] 이웃·대조 (사용자78 네 칸 + 전체137 24시간)")
    variants = [
        ("기준 T30 m15 g30 dev 40분 1회", dict()),
        ("g 25", dict(g=25)), ("g 35", dict(g=35)),
        ("T 25", dict(T=25)), ("T 35", dict(T=35)),
        ("m 10분", dict(m=10)), ("m 20분", dict(m=20)),
        ("conf(완성 40분봉)", dict(src="conf")),
        ("큰 봉 30분", dict(tf=30)), ("큰 봉 60분", dict(tf=60)),
        ("매 봉 확인형(i>=ei+k)", dict(once=False)),
        ("대조: 40분 조건 없음", dict(mode="none")),
        ("대조: 반대(dev<=30에서만)", dict(mode="rev")),
    ]
    res = {}
    for lab, kw in variants:
        dfv = df if not kw else compare(factory(**kw), lab, show=False)
        res[lab] = brief(lab, dfv)
    say("")
    nb = [v for v in res if v not in ("기준 T30 m15 g30 dev 40분 1회",) and not v.startswith("대조")]
    for p, sc in (("early", "24시간"), ("early", "정규장만"), ("late", "24시간"), ("late", "정규장만")):
        vals = [res[v][(p, sc, "사용자78")].차이 for v in nb]
        say(f"  이웃 {len(nb)}개 {p} {sc}: 플러스 {sum(x > 0 for x in vals)} / 마이너스 {sum(x < 0 for x in vals)} "
            f"(범위 {min(vals):+.2f} ~ {max(vals):+.2f})")
    say("")
    say("[5] 40분 RSI 독립 계산 점검(손절 봉 dev[i] vs 5분봉 직접 계산, 사용자78 24시간)")
    for p in ("early", "late"):
        S = load(p)
        t = run(S, best, p, False, syms=U78)
        st = t[t.why == "손절"]
        st = st.sample(min(120, len(st)), random_state=3)
        diffs, gate_flip = [], 0
        for x in st.itertuples():
            A = S[x.sym]
            i = int(x.xi)
            dv = htf_rsi(A, 40, 14)[0][i]
            mv = my_dev(x.sym, A["idx"][i - 1], A["c"][i - 1])
            if np.isnan(mv) or np.isnan(dv):
                continue
            diffs.append(abs(mv - dv))
            gate_flip += int((mv > 30) != (dv > 30))
        diffs = np.array(diffs)
        say(f"  {p}: 비교 {len(diffs)}건, 최대 차이 {diffs.max():.2e}, 0.01 넘는 건 {(diffs > 0.01).sum()}, 30 기준 판정 뒤집힘 {gate_flip}")
    say("")
    say("[6] 참고: 차순위 E_종가-3%&rsi>30&d40>35 (c[i-1] <= 매수가 x0.97, r[i-1] > 30, dev[i] > 35, 보유 중 매 봉)")

    def e_make(A):
        cc, r = A["c"], A["r"]
        dev = htf_rsi(A, 40, 14)[0]

        def rule(i, ei, ent):
            v = dev[i]
            return cc[i - 1] <= ent * 0.97 and r[i - 1] > 30 and (not np.isnan(v)) and v > 35
        return rule
    dfe = compare(e_make, "E", show=False)
    brief("E_종가-3%&rsi>30&d40>35", dfe)
    for p in ("early", "late"):
        d = daydiff(e_make, p)
        top = d.sort_values(ascending=False)
        say(f"  {p} 24시간: 합 {d.sum():+.3f}, 상위 2일 {', '.join(f'{k} {v:+.3f}' for k, v in top.head(2).items())}, "
            f"1일 빼면 {d.sum() - top.iloc[0]:+.3f}, 2일 빼면 {d.sum() - top.iloc[:2].sum():+.3f}")
    say("")
    say(f"걸린 시간 {time.time() - t0:.0f}초")
    FH.close()


if __name__ == "__main__":
    main()
