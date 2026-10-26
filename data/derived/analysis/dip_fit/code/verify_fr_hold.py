"""반박 검증: fr_hold(빠른 반등 뒤 보유 연장) 참고 행 3개(b10 K=30, a2 K=20, a3 K=20)와 핵심 가설(빠른 반등 뒤 더 오른다).

검증 항목(시험 전에 고정):
  V1 재현: 기준 거래 1856건, F(K) 판정을 내 RSI(Wilder RMA, Pine ta.rsi와 같은 식)로 다시 잰다(r[xi-1] > 70, ei..xi-2 봉은 <= 70).
  V2 독립 재계산: 1분봉 원본에서 a2·a3·a5·b10·e 매도가를 fr_hold 코드를 쓰지 않고 다시 계산해 거래별 차이를 비교한다.
  V3 꼬리 체결: (1) 정규장 매수인데 RSI 매도가가 애프터 봉 시가인 사건, (2) 연장 매도가 08시 가격(애프터 19시대 봉 시가)인 사건,
     (3) 정규장 위쪽 지정가가 1분 스파이크(고가만 닿고 종가는 아래, 고가 > 직전 1분 종가 +1%)로 체결된 사건. 직전 1분 종가 대비 |1%| 초과를 꼬리로 본다.
  V4 쏠림: 참고 행의 날짜별·종목별 차이 합, 상위 1일·1종목 제거, F(20) 날의 QQQ 일간 수익(전일 종가→종가)과 그날 F 사건 몰림.
  V5 공정 기준 대안: ±30분 창 대신 T 이후 30분 창([T, T+30분], 같은 날·같은 세션)으로 다시 잰다. S에서도 치우침이 사라지는지 본다.
  V6 자는 동안 실행: T(RSI 매도 순간)가 한국 00~08시면 지정가를 새로 걸 수 없다. 깨어 있는 T만으로 참고 행을 다시 잰다.
  V7 우연(순열): 같은 날 안에서 bars(빠름 표시)를 섞어 1000번(7~9월 보조도 같이 섞음). fr_hold의 고르기→확인 절차 전체를 그대로 돌려
     '확인됨'이 나올 확률, 참고 행 뒤 절반 합의 p값, 24조합 뒤 절반 최대 t의 분포, 핵심 경로(F20-S20 08시) p값을 낸다.
출력: ../results/verify_fr_hold.txt
사용: py verify_fr_hold.py (약 1~2분)"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fr_hold as FH
import build

RAW = FH.RAW
RES = FH.RES
NS = 60 * 10**9
RULES = FH.RULES
KS = FH.KS
MID = FH.MID
NPERM = 1000
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def pct(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{nd}f}%"


# ---------------- 독립 구현 ----------------

def my_rsi(c, n=14):
    """Wilder RSI: 첫 값은 n개 변화의 단순 평균, 이후 RMA(alpha=1/n)."""
    ch = np.diff(c, prepend=np.nan)
    up, dn = np.where(ch > 0, ch, 0.0), np.where(ch < 0, -ch, 0.0)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    au, ad = up[1:n + 1].mean(), dn[1:n + 1].mean()
    for i in range(n, len(c)):
        if i > n:
            au = (au * (n - 1) + up[i]) / n
            ad = (ad * (n - 1) + dn[i]) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


_M = {}


def load(s):
    if s not in _M:
        m = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
        m = m[m.High > 0]
        b = m.resample("3min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
        mod1 = m.index.hour * 60 + m.index.minute
        _M[s] = dict(m=m, t1=m.index.as_unit("ns").asi8, o1=m.Open.values, h1=m.High.values, l1=m.Low.values,
                     c1=m.Close.values, reg1=np.asarray((mod1 >= 570) & (mod1 < 960)),
                     b=b, tb=b.index.as_unit("ns").asi8, ob=b.Open.values, cb=b.Close.values,
                     e10=b.Close.ewm(span=10, adjust=False).mean().values, rsi=my_rsi(b.Close.values.astype(float)))
    return _M[s]


def dead_of(t_buy):
    k = t_buy.tz_convert("Asia/Seoul")
    t8 = k.normalize() + pd.Timedelta(hours=8)
    d = t8 if k < t8 else t8 + pd.Timedelta(days=1)
    return d.value


def indep_exit(M, T, dead, rule, P):
    """T(ns) = RSI 매도 봉 시작, P = 그 봉 시가. 반환 (가격, 시각 ns, 08시 여부)."""
    tb, ob = M["tb"], M["ob"]
    jd = int(np.searchsorted(tb, dead, "left"))
    s0 = int(np.searchsorted(tb, T, "left"))
    if jd >= len(tb):
        return np.nan, np.nan, False
    if jd <= s0:
        return ob[s0], tb[s0], False
    if rule == "e":
        return ob[jd], tb[jd], True
    if rule[0] == "a":
        L = P * (1 + int(rule[1:]) / 100)
        t1, h1, c1, o1, reg1 = M["t1"], M["h1"], M["c1"], M["o1"], M["reg1"]
        k0, k1 = int(np.searchsorted(t1, T, "left")), int(np.searchsorted(t1, dead, "left"))
        for k in range(max(k0, 1), k1):
            if h1[k] >= L and (reg1[k] or L <= c1[k - 1] * 1.01):
                return (max(L, o1[k]) if reg1[k] else L), t1[k], False
        return ob[jd], tb[jd], True
    if rule == "b10":
        cb, e10 = M["cb"], M["e10"]
        for j in range(s0, jd):
            if cb[j] < e10[j]:
                return ob[j + 1], tb[j + 1], (j + 1 == jd)
        return ob[jd], tb[jd], True
    raise ValueError(rule)


def tail_at(M, t_ns, px):
    """t_ns에 시작하는 1분의 가격 px가 직전 1분 종가 대비 몇 %인가."""
    k = int(np.searchsorted(M["t1"], t_ns, "left"))
    if k == 0 or k >= len(M["t1"]):
        return np.nan
    return px / M["c1"][k - 1] - 1


# ---------------- 통계 ----------------

def crt(x, g):
    return FH.crt(x, g)


def crt_mat(X, g_codes):
    """X: n x m, g_codes: 0..G-1 정수. 열마다 날짜 군집 t(CR1). G<3이면 nan."""
    n = X.shape[0]
    if n < 2:
        return np.full(X.shape[1], np.nan)
    u, gi = np.unique(g_codes, return_inverse=True)
    G = len(u)
    if G < 3:
        return np.full(X.shape[1], np.nan)
    m = X.mean(0)
    R = X - m
    S = np.vstack([np.bincount(gi, R[:, j], minlength=G) for j in range(X.shape[1])])
    se = np.sqrt((S ** 2).sum(1) * G / (G - 1)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, m / se, np.nan)


def drop_top(d, sym, day):
    """차이 합이 가장 큰 1종목·1일을 뺀 합."""
    bs = pd.Series(d).groupby(sym).sum()
    bd = pd.Series(d).groupby(day).sum()
    return d[sym != bs.idxmax()].sum(), d[day != bd.idxmax()].sum()


# ---------------- 실행 ----------------

def main():
    say("== 반박 검증: fr_hold 빠른 반등 뒤 보유 연장 ==")
    say(__doc__.split("검증 항목(시험 전에 고정):")[1].split("출력:")[0].rstrip())
    E, B, n_ref, n_all, ref = FH.late_events()
    Ee, Be = FH.early_events()
    E["half"] = E.day.map(FH.half_of)
    reg = E[E.ses == "정규"].reset_index(drop=True)
    rege = Ee[Ee.ses == "정규"].reset_index(drop=True)
    for X in (reg, rege):
        for ru in RULES:
            X[f"d_{ru}"] = (X[f"x_{ru}"] - X.px) / X.ent
    say(f"\n[V1 재현] 기준 거래 {n_all}건(trades_1009 {n_ref}건) | 정규장 매수 RSI 매도 {len(reg)}건 | F(10/20/30) "
        f"{(reg.bars <= 10).sum()}/{(reg.bars <= 20).sum()}/{(reg.bars <= 30).sum()} | 7~9월 정규 {len(rege)}건 F(20) {(rege.bars <= 20).sum()}")

    # V1: 내 RSI로 RSI 매도 봉 확인
    bad, chk = 0, 0
    for r in reg.itertuples():
        M = load(r.sym)
        tb = M["tb"]
        ei = int(np.searchsorted(tb, pd.Timestamp(r.t_buy).value, "left"))
        xi = int(np.searchsorted(tb, pd.Timestamp(r.t_sell).value, "left"))
        rs = M["rsi"]
        ok = (xi - ei == r.bars) and rs[xi - 1] > 70 and (xi - 1 == ei or np.all(rs[ei:xi - 1] <= 70)) \
            and abs(M["ob"][xi] - r.px) < 1e-6 * r.px
        chk += 1
        bad += (not ok)
    say(f"  내 RSI·내 3분봉으로 RSI 매도 봉·bars·매도가 일치: {chk - bad}/{chk}건")
    # 'bars'가 시간이 아닌 봉 수: 애프터로 넘어간 사건
    reg["min_held"] = (pd.to_datetime(reg.t_sell, utc=True) - pd.to_datetime(reg.t_buy, utc=True)).dt.total_seconds() / 60
    odd = reg[(reg.bars <= 30) & (reg.min_held > reg.bars * 3 + 1)]
    say(f"  F(30) 중 봉 수 x 3분보다 실제 보유 시간이 긴 사건(애프터 빈 봉): {len(odd)}건 " +
        ", ".join(f"{r.sym} {r.day} {r.bars}봉={r.min_held:.0f}분" for r in odd.itertuples()))

    # V2: 독립 재계산
    say("\n[V2 독립 재계산] 1분봉 원본에서 a2·a3·a5·b10·e 매도가(정규장 매수 RSI 매도 전 사건)")
    cmp_rules = ["a2", "a3", "a5", "b10", "e"]
    mx = {ru: 0.0 for ru in cmp_rules}
    nd = {ru: 0 for ru in cmp_rules}
    for i, r in reg.iterrows():
        M = load(r.sym)
        T = pd.Timestamp(r.t_sell).value
        dead = dead_of(pd.Timestamp(r.t_buy))
        for ru in cmp_rules:
            px, tx, at8 = indep_exit(M, T, dead, ru, r.px)
            reg.loc[i, f"my_{ru}"] = px
            reg.loc[i, f"my8_{ru}"] = at8
            reg.loc[i, f"mytx_{ru}"] = tx
            both_nan = np.isnan(px) and np.isnan(r[f"x_{ru}"])
            dv = 0.0 if both_nan else abs(px - r[f"x_{ru}"]) / r.ent
            dv = dv if np.isfinite(dv) else 9.0
            mx[ru] = max(mx[ru], dv)
            nd[ru] += int(dv > 1e-6)
    say(f"  08시 가격이 없는 사건(10/9 애프터 19시 뒤 봉 없음, 모두 S): {int(reg.x_e.isna().sum())}건 — 두 계산 모두 빈칸으로 둔다")
    say("  규칙별 fr_hold와 다른 사건 수(최대 차이): " + ", ".join(f"{ru} {nd[ru]}건({100 * mx[ru]:.4f}%p)" for ru in cmp_rules))

    # V3: 꼬리 체결
    say("\n[V3 꼬리 체결] 직전 1분 종가 대비 |1%| 초과")
    F30 = reg[reg.bars <= 30]
    ah = F30[F30.ses_x != "정규"]
    for r in ah.itertuples():
        M = load(r.sym)
        say(f"  정규 매수·애프터 RSI 매도: {r.sym} {r.day} {pd.Timestamp(r.t_sell):%H:%M} P {r.px:.2f} 직전 1분 대비 "
            f"{pct(tail_at(M, pd.Timestamp(r.t_sell).value, r.px))} (bars {r.bars})")
    n8, t8l = 0, []
    for i, r in reg.iterrows():
        M = load(r.sym)
        for ru in ("a2", "a3", "e"):
            if r[f"my8_{ru}"]:
                tv = tail_at(M, int(r[f"mytx_{ru}"]), r[f"my_{ru}"])
                reg.loc[i, f"t8_{ru}"] = tv
    for K in (20, 30):
        for ru in ("a2", "a3", "e"):
            z = reg[(reg.bars <= K) & reg[f"my8_{ru}"].astype(bool)]
            tl = z[f"t8_{ru}"].abs() > 0.01
            say(f"  F({K}) {ru} 08시 정리 {len(z)}건 중 08시 가격 꼬리 {int(tl.sum())}건 " +
                ", ".join(f"{r.sym} {r.day} {pct(r[f't8_{ru}'])}" for _, r in z[tl].iterrows()))
    # 정규장 지정가 1분 스파이크 체결
    sp = []
    for i, r in reg[reg.bars <= 30].iterrows():
        M = load(r.sym)
        for ru in ("a2", "a3", "a5"):
            if r[f"my8_{ru}"]:
                continue
            k = int(np.searchsorted(M["t1"], int(r[f"mytx_{ru}"]), "left"))
            L = r.px * (1 + int(ru[1:]) / 100)
            if M["reg1"][k] and M["c1"][k] < L and M["h1"][k] > M["c1"][k - 1] * 1.01:
                sp.append(f"{r.sym} {r.day} {ru}(고가 {M['h1'][k]:.2f} 지정가 {L:.2f} 종가 {M['c1'][k]:.2f})")
    say(f"  F(30) 정규장 지정가 체결 중 1분 스파이크(고가 > 직전 종가 +1%, 종가 < 지정가): {len(sp)}건 " + "; ".join(sp))

    # V4: 쏠림
    say("\n[V4 쏠림] 참고 행 날짜별 차이 합(%)과 QQQ 일간 수익")
    qd = pd.read_parquet(os.path.join(RAW, "1d_QQQ.parquet")).Close
    qd.index = pd.to_datetime(qd.index).date
    qret = (qd / qd.shift(1) - 1)
    for ru, K in (("b10", 30), ("a2", 20), ("a3", 20)):
        F = reg[reg.bars <= K]
        dd = F.groupby("day")[f"d_{ru}"].agg(["sum", "count"])
        say(f"  {ru} K={K}: " + " ".join(f"{d.strftime('%m/%d')}:{100 * v['sum']:+.1f}({int(v['count'])},QQQ{100 * qret.get(d, np.nan):+.1f})"
                                        for d, v in dd.iterrows()))
        for h in ("앞", "뒤"):
            Fh = F[F.half == h]
            d = Fh[f"d_{ru}"].values
            s1, s2 = drop_top(d, Fh.sym.values, Fh.day.values)
            say(f"    {h}: n {len(Fh)} 합 {pct(d.sum(), 1)} t {crt(d, Fh.day):+.2f} | 1종목 빼면 {pct(s1, 1)} 1일 빼면 {pct(s2, 1)} | "
                f"최대 1일 {pct(Fh.groupby('day')[f'd_{ru}'].sum().max(), 1)} 최소 1일 {pct(Fh.groupby('day')[f'd_{ru}'].sum().min(), 1)}")
    F20 = reg[reg.bars <= 20]
    qq = F20.day.map(qret)
    say(f"  F(20) 21건의 그날 QQQ: 평균 {pct(qq.mean())}, +1% 넘는 날 {int((qq > 0.01).sum())}건, -1% 밑 날 {int((qq < -0.01).sum())}건 | "
        f"S(20) 그날 QQQ 평균 {pct(reg[reg.bars > 20].day.map(qret).mean())}")
    for ru in ("a2", "a3"):
        say(f"  QQQ 통제 {ru} K=20 뒤: 평균 {pct((F20[F20.half == '뒤'][f'ext_{ru}'] - F20[F20.half == '뒤'][f'q_{ru}']).mean())}, "
            f"전체 {pct((F20[f'ext_{ru}'] - F20[f'q_{ru}']).mean())} (t {crt(F20[f'ext_{ru}'] - F20[f'q_{ru}'], F20.day):+.2f})")

    # V5: 공정 기준 대안(T 이후 30분 창)
    say("\n[V5 공정 기준 대안] [T, T+30분] 같은 날·같은 세션 봉 진입, 같은 규칙. 사건 ext - 공정 평균")
    syms = reg.sym.unique()
    post = {ru: np.full(len(reg), np.nan) for ru in RULES}
    for s in syms:
        d, dead_a, hot = build.arrays(s)
        m1 = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
        P = FH.pack(d, 3, dead_a, 10, 25, m1)
        for i in np.flatnonzero(reg.sym.values == s):
            xi = int(reg.xi.iloc[i])
            W = [j for j in FH.window(P, xi, 10) if j >= xi]
            for ru in RULES:
                fv = []
                for j in W:
                    y = FH.rule_exit(P, j, P["dead"][j], ru)
                    if y:
                        fv.append(y[0] / P["o"][j] - 1)
                post[ru][i] = np.mean(fv) if fv else np.nan
    for ru in RULES:
        reg[f"fp_{ru}"] = post[ru]
    say("  규칙 | F(20) 원 공정초과(t) → T 이후 창 초과(t) | S(20) 원 공정초과 → T 이후 창 초과(t) | 뒤 절반 F(20) T 이후 창 초과(t)")
    for ru in RULES:
        F, S_ = reg[reg.bars <= 20], reg[reg.bars > 20]
        a, b = F[f"ext_{ru}"] - F[f"f_{ru}"], F[f"ext_{ru}"] - F[f"fp_{ru}"]
        c, e = S_[f"ext_{ru}"] - S_[f"f_{ru}"], S_[f"ext_{ru}"] - S_[f"fp_{ru}"]
        Fb = F[F.half == "뒤"]
        g = Fb[f"ext_{ru}"] - Fb[f"fp_{ru}"]
        say(f"  {ru:>3} | {pct(a.mean())}({crt(a, F.day):+.1f}) → {pct(b.mean())}({crt(b, F.day):+.1f}) | "
            f"{pct(c.mean())}({crt(c, S_.day):+.1f}) → {pct(e.mean())}({crt(e, S_.day):+.1f}) | {pct(g.mean())}({crt(g, Fb.day):+.1f})")

    # V6: 자는 동안 실행
    say("\n[V6 자는 동안] T(RSI 매도 순간) 한국 시각. 한국 00~08시면 새 지정가를 못 건다(지금 RSI 시장가 매도도 못 함)")
    kh = pd.to_datetime(reg.t_sell, utc=True).dt.tz_convert("Asia/Seoul").dt.hour
    reg["awake"] = (kh >= 8)
    for ru, K in (("a2", 20), ("a3", 20), ("b10", 30)):
        F = reg[reg.bars <= K]
        for lab, Z in (("깨어 있는 T", F[F.awake]), ("자는 T", F[~F.awake])):
            Zb = Z[Z.half == "뒤"]
            say(f"  {ru} K={K} {lab}: 전체 n {len(Z)} 합 {pct(Z[f'd_{ru}'].sum(), 1)} | 뒤 n {len(Zb)} 합 {pct(Zb[f'd_{ru}'].sum(), 1)} "
                f"t {crt(Zb[f'd_{ru}'], Zb.day):+.2f}")
    a8 = reg[(reg.bars <= 20)]
    say(f"  F(20) 21건 중 T가 깨어 있는 시간 {int(a8.awake.sum())}건 / 자는 시간 {int((~a8.awake).sum())}건 "
        f"(자는 T: {', '.join(f'{r.sym} {r.day} {pd.Timestamp(r.t_sell):%H:%M}ET' for r in a8[~a8.awake].itertuples())})")
    nfill = {ru: int((~a8[f"my8_{ru}"].astype(bool)).sum()) for ru in ("a2", "a3")}
    say(f"  위쪽 지정가 체결 F(20): a2 {nfill['a2']}/21, a3 {nfill['a3']}/21. 못 닿으면 08시(애프터 19시대 봉) 정리")

    # V7: 순열
    say(f"\n[V7 우연] 같은 날 안에서 bars 섞기 {NPERM}번. fr_hold 고르기→확인 절차 전체를 그대로 돌린다")
    say("  08시 가격이 없는 5건(10/9 S)은 섞기에서 뺀다(fr_hold 합·t도 이 5건을 건너뛴다)")
    reg = reg[reg[[f"d_{ru}" for ru in RULES]].notna().all(axis=1)].reset_index(drop=True)
    rng = np.random.default_rng(20261010)
    dcode = pd.factorize(reg.day)[0]
    dcode_e = pd.factorize(rege.day)[0]
    front = (reg.half == "앞").values
    back = ~front
    D = reg[[f"d_{ru}" for ru in RULES]].values
    EX = (reg[[f"ext_{ru}" for ru in RULES]].values - reg[[f"f_{ru}" for ru in RULES]].values)
    QA = (reg[[f"ext_{ru}" for ru in RULES]].values - reg[[f"q_{ru}" for ru in RULES]].values)
    De = rege[[f"d_{ru}" for ru in RULES]].values
    H8 = reg["h_08시"].values
    sym_a, day_a = reg.sym.values, reg.day.values
    say(f"  NaN 확인: 정규 차이 {int(np.isnan(D).sum())}, 공정 {int(np.isnan(EX).sum())}, QQQ {int(np.isnan(QA).sum())}, 7~9월 {int(np.isnan(De).sum())}")
    EX = np.nan_to_num(EX)
    QA = np.nan_to_num(QA)
    De = np.nan_to_num(De)
    groups = [np.flatnonzero(dcode == g) for g in range(dcode.max() + 1)]
    groups_e = [np.flatnonzero(dcode_e == g) for g in range(dcode_e.max() + 1)]
    bars0, bars0e = reg.bars.values.copy(), rege.bars.values.astype(float).copy()

    def shuffle(b, grp):
        nb = b.copy()
        for ix in grp:
            if len(ix) > 1:
                nb[ix] = b[rng.permutation(ix)]
        return nb

    def procedure(bb, bbe):
        """fr_hold 3번과 같은 절차. 반환: 고른 수, 확인됨 수, 참고 행 뒤 합들, 24조합 뒤 최대 t, 경로 F20-S20."""
        cand = []
        back_t = []
        back_sum = {}
        for K in KS:
            mF = front & (bb <= K)
            mB = back & (bb <= K)
            sF, tF = D[mF].sum(0), crt_mat(D[mF], dcode[mF])
            sB, tB = D[mB].sum(0), crt_mat(D[mB], dcode[mB])
            back_t.extend(list(tB))
            for j, ru in enumerate(RULES):
                back_sum[(ru, K)] = sB[j]
                if sF[j] > 0:
                    cand.append((tF[j] if np.isfinite(tF[j]) else -np.inf, ru, K, j))
        cand.sort(key=lambda x: -x[0])
        seen, picks = set(), []
        for c in cand:
            if c[1] not in seen:
                seen.add(c[1])
                picks.append(c)
            if len(picks) == 3:
                break
        conf = 0
        for _, ru, K, j in picks:
            mB = back & (bb <= K)
            d = D[mB, j]
            if d.sum() <= 0:
                continue
            t = crt_mat(d[:, None], dcode[mB])[0]
            s1, s2 = drop_top(d, sym_a[mB], day_a[mB])
            ex, qa = EX[mB, j].mean(), QA[mB, j].mean()
            es = De[bbe <= K, j].sum()
            if np.isfinite(t) and t >= 2 and s1 > 0 and s2 > 0 and ex > 0 and qa > 0 and es > 0:
                conf += 1
        f20 = bb <= 20
        path = np.nanmean(H8[f20]) - np.nanmean(H8[~f20])
        return len(picks), conf, back_sum, np.nanmax(back_t), path

    obs = procedure(bars0, bars0e)
    say(f"  관측: 고른 수 {obs[0]}, 확인됨 {obs[1]}, 뒤 절반 24조합 최대 t {obs[3]:+.2f}, F20-S20 08시 경로 {pct(obs[4])}")
    keys = [("a2", 20), ("a3", 20), ("b10", 30)]
    n_pick, n_conf, mt, pth = [], [], [], []
    bs = {k: [] for k in keys}
    early_a2 = []
    obs_e = {k: De[bars0e <= k[1], RULES.index(k[0])].sum() for k in keys}
    for _ in range(NPERM):
        bb = shuffle(bars0, groups)
        bbe = shuffle(bars0e, groups_e)
        p = procedure(bb, bbe)
        n_pick.append(p[0])
        n_conf.append(p[1])
        mt.append(p[3])
        pth.append(p[4])
        for k in keys:
            bs[k].append(p[2][k])
        early_a2.append({k: De[bbe <= k[1], RULES.index(k[0])].sum() for k in keys})
    n_pick, n_conf, mt, pth = map(np.asarray, (n_pick, n_conf, mt, pth))
    say(f"  섞은 뒤: 앞 절반에서 1개 이상 고름 {np.mean(n_pick > 0):.1%}, '확인됨' 1개 이상 {np.mean(n_conf > 0):.1%} "
        f"(절차의 우연 통과율)")
    say(f"  뒤 절반 24조합 최대 t: 섞은 분포 중앙 {np.median(mt):+.2f}, 95% {np.quantile(mt, .95):+.2f} | 관측 {obs[3]:+.2f} → p {np.mean(mt >= obs[3]):.2f}")
    for k in keys:
        a = np.asarray(bs[k])
        e_null = np.asarray([x[k] for x in early_a2])
        say(f"  {k[0]} K={k[1]} 뒤 합 관측 {pct(obs[2][k], 1)}: 섞은 중앙 {pct(np.median(a), 1)} 평균 {pct(a.mean(), 1)}, p(>=관측) {np.mean(a >= obs[2][k]):.2f} | "
            f"7~9월 합 관측 {pct(obs_e[k], 1)}: 섞은 중앙 {pct(np.median(e_null), 1)} 평균 {pct(e_null.mean(), 1)}, p(>=관측) {np.mean(e_null >= obs_e[k]):.2f}")
    say(f"  핵심 경로 F20-S20(08시, 정규 전체): 관측 {pct(obs[4])}, 섞은 중앙 {pct(np.median(pth))}, "
        f"양쪽 p {np.mean(np.abs(pth - np.median(pth)) >= abs(obs[4] - np.median(pth))):.2f}")

    # V8: 같은 날 S로 날짜 효과 빼기
    say("\n[V8 같은 날 S 빼기] F 사건 차이 - 같은 날 S 사건 차이 평균(같은 규칙). 같은 날 S가 없으면 뺀다")
    for ru, K in keys + [("e", 20)]:
        for lab, X in (("앞", reg[reg.half == "앞"]), ("뒤", reg[reg.half == "뒤"]), ("전체", reg), ("7~9월", rege)):
            F, S_ = X[X.bars <= K], X[X.bars > K]
            sm = S_.groupby("day")[f"d_{ru}"].mean()
            adj = F[f"d_{ru}"] - F.day.map(sm)
            ok = adj.notna()
            say(f"  {ru} K={K} {lab}: n {int(ok.sum())} F 평균 {pct(F[f'd_{ru}'][ok].mean())} 같은 날 S 평균 {pct(F.day.map(sm)[ok].mean())} "
                f"→ 초과 평균 {pct(adj[ok].mean())} 합 {pct(adj[ok].sum(), 1)} t {crt(adj[ok], F.day[ok]):+.2f}")
    F20 = reg[reg.bars <= 20]
    sm = reg[reg.bars > 20].groupby("day")["h_08시"].mean()
    adj = F20["h_08시"] - F20.day.map(sm)
    say(f"  핵심 경로(RSI 매도가 대비 08시) F(20) - 같은 날 S(20): 정규 전체 {pct(adj.mean())}(t {crt(adj.dropna(), F20.day[adj.notna()]):+.2f}), "
        f"앞 {pct(adj[F20.half == '앞'].mean())}, 뒤 {pct(adj[F20.half == '뒤'].mean())}")
    F20e = rege[rege.bars <= 20]
    sme = rege[rege.bars > 20].groupby("day")["h_08시"].mean()
    adje = F20e["h_08시"] - F20e.day.map(sme)
    say(f"  7~9월 F(20) - 같은 날 S: {pct(adje.mean())}(t {crt(adje.dropna(), F20e.day[adje.notna()]):+.2f}), 원 F-S {pct(F20e['h_08시'].mean() - rege[rege.bars > 20]['h_08시'].mean())}")

    # 요약 표
    say("\n[요약] 참고 행 재계산(정규장 매수, 거래당 차이 = (연장 매도가 - RSI 매도가)/매수가)")
    for ru, K in keys:
        for h, Z in (("앞", reg[(reg.half == "앞") & (reg.bars <= K)]), ("뒤", reg[(reg.half == "뒤") & (reg.bars <= K)]),
                     ("전체", reg[reg.bars <= K]), ("7~9월", rege[rege.bars <= K])):
            d = Z[f"d_{ru}"]
            say(f"  {ru} K={K} {h}: n {len(Z)}({Z.day.nunique()}일) 합 {pct(d.sum(), 1)} 평균 {pct(d.mean())} t {crt(d, Z.day):+.2f} "
                f"승 {int((d > 0).sum())}/패 {int((d < 0).sum())}")
    with open(os.path.join(RES, "verify_fr_hold.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
