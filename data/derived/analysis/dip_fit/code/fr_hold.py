"""빠른 반등 뒤 보유 연장 연구(2026-10-10 사용자 가설: 빠르게 반등한 종목은 그 뒤로 더 오르지 않을까. CRCL 10/9, AMD 10/6).

질문: 이격매수 뒤 RSI 매도가 일찍 나온 거래(빠른 반등)는 그 매도가 뒤에도 더 오르는가? 그렇다면 RSI에서 팔지 말고 더 들고 가는 편이 나은가?

시험 전에 고정한 것(결과를 본 뒤 바꾸지 않음):
  기준 거래: 지금 지표(build.sim, 3분봉 EMA60[1] x (1-2%) 지정가 매수, RSI14>70 다음 봉 시가 또는 매수 뒤 첫 한국 08시 정리,
            전날 정규장 +7% 다음 날 정규장 매수 안 함). 매수일 ET 9/10~10/9, raw_1010. trades_1009.csv와 건수가 같은지 확인한다.
  빠른 반등 F(K): RSI 매도 봉(xi) - 매수 봉(ei) <= K, K in {10, 20, 30}(30·60·90분). 판정은 RSI 매도 순간(봉 xi 시가)까지의 봉으로만.
  느린 반등 S(K): RSI 매도인데 xi - ei > K.
  매도 순간 T = 봉 xi 시작 시각, 매도가 P = o[xi](지금 규칙 매도가).
  사후 경로(P 대비): +30분 +60분 +120분 = T+h 이후 첫 3분봉 시가. 매도가 정규장이면 정규장 종가에서 자르고(T+h가 16:00 넘으면 종가),
            모두 그 거래의 08시 정리 시각에서 자른다. 정규장 종가 = T 뒤 처음 끝나는 정규장의 마지막 3분봉 종가(08시 뒤면 08시 가격).
            08시 = 그 거래의 정리 시각 뒤 첫 3분봉 시가(지금 규칙의 08시 가격).
  보유 연장 규칙(F일 때만 RSI 매도 대신 적용, 나머지 거래는 지금 규칙 그대로). 모두 08시 정리를 마지막 안전장치로 둔다:
    a2 a3 a5  P x (1+X%) 위쪽 지정가 매도(1분봉 고가로 체결 판정, 정규장은 max(지정가, 그 분 시가), 프리·애프터는 지정가이며
              직전 1분 종가보다 1% 넘게 위인 지정가는 꼬리 체결로 보고 체결 안 함). 못 닿으면 08시.        자는 동안 가능
    b10 b25   3분봉 종가 < EMA10(또는 EMA25)로 마감하면 다음 봉 시가 매도(봉 xi 종가부터 판정).             자는 동안 불가
    c50       RSI14 < 50으로 마감하면 다음 봉 시가 매도(봉 xi부터).                                       자는 동안 불가
    d         정규장 종가(정규장 종가가 08시 뒤면 08시). 종가 시장가 주문이 필요하다.                        자는 동안 불가(05시 KST)
    e         08시까지 보유.                                                                           자는 동안 가능
  비교 값: 거래당 차이 = (연장 매도가 - RSI 매도가) / 매수가. 반별 합, 최악(연장 뒤 거래 수익 최소), 날짜 군집 t(매수일 묶음, CR1),
          상위 1종목·1일(차이 합이 가장 큰) 빼고 합·t.
  공정 기준(시간대 맞춤 무작위): 같은 종목·같은 날·같은 세션에서 T와 ±30분 안 모든 3분봉(그 봉 시가에 들어갔다 치고)에 같은 규칙·같은
          자르기를 적용한 수익의 평균. 사건의 연장 수익(P 대비) - 그 평균 = 공정 초과.
  시장 통제: 연장 수익 - 같은 구간(T → 연장 매도 시각) QQQ 수익(1분봉). 같은 날 사건 몰림도 적는다.
  범위: 주 결과 = 정규장 매수. 보조 = 프리·애프터 매수 중 매수 체결(직전 1분 종가보다 1% 넘게 아래)과 RSI 매도가(직전 1분 종가보다
          1% 넘게 위)가 모두 꼬리가 아닌 깨끗한 거래.
  고르기: 앞 절반(매수일 9/10~9/23) 정규장, 24조합(K 3 x 규칙 8). 차이 합 > 0인 조합 중 규칙마다 앞 t가 가장 큰 K 하나를 남기고,
          앞 t 상위 3개 규칙을 고른다. 확인: 뒤 절반(9/24~10/9).
  판정(미리 고정): 뒤에서 실패 = 뒤 차이 합 <= 0. 확인됨 = 뒤 차이 합 > 0, 뒤 t >= 2, 상위 1종목·1일 빼도 합 > 0, 뒤 공정 초과 평균 > 0,
          뒤 QQQ 통제 평균 > 0, 7/14~9/8 보조 차이 합 > 0을 모두 만족. 나머지 = 약함.
  보조 기간 7/14~9/8: stop_lab.load("early") 5분봉(EMA36·RSI9). K는 시간으로 맞춰 6·12·18봉(30·60·90분), EMA10·25는 같은 시간
          EMA6·15, RSI는 RSI9 < 50, 지정가 체결은 5분봉 고가(프리·애프터는 직전 5분 종가 +1% 넘는 지정가 제외), 공정 창 ±30분 = ±6봉.
          단위가 달라 보조 확인으로만 쓴다.
출력: ../results/fr_hold.txt, fr_hold_grid.csv, fr_hold_fs.csv(사후 추가 F-S 비교), fr_hold_events.csv
사용: py fr_hold.py"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
import build
from events import bars, prep
import stop_lab

SP = build.SP
RAW = os.path.join(SP, "raw_1010")
build.RAW = RAW
RES = os.path.join(HERE, "../results")
LO, MID, HI = date(2026, 9, 10), date(2026, 9, 24), date(2026, 10, 9)
KS = (10, 20, 30)
RULES = ["a2", "a3", "a5", "b10", "b25", "c50", "d", "e"]
SLEEP = {"a2": True, "a3": True, "a5": True, "b10": False, "b25": False, "c50": False, "d": False, "e": True}
RNAME = {"a2": "(a) +2% 지정가, 못 닿으면 08시", "a3": "(a) +3% 지정가, 못 닿으면 08시", "a5": "(a) +5% 지정가, 못 닿으면 08시",
         "b10": "(b) 종가 < EMA10이면 다음 봉 매도", "b25": "(b) 종가 < EMA25이면 다음 봉 매도", "c50": "(c) RSI < 50이면 다음 봉 매도",
         "d": "(d) 정규장 종가까지", "e": "(e) 08시까지"}
HZ = ["30분", "60분", "120분", "종가", "08시"]
NS_MIN = 60 * 10**9
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def pct(x, nd=2):
    return "   n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{nd}f}%"


# ---------------- 가격 묶음 ----------------

def pack(d, mins, dead, e_short, e_long, m1=None):
    """3분봉(또는 5분봉) 배열 묶음. 정규장 종가 색인(nrc)과 1분봉(있으면)을 붙인다."""
    tk = d.index.as_unit("ns").asi8
    reg = d.ses.values == "정규"
    n = len(tk)
    day = np.asarray(d.date)
    reg_idx = np.flatnonzero(reg)
    last_of = {}
    for j in reg_idx:
        last_of[day[j]] = j
    nxt = np.searchsorted(reg_idx, np.arange(n), "left")
    nrc = np.full(n, -1, np.int64)
    ok = nxt < len(reg_idx)
    nrc[ok] = [last_of[day[reg_idx[k]]] for k in nxt[ok]]
    c = d.c.values.astype(float)
    P = dict(tk=tk, o=d.o.values.astype(float), h=d.h.values.astype(float), l=d.l.values.astype(float), c=c,
             r=d.r.values.astype(float), reg=reg, mod=d["mod"].values.astype(np.int64), day=day, ses=d.ses.values,
             dead=np.asarray(dead, np.int64), nrc=nrc, step=mins * NS_MIN,
             e10=pd.Series(c).ewm(span=e_short, adjust=False).mean().values,
             e25=pd.Series(c).ewm(span=e_long, adjust=False).mean().values)
    if m1 is not None:
        m1 = m1[m1.High > 0]
        t1 = m1.index.as_unit("ns").asi8
        mod1 = m1.index.hour * 60 + m1.index.minute
        P.update(t1=t1, o1=m1.Open.values.astype(float), h1=m1.High.values.astype(float), c1=m1.Close.values.astype(float),
                 reg1=np.asarray((mod1 >= 570) & (mod1 < 960)))
    return P


def deadline(P, dead):
    """그 거래의 08시 정리: 정리 시각 뒤 첫 봉 시가. 데이터 끝이면 None."""
    j = np.searchsorted(P["tk"], dead, "left")
    if j >= len(P["tk"]):
        return None
    return P["o"][j], P["tk"][j], "o"


def reg_close(P, s0, dead):
    k = P["nrc"][s0]
    if k < 0:
        return deadline(P, dead)
    te = P["tk"][k] + P["step"]
    if te > dead:
        return deadline(P, dead)
    return P["c"][k], te, "c"


def horizon(P, s0, dead, h):
    """봉 s0 시가 진입 뒤 h 경로 가격. 반환 (가격, 시각 ns, 'o'|'c')."""
    if h == "08시":
        return deadline(P, dead)
    if h == "종가":
        return reg_close(P, s0, dead)
    tau = P["tk"][s0] + {"30분": 30, "60분": 60, "120분": 120}[h] * NS_MIN
    if P["reg"][s0]:
        k = P["nrc"][s0]
        if k >= 0 and P["day"][k] == P["day"][s0] and tau >= P["tk"][k] + P["step"]:
            return reg_close(P, s0, dead)
    if tau >= dead:
        return deadline(P, dead)
    j = np.searchsorted(P["tk"], tau, "left")
    if j >= len(P["tk"]):
        return None
    return P["o"][j], P["tk"][j], "o"


def rule_exit(P, s0, dead, rule):
    """봉 s0 시가에서 RSI 매도 대신 연장 규칙을 적용했을 때의 매도. 반환 (가격, 시각 ns, 'o'|'c', 매도 봉 색인)."""
    tk, o = P["tk"], P["o"]
    jd = np.searchsorted(tk, dead, "left")
    if jd >= len(tk):
        return None
    if jd <= s0:                                   # 이미 정리 시각을 지났다(RSI 매도와 같은 봉)
        return o[s0], tk[s0], "o", s0
    if rule == "e":
        return o[jd], tk[jd], "o", jd
    if rule == "d":
        x = reg_close(P, s0, dead)
        return x[0], x[1], x[2], (P["nrc"][s0] if x[2] == "c" else jd)
    if rule[0] == "a":
        L = o[s0] * (1 + int(rule[1:]) / 100)
        if "t1" in P:
            t1, h1, c1, o1, reg1 = P["t1"], P["h1"], P["c1"], P["o1"], P["reg1"]
            k0, k1 = np.searchsorted(t1, tk[s0], "left"), np.searchsorted(t1, dead, "left")
        else:                                      # 5분봉 보조: 봉 단위로 판정
            t1, h1, c1, o1, reg1 = tk, P["h"], P["c"], o, P["reg"]
            k0, k1 = s0, jd
        k0 = max(k0, 1)
        if k1 > k0:
            hh, rr, pc = h1[k0:k1], reg1[k0:k1], c1[k0 - 1:k1 - 1]
            hit = (hh >= L) & (rr | (L <= pc * 1.01))
            if hit.any():
                k = k0 + int(np.argmax(hit))
                px = max(L, o1[k]) if reg1[k] else L
                return px, t1[k], "o", np.searchsorted(tk, t1[k], "right") - 1
        return o[jd], tk[jd], "o", jd
    # b10 b25 c50: 봉 i-1(>= s0) 종가 조건이면 봉 i 시가
    seg = slice(s0, jd)
    if rule == "b10":
        cond = P["c"][seg] < P["e10"][seg]
    elif rule == "b25":
        cond = P["c"][seg] < P["e25"][seg]
    else:
        cond = P["r"][seg] < 50
    if cond.any():
        i = s0 + int(np.argmax(cond)) + 1
        return o[i], tk[i], "o", i
    return o[jd], tk[jd], "o", jd


def qret(Q, t0, t1, kind):
    """QQQ 1분봉(없으면 봉) 수익: t0 시가 → t1(시가 'o' 또는 그 전 분 종가 'c')."""
    ts, op, cl = Q["t"], Q["o"], Q["c"]
    a = np.searchsorted(ts, t0, "left")
    if a >= len(ts):
        return np.nan
    if kind == "c":
        b = np.searchsorted(ts, t1, "left") - 1
        p1 = cl[b] if b >= 0 else np.nan
    else:
        b = np.searchsorted(ts, t1, "left")
        p1 = op[b] if b < len(ts) else cl[-1]
    return p1 / op[a] - 1


def window(P, s0, win_bars):
    """공정 기준 창: 같은 날·같은 세션, s0 ± win_bars 봉 시각(분 기준 ±30분) 안 모든 봉."""
    m0, d0, s_ = P["mod"][s0], P["day"][s0], P["ses"][s0]
    lo, hi = max(0, s0 - win_bars - 2), min(len(P["tk"]), s0 + win_bars + 3)
    span = win_bars * P["step"] // NS_MIN
    return [j for j in range(lo, hi) if P["day"][j] == d0 and P["ses"][j] == s_ and abs(P["mod"][j] - m0) <= span]


# ---------------- 사건 만들기 ----------------

def tail_buy(P, t_ns, ent):
    """프리·애프터 매수 꼬리: 체결 분(매수 봉 안에서 저가가 매수가 이하가 된 첫 1분)의 직전 1분 종가보다 1% 넘게 아래."""
    t1 = P["t1"]
    w, e = np.searchsorted(t1, t_ns, "left"), np.searchsorted(t1, t_ns + 3 * NS_MIN, "left")
    for k in range(w, e):
        if P["l1"][k] <= ent * (1 + 1e-9):
            return np.nan if k == 0 else float(ent < P["c1"][k - 1] * 0.99)
    return np.nan


def tail_sell(P, t_ns, px):
    t1 = P["t1"]
    k = np.searchsorted(t1, t_ns, "left")
    if k == 0 or k >= len(t1):
        return np.nan
    return float(px > P["c1"][k - 1] * 1.01)


def events_for(P, Q, sym, trades, win_bars, period):
    """RSI 매도 거래마다 사후 경로·연장 규칙·공정 기준을 잰다."""
    rows = []
    for ei, xi, ent, px, why, dead in trades:
        if why != "RSI":
            continue
        t0 = P["tk"][xi]
        row = dict(sym=sym, period=period, day=P["day"][ei], t_buy=pd.Timestamp(P["tk"][ei], tz="UTC").tz_convert("America/New_York"),
                   t_sell=pd.Timestamp(t0, tz="UTC").tz_convert("America/New_York"), ses=P["ses"][ei], ses_x=P["ses"][xi],
                   bars=xi - ei, ent=ent, px=px, base=px / ent - 1)
        if "t1" in P and P["ses"][ei] != "정규":
            row["tail_b"] = tail_buy(P, P["tk"][ei], ent)
            row["tail_s"] = tail_sell(P, t0, px)
        else:
            row["tail_b"] = row["tail_s"] = 0.0
        W = window(P, xi, win_bars)
        for h in HZ:
            x = horizon(P, xi, dead, h)
            row[f"h_{h}"] = x[0] / px - 1 if x else np.nan
            row[f"hq_{h}"] = qret(Q, t0, x[1], x[2]) if x else np.nan
            fv = []
            for j in W:
                y = horizon(P, j, P["dead"][j], h)
                if y:
                    fv.append(y[0] / P["o"][j] - 1)
            row[f"hf_{h}"] = np.mean(fv) if fv else np.nan
        for ru in RULES:                                       # 느린 반등 S에도 재서 F와 비교한다
            x = rule_exit(P, xi, dead, ru)
            if x is None:
                row[f"x_{ru}"] = np.nan
                continue
            row[f"x_{ru}"] = x[0]                                   # 연장 매도가
            row[f"ext_{ru}"] = x[0] / px - 1                        # P 대비 연장 수익
            row[f"q_{ru}"] = qret(Q, t0, x[1], x[2])
            row[f"xi_{ru}"] = x[3]
            row[f"tx_{ru}"] = pd.Timestamp(x[1], tz="UTC").tz_convert("America/New_York")
            fv = []
            for j in W:
                y = rule_exit(P, j, P["dead"][j], ru)
                if y:
                    fv.append(y[0] / P["o"][j] - 1)
            row[f"f_{ru}"] = np.mean(fv) if fv else np.nan
        row["xi"], row["ei"] = xi, ei
        rows.append(row)
    return rows


def qpack(raw, mins, el, rl):
    if mins == 3:
        m = pd.read_parquet(os.path.join(raw, "1m_QQQ.parquet"))
        m = m[m.High > 0]
        return dict(t=m.index.as_unit("ns").asi8, o=m.Open.values.astype(float), c=m.Close.values.astype(float))
    d = prep(bars(raw, "QQQ", mins), el, rl)
    return dict(t=d.index.as_unit("ns").asi8, o=d.o.values.astype(float), c=d.c.values.astype(float))


def late_events():
    Q = qpack(RAW, 3, 60, 14)
    ref = pd.read_csv(os.path.join(RES, "trades_1009.csv"))
    rows, base_rows, n_all = [], [], 0
    for s in build.SYMS:
        try:
            d, dead_of, hot = build.arrays(s)
        except FileNotFoundError:
            continue
        m1 = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
        P = pack(d, 3, dead_of, 10, 25, m1)
        P["l1"] = m1[m1.High > 0].Low.values.astype(float)
        tr = []
        for ei, xi, ent, px, why in build.sim(d, dead_of, hot, 0.03 if s in build.LEV else 0.02):
            dy = P["day"][ei]
            if LO <= dy <= HI:
                tr.append((ei, xi, ent, px, why, dead_of[ei]))
                base_rows.append(dict(sym=s, day=dy, ei=ei, xi=xi, ses=P["ses"][ei], ret=px / ent - 1, why=why))
        n_all += len(tr)
        rows += events_for(P, Q, s, tr, 10, "late")
    B = pd.DataFrame(base_rows)
    return pd.DataFrame(rows), B, len(ref), n_all, ref


def early_events():
    S = stop_lab.load("early")
    Q = qpack(stop_lab.RAW, 5, 36, 9)
    b = stop_lab.run(S, None, "early")
    rows = []
    for s, A in S.items():
        g = b[b.sym == s]
        if not len(g):
            continue
        d = pd.DataFrame({"o": A["o"], "h": A["h"], "l": A["l"], "c": A["c"], "r": A["r"], "mod": A["mod"], "ses": A["ses"],
                          "date": A["day"]}, index=A["idx"])
        P = pack(d, 5, A["dead"], 6, 15, None)
        tr = [(int(r.ei), int(r.xi), r.ent, r.px, r.why, A["dead"][int(r.ei)]) for r in g.itertuples()]
        rows += events_for(P, Q, s, tr, 6, "early")
    E = pd.DataFrame(rows)
    E["bars5"] = E.bars
    E["bars"] = E.bars * 5 / 3                              # 3분봉 단위로 환산(6·12·18봉 = 10·20·30)
    B = b.rename(columns={"ret": "ret"})[["sym", "day", "ei", "xi", "ses", "ret", "why"]]
    return E, B


# ---------------- 통계 ----------------

def crt(x, g):
    """평균의 날짜 군집 t(CR1)."""
    x, g = np.asarray(x, float), np.asarray(g)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    G = len(set(g))
    if n < 2 or G < 3:
        return np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum().values
    se = np.sqrt((s ** 2).sum() * G / (G - 1)) / n
    return m / se if se > 0 else np.nan


def diff_stats(F, ru):
    """F 거래에서 규칙 ru로 바꿨을 때: 거래당 차이 = (연장 매도가 - RSI 매도가)/매수가."""
    if not len(F):
        return dict(n=0, days=0, 합=0.0, 평균=np.nan, t=np.nan, 최악=np.nan, 기준최악=np.nan, 합_1종목빼기=np.nan, t_1종목빼기=np.nan,
                    합_1일빼기=np.nan, t_1일빼기=np.nan, 공정초과=np.nan, 공정t=np.nan, QQQ통제=np.nan, QQQt=np.nan, 연장수익=np.nan)
    df = (F[f"x_{ru}"] - F.px) / F.ent
    new = F[f"x_{ru}"] / F.ent - 1
    out = dict(n=len(F), days=F.day.nunique(), 합=df.sum(), 평균=df.mean(), t=crt(df, F.day), 최악=new.min(), 기준최악=F.base.min())
    bs = df.groupby(F.sym).sum()
    bd = df.groupby(F.day).sum()
    ks, kd = F.sym != bs.idxmax(), F.day != bd.idxmax()
    out.update(합_1종목빼기=df[ks].sum(), t_1종목빼기=crt(df[ks], F.day[ks]), 합_1일빼기=df[kd].sum(), t_1일빼기=crt(df[kd], F.day[kd]))
    ex = F[f"ext_{ru}"] - F[f"f_{ru}"]
    qa = F[f"ext_{ru}"] - F[f"q_{ru}"]
    out.update(공정초과=ex.mean(), 공정t=crt(ex, F.day), QQQ통제=qa.mean(), QQQt=crt(qa, F.day), 연장수익=F[f"ext_{ru}"].mean())
    return out


def missed(F, B, ru):
    """연장하는 동안 같은 종목에서 지금 규칙이 새로 샀을 거래(연장하면 신호가 막힌다). 건수, 그 수익 합."""
    n, s, sr = 0, 0.0, 0.0
    for r in F.itertuples():
        x_end = getattr(r, f"xi_{ru}")
        if not np.isfinite(x_end):
            continue
        g = B[(B.sym == r.sym) & (B.ei > r.xi) & (B.ei < x_end)]
        n += len(g)
        s += g.ret.sum()
        sr += g[g.ses == "정규"].ret.sum()
    return n, s, sr


def half_of(dy):
    return "앞" if dy < MID else "뒤"


# ---------------- 실행 ----------------

def main():
    os.makedirs(RES, exist_ok=True)
    say("== 빠른 반등 뒤 보유 연장(fr_hold) ==")
    say(__doc__.split("시험 전에 고정한 것(결과를 본 뒤 바꾸지 않음):")[1].split("출력:")[0].rstrip())
    say(f"시험 조합: 고르기 24개(앞 절반 정규장, K 3 x 규칙 8). 사후 경로 기술 15개(K 3 x 구간 5, 고르기 아님). "
        f"보조로 같은 24조합을 뒤 절반·연장 깨끗·7~9월 5분봉에 보고(고르기에 안 씀).")

    E, B, n_ref, n_all, ref = late_events()
    say(f"\n[재현] 기준 거래 {n_all}건(trades_1009.csv {n_ref}건) | RSI 매도 사건 {len(E)}건")
    E["half"] = E.day.map(half_of)
    reg = E[E.ses == "정규"]
    ext = E[(E.ses != "정규") & (E.tail_b == 0) & (E.tail_s == 0)]
    say(f"정규장 매수 RSI 매도 {len(reg)}건 | 프리·애프터 RSI 매도 {int((E.ses != '정규').sum())}건 중 깨끗 {len(ext)}건")
    Ee, Be = early_events()
    rege = Ee[Ee.ses == "정규"]

    # 1) 사후 경로
    say("\n== 1. RSI 매도 뒤 경로(매도가 P 대비 평균) — F(빠름) vs S(느림), 공정 초과 = 사건 - 같은 종목·날·±30분 무작위 진입 ==")
    hdr = "범위 | 묶음 | n(일) | " + " | ".join(f"{h} 원/공정초과(t)/QQQ통제" for h in HZ)
    say(hdr)
    for scope, X in (("정규 앞", reg[reg.half == "앞"]), ("정규 뒤", reg[reg.half == "뒤"]), ("정규 전체", reg),
                     ("연장깨끗 전체", ext), ("7~9월 5분봉 정규", rege)):
        for K in KS:
            for lab, G in ((f"F({K})", X[X.bars <= K]), (f"S({K})", X[X.bars > K])):
                cells = []
                for h in HZ:
                    raw_ = G[f"h_{h}"]
                    ex = G[f"h_{h}"] - G[f"hf_{h}"]
                    qa = G[f"h_{h}"] - G[f"hq_{h}"]
                    cells.append(f"{pct(raw_.mean())}/{pct(ex.mean())}({crt(ex, G.day):+.1f})/{pct(qa.mean())}")
                say(f"{scope} | {lab} | {len(G)}({G.day.nunique()}) | " + " | ".join(cells))
    say("  같은 날 F(20) 몰림: " + ", ".join(f"{d}:{n}" for d, n in reg[reg.bars <= 20].groupby("day").size().sort_values(ascending=False).head(5).items()))
    say("  F(20) 날 QQQ 그날 정규장 수익과 무관하게 보려면 위 QQQ통제 열을 본다.")

    # 2) 보유 연장 격자
    say("\n== 2. 보유 연장 격자(F일 때만 적용) — 차이 = (연장 매도가 - RSI 매도가)/매수가 ==")
    grid = []
    for scope, X in (("앞", reg[reg.half == "앞"]), ("뒤", reg[reg.half == "뒤"]), ("전체", reg), ("연장깨끗", ext), ("7~9월", rege)):
        for K in KS:
            F = X[X.bars <= K]
            for ru in RULES:
                st = diff_stats(F, ru)
                grid.append(dict(범위=scope, K=K, 규칙=ru, 자는동안=SLEEP[ru], **st))
    G = pd.DataFrame(grid)
    G.to_csv(os.path.join(RES, "fr_hold_grid.csv"), index=False, encoding="utf-8-sig")
    show = G[G.범위.isin(["앞", "뒤", "7~9월"])]
    say("범위 K 규칙 | n(일) | 차이합 | 평균 | t | 연장후 최악(기준 최악) | 공정초과(t) | QQQ통제(t)")
    for r in show.itertuples():
        say(f"{r.범위} {r.K} {r.규칙:>3} | {r.n}({r.days}) | {pct(r.합, 1)} | {pct(r.평균)} | {r.t:+.2f} | {pct(r.최악, 1)}({pct(r.기준최악, 1)}) | "
            f"{pct(r.공정초과)}({r.공정t:+.2f}) | {pct(r.QQQ통제)}({r.QQQt:+.2f})")

    # 2b) 같은 연장을 느린 반등 S에 했을 때와 비교(사후 추가 통제)
    say("\n== 2b. [사후 추가 통제] 같은 연장 규칙을 느린 반등 S(K)에도 적용: 거래당 차이 평균 F vs S ==")
    say("  이유: 미리 정한 공정 창(±30분)은 매도 전 오르던 봉을 포함해, RSI 매도 사건이면 F·S 모두 음수로 치우친다(1번 표 S 공정초과 -0.3~-0.4%, t -5~-14).")
    say("  그래서 '빠른 반등만의 효과'는 같은 RSI 매도 사건 중 느린 쪽과의 차이로 본다. 고르기에는 안 쓴다.")
    say("범위 K 규칙 | F 평균(n) | S 평균(n) | F-S (t)")
    fs = []
    for scope, X in (("앞", reg[reg.half == "앞"]), ("뒤", reg[reg.half == "뒤"]), ("7~9월", rege)):
        for K in KS:
            F, S_ = X[X.bars <= K], X[X.bars > K]
            for ru in RULES:
                dF, dS = (F[f"x_{ru}"] - F.px) / F.ent, (S_[f"x_{ru}"] - S_.px) / S_.ent
                tF, tS = crt(dF, F.day), crt(dS, S_.day)
                seF = abs(dF.mean() / tF) if np.isfinite(tF) and tF != 0 else np.nan
                seS = abs(dS.mean() / tS) if np.isfinite(tS) and tS != 0 else np.nan
                gap = dF.mean() - dS.mean()
                tg = gap / np.sqrt(seF ** 2 + seS ** 2) if np.isfinite(seF) and np.isfinite(seS) else np.nan
                fs.append(dict(범위=scope, K=K, 규칙=ru, F=dF.mean(), nF=len(F), S=dS.mean(), nS=len(S_), gap=gap, t=tg))
                say(f"{scope} {K} {ru:>3} | {pct(dF.mean())}({len(F)}) | {pct(dS.mean())}({len(S_)}) | {pct(gap)} ({tg:+.2f})")
    FS = pd.DataFrame(fs)
    FS.to_csv(os.path.join(RES, "fr_hold_fs.csv"), index=False, encoding="utf-8-sig")

    # 3) 고르기 → 확인
    say("\n== 3. 앞 절반에서 고르기 → 뒤 절반 확인 ==")
    fr = G[(G.범위 == "앞") & (G.합 > 0)].copy()
    fr = fr.sort_values("t", ascending=False).drop_duplicates("규칙").head(3)
    say(f"앞 절반 차이 합 > 0 조합 {int(((G.범위 == '앞') & (G.합 > 0)).sum())}/24개. 고른 규칙: " +
        (", ".join(f"{r.규칙}(K={r.K}, 앞 합 {pct(r.합, 1)}, t {r.t:+.2f})" for r in fr.itertuples()) or "없음"))
    ref_only = not len(fr)
    if ref_only:
        fr = G[G.범위 == "앞"].sort_values("t", ascending=False).drop_duplicates("규칙").head(3)
        say("  미리 정한 선택 조건(앞 차이 합 > 0)을 통과한 규칙이 없다. 아래 3개는 앞 t가 가장 덜 나쁜 규칙을 참고로만 뒤에 대 본 것이다(선택 아님).")
    picks = []
    for r in fr.itertuples():
        K, ru = r.K, r.규칙
        b = G[(G.범위 == "뒤") & (G.K == K) & (G.규칙 == ru)].iloc[0]
        e = G[(G.범위 == "7~9월") & (G.K == K) & (G.규칙 == ru)].iloc[0]
        x = G[(G.범위 == "연장깨끗") & (G.K == K) & (G.규칙 == ru)].iloc[0]
        Fb = reg[(reg.half == "뒤") & (reg.bars <= K)]
        mn, ms, msr = missed(Fb, B, ru)
        a = G[(G.범위 == "전체") & (G.K == K) & (G.규칙 == ru)].iloc[0]
        if b.합 <= 0:
            vd = "뒤에서 실패"
        elif (b.t >= 2 and b.합_1종목빼기 > 0 and b.합_1일빼기 > 0 and b.공정초과 > 0 and b.QQQ통제 > 0 and e.합 > 0):
            vd = "확인됨"
        else:
            vd = "약함"
        say(f"[{ru} K={K}] {RNAME[ru]} | 자는 동안 {'가능' if SLEEP[ru] else '불가'}")
        say(f"  앞: n {r.n}({r.days}일) 차이합 {pct(r.합, 1)} t {r.t:+.2f} | 공정초과 {pct(r.공정초과)} QQQ통제 {pct(r.QQQ통제)}")
        say(f"  뒤: n {b.n}({b.days}일) 차이합 {pct(b.합, 1)} t {b.t:+.2f} | 1종목 빼면 {pct(b.합_1종목빼기, 1)}(t {b.t_1종목빼기:+.2f}) "
            f"1일 빼면 {pct(b.합_1일빼기, 1)}(t {b.t_1일빼기:+.2f}) | 공정초과 {pct(b.공정초과)}(t {b.공정t:+.2f}) QQQ통제 {pct(b.QQQ통제)}(t {b.QQQt:+.2f})"
            f" | 연장 뒤 최악 {pct(b.최악, 1)} (기준 {pct(b.기준최악, 1)}) | 연장 중 막힌 재매수 {mn}건 합 {pct(ms, 1)}(그중 정규장 {pct(msr, 1)})")
        Ff = reg[(reg.half == "앞") & (reg.bars <= K)]
        dd = ((Ff[f"x_{ru}"] - Ff.px) / Ff.ent).groupby(Ff.day).sum()
        say(f"  전체 9/10~10/9: n {a.n}({a.days}일) 차이합 {pct(a.합, 1)} t {a.t:+.2f} | 참고: 앞에서 손해가 가장 큰 날({dd.idxmin()}, "
            f"{pct(dd.min(), 1)}) 빼면 앞 합 {pct(dd.sum() - dd.min(), 1)}")
        say(f"  7~9월 5분봉: n {e.n}({e.days}일) 차이합 {pct(e.합, 1)} t {e.t:+.2f} 공정초과 {pct(e.공정초과)} | 연장깨끗(9/10~10/9): n {x.n} 차이합 {pct(x.합, 1)}")
        if ref_only:
            vd = vd + "(참고 행: 앞에서 이미 음수)"
        say(f"  판정: {vd}")
        picks.append(dict(rule=ru, K=K, front=r, back=b, early=e, ext=x, verdict=vd, missed=(mn, ms)))

    # 4) 사례
    say("\n== 4. 사례: CRCL 10/9, AMD 10/6 ==")
    for sym, dy in (("CRCL", date(2026, 10, 9)), ("AMD", date(2026, 10, 6))):
        z = reg[(reg.sym == sym) & (reg.day == dy)]
        for r in z.itertuples():
            say(f"{sym} {dy}: 매수 {r.t_buy:%H:%M} {r.ent:.2f} → RSI 매도 {r.t_sell:%H:%M} {r.px:.2f} ({pct(r.base)}, {r.bars}봉) | F10 {r.bars <= 10} F20 {r.bars <= 20} F30 {r.bars <= 30}")
            say("  사후(P 대비): " + " ".join(f"{h} {pct(getattr(r, 'h_' + h))}(공정 {pct(getattr(r, 'hf_' + h))})" for h in HZ))
            for ru in RULES:
                xv = getattr(r, f"x_{ru}")
                say(f"  {ru}: 매도 {getattr(r, f'tx_{ru}'):%m-%d %H:%M} {xv:.2f} → 거래 수익 {pct(xv / r.ent - 1)} (RSI 대비 {pct((xv - r.px) / r.ent)})")

    E.to_csv(os.path.join(RES, "fr_hold_events.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RES, "fr_hold.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    return E, Ee, G, picks


if __name__ == "__main__":
    main()
