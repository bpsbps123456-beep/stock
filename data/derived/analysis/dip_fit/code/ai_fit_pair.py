"""이격매수매도 AI 종목 궁합 4부 — NBIS vs IREN 짝 비교 갈래 (2026-10-23 사용자: 'IREN보다 NBIS가 더 잘 맞는 것 같다').

입력: ../results/ai_fit_trades.csv (ai_fit_build.py가 만든 거래표, 고치지 않는다), raw 폴더(1m·5m·1d parquet, ET 시각).
출력: ../results/ai_fit_pair.txt (본문), ../results/ai_fit_pair_days.csv (한 종목만 산 날, 다른 종목이 안 산 이유)
주 잣대는 정규장만(꼬리 체결 없음), 보조로 전체 세션과 '꼬리 뺀 전체'를 본다.
t는 날짜 군집으로 잰다(같은 날 거래는 독립이 아님). 차이 구간은 날짜 부트스트랩(1만 번)이다.
보유 중 비트코인 수익은 매수 뒤에야 아는 값이라 '노출' 설명용이다. 사전 값은 기간 시작 전 60거래일 일봉 상관과
앞 구간 5분봉 상관뿐이다.
사용: py ai_fit_pair.py <raw 폴더>
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
from events import bars, prep
from rules import hot_days

RES = os.path.join(HERE, "../results")
ET = "America/New_York"
A, B = "NBIS", "IREN"
NEO = ["CRWV", "NBIS"]                                   # watchlist 네오클라우드
MIN = ["APLD", "CIFR", "GLXY", "HUT", "IREN", "WULF"]    # watchlist 채굴→AI 전환
PER = {"P0": (date(2026, 8, 3), date(2026, 9, 9)), "P1": (date(2026, 9, 10), date(2026, 10, 8)),
       "P2": (date(2026, 10, 9), date(2026, 10, 22))}
NBOOT = 10000
rng = np.random.default_rng(0)
L = []                                                    # 출력 줄
NTEST = {}                                                # 갈래 안 시험 수 집계
KEY = {}                                                  # 요약에 쓸 숫자


def P(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s)


def pct(x, d=2):
    return "  -  " if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


# ── 통계 도구 ─────────────────────────────────────────
def ct(r, g):
    """평균의 날짜 군집 t."""
    r, g = np.asarray(r, float), np.asarray(g)
    n = len(r)
    if n < 2:
        return np.nan
    e = r - r.mean()
    s = pd.Series(e).groupby(g).sum().values
    G = len(s)
    if G < 2:
        return np.nan
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    return r.mean() / se if se > 0 else np.nan


def diff_ct(ra, ga, rb, gb):
    """두 묶음 평균 차이(a-b)의 날짜 군집 t. 같은 날 거래는 한 군집."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    if len(ra) < 2 or len(rb) < 2:
        return np.nan
    ia = (ra - ra.mean()) / len(ra)
    ib = -(rb - rb.mean()) / len(rb)
    s = pd.Series(np.r_[ia, ib]).groupby(np.r_[np.asarray(ga), np.asarray(gb)]).sum().values
    G = len(s)
    se = np.sqrt(G / (G - 1) * (s ** 2).sum())
    return (ra.mean() - rb.mean()) / se if se > 0 else np.nan


def boot_diff(ra, ga, rb, gb):
    """날짜 부트스트랩: 날짜를 복원 추출해 그날 거래를 모두 넣는다. 반환 (2.5%, 97.5%, 차이<=0 비율)."""
    days = np.array(sorted(set(ga) | set(gb)))
    ix = {d: i for i, d in enumerate(days)}
    G = len(days)
    sa, na_, sb, nb = (np.zeros(G) for _ in range(4))
    for r, g in zip(ra, ga):
        sa[ix[g]] += r; na_[ix[g]] += 1
    for r, g in zip(rb, gb):
        sb[ix[g]] += r; nb[ix[g]] += 1
    W = rng.multinomial(G, np.full(G, 1 / G), size=NBOOT).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = W @ sa / (W @ na_) - W @ sb / (W @ nb)
    d = d[np.isfinite(d)]
    return np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()


def boot_mean(x):
    """날짜별 값 하나씩(짝 차이 등)의 평균 부트스트랩 구간."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = x[rng.integers(0, len(x), size=(NBOOT, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ── 거래표 ────────────────────────────────────────────
def fill_exit(T, raw):
    """P1(지난 표)에는 매도 시각이 없다 → 봉 격자에서 되살린다.
    9/25 04시 이후 매수: 새 1분봉으로 만든 3분봉에서 매수 봉 + bars 번째 봉(정확).
    그 전 매수: 8시 정리는 매수 뒤 첫 한국 8시 이후 첫 5분봉, RSI 매도는 5분봉 격자에서 bars×3분만큼 간 봉(근사)."""
    miss = T.t_exit.isna()
    T["exit_src"] = np.where(miss, "", "표")
    cut = pd.Timestamp("2026-09-25 04:00", tz=ET)
    for s, g in T[miss].groupby("sym"):
        i3 = bars(raw, s, 3).index
        i5 = pd.read_parquet(f"{raw}/5m_{s}.parquet").index
        for k, r in g.iterrows():
            if r.t >= cut:
                p = i3.searchsorted(r.t)
                if p < len(i3) and i3[p] == r.t and p + r.bars < len(i3):
                    T.at[k, "t_exit"], T.at[k, "exit_src"] = i3[p + int(r.bars)], "3분봉"
                    continue
            if r.why == "8시":
                kst = r.t.tz_convert("Asia/Seoul")
                t8 = kst.normalize() + pd.Timedelta(hours=8)
                dead = (t8 if kst < t8 else t8 + pd.Timedelta(days=1)).tz_convert(ET)
                p = i5.searchsorted(dead)
            else:
                p = i5.searchsorted(r.t) + int(round(r.bars * 3 / 5))
            if p < len(i5):
                T.at[k, "t_exit"], T.at[k, "exit_src"] = i5[p], "5분봉 근사"
    return T


def load(raw):
    T = pd.read_csv(os.path.join(RES, "ai_fit_trades.csv"))
    T["t"] = pd.to_datetime(T.t, utc=True).dt.tz_convert(ET)
    T["t_exit"] = pd.to_datetime(T.t_exit, utc=True).dt.tz_convert(ET)
    T["day"] = pd.to_datetime(T.day).dt.date
    T = fill_exit(T, raw)
    # 같은 날·같은 세션(정규/프리/애프터) 유니버스 평균(자기 거래 빼고) → 초과 수익
    g = T.groupby(["day", "ses"]).ret
    s, n = g.transform("sum"), g.transform("size")
    T["ex"] = np.where(n > 1, T.ret - (s - T.ret) / (n - 1).clip(lower=1), np.nan)
    return T


SESF = {"정규": lambda x: x[x.ses == "정규"], "프리": lambda x: x[x.ses == "프리"], "애프터": lambda x: x[x.ses == "애프터"],
        "전체": lambda x: x, "꼬리뺀전체": lambda x: x[x["tail"] != 1]}


def pick(T, per):
    if per == "P0+P1":
        return T[T.period.isin(["P0", "P1"])]
    if per == "전체":
        return T
    return T[T.period == per]


def desc(x):
    r = x.ret.values
    if len(r) == 0:
        return None
    rs, h8 = x[x.why == "RSI"].ret, x[x.why == "8시"].ret
    return dict(n=len(r), mean=r.mean(), t=ct(r, x.day.values), win=(r > 0).mean(), worst=r.min(), best=r.max(), sum=r.sum(),
                nrsi=len(rs), mrsi=rs.mean() if len(rs) else np.nan, m8=h8.mean() if len(h8) else np.nan,
                hold=np.median(x.hold_min), mae=x.mae.mean(), maemin=x.mae.min(),
                tail1=int((x["tail"] == 1).sum()), tailk=int(x["tail"].notna().sum()), days=x.day.nunique())


def sec1(T):
    P("\n== 1. 기간별·세션별 분해 ==")
    P("열: 건수 | 평균(날짜군집 t) | 승률 | 최악 | 최고 | 합 | RSI매도 건수·평균 | 8시정리 평균 | 보유 중앙(분) | MAE 평균·최악 | 꼬리(1/판정가능)")
    for per in ["P0", "P1", "P2", "P0+P1", "전체"]:
        P(f"\n[{per}]")
        for sf in ["정규", "프리", "애프터", "전체", "꼬리뺀전체"]:
            for s in [A, B, "AI전체"]:
                x = SESF[sf](pick(T, per))
                x = x if s == "AI전체" else x[x.sym == s]
                d = desc(x)
                if d is None:
                    P(f"  {sf:6s} {s:6s} 0건")
                    continue
                if s == "AI전체":
                    P(f"  {sf:6s} {s:6s} {d['n']:4d}건 평균 {pct(d['mean'])} (t {d['t']:+.2f}) 승률 {d['win']:.0%} | RSI {d['nrsi']}건 {pct(d['mrsi'])} 8시 {pct(d['m8'])} | 보유 {d['hold']:.0f}분")
                    continue
                P(f"  {sf:6s} {s:6s} {d['n']:3d}건 평균 {pct(d['mean'])} (t {d['t']:+.2f}) 승률 {d['win']:.0%} 최악 {pct(d['worst'])} 최고 {pct(d['best'])} 합 {pct(d['sum'], 1)}"
                  f" | RSI {d['nrsi']}건 {pct(d['mrsi'])} 8시 {pct(d['m8'])} | 보유 {d['hold']:.0f}분 | MAE {pct(d['mae'])} 최악 {pct(d['maemin'])} | 꼬리 {d['tail1']}/{d['tailk']}")
    # 유니버스 안 순위(정규장, 5건 이상)
    P("\n유니버스 안 순위(종목 평균, 5건 이상 종목만):")
    for per in ["P0", "P1", "P0+P1", "P2"]:
        for sf in ["정규", "전체"]:
            x = SESF[sf](pick(T, per))
            m = x.groupby("sym").ret.agg(["size", "mean"])
            m = m[m["size"] >= 5].sort_values("mean", ascending=False)
            rk = {s: (list(m.index).index(s) + 1 if s in m.index else None) for s in [A, B]}
            P(f"  {per:5s} {sf:3s}: 대상 {len(m)}종목, NBIS {rk[A]}위, IREN {rk[B]}위")


def sec1b(T):
    P("\n== 1b. NBIS − IREN 차이 (거래 평균 차이, 날짜 군집 t, 날짜 부트스트랩 95% 구간) ==")
    P("  '초과'는 같은 날·같은 세션(정규·프리·애프터) AI 유니버스 평균(자기 거래 뺀)을 뺀 값이다. 그날 장 전체 효과를 지운다.")
    n = 0
    for per in ["P0", "P1", "P2", "P0+P1", "전체"]:
        for sf in ["정규", "전체", "꼬리뺀전체"]:
            x = SESF[sf](pick(T, per))
            a, b = x[x.sym == A], x[x.sym == B]
            for col, nm in [("ret", "원수익"), ("ex", "초과")]:
                aa, bb = a[a[col].notna()], b[b[col].notna()]
                if len(aa) < 2 or len(bb) < 2:
                    continue
                d = aa[col].mean() - bb[col].mean()
                t = diff_ct(aa[col], aa.day, bb[col], bb.day)
                lo, hi, p0 = boot_diff(aa[col].values, aa.day.values, bb[col].values, bb.day.values)
                n += 1
                KEY[("1b", per, sf, col)] = (d, t, lo, hi, len(aa), len(bb))
                P(f"  {per:5s} {sf:5s} {nm}: NBIS {len(aa)}건 {pct(aa[col].mean())} / IREN {len(bb)}건 {pct(bb[col].mean())} → 차이 {pct(d)} (t {t:+.2f}) 구간 [{pct(lo)}, {pct(hi)}] 차이≤0 비율 {p0:.2f}")
    NTEST["1b NBIS-IREN 차이(기간5×세션3×원/초과2, 겹침 포함 서술용)"] = n


# ── 3·5분봉 상태 기계(한 종목만 산 날의 이유) ──────────────
def dead_times(d):
    kst = d.index.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    return np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)


def sim_state(d, hot, gap=0.02, x=70.0):
    """ai_fit_build.sim과 같은 상태 기계 + 봉마다 '시작할 때 보유 중' 표시."""
    o, l, r, ema, tk = d.o.values, d.l.values, d.r.values, d.ema.values, d.index.as_unit("ns").asi8
    dead_of = dead_times(d)
    hs = np.zeros(len(d), bool)
    out, hold, ent, ei, dead = [], False, 0.0, 0, 0
    for i in range(61, len(d)):
        hs[i] = hold
        if hold:
            why = "RSI" if (i - 1 >= ei and r[i - 1] > x) else ("8시" if tk[i] >= dead else None)
            if why:
                out.append((ei, i, ent, o[i], why))
                hold = False
        elif not hot[i]:
            lvl = ema[i - 1] * (1 - gap)
            if l[i] <= lvl:
                ent, ei, hold, dead = min(o[i], lvl), i, True, dead_of[i]
    return out, hs


def state_frames(raw, s):
    res = {}
    for mins, el, rl in [(5, 36, 9), (3, 60, 14)]:
        d = prep(bars(raw, s, mins), el, rl)
        hot = hot_days(raw, s, d, 0.07)
        _, hs = sim_state(d, hot)
        line = np.r_[np.nan, d.ema.values[:-1]] * 0.98
        d = d.assign(hot=hot, hs=hs, line=line, gapln=d.l / line - 1)
        d.loc[d.index[:61], "gapln"] = np.nan
        res[mins] = d
    return res


def why_not(fr, dy):
    """그날(ET) 그 종목이 안 산 이유. 9/28 이후는 3분봉, 그 전은 5분봉 근사(EMA36·RSI9)."""
    src = 3 if dy >= date(2026, 9, 28) else 5
    d = fr[src]
    x = d[d.date == dy]
    if len(x) == 0:
        return dict(src=src, why="봉 없음", mingap=np.nan, minses="")
    touch = x.gapln <= 0
    if (touch & ~x.hot & ~x.hs).any():
        why = "닿음(근사 불일치)" if src == 5 else "닿음(?)"
    elif (touch & x.hot).any() and not (touch & x.hs & ~x.hot).any():
        why = "전날+7% 쉼"
    elif (touch & x.hs).any():
        why = "이미 보유 중"
    else:
        why = "선에 안 닿음"
    j = x.gapln.idxmin()
    return dict(src=src, why=why, mingap=x.gapln.min(), minses=x.ses.loc[j])


def sec2(T, raw):
    P("\n== 2. 같은 날 짝 비교 ==")
    fr = {s: state_frames(raw, s) for s in [A, B]}
    rows = []
    for per in ["P0", "P1", "P2"]:
        x = pick(T, per)
        days = sorted(set(pd.read_parquet(f"{raw}/1d_QQQ.parquet").index.date) & set(
            pd.date_range(*PER[per]).date))
        for sf in ["전체", "정규", "프리"]:
            y = SESF[sf](x)
            da = y[y.sym == A].groupby("day").ret.mean()
            db = y[y.sym == B].groupby("day").ret.mean()
            both = sorted(set(da.index) & set(db.index))
            oa, ob = sorted(set(da.index) - set(db.index)), sorted(set(db.index) - set(da.index))
            neither = len(days) - len(both) - len(oa) - len(ob)
            dd = (da[both] - db[both]).values
            lo, hi = boot_mean(dd)
            cor = np.corrcoef(da[both], db[both])[0, 1] if len(both) > 2 else np.nan
            P(f"  {per} {sf}: 거래일 {len(days)} | 둘 다 {len(both)}일, NBIS만 {len(oa)}일, IREN만 {len(ob)}일, 둘 다 없음 {neither}일")
            if len(both):
                P(f"      둘 다 산 날 하루 평균 차이(NBIS−IREN) {pct(dd.mean())} 구간 [{pct(lo)}, {pct(hi)}], NBIS가 나은 날 {(dd > 0).sum()}/{len(dd)}, 두 종목 그날 수익 상관 {cor:+.2f}")
            if sf == "전체":
                for s, only, other in [(A, oa, B), (B, ob, A)]:
                    for dy in only:
                        w = why_not(fr[other], dy)
                        rows.append(dict(period=per, day=dy, bought=s, other=other, ret=(da if s == A else db)[dy],
                                         src=f"{w['src']}m", why=w["why"], other_min_gap=w["mingap"], gap_ses=w["minses"]))
    P("\n  전 기간(P0~P2) 같은 날 짝:")
    for sf in ["전체", "정규", "프리"]:
        y = SESF[sf](T)
        da = y[y.sym == A].groupby("day").ret.mean()
        db = y[y.sym == B].groupby("day").ret.mean()
        both = sorted(set(da.index) & set(db.index))
        dd = (da[both] - db[both]).values
        lo, hi = boot_mean(dd)
        KEY[("2", sf)] = (dd.mean(), lo, hi, len(dd), int((dd > 0).sum()))
        P(f"   {sf}: 둘 다 산 날 {len(both)}일, 하루 평균 차이(NBIS−IREN) {pct(dd.mean())} 구간 [{pct(lo)}, {pct(hi)}], NBIS가 나은 날 {(dd > 0).sum()}/{len(dd)}")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, "ai_fit_pair_days.csv"), index=False, encoding="utf-8-sig")
    P("\n  한 종목만 산 날, 다른 종목이 안 산 이유(전체 세션 기준, 9/28 이전은 5분봉 근사):")
    for (per, s), g in D.groupby(["period", "bought"]):
        c = g.why.value_counts().to_dict()
        ng = g[g.why == "선에 안 닿음"].other_min_gap
        P(f"   {per} {s}만 산 {len(g)}일: 그날 {s} 평균 {pct(g.ret.mean())} | 다른 종목 이유 {c} | 안 닿은 날 선까지 남은 거리 중앙 {pct(ng.median())}")
    # 신호 빈도와 사전 변동성
    P("\n  신호 빈도(거래일당 매수 건수)와 사전 변동성:")
    for per in ["P0", "P1", "P2"]:
        lo_, hi_ = PER[per]
        nd = len(set(pd.read_parquet(f"{raw}/1d_QQQ.parquet").index.date) & set(pd.date_range(lo_, hi_).date))
        for s in [A, B]:
            x = pick(T, per)
            x = x[x.sym == s]
            c = pd.read_parquet(f"{raw}/1d_{s}.parquet").Close
            c.index = pd.to_datetime(c.index).date
            r = c.pct_change()
            pre = r[[d < lo_ for d in r.index]].tail(20)
            fr5 = fr[s][5]
            # 앞 기간 5분봉 수익 표준편차(정규장) — 기간 시작 전 20거래일
            q = fr5[(fr5.date < lo_) & (fr5.ses == "정규")]
            qd = sorted(set(q.date))[-20:]
            q = q[q.date.isin(qd)]
            v5 = np.log(q.c).diff()[q.date == q.date.shift(1)].std() if len(q) else np.nan
            P(f"   {per} {s}: 거래일당 매수 {len(x) / nd:.2f}건 (정규 {(x.ses == '정규').sum() / nd:.2f}, 프리 {(x.ses == '프리').sum() / nd:.2f}, 애프터 {(x.ses == '애프터').sum() / nd:.2f})"
              f" | 시작 전 20일 일간 변동 {pre.std() * 100:.2f}% | 5분봉 변동 {v5 * 100 if np.isfinite(v5) else np.nan:.3f}%({len(qd)}일)")
    return D


def sec3(T):
    P("\n== 3. 차이 크기 대 잡음 ==")
    for sf in ["정규", "전체"]:
        x = SESF[sf](T)
        for per in ["P0+P1", "전체"]:
            y = pick(x, per)
            a, b = y[y.sym == A].ret, y[y.sym == B].ret
            d = a.mean() - b.mean()
            sa, sb = a.std(), b.std()
            n80 = 7.84 * (sa ** 2 + sb ** 2) / d ** 2
            KEY[("3", sf, per)] = (d, sa, sb, 4 * (sa ** 2 + sb ** 2) / d ** 2, n80)
            nt2 = 4 * (sa ** 2 + sb ** 2) / d ** 2
            nd = y.day.nunique()
            rate = (len(a) + len(b)) / 2 / nd * 21
            P(f"  {sf} {per}: NBIS {len(a)}건 sd {sa * 100:.2f}% / IREN {len(b)}건 sd {sb * 100:.2f}% / 차이 {pct(d)}"
              f" → t=2에 종목당 {nt2:.0f}건, 검정력 80%에 종목당 {n80:.0f}건 (지금 속도 종목당 월 {rate:.1f}건 → {n80 / rate:.0f}개월)")
    # 같은 날 짝 차이로 줄어드는 잡음
    for sf in ["전체", "정규"]:
        y = SESF[sf](T)
        da = y[y.sym == A].groupby("day").ret.mean()
        db = y[y.sym == B].groupby("day").ret.mean()
        both = sorted(set(da.index) & set(db.index))
        dd = da[both] - db[both]
        npair = 7.84 * dd.std() ** 2 / dd.mean() ** 2 if dd.mean() != 0 else np.inf
        P(f"  같은 날 짝({sf}, 전 기간 {len(both)}일): 차이 평균 {pct(dd.mean())} sd {dd.std() * 100:.2f}% → 검정력 80%에 짝 {npair:.0f}일")
    # 종목 간 '진짜' 차이 크기(무작위 효과): 관측 평균의 분산 − 표본 잡음
    P("\n  AI 유니버스 종목 간 진짜 차이 크기(무작위 효과 추정, 5건 이상 종목):")
    for sf in ["정규", "전체", "꼬리뺀전체"]:
        for per in ["P0", "P1", "P0+P1", "P2"]:
            y = SESF[sf](pick(T, per))
            g = y.groupby("sym").ret.agg(["size", "mean", "var"])
            g = g[g["size"] >= 5]
            # 종목 평균의 잡음 분산 = 그날 전체 공통 효과 때문에 군집이 있지만, 여기선 거래 독립 가정(하한 잡음 → tau 과대 쪽)
            se2 = (g["var"] / g["size"]).values
            w = 1 / se2
            mu = (w * g["mean"]).sum() / w.sum()
            Q = (w * (g["mean"] - mu) ** 2).sum()
            k = len(g)
            tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
            P(f"   {sf:5s} {per:5s}: {k}종목, 관측 평균의 sd {g['mean'].std() * 100:.2f}%, 잡음만의 sd {np.sqrt(se2.mean()) * 100:.2f}%,"
              f" 진짜 차이 sd(tau) {np.sqrt(tau2) * 100:.2f}% (Q {Q:.1f}, 자유도 {k - 1})")
    # 짝 방향 유지 기준선: 모든 종목 짝에서 앞 구간 차이 부호가 뒤 구간에서 유지된 비율
    P("\n  기준선: AI 유니버스 모든 종목 짝에서 '앞 구간에 나았던 쪽이 뒤 구간에도 나았던' 비율:")
    nt = 0
    for sf in ["정규", "전체", "꼬리뺀전체"]:
        for p1, p2, mn1, mn2 in [("P0", "P1", 5, 5), ("P0+P1", "P2", 5, 3)]:
            m1 = SESF[sf](pick(T, p1)).groupby("sym").ret.agg(["size", "mean"])
            m2 = SESF[sf](pick(T, p2)).groupby("sym").ret.agg(["size", "mean"])
            ok = sorted(set(m1[m1["size"] >= mn1].index) & set(m2[m2["size"] >= mn2].index))
            a1, a2 = m1.loc[ok, "mean"].values, m2.loc[ok, "mean"].values
            s1 = np.sign(a1[:, None] - a1[None, :])
            s2 = np.sign(a2[:, None] - a2[None, :])
            iu = np.triu_indices(len(ok), 1)
            keep = (s1[iu] != 0) & (s2[iu] != 0)
            same = (s1[iu] == s2[iu])[keep].mean()
            # 큰 차이(앞 구간 차이 상위 1/4) 짝만
            dif1 = np.abs(a1[:, None] - a1[None, :])[iu][keep]
            big = dif1 >= np.quantile(dif1, 0.75)
            same_big = (s1[iu] == s2[iu])[keep][big].mean()
            nt += 1
            nb_ = (A in ok and B in ok)
            ex = ""
            if nb_:
                i, j = ok.index(A), ok.index(B)
                ex = f" | NBIS−IREN 앞 {pct(a1[i] - a1[j])} → 뒤 {pct(a2[i] - a2[j])}"
            P(f"   {sf:5s} {p1}→{p2}: {len(ok)}종목 {keep.sum()}짝, 방향 유지 {same:.1%}, 앞 차이 상위 1/4 짝만 {same_big:.1%}{ex}")
    NTEST["3 짝 방향 유지 기준선(세션3×구간쌍2)"] = nt


# ── 4. 비트코인 연동과 묶음 비교 ─────────────────────────
def btc_series(raw):
    b1 = pd.read_parquet(f"{raw}/1m_BTC-USD.parquet").Close
    b5 = pd.read_parquet(f"{raw}/5m_BTC-USD.parquet").Close
    return b1, b5


def btc_ret(b1, b5, t0, t1):
    """보유 중 BTC 수익: 매수 봉 시작 직전 마지막 종가 → 매도 봉 시작 직전 마지막 종가. 9/25 이후 1분봉, 그 전 5분봉."""
    s = b1 if t0 >= pd.Timestamp("2026-09-25", tz=ET) else b5
    if t0 <= s.index[0]:
        return np.nan
    p0 = s.asof(t0 - pd.Timedelta(seconds=1))
    p1 = s.asof(t1 - pd.Timedelta(seconds=1))
    return p1 / p0 - 1


def pre_btc_corr(raw, s, start, n=60):
    """사전 값: 기간 시작 전 n거래일 일간 수익의 BTC 상관·베타(BTC는 UTC 일봉, 약 4시간 어긋남)."""
    c = pd.read_parquet(f"{raw}/1d_{s}.parquet").Close
    c.index = pd.to_datetime(c.index).date
    bt = pd.read_parquet(f"{raw}/1d_BTC-USD.parquet").Close
    bt.index = pd.to_datetime(bt.index).date
    bt = bt.reindex(c.index).ffill()
    r = pd.DataFrame({"s": c.pct_change(), "b": bt.pct_change()}).dropna()
    r = r[[d < start for d in r.index]].tail(n)
    return r.s.corr(r.b), r.s.cov(r.b) / r.b.var()


def pre_btc_corr5(raw, s, lo, hi):
    """사전 값: [lo, hi] 구간 5분봉(정규장) 수익의 BTC 5분봉 상관."""
    q = pd.read_parquet(f"{raw}/5m_{s}.parquet").Close
    b = pd.read_parquet(f"{raw}/5m_BTC-USD.parquet").Close
    q = q[(q.index.date >= lo) & (q.index.date <= hi)]
    m = q.index.hour * 60 + q.index.minute
    q = q[(m >= 570) & (m < 960)]
    rq = np.log(q).diff()
    rb = np.log(b.reindex(q.index)).diff()
    ok = (pd.Series(q.index.date, index=q.index) == pd.Series(q.index.date, index=q.index).shift(1)) & rq.notna() & rb.notna()
    return rq[ok].corr(rb[ok]), int(ok.sum())


def sec4(T, raw):
    P("\n== 4. 구조 가설: 채굴→AI 종목은 보유 중 비트코인과 같이 움직이는가 ==")
    b1, b5 = btc_series(raw)
    grp = NEO + MIN
    X = T[T.day >= date(2026, 8, 25)].copy()           # BTC 5분봉이 8/24 20시부터
    X["btc"] = [btc_ret(b1, b5, a, b) for a, b in zip(X.t, X.t_exit)]
    X["grp"] = np.where(X.sym.isin(NEO), "네오클라우드", np.where(X.sym.isin(MIN), "채굴→AI", "나머지AI"))
    # 사전 값
    P("\n  4a-1. 사전 값(기간 시작 전에 아는 BTC 연동):")
    for per in ["P1", "P2"]:
        lo = PER[per][0]
        p5lo, p5hi = (date(2026, 8, 25), date(2026, 9, 9)) if per == "P1" else (date(2026, 9, 10), date(2026, 10, 8))
        for s in grp:
            c, be = pre_btc_corr(raw, s, lo)
            c5, n5 = pre_btc_corr5(raw, s, p5lo, p5hi)
            P(f"   {per} 시작 전 {s:5s}: 60거래일 일간 상관 {c:+.2f} 베타 {be:+.2f} | 앞 구간({p5lo:%m/%d}~{p5hi:%m/%d}) 5분봉 상관 {c5:+.2f} ({n5}봉)")
    # 사후 노출: 8시 정리 거래 수익과 보유 중 BTC 수익
    P("\n  4a-2. 사후 노출(설명용): 거래 수익 vs 보유 중 BTC 수익, 정규·프리 매수만(애프터 꼬리 잡음 제외). BTC 5분봉 8/25~, 1분봉 9/25~. P1 매도 시각은 봉 격자에서 되살림(9/25 전은 5분봉 근사)")
    win = {"8/25~10/8": (date(2026, 8, 25), date(2026, 10, 8)), "9/25~10/8(1분봉)": (date(2026, 9, 25), date(2026, 10, 8)),
           "P2": PER["P2"]}
    for wn, (lo, hi) in win.items():
        for why in ["8시", "RSI"]:
            y = X[(X.day >= lo) & (X.day <= hi) & (X.why == why) & X.btc.notna() & (X.ses != "애프터")]
            for s in [A, B, "네오클라우드", "채굴→AI", "나머지AI"]:
                z = y[y.sym == s] if s in (A, B) else y[y.grp == s]
                if len(z) < 3:
                    P(f"   {wn:16s} {why} {s:6s}: {len(z)}건(3건 미만)")
                    continue
                c = z.ret.corr(z.btc)
                sl = np.polyfit(z.btc, z.ret, 1)[0] if z.btc.std() > 0 else np.nan
                P(f"   {wn:16s} {why} {s:6s}: {len(z):3d}건 상관 {c:+.2f} 기울기 {sl:+.2f} | 평균 수익 {pct(z.ret.mean())} 평균 보유 중 BTC {pct(z.btc.mean())}")
    # 채굴 묶음의 BTC 기울기가 나머지 AI보다 큰가(8시 정리 거래, 날짜 부트스트랩)
    P("\n  4a-3. 8시 정리 거래의 BTC 기울기: 채굴→AI − 나머지 AI (날짜 부트스트랩 2천 번, 설명용):")
    for wn, (lo, hi) in win.items():
        y = X[(X.day >= lo) & (X.day <= hi) & (X.why == "8시") & X.btc.notna() & (X.ses != "애프터") & X.grp.isin(["채굴→AI", "나머지AI"])]
        days = np.array(sorted(set(y.day)))

        def slope_gap(z):
            out = []
            for g in ["채굴→AI", "나머지AI"]:
                w_ = z[z.grp == g]
                out.append(np.polyfit(w_.btc, w_.ret, 1)[0] if len(w_) > 2 and w_.btc.std() > 0 else np.nan)
            return out[0] - out[1]
        g0 = slope_gap(y)
        byd = {d: y[y.day == d] for d in days}
        bs = []
        for _ in range(2000):
            pickd = rng.choice(days, len(days))
            bs.append(slope_gap(pd.concat([byd[d] for d in pickd])))
        bs = np.array(bs)
        bs = bs[np.isfinite(bs)]
        P(f"   {wn:16s}: 채굴 {(y.grp == '채굴→AI').sum()}건, 나머지 {(y.grp == '나머지AI').sum()}건, 기울기 차이 {g0:+.2f} 구간 [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
    NTEST["4a 사후 노출 표(설명용, 판정 아님)"] = 0

    # 4c. 유니버스 전체: 사전 BTC 상관이 높은 종목일수록 이격매수 성적이 나쁜가(순위 상관, 날짜 군집 아님 — 종목 단위)
    P("\n  4c. AI 유니버스 전체: 기간 시작 전 60거래일 BTC 일간 상관 vs 그 기간 종목 평균(5건 이상 종목, 스피어만):")
    nt = 0
    for per in ["P0", "P1", "P2"]:
        lo = PER[per][0]
        cs = {s: pre_btc_corr(raw, s, lo)[0] for s in T.sym.unique()}
        for sf in ["정규", "전체", "꼬리뺀전체"]:
            y = SESF[sf](pick(T, per))
            m = y.groupby("sym").ret.agg(["size", "mean"])
            m = m[m["size"] >= 5]
            c = pd.Series({s: cs[s] for s in m.index})
            rho = m["mean"].rank().corr(c.rank())
            top = c[c >= c.quantile(2 / 3)].index
            gap = m.loc[top, "mean"].mean() - m.loc[~m.index.isin(top), "mean"].mean()
            nt += 1
            P(f"   {per} {sf:5s}: {len(m)}종목, 순위 상관 {rho:+.2f}, BTC 상관 상위 1/3 − 나머지 {pct(gap)} (상위 1/3: {', '.join(sorted(top)[:8])}{'…' if len(top) > 8 else ''})")
    NTEST["4c 사전 BTC 상관 vs 종목 평균(기간3×세션3)"] = nt

    # 4b. 묶음 비교: 앞(P0+P1)에서 방향 → 뒤(P2) 확인
    P("\n  4b. 묶음 비교: 네오클라우드(CRWV·NBIS) vs 채굴→AI(APLD·CIFR·GLXY·HUT·IREN·WULF)")
    P("      앞 구간 P0+P1에서 방향을 정하고 P2에서 같은 방향인지 본다. '초과'는 같은 날·같은 세션 AI 유니버스 평균 대비.")
    nt = 0
    for nm, ga, gb in [("묶음 전체", NEO, MIN), ("두 종목 뺀 묶음(CRWV vs 채굴 5종목)", ["CRWV"], [m for m in MIN if m != B])]:
        P(f"   [{nm}]")
        for sf in ["정규", "전체", "꼬리뺀전체"]:
            for col, cn in [("ret", "원"), ("ex", "초과")]:
                line = []
                sgn = None
                for per in ["P0", "P1", "P0+P1", "P2"]:
                    y = SESF[sf](pick(T, per))
                    a, b = y[y.sym.isin(ga) & y[col].notna()], y[y.sym.isin(gb) & y[col].notna()]
                    d = a[col].mean() - b[col].mean()
                    t = diff_ct(a[col], a.day, b[col], b.day)
                    lo, hi, _ = boot_diff(a[col].values, a.day.values, b[col].values, b.day.values)
                    if per == "P0+P1":
                        sgn = np.sign(d)
                    mark = ""
                    if per == "P2" and sgn is not None:
                        mark = " 방향 유지" if np.sign(d) == sgn else " 방향 반대"
                        KEY.setdefault("4b", []).append(np.sign(d) == sgn)
                    line.append(f"{per} {len(a)}/{len(b)}건 {pct(d)}(t {t:+.2f}, [{pct(lo)},{pct(hi)}]){mark}")
                nt += 1
                P(f"    {sf:5s} {cn}: " + " | ".join(line))
    NTEST["4b 묶음 비교(묶음정의2×세션3×원/초과2, 앞에서 방향→P2 확인)"] = nt
    # 묶음 안 종목별 정규장 평균(참고)
    P("\n   묶음 안 종목별 평균(정규 / 전체):")
    for s in grp:
        r = []
        for per in ["P0", "P1", "P2"]:
            y = pick(T, per)
            y = y[y.sym == s]
            yr = y[y.ses == "정규"]
            r.append(f"{per} 정규 {len(yr)}건 {pct(yr.ret.mean())} 전체 {len(y)}건 {pct(y.ret.mean())}")
        P(f"    {s:5s}: " + " | ".join(r))
    return X


def sec5(T, X, raw):
    P("\n== 5. 사후 사례(설명용, 근거 아님): 차트에 보였을 최근 라벨 ==")
    q = pd.read_parquet(f"{raw}/1d_QQQ.parquet").Close
    q.index = pd.to_datetime(q.index).date
    qr = q.pct_change()
    # 같은 날 묶음 평균
    gm = T.groupby(["day", "sub"]).ret.mean()
    lo = date(2026, 9, 23)
    for s in [A, B]:
        y = X[(X.sym == s) & (X.day >= lo)].sort_values("ret")
        P(f"\n  {s} 9/23~10/22 라벨 {len(y)}건 합 {pct(y.ret.sum(), 1)} (9/23~10/8 {pct(y[y.period == 'P1'].ret.sum(), 1)}, 10/9~10/22 {pct(y[y.period == 'P2'].ret.sum(), 1)})")
        for _, r in pd.concat([y.head(4), y.tail(3)]).iterrows():
            P(f"    {r.day:%m/%d} {r.t:%H:%M} {r.ses:3s} {r.why:3s} {pct(r.ret):>8s} 보유 {r.hold_min:4.0f}분 MAE {pct(r.mae):>8s} 꼬리 {'' if pd.isna(r["tail"]) else int(r["tail"])}"
              f" | 보유 중 BTC {pct(r.btc)} | QQQ 그날 {pct(qr.get(r.day, np.nan))} | 같은 소분류 그날 평균 {pct(gm.get((r.day, r['sub']), np.nan))}")
    # 9/10~10/8(사용자가 인상을 얻었을 법한 구간) 합과 큰 손실 수
    P("\n  구간별 라벨 합(전체 세션)과 −5% 이하 손실 건수:")
    for per in ["P0", "P1", "P2"]:
        for s in [A, B]:
            y = pick(T, per)
            y = y[y.sym == s]
            P(f"   {per} {s}: {len(y)}건 합 {pct(y.ret.sum(), 1)}, +3% 이상 {(y.ret >= 0.03).sum()}건, −5% 이하 {(y.ret <= -0.05).sum()}건")


def main(raw):
    T = load(raw)
    P("이격매수매도 AI 종목 궁합 4부 — NBIS vs IREN (2026-10-23)")
    P(f"거래표 {len(T)}건, NBIS {(T.sym == A).sum()}건, IREN {(T.sym == B).sum()}건. 주 잣대 정규장, 보조 전체·꼬리뺀전체.")
    P("기간: P0 8/3~9/9(5분봉 근사), P1 9/10~10/8(3분봉), P2 10/9~10/22(3분봉, 새 구간).")
    sec1(T)
    sec1b(T)
    sec2(T, raw)
    sec3(T)
    X = sec4(T, raw)
    sec5(T, X, raw)
    P("\n== 요약(위 숫자에서 뽑음) ==")
    for per in ["P0", "P1", "P2", "전체"]:
        d, t, lo, hi, na_, nb = KEY[("1b", per, "정규", "ret")]
        P(f"  정규장 NBIS−IREN {per}: {pct(d)} (t {t:+.2f}, 구간 [{pct(lo)}, {pct(hi)}], {na_}/{nb}건)")
    for per in ["P1", "전체"]:
        d, t, lo, hi, na_, nb = KEY[("1b", per, "전체", "ret")]
        e = KEY[("1b", per, "전체", "ex")]
        P(f"  전체 세션 {per}: 원 {pct(d)} (t {t:+.2f}) → 같은 날·같은 세션 유니버스 평균 뺀 초과 {pct(e[0])} (t {e[1]:+.2f})")
    for sf in ["정규", "전체"]:
        m, lo, hi, n, w = KEY[("2", sf)]
        P(f"  같은 날 짝({sf}): {n}일 평균 {pct(m)} 구간 [{pct(lo)}, {pct(hi)}], NBIS가 나은 날 {w}/{n}")
    d, sa, sb, nt2, n80 = KEY[("3", "정규", "P0+P1")]
    P(f"  잡음: 정규장 거래당 sd NBIS {sa * 100:.2f}% IREN {sb * 100:.2f}%, P0+P1 차이 {pct(d)}를 가리려면 종목당 {nt2:.0f}건(t=2)·{n80:.0f}건(검정력 80%)")
    k = KEY.get("4b", [])
    P(f"  묶음 비교(네오클라우드 vs 채굴→AI): P0+P1 방향이 P2에서 유지된 조합 {sum(k)}/{len(k)}")
    NTEST["2 같은 날 짝(기간3×세션3 + 전 기간3)"] = 12
    NTEST["4a-3 BTC 기울기 차이(구간3, 설명용)"] = 3
    P("\n== 시험 수 ==")
    for k, v in NTEST.items():
        P(f"  {k}: {v}")
    open(os.path.join(RES, "ai_fit_pair.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
