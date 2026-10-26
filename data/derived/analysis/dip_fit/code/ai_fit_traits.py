"""이격매수매도 AI 종목 궁합 4부 — 갈래 '미리 아는 종목 성질'(2026-10-23 사용자: AI 종목 중 이격매수매도에 더 잘 맞는 종목은? 예: IREN보다 NBIS).

질문: 구간 시작 전에 알 수 있는 종목 성질로 '잘 맞는 종목'을 미리 고를 수 있나. P0+P1(정규장)에서 고르고 새 구간 P2에서 확인한다.
입력: ../results/ai_fit_trades.csv (ai_fit_build.py 산출, 고치지 않음), raw_1023 일봉·5분봉(1분봉은 P2 기준값 점검에만).

[시험 전에 고정한 후보 성질 11개] 값은 각 구간 시작 전날까지의 자료로만 잰다.
  기준일: P0(8/3~9/9) → 7/31 종가, P1(9/10~10/8) → 9/9 종가, P2(10/9~10/22) → 10/8 종가.
  지난 연구 성질(일봉):
    1 vol20     일봉 수익률 표준편차, 20거래일
    2 beta60    QQQ 대비 베타, 60거래일
    3 corr60    QQQ 일봉 상관, 60거래일
    4 ma50      종가 / 50일 이동평균 - 1 (추세)
    5 adv20     log10(종가 x 거래량) 20일 평균 (거래대금)
    6 gap60     시가 갭 |시가/전날 종가-1| > 2%인 날 비율, 60거래일
  장중 성질(5분봉 정규장, 기준일까지 20거래일. 5분봉은 7/31부터라 P0 기준값은 없다 → 이 성질들은 P1만으로 고른다):
    7 ac1       5분 로그수익률 1차 자기상관(같은 날 안의 쌍만)
    8 vr12      5분봉 분산비 = 1시간(12봉) 수익률 분산 / (12 x 5분 수익률 분산). 1보다 작으면 장중 평균회귀 성향
    9 depth     0.02 / (정규장 5분봉 종가의 EMA36 대비 이격 표준편차). EMA36은 프리·정규·애프터 전체 봉으로(지표처럼 24시간) 계산.
                값이 클수록 그 종목에는 2% 이격이 깊다(변동성 큰 종목에는 2%가 얕다). 5분봉 EMA36 = 3분봉 EMA60과 같은 180분.
                점검: P2 기준값은 1분봉→3분봉 EMA60(9/25~, 첫날 길들이기)으로도 재서 5분봉 값과 순위상관을 보인다(결과에는 5분봉 값만 쓴다).
  신호 성질(앞 구간 거래표):
   10 rsi_exit  앞 구간 신호(전체 세션) 중 RSI 매도로 끝난 비율(8시 정리까지 가지 않은 비율). 5건 미만이면 없음. P0은 앞 구간이 없어 없음.
                P1 값은 P0(5분봉 근사) 거래, P2 값은 P1 거래로 잰다. 앞 구간 마지막 거래도 다음 구간 시작 전에 끝난다.
  새 성질(일봉):
   11 btc60     IBIT 일봉 수익률 상관, 60거래일 (비트코인 대용. BTC-USD 일봉은 UTC 자정 기준이라 미국 장 종가와 어긋나서 IBIT를 쓴다)

[방법(미리 고정)]
  - 성질마다 구간별로 115종목을 상위 1/3·하위 1/3로 나눈다(문턱은 그 구간 기준값의 종목 분포, 거래 결과와 무관).
  - 고르는 표본: P0+P1 정규장 거래(주 잣대). 장중·신호 성질(7~10)은 P0 값이 없어 P1 정규장만.
  - 두 잣대, 모두 날짜 군집 표준오차:
      원값 차   = 상위 1/3 거래당 평균 - 하위 1/3 거래당 평균 (ret ~ 1 + 상위, 날짜 군집)
      날짜 뺀 차 = 매수일 고정효과 안에서의 차 (같은 날 안 비교. 업종 동반 하락일 영향을 뺀다)
  - 고르기 점수 = 두 t의 부호가 같으면 min(|t원값|, |t날짜뺌|), 다르면 0. 점수 2.0 이상만 후보 → P2에서 확인.
  - P2 확인(미리 고정): P2 정규장 날짜 뺀 차가 고른 방향이고 t >= 2.0, 원값 차도 같은 방향, 전체 세션(날짜x세션 고정효과)도 같은 방향.
    가장 유리한 하루를 빼도 같은 방향인지, 날짜별 차이가 며칠에 몰렸는지 함께 보인다.
  - 보조(판정에 안 씀): P0만·P1만 날짜 뺀 t, 전체 세션(날짜x세션 고정효과) t, P2 값은 11개 모두 표시.
  - 우연 수준: 가짜 성질 = 진짜 성질 값을 종목끼리 무작위로 섞은 것(같은 섞기를 세 구간에 그대로 적용 → 구간 간 지속성·빈칸 모양 유지).
    성질마다 60개, 모두 660개에 같은 고르기·확인을 돌린다. 11개 묶음 최고 점수의 우연 분포는 성질마다 가짜 1개씩 뽑아 2,000회.
  - 결과를 본 뒤 덧붙인 것(판정에 안 씀, 보고서에 '사후'로 표시): 느슨한 고르기(날짜 뺀 |t| >= 1.5) 민감도,
    P2 전체 세션 |t| >= 2인 성질의 세션별·꼬리 체결 뺀 값과 날짜별 분해, P2 프리장 11개 t와 그 최댓값의 가짜 성질 우연 수준.
저장: ../results/ai_fit_traits.txt(보고), ../results/ai_fit_traits_values.csv(종목x구간 성질 값), ../results/ai_fit_traits_tests.csv(11개 성질 시험값)
사용: py ai_fit_traits.py [raw 폴더]"""
import os, re, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.insert(0, os.path.join(HERE, "../../../../.."))
from events import bars, prep
import watchlist as w

RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4b7d1886-4bb9-4d2c-b134-8841c35220ef/scratchpad/dip/raw_1023"
BASE = {"P0": date(2026, 7, 31), "P1": date(2026, 9, 9), "P2": date(2026, 10, 8)}
PREV = {"P1": "P0", "P2": "P1"}
TRAITS = ["vol20", "beta60", "corr60", "ma50", "adv20", "gap60", "ac1", "vr12", "depth", "rsi_exit", "btc60"]
INTRA = {"ac1", "vr12", "depth", "rsi_exit"}          # P0 값이 없는 성질
NAME = {"vol20": "일봉 변동성 20일", "beta60": "QQQ 베타 60일", "corr60": "QQQ 상관 60일", "ma50": "50일선 대비 위치",
        "adv20": "거래대금 20일", "gap60": "갭 2% 넘는 날 비율", "ac1": "5분 자기상관", "vr12": "5분 분산비(1시간)",
        "depth": "2% 이격 깊이(이격 표준편차 단위)", "rsi_exit": "앞 구간 RSI 매도 비율", "btc60": "IBIT 상관 60일"}
N_PERM, N_FAM, SEL, CONF = 60, 2000, 2.0, 2.0
rng = np.random.default_rng(20261023)
AI = {}
for big, subs in w.AI_UNIVERSE.items():
    for sub, ts in subs.items():
        for t in ts:
            if re.fullmatch(r"[A-Z]{1,5}", t):
                AI.setdefault(t, (big, sub))
SYMS = sorted(AI)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)


def pct(v):
    return f"{v * 100:+.2f}%" if np.isfinite(v) else "  n/a"


# ---------------------------------------------------------------- 성질 계산

def daily(s):
    x = pd.read_parquet(f"{RAW}/1d_{s}.parquet")
    x.index = pd.to_datetime(x.index).date
    return x


def reg5(s):
    """5분봉 전체(24시간) EMA36 이격 + 정규장 표시."""
    b = pd.read_parquet(f"{RAW}/5m_{s}.parquet")
    b = b[b.High > 0]
    ema = b.Close.ewm(span=36, adjust=False).mean()
    mod = b.index.hour * 60 + b.index.minute
    return pd.DataFrame({"c": b.Close.values, "dev": (b.Close / ema - 1).values, "reg": (mod >= 570) & (mod < 960),
                         "date": b.index.date}, index=b.index)


def intraday_traits(m, days):
    x = m[m.reg & np.isin(m.date, days)]
    if x.date.nunique() < 15:
        return dict(ac1=np.nan, vr12=np.nan, depth=np.nan)
    lc = np.log(x.c)
    r = lc.groupby(x.date).diff()
    r1 = r.groupby(x.date).shift(1)
    ok = r.notna() & r1.notna()
    h = lc.groupby(x.date).diff(12)
    return dict(ac1=np.corrcoef(r[ok], r1[ok])[0, 1], vr12=h.var() / (12 * r.var()), depth=0.02 / x.dev.std())


def depth_3m(s, lo, hi):
    """P2 기준값 점검: 1분봉→3분봉 EMA60 이격 표준편차(정규장, lo~hi)."""
    d = prep(bars(RAW, s, 3), 60, 14)
    x = d[(d.ses == "정규") & (d.date >= lo) & (d.date <= hi)]
    return 0.02 / x.dev.std()


def traits(T):
    q = daily("QQQ").Close.pct_change()
    ib = daily("IBIT").Close.pct_change()
    tdays = list(daily("QQQ").index)
    rows = []
    for s in SYMS:
        x = daily(s)
        r = x.Close.pct_change()
        m = reg5(s)
        for p, b in BASE.items():
            xx, rr = x[x.index <= b], r[r.index <= b]
            jq = pd.concat([rr, q], axis=1, join="inner").dropna().iloc[-60:]
            jb = pd.concat([rr, ib], axis=1, join="inner").dropna().iloc[-60:]
            gp = (xx.Open / xx.Close.shift() - 1).iloc[-60:]
            f = dict(sym=s, period=p, vol20=rr.iloc[-20:].std(),
                     beta60=jq.iloc[:, 0].cov(jq.iloc[:, 1]) / jq.iloc[:, 1].var(), corr60=jq.iloc[:, 0].corr(jq.iloc[:, 1]),
                     ma50=xx.Close.iloc[-1] / xx.Close.iloc[-50:].mean() - 1, adv20=np.log10((xx.Close * xx.Volume).iloc[-20:].mean()),
                     gap60=(gp.abs() > 0.02).mean(), btc60=jb.iloc[:, 0].corr(jb.iloc[:, 1]))
            win = [d for d in tdays if d <= b][-20:]
            f.update(intraday_traits(m, win) if p != "P0" else dict(ac1=np.nan, vr12=np.nan, depth=np.nan))
            if p in PREV:
                g = T[(T.sym == s) & (T.period == PREV[p])]
                f["rsi_exit"] = (g.why == "RSI").mean() if len(g) >= 5 else np.nan
            else:
                f["rsi_exit"] = np.nan
            rows.append(f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 검정

def ols_cl(y, a, g, fe=None):
    """y ~ a (fe 묶음 고정효과면 묶음 안 평균을 뺀다), 날짜(g) 군집 표준오차. 반환: (계수, t)."""
    if fe is not None:
        y = y - pd.Series(y).groupby(fe).transform("mean").values
        a = a - pd.Series(a).groupby(fe).transform("mean").values
        X = a[:, None]
    else:
        X = np.column_stack([np.ones_like(a), a])
    XtX = X.T @ X
    if np.linalg.cond(XtX) > 1e12:
        return np.nan, np.nan
    inv = np.linalg.inv(XtX)
    b = inv @ X.T @ y
    u = y - X @ b
    gi = pd.factorize(g)[0]
    G = gi.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, gi, X * u[:, None])
    V = inv @ (S.T @ S) @ inv * G / (G - 1)
    se = np.sqrt(V[-1, -1])
    return b[-1], (b[-1] / se if se > 0 else np.nan)


def split_tests(x, lab, scope="정규"):
    """x: 거래(열 ret, day, ses), lab: 1=상위 1/3, 0=하위 1/3, nan=가운데. 반환 dict(원값 차·t, 날짜 뺀 차·t, 건수)."""
    k = ~np.isnan(lab)
    if scope == "정규":
        k &= (x.ses.values == "정규")
    y, a, d = x.ret.values[k], lab[k], x.day.values[k]
    if len(y) < 10 or a.min() == a.max():
        return dict(raw=np.nan, t_raw=np.nan, fe=np.nan, t_fe=np.nan, nA=int((a == 1).sum()), nB=int((a == 0).sum()))
    raw, t_raw = ols_cl(y, a, d)
    fe_g = d if scope == "정규" else pd.Series(d).astype(str).values + "|" + x.ses.values[k]
    fe, t_fe = ols_cl(y, a, d, fe=fe_g)
    return dict(raw=raw, t_raw=t_raw, fe=fe, t_fe=t_fe, nA=int((a == 1).sum()), nB=int((a == 0).sum()))


def score(r):
    if not (np.isfinite(r["t_raw"]) and np.isfinite(r["t_fe"])) or np.sign(r["t_raw"]) != np.sign(r["t_fe"]):
        return 0.0
    return min(abs(r["t_raw"]), abs(r["t_fe"]))


def tercile_labels(V, col, perm=None):
    """구간별 상위 1/3 = 1, 하위 1/3 = 0. perm이 있으면 종목 값을 그 순서로 섞는다(세 구간 같은 섞기)."""
    out = {}
    for p in ["P0", "P1", "P2"]:
        v = V[V.period == p].set_index("sym")[col].reindex(SYMS)
        if perm is not None:
            v = pd.Series(v.values[perm], index=SYMS)
        lo, hi = v.quantile([1 / 3, 2 / 3]).values if v.notna().sum() >= 30 else (np.nan, np.nan)
        lab = pd.Series(np.nan, index=SYMS)
        if np.isfinite(lo):
            lab[v >= hi], lab[v <= lo] = 1.0, 0.0
        out[p] = lab
    return out


def label_trades(T, labs):
    return np.array([labs[p].get(s, np.nan) for s, p in zip(T.sym.values, T.period.values)], float)


def run_trait(T, labs, col):
    lab = label_trades(T, labs)
    sel_p = ["P1"] if col in INTRA else ["P0", "P1"]
    ks = T.period.isin(sel_p).values
    k2 = (T.period == "P2").values
    r = {}
    r["sel"] = split_tests(T[ks], lab[ks])
    r["sel_all"] = split_tests(T[ks], lab[ks], "전체")
    for p in ["P0", "P1"]:
        kp = (T.period == p).values
        r[p] = split_tests(T[kp], lab[kp])
    r["P2"] = split_tests(T[k2], lab[k2])
    r["P2_all"] = split_tests(T[k2], lab[k2], "전체")
    r["score"] = score(r["sel"])
    r["dir"] = np.sign(r["sel"]["fe"]) if np.isfinite(r["sel"]["fe"]) else 0.0
    return r, lab


def confirm(r):
    d, p2, p2a = r["dir"], r["P2"], r["P2_all"]
    return bool(d != 0 and np.isfinite(p2["t_fe"]) and d * p2["t_fe"] >= CONF and d * p2["raw"] > 0 and d * p2a["fe"] > 0)


# ---------------------------------------------------------------- 보고

def day_table(x, lab):
    """P2 정규장 날짜별 상위-하위 차(양쪽 다 있는 날)."""
    k = ~np.isnan(lab) & (x.ses.values == "정규")
    y = x[k].assign(A=lab[k])
    m = y.groupby(["day", "A"]).ret.agg(["mean", "size"]).unstack()
    return m


def main():
    T = pd.read_csv(os.path.join(RES, "ai_fit_trades.csv"))
    T["day"] = T.day.astype(str)
    T = T[T.sym.isin(AI)].reset_index(drop=True)
    V = traits(T)
    V.to_csv(os.path.join(RES, "ai_fit_traits_values.csv"), index=False, encoding="utf-8-sig")

    say("이격매수매도 AI 종목 궁합 — 갈래: 미리 아는 종목 성질로 잘 맞는 종목을 고를 수 있나 (ai_fit_traits.py, 2026-10-23)")
    say(f"거래: {len(T)}건 (P0 {sum(T.period == 'P0')}, P1 {sum(T.period == 'P1')}, P2 {sum(T.period == 'P2')}), 정규장 "
        f"P0 {sum((T.period == 'P0') & (T.ses == '정규'))} / P1 {sum((T.period == 'P1') & (T.ses == '정규'))} / "
        f"P2 {sum((T.period == 'P2') & (T.ses == '정규'))}건, 매수일 P0 {T[T.period == 'P0'].day.nunique()} / "
        f"P1 {T[T.period == 'P1'].day.nunique()} / P2 {T[T.period == 'P2'].day.nunique()}일")
    say("성질 기준일: P0 7/31, P1 9/9, P2 10/8 종가까지. 장중·신호 성질(ac1 vr12 depth rsi_exit)은 P0 값이 없어 P1만으로 고른다.")
    say("")

    # 성질 지속성(앞 구간 → 다음 구간 순위상관, 같은 1/3 유지 비율)
    say("[0] 성질 값의 구간 간 지속성 (종목 순위상관 / 상·하 1/3 같은 칸 유지 비율)")
    for c in TRAITS:
        a = V[V.period == "P1"].set_index("sym")[c]
        b = V[V.period == "P2"].set_index("sym")[c]
        z = V[V.period == "P0"].set_index("sym")[c]
        s12 = a.corr(b, method="spearman")
        s01 = z.corr(a, method="spearman") if z.notna().sum() > 30 else np.nan
        la, lb = tercile_labels(V, c)["P1"], tercile_labels(V, c)["P2"]
        both = la.notna() & lb.notna()
        say(f"  {c:9s} P0→P1 {s01:+.2f}  P1→P2 {s12:+.2f}  P1·P2 모두 상·하 1/3인 종목 {both.sum()}개 중 같은 칸 {(la[both] == lb[both]).mean():.0%}")
    v5 = V[V.period == "P2"].set_index("sym").depth
    v3 = pd.Series({s: depth_3m(s, date(2026, 9, 29), BASE["P2"]) for s in SYMS})
    say(f"  depth 점검: P2 기준값 5분봉(EMA36, 20일) vs 3분봉(EMA60, 9/29~10/8) 순위상관 {v5.corr(v3, method='spearman'):+.2f}")
    say("")

    # 진짜 성질
    res, labs_all = {}, {}
    for c in TRAITS:
        labs = tercile_labels(V, c)
        res[c], labs_all[c] = run_trait(T, labs, c)
    say("[1] 고르기: P0+P1 정규장(장중·신호 성질은 P1만), 상위 1/3 - 하위 1/3 거래당 평균 차, 날짜 군집 t")
    say("    성질       상위건 하위건   원값 차   t     날짜 뺀 차  t   | 점수 | P0 날짜뺌 t | P1 날짜뺌 t | 전체세션 t")
    rows = []
    for c in TRAITS:
        r = res[c]
        s = r["sel"]
        say(f"  {c:9s} {s['nA']:5d} {s['nB']:5d}  {pct(s['raw'])} {s['t_raw']:+5.2f}  {pct(s['fe'])} {s['t_fe']:+5.2f} | {r['score']:4.2f} |"
            f"   {r['P0']['t_fe']:+5.2f}    |   {r['P1']['t_fe']:+5.2f}    |   {r['sel_all']['t_fe']:+5.2f}")
        rows.append(dict(trait=c, name=NAME[c], sel_period="P1" if c in INTRA else "P0+P1", nA=s["nA"], nB=s["nB"],
                         sel_raw=s["raw"], sel_t_raw=s["t_raw"], sel_fe=s["fe"], sel_t_fe=s["t_fe"], score=r["score"],
                         P0_t_fe=r["P0"]["t_fe"], P1_t_fe=r["P1"]["t_fe"], sel_all_t_fe=r["sel_all"]["t_fe"],
                         P2_nA=r["P2"]["nA"], P2_nB=r["P2"]["nB"], P2_raw=r["P2"]["raw"], P2_t_raw=r["P2"]["t_raw"],
                         P2_fe=r["P2"]["fe"], P2_t_fe=r["P2"]["t_fe"], P2_all_fe=r["P2_all"]["fe"], P2_all_t_fe=r["P2_all"]["t_fe"],
                         candidate=r["score"] >= SEL, confirmed=(r["score"] >= SEL) and confirm(r)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "ai_fit_traits_tests.csv"), index=False, encoding="utf-8-sig")
    cands = [c for c in TRAITS if res[c]["score"] >= SEL]
    say(f"  후보(점수 >= {SEL}): {', '.join(cands) if cands else '없음'}")
    say("")

    # 가짜 성질
    say(f"[2] 우연 수준: 가짜 성질 {len(TRAITS)} x {N_PERM} = {len(TRAITS) * N_PERM}개(진짜 값을 종목끼리 섞음)에 같은 절차")
    fake = {c: [] for c in TRAITS}
    for c in TRAITS:
        for _ in range(N_PERM):
            perm = rng.permutation(len(SYMS))
            r, lab = run_trait(T, tercile_labels(V, c, perm), c)
            kp = ((T.period == "P2") & (T.ses == "프리")).values
            pre = split_tests(T[kp].reset_index(drop=True), lab[kp], "전체")
            fake[c].append(dict(score=r["score"], dir=r["dir"], p2t=r["P2"]["t_fe"], p2fe=r["P2"]["fe"], conf=confirm(r),
                                cand=r["score"] >= SEL, pre_t=pre["t_fe"]))
    F = pd.DataFrame([dict(trait=c, **f) for c in TRAITS for f in fake[c]])
    nsel = F.cand.sum()
    say(f"  가짜 중 고르기 점수 >= {SEL}: {nsel}/{len(F)} ({nsel / len(F):.1%}). 그중 P2 확인 통과 {F[F.cand].conf.sum()}개")
    say(f"  가짜 전체 P2 확인 통과(고르기와 무관하게 방향만 맞춘 경우): {F.conf.sum()}/{len(F)} ({F.conf.mean():.1%})")
    real_max = max(res[c]["score"] for c in TRAITS)
    fam_max = np.array([max(F[F.trait == c].score.values[rng.integers(N_PERM)] for c in TRAITS) for _ in range(N_FAM)])
    say(f"  11개 묶음 최고 점수: 진짜 {real_max:.2f} / 가짜 묶음 {N_FAM}회 중앙 {np.median(fam_max):.2f}, 95% {np.quantile(fam_max, 0.95):.2f}, "
        f"진짜 이상 비율 {np.mean(fam_max >= real_max):.2f}")
    say(f"  진짜 성질 중 점수 >= {SEL}: {len(cands)}개 / 가짜 묶음 기대 {11 * nsel / len(F):.2f}개")
    # 검정력: P2 날짜 뺀 차의 표준오차(가짜 성질에서)
    se2 = (F.p2fe / F.p2t).abs().replace([np.inf], np.nan).dropna()
    say(f"  P2 검정력: 상·하 1/3 날짜 뺀 차의 표준오차 중앙 {se2.median() * 100:.2f}%p → t 2를 넘으려면 거래당 약 {2 * se2.median() * 100:.1f}%p 차가 필요")
    say("")

    # P2 확인
    say("[3] P2(10/9~10/22) 확인 — 정규장, 성질 기준일 10/8. 방향은 고르는 구간의 날짜 뺀 차 부호")
    say("    성질       상위건 하위건  원값 차   t     날짜 뺀 차  t   | 고른 방향으로 본 t | 가짜 중 이 이상 | 전체세션 날짜x세션 t | 판정")
    for c in TRAITS:
        r = res[c]
        p2 = r["P2"]
        sd = r["dir"] * p2["t_fe"]
        fk = F[F.trait == c]
        fsd = (fk.dir * fk.p2t).values
        pv = np.mean(fsd >= sd) if np.isfinite(sd) else np.nan
        tag = ("후보·확인" if confirm(r) else "후보·탈락") if c in cands else "후보 아님"
        say(f"  {c:9s} {p2['nA']:5d} {p2['nB']:5d}  {pct(p2['raw'])} {p2['t_raw']:+5.2f}  {pct(p2['fe'])} {p2['t_fe']:+5.2f} |"
            f"       {sd:+5.2f}       |     {pv:.2f}      |       {r['P2_all']['t_fe']:+5.2f}         | {tag}")
    loose = [c for c in TRAITS if abs(res[c]["sel"]["t_fe"]) >= 1.5]
    say("  민감도(판정 아님): 느슨한 규칙 '고르는 구간 날짜 뺀 |t| >= 1.5'만 썼다면 → " +
        (", ".join(f"{c}(고르기 t {res[c]['sel']['t_fe']:+.2f} → P2 고른 방향 t {res[c]['dir'] * res[c]['P2']['t_fe']:+.2f})" for c in loose) or "없음"))
    say("")

    # 후보(없으면 점수 상위 2개)의 P2 날짜별 분해
    show = cands if cands else sorted(TRAITS, key=lambda c: -res[c]["score"])[:2]
    say("[4] " + ("후보" if cands else "후보 없음 → 고르기 점수 상위 2개(설명용)") + "의 P2 날짜별 상위-하위 차(정규장)")
    k2 = (T.period == "P2").values
    for c in show:
        lab = labs_all[c][k2]
        m = day_table(T[k2], lab)
        d = res[c]["dir"]
        say(f"  {c} ({NAME[c]}), 고른 방향: {'상위' if d > 0 else '하위'} 1/3이 낫다")
        diffs = []
        for dy, row in m.iterrows():
            a, b = row[("mean", 1.0)] if ("mean", 1.0) in row else np.nan, row[("mean", 0.0)] if ("mean", 0.0) in row else np.nan
            na, nb = row.get(("size", 1.0), 0), row.get(("size", 0.0), 0)
            df_ = a - b
            diffs.append((dy, df_, na, nb))
            say(f"    {dy}  상위 {pct(a)}({0 if pd.isna(na) else int(na)}건)  하위 {pct(b)}({0 if pd.isna(nb) else int(nb)}건)  차 {pct(df_)}")
        dd = pd.Series({x[0]: x[1] for x in diffs}).dropna() * d
        if len(dd):
            best = dd.idxmax()
            say(f"    고른 방향 날짜별 차: 양수 {int((dd > 0).sum())}일 / 음수 {int((dd < 0).sum())}일, 합 {pct(dd.sum())}, "
                f"가장 유리한 날 {best}({pct(dd[best])}) 빼면 평균 {pct(dd.drop(best).mean())}")
    say("")

    # NBIS·IREN 성질 위치
    say("[5] NBIS·IREN 성질 값과 칸(상=상위 1/3, 하=하위 1/3, 중=가운데). P2 = 10/8 기준")
    for s in ["NBIS", "IREN"]:
        parts = []
        for c in TRAITS:
            v = V[(V.sym == s) & (V.period == "P2")][c].iloc[0]
            l = tercile_labels(V, c)["P2"].get(s, np.nan)
            parts.append(f"{c} {v:.3g}({'상' if l == 1 else '하' if l == 0 else '중'})")
        say(f"  {s}: " + ", ".join(parts))
    for s in ["NBIS", "IREN"]:
        g = T[(T.sym == s) & (T.ses == "정규")]
        say(f"  {s} 정규장 거래당: " + ", ".join(f"{p} {pct(g[g.period == p].ret.mean())}({(g.period == p).sum()}건)" for p in ["P0", "P1", "P2"]))
    say("")
    # 사후 관찰(판정 아님): P2 전체 세션에서 |t| >= 2가 나온 성질을 세션별·꼬리 체결 뺀 값으로 나눠 본다
    post = [c for c in TRAITS if abs(res[c]["P2_all"]["t_fe"]) >= 2]
    say("[6] 사후 관찰(판정 아님, 결과를 본 뒤 고름): P2 전체 세션 날짜x세션 |t| >= 2인 성질 " + (", ".join(post) if post else "없음"))
    x2 = T[k2].reset_index(drop=True)
    for c in post:
        lab = labs_all[c][k2]
        parts = []
        for sc in ["프리", "애프터"]:
            ks = (x2.ses == sc).values
            r = split_tests(x2[ks], lab[ks], "전체")
            parts.append(f"{sc} {pct(r['fe'])}(t {r['t_fe']:+.2f}, {r['nA']}/{r['nB']}건)")
        nt = (x2["tail"] != 1).values
        r = split_tests(x2[nt], lab[nt], "전체")
        parts.append(f"꼬리 체결 뺀 전체 {pct(r['fe'])}(t {r['t_fe']:+.2f})")
        sel_a = res[c]["sel_all"]
        say(f"  {c}: 정규 {pct(res[c]['P2']['fe'])}(t {res[c]['P2']['t_fe']:+.2f}), " + ", ".join(parts) +
            f" | 고르는 구간 전체 세션 {pct(sel_a['fe'])}(t {sel_a['t_fe']:+.2f})")
        sel_p = ["P1"] if c in INTRA else ["P0", "P1"]
        lab_all = labs_all[c]
        for sc in ["프리", "애프터"]:
            ks = (T.period.isin(sel_p) & (T.ses == sc)).values
            r = split_tests(T[ks].reset_index(drop=True), lab_all[ks], "전체")
            say(f"      고르는 구간({'+'.join(sel_p)}) {sc}: {pct(r['fe'])}(t {r['t_fe']:+.2f}, {r['nA']}/{r['nB']}건)")
        kp = (x2.ses == "프리").values
        y = x2[kp].assign(A=lab[kp]).dropna(subset=["A"])
        m = y.groupby(["day", "A"]).ret.mean().unstack()
        dd = (m[1.0] - m[0.0]).dropna()
        say(f"      P2 프리 날짜별 상위-하위: " + ", ".join(f"{d[5:]} {pct(v)}" for d, v in dd.items()) +
            f" | 합 {pct(dd.sum())}, 같은 방향 가장 큰 2일이 합의 {(dd.nsmallest(2) if dd.sum() < 0 else dd.nlargest(2)).sum() / dd.sum():.0%}, "
            f"합과 같은 방향인 날 {int((np.sign(dd) == np.sign(dd.sum())).sum())}/{len(dd)}일" if len(dd) else "")
    if post:
        kp = ((T.period == "P2") & (T.ses == "프리")).values
        real_pre = {c: split_tests(T[kp].reset_index(drop=True), labs_all[c][kp], "전체")["t_fe"] for c in TRAITS}
        mc = max(real_pre, key=lambda c: abs(real_pre[c]))
        mx = abs(real_pre[mc])
        ks = (T.period.isin(["P1"] if mc in INTRA else ["P0", "P1"]) & (T.ses == "프리")).values
        sp = split_tests(T[ks].reset_index(drop=True), labs_all[mc][ks], "전체")
        say(f"  P2 프리장 날짜 뺀 t, 11개 모두: " + ", ".join(f"{c} {v:+.2f}" for c, v in real_pre.items()))
        say(f"  가장 큰 것은 {mc}(P2 프리 t {real_pre[mc]:+.2f}). 같은 성질의 고르는 구간 프리장: {pct(sp['fe'])}(t {sp['t_fe']:+.2f})")
        fam_pre = np.array([max(abs(F[F.trait == c].pre_t.values[rng.integers(N_PERM)]) for c in TRAITS) for _ in range(N_FAM)])
        say(f"  우연 수준(사후 관찰용): P2 프리장 날짜 뺀 |t| 진짜 11개 최대 {mx:.2f}. 가짜 한 개가 이 이상 {np.mean(np.abs(F.pre_t) >= mx):.1%}, "
            f"가짜 11개 묶음 최대가 이 이상 {np.mean(fam_pre >= mx):.1%} (세션 3개까지 고르면 더 흔하다)")
    say("")
    say("[시험 수] 고르기: 성질 11개 x 상·하 1/3 1구분 x 잣대 2(원값·날짜 뺌) = 22개 t (점수 11개). 보조 표시: P0·P1·전체세션 33개 t.")
    say(f"          확인: P2에서 후보 {len(cands)}개 판정, 11개 모두 값 표시. 가짜 성질 {len(F)}개 x 같은 절차, 11개 묶음 최고 점수 {N_FAM}회.")
    open(os.path.join(RES, "ai_fit_traits.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RAW = sys.argv[1]
    main()
