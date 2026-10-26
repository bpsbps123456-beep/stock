"""이격매수매도 AI 종목 궁합 4부 — 115종목 성적표와 '과장 줄인' 궁합 점수 (2026-10-23 사용자: IREN보다 NBIS가 더 잘 맞는 것 같다).

입력: ../results/ai_fit_trades.csv (ai_fit_build.py가 만든 거래표, P0 8/3~9/9 5분봉 근사 / P1 9/10~10/8 / P2 10/9~10/22 3분봉)
출력: ../results/ai_fit_board.txt (요약·표), ../results/ai_fit_board.csv (115종목 전 항목)

계산
1) 성적표: 종목 × (P0, P1, P2, 합계) 건수·거래당 평균·승률·최악·합·정규장만 평균·RSI 매도 비율·꼬리 체결 비중.
   꼬리 비중 = 꼬리 체결(프리·애프터 1분 꼬리) 건수 / 꼬리를 잴 수 있는 9/28 이후 전체 거래(정규장은 꼬리 아님으로 셈).
2) 날짜 효과 뺀 수익: 거래 수익 − 그날(매수일 ET) 같은 잣대 전체 거래 평균. 정규장 잣대는 정규장 거래끼리, 전체 잣대는 전체 거래끼리.
   종목 평균 m_i와 날짜 군집 분산 v_i(종목의 같은 날 거래를 한 묶음으로).
   v_i는 건수 적은 종목에서 우연히 작게 나오는 것을 막으려고 종목 한 거래 분산을 전 종목 중앙값 쪽으로 4일치 무게만큼 당긴다(분산 수축, d0=4).
   종목 간 참 분산 τ²: 날짜를 두 무더기로 무작위로 나눠(구간별로 반씩) 두 무더기 종목 평균의 공분산을 200번 평균한다.
     같은 날의 업종 동반 충격은 한 무더기에만 들어가므로 참 차이로 잡히지 않는다. 확인용으로 DerSimonian-Laird τ²와 구간 사이 공분산도 낸다.
   수축 점수 θ_i = τ²/(τ²+v_i)·m_i (0 쪽으로 당김: 날짜 효과 뺀 값은 거래 수 가중 평균이 정확히 0).
   날짜 부트스트랩(구간별 층화, 1,000번)으로 θ_i와 순위의 90% 구간을 낸다.
   τ² 점검: 같은 날 안 종목 이름 섞기(참 차이 0) 100번 분포와, 표준편차 0.5%p 효과를 심어 되찾는지 20번.
3) 115종목 동시 검정: 날짜 군집 t(자유도 날짜 수−1, 날짜 3일 이상 종목)와 날짜 부호 뒤집기 p. BH-FDR 10%.
   우연 기대 수: 같은 날 안에서 종목 이름만 섞은 가짜 표 500개에서 같은 검정을 돌린 수.
4) 상위·하위 15(정규장 잣대, 전체 잣대), 대분류·소분류 평균(날짜 효과 뺀 값, 날짜 군집 t).
5) 전체 세션 순위 점검: 프리·애프터 비중·꼬리 비중과의 순위 상관, 날짜×세션 평균을 뺀 판, 9/28 이후 꼬리 체결을 뺀 판.
자료 점검: 원자료 1분봉에 10/7·10/8·10/9·10/12·10/13 애프터 봉이 없어(16:00 한 봉) 8시 매도가 다음 프리장 04:00로 밀린 거래 수와 뺀 평균을 적는다.
사후 값 주의: 이 표의 수익·순위는 모두 거래가 끝난 뒤의 값이다. 설명용이며, '지금 순위 상위를 고르면 다음에도 낫나'는 persist 갈래가 본다.
사용: py ai_fit_board.py
"""
import os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
TR = os.path.join(RES, "ai_fit_trades.csv")
OUT_TXT = os.path.join(RES, "ai_fit_board.txt")
OUT_CSV = os.path.join(RES, "ai_fit_board.csv")
PER = ["P0", "P1", "P2"]
D0 = 4          # 분산 수축 무게(일)
NSPLIT = 200    # τ² 무작위 날짜 나눔 횟수(본 계산)
NBOOT = 1000    # 날짜 부트스트랩
NSPLIT_B = 20   # 부트스트랩 한 번 안의 날짜 나눔 횟수
NPERM = 500     # 종목 이름 섞기
MARK = {"NBIS", "IREN"}
rng = np.random.default_rng(20261023)


def f(x, p=2):
    return "  -  " if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{p}f}"


# ---------- 1) 성적표 ----------
def card(g):
    r = g[g.ses == "정규"]
    tk = g[g.day >= "2026-09-28"]
    return pd.Series(dict(n=len(g), mean=g.ret.mean() if len(g) else np.nan, win=(g.ret > 0).mean() if len(g) else np.nan,
                          worst=g.ret.min() if len(g) else np.nan, sum=g.ret.sum(), reg_n=len(r),
                          reg_mean=r.ret.mean() if len(r) else np.nan,
                          rsi=(g.why == "RSI").mean() if len(g) else np.nan,
                          tail_n=int((g["tail"] == 1).sum()),
                          tail_pct=(tk["tail"] == 1).sum() / len(tk) if len(tk) else np.nan))


# ---------- 2) 날짜 효과·수축 ----------
def mats(df, syms, days):
    si = pd.Index(syms).get_indexer(df.sym)
    di = pd.Index(days).get_indexer(df.day)
    N = np.zeros((len(syms), len(days)))
    S = np.zeros_like(N)
    np.add.at(N, (si, di), 1.0)
    np.add.at(S, (si, di), df.res.values)
    return N, S


def core(N, S, w):
    """가중(날짜 배수) w로 종목 평균 m, 건수 n, 날짜 수 G, 날짜 군집 분산 vo, 수축 분산 v."""
    n = N @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (S @ w) / n
        E = S - N * np.nan_to_num(m)[:, None]
        G = (N > 0).astype(float) @ w
        vo = ((E ** 2) @ w) / n ** 2 * G / (G - 1)
        c = n * vo                                      # 한 거래당 실효 분산
        cbar = np.nanmedian(c[G >= 5])
        cm = (np.where(G > 1, (G - 1) * np.nan_to_num(c), 0) + D0 * cbar) / (np.maximum(G - 1, 0) + D0)
        v = cm / n
    return n, m, G, vo, v


def tau_split(N, S, w, pday, nsplit):
    """날짜(서로 다른 날)를 구간별로 반씩 나눠 두 무더기 종목 평균의 공분산 평균."""
    act = w > 0
    est = []
    for _ in range(nsplit):
        a = np.zeros(len(w))
        for p in PER:
            idx = np.where(act & (pday == p))[0]
            pick = rng.permutation(idx)[: len(idx) // 2]
            a[pick] = 1
        wa, wb = w * a, w * (1 - a) * act
        nA, nB = N @ wa, N @ wb
        ok = (nA > 0) & (nB > 0)
        mA, mB = (S @ wa)[ok] / nA[ok], (S @ wb)[ok] / nB[ok]
        est.append(np.cov(mA, mB)[0, 1])
    return float(np.mean(est))


def tau_dl(m, v):
    ok = np.isfinite(m) & np.isfinite(v) & (v > 0)
    m, v = m[ok], v[ok]
    wt = 1 / v
    mb = (wt * m).sum() / wt.sum()
    Q = (wt * (m - mb) ** 2).sum()
    return max(0.0, (Q - (len(m) - 1)) / (wt.sum() - (wt ** 2).sum() / wt.sum()))


def shrink(m, v, t2):
    """0 쪽으로 당긴다: 날짜 효과 뺀 값은 거래 수로 가중하면 평균이 정확히 0(그날 평균 종목 = 0).
    정밀도 가중 평균 mu_w는 참고로만 돌려준다(저잡음 종목 쪽으로 쏠려 0이 아닐 수 있다)."""
    ok = np.isfinite(m) & np.isfinite(v)
    t2 = max(t2, 0.0)
    wt = 1 / (t2 + v[ok])
    mu_w = (wt * m[ok]).sum() / wt.sum()
    B = t2 / (t2 + v)
    th = B * m
    return th, B, mu_w


def boot_w(pday):
    w = np.zeros(len(pday))
    for p in PER:
        idx = np.where(pday == p)[0]
        np.add.at(w, rng.choice(idx, len(idx)), 1.0)
    return w


def ranks(x):
    """큰 값이 1위. 거래 없는 종목(nan)은 순위 없음."""
    o = pd.Series(x).rank(ascending=False, method="average", na_option="keep").values
    return o


# ---------- 3) 동시 검정 ----------
def ttest(n, m, G, vo):
    with np.errstate(invalid="ignore", divide="ignore"):
        t = m / np.sqrt(vo)
    p = np.where(G >= 3, 2 * stats.t.sf(np.abs(t), np.maximum(G - 1, 1)), np.nan)
    return t, p


def signflip_p(Srow, Nrow, nflip=20000):
    s = Srow[Nrow > 0]
    G = len(s)
    if G < 3:
        return np.nan
    obs = abs(s.sum())
    if 2 ** G <= nflip:
        e = ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2 - 1
    else:
        e = rng.choice([-1, 1], size=(nflip, G))
    return float((np.abs(e @ s) >= obs - 1e-15).mean())


def bh(p, q=0.10):
    ok = np.isfinite(p)
    pv = p[ok]
    o = np.argsort(pv)
    k = len(pv)
    thr = q * np.arange(1, k + 1) / k
    passed = pv[o] <= thr
    cut = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(k, bool)
    rej[o[:cut]] = True
    out = np.zeros(len(p), bool)
    out[np.where(ok)[0]] = rej
    return out


def perm_null(df, syms, days, nperm=NPERM):
    """같은 날 안에서 종목 이름만 섞는다(종목별 날짜당 건수는 그대로)."""
    res = []
    grp = [np.where(df.day.values == d)[0] for d in days]
    symv = df.sym.values.copy()
    for _ in range(nperm):
        s2 = symv.copy()
        for g in grp:
            s2[g] = symv[rng.permutation(g)]
        N, S = mats(df.assign(sym=s2), syms, days)
        n, m, G, vo, v = core(N, S, np.ones(len(days)))
        t, p = ttest(n, m, G, vo)
        res.append(((p < 0.05).sum(), (p < 0.10).sum(), bh(p).sum()))
    return np.array(res)


def cat_stats(df, col):
    rows = []
    for k, g in df.groupby(col):
        s = g.groupby("day").res.agg(["sum", "size"])
        n = s["size"].sum()
        m = s["sum"].sum() / n
        G = len(s)
        e = s["sum"] - s["size"] * m
        se = np.sqrt((e ** 2).sum() / n ** 2 * G / (G - 1)) if G > 1 else np.nan
        rows.append(dict(cat=k, stocks=g.sym.nunique(), n=n, days=G, raw=g.ret.mean(), adj=m, t=m / se if se else np.nan,
                         p=2 * stats.t.sf(abs(m / se), G - 1) if G > 2 and se else np.nan))
    out = pd.DataFrame(rows).sort_values("adj", ascending=False)
    out["bh10"] = bh(out.p.values)
    return out


def analyze(df, syms, label, out):
    """df: 한 잣대(정규장 또는 전체)의 거래. res = ret − 그날 평균."""
    df = df.copy()
    df["res"] = df.ret - df.groupby("day").ret.transform("mean")
    days = np.array(sorted(df.day.unique()))
    pday = df.drop_duplicates("day").set_index("day").period.reindex(days).values
    N, S = mats(df, syms, days)
    w1 = np.ones(len(days))
    n, m, G, vo, v = core(N, S, w1)
    t2s = tau_split(N, S, w1, pday, NSPLIT)
    t2d = tau_dl(m, v)
    # 구간 사이 공분산(종목 효과가 구간을 넘어 이어지는 몫)
    cross = {}
    pm = {}
    for p in PER:
        wp = (pday == p).astype(float)
        np_, mp, *_ = core(N, S, wp)
        pm[p] = (np_, mp)
    for a, b in [("P0", "P1"), ("P1", "P2"), ("P0", "P2")]:
        ok = (pm[a][0] > 0) & (pm[b][0] > 0)
        cross[f"{a}-{b}"] = (np.cov(pm[a][1][ok], pm[b][1][ok])[0, 1], stats.spearmanr(pm[a][1][ok], pm[b][1][ok])[0], ok.sum())
    wa = np.isin(pday, ["P0", "P1"]).astype(float)
    wb = (pday == "P2").astype(float)
    nA, mA, *_ = core(N, S, wa)
    nB, mB, *_ = core(N, S, wb)
    ok = (nA > 0) & (nB > 0)
    cross["P0+P1-P2"] = (np.cov(mA[ok], mB[ok])[0, 1], stats.spearmanr(mA[ok], mB[ok])[0], ok.sum())
    th, B, mu = shrink(m, v, t2s)
    th_dl, B_dl, _ = shrink(m, v, t2d)
    with np.errstate(invalid="ignore"):
        post_sd = np.sqrt(np.maximum(t2s, 0) * v / (np.maximum(t2s, 0) + v))
    # 부트스트랩
    TH, MM, T2 = np.zeros((NBOOT, len(syms))), np.zeros((NBOOT, len(syms))), np.zeros(NBOOT)
    for b in range(NBOOT):
        w = boot_w(pday)
        nb, mb, Gb, vob, vb = core(N, S, w)
        t2b = tau_split(N, S, w, pday, NSPLIT_B)
        thb, _, _ = shrink(mb, vb, t2b)
        TH[b], MM[b], T2[b] = thb, mb, t2b
    RK_TH = np.array([ranks(x) for x in TH])
    RK_M = np.array([ranks(x) for x in MM])
    # 검정
    t, p = ttest(n, m, G, vo)
    psf = np.array([signflip_p(S[i], N[i]) for i in range(len(syms))])
    rej_t, rej_sf = bh(p), bh(psf)
    null = perm_null(df, syms, days)
    rk_now = ranks(th)
    K = int(np.isfinite(th).sum())
    R = pd.DataFrame(dict(sym=syms, n=n, days=G, adj=m, se=np.sqrt(vo), v_mod=np.sqrt(v), B=B, score=th,
                          score_lo=np.nanpercentile(TH, 5, axis=0), score_hi=np.nanpercentile(TH, 95, axis=0),
                          post_lo=th - 1.645 * post_sd, post_hi=th + 1.645 * post_sd,
                          rank=rk_now, rank_lo=np.nanpercentile(RK_TH, 5, axis=0), rank_hi=np.nanpercentile(RK_TH, 95, axis=0),
                          adj_lo=np.nanpercentile(MM, 5, axis=0), adj_hi=np.nanpercentile(MM, 95, axis=0),
                          rawrank=ranks(m), rawrank_lo=np.nanpercentile(RK_M, 5, axis=0), rawrank_hi=np.nanpercentile(RK_M, 95, axis=0),
                          p_top15=np.nanmean(RK_TH <= 15, axis=0), p_bot15=np.nanmean(RK_TH > K - 15, axis=0),
                          rawp_top15=np.nanmean(RK_M <= 15, axis=0),
                          score_dl=th_dl, t=t, p=p, p_sf=psf, fdr_t=rej_t, fdr_sf=rej_sf))
    for pp in PER:
        R[f"adj_{pp}"] = pm[pp][1]
        R[f"n_{pp}"] = pm[pp][0]
    out[label] = dict(K=K, R=R, t2s=t2s, t2d=t2d, T2=T2, cross=cross, mu=mu, null=null, df=df, days=days, TH=TH, MM=MM,
                      cats={c: cat_stats(df, c) for c in ["big", "sub"]})
    return out


def tau_check(df, syms, t2_obs, nrep=100, plant=0.005, nplant=20):
    """τ² 점검: 같은 날 안 종목 이름 섞기(참 차이 0)에서 반나눔 τ² 분포, 그리고 표준편차 0.5%p 종목 효과를 심었을 때 되찾는지."""
    df = df.copy()
    df["res"] = df.ret - df.groupby("day").ret.transform("mean")
    days = np.array(sorted(df.day.unique()))
    pday = df.drop_duplicates("day").set_index("day").period.reindex(days).values
    w1 = np.ones(len(days))
    nul, rec, mse = [], [], []
    for k in range(nrep + nplant):
        d2 = df.assign(sym=df.groupby("day").sym.transform(lambda x: rng.permutation(x.values)))
        if k >= nrep:
            eff = rng.normal(0, plant, len(syms))
            d2["res"] = d2.res + eff[pd.Index(syms).get_indexer(d2.sym)]
        N, S = mats(d2, syms, days)
        n, m, G, vo, v = core(N, S, w1)
        t2 = tau_split(N, S, w1, pday, 50)
        if k < nrep:
            nul.append(t2)
        else:
            rec.append(np.sqrt(max(t2, 0)))
            th = shrink(m, v, t2)[0]
            ok = np.isfinite(th)
            mse.append(np.mean((th[ok] - eff[ok]) ** 2) / np.mean((m[ok] - eff[ok]) ** 2))
    nul = np.array(nul)
    return (f"τ² 점검: 이름 섞기(참 차이 0) {nrep}번의 반나눔 표준편차 95% 분위 {np.sqrt(max(np.percentile(nul, 95), 0)) * 100:.2f}%p, "
            f"관찰값 이상 비율 {(nul >= t2_obs).mean():.2f} | 표준편차 {plant * 100:.1f}%p 심기 {nplant}번: 되찾은 표준편차 평균 {np.mean(rec) * 100:.2f}%p, "
            f"수축 점수 오차제곱 = 원평균의 {np.mean(mse):.0%}")


def light(df, syms, by, nperm=200):
    """세션 구성 점검용 가벼운 판: τ²(반나눔), 동시 검정 수, 우연 기대 수, 수축 점수."""
    df = df.copy()
    df["res"] = df.ret - df.groupby(by).ret.transform("mean")
    days = np.array(sorted(df.day.unique()))
    pday = df.drop_duplicates("day").set_index("day").period.reindex(days).values
    N, S = mats(df, syms, days)
    w1 = np.ones(len(days))
    n, m, G, vo, v = core(N, S, w1)
    t2 = tau_split(N, S, w1, pday, NSPLIT)
    t, p = ttest(n, m, G, vo)
    psf = np.array([signflip_p(S[i], N[i]) for i in range(len(syms))])
    null = perm_null(df, syms, days, nperm)
    return dict(n=n, m=m, t2=t2, p=p, psf=psf, null=null, th=shrink(m, v, t2)[0], days=len(days), trades=len(df))


def lline(name, o):
    nl = o["null"]
    return (f"{name}: {o['trades']}건 {o['days']}일, 참 차이 표준편차 {np.sqrt(max(o['t2'], 0)) * 100:.2f}%p, "
            f"p<0.05 {np.nansum(o['p'] < 0.05)}개(우연 {nl[:, 0].mean():.1f}, 95% {np.percentile(nl[:, 0], 2.5):.0f}~{np.percentile(nl[:, 0], 97.5):.0f}), "
            f"BH 10% 통과 t {bh(o['p']).sum()}개·부호 {bh(o['psf']).sum()}개")


def main():
    D = pd.read_csv(TR)
    D["day"] = D.day.astype(str)
    syms = sorted(D.sym.unique())
    info = D.drop_duplicates("sym").set_index("sym")[["big", "sub"]]
    # 1) 성적표
    board = {}
    for p in PER + ["ALL"]:
        g = D if p == "ALL" else D[D.period == p]
        c = g.groupby("sym").apply(card, include_groups=False).reindex(syms)
        c["n"] = c["n"].fillna(0)
        board[p] = c
    out = {}
    analyze(D[D.ses == "정규"], syms, "reg", out)
    analyze(D, syms, "all", out)

    # CSV
    parts = [info.reindex(syms)]
    for p in PER + ["ALL"]:
        parts.append(board[p].add_prefix(f"{p}_"))
    for lab in ["reg", "all"]:
        parts.append(out[lab]["R"].set_index("sym").reindex(syms).add_prefix(f"{lab}_"))
    rows = pd.concat(parts, axis=1)
    rows.index.name = "sym"
    rows.to_csv(OUT_CSV, encoding="utf-8-sig")

    L = []
    w = L.append
    w("이격매수매도 AI 115종목 성적표와 '과장 줄인' 궁합 점수 (ai_fit_board.py, 2026-10-23)")
    w("거래표: results/ai_fit_trades.csv (P0 8/3~9/9 5분봉 근사, P1 9/10~10/8, P2 10/9~10/22 3분봉). 수익은 거래당, 수수료 전.")
    w("날짜 효과 뺀 값(adj) = 거래 수익 − 그날 같은 잣대 전체 거래 평균. 정규장 잣대(꼬리 없음)가 주 잣대, 전체 세션은 보조.")
    w("수축 점수 = 그날 효과 뺀 종목 평균을 '종목 간 참 차이 / (참 차이 + 표본 잡음)' 비율만큼만 믿는 점수(경험적 베이즈).")
    w("주의: 수익·순위는 모두 거래가 끝난 뒤의 값이다(설명용). 순위가 다음 구간에도 이어지는지는 persist 갈래가 본다.")
    w("")
    w("== 0. 전체 ==")
    for p in PER + ["ALL"]:
        g = D if p == "ALL" else D[D.period == p]
        r = g[g.ses == "정규"]
        w(f"{p:>3}: 전체 {len(g):4d}건 {g.day.nunique():2d}일 거래당 {f(g.ret.mean())}% 승률 {(g.ret > 0).mean():.0%} | "
          f"정규장 {len(r):4d}건 거래당 {f(r.ret.mean())}% 승률 {(r.ret > 0).mean():.0%}")
    # 자료 점검: 원자료 1분봉에 10/7·10/8·10/9·10/12·10/13 애프터 봉이 없어(16:00 한 봉) 8시 매도가 다음 프리장 04:00로 밀린 거래
    sh = D[(D.why == "8시") & (D.t_exit.str[11:16] == "04:00")]
    w(f"자료 점검: 8시 매도가 다음 프리장 04:00 첫 봉으로 밀린 거래 {len(sh)}건(매수일 " + ", ".join(f"{k[5:]} {v}" for k, v in sh.day.value_counts().sort_index().items()) + ")")
    for p in PER:
        g = D[D.period == p]
        r = g[g.ses == "정규"]
        k = ~g.index.isin(sh.index)
        kr = ~r.index.isin(sh.index)
        w(f"  {p}: 밀린 거래 빼면 전체 거래당 {f(g.ret[k].mean())}% ({k.sum()}건), 정규장 {f(r.ret[kr].mean())}% ({kr.sum()}건)")
    w("")
    syms_ = syms
    for lab, name in [("reg", "정규장 잣대"), ("all", "전체 세션 잣대")]:
        o = out[lab]
        R = o["R"]
        w(f"== 1. 종목 간 참 차이 ({name}) ==")
        sd_s = np.sqrt(max(o["t2s"], 0))
        w(f"날짜 무작위 반나눔 공분산 τ² = {o['t2s'] * 1e4:+.3f} (%p²) → 참 차이 표준편차 {sd_s * 100:.2f}%p  "
          f"[부트스트랩 90%: 표준편차 {np.sqrt(max(np.percentile(o['T2'], 5), 0)) * 100:.2f}~{np.sqrt(max(np.percentile(o['T2'], 95), 0)) * 100:.2f}%p, τ²≤0 비율 {(o['T2'] <= 0).mean():.0%}]")
        w(f"확인: DerSimonian-Laird τ² 표준편차 {np.sqrt(o['t2d']) * 100:.2f}%p")
        w(tau_check(o["df"], syms, o["t2s"]))
        for k, (cv, rho, k_n) in o["cross"].items():
            w(f"  구간 사이 {k:>8}: 공분산 {cv * 1e4:+.3f} (%p²) → 이어지는 몫 표준편차 {np.sqrt(max(cv, 0)) * 100:.2f}%p, 순위 상관 {rho:+.2f} ({k_n}종목)")
        se_med = np.nanmedian(R.v_mod)
        w(f"종목 평균의 표본 잡음(수축 분산) 중앙 {se_med * 100:.2f}%p, 건수 중앙 {R.n.median():.0f}건·{R.days.median():.0f}일")
        w(f"믿음 비율 B = τ²/(τ²+잡음): 중앙 {np.nanmedian(R.B):.2f}, 범위 {np.nanmin(R.B):.2f}~{np.nanmax(R.B):.2f}, B≥0.5 종목 {(R.B >= 0.5).sum()}개")
        w(f"수축 점수 범위 {f(R.score.min())}~{f(R.score.max())}% (원래 날짜 효과 뺀 평균 범위 {f(R.adj.min())}~{f(R.adj.max())}%). "
          f"0 쪽으로 당김(참고: 정밀도 가중 평균 {f(o['mu'])}%)")
        w("")
        w(f"== 2. 115종목 동시 검정 ({name}): 날짜 효과 뺀 평균 ≠ 0 ==")
        ok = np.isfinite(R.p)
        nl = o["null"]
        w(f"검정 가능(날짜 3일 이상) {ok.sum()}종목")
        w(f"날짜 군집 t: p<0.05 {(R.p < 0.05).sum()}개, p<0.10 {(R.p < 0.10).sum()}개, BH-FDR 10% 통과 {R.fdr_t.sum()}개")
        w(f"부호 뒤집기: p<0.05 {(R.p_sf < 0.05).sum()}개, BH-FDR 10% 통과 {R.fdr_sf.sum()}개")
        w(f"우연 기대(같은 날 종목 이름 섞기 {NPERM}번, 날짜 군집 t): p<0.05 평균 {nl[:, 0].mean():.1f}개 (95% {np.percentile(nl[:, 0], 2.5):.0f}~{np.percentile(nl[:, 0], 97.5):.0f}), "
          f"p<0.10 평균 {nl[:, 1].mean():.1f}개, BH 통과 평균 {nl[:, 2].mean():.2f}개 (1개 이상 나온 비율 {(nl[:, 2] > 0).mean():.0%})")
        w(f"  관찰 p<0.05 수가 가짜 표 이상인 비율 {(nl[:, 0] >= (R.p < 0.05).sum()).mean():.2f}")
        hit = R[R.fdr_t | R.fdr_sf]
        for _, r in hit.iterrows():
            w(f"  통과: {r.sym} adj {f(r.adj)}% t {r.t:+.2f} p {r.p:.4f} 부호p {r.p_sf:.4f} ({r.n:.0f}건 {r.days:.0f}일)")
        sm = R[ok].sort_values("p").head(8)
        w("  p 작은 순 8개: " + ", ".join(f"{r.sym} {f(r.adj)}% (p {r.p:.3f})" for _, r in sm.iterrows()))
        w("")
    # 표
    for lab, name in [("reg", "정규장 잣대(주)"), ("all", "전체 세션 잣대(보조)")]:
        R = out[lab]["R"].dropna(subset=["score"]).copy()
        R = R.join(info, on="sym")
        b = board["ALL"]
        R["raw"] = (b.reg_mean if lab == "reg" else b["mean"]).reindex(R.sym).values
        R["tailp"] = b.tail_pct.reindex(R.sym).values
        R = R.sort_values("score", ascending=False).reset_index(drop=True)
        R["pos"] = np.arange(1, len(R) + 1)
        w(f"== 3. 상위 15 / 하위 15 ({name}) — 수축 점수 순, ★=NBIS·IREN ==")
        w("순위 종목  소분류            건수  날  원평균  날짜뺀  수축점수 [부트 90%]        순위90%   상위15확률 P0뺀  P1뺀  P2뺀" + ("  꼬리" if lab == "all" else ""))

        def line(r):
            st = "★" if r.sym in MARK else " "
            s = (f"{r.pos:3d} {st}{r.sym:<5} {r['sub'][:12]:<12} {r.n:4.0f} {r.days:3.0f} {f(r.raw):>7} {f(r.adj):>7} {f(r.score):>7} [{f(r.score_lo)}, {f(r.score_hi)}] "
                 f"{r.rank_lo:4.0f}~{r.rank_hi:<4.0f} {r.p_top15:5.0%}  {f(R_adjp(r, 'P0'), 1):>5} {f(R_adjp(r, 'P1'), 1):>5} {f(R_adjp(r, 'P2'), 1):>5}")
            if lab == "all":
                s += f"  {r.tailp:.0%}" if np.isfinite(r.tailp) else "   -"
            return s

        def R_adjp(r, p):
            return r[f"adj_{p}"] if r[f"n_{p}"] > 0 else np.nan

        for _, r in R.head(15).iterrows():
            w(line(r))
        w("  ...")
        for _, r in R.tail(15).iterrows():
            w(line(r))
        for _, r in R[R.sym.isin(MARK) & ((R.pos > 15) & (R.pos <= len(R) - 15))].iterrows():
            w(line(r) + "   (중간)")
        w("")
        out[lab]["sorted"] = R
    # 순위 불확실성
    for lab, name in [("reg", "정규장"), ("all", "전체")]:
        R = out[lab]["sorted"]
        K = out[lab]["K"]
        wd = R.rank_hi - R.rank_lo
        wr = R.rawrank_hi - R.rawrank_lo
        w(f"== 4. 순위 불확실성 ({name}) ==")
        w(f"수축 점수 순위 90% 구간 폭: 중앙 {wd.median():.0f}위, 폭 ≥{K // 2}위(절반 넘게) {(wd >= K // 2).sum()}종목 (순위 매긴 종목 {K}개)")
        w(f"날짜 효과 뺀 원평균 순위 90% 구간 폭: 중앙 {wr.median():.0f}위, 폭 ≥{K // 2}위 {(wr >= K // 2).sum()}종목")
        top = R.head(15)
        w(f"지금 상위 15 중 부트스트랩 90%에서도 늘 상위 30 안(순위 상한 ≤30): {(top.rank_hi <= 30).sum()}개, "
          f"상위 15 확률 ≥50%: {(R.p_top15 >= 0.5).sum()}개, ≥80%: {(R.p_top15 >= 0.8).sum()}개")
        w(f"지금 하위 15 중 늘 하위 30 안(순위 하한 ≥{K - 29}): {(R.tail(15).rank_lo >= K - 29).sum()}개, 하위 15 확률 ≥50%: {(R.p_bot15 >= 0.5).sum()}개")
        w(f"수축 점수 0 위(90% 하한 > 0): {(R.score_lo > 0).sum()}개, 0 아래(상한 < 0): {(R.score_hi < 0).sum()}개")
        nl = out[lab]["null"]
        w(f"주의: 날짜 효과 뺀 원평균의 부트 90%가 0을 안 덮는 종목은 {(R.adj_lo > 0).sum() + (R.adj_hi < 0).sum()}개인데, 같은 뜻의 날짜 군집 t p<0.10은 {(R.p < 0.10).sum()}개"
          f"(우연 기대 {nl[:, 1].mean():.1f}개)다. 부트스트랩은 날짜 수가 적은 종목에서 t분포 보정이 없어 구간이 좁다. 순위 구간도 실제보다 좁게 나왔을 수 있다.")
        rho = stats.spearmanr(R.score, R.score_dl, nan_policy="omit")[0]
        w(f"확인: DL τ²로 수축한 점수와의 순위 상관 {rho:.2f}")
        w("")
    # NBIS vs IREN
    w("== 5. NBIS 대 IREN ==")
    for lab, name in [("reg", "정규장"), ("all", "전체")]:
        o = out[lab]
        R = o["R"].set_index("sym")
        i, j = list(o["R"].sym).index("NBIS"), list(o["R"].sym).index("IREN")
        dTH = o["TH"][:, i] - o["TH"][:, j]
        dM = o["MM"][:, i] - o["MM"][:, j]
        a, b = R.loc["NBIS"], R.loc["IREN"]
        w(f"[{name}] NBIS 날짜뺀 {f(a.adj)}% ({a.n:.0f}건) 수축 {f(a.score)}% 순위 {a['rank']:.0f}위({a.rank_lo:.0f}~{a.rank_hi:.0f}) | "
          f"IREN 날짜뺀 {f(b.adj)}% ({b.n:.0f}건) 수축 {f(b.score)}% 순위 {b['rank']:.0f}위({b.rank_lo:.0f}~{b.rank_hi:.0f})")
        w(f"  날짜뺀 차이 NBIS−IREN {f(a.adj - b.adj)}%p, 부트 90% [{f(np.nanpercentile(dM, 5))}, {f(np.nanpercentile(dM, 95))}], NBIS가 앞선 비율 {np.nanmean(dM > 0):.0%}")
        w(f"  수축 점수 차이 {f(a.score - b.score)}%p, 부트 90% [{f(np.nanpercentile(dTH, 5))}, {f(np.nanpercentile(dTH, 95))}], NBIS가 앞선 비율 {np.nanmean(dTH > 0):.0%}")
        for p in PER:
            w(f"  {p}: NBIS 날짜뺀 {f(R.loc['NBIS', f'adj_{p}'])}% ({R.loc['NBIS', f'n_{p}']:.0f}건) / IREN {f(R.loc['IREN', f'adj_{p}'])}% ({R.loc['IREN', f'n_{p}']:.0f}건)")
        # 둘 다 거래한 날만 짝 비교
        df = o["df"]
        x = df[df.sym.isin(["NBIS", "IREN"])].groupby(["day", "sym"]).res.mean().unstack()
        x = x.dropna()
        if len(x) > 2:
            dd = x.NBIS - x.IREN
            tt = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
            w(f"  둘 다 산 날 {len(x)}일 짝 비교: NBIS−IREN 평균 {f(dd.mean())}%p, t {tt:+.2f}, NBIS가 나은 날 {(dd > 0).mean():.0%}")
    for lab, name in [("reg", "정규장"), ("all", "전체")]:
        bd = out[lab]["df"]
        w(f"  [{name}] 데이터센터 소분류: " + ", ".join(
            f"{k} {f(v.res.mean())}% ({len(v)}건)" for k, v in bd[bd.big == "데이터센터"].groupby("sub")))
    w("")
    # 분류별
    for lab, name in [("reg", "정규장"), ("all", "전체")]:
        for c, cn in [("big", "대분류"), ("sub", "소분류")]:
            C = out[lab]["cats"][c]
            w(f"== 6. {cn} 평균 ({name}) — 날짜 효과 뺀 평균, 날짜 군집 t, BH-FDR 10% 통과 {C.bh10.sum()}/{len(C)} ==")
            for _, r in C.iterrows():
                mk = " ★" if (c == "sub" and r["cat"] in ("네오클라우드", "채굴→AI 전환")) or (c == "big" and r["cat"] == "데이터센터") else ""
                w(f"  {r["cat"][:16]:<16} {r.stocks:3d}종목 {r.n:4d}건 원평균 {f(r.raw):>7}% 날짜뺀 {f(r.adj):>7}% t {r.t:+5.2f}{mk}")
            w("")
    # 세션 구성·꼬리 점검
    w("== 7. 전체 세션 순위는 세션 구성·꼬리 때문인가 ==")
    ext = D.assign(e=D.ses != "정규").groupby("sym").e.mean().reindex(syms).values
    tlp = board["ALL"].tail_pct.reindex(syms).values
    ra_ = out["all"]["R"]
    rg_ = out["reg"]["R"]
    sp = lambda a, b: stats.spearmanr(a, b, nan_policy="omit")[0]
    w(f"종목별 프리·애프터 거래 비중과 전체 세션 수축 점수의 순위 상관 {sp(ra_.score, ext):+.2f}, 꼬리 비중(9/28 이후)과 {sp(ra_.score, tlp):+.2f}, "
      f"정규장 날짜뺀 평균과 {sp(ra_.adj, rg_.adj):+.2f}, 정규장 수축 점수와 {sp(ra_.score, rg_.score):+.2f}")
    top = ra_.sort_values("score", ascending=False).head(15).sym
    ix = np.isin(syms, top)
    w(f"전체 잣대 상위 15의 프리·애프터 비중 평균 {np.nanmean(ext[ix]):.0%}, 꼬리 비중 평균 {np.nanmean(tlp[ix]):.0%} / 나머지 {np.nanmean(ext[~ix]):.0%}, {np.nanmean(tlp[~ix]):.0%}")
    w(f"  전체 잣대 상위 15의 정규장 날짜뺀 평균(종목 평균) {np.nanmean(rg_.adj.values[ix]) * 100:+.2f}% / 나머지 {np.nanmean(rg_.adj.values[~ix]) * 100:+.2f}% "
      f"(정규장 거래가 전체 점수에도 들어 있어 독립 비교가 아니다. 상위 15 중 정규장 날짜 3일 미만 {int((rg_.days.values[ix] < 3).sum())}종목)")
    ds = light(D, syms, ["day", "ses"])
    w(lline("날짜×세션 평균을 뺀 전체 세션(세션 구성 효과 제거)", ds) + f", 원래 전체 점수와 순위 상관 {sp(ds['th'], ra_.score):+.2f}")
    Dx = D.assign(res=D.ret - D.groupby(["day", "ses"]).ret.transform("mean"))
    cs = cat_stats(Dx, "sub").set_index("cat")
    w("  날짜×세션 뺀 소분류: " + ", ".join(f"{k} {f(cs.loc[k, 'adj'])}% (t {cs.loc[k, 't']:+.2f})" for k in ["채굴→AI 전환", "네오클라우드", "미국 빅테크"])
      + f" | NBIS {f(ds['m'][syms.index('NBIS')])}%, IREN {f(ds['m'][syms.index('IREN')])}%")
    W = D[D.day >= "2026-09-28"]
    wa_ = light(W, syms, "day")
    wb_ = light(W[W["tail"] != 1], syms, "day")
    w(lline("9/28~10/22 전체 세션(꼬리 포함)", wa_))
    w(lline("9/28~10/22 전체 세션(꼬리 체결 뺌)", wb_) + f", 꼬리 포함 판과 날짜뺀 평균 순위 상관 {sp(wa_['m'], wb_['m']):+.2f}")
    for s_ in ["NBIS", "IREN"]:
        i = syms.index(s_)
        w(f"  {s_}: 프리·애프터 비중 {ext[i]:.0%}, 꼬리 비중 {tlp[i]:.0%}, 9/28 이후 날짜뺀 꼬리 포함 {f(wa_['m'][i])}% ({wa_['n'][i]:.0f}건) → 꼬리 뺌 {f(wb_['m'][i])}% ({wb_['n'][i]:.0f}건)")
    w("")
    # 전체 성적표
    w("== 8. 115종목 성적표 (거래당 %, 승률, 최악, 합, 정규장만 평균, RSI 매도 비율, 꼬리 비중) ==")
    w("종목  | P0 건·평균·정규 | P1 건·평균·정규 | P2 건·평균·정규 | 합계 건 평균 승률 최악 합 정규건 정규평균 RSI 꼬리 | 수축(정규) 수축(전체)")
    rr = out["reg"]["R"].set_index("sym")
    ra = out["all"]["R"].set_index("sym")
    for s in syms:
        seg = []
        for p in PER:
            c = board[p].loc[s]
            seg.append(f"{c.n:3.0f} {f(c['mean'], 1):>5} {f(c.reg_mean, 1):>5}")
        c = board["ALL"].loc[s]
        st = "★" if s in MARK else " "
        tl = f"{c.tail_pct:4.0%}" if np.isfinite(c.tail_pct) else "  - "
        w(f"{st}{s:<5}| " + " | ".join(seg) + f" | {c.n:3.0f} {f(c['mean'], 2):>6} {c.win:4.0%} {f(c.worst, 1):>6} {f(c['sum'], 0):>5} {c.reg_n:3.0f} {f(c.reg_mean, 2):>6} {c.rsi:4.0%} {tl}"
          f" | {f(rr.loc[s, 'score'], 2):>6} {f(ra.loc[s, 'score'], 2):>6}")
    w("")
    w(f"시험 수: 종목 검정 115×2잣대=230(검정 가능 정규 {np.isfinite(out['reg']['R'].p).sum()}·전체 {np.isfinite(out['all']['R'].p).sum()}), "
      f"분류 검정 (대 {len(out['reg']['cats']['big'])} + 소 {len(out['reg']['cats']['sub'])})×2잣대, τ² 추정 2잣대×(반나눔·DL·구간 사이 4)=12, "
      f"τ² 점검 2잣대×(이름 섞기 100+심기 20), 세션 점검 3판(날짜×세션, 9/28 이후 꼬리 포함·뺌)×115종목, 부트스트랩 {NBOOT}번, 이름 섞기 {NPERM}번(세션 점검은 200번).")
    open(OUT_TXT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[: 200]))


if __name__ == "__main__":
    main()
