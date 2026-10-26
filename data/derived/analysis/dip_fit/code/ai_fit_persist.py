"""이격매수매도 AI 종목 궁합 4부 — '종목 궁합이 다음 구간에도 이어지는가' 정면 검정(2026-10-23).

사용자 질문: AI 종목 중 이격매수매도에 더 잘 맞는 종목은? 예: IREN보다 NBIS가 더 잘 맞는 것 같다.
이 갈래가 답하는 것: 한 구간에서 잘 맞던 종목이 다음 구간에도 잘 맞는가(= 미리 고를 수 있는가).

입력: ../results/ai_fit_trades.csv (ai_fit_build.py가 만든 표, 고치지 않는다)
  P0 8/3~9/9 5분봉 근사 / P1 9/10~10/8 3분봉 / P2 10/9~10/22 3분봉(새 구간)
표본 3가지(주 잣대 = 정규장만):
  정규    정규장 매수만
  전체    프리·정규·애프터 전부
  꼬리뺀  전체에서 프리·애프터 '1분 꼬리 체결'을 뺀 것. 꼬리 표시가 없는 거래(P0 전부, P1 9/27까지)는
          5분봉 대리 규칙으로 채운다([0]에서 1분봉이 있는 9/28~10/22로 대리 규칙의 맞춤률을 잰다).
수익 3가지:
  ret  거래 수익 그대로
  e    같은 날(전체·꼬리뺀은 같은 날·같은 세션) 전체 거래 평균을 뺀 값 = 날짜 운을 뺀 종목 몫
  e2   같은 날(·세션)·같은 대분류(watchlist 대분류 12개) 평균을 뺀 값 = 업종 동반 움직임까지 뺀 종목 몫
검정:
  [1] 앞→뒤 예측 (a) P0→P1 (b) P1→P2 (c) P0+P1→P2, 진단용 (d) P0→P2. 종목 평균의 스피어만 순위 상관 +
      종목 라벨 섞기 순열 p(한쪽, 5,000회) + 같은 대분류 안에서만 섞는 순열 p(업종 순위 효과를 빼고도 남나).
      앞 구간 상위 1/3 vs 하위 1/3 종목의 뒤 구간 거래당 평균 차이(날짜 군집 t). 앞 구간 최소 건수 1·3·5·8건, 3건&뒤 3건.
      정규 앞≥3·뒤≥1은 뒤 구간 하루씩 빼기(날짜 잭나이프)로 ρ·t의 최솟값을 본다.
      영속 종목 분산 추정: 앞·뒤 종목 평균의 공분산(잡음은 구간끼리 독립이라 공분산에는 참 종목 효과만 남는다).
      대분류 수준 순위 상관(12개).
  [2] 구간 안 종목 효과: 같은 날(·세션) 안에서 종목 라벨을 섞는 순열(2,000회). 단위는 종목x날짜(x세션) 묶음.
      변동성 큰 종목이 거래도 많아 그냥 SS는 섞기 쪽이 부풀므로(이분산), 종목마다 자기 분산으로 나눈 표준화 SS를 주 통계로 쓴다.
      보조: 같은 구간 안 홀수 거래일 vs 짝수 거래일 종목 평균 순위 상관(이분산에 영향 없음).
  [3] 검정력 모의실험: 실제 거래(날짜·세션·종목별 건수)를 그대로 두고 같은 날·세션 안에서 종목 라벨을 섞어 종목 효과를 지운 뒤,
      참 종목 효과(표준편차 σ, 모든 구간에서 같다 = 완전 지속)를 더해 [1]을 다시 잰다. σ마다 400회.
  [4] 앞 구간 상위 1/3 종목이 뒤 구간에도 상위 1/3에 남는 비율, 상위 10 겹침, 두 종목 순서 유지 비율, NBIS·IREN 순위.
  [5] 설명(사후): 구간 시작 전 60거래일 QQQ 베타(사전 값)와 구간별 종목 성적의 관계. 시장 방향은 사후 값이라 설명용.
사후 값: 꼬리 대리 규칙은 매수 봉 자체 값(매수 봉 종가)을 쓰므로 매매 신호가 아니라 자료 품질 표시로만 쓴다.
저장: ../results/ai_fit_persist.txt
사용: py ai_fit_persist.py [raw 폴더(1m·5m·1d parquet)]"""
import io, os, sys
from datetime import date
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ai_fit_build as B  # sim·tail_flag·dead_times·bars·prep·hot_days (같은 규칙 재사용, 파일은 고치지 않는다)

RES = os.path.join(HERE, "../results")
RAW = sys.argv[1] if len(sys.argv) > 1 else ("C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/"
                                              "4b7d1886-4bb9-4d2c-b134-8841c35220ef/scratchpad/dip/raw_1023")
RNG = np.random.default_rng(20261023)
NPERM_RHO, NPERM_SS, NSIM = 5000, 2000, 400
SIGMAS = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
VARS = ["정규", "전체", "꼬리뺀"]
COLS = ["ret", "e", "e2"]
PSTART = {"P0": "2026-08-03", "P1": "2026-09-10", "P2": "2026-10-09"}
PEND = {"P0": "2026-09-09", "P1": "2026-10-08", "P2": "2026-10-22"}
OUT = io.StringIO()
NT = {"spearman": 0, "tercile_t": 0, "sector_rho": 0, "within": 0, "split": 0, "retain": 0, "beta": 0}


def p(*a):
    print(*a)
    print(*a, file=OUT)


def pc(x, d=2):
    return f"{x * 100:+.{d}f}%" if pd.notna(x) else "N/A"


# ---------------------------------------------------------------- [0] 꼬리 대리 규칙
def proxy_cols(sym, ts):
    """5분봉에서 매수 시각이 든 봉: 직전 5분 종가(pc5), 그 봉 종가(bc5)."""
    b = pd.read_parquet(f"{RAW}/5m_{sym}.parquet")
    idx = b.index.as_unit("ns").asi8
    c = b.Close.values
    k = np.searchsorted(idx, ts, "right") - 1
    pc5 = np.where(k >= 1, c[np.maximum(k - 1, 0)], np.nan)
    bc5 = np.where(k >= 0, c[np.maximum(k, 0)], np.nan)
    return pc5, bc5


RULES = {"A 직전 5분 종가보다 1% 넘게 아래": lambda e, a, b: e < a * 0.99,
         "B A + 그 봉 종가보다도 1% 넘게 아래(되돌림)": lambda e, a, b: (e < a * 0.99) & (e < b * 0.99),
         "C 직전 5분 종가보다 0.5% 넘게 아래": lambda e, a, b: e < a * 0.995}


def tail_validation(T):
    """대리 규칙 맞춤률: (가) 5분봉 근사 전략을 9/28~10/22에 돌린 프리·애프터 거래(1분봉 꼬리 표시와 비교)
    (나) 꼬리 표시가 있는 3분봉 거래(P1 9/28~·P2)."""
    rows = []
    for s in sorted(T.sym.unique()):
        if not os.path.exists(f"{RAW}/1m_{s}.parquet"):
            continue
        d = B.prep(B.bars(RAW, s, 5), 36, 9)
        hot = B.hot_days(RAW, s, d, 0.07)
        m1 = pd.read_parquet(f"{RAW}/1m_{s}.parquet")
        for ei, xi, ent, px, why in B.sim(d, B.dead_times(d), hot):
            dy = d.date.iloc[ei]
            if not (date(2026, 9, 28) <= dy <= date(2026, 10, 22)) or d.ses.iloc[ei] == "정규":
                continue
            tf = B.tail_flag(m1, d.index[ei].as_unit("ns").value, ent, 5)
            rows.append(dict(sym=s, tail=tf, ent=ent, pc5=d.c.values[ei - 1], bc5=d.c.values[ei], ret=px / ent - 1))
    V5 = pd.DataFrame(rows).dropna(subset=["tail"])
    K = T[(T.ses != "정규") & T["tail"].notna()].copy()
    K["pc5"], K["bc5"] = np.nan, np.nan
    for s, g in K.groupby("sym"):
        a, b = proxy_cols(s, g.t_ns.values)
        K.loc[g.index, "pc5"], K.loc[g.index, "bc5"] = a, b
    p("[0] 꼬리 대리 규칙 맞춤률 (꼬리 = 프리·애프터 매수가 체결 분의 직전 1분 종가보다 1% 넘게 아래)")
    best, best_acc = None, -1
    for name, f in RULES.items():
        out, a5 = [], np.nan
        for lab, V in [("5분봉 전략 9/28~10/22", V5), ("3분봉 거래 P1 9/28~·P2", K)]:
            pr = f(V.ent.values, V.pc5.values, V.bc5.values)
            tl = V["tail"].values == 1
            acc = (pr == tl).mean()
            sens, spec = pr[tl].mean(), (~pr[~tl]).mean()
            out.append(f"{lab}: {len(V)}건 꼬리 {tl.mean():.0%} | 맞춤 {acc:.0%} 꼬리 잡음 {sens:.0%} 비꼬리 맞춤 {spec:.0%}"
                       f" | 대리로 뺀 뒤 평균 {pc(V[~pr].ret.mean())} (1분 기준 {pc(V[~tl].ret.mean())})")
            if lab.startswith("5분"):
                a5 = (sens + spec) / 2
        p(f"  {name}")
        for o in out:
            p(f"    {o}")
        if a5 > best_acc:
            best, best_acc = name, a5
    p(f"  → 고른 규칙(5분봉 전략 기준 균형 맞춤률 최고): {best} ({best_acc:.0%})")
    return RULES[best]


def add_tail2(T, rule):
    T["tail2"] = np.where(T.ses == "정규", 0.0, T["tail"])
    need = T.tail2.isna()
    for s, g in T[need].groupby("sym"):
        a, b = proxy_cols(s, g.t_ns.values)
        T.loc[g.index, "tail2"] = rule(g.ent.values, a, b).astype(float)
    p("  꼬리(1분 표시 또는 대리) 비율, 프리·애프터 중:")
    pa = T[T.ses != "정규"]
    for per, g in pa.groupby("period"):
        known = g["tail"].notna()
        p(f"    {per}: {len(g)}건, 1분 표시 {known.sum()}건, 대리 {(~known).sum()}건 → 꼬리 {g.tail2.mean():.0%}"
          f" (꼬리 평균 {pc(g[g.tail2 == 1].ret.mean())}, 비꼬리 평균 {pc(g[g.tail2 == 0].ret.mean())})")
    return T


# ---------------------------------------------------------------- 공통 계산
def sample(T, v):
    if v == "정규":
        X = T[T.ses == "정규"].copy()
        X["cell"] = X.day
    elif v == "전체":
        X = T.copy()
        X["cell"] = X.day + "|" + X.ses
    else:
        X = T[(T.ses == "정규") | (T.tail2 == 0)].copy()
        X["cell"] = X.day + "|" + X.ses
    X["e"] = X.ret - X.groupby("cell").ret.transform("mean")
    X["e2"] = X.ret - X.groupby(["cell", "big"]).ret.transform("mean")
    return X.reset_index(drop=True)


def cl_t(y, x, g):
    """y = a + b·x, 날짜 군집 표준오차(CR1). 반환 b, t, p(df=G-1), G."""
    X = np.column_stack([np.ones(len(y)), x])
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    codes = pd.factorize(g)[0]
    G = codes.max() + 1
    S = np.zeros((G, 2))
    np.add.at(S, codes, X * u[:, None])
    N = len(y)
    V = XtX @ (S.T @ S) @ XtX * G / (G - 1) * (N - 1) / (N - 2)
    t = b[1] / np.sqrt(V[1, 1]) if V[1, 1] > 0 else np.nan
    return b[1], t, 2 * stats.t.sf(abs(t), G - 1), G


def rank_z(a):
    r = stats.rankdata(a)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else r * 0.0


def spear_perm(x, y, groups=None, n=NPERM_RHO):
    """한쪽 순열 p. groups가 있으면 같은 그룹 안에서만 섞는다."""
    rx, ry = rank_z(x), rank_z(y)
    rho = float((rx * ry).mean())
    M = np.tile(ry, (n, 1))
    if groups is None:
        M = RNG.permuted(M, axis=1)
    else:
        for gv in np.unique(groups):
            ix = np.flatnonzero(groups == gv)
            if len(ix) > 1:
                M[:, ix] = RNG.permuted(M[:, ix], axis=1)
    null = M @ rx / len(rx)
    return rho, (1 + (null >= rho - 1e-12).sum()) / (n + 1)


PAIRS = {"(a) P0→P1": (["P0"], "P1"), "(b) P1→P2": (["P1"], "P2"), "(c) P0+P1→P2": (["P0", "P1"], "P2"),
         "(d) P0→P2": (["P0"], "P2")}
MAIN_PAIRS = ["(a) P0→P1", "(b) P1→P2", "(c) P0+P1→P2"]
KMINS = [(1, 1), (3, 1), (5, 1), (8, 1), (3, 3)]


def sym_means(X, pers, col):
    g = X[X.period.isin(pers)].groupby("sym")[col]
    return g.mean(), g.size()


def front_back(X, pair, col, kf, kb, drop_day=None):
    fr, bk = PAIRS[pair]
    mf, nf = sym_means(X, fr, col)
    Xb = X if drop_day is None else X[X.day != drop_day]
    mb, nb = sym_means(Xb, [bk], col)
    ok = [s for s in mf.index if nf[s] >= kf and s in mb.index and nb[s] >= kb]
    return mf[ok], mb[ok], nf[ok], nb[ok]


def tercile_diff(X, pair, col, mf, drop_day=None):
    """앞 구간 순위 상위 1/3 vs 하위 1/3 종목의 뒤 구간 거래당 평균 차이(날짜 군집 t)."""
    bk = PAIRS[pair][1]
    r = mf.rank(method="first")
    n = len(r)
    top = set(r[r > n * 2 / 3].index)
    bot = set(r[r <= n / 3].index)
    Y = X[(X.period == bk) & X.sym.isin(top | bot)]
    if drop_day is not None:
        Y = Y[Y.day != drop_day]
    x = Y.sym.isin(top).values.astype(float)
    return cl_t(Y[col].values, x, Y.day.values), len(Y)


def persist_cov(mf, mb, nf, nb, nboot=2000):
    """영속 종목 분산 = 앞·뒤 종목 평균의 (건수 가중) 공분산. 종목 재표본 부트스트랩 90% 구간."""
    w = np.sqrt(nf.values * nb.values).astype(float)
    a, b = mf.values, mb.values

    def cov(ix):
        ww = w[ix] / w[ix].sum()
        return np.sum(ww * (a[ix] - np.sum(ww * a[ix])) * (b[ix] - np.sum(ww * b[ix])))
    c0 = cov(np.arange(len(a)))
    bs = np.array([cov(RNG.integers(0, len(a), len(a))) for _ in range(nboot)])
    sd = lambda c: np.sign(c) * np.sqrt(abs(c))
    return sd(c0), sd(np.quantile(bs, 0.05)), sd(np.quantile(bs, 0.95))


# ---------------------------------------------------------------- [1]
def part1(S, BIG):
    p("\n[1] 앞→뒤 예측: 앞 구간 종목 평균으로 뒤 구간 종목 평균을 맞히나")
    p("  ρ = 스피어만 순위 상관, p = 종목 라벨 섞기 5,000회 한쪽(ρ가 이만큼 높게 나올 확률), p업종내 = 같은 대분류 안에서만 섞은 p.")
    p("  상·하 = 앞 상위 1/3 − 하위 1/3 종목의 뒤 구간 거래당 차이, t·p는 날짜 군집(자유도 날짜−1). (d)는 (c)를 풀어 보는 진단용.")
    rows = []
    for pair in PAIRS:
        p(f"\n  {pair}")
        p(f"  {'표본':<6}{'수익':<4}{'앞≥':>4}{'뒤≥':>4}{'종목':>5}  {'ρ':>6} {'p':>6} {'p업종내':>7}   {'상·하 차이':>9} {'t':>6} {'p':>6} {'날짜':>4} {'뒤 건수':>6}")
        for v in VARS:
            X = S[v]
            for col in COLS:
                for kf, kb in KMINS:
                    mf, mb, nf, nb = front_back(X, pair, col, kf, kb)
                    if len(mf) < 9:
                        continue
                    rho, pp = spear_perm(mf.values, mb.values)
                    _, pw = spear_perm(mf.values, mb.values, groups=np.array([BIG[s] for s in mf.index]))
                    (d, t, tp, G), ny = tercile_diff(X, pair, col, mf)
                    NT["spearman"] += 1
                    NT["tercile_t"] += 1
                    row = dict(pair=pair, v=v, col=col, kf=kf, kb=kb, n=len(mf), rho=rho, p=pp, pw=pw, d=d, t=t, tp=tp, G=G)
                    if v == "정규" and kf == 3 and kb == 1:   # 날짜 잭나이프
                        jr, jt = [], []
                        for dd in sorted(X[X.period == PAIRS[pair][1]].day.unique()):
                            a, b, _, _ = front_back(X, pair, col, 3, 1, drop_day=dd)
                            jr.append(np.mean(rank_z(a.values) * rank_z(b.values)))
                            jt.append(tercile_diff(X, pair, col, a, drop_day=dd)[0][1])
                        row.update(jr_min=min(jr), jr_max=max(jr), jt_min=min(jt), jt_max=max(jt),
                                   jr_day=sorted(X[X.period == PAIRS[pair][1]].day.unique())[int(np.argmin(jr))])
                    rows.append(row)
                    p(f"  {v:<6}{col:<4}{kf:>4}{kb:>4}{len(mf):>5}  {rho:+.3f} {pp:6.3f} {pw:7.3f}   {pc(d):>9} {t:+6.2f} {tp:6.3f} {G:>4} {ny:>6}")
    R = pd.DataFrame(rows)
    M = R[R.pair.isin(MAIN_PAIRS)]
    p(f"\n  요약(본 검정 3쌍): {len(M)}조합 중 ρ>0 {(M.rho > 0).sum()}개, 순열 p<0.05 {(M.p < 0.05).sum()}개(우연 기대 {len(M) * 0.05:.1f}개),"
      f" 업종 안 섞기 p<0.05 {(M.pw < 0.05).sum()}개, 상·하 t>2 {(M.t > 2).sum()}개, t<−2 {(M.t < -2).sum()}개")
    for v in VARS:
        for pair in PAIRS:
            for col in COLS:
                r = R[(R.v == v) & (R.pair == pair) & (R.col == col)]
                p(f"    {v:<6}{pair:<14}{col:<4} ρ {r.rho.min():+.2f}~{r.rho.max():+.2f}(중앙 {r.rho.median():+.2f}, p<0.05 {(r.p < 0.05).sum()}/{len(r)},"
                  f" 업종내 p<0.05 {(r.pw < 0.05).sum()}/{len(r)}) | 상·하 차이 중앙 {pc(r.d.median())}, t {r.t.min():+.1f}~{r.t.max():+.1f}")
    p("\n  날짜 잭나이프(정규, 앞≥3·뒤≥1): 뒤 구간 하루씩 빼고 다시 잰 ρ·상·하 t의 범위")
    for _, r in R[(R.v == "정규") & (R.kf == 3) & (R.kb == 1)].iterrows():
        p(f"    {r.pair:<14}{r.col:<4} ρ {r.rho:+.3f} → {r.jr_min:+.3f}~{r.jr_max:+.3f} (가장 낮아지는 날 {r.jr_day}) | t {r.t:+.2f} → {r.jt_min:+.2f}~{r.jt_max:+.2f}")
    p("\n  영속 종목 효과 크기(앞·뒤 종목 평균 공분산의 제곱근, 음수면 −√|공분산|; 앞≥3·뒤≥1; 90% 종목 부트스트랩)")
    for v in VARS:
        for pair in PAIRS:
            txt = []
            for col in ["e", "e2"]:
                mf, mb, nf, nb = front_back(S[v], pair, col, 3, 1)
                c0, lo, hi = persist_cov(mf, mb, nf, nb)
                txt.append(f"{col} {pc(c0)} [{pc(lo)}~{pc(hi)}]")
            p(f"    {v:<6}{pair:<14} " + "  ".join(txt) + f"  종목 {len(mf)}")
    p("\n  대분류(12개) 수준: 대분류 평균 e(종목 평균의 평균, 앞≥3건 종목)의 앞→뒤 순위 상관(순열 5,000회)")
    for v in VARS:
        for pair in PAIRS:
            mf, mb, nf, nb = front_back(S[v], pair, "e", 3, 1)
            gf = mf.groupby(lambda s: BIG[s]).mean()
            gb = mb.groupby(lambda s: BIG[s]).mean()
            rho, pp = spear_perm(gf.values, gb.values)
            NT["sector_rho"] += 1
            best = gf.idxmax(), gf.idxmin()
            p(f"    {v:<6}{pair:<14} ρ {rho:+.2f} (p {pp:.3f}) 앞 1위 {best[0]}→뒤 {int(gb.rank(ascending=False)[best[0]])}위,"
              f" 앞 꼴찌 {best[1]}→뒤 {int(gb.rank(ascending=False)[best[1]])}위 / {len(gf)}")
    return R


# ---------------------------------------------------------------- [2]
def part2(S):
    p("\n[2] 구간 안 종목 효과(날짜 운을 뺀 e): 같은 날(·세션) 안에서 종목 라벨을 섞어도 이만큼 종목 차이가 나나 (2,000회)")
    p("  단위 = 종목x날짜x세션 묶음의 평균 e. 그냥 SS = Σ 묶음 수x(종목 평균)². 표준화 SS = Σ 묶음 수x(종목 평균)² / (그 종목 묶음 제곱 평균).")
    p("  그냥 SS는 '변동 큰 종목이 거래도 많다'(이분산) 때문에 섞기 쪽이 부풀어 p가 1 쪽으로 치우친다 → 표준화 SS가 주 통계.")
    p("  σ̂ = 그냥 SS 기준 섞기 평균을 넘는 몫(대략). 홀짝 = 같은 구간 안 홀수 거래일 vs 짝수 거래일 종목 평균(e, 각 2건 이상) 순위 상관.")
    p(f"  {'표본':<6}{'구간':<4}{'건수':>6}{'종목':>5}{'날짜':>4}  {'그냥 p':>6} {'σ̂':>7}  {'표준화 관측':>9} {'섞기 평균':>8} {'p':>6}   {'홀짝 ρ':>7} {'p':>6} {'종목':>4}")
    for v in VARS:
        for per in ["P0", "P1", "P2"]:
            X = S[v][S[v].period == per]
            U = X.groupby(["cell", "sym"]).agg(n=("e", "size"), s=("e", "sum")).reset_index()
            m = (U.s / U.n).values
            cc = pd.factorize(U.cell)[0]
            sc, syms = pd.factorize(U.sym)
            nS = len(syms)
            ordc = np.argsort(cc, kind="stable")

            def stat(lab):
                k = np.bincount(lab, minlength=nS).astype(float)
                sm = np.bincount(lab, m, nS)
                sq = np.bincount(lab, m * m, nS)
                ok = k > 1
                raw = np.sum(sm[k > 0] ** 2 / k[k > 0])
                std = np.sum((sm[ok] ** 2 / k[ok]) / (sq[ok] / k[ok]))
                return raw, std
            o_raw, o_std = stat(sc)
            nr, ns_ = np.empty(NPERM_SS), np.empty(NPERM_SS)
            for i in range(NPERM_SS):
                order = np.lexsort((RNG.random(len(U)), cc))
                lab = np.empty(len(U), int)
                lab[ordc] = sc[order]
                nr[i], ns_[i] = stat(lab)
            p_raw = (1 + (nr >= o_raw).sum()) / (NPERM_SS + 1)
            p_std = (1 + (ns_ >= o_std).sum()) / (NPERM_SS + 1)
            sig = np.sqrt(max(0.0, (o_raw - nr.mean()) / len(U)))
            # 홀짝 거래일
            days = sorted(X.day.unique())
            odd = set(days[0::2])
            g = X.assign(h=X.day.isin(odd)).groupby(["sym", "h"]).e.agg(["mean", "size"]).unstack()
            g = g[(g["size"][True] >= 2) & (g["size"][False] >= 2)]
            r_s, p_s = spear_perm(g["mean"][True].values, g["mean"][False].values) if len(g) >= 9 else (np.nan, np.nan)
            NT["within"] += 1
            NT["split"] += 1
            p(f"  {v:<6}{per:<4}{len(X):>6}{nS:>5}{X.day.nunique():>4}  {p_raw:6.3f} {pc(sig):>7}  {o_std:9.1f} {ns_.mean():8.1f} {p_std:6.3f}"
              f"   {r_s:+7.3f} {p_s:6.3f} {len(g):>4}")


# ---------------------------------------------------------------- [3]
def part3(S, R):
    p("\n[3] 검정력: 참 종목 효과(표준편차 σ, 세 구간 모두 같음)가 있었다면 ρ와 상·하 t는 얼마로 나왔어야 하나 (σ마다 400회)")
    p("  잡음 = 실제 거래를 같은 날·세션 안에서 종목 라벨만 섞은 것(날짜 효과·두꺼운 꼬리 유지). 앞≥3·뒤≥1. 검출률 = σ=0 모의 상위 5%를 넘는 비율.")
    p("  관측 이하 비율 = 모의 ρ가 실제 관측 ρ 이하로 나온 비율(5% 미만이면 그 σ는 관측과 맞지 않는다 = 배제).")
    out = []
    for v in VARS:
        X = S[v]
        cell = pd.factorize(X.cell)[0]
        symc, syms = pd.factorize(X.sym)
        nS = len(syms)
        perc = X.period.values
        ret = X.ret.values
        days = X.day.values
        ordc = np.argsort(cell, kind="stable")
        ncell = np.bincount(cell)
        for pair in MAIN_PAIRS:
            fr, bk = PAIRS[pair]
            mf_ = np.isin(perc, fr)
            mb_ = perc == bk
            for col in ["ret", "e"]:
                obs = R[(R.v == v) & (R.pair == pair) & (R.col == col) & (R.kf == 3) & (R.kb == 1)].iloc[0]
                res = {}
                for sg in SIGMAS:
                    rhos, ts, ds = np.empty(NSIM), np.empty(NSIM), np.empty(NSIM)
                    for i in range(NSIM):
                        order = np.lexsort((RNG.random(len(X)), cell))
                        lab = np.empty(len(X), int)
                        lab[ordc] = symc[order]
                        y = ret + RNG.normal(0, sg, nS)[lab] if sg > 0 else ret.copy()
                        if col == "e":
                            y = y - (np.bincount(cell, y) / ncell)[cell]
                        nfl = np.bincount(lab[mf_], minlength=nS)
                        nbl = np.bincount(lab[mb_], minlength=nS)
                        sf = np.bincount(lab[mf_], y[mf_], nS) / np.maximum(nfl, 1)
                        sb = np.bincount(lab[mb_], y[mb_], nS) / np.maximum(nbl, 1)
                        okl = (nfl >= 3) & (nbl >= 1)
                        a, b = sf[okl], sb[okl]
                        rhos[i] = np.mean(rank_z(a) * rank_z(b))
                        rk = stats.rankdata(a, method="ordinal")
                        n = len(a)
                        ids = np.flatnonzero(okl)
                        top, bot = ids[rk > n * 2 / 3], ids[rk <= n / 3]
                        sel = mb_ & (np.isin(lab, top) | np.isin(lab, bot))
                        ds[i], ts[i] = cl_t(y[sel], np.isin(lab[sel], top).astype(float), days[sel])[:2]
                    res[sg] = (rhos, ts, ds)
                crit_r = np.quantile(res[0.0][0], 0.95)
                crit_t = np.quantile(res[0.0][1], 0.95)
                p(f"\n  {v} {pair} 수익 {col}: 관측 ρ {obs.rho:+.3f}, 상·하 t {obs.t:+.2f} (σ=0 모의 상위 5% 문턱 ρ {crit_r:+.3f}, t {crit_t:+.2f})")
                p(f"    {'σ':>6} {'ρ 평균':>7} {'ρ 5~95%':>15} {'ρ 검출률':>8} {'t 검출률':>8} {'관측 이하':>8} {'관측 이상':>8} {'상·하 평균':>9}")
                for sg in SIGMAS:
                    rh, tt, dd = res[sg]
                    below, above = (rh <= obs.rho).mean(), (rh >= obs.rho).mean()
                    out.append(dict(v=v, pair=pair, col=col, sigma=sg, rho_mean=rh.mean(), pow_r=(rh > crit_r).mean(),
                                    pow_t=(tt > crit_t).mean(), below=below, above=above, obs=obs.rho, d=dd.mean()))
                    p(f"    {sg * 100:5.2f}% {rh.mean():+7.3f} {np.quantile(rh, 0.05):+6.3f}~{np.quantile(rh, 0.95):+6.3f} "
                      f"{(rh > crit_r).mean():8.0%} {(tt > crit_t).mean():8.0%} {below:8.0%} {above:8.0%} {pc(dd.mean()):>9}")
    return pd.DataFrame(out)


# ---------------------------------------------------------------- [4]
def part4(S):
    p("\n[4] 앞 구간 상위 종목이 뒤 구간에도 상위에 남나 (앞≥3·뒤≥3, 순열 5,000회)")
    p("  상위 1/3 유지 = 앞 상위 1/3 중 뒤에도 상위 1/3인 비율(우연 33%). 순서 유지 = 두 종목 순서가 앞·뒤에서 같은 쌍 비율(우연 50%).")
    out = []
    for v in VARS:
        for pair in PAIRS:
            for col in ["ret", "e"]:
                mf, mb, nf, nb = front_back(S[v], pair, col, 3, 3)
                n = len(mf)
                rf, rb = mf.rank(method="first").values, mb.rank(method="first").values
                topf, topb = rf > n * 2 / 3, rb > n * 2 / 3
                keep = (topf & topb).sum() / topf.sum()
                k10 = len(set(mf.nlargest(10).index) & set(mb.nlargest(10).index))
                tau = stats.kendalltau(mf.values, mb.values).statistic
                null = np.array([(topf & RNG.permutation(topb)).sum() for _ in range(NPERM_RHO)]) / topf.sum()
                pv = (1 + (null >= keep - 1e-12).sum()) / (NPERM_RHO + 1)
                NT["retain"] += 1
                out.append(dict(v=v, pair=pair, col=col, n=n, keep=keep, p=pv, k10=k10, conc=(tau + 1) / 2))
                p(f"  {v:<6}{pair:<14}{col:<4} 종목 {n:>3}  상위 1/3 유지 {keep:5.0%} (p {pv:.3f})  상위 10 겹침 {k10:>2}개(우연 {100 / n:.1f}개)"
                  f"  순서 유지 {(tau + 1) / 2:5.1%}")
    return pd.DataFrame(out)


def nbis_iren(S):
    p("\n[4-2] NBIS·IREN: 구간별 거래당 평균과 순위(그 구간 3건 이상 종목 중 '나보다 높은 종목' 비율; 0%가 1등)")
    p(f"  {'표본':<6}{'구간':<7}{'종목':<5}{'건수':>4} {'ret':>7} {'e':>7} {'ret 순위':>8} {'e 순위':>7} {'종목 수':>6}")
    for v in VARS:
        X = S[v]
        for per in [["P0"], ["P1"], ["P2"], ["P0", "P1"]]:
            Y = X[X.period.isin(per)]
            g = Y.groupby("sym").agg(n=("ret", "size"), ret=("ret", "mean"), e=("e", "mean"))
            g = g[g.n >= 3]
            for s in ["NBIS", "IREN"]:
                if s not in g.index:
                    p(f"  {v:<6}{'+'.join(per):<7}{s:<5} 3건 미만")
                    continue
                r1 = (g.ret > g.ret[s]).sum() / len(g)
                r2 = (g.e > g.e[s]).sum() / len(g)
                p(f"  {v:<6}{'+'.join(per):<7}{s:<5}{g.n[s]:>4} {pc(g.ret[s]):>7} {pc(g.e[s]):>7} {r1:8.0%} {r2:7.0%} {len(g):>6}")
    p("\n  NBIS − IREN 거래당 차이(두 종목 거래만, 날짜 군집 t):")
    for v in VARS:
        X = S[v]
        for per in [["P0"], ["P1"], ["P2"], ["P0", "P1"]]:
            Y = X[X.period.isin(per) & X.sym.isin(["NBIS", "IREN"])]
            txt = []
            for col in ["ret", "e"]:
                d, t, tp, G = cl_t(Y[col].values, (Y.sym == "NBIS").values.astype(float), Y.day.values)
                txt.append(f"{col} {pc(d):>7} (t {t:+.2f})")
            p(f"  {v:<6}{'+'.join(per):<7}" + "  ".join(txt) + f"  날짜 {G}")
    p("\n  앞 구간 차이가 NBIS−IREN 이상인 종목 쌍에서 앞 승자가 뒤에서도 이긴 비율(앞≥3·뒤≥3; 우연 50%):")
    for v in VARS:
        for pair, fr in [("(b) P1→P2", ["P1"]), ("(c) P0+P1→P2", ["P0", "P1"]), ("(a) P0→P1", ["P0"])]:
            for col in ["ret", "e"]:
                mf, mb, nf, nb = front_back(S[v], pair, col, 3, 3)
                Y = S[v][S[v].period.isin(fr)]
                gap = Y[Y.sym == "NBIS"][col].mean() - Y[Y.sym == "IREN"][col].mean()
                a, b = mf.values, mb.values
                m = (a[:, None] - a[None, :]) >= max(gap, 0)
                np.fill_diagonal(m, False)
                win = ((b[:, None] - b[None, :])[m] > 0).mean()
                p(f"  {v:<6}{pair:<14}{col:<4} NBIS−IREN 앞 차이 {pc(gap)} → 그 이상 벌어진 쌍 {m.sum()}개 중 뒤에서도 앞 승자가 이김 {win:.1%}")


def top_list(S):
    p("\n[4-3] P0+P1 정규장 상위·하위 10종목(앞≥5건, 수익 e)과 P2 성적 — '잘 맞는 종목'으로 고를 만한 목록이 다음 구간에 어땠나")
    X = S["정규"]
    a = X[X.period.isin(["P0", "P1"])].groupby("sym").agg(n=("e", "size"), e=("e", "mean"), ret=("ret", "mean"))
    b = X[X.period == "P2"].groupby("sym").agg(n2=("e", "size"), e2=("e", "mean"), ret2=("ret", "mean"))
    a = a[a.n >= 5].join(b, how="left")
    a["rk2"] = a.e2.rank(ascending=False)
    n2 = a.e2.notna().sum()
    for lab, idx in [("상위 10", a.e.nlargest(10).index), ("하위 10", a.e.nsmallest(10).index)]:
        q = a.loc[idx]
        p(f"  {lab}: P0+P1 e 평균 {pc(q.e.mean())} → P2 e 평균 {pc(q.e2.mean())}, ret 평균 {pc(q.ret2.mean())} (P2 거래 {int(q.n2.sum())}건)")
        for s, r in q.iterrows():
            n2s = f"{int(r.n2)}" if pd.notna(r.n2) else "0"
            rks = f"{int(r.rk2)}/{n2}" if pd.notna(r.rk2) else "-"
            p(f"    {s:<6} P0+P1 {int(r.n):>3}건 e {pc(r.e)} ret {pc(r.ret)} | P2 {n2s:>2}건 e {pc(r.e2)} ret {pc(r.ret2)} 순위 {rks}")


# ---------------------------------------------------------------- [5]
def part5(S, syms):
    p("\n[5] 설명(사후): 구간 시작 전 60거래일 QQQ 베타(사전 값)와 구간별 종목 성적(e, 3건 이상)의 순위 상관")
    q = pd.read_parquet(f"{RAW}/1d_QQQ.parquet").Close
    q.index = pd.to_datetime(q.index).tz_localize(None).normalize() if getattr(q.index, "tz", None) else pd.to_datetime(q.index).normalize()
    rq = q.pct_change()
    beta = {}
    for s in syms:
        c = pd.read_parquet(f"{RAW}/1d_{s}.parquet").Close
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize() if getattr(c.index, "tz", None) else pd.to_datetime(c.index).normalize()
        r = c.pct_change()
        for per, st in PSTART.items():
            j = pd.concat([r, rq], axis=1, keys=["s", "q"]).dropna()
            j = j[j.index < pd.Timestamp(st)].tail(60)
            beta[(s, per)] = np.cov(j.s, j.q)[0, 1] / j.q.var() if len(j) >= 40 else np.nan
    for per in PSTART:
        a = q[q.index < pd.Timestamp(PSTART[per])].iloc[-1]
        b = q[q.index <= pd.Timestamp(PEND[per])].iloc[-1]
        p(f"  {per} QQQ 구간 수익 {pc(b / a - 1)} (사후)")
    for v in VARS:
        for col in ["e", "e2"]:
            txt = []
            for per in ["P0", "P1", "P2"]:
                m, n = sym_means(S[v], [per], col)
                m = m[n >= 3]
                bb = np.array([beta.get((s, per), np.nan) for s in m.index])
                ok = ~np.isnan(bb)
                rho, pp = spear_perm(bb[ok], m.values[ok])
                NT["beta"] += 1
                txt.append(f"{per} ρ {rho:+.2f} (한쪽 p {pp:.3f})")
            p(f"  {v:<6}{col:<4} 베타 vs 종목 성적: " + " | ".join(txt))
    p("  베타를 뺀 앞→뒤(정규, e, 앞≥3·뒤≥1): 구간마다 종목 평균 순위를 그 구간 시작 전 베타 순위로 회귀한 잔차끼리 순위 상관")
    for pair in PAIRS:
        fr, bk = PAIRS[pair]
        mf, mb, nf, nb = front_back(S["정규"], pair, "e", 3, 1)
        bf = np.array([np.nanmean([beta.get((s, x), np.nan) for x in fr]) for s in mf.index])
        bb = np.array([beta.get((s, bk), np.nan) for s in mf.index])
        ok = ~(np.isnan(bf) | np.isnan(bb))
        rf, rb = rank_z(mf.values[ok]), rank_z(mb.values[ok])
        zf, zb = rank_z(bf[ok]), rank_z(bb[ok])
        ef = rf - np.polyval(np.polyfit(zf, rf, 1), zf)
        eb = rb - np.polyval(np.polyfit(zb, rb, 1), zb)
        r0, p0 = spear_perm(mf.values[ok], mb.values[ok])
        r1, p1 = spear_perm(ef, eb)
        NT["beta"] += 1
        p(f"    {pair:<14} 그대로 ρ {r0:+.3f} (p {p0:.3f}) → 베타 뺀 뒤 ρ {r1:+.3f} (p {p1:.3f}), 종목 {ok.sum()}")


def main():
    T = pd.read_csv(os.path.join(RES, "ai_fit_trades.csv"))
    T["day"] = T.day.astype(str)
    T["t_ns"] = pd.to_datetime(T.t, utc=True).dt.as_unit("ns").astype("int64").values
    BIG = dict(zip(T.sym, T.big))
    p("이격매수매도 AI 종목 궁합 — 종목 궁합이 이어지는가 (ai_fit_persist.py, 2026-10-23)")
    p(f"입력: ai_fit_trades.csv {len(T)}건 = " + ", ".join(f"{k} {v}건" for k, v in T.groupby("period").size().items()))
    rule = tail_validation(T)
    T = add_tail2(T, rule)
    S = {v: sample(T, v) for v in VARS}
    p("\n표본 크기(구간별 건수 / 종목 / 날짜 / 종목당 중앙 건수 / 거래당 평균 / 거래 표준편차 ret·e):")
    for v, X in S.items():
        for per, g in X.groupby("period"):
            p(f"  {v:<6}{per}: {len(g):>5}건 {g.sym.nunique():>4}종목 {g.day.nunique():>3}일 중앙 {g.groupby('sym').size().median():>4.0f}건 평균 {pc(g.ret.mean())} 표준편차 {g.ret.std() * 100:.2f}%·{g.e.std() * 100:.2f}%")
    R = part1(S, BIG)
    part2(S)
    P = part3(S, R)
    part4(S)
    nbis_iren(S)
    top_list(S)
    part5(S, sorted(T.sym.unique()))
    p("\n[요약] 주 잣대 = 정규장, 앞≥3·뒤≥1")
    for pair in MAIN_PAIRS:
        for col in ["ret", "e"]:
            r = R[(R.v == "정규") & (R.pair == pair) & (R.col == col) & (R.kf == 3) & (R.kb == 1)].iloc[0]
            pw = P[(P.v == "정규") & (P.pair == pair) & (P.col == col)]
            ruled_hi = pw[pw.below < 0.05].sigma.min()
            p50 = pw[pw.pow_r >= 0.5].sigma.min()
            d50 = pw[pw.sigma == 0.005].d.iloc[0]
            pw25 = pw[pw.sigma == 0.0025].pow_r.iloc[0]
            p(f"  {pair:<14}{col:<4} ρ {r.rho:+.3f} (p {r.p:.3f}, 업종내 p {r.pw:.3f}), 상·하 {pc(r.d)} t {r.t:+.2f} |"
              f" σ 0.25%p 검출률 {pw25:.0%}, 검출률 50% 넘는 최소 σ {pc(p50) if pd.notna(p50) else '>2%'},"
              f" 관측보다 커서 배제되는 최소 σ {pc(ruled_hi) if pd.notna(ruled_hi) else '없음'}, σ 0.5%p일 때 기대 상·하 차이 {pc(d50)}")
    n1 = len(R[R.pair.isin(MAIN_PAIRS)])
    p(f"\n시험 수: [1] 스피어만 {NT['spearman']}개(본 검정 3쌍 {n1}개 + 진단 (d) {NT['spearman'] - n1}개; 3표본 x 3수익 x 최소 건수 5단계)"
      f" 각각 순열 p 2종 + 상·하 t {NT['tercile_t']}개, 대분류 순위 상관 {NT['sector_rho']}개, [2] 구간 안 {NT['within']}개 + 홀짝 {NT['split']}개,"
      f" [4] 상위 유지 {NT['retain']}개, [5] 베타 {NT['beta']}개. [3] 검정력 모의 {len(P) // len(SIGMAS)}칸 x σ {len(SIGMAS)}단계 x {NSIM}회(시험 아님).")
    p("꼬리 대리 규칙은 3개 중 1개를 9/28~10/22 맞춤률로 골랐다(종목 순위와 무관한 자료 품질 표시).")
    open(os.path.join(RES, "ai_fit_persist.txt"), "w", encoding="utf-8").write(OUT.getvalue())


if __name__ == "__main__":
    main()
