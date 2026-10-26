"""반박 검증: fam_micro_past.py 후보(early_avg h2, ac1 h2, early_sum h2, early_avg t3)를 처음부터 다시 계산해 깬다(2026-10-09).

기본값은 '살아남지 못함'. 확인 항목:
  [1] 재현: 두 절반 x 전체/정규 차이와 날짜 t를 분석가 숫자와 맞춘다.
  [2] 미래 참조: early 거래(7/14~9/8 매수)의 마지막 매도 시각, 특성 기준일, early_avg 재계산 일치.
  [3] 무작위 특성(종목 라벨 섞기) 2000개: 같은 구분·같은 날짜 t를 돌려 관측값이 몇 번째인지. 날짜 t는 종목 고정 효과
      (같은 종목이 매일 같은 쪽에 있음)를 반영하지 못해 부풀 수 있으므로, 이 순열 분포가 정직한 기준이다.
      + 종목 단위 t(종목별 평균을 표본으로 한 Welch t).
  [4] 애프터 설명: 애프터 거래 제외, 날짜x세션 평균을 뺀 초과, 날짜x세션xvol 삼분위 초과, 사전 애프터 비중(early_aft) 단독,
      early_avg를 사전 early_aft·vol20 순위로 회귀한 잔차로 다시 나누기.
  [5] 변동성 설명: 수익/atr14, vol20 삼분위 안.
  [6] 쏠림: 차이를 가장 줄이는 1·3종목, 1일 빼기.
  [7] 프리장만 남는다는 주장(early_avg t3·h2): 순열 p, 세션 분해 6칸 중 최대를 고른 선택 보정, 종목 단위 t,
      같은 봉 1% 되돌림(얇은 체결) 제외, 보수적 체결가(매수 봉 종가와 매수가 중 높은 값, 사후 설명용), 매도 세션, 잔차 특성.
  [8] ac1 h2 뒤 절반 실패 확인.
특성은 모두 9/9 종가까지 알 수 있다. 같은 봉 되돌림·보수 체결가·매도 세션은 사후 값이라 설명용이다.
시험 수는 결과 끝에 적는다.
저장: ../results/verify_micro_past.txt
사용: py verify_micro_past.py"""
import os, sys, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.append(HERE)

NPERM = 2000
RNG = np.random.default_rng(20261009)
OUT = io.StringIO()
NTEST = {"n": 0}


def p(*a):
    print(*a)
    print(*a, file=OUT)


def pct(x):
    return f"{x * 100:+.2f}%" if pd.notna(x) else "N/A"


def load():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    import stop_lab
    S = stop_lab.load("early")
    e = stop_lab.run(S, None, "early")
    xt = [S[s]["idx"][xi] for s, xi in zip(e.sym, e.xi)]
    e["xt"] = xt
    f["early_aft"] = f.sym.map((e.ses == "애프터").groupby(e.sym).mean())
    f["early_avg_chk"] = f.sym.map(e.groupby("sym").ret.mean())
    t = t[t.sym.isin(f.sym)].merge(f[["sym", "vol20", "atr14"]], on="sym").reset_index(drop=True)
    t["day"] = t.day.astype(str)
    f["volT"] = pd.qcut(f.vol20, 3, labels=[0, 1, 2]).astype(int)
    t["volT"] = t.sym.map(f.set_index("sym").volT)
    t["ret_atr"] = t.ret / t.atr14
    t["adj_s"] = t.ret - t.groupby(["day", "ses"]).ret.transform("mean")
    t["adj_sv"] = t.ret - t.groupby(["day", "ses", "volT"]).ret.transform("mean")
    return t, f, e


def bar_info(t):
    """매수 봉 종가(같은 봉 되돌림), 매도 봉 세션·시가·종가. build.py와 같은 3분봉. 사후 값(설명용)."""
    import build as B
    rec = np.full(len(t), np.nan)
    xses = np.empty(len(t), dtype=object)
    xc = np.full(len(t), np.nan)
    xo = np.full(len(t), np.nan)
    ec = np.full(len(t), np.nan)
    bad = 0
    for s, x in t.groupby("sym"):
        d = B.prep(B.bars(B.RAW, s, 3), 60, 14)
        pos = d.index.get_indexer(pd.to_datetime(x.t.values, utc=True).tz_convert(d.index.tz))
        xi = pos + x.bars.values
        ok = (pos >= 0) & (xi < len(d))
        bad += int((~ok).sum())
        c, o, ses = d.c.values, d.o.values, d.ses.values
        ii = x.index.values[ok]
        ec[ii] = c[pos[ok]]
        rec[ii] = c[pos[ok]] / x.ent.values[ok] - 1
        xses[ii] = ses[xi[ok]]
        xo[ii] = o[xi[ok]]
        xc[ii] = c[xi[ok]]
    t = t.copy()
    t["rec"], t["xses"], t["xo"], t["xc"], t["ec"] = rec, xses, xo, xc, ec
    t["px_err"] = (t.xo / t.px - 1).abs()
    # 보수 체결: 매수가 = max(매수가, 매수 봉 종가), 매도가 = min(매도 봉 시가, 매도 봉 종가) — 꼬리 체결을 모두 불리하게
    t["ret_cons"] = np.minimum(t.xo, t.xc) / np.maximum(t.ent, t.ec) - 1
    return t, bad


# ---------- 행렬 방식 통계(순열용) ----------
def mats(tt, col, syms):
    S = tt.pivot_table(index="day", columns="sym", values=col, aggfunc="sum").reindex(columns=syms).fillna(0.0)
    C = tt.pivot_table(index="day", columns="sym", values=col, aggfunc="count").reindex(columns=syms).fillna(0.0)
    return S.values, C.values


def tstats(S, C, Ah, Al):
    """Ah/Al: 종목 x P(0/1). 반환 거래 평균 차, 날짜 t, 날짜 수(둘 다 거래 있는 날)."""
    sh, ch, sl, cl = S @ Ah, C @ Ah, S @ Al, C @ Al
    ok = (ch > 0) & (cl > 0)
    with np.errstate(all="ignore"):
        d = np.where(ok, sh / ch - sl / cl, np.nan)
        n = ok.sum(0)
        m = np.nanmean(d, 0)
        sd = np.nanstd(d, 0, ddof=1)
        tv = m / (sd / np.sqrt(n))
        tv = np.where(n >= 3, tv, np.nan)
        pooled = (S.sum(0) @ Ah) / (C.sum(0) @ Ah) - (S.sum(0) @ Al) / (C.sum(0) @ Al)
    return pooled, tv, n


def assign(x, split):
    """x: 종목 순서 특성값(NaN 허용). fam_micro_past.grp와 같은 문턱."""
    v = pd.Series(x)
    z = v.dropna()
    if split == "h2":
        m = z.quantile(0.5)
        hi, lo = (v > m).values, (v <= m).values
    else:
        a, b = z.quantile(1 / 3), z.quantile(2 / 3)
        hi, lo = (v > b).values, (v <= a).values
    return hi.astype(float), lo.astype(float)


def perm_assign(x, split, n):
    Ah, Al = np.zeros((len(x), n)), np.zeros((len(x), n))
    for k in range(n):
        h, l = assign(np.asarray(x)[RNG.permutation(len(x))], split)
        Ah[:, k], Al[:, k] = h, l
    return Ah, Al


def sym_t(tt, hi, lo, syms, col="ret"):
    """종목 단위 Welch t: 그 범위 거래가 있는 종목의 종목별 평균을 표본으로."""
    m = tt.groupby("sym")[col].mean().reindex(syms)
    a, b = m[hi.astype(bool)].dropna(), m[lo.astype(bool)].dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, len(a), len(b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se, len(a), len(b)


def sub(t, half=None, ses=None, excl_aft=False):
    x = t
    if half:
        x = x[x.half == half]
    if ses:
        x = x[x.ses == ses]
    if excl_aft:
        x = x[x.ses != "애프터"]
    return x


def one(t, syms, x, split, half, ses=None, col="ret", excl_aft=False, perm=None, sign=1.0):
    """관측 차·날짜 t·종목 t + (perm이 있으면) 순열 단측 p(앞에서 정한 방향 sign)."""
    NTEST["n"] += 1
    tt = sub(t, half, ses, excl_aft)
    S, C = mats(tt, col, syms)
    h, l = assign(x, split)
    d, tv, n = tstats(S, C, h[:, None], l[:, None])
    st, na, nb = sym_t(tt, h, l, syms, col)
    r = dict(diff=d[0], t=tv[0], nd=int(n[0]), st=st, nsa=na, nsb=nb,
             n_hi=int((tt.sym.map(dict(zip(syms, h))) == 1).sum()), n_lo=int((tt.sym.map(dict(zip(syms, l))) == 1).sum()))
    if perm is not None:
        Ah, Al = perm
        _, tp, _ = tstats(S, C, Ah, Al)
        tp = tp[~np.isnan(tp)]
        r["p"] = (np.sum(tp * sign >= tv[0] * sign) + 1) / (len(tp) + 1)
        r["q95"] = np.nanquantile(np.abs(tp), 0.95)
        r["n2"] = int(np.sum(np.abs(tp[:200]) >= 2))
    return r


def fmt(r, perm=True):
    s = f"차 {pct(r['diff'])} 날짜t {r['t']:+.2f}({r['nd']}일) 종목t {r['st']:+.2f}({r['nsa']}/{r['nsb']}종목) 건수 {r['n_hi']}/{r['n_lo']}"
    if perm and "p" in r:
        s += f" | 순열 p {r['p']:.3f} (무작위 |t| 95% {r['q95']:.2f}, 무작위 200개 중 |t|>=2: {r['n2']}개)"
    return s


def drop_k(tt, hi_set, lo_set, sign, key, k):
    """차이(방향 곱)를 가장 줄이는 key 값을 하나씩 k번 뺀다(탐욕)."""
    x = tt[tt.sym.isin(hi_set | lo_set)].copy()
    gone = []
    for _ in range(k):
        best = (np.inf, None)
        for v in x[key].unique():
            y = x[x[key] != v]
            dd = (y[y.sym.isin(hi_set)].ret.mean() - y[y.sym.isin(lo_set)].ret.mean()) * sign
            if dd < best[0]:
                best = (dd, v)
        gone.append(best[1])
        x = x[x[key] != best[1]]
    return best[0] * sign, gone


CAND = [("early_avg", "h2", 1.0), ("ac1", "h2", 1.0), ("early_sum", "h2", 1.0), ("early_avg", "t3", 1.0)]
ANALYST = {  # 분석가 보고(앞 전체, 뒤 전체, 앞 정규, 뒤 정규): (차, t)
    ("early_avg", "h2"): [(0.0064, 2.88), (0.0099, 9.23), (0.0007, 0.32), (0.0018, 0.32)],
    ("ac1", "h2"): [(0.0026, 0.63), (-0.0001, -0.40), (0.0018, 1.30), (0.0015, 0.31)],
    ("early_sum", "h2"): [(0.0032, 2.00), (0.0073, 5.32), (-0.0016, -0.18), (0.0005, -0.01)],
    ("early_avg", "t3"): [(0.0104, 2.61), (0.0131, 7.00), (-0.0002, -0.14), (-0.0005, -0.37)],
}


def main():
    t, f, e = load()
    syms = list(f.sym)
    F = f.set_index("sym")
    p(f"거래 {len(t)}건 {t.sym.nunique()}종목 {t.day.nunique()}일, 순열 {NPERM}회(종목 라벨 섞기, 시드 20261009)")

    # [2] 미래 참조
    p("\n[2] 미래 참조 점검")
    p(f"  early 거래 {len(e)}건, 매수일 {e.day.min()}~{e.day.max()}, 마지막 매도 시각 {max(e.xt)} (최근 구간 첫 매수일 {t.day.min()})")
    dd = (F.early_avg - F.early_avg_chk).abs().max()
    p(f"  features.csv early_avg vs 재계산 최대 차 {dd:.2e} (early_n 합 {int(F.early_n.sum())} vs 재계산 {len(e)}건)")
    p("  build.py: 일봉 특성은 x.index <= 2026-09-09, 5분봉 ac1·vr12는 9/9 정규장까지. 문턱은 종목 특성 분포(두 절반 같음)")

    # [1] 재현
    p("\n[1] 재현(분석가 숫자와 대조): 차(거래 평균) / 날짜 t")
    for feat, sp, sg in CAND:
        x = F[feat].reindex(syms).values
        got = []
        for (h, s), (ad, at) in zip([("앞", None), ("뒤", None), ("앞", "정규"), ("뒤", "정규")], ANALYST[(feat, sp)]):
            r = one(t, syms, x, sp, h, s)
            got.append(f"{h}/{'정규' if s else '전체'} {pct(r['diff'])} t{r['t']:+.2f} (분석가 {pct(ad)} t{at:+.2f})")
        p(f"  {feat} {sp}: " + " | ".join(got))

    # 순열 준비
    PERM = {}
    for feat, sp, _ in CAND:
        PERM[(feat, sp)] = perm_assign(F[feat].reindex(syms).values, sp, NPERM)

    # [3] 무작위 특성 순열 + 종목 단위 t
    p("\n[3] 무작위 특성 대비(같은 구분, 종목 라벨 섞기 2000회). p = 무작위가 관측 이상(앞에서 정한 방향) 비율")
    for feat, sp, sg in CAND:
        x = F[feat].reindex(syms).values
        p(f"== {feat} {sp}")
        for h in ["앞", "뒤"]:
            for s in [None, "정규"]:
                r = one(t, syms, x, sp, h, s, perm=PERM[(feat, sp)], sign=sg)
                p(f"  {h}/{'정규' if s else '전체'}: " + fmt(r))

    # 분석가 고르기 절차 전체를 무작위 특성 200개에 적용: 7특성 중 최고 점수는 아니고, '한 특성이 앞 점수 >= 관측 & 뒤 t >= 관측'
    p("\n[3b] 고르기 절차 재현(무작위 특성 2000개, 각자 h2·t3 중 앞 점수 |(t전체+t정규)/2| 큰 쪽 선택, 그 방향으로 뒤 전체 t)")
    sc = {}
    for h in ["앞", "뒤"]:
        for s in [None, "정규"]:
            sc[(h, s)] = mats(sub(t, h, s), "ret", syms)
    rand = RNG.standard_normal((len(syms), NPERM))
    res = {}
    for sp in ["h2", "t3"]:
        Ah = np.zeros_like(rand)
        Al = np.zeros_like(rand)
        for k in range(NPERM):
            Ah[:, k], Al[:, k] = assign(rand[:, k], sp)
        res[sp] = {k: tstats(*sc[k], Ah, Al)[1] for k in sc}
    score = {sp: (res[sp][("앞", None)] + res[sp][("앞", "정규")]) / 2 for sp in res}
    pick = np.where(np.abs(score["h2"]) >= np.abs(score["t3"]), "h2", "t3")
    bs = np.where(pick == "h2", score["h2"], score["t3"])
    t2 = np.where(pick == "h2", res["h2"][("뒤", None)], res["t3"][("뒤", None)]) * np.sign(bs)
    t2r = np.where(pick == "h2", res["h2"][("뒤", "정규")], res["t3"][("뒤", "정규")]) * np.sign(bs)
    p(f"  무작위 특성의 앞 점수 |.|: 중앙 {np.nanmedian(np.abs(bs)):.2f}, 95% {np.nanquantile(np.abs(bs), .95):.2f}; 관측 early_avg 1.60은 상위 {np.mean(np.abs(bs) >= 1.60):.1%}")
    p(f"  무작위 특성의 뒤 전체 t(앞 방향): |t|>=2 비율 {np.nanmean(np.abs(t2) >= 2):.1%}, t>=2 {np.nanmean(t2 >= 2):.1%}, 95% {np.nanquantile(t2, .95):+.2f}, 최대 {np.nanmax(t2):+.2f}")
    p(f"  무작위 특성의 뒤 정규 t(앞 방향): t>=1 비율 {np.nanmean(t2r >= 1):.1%}")
    p(f"  앞 점수 >= 1.60 이고 뒤 전체 t >= 9.23: {int(np.sum((np.abs(bs) >= 1.60) & (t2 >= 9.23)))}/{NPERM}; "
      f"앞 점수 >= 0.91 이고 뒤 전체 t >= 2: {int(np.sum((np.abs(bs) >= 0.91) & (t2 >= 2)))}/{NPERM}")
    p("  → 날짜 t는 종목 묶음이 매일 같아 무작위로도 |t|>=2가 자주 나온다(위 비율). 판단은 순열 p와 종목 t로 한다.")

    # [4] 애프터 설명
    p("\n[4] 애프터·세션 구성으로 설명되는가(뒤 절반 중심, 앞도 같이). 순열 p는 같은 구분 2000회")
    F["early_aft"] = F.early_aft.fillna(F.early_aft.median())
    for feat, sp, sg in CAND:
        if feat == "ac1":
            continue
        x = F[feat].reindex(syms).values
        p(f"== {feat} {sp}")
        for h in ["앞", "뒤"]:
            r1 = one(t, syms, x, sp, h, excl_aft=True, perm=PERM[(feat, sp)], sign=sg)
            r2 = one(t, syms, x, sp, h, col="adj_s", perm=PERM[(feat, sp)], sign=sg)
            r3 = one(t, syms, x, sp, h, col="adj_sv", perm=PERM[(feat, sp)], sign=sg)
            p(f"  {h} 애프터 제외(프리+정규): " + fmt(r1))
            p(f"  {h} 날짜x세션 초과: " + fmt(r2))
            p(f"  {h} 날짜x세션xvol 초과: " + fmt(r3))
    # 사전 애프터 비중 단독, 잔차 특성
    rk = F[["early_avg", "early_aft", "vol20"]].rank()
    X = np.column_stack([np.ones(len(rk)), rk.early_aft, rk.vol20])
    b = np.linalg.lstsq(X, rk.early_avg.values, rcond=None)[0]
    F["ea_resid"] = rk.early_avg.values - X @ b
    r2 = 1 - F.ea_resid.var() / rk.early_avg.var()
    p(f"\n  early_avg 순위를 사전 early_aft·vol20 순위로 회귀: 설명력 R2 {r2:.2f}. 잔차로 다시 나눔. early_aft~뒤 실제 애프터 비중 스피어만 "
      f"{F.early_aft.corr(t[t.half == '뒤'].groupby('sym').ses.apply(lambda s: (s == '애프터').mean()).reindex(F.index), method='spearman'):+.2f}")
    for feat in ["early_aft", "ea_resid"]:
        for sp in ["h2", "t3"]:
            x = F[feat].reindex(syms).values
            pa = perm_assign(x, sp, NPERM)
            for h in ["앞", "뒤"]:
                ra = one(t, syms, x, sp, h, perm=pa)
                rr = one(t, syms, x, sp, h, "정규", perm=pa)
                rp = one(t, syms, x, sp, h, "프리", perm=pa)
                p(f"  {feat} {sp} {h}: 전체 {fmt(ra)}")
                p(f"      정규 {fmt(rr, False)} p {rr['p']:.3f} | 프리 {fmt(rp, False)} p {rp['p']:.3f}")

    # [5] 변동성
    p("\n[5] 변동성으로 설명되는가")
    for feat, sp, sg in CAND:
        if feat == "ac1":
            continue
        x = F[feat].reindex(syms).values
        h_, l_ = assign(x, sp)
        p(f"== {feat} {sp}: vol20 평균 상 {F.vol20[h_ == 1].mean():.3f} 하 {F.vol20[l_ == 1].mean():.3f}, 스피어만(특성, vol20) {F[feat].corr(F.vol20, method='spearman'):+.2f}")
        for h in ["앞", "뒤"]:
            r = one(t, syms, x, sp, h, col="ret_atr", perm=PERM[(feat, sp)], sign=sg)
            p(f"  {h} 수익/atr14 전체: 차 {r['diff']:+.3f} 날짜t {r['t']:+.2f} 종목t {r['st']:+.2f} 순열 p {r['p']:.3f}")
            s = []
            for vt in [0, 1, 2]:
                for ses in [None, "정규"]:
                    tt = sub(t, h, ses)
                    tt = tt[tt.volT == vt]
                    a = tt[tt.sym.isin(F.index[h_ == 1])].ret
                    c = tt[tt.sym.isin(F.index[l_ == 1])].ret
                    s.append(f"{'저중고'[vt]}/{'정규' if ses else '전체'} {pct(a.mean() - c.mean())}({len(a)}/{len(c)})")
            p(f"  {h} vol 삼분위 안 상-하: " + " ".join(s))

    # [6] 쏠림
    p("\n[6] 쏠림: 차이(앞에서 정한 방향)를 가장 줄이는 종목·날짜 빼기(탐욕)")
    for feat, sp, sg in CAND:
        x = F[feat].reindex(syms).values
        h_, l_ = assign(x, sp)
        hs, ls = set(F.index[h_ == 1]), set(F.index[l_ == 1])
        for h, ses in [("뒤", None), ("뒤", "프리"), ("앞", "프리")]:
            tt = sub(t, h, ses)
            a1, g1 = drop_k(tt, hs, ls, sg, "sym", 1)
            a3, g3 = drop_k(tt, hs, ls, sg, "sym", 3)
            d1, gd = drop_k(tt, hs, ls, sg, "day", 1)
            d2, gd2 = drop_k(tt, hs, ls, sg, "day", 2)
            p(f"  {feat} {sp} {h}/{ses or '전체'}: 1종목 {pct(a1)}({g1[0]}) 3종목 {pct(a3)}({','.join(g3)}) 1일 {pct(d1)}({gd[0]}) 2일 {pct(d2)}")

    # [7] 프리장 주장
    t2_, bad = bar_info(t)
    p(f"\n[7] 프리장만 남는다는 주장. 봉 재구성 실패 {bad}건, 매도가 재현 오차 > 0.1% {int((t2_.px_err > 0.001).sum())}건")
    aft = t2_[t2_.ses == "애프터"]
    p(f"  (확인) 애프터 승률 {(aft.ret > 0).mean():.0%}, 같은 봉 1% 되돌림 {(aft.rec >= 0.01).mean():.0%}, 보수 체결 평균 {pct(aft.ret_cons.mean())} (원래 {pct(aft.ret.mean())})")
    for ses in ["프리", "정규", "애프터"]:
        z = t2_[t2_.ses == ses]
        p(f"  {ses}: 원래 {pct(z.ret.mean())} 보수 체결 {pct(z.ret_cons.mean())} 되돌림1% {(z.rec >= 0.01).mean():.0%} 매도 세션 "
          + ", ".join(f"{k} {v:.0%}" for k, v in z.xses.value_counts(normalize=True).items()))
    t2_["ret_norec"] = np.where(t2_.rec < 0.01, t2_.ret, np.nan)
    for feat, sp in [("early_avg", "t3"), ("early_avg", "h2")]:
        x = F[feat].reindex(syms).values
        pa = PERM[(feat, sp)]
        p(f"== {feat} {sp} 프리장 매수만")
        for h in ["앞", "뒤"]:
            r = one(t2_, syms, x, sp, h, "프리", perm=pa)
            rc = one(t2_, syms, x, sp, h, "프리", col="ret_cons", perm=pa)
            rv = one(t2_, syms, x, sp, h, "프리", col="adj_sv", perm=pa)
            tt = sub(t2_, h, "프리")
            rn = one(tt[tt.rec < 0.01], syms, x, sp, None, perm=pa)
            p(f"  {h} 원래: {fmt(r)}")
            p(f"  {h} 보수 체결(사후): {fmt(rc)}")
            p(f"  {h} 날짜x세션xvol 초과: {fmt(rv)}")
            p(f"  {h} 같은 봉 1% 되돌림 제외(사후): {fmt(rn)}")
            hh, ll = assign(x, sp)
            for xs in ["프리", "정규", "애프터"]:
                q = tt[tt.xses == xs]
                a = q[q.sym.isin(F.index[hh == 1])].ret
                c = q[q.sym.isin(F.index[ll == 1])].ret
                p(f"     매도 세션 {xs}: 상 {pct(a.mean())}({len(a)}) 하 {pct(c.mean())}({len(c)})")
    # 세션 분해 6칸 선택 보정: 무작위 특성마다 max over {h2,t3}x{프리,애프터,정규} of min(t앞, t뒤)(두 방향)
    p("\n  [7b] 세션 분해 선택 보정: 무작위 특성 2000개마다 {h2,t3}x{프리,애프터,정규} 6칸 중 '두 절반 같은 방향 최소 t'의 최대값")
    sm = {}
    for h in ["앞", "뒤"]:
        for s in ["프리", "애프터", "정규"]:
            sm[(h, s)] = mats(sub(t, h, s), "ret", syms)
    best = np.full(NPERM, -np.inf)
    for sp in ["h2", "t3"]:
        Ah = np.zeros_like(rand)
        Al = np.zeros_like(rand)
        for k in range(NPERM):
            Ah[:, k], Al[:, k] = assign(rand[:, k], sp)
        for s in ["프리", "애프터", "정규"]:
            ta = tstats(*sm[("앞", s)], Ah, Al)[1]
            tb = tstats(*sm[("뒤", s)], Ah, Al)[1]
            m = np.fmax(np.fmin(ta, tb), np.fmin(-ta, -tb))
            best = np.fmax(best, m)
    x = F["early_avg"].reindex(syms).values
    obs = {}
    for sp in ["h2", "t3"]:
        for s in ["프리", "애프터", "정규"]:
            ta = one(t, syms, x, sp, "앞", s)["t"]
            tb = one(t, syms, x, sp, "뒤", s)["t"]
            obs[(sp, s)] = max(min(ta, tb), min(-ta, -tb))
    ob = max(obs.values())
    p("  관측 early_avg 6칸: " + ", ".join(f"{k[0]}/{k[1]} {v:+.2f}" for k, v in obs.items()))
    p(f"  관측 최대 {ob:+.2f} 이상인 무작위 특성 {int(np.sum(best >= ob))}/{NPERM} ({np.mean(best >= ob):.1%}); 무작위 최대값 중앙 {np.nanmedian(best):+.2f}, 95% {np.nanquantile(best, .95):+.2f}")
    # 프리장: 잔차 특성(사전 애프터 비중·vol 제거)으로도 남는가
    x = F["ea_resid"].reindex(syms).values
    pa = perm_assign(x, "t3", NPERM)
    for h in ["앞", "뒤"]:
        r = one(t, syms, x, "t3", h, "프리", perm=pa)
        p(f"  잔차 early_avg t3 프리 {h}: {fmt(r)}")
    # 프리장 vol 삼분위 안
    x = F["early_avg"].reindex(syms).values
    hh, ll = assign(x, "t3")
    s = []
    for h in ["앞", "뒤"]:
        for vt in [0, 1, 2]:
            tt = sub(t, h, "프리")
            tt = tt[tt.volT == vt]
            a = tt[tt.sym.isin(F.index[hh == 1])]
            c = tt[tt.sym.isin(F.index[ll == 1])]
            s.append(f"{h}/{'저중고'[vt]} {pct(a.ret.mean())}({len(a)}건/{a.sym.nunique()}종목)-{pct(c.ret.mean())}({len(c)}/{c.sym.nunique()})")
    p("  early_avg t3 프리 vol 삼분위 안 상-하: " + " ".join(s))

    # [7c] 정직한 앞→뒤 고르기(프리장): 분석가가 프리장을 본 7특성(상세 6 + early_pre_avg) 중 앞 절반 프리 t 최고를 뒤에 적용
    F["early_pre_avg"] = F.index.map(e[e.ses == "프리"].groupby("sym").ret.mean())
    cands = [("early_avg", "h2"), ("ac1", "h2"), ("early_sum", "h2"), ("early_avg", "t3"), ("early_sum", "t3"),
             ("early_reg_avg", "t3"), ("early_pre_avg", "t3")]
    F["early_reg_avg"] = F.index.map(e[e.ses == "정규"].groupby("sym").ret.mean())
    p("\n  [7c] 프리장 정직한 앞→뒤 고르기(7후보, 앞 프리 날짜 t 최고를 뒤에 그대로):")
    rows = []
    for feat, sp in cands:
        x = F[feat].reindex(syms).values
        pa = perm_assign(x, sp, NPERM)
        ra = one(t, syms, x, sp, "앞", "프리", perm=pa)
        rb = one(t, syms, x, sp, "뒤", "프리", perm=pa, sign=np.sign(ra["t"]))
        rows.append((feat, sp, ra, rb))
        p(f"    {feat} {sp}: 앞 {pct(ra['diff'])} t{ra['t']:+.2f} p {ra['p']:.3f} → 뒤 {pct(rb['diff'])} t{rb['t']:+.2f} 종목t {rb['st']:+.2f} p {rb['p']:.3f}")
    top = max(rows, key=lambda r: r[2]["t"])
    p(f"    앞 최고 = {top[0]} {top[1]}(t {top[2]['t']:+.2f}) → 뒤 t {top[3]['t']:+.2f}, 순열 p {top[3]['p']:.3f}")
    # 두 절반 합친 프리장 종목 단위 t(early_avg t3)
    x = F["early_avg"].reindex(syms).values
    r = one(t, syms, x, "t3", None, "프리", perm=PERM[("early_avg", "t3")])
    p(f"  early_avg t3 프리 두 절반 합: {fmt(r)}")

    p(f"\n시험 수(이 검증): one() 호출 {NTEST['n']}회(재현 포함) + 무작위 특성 순열 {NPERM}회 x 여러 조합. 분석가 51조합은 따로.")
    with open(os.path.join(RES, "verify_micro_past.txt"), "w", encoding="utf-8") as fh:
        fh.write(OUT.getvalue())


if __name__ == "__main__":
    main()
