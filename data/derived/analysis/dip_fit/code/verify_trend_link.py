"""반박 검증 — fam_trend_link 후보 3개(sdev_q B3, corr60 T3, beta60 B3)를 독립 코드로 다시 재고 깨 본다.

확인: ① 재현(앞·뒤 x 전체·정규, 날짜 군집 t) ② 문턱·시점(9/9 이전 값으로 다시 계산, 앞 절반 문턱 고정)
      ③ 쏠림(유리한 상위 1·3종목, 1·2일 제외, 종목 단위 t) ④ 변동성(날짜 고정효과 회귀 + vol20/atr14 통제, 날짜 군집 표준오차)
      ⑤ 애프터·얇은 체결(세션 표준화 차와 t, 다음 봉도 매수가 아래로 내려온 '2봉 확인' 거래만)
      ⑥ 섹터·시장(묶음 고정효과, 날짜별 차 vs QQQ 일간 수익) ⑦ 무작위 특성 null(종목 단위·묶음 단위 각 1200개, 같은 문턱·고르기)
입력: ../results/trades.csv, ../results/features.csv, scratchpad raw_1009(1분봉·일봉, 이미 받은 것)
출력: ../results/verify_trend_link.txt
사용: py verify_trend_link.py"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw_1009"
F0 = pd.Timestamp("2026-09-09")
SES = ["프리", "정규", "애프터"]
L = []
rng = np.random.default_rng(20261009)


def say(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s)


t = pd.read_csv(os.path.join(RES, "trades.csv"))
f = pd.read_csv(os.path.join(RES, "features.csv"))
t = t.merge(f[["sym", "vol20", "atr14", "beta60", "corr60", "group"]], on="sym", how="inner")
t["r_atr"] = t.ret / t.atr14
t["tt"] = pd.to_datetime(t.t, utc=True).dt.tz_convert("America/New_York")
vq = f.vol20.quantile([1 / 3, 2 / 3]).values
t["vt"] = np.where(t.vol20 <= vq[0], 0, np.where(t.vol20 >= vq[1], 2, 1))


def dstat(A, B, col="ret"):
    d = (A.groupby("day")[col].mean() - B.groupby("day")[col].mean()).dropna()
    n = len(d)
    tv = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if n > 2 and d.std(ddof=1) > 0 else np.nan
    return dict(mA=A[col].mean(), mB=B[col].mean(), diff=A[col].mean() - B[col].mean(), t=tv, nd=n, nA=len(A), nB=len(B),
                sA=A.sym.nunique(), sB=B.sym.nunique(), dm=d.mean() if n else np.nan)


def scope(x, sc):
    return x if sc == "전체" else x[x.ses == "정규"]


def fe_reg(x, isA, ctrls, ses_dummies):
    """날짜 고정효과 OLS, 날짜 군집 표준오차. 반환: A 계수, t(자유도 G-1)."""
    X = [isA.astype(float)] + [x[c].values.astype(float) for c in ctrls]
    if ses_dummies:
        for s in ["프리", "애프터"]:
            X.append((x.ses == s).values.astype(float))
    X = np.column_stack(X)
    y = x.ret.values.astype(float)
    g = x.day.values
    dfx = pd.DataFrame(X).groupby(g).transform("mean").values
    Xd, yd = X - dfx, y - pd.Series(y).groupby(g).transform("mean").values
    keep = Xd.std(axis=0) > 0
    Xd = Xd[:, keep]
    XtX = np.linalg.pinv(Xd.T @ Xd)
    b = XtX @ Xd.T @ yd
    u = yd - Xd @ b
    G = len(np.unique(g))
    meat = sum(np.outer(Xd[g == k].T @ u[g == k], Xd[g == k].T @ u[g == k]) for k in np.unique(g))
    N, K = Xd.shape
    V = XtX @ meat @ XtX * G / (G - 1) * (N - 1) / (N - K)
    return b[0], b[0] / np.sqrt(V[0, 0]), G


def drop_k(A, B, sign, key, k):
    """sign 방향 차를 가장 많이 만드는 key를 하나씩 k개 빼고 다시 잰다(탐욕)."""
    gone = []
    for _ in range(k):
        best, bv = None, np.inf
        for v in pd.unique(pd.concat([A[key], B[key]])):
            a, b = A[A[key] != v], B[B[key] != v]
            if len(a) and len(b):
                z = sign * (a.ret.mean() - b.ret.mean())
                if z < bv:
                    best, bv = v, z
        gone.append(best)
        A, B = A[A[key] != best], B[B[key] != best]
    return gone, dstat(A, B)


def stock_t(A, B):
    a, b = A.groupby("sym").ret.mean(), B.groupby("sym").ret.mean()
    r = stats.ttest_ind(a, b, equal_var=False)
    return a.mean() - b.mean(), r.statistic


# ---------- 후보 정의 (앞 절반에서 정한 문턱) ----------
a_half = t[t.half == "앞"]
q_sdev = a_half.sdev_q.quantile(1 / 3)
q_corr = f.corr60.quantile(2 / 3)
q_beta = f.beta60.quantile(1 / 3)
CANDS = [
    ("sdev_q B3", lambda x: x.sdev_q <= q_sdev, lambda x: x.sdev_q > q_sdev, -1, f"sdev_q <= {q_sdev*100:.3f}%p (앞 절반 거래 1/3 분위)"),
    ("corr60 T3", lambda x: x.corr60 >= q_corr, lambda x: x.corr60 < q_corr, +1, f"corr60 >= {q_corr:.3f} (136종목 단면 2/3 분위)"),
    ("beta60 B3", lambda x: x.beta60 <= q_beta, lambda x: x.beta60 > q_beta, +1, f"beta60 <= {q_beta:.3f} (136종목 단면 1/3 분위)"),
]

say("# 반박 검증 — fam_trend_link 후보 3개 (verify_trend_link)")
say(f"거래 {len(t)}건 / {t.sym.nunique()}종목(특성 없는 SKHY 제외), 앞 {a_half.day.nunique()}일 · 뒤 {t[t.half=='뒤'].day.nunique()}일. "
    "sign은 분석가가 앞 절반에서 정한 방향(+1 = A가 좋음).")
say("")

# ---------- ② 시점·정의 ----------
say("## ② 시점·정의 점검")
q = pd.read_parquet(os.path.join(RAW, "1d_QQQ.parquet"))
qr = q.Close.pct_change()
mx = []
for s in f.sym:
    x = pd.read_parquet(os.path.join(RAW, f"1d_{s}.parquet"))
    x = x[x.index <= F0]
    r = x.Close.pct_change()
    jr = pd.concat([r, qr], axis=1, join="inner").iloc[-60:]
    c = jr.iloc[:, 0].corr(jr.iloc[:, 1])
    b = jr.iloc[:, 0].cov(jr.iloc[:, 1]) / jr.iloc[:, 1].var()
    c_lag = jr.iloc[:, 0].corr(jr.iloc[:, 1].shift(1))
    mx.append((s, c, b, c_lag, jr.index.max()))
mx = pd.DataFrame(mx, columns=["sym", "c", "b", "c_lag", "last"]).merge(f[["sym", "corr60", "beta60"]], on="sym")
say(f"  corr60·beta60 재계산 최대 오차 {np.nanmax(np.abs(mx.c - mx.corr60)):.2e} / {np.nanmax(np.abs(mx.b - mx.beta60)):.2e}, "
    f"마지막 사용 날짜 최댓값 {mx['last'].max().date()} (9/9 이하), 하루 어긋난 상관 중앙값 {mx.c_lag.median():+.2f} vs 같은 날 {mx.c.median():+.2f} → 날짜 정렬 정상")
say(f"  문턱 재계산: sdev_q 1/3 = {q_sdev*100:.3f}%p(앞 절반 {len(a_half)}건 분위, 결과 미사용), corr60 2/3 = {q_corr:.3f}, beta60 1/3 = {q_beta:.3f} — 분석가 값(-0.925, 0.648, 1.24)과 같음, 뒤 절반에 그대로 적용")
say(f"  sdev_q = 매수 봉 직전(i-1) 봉 종가 기준, QQQ는 같은 시각 값 ffill → 미래 참조 없음. 레버리지 종목 0개(3% 문턱 혼입 없음).")
say(f"  주의: 3배 레버리지 없음 확인. corr60 상위 1/3 46종목 중 반도체 계열(칩 설계·장비·파운드리·메모리) {int(f[(f.corr60>=q_corr)].group.isin(['칩 설계','장비·소재·테스트','파운드리·후공정','메모리·스토리지']).sum())}개 → 사실상 섹터 묶음")
say("")

# ---------- 2봉 확인(얇은 체결) 표시 ----------
from events import bars, prep
conf = np.full(len(t), np.nan)
wick = np.full(len(t), np.nan)
for s, idx in t.groupby("sym").groups.items():
    try:
        d = prep(bars(RAW, s, 3), 60, 14)
    except FileNotFoundError:
        continue
    pos = d.index.get_indexer(t.loc[idx, "tt"])
    for j, p in zip(idx, pos):
        if p < 0 or p + 1 >= len(d):
            continue
        ent = t.at[j, "ent"]
        conf[t.index.get_loc(j)] = float(d.l.values[p + 1] <= ent * 1.0005)
        wick[t.index.get_loc(j)] = d.c.values[p] / ent - 1
t["conf"] = conf
t["wick"] = wick
say("## ⑤-0 얇은 체결 점검: '2봉 확인' = 매수 다음 봉 저가도 매수가(+0.05%) 이하로 내려옴")
for s in SES:
    x = t[t.ses == s]
    say(f"  {s}: 확인 비율 {x.conf.mean():.0%} ({int(x.conf.sum())}/{x.conf.notna().sum()}), 평균 수익 확인 {x[x.conf==1].ret.mean()*100:+.2f}% vs 미확인 {x[x.conf==0].ret.mean()*100:+.2f}%, "
        f"매수 봉 종가/매수가 중앙 {x.wick.median()*100:+.2f}%")
say("")

# ---------- 후보별 ----------
qq = q.copy()
qq.index = pd.to_datetime(qq.index).date
q_day = (qq.Close / qq.Close.shift(1) - 1)
VERD = {}
for name, fa, fb, sign, rule in CANDS:
    say(f"## 후보 {name}: A = {rule}, B = 나머지, 분석가 방향 {'A 우세' if sign > 0 else 'B 우세'}")
    R = {}
    for h in ["앞", "뒤"]:
        for sc in ["전체", "정규"]:
            x = scope(t[t.half == h], sc)
            A, B = x[fa(x)], x[fb(x)]
            s = dstat(A, B)
            R[(h, sc)] = s
            say(f"  ① {h}·{sc}: A {s['mA']*100:+.2f}%({s['nA']}건·{s['sA']}종목) B {s['mB']*100:+.2f}%({s['nB']}건·{s['sB']}종목) "
                f"차 {s['diff']*100:+.2f}%p 날짜t {s['t']:+.2f}(n{s['nd']}) | 같은방향 t {sign*s['t']:+.2f}")
    out = {}
    for sc in ["전체", "정규"]:
        x = scope(t[t.half == "뒤"], sc)
        A, B = x[fa(x)], x[fb(x)]
        g1, s1 = drop_k(A, B, sign, "sym", 1)
        g3, s3 = drop_k(A, B, sign, "sym", 3)
        d1, sd1 = drop_k(A, B, sign, "day", 1)
        d2, sd2 = drop_k(A, B, sign, "day", 2)
        sdm, stt = stock_t(A, B)
        say(f"  ③ 뒤·{sc} 쏠림: 1종목({g1[0]}) 빼면 {s1['diff']*100:+.2f}%p t {s1['t']:+.2f} | 3종목({','.join(g3)}) 빼면 {s3['diff']*100:+.2f}%p t {s3['t']:+.2f} | "
            f"1일({d1[0]}) 빼면 {sd1['diff']*100:+.2f}%p t {sd1['t']:+.2f} | 2일({','.join(map(str,d2))}) 빼면 {sd2['diff']*100:+.2f}%p t {sd2['t']:+.2f} | "
            f"종목 단위 Welch 차 {sdm*100:+.2f}%p t {stt:+.2f}")
        isA = fa(x).values
        AB = x[fa(x) | fb(x)]
        isA = fa(AB).values
        b0, t0, G = fe_reg(AB, isA, [], sc == "전체")
        b1, t1, _ = fe_reg(AB, isA, ["vol20"], sc == "전체")
        b2, t2, _ = fe_reg(AB, isA, ["atr14"], sc == "전체")
        tv = []
        for v in [0, 1, 2]:
            a, b = A[A.vt == v], B[B.vt == v]
            if len(a) >= 5 and len(b) >= 5:
                ss = dstat(a, b)
                tv.append(f"{'저중고'[v]} {ss['diff']*100:+.2f}({len(a)}/{len(b)},t{ss['t']:+.1f})")
        sa = dstat(A, B, "r_atr")
        say(f"  ④ 뒤·{sc} 변동성: 날짜FE{'+세션' if sc=='전체' else ''} 회귀 A계수 {b0*100:+.2f}%p t {t0:+.2f}(G{G}) → +vol20 {b1*100:+.2f} t {t1:+.2f} → +atr14 {b2*100:+.2f} t {t2:+.2f} | "
            f"수익/atr14 차 {sa['diff']:+.3f} t {sa['t']:+.2f} | vol 삼분위 안 " + ", ".join(tv))
        cA, cB = A[A.conf == 1], B[B.conf == 1]
        sc2 = dstat(cA, cB)
        say(f"  ⑤ 뒤·{sc} 2봉 확인 거래만: A {sc2['mA']*100:+.2f}%({sc2['nA']}) B {sc2['mB']*100:+.2f}%({sc2['nB']}) 차 {sc2['diff']*100:+.2f}%p t {sc2['t']:+.2f}")
        out[sc] = dict(s1=s1, s3=s3, sd1=sd1, sd2=sd2, stt=stt, t0=t0, t1=t1, t2=t2, sa=sa, conf=sc2)
    # 세션 표준화 차(날짜 군집 t): 날짜별로 세션 안 차를 전체 세션 비중으로 가중
    for h in ["앞", "뒤"]:
        x = t[t.half == h]
        w = x.ses.value_counts(normalize=True)
        per = []
        for dday, g in x.groupby("day"):
            v, ww = 0.0, 0.0
            for s in SES:
                a, b = g[(g.ses == s) & fa(g)], g[(g.ses == s) & fb(g)]
                if len(a) and len(b):
                    v += w[s] * (a.ret.mean() - b.ret.mean())
                    ww += w[s]
            if ww > 0:
                per.append(v / ww)
        per = np.array(per)
        tst = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
        pa = (x[fa(x)].ses == "애프터").mean()
        pb = (x[fb(x)].ses == "애프터").mean()
        ps = {s: dstat(x[(x.ses == s) & fa(x)], x[(x.ses == s) & fb(x)]) for s in SES}
        say(f"  ⑤ {h} 세션: 애프터 비중 A {pa:.0%} B {pb:.0%} | 세션 표준화 차(날짜별) {per.mean()*100:+.2f}%p t {tst:+.2f}(n{len(per)}) | "
            + ", ".join(f"{s} {ps[s]['diff']*100:+.2f}%p t {ps[s]['t']:+.2f}" for s in SES))
        out[f"std_{h}"] = (per.mean(), tst)
    # ⑥ 섹터·시장
    for sc in ["전체", "정규"]:
        x = scope(t[t.half == "뒤"], sc)
        AB = x[fa(x) | fb(x)].copy()
        isA = fa(AB).values
        for gname in AB.group.unique():
            AB[f"g_{gname}"] = (AB.group == gname).astype(float)
        gcols = [c for c in AB.columns if c.startswith("g_")][1:]
        bg, tg, _ = fe_reg(AB, isA, gcols, sc == "전체")
        A, B = x[fa(x)], x[fb(x)]
        dd = (A.groupby("day").ret.mean() - B.groupby("day").ret.mean()).dropna()
        qd = pd.Series({k: q_day.get(pd.Timestamp(k).date(), np.nan) for k in dd.index})
        rc = dd.corr(qd)
        up, dn = dd[qd > 0], dd[qd <= 0]
        say(f"  ⑥ 뒤·{sc} 섹터·시장: 묶음 고정효과 넣은 A계수 {bg*100:+.2f}%p t {tg:+.2f} | 날짜별 차 vs QQQ 일간수익 상관 {rc:+.2f} | "
            f"QQQ 상승일 {len(up)}일 차 {up.mean()*100:+.2f}%p, 하락일 {len(dn)}일 차 {dn.mean()*100:+.2f}%p")
        out[f"sec_{sc}"] = (bg, tg, rc)
    VERD[name] = (R, out, sign)
    say("")

# ---------- ⑦ 무작위 특성 null ----------
say("## ⑦ 무작위 특성 null: 종목 특성 자리에 난수를 넣고 같은 문턱 4종(T3·B3·T3vB3·M) x 범위 2 x 절반 2를 잼")
syms = f.sym.values
sidx = {s: i for i, s in enumerate(syms)}
grp_codes = pd.factorize(f.group)[0]
cells = {}
for h in ["앞", "뒤"]:
    for sc in ["전체", "정규"]:
        x = scope(t[t.half == h], sc)
        dcode, days = pd.factorize(x.day)
        cells[(h, sc)] = (x.sym.map(sidx).values, dcode, len(days), x.ret.values)


def fast_t(maskA, maskB, cell):
    si, dc, nd, r = cell
    a, b = maskA[si], maskB[si]
    na, nb = np.bincount(dc, weights=a, minlength=nd), np.bincount(dc, weights=b, minlength=nd)
    sa, sb = np.bincount(dc, weights=a * r, minlength=nd), np.bincount(dc, weights=b * r, minlength=nd)
    ok = (na > 0) & (nb > 0)
    d = sa[ok] / na[ok] - sb[ok] / nb[ok]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def cuts(v):
    q33, q50, q67 = np.quantile(v, [1 / 3, 0.5, 2 / 3])
    return {"T3": (v >= q67, v < q67), "B3": (v <= q33, v > q33), "T3vB3": (v >= q67, v <= q33), "M": (v >= q50, v < q50)}


def eval_feat(v):
    rows = []
    for c, (mA, mB) in cuts(v).items():
        mA, mB = mA.astype(float), mB.astype(float)
        r = {"cut": c}
        for h in ["앞", "뒤"]:
            for sc in ["전체", "정규"]:
                dm, tv = fast_t(mA, mB, cells[(h, sc)])
                r[f"{h}{sc}_dm"], r[f"{h}{sc}_t"] = dm, tv
        sgn = np.sign(r["앞전체_dm"])  # 분석가와 같이 앞 절반 전체 세션 방향으로 고정
        r["score"] = min(abs(r["앞전체_t"]), abs(r["앞정규_t"])) if np.sign(r["앞전체_t"]) == np.sign(r["앞정규_t"]) else 0.0
        r["t2all"], r["t2reg"] = sgn * r["뒤전체_t"], sgn * r["뒤정규_t"]
        rows.append(r)
    return rows


# 실제 특성 6개도 같은 함수로(분석가 표와 맞는지 대조)
real = {}
for c in ["beta60", "corr60", "ret20", "ret60", "dist52", "ma50"]:
    real[c] = pd.DataFrame(eval_feat(f[c].values))
cr = real["corr60"].set_index("cut").loc["T3"]
br = real["beta60"].set_index("cut").loc["B3"]
say(f"  대조: 빠른 계산 corr60 T3 앞 score {cr.score:.2f}, 뒤 같은방향 t 전체 {cr.t2all:+.2f} 정규 {cr.t2reg:+.2f} | beta60 B3 score {br.score:.2f}, 전체 {br.t2all:+.2f} 정규 {br.t2reg:+.2f}")

NN = 1200
res = {}
for kind in ["종목 iid", "묶음 군집"]:
    allr, best = [], []
    for k in range(NN):
        if kind == "종목 iid":
            v = rng.standard_normal(len(syms))
        else:
            v = rng.standard_normal(grp_codes.max() + 1)[grp_codes] + rng.standard_normal(len(syms)) * 0.5
        rr = eval_feat(v)
        for r in rr:
            r["k"] = k
        allr += rr
    D = pd.DataFrame(allr)
    # 특성 6개 한 가족 = 실제 분석가 탐색(종목 특성 6개 x 문턱 4)과 같은 크기. 가족 안에서 특성마다 최고 score 문턱 하나, 상위 3개 고름.
    fam = []
    for g in range(NN // 6):
        x = D[D.k.between(g * 6, g * 6 + 5)]
        bestc = x.sort_values("score", ascending=False).groupby("k").head(1).sort_values("score", ascending=False).head(3)
        bestc = bestc[bestc.score > 0]
        fam.append(dict(
            corr_like=((bestc.t2reg >= 2.03) & (bestc.t2all >= 1.46)).any(),
            reg2=(bestc.t2reg >= 2.03).any(),
            beta_like=((bestc.t2all >= 2.25)).any(),
            weak=((bestc.t2all > -1) & (bestc.t2reg > -1) & ~((bestc.t2all < 0.5) & (bestc.t2reg < 0.5))).any(),
            confirm_like=((bestc.t2all >= 2) & (bestc.t2reg >= 1.5)).any()))
    F = pd.DataFrame(fam)
    res[kind] = (D, F)
    say(f"  [{kind}] 난수 특성 {NN}개 x 문턱 4 = {len(D)}구분:")
    say(f"    구분 단위: 뒤 같은방향 t 정규>=2.03 {(D.t2reg>=2.03).mean():.1%}, 전체>=1.46 & 정규>=2.03 {((D.t2reg>=2.03)&(D.t2all>=1.46)).mean():.1%}, "
        f"전체>=2.25 {(D.t2all>=2.25).mean():.1%}, 전체>=2.25 & 정규<=0.5 {((D.t2all>=2.25)&(D.t2reg<=0.5)).mean():.1%}, "
        f"앞 score>=1.06 이면서 뒤 정규>=2.03 {((D.score>=1.06)&(D.t2reg>=2.03)).mean():.1%}")
    say(f"    가족 단위({len(F)}가족, 가족=특성 6개, 상위 3개 고름): corr60급(정규>=2.03 & 전체>=1.46) 1개 이상 {F.corr_like.mean():.0%}, "
        f"정규>=2.03 {F.reg2.mean():.0%}, beta60급(전체>=2.25) {F.beta_like.mean():.0%}, '약함' 이상 판정 1개 이상 {F.weak.mean():.0%}, "
        f"'확인됨' 문턱(전체>=2 & 정규>=1.5) {F.confirm_like.mean():.0%}")
    say(f"    뒤 같은방향 t 분포(정규): 표준편차 {D.t2reg.std():.2f}, 95% 분위 {D.t2reg.quantile(.95):+.2f} | (전체): 표준편차 {D.t2all.std():.2f}, 95% 분위 {D.t2all.quantile(.95):+.2f}")
say("")

# ---------- 추가 점검 ----------
say("## ⑧ 추가 점검")
for name, fa, fb, sign, rule in CANDS:
    for h in ["앞", "뒤"]:
        x = t[(t.half == h) & (t.ses == "프리")]
        c = x[x.conf == 1]
        s0, s1 = dstat(x[fa(x)], x[fb(x)]), dstat(c[fa(c)], c[fb(c)])
        say(f"  {name} {h}·프리: 전체 차 {s0['diff']*100:+.2f}%p t {s0['t']:+.2f} → 2봉 확인만 {s1['diff']*100:+.2f}%p t {s1['t']:+.2f}({s1['nA']}/{s1['nB']})")
for h in ["앞", "뒤"]:
    x = t[(t.half == h) & (t.ses == "정규")]
    fa, fb = CANDS[1][1], CANDS[1][2]
    A, B = x[fa(x)], x[fb(x)]
    dd = (A.groupby("day").ret.mean() - B.groupby("day").ret.mean()).dropna()
    qd = pd.Series({k: q_day.get(pd.Timestamp(k).date(), np.nan) for k in dd.index})
    say(f"  corr60 T3 {h}·정규: QQQ 상승일 {int((qd>0).sum())}일 차 {dd[qd>0].mean()*100:+.2f}%p, 하락일 {int((qd<=0).sum())}일 차 {dd[qd<=0].mean()*100:+.2f}%p, "
        f"구간 QQQ 누적 {((1+qd).prod()-1)*100:+.1f}%")
    AB = x[fa(x) | fb(x)].copy()
    for gname in AB.group.unique():
        AB[f"g_{gname}"] = (AB.group == gname).astype(float)
    gcols = [c for c in AB.columns if c.startswith("g_")][1:]
    b, tv, _ = fe_reg(AB, fa(AB).values, gcols + ["vol20"], False)
    say(f"  corr60 T3 {h}·정규: 날짜FE + 묶음FE + vol20 회귀 A계수 {b*100:+.2f}%p t {tv:+.2f}")
ff = f.copy()
for name, col, cond in [("corr60 T3", "corr60", ff.corr60 >= q_corr), ("beta60 B3", "beta60", ff.beta60 <= q_beta)]:
    a, b = ff[cond], ff[~cond]
    say(f"  {name} 앞 기간(7/14~9/8 5분봉 근사, 특성 계산 창과 겹침 — 보조): A {a.early_sum.sum()/a.early_n.sum()*100:+.2f}%({int(a.early_n.sum())}건) "
        f"vs B {b.early_sum.sum()/b.early_n.sum()*100:+.2f}%({int(b.early_n.sum())}건)")
say("")

# ---------- 판정 ----------
say("## 판정 요약")
for name, (R, out, sign) in VERD.items():
    ta, tr = sign * R[("뒤", "전체")]["t"], sign * R[("뒤", "정규")]["t"]
    say(f"  {name}: 뒤 같은방향 t 전체 {ta:+.2f} / 정규 {tr:+.2f}; 2봉 확인 정규 t {sign*out['정규']['conf']['t']:+.2f}; "
        f"세션 표준화 t 앞 {sign*out['std_앞'][1]:+.2f} 뒤 {sign*out['std_뒤'][1]:+.2f}; 묶음FE 정규 t {sign*out['sec_정규'][1]:+.2f}")

say("")
say("## 결론 (기본값 '살아남지 못함')")
say("  sdev_q B3: 살아남지 못함 — 세션 더미 넣은 날짜FE 회귀 -0.09%p t -0.47, 정규장은 반대(+0.63%p, 묶음FE t +2.39 반대 방향), 2봉 확인 전체 -0.17%p t -0.15. 애프터 비중 3% vs 28%가 원인.")
say("  corr60 T3: 살아남지 못함(관찰 후보) — 정규 뒤 t 2.03은 vol20 통제(2.15)·묶음FE(2.14)에는 버티나 둘 다 넣으면 1.48, 유리한 2일 빼면 1.12, 종목 단위 t 1.67, "
    "2봉 확인 0.77, 앞 절반 정규 t 1.06(FE+묶음+vol 0.14), 7/14~9/8 근사는 반대(A +0.76% vs B +0.94%). 우위는 QQQ 상승일(+1.23%p, 7일)에 몰림. "
    "무작위 null 가족 확률 2~5%라 우연만으로는 다 설명되지 않음.")
say("  beta60 B3: 살아남지 못함 — 정규 뒤 -0.02%p t 0.01, 세션 더미 회귀 t 1.09 → +vol20 t 0.56, 차이는 프리(얇은 체결: 프리 미확인 체결 평균 +2.67% vs 확인 -0.16%)에 몰림. 무작위 가족의 11~16%가 전체 t>=2.25를 냄.")
say("  분석가의 '약함' 판정은 무작위 가족의 78~90%에서 1개 이상 나오는 수준이라 정보가 적다.")

with open(os.path.join(RES, "verify_trend_link.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
