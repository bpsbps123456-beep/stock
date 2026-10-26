"""이격매수매도 종목 궁합 연구 — 매수 순간 상황(거래 단위) 묶음. 2026-10-09.

질문 ① 매수 순간 상황(qdev 시장 이격, sdev_q 혼자 빠짐, day_ret 그날 이미 빠진 폭, prev_ret 전날 수익, earn 실적 구간, ses·시각)에 따라 거래 성적이 갈리는가.
     ② 종목 간 성적 차이가 이 상황 구성 차이(예: 어떤 종목은 혼자 빠지는 매수가 많음)로 설명되는가.
방법: 문턱은 앞 절반(9/10~9/23) 분위로 정하고 방향도 앞에서 정해 뒤 절반(9/24~10/8)에 그대로 적용한다.
      t는 날짜 군집: 날짜마다 (A 무리 평균 - B 무리 평균)을 구해 그 평균/표준오차(날짜 수 = 자유도+1).
      통제: ret/atr14 차이, vol20 삼분위(종목 단위, 9/9까지 값) 안 차이의 날짜 평균, 애프터 비중, 정규장만, 상위 1종목·상위 1일 빼기.
      모든 거래 단위 값은 매수 봉 직전 봉까지 값(build.py 정의)이므로 매수 전에 알 수 있다. 실적 일정도 미리 알려진다.
입력: ../results/trades.csv, ../results/features.csv  (build.py가 만든 것, 다시 받지 않음)
출력: ../results/fam_context.txt(요약), ../results/fam_context_tests.csv(모든 시험 행)
사용: py fam_context.py"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
CONT = ["qdev", "sdev_q", "day_ret", "prev_ret"]
TOD_ALL = [(240, 300, "04:00-04:59"), (300, 570, "05:00-09:29"), (570, 600, "09:30-09:59"), (600, 960, "10:00-15:59"), (960, 1200, "16:00-19:59")]
TOD_REG = [(570, 600, "09:30-09:59"), (600, 660, "10:00-10:59"), (660, 960, "11:00-15:59")]
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def load():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    ts = pd.to_datetime(t.t.str[:19])
    t["mod"] = ts.dt.hour * 60 + ts.dt.minute
    t = t.merge(f[["sym", "vol20", "atr14"]], on="sym", how="left")
    t["ra"] = t.ret / t.atr14
    q1, q2 = f.vol20.quantile([1 / 3, 2 / 3])
    t["vt"] = np.select([t.vol20 <= q1, t.vol20 <= q2, t.vol20 > q2], [0, 1, 2], -1)
    t["aft"] = t.ses == "애프터"
    return t, f


def tstat(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return (x.mean() if n else np.nan), np.nan, n
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(n)), n


def daypair(d, a, col="ret"):
    """날짜마다 A평균 - B평균 → 날짜 평균, t, 날짜 수."""
    g = d.groupby([d.day, a])[col].mean().unstack()
    if True not in g.columns or False not in g.columns:
        return np.nan, np.nan, 0
    return tstat(g[True] - g[False])


def volpair(d, a):
    """vol20 삼분위 안에서 날짜마다 A-B 차이 → 삼분위 평균 → 날짜 t."""
    d = d[d.vt >= 0]
    g = d.groupby([d.day, d.vt, a[d.index]]).ret.mean().unstack()
    if True not in g.columns or False not in g.columns:
        return np.nan, np.nan, 0
    return tstat((g[True] - g[False]).dropna().groupby(level=0).mean())


def sespair(d, a):
    """같은 세션 안에서 날짜마다 A-B 차이 → 세션 평균 → 날짜 t(애프터·프리 비중 차이 통제)."""
    g = d.groupby([d.day, d.ses, a]).ret.mean().unstack()
    if True not in g.columns or False not in g.columns:
        return np.nan, np.nan, 0
    return tstat((g[True] - g[False]).dropna().groupby(level=0).mean())


def stats(d, a, sgn, full=True):
    """a: d와 같은 인덱스의 bool. sgn: 규칙 방향(+1이면 A가 좋다고 본 것)."""
    A, B = d[a], d[~a]
    r = dict(nA=len(A), nB=len(B), symA=A.sym.nunique(), symB=B.sym.nunique(), mA=A.ret.mean(), mB=B.ret.mean())
    r["diff"] = r["mA"] - r["mB"]
    r["dd"], r["t"], r["nd"] = daypair(d, a)
    if not full:
        return r
    r["ra_dd"], r["ra_t"], _ = daypair(d, a, "ra")
    r["vol_dd"], r["vol_t"], _ = volpair(d, a)
    r["ses_dd"], r["ses_t"], _ = sespair(d, a)
    r["aftA"], r["aftB"] = A.aft.mean(), B.aft.mean()
    for key, col in (("xs", "sym"), ("xd", "day")):
        best = None
        for v in d[col].unique():
            k = d[col] != v
            dd, tt, _ = daypair(d[k], a[k])
            if np.isnan(dd):
                continue
            if best is None or sgn * dd < sgn * best[1]:
                best = (v, dd, tt)
        r[key], r[key + "_dd"], r[key + "_t"] = best if best else (None, np.nan, np.nan)
    return r


def splits(d1):
    """(특성, 이름, 문턱 설명, 마스크 함수) — 문턱은 앞 절반 d1에서."""
    out = []
    for c in CONT:
        q33, q50, q67 = d1[c].quantile([1 / 3, 0.5, 2 / 3])
        out += [(c, "하위1/3", f"{c}<={q33:+.4f}", lambda x, c=c, v=q33: x[c] <= v),
                (c, "하위1/2", f"{c}<={q50:+.4f}", lambda x, c=c, v=q50: x[c] <= v),
                (c, "상위1/3", f"{c}>{q67:+.4f}", lambda x, c=c, v=q67: x[c] > v)]
        if c == "qdev":
            out.append((c, "음수", "qdev<0", lambda x: x.qdev < 0))
    out.append(("earn", "실적구간", "earn=True", lambda x: x.earn.astype(bool)))
    return out


def run_tests(t):
    rows = []
    scopes = {"전체": t, "정규": t[t.ses == "정규"]}
    for sc, d in scopes.items():
        d1, d2 = d[d.half == "앞"], d[d.half == "뒤"]
        for feat, nm, rule, fn in splits(d1):
            a1, a2 = fn(d1), fn(d2)
            r1 = stats(d1, a1, 1, full=False)
            sgn = 1 if (r1["dd"] if not np.isnan(r1["dd"]) else r1["diff"]) >= 0 else -1
            rows.append(dict(scope=sc, feat=feat, split=nm, rule=rule, pool="후보", sgn=sgn,
                             **{f"1_{k}": v for k, v in stats(d1, a1, sgn).items()},
                             **{f"2_{k}": v for k, v in stats(d2, a2, sgn).items()}))
    # 세션·시각: 애프터 얇은 체결 때문에 정규장만 확인이 불가능한 것은 설명용
    d = t
    d1, d2 = d[d.half == "앞"], d[d.half == "뒤"]
    tests = [("ses", s, f"ses={s}", lambda x, s=s: x.ses == s, "전체", "설명용") for s in ["프리", "정규", "애프터"]]
    tests += [("tod", lab, f"{lab} ET", lambda x, lo=lo, hi=hi: (x["mod"] >= lo) & (x["mod"] < hi), "전체", "설명용") for lo, hi, lab in TOD_ALL]
    tests += [("tod", lab, f"{lab} ET(정규 안)", lambda x, lo=lo, hi=hi: (x["mod"] >= lo) & (x["mod"] < hi), "정규", "후보") for lo, hi, lab in TOD_REG]
    for feat, nm, rule, fn, sc, pool in tests:
        dd = t if sc == "전체" else t[t.ses == "정규"]
        e1, e2 = dd[dd.half == "앞"], dd[dd.half == "뒤"]
        a1, a2 = fn(e1), fn(e2)
        r1 = stats(e1, a1, 1, full=False)
        sgn = 1 if r1["dd"] >= 0 else -1
        rows.append(dict(scope=sc, feat=feat, split=nm, rule=rule, pool=pool, sgn=sgn,
                         **{f"1_{k}": v for k, v in stats(e1, a1, sgn).items()},
                         **{f"2_{k}": v for k, v in stats(e2, a2, sgn).items()}))
    return pd.DataFrame(rows)


def pick(tab):
    """앞 절반 점수: 전체·정규 둘 다 있는 분할은 min(|t전체|,|t정규|)(부호 같을 때만), 정규 시각 분할은 |t정규|."""
    c = tab[tab.pool == "후보"]
    sc = []
    for (feat, split), g in c.groupby(["feat", "split"], sort=False):
        ts = g.set_index("scope")["1_t"]
        if feat == "tod":
            s = abs(ts.get("정규", np.nan))
        elif ts.notna().all() and len(ts) == 2 and np.sign(ts["전체"]) == np.sign(ts["정규"]):
            s = min(abs(ts["전체"]), abs(ts["정규"]))
        else:
            s = 0.0
        sc.append((feat, split, s))
    sc = pd.DataFrame(sc, columns=["feat", "split", "score"]).sort_values("score", ascending=False)
    return sc


def pct(x):
    return "nan" if pd.isna(x) else f"{100 * x:+.2f}%"


def fmt_row(r, h):
    p = f"{h}_"
    s = (f"A {r[p + 'nA']}건/{r[p + 'symA']}종목 {pct(r[p + 'mA'])} vs B {r[p + 'nB']}건/{r[p + 'symB']}종목 {pct(r[p + 'mB'])}"
         f" | 날짜차 {pct(r[p + 'dd'])} t={r[p + 't']:+.2f} ({r[p + 'nd']}일)")
    if p + "ra_dd" in r and not pd.isna(r.get(p + "ra_dd")):
        s += (f" | ret/atr 차 {r[p + 'ra_dd']:+.3f} t={r[p + 'ra_t']:+.2f} | vol삼분위내 {pct(r[p + 'vol_dd'])} t={r[p + 'vol_t']:+.2f}"
              f" | 세션내 {pct(r[p + 'ses_dd'])} t={r[p + 'ses_t']:+.2f}"
              f" | 애프터비중 A {100 * r[p + 'aftA']:.0f}% B {100 * r[p + 'aftB']:.0f}%"
              f" | -{r[p + 'xs']} {pct(r[p + 'xs_dd'])} t={r[p + 'xs_t']:+.2f} | -{r[p + 'xd']} {pct(r[p + 'xd_dd'])} t={r[p + 'xd_t']:+.2f}")
    return s


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def design(d, lim, ses_only=False):
    X = [np.ones(len(d)), (d.ses == "프리").astype(float), (d.ses == "애프터").astype(float),
         ((d["mod"] >= 240) & (d["mod"] < 300)).astype(float), d.earn.astype(float)]
    if ses_only:
        return np.column_stack(X[:4])
    for c in CONT:
        X.append(d[c].clip(*lim[c]))
    return np.column_stack(X)


def spear(a, b):
    k = a.notna() & b.notna()
    n = int(k.sum())
    r = a[k].corr(b[k], method="spearman") if n > 3 else np.nan
    t = r * np.sqrt((n - 2) / (1 - r * r)) if n > 3 and abs(r) < 1 else np.nan
    return r, t, n


def composition(t, cands):
    """② 종목 차이가 상황 구성 차이로 설명되는가."""
    say("\n== ② 종목 간 차이 vs 상황 구성 ==")
    say("상황 모형: 앞 절반 거래로 ret ~ 프리·애프터·04시대·earn·qdev·sdev_q·day_ret·prev_ret(앞 절반 1/99% 절단) OLS → 두 절반 예측. 잔차 = ret - 예측.")
    say("세션만잔차: 프리·애프터·04시대만 넣은 같은 모형의 잔차. 날짜 제거: ret - 그날 전체 거래 평균(날짜 공통 장세 제거, 설명용).")
    out = {}
    for sc, d in (("전체", t), ("정규", t[t.ses == "정규"])):
        d = d.copy()
        d1 = d[d.half == "앞"]
        lim = {c: tuple(d1[c].quantile([0.01, 0.99])) for c in CONT}
        X1 = design(d1, lim)
        if sc == "정규":
            X1 = X1[:, [0, 4, 5, 6, 7, 8]]
        b = ols(X1, d1.ret.values)
        Xa = design(d, lim) if sc == "전체" else design(d, lim)[:, [0, 4, 5, 6, 7, 8]]
        d["pred"] = Xa @ b
        d["res"] = d.ret - d.pred
        if sc == "전체":
            bs = ols(design(d1, lim, True), d1.ret.values)
            d["sres"] = d.ret - design(d, lim, True) @ bs
        else:
            d["sres"] = d.ret - d1.ret.mean()
        d["dm"] = d.ret - d.groupby("day").ret.transform("mean")
        d["dmres"] = d.dm - (d.pred - d.groupby("day").pred.transform("mean"))
        names = ["상수", "프리", "애프터", "04시대", "earn"] + CONT if sc == "전체" else ["상수", "earn"] + CONT
        say(f"[{sc}] 계수: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(names, b)))
        r2 = 1 - ((d.res[d.half == '앞']) ** 2).sum() / ((d1.ret - d1.ret.mean()) ** 2).sum()
        d2 = d[d.half == "뒤"]
        r2b = 1 - (d2.res ** 2).sum() / ((d2.ret - d2.ret.mean()) ** 2).sum()
        say(f"[{sc}] 거래 단위 R² 앞(적합) {r2:.3f}, 뒤(표본 밖) {r2b:+.3f}")
        g = d.groupby(["sym", "half"]).agg(n=("ret", "size"), raw=("ret", "mean"), pred=("pred", "mean"), res=("res", "mean"),
                                            dm=("dm", "mean"), dmres=("dmres", "mean"), ra=("ra", "mean"),
                                            sres=("sres", "mean")).unstack("half")
        for mn in (1, 3):
            k = (g["n"]["앞"] >= mn) & (g["n"]["뒤"] >= mn)
            gg = g[k]
            parts = []
            for col, lab in (("raw", "원수익"), ("ra", "ret/atr"), ("dm", "날짜제거"), ("sres", "세션만잔차"), ("res", "상황잔차"), ("dmres", "날짜+상황잔차"), ("pred", "상황예측")):
                r, tt, n = spear(gg[col]["앞"], gg[col]["뒤"])
                parts.append(f"{lab} {r:+.2f}(t{tt:+.1f})")
            say(f"[{sc}] 종목 지속성 앞→뒤 스피어만, 두 절반 각 {mn}건 이상 {int(k.sum())}종목: " + " | ".join(parts))
        for h in ("앞", "뒤"):
            k = g["n"][h] >= 3
            r, tt, n = spear(g["raw"][h][k], g["pred"][h][k])
            cv = np.corrcoef(g["raw"][h][k], g["pred"][h][k])[0, 1]
            sd_raw, sd_pred = g["raw"][h][k].std(), g["pred"][h][k].std()
            say(f"[{sc}] {h} 절반(3건 이상 {n}종목): 종목 평균 원수익 vs 상황예측 스피어만 {r:+.2f}(t{tt:+.1f}), 피어슨² {cv * cv:.2f}"
                f" | 종목 평균 표준편차 원수익 {100 * sd_raw:.2f}%p, 상황예측 {100 * sd_pred:.2f}%p")
        k = (g["n"]["앞"] >= 3) & (g["n"]["뒤"] >= 3)
        r, tt, n = spear(g["pred"]["앞"][k], g["raw"]["뒤"][k])
        say(f"[{sc}] 앞 절반 상황구성(예측평균)으로 뒤 절반 종목 원수익 맞히기: 스피어만 {r:+.2f}(t{tt:+.1f}, {n}종목)")
        out[sc] = (d, g)
    # 잘 통한 종목 vs 안 통한 종목의 상황 구성
    d, g = out["전체"]
    k = g["n"]["앞"] >= 4
    raw1 = g["raw"]["앞"][k]
    lo, hi = raw1.quantile([1 / 3, 2 / 3])
    grp = {"앞 상위1/3": raw1[raw1 > hi].index, "앞 하위1/3": raw1[raw1 <= lo].index}
    say(f"\n잘 통한/안 통한 종목(앞 절반 4건 이상 {int(k.sum())}종목을 앞 평균수익 삼분위로) 상황 구성:")
    say("무리 | 절반 | 거래 | 평균 | 정규만 평균 | 애프터% | 04시대% | qdev<0% | sdev_q 평균 | day_ret 평균 | prev_ret 평균 | 상황예측 | 상황잔차 | 날짜+상황잔차")
    for nm, syms in grp.items():
        for h in ("앞", "뒤"):
            x = d[d.sym.isin(syms) & (d.half == h)]
            rg = x[x.ses == "정규"].ret.mean()
            say(f"{nm}({len(syms)}) | {h} | {len(x)} | {pct(x.ret.mean())} | {pct(rg)} | {100 * x.aft.mean():.0f} | "
                f"{100 * ((x['mod'] >= 240) & (x['mod'] < 300)).mean():.0f} | {100 * (x.qdev < 0).mean():.0f} | {pct(x.sdev_q.mean())} | "
                f"{pct(x.day_ret.mean())} | {pct(x.prev_ret.mean())} | {pct(x.pred.mean())} | {pct(x.res.mean())} | {pct(x.dmres.mean())}")
    # 애프터 매수 비중이 높은 종목은 어떤 종목인가(9/9까지 특성, 설명용)
    f = pd.read_csv(os.path.join(RES, "features.csv")).set_index("sym")
    sh = d.groupby(["sym", "half"]).aft.mean().unstack()
    r, tt, n = spear(sh["앞"], sh["뒤"])
    say(f"종목별 애프터 매수 비중 앞→뒤 지속 스피어만 {r:+.2f}(t{tt:+.1f}, {n}종목)")
    shall = d.groupby("sym").aft.mean()
    parts = []
    for c in ("adv20", "vol20", "atr14", "price", "beta60"):
        r, tt, n = spear(shall.reindex(f.index), f[c])
        parts.append(f"{c} {r:+.2f}(t{tt:+.1f})")
    say(f"종목별 애프터 매수 비중(21일) vs 9/9 특성 스피어만 {n}종목: " + " | ".join(parts))
    # 두 무리 뒤 절반 차이의 날짜 t
    x = d[d.half == "뒤"].copy()
    x = x[x.sym.isin(grp["앞 상위1/3"]) | x.sym.isin(grp["앞 하위1/3"])]
    a = x.sym.isin(grp["앞 상위1/3"])
    for col, lab in (("ret", "원수익"), ("res", "상황잔차")):
        dd, tt, nd = daypair(x, a, col)
        say(f"뒤 절반 (앞 상위1/3 - 앞 하위1/3) {lab} 날짜차 {pct(dd)} t={tt:+.2f} ({nd}일)")
    xr = x[x.ses == "정규"]
    dd, tt, nd = daypair(xr, xr.sym.isin(grp["앞 상위1/3"]))
    say(f"뒤 절반 정규장만 (앞 상위1/3 - 앞 하위1/3) 날짜차 {pct(dd)} t={tt:+.2f} ({nd}일)")
    # 후보 상황의 종목별 비중이 종목 성적과 연결되는가
    for feat, split, rule, fn in cands:
        d["_c"] = fn(d)
        sh = d.groupby(["sym", "half"])._c.mean().unstack()
        for h in ("앞", "뒤"):
            kk = g["n"][h] >= 3
            r, tt, n = spear(sh[h][kk], g["raw"][h][kk])
            say(f"후보 {rule}: {h} 절반 종목별 비중 vs 종목 평균수익 스피어만 {r:+.2f}(t{tt:+.1f}, {n}종목)")
        r, tt, n = spear(sh["앞"], sh["뒤"])
        say(f"후보 {rule}: 종목별 비중 앞→뒤 지속 스피어만 {r:+.2f}(t{tt:+.1f}, {n}종목)")


def main():
    t, f = load()
    say(f"거래 {len(t)}건, {t.sym.nunique()}종목, {t.day.nunique()}일 | 앞 {int((t.half == '앞').sum())}건 {t[t.half == '앞'].day.nunique()}일, "
        f"뒤 {int((t.half == '뒤').sum())}건 {t[t.half == '뒤'].day.nunique()}일 | 특성 없음(SKHY) {int(t.vol20.isna().sum())}건은 vol 통제에서 빠짐")
    say("단위: 거래당 평균 수익(%), 날짜차 = 날짜별 (A평균-B평균)의 평균, t = 날짜 군집. A = 규칙 조건에 맞는 무리, B = 나머지.")
    say("통제 열: ret/atr = 수익÷종목 atr14 의 날짜차 | vol삼분위내 = 종목 vol20 삼분위 안 차이의 날짜 평균 | 세션내 = 같은 세션(프리·정규·애프터) 안 차이의 날짜 평균 | 애프터비중 | -X = 그 종목/날 하나를 빼서 규칙 방향 차이가 가장 줄어든 경우")
    tab = run_tests(t)
    n_c = int((tab.pool == "후보").sum())
    n_d = int((tab.pool == "설명용").sum())
    say(f"\n시험 수: 후보 {n_c}개(연속 4특성 x 분위 3 + qdev 부호 + earn = 14 분할 x 전체·정규 2범위 = 28, 정규장 시각 3) + 설명용 {n_d}개(세션 3, 하루 시각 5) = {n_c + n_d}개")
    say("문턱(앞 절반 분위)과 방향(앞 절반 날짜차 부호)을 앞에서 정해 뒤에 그대로 적용.")
    say("\n== ① 모든 시험 (앞 | 뒤) ==")
    for _, r in tab.iterrows():
        say(f"\n[{r.pool}·{r.scope}] {r.rule} ({r.feat} {r.split}) 방향 {'A가 좋음' if r.sgn > 0 else 'A가 나쁨'}")
        say("  앞: " + fmt_row(r, 1))
        say("  뒤: " + fmt_row(r, 2))
    sc = pick(tab)
    say("\n== 앞 절반 점수 순위(후보 17 분할) ==")
    for _, r in sc.iterrows():
        say(f"{r.feat} {r.split}: {r.score:.2f}")
    top = sc[sc.score > 0].drop_duplicates("feat").head(3)
    say("\n== 고른 후보 최대 3개(특성마다 앞 점수 최고 분할 1개 — 같은 특성의 다른 문턱은 중복): 뒤 절반 확인 ==")
    cands = []
    for _, s in top.iterrows():
        g = tab[(tab.feat == s.feat) & (tab.split == s.split) & (tab.pool == "후보")]
        for _, r in g.iterrows():
            say(f"[{r.scope}] {r.rule} 방향 {'A가 좋음' if r.sgn > 0 else 'A가 나쁨'} (점수 {s.score:.2f})")
            say("  앞: " + fmt_row(r, 1))
            say("  뒤: " + fmt_row(r, 2))
        if s.feat != "tod":
            sp = [x for x in splits(t[t.half == "앞"]) if x[0] == s.feat and x[1] == s.split][0]
            cands.append(sp)
            for ss in ("프리", "애프터"):
                e = t[t.ses == ss]
                e1, e2 = e[e.half == "앞"], e[e.half == "뒤"]
                sg = int(g.iloc[0].sgn)
                r1, r2 = stats(e1, sp[3](e1), sg, full=False), stats(e2, sp[3](e2), sg, full=False)
                say(f"  [{ss}만, 전체 문턱] 앞 A {r1['nA']}건 {pct(r1['mA'])} vs B {r1['nB']}건 {pct(r1['mB'])} 날짜차 {pct(r1['dd'])} t={r1['t']:+.2f}"
                    f" | 뒤 A {r2['nA']}건 {pct(r2['mA'])} vs B {r2['nB']}건 {pct(r2['mB'])} 날짜차 {pct(r2['dd'])} t={r2['t']:+.2f}")
    composition(t, cands)
    tab.to_csv(os.path.join(RES, "fam_context_tests.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RES, "fam_context.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
