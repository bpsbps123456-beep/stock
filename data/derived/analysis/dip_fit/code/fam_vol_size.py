"""이격매수매도 종목 궁합 — 특성 묶음 '변동성·규모·유동성'(vol20 atr14 adv20 price gap60 ovn60 lev) + group(섹터 묶음).

입력: ../results/trades.csv, ../results/features.csv (build.py가 만든 것, 다시 받지 않음)
방법(고르기와 확인을 나눔):
  1) 구분: 연속 특성 6개 x {상위 1/3, 하위 1/3, 중앙값 위} = 18개 + group(5종목 이상 묶음) 1개 대 나머지 12개 = 30개 구분.
     문턱은 136종목의 9/9 특성 분포(결과와 무관, 매수 전에 앎)로 정한다. lev는 특성표에 해당 종목 0개라 시험 불가.
  2) 앞 절반(9/10~9/23, 10거래일)에서 30 구분 x 범위 2(전체 세션·정규장만) x 단위 2(거래·종목) = 120 조합의 날짜 군집 t를 잰다.
     거래 단위: 날짜별 (A 거래 평균 - B 거래 평균)의 평균과 t. 종목 단위: 날짜별로 종목 평균을 먼저 낸 뒤 A·B 종목 평균의 차.
  3) 고르기 점수 = 네 조합(전체·정규 x 거래·종목) t의 부호가 모두 같을 때 그 |t|의 최솟값, 아니면 0.
     점수 순으로 특성당 1개(vol20·atr14는 순위상관 0.93이라 한 묶음), 점수 > 0인 것 최대 3개.
  4) 고른 문턱·방향을 뒤 절반(9/24~10/8, 11거래일)에 그대로 적용한다.
     통제: 수익/atr14, vol20 삼분위 안 차이, 세션 구성 고정 차이, 애프터 비중, 차이를 가장 줄이는 1종목·1일 제외, 수익 ±10% 절단.
  판정(미리 고정): 뒤 절반 전체·거래 단위 차이가 앞과 같은 방향이고 t>=2, 정규장만도 같은 방향, 수익/atr14도 같은 방향 -> 확인됨(기계 판정).
                   전체·거래 단위가 같은 방향이지만 나머지를 못 채움 -> 약함. 반대 방향 -> 뒤에서 실패.
                   과제 주의 1(애프터 비중으로 설명되는지)을 판정에 넣는다: 기계 판정이 확인됨이어도 뒤 절반 세션 고정 차 t<2
                   또는 정규장만 거래 단위 t<1이면 '약함(세션 구성으로 설명됨)'으로 낮춘다. 이 보정은 처음 실행에서 정규장 차 0.00%가
                   부호 검사를 통과한 것을 보고 넣었다(더 엄격하게만 바꿈).
  보조: 정규장만·거래 단위 앞 절반 |t| 상위 3개(특성 묶음당 1개)를 뒤 절반에 그대로 적용(30 구분에서 고름).
저장: ../results/fam_vol_size.txt(보고), ../results/fam_vol_size_tests.csv(120 조합 전부 + 앞·뒤 값)
사용: py fam_vol_size.py"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
CONT = ["vol20", "atr14", "adv20", "price", "gap60", "ovn60"]
FAMILY = {"vol20": "변동성", "atr14": "변동성"}
SES_ALL = ["프리", "정규", "애프터"]
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)


def load():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    miss = sorted(set(t.sym) - set(f.sym))
    t = t.merge(f, on="sym", how="inner")
    t["nret"] = t.ret / t.atr14
    t["wret"] = t.ret.clip(-0.10, 0.10)
    q = f.vol20.quantile([1 / 3, 2 / 3]).values
    t["vt"] = np.where(t.vol20 <= q[0], 0, np.where(t.vol20 <= q[1], 1, 2))
    return t, f, miss


def splits(f):
    out = []
    for c in CONT:
        q33, q50, q67 = f[c].quantile([1 / 3, 0.5, 2 / 3]).values
        out += [(c, "상위 1/3", f"{c}>={q67:.4g}", lambda x, c=c, v=q67: x[c] >= v),
                (c, "하위 1/3", f"{c}<={q33:.4g}", lambda x, c=c, v=q33: x[c] <= v),
                (c, "중앙값 위", f"{c}>{q50:.4g}", lambda x, c=c, v=q50: x[c] > v)]
    for g, n in f.group.value_counts().items():
        if n >= 5:
            out.append(("group", g, f"group={g}({n}종목)", lambda x, g=g: x.group == g))
    return out


def tstat(d):
    d = pd.Series(d).dropna()
    if len(d) < 3 or d.std(ddof=1) == 0:
        return d.mean() if len(d) else np.nan, np.nan, len(d)
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), len(d)


def daily_diff(x, a, col="ret", unit="거래"):
    """날짜별 A-B 차이 시계열. 종목 단위면 날짜 안에서 종목 평균을 먼저 낸다."""
    x = x.assign(A=a.values)
    if unit == "종목":
        x = x.groupby(["day", "sym", "A"], as_index=False)[col].mean()
    m = x.groupby(["day", "A"])[col].mean().unstack()
    if True not in m or False not in m:
        return pd.Series(dtype=float)
    return (m[True] - m[False]).dropna()


def summ(x, a, col="ret", unit="거래"):
    """A·B 평균(거래 단위: 거래당, 종목 단위: 종목별 거래당 평균의 평균), 건수, 종목수, 날짜 군집 t"""
    A, B = x[a.values], x[~a.values]
    if unit == "종목":
        ma, mb = A.groupby("sym")[col].mean().mean(), B.groupby("sym")[col].mean().mean()
    else:
        ma, mb = A[col].mean(), B[col].mean()
    dm, tt, k = tstat(daily_diff(x, a, col, unit))
    return dict(mA=ma, mB=mb, nA=len(A), nB=len(B), sA=A.sym.nunique(), sB=B.sym.nunique(), diff=ma - mb, dday=dm, t=tt, k=k)


def scope(x, sc):
    return x if sc == "전체" else x[x.ses == "정규"]


def pct(v):
    return f"{v * 100:+.2f}%"


def fmt(r):
    return (f"A {pct(r['mA'])}({r['nA']}건·{r['sA']}종목) B {pct(r['mB'])}({r['nB']}건·{r['sB']}종목) "
            f"차 {pct(r['diff'])} 날짜평균차 {pct(r['dday'])} t={r['t']:+.2f}(k={r['k']})")


def drop_worst(x, a, sign, key):
    """차이를 sign 방향에서 가장 많이 줄이는 1개(종목 또는 날짜)를 빼고 다시 잰다."""
    best = None
    for v in x[key].unique():
        keep = (x[key] != v).values
        dm, tt, k = tstat(daily_diff(x[keep], a[keep]))
        if best is None or sign * dm < best[1] * sign:
            best = (v, dm, tt, k)
    return best


def vol_within(x, a, col="ret"):
    """vol20 삼분위 안에서 날짜별 A-B 차이, 삼분위 평균(양쪽 다 있는 칸만)."""
    rows = []
    for (d, v), g in x.assign(A=a.values).groupby(["day", "vt"]):
        if g.A.any() and (~g.A).any():
            rows.append((d, v, g[g.A][col].mean() - g[~g.A][col].mean()))
    if not rows:
        return np.nan, np.nan, 0
    r = pd.DataFrame(rows, columns=["day", "vt", "d"]).groupby("day").d.mean()
    return tstat(r)


def ses_fixed(x, a):
    """세션 구성 고정: 날짜별로 세션마다 A-B 차를 내고 이 구간 전체 세션 비중으로 가중."""
    w = x.ses.value_counts(normalize=True)
    rows = []
    for d, g in x.assign(A=a.values).groupby("day"):
        num = den = 0.0
        for s, gs in g.groupby("ses"):
            if gs.A.any() and (~gs.A).any():
                num += w[s] * (gs[gs.A].ret.mean() - gs[~gs.A].ret.mean())
                den += w[s]
        if den:
            rows.append(num / den)
    return tstat(rows)


def after_share(x, a):
    return (x[a.values].ses == "애프터").mean(), (x[~a.values].ses == "애프터").mean()


def controls(x, a, sign, name):
    out = {}
    out["nret"] = summ(x, a, "nret")
    out["vt"] = vol_within(x, a) if name not in ("vol20", "atr14") else (np.nan, np.nan, 0)
    out["sesfix"] = ses_fixed(x, a)
    out["aft"] = after_share(x, a)
    out["volAB"] = (x[a.values].drop_duplicates("sym").vol20.median(), x[~a.values].drop_duplicates("sym").vol20.median())
    out["d_sym"] = drop_worst(x, a, sign, "sym")
    out["d_day"] = drop_worst(x, a, sign, "day")
    out["w"] = summ(x, a, "wret")
    reg = x.ses == "정규"
    out["reg_nret"] = summ(x[reg], a[reg], "nret")
    out["per_ses"] = {ss: (x[a.values & (x.ses == ss).values].ret.mean(), x[~a.values & (x.ses == ss).values].ret.mean(),
                           int((a.values & (x.ses == ss).values).sum()), int((~a.values & (x.ses == ss).values).sum())) for ss in SES_ALL}
    return out


def ctrl_text(c):
    vt = c["vt"]
    vtxt = "해당 없음(변동성 자체)" if np.isnan(vt[0]) and vt[2] == 0 else f"{pct(vt[0])} t={vt[1]:+.2f}"
    return (f"수익/atr14 차 {c['nret']['diff']:+.3f}ATR t={c['nret']['t']:+.2f}(정규만 {c['reg_nret']['diff']:+.3f} t={c['reg_nret']['t']:+.2f}) | "
            f"vol 삼분위 안 차 {vtxt} | 세션 고정 차 {pct(c['sesfix'][0])} t={c['sesfix'][1]:+.2f} | "
            f"애프터 비중 A {c['aft'][0] * 100:.0f}% B {c['aft'][1] * 100:.0f}% | vol20 중앙값 A {c['volAB'][0] * 100:.1f}% B {c['volAB'][1] * 100:.1f}% | "
            f"1종목({c['d_sym'][0]}) 빼면 {pct(c['d_sym'][1])} t={c['d_sym'][2]:+.2f} | 1일({c['d_day'][0]}) 빼면 {pct(c['d_day'][1])} t={c['d_day'][2]:+.2f} | "
            f"±10% 절단 차 {pct(c['w']['diff'])} t={c['w']['t']:+.2f} | 세션별 A/B: " +
            " ".join(f"{k} {pct(v[0])}({v[2]})/{pct(v[1])}({v[3]})" for k, v in c["per_ses"].items()))


def main():
    t, f, miss = load()
    say(f"# 이격매수매도 종목 궁합 — 변동성·규모·유동성·섹터 묶음 (fam_vol_size, 2026-10-09)")
    say(f"거래 {len(t)}건·{t.sym.nunique()}종목(특성 없는 {miss} 제외), 매수일 {t.day.min()}~{t.day.max()} {t.day.nunique()}거래일")
    say("lev: 특성표 136종목 중 lev=True 0개(3배 레버리지 종목이 유니버스에 없음) -> 시험 불가")
    say("문턱: 136종목의 9/9 특성 분포 삼분위·중앙값(결과를 보기 전, 매수 전에 아는 값)")
    sp = splits(f)
    first, second = t[t.half == "앞"], t[t.half == "뒤"]
    rows = []
    for name, lab, rule, fn in sp:
        for sc in ["전체", "정규"]:
            for un in ["거래", "종목"]:
                for h, x in [("앞", first), ("뒤", second)]:
                    xs = scope(x, sc)
                    r = summ(xs, fn(xs), "ret", un)
                    rows.append(dict(feature=name, split=lab, rule=rule, scope=sc, unit=un, half=h, **r))
    R = pd.DataFrame(rows)
    n_tested = len(R[R.half == "앞"])
    say(f"\n## 시험 범위: 구분 {len(sp)}개(연속 6특성 x 3문턱 = 18, group 12) x 범위 2 x 단위 2 = {n_tested} 조합(앞 절반에서 고름)")
    say("선택 편향: 120 조합 중 최대값을 고르므로 앞 절반 t는 부풀려져 있다. 판정은 뒤 절반만 본다.")

    # --- 고르기(앞 절반만 사용) ---
    F = R[R.half == "앞"].pivot_table(index=["feature", "split", "rule"], columns=["scope", "unit"], values="t")
    D = R[R.half == "앞"].pivot_table(index=["feature", "split", "rule"], columns=["scope", "unit"], values="diff")
    sg = np.sign(F)
    same = (sg.nunique(axis=1) == 1) & F.notna().all(axis=1)
    score = F.abs().min(axis=1).where(same, 0.0)
    S = pd.DataFrame({"score": score, "sign": sg.iloc[:, 0], "t_전체거래": F[("전체", "거래")], "t_전체종목": F[("전체", "종목")],
                      "t_정규거래": F[("정규", "거래")], "t_정규종목": F[("정규", "종목")],
                      "차_전체거래": D[("전체", "거래")], "차_정규거래": D[("정규", "거래")]}).sort_values("score", ascending=False)
    say("\n## 앞 절반 30 구분 전체 (점수 = 네 조합 t 부호가 같을 때 |t| 최솟값)")
    say("feature | 구분 | 문턱 | 점수 | t 전체·거래 | t 전체·종목 | t 정규·거래 | t 정규·종목 | 차 전체·거래 | 차 정규·거래")
    for (fe, lab, rule), r in S.iterrows():
        say(f"{fe} | {lab} | {rule} | {r.score:.2f} | {r['t_전체거래']:+.2f} | {r['t_전체종목']:+.2f} | {r['t_정규거래']:+.2f} | {r['t_정규종목']:+.2f} | "
            f"{pct(r['차_전체거래'])} | {pct(r['차_정규거래'])}")
    picked, fams = [], set()
    for (fe, lab, rule), r in S.iterrows():
        fam = FAMILY.get(fe, fe if fe != "group" else f"group:{lab}")
        if r.score > 0 and fam not in fams and len(picked) < 3:
            picked.append((fe, lab, rule, int(r.sign)))
            fams.add(fam)
    say(f"\n선택(점수 > 0, 특성 묶음당 1개, 최대 3개): {[(p[0], p[1], '우세' if p[3] > 0 else '열세') for p in picked]}")

    # --- 확인(뒤 절반) ---
    cand = []
    fnmap = {(n, l): fn for n, l, _, fn in sp}
    for fe, lab, rule, sign in picked:
        fn = fnmap[(fe, lab)]
        say(f"\n## 후보: {rule} (A=구분 안, 앞 절반 방향: A {'우세' if sign > 0 else '열세'})")
        res = {}
        for h, x in [("앞", first), ("뒤", second)]:
            for sc in ["전체", "정규"]:
                for un in ["거래", "종목"]:
                    xs = scope(x, sc)
                    r = summ(xs, fn(xs), "ret", un)
                    res[(h, sc, un)] = r
                    say(f"[{h}·{sc}·{un}] {fmt(r)}")
            c = controls(x, fn(x), sign, fe)
            res[(h, "ctrl")] = c
            say(f"[{h}·통제] {ctrl_text(c)}")
        b, br, bn = res[("뒤", "전체", "거래")], res[("뒤", "정규", "거래")], res[("뒤", "ctrl")]["nret"]
        if np.sign(b["dday"]) != sign:
            v = "뒤에서 실패"
        elif b["t"] * sign >= 2 and np.sign(br["dday"]) == sign and np.sign(bn["dday"]) == sign:
            v = "확인됨"
        else:
            v = "약함"
        mech = v
        sf = res[("뒤", "ctrl")]["sesfix"]
        if v == "확인됨" and (sf[1] * sign < 2 or br["t"] * sign < 1):
            v = "약함"
        say(f"판정: {v} (기계 판정 {mech}; 뒤 세션 고정 차 t={sf[1]:+.2f}, 정규·거래 t={br['t']:+.2f})")
        cand.append(dict(feature=fe, rule=rule, sign=sign, res=res, verdict=v))

    # --- 보조: 정규장만·거래 단위로 골랐다면 ---
    say("\n## 보조: 정규장만·거래 단위 앞 절반 |t| 상위 3개(특성 묶음당 1개) -> 뒤 절반 같은 문턱·방향")
    G = R[(R.scope == "정규") & (R.unit == "거래")].pivot_table(index=["feature", "split", "rule"], columns="half", values=["t", "diff", "dday"])
    G = G.reindex(G[("t", "앞")].abs().sort_values(ascending=False).index)
    fams, k = set(), 0
    for (fe, lab, rule), r in G.iterrows():
        fam = FAMILY.get(fe, fe if fe != "group" else f"group:{lab}")
        if fam in fams or k >= 3:
            continue
        fams.add(fam); k += 1
        sg = np.sign(r[("t", "앞")])
        fn = fnmap[(fe, lab)]
        xa = scope(second, "전체")
        ra = summ(xa, fn(xa))
        say(f"{rule}: 앞 정규 차 {pct(r[('diff', '앞')])} t={r[('t', '앞')]:+.2f} -> 뒤 정규 차 {pct(r[('diff', '뒤')])} t={r[('t', '뒤')]:+.2f} "
            f"({'같은 방향' if np.sign(r[('dday', '뒤')]) == sg else '반대'}), 뒤 전체 차 {pct(ra['diff'])} t={ra['t']:+.2f}")
        for h, x in [("앞", first), ("뒤", second)]:
            vv = x[x.ses == "정규"]
            say(f"   [{h}·정규] {fmt(summ(vv, fn(vv)))}")
        vv = second[second.ses == "정규"]
        c = controls(vv, fn(vv), sg, fe)
        say(f"   뒤 정규 통제: 수익/atr {c['nret']['diff']:+.3f}ATR t={c['nret']['t']:+.2f} | 1종목({c['d_sym'][0]}) 빼면 {pct(c['d_sym'][1])} t={c['d_sym'][2]:+.2f}(k={c['d_sym'][3]}) | "
            f"1일({c['d_day'][0]}) 빼면 {pct(c['d_day'][1])} t={c['d_day'][2]:+.2f}(k={c['d_day'][3]})")

    # --- 설명용: 종목별 평균과 특성의 순위상관(날짜 군집 아님) ---
    say("\n## 설명용: 종목별 거래당 평균 수익과 특성의 스피어만(앞/뒤, 전체/정규, vol20 순위 통제 편상관)")
    for sc in ["전체", "정규"]:
        for h, x in [("앞", first), ("뒤", second)]:
            m = scope(x, sc).groupby("sym").agg(ret=("ret", "mean"), nret=("nret", "mean")).join(f.set_index("sym")[CONT])
            rk = m.rank()
            parts = []
            for c in CONT:
                rho = rk.ret.corr(rk[c])
                if c in ("vol20", "atr14"):
                    parts.append(f"{c} {rho:+.2f}(수익/atr {rk.nret.corr(rk[c]):+.2f})")
                    continue
                e1 = rk.ret - np.polyval(np.polyfit(rk.vol20, rk.ret, 1), rk.vol20)
                e2 = rk[c] - np.polyval(np.polyfit(rk.vol20, rk[c], 1), rk.vol20)
                parts.append(f"{c} {rho:+.2f}(편 {np.corrcoef(e1, e2)[0, 1]:+.2f})")
            say(f"[{sc}·{h}, {len(m)}종목] " + " | ".join(parts))

    # --- 세션 구성: 변동성 삼분위별 애프터 비중과 세션별 평균 ---
    say("\n## 설명용: vol20 삼분위별 세션 구성과 세션별 거래당 평균(전체 기간)")
    for v in [0, 1, 2]:
        g = t[t.vt == v]
        ss = g.groupby("ses").ret.agg(["count", "mean"])
        say(f"vol 삼분위 {v + 1}: {len(g)}건 애프터 {(g.ses == '애프터').mean() * 100:.0f}% | " +
            " ".join(f"{s} {ss.loc[s, 'mean'] * 100:+.2f}%({int(ss.loc[s, 'count'])})" for s in SES_ALL if s in ss.index) +
            f" | 수익/atr {g.nret.mean():+.3f}")

    R.to_csv(os.path.join(RES, "fam_vol_size_tests.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RES, "fam_vol_size.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return n_tested, cand


if __name__ == "__main__":
    main()
