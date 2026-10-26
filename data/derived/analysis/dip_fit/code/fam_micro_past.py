"""이격매수매도 종목 궁합 — 특성 묶음 '장중 성격·과거 궁합'(2026-10-09).

질문: 과거(7/14~9/8, 5분봉 근사)에 잘 통한 종목이 최근 1개월(매수일 9/10~10/8)에도 잘 통하는가, 그리고 그 차이가
변동성(vol20·atr14)이나 애프터 비중으로 설명되는가. 장중 성격 ac1·vr12·trend60도 같은 잣대로 본다.

특성(모두 9/9 종가까지 알 수 있음): ac1 vr12 trend60 early_avg early_sum early_n (features.csv)
  + early_reg_avg(같은 앞 기간 거래 중 정규장 매수만의 평균, stop_lab early를 다시 돌려 계산) — 정규장만 결과와 짝을 맞추려고 추가.
  + early_aft(앞 기간 거래 중 애프터 비중) — 후보가 아니라 통제용.
구분: 종목 136개 기준 상위 1/3 vs 하위 1/3(t3), 상위 1/2 vs 하위 1/2(h2). 문턱은 종목 특성 분포에서 정하므로 두 절반에 같다.
범위: 전체 세션 / 정규장만. 시험 조합 = 7 특성 × 2 구분 × 2 범위 = 28(앞 절반에서 고르기). 뒤 절반은 고른 것만 확인(+ 핵심 질문 early_avg·early_sum t3는 고르기와 무관하게 보고).
고르기 점수(미리 정함): 앞 절반 날짜 t(전체)와 날짜 t(정규)의 평균 절댓값, 특성별로 더 강한 구분 하나 → 상위 3 특성. 방향 = 그 평균의 부호.
t값: 날짜 군집 — 같은 날 '상' 거래 평균 - '하' 거래 평균(둘 다 있는 날), 날짜 수로 t.
통제: ①수익/atr14 ②날짜×세션×vol20 삼분위 평균을 뺀 초과수익 ③회귀 ret ~ 상 더미 + log vol20 + 세션 더미 + 날짜 고정효과(날짜 군집 SE)
      ④세션 비중을 전체 비중으로 맞춘 평균 ⑤상위 1종목·상위 1일 빼기(차이를 가장 줄이는 것) ⑥꼬리 자름(±10%)
      ⑦세션별 분해와 '같은 봉 되돌림'(매수 봉 종가가 매수가보다 1% 이상 위 — 얇은 호가 꼬리 체결 흔적, 사후 값이라 설명용)
판정(미리 정함): 뒤 절반 전체 같은 방향 t>=2 & 정규만 같은 방향 t>=1 & 날짜·세션·vol 초과 같은 방향 t>=1.5 → 확인됨,
      뒤 절반 전체 반대 방향 또는 t<0.5 → 뒤에서 실패, 나머지 → 약함.
저장: ../results/fam_micro_past.txt, ../results/fam_micro_past.csv(모든 시험)
사용: py fam_micro_past.py"""
import os, sys, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))

FEATS = ["ac1", "vr12", "trend60", "early_avg", "early_sum", "early_n", "early_reg_avg"]
SPLITS = {"t3": (1 / 3, 2 / 3), "h2": (0.5, 0.5)}
SCOPES = {"전체": None, "정규": "정규"}
HALVES = ["앞", "뒤"]
OUT = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=OUT)


def load():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    import stop_lab
    e = stop_lab.run(stop_lab.load("early"), None, "early")
    f["early_reg_avg"] = f.sym.map(e[e.ses == "정규"].groupby("sym").ret.mean())
    f["early_reg_n"] = f.sym.map(e[e.ses == "정규"].groupby("sym").size()).fillna(0)
    f["early_aft"] = f.sym.map((e.ses == "애프터").groupby(e.sym).mean())
    f["volT"] = pd.qcut(f.vol20, 3, labels=["저", "중", "고"]).astype(str)
    miss = sorted(set(t.sym) - set(f.sym))
    miss = (miss, int(t.sym.isin(miss).sum()))
    t = t[t.sym.isin(f.sym)].merge(f[["sym", "vol20", "atr14", "volT", "early_aft"] + FEATS], on="sym")
    t["ret_atr"] = t.ret / t.atr14
    t["adj"] = t.ret - t.groupby(["day", "ses", "volT"]).ret.transform("mean")
    t["ret_clip"] = t.ret.clip(-0.10, 0.10)
    return t, f, e, miss


def grp(f, feat, split):
    """종목 → '상'/'하'/'중'. 문턱은 종목 특성 분포(9/9 기준)."""
    x = f.set_index("sym")[feat].dropna()
    lo, hi = x.quantile(SPLITS[split][0]), x.quantile(SPLITS[split][1])
    g = pd.Series("중", index=x.index)
    g[x <= lo] = "하"
    g[x > hi] = "상"
    if split == "h2":
        g[x > lo] = "상"
    return g


def day_t(tt, g, col="ret"):
    gg = tt.sym.map(g)
    a = tt[gg == "상"].groupby("day")[col].mean()
    b = tt[gg == "하"].groupby("day")[col].mean()
    d = (a - b).dropna()
    if len(d) < 3 or d.std(ddof=1) == 0:
        return np.nan, len(d), d
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), len(d), d


def diff(tt, g, col="ret"):
    gg = tt.sym.map(g)
    return tt[gg == "상"][col].mean() - tt[gg == "하"][col].mean()


def sess_std(tt, g, w):
    """세션 비중을 w(전체 세션 비중)로 맞춘 평균 차이."""
    gg = tt.sym.map(g)
    m = tt.assign(g=gg).groupby(["g", "ses"]).ret.mean().unstack()
    m = m.reindex(columns=w.index)
    return (m.loc["상"] * w).sum() - (m.loc["하"] * w).sum() if {"상", "하"} <= set(m.index) else np.nan


def reg(tt, g):
    """ret ~ 상 + log vol20 + 세션 더미 + 날짜 고정효과, 날짜 군집 SE. '상'/'하' 거래만."""
    gg = tt.sym.map(g)
    x = tt[gg.isin(["상", "하"])].copy()
    x["hi"] = (x.sym.map(g) == "상").astype(float)
    X = pd.concat([x[["hi"]], np.log(x.vol20).rename("lv"), pd.get_dummies(x.ses, drop_first=True, dtype=float),
                   pd.get_dummies(x.day, drop_first=False, dtype=float)], axis=1).values.astype(float)
    y = x.ret.values
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    days = x.day.values
    G = 0
    for d in np.unique(days):
        k = days == d
        s = X[k].T @ u[k]
        meat += np.outer(s, s)
        G += 1
    n, K = X.shape
    V = XtX @ meat @ XtX * G / (G - 1) * (n - 1) / max(n - K, 1)
    return b[0], b[0] / np.sqrt(V[0, 0]) if V[0, 0] > 0 else np.nan


def drop_one(tt, g, sign, key):
    """key(sym/day) 하나를 빼서 차이(방향 부호 곱)가 가장 작아지는 경우."""
    best = (np.inf, None)
    for v in tt[key].unique():
        r = diff(tt[tt[key] != v], g) * sign
        if r < best[0]:
            best = (r, v)
    return best[0] * sign, best[1]


def block(tt, g, w, sign=None):
    gg = tt.sym.map(g)
    H, L = tt[gg == "상"], tt[gg == "하"]
    tv, nd, _ = day_t(tt, g)
    d = H.ret.mean() - L.ret.mean()
    s = np.sign(d) if sign is None else sign
    r = dict(n_hi=len(H), sym_hi=H.sym.nunique(), m_hi=H.ret.mean(), n_lo=len(L), sym_lo=L.sym.nunique(), m_lo=L.ret.mean(),
             diff=d, t=tv, ndays=nd, med_hi=H.ret.median(), med_lo=L.ret.median(),
             aft_hi=(H.ses == "애프터").mean(), aft_lo=(L.ses == "애프터").mean(),
             sum_per_sym_hi=H.ret.sum() / max(H.sym.nunique(), 1), sum_per_sym_lo=L.ret.sum() / max(L.sym.nunique(), 1),
             vol_hi=H.vol20.mean(), vol_lo=L.vol20.mean())
    r["d_atr"], r["t_atr"] = diff(tt, g, "ret_atr"), day_t(tt, g, "ret_atr")[0]
    r["d_adj"], r["t_adj"] = diff(tt, g, "adj"), day_t(tt, g, "adj")[0]
    r["d_clip"], r["t_clip"] = diff(tt, g, "ret_clip"), day_t(tt, g, "ret_clip")[0]
    r["d_sess"] = sess_std(tt, g, w)
    r["b_reg"], r["t_reg"] = reg(tt, g)
    r["d_drop_sym"], r["drop_sym"] = drop_one(tt[gg.isin(["상", "하"])], g, s, "sym")
    r["d_drop_day"], r["drop_day"] = drop_one(tt[gg.isin(["상", "하"])], g, s, "day")
    return r


def same_bar_rec(t):
    """매수 봉(3분봉) 종가 / 매수가 - 1. 사후 값이라 설명용으로만 쓴다(build.py와 같은 봉 데이터)."""
    import build as B
    out = pd.Series(np.nan, index=t.index)
    for s, x in t.groupby("sym"):
        d = B.prep(B.bars(B.RAW, s, 3), 60, 14)
        pos = d.index.get_indexer(pd.to_datetime(x.t.values, utc=True).tz_convert(d.index.tz))
        c = d.c.values
        out[x.index] = np.where(pos >= 0, c[np.maximum(pos, 0)] / x.ent.values - 1, np.nan)
    return out


def pct(x):
    return f"{x * 100:+.2f}%" if pd.notna(x) else "N/A"


def line(r):
    return (f"상 {pct(r['m_hi'])}({r['n_hi']}건/{r['sym_hi']}종목) 하 {pct(r['m_lo'])}({r['n_lo']}건/{r['sym_lo']}종목) "
            f"차 {pct(r['diff'])} t={r['t']:.2f}({r['ndays']}일)")


def main():
    t, f, e, miss = load()
    w_all = t.ses.value_counts(normalize=True)
    p(f"거래 {len(t)}건 {t.sym.nunique()}종목 {t.day.nunique()}일 (특성 없는 종목 제외: {miss[0]}, {miss[1]}건)")
    p(f"앞 기간(7/14~9/8) 재계산 거래 {len(e)}건 — features.csv early_n 합 {int(f.early_n.sum())}건과 대조")
    p(f"특성 스피어만(종목 136): early_avg~vol20 {f.early_avg.corr(f.vol20, method='spearman'):.2f}, "
      f"early_avg~early_aft {f.early_avg.corr(f.early_aft, method='spearman'):.2f}, early_reg_avg~vol20 {f.early_reg_avg.corr(f.vol20, method='spearman'):.2f}, "
      f"ac1~vr12 {f.ac1.corr(f.vr12, method='spearman'):.2f}")
    p("")

    # 1) 종목 단위 순위 상관(설명용): 특성 vs 최근 두 절반 종목 평균
    p("[1] 종목 단위 스피어만: 특성 vs 그 절반 종목별 거래당 평균(거래 2건 이상 종목)")
    for feat in FEATS + ["vol20", "early_aft"]:
        s = []
        for h in HALVES:
            for sc, ses in SCOPES.items():
                x = t[(t.half == h) & ((t.ses == ses) if ses else True)]
                m = x.groupby("sym").ret.agg(["mean", "count"])
                m = m[m["count"] >= 2]
                a = x.groupby("sym").adj.mean().reindex(m.index)
                fv = f.set_index("sym")[feat].reindex(m.index)
                s.append(f"{h}/{sc} {fv.corr(m['mean'], method='spearman'):+.2f}(초과 {fv.corr(a, method='spearman'):+.2f}, n{len(m)})")
        p(f"  {feat:13s} " + " | ".join(s))
    p("")

    # 2) 앞 절반 고르기: 7 특성 × 2 구분 × 2 범위 = 28
    rows = []
    for feat in FEATS:
        for sp in SPLITS:
            g = grp(f, feat, sp)
            for sc, ses in SCOPES.items():
                for h in HALVES:
                    x = t[(t.half == h) & ((t.ses == ses) if ses else True)]
                    r = block(x, g, w_all if ses is None else pd.Series({"정규": 1.0}))
                    rows.append(dict(feat=feat, split=sp, scope=sc, half=h, **r))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "fam_micro_past.csv"), index=False, encoding="utf-8-sig")
    first = R[R.half == "앞"].pivot_table(index=["feat", "split"], columns="scope", values="t")
    first["score"] = (first["전체"] + first["정규"]) / 2
    best = first.reset_index().assign(a=lambda d: d.score.abs()).sort_values("a", ascending=False).groupby("feat").head(1)
    p("[2] 앞 절반 고르기(28조합: 7특성×2구분×2범위). 점수 = (날짜t 전체 + 날짜t 정규)/2")
    for _, r in first.reset_index().iterrows():
        p(f"  {r.feat:13s} {r.split}  t전체 {r['전체']:+.2f}  t정규 {r['정규']:+.2f}  점수 {r.score:+.2f}")
    picks = best.head(3)
    p("  → 고른 후보: " + ", ".join(f"{r.feat}/{r.split}(점수 {r.score:+.2f}, 방향 {'상>하' if r.score > 0 else '하>상'})" for _, r in picks.iterrows()))
    p("")

    # 3) 모든 조합의 두 절반 결과(전체·정규)
    p("[3] 모든 조합 두 절반(거래당 평균, 날짜 t)")
    for (feat, sp, sc), d in R.groupby(["feat", "split", "scope"], sort=False):
        a, b = d[d.half == "앞"].iloc[0], d[d.half == "뒤"].iloc[0]
        p(f"  {feat:13s} {sp} {sc}: 앞 {line(a)} || 뒤 {line(b)}")
    p("")

    # 4) 후보 + 핵심 질문 상세
    detail = [(r.feat, r.split, np.sign(r.score)) for _, r in picks.iterrows()]
    for k in [("early_avg", "t3"), ("early_sum", "t3"), ("early_reg_avg", "t3")]:
        if k not in [(a, b) for a, b, _ in detail]:
            sg = np.sign(first.loc[k, "score"])
            detail.append((k[0], k[1], sg))
    p("[4] 상세(후보 3 + 핵심 질문). 방향은 앞 절반 점수 부호로 고정")
    for feat, sp, sg in detail:
        g = grp(f, feat, sp)
        p(f"== {feat} {sp} (앞에서 정한 방향: {'상>하' if sg > 0 else '하>상'}) 문턱 {f[feat].quantile(SPLITS[sp][0]):.4g}/{f[feat].quantile(SPLITS[sp][1]):.4g}")
        for sc, ses in SCOPES.items():
            for h in HALVES:
                x = t[(t.half == h) & ((t.ses == ses) if ses else True)]
                r = block(x, g, w_all if ses is None else pd.Series({"정규": 1.0}), sg)
                p(f"  {h}/{sc}: {line(r)} 중앙 {pct(r['med_hi'])}/{pct(r['med_lo'])} 종목당합 {pct(r['sum_per_sym_hi'])}/{pct(r['sum_per_sym_lo'])}")
                p(f"     애프터비중 {r['aft_hi']:.0%}/{r['aft_lo']:.0%} vol20 {r['vol_hi']:.3f}/{r['vol_lo']:.3f} | 수익/atr 차 {r['d_atr']:+.3f} t={r['t_atr']:.2f}"
                  f" | 날짜·세션·vol삼분위 초과 차 {pct(r['d_adj'])} t={r['t_adj']:.2f} | 회귀 {pct(r['b_reg'])} t={r['t_reg']:.2f}"
                  f" | 세션비중 맞춤 차 {pct(r['d_sess'])} | ±10%자름 차 {pct(r['d_clip'])} t={r['t_clip']:.2f}"
                  f" | 1종목 빼기 {pct(r['d_drop_sym'])}({r['drop_sym']}) 1일 빼기 {pct(r['d_drop_day'])}({r['drop_day']})")
        # vol 삼분위 안에서
        s = []
        for vt in ["저", "중", "고"]:
            for h in HALVES:
                x = t[(t.half == h) & (t.volT == vt)]
                gg = x.sym.map(g)
                s.append(f"{vt}/{h} {pct(x[gg == '상'].ret.mean())}({(gg == '상').sum()})-{pct(x[gg == '하'].ret.mean())}({(gg == '하').sum()})")
        p("     vol20 삼분위 안(전체 세션, 상-하): " + " ".join(s))
        s = []
        for vt in ["저", "중", "고"]:
            for h in HALVES:
                x = t[(t.half == h) & (t.volT == vt) & (t.ses == "정규")]
                gg = x.sym.map(g)
                s.append(f"{vt}/{h} {pct(x[gg == '상'].ret.mean())}({(gg == '상').sum()})-{pct(x[gg == '하'].ret.mean())}({(gg == '하').sum()})")
        p("     vol20 삼분위 안(정규만, 상-하): " + " ".join(s))
        p("")

    # 5) 참고: 변동성 자체(vol20 t3)
    p("[5] 참고(후보 아님): vol20 t3 상(고변동)-하(저변동)")
    g = grp(f, "vol20", "t3")
    for sc, ses in SCOPES.items():
        for h in HALVES:
            x = t[(t.half == h) & ((t.ses == ses) if ses else True)]
            r = block(x, g, w_all if ses is None else pd.Series({"정규": 1.0}))
            p(f"  {h}/{sc}: {line(r)} 수익/atr 차 {r['d_atr']:+.3f} t={r['t_atr']:.2f} 애프터 {r['aft_hi']:.0%}/{r['aft_lo']:.0%}")
    p("")

    # 6) 세션별 분해 + 같은 봉 되돌림(설명용, 사후 값): 매수 봉 종가가 매수가보다 1% 이상 위 = 얇은 호가 꼬리 체결 흔적
    t["rec"] = same_bar_rec(t)
    p("[6] 세션별 분해(앞에서 정한 방향 그대로). 되돌림1% = 매수 봉 종가 >= 매수가 x1.01 비율(사후, 설명용)")
    for feat, sp, sg in detail:
        g = grp(f, feat, sp)
        p(f"== {feat} {sp}")
        for ses in ["프리", "애프터", "정규"]:
            for h in HALVES:
                x = t[(t.half == h) & (t.ses == ses)]
                gg = x.sym.map(g)
                H, L = x[gg == "상"], x[gg == "하"]
                tv, nd, _ = day_t(x, g)
                ta = day_t(x, g, "adj")[0]
                b, tb = reg(x, g)
                z = x[x.rec < 0.01]
                tz = day_t(z, g)[0]
                zg = z.sym.map(g)
                p(f"  {ses}/{h}: 상 {pct(H.ret.mean())}({len(H)}) 하 {pct(L.ret.mean())}({len(L)}) t={tv:.2f}({nd}일) 승률 {(H.ret > 0).mean():.0%}/{(L.ret > 0).mean():.0%}"
                  f" | 초과 차 {pct(diff(x, g, 'adj'))} t={ta:.2f} | 회귀(vol·날짜) {pct(b)} t={tb:.2f} | 되돌림1% {(H.rec >= 0.01).mean():.0%}/{(L.rec >= 0.01).mean():.0%}"
                  f" | 되돌림 없는 거래만 상 {pct(z[zg == '상'].ret.mean())}({(zg == '상').sum()}) 하 {pct(z[zg == '하'].ret.mean())}({(zg == '하').sum()}) t={tz:.2f}")
    p("")

    # 7) 판정(미리 정한 기준): 뒤 절반 전체 같은 방향 t>=2, 정규만 같은 방향 t>=1, 날짜·세션·vol 초과 같은 방향 t>=1.5 → 확인됨
    #    뒤 절반 전체가 반대 방향이거나 |t|<0.5 → 뒤에서 실패, 나머지 → 약함
    p("[7] 판정(후보 3 + 핵심 질문)")
    for feat, sp, sg in detail:
        g = grp(f, feat, sp)
        rr = {}
        for sc, ses in SCOPES.items():
            x = t[(t.half == "뒤") & ((t.ses == ses) if ses else True)]
            rr[sc] = block(x, g, w_all if ses is None else pd.Series({"정규": 1.0}), sg)
        ta, tr, tad = rr["전체"]["t"] * sg, rr["정규"]["t"] * sg, rr["전체"]["t_adj"] * sg
        v = "뒤에서 실패" if ta < 0.5 else ("확인됨" if (ta >= 2 and tr >= 1 and tad >= 1.5) else "약함")
        p(f"  {feat} {sp}: 뒤 전체 t {ta:+.2f}, 정규 t {tr:+.2f}, 초과 t {tad:+.2f} → {v}")
    p("")

    # 8) 추가 시험(고르기 격자 밖, 공개): 앞 기간 '프리장 매수만' 평균(early_pre_avg, 9/8까지 알 수 있음) → 최근 프리/정규/전체
    f["early_pre_avg"] = f.sym.map(e[e.ses == "프리"].groupby("sym").ret.mean())
    g = grp(f, "early_pre_avg", "t3")
    p("[8] 추가 시험: early_pre_avg t3(앞 기간 프리장 거래만 평균) 상-하")
    for sc, ses in [("프리", "프리"), ("정규", "정규"), ("전체", None)]:
        for h in HALVES:
            x = t[(t.half == h) & ((t.ses == ses) if ses else True)]
            r = block(x, g, w_all if ses is None else pd.Series({ses: 1.0}), 1.0)
            p(f"  {sc}/{h}: {line(r)} | 초과 차 {pct(r['d_adj'])} t={r['t_adj']:.2f} | 회귀 {pct(r['b_reg'])} t={r['t_reg']:.2f}"
              f" | 1종목 빼기 {pct(r['d_drop_sym'])}({r['drop_sym']}) 1일 빼기 {pct(r['d_drop_day'])}({r['drop_day']})")
    p("")
    p("시험 수: 고르기 격자 28(7특성×2구분×2범위) + 세션 분해 18(상세 6개×프리·애프터·정규) + 추가 3(early_pre_avg×3범위) + 참고 vol20 2 = 51")
    with open(os.path.join(RES, "fam_micro_past.txt"), "w", encoding="utf-8") as fh:
        fh.write(OUT.getvalue())


if __name__ == "__main__":
    main()
