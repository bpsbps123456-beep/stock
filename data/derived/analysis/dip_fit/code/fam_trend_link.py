"""이격매수매도 종목 궁합 — 특성 묶음 '추세·시장 연동'(beta60 corr60 ret20 ret60 dist52 ma50, 거래 단위 보조 sdev_q).

질문: 시장(QQQ)과 같이 움직이는 종목 vs 혼자 움직이는 종목, 9/9 기준 강세 vs 약세 종목에서 규칙 성적이 다른가.
입력: ../results/trades.csv, ../results/features.csv (build.py가 만든 것, 다시 받지 않는다)
방법:
  - 구분(split) = 특성 x 문턱. 종목 특성 문턱은 136종목 단면 분위(9/9 값, 성과와 무관), sdev_q 문턱은 앞 절반 거래 분위.
    문턱: 상위1/3 vs 나머지(T3), 하위1/3 vs 나머지(B3), 상위1/3 vs 하위1/3(T3vB3), 중앙값 위/아래(M), 0 위/아래(Z: ret20 ret60 ma50만)
  - 범위 = 전체 세션 / 정규장만. 앞 절반(9/10~9/23)에서 방향을 정하고 그대로 뒤 절반(9/24~10/8)에 적용.
  - t값 = 날짜 군집: 날짜별 (A 평균 - B 평균)의 평균 / 표준오차.
  - 고르기 점수 = 앞 절반 min(|t 전체|, |t 정규|) (두 범위 부호가 같을 때만), 특성이 다른 상위 3개.
  - 통제: 수익/atr14, vol20 삼분위 안 차이, 상위 1종목·1일 제외, 애프터 비중(세션 구성 효과·세션 표준화 차이), 꼬리 ±10% 자름.
출력: ../results/fam_trend_link.txt, ../results/fam_trend_link_tests.csv
사용: py fam_trend_link.py"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
FEATS = ["beta60", "corr60", "ret20", "ret60", "dist52", "ma50"]
ZERO = {"ret20", "ret60", "ma50"}
TRADE_FEATS = ["sdev_q"]
SES = ["프리", "정규", "애프터"]
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def load():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    miss = sorted(set(t.sym) - set(f.sym))
    t = t.merge(f[["sym", "vol20", "atr14"] + FEATS], on="sym", how="inner")
    t["r_atr"] = t.ret / t.atr14
    t["ret_clip"] = t.ret.clip(-0.10, 0.10)
    q = f.vol20.quantile([1 / 3, 2 / 3]).values
    f["vt"] = np.where(f.vol20 <= q[0], "저", np.where(f.vol20 >= q[1], "고", "중"))
    t = t.merge(f[["sym", "vt"]], on="sym")
    return t, f, miss


def splits(f, t):
    """(이름, 특성, 문턱, A 판정 함수, B 판정 함수) 목록. 문턱은 결과를 보지 않고 정한다."""
    out = []
    for c in FEATS:
        q33, q50, q67 = f[c].quantile([1 / 3, 0.5, 2 / 3]).values
        out += [(c, "T3", f"상위1/3(>={q67:.3g}) vs 나머지", lambda x, v=q67: x >= v, lambda x, v=q67: x < v),
                (c, "B3", f"하위1/3(<={q33:.3g}) vs 나머지", lambda x, v=q33: x <= v, lambda x, v=q33: x > v),
                (c, "T3vB3", f"상위1/3(>={q67:.3g}) vs 하위1/3(<={q33:.3g})", lambda x, v=q67: x >= v, lambda x, v=q33: x <= v),
                (c, "M", f"중앙값(>={q50:.3g}) 위 vs 아래", lambda x, v=q50: x >= v, lambda x, v=q50: x < v)]
        if c in ZERO:
            out.append((c, "Z", "0 초과 vs 이하", lambda x: x > 0, lambda x: x <= 0))
    a = t[t.half == "앞"]
    for c in TRADE_FEATS:
        q33, q50, q67 = a[c].quantile([1 / 3, 0.5, 2 / 3]).values
        out += [(c, "T3", f"상위1/3(>={q67:.3g}) vs 나머지", lambda x, v=q67: x >= v, lambda x, v=q67: x < v),
                (c, "B3", f"하위1/3(<={q33:.3g}) vs 나머지", lambda x, v=q33: x <= v, lambda x, v=q33: x > v),
                (c, "T3vB3", f"상위1/3(>={q67:.3g}) vs 하위1/3(<={q33:.3g})", lambda x, v=q67: x >= v, lambda x, v=q33: x <= v),
                (c, "M", f"중앙값(>={q50:.3g}) 위 vs 아래", lambda x, v=q50: x >= v, lambda x, v=q50: x < v)]
    return out


def dstat(A, B, col="ret"):
    ma, mb = A.groupby("day")[col].mean(), B.groupby("day")[col].mean()
    d = (ma - mb).dropna()
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else np.nan
    t = d.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    ew = A.groupby("sym")[col].mean().mean() - B.groupby("sym")[col].mean().mean() if len(A) and len(B) else np.nan
    return dict(mA=A[col].mean(), mB=B[col].mean(), diff=A[col].mean() - B[col].mean(), nA=len(A), nB=len(B),
                sA=A.sym.nunique(), sB=B.sym.nunique(), dmean=d.mean() if n else np.nan, t=t, nd=n, ew=ew)


def parts(t, sp):
    c, _, _, fa, fb = sp
    return t[fa(t[c])], t[fb(t[c])]


def scope(t, sc):
    return t if sc == "전체" else t[t.ses == "정규"]


def drop_top(A, B, sign, key):
    """sign 방향 차이를 가장 많이 만드는 key 하나(종목 또는 날짜)를 빼고 다시 잰다."""
    best, bv = None, np.inf
    for k in pd.unique(pd.concat([A[key], B[key]])):
        a, b = A[A[key] != k], B[B[key] != k]
        if len(a) == 0 or len(b) == 0:
            continue
        v = sign * (a.ret.mean() - b.ret.mean())
        if v < bv:
            best, bv = k, v
    a, b = A[A[key] != best], B[B[key] != best]
    s = dstat(a, b)
    return best, s


def ses_check(A, B, T):
    """애프터 비중과 세션 구성 효과. m_s는 그 절반·전체 거래의 세션 평균."""
    m = T.groupby("ses").ret.mean()
    w = T.ses.value_counts(normalize=True)
    pa, pb = A.ses.value_counts(normalize=True), B.ses.value_counts(normalize=True)
    mix = sum((pa.get(s, 0) - pb.get(s, 0)) * m.get(s, 0) for s in SES)
    std = sum(w.get(s, 0) * (A[A.ses == s].ret.mean() - B[B.ses == s].ret.mean()) for s in SES
              if (A.ses == s).any() and (B.ses == s).any())
    per = {s: (A[A.ses == s].ret.mean() - B[B.ses == s].ret.mean(), int((A.ses == s).sum()), int((B.ses == s).sum())) for s in SES}
    return pa.get("애프터", 0), pb.get("애프터", 0), mix, std, per


def fmt(s, pct=True):
    k = 100 if pct else 1
    u = "%" if pct else ""
    return (f"A {s['mA']*k:+.2f}{u}({s['nA']}건·{s['sA']}종목) B {s['mB']*k:+.2f}{u}({s['nB']}건·{s['sB']}종목) "
            f"차 {s['diff']*k:+.2f}{u} 날짜t {s['t']:+.2f}(n{s['nd']}) 종목동일가중차 {s['ew']*k:+.2f}{u}")


def examine(t, sp, sign, tag=""):
    """고른 구분 하나를 앞·뒤 x 전체·정규로 재고 통제와 판정을 붙인다."""
    lab = "A 우세" if sign > 0 else "B 우세"
    say(f"### {tag}{sp[0]} {sp[1]}: A={sp[2].split(' vs ')[0]}, B={sp[2].split(' vs ')[1]} — 앞 절반 날짜t 기준 {lab}")
    res = {}
    for h in ["앞", "뒤"]:
        for sc in ["전체", "정규"]:
            A, B = parts(scope(t[t.half == h], sc), sp)
            res[(h, sc)] = dstat(A, B)
            say(f"  {h}·{sc}: {fmt(res[(h, sc)])}")
    ctrl = {}
    for h in ["앞", "뒤"]:
        Th = t[t.half == h]
        for sc in ["전체", "정규"]:
            T = scope(Th, sc)
            A, B = parts(T, sp)
            sa = dstat(A, B, "r_atr")
            sclip = dstat(A, B, "ret_clip")
            vt = []
            for v in ["저", "중", "고"]:
                a, b = A[A.vt == v], B[B.vt == v]
                if len(a) >= 5 and len(b) >= 5:
                    vt.append((v, a.ret.mean() - b.ret.mean(), len(a), len(b)))
            vavg = np.mean([x[1] for x in vt]) if vt else np.nan
            ts, s_ts = drop_top(A, B, sign, "sym")
            td, s_td = drop_top(A, B, sign, "day")
            c = dict(atr=sa, clip=sclip, vt=vt, vavg=vavg, ts=(ts, s_ts), td=(td, s_td))
            say(f"  [{h}·{sc}] 수익/atr14 차 {sa['diff']:+.3f} t {sa['t']:+.2f} | 꼬리±10%자름 차 {sclip['diff']*100:+.2f}% t {sclip['t']:+.2f}"
                f" | vol20 삼분위 안 차 " + ", ".join(f"{v} {d*100:+.2f}%({na}/{nb})" for v, d, na, nb in vt) + f" → 평균 {vavg*100:+.2f}%")
            say(f"  [{h}·{sc}] 상위1종목({ts}) 빼면 차 {s_ts['diff']*100:+.2f}% t {s_ts['t']:+.2f} | 상위1일({td}) 빼면 차 {s_td['diff']*100:+.2f}% t {s_td['t']:+.2f}")
            if sc == "전체":
                pa, pb, mix, std, per = ses_check(A, B, T)
                c.update(pa=pa, pb=pb, mix=mix, std=std)
                say(f"  [{h}·전체] 애프터 비중 A {pa:.0%} B {pb:.0%} | 세션 구성만으로 설명되는 차 {mix*100:+.2f}% (실제 차 {res[(h, '전체')]['diff']*100:+.2f}%)"
                    f" | 세션 표준화 차 {std*100:+.2f}% | 세션별 차 " + ", ".join(f"{s} {d*100:+.2f}%({na}/{nb})" for s, (d, na, nb) in per.items()))
            ctrl[(h, sc)] = c
    # 판정(뒤 절반, 앞 방향 고정). 규칙: 한 범위라도 같은방향t<=-1 이거나 둘 다 <0.5면 실패,
    # 전체 t>=2 · 정규 t>=1.5 · 두 범위 통제(수익/atr, vol 삼분위 평균, 상위1종목·1일 제외) 같은 방향 · 세션 구성 설명 <50% 이면 확인됨, 나머지 약함
    ta, tr = sign * res[("뒤", "전체")]["t"], sign * res[("뒤", "정규")]["t"]
    def okc(c):
        return all(sign * x > 0 for x in [c["atr"]["diff"], c["vavg"], c["ts"][1]["diff"], c["td"][1]["diff"]])
    cf, cr = ctrl[("뒤", "전체")], ctrl[("뒤", "정규")]
    mix_share = cf["mix"] / res[("뒤", "전체")]["diff"] if res[("뒤", "전체")]["diff"] else np.nan
    if ta <= -1 or tr <= -1 or (ta < 0.5 and tr < 0.5):
        v = "뒤에서 실패"
    elif ta >= 2 and tr >= 1.5 and okc(cf) and okc(cr) and sign * cf["std"] > 0 and mix_share < 0.5:
        v = "확인됨"
    else:
        v = "약함"
    say(f"  판정: {v} (뒤 같은방향 t 전체 {ta:+.2f}, 정규 {tr:+.2f} | 통제 같은 방향 전체={okc(cf)} 정규={okc(cr)} | 세션 구성 설명 비율 {mix_share:.0%})")
    say("")
    return dict(sp=sp, res=res, ctrl=ctrl, verdict=v, sign=sign, ta=ta, tr=tr, mix_share=mix_share)


def main():
    t, f, miss = load()
    say("# 이격매수매도 종목 궁합 — 추세·시장 연동 묶음 (fam_trend_link)")
    say(f"거래 {len(t)}건 / {t.sym.nunique()}종목 (특성 없는 종목 제외: {miss}), 앞 절반 {t[t.half=='앞'].day.nunique()}일 · 뒤 절반 {t[t.half=='뒤'].day.nunique()}일")
    say("특성 시점: beta60 corr60 ret20 ret60 dist52 ma50 = 9/9 종가까지(사전). sdev_q = 매수 직전 봉(사전, 거래 단위 보조).")
    say("차 = A 평균 - B 평균(거래 단위). 날짜t = 날짜별 (A평균-B평균) 군집 t. 방향은 앞 절반 부호로 고정.")
    say("")

    # 0) 특성 간·변동성과의 관계(종목 단면 스피어만)
    say("## 0. 종목 단면 스피어만 (136종목, 9/9 값)")
    cm = f[FEATS + ["vol20", "atr14"]].corr(method="spearman")
    say(cm.round(2).to_string())
    say("")

    # 1) 종목 평균 수익과 특성의 스피어만(설명용, 절반별)
    say("## 1. 종목별 평균 수익 vs 특성 스피어만 (절반에 3건 이상인 종목)")
    rows = []
    for h in ["앞", "뒤"]:
        for sc in ["전체", "정규"]:
            x = scope(t[t.half == h], sc)
            g = x.groupby("sym").agg(n=("ret", "size"), m=("ret", "mean"), ma=("r_atr", "mean"))
            g = g[g.n >= 3].join(f.set_index("sym")[FEATS + ["vol20"]])
            r = {c: g.m.corr(g[c], method="spearman") for c in FEATS + ["vol20"]}
            ra = {c: g.ma.corr(g[c], method="spearman") for c in FEATS}
            rows.append(dict(절반=h, 범위=sc, 종목=len(g), **{c: round(r[c], 2) for c in FEATS + ["vol20"]},
                             **{c + "/atr": round(ra[c], 2) for c in FEATS}))
    say(pd.DataFrame(rows).to_string(index=False))
    say("")

    # 2) 전 조합 시험(앞·뒤 모두 기록, 고르기는 앞만)
    SP = splits(f, t)
    rec = []
    for sp in SP:
        for sc in ["전체", "정규"]:
            r = dict(feature=sp[0], cut=sp[1], rule=sp[2], scope=sc)
            for h in ["앞", "뒤"]:
                A, B = parts(scope(t[t.half == h], sc), sp)
                s = dstat(A, B)
                for k, v in s.items():
                    r[f"{h}_{k}"] = v
            rec.append(r)
    R = pd.DataFrame(rec)
    # 방향 = 앞 절반 날짜 군집 평균차(dmean) 부호. t와 같은 측도로 맞춘다(거래 묶음 평균차와 다를 수 있음).
    R["dir"] = np.sign(R["앞_dmean"])
    R["뒤_같은방향t"] = R["dir"] * R["뒤_t"]
    R["뒤_같은방향차"] = R["dir"] * R["뒤_diff"]
    R.to_csv(os.path.join(RES, "fam_trend_link_tests.csv"), index=False, encoding="utf-8-sig")
    n_split = len(SP)
    say(f"## 2. 시험 범위: 구분 {n_split}개(종목 특성 6개 x 문턱 4 + 0문턱 3 = 27, sdev_q x 4 = 4) x 범위 2(전체/정규) = {len(R)}조합")
    show = R[["feature", "cut", "scope", "앞_diff", "앞_t", "뒤_diff", "뒤_t", "뒤_같은방향t"]].copy()
    for c in ["앞_diff", "뒤_diff"]:
        show[c] = (show[c] * 100).round(2)
    for c in ["앞_t", "뒤_t", "뒤_같은방향t"]:
        show[c] = show[c].round(2)
    say("(차는 거래 묶음 평균차 %p, t는 날짜 군집, 뒤_같은방향t = 앞 절반 날짜t 부호로 본 뒤 절반 t)")
    say(show.to_string(index=False))
    say("")
    say(f"전 조합 뒤_같은방향t: 평균 {R['뒤_같은방향t'].mean():+.2f}, 양수 비율 {(R['뒤_같은방향t']>0).mean():.0%} "
        f"(조합끼리 겹쳐 독립이 아님), 앞 |t|>=2 조합 {(R['앞_t'].abs()>=2).sum()}/{len(R)} (우연 기대 약 {0.05*len(R):.0f}~{0.1*len(R):.0f}개, 날짜 10개라 t 꼬리 두꺼움)")
    for sc in ["전체", "정규"]:
        x = R[R.scope == sc]
        say(f"  {sc}: 뒤_같은방향t 평균 {x['뒤_같은방향t'].mean():+.2f}, 양수 {(x['뒤_같은방향t']>0).mean():.0%}, 뒤 |t|>=2 같은방향 {(x['뒤_같은방향t']>=2).sum()} / 반대 {(x['뒤_같은방향t']<=-2).sum()}")
    say("")

    # 3) 앞 절반에서 후보 고르기 (규칙은 결과 보기 전에 정함)
    W = R[["feature", "cut", "rule"]].drop_duplicates().copy()
    for sc, k in [("전체", "all"), ("정규", "reg")]:
        x = R[R.scope == sc].set_index(["feature", "cut"])
        W[f"d_{k}"] = [x.loc[(a, b), "앞_diff"] for a, b in zip(W.feature, W.cut)]
        W[f"t_{k}"] = [x.loc[(a, b), "앞_t"] for a, b in zip(W.feature, W.cut)]
    W["score"] = np.where(np.sign(W.t_reg) == np.sign(W.t_all), np.minimum(W.t_reg.abs(), W.t_all.abs()), 0.0)
    W = W.sort_values("score", ascending=False)
    say("## 3. 앞 절반 고르기 (점수 = min(|t 전체|,|t 정규|), 두 범위 날짜t 부호 같을 때만, 특성별 1개, 상위 3개)")
    say(W.head(10).round(3).to_string(index=False))
    pick, seen = [], set()
    for _, r in W.iterrows():
        if r.feature in seen or r.score <= 0:
            continue
        pick.append(r)
        seen.add(r.feature)
        if len(pick) == 3:
            break
    say("")

    out = []
    say("## 4. 고른 후보의 뒤 절반 확인과 통제 (A/B 정의는 앞 절반 그대로)")
    for r in pick:
        sp = next(s for s in SP if s[0] == r.feature and s[1] == r.cut)
        e = examine(t, sp, np.sign(r.t_all))
        e["r"] = r
        out.append(e)

    # 4b) 사후 참고: 고르지 않았지만 전체 세션에서 두 절반 모두 |t|>=2인 dist52 M에 같은 통제를 붙인다(선택 편향 있음, 후보 아님)
    say("## 4b. 사후 참고(후보 아님): dist52 M — 전체 세션 두 절반 모두 |t|>=2였던 구분에 같은 통제")
    sp = next(s for s in SP if s[0] == "dist52" and s[1] == "M")
    examine(t, sp, 1.0, "[사후] ")

    # 4c) 고르지 않은 조합 중 뒤 절반 |t|>=2 (사후 관찰, 선택 편향 있음 — 매매 근거 아님)
    say("## 4c. 고르지 않은 조합 중 뒤 절반 같은방향 |t|>=2 (사후 관찰, 62조합 중에서 본 것이라 선택 편향 큼)")
    picked = {(r.feature, r.cut) for r in pick}
    z = R[(R["뒤_같은방향t"].abs() >= 2) & ~R.apply(lambda x: (x.feature, x.cut) in picked, axis=1)]
    z = z[["feature", "cut", "scope", "앞_diff", "앞_t", "뒤_diff", "뒤_t", "뒤_같은방향t"]].copy()
    z["앞_diff"] = (z["앞_diff"] * 100).round(2)
    z["뒤_diff"] = (z["뒤_diff"] * 100).round(2)
    say(z.round(2).to_string(index=False))
    say("")

    # 5) 설명용: 전 기간 종목 성적 상·하위 1/3의 특성 중앙값(사후 구분, 매매 근거 아님)
    say("## 5. 설명용(사후): 21일 전체 종목 평균 수익 상위/하위 1/3 종목의 특성 중앙값 (5건 이상 종목)")
    for sc in ["전체", "정규"]:
        g = scope(t, sc).groupby("sym").agg(n=("ret", "size"), m=("ret", "mean"))
        g = g[g.n >= 5].join(f.set_index("sym")[FEATS + ["vol20", "atr14"]])
        q = g.m.quantile([1 / 3, 2 / 3]).values
        top, bot = g[g.m >= q[1]], g[g.m <= q[0]]
        say(f"  {sc}: 상위 {len(top)}종목(평균 {top.m.mean()*100:+.2f}%) vs 하위 {len(bot)}종목(평균 {bot.m.mean()*100:+.2f}%)")
        say("    " + " | ".join(f"{c} {top[c].median():+.3f}/{bot[c].median():+.3f}" for c in FEATS + ["vol20", "atr14"]))
    say("")

    with open(os.path.join(RES, "fam_trend_link.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    return R, out


if __name__ == "__main__":
    main()
