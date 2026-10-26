"""반박 검증: fam_vol_size.py 후보 3개(vol20 하위 1/3, gap60 하위 1/3, group=소프트웨어·보안)를 다시 계산해 무너뜨려 본다.

입력: ../results/trades.csv, ../results/features.csv (build.py 산출, 다시 받지 않음)
      체결 현실성 점검에만 build.py가 이미 받아 둔 1분봉(스크래치패드 raw_1009)을 읽는다(새로 받지 않음).
점검:
  1) 재현: 앞(9/10~9/23)·뒤(9/24~10/8) x 전체·정규·프리·애프터, 날짜 군집 t(날짜별 A-B 거래 평균 차).
     문턱은 136종목 9/9 특성 분포 삼분위 그대로(앞에서 정하고 뒤에 고정).
  2) 세션 구성: 날짜x세션 고정효과 회귀(날짜 군집 표준오차) -> 같은 날 같은 세션 안의 A-B 차.
     변동성 통제: 날짜x세션xvol20 삼분위 고정효과, 수익/atr14.
  3) 쏠림: 차이를 가장 줄이는 1종목·3종목(탐욕적)·1일 제외.
  4) 꼬리 체결: 매수 3분봉 안 1분봉에서 '종가가 매수가 이하인 분이 있었는가'(머문 체결) 또는
     '다음 3분봉도 매수가 이하로 내려왔는가'(2봉 연속)로 확인된 거래만 남겨 다시 잼.
     매도가는 매도 봉 시가 대신 매도 봉부터 6분간 1분 종가 중앙값으로 바꿔 다시 잼.
  5) 우연: 무작위 특성 200개 x 문턱 3 = 600 구분, 무작위 19종목 묶음 200개에 같은 고르기(네 조합 부호 일치 시 최소 |t|)와
     뒤 절반 확인을 돌려 후보 점수·뒤 t가 몇 번 나오는지 센다. 30구분 중 최댓값 분포는 무작위 구분 30개(연속 18+묶음 12) 2000회 재추출.
  6) 메커니즘: 앞 절반 종목별 애프터 거래 비중(9/23까지 앎) 상위 1/3을 뒤 절반에 적용해 같은 크기 차가 나오는지.
저장: ../results/verify_vol_size.txt
사용: py verify_vol_size.py"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw_1009"
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
SES = ["프리", "정규", "애프터"]
rng = np.random.default_rng(20261009)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)


def pct(v):
    return f"{v * 100:+.2f}%"


def tstat(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return (d.mean() if len(d) else np.nan), np.nan, len(d)
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), len(d)


def ddiff(di, y, a, nd):
    """날짜별 A 평균 - B 평균(양쪽 다 있는 날만)."""
    a = a.astype(float)
    sA = np.bincount(di, y * a, nd); nA = np.bincount(di, a, nd)
    sB = np.bincount(di, y * (1 - a), nd); nB = np.bincount(di, 1 - a, nd)
    ok = (nA > 0) & (nB > 0)
    return sA[ok] / nA[ok] - sB[ok] / nB[ok]


def load():
    t = pd.read_csv(os.path.join(RES, "trades.csv"), parse_dates=["t"])
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    t = t.merge(f, on="sym", how="inner")
    t["nret"] = t.ret / t.atr14
    q = f.vol20.quantile([1 / 3, 2 / 3]).values
    t["vt"] = np.where(t.vol20 <= q[0], 0, np.where(t.vol20 <= q[1], 1, 2))
    syms = f.sym.tolist()
    t["si"] = t.sym.map({s: i for i, s in enumerate(syms)})
    days = sorted(t.day.unique())
    t["di"] = t.day.map({d: i for i, d in enumerate(days)})
    return t, f, syms, len(days)


def fe_reg(x, a, fe, col="ret"):
    """고정효과(fe 열 묶음) 안에서 A 계수, 날짜 군집 t."""
    x = x.assign(A=a.astype(float))
    g = x.groupby(fe).ngroup()
    y = x[col] - x.groupby(g)[col].transform("mean")
    aa = x.A - x.groupby(g).A.transform("mean")
    den = (aa * aa).sum()
    if den == 0:
        return np.nan, np.nan, 0
    b = (aa * y).sum() / den
    s = (aa * (y - b * aa)).groupby(x.day).sum()
    G = len(s)
    v = G / (G - 1) * (s ** 2).sum() / den ** 2
    return b, b / np.sqrt(v), G


def drop_k(x, a, sign, nd, key, k):
    """차이를 sign 방향으로 가장 줄이는 key 값을 하나씩 k개 뺀다(탐욕적)."""
    keep = np.ones(len(x), bool)
    gone = []
    for _ in range(k):
        best = None
        for v in x.loc[keep, key].unique():
            kk = keep & (x[key] != v).values
            m = tstat(ddiff(x.di.values[kk], x.ret.values[kk], a[kk], nd))
            if best is None or sign * m[0] < sign * best[1][0]:
                best = (v, m, kk)
        gone.append(str(best[0])); keep = best[2]
    m = best[1]
    return gone, m


# ---------- 꼬리 체결 점검용: 1분봉에서 매수 봉·매도 봉 다시 보기 ----------
def fill_check(t):
    from events import bars
    out = []
    for s, g in t.groupby("sym"):
        m = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
        b3 = bars(RAW, s, 3)
        idx3 = b3.index
        for r in g.itertuples():
            t0 = r.t.tz_convert(m.index.tz)
            w = m[(m.index >= t0) & (m.index < t0 + pd.Timedelta(minutes=3))]
            eps = r.ent * 1e-6
            settle = bool((w.Close <= r.ent + eps).any())
            touch_n = int((w.Low <= r.ent + eps).sum())
            pos = idx3.get_indexer([t0])[0]
            nxt = b3.iloc[pos + 1] if 0 <= pos < len(b3) - 1 else None
            conf2 = bool(nxt is not None and nxt.Low <= r.ent + eps and (idx3[pos + 1] - t0) <= pd.Timedelta(minutes=6))
            xp = pos + r.bars
            px_ok = px_med = np.nan
            if 0 <= pos and xp < len(b3):
                tx = idx3[xp]
                px_ok = abs(b3.Open.iloc[xp] / r.px - 1) < 1e-6
                wx = m[(m.index >= tx) & (m.index < tx + pd.Timedelta(minutes=6))]
                px_med = wx.Close.median() if len(wx) else np.nan
            hit = w.index[w.Low <= r.ent + eps]
            prv = nx1 = np.nan
            if len(hit):
                p1 = m.index.get_loc(hit[0])
                prv = m.Close.iloc[p1 - 1] / r.ent - 1 if p1 > 0 else np.nan
                nx1 = m.Close.iloc[p1 + 1] / r.ent - 1 if p1 + 1 < len(m) else np.nan
            out.append(dict(i=r.Index, settle=settle, touch_n=touch_n, conf2=conf2, px_ok=px_ok, px_med=px_med, prv=prv, nx1=nx1))
    o = pd.DataFrame(out).set_index("i")
    return t.join(o)


def main():
    t, f, syms, nd = load()
    say("# 반박 검증 — fam_vol_size 후보 3개 (verify_vol_size, 2026-10-09)")
    say(f"거래 {len(t)}건·{t.sym.nunique()}종목, {nd}거래일. 날짜 군집 t = 날짜별 (A 거래 평균 - B 거래 평균)의 t.")
    q = {c: f[c].quantile([1 / 3, 0.5, 2 / 3]).values for c in ["vol20", "gap60"]}
    C = {"vol20 하위 1/3": (f.vol20 <= q["vol20"][0]).values,
         "gap60 하위 1/3": (f.gap60 <= q["gap60"][0]).values,
         "소프트웨어·보안": (f.group == "소프트웨어·보안").values}
    say(f"문턱(9/9 분포): vol20<={q['vol20'][0]:.5f}, gap60<={q['gap60'][0]:.4f}, 소프트웨어·보안 {C['소프트웨어·보안'].sum()}종목")
    say(f"A 종목 수: " + ", ".join(f"{k} {v.sum()}" for k, v in C.items()))
    H = {"앞": t[t.half == "앞"], "뒤": t[t.half == "뒤"]}

    # ---------- 1) 재현 ----------
    say("\n## 1) 재현 (거래 단위, 날짜 군집 t)")
    say("후보 | 절반 | 범위 | A 평균(건) | B 평균(건) | 차 | 날짜평균차 | t(k)")
    for name, mask in C.items():
        for h, x in H.items():
            for sc in ["전체", "정규", "프리", "애프터"]:
                xs = x if sc == "전체" else x[x.ses == sc]
                a = mask[xs.si.values]
                d = tstat(ddiff(xs.di.values, xs.ret.values, a, nd))
                say(f"{name} | {h} | {sc} | {pct(xs.ret[a].mean())}({a.sum()}) | {pct(xs.ret[~a].mean())}({(~a).sum()}) | "
                    f"{pct(xs.ret[a].mean() - xs.ret[~a].mean())} | {pct(d[0])} | {d[1]:+.2f}({d[2]})")

    # ---------- 2) 세션 구성·변동성 통제 ----------
    say("\n## 2) 세션 구성·변동성 통제 (고정효과 회귀, 날짜 군집 t)")
    say("후보 | 절반 | 날짜FE | 날짜x세션FE | 날짜x세션xvol삼분위FE | 수익/atr14 날짜x세션FE | 정규만 수익/atr14")
    for name, mask in C.items():
        for h, x in H.items():
            a = mask[x.si.values]
            r1 = fe_reg(x, a, ["day"])
            r2 = fe_reg(x, a, ["day", "ses"])
            r3 = fe_reg(x, a, ["day", "ses", "vt"]) if not name.startswith("vol20") else (np.nan, np.nan, 0)
            r4 = fe_reg(x, a, ["day", "ses"], "nret")
            xr = x[x.ses == "정규"]
            r5 = fe_reg(xr, mask[xr.si.values], ["day"], "nret")
            r3t = "해당 없음" if np.isnan(r3[0]) else f"{pct(r3[0])} t={r3[1]:+.2f}"
            say(f"{name} | {h} | {pct(r1[0])} t={r1[1]:+.2f} | {pct(r2[0])} t={r2[1]:+.2f} | {r3t} | "
                f"{r4[0]:+.3f}ATR t={r4[1]:+.2f} | {r5[0]:+.3f}ATR t={r5[1]:+.2f}")
        for h, x in H.items():
            a = mask[x.si.values]
            say(f"   {name} {h} 애프터 비중 A {(x.ses[a] == '애프터').mean() * 100:.0f}% B {(x.ses[~a] == '애프터').mean() * 100:.0f}% | "
                f"프리 비중 A {(x.ses[a] == '프리').mean() * 100:.0f}% B {(x.ses[~a] == '프리').mean() * 100:.0f}%")

    # ---------- 3) 쏠림 ----------
    say("\n## 3) 쏠림 (뒤 절반 전체 세션, 차이를 가장 줄이는 순서로 제외)")
    for name, mask in C.items():
        for sc in ["전체", "프리"]:
            x = H["뒤"] if sc == "전체" else H["뒤"][H["뒤"].ses == sc]
            a = mask[x.si.values]
            base = tstat(ddiff(x.di.values, x.ret.values, a, nd))
            sg = np.sign(base[0])
            g1, m1 = drop_k(x, a, sg, nd, "sym", 1)
            g3, m3 = drop_k(x, a, sg, nd, "sym", 3)
            gd, md = drop_k(x, a, sg, nd, "day", 1)
            say(f"{name} 뒤·{sc}: 기본 {pct(base[0])} t={base[1]:+.2f} | -{g1} {pct(m1[0])} t={m1[1]:+.2f} | "
                f"-{'/'.join(g3)} {pct(m3[0])} t={m3[1]:+.2f} | -{gd} {pct(md[0])} t={md[1]:+.2f}")

    # ---------- 4) 꼬리 체결 ----------
    say("\n## 4) 꼬리 체결 점검 (1분봉)")
    t2 = fill_check(t)
    t2["conf"] = t2.settle | t2.conf2
    t2["ret_med"] = t2.px_med / t2.ent - 1
    t2["spike"] = np.minimum(t2.prv, t2.nx1) > 0.01   # 닿은 1분봉 직전·직후 1분 종가가 모두 매수가보다 1% 넘게 위 = 한 분짜리 튐 체결(사후 정보 포함, 진단용)
    t2["jump"] = t2.prv > 0.01                        # 직전 1분 종가보다 1% 넘게 아래에서 닿음(체결 순간에 앎)
    say(f"매도가 재현(매도 봉 시가 = px): {t2.px_ok.mean() * 100:.1f}% ({t2.px_ok.notna().sum()}건 중)")
    say("세션 | 건수 | 원래 평균 | 머문 체결(1분 종가<=매수가) 비율 | 2봉 연속 비율 | 확인된 체결만 평균(건) | 확인 안 된 체결 평균(건) | 매도가 6분 중앙값 평균")
    for s in SES:
        g = t2[t2.ses == s]
        c, u = g[g.conf], g[~g.conf]
        say(f"{s} | {len(g)} | {pct(g.ret.mean())} | {g.settle.mean() * 100:.0f}% | {g.conf2.mean() * 100:.0f}% | "
            f"{pct(c.ret.mean())}({len(c)}) | {pct(u.ret.mean())}({len(u)}) | {pct(g.ret_med.mean())}")
    say("튐 체결(닿은 1분봉 직전·직후 1분 종가가 모두 매수가 +1% 초과): 세션 | 비율 | 직후 1분 종가 중앙값(매수가 대비) | 튐 평균(건) | 튐 아님 평균(건)")
    for s in SES:
        g = t2[t2.ses == s]
        say(f"{s} | {g.spike.mean() * 100:.0f}% | {pct(g.nx1.median())} | {pct(g.ret[g.spike].mean())}({g.spike.sum()}) | "
            f"{pct(g.ret[~g.spike].mean())}({(~g.spike).sum()})")
    say("후보별 튐 체결 비율(A/B)과 튐 아닌 거래만 A-B (날짜 군집 t, 날짜x세션FE)")
    for name, mask in C.items():
        for h in ["앞", "뒤"]:
            x = t2[t2.half == h]
            a = mask[x.si.values]
            sp = " ".join(f"{s} {x.spike[a & (x.ses == s).values].mean() * 100:.0f}/{x.spike[~a & (x.ses == s).values].mean() * 100:.0f}%" for s in SES)
            xn = x[~x.spike]
            an = mask[xn.si.values]
            d = tstat(ddiff(xn.di.values, xn.ret.values, an, nd))
            xr = xn[xn.ses != "정규"]
            ar = mask[xr.si.values]
            dr = tstat(ddiff(xr.di.values, xr.ret.values, ar, nd))
            fe = fe_reg(xn, an, ["day", "ses"])
            say(f"{name} {h}: 튐 비율 A/B {sp} | 튐 아님 전체 A {pct(xn.ret[an].mean())}({an.sum()}) B {pct(xn.ret[~an].mean())}({(~an).sum()}) "
                f"날짜차 {pct(d[0])} t={d[1]:+.2f} | 튐 아님 프리+애프터 날짜차 {pct(dr[0])} t={dr[1]:+.2f}({ar.sum()}/{(~ar).sum()}) | "
                f"튐 아님 날짜x세션FE {pct(fe[0])} t={fe[1]:+.2f}")
    say("체결 순간에 아는 기준만 쓴 변형(사후 정보 없음): 급락 닿음(직전 1분 종가 대비 -1% 넘게) 제외 / 2봉 연속(사용자 기존 거름 규칙)만")
    for s in SES:
        g = t2[t2.ses == s]
        say(f"  {s}: 급락 닿음 {g.jump.mean() * 100:.0f}% (그 평균 {pct(g.ret[g.jump].mean())}, 나머지 {pct(g.ret[~g.jump].mean())}) | "
            f"2봉 연속 {g.conf2.mean() * 100:.0f}% (그 평균 {pct(g.ret[g.conf2].mean())})")
    for name, mask in C.items():
        parts = []
        for h in ["앞", "뒤"]:
            x = t2[t2.half == h]
            for lab, sub in [("급락 닿음 제외", x[~x.jump]), ("2봉 연속만", x[x.conf2])]:
                a = mask[sub.si.values]
                d = tstat(ddiff(sub.di.values, sub.ret.values, a, nd))
                fe = fe_reg(sub, a, ["day", "ses"])
                parts.append(f"{h}·{lab} A {pct(sub.ret[a].mean())}({a.sum()}) B {pct(sub.ret[~a].mean())}({(~a).sum()}) 날짜차 {pct(d[0])} "
                             f"t={d[1]:+.2f}, 세션FE {pct(fe[0])} t={fe[1]:+.2f}")
        say(f"{name}: " + " | ".join(parts))
    say("후보별 A-B (날짜 군집 t): 원래 / 확인된 체결만 / 매도가 6분 중앙값 / 확인 체결+중앙값 매도"
        " (주의: '머문 체결'은 1분봉에 체결이 하나뿐이면 종가=저가라 튐 체결을 거르지 못함)")
    for name, mask in C.items():
        for h in ["앞", "뒤"]:
            x = t2[t2.half == h]
            parts = []
            for lab, sub, col in [("원래", x, "ret"), ("확인 체결", x[x.conf], "ret"), ("중앙값 매도", x[x.ret_med.notna()], "ret_med"),
                                  ("둘 다", x[x.conf & x.ret_med.notna()], "ret_med")]:
                a = mask[sub.si.values]
                d = tstat(ddiff(sub.di.values, sub[col].values, a, nd))
                parts.append(f"{lab} A {pct(sub[col][a].mean())}({a.sum()}) B {pct(sub[col][~a].mean())}({(~a).sum()}) "
                             f"날짜차 {pct(d[0])} t={d[1]:+.2f}")
            say(f"{name} {h}: " + " | ".join(parts))
        x = t2[t2.half == "뒤"]
        a = mask[x.si.values]
        say(f"   {name} 뒤 애프터: 머문 체결 비율 A {x.settle[a & (x.ses == '애프터').values].mean() * 100:.0f}% "
            f"B {x.settle[~a & (x.ses == '애프터').values].mean() * 100:.0f}% | 프리: A {x.settle[a & (x.ses == '프리').values].mean() * 100:.0f}% "
            f"B {x.settle[~a & (x.ses == '프리').values].mean() * 100:.0f}%")
        xc = x[x.conf]
        r2 = fe_reg(xc, mask[xc.si.values], ["day", "ses"])
        say(f"   {name} 뒤 확인 체결만 날짜x세션FE {pct(r2[0])} t={r2[1]:+.2f}")

    # ---------- 5) 우연(무작위 특성) ----------
    say("\n## 5) 우연: 무작위 특성에 같은 고르기·확인을 돌림")
    pre = {}
    for h, x in H.items():
        for sc in ["전체", "정규"]:
            xs = x if sc == "전체" else x[x.ses == "정규"]
            pre[(h, sc, "거래")] = (xs.di.values, xs.ret.values, xs.si.values)
            ds = xs.groupby(["di", "si"]).ret.mean().reset_index()
            pre[(h, sc, "종목")] = (ds.di.values, ds.ret.values, ds.si.values)

    def evaluate(mask):
        ts = [tstat(ddiff(pre[k][0], pre[k][1], mask[pre[k][2]], nd))[1] for k in
              [("앞", "전체", "거래"), ("앞", "전체", "종목"), ("앞", "정규", "거래"), ("앞", "정규", "종목")]]
        ts = np.array(ts)
        sg = np.sign(ts)
        same = np.all(sg == sg[0]) and not np.isnan(ts).any()
        score = np.abs(ts).min() if same else 0.0
        s0 = sg[0] if same else np.sign(ts[0])
        t2a = tstat(ddiff(*[pre[("뒤", "전체", "거래")][i] for i in (0, 1)], mask[pre[("뒤", "전체", "거래")][2]], nd))[1]
        t2r = tstat(ddiff(*[pre[("뒤", "정규", "거래")][i] for i in (0, 1)], mask[pre[("뒤", "정규", "거래")][2]], nd))[1]
        return score, s0, s0 * t2a, s0 * t2r

    real = {k: evaluate(v) for k, v in C.items()}
    for k, v in real.items():
        say(f"실제 {k}: 앞 점수 {v[0]:.2f}, 뒤 전체·거래 t(앞 방향 기준) {v[2]:+.2f}, 뒤 정규·거래 t {v[3]:+.2f}")
    n = len(f)
    cont, grp = [], []
    for _ in range(200):
        z = rng.standard_normal(n)
        q33, q50, q67 = np.quantile(z, [1 / 3, 0.5, 2 / 3])
        for m in (z >= q67, z <= q33, z > q50):
            cont.append(evaluate(m))
        g = np.zeros(n, bool); g[rng.choice(n, 19, replace=False)] = True
        grp.append(evaluate(g))
    cont, grp = np.array(cont), np.array(grp)
    allr = np.vstack([cont, grp])
    say(f"무작위 연속 구분 {len(cont)}개 + 무작위 19종목 묶음 {len(grp)}개 = {len(allr)}개")
    for k, v in real.items():
        say(f"  {k}: 앞 점수>={v[0]:.2f} {np.mean(allr[:, 0] >= v[0]) * 100:.1f}% | 뒤 전체 t>={v[2]:.2f}(앞 방향) "
            f"{np.mean(allr[:, 2] >= v[2]) * 100:.1f}% | 둘 다 {np.mean((allr[:, 0] >= v[0]) & (allr[:, 2] >= v[2])) * 100:.1f}% | "
            f"뒤 정규 t>={v[3]:.2f} {np.mean(allr[:, 3] >= v[3]) * 100:.1f}%")
    say(f"  무작위 구분 중 앞 점수>0이고 뒤 전체 t>=2: {np.mean((allr[:, 0] > 0) & (allr[:, 2] >= 2)) * 100:.1f}%, "
        f"뒤 정규 t>=2: {np.mean((allr[:, 0] > 0) & (allr[:, 3] >= 2)) * 100:.1f}%")
    mx = np.array([max(cont[rng.choice(len(cont), 18, replace=False), 0].max(), grp[rng.choice(len(grp), 12, replace=False), 0].max())
                   for _ in range(2000)])
    say(f"  30구분(연속 18+묶음 12) 최고 점수 분포: 중앙값 {np.median(mx):.2f}, 90% {np.quantile(mx, 0.9):.2f}; "
        f"실제 1위 0.88 이상 비율 {np.mean(mx >= 0.88) * 100:.0f}%")

    # ---------- 6) 메커니즘: 앞 절반 애프터 비중 ----------
    say("\n## 6) 메커니즘: 앞 절반 종목별 애프터 거래 비중(9/23까지 앎) 상위 1/3 -> 뒤 절반")
    sh = H["앞"].groupby("sym").ses.apply(lambda s: (s == "애프터").mean())
    thr = sh.quantile(2 / 3)
    hi = f.sym.map(sh).fillna(-1).values >= thr
    say(f"문턱 애프터 비중>={thr * 100:.0f}% ({hi.sum()}종목, 앞 거래 있는 {len(sh)}종목 기준)")
    x = H["뒤"]
    for sc in ["전체", "정규"]:
        xs = x if sc == "전체" else x[x.ses == "정규"]
        a = hi[xs.si.values]
        d = tstat(ddiff(xs.di.values, xs.ret.values, a, nd))
        say(f"뒤·{sc}: A {pct(xs.ret[a].mean())}({a.sum()}) B {pct(xs.ret[~a].mean())}({(~a).sum()}) 날짜차 {pct(d[0])} t={d[1]:+.2f}")
    a = hi[x.si.values]
    r2 = fe_reg(x, a, ["day", "ses"])
    say(f"뒤 날짜x세션FE: {pct(r2[0])} t={r2[1]:+.2f}")
    for name, mask in C.items():
        say(f"  겹침: {name} 종목 중 애프터 비중 상위 1/3 {np.mean(hi[mask]) * 100:.0f}% (나머지 {np.mean(hi[~mask]) * 100:.0f}%)")
    rk = f.set_index("sym")[["vol20", "gap60"]].join(sh.rename("aft")).dropna().rank()
    say(f"  스피어만(종목, 설명용): 앞 애프터 비중 vs vol20 {rk.aft.corr(rk.vol20):+.2f}, vs gap60 {rk.aft.corr(rk.gap60):+.2f}")

    with open(os.path.join(RES, "verify_vol_size.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
