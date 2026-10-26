"""반박 검증: fam_context.py 후보 3개(sdev_q 하위 1/3 나쁨, qdev<0 좋음, prev_ret 하위 1/3 좋음)를 다시 계산해 무너뜨려 본다. 2026-10-09.

입력: ../results/trades.csv, ../results/features.csv (build.py 산출, 다시 받지 않음)
      정의 재확인·체결 현실성 점검에만 build.py가 이미 받아 둔 1분봉(스크래치패드 raw_1009)을 읽는다(새로 받지 않음).
점검:
  0) 정의: 3분봉을 다시 만들어 매수가(min(시가, EMA60[1]x(1-갭)))·sdev_q·qdev가 trades.csv와 같은지, 값이 매수 봉 직전 봉(i-1)까지인지.
     날짜별 세션 거래 수(데이터 빈 곳) 확인.
  1) 재현: 문턱(앞 절반 1/3 분위)·방향(앞 절반)을 앞에서 정해 뒤에 고정. 전체·정규·프리·애프터. t = 날짜 군집(날짜별 A평균-B평균).
  2) 통제: 날짜x세션 고정효과(같은 날 같은 세션 안 A-B), 날짜x세션xvol20삼분위 고정효과, 수익/atr14, 애프터 비중.
  3) 쏠림: 규칙 방향 차이를 가장 줄이는 1종목·3종목(탐욕적)·1일 제외.
  4) 꼬리 체결: 매수 3분봉 안 1분 종가가 매수가 이하였던 적이 있거나(머문 체결) 다음 3분봉 저가도 매수가 이하(2봉 연속)인 거래만 '확인된 체결'.
     매도가는 매도 봉 시가 대신 매도 봉부터 6분간 1분 종가 중앙값(대체 매도가)으로도 잰다.
  5) 우연: (a) 거래마다 무작위 정규값 특성 200개 x 분할 3(하위1/3·하위1/2·상위1/3) x 범위 2(전체·정규)에 같은 앞 점수(두 범위 부호 같을 때 min|t|)·뒤 확인.
            앞 점수 최댓값 분포는 무작위 특성 6개(=18분할, 후보 17분할과 비슷) 묶음 2000회 재추출.
         (b) sdev_q를 같은 날·같은 세션 안에서만 섞기 200회(세션 구성은 그대로, 세션 안 연결만 끊음) -> 전체 범위 t가 몇 번 나오는가.
         (c) qdev를 다른 날의 QQQ 경로(같은 시각)로 바꾸기 200회(날짜 순열, 자기 날 제외) -> 같은 점수·뒤 t가 몇 번 나오는가.
  6) 종목 지속성(앞->뒤 종목 평균 스피어만)을 확인된 체결·대체 매도가로 다시 잼(분석가 ② 점검).
저장: ../results/verify_context.txt
사용: py verify_context.py"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw_1009"
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
LEV = set("SOXL,SOXS,TQQQ,SQQQ,TECL,TECS,LABU,LABD,NAIL,DPST,KORU,SPXL,SPXS,UPRO,SPXU,TNA,TZA,FNGU,FNGD,YINN,YANG".split(","))
SES = ["프리", "정규", "애프터"]
NS3, NS6 = 3 * 60 * 10 ** 9, 6 * 60 * 10 ** 9
rng = np.random.default_rng(20261009)
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def pct(v):
    return "nan" if pd.isna(v) else f"{100 * v:+.2f}%"


def tst(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return (v.mean() if len(v) else np.nan), np.nan, len(v)
    return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), len(v)


ND = 21


def dpair(di, y, a):
    a = a.astype(float)
    sA, nA = np.bincount(di, y * a, ND), np.bincount(di, a, ND)
    sB, nB = np.bincount(di, y * (1 - a), ND), np.bincount(di, 1 - a, ND)
    ok = (nA > 0) & (nB > 0)
    return sA[ok] / nA[ok] - sB[ok] / nB[ok]


def dt(x, a, col="ret"):
    return tst(dpair(x.di.values, x[col].values, np.asarray(a, bool)))


def fe(x, a, keys, col="ret"):
    """keys 고정효과 안 A 계수, 날짜 군집 표준오차."""
    g = x.groupby(keys).ngroup().values
    y = x[col].values.astype(float)
    A = np.asarray(a, float)
    cnt = np.bincount(g)
    ym = y - (np.bincount(g, y) / cnt)[g]
    am = A - (np.bincount(g, A) / cnt)[g]
    den = (am * am).sum()
    if den == 0:
        return np.nan, np.nan, 0
    b = (am * ym).sum() / den
    s = np.bincount(x.di.values, am * (ym - b * am), ND)
    used = np.bincount(x.di.values, np.abs(am), ND) > 0
    G = int(used.sum())
    v = G / (G - 1) * (s[used] ** 2).sum() / den ** 2
    return b, b / np.sqrt(v), G


def drop_greedy(x, a, sgn, key, k):
    keep = np.ones(len(x), bool)
    gone, best = [], None
    for _ in range(k):
        best = None
        for v in x.loc[keep, key].unique():
            kk = keep & (x[key] != v).values
            m = tst(dpair(x.di.values[kk], x.ret.values[kk], a[kk]))
            if np.isnan(m[0]):
                continue
            if best is None or sgn * m[0] < sgn * best[1][0]:
                best = (v, m, kk)
        gone.append(str(best[0]))
        keep = best[2]
    return gone, best[1]


# ---------------- 데이터 ----------------
def load():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    t["ts"] = pd.to_datetime(t.t, utc=True).dt.tz_convert("America/New_York")
    t["mod"] = t.ts.dt.hour * 60 + t.ts.dt.minute
    t = t.merge(f[["sym", "vol20", "atr14"]], on="sym", how="left")
    q1, q2 = f.vol20.quantile([1 / 3, 2 / 3])
    t["vt"] = np.select([t.vol20 <= q1, t.vol20 <= q2, t.vol20 > q2], [0, 1, 2], -1)
    t["ra"] = t.ret / t.atr14
    days = sorted(t.day.unique())
    t["di"] = t.day.map({d: i for i, d in enumerate(days)})
    t["aft"] = t.ses == "애프터"
    return t, f, days


def bar_check(t):
    """3분봉 재구성으로 정의 재확인 + 1분봉으로 체결 현실성·대체 매도가."""
    from events import bars, prep
    q = prep(bars(RAW, "QQQ", 3), 60, 14)
    qs = pd.Series((q.c / q.ema - 1).values, index=q.index.as_unit("ns"))
    out = []
    for s, g in t.groupby("sym"):
        d = prep(bars(RAW, s, 3), 60, 14)
        idx = d.index.as_unit("ns")
        ii = idx.asi8
        qq = qs.reindex(idx).ffill().values
        dev = (d.c / d.ema - 1).values
        o, c, l, ema = d.o.values, d.c.values, d.l.values, d.ema.values
        m = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
        mi = m.index.as_unit("ns").asi8
        mc = m.Close.values
        gap = 0.03 if s in LEV else 0.02
        for r in g.itertuples():
            t0 = r.ts.value
            pos = int(np.searchsorted(ii, t0))
            row = dict(i=r.Index)
            if pos >= len(ii) or ii[pos] != t0 or pos < 1:
                out.append(row)
                continue
            lvl = ema[pos - 1] * (1 - gap)
            eps = r.ent * 1e-6
            lo, hi = np.searchsorted(mi, t0), np.searchsorted(mi, t0 + NS3)
            settle = bool((mc[lo:hi] <= r.ent + eps).any())
            conf2 = bool(pos + 1 < len(ii) and ii[pos + 1] - t0 <= NS6 and l[pos + 1] <= r.ent + eps)
            xp = pos + r.bars
            px_chk = px_alt = np.nan
            if xp < len(ii):
                px_chk = o[xp]
                a1, a2 = np.searchsorted(mi, ii[xp]), np.searchsorted(mi, ii[xp] + NS6)
                px_alt = np.median(mc[a1:a2]) if a2 > a1 else np.nan
            row.update(ent_chk=min(o[pos], lvl), sdev_chk=dev[pos - 1] - qq[pos - 1], qdev_chk=qq[pos - 1],
                       dev1=dev[pos - 1], gapfill=o[pos] <= lvl, drop1=r.ent / c[pos - 1] - 1, cbar=c[pos] / r.ent - 1,
                       settle=settle, conf2=conf2, px_chk=px_chk, px_alt=px_alt)
            out.append(row)
    o = pd.DataFrame(out).set_index("i")
    t = t.join(o)
    t["ok"] = t.settle.fillna(False).astype(bool) | t.conf2.fillna(False).astype(bool)
    t["ret_alt"] = t.px_alt / t.ent - 1
    return t, q


# ---------------- 후보 규칙 ----------------
def cand_defs(t):
    """(이름, 방향 sgn, 범위별 마스크 함수). 문턱은 범위의 앞 절반 분위(프리·애프터는 전체 문턱, 분석가와 같음)."""
    a1 = t[t.half == "앞"]
    r1 = a1[a1.ses == "정규"]
    th = dict(sd_all=a1.sdev_q.quantile(1 / 3), sd_reg=r1.sdev_q.quantile(1 / 3),
              pr_all=a1.prev_ret.quantile(1 / 3), pr_reg=r1.prev_ret.quantile(1 / 3))
    C = [("C1 sdev_q 하위1/3(나쁨)", -1, lambda x, sc: x.sdev_q <= (th["sd_reg"] if sc == "정규" else th["sd_all"])),
         ("C2 qdev<0(좋음)", +1, lambda x, sc: x.qdev < 0),
         ("C3 prev_ret 하위1/3(좋음)", +1, lambda x, sc: x.prev_ret <= (th["pr_reg"] if sc == "정규" else th["pr_all"]))]
    return C, th


def scope(t, sc):
    return t if sc == "전체" else t[t.ses == sc]


def row_line(x, a, sgn, col="ret", full=False):
    a = np.asarray(a, bool)
    A, B = x[a], x[~a]
    m, tt, n = dt(x, a, col)
    s = f"A {len(A)}건 {pct(A[col].mean())} vs B {len(B)}건 {pct(B[col].mean())} | 날짜차 {pct(m)} t={tt:+.2f}({n}일)"
    if full:
        b1, t1, _ = fe(x, a, ["day", "ses"], col)
        xv = x[x.vt >= 0]
        av = a[(x.vt >= 0).values]
        b2, t2, _ = fe(xv, av, ["day", "ses", "vt"], col)
        b3, t3, _ = fe(xv, av, ["day", "vt"], col)
        m4, t4, _ = dt(xv, av, "ra")
        s += (f" | 날짜x세션 FE {pct(b1)} t={t1:+.2f} | 날짜xvol삼분위 FE {pct(b3)} t={t3:+.2f} | 날짜x세션xvol FE {pct(b2)} t={t2:+.2f}"
              f" | 수익/atr t={t4:+.2f} | 애프터비중 A {100 * A.aft.mean():.0f}% B {100 * B.aft.mean():.0f}%")
        if col == "ret":
            g1, m1 = drop_greedy(x, a, sgn, "sym", 1)
            g3, m3 = drop_greedy(x, a, sgn, "sym", 3)
            gd, md = drop_greedy(x, a, sgn, "day", 1)
            s += (f" | -{g1[0]} {pct(m1[0])} t={m1[1]:+.2f} | -{','.join(g3)} {pct(m3[0])} t={m3[1]:+.2f}"
                  f" | -{gd[0]} {pct(md[0])} t={md[1]:+.2f}")
    return s


# ---------------- 우연 점검 ----------------
def score_split(t, a, a_reg=None):
    """앞 절반 점수(전체·정규 t 부호 같을 때 min|t|), 방향, 뒤 전체·정규 부호 맞춘 t. a = 전체 범위 마스크, a_reg = 정규 범위 마스크(정규 문턱)."""
    a_reg = a if a_reg is None else a_reg
    h1 = (t.half == "앞").values
    reg = (t.ses == "정규").values
    r = {}
    for nm, k, m in (("all1", h1, a), ("reg1", h1 & reg, a_reg), ("all2", ~h1, a), ("reg2", ~h1 & reg, a_reg)):
        x = t[k]
        r[nm] = tst(dpair(x.di.values, x.ret.values, m[k]))[1]
    sgn = np.sign(r["all1"]) if not np.isnan(r["all1"]) else 0
    sc = min(abs(r["all1"]), abs(r["reg1"])) if (sgn != 0 and np.sign(r["reg1"]) == sgn) else 0.0
    return sc, sgn, sgn * r["all2"], sgn * r["reg2"]


def null_random(t, n=200):
    h1 = (t.half == "앞").values
    reg = (t.ses == "정규").values
    qs = {"하위1/3": 1 / 3, "하위1/2": 0.5, "상위1/3": 2 / 3}
    res = []
    for j in range(n):
        z = rng.standard_normal(len(t))
        for nm, qq in qs.items():
            qa, qr = np.quantile(z[h1], qq), np.quantile(z[h1 & reg], qq)
            a, ar = ((z > qa), (z > qr)) if nm == "상위1/3" else ((z <= qa), (z <= qr))
            sc, sgn, a2, r2 = score_split(t, a, ar)
            res.append(dict(feat=j, split=nm, score=sc, all2=a2, reg2=r2))
    return pd.DataFrame(res)


def qdev_table(q):
    """날짜 -> (분, QQQ 이격) 배열."""
    tab = {}
    for dte, g in q.groupby(q.index.date):
        tab[str(dte)] = (np.asarray(g.index.hour * 60 + g.index.minute), (g.c / g.ema - 1).values)
    return tab


def lookup_q(tab, day, mod):
    mm, vv = tab[day]
    p = np.searchsorted(mm, mod - 3, side="right") - 1
    return vv[max(p, 0)]


def main():
    t, f, days = load()
    t, q = bar_check(t)
    say("# 반박 검증 — fam_context 후보 3개 (verify_context.py, 2026-10-09)")
    say(f"거래 {len(t)}건, {t.sym.nunique()}종목, {t.day.nunique()}일(앞 {t[t.half == '앞'].day.nunique()}·뒤 {t[t.half == '뒤'].day.nunique()})."
        " 단위: 거래당 수익. t = 날짜 군집. A = 규칙 조건 무리.")

    # ---- 0) 정의 ----
    say("\n## 0) 정의·미래 참조 점검")
    k = t.ent_chk.notna()
    say(f"3분봉 재구성 매칭 {int(k.sum())}/{len(t)}건 | 매수가 일치(상대오차<1e-6) {np.mean(np.abs(t.ent_chk[k] / t.ent[k] - 1) < 1e-6) * 100:.1f}%"
        f" | 매도가(매도 봉 시가) 일치 {np.mean(np.abs(t.px_chk[k] / t.px[k] - 1) < 1e-6) * 100:.1f}%"
        f" | sdev_q 일치(|차|<1e-9) {np.mean(np.abs(t.sdev_chk[k] - t.sdev_q[k]) < 1e-9) * 100:.1f}% | qdev 일치 {np.mean(np.abs(t.qdev_chk[k] - t.qdev[k]) < 1e-9) * 100:.1f}%")
    say("→ sdev_q·qdev 모두 매수 봉 직전 봉(i-1) 종가·EMA로 계산(매수 전 앎). prev_ret = 전날 정규장 수익(앎). 미래 참조 없음.")
    say("구조: sdev_q + qdev = 종목 직전 봉 이격(dev[i-1]). 매수는 저가가 EMA[i-1]x0.98에 닿을 때이므로 dev[i-1]이 높을수록 '한 봉 안 급락'으로 체결된다.")
    rs = t[["sdev_q", "dev1", "drop1"]].corr(method="spearman")
    say(f"스피어만: sdev_q~dev[i-1] {rs.loc['sdev_q', 'dev1']:+.2f}, sdev_q~한봉낙폭(매수가/직전 종가-1) {rs.loc['sdev_q', 'drop1']:+.2f}")
    cnt = t.groupby(["day", "ses"]).size().unstack(fill_value=0)[SES]
    say("날짜별 세션 거래 수(데이터 빈 곳): " + " ".join(f"{d[5:]}:{r['프리']}/{r['정규']}/{r['애프터']}" for d, r in cnt.iterrows()))
    say("→ 10/07·10/08 애프터 거래 0건(10/07 애프터 1분봉 없음, 10/06 애프터는 19:01에 끊김). 뒤 절반 애프터는 9일치.")

    # ---- 4) 체결 현실성 먼저 요약 ----
    say("\n## 4) 꼬리 체결 점검(1분봉)")
    say("세션 | 건수 | 원래 평균 | 확인된 체결 비율 | 확인된 체결만 평균(건) | 확인 안 된 체결 평균(건) | 대체 매도가 평균 | 확인+대체 평균")
    for sname in SES + ["전체"]:
        x = scope(t, sname)
        ok = x.ok
        say(f"{sname} | {len(x)} | {pct(x.ret.mean())} | {100 * ok.mean():.0f}% | {pct(x.ret[ok].mean())}({int(ok.sum())}) | "
            f"{pct(x.ret[~ok].mean())}({int((~ok).sum())}) | {pct(x.ret_alt.mean())} | {pct(x.ret_alt[ok].mean())}")
    a1 = t[t.half == "앞"]
    thr = a1.sdev_q.quantile(1 / 3)
    for sname in ("프리", "애프터"):
        x = t[t.ses == sname]
        lo = x.sdev_q <= thr
        say(f"{sname}: 확인된 체결 비율 sdev_q 하위1/3 {100 * x.ok[lo].mean():.0f}%({int(lo.sum())}건) vs 나머지 {100 * x.ok[~lo].mean():.0f}%({int((~lo).sum())}건)"
            f" | 매수 봉 종가가 매수가보다 1% 넘게 위(꼬리 뒤 즉시 복귀) {100 * (x.cbar[lo] > 0.01).mean():.0f}% vs {100 * (x.cbar[~lo] > 0.01).mean():.0f}%")
    say("세션·체결 | 건수 | 수익 중앙값 | 승률 | 매수 봉 종가가 매수가+1% 위 | 수익이 1.5~2.5%(=갭만큼 제자리 복귀) | 최대 역행 중앙값")
    for sname in SES:
        for okv in (True, False):
            y = t[(t.ses == sname) & (t.ok == okv)]
            say(f"{sname}·{'확인' if okv else '미확인'} | {len(y)} | {pct(y.ret.median())} | {100 * (y.ret > 0).mean():.0f}% | {100 * (y.cbar > 0.01).mean():.0f}% | "
                f"{100 * y.ret.between(0.015, 0.025).mean():.0f}% | {pct(y.mae.median())}")
    say("→ 애프터는 확인된 체결도 승률 96%·절반 이상이 갭만큼 제자리 복귀: 1분 종가 확인으로도 걸러지지 않는 가짜 저가가 대부분. 애프터 결과는 믿을 수 없음.")
    x = t[(t.sym == "NVDA") & (t.ses == "애프터")]
    say(f"예: NVDA 애프터 {len(x)}건 평균 {pct(x.ret.mean())}, 확인된 체결 {int(x.ok.sum())}건, 대체 매도가 평균 {pct(x.ret_alt.mean())}"
        " (10/06 16:57 1분봉 저가 224.41·종가 239.51 같은 꼬리에 체결)")

    # ---- 1~3) 후보 재현·통제 ----
    C, th = cand_defs(t)
    say(f"\n## 1~3) 후보 재현(문턱: sdev_q 전체 {th['sd_all']:+.4f}·정규 {th['sd_reg']:+.4f}, prev_ret 전체 {th['pr_all']:+.4f}·정규 {th['pr_reg']:+.4f}; 앞 절반 분위)")
    variants = [("원래", "ret", None), ("확인된 체결만", "ret", "ok"), ("대체 매도가", "ret_alt", None), ("확인된 체결+대체 매도가", "ret_alt", "ok")]
    summary = {}
    for name, sgn, fn in C:
        say(f"\n### {name}")
        for sc in ["전체", "정규", "프리", "애프터"]:
            for vn, col, flt in variants:
                x = scope(t, sc)
                if flt:
                    x = x[x[flt]]
                x = x[x[col].notna()]
                x1, x2 = x[x.half == "앞"], x[x.half == "뒤"]
                full = (vn in ("원래", "확인된 체결+대체 매도가")) and sc in ("전체", "정규")
                l1 = row_line(x1, fn(x1, sc).values, sgn, col)
                l2 = row_line(x2, fn(x2, sc).values, sgn, col, full=full)
                say(f"[{sc}·{vn}] 앞: {l1}")
                say(f"[{sc}·{vn}] 뒤: {l2}")
                summary[(name, sc, vn)] = (dt(x1, fn(x1, sc).values, col)[1], dt(x2, fn(x2, sc).values, col)[1])

    # ---- 5a) 무작위 특성 ----
    say("\n## 5a) 무작위 특성 200개 x 분할 3 = 600분할(전체·정규 각각 문턱), 같은 앞 점수·뒤 확인")
    nr = null_random(t)
    best = nr.sort_values("score", ascending=False).drop_duplicates("feat")
    for s0, a0, lab in ((2.67, 2.61, "C1 수준(앞 점수≥2.67, 뒤 전체 같은 방향 t≥2.61)"),
                        (1.85, 1.76, "C2 수준(앞 점수≥1.85, 뒤 전체 같은 방향 t≥1.76)"),
                        (1.44, 0.0, "C3 수준(앞 점수≥1.44)")):
        k1 = nr.score >= s0
        k2 = k1 & (nr.all2 >= a0)
        say(f"{lab}: 600분할 중 앞 통과 {int(k1.sum())}개({100 * k1.mean():.1f}%), 앞+뒤 통과 {int(k2.sum())}개({100 * k2.mean():.1f}%)"
            f" | 특성별 최고 분할 200개 중 앞 통과 {int((best.score >= s0).sum())}, 앞+뒤 {int(((best.score >= s0) & (best.all2 >= a0)).sum())}")
    say(f"무작위 분할 뒤 전체 t 분포(앞 점수≥1.44인 것): 중앙값 {nr.all2[nr.score >= 1.44].median():+.2f}, |t|≥1.96 비율 {100 * (nr.all2[nr.score >= 1.44].abs() >= 1.96).mean():.0f}%")
    mx = []
    sc_by = nr.groupby("feat").score.max()
    for _ in range(2000):
        pick = rng.choice(sc_by.index.values, 6, replace=False)
        mx.append(sc_by.loc[pick].max())
    mx = np.array(mx)
    say(f"무작위 특성 6개(18분할) 묶음의 앞 점수 최댓값: 중앙값 {np.median(mx):.2f}, 90% {np.quantile(mx, 0.9):.2f}, ≥2.67 비율 {100 * (mx >= 2.67).mean():.1f}%, ≥1.85 비율 {100 * (mx >= 1.85).mean():.1f}%")

    # ---- 5b) sdev_q 날짜x세션 안 섞기 ----
    say("\n## 5b) sdev_q를 같은 날·같은 세션 안에서만 섞기 200회(세션 구성 유지, 세션 안 연결 끊음), 전체 범위·문턱 고정")
    thr = th["sd_all"]
    grp = t.groupby(["day", "ses"]).indices
    real1 = dt(t[t.half == "앞"], (t[t.half == "앞"].sdev_q <= thr).values)[1]
    real2 = dt(t[t.half == "뒤"], (t[t.half == "뒤"].sdev_q <= thr).values)[1]
    h1 = (t.half == "앞").values
    for lab, keyset in (("날짜x세션 안", grp), ("날짜 안(세션 무시)", t.groupby("day").indices)):
        r1, r2 = [], []
        v = t.sdev_q.values
        for _ in range(200):
            z = v.copy()
            for ix in keyset.values():
                z[ix] = v[rng.permutation(ix)]
            a = z <= thr
            r1.append(tst(dpair(t.di.values[h1], t.ret.values[h1], a[h1]))[1])
            r2.append(tst(dpair(t.di.values[~h1], t.ret.values[~h1], a[~h1]))[1])
        r1, r2 = np.array(r1), np.array(r2)
        say(f"[{lab}] 실제 앞 t={real1:+.2f}, 뒤 t={real2:+.2f} | 섞은 것 앞 t 중앙값 {np.median(r1):+.2f}(≤{real1:.2f} 비율 {100 * (r1 <= real1).mean():.0f}%),"
            f" 뒤 t 중앙값 {np.median(r2):+.2f}(≤{real2:.2f} 비율 {100 * (r2 <= real2).mean():.0f}%), 둘 다 {100 * ((r1 <= real1) & (r2 <= real2)).mean():.0f}%")

    # ---- 5c) qdev 다른 날 QQQ 경로 ----
    say("\n## 5c) qdev를 다른 날 같은 시각 QQQ 이격으로 바꾸기 200회(날짜 순열, 자기 날 제외)")
    tab = qdev_table(q)
    chk = np.array([lookup_q(tab, d, m) for d, m in zip(t.day, t["mod"])])
    say(f"같은 날 조회값 vs 실제 qdev 스피어만 {pd.Series(chk).corr(t.qdev.reset_index(drop=True), method='spearman'):+.3f}, 부호 일치 {100 * np.mean(np.sign(chk) == np.sign(t.qdev.values)):.0f}% (조회 방식 검증)")
    real = score_split(t, (t.qdev < 0).values)
    rs_ = fe(t[~h1], (t.qdev < 0).values[~h1], ["day", "ses"])[1]
    say(f"실제: 앞 점수 {real[0]:.2f}, 방향 {'+' if real[1] > 0 else '-'}, 뒤 전체 t {real[2]:+.2f}, 뒤 정규 t {real[3]:+.2f}, 뒤 날짜x세션 FE t {rs_:+.2f}")
    res = []
    dl = np.array(days)
    for _ in range(200):
        while True:
            p = rng.permutation(len(dl))
            if not np.any(p == np.arange(len(dl))):
                break
        mp = dict(zip(dl, dl[p]))
        fq = np.array([lookup_q(tab, mp[d], m) for d, m in zip(t.day, t["mod"])])
        a = fq < 0
        sc, sgn, a2, r2 = score_split(t, a)
        s2 = sgn * fe(t[~h1], a[~h1], ["day", "ses"])[1] if sgn != 0 else np.nan
        res.append((sc, a2, r2, s2))
    res = pd.DataFrame(res, columns=["score", "all2", "reg2", "fe2"])
    say(f"가짜 qdev 200개: 앞 점수≥{real[0]:.2f} {int((res.score >= real[0]).sum())}개, 그중 뒤 전체 같은 방향 t≥{real[2]:.2f} {int(((res.score >= real[0]) & (res.all2 >= real[2])).sum())}개"
        f" | 조건 없이 뒤 전체 t≥{real[2]:.2f} {int((res.all2 >= real[2]).sum())}개, 뒤 날짜x세션 FE t≥{rs_:.2f} {int((res.fe2 >= rs_).sum())}개, |뒤 FE t|≥{abs(rs_):.2f} {int((res.fe2.abs() >= abs(rs_)).sum())}개")

    # ---- 7) C2 프리장 하위 결과(분석가 사후 분해) 견고성 ----
    say("\n## 7) C2 qdev<0 프리장만(분석가가 고른 범위가 아닌 사후 분해) 견고성")
    for lab, y in (("프리 원래", t[t.ses == "프리"]), ("프리 확인된 체결", t[(t.ses == "프리") & t.ok])):
        y2 = y[y.half == "뒤"]
        a = (y2.qdev < 0).values
        d = dpair(y2.di.values, y2.ret.values, a)
        m, tt, n = tst(d)
        yv = y2[y2.vt >= 0]
        av = a[(y2.vt >= 0).values]
        b3, t3, _ = fe(yv, av, ["day", "vt"])
        g3, m3 = drop_greedy(y2, a, 1, "sym", 3)
        gd, md = drop_greedy(y2, a, 1, "day", 1)
        cnt = y2.groupby("day").qdev.agg(lambda z: f"{int((z < 0).sum())}/{int((z >= 0).sum())}")
        say(f"[{lab}] 뒤 날짜차 {pct(m)} t={tt:+.2f}({n}일, 양수 {int((d > 0).sum())}일) | 날짜xvol FE t={t3:+.2f} | 수익/atr t={dt(yv, av, 'ra')[1]:+.2f}"
            f" | -{','.join(g3)} t={m3[1]:+.2f} | -{gd[0]} t={md[1]:+.2f} | 날짜별 A/B 건수 " + " ".join(f"{k[5:]}:{v}" for k, v in cnt.items()))
    pre = (t.ses == "프리").values
    tp = t[pre]
    h1p = (tp.half == "앞").values
    rp1 = tst(dpair(tp.di.values[h1p], tp.ret.values[h1p], (tp.qdev < 0).values[h1p]))[1]
    rp2 = tst(dpair(tp.di.values[~h1p], tp.ret.values[~h1p], (tp.qdev < 0).values[~h1p]))[1]
    f1, f2 = [], []
    for _ in range(200):
        while True:
            p = rng.permutation(len(dl))
            if not np.any(p == np.arange(len(dl))):
                break
        mp = dict(zip(dl, dl[p]))
        a = np.array([lookup_q(tab, mp[d], m) for d, m in zip(tp.day, tp["mod"])]) < 0
        f1.append(tst(dpair(tp.di.values[h1p], tp.ret.values[h1p], a[h1p]))[1])
        f2.append(tst(dpair(tp.di.values[~h1p], tp.ret.values[~h1p], a[~h1p]))[1])
    f1, f2 = np.array(f1), np.array(f2)
    say(f"프리만 가짜 qdev(다른 날 QQQ 경로) 200개: 실제 앞 t={rp1:+.2f}·뒤 t={rp2:+.2f} | 가짜 앞 t≥{rp1:.2f} {int((f1 >= rp1).sum())}개, 뒤 t≥{rp2:.2f} {int((f2 >= rp2).sum())}개,"
        f" 둘 다 {int(((f1 >= rp1) & (f2 >= rp2)).sum())}개 | 가짜 뒤 t 중앙값 {np.nanmedian(f2):+.2f}, 90% {np.nanquantile(f2, 0.9):+.2f}")

    # ---- 6) 종목 지속성 ----
    say("\n## 6) 종목 평균 수익 앞→뒤 스피어만(두 절반 각 3건 이상) — 확인된 체결·대체 매도가로")
    for lab, x, col in (("전체 원래", t, "ret"), ("전체 확인+대체", t[t.ok], "ret_alt"), ("정규 원래", t[t.ses == "정규"], "ret"),
                        ("정규 확인+대체", t[(t.ses == "정규") & t.ok], "ret_alt"), ("애프터 제외 원래", t[t.ses != "애프터"], "ret"),
                        ("애프터 제외 확인+대체", t[(t.ses != "애프터") & t.ok], "ret_alt")):
        x = x[x[col].notna()]
        g = x.groupby(["sym", "half"])[col].agg(["size", "mean"]).unstack("half")
        k = (g["size"]["앞"] >= 3) & (g["size"]["뒤"] >= 3)
        r = g["mean"]["앞"][k].corr(g["mean"]["뒤"][k], method="spearman")
        n = int(k.sum())
        say(f"{lab}: {r:+.2f}(t{r * np.sqrt((n - 2) / (1 - r * r)):+.1f}, {n}종목)")
    sh = t.groupby(["sym", "half"]).aft.mean().unstack()
    say(f"종목별 애프터 비중 앞→뒤 {sh['앞'].corr(sh['뒤'], method='spearman'):+.2f}")

    with open(os.path.join(RES, "verify_context.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
