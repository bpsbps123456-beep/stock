"""검증 #2 (2026-10-10, 연구용): K3 '극상방 = 팔 자리'는 SOXX 경고 주에만 성립한다 — 반박 시도.
zones.py·zones_regime.py는 고치지 않고 불러 쓴다. 결과를 본 뒤 정한 시험이 아니라 검증 지시(a~e)를 그대로 수행한다.

a) 베타 착시: 사전 60일 베타(종목 일수익 vs 동일가중 유니버스 / SOXX / QQQ, t일 종가까지)로
   잔차 = 종목 10일 수익 - 베타 × 같은 구간 기준 수익. T3(상방-극상방)를 잔차로 다시 재고 평상/경고 비교.
b) 집중: 경고 사건(연속 경고 구간) 수, 사건 하나씩 빼기, 주 단위·사건 단위 블록 부트스트랩 CI(경고 T3 - 평상 T3),
   전 구간 묶음에서 경고 주 순서를 원형으로 밀어 본 위약 시험(사건 구조 보존).
c) h 5·10·20, 경고 정의 6개(SOXX 주간 SMA20 원본, SOXX 일간 SMA20, SOXX 주간 SMA10·SMA50, QQQ 주간·일간 SMA20).
d) 반도체 / 비반도체(desk.config.SEMIS_SECTORS, desk.strategies.sector_of(sym, sym)).
e) 1위 설정의 극상방·상방 10일 수익(원수익·초과)을 평상/경고로 — 극상방이 나빠진 건지 상방이 좋아진 건지.
대상: 자격 107개 전체(중앙값·양수 비율)와 사전 등록 1위 mz(60)≥0.5 + xrsi≥70.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_k3_attack.py
결과: results/verify_k3_attack.txt, verify_k3_attack.json
"""
import json, os, sys, warnings
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import zones_regime as ZR  # noqa: E402

sys.path.insert(0, Zm.ROOT)
PER = ("앞", "뒤", "2025", "2024")
ALLP = PER + ("묶음",)
HS = (5, 10, 20)
NB = 2000
SEED = 20261010
pct = lambda v: "   nan" if not np.isfinite(v) else f"{v * 100:+6.2f}"


# ---------------------------------------------------------------- 기준 지수·경고 정의
def load_bench(dates):
    o = pd.read_parquet(ZR.SOXX_SRC, columns=["date", "sym", "Open", "Close"])
    s = o[o.sym == "SOXX"].copy()
    s["d"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
    s = s.drop_duplicates("d").set_index("d").sort_index()[["Open", "Close"]].astype(float)
    x = pd.read_parquet(Zm.SRC)
    q = x[x.sym == "QQQ"].copy()
    q["d"] = pd.to_datetime(q["ts"]).dt.strftime("%Y-%m-%d")
    q = q.drop_duplicates("d").set_index("d").sort_index()[["Open", "Close"]].astype(float)
    miss = {k: sorted(set(dates) - set(v.index)) for k, v in (("SOXX", s), ("QQQ", q))}
    return {"SOXX": s, "QQQ": q}, miss


def week_warn(close: pd.Series, n, dates):
    """zones_regime.soxx_week_warning과 같은 규칙을 기준·기간만 바꿔 일반화."""
    below = close < close.rolling(n, min_periods=n).mean()
    days = list(close.index)
    ws, cur, st = {}, None, False
    for i, d in enumerate(days):
        wk = date.fromisoformat(d).isocalendar()[:2]
        if wk != cur:
            cur = wk
            st = bool(below.iloc[i - 1]) if i > 0 else False
        ws[d] = st
    out = np.zeros(len(dates), dtype=bool)
    for i in range(len(dates) - 1):
        out[i] = ws.get(dates[i + 1], False)
    return out


def day_warn(close: pd.Series, n, dates):
    """t일 종가가 SMA 아래면 t 신호(→ t+1 시가부터)를 경고로."""
    below = (close < close.rolling(n, min_periods=n).mean()).astype(float)
    below[close.rolling(n, min_periods=n).mean().isna()] = np.nan
    return (below.reindex(dates).ffill().fillna(0.0).to_numpy() > 0.5)


# ---------------------------------------------------------------- 날짜별 단계 평균
def zone_means(Z, V, zs=(1, 2)):
    have = np.isfinite(V) & (Z != -9)
    m, n = {}, {}
    for z in zs:
        M = have & (Z == z)
        c = M.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m[z] = np.where(c > 0, np.where(M, V, 0).sum(axis=1) / np.maximum(c, 1), np.nan)
        n[z] = c
    return m, n


def nmean(D, rows):
    """D: (k, 날짜) → 행 subset 평균(nan 무시). rows가 비면 nan."""
    if len(rows) == 0:
        return np.full(D.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(D[:, rows], axis=1)


def split_stats(D, rw, rn, wi):
    tw, tn = nmean(D, rw), nmean(D, rn)
    df = tw - tn
    g = slice(0, wi_n[0])
    fin = lambda v: v[np.isfinite(v)]
    return {"1위 평상": float(tn[wi]), "1위 경고": float(tw[wi]), "1위 차": float(df[wi]),
            "격자 평상 양수%": float(np.mean(fin(tn[g]) > 0)), "격자 경고 양수%": float(np.mean(fin(tw[g]) > 0)),
            "격자 평상 중앙": float(np.median(fin(tn[g]))), "격자 경고 중앙": float(np.median(fin(tw[g]))),
            "격자 차 중앙": float(np.median(fin(df[g]))), "격자 차 양수%": float(np.mean(fin(df[g]) > 0)),
            "경고일": int(len(rw)), "평상일": int(len(rn))}


wi_n = [0]  # 격자 크기(자격 설정 수), main에서 채움


def diff_pair(D, rw, rn, wi):
    df = nmean(D, rw) - nmean(D, rn)
    g = df[: wi_n[0]]
    return float(df[wi]), float(np.median(g[np.isfinite(g)]))


def boot(D, rw, rn, bw, bn, wi, rng, nb=NB):
    """경고·평상 각각 블록(bw, bn: 행별 블록 id)을 복원 추출 → (1위 차, 격자 중앙 차) 분포."""
    def groups(rows, bid):
        u = {}
        for r, b in zip(rows, bid):
            u.setdefault(b, []).append(r)
        return [np.array(v) for v in u.values()]
    Gw, Gn = groups(rw, bw), groups(rn, bn)
    if len(Gw) < 2 or len(Gn) < 2:
        return None
    out1, out2 = np.empty(nb), np.empty(nb)
    for k in range(nb):
        iw = np.concatenate([Gw[j] for j in rng.integers(0, len(Gw), len(Gw))])
        inn = np.concatenate([Gn[j] for j in rng.integers(0, len(Gn), len(Gn))])
        out1[k], out2[k] = diff_pair(D, iw, inn, wi)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), float(np.nanmean(v <= 0))]
    return {"블록수(경고/평상)": [len(Gw), len(Gn)], "1위 CI·P(≤0)": q(out1), "격자중앙 CI·P(≤0)": q(out2)}


# ---------------------------------------------------------------- 실행
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    I = Zm.Ind(*Zm.load())
    dates = I.dates
    nd = len(dates)
    F = {h: Zm.Fwd(I, h) for h in HS}
    res = json.load(open(os.path.join(Zm.OUT, "zones.json"), encoding="utf-8"))
    cfgs = [g["cfg"] for g in res["grid"] if g["eligible"]]
    names = [Zm.cname(c) for c in cfgs]
    wi = names.index(res["best"])
    wi_n[0] = len(cfgs)
    allc = cfgs + [Zm.BASELINE]
    L = []
    J = {"격자수": len(cfgs), "1위": res["best"]}

    # 서로 다른 극상방 정의 수(격자 107개가 독립 표본이 아님을 보이기)
    ext = {(c["x"], c["xv"]) if c["fam"] == "tx" else ("dev", c["L"], c["w"], c["b"]) for c in cfgs}
    xr70 = sum(1 for c in cfgs if c["fam"] == "tx" and c["x"] == "xrsi" and c["xv"] == 70)
    L.append(f"자격 {len(cfgs)}개 · 서로 다른 극상방 정의 {len(ext)}개 · 그중 RSI≥70 극상방을 쓰는 설정 {xr70}개")
    J["극상방정의수"] = len(ext)

    # 기준 지수
    B, miss = load_bench(dates)
    L.append(f"기준 지수 날짜 누락(주식 달력 기준): SOXX {len(miss['SOXX'])}일, QQQ {len(miss['QQQ'])}일")
    bO = {k: v["Open"].reindex(dates) for k, v in B.items()}
    bC = {k: v["Close"].reindex(dates).ffill() for k, v in B.items()}

    # 경고 정의
    W = {"SOXX주SMA20(원본)": week_warn(B["SOXX"]["Close"], 20, dates),
         "SOXX일SMA20": day_warn(B["SOXX"]["Close"], 20, dates),
         "SOXX주SMA10": week_warn(B["SOXX"]["Close"], 10, dates),
         "SOXX주SMA50": week_warn(B["SOXX"]["Close"], 50, dates),
         "QQQ주SMA20": week_warn(B["QQQ"]["Close"], 20, dates),
         "QQQ일SMA20": day_warn(B["QQQ"]["Close"], 20, dates)}
    orig = ZR.soxx_week_warning(dates)
    assert (orig == W["SOXX주SMA20(원본)"]).all(), "원본 경고 재현 실패"
    W0 = W["SOXX주SMA20(원본)"]

    # 행 집합
    rows = {(p, h): Zm.period_rows(I, p, h) for p in PER for h in HS}
    for h in HS:
        rows["묶음", h] = np.unique(np.concatenate([rows[p, h] for p in PER]))
    wk_of = [date.fromisoformat(dates[min(t + 1, nd - 1)]).isocalendar()[:2] for t in range(nd)]
    wk_u = {w: i for i, w in enumerate(sorted(set(wk_of)))}
    wkidx = np.array([wk_u[w] for w in wk_of])

    def segs(warn):
        return np.r_[0, np.cumsum(warn[1:] != warn[:-1])]

    # 베타(사전 60일, t일 종가까지)
    r = I.C.pct_change(fill_method=None)
    rb = {"EW": r.where(I.valid).mean(axis=1), "SOXX": bC["SOXX"].pct_change(fill_method=None),
          "QQQ": bC["QQQ"].pct_change(fill_method=None)}
    beta = {}
    for k, b in rb.items():
        mxy = r.mul(b, axis=0).rolling(60, min_periods=50).mean()
        mx = r.rolling(60, min_periods=50).mean()
        my = b.rolling(60, min_periods=50).mean()
        vy = (b ** 2).rolling(60, min_periods=50).mean() - my ** 2
        beta[k] = (mxy.sub(mx.mul(my, axis=0))).div(vy, axis=0).to_numpy()
    bfwd = {}
    for h in HS:
        bfwd[h] = {"EW": np.nanmean(F[h].raw, axis=1)}
        for k in ("SOXX", "QQQ"):
            o = bO[k]
            bfwd[h][k] = (o.shift(-(1 + h)) / o.shift(-1) - 1).to_numpy()

    # 설정별 날짜 T3 시계열
    D3 = {h: np.full((len(allc), nd), np.nan) for h in HS}          # 원본(초과)
    D3c = {h: {} for h in HS}                                        # 베타 공통 표본 원수익
    D3b = {h: {} for h in HS}                                        # 베타 잔차
    Bg = {}                                                          # 1위·격자: 극상방-상방 평균 베타 차(날짜별)
    M1 = {h: np.full((len(allc), nd), np.nan) for h in HS}
    M2 = {h: np.full((len(allc), nd), np.nan) for h in HS}
    common = {h: np.isfinite(F[h].raw) & np.isfinite(beta["EW"]) & np.isfinite(beta["SOXX"]) & np.isfinite(beta["QQQ"])
              for h in HS}
    for h in HS:
        for k in ("EW", "SOXX", "QQQ"):
            D3b[h][k] = np.full((len(allc), nd), np.nan)
            Bg.setdefault(k, np.full((len(allc), nd), np.nan))
        D3c[h] = np.full((len(allc), nd), np.nan)
    semis_mask = None
    from desk import config
    from desk.strategies import sector_of
    semis_mask = np.array([sector_of(s, s) in config.SEMIS_SECTORS for s in I.syms])
    D3s = {lab: np.full((len(allc), nd), np.nan) for lab in ("반도체", "비반도체")}
    Zwin = None
    for ci, c in enumerate(allc):
        Z = Zm.zones(I, c)
        if ci == wi:
            Zwin = Z
        for h in HS:
            m, _ = zone_means(Z, F[h].ex)
            D3[h][ci] = m[1] - m[2]
            M1[h][ci], M2[h][ci] = m[1], m[2]
            raw_c = np.where(common[h], F[h].raw, np.nan)
            m, _ = zone_means(Z, raw_c)
            D3c[h][ci] = m[1] - m[2]
            for k in ("EW", "SOXX", "QQQ"):
                res_k = np.where(common[h], F[h].raw - beta[k] * bfwd[h][k][:, None], np.nan)
                m, _ = zone_means(Z, res_k)
                D3b[h][k][ci] = m[1] - m[2]
                if h == 10:
                    mb, _ = zone_means(Z, np.where(common[h], beta[k], np.nan))
                    Bg[k][ci] = mb[2] - mb[1]
        for lab, msk in (("반도체", semis_mask), ("비반도체", ~semis_mask)):
            Zs = np.where(msk[None, :], Z, -9).astype(np.int8)
            m, _ = zone_means(Zs, F[10].ex)
            D3s[lab][ci] = m[1] - m[2]

    # ---- 0. 재현 확인
    zr = json.load(open(os.path.join(Zm.OUT, "zones_regime.json"), encoding="utf-8"))
    mx = 0.0
    for ci, c in enumerate(allc):
        nm = Zm.cname(c)
        for p in PER:
            rp = rows[p, 10]
            for s, sel in (("경고", rp[W0[rp]]), ("평상", rp[~W0[rp]])):
                v = np.nanmean(D3[10][ci, sel])
                ref = zr[nm][p][s]["T3 상방-극상방"]
                if np.isfinite(ref):
                    mx = max(mx, abs(v - ref))
    L.append(f"[재현] zones_regime.json T3와 최대 차 {mx:.2e} (0이면 같은 계산)")
    J["재현최대차"] = mx

    def per_split(p, h, warn):
        rp = rows[p, h]
        return rp[warn[rp]], rp[~warn[rp]]

    # ---- e. 1위 단계별 10일 수익(원·초과), 평상/경고
    L += ["", "== e. 1위 단계별 10일 수익: 평상 vs 경고 (날짜 평균, %, 괄호=그 단계가 있는 날 수·하루 평균 종목 수)"]
    zs5 = (2, 1, 0, -1, -2)
    me, ne = zone_means(Zwin, F[10].ex, zs5)
    mr, _ = zone_means(Zwin, F[10].raw, zs5)
    J["e"] = {}
    for p in ALLP:
        rw, rn = per_split(p, 10, W0)
        J["e"][p] = {}
        for s, rr in (("평상", rn), ("경고", rw)):
            parts, d = [], {}
            for z in zs5:
                sel = rr[ne[z][rr] > 0]
                d[Zm.NAMES[z]] = {"원수익": float(np.nanmean(mr[z][sel])) if len(sel) else np.nan,
                                  "초과": float(np.nanmean(me[z][sel])) if len(sel) else np.nan,
                                  "날짜수": int(len(sel)), "평균종목수": float(ne[z][sel].mean()) if len(sel) else 0.0}
            both = rr[(ne[1][rr] > 0) & (ne[2][rr] > 0)]
            d["T3날짜"] = {"날짜수": int(len(both)), "상방초과": float(np.nanmean(me[1][both])),
                         "극상방초과": float(np.nanmean(me[2][both])), "상방원수익": float(np.nanmean(mr[1][both])),
                         "극상방원수익": float(np.nanmean(mr[2][both])),
                         "유니버스원수익": float(np.nanmean(bfwd[10]["EW"][both])),
                         "SOXX": float(np.nanmean(bfwd[10]["SOXX"][both])), "QQQ": float(np.nanmean(bfwd[10]["QQQ"][both]))}
            J["e"][p][s] = d
            t = d["T3날짜"]
            L.append(f"[{p}·{s}] T3날짜 {t['날짜수']}일: 극상방 원{pct(t['극상방원수익'])} 초과{pct(t['극상방초과'])} | "
                     f"상방 원{pct(t['상방원수익'])} 초과{pct(t['상방초과'])} | 유니버스{pct(t['유니버스원수익'])} "
                     f"SOXX{pct(t['SOXX'])} QQQ{pct(t['QQQ'])}  (극상방 하루 {d['극상방']['평균종목수']:.1f}종목, "
                     f"상방 {d['상방']['평균종목수']:.1f}종목)")
        a, b = J["e"][p]["평상"]["T3날짜"], J["e"][p]["경고"]["T3날짜"]
        L.append(f"      경고-평상 변화: 상방 초과 {pct(b['상방초과'] - a['상방초과'])}, 극상방 초과 {pct(b['극상방초과'] - a['극상방초과'])}"
                 f"  → T3 변화 {pct((b['상방초과'] - b['극상방초과']) - (a['상방초과'] - a['극상방초과']))}")
    # 격자 전체: 극상방·상방 초과의 중앙값
    L += ["", "-- e'. 자격 107개 중앙값: 극상방·상방 초과(%, T3 날짜 기준) 평상 → 경고"]
    for p in ALLP:
        rw, rn = per_split(p, 10, W0)
        out = []
        for s, rr in (("평상", rn), ("경고", rw)):
            ok = np.isfinite(M1[10][:, rr]) & np.isfinite(M2[10][:, rr])
            m1 = np.array([np.nanmean(M1[10][ci, rr][ok[ci]]) for ci in range(len(cfgs))])
            m2 = np.array([np.nanmean(M2[10][ci, rr][ok[ci]]) for ci in range(len(cfgs))])
            out.append(f"{s} 극상방{pct(np.nanmedian(m2))} 상방{pct(np.nanmedian(m1))}")
        L.append(f"[{p}] " + " | ".join(out))

    # ---- a. 베타 착시
    L += ["", "== a. 베타 조정 T3 (h10). 원=같은 표본 원수익 T3, 조정=종목 수익-사전60일 베타×기준 수익",
          "   형식: 1위 평상/경고 | 격자 중앙 평상/경고 · 경고 양수% · 차 중앙 | 극상방-상방 베타 차(1위, 평상/경고), 기준 10일 수익 평상/경고"]
    J["a"] = {}
    for k in ("원", "EW", "SOXX", "QQQ"):
        J["a"][k] = {}
        for p in ALLP:
            rw, rn = per_split(p, 10, W0)
            D = D3c[10] if k == "원" else D3b[10][k]
            st = split_stats(D, rw, rn, wi)
            extra = ""
            if k != "원":
                bgn, bgw = np.nanmean(Bg[k][wi, rn]), np.nanmean(Bg[k][wi, rw])
                bgn_g, bgw_g = np.nanmedian(nmean(Bg[k][:len(cfgs)], rn)), np.nanmedian(nmean(Bg[k][:len(cfgs)], rw))
                fn, fw = np.nanmean(bfwd[10][k][rn]), np.nanmean(bfwd[10][k][rw])
                st.update({"베타차 1위 평상": float(bgn), "베타차 1위 경고": float(bgw), "베타차 격자중앙 평상": float(bgn_g),
                           "베타차 격자중앙 경고": float(bgw_g), "기준수익 평상": float(fn), "기준수익 경고": float(fw)})
                extra = (f" | β차 {bgn:+.2f}/{bgw:+.2f} (격자 {bgn_g:+.2f}/{bgw_g:+.2f}) 기준{pct(fn)}/{pct(fw)}")
            J["a"][k][p] = st
            L.append(f"[{k:>4}·{p}] 1위 {pct(st['1위 평상'])}/{pct(st['1위 경고'])} | 격자 {pct(st['격자 평상 중앙'])}/"
                     f"{pct(st['격자 경고 중앙'])} 경고양수 {st['격자 경고 양수%'] * 100:3.0f}% 평상양수 {st['격자 평상 양수%'] * 100:3.0f}% "
                     f"차중앙 {pct(st['격자 차 중앙'])}{extra}")

    # ---- b. 집중: 사건, 하나씩 빼기, 부트스트랩, 위약
    rng = np.random.default_rng(SEED)
    sg = segs(W0)
    L += ["", "== b. 경고 사건 집중 (원본 경고, h10, 초과). 사건 = 연속 경고 구간"]
    J["b"] = {}
    D = D3[10]
    for p in ALLP:
        rw, rn = per_split(p, 10, W0)
        base1, base2 = diff_pair(D, rw, rn, wi)
        eps = sorted(set(sg[rw]))
        epl = []
        for e in eps:
            er = rw[sg[rw] == e]
            t1 = float(np.nanmean(D[wi, er])) if np.isfinite(D[wi, er]).any() else np.nan
            g = nmean(D[:len(cfgs)], er)
            epl.append({"시작": dates[er[0]], "끝": dates[er[-1]], "신호일": int(len(er)),
                        "1위T3유효일": int(np.isfinite(D[wi, er]).sum()), "1위T3": t1,
                        "격자중앙T3": float(np.nanmedian(g)) if np.isfinite(g).any() else np.nan,
                        "빼면 1위 차": diff_pair(D, rw[sg[rw] != e], rn, wi)[0],
                        "빼면 격자중앙 차": diff_pair(D, rw[sg[rw] != e], rn, wi)[1]})
        bw = boot(D, rw, rn, wkidx[rw], wkidx[rn], wi, rng)
        be = boot(D, rw, rn, sg[rw], sg[rn], wi, rng)
        J["b"][p] = {"1위 차": base1, "격자중앙 차": base2, "사건": epl, "주블록": bw, "사건블록": be}
        L.append(f"[{p}] 경고-평상 T3 차: 1위 {pct(base1)}, 격자 중앙 {pct(base2)} · 경고 사건 {len(eps)}개")
        for e in epl:
            L.append(f"   {e['시작']}~{e['끝']} {e['신호일']:3d}일(1위 유효 {e['1위T3유효일']:3d}) 1위T3 {pct(e['1위T3'])} "
                     f"격자중앙T3 {pct(e['격자중앙T3'])} | 빼면 차: 1위 {pct(e['빼면 1위 차'])} 격자 {pct(e['빼면 격자중앙 차'])}")
        if epl:
            v1 = [e["빼면 1위 차"] for e in epl]
            v2 = [e["빼면 격자중앙 차"] for e in epl]
            L.append(f"   하나씩 빼기 범위: 1위 {pct(np.nanmin(v1))}~{pct(np.nanmax(v1))}, 격자 중앙 {pct(np.nanmin(v2))}~{pct(np.nanmax(v2))}")
        for lab, bb in (("주 블록", bw), ("사건 블록", be)):
            if bb:
                a1, a2 = bb["1위 CI·P(≤0)"], bb["격자중앙 CI·P(≤0)"]
                L.append(f"   {lab} 부트스트랩 {NB}회 (블록 {bb['블록수(경고/평상)']}): 1위 95% [{pct(a1[0])},{pct(a1[1])}] P(≤0) {a1[2]:.3f}"
                         f" | 격자 중앙 [{pct(a2[0])},{pct(a2[1])}] P(≤0) {a2[2]:.3f}")
            else:
                L.append(f"   {lab} 부트스트랩: 블록 2개 미만이라 생략")
    # 위약: 묶음 구간 주 단위 경고 순서를 원형 이동
    rp = rows["묶음", 10]
    wks = np.unique(wkidx[rp])
    wstate = {w: bool(W0[rp][wkidx[rp] == w][0]) for w in wks}
    seq = np.array([wstate[w] for w in wks])
    pos = {w: j for j, w in enumerate(wks)}
    act1, act2 = diff_pair(D, rp[W0[rp]], rp[~W0[rp]], wi)
    sh = []
    for k in range(1, len(wks)):
        s2 = np.roll(seq, k)
        wv = np.array([s2[pos[w]] for w in wkidx[rp]])
        d1, d2 = diff_pair(D, rp[wv], rp[~wv], wi)
        sh.append((k, d1, d2))
    sh = np.array(sh)
    far = np.minimum(sh[:, 0], len(wks) - sh[:, 0]) >= 4
    p1, p2 = float(np.mean(sh[far, 1] >= act1)), float(np.mean(sh[far, 2] >= act2))
    L.append(f"[위약] 묶음 {len(wks)}주, 경고 주 순서를 4주 이상 원형 이동 {int(far.sum())}가지: 실제 차 1위 {pct(act1)} → 이동 중 이상 비율 {p1:.3f}"
             f" (이동 분포 중앙 {pct(np.median(sh[far, 1]))}) | 격자 중앙 {pct(act2)} → {p2:.3f} (이동 분포 중앙 {pct(np.median(sh[far, 2]))})")
    J["b"]["위약"] = {"주수": int(len(wks)), "이동수": int(far.sum()), "1위 실제": act1, "1위 p": p1, "격자 실제": act2, "격자 p": p2,
                    "1위 이동중앙": float(np.median(sh[far, 1])), "격자 이동중앙": float(np.median(sh[far, 2]))}

    # ---- c. h·경고 정의 바꾸기 (전부 보고)
    L += ["", "== c. 견고성: 경고 정의 6개 × h 5·10·20 (초과). 1위 평상/경고(차) | 격자 경고양수%·평상양수% · 차 중앙 · 차 양수%",
          "   묶음 열에 주 블록 부트스트랩(1000회) 95% CI(1위 차 / 격자 중앙 차)"]
    J["c"] = {}
    rng2 = np.random.default_rng(SEED + 1)
    for wn, warn in W.items():
        for h in HS:
            key = f"{wn}·h{h}"
            J["c"][key] = {}
            for p in ALLP:
                rw, rn = per_split(p, h, warn)
                st = split_stats(D3[h], rw, rn, wi)
                if p == "묶음":
                    st["주블록"] = boot(D3[h], rw, rn, wkidx[rw], wkidx[rn], wi, rng2, nb=1000)
                J["c"][key][p] = st
            row = []
            for p in ALLP:
                st = J["c"][key][p]
                row.append(f"{p} {pct(st['1위 평상'])}/{pct(st['1위 경고'])} {st['격자 경고 양수%'] * 100:3.0f}%·{st['격자 평상 양수%'] * 100:3.0f}%"
                           f" 중앙차{pct(st['격자 차 중앙'])} 차+{st['격자 차 양수%'] * 100:3.0f}%")
            bb = J["c"][key]["묶음"]["주블록"]
            ci = (f"CI 1위 [{pct(bb['1위 CI·P(≤0)'][0])},{pct(bb['1위 CI·P(≤0)'][1])}] 격자 [{pct(bb['격자중앙 CI·P(≤0)'][0])},"
                  f"{pct(bb['격자중앙 CI·P(≤0)'][1])}]") if bb else ""
            L.append(f"{key} 경고비율 {np.mean(warn[rows['묶음', h]]) * 100:.0f}%")
            L.append("   " + " | ".join(row[:3]))
            L.append("   " + " | ".join(row[3:]) + "  " + ci)

    # ---- d. 반도체 / 비반도체
    L += ["", f"== d. 반도체 {int(semis_mask.sum())}종목 / 비반도체 {int((~semis_mask).sum())}종목 (원본 경고, h10, 초과). 1위 평상/경고 | 격자 중앙 평상/경고 · 경고양수% · 평상양수%"]
    J["d"] = {}
    for lab in ("반도체", "비반도체"):
        J["d"][lab] = {}
        for p in ALLP:
            rw, rn = per_split(p, 10, W0)
            st = split_stats(D3s[lab], rw, rn, wi)
            st["1위 유효일(평상/경고)"] = [int(np.isfinite(D3s[lab][wi, rn]).sum()), int(np.isfinite(D3s[lab][wi, rw]).sum())]
            J["d"][lab][p] = st
            L.append(f"[{lab}·{p}] 1위 {pct(st['1위 평상'])}/{pct(st['1위 경고'])} (유효일 {st['1위 유효일(평상/경고)']}) | 격자 "
                     f"{pct(st['격자 평상 중앙'])}/{pct(st['격자 경고 중앙'])} 경고양수 {st['격자 경고 양수%'] * 100:3.0f}% "
                     f"평상양수 {st['격자 평상 양수%'] * 100:3.0f}%")

    # ---- f. 수준 CI: 차가 아니라 경고 주 T3 자체가 0보다 큰가(지표 문구를 정하려면 이것이 필요)
    L += ["", "== f. 수준 CI (묶음, h10, 주 블록 2000회): T3 = 상방-극상방, 극상방 초과 = 극상방 - 유니버스"]
    J["f"] = {}
    rng3 = np.random.default_rng(SEED + 2)
    rp = rows["묶음", 10]

    def level_boot(D, rr, nb=NB):
        G = {}
        for r_, b_ in zip(rr, wkidx[rr]):
            G.setdefault(b_, []).append(r_)
        G = [np.array(v) for v in G.values()]
        o1, o2 = np.empty(nb), np.empty(nb)
        for k in range(nb):
            ii = np.concatenate([G[j] for j in rng3.integers(0, len(G), len(G))])
            v = nmean(D, ii)
            o1[k] = v[wi]
            g = v[: len(cfgs)]
            o2[k] = np.median(g[np.isfinite(g)])
        v = nmean(D, rr)
        g = v[: len(cfgs)]
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        return {"1위": float(v[wi]), "1위 CI": ci(o1), "격자중앙": float(np.median(g[np.isfinite(g)])), "격자중앙 CI": ci(o2)}

    M2x = np.where(np.isfinite(D3[10]), M2[10], np.nan)  # T3 날짜에 맞춘 극상방 초과
    for lab, D in (("T3 원본", D3[10]), ("T3 SOXX베타조정", D3b[10]["SOXX"]), ("T3 EW베타조정", D3b[10]["EW"]),
                   ("극상방 초과", M2x), ("T3 반도체", D3s["반도체"]), ("T3 비반도체", D3s["비반도체"])):
        J["f"][lab] = {}
        for s, rr in (("평상", rp[~W0[rp]]), ("경고", rp[W0[rp]])):
            st = level_boot(D, rr)
            J["f"][lab][s] = st
            L.append(f"[{lab}·{s}] 1위 {pct(st['1위'])} [{pct(st['1위 CI'][0])},{pct(st['1위 CI'][1])}] | 격자 중앙 "
                     f"{pct(st['격자중앙'])} [{pct(st['격자중앙 CI'][0])},{pct(st['격자중앙 CI'][1])}]")
    # 반도체 차 부트스트랩
    for lab in ("반도체", "비반도체"):
        bb = boot(D3s[lab], rp[W0[rp]], rp[~W0[rp]], wkidx[rp[W0[rp]]], wkidx[rp[~W0[rp]]], wi, rng3)
        J["f"][f"{lab} 차 주블록"] = bb
        L.append(f"[{lab} 경고-평상 차] 1위 95% [{pct(bb['1위 CI·P(≤0)'][0])},{pct(bb['1위 CI·P(≤0)'][1])}] | 격자 중앙 "
                 f"[{pct(bb['격자중앙 CI·P(≤0)'][0])},{pct(bb['격자중앙 CI·P(≤0)'][1])}]")
    # 마지막 신호일의 경고 상태(다음 주에 적용될 값)와 1위 극상방 종목
    sc = B["SOXX"]["Close"]
    sma = sc.rolling(20).mean()
    L.append(f"[현재] {sc.index[-1]} SOXX 종가 {sc.iloc[-1]:.2f}, SMA20 {sma.iloc[-1]:.2f} → 다음 주 경고 {bool(sc.iloc[-1] < sma.iloc[-1])}; "
             f"이번 주 경고 {bool(W0[-2])}; 1위 극상방 종목: " + ", ".join(I.syms[j] for j in range(len(I.syms)) if Zwin[-1, j] == 2))

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(Zm.OUT, "verify_k3_attack.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(Zm.OUT, "verify_k3_attack.json"), "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))


if __name__ == "__main__":
    main()
