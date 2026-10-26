"""갈래 K(레드팀): 1차 8갈래(A~H) 방법론 반박·완결성 검사(2026-10-10).

공유 파일은 고치지 않는다. v2_K_regen이 각 갈래 함수로 전 변형(1,299개)을 다시 만들고 results/*.json과 대조한다.
검사
  a 시점: at_open 단위 시험, '미래 자료 교란' 시험(T 뒤 자료를 흔들어 T+1 시가 노출이 그대로인지), FRED 영업일 지연, PDF 시점, StockEngine
  b 갈래 간 모순: D 상태형 vs G 하향돌파+타이머(SOXX<SMA50), C vs F·H(그림자 변동성 60%)
  c AI쇼크 구간(사후 고점·저점): 반등 포함 구간·같은 평균 노출 고정·이동 27일 최악과 비교
  d 앞 고르기: 앞 낙폭이 어느 하락에서 나왔나, 대체 고르기 4종
  e 보유 기간: 투자 비중·미분류·변동성
  f 현금 이자 연 4%: 전 변형 다시 고르기
  g 원화: USD 현금 / 원화 현금
  h 빠진 방법: 고정 노출 효율선, 종가 직후(프리장·주간거래) 체결 상한
실행: cd code && PYTHONIOENCODING=utf-8 py v2_K_critic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import v2_K_regen as R  # noqa: E402

RATE = 0.04
OUT = H.RESULTS / "v2_K_critic.json"
FAMS = "ABCDEFGH"
AI = ("2026-06-22", "2026-07-29")
AI_PLUS = ("2026-06-22", "2026-09-30")
FEB_EP = ("2026-02-03", "2026-02-05")

# 갈래별 앞 고르기·'관찰'·사후 후보(각 md에 나온 이름 그대로)
FOCUS = {
    "A": ["DD-6%→저점+10% 노출0.5"],
    "B": ["장중 추적 15% / 재조정까지 현금"],
    "C": ["하방20|0.6|비대칭10", "하방20|0.4|주1회"],
    "D": ["A RSP<SMA10 →0.0", "A SMH<SMA10 →0.5", "C SOXX ret5≤-12% →0.0 복귀>SMA20"],
    "E": ["MOVE20d>+20% | 노출0.5 | 해제즉시", "HYG/IEF<SMA50 | 노출0.5 | 최소5일", "HYG/IEF<SMA50 | 노출0 | 최소5일"],
    "F": ["현금|T1_SOXX<SMA50|SOXS|q=0.2", "헤지|T1_SOXX<SMA50|SOXS|q=0.2"],
    "G": ["P3 SOXX<SMA50 → time20", "P2 X10% (ii)산다 D15%"],
    "H": ["펀드[TIME semiD2.0] 노출0.5 유지10", "펀드[KoAct cashL2.0] 노출0.5 유지10", "사건[반도체4] gap 노출0.5"],
}


def rnd(x, n=4):
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def st(r, lo, hi):
    return H.stats(r, lo, hi)


def win_metrics(bn, r):
    b = H.base(bn)
    return {k: st(r, *v) for k, v in b.periods.items()}


def row_metrics(row, rate=0.0, mod=None):
    """harness.select_front가 읽는 모양({'2026': {'앞': {...}}, 'hold': ...})."""
    out = {"name": row["name"]}
    for bn in R.BNS:
        r = R.returns(row, bn, rate)
        if mod is not None and bn == "2026":
            r = mod(r)
        m = win_metrics(bn, r)
        m["평균노출"] = float(row["e"][bn].reindex(H.base(bn).ret.index).mean())
        out[bn] = m
    return out


# ============================================================================ a 시점
def test_at_open():
    b = H.base("2026")
    d0 = b.days[10]
    sig = pd.Series([0.3], index=[d0])
    e = H.at_open(sig, b, fill=1.0)
    ok = (e[d0] == 1.0) and (e[b.days[11]] == 0.3)
    # 주말 날짜(토) 값도 다음 거래일 시가에 쓰이는지
    sat = (pd.Timestamp(b.days[14]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sig2 = pd.Series([0.7], index=[sat])
    e2 = H.at_open(sig2, b, fill=1.0)
    nxt = next(d for d in b.days if d > sat)
    return {"같은날 미사용·다음날 사용": bool(ok), "주말 값 → 다음 거래일": bool(e2[nxt] == 0.7),
            "검사일": [d0, b.days[11], sat, nxt]}


def _perturb_env(T: str, seed: int = 7, sigma: float = 0.05):
    """T보다 뒤 날짜의 모든 가격·기본 경로·시장폭·FRED를 무작위로 흔든 환경을 harness에 끼운다. 원래 값을 돌려준다."""
    rng = np.random.default_rng(seed)
    saved = {"ohlc": H._OHLC, "bases": dict(H._BASES), "breadth": H.breadth, "fred": H.fred_known}
    o = H.ohlc().copy()
    m = o["date"] > T
    f = np.exp(rng.normal(0, sigma, m.sum()))
    for c in ("Open", "High", "Low", "Close", "Adj Close"):
        o.loc[m, c] = o.loc[m, c].to_numpy() * f
    H._OHLC = o
    for bn in ("2026", "hold"):
        b = saved["bases"].get(bn) or H.base(bn)
        op, cl = np.array(b.open, float), np.array(b.close, float)
        mm = np.array([d > T for d in b.days])
        op[mm] *= np.exp(rng.normal(0, sigma, mm.sum()))
        cl[mm] *= np.exp(rng.normal(0, sigma, mm.sum()))
        H._BASES[bn] = H.Base(bn, b.days, op, cl, b.weights, b.ticker, b.sector)
    br0 = saved["breadth"]()
    br1 = br0.copy()
    mm = br1.index > T
    br1.loc[mm] = br1.loc[mm] * np.exp(rng.normal(0, sigma, br1.loc[mm].shape))
    H.breadth = lambda: br1
    fr0 = saved["fred"]()
    fr1 = fr0.copy()
    mm = fr1.index > T
    fr1.loc[mm] = fr1.loc[mm] + rng.normal(0, 0.3, fr1.loc[mm].shape)
    H.fred_known = lambda: fr1
    return saved


def _restore(saved):
    H._OHLC = saved["ohlc"]
    H._BASES.clear()
    H._BASES.update(saved["bases"])
    H.breadth = saved["breadth"]
    H.fred_known = saved["fred"]


def lookahead_test(clean: dict, T: str, bn: str) -> dict:
    """clean = {fam: rows}. T 뒤 자료를 흔든 뒤 노출을 다시 만들고, T 다음 첫 거래일 시가까지 노출이 같은지."""
    days = H.base(bn).days
    upto = next(d for d in days if d > T)
    saved = _perturb_env(T)
    out = {}
    try:
        for fam in "ACDEFGH":
            rows = R.GEN[fam]()
            ref = {r["name"]: r for r in clean[fam]}
            bad, changed_after = [], 0
            for r in rows:
                a = r["e"][bn][r["e"][bn].index <= upto].to_numpy(float)
                c = ref[r["name"]]["e"][bn][ref[r["name"]]["e"][bn].index <= upto].to_numpy(float)
                if not np.allclose(a, c, atol=1e-12, equal_nan=True):
                    bad.append(r["name"])
                hs = (r["h"] or {}).get(bn) or {}
                for sym, h in hs.items():
                    h0 = ref[r["name"]]["h"][bn][sym]
                    if not np.allclose(h[h.index <= upto].to_numpy(float), h0[h0.index <= upto].to_numpy(float)):
                        bad.append(r["name"] + " 헤지")
                aa = r["e"][bn][r["e"][bn].index > upto].to_numpy(float)
                cc = ref[r["name"]]["e"][bn][ref[r["name"]]["e"][bn].index > upto].to_numpy(float)
                changed_after += int(not np.allclose(aa, cc))
            out[fam] = {"변형": len(rows), "T+1까지 달라진 변형": len(bad), "예": bad[:5],
                        "T+1 뒤 달라진 변형(교란이 먹었는지)": changed_after}
    finally:
        _restore(saved)
        import fam_A_portfolio_dd as A, fam_D_trend_breadth as D, fam_G_hold_reentry as G
        A._FEAT.clear(); D._BR.clear(); G._SIG.clear()
    return {"T": T, "base": bn, "비교 끝(포함)": upto, "갈래": out}


def lookahead_engine(T: str = "2026-07-10") -> dict:
    """B StockEngine: T 뒤 고가·저가·시가·종가를 흔들어도 T까지의 사건(손절·재진입)과 T+1 손절선이 같은지."""
    import fam_B_stock_stops as B
    eng = B.engine("2026")
    kT = eng.k_of[T]
    k1 = eng.k_of[next(d for d in eng.days if d > T)]
    rng = np.random.default_rng(11)
    orig = {a: {k: v.copy() for k, v in getattr(eng, a).items()} for a in ("O", "C", "H", "L")}
    res = {"T": T, "변형": 0, "T까지 사건 다름": 0, "T+1 손절선 다름": 0, "예": []}
    clean = {}
    for sp in B.grid():
        pol = B.StopPolicy(eng, *sp)
        _, ev = eng.run(pol)
        clean[sp] = ev
    for a in ("O", "C", "H", "L"):
        for k in getattr(eng, a):
            arr = np.array(getattr(eng, a)[k], float, copy=True)
            n = len(arr) - (kT + 1)
            if n > 0:
                arr[kT + 1:] = arr[kT + 1:] * np.exp(rng.normal(0, 0.05, n))
            getattr(eng, a)[k] = arr
    eng._atr = {}
    try:
        for sp in B.grid():
            pol = B.StopPolicy(eng, *sp)
            _, ev = eng.run(pol)
            res["변형"] += 1
            a = [x for x in clean[sp] if x["day"] <= T]
            c = [x for x in ev if x["day"] <= T]
            if json.dumps(a, default=str) != json.dumps(c, default=str):
                res["T까지 사건 다름"] += 1
                res["예"].append(str(sp))
            d1 = next(d for d in eng.days if d > T)
            la = {x["key"]: x.get("level") for x in clean[sp] if x["day"] == d1 and x["type"] == "stop"}
            lc = {x["key"]: x.get("level") for x in ev if x["day"] == d1 and x["type"] == "stop"}
            for key in set(la) & set(lc):
                if la[key] is not None and abs(la[key] - lc[key]) > 1e-9:
                    res["T+1 손절선 다름"] += 1
    finally:
        for a in ("O", "C", "H", "L"):
            setattr(eng, a, orig[a])
    return res


def fred_lag_test(clean_E: list) -> dict:
    rows_b = R.fam_E(fred_extra_bday=True)
    ref = {r["name"]: r for r in clean_E}
    out = []
    for r in rows_b:
        if not r["name"].startswith("HY"):
            continue
        m0, m1 = row_metrics(ref[r["name"]]), row_metrics(r)
        out.append({"name": r["name"], "앞낙폭": [m0["2026"]["앞"]["낙폭"], m1["2026"]["앞"]["낙폭"]],
                    "전체수익": [m0["2026"]["전체"]["수익"], m1["2026"]["전체"]["수익"]],
                    "2025수익": [m0["hold"]["2025"]["수익"], m1["hold"]["2025"]["수익"]],
                    "관세": [m0["hold"]["관세"]["수익"], m1["hold"]["관세"]["수익"]]})
    diff = max(abs(x["전체수익"][1] - x["전체수익"][0]) for x in out)
    d25 = max(abs(x["2025수익"][1] - x["2025수익"][0]) for x in out)
    import fam_E_macro as E
    hy = R.fred_known_bday()["BAMLH0A0HYM2"].dropna()
    b = H.base("2026")
    rowsE = rnd(sorted(out, key=lambda x: -abs(x["2025수익"][1] - x["2025수익"][0]))[:5])
    # 앞 고르기가 바뀌는지
    br = H.base_result()
    pick0 = H.select_front([row_metrics(r) for r in clean_E], br, 0.9)
    pick1 = H.select_front([row_metrics(r) for r in rows_b], br, 0.9)
    return {"HY 변형 수": len(out), "2026 전체 수익 최대 차": diff, "2025 수익 최대 차": d25, "차이 큰 5개": rowsE,
            "앞 고르기 keep90": [pick0["name"] if pick0 else None, pick1["name"] if pick1 else None]}


def pdf_earlier_test(clean_H: list) -> dict:
    """PDF D는 21:45 KST 공개 → 같은 날 미국 시가(22:30/23:30 KST)에 쓸 수 있다. 하루 앞당겨 쓰면?"""
    import fam_H_events_funds as HH
    fs = HH.load_fund_series()
    b26 = H.base("2026")
    days = b26.days
    out = []
    ref = {r["name"]: r for r in clean_H}
    for kind, thr in HH.FUND_SIGNALS:
        t26 = HH.fund_signal(fs["TIME26"], kind, thr, 5)
        k26 = HH.fund_signal(fs["KoAct26"], kind, thr, 5)
        both26 = (t26 & k26.reindex(t26.index).fillna(False))
        for fund, s26 in (("TIME", t26), ("KoAct", k26), ("둘다", both26)):
            # 날짜 D의 값을 D 시가부터: index를 D 바로 앞 거래일(없으면 하루 전)로 옮긴다
            s = s26.copy()
            s.index = [(max([d for d in days if d < x], default=(pd.Timestamp(x) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
                       for x in s.index]
            s = s[~s.index.duplicated(keep="last")]
            for lvl in HH.FUND_EXPO:
                for n in HH.FUND_HOLD:
                    nm = f"펀드[{fund} {kind}{thr}] 노출{lvl} 유지{n}"
                    e = HH.fund_exposure(b26, s, lvl, n)
                    r0 = R.returns(ref[nm], "2026")
                    r1 = H.overlay(b26, e)
                    out.append({"name": nm, "뒤": [st(r0, *H.P26["뒤"])["수익"], st(r1, *H.P26["뒤"])["수익"]],
                                "앞": [st(r0, *H.P26["앞"])["수익"], st(r1, *H.P26["앞"])["수익"]],
                                "앞낙폭": [st(r0, *H.P26["앞"])["낙폭"], st(r1, *H.P26["앞"])["낙폭"]],
                                "AI": [st(r0, *AI)["수익"], st(r1, *AI)["수익"]]})
    keep = {nm: next(x for x in out if x["name"] == nm) for nm in FOCUS["H"] if nm.startswith("펀드")}
    return {"고른 변형(전→앞당김)": rnd(keep),
            "뒤 수익 차 최대": rnd(max(abs(x["뒤"][1] - x["뒤"][0]) for x in out)),
            "앞당겨 뒤 수익이 기본(+4.6%) 넘는 변형": sum(x["뒤"][1] > 0.0459 for x in out),
            "원래 넘는 변형": sum(x["뒤"][0] > 0.0459 for x in out)}


# ============================================================================ b 모순
def soxx_cross_dates():
    sx = H.px("SOXX")
    sma = sx.rolling(50).mean()
    below = (sx < sma)
    x = pd.DataFrame({"SOXX": sx, "SMA50": sma, "아래": below})
    return rnd({d: {"SOXX": float(v.SOXX), "SMA50": float(v.SMA50), "아래": bool(v.아래)}
                for d, v in x.loc["2026-07-08":"2026-07-20"].iterrows()}, 2)


def soxx_timer_sensitivity():
    import fam_G_hold_reentry as G
    out = []
    for n in (5, 10, 15, 20, 25, 30, 40):
        row = {"name": f"SOXX<SMA50 하향 → {n}일", "e": {}, "h": None}
        eps = {}
        for bn in R.BNS:
            e_close, ep = G.simulate(G.signals(bn), "soxx_sma50", None, ("time", n))
            row["e"][bn] = H.at_open(e_close, H.base(bn), fill=1.0)
            eps[bn] = [(H.base(bn).days[x["xi"]], H.base(bn).days[x["ri"]] if x["ri"] is not None else None)
                       for x in ep if x["type"] == "out"]
        m = row_metrics(row)
        out.append({"규칙": row["name"], "2026전체": [m["2026"]["전체"]["수익"], m["2026"]["전체"]["낙폭"]],
                    "앞": [m["2026"]["앞"]["수익"], m["2026"]["앞"]["낙폭"]], "뒤": [m["2026"]["뒤"]["수익"], m["2026"]["뒤"]["낙폭"]],
                    "AI쇼크": m["2026"]["AI쇼크"]["수익"], "2025": m["hold"]["2025"]["수익"], "2024": m["hold"]["2024"]["수익"],
                    "2026 이탈(신호일→복귀신호일)": [p for p in eps["2026"]]})
    return rnd(out)


def vol_consistency():
    rch = pd.concat([(H.base(bn).close_s / H.base(bn).close_s.shift(1) - 1).dropna() for bn in ("hold", "2026")])
    rch = rch[~rch.index.duplicated(keep="last")].sort_index()
    v20 = rch.rolling(20).std() * np.sqrt(252)
    up = np.sqrt((rch.clip(lower=0) ** 2).rolling(20).mean() * 252 * 2)
    dn = np.sqrt((rch.clip(upper=0) ** 2).rolling(20).mean() * 252 * 2)
    x = pd.DataFrame({"v20": v20, "상방": up, "하방": dn})
    x26 = x[x.index >= "2026-01-02"]
    by_m = x26.groupby(x26.index.str[:7]).mean()
    on = (x26["v20"] > 0.60)
    ret20 = (H.base("2026").close_s / H.base("2026").close_s.shift(20) - 1).reindex(x26.index)
    return {"월평균(20일 실현·상방·하방 반편차×√2)": rnd(by_m.to_dict(orient="index"), 3),
            "2026 >60% 비율": round(float(on.mean()), 3),
            "앞 >60% 비율": round(float(on[on.index <= "2026-05-29"].mean()), 3),
            "AI쇼크 >60% 비율": round(float(on[(on.index >= AI[0]) & (on.index <= AI[1])].mean()), 3),
            "2026 20일 실현 중앙값": round(float(x26["v20"].median()), 3),
            "2025 20일 실현 중앙값": round(float(x[(x.index >= "2025-01-01") & (x.index <= "2025-12-31")]["v20"].median()), 3),
            "2025하반기 중앙값": round(float(x[(x.index >= "2025-07-01") & (x.index <= "2025-12-31")]["v20"].median()), 3),
            "corr(20일 변동성, 같은 20일 수익) 2026": round(float(pd.concat([x26["v20"], ret20], axis=1).corr().iloc[0, 1]), 3)}


# ============================================================================ c AI쇼크·같은 노출 고정
def matched_fixed(row, bn, lo, hi):
    b = H.base(bn)
    e = row["e"][bn].reindex(b.ret.index).astype(float)
    m = (e.index >= lo) & (e.index <= hi)
    c = float(e[m].mean())
    return c, st(c * b.ret, lo, hi)


def worst_roll(r, lo, hi, n=27):
    x = r[(r.index >= lo) & (r.index <= hi)]
    v = (1 + x).cumprod().to_numpy()
    v = np.concatenate([[1.0], v])
    if len(v) <= n:
        return float(v[-1] - 1)
    return float(np.min(v[n:] / v[:-n] - 1))


def ai_window_all(allrows):
    b = H.base("2026")
    base_ai, base_plus, base_back = st(b.ret, *AI)["수익"], st(b.ret, *AI_PLUS)["수익"], st(b.ret, *H.P26["뒤"])["수익"]
    out = {}
    for fam in FAMS:
        n_ai = n_ai_plus = n_ai_back = n_ai_fixed = n_ai_plus_fixed = 0
        d_ai, d_plus = [], []
        for row in allrows[fam]:
            r = R.returns(row, "2026")
            a, p, bk = st(r, *AI)["수익"], st(r, *AI_PLUS)["수익"], st(r, *H.P26["뒤"])["수익"]
            if a > base_ai + 1e-4:
                n_ai += 1
                n_ai_plus += p > base_plus + 1e-4
                n_ai_back += bk > base_back + 1e-4
                if row.get("h") is None:
                    _, fa = matched_fixed(row, "2026", *AI)
                    _, fp = matched_fixed(row, "2026", *AI_PLUS)
                    n_ai_fixed += a > fa["수익"] + 1e-4
                    n_ai_plus_fixed += p > fp["수익"] + 1e-4
            d_ai.append(a - base_ai)
            d_plus.append(p - base_plus)
        out[fam] = {"변형": len(allrows[fam]), "AI쇼크 개선": n_ai, "그중 6/22~9/30도 개선": int(n_ai_plus),
                    "그중 뒤 전체도 개선": int(n_ai_back), "그중 AI쇼크에서 같은 노출 고정보다 나음": int(n_ai_fixed),
                    "그중 6/22~9/30에서 같은 노출 고정보다 나음": int(n_ai_plus_fixed),
                    "corr(AI쇼크 개선, 6/22~9/30 개선)": round(float(np.corrcoef(d_ai, d_plus)[0, 1]), 3)}
    return {"기본": rnd({"AI쇼크": base_ai, "6/22~9/30": base_plus, "뒤": base_back}), "갈래": out}


def focus_table(allrows, rate=0.0):
    """후보 변형마다: 앞·뒤·AI쇼크·6/22~9/30·2025, 같은 평균 노출 고정 대비, 이동 27일 최악."""
    b, bh = H.base("2026"), H.base("hold")
    out = []
    by = {fam: {r["name"]: r for r in allrows[fam]} for fam in FAMS}
    rb = R.returns(R.base_row(), "2026", rate)
    rbh = R.returns(R.base_row(), "hold", rate)
    for fam, names in FOCUS.items():
        for nm in names:
            row = by[fam][nm]
            r, rh = R.returns(row, "2026", rate), R.returns(row, "hold", rate)
            d = {"갈래": fam, "변형": nm}
            for lab, (bn, lo, hi) in {"2026전체": ("2026",) + H.P26["전체"], "앞": ("2026",) + H.P26["앞"],
                                      "뒤": ("2026",) + H.P26["뒤"], "AI쇼크": ("2026",) + AI,
                                      "6/22~9/30": ("2026",) + AI_PLUS, "2025": ("hold",) + H.PHOLD["2025"],
                                      "관세": ("hold",) + H.PHOLD["관세"]}.items():
                rr, bb = (r, rb) if bn == "2026" else (rh, rbh)
                v, bs = st(rr, lo, hi), st(bb, lo, hi)
                cell = {"변형": [v["수익"], v["낙폭"]], "기본": [bs["수익"], bs["낙폭"]]}
                if row.get("h") is None:
                    c, f = matched_fixed(row, bn, lo, hi)
                    if rate:
                        bobj = H.base(bn)
                        inv = pd.Series([sum(w.values()) for w in bobj.weights], index=bobj.days).reindex(bobj.ret.index)
                        f = st(c * bobj.ret + (1 - c * inv) * rate / 252, lo, hi)
                    cell["같은노출고정"] = [c, f["수익"], f["낙폭"]]
                d[lab] = cell
            d["뒤 이동27일 최악"] = {"변형": worst_roll(r, *H.P26["뒤"]), "기본": worst_roll(rb, *H.P26["뒤"])}
            d["2025 이동27일 최악"] = {"변형": worst_roll(rh, *H.PHOLD["2025"]), "기본": worst_roll(rbh, *H.PHOLD["2025"])}
            if row.get("h") is None:
                c = float(row["e"]["2026"].reindex(b.ret.index)[(b.ret.index >= H.P26["뒤"][0])].mean())
                d["뒤 이동27일 최악"]["같은노출고정"] = worst_roll(c * b.ret, *H.P26["뒤"])
            out.append(d)
    return rnd(out)


# ============================================================================ d 앞 고르기
def dd_episode(r, lo, hi):
    x = r[(r.index >= lo) & (r.index <= hi)]
    v = pd.Series(np.concatenate([[1.0], (1 + x).cumprod().to_numpy()]), index=["start"] + list(x.index))
    dd = v / v.cummax() - 1
    t = dd.idxmin()
    pk = v.loc[:t].idxmax()
    return float(dd.min()), pk, t


def episode_label(t):
    if t == "start":
        return "없음"
    for lab, lo, hi in (("2/3~2/5", "2026-02-02", "2026-02-09"), ("2/25~3/6", "2026-02-24", "2026-03-10"),
                        ("3/19~3/30", "2026-03-18", "2026-04-01"), ("5/13~5/19", "2026-05-12", "2026-05-21")):
        if lo <= t <= hi:
            return lab
    return "기타"


def front_episode_census(allrows):
    out = {}
    b = H.base("2026")
    dd, pk, t = dd_episode(b.ret, *H.P26["앞"])
    out["기본"] = {"앞 낙폭": round(dd, 4), "고점(시가 경로 index)": pk, "저점": t, "하락": episode_label(t)}
    bw = 1 + st(b.ret, *H.P26["앞"])["수익"]
    for fam in FAMS:
        cnt, cnt90 = {}, {}
        best = []
        for row in allrows[fam]:
            r = R.returns(row, "2026")
            d, _, t = dd_episode(r, *H.P26["앞"])
            lab = episode_label(t)
            cnt[lab] = cnt.get(lab, 0) + 1
            if 1 + st(r, *H.P26["앞"])["수익"] >= 0.9 * bw:
                cnt90[lab] = cnt90.get(lab, 0) + 1
                best.append(d)
        best = sorted(best, reverse=True)
        out[fam] = {"전 변형 앞 최대낙폭이 나온 하락": cnt, "keep90 통과 변형": cnt90,
                    "keep90 최선 앞 낙폭": round(best[0], 4) if best else None,
                    "최선과 0.5%p 안 변형 수": int(sum(x >= best[0] - 0.005 for x in best)) if best else 0}
    return out


def ulcer(r, lo, hi):
    x = r[(r.index >= lo) & (r.index <= hi)]
    v = np.concatenate([[1.0], (1 + x).cumprod().to_numpy()])
    dd = v / np.maximum.accumulate(v) - 1
    return float(np.sqrt(np.mean(dd ** 2)))


def zero_feb(r):
    r = r.copy()
    m = (r.index >= FEB_EP[0]) & (r.index <= FEB_EP[1])
    r[m] = 0.0
    return r


def alt_selection(allrows, rate=0.0):
    """앞 구간 고르기를 4가지로 바꿔 본다. 고른 변형의 뒤·AI쇼크·2025와 같은 노출 고정 대비."""
    br0 = row_metrics(R.base_row(), rate)
    brF = row_metrics(R.base_row(), rate, zero_feb)
    b = H.base("2026")
    rb = R.returns(R.base_row(), "2026", rate)
    base_ul = ulcer(rb, *H.P26["앞"])
    out = {}
    for fam in FAMS:
        rows = allrows[fam]
        M0 = [row_metrics(r, rate) for r in rows]
        MF = [row_metrics(r, rate, zero_feb) for r in rows]
        res = {}
        p = H.select_front(M0, br0, 0.9)
        res["기본 규칙(keep90)"] = p["name"] if p else None
        p = H.select_front(MF, brF, 0.9)
        res["2/3~2/5 제외 keep90"] = p["name"] if p else None
        bw = 1 + br0["2026"]["앞"]["수익"]
        ok = [(i, m) for i, m in enumerate(M0) if 1 + m["2026"]["앞"]["수익"] >= 0.9 * bw]
        if ok:
            ul = [(ulcer(R.returns(rows[i], "2026", rate), *H.P26["앞"]), i) for i, _ in ok]
            u, i = min(ul)
            res["앞 울서지수 최소 keep90"] = rows[i]["name"] if u < base_ul * 0.95 else None
            cal = max(ok, key=lambda t: t[1]["2026"]["앞"]["수익"] / max(1e-9, -t[1]["2026"]["앞"]["낙폭"]))
            res["앞 칼마 최대 keep90"] = cal[1]["name"]
        else:
            res["앞 울서지수 최소 keep90"] = res["앞 칼마 최대 keep90"] = None
        # 타이밍 가치: 앞 구간에서 같은 평균 노출 고정보다 수익·낙폭 둘 다 나은 변형 중 앞 낙폭 최소
        tv = []
        for i, m in enumerate(M0):
            if rows[i].get("h") is not None:
                continue
            c, f = matched_fixed(rows[i], "2026", *H.P26["앞"])
            if m["2026"]["앞"]["수익"] > f["수익"] and m["2026"]["앞"]["낙폭"] > f["낙폭"] + 0.01 and c < 0.995:
                tv.append((m["2026"]["앞"]["낙폭"], m["2026"]["앞"]["수익"], i))
        res["앞에서 같은노출고정을 이긴 변형 수"] = len(tv)
        res["그중 앞 낙폭 최소"] = rows[max(tv)[2]]["name"] if tv else None
        detail = {}
        by = {r["name"]: (r, m) for r, m in zip(rows, M0)}
        for k, nm in res.items():
            if not isinstance(nm, str) or nm in detail:
                continue
            r, m = by[nm]
            dct = {"앞": [m["2026"]["앞"]["수익"], m["2026"]["앞"]["낙폭"]], "뒤": [m["2026"]["뒤"]["수익"], m["2026"]["뒤"]["낙폭"]],
                   "AI쇼크": m["2026"]["AI쇼크"]["수익"], "2026전체": [m["2026"]["전체"]["수익"], m["2026"]["전체"]["낙폭"]],
                   "2025": [m["hold"]["2025"]["수익"], m["hold"]["2025"]["낙폭"]]}
            if r.get("h") is None:
                c, f = matched_fixed(r, "2026", *H.P26["뒤"])
                dct["뒤 같은노출고정"] = [c, f["수익"], f["낙폭"]]
            dct["뒤에서 기본 이김(수익·낙폭 둘 다)"] = bool(m["2026"]["뒤"]["수익"] > br0["2026"]["뒤"]["수익"]
                                               and m["2026"]["뒤"]["낙폭"] > br0["2026"]["뒤"]["낙폭"])
            detail[nm] = dct
        out[fam] = {"고른 변형": res, "상세": rnd(detail)}
    return {"기본": rnd({"앞": [br0["2026"]["앞"]["수익"], br0["2026"]["앞"]["낙폭"]],
                        "앞(2/3~2/5 제외)": [brF["2026"]["앞"]["수익"], brF["2026"]["앞"]["낙폭"]],
                        "뒤": [br0["2026"]["뒤"]["수익"], br0["2026"]["뒤"]["낙폭"]], "앞 울서": base_ul}), "갈래": out}


# ============================================================================ e 보유 기간
def hold_diag():
    out = {}
    for bn, lo, hi in (("hold", "2024-01-01", "2024-12-31"), ("hold", "2025-01-01", "2025-12-31"),
                       ("2026", "2026-01-02", "2026-10-09")):
        b = H.base(bn)
        inv = pd.Series([sum(w.values()) for w in b.weights], index=b.days)
        semi = pd.Series([sum(v for k, v in w.items() if b.sector.get(k) in
                              {"AI 반도체", "메모리", "반도체 장비", "반도체 장비(적자)", "반도체(적자)", "반도체 소재", "미분류"})
                          for w in b.weights], index=b.days)
        unk = pd.Series([sum(v for k, v in w.items() if b.sector.get(k) == "미분류") for w in b.weights], index=b.days)
        m = (inv.index >= lo) & (inv.index <= hi)
        cr = b.close_s.pct_change()
        x = cr[(cr.index >= lo) & (cr.index <= hi)]
        sx = H.px("SOXX", "Adj Close").pct_change().reindex(x.index)
        beta = float(x.cov(sx) / sx.var())
        out[f"{bn} {lo[:4]}"] = {"평균 투자비중": float(inv[m].mean()), "상한 현금 있는 날 비율": float((inv[m] < 0.999).mean()),
                                 "반도체+미분류 평균": float(semi[m].mean()), "미분류 평균": float(unk[m].mean()),
                                 "연율 변동성": float(x.std() * np.sqrt(252)), "SOXX β": beta,
                                 "교체일 수": int(sum(1 for d in b.rebalance_days() if lo <= d <= hi))}
    return rnd(out, 3)


# ============================================================================ f 이자
def interest_reselect(allrows):
    br0, br4 = row_metrics(R.base_row()), row_metrics(R.base_row(), RATE)
    out = {"기본": {"이자0": {k: [br0["2026"][k]["수익"], br0["2026"][k]["낙폭"]] for k in ("전체", "앞", "뒤", "AI쇼크")},
                    "이자4%": {k: [br4["2026"][k]["수익"], br4["2026"][k]["낙폭"]] for k in ("전체", "앞", "뒤", "AI쇼크")},
                    "2025": [br0["hold"]["2025"]["수익"], br4["hold"]["2025"]["수익"]]}}
    b, bh = H.base("2026"), H.base("hold")
    fx = {}
    for c in (0.9, 0.8, 0.7):
        for bn, bb in (("2026", b), ("hold", bh)):
            inv = pd.Series([sum(w.values()) for w in bb.weights], index=bb.days).reindex(bb.ret.index)
            r0, r4 = c * bb.ret, c * bb.ret + (1 - c * inv) * RATE / 252
            for k in (("전체", "뒤", "AI쇼크") if bn == "2026" else ("2025",)):
                fx[f"고정{c} {k}"] = [st(r0, *bb.periods[k])["수익"], st(r4, *bb.periods[k])["수익"]]
    out["고정노출(이자0→4%)"] = fx
    fams = {}
    for fam in FAMS:
        M0 = [row_metrics(r) for r in allrows[fam]]
        M4 = [row_metrics(r, RATE) for r in allrows[fam]]
        d = {}
        for kp in (0.9, 0.8, 0.7):
            p0, p4 = H.select_front(M0, br0, kp), H.select_front(M4, br4, kp)
            d[f"keep{int(kp * 100)}"] = {"이자0": p0["name"] if p0 else None, "이자4%": p4["name"] if p4 else None}
            if p4:
                d[f"keep{int(kp * 100)}"]["이자4% 고른 변형 뒤/AI/2025"] = [p4["2026"]["뒤"]["수익"], p4["2026"]["뒤"]["낙폭"],
                                                                     p4["2026"]["AI쇼크"]["수익"], p4["hold"]["2025"]["수익"]]
        bw0, bw4 = 1 + br0["2026"]["앞"]["수익"], 1 + br4["2026"]["앞"]["수익"]
        d["keep90 통과 수(이자0→4%)"] = [sum(1 + m["2026"]["앞"]["수익"] >= 0.9 * bw0 for m in M0),
                                    sum(1 + m["2026"]["앞"]["수익"] >= 0.9 * bw4 for m in M4)]
        d["뒤 수익이 기본보다 높은 변형(이자0→4%)"] = [sum(m["2026"]["뒤"]["수익"] > br0["2026"]["뒤"]["수익"] for m in M0),
                                           sum(m["2026"]["뒤"]["수익"] > br4["2026"]["뒤"]["수익"] for m in M4)]
        fams[fam] = d
    out["갈래"] = fams
    return rnd(out)


# ============================================================================ g 원화
def krw_view(allrows):
    fx = H.px("KRW=X", "Close")
    out = {"USDKRW": {}}
    for lab, (lo, hi) in {"2026 전체": H.P26["전체"], "AI쇼크": AI, "뒤": H.P26["뒤"], "2025": H.PHOLD["2025"],
                          "관세": H.PHOLD["관세"]}.items():
        f = fx.reindex(sorted(set(fx.index) | {lo, hi})).ffill()
        nxt = H.base("2026" if lo >= "2026" else "hold").days
        hi2 = next((d for d in nxt if d > hi), hi)
        out["USDKRW"][lab] = [round(float(f[lo]), 1), round(float(f[hi2]), 1), round(float(f[hi2] / f[lo] - 1), 4)]

    def krw_ret(row, bn, mode):
        b = H.base(bn)
        f = fx.reindex(sorted(set(fx.index) | set(b.days))).ffill().reindex(b.days)
        fr = (f.shift(-1) / f).reindex(b.ret.index)
        r = R.returns(row, bn)
        if mode == "USD현금":
            return (1 + r) * fr - 1
        a = R.invested(row, bn)
        return (a + r) * fr - a          # 주식·헤지만 달러, 현금(상한 현금 포함)은 원화

    focus = []
    by = {fam: {r["name"]: r for r in allrows[fam]} for fam in FAMS}
    base = R.base_row()
    for fam, names in FOCUS.items():
        for nm in names:
            row = by[fam][nm]
            d = {"갈래": fam, "변형": nm}
            for mode in ("USD", "USD현금", "원화현금"):
                cells = {}
                for lab, (bn, lo, hi) in {"2026전체": ("2026",) + H.P26["전체"], "뒤": ("2026",) + H.P26["뒤"],
                                          "AI쇼크": ("2026",) + AI, "2025": ("hold",) + H.PHOLD["2025"],
                                          "관세": ("hold",) + H.PHOLD["관세"]}.items():
                    if mode == "USD":
                        rv, rb = R.returns(row, bn), R.returns(base, bn)
                    else:
                        rv, rb = krw_ret(row, bn, mode), krw_ret(base, bn, mode)
                    v, s = st(rv, lo, hi), st(rb, lo, hi)
                    cells[lab] = {"변형": [v["수익"], v["낙폭"]], "기본": [s["수익"], s["낙폭"]],
                                  "수익차": v["수익"] - s["수익"], "낙폭차": v["낙폭"] - s["낙폭"]}
                d[mode] = cells
            focus.append(d)
    out["후보"] = rnd(focus)
    # 앞 고르기가 원화(원화 현금)에서 바뀌는지
    picks = {}
    for fam in FAMS:
        rows = allrows[fam]

        def mk(row):
            m = {"name": row["name"]}
            for bn in R.BNS:
                m[bn] = win_metrics(bn, krw_ret(row, bn, "원화현금"))
            return m
        M = [mk(r) for r in rows]
        p = H.select_front(M, mk(base), 0.9)
        p0 = H.select_front([row_metrics(r) for r in rows], row_metrics(base), 0.9)
        picks[fam] = {"USD": p0["name"] if p0 else None, "원화현금": p["name"] if p else None,
                      "원화현금 고른 변형 뒤수익/낙폭": rnd([p["2026"]["뒤"]["수익"], p["2026"]["뒤"]["낙폭"]]) if p else None}
    bk = mk(base)
    picks["기본(원화현금) 뒤"] = rnd([bk["2026"]["뒤"]["수익"], bk["2026"]["뒤"]["낙폭"]])
    out["앞 고르기(원화현금)"] = picks
    return out


# ============================================================================ h 효율선·종가 직후 체결
def fixed_frontier():
    out = {}
    cs = np.round(np.arange(0.0, 1.0001, 0.005), 3)
    for bn, keys in (("2026", ("전체", "앞", "뒤")), ("hold", ("2025",))):
        b = H.base(bn)
        for k in keys:
            out[(bn, k)] = [(c, st(c * b.ret, *b.periods[k])) for c in cs]
    return out


def frontier_census(allrows):
    F = fixed_frontier()
    res = {}
    for fam in FAMS:
        d = {}
        for (bn, k), grid in F.items():
            beat = 0
            for row in allrows[fam]:
                r = R.returns(row, bn)
                v = st(r, *H.base(bn).periods[k])
                ok = [g for c, g in grid if g["낙폭"] >= v["낙폭"] - 1e-9]
                best = max(g["수익"] for g in ok) if ok else -1
                beat += v["수익"] > best + 1e-4
            d[f"{bn} {k}"] = beat
        d["변형"] = len(allrows[fam])
        res[fam] = d
    return res


def zero_delay(allrows):
    """노출 e[k](k−1 종가에 정함)를 k−1 종가에 바로 체결했다고 본 상한(프리장·주간거래 체결의 최대 이득).
    기본·변형 모두 종가→종가 수익으로 재고, 개선폭(변형−기본)을 시가 체결 때와 비교한다."""
    out = []
    by = {fam: {r["name"]: r for r in allrows[fam]} for fam in FAMS}
    for fam, names in FOCUS.items():
        if fam == "B":
            continue
        for nm in names:
            row = by[fam][nm]
            d = {"갈래": fam, "변형": nm}
            for lab, (bn, lo, hi) in {"2026전체": ("2026",) + H.P26["전체"], "앞": ("2026",) + H.P26["앞"],
                                      "뒤": ("2026",) + H.P26["뒤"], "AI쇼크": ("2026",) + AI,
                                      "2025": ("hold",) + H.PHOLD["2025"]}.items():
                b = H.base(bn)
                rc = (b.close_s / b.close_s.shift(1) - 1).dropna()
                e = row["e"][bn].reindex(rc.index).astype(float)
                rv = e * rc
                for sym, h in ((row.get("h") or {}).get(bn) or {}).items():
                    hc = H.px(sym, "Close").reindex(b.days).ffill()
                    rv = rv + h.reindex(rc.index).fillna(0.0) * (hc / hc.shift(1) - 1).reindex(rc.index).fillna(0.0)
                vo, bo = st(R.returns(row, bn), lo, hi), st(b.ret, lo, hi)
                vz, bz = st(rv, lo, hi), st(rc, lo, hi)
                d[lab] = {"시가체결 개선(수익,낙폭)": [vo["수익"] - bo["수익"], vo["낙폭"] - bo["낙폭"]],
                          "종가즉시 개선(수익,낙폭)": [vz["수익"] - bz["수익"], vz["낙폭"] - bz["낙폭"]]}
            out.append(d)
    return rnd(out)


def overnight_gap_share():
    """기본 경로 수익 중 밤사이(종가→다음 시가) 몫: 종가 신호를 시가에 체결할 때 놓치는 구간의 크기."""
    out = {}
    for bn, lo, hi in (("2026", "2026-01-02", "2026-10-09"), ("2026", *AI), ("hold", "2025-01-01", "2025-12-31")):
        b = H.base(bn)
        o = pd.Series(b.open, index=b.days)
        c = b.close_s
        gap = (o / c.shift(1) - 1).dropna()
        intr = (c / o - 1)
        m = (gap.index >= lo) & (gap.index <= hi)
        g, i = gap[m], intr.reindex(gap.index)[m]
        out[f"{bn} {lo}~{hi}"] = {"밤사이 누적": float((1 + g).prod() - 1), "장중 누적": float((1 + i).prod() - 1),
                                  "밤사이 |평균|": float(g.abs().mean()), "장중 |평균|": float(i.abs().mean()),
                                  "밤사이 −3% 이하 날": int((g <= -0.03).sum())}
    return rnd(out)


# ============================================================================ 실행
def main():
    allrows, repro = {}, {}
    for fam in FAMS:
        allrows[fam] = R.GEN[fam]()
        repro[fam] = R.check_reproduction(allrows[fam], fam)
        print(fam, len(allrows[fam]), "불일치", repro[fam]["n_mismatch"])
    res = {"재현": repro}

    print("[a] 시점")
    res["a_at_open"] = test_at_open()
    res["a_미래교란_2026"] = lookahead_test(allrows, "2026-07-10", "2026")
    res["a_미래교란_2026_2월"] = lookahead_test(allrows, "2026-02-03", "2026")
    res["a_미래교란_hold"] = lookahead_test(allrows, "2025-03-20", "hold")
    res["a_StockEngine"] = lookahead_engine("2026-07-10")
    res["a_FRED_영업일지연"] = fred_lag_test(allrows["E"])
    res["a_PDF_하루앞당김"] = pdf_earlier_test(allrows["H"])
    print(json.dumps(rnd({k: res[k] for k in res if k.startswith("a_")}), ensure_ascii=False, default=str)[:4000])

    print("[b] 모순")
    res["b_SOXX_교차"] = soxx_cross_dates()
    res["b_SOXX_타이머"] = soxx_timer_sensitivity()
    res["b_변동성"] = vol_consistency()

    print("[c] AI쇼크")
    res["c_AI쇼크_전변형"] = ai_window_all(allrows)
    res["c_후보표"] = focus_table(allrows)

    print("[d] 앞 고르기")
    res["d_앞낙폭_출처"] = front_episode_census(allrows)
    res["d_대체고르기"] = alt_selection(allrows)

    print("[e] 보유 기간")
    res["e_보유진단"] = hold_diag()

    print("[f] 이자")
    res["f_이자4%"] = interest_reselect(allrows)
    res["f_후보표_이자4%"] = focus_table(allrows, RATE)

    print("[g] 원화")
    res["g_원화"] = krw_view(allrows)

    print("[h] 효율선·종가즉시")
    res["h_고정노출효율선_이긴변형수"] = frontier_census(allrows)
    res["h_종가즉시체결"] = zero_delay(allrows)
    res["h_밤사이몫"] = overnight_gap_share()

    OUT.write_text(json.dumps(rnd(res), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("저장", OUT)
    return res


if __name__ == "__main__":
    main()
