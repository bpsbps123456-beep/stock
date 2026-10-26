"""leadlag 갈래(설명용 + 사전 지정 시험 1개). 헤지펀드 쪽 주간 변화 ↔ 개인·옵션 심리·QQQ의 주간 시차 상관, 그리고
'둘이 엇갈린 주에 누가 맞았나'.

시점: 모든 원값은 harness._place(패널과 같은 규칙: when 다음 거래일부터, 일별 3·주간 15·월 2회 25거래일까지)로 진입일에 놓는다.
주간 표본 = 각 주(W-FRI)의 마지막 진입일 d_w. 그 값은 d_w 시가 전에 공개된 것만 쓴다. 마지막 미완성 주는 버린다.
QQQ 주간 수익 r_w = 시가(d_{w+1}) ÷ 시가(d_w) − 1. 앞 수익 F_k(w) = 시가(d_{w+k}) ÷ 시가(d_w) − 1, k = 1·2·4주.

(1) 시차 상관: corr(X_t, Y_{t+k}), k = −8..+8주, 2024-01 ~ 2026-10 주간. 각 계열을 전 구간 순위(0~1)로 바꾼 뒤 피어슨(이상치에 강함).
    95% 구간 = 이동 블록 부트스트랩(블록 8주, 1,000번). 모든 시차·쌍이 같은 시점 재표본을 쓰므로 '앞섬 비대칭'
    A = 평균 ρ²(앞쪽 8칸) − 평균 ρ²(뒤쪽 8칸)의 구간도 같은 재표본으로 낸다. A > 0 = 헤지펀드 쪽이 앞섬.
    QQQ와 짝지을 때는 X_t가 d_t 시가 전에 알려지고 r_t는 그 뒤에 생기므로 k ≥ 0이 '앞으로', k ≤ −1이 '지난 수익'이다.
    보조(매매 불가, 설명만): 측정 시점 정렬(CFTC 화요일 기준일, 그 밖 측정일이 든 주; QQQ는 그 주 종가 대비 종가).
(2) 엇갈린 주(사전 지정, 코드 작성 시 결과 보기 전에 정함):
    HF매수·개인매도 = CFTC nq 레버리지 순포지션/OI 1주 변화 > 0 그리고 ProShares TQQQ−SQQQ 순유입 1주 합 < 0.
    HF매도·개인매수 = 반대. 1주 순유입 = 직전 표본 d_{w−1} ≤ 측정일 < d_w 인 날의 합(두 표본 사이 새로 공개된 흐름, 겹침 없음).
    CFTC 보고서가 직전 주 표본과 같은(새 보고서 없는) 주는 같은 사건을 두 번 세지 않도록 뺀다.
    비교: 앞 1·2·4주 QQQ 수익의 신호 주 평균 − 그 기간 전체 주 평균. p = 기간 안 유효 주에서 신호열을 모든 0 아닌 거리만큼
    원형 이동한 귀무(작은 이동은 지속성 때문에 보수 쪽). 판정: front 부호 방향으로 back 같은 부호 + back 한쪽 p < 0.05.
결과: results/leadlag.json, results/leadlag_cells.parquet(모든 상관 칸), results/leadlag.md(요약, 손으로 씀)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import harness as H

LAGS = np.arange(-8, 9)
NB, BLOCK, SEED = 1000, 8, 20261014
START = "2024-01-01"
MIN_PAIRS = 30

HF = {  # 헤지펀드·큰손 쪽 주간 변화
    "nq_lev_1w": "CFTC 나스닥 레버리지 순/OI 1주 변화", "nq_lev_4w": "CFTC 나스닥 레버리지 순/OI 4주 변화",
    "nq_am_1w": "CFTC 나스닥 자산운용 순/OI 1주 변화", "nq_am_4w": "CFTC 나스닥 자산운용 순/OI 4주 변화",
    "es_lev_1w": "CFTC S&P 레버리지 순/OI 1주 변화", "es_lev_4w": "CFTC S&P 레버리지 순/OI 4주 변화",
    "vx_lev_1w": "CFTC VIX 레버리지 순/OI 1주 변화", "vx_lev_4w": "CFTC VIX 레버리지 순/OI 4주 변화",
    "dix5_chg": "DIX 5일 평균 주간 변화", "gvip_rel20": "GVIP/QQQ 20거래일 변화", "si_ai_chg": "AI 공매도 잔고 변화(월 2회)",
}
RETAIL = {  # 개인·옵션 심리
    "eq_pcr5": "CBOE 주식 풋/콜 5일", "ai_pcr5": "OCC AI 고객 풋/콜 5일", "aaii_spread": "AAII 강세−약세",
    "flow20p": "ProShares TQQQ−SQQQ 순유입 20일 ÷ 순자산", "tqqq_share5": "TQQQ 거래대금 몫 5일",
}
MKT = {"qqq_ret": "QQQ 주간 수익"}


# ---------- 일별 원값(패널 시점) ----------
def _ser(v: pd.Series, when=None) -> pd.Series:
    return pd.Series(v.index if when is None else np.asarray(when), index=v.index)


def daily(dd: pd.Index) -> pd.DataFrame:
    import build as osb  # options_sentiment
    import fetch as osf
    out = {}
    cf = pd.read_parquet(H.CACHE / "cftc_tff.parquet")
    cf = cf.iloc[np.argsort(pd.to_datetime(cf["asof"]).to_numpy(), kind="stable")]
    wc = _ser(cf)
    for g in ("nq_lev", "nq_am", "es_lev", "vx_lev"):
        x = cf[f"{g}_net"] / cf[f"{g.split('_')[0]}_oi"]
        out[f"{g}_1w"] = H._place(x.diff(1), wc, dd, H.WEEKLY_AGE)
        out[f"{g}_4w"] = H._place(x.diff(4), wc, dd, H.WEEKLY_AGE)
    ordn = pd.Series([pd.Timestamp(a).toordinal() for a in cf["asof"]], index=cf.index, dtype=float)
    out["cftc_asof"] = H._place(ordn, wc, dd, H.WEEKLY_AGE)
    # DIX(측정일 저녁 공개 → 다음 거래일)
    sq = pd.read_parquet(H.CACHE / "squeeze.parquet").set_index("asof")
    d5 = sq["dix"].rolling(5).mean()
    out["dix5"] = H._place(d5, _ser(d5), dd, H.DAILY_AGE)
    # Yahoo 가격 지표(known 그날 종가 → 다음 거래일)
    hp = pd.read_parquet(H.CACHE / "hf_proxies.parquet")
    out["gvip_rel20"] = H._place(hp["gvip_rel20"], _ser(hp), dd, H.DAILY_AGE)
    out["tqqq_share5"] = H._place(hp["lev_qqq_bull5"], _ser(hp), dd, H.DAILY_AGE)
    # ProShares(측정일 asof 저녁 공개 → asof 다음 거래일)
    fl = hp.dropna(subset=["lev_qqq_flow20"]).set_index("asof")
    f20 = fl["lev_qqq_flow20"] / (fl["aum_tqqq"] + fl["aum_sqqq"])
    out["flow20p"] = H._place(f20, _ser(f20), dd, H.DAILY_AGE)
    # 공매도 잔고(월 2회, known 다음 거래일)
    si = pd.read_parquet(H.CACHE / "finra_si.parquet")
    out["si_ai_chg"] = H._place(si["si_ai_chg"], _ser(si), dd, H.BIMONTH_AGE)
    # 옵션(장 마감 뒤 → 다음 거래일)
    cb = osf.load_cboe()
    eq = cb["equity_pv"].rolling(5).sum() / cb["equity_cv"].rolling(5).sum()
    out["eq_pcr5"] = H._place(eq, _ser(eq), dd, H.DAILY_AGE)
    occ = osf.load_occ(osb.ai_us_symbols())
    cc = occ.pivot_table(index="date", columns="underlying", values="C_C", aggfunc="sum").reindex(cb.index)
    cp = occ.pivot_table(index="date", columns="underlying", values="C_P", aggfunc="sum").reindex(cb.index)
    have = cc.notna().any(axis=1) | cp.notna().any(axis=1)
    cc, cp = cc.fillna(0).where(have, axis=0), cp.fillna(0).where(have, axis=0)
    ai = cp.sum(axis=1, min_count=1).rolling(5).sum() / cc.sum(axis=1, min_count=1).rolling(5).sum()
    out["ai_pcr5"] = H._place(ai, _ser(ai), dd, H.DAILY_AGE)
    # AAII(목 아침 공개 → known 다음 거래일, 패널과 같음)
    a = pd.read_parquet(H.CACHE / "surveys.parquet")["aaii_spread"].dropna()
    out["aaii_spread"] = H._place(a, _ser(a), dd, H.WEEKLY_AGE)
    return pd.DataFrame(out, index=dd)


def weekly_days(dd: pd.Index) -> pd.Index:
    t = pd.to_datetime(dd)
    last = pd.Series(np.asarray(dd), index=t.to_period("W-FRI")).groupby(level=0).last()
    if pd.Timestamp(last.iloc[-1]).dayofweek != 4:  # 진행 중인 마지막 주
        last = last.iloc[:-1]
    return pd.Index(last.to_numpy())


def weekly() -> tuple[pd.DataFrame, dict]:
    px = H.prices()
    dd = px.index
    D = daily(dd)
    wd = weekly_days(dd)
    W = D.reindex(wd).copy()
    W["dix5_chg"] = W["dix5"].diff()
    o = px["QQQ_o"].reindex(wd)
    W["qqq_ret"] = (o.shift(-1) / o - 1) * 100
    for k in (1, 2, 4):
        W[f"fwd{k}"] = (o.shift(-k) / o - 1) * 100
    # 1주 순유입: d_{w-1} <= asof < d_w
    hp = pd.read_parquet(H.CACHE / "hf_proxies.parquet")
    f = hp.dropna(subset=["lev_qqq_flow"]).set_index("asof")["lev_qqq_flow"]
    fa = f.index.to_numpy()
    flow1 = [np.nan]
    for a, b in zip(wd[:-1], wd[1:]):
        m = (fa >= a) & (fa < b)
        flow1.append(f.to_numpy()[m].sum() if m.any() else np.nan)
    W["flow1w"] = flow1
    W["cftc_age"] = np.array([pd.Timestamp(d).toordinal() for d in W.index], dtype=float) - W["cftc_asof"].to_numpy()
    W["cftc_new"] = W["cftc_asof"].ne(W["cftc_asof"].shift()) & W["cftc_asof"].notna()
    meta = {"weeks_total": len(W), "first": wd[0], "last": wd[-1]}
    return W, meta


def weekly_measured() -> pd.DataFrame:
    """보조: 측정 시점 정렬(공개 지연 무시, 매매 불가). 각 원값을 측정일이 든 주(W-FRI)에 놓고 그 주 마지막 값."""
    import build as osb
    import fetch as osf
    wk = lambda idx: pd.to_datetime(pd.Index(idx)).to_period("W-FRI")
    cols = {}
    cf = pd.read_parquet(H.CACHE / "cftc_tff.parquet")
    cf = cf.iloc[np.argsort(pd.to_datetime(cf["asof"]).to_numpy(), kind="stable")]
    for g in ("nq_lev", "nq_am", "es_lev", "vx_lev"):
        x = pd.Series((cf[f"{g}_net"] / cf[f"{g.split('_')[0]}_oi"]).to_numpy(), index=wk(cf["asof"]))
        cols[f"{g}_1w"] = x.diff(1).groupby(level=0).last()
        cols[f"{g}_4w"] = x.diff(4).groupby(level=0).last()
    sq = pd.read_parquet(H.CACHE / "squeeze.parquet").set_index("asof")
    d5 = sq["dix"].rolling(5).mean()
    cols["dix5"] = pd.Series(d5.to_numpy(), index=wk(d5.index)).groupby(level=0).last()
    hp = pd.read_parquet(H.CACHE / "hf_proxies.parquet")
    for c, s in (("gvip_rel20", "gvip_rel20"), ("tqqq_share5", "lev_qqq_bull5")):
        cols[c] = pd.Series(hp[s].to_numpy(), index=wk(hp.index)).dropna().groupby(level=0).last()
    fl = hp.dropna(subset=["lev_qqq_flow20"])
    f20 = (fl["lev_qqq_flow20"] / (fl["aum_tqqq"] + fl["aum_sqqq"])).to_numpy()
    cols["flow20p"] = pd.Series(f20, index=wk(fl["asof"])).groupby(level=0).last()
    si = pd.read_parquet(H.CACHE / "finra_si.parquet")
    cols["si_ai_chg"] = pd.Series(si["si_ai_chg"].to_numpy(), index=wk(si["asof"])).groupby(level=0).last()
    cb = osf.load_cboe()
    eq = cb["equity_pv"].rolling(5).sum() / cb["equity_cv"].rolling(5).sum()
    cols["eq_pcr5"] = pd.Series(eq.to_numpy(), index=wk(eq.index)).dropna().groupby(level=0).last()
    occ = osf.load_occ(osb.ai_us_symbols())
    cc = occ.pivot_table(index="date", columns="underlying", values="C_C", aggfunc="sum").fillna(0).sum(axis=1)
    cp = occ.pivot_table(index="date", columns="underlying", values="C_P", aggfunc="sum").fillna(0).sum(axis=1)
    ai = cp.rolling(5).sum() / cc.rolling(5).sum()
    cols["ai_pcr5"] = pd.Series(ai.to_numpy(), index=wk(ai.index)).dropna().groupby(level=0).last()
    sv = pd.read_parquet(H.CACHE / "surveys.parquet").dropna(subset=["aaii_spread"])
    cols["aaii_spread"] = pd.Series(sv["aaii_spread"].to_numpy(), index=wk(sv["aaii_asof"])).groupby(level=0).last()
    px = H.prices()
    c = pd.Series(px["QQQ_c"].to_numpy(), index=wk(px.index)).groupby(level=0).last()
    cols["qqq_ret"] = (c / c.shift() - 1) * 100
    M = pd.DataFrame(cols).sort_index()
    M = M[(M.index >= pd.Period(START, "W-FRI")) & (M.index <= pd.Period(px.index[-1], "W-FRI") - 1)]
    M["si_ai_chg"] = M["si_ai_chg"].ffill(limit=2)
    M["dix5_chg"] = M["dix5"].diff()
    return M


# ---------- 시차 상관 + 블록 부트스트랩 ----------
def _rank(s: pd.Series) -> np.ndarray:
    return s.rank(pct=True).to_numpy()


def _shift(y: np.ndarray, k: int) -> np.ndarray:
    """out[t] = y[t+k]"""
    out = np.full(len(y), np.nan)
    if k >= 0:
        out[:len(y) - k] = y[k:]
    else:
        out[-k:] = y[:len(y) + k]
    return out


def _corr_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """행마다 NaN 빼고 피어슨. X, Y: (B, n)."""
    m = ~np.isnan(X) & ~np.isnan(Y)
    c = m.sum(1)
    Xz, Yz = np.where(m, X, 0.0), np.where(m, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = Xz.sum(1) / c, Yz.sum(1) / c
        dx, dy = np.where(m, X - mx[:, None], 0.0), np.where(m, Y - my[:, None], 0.0)
        r = (dx * dy).sum(1) / np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
    r[c < MIN_PAIRS] = np.nan
    return r


def block_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / BLOCK))
    st = rng.integers(0, n - BLOCK + 1, size=(NB, nb))
    return (st[:, :, None] + np.arange(BLOCK)).reshape(NB, -1)[:, :n]


def xcorr(Wf: pd.DataFrame, pairs: list[tuple[str, str]], fwd_from: dict, I: np.ndarray, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """모든 쌍 × 시차의 관측 ρ·95% 구간, 쌍별 앞섬 비대칭. fwd_from[pair] = '앞쪽' 첫 시차(1 또는 QQQ 앎 시점이면 0)."""
    R = {c: _rank(Wf[c]) for c in set(sum(map(list, pairs), []))}
    raw = {c: Wf[c].to_numpy(dtype=float) for c in R}
    cells, asym = [], []
    for a, b in pairs:
        x, y = R[a], R[b]
        boot = {}
        for k in LAGS:
            yk = _shift(y, k)
            obs = _corr_rows(x[None, :], yk[None, :])[0]
            bb = _corr_rows(x[I], yk[I])
            boot[k] = bb
            lo, hi = np.nanpercentile(bb, [2.5, 97.5]) if np.isfinite(bb).sum() > 100 else (np.nan, np.nan)
            pr = pd.Series(raw[a]).corr(pd.Series(_shift(raw[b], k)))
            n = int((~np.isnan(x) & ~np.isnan(yk)).sum())
            cells.append({"view": tag, "x": a, "y": b, "lag": int(k), "rho": obs, "lo": lo, "hi": hi, "n": n,
                          "sig": bool(lo > 0 or hi < 0), "pearson_raw": pr})
        f0 = fwd_from[(a, b)]
        fw = [k for k in LAGS if f0 <= k <= f0 + 7]
        bw = [k for k in LAGS if -8 <= k <= -1]
        ob = {k: next(c["rho"] for c in cells[-len(LAGS):] if c["lag"] == k) for k in LAGS}
        A_obs = np.nanmean([ob[k] ** 2 for k in fw]) - np.nanmean([ob[k] ** 2 for k in bw])
        A_b = np.nanmean(np.column_stack([boot[k] ** 2 for k in fw]), 1) - np.nanmean(np.column_stack([boot[k] ** 2 for k in bw]), 1)
        lo, hi = np.nanpercentile(A_b, [2.5, 97.5])
        pk = max(LAGS, key=lambda k: abs(ob[k]) if np.isfinite(ob[k]) else -1)
        asym.append({"view": tag, "x": a, "y": b, "fwd_lags": f"{fw[0]}..{fw[-1]}", "A": A_obs, "A_lo": lo, "A_hi": hi,
                     "lead": "x앞섬" if lo > 0 else ("y앞섬" if hi < 0 else "불명"),
                     "peak_lag": int(pk), "peak_rho": ob[pk]})
    return pd.DataFrame(cells), pd.DataFrame(asym)


# ---------- (2) 엇갈린 주 ----------
def circ_p(sig: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """sig(0/1), y(유효 주만). 모든 0 아닌 원형 이동. (p_greater, p_less, 이동 수)"""
    n = len(sig)
    obs = y[sig > 0].mean() - y.mean()
    null = np.array([y[np.roll(sig, s) > 0].mean() - y.mean() for s in range(1, n) if np.roll(sig, s).sum() > 0])
    return float(((null >= obs).sum() + 1) / (len(null) + 1)), float(((null <= obs).sum() + 1) / (len(null) + 1)), len(null)


def perm_p(sig: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float]:
    """보조: 신호 주 이름표 무작위 재배치(자기상관 무시 → 원형 이동보다 관대). (p_greater, p_less)"""
    obs = y[sig > 0].mean()
    k = int(sig.sum())
    null = np.array([y[rng.permutation(len(y))[:k]].mean() for _ in range(n)])
    return float(((null >= obs).sum() + 1) / (n + 1)), float(((null <= obs).sum() + 1) / (n + 1))


def disagree(W: pd.DataFrame, max_age: float | None = None) -> dict:
    ok = W["nq_lev_1w"].notna() & W["flow1w"].notna() & W["cftc_new"]
    if max_age is not None:  # 민감도: 기준일이 오래된(셧다운 밀린 공개) 주 제외
        ok &= W["cftc_age"] <= max_age
    prng = np.random.default_rng(SEED + 7)
    A = ((W["nq_lev_1w"] > 0) & (W["flow1w"] < 0)).where(ok)
    Bs = ((W["nq_lev_1w"] < 0) & (W["flow1w"] > 0)).where(ok)
    sigs = {"HF매수_개인매도": (A, +1), "HF매도_개인매수": (Bs, -1)}  # 부호 = 헤지펀드가 맞았을 때 차이 부호
    idx = W.index
    rows = []
    for nm, (s, hf_sign) in sigs.items():
        for k in (1, 2, 4):
            y = W[f"fwd{k}"]
            for per in ("front", "back", "y2025"):
                m = H.period_mask(idx, per) & s.notna().to_numpy() & y.notna().to_numpy()
                ss, yy = s.to_numpy()[m].astype(float), y.to_numpy()[m]
                n_sig = int(ss.sum())
                rec = {"cond": nm, "k_weeks": k, "period": per, "n_weeks": int(m.sum()), "n_sig": n_sig,
                       "mean_sig": float(yy[ss > 0].mean()) if n_sig else np.nan, "mean_all": float(yy.mean()) if m.any() else np.nan}
                rec["diff"] = rec["mean_sig"] - rec["mean_all"] if n_sig else np.nan
                if n_sig >= 2 and n_sig < m.sum():
                    pg, pl, ns = circ_p(ss, yy)
                    qg, ql = perm_p(ss, yy, prng)
                    rec.update({"p_greater": pg, "p_less": pl, "p_two": min(1.0, 2 * min(pg, pl)), "n_shifts": ns,
                                "p_floor": 1 / (ns + 1), "perm_p_greater": qg, "perm_p_less": ql})
                rec["hf_right"] = bool(np.sign(rec["diff"]) == hf_sign) if n_sig else None
                rows.append(rec)
    df = pd.DataFrame(rows)
    verdict = []
    for (nm, k), g in df.groupby(["cond", "k_weeks"]):
        f, b = g[g.period == "front"].iloc[0], g[g.period == "back"].iloc[0]
        if np.isnan(f["diff"]) or np.isnan(b["diff"]):
            ok_ = False
            pb = qb = np.nan
        else:
            same = np.sign(f["diff"]) == np.sign(b["diff"])
            pb = b["p_greater"] if f["diff"] > 0 else b["p_less"]
            qb = b["perm_p_greater"] if f["diff"] > 0 else b["perm_p_less"]
            ok_ = bool(same and pb < 0.05)
        verdict.append({"cond": nm, "k_weeks": int(k), "front_diff": f["diff"], "back_diff": b["diff"], "back_p_dir": pb,
                        "back_p_floor": b.get("p_floor", np.nan), "back_perm_p_dir": qb,
                        "y2025_diff": g[g.period == "y2025"].iloc[0]["diff"], "pass": ok_})
    # 단면 수: 기간별 신호 주
    counts = {per: {nm: int(np.nansum(s.to_numpy()[H.period_mask(idx, per)])) for nm, (s, _) in sigs.items()}
              for per in ("front", "back", "y2025")}
    counts["all_2024_2026"] = {nm: int(np.nansum(s.to_numpy())) for nm, (s, _) in sigs.items()}
    counts["valid_weeks"] = {per: int((ok.to_numpy() & H.period_mask(idx, per)).sum()) for per in ("front", "back", "y2025")}
    counts["excluded_no_new_cftc"] = int((W["nq_lev_1w"].notna() & ~W["cftc_new"]).sum())
    return {"rows": df, "verdict": pd.DataFrame(verdict), "counts": counts}


def aligned_variant() -> pd.DataFrame:
    """사후 보조(판정 없음, 결과를 본 뒤 추가). 본 시험은 CFTC 변화(측정 약 1.5주 전)와 개인 흐름(지난 1주)의 측정 창이 어긋난다.
    여기서는 같은 창으로 맞춘다: CFTC 기준일 T_{i-1}→T_i의 nq 레버리지 순/OI 변화, ProShares 순유입은 T_{i-1} < 측정일 ≤ T_i.
    진입 e = 그 보고서 known 다음 거래일(두 값 모두 그 전에 공개). 앞 수익 = 시가(e+5k거래일) ÷ 시가(e) − 1, k = 1·2·4.
    기준일→known이 7일을 넘는(셧다운·ION으로 밀린) 보고서는 뺀다."""
    px = H.prices()
    dd = px.index.to_numpy()
    o = px["QQQ_o"].to_numpy()
    cf = pd.read_parquet(H.CACHE / "cftc_tff.parquet")
    cf = cf.iloc[np.argsort(pd.to_datetime(cf["asof"]).to_numpy(), kind="stable")]
    x = (cf["nq_lev_net"] / cf["nq_oi"]).diff(1).to_numpy()
    hp = pd.read_parquet(H.CACHE / "hf_proxies.parquet")
    f = hp.dropna(subset=["lev_qqq_flow"]).set_index("asof")["lev_qqq_flow"]
    fa, fv = f.index.to_numpy(), f.to_numpy()
    asofs, knowns = cf["asof"].to_numpy(), cf.index.to_numpy()
    rows = []
    for i in range(1, len(cf)):
        T0, T1, K = asofs[i - 1], asofs[i], knowns[i]
        if (pd.Timestamp(K) - pd.Timestamp(T1)).days > 7:
            continue
        e = int(np.searchsorted(dd, K, side="right"))
        if e >= len(dd):
            continue
        m = (fa > T0) & (fa <= T1)
        rec = {"entry": dd[e], "nq_lev_1w": x[i], "flow1w": fv[m].sum() if m.any() else np.nan, "cftc_new": True, "cftc_age": 0.0}
        for k in (1, 2, 4):
            j = e + 5 * k
            rec[f"fwd{k}"] = (o[j] / o[e] - 1) * 100 if j < len(dd) else np.nan
        rows.append(rec)
    E = pd.DataFrame(rows).set_index("entry")
    return E[~E.index.duplicated(keep="last")]


def prewhiten(Wf: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """사후 보조: 각 계열(순위)을 자기 AR(1) 잔차로 바꿔 지속성이 시차 상관을 넓게 번지게 하는 것을 줄인다(계수는 전 구간 추정, 설명용)."""
    out = Wf.copy()
    for c in cols:
        r = Wf[c].rank(pct=True)
        ok = r.notna() & r.shift(1).notna()
        phi = float(np.polyfit(r.shift(1)[ok], r[ok], 1)[0])
        out[c] = r - phi * r.shift(1)
    return out


def calibration(Wf: pd.DataFrame, pairs: list, fwd: dict, I: np.ndarray, shifts=(30, 50, 70, 90)) -> dict:
    """보조: y 계열을 원형으로 크게 밀어 관계를 끊은 뒤 같은 절차로 '구간이 0을 뺀 칸' 비율을 잰다(5%면 구간이 맞게 잡힌 것)."""
    out = {}
    for sh in shifts:
        Ws = Wf.copy()
        ps, fw = [], {}
        for a, b in pairs:
            nb = f"{b}__s"
            Ws[nb] = np.roll(Wf[b].to_numpy(), sh)
            ps.append((a, nb))
            fw[(a, nb)] = fwd[(a, b)]
        c, a_ = xcorr(Ws, ps, fw, I, f"calib{sh}")
        out[sh] = {"cells": len(c), "sig_rate": float(c.sig.mean()), "lead_nonnull_pairs": int((a_["lead"] != "불명").sum())}
    return out


def partial_check(Wf: pd.DataFrame, fwd_cells: pd.DataFrame) -> pd.DataFrame:
    """보조: 앞쪽(k ≥ 0)에서 구간이 0을 뺀 QQQ 칸이 '지난 QQQ 수익(4·8주, d_t 시가까지)'의 대리인지 본다.
    x 순위와 r_{t+k} 순위를 지난 수익 순위 2개에 회귀한 잔차끼리 상관(부분 순위 상관)."""
    o = H.prices()["QQQ_o"].reindex(Wf.index)
    past = pd.DataFrame({"p4": o / o.shift(4) - 1, "p8": o / o.shift(8) - 1}).rank(pct=True)
    rows = []
    for _, r in fwd_cells.iterrows():
        x = Wf[r.x].rank(pct=True)
        y = pd.Series(_shift(Wf["qqq_ret"].rank(pct=True).to_numpy(), int(r.lag)), index=Wf.index)
        d = pd.concat([x.rename("x"), y.rename("y"), past], axis=1).dropna()
        Z = np.column_stack([np.ones(len(d)), d[["p4", "p8"]].to_numpy()])
        res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        rows.append({"x": r.x, "lag": int(r.lag), "rho": r.rho, "rho_partial": float(np.corrcoef(res(d.x.to_numpy()), res(d.y.to_numpy()))[0, 1]),
                     "corr_x_past8": float(d.x.corr(d.p8)), "n": len(d)})
    return pd.DataFrame(rows)


def main() -> None:
    W, meta = weekly()
    Wf = W[W.index >= START]
    rng = np.random.default_rng(SEED)
    I = block_idx(len(Wf), rng)
    pairs = [(a, b) for a in HF for b in list(RETAIL) + list(MKT)] + [(b, "qqq_ret") for b in RETAIL]
    fwd = {p: (0 if p[1] == "qqq_ret" else 1) for p in pairs}
    cells, asym = xcorr(Wf, pairs, fwd, I, "know")
    # 보조 1: 측정 시점 정렬(매매 불가)
    M = weekly_measured()
    I2 = block_idx(len(M), np.random.default_rng(SEED + 1))
    cells_m, asym_m = xcorr(M, pairs, {p: 1 for p in pairs}, I2, "measured")
    # 보조 2(사후): 앎 시점 + AR(1) 잔차
    allcols = list(HF) + list(RETAIL) + list(MKT)
    cells_w, asym_w = xcorr(prewhiten(Wf, allcols), pairs, fwd, I, "know_prewhitened")
    allc = pd.concat([cells, cells_m, cells_w], ignore_index=True)
    allc.to_parquet(H.RES / "leadlag_cells.parquet")
    # 하위 기간 안정성(설명용): 관측 ρ만
    def sub_rho(a, b, k, lo, hi):
        m = (Wf.index >= lo) & (Wf.index <= hi)
        x, y = Wf[a].rank(pct=True).to_numpy(), _shift(Wf[b].rank(pct=True).to_numpy(), k)
        mm = m & ~np.isnan(x) & ~np.isnan(y)
        return (float(np.corrcoef(x[mm], y[mm])[0, 1]) if mm.sum() >= 8 else np.nan, int(mm.sum()))
    subs = {"y2024": ("2024-01-01", "2024-12-31"), "y2025": H.PERIODS["y2025"], "front": H.PERIODS["front"], "back": H.PERIODS["back"]}
    def stability(sel: pd.DataFrame) -> pd.DataFrame:
        out = []
        for _, r in sel.iterrows():
            rec = {"x": r.x, "y": r.y, "lag": int(r.lag), "rho": r.rho, "lo": r.lo, "hi": r.hi}
            for nm, (lo, hi) in subs.items():
                rec[nm], rec[f"{nm}_n"] = sub_rho(r.x, r.y, int(r.lag), lo, hi)
            out.append(rec)
        return pd.DataFrame(out)
    stab = stability(cells[cells.sig].assign(ab=lambda d: d.rho.abs()).nlargest(25, "ab"))
    fwd_cells = stability(cells[cells.sig & (cells.y == "qqq_ret") & (cells.lag >= 0)].sort_values(["x", "lag"]))
    ac1 = {c: float(Wf[c].autocorr(1)) for c in allcols}
    pchk = partial_check(Wf, fwd_cells)
    # 보조: 앞쪽 QQQ 칸을 2024-01~2026-05(front 끝)로만 고르고 back(6~10월)에서 부호만 확인
    from scipy.stats import binomtest
    Wt = Wf[Wf.index <= H.PERIODS["front"][1]]
    qp = [p for p in pairs if p[1] == "qqq_ret"]
    ct, _ = xcorr(Wt, qp, {p: 0 for p in qp}, block_idx(len(Wt), np.random.default_rng(SEED + 2)), "train_to_front")
    pick = ct[ct.sig & (ct.lag >= 0)]
    oos = []
    for _, r in pick.iterrows():
        rb, nb = sub_rho(r.x, r.y, int(r.lag), *H.PERIODS["back"])
        oos.append({"x": r.x, "lag": int(r.lag), "rho_train": r.rho, "rho_back": rb, "n_back": nb, "same_sign": bool(np.sign(rb) == np.sign(r.rho))})
    oos = pd.DataFrame(oos)
    oos_sum = {"train_cells": int(len(ct[ct.lag >= 0])), "picked": len(oos), "same_sign_back": int(oos.same_sign.sum()) if len(oos) else 0,
               "binom_p_one_sided": float(binomtest(int(oos.same_sign.sum()), len(oos), 0.5, alternative="greater").pvalue) if len(oos) else np.nan}
    calib = calibration(Wf, pairs, fwd, I)
    pair_sig = (cells.groupby(["x", "y"]).agg(n_sig=("sig", "sum"), peak=("rho", lambda r: r.loc[r.abs().idxmax()]),
                                              peak_lag=("rho", lambda r: int(cells.loc[r.abs().idxmax(), "lag"])))
                .reset_index().sort_values("n_sig", ascending=False))
    dis = disagree(W)
    dis_age = disagree(W, max_age=17)
    E = aligned_variant()
    dis_al = disagree(E)
    # 요약
    def tally(c: pd.DataFrame) -> dict:
        out = {}
        for grp, g in (("HF×개인·옵션", c[c.y.isin(RETAIL) & c.x.isin(HF)]), ("HF×QQQ", c[c.x.isin(HF) & (c.y == "qqq_ret")]),
                       ("개인·옵션×QQQ", c[c.x.isin(RETAIL)])):
            out[grp] = {"cells": len(g), "sig": int(g.sig.sum()), "expected_5pct": round(0.05 * len(g), 1),
                        "sig_lag_neg": int(g[g.lag < 0].sig.sum()), "sig_lag_0": int(g[g.lag == 0].sig.sum()),
                        "sig_lag_pos": int(g[g.lag > 0].sig.sum())}
        return out
    def lead_tally(a: pd.DataFrame) -> dict:
        out = {}
        for grp, g in (("HF×개인·옵션", a[a.y.isin(RETAIL) & a.x.isin(HF)]), ("HF×QQQ", a[a.x.isin(HF) & (a.y == "qqq_ret")]),
                       ("개인·옵션×QQQ", a[a.x.isin(RETAIL)])):
            out[grp] = g["lead"].value_counts().to_dict() | {"pairs": len(g)}
        return out
    pack = lambda d: {"counts": d["counts"], "rows": d["rows"].to_dict("records"), "verdict": d["verdict"].to_dict("records")}
    res = {
        "meta": meta | {"weeks_xcorr_know": len(Wf), "weeks_xcorr_measured": len(M), "pairs": len(pairs), "lags": len(LAGS),
                        "cells_per_view": len(pairs) * len(LAGS), "boot": NB, "block_weeks": BLOCK, "seed": SEED,
                        "corr": "전 구간 순위(0~1)로 바꾼 뒤 피어슨", "min_pairs": MIN_PAIRS,
                        "series_n_know": {c: int(Wf[c].notna().sum()) for c in allcols}, "ac1_know": ac1,
                        "cftc_age_days_know": W["cftc_age"].value_counts().sort_index().to_dict()},
        "tally": {"know": tally(cells), "measured": tally(cells_m), "know_prewhitened": tally(cells_w)},
        "lead": {"know": lead_tally(asym), "measured": lead_tally(asym_m), "know_prewhitened": lead_tally(asym_w)},
        "asym_know": asym.to_dict("records"), "asym_measured": asym_m.to_dict("records"), "asym_know_prewhitened": asym_w.to_dict("records"),
        "calibration_shifted_null": calib,
        "pair_sig_know": pair_sig.to_dict("records"),
        "top_cells_know_stability": stab.to_dict("records"),
        "forward_qqq_cells_know": fwd_cells.to_dict("records"),
        "forward_qqq_cells_partial_past_ret": pchk.to_dict("records"),
        "forward_qqq_oos": {"summary": oos_sum, "cells": oos.to_dict("records")},
        "disagree": pack(dis),
        "disagree_sens_age17": pack(dis_age),
        "disagree_posthoc_aligned": pack(dis_al) | {"events": len(E), "first": E.index[0], "last": E.index[-1]},
    }
    H.save("leadlag", res)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    print(meta, "weeks know", len(Wf), "measured", len(M))
    print("tally", res["tally"])
    print("lead", res["lead"])
    print("calib", calib)
    print(pair_sig.head(25).round(3).to_string())
    print(stab.round(2).to_string())
    print(asym_w[asym_w.lead != "불명"].round(3).to_string())
    print(asym_m[asym_m.lead != "불명"].round(3).to_string())
    print(asym[asym.lead != "불명"].round(3).to_string())
    print(fwd_cells.round(2).to_string())
    print(pchk.round(3).to_string())
    print("OOS", oos_sum)
    print(oos.round(3).to_string())
    for nm, d in (("main", dis), ("age17", dis_age), ("aligned", dis_al)):
        print("==", nm, d["counts"])
        print(d["rows"].round(3).to_string())
        print(d["verdict"].round(3).to_string())


if __name__ == "__main__":
    main()
