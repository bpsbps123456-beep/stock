"""N4 갈래 h14 — PB 자금 흐름 전수 표본(252그룹)으로 H1~H4 검정. BRIEF.md §N4(2026-10-26 결과 전 고정).

py code/n4h14_run.py

입력(읽기만)
  research/pb_flow_census_20261026/{census.csv, coverage.csv, legacy_crosswalk.csv}
  research/news_turning_points_20261025/{event_outcomes.csv, metrics.json}  (Codex 31건 비교용)
  cache/n1_raw.parquet  (yfinance 일봉, 분할 반영·배당 미포함 = Codex와 같은 기준)
출력: results/n4h14_*.csv / .json / .txt

정의(등록)
  H1  family D 이고 lookback_months >= 12  → 5·10·20일 SOXX 가 대조보다 높지 않다
  H2  family R                              → 5·10·20일 SOXX 가 대조보다 높다
  H3  순매도 노트(flow_direction=net_sell 또는 long_flow=sell, lookback 있음) → lookback 과 20일 SOXX 의 스피어만 > 0
  H4  family T                              → 20일 SOXX 가 대조보다 낮다
  수익 r_h = 종가[진입+h-1] / 시가[진입] - 1 (Codex analyze.py 와 같음)
  대조 A(등록) = 같은 해 + 직전 20일 고점 대비 낙폭 구간(0~5·5~10·10%+, 진입 전날 종가까지) 날의 평균
  시점 = published_at 만 사용. exact 이면 뉴욕 시각 09:30 전 공개 → 그날 시가, 아니면 다음 정규장 시가.
         날짜만 있는 행(date_article·date_feed·date_url) → 그 날짜 다음 정규장 시가. published_at 없는 7그룹은 제외.
  판정 = 날짜 묶음 부트스트랩 95% 구간 + 원형 이동 귀무(공통 이동 k, 15 <= k <= D-15)

사후로 정한 것(결과 표에 '사후' 표시)
  - 판정 규칙의 세부: H2·H3·H4 통과 = 구간이 0을 넘고(방향 맞음) 귀무 한쪽 p < 0.05.
    H1('높지 않다')은 반증 조건만 정할 수 있어, 구간 하한 > 0 이고 귀무 위쪽 p < 0.05 이면 '반증', 아니면 '유지',
    구간 상한 < 0 이고 귀무 아래쪽 p < 0.05 이면 '유지(낮음 확인)'. 신호 5개 미만 또는 묶음 3개 미만이면 '표본 부족'.
  - family 대안 표(coding_notes·reliability.csv 를 읽고 손으로 옮김), 범위(SOXX 관련) 거르기, D 전체(lookback 무관).
"""
from __future__ import annotations

import json
import pathlib
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore", "Mean of empty slice")
T0 = time.time()
HERE = pathlib.Path(__file__).resolve().parent.parent
ROOT = HERE.parents[3]
CENSUS = ROOT / "research" / "pb_flow_census_20261026"
CODEX = ROOT / "research" / "news_turning_points_20261025"
RES = HERE / "results"
RES.mkdir(exist_ok=True)

HS = (5, 10, 20)
ASSETS = ("SOXX", "QQQ", "SPY")
W0, W1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-10-23")
B_MEAN = 10000
B_RHO = 4000
SEED = 20261026
KMIN = 15
CHAIN = 5          # 같은 family 연속 노트: 직전 채택 신호와 진입일 차 <= 5 거래일이면 뒤 신호 제외(Codex 규칙)
MIN_N, MIN_CL = 5, 3

# ---------------------------------------------------------------- family 대안(사후, 손 옮김)
# A = 명시적 대안: 다른 부호자의 family=, 'X 대안', '엄격 기준이면 X', '그 경우 지배 사실은 X', 'X로 재검토',
#     reliability.csv 맹검 재부호화 불일치.
ALT_A = {
    "G0001": ("R", "D", "다른 부호자 행 family=D(claude 9f7827b16caf, codex d072556e3983)"),
    "G0007": ("D", "N", "Codex 검토행 N(상충 보존)"),
    "G0012": ("R", "T", "'T 대안 보존'"),
    "G0032": ("N", "S", "reliability 재부호화 N→S(ad98d9050c54)"),
    "G0043": ("N", "T", "reliability 재부호화 N→T(89c0590362e8#1 JPM)"),
    "G0061": ("N", "S", "'S 대안은 상충으로 보존'"),
    "G0066": ("D", "N", "다른 부호자 행 family=N(claude 9a0a216e9cbb)"),
    "G0082": ("D", "N", "'기준이 짧은 기간이면 N으로 재검토'"),
    "G0086": ("D", "N", "다른 부호자 행 family=N(claude d33ea50767ac·7658318ff8a6)"),
    "G0131": ("D", "R", "reliability 재부호화 D→R(6ebf9522baab), '미국 재매수(R 성격)와 상충 보존'"),
    "G0132": ("S", "D", "'엄격 기준이면 D/N 대안'(앞의 D)"),
    "G0150": ("R", "T", "'순노출 수준의 5년 최고라면 T 대안'"),
    "G0157": ("R", "N", "'세계 기준이면 매도(N), 상충 보존'"),
    "G0164": ("D", "N", "'N 대안 보존'"),
    "G0167": ("N", "S", "'S 대안 보존'"),
    "G0183": ("R", "N", "'N 대안'"),
    "G0187": ("S", "D", "'대안 D(기록적 매도 크기)'"),
    "G0199": ("S", "D", "'그 경우 지배 사실은 D'"),
    "G0211": ("T", "R", "reliability 재부호화 T→R(44c26ac783a2)"),
    "G0228": ("N", "T", "'T 대안은 상충으로 기록'"),
    "G0232": ("R", "N", "다른 부호자 행 family=N(claude 71143e864253)"),
    "G0238": ("T", "N", "'일반 매도(N형)와 높은 총레버리지(T형)가 공존 … 상충 보존'"),
    "G0246": ("D", "S", "reliability 재부호화 D→S(2da96dc2e812); 메모의 N 대안도 있음"),
    "G0249": ("D", "N", "'엄격 기준이면 N 대안'"),
    "G0250": ("D", "N", "다른 부호자 행 family=N(claude ba5d5a5e39c0)"),
    "G0252": ("S", "N", "다른 부호자 행 family=N(claude a230b95e1f70·2dac9b7885b3)"),
}
# B = A + 같은 노트에 함께 적힌 다른 family 사실('R 하위 사실', 'T 보조사실', 'D형 하위 범위 사실', '상충 R 사실' 등).
ALT_B_EXTRA = {
    "G0004": ("D", "R", "'단일종목 롱 매수>숏 매도(R 하위 사실)'"),
    "G0009": ("N", "R", "'일본·유럽 R 하위 사실 공존'"),
    "G0042": ("N", "T", "'기록 근처 유지(T 상충 보조사실)'"),
    "G0076": ("T", "R", "'공존 R 사실: 저가 매수'"),
    "G0089": ("N", "T", "'팩터 보유 수준(T 보조사실)'"),
    "G0094": ("D", "R", "'숏 청산 R은 상충으로 보존'"),
    "G0105": ("R", "T", "'레버리지 고점권은 T 보조 사실'"),
    "G0115": ("D", "T", "'상충 T 사실: 유틸리티 배분 5년 96·99백분위'"),
    "G0120": ("N", "R", "'상충 R 사실: TMT 3개월 최고 속도로 추가'"),
    "G0137": ("R", "D", "'1주 전 순매도 약 15년 최대(D/S 사실)'"),
    "G0139": ("N", "R", "'상충 R 사실: 개인 19주 연속 순매수'"),
    "G0141": ("R", "T", "'T 보조사실: 금융 보유 비중 2년 최고'"),
    "G0143": ("S", "R", "'숏 커버 관측은 R 보조사실'"),
    "G0149": ("R", "N", "'평범한 매도(N) … 상충 보존'"),
    "G0171": ("R", "S", "'숏 급증(S형)과 청산(R형) 상충'"),
    "G0172": ("R", "D", "'섹터 범위 D 성격 사실'"),
    "G0181": ("N", "D", "'섹터 D형 보조 사실'"),
    "G0184": ("T", "D", "'D형 보조 사실: 북미 포지션 해소 4개월 최고 속도'"),
    "G0192": ("S", "T", "'총노출 사상 최고 근접(T 보조 사실)'"),
    "G0200": ("R", "D", "'기술주 매도 5년 최대는 D형 하위 범위 사실'"),
    "G0207": ("N", "S", "'상충: 매크로 숏 10년 최고(S), AI 바구니 노출 기록 근처(T)'(앞의 S)"),
    "G0216": ("N", "R", "'개인 매수는 R 성격 하위 사실'"),
    "G0230": ("R", "D", "'직전 D 사실로 상충 기록'"),
    "G0241": ("R", "T", "''crowded' … T로 올리지 않음(T 재검토)'"),
    "G0243": ("N", "R", "'반도체장비·소프트웨어 유입은 R 성격 하위 사실'"),
}
ALT_B = {**ALT_A, **ALT_B_EXTRA}

# 범위 거르기(사후): SOXX 와 무관해 보이는 범위만 뺀다.
NONTECH = re.compile(r"(금융|은행|헬스케어|에너지|필수소비재|유틸리티|산업재|원자재|방어주|경기민감|소형주|가치|부동산|소비재|방산)")
TECH = re.compile(r"(정보기술|기술|TMT|반도체|메가캡|AI|MAGMA|Mag)")
NONUS = re.compile(r"(유럽|아시아|중국|일본|홍콩|신흥|EM)")
US_GLOBAL = re.compile(r"(US|미국|북미|세계|전체|Prime book|지역 미기재)")


def scope_ok(s) -> bool:
    s = "" if pd.isna(s) else str(s)
    if NONTECH.search(s) and not TECH.search(s):
        return False
    if NONUS.search(s) and not US_GLOBAL.search(s):
        return False
    return True


# ---------------------------------------------------------------- 1. 가격·일별 표
def load_prices():
    raw = pd.read_parquet(HERE / "cache" / "n1_raw.parquet")
    raw = raw[raw.sym.isin(ASSETS) & (raw.date >= "2022-06-01") & (raw.date <= W1)]
    O = raw.pivot(index="date", columns="sym", values="O")[list(ASSETS)]
    C = raw.pivot(index="date", columns="sym", values="C")[list(ASSETS)]
    assert O.notna().all().all() and C.notna().all().all(), "가격 결측"
    assert (O > 0).all().all() and (C > 0).all().all()
    return O, C


def build_daily(O, C, notes_weeks):
    sess = O.index
    ev = (sess >= W0) & (sess <= W1)
    pos0 = int(np.flatnonzero(ev)[0])
    days = sess[ev]
    D = len(days)
    info = pd.DataFrame(index=days)
    info["pos"] = np.arange(D)
    info["year"] = days.year
    info["wd"] = days.weekday
    info["week"] = (days - pd.to_timedelta(days.weekday, unit="D")).normalize()
    info = info.join(notes_weeks, on="week")
    info["nw_strict"] = info["nw_strict"].fillna(False).astype(bool)
    info["nw_any"] = info["nw_any"].fillna(False).astype(bool)
    arrs = {}
    for a in ASSETS:
        o, c = O[a], C[a]
        dd20 = (c / c.rolling(20, min_periods=20).max() - 1).shift(1)
        bucket = pd.cut(-dd20, [-np.inf, .05, .10, np.inf], right=False, labels=[0, 1, 2]).astype("Int64")
        info[f"{a}_bucket"] = bucket.reindex(days).to_numpy()
        for h in HS:
            r = (c.shift(-(h - 1)) / o - 1).reindex(days)
            # 창 끝 넘어가는 날은 NaN 이어야 한다
            r = r.where(np.arange(D) + h - 1 <= D - 1 + 0)
            arrs[(a, h)] = r.to_numpy(float)
    assert info[[f"{a}_bucket" for a in ASSETS]].notna().all().all()
    return info, arrs, pos0, D


def control_means(info, arrs):
    """날마다 대조 칸 평균(수익·상승 비율). 키 (asset, h, control) -> (cm, cu, cell_n)."""
    out = {}
    for a in ASSETS:
        for h in HS:
            r = arrs[(a, h)]
            df = pd.DataFrame(dict(r=r, up=np.where(np.isnan(r), np.nan, (r > 0).astype(float)),
                                   year=info.year.to_numpy(), b=info[f"{a}_bucket"].to_numpy(int),
                                   wd=info.wd.to_numpy(), s=info.nw_strict.to_numpy(), u=info.nw_any.to_numpy()))
            v = df[df.r.notna()]
            for name, keys, pool in (("A", ["year", "b"], v), ("Y", ["year"], v),
                                     ("Ws", ["wd"], v[v.s]), ("Wu", ["wd"], v[v.u])):
                g = pool.groupby(keys).agg(m=("r", "mean"), u=("up", "mean"), n=("r", "size"))
                j = df[keys].merge(g, left_on=keys, right_index=True, how="left")
                out[(a, h, name)] = (j.m.to_numpy(float), j.u.to_numpy(float), j.n.fillna(0).to_numpy(int))
    return out


# ---------------------------------------------------------------- 2. 신호
def entry_positions(cen: pd.DataFrame, sess: pd.DatetimeIndex) -> pd.Series:
    out = []
    for _, r in cen.iterrows():
        p = r.published_at
        if pd.isna(p):
            out.append(np.nan)
            continue
        if r.time_precision == "exact":
            ts = pd.Timestamp(p).tz_convert("America/New_York")
            d = pd.Timestamp(ts.date())
            k = int(sess.searchsorted(d, side="left"))
            same_day = k < len(sess) and sess[k] == d
            if same_day and (ts.hour, ts.minute) < (9, 30):
                out.append(k)
            else:
                out.append(int(sess.searchsorted(d, side="right")))
        else:
            d = pd.Timestamp(str(p)[:10])
            out.append(int(sess.searchsorted(d, side="right")))
    return pd.Series(out, index=cen.index, dtype="float")


def load_signals(O):
    cen = pd.read_csv(CENSUS / "census.csv")
    assert len(cen) == 252 and cen.group_id.is_unique
    for g, (f0, f1, _) in ALT_B.items():
        row = cen[cen.group_id == g]
        assert len(row) == 1 and row.family.iloc[0] == f0, (g, f0, row.family.tolist())
        assert f1 != f0
    sess = O.index
    cen["full_i"] = entry_positions(cen, sess)
    cen["entry"] = [sess[int(i)] if pd.notna(i) and int(i) < len(sess) else pd.NaT for i in cen.full_i]
    cen["sell_note"] = (cen.flow_direction == "net_sell") | (cen.long_flow == "sell")
    cen["fam_altA"] = [ALT_A[g][1] if g in ALT_A else f for g, f in zip(cen.group_id, cen.family)]
    cen["fam_altB"] = [ALT_B[g][1] if g in ALT_B else f for g, f in zip(cen.group_id, cen.family)]
    cen["scope_ok"] = cen.scope.map(scope_ok)
    return cen


def first_only(pos: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """keys 순으로 정렬된 상태에서 직전 채택 신호와 진입일 차 <= CHAIN 이면 제외."""
    order = np.lexsort((keys, pos))
    keep = np.zeros(len(pos), bool)
    prev = -10**9
    for t in order:
        if pos[t] - prev <= CHAIN:
            continue
        keep[t] = True
        prev = pos[t]
    return keep


# ---------------------------------------------------------------- 3. 통계
def clusters(pos: np.ndarray, h: int) -> np.ndarray:
    o = np.argsort(pos, kind="stable")
    lab = np.empty(len(pos), int)
    k = 0
    for j, t in enumerate(o):
        if j > 0 and pos[t] - pos[o[j - 1]] >= h:
            k += 1
        lab[t] = k
    return lab


def boot_mean(vals, lab, rng, B=B_MEAN):
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    sums = np.bincount(inv, weights=vals)
    cnts = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def spearman(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def spearman_rows(X, Y):
    """X, Y: (K, n) 행별 스피어만(NaN 없음 가정)."""
    rx = rankdata(X, axis=1)
    ry = rankdata(Y, axis=1)
    rx = rx - rx.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(1) / den


def boot_rho(x, y, lab, rng, B=B_RHO):
    u = np.unique(lab)
    if len(u) < 3:
        return (np.nan, np.nan)
    groups = [np.flatnonzero(lab == g) for g in u]
    pick = rng.integers(0, len(u), size=(B, len(u)))
    vals = np.empty(B)
    for b in range(B):
        idx = np.concatenate([groups[i] for i in pick[b]])
        vals[b] = spearman(x[idx], y[idx])
    vals = vals[~np.isnan(vals)]
    return tuple(np.quantile(vals, [0.025, 0.975])) if len(vals) > 100 else (np.nan, np.nan)


def shift_matrix(pos, p0, p1):
    D = p1 - p0 + 1
    ks = np.arange(KMIN, D - KMIN + 1)
    rel = pos - p0
    return p0 + (rel[None, :] + ks[:, None]) % D, len(ks)


PERIODS = [("all", "2023-01-01", "2026-10-23"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
           ("2026", "2026-01-01", "2026-10-23"), ("2026_06_10", "2026-06-01", "2026-10-23")]


def period_bounds(info, a, b):
    m = (info.index >= a) & (info.index <= b)
    p = info.pos.to_numpy()[m]
    return int(p[0]), int(p[-1])


def mean_stats(pos, h, r, cm, cu, period_pb, rng, direction):
    """pos: 평가 창 위치(이미 기간 안). r·cm·cu: 날짜 배열."""
    ok = ~np.isnan(r[pos]) & ~np.isnan(cm[pos])
    pos = pos[ok]
    n = len(pos)
    row = dict(n=n)
    if n == 0:
        return row
    rv, cv, uv = r[pos], cm[pos], cu[pos]
    up = (rv > 0).astype(float)
    ex = rv - cv
    lab = clusters(pos, h)
    lo, hi = boot_mean(ex, lab, rng)
    ulo, uhi = boot_mean(up - uv, lab, rng)
    e_all = r - cm
    S, K = shift_matrix(pos, *period_pb)
    nullm = np.nanmean(e_all[S], axis=1)
    obs = ex.mean()
    row.update(clusters=len(np.unique(lab)), mean=rv.mean(), up=up.mean(), n_up=int(up.sum()), ctl_mean=cv.mean(),
               ctl_up=uv.mean(), ex_mean=obs, ex_lo=lo, ex_hi=hi, ex_up=(up - uv).mean(), ex_up_lo=ulo,
               ex_up_hi=uhi, null_K=K, null_q05=np.nanquantile(nullm, .05), null_q95=np.nanquantile(nullm, .95),
               p_hi=(np.sum(nullm >= obs) + 1) / (K + 1), p_lo=(np.sum(nullm <= obs) + 1) / (K + 1))
    return row


def rho_stats(pos, lb, y, period_pb, rng):
    ok = ~np.isnan(y[pos])
    pos, lb = pos[ok], lb[ok]
    n = len(pos)
    row = dict(n=n)
    if n < 3:
        return row
    lab = clusters(pos, 20)
    obs = spearman(lb, y[pos])
    lo, hi = boot_rho(lb, y[pos], lab, rng)
    S, K = shift_matrix(pos, *period_pb)
    Y = y[S]
    good = ~np.isnan(Y)
    nullr = np.full(K, np.nan)
    full = good.all(1)
    if full.any():
        nullr[full] = spearman_rows(np.broadcast_to(lb, Y[full].shape), Y[full])
    for k in np.flatnonzero(~full):
        g = good[k]
        nullr[k] = spearman(lb[g], Y[k, g])
    nn = nullr[~np.isnan(nullr)]
    row.update(clusters=len(np.unique(lab)), rho=obs, rho_lo=lo, rho_hi=hi, null_K=len(nn),
               null_q05=np.quantile(nn, .05), null_q95=np.quantile(nn, .95),
               p_hi=(np.sum(nn >= obs) + 1) / (len(nn) + 1), p_lo=(np.sum(nn <= obs) + 1) / (len(nn) + 1),
               mean_r20=float(np.mean(y[pos])))
    return row


def verdict(hyp, row):
    n, cl = row.get("n", 0), row.get("clusters", 0)
    if n < MIN_N or cl < MIN_CL:
        return "표본 부족"
    if hyp == "H3":
        lo, hi = row.get("rho_lo", np.nan), row.get("rho_hi", np.nan)
    else:
        lo, hi = row.get("ex_lo", np.nan), row.get("ex_hi", np.nan)
    ph, pl = row.get("p_hi", 1), row.get("p_lo", 1)
    if hyp in ("H2", "H3"):
        return "통과" if (lo > 0 and ph < .05) else "불통과"
    if hyp == "H4":
        return "통과" if (hi < 0 and pl < .05) else "불통과"
    # H1: '높지 않다'
    if lo > 0 and ph < .05:
        return "반증"
    if hi < 0 and pl < .05:
        return "유지(낮음 확인)"
    return "유지"


# ---------------------------------------------------------------- 4. 실행
def main():
    O, C = load_prices()
    cov = pd.read_csv(CENSUS / "coverage.csv")
    wk = cov.groupby("week_start").agg(pub=("groups_published", "sum"), unv=("unverified_candidates", "sum"))
    wk.index = pd.to_datetime(wk.index)
    notes_weeks = pd.DataFrame(dict(nw_strict=(wk.pub == 0) & (wk.unv == 0), nw_any=(wk.pub == 0)))
    info, arrs, pos0, D = build_daily(O, C, notes_weeks)
    ctl = control_means(info, arrs)
    cen = load_signals(O)
    cen["pos"] = cen.full_i - pos0
    cen["in_window"] = cen.pos.notna() & (cen.pos >= 0) & (cen.pos <= D - 1)
    cen["year"] = cen.entry.dt.year

    sig = cen[cen.in_window].copy()
    sig["pos"] = sig.pos.astype(int)
    excluded = cen[~cen.in_window][["group_id", "bank", "published_at", "time_precision", "family"]]

    # 신호 표 저장(진입·수익·대조)
    st = sig[["group_id", "bank", "published_at", "time_precision", "note_date", "entry", "family", "fam_altA",
              "fam_altB", "lookback_months", "flow_direction", "long_flow", "sell_note", "scope", "scope_ok",
              "verification", "uncertain_dup", "coder", "legacy_ids"]].copy()
    for a in ASSETS:
        for h in HS:
            st[f"{a}_r{h}"] = arrs[(a, h)][sig.pos]
            st[f"{a}_A{h}"] = ctl[(a, h, "A")][0][sig.pos]
    st["SOXX_bucket"] = info["SOXX_bucket"].to_numpy()[sig.pos]
    st.sort_values(["entry", "group_id"]).to_csv(RES / "n4h14_signals.csv", index=False, encoding="utf-8-sig")
    alt_rows = [dict(group_id=g, family=v[0], alt=v[1], kind="A" if g in ALT_A else "B", basis=v[2]) for g, v in ALT_B.items()]
    pd.DataFrame(alt_rows).to_csv(RES / "n4h14_family_alt.csv", index=False, encoding="utf-8-sig")

    # 변형(신호 집합) 정의: (이름, 사후 여부, 마스크 함수, family 열)
    base_mask = pd.Series(True, index=sig.index)
    variants = [
        ("all", False, base_mask, "family"),
        ("first", False, base_mask, "family"),
        ("GS", False, sig.bank == "GS", "family"),
        ("no_uncertain_dup", False, sig.uncertain_dup.isna(), "family"),
        ("no_headline_only", False, sig.verification != "headline_summary_only", "family"),
        ("altA", True, base_mask, "fam_altA"),
        ("altB", True, base_mask, "fam_altB"),
        ("scope_soxx", True, sig.scope_ok, "family"),
    ]

    def hyp_mask(hyp, famcol):
        f = sig[famcol]
        lb = sig.lookback_months
        if hyp == "H1":
            return (f == "D") & (lb >= 12)
        if hyp == "H1_Dall":
            return f == "D"
        if hyp == "H2":
            return f == "R"
        if hyp == "H4":
            return f == "T"
        if hyp == "H3":
            return sig.sell_note & lb.notna()
        raise KeyError(hyp)

    HYPS = [("H1", HS), ("H1_Dall", HS), ("H2", HS), ("H4", (20,)), ("H3", (20,))]
    rng = np.random.default_rng(SEED)
    pbs = {p: period_bounds(info, a, b) for p, a, b in PERIODS}
    rows, rho_rows = [], []
    for hyp, hs in HYPS:
        for vname, posthoc, vmask, famcol in variants:
            if hyp == "H3" and vname in ("altA", "altB"):
                continue
            if hyp == "H1_Dall" and vname not in ("all", "first"):
                continue
            m = vmask & hyp_mask(hyp, famcol)
            g = sig[m]
            pos_all = g.pos.to_numpy(int)
            keys = g.group_id.to_numpy()
            if vname == "first":
                keep = first_only(pos_all, keys)
                g = g[keep]
                pos_all = pos_all[keep]
            ph = posthoc or hyp == "H1_Dall"
            for pname, pa, pb_ in PERIODS:
                p0, p1 = pbs[pname]
                inp = (pos_all >= p0) & (pos_all <= p1)
                pos = pos_all[inp]
                for a in ASSETS:
                    if hyp == "H3":
                        lb = g.lookback_months.to_numpy(float)[inp]
                        for cname in ("raw", "A"):
                            y = arrs[(a, 20)] if cname == "raw" else arrs[(a, 20)] - ctl[(a, 20, "A")][0]
                            r = rho_stats(pos, lb, y, (p0, p1), rng)
                            r.update(hyp=hyp, h=20, variant=vname, posthoc=ph, asset=a, control=cname, period=pname)
                            r["verdict"] = verdict("H3", r) if (a == "SOXX" and cname == "raw" and pname != "2026_06_10") else ""
                            rho_rows.append(r)
                        continue
                    for h in hs:
                        cnames = ("A", "Y", "Ws", "Wu") if a == "SOXX" else ("A",)
                        for cname in cnames:
                            cm, cu, _ = ctl[(a, h, cname)]
                            d = -1 if hyp == "H4" else 1
                            r = mean_stats(pos, h, arrs[(a, h)], cm, cu, (p0, p1), rng, d)
                            r.update(hyp=hyp, h=h, variant=vname, posthoc=ph, asset=a, control=cname, period=pname)
                            hk = "H1" if hyp == "H1_Dall" else hyp
                            r["verdict"] = verdict(hk, r) if (a == "SOXX" and cname == "A" and pname != "2026_06_10") else ""
                            rows.append(r)
    cells = pd.DataFrame(rows)
    rcells = pd.DataFrame(rho_rows)
    front = ["hyp", "h", "variant", "posthoc", "asset", "control", "period", "verdict"]
    cells = cells[front + [c for c in cells.columns if c not in front]]
    rcells = rcells[front + [c for c in rcells.columns if c not in front]]
    cells.to_csv(RES / "n4h14_cells.csv", index=False, encoding="utf-8-sig")
    rcells.to_csv(RES / "n4h14_h3_cells.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ Codex 31건과 비교(5일, SOXX, 대조 A)
    mx = json.loads((CODEX / "metrics.json").read_text(encoding="utf-8"))["families"]
    comp = []
    for fam in ("D", "S", "R", "T"):
        x = mx[fam]
        comp.append(dict(source="Codex 10/25 (2024~2026, 첫 신호)", family=fam, n5=x["n5"], n_up=x["wins5"],
                         up=x["wins5"] / x["n5"], mean5=x["mean5"], ctl_mean5=x["matched_mean5"], ctl_up5=x["matched_hit5"] if fam != "T" else 1 - x["matched_hit5"]))
    for label, yrs, mode in (("전수 2024~2026, 첫 신호", (2024, 2025, 2026), "first"),
                             ("전수 2024~2026, 전부", (2024, 2025, 2026), "all"),
                             ("전수 2023~2026, 첫 신호", (2023, 2024, 2025, 2026), "first"),
                             ("전수 2023~2026, 전부", (2023, 2024, 2025, 2026), "all")):
        for fam, extra in (("D", None), ("D>=12", None), ("S", None), ("R", None), ("T", None)):
            f = "D" if fam.startswith("D") else fam
            m = (sig.family == f) & sig.year.isin(yrs)
            if fam == "D>=12":
                m &= sig.lookback_months >= 12
            g = sig[m]
            pos = g.pos.to_numpy(int)
            if mode == "first":
                pos = pos[first_only(pos, g.group_id.to_numpy())]
            r5 = arrs[("SOXX", 5)][pos]
            cm = ctl[("SOXX", 5, "A")][0][pos]
            cu = ctl[("SOXX", 5, "A")][1][pos]
            ok = ~np.isnan(r5)
            comp.append(dict(source=label, family=fam, n5=int(ok.sum()), n_up=int((r5[ok] > 0).sum()),
                             up=float((r5[ok] > 0).mean()) if ok.any() else np.nan,
                             mean5=float(np.nanmean(r5)) if ok.any() else np.nan,
                             ctl_mean5=float(np.nanmean(cm[ok])) if ok.any() else np.nan,
                             ctl_up5=float(np.nanmean(cu[ok])) if ok.any() else np.nan))
    comp = pd.DataFrame(comp)
    comp.to_csv(RES / "n4h14_codex_compare.csv", index=False, encoding="utf-8-sig")

    # 10/25 원 기사 → 그룹 진입일 대조
    lc = pd.read_csv(CENSUS / "legacy_crosswalk.csv")
    eo = pd.read_csv(CODEX / "event_outcomes.csv", encoding="utf-8-sig")
    leg = []
    for _, x in lc.iterrows():
        e = eo[eo.id == x.legacy_id]
        ce = e.entry.iloc[0] if len(e) else None
        for gid in (str(x.group_ids).split(";") if isinstance(x.group_ids, str) else [None]):
            s = sig[sig.group_id == gid] if gid else sig.iloc[0:0]
            leg.append(dict(legacy_id=x.legacy_id, legacy_family=x.legacy_family, codex_entry=ce,
                            codex_r5=float(e.SOXX_r5.iloc[0]) if len(e) else np.nan, codex_keep=bool(e.keep.iloc[0]) if len(e) else None,
                            status=x.status, group_id=gid,
                            census_family=s.family.iloc[0] if len(s) else None,
                            census_entry=s.entry.iloc[0].date().isoformat() if len(s) else None,
                            census_r5=float(arrs[("SOXX", 5)][s.pos.iloc[0]]) if len(s) else np.nan,
                            census_lookback=s.lookback_months.iloc[0] if len(s) else np.nan))
    leg = pd.DataFrame(leg)
    leg["same_entry"] = leg.codex_entry == leg.census_entry

    # Codex 계산 재현: Codex 진입일에 우리 r5·대조 A 를 붙여 metrics.json 과 비교
    rep = []
    eok = eo[eo.keep & eo.SOXX_r5.notna()].copy()
    eok["pos"] = [int(info.pos[pd.Timestamp(d)]) for d in eok.entry]
    for fam, g in eok.groupby("family"):
        pp = g.pos.to_numpy(int)
        rep.append(dict(family=fam, n=len(g), codex_mean5=mx[fam]["mean5"], ours_mean5=float(arrs[("SOXX", 5)][pp].mean()),
                        codex_ctl5=mx[fam]["matched_mean5"], ours_ctl5=float(ctl[("SOXX", 5, "A")][0][pp].mean()),
                        max_abs_r5_diff=float(np.max(np.abs(arrs[("SOXX", 5)][pp] - g.SOXX_r5.to_numpy())))))
    rep = pd.DataFrame(rep)
    rep.to_csv(RES / "n4h14_codex_repro.csv", index=False, encoding="utf-8-sig")
    leg.to_csv(RES / "n4h14_legacy_entry.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 요약 출력
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    lines = []
    P = lines.append
    P(f"N4 h14  실행 {time.strftime('%Y-%m-%d %H:%M')}  소요 {time.time() - T0:.0f}s")
    P(f"평가 창 {info.index[0].date()}~{info.index[-1].date()}  거래일 D={D}")
    P(f"census 252 → 창 안 신호 {len(sig)}  제외 {len(excluded)}: {excluded.to_dict('records')}")
    P(f"time_precision: {sig.time_precision.value_counts().to_dict()}")
    P(f"노트 미발견 주(공개일 기준, 4개 은행 합 0): strict {int(notes_weeks.nw_strict.sum())} / 미확인 후보 포함 {int(notes_weeks.nw_any.sum())} (200주)")
    P(f"대조 Ws 풀 크기(요일별, r5): {pd.Series(info.wd[info.nw_strict]).value_counts().sort_index().to_dict()}")
    P(f"범위 거르기(사후)에서 빠진 그룹 {int((~sig.scope_ok).sum())}: {sig[~sig.scope_ok].scope.value_counts().to_dict()}")
    for hyp in ("H1", "H1_Dall", "H2", "H4", "H3"):
        mk = hyp_mask(hyp, "family")
        g = sig[mk]
        P(f"{hyp}: 신호 {len(g)} (연도 {g.year.value_counts().sort_index().to_dict()}, 은행 {g.bank.value_counts().to_dict()}), 첫 신호만 {int(first_only(g.pos.to_numpy(int), g.group_id.to_numpy()).sum())}")
    P("")
    P("== 주 표: SOXX, 대조 A, all/first ==")
    cols = ["hyp", "h", "variant", "period", "verdict", "n", "clusters", "n_up", "up", "mean", "ctl_mean", "ex_mean", "ex_lo",
            "ex_hi", "ex_up", "p_hi", "p_lo"]
    mt = cells[(cells.asset == "SOXX") & (cells.control == "A") & cells.variant.isin(["all", "first"])]
    P(mt[cols].round(4).to_string(index=False))
    P("")
    P("== H3 (SOXX, raw/A) ==")
    rc = ["hyp", "variant", "control", "period", "verdict", "n", "clusters", "rho", "rho_lo", "rho_hi", "p_hi", "p_lo", "mean_r20"]
    P(rcells[(rcells.asset == "SOXX")][rc].round(4).to_string(index=False))
    P("")
    P("== 민감도(SOXX, 대조 A, 전체 기간) ==")
    sm = cells[(cells.asset == "SOXX") & (cells.control == "A") & (cells.period == "all")]
    P(sm[["hyp", "h", "variant", "posthoc", "verdict", "n", "clusters", "mean", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_hi", "p_lo"]].round(4).to_string(index=False))
    P("")
    P("== 다른 대조(SOXX, all, 전체 기간) ==")
    oc = cells[(cells.asset == "SOXX") & (cells.variant == "all") & (cells.period == "all")]
    P(oc[["hyp", "h", "control", "n", "mean", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_hi", "p_lo"]].round(4).to_string(index=False))
    P("")
    P("== QQQ·SPY (대조 A, all, 전체 기간) ==")
    qc = cells[(cells.asset != "SOXX") & (cells.variant == "all") & (cells.period == "all")]
    P(qc[["hyp", "h", "asset", "n", "mean", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_hi", "p_lo"]].round(4).to_string(index=False))
    P("")
    P("== Codex 비교(5일 SOXX) ==")
    P(comp.round(4).to_string(index=False))
    P("")
    P("== 10/25 기사 → 그룹 진입일 ==")
    P(leg.round(4).to_string(index=False))
    P(f"진입일 같음 {int(leg.same_entry.sum())}/{int(leg.census_entry.notna().sum())}")
    P("== Codex 계산 재현(Codex 진입일, 우리 가격·대조) ==")
    P(rep.round(5).to_string(index=False))
    ver = {}
    for _, r in pd.concat([cells, rcells])[lambda x: x.verdict != ""].iterrows():
        ver[f"{r.hyp}|h{r.h}|{r.variant}|{r.period}"] = r.verdict
    tests = dict(
        mean_cells=len(cells), rho_cells=len(rcells), verdict_cells=len(ver),
        verdict_main=sum(1 for k in ver if k.split("|")[2] in ("all", "first") and not k.startswith("H1_Dall")),
        hyp_horizons_registered=8, periods_verdict=5, periods_display=1, variants=[v[0] for v in variants],
        controls_SOXX=["A(등록)", "Y(같은 해)", "Ws(노트 미발견 주 같은 요일, 미확인 후보 없음)", "Wu(노트 미발견 주 같은 요일, 미확인 후보 포함)"],
        assets=list(ASSETS), bootstrap=dict(B_mean=B_MEAN, B_rho=B_RHO, seed=SEED, cluster="진입일 차 < h 거래일이면 같은 묶음"),
        null=dict(kind="공통 원형 이동", kmin=KMIN, kmax="D-15"))
    (RES / "n4h14_verdict.json").write_text(json.dumps(dict(verdicts=ver, tests=tests), ensure_ascii=False, indent=1), encoding="utf-8")
    vs = pd.Series(ver)
    main_keys = [k for k in ver if k.split("|")[2] in ("all", "first") and not k.startswith("H1_Dall")]
    tests["verdict_counts_main"] = vs[main_keys].value_counts().to_dict()
    tests["verdict_counts_all"] = vs.value_counts().to_dict()
    tests["pass_or_refute_cells"] = {k: v for k, v in ver.items() if v in ("통과", "반증", "유지(낮음 확인)")}
    (RES / "n4h14_verdict.json").write_text(json.dumps(dict(verdicts=ver, tests=tests), ensure_ascii=False, indent=1), encoding="utf-8")
    P(json.dumps(tests, ensure_ascii=False))
    txt = "\n".join(lines)
    (RES / "n4h14_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
