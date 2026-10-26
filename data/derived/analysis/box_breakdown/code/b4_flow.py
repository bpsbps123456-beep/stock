"""갈래 4: 수급·레버리지·옵션 특징이 박스 하단 이탈을 '진짜'로 만드는가 (BRIEF 갈래 4).

사용
    py b4_flow.py features   # 특징만 만들고 범위·시점 점검(결과값은 보지 않음)
    py b4_flow.py audit      # 미래 자료 교란 시험(특징이 t+1 시가 전 공개 자료만 쓰는지)
    py b4_flow.py run        # 특징 → 칸 → harness.evaluate_cells → results/b4_flow_cells.csv, b4_flow.md

정의(결과를 보기 전에 정함). 모든 특징은 '진입일 t+1 시가 전에 공개된 관측'만 쓴다.
  배치 규칙(hf_options/code/harness.py `_place`와 같다): 원자료 행마다 when(아래)을 정하고, when 다음 거래일
  (= 그 값으로 처음 시가 진입할 수 있는 날)에 값을 놓은 뒤 age거래일까지 이어 쓴다. 이탈일 t 사건의 특징 =
  진입일 t+1에 놓인 값. 따라서 when ≤ t인 행만 들어간다.
    - 장 마감 뒤 공개 일별 자료(ProShares·GraniteShares 좌수/NAV, CBOE 풋/콜, Yahoo 종가, SqueezeMetrics
      DIX/GEX, FINRA 일별 공매도 거래량): when = 측정일 D(공개 D 저녁 → D+1 시가). FINRA는 open_ok=0(다음 날
      9:30 ET 뒤 게시)인 날만 when = known. age 3.
    - CFTC TFF(화요일 측정, 금 15:30 ET 공개, 16시 뒤 공개·휴일 예외는 QA의 known): when = known. age 15.
    - FINRA 공매도 잔고(월 2회): when = known(공개일 다음 거래일, QA 규칙). age 25.
  백분위: 원자료 자기 빈도로 지난 win개(자기 포함) 안 순위, 같은 값은 반(leverage_flood/stage2_eval.pct_rank와
  같은 식). 일별 win 250·최소 60, 주간 win 52·최소 26. 0~1.

특징 23개(날짜 고정 18, 종목별 5) — FEATS 표. 칸 = harness.tercile_cells(N별 앞 구간 3분위, 두 값이면 =1/=0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

ANA = H.ANALYSIS
LF = ANA / "leverage_flood/results"
HO = ANA / "hf_options/cache"
OUT_MD = H.RESULTS / "b4_flow.md"
OUT_CSV = H.RESULTS / "b4_flow_cells.csv"
OUT_EV = H.RESULTS / "b4_flow_events.parquet"
OUT_JSON = H.RESULTS / "b4_flow_summary.json"
OUT_AUDIT = H.RESULTS / "b4_flow_audit.json"

DAILY_AGE, WEEKLY_AGE, BIMONTH_AGE = 3, 15, 25

# 이름: (수준, 원자료, 정의, 언제 알 수 있나)
FEATS = {
    # ── 개인 레버리지 자금 흐름(날짜 고정) ──
    "nvdl_flow20_p": ("날짜", "leverage_flood us_daily_panel NVDL_flow_20d",
                      "NVDL(엔비디아 2배) 20일 순유입(좌수 변화×NAV 합)의 250일 백분위",
                      "GraniteShares 측정일 D 저녁(가정, leverage_flood 규칙) → D+1 시가"),
    "upro_flow20_p": ("날짜", "us_daily_panel UPRO_flow_20d", "UPRO(S&P 3배) 20일 순유입의 250일 백분위",
                      "ProShares D 저녁(20시 뒤) → D+1 시가"),
    "tqqq_flow20_p": ("날짜", "us_daily_panel TQQQ_flow_20d", "TQQQ(나스닥 3배) 20일 순유입의 250일 백분위",
                      "ProShares D 저녁 → D+1 시가"),
    "sqqq_aum_chg20_p": ("날짜", "us_daily_panel SQQQ_aum", "SQQQ 순자산 20거래일 변화율의 250일 백분위(휴장 빈 줄 뺀 뒤)",
                         "ProShares D 저녁 → D+1 시가"),
    "nvdl_top10": ("날짜", "us_daily_panel NVDL_flow_20d", "nvdl_flow20_p ≥ 0.90(개인 레버리지 유입 상위 10%, leverage_flood 후보)",
                   "D 저녁 → D+1 시가"),
    "upro_top10": ("날짜", "us_daily_panel UPRO_flow_20d", "upro_flow20_p ≥ 0.90", "D 저녁 → D+1 시가"),
    # ── 옵션·헤지펀드 대용(날짜 고정) ──
    "pc_equity5_p": ("날짜", "us_daily_panel pc_equity_5d", "CBOE 주식 풋/콜 5일 평균의 250일 백분위",
                     "CBOE D 저녁(가정) → D+1 시가"),
    "gvip_rel20_p": ("날짜", "us_daily_panel GVIP_SPY", "GVIP/SPY 20거래일 변화율의 250일 백분위(휴장 빈 줄 뺀 뒤 다시 계산)",
                     "종가 D 16시 → D+1 시가"),
    "dix_p": ("날짜", "hf_options squeeze dix", "DIX(다크풀 공매도 지수)의 250일 백분위", "D 17:56~19:55 ET → D+1 시가"),
    "gex_p": ("날짜", "hf_options squeeze gex_bn", "GEX(딜러 감마, 10억 달러)의 250일 백분위", "D 저녁 → D+1 시가"),
    "gex_neg": ("날짜", "hf_options squeeze gex_bn", "GEX < 0(딜러 숏감마)", "D 저녁 → D+1 시가"),
    # ── ETF 일별 공매도 거래량 비율(날짜 고정) ──
    "soxl_shr5_p": ("날짜", "hf_options finra_shvol shvr_soxl", "SOXL 공매도 거래량 비율 5일 평균의 250일 백분위",
                    "FINRA D 17시대 게시 → D+1 시가"),
    "tqqq_shr5_p": ("날짜", "finra_shvol shvr_tqqq", "TQQQ 공매도 거래량 비율 5일 평균의 250일 백분위", "D 저녁 → D+1 시가"),
    "spy_shr5_p": ("날짜", "finra_shvol shvr_spy", "SPY 공매도 거래량 비율 5일 평균의 250일 백분위", "D 저녁 → D+1 시가"),
    "smh_shr5_p": ("날짜", "finra_shvol shvr_smh", "SMH 공매도 거래량 비율 5일 평균의 250일 백분위", "D 저녁 → D+1 시가"),
    # ── CFTC TFF(날짜 고정, 주간) ──
    "cftc_nq_lev_p": ("날짜", "hf_options cftc_tff", "나스닥100 선물 레버리지 펀드 순포지션÷미결제의 52주 백분위",
                      "화 측정 → 금 15:30 ET 공개(known) → 다음 거래일 시가"),
    "cftc_nq_lev_chg4_p": ("날짜", "cftc_tff", "위 비율의 4주 변화의 52주 백분위", "known 다음 거래일 시가"),
    "cftc_es_lev_p": ("날짜", "cftc_tff", "S&P500 선물 레버리지 펀드 순포지션÷미결제의 52주 백분위", "known 다음 거래일 시가"),
    # ── 종목별 공매도 거래량·잔고 ──
    "shvr_z20": ("종목", "hf_options finra_shvol_by_symbol", "그 종목 공매도 거래량 비율(short/total) D값의 직전 20개 대비 z값(최소 15개)",
                 "FINRA D 저녁 → D+1 시가"),
    "shvr5_chg": ("종목", "finra_shvol_by_symbol", "비율 5일 평균 − 그 앞 20개 평균(최소 15개)", "D 저녁 → D+1 시가"),
    "shvr_spike": ("종목", "finra_shvol_by_symbol", "shvr_z20 ≥ 2(종목별 숏볼륨 비율 급증)", "D 저녁 → D+1 시가"),
    "si_chg": ("종목", "hf_options finra_si_by_symbol chg", "가장 최근 공개 공매도 잔고의 직전 대비 변화율(분할 보정)",
               "기준일 → 공개일 다음 거래일(known) → 그 다음 거래일 시가"),
    "si_dtc": ("종목", "finra_si_by_symbol days_to_cover", "같은 보고의 상환 일수(잔고÷평균 거래량)", "known 다음 거래일 시가"),
}
FEAT_NAMES = list(FEATS)
HYP_FLAGS = ["shvr_spike", "gex_neg", "nvdl_top10", "upro_top10"]


# ───────────────────────────── 자료 ─────────────────────────────
def load_sources():
    pan = pd.read_csv(LF / "us_daily_panel.csv", parse_dates=["date"])
    shv = pd.read_parquet(HO / "finra_shvol.parquet").reset_index()
    shb = pd.read_parquet(HO / "finra_shvol_by_symbol.parquet")
    si = pd.read_parquet(HO / "finra_si_by_symbol.parquet")
    cf = pd.read_parquet(HO / "cftc_tff.parquet").reset_index()
    sq = pd.read_parquet(HO / "squeeze.parquet").reset_index()
    for d, cols in ((shv, ("known", "asof")), (shb, ("known", "asof")), (si, ("known", "asof")),
                    (cf, ("known", "asof")), (sq, ("known", "asof"))):
        for c in cols:
            d[c] = pd.to_datetime(d[c])
    # FINRA: 다음 날 9:30 ET 전 게시(open_ok=1)면 when = asof, 아니면 known
    shv["when"] = np.where(shv["open_ok"].to_numpy() == 1, shv["asof"].to_numpy(), shv["known"].to_numpy())
    ok = shv.set_index("asof")["when"]
    shb["when"] = shb["asof"].map(ok)
    shb["when"] = shb["when"].fillna(shb["known"])       # 결과 표에서 빠진 늦은 수정일은 종목표에도 없다(QA)
    pan["when"] = pan["date"]
    sq["when"] = sq["asof"]
    cf["when"] = cf["known"]
    si["when"] = si["known"]
    return {"pan": pan, "shv": shv, "shb": shb, "si": si, "cf": cf, "sq": sq}


def pct_rank(s, win, minp):
    """지난 win개(자기 포함) 안 순위 백분위: (앞 값 < x) + 0.5·(앞 값 = x), 0~1. 앞 값만 쓴다."""
    s = s.dropna()

    def f(w):
        w = w[~np.isnan(w)]
        if len(w) < minp:
            return np.nan
        return (w[:-1] < w[-1]).mean() + 0.5 * (w[:-1] == w[-1]).mean()
    return s.rolling(win, min_periods=minp).apply(f, raw=True)


def place(vals, when, cal, age):
    """값을 when 다음 거래일(진입 가능 첫날)에 놓고 age거래일까지 이어 쓴다. 반환 (값 배열, 원자료 when 배열)."""
    d = pd.DataFrame({"v": np.asarray(vals, float), "w": pd.to_datetime(np.asarray(when))}).dropna()
    d = d.sort_values("w", kind="stable")
    pos = cal.searchsorted(d["w"].to_numpy(), side="right")
    d["pos"] = pos
    d = d[d["pos"] < len(cal)].drop_duplicates("pos", keep="last")
    v = pd.Series(np.nan, index=range(len(cal)))
    w = pd.Series(pd.NaT, index=range(len(cal)), dtype="datetime64[ns]")
    v.loc[d["pos"].to_numpy()] = d["v"].to_numpy()
    w.loc[d["pos"].to_numpy()] = d["w"].to_numpy()
    filled = v.notna()
    grp = filled.cumsum()
    run = v.groupby(grp).cumcount()            # 마지막 값 뒤 경과 일수
    keep = (grp > 0) & (run <= age)
    vv = v.groupby(grp).transform("first").where(keep)
    ww = w.groupby(grp).transform("first").where(keep)
    return vv.to_numpy(), ww.to_numpy()


def _series(df, col):
    x = df[["when", col]].dropna().sort_values("when")
    return x.set_index("when")[col].astype(float)


def date_features(src, cal):
    """진입일 색인(cal 위치) × 날짜 고정 특징, 그리고 특징별 원자료 when."""
    pan, shv, cf, sq = src["pan"], src["shv"], src["cf"], src["sq"]
    vals, whens = {}, {}

    def put(name, s, age, win, minp, transform="pct"):
        p = pct_rank(s, win, minp) if transform == "pct" else s
        v, w = place(p.to_numpy(), p.index, cal, age)
        vals[name], whens[name] = v, w

    for nm, col in (("nvdl_flow20_p", "NVDL_flow_20d"), ("upro_flow20_p", "UPRO_flow_20d"),
                    ("tqqq_flow20_p", "TQQQ_flow_20d"), ("pc_equity5_p", "pc_equity_5d")):
        put(nm, _series(pan, col), DAILY_AGE, 250, 60)
    a = _series(pan, "SQQQ_aum")
    put("sqqq_aum_chg20_p", a / a.shift(20) - 1, DAILY_AGE, 250, 60)
    g = _series(pan, "GVIP_SPY")
    put("gvip_rel20_p", g / g.shift(20) - 1, DAILY_AGE, 250, 60)
    vals["nvdl_top10"] = np.where(np.isnan(vals["nvdl_flow20_p"]), np.nan, (vals["nvdl_flow20_p"] >= 0.90).astype(float))
    whens["nvdl_top10"] = whens["nvdl_flow20_p"]
    vals["upro_top10"] = np.where(np.isnan(vals["upro_flow20_p"]), np.nan, (vals["upro_flow20_p"] >= 0.90).astype(float))
    whens["upro_top10"] = whens["upro_flow20_p"]
    # squeeze
    put("dix_p", _series(sq, "dix"), DAILY_AGE, 250, 60)
    gx = _series(sq, "gex_bn")
    put("gex_p", gx, DAILY_AGE, 250, 60)
    put("gex_neg", (gx < 0).astype(float), DAILY_AGE, 0, 0, transform="raw")
    # FINRA ETF 공매도 거래량 비율 5일 평균(asof 순서 5행)
    sv = shv.sort_values("asof")
    for nm, col in (("soxl_shr5_p", "shvr_soxl"), ("tqqq_shr5_p", "shvr_tqqq"), ("spy_shr5_p", "shvr_spy"),
                    ("smh_shr5_p", "shvr_smh")):
        r5 = sv[col].rolling(5).mean()
        s = pd.Series(r5.to_numpy(), index=pd.to_datetime(sv["when"].to_numpy()))
        put(nm, s, DAILY_AGE, 250, 60)
    # CFTC
    c = cf.sort_values("known")
    nq = pd.Series((c["nq_lev_net"] / c["nq_oi"]).to_numpy(), index=c["when"].to_numpy())
    es = pd.Series((c["es_lev_net"] / c["es_oi"]).to_numpy(), index=c["when"].to_numpy())
    put("cftc_nq_lev_p", nq, WEEKLY_AGE, 52, 26)
    put("cftc_nq_lev_chg4_p", nq.diff(4), WEEKLY_AGE, 52, 26)
    put("cftc_es_lev_p", es, WEEKLY_AGE, 52, 26)
    return pd.DataFrame(vals), pd.DataFrame(whens)


def stock_features(src, cal, syms):
    """{sym: (DataFrame 값, DataFrame when)} 진입일 색인. 종목별 공매도 거래량 비율·공매도 잔고."""
    shb, si = src["shb"], src["si"]
    out = {}
    for s in syms:
        vals, whens = {}, {}
        g = shb[shb["symbol"] == s].sort_values("asof")
        if len(g):
            r = (g["short"] / g["total"].where(g["total"] > 0)).to_numpy()
            r = pd.Series(r)
            prev = r.shift(1).rolling(20, min_periods=15)
            z = (r - prev.mean()) / prev.std()
            z[~np.isfinite(z)] = np.nan
            chg5 = r.rolling(5, min_periods=5).mean() - r.shift(5).rolling(20, min_periods=15).mean()
            for nm, x in (("shvr_z20", z), ("shvr5_chg", chg5)):
                v, w = place(x.to_numpy(), g["when"].to_numpy(), cal, DAILY_AGE)
                vals[nm], whens[nm] = v, w
            vals["shvr_spike"] = np.where(np.isnan(vals["shvr_z20"]), np.nan, (vals["shvr_z20"] >= 2).astype(float))
            whens["shvr_spike"] = whens["shvr_z20"]
        q = si[si["symbol"] == s].sort_values("known")
        if len(q):
            for nm, col in (("si_chg", "chg"), ("si_dtc", "days_to_cover")):
                v, w = place(q[col].to_numpy(), q["when"].to_numpy(), cal, BIMONTH_AGE)
                vals[nm], whens[nm] = v, w
        out[s] = (pd.DataFrame(vals, index=range(len(cal))), pd.DataFrame(whens, index=range(len(cal))))
    return out


def attach(ev, src=None, cal=None, lag=0):
    """사건표에 특징을 붙인다. 사건 t(인덱스 i)의 특징 = 진입일 i+1+lag에 놓인 값(lag=1이면 하루 늦춤, 민감도용)."""
    src = src or load_sources()
    cal = cal if cal is not None else H.load_panel().cal
    dv, dw = date_features(src, cal)
    sf = stock_features(src, cal, sorted(ev["sym"].unique()))
    ev = ev.copy()
    e = ev["i"].to_numpy() + 1 + lag
    okp = e < len(cal)
    ec = np.where(okp, e, 0)
    for f in FEAT_NAMES:
        ev[f] = np.nan
        ev[f"w_{f}"] = pd.NaT
    for f in dv.columns:
        ev[f] = np.where(okp, dv[f].to_numpy()[ec], np.nan)
        ev[f"w_{f}"] = np.where(okp, dw[f].to_numpy()[ec], np.datetime64("NaT"))
    for s, idx in ev.groupby("sym").groups.items():
        v, w = sf[s]
        ii = ec[ev.index.get_indexer(idx)]
        okk = okp[ev.index.get_indexer(idx)]
        for f in v.columns:
            ev.loc[idx, f] = np.where(okk, v[f].to_numpy()[ii], np.nan)
            ev.loc[idx, f"w_{f}"] = np.where(okk, w[f].to_numpy()[ii], np.datetime64("NaT"))
    for f in FEAT_NAMES:
        ev[f"w_{f}"] = pd.to_datetime(ev[f"w_{f}"])
    return ev


def check_timing(ev, cal, lag=0):
    """모든 특징 값의 원자료 when < 진입일(t+1) 인지. 위반 수 반환."""
    entry = pd.Series(cal[np.minimum(ev["i"].to_numpy() + 1 + lag, len(cal) - 1)], index=ev.index)
    bad = {}
    for f in FEAT_NAMES:
        w = ev[f"w_{f}"]
        m = ev[f].notna()
        bad[f] = int(((w[m] >= entry[m]) | w[m].isna()).sum())
    return bad


# ───────────────────────────── 범위 점검 ─────────────────────────────
def coverage(ev):
    rows = []
    for f in FEAT_NAMES:
        r = {"feature": f, "level": FEATS[f][0]}
        for pn, p in (("front", H.FRONT), ("back", H.BACK), ("aux2025", H.AUX)):
            m = H.in_period(ev, p) & ev["short_ret_h10"].notna()
            r[f"{pn}_nonnan"] = float(ev.loc[m, f].notna().mean()) if m.sum() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_features():
    ev = H.load_events_all()
    cal = H.load_panel().cal
    ev = attach(ev, cal=cal)
    bad = check_timing(ev, cal)
    print("시점 위반:", {k: v for k, v in bad.items() if v} or "0")
    print(coverage(ev).to_string())
    return ev


def cmd_audit(n_pick=6, seed=11):
    """미래 자료 교란: 기준일 c 뒤에 when이 있는 원자료 행만 무작위로 바꾸고 다시 붙였을 때, t ≤ c 사건의 특징이
    그대로인지 본다(검출력 확인: t > c 사건은 바뀌어야 한다)."""
    rng = np.random.default_rng(seed)
    cal = H.load_panel().cal
    ev = H.load_events_all()
    base_src = load_sources()
    base = attach(ev, src=base_src, cal=cal)
    dates = sorted(ev.loc[H.in_period(ev, ("2025-01-01", "2026-09-29")), "t"].unique())
    picks = sorted(rng.choice(dates, n_pick, replace=False))
    res = []
    for c in picks:
        c = pd.Timestamp(c)
        src = {k: v.copy() for k, v in base_src.items()}
        for k, d in src.items():
            fut = d["when"] > c
            num = [x for x in d.columns if d[x].dtype.kind == "f"]
            noise = rng.uniform(0.6, 1.4, size=(int(fut.sum()), len(num)))
            if k == "sq":
                noise[:, num.index("gex_bn")] *= rng.choice([-1, 1], size=int(fut.sum()))
                noise[:, num.index("gex")] = noise[:, num.index("gex_bn")]
            d.loc[fut, num] = d.loc[fut, num].to_numpy() * noise
        pert = attach(ev, src=src, cal=cal)
        past = base["t"] <= c
        diff_past = 0
        diff_fut = 0
        for f in FEAT_NAMES:
            a, b = base[f].to_numpy(), pert[f].to_numpy()
            neq = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            diff_past += int(neq[past.to_numpy()].sum())
            diff_fut += int(neq[~past.to_numpy()].sum())
        res.append({"cutoff": str(c.date()), "past_cells_changed": diff_past, "future_cells_changed": diff_fut,
                    "past_events": int(past.sum())})
        print(res[-1])
    OUT_AUDIT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ───────────────────────────── 판정 ─────────────────────────────
def parse_cell(name):
    nkey, rest = name.split("|", 1)
    N = int(nkey[1:])
    for f in sorted(FEAT_NAMES, key=len, reverse=True):
        if rest.startswith(f):
            tail = rest[len(f):]
            if tail.startswith("="):
                direction = "=1(켜짐)" if "(hi)" in tail else "=0(꺼짐)"
                thr = tail[1:tail.index("(")]
            elif tail.startswith("≥"):
                direction, thr = "상위3분위(≥)", tail[1:tail.index("(")]
            else:
                direction, thr = "하위3분위(≤)", tail[1:tail.index("(")]
            return N, f, direction, thr
    raise ValueError(name)


def cell_definition(name):
    N, f, d, thr = parse_cell(name)
    lvl, srcn, defi, when = FEATS[f]
    if d.startswith("="):
        cond = f"{f} {'= 1' if '켜짐' in d else '= 0'}"
    else:
        cond = f"{f} {'≥' if d.startswith('상위') else '≤'} {thr}"
    return f"N={N} 박스 이탈 사건 중 {cond} (문턱은 2026-01~05 사건 3분위). {f}: {defi} [{srcn}]", when


def hyp_table(ev):
    """가설 깃발 4개: N·구간별 켜짐/꺼짐 공매도 손익, 날짜 묶음 t, 진짜·가짜 비율(설명용)."""
    rows = []
    for f in HYP_FLAGS + ["nvdl_top10_lag1", "upro_top10_lag1"]:
        for N in H.NS:
            for pn, p in (("앞", H.FRONT), ("뒤", H.BACK), ("2025", H.AUX)):
                g = ev[(ev["N"] == N) & H.in_period(ev, p) & ev["short_ret_h10"].notna() & ev[f].notna()]
                if g.empty:
                    continue
                M = g[f].to_numpy(float)
                y = g["short_ret_h10"].to_numpy(float)
                gc = H._codes(g["t"].to_numpy())
                n1, n0, m1, m0, d, t = H.cluster_diff(M, y, gc)
                on = g[f] == 1
                c_on, c_off = g.loc[on, "cls"].dropna(), g.loc[~on, "cls"].dropna()
                rows.append({"flag": f, "N": N, "period": pn, "n_on": int(n1[0]), "dates_on": int(g.loc[on, "t"].nunique()),
                             "n_off": int(n0[0]), "mean_on": m1[0], "mean_off": m0[0], "diff": d[0], "t": t[0],
                             "true_on": (c_on == "진짜").mean() if len(c_on) else np.nan,
                             "true_off": (c_off == "진짜").mean() if len(c_off) else np.nan,
                             "fake_on": (c_on == "가짜").mean() if len(c_on) else np.nan,
                             "fake_off": (c_off == "가짜").mean() if len(c_off) else np.nan})
    return pd.DataFrame(rows)


def cmd_run():
    cal = H.load_panel().cal
    ev0 = H.load_events_all()
    src = load_sources()
    ev = attach(ev0, src=src, cal=cal)
    bad = check_timing(ev, cal)
    assert sum(bad.values()) == 0, bad
    ev1 = attach(ev0, src=src, cal=cal, lag=1)
    ev["nvdl_top10_lag1"] = ev1["nvdl_top10"].to_numpy()
    ev["upro_top10_lag1"] = ev1["upro_top10"].to_numpy()
    keep = [c for c in ev.columns if not c.startswith("w_")]
    ev[keep].to_parquet(OUT_EV)
    cells = H.tercile_cells(ev, FEAT_NAMES, by=("N",))
    tab, summ = H.evaluate_cells(ev, cells)
    meta = [parse_cell(n) for n in tab["cell"]]
    tab.insert(1, "N", [m[0] for m in meta])
    tab.insert(2, "feature", [m[1] for m in meta])
    tab.insert(3, "direction", [m[2] for m in meta])
    tab.insert(4, "threshold", [m[3] for m in meta])
    tab.insert(5, "level", [FEATS[m[1]][0] for m in meta])
    tab["definition"] = [cell_definition(n)[0] for n in tab["cell"]]
    tab["known_when"] = [cell_definition(n)[1] for n in tab["cell"]]
    tab.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    hyp = hyp_table(ev)
    cov = coverage(ev)
    # 칸 수 정확히
    per_feat = tab.groupby("feature").size()
    summ_out = {k: v for k, v in summ.items() if k not in ("null", "null_shift")}
    for k in ("null", "null_shift"):
        if k in summ:
            summ_out[k] = {kk: vv for kk, vv in summ[k].items() if kk != "dist"}
    summ_out["cells_per_feature"] = per_feat.to_dict()
    summ_out["timing_violations"] = bad
    summ_out["events_by_N"] = {
        str(N): {pn: {"events": int((H.in_period(ev, p) & (ev["N"] == N) & ev["short_ret_h10"].notna()).sum()),
                      "dates": int(ev.loc[H.in_period(ev, p) & (ev["N"] == N) & ev["short_ret_h10"].notna(), "t"].nunique())}
                 for pn, p in (("front", H.FRONT), ("back", H.BACK), ("aux", H.AUX))} for N in H.NS}
    OUT_JSON.write_text(json.dumps(summ_out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(tab, summ, hyp, cov, ev)
    return tab, summ, hyp, cov


def _on_rows(ev, r, period, off=False):
    """칸 r(표 행 dict)의 구간 사건 중 켜진(off=True면 꺼진) 행. 표지는 tercile_cells가 만든 그 칸 그대로."""
    x = ev[(ev["N"] == r["N"]) & H.in_period(ev, period) & ev["short_ret_h10"].notna()]
    cells = dict(H.tercile_cells(ev, [r["feature"]], by=("N",)))
    lab = pd.Series(cells[r["cell"]]).reindex(x.index)
    return x[lab == (0 if off else 1)]


CONCLUSION = [
    "## 결론 (표를 본 뒤 씀)", "",
    "**수급·레버리지·옵션 자료로 박스 하단 이탈의 '진짜'를 미리 가려내는 규칙은 확인되지 않았다.**", "",
    "- 138칸 중 BRIEF 통과는 1칸이다. 같은 고르기·확인을 날짜 묶음째 섞어 200번 되풀이하면 평균 0.125칸이 통과한다(p 0.12). "
    "원형 이동 귀무도 평균 0.15칸(p 0.13)이고, 원형 이동 p까지 걸면 통과 0칸이다. 1칸은 우연과 구별되지 않는다.",
    "- 통과한 `N20|gvip_rel20_p ≥ 0.538`(헤지펀드 인기주 GVIP가 지난 20일 SPY보다 강했던 날의 20일 박스 이탈)은 규칙으로 쓸 수 없다.",
    "  - 뒤 구간 켜진 56건이 모두 6/1~7/7의 11일(episode 34 앞부분)에 있다. 꺼진 쪽은 6/4~6/10 4일을 빼면 7/13 이후이고, "
    "7/16~7/29 반등(그날 공매도 평균 −3~−13%)을 비롯해 대부분 손실이다. 결국 '6월~7월 초 대 그 뒤'라는 시간 구분이다.",
    "  - 원형 이동 p 0.058(보조 기준 탈락). N40은 +2.2%p(t 1.13), N60은 −0.5%p로 다른 박스 길이에서 되풀이되지 않는다. "
    "2025는 −0.3%p로 반대다.",
    "  - 켜진 사건의 공매도 손익도 평균 +1.13%(비용 0.10% 뒤, 대차 비용 전)로 얇다.",
    "- 상위 10칸 중 8칸이 날짜 고정 특징(시장 전체 하나의 값)이다. 이런 칸은 '언제'만 가르고 '어느 종목'은 가르지 못한다.", "",
    "### 물어본 세 신호(N = 20·40·60 범위, 차이 = 켜짐 − 꺼짐 공매도 손익)", "",
    "| 신호 | 2026 앞(1~5월) | 2026 뒤(6~9월) | 2025 | 읽기 |",
    "|---|---|---|---|---|",
    "| 종목 숏볼륨 비율 급증(직전 20일 대비 z ≥ 2) | +3.9~+9.7%p, N마다 4~5건/3~4일 | −3.9~+6.2%p, N마다 3건 | +1.8~+5.8%p | "
    "이탈일에 급증하는 일이 드물어(고르기 기준 15건 미달) 판단 불가 |",
    "| 숏볼륨 비율 상대 상승(z 상위 3분위) | +1.0~+2.6%p | +1.0~+3.5%p(t 0.5~1.5, N60 순열 p 0.13) | −0.3~+0.9%p | "
    "2026 두 구간 모두 같은 방향이지만 유의하지 않다 |",
    "| GEX 음수(딜러 숏감마) | −1.0~+0.9%p(3월 8일) | 6/26 하루(9건 이하) | −6.5~−7.9%p(t −4.4~−5.5, 5~6일) | "
    "진짜로 만들지 않는다. 2025-03~04엔 오히려 공매도 손실 |",
    "| 개인 레버리지 유입 상위 10%(NVDL 20일 순유입) | +1.4~+4.1%p(N20 t 3.8, 6일) | +0.2~+7.6%p(N20 t 0.1) | "
    "+7.5~+10.0%p(2~3일) | 확인 안 됨. 하루 늦추면 앞 구간 N20이 +0.5%p로 사라진다 |",
    "| 같은 무리(UPRO 20일 순유입 상위 10%) | +2.3~+3.0%p(2~4일) | 0.0~+2.3%p | −4.6~−5.6%p(t −2.9~−3.4, 14~15일) | "
    "확인 안 됨. 2025엔 반대 |", "",
    "- **'더 빠지는 이탈'과 '공매도로 버는 이탈'은 다르다.** 2025년 GEX 음수 날의 이탈은 '진짜'(10일 안에 L − 2·ATR 먼저 닿음) "
    "비율이 73~75%로 꺼진 날(31~35%)의 두 배였다. 그런데도 공매도는 평균 −7.6~−8.1% 손실이었다. 더 깊이 판 뒤 10일 안에 박스 안으로 "
    "V자로 돌아와 되돌림 손절이 진입가보다 위에서 걸렸기 때문이다. UPRO 유입 상위 10%(2025-04 저가 매수)도 같은 모습이다(진짜 55~62% 대 30~34%, 손익 −5.3~−5.8%).",
    "- 지금까지의 박스 이탈 연구와 같은 결론이다. 2026년 이탈은 평균적으로 공매도에 불리했고(기준선 종목 h10 −3.0~−6.0%), "
    "수급 자료가 그 가운데 '진짜'를 앞서 골라 주지 못했다.", "",
]

LIMITS = [
    "## 한계", "",
    "- **공개 시각 가정.** NVDL(GraniteShares)과 CBOE 풋/콜은 '측정일 저녁 공개'라는 직접 근거가 이 연구에 없다. leverage_flood·"
    "hf_options timing_audit와 같은 가정이다. NVDL·UPRO 깃발은 하루 늦춘 값(`*_lag1`)도 가설 표에 넣었다.",
    "- **FINRA 늦게 고친 파일 3일**(2023-02-13, 2025-08-20, 2026-02-05)은 hf_options 규칙대로 뺐다. 그래서 leverage_flood 패널의 "
    "`*_short_ratio_5d`와 8행(두 날 뒤 4거래일씩)에서 값이 다르다(최대 0.054, 나머지 432행은 같음). 이 갈래는 2025 백분위에 "
    "이력이 더 필요해 hf_options 원자료(2023-01부터)로 다시 계산했다.",
    "- GVIP/SPY 20일 변화는 패널 칸(휴장 빈 줄이 낀 pct_change)을 쓰지 않고 휴장 줄을 뺀 뒤 다시 계산했다.",
    "- CFTC는 2025-10~12 셧다운 구간 값이 15거래일 넘게 묵어 비웠다(2025 사건의 12%가 빈칸).",
    "- 종목 숏볼륨은 FINRA 장외(TRF) 정규장 거래만이다. 높은 비율은 대부분 고객 매수를 받아 주는 시장조성자 매도라 '약세 베팅'이 "
    "아니다(qa_finra_shvol). IWM·XLK·IGV 사건은 종목별 숏볼륨·잔고가 없어 해당 칸 밖이고, SPY는 잔고 자료가 없다.",
    "- 뒤 구간 사건의 97~100%가 episode 34·35에 몰려 있다. 날짜 고정 특징의 독립 관측은 사실상 두 국면이다.",
    "- **선택편향.** 같은 자료로 레버리지 홍수(422칸)·헤지펀드×옵션(약 1.7만 칸)·옵션 심리(352칸) 연구를 본 뒤의 탐색이다. 이 갈래의 "
    "23특징은 결과를 보기 전에 정했지만(ETF 숏볼륨은 SOXL·TQQQ·SPY·SMH 4개: SOXX는 SMH와 겹치고 NVDA·MU는 종목별 특징이 덮음, "
    "SQQQ 순자산은 leverage_flood 후보인 20일 변화, CFTC는 레버리지 펀드 3칸), 자료 고르기 자체가 앞선 결과에 기대고 있다.",
    "- 대차 비용과 배당 지급은 빠졌다(하네스 한계). 결과 뒤 정의 변경은 없다.", "",
]


def _p(x, nd=2):
    return "못 구함" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{nd}f}%"


def _s(x):
    return "못 구함" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.0%}"


def _f(x, nd=2):
    return "못 구함" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_md(tab, summ, hyp, cov, ev):
    sel = tab[tab["selected"]].sort_values("rank")
    L = ["# 갈래 4: 수급·레버리지·옵션 — 박스 하단 이탈을 '진짜'로 만드나 (2026-10-14)", "",
         "코드 `code/b4_flow.py`(`py b4_flow.py run`, 약 2.5분), 칸 전체 `results/b4_flow_cells.csv`, 사건별 특징 "
         "`results/b4_flow_events.parquet`, 요약 `results/b4_flow_summary.json`, 시점 교란 시험 `results/b4_flow_audit.json`. "
         "판정은 `harness.evaluate_cells` 그대로(BRIEF). 아래 '결론'과 '한계'는 표를 본 뒤 쓴 글이고, 나머지 표는 코드가 만든다.", ""]
    L += CONCLUSION
    L += ["## 판정 요약", "",
          f"- 시험한 칸: **{summ['n_cells']}칸** = 특징 {len(FEAT_NAMES)}개 × 방향 2(3분위 위·아래, 두 값 특징은 =1·=0) × N 3. "
          "앞 구간에서 값이 하나뿐인 특징은 없어 빠진 칸이 없다. "
          f"고르기 조건(앞 차이 > 0, 켜짐 ≥ 15건, 켜짐 날짜 ≥ 8일)을 만족한 칸 {summ['n_eligible']}개, 상위 {summ['n_selected']}칸 확인.",
          f"- 통과(BRIEF): **{summ['n_pass']}칸** {summ['passed']}. 원형 이동 p까지(보조): {summ['n_pass_shift']}칸.",
          f"- 선택편향 귀무(날짜 묶음 i.i.d. 200번): 평균 {summ['null']['mean']:.3f}칸, 95% 분위 {summ['null']['q95']:.0f}, "
          f"p = {summ['null']['p_value']:.3f}. 원형 이동 귀무: 평균 {summ['null_shift']['mean']:.3f}칸, p = {summ['null_shift']['p_value']:.3f}.",
          f"- 사건(세 N 합, h10 있는 것): 앞 {summ['events']['f']}건/{summ['dates']['f']}일, 뒤 {summ['events']['b']}건/{summ['dates']['b']}일, "
          f"2025 {summ['events']['a']}건/{summ['dates']['a']}일.", ""]
    L += ["## 앞 구간 상위 10칸(통과 여부와 무관)", "",
          "| 순위 | 칸 | 수준 | 앞 차이 | 앞 켜짐 n/일 | 뒤 차이 | 뒤 t | 뒤 순열 p | 원형 이동 p | 뒤 켜짐 평균 | 뒤 켜짐 n/일 | 뒤 초과 차이 | 2025 차이 | 한 국면 최대 비중 | 통과 |",
          "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sel.to_dict("records"):
        ep = "" if not np.isfinite(r["regime_top_ep"]) else f" (ep {int(r['regime_top_ep'])})"
        L.append(f"| {int(r['rank'])} | `{r['cell']}` | {r['level']} | {_p(r['f_diff'])} | {int(r['f_n_on'])}/{int(r['f_dates_on'])} | "
                 f"{_p(r['b_diff'])} | {_f(r['b_t'])} | {_f(r['b_p'], 3)} | {_f(r['b_p_shift'], 3)} | {_p(r['b_mean_on'])} | "
                 f"{int(r['b_n_on'])}/{int(r['b_dates_on'])} | {_p(r['b_ex_diff'])} | {_p(r['a_diff'])} | {_f(r['regime_top_share'])}{ep} | "
                 f"{'통과' if r['pass'] else '아님'} |")
    L += ["", "차이 = 켜진 사건 평균 공매도 손익(h10, 비용 0.10% 뒤) − 꺼진 사건. 뒤 초과 차이는 같은 날 같은 유형 평균을 뺀 손익의 "
          "차이다. 날짜 고정 특징은 같은 날 같은 N 사건이 모두 같은 표지라 정의상 종목을 가르지 못한다(초과 차이는 그날 사건이 1건이라 "
          "무작위 대조를 쓴 날에서만 생긴다). '한 국면 최대 비중'은 뒤 구간 켜진 사건 손익 합이 양수일 때만 하네스가 계산한다.", ""]
    L += ["### 상위 10칸의 뒤 구간 켜진 사건이 몰린 국면(사건 수 기준, 설명용)", "",
          "| 순위 | 칸 | 뒤 켜짐 사건 | 가장 많은 국면 | 그 비중 | 켜짐 사건 손익 합(%의 합) |", "|---:|---|---:|---:|---:|---:|"]
    epname = {34: "34(5/29~7/31)", 35: "35(8/11~9/16)", -1: "국면 밖"}
    for r in sel.to_dict("records"):
        x = _on_rows(ev, r, H.BACK)
        vc = x["episode"].value_counts(normalize=True)
        top = epname.get(int(vc.index[0]), str(int(vc.index[0]))) if len(vc) else "-"
        L.append(f"| {int(r['rank'])} | `{r['cell']}` | {len(x)} | {top} | {vc.iloc[0]:.0%} | {_p(x['short_ret_h10'].sum())} |")
    L.append("")
    for r in tab[tab["pass"]].to_dict("records"):
        x = _on_rows(ev, r, H.BACK)
        off = _on_rows(ev, r, H.BACK, off=True)
        g = x.groupby("t").agg(n=("sym", "size"), m=("short_ret_h10", "mean"), ep=("episode", "first"))
        L += [f"### 통과 칸 `{r['cell']}`의 뒤 구간 켜진 날", "",
              "| 이탈일 | 사건 | 평균 공매도 손익 | 국면 |", "|---|---:|---:|---:|"]
        L += [f"| {d.date()} | {int(q.n)} | {_p(q.m)} | {int(q.ep)} |" for d, q in g.iterrows()]
        cl_on = x["cls"].value_counts(normalize=True)
        cl_off = off["cls"].value_counts(normalize=True)
        worst = off.groupby("t")["short_ret_h10"].agg(["size", "mean"]).sort_values("mean").head(5)
        L += ["", f"- 분류(켜짐/꺼짐): 진짜 {cl_on.get('진짜', 0):.0%}/{cl_off.get('진짜', 0):.0%}, "
              f"가짜 {cl_on.get('가짜', 0):.0%}/{cl_off.get('가짜', 0):.0%}, 중간 {cl_on.get('중간', 0):.0%}/{cl_off.get('중간', 0):.0%}.",
              "- 꺼진 쪽 손실이 큰 날: " + ", ".join(f"{d.date()} {int(q['size'])}건 {_p(q['mean'])}" for d, q in worst.iterrows()) + ".", ""]
    # 가설 깃발
    L += ["## 가설 깃발 4개(설명용, 판정은 위 칸 표)", "",
          "| 깃발 | N | 구간 | 켜짐 n/일 | 켜짐 평균 | 꺼짐 평균 | 차이 | 날짜 묶음 t | 진짜 켜짐/꺼짐 | 가짜 켜짐/꺼짐 |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in hyp.itertuples():
        L.append(f"| {r.flag} | {r.N} | {r.period} | {r.n_on}/{r.dates_on} | {_p(r.mean_on)} | {_p(r.mean_off)} | {_p(r.diff)} | "
                 f"{_f(r.t)} | {_s(r.true_on)}/{_s(r.true_off)} | {_s(r.fake_on)}/{_s(r.fake_off)} |")
    L += ["", "`*_lag1`는 NVDL·UPRO 좌수에 다음 날 결제분이 섞였을 가능성(leverage_flood 한계)에 대비해 하루 늦춘 값이다(칸 수에 넣지 않음).", ""]
    if OUT_AUDIT.exists():
        au = json.loads(OUT_AUDIT.read_text(encoding="utf-8"))
        L += ["## 시점 점검", "",
              "- 모든 특징 값에 원자료 when을 붙여 '진입일(t+1) 전'인지 확인했다. 위반 0건(23특징 전부).",
              "- 미래 자료 교란(`py b4_flow.py audit`): 기준일 c를 사건 날짜에서 무작위로 고르고, when > c인 원자료 행(흐름·풋/콜·가격 비율·"
              "DIX/GEX·FINRA·CFTC·잔고)의 숫자 칸 전부에 ×U(0.6, 1.4)(GEX는 부호도 무작위)를 곱해 다시 붙였다.", "",
              "| 기준일 c | t ≤ c 사건 | 바뀐 특징 값(t ≤ c) | 바뀐 특징 값(t > c, 검출력) |", "|---|---:|---:|---:|"]
        L += [f"| {a['cutoff']} | {a['past_events']} | {a['past_cells_changed']} | {a['future_cells_changed']} |" for a in au]
        L.append("")
    L += LIMITS
    # 칸 수
    L += ["## 칸 수(정확히)", "", "| 특징 | 수준 | 칸 수 | 정의 | 언제 알 수 있나 |", "|---|---|---:|---|---|"]
    cpf = tab.groupby("feature").size()
    for f in FEAT_NAMES:
        L.append(f"| {f} | {FEATS[f][0]} | {int(cpf.get(f, 0))} | {FEATS[f][2]} | {FEATS[f][3]} |")
    L += [f"| 합계 | | {len(tab)} | | |", ""]
    L += ["## 자료 범위(h10 있는 사건 중 특징 값이 있는 비율)", "", "| 특징 | 앞 | 뒤 | 2025 |", "|---|---:|---:|---:|"]
    for r in cov.itertuples():
        L.append(f"| {r.feature} | {r.front_nonnan:.0%} | {r.back_nonnan:.0%} | {r.aux2025_nonnan:.0%} |")
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"features": cmd_features, "audit": cmd_audit, "run": cmd_run}[cmd]()
