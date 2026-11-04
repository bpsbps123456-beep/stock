"""성적표 — 앞 연구에서 남은 후보 6개를 2026년 하나의 잣대로 나란히 잰다 (2026-11-04).

규칙: AGENTS.md 연구 보고 규칙(2026-11-04 갱신). 판정·순위는 2026-01-02~10-28 207일만 쓴다. 2025는 한 줄 보조이고,
2024년 이전 자료는 쓰지 않는다. 후보 정의는 원래 연구 코드 그대로 불러 쓴다(바꾸지 않는다).

후보(판 = 11시판 / 10시판)
  S1 F1-A 첫 1분봉 방향 돌파: fam_F1.make(F1A). 11시판 = Engine(scheme='actual')(2026 1/2~3/6 표준시 10:00, 3/9~ 11:00),
     10시판 = Engine(scheme='std').
  S2 SOXL 3% → SOXS(사용자 방향): ev4_events side='user', x=0.03 단독 사슬. 11시판 r_w90, 10시판 r_std(결정 ≤ 09:44 봉).
  S3 윗꼬리 거부 H4: S2 사건 중 c10_wick ≥ 0.5 − 1e-9(c3_hyp H4). 판은 S2와 같다.
  S4 극상방 다음 날 SOXS: 단계(D−1) = +2, stage5.Always(-1), Engine(scheme='std')(09:31 봉 종가 + 반 호가 → 09:59 봉 종가 − 반 호가).
     09:59에 끝나 두 판이 같은 매매다(한 줄).
  S5 빨강 첫날 SOXS: redfirst.flags R1 날, stage5.Always(-1), 11시판 Engine('w90')(10:59), 10시판 Engine('std')(09:59).
  S6 하방 다음 날 SOXL: 단계(D−1) = −1, stage5.Always(+1), 11시판 w90, 10시판 std.
  단계: stage5.zone_calc. S4·S6은 stage5 캐시(SOXX 2021-01-01~2026-10-28), S5는 redfirst 캐시(2001-01-01~2026-10-29) 그대로.

같은 잣대(후보·판마다, 단위 % — 1.0 = 1%)
  하루 손익 = 그날 매매 손익의 곱 − 1, 매매 없는 날 0. 하루 평균 = 2026 207일 평균.
  활동일, 매매 수, 이긴 매매 비율(손익 > 0), +1%·2R 목표가 있으면 도달률(why == 'target'), 매매당 평균, 최악 매매.
  날 묶음 부트스트랩 95%: 207일 복원 추출 10,000회, seed 20261104(계열마다 새 rng), 2.5·97.5 분위. 단측 p = 평균 ≤ 0 비율(참고).
  1~6월(123일)·7~10월(84일) 하루 평균, 비용 2배(반 호가 × 2) 하루 평균, 상위 2일(하루 손익 큰 두 날)을 빼고 205일 평균.
  최대 낙폭 = 하루 손익을 이은 자본(시작 1.0 포함)의 고점 대비 최저(%). 달마다 매매 수. 2025 하루 평균(같은 식, 250일).
  S2·S3 10시판 비용 2배 = h0.sim_trade(결정 봉, cut 10:00, cm=2)(ev4_build의 r_w90_c2와 같은 방식).
점검: S2·S3 사건 표 날별 곱 = h0.H0(sides=(-1,)) 엔진 단독 실행(w90·std), 단계 D−1 = stage5_daily_A.csv, S5 R1 = redfirst 2026 13건.

새 거래일(2026-10-29·10-30·11-02·11-03)
  Alpaca 1분봉 cache/alp1m_recent(fetch.py 2026-10-01 '2026-11-03 20:00' _recent; 무료 키는 15분 안 자료를 막아 끝을
  11-03 20:00 ET로 줌 = 11-04 00:00 ET와 같은 범위). yfinance SOXX 일봉 2001-01-01~2026-11-03을 새로 받아 단계를 잇는다
  (옛 캐시와 겹치는 날 단계 대조). S2·S3은 h0.H0(ref='low', thr=0.03, sides=(-1,)) 엔진 사슬(ev4 사슬과 비트 일치가 확인된
  정의)로 다시 만들고, 결정 봉 SOXL 윗꼬리를 ev4_build 식 (H − max(O, C)) ÷ (H − L)로 잰다.
  겹치는 10-01~10-28은 새 자료로 다시 돌려 성적표 계열과 같은지 확인한다. 실행판: 10-29·10-30 11시판, 11-02·11-03 10시판.
  덤: 11-03 종가 단계로 11-04 세션에 일봉 후보(S4·S5·S6)가 켜지는지.
실행: cd code && py scoreboard.py → ../results/scoreboard_2026.json, ../results/scoreboard_daily.csv
"""
import hashlib, json, os, sys, time

os.environ["SOS_UNLOCK"] = "frozen"          # 얼린 후보를 2025·2026에 한 번 잰다(잠금 구간 확인)
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, Engine, idx, hm, cutoff_idx  # noqa: E402
import fam_F1  # noqa: E402
import stage5 as S5  # noqa: E402
import redfirst as RF  # noqa: E402
import h0 as H0M  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
SEED, NB, EPS = 20261104, 10000, 1e-9
F1A = {"N": 1, "entry": "confirm", "tgt": 2, "stop": None, "rev": False, "filt": None}
X = 0.03
CUT_STD, STD_LAST_DEC = idx("10:00"), idx("10:00") - 16
H1 = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30"))
H2 = (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-31"))
RECENT = [pd.Timestamp(d) for d in ("2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03")]
T0 = time.time()
fnum = S5.fnum


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


HOW = {
    "S1": "사전 등록(BRIEF2) F1 시가 범위 돌파 묶음 300설정(5묶음 전체 1,484설정, 9개 얼림) 중 2026 1~6월(sel)로 이웃 t 중앙값 1위로 "
          "고른 설정. 1~6월은 고른 구간(표본 안), 7~10월만 표본 밖 확인 → 11시판(actual) '관찰', 10시판(std) '탈락'(report.md).",
    "S2": "사용자 원문 규칙(SOXL 장중 저점 +3% → SOXS, +1% 지정가, 손절 없음). 사전 등록 H0(BRIEF3)는 양방향이었고 이 한 방향 "
          "단독 실행은 사후 분할(면 2 × 판 2 = 4칸 중 하나, 판정 안 함, report3.md). 고른 매개변수 없음. 2025는 음(−).",
    "S3": "BRIEF4 C3 사전 등록 이론 가설 10개 중 H4(같은 연구에서 C1 자격 칸 9,676개·모형 6개 포함 2만 2천여 칸 시험). 2026 형식 통과"
          "(Holm 0.004) 뒤 감사 불인정(공정 귀무 보정 0.43), 2025는 음(−)(report4.md).",
    "S4": "BRIEF5 표 A 사후 단서: 2026 결과 24칸(단계 × 종목 × 마감 16칸 + 표 B 8칸) 중 가장 큰 값을 보고 고름(표본 안). "
          "5/27~6/5 8일이 칸 합의 103%(report5.md).",
    "S5": "BRIEF6 사전 등록 주 가설(R1 × O2, 사용자 아이디어 '빨강 첫날 숏'). 변형 4 × 청산 4 = 16칸, Holm m 4 → O2 0.133 불통과. "
          "아이디어가 2026 차트에서 나와 2026은 사실상 표본 안(report6.md). 10시판 = O1.",
    "S6": "BRIEF5 표 A 사후 단서: 같은 2026 24칸 중 둘째로 큰 값(하방 다음 날 늘 SOXL 11시 +0.303), 10/14~10/28 11일이 전부"
          "(report5.md). BRIEF7 되살리기 후보에는 없었음(새 자료 확인 안 됨).",
}
NAME = {"S1": "첫 1분봉 방향 돌파 F1-A", "S2": "SOXL 3% → SOXS(사용자 방향)", "S3": "윗꼬리 거부 H4",
        "S4": "극상방 다음 날 SOXS 09:31→09:59", "S5": "빨강 첫날 SOXS", "S6": "하방 다음 날 SOXL"}


# ---------------------------------------------------------------- 통계
def comp_by_day(day_idx, r, nd_index):
    """사건 손익(%)을 날 색인으로 곱해 묶는다. 반환: (하루 손익 %, 건수) — nd_index 순서."""
    s = pd.DataFrame({"d": day_idx, "r": r})
    g = s.groupby("d")["r"]
    comp = g.apply(lambda v: (np.prod(1 + v.values / 100) - 1) * 100)
    n = g.size()
    return comp.reindex(nd_index, fill_value=0.0).values.astype(float), n.reindex(nd_index, fill_value=0).values.astype(int)


def boot(v, seed=SEED):
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    return [fnum(np.percentile(bs, 2.5)), fnum(np.percentile(bs, 97.5))], fnum((bs <= 0).mean())


def max_dd(v):
    eq = np.r_[1.0, np.cumprod(1 + v / 100)]
    return fnum(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)


def stats(days, dly, n, dly_c2, trades, has_target, dly25):
    """days: 2026 날(Timestamp) 배열, dly·n·dly_c2: 하루 손익 %·건수·비용 2배, trades: 매매 손익 %(DataFrame: day, ret, why)."""
    days = pd.DatetimeIndex(days)
    h1 = (days >= H1[0]) & (days <= H1[1])
    h2 = (days >= H2[0]) & (days <= H2[1])
    ci, p = boot(dly)
    top = np.argsort(-dly, kind="stable")[:2]
    keep = np.ones(len(dly), bool)
    keep[top] = False
    r = trades["ret"].values
    months = trades["day"].dt.strftime("%Y-%m").value_counts().sort_index()
    allm = [f"2026-{m:02d}" for m in range(1, 11)]
    out = {"days_2026": int(len(dly)), "per_day": fnum(dly.mean()), "active_days": int((n > 0).sum()), "trades": int(n.sum()),
           "win_rate": fnum((r > 0).mean()) if len(r) else None,
           "target_hit": fnum((trades["why"] == "target").mean()) if (has_target and len(r)) else None,
           "per_trade": fnum(r.mean()) if len(r) else None, "median_trade": fnum(np.median(r)) if len(r) else None,
           "ci95": ci, "p_one_sided_boot": p,
           "t": fnum(dly.mean() / (dly.std(ddof=1) / np.sqrt(len(dly)))) if dly.std() > 0 else None,
           "h1_jan_jun": fnum(dly[h1].mean()), "h1_days": int(h1.sum()),
           "h2_jul_oct": fnum(dly[h2].mean()), "h2_days": int(h2.sum()),
           "cost2x": fnum(dly_c2.mean()),
           "drop_top2_days": fnum(dly[keep].mean()),
           "top2_days": [{"day": str(days[k].date()), "pnl": fnum(dly[k])} for k in top],
           "worst_trade": fnum(r.min()) if len(r) else None, "best_trade": fnum(r.max()) if len(r) else None,
           "worst_day": fnum(dly.min()), "max_dd": max_dd(dly),
           "monthly_signals": {m: int(months.get(m, 0)) for m in allm},
           "per_month": fnum(n.sum() / len(allm)),
           "y2025_per_day": fnum(dly25.mean()), "y2025_days": int(len(dly25))}
    return out


# ---------------------------------------------------------------- 단계
def stages_from(close_series, days):
    """stage5 방식: 날 D에는 yfinance 날짜열에서 D 바로 앞 거래일 종가의 단계."""
    dates = pd.DatetimeIndex(close_series.index)
    zn, r, z = S5.zone_calc(close_series.to_numpy(float))
    pos = np.searchsorted(dates.values, np.array(days, dtype="datetime64[ns]"), side="left") - 1
    return zn[pos], dates[pos], zn, r, z, dates


def yf_soxx_new():
    p = os.path.join(CACHE, "soxx_1d_yf_ohlc_20010101_20261103.csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True), "캐시 " + os.path.basename(p)
    import yfinance as yf
    df = yf.download("SOXX", start="2001-01-01", end="2026-11-04", auto_adjust=False, progress=False, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[df["Close"].notna()]
    assert df.index[-1] == pd.Timestamp("2026-11-03"), df.index[-1]
    df.to_csv(p)
    return df, "yfinance 새로 받음 2001-01-01~2026-11-03"


# ---------------------------------------------------------------- 엔진 계열
def always_runs(D, days_k):
    """늘 SOXL·늘 SOXS × w90·std × 비용 1·2배. 반환 {(pos, scheme, cm): 하루 손익 %}, 매매 표."""
    out, trs = {}, {}
    for pos in (1, -1):
        for sch in ("w90", "std"):
            for cm in (1.0, 2.0):
                daily, tr = Engine(D, scheme=sch, cost_mult=cm).run(lambda pos=pos: S5.Always(pos), days=days_k)
                assert (daily["n"] == 1).all()
                out[(pos, sch, cm)] = daily["ret"].values * 100
                trs[(pos, sch, cm)] = tr
    return out, trs


def wick_of(D, d, i):
    A = D.A["SOXL"]
    h_, o_, l_, c = A["High"][d, i], A["Open"][d, i], A["Low"][d, i], A["Close"][d, i]
    rng = h_ - l_
    return float((h_ - max(o_, c)) / rng) if rng > 0 else 0.0


def h0_chain(D, days_k, scheme):
    """h0.H0 사용자 방향 단독 사슬. 반환 매매 표(+ 결정 봉·윗꼬리)."""
    lg = []
    daily, tr = Engine(D, scheme=scheme).run(lambda: H0M.H0(ref="low", thr=X, sides=(-1,), log=lg), days=days_k)
    lgd = pd.DataFrame(lg)
    tr = tr.reset_index(drop=True)
    assert len(lgd) == len(tr)
    if len(tr):
        assert (tr["entry"].values == np.array([hm(f) for f in lgd["fill"]])).all()
        tr["d"] = lgd["d"].values
        tr["dec_i"] = lgd["dec"].values
        tr["dec_close"] = [hm(i + 1) for i in lgd["dec"]]
        tr["wick"] = [wick_of(D, int(d), int(i)) for d, i in zip(lgd["d"], lgd["dec"])]
    return daily, tr


def main():
    os.makedirs(RES, exist_ok=True)
    checks = {}
    D = Data("alp1m_long")
    days_all = [pd.Timestamp(d) for d in D.days]
    k26 = [k for k, d in enumerate(days_all) if d.year == 2026]
    k25 = [k for k, d in enumerate(days_all) if d.year == 2025]
    days26 = pd.DatetimeIndex([days_all[k] for k in k26])
    assert len(k26) == 207 and len(k25) == 250 and days26[-1] == pd.Timestamp("2026-10-28")
    K = k25 + k26
    log(f"Data alp1m_long 2025 {len(k25)}일, 2026 {len(k26)}일 (2024년 이전 날은 계산에 넣지 않음)")

    # ---- 단계(원래 캐시 그대로)
    c21 = S5.soxx_daily()                                      # stage5 캐시 2021-01-01~2026-10-28
    st_prev_all, prev_dates, *_ = stages_from(c21, [days_all[k] for k in K])
    st_prev = dict(zip(K, st_prev_all))
    A5 = pd.read_csv(os.path.join(RES, "stage5_daily_A.csv"), parse_dates=["day"]).set_index("day")
    a5 = A5["stage_prev"].reindex([days_all[k] for k in K]).values
    checks["stage_prev_vs_stage5_daily_A"] = {"n": len(K), "agree": int((a5 == st_prev_all).sum())}
    X01 = RF.yf_daily("SOXX", "2001-01-01")                   # redfirst 캐시 2001-01-01~2026-10-29
    znX, _, _ = S5.zone_calc(X01["Close"].to_numpy(float))
    datesX = pd.DatetimeIndex(X01.index)
    tX = {d: t for t, d in enumerate(datesX)}
    FX = RF.flags(znX)
    r1 = {k: bool(FX["R1"][tX[days_all[k]]]) for k in K}
    checks["S5_R1_2026_n"] = int(sum(r1[k] for k in k26))
    log("단계 대조", checks["stage_prev_vs_stage5_daily_A"], "R1 2026", checks["S5_R1_2026_n"])

    # ---- 엔진 계열
    AL, ALT = always_runs(D, K)
    pos_in = {k: j for j, k in enumerate(K)}
    j26 = np.array([pos_in[k] for k in k26])
    j25 = np.array([pos_in[k] for k in k25])

    series = {}       # (id, version) -> dict(d26, n26, c2_26, trades26, has_target, d25)

    def masked(mask_k, pos, sch):
        m = np.array([mask_k[k] for k in K], bool)
        d1 = np.where(m, AL[(pos, sch, 1.0)], 0.0)
        d2 = np.where(m, AL[(pos, sch, 2.0)], 0.0)
        n = m.astype(int)
        tr = ALT[(pos, sch, 1.0)].reset_index(drop=True)
        tr = tr[m].copy()
        tr["ret"] = tr["ret"] * 100
        tr["day"] = pd.to_datetime(tr["day"])
        return d1, d2, n, tr

    def put(cid, ver, d1, d2, n, tr, has_target):
        t26 = tr[tr["day"].dt.year == 2026]
        series[(cid, ver)] = dict(d26=d1[j26], n26=n[j26], c2_26=d2[j26], tr26=t26, has_target=has_target, d25=d1[j25],
                                  tr_all=tr)

    # S1 F1-A
    for ver, sch in (("11시판", "actual"), ("10시판", "std")):
        dd, tt = {}, None
        for cm in (1.0, 2.0):
            daily, tr = Engine(D, scheme=sch, cost_mult=cm).run(fam_F1.make(F1A), days=K)
            dd[cm] = daily
            if cm == 1.0:
                tt = tr.copy()
        tt["ret"] = tt["ret"] * 100
        tt["day"] = pd.to_datetime(tt["day"])
        put("S1", ver, dd[1.0]["ret"].values * 100, dd[2.0]["ret"].values * 100, dd[1.0]["n"].values, tt, True)
    # 원래 confirm 결과와 대조(conf 7~10월 날별)
    for ver, sch in (("11시판", "actual"), ("10시판", "std")):
        p = os.path.join(RES, f"confirm_daily_F1-A_{sch}.csv")
        if os.path.exists(p):
            cf = pd.read_csv(p, parse_dates=["day"])
            mine = pd.Series(series[("S1", ver)]["d26"], index=days26)
            v = cf.set_index("day")["ret_pct"]
            com = v.index.intersection(mine.index)
            checks[f"S1_{sch}_vs_confirm_daily_max_abs_diff"] = fnum(np.abs(v.reindex(com).values - mine.reindex(com).values).max())
            checks[f"S1_{sch}_vs_confirm_daily_n"] = int(len(com))

    # S2·S3 사건 표
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    ev = ev[(ev["side"] == "user") & np.isclose(ev["x"], X)].copy()
    ev = ev[ev["d"].isin(set(K))].copy()
    assert (pd.to_datetime(ev["day"]).values == np.array([np.datetime64(days_all[d]) for d in ev["d"]])).all()
    # 10시판 비용 2배: 결정 ≤ 09:44 사건만 sim_trade(cut 10:00, cm 2)
    rs2, rs1 = [], []
    for dec, d, pos in zip(ev["dec_i"].values, ev["d"].values, ev["buy"].map({"SOXS": -1, "SOXL": 1}).values):
        if dec <= STD_LAST_DEC:
            a, _, _, _, _ = H0M.sim_trade(D, int(d), int(dec), int(pos), CUT_STD, cm=2.0)
            b, _, _, _, _ = H0M.sim_trade(D, int(d), int(dec), int(pos), CUT_STD, cm=1.0)
            rs2.append(a * 100)
            rs1.append(b * 100)
        else:
            rs2.append(np.nan)
            rs1.append(np.nan)
    ev["r_std_c2"] = rs2
    checks["S2_r_std_recomputed_max_abs_diff"] = fnum(np.nanmax(np.abs(np.array(rs1) - ev["r_std"].values)))
    ev["is_h4"] = ev["c10_wick"] >= 0.5 - EPS
    for cid, sub in (("S2", ev), ("S3", ev[ev["is_h4"]])):
        for ver, col, c2col, wcol in (("11시판", "r_w90", "r_w90_c2", "why"), ("10시판", "r_std", "r_std_c2", "why_std")):
            s = sub[sub[col].notna()]
            d1, n = comp_by_day(s["d"].values, s[col].values, K)
            d2, _ = comp_by_day(s["d"].values, s[c2col].values, K)
            tr = pd.DataFrame({"day": pd.to_datetime(s["day"]).values, "ret": s[col].values, "why": s[wcol].astype(str).values,
                               "entry": s["entry"].values, "dec_close": s["dec_close"].values, "wick": s["c10_wick"].values})
            put(cid, ver, d1, d2, n, tr, True)
    # 점검: h0.H0 엔진 단독 실행과 날별 일치(2026, 2025)
    for ver, sch in (("11시판", "w90"), ("10시판", "std")):
        daily, _ = h0_chain(D, K, sch)
        a = daily["ret"].values * 100
        b = np.r_[series[("S2", ver)]["d25"], series[("S2", ver)]["d26"]]
        checks[f"S2_{sch}_ev4_vs_h0_engine_daily_max_abs_diff"] = fnum(np.abs(a - b).max())
        checks[f"S2_{sch}_ev4_vs_h0_engine_trades"] = [int(daily["n"].sum()),
                                                       int(series[("S2", ver)]["n26"].sum()) + int(series[("S2", ver)]["tr_all"]["day"].dt.year.eq(2025).sum())]
    log("S2 대조", {k: v for k, v in checks.items() if k.startswith("S2")})

    # S4·S5·S6
    d1, d2, n, tr = masked({k: st_prev[k] == 2 for k in K}, -1, "std")
    put("S4", "10시판(=11시판)", d1, d2, n, tr, False)
    for ver, sch in (("11시판", "w90"), ("10시판", "std")):
        d1, d2, n, tr = masked(r1, -1, sch)
        put("S5", ver, d1, d2, n, tr, False)
        d1, d2, n, tr = masked({k: st_prev[k] == -1 for k in K}, 1, sch)
        put("S6", ver, d1, d2, n, tr, False)
    # 원래 보고값 대조(조건 날 평균)
    def ev_day_mean(cid, ver):
        s = series[(cid, ver)]
        return fnum(s["d26"][s["n26"] > 0].mean())
    checks["reported_vs_now"] = {
        "S4 극상방 SOXS 10시 날 평균(보고 +0.526, 36일)": [ev_day_mean("S4", "10시판(=11시판)"), int(series[("S4", "10시판(=11시판)")]["n26"].sum())],
        "S5 R1 O2 날 평균(보고 +2.16, 13건)": [ev_day_mean("S5", "11시판"), int(series[("S5", "11시판")]["n26"].sum())],
        "S5 R1 O1 날 평균(보고 +1.05)": [ev_day_mean("S5", "10시판"), None],
        "S6 하방 SOXL 11시 날 평균(보고 +0.303, 43일)": [ev_day_mean("S6", "11시판"), int(series[("S6", "11시판")]["n26"].sum())],
        "S2 w90 하루 평균(보고 +0.078)": [fnum(series[("S2", "11시판")]["d26"].mean()), None],
        "S2 std 하루 평균(보고 +0.152)": [fnum(series[("S2", "10시판")]["d26"].mean()), None],
        "S3 w90 하루 평균(보고 +0.038, 9건)": [fnum(series[("S3", "11시판")]["d26"].mean()), int(series[("S3", "11시판")]["n26"].sum())],
        "S1 actual 7~10월(보고 +0.174)": [fnum(series[("S1", "11시판")]["d26"][days26 >= H2[0]].mean()), None],
        "S1 actual 2025(보고 −0.29)": [fnum(series[("S1", "11시판")]["d25"].mean()), None]}
    log("보고값 대조", checks["reported_vs_now"])

    # ---- 성적표
    rows, daily_rows = [], []
    for (cid, ver), s in series.items():
        st = stats(days26, s["d26"], s["n26"], s["c2_26"], s["tr26"], s["has_target"], s["d25"])
        row = {"id": cid, "name": NAME[cid], "version": ver, **st, "how_found": HOW[cid]}
        row["hit"] = row["win_rate"]
        if cid == "S1" and ver == "11시판":
            row["version_note"] = "scheme actual: 2026-01-02~03-06 표준시 10:00 마감, 03-09~ 11:00"
        if cid == "S4":
            row["version_note"] = "09:59 청산이라 11시판·10시판이 같은 매매(한 줄)"
        rows.append(row)
        for k, dday in enumerate(days26):
            daily_rows.append({"day": str(dday.date()), "id": cid, "version": ver, "ret_pct": s["d26"][k],
                               "n_trades": int(s["n26"][k]), "ret_c2_pct": s["c2_26"][k]})
    order = {f"S{i}": i for i in range(1, 7)}
    rows.sort(key=lambda r: (order[r["id"]], r["version"] != "11시판"))
    pd.DataFrame(daily_rows).to_csv(os.path.join(RES, "scoreboard_daily.csv"), index=False, encoding="utf-8-sig")
    log("성적표", [(r["id"], r["version"], round(r["per_day"], 4), r["trades"]) for r in rows])

    # 매매 목록(2026, 설명용)
    trade_lists = {}
    for (cid, ver), s in series.items():
        t = s["tr26"]
        cols = [c for c in ("day", "entry", "exit", "dec_close", "why", "ret", "wick", "sym") if c in t.columns]
        tt = t[cols].copy()
        tt["day"] = tt["day"].dt.strftime("%Y-%m-%d")
        if cid in ("S3", "S4", "S5"):
            trade_lists[f"{cid}_{ver}"] = tt.to_dict("records")

    # ---------------------------------------------------------------- 새 거래일
    recent = recent_check(series, days26, checks, D)

    out = {"brief": "성적표(오케스트레이터 지시 2026-11-04)", "rule": "AGENTS.md 연구 보고 규칙 2026-11-04: 판정·순위 2026, 2025 한 줄 보조, 2024년 이전 미사용",
           "code_sha256_16": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
           "units": "% (1.0 = 1%)", "period_2026": "2026-01-02~2026-10-28 207일", "period_2025": "2025-01-02~2025-12-31 250일",
           "settings": {"seed": SEED, "boot": NB, "F1A": F1A, "x": X, "costs": "cost_table.json(2025~26 실측), 비용 2배 = 반 호가 × 2",
                        "daily": "하루 손익 = 매매 곱 − 1, 매매 없는 날 0", "drop_top2": "하루 손익 큰 두 날을 표본에서 빼고 205일 평균",
                        "max_dd": "하루 손익을 이은 자본(시작 1.0 포함) 고점 대비 최저", "hit": "이긴 매매 비율(손익 > 0); target_hit = 목표가 도달률(S1 2R·S2·S3 +1%)"},
           "rows": rows, "checks": checks, "trades_2026": trade_lists, "recent": recent,
           "files": ["scoreboard_2026.json", "scoreboard_daily.csv"]}
    json.dump(out, open(os.path.join(RES, "scoreboard_2026.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("끝")


def recent_check(series, days26, checks, Dl):
    Dr = Data("alp1m_recent")
    rdays = [pd.Timestamp(d) for d in Dr.days]
    log(f"Data alp1m_recent {rdays[0].date()}~{rdays[-1].date()} {len(rdays)}일")
    assert all(d in rdays for d in RECENT), rdays[-6:]
    kr = [rdays.index(d) for d in RECENT]
    k_over = [k for k, d in enumerate(rdays) if d <= pd.Timestamp("2026-10-28")]
    rc = {"data": {"first": str(rdays[0].date()), "last": str(rdays[-1].date()), "days": len(rdays),
                   "fetch": "py fetch.py 2026-10-01 \"2026-11-03 20:00\" _recent (끝 = 11-04 00:00 ET와 같은 범위, 무료 키 15분 제한 회피)"}}

    # 겹치는 날 자료 대조
    dl = [pd.Timestamp(d) for d in Dl.days]
    mx = 0.0
    for k in k_over:
        kl = dl.index(rdays[k])
        for s in ("SOXL", "SOXS"):
            a, b = Dr.A[s]["Close"][k, 330:450], Dl.A[s]["Close"][kl, 330:450]
            mx = max(mx, float(np.nanmax(np.abs(a / b - 1))))
    rc["overlap_close_0930_1129_max_rel_diff"] = mx
    rc["half_day_flags_recent"] = {str(rdays[k].date()): bool(Dr.rth_end[k] != 720) for k in kr}
    rc["cutoff_actual"] = {str(rdays[k].date()): hm(cutoff_idx(rdays[k].date(), "actual")) for k in kr}

    # 단계: 새 yfinance
    Xn, src = yf_soxx_new()
    rc["soxx_daily"] = {"source": src, "first": str(Xn.index[0].date()), "last": str(Xn.index[-1].date())}
    zn_n, r_n, z_n = S5.zone_calc(Xn["Close"].to_numpy(float))
    dn = pd.DatetimeIndex(Xn.index)
    # 2021 시작판(stage5 방식)도 같은지
    m21 = dn >= pd.Timestamp("2021-01-01")
    zn21, _, _ = S5.zone_calc(Xn["Close"].to_numpy(float)[m21])
    s21 = pd.Series(zn21, index=dn[m21])
    o21 = S5.soxx_daily()
    zo21, _, _ = S5.zone_calc(o21.to_numpy(float))
    so21 = pd.Series(zo21, index=pd.DatetimeIndex(o21.index))
    com = so21.index.intersection(s21.index)
    okm = so21.reindex(com).notna() & s21.reindex(com).notna()
    o01 = RF.yf_daily("SOXX", "2001-01-01")
    zo01, _, _ = S5.zone_calc(o01["Close"].to_numpy(float))
    so01 = pd.Series(zo01, index=pd.DatetimeIndex(o01.index))
    sn = pd.Series(zn_n, index=dn)
    com1 = so01.index.intersection(sn.index)
    ok1 = so01.reindex(com1).notna() & sn.reindex(com1).notna()
    rc["stage_checks"] = {
        "new2021start_vs_stage5_cache": {"n": int(okm.sum()), "agree": fnum((so21.reindex(com)[okm] == s21.reindex(com)[okm]).mean()),
                                         "close_max_rel_diff": fnum(np.abs(o21.reindex(com).values / Xn["Close"].reindex(com).values - 1).max())},
        "new2001start_vs_redfirst_cache": {"n": int(ok1.sum()), "agree": fnum((so01.reindex(com1)[ok1] == sn.reindex(com1)[ok1]).mean())},
        "new2001_vs_new2021_from_2022": fnum((sn.reindex(s21.index)[s21.index >= "2022-01-01"] == s21[s21.index >= "2022-01-01"]).mean())}
    log("새 단계 대조", rc["stage_checks"])
    tbl = []
    for t in range(len(dn)):
        if dn[t] >= pd.Timestamp("2026-10-20"):
            tbl.append({"date": str(dn[t].date()), "close": fnum(Xn["Close"].iloc[t]), "rsi14": fnum(r_n[t]), "z20": fnum(z_n[t]),
                        "stage": int(zn_n[t]), "stage_name": S5.NAMES[int(zn_n[t])]})
    rc["soxx_stage_table"] = tbl
    tn = {d: t for t, d in enumerate(dn)}
    FXn = RF.flags(zn_n)
    pos = np.searchsorted(dn.values, np.array(rdays, dtype="datetime64[ns]"), side="left") - 1
    stp = zn_n[pos]
    r1n = np.array([bool(FXn["R1"][tn[d]]) for d in rdays])

    # 엔진 실행(새 자료 전체 날: 겹침 대조 + 새 4일)
    allk = list(range(len(rdays)))
    res = {}
    for ver, sch in (("11시판", "actual"), ("10시판", "std")):
        daily, tr = Engine(Dr, scheme=sch).run(fam_F1.make(F1A), days=allk)
        res[("S1", ver)] = (daily, tr)
    ch = {}
    for ver, sch in (("11시판", "w90"), ("10시판", "std")):
        ch[ver] = h0_chain(Dr, allk, sch)
    AL = {}
    for p_ in (1, -1):
        for sch in ("w90", "std"):
            AL[(p_, sch)] = Engine(Dr, scheme=sch).run(lambda p_=p_: S5.Always(p_), days=allk)

    # 겹치는 날(10-01~10-28) 성적표 계열과 같은지
    def ser(cid, ver):
        return pd.Series(series[(cid, ver)]["d26"], index=days26)
    ov = {}
    idx_over = pd.DatetimeIndex([rdays[k] for k in k_over])

    def cmp(cid, ver, vals):
        a = ser(cid, ver).reindex(idx_over).values
        ov[f"{cid}_{ver}"] = fnum(np.abs(a - vals).max())
    for ver in ("11시판", "10시판"):
        cmp("S1", ver, res[("S1", ver)][0]["ret"].values[k_over] * 100)
        dly, tr = ch[ver]
        cmp("S2", ver, dly["ret"].values[k_over] * 100)
        t3 = tr[tr["wick"] >= 0.5 - EPS] if len(tr) else tr
        d3 = np.zeros(len(rdays))
        for d, g in t3.groupby("d"):
            d3[int(d)] = (np.prod(1 + g["ret"].values) - 1) * 100
        cmp("S3", ver, d3[k_over])
        sch = "w90" if ver == "11시판" else "std"
        cmp("S5", ver, np.where(r1n, AL[(-1, sch)][0]["ret"].values * 100, 0.0)[k_over])
        cmp("S6", ver, np.where(stp == -1, AL[(1, sch)][0]["ret"].values * 100, 0.0)[k_over])
    cmp("S4", "10시판(=11시판)", np.where(stp == 2, AL[(-1, "std")][0]["ret"].values * 100, 0.0)[k_over])
    rc["overlap_2026_10_01_to_10_28_vs_scoreboard_max_abs_diff"] = ov
    rc["overlap_days"] = len(k_over)
    log("겹침 대조", ov)

    # 새 4일
    exec_ver = {RECENT[0]: "11시판", RECENT[1]: "11시판", RECENT[2]: "10시판", RECENT[3]: "10시판"}
    per = {}
    for k, dday in zip(kr, RECENT):
        key = str(dday.date())
        ev_ = exec_ver[dday]
        info = {"exec_version": ev_, "cutoff_actual_ET": hm(cutoff_idx(dday.date(), "actual")),
                "soxx_stage_D-1": int(stp[k]), "soxx_stage_D-1_name": S5.NAMES[int(stp[k])],
                "soxx_D-1_date": str(dn[pos[k]].date()), "R1": bool(r1n[k])}
        cands = {}
        # S1
        c1 = {}
        for ver in ("11시판", "10시판"):
            daily, tr = res[("S1", ver)]
            t = tr[pd.to_datetime(tr["day"]) == dday]
            c1[ver] = {"signal": bool(len(t)), "trades": [{"sym": r.sym, "entry": r.entry, "exit": r.exit, "why": r.why,
                                                           "ret": fnum(r.ret * 100)} for r in t.itertuples()],
                       "day_pnl": fnum(daily["ret"].values[k] * 100)}
        o, cc = Dr.A["SOXL"]["Open"][k, 330], Dr.A["SOXL"]["Close"][k, 330]
        c1["first_bar_SOXL"] = {"open": fnum(o), "close": fnum(cc), "dir": "양봉→SOXL 대기" if cc > o else ("음봉→SOXS 대기" if cc < o else "같음(쉼)")}
        cands["S1"] = c1
        # S2·S3
        for cid in ("S2", "S3"):
            cx = {}
            for ver in ("11시판", "10시판"):
                dly, tr = ch[ver]
                t = tr[tr["d"] == k] if len(tr) else tr
                if cid == "S3" and len(t):
                    t = t[t["wick"] >= 0.5 - EPS]
                cx[ver] = {"signal": bool(len(t)),
                           "trades": [{"dec_close": r.dec_close, "entry": r.entry, "exit": r.exit, "why": r.why,
                                       "ret": fnum(r.ret * 100), "wick": fnum(r.wick)} for r in t.itertuples()],
                           "day_pnl": fnum((np.prod(1 + t["ret"].values) - 1) * 100) if len(t) else 0.0}
            if cid == "S2":
                t11 = ch["11시판"][1]
                cx["all_chain_wicks_11"] = [fnum(w) for w in t11.loc[t11["d"] == k, "wick"]] if len(t11) else []
            cands[cid] = cx
        # S4
        sig4 = bool(stp[k] == 2)
        t4 = AL[(-1, "std")][1]
        t4 = t4[pd.to_datetime(t4["day"]) == dday]
        cands["S4"] = {"10시판(=11시판)": {"signal": sig4, "ret_if_traded": fnum(t4["ret"].iloc[0] * 100),
                                          "day_pnl": fnum(t4["ret"].iloc[0] * 100) if sig4 else 0.0}}
        for cid, flag, p_ in (("S5", bool(r1n[k]), -1), ("S6", bool(stp[k] == -1), 1)):
            cx = {}
            for ver, sch in (("11시판", "w90"), ("10시판", "std")):
                t = AL[(p_, sch)][1]
                t = t[pd.to_datetime(t["day"]) == dday]
                v = fnum(t["ret"].iloc[0] * 100)
                cx[ver] = {"signal": flag, "ret_if_traded": v, "day_pnl": v if flag else 0.0, "exit": t["exit"].iloc[0]}
            cands[cid] = cx
        info["candidates"] = cands
        info["exec_pnl"] = {cid: (cands[cid]["10시판(=11시판)"]["day_pnl"] if cid == "S4" else cands[cid][ev_]["day_pnl"])
                            for cid in cands}
        per[key] = info
    rc["days"] = per
    # 4일 합(실행판)
    rc["exec_total_4days"] = {cid: fnum((np.prod([1 + per[str(d.date())]["exec_pnl"][cid] / 100 for d in RECENT]) - 1) * 100)
                              for cid in ("S1", "S2", "S3", "S4", "S5", "S6")}
    rc["exec_signals_4days"] = {cid: int(sum(
        (per[str(d.date())]["candidates"][cid]["10시판(=11시판)"]["signal"] if cid == "S4"
         else per[str(d.date())]["candidates"][cid][exec_ver[d]]["signal"]) for d in RECENT)) for cid in ("S1", "S2", "S3", "S4", "S5", "S6")}
    # 덤: 11-04 세션(11-03 종가 단계)
    t_last = len(dn) - 1
    st_today_prev = int(zn_n[t_last])
    r1_today = bool(zn_n[t_last] <= -1 and not (zn_n[t_last - 1] <= -1))
    rc["preview_2026_11_04"] = {"soxx_stage_2026_11_03": st_today_prev, "name": S5.NAMES[st_today_prev],
                                "rsi14": fnum(r_n[t_last]), "z20": fnum(z_n[t_last]),
                                "S4_on": st_today_prev == 2, "S5_on": r1_today, "S6_on": st_today_prev == -1,
                                "exec_version": "10시판(표준시, 마감 10:00 ET = 24:00 KST)"}
    log("새 4일", {d: per[d]["exec_pnl"] for d in per}, rc["preview_2026_11_04"])
    return rc


if __name__ == "__main__":
    main()
