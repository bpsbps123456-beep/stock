"""bottom_confirm 공통 계산 — BRIEF.md(2026-10-26, 결과 전 고정).

질문(사용자 2026-10-26): "완전한 바닥이 아니더라도 바닥으로 확인할 수 있다는 게 하나도 없나? 확률 70% 이상이면 충분하다."

모든 갈래가 이 파일의 결과 정의·기준선·통계·판정을 그대로 쓴다. 갈래 코드는 신호(날짜 × 종목 bool)만 만든다.

자료(읽기만, 클라우드 세션에서 Yahoo 차단이라 저장소 자료만 쓴다)
  SOXX OHLC  research/news_turning_points_20261025/prices.csv (2023-12-01~2026-10-23, Codex 10/25 스냅샷)
             + 2023 종가 data/derived/analysis/leverage_flood/results/us_prices_close.csv (같은 yfinance 값, 겹치는 날 오차 0)
  시장 지표   us_prices_close.csv(^VIX·^VIX3M·^VXN·^SKEW·^SOX 등, ~2026-10-13), us_daily_panel.csv(풋콜·레버리지 ETF 자금, 2023-06~2026-10-13)
  종목       data/derived/desk/prices_open.csv·prices_close.csv(미국 82종목, QQQ·한국·환율 제외, 2025-06-02~2026-10-23)
  뉴스       data/derived/analysis/news_followup/results/n4h14_signals.csv(전수 노트 245개의 진입일·family·lookback)

정의(신호일 t = 그날 종가까지 앎, 진입 e = t+1 정규장 시가)
  pk[t]    = 최근 60거래일 종가 최고점의 위치(같은 값이면 늦은 날)
  Ldd[t]   = min(C[pk[t]..t])                       지금 하락의 저점(60일 고점 이후 최저 종가)
  dd60[t]  = C[t]/max(C[t-59..t]) − 1               60일 고점 대비 낙폭
  up[t]    = C[t]/Ldd[t] − 1                        하락 저점 대비 이미 오른 폭(늦음의 값)
  in_dd[t] = min(dd60[t-19..t]) ≤ −THR              최근 20거래일 안에 60일 고점 대비 −THR 이하였다(하락 국면)
  hold20   = min(C[t+1..t+20]) ≥ Ldd[t]             주 결과 '바닥 확인': 앞으로 한 달 안에 그 하락 저점을 다시 깨지 않음
  hold40   = min(C[t+1..t+40]) ≥ Ldd[t]             보조(두 달)
  (처음 정의한 '60일 최저 종가'는 큰 상승 뒤 하락에서 저점이 석 달 전 값이라 거의 항상 지켜져(2026 in_dd 날 100%)
   신호를 계산하기 전에 이 정의로 바꿨다.)
  near5    = min(C[e..e+39]) ≥ 0.95·O[e]            보조 '바닥 근처 매수': 산 값에서 두 달 안에 5% 넘게 더 빠지지 않음
  r20      = C[e+19]/O[e] − 1, win20 = r20 > 0;  r60 = C[e+59]/O[e] − 1
  THR: SOXX 10%, 종목 20%
  첫 신호: 같은 계열(종목)에서 직전 채택 신호와 20거래일 이하로 붙은 신호는 버린다.
  대조(matched): 같은 계열 묶음·같은 기간 조각(SOXX는 같은 해, 종목은 같은 반기)의 in_dd 날 가운데 up 구간(0~3·3~8·8~15·15%+)이 같은 날의 평균.
  통계: 신호 날짜를 20거래일 안이면 같은 묶음으로 묶은 부트스트랩(B=5000) 95% 구간.
"""
from __future__ import annotations

import pathlib

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[5]
HERE = pathlib.Path(__file__).resolve().parents[1]
RES = HERE / "results"
RES.mkdir(exist_ok=True)

LAST = pd.Timestamp("2026-10-23")
COOL = 20
B = 5000
SEED = 20261026
UP_BINS = [0.0, 0.03, 0.08, 0.15, np.inf]
THR = {"SOXX": 0.10, "STK": 0.20}
PERIODS = {
    "SOXX": [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
             ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
             ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")],
    "STK": [("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
            ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
            ("2025", "2025-09-01", "2025-12-31")],
}
EVAL_START = {"SOXX": pd.Timestamp("2024-01-02"), "STK": pd.Timestamp("2025-09-01")}
OUTCOMES = ("hold20", "hold40", "near5", "win20", "r20", "r60")


def xnys(a, b) -> pd.DatetimeIndex:
    s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(a, b))
    return s.tz_localize(None) if s.tz is not None else s


# ---------------------------------------------------------------- 자료
def load_soxx() -> pd.DataFrame:
    """SOXX 일봉: 2023-01-03~2026-10-23. O·H·L 은 2023-12-01 부터."""
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()].dropna().sort_index()
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna()
    cal = xnys("2023-01-01", "2026-10-23")
    d = pd.DataFrame(index=cal)
    d["C"] = u.reindex(cal)
    d.loc[p.index, "C"] = p["close"]["SOXX"]
    for k, col in (("O", "open"), ("H", "high"), ("L", "low")):
        d[k] = p[col]["SOXX"].reindex(cal)
    ov = u.index.intersection(p.index)
    assert np.allclose(u[ov].to_numpy(), p["close"]["SOXX"][ov].to_numpy(), rtol=0, atol=1e-9)
    assert d.C.notna().all(), "SOXX 종가 결측"
    assert d.loc["2023-12-01":, ["O", "H", "L"]].notna().all().all()
    return d


def load_market() -> pd.DataFrame:
    """시장 지표(종가). 세션 색인 2023-01-03~2026-10-23, 자료 없는 날은 NaN(10/14 이후 등)."""
    cal = xnys("2023-01-01", "2026-10-23")
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)
    pnl = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_daily_panel.csv",
                      encoding="utf-8-sig", index_col=0, parse_dates=True)
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()].dropna()
    m = pd.DataFrame(index=cal)
    for c in u.columns:
        m[c] = u[c].reindex(cal)
    for c in pnl.columns:
        if c not in m.columns:
            m[c] = pnl[c].reindex(cal)
    m["TNX"] = p["close"]["^TNX"].reindex(cal)
    return m


def load_stocks():
    """종목 시가·종가(desk 캐시 미국 82종목), XNYS 세션 2025-06-02~2026-10-23."""
    o = pd.read_csv(ROOT / "data/derived/desk/prices_open.csv", index_col=0, parse_dates=True)
    c = pd.read_csv(ROOT / "data/derived/desk/prices_close.csv", index_col=0, parse_dates=True)
    cal = xnys("2025-06-02", "2026-10-23")
    drop = [x for x in c.columns if x.endswith(".KS") or x.endswith("=X") or x == "QQQ"]
    o, c = o.drop(columns=drop).reindex(cal), c.drop(columns=drop).reindex(cal)
    return o, c


def load_notes() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "data/derived/analysis/news_followup/results/n4h14_signals.csv",
                    encoding="utf-8-sig", parse_dates=["entry"])
    return s[["group_id", "bank", "entry", "family", "lookback_months", "sell_note", "flow_direction", "scope",
              "verification", "legacy_ids"]].copy()


# ---------------------------------------------------------------- 결과 배열
def outcome_frame(O: pd.DataFrame, C: pd.DataFrame, thr: float) -> dict[str, pd.DataFrame]:
    """O, C: 같은 세션 색인 × 계열 열. 반환: 이름 → (세션 × 계열) 표."""
    Cv, Ov = C.to_numpy(float), O.to_numpy(float)
    N, K = Cv.shape
    roll_max = C.rolling(60, min_periods=60).max()
    dd60 = C / roll_max - 1
    Ldd = np.full((N, K), np.nan)
    for k in range(K):
        c = Cv[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            pk = t - 59 + (59 - int(np.argmax(w[::-1])))          # 최고값의 가장 늦은 위치
            Ldd[t, k] = c[pk:t + 1].min()
    roll_min = pd.DataFrame(Ldd, index=C.index, columns=C.columns)
    up = C / roll_min - 1
    in_dd = dd60.rolling(20, min_periods=1).min() <= -thr
    L60 = Ldd

    def fwd_min(start, length):
        out = np.full((N, K), np.nan)
        for t in range(N):
            a, b = t + start, t + start + length
            if b <= N:
                blk = Cv[a:b]
                out[t] = np.where(np.isnan(blk).any(0), np.nan, np.nanmin(blk, 0))
        return out

    m20, m40 = fwd_min(1, 20), fwd_min(1, 40)
    hold20 = np.where(np.isnan(m20) | np.isnan(L60), np.nan, (m20 >= L60 - 1e-12).astype(float))
    hold40 = np.where(np.isnan(m40) | np.isnan(L60), np.nan, (m40 >= L60 - 1e-12).astype(float))
    Oe = np.full((N, K), np.nan)
    Oe[:-1] = Ov[1:]
    me40 = fwd_min(1, 40)                                      # C[e..e+39] = C[t+1..t+40]
    near5 = np.where(np.isnan(me40) | np.isnan(Oe), np.nan, (me40 >= 0.95 * Oe).astype(float))
    r20 = np.full((N, K), np.nan)
    r20[:N - 20] = Cv[20:] / Oe[:N - 20] - 1                   # C[t+20] = C[e+19]
    r60 = np.full((N, K), np.nan)
    r60[:N - 60] = Cv[60:] / Oe[:N - 60] - 1
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    idx, cols = C.index, C.columns
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return dict(hold20=f(hold20), hold40=f(hold40), near5=f(near5), win20=f(win20), r20=f(r20), r60=f(r60),
                up=up, dd60=dd60, in_dd=in_dd.fillna(False).astype(bool), L60=roll_min)


def soxx_frames():
    d = load_soxx()
    return outcome_frame(d[["O"]].rename(columns={"O": "SOXX"}), d[["C"]].rename(columns={"C": "SOXX"}), THR["SOXX"])


def stock_frames():
    o, c = load_stocks()
    return outcome_frame(o, c, THR["STK"])


# ---------------------------------------------------------------- 평가
def first_signals(sig: pd.DataFrame, fr: dict, kind: str) -> pd.DataFrame:
    """sig: 세션 × 계열 bool(그날 종가 기준 신호). in_dd 이고 평가 창 안인 날만, 계열마다 20거래일 쿨다운.
    반환: 행 = 채택 신호(date, series, pos)."""
    sig = sig.reindex(index=fr["in_dd"].index, columns=fr["in_dd"].columns).fillna(False).astype(bool)
    ok = sig & fr["in_dd"] & fr["L60"].notna()
    ok &= pd.DataFrame(np.broadcast_to((ok.index >= EVAL_START[kind])[:, None], ok.shape), index=ok.index, columns=ok.columns)
    ok &= pd.DataFrame(np.broadcast_to((ok.index <= LAST)[:, None], ok.shape), index=ok.index, columns=ok.columns)
    rows = []
    for col in ok.columns:
        pos = np.flatnonzero(ok[col].to_numpy())
        prev = -10 ** 9
        for p in pos:
            if p - prev <= COOL:
                continue
            rows.append((ok.index[p], col, p))
            prev = p
    return pd.DataFrame(rows, columns=["date", "series", "pos"])


def _slice_key(dates: pd.DatetimeIndex, kind: str) -> np.ndarray:
    y, m = np.asarray(dates.year), np.asarray(dates.month)
    return y if kind == "SOXX" else y * 10 + (m > 6)


def matched_baseline(fr: dict, kind: str) -> dict[str, pd.DataFrame]:
    """같은 조각 × up 구간의 in_dd 날 평균. 반환 결과명 → 세션 × 계열(값은 계열 공통, 조각·구간별)."""
    up = fr["up"]
    ub = pd.DataFrame(np.digitize(up.to_numpy(float), UP_BINS[1:-1]), index=up.index, columns=up.columns)
    ub = ub.where(up.notna())
    sk = pd.DataFrame(np.broadcast_to(_slice_key(up.index, kind)[:, None], up.shape), index=up.index, columns=up.columns)
    out = {}
    st = fr["in_dd"] & up.notna() & pd.DataFrame(np.broadcast_to((up.index >= EVAL_START[kind])[:, None], up.shape),
                                                  index=up.index, columns=up.columns)
    for k in OUTCOMES:
        y = fr[k]
        long = pd.DataFrame(dict(y=y.stack(future_stack=True), b=ub.stack(future_stack=True), s=sk.stack(future_stack=True),
                                 st=st.stack(future_stack=True)))
        g = long[long.st & long.y.notna() & long.b.notna()].groupby(["s", "b"]).y.mean()
        val = pd.Series([g.get((s, b), np.nan) if not (pd.isna(b) or pd.isna(s)) else np.nan
                         for s, b in zip(long.s, long.b)], index=long.index)
        out[k] = val.unstack()
    return out


def base_rates(fr: dict, kind: str) -> pd.DataFrame:
    """무조건 기준선: 평가 창의 in_dd 날 전부(종목은 종목·날 전부)."""
    rows = []
    for pname, a, b in PERIODS[kind]:
        m = (fr["in_dd"].index >= a) & (fr["in_dd"].index <= b)
        st = fr["in_dd"].loc[m]
        r = dict(period=pname, state_days=int(st.to_numpy().sum()))
        for k in OUTCOMES:
            y = fr[k].loc[m].where(st)
            r[k] = float(np.nanmean(y.to_numpy())) if np.isfinite(y.to_numpy()).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _clusters(dates: pd.DatetimeIndex, pos: np.ndarray) -> np.ndarray:
    o = np.argsort(pos, kind="stable")
    lab = np.empty(len(pos), int)
    k = 0
    for j, t in enumerate(o):
        if j > 0 and pos[t] - pos[o[j - 1]] > COOL:
            k += 1
        lab[t] = k
    return lab


def _boot(v: np.ndarray, lab: np.ndarray, rng) -> tuple[float, float]:
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    sums = np.bincount(inv, weights=v)
    cnts = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def evaluate(sig: pd.DataFrame, fr: dict, base: dict, kind: str, rule: str, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (기간별 요약 표, 채택 신호 표)."""
    fs = first_signals(sig, fr, kind)
    rng = np.random.default_rng(SEED)
    if len(fs):
        for k in OUTCOMES:
            fs[k] = [fr[k].iat[p, fr[k].columns.get_loc(s)] for p, s in zip(fs.pos, fs.series)]
            fs[f"base_{k}"] = [base[k].iat[p, base[k].columns.get_loc(s)] for p, s in zip(fs.pos, fs.series)]
        fs["up"] = [fr["up"].iat[p, fr["up"].columns.get_loc(s)] for p, s in zip(fs.pos, fs.series)]
        fs["dd60"] = [fr["dd60"].iat[p, fr["dd60"].columns.get_loc(s)] for p, s in zip(fs.pos, fs.series)]
    rows = []
    for pname, a, b in PERIODS[kind]:
        g = fs[(fs.date >= a) & (fs.date <= b)] if len(fs) else fs
        r = dict(family=family, rule=rule, kind=kind, period=pname, n=len(g))
        if len(g):
            lab = _clusters(pd.DatetimeIndex(g.date), g.pos.to_numpy())
            r["clusters"] = int(len(np.unique(lab)))
            r["up_median"] = float(np.median(g.up))
            r["dd60_median"] = float(np.median(g.dd60))
            for k in OUTCOMES:
                ok = g[k].notna().to_numpy() & g[f"base_{k}"].notna().to_numpy()
                v, bv = g[k].to_numpy(float)[ok], g[f"base_{k}"].to_numpy(float)[ok]
                r[f"{k}_n"] = int(ok.sum())
                if ok.sum() == 0:
                    continue
                r[k] = float(v.mean())
                r[f"{k}_base"] = float(bv.mean())
                lo, hi = _boot(v, lab[ok], rng)
                dlo, dhi = _boot(v - bv, lab[ok], rng)
                r.update({f"{k}_lo": lo, f"{k}_hi": hi, f"{k}_diff": float((v - bv).mean()), f"{k}_dlo": dlo, f"{k}_dhi": dhi})
        rows.append(r)
    return pd.DataFrame(rows), fs


def verdict(tab: pd.DataFrame, kind: str) -> dict:
    """사전 등록 판정. L1 = 70% 바닥 확인, L2 = L1 + 같은 상태 대조보다 높음."""
    t = tab.set_index("period")

    def get(p, k, default=np.nan):
        return t.loc[p, k] if (p in t.index and k in t.columns and pd.notna(t.loc[p, k])) else default

    if kind == "SOXX":
        main, n_main = "2024~2026", get("2024~2026", "hold20_n", 0)
        c1 = n_main >= 8 and get(main, "hold20", 0) >= 0.70
        c2 = get("2026", "hold20_n", 0) >= 3 and get("2026", "hold20", 0) >= 0.70
        c3 = get("2025", "hold20_n", 0) < 3 or get("2025", "hold20", 0) >= 0.60
        l1 = bool(c1 and c2 and c3)
        l2 = bool(l1 and get(main, "hold20_dlo", -1) > 0)
    else:
        main = "2026"
        ch = [get(h, "hold20_n", 0) >= 5 and get(h, "clusters", 0) >= 2 and get(h, "hold20", 0) >= 0.70 for h in ("2026H1", "2026H2")]
        c3 = get("2025", "hold20_n", 0) < 5 or get("2025", "hold20", 0) >= 0.60
        l1 = bool(all(ch) and c3 and get("2026", "hold20", 0) >= 0.70)
        l2 = bool(l1 and get(main, "hold20_dlo", -1) > 0)
    return dict(L1=l1, L2=l2, main_period=main, hold20=get(main, "hold20"), hold20_base=get(main, "hold20_base"),
                hold20_n=get(main, "hold20_n", 0), near5=get(main, "near5"), win20=get(main, "win20"),
                r20=get(main, "r20"), up_median=get(main, "up_median"))


def run_rules(rules: dict, kind: str, family: str, fr=None, base=None, tag=None):
    """rules: 규칙명 → 신호표(세션 × 계열 bool). 결과를 results/<tag>_{summary,signals,verdict}.csv 로 쓴다."""
    fr = fr if fr is not None else (soxx_frames() if kind == "SOXX" else stock_frames())
    base = base if base is not None else matched_baseline(fr, kind)
    tabs, sigs, vers = [], [], []
    for name, sig in rules.items():
        tab, fs = evaluate(sig, fr, base, kind, name, family)
        tabs.append(tab)
        if len(fs):
            fs.insert(0, "rule", name)
            sigs.append(fs)
        v = verdict(tab, kind)
        v.update(family=family, rule=name, kind=kind)
        vers.append(v)
    tag = tag or f"{family}_{kind}"
    summ = pd.concat(tabs, ignore_index=True)
    summ.to_csv(RES / f"{tag}_summary.csv", index=False, encoding="utf-8-sig")
    (pd.concat(sigs, ignore_index=True) if sigs else pd.DataFrame()).to_csv(RES / f"{tag}_signals.csv", index=False, encoding="utf-8-sig")
    ver = pd.DataFrame(vers)
    ver.to_csv(RES / f"{tag}_verdict.csv", index=False, encoding="utf-8-sig")
    return summ, ver


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    fr = soxx_frames()
    print("SOXX 기준선(in_dd 날 전부)")
    print(base_rates(fr, "SOXX").round(3).to_string(index=False))
    sf = stock_frames()
    print("종목 기준선(종목·날 전부, in_dd)")
    print(base_rates(sf, "STK").round(3).to_string(index=False))
    # 자기 점검: 모든 in_dd 날을 신호로 쓰면 대조와 거의 같아야 한다(쿨다운 때문에 같지는 않음)
    allsig = fr["in_dd"].copy()
    tab, _ = evaluate(allsig, fr, matched_baseline(fr, "SOXX"), "SOXX", "all_in_dd", "selftest")
    print(tab[["period", "n", "hold20", "hold20_base", "near5", "near5_base", "win20", "win20_base"]].round(3).to_string(index=False))
