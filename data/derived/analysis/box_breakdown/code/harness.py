"""박스 하단 이탈 연구 공통 하네스. 정의·구간·판정은 ../BRIEF.md(사전 등록) 그대로다.

사용
    py harness.py build      # 일봉 캐시 → 사건표 events_N{20,40,60}.parquet → baseline.md
    갈래 스크립트:
        sys.path.insert(0, r"<box_breakdown>/code"); import harness as H
        ev = H.load_events(20)                 # 사건표(특징·결과 포함)
        P = H.load_panel()                     # 일봉 패널(cal, syms, O/H/L/C/V 배열)
        cells = H.tercile_cells(ev, H.FEATURES_BASE)
        tab, summ = H.evaluate_cells(ev, cells)

정의 해석(결과를 보기 전에 정함, BRIEF가 열어 둔 부분)
- 같은 사건 묶기: 같은 종목에서 '직전 이탈일'(묶인 날 포함) 뒤 10거래일 안의 이탈은 같은 사건이다(연쇄).
  사건의 t는 묶음의 첫 이탈일이다.
- 되돌림 손절: t+1..t+h−1 종가 > L이면 다음 날 시가에 청산, 아니면 t+h 종가. (t+h 종가가 L 위여도
  t+h 종가 청산이 더 이르다.)
- 분류: 진짜 = t+1..t+10에서 L 위 종가가 나오기 전에 저가 ≤ L−2·ATR(같은 날 저가 도달 뒤 L 위 종가는
  진짜). 가짜 = t+1..t+5에서 L 위 종가, 그날까지(t+1부터) 저가 > L−1·ATR. 나머지 = 중간. 둘은 서로 배타다.
  t+10까지 자료가 없고 아직 정해지지 않았으면 None.
- 초과 손익: 같은 날 같은 kind 사건이 2개 이상이면 그 평균(자기 포함)을 뺀다. 1개면 같은 날 이탈 상태가
  아닌(종가 ≥ 자기 L_N) 같은 kind 종목 하나를 시드 고정으로 뽑아, 같은 보유 창(t+1 시가 진입 → 사건과 같은
  청산일·같은 가격 종류)의 공매도 손익(비용 포함)을 뺀다. 비이탈 종목에 되돌림 규칙을 그대로 쓰면 첫날
  바로 청산되어 시장 효과 대조가 안 되므로 창을 맞췄다.
- 국면 표지 episode: QQQ 또는 SOXX 종가 ≤ 직전 5거래일 최고 고가 × 0.95인 날을 표시하고, 표시일 간격이
  10거래일 이하이면 한 묶음. 묶음 창 = [첫 표시일 − 5거래일, 마지막 표시일](5일 고점 비교라 하락 시작이
  첫 표시 5일 안에 있다). 창 밖은 −1. t 뒤 자료를 쓰므로 특징이 아니라 사후 표지다(판정의 한 국면 점검 전용).
- ret5_pre·ret20_pre는 이탈 전(t−1 종가까지) 수익. t 당일 수익은 ret_t.
- semi: AI_UNIVERSE 대분류 칩 설계·파운드리·후공정·메모리·스토리지·장비·소재·테스트. ETF는 SOXX·SMH만 1.
- cobreak_share: 같은 날 같은 kind에서 새 사건(묶기 뒤 첫날)이 난 종목 수 ÷ 그날 L_N이 있는 같은 kind 종목 수.
- earn_pm3: 실적 반응일(발표 시각 16시 이후면 다음 거래일, 아니면 그날 또는 다음 첫 거래일)이 t±3거래일 안.
  발표 날짜만 쓰고 EPS·서프라이즈는 쓰지 않는다(t 뒤 결과 사용 금지).
- 날짜 묶음 t: 켜짐 표지 회귀를 이탈일 t로 묶은 군집 강건 표준오차(CR1)로 낸 (켜짐 평균 − 꺼짐 평균)의 t.
  날짜 고정 특징(시장 맥락)과 종목 간 특징을 같은 식으로 다룬다.
- 날짜 묶음 순열(Permuter, 2026-10-14 감사에서 고침 — results/qa_harness.md): 구간의 이탈 날짜 전체에 균일
  무작위 순열 pi 하나를 뽑는다. 날짜 고정 칸(같은 날 표지가 모두 같은 칸: 시장 맥락)은 그 칸 모집단 날짜로
  줄인 첫 귀환 지도로 날짜 표지를 통째 옮긴다(N·kind 층을 가로질러 같은 날 사건이 같은 원천 날짜를 받는다).
  나머지 칸은 층(N·kind) 안에서 같은 pi의 첫 귀환 지도로 행 묶음을 옮기고, 원천 사건 수가 모자라면 복원
  추출로 채운다. 칸 사이 상관은 모든 칸에 같은 pi를 써서 보존한다. 순열 p는 (1+#{≥관측})/(1+#{유한 회차}),
  관측 차이가 NaN이면 p도 NaN.
- 고르기는 앞 구간 차이 > 0인 칸 중 상위 top개(켜짐 ≥ min_on, 켜짐 날짜 ≥ min_dates). 통과 = 뒤 구간
  차이 > 0, t ≥ 2, 순열 p < 0.05(한쪽), 켜진 사건 평균 공매도 손익 > 0. 한 국면 의존은 표시만 한다.
- 보조 지표(판정 기준 아님, 감사에서 추가): b_p_shift = 이탈 날짜를 시간순 1..D−1칸 원형 이동한 순열 p
  (국면 지속성 보존), pass_shift = pass & b_p_shift < 0.05, summ['null_shift'] = 앞·뒤 표지를 원형 이동한
  선택편향 귀무. 날짜 i.i.d. 순열과 날짜 묶음 t는 오래 이어지는 날짜 고정 특징(VIX·시장폭·50일선 등)에서
  거짓 양성이 명목보다 많다(가짜 AR(1) 특징으로 P(t≥2 & p<0.05) 5~8%, 원형 이동 p를 더하면 2~5%).
- 실적일: 달력 첫날(2022-06-01) 전 발표는 버린다(예전에는 0번 날로 붙었다. 사건은 i ≥ 21이라 영향 없음).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent                      # .../analysis/box_breakdown
ANALYSIS = ROOT.parent
REPO = ROOT.parents[3]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import watchlist  # noqa: E402

START, END = "2022-06-01", "2026-10-13"
FRONT = ("2026-01-02", "2026-05-29")
BACK = ("2026-06-01", "2026-09-29")
AUX = ("2025-01-01", "2025-12-31")
NS = (20, 40, 60)
COST = 0.001        # 왕복 스프레드 0.10%
DEDUP = 10          # 같은 사건 묶기(거래일)
SEED = 20261014

SEMI_GROUPS = ("칩 설계", "파운드리·후공정", "메모리·스토리지", "장비·소재·테스트")
STOCKS = [s for g in watchlist.AI_UNIVERSE.values() for ss in g.values() for s in ss if "." not in s]
SEMI = {s for gn, g in watchlist.AI_UNIVERSE.items() if gn in SEMI_GROUPS for ss in g.values() for s in ss}
ETFS = ["QQQ", "SPY", "SOXX", "SMH", "IWM", "XLK", "IGV"]
SEMI_ETFS = {"SOXX", "SMH"}
HEDGES = ["SOXS", "SQQQ", "QID", "PSQ", "SH"]
VOLS = ["^VIX", "^VIX3M"]
ALL_SYMS = STOCKS + ETFS + HEDGES + VOLS
EVENT_SYMS = STOCKS + ETFS

FEATURES_BASE = [
    "box_w_atr", "slope50", "touches", "depth_atr", "gap_break", "close_loc", "vol_ratio",
    "ret5_pre", "ret20_pre", "earn_pm3", "qqq_above50", "qqq_above200", "soxx_above50",
    "cobreak_share", "breadth50", "vix", "vix_ts", "semi",
]
# 같은 시점(t 종가)의 보조 특징. 갈래가 쓰면 칸 수에 넣는다.
FEATURES_EXTRA = ["ret_t", "gap_pct", "atr_pct", "rs20_qqq", "earn_dist"]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


# ───────────────────────────── 자료 ─────────────────────────────
def _yf(tickers, start, end):
    """yfinance 일봉(분할 반영, 배당 미반영 = auto_adjust=False) → 긴 형식."""
    import yfinance as yf
    if not tickers:
        return pd.DataFrame(columns=["date", "sym"] + FIELDS)
    f = yf.download(list(tickers), start=start, end=end, auto_adjust=False, progress=False,
                    threads=True, group_by="column")
    rows = []
    for s in tickers:
        try:
            x = pd.DataFrame({c: f[c][s] for c in FIELDS})
        except KeyError:
            continue
        x = x.dropna(subset=["Close"])
        if len(x):
            x.index.name = "date"
            rows.append(x.reset_index().assign(sym=s))
    if not rows:
        return pd.DataFrame(columns=["date", "sym"] + FIELDS)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out[["date", "sym"] + FIELDS]


def _overlap_diff(a, b):
    m = a.set_index("date")["Close"].to_frame("a").join(b.set_index("date")["Close"].to_frame("b"), how="inner")
    if m.empty:
        return np.nan, 0
    return float((m["a"] / m["b"] - 1).abs().max()), len(m)


def build_daily(force=False, tol=0.005):
    """기존 캐시 우선 → 빈 곳만 yfinance. 겹치는 날 종가가 tol 넘게 다르면(분할 의심) 그 종목은 yfinance 전 구간."""
    out = CACHE / "daily.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    sr = pd.read_parquet(ANALYSIS / "stop_research/cache/ohlc.parquet")
    sr["date"] = pd.to_datetime(sr["date"]).dt.normalize()
    sr = sr[sr["sym"].isin(ALL_SYMS)][["date", "sym"] + FIELDS]
    sr_syms = set(sr["sym"])
    dip_dir = ANALYSIS / "trend_zones/cache/dip_raw_1010"
    dip = {}
    for s in ALL_SYMS:
        p = dip_dir / f"1d_{s}.parquet"
        if p.exists():
            x = pd.read_parquet(p)
            x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
            x.index.name = "date"
            dip[s] = x[FIELDS].astype(float).reset_index().assign(sym=s)[["date", "sym"] + FIELDS]
    yend = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    need_full = [s for s in ALL_SYMS if s not in sr_syms]
    need_tail = [s for s in ALL_SYMS if s in sr_syms]
    yf_full = _yf(need_full, START, yend)
    yf_tail = _yf(need_tail, "2026-08-01", yend)
    parts, man = [], {}
    refetch = []
    for s in ALL_SYMS:
        info = {}
        if s in sr_syms:
            base = sr[sr["sym"] == s]
            tail = yf_tail[yf_tail["sym"] == s]
            d, n = _overlap_diff(base, tail)
            info.update(source="stop_research+yf_tail", overlap_days=n, overlap_maxdiff=d)
            if n and d > tol:
                refetch.append(s)
                continue
            x = pd.concat([base, tail[tail["date"] > base["date"].max()]])
        else:
            full = yf_full[yf_full["sym"] == s]
            if s in dip:
                d, n = _overlap_diff(dip[s], full)
                info.update(source="dip_raw_1010+yf", overlap_days=n, overlap_maxdiff=d)
                if n and d > tol:
                    refetch.append(s)
                    continue
                x = pd.concat([full[~full["date"].isin(dip[s]["date"])], dip[s]])
            else:
                info.update(source="yf")
                x = full
        man[s] = info
        parts.append(x)
    if refetch:
        rf = _yf(refetch, START, yend)
        for s in refetch:
            parts.append(rf[rf["sym"] == s])
            man[s] = {"source": "yf_full(겹침 불일치로 재수신)"}
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df.loc[df["Open"] <= 0, "Open"] = np.nan
    df = df.dropna(subset=["Close"]).drop_duplicates(["date", "sym"], keep="last")
    df = df.sort_values(["sym", "date"]).reset_index(drop=True)
    df.to_parquet(out)
    got = set(df["sym"])
    for s in ALL_SYMS:
        if s in got:
            g = df[df["sym"] == s]
            man.setdefault(s, {}).update(first=str(g["date"].min().date()), last=str(g["date"].max().date()), rows=len(g))
    manifest = {
        "requested": len(ALL_SYMS), "received": len(got & set(ALL_SYMS)),
        "missing": [s for s in ALL_SYMS if s not in got],
        "not_to_end": [s for s in got if man[s]["last"] < END],
        "late_start": {s: man[s]["first"] for s in got if man[s]["first"] > "2022-06-03"},
        "refetched": refetch, "stocks": len([s for s in STOCKS if s in got]),
        "etfs": len([s for s in ETFS if s in got]), "per_symbol": man,
    }
    (RESULTS / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str),
                                                encoding="utf-8")
    return df


@dataclass
class Panel:
    cal: pd.DatetimeIndex
    syms: list
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray

    def __post_init__(self):
        self.j = {s: k for k, s in enumerate(self.syms)}

    def col(self, field, sym):
        return getattr(self, field)[:, self.j[sym]]

    @classmethod
    def from_long(cls, df, calendar_sym="QQQ"):
        cal = pd.DatetimeIndex(sorted(df.loc[df["sym"] == calendar_sym, "date"].unique()))
        syms = sorted(df["sym"].unique())
        arr = {}
        for f, k in zip(FIELDS, "OHLCV"):
            w = df.pivot(index="date", columns="sym", values=f).reindex(index=cal, columns=syms)
            arr[k] = w.to_numpy(dtype=float)
        return cls(cal, syms, arr["O"], arr["H"], arr["L"], arr["C"], arr["V"])


_PANEL = None


def load_panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = Panel.from_long(build_daily())
    return _PANEL


# ───────────────────────────── 사건 정의(배열 함수) ─────────────────────────────
def box_levels(o, h, l, c, N):
    """L_N(t)=최저 저가(t−N..t−1), H_N(t)=최고 고가(t−N..t−1), ATR20(t−1). 창에 빈칸이 있으면 NaN."""
    Ls = pd.Series(l).shift(1).rolling(N, min_periods=N).min().to_numpy()
    Hs = pd.Series(h).shift(1).rolling(N, min_periods=N).max().to_numpy()
    cp = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - cp), np.abs(l - cp)))
    tr[np.isnan(h) | np.isnan(l) | np.isnan(cp)] = np.nan
    atr = pd.Series(tr).rolling(20, min_periods=20).mean().shift(1).to_numpy()
    return Ls, Hs, atr


def detect_events(o, h, l, c, N, gap=DEDUP):
    """이탈일(종가 < L_N) 중 사건 첫날 인덱스. 직전 이탈일 뒤 gap거래일 안의 이탈은 같은 사건(연쇄)."""
    Ls, Hs, atr = box_levels(o, h, l, c, N)
    ok = ~np.isnan(Ls) & ~np.isnan(atr) & ~np.isnan(c) & (atr > 0)
    brk = ok & (c < np.where(ok, Ls, -np.inf))
    ev, last = [], -10 ** 9
    for i in np.flatnonzero(brk):
        if i - last > gap:
            ev.append(i)
        last = i
    return np.array(ev, dtype=int), brk, Ls, Hs, atr


def short_outcome(o, c, i, Lv, h, cost=COST):
    """t=i 이탈, t+1 시가 공매도. t+1..t+h−1 종가 > L → 다음 시가 청산, 아니면 t+h 종가.
    반환 (수익, 사유, 청산 인덱스, 'open'|'close'). 자료 부족이면 (nan, None, -1, None)."""
    n = len(c)
    e = i + 1
    if e >= n or np.isnan(o[e]):
        return np.nan, None, -1, None
    entry = o[e]
    for k in range(i + 1, i + h):
        if k >= n or np.isnan(c[k]):
            return np.nan, None, -1, None
        if c[k] > Lv:
            if k + 1 >= n or np.isnan(o[k + 1]):
                return np.nan, None, -1, None
            return (entry - o[k + 1]) / entry - cost, "reclaim", k + 1, "open"
    k = i + h
    if k >= n or np.isnan(c[k]):
        return np.nan, None, -1, None
    return (entry - c[k]) / entry - cost, "time", k, "close"


def window_short(o, c, i, exit_i, exit_type, cost=COST):
    """대조 종목용: t+1 시가 진입 → exit_i의 시가/종가 청산 공매도 손익."""
    n = len(c)
    if i + 1 >= n or exit_i < 0 or exit_i >= n or np.isnan(o[i + 1]):
        return np.nan
    px = o[exit_i] if exit_type == "open" else c[exit_i]
    if np.isnan(px):
        return np.nan
    return (o[i + 1] - px) / o[i + 1] - cost


def classify(l, c, i, Lv, atr):
    """진짜/가짜/중간(BRIEF). 정해지지 않았는데 자료가 끝나면 None."""
    n = len(c)
    true_lvl, fake_lvl = Lv - 2 * atr, Lv - atr
    minlow = np.inf
    for k in range(i + 1, i + 11):
        if k >= n or np.isnan(c[k]) or np.isnan(l[k]):
            return None
        if l[k] <= true_lvl:
            return "진짜"
        minlow = min(minlow, l[k])
        if c[k] > Lv:
            return "가짜" if (k <= i + 5 and minlow > fake_lvl) else "중간"
    return "중간"


# ───────────────────────────── 특징 ─────────────────────────────
def _sma(x, n):
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def earnings_react_dates(cal, path=None):
    """실적 반응 거래일 인덱스 {ticker: np.array}. 16시 이후 발표 → 다음 거래일."""
    path = path or ANALYSIS / "box_trade/results/earnings_cache.csv"
    e = pd.read_csv(path)
    local = pd.to_datetime(e["earnings_dt"].str[:19])
    d = local.dt.normalize() + pd.to_timedelta((local.dt.hour >= 16).astype(int), unit="D")
    idx = np.searchsorted(np.asarray(cal.values, "datetime64[ns]"), d.values.astype("datetime64[ns]"), side="left")
    e = e.assign(ri=idx)
    # 달력 첫날 전 발표는 0번 날로 붙지 않게 뺀다(예전에는 searchsorted가 0으로 보냈다. 사건은 i ≥ 21이라 영향 없음).
    e = e[(e["ri"] < len(cal)) & (d.values >= np.datetime64(cal[0]))]
    return {t: np.unique(g["ri"].to_numpy()) for t, g in e.groupby("ticker")}


def _market_series(P):
    q = P.col("C", "QQQ")
    sx = P.col("C", "SOXX")
    m = {
        "qqq_above50": (q > _sma(q, 50)).astype(float),
        "qqq_above200": (q > _sma(q, 200)).astype(float),
        "soxx_above50": (sx > _sma(sx, 50)).astype(float),
        "vix": P.col("C", "^VIX"),
    }
    m["qqq_above50"][np.isnan(_sma(q, 50))] = np.nan
    m["qqq_above200"][np.isnan(_sma(q, 200))] = np.nan
    m["soxx_above50"][np.isnan(_sma(sx, 50))] = np.nan
    m["vix_ts"] = P.col("C", "^VIX") / P.col("C", "^VIX3M")
    st = [s for s in STOCKS if s in P.j]
    C = P.C[:, [P.j[s] for s in st]]
    S50 = pd.DataFrame(C).rolling(50, min_periods=50).mean().to_numpy()
    valid = ~np.isnan(S50) & ~np.isnan(C)
    below = ((C < S50) & valid).sum(1)
    m["breadth50"] = np.where(valid.sum(1) > 0, below / np.maximum(valid.sum(1), 1), np.nan)
    return m


def episodes(P, drop=0.05, look=5, gap=10, pre=5):
    """시장 전체 급락 묶음: [(id, start_i, end_i, first_flag_i, last_flag_i)] 와 날짜별 id 배열."""
    flag = np.zeros(len(P.cal), bool)
    for s in ("QQQ", "SOXX"):
        c, h = P.col("C", s), P.col("H", s)
        hh = pd.Series(h).shift(1).rolling(look, min_periods=look).max().to_numpy()
        flag |= np.nan_to_num(c <= (1 - drop) * hh, nan=0).astype(bool)
    fi = np.flatnonzero(flag)
    groups = []
    for i in fi:
        if groups and i - groups[-1][-1] <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    lab = np.full(len(P.cal), -1, int)
    rows = []
    for k, g in enumerate(groups):
        s0, s1 = max(g[0] - pre, 0), g[-1]
        lab[s0:s1 + 1] = k
        rows.append((k, s0, s1, g[0], g[-1], len(g)))
    return rows, lab


def build_events(P, N, earnings=None, seed=SEED, event_syms=None, etfs=None, semi=None):
    """사건표 한 개(N). 특징은 모두 t 종가까지의 정보만 쓴다."""
    etfs = set(ETFS if etfs is None else etfs)
    semi = (SEMI | SEMI_ETFS) if semi is None else set(semi)
    event_syms = [s for s in (EVENT_SYMS if event_syms is None else event_syms) if s in P.j]
    if earnings is None:
        earnings = earnings_react_dates(P.cal)
    mk = _market_series(P)
    _, eplab = episodes(P)
    qc = P.col("C", "QQQ")
    T = len(P.cal)
    # 1) 종목별 사건·상태
    per = {}
    for s in event_syms:
        o, h, l, c, v = (P.col(f, s) for f in "OHLCV")
        ev, brk, Ls, Hs, atr = detect_events(o, h, l, c, N)
        per[s] = (ev, brk, Ls, Hs, atr)
    kinds = {s: ("etf" if s in etfs else "stock") for s in event_syms}
    # 같은 날 새 사건 비율
    newcnt = {"stock": np.zeros(T), "etf": np.zeros(T)}
    valid = {"stock": np.zeros(T), "etf": np.zeros(T)}
    for s, (ev, brk, Ls, Hs, atr) in per.items():
        newcnt[kinds[s]][ev] += 1
        valid[kinds[s]] += ~np.isnan(Ls)
    rows = []
    for s, (ev, brk, Ls, Hs, atr) in per.items():
        o, h, l, c, v = (P.col(f, s) for f in "OHLCV")
        k = kinds[s]
        s50 = _sma(c, 50)
        vma = pd.Series(v).shift(1).rolling(20, min_periods=20).mean().to_numpy()
        er = earnings.get(s, np.array([], int)) if k == "stock" else np.array([], int)
        for i in ev:
            Lv, Hv, a = Ls[i], Hs[i], atr[i]
            r10, why, xi, xt = short_outcome(o, c, i, Lv, 10)
            r5 = short_outcome(o, c, i, Lv, 5)[0]
            r20 = short_outcome(o, c, i, Lv, 20)[0]
            lows = l[i - N:i]
            rng_ = h[i] - l[i]
            ed = er - i
            near = ed[np.abs(ed) <= 10]
            rows.append({
                "sym": s, "kind": k, "N": N, "t": P.cal[i], "i": i,
                "L": Lv, "H": Hv, "close_t": c[i], "atr20": a,
                "entry_open": o[i + 1] if i + 1 < T else np.nan,
                "short_ret_h10": r10, "short_ret_h5": r5, "short_ret_h20": r20,
                "exit_reason": why, "exit_date": P.cal[xi] if xi >= 0 else pd.NaT,
                "exit_i": xi, "exit_type": xt,
                "cls": classify(l, c, i, Lv, a),
                "box_w_atr": (Hv - Lv) / a,
                "slope50": (s50[i] / s50[i - 20] - 1) * 100 if i >= 20 else np.nan,
                "touches": int(np.sum(np.abs(lows - Lv) <= 0.5 * a)),
                "depth_atr": (Lv - c[i]) / a,
                "gap_break": float(o[i] < Lv) if not np.isnan(o[i]) else np.nan,
                "close_loc": (c[i] - l[i]) / rng_ if rng_ > 0 else 0.5,
                "vol_ratio": v[i] / vma[i] if vma[i] and vma[i] > 0 else np.nan,
                "ret5_pre": c[i - 1] / c[i - 6] - 1 if i >= 6 else np.nan,
                "ret20_pre": c[i - 1] / c[i - 21] - 1 if i >= 21 else np.nan,
                "earn_pm3": float(np.any(np.abs(ed) <= 3)) if k == "stock" else 0.0,
                "qqq_above50": mk["qqq_above50"][i], "qqq_above200": mk["qqq_above200"][i],
                "soxx_above50": mk["soxx_above50"][i],
                "cobreak_share": newcnt[k][i] / valid[k][i] if valid[k][i] else np.nan,
                "breadth50": mk["breadth50"][i], "vix": mk["vix"][i], "vix_ts": mk["vix_ts"][i],
                "semi": float(s in semi),
                "ret_t": c[i] / c[i - 1] - 1,
                "gap_pct": o[i] / c[i - 1] - 1,
                "atr_pct": a / c[i],
                "rs20_qqq": (c[i] / c[i - 20]) / (qc[i] / qc[i - 20]) - 1 if i >= 20 else np.nan,
                "earn_dist": float(near[np.argmin(np.abs(near))]) if len(near) else np.nan,
                "episode": int(eplab[i]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["t", "kind", "sym"]).reset_index(drop=True)
    df = add_excess(df, P, per, kinds, seed=seed)
    return df


def add_excess(df, P, per, kinds, seed=SEED, ret="short_ret_h10", out="excess_h10"):
    """같은 날 같은 kind 평균을 뺀 초과 손익. 그날 사건 1개면 비이탈 무작위 종목의 같은 보유 창 손익을 뺀다."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df[out] = np.nan
    df["excess_ref"] = None
    ok = df[ret].notna()
    for (t, k), g in df[ok].groupby(["t", "kind"], sort=True):
        if len(g) >= 2:
            df.loc[g.index, out] = g[ret] - g[ret].mean()
            df.loc[g.index, "excess_ref"] = "group"
            continue
        r = g.iloc[0]
        i = int(r["i"])
        cand = []
        for s, (ev, brk, Ls, Hs, atr) in per.items():
            if s == r["sym"] or kinds[s] != k:
                continue
            if np.isnan(Ls[i]) or brk[i]:
                continue
            o, c = P.col("O", s), P.col("C", s)
            w = window_short(o, c, i, int(r["exit_i"]), r["exit_type"])
            if not np.isnan(w):
                cand.append((s, w))
        if cand:
            s, w = cand[rng.integers(len(cand))]
            df.loc[g.index, out] = r[ret] - w
            df.loc[g.index, "excess_ref"] = f"random:{s}"
    return df


def build_all(force_data=False):
    P = Panel.from_long(build_daily(force=force_data))
    global _PANEL
    _PANEL = P
    er = earnings_react_dates(P.cal)
    for N in NS:
        ev = build_events(P, N, earnings=er)
        ev.to_parquet(RESULTS / f"events_N{N}.parquet")
    rows, _ = episodes(P)
    pd.DataFrame([{"episode": k, "start": P.cal[a].date(), "end": P.cal[b].date(),
                   "first_flag": P.cal[f].date(), "last_flag": P.cal[g].date(), "flag_days": n}
                  for k, a, b, f, g, n in rows]).to_csv(RESULTS / "episodes.csv", index=False, encoding="utf-8-sig")
    return P


def load_events(N):
    p = RESULTS / f"events_N{N}.parquet"
    if not p.exists():
        build_all()
    return pd.read_parquet(p)


def load_events_all(Ns=NS):
    """N=20·40·60 사건표를 이어 붙인다(칸이 N을 가로지를 때 evaluate_cells에 그대로 넣는다)."""
    return pd.concat([load_events(N) for N in Ns], ignore_index=True)


def in_period(df, period, col="t"):
    return (df[col] >= period[0]) & (df[col] <= period[1])


# ───────────────────────────── 판정 ─────────────────────────────
def _codes(x):
    return pd.factorize(pd.Series(x), sort=True)[0]


def cluster_diff(M, y, g):
    """M: n×C (1=켜짐, 0=꺼짐, NaN=밖), y: n, g: 날짜 코드. 반환 (n_on, n_off, mean_on, mean_off, diff, t)."""
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    on, off = (M == 1), (M == 0)
    n1, n0 = on.sum(0), off.sum(0)
    yy = y[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (on * yy).sum(0) / n1
        m0 = (off * yy).sum(0) / n0
        diff = m1 - m0
        G = int(g.max()) + 1 if len(g) else 0
        D = np.zeros((G, len(y)))
        D[g, np.arange(len(y))] = 1.0
        u1 = np.where(on, yy - m1, 0.0)
        u0 = np.where(off, yy - m0, 0.0)
        infl = D @ u1 / n1 - D @ u0 / n0
        Gc = ((D @ (on | off).astype(float)) > 0).sum(0)
        Nc = n1 + n0
        var = (infl ** 2).sum(0) * Gc / (Gc - 1) * (Nc - 1) / (Nc - 2)
        t = diff / np.sqrt(var)
    return n1, n0, m1, m0, diff, t


def cluster_mean(y, g):
    """날짜로 묶은 평균과 t(CR1)."""
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    y, g = y[ok], _codes(np.asarray(g)[ok])
    n = len(y)
    if n < 3:
        return (float(y.mean()) if n else np.nan), np.nan, n, len(np.unique(g))
    m = y.mean()
    s = np.bincount(g, weights=y - m) / n
    G = len(s)
    se = np.sqrt((s ** 2).sum() * G / (G - 1))
    return float(m), float(m / se) if se > 0 else np.nan, n, G


def _groups(dc, sc):
    groups = {}
    for s in np.unique(sc):
        idx = np.flatnonzero(sc == s)
        dd = dc[idx]
        u = np.unique(dd)
        groups[s] = (u, {d: idx[dd == d] for d in u})
    return groups


def _first_return(pi, D):
    """순열 pi(날짜 → 원천 날짜)를 날짜 부분집합 D(bool)로 줄인 첫 귀환 지도 f: D → D.
    f(d) = pi(d), pi(pi(d)), … 중 처음 D에 드는 날짜. f는 D 위의 전단사이고 pi가 균일 무작위면 f도 균일 무작위다
    (순환 표기에서 D 밖 원소를 지우는 것과 같다). D 밖 날짜는 −1."""
    f = np.full(len(pi), -1)
    ds = np.flatnonzero(D)
    x = pi[ds]
    bad = ~D[x]
    while bad.any():
        x[bad] = pi[x[bad]]
        bad = ~D[x]
    f[ds] = x
    return f


def _fill_map(pi, nd, n, groups, rng):
    """날짜 전역 순열 pi를 층마다 첫 귀환 지도로 줄여 행 지도를 만든다. pi(d)에 층 s 사건이 있으면 그 날짜를,
    없으면 pi(pi(d)), … 중 처음 s 사건이 있는 날짜를 원천으로 쓴다. pi(d)가 두 층에 모두 있으면 두 층이 같은
    원천 날짜를 받는다(같은 날 사건끼리 함께 이동). 원천 사건 수가 모자라면 복원 추출로 채운다."""
    out = np.arange(n)
    for s, (u, ev) in groups.items():
        pres = np.zeros(nd, bool)
        pres[u] = True
        f = _first_return(pi, pres)
        for d_t in u:
            tgt, src = ev[d_t], ev[f[d_t]]
            p = rng.permutation(src)
            if len(p) < len(tgt):
                p = np.r_[p, rng.choice(src, len(tgt) - len(p), replace=True)]
            out[tgt] = p[:len(tgt)]
    return out


def date_constant(M, dc):
    """칸마다 '같은 날 켜짐 표지가 모두 같은가'(시장 맥락처럼 날짜 고정 특징). NaN 행은 빼고 본다."""
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    nd = int(dc.max()) + 1 if len(dc) else 0
    mn = np.full((nd, M.shape[1]), np.inf)
    mx = np.full((nd, M.shape[1]), -np.inf)
    rr, cc = np.nonzero(~np.isnan(M))
    np.minimum.at(mn, (dc[rr], cc), M[rr, cc])
    np.maximum.at(mx, (dc[rr], cc), M[rr, cc])
    has = np.isfinite(mn)
    return np.all(~has | (mn == mx), axis=0)


class Permuter:
    """날짜 묶음 순열 은행. mode='iid': 날짜 전역 균일 순열 k개(BRIEF 판정용). mode='shift': 구간의 이탈 날짜를
    시간순으로 놓고 1..D−1칸 원형 이동(D−1개, 보조: 국면 지속성 보존).
    apply(M, r, dconst): 날짜 고정 칸(dconst)은 날짜 표지를 날짜째 옮긴다(그 칸의 모집단 날짜로 줄인 첫 귀환
    지도라 N·kind 층을 가로질러도 같은 날 사건이 같은 원천 날짜를 받는다. 모집단 행은 그대로).
    나머지 칸은 층(N·kind) 안 행 지도(_fill_map)로 옮긴다."""

    def __init__(self, dates, strata, k=200, rng=None, mode="iid"):
        rng = rng if rng is not None else np.random.default_rng(SEED)
        self.dc = _codes(dates)
        sc = _codes(strata)
        n = len(self.dc)
        self.nd = nd = int(self.dc.max()) + 1 if n else 0
        groups = _groups(self.dc, sc)
        if mode == "iid":
            pis = [rng.permutation(nd) for _ in range(k)]
        elif mode == "shift":
            pis = [(np.arange(nd) + s) % nd for s in range(1, nd)]
        else:
            raise ValueError(mode)
        self.mode = mode
        self.pis = np.array(pis, dtype=int).reshape(len(pis), nd)
        self.rowmaps = np.array([_fill_map(pi, nd, n, groups, rng) for pi in pis], dtype=int).reshape(len(pis), n)
        self._fr = {}

    def __len__(self):
        return len(self.pis)

    def _src(self, r, D):
        key = (r, D.tobytes())
        f = self._fr.get(key)
        if f is None:
            f = _first_return(self.pis[r], D)[self.dc]
            f[f < 0] = 0          # 모집단 밖 행(그 칸 NaN)은 아래에서 다시 NaN으로 덮는다
            self._fr[key] = f
        return f

    def prepare(self, M, dconst):
        """날짜 고정 칸의 날짜 표지·모집단 날짜 집합을 한 번만 계산(같은 M을 여러 회차에 옮길 때)."""
        cols = np.flatnonzero(dconst) if dconst is not None else np.array([], int)
        if not len(cols):
            return None
        Mc = M[:, cols]
        nan = np.isnan(Mc)
        lab = np.full((self.nd, len(cols)), np.nan)
        rr, cc = np.nonzero(~nan)
        lab[self.dc[rr], cc] = Mc[rr, cc]
        pres = ~np.isnan(lab)
        groups = {}
        for c in range(len(cols)):
            groups.setdefault(pres[:, c].tobytes(), (pres[:, c], []))[1].append(c)
        return cols, nan, lab, [(D, np.array(cs)) for D, cs in groups.values()]

    def apply(self, M, r, dconst=None, prep=None):
        out = M[self.rowmaps[r]]
        if prep is None:
            prep = self.prepare(M, dconst)
        if prep is None:
            return out
        cols, nan, lab, groups = prep
        newc = np.empty(nan.shape)
        for D, cs in groups:
            newc[:, cs] = lab[self._src(r, D)][:, cs]
        newc[nan] = np.nan
        out[:, cols] = newc
        return out


def block_maps(dates, strata, k, rng):
    """날짜 묶음 순열 행 지도 k개(k×n 원천 행 인덱스). 날짜 전역 순열 하나를 층마다 첫 귀환 지도로 쓴다.
    날짜 고정 칸까지 층을 가로질러 정확히 옮기려면 Permuter.apply를 쓴다(evaluate_cells가 그렇게 한다)."""
    return Permuter(dates, strata, k, rng, "iid").rowmaps


def shift_maps(dates, strata, rng):
    """보조 원형 이동 행 지도 D−1개(Permuter mode='shift'의 행 지도)."""
    return Permuter(dates, strata, rng=rng, mode="shift").rowmaps


def tercile_cells(df, features, front=FRONT, by=("N",), kind=None, tag=""):
    """특징 3분위 칸. 문턱은 앞 구간 사건 분포(그룹 by 안)로만 정한다. 값이 두 개 이하면 =1/=0 칸.
    반환 [(이름, Series 1/0/NaN)]. 그룹 밖·특징 NaN은 NaN(모집단 밖)."""
    pop = pd.Series(True, index=df.index) if kind is None else (df["kind"] == kind)
    fr = in_period(df, front)
    cells = []
    for key, g in df[pop].groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        kname = ",".join(f"{b}{v}" for b, v in zip(by, key))
        for f in features:
            x = g[f].astype(float)
            xf = x[fr.loc[g.index] & x.notna()]
            if xf.empty:
                continue
            u = np.unique(xf)
            base = pd.Series(np.nan, index=df.index)
            if len(u) <= 2:
                for val, nm in ((u.max(), "hi"), (u.min(), "lo")):
                    s = base.copy()
                    s.loc[g.index] = np.where(x.isna(), np.nan, (x == val).astype(float))
                    cells.append((f"{tag}{kname}|{f}={val:g}({nm})", s))
                    if len(u) == 1:
                        break
                continue
            q1, q2 = np.quantile(xf, [1 / 3, 2 / 3])
            for nm, on in ((f"≥{q2:.4g}(상위3분위)", x >= q2), (f"≤{q1:.4g}(하위3분위)", x <= q1)):
                s = base.copy()
                s.loc[g.index] = np.where(x.isna(), np.nan, on.astype(float))
                cells.append((f"{tag}{kname}|{f}{nm}", s))
    return cells


def _cell_matrix(df, cells):
    M = np.full((len(df), len(cells)), np.nan)
    for j, (_, s) in enumerate(cells):
        v = pd.Series(s).reindex(df.index)
        try:
            M[:, j] = v.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError):
            M[:, j] = [np.nan if pd.isna(x) else float(x) for x in v]
    M[(M != 0) & (M != 1)] = np.nan
    return M


def _date_count(M, g):
    """칸별 켜진 사건의 서로 다른 날짜 수."""
    G = int(g.max()) + 1 if len(g) else 0
    D = np.zeros((G, len(g)))
    D[g, np.arange(len(g))] = 1.0
    return ((D @ (M == 1).astype(float)) > 0).sum(0)


def _select(diff, n1, nd, top, min_on, min_dates):
    elig = (n1 >= min_on) & (nd >= min_dates) & np.isfinite(diff) & (diff > 0)
    order = np.argsort(-np.where(elig, diff, -np.inf), kind="stable")
    return order[:min(top, int(elig.sum()))], elig


def _perm_p(Mb_sel, yb, gb, bank, obs, dconst=None):
    """뒤 구간 한쪽 순열 p: (1 + #{perm diff ≥ obs}) / (1 + #{perm diff 유한}).
    bank는 Permuter(날짜 고정 칸은 dconst로 날짜째 이동) 또는 행 지도 배열(예전 방식).
    관측 차이가 NaN(켜짐 또는 꺼짐 0건)이면 p도 NaN. 순열에서 NaN이 난 회차는 분모에서 뺀다
    (예전에는 NaN 관측도 p = 1/201로, NaN 순열 회차는 '관측보다 작음'으로 셌다)."""
    if Mb_sel.shape[1] == 0:
        return np.array([])
    obs = np.asarray(obs, float)
    cnt = np.zeros(Mb_sel.shape[1])
    fin = np.zeros(Mb_sel.shape[1])
    isp = isinstance(bank, Permuter)
    prep = bank.prepare(Mb_sel, dconst) if isp else None
    for r in range(len(bank)):
        Mp = bank.apply(Mb_sel, r, prep=prep) if isp else Mb_sel[bank[r]]
        d = cluster_diff_fast(Mp, yb)
        ok = np.isfinite(d)
        fin += ok
        cnt += ok & (d >= obs - 1e-15)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (1 + cnt) / (1 + fin)
    p[~np.isfinite(obs)] = np.nan
    return p


def cluster_diff_fast(M, y):
    on, off = (M == 1), (M == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (on * y[:, None]).sum(0) / on.sum(0) - (off * y[:, None]).sum(0) / off.sum(0)


def evaluate_cells(df, cells, front=FRONT, back=BACK, aux=AUX, top=10, min_on=15, min_dates=8,
                   nperm=200, seed=SEED, y="short_ret_h10", y_ex="excess_h10", strata=("N", "kind"),
                   nnull=200, null=True, regime_share=0.5, date_col="t", episode_col="episode"):
    """BRIEF 판정. cells = [(이름, Series 1/0/NaN 또는 bool)]. 반환 (칸 표 DataFrame, 요약 dict)."""
    rng = np.random.default_rng(seed)
    df = df[df[y].notna()]
    names = [n for n, _ in cells]
    M = _cell_matrix(df, cells)
    yv = df[y].to_numpy(float)
    yex = df[y_ex].to_numpy(float) if y_ex in df else np.full(len(df), np.nan)
    strata = [c for c in (strata or ()) if c in df]
    st = df[strata].astype(str).agg("|".join, axis=1).to_numpy() if strata else np.zeros(len(df))
    dt = df[date_col].to_numpy()
    ep = df[episode_col].to_numpy() if episode_col in df else np.full(len(df), -1)
    per = {nm: in_period(df, p, date_col).to_numpy() for nm, p in (("f", front), ("b", back), ("a", aux))}

    def stats(mask, yy):
        g = _codes(dt[mask])
        Mm = M[mask]
        okm = ~np.isnan(yy[mask])
        Mm = np.where(okm[:, None], Mm, np.nan)
        n1, n0, m1, m0, d, t = cluster_diff(Mm, np.nan_to_num(yy[mask]), g)
        return n1, n0, m1, m0, d, t, _date_count(Mm, g)

    tab = pd.DataFrame({"cell": names})
    for p in ("f", "b", "a"):
        n1, n0, m1, m0, d, t, nd = stats(per[p], yv)
        tab[f"{p}_n_on"], tab[f"{p}_n_off"], tab[f"{p}_dates_on"] = n1, n0, nd
        tab[f"{p}_mean_on"], tab[f"{p}_mean_off"], tab[f"{p}_diff"], tab[f"{p}_t"] = m1, m0, d, t
        _, _, mx1, _, dx, tx, _ = stats(per[p], yex)
        tab[f"{p}_ex_mean_on"], tab[f"{p}_ex_diff"], tab[f"{p}_ex_t"] = mx1, dx, tx
    sel, elig = _select(tab["f_diff"].to_numpy(), tab["f_n_on"].to_numpy(), tab["f_dates_on"].to_numpy(),
                        top, min_on, min_dates)
    tab["eligible"] = elig
    tab["rank"] = np.nan
    tab.loc[sel, "rank"] = np.arange(1, len(sel) + 1)
    tab["selected"] = tab["rank"].notna()
    # 뒤 구간 순열(BRIEF: 날짜 묶음 i.i.d. 순열) + 보조(원형 이동, 국면 지속성 보존)
    mb = per["b"]
    yb, gb = yv[mb], _codes(dt[mb])
    Mb = M[mb]
    dconst_b = date_constant(Mb, gb)
    bank = Permuter(dt[mb], st[mb], nperm, rng, "iid")
    shift_b = Permuter(dt[mb], st[mb], rng=rng, mode="shift")
    tab["date_const"] = dconst_b
    tab["b_p"] = np.nan
    tab["b_p_shift"] = np.nan
    if len(sel):
        obs = tab.loc[sel, "b_diff"].to_numpy()
        tab.loc[sel, "b_p"] = _perm_p(Mb[:, sel], yb, gb, bank, obs, dconst_b[sel])
        tab.loc[sel, "b_p_shift"] = _perm_p(Mb[:, sel], yb, gb, shift_b, obs, dconst_b[sel])
    tab["a_same_dir"] = np.sign(tab["a_diff"]) == np.sign(tab["f_diff"])
    tab["pass"] = tab["selected"] & (tab["b_diff"] > 0) & (tab["b_t"] >= 2) & (tab["b_p"] < 0.05) & (tab["b_mean_on"] > 0)
    # 보조(판정 아님): 원형 이동 p까지 < 0.05
    tab["pass_shift"] = tab["pass"] & (tab["b_p_shift"] < 0.05)
    # 한 국면 의존
    tab["regime_top_ep"], tab["regime_top_share"], tab["one_regime"] = np.nan, np.nan, False
    epb = ep[mb]
    for j in sel:
        on = Mb[:, j] == 1
        tot = yb[on].sum()
        if tot <= 0:
            continue
        s = pd.Series(yb[on]).groupby(epb[on]).sum()
        s = s[s.index >= 0]
        if len(s):
            tab.loc[j, "regime_top_ep"] = int(s.idxmax())
            tab.loc[j, "regime_top_share"] = float(s.max() / tot)
            tab.loc[j, "one_regime"] = bool(s.max() / tot > regime_share)
    n_pass = int(tab["pass"].sum())
    summ = {
        "n_cells": len(cells), "n_eligible": int(elig.sum()), "n_selected": len(sel),
        "selected": [names[j] for j in sel], "n_pass": n_pass,
        "passed": tab.loc[tab["pass"], "cell"].tolist(),
        "passed_one_regime": tab.loc[tab["pass"] & tab["one_regime"], "cell"].tolist(),
        "n_pass_shift": int(tab["pass_shift"].sum()),
        "passed_shift": tab.loc[tab["pass_shift"], "cell"].tolist(),
        "events": {p: int(per[p].sum()) for p in per},
        "dates": {p: int(len(np.unique(dt[per[p]]))) for p in per},
        "params": dict(front=front, back=back, aux=aux, top=top, min_on=min_on, min_dates=min_dates,
                       nperm=nperm, nnull=nnull, seed=seed, y=y, n_shift=len(shift_b)),
    }
    if null and nnull > 0:
        mf = per["f"]
        yf_, gf = yv[mf], _codes(dt[mf])
        Mf = M[mf]
        dconst_f = date_constant(Mf, gf)
        nf = Permuter(dt[mf], st[mf], nnull, rng, "iid")
        nb = Permuter(dt[mb], st[mb], nnull, rng, "iid")
        sf = Permuter(dt[mf], st[mf], rng=rng, mode="shift")
        pairs = (rng.integers(len(sf), size=nnull), rng.integers(len(shift_b), size=nnull))             if len(sf) and len(shift_b) else None

        def one(Mf_r, Mb_full_r):
            _, _, _, _, d_f, _ = cluster_diff(Mf_r, yf_, gf)
            s_r, _ = _select(d_f, (Mf_r == 1).sum(0), _date_count(Mf_r, gf), top, min_on, min_dates)
            if not len(s_r):
                return 0
            Mb_r = Mb_full_r[:, s_r]
            _, _, m1b, _, d_b, t_b = cluster_diff(Mb_r, yb, gb)
            p_b = _perm_p(Mb_r, yb, gb, bank, d_b, dconst_b[s_r])
            with np.errstate(invalid="ignore"):
                return int(np.sum((d_b > 0) & (t_b >= 2) & (p_b < 0.05) & (m1b > 0)))

        def pack(dist):
            return {"mean": float(dist.mean()), "q50": float(np.quantile(dist, 0.5)),
                    "q90": float(np.quantile(dist, 0.9)), "q95": float(np.quantile(dist, 0.95)),
                    "max": int(dist.max()), "p_value": float((1 + np.sum(dist >= n_pass)) / (1 + len(dist))),
                    "dist": dist.tolist()}

        # BRIEF 귀무: 앞·뒤 표지를 날짜 묶음째 i.i.d.로 섞고 고르기·확인을 반복
        pf, pb = nf.prepare(Mf, dconst_f), nb.prepare(Mb, dconst_b)
        dist = np.array([one(nf.apply(Mf, r, prep=pf), nb.apply(Mb, r, prep=pb)) for r in range(nnull)])
        summ["null"] = pack(dist)
        # 보조 귀무: 앞·뒤 표지를 원형 이동(국면 지속성 보존)한 뒤 같은 고르기·확인(통과 기준은 BRIEF 그대로)
        if pairs is not None:
            qf, qb = sf.prepare(Mf, dconst_f), shift_b.prepare(Mb, dconst_b)
            dist_s = np.array([one(sf.apply(Mf, a, prep=qf), shift_b.apply(Mb, b, prep=qb))
                               for a, b in zip(*pairs)])
            summ["null_shift"] = pack(dist_s)
    return tab, summ


# ───────────────────────────── 기준선 ─────────────────────────────
def _fmt(x, pct=True, nd=2, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "못 구함"
    return (f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%") if pct else f"{x:.{nd}f}"


def baseline_report(path=None):
    path = path or RESULTS / "baseline.md"
    man = json.loads((RESULTS / "data_manifest.json").read_text(encoding="utf-8"))
    L = ["# 기준선: 모든 박스 하단 이탈 공매도(조건 없음)", "",
         f"- 자료: 요청 {man['requested']}종목, 받음 {man['received']}(AI 미국 종목 {man['stocks']}/{len(STOCKS)}, "
         f"ETF {man['etfs']}/{len(ETFS)}), 빠짐 {man['missing'] or '없음'}, "
         f"{START}~{END}(10/13까지 못 받은 종목: {man['not_to_end'] or '없음'}).",
         f"- 늦게 상장(첫 날짜 > 2022-06-03): {len(man['late_start'])}종목 — "
         + ", ".join(f"{k} {v}" for k, v in sorted(man['late_start'].items(), key=lambda kv: kv[1])),
         "- 진입 t+1 시가 공매도, 되돌림 손절(종가 > L → 다음 시가) 또는 t+h 종가, 비용 왕복 0.10%(대차 비용 없음).",
         "- excess = 같은 날 같은 kind 사건 평균을 뺀 값(그날 1건이면 비이탈 무작위 종목의 같은 보유 창). "
         "날짜 묶음 t는 이탈일로 묶은 CR1.",
         f"- 구간: 앞 {FRONT[0]}~{FRONT[1]}, 뒤 {BACK[0]}~{BACK[1]}, 보조 2025. 사건 h20은 t+20 자료가 있어야 잡힌다.", ""]
    for N in NS:
        ev = load_events(N)
        L += [f"## N = {N}", "",
              "| 유형 | 구간 | 사건 | 서로 다른 날짜 | h10 평균 | 날짜 묶음 t | excess h10 | h5 평균 | h20 평균 | 진짜 | 가짜 | 중간 | 되돌림 청산 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for kind, kn in (("stock", "종목"), ("etf", "ETF")):
            for pn, p in (("앞(2026-01~05)", FRONT), ("뒤(2026-06~09)", BACK), ("보조 2025", AUX)):
                g = ev[(ev["kind"] == kind) & in_period(ev, p)]
                gg = g[g["short_ret_h10"].notna()]
                m, t, n, G = cluster_mean(gg["short_ret_h10"], gg["t"])
                ex = gg["excess_h10"].mean() if len(gg) else np.nan
                c = g["cls"].dropna()
                sh = {k: (c == k).mean() if len(c) else np.nan for k in ("진짜", "가짜", "중간")}
                rc = (gg["exit_reason"] == "reclaim").mean() if len(gg) else np.nan
                L.append(f"| {kn} | {pn} | {n} | {G} | {_fmt(m)} | {_fmt(t, False)} | {_fmt(ex)} | "
                         f"{_fmt(g['short_ret_h5'].mean())} | {_fmt(g['short_ret_h20'].mean())} | "
                         f"{_fmt(sh['진짜'], nd=0, sign=False)} | {_fmt(sh['가짜'], nd=0, sign=False)} | "
                         f"{_fmt(sh['중간'], nd=0, sign=False)} | {_fmt(rc, nd=0, sign=False)} |")
        L.append("")
    ep = pd.read_csv(RESULTS / "episodes.csv")
    ep = ep[ep["end"] >= "2025-01-01"]
    L += ["## 시장 급락 묶음(episode, 2025 이후)", "", "| id | 창 시작 | 창 끝 | 표시일 수 |", "|---:|---|---|---:|"]
    L += [f"| {r.episode} | {r.start} | {r.end} | {r.flag_days} |" for r in ep.itertuples()]
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build_all(force_data="--refetch" in sys.argv)
        print(baseline_report())
