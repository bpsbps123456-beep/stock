"""SOXL 긴급탈출·바닥 연구 공통 채점 틀 (2026-11-20).

모든 갈래가 이 파일의 자료·지표·체결·채점·대조군을 그대로 쓴다. 신호 정의만 갈래가 만든다.

자료
- cache/SOXL_1m.parquet: Alpaca SIP 1분봉, 04:00~20:00 ET, 분할 반영. 트레이딩뷰 연장시간 차트처럼
  지표는 연장시간 봉까지 이어서 계산한다(사용자 차트가 프리장을 켜 둔 상태).
- 3분봉 등 상위봉은 자정 기준 고정 칸(트레이딩뷰와 같은 정렬)으로 1분봉을 모아 만든다. 빈 칸은 만들지 않는다.

체결 규약(미래 정보 차단)
- 신호는 봉이 '완성된 뒤'에만 안다. 상위봉 신호의 체결은 그 봉 완성 시각 이후 첫 1분봉의 시가다.
- 1분봉 신호도 다음 1분봉 시가에 체결한다. 비용 0(사용자 수수료 0). 민감도는 cost 인자로 편도 차감.
- 주 평가는 정규장(09:30~16:00 ET)에서 체결되는 신호만 본다. 앞으로 창은 그날 16:00 마지막 정규장 봉에서 자른다.

구간
- TRAIN 2026-10-15~11-04(15세션, 고르는 구간), TEST 2026-11-05~11-19(11세션, 확인 구간, 11-19는 진행 중).
- PRE 2026-08-04~10-14는 장세 위험 한 줄용 보조이며 선택에 쓰지 않는다.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "cache", "SOXL_1m.parquet")
ET = "America/New_York"

TRAIN = ("2026-10-15", "2026-11-04")
TEST = ("2026-11-05", "2026-11-19")
PRE = ("2026-08-04", "2026-10-14")
REG_OPEN, REG_CLOSE = 9 * 60 + 30, 16 * 60

_CACHE: dict = {}


# ───────────────────────── 자료 ─────────────────────────
def load_1m() -> pd.DataFrame:
    """1분봉 + 보조 열. 열: Open High Low Close Volume VWAP(봉 내 VWAP) date mod reg."""
    if "1m" in _CACHE:
        return _CACHE["1m"]
    d = pd.read_parquet(CACHE)[["Open", "High", "Low", "Close", "Volume", "VWAP"]].copy()
    d = d[~d.index.duplicated()].sort_index()
    d["date"] = pd.to_datetime(d.index.date)
    d["mod"] = d.index.hour * 60 + d.index.minute
    d["reg"] = (d["mod"] >= REG_OPEN) & (d["mod"] < REG_CLOSE)
    _CACHE["1m"] = d
    return d


def resample(minutes: int) -> pd.DataFrame:
    """자정 기준 고정 칸으로 모은 상위봉. index=칸 시작, 'end'=완성 시각(칸 시작+minutes).
    'reg'는 칸 시작이 정규장 안인지. 정규장 마지막 칸은 16:00에서 끝나도록 칸이 정렬된다(3·5·15·30분 모두 390의 약수 아님 주의:
    3분은 09:30 시작 칸이 정렬되어 16:00에 정확히 끝난다)."""
    key = f"r{minutes}"
    if key in _CACHE:
        return _CACHE[key]
    d = load_1m()
    if minutes == 1:
        out = d.copy()
        out["end"] = out.index + pd.Timedelta(minutes=1)
        _CACHE[key] = out
        return out
    g = d.groupby(d.index.floor(f"{minutes}min"))
    out = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
        "Close": g["Close"].last(), "Volume": g["Volume"].sum(),
    })
    pv = (d["VWAP"] * d["Volume"]).groupby(d.index.floor(f"{minutes}min")).sum()
    out["VWAP"] = (pv / out["Volume"].replace(0, np.nan)).fillna(out["Close"])
    out.index = pd.DatetimeIndex(out.index)
    out["end"] = out.index + pd.Timedelta(minutes=minutes)
    out["date"] = pd.to_datetime(out.index.date)
    out["mod"] = out.index.hour * 60 + out.index.minute
    out["reg"] = (out["mod"] >= REG_OPEN) & (out["mod"] < REG_CLOSE)
    _CACHE[key] = out
    return out


# ───────────────────────── 지표(파인과 같은 정의) ─────────────────────────
def ema(x: pd.Series, n: int) -> pd.Series:
    """ta.ema: alpha=2/(n+1), 첫 값은 n개 SMA로 시작."""
    x = x.astype(float)
    a = 2.0 / (n + 1)
    v = x.to_numpy()
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        out[n - 1] = v[:n].mean()
        for i in range(n, len(v)):
            out[i] = a * v[i] + (1 - a) * out[i - 1]
    return pd.Series(out, index=x.index)


def rma(x: pd.Series, n: int) -> pd.Series:
    """ta.rma(Wilder): alpha=1/n, 첫 값은 n개 SMA."""
    x = x.astype(float)
    v = x.to_numpy()
    out = np.full(len(v), np.nan)
    start = np.argmax(~np.isnan(v)) if np.any(~np.isnan(v)) else len(v)
    if len(v) - start >= n:
        out[start + n - 1] = np.nanmean(v[start:start + n])
        for i in range(start + n, len(v)):
            out[i] = (v[i] + (n - 1) * out[i - 1]) / n
    return pd.Series(out, index=x.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    ch = close.diff()
    up = rma(ch.clip(lower=0), n)
    dn = rma((-ch).clip(lower=0), n)
    r = 100 - 100 / (1 + up / dn)
    r[(dn == 0) & up.notna()] = 100.0
    return r


def stoch_rsi(close: pd.Series, k=3, d=3, n_rsi=14, n_st=14):
    """트레이딩뷰 기본 Stoch RSI 3 3 14 14 → (K, D)."""
    r = rsi(close, n_rsi)
    lo, hi = r.rolling(n_st).min(), r.rolling(n_st).max()
    st = 100 * (r - lo) / (hi - lo)
    K = st.rolling(k).mean()
    D = K.rolling(d).mean()
    return K, D


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """트레이딩뷰 VWAP(앵커=세션, 연장시간 차트면 04:00부터) — hlc3 가중."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = (tp * df["Volume"]).groupby(df["date"]).cumsum()
    vv = df["Volume"].groupby(df["date"]).cumsum()
    return pv / vv.replace(0, np.nan)


# ───────────────────────── 체결 위치 ─────────────────────────
def exec_index(sig: pd.Series, bars: pd.DataFrame) -> np.ndarray:
    """상위봉(또는 1분봉) bool 신호 → 1분봉 체결 위치(정수). 봉 완성 시각 이후 첫 1분봉, 같은 날짜만.
    반환: 신호가 있는 봉마다 1분봉 정수 위치(없으면 -1)."""
    d1 = load_1m()
    t1 = d1.index.values
    s = sig.reindex(bars.index).fillna(False).astype(bool)
    ends = bars.loc[s.values, "end"].values
    dates = bars.loc[s.values, "date"].values
    pos = np.searchsorted(t1, ends, side="left")
    ok = pos < len(t1)
    pos = np.where(ok, pos, -1)
    same = np.zeros(len(pos), dtype=bool)
    same[ok] = d1["date"].values[pos[ok]] == dates[ok]
    return np.where(same, pos, -1)


def in_range(dates: np.ndarray, rng) -> np.ndarray:
    a, b = np.datetime64(rng[0]), np.datetime64(rng[1])
    return (dates >= a) & (dates <= b)


def dedup(pos: np.ndarray, cooldown: int = 30) -> np.ndarray:
    """같은 날 cooldown분 안의 겹친 신호는 첫 신호만 남긴다(시각 기준)."""
    d1 = load_1m()
    pos = np.sort(pos[pos >= 0])
    keep, last_t, last_d = [], None, None
    tt = d1.index.values
    dd = d1["date"].values
    for p in pos:
        if last_t is not None and dd[p] == last_d and (tt[p] - last_t) < np.timedelta64(cooldown, "m"):
            continue
        keep.append(p)
        last_t, last_d = tt[p], dd[p]
    return np.array(keep, dtype=int)


# ───────────────────────── 앞으로 경로 지표 ─────────────────────────
def _fwd_tables():
    """모든 1분봉에 대해, 그 봉 시가 체결 기준 앞으로 경로 지표를 한 번 계산해 둔다(정규장 끝에서 자름)."""
    if "fwd" in _CACHE:
        return _CACHE["fwd"]
    d = load_1m()
    n = len(d)
    O, H, L, C = (d[c].to_numpy(float) for c in ["Open", "High", "Low", "Close"])
    dates = d["date"].values
    mod = d["mod"].to_numpy()
    # 각 봉이 속한 날의 마지막 정규장 봉 위치
    reg = d["reg"].to_numpy()
    last_reg = np.full(n, -1)
    df_idx = pd.Series(np.arange(n), index=d.index)
    lr = df_idx[reg].groupby(dates[reg]).max()
    m = pd.Series(lr.values, index=pd.to_datetime(lr.index))
    last_reg = pd.Series(dates).map(m).fillna(-1).astype(int).to_numpy()
    t = d.index.values
    out = {}
    for h in (15, 30, 60, 120):
        # 체결 봉 i 기준, i+h분 시각 직전 봉의 종가(정규장 끝에서 자름)
        tgt = t + np.timedelta64(h, "m")
        j = np.searchsorted(t, tgt, side="left") - 1
        j = np.minimum(j, np.where(last_reg >= 0, last_reg, j))
        out[f"r{h}"] = np.where(last_reg >= 0, C[np.clip(j, 0, n - 1)] / O - 1, np.nan)
        out[f"j{h}"] = j
    out["rclose"] = np.where(last_reg >= 0, C[np.clip(last_reg, 0, n - 1)] / O - 1, np.nan)
    # 60분 안 최대 역행·순행(창 안 최저·최고)
    for h in (30, 60, 120):
        j = out[f"j{h}"]
        mn = np.full(n, np.nan)
        mx = np.full(n, np.nan)
        # 슬라이딩 최소·최대: 길이가 거의 같으므로 단순 루프 대신 누적 블록 계산
        Ls, Hs = pd.Series(L), pd.Series(H)
        rmin = Ls[::-1].rolling(h, min_periods=1).min()[::-1].to_numpy()
        rmax = Hs[::-1].rolling(h, min_periods=1).max()[::-1].to_numpy()
        # rolling 창은 위치 기준 h개 봉 — 정규장 끝을 넘는 봉은 아래서 잘라낸다
        for i in range(n):
            if last_reg[i] < 0 or j[i] < i:
                continue
            if j[i] - i + 1 == h:
                mn[i], mx[i] = rmin[i], rmax[i]
            else:
                mn[i], mx[i] = L[i:j[i] + 1].min(), H[i:j[i] + 1].max()
        out[f"mae{h}"] = mn / O - 1
        out[f"mfe{h}"] = mx / O - 1
    out["last_reg"] = last_reg
    out["avail_end"] = t[-1]
    # 진행 중인 마지막 날: 자료 끝을 넘는 창은 잘린 값이므로 비운다(대조군 오염 방지, 2026-11-20 검증자 지적).
    if (mod[-1]) < REG_CLOSE - 1:
        today = dates == dates[-1]
        out["rclose"][today] = np.nan
        for h in (15, 30, 60, 120):
            cut = today & ((t + np.timedelta64(h, "m")) > t[-1])
            out[f"r{h}"][cut] = np.nan
            if h in (30, 60, 120):
                out[f"mae{h}"][cut] = np.nan
                out[f"mfe{h}"][cut] = np.nan
    _CACHE["fwd"] = out
    return out


def first_passage(pos: np.ndarray, up: float, dn: float) -> np.ndarray:
    """체결가 기준 +up 먼저(1) / -dn 먼저(0) / 둘 다 같은 봉(0.5) / 정규장 끝까지 미도달(nan)."""
    d = load_1m()
    H, L, O = d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Open"].to_numpy(float)
    f = _fwd_tables()
    res = np.full(len(pos), np.nan)
    for k, i in enumerate(pos):
        e = f["last_reg"][i]
        if e < i:
            continue
        P = O[i]
        hu = np.nonzero(H[i:e + 1] >= P * (1 + up))[0]
        hd = np.nonzero(L[i:e + 1] <= P * (1 - dn))[0]
        a = hu[0] if len(hu) else 10**9
        b = hd[0] if len(hd) else 10**9
        if a == b == 10**9:
            continue
        res[k] = 1.0 if a < b else (0.0 if b < a else 0.5)
    return res


def event_metrics(pos: np.ndarray, cost: float = 0.0, fp=(0.015, 0.015)) -> pd.DataFrame:
    """체결 위치마다 경로 지표 표. cost는 편도 비용(수익률에서 뺌, 방향은 호출자가 해석)."""
    d = load_1m()
    f = _fwd_tables()
    pos = np.asarray(pos, dtype=int)
    pos = pos[pos >= 0]
    t = d.index.values[pos]
    rows = pd.DataFrame({
        "pos": pos, "time": d.index[pos], "date": d["date"].values[pos], "mod": d["mod"].values[pos],
        "reg": d["reg"].values[pos], "px": d["Open"].values[pos],
    })
    for k in ("r15", "r30", "r60", "r120", "rclose", "mae30", "mfe30", "mae60", "mfe60", "mae120", "mfe120"):
        rows[k] = f[k][pos]
    # 진행 중인 마지막 날의 잘린 창은 _fwd_tables에서 이미 비웠다(자료 마지막 날짜 기준).
    # 옛 코드는 '사건이 있는 마지막 날'을 진행 중인 날로 잡아 11-18 사건의 rclose를 지우는 버그가 있었다(2026-11-20 수정).
    rows["fp"] = first_passage(pos, fp[0], fp[1])
    return rows


def baseline_positions(rng, mask: np.ndarray | None = None) -> np.ndarray:
    """구간 안 정규장 1분봉 전부(대조군). mask(1분봉 길이 bool)로 상황 조건을 걸 수 있다."""
    d = load_1m()
    sel = d["reg"].to_numpy() & in_range(d["date"].values, rng)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    return np.nonzero(sel)[0]


def day_matched(ev: pd.DataFrame, col: str, window: int = 30) -> np.ndarray:
    """각 사건과 같은 날 ±window분 정규장 봉들의 같은 지표 평균(가까운 시각 무작위 체결 대조)."""
    d = load_1m()
    f = _fwd_tables()
    vals = f[col] if col in f else None
    reg = d["reg"].to_numpy()
    t = d.index.values
    out = np.full(len(ev), np.nan)
    for k, (p, dt) in enumerate(zip(ev["pos"].values, ev["date"].values)):
        lo = np.searchsorted(t, t[p] - np.timedelta64(window, "m"))
        hi = np.searchsorted(t, t[p] + np.timedelta64(window, "m"), side="right")
        idx = np.arange(lo, hi)
        idx = idx[reg[idx] & (d["date"].values[idx] == dt)]
        if len(idx):
            out[k] = np.nanmean(vals[idx])
    return out


def summarize(ev: pd.DataFrame, rng, kind: str, ctx_mask: np.ndarray | None = None, cost: float = 0.0) -> dict:
    """한 신호의 구간 성적. kind='exit'면 '체결 뒤 더 빠질수록 좋음'(benefit = -r), 'bottom'이면 r 그대로.
    대조: all(구간 정규장 전체 봉), ctx(같은 상황 조건 봉), dm(같은 날 ±30분 봉)."""
    e = ev[in_range(ev["date"].values, rng) & ev["reg"].values].copy()
    sgn = -1.0 if kind == "exit" else 1.0
    res = {"n": int(len(e)), "days": int(e["date"].nunique()) if len(e) else 0}
    if not len(e):
        return res
    ndays = len(np.unique(load_1m().loc[in_range(load_1m()["date"].values, rng) & load_1m()["reg"].values, "date"]))
    res["per_day"] = round(len(e) / max(ndays, 1), 2)
    f = _fwd_tables()
    base_all = baseline_positions(rng)
    base_ctx = baseline_positions(rng, ctx_mask) if ctx_mask is not None else None
    for h in ("r30", "r60", "rclose"):
        v = sgn * e[h].to_numpy() - cost * (1 if kind == "bottom" else 0)
        res[f"{h}_mean%"] = round(100 * np.nanmean(v), 3)
        res[f"{h}_med%"] = round(100 * np.nanmedian(v), 3)
        b = sgn * f[h][base_all]
        res[f"{h}_x_all%"] = round(100 * (np.nanmean(v) - np.nanmean(b)), 3)
        if base_ctx is not None and len(base_ctx):
            bc = sgn * f[h][base_ctx]
            res[f"{h}_x_ctx%"] = round(100 * (np.nanmean(v) - np.nanmean(bc)), 3)
        dm = sgn * day_matched(e, h)
        xd = v - dm
        res[f"{h}_x_dm%"] = round(100 * np.nanmean(xd), 3)
        # 날짜 묶음 t: 날짜별 평균 초과(dm)의 평균/표준오차
        g = pd.Series(xd, index=e["date"].values).groupby(level=0).mean().dropna()
        res[f"{h}_x_dm_t"] = round(float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), 2) if len(g) > 2 and g.std(ddof=1) > 0 else None
        res[f"{h}_win%"] = round(100 * np.nanmean(v > 0), 1)
    res["mae60_med%"] = round(100 * np.nanmedian(e["mae60"]), 3)
    res["mfe60_med%"] = round(100 * np.nanmedian(e["mfe60"]), 3)
    fpv = e["fp"].to_numpy()
    good = (1 - fpv) if kind == "exit" else fpv
    res["fp_good%"] = round(100 * np.nanmean(good), 1)
    ck = ("bfp", rng)
    if ck not in _CACHE:
        _CACHE[ck] = first_passage(base_all[::7], 0.015, 0.015)
    bfp = _CACHE[ck]
    res["fp_good_base%"] = round(100 * np.nanmean((1 - bfp) if kind == "exit" else bfp), 1)
    return res


# ───────────────────────── 보유 모의(손절 규칙 평가) ─────────────────────────
def hold_sim(exit_pos: np.ndarray, rng, entry_mask: np.ndarray | None = None, step: int = 5,
             n_perm: int = 20, seed: int = 7) -> dict:
    """정규장 매 step분마다 가상 진입(entry_mask로 상황 제한) → 진입 뒤 첫 탈출 체결(exit_pos, 같은 날)
    또는 16:00 종가. 비교: (a) 종가까지 보유, (b) 같은 보유시간 분포를 날짜 안에서 섞은 무작위 탈출.
    상태형 규칙은 진입 다음 봉부터 첫 True가 탈출이다. 진입 기준 규칙(진입 뒤 고점 추적 등)은 갈래가 직접 시뮬레이션한다."""
    d = load_1m()
    O, C = d["Open"].to_numpy(float), d["Close"].to_numpy(float)
    f = _fwd_tables()
    ent = baseline_positions(rng, entry_mask)
    ent = ent[(d["mod"].values[ent] - REG_OPEN) % step == 0]
    ent = ent[ent < f["last_reg"][ent]]
    # 오늘 진행 중이면 오늘 진입 제외(종가 미확정)
    if (d.index[-1].hour * 60 + d.index[-1].minute) < REG_CLOSE:
        ent = ent[d["date"].values[ent] != d["date"].values[-1]]
    xs = np.sort(np.asarray(exit_pos)[np.asarray(exit_pos) >= 0])
    k = np.searchsorted(xs, ent + 1)          # 진입 봉 다음 봉부터
    ex = np.where(k < len(xs), xs[np.minimum(k, len(xs) - 1)], 10**12)
    lr = f["last_reg"][ent]
    use = ex <= lr
    exit_px = np.where(use, O[np.minimum(ex, len(O) - 1)], C[lr])
    pnl_rule = exit_px / O[ent] - 1
    pnl_hold = C[lr] / O[ent] - 1
    dur = np.where(use, ex - ent, lr - ent + 1)
    # 무작위 탈출: 같은 날 안에서 보유시간을 섞는다(규칙이 시점을 잘 고르는지)
    rng_ = np.random.default_rng(seed)
    dates = d["date"].values[ent]
    pr = []
    for _ in range(n_perm):
        dd = dur.copy()
        for dt in np.unique(dates):
            m = np.nonzero(dates == dt)[0]
            dd[m] = rng_.permutation(dd[m])
        e2 = ent + dd
        px2 = np.where(e2 <= lr, O[np.minimum(e2, len(O) - 1)], C[lr])   # 넘어가면 그날 정규장 종가
        pr.append(px2 / O[ent] - 1)
    pnl_rand = np.mean(pr, axis=0)

    def tail(x, q=0.05):
        c = np.nanquantile(x, q)
        return float(np.nanmean(x[x <= c]))

    g = pd.DataFrame({"d": dates, "x": pnl_rule - pnl_rand}).groupby("d")["x"].mean()
    return {
        "entries": int(len(ent)), "exit_rate%": round(100 * use.mean(), 1),
        "avg_hold_min": round(float(dur.mean()), 1),
        "rule_mean%": round(100 * np.nanmean(pnl_rule), 3), "hold_mean%": round(100 * np.nanmean(pnl_hold), 3),
        "rand_mean%": round(100 * np.nanmean(pnl_rand), 3),
        "rule_cvar5%": round(100 * tail(pnl_rule), 3), "hold_cvar5%": round(100 * tail(pnl_hold), 3),
        "rand_cvar5%": round(100 * tail(pnl_rand), 3),
        "x_rand_mean%": round(100 * float(np.nanmean(pnl_rule - pnl_rand)), 3),
        "x_rand_t_days": round(float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), 2) if len(g) > 2 and g.std(ddof=1) > 0 else None,
        "rule_worst%": round(100 * float(np.nanmin(pnl_rule)), 2), "hold_worst%": round(100 * float(np.nanmin(pnl_hold)), 2),
    }


# ───────────────────────── 상황 조건 도우미 ─────────────────────────
def ctx_rally(bars: pd.DataFrame, lookback: int = 60, pct: float = 0.015) -> pd.Series:
    """'올라가는 척' 상황: 최근 lookback봉 최저가 대비 종가가 pct 이상 위(완성봉 기준, 미래 없음)."""
    lo = bars["Low"].rolling(lookback, min_periods=1).min()
    return bars["Close"] >= lo * (1 + pct)


def ctx_decline(bars: pd.DataFrame, lookback: int = 120, pct: float = 0.03) -> pd.Series:
    """하락 중 상황: 최근 lookback봉 최고가 대비 종가가 pct 이상 아래."""
    hi = bars["High"].rolling(lookback, min_periods=1).max()
    return bars["Close"] <= hi * (1 - pct)


def to_1m_mask(sig: pd.Series, bars: pd.DataFrame) -> np.ndarray:
    """상위봉 상태(bool)를 1분봉 길이 마스크로 — 1분봉 시각에 '이미 완성된' 최신 상위봉의 상태(미래 없음)."""
    d1 = load_1m()
    s = sig.reindex(bars.index).fillna(False).astype(bool)
    ends = bars["end"].values
    k = np.searchsorted(ends, d1.index.values, side="right") - 1
    v = s.to_numpy()
    return np.where(k >= 0, v[np.clip(k, 0, len(v) - 1)], False)


if __name__ == "__main__":
    d = load_1m()
    print(d.index.min(), d.index.max(), len(d))
    b3 = resample(3)
    print(b3.tail(3))
    f = _fwd_tables()
    for rng, nm in ((TRAIN, "TRAIN"), (TEST, "TEST")):
        b = baseline_positions(rng)
        print(nm, len(b), "r60 mean%", round(100 * np.nanmean(f["r60"][b]), 3), "rclose%", round(100 * np.nanmean(f["rclose"][b]), 3))
