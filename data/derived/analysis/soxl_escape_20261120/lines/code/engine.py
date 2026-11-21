"""SOXL 경고선·재매수선(대피 ↔ 복귀) 연구 공통 엔진 (2026-11-20).

생각: 정확한 바닥은 못 맞히니, 경고선 아래로 내려가면 대피(현금)하고 재매수선 위로 올라타면 복귀한다.
모든 갈래가 이 엔진의 자료·봉 만들기·체결·채점·대조군을 그대로 쓴다. 선과 신호만 갈래가 만든다.

자료: cache/<SYM>_1m.parquet (Alpaca SIP 1분봉, 04:00~20:00 ET, 분할 반영, 2025-01-02~2026-11-19).
봉: tf분봉은 세션 시작에 맞춘다(트레이딩뷰와 같음) — 정규장('reg') 09:30, 연장 포함('ext') 04:00. 'D'는 그 세션의 하루 봉.
체결: 신호는 봉이 완성된 뒤에만 안다 → 완성 시각 이후 첫 SOXL 1분봉 시가(exec_sess='reg'면 정규장 1분봉만).
      닿으면 체결(touch): 봉 시작 때 이미 아는 선 값에 그 봉 안의 1분봉 저가(고가)가 닿은 첫 1분봉에서 min(시가, 선)에 매도 / max(시가, 선)에 매수.
손익: SOXL 1분봉 시가→다음 시가 로그수익을 보유 중인 봉만 더한다(밤사이 갭 포함). 수수료 0, 비용은 cost 인자(편도).
대조: ① 그냥 보유 ② 같은 평균 비중을 늘 들고 있기(f) ③ 보유/대피 경로를 시간축으로 돌려 끼운 위약(circular shift).
      '시점 값'(TV) = Σ(보유−f)·로그수익 = 규칙 로그수익 − 위약 기대값. 선이 대피 시점을 잘 골랐는지의 1순위 지표다.

구간(사전 고정)
- SEL  2026-01-02~10-14: 고르는 구간.
- CONF 2026-10-15~11-19: 확인 구간(사용자가 본 급락 장세). 여기 성적으로 다시 고르지 않는다.
- 보조: Y2025(2025-03-03~12-31, 장세 위험 한 줄), 창 안 분할 B1 10-15~11-04 / B2 11-05~11-19(지난 연구와 같은 분할, 참고).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ET = "America/New_York"
WINDOWS = {
    "SEL": ("2026-01-02", "2026-10-14"),
    "CONF": ("2026-10-15", "2026-11-19"),
    "Y2025": ("2025-03-03", "2025-12-31"),
    "B1": ("2026-10-15", "2026-11-04"),
    "B2": ("2026-11-05", "2026-11-19"),
}
REG_OPEN, REG_CLOSE, EXT_OPEN, EXT_CLOSE = 570, 960, 240, 1200
_C: dict = {}
SYM = "SOXL"          # 체결·손익을 계산하는 종목. use()로 바꾼다(기본 SOXL — 기존 스크립트는 그대로 돈다).


def use(sym: str) -> None:
    """체결·손익 종목을 바꾼다. 종목별 1분봉·봉 캐시(("1m",종목), ("b",종목,…))는 남기고 나머지 캐시는 비운다."""
    global SYM
    SYM = sym
    for k in list(_C):
        if not (isinstance(k, tuple) and k and k[0] in ("1m", "b")):
            del _C[k]


# ───────────────────────── 자료 ─────────────────────────
def load(sym: str = "SOXL") -> pd.DataFrame:
    k = ("1m", sym)
    if k not in _C:
        d = pd.read_parquet(os.path.join(ROOT, "cache", f"{sym}_1m.parquet"))[["Open", "High", "Low", "Close", "Volume"]]
        d = d[~d.index.duplicated()].sort_index().copy()
        d["date"] = pd.to_datetime(d.index.date)
        d["mod"] = d.index.hour * 60 + d.index.minute
        d["reg"] = (d["mod"] >= REG_OPEN) & (d["mod"] < REG_CLOSE)
        _C[k] = d
    return _C[k]


def bars(sym: str, tf, sess: str) -> pd.DataFrame:
    """tf: 분(int) 또는 'D'. sess: 'reg'(09:30~16:00) | 'ext'(04:00~20:00). index=봉 시작, 'end'=완성 시각.
    마지막 칸이 세션 끝을 넘으면 세션 끝(그날 마지막 1분봉+1분)에서 완성된다(조기 종료일 포함)."""
    k = ("b", sym, tf, sess)
    if k in _C:
        return _C[k]
    d = load(sym)
    if sess == "reg":
        d = d[d["reg"]]
    start = REG_OPEN if sess == "reg" else EXT_OPEN
    day0 = d.index.normalize()
    if tf == "D":
        key = day0
    else:
        key = day0 + pd.to_timedelta(start + ((d["mod"].to_numpy() - start) // tf) * tf, unit="m")
    g = d.groupby(key)
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    naive = lambda x: pd.DatetimeIndex(x).tz_convert("UTC").tz_localize(None)          # noqa: E731
    sess_end = pd.Series(naive(d.index) + pd.Timedelta(minutes=1), index=day0).groupby(level=0).max()
    day_end = sess_end.reindex(out.index.normalize()).to_numpy()
    if tf == "D":
        end = day_end
    else:
        end = np.minimum((naive(out.index) + pd.Timedelta(minutes=tf)).to_numpy(), day_end)
    out["end"] = pd.DatetimeIndex(end).tz_localize("UTC").tz_convert(ET)
    out["date"] = pd.to_datetime(out.index.date)
    _C[k] = out
    return out


# ───────────────────────── 지표 ─────────────────────────
def ema(x: pd.Series, n: int) -> pd.Series:
    """ta.ema와 같은 재귀식(alpha=2/(n+1)). 시작값만 다르며(첫 값) 2025년 예열 뒤에는 차이가 없다."""
    return x.ewm(span=n, adjust=False).mean()


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def atr(b: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = b["Close"].shift(1)
    tr = pd.concat([b["High"] - b["Low"], (b["High"] - pc).abs(), (b["Low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr.fillna(b["High"] - b["Low"]), n)


# ───────────────────────── 체결·보유 경로 ─────────────────────────
def _base():
    if "base" not in _C:
        d = load(SYM)
        O = d["Open"].to_numpy(float)
        r = np.zeros(len(O))
        r[:-1] = np.log(O[1:] / O[:-1])
        _C["base"] = {"t": d.index.values, "O": O, "H": d["High"].to_numpy(float), "L": d["Low"].to_numpy(float),
                      "r": r, "reg": d["reg"].to_numpy(), "date": d["date"].values}
    return _C["base"]


def simulate(b: pd.DataFrame, exit_close=None, entry_close=None, exit_touch=None, entry_touch=None,
             exec_sess: str = "ext", start_in: bool = True):
    """두 줄 상태 기계 → SOXL 1분봉 보유 경로.
    b: 신호를 만든 봉(SOXL이든 SOXX든). exit_close/entry_close: 봉 종가에 판정되는 bool Series(완성 뒤 다음 1분봉 시가 체결).
    exit_touch/entry_touch: '봉 시작 때 이미 아는' 선 값 Series(직전 봉까지로 계산해 shift(1)한 값). 그 봉 안에서 닿으면 선 가격 체결.
      touch는 SOXL 봉에서만 쓴다(SOXX 선은 SOXL 가격에 걸 수 없다).
    한 봉에서는 한 번만 바뀐다. 보유 중이면 탈출만, 대피 중이면 복귀만 본다.
    반환: pos(1분봉 길이 0/1), corr(체결가 보정 로그수익), sw(바뀐 1분봉 위치 목록)."""
    B = _base()
    n1 = len(B["t"])
    idx = b.index
    nb = len(b)
    naive = lambda x: pd.DatetimeIndex(x).tz_convert("UTC").tz_localize(None).to_numpy()   # noqa: E731
    if "t_naive" not in _C:
        _C["t_naive"] = naive(load(SYM).index)
    tn = _C["t_naive"]
    ends = naive(b["end"])

    def epos(e):
        p = np.searchsorted(tn, e, side="left")
        if exec_sess == "reg":
            if "regpos" not in _C:
                _C["regpos"] = np.nonzero(B["reg"])[0]
            rp = _C["regpos"]
            k = np.searchsorted(rp, p, side="left")
            p = np.where(k < len(rp), rp[np.minimum(k, len(rp) - 1)], n1)
        return p

    ex_pos = epos(ends)
    starts = naive(idx)
    s_pos = np.searchsorted(tn, starts, side="left")          # 봉의 첫 1분봉 위치
    e_pos_bar = np.searchsorted(tn, ends, side="left")        # 봉 끝(다음 봉 시작) 위치

    xc = np.zeros(nb, bool) if exit_close is None else pd.Series(exit_close).reindex(idx).fillna(False).to_numpy(bool)
    ec = np.zeros(nb, bool) if entry_close is None else pd.Series(entry_close).reindex(idx).fillna(False).to_numpy(bool)
    xt = None if exit_touch is None else pd.Series(exit_touch).reindex(idx).to_numpy(float)
    et = None if entry_touch is None else pd.Series(entry_touch).reindex(idx).to_numpy(float)
    lo, hi = b["Low"].to_numpy(float), b["High"].to_numpy(float)
    x_cand = xc.copy()
    if xt is not None:
        x_cand |= np.nan_to_num(lo - xt, nan=1.0) <= 0
    e_cand = ec.copy()
    if et is not None:
        e_cand |= np.nan_to_num(hi - et, nan=-1.0) >= 0
    xi, ei = np.nonzero(x_cand)[0], np.nonzero(e_cand)[0]

    O, Hh, Ll, reg = B["O"], B["H"], B["L"], B["reg"]
    events = []            # (1분봉 위치, 새 상태, 보정)
    state = start_in
    i = -1                 # 마지막으로 처리한 봉
    free = 0               # 이 1분봉 위치부터 새 체결 가능(직전 체결 이후)
    while True:
        cand = xi if state else ei
        k = np.searchsorted(cand, i + 1)
        if k >= len(cand):
            break
        j = int(cand[k])
        i = j
        lvl = (xt if state else et)
        done = False
        if lvl is not None and lvl[j] == lvl[j]:
            a, z = max(int(s_pos[j]), free), int(e_pos_bar[j])
            if z > a:
                seg = slice(a, z)
                hit = (Ll[seg] <= lvl[j]) if state else (Hh[seg] >= lvl[j])
                if exec_sess == "reg":
                    hit &= reg[seg]
                h = np.nonzero(hit)[0]
                if len(h):
                    p = a + int(h[0])
                    if state:      # 매도: min(시가, 선) → 그 봉 시가에 판 것으로 두고 시가→체결가 보정(≤0)
                        fill = min(O[p], lvl[j])
                        events.append((p, False, float(np.log(fill / O[p]))))
                    else:          # 매수: max(시가, 선) → 그 봉 시가에 산 것으로 두고 체결가→시가 보정(≤0)
                        fill = max(O[p], lvl[j])
                        events.append((p, True, float(np.log(O[p] / fill))))
                    free = p + 1
                    state = not state
                    done = True
        if not done and (xc[j] if state else ec[j]):
            p = int(ex_pos[j])
            if p >= n1:
                break
            p = max(p, free)
            if p >= n1:
                break
            events.append((p, not state, 0.0))
            free = p + 1
            state = not state
    pos = np.zeros(n1)
    corr = np.zeros(n1)
    cur, last = (1.0 if start_in else 0.0), 0
    for p, st, c in events:
        pos[last:p] = cur
        cur, last = (1.0 if st else 0.0), p
        corr[p] += c
    pos[last:] = cur
    return pos, corr, np.array([e[0] for e in events], dtype=int)


# ───────────────────────── 채점 ─────────────────────────
def _win(name_or_rng):
    rng = WINDOWS[name_or_rng] if isinstance(name_or_rng, str) else name_or_rng
    B = _base()
    k = ("win", rng)
    if k not in _C:
        m = (B["date"] >= np.datetime64(rng[0])) & (B["date"] <= np.datetime64(rng[1]))
        ii = np.nonzero(m)[0]
        _C[k] = (ii[0], ii[-1])          # 마지막 봉의 수익(다음 날 시가까지)은 뺀다
    return _C[k]


def _mdd(logeq: np.ndarray) -> float:
    e = np.exp(logeq)
    return float((e / np.maximum.accumulate(e) - 1).min())


def evaluate(pos, corr, sw, window="SEL", cost: float = 0.0) -> dict:
    """한 구간 성적. 수익률은 %, tv는 로그수익 %p(규칙 − 같은 평균 비중 위약 기대값)."""
    B = _base()
    a, z = _win(window)
    r = B["r"][a:z]
    p = pos[a:z]
    c = corr[a:z]
    nsw = int(((sw >= a) & (sw < z)).sum())
    inc = p * r + c
    if cost:
        cc = np.zeros(z - a)
        s = sw[(sw >= a) & (sw < z)] - a
        np.add.at(cc, s, -cost)
        inc = inc + cc
    f = float(p.mean())
    tot, bh = float(inc.sum()), float(r.sum())
    # 날짜별 시점 값(밤사이 갭은 그 갭을 여는 봉의 날짜에 넣는다)
    dts = B["date"][a:z]
    dv = pd.Series((p - f) * r + c).groupby(dts).sum()
    t = float(dv.mean() / (dv.std(ddof=1) / np.sqrt(len(dv)))) if len(dv) > 2 and dv.std(ddof=1) > 0 else float("nan")
    top3 = float(dv.sort_values(ascending=False).iloc[:3].sum())
    fixed = np.cumsum(np.log1p(f * np.expm1(r)))
    return {
        "ret%": round(100 * np.expm1(tot), 2), "bh%": round(100 * np.expm1(bh), 2),
        "fixed%": round(100 * np.expm1(float(fixed[-1])), 2),
        "tv%p": round(100 * (tot - f * bh), 2), "tv_t": round(t, 2), "tv_ex_top3%p": round(100 * (tot - f * bh - top3), 2),
        "mdd%": round(100 * _mdd(np.cumsum(inc)), 2), "bh_mdd%": round(100 * _mdd(np.cumsum(r)), 2),
        "fixed_mdd%": round(100 * _mdd(fixed), 2),
        "expo%": round(100 * f, 1), "switches": nsw, "days": int(len(dv)),
    }


def placebo(pos, corr, window="SEL", n: int = 200, seed: int = 5) -> dict:
    """보유/대피 경로를 구간 안에서 무작위로 돌려 끼운 위약 n개 대비 백분위(수익, 최대낙폭)."""
    B = _base()
    a, z = _win(window)
    r = B["r"][a:z]
    p = pos[a:z]
    real = float((p * r + corr[a:z]).sum())
    real_mdd = _mdd(np.cumsum(p * r + corr[a:z]))
    rs = np.random.default_rng(seed)
    offs = rs.integers(1, len(p) - 1, size=n)
    rets, mdds = [], []
    for o in offs:
        q = np.roll(p, int(o))
        inc = q * r
        rets.append(float(inc.sum()))
        mdds.append(_mdd(np.cumsum(inc)))
    rets, mdds = np.array(rets), np.array(mdds)
    return {"ret_pct": round(100 * float((rets < real).mean()), 1), "mdd_pct": round(100 * float((mdds < real_mdd).mean()), 1),
            "placebo_ret_med%": round(100 * float(np.expm1(np.median(rets))), 2),
            "placebo_mdd_med%": round(100 * float(np.median(mdds)), 2)}


def score_all(pos, corr, sw, windows=("SEL", "CONF", "Y2025", "B1", "B2"), cost: float = 0.0) -> dict:
    return {w: evaluate(pos, corr, sw, w, cost) for w in windows}


if __name__ == "__main__":
    import time
    t0 = time.time()
    for sess in ("reg", "ext"):
        for tf in (60, "D"):
            b = bars("SOXL", tf, sess)
            e = ema(b["Close"], 20)
            pos, corr, sw = simulate(b, exit_close=b["Close"] < e, entry_close=b["Close"] > e, exec_sess=sess)
            print(sess, tf, len(b), {w: {k: v for k, v in evaluate(pos, corr, sw, w).items() if k in ("ret%", "bh%", "fixed%", "tv%p", "tv_t", "mdd%", "bh_mdd%", "expo%", "switches")} for w in ("SEL", "CONF")})
    print(b.tail(2)[["Close", "end"]])
    print(round(time.time() - t0, 1), "s")
