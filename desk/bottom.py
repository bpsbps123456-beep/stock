"""바닥 감지기(뉴스 근거 탭 맨 위): SOXX가 하락한 뒤 저점에서 얼마나 올라왔는지로 '그 저점이 한 달 안에
다시 깨질 가능성'을 본다.

정본은 docs/desk_system.md §13.6, 근거는 data/derived/analysis/bottom_confirm/report.md(§13.5)다. 정의는 연구 엔진
code/bc_lib.py(BRIEF 2026-10-26, 결과 전 고정)와 같고, tests/test_bottom.py가 연구 숫자(147일 99.3%)를 다시 낸다.
관찰용 화면이다. 매수 신호·목표 비중·주문 수량을 만들지 않는다(§1·§13 적용 범위).

  고점      = 최근 60거래일 최고 종가(같은 값이면 늦은 날)
  저점      = 그 고점 이후 최저 종가(지금 하락의 저점)
  하락 국면 = 최근 20거래일 안에 종가가 60일 최고 종가보다 10% 이상 낮았던 날이 있음
  반등 폭   = 오늘 종가 ÷ 저점 − 1
  확인      = 하락 국면이고 반등 폭 ≥ 7%(7%와 8%는 구별되지 않았다, §13.5)
  결과      = 확인한 날 뒤 20거래일 동안 종가가 그 저점 아래로 내려가지 않음

가격은 yfinance 일봉 종가(분할 반영)다. 장이 끝나지 않은 날의 봉은 버린다. TradingView 값은 넣지 않는다(AGENTS.md).
확률표는 고정 숫자가 아니라 같은 함수로 SOXX(2024~)와 ^SOX(1994~)를 매번 다시 센다. 그래서 앞으로 쌓이는 날도 들어간다.
"""

from __future__ import annotations

import json
import threading
import time as _time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from . import config, marketdata
from .news import market_calendar, write_atomic
from .state import clean

# 계산 방식이 바뀌면 올린다. 캐시의 방식이 다르면 30분 안이어도 다시 계산한다.
SPEC = "2026-10-28 SOXX 하락 10%·반등 7%·20거래일·새 고점이면 새 신호"
CACHE_PATH = config.CACHE / "bottom.json"
MAX_AGE_MIN = 30
SYMBOL, LONG_SYMBOL = "SOXX", "^SOX"
START, LONG_START = "2023-06-01", "1994-01-01"   # SOXX는 60일 창 예열 뒤 2024-01부터 센다
SIGNAL_FROM = "2024-01-02"                         # 연구 평가 시작일
HIGH_WIN, DD_WIN, DD_THR = 60, 20, 0.10
CONFIRM = 0.07
LINES = (0.07, 0.08, 0.10)                         # 화면에 긋는 반등 선
HOLD = COOL = 20                                   # 결과 창, 같은 하락의 신호 간격(bc_lib.first_signals)
SPLIT_JUMP = 0.40                                  # 하루 ±40%를 넘는 종가 변화는 분할 섞임으로 본다
LIVE_RATIO = (1 / 1.6, 1.6)                        # 실시간 값 ÷ 마지막 종가가 이 밖이면 분할 반영 차이로 본다
LIVE_MAX_AGE_SEC = 90
TAIL = HIGH_WIN + DD_WIN + 5                       # 잠정 계산에 필요한 최근 종가 수
CHART_DAYS = 130
BANDS = ((0.0, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, None), (0.09, None))
PERIODS = (  # (키, 이름, 종목, 시작, 끝) — 끝이 None이면 자료 끝까지
    ("soxx", "2024~ SOXX", SYMBOL, SIGNAL_FROM, None),
    ("soxx26", "2026 SOXX", SYMBOL, "2026-01-02", None),
    ("long", "1994~2023 ^SOX", LONG_SYMBOL, "1994-05-04", "2023-12-29"),
    ("bear", "약세장 ^SOX", LONG_SYMBOL, None, None),
)
BEARS = (("2000-01-03", "2002-12-31"), ("2007-01-03", "2009-12-31"), ("2022-01-03", "2022-12-30"))
NOTICES = (  # (표시 마지막 날, 문구)
    ("2026-11-19", "SOXX 3:1 분할: 11/4 장 마감 뒤 시행, 11/5부터 분할 가격으로 거래합니다. 그날부터 모든 가격이 약 1/3로 "
                   "바뀌지만 반등 폭(비율)은 그대로입니다. 분할 전 저점 475.69는 158.56, 7% 선 508.99는 169.66입니다."),
)
_LOCK = threading.Lock()
_LIVE: dict = {}


class BottomError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- 계산(순수 함수)
def frame(close: pd.Series) -> pd.DataFrame:
    """날마다 고점·저점·하락 국면·반등 폭·20거래일 결과(1 유지, 0 이탈, 창이 덜 찼고 아직 안 깨졌으면 NaN)."""
    c = close.to_numpy(float)
    n = len(c)
    high, low = np.full(n, np.nan), np.full(n, np.nan)
    peak_i, low_i = np.full(n, -1), np.full(n, -1)
    for t in range(HIGH_WIN - 1, n):
        w = c[t - HIGH_WIN + 1:t + 1]
        if np.isnan(w).any():
            continue
        pk = t - int(np.argmax(w[::-1]))              # 최고값의 가장 늦은 위치
        j = int(np.argmin(c[pk:t + 1]))
        high[t], low[t], peak_i[t], low_i[t] = c[pk], c[pk + j], pk, pk + j
    dd = c / high - 1
    in_dd = pd.Series(dd).rolling(DD_WIN, min_periods=1).min().to_numpy() <= -DD_THR
    fwd, full = np.full(n, np.nan), np.zeros(n, bool)
    for t in range(n - 1):
        blk = c[t + 1:t + 1 + HOLD]
        fwd[t], full[t] = blk.min(), len(blk) == HOLD
    broke = fwd < low - 1e-12
    hold = np.where(np.isnan(low) | np.isnan(fwd), np.nan, np.where(broke, 0.0, np.where(full, 1.0, np.nan)))
    idx = close.index
    pick = lambda pos: [idx[p] if p >= 0 else pd.NaT for p in pos]
    return pd.DataFrame({"close": c, "high": high, "low": low, "peak": pick(peak_i), "low_date": pick(low_i),
                         "dd": dd, "up": c / low - 1, "in_dd": in_dd, "hold": hold}, index=idx)


def _between(idx: pd.DatetimeIndex, start, end) -> np.ndarray:
    ok = np.ones(len(idx), bool)
    if start:
        ok &= idx >= pd.Timestamp(start)
    if end:
        ok &= idx <= pd.Timestamp(end)
    return ok


def _mask(idx: pd.DatetimeIndex, key: str, start, end) -> np.ndarray:
    if key == "bear":
        return np.logical_or.reduce([_between(idx, a, b) for a, b in BEARS])
    return _between(idx, start, end)


def band_of(up: float) -> int | None:
    """반등 폭이 속한 BANDS 칸(겹치는 '9% 이상'이 아니라 배타적인 앞 네 칸 중 하나)."""
    if up is None or not np.isfinite(up) or up < 0:
        return None
    for i, (lo, hi) in enumerate(BANDS[:4]):
        if up >= lo and (hi is None or up < hi):
            return i
    return None


def hold_rates(fr: pd.DataFrame, mask: np.ndarray) -> dict:
    """하락 국면 날 가운데 반등 칸별 20거래일 저점 유지 비율(쿨다운 없는 날 기준, 연구 calib·sweep 표와 같은 셈)."""
    base = fr["in_dd"].to_numpy() & mask & ~np.isnan(fr["hold"].to_numpy())
    up, hold = fr["up"].to_numpy(), fr["hold"].to_numpy()
    rows = []
    for lo, hi in BANDS:
        m = base & (up >= lo) & ((up < hi) if hi is not None else True)
        rows.append({"n": int(m.sum()), "rate": float(hold[m].mean()) if m.any() else None})
    return {"bands": rows, "base": {"n": int(base.sum()), "rate": float(hold[base].mean()) if base.any() else None}}


def signals(fr: pd.DataFrame, start: str = SIGNAL_FROM) -> list[dict]:
    """하락 국면에서 반등 폭이 처음 7%를 넘은 날과 그 뒤 20거래일 결과.

    연구(bc_lib.first_signals)처럼 직전 신호에서 20거래일 안의 신호는 다시 세지 않는다. 다만 직전 신호 뒤에 새 60일
    고점이 났으면 다른 하락이므로 센다(예: 9/29 신호 뒤 10/5 고점 → 10/27 확인). 확률표는 날 기준이라 이 규칙과 무관하다.
    """
    ok = fr["in_dd"].to_numpy() & (fr["up"].to_numpy() >= CONFIRM) & (fr.index >= pd.Timestamp(start))
    c, n, out, prev = fr["close"].to_numpy(), len(fr), [], -10 ** 9
    peaks = fr["peak"]
    for p in np.flatnonzero(ok):
        if p - prev <= COOL and not peaks.iloc[p] > fr.index[prev]:
            continue
        prev = p
        r = fr.iloc[p]
        after = c[p + 1:p + 1 + HOLD]
        below = np.flatnonzero(after < r["low"] - 1e-12)
        if below.size:
            result = {"status": "broke", "on": fr.index[p + 1 + below[0]].date().isoformat(), "day": int(below[0] + 1)}
        elif after.size == HOLD:
            result = {"status": "held"}
        else:
            result = {"status": "watching", "day": int(after.size)}
        out.append({"date": fr.index[p].date().isoformat(), "close": r["close"], "low": r["low"],
                    "low_date": r["low_date"].date().isoformat(), "high": r["high"], "up": r["up"], **result,
                    "worst": float(after.min() / r["close"] - 1) if after.size else None,
                    "ret20": float(c[p + HOLD] / r["close"] - 1) if p + HOLD < n else None})
    return out


def status(fr: pd.DataFrame, sigs: list[dict] | None = None) -> dict:
    """마지막 날의 상태: calm(하락 국면 아님) · falling(하락 국면, 확인 전) · confirmed(하락 국면, 반등 ≥ 7%)."""
    if fr.empty or not np.isfinite(fr["low"].iloc[-1]):
        raise BottomError("60거래일 넘는 종가가 필요합니다")
    r, n = fr.iloc[-1], len(fr)
    kind = "calm" if not r["in_dd"] else "confirmed" if r["up"] >= CONFIRM else "falling"
    deep = np.flatnonzero(fr["dd"].to_numpy()[-DD_WIN:] <= -DD_THR)
    out = {"kind": kind, "date": fr.index[-1].date().isoformat(), "close": r["close"], "high": r["high"],
           "peak": r["peak"].date().isoformat(), "low": r["low"], "low_date": r["low_date"].date().isoformat(),
           "dd": r["dd"], "up": r["up"], "band": band_of(r["up"]) if r["in_dd"] else None,
           "lines": [{"k": k, "price": r["low"] * (1 + k), "gap": r["low"] * (1 + k) / r["close"] - 1} for k in LINES],
           "dd_trigger": r["high"] * (1 - DD_THR),
           # 새 −10% 날이 없으면 하락 국면이 앞으로 며칠 더 이어지는지(0이면 오늘이 마지막 날)
           "dd_left": int(deep[-1]) if deep.size else None}
    last = (sigs or [None])[-1]
    if last:
        age = n - 1 - fr.index.get_loc(pd.Timestamp(last["date"]))
        out["signal"] = {**last, "age": int(age), "current": age < HOLD or last["status"] == "watching"}
    return out


# ----------------------------------------------------------------------------- 자료
def completed(close: pd.Series, now: datetime | None = None) -> pd.Series:
    """장이 끝나지 않은 날의 봉(장중 현재가가 종가처럼 들어온 행)을 버린다. 끝의 몇 행만 보면 된다."""
    now = now or datetime.now(timezone.utc)
    cal, drop = market_calendar(), []
    for d in close.index[-3:]:
        day = d.strftime("%Y-%m-%d")
        if not cal.is_session(day) or cal.session_close(day).to_pydatetime() > now:
            drop.append(d)
    return close.drop(drop)


def check_split(close: pd.Series, symbol: str) -> None:
    jump = close.pct_change().abs()
    bad = jump[jump > SPLIT_JUMP]
    if len(bad):
        d = bad.index[0]
        raise BottomError(f"{symbol} {d.date()} 하루 {close.pct_change()[d]:+.0%} — 주식 분할 전후 가격이 섞인 것 같습니다. "
                          "‘가격 갱신’을 다시 누르세요(Yahoo가 분할을 늦게 반영하면 몇 시간 뒤 맞춰집니다)")


def fetch(symbol: str, start: str) -> pd.Series:
    import yfinance as yf

    frame_ = yf.download(symbol, start=start, auto_adjust=False, progress=False, threads=False)
    if frame_ is None or frame_.empty:
        raise BottomError(f"{symbol} 일봉을 받지 못했습니다")
    close = frame_["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[symbol] if symbol in close.columns else close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return close[~close.index.duplicated(keep="last")].astype(float)


def _series(fr: pd.DataFrame, col: str, days: int) -> list:
    v = fr[col].iloc[-days:]
    return [float(x) if pd.notna(x) else None for x in v]


def compute(now: datetime | None = None) -> dict:
    close = completed(fetch(SYMBOL, START), now)
    check_split(close, SYMBOL)
    fr = frame(close)
    sigs = signals(fr)
    st = status(fr, sigs)
    tables = {k: None for k, *_ in PERIODS}
    warnings = []
    lfr = None
    try:
        long_close = completed(fetch(LONG_SYMBOL, LONG_START), now)
        check_split(long_close, LONG_SYMBOL)
        lfr = frame(long_close)
    except Exception as exc:  # 긴 이력이 없어도 SOXX 상태는 보여 준다
        warnings.append(f"^SOX 긴 이력을 받지 못해 긴 이력 칸을 비웠습니다({type(exc).__name__})")
    for key, _name, sym, a, b in PERIODS:
        src = fr if sym == SYMBOL else lfr
        if src is not None:
            tables[key] = hold_rates(src, _mask(src.index, key, a, b))
    tail = fr.iloc[-CHART_DAYS:]
    in_dd = tail["in_dd"].to_numpy()
    chart = {"dates": [d.date().isoformat() for d in tail.index], "close": _series(fr, "close", CHART_DAYS),
             "high": _series(fr, "high", CHART_DAYS),
             "low": [v if m else None for v, m in zip(_series(fr, "low", CHART_DAYS), in_dd)],
             "line": [v * (1 + CONFIRM) if v is not None and m else None
                      for v, m in zip(_series(fr, "low", CHART_DAYS), in_dd)]}
    return {"symbol": SYMBOL, "status": st, "signals": sigs, "tables": tables,
            "periods": [{"key": k, "label": name} for k, name, *_ in PERIODS],
            "bands": [{"lo": lo, "hi": hi} for lo, hi in BANDS], "chart": chart,
            "tail": {"dates": [d.date().isoformat() for d in close.index[-TAIL:]],
                     "close": [float(x) for x in close.iloc[-TAIL:]]},
            "long_range": [lfr.index[0].date().isoformat(), lfr.index[-1].date().isoformat()] if lfr is not None else None,
            "warnings": warnings, "fetched_at": datetime.now(config.KST).isoformat(timespec="seconds"),
            "source": "Yahoo Finance 일봉 종가(분할 반영) · 장이 끝난 날만"}


def provisional(data: dict, quote: dict) -> dict | None:
    """실시간 값(프리·정규·애프터)을 오늘 종가로 가정한 잠정 상태. 마지막 종가 뒤의 값일 때만 만든다."""
    tail = data.get("tail") or {}
    if not quote or "price" not in quote or not tail.get("close"):
        return None
    cal, last_day = market_calendar(), tail["dates"][-1]
    if quote.get("epoch") is None or quote["epoch"] <= cal.session_close(last_day).timestamp():
        return None
    # 애프터마켓 값도 다음 정규장 종가를 가정한 것으로 다룬다(날짜 이름만 다음 거래일)
    nxt = cal.next_session(last_day).strftime("%Y-%m-%d")
    s = pd.Series(tail["close"] + [quote["price"]], index=pd.DatetimeIndex(tail["dates"] + [nxt]))
    st = status(frame(s))
    return {"price": quote["price"], "session": quote.get("session"), "time_kst": quote.get("time_kst"),
            "kind": st["kind"], "up": st["up"], "dd": st["dd"], "low": st["low"], "band": st["band"]}


def live(data: dict) -> tuple[dict | None, str | None]:
    """(잠정 상태, 경고). 실시간 시세는 LIVE_MAX_AGE_SEC 동안 메모리에 둔다."""
    if _time.time() - _LIVE.get("epoch", 0) > LIVE_MAX_AGE_SEC:
        try:
            q = marketdata.live_quotes([SYMBOL])
        except Exception as exc:  # 시세가 없어도 종가 기준 상태는 보여 준다
            q = {"_error": {"message": f"{type(exc).__name__}: {exc}"}}
        _LIVE.update(epoch=_time.time(), quote=q.get(SYMBOL), error=(q.get("_error") or {}).get("message"))
    quote = _LIVE.get("quote")
    if not quote:
        return None, ("실시간 시세를 받지 못했습니다: " + _LIVE["error"]) if _LIVE.get("error") else None
    last = (data.get("tail") or {}).get("close", [None])[-1]
    if last and not LIVE_RATIO[0] < quote["price"] / last < LIVE_RATIO[1]:
        return None, "MISMATCH"
    return provisional(data, quote), None


def _fresh(cached: dict) -> bool:
    return cached.get("_spec") == SPEC and _time.time() - cached.get("_epoch", 0) < MAX_AGE_MIN * 60


def _daily(force: bool) -> dict:
    with _LOCK:
        if not force and CACHE_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if _fresh(cached):
                return cached
        data = clean(compute())
        data["_epoch"], data["_spec"] = _time.time(), SPEC
        write_atomic(CACHE_PATH, data)
        return data


def payload(force: bool = False, with_live: bool = True) -> dict:
    """일봉 계산은 MAX_AGE_MIN 동안 캐시, 실시간 잠정 상태는 요청마다 붙인다(디스크에 저장하지 않음)."""
    data = _daily(force)
    prov, warn = live(data) if with_live else (None, None)
    if warn == "MISMATCH" and _time.time() - data.get("_epoch", 0) > 60:
        data = _daily(True)        # 캐시가 분할 전 값이면 일봉을 다시 받아 맞춘다
        prov, warn = live(data)
    if warn == "MISMATCH":
        prov, warn = None, "실시간 값이 마지막 종가와 크게 달라(분할 반영 차이 의심) 잠정 계산을 생략했습니다"
    today = datetime.now(config.KST).date().isoformat()
    return {**{k: v for k, v in data.items() if k != "tail"}, "live": prov,
            "warnings": data.get("warnings", []) + ([warn] if warn else []),
            "notices": [text for until, text in NOTICES if today <= until], "rules": rules()}


def rules() -> dict:
    return {"high_win": HIGH_WIN, "dd_win": DD_WIN, "dd_thr": DD_THR, "confirm": CONFIRM, "hold": HOLD, "cool": COOL,
            "notes": [
                "하락 국면: 최근 20거래일 안에 SOXX 종가가 60거래일 최고 종가보다 10% 이상 낮았던 날이 있으면 하락 국면으로 봅니다.",
                "저점: 그 60일 고점 이후 가장 낮은 종가입니다. 반등 폭은 오늘 종가가 이 저점보다 몇 % 높은지입니다.",
                "바닥 확인: 하락 국면에서 반등 폭이 7% 이상인 날입니다. 연구(2024~2026)에서 7%와 8%는 구별되지 않았고, "
                "경계는 실패한 반등 4~5개가 멈춘 자리(+6.5~6.9%, +8.6%)로 정해졌습니다.",
                "뜻: '앞으로 20거래일 안에 그 저점이 종가로 다시 깨지지 않는다'까지입니다. 가장 낮은 바닥 보증이 아닙니다 — "
                "8% 반등이 나온 하락 7개 중 5개만 최종 저점이었고, 두 달 유지는 약 83%였습니다.",
                "매수 신호가 아닙니다. 확인 뒤에도 진입가에서 5% 넘게 다시 빠진 경우가 절반 이상이었습니다(두 달 안).",
                "장세 위험: 1994~2023 ^SOX에서는 7% 반등 뒤 한 달 유지가 73%, 약세장(2000~02·2007~09·2022)은 52~70%였습니다. "
                "2024~2026의 99%는 지금 장세의 값입니다.",
                "확률표는 하락 국면 날 전부를 셉니다(쿨다운 없음). 결과 창이 덜 찬 날은 이미 저점이 깨졌으면 실패로, 아니면 빼고 셉니다.",
                "잠정 값은 지금 시세(프리·정규·애프터)를 오늘 종가로 가정해 같은 계산을 한 것입니다. 확정은 정규장 종가입니다.",
                "개별 종목은 이 표를 그대로 쓰지 않습니다. 연구에서 AI 종목은 저점 대비 20% 이상 반등해야 76%였습니다.",
            ],
            "source": "docs/desk_system.md §13.5·§13.6 · data/derived/analysis/bottom_confirm/report.md"}


def summary_lines(data: dict) -> list[str]:
    st, lv = data["status"], data.get("live")
    label = {"calm": "하락 국면 아님", "falling": "하락 중 · 확인 전", "confirmed": "바닥 확인(반등 ≥ 7%)"}
    pct = lambda v: "—" if v is None else f"{v * 100:+.1f}%"
    lines = [f"[바닥 감지기] {data['symbol']} {st['date']} 종가 {st['close']:.2f} — {label[st['kind']]}",
             f"60일 고점 {st['high']:.2f}({st['peak']}) 대비 {pct(st['dd'])} · 저점 {st['low']:.2f}({st['low_date']}) 대비 {pct(st['up'])}"]
    if st["kind"] != "calm":
        lines.append("반등 선: " + " · ".join(f"{int(x['k'] * 100)}% {x['price']:.2f}" for x in st["lines"]))
    sig = st.get("signal")
    if sig and sig.get("current"):
        res = {"broke": f"저점 이탈 {sig.get('on')}", "held": "20거래일 유지", "watching": f"관찰 {sig.get('day')}/{HOLD}일 유지 중"}
        lines.append(f"최근 확인 {sig['date']}(반등 {pct(sig['up'])}) → {res[sig['status']]}")
    if lv:
        lines.append(f"잠정({lv['session']} {lv['time_kst']} KST) {lv['price']:.2f}: 저점 대비 {pct(lv['up'])} → {label[lv['kind']]}")
    lines += [f"주의: {w}" for w in data.get("warnings", [])] + data.get("notices", [])
    return lines


def main(argv: list[str]) -> int:
    data = payload(force="--refresh" in argv)
    print("\n".join(summary_lines(data)))
    return 0
