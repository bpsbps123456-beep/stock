"""갈래 8 — 60분봉 박스 하단 이탈(BRIEF 갈래 8). 하네스의 사건 함수와 evaluate_cells를 그대로 쓴다.

실행: py b8_intraday.py fetch      # yfinance 1시간봉(정규장) → ../cache/h1/<티커>.parquet (+ CRCL 일봉)
      py b8_intraday.py qa         # 자료 점검(일봉 대조·분할·빈 봉·튐) + 미래 값 교란 시험
      py b8_intraday.py run        # 사건표 → 기준선 → 칸 판정 → ../results/b8_intraday*.{md,csv,json}

대상(41종목)
- ETF 4: QQQ, SPY, SOXX, SMH.
- 종목 37: ../box_trade/code/line5d_depth.py가 쓰는 트레이딩뷰 AI 목록 38개(../dip_reason/code/fetch.py의 TV_AI,
  코드 실행 없이 정규식+JSON으로 읽음) 중 SOXX(ETF, 위에 포함)를 뺀 37개. CRCL은 AI_UNIVERSE 밖이지만 목록에 있어 넣는다.
- 자료: yfinance interval=1h, prepost=False(정규장 9:30~16:00 ET, 봉 시작 9:30·10:30·…·15:30, 마지막 봉 30분),
  auto_adjust=False(분할 반영), period=730d(빈 자료면 start=오늘−725일). 2026-10-13 마지막 봉까지(하네스 END와 같게).
- 일봉 특징(일봉 박스·일봉 ATR·전일 종가)은 하네스 일봉 패널(../cache/daily.parquet)을 쓴다. CRCL만 그 패널에 없어
  yfinance 일봉을 ../cache/h1/CRCL_1d.parquet로 받는다.

Yahoo 1시간봉 구멍(점검에서 찾음, 결과를 보기 전에 처리 규칙을 정함)
- 2026-01-30 10:30 봉은 잘린 봉(QQQ 거래량 0.86백만 = 평소의 약 1/6, 7종목은 아예 없음)이고, 그 뒤 01-30 11:30 ~
  02-02 12:30 봉 9개가 모든 종목에서 없다(10/7·10/9에 받은 다른 연구 캐시도 같은 구멍). 이렇게 다룬다.
  ① 01-30 10:30 봉은 버린다(달력이 01-30 9:30 → 02-02 13:30으로 바로 이어짐, 구멍 = 봉 10칸).
  ② 박스 고저 보정: 하네스 일봉의 01-30·02-02 저가가 그날 남은 봉의 최저 저가보다 낮으면 그 저가는 구멍 안에서
     났으므로, 구멍에 붙은 봉(01-30 9:30 봉, 02-02 13:30 봉)의 저가에 넣는다. 고가도 같게. 날 안 남은 봉에서 저가가
     났으면 고치지 않는다(그래서 02-02 13:30 봉에 그 뒤 봉의 저가가 들어가는 일은 없다). 시가·종가는 고치지 않는다.
  ③ 이탈일 t가 01-30 또는 02-02인 사건은 결과를 NaN으로 두어 판정에서 뺀다(진입 봉이 구멍 너머이거나 구멍 안
     경로를 모름). 보유 창이 구멍을 지나는 사건은 남기되 수를 적는다(70봉이 실제로는 80봉 칸).
  ④ 같은 시각대 평균 거래량에서 01-30·02-02는 뺀다.

정의(결과를 보기 전에 정함. BRIEF 일봉 정의를 봉 수로 환산; 정규장 하루 = 7봉)
- 박스 N = 7·14·28 거래일 → Nb = 49·98·196봉. L = 최저 저가(봉 i−Nb..i−1), H = 최고 고가(같은 창).
  ATR = 20봉 ATR(봉 i−1까지). 모두 harness.box_levels / detect_events 그대로(gap = 70봉 = 10거래일).
- 이탈 봉 i: 봉 종가 < L. 같은 종목에서 직전 이탈 봉 뒤 70봉 안의 이탈은 같은 사건(연쇄, 하네스와 같음).
- 진입: 봉 i+1 시가 공매도(i가 그날 마지막 봉이면 다음 거래일 9:30 봉 시가 — 밤사이 갭 포함).
- 청산: harness.short_outcome(h = 70봉): 봉 i+1..i+69 종가 > L → 다음 봉 시가, 아니면 봉 i+70 종가.
  보조 h5 = 35봉, h20 = 140봉. 비용 왕복 0.10%. 반나절 장(Yahoo 봉 3개)이 끼면 70봉은 10거래일보다 몇 시간 길다.
- 분류(설명용): 진짜 = 봉 i+1..i+70에서 L 위 종가 전에 저가 ≤ L − 2·ATRd, 가짜 = 봉 i+1..i+35에서 L 위 종가이고
  그때까지 저가 > L − 1·ATRd, 나머지 중간. ATRd = 일봉 ATR20(전 거래일까지, 하네스 일봉).
  일봉 ATR을 쓰는 까닭: '2·ATR 하락'의 크기를 일봉 BRIEF와 같게 두려고(20봉 ATR은 일봉보다 훨씬 작다).
- 날짜(t) = 이탈 봉의 거래일. 구간 나누기·날짜 묶음 t·순열·초과 손익의 '같은 날'은 모두 이 날짜 기준(하네스와 같음).
  초과 손익은 harness.add_excess 그대로(같은 날 같은 유형 사건 평균, 1건이면 그 봉에 이탈 상태가 아닌 무작위
  같은 유형 종목의 같은 보유 창).
- 국면 표지 episode: 하네스 일봉 패널의 episodes()를 이탈 거래일에 붙인다(사후 표지, 판정의 한 국면 점검 전용).

특징 9개(모두 이탈 봉 i의 종가 뒤 알 수 있음, 진입은 봉 i+1 시가)
- depth_atr: (L − 종가_i) / ATR(20봉, 봉 i−1까지).
- vol_ratio: 거래량_i ÷ 같은 시각대 봉의 직전 20거래일 평균 거래량(오늘 제외, 15일 이상 있을 때). 장중 U자 거래량 보정.
- tod_open / tod_mid / tod_close: 이탈 봉이 개장 1시간(9:30 봉) / 중간 / 그날 마지막 봉(보통 15:30 봉, 반나절 장은
  Yahoo 마지막 봉 11:30 — 반나절 일정은 미리 공시됨). 이진(=1/=0) 칸.
- dbox_pos: 일봉 20일 박스(전 거래일까지의 최저 저가·최고 고가) 안 종가_i 위치 = (종가_i − Ld20) / (Hd20 − Ld20).
  0보다 작으면 일봉 박스도 이미 깨진 상태.
- box_w_atr: (H − L) / ATR(20봉) — 봉 박스 폭.
- gap_break: 이탈 봉 시가 < L(봉이 이미 박스 아래에서 열림; 9:30 봉이면 밤사이 갭 이탈).
- ret_day: 종가_i / 전 거래일 종가(하네스 일봉) − 1(그날 낙폭).

칸: harness.tercile_cells(N 세 표 이어 붙임, by=("N",), 종목+ETF 함께) — 하네스 108칸·갈래 3과 같은 방식.
문턱은 앞 구간(2026-01-02~05-29) 사건 분포. 판정은 evaluate_cells 기본값(top 10, 켜짐 ≥ 15건·날짜 ≥ 8일,
순열 200번, 귀무 200번, 층 N·kind) 그대로. 보조로 원형 이동 p·귀무를 같이 적는다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

RES = H.RESULTS
H1 = H.CACHE / "h1"
END_TS = pd.Timestamp("2026-10-13 23:59")
ETF8 = ["QQQ", "SPY", "SOXX", "SMH"]
NDAYS = (7, 14, 28)
BPD = 7                 # 정규장 하루 봉 수
HZ = {5: 35, 10: 70, 20: 140}
DEDUP_B = 70
FAKE_WIN = 35
DROP_TS = [pd.Timestamp("2026-01-30 10:30")]
HOLE_DAYS = [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-02")]
HOLE_SPAN = (pd.Timestamp("2026-01-30 10:30"), pd.Timestamp("2026-02-02 12:30"))   # 빠진 봉 시작 시각 범위
FEATS = ["depth_atr", "vol_ratio", "tod_open", "tod_mid", "tod_close", "dbox_pos", "box_w_atr", "gap_break",
         "ret_day"]
KNOWN = {
    "depth_atr": "이탈 봉 종가(L은 직전 Nb봉, ATR은 직전 20봉) → 다음 봉 시가 진입 전",
    "vol_ratio": "이탈 봉 종료 시 거래량, 비교 평균은 전 거래일까지 같은 시각대 20일 → 다음 봉 시가 전",
    "tod_open": "이탈 봉 시각(9:30 봉) — 봉 종료 시 확정",
    "tod_mid": "이탈 봉 시각(10:30~14:30 봉) — 봉 종료 시 확정",
    "tod_close": "이탈 봉 시각(그날 마지막 봉, 반나절 장 일정은 사전 공시) — 봉 종료 시 확정, 진입은 다음 거래일 시가",
    "dbox_pos": "일봉 20일 박스는 전 거래일까지, 위치는 이탈 봉 종가 → 다음 봉 시가 전",
    "box_w_atr": "직전 Nb봉 고저·직전 20봉 ATR → 이탈 봉 전",
    "gap_break": "이탈 봉 시가 — 봉 시작 시 확정",
    "ret_day": "전 거래일 종가·이탈 봉 종가 → 다음 봉 시가 전",
}


def tv_ai():
    """../dip_reason/code/fetch.py의 TV_AI 리스트를 코드 실행 없이 읽는다(정규식 + JSON)."""
    src = (H.ANALYSIS / "dip_reason/code/fetch.py").read_text(encoding="utf-8")
    m = re.search(r"^TV_AI\s*=\s*(\[.*?\])", src, re.S | re.M)
    if not m:
        raise RuntimeError("TV_AI를 못 찾음")
    return json.loads(m.group(1))


TV_AI = tv_ai()
STOCKS8 = [s for s in TV_AI if s not in ETF8]
SYMS8 = STOCKS8 + ETF8


# ───────────────────────────── 자료 ─────────────────────────────
def fetch(force=False):
    import yfinance as yf
    H1.mkdir(parents=True, exist_ok=True)
    need = [s for s in SYMS8 if force or not (H1 / f"{s}.parquet").exists()
            or len(pd.read_parquet(H1 / f"{s}.parquet")) == 0]
    # period=730d는 상장일이 730일 한도 밖인 종목(예: ALAB 2024-03 상장)에서 Yahoo가 시작 시각을 상장일로 잡아
    # 거절한다. 그런 종목은 start = 오늘 − 725일로 다시 받는다.
    for kw in (dict(period="730d"),
               dict(start=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=725)).strftime("%Y-%m-%d"))):
        if not need:
            break
        d = yf.download(need, interval="1h", prepost=False, auto_adjust=False, progress=False,
                        group_by="ticker", threads=True, **kw)
        left = []
        for s in need:
            try:
                x = d[s][H.FIELDS].dropna(subset=["Close"]).copy()
            except KeyError:
                x = pd.DataFrame()
            if x.empty:
                print("빈 자료:", s, kw)
                left.append(s)
                continue
            x.index = x.index.tz_convert("America/New_York").tz_localize(None)
            x.index.name = "ts"
            x.to_parquet(H1 / f"{s}.parquet")
        need = left
    # 하네스 일봉 패널에 없는 종목의 일봉
    daily_syms = set(pd.read_parquet(H.CACHE / "daily.parquet", columns=["sym"])["sym"])
    for s in SYMS8:
        p = H1 / f"{s}_1d.parquet"
        if s in daily_syms or (p.exists() and not force):
            continue
        x = H._yf([s], H.START, (pd.Timestamp(H.END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        x.to_parquet(p)
    man = {}
    for s in SYMS8:
        p = H1 / f"{s}.parquet"
        if not p.exists():
            man[s] = None
            continue
        x = pd.read_parquet(p)
        man[s] = {"first": str(x.index.min()), "last": str(x.index.max()), "bars": len(x),
                  "days": int(x.index.normalize().nunique())}
    out = {"source": "yfinance 1h, prepost=False, auto_adjust=False, period=730d "
                     "(빈 자료면 start=오늘−725일로 재수신: 2026-10-14 ALAB)",
           "daily_extra": sorted(s for s in SYMS8 if s not in daily_syms),
           "fetched_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
           "requested": len(SYMS8), "received": sum(v is not None for v in man.values()),
           "missing": [s for s, v in man.items() if v is None], "per_symbol": man}
    (RES / "b8_h1_manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_bars():
    parts = []
    for s in SYMS8:
        x = pd.read_parquet(H1 / f"{s}.parquet")
        x = x[(x.index <= END_TS) & ~x.index.isin(DROP_TS)]
        parts.append(x.reset_index().assign(sym=s))
    return pd.concat(parts, ignore_index=True)


def load_daily_official(syms):
    """하네스 일봉 패널(+ 없는 종목은 ../cache/h1/<티커>_1d.parquet) → (cal, {필드: T×S 배열})."""
    Pd = H.load_panel()
    out = {k: np.full((len(Pd.cal), len(syms)), np.nan) for k in "OHLC"}
    for j, s in enumerate(syms):
        if s in Pd.j:
            for k in "OHLC":
                out[k][:, j] = Pd.col(k, s)
        else:
            x = pd.read_parquet(H1 / f"{s}_1d.parquet").set_index("date").reindex(Pd.cal)
            for k, f in zip("OHLC", ("Open", "High", "Low", "Close")):
                out[k][:, j] = x[f].to_numpy(float)
    return Pd.cal, out


class Bars:
    """봉 패널(하네스 Panel) + 거래일·시각대 표지 + 하네스 일봉(일봉 특징용) + 구멍 보정."""

    def __init__(self, long, calendar_sym="QQQ", repair=True):
        cal = pd.DatetimeIndex(sorted(long.loc[long["sym"] == calendar_sym, "ts"].unique()))
        syms = list(SYMS8)
        arr = {}
        for f, k in zip(H.FIELDS, "OHLCV"):
            w = long.pivot(index="ts", columns="sym", values=f).reindex(index=cal, columns=syms)
            arr[k] = np.array(w.to_numpy(dtype=float), copy=True)
        self.extra_ts = int((~long["ts"].isin(cal)).sum())   # 달력(QQQ) 밖 봉 수
        self.cal = cal
        mins = cal.hour * 60 + cal.minute
        self.slot = np.asarray((mins - 570) // 60)
        days = cal.normalize()
        self.dix = pd.factorize(days, sort=True)[0]
        self.days = pd.DatetimeIndex(sorted(days.unique()))
        self.is_last = np.r_[self.dix[1:] != self.dix[:-1], True]
        self.is_first = np.r_[True, self.dix[1:] != self.dix[:-1]]
        self.hole_day = np.isin(self.days, HOLE_DAYS)
        # 하네스 일봉
        self.dcal, self.D = load_daily_official(syms)
        self.didx = np.searchsorted(np.asarray(self.dcal.values, "datetime64[ns]"),
                                    np.asarray(self.days.values, "datetime64[ns]"))
        assert np.all(self.dcal[self.didx] == self.days), "1시간봉 거래일이 일봉 달력에 없음"
        self.repairs = self._repair_hole(arr, syms) if repair else []
        self.P = H.Panel(cal, syms, arr["O"], arr["H"], arr["L"], arr["C"], arr["V"])
        self._agg(arr)

    def _repair_hole(self, arr, syms):
        """구멍 안 고저를 구멍에 붙은 봉에 넣는다(위 ② 규칙). 반환: 고친 목록."""
        fixes = []
        for hd, side in ((HOLE_DAYS[0], "last"), (HOLE_DAYS[1], "first")):
            k = int(np.searchsorted(self.days, hd))
            if k >= len(self.days) or self.days[k] != hd:
                continue
            bars = np.flatnonzero(self.dix == k)
            b = bars[-1] if side == "last" else bars[0]
            dk = self.didx[k]
            for j, s in enumerate(syms):
                lo, hi = self.D["L"][dk, j], self.D["H"][dk, j]
                avail_lo, avail_hi = np.nanmin(arr["L"][bars, j]), np.nanmax(arr["H"][bars, j])
                if np.isfinite(lo) and np.isfinite(avail_lo) and lo < avail_lo - 1e-9:
                    fixes.append((s, str(self.cal[b]), "L", float(arr["L"][b, j]), float(lo)))
                    arr["L"][b, j] = lo
                if np.isfinite(hi) and np.isfinite(avail_hi) and hi > avail_hi + 1e-9:
                    fixes.append((s, str(self.cal[b]), "H", float(arr["H"][b, j]), float(hi)))
                    arr["H"][b, j] = hi
        return fixes

    def _agg(self, arr):
        """1시간봉 → 일봉(점검용 대조에만 쓴다)."""
        g = pd.Series(self.dix)
        self.aC = pd.DataFrame(arr["C"]).groupby(g).last().to_numpy()
        self.aL = pd.DataFrame(arr["L"]).groupby(g).min().to_numpy()


def slot_vol_avg(v, dix, slot, nday, skip_day=None, n=20, minp=15):
    """같은 시각대 봉의 직전 n거래일 평균 거래량(오늘 제외)을 봉마다 돌려준다. skip_day(bool, 거래일)는 평균에서 뺀다."""
    Vm = np.full((nday, 7), np.nan)
    ok = (slot >= 0) & (slot < 7)
    Vm[dix[ok], slot[ok]] = v[ok]
    if skip_day is not None:
        Vm[skip_day] = np.nan
    avg = pd.DataFrame(Vm).rolling(n, min_periods=minp).mean().shift(1).to_numpy()
    out = np.full(len(v), np.nan)
    out[ok] = avg[dix[ok], slot[ok]]
    return out


def classify_bars(l, c, i, Lv, atrd, hz=HZ[10], fw=FAKE_WIN):
    """harness.classify와 같은 규칙을 봉 수(10거래일 = 70봉, 5거래일 = 35봉)와 일봉 ATR로."""
    n = len(c)
    if not np.isfinite(atrd):
        return None
    true_lvl, fake_lvl = Lv - 2 * atrd, Lv - atrd
    minlow = np.inf
    for k in range(i + 1, i + hz + 1):
        if k >= n or np.isnan(c[k]) or np.isnan(l[k]):
            return None
        if l[k] <= true_lvl:
            return "진짜"
        minlow = min(minlow, l[k])
        if c[k] > Lv:
            return "가짜" if (k <= i + fw and minlow > fake_lvl) else "중간"
    return "중간"


def daily_episode_map():
    Pd = H.load_panel()
    _, lab = H.episodes(Pd)
    return dict(zip(Pd.cal, lab))


def build_events_b8(B, Nd, epmap=None):
    """사건표 한 개(N 거래일). 특징은 이탈 봉 종가까지의 정보만 쓴다."""
    P = B.P
    Nb = Nd * BPD
    nday = len(B.days)
    kinds = {s: ("etf" if s in ETF8 else "stock") for s in P.syms}
    hole_after = int(np.searchsorted(P.cal, HOLE_SPAN[1], side="right"))   # 구멍 뒤 첫 봉 인덱스
    per = {}
    for s in P.syms:
        o, h, l, c = (P.col(f, s) for f in "OHLC")
        per[s] = H.detect_events(o, h, l, c, Nb, gap=DEDUP_B)[:5]
    rows = []
    for s, (ev, brk, Ls, Hs, atr) in per.items():
        j = P.j[s]
        o, h, l, c, v = (P.col(f, s) for f in "OHLCV")
        dLs, dHs, datr = H.box_levels(B.D["O"][:, j], B.D["H"][:, j], B.D["L"][:, j], B.D["C"][:, j], 20)
        dC = B.D["C"][:, j]
        va = slot_vol_avg(v, B.dix, B.slot, nday, B.hole_day)
        for i in ev:
            Lv, Hv, a = Ls[i], Hs[i], atr[i]
            k = B.dix[i]
            d = B.didx[k]
            day = B.days[k]
            hole_t = bool(B.hole_day[k])
            if hole_t:
                r10, why, xi, xt, r5, r20, cls = np.nan, None, -1, None, np.nan, np.nan, None
            else:
                r10, why, xi, xt = H.short_outcome(o, c, i, Lv, HZ[10])
                r5 = H.short_outcome(o, c, i, Lv, HZ[5])[0]
                r20 = H.short_outcome(o, c, i, Lv, HZ[20])[0]
                cls = classify_bars(l, c, i, Lv, datr[d])
            dw = dHs[d] - dLs[d]
            rows.append({
                "sym": s, "kind": kinds[s], "N": Nd, "Nb": Nb, "t": day, "ts": P.cal[i], "i": int(i),
                "L": Lv, "H": Hv, "close_t": c[i], "atr20": a, "atr_d": datr[d],
                "entry_open": o[i + 1] if i + 1 < len(c) else np.nan,
                "entry_ts": P.cal[i + 1] if i + 1 < len(c) else pd.NaT,
                "short_ret_h10": r10, "short_ret_h5": r5, "short_ret_h20": r20,
                "exit_reason": why, "exit_date": P.cal[xi] if xi >= 0 else pd.NaT, "exit_i": xi, "exit_type": xt,
                "cls": cls,
                "hole_t": hole_t,
                "hole_hold": bool(xi >= 0 and i < hole_after <= xi),
                "hole_box": bool(i - Nb < hole_after <= i and not hole_t),
                "depth_atr": (Lv - c[i]) / a,
                "vol_ratio": v[i] / va[i] if np.isfinite(va[i]) and va[i] > 0 else np.nan,
                "tod_open": float(B.slot[i] == 0),
                "tod_close": float(B.is_last[i]),
                "tod_mid": float(B.slot[i] != 0 and not B.is_last[i]),
                "dbox_pos": (c[i] - dLs[d]) / dw if np.isfinite(dw) and dw > 0 else np.nan,
                "box_w_atr": (Hv - Lv) / a,
                "gap_break": float(o[i] < Lv) if not np.isnan(o[i]) else np.nan,
                "ret_day": c[i] / dC[d - 1] - 1 if d >= 1 and np.isfinite(dC[d - 1]) else np.nan,
                "slot": int(B.slot[i]),
                "episode": int(epmap.get(day, -1)) if epmap is not None else -1,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["ts", "kind", "sym"]).reset_index(drop=True)
    df = H.add_excess(df, P, per, kinds)
    return df


def build_all(B=None):
    B = B or Bars(load_bars())
    ep = daily_episode_map()
    evs = {N: build_events_b8(B, N, ep) for N in NDAYS}
    for N, e in evs.items():
        e.to_parquet(RES / f"b8_events_N{N}.parquet")
    return B, evs


# ───────────────────────────── 점검 ─────────────────────────────
def cmd_qa():
    long = load_bars()
    B = Bars(long)
    P = B.P
    print(f"달력(QQQ 봉, 01-30 10:30 뺌) {len(B.cal)}봉, 거래일 {len(B.days)}일, {B.cal[0]} ~ {B.cal[-1]}, "
          f"달력 밖 봉 {B.extra_ts}")
    print("시각대 분포:", pd.Series(B.slot).value_counts().sort_index().to_dict())
    bpd = pd.Series(B.dix).value_counts()
    print("하루 봉 수 분포:", bpd.value_counts().sort_index().to_dict(),
          "| 7봉 아닌 날:", [f"{B.days[k].date()}({bpd[k]})" for k in sorted(bpd[bpd != 7].index)])
    dc = B.dcal[(B.dcal >= B.days[0]) & (B.dcal <= B.days[-1])]
    print("일봉 달력에만 있는 날:", [str(x.date()) for x in dc.difference(B.days)],
          "| 1시간봉에만 있는 날:", [str(x.date()) for x in B.days.difference(dc)])
    miss = {}
    for s in P.syms:
        c = P.col("C", s)
        f = np.flatnonzero(~np.isnan(c))
        seg = np.flatnonzero(np.isnan(c[f[0]:f[-1] + 1])) + f[0]
        if len(seg):
            miss[s] = [str(P.cal[k]) for k in seg]
    print("상장 구간 안 빈 봉:", miss or "없음")
    print("첫 봉이 달력 첫날 뒤:", {s: str(P.cal[np.flatnonzero(~np.isnan(P.col('C', s)))[0]].date())
                              for s in P.syms if np.isnan(P.col("C", s)[0])})
    print(f"구멍 보정 {len(B.repairs)}건(종목, 봉, 필드, 원래, 고친 값):")
    for r in B.repairs:
        print("   ", r)
    bad = {}
    for s in P.syms:
        o, h, l, c = (P.col(f, s) for f in "OHLC")
        m = ~np.isnan(c)
        b = (l[m] > np.minimum(o[m], c[m]) + 1e-9) | (h[m] < np.maximum(o[m], c[m]) - 1e-9)
        if b.sum():
            bad[s] = int(b.sum())
    print("OHLC 불일치 봉:", bad or "없음")
    jumps = []
    for s in P.syms:
        c = P.col("C", s)
        r = c[1:] / c[:-1] - 1
        for k in np.flatnonzero(np.abs(np.nan_to_num(r)) > 0.15):
            jumps.append((s, str(P.cal[k + 1]), round(float(r[k]) * 100, 1), int(B.slot[k + 1])))
    js = pd.DataFrame(jumps, columns=["sym", "ts", "pct", "slot"])
    print(f"봉 사이 ±15% 넘는 튐 {len(js)}건, 시각대별 {js['slot'].value_counts().to_dict()}; 9:30 봉 밖:",
          js[js["slot"] != 0].values.tolist())
    wild = []
    for s in P.syms:
        r = P.col("H", s) / P.col("L", s)
        for k in np.flatnonzero(np.nan_to_num(r) > 1.25):
            wild.append((s, str(P.cal[k]), round(float(r[k]), 3)))
    print(f"봉 안 고가/저가 > 1.25 {len(wild)}건:", wild)
    print("1시간봉 일 종가/저가 vs 하네스 일봉(겹치는 날, 종가 중앙·최대 상대차, 최대 날짜, 종가 1% 넘는 날, 저가 1% 넘는 날),"
          " 구멍 이틀 제외:")
    keep = ~B.hole_day
    for s in P.syms:
        j = P.j[s]
        a = pd.DataFrame({"c": B.aC[keep, j], "l": B.aL[keep, j], "C": B.D["C"][B.didx[keep], j],
                          "L": B.D["L"][B.didx[keep], j]}, index=B.days[keep]).dropna()
        rc = (a["c"] / a["C"] - 1).abs()
        rl = (a["l"] / a["L"] - 1).abs()
        print("  ", s, len(a), f"{rc.median():.5f}", f"{rc.max():.4f}", str(rc.idxmax().date()),
              int((rc > 0.01).sum()), int((rl > 0.01).sum()))
    for s, dte in (("NVDA", "2024-06-10"), ("AVGO", "2024-07-15")):
        k = np.searchsorted(B.days, pd.Timestamp(dte))
        c = B.aC[:, P.j[s]]
        print(f"분할 {s} {dte}: 1시간봉 전일 마지막 종가 {c[k - 1]:.2f} → 당일 {c[k]:.2f} ({c[k] / c[k - 1] - 1:+.1%})")
    lookahead_test(B)


def lookahead_test(B, n_pick=6, seed=11):
    """이탈 봉 i 뒤 모든 종목 봉 OHLCV와 그날(포함) 뒤 일봉에 ×U(0.6,1.4)를 곱해도 봉 i 이하 사건 집합·특징이
    같은지, 결과값(h10)은 바뀌는지."""
    rng = np.random.default_rng(seed)
    ep = daily_episode_map()
    feats = FEATS + ["L", "H", "atr20", "atr_d", "close_t"]
    P = B.P
    for N in NDAYS:
        base = build_events_b8(B, N, ep)
        cand = base[(base["t"] >= "2025-01-01") & base["short_ret_h10"].notna()]
        picks = cand.sample(n_pick, random_state=int(rng.integers(1e9)))
        fails, changed, tot, nchk = 0, 0, 0, 0
        for _, r in picks.iterrows():
            i0 = int(r["i"])
            d0 = int(B.didx[B.dix[i0]])
            arrs = {}
            for k in "OHLCV":
                A = getattr(P, k).copy()
                A[i0 + 1:] = A[i0 + 1:] * rng.uniform(0.6, 1.4, size=A[i0 + 1:].shape)
                arrs[k] = A
            arrs["H"] = np.fmax(arrs["H"], np.fmax(arrs["O"], arrs["C"]))
            arrs["L"] = np.fmin(arrs["L"], np.fmin(arrs["O"], arrs["C"]))
            for k in "HL":
                arrs[k][np.isnan(P.C)] = np.nan
            D2 = {k: v.copy() for k, v in B.D.items()}
            for k in "OHLC":
                D2[k][d0:] = D2[k][d0:] * rng.uniform(0.6, 1.4, size=D2[k][d0:].shape)
            D2["H"] = np.fmax(D2["H"], np.fmax(D2["O"], D2["C"]))
            D2["L"] = np.fmin(D2["L"], np.fmin(D2["O"], D2["C"]))
            B2 = object.__new__(Bars)
            B2.__dict__.update(B.__dict__)
            B2.D = D2
            B2.P = H.Panel(P.cal, P.syms, arrs["O"], arrs["H"], arrs["L"], arrs["C"], arrs["V"])
            pert = build_events_b8(B2, N, ep)
            a = base[base["i"] <= i0].set_index(["sym", "i"])
            b = pert[pert["i"] <= i0].set_index(["sym", "i"])
            if set(a.index) != set(b.index):
                fails += 1
                continue
            x, y = a[feats].sort_index(), b.loc[a.index, feats].sort_index()
            nchk += len(x)
            diff = np.nanmax(np.abs(x.to_numpy(float) - y.to_numpy(float))) if len(x) else 0.0
            nanmis = int((x.isna().to_numpy() != y.isna().to_numpy()).sum())
            if diff > 1e-9 or nanmis:
                fails += 1
            same = base[base["i"] == i0].set_index("sym")["short_ret_h10"]
            new = pert[pert["i"] == i0].set_index("sym")["short_ret_h10"].reindex(same.index)
            changed += int((np.abs(same - new) > 1e-12).sum())
            tot += len(same)
        print(f"미래 값 교란 N{N}: 시험 {n_pick}봉, 비교한 봉 i 이하 사건 {nchk}행, 사건·특징 불일치 {fails}, "
              f"대조 h10 바뀜 {changed}/{tot}")


# ───────────────────────────── 기준선·판정 ─────────────────────────────
def baseline_rows(ev):
    out = []
    for N in NDAYS:
        for kind, kn in (("stock", "종목"), ("etf", "ETF")):
            for pn, p in (("앞(2026-01~05)", H.FRONT), ("뒤(2026-06~09)", H.BACK), ("보조 2025", H.AUX)):
                g = ev[(ev["N"] == N) & (ev["kind"] == kind) & H.in_period(ev, p) & ~ev["hole_t"]]
                gg = g[g["short_ret_h10"].notna()]
                m, t, n, G = H.cluster_mean(gg["short_ret_h10"], gg["t"])
                c = g["cls"].dropna()
                out.append({"N": N, "유형": kn, "구간": pn, "사건": n, "날짜": G, "h10": m, "t": t,
                            "excess": gg["excess_h10"].mean() if len(gg) else np.nan,
                            "h5": g["short_ret_h5"].mean(), "h20": g["short_ret_h20"].mean(),
                            "진짜": (c == "진짜").mean() if len(c) else np.nan,
                            "가짜": (c == "가짜").mean() if len(c) else np.nan,
                            "중간": (c == "중간").mean() if len(c) else np.nan,
                            "되돌림": (gg["exit_reason"] == "reclaim").mean() if len(gg) else np.nan})
    return pd.DataFrame(out)


def tod_rows(ev):
    """시각대별 기술 통계(판정 아님, 종목 사건)."""
    lab = np.where(ev["tod_open"] == 1, "개장 1시간", np.where(ev["tod_close"] == 1, "마감 봉", "중간"))
    out = []
    for N in NDAYS:
        for pn, p in (("앞", H.FRONT), ("뒤", H.BACK), ("2025", H.AUX)):
            for tl in ("개장 1시간", "중간", "마감 봉"):
                g = ev[(ev["N"] == N) & (ev["kind"] == "stock") & H.in_period(ev, p) & (lab == tl)]
                gg = g[g["short_ret_h10"].notna()]
                m, t, n, G = H.cluster_mean(gg["short_ret_h10"], gg["t"])
                c = g["cls"].dropna()
                out.append({"N": N, "구간": pn, "시각대": tl, "사건": n, "날짜": G, "h10": m, "t": t,
                            "가짜": (c == "가짜").mean() if len(c) else np.nan,
                            "진짜": (c == "진짜").mean() if len(c) else np.nan})
    return pd.DataFrame(out)


def parse_cell(name):
    """'N7|depth_atr≥1.23(상위3분위)' → (7, 'depth_atr', '≥1.23(상위3분위)')."""
    head, rest = name.split("|", 1)
    N = int(head[1:])
    for f in sorted(FEATS, key=len, reverse=True):
        if rest.startswith(f):
            return N, f, rest[len(f):]
    return N, rest, ""


def definition(N, f, tail):
    return f"N={N}거래일({N * BPD}봉) 60분봉 박스 하단 이탈 사건 중 {f}{tail}(문턱은 앞 구간 사건 분포)"


def cmd_run():
    B, evs = build_all()
    ev = pd.concat([evs[N] for N in NDAYS], ignore_index=True)
    base = baseline_rows(ev)
    tods = tod_rows(ev)
    cells = H.tercile_cells(ev, FEATS, by=("N",))
    tab, summ = H.evaluate_cells(ev, cells)
    meta = [parse_cell(n) for n in tab["cell"]]
    tab.insert(1, "N", [m[0] for m in meta])
    tab.insert(2, "feature", [m[1] for m in meta])
    tab.insert(3, "direction", [m[2] for m in meta])
    tab.insert(4, "definition", [definition(*m) for m in meta])
    tab.insert(5, "known_when", [KNOWN.get(m[1], "") for m in meta])
    tab.to_csv(RES / "b8_intraday_cells.csv", index=False, encoding="utf-8-sig")
    base.to_csv(RES / "b8_intraday_baseline.csv", index=False, encoding="utf-8-sig")
    tods.to_csv(RES / "b8_intraday_tod.csv", index=False, encoding="utf-8-sig")
    okh = ev["short_ret_h10"].notna()
    pers = (("f", H.FRONT), ("b", H.BACK), ("a", H.AUX))
    out = {"n_cells": int(summ["n_cells"]), "n_eligible": summ["n_eligible"], "n_selected": summ["n_selected"],
           "selected": summ["selected"], "n_pass": summ["n_pass"], "passed": summ["passed"],
           "passed_one_regime": summ["passed_one_regime"], "n_pass_shift": summ["n_pass_shift"],
           "passed_shift": summ["passed_shift"], "events": summ["events"], "dates": summ["dates"],
           "params": summ["params"],
           "null": {k: v for k, v in summ.get("null", {}).items()},
           "null_shift": {k: v for k, v in summ.get("null_shift", {}).items()},
           "per_N_events": {p: ev[H.in_period(ev, per) & okh].groupby("N").size().to_dict() for p, per in pers},
           "per_N_dates": {p: ev[H.in_period(ev, per) & okh].groupby("N")["t"].nunique().to_dict()
                           for p, per in pers},
           "per_N_all": {int(N): int(len(evs[N])) for N in NDAYS},
           "h10_nan": {int(N): int(evs[N]["short_ret_h10"].isna().sum()) for N in NDAYS},
           "hole_t": {int(N): int(evs[N]["hole_t"].sum()) for N in NDAYS},
           "hole_hold": {int(N): int(evs[N]["hole_hold"].sum()) for N in NDAYS},
           "hole_box_front": {int(N): int((evs[N]["hole_box"] & H.in_period(evs[N], H.FRONT)).sum()) for N in NDAYS},
           "repairs": len(B.repairs),
           "excess_nan_in_periods": int(ev[okh & (H.in_period(ev, H.FRONT) | H.in_period(ev, H.BACK)
                                                   | H.in_period(ev, H.AUX))]["excess_h10"].isna().sum())}
    (RES / "b8_intraday_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    return ev, base, tods, tab, summ, out


# ───────────────────────────── 보고서 ─────────────────────────────
def pct(x, nd=2, sign=True):
    if x is None or not np.isfinite(x):
        return "못 구함"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def num(x, nd=2):
    return "못 구함" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def exact_thresholds(ev):
    """tercile_cells와 같은 방식(앞 구간, N별 종목+ETF 함께, 특징 NaN 제외)으로 문턱을 전체 정밀도로."""
    th = {}
    fr = H.in_period(ev, H.FRONT)
    for N in NDAYS:
        for f in FEATS:
            x = ev.loc[(ev["N"] == N) & fr, f].astype(float).dropna()
            u = np.unique(x)
            th[(N, f)] = tuple(u) if len(u) <= 2 else tuple(np.quantile(x, [1 / 3, 2 / 3]))
    return th


FEAT_TXT = {
    "depth_atr": "이탈 깊이 (L − 이탈 봉 종가) ÷ 20봉 ATR",
    "vol_ratio": "이탈 봉 거래량 ÷ 같은 시각대 직전 20거래일 평균",
    "tod_open": "이탈 봉이 9:30 봉(개장 1시간)",
    "tod_mid": "이탈 봉이 중간 시각대(10:30~14:30 봉, 그날 마지막 봉 아님)",
    "tod_close": "이탈 봉이 그날 마지막 봉",
    "dbox_pos": "일봉 20일 박스(전일까지) 안 이탈 봉 종가 위치 (C − Ld20)/(Hd20 − Ld20)",
    "box_w_atr": "봉 박스 폭 (H − L) ÷ 20봉 ATR",
    "gap_break": "이탈 봉 시가가 이미 L 아래",
    "ret_day": "이탈 봉 종가 ÷ 전일 종가 − 1",
}


def exact_definition(N, f, tail, th):
    base = (f"N={N}거래일 60분봉 박스(직전 {N * BPD}봉 최저 저가 L) 하단 이탈(봉 종가 < L, 같은 종목 70봉 안 연쇄 묶기), "
            f"다음 봉 시가 공매도·되돌림 손절(봉 종가 > L → 다음 봉 시가) 또는 70봉 뒤 종가, 비용 0.10%; 켜짐 = ")
    q = th[(N, f)]
    if "상위3분위" in tail:
        cond = f"{FEAT_TXT[f]} ≥ {q[1]:.6g}"
    elif "하위3분위" in tail:
        cond = f"{FEAT_TXT[f]} ≤ {q[0]:.6g}"
    else:
        val = tail.split("=")[1].split("(")[0]
        cond = f"{FEAT_TXT[f]} {'예' if float(val) == 1 else '아니오'}"
        return base + cond + " (이진 특징)"
    return base + cond + " (문턱 = 앞 구간 사건 3분위, 종목+ETF 함께)"


def write_md():
    tab = pd.read_csv(RES / "b8_intraday_cells.csv")
    base = pd.read_csv(RES / "b8_intraday_baseline.csv")
    tods = pd.read_csv(RES / "b8_intraday_tod.csv")
    out = json.loads((RES / "b8_intraday_summary.json").read_text(encoding="utf-8"))
    man = json.loads((RES / "b8_h1_manifest.json").read_text(encoding="utf-8"))
    ev = pd.concat([pd.read_parquet(RES / f"b8_events_N{N}.parquet") for N in NDAYS], ignore_index=True)
    th = exact_thresholds(ev)
    sel = tab[tab["selected"]].sort_values("rank")
    nl, ns = out["null"], out["null_shift"]
    late = {s: v["first"][:10] for s, v in man["per_symbol"].items() if v and v["first"][:10] > "2023-11-14"}
    cands = []
    for _, r in sel.iterrows():
        N, f, tail = parse_cell(r["cell"])
        cands.append({
            "name": r["cell"], "definition": exact_definition(N, f, tail, th), "known_when": KNOWN[f],
            "front_diff": round(float(r["f_diff"]) * 100, 3), "back_diff": round(float(r["b_diff"]) * 100, 3),
            "back_on_mean": round(float(r["b_mean_on"]) * 100, 3), "back_t": round(float(r["b_t"]), 3),
            "perm_p": round(float(r["b_p"]), 4), "perm_p_shift": round(float(r["b_p_shift"]), 4),
            "aux2025_diff": round(float(r["a_diff"]) * 100, 3) if np.isfinite(r["a_diff"]) else None,
            "n_on_back": int(r["b_n_on"]), "dates_on_back": int(r["b_dates_on"]),
            "n_on_front": int(r["f_n_on"]), "n_off_front": int(r["f_n_off"]),
            "episode_share_max": (round(float(r["regime_top_share"]), 3)
                                  if np.isfinite(r["regime_top_share"]) else None),
            "passed": bool(r["pass"]),
        })
    (RES / "b8_intraday_candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")

    def brow(r):
        return (f"| {r['N']} | {r['유형']} | {r['구간']} | {r['사건']} | {r['날짜']} | {pct(r['h10'])} | {num(r['t'])} | "
                f"{pct(r['excess'])} | {pct(r['h5'])} | {pct(r['h20'])} | {pct(r['진짜'], 0, False)} | "
                f"{pct(r['가짜'], 0, False)} | {pct(r['중간'], 0, False)} | {pct(r['되돌림'], 0, False)} |")

    st = base[base["유형"] == "종목"].set_index(["N", "구간"])
    et = base[base["유형"] == "ETF"].set_index(["N", "구간"])
    F, Bk, A = "앞(2026-01~05)", "뒤(2026-06~09)", "보조 2025"
    s26 = st[st.index.get_level_values(1).isin([F, Bk])]
    hold = ev[ev["short_ret_h10"].notna() & (H.in_period(ev, H.FRONT) | H.in_period(ev, H.BACK))]
    hold_med = hold.groupby("N").apply(lambda g: float((g["exit_i"] - g["i"]).median()), include_groups=False)
    L = ["# 갈래 8 — 60분봉 박스 하단 이탈: 진짜 하락을 장중에 먼저 가를 수 있나 (2026-10-14)", ""]
    L += ["## 결론", "",
          f"**통과 0칸 / {out['n_cells']}칸.** 60분봉 박스(7·14·28거래일) 하단 이탈에서 '진짜 하락'을 미리 가르는 특징"
          f"(깊이·거래량·시각대·일봉 박스 위치·박스 폭·갭 이탈·당일 낙폭)은 찾지 못했다. 선택편향 귀무의 통과 칸 수 평균 "
          f"{nl['mean']:.3f}(95% 분위 {nl['q95']:.0f}), 관측 0이라 p = {nl['p_value']:.2f}.", "",
          "| 무엇 | 2026 앞(1~5월) | 2026 뒤(6~9월) | 2025 |", "|---|---:|---:|---:|"]
    for N in NDAYS:
        L.append(f"| 종목 N{N} 이탈 공매도 h10(날짜 묶음 t) | {pct(st.loc[(N, F), 'h10'])} ({num(st.loc[(N, F), 't'])}) | "
                 f"{pct(st.loc[(N, Bk), 'h10'])} ({num(st.loc[(N, Bk), 't'])}) | "
                 f"{pct(st.loc[(N, A), 'h10'])} ({num(st.loc[(N, A), 't'])}) |")
    for N in NDAYS:
        L.append(f"| ETF N{N} 이탈 공매도 h10(건) | {pct(et.loc[(N, F), 'h10'])} ({int(et.loc[(N, F), '사건'])}) | "
                 f"{pct(et.loc[(N, Bk), 'h10'])} ({int(et.loc[(N, Bk), '사건'])}) | "
                 f"{pct(et.loc[(N, A), 'h10'])} ({int(et.loc[(N, A), '사건'])}) |")
    top_pos = sel[sel["b_mean_on"] > 0]
    L += ["",
          f"- 60분봉 이탈 공매도는 조건 없이 2026 앞·뒤 모두 손실이다. 2026 종목 사건의 가짜 비율은 "
          f"{pct(s26['가짜'].min(), 0, False)}~{pct(s26['가짜'].max(), 0, False)}, 되돌림 청산은 "
          f"{pct(s26['되돌림'].min(), 0, False)}~{pct(s26['되돌림'].max(), 0, False)}다. 보유 봉 수 중앙값은 "
          f"{(f'{hold_med.min():g}봉' if hold_med.min() == hold_med.max() else f'{hold_med.min():g}~{hold_med.max():g}봉')}(하루 안, N별 앞·뒤 합쳐)이다. ETF 이탈은 2026 앞·뒤 모두 되돌림 청산 100%다.",
          f"- 앞 구간 상위 10칸 중 뒤 구간 켜짐 평균이 플러스인 칸은 {len(top_pos)}개"
          + ("" if top_pos.empty else
             f"(`{top_pos.iloc[0]['cell']}` {pct(top_pos.iloc[0]['b_mean_on'])}, {int(top_pos.iloc[0]['b_n_on'])}건/"
             f"{int(top_pos.iloc[0]['b_dates_on'])}일, t {num(top_pos.iloc[0]['b_t'])}, p {num(top_pos.iloc[0]['b_p'], 3)}, "
             f"손익의 {pct(top_pos.iloc[0]['regime_top_share'], 0, False)}가 국면 {int(top_pos.iloc[0]['regime_top_ep'])}"
             " — 일봉 20일 박스도 이미 깨진 상태의 N28 이탈)")
          + ". 나머지는 켜져도 공매도 손실이다.",
          "- 1위 칸 `N14|tod_close=0`은 앞 구간 꺼짐(마감 봉 이탈)이 1건뿐인 퇴화 칸이다(하네스 고르기는 켜짐 쪽만 "
          "최소 건수를 본다). `N28|tod_open=0`과 `N28|tod_mid=1`도 앞 구간 N28 마감 봉 이탈이 0건이라 앞 구간에서는 같은 칸이다.",
          "- 일봉 결과(baseline.md, 종목 N20 뒤 −3.02%)와 같은 방향이다. 봉을 잘게 쪼개도 '박스를 깼다'는 사실은 "
          "공매도 근거가 되지 않았고, 2026 이탈의 대부분은 몇 시간 안에 박스로 되돌아왔다.", ""]
    L += ["## 시험 범위", "",
          f"- 특징 9개 × 방향(연속형은 앞 구간 3분위 위/아래, 이진형은 =1/=0) × N 3개(7·14·28거래일 = 49·98·196봉) = 54칸 중, "
          f"앞 구간 N28에 마감 봉 이탈이 없어 `N28|tod_close=1`을 만들지 않음 → **{out['n_cells']}칸**. "
          "tod_open·tod_mid·tod_close는 서로 겹치는 이진 칸이다(셋 중 하나만 1).",
          f"- 고르기 자격(앞 차이 > 0, 켜짐 ≥ 15건, 켜짐 날짜 ≥ 8일) {out['n_eligible']}칸 → 상위 {out['n_selected']}칸 확인.",
          "- 판정·순열·귀무는 `harness.evaluate_cells` 기본값 그대로(순열 200번, 귀무 200번, 층 N·kind, 이탈 날짜로 묶음). "
          f"보조로 원형 이동 p(`b_p_shift`, 뒤 구간 이탈 날짜 {out['params']['n_shift'] + 1}일 → 이동 "
          f"{out['params']['n_shift']}개)와 원형 이동 귀무를 같이 적는다.",
          f"- 사건(N 세 표 합친 행, h10 있는 것): 앞 {out['events']['f']}건/{out['dates']['f']}일, 뒤 {out['events']['b']}건/"
          f"{out['dates']['b']}일, 2025 {out['events']['a']}건/{out['dates']['a']}일. N별(7·14·28) 앞 "
          f"{'·'.join(str(out['per_N_events']['f'][str(N)]) for N in NDAYS)}건, 뒤 "
          f"{'·'.join(str(out['per_N_events']['b'][str(N)]) for N in NDAYS)}건, 2025 "
          f"{'·'.join(str(out['per_N_events']['a'][str(N)]) for N in NDAYS)}건 (서로 다른 날짜 앞 "
          f"{'·'.join(str(out['per_N_dates']['f'][str(N)]) for N in NDAYS)}일, 뒤 "
          f"{'·'.join(str(out['per_N_dates']['b'][str(N)]) for N in NDAYS)}일). "
          "같은 이탈이 여러 N에 들어가므로 합친 행 수는 서로 다른 사건 수가 아니다.",
          "- 앞선 박스·손절·레버리지·옵션 연구와 다른 갈래를 본 뒤의 탐색이라 선택편향이 쌓여 있다.", ""]
    L += ["## 판정", "",
          f"- 통과(BRIEF): **{out['n_pass']}칸**. 원형 이동 p까지 통과: {out['n_pass_shift']}칸. 한 국면 의존 통과: 없음.",
          f"- 선택편향 귀무(날짜 i.i.d. 순열 200번): 평균 {nl['mean']:.3f}, 중앙 {nl['q50']:.0f}, 90% {nl['q90']:.0f}, "
          f"95% {nl['q95']:.0f}, 최대 {nl['max']}, 관측(0) 이상 비율 p = {nl['p_value']:.3f}.",
          f"- 보조 원형 이동 귀무: 평균 {ns['mean']:.3f}, 95% {ns['q95']:.0f}, 최대 {ns['max']}, p = {ns['p_value']:.3f}.", ""]
    L += ["### 앞 구간 상위 10칸 (통과 여부 무관)", "",
          "공매도 손익 = 다음 봉 시가 공매도, 되돌림 손절 또는 70봉 뒤 종가, 비용 0.10% 뺀 값(%/거래). 차이 = 켜짐 − 꺼짐. "
          "건/일 = 사건 수 / 서로 다른 이탈 날짜 수. 정확한 문턱은 아래 '칸 정의'와 b8_intraday_candidates.json.", "",
          "| 순위 | 칸 | 앞 차이 | 앞 켜짐 평균 | 앞 켜짐/꺼짐(건) | 뒤 차이 | 뒤 켜짐 평균 | 뒤 켜짐(건/일) | 뒤 t | p | p(원형) | "
          "2025 차이 | 최대 국면 비중 | 통과 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|"]
    for (_, r), c in zip(sel.iterrows(), cands):
        L.append(f"| {int(r['rank'])} | `{r['cell']}` | {pct(r['f_diff'])} | {pct(r['f_mean_on'])} | "
                 f"{int(r['f_n_on'])}/{int(r['f_n_off'])} | {pct(r['b_diff'])} | {pct(r['b_mean_on'])} | "
                 f"{int(r['b_n_on'])}/{int(r['b_dates_on'])} | {num(r['b_t'])} | {num(r['b_p'], 3)} | "
                 f"{num(r['b_p_shift'], 3)} | {pct(r['a_diff'])} | "
                 f"{pct(r['regime_top_share'], 0, False) if np.isfinite(r['regime_top_share']) else '해당 없음(켜짐 손익 합 ≤ 0)'} | "
                 f"{'통과' if c['passed'] else '탈락'} |")
    L += ["", "칸 정의(정확한 문턱):", ""]
    for c in cands:
        L.append(f"- `{c['name']}`: {c['definition']}. 알 수 있는 때: {c['known_when']}.")
    L.append("")
    L += ["## 기준선 (조건 없음, 판정 아님)", "",
          "| N | 유형 | 구간 | 사건 | 날짜 | h10 평균 | 날짜 묶음 t | excess h10 | h5 | h20 | 진짜 | 가짜 | 중간 | 되돌림 청산 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    L += [brow(r) for _, r in base.iterrows()]
    L += ["", "h5 = 35봉, h20 = 140봉(평균은 결과가 있는 사건만). 진짜/가짜는 일봉 ATR20 기준(설명용, t 뒤 가격 사용).", ""]
    L += ["## 시각대별 기술 통계 (종목, 판정 아님)", "",
          "| N | 구간 | 시각대 | 사건 | 날짜 | h10 | t | 가짜 | 진짜 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in tods.iterrows():
        L.append(f"| {r['N']} | {r['구간']} | {r['시각대']} | {r['사건']} | {r['날짜']} | {pct(r['h10'])} | {num(r['t'])} | "
                 f"{pct(r['가짜'], 0, False)} | {pct(r['진짜'], 0, False)} |")
    L += ["", "60분봉 마감 봉은 30분짜리라 이탈이 드물다(2026 앞 N7·N14 각 1건, N28 0건). 2026 이탈의 50~60%가 9:30 봉"
          "(밤사이 갭 포함)에서 난다.", ""]
    L += ["## 자료와 정의", "",
          f"- 1시간봉: yfinance interval=1h, 정규장만(prepost=False), 분할 반영. 요청 {man['requested']}종목, 받음 {man['received']}. "
          "QQQ 달력 기준 2023-11-14 9:30 ~ 2026-10-13 15:30(730거래일). Yahoo가 대부분 종목에 2023-11부터 줬다"
          "(BRIEF가 적은 '2024-10 이후'보다 길다. 판정은 2025·2026만 쓴다). ALAB은 period=730d가 거절되어 start=오늘−725일로 받음.",
          "- 늦게 시작: " + ", ".join(f"{k} {v}" for k, v in sorted(late.items(), key=lambda kv: kv[1])) + ".",
          "- 대상: ETF 4(QQQ·SPY·SOXX·SMH) + 트레이딩뷰 AI 목록 38개(../dip_reason/code/fetch.py `TV_AI`, box_trade가 쓴 목록) 중 "
          "SOXX를 뺀 종목 37. CRCL은 AI_UNIVERSE 밖.",
          "- 일봉 특징(일봉 20일 박스·일봉 ATR20·전일 종가)은 하네스 일봉(../cache/daily.parquet), CRCL만 yfinance 일봉.",
          "- 정의: 박스 L = 직전 Nb봉 최저 저가, 이탈 = 봉 종가 < L, 같은 종목 직전 이탈 뒤 70봉(10거래일) 안은 같은 사건(연쇄), "
          "진입 = 다음 봉 시가(마감 봉 이탈이면 다음 날 9:30 시가), 청산 = 봉 종가 > L이면 다음 봉 시가, 아니면 70봉 뒤 종가. "
          "harness.detect_events·short_outcome·add_excess·tercile_cells·evaluate_cells를 그대로 썼다. 날짜(t) = 이탈 봉의 거래일.",
          "- 분류: 진짜 = 70봉 안 L 위 종가 전에 저가 ≤ L − 2·일봉 ATR20, 가짜 = 35봉 안 L 위 종가이고 그때까지 저가 > L − 1·일봉 ATR20.",
          "- 특징 9개와 '알 수 있는 때'는 b8_intraday.py 머리말과 칸 표 `known_when` 열. 모두 이탈 봉 종가 뒤, 다음 봉 시가 전.", ""]
    L += ["## 자료 점검 (`py b8_intraday.py qa`)", "",
          "- 미래 값 교란: N마다 이탈 봉 6개를 골라 그 뒤 모든 봉 OHLCV와 그날부터의 일봉에 ×U(0.6, 1.4)를 곱함 → 봉 i 이하 "
          "사건 집합·특징 14열 불일치 0(N7 4,132행·N14 3,045행·N28 1,658행 비교), 대조 h10은 그 봉 사건 55/55 모두 바뀜.",
          "- 1시간봉 거래일 730일 = 하네스 일봉 달력과 같음. OHLC 불일치 0. 분할일(NVDA 2024-06-10 +0.6%, AVGO 2024-07-15 +0.8%) 튐 없음.",
          "- 1시간봉 마지막 봉 종가 vs 일봉 종가(구멍 이틀 제외): 종목별 중앙 상대차 0.007~0.11%, 1% 넘는 날 0~3일. 차이가 가장 큰 날은 41종목 중 33종목이 반나절 장.",
          "- **Yahoo 1시간봉 구멍**: 2026-01-30 10:30 봉이 잘렸고(QQQ 거래량 평소 약 1/6, 7종목 없음) 01-30 11:30 ~ 02-02 12:30 "
          "봉 9개가 모든 종목에서 없다(10/7·10/9 다른 연구 캐시도 같음). 결과를 보기 전에 정한 처리: ① 01-30 10:30 봉 버림 "
          f"② 구멍 안에서 난 일봉 저가·고가를 구멍에 붙은 봉(01-30 9:30, 02-02 13:30)에 넣음({out['repairs']}건, 시가·종가는 안 고침) "
          f"③ 이탈일이 01-30·02-02인 사건 {sum(out['hole_t'].values())}건(N7 {out['hole_t']['7']}, N14 {out['hole_t']['14']}, "
          f"N28 {out['hole_t']['28']})은 결과 NaN으로 판정에서 뺌 ④ 그 이틀은 같은 시각대 평균 거래량에서 뺌. 보유 창이 구멍을 "
          f"지나는 사건 {sum(out['hole_hold'].values())}건, 박스 창이 구멍을 지나는 앞 구간 사건 N7 {out['hole_box_front']['7']}·"
          f"N14 {out['hole_box_front']['14']}·N28 {out['hole_box_front']['28']}건(보정한 고저로 L 계산).",
          "- 상장 구간 안 빈 봉: BE 2024-11-14 13:30~15:30, CRCL 상장일 2025-06-05 10:30·11:30(둘 다 판정 구간 밖).",
          "- 봉 사이 ±15% 넘는 튐 158건 중 156건이 9:30 봉(밤사이 실적 갭). 봉 안 고가/저가 > 1.25는 15건(소형주 개장 봉·CRCL 상장일).",
          f"- 초과 손익 NaN {out['excess_nan_in_periods']}건(SOXX N7 2025-11-07·2026-03-27: 그날 ETF 사건 1건인데 다른 ETF도 모두 이탈 중).",
          ""]
    L += ["## 한계", "",
          "1. 반나절 장 Yahoo 봉은 3개(9:30·10:30·11:30)뿐이고, 70봉 보유는 반나절 장이 끼면 10거래일보다 길다.",
          f"2. 구멍 뒤 박스는 일봉 고저로 보정했지만 구멍 안 종가 경로(되돌림 여부)는 모른다. 보유 창이 구멍을 지나는 사건은 "
          f"{sum(out['hole_hold'].values())}건이다.",
          "3. 대차 비용과 배당은 넣지 않았다. 다음 봉 시가 체결을 가정했고, 9:30 봉 시가는 개장 경매가라 실제 체결과 다를 수 있다.",
          "4. 날짜 묶음 순열은 종목 수준 특징에서 보수적이다(qa_harness.md 한계 2). 다만 상위 10칸 중 뒤 구간 켜짐 평균이 "
          "플러스인 칸이 하나뿐이라 검정력 부족으로 놓쳤다고 보기 어렵다.",
          "5. 하네스 고르기는 꺼짐 쪽 최소 건수를 보지 않아, 앞 구간 꺼짐이 1건인 시각대 칸이 1위가 됐다(정의대로 둠).",
          f"6. 사후 참고(판정 아님): {out['n_cells']}칸 중 뒤 구간 켜짐 평균이 플러스인 칸은 {int((tab['b_mean_on'] > 0).sum())}칸, "
          f"가장 큰 것도 {pct(tab['b_mean_on'].max())}다.",
          "7. 다른 해: 2025는 이탈 공매도 손실이 작았다(종목 N7 −0.25%, t −0.30). 같은 이탈이 장세에 따라 반대로 갈 수 있다.", ""]
    L += ["## 파일", "",
          "- 코드: `code/b8_intraday.py` (`fetch` / `qa` / `run` / `report`)",
          "- 봉 자료: `cache/h1/<티커>.parquet`(41개), `cache/h1/CRCL_1d.parquet`, 목록 `results/b8_h1_manifest.json`",
          "- 사건표: `results/b8_events_N7.parquet`, `results/b8_events_N14.parquet`, `results/b8_events_N28.parquet`",
          "- 칸 전체: `results/b8_intraday_cells.csv`, 요약·귀무 분포: `results/b8_intraday_summary.json`, "
          "상위 10칸: `results/b8_intraday_candidates.json`, 기준선 `results/b8_intraday_baseline.csv`, 시각대 `results/b8_intraday_tod.csv`",
          ""]
    (RES / "b8_intraday.md").write_text("\n".join(L), encoding="utf-8")
    return cands


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        print(json.dumps(fetch("--force" in sys.argv), ensure_ascii=False, indent=1)[:4000])
    elif cmd == "qa":
        cmd_qa()
    elif cmd == "run":
        ev, base, tods, tab, summ, out = cmd_run()
        print(json.dumps({k: v for k, v in out.items() if k not in ("null", "null_shift")}, ensure_ascii=False,
                         indent=1, default=str))
        write_md()
    elif cmd == "report":
        print(json.dumps(write_md(), ensure_ascii=False, indent=1))
