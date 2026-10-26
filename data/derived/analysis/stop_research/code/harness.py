"""손절·재진입 연구 공통 평가 틀(2026-10-10). 모든 갈래가 이 파일 하나로 같은 시점·같은 지표로 잰다.

기본 경로(build_base.py): 데스크 7종목 규칙 그대로(반도체 50%), 시가→다음 시가 수익(USD, 수수료 0, 현금 이자 0).
  ret[k] = open[k+1]/open[k] - 1 : k일 시가 거래 뒤 보유로 번 수익(마지막 날은 없음).

시점 규칙(어기면 안 됨)
  - 종가 신호: t일 종가까지의 정보로 정한 값은 t 다음 거래일 시가부터 적용한다 → at_open(sig).
    at_open은 '그 거래일보다 엄격히 앞선 날짜'의 최신 값을 쓴다. 미국 장 마감(KST 05:00) 뒤 사용자가 보고 다음 시가에 체결.
  - FRED 시계열은 관측일 다음 날 공개되는 것이 많아 fred_known()이 하루 더 늦춰 준다.
  - 기본 경로 자체의 종가 평가값(base.close)도 t일 종가에 알 수 있다(그림자 경로 신호에 사용).
  - 장중 손절 주문(종목별)은 StockEngine으로만 잰다(고가·저가 필요).

평가 구간(측정용. 구간 경계는 사후에 기본 경로 낙폭으로 정했으므로 신호에 쓰지 않는다)
  2026: 전체 1/2~10/9, 앞 1/2~5/29(고르기), 뒤 6/1~10/9(확인), AI쇼크 6/22~7/29(7종목 고점→저점), 7~9월, 9/1~10/9
  보유(주간 PDF 2023-05-22~2025-12-31, 보조): 2025, 2024, 딥시크 25/1/23~1/27, 관세 25/2/18~4/4, 25가을 10/13~11/21,
  24여름 6/18~8/5, 23가을 8/1~10/26

고르기 규칙(갈래 공통, 사전 고정)
  1) 앞 구간 자산이 기본의 90% 이상(= 수익 유지 조건)인 변형 중 앞 구간 최대 낙폭이 가장 얕은 것. 같으면 앞 수익이 큰 것.
  2) 통과 변형이 없으면 그 갈래는 '앞 구간 탈락'.
  3) 고른 변형을 고정한 채 뒤 구간·AI쇼크·보유 기간(2025·2024 쇼크) 결과를 보고한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"

P26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30"), "9/1~": ("2026-09-01", "2026-10-09")}
PHOLD = {"전체": ("2023-05-22", "2025-12-31"), "2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"),
         "딥시크": ("2025-01-23", "2025-01-27"), "관세": ("2025-02-18", "2025-04-04"), "25가을": ("2025-10-13", "2025-11-21"),
         "24여름": ("2024-06-18", "2024-08-05"), "23가을": ("2023-08-01", "2023-10-26")}
PERIODS = {"2026": P26, "hold": PHOLD, "774": P26}
FILES = {"2026": "base_2026", "hold": "base_hold", "774": "base_2026_774"}


# ----------------------------------------------------------------------------- 자료
@dataclass
class Base:
    name: str
    days: list
    open: np.ndarray
    close: np.ndarray
    weights: list
    ticker: dict
    sector: dict
    ret: pd.Series = field(init=False)          # index = day k, 시가 k → 시가 k+1
    close_s: pd.Series = field(init=False)      # t일 종가 평가값(그림자 경로)

    def __post_init__(self):
        v = pd.Series(self.open, index=self.days)
        self.ret = (v.shift(-1) / v - 1).dropna()
        self.close_s = pd.Series(self.close, index=self.days)

    @property
    def periods(self):
        return PERIODS[self.name]

    def rebalance_days(self):
        out = [self.days[0]]
        for a, b in zip(self.days, self.days[1:]):
            if date.fromisoformat(a).isocalendar()[:2] != date.fromisoformat(b).isocalendar()[:2]:
                out.append(b)
        return out


_BASES: dict = {}


def base(name: str = "2026") -> Base:
    if name not in _BASES:
        b = json.loads((CACHE / f"{FILES[name]}.json").read_text(encoding="utf-8"))
        _BASES[name] = Base(name, b["days"], np.array(b["open"]), np.array(b["close"]), b["weights"], b["ticker"],
                            b["sector"])
    return _BASES[name]


_OHLC = None


def ohlc() -> pd.DataFrame:
    """long 형식 일봉(date, sym, Open, High, Low, Close, Adj Close, Volume). 분할 수정, 배당 미수정(Adj Close는 배당 수정)."""
    global _OHLC
    if _OHLC is None:
        _OHLC = pd.read_parquet(CACHE / "ohlc.parquet")
        _OHLC["date"] = pd.to_datetime(_OHLC["date"]).dt.strftime("%Y-%m-%d")
    return _OHLC


def px(sym: str, field: str = "Close") -> pd.Series:
    """한 종목·지수의 일봉 한 칸(날짜 문자열 index). 예: px('^VIX'), px('SOXX', 'Adj Close')."""
    o = ohlc()
    s = o[o.sym == sym].set_index("date")[field].astype(float).sort_index()
    return s[~s.index.duplicated()]


def wide(field: str = "Close", syms=None) -> pd.DataFrame:
    o = ohlc()
    if syms is not None:
        o = o[o.sym.isin(list(syms))]
    return o.pivot_table(index="date", columns="sym", values=field).sort_index()


def breadth() -> pd.DataFrame:
    """watchlist.AI_UNIVERSE 미국 상장 115종목 Adj Close(wide). 시장 폭 계산용(생존편향: 2026-10 기준 목록)."""
    d = pd.read_parquet(CACHE / "breadth.parquet")
    d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
    return d.sort_index()


def fred_known() -> pd.DataFrame:
    """FRED 시계열을 '알 수 있는 날' 기준으로 하루 늦춘 값(관측일 t → t+1 영업일에 사용 가능으로 본다).
    주간 시계열(STLFSI4·NFCI)은 관측일(금요일 기준) 뒤 다음 주에 나오므로 7일 늦춘다."""
    f = pd.read_parquet(CACHE / "fred.parquet")
    f.index = pd.to_datetime(f.index)
    out = {}
    for c in f.columns:
        s = f[c].dropna()
        lag = pd.Timedelta(days=7) if c in ("STLFSI4", "NFCI") else pd.Timedelta(days=1)
        s.index = (s.index + lag).strftime("%Y-%m-%d")
        out[c] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(out).sort_index()


# ----------------------------------------------------------------------------- 시점 맞추기
def at_open(sig: pd.Series, b: Base, fill=1.0) -> pd.Series:
    """종가 신호(index=그 값을 알게 된 날짜) → 각 거래일 k 시가에 쓸 값 = k보다 엄격히 앞선 날짜의 최신 값."""
    s = sig.dropna().sort_index()
    idx = np.searchsorted(s.index.to_numpy(), np.asarray(b.days), side="left") - 1
    vals = np.where(idx >= 0, s.to_numpy()[np.clip(idx, 0, None)], np.nan) if len(s) else np.full(len(b.days), np.nan)
    return pd.Series(vals, index=b.days).fillna(fill)


def open_ret(sym: str, b: Base) -> pd.Series:
    """참고 종목(헤지 등)의 시가 k → 다음 기준 거래일 시가 수익. 기본 경로 ret와 같은 index."""
    o = px(sym, "Open").reindex(b.days).ffill()
    r = (o.shift(-1) / o - 1)
    return r.reindex(b.ret.index).fillna(0.0)


# ----------------------------------------------------------------------------- 평가
def overlay(b: Base, exposure: pd.Series, hedge: dict | None = None, cost: float = 0.0) -> pd.Series:
    """노출 비율 e[k](0~1, k일 시가에 맞춤)로 기본 경로를 줄인 일별 수익.
    hedge = {종목: 비중 시계열}: 그 종목을 k일 시가에 비중만큼 산다(인버스 ETF는 양수 비중). 합계 e+Σ헤지 ≤ 1(레버리지·공매도 없음).
    cost = 회전율 1 단위당 비용(편도, 예 0.001). 기본 0(수수료 0원). 스프레드 민감도로만 쓴다."""
    e = exposure.reindex(b.ret.index).astype(float)
    if e.isna().any():
        raise ValueError("exposure에 빈 날이 있다")
    r = e * b.ret
    turn = e.diff().abs().fillna((e.iloc[0] - 1.0).__abs__())
    if hedge:
        tot = e.copy()
        for sym, h in hedge.items():
            h = h.reindex(b.ret.index).astype(float).fillna(0.0)
            r = r + h * open_ret(sym, b)
            turn = turn + h.diff().abs().fillna(h.iloc[0])
            tot = tot + h
        if (tot > 1 + 1e-9).any():
            raise ValueError(f"주식+헤지 합계가 1을 넘는 날 {int((tot > 1 + 1e-9).sum())}일")
    return r - cost * turn


def stats(r: pd.Series, lo: str, hi: str) -> dict:
    x = r[(r.index >= lo) & (r.index <= hi)]
    if not len(x):
        return {"수익": float("nan"), "낙폭": float("nan")}
    c = np.concatenate([[1.0], (1 + x.to_numpy()).cumprod()])
    return {"수익": float(c[-1] - 1), "낙폭": float((c / np.maximum.accumulate(c) - 1).min())}


def metrics(b: Base, r: pd.Series, exposure: pd.Series | None = None) -> dict:
    out = {k: stats(r, *v) for k, v in b.periods.items()}
    if exposure is not None:
        e = exposure.reindex(b.ret.index).astype(float)
        out["평균노출"] = float(e.mean())
        out["빠짐횟수"] = int(((e.shift(1).fillna(1.0) >= 0.999) & (e < 0.999)).sum())
        out["회전율"] = float(e.diff().abs().sum())
    return out


def evaluate(name: str, exposure_by_base: dict, hedge_by_base: dict | None = None, params=None) -> dict:
    """exposure_by_base = {'2026': Series, 'hold': Series} (at_open을 거친 값). 각 기본 경로에서 지표 + 스프레드 0.1% 민감도."""
    res = {"name": name, "params": params or {}}
    for bn, e in exposure_by_base.items():
        b = base(bn)
        h = (hedge_by_base or {}).get(bn)
        r = overlay(b, e, h)
        m = metrics(b, r, e)
        m["전체_스프레드0.1%"] = stats(overlay(b, e, h, cost=0.001), *b.periods["전체"])["수익"]
        res[bn] = m
    return res


def base_result() -> dict:
    return evaluate("기본(7종목 현행)", {bn: pd.Series(1.0, index=base(bn).days) for bn in ("2026", "hold")})


def select_front(rows: list[dict], base_row: dict, keep: float = 0.90, min_gain: float = 0.01) -> dict | None:
    """사전 고정 고르기: 2026 앞 구간 자산 ≥ 기본의 keep배이고 앞 낙폭이 기본보다 min_gain(1%p) 이상 얕은 변형 중
    앞 낙폭 최소 → 앞 수익 최대. 없으면 None(= 앞 구간에서 '수익 유지 + 낙폭 개선'을 보인 변형 없음)."""
    bw, bd = 1 + base_row["2026"]["앞"]["수익"], base_row["2026"]["앞"]["낙폭"]
    ok = [r for r in rows if 1 + r["2026"]["앞"]["수익"] >= keep * bw and r["2026"]["앞"]["낙폭"] >= bd + min_gain]
    if not ok:
        return None
    return max(ok, key=lambda r: (round(r["2026"]["앞"]["낙폭"], 6), r["2026"]["앞"]["수익"]))


def evaluate_returns(name: str, r_by_base: dict, params=None, exposure_by_base: dict | None = None) -> dict:
    """이미 계산한 일별 수익(StockEngine 등)을 같은 지표로. r의 index는 기본 경로 ret와 같아야 한다."""
    res = {"name": name, "params": params or {}}
    for bn, r in r_by_base.items():
        b = base(bn)
        if not r.index.equals(b.ret.index):
            raise ValueError(f"{bn}: 수익 index가 기본 경로와 다르다")
        res[bn] = metrics(b, r, (exposure_by_base or {}).get(bn))
    return res


def trace(exposure: pd.Series, lo: str = "2026-06-15", hi: str = "2026-09-30") -> list:
    """노출이 바뀐 날만 [(날짜, 노출)] — 7월 쇼크 때 언제 줄이고 언제 돌아왔는지 보고용."""
    e = exposure[(exposure.index >= lo) & (exposure.index <= hi)].round(3)
    ch = e[e.ne(e.shift())]
    return [(d, float(v)) for d, v in ch.items()]


PICK_KEYS = (("2026", "전체"), ("2026", "앞"), ("2026", "뒤"), ("2026", "AI쇼크"), ("2026", "7~9월"), ("2026", "9/1~"),
             ("hold", "2025"), ("hold", "2024"), ("hold", "관세"), ("hold", "딥시크"), ("hold", "24여름"), ("hold", "25가을"),
             ("hold", "23가을"))


def pick_row(label: str, r: dict | None) -> dict | None:
    if r is None:
        return None
    out = {"label": label, "name": r["name"], "params": json.dumps(r.get("params", {}), ensure_ascii=False, default=str)}
    for bn, k in PICK_KEYS:
        if bn in r and k in r[bn]:
            out[f"{bn}_{k}_수익"] = round(r[bn][k]["수익"], 4)
            out[f"{bn}_{k}_낙폭"] = round(r[bn][k]["낙폭"], 4)
    out["2026_평균노출"] = round(r["2026"].get("평균노출", float("nan")), 3) if "2026" in r else None
    out["2026_빠짐횟수"] = r["2026"].get("빠짐횟수") if "2026" in r else None
    return out


def summarize_family(family: str, rows: list[dict], grid_desc: str, extra: dict | None = None) -> dict:
    """갈래 공통 저장·출력: 앞 고르기(수익 유지 90%·80%), 사후 파레토(90%), 전 변형 표. results/<family>.json."""
    br = base_result()
    picks = {"keep90": select_front(rows, br, 0.90), "keep80": select_front(rows, br, 0.80),
             "keep70": select_front(rows, br, 0.70)}
    par = sorted(pareto_2026(rows, br, 0.90), key=lambda r: -r["2026"]["전체"]["수익"])
    payload = {"family": family, "grid_size": len(rows), "grid_desc": grid_desc,
               "base": pick_row("기본", br),
               "front_picks": {k: pick_row(f"앞 고르기 {k}", v) for k, v in picks.items()},
               "pareto_2026_expost": [pick_row("사후 파레토", r) for r in par[:10]],
               "rows": [pick_row("전 변형", r) for r in rows], "extra": extra or {}}
    p = save(family, payload)
    print(row_text(br))
    for k, v in picks.items():
        print(f"[앞 고르기 {k}]", row_text(v) if v else "없음")
    print(f"[사후 파레토 90%] {len(par)}개")
    for r in par[:10]:
        print("  ", row_text(r))
    print("저장:", p)
    return payload


def pareto_2026(rows: list[dict], base_row: dict, keep: float = 0.90) -> list[dict]:
    """사후(2026 전체를 본) 비교: 전체 자산 ≥ 기본의 keep배이면서 전체 낙폭이 기본보다 얕은 변형. 선택편향 있음."""
    bw, bd = 1 + base_row["2026"]["전체"]["수익"], base_row["2026"]["전체"]["낙폭"]
    return [r for r in rows if 1 + r["2026"]["전체"]["수익"] >= keep * bw and r["2026"]["전체"]["낙폭"] > bd]


def row_text(r: dict) -> str:
    a, h = r["2026"], r.get("hold", {})
    f = lambda d, k: f"{d[k]['수익']:+.1%}/{d[k]['낙폭']:.1%}" if k in d else "-"
    return (f"{r['name']:<40} 2026전체 {f(a, '전체')} 앞 {f(a, '앞')} 뒤 {f(a, '뒤')} AI쇼크 {a['AI쇼크']['수익']:+.1%} "
            f"7~9월 {a['7~9월']['수익']:+.1%} | 2025 {f(h, '2025')} 관세 {h['관세']['수익']:+.1%} 24여름 {h['24여름']['수익']:+.1%} "
            f"25가을 {h['25가을']['수익']:+.1%} | 노출 {a.get('평균노출', 1):.2f} 빠짐 {a.get('빠짐횟수', 0)}") if h else r["name"]


def save(family: str, payload: dict):
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{family}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- 종목별 손절(장중 주문)
class StockEngine:
    """기본 경로의 주간 목표를 그대로 두고, 종목별 손절 주문·재진입만 얹는 엔진.

    매 거래일 k:
      1) 재조정일이면 목표(기본 경로 그날 비중)로 맞춘다. policy.allow(key, k, st)가 False면 그 종목 몫은 현금.
      2) k-1일 종가까지 정보로 policy.exit_open(key, k, st)가 True면 k일 시가에 판다(현금).
      3) 손절로 나간 종목이 이번 주 목표에 남아 있고 policy.reenter(key, k, st)가 True면 k일 시가에 그 주 목표 비중까지(현금 한도) 다시 산다.
      4) 장중: policy.stop(key, k, st)가 준 가격 L에 손절 주문. 시가 ≤ L이면 시가에 체결, 저가 ≤ L이면 L×(1-slip)에 체결.
    st = 종목별 상태 dict: entry(산 가격), peak(산 뒤 최고 종가, k-1까지), high_peak(산 뒤 최고 고가), entry_k, stopped_k, stop_px.
    가격: open/close는 기본 경로와 같은 고정본(build_base), 고가·저가는 ohlc.parquet.
    정책이 아무것도 안 하면 기본 경로를 그대로 재현해야 한다(check_engine)."""

    def __init__(self, bn: str = "2026"):
        import sys
        sys.path.insert(0, str(HERE))
        import build_base as BB
        self.b = base(bn)
        self.book = BB.book_2026() if bn in ("2026", "774") else BB.book_hold()
        bk = self.book
        days = self.b.days
        self.k_of = {d: i for i, d in enumerate(bk.days)}
        keys = sorted({k for w in self.b.weights for k in w})
        self.keys = keys
        self.tk = {k: bk.ticker[k] for k in keys}
        idx = bk.days
        self.O = {k: bk.opn[self.tk[k]].to_numpy(float) if self.tk[k] in bk.opn else np.full(len(idx), np.nan) for k in keys}
        self.C = {k: bk.cls[self.tk[k]].to_numpy(float) if self.tk[k] in bk.cls else np.full(len(idx), np.nan) for k in keys}
        hi, lo = wide("High", self.tk.values()), wide("Low", self.tk.values())
        oo = wide("Open", self.tk.values())
        self.H, self.L = {}, {}
        for k in keys:
            t = self.tk[k]
            h = hi[t].reindex(idx).to_numpy(float) if t in hi else np.full(len(idx), np.nan)
            l = lo[t].reindex(idx).to_numpy(float) if t in lo else np.full(len(idx), np.nan)
            # 고정본 시가와 새로 받은 시가의 비(분할 수정 차이 대비)로 고가·저가를 고정본 단위에 맞춘다
            o2 = oo[t].reindex(idx).to_numpy(float) if t in oo else np.full(len(idx), np.nan)
            ratio = np.where(np.isfinite(o2) & (o2 > 0) & np.isfinite(self.O[k]), self.O[k] / np.where(o2 > 0, o2, 1), 1.0)
            self.H[k], self.L[k] = h * ratio, l * ratio
        self.reb = set(self.b.rebalance_days())
        # 첫 PDF가 시작일과 같은 날이면 run_model은 다음 거래일에 처음 산다(주 중간 첫 매수)
        self.reb |= {d for i, d in enumerate(days) if i and not self.b.weights[i - 1] and self.b.weights[i]}
        self.days = days

    def atr(self, key: str, k: int, n: int = 14) -> float:
        """k-1일까지 n일 평균 진폭(고가-저가, 전일 종가 갭 포함)."""
        H, L, C = self.H[key], self.L[key], self.C[key]
        lo = max(1, k - n)
        tr = [max(H[j] - L[j], abs(H[j] - C[j - 1]), abs(L[j] - C[j - 1])) for j in range(lo, k)
              if np.isfinite(H[j]) and np.isfinite(L[j]) and np.isfinite(C[j - 1])]
        return float(np.mean(tr)) if tr else float("nan")

    def run(self, policy=None, slip: float = 0.0):
        b, days = self.b, self.days
        w: dict = {}
        target: dict = {}
        st: dict = {}
        stopped: dict = {}          # 이번 주 목표인데 손절로 나간 종목 → 상태
        value = [1.0]
        events = []
        for i, day in enumerate(days):
            k = self.k_of[day]
            if i > 0:
                kp = self.k_of[days[i - 1]]
                gross, new = 0.0, {}
                for key, wt in w.items():
                    o0, o1 = self.O[key][kp], self.O[key][k]
                    r = (o1 / o0 - 1) if np.isfinite(o0) and np.isfinite(o1) and o0 > 0 else 0.0
                    if key in intraday_exit:
                        r = intraday_exit[key] / o0 - 1
                    gross += wt * r
                    new[key] = wt * (1 + r)
                tot = sum(new.values()) + (1 - sum(w.values()))
                w = {key: v / tot for key, v in new.items() if key not in intraday_exit}
                for key in intraday_exit:
                    stopped[key] = st.pop(key)
                value.append(value[-1] * (1 + gross))
                for key in w:        # 산 뒤 최고 종가·고가(k-1일까지)
                    c, h = self.C[key][kp], self.H[key][kp]
                    if np.isfinite(c):
                        st[key]["peak"] = max(st[key]["peak"], c)
                    if np.isfinite(h):
                        st[key]["high_peak"] = max(st[key]["high_peak"], h)
            intraday_exit = {}
            if day in self.reb:
                target = dict(b.weights[i])
                stopped = {}
                nw = {}
                for key, wt in target.items():
                    if policy is not None and not policy.allow(key, k, st.get(key), self):
                        continue
                    nw[key] = wt
                for key in nw:
                    if key not in w:
                        o = self.O[key][k]
                        st[key] = {"entry": o, "peak": o, "high_peak": o, "entry_k": k}
                for key in list(st):
                    if key not in nw:
                        st.pop(key)
                w = nw
            elif policy is not None:
                for key in list(w):
                    if policy.exit_open(key, k, st[key], self):
                        events.append({"day": day, "key": self.tk[key], "type": "exit_open", "px": self.O[key][k]})
                        w.pop(key)
                        s = st.pop(key)
                        s["stopped_k"] = k
                        stopped[key] = s
                for key in list(stopped):
                    if key in target and key not in w and policy.reenter(key, k, stopped[key], self):
                        room = 1 - sum(w.values())
                        add = min(target[key], room)
                        o = self.O[key][k]
                        if add > 1e-9 and np.isfinite(o):
                            w[key] = add
                            st[key] = {"entry": o, "peak": o, "high_peak": o, "entry_k": k}
                            stopped.pop(key)
                            events.append({"day": day, "key": self.tk[key], "type": "reenter", "px": o})
            if policy is not None:
                for key in list(w):
                    L = policy.stop(key, k, st[key], self)
                    if L is None or not np.isfinite(L):
                        continue
                    o, lo = self.O[key][k], self.L[key][k]
                    if np.isfinite(o) and o <= L:
                        fill = o * (1 - slip)
                    elif np.isfinite(lo) and lo <= L:
                        fill = L * (1 - slip)
                    else:
                        continue
                    intraday_exit[key] = fill
                    st[key]["stopped_k"] = k
                    st[key]["stop_px"] = fill
                    events.append({"day": day, "key": self.tk[key], "type": "stop", "px": fill, "level": L})
        v = pd.Series(value, index=days)
        r = (v.shift(-1) / v - 1).dropna()
        return r, events


class NoPolicy:
    def allow(self, key, k, st, eng): return True
    def exit_open(self, key, k, st, eng): return False
    def reenter(self, key, k, st, eng): return False
    def stop(self, key, k, st, eng): return None


def check_engine():
    for bn in ("2026", "hold"):
        eng = StockEngine(bn)
        r, ev = eng.run(NoPolicy())
        d = float(np.max(np.abs(r.to_numpy() - base(bn).ret.to_numpy())))
        print(bn, "StockEngine(정책 없음) vs 기본 ret 최대 차", d, "사건", len(ev))
        lo_bad = sum(int(np.nansum(eng.L[k] > eng.O[k] * 1.0001)) for k in eng.keys)
        print("   저가>시가 인 봉 수(자료 이상)", lo_bad)


if __name__ == "__main__":
    import sys
    br = base_result()
    print(row_text(br))
    if sys.argv[1:] == ["check"]:
        check_engine()
