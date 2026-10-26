"""추세 등급 연구 공통 채점기 (2026-10-12).

질문: 두 펀드(TIME·KoAct) 미국 보유 종목을 가격·이평선만으로 매일 S~D로 나눌 때, 어떤 점수가
'지금 강한 종목'을 가장 일관되게 고르는가. 점수 함수 하나를 받아 같은 잣대로 잰다.

점수 함수 계약
    score_fn(P) -> DataFrame (날짜 × 종목), 클수록 강한 추세. P는 load_panel()의 dict
    (O·H·L·C·V 넓은 표, 분할 반영 일봉). t일 값은 t일 종가까지의 자료만 써야 한다.
    no_lookahead(score_fn)가 잘라 낸 자료로 다시 계산해 같은지 확인한다.

시점
    t일 종가로 점수 → t+1 시가 매수 → t+1+h 시가 매도 (h = 5·10·20거래일).
    보유 종목: t일 이전(같은 날 포함) 최신 PDF. PDF D는 D 21:45 KST 공개 → 미국 D 장 전에 안다.
    2026 이전 구간은 2026 보유 이력 전체(150종목)를 쓴다(생존 편향, 보조 확인용).

구간 (매도일까지 구간 안)
    front 2026-01-02~05-29 (고르는 구간) / back 2026-06-01~끝 (확인 구간) / y2025 / y2024(DRUG 제외)

주 지표(사전 등록): front의 날짜별 Spearman IC(h=20) 평균. 보조: 상위 20% 초과수익, 등급별
평균(피라미드 10/20/40/20/10), 이후 20일 최대 낙폭, 등급 바뀜, S·S+A 주간 보유 성과.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
PANEL = os.path.join(ROOT, "data", "derived", "analysis", "rsi_breakout", "cache", "1d_long.parquet")
SNAPS = os.path.join(ROOT, "data", "normalized", "etf_holdings")
ALIASES = {"GOOGL": "GOOG", "SATS": "ECHO", "SKHYV": "SKHY", "MOG/A": "MOG-A"}
BENCH = "QQQ"
MIN_BARS = 130
HORIZONS = (5, 10, 20)
WINDOWS = {
    "front": ("2026-01-02", "2026-05-29"),
    "back": ("2026-06-01", "2026-12-31"),
    "y2025": ("2025-01-02", "2025-12-31"),
    "y2024": ("2024-01-02", "2024-12-31"),
    # 사후 진단(선택에 쓰지 않음): 6~8월 충격 뒤 최근 장세. h20은 날짜가 모자라 비고 h5·h10만 나온다.
    "late": ("2026-09-01", "2026-12-31"),
}
EXCLUDE = {"y2024": {"DRUG"}}  # 2024-10-15 하루 +1446% (trend_zones 보고서 한계 절)
PYRAMID = (("S", 0.90), ("A", 0.70), ("B", 0.30), ("C", 0.10))  # 백분위 하한, 그 아래 D
QUINTILE = (("S", 0.80), ("A", 0.60), ("B", 0.40), ("C", 0.20))
GRADES = ("S", "A", "B", "C", "D")


@lru_cache(maxsize=1)
def load_panel() -> dict:
    df = pd.read_parquet(PANEL)
    df["ts"] = pd.to_datetime(df["ts"])
    out = {}
    for k, col in (("O", "Open"), ("H", "High"), ("L", "Low"), ("C", "Close"), ("V", "Volume")):
        w = df.pivot(index="ts", columns="sym", values=col).sort_index()
        out[k] = w
    out["bench"] = {k: out[k].pop(BENCH) for k in ("O", "H", "L", "C", "V")}
    return out


def _snapshot_members() -> pd.DataFrame:
    rows = {}
    for f in sorted(glob.glob(os.path.join(SNAPS, "2026-*.json"))):
        day = os.path.basename(f)[:10]
        d = json.load(open(f, encoding="utf-8"))
        syms = rows.setdefault(day, set())
        for h in d.get("holdings", []):
            p = str(h.get("code", "")).split()
            if h.get("asset_type") == "security" and len(p) >= 3 and p[1] == "US":
                syms.add(ALIASES.get(p[0], p[0]))
    return rows


@lru_cache(maxsize=1)
def membership() -> pd.DataFrame:
    """날짜 × 종목 bool. 2026-01-02 전에는 패널 전 종목(150)."""
    P = load_panel()
    cal, cols = P["C"].index, P["C"].columns
    snaps = _snapshot_members()
    days = sorted(snaps)
    m = pd.DataFrame(False, index=cal, columns=cols)
    m.loc[cal < pd.Timestamp(days[0]), :] = True
    j = 0
    for t in cal[cal >= pd.Timestamp(days[0])]:
        while j + 1 < len(days) and pd.Timestamp(days[j + 1]) <= t:
            j += 1
        members = [s for s in snaps[days[j]] if s in m.columns]
        m.loc[t, members] = True
    return m


def universe_mask(P: dict) -> pd.DataFrame:
    """점수 안에서 단면 순위를 낼 때 쓰는 t일까지 아는 종목군(보유 + 봉 130개 이상).
    eligible()은 다음 날 시가가 있는지까지 보므로 채점 전용이다(점수 계산에 쓰면 미래 정보)."""
    C = P["C"]
    m = membership().reindex(index=C.index, columns=C.columns).fillna(False).astype(bool)
    return m & (C.notna().cumsum() >= MIN_BARS) & C.notna()


@lru_cache(maxsize=1)
def eligible() -> pd.DataFrame:
    P = load_panel()
    bars = P["C"].notna().cumsum()
    nxt_open = P["O"].shift(-1).notna()
    return membership() & (bars >= MIN_BARS) & P["C"].notna() & nxt_open


@lru_cache(maxsize=4)
def forward(h: int) -> pd.DataFrame:
    """t 신호 → t+1 시가 매수, t+1+h 시가 매도 수익."""
    O = load_panel()["O"]
    return O.shift(-(1 + h)) / O.shift(-1) - 1.0


@lru_cache(maxsize=1)
def forward_mdd20() -> pd.DataFrame:
    """t+1 시가 매수 뒤 20거래일 안 최저 저가 대비 낙폭."""
    P = load_panel()
    low_min = P["L"][::-1].rolling(20, min_periods=1).min()[::-1].shift(-1)
    return low_min / P["O"].shift(-1) - 1.0


def exit_date(h: int) -> pd.Series:
    cal = load_panel()["C"].index
    s = pd.Series(cal, index=cal).shift(-(1 + h))
    return s


def window_dates(name: str, h: int) -> pd.DatetimeIndex:
    a, b = WINDOWS[name]
    cal = load_panel()["C"].index
    ex = exit_date(h)
    ok = (cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b)) & ex.notna().to_numpy()
    ok &= (ex.fillna(pd.Timestamp("2100-01-01")) <= pd.Timestamp(b)).to_numpy()
    return cal[ok]


def pct_rank(score: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    s = score.where(mask)
    return s.rank(axis=1, pct=True, method="average")


def grade_of(pr: pd.DataFrame, cuts=PYRAMID) -> pd.DataFrame:
    vals = pr.to_numpy(float)
    arr = np.full(vals.shape, None, dtype=object)
    finite = np.isfinite(vals)
    arr[finite] = "D"
    for name, floor in reversed(cuts):  # C, B, A, S 순으로 덮어쓴다
        arr[finite & (vals > floor)] = name
    return pd.DataFrame(arr, index=pr.index, columns=pr.columns)


def newey_west_t(x: pd.Series, lag: int) -> float:
    x = x.dropna().to_numpy(float)
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    return float(mu / np.sqrt(var / n)) if var > 0 else float("nan")


def _row_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    ok = a.notna() & b.notna()
    ra = a.where(ok).rank(axis=1)
    rb = b.where(ok).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    out = num / den
    return out.where(ok.sum(axis=1) >= 10)


def weekly_portfolio(pr: pd.DataFrame, lo: float, start: str, end: str, mask: pd.DataFrame) -> dict:
    """매주 첫 거래일 시가에 직전 종가 백분위 > lo 종목을 같은 비중으로 사서 다음 주 첫 시가까지 보유.
    매일 종가로 평가해 최대 낙폭을 잰다. 보유 0이면 그 주는 현금."""
    P = load_panel()
    O, C = P["O"], P["C"]
    cal = C.index
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(days) < 3:
        return {}
    week = pd.Series(days.to_period("W-SUN"), index=days)
    firsts = [d for d, w in zip(days, week) if d == week[week == w].index[0]]
    eq, curve, picks = 1.0, [], []
    for i, w0 in enumerate(firsts):
        w1 = firsts[i + 1] if i + 1 < len(firsts) else None
        prev = cal[cal.get_loc(w0) - 1]
        row = pr.loc[prev]
        names = row[(row > lo) & mask.loc[prev]].index
        names = [n for n in names if np.isfinite(O.at[w0, n])]
        picks.append(len(names))
        span = days[(days >= w0) & ((days < w1) if w1 is not None else (days <= days[-1]))]
        if not names:
            curve += [eq] * len(span)
            continue
        entry = O.loc[w0, names]
        for d in span:
            curve.append(eq * float((C.loc[d, names] / entry).mean(skipna=True)))
        if w1 is not None:
            ex = O.loc[w1, names].fillna(C.loc[span[-1], names])
            eq *= float((ex / entry).mean(skipna=True))
        else:
            eq = curve[-1]
    curve = pd.Series(curve)
    mdd = float((curve / curve.cummax() - 1).min()) if len(curve) else float("nan")
    return {"ret": round((eq - 1) * 100, 2), "mdd": round(mdd * 100, 2), "avg_n": round(float(np.mean(picks)), 1)}


def bench_weekly(start: str, end: str, which: str) -> dict:
    """비교: 같은 주간 방식의 보유 종목 전체 같은 비중(universe) 또는 QQQ."""
    P = load_panel()
    if which == "universe":
        pr = pd.DataFrame(1.0, index=P["C"].index, columns=P["C"].columns)
        return weekly_portfolio(pr, 0.0, start, end, eligible())
    O, C = P["bench"]["O"], P["bench"]["C"]
    s = C[(C.index >= pd.Timestamp(start)) & (C.index <= pd.Timestamp(end))]
    first = O[O.index >= pd.Timestamp(start)].iloc[0]
    curve = s / first
    return {"ret": round((float(curve.iloc[-1]) - 1) * 100, 2),
            "mdd": round(float((curve / curve.cummax() - 1).min()) * 100, 2)}


def evaluate(score: pd.DataFrame, windows=("front", "back", "y2025", "y2024", "late"), portfolio=True) -> dict:
    """한 점수의 모든 지표. 값은 % 단위(IC 제외)."""
    P = load_panel()
    score = score.reindex(index=P["C"].index, columns=P["C"].columns)
    mask = eligible() & score.notna()
    pr = pct_rank(score, mask)
    gp = grade_of(pr, PYRAMID)
    gq = grade_of(pr, QUINTILE)
    mdd = forward_mdd20()
    res = {}
    for wname in windows:
        excl = EXCLUDE.get(wname, set())
        cols = [c for c in P["C"].columns if c not in excl]
        out = {}
        for h in HORIZONS:
            dates = window_dates(wname, h)
            if len(dates) < 10:
                continue
            fr = forward(h).loc[dates, cols].where(mask.loc[dates, cols])
            sc = pr.loc[dates, cols]
            ic = _row_spearman(sc, fr).dropna()
            mean_all = fr.mean(axis=1)
            top = fr.where(sc > 0.8).mean(axis=1) - mean_all
            bot = fr.where(sc <= 0.2).mean(axis=1) - mean_all
            d = {"ic": round(float(ic.mean()), 4), "ic_t": round(newey_west_t(ic, h), 2),
                 "ic_pos": round(float((ic > 0).mean()) * 100, 1), "days": int(len(ic)),
                 "top20": round(float(top.mean()) * 100, 3), "bot20": round(float(bot.mean()) * 100, 3),
                 "top20_hit": round(float((top > 0).mean()) * 100, 1)}
            if h == 20:
                g = gp.loc[dates, cols]
                d["pyramid"] = {k: round(float((fr.where(g == k).mean(axis=1) - mean_all).mean()) * 100, 3)
                                for k in GRADES}
                gq_ = gq.loc[dates, cols]
                d["quintile"] = {k: round(float((fr.where(gq_ == k).mean(axis=1) - mean_all).mean()) * 100, 3)
                                 for k in GRADES}
                md = mdd.loc[dates, cols].where(mask.loc[dates, cols])
                d["mdd20_S"] = round(float(md.where(g == "S").mean(axis=1).mean()) * 100, 2)
                d["mdd20_D"] = round(float(md.where(g == "D").mean(axis=1).mean()) * 100, 2)
                d["mdd20_all"] = round(float(md.mean(axis=1).mean()) * 100, 2)
            out[f"h{h}"] = d
        # 등급 바뀜: 하루에 등급이 바뀐 종목 비율(피라미드)
        dates = window_dates(wname, 1)
        g = gp.loc[dates, cols]
        prev = gp.shift(1).loc[dates, cols]
        both = g.notna() & prev.notna()
        out["turnover"] = round(float(((g != prev) & both).sum().sum() / max(both.sum().sum(), 1)) * 100, 1)
        if portfolio:
            a, b = WINDOWS[wname]
            last = P["C"].index[-1].strftime("%Y-%m-%d")
            b = min(b, last)
            out["port_S"] = weekly_portfolio(pr, 0.90, a, b, mask)
            out["port_SA"] = weekly_portfolio(pr, 0.70, a, b, mask)
        res[wname] = out
    return res


@lru_cache(maxsize=1)
def benchmarks() -> dict:
    P = load_panel()
    last = P["C"].index[-1].strftime("%Y-%m-%d")
    out = {}
    for wname, (a, b) in WINDOWS.items():
        b = min(b, last)
        out[wname] = {"universe": bench_weekly(a, b, "universe"), "QQQ": bench_weekly(a, b, "QQQ")}
    return out


def no_lookahead(score_fn, cutoffs=("2026-03-13", "2026-07-22", "2026-09-30")) -> bool:
    """자료를 cutoff까지 잘라 다시 계산한 마지막 행이 전체 계산의 그 행과 같은지."""
    P = load_panel()
    full = score_fn(P)
    for c in cutoffs:
        t = pd.Timestamp(c)
        cut = {k: (v.loc[:t] if isinstance(v, pd.DataFrame) else v) for k, v in P.items() if k != "bench"}
        cut["bench"] = {k: v.loc[:t] for k, v in P["bench"].items()}
        part = score_fn(cut)
        a, b = full.loc[t], part.loc[t].reindex(full.columns)
        same = (a.isna() & b.isna()) | (np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12))
        if not bool(same.all()):
            return False
    return True


def summary_line(name: str, r: dict) -> str:
    f, b = r.get("front", {}), r.get("back", {})
    g = lambda w, h, k: w.get(h, {}).get(k)
    return (f"{name:40s} front IC20 {g(f,'h20','ic')} (t {g(f,'h20','ic_t')}) top20 {g(f,'h20','top20')} | "
            f"back IC20 {g(b,'h20','ic')} (t {g(b,'h20','ic_t')}) top20 {g(b,'h20','top20')} | "
            f"2025 IC20 {g(r.get('y2025', {}),'h20','ic')} | S주간 front {f.get('port_S', {}).get('ret')} back {b.get('port_S', {}).get('ret')}")


RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def save_results(family: str, rows: list[dict]) -> str:
    """rows: [{"name", "spec", "result"(evaluate 출력), "no_lookahead"}] → results/<family>.json"""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{family}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"family": family, "benchmarks": benchmarks(), "rows": rows}, fh, ensure_ascii=False, indent=1)
    return path


if __name__ == "__main__":
    P = load_panel()
    m = membership()
    e = eligible()
    print("panel", P["C"].shape, P["C"].index[0].date(), P["C"].index[-1].date())
    for w in WINDOWS:
        a, b = WINDOWS[w]
        sl = e.loc[a:b]
        print(w, "dates", len(sl), "eligible per day mean", round(float(sl.sum(axis=1).mean()), 1),
              "min", int(sl.sum(axis=1).min()))
    print(json.dumps(benchmarks(), ensure_ascii=False))
    # 기준 점수: 20일 수익률(단순 모멘텀)
    mom = lambda P: P["C"] / P["C"].shift(20) - 1
    assert no_lookahead(mom)
    print(summary_line("ret20 (sanity)", evaluate(mom(P))))
