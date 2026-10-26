"""사용자 아이디어(2026-10-10): 급락일에 특정 시각에 사면 어떤가 — 정규장을 5분마다 훑는다.

대상: 그 시각(봉이 끝나는 시각 T, 09:35~15:55 ET)에 전일 종가 대비 -7% 이하인 종목·날(그 순간 급락 목록에 있는 것).
      lowfind 사건(그날 한 번이라도 -7%)에서 고르므로 T에 -7% 이하인 종목은 모두 들어 있다(미래 정보 없음).
매수: T에 끝난 봉의 종가. 앞·뒤 5분봉, 뒤는 1분봉(그 시각 가격을 정확히)으로도 잰다.
매도(lowfind.outcome과 같은 판정):
  +2%          +2% 지정가(다음 봉부터 고가), 못 닿으면 다음 날 시가.
  +2%·손절     위에 '매수 때까지 그날 저가 -1%' 손절을 하루 종일(같은 봉이면 손절 먼저, 시가가 이미 아래면 시가).
  종가         그날 종가에 판다(시각 효과만 본다).
  다음날시가
사용: py timeslot.py run → results/timeslot.txt, timeslot.csv"""
import io, os, sys
import numpy as np, pandas as pd
from lowfind import RES, CRASH, AWAKE_END, regular_bars

SLOTS = list(range(9 * 60 + 35, 16 * 60, 5))          # 09:35 ~ 15:55, 77개
TP, STOP_BELOW = 0.02, 0.01
CELLS = (("앞", 5), ("뒤", 5), ("뒤", 1))
METRICS = ("+2%", "+2%·손절", "종가", "다음날시가")


def first_true(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def slot_rows(b, c0, nxt, minutes, key):
    end = (b.index.hour * 60 + b.index.minute).to_numpy() + minutes
    lo, hi, cl, op = (b[c].to_numpy() for c in ("Low", "High", "Close", "Open"))
    run_low = np.minimum.accumulate(lo)
    fallback = nxt if nxt == nxt else cl[-1]
    rows = []
    for t in SLOTS:
        i0 = int(np.searchsorted(end, t, side="right")) - 1     # T까지 끝난 마지막 봉(5분 안에 끝난 것만)
        if i0 < 0 or end[i0] <= t - 5 or cl[i0] / c0 - 1 > CRASH:
            continue
        px = cl[i0]
        stop = run_low[i0] * (1 - STOP_BELOW)
        tp_j = first_true(hi[i0 + 1:] >= px * (1 + TP))
        st_j = first_true(lo[i0 + 1:] <= stop)
        plain = TP if tp_j is not None else fallback / px - 1
        if st_j is not None and (tp_j is None or st_j <= tp_j):
            guarded = min(stop, op[i0 + 1 + st_j]) / px - 1
        else:
            guarded = plain
        rows.append(dict(**key, t=t, dd=cl[i0] / c0 - 1, near=px / run_low[i0] - 1, hit=tp_j is not None,
                         **{"+2%": plain, "+2%·손절": guarded, "종가": cl[-1] / px - 1,
                            "다음날시가": (nxt / px - 1) if nxt == nxt else np.nan}))
    return rows


def run():
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    rows = []
    for t, d, win in zip(ev.ticker, ev.day, ev.win):
        for minutes in ((5, 1) if win == "뒤" else (5,)):
            got = regular_bars(t, d, minutes)
            if got is None:
                continue
            b, c0, nxt = got
            rows += slot_rows(b, c0, nxt, minutes, dict(ticker=t, day=d, win=win, bar=minutes))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "timeslot.csv"), index=False)
    report(df)


def tday(x, col):
    g = x.dropna(subset=[col]).groupby("day")[col].mean()
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan


def hhmm(m, shift=0):
    m = (m + shift * 60) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def report(df):
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("[특정 시각 매수 — 그 시각 -7% 이하인 종목을 그 봉 종가에 산다. 칸: 평균(+2% 익절 닿음) · 종목 수]")
    p("  시각 ET(한국)  | " + " | ".join(f"{w}{b}분 +2% / +2%·손절 / 종가 / 다음날시가" for w, b in CELLS))
    for t in SLOTS:
        cells = []
        for w, b in CELLS:
            x = df[(df.t == t) & (df.win == w) & (df.bar == b)]
            if len(x) < 20:
                cells.append(f"({len(x)}건)")
                continue
            cells.append(f"{x['+2%'].mean():+.2%}({x.hit.mean():.0%}) / {x['+2%·손절'].mean():+.2%} / {x['종가'].mean():+.2%} / "
                         f"{x['다음날시가'].mean():+.2%} · {len(x)}")
        p(f"  {hhmm(t)}({hhmm(t, 13)}){'*' if t <= AWAKE_END else ' '} | " + " | ".join(cells))
    p("  * = 서머타임 한국 24:00 전(사용자가 깨어 있는 시간). 표준시(11/1부터)는 10:00 ET까지.")
    # 30분 묶음: 칸마다 종목·날 한 번(묶음 안 첫 시각)
    p("\n[30분 묶음 — 묶음 안에서 처음 -7% 이하였던 시각에 한 번 산다. 평균·t(일)]")
    bands = [(s, s + 30) for s in range(9 * 60 + 30, 16 * 60, 30)]
    for w, b in CELLS:
        g0 = df[(df.win == w) & (df.bar == b)]
        p(f"  ==== {w}{b}분 ====")
        for s, e in bands:
            x = g0[(g0.t > s) & (g0.t <= e)].sort_values("t").drop_duplicates(["ticker", "day"])
            if len(x) < 20:
                continue
            parts = [f"{m} {x[m].mean():+.2%}(t {tday(x, m):+.1f})" for m in METRICS]
            p(f"    {hhmm(s + 5)}~{hhmm(e)} ET({hhmm(s + 5, 13)}~{hhmm(e, 13)}) n={len(x):>4} 익절 {x.hit.mean():.0%} · 저가 대비 중앙 {x.near.median():+.1%} · " + " · ".join(parts))
    # 앞 기간으로 고르고 뒤 기간으로 확인
    p("\n[앞 기간 +2% 평균 상위 8개 시각(n ≥ 30) → 뒤 기간]")
    a = df[(df.win == "앞") & (df.bar == 5)].groupby("t").agg(n=("+2%", "size"), avg=("+2%", "mean"))
    for t in a[a.n >= 30].sort_values("avg", ascending=False).index[:8]:
        line = []
        for w, b in CELLS:
            x = df[(df.t == t) & (df.win == w) & (df.bar == b)]
            line.append(f"{w}{b} {x['+2%'].mean():+.2%}(n {len(x)}, t {tday(x, '+2%'):+.1f})")
        p(f"  {hhmm(t)} ET({hhmm(t, 13)}): " + " · ".join(line))
    p(f"\n시험 범위: 시각 {len(SLOTS)}개(5분 간격, 09:35~15:55) × 매도 {len(METRICS)}개 × 기간·봉 3칸. "
      f"30분 묶음 13개는 같은 자료의 요약이다. 익절 +2%·손절 저가 -1%는 앞선 연구와 같은 값이다.")
    txt = o.getvalue()
    open(os.path.join(RES, "timeslot.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def report_only():
    report(pd.read_csv(os.path.join(RES, "timeslot.csv")))


def drift():
    """급락주만 봤는데도 종가까지 더 빠지나: 같은 날·같은 시각 SPY의 그 시각→종가와 비교, 그 시각 하락 깊이별(5분봉)."""
    df = pd.read_csv(os.path.join(RES, "timeslot.csv"))
    df = df[df.bar == 5].copy()
    spy = pd.read_parquet(os.path.join(RES, "lowfind", "spy_5m.parquet"))
    spy.index = spy.index.tz_convert("America/New_York")
    mm = spy.index.hour * 60 + spy.index.minute
    spy = spy[(mm >= 570) & (mm < 960)]
    at = {}
    for day, g in spy.groupby(spy.index.date):
        end = (g.index.hour * 60 + g.index.minute).to_numpy() + 5
        at[str(day)] = (dict(zip(end.tolist(), g.Close.tolist())), float(g.Close.iloc[-1]))
    df["spy"] = [(at[d][1] / at[d][0][t] - 1) if d in at and t in at[d][0] else np.nan for d, t in zip(df.day, df.t)]
    df["excess"] = df["종가"] - df["spy"]
    df["depth"] = pd.cut(df.dd, [-1, -0.15, -0.10, -0.07], labels=["-15% 이하", "-10~-15%", "-7~-10%"])
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("[그 시각 → 그날 종가: 급락주 vs 같은 순간 SPY (모든 시각·종목 행, t는 날짜별 평균)]")
    for win in ("앞", "뒤"):
        x = df[df.win == win]
        p(f"  ==== {win} 5분봉 ({len(x)}행, {x.day.nunique()}일) ====")
        groups = [("전체", x)] + [(str(k), y) for k, y in x.groupby("depth", observed=True)]
        for lab, y in groups:
            p(f"  {lab:<9} 급락주 {y['종가'].mean():+.2%}(t {tday(y, '종가'):+.1f}) · 종가가 산 값보다 낮음 {(y['종가'] < 0).mean():.0%} · "
              f"SPY {y.spy.mean():+.2%} · 차이 {y.excess.mean():+.2%}(t {tday(y, 'excess'):+.1f})")
    txt = o.getvalue()
    open(os.path.join(RES, "timeslot_drift.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    {"run": run, "report": report_only, "drift": drift}[sys.argv[1]]()
