"""사용자 아이디어(2026-10-10): ASTS 10/9처럼 '지지선을 N개 무너뜨린 뒤 사면' 익절할 수 있나 — 지난 급락일에 시험.

선: 전날 정규장과 오늘 이미 확정된 피벗 고점·저점(좌우 W분). 0.5% 폭 안의 값은 한 선으로 묶는다.
    오늘 피벗은 오른쪽 W분이 지나야 보인다. 장 마감 뒤에 그린 차트에는 저점 뒤에 생긴 선이 섞이므로 그때 보이던 선만 센다.
무너진 선의 수: 봉 마감 때 오늘 고가 이하이고 종가보다 0.2% 넘게 위에 있는 선의 수(차트에서 세는 방식).
매수: 무너진 선이 처음 N개 이상이고 전일 종가 대비 -7% 이하인 봉의 종가(09:45~15:30 ET에 마감한 봉, 사건마다 한 번).
매도: lowfind.outcome 그대로 — +TP 지정가(다음 봉부터 고가로 판정), 손절 '매수 때까지 저가 -1%'(하루 종일·11:00 ET까지·없음),
      못 닿으면 다음 날 시가.
비교: 같은 사건·같은 매도로 -7%·-10%·-15% 첫 마감에 산 경우와 바닥 15분(NL15_2).
사용: py supports.py run → results/supports.txt · py supports.py asts → results/supports_asts.txt"""
import io, os, sys
import numpy as np, pandas as pd
from lowfind import RES, CRASH, AWAKE_END, regular_bars, outcome, stat, fmt, signals

W_GRID = (15, 30)            # 피벗 좌우 창(분): 5분봉 3·6봉, 3분봉 5·10봉
N_GRID = (3, 4, 5, 6, 7)
TOL = 0.005                  # 이 폭 안의 피벗은 한 선
BREAK = 0.002                # 종가가 선보다 이만큼 넘게 아래여야 무너진 것
TPS = (0.02, 0.03)
STOPS = ("none", "awake", "all")
BASE = (("첫 -7%", -0.07), ("첫 -10%", -0.10), ("첫 -15%", -0.15))


def pivots(b, k):
    """좌우 k봉 안에서 가장 낮은 저가·가장 높은 고가인 봉: [(위치, 값)]. 장 시작 쪽 왼쪽 창은 잘린다."""
    lo, hi = b.Low.to_numpy(), b.High.to_numpy()
    out = []
    for i in range(len(b) - k):                  # 오른쪽 k봉이 있어야 확정
        a = max(0, i - k)
        if lo[i] <= lo[a:i + k + 1].min():
            out.append((i, float(lo[i])))
        if hi[i] >= hi[a:i + k + 1].max():
            out.append((i, float(hi[i])))
    return out


def zones(prices):
    """아래에서부터 TOL 폭으로 묶은 선의 아래 끝."""
    out = []
    for p in sorted(prices):
        if not out or p > out[-1] * (1 + TOL):
            out.append(p)
    return out


def broken_counts(b, prev, minutes, w):
    """봉마다 그때 보이는 선 중 무너진 수, 그리고 봉마다 보이던 선(재생용)."""
    k = max(1, w // minutes)
    known = [p for _, p in pivots(prev, k)] if prev is not None and len(prev) else []
    today = sorted(pivots(b, k))
    hi, cl = b.High.to_numpy(), b.Close.to_numpy()
    sh = np.maximum.accumulate(hi)
    counts, seen = np.zeros(len(b), int), []
    zs, j = zones(known), 0
    for i in range(len(b)):
        added = False
        while j < len(today) and today[j][0] + k <= i:   # 오른쪽 k봉이 이 봉까지 마감했으면 보인다
            known.append(today[j][1])
            j += 1
            added = True
        if added:
            zs = zones(known)
        counts[i] = sum(1 for z in zs if z <= sh[i] and cl[i] < z * (1 - BREAK))
        seen.append(zs)
    return counts, seen


def first(mask):
    idx = np.nonzero(mask)[0]
    return int(idx[0]) if len(idx) else None


def trades_for(b, c0, nxt, minutes, prev, key):
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    cl = b.Close.to_numpy()
    run_low = np.minimum.accumulate(b.Low.to_numpy())
    ok = (end >= 585) & (end <= 930)
    dd = cl / c0 - 1
    picks, shape = [], dict(key)
    for w in W_GRID:
        counts, _ = broken_counts(b, prev, minutes, w)
        lo_i = int(np.argmin(b.Low.to_numpy()))
        shape[f"max{w}"] = int(counts[ok & (dd <= CRASH)].max()) if (ok & (dd <= CRASH)).any() else 0
        shape[f"atlow{w}"] = int(counts[lo_i])
        for n in N_GRID:
            i0 = first(ok & (dd <= CRASH) & (counts >= n))
            if i0 is not None:
                picks.append((f"선{n}개 W{w}", w, n, i0))
    for lab, th in BASE:
        i0 = first(ok & (dd <= th))
        if i0 is not None:
            picks.append((lab, 0, 0, i0))
    nl = signals(b, c0, minutes).get("NL15_2")
    if nl:
        picks.append(("바닥15(NL15_2)", 0, 0, nl[0]))
    rows = []
    for rule, w, n, i0 in picks:
        px, lowb = float(cl[i0]), float(run_low[i0])
        at = int(end[i0])
        for tp in TPS:
            for sm in STOPS:
                r, why = outcome(b, minutes, i0, px, lowb, nxt, tp, sm)
                rows.append(dict(**key, rule=rule, w=w, n=n, tp=tp, stop=sm, at=f"{at // 60:02d}:{at % 60:02d}",
                                 awake=at <= AWAKE_END, dd=px / c0 - 1, ret=r, why=why))
    return rows, shape


def run():
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    rows, shapes = [], []
    for t, d, win in zip(ev.ticker, ev.day, ev.win):
        for minutes in ((5, 3) if win == "뒤" else (5,)):
            got = regular_bars(t, d, minutes, with_prev=True)
            if got is None:
                continue
            b, c0, nxt, prev = got
            r, s = trades_for(b, c0, nxt, minutes, prev, dict(ticker=t, day=d, win=win, bar=minutes))
            rows += r
            shapes.append(s)
    tr, sh = pd.DataFrame(rows), pd.DataFrame(shapes)
    tr.to_csv(os.path.join(RES, "supports_trades.csv"), index=False)
    sh.to_csv(os.path.join(RES, "supports_shape.csv"), index=False)
    report(tr, sh)


CELLS = (("앞", 5), ("뒤", 5), ("뒤", 3))


def hhmm(at):
    """'HH:MM' 열의 중앙 시각."""
    if not len(at):
        return "—"
    m = int(np.median([int(s[:2]) * 60 + int(s[3:]) for s in at]))
    return f"{m // 60:02d}:{m % 60:02d}"


def report(tr, sh):
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("[무너진 선 수 — 사건 중 그날(09:45~15:30, -7% 이하 봉) N개 이상 무너진 비율 · 그날 저가 봉에서 센 수 중앙]")
    for (win, bar), g in sh.groupby(["win", "bar"], sort=False):
        for w in W_GRID:
            reach = " · ".join(f"{n}개 {(g[f'max{w}'] >= n).mean():.0%}" for n in N_GRID)
            p(f"  {win}{bar}분 W{w}: {reach} · 저가 봉 중앙 {g[f'atlow{w}'].median():.0f}개 (사건 {len(g)})")
    for only_awake in (False, True):
        x = tr[tr.awake] if only_awake else tr
        p(f"\n==== {'11:00 ET 전 신호만' if only_awake else '모든 신호(09:45~15:30)'} — 앞5 / 뒤5 / 뒤3, ★ = 세 칸 모두 평균 플러스 ====")
        order = [f"선{n}개 W{w}" for w in W_GRID for n in N_GRID] + [lab for lab, _ in BASE] + ["바닥15(NL15_2)"]
        for rule in order:
            g0 = x[x.rule == rule]
            if g0.empty:
                continue
            for tp in TPS:
                for sm in STOPS:
                    g = g0[(g0.tp == tp) & (g0.stop == sm)]
                    cells = [stat(g[(g.win == w_) & (g.bar == b_)]) for w_, b_ in CELLS]
                    star = all(c and c["avg"] > 0 for c in cells)
                    a = g[(g.win == "앞") & (g.bar == 5)]
                    p(f"  {'★' if star else ' '} {rule:<14} +{tp:.0%} 손절:{sm:<5} 앞5 매수 순간 중앙 {a.dd.median():+.1%}·{hhmm(a['at'])}")
                    for (w_, b_), c in zip(CELLS, cells):
                        p(f"       {w_}{b_} {fmt(c)}")
    p("\n[앞 기간(5분봉) 평균 상위 8개(n ≥ 30, 선 규칙만) → 뒤 기간]")
    cand = []
    for (rule, tp, sm, aw), g in pd.concat([tr.assign(aw=False), tr[tr.awake].assign(aw=True)]).groupby(["rule", "tp", "stop", "aw"]):
        if not rule.startswith("선"):
            continue
        a = stat(g[(g.win == "앞") & (g.bar == 5)])
        if a and a["n"] >= 30:
            cand.append((a["avg"], rule, tp, sm, aw, g))
    for _, rule, tp, sm, aw, g in sorted(cand, key=lambda z: -z[0])[:8]:
        p(f"  {rule} +{tp:.0%} 손절:{sm} {'11시 전' if aw else '모든 신호'}")
        for (w_, b_) in CELLS:
            p(f"       {w_}{b_} {fmt(stat(g[(g.win == w_) & (g.bar == b_)]))}")
    n_rules = len(W_GRID) * len(N_GRID)
    p(f"\n시험 범위: 선 규칙 {n_rules}개(W {len(W_GRID)} × N {len(N_GRID)}) × 익절 {len(TPS)} × 손절 {len(STOPS)} × 신호 시간 2 = "
      f"{n_rules * len(TPS) * len(STOPS) * 2}개. 선 묶음 폭 {TOL:.1%}·붕괴 {BREAK:.1%}는 결과 보기 전에 한 값으로 정했다. "
      f"비교 4개(첫 -7·-10·-15%, NL15_2)는 같은 매도.")
    txt = o.getvalue()
    open(os.path.join(RES, "supports.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def report_only():
    report(pd.read_csv(os.path.join(RES, "supports_trades.csv")), pd.read_csv(os.path.join(RES, "supports_shape.csv")))


# ---------------------------------------------------------------- ASTS 10/9 재생
def asts(t="ASTS", day="2026-10-09", prev_day="2026-10-08"):
    import yfinance as yf
    cache = os.path.join(RES, "supports")
    os.makedirs(cache, exist_ok=True)
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    for minutes, interval in ((5, "5m"), (3, "1m")):
        path = os.path.join(cache, f"{interval}_{t}_{day}.parquet")
        if not os.path.exists(path):
            x = yf.download(t, start=prev_day, end=(pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            interval=interval, prepost=False, progress=False, auto_adjust=False)
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = x.columns.get_level_values(0)
            x.index = x.index.tz_convert("America/New_York")
            x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(path)
        x = pd.read_parquet(path)
        mm = x.index.hour * 60 + x.index.minute
        x = x[(mm >= 570) & (mm < 960)]
        dates = np.asarray(x.index.date)
        b, prev = x[dates == pd.Timestamp(day).date()], x[dates == pd.Timestamp(prev_day).date()]
        if minutes == 3:
            from lowfind import _resample
            b, prev = _resample(b, 3), _resample(prev, 3)
        c0 = float(prev.Close.iloc[-1])
        lo_i = int(np.argmin(b.Low.to_numpy()))
        p(f"\n==== {t} {day} {minutes}분봉 · 전일 종가 {c0:.2f} · 시가 {b.Open.iloc[0]:.2f} · 저가 {b.Low.iloc[lo_i]:.2f}"
          f"({(b.index[lo_i] + pd.Timedelta(minutes=minutes)).strftime('%H:%M')}) · 종가 {b.Close.iloc[-1]:.2f} ====")
        end = (b.index.hour * 60 + b.index.minute).to_numpy() + minutes
        for w in W_GRID:
            counts, seen = broken_counts(b, prev, minutes, w)
            p(f"  W{w}: 저가 봉에서 무너진 선 {counts[lo_i]}개, 그때 보이던 선 중 저가~오늘 고가 사이: "
              + ", ".join(f"{z:.2f}" for z in seen[lo_i] if b.Low.iloc[lo_i] < z <= b.High.iloc[:lo_i + 1].max()))
            # 마감 뒤 그린 차트처럼 하루 전체 피벗으로 세면(사후)
            k = max(1, w // minutes)
            allz = zones([v for _, v in pivots(prev, k)] + [v for _, v in pivots(b, k)])
            post = [z for z in allz if b.Low.iloc[lo_i] < z <= b.High.max()]
            p(f"       마감 뒤 하루 전체 피벗으로 그린 선(저가~고가 사이) {len(post)}개: " + ", ".join(f"{z:.2f}" for z in post))
            for n in N_GRID:
                ok = (end >= 585) & (end <= 930) & (b.Close.to_numpy() / c0 - 1 <= CRASH) & (counts >= n)
                i0 = first(ok)
                if i0 is None:
                    p(f"     {n}개: 신호 없음")
                    continue
                px = float(b.Close.iloc[i0])
                run_low = float(b.Low.iloc[:i0 + 1].min())
                hits = []
                for tp in TPS:
                    after = b.iloc[i0 + 1:]
                    h = after[after.High >= px * (1 + tp)]
                    hits.append(f"+{tp:.0%} " + ((h.index[0] + pd.Timedelta(minutes=minutes)).strftime("%H:%M") if len(h) else "못 닿음"))
                r_aw, why_aw = outcome(b, minutes, i0, px, run_low, np.nan, 0.02, "awake")
                p(f"     {n}개: {end[i0] // 60:02d}:{end[i0] % 60:02d} ET(한국 {(end[i0] // 60 + 13) % 24:02d}:{end[i0] % 60:02d}) "
                  f"매수 {px:.2f}({px / c0 - 1:+.1%}) · 그 뒤 최저 {b.Low.iloc[i0 + 1:].min() / px - 1:+.1%} · " + " · ".join(hits)
                  + f" · +2%·11시 손절이면 {r_aw:+.1%}({why_aw}, 다음 날 시가 대신 종가)")
    txt = o.getvalue()
    open(os.path.join(RES, "supports_asts.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    {"run": run, "report": report_only, "asts": asts}[sys.argv[1]]()
