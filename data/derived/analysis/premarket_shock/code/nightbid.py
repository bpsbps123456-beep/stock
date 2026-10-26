"""사용자 아이디어(2026-10-10): 자기 전(11:00 ET = 서머타임 24:00 KST)에 그때까지 저가보다 더 아래에 매수 지정가를 걸고 잔다.

대상: 11:00 ET 봉 종가가 전일 종가 대비 -7% 이하인 급락일(lowfind 사건, 그 순간 급락 목록에 있는 종목).
매수: 그때까지 저가의 -k%(k 0·2·3·5·8%). 체결은 그 값보다 0.1% 더 내려간 봉, 봉 시가가 이미 아래면 시가(lowfind.night과 같다).
      비교: 11:00 종가에 바로 산 경우.
매도(자는 동안에는 체결을 모르므로 익절·손절 주문은 아침에 건다):
  아침      다음 날 아침 주간거래에서 판다. 장후·주간거래 시세가 없어 당일 종가로 대신 잰다.
  다음날시가
  다음날+2%·+3%   아침에 지정가를 걸고 다음 날 정규장에서 닿으면 판다. 시가가 이미 위면 시가, 못 닿으면 다음 날 종가.
  다음날+2%/-3%   위 +2%에 체결가 -3% 손절을 더한다. 같은 봉이면 손절 먼저, 시가가 이미 아래면 시가.
사용: py nightbid.py run → results/nightbid.txt"""
import io, os, sys
import numpy as np, pandas as pd
from lowfind import RES, CRASH, AWAKE_END, regular_bars

KS = (0.0, 0.02, 0.03, 0.05, 0.08)
STRICT = 0.001
EXITS = ("아침", "다음날시가", "다음날+2%", "다음날+3%", "다음날+2%/-3%")


def next_day_exit(nb, px, tp, stop=None):
    """다음 날 정규장 봉 nb에서 +tp 지정가(와 손절)로 판 수익률. 자료가 없으면 nan."""
    if nb is None or nb.empty:
        return np.nan
    target, floor = px * (1 + tp), (px * (1 - stop) if stop else None)
    o = float(nb.Open.iloc[0])
    if floor is not None and o <= floor:
        return o / px - 1
    if o >= target:
        return o / px - 1
    for lo, hi, op in zip(nb.Low.to_numpy(), nb.High.to_numpy(), nb.Open.to_numpy()):
        if floor is not None and lo <= floor:
            return min(floor, op) / px - 1
        if hi >= target:
            return tp
    return float(nb.Close.iloc[-1]) / px - 1


def exits(px, close, nxt_open, nb):
    return {"아침": close / px - 1, "다음날시가": (nxt_open / px - 1) if nxt_open == nxt_open else np.nan,
            "다음날+2%": next_day_exit(nb, px, 0.02), "다음날+3%": next_day_exit(nb, px, 0.03),
            "다음날+2%/-3%": next_day_exit(nb, px, 0.02, 0.03)}


def night_rows(b, c0, nxt_open, nb, minutes, key):
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    idx = np.nonzero(end == AWAKE_END)[0]
    if not len(idx):
        return []
    i11 = int(idx[0])
    cl = b.Close.to_numpy()
    dd11 = cl[i11] / c0 - 1
    if dd11 > CRASH:
        return []
    rl = float(b.Low.iloc[:i11 + 1].min())
    later = b.iloc[i11 + 1:]
    close = float(cl[-1])
    rows = [dict(**key, entry="11시 종가", filled=True, fill_at="11:00", dd11=dd11, px_vs_low=cl[i11] / rl - 1,
                 **exits(float(cl[i11]), close, nxt_open, nb))]
    for k in KS:
        level = rl * (1 - k)
        hit = later[later.Low <= level * (1 - STRICT)]
        name = f"저가-{k:.0%}"
        if hit.empty:
            rows.append(dict(**key, entry=name, filled=False, dd11=dd11))
            continue
        px = min(level, float(hit.Open.iloc[0]))
        at = hit.index[0] + pd.Timedelta(minutes=minutes)
        rows.append(dict(**key, entry=name, filled=True, fill_at=at.strftime("%H:%M"), dd11=dd11,
                         **exits(px, close, nxt_open, nb)))
    return rows


def run():
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    rows = []
    for t, d, win in zip(ev.ticker, ev.day, ev.win):
        for minutes in ((5, 3) if win == "뒤" else (5,)):
            got = regular_bars(t, d, minutes, with_next=True)
            if got is None:
                continue
            b, c0, nxt_open, nb = got
            rows += night_rows(b, c0, nxt_open, nb, minutes, dict(ticker=t, day=d, win=win, bar=minutes))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "nightbid.csv"), index=False)
    report(df)


def cell(x, col):
    r = x[col].dropna()
    if len(r) < 5:
        return f"(체결 {len(r)}건)"
    g = x.assign(ret=x[col]).dropna(subset=["ret"]).groupby("day").ret.mean()
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan
    return (f"평균 {r.mean():+.2%} 중앙 {r.median():+.1%} 승률 {(r > 0).mean():.0%} 하위10% {r.quantile(.1):+.1%} "
            f"최악 {r.min():+.1%} n={len(r)} t(일) {t:+.1f}")


def report(df):
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("[자기 전 지정가 — 11:00 ET에 -7% 이하인 종목, 그때까지 저가의 -k%에 매수 주문]")
    for (win, bar), g in df.groupby(["win", "bar"], sort=False):
        targets = g[g.entry == "11시 종가"]
        p(f"\n==== {win} {bar}분봉 · 대상 {len(targets)}종목·날 ====")
        for entry in ["11시 종가"] + [f"저가-{k:.0%}" for k in KS]:
            x = g[g.entry == entry]
            f = x[x.filled == True]
            when = f["fill_at"].dropna()
            p(f"  {entry:<8} 체결 {len(f) / max(len(x), 1):.0%}({len(f)})"
              + (f" · 체결 시각 중앙 {sorted(when)[len(when) // 2]}" if len(when) else ""))
            for col in EXITS:
                p(f"      {col:<12} {cell(f, col)}")
    p(f"\n시험 범위: 매수 깊이 {len(KS)}개(+11시 종가 비교) × 매도 {len(EXITS)}개 × 기간·봉 3칸. 깊이와 매도는 결과 보기 전에 정했다.")
    txt = o.getvalue()
    open(os.path.join(RES, "nightbid.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def report_only():
    report(pd.read_csv(os.path.join(RES, "nightbid.csv")))


if __name__ == "__main__":
    {"run": run, "report": report_only}[sys.argv[1]]()
