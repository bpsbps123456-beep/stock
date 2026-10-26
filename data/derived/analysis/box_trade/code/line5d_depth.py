"""'최근 N일 저가 x (1-d)' 지정가선의 깊이 d를 올해(2026) 트레이딩뷰 AI 목록 38종목으로 본다(2026-10-09 사용자: -2%면 충분한가).
분봉은 60일 한도라 올해 전체는 yfinance 1시간봉(프리·애프터 포함)으로 계산한다.
규칙(intra_B 실행판 TP2W와 같게):
  L = 오늘을 뺀 직전 N거래일 정규장 저가 최소(일봉), 매수선 P = L x (1-d), 정규장(9:30~16:00 ET) 1시간봉에서 처음 저가 <= P인 봉에 min(시가, P) 체결, 종목당 하루 1회.
  매도: 체결 봉 시작이 11:00 ET 전이면 다음 봉부터 체결가 x 1.02 지정가(시가가 위면 시가), 못 닿으면(또는 11:00 뒤 체결) 매수 뒤 처음 오는 한국시간 08:00 이후 첫 봉 시가.
  1시간봉이라 체결 봉 안의 +2%는 보지 않는다(보수적).
비교 기준: 같은 날 정규장 시가에 사서 같은 매도 규칙(무조건 매수). 초과 = 규칙 거래 - 같은 종목·같은 날 시가 매수.
사용: py line5d_depth.py <일봉 폴더> <1시간봉 저장 폴더>"""
import os, sys
import numpy as np, pandas as pd, yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dip_reason/code"))
from fetch import TV_AI

DEPTHS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
NS = [3, 5, 10, 20]


def hourly(out):
    p = os.path.join(out, "h1.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    d = yf.download(TV_AI, start="2025-12-01", interval="1h", prepost=True, progress=False, auto_adjust=False, group_by="ticker", threads=True)
    rows = []
    for s in TV_AI:
        try:
            x = d[s][["Open", "High", "Low", "Close"]].dropna(subset=["Close"]).copy()
        except KeyError:
            continue
        x.index = x.index.tz_convert("America/New_York")
        x["sym"] = s
        rows.append(x)
    h = pd.concat(rows)
    h.to_parquet(p)
    return h


def run(h, daily_dir):
    out = []
    for s, x in h.groupby("sym"):
        dpath = os.path.join(daily_dir, f"1d_{s}.parquet")
        if not os.path.exists(dpath):
            continue
        dl = pd.read_parquet(dpath).Low
        dl.index = pd.to_datetime(dl.index).date
        x = x.sort_index()
        t = x.index
        o, hi, lo = x.Open.values, x.High.values, x.Low.values
        mod = t.hour * 60 + t.minute
        reg = (mod >= 570) & (mod < 960)
        day = np.asarray(t.date)
        kst = t.tz_convert("Asia/Seoul")
        tk = t.as_unit("ns").asi8
        t8 = kst.normalize() + pd.Timedelta(hours=8)
        dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)

        def exit_from(i, ent):
            """i봉 체결, 체결가 ent → (매도가, 이유)."""
            tp = ent * 1.02 if mod[i] < 660 else None
            for k in range(i + 1, len(o)):
                if tk[k] >= dead[i]:
                    return o[k], "8시"
                if tp is not None and hi[k] >= tp:
                    return max(o[k], tp), "+2%"
            return np.nan, "미완"

        days = sorted(set(day[reg]))
        dli = pd.Series(dl)
        for D in days:
            if D.year != 2026:
                continue
            idx = np.where((day == D) & reg)[0]
            if not len(idx):
                continue
            prev = dli[[k for k in dli.index if k < D]]
            if len(prev) < max(NS):
                continue
            i0 = idx[0]
            b_px, _ = exit_from(i0, o[i0])                 # 같은 날 시가 매수 기준
            base = b_px / o[i0] - 1
            for N in NS:
                L = prev.iloc[-N:].min()
                for dd in DEPTHS:
                    P = L * (1 - dd)
                    hit = [i for i in idx if lo[i] <= P]
                    if not hit:
                        continue
                    i = hit[0]
                    ent = min(o[i], P)
                    px, why = exit_from(i, ent)
                    out.append((s, D, N, dd, ent, px / ent - 1, base, why, o[i] < P))
    return pd.DataFrame(out, columns=["sym", "day", "N", "d", "ent", "ret", "base", "why", "gap"])


if __name__ == "__main__":
    daily_dir, hdir = sys.argv[1], sys.argv[2]
    os.makedirs(hdir, exist_ok=True)
    h = hourly(hdir)
    t = run(h, daily_dir).dropna(subset=["ret"])
    t["excess"] = t.ret - t.base
    t["반기"] = np.where(pd.to_datetime(t.day) < pd.Timestamp("2026-07-01"), "1~6월", "7~10월")
    t["월"] = pd.to_datetime(t.day).dt.month
    t.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/line5d_depth_trades.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)

    def summ(v):
        g = v.groupby("day").excess.sum()
        te = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std(ddof=1) > 0 else np.nan
        return pd.Series(dict(건수=len(v), 평균=v.ret.mean(), 승률=(v.ret > 0).mean(), 최악=v.ret.min(), 합=v.ret.sum(),
                              초과=v.excess.mean(), 날짜t=te, 갭체결=v.gap.mean()))

    for N in NS:
        r = t[t.N == N].groupby(["반기", "d"]).apply(summ).reset_index()
        r["d"] = r.d.map(lambda x: f"-{x:.0%}")
        for c, f in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("합", "{:+.2f}"), ("초과", "{:+.2%}"), ("날짜t", "{:+.2f}"), ("갭체결", "{:.0%}")):
            r[c] = r[c].map(lambda v: "" if pd.isna(v) else f.format(v))
        print(f"\n===== 최근 {N}일 저가 기준, 트레이딩뷰 AI 38종목, 1시간봉 (초과 = 같은 날 시가 매수 대비)")
        print(r.to_string(index=False))
    m = t[t.N == 5].pivot_table(index="월", columns="d", values="excess", aggfunc="mean").map(lambda v: f"{v:+.2%}")
    m.columns = [f"-{c:.0%}" for c in m.columns]
    print("\n월별 초과(5일 기준, 깊이별)\n" + m.to_string())
    c = t[t.N == 5].pivot_table(index="월", columns="d", values="ret", aggfunc="count").fillna(0).astype(int)
    c.columns = [f"-{x:.0%}" for x in c.columns]
    print("\n월별 건수(5일 기준)\n" + c.to_string())
