"""장세 전환 위험 줄(2026-10-28, 인계 문서 할 일 #3): 긴 이력 ^SOX(1994~)에서 바닥 확인 문턱 7·8·9%를 다시 잰다.

합격 기준이 아니다(AGENTS.md 연구 보고 규칙: 2026 장세가 기준, 다른 해의 실패는 위험 한두 줄). 클라우드 표본(SOXX 2024~2026)에
긴 약세장(2000~02·2007~09·2022)이 없어서 같은 규칙이 그런 장세에서 어땠는지만 본다.

정의는 bc_lib 그대로(outcome_frame 재사용, THR 10%):
  in_dd  = 최근 20거래일 안에 종가가 60일 최고 종가보다 10% 이상 낮았음
  up     = 오늘 종가 / 60일 고점 이후 최저 종가(Ldd) − 1
  hold20 = 다음 20거래일 종가가 Ldd 를 깨지 않음(덜 찬 창은 이미 깨졌을 때만 0)
지표(문턱 k = 7·8·9%, 3개만 시험)
  날 기준     in_dd 이고 up ≥ k 인 날 전부의 hold20 평균(쿨다운 없음, calib.py·threshold_sweep 표와 같은 셈)
  국면 기준   그 날들의 하락 국면(Ldd 값이 같은 날 묶음) 수와, 그중 hold20 = 0 인 날이 하나라도 있는 국면 수
  첫 신호     bc_lib.first_signals(20거래일 쿨다운) hold20 — 보조
자료: yfinance ^SOX 일봉 종가 1994-05-04~(results/long_history_sox_prices.csv 스냅샷, --refresh 로 다시 받음).
대조: 2024~2026(~2026-10-23)은 SOXX 결과(7%·8% 날 기준 99.3%, 실패 국면 1)와 같은지 본다.

py code/long_history.py [--refresh] → results/long_history_summary.csv · long_history_fails.csv · long_history.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import bc_lib as L

KS = [0.07, 0.08, 0.09]
SNAP = L.RES / "long_history_sox_prices.csv"
PERIODS = [("1994~2023", "1994-05-04", "2023-12-29"), ("2000~02", "2000-01-03", "2002-12-31"),
           ("2007~09", "2007-01-03", "2009-12-31"), ("2022", "2022-01-03", "2022-12-30"),
           ("1994~2023 약세장 밖", None, None), ("2024~2026", "2024-01-02", "2026-10-23")]
BEARS = [p for p in PERIODS if p[0] in ("2000~02", "2007~09", "2022")]


def load(refresh: bool) -> pd.DataFrame:
    if refresh or not SNAP.exists():
        import yfinance as yf
        d = yf.download("^SOX", period="max", auto_adjust=False, progress=False)
        out = pd.DataFrame({"O": d["Open"].squeeze(), "C": d["Close"].squeeze()})
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
        out.index.name = "date"
        out.to_csv(SNAP, encoding="utf-8-sig")
    d = pd.read_csv(SNAP, encoding="utf-8-sig", index_col=0, parse_dates=True)
    import exchange_calendars as xc
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS", start="1994-01-01").sessions_in_range(d.index[0], d.index[-1]))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    miss = cal.difference(d.index)
    extra = d.index.difference(cal)
    d = d.reindex(cal)
    return d, list(miss.strftime("%Y-%m-%d")), list(extra.strftime("%Y-%m-%d"))


def mask(idx: pd.DatetimeIndex, name: str, a, b) -> np.ndarray:
    if name == "1994~2023 약세장 밖":
        m = (idx >= "1994-05-04") & (idx <= "2023-12-29")
        for _, x, y in BEARS:
            m &= ~((idx >= x) & (idx <= y))
        return m
    return (idx >= a) & (idx <= b)


def main(refresh: bool = False):
    pd.set_option("display.width", 220)
    d, miss, extra = load(refresh)
    O, C = d[["O"]].rename(columns={"O": "SOX"}), d[["C"]].rename(columns={"C": "SOX"})
    fr = L.outcome_frame(O, C, L.THR["SOXX"])
    up, st, h20, h40, Ldd = (fr[k]["SOX"] for k in ("up", "in_dd", "hold20", "hold40", "L60"))
    L.EVAL_START["SOX"] = pd.Timestamp("1994-05-04")
    rows, fails = [], []
    for k in KS:
        sig = fr["up"] >= k
        fs = L.first_signals(sig, fr, "SOX")
        fs["hold20"] = [h20.iat[p] for p in fs.pos]
        for name, a, b in PERIODS:
            m = mask(up.index, name, a, b) & st.to_numpy() & (up >= k).to_numpy() & Ldd.notna().to_numpy()
            v = h20[m].dropna()
            v40 = h40[m].dropna()
            ep = Ldd[v.index].round(4)
            bad = ep[v == 0].unique()
            g = fs[mask(pd.DatetimeIndex(fs.date), name, a, b)] if len(fs) else fs
            gv = g.hold20.dropna()
            base = h20[mask(up.index, name, a, b) & st.to_numpy() & Ldd.notna().to_numpy()].dropna()
            rows.append(dict(k=k, period=name, days=len(v), hold20_days=v.mean() if len(v) else np.nan,
                             base_hold20=base.mean() if len(base) else np.nan,
                             hold40_days=v40.mean() if len(v40) else np.nan, episodes=int(ep.nunique()),
                             failed_episodes=int(len(bad)), first_n=len(gv), first_hold20=int(gv.sum()),
                             in_dd_days=int((mask(up.index, name, a, b) & st.to_numpy()).sum())))
            if name in ("1994~2023", "2024~2026"):
                for L_ in bad:
                    days = v.index[(ep == L_).to_numpy() & (v == 0).to_numpy()]
                    t0 = up.index.get_loc(days[0])
                    fut = C["SOX"].iloc[t0 + 1:]
                    brk = fut.index[fut < L_ - 1e-9]
                    low = C["SOX"].iloc[t0 + 1:t0 + 61].min()
                    fails.append(dict(k=k, period=name, Ldd=float(L_), first_fail_day=days[0].date().isoformat(),
                                      up_then=float(up.loc[days[0]]), fail_days=len(days),
                                      break_date=brk[0].date().isoformat() if len(brk) else None,
                                      low_next60_vs_Ldd=float(low / L_ - 1)))
    t = pd.DataFrame(rows)
    f = pd.DataFrame(fails)
    t.to_csv(L.RES / "long_history_summary.csv", index=False, encoding="utf-8-sig")
    f.to_csv(L.RES / "long_history_fails.csv", index=False, encoding="utf-8-sig")
    info = dict(source="yfinance ^SOX Close", first=str(d.index[0].date()), last=str(d.index[-1].date()),
                sessions=len(d), missing_sessions=len(miss), missing_sample=miss[:10], non_xnys_rows=extra,
                thresholds=KS, periods=[p[0] for p in PERIODS], thr_in_dd=L.THR["SOXX"])
    (L.RES / "long_history.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    print(t.round(3).to_string(index=False))
    print(f.round(3).to_string(index=False))


if __name__ == "__main__":
    main("--refresh" in sys.argv)
