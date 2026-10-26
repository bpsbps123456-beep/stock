"""가격 전 점검표 #9: 가격만 쓰는 셈(포럼과 잇지 않는다). HYPOTHESES.md §5.1 #9, §3 H24~H26·H29·H31·H33·H36.

지난 종가만 쓴다: ^KS11(KOSPI)·000660.KS(SK하이닉스)·^SOX의 종가, 2026-10-14(마지막 창 10/15의 P일)까지.
앞 수익(R_h)·낙폭(MDD_h)·갭·시가·고가·저가·거래대금은 받지도 만들지도 않는다. 모든 값은 그 날까지의 종가로만 정해진다
(σ60은 그날 앞 60개 일수익, 60·20일 최고는 그날까지, r20은 그날 종가 ÷ 20거래일 전 종가).
§12.4 급락 시작은 확인일(5일 고점 대비 −8%) 기준으로 셈하므로 10/14까지 확인된 사건만 든다.

산출: results/prechecks_price.json (셈과 사건 날짜 목록만. 가격 열은 저장하지 않는다)
실행: py prechecks_price.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
LAST_DAY = date(2026, 10, 14)        # 마지막 창 K/U2026-10-15의 직전 거래일. 이 뒤 종가는 쓰지 않는다
START = date(2024, 8, 1)             # 2025-01 σ60·60일 최고 예열
PERIODS = {
    "2025": (date(2025, 1, 1), date(2025, 12, 31)),
    "2026앞": (date(2026, 1, 2), date(2026, 5, 29)),
    "2026뒤(10/14까지)": (date(2026, 6, 1), LAST_DAY),
}
CRASH, HORIZON_EVT, GAP = -0.08, 5, 10   # §12.4(leverage_flood stage2_eval.events와 같은 정의)


def past_closes(ticker: str) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=START.isoformat(), end=(LAST_DAY + timedelta(days=1)).isoformat(),
                                  auto_adjust=False)
    s = pd.Series(h["Close"].to_numpy(float), index=[pd.Timestamp(t).date() for t in h.index])
    s = s[~s.index.duplicated(keep="last")].dropna()
    s = s[s.index <= LAST_DAY]
    if len(s) and max(s.index) > LAST_DAY:
        raise RuntimeError("방화벽: 10/14 뒤 종가가 들어왔다")
    return s


def shock_z(close: pd.Series) -> pd.Series:
    """r(d) ÷ σ60(d), σ60 = d 앞 60개 일수익 SD(d 제외)."""
    r = close.pct_change()
    return r / r.shift(1).rolling(60, min_periods=60).std(ddof=1)


def cluster_events(days: list[date], cal: list[date], gap: int = 5) -> list[date]:
    """사건일 목록 → 묶음 첫날. 앞 사건(묶음의 마지막 사건) 뒤 gap 거래일 안이면 같은 묶음(이어지면 계속)."""
    pos = {d: i for i, d in enumerate(cal)}
    out, last = [], None
    for d in sorted(days):
        i = pos[d]
        if last is None or i - last > gap:
            out.append(d)
        last = i
    return out


def crash_starts(close: pd.Series) -> list[date]:
    """§12.4: 종가가 직전 5거래일 최고 종가보다 8% 이상 낮아진 날(확인일) → 그 5일 고점 다음 날이 시작. GAP 안이면 같은 사건."""
    c = close.dropna()
    v = c.to_numpy()
    starts, last = [], -10**9
    for r in range(HORIZON_EVT, len(v)):
        w = v[r - HORIZON_EVT:r]
        if v[r] / w.max() - 1 <= CRASH:
            a = r - HORIZON_EVT + int(w.argmax()) + 1
            if a - last > GAP:
                starts.append(c.index[a])
                last = a
    return starts


def in_period(d: date, p) -> bool:
    return p[0] <= d <= p[1]


def runs(flags: pd.Series) -> list[tuple[date, date, int]]:
    """True 연속 구간(시작, 끝, 길이)."""
    out, start, n, prev = [], None, 0, None
    for d, f in flags.items():
        if f:
            if start is None:
                start, n = d, 0
            n += 1
            prev = d
        elif start is not None:
            out.append((start, prev, n))
            start = None
    if start is not None:
        out.append((start, prev, n))
    return out


def main() -> int:
    k = past_closes("^KS11")
    hx = past_closes("000660.KS")
    sox = past_closes("^SOX")
    res: dict = {"last_close_day": LAST_DAY.isoformat(), "start": START.isoformat(),
                 "tickers": {"KOSPI": "^KS11", "HYNIX": "000660.KS", "SOX": "^SOX"},
                 "n_closes": {"KOSPI": int(len(k)), "HYNIX": int(len(hx)), "SOX": int(len(sox))},
                 "last_dates": {"KOSPI": str(max(k.index)), "HYNIX": str(max(hx.index)), "SOX": str(max(sox.index))}}
    kcal = list(k.index)

    # H24 사건: KOSPI r/σ60 ≤ −2(하닉 같은 날이면 하나), 5일 안 사건은 묶음
    zk, zh, zs = shock_z(k), shock_z(hx).reindex(k.index), shock_z(sox)
    ev_k = [d for d, z in zk.items() if z <= -2]
    ev_h = [d for d, z in zh.items() if z == z and z <= -2]
    ev_u = sorted(set(ev_k) | set(ev_h))
    cl_union = cluster_events(ev_u, kcal)
    cl_kospi = cluster_events(ev_k, kcal)
    cl_sox = cluster_events([d for d, z in zs.items() if z <= -2], list(sox.index))
    h24 = {}
    for name, p in PERIODS.items():
        h24[name] = {
            "kospi_or_hynix_clusters": [str(d) for d in cl_union if in_period(d, p)],
            "kospi_only_clusters": [str(d) for d in cl_kospi if in_period(d, p)],
            "sox_clusters": [str(d) for d in cl_sox if in_period(d, p)],
            "kospi_days_le_-2sigma": int(sum(in_period(d, p) for d in ev_k)),
        }
    res["h24_events"] = h24

    # 하락 상태 날(D 기준, D−1 종가로 판정): H26 ×0.90·60일 최고, F5 보조 ×0.92·20일 최고
    cal_next = {kcal[i]: kcal[i + 1] for i in range(len(kcal) - 1)}
    cal_next[kcal[-1]] = date(2026, 10, 15)   # 10/14 종가로 판정한 D = 10/15(마지막 창)
    st90 = k <= k.rolling(60, min_periods=60).max() * 0.90
    st92 = k <= k.rolling(20, min_periods=20).max() * 0.92
    r20 = k / k.shift(20) - 1
    down = {}
    for name, p in PERIODS.items():
        d90 = [cal_next[d] for d, f in st90.items() if f and in_period(cal_next[d], p)]
        d92 = [cal_next[d] for d, f in st92.items() if f and in_period(cal_next[d], p)]
        down[name] = {"down_state_60max_x0.90_days": len(d90), "down_state_20max_x0.92_days": len(d92)}
    res["down_state"] = down

    # H36 하락 조건 국면: D−1 종가 기준 r20 ≤ 0인 날의 연속 구간(국면) 수
    fall = pd.Series({cal_next[d]: bool(v <= 0) for d, v in r20.items() if v == v})
    rise = pd.Series({cal_next[d]: bool(v > 0) for d, v in r20.items() if v == v})
    regimes = {}
    for name, p in PERIODS.items():
        f = fall[[in_period(d, p) for d in fall.index]]
        r_ = rise[[in_period(d, p) for d in rise.index]]
        fr, rr = runs(f), runs(r_)
        regimes[name] = {"fall_regimes": len(fr), "fall_regimes_ge5d": sum(1 for x in fr if x[2] >= 5),
                         "fall_days": int(f.sum()), "rise_regimes": len(rr), "rise_days": int(r_.sum()),
                         "fall_list": [[str(a), str(b), n] for a, b, n in fr]}
    res["h36_regimes"] = regimes

    # H29 ±3% 날(r(D−1) 기준 → 세션일로 셈)
    rk = k.pct_change()
    pm3 = {}
    for name, p in PERIODS.items():
        up = [str(d) for d, v in rk.items() if v >= 0.03 and in_period(d, p)]
        dn = [str(d) for d, v in rk.items() if v <= -0.03 and in_period(d, p)]
        pm3[name] = {"up3": len(up), "down3": len(dn), "up_list": up, "down_list": dn}
    pm3["after_2026-10-14"] = {"up3": 0, "down3": 0, "note": "10/14 뒤 종가는 쓰지 않았다(새 날 셈은 판정 시점에)"}
    res["h29_pm3"] = pm3

    # §12.4 급락 시작
    cs = {}
    for nm, s in (("KOSPI", k), ("SOX", sox)):
        st = crash_starts(s)
        cs[nm] = {name: [str(d) for d in st if in_period(d, p)] for name, p in PERIODS.items()}
    cs["doc_12_4_2026"] = {"KOSPI": ["2026-02-27", "2026-03-19", "2026-05-15", "2026-06-04", "2026-06-23",
                                     "2026-07-08", "2026-07-24", "2026-10-06"],
                           "SOX": ["2026-01-30", "2026-02-26", "2026-03-26", "2026-06-04", "2026-06-23",
                                   "2026-07-13", "2026-08-18"]}
    res["crash_starts_12_4"] = cs

    out = RESULTS / "prechecks_price.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("h36_regimes",)}, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
