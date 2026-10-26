"""N2 사건표: 반전 확인 사건에 금리 단서 R1·R2·M을 붙인다(BRIEF.md N2).

py code/n2_events.py → results/n2_events_stock.parquet · results/n2_events_soxx.parquet · results/n2_events_check.json

단서(모두 확정일 d 종가까지 알 수 있는 값, ^TNX는 주식 거래일 달력에 앞 값 채우기)
- R1 = TNX[d] − TNX[d−20거래일] (%p)
- R2 = TNX[이번 극점] − TNX[직전 극점] (%p). 직전 반대 구간(전 극점 → 극점) 동안의 금리 변화. 극점은 확정일 이전 날이다.
- M  = SOXX 5거래일 수익 < 0 그리고 TNX 5거래일 변화 < 0 (Codex analyze.py 정의, 확정일 종가 기준) → 1/0
- 민감도(^TNX 종가 시각 대비): R1L·ML = 금리만 하루 늦춘 값(TNX[d−1] 기준).

사건
- 주식: swing_trend/results/events_ZZA3.parquet(읽기만). 직전 극점은 swing.zigzag(3×ATR20)를 같은 가격으로 다시 돌려 찾고, 원 사건표의 극점·확정일과 전부 맞는지 확인한다.
- SOXX 긴 이력: n2_prices_adj SOXX(수정주가, swing_trend와 같은 auto_adjust=True)에 swing.zigzag(3×ATR20). 결과는 events.py와 같은 식
  (y1 = 반대 확정 다음 날 시가까지, y10·y20 = 고정 10·20거래일 뒤 시가, 확정 다음 날 시가 진입, 매도 확정은 부호 반대).
  구간 L1 2003~2013(경계 고르기), L2 2014~2023(확인), L3 2024~2026-10-23(표시).
  ^SOX는 2008년 전 시가가 대부분 전날 종가와 같아(1997~2004년 85~99%) 시가 체결 결과에 못 쓰므로 SOXX를 쓴다.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
os.environ["NUMBA_CACHE_DIR"] = str(HERE / "cache" / "numba")   # swing_trend 폴더에 캐시를 쓰지 않는다
sys.dont_write_bytecode = True
SWING = HERE.parent / "swing_trend"
sys.path.insert(0, str(SWING / "code"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from swing import atr, load, zigzag  # noqa: E402

RES = HERE / "results"
LPER = {"L1": ("2003-01-01", "2013-12-31"), "L2": ("2014-01-01", "2023-12-31"), "L3": ("2024-01-01", "2026-10-23")}


def rate_frame(cal: pd.DatetimeIndex, raw: pd.DataFrame) -> pd.DataFrame:
    """주식 거래일 달력 위 금리·SOXX 단서(날짜 단위)."""
    tnx = raw[raw.sym == "^TNX"].set_index("date").C.reindex(cal).ffill()
    sx = raw[raw.sym == "SOXX"].set_index("date").C.reindex(cal)
    f = pd.DataFrame(index=cal)
    f["tnx"] = tnx.values
    f["R1"] = (tnx - tnx.shift(20)).values
    f["R1L"] = (tnx.shift(1) - tnx.shift(21)).values
    fall = (sx / sx.shift(5) - 1) < 0
    f["M"] = (fall & ((tnx - tnx.shift(5)) < 0)).astype(float).values
    f["ML"] = (fall & ((tnx.shift(1) - tnx.shift(6)) < 0)).astype(float).values
    # Codex 원식(앞 값 채우기 없이 금리 결측이면 꺼짐)과 다른 날 수 확인용
    tnx0 = raw[raw.sym == "^TNX"].set_index("date").C.reindex(cal)
    f["M_noffill"] = (fall & (tnx0.diff(5) < 0)).astype(float).values
    nanmask = sx.shift(5).isna() | tnx.shift(5).isna()
    for k in ("M", "M_noffill"):
        f.loc[nanmask.values, k] = np.nan
    f.loc[(sx.shift(5).isna() | tnx.shift(6).isna()).values, "ML"] = np.nan
    return f


def stock_events(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ev = pd.read_parquet(SWING / "results" / "events_ZZA3.parquet")
    D = load()
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(D["SOXX"].date)))
    rf = rate_frame(cal, raw)
    tnx = rf.tnx
    out, bad = [], 0
    by = {s: g for s, g in ev.groupby("sym")}
    for s, g0 in by.items():
        g = D[s]
        O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
        a20 = atr(H, L, C, 20)
        st, pi, pc, pt = zigzag(C, 0.0, 3.0 * a20)
        dates = pd.to_datetime(g.date).values
        pos = {int(c): e for e, c in enumerate(pc)}
        for _, r in g0.iterrows():
            e = pos.get(int(r.conf_idx))
            if e is None or pt[e] != r.type or pd.Timestamp(dates[pi[e]]).date() != r.pivot_date:
                bad += 1
                continue
            d = pd.Timestamp(r.date)
            p_d = pd.Timestamp(dates[pi[e]])
            pp_d = pd.Timestamp(dates[pi[e - 1]]) if e > 0 else None
            row = r.to_dict()
            row.update(R1=rf.at[d, "R1"], R1L=rf.at[d, "R1L"], M=rf.at[d, "M"], ML=rf.at[d, "ML"], M_noffill=rf.at[d, "M_noffill"],
                       R2=(tnx.at[p_d] - tnx.at[pp_d]) if pp_d is not None else np.nan,
                       prev_pivot_date=pp_d.date() if pp_d is not None else None)
            out.append(row)
    res = pd.DataFrame(out)
    chk = dict(events_in=len(ev), events_out=len(res), pivot_mismatch=bad,
               M_ffill_vs_noffill_diff_dates=int((rf.M.fillna(-1) != rf.M_noffill.fillna(-1)).sum()))
    return res, chk


def soxx_events(adj: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = adj[adj.sym == "SOXX"].sort_values("date").reset_index(drop=True)
    cal = pd.DatetimeIndex(g.date)
    rf = rate_frame(cal, raw)
    tnx = rf.tnx.values
    O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
    a20 = atr(H, L, C, 20)
    st, pi, pc, pt = zigzag(C, 0.0, 3.0 * a20)
    n = len(C)
    rows = []
    for e, (p, c, t) in enumerate(zip(pi, pc, pt)):
        d = cal[c]
        per = next((k for k, (a, b) in LPER.items() if pd.Timestamp(a) <= d <= pd.Timestamp(b)), None)
        if per is None or c < 260 or c + 1 >= n:
            continue
        nxt = [k for k in range(e + 1, len(pc)) if pt[k] == -t]
        c2 = pc[nxt[0]] if nxt else None
        exit_px = O[c2 + 1] if (c2 is not None and c2 + 1 < n) else C[-1]
        y1 = t * np.log(exit_px / O[c + 1])
        y10 = t * np.log(O[c + 11] / O[c + 1]) if c + 11 < n else np.nan
        y20 = t * np.log(O[c + 21] / O[c + 1]) if c + 21 < n else np.nan
        pp = pi[e - 1] if e > 0 else None
        rows.append(dict(sym="SOXX", type=int(t), date=d.date(), period=per, pivot_date=cal[p].date(), conf_idx=int(c),
                         prev_pivot_date=cal[pp].date() if pp is not None else None,
                         exit_date=cal[c2].date() if c2 is not None else None, censored=c2 is None or c2 + 1 >= n,
                         hold_days=(c2 + 1 - (c + 1)) if c2 is not None else n - 1 - (c + 1),
                         y1=y1, y10=y10, y20=y20, S1=int(c - p),
                         R1=rf.R1.iat[c], R1L=rf.R1L.iat[c], M=rf.M.iat[c], ML=rf.ML.iat[c], M_noffill=rf.M_noffill.iat[c],
                         R2=(tnx[p] - tnx[pp]) if pp is not None else np.nan))
    res = pd.DataFrame(rows)
    # 확인: 극점은 확정일보다 앞서고, 직전 극점은 극점보다 앞서며, 극점 사이 종가 순서가 지그재그와 맞다
    ok_order = bool((pd.to_datetime(res.pivot_date) < pd.to_datetime(res.date)).all()
                    and (pd.to_datetime(res.prev_pivot_date) < pd.to_datetime(res.pivot_date)).all())
    chk = dict(soxx_rows=n, soxx_first=str(cal[0].date()), pivots_total=len(pc), events=len(res), order_ok=ok_order,
               by_period_type=res.groupby(["period", "type"]).size().rename(lambda x: str(x)).to_dict() if len(res) else {})
    chk["by_period_type"] = {f"{k[0]}_{k[1]}": int(v) for k, v in res.groupby(["period", "type"]).size().items()}
    return res, chk


def main():
    raw = pd.read_parquet(HERE / "cache" / "n2_prices_raw.parquet")
    adj = pd.read_parquet(HERE / "cache" / "n2_prices_adj.parquet")
    for x in (raw, adj):
        x["date"] = pd.to_datetime(x.date)
    se, c1 = stock_events(raw)
    xe, c2 = soxx_events(adj, raw)
    se.to_parquet(RES / "n2_events_stock.parquet")
    xe.to_parquet(RES / "n2_events_soxx.parquet")
    chk = dict(stock=c1, soxx=c2)
    for nm, df in (("stock", se), ("soxx", xe)):
        chk[nm]["feature_nan"] = {k: int(df[k].isna().sum()) for k in ("R1", "R2", "M", "R1L", "ML")}
        chk[nm]["M_vs_noffill_event_diff"] = int((df.M.fillna(-1) != df.M_noffill.fillna(-1)).sum())
    (RES / "n2_events_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
    for nm, df in (("stock", se), ("soxx", xe)):
        print(nm, df.groupby(["period", "type"]).agg(n=("y1", "size"), dates=("date", "nunique"), M=("M", "mean"),
                                                       R1=("R1", "median"), R2=("R2", "median"), y1=("y1", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
