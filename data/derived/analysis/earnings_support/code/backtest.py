"""본 백테스트: 실적 발표 전(기준일 정규장 마감) 160분봉 지지 구역 Zk에 매수 지정가 → 반응일 체결 → 매도.

실행: py data/derived/analysis/earnings_support/code/backtest.py
출력: cache/trades.parquet(사건 × 전략 × 창), results/summary_all.csv(전 칸), results/null_n2.csv(N2 귀무 집계),
      results/main_table.csv(주 설정 표)
전략: Z1~Z6 하단·상단, N0 기준 종가 매수, N1 고정 −2/−4/−6/−8/−10/−15%, N3 −m·s(m 0.5·1·1.5·2), N2 지터 귀무(하단, 50번).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from events import load  # noqa: E402
from lib import CACHE, RES  # noqa: E402
from sim import EXITS, fills, returns, summarize  # noqa: E402

K = 6
N1 = [2, 4, 6, 8, 10, 15]
N3 = [0.5, 1.0, 1.5, 2.0]
NULL_DRAWS = 50
WINDOWS = ("A", "B", "C")


def jitter(eid: int) -> np.ndarray:
    rng = np.random.default_rng(1000003 * (eid + 1))
    out = []
    while len(out) < NULL_DRAWS:
        u = rng.uniform(-0.015, 0.015, NULL_DRAWS * 2)
        out.extend(u[np.abs(u) >= 0.003].tolist())
    return np.array(out[:NULL_DRAWS])


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta, bars, zs = load()
    sup = zs[zs.side == "S"].set_index(["eid", "k"])[["lo", "hi"]]
    sup_by = {e: g.droplevel(0) for e, g in sup.groupby(level=0)}
    trades, nulls = [], []
    t0 = time.time()
    for n, r in enumerate(meta.itertuples()):
        b = bars.get(r.eid)
        if b is None:
            continue
        arr = {k: b[k].to_numpy() for k in ("seg", "o", "h", "l", "c", "nl")}
        names, levels = [], []
        z = sup_by.get(r.eid)
        for k in range(1, K + 1):
            if z is not None and k in z.index:
                names += [f"Z{k}_bot", f"Z{k}_top"]
                levels += [z.at[k, "lo"], z.at[k, "hi"]]
        for d in N1:
            names.append(f"N1_{d}")
            levels.append(r.base_close * (1 - d / 100))
        if r.s is not None and np.isfinite(r.s):
            for m in N3:
                names.append(f"N3_{m}")
                levels.append(r.base_close * (1 - m * r.s / 100))
        levels = np.array(levels, float)
        # N2: 하단 구역마다 50번
        null_names, null_lv = [], []
        if z is not None:
            u = jitter(int(r.eid))
            for k in range(1, K + 1):
                if k in z.index:
                    null_names += [(k, j) for j in range(NULL_DRAWS)]
                    null_lv.append(z.at[k, "lo"] * (1 + u))
        null_lv = np.concatenate(null_lv) if null_lv else np.empty(0)
        for w in WINDOWS:
            pos, px = fills(arr, levels, w, r.amc)
            rets = returns(arr, r, pos, px)
            for i, nm in enumerate(names):
                trades.append((r.eid, nm, w, pos[i] >= 0, px[i], *(rets[x][i] for x in EXITS)))
            if len(null_lv):
                pos, px = fills(arr, null_lv, w, r.amc)
                rets = returns(arr, r, pos, px)
                for i, (k, j) in enumerate(null_names):
                    nulls.append((r.eid, k, j, w, pos[i] >= 0, *(rets[x][i] for x in ("x1", "x2", "x4"))))
        # N0: 기준 종가 매수(체결 봉 위치 −1 → 반응일 정규장 전체에서 익절 확인)
        rets = returns(arr, r, np.array([-1]), np.array([r.base_close]))
        trades.append((r.eid, "N0", "-", True, r.base_close, *(rets[x][0] for x in EXITS)))
        if n % 1000 == 0:
            print(f"{n}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    cols = ["eid", "strategy", "window", "filled", "fill_px", *EXITS]
    tr = pd.DataFrame(trades, columns=cols)
    nl = pd.DataFrame(nulls, columns=["eid", "k", "draw", "window", "filled", "x1", "x2", "x4"])
    ok = meta[["eid"]].assign(**{f"ok_{x}": meta[x].notna() for x in ("x1", "x2", "x3", "x4")})
    ok["ok_x5"] = ok["ok_x4"]
    tr = tr.merge(meta[["eid", "ticker", "reaction_date", "period", "amc", "mcap_b", "ai", "swings", "s"]], on="eid")
    tr = tr.merge(ok, on="eid")
    nl = nl.merge(meta[["eid", "reaction_date", "period", "amc", "mcap_b", "ai", "swings"]], on="eid")
    nl = nl.merge(ok[["eid", "ok_x1", "ok_x2", "ok_x4"]], on="eid")
    return tr, nl


SPLITS = {
    "all": lambda d: d,
    "amc": lambda d: d[d.amc],
    "bmo": lambda d: d[~d.amc],
    "cap>=10B": lambda d: d[d.mcap_b >= 10],
    "cap>=50B": lambda d: d[d.mcap_b >= 50],
    "ai": lambda d: d[d.ai],
}
PERIODS = {"2025": lambda d: d[d.period == "2025"], "2026H1": lambda d: d[d.period == "2026H1"],
           "2026H2": lambda d: d[d.period == "2026H2"], "2026": lambda d: d[d.period.str.startswith("2026")]}


def summary(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (st, w), g in tr.groupby(["strategy", "window"]):
        for pn, pf in PERIODS.items():
            gp = pf(g)
            for sn, sf in SPLITS.items():
                gs = sf(gp)
                if len(gs) < 10:
                    continue
                for x in EXITS:
                    rows.append({"strategy": st, "window": w, "period": pn, "split": sn, "exit": x,
                                 **summarize(gs, x)})
    return pd.DataFrame(rows)


def null_summary(nl: pd.DataFrame) -> pd.DataFrame:
    """N2: 뽑기 번호마다 사건 전체를 모은 집계 → (구역, 창, 구간, 나눔, 매도)마다 50개 귀무 값."""
    rows = []
    for pn, pf in PERIODS.items():
        gp = pf(nl)
        for sn, sf in SPLITS.items():
            gs = sf(gp)
            if len(gs) < 10 * NULL_DRAWS:
                continue
            for x in ("x1", "x2", "x4"):
                v = gs[gs[f"ok_{x}"]]
                fin = np.isfinite(v[x].to_numpy())
                v = v[~(v.filled.to_numpy() & ~fin)]
                f = v.filled.to_numpy() & np.isfinite(v[x].to_numpy())
                xv = np.where(f, v[x].to_numpy(), 0.0)
                agg = pd.DataFrame({"k": v.k.to_numpy(), "window": v.window.to_numpy(), "draw": v.draw.to_numpy(),
                                    "n": 1, "nf": f.astype(int), "nw": (f & (xv > 0)).astype(int), "sx": xv})
                g = agg.groupby(["k", "window", "draw"]).sum()
                g = g.assign(fill_rate=g.nf / g.n, win=g.nw / g.nf.replace(0, np.nan), mean=g.sx / g.nf.replace(0, np.nan),
                             per_event=g.sx / g.n, period=pn, split=sn, exit=x).reset_index()
                rows.append(g[["k", "window", "period", "split", "exit", "draw", "n", "nf", "fill_rate", "win", "mean",
                               "per_event"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    """--from-cache면 사건 계산을 건너뛰고 저장된 거래로 집계만 다시 한다.
    주 표본: 판단 시점 160분봉 스윙 60개 이상 + 1시간봉 종가와 일봉 종가가 3% 안(BNY처럼 1시간봉 기호가 다른 상품을
    가리키는 자료 오류 4건을 뺀다)."""
    if "--from-cache" in sys.argv:
        tr, nl = pd.read_parquet(CACHE / "trades.parquet"), pd.read_parquet(CACHE / "null_n2.parquet")
    else:
        tr, nl = run()
        tr.to_parquet(CACHE / "trades.parquet")
        nl.to_parquet(CACHE / "null_n2.parquet")
    meta = pd.read_parquet(CACHE / "ev_meta.parquet")
    good = set(meta.loc[(meta.ref_bar / meta.base_close - 1).abs() <= 0.03, "eid"])
    RES.mkdir(parents=True, exist_ok=True)
    s = summary(tr[(tr.swings >= 60) & tr.eid.isin(good)])
    s.to_csv(RES / "summary_all.csv", index=False)
    ns = null_summary(nl[(nl.swings >= 60) & nl.eid.isin(good)])
    ns.to_csv(RES / "null_n2.csv", index=False)
    print("완료", len(tr), len(nl), "주 표본 사건", len(set(tr.loc[(tr.swings >= 60) & tr.eid.isin(good), "eid"])), flush=True)


if __name__ == "__main__":
    main()
