"""c2 audit 2: events.py 미리보기 감사(실제 events.build를 자른·더럽힌 자료로 다시 돌린다).

두 방식 × 날짜 T 16개 × 판정 2개.
- cut  : 일봉·VIX·실적표를 T에서 자른 사본. T까지 사건 중 '다음 날 시가'가 있는 것(종목별 마지막 날 앞)만 생기므로
         전체 사건표의 같은 사건과 단서 18개가 같아야 한다.
- noise: T 뒤 일봉(주식·ETF 전부)·VIX를 무작위 값으로 바꾸고, T 뒤 실적일을 ±10일 흔들고 세션을 무작위로 바꾼다.
         날짜 행은 그대로라 T 당일 사건까지 생긴다. T 이하 사건의 단서가 하나라도 바뀌면 미래 자료를 본 것이다.
결과(y1·y10·y20)는 끝나는 시가가 T 이하인 사건만 같아야 한다(그 밖은 바뀌는 게 정상).

py code/c2audit_lookahead.py → results/c2audit_lookahead.csv, results/c2audit_lookahead_focus.csv
"""
from __future__ import annotations

import pathlib
import sys
from datetime import date, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

FOCUS = ["MU", "BE", "NBIS", "NVDA", "AMD"]
FEATS = ["M1", "M2", "M3a", "M3b", "M4a", "M4b", "M4c", "M5",
         "S1", "S2", "S3a", "S3b", "S4", "S5", "S6", "S7", "S8", "S9"]
TS = [date(2025, 1, 27), date(2025, 4, 24), date(2025, 12, 17), date(2025, 12, 31),
      date(2026, 3, 6), date(2026, 4, 8), date(2026, 5, 29), date(2026, 6, 1),
      date(2026, 7, 2), date(2026, 7, 28), date(2026, 8, 4), date(2026, 8, 12),
      date(2026, 8, 26), date(2026, 9, 30), date(2026, 10, 14), date(2026, 10, 21)]
# 2025-12-17 MU amc, 2026-07-28 BE amc, 08-04 AMD amc, 08-12 NBIS bmo, 08-26 NVDA amc, 09-30 MU amc: T 당일 저녁 실적
TOL = 1e-12


class PdProxy:
    """events.py 안의 pd를 바꿔 vix·실적 파일만 사본으로 돌려준다."""

    def __init__(self, vix, earn):
        self._vix, self._earn = vix, earn

    def __getattr__(self, k):
        return getattr(pd, k)

    def read_parquet(self, p, *a, **k):
        nm = pathlib.Path(p).name
        if nm == "vix.parquet":
            return self._vix.copy()
        if nm == "earnings.parquet":
            return self._earn.copy()
        return pd.read_parquet(p, *a, **k)


def make_copy(D, vix, earn, T, mode, seed):
    rng = np.random.default_rng(seed)
    if mode == "cut":
        D2 = {s: g[g.date <= T].reset_index(drop=True) for s, g in D.items()}
        D2 = {s: g for s, g in D2.items() if len(g)}
        return D2, vix[vix.index <= T].copy(), earn[earn.date_et <= T].copy()
    D2 = {}
    for s, g in D.items():
        g = g.copy()
        fut = (g.date > T).values
        m = int(fut.sum())
        if m:
            last = float(g.loc[~fut, "C"].iloc[-1]) if (~fut).any() else float(g.C.iloc[0])
            c = last * np.exp(np.cumsum(rng.normal(0, 0.05, m)))
            o = np.r_[last, c[:-1]] * np.exp(rng.normal(0, 0.03, m))
            h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.02, m)))
            lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.02, m)))
            v = float(g.V.median()) * np.exp(rng.normal(0, 1, m))
            g.loc[fut, ["O", "H", "L", "C", "V"]] = np.c_[o, h, lo, c, v]
        D2[s] = g
    v2 = vix.copy()
    fut = v2.index > T
    v2.loc[fut, "VIX"] = rng.uniform(10, 60, int(fut.sum()))
    v2.loc[fut, "VIX3M"] = rng.uniform(10, 60, int(fut.sum()))
    e2 = earn.copy()
    fut = (e2.date_et > T).values
    sh = rng.integers(-10, 11, int(fut.sum()))
    nd = [max(d + timedelta(days=int(k)), T + timedelta(days=1)) for d, k in zip(e2.date_et[fut], sh)]
    e2.loc[fut, "date_et"] = nd
    e2.loc[fut, "session"] = rng.choice(["amc", "bmo", "day"], int(fut.sum()))
    return D2, v2, e2


def run_one(T):
    import events as E
    import swing as S
    D = S.load()
    vix = pd.read_parquet(ROOT / "cache" / "vix.parquet")
    earn = pd.read_parquet(ROOT / "cache" / "earnings.parquet")
    rows, focus = [], []
    for mode in ("cut", "noise"):
        D2, v2, e2 = make_copy(D, vix, earn, T, mode, seed=T.toordinal() * 2 + (mode == "noise"))
        E.load = lambda D2=D2: {s: g.copy() for s, g in D2.items()}
        E.pd = PdProxy(v2, e2)
        for det in ("ZZA3", "ZZP10"):
            full = pd.read_parquet(ROOT / "results" / f"events_{det}.parquet")
            got = E.build(det)
            # 기대 사건: cut은 종목별 마지막 날(T 이하) 앞, noise는 T 이하 전부
            lastrow = {s: len(g) - 1 for s, g in D2.items()}
            exp = full[full.date <= T]
            if mode == "cut":
                exp = exp[exp.conf_idx < exp.sym.map(lastrow)]
            got = got[got.date <= T]
            key = ["sym", "date", "type"]
            m = exp.merge(got, on=key, how="outer", suffixes=("_f", "_t"), indicator=True)
            both = m[m._merge == "both"]
            fd = {}
            for f in FEATS:
                a, b = both[f + "_f"].astype(float).values, both[f + "_t"].astype(float).values
                d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.where(np.isnan(a) != np.isnan(b), np.inf, np.abs(a - b)))
                fd[f] = d
            bad = np.zeros(len(both), bool)
            for f in FEATS:
                bad |= fd[f] > TOL
            # 결과: 끝나는 시가가 T 이하인 사건만 같아야 한다
            dates_of = {s: list(g.date) for s, g in D.items()}

            def ends_by(r, h):
                ds = dates_of[r.sym]
                j = r.conf_idx_f + 1 + h
                return j < len(ds) and ds[j] <= T and (mode != "cut" or j <= lastrow[r.sym])

            yb = {}
            for y, h in (("y10", 10), ("y20", 20)):
                ok = both.apply(lambda r: ends_by(r, h), axis=1).values if len(both) else np.zeros(0, bool)
                yb[y] = (int(ok.sum()), int((ok & (np.abs(both[y + "_f"].values - both[y + "_t"].values) > TOL)).sum()))
            ok1 = np.zeros(len(both), bool)
            for i, (_, r) in enumerate(both.iterrows()):
                if r.exit_date_f is None or pd.isna(r.exit_date_f) or bool(r.censored_f):
                    continue
                ds = dates_of[r.sym]
                j = ds.index(r.exit_date_f) + 1
                ok1[i] = ds[j] <= T and (mode != "cut" or j <= lastrow[r.sym])
            y1bad = int((ok1 & (np.abs(both.y1_f.values - both.y1_t.values) > TOL)).sum()) if len(both) else 0
            rows.append(dict(T=T, mode=mode, det=det, n_expected=len(exp), n_got=len(got),
                             only_full=int((m._merge == "left_only").sum()), only_cut=int((m._merge == "right_only").sum()),
                             n_on_T=int((both.date == T).sum()), feat_bad=int(bad.sum()),
                             max_feat_diff=float(max(fd[f].max() if len(fd[f]) else 0 for f in FEATS)),
                             worst_feat=max(FEATS, key=lambda f: fd[f].max() if len(fd[f]) else 0),
                             y1_checked=int(ok1.sum()), y1_bad=y1bad,
                             y10_checked=yb["y10"][0], y10_bad=yb["y10"][1], y20_checked=yb["y20"][0], y20_bad=yb["y20"][1],
                             S6_on=int(both.S6_f.sum())))
            fb = both[both.sym.isin(FOCUS)]
            for s in FOCUS:
                sel = (both.sym == s).values
                focus.append(dict(T=T, mode=mode, det=det, sym=s, n=int(sel.sum()), feat_bad=int((bad & sel).sum()),
                                  n_on_T=int(((both.date == T).values & sel).sum())))
            del fb
    return rows, focus


if __name__ == "__main__":
    with Pool(3) as pool:
        res = pool.map(run_one, TS)
    R = pd.DataFrame([r for a, _ in res for r in a])
    F = pd.DataFrame([r for _, b in res for r in b])
    R.to_csv(ROOT / "results" / "c2audit_lookahead.csv", index=False)
    F.to_csv(ROOT / "results" / "c2audit_lookahead_focus.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.to_string())
    print("합계", R[["n_expected", "n_got", "only_full", "only_cut", "n_on_T", "feat_bad", "y1_checked", "y1_bad",
                   "y10_checked", "y10_bad", "y20_checked", "y20_bad"]].sum().to_dict())
    print("5종목", F.groupby(["mode", "det"])[["n", "feat_bad", "n_on_T"]].sum().to_string())
