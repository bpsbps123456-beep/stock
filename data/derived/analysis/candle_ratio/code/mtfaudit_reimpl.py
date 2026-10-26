"""mtfaudit 1: 독립 재계산(mtfaudit_core)으로 mtf_rows·mtf_diag(k=0) 대조 + 묶음 대조(engine.tf_bars) + 1시간봉 30분 봉 확인.

py code/mtfaudit_reimpl.py → results/mtfaudit_reimpl_cells.csv, results/mtfaudit_reimpl_diag.csv, results/mtfaudit_reimpl.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import mtfaudit_core as M  # noqa: E402
from engine import tf_bars  # noqa: E402  (묶음 대조용으로만)

OUT = M.HERE / "results"
SYMS = ["MU", "BE", "NBIS"]
CELLS = [(10, "BASE", "A3"), (40, "BASE", "A3"), (240, "BASE", "A3"),
         (40, "z160_AGREE", "A0"), (40, "z320_ER", "A0"), (60, "z240_ER", "A2"), (20, "zD_ER", "A0"),
         (240, "z960_AGREE", "A2"), (10, "z80_ADX", "A0"), (90, "zD_EXT", "A0"),
         (30, "z30_ADX", "A2"), (120, "z120_ER", "A0"), (180, "z360_EXT", "A2"), (20, "z20_EXT", "A0")]
BOARDS = {"real": (0.001, False), "real0": (0.0, False), "ideal": (0.0, True)}
COLS = ["ret", "mdd", "x", "static", "static_mdd", "bh"]


def bars_check(raw):
    """engine.tf_bars와 내 묶음(5분봉·1시간봉)이 O·H·L·C·V·start·day에서 같은지."""
    bad = 0
    tot = 0
    for kind, base in (("5m", raw["5m"]), ("1h", raw["1h"])):
        for tf in sorted({z for b in M.BASE_TFS for z in M.zone_list(b) if z != "D"}):
            a = M.group(base, tf, kind)
            e = tf_bars(base, tf, "eth")
            tot += 1
            same = (len(a) == len(e) and np.allclose(a[["O", "H", "L", "C", "V"]].values, e[["O", "H", "L", "C", "V"]].values,
                                                     rtol=0, atol=1e-9)
                    and (a.start.values == pd.DatetimeIndex(e.start).as_unit("ns").asi8).all() and (a.day.values == e.day.values).all())
            bad += not same
    return dict(tf_checked=tot, mismatched=bad)


def hour_bar_check(raw):
    """1시간봉 09:00·15:30이 30분 봉인지: 7/17 이후 5분봉 묶음과 O·H·L·C 대조(끝 시각 30분 vs 60분 가정)."""
    h1, d5 = raw["1h"], raw["5m"]
    h1 = h1[h1.index.date >= M.START5]
    res = {}
    for hm in ("09:00", "15:30", "10:30", "16:00"):
        hh, mm = map(int, hm.split(":"))
        rows = h1[(h1.index.hour == hh) & (h1.index.minute == mm)]
        out = {}
        for ln in (30, 60):
            ok = n = 0
            for t, r in rows.iterrows():
                sub = d5[(d5.index >= t) & (d5.index < t + pd.Timedelta(minutes=ln))]
                if not len(sub):
                    continue
                n += 1
                ok += abs(sub.Close.iloc[-1] - r.Close) < 1e-6 * r.Close and abs(sub.High.max() - r.High) < 1e-6 * r.High
            out[f"{ln}분 가정 일치"] = f"{ok}/{n}"
        res[hm] = out
    return res


def main():
    g = pd.read_parquet(OUT / "mtf_rows.parquet", filters=[("k", "==", 0), ("sym", "in", SYMS)])
    dg = pd.read_parquet(OUT / "mtf_diag.parquet", filters=[("k", "==", 0), ("sym", "in", SYMS)])
    rows, drows, info = [], [], {}
    for sym in SYMS:
        raw = M.read_raw(sym)
        info[sym] = dict(bars=bars_check(raw), hour_bars=hour_bar_check(raw))
        cache = {}
        for b, clf, act in CELLS:
            P = cache.get(b) or cache.setdefault(b, M.build(raw, b))
            flag = P["flags"][clf] if clf != "BASE" else np.zeros(len(P["O"]), bool)
            tgt = M.target(P["s"], flag, act)
            for bd, (cost, ideal) in BOARDS.items():
                mine, _ = M.metrics(P, tgt, cost, ideal)
                ref = g[(g.sym == sym) & (g.b == b) & (g.clf == clf) & (g.action == act) & (g.board == bd)]
                assert len(ref) == 1, (sym, b, clf, act, bd, len(ref))
                ref = ref.iloc[0]
                for w in ("sel", "conf"):
                    for c in COLS:
                        rows.append(dict(sym=sym, b=b, clf=clf, action=act, board=bd, window=w, col=c,
                                         mine=mine[f"{c}_{w}"], ref=float(ref[f"{c}_{w}"])))
            if clf != "BASE":
                for bd, ideal in (("real0", False), ("ideal", True)):
                    _, pos = M.metrics(P, M.target(P["s"], flag, "A3"), 0.0, ideal)
                    mine = M.diag(P, flag, pos)
                    ref = dg[(dg.sym == sym) & (dg.b == b) & (dg.clf == clf) & (dg.board == bd)].iloc[0]
                    for c, v in mine.items():
                        drows.append(dict(sym=sym, b=b, clf=clf, board=bd, col=c, mine=v, ref=float(ref[c])))
        print(sym, info[sym], flush=True)
    r = pd.DataFrame(rows)
    r["diff"] = (r.mine - r.ref).abs()
    # 결과 파일은 float32 저장이라 값 크기에 비례한 반올림 오차(≤ 6e-8 × |값|)를 허용 범위에 더한다
    r["tol"] = 1e-6 + 6e-8 * r.ref.abs()
    r["ok"] = r["diff"] <= r["tol"]
    d = pd.DataFrame(drows)
    d["diff"] = (d.mine - d.ref).abs()
    d["ok"] = (d["diff"] <= 1e-9) | (d.mine.isna() & d.ref.isna())
    r.to_csv(OUT / "mtfaudit_reimpl_cells.csv", index=False)
    d.to_csv(OUT / "mtfaudit_reimpl_diag.csv", index=False)
    summ = dict(cells=len(CELLS), syms=SYMS, values=len(r), mismatched=int((~r.ok).sum()), max_abs_diff=float(r["diff"].max()),
                max_abs_diff_by_col=r.groupby("col")["diff"].max().to_dict(),
                diag_values=len(d), diag_mismatched=int((~d.ok).sum()), diag_max_abs_diff=float(d["diff"].max()), per_sym=info)
    json.dump(summ, open(OUT / "mtfaudit_reimpl.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_sym"}, ensure_ascii=False, indent=1, default=str))
    if (~r.ok).any():
        print(r[~r.ok].head(30).to_string())


if __name__ == "__main__":
    main()
