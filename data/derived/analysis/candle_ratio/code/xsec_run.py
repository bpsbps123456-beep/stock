"""xsec 갈래: 144종목 전부에 run_grid.grid를 비용 0으로 돌려 칸별 결과를 npy로 남긴다.

py code/xsec_run.py 5m [프로세스 수=3]
→ results/xsec/<base>/<SYM>.npy  float32 (4, 칸): ret_sel, ret_conf(로그), hold_sel/bars_sel, hold_conf/bars_conf
→ results/xsec/<base>/<SYM>_ent.npy uint16 (2, 칸): ent_sel, ent_conf
→ results/xsec/<base>/<SYM>_meta.json  보유 수익(세션·시간봉별), 하루 봉 수 중앙값
→ results/xsec/cells_<base>.parquet  칸 순서표(모든 종목 공통)
"""
import itertools
import json
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import BS, CACHE, HERE, NS, RATIOS, SS, TF_1H, TF_5M, WIN, load  # noqa: E402
import run_grid  # noqa: E402

OUT = HERE / "results" / "xsec"
KEY = ["sess", "tf", "N", "ratio", "dir", "dayflat", "b", "s"]


def cell_table(base):
    tfs = TF_5M if base == "5m" else TF_1H
    rows = []
    for sess, tf, n, kind in itertools.product(("eth", "rth"), tfs, NS, RATIOS):
        for d in ("follow", "fade"):
            for df in (False, True):
                for b in BS:
                    for s in SS:
                        rows.append((sess, tf, n, kind, d, df, b, s))
    return pd.DataFrame(rows, columns=KEY)


def bars_per_day(sym, base):
    d = load(sym, base)
    t = d.index
    mins = np.asarray(t.hour * 60 + t.minute)
    day = pd.Series(t.date)
    w = WIN[base]
    in_win = np.asarray((day >= w["sel"][0]) & (day <= w["conf"][1]))
    out = {}
    for sess, (a, b) in (("eth", (240, 1200)), ("rth", (570, 960))):
        k = (mins >= a) & (mins < b) & in_win
        vc = day[k].value_counts()
        out[f"bpd_{sess}"] = float(vc.median()) if len(vc) else 0.0
        out[f"days_{sess}"] = int(len(vc))
    # 연장 봉 중 프리(04:00~09:30)·애프터(16:00~20:00) 하루 봉 수 중앙값
    for nm, (a, b) in (("pre", (240, 570)), ("post", (960, 1200))):
        k = (mins >= a) & (mins < b) & in_win
        vc = day[k].value_counts().reindex(sorted(set(day[in_win])), fill_value=0)
        out[f"bpd_{nm}"] = float(vc.median()) if len(vc) else 0.0
    return out


def work(args):
    sym, base = args
    d = OUT / base
    f = d / f"{sym}.npy"
    if f.exists() and (d / f"{sym}_meta.json").exists():
        return sym, "skip", 0.0
    t0 = time.time()
    cells = cell_table(base)
    g = run_grid.grid(sym, base, costs_fn=lambda s: [0.0])
    if len(g) == len(cells) and (g[KEY].reset_index(drop=True) == cells).all().all():
        a = g
    else:  # 봉이 모자라 빠진 시간봉이 있으면 칸 순서에 맞춰 NaN으로 채운다
        a = cells.merge(g, on=KEY, how="left")
    hs = (a.hold_sel / a.bars_sel).to_numpy(np.float32)
    hc = (a.hold_conf / a.bars_conf).to_numpy(np.float32)
    arr = np.vstack([a.ret_sel.to_numpy(np.float32), a.ret_conf.to_numpy(np.float32), hs, hc])
    ent = np.vstack([a.ent_sel.fillna(0).to_numpy(np.uint16), a.ent_conf.fillna(0).to_numpy(np.uint16)])
    bhv = g.groupby(["sess", "tf"])[["bh_sel", "bh_conf", "bars_sel", "bars_conf"]].first().reset_index()
    meta = dict(sym=sym, base=base, n_rows=int(len(g)), n_cells=int(len(cells)),
                bh=bhv.to_dict(orient="list"), **bars_per_day(sym, base))
    np.save(d / f"{sym}_ent.npy", ent)
    (d / f"{sym}_meta.json").write_text(json.dumps(meta, default=float), encoding="utf-8")
    np.save(f, arr)  # 마지막에 써서 중단되면 다시 돈다
    return sym, len(g), time.time() - t0


if __name__ == "__main__":
    base = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    (OUT / base).mkdir(parents=True, exist_ok=True)
    cell_table(base).to_parquet(OUT / f"cells_{base}.parquet")
    syms = sorted(p.name[: -len(f"_{base}.parquet")] for p in CACHE.glob(f"*_{base}.parquet"))
    if len(sys.argv) > 3:
        syms = sys.argv[3].split(",")
    t0 = time.time()
    with Pool(min(nproc, 3)) as pool:
        for i, (sym, n, dt) in enumerate(pool.imap_unordered(work, [(s, base) for s in syms]), 1):
            print(f"{i}/{len(syms)} {sym} {n} {dt:.1f}s 누적 {time.time() - t0:.0f}s", flush=True)
