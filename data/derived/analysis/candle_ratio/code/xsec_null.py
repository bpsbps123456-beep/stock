"""xsec 갈래 귀무: 비율 p를 거래일 단위로 원형 이동(run_grid.grid의 shift_days)한 뒤 같은 격자를 전 종목에 돌린다.
신호와 가격의 시점이 어긋난 상태에서 '칸별 영역' 판정이 몇 칸 나오는지 우연 수준을 잰다.

py code/xsec_null.py 5m 20   → results/xsec/5m_shift20/<SYM>.npy (형식은 xsec_run과 같다)
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from xsec_run import CACHE, OUT, cell_table, run_grid, KEY


def work(args):
    sym, base, sh = args
    d = OUT / f"{base}_shift{sh}"
    f = d / f"{sym}.npy"
    if f.exists():
        return sym, "skip", 0.0
    t0 = time.time()
    cells = cell_table(base)
    g = run_grid.grid(sym, base, costs_fn=lambda s: [0.0], shift_days=sh)
    a = g if (len(g) == len(cells) and (g[KEY].reset_index(drop=True) == cells).all().all()) else cells.merge(g, on=KEY, how="left")
    arr = np.vstack([a.ret_sel.to_numpy(np.float32), a.ret_conf.to_numpy(np.float32),
                     (a.hold_sel / a.bars_sel).to_numpy(np.float32), (a.hold_conf / a.bars_conf).to_numpy(np.float32)])
    ent = np.vstack([a.ent_sel.fillna(0).to_numpy(np.uint16), a.ent_conf.fillna(0).to_numpy(np.uint16)])
    bhv = g.groupby(["sess", "tf"])[["bh_sel", "bh_conf", "bars_sel", "bars_conf"]].first().reset_index()
    np.save(d / f"{sym}_ent.npy", ent)
    (d / f"{sym}_meta.json").write_text(json.dumps(dict(sym=sym, base=base, shift_days=sh, bh=bhv.to_dict(orient="list")),
                                                   default=float), encoding="utf-8")
    np.save(f, arr)
    return sym, len(g), time.time() - t0


if __name__ == "__main__":
    base, sh = sys.argv[1], int(sys.argv[2])
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    (OUT / f"{base}_shift{sh}").mkdir(parents=True, exist_ok=True)
    syms = sorted(p.name[: -len(f"_{base}.parquet")] for p in CACHE.glob(f"*_{base}.parquet"))
    t0 = time.time()
    with Pool(min(nproc, 3)) as pool:
        for i, (sym, n, dt) in enumerate(pool.imap_unordered(work, [(s, base, sh) for s in syms]), 1):
            print(f"{i}/{len(syms)} {sym} {n} {dt:.1f}s 누적 {time.time() - t0:.0f}s", flush=True)
