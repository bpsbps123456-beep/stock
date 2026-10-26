"""dense.py와 같은 탐색이지만 판단을 '정규장 밖(개장 전·애프터)'에서만 한다: 개장 직후 봉에 반응하지 않는 운용."""
import time
from multiprocessing import Pool
import pandas as pd
import dense
from core import SYMS


def job2(args):
    win, tf, eth = args
    D, H, S, common, sel = dense.ctx(win)
    for s in SYMS:
        if "awake0" not in S[s]:
            S[s]["awake0"] = S[s]["is_awake"].copy()
            S[s]["is_awake"] = S[s]["awake0"] & ~S[s]["is_reg"]
    return dense.job(args)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("short", tf, eth) for tf in range(10, 365, 5) for eth in (False, True)]
    parts = []
    with Pool(11) as pool:
        for i, df in enumerate(pool.imap_unordered(job2, jobs), 1):
            parts.append(df)
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet("dense_preopen.parquet")
    print("done")
