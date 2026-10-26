"""z3critic: zones3.prepare 미리보기 점검. 자료를 시각 T에서 자른 사본(5분봉 T 전, 일봉은 16:00 ET 마감이 T 이하인 날만)으로
다시 계산해, 끝 시각 ≤ T인 봉의 s·flag 28개·z5가 전체 자료 계산과 같은지 본다."""
from __future__ import annotations
import shutil, tempfile, pathlib
import numpy as np, pandas as pd
import engine, zones3
from engine import ET

SYMS = ["MU", "BE", "NBIS", "NVDA", "SKHY"]
TS = ["2026-07-24 12:00", "2026-08-12 16:00", "2026-08-12 15:20", "2026-09-04 19:20", "2026-09-17 10:40", "2026-10-01 16:00", "2026-10-14 06:40"]
orig = engine.CACHE
tmp = pathlib.Path(tempfile.mkdtemp())
bad = 0; checked = 0
for sym in SYMS:
    if not (orig / f"{sym}_5m.parquet").exists():
        continue
    engine.CACHE = orig; zones3.CACHE = orig
    full = zones3.prepare(sym, "5m", "dst")
    for T in TS:
        t = pd.Timestamp(T, tz=ET)
        d5 = pd.read_parquet(orig / f"{sym}_5m.parquet"); d5 = d5[d5.index.tz_convert(ET) < t]
        d1 = pd.read_parquet(orig / f"{sym}_1d.parquet")
        dd = pd.DatetimeIndex(d1.index).date
        close = pd.DatetimeIndex(pd.to_datetime(dd)).tz_localize(ET) + pd.Timedelta(minutes=960)
        d1 = d1[np.asarray(close <= t)]
        d5.to_parquet(tmp / f"{sym}_5m.parquet"); d1.to_parquet(tmp / f"{sym}_1d.parquet")
        engine.CACHE = tmp; zones3.CACHE = tmp
        cut = zones3.prepare(sym, "5m", "dst")
        assert full["end"][0] > 1e15 and full["end"][0] < 1e16  # 마이크로초
        m = full["end"] <= t.value // 1000
        nb = int(m.sum()); assert len(cut["s"]) >= nb
        diff = {"s": int((full["s"][:nb] != cut["s"][:nb]).sum()),
                "z5": int((~np.isclose(np.nan_to_num(full["z5"][:nb], nan=-9), np.nan_to_num(cut["z5"][:nb], nan=-9))).sum())}
        for k in full["flags"]:
            diff[k] = int((full["flags"][k][:nb] != cut["flags"][k][:nb]).sum())
        tot = sum(diff.values()); bad += tot; checked += 1
        print(sym, T, "봉", nb, "불일치", tot, {k: v for k, v in diff.items() if v})
engine.CACHE = orig; zones3.CACHE = orig
shutil.rmtree(tmp)
print("점검", checked, "총 불일치", bad)
