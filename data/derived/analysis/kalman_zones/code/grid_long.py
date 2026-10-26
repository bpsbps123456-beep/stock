"""2026년 창(1시간봉 기반): 정규장 60~360분 x 지표 x 5단계 기준 x 규칙. 선택 1~5월, 확인 6~9월."""
import pandas as pd
from core import *
from grid_short import run

if __name__ == "__main__":
    D = load()
    halfs = set()
    for s in SYMS:
        halfs |= half_days(D[(s, "1h")])
    df = run(D, halfs, [60, 120, 180, 240, 360], (False,), base_key="1h",
             eval_start=pd.Timestamp("2026-01-02").date(), split=pd.Timestamp("2026-06-01").date(), tag="long")
    df.to_parquet("grid_long2.parquet")
    print(df.groupby("kind").size())
