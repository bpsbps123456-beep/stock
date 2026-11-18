"""Print the frozen replay using the production forecaster. Run from any directory."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from desk.bottom_outlook import replay_outlook


if __name__ == "__main__":
    close = pd.read_csv(HERE / "soxx_close.csv", index_col="date", parse_dates=True)["SOXX"]
    outlook, cases = replay_outlook(close)
    print(json.dumps(outlook, indent=2, ensure_ascii=False, allow_nan=False))
    print("예측일       결과일        당일종가   하단   중앙   상단   실제 최저")
    for case in cases:
        c = close.loc[case["date"]]
        levels = c * (1 + case["bands"][outlook["method"]])
        actual = c * (1 + case["actual"])
        print(case["date"], case["end"], f"{c:8.2f}", *(f"{v:6.2f}" for v in levels), f"{actual:8.2f}")
