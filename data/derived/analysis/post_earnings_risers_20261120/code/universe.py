"""TIME(456600)·KoAct(2ETFQ1) 2026 일별 구성 스냅샷에서 미국 상장 보유 종목과 보유 날짜를 뽑는다.

결과: results/universe.csv(종목 요약), results/held_days.csv(종목 × 펀드 × 날짜 × 수량).
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1] / "results"
FUNDS = {"456600": "TIME", "2ETFQ1": "KoAct"}


def main() -> None:
    rows = []
    for fid, fname in FUNDS.items():
        for f in sorted(glob.glob(str(ROOT / f"data/normalized/etf_holdings/*_{fid}.json"))):
            d = json.load(open(f, encoding="utf-8"))
            dt = os.path.basename(f)[:10]
            for h in d["holdings"]:
                code = h.get("code", "")
                if h.get("asset_type") != "security" or not code.endswith(" US EQUITY"):
                    continue
                tk = code[: -len(" US EQUITY")].strip().replace("/", "-")
                rows.append({"date": dt, "fund": fname, "ticker": tk, "name": h.get("name"),
                             "qty": h.get("qty"), "weight": h.get("weight")})
    held = pd.DataFrame(rows)
    held.to_csv(OUT / "held_days.csv", index=False)
    g = held.groupby("ticker")
    uni = pd.DataFrame({
        "name": g["name"].last(),
        "funds": g["fund"].agg(lambda s: "+".join(sorted(set(s)))),
        "first_held": g["date"].min(),
        "last_held": g["date"].max(),
        "days_held": g["date"].nunique(),
        "max_weight": g["weight"].max().round(2),
    }).reset_index()
    uni.to_csv(OUT / "universe.csv", index=False)
    print(len(uni), "tickers;", uni.funds.value_counts().to_dict())


if __name__ == "__main__":
    main()
