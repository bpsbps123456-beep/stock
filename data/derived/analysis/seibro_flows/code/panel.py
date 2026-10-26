"""SEIBRO 원자료(raw/*.json) → 창별 종목 표와 목록 합계. 가격은 쓰지 않는다.

출력(derived/):
- settle_{weekly,daily}.parquet: 창(start, end) × ISIN. 네 목록(d1 매수·d2 매도·d3 합계·d4 순매수) 합집합.
  in_d1..in_d4 = 그 목록에 들었는지, rank_d1..rank_d4 = 순위.
- totals_{weekly,daily}.parquet: 창마다 목록 합계. sum_buy_d1(ΣB), sum_net_d4(ΣN⁺), sum_sell_d2, sum_tot_d3.
- custody.parquet: 날짜 × ISIN 보관금액(Top50), custody_totals.parquet: 날짜마다 Top50 합계.
- isins.csv: ISIN, 이름(가장 최근), 등장 창 수, 금액 합 — 분류(classes.csv) 재료.
- qa.json: 같은 창·ISIN인데 목록마다 금액이 다른 경우, 빈 창, 파일 오류 수.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "derived"


def load(kind: str) -> tuple[pd.DataFrame, list[str]]:
    recs, bad = [], []
    for p in sorted((RAW / kind).glob("*/*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad.append(str(p))
            continue
        base = {"start": rec["start"], "end": rec["end"], "dtype": rec["dtype"], "n_list": rec["n"]}
        if rec["meta"]:
            base.update({f"meta_{k}": v for k, v in rec["meta"].items()})
        if not rec["rows"]:
            recs.append({**base, "isin": None})
        for r in rec["rows"]:
            recs.append({**base, **r})
    return pd.DataFrame(recs), bad


def settle_panel(kind: str, qa: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, bad = load(kind)
    qa[f"{kind}_bad_files"] = bad
    key = ["start", "end"]
    lists = df.dropna(subset=["isin"])
    tot = (lists.groupby(key + ["dtype"]).agg(sum_buy=("buy", "sum"), sum_sell=("sell", "sum"),
                                               sum_tot=("tot", "sum"), sum_net=("net", "sum"), n=("isin", "size"))
           .reset_index())
    piv = tot.pivot_table(index=key, columns="dtype", values=["sum_buy", "sum_sell", "sum_tot", "sum_net", "n"])
    totals = pd.DataFrame({
        "sum_buy_d1": piv[("sum_buy", "1")], "sum_sell_d2": piv[("sum_sell", "2")],
        "sum_tot_d3": piv[("sum_tot", "3")], "sum_net_d4": piv[("sum_net", "4")],
        "n_d1": piv[("n", "1")], "n_d2": piv[("n", "2")], "n_d3": piv[("n", "3")], "n_d4": piv[("n", "4")],
    }).reset_index()
    # 빈 목록(휴장 창)과 받은 목록 수
    got = df.groupby(key)["dtype"].nunique().rename("lists_fetched").reset_index()
    totals = got.merge(totals, on=key, how="left")
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    if meta_cols:
        totals = totals.merge(df.groupby(key)[meta_cols].first().reset_index(), on=key, how="left")

    # 같은 창·ISIN은 목록이 달라도 금액이 같아야 한다
    amt = ["buy", "sell", "tot", "net"]
    spread = lists.groupby(key + ["isin"])[amt].agg(lambda s: s.max() - s.min())
    incons = spread[(spread > 0.5).any(axis=1)]
    qa[f"{kind}_inconsistent_rows"] = int(len(incons))
    qa[f"{kind}_inconsistent_sample"] = incons.head(10).reset_index().astype(str).to_dict("records")

    first = lists.sort_values("dtype").groupby(key + ["isin"])[["name"] + amt].first()
    ranks = lists.pivot_table(index=key + ["isin"], columns="dtype", values="rank", aggfunc="min")
    ranks.columns = [f"rank_d{c}" for c in ranks.columns]
    panel = first.join(ranks).reset_index()
    for c in "1234":
        panel[f"in_d{c}"] = panel[f"rank_d{c}"].notna()
    qa[f"{kind}_windows"] = int(totals.shape[0])
    qa[f"{kind}_windows_incomplete_lists"] = int((totals["lists_fetched"] < 4).sum())
    return panel, totals


def custody_panel(qa: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, bad = load("custody")
    qa["custody_bad_files"] = bad
    lists = df.dropna(subset=["isin"])
    panel = lists[["start", "isin", "name", "rank", "amt"]].rename(columns={"start": "date"})
    totals = lists.groupby("start").agg(sum_amt=("amt", "sum"), n=("isin", "size")).reset_index()
    totals = totals.rename(columns={"start": "date"})
    empty = sorted(set(df["start"]) - set(totals["date"]))
    qa["custody_dates"] = int(df["start"].nunique())
    qa["custody_empty_dates"] = empty
    return panel, totals


def main() -> None:
    OUT.mkdir(exist_ok=True)
    qa: dict = {}
    names = []
    for kind in ["weekly", "daily"]:
        if not (RAW / kind).exists():
            continue
        panel, totals = settle_panel(kind, qa)
        panel.to_parquet(OUT / f"settle_{kind}.parquet", index=False)
        totals.to_parquet(OUT / f"totals_{kind}.parquet", index=False)
        names.append(panel.assign(src=kind, amount=panel["tot"]).rename(columns={"start": "date"})
                     [["isin", "name", "date", "src", "amount"]])
    if (RAW / "custody").exists():
        panel, totals = custody_panel(qa)
        panel.to_parquet(OUT / "custody.parquet", index=False)
        totals.to_parquet(OUT / "custody_totals.parquet", index=False)
        names.append(panel.assign(src="custody", amount=panel["amt"])[["isin", "name", "date", "src", "amount"]])
    allp = pd.concat(names)
    latest = allp.sort_values("date").groupby("isin")["name"].last()
    stats = allp.groupby(["isin", "src"]).agg(windows=("date", "nunique"), amount=("amount", "sum")).unstack("src")
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    isins = stats.join(latest).reset_index().sort_values(stats.columns[0], ascending=False)
    isins.to_csv(OUT / "isins.csv", index=False, encoding="utf-8-sig")
    qa["unique_isins"] = int(len(isins))
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in qa.items() if not k.endswith("sample")}, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
