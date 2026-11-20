"""펀드 밖 종목 확인: 앞선 넓은 연구의 2026 실적 사건(시총 20억 달러+, 1,629종목) 중 두 펀드 161종목을 뺀 것에
가격만 쓰는 규칙을 그대로(펀드 H1 분포·잠근 문턱) 적용한다. 가격은 비수정 일봉(2026-10-28까지).

- A1: Astra E1 열위 1 = (R(발표 전 20일 QQQ 대비) + R(20일 장중 수익 QQQ 대비))/2 ≤ 0.3338 & 반응일 거래량 ≤ 2.191배
- S1: 반응일 거래량 ≤ 2.191배 하나만
결과: results/wide_check.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
WIDE = BASE.parent / "earnings_trend"
sys.path.insert(0, str(BASE / "code"))
import rules as R  # noqa: E402


def main() -> None:
    fund = R.load()
    spec = {c["family"] + str(c["rank"]): c["spec"] for c in
            json.load(open(BASE / "astra/rules/results.json", encoding="utf-8"))["confirmation"] if c["hit70"]}["E1_down1"]
    px = {k: pd.read_parquet(WIDE / f"cache/wide/{k}.parquet") for k in ("open", "close", "volume")}
    idx = px["close"]["QQQ"].dropna().index
    px = {k: v.reindex(idx) for k, v in px.items()}
    op, cl, vo = px["open"], px["close"], px["volume"]
    intra = np.log(cl / op)
    ev = pd.read_csv(WIDE / "results/events_wide.csv")
    ev = ev[(ev.reaction_date >= "2026-01-01") & ~ev.ticker.isin(set(fund.ticker))]
    n = len(idx)

    def feats(t, k):
        pre = 100 * ((cl[t].iloc[k - 1] / cl[t].iloc[k - 21] - 1) - (cl["QQQ"].iloc[k - 1] / cl["QQQ"].iloc[k - 21] - 1))
        i20 = 100 * float((intra[t].iloc[k - 20:k] - intra["QQQ"].iloc[k - 20:k]).sum())
        med = vo[t].iloc[k - 50:k].median()
        v0 = vo[t].iloc[k] / med if med and med > 0 else np.nan
        y = np.nan
        if k + 20 < n:
            y = 100 * ((cl[t].iloc[k + 20] / op[t].iloc[k + 1] - 1) - (cl["QQQ"].iloc[k + 20] / op["QQQ"].iloc[k + 1] - 1))
        return pre, i20, v0, y

    rows = []
    for e in ev.itertuples():
        if e.ticker not in cl.columns:
            continue
        rdt = pd.Timestamp(e.reaction_date)
        if rdt not in idx:
            continue
        k = idx.get_loc(rdt)
        if k < 60:
            continue
        pre, i20, v0, y = feats(e.ticker, k)
        rows.append({"ticker": e.ticker, "reaction_date": e.reaction_date, "pre20": pre, "intra20": i20, "vol0": v0, "y": y})
    w = pd.DataFrame(rows)
    # 펀드 H1 분포로 순위
    fd = fund  # 교차 재계산(crosscheck_astra.py)과 같은 정의
    pxf = {k: pd.read_parquet(BASE / f"cache/px_{k}.parquet") for k in ("open", "close", "adjclose")}
    fidx = pxf["close"]["QQQ"].dropna().index
    pxf = {k: v.reindex(fidx) for k, v in pxf.items()}
    fac = pxf["adjclose"]
    fintra = np.log(fac / (pxf["open"] * fac / pxf["close"]))
    fd["intraday20"] = [100 * float((fintra[r.ticker].iloc[int(r.k) - 20:int(r.k)] - fintra["QQQ"].iloc[int(r.k) - 20:int(r.k)]).sum())
                        for r in fd.itertuples()]
    yv = fd.d20_x.to_numpy().copy()
    yv[(fd.timing == "unk").to_numpy()] = np.nan
    h1 = fd.season.isin(["2026Q1", "2026Q2"]).to_numpy() & ~np.isnan(yv)
    refa = fd.px_pre20_x.to_numpy()[h1]
    refa = refa[~np.isnan(refa)]
    refb = fd.intraday20.to_numpy()[h1]
    refb = refb[~np.isnan(refb)]

    def rk(x, ref):
        return np.array([((ref < v).sum() + 0.5 * (ref == v).sum()) / len(ref) if not np.isnan(v) else np.nan for v in x])
    w["score"] = (rk(w.pre20.to_numpy(), refa) + rk(w.intra20.to_numpy(), refb)) / 2
    w["A1"] = (w.score <= spec["score_cut"]) & (w.vol0 <= spec["gate_cut"])
    w["S1"] = w.vol0 <= spec["gate_cut"]
    w["half"] = np.where(w.reaction_date < "2026-07-01", "H1", "H2")
    w = w[w.y.notna()]
    out = []
    for half, g in w.groupby("half"):
        for nm, m in (("기준(조건 없음)", np.ones(len(g), bool)), ("S1 반응일 거래량 ≤2.19배", g.S1.to_numpy()), ("A1 Astra 열위 1", g.A1.to_numpy())):
            yy = g.y.to_numpy()[m]
            out.append({"구간": half, "규칙": nm, "건수": int(m.sum()), "QQQ보다 못 간 비율": round(float((yy < 0).mean()), 3),
                        "중앙 초과수익": round(float(np.median(yy)), 2), "종목 수": int(g.ticker[m].nunique())})
    res = pd.DataFrame(out)
    res.to_csv(BASE / "results/wide_check.csv", index=False)
    print(res.to_string(index=False))
    print("H2 마지막 반응일", w[w.half == "H2"].reaction_date.max())


if __name__ == "__main__":
    main()
