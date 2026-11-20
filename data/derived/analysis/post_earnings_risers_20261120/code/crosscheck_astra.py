"""Astra가 H1에서 잠가 H2를 통과한 열위 규칙 3개를 Claude가 정의만 보고 독립 재계산한다(Astra 코드 미사용).

정의 출처: astra/rules/REPORT.md(요인 시점 표), 문턱은 astra/rules/results.json의 잠근 값.
- intraday20  = 반응일 전 20거래일 Σ[ln(수정종가/수정시가)종목 − 같은 값 QQQ] × 100
- efficiency20 = |Σ 20일 일별 로그수익| / Σ|일별 로그수익|
- R(x) = (H1 유효값 중 x보다 작은 수 + 같은 수/2) / H1 유효 수,  점수 = (R(A)+R(B))/2
- H1 = 1·2분기 시즌 중 그 진입의 결과가 끝난 사건. E1은 시각 불명 1건(SAP 1/29) 제외
결과: results/crosscheck_astra.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))
import rules as R  # noqa: E402


def main() -> None:
    d = R.load()
    px = {k: pd.read_parquet(BASE / f"cache/px_{k}.parquet") for k in ("open", "close", "adjclose")}
    idx = px["close"]["QQQ"].dropna().index
    px = {k: v.reindex(idx) for k, v in px.items()}
    ac = px["adjclose"]
    ao = px["open"] * ac / px["close"]
    intra = np.log(ac / ao)
    lr = np.log(ac / ac.shift(1))
    i20, e20 = [], []
    for e in d.itertuples():
        k, t = int(e.k), e.ticker
        w = slice(k - 20, k)
        i20.append(100 * float((intra[t].iloc[w] - intra["QQQ"].iloc[w]).sum()))
        r = lr[t].iloc[w]
        e20.append(float(abs(r.sum()) / r.abs().sum()) if r.abs().sum() > 0 else np.nan)
    d["intraday20"], d["efficiency20"] = i20, e20
    seas, rd = d.season.to_numpy(), d.reaction_date.astype(str).to_numpy()
    spec = {c["family"] + str(c["rank"]): c["spec"] for c in
            json.load(open(BASE / "astra/rules/results.json", encoding="utf-8"))["confirmation"] if c["hit70"]}
    astra = {"E1_down1": ((29, 36), (20, 26)), "E1_down3": ((28, 35), (30, 41)), "E6_down1": ((24, 30), (13, 16))}
    rows = []
    for key, sp in spec.items():
        y = d["d20_x" if key.startswith("E1") else "e6_x"].to_numpy().copy()
        if key.startswith("E1"):
            y[(d.timing == "unk").to_numpy()] = np.nan
        valid = ~np.isnan(y)
        h1 = np.isin(seas, ["2026Q1", "2026Q2"]) & valid
        h2 = ((seas == "2026Q3") | ((seas == "2026Q4") & (rd <= "2026-10-21"))) & valid
        a, b = d[sp["a"]].to_numpy(float), d[sp["b"]].to_numpy(float)

        def rank(x):
            ref = x[h1 & ~np.isnan(x)]
            out = np.array([((ref < v).sum() + 0.5 * (ref == v).sum()) / len(ref) if not np.isnan(v) else np.nan for v in x])
            return out
        ra, rb = rank(a), rank(b)
        score = (ra + (rb if sp["sign"] == 1 else 1 - rb)) / 2
        sig = (score <= sp["score_cut"]) if sp["score_op"] == "le" else (score >= sp["score_cut"])
        g = d[sp["gate"]].to_numpy(float)
        sig &= (g <= sp["gate_cut"]) if sp["gate_op"] == "le" else (g >= sp["gate_cut"])
        sig &= ~np.isnan(score) & ~np.isnan(g)
        rec = {"rule": key, "spec": json.dumps(sp, ensure_ascii=False)}
        for nm, m in (("H1", h1), ("Q1", (seas == "2026Q1") & valid), ("Q2", (seas == "2026Q2") & valid), ("H2", h2)):
            s = sig & m
            rec[f"{nm}_hit"], rec[f"{nm}_n"] = int((y[s] < 0).sum()), int(s.sum())
        rec["H2_mean"] = round(float(np.nanmean(y[sig & h2])), 2)
        # 조기 실전(판정 아님): 반응일 10/22~11/12, 진입 시가 → min(만기, 11/18 종가)
        n = len(idx)
        s0 = 1 if key.startswith("E1") else 6
        late = sig & (rd >= "2026-10-22") & (rd <= "2026-11-12")
        ys = []
        for i in np.where(late)[0]:
            k, t = int(d.k.iloc[i]), d.ticker.iloc[i]
            end = min(k + (20 if s0 == 1 else 25), n - 2)  # yfinance 11-19 행 일부 결측
            if k + s0 <= end:
                ys.append(100 * ((ac[t].iloc[end] / ao[t].iloc[k + s0] - 1) - (ac["QQQ"].iloc[end] / ao["QQQ"].iloc[k + s0] - 1)))
        ys = np.array([v for v in ys if not np.isnan(v)])
        rec["early_n"], rec["early_lag"] = len(ys), round(float((ys < 0).mean()), 3) if len(ys) else np.nan
        rec["H2_tickers"] = int(d.ticker[sig & h2].nunique())
        (ah, an), (bh, bn) = astra[key]
        rec["astra_H1"], rec["astra_H2"] = f"{ah}/{an}", f"{bh}/{bn}"
        rec["match"] = (rec["H1_hit"], rec["H1_n"], rec["H2_hit"], rec["H2_n"]) == (ah, an, bh, bn)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "results/crosscheck_astra.csv", index=False)
    print(out.drop(columns="spec").to_string(index=False))


if __name__ == "__main__":
    main()
