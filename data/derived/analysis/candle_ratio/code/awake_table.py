"""awake 갈래 요약표: awake_summary_{5m,1h}.json → results/awake_table.csv (값은 % 또는 비율, 귀무 분위는 공통 k 이동 중 실제보다 낮은 비율).
py code/awake_table.py"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import HERE  # noqa: E402

ROWS = [("orig", "원안 연장 40분(1h창 60분) N10 count follow 0.5/0.5 — 깨어 있음+0.1%"),
        ("origI", "원안 이상적 체결·비용 0 (참고)"),
        ("t1", "종목별 고르기 1위(수익) → 확인"),
        ("t1x", "종목별 고르기 1위(노출 맞춘 초과) → 확인"),
        ("t20", "칸별 종목 중앙 고르기 상위 20칸(수익)"),
        ("t20x", "칸별 종목 중앙 고르기 상위 20칸(초과)"),
        ("lead", "단서 칸 정규장 60분 N5 body fade 0.5/0.5 — 깨어 있음+0.05%"),
        ("leadI", "단서 칸 이상적·비용 0 (참고)")]


def g(st, k):
    v = st.get(k)
    return (v["actual"], v["q"]) if v else (np.nan, None)


rows = []
for base in ("5m", "1h"):
    o = json.loads((HERE / "results" / f"awake_summary_{base}.json").read_text(encoding="utf-8"))
    st = o["stats"]
    for tag, name in ROWS:
        r = dict(window=base, rule=name)
        for c, nm in (("rc", "확인수익"), ("dbh_c", "보유대비차"), ("ex_c", "노출초과"), ("dd_c", "낙폭축소")):
            v, q = g(st, f"{tag}_{c}_med")
            r[nm + "_중앙%"] = round(float(np.expm1(v)) * 100 if c in ("rc", "ex_c") else v * 100, 1)
            if c != "rc":
                vp, qp = g(st, f"{tag}_{c}_pos")
                r[nm + "_+종목%"] = round(vp * 100, 1)
                r[nm + "_+종목_귀무분위"] = qp
        r["노출초과_중앙_귀무분위"] = g(st, f"{tag}_ex_c_med")[1]
        if tag in ("orig", "origI", "lead", "leadI"):
            v, q = g(st, f"{tag}_ex_s_med"); r["고르기구간_노출초과중앙%"] = round(float(np.expm1(v)) * 100, 1); r["고르기구간_초과중앙_귀무분위"] = q
            v, q = g(st, f"{tag}_ex_s_pos"); r["고르기구간_초과+종목%"] = round(v * 100, 1)
            v, q = g(st, f"{tag}_dd_s_pos"); r["고르기구간_낙폭축소+종목%"] = round(v * 100, 1)
        r["귀무이동수"] = o["n_null"]; r["칸수"] = o["n_cells"]; r["종목수"] = o["n_syms"]
        rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(HERE / "results" / "awake_table.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(T.drop(columns=["rule"]).to_string())
