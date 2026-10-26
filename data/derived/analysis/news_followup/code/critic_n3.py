"""반박 검산 N3: '하락 상태 R·S 기사 → 확정 대기보다 +7.5% 싸게'를 실시간으로 알 수 있는 반등 조건의 아무 날과 비교."""
import pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
import n3_analyze as A
raw = pd.read_parquet(HERE / "cache" / "n3_raw.parquet")
z = A.zz_runs(raw, "SOXX")
C, O = z["C"], z["O"]
idx = np.flatnonzero((z["dates"] >= "2024-01-02") & (z["dates"] <= "2026-10-23"))
st = z["ZZA3"]["st"]
rows = []
for ie in idx:
    if st[ie - 1] != 0:
        continue
    r = A.classify(z, "ZZA3", int(ie), "R")
    if not r["cat"].startswith("선행"):
        continue
    r5 = C[ie - 1] / C[ie - 6] - 1
    # 마지막 하락 확정 뒤 최저 종가 대비 (실시간)
    pcs = [k for k in range(len(z["ZZA3"]["pc"])) if z["ZZA3"]["pt"][k] == -1 and z["ZZA3"]["pc"][k] <= ie - 1]
    start = z["ZZA3"]["pi"][pcs[-1]] if pcs else 0
    runmin = C[start:ie].min()
    rows.append(dict(ie=ie, date=z["dates"][ie].date(), year=z["dates"][ie].year, cat=r["cat"], gain=r["gain"], r5=r5, off_low=C[ie - 1] / runmin - 1))
b = pd.DataFrame(rows)
print("하락 상태 아무 날", len(b), "평균 %.2f%%" % (100 * b.gain.mean()))
for lab, m in {"직전5일>0": b.r5 > 0, "저점대비>=3%": b.off_low >= .03, "저점대비>=5%": b.off_low >= .05, "둘다": (b.r5 > 0) & (b.off_low >= .03)}.items():
    s = b[m]
    print(lab, len(s), "평균 %.2f%% 양수 %.0f%% 저점 뒤 비율 %.0f%%" % (100 * s.gain.mean(), 100 * (s.gain > 0).mean(), 100 * s.cat.ne("선행·극점 앞").mean()),
          "국면(10일 간격)", int(1 + (np.diff(s.ie.values) > 10).sum()) if len(s) else 0)
ev = {"N07": "2024-08-07", "N09": "2024-09-24", "N15": "2025-04-11", "N23": "2026-03-31", "N30": "2026-07-31"}
for k, d in ev.items():
    x = b[b.date == pd.Timestamp(d).date()]
    print(k, x[["r5", "off_low", "cat", "gain"]].round(3).to_dict("records"))
print("저점 뒤 비율(아무 하락 날):", round(b.cat.ne("선행·극점 앞").mean(), 3))
