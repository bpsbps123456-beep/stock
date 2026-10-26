"""반박 검산 N1: (1) 20일 결과의 묶음 길이 민감도(월→분기→반년→해), (2) 2003~19 대 2020~26 차이 검정,
(3) 2003~23 구간이 2024~26 효과 크기(+0.73%p)를 배제하는지."""
import pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
import n1_run as R
from n1_lib import block_boot

op, cl = R.load("raw")
for s in R.SYMS:
    d, ev = R.build(op, cl, s)
    for k in [d] + list(ev.values()):
        k["half"] = k.year * 2 + (k.index.month - 1) // 6
    # 같은 날 C 분할(금리 하락 vs 상승)
    C = ev["C"]
    for h in [5, 20]:
        for col in ["month", "quarter", "half", "year"]:
            g = {k: R.in_period(v, 2003, 2023) for k, v in ev.items()}
            bt = block_boot({"M": g["M"], "C": g["C"], "D": g["D"]}, h, R.months_of(d, 2003, 2023, col), B=3000, col=col)
            gc = R.in_period(C, 2003, 2023)
            bs = block_boot({"M": gc[gc.dy5_sig < 0], "X": gc[gc.dy5_sig > 0]}, h, R.months_of(d, 2003, 2023, col), B=3000, col=col)
            print(s, h, col, "M-b %.2f [%.2f,%.2f]" % (100*bt["dmean_b"], *(100*np.array(bt["dmean_b_ci"]))),
                  "M-C %.2f [%.2f,%.2f]" % (100*bt["dmean_C"], *(100*np.array(bt["dmean_C_ci"]))),
                  "M-D %.2f [%.2f,%.2f]" % (100*bt["dmean_D"], *(100*np.array(bt["dmean_D_ci"]))),
                  "splitC %.2f [%.2f,%.2f]" % (100*bs["dmean_X"], *(100*np.array(bs["dmean_X_ci"]))))
    # 시기 차이: (2020~26 M-b) - (2003~19 M-b), 5일, 월 묶음 독립 부트스트랩
    M = ev["M"]
    rng = np.random.default_rng(1)
    out = {}
    for lab, (y0, y1) in {"early": (2003, 2019), "late": (2020, 2026)}.items():
        v = R.in_period(M, y0, y1); v = v[v.r5.notna() & v.b_mean5.notna()]
        e = (v.r5 - v.b_mean5).to_numpy(); mo = v.month.to_numpy()
        months = np.unique(R.in_period(d, y0, y1).month)
        bs = []
        for _ in range(3000):
            pick = rng.choice(months, len(months))
            cnt = pd.Series(pick).value_counts()
            w = pd.Series(mo).map(cnt).fillna(0).to_numpy()
            bs.append((w @ e) / w.sum() if w.sum() else np.nan)
        out[lab] = (e.mean(), np.array(bs), len(e))
    dlt = out["late"][1] - out["early"][1]
    print(s, "5d M-b early %.2f (n%d) late %.2f (n%d) late-early %.2f [%.2f,%.2f]" % (100*out["early"][0], out["early"][2], 100*out["late"][0], out["late"][2],
          100*(out["late"][0]-out["early"][0]), *np.nanpercentile(100*dlt, [2.5, 97.5])))
