"""BRIEF4 C3 H4 감사 보조 — 단독 실행 판(phantom/reset)의 2026 매매 목록과 '첫 교차 뒤 봉' 판의 성분.
사용: cd code && py audit4_H4_alone.py → results/audit4_H4_alone.json"""
import json, os, sys
os.environ["SOS_UNLOCK"] = "frozen"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lib import Data, hm
from audit4_H4 import chain_day, RES

D = Data("alp1m_long")
d26 = [k for k, dd in enumerate(D.days) if dd.year == 2026]
out = {}
for mode in ("phantom", "reset", "any"):
    tr = []
    for d in d26:
        tr += [t for t in chain_day(D, d, mode) if t["ok"]]
    df = pd.DataFrame(tr)
    df["day"] = [str(D.days[d]) for d in df["d"]]
    df["dec_close"] = [hm(i + 1) for i in df["dec_i"]]
    df["first_cross_since_s0"] = [bool(D.A["SOXL"]["Close"][r.d, r.s0:r.dec_i] .size == 0 or
                                       not (D.A["SOXL"]["Close"][r.d, r.s0:r.dec_i] >= np.minimum.accumulate(D.A["SOXL"]["Low"][r.d, r.s0:r.dec_i]) * 1.03).any())
                                  for r in df.itertuples()]
    out[mode] = {"n": int(len(df)), "mean": float(df["r"].mean()),
                 "first_cross_n": int(df["first_cross_since_s0"].sum()),
                 "first_cross_mean": float(df.loc[df["first_cross_since_s0"], "r"].mean()) if df["first_cross_since_s0"].any() else None,
                 "later_bar_n": int((~df["first_cross_since_s0"]).sum()),
                 "later_bar_mean": float(df.loc[~df["first_cross_since_s0"], "r"].mean()) if (~df["first_cross_since_s0"]).any() else None}
    if mode != "any":
        out[mode]["trades"] = df[["day", "dec_close", "wick", "r", "why"]].round(4).to_dict("records")
    print(mode, {k: v for k, v in out[mode].items() if k != "trades"})
    if mode != "any":
        print(df[["day", "dec_close", "wick", "r", "why"]].round(3).to_string())
json.dump(out, open(os.path.join(RES, "audit4_H4_alone.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
