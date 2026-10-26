"""xsec 갈래: 영역 칸(비용 0)이 편도 비용 몇 %까지 영역 조건을 지키는지, 왕복 1회당 초과가 얼마인지.

py code/xsec_breakeven.py → results/xsec_breakeven_5m.json
영역 조건: 종목 중앙 노출 맞춘 초과가 고르기·확인 모두 + 이고, 두 구간 모두 종목 60% 이상에서 +.
"""
import json

import numpy as np
import pandas as pd

from xsec_analyze import HERE, SPARSE_BPD, XS, load_all, pct

CS = [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]

if __name__ == "__main__":
    cells, syms, R, meta = load_all("5m", "5m")
    m5 = {p.stem[:-5]: json.loads(p.read_text(encoding="utf-8"))["bpd_eth"] for p in (XS / "5m").glob("*_meta.json")}
    sparse = (pd.Series(m5).reindex(syms) < SPARSE_BPD).fillna(False).to_numpy()
    eth = (cells.sess == "eth").to_numpy()
    C = pd.read_parquet(HERE / "results" / "xsec_cells_5m.parquet")
    reg = np.flatnonzero(C.region.to_numpy())
    fam = np.flatnonzero(((cells.sess == "rth") & (cells.dir == "fade") & (cells.tf <= 15)).to_numpy())
    out = {"n_region": int(len(reg))}
    for nm, idx in (("region", reg), ("rth_fade_tf<=15", fam)):
        res = {}
        for c in CS:
            lc = np.log(1 - c)
            ok = np.zeros(len(idx), bool)
            xs_meds, xc_meds = [], []
            for k, j in enumerate(idx):
                rows = ~sparse if eth[j] else np.ones(len(syms), bool)
                xs = R["rs"][rows, j] + 2 * R["es"][rows, j] * lc - R["hs"][rows, j] * R["bs"][rows, j]
                xc = R["rc"][rows, j] + 2 * R["ec"][rows, j] * lc - R["hc"][rows, j] * R["bc"][rows, j]
                ms, mc = np.median(xs), np.median(xc)
                ok[k] = ms > 0 and mc > 0 and np.mean(xs > 0) >= 0.6 and np.mean(xc > 0) >= 0.6
                xs_meds.append(ms); xc_meds.append(mc)
            res[f"{c * 100:.2f}%"] = {"n_ok": int(ok.sum()), "xs_med_of_med%": round(pct(np.median(xs_meds)), 2),
                                      "xc_med_of_med%": round(pct(np.median(xc_meds)), 2)}
        # 왕복 1회당 초과(비용 0, 종목 중앙): (xs+xc)/(ent_sel+ent_conf)
        per_trade = []
        for j in idx:
            rows = ~sparse if eth[j] else np.ones(len(syms), bool)
            x = (R["xs"][rows, j] + R["xc"][rows, j]) / np.maximum(R["es"][rows, j] + R["ec"][rows, j], 1)
            per_trade.append(np.median(x))
        res["per_roundtrip_excess_med_bp"] = round(float(np.median(per_trade)) * 1e4, 2)
        res["entries_med"] = float(np.median([np.median(R["es"][:, j] + R["ec"][:, j]) for j in idx]))
        out[nm] = res
    (HERE / "results" / "xsec_breakeven_5m.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
