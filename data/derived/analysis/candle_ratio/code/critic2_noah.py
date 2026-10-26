"""critic2: 애프터마켓(16:00~20:00 ET) 체결이 안 된다고 볼 때(desk_system §1 '애프터마켓 체결 확인하지 않음') 원안 실제 판.
깨어 있음 = awake_mask & 시작 < 16:00 ET. 편도 0.1%. 비교: 기존 awake. 5분봉 창 전체 구간(고르기+확인)."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from engine import CACHE, HERE, WIN, awake_mask, load, ratio, segments, sim, tf_bars
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))

def one(sym):
    tb = tf_bars(load(sym, "5m"), 40, "eth"); tb = tb[tb.day >= WIN["5m"]["warm"]].reset_index(drop=True)
    O, C, V = (tb[k].values.astype(np.float64) for k in ("O", "C", "V"))
    seg = segments(tb.day.values, WIN["5m"]); p = ratio(O, C, V, 10, "count")
    st = pd.DatetimeIndex(tb.start); aw = awake_mask(tb.start); noah = aw & np.asarray(st.hour < 16)
    out = np.zeros(11); r = {"sym": sym}
    for nm, m in (("awake", aw), ("noah", noah)):
        sim(O, C, p, tb.lod.values, seg, m, 0.5, 0.5, True, False, 0.001, out)
        r[nm] = np.exp(out[0] + out[1]) - 1; r[nm + "_mdd"] = np.exp(out[10]) - 1
        r[nm + "_x"] = (out[4] + out[5]) / (out[6] + out[7]); r[nm + "_conf"] = np.exp(out[1]) - 1
    return r

if __name__ == "__main__":
    with Pool(3) as pool: df = pd.DataFrame(pool.map(one, SYMS))
    df.to_csv(HERE / "results" / "critic2_noah.csv", index=False)
    print((df.drop(columns="sym").median() * 100).round(1).to_dict())
    print(df[df.sym.isin(["MU", "BE", "NBIS"])].set_index("sym").mul(100).round(1).to_string())
