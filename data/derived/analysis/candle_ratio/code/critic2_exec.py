"""critic2: 실행 관점 검산. 원안(40분·eth·N10·count·follow·b0.5·s0.5·밤 보유) 실제 판의
(1) 체결 시간대 분포, (2) 시간대별 5분봉 호가폭 근사(Abdi-Ranaldo, 고저폭), 봉 결측률,
(3) 시간대별 비용을 넣은 재계산(sim과 0.1% 일치 확인 후), (4) 표준시(11/1~) 깨어 있음으로 다시 잼.
결과: results/critic2_exec_*.csv
"""
import sys
import numpy as np
import pandas as pd
from engine import CACHE, HERE, WIN, awake_mask, load, positions, ratio, segments, sim, tf_bars

OUT = HERE / "results"
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
FOCUS = ["MU", "BE", "NBIS"]


def part_of(mins):
    return np.where(mins < 570, "pre", np.where(mins < 960, "rth", "post"))


def spread_stats(d):
    """5분봉 호가폭 근사. AR: s^2 = 4 E[(c_t-eta_t)(c_t-eta_{t+1})], 같은 세션 부분 안 연속 봉만."""
    d = d[(d.index.date >= WIN["5m"]["warm"])]
    m = np.asarray(d.index.hour * 60 + d.index.minute)
    keep = (m >= 240) & (m < 1200)
    d = d[keep]; m = m[keep]
    pt = part_of(m)
    # 깨어 있는 체결 시간대 세분: pre_early 04:00-07:00, pre_late 07:00-09:30, rth_open 09:30-11:00, post_wake 19:00-20:00
    sub = np.select([m < 420, m < 570, m < 660, m < 960, m < 1140], ["pre_e", "pre_l", "rth_o", "rth_x", "post_x"], "post_w")
    lc = np.log(d.Close.values); eta = (np.log(d.High.values) + np.log(d.Low.values)) / 2
    hl = (d.High.values - d.Low.values) / ((d.High.values + d.Low.values) / 2)
    nxt_ok = np.r_[(np.diff(m) == 5) & (np.asarray(d.index.date[1:]) == np.asarray(d.index.date[:-1])), False]
    prod = np.full(len(d), np.nan)
    prod[:-1] = (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    prod[~nxt_ok] = np.nan
    flat = d.High.values <= d.Low.values
    days = len(np.unique(d.index.date))
    res = {}
    for s, nslots in (("pre_e", 36), ("pre_l", 30), ("rth_o", 18), ("rth_x", 60), ("post_x", 36), ("post_w", 12)):
        k = sub == s
        x = prod[k]; x = x[~np.isnan(x)]
        ar2 = 4 * x.mean() if len(x) else np.nan
        res[s] = dict(ar=np.sqrt(max(ar2, 0)) if not np.isnan(ar2) else np.nan,
                      hl_med=float(np.median(hl[k])) if k.any() else np.nan,
                      flat=float(flat[k].mean()) if k.any() else np.nan,
                      present=float(k.sum() / (days * nslots)))
    return res


def run_sym(sym):
    d5 = load(sym, "5m")
    tb = tf_bars(d5, 40, "eth")
    tb = tb[tb.day >= WIN["5m"]["warm"]].reset_index(drop=True)
    O, C, V = (tb[k].values.astype(np.float64) for k in ("O", "C", "V"))
    lod = tb.lod.values
    seg = segments(tb.day.values, WIN["5m"])
    p = ratio(O, C, V, 10, "count")
    st = pd.DatetimeIndex(tb.start)
    aw = awake_mask(tb.start)
    # 표준시 깨어 있음: ET+14h 기준 KST 시각 >= 8 (같은 ET 시각에 1시간 더함)
    kst_std = (st.hour + 14) % 24
    aw_std = np.asarray(kst_std >= 8)
    mins = np.asarray(st.hour * 60 + st.minute)
    out = np.zeros(11)
    sim(O, C, p, lod, seg, aw, 0.5, 0.5, True, False, 0.001, out)
    ref = out[0] + out[1]
    sim(O, C, p, lod, seg, aw_std, 0.5, 0.5, True, False, 0.001, out)
    std = out[0] + out[1]; std_hold = (out[4] + out[5]) / (out[6] + out[7])
    pos, ev = positions(O, C, p, lod, aw, 0.5, 0.5, True, False)
    rb = np.r_[np.log(O[1:] / O[:-1]), np.log(C[-1] / O[-1])]
    m = seg >= 0
    chg = np.r_[pos[0] != 0, pos[1:] != pos[:-1]]
    gross = float((pos * rb)[m].sum())
    fills = np.flatnonzero(chg & m)
    my = gross + len(fills) * np.log(1 - 0.001)
    fm = mins[fills]
    sub = np.select([fm < 420, fm < 570, fm < 660, fm < 960, fm < 1140], ["pre_e", "pre_l", "rth_o", "rth_x", "post_x"], "post_w")
    # 09:20 묶음봉은 프리장 시가(09:20)로 체결 → pre_l로 셈(위 select가 그렇게 함)
    sp = spread_stats(d5)
    row = dict(sym=sym, ref=ref, mine=my, gross=gross, n_fill=len(fills), std_ret=std, std_hold=std_hold)
    for s in ("pre_e", "pre_l", "rth_o", "rth_x", "post_x", "post_w"):
        row[f"n_{s}"] = int((sub == s).sum())
        for kk, v in sp[s].items():
            row[f"{kk}_{s}"] = v
    # 시간대별 비용: 편도 = max(0.05%, AR 호가폭/2), 정규장은 0.05% 하한
    c = np.array([max(0.0005, (sp[s]["ar"] if not np.isnan(sp[s]["ar"]) else 0.002) / 2) for s in sub])
    row["ret_arcost"] = gross + float(np.log(1 - c).sum())
    row["mean_cost_ar"] = float(c.mean()) if len(c) else np.nan
    # 같은 비중 그냥 보유(단순), x = 보유 비율
    x = pos[m].mean()
    px = np.r_[O, C[-1]]; w = np.flatnonzero(m); a, b = w[0], w[-1]
    row["x"] = x
    row["static"] = x * (px[b + 1] / px[a] * (1 - 0.001) - 1)
    return row


if __name__ == "__main__":
    from multiprocessing import Pool
    syms = SYMS if len(sys.argv) < 2 else sys.argv[1:]
    with Pool(3) as pool:
        rows = pool.map(run_sym, syms)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "critic2_exec_syms.csv", index=False)
    pd.set_option("display.width", 250)
    print("sim 일치 최대 오차", float((df.ref - df.mine).abs().max()))
    pct = lambda v: (np.exp(v) - 1) * 100
    for s in FOCUS:
        r = df[df.sym == s].iloc[0]
        print(s, "0.1%:", round(pct(r.ref), 1), "AR비용:", round(pct(r.ret_arcost), 1), "평균 편도", round(r.mean_cost_ar * 100, 3),
              "체결", r.n_fill, "표준시 깨어있음:", round(pct(r.std_ret), 1), "static", round(r.static * 100, 1))
        print("   ", {k: (int(r[f'n_{k}']), round(r[f'ar_{k}'] * 100, 3), round(r[f'hl_med_{k}'] * 100, 3), round(r[f'present_{k}'], 2), round(r[f'flat_{k}'], 2)) for k in ("pre_e", "pre_l", "rth_o", "rth_x", "post_x", "post_w")})
    print("144 중앙 0.1%", round(pct(df.ref).median(), 1), "AR비용", round(pct(df.ret_arcost).median(), 1), "표준시", round(pct(df.std_ret).median(), 1),
          "static", round(df.static.median() * 100, 1), "평균 편도 중앙", round(df.mean_cost_ar.median() * 100, 3))
    print("체결 시간대 합", {k: int(df[f'n_{k}'].sum()) for k in ("pre_e", "pre_l", "rth_o", "rth_x", "post_x", "post_w")})
    print("AR 호가폭 중앙(%)", {k: round(df[f'ar_{k}'].median() * 100, 3) for k in ("pre_e", "pre_l", "rth_o", "rth_x", "post_x", "post_w")})
    print("고저폭 중앙(%)", {k: round(df[f'hl_med_{k}'].median() * 100, 3) for k in ("pre_e", "pre_l", "rth_o", "rth_x", "post_x", "post_w")})
    print("봉 있음 중앙", {k: round(df[f'present_{k}'].median(), 2) for k in ("pre_e", "pre_l", "rth_o", "rth_x", "post_x", "post_w")})
    print("원안>static 종목 비율 0.1%:", round(((np.exp(df.ref) - 1) > df.static).mean(), 3), "AR비용:", round(((np.exp(df.ret_arcost) - 1) > df.static).mean(), 3))
