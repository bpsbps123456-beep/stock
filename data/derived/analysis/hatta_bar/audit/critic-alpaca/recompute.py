"""원봉에서 다시 계산(탐색용, 사전 등록 아님):
 1) 변동성 맞춤 기준선 분해 — 같은 날·시간대·ATR 3분위 묶음 봉을 신호 시각 앞(창 끝남)/앞(창 겹침)/자기/뒤로 나눈 평균 R1(E2)
 2) 과거 날만 쓰는 변동성 맞춤(직전 20거래일·같은 시간대·ATR% ±20%) — 오늘 경로에 조건 걸지 않는 대안
 3) 엄격 체결 R1(E2): 목표가를 1틱(0.01$) 또는 0.1% 넘겨 거래돼야 체결
 4) 진입·청산 봉의 실제 시간대(비용 민감도용), 진입 봉 거래대금
 5) 무조건 기준: 판정 모집단(주문 가능·위생) 모든 유효 봉 평균 R1(E2)
 6) 종목 유동성: 앞 구간(7~8월) 정규장 분당 거래대금 중앙값(뒤 구간 신호엔 사전 값)"""
import time
from common import *
from build import SYMS


def strict_r1(a, zz, t, delta_abs=0.0, delta_rel=0.0):
    e = t + 1
    z = zz[t]
    if z < 0 or e + 1 > z:
        return np.nan
    en = a["c"][e]
    up = en * 1.01
    need = max(up + delta_abs, en * (1.01 + delta_rel))
    for j in range(e + 1, z + 1):
        if a["h"][j] >= need:
            return 0.01
    return a["c"][z] / en - 1.0


def main():
    sigs = []
    for nm in CELLS:
        s = pd.read_csv(os.path.join(RES, f"signals_{nm}.csv"), usecols=["sid", "t", "ts", "day", "sess", "r1_e2", "bv_r1_e2"])
        s["cell"] = nm
        sigs.append(s)
    sigs = pd.concat(sigs, ignore_index=True)
    rows, uncond, liq = [], [], {}
    t0 = time.time()
    for sid, sym in enumerate(SYMS):
        d = E.load(sym, "alp1m")
        if d is None or len(d) < 500:
            continue
        a = E.prep(d)
        out = E.bar_outcomes(a)
        zz = out["zz"]
        atrp = a["atr"] / a["c"]
        df = pd.DataFrame({"day": a["day"], "sess": a["sess"], "atrp": atrp})
        pct = df.groupby(["day", "sess"])["atrp"].rank(pct=True)
        ter = np.minimum(np.floor(pct * 3), 2).to_numpy()
        v2 = out["valid_e2"]
        r1 = np.where(v2, out["r1_e2"], np.nan)
        inper = (a["day"] >= 20260701) & (a["day"] <= 20261028)
        # 5) 무조건 기준(판정 모집단 봉 전체: 주문 가능·위생·밀도·유효)
        m = inper & a["tradable"] & a["sane"] & (a["dens"] >= E.MIN_DENS) & v2 & out["valid_e0"]
        uncond.append(pd.DataFrame({"sid": sid, "day": a["day"][m], "sess": a["sess"][m], "r1_e2": r1[m],
                                    "r1net_e2": out["r1net_e2"][m], "ter": ter[m]}))
        # 6) 유동성(앞 구간 정규장)
        reg = (a["mins"] >= 570) & (a["mins"] < 960) & (a["day"] >= 20260701) & (a["day"] <= 20260831)
        liq[sid] = float(np.median((a["v"] * a["c"])[reg])) if reg.any() else np.nan
        S = sigs[sigs.sid == sid]
        if len(S) == 0:
            continue
        days_u = np.unique(a["day"])
        for _, r in S.iterrows():
            t = int(r.t)
            ts_s = a["ts"][t]
            assert ts_s == r.ts
            g = (a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]) & (ter == ter[t]) & v2
            tsb = a["ts"]
            parts = {"pre_done": g & (tsb + 30 < ts_s), "pre_overlap": g & (tsb >= ts_s - 30) & (tsb < ts_s),
                     "self": g & (tsb == ts_s), "post": g & (tsb > ts_s)}
            rec = dict(cell=r.cell, sid=sid, t=t, ts=ts_s, day=int(r.day), sess=int(a["sess"][t]), ter=float(ter[t]),
                       r1_e2=r.r1_e2, bv_csv=r.bv_r1_e2, bv_re=float(np.nanmean(r1[g])) if g.any() else np.nan)
            for k, mk in parts.items():
                rec[f"bv_{k}"] = float(np.nanmean(r1[mk])) if mk.any() else np.nan
                rec[f"bvn_{k}"] = int(mk.sum())
            # 2) 과거 20거래일 같은 시간대 ATR% ±20%
            di = np.searchsorted(days_u, a["day"][t])
            past = days_u[max(0, di - 20):di]
            ap = atrp[t]
            pm = np.isin(a["day"], past) & (a["sess"] == a["sess"][t]) & v2 & (atrp >= ap / 1.2) & (atrp <= ap * 1.2)
            rec["pv_r1"] = float(np.nanmean(r1[pm])) if pm.sum() >= 20 else np.nan
            rec["pv_n"] = int(pm.sum())
            pm2 = np.isin(a["day"], past) & (a["sess"] == a["sess"][t]) & v2
            rec["ps_r1"] = float(np.nanmean(r1[pm2])) if pm2.sum() >= 20 else np.nan
            # 3) 엄격 체결
            rec["r1_tick"] = strict_r1(a, zz, t, delta_abs=0.01)
            rec["r1_011"] = strict_r1(a, zz, t, delta_rel=0.001)
            # 4) 진입·청산 시간대, 가격
            e, z = t + 1, zz[t]
            rec["sess_e"], rec["sess_z"] = int(a["sess"][e]), int(a["sess"][z])
            rec["hit"] = float(out["hit_e2"][t])
            rec["price"] = float(a["c"][t])
            rec["notional_e"] = float(a["v"][e] * a["c"][e])
            rows.append(rec)
        print(sym, len(S), f"{time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R["liq"] = R.sid.map(liq)
    R.to_csv(os.path.join(OUT, "recompute_signals.csv"), index=False)
    U = pd.concat(uncond, ignore_index=True)
    U.to_parquet(os.path.join(OUT, "uncond_bars.parquet"))
    json.dump({str(k): v for k, v in liq.items()}, open(os.path.join(OUT, "liq_front.json"), "w"))
    print("done", len(R), len(U))


if __name__ == "__main__":
    main()
