"""자료 대조: 9/30~10/28 신호 봉을 야후 같은 분 봉과 맞춰 본다(저가 차이 ≤ max(0.15%, 0.1×ATR14)이면 '야후 확인').
그리고 같은 기간 전체 봉의 일치율(봉 있음·저가·고가). 사용: py xcheck.py → results/xcheck.json"""
import os, json
import numpy as np
import pandas as pd
import engine as E
from build import SYMS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
A0, B0 = pd.Timestamp("2026-09-30", tz=E.ET), pd.Timestamp("2026-10-29", tz=E.ET)


def main():
    bars = {}
    tot = {"alp_bars": 0, "both": 0, "low_ok": 0, "high_ok": 0}
    for sym in SYMS:
        a, y = E.load(sym, "alp1m"), E.load(sym, "yf1m")
        if a is None or y is None:
            continue
        a = a.assign(ATR=E._rma_atr(a.High.to_numpy(), a.Low.to_numpy(), a.Close.to_numpy(), 14))
        a, y = a[(a.index >= A0) & (a.index < B0)], y[(y.index >= A0) & (y.index < B0)]
        j = a.join(y, how="left", rsuffix="_y")
        tot["alp_bars"] += len(j)
        both = j.Low_y.notna()
        tot["both"] += int(both.sum())
        tol = np.maximum(0.0015 * j.Close, 0.1 * j.ATR)
        tot["low_ok"] += int(((j.Low - j.Low_y).abs() <= tol)[both].sum())
        tot["high_ok"] += int(((j.High - j.High_y).abs() <= tol)[both].sum())
        bars[sym] = (a, y)
    res = {"bars": tot}
    for f in sorted(os.listdir(RES)):
        if not (f.startswith("signals_") and f.endswith(".csv")):
            continue
        s = pd.read_csv(os.path.join(RES, f))
        s["time"] = pd.to_datetime(s.ts * 60, unit="s", utc=True).dt.tz_convert(E.ET)
        s = s[(s.time >= A0) & (s.time < B0)].copy()
        ok = []
        for r in s.itertuples():
            sym = SYMS[int(r.sid)]
            if sym not in bars:
                ok.append(np.nan)
                continue
            a, y = bars[sym]
            # ①은 신호 봉 = 이탈 봉, ②는 이탈 봉 = 신호 봉 − 1분 앞 봉
            k_time = r.time if f.startswith("signals_orig_A") or f.startswith("signals_top_A") else a.index[a.index.get_loc(r.time) - 1]
            if k_time not in y.index:
                ok.append(False)
                continue
            lo_a, lo_y = a.at[k_time, "Low"], y.at[k_time, "Low"]
            ok.append(abs(lo_a - lo_y) <= max(0.0015 * a.at[k_time, "Close"], 0.1 * a.at[k_time, "ATR"]))
        s["yf_ok"] = ok
        conf = s[s.yf_ok == True]  # noqa: E712
        res[f[8:-4]] = {"n": int(len(s)), "yf_ok_share": float(np.nanmean(s.yf_ok.astype(float))) if len(s) else None,
                        "r1net_all": float(s.r1net_e2.mean()) if len(s) else None,
                        "r1net_yf_ok": float(conf.r1net_e2.mean()) if len(conf) else None,
                        "hit_all": float(s.hit_e2.mean()) if len(s) else None,
                        "hit_yf_ok": float(conf.hit_e2.mean()) if len(conf) else None}
    json.dump(res, open(os.path.join(RES, "xcheck.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
