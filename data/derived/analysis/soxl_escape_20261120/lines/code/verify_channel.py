"""Claude가 Codex 후보 channel을 검증한다(2026-11-20).
정의: SOXL 연장 30분봉(04:00 기준). 직전 32봉 최저가에 봉 안 첫 1분봉 저가가 닿으면 min(시가, 선) 매도, 종가 > 직전 32봉 최고가면 봉 완성 뒤 첫 1분봉 시가 매수.
1) 엔진을 쓰지 않는 단순 반복문으로 다시 계산해 Codex 숫자와 비교 2) 이웃(N·M, 봉, 세션, 스탑↔종가) 3) 위약 백분위
결과: ../results/verify_channel.json"""
import json, os
import numpy as np, pandas as pd
import engine as E

OUT = os.path.join(E.ROOT, "results", "verify_channel.json")


def naive(n=32, m=32, tf=30):
    d = pd.read_parquet(os.path.join(E.ROOT, "cache", "SOXL_1m.parquet"))[["Open", "High", "Low", "Close"]].sort_index()
    mod = d.index.hour * 60 + d.index.minute
    binid = (d.index.normalize() + pd.to_timedelta(240 + ((mod - 240) // tf) * tf, unit="m"))
    g = d.groupby(binid)
    b = pd.DataFrame({"High": g["High"].max(), "Low": g["Low"].min(), "Close": g["Close"].last()})
    lower = b["Low"].rolling(n).min().shift(1).to_numpy()
    upper = b["High"].rolling(m).max().shift(1).to_numpy()
    codes, _ = pd.factorize(binid)                 # 1분봉 → 봉 번호(시간순)
    O, L = d["Open"].to_numpy(), d["Low"].to_numpy()
    Cb = b["Close"].to_numpy()
    first = np.r_[0, np.nonzero(np.diff(codes))[0] + 1]      # 각 봉의 첫 1분봉
    last = np.r_[first[1:] - 1, len(d) - 1]
    hold = np.ones(len(d))
    adj = np.zeros(len(d))
    state, start_from, nsw = True, 0, []
    for j in range(len(b)):
        a, z = first[j], last[j]
        if state:
            lv = lower[j]
            if lv == lv:
                hit = np.nonzero(L[max(a, start_from):z + 1] <= lv)[0]
                if len(hit):
                    p = max(a, start_from) + hit[0]
                    hold[p:] = 0
                    adj[p] += np.log(min(O[p], lv) / O[p])
                    state, start_from = False, p + 1
                    nsw.append(p)
        else:
            if upper[j] == upper[j] and Cb[j] > upper[j] and z + 1 < len(d):
                p = max(z + 1, start_from)
                hold[p:] = 1
                state, start_from = True, p + 1
                nsw.append(p)
    r = np.zeros(len(d)); r[:-1] = np.log(O[1:] / O[:-1])
    dates = pd.to_datetime(d.index.date)
    res = {}
    for w, (s, e) in E.WINDOWS.items():
        ii = np.nonzero((dates >= s) & (dates <= e))[0]
        a, z = ii[0], ii[-1]
        inc = hold[a:z] * r[a:z] + adj[a:z]
        f = hold[a:z].mean()
        eq = np.exp(np.cumsum(inc))
        res[w] = {"ret%": round(100 * np.expm1(inc.sum()), 2), "tv%p": round(100 * (inc.sum() - f * r[a:z].sum()), 2),
                  "mdd%": round(100 * float((eq / np.maximum.accumulate(eq) - 1).min()), 2), "expo%": round(100 * f, 1),
                  "switches": int(sum(a <= p < z for p in nsw))}
    return res


def eng(n, m, tf, sess, mode):
    b = E.bars("SOXL", tf, sess)
    lower = b.Low.rolling(n).min().shift(1)
    upper = b.High.rolling(m).max().shift(1)
    kw = dict(exit_touch=lower) if mode == "touch" else dict(exit_close=b.Close < lower)
    return E.simulate(b, **kw, entry_close=b.Close > upper, exec_sess=sess)


L = lambda x: 100 * np.log1p(x / 100)   # noqa: E731
out = {"naive": naive()}
print("naive", json.dumps({w: out["naive"][w] for w in ("SEL", "CONF", "Y2025")}))
pos, corr, sw = eng(32, 32, 30, "ext", "touch")
out["engine"] = {w: E.evaluate(pos, corr, sw, w) for w in E.WINDOWS}
out["placebo"] = {w: E.placebo(pos, corr, w) for w in ("SEL", "CONF", "Y2025")}
print("placebo", out["placebo"])
rows = []
for tf in (30, 60):
    for sess in ("ext", "reg"):
        for n in (16, 32, 64):
            for m in (16, 32, 64):
                for mode in ("touch", "close"):
                    p, c, s = eng(n, m, tf, sess, mode)
                    S, Cn, Y = (E.evaluate(p, c, s, w) for w in ("SEL", "CONF", "Y2025"))
                    rows.append(dict(tf=tf, sess=sess, n=n, m=m, mode=mode, S_sw=S["switches"], S_tvnet=round(S["tv%p"] - 0.1 * S["switches"], 1),
                                     C_tvL=round(L(Cn["ret%"]) - S["expo%"] / 100 * L(Cn["bh%"]) - 0.1 * Cn["switches"], 1),
                                     Y_tvnet=round(Y["tv%p"] - 0.1 * Y["switches"], 1), S_mdd=S["mdd%"], Y_ret=Y["ret%"]))
t = pd.DataFrame(rows)
out["neighbors"] = rows
out["neighbor_summary"] = {"n": len(t), "SEL_pos%": round(100 * float((t.S_tvnet > 0).mean()), 1), "CONF_tvL_pos%": round(100 * float((t.C_tvL > 0).mean()), 1),
                           "Y2025_pos%": round(100 * float((t.Y_tvnet > 0).mean()), 1),
                           "all3_pos%": round(100 * float(((t.S_tvnet > 0) & (t.C_tvL > 0) & (t.Y_tvnet > 0)).mean()), 1),
                           "median": t[["S_tvnet", "C_tvL", "Y_tvnet"]].median().round(1).to_dict()}
print(out["neighbor_summary"])
print(t[(t.tf == 30) & (t.sess == "ext")].to_string(index=False))
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
