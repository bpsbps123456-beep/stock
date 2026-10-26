"""반박: '시장 공통 반전'을 실시간 거름망으로 쓸 수 있나.
매수 확정 날 기준, 직전 3거래일(그날 포함) 안에 같은 설정으로 매수 확정이 난 다른 주식 비율(그날 종가에 알 수 있음)로
확정 사건을 셋으로 나눠 먹은 몫(capt, cut)을 비교한다."""
import numpy as np, pandas as pd
import swing
ETF = {"SPY", "QQQ", "SOXX", "SMH", "IGV", "IWM", "XBI", "ARKG"}
e = pd.read_csv(swing.HERE / "results" / "capture_confirm_events.csv")
e = e[(e.ver == "cut") & (~e.sym.isin(ETF))].copy()
e["date"] = pd.to_datetime(e["date"])
cal = pd.to_datetime(pd.Series(sorted(swing.load()["SPY"].date.unique())))
pos = pd.Series(np.arange(len(cal)), index=cal)
e["di"] = e.date.map(pos)
out = []
for (cfg, per), g in e.groupby(["cfg", "per"]):
    nsym = g.sym.nunique()
    for typ in (1, -1):
        b = g[g.typ == typ]
        cnt = b.groupby("di").sym.nunique()
        full = cnt.reindex(range(int(b.di.min()) - 3, int(b.di.max()) + 1), fill_value=0)
        tr = full.rolling(3).sum()
        br = (b.di.map(tr) - 1) / (nsym - 1)
        b = b.assign(br=br)
        q = pd.qcut(b.br.rank(method="first"), 3, labels=["low", "mid", "high"])
        s = b.groupby(q, observed=True).capt.agg(["mean", "median", "size"])
        out.append(dict(cfg=cfg, per=per, typ=typ, **{f"{k}_{m}": s.loc[k, m] for k in s.index for m in ("mean", "median", "size")},
                        br_hi=b.br[q == "high"].median(), br_lo=b.br[q == "low"].median()))
r = pd.DataFrame(out)
r.to_csv(swing.HERE / "results" / "critic_breadth.csv", index=False)
x = r[r.cfg.isin(["ZZA_k3.0", "ZZP_th0.15", "ZZP_th0.07"])]
pd.set_option("display.width", 250)
print(x[["cfg", "per", "typ", "br_lo", "br_hi", "low_mean", "mid_mean", "high_mean", "low_median", "high_median", "high_size"]].round(3).to_string())
r["hi_minus_lo"] = r.high_mean - r.low_mean
print(r.groupby(["per", "typ"]).hi_minus_lo.agg(["median", lambda v: (v > 0).sum(), "size"]).round(3))
