"""감사 보조: 확정 사건·가짜 대표의 오프셋 0·+1 창 글이 D+1 08:00 전에 처음 받혔는지(조회 수치 측정 시점) → results/audit_views_timing.csv"""
import sys, json
from pathlib import Path
sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/events/code")
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import event_study as es
RES = es.RESULTS
ev, _ = es.confirmed_only_events(json.loads((RES/"events.json").read_text("utf-8")), es.CONFIRMED_ASOF)
F, meta = es.load_features()
P = pd.read_parquet(es.EV/"snapshot_1023"/"results"/"views_posts.parquet")
P["T_first"] = pd.to_datetime(P["T_first"])
panels = {m: es.build_panel(F, meta, m) for m in ("kr","us")}
out = []
for c in es.load_contrasts(ev):
    p = panels[c.market]
    wcol = "win_kr" if c.market == "kr" else "win_u"
    for grp, df in (("사건", c.events), ("가짜", c.fake_reps)):
        pos = es.positions(p, df["win_id"])
        for d, ps in zip(df["date"], pos):
            if ps < 0: continue
            cut = pd.Timestamp(d) + pd.Timedelta(days=1, hours=8)
            for k in (0, 1):
                if ps + k >= len(p.win_id): continue
                w = p.win_id[ps + k]
                g = P[P[wcol] == w]
                if not len(g): continue
                out.append({"contrast": c.name, "grp": grp, "date": d, "k": k, "win": w, "n_posts": len(g),
                            "share_first_fetch_le_cut": float((g["T_first"] <= cut).mean()),
                            "median_age_first_d": float(np.median(g["age_h_first"])/24)})
o = pd.DataFrame(out)
print(len(o))
print(o.groupby(["grp","k"])[["share_first_fetch_le_cut","median_age_first_d"]].describe().T.round(3).to_string())
print("any window with >50% fetched before cut:", int((o["share_first_fetch_le_cut"]>0.5).sum()), "of", len(o))
print(o[o["share_first_fetch_le_cut"]>0].head(20).to_string())
# 고유 창
u = o.drop_duplicates("win")
print("unique windows", len(u), "median share", u["share_first_fetch_le_cut"].median(), "max", u["share_first_fetch_le_cut"].max(), "median age d", u["median_age_first_d"].median(), "min age d", u["median_age_first_d"].min())
o.to_csv(RES/"audit_views_timing.csv", index=False, encoding="utf-8-sig")
