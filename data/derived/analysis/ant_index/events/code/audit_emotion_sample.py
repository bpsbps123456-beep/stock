"""감사 보조: 감정 층 표본 중 바닥 사례(가격 사건 둘레 51일) 재사용 단위가 든 K창과 사건·가짜 창의 몫. 출력만 한다."""
import sys, json
sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/events/code")
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import event_study as es
RES = es.RESULTS
ev, _ = es.confirmed_only_events(json.loads((RES/"events.json").read_text("utf-8")), es.CONFIRMED_ASOF)
F, meta = es.load_features()
key = pd.read_parquet(es.EV/"label_key.parquet")
bc = key[key["src"]=="bottom_case"].groupby("win_id").size()
F["bc_units"] = F["win_id"].map(bc).fillna(0).astype(int)
print("windows with bottom_case reuse:", int((F["bc_units"]>0).sum()), "units", int(F["bc_units"].sum()))
print("em_n_units median kr:", F.loc[F.market=="kr","em_n_units"].median(), " with bc:", F.loc[(F.market=="kr")&(F.bc_units>0),"em_n_units"].median())
rows=[]
for c in es.load_contrasts(ev):
    g = F[F["market"]==c.market].sort_values("close_kst").reset_index(drop=True)
    pos = {w:i for i,w in enumerate(g["win_id"])}
    for grp, df in (("ev", c.events), ("fk", c.fake_reps)):
        for k in (0,1):
            ii=[pos[w]+k for w in df["win_id"] if w in pos and pos[w]+k < len(g)]
            sub=g.iloc[ii]
            rows.append({"contrast":c.name,"grp":grp,"k":k,"n":len(sub),"share_bc_windows":float((sub["bc_units"]>0).mean()) if len(sub) else np.nan,
                         "em_n_units_med":float(sub["em_n_units"].median()) if len(sub) else np.nan})
o=pd.DataFrame(rows)
print(o.pivot_table(index=["contrast","k"],columns="grp",values=["share_bc_windows","em_n_units_med"]).round(2).to_string())
