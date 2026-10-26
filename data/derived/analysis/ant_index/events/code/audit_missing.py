"""감사 보조: 사건·가짜 창의 지운 글 몫(missing_rate) 비교(소급 창, 오프셋 0·+1, 양측 Mann–Whitney). 출력만 한다."""
import sys, json
sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/events/code")
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import event_study as es
from scipy.stats import mannwhitneyu
RES = es.RESULTS
ev, _ = es.confirmed_only_events(json.loads((RES/"events.json").read_text("utf-8")), es.CONFIRMED_ASOF)
F, meta = es.load_features()
F = F.sort_values(["market","close_kst"]).reset_index(drop=True)
rows = []
for c in es.load_contrasts(ev):
    g = F[F["market"] == c.market].reset_index(drop=True)
    pos = {w:i for i,w in enumerate(g["win_id"])}
    for k in (0,1):
        def vals(df):
            out=[]
            for w in df["win_id"]:
                i = pos.get(w)
                if i is None or i+k >= len(g): continue
                r = g.iloc[i+k]
                if r["retro_window"] and r["judge_ok"]:
                    out.append(r["missing_rate"])
            return np.array(out, float)
        a, b = vals(c.events), vals(c.fake_reps)
        rows.append({"contrast": c.name, "k": k, "n_ev": len(a), "n_fk": len(b), "med_ev": np.nanmedian(a) if len(a) else np.nan,
                     "med_fk": np.nanmedian(b) if len(b) else np.nan,
                     "p_two": mannwhitneyu(a[np.isfinite(a)], b[np.isfinite(b)]).pvalue if len(a)>2 and len(b)>2 else np.nan})
o = pd.DataFrame(rows)
print(o.round(4).to_string())
allr = F[F["retro_window"] & F["judge_ok"]]
print("all retro windows median missing_rate", allr.groupby("market")["missing_rate"].median().round(4).to_dict())
