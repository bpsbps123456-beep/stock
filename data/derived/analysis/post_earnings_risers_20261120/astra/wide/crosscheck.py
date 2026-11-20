"""Reconstruct all peer A-pass rules after the independent ML report is written."""
import hashlib
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from models import OUT, SOURCE
from study import cluster


def verify():
    assert (OUT / "REPORT.md").is_file()
    p = pd.read_csv(SOURCE)
    dates = pd.to_datetime(p.reaction_date)
    yearhalf = dates.between("2026-01-01", "2026-06-30")
    gates = dates.between("2026-07-01", "2026-10-21")
    parts = {"S": p.src.eq("wide") & yearhalf,
             "A": p.src.eq("wide") & ~p.fund_ticker & gates,
             "B": p.src.eq("fund") & gates}
    parts["q1"] = parts["S"] & dates.dt.quarter.eq(1)
    parts["q2"] = parts["S"] & dates.dt.quarter.eq(2)
    labels = np.select([parts[s] for s in ("S","A","B")], ["S","A","B"], default="other")

    @lru_cache(None)
    def condition(token):
        field, op, word = re.fullmatch(r"(.+?)([=≤≥])(.+)", token).groups()
        raw = p[field]
        if op=="=":
            return raw.eq({"참":True, "거짓":False}.get(word,word))
        cut = raw[parts["S"]].quantile(.33333 if op=="≤" else .66667)
        assert f"{cut:.4g}"==word
        return raw.le(cut) if op=="≤" else raw.ge(cut)

    def signed_values(target,side,dm):
        y = p[[target, target+"_s"]].copy()
        if dm:
            group = parts["A"].to_numpy() if dm=="peer" and target!="e1_h5" else labels
            y -= y.groupby([group, dates]).transform("mean")
        return y.iloc[:,0], y.iloc[:,1], (1 if side=="up" else -1)

    def recount(row, dm):
        target, side = row.family.rsplit("_",1)
        y, ys, sign = signed_values(target,side,dm)
        mask = pd.Series(np.logical_and.reduce([condition(c).to_numpy() for c in row.rule.split(" & ")]))
        valid = mask & y.notna()
        out = dict()
        for g, m in parts.items():
            z = y[valid & m]
            out[g] = [int((sign*z>0).sum()), len(z)]
        a = valid & parts["A"]
        b = valid & parts["B"]
        out["A_sector"] = [int((sign*ys[a]>0).sum()),int(a.sum())]
        out.update(med_A=float(y[a].median()), med_B=float(y[b].median()) if b.any() else None,
                   tickers_A=int(p.ticker[a].nunique()), dates_A=int(dates[a].nunique()))
        # Maximum count in any fourteen consecutive calendar dates (inclusive).
        cells = dates[a].value_counts().sort_index()
        out["max14_A"] = int(cells.rolling("14D").sum().max())
        assert np.array_equal(np.flatnonzero(a), np.fromstring(row.members_A, sep=",", dtype=int))
        return out, a.to_numpy()

    def passes(rec):
        def ok(name, minimum, pct):
            w,n = rec[name]
            return n>=minimum and 100*w>=pct*n
        s=ok("S",60,70) and ok("q1",20,65) and ok("q2",20,65)
        a=ok("A",40,70) and ok("A_sector",40,60)
        b=ok("B",10,65)
        return (s and a, s and a and b)

    def matches(r, item):
        for name in ("S","A","B","q1","q2","A_sector"):
            w,n=item[name]
            if hasattr(r,"n_"+name) and n!=getattr(r,"n_"+name):
                return False
            if not np.isclose(w/n if n else np.nan, getattr(r,"hit_"+name), atol=.00051, rtol=0, equal_nan=True):
                return False
        return (r.tickers_A==item["tickers_A"] and
                np.isclose(r.med_A,item["med_A"],atol=.00501,rtol=0) and
                np.isclose(r.med_B,np.nan if item["med_B"] is None else item["med_B"],atol=.00501,rtol=0,equal_nan=True))

    output={}
    for tag in ("", "_datedm"):
        file=SOURCE.parent / ("rules_wide_pass"+tag+".csv")
        rows=pd.read_csv(file)
        out={"rows":len(rows), "valid_B":0,"small_B":0,"B_fails":0,"corrected_SA":0,
             "corrected_SAB":0,"peer_match":0,"intended_match":0,"by_family":{},
             "source_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"confirmed":[]}
        passed, both, corrected = {},{},{}
        for i,row in enumerate(rows.itertuples()):
            intended, mask = recount(row,"intended" if tag else None)
            published = intended
            if tag and not matches(row,intended):
                published, mask = recount(row,"peer")
            assert matches(row,published), (i,row.family,row.rule,published)
            out["peer_match"] += 1
            out["intended_match"] += int(matches(row,intended))
            sa,sab=passes(published)
            assert sa
            sa2,sab2=passes(intended)
            out["corrected_SA"]+=int(sa2)
            out["corrected_SAB"]+=int(sab2)
            out["by_family"][row.family]=1+out["by_family"].get(row.family,0)
            key=str(i)
            passed[key]=mask
            if sab:
                both[key]=mask
                out["valid_B"]+=1
                out["confirmed"].append({"family":row.family,"rule":row.rule,"published":published,"intended":intended})
            elif published["B"][1]<10:
                out["small_B"]+=1
            else:
                out["B_fails"]+=1
            if sab2:
                corrected[key]=mask
        out["SA_groups"]=len(cluster(passed))
        out["SAB_groups"]=len(cluster(both))
        out["corrected_SAB_groups"]=len(cluster(corrected))
        output[tag or "registered"]=out
    return output


if __name__ == "__main__":
    obj=verify()
    print(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

