"""Lock on S, assess real gates once, then rerun the entire pipeline on joint permutations."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models import BUNDLES, HYPERS, INTERACTIONS, OUT, SEED, SOURCE, TAILS, TARGETS, Design
from models import candidate_mask, digest, panel, select_models


RUNS = 100
RESULT = OUT / "results.json"


def timestamp():
    return datetime.now().astimezone().isoformat()


def read_result():
    return json.loads(RESULT.read_text(encoding="utf-8"))


def save(data):
    stage = RESULT.with_suffix(".tmp")
    stage.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    stage.replace(RESULT)


def lock():
    design = Design(panel(train_only=True))
    frozen, _ = select_models(design, design.y)
    config = dict(bundles=BUNDLES, models=HYPERS, tails=TAILS, targets=TARGETS, interactions=INTERACTIONS,
                  seed=SEED, null_runs=RUNS, fit=design.scale_sources,
                  implementation_sha256=hashlib.sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest())
    token = digest({"config": config, "candidates": frozen})
    if RESULT.exists():
        assert read_result()["lock_sha256"] == token
        return
    obj = dict(locked_at=timestamp(), lock_sha256=token, configuration=config, locked=frozen,
               panel_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    save(obj)
    print("Locked", token, "selection_pass", sum(r["selection_pass"] for r in frozen.values()), flush=True)


def stats(frame, mask, returns, sign):
    qq, sect = returns.T
    qgood = mask & np.isfinite(qq)
    sgood = qgood & np.isfinite(sect)
    n, n_s = int(qgood.sum()), int(sgood.sum())
    wins, wins_s = int((sign*qq[qgood] > 0).sum()), int((sign*sect[sgood] > 0).sum())
    return dict(signals=int(mask.sum()), n=n, wins=wins, rate=wins/n if n else None,
                sector_n=n_s, sector_wins=wins_s, sector_rate=wins_s/n_s if n_s else None,
                tickers=int(frame.ticker[qgood].nunique()), fallback_qqq=int((qgood & frame.etf.eq("QQQ")).sum()),
                mean=float(qq[qgood].mean()) if n else None, median=float(np.median(qq[qgood])) if n else None)


def check_s(q1, q2):
    w1,n1=q1["wins"],q1["n"]
    w2,n2=q2["wins"],q2["n"]
    return n1>=20 and n2>=20 and n1+n2>=60 and 100*w1>=65*n1 and 100*w2>=65*n2 and 10*(w1+w2)>=7*(n1+n2)


def check_a(a):
    return a["n"]>=40 and 10*a["wins"]>=7*a["n"] and a["sector_n"]>=40 and 10*a["sector_wins"]>=6*a["sector_n"]


def audit_real(design, frozen):
    collected = {}
    for target, j, bundle, hyper, scores in design.model_scores(design.y):
        for side in ("up", "down"):
            key = target+"_"+side
            item = frozen[key]
            if (bundle, hyper) != (item["bundle"], item["model"]):
                continue
            actual = hashlib.sha256(np.round(scores[design.s], 10).astype("<f8").tobytes()).hexdigest()
            assert actual == item["score_sha256"], (key, "model score hash")
            ready = design.ready6 if target.startswith("e6") else np.ones(len(scores), bool)
            sign = 1 if side == "up" else -1
            hit = candidate_mask(scores, ready, item["cut"], side)
            # Benchmark agreement is measured on the same A cases.
            parts = dict(design.part)
            parts.update(Q1=design.s & (design.q==1), Q2=design.s & (design.q==2))
            parts["A"] = parts["A"] & np.isfinite(design.y[:,j,1])
            ss = {name:stats(design.df, hit & m, design.y[:,j], sign) for name,m in parts.items()}
            passed_s = check_s(ss["Q1"],ss["Q2"])
            passed_a = check_a(ss["A"])
            b = ss["B"]
            fund_status = "unconfirmed" if b["n"]<10 else ("pass" if 100*b["wins"]>=65*b["n"] else "fail")
            collected[key] = (dict(stats=ss, pass_S=passed_s, pass_A=passed_a,
                                   pass_SA=passed_s and passed_a, B_status=fund_status,
                                   pass_SAB=passed_s and passed_a and fund_status=="pass"),
                              hit & parts["A"] & np.isfinite(design.y[:,j,0]))
    assert len(collected)==12
    return collected


def cluster(masks):
    components = []
    for key, mask in masks.items():
        known = [i for i, group in enumerate(components) if any(
            2*np.count_nonzero(mask & masks[other]) >= min(int(mask.sum()),int(masks[other].sum())) for other in group)]
        group = [key]
        for i in reversed(known):
            group.extend(components.pop(i))
        components.append(sorted(group))
    return components


def confirm():
    obj = read_result()
    assert "observed" not in obj, "Real gates have already been recorded."
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==obj["panel_sha256"]
    design = Design(panel())
    audited = audit_real(design, obj["locked"])
    obs = {k:v[0] for k,v in audited.items()}
    clusters = {phase:cluster({k:m for k,(r,m) in audited.items() if r[phase]})
                for phase in ("pass_SA","pass_SAB")}
    bases = {}
    for j,t in enumerate(TARGETS):
        bases[t] = {part:stats(design.df,m,design.y[:,j],1) for part,m in design.part.items()}
    obj.update(observed=obs, observed_clusters=clusters, evaluated_at=timestamp(), baseline=bases,
               panel_counts={p:int(m.sum()) for p,m in design.part.items()},
               n_rows=len(design.df), n_s_train=int(design.train.sum()))
    save(obj)
    flat = []
    for key,item in obj["locked"].items():
        row = {"candidate":key, **item}
        row.update({k:v for k,v in obs[key].items() if k!="stats"})
        for band, values in obs[key]["stats"].items():
            row.update({band+"_"+k:v for k,v in values.items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUT/"candidates.csv", index=False)
    print("Observed",sum(v["pass_S"] for v in obs.values()),clusters,flush=True)


def shuffle_groups(design):
    buckets = {}
    valid = np.isfinite(design.y)
    for part in ("S","A"):
        for i in np.flatnonzero(design.part[part]):
            signature = (part, int(design.q[i]), valid[i].tobytes())
            if part=="S":
                signature += (bool(design.train[i]), tuple(bool(design.mature[t][i]) for t in TARGETS))
            buckets.setdefault(signature,[]).append(int(i))
    return [np.array(rows, dtype=int) for rows in buckets.values()]


def simulate(design, groups, index):
    rng = np.random.default_rng(np.random.SeedSequence([SEED,index]))
    y = design.y.copy()
    for rows in groups:
        y[rows] = design.y[rng.permutation(rows)]
    frozen, scores = select_models(design,y)
    pass_s,pass_sa={},{}
    for key,r in frozen.items():
        j=TARGETS.index(r["target"])
        ready=design.ready6 if r["target"].startswith("e6") else np.ones(len(design.df),bool)
        signal=candidate_mask(scores[key],ready,r["cut"],r["side"])
        signed=y[:,j,:]*(1 if r["side"]=="up" else -1)
        def split_counts(m):
            valid=m & np.isfinite(signed[:,0])
            return dict(n=int(valid.sum()),wins=int((signed[valid,0]>0).sum()))
        sm=signal & design.s
        pass_s[key]=check_s(split_counts(sm & (design.q==1)), split_counts(sm & (design.q==2)))
        if not pass_s[key]:
            continue
        am=signal & design.part["A"] & np.isfinite(signed).all(axis=1)
        a=split_counts(am)
        a.update(sector_n=a["n"], sector_wins=int((signed[am,1]>0).sum()))
        if check_a(a):
            pass_sa[key]=am
    return dict(trial=index, lock=digest(frozen), S=[k for k,v in pass_s.items() if v],
                SA=list(pass_sa), clusters=cluster(pass_sa))


def null():
    obj=read_result()
    assert "observed" in obj
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==obj["panel_sha256"]
    design=Design(panel())
    shuffle=shuffle_groups(design)
    runs=obj.setdefault("null_trials",[])
    for i in range(len(runs),RUNS):
        runs.append(simulate(design,shuffle,i))
        if len(runs)%10==0:
            save(obj)
            print("Completed simulated locks",len(runs),flush=True)
    observed=len(obj["observed_clusters"]["pass_SA"])
    counts=np.array([len(r["clusters"]) for r in runs])
    obj["null_summary"]=dict(runs=RUNS,observed_SA_clusters=observed,
                             p95=float(np.quantile(counts,.95)), exceed_or_equal=int((counts>=observed).sum()),
                             add_one_p=float(((counts>=observed).sum()+1)/(RUNS+1)),
                             trials_with_any=int((counts>0).sum()),
                             counts={str(int(n)):int((counts==n).sum()) for n in np.unique(counts)},
                             family_SA_trials={key:sum(key in r["SA"] for r in runs) for key in obj["locked"]})
    save(obj)
    print(obj["null_summary"])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        {"lock":lock,"confirm":confirm,"null":null}[sys.argv[1]]()

