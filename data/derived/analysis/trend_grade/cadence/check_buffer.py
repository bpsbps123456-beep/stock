"""E1의 '이력 띠(S로 들어와 A까지 유지)' 대 'S만' 주간 결과를 독립 계산으로 확인(메인 세션, 2026-10-12)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import harness as H, fam_composite as FC, fam_ladder as FL
P = H.load_panel(); O, C = P["O"], P["C"]
um = H.universe_mask(P)
sc = FC.make(FC.GRID["E14"], 10)(P)
G = FL.hyb_grades(sc, P, um)
cal = C.index
def weeks(a, b):
    d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))]
    wk = pd.Series(d.to_period("W-SUN"), index=d)
    return [x for x, w in zip(d, wk) if x == wk[wk == w].index[0]]
def run(a, b, rule):
    fs = weeks(a, b); eq = 1.0; held = set(); curve = []; turn = 0
    for i in range(len(fs) - 1):
        w0, w1 = fs[i], fs[i + 1]
        prev = cal[cal.get_loc(w0) - 1]
        if prev < pd.Timestamp(a): continue
        g = G.loc[prev]; ok = um.loc[prev]
        if rule == "S": new = {s for s in g.index if g[s] == "S" and ok[s]}
        elif rule == "BUF": new = {s for s in g.index if ok[s] and (g[s] == "S" or (s in held and g[s] in ("S", "A")))}
        else: new = {s for s in g.index if ok[s] and isinstance(g[s], str)}
        turn += len(new ^ held) / max(len(new), 1); held = new
        names = [s for s in held if np.isfinite(O.at[w0, s]) and np.isfinite(O.at[w1, s])]
        if not names: continue
        span = cal[(cal >= w0) & (cal < w1)]
        for d in span: curve.append(eq * float((C.loc[d, names] / O.loc[w0, names]).mean()))
        eq *= float((O.loc[w1, names] / O.loc[w0, names]).mean())
    c = pd.Series(curve)
    return round((eq - 1) * 100, 1), round(float((c / c.cummax() - 1).min()) * 100, 1)
for name, (a, b) in {"front": ("2026-01-02", "2026-05-29"), "back": ("2026-06-01", "2026-10-09"),
                     "y2025": ("2025-01-02", "2025-12-31"), "y2024": ("2024-01-02", "2024-12-31")}.items():
    print(name, {r: run(a, b, r) for r in ("S", "BUF", "ALL")})
