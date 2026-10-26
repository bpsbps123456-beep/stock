"""N4 비판 검산: '5거래일 안' 첫 신호 규칙을 H1-4(차 <= 5 제외)와 H5(차 < 5 제외) 중 어느 쪽으로 해도 H1·H2·H4 첫 신호 판정이 같은가."""
import importlib.util, pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m", HERE / "n4h14_run.py"); m = importlib.util.module_from_spec(spec); sys.modules["m"] = m; spec.loader.exec_module(m)
O, C = m.load_prices()
cov = pd.read_csv(m.CENSUS / "coverage.csv")
wk = cov.groupby("week_start").agg(pub=("groups_published", "sum"), unv=("unverified_candidates", "sum")); wk.index = pd.to_datetime(wk.index)
nw = pd.DataFrame(dict(nw_strict=(wk.pub == 0) & (wk.unv == 0), nw_any=(wk.pub == 0)))
info, arrs, pos0, D = m.build_daily(O, C, nw); ctl = m.control_means(info, arrs); cen = m.load_signals(O)
cen["pos"] = cen.full_i - pos0; sig = cen[cen.pos.notna() & (cen.pos >= 0) & (cen.pos <= D - 1)].copy(); sig["pos"] = sig.pos.astype(int)
rng = np.random.default_rng(7); p0, p1 = m.period_bounds(info, "2023-01-01", "2026-10-23")
rows = []
for chain in (5, 4):
    m.CHAIN = chain
    for hyp, fam, hs in (("H1", "D", (5, 10, 20)), ("H2", "R", (5, 10, 20)), ("H4", "T", (20,))):
        g = sig[sig.family == fam]
        if hyp == "H1": g = g[g.lookback_months >= 12]
        pos = g.pos.to_numpy(int); pos = pos[m.first_only(pos, g.group_id.to_numpy())]
        for h in hs:
            cm, cu, _ = ctl[("SOXX", h, "A")]
            r = m.mean_stats(pos, h, arrs[("SOXX", h)], cm, cu, (p0, p1), rng, 1); r.update(chain=chain, hyp=hyp, h=h); r["verdict"] = m.verdict(hyp, r); rows.append(r)
out = pd.DataFrame(rows)[["chain", "hyp", "h", "n", "ex_mean", "ex_lo", "ex_hi", "p_hi", "p_lo", "verdict"]]
print(out.round(4).to_string(index=False)); out.to_csv(HERE.parent / "results" / "n4critic_chain.csv", index=False, encoding="utf-8-sig")
