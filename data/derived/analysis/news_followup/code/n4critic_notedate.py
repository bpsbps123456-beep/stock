"""N4 비판 검산(설명용 경계, 매매 불가): 노트 날짜 다음 정규장 시가로 진입하면(공개가 노트 날짜에 있었다는 낙관 가정) H1·H2·H4 판정이 바뀌는가.
published_at 보다 늦어지지 않게 min(노트 기준, 공개 기준)을 쓴다. 공개일 대리(publication_proxy) 행은 그대로다."""
import importlib.util, pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m", HERE / "n4h14_run.py"); m = importlib.util.module_from_spec(spec); sys.modules["m"] = m; spec.loader.exec_module(m)
O, C = m.load_prices(); sess = O.index
cov = pd.read_csv(m.CENSUS / "coverage.csv")
wk = cov.groupby("week_start").agg(pub=("groups_published", "sum"), unv=("unverified_candidates", "sum")); wk.index = pd.to_datetime(wk.index)
nw = pd.DataFrame(dict(nw_strict=(wk.pub == 0) & (wk.unv == 0), nw_any=(wk.pub == 0)))
info, arrs, pos0, D = m.build_daily(O, C, nw); ctl = m.control_means(info, arrs); cen = m.load_signals(O)
nd = [int(sess.searchsorted(pd.Timestamp(x), side="right")) for x in cen.note_date]
cen["nd_i"] = np.where(cen.full_i.notna(), np.minimum(cen.full_i.fillna(1e9), nd), np.nan)
print("앞당겨진 신호 수:", int((cen.nd_i < cen.full_i).sum()), "/", int(cen.full_i.notna().sum()),
      "· 평균 앞당김(세션):", round(float((cen.full_i - cen.nd_i)[cen.full_i.notna()].mean()), 2))
rng = np.random.default_rng(7); p0, p1 = m.period_bounds(info, "2023-01-01", "2026-10-23")
rows = []
for var in ("full_i", "nd_i"):
    pc = cen[var] - pos0; ok = pc.notna() & (pc >= 0) & (pc <= D - 1)
    sig = cen[ok].copy(); sig["pos"] = pc[ok].astype(int)
    for hyp, fam, hs in (("H1", "D", (5, 10, 20)), ("H2", "R", (5, 10, 20)), ("H4", "T", (20,))):
        g = sig[sig.family == fam]
        if hyp == "H1": g = g[g.lookback_months >= 12]
        for vname in ("all", "first"):
            pos = g.pos.to_numpy(int)
            if vname == "first": pos = pos[m.first_only(pos, g.group_id.to_numpy())]
            for h in hs:
                cm, cu, _ = ctl[("SOXX", h, "A")]
                r = m.mean_stats(pos, h, arrs[("SOXX", h)], cm, cu, (p0, p1), rng, 1); r.update(var=var, hyp=hyp, variant=vname, h=h); r["verdict"] = m.verdict(hyp, r); rows.append(r)
out = pd.DataFrame(rows)[["var", "hyp", "variant", "h", "n", "mean", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_hi", "p_lo", "verdict"]]
print(out.round(4).to_string(index=False)); out.to_csv(HERE.parent / "results" / "n4critic_notedate.csv", index=False, encoding="utf-8-sig")
