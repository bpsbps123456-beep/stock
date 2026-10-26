"""N4 비판(시점) 검산 — 진입 규칙을 바꾸면 H1·H2·H4 판정이 바뀌는가.

변형(날짜만 있는 행만 바꾼다; exact 는 그대로)
  base  : 날짜 D 다음 정규장 시가(본 연구 규칙)
  same  : D 가 거래일이면 D 시가(장전 공개 가정, 장중·장후 공개면 사전 정보 섞임 = 낙관 경계)
  late1 : base 보다 1세션 뒤(아시아 재게재·피드 색인 지연 가정)
그리고 출처 시점 점검: 그룹 published_at 이 group_sources 최소 날짜와 같은지, 아시아 도메인 대표 출처 수.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m", HERE / "n4h14_run.py")
m = importlib.util.module_from_spec(spec)
sys.modules["m"] = m
spec.loader.exec_module(m)

O, C = m.load_prices()
cov = pd.read_csv(m.CENSUS / "coverage.csv")
wk = cov.groupby("week_start").agg(pub=("groups_published", "sum"), unv=("unverified_candidates", "sum"))
wk.index = pd.to_datetime(wk.index)
nw = pd.DataFrame(dict(nw_strict=(wk.pub == 0) & (wk.unv == 0), nw_any=(wk.pub == 0)))
info, arrs, pos0, D = m.build_daily(O, C, nw)
ctl = m.control_means(info, arrs)
cen = m.load_signals(O)
sess = O.index

# 변형 진입 위치
same = []
for _, r in cen.iterrows():
    if pd.isna(r.published_at) or r.time_precision == "exact":
        same.append(r.full_i)
    else:
        d = pd.Timestamp(str(r.published_at)[:10])
        same.append(int(sess.searchsorted(d, side="left")))
cen["same_i"] = same
cen["late_i"] = cen.full_i + 1
dateonly = cen.published_at.notna() & (cen.time_precision != "exact")
wd = pd.to_datetime(cen.published_at.str[:10], errors="coerce")
is_sess = wd.isin(sess)
print("날짜만 있는 행", int(dateonly.sum()), "· 그날이 거래일", int((dateonly & is_sess).sum()),
      "· same 이 base 와 다름", int((dateonly & (cen.same_i != cen.full_i)).sum()))

rng = np.random.default_rng(7)
p0, p1 = m.period_bounds(info, "2023-01-01", "2026-10-23")
rows = []
for var in ("full_i", "same_i", "late_i"):
    pos_all_c = cen[var] - pos0
    ok = pos_all_c.notna() & (pos_all_c >= 0) & (pos_all_c <= D - 1)
    sig = cen[ok].copy()
    sig["pos"] = pos_all_c[ok].astype(int)
    for hyp, fam, cond, hs in (("H1", "D", True, (5, 10, 20)), ("H2", "R", None, (5, 10, 20)), ("H4", "T", None, (20,))):
        g = sig[sig.family == fam]
        if hyp == "H1":
            g = g[g.lookback_months >= 12]
        for vname in ("all", "first"):
            pos = g.pos.to_numpy(int)
            if vname == "first":
                pos = pos[m.first_only(pos, g.group_id.to_numpy())]
            for h in hs:
                cm, cu, _ = ctl[("SOXX", h, "A")]
                r = m.mean_stats(pos, h, arrs[("SOXX", h)], cm, cu, (p0, p1), rng, 1)
                r.update(var=var, hyp=hyp, variant=vname, h=h)
                r["verdict"] = m.verdict(hyp, r)
                rows.append(r)
out = pd.DataFrame(rows)[["var", "hyp", "variant", "h", "n", "clusters", "mean", "ctl_mean", "ex_mean", "ex_lo", "ex_hi",
                         "p_hi", "p_lo", "verdict"]]
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
res = HERE.parent / "results"
out.to_csv(res / "n4critic_timing.csv", index=False, encoding="utf-8-sig")

# 출처 시점 점검
gs = pd.read_csv(m.CENSUS / "group_sources.csv")
gs["d"] = pd.to_datetime(gs.published_at.astype(str).str[:10], errors="coerce")
mn = gs.groupby("group_id").d.min()
cen["d"] = wd
cen["src_min"] = cen.group_id.map(mn)
earlier = cen[cen.src_min < cen.d]
print("\n그룹 published_at 보다 이른 출처가 있는 그룹:", len(earlier))
print(earlier[["group_id", "family", "published_at", "time_precision", "src_min"]].to_string(index=False))
asia = cen.source.astype(str).str.contains(r"(\bsg\.|\bau\.|\bin\.|\bhk\.|\.kr\b|\.jp\b|\.sg\b|\.hk\b|\.cn\b|\.tw\b|\.in\b|scmp|nikkei|edaily|hankyung|mk\.co|yna|businesstimes|theedge|straitstimes)", regex=True)
print("\n아시아 도메인 대표 출처:", int(asia.sum()))
print(cen[asia][["group_id", "family", "published_at", "time_precision", "note_date", "source"]].to_string(index=False))
