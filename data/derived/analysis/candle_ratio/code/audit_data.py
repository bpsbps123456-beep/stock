"""감사(audit) 3: MU 자료 품질과 봉 색 영향.

1. 튀는 체결: 5분봉 시가가 앞 봉 종가에서 1% 넘게 떨어졌다가 그 봉 종가가 다시 앞 종가 근처(0.5% 안)로 돌아오면 '시가 튐',
   종가가 시가에서 1% 넘게 벗어났다가 다음 봉 시가가 이 봉 시가 근처(0.5% 안)로 돌아오면 '종가 튐'. 고가·저가 꼬리 2% 넘는 봉은 따로 센다.
   깨끗한 판(튄 시가→앞 종가, 튄 종가→다음 시가)으로 격자를 다시 돌려(엔진 그대로, 비용 0) 결과가 얼마나 바뀌는지 본다.
2. 거래량 0 봉: 프리·정규·애프터별.
3. 빠진 애프터: 날짜별 16:00 이후 5분봉 수, 40분 연장 봉 중 빠지거나 덜 찬 봉. 다른 날 애프터를 같은 모양으로 지운 위약(placebo)으로 결과 민감도를 잰다.
4. 대조: structure_map 40분 정규장 봉(09:30 기준, 따로 받은 prepost=False 5분봉), yfinance 1시간 정규장 봉, 10/10 받은 옛 5분봉 캐시.
사용: py code/audit_data.py → results/audit_data.json, results/audit_data_spikes.csv, results/audit_data_clean_grid.parquet
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_reimpl as R  # noqa: E402  (엔진과 따로 짠 재계산)

ROOT = HERE.parent
ET = "America/New_York"
LAST = pd.Timestamp("2026-10-22").date()
out = {}


def load5():
    d = pd.read_parquet(ROOT / "cache" / "MU_5m.parquet")
    d.index = d.index.tz_convert(ET)
    return d[d.index.date <= LAST].copy()


d = load5()
mins = d.index.hour * 60 + d.index.minute
part = np.where(mins < 570, "pre", np.where(mins < 960, "rth", "post"))

# 1. 튀는 체결 ------------------------------------------------------------------------------
O, H, L, C = d.Open.values, d.High.values, d.Low.values, d.Close.values
pc = np.r_[np.nan, C[:-1]]
no = np.r_[O[1:], np.nan]
TH, BACK = 0.01, 0.005
bad_o = (np.abs(np.log(O / pc)) > TH) & (np.abs(np.log(C / pc)) < BACK)
bad_c = (np.abs(np.log(C / O)) > TH) & (np.abs(np.log(no / O)) < BACK)
bad_o = np.nan_to_num(bad_o, nan=0).astype(bool); bad_c = np.nan_to_num(bad_c, nan=0).astype(bool)
wick = (np.log(H / np.maximum(O, C)) > 0.02) | (np.log(np.minimum(O, C) / L) > 0.02)
sp = d.assign(part=part, bad_o=bad_o, bad_c=bad_c, wick2=wick,
              gap_pct=np.log(O / pc) * 100, body_pct=np.log(C / O) * 100, next_pct=np.log(no / C) * 100)
sp[(sp.bad_o | sp.bad_c)].round(3).to_csv(ROOT / "results" / "audit_data_spikes.csv", encoding="utf-8-sig")
clean = d.copy()
clean.loc[bad_o, "Open"] = pc[bad_o]
clean.loc[bad_c, "Close"] = no[bad_c]
clean["High"] = clean[["High", "Open", "Close"]].max(axis=1)
clean["Low"] = clean[["Low", "Open", "Close"]].min(axis=1)
col_raw = np.sign(C - O); col_cln = np.sign(clean.Close.values - clean.Open.values)
out["spikes"] = dict(rule="시가 튐 |O/앞C|>1% & |C/앞C|<0.5%, 종가 튐 |C/O|>1% & |다음O/O|<0.5%",
                     bad_open=int(bad_o.sum()), bad_close=int(bad_c.sum()),
                     by_part={k: int(((bad_o | bad_c) & (part == k)).sum()) for k in ("pre", "rth", "post")},
                     wick_gt2pct=int(wick.sum()), wick_by_part={k: int((wick & (part == k)).sum()) for k in ("pre", "rth", "post")},
                     color_flips_5m=int((col_raw != col_cln).sum()),
                     worst=sp[(sp.bad_o | sp.bad_c)].assign(a=lambda x: np.maximum(np.abs(x.gap_pct), np.abs(x.body_pct)))
                     .sort_values("a", ascending=False).head(6)[["part", "Open", "Close", "gap_pct", "body_pct", "next_pct"]]
                     .round(2).reset_index().astype(str).values.tolist())


def bars_from(df, tf, sess):
    raw = list(zip(df.index.to_pydatetime(), df.Open.tolist(), df.High.tolist(), df.Low.tolist(), df.Close.tolist(), df.Volume.tolist()))
    return R.make_bars(raw, tf, sess, R.WIN["5m"]["warm"])


flips = {}
for tf in (5, 40, 70):
    a = bars_from(d, tf, "eth"); b = bars_from(clean, tf, "eth")
    ca = np.sign([x["C"] - x["O"] for x in a]); cb = np.sign([x["C"] - x["O"] for x in b])
    flips[f"eth{tf}"] = [int((ca != cb).sum()), len(a)]
out["spikes"]["color_flips_tf"] = flips

# 깨끗한 판으로 엔진 격자 다시 (비용 0) → 원래 격자와 비교
import run_grid as RG  # noqa: E402
RG.load = lambda sym, base: clean
t0 = time.time()
gc = RG.grid("MU", "5m", costs_fn=lambda sess: [0.0])
g0 = pd.read_parquet(ROOT / "results" / "grid_MU_5m.parquet")
g0 = g0[g0.cost == 0].reset_index(drop=True)
key = ["tf", "sess", "N", "ratio", "dir", "dayflat", "b", "s"]
m = g0.merge(gc[key + ["ret_sel", "ret_conf"]], on=key, suffixes=("", "_cl"))
dsel = (m.ret_sel_cl - m.ret_sel) * 100; dconf = (m.ret_conf_cl - m.ret_conf) * 100
top_raw = m.sort_values("ret_sel", ascending=False).iloc[0]
top_cl = m.sort_values("ret_sel_cl", ascending=False).iloc[0]
top20_raw = m.nlargest(20, "ret_sel"); top20_cl = m.nlargest(20, "ret_sel_cl")
out["clean_grid"] = dict(cells=len(m), secs=round(time.time() - t0, 1),
                         changed_cells_sel=int((dsel.abs() > 1e-9).sum()), changed_cells_conf=int((dconf.abs() > 1e-9).sum()),
                         abs_dsel_logpp_med=float(dsel.abs().median()), abs_dsel_logpp_p95=float(dsel.abs().quantile(0.95)),
                         abs_dconf_logpp_med=float(dconf.abs().median()), abs_dconf_logpp_p95=float(dconf.abs().quantile(0.95)),
                         by_sess_ratio=m.assign(a=dsel.abs()).groupby(["sess", "ratio"]).a.median().round(3).reset_index().astype(str).values.tolist(),
                         spearman_sel=float(m.ret_sel.rank().corr(m.ret_sel_cl.rank())),
                         top_raw=[str(top_raw[k]) for k in key] + [round(float(np.expm1(top_raw.ret_sel)) * 100, 1), round(float(np.expm1(top_raw.ret_sel_cl)) * 100, 1)],
                         top_clean=[str(top_cl[k]) for k in key] + [round(float(np.expm1(top_cl.ret_sel_cl)) * 100, 1), round(float(np.expm1(top_cl.ret_conf_cl)) * 100, 1)],
                         top20_conf_median_raw=round(float(np.expm1(top20_raw.ret_conf.median())) * 100, 1),
                         top20_conf_median_clean=round(float(np.expm1(top20_cl.ret_conf_cl.median())) * 100, 1),
                         top20_overlap=int(len(set(top20_raw.index) & set(top20_cl.index))))
m[key + ["ret_sel", "ret_conf", "ret_sel_cl", "ret_conf_cl"]].to_parquet(ROOT / "results" / "audit_data_clean_grid.parquet")

# 2. 거래량 0 ------------------------------------------------------------------------------
out["zero_volume"] = {k: [int(((d.Volume == 0) & (part == k)).sum()), int((part == k).sum())] for k in ("pre", "rth", "post")}

# 3. 빠진 애프터 ----------------------------------------------------------------------------
post = pd.Series(part == "post", index=d.index).groupby(d.index.date).sum()
pre = pd.Series(part == "pre", index=d.index).groupby(d.index.date).sum()
rth = pd.Series(part == "rth", index=d.index).groupby(d.index.date).sum()
short = post[post < 48]
out["missing_post"] = {str(k): int(v) for k, v in short.items()}
out["missing_pre_days"] = int((pre < 66).sum()); out["missing_rth_days"] = int((rth < 78).sum())
b40 = bars_from(d, 40, "eth")
cnt = pd.DataFrame([(x["day"], x["k"], x["n"]) for x in b40], columns=["day", "k", "n"])
per_day = cnt.groupby("day").size()
partial = cnt[cnt.n < 8]
out["eth40_effect"] = dict(days_lt24=per_day[per_day < 24].astype(int).rename(str).to_dict() if hasattr(per_day, "rename") else {},
                           partial_bars=[[str(r.day), int(r.k), int(r.n)] for r in partial.itertuples()],
                           conf_bars=int(sum(1 for x in b40 if R.seg_of(x["day"], R.WIN["5m"]) == 1)))
out["eth40_effect"]["days_lt24"] = {str(k): int(v) for k, v in per_day[per_day < 24].items()}

# 위약: 확인 구간의 다른 연속 4거래일(애프터가 다 있는 날)에서 16:05 이후를 지우고 결과 변화를 잰다
cells = [("원안 eth 밤보유", 40, "eth", 10, "count", "follow", 0.5, 0.5, False),
         ("원안 eth dayflat", 40, "eth", 10, "count", "follow", 0.5, 0.5, True),
         ("고르기 1위", 70, "eth", 15, "vol", "fade", 0.8, 0.2, False),
         ("body 40 eth N10", 40, "eth", 10, "body", "follow", 0.5, 0.5, False)]
days = sorted(set(d.index.date))
full = [x for x in days if post.get(x, 0) == 48 and R.WIN["5m"]["conf"][0] <= x <= R.WIN["5m"]["conf"][1]]
rng = random.Random(7)


def run_cell(df, c):
    _, tf, sess, n, kind, dr, b, s, dfl = c
    bars = bars_from(df, tf, sess)
    seg = [R.seg_of(x["day"], R.WIN["5m"]) for x in bars]
    p = R.ratio_direct(bars, n, kind)
    r = R.ledger(bars, p, seg, b, s, dr == "follow", dfl, 0.0)
    return r["ret_conf"], r["ret_sel"]


base_res = {c[0]: run_cell(d, c) for c in cells}
plac = {c[0]: [] for c in cells}
for _ in range(20):
    i = rng.randrange(0, len(full) - 4)
    drop_days = set(full[i:i + 4])
    mask = np.array([(t.date() in drop_days) and (t.hour * 60 + t.minute >= 965) for t in d.index])
    dd = d[~mask]
    for c in cells:
        rc, _ = run_cell(dd, c)
        plac[c[0]].append((rc - base_res[c[0]][0]) * 100)
out["missing_post_placebo"] = {k: dict(base_conf_pct=round(float(np.expm1(base_res[k][0])) * 100, 2),
                                       abs_dconf_logpp_median=round(float(np.median(np.abs(v))), 2),
                                       abs_dconf_logpp_max=round(float(np.max(np.abs(v))), 2)) for k, v in plac.items()}

# 4-1. structure_map 40분 정규장 봉 대조 ------------------------------------------------------
sm = pd.DataFrame(json.loads((ROOT.parent / "structure_map" / "cache" / "hist" / "MU_40m.json").read_text()),
                  columns=["ms", "o", "h", "l", "c", "v"])
sm["t"] = pd.to_datetime(sm.ms, unit="ms", utc=True).dt.tz_convert(ET)
mine = pd.DataFrame([dict(t=x["t0"], k=x["k"], day=x["day"], O=x["O"], C=x["C"], V=x["V"], n=x["n"]) for x in bars_from(d, 40, "rth")])
mine["start"] = [pd.Timestamp(x).normalize() + pd.Timedelta(minutes=570 + 40 * k) for x, k in zip(mine.t, mine.k)]
j = mine.merge(sm, left_on="start", right_on="t", how="inner", suffixes=("", "_sm"))
rel_o = np.abs(j.O / j.o - 1); rel_c = np.abs(j.C / j.c - 1)
col_eq = np.sign(j.C - j.O) == np.sign(j.c - j.o)
out["vs_structure_map_40m_rth"] = dict(overlap_bars=len(j), overlap_days=int(j.day.nunique()),
                                       first=str(j.start.min()), last=str(j.start.max()),
                                       open_match_1e5=float((rel_o < 1e-5).mean()), close_match_1e5=float((rel_c < 1e-5).mean()),
                                       color_match=float(col_eq.mean()), color_mismatch=int((~col_eq).sum()),
                                       vol_ratio_median=float((j.V / j.v).median()),
                                       mismatches=j.loc[~col_eq, ["start", "O", "C", "o", "c"]].astype(str).values.tolist()[:10])

# 4-2. yfinance 1시간 정규장 봉 vs 5분봉으로 묶은 60분 정규장 봉 ---------------------------------
h = pd.read_parquet(ROOT / "cache" / "MU_1h.parquet"); h.index = h.index.tz_convert(ET)
h = h[(h.index.date <= LAST) & (h.index.date >= R.WIN["5m"]["warm"])]
hm = h.index.hour * 60 + h.index.minute
h = h[(hm >= 570) & (hm < 960)]
m60 = pd.DataFrame([dict(t=x["t0"], k=x["k"], day=x["day"], O=x["O"], C=x["C"], V=x["V"]) for x in bars_from(d, 60, "rth")])
m60["start"] = [pd.Timestamp(x).normalize() + pd.Timedelta(minutes=570 + 60 * k) for x, k in zip(m60.t, m60.k)]
jj = m60.merge(h.reset_index().rename(columns={h.index.name or "index": "start"}), on="start", how="inner")
ce = np.sign(jj.C - jj.O) == np.sign(jj.Close - jj.Open)
out["vs_yf_1h_rth"] = dict(overlap_bars=len(jj), open_match_1e5=float((np.abs(jj.O / jj.Open - 1) < 1e-5).mean()),
                           close_match_1e5=float((np.abs(jj.C / jj.Close - 1) < 1e-5).mean()), color_match=float(ce.mean()),
                           color_mismatch=int((~ce).sum()), vol_ratio_median=float((jj.V / jj.Volume).median()))

# 4-3. 10/10 옛 5분봉 캐시 vs 지금 캐시(겹치는 시각은 새 값) ------------------------------------
old = pd.read_parquet(ROOT.parent / "trend_zones" / "cache" / "dip_raw_1010" / "5m_MU.parquet")
if isinstance(old.columns, pd.MultiIndex):
    old.columns = old.columns.get_level_values(0)
old.index = old.index.tz_convert(ET)
ov = d.join(old[["Open", "Close", "Volume"]], rsuffix="_old", how="inner")
diff = (np.abs(ov.Open / ov.Open_old - 1) > 1e-6) | (np.abs(ov.Close / ov.Close_old - 1) > 1e-6)
cflip = np.sign(ov.Close - ov.Open) != np.sign(ov.Close_old - ov.Open_old)
pm = ov.index.hour * 60 + ov.index.minute
pp = np.where(pm < 570, "pre", np.where(pm < 960, "rth", "post"))
only_old = old.index.difference(d.index)
new_from = d.index.difference(old.index)
out["vs_old_cache"] = dict(overlap=len(ov), price_diff=int(diff.sum()), color_flip=int(cflip.sum()),
                           diff_by_part={k: int((diff & (pp == k)).sum()) for k in ("pre", "rth", "post")},
                           flip_by_part={k: int((cflip & (pp == k)).sum()) for k in ("pre", "rth", "post")},
                           overlap_days=[str(min(ov.index.date)), str(max(ov.index.date))],
                           old_only_bars=len(only_old), new_only_first=str(new_from.min()) if len(new_from) else None)

(ROOT / "results" / "audit_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
