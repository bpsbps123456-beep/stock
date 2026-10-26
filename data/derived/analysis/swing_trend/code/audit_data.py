"""audit 5: 자료 품질과 상장 늦은 종목 처리.
- 결측·0·음수, H<L, 시가·종가가 고저 밖, 중복 날짜, 거래량 0, SPY 달력 대비 빠진 날
- 튀는 날: |종가 수익| > 25%, 시가 갭 > 20%, 하루 튀었다 되돌림(±15% 뒤 반대 ±12%), 고가·저가 꼬리 > 15%
- 수정 배율 일관성: cache O·H·L·C가 같은 배율로 조정됐는지(audit_raw_daily.parquet 대조)
- 구간별 사용 가능 종목(구간 첫날 전 60거래일 이상 + 구간 전 날짜 보유)
→ results/audit_data_flags.csv, results/audit_data_spikes.csv, results/audit_eligibility.csv"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import swing as S  # noqa: E402

raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
cal = sorted(G["SPY"]["date"].tolist())
cal_i = {d: i for i, d in enumerate(cal)}
U = None
up = ROOT / "cache" / "audit_raw_daily.parquet"
if up.exists():
    U = {s: g for s, g in pd.read_parquet(up).groupby("sym")}

flags, spikes = [], []
for s, g in G.items():
    O, H, L, C, V = (g[k].to_numpy(float) for k in ("O", "H", "L", "C", "V"))
    d = g["date"].tolist()
    f = dict(sym=s, first=d[0], last=d[-1], rows=len(g))
    f["nan"] = int(g[["O", "H", "L", "C", "V"]].isna().any(axis=1).sum())
    f["nonpos"] = int(((g[["O", "H", "L", "C"]] <= 0).any(axis=1)).sum())
    f["h_lt_l"] = int((H < L - 1e-9).sum())
    f["oc_outside"] = int(((O > H * (1 + 1e-6)) | (O < L * (1 - 1e-6)) | (C > H * (1 + 1e-6)) | (C < L * (1 - 1e-6))).sum())
    f["dup_date"] = int(g["date"].duplicated().sum())
    f["vol0"] = int((V <= 0).sum())
    exp_days = cal[cal_i[d[0]]: cal_i[d[-1]] + 1] if d[0] in cal_i else []
    f["missing_vs_spy"] = len(set(exp_days) - set(d))
    f["extra_vs_spy"] = len(set(d) - set(cal))
    r = C[1:] / C[:-1] - 1
    gap = O[1:] / C[:-1] - 1
    f["ret_gt25"] = int((np.abs(r) > 0.25).sum())
    f["gap_gt20"] = int((np.abs(gap) > 0.20).sum())
    rev = (np.abs(r[:-1]) > 0.15) & (np.abs(r[1:]) > 0.12) & (np.sign(r[:-1]) != np.sign(r[1:]))
    f["spike_revert"] = int(rev.sum())
    tail_hi = H / np.maximum(O, C) - 1
    tail_lo = 1 - L / np.minimum(O, C)
    f["tail_gt15"] = int(((tail_hi > 0.15) | (tail_lo > 0.15)).sum())
    if U is not None and s in U:
        m = g.merge(U[s], on="date", suffixes=("", "_u"))
        fc = m.C / m.C_u
        f["adj_ratio_inconsistent"] = int(((m.O / m.O_u / fc - 1).abs().gt(1e-5) | (m.H / m.H_u / fc - 1).abs().gt(1e-5)
                                           | (m.L / m.L_u / fc - 1).abs().gt(1e-5)).sum())
    flags.append(f)
    for i in range(1, len(C)):
        rr = C[i] / C[i - 1] - 1; gg = O[i] / C[i - 1] - 1
        th = tail_hi[i]; tl = tail_lo[i]
        nxt = C[i + 1] / C[i] - 1 if i + 1 < len(C) else np.nan
        if abs(rr) > 0.25 or abs(gg) > 0.20 or th > 0.15 or tl > 0.15:
            spikes.append(dict(sym=s, date=d[i], ret=rr, gap=gg, next_ret=nxt, tail_hi=th, tail_lo=tl,
                               revert=bool(abs(rr) > 0.15 and abs(nxt) > 0.12 and np.sign(nxt) != np.sign(rr)),
                               O=O[i], H=H[i], L=L[i], C=C[i], Cprev=C[i - 1]))
fl = pd.DataFrame(flags)
fl.to_csv(ROOT / "results" / "audit_data_flags.csv", index=False)
sp = pd.DataFrame(spikes)
sp.to_csv(ROOT / "results" / "audit_data_spikes.csv", index=False)
cols = ["nan", "nonpos", "h_lt_l", "oc_outside", "dup_date", "vol0", "missing_vs_spy", "extra_vs_spy", "ret_gt25", "gap_gt20",
        "spike_revert", "tail_gt15"] + (["adj_ratio_inconsistent"] if "adj_ratio_inconsistent" in fl else [])
print("종목 144 합계:", fl[cols].sum().to_string())
print("\n문제 있는 종목(결측·0·고저·중복·빈날):")
bad = fl[(fl[["nan", "nonpos", "h_lt_l", "oc_outside", "dup_date", "missing_vs_spy", "extra_vs_spy"]].sum(axis=1) > 0)]
print(bad[["sym", "first", "last", "rows"] + cols].to_string(index=False))
print("\n거래량 0 날 있는 종목:", fl.loc[fl.vol0 > 0, ["sym", "vol0"]].to_string(index=False))
sp25 = sp[sp.date >= pd.Timestamp("2025-01-01").date()]
print(f"\n튀는 날(2025-01 이후) {len(sp25)}건, 되돌림 모양 {int(sp25.revert.sum())}건, 꼬리>15% {int(((sp25.tail_hi > .15) | (sp25.tail_lo > .15)).sum())}건")
print(sp25[sp25.revert | (sp25.tail_hi > .15) | (sp25.tail_lo > .15)].round(3).to_string(index=False))
print("\n|수익|>25% 상위(2025-01 이후):")
print(sp25.reindex(sp25.ret.abs().sort_values(ascending=False).index).head(15).round(3).to_string(index=False))
for s in ("MU", "BE", "NBIS", "NVDA", "SPY"):
    print(s, fl.loc[fl.sym == s, cols].to_dict("records")[0])

# 구간별 사용 가능 종목
el = []
for per, (a, b) in S.PERIODS.items():
    pdays = [x for x in cal if a <= x <= b]
    for s, g in G.items():
        d = g["date"].tolist()
        ix = S.period_index(g, per)
        j0 = int(ix[0]) if len(ix) else -1
        cover = len(ix) == len(pdays)
        el.append(dict(per=per, sym=s, warm=j0, cover=cover, ok=bool(cover and j0 >= 60), first=d[0]))
el = pd.DataFrame(el)
el.to_csv(ROOT / "results" / "audit_eligibility.csv", index=False)
print("\n구간별 사용 가능 종목 수(구간 다 덮음 + 예열 60일 이상):")
for per, q in el.groupby("per", sort=False):
    out = q[~q.ok]
    print(f"  {per}: {int(q.ok.sum())}/144  빼는 종목: " + ", ".join(f"{r.sym}(예열 {max(r.warm, 0)}{'' if r.cover else ', 구간 일부'})" for r in out.itertuples()))
print("예열 60~128일(Pine 129봉 규칙과 다름) 종목·구간:", el[(el.ok) & (el.warm < 129)][["per", "sym", "warm"]].to_string(index=False))
