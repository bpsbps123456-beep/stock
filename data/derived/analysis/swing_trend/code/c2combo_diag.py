"""c2combo 진단: 고른 규칙(1위)의 P2 결과가 10/22에 잘린 사건·한 달에 몰렸는지, 단서 하나(S9)만으로 같은지.

py code/c2combo_diag.py → results/c2combo_diag.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from c2combo_lib import FEATURES, RES, binize, boot_diff_ci, edges, group_rows, load
from c2combo_rules import rule_mask, side

rows = []
for det in ("ZZA3", "ZZP10"):
    ev = load(det)
    X = ev[FEATURES].values.astype(float)
    y = ev.y1.values
    top = pd.read_csv(RES / f"c2combo_rules_{det}_top.csv")
    for t in (1, -1):
        B = binize(X, edges(ev, t))
        for train in ("P1", "Y25P1"):
            tt = top[(top.type == t) & (top.train == train) & (top["rank"] == 1)]
            if tt.empty:
                continue
            rid, rule = int(tt.rid.iloc[0]), tt.rule.iloc[0]
            for per in ("P1", "P2", "Y25", "Y24", "Y23"):
                r = group_rows(ev, per, t)
                m = rule_mask(B[r], rid)
                unc = ~ev.censored.values[r]
                mon = ev.date.dt.month.values[r]
                oc = mon != 10
                lo, hi = boot_diff_ci(y[r][unc], m[unc], ev.di.values[r][unc]) if (m & unc).sum() >= 5 else (np.nan, np.nan)
                rows.append(dict(det=det, type=t, train=train, rule=rule, period=per, N=len(r), n_sel=int(m.sum()),
                                 n_sel_censored=int((m & ~unc).sum()), n_sel_oct=int((m & (mon == 10)).sum()) if per == "P2" else np.nan,
                                 diff=side(y[r], m), diff_uncensored=side(y[r][unc], m[unc]), unc_ci_lo=lo, unc_ci_hi=hi,
                                 diff_ex_oct=side(y[r][oc], m[oc]) if per == "P2" else np.nan,
                                 diff_y10=side(ev.y10.values[r], m), diff_y20=side(ev.y20.values[r], m),
                                 S9_low_alone_diff=side(y[r], B[r, FEATURES.index("S9")] == 0)))
out = pd.DataFrame(rows)
out.to_csv(RES / "c2combo_diag.csv", index=False, encoding="utf-8-sig")
print(out.drop(columns="rule").round(4).to_string())

# 모형 진단: P1 학습 로지스틱(18개·종목 10개)의 P2 AUC를 잘림 뺀·10월 뺀 사건으로 다시 잰다(학습은 결정적이라 그대로 재현).
from c2combo_models import MF, SF, auc, fit_logit, top_diff  # noqa: E402

mrows = []
for det in ("ZZA3", "ZZP10"):
    ev = load(det)
    y = ev.y1.values
    lab = (y > 0).astype(int)
    for t in (1, -1):
        r1, r2 = group_rows(ev, "P1", t), group_rows(ev, "P2", t)
        for name, cols in (("logit", FEATURES), ("logit_S", SF)):
            Xc = ev[cols].values.astype(float)
            score = fit_logit(Xc[r1], lab[r1], ev.di.values[r1], False)[0]
            sc = score(Xc[r2])
            unc = ~ev.censored.values[r2]
            oc = ev.date.dt.month.values[r2] != 10
            mrows.append(dict(det=det, type=t, model=name, P2_auc=auc(lab[r2], sc), P2_auc_uncensored=auc(lab[r2][unc], sc[unc]),
                              P2_auc_ex_oct=auc(lab[r2][oc], sc[oc]), P2_top3=top_diff(y[r2], sc)[0],
                              P2_top3_uncensored=top_diff(y[r2][unc], sc[unc])[0], P2_top3_ex_oct=top_diff(y[r2][oc], sc[oc])[0],
                              n_uncensored=int(unc.sum()), n_ex_oct=int(oc.sum())))
mo = pd.DataFrame(mrows)
mo.to_csv(RES / "c2combo_diag_models.csv", index=False, encoding="utf-8-sig")
print(mo.round(4).to_string())
