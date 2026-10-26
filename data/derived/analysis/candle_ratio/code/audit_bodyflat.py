"""감사(audit) 4: body 비율에서 창이 모두 도지(Σ|C−O|=0)일 때 엔진 값 0.75(사양상 중립 0.5)가 다른 종목 결과에 주는 영향.

engine.ratio를 그대로 쓰되 모두 도지인 body 창만 0.5로 바꾼 판으로 run_grid.grid(비용 0)를 다시 돌려 원래 판과 비교한다.
엔진·run_grid 파일은 고치지 않는다(이 프로세스 안에서만 run_grid.ratio를 바꿔 끼운다).
종목: 5분 연장 3봉 모두 도지 창 비율의 중앙(ANET)·75분위(MPWR)·최대(KEYS). MU는 그런 창이 0개.
사용: py code/audit_bodyflat.py → results/audit_bodyflat.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import run_grid as RG  # noqa: E402

ORIG = E.ratio


def ratio_fixed(O, C, V, n, kind):
    p = ORIG(O, C, V, n, kind)
    if kind == "body" and len(O) >= n:
        den = np.r_[np.full(n - 1, np.nan), np.lib.stride_tricks.sliding_window_view(np.abs(C - O), n).sum(1)]
        p[den == 0] = 0.5
    return p


key = ["tf", "sess", "N", "ratio", "dir", "dayflat", "b", "s"]
res = {}
for sym in sys.argv[1:] or ["ANET", "MPWR", "KEYS"]:
    t0 = time.time()
    RG.ratio = ORIG
    a = RG.grid(sym, "5m", costs_fn=lambda sess: [0.0])
    RG.ratio = ratio_fixed
    b = RG.grid(sym, "5m", costs_fn=lambda sess: [0.0])
    m = a.merge(b[key + ["ret_sel", "ret_conf"]], on=key, suffixes=("", "_fix"))
    body = m[m.ratio == "body"]
    ch = (body.ret_sel - body.ret_sel_fix).abs() > 1e-12
    ch2 = (body.ret_conf - body.ret_conf_fix).abs() > 1e-12
    ta = m.sort_values("ret_sel", ascending=False).iloc[0]
    tb = m.sort_values("ret_sel_fix", ascending=False).iloc[0]
    t20a = m.nlargest(20, "ret_sel"); t20b = m.nlargest(20, "ret_sel_fix")
    res[sym] = dict(cells=len(m), body_cells=len(body), body_changed_sel=int(ch.sum()), body_changed_conf=int(ch2.sum()),
                    abs_dsel_pct_med_changed=round(float(((np.expm1(body.ret_sel) - np.expm1(body.ret_sel_fix)).abs()[ch]).median() * 100), 2) if ch.any() else 0,
                    abs_dconf_pct_med_changed=round(float(((np.expm1(body.ret_conf) - np.expm1(body.ret_conf_fix)).abs()[ch2]).median() * 100), 2) if ch2.any() else 0,
                    abs_dconf_pct_max=round(float((np.expm1(body.ret_conf) - np.expm1(body.ret_conf_fix)).abs().max() * 100), 1),
                    top1_same=bool(all(ta[k] == tb[k] for k in key)),
                    top1_engine=[str(ta[k]) for k in key] + [round(float(np.expm1(ta.ret_sel)) * 100, 1), round(float(np.expm1(ta.ret_conf)) * 100, 1)],
                    top1_fixed=[str(tb[k]) for k in key] + [round(float(np.expm1(tb.ret_sel_fix)) * 100, 1), round(float(np.expm1(tb.ret_conf_fix)) * 100, 1)],
                    top20_conf_median_engine=round(float(np.expm1(t20a.ret_conf.median())) * 100, 1),
                    top20_conf_median_fixed=round(float(np.expm1(t20b.ret_conf_fix.median())) * 100, 1),
                    top20_overlap=int(len(set(t20a.index) & set(t20b.index))), secs=round(time.time() - t0, 1))
    print(sym, json.dumps(res[sym], ensure_ascii=False), flush=True)
RG.ratio = ORIG
path = E.HERE / "results" / "audit_bodyflat.json"
(path).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
