"""grid.py 결과의 후보에 실제 투자 비중(상한이 남긴 현금) 점검을 더한다.

엔진의 avg_expo는 exposure 인자(기본 1)만 재므로, caps=True가 남긴 현금은 보이지 않는다.
여기서는 세션별 실제 투자 비중 = 그 세션 보유 비중 합(리밸런스 사이 표류 반영)을 다시 재고,
같은 평균 투자 비중 대조 = caps=False인 같은 설정 바구니를 그 평균 비중으로 늘 든 경로(E.exposure_control)의 H2를 적는다.
실행: py data/derived/analysis/momentum_variants_20261120/branches/construction/finalize.py (grid.py 뒤)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
import engine as E  # noqa: E402
from grid import KEYS, cid, clean  # noqa: E402


def invested_path(e, res):
    """세션별 시가 기준 실제 투자 비중(리밸런스 날 holdings 합에서 출발, 다음 리밸런스까지 표류)."""
    days = res.days
    reb = {d: sum(v.values()) for d, v in res.holdings.items()}
    # holdings 키는 결정일(kd). delay 0이면 체결일과 같다.
    f = np.zeros(len(days))
    cur = 0.0
    for j, d in enumerate(days):
        if d in reb:
            cur = reb[d]
        f[j] = cur
        # 표류: 투자분은 r_oo/cur 비율로 자라고 현금은 그대로
        if j + 1 < len(days) and cur > 0:
            basket = res.r_oo[j] / cur  # 투자분 수익
            cur = cur * (1 + basket) / (cur * (1 + basket) + (1 - cur))
    return f


def main():
    path = HERE / "results.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    e = E.Engine()
    S = {60: e.ret(60), 120: e.ret(120)}

    def run(p, **kw):
        q = {k: v for k, v in p.items() if k != "lb"}
        return e.run(S[p["lb"]], **q, **kw)

    checks = {}
    targets = [f for f in out["finalists"]] + [{**out["baseline"], **dict(universe="TIME", lb=120, freq="W", buffer=5, n=7, weight="time", caps=True)}]
    for f in targets:
        p = {k: f[k] for k in KEYS}
        r = run(p)
        inv = invested_path(e, r)
        c = {"avg_invested": float(inv.mean()), "min_invested": float(inv.min())}
        if p["caps"]:
            q = dict(p, caps=False)
            ru = run(q)
            ctrl = e.exposure_control(ru, c["avg_invested"])
            c["ctrl_uncapped_same_avg_H2"] = E.Engine.span(ctrl, "2026-07-01", E.LAST)
            c["ctrl_uncapped_same_avg_H1"] = E.Engine.span(ctrl, "2026-01-02", "2026-06-30")
            c["uncapped_full_H2"] = E.Engine.span(ru, "2026-07-01", E.LAST)
        checks[cid(p)] = c
    out["invested_checks"] = checks

    # 독립 재계산(같은 엔진 다른 호출 순서) — 후보 H1/H2 재현 차이
    rep = {}
    for f in out["finalists"]:
        p = {k: f[k] for k in KEYS}
        m = e.metrics(run(p))
        rep[f["id"]] = {"dH1": m["H1"] - f["H1"], "dH2": m["H2"] - f["H2"]}
    out["rerun_diff"] = rep
    path.write_text(json.dumps(clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(clean({"checks": checks, "rerun": rep}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
