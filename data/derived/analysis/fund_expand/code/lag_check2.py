"""lag_check 후속: (1) 매일 보유의 급변 날짜 진단 (2) 같은 조건 비교 — 모든 펀드를 월말로 맞춘 U0~U4
(3) 지금 범위를 주간·격주로 낮췄을 때 매일과의 차이(얼마나 촘촘히 받아야 하는가).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as X  # noqa: E402
import lag_check as L  # noqa: E402

H = X.H
RULES = ("S", "SA", "BUF", "TOP7", "ALL")


def snapped(base, cal, snaps):
    m = base.copy()
    snaps = [pd.Timestamp(s) for s in snaps]
    for d in cal[cal >= pd.Timestamp("2026-01-02")]:
        anchor = max(s for s in snaps if s <= d)
        m.loc[d] = base.loc[:anchor].iloc[-1].values
    return m


def with_extras(m, funds, extra, cols, cal):
    m = m.copy()
    for f in funds:
        days = sorted(extra[f])
        for i, d in enumerate(days):
            a = pd.Timestamp(d)
            b = pd.Timestamp(days[i + 1]) if i + 1 < len(days) else cal[-1] + pd.Timedelta(days=1)
            m.loc[(cal >= a) & (cal < b), [c for c in extra[f][d] if c in cols]] = True
    return m


def main():
    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    cal, cols = P["C"].index, P["C"].columns
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    rows26 = cal >= pd.Timestamp("2026-01-02")

    # (1) 진단: 매일 보유 수와 전날 대비 겹침
    b26 = base.loc[rows26]
    n = b26.sum(axis=1)
    prev = b26.shift(1).fillna(False).astype(bool)
    inter = (b26 & prev).sum(axis=1)
    union = (b26 | prev).sum(axis=1)
    j = (inter / union.where(union > 0)).iloc[1:]
    diag = {"n_min": int(n.min()), "n_max": int(n.max()), "n_min_day": str(n.idxmin().date()),
            "day_overlap_low": {str(k.date()): round(float(v), 2) for k, v in j.nsmallest(6).items()},
            "n_around_low": {str(k.date()): int(n[k]) for k in j.nsmallest(6).index}}
    print("진단", diag, flush=True)

    aux = X.membership(P, [], extra, aux_all=True)  # 2025 이전 처리(합집합)는 compare와 같게
    month_snaps = sorted(extra["bio"])
    res = {"diag": diag, "monthly_all": {}, "u0_cadence": {}}

    # (2) 모든 펀드 월말: U0~U4
    mon = snapped(base, cal, month_snaps)
    for name, funds in X.VARIANTS.items():
        m = aux.copy()
        m.loc[rows26] = with_extras(mon, funds, extra, cols, cal).loc[rows26]
        r = L.run(P, m)
        res["monthly_all"][name] = r
        print("월말", name, " | ".join(f"{w}: " + " ".join(f"{k} {r[w][k]['ret']:+.1f}" for k in RULES)
                                      for w in ("front", "back", "y2026")), flush=True)

    # (3) 지금 범위를 주간(그 주 마지막 거래일)·격주로 낮춤
    d26 = cal[rows26]
    wk = pd.Series(d26.to_period("W-SUN"), index=d26)
    week_last = [g.index[-1] for _, g in wk.groupby(wk)]
    week_first = [g.index[0] for _, g in wk.groupby(wk)]
    # 주간 PDF는 그 주 마지막 날 받은 것을 다음 주부터 쓴다: 앵커 = 그 주 마지막 날 → 그날부터 이어 씀
    for label, snaps in (("weekly", [d26[0]] + week_last), ("biweekly", [d26[0]] + week_last[1::2])):
        m = aux.copy()
        m.loc[rows26] = snapped(base, cal, snaps).loc[rows26]
        r = L.run(P, m)
        res["u0_cadence"][label] = r
        print("U0", label, " | ".join(f"{w}: " + " ".join(f"{k} {r[w][k]['ret']:+.1f}" for k in RULES)
                                      for w in ("front", "back", "y2026")), flush=True)
    _ = week_first
    (HERE.parent / "results" / "lag_check2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
