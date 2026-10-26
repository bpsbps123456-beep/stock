"""월말 보유만 쓴 것이 결과를 얼마나 바꾸는가 (2026-10-12, 사용자 질문 "더 수집할 가치가 없다는 거야?").

매일 보유가 있는 지금 범위(TIME + 2ETFQ1)를 추가 펀드와 같은 월말 날짜 11개로 낮춰서 같은 시험을 다시 돌린다.
차이가 작으면 추가 펀드를 매주·매일로 더 받아도 과거 시험 결론은 거의 안 바뀐다.
함께 월말 사이 보유 변화(추가 펀드 월말끼리 겹침, 두 펀드 매일 기준 한 달 안 바뀜)를 잰다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as X  # noqa: E402

H = X.H


def monthly_base(cal, cols, snaps):
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    m = base.copy()
    first = pd.Timestamp("2026-01-02")
    snaps = [pd.Timestamp(s) for s in snaps]
    for d in cal[cal >= first]:
        anchor = max(s for s in snaps if s <= d)
        m.loc[d] = base.loc[:anchor].iloc[-1].values
    return base, m


def run(P, mask):
    G, um = X.grades(P, mask)
    out = {}
    for w in ("front", "back", "y2026"):
        a, b = X.WINDOWS[w]
        out[w] = {r: X.weekly(P, G, um, a, b, r) for r in ("S", "SA", "BUF", "TOP7", "ALL")}
        for r in out[w].values():
            r.pop("picks", None)
    return out


def jacc(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    snaps = sorted(extra["bio"])
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    cal, cols = P["C"].index, P["C"].columns
    base, mon = monthly_base(cal, cols, snaps)
    aux = X.membership(P, [], extra, aux_all=True)  # 2025 처리만 같게
    daily_mask = aux.copy()
    monthly_mask = aux.copy()
    rows26 = cal >= pd.Timestamp("2026-01-02")
    monthly_mask.loc[rows26] = mon.loc[rows26]
    res = {"daily": run(P, daily_mask), "monthly": run(P, monthly_mask)}

    # 월말 사이 보유 변화
    drift = {}
    for k, v in extra.items():
        ds = sorted(v)
        drift[X.EXTRA[k]] = [round(jacc(v[ds[i]], v[ds[i + 1]]), 2) for i in range(len(ds) - 1)]
    b26 = base.loc[rows26]
    sets = {d: set(b26.columns[b26.loc[d]]) for d in b26.index}
    snap_ts = [pd.Timestamp(s) for s in snaps]
    gap = []  # 그날 실제 보유와 직전 월말 보유의 겹침
    for d, s in sets.items():
        anchor = max(x for x in snap_ts if x <= d)
        gap.append(jacc(s, set(base.columns[base.loc[:anchor].iloc[-1]])))
    drift["TIME+2ETFQ1 매일 대비 직전 월말 겹침(평균·최저)"] = [round(sum(gap) / len(gap), 3), round(min(gap), 3)]
    res["drift"] = drift
    (HERE.parent / "results" / "lag_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for w in ("front", "back", "y2026"):
        print(w, " | ".join(f"{r} 매일 {res['daily'][w][r]['ret']:+.1f} 월말 {res['monthly'][w][r]['ret']:+.1f}"
                            for r in ("S", "SA", "BUF", "TOP7", "ALL")))
    for k, v in drift.items():
        print(k, v)


if __name__ == "__main__":
    main()
