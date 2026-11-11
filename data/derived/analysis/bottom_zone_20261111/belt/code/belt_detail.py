"""벨트 탐색의 사후 정리: 표본이 큰 후보의 우연 대조와 최종 후보의 국면별 내역.

belt_search.py가 만든 표를 다시 계산하지 않고, 같은 조건·표시 함수를 불러 쓴다.
실행: py data/derived/analysis/bottom_zone_20261111/belt/code/belt_detail.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import belt_search as bs  # noqa: E402

OUT = bs.OUT
FINALISTS = [
    "낙폭20 + SOXL비중상위20",
    "낙폭20 + 신저가3일내 + SOXL비중상위20",
    "낙폭20 + 신저가3일내",
    "낙폭20",
    "5일-12",
    "공포25 + 안전자산 + SOXL비중상위20",
    "신저가3일내 + SOXS거래량 + SOXL비중상위20",
]


def main():
    panel, wide, cnn, news = bs.load()
    atoms = bs.build_atoms(panel, wide, cnn, news)
    lab = bs.labels(panel, wide)
    names = list(atoms.columns)
    keep = (panel["down"] == True) & lab["belt10"].notna() & (panel.index >= "2024-01-02")  # noqa: E712
    idx = panel.index[keep]
    A = atoms.loc[idx].to_numpy()
    l10 = lab.loc[idx, "belt10"].to_numpy(float)
    e = lab.loc[idx, "episode"].to_numpy()
    sel = np.asarray(idx <= bs.SELECT_END)
    con = ~sel
    cs = list(bs.combos(len(names)))
    M = bs.on_matrix(A, cs)

    def epi_count(mask):
        out = np.zeros(len(M), int)
        for k in np.unique(e[mask]):
            out += M[:, mask & (e == k)].any(1)
        return out

    n1, n2 = M[:, sel].sum(1), M[:, con].sum(1)
    e1, e2 = epi_count(sel), epi_count(con)
    rob = (n1 >= 8) & (n2 >= 8) & (e1 >= 3) & (e2 >= 3)        # 사후 기준: 구간마다 8일·3국면 이상
    Mr = M[rob]
    r1, r2 = n1[rob], n2[rob]
    obs_min = np.minimum(Mr[:, sel] @ l10[sel] / r1, Mr[:, con] @ l10[con] / r2)
    rng = np.random.default_rng(bs.SEED + 1)
    years = pd.DatetimeIndex(idx).year.to_numpy()
    cnt75, best = [], []
    for _ in range(1000):
        sh = l10.copy()
        for y in np.unique(years):
            cols = np.where(years == y)[0]
            sh[cols] = np.roll(l10[cols], int(rng.integers(1, len(cols))))
        m = np.minimum(Mr[:, sel] @ sh[sel] / r1, Mr[:, con] @ sh[con] / r2)
        cnt75.append(int((m >= 0.75).sum()))
        best.append(float(m.max()))
    robust_null = dict(
        criterion="각 구간 켜진 날 8일·국면 3개 이상(사후 기준)", n_robust=int(rob.sum()), shifts=1000,
        observed_ge75=int((obs_min >= 0.75).sum()), null_ge75_mean=float(np.mean(cnt75)),
        null_ge75_p95=float(np.percentile(cnt75, 95)),
        p_ge75=float((1 + sum(c >= (obs_min >= 0.75).sum() for c in cnt75)) / 1001),
        observed_best=float(obs_min.max()), null_best_mean=float(np.mean(best)),
        p_best=float((1 + sum(b >= obs_min.max() for b in best)) / 1001),
    )

    # 최종 후보의 국면별 내역 (하락 상태 관찰일 전부, 미정 포함)
    full = pd.concat([panel[["open", "close", "dd60", "down"]], lab, atoms], axis=1)
    full = full[(full.index >= "2024-01-02") & (full["down"] == True) & (full["anchor"] != "")]  # noqa: E712
    pos = {d: i for i, d in enumerate(panel.index)}
    rows, summary = [], {}
    for name in FINALISTS:
        parts = name.split(" + ")
        on = full[full[parts].all(axis=1)]
        sc = on[on.belt10.notna()]
        pre = sc[[pos[d] <= pos[pd.Timestamp(a)] for d, a in zip(sc.index, sc.anchor)]]
        summary[name] = dict(
            days=int(len(sc)), belt10=float(sc.belt10.mean()), belt5=float(sc.belt5.mean()),
            episodes=int(sc.anchor.nunique()),
            before_low_days=int(len(pre)), after_low_days=int(len(sc) - len(pre)),
            to_low_mean=float((sc.low / sc.open - 1).mean()), to_low_worst=float((sc.low / sc.open - 1).min()),
            r20_mean=float(sc.r20.mean()), r20_win=float((sc.r20.dropna() > 0).mean()), r20_n=int(sc.r20.notna().sum()),
        )
        for anchor, g in on.groupby("anchor"):
            first = g.index[0]
            rows.append(dict(
                조합=name, 저점=anchor, 저점종가=round(float(g.low.iloc[0]), 2), 켜진날=len(g),
                벨트10=int((g.belt10 == 1).sum()), 벨트5=int((g.belt5 == 1).sum()),
                첫켜짐=str(first.date()), 첫켜짐_저점대비세션=pos[first] - pos[pd.Timestamp(anchor)],
                첫켜짐_시가_저점대비=round(float(g.open.iloc[0] / g.low.iloc[0] - 1), 4),
                마지막켜짐=str(g.index[-1].date()),
                첫켜짐_20세션수익=None if pd.isna(g.r20.iloc[0]) else round(float(g.r20.iloc[0]), 4),
            ))
    det = pd.DataFrame(rows)
    det.to_csv(OUT / "finalists_by_low.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(robust_null=robust_null, finalists=summary),
              open(OUT / "finalists.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 50)
    print(json.dumps(robust_null, ensure_ascii=False, indent=1))
    print(pd.DataFrame(summary).T.round(3).to_string())
    for name in FINALISTS[:2] + FINALISTS[4:6]:
        print("---", name)
        print(det[det.조합 == name].drop(columns="조합").to_string(index=False))


if __name__ == "__main__":
    main()
