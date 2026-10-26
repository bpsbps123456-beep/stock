"""불일치 진단: m·p 보정이 A(12.5·1.55)와 B(12.0·1.50)로 갈린 원인.
변형 = {창 시작에서 상태 초기화(A) / 자료 시작부터 상태 이어받기(B)} × {보정 종목 최소 유효일 400(A) / 252(B)}
선택 규칙 = {중앙이 4에 가장 가까운 값(동률은 깊은 쪽 / 얕은 쪽)} 과 {중앙 ≥ 4 가운데 가장 깊은 문턱(X4 정의와 같은 꼴)}
실행: PYTHONIOENCODING=utf-8 py diag_calib.py  → diag_calib.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rare_A"))
sys.dont_write_bytecode = True
import rare_compare as RA  # noqa: E402

TC = RA.TC
H = RA.H_2026


def main():
    raw = pd.read_parquet(RA.PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    rows = {}
    for sym, g in raw.groupby("sym", sort=True):
        if bool(g["is_etf"].iloc[0]):
            continue
        o, piv, pdd, pru, _, _ = RA.per_symbol(g)
        rows[sym] = (o, pdd)
    out = {"m": {}, "p": {}}
    universe = {}
    for kind, grid in (("m", RA.MGRID), ("p", RA.PGRID)):
        rates = {v: [] for v in ("reset400", "reset252", "carry400", "carry252")}
        nd_by_sym = {}
        for sym, (o, pdd) in rows.items():
            d = o["date"].to_numpy()
            dd = o["dd20"].to_numpy()
            if kind == "m":
                thr_all = -(grid[:, None] * o["u252"].to_numpy()[None, :])
            else:
                thr_all = np.minimum(pdd, -1e-12)
            ok = np.isfinite(dd) & np.isfinite(thr_all[0])
            w = RA.inper(d, H)
            nd = int((ok & w).sum())
            nd_by_sym[sym] = nd
            if nd < 252:
                continue
            yrs = nd / 252.0
            # reset: 창 안 유효값만
            t, r = RA.thr_from_log(thr_all[:, ok & w])
            n_reset, _, _ = RA.episodes(dd[ok & w][None, :], t, r)
            # carry: 자료 시작부터(유효값만), 창 안에서 시작한 것만
            t2, r2 = RA.thr_from_log(thr_all[:, ok])
            _, _, start = RA.episodes(dd[ok][None, :], t2, r2)
            n_carry = start[:, w[ok]].sum(axis=1)
            for tag, n in (("reset", n_reset), ("carry", n_carry)):
                rates[tag + "252"].append(n / yrs)
                if nd >= 400:
                    rates[tag + "400"].append(n / yrs)
        universe[kind] = {k: len(v) for k, v in rates.items()}
        universe[kind + "_between_252_400"] = {s: n for s, n in nd_by_sym.items() if 252 <= n < 400}
        for tag, lst in rates.items():
            A = np.array(lst)                      # (종목, 격자)
            med = np.median(A, axis=0)
            deep = (lambda i: -grid[i]) if kind == "m" else (lambda i: grid[i])
            closest_deep = int(min(range(len(grid)), key=lambda i: (abs(med[i] - 4), deep(i))))
            closest_shallow = int(min(range(len(grid)), key=lambda i: (abs(med[i] - 4), -deep(i))))
            ge4 = [i for i in range(len(grid)) if med[i] >= 4]
            ge4_deep = (max(ge4, key=lambda i: grid[i]) if kind == "m" else min(ge4, key=lambda i: grid[i]))
            nb = {float(grid[i]): round(float(med[i]), 3) for i in range(len(grid))
                  if (kind == "m" and 1.35 <= grid[i] <= 1.7) or (kind == "p" and 10.5 <= grid[i] <= 13.5)}
            # p=12.0 / m=1.55 에서 중앙 근처 종목 순위값
            out[kind][tag] = {"n": len(lst), "near": nb,
                              "closest_tie_deep": float(grid[closest_deep]),
                              "closest_tie_shallow": float(grid[closest_shallow]),
                              "ge4_deepest": float(grid[ge4_deep])}
            if kind == "p":
                i12 = int(np.flatnonzero(np.isclose(grid, 12.0))[0])
                srt = np.sort(A[:, i12])
                k = len(srt)
                out[kind][tag]["p12_middle_values"] = [round(float(x), 3) for x in srt[(k - 1) // 2:k // 2 + 1]]
    out["universe"] = universe
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(HERE, "diag_calib.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
