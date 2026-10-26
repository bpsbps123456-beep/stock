"""재계산판(K, 규칙 P)의 묶음 단위 우연 기대: 모든 칸에 같은 섞기(대비마다 단위 전체의 표지 1,000번 섞기).
통과 묶음 칸만. 누출 시험((b) vw_ln_b_median 사건 중앙 z > 가짜 중앙 z, 오프셋 0·+1 중 하나라도)이면 조회·추천 칸을 뺀다."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import recompute as R

OUT = Path(__file__).resolve().parent
COMBOS = [("KOSPI", "8"), ("KOSPI", "12"), ("SKHYNIX", "8"), ("SKHYNIX", "12"), ("SOXX", "8"), ("SOXX", "12")]


def main(tag="K"):
    R.VALID_ONLY = (tag == "V")
    res, leak_rows = [], []
    cache = {}
    for a, x in COMBOS:
        mk = R.MARKET[a]
        if mk not in cache:
            cache[mk] = R.prep_market(mk, kind_adj=True)
        d, Z, PCT, pos = cache[mk]
        th = R.ev["assets"][a]["thresholds"][x]
        for side in ["bottom", "top"]:
            evd = th["bottoms"] if side == "bottom" else th["tops"]
            fk = th["fake_bottoms"] if side == "bottom" else th["fake_tops"]
            reps = {}
            for f in fk:
                r = f["run"]
                if r not in reps or (f["close"] < reps[r]["close"] if side == "bottom" else f["close"] > reps[r]["close"]):
                    reps[r] = f
            repd = [reps[r]["date"] for r in sorted(reps)]
            U = evd + repd
            y0 = np.r_[np.ones(len(evd), int), np.zeros(len(repd), int)]
            rng = np.random.default_rng(R.seed_of(f"family_{a}_{x}_{side}"))
            Yall = np.vstack([y0] + [rng.permutation(y0) for _ in range(R.NPERM)])

            def zval(f, off):
                out = []
                for s in U:
                    i = pos.get(s)
                    out.append(np.nan if i is None or i + off >= len(Z) else Z[f].iloc[i + off])
                return np.array(out)
            # 누출 시험
            leak = False
            for off in [0, 1]:
                v = zval("vw_ln_b_median", off)
                me = np.nanmedian(v[:len(evd)]); mf = np.nanmedian(v[len(evd):])
                leak_rows.append(dict(combo=f"{a}_{x}", side=side, offset=off, med_z_events=me, med_z_fakes=mf, suspect=bool(me > mf)))
                leak |= bool(me > mf)
            cells = []
            for f in R.FEATS:
                if R.meta[f]["status"] != "통과":
                    continue
                if leak and R.meta[f]["bundle"] == "조회·추천":
                    continue
                for off in [0, 1]:
                    v = zval(f, off); m = ~np.isnan(v)
                    Y = Yall[:, m]
                    if y0[m].sum() < 4 or (1 - y0[m]).sum() < 4:  # 관측 표지로 시험 자격
                        continue
                    E = Y.sum(1); F = Y.shape[1] - E
                    ok = (E >= 2) & (F >= 2)
                    ba = np.full(Y.shape[0], 0.5)
                    if ok.any():
                        b, _ = R.loo_ba(v[m], Y[ok], "P")
                        ba[ok] = b
                    null = ba[1:]
                    # 섞기마다의 p(그 섞기 값을 나머지 섞기 분포와 비교)
                    srt = np.sort(null)
                    ge = len(null) - np.searchsorted(srt, ba - 1e-9, side="left")
                    p_all = (1 + ge) / (len(null) + 1)
                    p_all[1:] = (ge[1:]) / len(null)  # 자기 자신 포함 순위(근사)
                    cells.append(dict(feature=f, offset=off, ba=ba[0], p=p_all[0], p_perm=p_all[1:]))
            if not cells:
                continue
            obs = sum(c["p"] < 0.05 for c in cells)
            cnt = np.sum([c["p_perm"] < 0.05 for c in cells], axis=0)
            res.append(dict(tag=tag, combo=f"{a}_{x}", side=side, cells=len(cells), leak_suspect=leak, obs_p05=obs,
                            chance_expect=float(cnt.mean()), null_q95=float(np.quantile(cnt, 0.95)),
                            family_p=float((1 + np.sum(cnt >= obs)) / (len(cnt) + 1)),
                            hits=";".join(f"{c['feature']}@{c['offset']}(p={c['p']:.3f})" for c in sorted(cells, key=lambda c: c["p"]) if c["p"] < 0.05)))
    r = pd.DataFrame(res); l = pd.DataFrame(leak_rows)
    r.to_csv(OUT / f"family_{tag}.csv", index=False, encoding="utf-8-sig"); l.to_csv(OUT / f"leak_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print(r.to_string()); print(l.to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "K")
