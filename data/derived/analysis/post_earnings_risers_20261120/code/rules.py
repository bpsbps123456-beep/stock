"""사전 등록 2(PREREG2.md) 격자 탐색: 1~3조건 AND, H1 고르기 → H2 확인, 시즌 안 섞기 200번 우연 기준.

결과: results/rules_summary.json, results/rules_{e1,e6}_{up,down}.csv(H1 통과 규칙 전부와 H2 성적)
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "cache"
PERM = 200
RNG = np.random.default_rng(20261120)
DROP = {"ticker", "reaction_date", "ann_date", "time_src", "k", "eps_est", "eps_rep", "eps_surprise", "snap", "season",
        "nx_end_k", "nx_complete", "last_k", "held_TIME", "held_KoAct", "held_any", "w_TIME", "w_KoAct",
        "r0", "d5", "d20", "d40", "t5", "t20", "t40", "nx", "d20_x", "d40_x", "t5_x", "t20_x", "t40_x", "nx_x",
        "group", "timing", "t_peer_n", "kr_custody_rank"}


def load():
    ev = pd.read_csv(BASE / "results/events.csv")
    ft = pd.read_csv(BASE / "results/features.csv")
    fx = pd.read_csv(BASE / "results/features_extra.csv")
    d = ev.merge(ft, on=["ticker", "reaction_date"]).merge(fx, on=["ticker", "reaction_date"])
    px = {k: pd.read_parquet(CACHE / f"px_{k}.parquet") for k in ("open", "close", "adjclose")}
    idx = px["close"]["QQQ"].dropna().index
    px = {k: v.reindex(idx) for k, v in px.items()}
    ac = px["adjclose"]
    ao = px["open"] * ac / px["close"]
    n = len(idx)
    e6 = []
    for e in d.itertuples():
        k = int(e.k)
        if k + 25 < n:
            r = ac[e.ticker].iloc[k + 25] / ao[e.ticker].iloc[k + 6] - 1
            q = ac["QQQ"].iloc[k + 25] / ao["QQQ"].iloc[k + 6] - 1
            e6.append(100 * (r - q))
        else:
            e6.append(np.nan)
    d["e6_x"] = e6
    d["bmo"] = d.timing == "bmo"
    return d


def conditions(d: pd.DataFrame, h1: np.ndarray, entry: str):
    feats = [c for c in d.columns if c not in DROP and not c.startswith("post_") and c != "e6_x"]
    if entry == "e1":  # 반응일 다음 날 조회수는 E1 진입(다음 날 시가) 때 아직 다 집계되지 않는다(Astra 지적, 2026-11-20)
        feats = [c for c in feats if c not in ("d5_x", "a_h07_wiki_post")]
    names, masks = [], []
    for c in feats:
        s = d[c]
        if s.dropna().isin([True, False, 0, 1]).all() and s.nunique() <= 2:
            b = s.fillna(False).astype(bool).to_numpy()
            nb = (~s.fillna(True).astype(bool)).to_numpy()
            for nm, m in ((f"{c}=참", b), (f"{c}=거짓", nb)):
                if m[h1].sum() >= 25:
                    names.append((c, nm))
                    masks.append(m)
            continue
        x = pd.to_numeric(s, errors="coerce")
        lo, hi = np.nanpercentile(x[h1], 33.333), np.nanpercentile(x[h1], 66.667)
        names.append((c, f"{c}≤{lo:.4g}"))
        masks.append((x <= lo).to_numpy())
        names.append((c, f"{c}≥{hi:.4g}"))
        masks.append((x >= hi).to_numpy())
    for g, cnt in d.group.value_counts().items():  # 업종 묶음(20건 이상)
        if cnt >= 20:
            names.append(("group", f"group={g}"))
            masks.append((d.group == g).to_numpy())
    return names, np.array(masks, dtype=bool)


def combos(names):
    m = len(names)
    out = [(i,) for i in range(m)]
    for i, j in itertools.combinations(range(m), 2):
        if names[i][0] != names[j][0]:
            out.append((i, j))
    for i, j, k in itertools.combinations(range(m), 3):
        if len({names[i][0], names[j][0], names[k][0]}) == 3:
            out.append((i, j, k))
    return out


def run(d: pd.DataFrame, entry: str):
    y = d["d20_x" if entry == "e1" else "e6_x"].to_numpy()
    seas = d.season.to_numpy()
    rd = d.reaction_date.astype(str).to_numpy()
    valid = ~np.isnan(y)
    part = np.full(len(d), "", dtype=object)
    part[(seas == "2026Q1") & valid] = "Q1"
    part[(seas == "2026Q2") & valid] = "Q2"
    part[((seas == "2026Q3") | ((seas == "2026Q4") & (rd <= "2026-10-21"))) & valid] = "H2"
    h1 = (part == "Q1") | (part == "Q2")
    names, masks = conditions(d, h1, entry)
    cmb = combos(names)
    # 결과 행렬: 열 0 = 진짜, 1..PERM = 시즌 안 섞기
    Y = np.tile(np.nan_to_num(y), (PERM + 1, 1)).T
    for p in range(1, PERM + 1):
        for s in ("2026Q1", "2026Q2", "2026Q3", "2026Q4"):
            rows = np.where((seas == s) & valid)[0]
            Y[rows, p] = y[RNG.permutation(rows)]
    out = {}
    for direction in ("up", "down"):
        W = (Y > 0) if direction == "up" else (Y < 0)
        res_rows, pass_h1 = [], np.zeros(PERM + 1, int)
        votes = np.zeros((PERM + 1, len(d)), dtype=np.float32)  # 추가: H1 통과 규칙들의 투표(H2를 보지 않음)
        pass_all = np.zeros(PERM + 1, int)
        parts = {pt: np.where(part == pt)[0] for pt in ("Q1", "Q2", "H2")}
        for start in range(0, len(cmb), 20000):
            chunk = cmb[start:start + 20000]
            M = np.ones((len(chunk), len(d)), dtype=bool)
            for r, c in enumerate(chunk):
                for i in c:
                    M[r] &= masks[i]
            Mf = M.astype(np.float32)
            n = {pt: Mf[:, rows].sum(1) for pt, rows in parts.items()}
            w = {pt: Mf[:, rows] @ W[rows].astype(np.float32) for pt, rows in parts.items()}
            nh1 = n["Q1"] + n["Q2"]
            wh1 = w["Q1"] + w["Q2"]
            with np.errstate(invalid="ignore", divide="ignore"):
                ok1 = ((nh1 >= 25)[:, None] & (wh1 >= 0.70 * nh1[:, None])
                       & (n["Q1"] >= 8)[:, None] & (w["Q1"] >= 0.60 * n["Q1"][:, None])
                       & (n["Q2"] >= 8)[:, None] & (w["Q2"] >= 0.60 * n["Q2"][:, None]))
                ok2 = ok1 & (n["H2"] >= 15)[:, None] & (w["H2"] >= 0.70 * n["H2"][:, None])
            pass_h1 += ok1.sum(0)
            votes += ok1.T.astype(np.float32) @ Mf
            pass_all += ok2.sum(0)
            for r in np.where(ok1[:, 0])[0]:
                c = chunk[r]
                sel = M[r]
                yy = y[sel & (part == "H2")]
                res_rows.append({"rule": " & ".join(names[i][1] for i in c), "k": len(c),
                                 "n_h1": int(nh1[r]), "hit_h1": round(wh1[r, 0] / nh1[r], 3),
                                 "n_q1": int(n["Q1"][r]), "hit_q1": round(w["Q1"][r, 0] / n["Q1"][r], 3),
                                 "n_q2": int(n["Q2"][r]), "hit_q2": round(w["Q2"][r, 0] / n["Q2"][r], 3),
                                 "n_h2": int(n["H2"][r]),
                                 "hit_h2": round(w["H2"][r, 0] / n["H2"][r], 3) if n["H2"][r] else np.nan,
                                 "mean_h2": round(float(np.mean(yy)), 2) if len(yy) else np.nan,
                                 "tickers_h2": int(len(set(d.ticker.to_numpy()[sel & (part == "H2")]))),
                                 "pass_h2": bool(ok2[r, 0]), "members": ",".join(map(str, np.where(sel)[0]))})
        # 투표 상위: H1 사건 투표의 80분위 이상(투표 > 0)인 H2 사건의 맞힘률, 진짜 vs 섞기
        vh = []
        for p in range(PERM + 1):
            thr = max(np.percentile(votes[p, h1], 80), 1e-9)
            sel = (part == "H2") & (votes[p] >= thr)
            vh.append((W[sel, p].mean() if sel.sum() else np.nan, int(sel.sum())))
        vh = np.array(vh)
        sel0 = (part == "H2") & (votes[0] >= max(np.percentile(votes[0, h1], 80), 1e-9))
        pd.DataFrame({"ticker": d.ticker[sel0], "reaction_date": d.reaction_date[sel0], "votes": votes[0, sel0],
                      "y": y[sel0]}).to_csv(BASE / f"results/rules_{entry}_{direction}_vote_h2.csv", index=False)
        df = pd.DataFrame(res_rows)
        df.to_csv(BASE / f"results/rules_{entry}_{direction}.csv", index=False)
        out[direction] = {"tested": len(cmb), "conditions": len(names),
                          "base_hit_h1": round(float(W[h1, 0].mean()), 3), "base_hit_h2": round(float(W[part == "H2", 0].mean()), 3),
                          "real_pass_h1": int(pass_h1[0]), "perm_pass_h1_mean": round(float(pass_h1[1:].mean()), 1),
                          "perm_pass_h1_p95": float(np.percentile(pass_h1[1:], 95)),
                          "real_pass_all": int(pass_all[0]), "perm_pass_all_mean": round(float(pass_all[1:].mean()), 2),
                          "perm_pass_all_p95": float(np.percentile(pass_all[1:], 95)),
                          "perm_share_ge_real": round(float((pass_all[1:] >= pass_all[0]).mean()), 3),
                          "n_h1": int(h1.sum()), "n_h2": int((part == "H2").sum()),
                          "vote_h2_hit": round(float(vh[0, 0]), 3), "vote_h2_n": int(vh[0, 1]),
                          "vote_perm_hit_mean": round(float(np.nanmean(vh[1:, 0])), 3),
                          "vote_perm_hit_p95": round(float(np.nanpercentile(vh[1:, 0], 95)), 3),
                          "vote_perm_share_ge_real": round(float(np.nanmean(vh[1:, 0] >= vh[0, 0])), 3)}
        print(entry, direction, json.dumps(out[direction], ensure_ascii=False))
    return out


def main() -> None:
    d = load()
    summary = {e: run(d, e) for e in ("e1", "e6")}
    (BASE / "results/rules_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
