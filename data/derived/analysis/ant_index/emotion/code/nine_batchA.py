"""부록 6 감정 9가지 × 예측 두 층 라벨, 묶음 A 집계 (가격 자료 미사용).

입력: emotion/labels4/primary_01~10.json, second_1~3.json, review200_ai.json,
      emotion/labels/primary_*.json(1차 7범주), sample_key.parquet, sample_a_summary.json, review_label_key.parquet
출력: emotion/nine_batchA.json (md는 별도 작성)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

EMO = Path(r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/emotion")
L4 = EMO / "labels4"
SEED = 20261017
N_BOOT = 2000
E9 = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "HOPE", "REGRET", "ENVY"]
OLD6 = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR"]
NEW3 = ["HOPE", "REGRET", "ENVY"]
TIE = ["MOCK", "ANGER", "DESP", "FEAR", "REGRET", "HOPE", "ENVY", "HYPE", "JOY"]
PRIM = TIE + ["NEUT"]
TGTS = ["bear", "bull", "other"]
PTG = ["market", "stock", "unknown"]
DIRS = ["none", "down", "amb", "up"]


def load(path: Path, tag: str) -> pd.DataFrame:
    rows = json.loads(path.read_text("utf-8"))
    out = []
    for r in rows:
        d = {"id": r["id"], "file": tag}
        assert set(r["emo"]) == set(E9), f"{tag} {r['id']} 감정 키"
        for e in E9:
            v = r["emo"][e]
            assert isinstance(v, int) and 0 <= v <= 3, f"{tag} {r['id']} {e}={v}"
            d[e] = v
        d["mock_target"] = r.get("mock_target")
        d["pred"] = r["pred"]
        d["pred_target"] = r.get("pred_target")
        d["conf"] = r["conf"]
        out.append(d)
    df = pd.DataFrame(out)
    assert df["id"].is_unique, f"{tag} id 중복"
    assert ((df["MOCK"] >= 1) == df["mock_target"].notna()).all(), f"{tag} mock_target 규칙"
    assert set(df["mock_target"].dropna()) <= set(TGTS), f"{tag} mock_target 값"
    assert df["pred"].isin(range(6)).all(), f"{tag} pred 값"
    assert ((df["pred"] == 0) == df["pred_target"].isna()).all(), f"{tag} pred_target 규칙"
    assert set(df["pred_target"].dropna()) <= set(PTG), f"{tag} pred_target 값"
    assert df["conf"].isin([1, 2, 3]).all(), f"{tag} conf 값"
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[E9].to_numpy()
    df["n_emo"] = (X >= 1).sum(1)
    df["neutral"] = df["n_emo"] == 0
    df["mixed"] = df["n_emo"] >= 2
    # 주된 감정: 가장 센 것, 같으면 TIE 순서, 모두 0이면 NEUT
    prim = []
    for row in df[TIE].to_numpy():
        m = row.max()
        prim.append("NEUT" if m == 0 else TIE[int(np.argmax(row == m))])
    df["primary"] = prim
    df["dir"] = df["pred"].map({0: "none", 1: "down", 2: "down", 3: "amb", 4: "up", 5: "up"})
    df["anx_joy"] = ((df["JOY"] >= 1) | (df["HYPE"] >= 1)) & (df["FEAR"] >= 1)
    df["anger_desp"] = (df["ANGER"] >= 1) & (df["DESP"] >= 1)
    df["old6_zero"] = (df[OLD6].to_numpy() == 0).all(1)
    df["only_new3"] = df["old6_zero"] & ~df["neutral"]
    df["only_hope"] = (df["n_emo"] == 1) & (df["HOPE"] >= 1)
    df["combo"] = ["+".join(e for e in E9 if r[e] >= 1) for _, r in df[E9].iterrows()]
    df["elation"] = df["HYPE"] + df["JOY"] - df["DESP"] - df["FEAR"]
    return df


# ------------------------------------------------------------- κ (빠른 판, sklearn으로 점검)

def kappa(a: np.ndarray, b: np.ndarray, k: int, weights: str | None = None) -> float:
    cm = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    ii, jj = np.indices((k, k))
    if weights == "quadratic":
        W = (ii - jj) ** 2 / (k - 1) ** 2
    else:
        W = (ii != jj).astype(float)
    E = np.outer(cm.sum(1), cm.sum(0)) / n
    de = (W * E).sum()
    if de == 0:
        return float("nan")
    return float(1 - (W * cm).sum() / de)


def ci(v: list[float]) -> list[float] | None:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 100:
        return None
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def agreement(P: pd.DataFrame, S: pd.DataFrame, key: pd.DataFrame) -> dict:
    m = S.merge(P, on="id", suffixes=("_2", "_1"), how="left").merge(key[["id", "kind", "shard"]], on="id")
    assert m["HYPE_1"].notna().all() and len(m) == 600 and set(m["shard"]) == {1, 5, 9}
    n = len(m)
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    res: dict = {"n": n, "by_kind_n": m["kind"].value_counts().to_dict(), "per_emotion": {}}

    def boot(f):
        return ci([f(i) for i in idx])

    for e in E9:
        a, b = m[f"{e}_1"].to_numpy().astype(int), m[f"{e}_2"].to_numpy().astype(int)
        pa, pb = (a >= 1).astype(int), (b >= 1).astype(int)
        kb = kappa(pa, pb, 2)
        kw = kappa(a, b, 4, "quadratic")
        # sklearn 점검
        if not np.isnan(kb):
            assert abs(kb - cohen_kappa_score(pa, pb, labels=[0, 1])) < 1e-9
        if not np.isnan(kw):
            assert abs(kw - cohen_kappa_score(a, b, labels=[0, 1, 2, 3], weights="quadratic")) < 1e-9
        both = int(((pa == 1) & (pb == 1)).sum())
        res["per_emotion"][e] = {
            "n_primary": int(pa.sum()), "n_second": int(pb.sum()), "n_both": both,
            "agree_presence": float((pa == pb).mean()),
            "pos_agree": float(2 * both / (pa.sum() + pb.sum())) if pa.sum() + pb.sum() else None,
            "kappa_presence": kb, "kappa_presence_ci95": boot(lambda i: kappa(pa[i], pb[i], 2)),
            "kappa_intensity_qw": kw, "kappa_intensity_qw_ci95": boot(lambda i: kappa(a[i], b[i], 4, "quadratic")),
            "agree_intensity_exact": float((a == b).mean()),
            "presence_ok_0.6": bool(kb >= 0.6), "intensity_ok_0.6": bool(kw >= 0.6),
            "by_kind": {k: {"n": int((m["kind"] == k).sum()),
                            "kappa_presence": kappa(pa[m["kind"].to_numpy() == k], pb[m["kind"].to_numpy() == k], 2),
                            "kappa_intensity_qw": kappa(a[m["kind"].to_numpy() == k], b[m["kind"].to_numpy() == k], 4,
                                                        "quadratic")} for k in ("글", "댓글")},
        }
    # 주된 감정
    pmap = {c: i for i, c in enumerate(PRIM)}
    a = m["primary_1"].map(pmap).to_numpy()
    b = m["primary_2"].map(pmap).to_numpy()
    kp = kappa(a, b, len(PRIM))
    assert abs(kp - cohen_kappa_score(a, b, labels=list(range(len(PRIM))))) < 1e-9
    res["primary"] = {"agree": float((a == b).mean()), "kappa": kp,
                      "kappa_ci95": boot(lambda i: kappa(a[i], b[i], len(PRIM))),
                      "agree_ok_0.75": bool((a == b).mean() >= 0.75),
                      "confusion": pd.crosstab(m["primary_1"], m["primary_2"]).reindex(index=PRIM, columns=PRIM,
                                                                                      fill_value=0).astype(int).to_dict(orient="index"),
                      "by_kind": {k: {"agree": float((a[m["kind"].to_numpy() == k] == b[m["kind"].to_numpy() == k]).mean()),
                                      "kappa": kappa(a[m["kind"].to_numpy() == k], b[m["kind"].to_numpy() == k], len(PRIM))}
                                  for k in ("글", "댓글")}}
    # 중립 여부
    na, nb = m["neutral_1"].to_numpy().astype(int), m["neutral_2"].to_numpy().astype(int)
    res["neutral"] = {"agree": float((na == nb).mean()), "kappa": kappa(na, nb, 2),
                      "n_primary": int(na.sum()), "n_second": int(nb.sum())}
    # 섞인 감정 여부
    xa, xb = m["mixed_1"].to_numpy().astype(int), m["mixed_2"].to_numpy().astype(int)
    res["mixed"] = {"agree": float((xa == xb).mean()), "kappa": kappa(xa, xb, 2),
                    "n_primary": int(xa.sum()), "n_second": int(xb.sum())}
    # 예측
    pa, pb = m["pred_1"].to_numpy().astype(int), m["pred_2"].to_numpy().astype(int)
    k0 = kappa(pa, pb, 6, "quadratic")
    assert abs(k0 - cohen_kappa_score(pa, pb, labels=list(range(6)), weights="quadratic")) < 1e-9
    nz = (pa != 0) & (pb != 0)
    a5, b5 = pa[nz] - 1, pb[nz] - 1
    k5 = kappa(a5, b5, 5, "quadratic")
    dmap = {c: i for i, c in enumerate(DIRS)}
    da, db = m["dir_1"].map(dmap).to_numpy(), m["dir_2"].map(dmap).to_numpy()

    def k5b(i):
        aa, bb = pa[i], pb[i]
        z = (aa != 0) & (bb != 0)
        return kappa(aa[z] - 1, bb[z] - 1, 5, "quadratic")

    has_a, has_b = (pa != 0).astype(int), (pb != 0).astype(int)
    res["pred"] = {
        "qw_kappa_with0": k0, "qw_kappa_with0_ci95": boot(lambda i: kappa(pa[i], pb[i], 6, "quadratic")),
        "qw_kappa_1to5": k5, "qw_kappa_1to5_ci95": boot(k5b), "n_both_nonzero": int(nz.sum()),
        "exact_agree_with0": float((pa == pb).mean()),
        "exact_agree_1to5": float((a5 == b5).mean()) if nz.sum() else None,
        "dir_agree": float((da == db).mean()), "dir_kappa": kappa(da, db, 4),
        "dir_agree_ok_0.75": bool((da == db).mean() >= 0.75),
        "dir_agree_both_nonzero": float((da[nz] == db[nz]).mean()) if nz.sum() else None,
        "has_pred_agree": float((has_a == has_b).mean()), "has_pred_kappa": kappa(has_a, has_b, 2),
        "n_pred_primary": int(has_a.sum()), "n_pred_second": int(has_b.sum()),
        "qw_ok_0.6_with0": bool(k0 >= 0.6), "qw_ok_0.6_1to5": bool(k5 >= 0.6),
        "dir_confusion": pd.crosstab(m["dir_1"], m["dir_2"]).reindex(index=DIRS, columns=DIRS, fill_value=0)
        .astype(int).to_dict(orient="index"),
        "pred_confusion": pd.crosstab(m["pred_1"], m["pred_2"]).reindex(index=range(6), columns=range(6), fill_value=0)
        .astype(int).rename(index=str, columns=str).to_dict(orient="index"),
    }
    # 대상
    bm = m[(m["MOCK_1"] >= 1) & (m["MOCK_2"] >= 1)]
    res["mock_target"] = {"n_both_mock": int(len(bm)),
                          "agree": float((bm["mock_target_1"] == bm["mock_target_2"]).mean()) if len(bm) else None,
                          "kappa": float(cohen_kappa_score(bm["mock_target_1"], bm["mock_target_2"], labels=TGTS))
                          if len(bm) else None}
    bp = m[(m["pred_1"] != 0) & (m["pred_2"] != 0)]
    res["pred_target"] = {"n_both_pred": int(len(bp)),
                          "agree": float((bp["pred_target_1"] == bp["pred_target_2"]).mean()) if len(bp) else None,
                          "kappa": float(cohen_kappa_score(bp["pred_target_1"], bp["pred_target_2"], labels=PTG))
                          if len(bp) else None}
    # 1차 라벨러 확신도별
    res["by_primary_conf"] = {}
    for c in (1, 2, 3):
        s = (m["conf_1"] == c).to_numpy()
        res["by_primary_conf"][str(c)] = {"n": int(s.sum()), "primary_agree": float((a[s] == b[s]).mean()),
                                          "dir_agree": float((da[s] == db[s]).mean())}
    return res


# ------------------------------------------------------------- 분포

def wshare(df: pd.DataFrame, mask) -> float:
    return float(df.loc[mask, "w"].sum() / df["w"].sum())


def dist_block(df: pd.DataFrame, weighted: bool) -> dict:
    df = df.copy()
    if not weighted:
        df["w"] = 1.0
    W = df["w"].sum()
    out: dict = {"n": int(len(df)), "weighted": weighted}

    def sh(mask):
        return float(df.loc[mask, "w"].sum() / W)

    out["emotion"] = {e: {"share_ge1": sh(df[e] >= 1), "mean_intensity": float((df[e] * df["w"]).sum() / W),
                          "n_ge1": int((df[e] >= 1).sum()),
                          "intensity_share": {str(v): sh(df[e] == v) for v in (1, 2, 3)}} for e in E9}
    out["neutral"] = sh(df["neutral"])
    out["n_neutral"] = int(df["neutral"].sum())
    out["mixed"] = sh(df["mixed"])
    out["n_mixed"] = int(df["mixed"].sum())
    out["n_emo_dist"] = {str(k): sh(df["n_emo"] == k) for k in range(int(df["n_emo"].max()) + 1)}
    emo_units = df[~df["neutral"]]
    out["mixed_among_emotional"] = float(emo_units.loc[emo_units["mixed"], "w"].sum() / emo_units["w"].sum())
    out["mean_n_emo_among_emotional"] = float((emo_units["n_emo"] * emo_units["w"]).sum() / emo_units["w"].sum())
    mx = df[df["mixed"]]
    combos = mx.groupby("combo").agg(n=("w", "size"), w=("w", "sum")).sort_values(["w", "n"], ascending=False)
    out["top_mixed_combos"] = [{"combo": c, "n": int(r.n), "share_all": float(r.w / W),
                                "share_mixed": float(r.w / mx["w"].sum())} for c, r in combos.head(10).iterrows()]
    out["n_distinct_mixed_combos"] = int(len(combos))
    out["anx_joy"] = sh(df["anx_joy"])
    out["n_anx_joy"] = int(df["anx_joy"].sum())
    out["anger_desp"] = sh(df["anger_desp"])
    out["n_anger_desp"] = int(df["anger_desp"].sum())
    out["only_new3"] = sh(df["only_new3"])
    out["n_only_new3"] = int(df["only_new3"].sum())
    out["only_hope"] = sh(df["only_hope"])
    out["n_only_hope"] = int(df["only_hope"].sum())
    out["new3_any"] = sh((df[NEW3].to_numpy() >= 1).any(1))
    out["primary_emotion"] = {c: sh(df["primary"] == c) for c in PRIM}
    out["primary_emotion_n"] = {c: int((df["primary"] == c).sum()) for c in PRIM}
    mk = df[df["MOCK"] >= 1]
    out["mock_target"] = {t: float(mk.loc[mk["mock_target"] == t, "w"].sum() / mk["w"].sum()) for t in TGTS}
    out["elation_mean"] = float((df["elation"] * df["w"]).sum() / W)
    # 예측
    out["pred"] = {str(k): sh(df["pred"] == k) for k in range(6)}
    out["pred_n"] = {str(k): int((df["pred"] == k).sum()) for k in range(6)}
    out["pred_share"] = sh(df["pred"] != 0)
    pz = df[df["pred"] != 0]
    out["crowd_mean_P"] = float((pz["pred"] * pz["w"]).sum() / pz["w"].sum() - 3)
    out["pred_dir"] = {d: sh(df["dir"] == d) for d in DIRS}
    out["strong_down"] = sh(df["pred"] == 1)
    out["strong_up"] = sh(df["pred"] == 5)
    out["pred_target"] = {t: float(pz.loc[pz["pred_target"] == t, "w"].sum() / pz["w"].sum()) for t in PTG}
    # 감정 × 예측
    down = df["pred"].isin([1, 2])
    up = df["pred"].isin([4, 5])
    cells = {"MOCK×down": (df["MOCK"] >= 1) & down, "DESP×down": (df["DESP"] >= 1) & down,
             "HYPE×up": (df["HYPE"] >= 1) & up, "REGRET×down": (df["REGRET"] >= 1) & down,
             "REGRET×up": (df["REGRET"] >= 1) & up, "HOPE×down": (df["HOPE"] >= 1) & down,
             "HOPE×up": (df["HOPE"] >= 1) & up, "ENVY×up": (df["ENVY"] >= 1) & up}
    base = {"MOCK×down": "MOCK", "DESP×down": "DESP", "HYPE×up": "HYPE", "REGRET×down": "REGRET",
            "REGRET×up": "REGRET", "HOPE×down": "HOPE", "HOPE×up": "HOPE", "ENVY×up": "ENVY"}
    out["emo_x_pred"] = {}
    for nm, msk in cells.items():
        e = base[nm]
        em = df[e] >= 1
        out["emo_x_pred"][nm] = {"share_all": sh(msk), "n": int(msk.sum()),
                                 "share_within_emotion": float(df.loc[msk, "w"].sum() / df.loc[em, "w"].sum())
                                 if em.any() else None}
    out["H69_diff_MOCKdown_minus_DESPdown"] = out["emo_x_pred"]["MOCK×down"]["share_all"] - out["emo_x_pred"]["DESP×down"]["share_all"]
    out["H75_diff_REGRETdown_minus_REGRETup"] = out["emo_x_pred"]["REGRET×down"]["share_all"] - out["emo_x_pred"]["REGRET×up"]["share_all"]
    # 바람과 예측
    hp = df[df["HOPE"] >= 1]
    out["hope_pred"] = {d: float(hp.loc[hp["dir"] == d, "w"].sum() / hp["w"].sum()) for d in DIRS} if len(hp) else None
    out["hope_with_fear"] = float(hp.loc[hp["FEAR"] >= 1, "w"].sum() / hp["w"].sum()) if len(hp) else None
    out["conf"] = {str(c): sh(df["conf"] == c) for c in (1, 2, 3)}
    return out


def main() -> int:
    key = pd.read_parquet(EMO / "sample_key.parquet")
    P = pd.concat([load(L4 / f"primary_{k:02d}.json", f"primary_{k:02d}") for k in range(1, 11)], ignore_index=True)
    S = pd.concat([load(L4 / f"second_{k}.json", f"second_{k}") for k in range(1, 4)], ignore_index=True)
    R = load(L4 / "review200_ai.json", "review200_ai")
    assert P["id"].is_unique and len(P) == 2000 and set(P["id"]) == set(key["id"]), "1차 id ≠ 표본 id"
    assert S["id"].is_unique and set(S["id"]) == set(key.loc[key["overlap"].notna(), "id"]), "2차 id ≠ 겹침 id"
    # 조각 순서 점검
    for k in range(1, 11):
        want = key[key["shard"] == k].sort_values("shard_pos")["id"].tolist()
        got = P[P["file"] == f"primary_{k:02d}"]["id"].tolist()
        assert want == got, f"primary_{k:02d} 순서"
    for k in range(1, 4):
        want = key[key["overlap"] == k].sort_values("overlap_pos")["id"].tolist()
        got = S[S["file"] == f"second_{k}"]["id"].tolist()
        assert want == got, f"second_{k} 순서"
    P, S, R = add_derived(P), add_derived(S), add_derived(R)

    out: dict = {"made": datetime.now().replace(microsecond=0).isoformat(),
                 "spec": "HYPOTHESES_addendum6.md(+5·4) 감정 9가지 × 예측, 묶음 A 2,000단위",
                 "seed": SEED, "n_boot": N_BOOT, "price_data_read": False,
                 "code": "세션 임시 폴더 scratchpad/nine_batchA.py, nine_batchA_report.py (저장소에 넣지 않음)",
                 "weights": "층(월×시간대×글/댓글×길이)마다 N_frame ÷ n_sample (sample_a_summary.json)",
                 "kappa_note": "있음/없음은 Cohen κ, 세기 0~3과 예측은 이차 가중 κ, 95%는 단위 부트스트랩 2,000회",
                 "inputs": {"primary": "emotion/labels4/primary_01~10.json", "second": "emotion/labels4/second_1~3.json",
                            "review": "emotion/labels4/review200_ai.json", "first_labels": "emotion/labels/primary_*.json"},
                 "checks": {"primary_n": int(len(P)), "second_n": int(len(S)), "review_n": int(len(R)),
                            "id_order_ok": True, "format_ok": True}}

    # 1. 일치도
    out["agreement"] = agreement(P, S, key)

    # 2. 분포 (층 가중)
    strata = pd.DataFrame(json.loads((EMO / "sample_a_summary.json").read_text("utf-8"))["strata"])
    wmap = dict(zip(strata["stratum"], strata["N_frame"] / strata["n_sample"]))
    D = P.merge(key[["id", "kind", "stratum"]], on="id")
    D["w"] = D["stratum"].map(wmap)
    assert D["w"].notna().all()
    out["frame_units"] = float(D["w"].sum())
    out["distribution"] = {"all": dist_block(D, True), "글": dist_block(D[D["kind"] == "글"], True),
                           "댓글": dist_block(D[D["kind"] == "댓글"], True), "raw": dist_block(D, False)}

    # 3. 1차 라벨과 비교
    first = pd.concat([pd.DataFrame(json.loads((EMO / "labels" / f"primary_{k:02d}.json").read_text("utf-8")))
                       for k in range(1, 11)], ignore_index=True)[["id", "cat"]].rename(columns={"cat": "cat_first"})
    C = D.merge(first, on="id", how="left")
    assert C["cat_first"].notna().all()
    nt = C[C["cat_first"] == "NEUT"]
    tagged = (~nt["neutral"]) | (nt["pred"] != 0)

    def both(mask, df):
        return {"n": int(mask.sum()), "raw": float(mask.mean()), "weighted": float(df.loc[mask, "w"].sum() / df["w"].sum())}

    cmp = {"n_first_neut": int(len(nt)),
           "first_neut_weighted_share": float(nt["w"].sum() / C["w"].sum()),
           "tagged_emo_or_pred": both(tagged, nt),
           "emo_only": both((~nt["neutral"]) & (nt["pred"] == 0), nt),
           "pred_only": both(nt["neutral"] & (nt["pred"] != 0), nt),
           "emo_and_pred": both((~nt["neutral"]) & (nt["pred"] != 0), nt),
           "emo_any": both(~nt["neutral"], nt),
           "pred_any": both(nt["pred"] != 0, nt),
           "still_empty": both(nt["neutral"] & (nt["pred"] == 0), nt),
           "emotion_ge1_within_first_neut": {e: both(nt[e] >= 1, nt) for e in E9},
           "only_new3_within_first_neut": both(nt["only_new3"], nt)}
    emo1 = C[C["cat_first"] != "NEUT"]
    cmp["first_emotional_now_neutral"] = both(emo1["neutral"], emo1)
    cmp["first_emotional_now_neutral_and_no_pred"] = both(emo1["neutral"] & (emo1["pred"] == 0), emo1)
    same = []
    for c in ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR"]:
        s = C[C["cat_first"] == c]
        same.append((c, int(len(s)), float((s[c] >= 1).mean()), float((s["primary"] == c).mean())))
    cmp["first_cat_kept"] = {c: {"n": n, "still_ge1": a, "still_primary": b} for c, n, a, b in same}
    cmp["crosstab_first_cat_x_new_primary"] = pd.crosstab(C["cat_first"], C["primary"]).reindex(
        columns=PRIM, fill_value=0).astype(int).to_dict(orient="index")
    out["vs_first_labels"] = cmp

    # 4. 검수지 AI 판정 분포(그대로, 비교 없음)
    rk = pd.read_parquet(EMO / "review_label_key.parquet")
    assert rk["review_id"].tolist() == R["id"].tolist(), "검수지 순서"
    R2 = R.merge(rk[["review_id", "kind"]].rename(columns={"review_id": "id"}), on="id")
    R2["w"] = 1.0
    out["review200_ai"] = {"all": dist_block(R2, False), "글": dist_block(R2[R2["kind"] == "글"], False),
                           "댓글": dist_block(R2[R2["kind"] == "댓글"], False),
                           "note": "가중 없음(검수지는 묶음 A 층 표본이 아님). 사용자 판정과 비교는 하지 않았다."}

    (EMO / "nine_batchA.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
