"""감정 범주 묶음 A: 일치도 → 학습 자료 → 분류기 → 전체 적용 → 창별 비율 → 사용자 검수지 (HYPOTHESES_addendum3.md 3·5·6단계).

입력(읽기만):
  emotion/labels/primary_01~10.json (조각별 1차 라벨), emotion/labels/second_1~3.json (이중 라벨 600)
  emotion/sample_key.parquet (id → units 행), emotion/sample_a_summary.json (층 크기)
  interim/results/units.parquet, windows_kr.parquet, windows_us.parquet (로더 산출, 10/15 스냅샷)
가격·지수·수익 자료는 읽지 않는다. 급락일(7/29~31, 10/13)은 지시로 받은 날짜 목록이고 가격에서 계산하지 않았다.

고정 설정(가격 연결 전, 결과를 보기 전에 정함):
  가림 = sample_a.unit_text(글: '제목: …\\n본문: 앞 600자', 댓글: 앞 400자, 날짜·시각·사건·숫자·주소 가림) 그대로.
         분류기 입력에서는 자리표시 [숫자]·[사건]·[시각]·[링크]를 사용 영역 글자 하나로 바꾼다('숫자' 같은 낱말과 섞이지 않게).
  특징 = TfidfVectorizer(analyzer='char', ngram_range=(2,4), sublinear_tf=True, min_df=2), 소문자화(기본값).
  모형 = LogisticRegression(lbfgs, L2, 다항, class_weight=None, max_iter=5000).
  C = logspace(-2, 2, 9) 가운데 떼어 두지 않은 80% 안 5겹 층화 교차검증 macro-F1 최대(동점이면 작은 C).
  나눔 = 범주 층화 80/20, random_state=SEED. 이중 라벨에서 두 라벨이 다른 단위는 '불일치'로 학습·평가에서 뺀다.
  배포 모형 = 80%로 학습한 모형(떼어 둔 20% 정밀도가 그대로 그 모형의 값이 되게).
  조롱 대상 = MOCK 단위만으로 같은 설정의 2단 분류기(other·bull·bear). 적용은 주 범주가 MOCK인 단위에만.
  창 비율 = 창(본판·완전한 창) 안 (작성자, 범주) 쌍을 한 번씩 센다. 전체판 분모 = 7범주 쌍 합, 감정판 분모 = NEUT 뺀 6범주 쌍 합.
  들뜸 지수 = (HYPE + JOY) − (DESP + FEAR) 감정판. 엔트로피 = 감정판 6범주 −Σp ln p (와 ÷ ln 6).
  장애 창(outage = True)은 비율·지수를 결측으로 둔다(§2.4). 쌍 수는 남긴다.

실행: py emotion_batchA.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
EMO = HERE.parent
ANT = EMO.parent
RES = ANT / "interim" / "results"
sys.path.insert(0, str(HERE))
import sample_a as SA  # noqa: E402  (가림 규칙·틀 정의를 그대로 쓴다)

SEED = 20261017
CATS = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "NEUT"]
EMO6 = [c for c in CATS if c != "NEUT"]
KO = {"HYPE": "들뜸·자랑", "JOY": "기쁨·안도", "MOCK": "조롱·비꼼", "DESP": "우울·절망", "ANGER": "분노·욕",
      "FEAR": "불안·공포", "NEUT": "중립·정보"}
TGTS = ["other", "bull", "bear"]
C_GRID = [float(x) for x in np.logspace(-2, 2, 9)]
PRECISION_MIN, KAPPA_MIN = 0.60, 0.60
CRASH_DAYS = {date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31), date(2026, 10, 13)}
N_REVIEW_POST, N_REVIEW_COMMENT = 100, 100
N_BOOT = 2000
PH = {"[숫자]": "", "[사건]": "", "[시각]": "", "[링크]": ""}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feat_text(s: str) -> str:
    for k, v in PH.items():
        s = s.replace(k, v)
    return s


def make_pipe(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2)),
        ("lr", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
    ])


# ================================================================ 1. 라벨·일치도

def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    prim = []
    for k in range(1, 11):
        for r in json.loads((EMO / "labels" / f"primary_{k:02d}.json").read_text("utf-8")):
            r = dict(r)
            r["file"] = f"primary_{k:02d}"
            prim.append(r)
    sec = []
    for k in range(1, 4):
        for r in json.loads((EMO / "labels" / f"second_{k}.json").read_text("utf-8")):
            r = dict(r)
            r["file"] = f"second_{k}"
            sec.append(r)
    P, S = pd.DataFrame(prim), pd.DataFrame(sec)
    for df, nm in ((P, "primary"), (S, "second")):
        assert df["id"].is_unique, f"{nm} id 중복"
        assert set(df["cat"]) <= set(CATS), f"{nm} 범주 값"
        bad = df[(df["cat"] == "MOCK") != df["mock_target"].notna()]
        assert bad.empty, f"{nm} mock_target 규칙 위반 {len(bad)}"
        assert set(df.loc[df["cat"] == "MOCK", "mock_target"]) <= set(TGTS)
    return P, S


def kappa_bin(a: np.ndarray, b: np.ndarray, c: str) -> float:
    x, y = (a == c).astype(int), (b == c).astype(int)
    if x.sum() == 0 and y.sum() == 0:
        return float("nan")
    return float(cohen_kappa_score(x, y))


def agreement(P: pd.DataFrame, S: pd.DataFrame, key: pd.DataFrame) -> dict:
    m = S.merge(P, on="id", suffixes=("_2", "_1"), how="left").merge(key[["id", "kind", "shard"]], on="id")
    assert m["cat_1"].notna().all() and len(m) == 600 and set(m["shard"]) == {1, 5, 9}
    a, b = m["cat_1"].to_numpy(), m["cat_2"].to_numpy()
    rng = np.random.default_rng(SEED)
    boots = {c: [] for c in CATS}
    boots["_all"] = []
    n = len(m)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        aa, bb = a[i], b[i]
        boots["_all"].append(cohen_kappa_score(aa, bb, labels=CATS))
        for c in CATS:
            boots[c].append(kappa_bin(aa, bb, c))

    def ci(v):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    per = {}
    for c in CATS:
        per[c] = {"kappa": kappa_bin(a, b, c), "ci95": ci(boots[c]), "n_primary": int((a == c).sum()),
                  "n_second": int((b == c).sum()), "n_both": int(((a == c) & (b == c)).sum()),
                  "pos_agree": float(2 * ((a == c) & (b == c)).sum() / max(1, (a == c).sum() + (b == c).sum()))}
    cm = pd.DataFrame(confusion_matrix(a, b, labels=CATS), index=CATS, columns=CATS)
    by_kind = {k: {"n": int((m["kind"] == k).sum()),
                   "kappa": float(cohen_kappa_score(a[m["kind"] == k], b[m["kind"] == k], labels=CATS)),
                   "agree": float((a[m["kind"] == k] == b[m["kind"] == k]).mean())} for k in ("글", "댓글")}
    both_mock = m[(m["cat_1"] == "MOCK") & (m["cat_2"] == "MOCK")]
    tgt = {"n_both_mock": int(len(both_mock)),
           "agree": float((both_mock["mock_target_1"] == both_mock["mock_target_2"]).mean()) if len(both_mock) else None,
           "kappa": float(cohen_kappa_score(both_mock["mock_target_1"], both_mock["mock_target_2"], labels=TGTS))
           if len(both_mock) else None,
           "table": pd.crosstab(both_mock["mock_target_1"], both_mock["mock_target_2"]).to_dict()}
    aux = {"stance_kappa": float(cohen_kappa_score(m["stance_1"], m["stance_2"])),
           "stance_agree": float((m["stance_1"] == m["stance_2"]).mean()),
           "px_kappa": float(cohen_kappa_score(m["px_1"].astype(int), m["px_2"].astype(int))),
           "px_agree": float((m["px_1"] == m["px_2"]).mean())}
    conf_agree = {int(c): {"n": int((m["conf_1"] == c).sum()),
                           "agree": float((a[m["conf_1"] == c] == b[m["conf_1"] == c]).mean())}
                  for c in sorted(m["conf_1"].unique())}
    return {"n": n, "kappa_all": float(cohen_kappa_score(a, b, labels=CATS)), "kappa_all_ci95": ci(boots["_all"]),
            "agree_all": float((a == b).mean()), "per_cat": per, "confusion": cm, "by_kind": by_kind,
            "mock_target": tgt, "aux": aux, "agree_by_primary_conf": conf_agree, "merged": m}


# ================================================================ 2. 학습 자료

def build_train(P: pd.DataFrame, S: pd.DataFrame, key: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    t = key[["id", "kind", "utype", "unit_idx", "shard", "overlap", "text_sha256", "stratum"]].merge(
        P[["id", "cat", "mock_target", "conf"]], on="id", how="left")
    assert t["cat"].notna().all() and len(t) == 2000
    t = t.merge(S[["id", "cat", "mock_target"]].rename(columns={"cat": "cat_2", "mock_target": "mock_target_2"}),
                on="id", how="left")
    # 원문은 units.parquet에서 가져와 같은 가림을 다시 적용하고, 라벨러가 본 글과 같은지 지문으로 확인한다
    rows = units.loc[t["unit_idx"].to_numpy(), ["utype", "title", "text"]]
    texts = [SA.unit_text({"utype": r.utype, "title": r.title, "text": r.text})[0] for r in rows.itertuples()]
    t["text_masked"] = texts
    t["sha_ok"] = [hashlib.sha256(x.encode("utf-8")).hexdigest() == h for x, h in zip(texts, t["text_sha256"])]
    assert t["sha_ok"].all(), f"가림 재현 실패 {int((~t['sha_ok']).sum())}"
    dbl = t["cat_2"].notna()
    t["cat_train"] = t["cat"]
    t.loc[dbl & (t["cat"] != t["cat_2"]), "cat_train"] = "불일치"
    # 조롱 대상: 두 라벨 모두 MOCK인데 대상이 다르면 대상 학습에서만 뺀다
    t["tgt_train"] = t["mock_target"].where(t["cat_train"] == "MOCK")
    tdis = dbl & (t["cat_train"] == "MOCK") & (t["mock_target"] != t["mock_target_2"])
    t.loc[tdis, "tgt_train"] = "불일치"
    return t.drop(columns=["sha_ok"])


# ================================================================ 3. 분류기

def cv_select(X: list[str], y: np.ndarray) -> tuple[float, list[dict]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    X = np.asarray(X, dtype=object)
    rows = []
    for C in C_GRID:
        sc = []
        for tr, va in folds:
            p = make_pipe(C).fit(X[tr], y[tr])
            sc.append(f1_score(y[va], p.predict(X[va]), average="macro", labels=sorted(set(y)), zero_division=0))
        rows.append({"C": C, "macro_f1_mean": float(np.mean(sc)), "macro_f1_sd": float(np.std(sc, ddof=1)),
                     "folds": [float(s) for s in sc]})
    best = max(rows, key=lambda r: (round(r["macro_f1_mean"], 12), -r["C"]))
    return best["C"], rows


def eval_block(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    npred = {c: int((y_pred == c).sum()) for c in labels}
    per = {c: {"precision": (float(p[i]) if npred[c] else None), "recall": float(r[i]), "f1": float(f[i]),
               "support": int(s[i]), "n_pred": npred[c]} for i, c in enumerate(labels)}
    return {"n": int(len(y_true)), "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
            "kappa_vs_label": float(cohen_kappa_score(y_true, y_pred, labels=labels)), "per_cat": per,
            "confusion": pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)}


# ================================================================ 4. 적용·창

def mask_all(units: pd.DataFrame) -> list[str]:
    return [feat_text(SA.unit_text({"utype": ut, "title": ti, "text": tx})[0])
            for ut, ti, tx in zip(units["utype"], units["title"], units["text"])]


def window_table(market: str, units: pd.DataFrame, pred: pd.DataFrame, wins: pd.DataFrame) -> pd.DataFrame:
    key = "win_kr" if market == "kr" else "win_u"
    w = wins[(wins["variant"] == "main") & wins["complete"]].sort_values("close_kst").reset_index(drop=True)
    idx = pd.Index(w["win_id"], name="win_id")
    F = pd.DataFrame(index=idx)
    for c in ["market", "D", "P", "weekday", "kind", "main_kind", "length_h", "over96", "xholiday_mismatch",
              "spam_wave", "missing_rate", "n_posts", "n_comments", "n_authors"]:
        F[c] = w[c].to_numpy()
    F["outage_flag"] = w["outage"].fillna(False).astype(bool).to_numpy()
    F["valid"] = ~F["outage_flag"]
    d = pd.DataFrame({"win": units[key].to_numpy(), "a": units["a"].to_numpy(), "cat": pred["cat"].to_numpy(),
                      "tgt": pred["mock_target"].to_numpy()})
    d = d[d["win"].isin(idx)]
    F["n_units"] = d.groupby("win").size().reindex(idx).fillna(0).astype(int).to_numpy()
    F["n_units_emo"] = d[d["cat"] != "NEUT"].groupby("win").size().reindex(idx).fillna(0).astype(int).to_numpy()
    F["n_auth"] = d.groupby("win")["a"].nunique().reindex(idx).fillna(0).astype(int).to_numpy()
    pairs = d.drop_duplicates(["win", "a", "cat"])
    cnt = pairs.groupby(["win", "cat"]).size().unstack(fill_value=0).reindex(index=idx, columns=CATS, fill_value=0)
    for c in CATS:
        F[f"n_pair_{c}"] = cnt[c].to_numpy()
    tot_all = cnt[CATS].sum(axis=1).to_numpy(float)
    tot_emo = cnt[EMO6].sum(axis=1).to_numpy(float)
    F["n_pair_all"], F["n_pair_emo"] = tot_all.astype(int), tot_emo.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in CATS:
            F[f"r_all_{c}"] = np.where(tot_all > 0, cnt[c].to_numpy() / tot_all, np.nan)
        for c in EMO6:
            F[f"r_emo_{c}"] = np.where(tot_emo > 0, cnt[c].to_numpy() / tot_emo, np.nan)
        # 보조: 작성자 비율(§2.0 X_A 꼴) = 그 범주 단위를 한 번이라도 쓴 고유 작성자 ÷ 창 안 고유 작성자
        na = F["n_auth"].to_numpy(float)
        for c in CATS:
            F[f"ash_{c}"] = np.where(na > 0, cnt[c].to_numpy() / na, np.nan)
        F["hype_index"] = (F["r_emo_HYPE"] + F["r_emo_JOY"]) - (F["r_emo_DESP"] + F["r_emo_FEAR"])
        pe = F[[f"r_emo_{c}" for c in EMO6]].to_numpy(float)
        ent = -np.nansum(np.where(pe > 0, pe * np.log(np.where(pe > 0, pe, 1)), 0.0), axis=1)
        ent[np.isnan(pe).any(axis=1)] = np.nan
        F["entropy_emo"] = ent
        F["entropy_emo_norm"] = ent / math.log(len(EMO6))
    mp = d[d["cat"] == "MOCK"].drop_duplicates(["win", "a", "tgt"])
    tc = mp.groupby(["win", "tgt"]).size().unstack(fill_value=0).reindex(index=idx, columns=TGTS, fill_value=0)
    tt = tc.sum(axis=1).to_numpy(float)
    for t in TGTS:
        F[f"n_mock_{t}"] = tc[t].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            F[f"mock_share_{t}"] = np.where(tt > 0, tc[t].to_numpy() / tt, np.nan)
    F["mock_bear_minus_bull"] = F["mock_share_bear"] - F["mock_share_bull"]
    feat_cols = [c for c in F.columns if c.startswith(("r_all_", "r_emo_", "ash_", "mock_share_"))] + [
        "hype_index", "entropy_emo", "entropy_emo_norm", "mock_bear_minus_bull"]
    F.loc[~F["valid"], feat_cols] = np.nan
    return F.reset_index()


# ================================================================ 5. 검수지

def review_sample(units: pd.DataFrame, pred: pd.DataFrame, key: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    f, _ = SA.load_frame()                       # 묶음 A와 같은 틀(글 + 마감 전 댓글, 7/13~10/15, latest 제외)
    f = f[~f["dt_kst"].dt.date.isin(CRASH_DAYS)]
    f = f[~f["unit_idx"].isin(set(key["unit_idx"]))]
    picks = []
    for kind, n, nm in (("글", N_REVIEW_POST, "REVIEW_POST"), ("댓글", N_REVIEW_COMMENT, "REVIEW_COMMENT")):
        pos = f.index[f["kind"].eq(kind)].to_numpy()
        picks.extend(np.random.default_rng(SA._seed(nm)).choice(pos, size=n, replace=False))
    r = f.loc[picks].copy()
    r = r.iloc[np.random.default_rng(SA._seed("REVIEW_SHUFFLE")).permutation(len(r))].reset_index(drop=True)
    rng = np.random.default_rng(SA._seed("REVIEW_IDS"))
    ids: list[str] = []
    while len(ids) < len(r):
        x = f"R{int(rng.integers(0, 2**32)):08x}"
        if x not in ids:
            ids.append(x)
    r["id"] = ids
    r["text_masked"] = [SA.unit_text(row)[0] for _, row in r.iterrows()]
    pc = pred.set_index("unit_idx")
    r["model_cat"] = pc.loc[r["unit_idx"], "cat"].to_numpy()
    r["model_p"] = pc.loc[r["unit_idx"], [f"p_{c}" for c in CATS]].max(axis=1).to_numpy()
    r["model_mock_target"] = pc.loc[r["unit_idx"], "mock_target"].to_numpy()
    assert not set(r["unit_idx"]) & set(key["unit_idx"])
    assert not r["dt_kst"].dt.date.isin(CRASH_DAYS).any()
    items = [{"id": x.id, "text": x.text_masked, "model_cat": x.model_cat, "model_cat_ko": KO[x.model_cat],
              "user_ok": None, "user_cat": None} for x in r.itertuples()]
    rk = r[["id", "kind", "utype", "unit_idx", "no", "cid", "dt_kst", "model_cat", "model_p", "model_mock_target"]].copy()
    rk["date_kst"] = rk["dt_kst"].dt.date.astype(str)
    rk["cid"] = rk["cid"].astype("Int64")
    return items, rk


# ================================================================ 실행

def main() -> int:
    t0 = datetime.now()
    out: dict = {"made": t0.replace(microsecond=0).isoformat(), "spec": "HYPOTHESES_addendum3.md 3·5·6단계(묶음 A)",
                 "seed": SEED, "sklearn": sklearn.__version__, "price_data_read": False}
    key = pd.read_parquet(EMO / "sample_key.parquet")
    P, S = load_labels()
    assert set(P["id"]) == set(key["id"]), "1차 라벨 id ≠ 표본 id"
    assert set(S["id"]) == set(key.loc[key["overlap"].notna(), "id"]), "이중 라벨 id ≠ 겹침 id"

    # 1. 일치도
    ag = agreement(P, S, key)
    out["agreement"] = {k: v for k, v in ag.items() if k not in ("confusion", "merged")}
    out["agreement"]["confusion"] = ag["confusion"].to_dict()

    # 라벨 표본 비율(그대로, 층 가중)
    strata = pd.DataFrame(json.loads((EMO / "sample_a_summary.json").read_text("utf-8"))["strata"])
    wmap = dict(zip(strata["stratum"], strata["N_frame"] / strata["n_sample"]))
    lab = key[["id", "kind", "stratum"]].merge(P[["id", "cat", "mock_target"]], on="id")
    lab["w"] = lab["stratum"].map(wmap)
    assert lab["w"].notna().all()
    share = {}
    for nm, sub in (("all", lab), ("글", lab[lab["kind"] == "글"]), ("댓글", lab[lab["kind"] == "댓글"])):
        share[nm] = {"n": int(len(sub)),
                     "raw": {c: float((sub["cat"] == c).mean()) for c in CATS},
                     "weighted": {c: float(sub.loc[sub["cat"] == c, "w"].sum() / sub["w"].sum()) for c in CATS}}
        emo = sub[sub["cat"] != "NEUT"]
        share[nm]["weighted_emo"] = {c: float(emo.loc[emo["cat"] == c, "w"].sum() / emo["w"].sum()) for c in EMO6}
        mk = sub[sub["cat"] == "MOCK"]
        share[nm]["weighted_mock_target"] = {t: float(mk.loc[mk["mock_target"] == t, "w"].sum() / mk["w"].sum())
                                             for t in TGTS}
    out["label_share"] = share

    # 2. 학습 자료
    ucols = ["no", "cid", "a", "utype", "dt_kst", "title", "text", "dt_conf", "win_kr", "win_u", "pre_close_kr"]
    units = pd.read_parquet(RES / "units.parquet", columns=ucols)
    tr = build_train(P, S, key, units)
    ok = tr["cat_train"] != "불일치"
    out["train"] = {"n_units": int(len(tr)), "n_double": int(tr["cat_2"].notna().sum()),
                    "n_disagree_excluded": int((~ok).sum()), "n_usable": int(ok.sum()),
                    "usable_by_cat": tr.loc[ok, "cat_train"].value_counts().reindex(CATS).fillna(0).astype(int).to_dict()}
    use = tr[ok].copy()
    use["X"] = [feat_text(s) for s in use["text_masked"]]
    i_tr, i_te = train_test_split(use.index.to_numpy(), test_size=0.2, stratify=use["cat_train"], random_state=SEED)
    tr["split"] = "excluded"
    tr.loc[i_tr, "split"] = "train"
    tr.loc[i_te, "split"] = "heldout"
    Xtr, ytr = use.loc[i_tr, "X"].tolist(), use.loc[i_tr, "cat_train"].to_numpy()
    Xte, yte = use.loc[i_te, "X"].tolist(), use.loc[i_te, "cat_train"].to_numpy()
    out["train"]["split"] = {"train": len(i_tr), "heldout": len(i_te),
                             "heldout_by_cat": pd.Series(yte).value_counts().reindex(CATS).fillna(0).astype(int).to_dict()}

    # 3. 분류기(범주)
    C_best, cv_rows = cv_select(Xtr, ytr)
    pipe = make_pipe(C_best).fit(Xtr, ytr)
    assert list(pipe.classes_) == sorted(CATS)
    pte = pipe.predict(Xte)
    ev = eval_block(yte, pte, CATS)
    no_ov = use.loc[i_te, "overlap"].isna().to_numpy()
    ev_noov = eval_block(yte[no_ov], pte[no_ov], CATS)
    proba_te = pipe.predict_proba(Xte)
    out["classifier"] = {"C_grid": C_GRID, "cv": cv_rows, "C_best": C_best,
                         "n_features": int(len(pipe.named_steps["tfidf"].vocabulary_)),
                         "heldout": {k: v for k, v in ev.items() if k != "confusion"},
                         "heldout_confusion": ev["confusion"].to_dict(),
                         "heldout_no_overlap": {k: v for k, v in ev_noov.items() if k != "confusion"},
                         "heldout_share": {"label": {c: float((yte == c).mean()) for c in CATS},
                                           "pred_argmax": {c: float((pte == c).mean()) for c in CATS},
                                           "pred_proba_mean": {c: float(proba_te[:, list(pipe.classes_).index(c)].mean())
                                                               for c in CATS}}}
    gate = {}
    for c in CATS:
        pr = ev["per_cat"][c]["precision"]
        kp = ag["per_cat"][c]["kappa"]
        gate[c] = {"precision": pr, "kappa": kp, "precision_ok": bool(pr is not None and pr >= PRECISION_MIN),
                   "kappa_ok": bool(kp >= KAPPA_MIN), "pass": bool(pr is not None and pr >= PRECISION_MIN and kp >= KAPPA_MIN)}
    out["gate"] = gate

    # 3b. 조롱 대상 분류기
    mk = tr[(tr["split"] != "excluded") & tr["tgt_train"].isin(TGTS)].copy()
    mk["X"] = [feat_text(s) for s in mk["text_masked"]]
    mtr, mte = mk[mk["split"] == "train"], mk[mk["split"] == "heldout"]
    Ct_best, cvt_rows = cv_select(mtr["X"].tolist(), mtr["tgt_train"].to_numpy())
    tpipe = make_pipe(Ct_best).fit(mtr["X"].tolist(), mtr["tgt_train"].to_numpy())
    tev = eval_block(mte["tgt_train"].to_numpy(), tpipe.predict(mte["X"].tolist()), TGTS)
    out["target_classifier"] = {"n_train": int(len(mtr)), "n_heldout": int(len(mte)),
                                "train_by_target": mtr["tgt_train"].value_counts().reindex(TGTS).fillna(0).astype(int).to_dict(),
                                "n_target_disagree_excluded": int((tr["tgt_train"] == "불일치").sum()),
                                "cv": cvt_rows, "C_best": Ct_best,
                                "heldout": {k: v for k, v in tev.items() if k != "confusion"},
                                "heldout_confusion": tev["confusion"].to_dict(),
                                "note": "떼어 둔 MOCK 단위는 정답 MOCK 단위다(1단 분류기의 MOCK 오분류는 따로 반영되지 않는다)"}

    # 모델 저장·지문
    md = EMO / "model"
    md.mkdir(exist_ok=True)
    files = {"cat_vectorizer.joblib": pipe.named_steps["tfidf"], "cat_logreg.joblib": pipe.named_steps["lr"],
             "target_vectorizer.joblib": tpipe.named_steps["tfidf"], "target_logreg.joblib": tpipe.named_steps["lr"]}
    fp = {}
    for nm, obj in files.items():
        joblib.dump(obj, md / nm)
        fp[nm] = sha(md / nm)
    card = {"made": out["made"], "spec": out["spec"], "sklearn": sklearn.__version__, "seed": SEED,
            "classes": list(pipe.classes_), "target_classes": list(tpipe.classes_),
            "tfidf": {"analyzer": "char", "ngram_range": [2, 4], "sublinear_tf": True, "min_df": 2, "lowercase": True},
            "placeholder_map": {k: f"U+{ord(v):04X}" for k, v in PH.items()},
            "logreg": {"solver": "lbfgs", "penalty": "l2", "multinomial": True, "class_weight": None, "max_iter": 5000,
                       "C": C_best, "target_C": Ct_best},
            "C_grid": C_GRID, "cv": "StratifiedKFold(5, shuffle, random_state=SEED), macro-F1, 동점이면 작은 C",
            "split": "범주 층화 train_test_split(test_size=0.2, random_state=SEED), 배포 모형 = 80% 학습분",
            "mask": "emotion/code/sample_a.py unit_text (글 제목 + 본문 600자, 댓글 400자)",
            "sha256": fp, "code_sha256": {"emotion_batchA.py": sha(Path(__file__)), "sample_a.py": sha(HERE / "sample_a.py")}}
    (md / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), "utf-8")
    out["model_sha256"] = fp
    out["model_card_sha256"] = sha(md / "model_card.json")

    # 학습 표 저장
    tr_out = tr[["id", "kind", "utype", "unit_idx", "shard", "overlap", "cat", "cat_2", "cat_train", "mock_target",
                 "mock_target_2", "tgt_train", "conf", "split", "text_masked"]].copy()
    tr_out["overlap"] = tr_out["overlap"].astype("Int64")
    tr_out.to_parquet(EMO / "train_set.parquet", index=False)

    # 4. 전체 적용
    t1 = datetime.now()
    Xall = mask_all(units)
    proba = pipe.predict_proba(Xall)
    cls = list(pipe.classes_)
    pred = pd.DataFrame({"unit_idx": np.arange(len(units)), "no": units["no"].to_numpy(),
                         "cid": units["cid"].to_numpy(), "utype": units["utype"].to_numpy(),
                         "win_kr": units["win_kr"].to_numpy(), "win_u": units["win_u"].to_numpy()})
    for c in CATS:
        pred[f"p_{c}"] = proba[:, cls.index(c)].astype("float32")
    pred["cat"] = np.array(cls)[proba.argmax(axis=1)]
    is_m = pred["cat"].eq("MOCK").to_numpy()
    tcls = list(tpipe.classes_)
    tp = np.full((len(units), len(TGTS)), np.nan, dtype="float32")
    if is_m.any():
        pm = tpipe.predict_proba([x for x, b in zip(Xall, is_m) if b])
        for j, t in enumerate(TGTS):
            tp[is_m, j] = pm[:, tcls.index(t)]
    for j, t in enumerate(TGTS):
        pred[f"p_tgt_{t}"] = tp[:, j]
    pred["mock_target"] = None
    pred.loc[is_m, "mock_target"] = np.array(TGTS)[np.nanargmax(tp[is_m], axis=1)]
    smap = dict(zip(tr["unit_idx"], tr["split"]))
    pred["sampleA_split"] = pred["unit_idx"].map(smap).fillna("not_sample")
    pred.to_parquet(EMO / "pred_units.parquet", index=False)
    out["apply"] = {"n_units": int(len(pred)), "seconds": round((datetime.now() - t1).total_seconds(), 1),
                    "pred_share_all_units": pred["cat"].value_counts(normalize=True).reindex(CATS).fillna(0).to_dict(),
                    "pred_share_by_utype": {k: g["cat"].value_counts(normalize=True).reindex(CATS).fillna(0).to_dict()
                                            for k, g in pred.groupby("utype")},
                    "mock_target_share": pred.loc[is_m, "mock_target"].value_counts(normalize=True).reindex(TGTS)
                    .fillna(0).to_dict()}
    # 표본 단위에서 모형 예측과 라벨이 같은지(학습분은 낙관적, 떼어 둔 분과 같은 값이어야 함)
    chk = pred.loc[pred["unit_idx"].isin(tr.loc[tr["split"] == "heldout", "unit_idx"])].merge(
        tr[["unit_idx", "cat_train"]], on="unit_idx")
    out["apply"]["heldout_reproduced_accuracy"] = float((chk["cat"] == chk["cat_train"]).mean())
    assert abs(out["apply"]["heldout_reproduced_accuracy"] - ev["accuracy"]) < 1e-9, "적용 단계 가림·특징이 학습과 다르다"

    # 창
    wk = pd.read_parquet(RES / "windows_kr.parquet")
    wu = pd.read_parquet(RES / "windows_us.parquet")
    W = pd.concat([window_table("kr", units, pred, wk), window_table("us", units, pred, wu)], ignore_index=True)
    W.to_parquet(EMO / "window_emotion.parquet", index=False)
    summ = {}
    for mkt, g in W.groupby("market"):
        v = g[g["valid"]]
        cols = ([f"r_all_{c}" for c in CATS] + [f"r_emo_{c}" for c in EMO6] + [f"ash_{c}" for c in CATS]
                + ["hype_index", "entropy_emo", "entropy_emo_norm"] + [f"mock_share_{t}" for t in TGTS]
                + ["mock_bear_minus_bull", "n_pair_all", "n_pair_emo", "n_auth", "n_mock_bear", "n_mock_bull"])
        summ[mkt] = {"n_windows": int(len(g)), "n_valid": int(len(v)), "n_over96": int(g["over96"].fillna(False).sum()),
                     "first_D": str(pd.Timestamp(g["D"].min()).date()), "last_D": str(pd.Timestamp(g["D"].max()).date()),
                     "stats": {c: {"mean": float(v[c].mean()), "sd": float(v[c].std(ddof=1)), "median": float(v[c].median()),
                                   "min": float(v[c].min()), "max": float(v[c].max())} for c in cols}}
    out["windows"] = summ

    # 5. 검수지
    items, rk = review_sample(units, pred, key)
    (EMO / "review_200.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    rk.to_parquet(EMO / "review_key.parquet", index=False)
    out["review"] = {"n": len(items), "by_kind": rk["kind"].value_counts().to_dict(),
                     "model_cat": rk["model_cat"].value_counts().reindex(CATS).fillna(0).astype(int).to_dict(),
                     "excluded_days": sorted(str(d) for d in CRASH_DAYS),
                     "frame": "sample_a.load_frame()과 같은 틀에서 급락일·묶음 A 표본 단위를 뺀 뒤 글 100·댓글 100 단순 무작위"}

    out["files"] = {p: sha(EMO / p) for p in ["pred_units.parquet", "window_emotion.parquet", "review_200.json",
                                              "review_key.parquet", "train_set.parquet"]}
    out["seconds_total"] = round((datetime.now() - t0).total_seconds(), 1)
    (EMO / "emotion_batchA_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: out[k] for k in ("train", "gate", "model_sha256", "apply", "review", "seconds_total")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
