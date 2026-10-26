"""감정 범주 묶음 A 감사 (가격 연결 전). emotion/ 아래 코드·산출물을 다시 계산해 다섯 가지를 확인한다.

① 가격·지수·수익 자료를 읽거나 만든 곳이 없는가   → 코드 정적 점검 + 모든 parquet·json 열 이름 점검
② 라벨러가 다른 조각·sample_key를 열었는가          → 라벨 파일 순서·이중 라벨 일치 패턴(작업 기록 점검은 audit.md)
③ 라벨이 키워드 규칙으로 기계 생성된 흔적           → 낱말 순도, 얕은 규칙 나무, 조각 안 위치별 쏠림, 같은 글 일관성
④ 떼어 둔 20%가 학습·C 선택에 섞였는가              → 나눔 재현, CV 겹이 학습분 안인지, 벡터라이저 어휘 재현, 평가 재현, 같은 글 중복
⑤ 창 비율이 마감 전 단위만 쓰고 작성자당 1회인가    → window_emotion.parquet을 독립 재계산해 대조

가격·지수·수익 자료는 읽지 않는다. 기존 산출물(모델·창 표·보고서)은 고치지 않고 읽기만 한다.
산출: emotion/audit_batchA.json
실행: py audit_batchA.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).resolve().parent
EMO = HERE.parent
ANT = EMO.parent
RES = ANT / "interim" / "results"
sys.path.insert(0, str(HERE))
import emotion_batchA as EB  # noqa: E402  (feat_text·make_pipe·상수만 쓴다. main은 부르지 않는다)

SEED = EB.SEED
CATS, EMO6, TGTS = EB.CATS, EB.EMO6, EB.TGTS
PRICE_COL_RX = re.compile(r"(price|close_px|open_px|ohlc|kospi|kosdaq|nasdaq|sox|qqq|spx|vix|ret_?\d|^r\d+$|^rv\d+|"
                          r"return|수익|종가|시가|고가|저가|지수|hl$|^g_d$|mdd)", re.I)
ALLOWED_COLS = {"close_kst", "early_close_D", "early_close_P", "pre_close_kr", "pre_close_u", "hype_index"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- ①

def check_price() -> dict:
    code = {}
    read_rx = re.compile(r"read_parquet\(|read_csv\(|read_text\(|json\.load\(|open\(|yf\.|yfinance|download\(|requests|urllib")
    for p in sorted(HERE.glob("*.py")):
        if p.name == Path(__file__).name:
            continue
        hits = [f"{i}: {ln.strip()[:140]}" for i, ln in enumerate(p.read_text("utf-8").splitlines(), 1)
                if read_rx.search(ln)]
        imports = sorted({m.group(1) for m in re.finditer(r"^\s*(?:import|from)\s+([\w.]+)", p.read_text("utf-8"), re.M)})
        code[p.name] = {"read_lines": hits, "imports": imports}
    cols = {}
    for p in sorted(list(EMO.glob("*.parquet")) + [RES / "units.parquet", RES / "windows_kr.parquet",
                                                    RES / "windows_us.parquet"]):
        names = pq.read_schema(p).names
        sus = [c for c in names if PRICE_COL_RX.search(c) and c not in ALLOWED_COLS]
        cols[str(p.relative_to(ANT)).replace("\\", "/")] = {"n_cols": len(names), "suspicious": sus}
    js = {}
    for p in sorted(EMO.glob("*.json")) + sorted((EMO / "model").glob("*.json")):
        txt = p.read_text("utf-8")
        js[str(p.relative_to(EMO)).replace("\\", "/")] = {
            "price_data_read_flag": re.findall(r'"price_data_read":\s*(\w+)', txt),
            "price_words": len(re.findall(r"kospi|yfinance|ohlc|종가|수익률", txt, re.I))}
    return {"code": code, "parquet_columns": cols, "json": js}


# ---------------------------------------------------------------- ②

def load_labels():
    P = {k: json.loads((EMO / "labels" / f"primary_{k:02d}.json").read_text("utf-8")) for k in range(1, 11)}
    S = {k: json.loads((EMO / "labels" / f"second_{k}.json").read_text("utf-8")) for k in range(1, 4)}
    SH = {k: json.loads((EMO / "shards" / f"shard_{k:02d}.json").read_text("utf-8")) for k in range(1, 11)}
    OV = {k: json.loads((EMO / "shards" / f"overlap_{k}.json").read_text("utf-8")) for k in range(1, 4)}
    return P, S, SH, OV


def check_labeler(P, S, SH, OV, key: pd.DataFrame) -> dict:
    out = {"order_matches_own_file": {}, "label_fields": sorted({k for v in P.values() for r in v for k in r})}
    for k in P:
        out["order_matches_own_file"][f"primary_{k:02d}"] = [r["id"] for r in P[k]] == [u["id"] for u in SH[k]]
    for k in S:
        out["order_matches_own_file"][f"second_{k}"] = [r["id"] for r in S[k]] == [u["id"] for u in OV[k]]
    prim = {r["id"]: r for v in P.values() for r in v}
    per = {}
    for k, rows in S.items():
        a = [prim[r["id"]] for r in rows]
        same = lambda f: float(np.mean([x[f] == y[f] for x, y in zip(a, rows)]))  # noqa: E731
        full = float(np.mean([all(x[f] == y[f] for f in ("cat", "mock_target", "stance", "px", "conf"))
                              for x, y in zip(a, rows)]))
        from_shard = Counter(int(key.set_index("id").loc[r["id"], "shard"]) for r in rows)
        per[f"second_{k}"] = {"n": len(rows), "cat": same("cat"), "stance": same("stance"), "px": same("px"),
                              "conf": same("conf"), "all_5_fields": full,
                              "from_shard": {str(s): n for s, n in sorted(from_shard.items())}}
    out["second_vs_primary"] = per
    # 같은 라벨러가 겹침 단위를 다시 매긴 꼴(복사)이면 all_5_fields가 1에 가깝다. 같은 AI 두 번의 일치 수준과 비교한다
    return out


# ---------------------------------------------------------------- ③

FRAME_TOKENS = {"제목", "본문", "없음", "내용", "생략"}  # 가림 틀이 붙인 글자(제목: / 본문: / (없음) / …(뒤 생략))


def tokens(s: str) -> set[str]:
    s = EB.feat_text(s)
    return set(re.findall(r"[가-힣A-Za-z]{2,}|ㅋ{2,}|ㅠ{2,}|ㅜ{2,}|ㅎ{2,}|[!?]{2,}", s)) - FRAME_TOKENS


def check_keyword(train: pd.DataFrame, P, SH) -> dict:
    t = train.copy()
    lab = t["cat"].to_numpy()
    toks = [tokens(s) for s in t["text_masked"]]
    df = defaultdict(Counter)
    for tk, c in zip(toks, lab):
        for w in tk:
            df[w][c] += 1
    pure = []
    for w, cnt in df.items():
        n = sum(cnt.values())
        if n >= 10:
            c, k = cnt.most_common(1)[0]
            if c != "NEUT":
                pure.append({"token": w, "n": n, "top_cat": c, "purity": k / n})
    pure.sort(key=lambda r: (-r["purity"], -r["n"]))
    # 범주별: 그 범주 라벨을 가장 많이 덮는 낱말 하나가 덮는 몫(키워드로 매겼다면 소수 낱말이 거의 전부를 덮는다)
    cover = {}
    for c in EMO6:
        idx = [i for i, x in enumerate(lab) if x == c]
        best = max(((w, sum(1 for i in idx if w in toks[i])) for w in df), key=lambda r: r[1])
        no_tok = sum(1 for i in idx if not toks[i])
        cover[c] = {"n": len(idx), "best_token": best[0], "best_cover": best[1] / max(1, len(idx)),
                    "precision_of_best": df[best[0]][c] / sum(df[best[0]].values()), "units_without_tokens": no_tok}
    # 얕은 낱말 규칙 나무(깊이 3·6) 5겹 정확도: 키워드 규칙이면 깊이가 얕아도 거의 완벽히 맞힌다
    cv = CountVectorizer(analyzer=lambda s: list(tokens(s)), binary=True, min_df=3)
    X = cv.fit_transform(t["text_masked"])
    tree = {}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for d in (3, 6, 12):
        sc = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), X, lab, cv=skf)
        tree[f"depth_{d}"] = {"acc_mean": float(sc.mean()), "acc_sd": float(sc.std(ddof=1))}
    neut = float((lab == "NEUT").mean())
    # 조각 안 위치(4분위)별 NEUT 몫·확신도 3 몫: 뒤로 갈수록 대충 매긴 흔적(한 범주 연속)이 있는가
    pos = []
    for k, rows in P.items():
        for j, r in enumerate(rows):
            pos.append({"shard": k, "q": j // 50 + 1, "neut": r["cat"] == "NEUT", "conf3": r["conf"] == 3, "cat": r["cat"]})
    pos = pd.DataFrame(pos)
    by_q = pos.groupby("q")[["neut", "conf3"]].mean().round(3).to_dict()
    runs = []
    for k, g in pos.groupby("shard"):
        c = g["cat"].tolist()
        best, cur = 1, 1
        for a, b in zip(c, c[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        runs.append(best)
    # 무작위로 섞었을 때 가장 긴 같은 범주 연속의 기대(같은 조각 범주 구성 유지, 1,000번)
    rng = np.random.default_rng(SEED)
    sim = []
    for k, g in pos.groupby("shard"):
        c = g["cat"].to_numpy()
        m = []
        for _ in range(1000):
            x = rng.permutation(c)
            best, cur = 1, 1
            for a, b in zip(x, x[1:]):
                cur = cur + 1 if a == b else 1
                best = max(best, cur)
            m.append(best)
        sim.append(float(np.quantile(m, 0.95)))
    # 같은 가린 글이 여러 번 뽑혔을 때 라벨이 같은가
    dup = t.groupby("text_masked")["cat"].agg(["size", "nunique"])
    dup = dup[dup["size"] > 1]
    return {"pure_tokens_top": pure[:25], "n_pure_tokens_ge_0.9": sum(r["purity"] >= 0.9 for r in pure),
            "n_tokens_n_ge_10_nonneut_top": len(pure), "category_cover": cover,
            "word_tree_cv": tree, "majority_baseline": neut,
            "position_quartile": by_q, "longest_same_cat_run_by_shard": runs,
            "shuffle_95pct_longest_run_by_shard": sim,
            "duplicate_texts": {"n_groups": int(len(dup)), "n_units": int(dup["size"].sum()),
                                "groups_with_conflicting_labels": int((dup["nunique"] > 1).sum())}}


# ---------------------------------------------------------------- ④

def check_split(train: pd.DataFrame, res: dict) -> dict:
    t = train.copy()
    ok = t["cat_train"] != "불일치"
    use = t[ok].copy()
    i_tr, i_te = train_test_split(use.index.to_numpy(), test_size=0.2, stratify=use["cat_train"], random_state=SEED)
    split_same = (set(t.index[t["split"] == "train"]) == set(i_tr)) and (set(t.index[t["split"] == "heldout"]) == set(i_te))
    tr, te = t.loc[i_tr], t.loc[i_te]
    disjoint_units = not (set(tr["unit_idx"]) & set(te["unit_idx"]))
    disjoint_ids = not (set(tr["id"]) & set(te["id"]))
    # CV 겹: emotion_batchA.cv_select와 같은 방식으로 다시 만들어 모든 겹 색인이 학습분 안인지
    Xtr = [EB.feat_text(s) for s in tr["text_masked"]]
    ytr = tr["cat_train"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
    fold_ok = all(max(np.max(a), np.max(b)) < len(Xtr) for a, b in folds)
    fold_units = set()
    for a, b in folds:
        fold_units |= set(tr["unit_idx"].to_numpy()[b])
    cv_uses_only_train = fold_ok and fold_units == set(tr["unit_idx"])
    # 저장된 벡터라이저 어휘가 '학습 80%만으로 맞춘 것'과 같은지(떼어 둔 글이 어휘·idf에 들어갔는지)
    vec = joblib.load(EMO / "model" / "cat_vectorizer.joblib")
    lr = joblib.load(EMO / "model" / "cat_logreg.joblib")
    v_tr = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2).fit(Xtr)
    Xall_ = [EB.feat_text(s) for s in pd.concat([tr, te])["text_masked"]]
    v_all = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2).fit(Xall_)
    vocab_eq_train = set(vec.vocabulary_) == set(v_tr.vocabulary_)
    idf_eq_train = vocab_eq_train and bool(np.allclose(
        np.array([vec.idf_[vec.vocabulary_[w]] for w in sorted(vec.vocabulary_)]),
        np.array([v_tr.idf_[v_tr.vocabulary_[w]] for w in sorted(vec.vocabulary_)])))
    vocab_eq_all = set(vec.vocabulary_) == set(v_all.vocabulary_)
    # 평가 재현: 저장 모형으로 떼어 둔 391단위를 다시 예측
    Xte = [EB.feat_text(s) for s in te["text_masked"]]
    pte = lr.predict(vec.transform(Xte))
    acc = float((pte == te["cat_train"].to_numpy()).mean())
    rep = res["classifier"]["heldout"]["accuracy"]
    prec = {}
    for c in CATS:
        m = pte == c
        prec[c] = (int((te["cat_train"].to_numpy()[m] == c).sum()), int(m.sum()))
    # 같은 가린 글(또는 공백만 다른 글)이 학습과 떼어 둔 쪽에 함께 있는가
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    trn = Counter(norm(s) for s in tr["text_masked"])
    dup_te = [i for i, s in zip(te.index, te["text_masked"]) if trn.get(norm(s))]
    dup_detail = te.loc[dup_te, ["id", "kind", "cat_train"]].assign(
        train_labels=[sorted(set(tr.loc[tr["text_masked"].map(norm) == norm(s), "cat_train"]))
                      for s in te.loc[dup_te, "text_masked"]],
        text=[s[:60] for s in te.loc[dup_te, "text_masked"]])
    pred_dup = lr.predict(vec.transform([EB.feat_text(s) for s in te.loc[dup_te, "text_masked"]])) if dup_te else []
    acc_wo_dup = float((pte[~te.index.isin(dup_te)] == te["cat_train"].to_numpy()[~te.index.isin(dup_te)]).mean())
    # 조롱 대상: 학습분 MOCK만으로 학습했는가
    mk = t[(t["split"] != "excluded") & t["tgt_train"].isin(TGTS)]
    tvec = joblib.load(EMO / "model" / "target_vectorizer.joblib")
    tv_tr = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2).fit(
        [EB.feat_text(s) for s in mk.loc[mk["split"] == "train", "text_masked"]])
    # 파일 지문
    fp_ok = {k: sha(EMO / "model" / k) == v for k, v in res["model_sha256"].items()}
    files_ok = {k: sha(EMO / k) == v for k, v in res["files"].items()}
    return {"split_reproduced": bool(split_same), "n_train": int(len(tr)), "n_heldout": int(len(te)),
            "disjoint_unit_idx": disjoint_units, "disjoint_id": disjoint_ids,
            "cv_folds_only_train": bool(cv_uses_only_train),
            "vectorizer_vocab_equals_train_only_fit": vocab_eq_train, "vectorizer_idf_equals_train_only_fit": idf_eq_train,
            "vectorizer_vocab_equals_train_plus_heldout_fit": vocab_eq_all,
            "n_vocab_saved": len(vec.vocabulary_), "n_vocab_train_only": len(v_tr.vocabulary_),
            "n_vocab_train_plus_heldout": len(v_all.vocabulary_),
            "target_vocab_equals_train_only_fit": set(tvec.vocabulary_) == set(tv_tr.vocabulary_),
            "heldout_accuracy_recomputed": acc, "heldout_accuracy_reported": rep,
            "heldout_precision_recomputed": {c: f"{k}/{n}" for c, (k, n) in prec.items()},
            "heldout_dup_of_train_text": {"n": len(dup_te), "pred_correct": int(sum(p == y for p, y in zip(
                pred_dup, te.loc[dup_te, "cat_train"]))), "accuracy_without_dups": acc_wo_dup,
                "rows": dup_detail.to_dict(orient="records")},
            "model_sha_ok": fp_ok, "output_sha_ok": files_ok}


# ---------------------------------------------------------------- ⑤

def check_windows() -> dict:
    u = pd.read_parquet(RES / "units.parquet", columns=["a", "utype", "dt_kst", "dt_conf", "win_kr", "win_u",
                                                        "pwin_kr", "pwin_u", "pre_close_kr", "pre_close_u"])
    pred = pd.read_parquet(EMO / "pred_units.parquet", columns=["unit_idx", "cat", "mock_target", "win_kr", "win_u"])
    assert (pred["unit_idx"].to_numpy() == np.arange(len(u))).all()
    assert pred["win_kr"].equals(u["win_kr"]) and pred["win_u"].equals(u["win_u"])
    W = pd.read_parquet(EMO / "window_emotion.parquet")
    out = {}
    for mkt, key, pw, pc, wf in (("kr", "win_kr", "pwin_kr", "pre_close_kr", "windows_kr.parquet"),
                                 ("us", "win_u", "pwin_u", "pre_close_u", "windows_us.parquet")):
        wins = pd.read_parquet(RES / wf)
        wm = wins[(wins["variant"] == "main") & wins["complete"]]
        close = dict(zip(wm["win_id"], wm["close_kst"]))
        opn = dict(zip(wm["win_id"], wm["open_kst"]))
        d = pd.DataFrame({"win": u[key], "a": u["a"], "cat": pred["cat"], "tgt": pred["mock_target"],
                          "dt": u["dt_kst"], "utype": u["utype"], "conf": u["dt_conf"], "pw": u[pw], "pc": u[pc]})
        d = d[d["win"].isin(close)]
        cl = d["win"].map(close)
        op = d["win"].map(opn)
        after_close = int((d["dt"] >= cl).sum())
        before_open = int((d["dt"] < op).sum())
        com = d[d["utype"] == "comment"]
        g = W[W["market"] == mkt].set_index("win_id")
        pairs = d.drop_duplicates(["win", "a", "cat"])
        cnt = pairs.groupby(["win", "cat"]).size().unstack(fill_value=0).reindex(index=g.index, columns=CATS, fill_value=0)
        tot = cnt.sum(axis=1)
        emo = cnt[EMO6].sum(axis=1)
        diff_pair = int((cnt.to_numpy() != g[[f"n_pair_{c}" for c in CATS]].to_numpy()).sum())
        v = g["valid"]
        r_emo = cnt[EMO6].div(emo, axis=0)
        max_abs = float(np.nanmax(np.abs(r_emo[v].to_numpy() - g.loc[v, [f"r_emo_{c}" for c in EMO6]].to_numpy())))
        hype = (r_emo["HYPE"] + r_emo["JOY"]) - (r_emo["DESP"] + r_emo["FEAR"])
        hype_abs = float(np.nanmax(np.abs(hype[v] - g.loc[v, "hype_index"])))
        # 작성자당 1회 확인: 쌍 수 ≤ 단위 수, 그리고 같은 작성자가 한 창 한 범주에 여러 단위를 쓴 경우가 실제로 줄었는가
        n_units_cat = d.groupby(["win", "cat"]).size().unstack(fill_value=0).reindex(index=g.index, columns=CATS, fill_value=0)
        dedup_removed = int((n_units_cat - cnt).to_numpy().sum())
        max_per_author = int(d.groupby(["win", "a", "cat"]).size().max())
        # 96시간 넘는 창(§2.1: 판정에서 빼고 따로 보고)을 뺀 요약과 넣은 요약(보고서 표는 넣은 쪽)
        o96 = g["over96"].fillna(False).astype(bool)
        cols = [f"r_emo_{c}" for c in EMO6] + ["hype_index", "entropy_emo_norm"]
        summ96 = {c: {"with_over96_mean": float(g.loc[v, c].mean()), "without_over96_mean": float(g.loc[v & ~o96, c].mean()),
                      "with_over96_sd": float(g.loc[v, c].std(ddof=1)), "without_over96_sd": float(g.loc[v & ~o96, c].std(ddof=1)),
                      "over96_value": [float(x) for x in g.loc[v & o96, c]]} for c in cols}
        out[mkt] = {
            "n_windows": int(len(g)), "n_valid": int(v.sum()), "invalid_windows": list(g.index[~v]),
            "invalid_features_all_nan": bool(g.loc[~v, [f"r_emo_{c}" for c in EMO6] + ["hype_index"]].isna().all().all()),
            "over96_windows": list(g.index[g["over96"].fillna(False).astype(bool)]),
            "units_counted": int(len(d)), "units_dt_ge_close": after_close, "units_dt_lt_open": before_open,
            "comments_counted": int(len(com)),
            "comments_latest_date": int((com["conf"] == "latest").sum()),
            "comments_on_earlier_window_post": int((com["pw"] != com["win"]).sum()),
            "comments_not_pre_close_of_parent_window": int((~com["pc"]).sum()),
            "pair_count_cells_differ": diff_pair, "r_emo_max_abs_diff": max_abs, "hype_max_abs_diff": hype_abs,
            "units_removed_by_author_dedup": dedup_removed, "max_units_same_author_window_cat": max_per_author,
            "pair_total_le_units": bool((tot.to_numpy() <= g["n_units"].to_numpy()).all()),
            "summary_over96_effect": summ96,
        }
    return out


def main() -> int:
    res = json.loads((EMO / "emotion_batchA_results.json").read_text("utf-8"))
    key = pd.read_parquet(EMO / "sample_key.parquet")
    train = pd.read_parquet(EMO / "train_set.parquet")
    P, S, SH, OV = load_labels()
    out = {"spec": "emotion/ 감사(가격 연결 전)", "price_data_read": False,
           "p1_price": check_price(),
           "p2_labeler": check_labeler(P, S, SH, OV, key),
           "p3_keyword": check_keyword(train, P, SH),
           "p4_split": check_split(train, res),
           "p5_windows": check_windows(),
           "code_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}}
    (EMO / "audit_batchA.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "p1_price"}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
