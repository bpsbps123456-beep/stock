"""AI 라벨 검수지 (HYPOTHESES_addendum3.md 측정 절차 4단계 보완, 감사에서 추가).

왜: 기존 검수지(review_200.json)는 묶음 A 밖 단위에 '분류기' 범주를 붙였다(200 중 NEUT 181).
    그래서 사용자 검수가 3단계 AI 라벨(보고서의 '라벨로 본 비율' 답)을 확인하지 못한다.
    부록 3 순서는 3단계 라벨 → 4단계 사용자 검수 → 5단계 분류기다. 이 검수지는 AI 라벨 자체를 확인한다.
틀: 묶음 A 2,000단위(이미 AI 1차 라벨이 있음) 가운데 급락일(7/29~31, 10/13, 지시로 받은 날짜 목록)을 뺀 것.
뽑기: 글 100·댓글 100 단순 무작위(시드 sha256('20261017:REVIEW_LABEL_POST'/'..._COMMENT') 앞 8자리), 순서 섞기.
보여 주는 글: 라벨러가 본 가린 글 그대로(train_set.parquet text_masked = sample_a.unit_text).
산출: emotion/review_label_200.json(검수지), emotion/review_label_key.parquet(대응표, 사용자에게 주지 않아도 됨).
가격·지수·수익 자료는 읽지 않는다. 기존 산출물은 고치지 않는다.

실행: py review_labels.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMO = HERE.parent
sys.path.insert(0, str(HERE))
import sample_a as SA  # noqa: E402
import emotion_batchA as EB  # noqa: E402  (CATS·KO·CRASH_DAYS만 쓴다)

N_POST, N_COMMENT = 100, 100


def main() -> int:
    key = pd.read_parquet(EMO / "sample_key.parquet")
    tr = pd.read_parquet(EMO / "train_set.parquet")
    t = tr.merge(key[["id", "date_kst"]], on="id")
    assert len(t) == 2000
    crash = {str(d) for d in EB.CRASH_DAYS}
    f = t[~t["date_kst"].isin(crash)].reset_index(drop=True)
    picks = []
    for kind, n, nm in (("글", N_POST, "REVIEW_LABEL_POST"), ("댓글", N_COMMENT, "REVIEW_LABEL_COMMENT")):
        pos = f.index[f["kind"].eq(kind)].to_numpy()
        picks.extend(np.random.default_rng(SA._seed(nm)).choice(pos, size=n, replace=False))
    r = f.loc[picks].copy()
    r = r.iloc[np.random.default_rng(SA._seed("REVIEW_LABEL_SHUFFLE")).permutation(len(r))].reset_index(drop=True)
    rng = np.random.default_rng(SA._seed("REVIEW_LABEL_IDS"))
    ids: list[str] = []
    while len(ids) < len(r):
        x = f"L{int(rng.integers(0, 2**32)):08x}"
        if x not in ids:
            ids.append(x)
    r["rid"] = ids
    assert not r["date_kst"].isin(crash).any() and r["id"].is_unique
    items = [{"id": x.rid, "text": x.text_masked, "ai_cat": x.cat, "ai_cat_ko": EB.KO[x.cat],
              "ai_mock_target": (x.mock_target if x.cat == "MOCK" else None), "user_ok": None, "user_cat": None}
             for x in r.itertuples()]
    out = EMO / "review_label_200.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    k = r[["rid", "id", "kind", "utype", "unit_idx", "date_kst", "shard", "overlap", "cat", "cat_2", "cat_train",
           "mock_target", "conf", "split"]].rename(columns={"rid": "review_id", "id": "sample_id", "cat": "ai_cat"})
    k["overlap"] = k["overlap"].astype("Int64")
    k.to_parquet(EMO / "review_label_key.parquet", index=False)
    summ = {"n": len(items), "by_kind": r["kind"].value_counts().to_dict(),
            "ai_cat": r["cat"].value_counts().reindex(EB.CATS).fillna(0).astype(int).to_dict(),
            "frame_n": int(len(f)), "excluded_crash_units": int(len(t) - len(f)),
            "sha256": {"review_label_200.json": hashlib.sha256(out.read_bytes()).hexdigest(),
                       "review_label_key.parquet": hashlib.sha256((EMO / "review_label_key.parquet").read_bytes()).hexdigest()}}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
