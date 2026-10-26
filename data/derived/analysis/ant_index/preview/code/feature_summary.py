"""미리보기(부록 9) 특징 목록·결측 요약. 가격은 읽지 않는다(창 자격 표지만 panel에서 가져온다).

판정용 사전 묶음은 측정 통과 묶음뿐이다: LL_US, LS, MC, ANG, KX, MEM, SEMI, TILT_US, TILT_KR, ALIAS_*, PX.
동결 코드가 함께 만든 감정 사전 열(E·F·S·Pess·ED·BULL/BEAR 점수·D_SD·단계표 등)은 파일에 남지만 쓰지 않는다.

산출: preview/results/feature_registry.json
실행: py feature_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
D0701 = pd.Timestamp("2026-07-01")

ALLOWED_STRATA = ["LL_US", "LS", "MC", "ANG", "KX", "MEM", "SEMI", "TILT_US", "TILT_KR", "ALIAS_SEC", "ALIAS_SKH",
                  "ALIAS_BOTH", "ALIAS_US", "PX"]
NOT_USED_PREFIX = ["E_", "F_", "S_", "S", "Pess", "ED_", "H05_ED", "H19_E", "H21_FO", "FO_", "CAP_", "HES_", "RES",
                   "LL_KR", "B", "D_SD", "D_AF", "semi_B", "semi_D_SD", "semi_D_AF", "mean_post_score", "dD3", "H11_X",
                   "stage10", "stage20", "SG", "H59_", "week_S_mean", "week_stage10", "H24_R", "H25_CR", "H26_PH",
                   "h06_", "h10_", "h49_", "h51_", "h53_", "s_day_S", "h37_"]

CELLS = [
    # (칸, 시장, 파일, 표(feat_topic만), 열, 역할)
    ("H03", "kr", "feat_activity", None, "A_rz8", "주 특징(번호판)"),
    ("H03", "kr", "feat_activity", None, "A_saved_rz8", "부호 확인(저장 글 수판, 작성자당 5건)"),
    ("H08", "kr", "feat_activity", None, "fight", "주 특징(잘리지 않은 글)"),
    ("H08", "kr", "feat_activity", None, "ln_n_comments_onposts", "통제(ln 댓글 수)"),
    ("H08", "kr", "feat_activity", None, "trunc_share", "통제(잘림 비율, 첫 줄 표지)"),
    ("H35", "kr", "feat_activity", None, "NW90_rz8", "주 특징(90일 이력 + rz8)"),
    ("H35", "kr", "feat_activity", None, "NW90", "재료(rz8 전)"),
    ("H35", "kr", "feat_activity", None, "NW30_rz8", "비등록 30일 이력판(설명용)"),
    ("H46", "us", "feat_topic", "US_D", "h46_dtilt", "주 특징(ΔTilt = 5창 평균 − 20창 전)"),
    ("H46", "us", "feat_topic", "US_D", "h46_tilt", "재료(Tilt)"),
    ("H31", "us", "feat_text", None, "LL_US_d20", "주 특징 재료(20창 변화, SOX 잔차화는 가격 단계)"),
    ("H31", "us", "feat_text", None, "LL_US_A_all", "재료(작성자 비율)"),
    ("H31", "us", "feat_text", None, "LL_US_noNVDL_d20", "강건(NVDL 사후 단어 뺌)"),
    ("공통", "kr", "feat_activity", None, "missing_rate", "통제(결번율)"),
    ("공통", "us", "feat_activity", None, "missing_rate", "통제(결번율)"),
]


def load() -> dict[str, pd.DataFrame]:
    fa = pd.read_parquet(RES / "feat_activity.parquet")
    ft = pd.read_parquet(RES / "feat_topic.parquet")
    fx = pd.read_parquet(RES / "feat_text.parquet")
    pk = pd.read_parquet(RES / "panel_kr.parquet", columns=["win_id", "judge_ok"])
    pu = pd.read_parquet(RES / "panel_us.parquet", columns=["win_id", "judge_ok"])
    ok = pd.concat([pk, pu]).set_index("win_id")["judge_ok"]
    return {"feat_activity": fa, "feat_topic": ft, "feat_text": fx, "judge_ok": ok}


def stat(df: pd.DataFrame, col: str, ok: pd.Series) -> dict:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    j = df["win_id"].map(ok).fillna(False).astype(bool).to_numpy()
    late = (pd.to_datetime(df["D"]) >= D0701).to_numpy()
    v = s.notna().to_numpy()
    Dv = pd.to_datetime(df.loc[v, "D"])
    return {"rows": int(len(s)), "valid": int(v.sum()), "valid_judge_ok": int((v & j).sum()),
            "rows_judge_ok": int(j.sum()), "rows_D_ge_0701": int(late.sum()),
            "valid_judge_ok_D_ge_0701": int((v & j & late).sum()),
            "missing_rate_judge_ok": round(float(1 - (v & j).sum() / max(j.sum(), 1)), 4),
            "first_valid_D": str(Dv.min().date()) if len(Dv) else None,
            "last_valid_D": str(Dv.max().date()) if len(Dv) else None}


def main() -> int:
    d = load()
    out = {"allowed_lexicon_strata": ALLOWED_STRATA,
           "not_used_columns_rule": "감정 사전(E·F 등 측정 실패 묶음)과 그 파생 열. 접두어: " + ", ".join(NOT_USED_PREFIX),
           "cells": [], "windows": {}}
    for cell, mk, src, table, col, role in CELLS:
        df = d[src]
        if src == "feat_topic":
            df = df[df["table"] == table]
        else:
            df = df[df["market"] == mk]
        rec = {"cell": cell, "market": mk, "file": f"{src}.parquet", "column": col, "role": role}
        rec.update(stat(df, col, d["judge_ok"]) if col in df.columns else {"error": "열 없음"})
        out["cells"].append(rec)
    for mk in ("kr", "us"):
        fa = d["feat_activity"][d["feat_activity"]["market"] == mk]
        out["windows"][mk] = {"rows": int(len(fa)), "judge_ok": int(fa["win_id"].map(d["judge_ok"]).sum()),
                              "first_D": str(pd.to_datetime(fa["D"]).min().date()),
                              "last_D": str(pd.to_datetime(fa["D"]).max().date()),
                              "D_ge_0701": int((pd.to_datetime(fa["D"]) >= D0701).sum())}
    (RES / "feature_registry.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    for r in out["cells"]:
        print(r["cell"], r["market"], r["column"], {k: r.get(k) for k in ("valid_judge_ok", "rows_judge_ok",
                                                                        "valid_judge_ok_D_ge_0701", "first_valid_D")})
    print(out["windows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
