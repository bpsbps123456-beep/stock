"""사건 연구 감사: W_U 라벨 층(예측·감정) 가중이 창 마감 뒤 틀 크기에 기대는 문제와 그 고친 판, SOXX 대비 다시 계산.

문제(features_all.design_weights, W_U): W_U 창 U(D)의 단위 가중 = 그 단위가 뽑힌 W_KR 층의 '틀 수 ÷ 표본 수'.
  U(D)(D 21:30 마감)에 단위를 대는 W_KR 창 K(D+1)은 D+1 08:00에 닫히므로(금요일·국내 휴일 앞날이면 다음 국내 거래일 08:00),
  U(D) 값이 U(D) 마감 뒤(D 21:30 ~ K(D+1) 마감)에 쓴 글·댓글 '수'에 기댄다(내용은 아님). 감정 층 하루 칸 가중도 같다.
  - 실시간 규칙(D+1 07:59)으로 보면 월~목 사건의 오프셋 0은 안이지만, K(D+1)이 D+1 08:00 뒤에 닫히는 날(금요일·국내 휴일 앞날)은
    규칙 밖 정보(주말·연휴 글 수)가 오프셋 0 값에 섞인다.
고친 판: W_U는 칸(W_U 창 × W_KR 창 × 글/댓글, 감정 층은 × 하루) 사후 층 가중 = 칸 안 틀 수 ÷ 칸 안 표본 수(틀 수는 U 창 안 단위만).
  나머지(표본·라벨·최소 표본 규칙·덮임·W_KR 값)는 그대로다. 표본 뽑기 자체가 K창 전체 틀에서 이뤄진 것(칸 표본 수가 우연히
  마감 뒤 글 양에 따라 달라짐)은 고칠 수 없어 남는 한계로 적는다.
산출: events/results/features_all_wu_cellw.parquet(W_U 라벨 층 열만 고친 사본, 원본은 그대로),
      events/results/audit_wu_weights.json, audit_wu_weights_tests.csv
실행: py -B audit_wu_weights.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_study as es  # noqa: E402
import features_all as FAL  # noqa: E402

RES = es.RESULTS
_orig_dw = FAL.design_weights
ALLOW_MISSING_CELLS = False      # 실제 자료에서는 빈 칸이 없어야 한다(있으면 멈춤)
MISSING_FILLED: list[int] = []


def design_weights_cell(s: pd.DataFrame, base_w: pd.Series, fr: pd.DataFrame, win_col: str):
    """W_KR은 원래 함수 그대로. W_U는 칸 사후 층 가중(칸 = W_U 창 × W_KR 창 × 글/댓글 [× 하루]), 칸 틀 수는 U 창 안 단위만."""
    if win_col != "win_u":
        return _orig_dw(s, base_w, fr, win_col)
    cells = ["win_u", "win_kr", "utype"] + (["day"] if "day" in s.columns and "w_day" in s.columns else [])
    N = fr.groupby(cells).size()
    n = s.groupby(cells).size()
    k = pd.MultiIndex.from_frame(s[cells])
    w = pd.Series(N.reindex(k).to_numpy(float) / n.reindex(k).to_numpy(float), index=s.index)
    if ALLOW_MISSING_CELLS:      # 교란 시험 전용: T 뒤 내용을 바꿔 틀에서 빠진 표본 단위(마감 > T 창)는 가중 1로 둔다
        MISSING_FILLED.append(int(w.isna().sum()))
        w = w.fillna(1.0)
    if w.isna().any() or (w <= 0).any():
        raise RuntimeError("칸 가중이 비었거나 0 이하")
    w = 0.5 * w / w.groupby([s[win_col], s["utype"]]).transform("sum")
    _, cov = _orig_dw(s, base_w, fr, win_col)          # 덮임 정의는 그대로(칸 표본 유무)
    return w, cov


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    F0, meta = es.load_features()
    FAL.design_weights = design_weights_cell
    try:
        F1, _, checks = FAL.build()
    finally:
        FAL.design_weights = _orig_dw
    lab_cols = [c for c in F0.columns if c.startswith(("pr_", "em_"))]
    F1 = F1.set_index("win_id")
    Fc = F0.copy().set_index("win_id")
    us = Fc.index[Fc["market"] == "us"]
    # 원본 W_KR 열은 그대로 두고 W_U 라벨 층만 바꾼다(W_KR 계산은 같은 함수라 같아야 한다)
    kr = Fc.index[Fc["market"] == "kr"]
    same_kr = {c: bool(np.allclose(Fc.loc[kr, c].astype(float), F1.loc[kr, c].astype(float), equal_nan=True, atol=1e-12))
               for c in lab_cols}
    for c in lab_cols:
        Fc.loc[us, c] = F1.loc[us, c].astype(float)
    Fc = Fc.reset_index()[F0.columns]
    Fc.to_parquet(RES / "features_all_wu_cellw.parquet", index=False)
    d = (Fc.set_index("win_id").loc[us, lab_cols].astype(float) - F0.set_index("win_id").loc[us, lab_cols].astype(float))
    grid = [f for f in es.grid_features(meta) if f in lab_cols]
    delta = {c: {"max_abs": float(np.nanmax(np.abs(d[c]))) if d[c].notna().any() else None,
                 "median_abs": float(np.nanmedian(np.abs(d[c]))) if d[c].notna().any() else None,
                 "windows_changed": int((np.abs(d[c]) > 1e-12).sum())} for c in grid}

    # K(D+1) 마감이 D+1 08:00 뒤인 U창(금요일·국내 휴일 앞날): 원래 가중이 실시간 규칙 밖 글 수에 기대는 창
    wk = F0[F0["market"] == "kr"][["win_id", "open_kst", "close_kst"]].copy()
    wu = F0[F0["market"] == "us"][["win_id", "D", "close_kst"]].copy()
    late = []
    for _, r in wu.iterrows():
        D = pd.Timestamp(r["D"])
        ov = wk[(pd.to_datetime(wk["open_kst"]) < pd.Timestamp(r["close_kst"])) &
                (pd.to_datetime(wk["close_kst"]) > pd.Timestamp(r["close_kst"]))]
        kc = pd.to_datetime(ov["close_kst"]).max() if len(ov) else pd.NaT
        late.append({"win_id": r["win_id"], "D": str(D.date()), "K_overlap_close": str(kc),
                     "beyond_D1_0800": bool(pd.notna(kc) and kc > D + pd.Timedelta(days=1, hours=8))})
    late = pd.DataFrame(late)

    # SOXX 대비 다시 계산(원본 특징 vs 고친 특징)
    ev, mode = es.confirmed_only_events(json.loads((RES / "events.json").read_text("utf-8")), es.CONFIRMED_ASOF)
    prices = pd.read_csv(RES / "prices_close.csv")
    P0 = es.build_panel(F0, meta, "us")
    P1 = es.build_panel(Fc, meta, "us")
    out = {"made": t0.isoformat(), "checks_kr_label_cols_unchanged": same_kr, "delta_us_grid_label_cols": delta,
           "us_windows_K_overlap_beyond_D1_0800": late[late["beyond_D1_0800"]]["win_id"].tolist(),
           "contrasts": {}}
    rows = []
    for c in es.load_contrasts(ev):
        if c.market != "us":
            continue
        close = es.price_series(prices, c.asset)
        r0 = es.run_contrast(c, P0, close, with_now=False)
        r1 = es.run_contrast(c, P1, close, with_now=False)
        t0_, t1_ = r0["table"], r1["table"]
        m = t0_.merge(t1_, on=["feature", "offset"], suffixes=("_orig", "_fix"))
        lab = m["bundle_orig"].isin(["예측 층", "감정 층"])
        evd = set(c.events["date"]) | set(c.fake_reps["date"])
        bad_used = sorted(set(late.loc[late["beyond_D1_0800"], "D"]) & evd)
        lt0, lt1 = r0["lead"], r1["lead"]
        ml = lt0.merge(lt1, on=["feature", "side"], suffixes=("_orig", "_fix"))
        out["contrasts"][c.name] = {
            "events_or_fakes_with_U_D_weight_beyond_D1_0800": bad_used,
            "family_cls": {"orig": r0["family"], "fix": r1["family"]},
            "family_lead": {"orig": r0["lead_family"], "fix": r1["lead_family"]},
            "candidates_cls": {"orig": sorted(f"{a}@{b:+d}" for a, b in t0_.loc[t0_["candidate"], ["feature", "offset"]].values),
                               "fix": sorted(f"{a}@{b:+d}" for a, b in t1_.loc[t1_["candidate"], ["feature", "offset"]].values)},
            "candidates_lead": {"orig": sorted(f"{a}({b})" for a, b in lt0.loc[lt0["candidate"], ["feature", "side"]].values),
                                "fix": sorted(f"{a}({b})" for a, b in lt1.loc[lt1["candidate"], ["feature", "side"]].values)},
            "label_cells_p_changed": int((~np.isclose(m.loc[lab, "p_orig"].astype(float), m.loc[lab, "p_fix"].astype(float),
                                                      equal_nan=True)).sum()),
            "label_cells": int(lab.sum()),
            "label_cells_p_lt_005": {"orig": int((m.loc[lab, "p_orig"] < 0.05).sum()), "fix": int((m.loc[lab, "p_fix"] < 0.05).sum())},
            "lead_label_cells_p_changed": int((~np.isclose(ml["p_orig"].astype(float), ml["p_fix"].astype(float),
                                                           equal_nan=True)).sum()),
            "nonlabel_cells_p_changed": int((~np.isclose(m.loc[~lab, "p_orig"].astype(float), m.loc[~lab, "p_fix"].astype(float),
                                                         equal_nan=True)).sum())}
        mm = m[lab & ((m["p_orig"] < 0.10) | (m["p_fix"] < 0.10))]
        for _, x in mm.iterrows():
            rows.append({"contrast": c.name, "feature": x["feature"], "offset": int(x["offset"]), "ba_orig": x["ba_orig"],
                         "ba_fix": x["ba_fix"], "p_orig": x["p_orig"], "p_fix": x["p_fix"],
                         "candidate_orig": bool(x["candidate_orig"]), "candidate_fix": bool(x["candidate_fix"])})
    pd.DataFrame(rows).to_csv(RES / "audit_wu_weights_tests.csv", index=False, encoding="utf-8-sig")
    out["seconds"] = (datetime.now() - t0).total_seconds()
    (RES / "audit_wu_weights.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("delta_us_grid_label_cols",)}, ensure_ascii=False,
                     indent=1, default=str)[:6000])
    print(json.dumps(delta, ensure_ascii=False))
    print(pd.DataFrame(rows).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
