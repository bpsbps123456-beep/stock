"""사건 연구(부록 13·14) 감사 ②③④: 12개 대비를 event_study 함수로 다시 돌려 조합별 산출과 대조하고, 후보 표에 실시간·누출·검수·
지식 기간 표시를 붙인다. 분석 로직·문턱·귀무·씨앗은 event_study 그대로(고치지 않고 부름).

보는 것
  ② 확정 뒤 정보: 가르기·앞섬이 쓰는 창(오프셋 −5~+1)이 사건 확정 시각 전에 닫히는가(사건·가짜 모두). '알아챔' 마감
     = W_KR(D+1) 마감(D+1 07:59). W_U 오프셋 0 = U(D)(D 21:30 마감)는 안, +1 = U(D+1)(D+1 21:30 마감)은 밖.
  ③ 누출 시험: 등록 규칙(오프셋 0·+1 (b) z 중앙값 사건 > 가짜면 의심, 값 부족도 의심)을 다시 계산해 조합 산출의 표시와 같은지,
     누출 의심 대비의 조회·추천 칸이 후보에서 빠졌는지. 조회·추천 원값은 수집 시점 누적값이라(소급 창은 사건 몇 주~몇 달 뒤)
     누출 시험을 통과해도 D+1 07:59에 같은 값을 잴 수 없다 → 조회·추천 칸 전부 '실시간 측정 불가' 표시
     (candidate_realtime = 코드 후보 + 알아챔 마감 안 + 조회·추천 아님).
  ④ 감정 층(부록 8 검수 실패) = 설명용, 후보 아님. 예측 층 칸은 쓰인 창 중 D < 2026-07-01(라벨 모델 지식 기간) 몫을 적는다.
산출: events/results/audit_candidates.json, audit_candidates.csv(후보·근접 칸 표시판)
실행: py -B audit_candidates.py
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

RES = es.RESULTS
KNOW_END = np.datetime64("2026-07-01", "D")
COMBO_TESTS = {"KOSPI_8": RES / "run_KOSPI_8_tests.csv", "KOSPI_12": RES / "run_KOSPI_12" / "tests.csv",
               "SKHYNIX_8": RES / "run_SKHYNIX_8" / "tests.csv", "SKHYNIX_12": RES / "run_SKHYNIX_12_tests.csv",
               "SOXX_8": RES / "run_SOXX_8_tests.csv", "SOXX_12": RES / "run_SOXX_12_tests.csv"}
COMBO_LEAD = {"KOSPI_8": RES / "run_KOSPI_8_lead.csv", "KOSPI_12": RES / "run_KOSPI_12" / "lead.csv",
              "SKHYNIX_8": RES / "run_SKHYNIX_8" / "lead.csv", "SKHYNIX_12": RES / "run_SKHYNIX_12_lead.csv",
              "SOXX_8": RES / "run_SOXX_8_lead.csv", "SOXX_12": RES / "run_SOXX_12_lead.csv"}
KIND_KO = {"bottom": "바닥", "top": "고점"}


def kind_of(df: pd.DataFrame, c: es.Contrast) -> pd.DataFrame:
    if "kind" in df.columns:
        return df[df["kind"] == c.kind]
    if "contrast" in df.columns:
        return df[df["contrast"].astype(str).str.endswith(c.kind) | df["contrast"].astype(str).eq(KIND_KO[c.kind])]
    raise KeyError("kind 열 없음")


def close_map(F: pd.DataFrame) -> dict:
    return dict(zip(F["win_id"], pd.to_datetime(F["close_kst"])))


def confirm_ts(asset: str, d: str) -> pd.Timestamp:
    """확정일 장 마감 시각(KST). KOSPI·하이닉스 15:30, SOXX 미국 16:00 ET → 다음 날 05:00/06:00 KST(보수적으로 05:00)."""
    if asset == "SOXX":
        return pd.Timestamp(d) + pd.Timedelta(hours=29)
    return pd.Timestamp(d) + pd.Timedelta(hours=15, minutes=30)


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    ev_raw = json.loads((RES / "events.json").read_text("utf-8"))
    ev, mode = es.confirmed_only_events(ev_raw, es.CONFIRMED_ASOF)
    F, meta = es.load_features()
    prices = pd.read_csv(RES / "prices_close.csv")
    panels = {m: es.build_panel(F, meta, m) for m in ("kr", "us")}
    cmap = close_map(F)
    out = {"made": t0.isoformat(), "mode": mode, "contrasts": {}}
    cand_rows = []
    for c in es.load_contrasts(ev):
        combo = f"{c.asset}_{c.threshold}"
        p = panels[c.market]
        r = es.run_contrast(c, p, es.price_series(prices, c.asset), with_now=False)
        tb, lt, leak = r["table"], r["lead"], r["leak"]
        rec = {"n_events": int(len(c.events)), "n_fake_reps": int(len(c.fake_reps)), "leak": leak}
        # --- ③ 조합 산출과 대조
        try:
            t2 = kind_of(pd.read_csv(COMBO_TESTS[combo]), c)
            m = tb.merge(t2[["feature", "offset", "p", "candidate", "leak_suspect"]], on=["feature", "offset"],
                         suffixes=("", "_combo"))
            rec["combo_tests_match"] = {
                "cells": int(len(m)),
                "p_equal": int(np.isclose(m["p"].astype(float), m["p_combo"].astype(float), equal_nan=True).sum()),
                "candidate_equal": int((m["candidate"].astype(bool) == m["candidate_combo"].astype(bool)).sum()),
                "leak_equal": int((m["leak_suspect"].astype(bool) == m["leak_suspect_combo"].astype(bool)).sum())}
            l2 = kind_of(pd.read_csv(COMBO_LEAD[combo]), c)
            ml = lt.merge(l2[["feature", "side", "p", "candidate"]], on=["feature", "side"], suffixes=("", "_combo"))
            rec["combo_lead_match"] = {
                "cells": int(len(ml)),
                "p_equal": int(np.isclose(ml["p"].astype(float), ml["p_combo"].astype(float), equal_nan=True).sum()),
                "candidate_equal": int((ml["candidate"].astype(bool) == ml["candidate_combo"].astype(bool)).sum())}
        except Exception as e:  # noqa: BLE001
            rec["combo_match_error"] = repr(e)
        # 후보 규칙 재확인: 조회·추천 칸은 누출 의심이면 후보가 아니다
        vw = tb["bundle"].eq(es.LEAK_BUNDLE)
        rec["rule_check"] = {
            "candidate_rule_holds": bool((tb["candidate"] == ((tb["p"] < es.ALPHA) & tb["status_feature"].eq("통과")
                                                               & ~(vw & leak["flag"]))).all()),
            "views_candidates_when_leak_flag": int((tb["candidate"] & vw).sum()) if leak["flag"] else 0,
            "emotion_candidates": int((tb["candidate"] & tb["bundle"].eq("감정 층")).sum()),
            "lead_emotion_candidates": int((lt["candidate"] & lt["bundle"].eq("감정 층")).sum())}
        # --- ② 확정 뒤 정보: 쓰인 창의 마감 vs 확정 시각
        piv = {pp["date"]: pp for pp in ev["assets"][c.asset]["thresholds"][c.threshold]["pivots"]}
        ev_pos = es.positions(p, c.events["win_id"])
        worst = []
        for d, pos in zip(c.events["date"], ev_pos):
            if pos < 0:
                continue
            ct = confirm_ts(c.asset, piv[d]["confirm_date"])
            cl1 = cmap.get(p.win_id[pos + 1]) if pos + 1 < len(p.win_id) else None
            worst.append({"date": d, "confirm_date": piv[d]["confirm_date"], "lag_td": piv[d]["confirm_lag_td"],
                          "off1_win": p.win_id[pos + 1] if pos + 1 < len(p.win_id) else None,
                          "off1_close": str(cl1), "off1_closes_before_confirm": bool(cl1 is not None and cl1 < ct),
                          "off1_close_le_D1_0800": bool(cl1 is not None and (
                              c.market == "kr" or cl1 <= pd.Timestamp(d) + pd.Timedelta(days=1, hours=8)))})
        rec["confirm_timing"] = {"events": len(worst),
                                 "off1_before_confirm": int(sum(w["off1_closes_before_confirm"] for w in worst)),
                                 "off1_within_D1_0759_rule": int(sum(w["off1_close_le_D1_0800"] for w in worst)),
                                 "rows": worst}
        # --- 후보·근접 칸 표시판(가르기: p < 0.10 통과·설명용 모두, 앞섬: p < 0.05)
        fk_pos = es.positions(p, c.fake_reps["win_id"])
        for _, row in tb[tb["p"] < 0.10].iterrows():
            k = int(row["offset"])
            fi = p.feats.index(row["feature"])
            idx = np.r_[ev_pos, fk_pos]
            lab = np.r_[np.ones(len(ev_pos), bool), np.zeros(len(fk_pos), bool)]
            ok = (idx >= 0) & (idx + k >= 0) & (idx + k < len(p.win_id))
            idx, lab = idx[ok] + k, lab[ok]
            fin = np.isfinite(p.Z[idx, fi])
            idx, lab = idx[fin], lab[fin]
            kn = p.D[idx] < KNOW_END
            mt = meta[row["feature"]]
            realtime = "안" if (c.market == "kr" or k <= 0) else "밖(U(D+1) 마감 D+1 21:30 > D+1 07:59)"
            flags = []
            if mt["bundle"] == "감정 층":
                flags.append("감정 층 검수 실패 → 설명용(후보 아님)")
            elif mt["status"] != "통과":
                flags.append(f"설명용({mt['leak']} 누출·{mt['note'][:20]})")
            if mt["bundle"] in ("예측 층", "감정 층"):
                flags.append(f"지식 기간 창 사건 {int((kn & lab).sum())}/{int(lab.sum())}·가짜 {int((kn & ~lab).sum())}/{int((~lab).sum())}")
            if mt["bundle"] == es.LEAK_BUNDLE:
                flags.append("누출 의심(후보 아님)" if leak["flag"] else "누출 시험 통과")
                flags.append("조회·추천 수치 = 수집 시점 누적값(확정 사건 창 첫 수집 중앙 137일 뒤) → D+1 07:59에 같은 값 측정 불가")
            if realtime != "안":
                flags.append("알아챔 마감 밖")
            cand_rows.append({"combo": combo, "kind": KIND_KO[c.kind], "analysis": "가르기", "feature": row["feature"],
                              "cell": f"@{k:+d}", "bundle": mt["bundle"], "status": mt["status"], "p": row["p"],
                              "ba": row["ba"], "candidate_code": bool(row["candidate"]), "realtime": realtime,
                              "candidate_realtime": bool(row["candidate"]) and realtime == "안" and
                                                    mt["bundle"] != es.LEAK_BUNDLE,
                              "knowledge_events": f"{int((kn & lab).sum())}/{int(lab.sum())}",
                              "knowledge_fakes": f"{int((kn & ~lab).sum())}/{int((~lab).sum())}",
                              "flags": "; ".join(flags)})
        for _, row in lt[lt["p"] < 0.05].iterrows():
            mt = meta[row["feature"]]
            flags = []
            if mt["bundle"] == "감정 층":
                flags.append("감정 층 검수 실패 → 설명용(후보 아님)")
            if mt["bundle"] in ("예측 층", "감정 층"):
                kn_e = int((p.D[ev_pos[ev_pos >= 0]] < KNOW_END).sum())
                flags.append(f"지식 기간 사건 {kn_e}/{int((ev_pos >= 0).sum())}")
            if mt["bundle"] == es.LEAK_BUNDLE:
                flags.append("누출 의심(후보 아님)" if leak["flag"] else "누출 시험 통과")
            cand_rows.append({"combo": combo, "kind": KIND_KO[c.kind], "analysis": "앞섬", "feature": row["feature"],
                              "cell": row["side"], "bundle": mt["bundle"], "status": mt["status"], "p": row["p"],
                              "ba": np.nan, "candidate_code": bool(row["candidate"]), "realtime": "안(사건 전 창)",
                              "candidate_realtime": bool(row["candidate"]) and mt["bundle"] != es.LEAK_BUNDLE,
                              "knowledge_events": "", "knowledge_fakes": "", "flags": "; ".join(flags)})
        rec["families"] = {"cls": r["family"], "lead": r["lead_family"]}
        out["contrasts"][c.name] = rec
    cr = pd.DataFrame(cand_rows)
    cr.to_csv(RES / "audit_candidates.csv", index=False, encoding="utf-8-sig")
    out["candidates"] = cr[cr["candidate_code"]].to_dict("records")
    out["seconds"] = (datetime.now() - t0).total_seconds()
    (RES / "audit_candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    for nm, rec in out["contrasts"].items():
        print(nm, "leak", rec["leak"]["flag"], rec.get("combo_tests_match"), rec.get("combo_lead_match"),
              rec["rule_check"], {k: v for k, v in rec["confirm_timing"].items() if k != "rows"})
    print(cr[cr["candidate_code"]][["combo", "kind", "analysis", "feature", "cell", "p", "realtime", "candidate_realtime",
                                    "flags"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
