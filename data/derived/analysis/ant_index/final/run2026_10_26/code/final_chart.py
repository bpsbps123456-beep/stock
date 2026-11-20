"""개미 지수 최종 그림 자료(2026-10-26).

읽기만 하는 입력:
  final/run2026_10_26/results/feat_activity.parquet, pred_windows_full.parquet, prices.parquet,
  judgment.json, recheck_judge_indep/compare_summary.json, audit_timing/results/audit_sens.json
  events/run2026_10_26/results/events.json, prices_close.csv, new_event_check.json
출력: final/run2026_10_26/results/final_chart.json
판정을 다시 돌리지 않는다. 숫자는 위 결과 파일에서 옮긴다.
"""
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # .../ant_index
FIN = ROOT / "final" / "run2026_10_26"
RES = FIN / "results"
EV = ROOT / "events" / "run2026_10_26" / "results"
OUT = RES / "final_chart.json"

NEW_FROM, NEW_TO = "2025-01-21", "2025-08-20"   # 부록 15 새 사건 기간
STUDY_FROM, CUTOFF = "2025-01-21", "2026-10-23"
BACK_FROM = "2026-06-01"                         # 판정 뒤 구간(H03·H08)
FRONT_FROM = "2026-01-01"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def num(v, nd=6):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def period_of(d):
    if d < NEW_FROM:
        return "pre"
    if d <= NEW_TO:
        return "new"
    return "old"


def segment_of(d):
    if d < FRONT_FROM:
        return "2025"
    if d < BACK_FROM:
        return "front"
    return "back"


def daily_kr():
    fa = pd.read_parquet(RES / "feat_activity.parquet")
    fa = fa[fa["market"] == "kr"].copy()
    pw = pd.read_parquet(RES / "pred_windows_full.parquet")
    px = pd.read_parquet(RES / "prices.parquet")
    k = px[px["ticker"] == "^KS11"].copy()
    k["date"] = pd.to_datetime(k["date"]).dt.strftime("%Y-%m-%d")
    bad = k["flag_flat_bar"].fillna(False) & k["flag_zero_volume"].fillna(False)
    k["lnHL"] = (k["high"] / k["low"]).apply(lambda v: math.log(v) if v and v > 0 else float("nan"))
    k.loc[bad, "lnHL"] = float("nan")          # judge 동결 규칙: 평평한 막대+거래량 0 → OHLC 결측
    kmap = k.set_index("date")[["close", "lnHL"]].to_dict("index")
    flat_days = sorted(k.loc[bad, "date"].tolist())

    pw = pw.set_index("win_id")
    rows = []
    for _, r in fa.sort_values("D").iterrows():
        d = pd.Timestamp(r["D"]).strftime("%Y-%m-%d")
        if d > CUTOFF:
            continue
        p = pw.loc[r["win_id"]] if r["win_id"] in pw.index else None
        kp = kmap.get(d, {})
        rows.append({
            "win_id": r["win_id"],
            "D": d,
            "kind": r["kind"],
            "length_h": num(r["length_h"], 2),
            "realtime": bool(r["realtime"]),
            "judge_ok": bool(r["judge_ok"]),
            "segment": segment_of(d),
            "posts_no": num(r["d_no"], 0),          # 글 번호 증가(지운 글 포함) = 글 수
            "posts_saved": int(r["n_posts"]),       # 저장된 글
            "posts_per_h": num(r["d_no_per_h"], 4),
            "comments_pre": int(r["n_comments_win"]),
            "A_rz8": num(r["A_rz8"], 4),            # H03 특징
            "P_bar": num(p["P_bar"], 4) if p is not None else None,   # 군중 예측(H66 특징)
            "P_share": num(p["P_share"], 4) if p is not None else None,
            "n_pred": int(p["n_pred"]) if p is not None and pd.notna(p["n_pred"]) else None,
            "knowledge_period": bool(p["knowledge_period"]) if p is not None else None,
            "kospi_close": num(kp.get("close"), 2),
            "kospi_lnHL": num(kp.get("lnHL"), 6),
        })
    return rows, flat_days


def closes():
    pc = pd.read_csv(EV / "prices_close.csv")
    out = {}
    for a, g in pc.groupby("asset"):
        g = g.sort_values("date")
        out[a] = [[d, num(c, 2)] for d, c in zip(g["date"], g["close"])]
    return out


def events():
    ev = json.loads((EV / "events.json").read_text(encoding="utf-8"))
    out = {}
    for a, ad in ev["assets"].items():
        out[a] = {"window": ad["window"], "thresholds": {}}
        for x, t in ad["thresholds"].items():
            piv = []
            for p in t["pivots"]:
                per = period_of(p["date"])
                piv.append({
                    "kind": "bottom" if p["kind"] == "low" else "top",
                    "date": p["date"], "close": num(p["close"], 2),
                    "confirm_date": p.get("confirm_date"), "confirm_lag_td": p.get("confirm_lag_td"),
                    "period": per, "in_study": per != "pre",
                })
            tail = t["tail_unconfirmed"]
            out[a]["thresholds"][x] = {
                "pivots": piv,
                "fake_bottoms": [{"date": f["date"], "run": f.get("run"), "period": period_of(f["date"])}
                                 for f in t["fake_bottoms"] if f["date"] >= STUDY_FROM],
                "fake_tops": [{"date": f["date"], "run": f.get("run"), "period": period_of(f["date"])}
                              for f in t["fake_tops"] if f["date"] >= STUDY_FROM],
                "held_fake_bottoms": [f["date"] for f in t["held_near_tail"]["fake_bottoms"]],
                "tail": {
                    "kind": "bottom" if tail["kind"] == "low" else "top",
                    "date": tail["date"], "close": num(tail["close"], 2), "status": tail["status"],
                    "confirm_level": num(tail["confirm_level"], 2),
                    "last_date": tail["last_date"], "last_close": num(tail["last_close"], 2),
                    "move_from_tail": num(tail["move_from_tail"], 4),
                    "amp_in": num(tail["amp_in"], 4), "prev_pivot_date": tail["prev_pivot_date"],
                },
            }
    return out


def judgment():
    j = json.loads((RES / "judgment.json").read_text(encoding="utf-8"))
    cs = json.loads((FIN / "recheck_judge_indep" / "compare_summary.json").read_text(encoding="utf-8"))
    final_cat = {
        "H03": "통과(취약)", "H08": "기각", "H70": "판정 불가", "H66": "기각(재계산은 판정 불가)",
    }
    why = {
        "H03": "등록 판정은 통과. 독립 재계산 주판은 메아리 위약으로 기각(비율 0.72), 72개 명세 중 통과 12. "
               "KOSPI 6/29 막대를 네이버 OHLC로 채우면 Holm p 0.0581로 판정 불가. X⊥판 p 0.084.",
        "H08": "뒤 구간 부호 반대. 재계산 72/72 기각(근거는 메아리 위약으로 다름).",
        "H70": "보정 p 0.684 미달. 재계산 72/72 판정 불가. 전방 트랙.",
        "H66": "등록 판정은 메아리 위약 기각. 재계산 주판은 판정 불가(Hodrick p 0.1003), 72개 명세 중 판정 불가 66. "
               "시점 감사의 자르기 시험에서 M0 미래 정보 섞임은 나오지 않음. 차이 원인은 미해결.",
    }
    cells = []
    for cid in j["judgment_cells_order"]:
        c = j["judgment_cells"][cid]
        rc = cs["cells"].get(cid, {}).get("recheck_main", {})
        cells.append({
            "id": cid, "label": c["label"], "fam": c["fam"], "y": c["y"], "h": c["h"],
            "n_back": c["n_back"], "est": num(c["est"], 4), "p_final": num(c["p_final"], 4),
            "p_adj": num(c["p_adj"], 4), "thr_local": num(c["thr_local"], 4),
            "est_front": num(c["est_front"], 4), "front_ok": c["front_ok"],
            "robust_ratio": {k: num(v, 3) for k, v in (c["robust_ratio"] or {}).items()},
            "placebo_xhat": num(c["placebo_est"], 4), "placebo_x": num(c["placebo_x_est"], 4),
            "placebo_flag": c["placebo_flag"],
            "xperp_est": num(c["xperp_est"], 4), "xperp_p_final": num(c["xperp_p_final"], 4),
            "mde": num(c["mde"], 3), "expected_effect": c["expected_effect"],
            "harness_verdict": c["verdict"],
            "recheck_verdict": rc.get("verdict"), "recheck_est": num(rc.get("est"), 4),
            "recheck_p_final": num(rc.get("p_final"), 4),
            "final_category": final_cat[cid], "note": why[cid],
        })
    sens = {
        "기준": {"H03": [0.469, 0.0194, "통과"], "H08": [-0.070, None, "기각"], "H70": [0.062, None, "판정 불가"], "H66": [-0.108, 0.381, "기각"]},
        "S1 FOMC 5회 추가": {"H03": [0.469, 0.0194], "H08": [-0.070], "H70": [0.062], "H66": [-0.107, 0.387]},
        "S2 국내 행사 당일 세션": {"H03": [0.457, 0.0387], "H08": [-0.072], "H70": [0.063], "H66": [-0.116, 0.348]},
        "S3 삼전 잠정실적일 추가": {"H03": [0.479, 0.0194], "H08": [-0.038], "H70": [0.055], "H66": [-0.104, 0.394]},
        "S4 S1+S2+S3": {"H03": [0.471, 0.0194], "H08": [-0.041], "H70": [0.063], "H66": [-0.107, 0.361]},
        "S6 KOSPI 6/29 네이버 OHLC": {"H03": [0.447, 0.0581, "판정 불가"], "H08": [-0.033], "H70": [0.112], "H66": [-0.151, 0.316]},
        "S7 결번율 통제 뺌": {"H03": [0.482, 0.0194], "H08": [-0.074], "H70": [0.049], "H66": [-0.110, 0.374]},
    }
    return {
        "rule": j["rule"],
        "cells": cells,
        "global_rotation": j["global_rotation"],
        "observed_pass_ids": j["observed_pass_ids"],
        "recheck_spec_matrix_72": {"H03": {"통과": 12, "기각": 50, "판정 불가": 10},
                                   "H08": {"기각": 72}, "H70": {"판정 불가": 72},
                                   "H66": {"판정 불가": 66, "기각": 4, "통과": 2}},
        "audit_sensitivity": {"format": "칸: [점추정, 보정 p(H03 Holm, H66 BH), 범주가 바뀌면 범주]", "rows": sens},
        "trading_use_addendum12": {
            "T1 통과": "방향이 아니라 '내일 크게 움직일 날' 경고. 비중·손절 폭 참고",
            "H66 통과": "군중 예측 반대 방향을 일주일 기울기로 참고(이번에는 통과 아님 → 쓰지 않음)",
            "통과 없음": "매매 신호로 쓰지 않고 시장 온도 화면으로만",
        },
    }


def event_summary():
    combos = {
        # 종류: [사건(새/옛), 가르기 관측/기대/p, 앞섬 관측/기대/p, 누출, 앞선 실행 가르기 관측/기대]
        "KOSPI_8": {"bottom": [9, 1, 8, 2, 1.71, 0.468, 1, 0.76, 0.460, "통과", "0/1.67"],
                    "top": [10, 1, 9, 0, 2.11, 1.000, 1, 1.50, 0.686, "통과", "0/1.83"]},
        "KOSPI_12": {"bottom": [5, 1, 4, 1, 1.75, 0.786, 2, 0.74, 0.198, "통과", "1/1.29"],
                     "top": [6, 1, 5, 2, 1.95, 0.560, 0, 1.49, 1.000, "통과", "4/1.74"]},
        "SKHYNIX_8": {"bottom": [22, 4, 18, 1, 2.22, 0.866, 1, 1.56, 0.720, "의심", "0/2.14"],
                      "top": [23, 5, 18, 7, 2.28, 0.029, 2, 1.63, 0.442, "통과", "4/2.15"]},
        "SKHYNIX_12": {"bottom": [12, 3, 9, 0, 2.24, 1.000, 3, 1.64, 0.235, "통과", "1/2.17"],
                       "top": [13, 4, 9, 1, 2.26, 0.871, 3, 1.56, 0.209, "통과", "3/2.09"]},
        "SOXX_8": {"bottom": [11, 3, 8, 3, 2.14, 0.347, 0, 1.68, 1.000, "의심", "3/1.81"],
                   "top": [12, 3, 9, 1, 2.27, 0.856, 0, 1.72, 1.000, "의심", "0/2.04"]},
        "SOXX_12": {"bottom": [5, 1, 4, 3, 2.13, 0.346, 2, 1.55, 0.385, "통과", "1/1.78"],
                    "top": [6, 1, 5, 1, 1.81, 0.785, 0, 1.36, 1.000, "통과", "0/1.74"]},
    }
    keys = ["n_events", "n_new", "n_old", "split_obs", "split_exp", "split_p",
            "lead_obs", "lead_exp", "lead_p", "leak_test", "prev_split_obs_exp"]
    tab = {c: {k: dict(zip(keys, v)) for k, v in d.items()} for c, d in combos.items()}
    nec = json.loads((EV / "new_event_check.json").read_text(encoding="utf-8"))
    return {
        "source": "events/run2026_10_26/results/run_*.md 0·5절",
        "combos": tab,
        "beyond_chance": ["SKHYNIX_8 top 가르기 (7 / 2.28, p 0.029; 검증 재계산 8 / 2.29, p 0.016)"],
        "bundles_tested": 24,
        "new_event_check": {"cells": 19, "재현": 0, "같은 쪽": 6, "재현 안 됨": 6, "판정 불가": 7,
                            "source_sha256": sha(EV / "new_event_check.json"),
                            "keys": list(nec.keys())[:10]},
        "verification_flags": [
            "SKHYNIX_12 바닥 앞섬 pr_P_strong5 하위 10%는 창 종류 보정 부작용 → 후보에서 뺌",
            "W_U(미국 창) 라벨 가중 고침(10/23 감사)이 이번 재실행에 빠짐 → SOXX 라벨 층 칸 전부 영향",
            "SKHYNIX_8 고점 실시간 2칸은 20일 최고 고점 11개로만 재면 BA 0.32·0.51",
            "가짜 대표(묶음 극단 날)는 정의 밖·사후. 가짜 날 전부로 재면 여러 후보 BA 0.51~0.67",
        ],
    }


def main():
    rows, flat_days = daily_kr()
    kst = timezone(timedelta(hours=9))
    inputs = [RES / "feat_activity.parquet", RES / "pred_windows_full.parquet", RES / "prices.parquet",
              RES / "judgment.json", EV / "events.json", EV / "prices_close.csv", EV / "new_event_check.json",
              FIN / "recheck_judge_indep" / "compare_summary.json"]
    out = {
        "meta": {
            "title": "개미 지수 최종 그림 자료 (2026-10-26)",
            "made": datetime.now(kst).isoformat(timespec="seconds"),
            "status": "설명용 그림 자료. 판정을 다시 돌리지 않았다. 매매 권유나 '지금이 바닥/고점' 판단이 아니다.",
            "snapshot": ["2025-01-01", CUTOFF],
            "judge_sha256": sha(ROOT / "final" / "code" / "judge.py"),
            "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs},
            "fields": {
                "daily_kr": "국내 창(W_KR, 전 거래일 08:00 ~ D 08:00 KST)마다 한 줄. posts_no = 글 번호 증가(지운 글 포함), "
                            "posts_saved = 저장 글, A_rz8 = H03 특징, P_bar = 군중 예측(−1~+1, H66 특징), "
                            "knowledge_period = AI 라벨 지식 기간(2026-07-01 전, 판정 신호 아님), "
                            "kospi_lnHL = D일 ln(고가/저가)(H03·H08·H70 목표), segment = 2025/front(2026-01~05)/back(2026-06-01~10-23)",
                "closes": "자산별 [날짜, 종가] (events/run2026_10_26/results/prices_close.csv)",
                "events": "확정 꺾임점(period new = 2025-01-21~08-20 새 사건, old = 2025-08-21 뒤, pre = 연구 기간 앞), 가짜 사건, 미확정 끝 극값",
            },
            "kospi_flat_bar_days": flat_days,
        },
        "daily_kr": rows,
        "closes": closes(),
        "events": events(),
        "judgment": judgment(),
        "event_study": event_summary(),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT, len(rows), "rows", OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
