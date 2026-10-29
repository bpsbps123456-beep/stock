"""구현 A·B 결과 대조: 32칸 × {fade, follow, b0} 모든 필드(건수 정확히, 비율·% 1e-6 안)와 주 칸 매매 목록 csv 행 단위.

사용: py code/compare_ab.py  → 화면 요약 + results/compare_ab.json
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
TOL = 1e-6
INT_FIELDS = {"n", "n_days", "skipped_overlap", "skipped_nobar", "race_ambig_n"}
SCHEMA = {
    "fade": ["n", "n_days", "skipped_overlap", "skipped_nobar", "hit_rate", "ev_mean_pct", "ev_median_pct",
             "miss_mean_pct", "mae_median_pct", "race_win", "race_loss", "race_ambig_n", "race_none"],
    "b0": ["n", "hit_rate", "ev_mean_pct", "race_win", "race_loss"],
}
SCHEMA["follow"] = SCHEMA["fade"]
CSV_COLS = ["day", "signal_ticker", "signal_time", "buy", "entry_time", "entry_px", "hit", "exit_time",
            "pnl_pct", "mae_pct"]


def same(f, a, b):
    if a is None or b is None:
        return a is None and b is None
    if f in INT_FIELDS:
        return int(a) == int(b) and float(a) == int(a) and float(b) == int(b)
    return abs(float(a) - float(b)) <= TOL


def compare_cells(A, B):
    mism, nchk = [], 0
    ka, kb = set(A["cells"]), set(B["cells"])
    if ka != kb:
        mism.append(f"칸 키 차이: A만 {sorted(ka - kb)}, B만 {sorted(kb - ka)}")
    if len(ka) != 32:
        mism.append(f"A 칸 수 {len(ka)} ≠ 32")
    for key in sorted(ka & kb):
        for grp, fields in SCHEMA.items():
            ga, gb = A["cells"][key].get(grp), B["cells"][key].get(grp)
            if ga is None or gb is None:
                mism.append(f"{key}.{grp} 없음 (A {ga is not None}, B {gb is not None})")
                continue
            for side, g in (("A", ga), ("B", gb)):
                extra = set(g) - set(fields)
                missing = set(fields) - set(g)
                if extra or missing:
                    mism.append(f"{key}.{grp} {side} 스키마: 추가 {sorted(extra)}, 빠짐 {sorted(missing)}")
            for f in fields:
                nchk += 1
                if not same(f, ga.get(f), gb.get(f)):
                    mism.append(f"{key}.{grp}.{f}: A={ga.get(f)} B={gb.get(f)}")
    return mism, nchk


def compare_csv(pa, pb):
    a = pd.read_csv(pa, dtype={"signal_time": str, "entry_time": str, "exit_time": str})
    b = pd.read_csv(pb, dtype={"signal_time": str, "entry_time": str, "exit_time": str})
    mism = []
    for side, d in (("A", a), ("B", b)):
        if list(d.columns) != CSV_COLS:
            mism.append(f"csv {side} 열: {list(d.columns)}")
    if len(a) != len(b):
        mism.append(f"csv 행 수 A={len(a)} B={len(b)}")
        return mism, len(a), len(b), False
    # 저장 순서 그대로 행 단위 비교(같은 순서인지도 확인)
    order_same = True
    for i in range(len(a)):
        ra, rb = a.iloc[i], b.iloc[i]
        for c in CSV_COLS:
            va, vb = ra[c], rb[c]
            if c in ("entry_px",):
                ok = abs(float(va) - float(vb)) <= 5e-5   # 소수 4자리 기록
            elif c in ("pnl_pct", "mae_pct"):
                ok = abs(float(va) - float(vb)) <= TOL
            elif c == "hit":
                ok = int(va) == int(vb)
            else:
                ok = str(va) == str(vb)
            if not ok:
                order_same = False
                mism.append(f"csv 행 {i} {c}: A={va} B={vb}")
    return mism, len(a), len(b), order_same


AUX_CI = {"fade_ev_mean_pct": "fade_ev_ci", "fade_minus_b0_hit": "fade_minus_b0_hit_ci",
          "fade_minus_follow_ev_pct": "fade_minus_follow_ev_ci"}


def compare_aux(A, B):
    """보조 5분봉: 기간별 fade·follow·b0 필드와 부트스트랩 구간. B는 기간별 건너뜀 수를 두지 않아 합계로 맞춘다."""
    xa, xb = A.get("extra", {}).get("aux_5m"), B.get("aux_5m")
    if xa is None or xb is None:
        return [f"aux_5m 없음 (A {xa is not None}, B {xb is not None})"], 0
    mism, nchk = [], 0
    if set(xa["periods"]) != set(xb["periods"]):
        return [f"aux_5m 기간 키 A {sorted(xa['periods'])} B {sorted(xb['periods'])}"], 0
    skip_sum = {g: {"skipped_overlap": 0, "skipped_nobar": 0} for g in ("fade", "follow")}
    for per in sorted(xa["periods"]):
        pa, pb = xa["periods"][per], xb["periods"][per]
        nchk += 1
        if pa["n_days"] != pb["n_days"]:
            mism.append(f"aux {per} n_days A={pa['n_days']} B={pb['n_days']}")
        for grp, fields in SCHEMA.items():
            for f in fields:
                if f in ("skipped_overlap", "skipped_nobar"):
                    skip_sum[grp][f] += pa[grp][f]
                    continue
                nchk += 1
                if not same(f, pa[grp].get(f), pb[grp].get(f)):
                    mism.append(f"aux {per}.{grp}.{f}: A={pa[grp].get(f)} B={pb[grp].get(f)}")
        for ka, kb in AUX_CI.items():
            nchk += 2
            a = [pa["ci95"][ka]["lo"], pa["ci95"][ka]["hi"]]
            if not all(same("ci", x, y) for x, y in zip(a, pb["boot"][kb])):
                mism.append(f"aux {per} CI {ka}: A={a} B={pb['boot'][kb]}")
    for grp in ("fade", "follow"):
        for f in ("skipped_overlap", "skipped_nobar"):
            nchk += 1
            if skip_sum[grp][f] != xb[f][grp]:
                mism.append(f"aux 합계 {grp}.{f}: A={skip_sum[grp][f]} B={xb[f][grp]}")
    return mism, nchk


def main():
    A = json.load(open(os.path.join(RES, "impl_a.json"), encoding="utf-8"))
    B = json.load(open(os.path.join(RES, "impl_b.json"), encoding="utf-8"))
    cm, nchk = compare_cells(A, B)
    am, nchk_aux = compare_aux(A, B)
    csvm, na, nb, order_same = compare_csv(os.path.join(RES, "events_a.csv"), os.path.join(RES, "events_b.csv"))
    ev = []
    if A.get("events") != "results/events_a.csv":
        ev.append(f"A events 경로 {A.get('events')}")
    if B.get("events") != "results/events_b.csv":
        ev.append(f"B events 경로 {B.get('events')}")
    out = {"fields_checked": nchk, "cell_mismatches": cm, "csv_rows": [na, nb], "csv_rows_same_order": order_same,
           "csv_mismatches": csvm, "events_path_issues": ev,
           "aux_5m_fields_checked": nchk_aux, "aux_5m_mismatches": am,
           "all_match": not (cm or csvm or ev or am)}
    with open(os.path.join(RES, "compare_ab.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("비교 필드", nchk, "칸 불일치", len(cm), "csv 행", na, nb, "csv 불일치", len(csvm), "경로", ev,
          "| 보조 5분봉 필드", nchk_aux, "불일치", len(am))
    for m in (cm + csvm + am)[:60]:
        print(" ", m)
    print("전부 일치" if out["all_match"] else "불일치 있음")


if __name__ == "__main__":
    main()
