"""사건 연구(부록 13·14) 실시간성 감사 ①: 마감 T 뒤 글·댓글·다시 받은 줄을 바꿔도 마감 ≤ T 창의 사건 연구 특징이 그대로인가.

동결 감사 하네스(interim/code/audit_leak.py)의 교란(perturb)과 동결 로더·특징 코드를 고치지 않고 import해 쓴다.
바꾸는 것은 이 실행 안의 입력 경로뿐이다(임시 폴더). 가격은 읽지 않는다(feat_text H24 사건 표지는 prices=None, 사건 연구가 쓰지 않는 열).

교란 A(하네스 그대로): T 뒤에 쓴 글·댓글의 제목·본문·작성자를 바꾸고, 다시 받은 줄의 제목·본문·ncomment를 바꾸고,
  모든 줄의 조회·추천·다른의견(원 줄)을 무작위로 바꾸고, T 뒤 새 글 300개와 'T 뒤 댓글만 든 다시 받은 줄'(T 전 글의 30%)을 더한다.
  여기에 T 바로 뒤 12시간 안에 새 글 60개(댓글 0~5개)를 더한다(덧붙임: 마감 직후 몰리는 글. 하네스의 300개는 석 달에 흩어져
  T 바로 뒤 창에 거의 안 떨어진다).
교란 B: A + T 뒤 글·댓글 20%를 모든 줄에서 지운다(빠지는 쪽만 허용).
다시 돌리는 단계: 로더 → feat_activity → feat_topic → feat_text(가격 없음) → feat_views(조회 수치 줄 views_rows는 원본 그대로:
  수치 필드는 수집 시점 누적값이라 정의상 마감 뒤 값이며 부록 13 누출 시험이 따로 다룬다) → features_all.build
  (예측·감정 층 라벨과 표본은 고정, 틀 크기·가중만 다시 셈).
비교 기준: 저장된 사건 연구 입력(events/results/features_all.parquet, snapshot_1023/results/*). 마감 ≤ T 행의 모든 열.

T(실시간 규칙 'D+1 07:59' 기준과 창 자기 마감 기준을 함께 본다)
  T0 2026-07-29 21:30  U2026-07-29 마감(SOXX 7/29 바닥의 오프셋 0 창 자기 마감)
  T1 2026-07-30 08:00  SOXX 7/29 바닥의 D+1 07:59 마감 = K2026-07-30 마감(KOSPI·하이닉스 7/30 바닥의 오프셋 0)
  T2 2026-07-31 08:00  KOSPI·하이닉스 7/30 바닥의 D+1 07:59 마감 = K2026-07-31 마감(오프셋 +1)
  T3 2026-10-14 08:00  실시간 수집 시기(K2026-10-14 마감)

실행: py -B audit_realtime.py --tmp <임시 폴더> [--only T0,T1] [--variants A,B]
쓰기: events/results/audit_realtime.json(합침). 임시 산출은 --tmp 아래(끝나면 지운다).
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import snap_paths as SP  # noqa: E402

sys.path.insert(0, str(SP.ANT / "interim" / "code"))
import loader as LD  # noqa: E402
import audit_leak as AL  # noqa: E402
import feat_activity as FA  # noqa: E402
import feat_topic as FT  # noqa: E402
import feat_text as FX  # noqa: E402
import features_all as FAL  # noqa: E402
import feat_views as FV  # noqa: E402

AL.RESULTS = SP.SNAP_RES        # 하네스가 달력을 복사해 오는 곳(읽기만)
INTERIM_RES = SP.ANT / "interim" / "results"
OUT_JSON = SP.EV / "results" / "audit_realtime.json"
SEED = 20261023
CUTS = {"T0": pd.Timestamp("2026-07-29 21:30"), "T1": pd.Timestamp("2026-07-30 08:00"),
        "T2": pd.Timestamp("2026-07-31 08:00"), "T3": pd.Timestamp("2026-10-14 08:00")}
CUT_NOTE = {"T0": "U2026-07-29 마감(창 자기 마감)", "T1": "SOXX 7/29 바닥 D+1 07:59(= K2026-07-30 마감)",
            "T2": "KOSPI·하이닉스 7/30 바닥 D+1 07:59(= K2026-07-31 마감)", "T3": "실시간 수집 시기 K2026-10-14 마감"}
BURST_N, BURST_H = 60, 12
CELLW = False                    # --cellw: W_U 라벨 층을 칸 가중으로 고친 판(audit_wu_weights.design_weights_cell)으로 돌리고
                                 # 기준을 events/results/features_all_wu_cellw.parquet로 둔다


def log(msg: str) -> None:
    print(time_s(), msg, flush=True)


def time_s() -> str:
    return datetime.now().strftime("%H:%M:%S")


def load_man() -> dict:
    return json.loads((SP.SNAP / "snapshot_manifest.json").read_text("utf-8"))


def read_rows(man: dict) -> list[dict]:
    rows, _ = LD.read_snapshot(LD.POSTS_DIR, man["cutoff"], man)
    return rows


def calendar(tmp: Path) -> dict:
    d = tmp / "_cal"
    d.mkdir(parents=True, exist_ok=True)
    for n in ("loader_calendar.parquet", "loader_calendar.json"):
        shutil.copyfile(SP.SNAP_RES / n, d / n)
    return LD.trading_calendars(LD.CAL_START, date.fromisoformat(load_man()["cutoff"]), d / "loader_calendar.parquet")


def add_burst(rows_p: list[dict], units0: pd.DataFrame, T: pd.Timestamp, max_no: int, seed: int) -> tuple[int, dict]:
    """T 바로 뒤 BURST_H시간 안에 새 글 BURST_N개(하네스 새 글과 같은 꼴, 번호는 하네스 끝 번호 뒤)."""
    rng = np.random.default_rng(seed)
    authors = units0["a"].dropna().unique()
    line0 = 3 * 10 ** 7
    n_c = 0
    for k in range(BURST_N):
        t = (T + pd.Timedelta(seconds=float(rng.uniform(0, BURST_H * 3600)))).floor("min")
        no = max_no + 1 + k
        cs = []
        for j in range(int(rng.integers(0, 6))):
            ct = t + pd.Timedelta(seconds=float(rng.uniform(0, 6 * 3600)))
            cs.append({"dt": ct.strftime("%Y-%m-%d %H:%M:%S"), "text": AL._scramble_text(("bc", no, j)), "up": 0,
                       "down": 0, "depth": 0, "a": str(authors[AL._h("bc", no, j) % len(authors)])})
            n_c += 1
        rec = {"no": no, "dt": t.strftime("%Y-%m-%d %H:%M"), "title": AL._scramble_text(("bt", no)),
               "body": AL._scramble_text(("bb", no)), "views": 5, "up": 5, "down": 5, "ncomment": len(cs),
               "comments": cs, "a": str(authors[AL._h("bp", no) % len(authors)])}
        rows_p.append({"file": t.strftime("%Y-%m-%d"), "line": line0 + k, "rec": rec})
    return max_no + BURST_N, {"burst_posts": BURST_N, "burst_comments": n_c, "burst_hours_after_T": BURST_H}


def run_stages(b: dict, tmp: Path) -> None:
    """동결 특징 3단계 + 조회 특징 + 사건 연구 통합 특징을 tmp에서 돌린다(파일 왕복, 실제 실행과 같은 자료형)."""
    AL._write_inputs(b, tmp)
    for n in ("lexicon_v02.json", "lexicon_v02_full.json"):
        shutil.copyfile(INTERIM_RES / n, tmp / n)
    for n in ("pred_windows_full.parquet", "views_rows.parquet", "views_fetch_meta.json"):
        shutil.copyfile(SP.SNAP_RES / n, tmp / n)
    fb = FA.build(results=tmp)
    FA.write(fb, tmp)
    del fb
    FT.run(tmp, rebuild_hits=True)
    old = FX.RESULTS
    FX.RESULTS = tmp
    try:
        lex = FX.check_lexicon(tmp / "lexicon_v02.json")
        if lex["word_set_mismatch"]:
            raise RuntimeError("사전 단어 불일치")
        cols = ["no", "cid", "a", "utype", "dt_kst", "title", "text", "dt_conf", "win_kr", "win_u", "pwin_kr", "pwin_u",
                "pre_close_kr", "pre_close_u", "rank_ok_kr", "rank_ok_u", "c120_kr", "c120_u"]
        u = pd.read_parquet(tmp / "units.parquet", columns=cols)
        m = FX.match_units(u)
        m.reset_index().to_parquet(tmp / "feat_text_units.parquet", index=False)
        wk = pd.read_parquet(tmp / "windows_kr.parquet")
        wu = pd.read_parquet(tmp / "windows_us.parquet")
        feat = pd.concat([FX.build_market("kr", u, m, wk, None), FX.build_market("us", u, m, wu, None)],
                         ignore_index=True, sort=False)
        feat.to_parquet(tmp / "feat_text.parquet", index=False)
        del u, m, feat
    finally:
        FX.RESULTS = old
    FV.RES, FV.OUT = tmp, tmp
    FV.main()
    FAL.RES = tmp
    FAL.FLS.RES = tmp
    FAL.PWF.RES = tmp
    SP.SNAP_RES = tmp
    orig_dw = FAL.design_weights
    if CELLW:
        import audit_wu_weights as AW
        AW.ALLOW_MISSING_CELLS = True
        AW.MISSING_FILLED.clear()
        FAL.design_weights = AW.design_weights_cell
    try:
        F, _, _ = FAL.build()
    finally:
        FAL.design_weights = orig_dw
        SP.SNAP_RES = SP.SNAP / "results"
        FAL.RES = SP.SNAP_RES
        FAL.FLS.RES = SP.SNAP_RES
        FAL.PWF.RES = SP.SNAP_RES
        FV.RES = FV.OUT = SP.SNAP_RES
    F.to_parquet(tmp / "features_all_pert.parquet", index=False)
    if CELLW:
        import audit_wu_weights as AW
        (tmp / "cellw_missing.json").write_text(json.dumps(AW.MISSING_FILLED), "utf-8")


UCOLS = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
         "pwin_kr", "win_u", "root_cid", "depth"]


def load_units(p: Path) -> pd.DataFrame:
    u = pd.read_parquet(p, columns=UCOLS)
    for c in ("dt_kst", "post_dt_kst"):
        u[c] = u[c].astype("datetime64[ns]")
    return u


def _num(s: pd.Series):
    return AL._num(s)


def diff_table(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], skip: set[str] = frozenset()) -> dict:
    a = a.drop_duplicates(keys).set_index(keys).sort_index()
    b = b.drop_duplicates(keys).set_index(keys).sort_index()
    miss = sorted(map(str, set(a.index) ^ set(b.index)))
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    diff = {}
    cols = [c for c in a.columns.intersection(b.columns) if c not in skip]
    for c in cols:
        x, y = a[c], b[c]
        xn, yn = _num(x), _num(y)
        if xn is not None and yn is not None:
            bad = ~((np.isnan(xn) & np.isnan(yn)) | (np.abs(xn - yn) <= 1e-9 * np.maximum(1, np.abs(xn))))
        else:
            xm, ym = pd.isna(x).to_numpy(), pd.isna(y).to_numpy()
            xs = np.array([str(v) for v in x.to_numpy(object)], dtype=object)
            ys = np.array([str(v) for v in y.to_numpy(object)], dtype=object)
            bad = ~((xm & ym) | (~xm & ~ym & (xs == ys)))
        if bad.any():
            rows = [str(idx[i]) for i in np.flatnonzero(bad)[:6]]
            diff[c] = {"rows": int(bad.sum()), "examples": rows}
    return {"rows_compared": int(len(idx)), "cols_compared": len(cols), "n_key_mismatch": len(miss),
            "key_mismatch": miss[:5], "diff": diff}


def closes() -> pd.DataFrame:
    wk = pd.read_parquet(SP.SNAP_RES / "windows_kr.parquet")
    wu = pd.read_parquet(SP.SNAP_RES / "windows_us.parquet")
    w = pd.concat([wk[(wk["variant"] == "main") & wk["complete"]], wu[(wu["variant"] == "main") & wu["complete"]]])
    return w[["win_id", "close_kst"]].assign(close_kst=lambda d: pd.to_datetime(d["close_kst"]))


def compare_all(tmp: Path, T: pd.Timestamp, u0: pd.DataFrame) -> dict:
    cl = closes()
    out = {}
    # 1) 사건 연구 통합 특징(46 격자 + 나머지 특징 + 표지)
    F0 = pd.read_parquet(SP.EV / "results" / ("features_all_wu_cellw.parquet" if CELLW else "features_all.parquet"))
    F1 = pd.read_parquet(tmp / "features_all_pert.parquet")
    le = lambda d: d[pd.to_datetime(d["close_kst"]) <= T]  # noqa: E731
    gt = lambda d: d[pd.to_datetime(d["close_kst"]) > T]  # noqa: E731
    out["features_all_le_T"] = diff_table(le(F0), le(F1), ["win_id"])
    out["features_all_gt_T"] = {k: v for k, v in diff_table(gt(F0), gt(F1), ["win_id"]).items() if k != "diff"}
    out["features_all_gt_T"]["diff_cols"] = sorted(diff_table(gt(F0), gt(F1), ["win_id"])["diff"])
    # 2) 동결 3단계 + 조회 특징(마감 ≤ T 행)
    specs = [("feat_activity.parquet", ["win_id"], set()),
             ("feat_text.parquet", ["win_id"], None),
             ("feat_topic.parquet", ["table", "win_id"], set(AL.LATE_COLS["fto"])),
             ("feat_views.parquet", ["win_id"], set())]
    for fn, keys, skip in specs:
        a = pd.read_parquet(SP.SNAP_RES / fn)
        b = pd.read_parquet(tmp / fn)
        if fn == "feat_text.parquet":
            skip = {c for c in a.columns if c.startswith(("px_", "ev_"))}   # H24 가격 표지(사건 연구 미사용, 이 실행은 가격 없음)
        if "close_kst" not in a.columns:
            a = a.merge(cl, on="win_id", how="left")
            b = b.merge(cl, on="win_id", how="left")
        out[fn + "_le_T"] = diff_table(le(a), le(b), keys, skip)
    # 3) 단위 출입(마감 ≤ T 창)
    u1 = load_units(tmp / "units.parquet")
    out["units"] = AL.unit_shift_check(u0, u1, T)
    # 4) 라벨 표본 틀 크기: K창 층(창 × 글/댓글), U창 칸(U창 × K창 × 글/댓글)
    fr0, fr1 = FAL.frame_cells(u0), None
    old = FAL.FLS.RES
    FAL.FLS.RES = tmp
    try:
        fr1 = FAL.frame_cells(u1)
    finally:
        FAL.FLS.RES = old
    clm = dict(zip(cl["win_id"], cl["close_kst"]))
    for nm, cols in (("frame_K_strata", ["win_kr", "utype"]), ("frame_U_cells", ["win_u", "win_kr", "utype"])):
        a = fr0.groupby(cols, dropna=True).size().rename("N").reset_index()
        b = fr1.groupby(cols, dropna=True).size().rename("N").reset_index()
        wcol = cols[0]
        a = a[a[wcol].map(clm) <= T]
        b = b[b[wcol].map(clm) <= T]
        m = a.merge(b, on=cols, how="outer", suffixes=("_base", "_pert"))
        ch = m[~(m["N_base"].fillna(-1) == m["N_pert"].fillna(-1))]
        out[nm + "_le_T"] = {"cells": int(len(m)), "changed": int(len(ch)), "examples": ch.head(6).astype(str).to_dict("records")}
    # U창(마감 ≤ T)에 단위를 대는 K창 층 중 마감 > T인 층: 표본 가중(층 틀 수 ÷ 표본 수)이 T 뒤 단위 수에 기댄다
    fu = fr0[fr0["win_u"].map(clm) <= T]
    late_k = sorted({k for k in fu["win_kr"].dropna().unique() if clm.get(k, pd.Timestamp.max) > T})
    s0 = fr0.groupby(["win_kr", "utype"]).size()
    s1 = fr1.groupby(["win_kr", "utype"]).size()
    out["K_strata_feeding_U_le_T_with_close_gt_T"] = {
        "K_windows": late_k,
        "N_base": {f"{k}|{t}": int(s0.get((k, t), 0)) for k in late_k for t in ("post", "comment")},
        "N_pert": {f"{k}|{t}": int(s1.get((k, t), 0)) for k in late_k for t in ("post", "comment")}}
    return out


def one(tag: str, variant: str, T: pd.Timestamp, man: dict, cal: dict, u0: pd.DataFrame, tmp_root: Path, k: int) -> dict:
    t0 = _time.time()
    rows = read_rows(man)
    S = datetime.fromisoformat(man["S"])
    rows_p, st, max_no = AL.perturb(rows, u0, T, S, SEED + k, delete=(variant == "B"))
    del rows
    gc.collect()
    max_no, bst = add_burst(rows_p, u0, T, max_no, SEED + 100 + k)
    st.update(bst)
    log(f"{tag}{variant} perturbed {st}")
    bp = LD.build(rows_p, man, cal, {"high": max(int(man.get("state_high") or 0), max_no)})
    del rows_p
    gc.collect()
    d = tmp_root / f"{tag}{variant}"
    run_stages(bp, d)
    del bp
    gc.collect()
    cmp = compare_all(d, T, u0)
    if CELLW and (d / "cellw_missing.json").exists():
        cmp["cellw_missing_cells_filled_weight1"] = json.loads((d / "cellw_missing.json").read_text("utf-8"))
    shutil.rmtree(d, ignore_errors=True)
    return {"T": str(T), "note": CUT_NOTE[tag], "variant": {"A": "A(내용·작성자·수치·다시 받은 줄·새 글 + T 직후 몰림)",
                                                             "B": "B(A + T 뒤 글·댓글 20% 지움)"}[variant],
            "perturb": st, "compare": cmp, "seconds": round(_time.time() - t0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--only", default="T0,T1,T2,T3")
    ap.add_argument("--variants", default="A")
    ap.add_argument("--cellw", action="store_true")
    a = ap.parse_args(argv)
    global CELLW
    CELLW = a.cellw
    man = load_man()
    u0 = load_units(SP.SNAP_RES / "units.parquet")
    rep = json.loads(OUT_JSON.read_text("utf-8")) if OUT_JSON.exists() else {}
    rep.setdefault("runs", {})
    rep.update({"harness": "interim/code/audit_leak.py perturb·unit_shift_check(고치지 않고 import)",
                "snapshot": "events/snapshot_1023/snapshot_manifest.json", "S": man["S"],
                "baseline": "저장 산출(events/results/features_all.parquet, snapshot_1023/results/*)",
                "fixed_inputs": "예측·감정 라벨과 표본(대응표), views_rows(조회 수치), pred_windows_full(라벨·창 경계만 씀)"})
    tmp_root = Path(tempfile.mkdtemp(prefix="audit_realtime_", dir=a.tmp))
    try:
        cal = calendar(tmp_root)
        for k, tag in enumerate(CUTS):
            if tag not in a.only.split(","):
                continue
            for v in a.variants.split(","):
                log(f"start {tag}{v} T={CUTS[tag]}")
                r = one(tag, v, CUTS[tag], man, cal, u0, tmp_root, k)
                r["cellw"] = CELLW
                rep["runs"][f"{tag}_{v}" + ("_cellw" if CELLW else "")] = r
                c = r["compare"]
                log(f"{tag}{v} features_all≤T diff cols {sorted(c['features_all_le_T']['diff'])} "
                    f">T diff cols {len(c['features_all_gt_T']['diff_cols'])} units {c['units']} {r['seconds']}s")
                rep["made"] = datetime.now().replace(microsecond=0).isoformat()
                OUT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), "utf-8")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
