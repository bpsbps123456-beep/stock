"""미리보기(부록 9) 시점 감사 ①: 포럼 교란 시험을 이 스냅샷(2026-05-28~10-17)으로 다시 돌린다.

동결 감사 하네스(interim/code/audit_leak.py)를 고치지 않고 import해 쓴다. 바꾸는 것은 이 실행 안의 입력뿐이다.
  - 스냅샷: preview/snapshot_manifest.json의 바이트만(loader.read_snapshot manifest 재현)
  - 달력: preview/results/loader_calendar.parquet(읽기만, 임시 폴더에 복사)
  - 교란 A: 마감 T 뒤 글·댓글의 제목·본문·작성자를 바꾸고, 모든 수치 필드와 다시 받은 줄을 바꾸고, T 뒤 새 글·새 줄을 더한다
  - 교란 B: A + T 뒤 글·댓글 20%를 모든 줄에서 지운다
  → 마감 ≤ T 창의 특징 3단계(feat_activity·feat_text·feat_topic) 모든 열이 그대로여야 한다(B는 '빠지는 쪽'만 허용).
더해 보는 것(이 미리보기에서 새로 생긴 단계):
  - 예측 층 표본(label_sample.frame/draw): 마감 ≤ T 창(D ≥ 07-01)에서 뽑힌 단위가 그대로인가
가격은 읽지 않는다(feat_text H24 사건 표지는 prices=None, 미리보기 칸에 쓰지 않는 열).
쓰기: preview/results/audit_timing_forum.json. 임시 산출은 --tmp 아래(끝나면 지운다).
실행: py audit_timing_forum.py --tmp <임시 폴더> [--quick]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time as _time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
RES = PREVIEW / "results"
sys.path.insert(0, str(ANT / "interim" / "code"))
sys.path.insert(0, str(HERE))
import loader as LD  # noqa: E402
import audit_leak as AL  # noqa: E402
import label_sample as LS  # noqa: E402

AL.RESULTS = RES          # 하네스가 달력을 복사해 오는 곳만 바꾼다(읽기만)
SEED = 20261018


def load_base() -> dict:
    man = json.loads((PREVIEW / "snapshot_manifest.json").read_text("utf-8"))
    rows, man2 = LD.read_snapshot(LD.POSTS_DIR, man["cutoff"], man)
    cal = LD.trading_calendars(LD.CAL_START, date.fromisoformat(man["cutoff"]), RES / "loader_calendar.parquet")
    return {"rows": rows, "man": man2, "cal": cal, "high": man.get("state_high")}


def cutoffs(b0: dict, quick: bool) -> list[pd.Timestamp]:
    wk = b0["windows_kr"]
    wk = wk[(wk["variant"] == "main") & wk["complete"]].sort_values("close_kst")
    wu = b0["windows_us"]
    wu = wu[(wu["variant"] == "main") & wu["complete"]].sort_values("close_kst")
    kc = {w: pd.Timestamp(c) for w, c in zip(wk["win_id"], wk["close_kst"])}
    uc = {w: pd.Timestamp(c) for w, c in zip(wu["win_id"], wu["close_kst"])}
    ts = [kc["K2026-07-15"], uc["U2026-08-20"], kc["K2026-09-15"], kc["K2026-10-13"], uc["U2026-10-14"],
          kc["K2026-10-16"]]
    return [ts[2], ts[5]] if quick else ts


def sample_keys(tmp: Path) -> pd.DataFrame:
    """label_sample의 틀·뽑기를 tmp의 units·windows로 다시 돌려 뽑힌 단위의 내용 키를 낸다."""
    old = LS.RES
    LS.RES = tmp
    try:
        f, w, _ = LS.frame()
        smp, _ = LS.draw(f, w)
    finally:
        LS.RES = old
    out = smp[["win_kr", "utype", "no", "a", "dt_kst"]].copy()
    out["cid_text"] = smp["text"].fillna("").astype(str)
    out["title"] = smp["title"].fillna("").astype(str)
    out = out.merge(w[["win_id", "close_kst"]].rename(columns={"win_id": "win_kr"}), on="win_kr", how="left")
    return out


def compare_samples(s0: pd.DataFrame, s1: pd.DataFrame, T: pd.Timestamp) -> dict:
    a = s0[pd.to_datetime(s0["close_kst"]) <= T]
    b = s1[pd.to_datetime(s1["close_kst"]) <= T]
    key = ["win_kr", "utype", "no", "a", "dt_kst", "cid_text", "title"]
    ka = set(map(tuple, a[key].astype(str).to_numpy()))
    kb = set(map(tuple, b[key].astype(str).to_numpy()))
    wins_diff = sorted({k[0] for k in ka ^ kb})
    return {"windows_compared": int(a["win_kr"].nunique()), "units_base": len(ka), "units_pert": len(kb),
            "only_base": len(ka - kb), "only_pert": len(kb - ka), "windows_with_diff": wins_diff[:10],
            "n_windows_with_diff": len(wins_diff)}


def keeproots_only(a) -> int:
    t0 = _time.time()
    rp = RES / "audit_timing_forum.json"
    rep = json.loads(rp.read_text("utf-8"))
    base = load_base()
    S = datetime.fromisoformat(base["man"]["S"])
    b0 = AL.run_loader(base["rows"], base)
    tmp_root = Path(tempfile.mkdtemp(prefix="audit_timing_forum_kr_", dir=a.tmp))
    try:
        f0 = AL.run_features(b0, tmp_root / "base")
        s0 = sample_keys(tmp_root / "base")
        ts = cutoffs(b0, False)
        for k in [int(x) for x in a.keeproots.split(",")]:
            T = ts[k]
            rows_p, st, max_no = AL.perturb(base["rows"], b0["units"], T, S, SEED + k, delete=True, keep_roots=True)
            bp = AL.run_loader(rows_p, base, high=max(base["high"] or 0, max_no))
            d = tmp_root / f"p{k}Bk"
            fp = AL.run_features(bp, d)
            cmpr = AL.compare(f0, fp, T)
            rep["perturb"][f"T{k}_B_keeproots"] = {"T": str(T), "variant": "B, 깊이 0(뿌리) 미래 댓글은 지우지 않음",
                                                   "perturb": st, "compare": cmpr,
                                                   "units": AL.unit_shift_check(b0["units"], bp["units"], T),
                                                   "label_sample": compare_samples(s0, sample_keys(d), T)}
            nd = sum(len(v["diff"]) for v in cmpr.values())
            print(f"T{k} {T} B_keeproots: diff cols {nd}", round(_time.time() - t0), flush=True)
            shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), "utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keeproots", default="", help="B에서 차이가 난 T 번호(쉼표). 깊이 0(뿌리) 미래 댓글은 지우지 않는 B를 더 돌려 "
                                                    "기존 json에 더한다(감사 하네스 B_keeproots와 같은 판)")
    a = ap.parse_args(argv)
    if a.keeproots:
        return keeproots_only(a)
    t0 = _time.time()
    base = load_base()
    S = datetime.fromisoformat(base["man"]["S"])
    b0 = AL.run_loader(base["rows"], base)
    rep = {"made": datetime.now().replace(microsecond=0).isoformat(), "S": base["man"]["S"],
           "snapshot": "preview/snapshot_manifest.json", "harness": "interim/code/audit_leak.py(고치지 않고 import)"}
    tmp_root = Path(tempfile.mkdtemp(prefix="audit_timing_forum_", dir=a.tmp))
    try:
        f0 = AL.run_features(b0, tmp_root / "base")
        s0 = sample_keys(tmp_root / "base")
        print("base", round(_time.time() - t0), flush=True)
        # 하네스가 저장된 미리보기 산출을 그대로 다시 만드는가
        saved = {"fa": pd.read_parquet(RES / "feat_activity.parquet"),
                 "ft": pd.read_parquet(RES / "feat_text.parquet"),
                 "fto": pd.read_parquet(RES / "feat_topic.parquet")}
        rs = AL.compare({**f0, **saved}, f0, pd.Timestamp("2100-01-01"))
        rep["harness_vs_saved"] = {k: {"rows": v["rows_compared"], "cols": v["cols_compared"],
                                       "diff_cols": sorted(v["diff"]), "n_key_mismatch": v["n_key_mismatch"]}
                                   for k, v in rs.items() if k in saved}
        key = pd.read_parquet(PREVIEW / "label_key.parquet")
        s_saved = set(zip(key["win_id"], key["utype"], key["no"].astype(int), key["a"]))
        s_harn = set(zip(s0["win_kr"], s0["utype"], s0["no"].astype(int), s0["a"]))
        rep["label_sample_harness_vs_saved"] = {"saved": len(s_saved), "harness": len(s_harn),
                                                "only_saved": len(s_saved - s_harn), "only_harness": len(s_harn - s_saved)}
        print("harness vs saved", json.dumps(rep["harness_vs_saved"], ensure_ascii=False),
              rep["label_sample_harness_vs_saved"], flush=True)
        pert = {}
        for k, T in enumerate(cutoffs(b0, a.quick)):
            for tag, delete in (("A", False), ("B", True)):
                rows_p, st, max_no = AL.perturb(base["rows"], b0["units"], T, S, SEED + k, delete=delete)
                bp = AL.run_loader(rows_p, base, high=max(base["high"] or 0, max_no))
                d = tmp_root / f"p{k}{tag}"
                fp = AL.run_features(bp, d)
                cmpr = AL.compare(f0, fp, T)
                sp = sample_keys(d)
                pert[f"T{k}_{tag}"] = {"T": str(T), "variant": {"A": "A(내용·수치·새 줄)", "B": "B(A + 지우기)"}[tag],
                                       "perturb": st, "compare": cmpr,
                                       "units": AL.unit_shift_check(b0["units"], bp["units"], T),
                                       "label_sample": compare_samples(s0, sp, T)}
                nd = sum(len(v["diff"]) for v in cmpr.values())
                print(f"T{k} {T} {tag}: diff cols {nd}, sample {pert[f'T{k}_{tag}']['label_sample']}",
                      round(_time.time() - t0), flush=True)
                shutil.rmtree(d, ignore_errors=True)
        rep["perturb"] = pert
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    rep["seconds"] = round(_time.time() - t0)
    (RES / "audit_timing_forum.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), "utf-8")
    print("wrote audit_timing_forum.json", rep["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
