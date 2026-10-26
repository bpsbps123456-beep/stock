"""부록 15 전체 분석 실행 준비 1·2단계: 스냅샷 목록과 특징(가격 앞날 미사용).

final/code·interim/code는 한 줄도 고치지 않고 불러 쓴다. 바꾸는 것은 스냅샷 범위와 결과 폴더(final/run2026_10_26)뿐이다.
  snapshot   final/code/snapshot.py main, FIRST = 2025-01-01, CUTOFF = 2026-10-23, OUT = run2026_10_26/snapshot_manifest.json.
             앞선 두 스냅샷(final 2025-08-20~10-21, events/snapshot_1023)의 파일 앞 바이트가 그대로인지(덧붙이기만 했는지),
             날짜 파일이 빠짐없이 있는지, 10/23 파일이 K2026-10-23(07:59 마감)·U2026-10-23(08:29 ET 마감)을 넘겼는지 본다.
  features   final/code/run_features.py main(loader → feat_activity → feat_topic → feat_text), 결과 폴더만 바꿈.
             달력은 로더 그대로(캐시가 없으므로 pandas_market_calendars → 없으면 yfinance ^KS11·^GSPC 거래일 '날짜'만).
             만든 뒤 final·snapshot_1023 달력 캐시와 날짜별로 대조한다. 이어서 final/code/feature_summary.py(특징 목록)도 돌린다.
실행: py -B prep.py snapshot|features
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import run_paths as RP

import pandas as pd


def meta_update(section: str, value) -> None:
    m = json.loads(RP.META.read_text("utf-8")) if RP.META.exists() else {}
    m[section] = value
    RP.META.write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), "utf-8")


def prefix_check(new: dict, old_path: Path, posts: Path) -> dict:
    old = json.loads(old_path.read_text("utf-8"))
    same, grown, bad = 0, [], []
    for name, spec in old["files"].items():
        ns = new["files"].get(name)
        if ns is None:
            bad.append({"file": name, "problem": "새 스냅샷에 없음"})
            continue
        with open(posts / name, "rb") as fh:
            head = fh.read(spec["bytes"])
        if hashlib.sha256(head).hexdigest() != spec["sha256"] or ns["bytes"] < spec["bytes"]:
            bad.append({"file": name, "problem": "앞 스냅샷 바이트가 바뀜"})
        elif ns["bytes"] == spec["bytes"]:
            same += 1
        else:
            grown.append({"file": name, "old_bytes": spec["bytes"], "new_bytes": ns["bytes"],
                          "old_lines": spec["lines"], "new_lines": ns["lines"]})
    return {"old_manifest": str(old_path), "old_sha256": RP.sha(old_path), "old_S": old["S"],
            "old_n_files": old["n_files"], "same_bytes": same, "grown": grown, "problem": bad,
            "files_only_in_new": len(set(new["files"]) - set(old["files"]))}


def file_posts(posts: Path, name: str, nbytes: int) -> list[dict]:
    with open(posts / name, "rb") as fh:
        data = fh.read(nbytes)
    return [json.loads(x.decode("utf-8")) for x in data.split(b"\n") if x.strip()]


def cmd_snapshot() -> dict:
    j = RP.sha(RP.FINAL / "code" / "judge.py")
    SN = RP.load(RP.FINAL / "code" / "snapshot.py", "final_snapshot")
    SN.FIRST, SN.CUTOFF, SN.OUT = RP.FIRST, RP.CUTOFF, RP.RUN / "snapshot_manifest.json"
    RP.RUN.mkdir(parents=True, exist_ok=True)
    SN.main()
    new = json.loads(SN.OUT.read_text("utf-8"))
    posts = SN.LD.POSTS_DIR
    days = [str(date.fromisoformat(RP.FIRST) + timedelta(days=k))
            for k in range((date.fromisoformat(RP.CUTOFF) - date.fromisoformat(RP.FIRST)).days + 1)]
    have = {k[:10] for k in new["files"]}
    last = file_posts(posts, f"{RP.CUTOFF}.jsonl", new["files"][f"{RP.CUTOFF}.jsonl"]["bytes"])
    first = file_posts(posts, f"{RP.FIRST}.jsonl", new["files"][f"{RP.FIRST}.jsonl"]["bytes"])
    ldt = sorted(r["dt"] for r in last)
    fdt = sorted(r["dt"] for r in first)
    out = {"S": new["S"], "first": new["first"], "cutoff": new["cutoff"], "n_files": new["n_files"],
           "n_lines": new["n_lines"], "bytes": new["bytes"], "state_high": new["state_high"],
           "state_low": new["state_low"], "manifest_sha256": RP.sha(SN.OUT),
           "missing_day_files": sorted(set(days) - have),
           "dropped_files_outside_range": new["dropped_files_outside_range"],
           "files_after_cutoff_not_read": new["files_after_cutoff_not_read"],
           "file_first": {"name": f"{RP.FIRST}.jsonl", "lines": len(first), "first_post_dt": fdt[0] if fdt else None,
                          "last_post_dt": fdt[-1] if fdt else None},
           "file_cutoff": {"name": f"{RP.CUTOFF}.jsonl", "lines": len(last), "first_post_dt": ldt[0] if ldt else None,
                           "last_post_dt": ldt[-1] if ldt else None,
                           "max_no": max((r["no"] for r in last), default=None),
                           "past_K2026_10_23_close_0800": bool(ldt) and ldt[-1] >= f"{RP.CUTOFF} 08:00",
                           "past_U2026_10_23_close_2130_kst": bool(ldt) and ldt[-1] >= f"{RP.CUTOFF} 21:30"},
           "vs_final": prefix_check(new, RP.FINAL / "snapshot_manifest.json", posts),
           "vs_snapshot_1023": prefix_check(new, RP.SNAP1023 / "snapshot_manifest.json", posts),
           "judge_py_sha256": j, "judge_py_sha256_as_registered": j == RP.JUDGE_SHA256,
           "rule": "final/code/snapshot.py 그대로(완성된 줄만, 파일별 바이트 수·줄 수·SHA-256). 범위만 2025-01-01 ~ 2026-10-23"}
    meta_update("snapshot", out)
    assert not out["vs_final"]["problem"] and not out["vs_snapshot_1023"]["problem"], "앞 스냅샷 바이트가 바뀜"
    assert not out["missing_day_files"], out["missing_day_files"]
    assert out["file_cutoff"]["past_U2026_10_23_close_2130_kst"], "10/23 파일이 미국 창 마감을 넘기지 못함"
    brief = {k: v for k, v in out.items() if k not in ("vs_final", "vs_snapshot_1023")}
    brief["vs_final"] = {k: (len(v) if k == "grown" else v) for k, v in out["vs_final"].items()}
    brief["vs_snapshot_1023"] = {k: (len(v) if k == "grown" else v) for k, v in out["vs_snapshot_1023"].items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return out


def cal_diff(new: pd.DataFrame, old: pd.DataFrame, tag: str) -> dict:
    m = new.merge(old, on="date", how="inner", suffixes=("", "_old"))
    out = {"tag": tag, "overlap_days": int(len(m)), "first": str(m["date"].min().date()), "last": str(m["date"].max().date())}
    for c in ("krx", "nyse", "nyse_early"):
        d = m[m[c] != m[f"{c}_old"]]
        out[f"{c}_diff"] = [{"date": str(x.date()), "new": bool(a), "old": bool(b)} for x, a, b in zip(d["date"], d[c], d[f"{c}_old"])]
    return out


def cmd_features() -> int:
    RF = RP.load(RP.FINAL / "code" / "run_features.py", "final_run_features")
    RP.patch_final(RF)
    assert RF.RES == RP.RES and RF.FINAL == RP.RUN
    rc = RF.main([])
    s = json.loads((RP.RES / "loader_summary.json").read_text("utf-8"))
    cal = pd.read_parquet(RP.RES / "loader_calendar.parquet")
    cmeta = json.loads((RP.RES / "loader_calendar.json").read_text("utf-8"))
    calcheck = {"source": cmeta["source"], "start": cmeta["start"], "end": cmeta["end"], "check": cmeta.get("check"),
                "vs_final_cache": cal_diff(cal, pd.read_parquet(RP.FINAL / "results" / "loader_calendar.parquet"), "final"),
                "vs_snapshot_1023_cache": cal_diff(cal, pd.read_parquet(RP.SNAP1023 / "results" / "loader_calendar.parquet"),
                                                   "snapshot_1023"),
                "note": "로더 그대로(캐시 없음 → pandas_market_calendars 미설치 → yfinance ^KS11·^GSPC 거래일 날짜만). "
                        "snapshot_1023은 S(10/23 09:05) 전 개장 세션만 넣어 NYSE 10/23이 빠졌다. 이번 S는 그 뒤다"}
    meta_update("calendar", calcheck)
    meta_update("loader", {"snapshot": s["snapshot"], "windows_kr_main": s["windows_kr"]["main"],
                           "windows_us_main": s["windows_us"]["main"], "units": s["units"],
                           "calendar_check": s["calendar"].get("check")})
    FS = RP.load(RP.FINAL / "code" / "feature_summary.py", "final_feature_summary")
    FS.RES = RP.RES
    try:
        FS.main()
        reg = {"ok": True, "path": str(RP.RES / "feature_registry.json")}
    except Exception as e:  # 특징 목록은 보조 산출이다(실패해도 특징 파일은 그대로)
        reg = {"ok": False, "error": repr(e)}
    meta_update("feature_registry", reg)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": RP.sha(p)} for p in sorted(RP.RES.glob("*"))
             if p.is_file() and p.suffix in (".parquet", ".json", ".csv")}
    meta_update("results_files", files)
    print(json.dumps({"windows_kr_main": s["windows_kr"]["main"], "windows_us_main": s["windows_us"]["main"],
                      "units": s["units"], "calendar_vs_1023": {k: v for k, v in calcheck["vs_snapshot_1023_cache"].items()},
                      "calendar_vs_final": calcheck["vs_final_cache"], "feature_registry": reg},
                     ensure_ascii=False, indent=1, default=str))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "features"])
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        cmd_snapshot()
        return 0
    return cmd_features()


if __name__ == "__main__":
    raise SystemExit(main())
