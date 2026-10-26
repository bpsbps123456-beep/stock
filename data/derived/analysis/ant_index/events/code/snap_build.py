"""부록 14 갱신: 포럼 스냅샷을 2026-10-23 07:59 KST까지 늘려 동결 코드로 특징을 다시 만든다(가격 미사용).

final/code·interim/code는 한 줄도 고치지 않고 불러 쓴다. 바꾸는 것은 컷오프와 결과 폴더(events/snapshot_1023)뿐이다.
  snapshot   final/code/snapshot.py main, CUTOFF = 2026-10-23, OUT = snapshot_1023/snapshot_manifest.json.
             final 스냅샷 파일의 앞 바이트가 그대로인지(덧붙이기만 했는지) 대조한다.
  calendar   로더 달력 캐시(snapshot_1023/results/loader_calendar.parquet). 2026-10-21까지는 final 캐시(yfinance 거래일)
             그대로, 10/22·10/23은 거래소 달력(exchange_calendars)으로 채운다. 2026-10-23 09:00 확인에서 yfinance
             ^KS11은 10/21·10/22, ^GSPC는 10/21 막대가 없었다(빈 날을 두면 두 창이 하나로 합쳐진다).
             스냅샷 시각 S 전에 시작한 세션만 넣는다: KRX 10/23(09:00 개장)은 넣고 NYSE 10/23(23:30 KST 개장)은 뺀다
             — yfinance 날짜 원천이 S에 보였을 날과 같은 규칙. 그래서 마지막 창은 K2026-10-23(10/23 07:59 마감)·U2026-10-22.
  features   final/code/run_features.py main(loader → feat_activity → feat_topic → feat_text), 결과 폴더만 바꿈.
  pred       final/code/pred_windows_full.py main(예측 층 창 특징), 대응표·라벨 폴더만 바꿈(sample_new_windows.py 뒤).
실행: py -B snap_build.py snapshot|calendar|features|pred
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import date, datetime
from pathlib import Path

import snap_paths as SP

import pandas as pd

META = SP.SNAP / "update_meta.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_final(name: str):
    spec = importlib.util.spec_from_file_location(f"final_{name}", SP.FINAL / "code" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def meta_update(section: str, value) -> None:
    m = json.loads(META.read_text("utf-8")) if META.exists() else {}
    m[section] = value
    META.write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), "utf-8")


def cmd_snapshot() -> dict:
    SN = load_final("snapshot")
    SN.CUTOFF = SP.SNAP_CUTOFF
    SN.OUT = SP.SNAP / "snapshot_manifest.json"
    SP.SNAP.mkdir(parents=True, exist_ok=True)
    SN.main()
    new = json.loads(SN.OUT.read_text("utf-8"))
    old = json.loads((SP.FINAL / "snapshot_manifest.json").read_text("utf-8"))
    posts = SN.LD.POSTS_DIR
    same, grown, bad = 0, [], []
    for name, spec in old["files"].items():
        ns = new["files"].get(name)
        if ns is None:
            bad.append({"file": name, "problem": "새 스냅샷에 없음"})
            continue
        with open(posts / name, "rb") as fh:
            head = fh.read(spec["bytes"])
        if sha_bytes(head) != spec["sha256"] or ns["bytes"] < spec["bytes"]:
            bad.append({"file": name, "problem": "final 스냅샷 앞부분이 바뀜"})
        elif ns["bytes"] == spec["bytes"]:
            same += 1
        else:
            grown.append({"file": name, "old_bytes": spec["bytes"], "new_bytes": ns["bytes"],
                          "old_lines": spec["lines"], "new_lines": ns["lines"]})
    added = sorted(set(new["files"]) - set(old["files"]))
    # 10/23 파일: 앞으로 받기가 08:00을 넘었는지(K2026-10-23 창 완전성)
    rows, _ = SN.LD.read_snapshot(posts, SP.SNAP_CUTOFF, new)
    last = [r["rec"] for r in rows if r["file"] == SP.SNAP_CUTOFF]
    dts = sorted(r["dt"] for r in last)
    out = {"S": new["S"], "cutoff": new["cutoff"], "first": new["first"], "n_files": new["n_files"],
           "n_lines": new["n_lines"], "state_high": new["state_high"],
           "final_files_same_bytes": same, "final_files_grown": grown, "final_files_problem": bad,
           "files_added": added,
           "file_2026_10_23": {"lines": len(last), "first_post_dt": dts[0] if dts else None,
                               "last_post_dt": dts[-1] if dts else None,
                               "max_no": max((r["no"] for r in last), default=None),
                               "forward_past_0800": bool(dts) and dts[-1] >= "2026-10-23 08:00"},
           "rule": "final/code/snapshot.py 그대로(완성된 줄만, 바이트 수·SHA-256). 컷오프만 2026-10-23"}
    meta_update("snapshot", out)
    assert not bad, bad
    assert out["file_2026_10_23"]["forward_past_0800"], "10/23 08:00 뒤 글이 아직 없다(K2026-10-23 창 미완)"
    print(json.dumps({k: v for k, v in out.items() if k != "final_files_grown"}, ensure_ascii=False, indent=1))
    print("grown:", [(g["file"], g["old_lines"], g["new_lines"]) for g in grown])
    return out


def cmd_calendar() -> dict:
    import exchange_calendars as xc
    LD = load_final("run_features").LD          # run_features가 감싼 _xcal_check(대조 끝 자르기)까지 같은 모듈
    src = pd.read_parquet(SP.FINAL / "results" / "loader_calendar.parquet")
    smeta = json.loads((SP.FINAL / "results" / "loader_calendar.json").read_text("utf-8"))
    man = json.loads((SP.SNAP / "snapshot_manifest.json").read_text("utf-8"))
    S = datetime.fromisoformat(man["S"])
    end = date.fromisoformat(SP.SNAP_CUTOFF)
    start = date.fromisoformat(smeta["start"])
    xk = xc.get_calendar("XKRX")
    xn = xc.get_calendar("XNYS")
    add = []
    for d in pd.date_range(pd.Timestamp(smeta["end"]) + pd.Timedelta(days=1), pd.Timestamp(end), freq="D"):
        ts = pd.Timestamp(d.date())
        k_open = xk.is_session(ts) and xk.session_open(ts).tz_convert("Asia/Seoul").tz_localize(None) <= pd.Timestamp(S)
        n_open = xn.is_session(ts) and xn.session_open(ts).tz_convert("Asia/Seoul").tz_localize(None) <= pd.Timestamp(S)
        add.append({"date": ts, "krx": bool(k_open), "nyse": bool(n_open), "nyse_early": False,
                    "krx_session": bool(xk.is_session(ts)), "nyse_session": bool(xn.is_session(ts))})
    ext = pd.DataFrame(add)
    cal = pd.concat([src, ext[["date", "krx", "nyse", "nyse_early"]]], ignore_index=True)
    cal["date"] = cal["date"].astype(src["date"].dtype)
    kr = [d.date() for d in cal.loc[cal["krx"], "date"]]
    us = [d.date() for d in cal.loc[cal["nyse"], "date"]]
    check = LD._xcal_check(start, end, kr, us)
    SP.SNAP_RES.mkdir(parents=True, exist_ok=True)
    cp = SP.SNAP_RES / "loader_calendar.parquet"
    cal.to_parquet(cp, index=False)
    m = {"source": f"{smeta['source']}(~{smeta['end']}, final 캐시 그대로) + exchange_calendars XKRX·XNYS({ext['date'].min().date()}~{end}, "
                   "스냅샷 시각 S 전에 시작한 세션만)",
         "start": smeta["start"], "end": SP.SNAP_CUTOFF, "check": check,
         "added_days": [{k: (str(v.date()) if k == "date" else v) for k, v in r.items()} for r in add],
         "S": man["S"],
         "why": "2026-10-23 09:00 KST 확인: yfinance ^KS11 막대 10/19·10/20만(10/21·10/22 없음), ^GSPC 10/21 없음. "
                "빈 거래일은 창 두 개를 하나로 합치므로 거래소 달력으로 채운다. NYSE 10/23은 S 뒤 개장이라 넣지 않는다(U2026-10-23 창 없음)",
         "final_cache_sha256": sha(SP.FINAL / "results" / "loader_calendar.parquet")}
    (SP.SNAP_RES / "loader_calendar.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), "utf-8")
    meta_update("calendar", m)
    print(json.dumps(m, ensure_ascii=False, indent=1, default=str))
    return m


def cmd_features() -> int:
    RF = load_final("run_features")
    SP.patch_final(RF)
    assert RF.RES == SP.SNAP_RES and RF.FINAL == SP.SNAP
    rc = RF.main([])
    s = json.loads((SP.SNAP_RES / "loader_summary.json").read_text("utf-8"))
    meta_update("loader", {"snapshot": s["snapshot"], "windows_kr_main": s["windows_kr"]["main"],
                           "windows_us_main": s["windows_us"]["main"], "units": s["units"],
                           "calendar_check": s["calendar"].get("check")})
    return rc


def cmd_pred() -> int:
    PWF = load_final("pred_windows_full")
    SP.patch_final(PWF)
    assert PWF.RES == SP.SNAP_RES and PWF.FINAL == SP.SNAP
    return PWF.main()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "calendar", "features", "pred"])
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        cmd_snapshot()
    elif a.cmd == "calendar":
        cmd_calendar()
    elif a.cmd == "features":
        return cmd_features()
    else:
        return cmd_pred()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
