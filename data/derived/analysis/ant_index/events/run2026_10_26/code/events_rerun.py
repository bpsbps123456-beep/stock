"""부록 15 사건 연구 1단계: 사건 목록(가격만)을 2025-01-01 ~ 2026-10-23으로 다시 판정한다.

events/code/events.py를 한 줄도 고치지 않고 불러 쓴다. 바꾸는 것은 모듈 상수뿐이다.
  RESULTS·PRICES_CSV·FETCH_META·EVENTS_JSON  → events/run2026_10_26/results/
  EVENT_START  2025-08-21 → 2025-01-01  (부록 15: 포럼 스냅샷이 2025-01-01부터라 사건 기간을 앞으로 늘린다.
                                          사건 연구에서 쓰는 기간은 2025-01-21 이후 — event_study_rerun.py가 거른다)
  FINAL_RESULTS(창 표·달력)는 이 폴더의 snap_paths(final/run2026_10_26/results)에서 자동으로 온다.
끝 극값은 10/23 종가로 다시 판정한다(events.py fetch가 10/22·10/23 마감 뒤 종가를 넣고, build가 rejudge_tail을 돈다).

덧붙이는 것(사건 목록은 바꾸지 않는다, results/events_rerun_compare.json):
  - 앞선 실행(events/results/events.json, 10/22 종가)과 2025-08-21 뒤 꺾임점·가짜·끝 극값 비교
  - 앞선 종가 스냅샷과 겹치는 날 종가 비교
  - 예열 점검(설명용): 2024-06-03부터의 yfinance 원값을 앞에 붙여 지그재그를 다시 돌렸을 때 2025-01-21 뒤 꺾임점이 같은지
    (events.py robustness의 warmup_from_* 칸은 기간 시작이 2025-01-01이 되어 그 앞 꺾임점이 빠지므로 '같지 않음'으로 나온다)
실행: py -B events_rerun.py all | build | compare
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import pandas as pd  # noqa: E402

import snap_paths as SP  # noqa: E402

assert Path(SP.__file__).resolve().parent == HERE, "이 실행의 snap_paths가 아님"
OUT = SP.RUN_EV / "results"
OLD = SP.EV / "results"
NEW_START = "2025-01-01"
STUDY_START = "2025-01-21"
OLD_START = "2025-08-21"
WARM_EXT_START = "2024-06-03"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_events_module():
    spec = importlib.util.spec_from_file_location("events_code_events", SP.EV / "code" / "events.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclass가 모듈을 sys.modules에서 찾는다
    spec.loader.exec_module(mod)
    assert mod.SP is SP, "events.py가 이 실행의 snap_paths를 쓰지 않는다"
    assert mod.FINAL_RESULTS == SP.SNAP_RES
    mod.RESULTS = OUT
    mod.PRICES_CSV = OUT / "prices_close.csv"
    mod.FETCH_META = OUT / "prices_fetch_meta.json"
    mod.EVENTS_JSON = OUT / "events.json"
    mod.EVENT_START = NEW_START
    return mod


def annotate(EVM) -> None:
    """events.json meta에 이 실행의 실제 위치·상수를 덧붙인다(사건 목록·정의는 그대로)."""
    p = EVM.EVENTS_JSON
    ev = json.loads(p.read_text("utf-8"))
    ev["meta"]["forum_snapshot"]["path"] = str((SP.SNAP / "snapshot_manifest.json").relative_to(SP.ANT)).replace("\\", "/")
    ev["meta"]["rerun_2026_10_26"] = {
        "addendum": "HYPOTHESES_addendum15.md",
        "code": "events/code/events.py (변경 없음, sha256 " + sha(SP.EV / "code" / "events.py") + ")",
        "wrapper": "events/run2026_10_26/code/events_rerun.py (sha256 " + sha(Path(__file__)) + ")",
        "constants_changed": {"EVENT_START": [OLD_START, NEW_START],
                              "RESULTS": ["events/results", "events/run2026_10_26/results"]},
        "windows_and_calendar": str(SP.SNAP_RES.relative_to(SP.ANT)).replace("\\", "/"),
        "study_period_in_event_study": [STUDY_START, EVM.EVENT_END],
        "note": ("definitions의 'event_dates'·'win_id' 설명과 fetch meta의 'calendar' 설명 문자열은 events.py에 박힌 앞선 실행 문구다. "
                 "이 실행의 기간은 meta.period, 창 표·달력은 위 windows_and_calendar 경로다. robustness.warmup_from_*는 기간 시작이 "
                 "2025-01-01이라 그 앞 꺾임점이 빠져 '같지 않음'으로 나온다 — 예열 점검은 events_rerun_compare.json의 warmup_extended를 본다."),
    }
    p.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")


def _piv(t, lo=None, hi=None):
    return [(p["kind"], p["date"], p["confirm_date"]) for p in t["pivots"]
            if (lo is None or p["date"] >= lo) and (hi is None or p["date"] <= hi)]


def compare(EVM) -> dict:
    new = json.loads(EVM.EVENTS_JSON.read_text("utf-8"))
    old = json.loads((OLD / "events.json").read_text("utf-8"))
    out = {"made": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
           "new_events_json_sha256": sha(EVM.EVENTS_JSON), "old_events_json_sha256": sha(OLD / "events.json"),
           "assets": {}}
    for asset, a in new["assets"].items():
        oa = old["assets"][asset]
        blk = {"data_through": {"new": a["data_through"], "old": oa["data_through"]}, "thresholds": {}}
        for X, t in a["thresholds"].items():
            ot = oa["thresholds"][X]
            np_, op_ = _piv(t, OLD_START), _piv(ot, OLD_START)
            # 앞선 실행의 확정 꺾임점(2025-08-21 뒤)이 새 목록에 그대로 있는가 + 새로 확정된 것
            only_new = [x for x in np_ if x not in op_]
            only_old = [x for x in op_ if x not in np_]
            fb_n = sorted(f["date"] for f in t["fake_bottoms"] if f["date"] >= OLD_START)
            fb_o = sorted(f["date"] for f in ot["fake_bottoms"])
            ft_n = sorted(f["date"] for f in t["fake_tops"] if f["date"] >= OLD_START)
            ft_o = sorted(f["date"] for f in ot["fake_tops"])
            early = _piv(t, None, "2025-08-20")
            blk["thresholds"][X] = {
                "pivots_from_2025_08_21": {"new": len(np_), "old": len(op_), "only_new": only_new, "only_old": only_old},
                "pivots_2025_01_01_to_08_20": {"all": early,
                                               "study_period_from_2025_01_21": [x for x in early if x[1] >= STUDY_START]},
                "fake_bottoms_from_2025_08_21": {"new": len(fb_n), "old": len(fb_o),
                                                 "only_new": [d for d in fb_n if d not in fb_o],
                                                 "only_old": [d for d in fb_o if d not in fb_n]},
                "fake_tops_from_2025_08_21": {"new": len(ft_n), "old": len(ft_o),
                                              "only_new": [d for d in ft_n if d not in ft_o],
                                              "only_old": [d for d in ft_o if d not in ft_n]},
                "tail": {"new": {k: (t["tail_unconfirmed"] or {}).get(k) for k in
                                 ("kind", "date", "close", "status", "confirm_level", "last_date", "last_close", "move_from_tail")},
                         "old": {k: (ot["tail_unconfirmed"] or {}).get(k) for k in ("kind", "date", "close", "status", "confirm_level")}},
                "rejudge": {k: t["rejudge"].get(k) for k in ("deadline", "data_through", "deadline_close_in", "stage")},
                "held_near_tail": {k: [h["date"] for h in v] for k, v in t["held_near_tail"].items()},
            }
        out["assets"][asset] = blk
    # 종가 비교
    pn = pd.read_csv(EVM.PRICES_CSV, dtype={"date": str})
    po = pd.read_csv(OLD / "prices_close.csv", dtype={"date": str})
    m = pn.merge(po, on=["asset", "date"], suffixes=("_new", "_old"))
    diff = m[(m["close_new"] / m["close_old"] - 1).abs() > EVM.MATCH_REL]
    out["prices_vs_old_snapshot"] = {
        "overlap_rows": int(len(m)), "differ_rows": int(len(diff)),
        "differ": diff[["asset", "date", "close_new", "close_old", "source_new", "source_old"]].to_dict("records"),
        "new_rows_not_in_old": pn.merge(po[["asset", "date"]], on=["asset", "date"], how="left", indicator=True)
        .query("_merge == 'left_only'")[["asset", "date", "close", "source"]].to_dict("records")}
    out["warmup_extended"] = warmup_extended(EVM, pn)
    (OUT / "events_rerun_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    return out


def warmup_extended(EVM, pn: pd.DataFrame) -> dict:
    """설명용: 2024-06-03 ~ 2024-12-31 yfinance 원값(대조 안 함)을 앞에 붙여 지그재그를 다시 돌린다. 2025-01-21 뒤 꺾임점이 같은가."""
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        return {"status": f"yfinance 없음: {exc}"}
    res = {"rule": "앞 구간은 yfinance Close 원값(auto_adjust=False), 2025-01-02 뒤는 사건 목록과 같은 종가. 사건 목록은 바꾸지 않는다",
           "start": WARM_EXT_START, "assets": {}}
    for asset, cfg in EVM.ASSETS.items():
        try:
            d = yf.download(cfg["ticker"], start=WARM_EXT_START, end=NEW_START, auto_adjust=False, progress=False,
                            multi_level_index=False, actions=False, threads=False)
            pre = pd.DataFrame({"date": pd.to_datetime(d.index).strftime("%Y-%m-%d"), "close": d["Close"].astype(float).to_numpy()})
            pre = pre.dropna()
        except Exception as exc:  # noqa: BLE001
            res["assets"][asset] = {"status": f"받기 실패: {str(exc)[:120]}"}
            continue
        p = pn[pn["asset"] == asset].sort_values("date")
        p = p[p["date"] <= EVM.EVENT_END]
        dates = pre["date"].tolist() + p["date"].tolist()
        closes = pre["close"].tolist() + p["close"].astype(float).tolist()
        blk = {"pre_rows": int(len(pre)), "pre_first": pre["date"].iloc[0] if len(pre) else None}
        for key, x, _ in EVM.THRESHOLDS:
            base_pv, _t = EVM.zigzag(p["close"].astype(float).tolist(), x)
            bd = p["date"].tolist()
            base = [(q.kind, bd[q.idx]) for q in base_pv if bd[q.idx] >= STUDY_START]
            ext_pv, _t2 = EVM.zigzag(closes, x)
            ext = [(q.kind, dates[q.idx]) for q in ext_pv if dates[q.idx] >= STUDY_START]
            blk[key] = {"same_from_2025_01_21": ext == base, "only_extended": [a for a in ext if a not in base],
                        "only_used": [b for b in base if b not in ext],
                        "first_pivots_used": [(q.kind, bd[q.idx]) for q in base_pv[:3]],
                        "first_pivots_extended_2025": [(q.kind, dates[q.idx]) for q in ext_pv if dates[q.idx] >= NEW_START][:3]}
        res["assets"][asset] = blk
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "build", "compare"])
    a = ap.parse_args(argv)
    EVM = load_events_module()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("all", "build"):
        argv2 = ["all"] if a.cmd == "all" else ["build"]
        EVM.main(argv2)
        annotate(EVM)
    c = compare(EVM)
    for asset, b in c["assets"].items():
        for X, t in b["thresholds"].items():
            print(asset, X, "pivots>=08-21 new/old", t["pivots_from_2025_08_21"]["new"], t["pivots_from_2025_08_21"]["old"],
                  "only_new", t["pivots_from_2025_08_21"]["only_new"], "only_old", t["pivots_from_2025_08_21"]["only_old"],
                  "| study early", len(t["pivots_2025_01_01_to_08_20"]["study_period_from_2025_01_21"]),
                  "| tail", t["tail"]["new"]["kind"], t["tail"]["new"]["date"], t["rejudge"]["stage"])
    print("prices differ rows", c["prices_vs_old_snapshot"]["differ_rows"])
    print(json.dumps(c["warmup_extended"], ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
