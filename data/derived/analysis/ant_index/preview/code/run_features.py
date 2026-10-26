"""미리보기(부록 9) 2단계: 동결 로더·특징 코드를 고치지 않고 import해 이 스냅샷으로 다시 돌린다.

순서: loader.run(snapshot_manifest.json의 바이트만) → feat_activity → feat_topic → feat_text.
interim 코드는 한 줄도 바꾸지 않는다. 결과 폴더만 preview/results로 바꿔 넘긴다
(feat_activity·feat_topic은 results 인자, feat_text는 모듈 전역 RESULTS를 이 실행 안에서만 바꾼다).
사전: interim/results/lexicon_v02.json·lexicon_v02_full.json을 바이트 그대로 복사해 지문(SHA-256)을 맞춘다.
판정용 묶음은 측정 통과 묶음(LL_US, LS, MC, ANG, KX, MEM, SEMI, TILT_US, TILT_KR, ALIAS_*, PX)뿐이다.
동결 코드가 함께 만드는 감정 사전 열(E·F·S 등, 측정 실패)은 파일에 남지만 feature_registry.json에 '쓰지 않음'으로 적는다.
가격: feat_text가 H24 사건 표지용으로 KOSPI·하닉 '지난' 종가만 받는다(동결 코드 그대로). 앞날 가격은 이 단계에서 쓰지 않는다.

실행: py run_features.py [--skip-loader]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
INTERIM = ANT / "interim"
RES = PREVIEW / "results"
sys.path.insert(0, str(INTERIM / "code"))

import loader as LD  # noqa: E402

_xcal_orig = LD._xcal_check


def _xcal_clipped(start, end, kr, us):
    """컷오프(10-17)가 토요일이라 exchange_calendars가 end(세션이 아닌 날)를 받지 못한다(구간 끝 = 마지막 세션 규칙).
    대조 구간 끝만 두 달력 모두 end 이하 마지막 세션으로 줄인다(10-17은 토요일이라 거래일 차이 판정은 같다).
    달력 원천(yfinance 날짜)은 그대로다."""
    import exchange_calendars as xc
    import pandas as pd
    end2 = min(xc.get_calendar(c).date_to_session(pd.Timestamp(end), "previous") for c in ("XKRX", "XNYS")).date()
    out = _xcal_orig(start, end2, [d for d in kr if d <= end2], [d for d in us if d <= end2])
    out["check_end_clipped_to"] = str(end2)
    return out


LD._xcal_check = _xcal_clipped


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loader", action="store_true")
    ap.add_argument("--only", default="loader,activity,topic,text")
    a = ap.parse_args(argv)
    steps = set(a.only.split(","))
    RES.mkdir(parents=True, exist_ok=True)
    for name in ("lexicon_v02.json", "lexicon_v02_full.json"):
        src, dst = INTERIM / "results" / name, RES / name
        if not dst.exists() or sha(dst) != sha(src):
            shutil.copyfile(src, dst)
        assert sha(dst) == sha(src)
    man_path = PREVIEW / "snapshot_manifest.json"
    man = json.loads(man_path.read_text("utf-8"))
    timings = {}
    if "loader" in steps and not a.skip_loader:
        t = time.time()
        log("loader")
        s = LD.run(man["cutoff"], LD.POSTS_DIR, RES, man_path)
        timings["loader"] = round(time.time() - t, 1)
        log(f"loader done {timings['loader']}s windows_kr={s['windows_kr']['main']['n_windows']} "
            f"windows_us={s['windows_us']['main']['n_windows']}")
    if "activity" in steps:
        import feat_activity as FA
        t = time.time()
        log("feat_activity")
        b = FA.build(RES)
        FA.write(b, RES)
        timings["feat_activity"] = round(time.time() - t, 1)
        log(f"feat_activity done {timings['feat_activity']}s")
    if "topic" in steps:
        import feat_topic as FT
        t = time.time()
        log("feat_topic")
        FT.run(RES, rebuild_hits=True)
        timings["feat_topic"] = round(time.time() - t, 1)
        log(f"feat_topic done {timings['feat_topic']}s")
    if "text" in steps:
        import feat_text as FX
        t = time.time()
        log("feat_text")
        FX.RESULTS = RES          # 이 실행 안에서만(파일은 그대로)
        FX.run()
        timings["feat_text"] = round(time.time() - t, 1)
        log(f"feat_text done {timings['feat_text']}s")
    tp = RES / "run_features_timings.json"
    old = json.loads(tp.read_text("utf-8")) if tp.exists() else {}
    old.update(timings)
    old["code_sha256"] = {p.name: sha(p) for p in sorted((INTERIM / "code").glob("*.py"))
                          if p.name in ("loader.py", "feat_activity.py", "feat_topic.py", "feat_text.py",
                                        "lexicon_v02.py", "lexicon_v02_final.py")}
    tp.write_text(json.dumps(old, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
