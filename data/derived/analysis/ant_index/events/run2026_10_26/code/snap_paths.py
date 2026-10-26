"""부록 15 전체 분석 실행(2026-10-26)용 snap_paths — events/code/snap_paths.py와 같은 이름·같은 꼴.

events/code 모듈(views_fetch·feat_views·features_all)은 `import snap_paths as SP`로 스냅샷 위치를 읽는다. 이 파일을
그 이름으로 먼저 찾게 해(run_events.py가 이 폴더를 sys.path 맨 앞에 둔다) 로직은 한 줄도 고치지 않고 위치만 바꾼다.
  SNAP      final/run2026_10_26/pred_all   예측 층 대응표·라벨 모음(final/run2026_10_26/code/pred_all.py)
                                           + 스냅샷 목록 바이트 사본
  SNAP_RES  final/run2026_10_26/results    로더·feat_activity·feat_topic·feat_text·pred_windows_full 산출,
                                           views_fetch·feat_views 산출도 여기(snapshot_1023/results와 같은 꼴)
스냅샷: 2025-01-01 ~ 2026-10-23 파일(S 2026-10-26 02:38). W_KR K2025-01-03 ~ K2026-10-23, W_U U2025-01-03 ~ U2026-10-23.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True        # final/·interim/·preview/·events/code 폴더에 캐시를 쓰지 않는다

HERE = Path(__file__).resolve().parent          # events/run2026_10_26/code
RUN_EV = HERE.parent                             # events/run2026_10_26
EV = RUN_EV.parent                               # events(원래 코드·라벨, 읽기만)
ANT = EV.parent
FINAL = ANT / "final"                            # 동결 코드 원천(읽기만)
RUN_FINAL = FINAL / "run2026_10_26"
SNAP = RUN_FINAL / "pred_all"
SNAP_RES = RUN_FINAL / "results"
SNAP_CUTOFF = "2026-10-23"
SNAP_END_KST = "2026-10-24 00:00"                # 10/23 파일까지(U2026-10-23 21:30 마감 창까지 완전)
EXPECT_WINDOWS = {"kr": 438, "us": 453}          # 이 스냅샷 로더의 본판 완전 창 수
NEW_KR_WINDOWS: list[str] = []                   # 이번 실행은 새 표본을 뽑지 않는다(final/run2026_10_26/code/sample.py에서 끝)


def patch_final(mod, *, results: bool = True) -> None:
    """final/code 모듈의 경로 상수(FINAL·RES·SHARDS)만 이 실행으로 바꾼다."""
    if hasattr(mod, "FINAL"):
        mod.FINAL = SNAP
    if results and hasattr(mod, "RES"):
        mod.RES = SNAP_RES
    if hasattr(mod, "SHARDS"):
        mod.SHARDS = SNAP / "label_shards"
