"""사건 연구가 쓰는 포럼 스냅샷 위치(부록 14 갱신, 2026-10-23).

final/는 부록 12 판정용 동결 스냅샷(2025-08-20 ~ 2026-10-21 파일)이라 고치지 않는다. 사건 연구는 같은 코드로 스냅샷만
2026-10-23 07:59 KST까지 늘린 events/snapshot_1023/(final/과 같은 꼴)을 쓴다.
  snapshot_1023/snapshot_manifest.json   스냅샷 목록(final/code/snapshot.py 그대로, 컷오프만 2026-10-23)
  snapshot_1023/results/                 로더·feat_activity·feat_topic·feat_text·pred_windows_full 산출
  snapshot_1023/label_key.parquet        예측 층 대응표 = final 대응표 + 새 창(K2026-10-22·23) 표본
  snapshot_1023/labels/                  예측 층 라벨 = final/labels 사본 + 새 창 라벨(shard_044.json)
final/code 모듈은 고치지 않고 불러 쓰며, 경로 상수만 patch_final()로 이 스냅샷을 가리키게 한다(로직은 그대로).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True        # final/·interim/·preview/ 코드 폴더에 캐시를 쓰지 않는다

HERE = Path(__file__).resolve().parent
EV = HERE.parent
ANT = EV.parent
FINAL = ANT / "final"                  # 동결 판정 스냅샷(코드 원천, 읽기만)
SNAP = EV / "snapshot_1023"            # 사건 연구 스냅샷
SNAP_RES = SNAP / "results"
SNAP_CUTOFF = "2026-10-23"             # 이 날짜 이하 파일(10/23 파일은 읽은 시각 S까지의 완성된 줄)
SNAP_END_KST = "2026-10-23 08:00"      # 이 시각에 닫힌 창까지 완전(K2026-10-23 마감 07:59)
EXPECT_WINDOWS = {"kr": 285, "us": 295}   # W_KR K2025-08-21~K2026-10-23, W_U U2025-08-21~U2026-10-22
NEW_KR_WINDOWS = ["K2026-10-22", "K2026-10-23"]   # final 뒤에 더한 W_KR 창(라벨 표본을 새로 뽑는 창)


def patch_final(mod, *, results: bool = True) -> None:
    """final/code 모듈의 경로 상수(FINAL·RES·SHARDS)만 이 스냅샷으로 바꾼다."""
    if hasattr(mod, "FINAL"):
        mod.FINAL = SNAP
    if results and hasattr(mod, "RES"):
        mod.RES = SNAP_RES
    if hasattr(mod, "SHARDS"):
        mod.SHARDS = SNAP / "label_shards"
