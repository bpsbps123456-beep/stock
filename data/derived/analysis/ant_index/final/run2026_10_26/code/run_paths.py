"""부록 15 전체 분석 실행(2026-10-26) 경로·상수.

final/code·interim/code·events/code 모듈은 한 줄도 고치지 않고 불러 쓴다. 경로 상수(FINAL·RES·SHARDS)와
스냅샷 범위·시드만 이 실행 폴더(final/run2026_10_26/)를 가리키게 한다(로직은 그대로).
  snapshot_manifest.json   스냅샷 목록(final/code/snapshot.py 그대로, 범위만 2025-01-01 ~ 2026-10-23)
  results/                 로더·feat_activity·feat_topic·feat_text 산출(final/code/run_features.py 그대로)
  label_shards/            새 라벨 표본 조각(창 2025-01-02 ~ 2025-08-20, 두 층을 같은 표본에 매김)
  label_key.parquet        새 표본 대응표(라벨러에게 주지 않음)
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True        # final/·interim/·events/ 코드 폴더에 캐시를 쓰지 않는다

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                      # final/run2026_10_26
FINAL = RUN.parent                     # final/(코드 원천, 읽기만)
ANT = FINAL.parent
EV = ANT / "events"
SNAP1023 = EV / "snapshot_1023"        # 앞선 사건 연구 스냅샷(2025-08-20 ~ 2026-10-23 07:59, 읽기만)
RES = RUN / "results"
SHARDS = RUN / "label_shards"
META = RUN / "update_meta.json"

FIRST, CUTOFF = "2025-01-01", "2026-10-23"          # 부록 15: 2025-01-01 ~ 2026-10-23 완결 창
NEW_RANGE = ("2025-01-02", "2025-08-20")            # 새로 뽑는 W_KR 창(D 기준)
CHECK_WINDOWS = ["K2026-10-22", "K2026-10-23"]      # 예측 라벨이 없으면 새로 뽑는 창
SEED = 20261026
TAG = "2026-10-26"
ID_PREFIX = "Y"
JUDGE_SHA256 = "ddb1dec40a91364e148c4eb411bd184fd6b7da7a67efcb3affa8519a99575c8c"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_final(mod, *, results: bool = True) -> None:
    """final/code 모듈의 경로 상수(FINAL·RES·SHARDS)만 이 실행 폴더로 바꾼다."""
    if hasattr(mod, "FINAL"):
        mod.FINAL = RUN
    if results and hasattr(mod, "RES"):
        mod.RES = RES
    if hasattr(mod, "SHARDS"):
        mod.SHARDS = SHARDS


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
