"""최종 판정 준비 1단계: 포럼 스냅샷 목록.

posts/ 2025-08-20 ~ 2026-10-21 파일(완결된 날짜)의 완성된 줄만 읽어 파일별 바이트 수·줄 수·SHA-256을 남긴다.
수집기가 지금도 쓰는 중이라 읽기만 한다(마지막 미완성 줄은 버린다). 뒤 단계는 이 목록의 바이트만 다시 읽는다
(interim/code/loader.read_snapshot의 manifest 재현 규칙 그대로, preview/code/snapshot.py와 같은 방식).
범위 밖 파일(2025-08-18·19는 거꾸로 받기 중, 2026-10-22는 오늘)은 목록에 넣지 않는다.

산출: final/snapshot_manifest.json
실행: py snapshot.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ANT = FINAL.parent
sys.path.insert(0, str(ANT / "interim" / "code"))
import loader as LD  # noqa: E402  (고치지 않고 import)

FIRST, CUTOFF = "2025-08-20", "2026-10-21"
OUT = FINAL / "snapshot_manifest.json"


def main() -> int:
    rows, man = LD.read_snapshot(LD.POSTS_DIR, CUTOFF)
    files = {k: v for k, v in man["files"].items() if FIRST <= k[:10] <= CUTOFF}
    dropped = sorted(set(man["files"]) - set(files))
    later = sorted(p.name for p in LD.POSTS_DIR.glob("*.jsonl") if p.stem > CUTOFF)
    salt = LD.salt_check()
    days = sorted(k[:10] for k in files)
    out = {"cutoff": CUTOFF, "first": FIRST, "S": man["S"],
           "S_note": "기준 시각(KST, 읽기를 마친 시각). 모든 줄의 수집 시각 상한이며 로더의 댓글 날짜 상한 hi로 쓰인다",
           "made": datetime.now().replace(microsecond=0).isoformat(),
           "files": files, "n_files": len(files), "n_lines": int(sum(f["lines"] for f in files.values())),
           "bytes": int(sum(f["bytes"] for f in files.values())),
           "first_file": days[0], "last_file": days[-1],
           "dropped_files_outside_range": dropped, "files_after_cutoff_not_read": later,
           "salt_fingerprint": salt.get("fingerprint"), "state_high": salt.get("high"), "state_low": salt.get("low"),
           "rule": "완성된 줄(마지막 줄바꿈까지)만. 뒤 단계는 loader.read_snapshot(manifest=...)로 같은 바이트만 읽고 "
                   "SHA-256을 맞춘다. 목록에 없는 파일은 건너뛴다"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: out[k] for k in ("S", "n_files", "n_lines", "bytes", "first_file", "last_file",
                                          "dropped_files_outside_range", "files_after_cutoff_not_read",
                                          "state_high", "state_low")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
