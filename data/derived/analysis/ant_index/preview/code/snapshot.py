"""미리보기(부록 9) 1단계: 포럼 스냅샷 목록.

posts/ 2026-05-28 ~ 2026-10-17 파일의 완성된 줄만 읽어 파일별 바이트 수·줄 수·SHA-256을 남긴다.
수집기가 지금도 쓰는 중이라 읽기만 한다(마지막 미완성 줄은 버린다). 뒤 단계는 이 목록의 바이트만 다시 읽는다
(interim/code/loader.read_snapshot의 manifest 재현 규칙 그대로).

산출: preview/snapshot_manifest.json
실행: py snapshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
sys.path.insert(0, str(ANT / "interim" / "code"))
import loader as LD  # noqa: E402  (고치지 않고 import)

FIRST, CUTOFF = "2026-05-28", "2026-10-17"
OUT = PREVIEW / "snapshot_manifest.json"


def main() -> int:
    rows, man = LD.read_snapshot(LD.POSTS_DIR, CUTOFF)
    files = {k: v for k, v in man["files"].items() if FIRST <= k[:10] <= CUTOFF}
    dropped = sorted(set(man["files"]) - set(files))
    salt = LD.salt_check()
    out = {"cutoff": CUTOFF, "first": FIRST, "S": man["S"], "files": files,
           "n_files": len(files), "n_lines": int(sum(f["lines"] for f in files.values())),
           "bytes": int(sum(f["bytes"] for f in files.values())),
           "dropped_files_outside_range": dropped,
           "salt_fingerprint": salt.get("fingerprint"), "state_high": salt.get("high"), "state_low": salt.get("low"),
           "rule": "완성된 줄(마지막 줄바꿈까지)만. 뒤 단계는 loader.read_snapshot(manifest=...)로 같은 바이트만 읽고 "
                   "SHA-256을 맞춘다. 목록에 없는 파일(스냅샷 뒤 생긴 날)은 건너뛴다"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: out[k] for k in ("S", "n_files", "n_lines", "bytes", "dropped_files_outside_range",
                                          "state_high", "state_low")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
