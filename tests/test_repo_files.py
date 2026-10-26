"""GitHub에 올리는 저장소 파일 점검(AGENTS.md 공동 작업 규칙 '종결').

- 추적 파일은 50MB를 넘지 않는다(GitHub 한도 100MB, 경고 50MB).
- 개인 매매 기록(data/journal/)은 전략 설정 settings.json만 추적한다.
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 50 * 1024 * 1024


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


@unittest.skipIf(shutil.which("git") is None or not (ROOT / ".git").exists(), "git 저장소가 아님")
class RepoFilesTest(unittest.TestCase):
    def test_no_tracked_file_over_limit(self):
        big = [p for p in tracked() if (ROOT / p).is_file() and (ROOT / p).stat().st_size > LIMIT]
        self.assertEqual(big, [], "50MB 넘는 추적 파일은 .gitignore에 넣고 코드로 다시 만든다")

    def test_journal_only_settings(self):
        journal = [p for p in tracked() if p.startswith("data/journal/")]
        self.assertTrue(set(journal) <= {"data/journal/settings.json"}, journal)


if __name__ == "__main__":
    unittest.main()
