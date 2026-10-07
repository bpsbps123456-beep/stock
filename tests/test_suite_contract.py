"""Suite contract: `py -m unittest` and `py -m pytest` must collect the same tests."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent


class SuiteContractTests(unittest.TestCase):
    def test_module_level_test_functions_are_visible_to_unittest(self) -> None:
        # unittest는 TestCase만 모은다. 모듈 함수(test_*)를 쓰는 파일에 load_tests가 없으면
        # pytest(Codex)만 그 테스트를 돌리고 unittest(Claude·문서 명령)는 건너뛴다.
        missing = []
        for path in sorted(TESTS.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            if any(name.startswith("test_") for name in names) and "load_tests" not in names:
                missing.append(path.name)
        self.assertEqual(missing, [], "TestCase로 쓰거나 load_tests 훅을 두세요")


if __name__ == "__main__":
    unittest.main()
