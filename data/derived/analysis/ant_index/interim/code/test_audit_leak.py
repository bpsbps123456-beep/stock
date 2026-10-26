"""시점 누출 감사(audit_leak.py) 회귀 테스트: 잘림 표지는 첫 줄, 댓글 날짜는 늦게만 틀림.

test_loader의 가짜 jsonl을 그대로 쓴다(글 104: 첫 줄은 안 잘렸고, 다시 받은 줄은 ncomment 5 > 저장 2로 잘렸다).
실행: py -m unittest test_audit_leak   (이 폴더에서). 네트워크·실제 스냅샷을 쓰지 않는다.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_leak as A  # noqa: E402
import feat_activity as FA  # noqa: E402
import loader  # noqa: E402
import test_loader as TL  # noqa: E402


class TruncAndDateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        for day, recs in TL.FILES.items():
            with open(d / f"{day}.jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cls.rows, cls.man = loader.read_snapshot(d, "2026-10-20", now=TL.S)
        cls.b = loader.build(cls.rows, cls.man, TL.CAL, None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_trunc_share_uses_first_row(self):
        w = self.b["windows_kr"]
        w = w[(w["win_id"] == "K2026-10-15")].iloc[0]
        self.assertEqual(w["n_posts"], 3)            # 103(10/14 08:00), 104(10/14 22:00), 105(10/15 07:00)
        self.assertEqual(float(w["trunc_share"]), 0.0)   # 104는 첫 줄에서 안 잘렸다(마지막 줄 기준이면 1/3)
        legacy = A.legacy_trunc(self.b)["windows_kr"]
        self.assertAlmostEqual(float(legacy.loc[legacy["win_id"] == "K2026-10-15", "trunc_share"].iloc[0]), 1 / 3)

    def test_fight_filter_uses_first_row(self):
        u = self.b["units"][FA.UNIT_COLS]
        f = FA.activity_features(FA.window_frame(self.b["windows_kr"]), u, "win_kr", "pwin_kr", "pre_close_kr")
        r = f[f["win_id"] == "K2026-10-15"].iloc[0]
        self.assertEqual(r["n_posts_nontrunc"], 3)   # 다시 받은 줄의 ncomment로 104를 빼지 않는다(고치기 전 2)

    def test_dates_only_later_without_match(self):
        S = datetime.fromisoformat(self.man["S"])
        lo, hi, _ = A.row_bounds(self.rows, S, True)
        on = A.tod_dates(self.rows, lo, hi, True)
        lo0, hi0, _ = A.row_bounds(self.rows, S, False)
        off = A.tod_dates(self.rows, lo0, hi0, False)
        self.assertTrue(on)
        for k, v in on.items():
            self.assertGreaterEqual(off[k], v, k)        # 맞춤을 끄면 같거나 늦어진다(이르게 놓이지 않는다)
        c1 = (104, "x0000000000x", "c1", "22:10:00")
        self.assertEqual(str(on[c1]), "2026-10-14")
        self.assertGreater(off[c1], on[c1])

    def test_compare_treats_missing_as_equal_and_detects_change(self):
        T = pd.Timestamp("2026-10-16 08:00")
        base = {k: pd.DataFrame() for k in A.KEYS}
        df = pd.DataFrame({"win_id": ["K1", "K2"], "close_kst": pd.to_datetime(["2026-10-15 08:00"] * 2),
                           "x": [1.0, float("nan")], "s": ["a", None]})
        for k in A.KEYS:
            base[k] = df.assign(table="t", stock="s0", market="kr", week_close_day=df["close_kst"], sat=df["close_kst"])
        base["fto_stock"] = base["fto_stock"].drop(columns="close_kst")   # 종목 표는 짝 창(fto)의 마감으로 고른다
        same = A.compare(base, {k: v.copy() for k, v in base.items()}, T)
        self.assertTrue(all(not v["diff"] for v in same.values()))
        pert = {k: v.copy() for k, v in base.items()}
        pert["ft"].loc[0, "x"] = 2.0
        self.assertIn("x", A.compare(base, pert, T)["ft"]["diff"])


if __name__ == "__main__":
    unittest.main()
