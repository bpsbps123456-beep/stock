"""판정용 로더 테스트: 작은 가짜 jsonl로 창 경계·주말·첫 줄·댓글 날짜 확정·마감 뒤 댓글 제외를 본다.

실행: py -m unittest test_loader   (이 폴더에서)   또는   py test_loader.py
네트워크를 쓰지 않는다(거래일 달력을 직접 넣는다).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loader  # noqa: E402

KR_DAYS = [date(2026, 10, d) for d in (12, 13, 14, 15, 16, 19, 20)]
CAL = {"krx": KR_DAYS, "nyse": KR_DAYS, "nyse_early": set(), "source": "test"}
S = datetime(2026, 10, 21, 12, 0)


def post(no, dt, title="t", body="b", comments=(), ncomment=None, a="author00000a"):
    cs = [{"dt": d, "text": t, "up": 9, "down": 9, "depth": dep, "a": ca} for d, t, dep, ca in comments]
    return {"no": no, "dt": dt, "title": title, "body": body, "views": 999, "up": 99, "down": 9,
            "ncomment": len(cs) if ncomment is None else ncomment, "comments": cs, "a": a}


FILES = {
    "2026-10-13": [post(100, "2026-10-13 07:00")],
    "2026-10-14": [
        post(101, "2026-10-14 07:59"),
        post(103, "2026-10-14 08:00"),  # 102는 지운 글(결번)
        # 104 첫 줄(10/14 23:30 수집): 오늘 댓글은 시각만
        post(104, "2026-10-14 22:00", title="첫제목", body="첫본문",
             comments=[("22:10:00", "c1", 0, "x0000000000x"), ("22:20:00", "c2", 1, "y0000000000y")]),
        # 104 다시 받은 줄(10/15 22:30): c1은 날짜가 붙고, c2는 지워졌고, c3(10/15 01:30)은 시각만
        post(104, "2026-10-14 22:00", title="고친제목", body="고친본문", ncomment=5,
             comments=[("2026-10-14 22:10:00", "c1", 0, "x0000000000x"), ("01:30:00", "c3", 0, "z0000000000z")]),
    ],
    "2026-10-15": [
        # 106: 한 줄뿐, 시각만 댓글(맞출 줄 없음). 뒤 줄(107 첫 줄)의 수집 날짜가 확정되면 범위가 좁혀진다
        post(106, "2026-10-15 10:00", comments=[("10:30:00", "x106", 0, "r0000000000r")]),
        post(107, "2026-10-15 11:00", comments=[("11:05:00", "x107", 0, "s0000000000s")]),
        post(107, "2026-10-15 11:00", comments=[("2026-10-15 11:05:00", "x107", 0, "s0000000000s")]),
        # 105는 10/16 이후에 받은 줄이다(날짜 붙은 10/15 댓글). 파일 줄 순서 = 수집 순서라 맨 뒤에 둔다
        post(105, "2026-10-15 07:00",
             comments=[("2026-10-15 07:59:59", "before", 0, "p0000000000p"),
                       ("2026-10-15 08:00:00", "after", 0, "q0000000000q")]),
    ],
    "2026-10-17": [post(108, "2026-10-17 12:00")],  # 토요일 → 월요일 창
}


class LoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        for day, recs in FILES.items():
            with open(d / f"{day}.jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(d / "2026-10-17.jsonl", "a", encoding="utf-8") as f:
            f.write('{"no": 109, "dt": "2026-10-17 13:')  # 수집기가 쓰는 중인 미완성 줄
        with open(d / "2026-10-25.jsonl", "w", encoding="utf-8") as f:  # 컷오프 뒤 파일
            f.write(json.dumps(post(200, "2026-10-25 09:00")) + "\n")
        cls.dir = d
        rows, man = loader.read_snapshot(d, "2026-10-20", now=S)
        cls.rows, cls.man = rows, man
        cls.b = loader.build(rows, man, CAL, None)
        cls.u = cls.b["units"]
        cls.wk = cls.b["windows_kr"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def unit(self, utype, no, text=None):
        u = self.u[(self.u["utype"] == utype) & (self.u["no"] == no)]
        if text is not None:
            u = u[u["text"] == text]
        self.assertEqual(len(u), 1, (utype, no, text))
        return u.iloc[0]

    def win(self, wid):
        w = self.wk[self.wk["win_id"] == wid]
        self.assertEqual(len(w), 1, wid)
        return w.iloc[0]

    def test_snapshot_skips_partial_line_and_later_files(self):
        self.assertEqual(sum(f["lines"] for f in self.man["files"].values()), 10)
        self.assertNotIn("2026-10-25.jsonl", self.man["files"])
        self.assertFalse((self.u["no"] == 109).any())
        self.assertFalse((self.u["no"] == 200).any())

    def test_manifest_rereads_same_bytes(self):
        with open(self.dir / "2026-10-15.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(post(110, "2026-10-15 23:00")) + "\n")
        rows2, man2 = loader.read_snapshot(self.dir, "2026-10-20", manifest=self.man)
        self.assertEqual(len(rows2), len(self.rows))
        self.assertEqual(man2["S"], self.man["S"])

    def test_window_boundary_0759_in_0800_next(self):
        self.assertEqual(self.unit("post", 101)["win_kr"], "K2026-10-14")
        self.assertEqual(self.unit("post", 103)["win_kr"], "K2026-10-15")
        w = self.win("K2026-10-14")
        self.assertEqual(w["open_kst"], pd.Timestamp("2026-10-13 08:00"))
        self.assertEqual(w["close_kst"], pd.Timestamp("2026-10-14 08:00"))
        self.assertEqual(w["kind"], "weekday")
        self.assertEqual(w["length_h"], 24)

    def test_weekend_to_monday(self):
        self.assertEqual(self.unit("post", 108)["win_kr"], "K2026-10-19")
        w = self.win("K2026-10-19")
        self.assertEqual(w["kind"], "weekend")
        self.assertEqual(w["length_h"], 72)
        self.assertFalse(w["over96"])
        self.assertTrue(pd.isna(self.unit("post", 108)["win_kr24"]))  # 고정 24시간판(일 08:00~월 07:59)에는 없다

    def test_incomplete_first_window_has_no_id(self):
        self.assertTrue(pd.isna(self.unit("post", 100)["win_kr"]))
        self.assertFalse(self.win("K2026-10-13")["complete"])
        self.assertTrue(self.win("K2026-10-14")["complete"])

    def test_first_line_title_body(self):
        p = self.unit("post", 104)
        self.assertEqual(p["title"], "첫제목")
        self.assertEqual(p["text"], "첫본문")
        self.assertTrue(p["body_edited"])
        self.assertEqual(p["n_rows"], 2)
        self.assertTrue(p["trunc_last"])  # ncomment 5 > 저장 2
        self.assertFalse(p["trunc_first"])

    def test_numeric_fields_not_in_units(self):
        for col in ("views", "up", "down"):
            self.assertNotIn(col, self.u.columns)

    def test_comment_date_resolution(self):
        c1 = self.unit("comment", 104, "c1")
        self.assertEqual((c1["dt_kst"], c1["dt_conf"]), (pd.Timestamp("2026-10-14 22:10:00"), "matched"))
        c2 = self.unit("comment", 104, "c2")  # 같은 줄의 c1이 맞춰져 그 줄 수집 날짜가 10/14로 정해진다
        self.assertEqual((c2["dt_kst"], c2["dt_conf"]), (pd.Timestamp("2026-10-14 22:20:00"), "row"))
        c3 = self.unit("comment", 104, "c3")  # 맞출 수 없음 → 가능한 가장 늦은 날짜(S = 10/21)
        self.assertEqual((c3["dt_kst"], c3["dt_conf"]), (pd.Timestamp("2026-10-21 01:30:00"), "latest"))
        self.assertTrue(pd.isna(c3["win_kr"]))  # 늦게 놓여 마감 전 집계에서 빠진다
        x106 = self.unit("comment", 106, "x106")  # 파일 뒤 줄(107 첫 줄, 10/15 확정)이 상한을 준다
        self.assertEqual((x106["dt_kst"], x106["dt_conf"]), (pd.Timestamp("2026-10-15 10:30:00"), "row"))
        self.assertEqual(len(self.u[(self.u["utype"] == "comment") & (self.u["no"] == 107)]), 1)  # 중복 제거
        self.assertEqual(self.unit("comment", 107, "x107")["dt_conf"], "matched")

    def test_post_close_comment_excluded(self):
        before, after = self.unit("comment", 105, "before"), self.unit("comment", 105, "after")
        self.assertEqual(before["pwin_kr"], "K2026-10-15")
        self.assertTrue(before["pre_close_kr"])
        self.assertFalse(after["pre_close_kr"])
        self.assertEqual(before["win_kr"], "K2026-10-15")
        self.assertEqual(after["win_kr"], "K2026-10-16")  # 08:00:00 댓글은 다음 창
        pre = loader.comments_before_close(self.u, pd.Timestamp("2026-10-15 08:00"))
        self.assertIn("before", set(pre["text"]))
        self.assertNotIn("after", set(pre["text"]))
        p = self.unit("post", 105)
        self.assertEqual(p["c120_kr"], 1)  # 120분 안 댓글 2개 중 마감 전 1개
        self.assertFalse(p["rank_ok_kr"])  # 07:00 > 05:59
        self.assertEqual(self.win("K2026-10-15")["n_comments"], 3)  # c1·c2(10/14 밤) + before

    def test_number_missing_rate(self):
        w = self.win("K2026-10-15")  # 경계 번호 101 → 105: 102~105 중 102가 결번
        self.assertEqual((w["b_open"], w["b_close"], w["d_no"], w["n_saved_in_range"]), (101, 105, 4, 3))
        self.assertAlmostEqual(w["missing_rate"], 0.25)

    def test_window_index_ranges(self):
        for _, w in self.wk[self.wk["complete"] & (self.wk["variant"] == "main")].iterrows():
            seg = self.u.iloc[w["u_lo"]:w["u_hi"]]
            self.assertTrue((seg["win_kr"] == w["win_id"]).all())
            self.assertEqual(len(seg), int((self.u["win_kr"] == w["win_id"]).sum()))

    def test_us_window_dst(self):
        self.assertEqual(loader.et_to_kst(date(2026, 10, 14), 8, 30), datetime(2026, 10, 14, 21, 30))
        self.assertEqual(loader.et_to_kst(date(2026, 11, 2), 8, 30), datetime(2026, 11, 2, 22, 30))
        wu = self.b["windows_us"]
        w = wu[wu["win_id"] == "U2026-10-15"].iloc[0]
        self.assertEqual((w["open_kst"], w["close_kst"]),
                         (pd.Timestamp("2026-10-14 21:30"), pd.Timestamp("2026-10-15 21:30")))
        self.assertEqual(self.unit("post", 103)["win_u"], "U2026-10-14")
        self.assertEqual(self.unit("post", 106)["win_u"], "U2026-10-15")

    def test_salt_continuity_runs(self):
        sc = loader.salt_continuity(self.u)
        self.assertIn("overlap_prev", sc.columns)
        self.assertTrue(len(sc) >= 3)


if __name__ == "__main__":
    unittest.main()
