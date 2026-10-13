"""개미 지수 수집 contract: docs/desk_system.md §12의 파싱·속도·이어 받기·재수집 규칙을 지킨다."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from desk import ant


def view_page(no, dt, title, body="본문", comments=(), total=None):
    """뽐뿌 view.php에서 파서가 보는 부분만 남긴 쪽."""
    data = {"total_comment": len(comments) if total is None else total, "total_page": 1,
            "comments": list(comments)}
    span = f'<span id="comment">{len(comments)}</span>' if comments else ""
    return (
        f'<div id="topTitle"> <h1>{title}{span}</h1> <div class="topTitle-box"><ul>'
        f'<li class="topTitle-name"><strong><a><i class="nlevel lv5"></i>닉네임{no}</a></strong></li>'
        f"<li>등록일 {dt}</li> <li>조회수 321</li></ul></div></div>"
        f"<!--DCM_BODY--><table><tr><td class='board-contents'><p>{body}</p><p>둘째&amp;줄</p>"
        f"</td></tr></table><!--/DCM_BODY-->"
        '<span id="vote_list_btn_txt">3</span><span id="vote_anti_list_btn_txt">1</span>'
        f"<script> var initialCommentData = {json.dumps(data, ensure_ascii=False)}; </script>"
    )


def comment(text, detail="2026-10-12 10:00:00", blocked=False, replies=()):
    return {"no": 1, "depth": 0, "name": "<b><i></i>댓글러</b>", "memo": f"<p>{text}</p>",
            "vote_btn": {"vote_count": 2}, "anti_vote_btn": {"anti_vote_count": 1},
            "meta": {"time_display": "10:00:00", "time_detail": detail},
            "block_status": {"is_blocked": blocked, "is_spam": False}, "replies": list(replies)}


def list_page(newest):
    return "".join(f'<td class="baseList-space baseList-numb" colspan="2">{n}</td>'
                   for n in range(newest, newest - 3, -1))


class ParseTest(unittest.TestCase):
    def test_post_fields_comments_and_hashed_author(self):
        reply = comment("대댓글", detail="2026-10-12 11:00:00")
        page = view_page(7, "2026-10-12 09:30", "삼전 <b>가즈아</b>", body="풀매수",
                         comments=[comment("떡상"), comment("광고", blocked=True), comment("물림", replies=[reply])])
        r = ant.parse_view(page, 7, salt="s")
        self.assertEqual((r["no"], r["dt"], r["title"]), (7, "2026-10-12 09:30", "삼전 가즈아"))
        self.assertEqual(r["body"], "풀매수\n둘째&줄")
        self.assertEqual((r["views"], r["up"], r["down"], r["ncomment"]), (321, 3, 1, 3))
        self.assertEqual([c["text"] for c in r["comments"]], ["떡상", "물림", "대댓글"])  # 차단 댓글은 뺀다
        self.assertEqual(r["comments"][2]["dt"], "2026-10-12 11:00:00")
        self.assertEqual(r["a"], ant.author_key("닉네임7", "s"))
        self.assertNotIn("닉네임", json.dumps(r, ensure_ascii=False))  # 이름은 저장하지 않는다

    def test_today_comment_keeps_clock_time_and_missing_post_is_none(self):
        r = ant.parse_view(view_page(1, "2026-10-13 21:55", "제목", comments=[comment("오늘", detail="")]), 1)
        self.assertEqual(r["comments"][0]["dt"], "10:00:00")
        self.assertIsNone(ant.parse_view("<html><title>뽐뿌 - 사람이 좋아 함께하는 곳</title></html>", 2))

    def test_title_without_comment_span(self):
        r = ant.parse_view(view_page(3, "2026-10-13 22:13", "삼바 인수"), 3)
        self.assertEqual((r["title"], r["ncomment"], r["comments"]), ("삼바 인수", 0, []))

    def test_newest_number_from_list(self):
        self.assertEqual(ant.newest_no(list_page(441517)), 441517)
        with self.assertRaises(ValueError):
            ant.newest_no("<html></html>")


class FakeResponse:
    def __init__(self, code, text=""):
        self.status_code, self.content = code, text.encode("euc-kr")


class FetcherTest(unittest.TestCase):
    def make(self, codes):
        session = mock.Mock()
        session.get.side_effect = [FakeResponse(c, "본문" if c == 200 else "") for c in codes]
        slept = []
        clock = iter(range(0, 10**6, 100))  # 매번 충분히 시간이 흐른 것으로 둔다
        return ant.Fetcher(5.0, session, slept.append, lambda: next(clock)), slept

    def test_429_waits_penalty_then_succeeds(self):
        f, slept = self.make([429, 429, 200])
        self.assertEqual(f.get("u"), "본문")
        self.assertEqual(slept, [ant.PENALTY, ant.PENALTY])

    def test_persistent_error_raises_after_six_tries(self):
        f, slept = self.make([500] * 6)
        with self.assertRaises(IOError):
            f.get("u")
        self.assertEqual(slept, [10, 20, 40, 80, 160])  # 마지막 시도 뒤에는 쉬지 않고 바로 알린다

    def test_gap_between_requests(self):
        session = mock.Mock()
        session.get.return_value = FakeResponse(200, "x")
        slept, now = [], [0.0]
        f = ant.Fetcher(5.0, session, lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)), lambda: now[0])
        f.get("a")
        now[0] += 1.5
        f.get("b")
        self.assertEqual(slept, [3.5])


class FakeSite:
    """번호 → (날짜, 제목). 없는 번호는 지운 글이다."""

    def __init__(self, posts, newest):
        self.posts, self.newest, self.calls, self.fail = posts, newest, [], set()

    def get(self, url):
        if "zboard.php" in url:
            return list_page(self.newest)
        no = int(url.rsplit("no=", 1)[1])
        self.calls.append(no)
        if no in self.fail:
            raise IOError("HTTP 500")
        if no not in self.posts or no > self.newest:
            return "<html>삭제</html>"
        dt, title = self.posts[no]
        return view_page(no, dt, title)


class CollectTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        patch = mock.patch.object(ant, "ANT_DIR", self.dir)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(shutil.rmtree, self.dir, True)
        # 번호 1~100, 하루 10건씩 2024-12-28부터. 37번은 지운 글
        start = datetime(2024, 12, 28, 9, 0)
        self.posts = {n: ((start + timedelta(days=(n - 1) // 10, minutes=n)).strftime("%Y-%m-%d %H:%M"), f"글{n}")
                      for n in range(1, 101) if n != 37}
        self.now = datetime(2025, 1, 10, 12, 0)

    def collect(self, site, **kw):
        return ant.collect(since="2025-01-01", get=site.get, now=lambda: self.now, clock=lambda: 0.0,
                           log=lambda m: None, **kw)

    def test_backfill_resumes_and_stops_below_since(self):
        site = FakeSite(self.posts, 100)
        first = self.collect(site, max_posts=30)
        self.assertEqual((first["high"], first["low"], first["done"]), (100, 71, False))
        second = self.collect(site)
        self.assertTrue(second["done"])
        self.assertEqual(sorted(set(site.calls)), list(range(min(site.calls), 101)))
        self.assertLessEqual(min(site.calls), 41)  # 2025-01-01(41번) 아래로 한 묶음 안에서 멈춘다
        day = ant.load_day("2025-01-01")
        self.assertEqual(sorted(day), list(range(41, 51)))
        self.assertNotIn(37, ant.load_day("2024-12-31"))  # 지운 글은 저장하지 않는다
        self.assertEqual(second["missing"], 1)

    def test_new_posts_and_unsettled_posts_are_refetched_once_due(self):
        site = FakeSite(self.posts, 95)
        self.collect(site)
        st = ant.load_state()
        # 2025-01-06(번호 91~95)은 now(1/10 12:00) 기준 24시간이 지났고, 1/10 글만 다시 볼 대상이다
        self.posts.update({n: ("2025-01-10 11:00", f"새글{n}") for n in range(96, 101)})
        site.newest, site.calls = 100, []
        out = self.collect(site)
        self.assertEqual(sorted(site.calls), list(range(96, 101)))
        self.assertEqual(out["recheck"], 5)
        self.assertEqual(st["high"], 95)
        site.calls = []
        self.collect(site)
        self.assertEqual(site.calls, [])  # 아직 24시간이 안 지났다
        self.now += timedelta(hours=25)
        site.calls = []
        out = self.collect(site)
        self.assertEqual(sorted(site.calls), list(range(96, 101)))
        self.assertEqual(out["recheck"], 0)
        self.assertEqual(len(ant.load_day("2025-01-10")), 5)  # 다시 받은 줄은 뒤 줄이 이긴다

    def test_three_failures_in_a_row_stop_and_keep_progress(self):
        site = FakeSite(self.posts, 100)
        site.fail = {90, 89, 88}
        with self.assertRaises(ant.Blocked):
            self.collect(site)
        st = ant.load_state()
        self.assertEqual(st["low"], 101)  # 그 묶음은 low를 못 내렸으니 다음 실행이 다시 받는다
        self.assertEqual(sorted(int(k) for k in st["recheck"]), [88, 89, 90])
        self.assertEqual(len(ant.load_day("2025-01-06")), 10)  # 받은 글은 저장됐다
        site.fail = set()
        self.assertTrue(self.collect(site)["done"])

    def test_long_run_of_empty_numbers_stops(self):
        site = FakeSite(self.posts, 100)
        site.posts = {}  # 쪽 구조가 바뀌어 파서가 모두 None을 돌려주는 경우
        with mock.patch.object(ant, "EMPTY_STOP", 15), self.assertRaises(ant.Blocked):
            self.collect(site)
        self.assertEqual(len(site.calls), 15)


if __name__ == "__main__":
    unittest.main()
