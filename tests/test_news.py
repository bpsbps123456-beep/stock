"""Availability, deduplication, immutable recording and bounded outcome tests."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from desk import news, server


def example():
    event = dict(id="first", episode="test", source_group="one-report", title="매도 보도", source="원문",
                 url="https://example.com/report", published_date="2026-07-20",
                 available_at="2026-07-20T13:16:00+00:00", recorded_at="2026-10-25T07:00:00+00:00",
                 time_precision="exact", kind="pressure", stance="against", quality="article",
                 time_note="당시 기사, 후일 수집", scope="기술주", measurement="순매도", window="8주",
                 summary="기록적인 매도", interpretation="매도 소진과 다름", next_check="매도 감소",
                 bank=None, family=None)
    return {"version": 1, "episodes": [{"id": "test", "label": "시험 국면", "benchmark": "SOXX"}],
            "events": [event], "assessments": []}


class NewsPointInTimeTests(unittest.TestCase):
    def test_public_replay_and_actual_recording_are_different(self):
        data = example()
        before = news.stamp("2026-07-20T13:15:59Z")
        after = news.stamp("2026-07-20T13:16:00Z")
        self.assertEqual(news.visible_events(data, before, "public"), [])
        self.assertEqual(len(news.visible_events(data, after, "public")), 1)
        self.assertEqual(news.visible_events(data, after, "recorded"), [])

    def test_unknown_publication_time_is_not_backdated(self):
        data = example()
        data["events"][0]["time_precision"] = "unknown"
        with self.assertRaises(news.NewsError):
            news.validate(data)
        data["events"][0]["available_at"] = data["events"][0]["recorded_at"]
        news.validate(data)
        self.assertIsNone(news.reaction(data["events"][0], [])["anchor"])

    def test_repeated_media_do_not_add_independent_sources(self):
        data = example()
        duplicate = {**data["events"][0], "id": "repeat", "available_at": "2026-07-21T14:00:00Z"}
        data["events"].append(duplicate)
        ep = news.payload("2026-07-22T21:00:00Z", data=data, prices={})["episodes"][0]
        self.assertEqual(ep["groups"], 1)
        self.assertEqual(ep["counts"]["pressure"], 1)
        self.assertTrue(ep["events"][1]["duplicate_source"])
        self.assertIsNone(ep["assessment"])

    def test_correction_does_not_erase_original_in_earlier_replay(self):
        data = example()
        data["events"][0]["recorded_at"] = "2026-07-20T14:00:00Z"
        data["events"].append({**data["events"][0], "id": "corrected", "supersedes": "first",
                               "available_at": "2026-07-22T14:00:00Z", "recorded_at": "2026-07-22T15:00:00Z"})
        news.validate(data)
        self.assertEqual([e["id"] for e in news.visible_events(data, news.stamp("2026-07-21T21:00:00Z"), "public")], ["first"])
        self.assertEqual([e["id"] for e in news.visible_events(data, news.stamp("2026-07-23T21:00:00Z"), "public")], ["corrected"])

    def test_price_close_not_visible_before_session_end(self):
        prices = {"2026-07-20": 100, "2026-07-21": 102, "2026-07-22": 99}
        rows = news.completed_prices(prices, news.stamp("2026-07-21T19:59:59Z"))
        self.assertEqual([r[0] for r in rows], ["2026-07-20"])
        self.assertIsNone(news.reaction(example()["events"][0], rows)["returns"]["1"])

    def test_after_close_article_anchors_next_session_and_limits_outcomes(self):
        row = {**example()["events"][0], "available_at": "2026-07-20T20:30:00Z"}
        prices = dict(zip(["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24", "2026-07-27", "2026-07-28"],
                          [50, 100, 90, 105, 110, 95, 120]))
        result = news.reaction(row, news.completed_prices(prices, news.stamp("2026-07-28T21:00:00Z")))
        self.assertEqual(result["anchor"], "2026-07-21")
        self.assertAlmostEqual(result["returns"]["1"], -.1)
        self.assertAlmostEqual(result["returns"]["3"], .1)
        self.assertAlmostEqual(result["returns"]["5"], .2)
        self.assertAlmostEqual(result["worst_close_5"], -.1)
        partial = news.reaction(row, news.completed_prices(prices, news.stamp("2026-07-24T21:00:00Z")))
        self.assertIsNone(partial["returns"]["5"])
        self.assertIsNone(partial["worst_close_5"])

    def test_price_download_failure_preserves_cached_observations(self):
        prices = {"series": {"SOXX": {"2026-07-20": 100}}, "fetched_at": "2026-07-20T21:00:00Z"}
        with mock.patch.object(news, "refresh_prices", side_effect=OSError("offline")):
            result = news.payload("2026-07-21T21:00:00Z", force=True, data=example(), prices=prices)
        self.assertIn("offline", result["warning"])
        self.assertEqual(result["episodes"][0]["quote"]["close"], 100)
        self.assertEqual(result["prices_fetched_at"], prices["fetched_at"])

    def test_early_close_holiday_and_missing_price_are_handled(self):
        # 2026 Thanksgiving: closed Thursday, Friday ends at 13:00 ET.
        prices = {"2026-11-26": 999, "2026-11-27": 100, "2026-11-30": 102, "2026-12-02": 104}
        rows = news.completed_prices(prices, news.stamp("2026-11-27T18:01:00Z"))
        self.assertEqual([r[0] for r in rows], ["2026-11-27"])
        event = {**example()["events"][0], "available_at": "2026-11-27T18:30:00Z"}
        result = news.reaction(event, news.completed_prices(prices, news.stamp("2026-12-02T22:00:00Z")))
        self.assertEqual(result["anchor"], "2026-11-30")
        self.assertIsNone(result["returns"]["1"], "missing Tuesday must not shift D+1 to Wednesday")

    def test_missing_anchor_is_not_replaced_with_later_close(self):
        rows = news.completed_prices({"2026-07-21": 102}, news.stamp("2026-07-22T22:00:00Z"))
        self.assertIsNone(news.reaction(example()["events"][0], rows)["anchor"])


class NewsRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.json"

    def test_import_stamps_actual_time_and_is_idempotent(self):
        data = example()
        at = "2026-10-25T07:30:00Z"
        with mock.patch.object(news, "now_iso", return_value=at):
            self.assertEqual(news.import_records(data, self.path), 2)
            self.assertEqual(news.import_records(data, self.path), 0)
        self.assertEqual(news.read_ledger(self.path)["events"][0]["recorded_at"], at)

    def test_conflict_is_atomic_and_keeps_original(self):
        data = example()
        news.import_records(data, self.path)
        before = self.path.read_bytes()
        data["events"][0]["summary"] = "기존 판단을 바꾼다"
        with self.assertRaises(news.NewsError):
            news.import_records(data, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.path.with_suffix(".lock").exists())

    def test_invalid_input_never_creates_ledger(self):
        data = example()
        data["events"][0]["url"] = "javascript:alert(1)"
        with self.assertRaises(news.NewsError):
            news.import_records(data, self.path)
        self.assertFalse(self.path.exists())

    def test_concurrent_writer_is_refused(self):
        self.path.with_suffix(".lock").touch()
        with self.assertRaises(news.NewsError):
            news.import_records(example(), self.path)

    def test_assessment_cannot_reference_unrecorded_evidence(self):
        data = example()
        data["assessments"] = [dict(id="a", episode="test", recorded_at="2026-07-21T12:00:00Z",
                                    conclusion="반등", supports=["first"], contradicts=[], next_checks=["확인"])]
        with self.assertRaises(news.NewsError):
            news.validate(data)

    def test_seed_records_are_consistent_and_two_episodes(self):
        data = news.read_ledger()
        self.assertEqual({e["benchmark"] for e in data["episodes"]}, {"SOXX"})
        self.assertGreaterEqual(len(data["events"]), 6)
        self.assertEqual(news.visible_events(data, news.stamp("2026-07-31T21:00:00Z"), "recorded"), [])


def note(event_id, family, available, bank="GS"):
    row = example()["events"][0]
    row.update(id=event_id, source_group=event_id, family=family, bank=bank, available_at=available,
               published_date=available[:10])
    return row


class NewsFamilyTests(unittest.TestCase):
    """2026-10-28 사용자 결정: 은행·계열 칸, D의 앞 30거래일 T 유무, R 전방 채점(§13.1)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.json"

    def test_new_records_need_bank_and_family(self):
        data = example()
        del data["events"][0]["bank"]
        with mock.patch.object(news, "now_iso", return_value="2026-10-28T12:00:00Z"):
            with self.assertRaises(news.NewsError):
                news.import_records(data, self.path)
        bad = example()
        bad["events"][0]["family"] = "X"
        with self.assertRaises(news.NewsError):
            news.validate(bad)

    def test_prior_t_is_fixed_when_d_is_recorded(self):
        base = {"episodes": example()["episodes"]}
        with mock.patch.object(news, "now_iso", return_value="2026-10-28T12:00:00Z"):
            news.import_records({**base, "events": [note("t1", "T", "2026-10-01T13:00:00+00:00")]}, self.path)
            news.import_records({**base, "events": [note("d1", "D", "2026-10-20T13:00:00+00:00")]}, self.path)
        with mock.patch.object(news, "now_iso", return_value="2026-12-02T12:00:00Z"):
            news.import_records({**base, "events": [note("d2", "D", "2026-12-01T13:00:00+00:00"),
                                                    note("t2", "T", "2026-10-19T13:00:00+00:00")]}, self.path)
        rows = {e["id"]: e for e in news.read_ledger(self.path)["events"]}
        self.assertEqual((rows["d1"]["prior_t"]["found"], rows["d1"]["prior_t"]["ids"]), (True, ["t1"]))
        self.assertFalse(rows["d2"]["prior_t"]["found"])          # 10/1은 12/1 앞 30거래일 밖, t2는 d2보다 늦게 들어옴
        self.assertEqual(rows["d1"]["prior_t"]["ids"], ["t1"])    # 나중에 넣은 T(t2)가 d1을 바꾸지 않음

    def test_forward_r_scores_only_promptly_recorded_r(self):
        cal = news.market_calendar()
        days = [s.strftime("%Y-%m-%d") for s in cal.sessions_window("2026-03-02", 100)]
        values = [100 + i * 10 / 59 for i in range(60)] + [108, 105, 102, 99, 97, 95] + \
                 [95 + i * 15 / 33 for i in range(1, 35)]
        series = dict(zip(days, values))
        prompt = note("r1", "R", days[68] + "T14:00:00+00:00")
        prompt["recorded_at"] = days[68] + "T15:00:00+00:00"
        late = note("r2", "R", days[68] + "T14:00:00+00:00")
        late["recorded_at"] = days[75] + "T15:00:00+00:00"
        result = news.forward_r([prompt, late], series, news.stamp(days[-1] + "T23:00:00+00:00"))
        self.assertEqual([i["id"] for i in result["items"]], ["r1"])
        item = result["items"][0]
        self.assertEqual(item["anchor"], days[68])
        self.assertAlmostEqual(item["returns"]["10"], values[78] / values[68] - 1)
        self.assertEqual((item["in_dd"], item["hold20"]), (True, 1))
        self.assertEqual((result["gs"]["n"], result["all"]["held"]), (1, 1))


class NewsHttpTests(unittest.TestCase):
    def test_endpoint_filters_dates_and_rejects_naive_timestamp(self):
        httpd = server.DeskServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{httpd.server_port}/api/news"
            with urllib.request.urlopen(base + "?as_of=2026-07-20T13:15:59Z") as response:
                data = json.load(response)
            self.assertFalse(any(e["events"] for e in data["episodes"]))
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + "?as_of=2026-07-20")
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(3)


if __name__ == "__main__":
    unittest.main()
