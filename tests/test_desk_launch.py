"""Double-click launch contract (야간데스크.cmd → py -m desk open): the page must open while the
prices are still loading, a second double-click must reuse the running desk, and a desk that
never answers must end in a message instead of a silent console window."""

from __future__ import annotations

import threading
import unittest
from unittest import mock

from desk import server


class ServeOpensThePageFirstTest(unittest.TestCase):
    def test_page_opens_before_the_prices_are_ready(self):
        gate, seen = threading.Event(), []
        ensure_done = threading.Event()

        def slow_ensure(*_a, **_k):  # a first launch after the US close: prices still downloading
            gate.wait(5)
            ensure_done.set()

        def opened(url):
            seen.append(("open", url, ensure_done.is_set()))

        with mock.patch.object(server.DESK, "ensure", side_effect=slow_ensure), \
                mock.patch.object(server, "running_desk", return_value=None), \
                mock.patch.object(server.webbrowser, "open", side_effect=opened), \
                mock.patch.object(server.DeskServer, "serve_forever", side_effect=lambda: seen.append(("serve",))):
            self.assertEqual(server.main(["--port=0"]), 0)
        gate.set()
        for t in threading.enumerate():  # let the warm-up finish before the next test patches time
            if t.name == "desk-warm":
                t.join(2)
        self.assertEqual(seen[0][0], "open")
        self.assertTrue(seen[0][1].startswith("http://127.0.0.1:"))
        self.assertFalse(seen[0][2], "the browser waited for the price download")
        self.assertEqual(seen[1], ("serve",))


class LaunchTest(unittest.TestCase):
    def test_a_running_desk_is_reused_without_a_second_server(self):
        with mock.patch.object(server, "running_desk", return_value="http://127.0.0.1:8321/"), \
                mock.patch.object(server.subprocess, "Popen") as popen, \
                mock.patch.object(server.webbrowser, "open") as browse:
            self.assertEqual(server.launch([]), 0)
        popen.assert_not_called()
        browse.assert_called_once_with("http://127.0.0.1:8321/")

    def test_starts_the_server_in_its_own_window_then_opens_the_page(self):
        answers = iter([None, None, "http://127.0.0.1:8321/"])
        with mock.patch.object(server, "running_desk", side_effect=lambda ports=None: next(answers)), \
                mock.patch.object(server.subprocess, "Popen") as popen, \
                mock.patch.object(server.time, "sleep"), \
                mock.patch.object(server.webbrowser, "open") as browse:
            popen.return_value.poll.return_value = None
            self.assertEqual(server.launch([]), 0)
        args = popen.call_args.args[0]
        self.assertEqual(args[1:6], ["-m", "desk", "serve", "--launcher", "--no-browser"])
        browse.assert_called_once_with("http://127.0.0.1:8321/")

    def test_a_desk_that_never_answers_ends_in_a_message(self):
        clock = iter([0.0, 0.0] + [server.LAUNCH_WAIT_SEC + 1.0] * 5)
        with mock.patch.object(server, "running_desk", return_value=None), \
                mock.patch.object(server.subprocess, "Popen") as popen, \
                mock.patch.object(server.time, "time", side_effect=lambda: next(clock)), \
                mock.patch.object(server.time, "sleep"), \
                mock.patch.object(server.webbrowser, "open") as browse:
            popen.return_value.poll.return_value = None
            self.assertEqual(server.launch([]), 1)
        browse.assert_not_called()

    def test_a_server_that_exits_at_once_ends_in_a_message(self):
        with mock.patch.object(server, "running_desk", return_value=None), \
                mock.patch.object(server.subprocess, "Popen") as popen, \
                mock.patch.object(server.time, "sleep"), \
                mock.patch.object(server.webbrowser, "open") as browse:
            popen.return_value.poll.return_value = 1
            self.assertEqual(server.launch([]), 1)
        browse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
