"""Paper reads must stay read-only; refresh is explicit and separated from account routes."""

import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from desk import server


class PaperApiTest(unittest.TestCase):
    def setUp(self):
        self.http = server.DeskServer(("127.0.0.1", 0), server.Handler)
        self.worker = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.worker.start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.worker.join(2)

    def request(self, path, data=None):
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read())

    def test_get_including_refresh_query_never_updates_or_changes_account(self):
        with mock.patch.object(server.sox_paper, "payload", return_value={"initialized": False}) as payload, \
                mock.patch.object(server.DESK, "state") as account:
            self.assertEqual(self.request("/api/paper?refresh=1"), {"initialized": False})
        payload.assert_called_once_with()
        account.assert_not_called()

    def test_post_only_refreshes_paper(self):
        with mock.patch.object(server.sox_paper, "payload", return_value={"valid_sessions": 0}) as payload, \
                mock.patch.object(server.DESK, "ensure") as account:
            self.assertEqual(self.request("/api/paper/update", b"{}"), {"valid_sessions": 0})
        payload.assert_called_once_with(force=True)
        account.assert_not_called()

    def test_bad_json_does_not_refresh(self):
        with mock.patch.object(server.sox_paper, "payload") as payload:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/paper/update", b"[]")
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()
        payload.assert_not_called()

    def test_frozen_rule_error_is_shown_as_user_error(self):
        with mock.patch.object(server.sox_paper, "payload", side_effect=server.sox_paper.PaperError("동결 확인 실패")):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/paper/update", b"{}")
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.loads(caught.exception.read())["error"], "동결 확인 실패")
        caught.exception.close()


class PaperWatchTest(unittest.TestCase):
    def test_unregistered_desk_does_not_start_a_paper_experiment(self):
        stop = mock.Mock()
        stop.wait.side_effect = [False, True]
        with mock.patch.object(server.sox_paper, "payload", return_value={"initialized": False}) as payload:
            server._paper_watch(stop)
        payload.assert_called_once_with()

    def test_registered_desk_updates_without_waiting_for_user_to_open_tab(self):
        stop = mock.Mock()
        stop.wait.side_effect = [False, True]
        with mock.patch.object(server.sox_paper, "payload", return_value={"initialized": True}) as payload:
            server._paper_watch(stop)
        self.assertEqual(payload.call_args_list, [mock.call(), mock.call(force=True)])


if __name__ == "__main__":
    unittest.main()
