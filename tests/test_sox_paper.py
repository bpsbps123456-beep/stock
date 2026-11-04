"""Forward-only ledger contracts, with synthetic SIP bars and no network."""
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from desk import config, sox_paper as p


def moment(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def bars(session="2026-11-05"):
    start = pd.Timestamp(session + " 09:30", tz=config.NY)
    idx = pd.date_range(start, p.cutoff(session), freq="min", inclusive="left")
    out = {}
    for sym in ("SOXL", "SOXS"):
        out[sym] = pd.DataFrame({"Open": 100., "High": 100.1, "Low": 99.9,
                                 "Close": 100., "Volume": 1000.}, index=idx)
    return out


class PaperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = mock.patch.object(p, "STORE", Path(self.tmp.name))
        self.store.start()
        self.now = moment("2026-11-05T00:05:00+09:00")
        self.manifest = p.initialize(self.now)

    def tearDown(self):
        self.store.stop()
        self.tmp.cleanup()

    def test_unobserved_start_and_manifest_is_immutable(self):
        self.assertEqual(self.manifest["start_session"], "2026-11-05")
        before = (p.STORE / "manifest.json").read_bytes()
        p.initialize(moment("2026-11-10T00:00:00+09:00"))
        self.assertEqual(before, (p.STORE / "manifest.json").read_bytes())
        self.assertEqual(p._next_open_session(moment("2026-11-05T10:00:00-05:00")), "2026-11-06")

    def test_dst_cutoff_and_delay(self):
        self.assertEqual(p.cutoff("2026-11-05").strftime("%H:%M"), "10:00")
        self.assertEqual(p.cutoff("2027-03-15").strftime("%H:%M"), "11:00")
        early = moment("2026-11-05T10:14:59-05:00")
        self.assertEqual(p._eligible_sessions(self.manifest, early), [])
        ready = moment("2026-11-05T10:15:00-05:00")
        self.assertEqual(p._eligible_sessions(self.manifest, ready), ["2026-11-05"])
        with mock.patch.object(p, "fetch_bars") as fetch:
            for day in ("2026-11-04", "2026-11-06", "2026-11-07"):
                with self.assertRaises(p.PaperError):
                    p.update(day, now=ready)
            fetch.assert_not_called()

    def test_no_signal_is_valid_zero_missing_is_not_zero(self):
        record = p.replay("2026-11-05", bars(), self.manifest)
        self.assertEqual(record["status"], "valid")
        for rule in record["rules"].values():
            self.assertEqual(rule["return_pct"], 0)
            self.assertEqual(rule["trades"], [])
        bad = bars()
        bad["SOXS"] = bad["SOXS"].drop(bad["SOXS"].index[5])
        with self.assertRaisesRegex(p.PaperError, "누락"):
            p.replay("2026-11-05", bad, self.manifest)
        with mock.patch.object(p, "fetch_bars", return_value=bad):
            result = p.update(now=moment("2026-11-05T10:16:00-05:00"))
        self.assertEqual(result["valid_sessions"], 0)
        self.assertEqual(result["missing_sessions"], 1)
        self.assertIsNone(result["rules"][0]["mean_daily_pct"])
        self.assertFalse((p.STORE / "days/2026-11-05.json").exists())

    def test_missing_retry_then_valid_record_never_overwritten(self):
        ready = moment("2026-11-05T10:16:00-05:00")
        with mock.patch.object(p, "fetch_bars", side_effect=p.PaperError("자료 없음")):
            p.update(now=ready)
        with mock.patch.object(p, "fetch_bars", return_value=bars()) as fetch:
            result = p.update(now=ready)
            self.assertEqual(result["valid_sessions"], 1)
            self.assertEqual(result["missing_sessions"], 0)
            path = p.STORE / "days/2026-11-05.json"
            before = path.read_bytes()
            p.update(now=ready)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(fetch.call_count, 1)

    def test_soxl_trigger_delay_and_target_after_entry_bar(self):
        frames = bars()
        l = frames["SOXL"]
        # Signal at 09:32 close: first low 99.9, then 103.1 > 1.03*99.9.
        l.loc[l.index[2]:, ["Open", "Close"]] = 103.1
        l.loc[l.index[2]:, "High"] = 103.2
        l.loc[l.index[2]:, "Low"] = 103.
        s = frames["SOXS"]
        # SOXS target cannot be touched in its entry bar 09:33.
        s.loc[s.index[3], "High"] = 102.
        s.loc[s.index[4], "High"] = 102.
        record = p.replay("2026-11-05", frames, self.manifest)
        tr = record["rules"]["S2"]["trades"]
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr[0]["entry_bar_et"], "09:33")
        self.assertEqual(tr[0]["entry_at_et"], "2026-11-05T09:34:00-05:00")
        self.assertEqual(tr[0]["exit_bar_et"], "09:34")
        self.assertEqual(tr[0]["exit_time_basis"], "within_bar")
        self.assertEqual(tr[0]["why"], "target")
        self.assertAlmostEqual(tr[0]["return_pct"], 1.)

    def test_s2_resets_low_after_target_and_waits_for_new_three_percent_rise(self):
        frames = bars()
        l = frames["SOXL"]
        l.loc[l.index[2]:, ["Open", "Close"]] = 103.1
        l.loc[l.index[2]:, "High"] = 103.2
        l.loc[l.index[2]:, "Low"] = 103.
        l.loc[l.index[7]:, ["Open", "Close"]] = 106.2
        l.loc[l.index[7]:, "High"] = 106.3
        l.loc[l.index[7]:, "Low"] = 106.
        s = frames["SOXS"]
        s.loc[s.index[[4, 9]], "High"] = 102.
        trades = p.replay("2026-11-05", frames, self.manifest)["rules"]["S2"]["trades"]
        self.assertEqual([tr["entry_bar_et"] for tr in trades], ["09:33", "09:38"])
        self.assertEqual([tr["why"] for tr in trades], ["target", "target"])

    def test_s1_both_directions_use_decision_to_range_distance_and_only_one_entry(self):
        for sign, symbol in ((1, "SOXL"), (-1, "SOXS")):
            with self.subTest(sign=sign):
                frames = bars()
                l = frames["SOXL"]
                l.iloc[0] = [100, 100.2, 99.8, 100 + .1 * sign, 1000]
                for j in range(1, len(l)):
                    close = 100 + .4 * sign
                    l.iloc[j] = [close, close + .1, close - .1, close, 1000]
                bought = frames[symbol]
                bought.loc[bought.index[3], "High"] = 103
                trades = p.replay("2026-11-05", frames, self.manifest)["rules"]["S1"]["trades"]
                self.assertEqual(len(trades), 1)
                trade = trades[0]
                self.assertEqual(trade["symbol"], symbol)
                self.assertEqual(trade["entry_bar_et"], "09:32")
                self.assertEqual(trade["why"], "target")
                decision_close = 100 + .4 * sign
                r = .6 / decision_close
                self.assertAlmostEqual(trade["return_pct"], 2 * r * 100)

    def test_source_and_ledger_tamper_stop_update(self):
        changed = {**p._source_hashes(), "code/lib.py": "changed"}
        with mock.patch.object(p, "_source_hashes", return_value=changed):
            with self.assertRaisesRegex(p.PaperError, "바뀌었습니다"):
                p.payload()
        with mock.patch.object(p, "fetch_bars", return_value=bars()):
            p.update(now=moment("2026-11-05T10:16:00-05:00"))
        path = p.STORE / "days/2026-11-05.json"
        record = p._read(path)
        record["rules"]["S1"]["return_pct"] = 99
        p._atomic(path, record)
        with self.assertRaisesRegex(p.PaperError, "해시"):
            p.payload()

    def test_research_imports_leave_global_environment_and_paths_unchanged(self):
        path = list(sys.path)
        env = dict(os.environ)
        with mock.patch.object(p, "_MODULES", None):
            modules = p._research_modules()
        self.assertEqual(sys.path, path)
        self.assertEqual(dict(os.environ), env)
        self.assertEqual(modules[1].Order, modules[0].Order)
        self.assertEqual(modules[2].Order, modules[0].Order)

    def test_source_digest_normalizes_bom_and_checkout_newlines(self):
        a, b = p.STORE / "lf.py", p.STORE / "crlf.py"
        a.write_bytes(b"value = 1\nsecond = 2\n")
        b.write_bytes(b"\xef\xbb\xbfvalue = 1\r\nsecond = 2\r\n")
        self.assertEqual(p._file_digest(a), p._file_digest(b))

    def test_active_process_lock_and_recovery_after_termination(self):
        script = ("from desk import sox_paper as p; import sys; "
                  "cm=p._write_lock(); cm.__enter__(); print('locked',flush=True); sys.stdin.read()")
        process = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env={**os.environ, "SOX_PAPER_DIR": str(p.STORE)})
        try:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            with self.assertRaises(p.PaperError):
                with p._write_lock():
                    self.fail("active process lock must remain exclusive")
            process.terminate()
            process.wait(timeout=15)
            with p._write_lock():
                self.assertTrue((p.STORE / ".write.lock").exists())
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=15)

    def test_zero_volume_and_duplicate_bars_are_rejected(self):
        frames = bars()
        frames["SOXS"].loc[frames["SOXS"].index[3], "Volume"] = 0
        with self.assertRaises(p.PaperError):
            p.replay("2026-11-05", frames, self.manifest)
        frames = bars()
        frames["SOXS"] = pd.concat([frames["SOXS"], frames["SOXS"].iloc[:1]])
        with self.assertRaises(p.PaperError):
            p.replay("2026-11-05", frames, self.manifest)


if __name__ == "__main__":
    unittest.main()
