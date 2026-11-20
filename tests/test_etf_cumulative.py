"""Network-free end-to-end test of the cumulative briefing (prices come from its cache file)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

from gbrc.etf import cumulative
from gbrc.etf.snapshot import normalize_snapshot
from gbrc.etf.storage import save_normalized_snapshot

SESSIONS = [d.isoformat() for d in (date(2026, 9, 1) + timedelta(days=i) for i in range(40))
            if d.weekday() < 5][:26]
START, END = SESSIONS[-6], SESSIONS[-1]
MISSING = SESSIONS[-4]  # the D-3 session had no briefing → D-3 falls back one snapshot
NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def raw_snapshot(fund_id: str, day: str, qty: dict[str, float]) -> dict:
    holdings = [{"code": code, "name": code, "qty": q, "value_krw": q * 1000.0, "weight": 0.0}
                for code, q in qty.items()]
    holdings.append({"code": "", "name": "Cash", "qty": 500.0, "value_krw": 500.0, "weight": 0.0})
    return {"date": day, "actual_date": day, "etf_ticker": fund_id, "etf_name": f"Fund {fund_id}",
            "holdings": holdings, "fetched_at": f"{day}T18:00:00+09:00"}


def seed(root: Path, out_dir: Path) -> None:
    for day in SESSIONS:
        if day == MISSING:
            continue
        base = {f"{name} US EQUITY": 100.0 for name in NAMES}
        time_qty = {**base, "009150": 100.0}
        koact_qty = dict(base)
        if day == END:
            time_qty["AAA US EQUITY"] *= 1.3  # TIME 선별 매수 +30%
        for fund_id, qty in (("456600", time_qty), ("2ETFQ1", koact_qty)):
            save_normalized_snapshot(normalize_snapshot(raw_snapshot(fund_id, day, qty)), root)
    series = {d: 100.0 + i for i, d in enumerate(SESSIONS)}
    close = {ticker: dict(series) for ticker in
             ["456600.KS", "0015B0.KS", "009150.KS", *NAMES]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{END}_cumulative_prices.json").write_text(
        json.dumps({"provider": "fixture", "auto_adjust": False,
                    "retrieved_at": f"{END}T18:00:00+09:00", "close": close}),
        encoding="utf-8")


class CumulativeBriefingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.root, self.out = base / "normalized", base / "briefings"
        seed(self.root, self.out)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_report_has_every_section_row_and_real_trade(self) -> None:
        path, summary, report = cumulative.build(START, END, self.out, self.root)
        self.assertEqual(path, self.out / f"{END}_cumulative.md")
        for heading in ("## Fund 456600 (456600)", "## Fund 2ETFQ1 (2ETFQ1)"):
            self.assertIn(heading, report)
        self.assertEqual(summary["funds"]["time"]["holdings_count"], len(NAMES) + 2)
        self.assertEqual(summary["validation"]["time"]["all_holdings_rows"], len(NAMES) + 2)
        self.assertEqual(summary["funds"]["time"]["real_trades"]["누적"]["buy"], ["AAA +30%"])
        self.assertEqual(summary["funds"]["koact"]["real_trades"]["누적"]["buy"], [])
        self.assertTrue((self.out / f"{END}_cumulative.json").is_file())

    def test_missing_briefing_day_is_reported_and_d3_falls_back(self) -> None:
        _, summary, report = cumulative.build(START, END, self.out, self.root)
        self.assertEqual(summary["missing_snapshot_dates"], [MISSING])
        d1 = summary["horizons"][1]
        self.assertEqual(d1["price_base"], SESSIONS[-2])
        self.assertEqual((d1["snapshot_base"], d1["snapshot_fallback"]), (SESSIONS[-2], False))
        d3 = summary["horizons"][3]
        self.assertEqual((d3["price_base"], d3["snapshot_base"], d3["snapshot_fallback"]),
                         (MISSING, SESSIONS[-5], True))
        self.assertIn(f"미수집 {MISSING}", report)

    def test_etf_price_returns_and_default_start(self) -> None:
        payload = cumulative.run(end=END, out_dir=self.out, root=self.root)
        self.assertEqual(payload["start"], SESSIONS[-2])  # 직전 공통 스냅샷
        d1 = payload["performance"][0]
        expected = ((100.0 + 25) / (100.0 + 24) - 1) * 100
        self.assertAlmostEqual(d1["time_return"], expected)
        self.assertEqual(d1["winner"], "동률")



class PriceCacheTests(unittest.TestCase):
    """장중에 받은 잠정 종가는 캐시를 다시 쓰지 않고 확정 뒤 새로 받는다(2026-11-21 버그 수정)."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.root, self.out = base / "normalized", base / "briefings"
        seed(self.root, self.out)
        self.cache = self.out / f"{END}_cumulative_prices.json"
        self.corpus, _, _ = cumulative.load_corpus(END, self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def fake_download(self, value: float):
        close = {t: {END: value} for t in ["456600.KS", "0015B0.KS", "009150.KS", *NAMES]}
        return mock.patch.object(cumulative.price_history, "download_closes", return_value=close)

    def test_confirmed_cache_is_reused_without_download(self) -> None:
        with self.fake_download(1.0) as download:
            prices = cumulative.load_prices(self.corpus, END, self.cache)
        download.assert_not_called()
        self.assertEqual(prices["456600.KS"][END], 125.0)

    def test_intraday_download_is_marked_provisional_and_refetched_after_close(self) -> None:
        self.cache.unlink()
        intraday = datetime.fromisoformat(f"{END}T11:00:00+09:00")
        with self.fake_download(1.0):
            cumulative.load_prices(self.corpus, END, self.cache, now=intraday)
        self.assertTrue(json.loads(self.cache.read_text(encoding="utf-8"))["provisional"])

        evening = datetime.fromisoformat(f"{END}T18:00:00+09:00")
        with self.fake_download(2.0) as download:
            prices = cumulative.load_prices(self.corpus, END, self.cache, now=evening)
        download.assert_called_once()
        self.assertEqual(prices["456600.KS"][END], 2.0)
        saved = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertFalse(saved["provisional"])

        with self.fake_download(3.0) as download:  # 확정 캐시는 그대로 쓴다
            self.assertEqual(cumulative.load_prices(self.corpus, END, self.cache)["456600.KS"][END], 2.0)
        download.assert_not_called()

    def test_legacy_cache_fetched_before_settlement_is_refetched(self) -> None:
        payload = json.loads(self.cache.read_text(encoding="utf-8"))
        payload["retrieved_at"] = f"{END}T14:05:00+09:00"  # 예전 형식(표시 없음), 장중에 받음
        self.cache.write_text(json.dumps(payload), encoding="utf-8")
        with self.fake_download(2.0) as download:
            cumulative.load_prices(self.corpus, END, self.cache,
                                   now=datetime.fromisoformat(f"{END}T20:00:00+09:00"))
        download.assert_called_once()

    def test_failed_download_stops_without_writing_cache(self) -> None:
        self.cache.unlink()
        with mock.patch.object(cumulative.price_history, "download_closes", return_value={}):
            with self.assertRaises(SystemExit):
                cumulative.load_prices(self.corpus, END, self.cache)
        self.assertFalse(self.cache.exists())

    def test_intraday_warning_uses_korea_time(self) -> None:
        self.assertIsNotNone(cumulative.intraday_price_warning(
            END, now=datetime.fromisoformat(f"{END}T02:00:00+00:00")))  # 11:00 KST
        self.assertIsNone(cumulative.intraday_price_warning(
            END, now=datetime.fromisoformat(f"{END}T07:00:00+00:00")))  # 16:00 KST


if __name__ == "__main__":
    unittest.main()
