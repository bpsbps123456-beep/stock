"""Regression tests for ETF classification without scanner-list pollution."""

from __future__ import annotations

import unittest

import watchlist


def flattened(sector_map: dict[str, list[str]]) -> set[str]:
    return {ticker for tickers in sector_map.values() for ticker in tickers}


class WatchlistContractTests(unittest.TestCase):
    def test_etf_sectors_are_mapped_without_polluting_profit_or_loss_sets(self) -> None:
        profit_expected = flattened(watchlist.PROFIT_SECTORS)
        loss_expected = flattened(watchlist.LOSS_SECTORS)
        etf_expected = flattened(watchlist.ETF_SECTORS)
        all_classified = profit_expected | loss_expected | etf_expected

        self.assertEqual(set(watchlist.TICKER_TO_SECTOR), all_classified)
        for sector, tickers in watchlist.ETF_SECTORS.items():
            for ticker in tickers:
                self.assertEqual(watchlist.TICKER_TO_SECTOR[ticker], sector)

        self.assertEqual(set(watchlist.PROFIT_TICKERS), profit_expected)
        self.assertEqual(set(watchlist.LOSS_TICKERS), loss_expected)

        etf_only = etf_expected - profit_expected - loss_expected
        self.assertTrue(etf_only)
        self.assertTrue(etf_only.isdisjoint(watchlist.PROFIT_TICKERS))
        self.assertTrue(etf_only.isdisjoint(watchlist.LOSS_TICKERS))


if __name__ == "__main__":
    unittest.main()
