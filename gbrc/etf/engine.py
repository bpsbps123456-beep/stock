"""The briefing engine: collection, snapshot lookup, and report rendering over one store.

`workflow.py` and the night desk (`desk/digest.py`) take this object instead of reaching
into module state, and tests swap in a fake with the same methods. Every method names
its fund explicitly; nothing is "active".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gbrc.config import FundConfig

from . import digest, performance, prices, providers, report, storage
from .storage import NORMALIZED_HOLDINGS_DIR


@dataclass(frozen=True)
class BriefingEngine:
    root: Path = NORMALIZED_HOLDINGS_DIR  # validated snapshots (data/normalized/etf_holdings)

    def fetch_holdings(self, fund: FundConfig, requested_date: str) -> dict[str, Any] | None:
        return providers.fetch_holdings(fund, requested_date)

    def snapshot_dates(self, fund: FundConfig) -> list[str]:
        return storage.snapshot_dates(fund.snapshot_ticker, self.root)

    def fund_snapshots(self, fund_id: str, through_date: str) -> dict[str, dict[str, Any]]:
        return storage.load_fund_snapshots(fund_id, through_date, self.root)

    def krx_sessions(self, through_date: str) -> list[str]:
        return prices.krx_sessions(through_date, self.root)

    def fund_report(self, fund: FundConfig, target_date: str | None = None) -> str:
        return report.fund_report(fund, target_date, self.root)

    def core_digest(self, target_date: str) -> str:
        return digest.core_digest(target_date, self.root)

    def performance_report(self, target_date: str | None = None) -> str:
        return performance.performance_report(target_date, self.root)
