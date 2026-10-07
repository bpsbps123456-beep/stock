"""Load the canonical GBRC fund configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "etfs.json"


@dataclass(frozen=True)
class FundConfig:
    alias: str
    aliases: tuple[str, ...]
    display_name: str
    snapshot_ticker: str
    listing_ticker: str
    provider: str
    provider_id: str
    source_url: str

    @property
    def legacy_idx(self) -> int | None:
        if self.provider == "time":
            return int(self.provider_id)
        return None


@lru_cache(maxsize=1)
def load_fund_configs() -> dict[str, FundConfig]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported config/etfs.json schema_version")

    funds: dict[str, FundConfig] = {}
    for alias, raw in payload.get("funds", {}).items():
        funds[alias] = FundConfig(
            alias=alias,
            aliases=tuple(str(item).lower() for item in raw["aliases"]),
            display_name=raw["display_name"],
            snapshot_ticker=raw["snapshot_ticker"],
            listing_ticker=raw["listing_ticker"],
            provider=raw["provider"],
            provider_id=str(raw["provider_id"]),
            source_url=raw["source_url"],
        )

    if set(funds) != {"time", "koact"}:
        raise ValueError("Exactly the time and koact funds must be configured")
    return funds


def fund_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, fund in load_fund_configs().items():
        aliases[key] = key
        for alias in fund.aliases:
            if alias in aliases and aliases[alias] != key:
                raise ValueError(f"Duplicate ETF alias: {alias}")
            aliases[alias] = key
    return aliases


@lru_cache(maxsize=1)
def observation_primary_alias() -> str:
    """Fund whose US-listed Top 10 anchors the briefing watch list."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    alias = str(payload.get("observation_primary", "time"))
    if alias not in load_fund_configs():
        raise ValueError(f"observation_primary must be a configured fund, got {alias!r}")
    return alias


def resolve_fund(alias: str | None) -> FundConfig:
    requested = (alias or "time").lower()
    key = fund_alias_map().get(requested)
    if not key:
        choices = ", ".join(sorted(fund_alias_map()))
        raise ValueError(f"Unknown ETF alias {alias!r}. Choices: {choices}")
    return load_fund_configs()[key]


def performance_funds() -> list[dict[str, str]]:
    return [
        {
            "label": "TIME" if fund.alias == "time" else "KoAct",
            "snapshot_ticker": fund.snapshot_ticker,
            "price_ticker": fund.listing_ticker,
            "name": fund.display_name,
        }
        for fund in load_fund_configs().values()
    ]
