"""Liquidation data downloader -- stub.

TODO: Implement when a reliable liquidation data source is identified.
Binance um liquidationSnapshot returns 404 for all dates. Coinalyze
requires an API key and has rate limits. This is deferred to a later phase.
"""
from __future__ import annotations


class LiquidationDownloader:
    """Placeholder for liquidation data downloader."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "LiquidationDownloader is not yet implemented. "
            "See liquidations.py docstring for details."
        )
