"""
Market detection and stock ticker utilities.

Identifies the market (US, China A-shares, Hong Kong) from a ticker symbol
and provides associated metadata (currency, data source, display names).
"""

import re
from enum import Enum
from typing import TypedDict


class StockMarket(str, Enum):
    US = "us"
    CHINA = "china"
    HK = "hk"
    UNKNOWN = "unknown"


class MarketInfo(TypedDict):
    market: str          # StockMarket value
    market_name: str     # Human-readable name
    currency_name: str   # e.g. "USD", "CNY", "HKD"
    currency_symbol: str # e.g. "$", "¥", "HK$"
    data_source: str     # Recommended data vendor
    is_us: bool
    is_china: bool
    is_hk: bool


# Regex patterns for market detection
_CHINA_PATTERN = re.compile(r"^\d{6}(\.(SZ|SH|BJ|SS))?$", re.IGNORECASE)
_HK_PATTERN = re.compile(r"^(\d{4,5})(\.HK)?$", re.IGNORECASE)
_US_PATTERN = re.compile(r"^[A-Z]{1,5}$", re.IGNORECASE)


class StockUtils:
    """Utilities for identifying stock markets and normalizing tickers."""

    @staticmethod
    def identify_market(ticker: str) -> StockMarket:
        """Identify the market from a ticker symbol.

        Rules:
            - 6-digit numbers (with optional .SZ/.SH/.BJ suffix) -> China A-shares
            - 4-5 digit numbers (with optional .HK suffix) -> Hong Kong
            - 1-5 uppercase letters -> US stocks
        """
        ticker = ticker.strip()

        # China A-shares: 6-digit codes like 600519, 000001.SZ
        if _CHINA_PATTERN.match(ticker):
            return StockMarket.CHINA

        # Hong Kong: 4-5 digit codes like 0700, 09988.HK
        if _HK_PATTERN.match(ticker):
            # Distinguish from China: HK is 4-5 digits, China is exactly 6
            digits = re.match(r"^(\d+)", ticker).group(1)
            if len(digits) <= 5:
                return StockMarket.HK

        # US: 1-5 letter codes like AAPL, NVDA, TSLA
        if _US_PATTERN.match(ticker):
            return StockMarket.US

        return StockMarket.UNKNOWN

    @staticmethod
    def get_market_info(ticker: str) -> MarketInfo:
        """Get comprehensive market information for a ticker."""
        market = StockUtils.identify_market(ticker)

        if market == StockMarket.CHINA:
            return MarketInfo(
                market=market.value,
                market_name="China A-shares",
                currency_name="CNY",
                currency_symbol="¥",
                data_source="akshare",
                is_us=False,
                is_china=True,
                is_hk=False,
            )
        elif market == StockMarket.HK:
            return MarketInfo(
                market=market.value,
                market_name="Hong Kong",
                currency_name="HKD",
                currency_symbol="HK$",
                data_source="yfinance",
                is_us=False,
                is_china=False,
                is_hk=True,
            )
        elif market == StockMarket.US:
            return MarketInfo(
                market=market.value,
                market_name="US",
                currency_name="USD",
                currency_symbol="$",
                data_source="yfinance",
                is_us=True,
                is_china=False,
                is_hk=False,
            )
        else:
            return MarketInfo(
                market=market.value,
                market_name="Unknown",
                currency_name="USD",
                currency_symbol="$",
                data_source="yfinance",
                is_us=False,
                is_china=False,
                is_hk=False,
            )

    @staticmethod
    def normalize_ticker(ticker: str, market: StockMarket = None) -> str:
        """Normalize a ticker for data source consumption.

        - HK tickers get .HK suffix if missing
        - China tickers stay as bare 6-digit codes (for akshare)
        - US tickers are uppercased
        """
        ticker = ticker.strip().upper()
        if market is None:
            market = StockUtils.identify_market(ticker)

        if market == StockMarket.HK:
            # Ensure .HK suffix for yfinance
            if not ticker.endswith(".HK"):
                digits = re.match(r"^(\d+)", ticker).group(1)
                return f"{digits}.HK"
        elif market == StockMarket.CHINA:
            # Strip any suffix for akshare, keep bare 6 digits
            return re.match(r"^(\d{6})", ticker).group(1)

        return ticker

    @staticmethod
    def is_china_stock(ticker: str) -> bool:
        return StockUtils.identify_market(ticker) == StockMarket.CHINA

    @staticmethod
    def is_hk_stock(ticker: str) -> bool:
        return StockUtils.identify_market(ticker) == StockMarket.HK

    @staticmethod
    def is_us_stock(ticker: str) -> bool:
        return StockUtils.identify_market(ticker) == StockMarket.US
