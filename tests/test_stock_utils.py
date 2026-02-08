"""Unit tests for tradingagents.utils.stock_utils."""

import pytest
from tradingagents.utils.stock_utils import StockUtils, StockMarket, MarketInfo


class TestIdentifyMarket:
    """Tests for StockUtils.identify_market()."""

    # --- US stocks ---
    @pytest.mark.parametrize("ticker", ["AAPL", "NVDA", "TSLA", "MSFT", "A", "GOOGL"])
    def test_us_stocks(self, ticker):
        assert StockUtils.identify_market(ticker) == StockMarket.US

    def test_us_lowercase(self):
        assert StockUtils.identify_market("aapl") == StockMarket.US

    def test_us_with_whitespace(self):
        assert StockUtils.identify_market("  NVDA  ") == StockMarket.US

    # --- China A-shares ---
    @pytest.mark.parametrize("ticker", ["600519", "000001", "300750", "688001"])
    def test_china_bare_codes(self, ticker):
        assert StockUtils.identify_market(ticker) == StockMarket.CHINA

    @pytest.mark.parametrize("ticker", ["600519.SH", "000001.SZ", "300750.sz", "830799.BJ"])
    def test_china_with_suffix(self, ticker):
        assert StockUtils.identify_market(ticker) == StockMarket.CHINA

    def test_china_with_ss_suffix(self):
        assert StockUtils.identify_market("600519.SS") == StockMarket.CHINA

    # --- Hong Kong ---
    @pytest.mark.parametrize("ticker", ["0700", "9988", "1810", "3690"])
    def test_hk_bare_codes(self, ticker):
        assert StockUtils.identify_market(ticker) == StockMarket.HK

    def test_hk_five_digits(self):
        assert StockUtils.identify_market("09988") == StockMarket.HK

    def test_hk_with_suffix(self):
        assert StockUtils.identify_market("0700.HK") == StockMarket.HK

    def test_hk_suffix_lowercase(self):
        assert StockUtils.identify_market("9988.hk") == StockMarket.HK

    # --- Unknown ---
    def test_empty_string(self):
        assert StockUtils.identify_market("") == StockMarket.UNKNOWN

    def test_long_letters(self):
        # 6+ letter tickers don't match US (1-5 letters)
        assert StockUtils.identify_market("ABCDEF") == StockMarket.UNKNOWN

    def test_mixed_alphanumeric(self):
        assert StockUtils.identify_market("ABC123") == StockMarket.UNKNOWN


class TestGetMarketInfo:
    """Tests for StockUtils.get_market_info()."""

    def test_us_market_info(self):
        info = StockUtils.get_market_info("AAPL")
        assert info["market"] == "us"
        assert info["market_name"] == "US"
        assert info["currency_name"] == "USD"
        assert info["currency_symbol"] == "$"
        assert info["data_source"] == "yfinance"
        assert info["is_us"] is True
        assert info["is_china"] is False
        assert info["is_hk"] is False

    def test_china_market_info(self):
        info = StockUtils.get_market_info("600519")
        assert info["market"] == "china"
        assert info["market_name"] == "China A-shares"
        assert info["currency_name"] == "CNY"
        assert info["currency_symbol"] == "¥"
        assert info["data_source"] == "akshare"
        assert info["is_china"] is True
        assert info["is_us"] is False

    def test_hk_market_info(self):
        info = StockUtils.get_market_info("0700")
        assert info["market"] == "hk"
        assert info["market_name"] == "Hong Kong"
        assert info["currency_name"] == "HKD"
        assert info["currency_symbol"] == "HK$"
        assert info["data_source"] == "yfinance"
        assert info["is_hk"] is True

    def test_unknown_market_info(self):
        info = StockUtils.get_market_info("TOOLONG")
        assert info["market"] == "unknown"
        assert info["market_name"] == "Unknown"
        # Falls back to USD
        assert info["currency_symbol"] == "$"


class TestNormalizeTicker:
    """Tests for StockUtils.normalize_ticker()."""

    def test_us_uppercased(self):
        assert StockUtils.normalize_ticker("aapl") == "AAPL"

    def test_hk_adds_suffix(self):
        assert StockUtils.normalize_ticker("0700") == "0700.HK"

    def test_hk_keeps_suffix(self):
        assert StockUtils.normalize_ticker("0700.HK") == "0700.HK"

    def test_china_strips_suffix(self):
        assert StockUtils.normalize_ticker("600519.SH") == "600519"

    def test_china_bare_code(self):
        assert StockUtils.normalize_ticker("600519") == "600519"

    def test_explicit_market_parameter(self):
        # Force HK market for a 4-digit code
        assert StockUtils.normalize_ticker("1234", StockMarket.HK) == "1234.HK"


class TestConvenienceMethods:
    """Tests for is_china_stock, is_hk_stock, is_us_stock."""

    def test_is_china_stock(self):
        assert StockUtils.is_china_stock("600519") is True
        assert StockUtils.is_china_stock("AAPL") is False

    def test_is_hk_stock(self):
        assert StockUtils.is_hk_stock("0700") is True
        assert StockUtils.is_hk_stock("AAPL") is False

    def test_is_us_stock(self):
        assert StockUtils.is_us_stock("AAPL") is True
        assert StockUtils.is_us_stock("600519") is False
