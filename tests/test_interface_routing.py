"""Unit tests for tradingagents.dataflows.interface routing logic."""

import pytest
from unittest.mock import patch, MagicMock


class TestDetectMarketVendor:
    """Tests for _detect_market_vendor() auto-routing."""

    def test_china_ticker_routes_to_akshare(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_stock_data", ("600519",))
        assert result == "akshare"

    def test_china_ticker_with_suffix(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_stock_data", ("000001.SZ",))
        assert result == "akshare"

    def test_us_ticker_returns_none(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_stock_data", ("AAPL",))
        assert result is None

    def test_hk_ticker_returns_none(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_stock_data", ("0700",))
        assert result is None

    def test_empty_args_returns_none(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_stock_data", ())
        assert result is None

    def test_none_args_returns_none(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_stock_data", None)
        assert result is None

    def test_method_without_akshare_returns_none(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        # get_global_news has no akshare implementation
        result = _detect_market_vendor("get_global_news", ("600519",))
        assert result is None

    def test_unknown_method_returns_none(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("nonexistent_method", ("600519",))
        assert result is None

    def test_china_ticker_for_indicators(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_indicators", ("300750",))
        assert result == "akshare"

    def test_china_ticker_for_fundamentals(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_fundamentals", ("600519",))
        assert result == "akshare"

    def test_china_ticker_for_news(self):
        from tradingagents.dataflows.interface import _detect_market_vendor
        result = _detect_market_vendor("get_news", ("000001",))
        assert result == "akshare"


class TestGetCategoryForMethod:
    """Tests for get_category_for_method()."""

    def test_stock_data(self):
        from tradingagents.dataflows.interface import get_category_for_method
        assert get_category_for_method("get_stock_data") == "core_stock_apis"

    def test_indicators(self):
        from tradingagents.dataflows.interface import get_category_for_method
        assert get_category_for_method("get_indicators") == "technical_indicators"

    def test_fundamentals(self):
        from tradingagents.dataflows.interface import get_category_for_method
        assert get_category_for_method("get_fundamentals") == "fundamental_data"

    def test_news(self):
        from tradingagents.dataflows.interface import get_category_for_method
        assert get_category_for_method("get_news") == "news_data"

    def test_unknown_raises(self):
        from tradingagents.dataflows.interface import get_category_for_method
        with pytest.raises(ValueError, match="not found"):
            get_category_for_method("nonexistent")


class TestVendorMethods:
    """Tests for VENDOR_METHODS structure."""

    def test_akshare_registered_for_stock_data(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "akshare" in VENDOR_METHODS["get_stock_data"]

    def test_akshare_registered_for_indicators(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "akshare" in VENDOR_METHODS["get_indicators"]

    def test_akshare_registered_for_fundamentals(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "akshare" in VENDOR_METHODS["get_fundamentals"]

    def test_akshare_registered_for_news(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "akshare" in VENDOR_METHODS["get_news"]

    def test_yfinance_registered_for_stock_data(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "yfinance" in VENDOR_METHODS["get_stock_data"]

    def test_vendor_list_includes_akshare(self):
        from tradingagents.dataflows.interface import VENDOR_LIST
        assert "akshare" in VENDOR_LIST


class TestGetVendor:
    """Tests for get_vendor() config resolution."""

    def test_tool_level_overrides_category(self):
        from tradingagents.dataflows.interface import get_vendor
        from tradingagents.dataflows.config import set_config
        test_config = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {"get_stock_data": "alpha_vantage"},
        }
        set_config(test_config)
        result = get_vendor("core_stock_apis", "get_stock_data")
        assert result == "alpha_vantage"

    def test_falls_back_to_category(self):
        from tradingagents.dataflows.interface import get_vendor
        from tradingagents.dataflows.config import set_config
        test_config = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {},
        }
        set_config(test_config)
        result = get_vendor("core_stock_apis", "get_stock_data")
        assert result == "yfinance"
