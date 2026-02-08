"""Unit tests for tradingagents.default_config."""

import pytest


class TestDefaultConfig:
    """Verify DEFAULT_CONFIG has all expected keys and valid defaults."""

    def test_has_all_required_keys(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        required_keys = [
            "llm_provider",
            "deep_think_llm",
            "quick_think_llm",
            "backend_url",
            "api_key",
            "max_debate_rounds",
            "max_risk_discuss_rounds",
            "max_recur_limit",
            "data_vendors",
            "tool_vendors",
            "project_dir",
            "results_dir",
            "data_cache_dir",
        ]
        for key in required_keys:
            assert key in DEFAULT_CONFIG, f"Missing config key: {key}"

    def test_llm_provider_default(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["llm_provider"] == "openai"

    def test_api_key_none_by_default(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["api_key"] is None

    def test_data_vendors_structure(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        vendors = DEFAULT_CONFIG["data_vendors"]
        assert "core_stock_apis" in vendors
        assert "technical_indicators" in vendors
        assert "fundamental_data" in vendors
        assert "news_data" in vendors

    def test_debate_rounds_positive(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["max_debate_rounds"] >= 1
        assert DEFAULT_CONFIG["max_risk_discuss_rounds"] >= 1

    def test_recur_limit_reasonable(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["max_recur_limit"] >= 10

    def test_tool_vendors_is_dict(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG["tool_vendors"], dict)

    def test_backend_url_is_string(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG["backend_url"], str)
        assert DEFAULT_CONFIG["backend_url"].startswith("http")


class TestDataflowConfig:
    """Tests for the dataflows config singleton."""

    def test_set_and_get_config(self):
        from tradingagents.dataflows.config import get_config, set_config
        test_config = {"foo": "bar", "data_vendors": {}}
        set_config(test_config)
        result = get_config()
        assert result["foo"] == "bar"

    def test_get_config_returns_dict(self):
        from tradingagents.dataflows.config import get_config, set_config
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        config = get_config()
        assert isinstance(config, dict)
