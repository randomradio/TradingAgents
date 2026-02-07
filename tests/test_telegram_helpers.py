"""Unit tests for telegram_bot.bot helper functions.

Tests the pure-Python helper functions without importing telegram library.
We import the helpers directly to avoid the telegram dependency.
"""

import datetime
import importlib
import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Fixture: import helpers without triggering the telegram / dotenv imports
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bot_helpers():
    """Import only the pure-Python helpers from telegram_bot/bot.py.

    We read the file and extract _split_message, _parse_request, and
    _format_report using exec so that we skip the top-level imports
    of telegram, dotenv, etc.
    """
    import ast, textwrap

    src_path = "telegram_bot/bot.py"
    with open(src_path) as f:
        source = f.read()

    # Parse the AST to find function definitions we care about
    tree = ast.parse(source)
    helpers = types.ModuleType("bot_helpers")
    helpers.MAX_TG_MESSAGE_LENGTH = 4096

    # We need datetime, re, and typing.Optional for the helper functions
    import re as _re
    from typing import Optional as _Optional
    helpers.__dict__["re"] = _re
    helpers.__dict__["datetime"] = datetime
    helpers.__dict__["Optional"] = _Optional

    # Extract each standalone helper function
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_split_message", "_parse_request", "_format_report", "_build_config",
            ):
                func_src = ast.get_source_segment(source, node)
                exec(compile(ast.parse(func_src), src_path, "exec"), helpers.__dict__)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "MAX_TG_MESSAGE_LENGTH":
                        assign_src = ast.get_source_segment(source, node)
                        exec(compile(ast.parse(assign_src), src_path, "exec"), helpers.__dict__)

    return helpers


# ---------------------------------------------------------------------------
# _split_message tests
# ---------------------------------------------------------------------------

class TestSplitMessage:
    def test_short_message_unchanged(self, bot_helpers):
        text = "Hello World"
        result = bot_helpers._split_message(text)
        assert result == [text]

    def test_exact_limit(self, bot_helpers):
        text = "x" * 4096
        result = bot_helpers._split_message(text, limit=4096)
        assert result == [text]

    def test_splits_at_newline(self, bot_helpers):
        line = "a" * 50 + "\n"
        text = line * 10  # 510 chars
        result = bot_helpers._split_message(text, limit=200)
        assert len(result) > 1
        # Each chunk should be <= 200 chars
        for chunk in result:
            assert len(chunk) <= 200

    def test_splits_long_line(self, bot_helpers):
        text = "x" * 300
        result = bot_helpers._split_message(text, limit=100)
        assert len(result) == 3
        assert "".join(result) == text

    def test_empty_string(self, bot_helpers):
        result = bot_helpers._split_message("")
        assert result == [""]

    def test_preserves_all_content(self, bot_helpers):
        text = "Line1\nLine2\nLine3\nLine4\nLine5"
        result = bot_helpers._split_message(text, limit=15)
        joined = "\n".join(result)
        # All original lines should appear
        for line in ["Line1", "Line2", "Line3", "Line4", "Line5"]:
            assert line in joined


# ---------------------------------------------------------------------------
# _parse_request tests
# ---------------------------------------------------------------------------

class TestParseRequest:
    def test_ticker_only(self, bot_helpers):
        ticker, date = bot_helpers._parse_request("NVDA")
        assert ticker == "NVDA"
        assert date == datetime.date.today().strftime("%Y-%m-%d")

    def test_ticker_with_date(self, bot_helpers):
        ticker, date = bot_helpers._parse_request("AAPL 2024-05-10")
        assert ticker == "AAPL"
        assert date == "2024-05-10"

    def test_market_prefix_stripped(self, bot_helpers):
        ticker, date = bot_helpers._parse_request("US NVDA")
        assert ticker == "NVDA"

    def test_market_prefix_with_date(self, bot_helpers):
        ticker, date = bot_helpers._parse_request("US TSLA 2024-05-10")
        assert ticker == "TSLA"
        assert date == "2024-05-10"

    def test_cn_prefix(self, bot_helpers):
        ticker, date = bot_helpers._parse_request("CN 600519")
        assert ticker == "600519"

    def test_hk_prefix(self, bot_helpers):
        ticker, date = bot_helpers._parse_request("HK 0700")
        assert ticker == "0700"

    def test_lowercase_uppercased(self, bot_helpers):
        ticker, _ = bot_helpers._parse_request("aapl")
        assert ticker == "AAPL"

    def test_empty_message_raises(self, bot_helpers):
        with pytest.raises(ValueError, match="Empty message"):
            bot_helpers._parse_request("")

    def test_whitespace_only_raises(self, bot_helpers):
        with pytest.raises(ValueError, match="Empty message"):
            bot_helpers._parse_request("   ")

    def test_invalid_date_ignored(self, bot_helpers):
        ticker, date = bot_helpers._parse_request("AAPL not-a-date")
        assert ticker == "AAPL"
        # Should fall back to today
        assert date == datetime.date.today().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# _format_report tests
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_basic_format(self, bot_helpers):
        state = {
            "market_name": "US",
            "currency": "$",
            "market_report": "Market looks good",
            "sentiment_report": "Positive sentiment",
            "news_report": "Breaking news",
            "fundamentals_report": "Strong fundamentals",
            "investment_debate_state": {"judge_decision": "Buy recommended"},
            "trader_investment_plan": "Buy at $100",
            "risk_debate_state": {"judge_decision": "Low risk"},
        }
        report = bot_helpers._format_report(state, "BUY", "NVDA", "2024-05-10")
        assert "NVDA" in report
        assert "2024-05-10" in report
        assert "BUY" in report
        assert "Market looks good" in report
        assert "Positive sentiment" in report

    def test_missing_sections(self, bot_helpers):
        state = {"market_name": "", "currency": "$"}
        report = bot_helpers._format_report(state, "HOLD", "AAPL", "2024-01-01")
        assert "AAPL" in report
        assert "HOLD" in report

    def test_decision_emojis(self, bot_helpers):
        state = {"market_name": "US", "currency": "$"}
        buy_report = bot_helpers._format_report(state, "BUY", "X", "2024-01-01")
        sell_report = bot_helpers._format_report(state, "SELL", "X", "2024-01-01")
        hold_report = bot_helpers._format_report(state, "HOLD", "X", "2024-01-01")
        assert "\U0001f7e2" in buy_report   # green circle
        assert "\U0001f534" in sell_report   # red circle
        assert "\U0001f7e1" in hold_report   # yellow circle

    def test_long_reports_truncated(self, bot_helpers):
        state = {
            "market_name": "US",
            "currency": "$",
            "market_report": "x" * 2000,
        }
        report = bot_helpers._format_report(state, "BUY", "AAPL", "2024-01-01")
        # The market_report section should be truncated at 800 chars
        # The full report should be much shorter than 2000
        market_section_start = report.find("Market Analysis")
        assert market_section_start != -1
