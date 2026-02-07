"""Unit tests for agent factory functions.

Verifies that the factory pattern works: each create_* function returns
a callable node function that interacts with state correctly.
We use a mock LLM to avoid real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch


def _make_mock_llm(content="Mock response"):
    """Create a mock LLM that returns a predictable response."""
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    llm.invoke.return_value = response
    return llm


def _make_mock_memory(memories=None):
    """Create a mock memory object."""
    mem = MagicMock()
    mem.get_memories.return_value = memories or []
    return mem


def _base_state():
    """Return a minimal agent state dict for testing."""
    return {
        "messages": [],
        "company_of_interest": "AAPL",
        "trade_date": "2024-05-10",
        "market": "us",
        "market_name": "US",
        "currency": "$",
        "market_report": "Market report text",
        "sentiment_report": "Sentiment report text",
        "news_report": "News report text",
        "fundamentals_report": "Fundamentals report text",
        "investment_plan": "Buy AAPL at $150",
        "trader_investment_plan": "Buy AAPL at $150 with stop-loss at $140",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "risk_debate_state": {
            "history": "",
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }


class TestTraderFactory:
    """Tests for create_trader()."""

    def test_returns_callable(self):
        from tradingagents.agents.trader.trader import create_trader
        llm = _make_mock_llm()
        mem = _make_mock_memory()
        node = create_trader(llm, mem)
        assert callable(node)

    def test_node_returns_expected_keys(self):
        from tradingagents.agents.trader.trader import create_trader
        llm = _make_mock_llm("BUY at $200. FINAL TRANSACTION PROPOSAL: **BUY**")
        mem = _make_mock_memory()
        node = create_trader(llm, mem)
        result = node(_base_state())
        assert "messages" in result
        assert "trader_investment_plan" in result
        assert "sender" in result
        assert result["sender"] == "Trader"

    def test_node_invokes_llm(self):
        from tradingagents.agents.trader.trader import create_trader
        llm = _make_mock_llm()
        mem = _make_mock_memory()
        node = create_trader(llm, mem)
        node(_base_state())
        llm.invoke.assert_called_once()

    def test_node_queries_memory(self):
        from tradingagents.agents.trader.trader import create_trader
        llm = _make_mock_llm()
        mem = _make_mock_memory()
        node = create_trader(llm, mem)
        node(_base_state())
        mem.get_memories.assert_called_once()


class TestRiskyDebatorFactory:
    """Tests for create_risky_debator()."""

    def test_returns_callable(self):
        from tradingagents.agents.risk_mgmt.aggresive_debator import create_risky_debator
        node = create_risky_debator(_make_mock_llm())
        assert callable(node)

    def test_node_returns_risk_debate_state(self):
        from tradingagents.agents.risk_mgmt.aggresive_debator import create_risky_debator
        llm = _make_mock_llm("Risk is worth it!")
        node = create_risky_debator(llm)
        result = node(_base_state())
        assert "risk_debate_state" in result
        rds = result["risk_debate_state"]
        assert rds["latest_speaker"] == "Risky"
        assert rds["count"] == 1
        assert "Risky Analyst:" in rds["current_risky_response"]

    def test_increments_count(self):
        from tradingagents.agents.risk_mgmt.aggresive_debator import create_risky_debator
        state = _base_state()
        state["risk_debate_state"]["count"] = 3
        node = create_risky_debator(_make_mock_llm())
        result = node(state)
        assert result["risk_debate_state"]["count"] == 4


class TestSafeDebatorFactory:
    """Tests for create_safe_debator()."""

    def test_returns_callable(self):
        from tradingagents.agents.risk_mgmt.conservative_debator import create_safe_debator
        node = create_safe_debator(_make_mock_llm())
        assert callable(node)

    def test_node_returns_risk_debate_state(self):
        from tradingagents.agents.risk_mgmt.conservative_debator import create_safe_debator
        node = create_safe_debator(_make_mock_llm("Be cautious!"))
        result = node(_base_state())
        rds = result["risk_debate_state"]
        assert rds["latest_speaker"] == "Safe"
        assert "Safe Analyst:" in rds["current_safe_response"]


class TestNeutralDebatorFactory:
    """Tests for create_neutral_debator()."""

    def test_returns_callable(self):
        from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
        node = create_neutral_debator(_make_mock_llm())
        assert callable(node)

    def test_node_returns_risk_debate_state(self):
        from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
        node = create_neutral_debator(_make_mock_llm("Balanced view."))
        result = node(_base_state())
        rds = result["risk_debate_state"]
        assert rds["latest_speaker"] == "Neutral"
        assert "Neutral Analyst:" in rds["current_neutral_response"]


class TestRiskManagerFactory:
    """Tests for create_risk_manager()."""

    def test_returns_callable(self):
        from tradingagents.agents.managers.risk_manager import create_risk_manager
        node = create_risk_manager(_make_mock_llm(), _make_mock_memory())
        assert callable(node)

    def test_node_returns_final_decision(self):
        from tradingagents.agents.managers.risk_manager import create_risk_manager
        llm = _make_mock_llm("FINAL TRANSACTION PROPOSAL: **BUY**")
        mem = _make_mock_memory()
        node = create_risk_manager(llm, mem)
        result = node(_base_state())
        assert "final_trade_decision" in result
        assert "risk_debate_state" in result
        assert result["risk_debate_state"]["latest_speaker"] == "Judge"

    def test_risk_manager_reads_fundamentals_report(self):
        """Verify the bug fix: risk_manager reads fundamentals_report, not news_report.

        The fundamentals report is used in curr_situation (passed to memory),
        not directly in the LLM prompt. We verify that memory.get_memories
        receives a string containing the fundamentals marker.
        """
        from tradingagents.agents.managers.risk_manager import create_risk_manager
        llm = _make_mock_llm("Decision")
        mem = _make_mock_memory()
        node = create_risk_manager(llm, mem)

        state = _base_state()
        state["fundamentals_report"] = "UNIQUE_FUNDAMENTALS_MARKER"
        state["news_report"] = "DIFFERENT_NEWS_MARKER"

        node(state)
        # curr_situation is passed to memory.get_memories — it should include fundamentals
        mem_call_args = mem.get_memories.call_args
        situation_str = mem_call_args[0][0]
        assert "UNIQUE_FUNDAMENTALS_MARKER" in situation_str
