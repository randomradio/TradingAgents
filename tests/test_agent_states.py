"""Unit tests for agent states and propagation."""

import pytest


class TestAgentStateFields:
    """Verify AgentState has the expected market-related fields."""

    def test_agent_state_has_market_fields(self):
        from tradingagents.agents.utils.agent_states import AgentState
        annotations = AgentState.__annotations__
        assert "market" in annotations
        assert "market_name" in annotations
        assert "currency" in annotations

    def test_invest_debate_state_fields(self):
        from tradingagents.agents.utils.agent_states import InvestDebateState
        annotations = InvestDebateState.__annotations__
        assert "history" in annotations
        assert "current_response" in annotations
        assert "count" in annotations
        assert "judge_decision" in annotations
        assert "bull_history" in annotations
        assert "bear_history" in annotations

    def test_risk_debate_state_fields(self):
        from tradingagents.agents.utils.agent_states import RiskDebateState
        annotations = RiskDebateState.__annotations__
        assert "history" in annotations
        assert "risky_history" in annotations
        assert "safe_history" in annotations
        assert "neutral_history" in annotations
        assert "latest_speaker" in annotations
        assert "current_risky_response" in annotations
        assert "current_safe_response" in annotations
        assert "current_neutral_response" in annotations
        assert "judge_decision" in annotations
        assert "count" in annotations

    def test_agent_state_has_report_fields(self):
        from tradingagents.agents.utils.agent_states import AgentState
        annotations = AgentState.__annotations__
        assert "market_report" in annotations
        assert "sentiment_report" in annotations
        assert "news_report" in annotations
        assert "fundamentals_report" in annotations
        assert "investment_plan" in annotations
        assert "trader_investment_plan" in annotations
        assert "final_trade_decision" in annotations


class TestPropagator:
    """Tests for Propagator.create_initial_state()."""

    def test_us_stock_initial_state(self):
        from tradingagents.graph.propagation import Propagator
        prop = Propagator(max_recur_limit=50)
        state = prop.create_initial_state("NVDA", "2024-05-10")

        assert state["company_of_interest"] == "NVDA"
        assert state["trade_date"] == "2024-05-10"
        assert state["market"] == "us"
        assert state["market_name"] == "US"
        assert state["currency"] == "$"
        assert state["market_report"] == ""
        assert state["fundamentals_report"] == ""

    def test_china_stock_initial_state(self):
        from tradingagents.graph.propagation import Propagator
        prop = Propagator()
        state = prop.create_initial_state("600519", "2025-01-15")

        assert state["company_of_interest"] == "600519"
        assert state["market"] == "china"
        assert state["market_name"] == "China A-shares"
        assert state["currency"] == "¥"

    def test_hk_stock_initial_state(self):
        from tradingagents.graph.propagation import Propagator
        prop = Propagator()
        state = prop.create_initial_state("0700", "2025-01-15")

        assert state["market"] == "hk"
        assert state["market_name"] == "Hong Kong"
        assert state["currency"] == "HK$"

    def test_initial_state_has_debate_states(self):
        from tradingagents.graph.propagation import Propagator
        prop = Propagator()
        state = prop.create_initial_state("AAPL", "2024-01-01")

        assert "investment_debate_state" in state
        assert state["investment_debate_state"]["count"] == 0
        assert state["investment_debate_state"]["history"] == ""

        assert "risk_debate_state" in state
        assert state["risk_debate_state"]["count"] == 0
        assert state["risk_debate_state"]["history"] == ""

    def test_graph_args(self):
        from tradingagents.graph.propagation import Propagator
        prop = Propagator(max_recur_limit=75)
        args = prop.get_graph_args()

        assert args["stream_mode"] == "values"
        assert args["config"]["recursion_limit"] == 75
