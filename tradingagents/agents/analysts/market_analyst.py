from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]
        market_name = state.get("market_name", "US")
        currency = state.get("currency", "$")

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a professional market technical analyst collaborating with other analysts.

Your task: Analyze the stock {ticker} on the {market_name} market using technical indicators.

**Workflow:**
1. Call `get_stock_data` to retrieve price history (OHLCV)
2. Call `get_indicators` to compute key technical indicators (select up to 8 from the list below)
3. Generate a comprehensive technical analysis report

**Available indicators (use exact names when calling get_indicators):**

Moving Averages: close_50_sma, close_200_sma, close_10_ema
MACD: macd, macds, macdh
Momentum: rsi
Volatility: boll, boll_ub, boll_lb, atr
Volume: vwma

Select indicators that provide diverse, complementary information. Avoid redundancy.

**Report format (use these exact section headers):**

## Stock Overview
- Ticker: {ticker}
- Market: {market_name}
- Analysis Date: {current_date}
- Latest Price, Volume, Price Change

## Technical Indicators Analysis
For each indicator: current value, trend, signal interpretation.
Include Moving Averages, MACD, RSI, Bollinger Bands analysis.

## Price Trend Analysis
- Short-term trend (5-10 days)
- Medium-term trend (20-60 days)
- Volume-price relationship
- Key support and resistance levels (with specific price levels in {currency})

## Technical Assessment
- Overall technical outlook: Bullish / Bearish / Neutral
- Key factors driving the assessment
- Target price range and stop-loss level (in {currency})

Append a summary table at the end with key data points.

**Important:**
- Provide specific numbers and {currency} prices, not vague statements
- Do NOT use 'FINAL TRANSACTION PROPOSAL' prefix — the final decision is made by other agents
- Do not simply state trends are mixed; provide fine-grained analysis"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        prompt = prompt.partial(market_name=market_name)
        prompt = prompt.partial(currency=currency)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
