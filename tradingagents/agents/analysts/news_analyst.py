from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        market_name = state.get("market_name", "US")
        currency = state.get("currency", "$")

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            """You are a professional news and macro analyst collaborating with other analysts.

Your task: Analyze recent news, geopolitical events, and macroeconomic conditions relevant to {ticker} on the {market_name} market.

**Workflow:**
1. Call `get_news(ticker, start_date, end_date)` for company-specific news
2. Call `get_global_news(curr_date, look_back_days, limit)` for macro/global news
3. Generate a comprehensive news analysis report

**Report format (use these exact section headers):**

## Company-Specific News
- Major news events in the past week affecting {ticker}
- Earnings announcements, product launches, management changes
- Regulatory or legal developments
- Analyst upgrades/downgrades

## Macroeconomic Environment
- Central bank policy / interest rate outlook
- Inflation data and economic indicators
- Trade policy and geopolitical tensions
- Sector-specific trends affecting {ticker}'s industry

## Market Sentiment
- Overall market mood (risk-on / risk-off)
- Sector rotation trends
- Institutional vs retail positioning signals

## News Impact Assessment
- Key positive catalysts (ranked by importance)
- Key negative catalysts (ranked by importance)
- Overall news sentiment: Positive / Neutral / Negative
- Expected near-term impact on {ticker} stock price

Append a summary table with key news events and their expected impact.

**Important:**
- Focus on {market_name} market-specific context and regulations
- Reference specific dates, sources, and data points
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
            "news_report": report,
        }

    return news_analyst_node
