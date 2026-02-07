from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]
        market_name = state.get("market_name", "US")
        currency = state.get("currency", "$")

        tools = [
            get_news,
        ]

        system_message = (
            """You are a professional social media and sentiment analyst collaborating with other analysts.

Your task: Analyze social media posts, public sentiment, and community discussions about {ticker} on the {market_name} market.

**Workflow:**
1. Call `get_news(ticker, start_date, end_date)` to search for company-specific social media discussions and news
2. Analyze the tone, volume, and trends in public sentiment
3. Generate a comprehensive sentiment analysis report

**Report format (use these exact section headers):**

## Social Media Overview
- Key platforms and sources analyzed
- Volume of mentions / discussions over the past week
- Trending topics and hashtags related to {ticker}

## Sentiment Analysis
- Overall sentiment score: Bullish / Neutral / Bearish
- Day-by-day sentiment trend over the past week
- Key positive narratives (with quotes/examples)
- Key negative narratives (with quotes/examples)
- Sentiment shift detection (any sudden changes)

## Community & Institutional Signals
- Retail investor sentiment (Reddit, StockTwits, forums)
- Analyst commentary trends
- Short interest and options flow signals (if mentioned in news)

## Sentiment Assessment
- How social sentiment aligns with or diverges from fundamentals
- Key catalysts driving public opinion
- Risk of sentiment-driven volatility
- Overall social sentiment outlook: Positive / Neutral / Negative

Append a summary table with key sentiment indicators and their readings.

**Important:**
- Cite specific sources, dates, and quotes where possible
- Consider {market_name} market-specific social media platforms and sentiment patterns
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
