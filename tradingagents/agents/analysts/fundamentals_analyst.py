from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_sentiment, get_insider_transactions
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]
        market_name = state.get("market_name", "US")
        currency = state.get("currency", "$")

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            """You are a professional fundamentals analyst collaborating with other analysts.

Your task: Analyze the fundamental financial health of {ticker} on the {market_name} market.

**Workflow:**
1. Call `get_fundamentals` for a comprehensive company overview (key ratios, valuation metrics)
2. Call `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for detailed financial statements
3. Generate a comprehensive fundamental analysis report

**Report format (use these exact section headers):**

## Company Overview
- Ticker: {ticker}
- Market: {market_name}
- Analysis Date: {current_date}
- Market Cap, Sector/Industry, Key Business Description

## Valuation Analysis
- P/E Ratio (TTM and Forward)
- P/B Ratio
- EV/EBITDA
- PEG Ratio (if available)
- Comparison to sector averages
- Assessment: Overvalued / Fairly Valued / Undervalued

## Financial Health
- Revenue trend (YoY growth)
- Net income / profit margins
- Debt-to-equity ratio
- Current ratio / quick ratio
- Free cash flow trend
- ROE, ROA

## Growth & Competitive Position
- Revenue growth trajectory
- Earnings growth trajectory
- Competitive advantages (moat)
- Key risks and challenges

## Fundamental Assessment
- Overall fundamental outlook: Strong / Moderate / Weak
- Key factors driving the assessment
- Fair value estimate (in {currency}) if data supports it

Append a summary table at the end with key financial metrics.

**Important:**
- All monetary values in {currency}
- Include specific numbers from the financial statements
- Do NOT use 'FINAL TRANSACTION PROPOSAL' prefix — the final decision is made by other agents
- Do not simply state the fundamentals are mixed; provide fine-grained analysis"""
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
