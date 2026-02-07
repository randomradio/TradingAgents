import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]
        market_name = state.get("market_name", "US")
        currency = state.get("currency", "$")

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader.

**Context:**
- Stock: {company_name}
- Market: {market_name}
- Currency: {currency}

Your decision must result in a clear recommendation: **FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the {market_name} market context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on the analysts' insights.
4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision.
5. **Consider Market-Specific Risks**: Factor in {market_name} market regulations, trading rules, and macro conditions.

**Report format:**

## Debate Summary
- Key arguments from each analyst (Risky, Safe, Neutral)
- Points of agreement and disagreement

## Risk Assessment
- Overall risk level: High / Moderate / Low
- Key risk factors specific to {market_name} market
- Downside scenario and probability

## Final Decision
FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**
- Position size recommendation
- Entry/exit price levels (in {currency})
- Stop-loss level (in {currency})
- Time horizon

---

**Analysts Debate History:**
{history}

---

Choose Hold only if strongly justified by specific arguments, not as a fallback. Strive for clarity and decisiveness. Focus on actionable insights and continuous improvement."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
