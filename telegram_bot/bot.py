"""
TradingAgents Telegram Bot

Send a message like "US NVDA" or "AAPL" and receive a full multi-agent
analysis report with a BUY / HOLD / SELL recommendation.

Usage:
    python -m telegram.bot                     # uses env vars
    python -m telegram.bot --token BOT_TOKEN   # explicit token

Environment variables:
    TELEGRAM_BOT_TOKEN          - Telegram Bot API token (from @BotFather)
    TELEGRAM_ALLOWED_USERS      - Comma-separated user IDs (empty = allow all)
    OPENAI_API_KEY / etc.       - LLM provider credentials
    ALPHA_VANTAGE_API_KEY       - Data vendor credentials
"""

import argparse
import asyncio
import datetime
import logging
import os
import re
import threading
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_TG_MESSAGE_LENGTH = 4096


def _split_message(text: str, limit: int = MAX_TG_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def _parse_request(text: str) -> tuple[str, Optional[str]]:
    """Parse a user message into (ticker, trade_date).

    Accepted formats:
        NVDA                  -> ("NVDA", today)
        US NVDA               -> ("NVDA", today)
        AAPL 2024-05-10       -> ("AAPL", "2024-05-10")
        US TSLA 2024-05-10    -> ("TSLA", "2024-05-10")
    """
    parts = text.strip().split()
    if not parts:
        raise ValueError("Empty message")

    # Strip optional market prefix (informational only for now)
    market_prefixes = {"US", "CN", "HK", "JP", "EU", "UK"}
    if parts[0].upper() in market_prefixes and len(parts) > 1:
        parts = parts[1:]

    ticker = parts[0].upper()
    trade_date = None
    if len(parts) >= 2:
        date_str = parts[1]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            trade_date = date_str

    if trade_date is None:
        trade_date = datetime.date.today().strftime("%Y-%m-%d")

    return ticker, trade_date


def _build_config(overrides: dict | None = None) -> dict:
    """Build a TradingAgents config, applying any overrides."""
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def _format_report(final_state: dict, decision: str, ticker: str, trade_date: str) -> str:
    """Format the analysis results into a readable Telegram report."""
    sections = []
    market_name = final_state.get("market_name", "")
    currency = final_state.get("currency", "$")
    market_label = f" [{market_name}]" if market_name else ""
    sections.append(f"📊 *Analysis Report: {ticker}{market_label} ({trade_date})*\n")

    if final_state.get("market_report"):
        sections.append(f"📈 *Market Analysis*\n{final_state['market_report'][:800]}")

    if final_state.get("sentiment_report"):
        sections.append(f"💬 *Sentiment*\n{final_state['sentiment_report'][:800]}")

    if final_state.get("news_report"):
        sections.append(f"📰 *News*\n{final_state['news_report'][:800]}")

    if final_state.get("fundamentals_report"):
        sections.append(f"📋 *Fundamentals*\n{final_state['fundamentals_report'][:800]}")

    debate = final_state.get("investment_debate_state", {})
    if debate.get("judge_decision"):
        sections.append(f"⚖️ *Research Team Decision*\n{debate['judge_decision'][:800]}")

    if final_state.get("trader_investment_plan"):
        sections.append(f"🤝 *Trader Plan*\n{final_state['trader_investment_plan'][:800]}")

    risk = final_state.get("risk_debate_state", {})
    if risk.get("judge_decision"):
        sections.append(f"🛡️ *Risk Management Decision*\n{risk['judge_decision'][:800]}")

    # Final decision
    decision_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(
        decision.upper(), "⚪"
    )
    sections.append(
        f"\n{decision_emoji} *FINAL DECISION: {decision.upper()}* {decision_emoji}"
    )

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Bot handlers
# ---------------------------------------------------------------------------

class TradingBot:
    """Wraps the Telegram bot with TradingAgents integration."""

    def __init__(self, token: str, config_overrides: dict | None = None,
                 allowed_users: set[int] | None = None):
        self.token = token
        self.config_overrides = config_overrides or {}
        self.allowed_users = allowed_users
        self._running_analyses: dict[int, bool] = {}  # chat_id -> True while running

    def _is_authorized(self, user_id: int) -> bool:
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "👋 Welcome to *TradingAgents Bot*!\n\n"
            "Send me a stock ticker to analyze. I support:\n"
            "🇺🇸 *US stocks*: `NVDA`, `AAPL`, `TSLA`\n"
            "🇨🇳 *China A-shares*: `600519`, `000001`\n"
            "🇭🇰 *Hong Kong*: `0700`, `9988`\n\n"
            "You can also specify a date:\n"
            "  `AAPL 2024-05-10`\n"
            "  `600519 2025-01-15`\n\n"
            "I'll run the full multi-agent analysis pipeline and send you a report "
            "with a BUY / HOLD / SELL recommendation.\n\n"
            "Commands:\n"
            "/start - Show this help\n"
            "/status - Check if an analysis is running",
            parse_mode="Markdown",
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if self._running_analyses.get(chat_id):
            await update.message.reply_text("⏳ An analysis is currently running. Please wait.")
        else:
            await update.message.reply_text("✅ No analysis running. Send a ticker to start!")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if not self._is_authorized(user_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return

        text = update.message.text.strip()
        if not text or text.startswith("/"):
            return

        # Parse the request
        try:
            ticker, trade_date = _parse_request(text)
        except ValueError:
            await update.message.reply_text(
                "❌ Could not parse your request.\n"
                "Examples: `NVDA`, `US AAPL 2024-05-10`",
                parse_mode="Markdown",
            )
            return

        # Prevent concurrent analyses per chat
        if self._running_analyses.get(chat_id):
            await update.message.reply_text(
                "⏳ An analysis is already running. Please wait for it to finish."
            )
            return

        self._running_analyses[chat_id] = True
        await update.message.reply_text(
            f"🔍 Starting analysis for *{ticker}* on *{trade_date}*...\n"
            "This may take a few minutes.",
            parse_mode="Markdown",
        )

        try:
            # Run the analysis in a thread to avoid blocking the event loop
            final_state, decision = await asyncio.get_event_loop().run_in_executor(
                None, self._run_analysis, ticker, trade_date
            )

            report = _format_report(final_state, decision, ticker, trade_date)
            for chunk in _split_message(report):
                await update.message.reply_text(chunk, parse_mode="Markdown")

        except Exception as e:
            logger.exception("Analysis failed for %s", ticker)
            await update.message.reply_text(
                f"❌ Analysis failed: {e}\n\nPlease check your API keys and try again."
            )
        finally:
            self._running_analyses.pop(chat_id, None)

    def _run_analysis(self, ticker: str, trade_date: str) -> tuple:
        """Synchronous wrapper that runs the TradingAgents pipeline."""
        config = _build_config(self.config_overrides)
        ta = TradingAgentsGraph(debug=False, config=config)
        final_state, decision = ta.propagate(ticker, trade_date)
        return final_state, decision

    def run(self):
        """Start the bot (blocking)."""
        app = Application.builder().token(self.token).build()
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        logger.info("TradingAgents Telegram bot is running...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="TradingAgents Telegram Bot")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"),
                        help="Telegram Bot API token")
    parser.add_argument("--allowed-users", default=os.getenv("TELEGRAM_ALLOWED_USERS", ""),
                        help="Comma-separated allowed user IDs (empty = all)")
    # LLM overrides
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--quick-model", default=None)
    parser.add_argument("--deep-model", default=None)

    args = parser.parse_args()

    if not args.token:
        parser.error(
            "Telegram bot token required. Set TELEGRAM_BOT_TOKEN env var or use --token."
        )

    allowed_users = set()
    if args.allowed_users:
        for uid in args.allowed_users.split(","):
            uid = uid.strip()
            if uid.isdigit():
                allowed_users.add(int(uid))

    config_overrides = {}
    if args.llm_provider:
        config_overrides["llm_provider"] = args.llm_provider
    if args.backend_url:
        config_overrides["backend_url"] = args.backend_url
    if args.api_key:
        config_overrides["api_key"] = args.api_key
    if args.quick_model:
        config_overrides["quick_think_llm"] = args.quick_model
    if args.deep_model:
        config_overrides["deep_think_llm"] = args.deep_model

    bot = TradingBot(
        token=args.token,
        config_overrides=config_overrides,
        allowed_users=allowed_users or None,
    )
    bot.run()


if __name__ == "__main__":
    main()
