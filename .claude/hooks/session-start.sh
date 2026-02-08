#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Upgrade setuptools first to avoid build failures with legacy packages
pip install --upgrade setuptools > /dev/null 2>&1

# Install core project dependencies
pip install \
  typing-extensions \
  langchain-openai \
  langchain-experimental \
  pandas \
  yfinance \
  praw \
  feedparser \
  stockstats \
  langgraph \
  chromadb \
  akshare \
  requests \
  tqdm \
  pytz \
  rich \
  questionary \
  langchain_anthropic \
  langchain-google-genai \
  python-telegram-bot \
  python-dotenv \
  pytest \
  > /dev/null 2>&1

# Create .env from example if it doesn't exist
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
fi

# Export PYTHONPATH so imports resolve correctly
echo 'export PYTHONPATH="$CLAUDE_PROJECT_DIR"' >> "$CLAUDE_ENV_FILE"
