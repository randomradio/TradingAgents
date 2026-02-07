# CLAUDE.md - TradingAgents Codebase Guide

## Project Overview

TradingAgents is a **multi-agent LLM financial trading framework** that mirrors real-world trading firms. It deploys specialized LLM-powered agents (analysts, researchers, traders, risk managers) that collaboratively evaluate market conditions and make trading decisions through structured debates. Built with **LangGraph** for orchestration and **LangChain** for LLM integration.

**License:** Apache 2.0
**Python:** >=3.10 (recommended: 3.13)
**Paper:** [arXiv:2412.20138](https://arxiv.org/abs/2412.20138)

## Repository Structure

```
TradingAgents/
├── main.py                          # Main entry point for Python usage
├── test.py                          # Ad-hoc test script (not a test suite)
├── cli/                             # Interactive CLI application (Typer + Rich)
│   ├── main.py                      # CLI entry point, display logic, streaming
│   ├── models.py                    # Pydantic/Enum models (AnalystType)
│   ├── utils.py                     # Interactive prompts (questionary)
│   └── static/welcome.txt           # ASCII art welcome banner
├── telegram_bot/                    # Telegram bot integration
│   ├── __init__.py
│   └── bot.py                       # TradingBot: message parsing, analysis, reporting
├── tradingagents/                   # Core package
│   ├── default_config.py            # DEFAULT_CONFIG dict (LLMs, vendors, paths)
│   ├── agents/                      # All agent definitions
│   │   ├── __init__.py              # Re-exports all agent factory functions
│   │   ├── analysts/                # 4 analyst agents
│   │   │   ├── market_analyst.py        # Technical/price analysis (uses tools)
│   │   │   ├── social_media_analyst.py  # Social sentiment (uses tools)
│   │   │   ├── news_analyst.py          # News & insider info (uses tools)
│   │   │   └── fundamentals_analyst.py  # Company financials (uses tools)
│   │   ├── researchers/             # Bull vs Bear debate agents
│   │   │   ├── bull_researcher.py       # Advocates for buying
│   │   │   └── bear_researcher.py       # Advocates against buying
│   │   ├── managers/                # Decision-making judges
│   │   │   ├── research_manager.py      # Judges bull/bear debate
│   │   │   └── risk_manager.py          # Judges risk debate, final decision
│   │   ├── risk_mgmt/              # Risk assessment debate agents
│   │   │   ├── aggresive_debator.py     # High-risk advocate (note: typo in filename)
│   │   │   ├── conservative_debator.py  # Low-risk advocate
│   │   │   └── neutral_debator.py       # Balanced perspective
│   │   ├── trader/
│   │   │   └── trader.py               # Generates trade proposals
│   │   └── utils/
│   │       ├── agent_states.py          # TypedDict state definitions (AgentState, InvestDebateState, RiskDebateState)
│   │       ├── agent_utils.py           # Message helpers, tool re-exports
│   │       ├── core_stock_tools.py      # @tool: get_stock_data
│   │       ├── technical_indicators_tools.py  # @tool: get_indicators
│   │       ├── fundamental_data_tools.py      # @tool: get_fundamentals, balance sheet, etc.
│   │       ├── news_data_tools.py       # @tool: get_news, get_global_news, insider data
│   │       └── memory.py               # ChromaDB-backed FinancialSituationMemory
│   ├── dataflows/                   # Data vendor abstraction layer
│   │   ├── __init__.py
│   │   ├── interface.py             # Vendor routing: VENDOR_METHODS, route_to_vendor()
│   │   ├── config.py               # Global config singleton (get_config/set_config)
│   │   ├── y_finance.py            # yfinance implementations
│   │   ├── yfin_utils.py           # yfinance helpers
│   │   ├── alpha_vantage.py        # Alpha Vantage facade
│   │   ├── alpha_vantage_common.py # Shared AV utilities, rate limit error
│   │   ├── alpha_vantage_stock.py  # AV stock data
│   │   ├── alpha_vantage_indicator.py  # AV technical indicators
│   │   ├── alpha_vantage_fundamentals.py  # AV fundamentals
│   │   ├── alpha_vantage_news.py   # AV news data
│   │   ├── google.py               # Google News fetcher
│   │   ├── googlenews_utils.py     # Google News helpers
│   │   ├── openai.py               # OpenAI-based data retrieval
│   │   ├── local.py                # Local/offline data (Finnhub, SimFin, Reddit)
│   │   ├── reddit_utils.py         # Reddit API helpers
│   │   ├── stockstats_utils.py     # stockstats indicator helpers
│   │   └── utils.py                # Shared data utilities
│   └── graph/                       # LangGraph orchestration
│       ├── __init__.py
│       ├── trading_graph.py         # TradingAgentsGraph: main orchestrator class
│       ├── setup.py                 # GraphSetup: builds the StateGraph workflow
│       ├── propagation.py           # Propagator: initial state & graph args
│       ├── conditional_logic.py     # ConditionalLogic: routing decisions
│       ├── reflection.py            # Reflector: post-trade memory updates
│       └── signal_processing.py     # SignalProcessor: extracts BUY/SELL/HOLD
├── assets/                          # Images for README
├── pyproject.toml                   # uv/pip project metadata
├── setup.py                         # setuptools configuration
├── requirements.txt                 # pip dependencies
└── uv.lock                          # uv lockfile
```

## Architecture & Agent Pipeline

The system follows a **5-stage sequential pipeline**, orchestrated as a LangGraph `StateGraph`:

```
1. Analyst Team       Analysts run sequentially, each using tools to fetch data
   Market Analyst ──► Social Analyst ──► News Analyst ──► Fundamentals Analyst
        │                   │                 │                    │
   (tools_market)     (tools_social)    (tools_news)     (tools_fundamentals)

2. Research Team      Bull & Bear researchers debate (configurable rounds)
   Bull Researcher ◄──► Bear Researcher ──► Research Manager (judge)

3. Trader             Synthesizes all reports into a trade proposal
   Trader ──► FINAL TRANSACTION PROPOSAL: BUY/HOLD/SELL

4. Risk Management    Three risk perspectives debate (configurable rounds)
   Risky Analyst ──► Safe Analyst ──► Neutral Analyst ──► (loop or judge)

5. Portfolio Manager   Risk Judge makes final decision
   Risk Judge ──► END (final_trade_decision)
```

### Key State Objects

- **`AgentState`** (`MessagesState` subclass): Top-level state flowing through the graph. Contains company info, trade date, all analyst reports, debate states, and final decision.
- **`InvestDebateState`**: Tracks bull/bear debate history, current response, judge decision, round count.
- **`RiskDebateState`**: Tracks risky/safe/neutral debate history, latest speaker, round count.

### Memory System

Uses **ChromaDB** (in-memory) with **OpenAI embeddings** for similarity-based memory retrieval. Each agent type has its own `FinancialSituationMemory` collection. The `reflect_and_remember()` method stores lessons learned from trade outcomes.

## Key Patterns & Conventions

### Agent Factory Pattern

All agents use a **factory function pattern** — a `create_*` function that takes an LLM (and optionally memory) and returns a **node function** compatible with LangGraph:

```python
def create_market_analyst(llm):
    def market_analyst_node(state):
        # ... use llm and state ...
        return {"messages": [result], "market_report": report}
    return market_analyst_node
```

### Tool Abstraction Layer

Tools are defined as `@tool`-decorated functions in `agents/utils/*_tools.py` that delegate to `route_to_vendor()` in `dataflows/interface.py`. This routes calls to the configured vendor (yfinance, Alpha Vantage, OpenAI, Google, local) with automatic fallback.

### Data Vendor Configuration

Configured in `default_config.py` via two levels:
- **Category-level** (`data_vendors`): Sets default vendor per data category
- **Tool-level** (`tool_vendors`): Overrides for specific tools

Categories: `core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`
Vendors: `yfinance`, `alpha_vantage`, `openai`, `google`, `local`

### LLM Provider Support

The framework supports multiple LLM providers configured via `llm_provider`:
- **OpenAI** (default): Uses `ChatOpenAI` — also used for OpenRouter and Ollama via `base_url`
- **Anthropic**: Uses `ChatAnthropic`
- **Google**: Uses `ChatGoogleGenerativeAI`
- **Custom** (`"custom"`): Any OpenAI-compatible API — uses `ChatOpenAI` with explicit `base_url` and `api_key`

Two LLM tiers: `deep_think_llm` (for judges/managers) and `quick_think_llm` (for analysts/debaters).

API keys can be set via:
1. **Config dict** (`api_key` field) — takes precedence
2. **Environment variables** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) — fallback

Embedding model/endpoint for memory can be configured independently via `embedding_model`, `embedding_base_url`, and `embedding_api_key`.

## Development Workflows

### Setup

```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
conda create -n tradingagents python=3.13
conda activate tradingagents
pip install -r requirements.txt
```

### Environment Variables

Required (set via export or `.env` file — see `.env.example`):
```bash
OPENAI_API_KEY=...          # Required for LLM calls (default provider)
ALPHA_VANTAGE_API_KEY=...   # Required for fundamental/news data (default vendor)
```

Optional depending on provider/vendor:
```bash
ANTHROPIC_API_KEY=...       # If using Anthropic provider
GOOGLE_API_KEY=...          # If using Google provider
TELEGRAM_BOT_TOKEN=...     # For Telegram bot integration
TELEGRAM_ALLOWED_USERS=... # Comma-separated user IDs (empty = allow all)
```

### Running

```bash
# Python API usage
python main.py

# Interactive CLI
python -m cli.main

# Telegram bot
export TELEGRAM_BOT_TOKEN=your_token_here
python -m telegram_bot.bot

# Telegram bot with custom LLM provider
python -m telegram_bot.bot --backend-url https://api.example.com/v1 --api-key sk-xxx --quick-model gpt-4o-mini --deep-model o4-mini
```

### Testing

There is no formal test suite. `test.py` is an ad-hoc script for manual testing of specific dataflow functions. Run it with:
```bash
python test.py
```

### Output & Logs

- **Eval results**: Written to `eval_results/{TICKER}/TradingAgentsStrategy_logs/`
- **CLI results**: Written to `results/{TICKER}/{DATE}/` with reports in `reports/` subdirectory
- **Data cache**: Stored in `tradingagents/dataflows/data_cache/`

## Configuration Reference

All settings live in `DEFAULT_CONFIG` dict (`tradingagents/default_config.py`):

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `"openai"` | LLM provider: openai, anthropic, google, ollama, openrouter, custom |
| `deep_think_llm` | `"o4-mini"` | Model for judges/managers (deep reasoning) |
| `quick_think_llm` | `"gpt-4o-mini"` | Model for analysts/debaters (fast inference) |
| `backend_url` | `"https://api.openai.com/v1"` | API endpoint URL |
| `api_key` | `None` | Explicit API key (falls back to env vars if None) |
| `embedding_model` | `None` | Embedding model name (auto-detected if None) |
| `embedding_base_url` | `None` | Separate embedding endpoint (uses `backend_url` if None) |
| `embedding_api_key` | `None` | Separate embedding API key (uses `api_key` if None) |
| `max_debate_rounds` | `1` | Rounds of bull/bear debate |
| `max_risk_discuss_rounds` | `1` | Rounds of risk management debate |
| `max_recur_limit` | `100` | LangGraph recursion limit |
| `data_vendors` | See below | Category-level vendor config |
| `tool_vendors` | `{}` | Tool-level vendor overrides |
| `results_dir` | `"./results"` | CLI results output directory (env: `TRADINGAGENTS_RESULTS_DIR`) |
| `data_cache_dir` | Auto-computed | Cache for fetched data |

## Code Conventions

### Style
- No linter/formatter configuration is present in the repo — no `.flake8`, `ruff.toml`, etc.
- Code uses a mix of styles; new contributions should aim for consistency with surrounding code.
- Type hints are used in some places (TypedDict states, tool annotations) but not universally enforced.

### Naming
- Agent factory functions: `create_{agent_name}()` returning a node function
- Tool functions: `get_{data_type}()` decorated with `@tool`
- Vendor implementations: `get_{data_type}_{vendor}()` or `get_{vendor}_{data_type}()`
- States: PascalCase TypedDict classes (`AgentState`, `InvestDebateState`, `RiskDebateState`)

### Imports
- `tradingagents/agents/__init__.py` uses wildcard exports — all factory functions are available via `from tradingagents.agents import *`
- `tradingagents/graph/__init__.py` explicitly exports graph component classes

### Known Issues
- `aggresive_debator.py` filename has a typo (should be "aggressive") — maintain current spelling for compatibility
- `default_config.py` has a hardcoded local path in `data_dir` — this only matters for `local` vendor mode
- `risk_manager.py:14` has a bug: `fundamentals_report` is assigned from `news_report` instead of `fundamentals_report`

## Adding New Components

### New Analyst Agent
1. Create `tradingagents/agents/analysts/new_analyst.py` with a `create_new_analyst(llm)` factory function
2. The factory should return a node function that accepts `state` and returns a dict with `messages` and a report field
3. Add the new report field to `AgentState` in `agent_states.py`
4. Export from `tradingagents/agents/__init__.py`
5. Add tool nodes in `TradingAgentsGraph._create_tool_nodes()` if the analyst needs data tools
6. Add conditional logic in `ConditionalLogic` for tool routing
7. Register in `GraphSetup.setup_graph()` and update CLI's `AnalystType` enum

### New Data Vendor
1. Create `tradingagents/dataflows/new_vendor.py` implementing the required function signatures
2. Register implementations in `VENDOR_METHODS` dict in `dataflows/interface.py`
3. Add vendor name to `VENDOR_LIST` in `interface.py`
4. The `route_to_vendor()` function will automatically handle fallback

### New LLM Provider
1. Add the provider's LangChain integration package to `requirements.txt` and `pyproject.toml`
2. Add initialization logic in `TradingAgentsGraph.__init__()` under the provider check
3. Add provider options to `cli/utils.py` in `select_llm_provider()`, `select_shallow_thinking_agent()`, and `select_deep_thinking_agent()`

## Dependencies

Key framework dependencies:
- **langgraph** / **langchain-openai** / **langchain-anthropic** / **langchain-google-genai**: Agent orchestration and LLM integration
- **chromadb**: Vector store for agent memory
- **yfinance**: Default stock data provider
- **rich** / **typer** / **questionary**: CLI interface
- **pandas** / **stockstats**: Data manipulation and technical indicators
- **praw** / **feedparser**: Reddit/RSS data sources
- **requests**: HTTP calls for Alpha Vantage and other APIs
- **python-telegram-bot**: Telegram bot integration

## Important Notes for AI Assistants

- This is a **research framework**, not production trading software. It is not intended as financial advice.
- The framework makes **many LLM API calls** per analysis run. Use cheaper models (e.g., `gpt-4o-mini`, `gpt-4.1-nano`) for testing.
- Never commit `.env` files or API keys. Use environment variables or `.env` (gitignored).
- The `data_dir` in `default_config.py` contains a hardcoded development path — it only affects the `local` data vendor and can be safely ignored for non-local setups.
- When modifying agent prompts, preserve the `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**` format as it's parsed by downstream components.
- The graph uses LangGraph's `stream_mode="values"` — each chunk contains the full state, not deltas.
