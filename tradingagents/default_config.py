import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings (all overridable via environment variables)
    "llm_provider": os.getenv("TRADINGAGENTS_LLM_PROVIDER", "openai"),  # openai, anthropic, google, ollama, openrouter, custom
    "deep_think_llm": os.getenv("TRADINGAGENTS_DEEP_THINK_LLM", "o4-mini"),
    "quick_think_llm": os.getenv("TRADINGAGENTS_QUICK_THINK_LLM", "gpt-4o-mini"),
    "backend_url": os.getenv("TRADINGAGENTS_BASE_URL", "https://api.openai.com/v1"),
    "api_key": os.getenv("TRADINGAGENTS_API_KEY"),  # Explicit API key; if None, falls back to env vars (OPENAI_API_KEY, etc.)
    # Embedding settings (for agent memory)
    "embedding_model": os.getenv("TRADINGAGENTS_EMBEDDING_MODEL"),  # None = auto-select based on provider
    "embedding_base_url": os.getenv("TRADINGAGENTS_EMBEDDING_BASE_URL"),  # None = same as backend_url
    "embedding_api_key": os.getenv("TRADINGAGENTS_EMBEDDING_API_KEY"),  # None = same as api_key
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage", # Options: openai, alpha_vantage, local
        "news_data": "alpha_vantage",        # Options: openai, alpha_vantage, google, local
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
        # Example: "get_news": "openai",               # Override category default
    },
}
