"""Shared test configuration and fixtures.

Pre-mocks heavy third-party dependencies that cannot be installed in the
test environment (yfinance needs curl_cffi C extensions, akshare needs
specific wheels, etc.).  This file is loaded by pytest before any test
modules, so the mocks are in place before real imports happen.
"""

import sys
import types
from unittest.mock import MagicMock


def _mock_module(name: str) -> MagicMock:
    """Create and register a MagicMock module, returning it."""
    if name in sys.modules:
        mod = sys.modules[name]
        if isinstance(mod, MagicMock):
            return mod
        return mod  # already real — leave alone
    mock = MagicMock()
    mock.__name__ = name
    mock.__package__ = name
    mock.__path__ = []
    mock.__file__ = f"<mocked {name}>"
    sys.modules[name] = mock
    return mock


# ---------------------------------------------------------------------------
# Packages that are unavailable or too heavy for the test sandbox.
# Order matters: parent packages must be mocked before children.
# ---------------------------------------------------------------------------

_PACKAGES_TO_MOCK = [
    # yfinance and its transitive deps
    "curl_cffi",
    "curl_cffi.requests",
    "yfinance",
    "yfinance.search",
    "yfinance.data",
    "yfinance.utils",
    # akshare
    "akshare",
    # feedparser / sgmllib (feedparser build fails)
    "feedparser",
    "sgmllib",
    # simfin / finnhub
    "simfin",
    "finnhub",
    # telegram (python-telegram-bot needs cryptography C ext)
    "dotenv",
    "telegram",
    "telegram.ext",
    # langchain providers that may not be installed
    "langchain_anthropic",
    "langchain_google_genai",
]

for _pkg in _PACKAGES_TO_MOCK:
    _mock_module(_pkg)
