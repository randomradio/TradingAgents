"""
Factory function for creating LLM providers.

Provides a unified interface for creating providers based on configuration.
"""

import os
from typing import Optional

from .base import BaseLLMProvider
from .deepseek import DeepSeekProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider


# Provider name to class mapping
PROVIDER_CLASSES = {
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "ollama": OpenAIProvider,  # Ollama uses OpenAI-compatible API
    "openrouter": OpenAIProvider,  # OpenRouter uses OpenAI-compatible API
    "custom": OpenAIProvider,  # Custom endpoints use OpenAI-compatible API
}


def create_llm_provider(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider: Provider name (deepseek, openai, anthropic, google, ollama, openrouter, custom)
        model: Model name/identifier
        api_key: API key (if None, will try to get from environment)
        base_url: Base URL for API (optional)
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseLLMProvider instance

    Raises:
        ValueError: If provider is not supported

    Examples:
        >>> provider = create_llm_provider(
        ...     provider="deepseek",
        ...     model="deepseek-reasoner",
        ...     api_key="your-key"
        ... )

        >>> provider = create_llm_provider(
        ...     provider="openrouter",
        ...     model="deepseek/deepseek-chat-v3-0324:free",
        ...     api_key="your-key",
        ...     base_url="https://openrouter.ai/api/v1"
        ... )
    """
    provider = provider.lower()

    if provider not in PROVIDER_CLASSES:
        supported = ", ".join(PROVIDER_CLASSES.keys())
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {supported}"
        )

    provider_class = PROVIDER_CLASSES[provider]

    # Get API key from environment if not provided
    if api_key is None:
        api_key = _get_default_api_key(provider)

    if not api_key:
        raise ValueError(
            f"API key required for {provider}. "
            f"Set api_key parameter or {get_env_var_name(provider)} environment variable."
        )

    # Create provider instance
    return provider_class(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def _get_default_api_key(provider: str) -> Optional[str]:
    """Get API key from environment variable."""
    env_var = get_env_var_name(provider)
    return os.environ.get(env_var)


def get_env_var_name(provider: str) -> str:
    """Get the environment variable name for a provider's API key."""
    env_var_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "ollama": None,  # Ollama typically doesn't require an API key
        "openrouter": "OPENROUTER_API_KEY",
        "custom": None,  # Custom uses whatever is configured
    }
    return env_var_map.get(provider.lower())
