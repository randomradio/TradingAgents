"""
LLM Provider Abstraction Layer

This module provides a unified interface for multiple LLM providers,
bypassing LangChain's abstraction to support provider-specific features
like DeepSeek's reasoning_content.

Usage:
    from tradingagents.llm_providers import create_llm_provider

    provider = create_llm_provider(
        provider="deepseek",
        model="deepseek-reasoner",
        api_key="your-key",
        base_url="https://api.deepseek.com"
    )

    # Simple invoke
    response = provider.invoke("Your prompt here")
    print(response.content)
    if response.reasoning_content:
        print(response.reasoning_content)

    # With tools
    bound_llm = provider.bind_tools(tools)
    response = bound_llm.invoke(messages)
"""

from .base import BaseLLMProvider, LLMResponse, ToolCall
from .deepseek import DeepSeekProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .factory import create_llm_provider

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "ToolCall",
    "DeepSeekProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "create_llm_provider",
]
