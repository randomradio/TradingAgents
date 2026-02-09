"""
DeepSeek LLM Provider with reasoning_content support.

Implements direct API calls to DeepSeek's API, supporting:
- Thinking mode with reasoning_content
- Tool/function calling
- Multi-turn conversations with proper reasoning_content handling
"""

import os
import json
from typing import List, Dict, Any, Optional

import requests

from .base import BaseLLMProvider, LLMResponse, ToolCall


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek API provider with thinking mode support.

    Usage:
        provider = DeepSeekProvider(
            model="deepseek-reasoner",
            api_key="your-api-key",
            base_url="https://api.deepseek.com"
        )

        response = provider.invoke("Your prompt")
        print(response.content)  # Final answer
        print(response.reasoning_content)  # Chain of thought
    """

    # Default base URL for DeepSeek
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    # Models that support thinking mode
    THINKING_MODELS = {"deepseek-reasoner", "deepseek-chat"}

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        enable_thinking: bool = None,
        **kwargs
    ):
        """
        Initialize DeepSeek provider.

        Args:
            model: Model name (e.g., "deepseek-reasoner", "deepseek-chat")
            api_key: DeepSeek API key
            base_url: Base URL (defaults to https://api.deepseek.com)
            enable_thinking: Force enable/disable thinking mode.
                           None = auto-detect based on model
            **kwargs: Additional parameters (max_tokens, etc.)
        """
        # Support OpenRouter URLs
        if base_url and "openrouter" in base_url.lower():
            # OpenRouter format: deepseek/deepseek-chat-v3-0324:free
            if model.startswith("deepseek/"):
                self._is_openrouter = True
        else:
            self._is_openrouter = False

        super().__init__(model, api_key, base_url or self.DEFAULT_BASE_URL, **kwargs)

        # Auto-detect thinking mode support
        self.enable_thinking = enable_thinking
        if self.enable_thinking is None:
            # Enable for models that support it
            self.enable_thinking = any(
                m in model.lower() for m in ["reasoner", "deepseek-chat", "deepseek-v3"]
            )

    def _make_api_request(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Make API request to DeepSeek.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            **kwargs: Additional parameters

        Returns:
            LLMResponse with reasoning_content if available
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Inject reasoning_content into assistant messages with tool calls
        # This is required by DeepSeek's API when using thinking mode with tools
        messages = self._inject_reasoning_content(messages)

        # Build request body
        body = {
            "model": self.model,
            "messages": messages,
            **self.extra_params,
            **kwargs,
        }

        # Add tools if present
        if tools:
            body["tools"] = tools

        # Enable thinking mode for DeepSeek
        if self.enable_thinking and not self._is_openrouter:
            # Use extra_body format for OpenAI SDK compatibility
            # Note: Direct API uses "thinking" field
            body["thinking"] = {"type": "enabled"}

        # Make request
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=120,  # DeepSeek thinking mode can take longer
        )

        response.raise_for_status()
        result = response.json()

        # Parse response
        choice = result["choices"][0]
        message = choice["message"]

        # Extract content
        content = message.get("content", "")

        # Extract reasoning_content (DeepSeek-specific)
        reasoning_content = message.get("reasoning_content")

        # Cache reasoning_content for next tool call round
        if reasoning_content:
            self._reasoning_content_cache["last"] = reasoning_content

        # Extract tool calls
        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ))

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
                "provider": "deepseek",
            }
        )

    def _inject_reasoning_content(self, messages: List[Dict]) -> List[Dict]:
        """
        Inject cached reasoning_content into assistant messages that need it.

        When using tools with thinking mode, DeepSeek requires that assistant
        messages with tool_calls include the reasoning_content field.

        Args:
            messages: List of message dicts

        Returns:
            Messages with reasoning_content injected
        """
        if not self.enable_thinking or self._is_openrouter:
            return messages

        # Check if we have cached reasoning_content to inject
        cached_reasoning = self._reasoning_content_cache.get("last")
        if not cached_reasoning:
            return messages

        # Find assistant messages with tool_calls but no reasoning_content
        # and inject the cached reasoning_content
        modified_messages = []
        for msg in messages:
            modified_msg = msg.copy() if isinstance(msg, dict) else msg

            if (
                isinstance(modified_msg, dict)
                and modified_msg.get("role") == "assistant"
                and "tool_calls" in modified_msg
                and "reasoning_content" not in modified_msg
            ):
                # Inject cached reasoning_content
                modified_msg["reasoning_content"] = cached_reasoning

            modified_messages.append(modified_msg)

        return modified_messages

    def clear_reasoning_content(self, messages: List[Any]) -> List[Any]:
        """
        Clear reasoning_content from message history.

        Recommended between different questions to save bandwidth.

        Args:
            messages: List of message objects or dicts

        Returns:
            Messages with reasoning_content cleared
        """
        # Clear the cache
        self._reasoning_content_cache.clear()

        for msg in messages:
            if isinstance(msg, dict):
                msg.pop("reasoning_content", None)
            elif hasattr(msg, "reasoning_content"):
                msg.reasoning_content = None

        return messages

    def bind_tools(self, tools: List) -> "BoundLLM":
        """
        Bind tools with special handling for reasoning_content.

        When using tools with DeepSeek thinking mode, reasoning_content
        must be preserved between tool calls within the same question.
        """
        return BoundLLM(self, tools)


class BoundLLM:
    """Bound LLM for DeepSeek with reasoning_content preservation."""

    def __init__(self, provider: DeepSeekProvider, tools: List):
        self.provider = provider
        self.tools = tools

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Invoke with automatic reasoning_content handling."""
        # Clear reasoning cache between different conversations
        # Detect new conversation by checking if this is a fresh user message
        # (i.e., the last message is from user, not tool)
        if messages and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                is_fresh_user = (
                    last_msg.get("role") == "user" and
                    not any(m.get("role") == "tool" for m in messages if isinstance(m, dict))
                )
                if is_fresh_user:
                    # Clear cache for new conversation
                    self.provider._reasoning_content_cache.clear()

        response = self.provider.invoke(messages, tools=self.tools, **kwargs)
        return response

    def __getattr__(self, name):
        """Proxy other attributes to the underlying provider."""
        return getattr(self.provider, name)
