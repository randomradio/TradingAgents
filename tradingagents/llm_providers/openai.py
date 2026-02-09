"""
OpenAI LLM Provider.

Implements direct API calls to OpenAI's API.
Compatible with OpenAI-style APIs (OpenRouter, Ollama, etc.).
"""

import os
from typing import List, Dict, Any, Optional

import requests

from .base import BaseLLMProvider, LLMResponse, ToolCall


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider.

    Usage:
        provider = OpenAIProvider(
            model="gpt-4o-mini",
            api_key="your-api-key",
            base_url="https://api.openai.com/v1"  # optional
        )

        response = provider.invoke("Your prompt")
        print(response.content)
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: OpenAI API key
            base_url: Base URL (for custom endpoints like OpenRouter)
            **kwargs: Additional parameters
        """
        super().__init__(model, api_key, base_url or self.DEFAULT_BASE_URL, **kwargs)

    def _make_api_request(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Make API request to OpenAI."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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

        # Make request
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        # Parse response
        choice = result["choices"][0]
        message = choice["message"]

        # Extract content
        content = message.get("content", "")

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
            tool_calls=tool_calls,
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
                "provider": "openai",
            }
        )
