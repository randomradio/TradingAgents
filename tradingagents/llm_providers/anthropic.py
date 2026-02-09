"""
Anthropic (Claude) LLM Provider.

Implements direct API calls to Anthropic's Claude API.
"""

from typing import List, Dict, Any, Optional

import requests

from .base import BaseLLMProvider, LLMResponse, ToolCall


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.

    Usage:
        provider = AnthropicProvider(
            model="claude-sonnet-4-5-20250929",
            api_key="your-api-key"
        )

        response = provider.invoke("Your prompt")
        print(response.content)
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-sonnet-4-5-20250929")
            api_key: Anthropic API key
            base_url: Base URL (optional)
            **kwargs: Additional parameters
        """
        super().__init__(model, api_key, base_url or self.DEFAULT_BASE_URL, **kwargs)

    def _make_api_request(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Make API request to Anthropic."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Convert messages to Anthropic format
        # Anthropic requires system message to be separate
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(msg)

        # Build request body
        body = {
            "model": self.model,
            "messages": anthropic_messages,
            **self.extra_params,
            **kwargs,
        }

        if system_message:
            body["system"] = system_message

        # Add tools if present - convert to Anthropic format
        if tools:
            body["tools"] = [self._convert_tool_to_anthropic_format(t) for t in tools]

        # Make request
        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=body,
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        # Extract content
        content = ""
        for block in result.get("content", []):
            if block["type"] == "text":
                content += block["text"]

        # Extract tool calls
        tool_calls = []
        for block in result.get("content", []):
            if block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=json.dumps(block["input"]),
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
                "provider": "anthropic",
            }
        )

    def _convert_tool_to_anthropic_format(self, tool: Dict) -> Dict:
        """Convert OpenAI-style tool format to Anthropic format."""
        if "function" in tool:
            func = tool["function"]
            return {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            }
        return tool

    def _convert_tool_format(self, tool) -> Dict[str, Any]:
        """Convert LangChain tool to Anthropic format."""
        if hasattr(tool, 'name'):
            return {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.args_schema.schema() if hasattr(tool, 'args_schema') else {},
            }

        if isinstance(tool, dict):
            return self._convert_tool_to_anthropic_format(tool)

        raise ValueError(f"Unsupported tool format: {type(tool)}")


# Need json for tool arguments
import json
