"""
Google (Gemini) LLM Provider.

Implements direct API calls to Google's Gemini API.
"""

from typing import List, Dict, Any, Optional

import requests

from .base import BaseLLMProvider, LLMResponse, ToolCall


class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini API provider.

    Usage:
        provider = GoogleProvider(
            model="gemini-2.0-flash",
            api_key="your-api-key"
        )

        response = provider.invoke("Your prompt")
        print(response.content)
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google provider.

        Args:
            model: Model name (e.g., "gemini-2.0-flash")
            api_key: Google API key
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
        """Make API request to Google Gemini."""
        # Build request body for Gemini API
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        body = {
            "contents": contents,
            **self.extra_params,
            **kwargs,
        }

        if system_instruction:
            body["system_instruction"] = {"parts": [{"text": system_instruction}]}

        # Add tools if present
        if tools:
            body["tools"] = [self._convert_tool_to_google_format(t) for t in tools]

        # Make request
        # Note: API key goes in URL for Google
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        response = requests.post(
            url,
            json={"contents": contents, **body},
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        # Extract content
        content = ""
        tool_calls = []

        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(ToolCall(
                        id=f"call_{len(tool_calls)}",
                        name=fc["name"],
                        arguments=json.dumps(fc.get("args", {})),
                    ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            metadata={
                "model": result.get("model"),
                "usage": result.get("usageMetadata"),
                "provider": "google",
            }
        )

    def _convert_tool_to_google_format(self, tool: Dict) -> Dict:
        """Convert OpenAI-style tool format to Google format."""
        if "function" in tool:
            func = tool["function"]
            return {
                "function_declarations": [
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                ]
            }
        return {"function_declarations": [tool]}

    def _convert_tool_format(self, tool) -> Dict[str, Any]:
        """Convert LangChain tool to Google format."""
        if hasattr(tool, 'name'):
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.schema() if hasattr(tool, 'args_schema') else {},
            }

        if isinstance(tool, dict):
            if "function" in tool:
                return self._convert_tool_to_google_format(tool)
            return {"function_declarations": [tool]}

        raise ValueError(f"Unsupported tool format: {type(tool)}")


import json
