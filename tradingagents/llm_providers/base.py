"""
Base classes for LLM Provider abstraction.

Provides compatibility with LangChain's LLM interface while using
direct API calls under the hood.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Try to import LangChain message types for compatibility
try:
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        SystemMessage,
    )
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # Create dummy classes for type hints
    BaseMessage = object
    HumanMessage = object
    AIMessage = object
    ToolMessage = object
    SystemMessage = object


@dataclass
class ToolCall:
    """Represents a tool call from an LLM response."""

    id: str
    name: str
    arguments: str  # JSON string of arguments

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LangChain-compatible format."""
        return {
            "id": self.id,
            "name": self.name,
            "args": self.arguments,  # LangChain uses "args"
        }


@dataclass
class LLMResponse:
    """
    Response from an LLM provider.

    Compatible with LangChain's AIMessage interface.
    """

    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_content: Optional[str] = None  # For DeepSeek thinking mode
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def response_metadata(self) -> Dict[str, Any]:
        """LangChain compatibility."""
        return self.metadata

    def __repr__(self) -> str:
        reasoning_info = f" (reasoning: {len(self.reasoning_content)} chars)" if self.reasoning_content else ""
        tools_info = f" ({len(self.tool_calls)} tools)" if self.tool_calls else ""
        return f"LLMResponse{reasoning_info}{tools_info}: {self.content[:100]}..."


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides a unified interface that's compatible with LangChain's
    ChatOpenAI/ChatAnthropic usage patterns, while using direct API calls.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLM provider.

        Args:
            model: Model name/identifier
            api_key: API key for the provider
            base_url: Base URL for API (optional, for custom endpoints)
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = kwargs
        # Store reasoning content for tracking
        self._reasoning_content_cache: Dict[str, str] = {}

    def _convert_messages_to_dict(
        self,
        messages: Union[str, List[Union[Dict, BaseMessage]]]
    ) -> List[Dict[str, Any]]:
        """
        Convert various message formats to standard dict format.

        Handles:
        - String prompt (converted to user message)
        - List of dicts (returned as-is)
        - LangChain message objects (converted to dicts)

        Args:
            messages: Messages in various formats

        Returns:
            List of message dicts with role, content, and optional fields
        """
        # Convert string prompt to message format
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        if not isinstance(messages, list):
            raise ValueError(f"Messages must be str or list, got {type(messages)}")

        result = []
        for msg in messages:
            if isinstance(msg, dict):
                # Already a dict, just ensure it has required fields
                result.append(msg)
            elif HAS_LANGCHAIN and isinstance(msg, BaseMessage):
                # Convert LangChain message to dict
                msg_dict = self._langchain_message_to_dict(msg)
                result.append(msg_dict)
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        return result

    def _langchain_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """
        Convert a LangChain BaseMessage to dict format.

        Args:
            message: LangChain message object

        Returns:
            Message dict with role, content, and optional fields
        """
        if not HAS_LANGCHAIN:
            raise ValueError("LangChain messages require langchain-core to be installed")

        msg_dict = {"content": message.content}

        # Determine role from message type
        if isinstance(message, HumanMessage):
            msg_dict["role"] = "user"
        elif isinstance(message, AIMessage):
            msg_dict["role"] = "assistant"
            # Preserve tool_calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"],
                        }
                    }
                    for i, tc in enumerate(message.tool_calls)
                ]
        elif isinstance(message, ToolMessage):
            msg_dict["role"] = "tool"
            msg_dict["tool_call_id"] = message.tool_call_id
        elif isinstance(message, SystemMessage):
            msg_dict["role"] = "system"
        else:
            # Fallback for unknown message types
            msg_dict["role"] = "user"

        return msg_dict

    @abstractmethod
    def _make_api_request(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Make the actual API request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Additional request parameters

        Returns:
            LLMResponse object
        """
        pass

    def invoke(
        self,
        prompt: Union[str, List],
        tools: Optional[List] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Invoke the LLM with a prompt.

        Compatible with LangChain's LLM.invoke() interface.

        Args:
            prompt: String prompt or list of messages (dicts or LangChain objects)
            tools: Optional list of LangChain tool objects
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        # Convert messages to standard dict format
        messages = self._convert_messages_to_dict(prompt)

        # Convert LangChain tools to provider format
        tool_dicts = None
        if tools:
            tool_dicts = [self._convert_tool_format(t) for t in tools]

        return self._make_api_request(messages, tool_dicts, **kwargs)

    def bind_tools(self, tools: List) -> "BoundLLM":
        """
        Bind tools to this LLM, returning a new LLM instance.

        Compatible with LangChain's bind_tools() interface.

        Args:
            tools: List of LangChain tool objects

        Returns:
            BoundLLM instance
        """
        return BoundLLM(self, tools)

    def _convert_tool_format(self, tool) -> Dict[str, Any]:
        """
        Convert a LangChain tool to provider-specific format.

        Default OpenAI-style format. Override in provider subclasses if needed.

        Args:
            tool: LangChain tool object

        Returns:
            Tool definition dict
        """
        # Handle LangChain tool objects
        if hasattr(tool, 'name'):
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.schema() if hasattr(tool, 'args_schema') else {},
                }
            }

        # Handle dict format
        if isinstance(tool, dict):
            return tool

        raise ValueError(f"Unsupported tool format: {type(tool)}")


class BoundLLM:
    """
    An LLM with pre-bound tools.

    Compatible with LangChain's bind_tools() return value.
    """

    def __init__(self, provider: BaseLLMProvider, tools: List):
        self.provider = provider
        self.tools = tools

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Invoke with bound tools."""
        return self.provider.invoke(messages, tools=self.tools, **kwargs)

    def __getattr__(self, name):
        """Proxy other attributes to the underlying provider."""
        return getattr(self.provider, name)
