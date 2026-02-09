#!/usr/bin/env python3
"""
Test DeepSeek reasoning_content injection specifically.

This tests the logic for injecting reasoning_content into assistant messages
that have tool_calls, which is required by DeepSeek's API.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_reasoning_content_injection():
    """Test that reasoning_content is properly injected into assistant messages."""
    from tradingagents.llm_providers.deepseek import DeepSeekProvider

    print("Testing reasoning_content injection...")

    # Create a DeepSeek provider (no API key needed for this test)
    provider = DeepSeekProvider(
        model="deepseek-chat",
        api_key="test-key",
        enable_thinking=True,
    )

    # Simulate messages that LangChain would pass
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Sunny, 70°F"
        },
        # This is the message that should get reasoning_content injected
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_forecast", "arguments": '{"location": "NYC"}'}
                }
            ]
        },
    ]

    # Simulate having cached reasoning_content
    provider._reasoning_content_cache["last"] = "I need to get the weather first, then get the forecast."

    # Test the injection logic
    result = provider._inject_reasoning_content(messages)

    # Verify that the last assistant message got reasoning_content
    last_assistant = None
    for msg in result:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            last_assistant = msg

    if last_assistant and "reasoning_content" in last_assistant:
        print("✓ reasoning_content was injected into assistant message")
        print(f"  Injected content: {last_assistant['reasoning_content']}")
    else:
        print("✗ reasoning_content was NOT injected")
        print(f"  Last assistant message: {last_assistant}")

def test_message_conversion():
    """Test that LangChain messages are properly converted to dicts."""
    from tradingagents.llm_providers.deepseek import DeepSeekProvider

    print("\nTesting LangChain message conversion...")

    try:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        provider = DeepSeekProvider(
            model="deepseek-chat",
            api_key="test-key",
        )

        # Create LangChain messages
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "get_weather",
                    "args": {"location": "NYC"},
                }]
            ),
            ToolMessage(content="Sunny, 70°F", tool_call_id="call_1"),
        ]

        # Convert to dict format
        result = provider._convert_messages_to_dict(messages)

        print(f"✓ Converted {len(result)} messages")
        for i, msg in enumerate(result):
            print(f"  Message {i}: role={msg.get('role')}, has_tool_calls={'tool_calls' in msg}")

    except ImportError:
        print("⊘ Skipping (langchain-core not available)")

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek reasoning_content Test Suite")
    print("=" * 60)

    test_reasoning_content_injection()
    test_message_conversion()

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)
