import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
import pytest

from langchain_asynctools import AsyncTools


class FakeChatModelWithTools(GenericFakeChatModel):
    """GenericFakeChatModel that supports bind_tools (ignores bound tools)."""

    def bind_tools(self, tools: list[BaseTool | dict | Any], **kwargs: Any) -> "FakeChatModelWithTools": #type: ignore
        return self


@tool
def get_greeting(name: str) -> str:
    """Returns a greeting for the given name."""
    return f"Hello, {name}!"


@tool
async def slow_computation(n: int) -> int:
    """Computes n squared slowly."""
    await asyncio.sleep(1)
    return n * n

@pytest.mark.asyncio
async def test_sync_and_async_tool_calls():
    middleware = AsyncTools()

    model = FakeChatModelWithTools(messages=iter([
        AIMessage(content="", tool_calls=[
            ToolCall(name="get_greeting", args={"name": "World"}, id="call_sync_1"),
        ]),
        AIMessage(content="", tool_calls=[
            ToolCall(name="slow_computation", args={"n": 5}, id="call_async_1"),
        ]),
        AIMessage(content="", tool_calls=[
            ToolCall(name="query_tool_output", args={"job_id": "1"}, id="call_query_1"),
        ]),
        AIMessage(content="", tool_calls=[
            ToolCall(name="await_tool_output", args={"job_id": "1", "wait_time": 2}, id="call_await_1"),
        ]),
        AIMessage(content="All done."),
    ]))

    agent = create_agent(
        model=model,
        tools=[get_greeting, slow_computation],
        middleware=[middleware],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Greet and compute")]})

    tool_messages: list[ToolMessage] = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    # Sync tool should execute synchronously and return its real result
    sync_result = next(m for m in tool_messages if m.tool_call_id == "call_sync_1")
    assert "Hello, World!" in sync_result.content

    # Async tool should return immediately with a job ID placeholder
    async_result = next(m for m in tool_messages if m.tool_call_id == "call_async_1")
    assert "job ID" in async_result.content

    async_query_result = next(m for m in tool_messages if m.tool_call_id == "call_query_1")
    assert "still processing" in async_query_result.content

    async_await_query_result = next(m for m in tool_messages if m.tool_call_id == "call_await_1")
    assert "25" in async_await_query_result.content
    assert len(result.get("jobs", {})) == 0  # Job should be cleaned up after completion  
