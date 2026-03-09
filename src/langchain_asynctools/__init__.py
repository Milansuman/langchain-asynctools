"""
Middleware for non blocking tool call execution in Langchain.
"""
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

import asyncio

_ASYNC_TOOLS_SYSTEM_PROMPT = """\
Some tools in this agent run asynchronously in the background. When a tool call \
returns a job ID, use the following tools to retrieve results:
- `query_tool_output(job_id)`: Check whether the result is ready and return it immediately.
- `await_tool_output(job_id, wait_time)`: Wait up to `wait_time` seconds for the result \
before returning.
"""

class AsyncTools(AgentMiddleware):
    """
    Middleware for non blocking tool call execution in Langchain.
    """

    def __init__(self):
        super().__init__()
        self._jobs: dict[int, asyncio.Task] = {}

    def _make_async_tools(self) -> list[StructuredTool]:
        jobs = self._jobs

        def query_tool_output(job_id: str) -> str:
            """Returns the current output of a background tool call, or a status message if it is still running."""
            task = jobs.get(int(job_id))
            if task is None:
                return f"No job found with ID: {job_id}"
            if not task.done():
                return f"Job {job_id} is still processing."
            try:
                result = task.result()
                return result.text if isinstance(result, ToolMessage) else str(result)
            except Exception as e:
                return f"Job {job_id} failed with error: {e}"

        async def await_tool_output(job_id: str, wait_time: int) -> str:
            """Waits up to wait_time seconds for the output of a background tool call."""
            task = jobs.get(int(job_id))
            if task is None:
                return f"No job found with ID: {job_id}"
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=wait_time)
                return result.text if isinstance(result, ToolMessage) else str(result)
            except asyncio.TimeoutError:
                return f"Job {job_id} did not complete within {wait_time} seconds."
            except Exception as e:
                return f"Job {job_id} failed with error: {e}"

        return [
            StructuredTool.from_function(func=query_tool_output),
            StructuredTool.from_function(coroutine=await_tool_output),
        ]

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        async_tools = self._make_async_tools()
        existing = request.system_message.text if request.system_message else ""
        new_content = (existing + "\n\n" + _ASYNC_TOOLS_SYSTEM_PROMPT).strip()
        return handler(request.override(
            tools=list(request.tools) + async_tools,
            system_message=SystemMessage(content=new_content),
        ))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        async_tools = self._make_async_tools()
        existing = request.system_message.text if request.system_message else ""
        new_content = (existing + "\n\n" + _ASYNC_TOOLS_SYSTEM_PROMPT).strip()
        return await handler(request.override(
            tools=list(request.tools) + async_tools,
            system_message=SystemMessage(content=new_content),
        ))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
        """Sync version: delegates directly to the handler."""
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]) -> ToolMessage | Command[Any]:
        """
        Async version: schedules the tool call as a background task and returns immediately.

        Args:
            request: The tool call request.
            handler: The original async tool call handler.

        Returns:
            A ToolMessage indicating that the tool call is being processed.
        """
        job_id = id(request)
        self._jobs[job_id] = asyncio.ensure_future(handler(request))

        try:
            result = await asyncio.wait_for(asyncio.shield(self._jobs[job_id]), timeout=0.5)
            del self._jobs[job_id]
            return result if isinstance(result, ToolMessage) else ToolMessage(content=str(result), tool_call_id=request.tool_call["id"])
        except asyncio.TimeoutError:
            pass

        return ToolMessage(content=f"Tool call is being processed with job ID: {job_id}", tool_call_id=request.tool_call["id"])
    
    
    