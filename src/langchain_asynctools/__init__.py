"""
Middleware for non blocking tool call execution in Langchain.
"""
import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated, Any
from typing_extensions import NotRequired

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentMiddleware, AgentState, OmitFromInput
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain.tools import InjectedState, tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_ASYNC_TOOLS_SYSTEM_PROMPT = """\
Some tools in this agent run asynchronously in the background. When a tool call \
returns a job ID, use the following tools to retrieve results:
- `query_tool_output(job_id)`: Check whether the result is ready and return it immediately.
- `await_tool_output(job_id, wait_time)`: Wait up to `wait_time` seconds for the result \
before returning.
"""


class AsyncToolsState(AgentState[Any]):
    """State schema for async tool middleware internals."""

    jobs: Annotated[NotRequired[dict[str, asyncio.Task[Any]]], OmitFromInput]
    next_job_id: Annotated[NotRequired[int], OmitFromInput]


_INTERNAL_ASYNC_TOOL_NAMES = {"query_tool_output", "await_tool_output"}


class AsyncTools(AgentMiddleware[AsyncToolsState]):
    """
    Middleware for non blocking tool call execution in Langchain.
    """

    state_schema = AsyncToolsState

    def __init__(self):
        super().__init__()
        self.tools = self._make_async_tools()

    def _make_async_tools(self) -> list[BaseTool]:
        def _get_jobs(state: AsyncToolsState) -> dict[str, asyncio.Task[Any]]:
            jobs = state.get("jobs")
            if jobs is None:
                jobs = {}
                state["jobs"] = jobs
            return jobs

        @tool
        def query_tool_output(
            job_id: str,
            state: Annotated[AsyncToolsState, InjectedState],
        ) -> str:
            """Returns the current output of a background tool call, or a status message if it is still running."""
            jobs = _get_jobs(state)
            task = jobs.get(job_id)
            if task is None:
                return f"No job found with ID: {job_id}"
            if not task.done():
                return f"Job {job_id} is still processing."
            try:
                jobs.pop(job_id, None)
                result = task.result()
                return result.text if isinstance(result, ToolMessage) else str(result)
            except Exception as e:
                return f"Job {job_id} failed with error: {e}"

        @tool
        async def await_tool_output(
            job_id: str,
            wait_time: int,
            state: Annotated[AsyncToolsState, InjectedState],
        ) -> str:
            """Waits up to wait_time seconds for the output of a background tool call."""
            jobs = _get_jobs(state)
            task = jobs.get(job_id)
            if task is None:
                return f"No job found with ID: {job_id}"
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=wait_time)
                jobs.pop(job_id, None)
                return result.text if isinstance(result, ToolMessage) else str(result)
            except asyncio.TimeoutError:
                return f"Job {job_id} did not complete within {wait_time} seconds."
            except Exception as e:
                return f"Job {job_id} failed with error: {e}"

        return [
            query_tool_output,
            await_tool_output
        ]

    def before_agent(self, state: AsyncToolsState, runtime: Any) -> dict[str, Any] | None:
        updates: dict[str, Any] = {}
        if "jobs" not in state:
            updates["jobs"] = {}
        if "next_job_id" not in state:
            updates["next_job_id"] = 1
        return updates or None

    async def abefore_agent(self, state: AsyncToolsState, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        existing = request.system_message.text if request.system_message else ""
        new_content = (existing + "\n\n" + _ASYNC_TOOLS_SYSTEM_PROMPT).strip()
        return handler(request.override(
            system_message=SystemMessage(content=new_content),
        ))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        existing = request.system_message.text if request.system_message else ""
        new_content = (existing + "\n\n" + _ASYNC_TOOLS_SYSTEM_PROMPT).strip()
        return await handler(request.override(
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
        # Let middleware-internal async tools execute normally.
        if request.tool_call["name"] in _INTERNAL_ASYNC_TOOL_NAMES:
            return await handler(request)

        state = request.state
        jobs = state.get("jobs")
        if jobs is None:
            jobs = {}
            state["jobs"] = jobs

        next_job_id = state.get("next_job_id", 1)
        tool_name = request.tool_call["name"]
        job_id = f"{tool_name}-{next_job_id}"
        state["next_job_id"] = next_job_id + 1

        jobs[job_id] = asyncio.ensure_future(handler(request))

        try:
            result = await asyncio.wait_for(asyncio.shield(jobs[job_id]), timeout=0.5)
            jobs.pop(job_id, None)
            return result if isinstance(result, ToolMessage) else ToolMessage(content=str(result), tool_call_id=request.tool_call["id"])
        except asyncio.TimeoutError:
            pass

        return ToolMessage(content=f"Tool call is being processed with job ID: {job_id}", tool_call_id=request.tool_call["id"])
    
    
    