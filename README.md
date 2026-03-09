# langchain-asynctools

Non-blocking tool call execution middleware for LangChain agents. Async tools are dispatched as background tasks, allowing the agent to continue without waiting for long-running tool calls to complete. Results can be retrieved later using built-in polling and wait tools that are automatically injected into the agent.

## Requirements

- Python 3.12+
- `langchain >= 1.2.10`

## Installation

```bash
pip install langchain-asynctools
```

## How it works

When an async tool call is invoked, `AsyncTools` schedules it as an `asyncio.Task` and immediately returns a `ToolMessage` containing a job ID. If the tool completes within a 0.5-second window, the result is returned directly — no job ID is issued.

The middleware automatically appends two tools to every model call:

- `query_tool_output(job_id)` — returns the result if the task is done, or a status message if it is still running.
- `await_tool_output(job_id, wait_time)` — waits up to `wait_time` seconds for the task to complete before returning.

The agent receives instructions about these tools via an automatically injected addendum to the system prompt.

Synchronous tool calls pass through unchanged.

## Usage

```python
import asyncio
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_asynctools import AsyncTools


@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"Sunny in {city}."


@tool
async def run_long_analysis(data: str) -> str:
    """Runs a long analysis on the provided data."""
    await asyncio.sleep(10)
    return f"Analysis complete: {data}"


agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather, run_long_analysis],
    middleware=[AsyncTools()],
)

result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Analyse this data and get the weather in London."}]}))
```

### Retrieving results

After receiving a job ID, the agent can call `query_tool_output` to check whether a result is ready:

```
query_tool_output(job_id="140234567890")
# -> "Job 140234567890 is still processing."
# or
# -> "Analysis complete: ..."
```

Or block for a bounded period using `await_tool_output`:

```
await_tool_output(job_id="140234567890", wait_time=30)
# -> "Analysis complete: ..."
# or
# -> "Job 140234567890 did not complete within 30 seconds."
```

## Notes

- `AsyncTools` must be used with `agent.ainvoke` or `agent.astream`. The background tasks rely on a running asyncio event loop.
- Each `AsyncTools` instance maintains its own job registry. Do not share a single instance across concurrent agent invocations unless you manage job ID namespacing yourself.
- Background tasks are shielded from cancellation when awaited via `await_tool_output`, so they will continue running even if the wait times out.
