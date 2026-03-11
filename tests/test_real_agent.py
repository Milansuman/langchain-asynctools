"""
Integration tests with a real agent.
"""

import asyncio
import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_asynctools import AsyncTools

@tool
async def lookup_flight_delay(flight_number: str) -> str:
	"""Look up the latest delay estimate for a flight."""
	await asyncio.sleep(10)
	return f"Flight {flight_number} is delayed by 18 minutes."


@tool
async def lookup_gate_services(gate: str) -> str:
	"""Look up the currently available passenger services at a gate."""
	await asyncio.sleep(10)
	return f"Gate {gate} has coffee, charging, and standby assistance available."


@pytest.mark.asyncio
async def test_real_agent_with_async_tools(env, netra, llm):
	if not env.get("LITELLM_API_KEY"):
		pytest.skip("LITELLM_API_KEY is not configured")

	middleware = AsyncTools()
	agent = create_agent(
		model=llm,
		tools=[lookup_flight_delay, lookup_gate_services],
		middleware=[middleware],
	)

	result = await agent.ainvoke(
		{
			"messages": [
				HumanMessage(
					content=(
						"You are preparing a short airport ops update. "
						"Use the flight and gate tools for flight ZX-204 and gate B12. "
						"If either tool returns a job ID, wait for the result before answering. "
						"Your final answer must mention both the delay and the gate services."
					)
				)
			]
		}
	)

	tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
	final_ai_message = next(
		message for message in reversed(result["messages"]) if isinstance(message, AIMessage)
	)
	final_text = final_ai_message.text().lower()

	assert any("job ID" in str(message.content) for message in tool_messages)
	assert any("Flight ZX-204 is delayed by 18 minutes." in str(message.content) for message in tool_messages)
	assert any(
		"Gate B12 has coffee, charging, and standby assistance available." in str(message.content)
		for message in tool_messages
	)
	assert "zx-204" in final_text
	assert "18 minutes" in final_text
	assert "b12" in final_text
	assert "coffee" in final_text
	assert result.get("jobs", {}) == {}