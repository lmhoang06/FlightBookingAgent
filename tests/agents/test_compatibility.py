"""End-to-End compatibility test suite across all three flight booking agent patterns."""

import pytest
from langchain_core.messages import AIMessage

from agents import get_flight_agent
from tests.agents.conftest import ToolCallingFakeChatModel


@pytest.mark.parametrize("pattern", ["react", "plan_then_execute", "hybrid"])
def test_all_patterns_unified_interface(pattern):
    """Verify that all three agent patterns adhere strictly to BaseFlightAgent interface and return AgentResponse."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(content="I can help you search flights from Hanoi to Da Nang."),
            AIMessage(content="Flights found."),
            AIMessage(content="Booking completed."),
            AIMessage(content="Final summary."),
        ]
    )

    agent = get_flight_agent(
        pattern=pattern, llm=fake_llm, session_id=f"test_session_{pattern}"
    )
    resp = agent.invoke("Book a flight from HAN to DAD")

    # Assert unified contract
    assert resp.content is not None
    assert isinstance(resp.content, str)
    assert isinstance(resp.steps, list)
    assert resp.session_id == f"test_session_{pattern}"
    assert resp.pattern in ("react", "plan_then_execute", "hybrid")
    assert resp.stop_reason in (
        "completed",
        "max_iterations",
        "duplicate_action_prevented",
        "awaiting_user_input",
        "unachievable_goal",
    )


@pytest.mark.parametrize("pattern", ["react", "plan_then_execute", "hybrid"])
def test_all_patterns_stream_interface(pattern):
    """Verify that all three agent patterns support generator streaming."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(content="Stream test message 1."),
            AIMessage(content="Stream test message 2."),
        ]
    )

    agent = get_flight_agent(pattern=pattern, llm=fake_llm)
    gen = agent.stream("Stream query")
    assert hasattr(gen, "__iter__") or hasattr(gen, "__next__")
