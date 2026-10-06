"""Unit and integration tests for HybridFlightAgent."""

from langchain_core.messages import AIMessage

from agents import get_flight_agent
from tests.agents.conftest import ToolCallingFakeChatModel


def test_hybrid_agent_instantiation():
    """Verify default instantiation of Hybrid agent."""
    agent = get_flight_agent("hybrid")
    assert agent is not None
    assert agent.config.micro_max_steps == 4
    assert agent.session_id == "default_session"


def test_hybrid_milestone_progression():
    """Test macro milestone progression across micro loops."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            # Discovery milestone
            AIMessage(content="Airport resolved: SGN and HAN."),
            # Search milestone
            AIMessage(content="Found flights VN210 and VJ112."),
            # Offer selection milestone
            AIMessage(content="Selected offer VN210 for $110."),
            # Booking confirmation milestone
            AIMessage(content="Booking confirmed with ID BK-12345."),
        ]
    )
    agent = get_flight_agent("hybrid", llm=fake_llm)
    resp = agent.invoke("Book a flight from SGN to HAN")

    assert resp.pattern == "hybrid"
    assert resp.stop_reason in ("completed", "awaiting_user_input")
    assert len(resp.steps) > 0
    milestone_steps = [s for s in resp.steps if s.step_type == "milestone"]
    assert len(milestone_steps) >= 1


def test_hybrid_awaiting_user_input():
    """Test that agent halts when user choice is required."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(content="Discovered airports SGN and DAD."),
            AIMessage(
                content="Please select which flight offer you prefer: Flight A ($100) or Flight B ($150)?"
            ),
        ]
    )
    agent = get_flight_agent("hybrid", llm=fake_llm)
    resp = agent.invoke("Search flights")

    assert resp.stop_reason == "awaiting_user_input"
    assert "which flight" in resp.content.lower() or "select" in resp.content.lower()
