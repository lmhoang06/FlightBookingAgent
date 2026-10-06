"""Unit and integration tests for PlanThenExecuteFlightAgent."""

from langchain_core.messages import AIMessage

from agents import get_flight_agent
from tests.agents.conftest import ToolCallingFakeChatModel


def test_plan_execute_instantiation():
    """Verify default instantiation of Plan-then-Execute agent."""
    agent = get_flight_agent("plan_then_execute")
    assert agent is not None
    assert agent.config.max_plan_steps == 8
    assert agent.session_id == "default_session"


def test_plan_execute_basic_flow():
    """Test standard plan generation and execution loop."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            # Step execution 1 response
            AIMessage(content="Airport code for Da Nang is DAD."),
            # Step execution 2 response
            AIMessage(content="Flights found for DAD."),
            # Step execution 3 response
            AIMessage(content="Flight VN123 is available at $120."),
        ]
    )
    agent = get_flight_agent("plan_then_execute", llm=fake_llm)
    # Mock planner steps directly for deterministic flow
    agent.planner.plan = lambda query: [
        "Resolve airport code for Da Nang",
        "Search flights to Da Nang",
    ]

    resp = agent.invoke("Book a flight to Da Nang")
    assert resp.pattern == "plan_then_execute"
    assert resp.stop_reason in ("completed", "max_iterations")
    assert len(resp.steps) > 0


def test_plan_execute_replanning_triggers():
    """Test that replanner runs after step execution."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(content="Resolved airport"),
            AIMessage(content="Found flights"),
        ]
    )
    agent = get_flight_agent("plan_then_execute", llm=fake_llm)
    agent.planner.plan = lambda query: ["Step 1", "Step 2"]

    resp = agent.invoke("Find flights")
    replan_steps = [s for s in resp.steps if s.step_type == "replan"]
    assert len(replan_steps) >= 1
