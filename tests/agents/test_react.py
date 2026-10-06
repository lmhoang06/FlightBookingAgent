"""Unit and integration tests for ReActFlightAgent."""

from langchain_core.messages import AIMessage

from agents import get_flight_agent
from tests.agents.conftest import ToolCallingFakeChatModel


def test_react_agent_instantiation():
    """Verify default instantiation of ReAct agent."""
    agent = get_flight_agent("react")
    assert agent is not None
    assert agent.config.max_iterations == 10
    assert agent.session_id == "default_session"


def test_react_direct_response_without_tools():
    """Test ReAct conversation when no tool calls are needed."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(
                content="Hello! I am your AI flight booking assistant. How can I help you today?"
            )
        ]
    )
    agent = get_flight_agent("react", llm=fake_llm, session_id="test_session_1")
    resp = agent.invoke("Hello")

    assert resp.pattern == "react"
    assert resp.stop_reason == "completed"
    assert "flight booking assistant" in resp.content
    assert resp.session_id == "test_session_1"
    assert len(resp.steps) == 0


def test_react_single_tool_execution():
    """Test ReAct agent invoking search_airports_and_cities tool."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_123",
                "name": "search_airports_and_cities",
                "args": {"query": "Hanoi"},
            }
        ],
    )
    final_msg = AIMessage(
        content="I found Noi Bai International Airport (HAN) in Hanoi, Vietnam."
    )

    fake_llm = ToolCallingFakeChatModel(
        responses=[
            tool_call_msg,
            final_msg,
        ]
    )
    agent = get_flight_agent("react", llm=fake_llm, session_id="test_session_2")
    resp = agent.invoke("Find airport for Hanoi")

    assert resp.pattern == "react"
    assert resp.stop_reason == "completed"
    assert "Noi Bai International Airport (HAN)" in resp.content
    assert len(resp.steps) >= 2  # tool_call and tool_result
    assert any(s.name == "search_airports_and_cities" for s in resp.steps)


def test_react_multi_turn_session_persistence():
    """Test session state retention across multi-turn interactions."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(content="I can help you search for flights to Da Nang."),
            AIMessage(content="Da Nang airport code is DAD."),
        ]
    )
    agent = get_flight_agent(
        "react", llm=fake_llm, session_id="session_persistence_test"
    )

    resp1 = agent.invoke(
        "I want to visit Da Nang", session_id="session_persistence_test"
    )
    assert "Da Nang" in resp1.content

    resp2 = agent.invoke(
        "What is its airport code?", session_id="session_persistence_test"
    )
    assert "DAD" in resp2.content

    history = agent.get_history("session_persistence_test")
    assert len(history) >= 4  # 2 human messages + 2 AI messages


def test_react_reset_session():
    """Test session reset clears checkpointed state."""
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(content="First message."),
        ]
    )
    agent = get_flight_agent("react", llm=fake_llm, session_id="session_to_reset")
    agent.invoke("Hi", session_id="session_to_reset")

    agent.reset("session_to_reset")
    history = agent.get_history("session_to_reset")
    assert len(history) == 0
