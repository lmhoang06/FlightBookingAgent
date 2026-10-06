"""FlightBookingAgent agents package.

Provides ReAct, Plan-then-Execute, and Hybrid agent implementations,
along with unified configuration, interfaces, Harness architecture, and factory functions.
"""

from .base import AgentResponse, BaseFlightAgent, StepRecord
from .config import AgentConfig, get_llm
from .fake_model import ScriptedFakeChatModel
from .harness import (
    Constraints,
    StopAgentException,
    check_hallucination,
    check_permission,
    handoff,
    is_done,
    is_empty_or_error,
    is_looping,
    wrap_tool_with_harness,
)
from .prompts import (
    HYBRID_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
)


def get_flight_agent(
    pattern="react", config=None, llm=None, session_id="default_session", **kwargs
):
    """Factory helper to instantiate a flight agent by pattern name.

    Supported patterns:
        - 'react': Adaptive Reasoning + Acting agent with Harness interceptor
        - 'plan_then_execute': Structured One-shot Plan + Human Gate + Plain Code execution
        - 'hybrid': Macro Milestone + Micro ReAct + Observation Delta Replanner
    """
    pattern_lower = pattern.lower().replace("-", "_")
    if pattern_lower == "react":
        from .react.agent import ReActFlightAgent

        return ReActFlightAgent(config=config, llm=llm, session_id=session_id, **kwargs)
    elif pattern_lower in ("plan_then_execute", "plan_execute"):
        from .plan_then_execute.agent import PlanThenExecuteFlightAgent

        return PlanThenExecuteFlightAgent(
            config=config, llm=llm, session_id=session_id, **kwargs
        )
    elif pattern_lower == "hybrid":
        from .hybrid.agent import HybridFlightAgent

        return HybridFlightAgent(
            config=config, llm=llm, session_id=session_id, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown agent pattern: {pattern}. Choose from 'react', 'plan_then_execute', or 'hybrid'."
        )


__all__ = [
    "HYBRID_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "REACT_SYSTEM_PROMPT",
    "REPLANNER_SYSTEM_PROMPT",
    "AgentConfig",
    "AgentResponse",
    "BaseFlightAgent",
    "Constraints",
    "ScriptedFakeChatModel",
    "StepRecord",
    "StopAgentException",
    "check_hallucination",
    "check_permission",
    "get_flight_agent",
    "get_llm",
    "handoff",
    "is_done",
    "is_empty_or_error",
    "is_looping",
    "wrap_tool_with_harness",
]
