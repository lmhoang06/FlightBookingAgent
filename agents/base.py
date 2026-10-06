"""Base interfaces and unified response data structures for flight agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class StepRecord:
    """Represents an observable trace step executed during agent reasoning."""

    step_type: (
        str  # 'thought', 'tool_call', 'tool_result', 'plan', 'replan', 'milestone'
    )
    name: str
    input_data: any = None
    output_data: any = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class AgentResponse:
    """Unified response object returned by all flight agents."""

    content: str
    steps: list = field(default_factory=list)
    session_id: str = "default_session"
    pattern: str = "react"  # 'react', 'plan_then_execute', 'hybrid'
    stop_reason: str = "completed"  # 'completed', 'max_iterations', 'duplicate_action_prevented', 'awaiting_user_input', 'unachievable_goal'
    metadata: dict = field(default_factory=dict)


class BaseFlightAgent(ABC):
    """Abstract base class establishing the contract for all flight booking agents."""

    def __init__(self, config=None, llm=None, session_id="default_session"):
        from .config import AgentConfig, get_llm

        self.config = config or AgentConfig()
        self.llm = llm or get_llm(self.config)
        self.session_id = session_id

    @abstractmethod
    def invoke(self, query: str, session_id=None):
        """Synchronously execute the agent on a user query."""

    @abstractmethod
    def stream(self, query: str, session_id=None):
        """Yield intermediate steps and final chunks/response as generator."""

    @abstractmethod
    def reset(self, session_id=None):
        """Clear memory or context for the designated session."""

    @abstractmethod
    def get_history(self, session_id=None):
        """Retrieve historical messages or steps for the session."""
