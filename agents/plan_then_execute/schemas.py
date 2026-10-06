"""Schemas and state representations for Plan-then-Execute agent."""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A discrete step in an execution plan."""

    step_id: int = Field(description="Step sequence index starting at 1")
    description: str = Field(description="Actionable instruction for this step")
    tool_hint: str = Field(default="", description="Suggested tool name if applicable")


class Plan(BaseModel):
    """Structured plan composed of steps."""

    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Sequential list of steps to fulfill the user request",
    )


class ReplannerOutput(BaseModel):
    """Output schema for the replanner."""

    is_complete: bool = Field(
        default=False,
        description="Whether the goal has been fully met or completed",
    )
    final_response: str = Field(
        default="",
        description="Final answer to provide the user if is_complete is True",
    )
    remaining_steps: list[str] = Field(
        default_factory=list,
        description="Updated list of remaining steps if goal is not complete",
    )


@dataclass
class PlanExecuteState:
    """State object tracked throughout the Plan-then-Execute graph."""

    query: str = ""
    plan: list = field(default_factory=list)
    completed_steps: list = field(default_factory=list)
    current_step: str = ""
    current_step_result: str = ""
    response: str = ""
    replan_count: int = 0
    step_count: int = 0
    stop_reason: str = "completed"
    session_id: str = "default_session"
    history: list = field(default_factory=list)
