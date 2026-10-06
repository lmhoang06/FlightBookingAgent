"""Schemas and state representations for Hybrid (Macro + Micro) agent."""

from enum import Enum

from pydantic import BaseModel, Field


class MilestoneStatus(str, Enum):
    """Execution status of a macro milestone."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Milestone(BaseModel):
    """High-level macro milestone in the flight booking lifecycle."""

    name: str = Field(
        description="Name of milestone (e.g. DISCOVERY, SEARCH, OFFER_SELECTION, BOOKING_CONFIRMATION)"
    )
    description: str = Field(description="Objective to satisfy in this milestone")
    status: MilestoneStatus = Field(default=MilestoneStatus.PENDING)


class MilestoneResult(BaseModel):
    """Result returned by micro-ReAct solver after resolving a milestone."""

    milestone_name: str
    success: bool
    summary: str
    extracted_data: dict = Field(default_factory=dict)


class HybridPlan(BaseModel):
    """Macro plan containing all milestones."""

    milestones: list[Milestone] = Field(
        default_factory=lambda: [
            Milestone(
                name="DISCOVERY",
                description="Resolve airport IATA codes, cities, and traveler preferences",
            ),
            Milestone(
                name="SEARCH",
                description="Search available flight offers and check schedule/pricing",
            ),
            Milestone(
                name="OFFER_SELECTION",
                description="Inspect and select the most suitable flight offer",
            ),
            Milestone(
                name="BOOKING_CONFIRMATION",
                description="Verify passenger information and finalize or confirm booking",
            ),
        ]
    )
