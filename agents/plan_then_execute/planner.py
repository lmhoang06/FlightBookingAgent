"""Planner and Replanner logic for Plan-then-Execute agent."""

from langchain_core.messages import HumanMessage, SystemMessage

from ..harness import Constraints
from ..prompts import PLANNER_SYSTEM_PROMPT, REPLANNER_SYSTEM_PROMPT
from .schemas import Plan, PlanStep, ReplannerOutput


class Planner:
    """Generates a complete step-by-step plan in one shot based on the user's travel request."""

    def __init__(self, llm):
        self.llm = llm

    def plan_structured(self, query: str, constraints=None):
        """Generates a structured Plan with tool names and arguments."""
        c = constraints or Constraints()
        prompt = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"User Goal: {query}\n"
            f"Constraints: {c.to_prompt()}\n\n"
            "Produce the complete step-by-step plan."
        )
        try:
            planner_llm = self.llm.with_structured_output(Plan)
            res = planner_llm.invoke(prompt)
            if isinstance(res, Plan) and res.steps:
                return res
        except Exception:
            pass

        # Deterministic slide-compliant fallback plan
        return Plan(
            steps=[
                PlanStep(
                    step_id=1,
                    description=f"Search flights from {c.origin} to {c.destination} on {c.date}",
                    tool="search_flights",
                    tool_hint="search_flights",
                    args={
                        "origin": c.origin,
                        "destination": c.destination,
                        "departure_date": c.date,
                        "adults": c.passenger_count,
                        "trip_type": "ONE_WAY",
                    },
                ),
                PlanStep(
                    step_id=2,
                    description=f"Book flight offer for passenger {c.passenger_name}",
                    tool="create_flight_booking",
                    tool_hint="create_flight_booking",
                    args={
                        "offer_id": "$offer_id",
                        "contact_email": "passenger@example.com",
                        "contact_phone": "0901234567",
                    },
                ),
            ]
        )

    def plan(self, query: str):
        """Generates a list of plan step descriptions (maintains legacy compatibility)."""
        structured = self.plan_structured(query)
        if structured and structured.steps:
            return [s.description for s in structured.steps]

        try:
            res = self.llm.invoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            content = res.content if hasattr(res, "content") else str(res)
            steps = []
            for line in content.splitlines():
                line_str = line.strip()
                if line_str and (line_str[0].isdigit() or line_str.startswith("-")):
                    cleaned = line_str.lstrip("0123456789.-) ")
                    if cleaned:
                        steps.append(cleaned)
            if steps:
                return steps
        except Exception:
            pass

        return [
            f"Search flight offers matching criteria for: {query}",
            f"Book best matching flight offer for: {query}",
        ]


class Replanner:
    """Evaluates step progress and determines remaining steps or completion."""

    def __init__(self, llm):
        self.llm = llm

    def replan(
        self,
        query: str,
        completed_steps: list,
        remaining_steps: list,
        latest_result: str,
    ):
        """Returns ReplannerOutput indicating completion status and next steps."""
        prompt = (
            f"{REPLANNER_SYSTEM_PROMPT}\n\n"
            f"User Request: {query}\n"
            f"Completed Steps: {completed_steps}\n"
            f"Remaining Steps: {remaining_steps}\n"
            f"Latest Result: {latest_result}\n\n"
            "Assess whether we are finished, need more steps, or need user decision."
        )
        try:
            replanner_llm = self.llm.with_structured_output(ReplannerOutput)
            res = replanner_llm.invoke(prompt)
            if isinstance(res, ReplannerOutput):
                return res
        except Exception:
            pass

        if not remaining_steps:
            return ReplannerOutput(
                is_complete=True,
                final_response=latest_result or "Process completed successfully.",
                remaining_steps=[],
            )

        return ReplannerOutput(
            is_complete=False,
            final_response="",
            remaining_steps=remaining_steps,
        )
