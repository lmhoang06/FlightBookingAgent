"""Planner and Replanner logic for Plan-then-Execute agent."""

from langchain_core.messages import HumanMessage, SystemMessage

from ..prompts import PLANNER_SYSTEM_PROMPT, REPLANNER_SYSTEM_PROMPT
from .schemas import Plan, ReplannerOutput


class Planner:
    """Generates an initial plan of action based on the user's travel request."""

    def __init__(self, llm):
        self.llm = llm

    def plan(self, query: str):
        """Generates a structured Plan."""
        prompt = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"User Goal: {query}\n\n"
            "Produce the step-by-step plan."
        )
        try:
            planner_llm = self.llm.with_structured_output(Plan)
            res = planner_llm.invoke(prompt)
            if res and res.steps:
                return [s.description for s in res.steps]
        except Exception:
            pass

        # Fallback heuristic or direct text generation
        res = self.llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )
        content = res.content if hasattr(res, "content") else str(res)
        # Parse numbered list
        steps = []
        for line in content.splitlines():
            line_str = line.strip()
            if line_str and (line_str[0].isdigit() or line_str.startswith("-")):
                cleaned = line_str.lstrip("0123456789.-) ")
                if cleaned:
                    steps.append(cleaned)

        if not steps:
            steps = [
                f"Resolve airport codes and locations for request: {query}",
                f"Search flight offers matching criteria for: {query}",
                "Review offer details and summarize available flights",
            ]
        return steps


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

        # Fallback: if remaining_steps is empty, complete with summary
        if not remaining_steps:
            return ReplannerOutput(
                is_complete=True,
                final_response=latest_result or "Process completed successfully.",
                remaining_steps=[],
            )

        # Otherwise continue with next remaining steps
        return ReplannerOutput(
            is_complete=False,
            final_response="",
            remaining_steps=remaining_steps,
        )
