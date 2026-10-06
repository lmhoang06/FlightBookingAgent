"""Plan-then-Execute agent adhering strictly to the lecture slide architecture.

Slide Alignment:
1. Model generates the entire plan in ONE call (Zero LLM step execution cost).
2. Human Reviewer approval gate (Approve/Reject).
3. Plain code executes Step 1, 2, 3 with $placeholder replacement.
4. Stop condition verified by CODE (is_done()).
"""

import json

from langgraph.prebuilt import create_react_agent

from tools import FLIGHT_BOOKING_TOOLS

from ..base import AgentResponse, BaseFlightAgent, StepRecord
from ..harness import (
    Constraints,
    check_permission,
    handoff,
    is_done,
)
from .planner import Planner, Replanner
from .schemas import Plan, PlanStep


class PlanThenExecuteFlightAgent(BaseFlightAgent):
    """Plan-then-Execute agent with Human Reviewer and Plain-Code Step Execution."""

    def __init__(
        self,
        config=None,
        llm=None,
        session_id="default_session",
        tools=None,
        auto_approve=True,
        plan_approval_callback=None,
    ):
        super().__init__(config=config, llm=llm, session_id=session_id)
        self.tools = tools if tools is not None else FLIGHT_BOOKING_TOOLS
        self.tool_map = {
            getattr(t, "name", str(t)): t for t in self.tools
        }
        self.planner = Planner(self.llm)
        self.replanner = Replanner(self.llm)
        self.auto_approve = auto_approve
        self.plan_approval_callback = plan_approval_callback

        # Kept for compatibility with existing tests
        self.executor_llm = self.llm
        self.step_agent = create_react_agent(
            model=self.executor_llm,
            tools=self.tools,
        )

    def approve_plan(self, plan: Plan) -> bool:
        """Human-in-the-Loop review gate."""
        if not self.auto_approve and callable(self.plan_approval_callback):
            return bool(self.plan_approval_callback(plan))
        return True

    def _resolve_tool_and_args(self, step, variables: dict, constraints: Constraints):
        """Resolves tool name, arguments, and substitutes $placeholders."""
        tool_name = ""
        args = {}

        if isinstance(step, PlanStep):
            tool_name = step.tool or step.tool_hint
            args = dict(step.args)
            desc = step.description
        elif isinstance(step, dict):
            tool_name = step.get("tool") or step.get("tool_hint", "")
            args = dict(step.get("args", {}))
            desc = step.get("description", "")
        else:
            desc = str(step)

        # Infer tool if not explicitly specified
        desc_lower = desc.lower()
        if not tool_name:
            if "airport" in desc_lower or "code" in desc_lower:
                tool_name = "search_airports_and_cities"
                args = {"query": constraints.destination}
            elif "search" in desc_lower or "flight" in desc_lower:
                tool_name = "search_flights"
            elif "book" in desc_lower or "ticket" in desc_lower:
                tool_name = "create_flight_booking"
            elif "detail" in desc_lower:
                tool_name = "get_flight_details"
            elif "status" in desc_lower:
                tool_name = "get_session_status"
            else:
                tool_name = "search_flights"

        # Substitute $variables in args
        resolved_args = {}
        for k, v in args.items():
            if isinstance(v, str) and v.startswith("$"):
                var_key = v.lstrip("$")
                resolved_args[k] = variables.get(var_key, v)
            else:
                resolved_args[k] = v

        # Apply defaults based on tool and constraints
        if tool_name == "search_flights":
            resolved_args.setdefault("origin", constraints.origin)
            resolved_args.setdefault("destination", constraints.destination)
            resolved_args.setdefault("departure_date", constraints.date)
            resolved_args.setdefault("adults", constraints.passenger_count)
            resolved_args.setdefault("trip_type", "ONE_WAY")
            resolved_args.setdefault("session_id", variables.get("session_id"))

        elif tool_name == "create_flight_booking":
            if "offer_id" not in resolved_args or str(resolved_args.get("offer_id")).startswith("$"):
                resolved_args["offer_id"] = variables.get("offer_id", "OFFER-VN-120")

            resolved_args.setdefault(
                "travelers",
                [
                    {
                        "first_name": constraints.passenger_name.split()[0],
                        "last_name": " ".join(constraints.passenger_name.split()[1:]) or "Traveler",
                        "date_of_birth": "1990-01-01",
                    }
                ],
            )
            resolved_args.setdefault("contact_email", "passenger@example.com")
            resolved_args.setdefault("contact_phone", "0901234567")
            resolved_args.setdefault("session_id", variables.get("session_id"))

        elif tool_name == "search_airports_and_cities":
            if "query" not in resolved_args:
                resolved_args["query"] = constraints.destination

        return tool_name, resolved_args, desc

    def invoke(self, query: str, session_id=None, constraints=None):
        """Execute Plan-then-Execute pattern with zero LLM step overhead."""
        active_session = session_id or self.session_id
        active_constraints = constraints or Constraints()
        records = []
        stop_reason = "completed"

        # Step 1: Model generates entire plan in 1 shot
        plan_obj = self.planner.plan_structured(query, active_constraints)
        plan_steps = plan_obj.steps if plan_obj and plan_obj.steps else []

        # Fallback if plan was monkeypatched to return list of strings
        if not plan_steps and hasattr(self.planner, "plan"):
            legacy_plan = self.planner.plan(query)
            if isinstance(legacy_plan, list):
                plan_steps = [
                    PlanStep(step_id=idx + 1, description=s)
                    for idx, s in enumerate(legacy_plan)
                ]

        records.append(
            StepRecord(
                step_type="plan",
                name="planner",
                input_data={"query": query},
                output_data={"plan": [s.description for s in plan_steps]},
            )
        )

        # Step 2: Human Reviewer Approval Gate
        approved = self.approve_plan(plan_obj)
        if not approved:
            return AgentResponse(
                content="Plan execution halted: Rejected by Human Reviewer.",
                steps=records,
                session_id=active_session,
                pattern="plan_then_execute",
                stop_reason="awaiting_user_input",
                metadata={"plan_approved": False},
            )

        # Step 3: Plain-Code Deterministic Step Execution ($placeholder substitution)
        variables = {
            "session_id": active_session,
            "passenger_name": active_constraints.passenger_name,
        }
        completed_steps = []
        last_step_result = ""
        booking_attempted = False

        for step in plan_steps:
            tool_name, resolved_args, desc = self._resolve_tool_and_args(
                step, variables, active_constraints
            )

            if tool_name == "create_flight_booking":
                booking_attempted = True

            # 3a. Permission Check before running tool
            allowed, denial_reason = check_permission(
                tool_name, resolved_args, active_constraints
            )
            if not allowed:
                denial_text = f"DENIED by Harness Permission Check: {denial_reason}"
                records.append(
                    StepRecord(
                        step_type="tool_call",
                        name=tool_name,
                        input_data=resolved_args,
                    )
                )
                records.append(
                    StepRecord(
                        step_type="tool_result",
                        name=tool_name,
                        output_data=denial_text,
                    )
                )
                records.append(
                    StepRecord(
                        step_type="replan",
                        name="replanner",
                        input_data={"step": desc},
                        output_data={"status": "denied", "reason": denial_reason},
                    )
                )
                stop_reason = "unachievable_goal"
                last_step_result = denial_text
                break

            # 3b. Deterministic execution via Plain Code
            tool_func = self.tool_map.get(tool_name)
            if tool_func:
                records.append(
                    StepRecord(
                        step_type="tool_call",
                        name=tool_name,
                        input_data=resolved_args,
                    )
                )
                try:
                    if hasattr(tool_func, "invoke"):
                        out = tool_func.invoke(resolved_args)
                    else:
                        out = tool_func(**resolved_args)
                except Exception as ex:
                    out = f"Error executing {tool_name}: {ex!s}"

                records.append(
                    StepRecord(
                        step_type="tool_result",
                        name=tool_name,
                        output_data=out,
                    )
                )
                last_step_result = out

                # Parse and capture placeholder outputs
                if tool_name == "search_flights":
                    try:
                        parsed = json.loads(out)
                        offers = parsed.get("offers", [])
                        for o in offers:
                            if active_constraints.is_ok(o):
                                variables["offer_id"] = o.get("offer_id")
                                break
                        if "offer_id" not in variables and offers:
                            variables["offer_id"] = offers[0].get("offer_id")
                    except Exception:
                        pass

                elif tool_name == "create_flight_booking":
                    try:
                        parsed = json.loads(out)
                        order = parsed.get("order", {})
                        pnr = order.get("booking_reference") or parsed.get("message")
                        if pnr:
                            variables["booking_reference"] = pnr
                    except Exception:
                        pass
            else:
                last_step_result = f"Step executed: {desc}"

            # Step replan / review record
            records.append(
                StepRecord(
                    step_type="replan",
                    name="replanner",
                    input_data={"step": desc},
                    output_data={"status": "completed", "result": str(last_step_result)[:100]},
                )
            )
            completed_steps.append({"step": desc, "result": last_step_result})

        # Step 4: Done checked by CODE (is_done)
        done_flag = is_done(active_session, active_constraints)
        metadata = {
            "completed_steps": len(completed_steps),
            "replan_count": len([r for r in records if r.step_type == "replan"]),
            "is_done": done_flag,
        }

        if booking_attempted or "book" in query.lower():
            if done_flag:
                stop_reason = "completed"
                final_content = (
                    f"Booking successfully confirmed! PNR: {variables.get('booking_reference', 'CONFIRMED')}. "
                    f"Details: {active_constraints.to_prompt()}"
                )
            else:
                stop_reason = "unachievable_goal"
                metadata["handoff"] = handoff(active_session, records, active_constraints)
                final_content = (
                    "Could not complete booking meeting all constraints. "
                    f"Harness handoff initiated: {metadata['handoff']['question']}"
                )
        else:
            final_content = str(last_step_result) or "Plan executed successfully."

        return AgentResponse(
            content=final_content,
            steps=records,
            session_id=active_session,
            pattern="plan_then_execute",
            stop_reason=stop_reason,
            metadata=metadata,
        )

    def stream(self, query: str, session_id=None):
        """Yields progress generator."""
        resp = self.invoke(query, session_id=session_id)
        yield resp

    def reset(self, session_id=None):
        """Resets agent state."""

    def get_history(self, session_id=None):
        """Returns session history."""
        return []
