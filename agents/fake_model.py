"""Deterministic Scripted Fake Chat Model for offline evaluation and testing.

Provides predictable offline behaviors matching the course scenarios:
1. case='success': Valid search -> book valid flight -> paid/confirmed -> is_done() == True.
2. case='fail_drift': Valid search -> attempt out-of-budget/afternoon booking -> blocked by Harness -> handoff.
3. case='fail_loop': Repeated search calls (3x) -> interrupted by Harness loop detector.
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from .plan_then_execute.schemas import Plan, PlanStep, ReplannerOutput


class ScriptedFakeChatModel(BaseChatModel):
    """Deterministic offline chat model supporting scripted case trajectories."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case: str = "success"  # 'success', 'fail_drift', 'fail_loop', 'custom'
    responses: list = Field(default_factory=list)
    response_idx: int = 0
    loop_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted_fake_chat_model"

    def bind_tools(self, tools, **kwargs):
        """Pass-through for tool binding."""
        return self

    def with_structured_output(self, schema, **kwargs):
        """Supports structured output for Planner / Replanner schemas."""
        class StructuredRunner:
            def __init__(self, target_schema):
                self.target_schema = target_schema

            def invoke(self, prompt, **kw):
                if issubclass(self.target_schema, Plan):
                    return Plan(
                        steps=[
                            PlanStep(
                                step_id=1,
                                description="Search flights from SGN to DAD on 2026-10-07",
                                tool_hint="search_flights",
                            ),
                            PlanStep(
                                step_id=2,
                                description="Create flight booking for selected offer",
                                tool_hint="create_flight_booking",
                            ),
                        ]
                    )
                elif issubclass(self.target_schema, ReplannerOutput):
                    return ReplannerOutput(
                        is_complete=True,
                        final_response="Flight booking completed successfully.",
                        remaining_steps=[],
                    )
                return self.target_schema()

        return StructuredRunner(schema)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generates deterministic AIMessage based on current conversation history and case."""
        # If custom responses list was supplied, use standard sequential playback
        if self.responses and self.case == "custom":
            if self.response_idx < len(self.responses):
                msg = self.responses[self.response_idx]
                self.response_idx += 1
                return ChatResult(generations=[ChatGeneration(message=msg)])
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="Task completed.")
                    )
                ]
            )

        # Inspect last message
        last_msg = messages[-1] if messages else None

        # Case: fail_loop
        if self.case == "fail_loop":
            self.loop_count += 1
            call_id = f"loop_call_{self.loop_count}"
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": "search_flights",
                        "args": {
                            "origin": "SGN",
                            "destination": "DAD",
                            "departure_date": "2026-10-07",
                        },
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Initial turn (no ToolMessage yet in history)
        has_tool_msg = any(isinstance(m, ToolMessage) for m in messages)
        if not has_tool_msg:
            msg = AIMessage(
                content="I will search for available flights from SGN to DAD on 2026-10-07.",
                tool_calls=[
                    {
                        "id": "call_search_1",
                        "name": "search_flights",
                        "args": {
                            "origin": "SGN",
                            "destination": "DAD",
                            "departure_date": "2026-10-07",
                            "departure_time_window": "MORNING",
                        },
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Follow-up turns after ToolMessages
        if isinstance(last_msg, ToolMessage):
            tool_name = getattr(last_msg, "name", "")
            raw_content = last_msg.content or "{}"

            # After search_flights
            if tool_name == "search_flights" or "offers" in str(raw_content):
                offers = []
                try:
                    parsed = json.loads(raw_content)
                    offers = parsed.get("offers", [])
                except Exception:
                    pass

                if self.case == "success":
                    # Pick morning, in-budget offer (e.g. VN-120 or first offer)
                    chosen_offer_id = None
                    for o in offers:
                        itin = o.get("itineraries", [{}])[0]
                        segs = itin.get("segments", [{}])
                        if segs:
                            dep = segs[0].get("departure_time", "")
                            if "08:15" in dep or "10:30" in dep or "06:30" in dep:
                                chosen_offer_id = o.get("offer_id")
                                break
                    if not chosen_offer_id and offers:
                        chosen_offer_id = offers[0].get("offer_id")

                    chosen_offer_id = chosen_offer_id or "OFFER-VN-120"
                    msg = AIMessage(
                        content="Found flight VN-120. Proceeding to book passenger.",
                        tool_calls=[
                            {
                                "id": "call_book_1",
                                "name": "create_flight_booking",
                                "args": {
                                    "offer_id": chosen_offer_id,
                                    "travelers": [
                                        {
                                            "first_name": "Nguyen Van",
                                            "last_name": "A",
                                            "date_of_birth": "1990-01-01",
                                        }
                                    ],
                                    "contact_email": "passenger@example.com",
                                    "contact_phone": "0901234567",
                                },
                            }
                        ],
                    )
                    return ChatResult(generations=[ChatGeneration(message=msg)])

                elif self.case == "fail_drift":
                    # Attempt to book an offer that violates constraints (VN-128: 17:45, price 95)
                    chosen_offer_id = None
                    for o in offers:
                        itin = o.get("itineraries", [{}])[0]
                        segs = itin.get("segments", [{}])
                        if segs:
                            dep = segs[0].get("departure_time", "")
                            if "17:45" in dep or "20:15" in dep:
                                chosen_offer_id = o.get("offer_id")
                                break
                    if not chosen_offer_id and offers:
                        chosen_offer_id = offers[-1].get("offer_id")

                    chosen_offer_id = chosen_offer_id or "OFFER-VN-128"
                    msg = AIMessage(
                        content="Attempting to book flight VN-128 departing at 17:45.",
                        tool_calls=[
                            {
                                "id": "call_book_drift",
                                "name": "create_flight_booking",
                                "args": {
                                    "offer_id": chosen_offer_id,
                                    "travelers": [
                                        {
                                            "first_name": "Nguyen Van",
                                            "last_name": "A",
                                            "date_of_birth": "1990-01-01",
                                        }
                                    ],
                                    "contact_email": "passenger@example.com",
                                    "contact_phone": "0901234567",
                                },
                            }
                        ],
                    )
                    return ChatResult(generations=[ChatGeneration(message=msg)])

            # After create_flight_booking
            if tool_name == "create_flight_booking" or "order" in str(raw_content) or "denied" in str(raw_content).lower():
                if "denied" in str(raw_content).lower() or self.case == "fail_drift":
                    msg = AIMessage(
                        content=(
                            "I cannot complete the booking because the requested flight violates "
                            "your data constraints (departure time or maximum price)."
                        )
                    )
                    return ChatResult(generations=[ChatGeneration(message=msg)])
                else:
                    msg = AIMessage(
                        content=(
                            "Booking successfully confirmed for flight VN-120 from SGN to DAD on 2026-10-07. "
                            "Passenger: Nguyen Van A. Ticket status: CONFIRMED."
                        )
                    )
                    return ChatResult(generations=[ChatGeneration(message=msg)])

        # Default completion message
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="Process finished.")
                )
            ]
        )
