"""Harness Architecture Module for FlightBookingAgent.

Implements the 4 course harness layers:
1. Constraints are DATA (Structured Constraints class acting as data)
2. Permission Check (Intercepts before tools, enforces data constraints)
3. Done checked by CODE (Queries database directly, never trusts LLM text)
4. Handoff to Human (Structured handoff with done_so_far, tried, question)

Also provides resilience to 4 classic failure modes:
- Infinite Loop Detection (is_looping)
- Tool Hallucination Detection (check_hallucination)
- Goal Drift Prevention (enforced via check_permission)
- State Corruption Detection (is_empty_or_error)
"""

import json
import re
from dataclasses import dataclass

from langchain_core.tools import StructuredTool

from tools.db import db


class StopAgentException(Exception):
    """Exception raised when harness interrupts agent execution due to safety policies."""

    def __init__(self, reason: str, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


@dataclass
class Constraints:
    """Constraints as DATA (Ràng buộc là dữ liệu)."""

    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"  # Departure time threshold (HH:MM)
    depart_after: str = "00:00"   # HH:MM
    max_price: int = 2_000_000    # Max allowable price in VND (or equivalent USD)
    passenger_name: str = "Nguyen Van A"
    passenger_count: int = 1

    def to_prompt(self) -> str:
        """Deterministically generates prompt from structured data."""
        return (
            f"Book {self.passenger_count} ticket(s) {self.origin} -> {self.destination} on {self.date}, "
            f"departing between {self.depart_after} and {self.depart_before}, "
            f"price at most {self.max_price:,}. Passenger: {self.passenger_name}."
        )

    def normalize_price(self, price: float, currency: str = "USD") -> float:
        """Converts price to the constraint currency scale (VND vs USD)."""
        curr = (currency or "USD").upper()
        if self.max_price >= 10_000:
            # Constraints in VND (e.g. 2,000,000 VND)
            if curr == "USD" or price < 10_000:
                return float(price) * 25_000.0
            return float(price)
        else:
            # Constraints in USD (e.g. 100 USD)
            if curr == "VND" or price >= 10_000:
                return float(price) / 25_000.0
            return float(price)

    def is_ok(self, flight_or_booking) -> bool:
        """Validates if a flight offer or booking strictly satisfies all data constraints."""
        if not flight_or_booking:
            return False

        # Extract fields whether flight_or_booking is Pydantic model or dict
        if hasattr(flight_or_booking, "model_dump"):
            data = flight_or_booking.model_dump()
        elif isinstance(flight_or_booking, dict):
            data = flight_or_booking
        else:
            data = getattr(flight_or_booking, "__dict__", {})

        # Extract route and times from itineraries / segments
        itineraries = data.get("itineraries") or []
        first_segment = None
        last_segment = None

        if itineraries:
            segments = itineraries[0].get("segments", [])
            if segments:
                first_segment = segments[0]
                last_segment = segments[-1]
        elif "segments" in data:
            segments = data.get("segments", [])
            if segments:
                first_segment = segments[0]
                last_segment = segments[-1]

        # 1. Route validation
        origin = None
        destination = None
        if first_segment:
            origin = first_segment.get("departure_airport") or first_segment.get("origin")
        if last_segment:
            destination = last_segment.get("arrival_airport") or last_segment.get("destination")

        if not origin:
            origin = data.get("origin") or data.get("departure_airport")
        if not destination:
            destination = data.get("destination") or data.get("arrival_airport")

        if origin and origin.upper() != self.origin.upper():
            return False
        if destination and destination.upper() != self.destination.upper():
            return False

        # 2. Date and Departure Time validation
        dep_time_raw = None
        if first_segment:
            dep_time_raw = first_segment.get("departure_time")
        if not dep_time_raw:
            dep_time_raw = data.get("departure_time")

        if dep_time_raw:
            # Can be "2026-10-07T08:15:00" or "08:15"
            if "T" in dep_time_raw:
                dep_date, dep_time = dep_time_raw.split("T", 1)
                dep_time = dep_time[:5]
                if dep_date != self.date:
                    return False
            else:
                dep_time = dep_time_raw[:5]

            # Compare departure time window
            if not (self.depart_after <= dep_time <= self.depart_before):
                return False

        # 3. Price validation
        price_obj = data.get("price") or data.get("price_breakdown") or {}
        if isinstance(price_obj, dict):
            raw_price = price_obj.get("total_fare", price_obj.get("total", 0.0))
            currency = price_obj.get("currency", "USD")
        elif hasattr(price_obj, "total_fare"):
            raw_price = price_obj.total_fare
            currency = getattr(price_obj, "currency", "USD")
        else:
            raw_price = data.get("total_price", data.get("price", 0.0))
            currency = data.get("currency", "USD")

        try:
            norm_price = self.normalize_price(float(raw_price), currency)
            if norm_price > float(self.max_price):
                return False
        except (ValueError, TypeError):
            pass

        # 4. Passenger details validation (if travelers present in booking)
        travelers = data.get("travelers")
        return not (
            travelers is not None
            and isinstance(travelers, list)
            and len(travelers) > 0
            and len(travelers) != self.passenger_count
        )


def check_permission(tool_name: str, args: dict, constraints: Constraints, db_instance=None):
    """Permission Check (Kiểm quyền).

    Intercepts calls BEFORE tools execute.
    Blocks the call if it violates data constraints.
    Returns:
        tuple (is_allowed: bool, reason: str | None)
    """
    database = db_instance or db

    if tool_name == "create_flight_booking":
        offer_id = args.get("offer_id")
        if not offer_id:
            return False, "Missing offer_id in create_flight_booking."

        offer = database.get_flight_details(offer_id)
        if not offer:
            return False, f"Offer {offer_id} not found in database or expired."

        if not constraints.is_ok(offer):
            return False, f"{offer_id} breaks the constraints: {constraints.to_prompt()}"

        travelers = args.get("travelers", [])
        if travelers and len(travelers) != constraints.passenger_count:
            return (
                False,
                f"Traveler count {len(travelers)} does not match constraint {constraints.passenger_count}",
            )

    return True, None


def is_done(session_id: str, constraints: Constraints, db_instance=None) -> bool:
    """Done checked by CODE (Tiêu chí hoàn thành kiểm bằng code).

    Directly queries the database for session_id.
    Never trusts LLM text responses.
    Verifies:
    1. Booking record actually exists.
    2. Status is CONFIRMED.
    3. The booked flight strictly satisfies all data constraints.
    """
    database = db_instance or db
    session_orders = database._orders_by_session.get(session_id, {})

    for order in session_orders.values():
        status = getattr(order, "status", None)
        if hasattr(status, "value"):
            status_str = str(status.value).upper()
        else:
            status_str = str(status or "").upper()

        if status_str == "CONFIRMED" and constraints.is_ok(order):
            return True

    return False


def handoff(session_id: str, log: list, constraints: Constraints, db_instance=None) -> dict:
    """Structured Handoff to Human (Bàn giao có cấu trúc).

    Returns:
        dict containing:
        - 'done_so_far': Current bookings held or paid in the database
        - 'tried': Tools called, arguments, and outcomes
        - 'question': Clear question clarifying which constraint to relax
    """
    database = db_instance or db
    session_orders = database._orders_by_session.get(session_id, {})

    done_so_far = []
    seen_pnrs = set()
    for o in session_orders.values():
        pnr = getattr(o, "booking_reference", None) or (
            o.get("booking_reference") if isinstance(o, dict) else None
        )
        if pnr and pnr not in seen_pnrs:
            seen_pnrs.add(pnr)
            if hasattr(o, "model_dump"):
                done_so_far.append(o.model_dump())
            elif isinstance(o, dict):
                done_so_far.append(o)
            else:
                done_so_far.append(str(o))

    tried = []
    for step in log:
        if hasattr(step, "step_type"):
            tried.append(
                {
                    "step_type": step.step_type,
                    "name": getattr(step, "name", ""),
                    "input": getattr(step, "input_data", None),
                    "output": getattr(step, "output_data", None),
                }
            )
        elif isinstance(step, dict):
            tried.append(step)

    question = (
        f"No available flight matches all constraints ({constraints.to_prompt()}). "
        "Which constraint can be relaxed: departure time window or maximum price?"
    )

    return {
        "done_so_far": done_so_far,
        "tried": tried,
        "question": question,
    }


def is_looping(log: list, threshold: int = 3) -> bool:
    """Detects infinite loops by tracking consecutive or repeated identical tool calls."""
    if not log or len(log) < threshold:
        return False

    tool_calls = []
    for step in log:
        if hasattr(step, "step_type") and step.step_type == "tool_call":
            tool_calls.append((getattr(step, "name", ""), str(getattr(step, "input_data", ""))))
        elif isinstance(step, dict) and step.get("step_type") == "tool_call":
            tool_calls.append((step.get("name", ""), str(step.get("input", ""))))

    if len(tool_calls) < threshold:
        return False

    # Check last `threshold` calls for exact repetition
    last_window = tool_calls[-threshold:]
    if len(set(last_window)) == 1:
        return True

    # Check total occurrences of identical call
    counts = {}
    for tc in tool_calls:
        counts[tc] = counts.get(tc, 0) + 1
        if counts[tc] >= threshold:
            return True

    return False


def check_hallucination(final_text: str, log: list) -> list:
    """Tool Hallucination Detection.

    Checks flight numbers mentioned in agent's final text against the tool result log.
    Returns list of ungrounded flight numbers.
    """
    if not final_text:
        return []

    # Find flight numbers like VN-120, BA-304
    mentioned_flights = set(re.findall(r"\b[A-Z]{2}-\d{3,4}\b", final_text))
    if not mentioned_flights:
        return []

    # Gather all text in tool results
    log_text = ""
    for step in log:
        out = getattr(step, "output_data", None) if hasattr(step, "output_data") else step.get("output", "")
        log_text += f" {out!s}"

    hallucinated = [f for f in mentioned_flights if f not in log_text]
    return hallucinated


def is_empty_or_error(tool_output) -> bool:
    """State Corruption Detection.

    Ensures empty query responses from network/mock glitches are detected.
    """
    if tool_output is None:
        return True
    if isinstance(tool_output, str):
        cleaned = tool_output.strip()
        if not cleaned or cleaned in ("{}", "[]", "null"):
            return True
        if "error" in cleaned.lower() and "exception" in cleaned.lower():
            return True
    elif isinstance(tool_output, (dict, list)) and len(tool_output) == 0:
        return True
    return False


def wrap_tool_with_harness(tool_func, constraints: Constraints, execution_log: list, db_instance=None):
    """Wraps a LangChain tool with Harness permission check and loop detector."""
    orig_name = getattr(tool_func, "name", tool_func.__name__)
    orig_desc = getattr(tool_func, "description", "")

    def wrapped_func(**kwargs):
        # 1. Permission check
        allowed, reason = check_permission(orig_name, kwargs, constraints, db_instance)
        if not allowed:
            denial_msg = f"DENIED by Harness Permission Check: {reason}"
            return json.dumps({"status": "denied", "error": denial_msg})

        # 2. Loop check before running
        if is_looping(execution_log, threshold=3):
            raise StopAgentException("LOOP: Repeated tool call detected by Harness.")

        # 3. Execute underlying tool
        result = tool_func.invoke(kwargs) if hasattr(tool_func, "invoke") else tool_func(**kwargs)

        # 4. State corruption check
        if is_empty_or_error(result):
            return json.dumps({
                "status": "warning",
                "message": "Tool returned empty or malformed response; state validation flagged.",
            })

        return result

    return StructuredTool.from_function(
        func=wrapped_func,
        name=orig_name,
        description=orig_desc,
        args_schema=getattr(tool_func, "args_schema", None),
    )
