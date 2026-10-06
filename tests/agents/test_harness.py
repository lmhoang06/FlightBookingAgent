"""Unit and integration tests for Harness Architecture and Scripted Fake Model."""

from langchain_core.messages import AIMessage

from agents.fake_model import ScriptedFakeChatModel
from agents.harness import (
    Constraints,
    check_hallucination,
    check_permission,
    handoff,
    is_done,
    is_empty_or_error,
    is_looping,
)
from tools.db import db
from tools.schemas import (
    BookingStatus,
    FlightSearchCriteria,
    TravelerInfo,
    TravelerType,
)


def test_constraints_dataclass_and_prompt():
    """Verify Constraints dataclass deterministic prompt and validation."""
    c = Constraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        depart_after="00:00",
        max_price=2_000_000,
        passenger_name="Nguyen Van A",
        passenger_count=1,
    )
    prompt = c.to_prompt()
    assert "SGN -> DAD" in prompt
    assert "2026-10-07" in prompt
    assert "2,000,000" in prompt
    assert "Nguyen Van A" in prompt

    # Flight offer matching constraints (VN-120: 08:15, price 70 USD * 25k = 1.75M <= 2M)
    valid_offer = {
        "origin": "SGN",
        "destination": "DAD",
        "departure_time": "2026-10-07T08:15:00",
        "price": {"total_fare": 70.0, "currency": "USD"},
    }
    assert c.is_ok(valid_offer) is True

    # Flight offer departing after 12:00 (VN-128: 17:45)
    late_offer = {
        "origin": "SGN",
        "destination": "DAD",
        "departure_time": "2026-10-07T17:45:00",
        "price": {"total_fare": 70.0, "currency": "USD"},
    }
    assert c.is_ok(late_offer) is False

    # Flight offer exceeding max_price (price 95 USD * 25k = 2.375M > 2M)
    expensive_offer = {
        "origin": "SGN",
        "destination": "DAD",
        "departure_time": "2026-10-07T09:00:00",
        "price": {"total_fare": 95.0, "currency": "USD"},
    }
    assert c.is_ok(expensive_offer) is False

    # Flight offer with wrong destination
    wrong_route = {
        "origin": "SGN",
        "destination": "HAN",
        "departure_time": "2026-10-07T08:15:00",
        "price": {"total_fare": 70.0, "currency": "USD"},
    }
    assert c.is_ok(wrong_route) is False


def test_permission_check_enforcement():
    """Verify check_permission blocks violations before tool executes."""
    c = Constraints(origin="SGN", destination="DAD", max_price=2_000_000, depart_before="12:00")

    # Search flights is read-only, should pass
    allowed, reason = check_permission("search_flights", {"origin": "SGN"}, c)
    assert allowed is True
    assert reason is None

    # Missing offer_id
    allowed, reason = check_permission("create_flight_booking", {}, c)
    assert allowed is False
    assert "Missing offer_id" in reason

    # Non-existent offer_id
    allowed, reason = check_permission("create_flight_booking", {"offer_id": "OFFER-FAKE-999"}, c)
    assert allowed is False
    assert "not found" in reason


def test_is_done_verified_by_code():
    """Verify is_done directly queries database and validates against constraints."""
    c = Constraints(origin="SGN", destination="DAD", date="2026-10-07", passenger_count=1)
    session_id = "test_is_done_session"

    # Initially no bookings in DB
    assert is_done(session_id, c) is False

    # Search flight offers in db
    search_res = db.search_flights(
        FlightSearchCriteria(
            origin="SGN",
            destination="DAD",
            departure_date="2026-10-07",
        )
    )
    assert len(search_res.offers) > 0
    offer = search_res.offers[0]

    # Create confirmed booking via DB
    order = db.create_flight_booking(
        offer_id=offer.offer_id,
        travelers=[
            TravelerInfo(
                traveler_id="PAX-1",
                traveler_type=TravelerType.ADULT,
                first_name="Nguyen Van",
                last_name="A",
                date_of_birth="1990-01-01",
            )
        ],
        contact_email="a@example.com",
        contact_phone="0901234567",
        session_id=session_id,
    )

    assert is_done(session_id, c) is True
    assert order.status == BookingStatus.CONFIRMED


def test_structured_handoff():
    """Verify handoff returns structured done_so_far, tried, and question."""
    c = Constraints(origin="SGN", destination="DAD", max_price=2_000_000)
    session_id = "test_handoff_session"

    log = [
        {"step_type": "tool_call", "name": "search_flights", "input": {"origin": "SGN", "destination": "DAD"}},
        {"step_type": "tool_result", "name": "search_flights", "output": "No flights found"},
    ]
    h = handoff(session_id, log, c)
    assert "done_so_far" in h
    assert "tried" in h
    assert "question" in h
    assert len(h["tried"]) == 2
    assert "Which constraint can be relaxed" in h["question"]


def test_failure_mode_mitigations():
    """Verify detection of Loop, Hallucination, and State Corruption."""
    # 1. Loop detection
    log_looping = [
        {"step_type": "tool_call", "name": "search_flights", "input": '{"query": "DAD"}'},
        {"step_type": "tool_call", "name": "search_flights", "input": '{"query": "DAD"}'},
        {"step_type": "tool_call", "name": "search_flights", "input": '{"query": "DAD"}'},
    ]
    assert is_looping(log_looping, threshold=3) is True

    log_normal = [
        {"step_type": "tool_call", "name": "search_airports", "input": '{"query": "DAD"}'},
        {"step_type": "tool_call", "name": "search_flights", "input": '{"origin": "SGN"}'},
    ]
    assert is_looping(log_normal, threshold=3) is False

    # 2. Tool Hallucination detection
    tool_log = [
        {"step_type": "tool_result", "output": "Available flight is VN-120 priced at $70."},
    ]
    # Agent mentions real flight VN-120 -> no hallucination
    assert check_hallucination("I booked VN-120 for you.", tool_log) == []
    # Agent invents flight VN-999 never mentioned in tool log -> flagged!
    hallucinated = check_hallucination("I booked flight VN-999 for you.", tool_log)
    assert "VN-999" in hallucinated

    # 3. State corruption detection
    assert is_empty_or_error(None) is True
    assert is_empty_or_error("") is True
    assert is_empty_or_error("{}") is True
    assert is_empty_or_error({"offers": []}) is False  # valid structure with data
    assert is_empty_or_error("Internal Error: server exception") is True


def test_scripted_fake_model_cases():
    """Verify ScriptedFakeChatModel trajectories for offline benchmarking."""
    # Case: Success
    model_success = ScriptedFakeChatModel(case="success")
    res1 = model_success.invoke("Book SGN to DAD")
    assert isinstance(res1, AIMessage)
    assert res1.tool_calls[0]["name"] == "search_flights"

    # Case: Loop
    model_loop = ScriptedFakeChatModel(case="fail_loop")
    r1 = model_loop.invoke("Book SGN to DAD")
    r2 = model_loop.invoke("Book SGN to DAD")
    r3 = model_loop.invoke("Book SGN to DAD")
    assert r1.tool_calls[0]["name"] == "search_flights"
    assert r2.tool_calls[0]["name"] == "search_flights"
    assert r3.tool_calls[0]["name"] == "search_flights"
