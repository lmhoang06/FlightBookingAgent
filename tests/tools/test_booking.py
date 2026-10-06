"""Unit tests for flight booking, session isolation, and cancellation tools."""

import json

from tools.booking_tools import (
    cancel_flight_booking,
    create_flight_booking,
    get_booking_details,
    get_session_status,
)
from tools.flight_search_tools import search_flights


def test_create_flight_booking_success() -> None:
    """Creating a flight booking must confirm the order and generate a 6-character PNR."""
    search_raw = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-15",
            "session_id": "test_session_create",
        }
    )
    search_data = json.loads(search_raw)
    offer_id = search_data["offers"][0]["offer_id"]

    booking_raw = create_flight_booking.invoke(
        {
            "offer_id": offer_id,
            "travelers": [
                {
                    "first_name": "Jane",
                    "last_name": "Doe",
                    "date_of_birth": "1990-05-12",
                    "traveler_type": "ADULT",
                    "passport_number": "P12345678",
                    "seat_preference": "WINDOW",
                }
            ],
            "contact_email": "jane.doe@example.com",
            "contact_phone": "+44123456789",
            "session_id": "test_session_create",
        }
    )
    booking_data = json.loads(booking_raw)

    assert booking_data["status"] == "success"
    order = booking_data["order"]
    assert order["status"] == "CONFIRMED"
    assert len(order["booking_reference"]) == 6
    assert order["session_id"] == "test_session_create"
    assert "Jane Doe" in order["seat_assignments"]
    assert order["seat_assignments"]["Jane Doe"].endswith("A")  # WINDOW seat


def test_booking_inventory_decrement_and_restoration() -> None:
    """Booking seats must decrement available inventory; cancelling must restore it."""
    date_str = "2026-10-18"
    search_1 = json.loads(
        search_flights.invoke(
            {
                "origin": "LHR",
                "destination": "CDG",
                "departure_date": date_str,
                "session_id": "test_session_inv",
            }
        )
    )
    offer_1 = search_1["offers"][0]
    initial_seats = offer_1["available_seats"]

    # Book 2 passengers
    booking_raw = create_flight_booking.invoke(
        {
            "offer_id": offer_1["offer_id"],
            "travelers": [
                {
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "date_of_birth": "1988-02-14",
                },
                {
                    "first_name": "Bob",
                    "last_name": "Smith",
                    "date_of_birth": "1985-08-20",
                },
            ],
            "contact_email": "alice@example.com",
            "contact_phone": "+447700900077",
            "session_id": "test_session_inv",
        }
    )
    booking_data = json.loads(booking_raw)
    pnr = booking_data["order"]["booking_reference"]

    # Verify inventory decremented by 2
    search_2 = json.loads(
        search_flights.invoke(
            {
                "origin": "LHR",
                "destination": "CDG",
                "departure_date": date_str,
                "session_id": "test_session_inv",
            }
        )
    )
    assert search_2["offers"][0]["available_seats"] == initial_seats - 2

    # Cancel booking
    cancel_raw = cancel_flight_booking.invoke(
        {
            "booking_reference": pnr,
            "session_id": "test_session_inv",
        }
    )
    cancel_data = json.loads(cancel_raw)
    assert cancel_data["status"] == "success"

    # Verify inventory fully restored
    search_3 = json.loads(
        search_flights.invoke(
            {
                "origin": "LHR",
                "destination": "CDG",
                "departure_date": date_str,
                "session_id": "test_session_inv",
            }
        )
    )
    assert search_3["offers"][0]["available_seats"] == initial_seats


def test_multi_tenant_session_isolation() -> None:
    """User B cannot view or cancel User A's booking across different sessions."""
    # Alice books in session_alice
    search_alice = json.loads(
        search_flights.invoke(
            {
                "origin": "LHR",
                "destination": "CDG",
                "departure_date": "2026-10-15",
                "session_id": "session_alice",
            }
        )
    )
    booking_alice = json.loads(
        create_flight_booking.invoke(
            {
                "offer_id": search_alice["offers"][0]["offer_id"],
                "travelers": [
                    {
                        "first_name": "Alice",
                        "last_name": "Wonderland",
                        "date_of_birth": "1995-04-10",
                    }
                ],
                "contact_email": "alice@wonder.com",
                "contact_phone": "+1234567890",
                "session_id": "session_alice",
            }
        )
    )
    alice_pnr = booking_alice["order"]["booking_reference"]

    # Bob in session_bob tries to inspect Alice's booking -> Rejected
    bob_view = json.loads(
        get_booking_details.invoke(
            {
                "booking_reference": alice_pnr,
                "session_id": "session_bob",
            }
        )
    )
    assert bob_view["status"] == "error"
    assert "not found in current session 'session_bob'" in bob_view["message"]

    # Bob in session_bob tries to cancel Alice's booking -> Rejected
    bob_cancel = json.loads(
        cancel_flight_booking.invoke(
            {
                "booking_reference": alice_pnr,
                "session_id": "session_bob",
            }
        )
    )
    assert bob_cancel["status"] == "error"
    assert "not found in current session" in bob_cancel["message"]

    # Alice in session_alice accesses her own booking -> Success
    alice_view = json.loads(
        get_booking_details.invoke(
            {
                "booking_reference": alice_pnr,
                "session_id": "session_alice",
            }
        )
    )
    assert alice_view["status"] == "success"
    assert alice_view["order"]["booking_reference"] == alice_pnr


def test_cancel_flight_booking_refund_calculation() -> None:
    """Cancellation must calculate net refund and fees per airline policy and prevent duplicate cancellations."""
    session_id = "test_session_cancel_calc"
    search_res = json.loads(
        search_flights.invoke(
            {
                "origin": "LHR",
                "destination": "CDG",
                "departure_date": "2026-10-15",
                "session_id": session_id,
            }
        )
    )
    booking_res = json.loads(
        create_flight_booking.invoke(
            {
                "offer_id": search_res["offers"][0]["offer_id"],
                "travelers": [
                    {
                        "first_name": "Tom",
                        "last_name": "Hanks",
                        "date_of_birth": "1960-07-09",
                    }
                ],
                "contact_email": "tom@hollywood.com",
                "contact_phone": "+1987654321",
                "session_id": session_id,
            }
        )
    )
    pnr = booking_res["order"]["booking_reference"]
    total_fare = booking_res["order"]["price_breakdown"]["total_fare"]
    cancellation_fee = booking_res["order"]["cancellation_policy"]["cancellation_fee"]
    expected_refund = max(0.0, round(total_fare - cancellation_fee, 2))

    cancel_res = json.loads(
        cancel_flight_booking.invoke(
            {
                "booking_reference": pnr,
                "reason": "Change of travel plans",
                "session_id": session_id,
            }
        )
    )

    assert cancel_res["status"] == "success"
    cancellation = cancel_res["cancellation"]
    assert cancellation["booking_reference"] == pnr
    assert cancellation["status"] == "CANCELLED"
    assert cancellation["cancellation_fee"] == cancellation_fee
    assert cancellation["refund_amount"] == expected_refund

    # Double-cancellation must be rejected
    double_cancel = json.loads(
        cancel_flight_booking.invoke(
            {
                "booking_reference": pnr,
                "session_id": session_id,
            }
        )
    )
    assert double_cancel["status"] == "error"
    assert "already been cancelled" in double_cancel["message"]


def test_get_session_status() -> None:
    """Inspecting session status reflects active confirmed orders and audit counters."""
    session_id = "test_session_status_audit"
    search_res = json.loads(
        search_flights.invoke(
            {
                "origin": "LHR",
                "destination": "CDG",
                "departure_date": "2026-10-15",
                "session_id": session_id,
            }
        )
    )
    create_flight_booking.invoke(
        {
            "offer_id": search_res["offers"][0]["offer_id"],
            "travelers": [
                {
                    "first_name": "Clark",
                    "last_name": "Kent",
                    "date_of_birth": "1980-06-18",
                }
            ],
            "contact_email": "clark@dailyplanet.com",
            "contact_phone": "+1555123456",
            "session_id": session_id,
        }
    )

    status_raw = get_session_status.invoke({"session_id": session_id})
    status_data = json.loads(status_raw)

    assert status_data["status"] == "success"
    assert status_data["session_id"] == session_id
    assert status_data["searches_executed"] >= 1
    assert status_data["active_confirmed_bookings_count"] == 1
    assert len(status_data["active_pnr_references"]) == 1
    assert "ENFORCED" in status_data["isolation_status"]

def test_create_flight_booking_insufficient_seats() -> None:
    """Booking should fail gracefully if requested passengers exceed available seats."""
    session_id = "test_session_insufficient"
    search_res = json.loads(search_flights.invoke({
        "origin": "LHR",
        "destination": "CDG",
        "departure_date": "2026-10-15",
        "session_id": session_id,
    }))
    offer = search_res["offers"][0]
    
    # Try to book more travelers than available seats
    travelers = [
        {"first_name": f"Pax{i}", "last_name": "Test", "date_of_birth": "1990-01-01"}
        for i in range(offer["available_seats"] + 1)
    ]
    
    booking_raw = create_flight_booking.invoke({
        "offer_id": offer["offer_id"],
        "travelers": travelers,
        "contact_email": "test@example.com",
        "contact_phone": "+1234567890",
        "session_id": session_id,
    })
    
    booking_data = json.loads(booking_raw)
    assert booking_data["status"] == "error"
    assert "Booking creation failed" in booking_data["message"]
