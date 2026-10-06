"""LangChain tools for Flight Booking, PNR Management, Cancellation, and Session Audit.

Follows the Single Responsibility Principle (SRP) and real-world airline distribution schemas:
- Generates 6-character airline Passenger Name Records (PNR).
- Enforces strict multi-tenant session isolation (a session never sees or cancels another user's bookings).
- Real-time seat inventory reservation and restoration upon cancellation.
- Dynamic penalty fees and net refund calculation according to the specific airline's fare rules.
"""

import json
from typing import Any

from langchain_core.tools import tool

from .db import db
from .schemas import TravelerInfo, TravelerType


@tool
def create_flight_booking(
    offer_id: str,
    travelers: list[dict[str, Any]],
    contact_email: str,
    contact_phone: str,
    payment_method: str = "CREDIT_CARD",
    selected_seats: list[str] | None = None,
    special_requests: str | None = None,
    session_id: str | None = None,
) -> str:
    """Create a confirmed flight booking (PNR) for the specified flight offer.

    Decrements real-time seat inventory and isolates the booking under the active session ID,
    guaranteeing that no other user or session can inspect or modify this order.

    Args:
        offer_id: Unique flight offer ID from search_flights (e.g. 'OFFER-BA-A1B2').
        travelers: List of passenger traveler objects. Each object should include:
            - 'first_name': str (e.g. 'Jane')
            - 'last_name': str (e.g. 'Doe')
            - 'date_of_birth': str in YYYY-MM-DD format (e.g. '1992-05-14')
            - 'traveler_type': optional str ('ADULT', 'CHILD', 'INFANT', default 'ADULT')
            - 'passport_number': optional str (required for international itineraries)
            - 'passport_expiry': optional str (YYYY-MM-DD)
            - 'nationality': optional str
            - 'email': optional str
            - 'phone': optional str
            - 'seat_preference': optional str ('WINDOW', 'AISLE', or specific code '14A')
            - 'meal_preference': optional str ('VEGETARIAN', 'HALAL', 'STANDARD', etc.)
        contact_email: Primary contact email for booking confirmation and e-ticket.
        contact_phone: Primary contact telephone number.
        payment_method: Payment method used (e.g., 'CREDIT_CARD', 'DEBIT_CARD', 'PAYPAL').
        selected_seats: Optional list of specific seat assignments (e.g., ['14A', '14B']).
        special_requests: Optional special assistance notes (e.g., 'Wheelchair assistance requested').
        session_id: Session ID that owns this booking. If None, auto-generated and returned in response.

    Returns:
        JSON string containing the confirmed FlightOrder, Passenger Name Record (PNR),
        assigned seats, fare breakdown, and cancellation terms.
    """
    try:
        session = db.get_or_create_session(session_id)

        # Parse traveler profiles
        traveler_models: list[TravelerInfo] = []
        for idx, t in enumerate(travelers):
            t_type_str = t.get("traveler_type", "ADULT").upper()
            t_type = (
                TravelerType(t_type_str)
                if t_type_str in TravelerType._value2member_map_
                else TravelerType.ADULT
            )
            traveler_models.append(
                TravelerInfo(
                    traveler_id=f"PAX-{idx + 1}",
                    traveler_type=t_type,
                    first_name=t["first_name"],
                    last_name=t["last_name"],
                    date_of_birth=t["date_of_birth"],
                    passport_number=t.get("passport_number"),
                    passport_expiry=t.get("passport_expiry"),
                    nationality=t.get("nationality"),
                    email=t.get("email", contact_email),
                    phone=t.get("phone", contact_phone),
                    seat_preference=t.get("seat_preference"),
                    meal_preference=t.get("meal_preference"),
                )
            )

        order = db.create_flight_booking(
            offer_id=offer_id,
            travelers=traveler_models,
            contact_email=contact_email,
            contact_phone=contact_phone,
            payment_method=payment_method,
            selected_seats=selected_seats,
            special_requests=special_requests,
            session_id=session.session_id,
        )

        return json.dumps(
            {
                "status": "success",
                "message": f"Booking successfully confirmed! PNR reference: {order.booking_reference}",
                "order": order.model_dump(),
            },
            indent=2,
        )

    except ValueError as ve:
        return json.dumps(
            {
                "status": "error",
                "message": f"Booking creation failed: {ve!s}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Unexpected error during booking creation: {e!s}",
            },
            indent=2,
        )


@tool
def get_booking_details(
    booking_reference: str,
    session_id: str | None = None,
) -> str:
    """Retrieve full details of an existing flight booking by PNR or Order ID.

    Enforces strict multi-tenant privacy: only bookings created within the same session ID
    can be retrieved. Bookings from other users or sessions are strictly protected and inaccessible.

    Args:
        booking_reference: 6-character PNR code (e.g. 'BA9A2K') or order ID (e.g. 'ORD-3F8B').
        session_id: The session ID of the client requesting the booking.

    Returns:
        JSON string containing the FlightOrder record, or an error if not found in the session.
    """
    try:
        session = db.get_or_create_session(session_id)
        order = db.get_booking_details(booking_reference, session.session_id)

        if not order:
            return json.dumps(
                {
                    "status": "error",
                    "session_id": session.session_id,
                    "message": (
                        f"Booking '{booking_reference}' not found in current session '{session.session_id}'. "
                        "Note: To protect privacy, bookings from other sessions cannot be viewed."
                    ),
                },
                indent=2,
            )

        return json.dumps(
            {
                "status": "success",
                "session_id": session.session_id,
                "order": order.model_dump(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to retrieve booking: {e!s}",
            },
            indent=2,
        )


@tool
def cancel_flight_booking(
    booking_reference: str,
    reason: str | None = None,
    session_id: str | None = None,
) -> str:
    """Cancel an active flight booking, release seats back to inventory, and compute refund per policy.

    Strictly isolated to the current session ID. Cannot cancel bookings belonging to other sessions.

    Args:
        booking_reference: 6-character PNR code or order ID to cancel.
        reason: Optional user-provided reason for cancellation.
        session_id: Active session ID for authorization.

    Returns:
        JSON string containing cancellation status, refund amount, deduction fees, and confirmation.
    """
    try:
        session = db.get_or_create_session(session_id)
        response = db.cancel_flight_booking(
            booking_reference=booking_reference,
            reason=reason,
            session_id=session.session_id,
        )
        return json.dumps(
            {
                "status": "success",
                "cancellation": response.model_dump(),
            },
            indent=2,
        )

    except ValueError as ve:
        return json.dumps(
            {
                "status": "error",
                "message": f"Cancellation rejected: {ve!s}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to cancel booking: {e!s}",
            },
            indent=2,
        )


@tool
def get_session_status(
    session_id: str | None = None,
) -> str:
    """Inspect or verify the current session state, active bookings count, and audit trail.

    Helps agents running in ReAct, Plan-then-Execute, or Hybrid mode inspect tool call integrity,
    ensure session persistence, and verify booking state across multi-turn reasoning steps.

    Args:
        session_id: Session ID to inspect. If None, returns the active session or auto-generates one.

    Returns:
        JSON string containing session metadata and active booking count.
    """
    try:
        status_info = db.get_session_status(session_id)
        return json.dumps(
            {
                "status": "success",
                **status_info,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to get session status: {e!s}",
            },
            indent=2,
        )
