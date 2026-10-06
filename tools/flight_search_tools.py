"""LangChain tools for Flight Search and Offer Details Inspection.

Follows industry standards (IATA NDC / Amadeus v2 / Duffel API).
Provides complete multi-criteria search without simplification.
Integrates actionable error messages and hints when parameters are invalid.
"""

import json

from langchain_core.tools import tool

from .db import db
from .schemas import (
    DepartureTimeWindow,
    FlightSearchCriteria,
    SortBy,
    TravelClass,
    TripType,
)


@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    trip_type: str = "ONE_WAY",
    travel_class: str = "ECONOMY",
    adults: int = 1,
    children: int = 0,
    infants: int = 0,
    direct_flights_only: bool = False,
    max_stops: int | None = None,
    included_airlines: list[str] | None = None,
    excluded_airlines: list[str] | None = None,
    departure_time_window: str = "ANY",
    earliest_departure_time: str | None = None,
    latest_departure_time: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    currency: str = "USD",
    requires_checked_bag: bool = False,
    sort_by: str = "BEST_VALUE",
    limit: int = 5,
    offset: int = 0,
    session_id: str | None = None,
) -> str:
    """Search for flight offers matching comprehensive travel criteria using real API standard data.

    Supports direct and multi-segment connecting flights with dynamic, airline-specific pricing.
    NOTE: 'origin' and 'destination' MUST be 3-letter IATA airport codes (e.g. 'LHR', 'CDG', 'SGN', 'JFK').
    If you only have a city name (e.g. 'London', 'Paris') or aren't sure of the exact airport code,
    call 'search_airports_and_cities' FIRST to discover the valid IATA codes.

    Args:
        origin: 3-letter departure IATA airport code (e.g., 'LHR', 'CDG', 'SGN', 'JFK').
        destination: 3-letter destination IATA airport code (e.g., 'CDG', 'HND', 'HAN').
        departure_date: Outbound travel date in YYYY-MM-DD format (e.g. '2026-10-15').
        return_date: Return travel date in YYYY-MM-DD format (required if trip_type is 'ROUND_TRIP').
        trip_type: 'ONE_WAY' or 'ROUND_TRIP' (default: 'ONE_WAY').
        travel_class: Cabin class: 'ECONOMY', 'PREMIUM_ECONOMY', 'BUSINESS', or 'FIRST' (default: 'ECONOMY').
        adults: Number of adult passengers (age 12+, default: 1).
        children: Number of child passengers (age 2-11, default: 0).
        infants: Number of infant passengers (under 2 years, default: 0).
        direct_flights_only: If True, filters strictly for non-stop direct flights (default: False).
        max_stops: Maximum number of intermediate layover stops allowed (0 for direct, 1, or 2).
        included_airlines: Optional list of airline IATA codes to restrict results to (e.g. ['BA', 'VN']).
        excluded_airlines: Optional list of airline IATA codes to exclude from results.
        departure_time_window: Time-of-day preference: 'ANY', 'MORNING' (06:00-12:00), 'AFTERNOON' (12:00-18:00), 'EVENING' (18:00-24:00), or 'NIGHT' (00:00-06:00).
        earliest_departure_time: Filter for flights departing at or after this time in HH:MM format (e.g. '08:00').
        latest_departure_time: Filter for flights departing at or before this time in HH:MM format (e.g. '20:00').
        min_price: Minimum total budget limit in specified currency.
        max_price: Maximum total budget limit in specified currency.
        currency: Desired fare currency (default: 'USD').
        requires_checked_bag: If True, only returns offers with free checked baggage included (default: False).
        sort_by: Ordering strategy: 'PRICE_ASC', 'DURATION_ASC', 'DEPARTURE_ASC', 'ARRIVAL_ASC', or 'BEST_VALUE' (default: 'BEST_VALUE').
        limit: Max offers to return per page (default: 5, max: 50).
        offset: Pagination offset index (default: 0).
        session_id: Active session ID for multi-tenant isolation and audit tracing. If None, auto-assigned.

    Returns:
        JSON string representing FlightSearchResponse with matching flight offers, price breakdowns, and itinerary segments.
    """
    try:
        # Validate enums with helpful error messaging
        try:
            parsed_trip_type = TripType(trip_type.upper())
        except ValueError:
            valid_trip_types = ", ".join([e.value for e in TripType])
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid trip_type '{trip_type}'. Must be one of: {valid_trip_types}.",
                },
                indent=2,
            )

        try:
            parsed_travel_class = TravelClass(travel_class.upper())
        except ValueError:
            valid_classes = ", ".join([e.value for e in TravelClass])
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid travel_class '{travel_class}'. Must be one of: {valid_classes}.",
                },
                indent=2,
            )

        try:
            parsed_time_window = DepartureTimeWindow(departure_time_window.upper())
        except ValueError:
            valid_windows = ", ".join([e.value for e in DepartureTimeWindow])
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid departure_time_window '{departure_time_window}'. Must be one of: {valid_windows}.",
                },
                indent=2,
            )

        try:
            parsed_sort_by = SortBy(sort_by.upper())
        except ValueError:
            valid_sorts = ", ".join([e.value for e in SortBy])
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid sort_by '{sort_by}'. Must be one of: {valid_sorts}.",
                },
                indent=2,
            )

        if parsed_trip_type == TripType.ROUND_TRIP and not return_date:
            return json.dumps(
                {
                    "status": "error",
                    "message": "return_date in YYYY-MM-DD format is required when trip_type is ROUND_TRIP.",
                },
                indent=2,
            )

        criteria = FlightSearchCriteria(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            trip_type=parsed_trip_type,
            travel_class=parsed_travel_class,
            adults=adults,
            children=children,
            infants=infants,
            direct_flights_only=direct_flights_only,
            max_stops=max_stops,
            included_airlines=included_airlines,
            excluded_airlines=excluded_airlines,
            departure_time_window=parsed_time_window,
            earliest_departure_time=earliest_departure_time,
            latest_departure_time=latest_departure_time,
            min_price=min_price,
            max_price=max_price,
            currency=currency,
            requires_checked_bag=requires_checked_bag,
            sort_by=parsed_sort_by,
            limit=limit,
            offset=offset,
            session_id=session_id,
        )

        response = db.search_flights(criteria)
        return json.dumps(response.model_dump(), indent=2)

    except ValueError as ve:
        return json.dumps(
            {
                "status": "error",
                "message": f"Flight search validation error: {ve!s}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Unexpected error during flight search: {e!s}",
            },
            indent=2,
        )


@tool
def get_flight_details(
    offer_id: str,
    session_id: str | None = None,
) -> str:
    """Retrieve detailed quotation and service specifications for a specific flight offer ID.

    Provides in-depth baggage allowance rules, dynamic fare breakdown
    (base fare, taxes, airline-specific multipliers), and cancellation/change policies.

    Args:
        offer_id: Unique flight offer ID returned from search_flights (e.g. 'OFFER-BA-A1B2').
        session_id: Optional session identifier for tracking and session continuity.

    Returns:
        JSON string containing the full FlightOffer specifications or an error message if not found.
    """
    try:
        offer = db.get_flight_details(offer_id)
        if not offer:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"Flight offer '{offer_id}' not found or expired. "
                        "Flight quotes expire after 2 hours. Please perform search_flights again to receive fresh offers."
                    ),
                },
                indent=2,
            )

        return json.dumps(
            {
                "status": "success",
                "session_id": session_id,
                "offer": offer.model_dump(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to retrieve flight details: {e!s}",
            },
            indent=2,
        )
