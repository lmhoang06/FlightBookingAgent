"""FlightBookingAgent tools package.

Exports LangChain tools, industry-standard flight distribution schemas,
and mock database engine for agentic flight searching, airport discovery,
offer quotation, booking creation, order inspection, and cancellation.
Follows SOLID principles and modular design.
"""

from .airport_tools import (
    get_airline_info,
    search_airports_and_cities,
)
from .booking_tools import (
    cancel_flight_booking,
    create_flight_booking,
    get_booking_details,
    get_session_status,
)
from .db import (
    MockFlightDatabase,
    db,
)
from .flight_search_tools import (
    get_flight_details,
    search_flights,
)
from .schemas import (
    AirlineBaggagePolicy,
    AirlineInfo,
    AirlinePricingPolicy,
    AirportInfo,
    AirportSearchResult,
    BookingCancellationResponse,
    BookingStatus,
    CancellationPolicy,
    DepartureTimeWindow,
    FlightItinerary,
    FlightOffer,
    FlightOrder,
    FlightSearchCriteria,
    FlightSearchResponse,
    FlightSegment,
    PriceBreakdown,
    SessionInfo,
    SortBy,
    TravelClass,
    TravelerInfo,
    TravelerPricing,
    TravelerType,
    TripType,
)

# Complete list of LangChain tools ready for model binding (e.g. llm.bind_tools(FLIGHT_BOOKING_TOOLS))
FLIGHT_BOOKING_TOOLS = [
    search_airports_and_cities,
    get_airline_info,
    search_flights,
    get_flight_details,
    create_flight_booking,
    get_booking_details,
    cancel_flight_booking,
    get_session_status,
]

# Alias for general use
FLIGHT_TOOLS = FLIGHT_BOOKING_TOOLS

__all__ = [
    # LangChain Tools
    "search_airports_and_cities",
    "get_airline_info",
    "search_flights",
    "get_flight_details",
    "create_flight_booking",
    "get_booking_details",
    "cancel_flight_booking",
    "get_session_status",
    "FLIGHT_BOOKING_TOOLS",
    "FLIGHT_TOOLS",
    # Database
    "MockFlightDatabase",
    "db",
    # Schemas
    "TravelClass",
    "TripType",
    "TravelerType",
    "BookingStatus",
    "SortBy",
    "DepartureTimeWindow",
    "AirportInfo",
    "AirportSearchResult",
    "AirlineInfo",
    "AirlinePricingPolicy",
    "AirlineBaggagePolicy",
    "FlightSegment",
    "FlightItinerary",
    "PriceBreakdown",
    "TravelerPricing",
    "CancellationPolicy",
    "FlightOffer",
    "FlightSearchCriteria",
    "FlightSearchResponse",
    "TravelerInfo",
    "FlightOrder",
    "BookingCancellationResponse",
    "SessionInfo",
]
