"""Pydantic schemas and standard flight distribution models.

Adheres to real-world airline distribution schemas (IATA NDC / Amadeus v2 / Duffel).
Provides rich type annotations for agentic workflows (ReAct, Plan-then-Execute, Hybrid).
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TravelClass(str, Enum):
    """Cabin travel class."""

    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"


class TripType(str, Enum):
    """Flight trip type."""

    ONE_WAY = "ONE_WAY"
    ROUND_TRIP = "ROUND_TRIP"


class TravelerType(str, Enum):
    """Traveler age category for dynamic pricing and ticketing."""

    ADULT = "ADULT"  # Age 12+
    CHILD = "CHILD"  # Age 2-11
    INFANT = "INFANT"  # Under 2 years old (lap infant)


class SortBy(str, Enum):
    """Sorting strategy for flight search results."""

    PRICE_ASC = "PRICE_ASC"  # Lowest total fare first
    DURATION_ASC = "DURATION_ASC"  # Shortest flight duration first
    DEPARTURE_ASC = "DEPARTURE_ASC"  # Earliest departure timestamp first
    ARRIVAL_ASC = "ARRIVAL_ASC"  # Earliest arrival timestamp first
    BEST_VALUE = "BEST_VALUE"  # Weighted balance between price and duration


class DepartureTimeWindow(str, Enum):
    """Departure time of day preference."""

    ANY = "ANY"
    MORNING = "MORNING"  # 06:00 - 11:59
    AFTERNOON = "AFTERNOON"  # 12:00 - 17:59
    EVENING = "EVENING"  # 18:00 - 23:59
    NIGHT = "NIGHT"  # 00:00 - 05:59


# --- Airport and Airline Reference Models ---


class AirportInfo(BaseModel):
    """Airport details adhering to IATA standard specifications."""

    iata_code: str = Field(
        description="3-letter IATA airport code, e.g. LHR, CDG, SGN, JFK"
    )
    name: str = Field(description="Full commercial airport name")
    city: str = Field(description="Metropolitan city served by airport")
    city_code: str | None = Field(
        default=None,
        description="Metropolitan area code (e.g. LON for London, PAR for Paris, NYC for New York)",
    )
    country: str = Field(description="Country name")
    country_code: str = Field(
        description="2-letter ISO country code, e.g. GB, FR, VN, US"
    )
    timezone: str = Field(
        description="IANA timezone identifier, e.g. Europe/London, Asia/Ho_Chi_Minh"
    )
    terminals: list[str] = Field(
        default_factory=list, description="Operational passenger terminals"
    )


class AirportSearchResult(BaseModel):
    """Structured response returned by airport/city search."""

    query: str = Field(description="The original user/agent search term")
    matches_found: int = Field(description="Number of airports matched")
    airports: list[AirportInfo] = Field(
        default_factory=list, description="Matching airports"
    )
    suggestion: str | None = Field(
        default=None, description="Fuzzy correction suggestion if a typo is suspected"
    )
    message: str = Field(description="Human/agent-readable summary of the lookup")


class AirlineBaggagePolicy(BaseModel):
    """Airline baggage allowance policy."""

    cabin_bags_included: int = Field(
        default=1, description="Number of complimentary carry-on bags"
    )
    cabin_bag_weight_kg: float = Field(
        default=7.0, description="Maximum cabin bag weight in kg"
    )
    checked_bags_included: int = Field(
        default=1, description="Number of complimentary checked bags"
    )
    checked_bag_weight_kg: float = Field(
        default=23.0, description="Standard checked bag weight limit in kg"
    )


class AirlinePricingPolicy(BaseModel):
    """Airline dynamic pricing factors loaded from airline metadata."""

    cabin_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "ECONOMY": 1.0,
            "PREMIUM_ECONOMY": 1.45,
            "BUSINESS": 2.4,
            "FIRST": 4.0,
        },
        description="Fare multiplier per cabin class",
    )
    child_discount_ratio: float = Field(
        default=0.25,
        description="Discount ratio for child tickets (e.g. 0.25 = 25% off adult fare)",
    )
    infant_discount_ratio: float = Field(
        default=0.90,
        description="Discount ratio for infant tickets (e.g. 0.90 = 90% off adult fare)",
    )
    seat_selection_fee: float = Field(
        default=15.0, description="Advance seat selection surcharge"
    )
    extra_checked_bag_fee: float = Field(
        default=45.0, description="Fee for an extra checked piece"
    )
    tax_and_fee_rate: float = Field(
        default=0.14, description="Applicable tax and airport authority fee rate"
    )


class AirlineInfo(BaseModel):
    """Airline company metadata and service rules."""

    iata_code: str = Field(
        description="2-letter IATA airline code, e.g. VN, SQ, BA, JL"
    )
    name: str = Field(description="Official airline name")
    alliance: str | None = Field(
        default=None,
        description="Global airline alliance, e.g. SkyTeam, Star Alliance, oneworld",
    )
    baggage_policy: AirlineBaggagePolicy = Field(default_factory=AirlineBaggagePolicy)
    pricing_policy: AirlinePricingPolicy = Field(default_factory=AirlinePricingPolicy)


# --- Flight Offers, Itinerary, and Pricing Models ---


class FlightSegment(BaseModel):
    """Individual flight leg / segment in an itinerary."""

    segment_id: str = Field(description="Unique segment identifier")
    carrier_code: str = Field(description="2-letter IATA operating carrier code")
    carrier_name: str = Field(description="Operating airline name")
    flight_number: str = Field(description="Commercial flight number (e.g. BA-112)")
    aircraft: str = Field(
        description="Aircraft equipment model, e.g. Airbus A350-900, Boeing 787-9"
    )
    departure_airport: str = Field(description="Departure 3-letter IATA airport code")
    departure_terminal: str | None = Field(
        default=None, description="Departure terminal"
    )
    departure_time: str = Field(
        description="Local departure timestamp in ISO 8601 with timezone offset (e.g. '2026-10-15T19:45:00+09:00')"
    )
    departure_timezone: str = Field(
        description="IANA timezone identifier of departure airport (e.g. 'Asia/Tokyo', 'Europe/London')"
    )
    arrival_airport: str = Field(description="Arrival 3-letter IATA airport code")
    arrival_terminal: str | None = Field(default=None, description="Arrival terminal")
    arrival_time: str = Field(
        description="Local arrival timestamp in ISO 8601 with timezone offset (e.g. '2026-10-15T13:20:00-07:00')"
    )
    arrival_timezone: str = Field(
        description="IANA timezone identifier of arrival airport (e.g. 'America/Los_Angeles', 'Europe/Paris')"
    )
    duration_minutes: int = Field(description="Flight airtime in minutes")
    cabin_class: TravelClass = Field(
        default=TravelClass.ECONOMY, description="Class of service booked"
    )
    stops: int = Field(default=0, description="Technical intermediate stops")
    layover_airport: str | None = Field(
        default=None, description="Connecting transit airport code if layover"
    )
    layover_duration_minutes: int = Field(
        default=0, description="Transit layover waiting time in minutes"
    )


class FlightItinerary(BaseModel):
    """One direction itinerary (outbound or return) containing one or more flight segments."""

    itinerary_id: str = Field(description="Itinerary identifier")
    direction: str = Field(default="OUTBOUND", description="'OUTBOUND' or 'INBOUND'")
    total_duration_minutes: int = Field(
        description="Total travel time in minutes including layovers"
    )
    total_duration_iso: str = Field(
        description="ISO 8601 travel duration (e.g. PT6H30M)"
    )
    stop_count: int = Field(
        default=0, description="Total layovers / stops on this route"
    )
    segments: list[FlightSegment] = Field(description="Ordered list of flight segments")


class PriceBreakdown(BaseModel):
    """Transparent pricing breakdown reflecting airline policy."""

    currency: str = Field(default="USD", description="Currency ISO 4217 code")
    base_fare: float = Field(
        description="Base ticket fare before government/airport taxes"
    )
    taxes_and_fees: float = Field(description="Government, security, and airport taxes")
    baggage_fees: float = Field(
        default=0.0, description="Checked baggage additional fees"
    )
    seat_fees: float = Field(default=0.0, description="Seat selection surcharges")
    total_fare: float = Field(description="All-inclusive final payable fare")
    pricing_factors_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic factors retrieved from airline/flight metadata",
    )


class TravelerPricing(BaseModel):
    """Itemized pricing breakdown per passenger category."""

    traveler_type: TravelerType
    count: int
    fare_per_passenger: float
    taxes_per_passenger: float
    total_per_passenger: float
    subtotal: float


class CancellationPolicy(BaseModel):
    """Refund and ticket change rules according to fare conditions."""

    refundable: bool = Field(
        description="True if ticket allows refunds upon cancellation"
    )
    cancellation_fee: float = Field(description="Penalty fee deducted if cancelled")
    change_allowed: bool = Field(
        description="True if itinerary date/time modification is permitted"
    )
    change_fee: float = Field(description="Penalty fee required to change booking")
    terms: str = Field(description="Full text terms of cancellation and change")


class FlightOffer(BaseModel):
    """Comprehensive flight offer quote adhering to modern flight APIs."""

    offer_id: str = Field(
        description="Unique offer identifier to reference in details or booking"
    )
    trip_type: TripType = Field(description="'ONE_WAY' or 'ROUND_TRIP'")
    valid_until: str = Field(description="Quote expiration ISO timestamp")
    primary_airline_code: str = Field(description="Primary operating airline IATA code")
    primary_airline_name: str = Field(description="Primary operating airline name")
    itineraries: list[FlightItinerary] = Field(
        description="Outbound and optional inbound itineraries"
    )
    travel_class: TravelClass = Field(description="Cabin class booked")
    available_seats: int = Field(description="Remaining seat inventory on this offer")
    price: PriceBreakdown = Field(description="Itemized price breakdown")
    traveler_pricings: list[TravelerPricing] = Field(
        description="Per-passenger pricing details"
    )
    baggage: AirlineBaggagePolicy = Field(description="Baggage allowance included")
    cancellation_policy: CancellationPolicy = Field(
        description="Cancellation and refund rules"
    )


# --- Search Criteria and Response Models ---


class FlightSearchCriteria(BaseModel):
    """Comprehensive search parameters covering all standard booking portal fields."""

    origin: str = Field(
        description="Departure 3-letter IATA airport code (e.g. 'LHR', 'CDG', 'SGN')"
    )
    destination: str = Field(
        description="Arrival 3-letter IATA airport code (e.g. 'JFK', 'NRT', 'HAN')"
    )
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: str | None = Field(
        default=None,
        description="Return date in YYYY-MM-DD format (required if trip_type is ROUND_TRIP)",
    )
    trip_type: TripType = Field(
        default=TripType.ONE_WAY, description="Trip type: ONE_WAY or ROUND_TRIP"
    )
    travel_class: TravelClass = Field(
        default=TravelClass.ECONOMY,
        description="Cabin class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, or FIRST",
    )
    adults: int = Field(
        default=1, ge=1, le=9, description="Number of adult passengers (age 12+)"
    )
    children: int = Field(
        default=0, ge=0, le=8, description="Number of child passengers (age 2-11)"
    )
    infants: int = Field(
        default=0, ge=0, le=4, description="Number of infant passengers (under 2 years)"
    )
    direct_flights_only: bool = Field(
        default=False, description="Filter strictly for non-stop direct flights"
    )
    max_stops: int | None = Field(
        default=None,
        description="Maximum number of intermediate layovers (0 for direct, 1, or 2)",
    )
    included_airlines: list[str] | None = Field(
        default=None,
        description="List of airline IATA codes to filter by (e.g. ['VN', 'SQ', 'BA'])",
    )
    excluded_airlines: list[str] | None = Field(
        default=None, description="List of airline IATA codes to exclude"
    )
    departure_time_window: DepartureTimeWindow = Field(
        default=DepartureTimeWindow.ANY,
        description="Departure time-of-day window: ANY, MORNING, AFTERNOON, EVENING, NIGHT",
    )
    earliest_departure_time: str | None = Field(
        default=None, description="Earliest departure time (HH:MM, 24-hour format)"
    )
    latest_departure_time: str | None = Field(
        default=None, description="Latest departure time (HH:MM, 24-hour format)"
    )
    min_price: float | None = Field(
        default=None,
        ge=0.0,
        description="Minimum total price floor in specified currency",
    )
    max_price: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum total budget ceiling in specified currency",
    )
    currency: str = Field(
        default="USD", description="Fare currency code (default: 'USD')"
    )
    requires_checked_bag: bool = Field(
        default=False,
        description="Filter for offers with free checked baggage included",
    )
    sort_by: SortBy = Field(
        default=SortBy.BEST_VALUE,
        description="Sort order: PRICE_ASC, DURATION_ASC, DEPARTURE_ASC, ARRIVAL_ASC, BEST_VALUE",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of flight offers to return per page",
    )
    offset: int = Field(default=0, ge=0, description="Pagination offset index")
    session_id: str | None = Field(
        default=None, description="Session ID for tracking and audit tracing"
    )


class FlightSearchResponse(BaseModel):
    """Standardized search response returned by search_flights tool."""

    session_id: str = Field(description="Active session ID for tracking and isolation")
    search_criteria_summary: dict[str, Any] = Field(
        description="Normalized search parameters executed"
    )
    total_matches: int = Field(description="Total number of matching flights found")
    offset: int = Field(description="Current pagination offset")
    limit: int = Field(description="Current pagination limit")
    offers: list[FlightOffer] = Field(
        description="Matching flight offers with dynamic pricing"
    )
    notes: str | None = Field(
        default=None,
        description="Helpful instructions, tips, or guidance for the agent",
    )
    errors_or_warnings: list[str] | None = Field(
        default=None, description="Non-fatal warnings or resolution notices"
    )


# --- Booking, Order, and Session Management Models ---


class BookingStatus(str, Enum):
    """Status of an airline booking order."""

    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    PENDING_PAYMENT = "PENDING_PAYMENT"


class TravelerInfo(BaseModel):
    """Passenger details for ticket issuance."""

    traveler_id: str | None = Field(
        default=None, description="Passenger sequence identifier, e.g. PAX-1"
    )
    traveler_type: TravelerType = Field(
        default=TravelerType.ADULT,
        description="Passenger age type: ADULT, CHILD, or INFANT",
    )
    first_name: str = Field(
        description="Given name as displayed on passport/government ID"
    )
    last_name: str = Field(
        description="Family name / surname as displayed on passport/government ID"
    )
    date_of_birth: str = Field(description="Date of birth in YYYY-MM-DD format")
    passport_number: str | None = Field(
        default=None,
        description="Passport number (required for international itineraries)",
    )
    passport_expiry: str | None = Field(
        default=None, description="Passport expiration date (YYYY-MM-DD)"
    )
    nationality: str | None = Field(
        default=None, description="2-letter ISO country code or nationality"
    )
    email: str | None = Field(default=None, description="Traveler contact email")
    phone: str | None = Field(default=None, description="Traveler contact phone number")
    seat_preference: str | None = Field(
        default=None,
        description="Seat preference: 'WINDOW', 'AISLE', or specific code like '14A'",
    )
    meal_preference: str | None = Field(
        default=None,
        description="Special dietary meal request, e.g. 'VEGETARIAN', 'HALAL', 'STANDARD'",
    )


class FlightOrder(BaseModel):
    """Confirmed Flight Order / PNR Record strictly isolated to the owning session."""

    order_id: str = Field(
        description="Internal booking order identifier, e.g. ORD-B4F2"
    )
    booking_reference: str = Field(
        description="6-character airline PNR (Passenger Name Record), e.g. BA8X2K"
    )
    session_id: str = Field(
        description="Session ID that owns this booking, ensuring multi-tenant privacy"
    )
    status: BookingStatus = Field(
        default=BookingStatus.CONFIRMED, description="Booking status"
    )
    offer_id: str = Field(description="Referenced flight offer ID")
    primary_airline_code: str = Field(description="Primary operating airline code")
    primary_airline_name: str = Field(description="Primary operating airline name")
    travel_class: TravelClass = Field(description="Cabin class booked")
    itineraries: list[FlightItinerary] = Field(description="Booked flight itineraries")
    travelers: list[TravelerInfo] = Field(description="Registered passengers")
    contact_email: str = Field(description="Primary contact email for confirmation")
    contact_phone: str = Field(description="Primary contact phone number")
    seat_assignments: dict[str, str] = Field(
        default_factory=dict, description="Assigned seat mapping per passenger"
    )
    price_breakdown: PriceBreakdown = Field(description="Final price charged breakdown")
    cancellation_policy: CancellationPolicy = Field(
        description="Applicable cancellation terms"
    )
    payment_status: str = Field(
        default="PAID_CONFIRMED", description="Payment processing status"
    )
    created_at: str = Field(description="Booking creation ISO timestamp")
    updated_at: str = Field(description="Last update ISO timestamp")
    cancellation_details: dict[str, Any] | None = Field(
        default=None, description="Details if order was cancelled"
    )


class BookingCancellationResponse(BaseModel):
    """Response returned upon cancelling a flight booking."""

    session_id: str = Field(description="Active session ID")
    booking_reference: str = Field(description="PNR cancelled")
    order_id: str = Field(description="Order ID")
    status: BookingStatus = Field(default=BookingStatus.CANCELLED)
    original_total_amount: float = Field(description="Original fare paid in USD")
    cancellation_fee: float = Field(
        description="Deduction fee applied according to airline fare conditions"
    )
    refund_amount: float = Field(
        description="Net amount refunded to original payment method"
    )
    currency: str = Field(default="USD")
    cancelled_at: str = Field(description="Cancellation timestamp")
    message: str = Field(description="Confirmation message")


class SessionInfo(BaseModel):
    """Session diagnostic information for agentic execution tracing and multi-tenant audit."""

    session_id: str = Field(description="Unique active session identifier")
    created_at: str = Field(description="Session creation timestamp")
    active_orders_count: int = Field(
        default=0, description="Total active confirmed bookings owned by this session"
    )
    search_history_count: int = Field(
        default=0, description="Total searches executed in this session"
    )
    description: str = Field(description="Session summary for demo or audit")
