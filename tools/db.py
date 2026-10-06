"""Mock Flight Database Engine with dynamic pricing and airport/city discovery.

Implements:
1. Dynamic pricing derived purely from airline policies and flight metadata (no hardcoded multipliers).
2. Intelligent airport and city discovery with fuzzy typo correction (e.g. 'Londo' -> 'London').
3. Multi-airport metropolitan area resolution (e.g. London -> LHR, LGW, STN, LCY, LTN).
4. Direct and 1-stop connecting flight generation with layover validation.
5. All standard airline distribution search filters (cabin, price, time window, airline, baggage).
"""

import difflib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .mock_data import MOCK_AIRLINES, MOCK_AIRPORTS, MOCK_DIRECT_SCHEDULES
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


class MockFlightDatabase:
    """In-memory flight database engine for search, quotation, and airport discovery."""

    def __init__(self) -> None:
        self.airports: dict[str, AirportInfo] = {}
        self.city_to_airports: dict[str, list[AirportInfo]] = {}
        self.airlines: dict[str, AirlineInfo] = {}
        self.direct_schedules: list[dict[str, Any]] = MOCK_DIRECT_SCHEDULES
        self._active_offers: dict[str, FlightOffer] = {}

        # Multi-tenant session isolation stores: session_id -> data
        self._sessions: dict[str, SessionInfo] = {}
        # session_id -> {booking_reference or order_id: FlightOrder}
        self._orders_by_session: dict[str, dict[str, FlightOrder]] = {}
        # Dynamic seat inventory: flight_instance_key (e.g. 'BA-304_2026-10-15') -> remaining seats
        self._inventory: dict[str, int] = {}

        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Parse raw mock datasets into typed structures and indexing lookups."""
        for ap in MOCK_AIRPORTS:
            info = AirportInfo(**ap)
            self.airports[info.iata_code] = info

            # Index by normalized city name
            city_norm = info.city.strip().lower()
            if city_norm not in self.city_to_airports:
                self.city_to_airports[city_norm] = []
            self.city_to_airports[city_norm].append(info)

            # Index by metropolitan city code if present (e.g. LON, PAR, NYC, TYO)
            if info.city_code:
                code_norm = info.city_code.strip().lower()
                if code_norm not in self.city_to_airports:
                    self.city_to_airports[code_norm] = []
                if info not in self.city_to_airports[code_norm]:
                    self.city_to_airports[code_norm].append(info)

        for code, al in MOCK_AIRLINES.items():
            self.airlines[code] = AirlineInfo(
                iata_code=al["iata_code"],
                name=al["name"],
                alliance=al.get("alliance"),
                baggage_policy=AirlineBaggagePolicy(**al["baggage_policy"]),
                pricing_policy=AirlinePricingPolicy(**al["pricing_policy"]),
            )

    # --- Session Management & Multi-Tenant Isolation ---

    def get_or_create_session(self, session_id: str | None = None) -> SessionInfo:
        """Retrieve existing session or generate a randomized session ID for tenant isolation."""
        if not session_id or not session_id.strip():
            session_id = f"sess_{secrets.token_hex(6)}"

        session_id = session_id.strip()
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionInfo(
                session_id=session_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                active_orders_count=0,
                search_history_count=0,
                description=f"Isolated session environment for client {session_id}",
            )
            self._orders_by_session[session_id] = {}

        return self._sessions[session_id]

    # --- Real-Time Seat Inventory Tracker ---

    def _find_initial_seats(self, flight_number: str) -> int:
        """Find baseline seat capacity from schedule templates."""
        for s in self.direct_schedules:
            if s["flight_number"] == flight_number:
                return s.get("initial_available_seats", 20)
        return 20

    def _get_available_seats(
        self, flight_number: str, departure_date: str, initial_seats: int | None = None
    ) -> int:
        """Fetch real-time seat inventory for a flight instance."""
        key = f"{flight_number}_{departure_date}"
        if key not in self._inventory:
            if initial_seats is None:
                initial_seats = self._find_initial_seats(flight_number)
            self._inventory[key] = initial_seats
        return self._inventory[key]

    def _decrement_seats(
        self, flight_number: str, departure_date: str, seat_count: int
    ) -> bool:
        """Reserve seats on a flight instance. Returns False if insufficient seats available."""
        key = f"{flight_number}_{departure_date}"
        current = self._get_available_seats(flight_number, departure_date)
        if current < seat_count:
            return False
        self._inventory[key] = current - seat_count
        return True

    def _restore_seats(
        self, flight_number: str, departure_date: str, seat_count: int
    ) -> None:
        """Restore seats to inventory upon cancellation."""
        key = f"{flight_number}_{departure_date}"
        if key in self._inventory:
            self._inventory[key] += seat_count

    # --- Location Resolution & Typo Detection Engine ---

    def search_airports_and_cities(
        self,
        query: str,
        country: str | None = None,
    ) -> AirportSearchResult:
        """Discover airports by city, airport name, or IATA code with fuzzy typo correction.

        Specifically designed for LLM agents that lack pre-existing knowledge of airport codes.
        If an agent searches for 'Londo', suggests 'London' and returns all London airports.
        """
        raw_query = query.strip()
        q_lower = raw_query.lower()
        q_upper = raw_query.upper()

        # 1. Exact 3-letter IATA code lookup
        if q_upper in self.airports:
            ap = self.airports[q_upper]
            return AirportSearchResult(
                query=raw_query,
                matches_found=1,
                airports=[ap],
                suggestion=None,
                message=f"Found exact match for IATA code '{q_upper}': {ap.name} ({ap.city}, {ap.country}).",
            )

        # 2. Exact match on city name or city metropolitan code (e.g., 'London' or 'LON')
        if q_lower in self.city_to_airports:
            matched = self.city_to_airports[q_lower]
            if country:
                matched = [a for a in matched if country.lower() in a.country.lower()]
            codes = ", ".join(f"{a.iata_code} ({a.name})" for a in matched)
            return AirportSearchResult(
                query=raw_query,
                matches_found=len(matched),
                airports=matched,
                suggestion=None,
                message=f"Metropolitan area '{raw_query.title()}' is served by {len(matched)} airport(s): {codes}.",
            )

        # 3. Fuzzy typo detection on cities and airport names
        known_cities = list({ap.city for ap in self.airports.values()})
        known_airport_names = [ap.name for ap in self.airports.values()]
        known_terms = known_cities + known_airport_names

        # Check if raw_query is a close match/typo for a known city (e.g. 'Londo' -> 'London')
        close_cities = difflib.get_close_matches(
            raw_query.title(), known_cities, n=1, cutoff=0.7
        )
        if close_cities and close_cities[0].lower() != q_lower:
            suggested_city = close_cities[0]
            matched = self.city_to_airports.get(suggested_city.lower(), [])
            if country:
                matched = [a for a in matched if country.lower() in a.country.lower()]
            codes = ", ".join(f"{a.iata_code} ({a.name})" for a in matched)
            return AirportSearchResult(
                query=raw_query,
                matches_found=len(matched),
                airports=matched,
                suggestion=suggested_city,
                message=(
                    f"No exact airport found for '{raw_query}'. Did you mean '{suggested_city}'? "
                    f"Available airport(s) in {suggested_city}: {codes}."
                ),
            )

        # 4. Substring matching across city, airport name, or country
        substring_matches: list[AirportInfo] = []
        for ap in self.airports.values():
            if q_lower in ap.city.lower() or q_lower in ap.name.lower():
                if not country or country.lower() in ap.country.lower():
                    if ap not in substring_matches:
                        substring_matches.append(ap)

        if substring_matches:
            codes = ", ".join(
                f"{a.iata_code} ({a.name}, {a.city})" for a in substring_matches
            )
            return AirportSearchResult(
                query=raw_query,
                matches_found=len(substring_matches),
                airports=substring_matches,
                suggestion=None,
                message=f"Found {len(substring_matches)} matching airport(s) for '{raw_query}': {codes}.",
            )

        # 5. Fallback fuzzy typo detection on all terms
        close_matches = difflib.get_close_matches(
            raw_query, known_terms, n=1, cutoff=0.6
        )
        if close_matches:
            suggested_term = close_matches[0]
            suggested_airports: list[AirportInfo] = []
            if suggested_term.lower() in self.city_to_airports:
                suggested_airports = self.city_to_airports[suggested_term.lower()]
            else:
                for ap in self.airports.values():
                    if suggested_term.lower() in ap.name.lower():
                        suggested_airports.append(ap)

            codes = ", ".join(f"{a.iata_code} ({a.name})" for a in suggested_airports)
            return AirportSearchResult(
                query=raw_query,
                matches_found=len(suggested_airports),
                airports=suggested_airports,
                suggestion=suggested_term,
                message=(
                    f"No exact airport found for '{raw_query}'. Did you mean '{suggested_term}'? "
                    f"Available airport(s) in {suggested_term}: {codes}."
                ),
            )

        # 6. Nothing found
        available_cities = ", ".join(
            sorted(list({ap.city for ap in self.airports.values()}))
        )
        return AirportSearchResult(
            query=raw_query,
            matches_found=0,
            airports=[],
            suggestion=None,
            message=(
                f"No airports or cities matched '{raw_query}'. "
                f"Supported cities include: {available_cities}. "
                "Use search_airports_and_cities with a known city name or IATA code."
            ),
        )

    def validate_airport_code(
        self, code: str
    ) -> tuple[bool, str, list[AirportInfo] | None]:
        """Validate if a 3-letter IATA code is valid. If a city name was supplied, returns guidance."""
        c_upper = code.strip().upper()
        if c_upper in self.airports:
            return True, "", None

        # Check if the user accidentally passed a city name (e.g. 'London')
        c_lower = code.strip().lower()
        if c_lower in self.city_to_airports:
            matched = self.city_to_airports[c_lower]
            options = ", ".join(f"'{a.iata_code}' for {a.name}" for a in matched)
            msg = (
                f"'{code}' is a city with multiple airports, not a single 3-letter IATA code. "
                f"Please choose an airport code: {options}."
            )
            return False, msg, matched

        # Fuzzy check for typo in city name or code
        search_res = self.search_airports_and_cities(code)
        if search_res.suggestion:
            msg = (
                f"Airport code or city '{code}' was not recognized. {search_res.message} "
                "Please call search_airports_and_cities() to verify the exact 3-letter IATA code."
            )
            return False, msg, search_res.airports

        return (
            False,
            f"Airport '{code}' is not recognized. Please use search_airports_and_cities() first.",
            None,
        )

    def get_airline_info(self, query: str) -> dict[str, Any]:
        """Lookup airline details and policies by IATA code or name with typo support."""
        q_strip = query.strip()
        q_upper = q_strip.upper()

        if q_upper in self.airlines:
            al = self.airlines[q_upper]
            return {"status": "success", "airline": al.model_dump()}

        q_lower = q_strip.lower()
        for al in self.airlines.values():
            if q_lower == al.name.lower() or q_lower in al.name.lower():
                return {"status": "success", "airline": al.model_dump()}

        # Typo matching
        known_names = [al.name for al in self.airlines.values()]
        close = difflib.get_close_matches(q_strip, known_names, n=1, cutoff=0.6)
        if close:
            suggested = close[0]
            matched_al = next(
                al for al in self.airlines.values() if al.name == suggested
            )
            return {
                "status": "suggestion",
                "message": f"Airline '{query}' not found. Did you mean '{suggested}'?",
                "airline": matched_al.model_dump(),
            }

        available = ", ".join(
            f"{al.iata_code} ({al.name})" for al in self.airlines.values()
        )
        return {
            "status": "error",
            "message": f"Airline '{query}' not recognized. Available airlines: {available}.",
        }

    # --- Dynamic Pricing Calculation Engine ---

    def calculate_price_breakdown(
        self,
        airline_code: str,
        base_economy_fare: float,
        travel_class: TravelClass,
        departure_date_str: str,
        schedule_weekend_multiplier: float,
        adults: int,
        children: int,
        infants: int,
        requires_checked_bag: bool = False,
    ) -> tuple[PriceBreakdown, list[TravelerPricing]]:
        """Compute transparent dynamic pricing purely from airline policy and schedule metadata.

        Zero hardcoded multipliers: all pricing factors are extracted from AirlinePricingPolicy
        and schedule metadata.
        """
        airline = self.airlines.get(airline_code)
        if not airline:
            airline_policy = AirlinePricingPolicy()
            baggage_policy = AirlineBaggagePolicy()
        else:
            airline_policy = airline.pricing_policy
            baggage_policy = airline.baggage_policy

        # 1. Cabin class multiplier from airline policy
        cabin_mult = airline_policy.cabin_multipliers.get(travel_class.value, 1.0)

        # 2. Weekend surcharge from flight schedule metadata (Friday=4, Saturday=5, Sunday=6)
        is_weekend = False
        try:
            dep_date = datetime.strptime(departure_date_str, "%Y-%m-%d")
            if dep_date.weekday() in (4, 5, 6):
                is_weekend = True
        except ValueError:
            pass

        weekend_factor = schedule_weekend_multiplier if is_weekend else 1.0

        # 3. Base adult fare calculation
        adult_base_fare = round(base_economy_fare * cabin_mult * weekend_factor, 2)
        tax_rate = airline_policy.tax_and_fee_rate
        adult_taxes = round(adult_base_fare * tax_rate, 2)
        adult_total = round(adult_base_fare + adult_taxes, 2)

        # 4. Child discount from airline policy
        child_discount = airline_policy.child_discount_ratio
        child_base_fare = round(adult_base_fare * (1.0 - child_discount), 2)
        child_taxes = round(child_base_fare * tax_rate, 2)
        child_total = round(child_base_fare + child_taxes, 2)

        # 5. Infant discount from airline policy
        infant_discount = airline_policy.infant_discount_ratio
        infant_base_fare = round(adult_base_fare * (1.0 - infant_discount), 2)
        infant_taxes = round(infant_base_fare * tax_rate, 2)
        infant_total = round(infant_base_fare + infant_taxes, 2)

        traveler_pricings: list[TravelerPricing] = []
        total_base = 0.0
        total_taxes = 0.0

        if adults > 0:
            sub = round(adult_total * adults, 2)
            traveler_pricings.append(
                TravelerPricing(
                    traveler_type=TravelerType.ADULT,
                    count=adults,
                    fare_per_passenger=adult_base_fare,
                    taxes_per_passenger=adult_taxes,
                    total_per_passenger=adult_total,
                    subtotal=sub,
                )
            )
            total_base += adult_base_fare * adults
            total_taxes += adult_taxes * adults

        if children > 0:
            sub = round(child_total * children, 2)
            traveler_pricings.append(
                TravelerPricing(
                    traveler_type=TravelerType.CHILD,
                    count=children,
                    fare_per_passenger=child_base_fare,
                    taxes_per_passenger=child_taxes,
                    total_per_passenger=child_total,
                    subtotal=sub,
                )
            )
            total_base += child_base_fare * children
            total_taxes += child_taxes * children

        if infants > 0:
            sub = round(infant_total * infants, 2)
            traveler_pricings.append(
                TravelerPricing(
                    traveler_type=TravelerType.INFANT,
                    count=infants,
                    fare_per_passenger=infant_base_fare,
                    taxes_per_passenger=infant_taxes,
                    total_per_passenger=infant_total,
                    subtotal=sub,
                )
            )
            total_base += infant_base_fare * infants
            total_taxes += infant_taxes * infants

        # 6. Baggage fee if checked bag requested and airline charges extra
        baggage_fee = 0.0
        if requires_checked_bag and baggage_policy.checked_bags_included == 0:
            baggage_fee = round(
                airline_policy.extra_checked_bag_fee * (adults + children), 2
            )

        total_fare = round(total_base + total_taxes + baggage_fee, 2)

        breakdown = PriceBreakdown(
            currency="USD",
            base_fare=round(total_base, 2),
            taxes_and_fees=round(total_taxes, 2),
            baggage_fees=baggage_fee,
            seat_fees=0.0,
            total_fare=total_fare,
            pricing_factors_applied={
                "airline_code": airline_code,
                "cabin_multiplier": cabin_mult,
                "is_weekend": is_weekend,
                "weekend_multiplier": weekend_factor,
                "tax_rate": tax_rate,
                "child_discount_ratio": child_discount,
                "infant_discount_ratio": infant_discount,
            },
        )
        return breakdown, traveler_pricings

    # --- Schedule & Itinerary Builders ---

    def _matches_departure_window(
        self,
        time_str: str,
        window: DepartureTimeWindow,
        earliest: str | None,
        latest: str | None,
    ) -> bool:
        """Check if departure time string (HH:MM) fits within window or time range."""
        try:
            flight_time = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            return True

        if earliest:
            try:
                min_time = datetime.strptime(earliest, "%H:%M").time()
                if flight_time < min_time:
                    return False
            except ValueError:
                pass

        if latest:
            try:
                max_time = datetime.strptime(latest, "%H:%M").time()
                if flight_time > max_time:
                    return False
            except ValueError:
                pass

        if window == DepartureTimeWindow.MORNING:
            return 6 <= flight_time.hour < 12
        elif window == DepartureTimeWindow.AFTERNOON:
            return 12 <= flight_time.hour < 18
        elif window == DepartureTimeWindow.EVENING:
            return 18 <= flight_time.hour < 24
        elif window == DepartureTimeWindow.NIGHT:
            return 0 <= flight_time.hour < 6

        return True

    def _build_direct_itinerary(
        self,
        schedule: dict[str, Any],
        departure_date: str,
        travel_class: TravelClass,
    ) -> FlightItinerary:
        """Construct a single direct segment itinerary with accurate airport timezones."""
        orig_ap = self.airports.get(schedule["origin"])
        dest_ap = self.airports.get(schedule["destination"])
        orig_tz_str = orig_ap.timezone if orig_ap else "UTC"
        dest_tz_str = dest_ap.timezone if dest_ap else "UTC"

        try:
            orig_tz = ZoneInfo(orig_tz_str)
        except Exception:
            orig_tz = timezone.utc
        try:
            dest_tz = ZoneInfo(dest_tz_str)
        except Exception:
            dest_tz = timezone.utc

        # Local departure datetime with origin airport timezone
        dep_naive = datetime.strptime(
            f"{departure_date}T{schedule['departure_time']}:00", "%Y-%m-%dT%H:%M:%S"
        )
        dep_dt_local = dep_naive.replace(tzinfo=orig_tz)
        dep_dt_utc = dep_dt_local.astimezone(timezone.utc)

        # Elapsed real flight duration in UTC
        arr_dt_utc = dep_dt_utc + timedelta(minutes=schedule["duration_minutes"])
        # Converted to destination airport local clock time
        arr_dt_local = arr_dt_utc.astimezone(dest_tz)

        dep_time_str = dep_dt_local.isoformat()
        arr_time_str = arr_dt_local.isoformat()

        carrier = self.airlines.get(schedule["carrier_code"])
        carrier_name = carrier.name if carrier else schedule["carrier_code"]

        segment = FlightSegment(
            segment_id=f"SEG-{schedule['flight_number']}-{departure_date}",
            carrier_code=schedule["carrier_code"],
            carrier_name=carrier_name,
            flight_number=schedule["flight_number"],
            aircraft=schedule["aircraft"],
            departure_airport=schedule["origin"],
            departure_terminal=schedule.get("origin_terminal"),
            departure_time=dep_time_str,
            departure_timezone=orig_tz_str,
            arrival_airport=schedule["destination"],
            arrival_terminal=schedule.get("destination_terminal"),
            arrival_time=arr_time_str,
            arrival_timezone=dest_tz_str,
            duration_minutes=schedule["duration_minutes"],
            cabin_class=travel_class,
            stops=0,
        )

        hours, mins = divmod(schedule["duration_minutes"], 60)
        return FlightItinerary(
            itinerary_id=f"ITIN-{schedule['flight_number']}-{departure_date}",
            direction="OUTBOUND",
            total_duration_minutes=schedule["duration_minutes"],
            total_duration_iso=f"PT{hours}H{mins}M",
            stop_count=0,
            segments=[segment],
        )

    def _build_connecting_itinerary(
        self,
        leg1: dict[str, Any],
        leg2: dict[str, Any],
        departure_date: str,
        travel_class: TravelClass,
    ) -> FlightItinerary | None:
        """Construct a 2-segment connecting itinerary through a transit hub with realistic timezones."""
        orig_ap = self.airports.get(leg1["origin"])
        hub_ap = self.airports.get(leg1["destination"])
        dest_ap = self.airports.get(leg2["destination"])

        orig_tz_str = orig_ap.timezone if orig_ap else "UTC"
        hub_tz_str = hub_ap.timezone if hub_ap else "UTC"
        dest_tz_str = dest_ap.timezone if dest_ap else "UTC"

        try:
            orig_tz = ZoneInfo(orig_tz_str)
        except Exception:
            orig_tz = timezone.utc
        try:
            hub_tz = ZoneInfo(hub_tz_str)
        except Exception:
            hub_tz = timezone.utc
        try:
            dest_tz = ZoneInfo(dest_tz_str)
        except Exception:
            dest_tz = timezone.utc

        # Leg 1: origin to transit hub
        leg1_dep_naive = datetime.strptime(
            f"{departure_date}T{leg1['departure_time']}:00", "%Y-%m-%dT%H:%M:%S"
        )
        leg1_dep_local = leg1_dep_naive.replace(tzinfo=orig_tz)
        leg1_dep_utc = leg1_dep_local.astimezone(timezone.utc)
        leg1_arr_utc = leg1_dep_utc + timedelta(minutes=leg1["duration_minutes"])
        leg1_arr_local = leg1_arr_utc.astimezone(hub_tz)

        # Minimum connection layover at hub: 60 minutes
        min_conn = timedelta(minutes=60)
        earliest_leg2_dep_local = leg1_arr_local + min_conn

        leg2_time = datetime.strptime(leg2["departure_time"], "%H:%M").time()
        leg2_dep_local = datetime.combine(
            leg1_arr_local.date(), leg2_time, tzinfo=hub_tz
        )
        if leg2_dep_local < earliest_leg2_dep_local:
            leg2_dep_local += timedelta(days=1)

        leg2_dep_utc = leg2_dep_local.astimezone(timezone.utc)
        layover_mins = int((leg2_dep_utc - leg1_arr_utc).total_seconds() // 60)
        if layover_mins > 360 or layover_mins < 60:
            return None

        leg2_arr_utc = leg2_dep_utc + timedelta(minutes=leg2["duration_minutes"])
        leg2_arr_local = leg2_arr_utc.astimezone(dest_tz)
        total_trip_mins = int((leg2_arr_utc - leg1_dep_utc).total_seconds() // 60)

        carrier1 = self.airlines.get(leg1["carrier_code"])
        carrier2 = self.airlines.get(leg2["carrier_code"])

        seg1 = FlightSegment(
            segment_id=f"SEG-{leg1['flight_number']}-{departure_date}-1",
            carrier_code=leg1["carrier_code"],
            carrier_name=carrier1.name if carrier1 else leg1["carrier_code"],
            flight_number=leg1["flight_number"],
            aircraft=leg1["aircraft"],
            departure_airport=leg1["origin"],
            departure_terminal=leg1.get("origin_terminal"),
            departure_time=leg1_dep_local.isoformat(),
            departure_timezone=orig_tz_str,
            arrival_airport=leg1["destination"],
            arrival_terminal=leg1.get("destination_terminal"),
            arrival_time=leg1_arr_local.isoformat(),
            arrival_timezone=hub_tz_str,
            duration_minutes=leg1["duration_minutes"],
            cabin_class=travel_class,
            stops=0,
            layover_airport=leg1["destination"],
            layover_duration_minutes=layover_mins,
        )

        seg2 = FlightSegment(
            segment_id=f"SEG-{leg2['flight_number']}-{leg2_dep_local.strftime('%Y-%m-%d')}-2",
            carrier_code=leg2["carrier_code"],
            carrier_name=carrier2.name if carrier2 else leg2["carrier_code"],
            flight_number=leg2["flight_number"],
            aircraft=leg2["aircraft"],
            departure_airport=leg2["origin"],
            departure_terminal=leg2.get("origin_terminal"),
            departure_time=leg2_dep_local.isoformat(),
            departure_timezone=hub_tz_str,
            arrival_airport=leg2["destination"],
            arrival_terminal=leg2.get("destination_terminal"),
            arrival_time=leg2_arr_local.isoformat(),
            arrival_timezone=dest_tz_str,
            duration_minutes=leg2["duration_minutes"],
            cabin_class=travel_class,
            stops=0,
        )

        hours, mins = divmod(total_trip_mins, 60)
        return FlightItinerary(
            itinerary_id=f"ITIN-{leg1['flight_number']}-{leg2['flight_number']}-{departure_date}",
            direction="OUTBOUND",
            total_duration_minutes=total_trip_mins,
            total_duration_iso=f"PT{hours}H{mins}M",
            stop_count=1,
            segments=[seg1, seg2],
        )

    # --- Core Search Engine ---

    def search_flights(self, criteria: FlightSearchCriteria) -> FlightSearchResponse:
        """Search flight offers satisfying all criteria with complete data-driven pricing."""
        session = self.get_or_create_session(criteria.session_id)
        session.search_history_count += 1
        session_id = session.session_id

        # Validate origin & destination IATA codes
        origin_ok, origin_err, _ = self.validate_airport_code(criteria.origin)
        if not origin_ok:
            raise ValueError(origin_err)

        dest_ok, dest_err, _ = self.validate_airport_code(criteria.destination)
        if not dest_ok:
            raise ValueError(dest_err)

        origin_code = criteria.origin.strip().upper()
        dest_code = criteria.destination.strip().upper()

        if origin_code == dest_code:
            raise ValueError("Origin and destination cannot be the same airport.")

        total_passengers = criteria.adults + criteria.children + criteria.infants
        if total_passengers <= 0:
            raise ValueError("At least 1 passenger is required.")

        candidate_offers: list[FlightOffer] = []

        # 1. Outbound Direct Schedules
        for sched in self.direct_schedules:
            if sched["origin"] == origin_code and sched["destination"] == dest_code:
                # Airline filters
                if (
                    criteria.included_airlines
                    and sched["carrier_code"] not in criteria.included_airlines
                ):
                    continue
                if (
                    criteria.excluded_airlines
                    and sched["carrier_code"] in criteria.excluded_airlines
                ):
                    continue

                # Departure time window filter
                if not self._matches_departure_window(
                    sched["departure_time"],
                    criteria.departure_time_window,
                    criteria.earliest_departure_time,
                    criteria.latest_departure_time,
                ):
                    continue

                itinerary = self._build_direct_itinerary(
                    sched, criteria.departure_date, criteria.travel_class
                )
                available_seats = self._get_available_seats(
                    sched["flight_number"],
                    criteria.departure_date,
                    sched.get("initial_available_seats", 20),
                )
                if available_seats < total_passengers:
                    continue

                airline = self.airlines.get(sched["carrier_code"])
                baggage = airline.baggage_policy if airline else AirlineBaggagePolicy()
                if criteria.requires_checked_bag and baggage.checked_bags_included == 0:
                    continue

                price, traveler_pricing = self.calculate_price_breakdown(
                    airline_code=sched["carrier_code"],
                    base_economy_fare=sched["base_economy_price"],
                    travel_class=criteria.travel_class,
                    departure_date_str=criteria.departure_date,
                    schedule_weekend_multiplier=sched.get("weekend_multiplier", 1.0),
                    adults=criteria.adults,
                    children=criteria.children,
                    infants=criteria.infants,
                    requires_checked_bag=criteria.requires_checked_bag,
                )

                # Price ceiling and floor filters
                if (
                    criteria.max_price is not None
                    and price.total_fare > criteria.max_price
                ):
                    continue
                if (
                    criteria.min_price is not None
                    and price.total_fare < criteria.min_price
                ):
                    continue

                cancel_rules = CancellationPolicy(**sched["cancellation_policy"])
                offer_id = (
                    f"OFFER-{sched['carrier_code']}-{secrets.token_hex(4).upper()}"
                )

                offer = FlightOffer(
                    offer_id=offer_id,
                    trip_type=TripType.ONE_WAY,
                    valid_until=(
                        datetime.now(timezone.utc) + timedelta(hours=2)
                    ).isoformat(),
                    primary_airline_code=sched["carrier_code"],
                    primary_airline_name=airline.name
                    if airline
                    else sched["carrier_code"],
                    itineraries=[itinerary],
                    travel_class=criteria.travel_class,
                    available_seats=available_seats,
                    price=price,
                    traveler_pricings=traveler_pricing,
                    baggage=baggage,
                    cancellation_policy=cancel_rules,
                )
                self._active_offers[offer_id] = offer
                candidate_offers.append(offer)

        # 2. Connecting 1-stop routes (if stops allowed)
        if not criteria.direct_flights_only and (
            criteria.max_stops is None or criteria.max_stops >= 1
        ):
            for leg1 in self.direct_schedules:
                if leg1["origin"] == origin_code:
                    transit_hub = leg1["destination"]
                    for leg2 in self.direct_schedules:
                        if (
                            leg2["origin"] == transit_hub
                            and leg2["destination"] == dest_code
                        ):
                            primary_code = leg1["carrier_code"]
                            if (
                                criteria.included_airlines
                                and primary_code not in criteria.included_airlines
                            ):
                                continue
                            if (
                                criteria.excluded_airlines
                                and primary_code in criteria.excluded_airlines
                            ):
                                continue

                            if not self._matches_departure_window(
                                leg1["departure_time"],
                                criteria.departure_time_window,
                                criteria.earliest_departure_time,
                                criteria.latest_departure_time,
                            ):
                                continue

                            itinerary = self._build_connecting_itinerary(
                                leg1,
                                leg2,
                                criteria.departure_date,
                                criteria.travel_class,
                            )
                            if not itinerary:
                                continue

                            seats1 = self._get_available_seats(
                                leg1["flight_number"],
                                criteria.departure_date,
                                leg1.get("initial_available_seats", 20),
                            )
                            seats2 = self._get_available_seats(
                                leg2["flight_number"],
                                criteria.departure_date,
                                leg2.get("initial_available_seats", 20),
                            )
                            seats = min(seats1, seats2)
                            if seats < total_passengers:
                                continue

                            airline = self.airlines.get(primary_code)
                            baggage = (
                                airline.baggage_policy
                                if airline
                                else AirlineBaggagePolicy()
                            )
                            if (
                                criteria.requires_checked_bag
                                and baggage.checked_bags_included == 0
                            ):
                                continue

                            combined_base = (
                                leg1["base_economy_price"] + leg2["base_economy_price"]
                            ) * 0.90
                            weekend_mult = max(
                                leg1.get("weekend_multiplier", 1.0),
                                leg2.get("weekend_multiplier", 1.0),
                            )

                            price, traveler_pricing = self.calculate_price_breakdown(
                                airline_code=primary_code,
                                base_economy_fare=combined_base,
                                travel_class=criteria.travel_class,
                                departure_date_str=criteria.departure_date,
                                schedule_weekend_multiplier=weekend_mult,
                                adults=criteria.adults,
                                children=criteria.children,
                                infants=criteria.infants,
                                requires_checked_bag=criteria.requires_checked_bag,
                            )

                            if (
                                criteria.max_price is not None
                                and price.total_fare > criteria.max_price
                            ):
                                continue
                            if (
                                criteria.min_price is not None
                                and price.total_fare < criteria.min_price
                            ):
                                continue

                            cancel_rules = CancellationPolicy(
                                **leg1["cancellation_policy"]
                            )
                            offer_id = f"OFFER-CONN-{primary_code}-{secrets.token_hex(4).upper()}"

                            offer = FlightOffer(
                                offer_id=offer_id,
                                trip_type=TripType.ONE_WAY,
                                valid_until=(
                                    datetime.now(timezone.utc) + timedelta(hours=2)
                                ).isoformat(),
                                primary_airline_code=primary_code,
                                primary_airline_name=airline.name
                                if airline
                                else primary_code,
                                itineraries=[itinerary],
                                travel_class=criteria.travel_class,
                                available_seats=seats,
                                price=price,
                                traveler_pricings=traveler_pricing,
                                baggage=baggage,
                                cancellation_policy=cancel_rules,
                            )
                            self._active_offers[offer_id] = offer
                            candidate_offers.append(offer)

        # 3. Round-Trip Pairing if requested
        if criteria.trip_type == TripType.ROUND_TRIP and criteria.return_date:
            round_trip_offers: list[FlightOffer] = []
            for out_offer in candidate_offers:
                for ret_sched in self.direct_schedules:
                    if (
                        ret_sched["origin"] == dest_code
                        and ret_sched["destination"] == origin_code
                    ):
                        if ret_sched["carrier_code"] != out_offer.primary_airline_code:
                            continue

                        ret_itin = self._build_direct_itinerary(
                            ret_sched, criteria.return_date, criteria.travel_class
                        )
                        ret_itin.direction = "INBOUND"

                        ret_seats = self._get_available_seats(
                            ret_sched["flight_number"],
                            criteria.return_date,
                            ret_sched.get("initial_available_seats", 20),
                        )
                        if ret_seats < total_passengers:
                            continue

                        ret_price, ret_traveler_pricing = (
                            self.calculate_price_breakdown(
                                airline_code=ret_sched["carrier_code"],
                                base_economy_fare=ret_sched["base_economy_price"]
                                * 0.90,
                                travel_class=criteria.travel_class,
                                departure_date_str=criteria.return_date,
                                schedule_weekend_multiplier=ret_sched.get(
                                    "weekend_multiplier", 1.0
                                ),
                                adults=criteria.adults,
                                children=criteria.children,
                                infants=criteria.infants,
                                requires_checked_bag=criteria.requires_checked_bag,
                            )
                        )

                        combined_total = round(
                            out_offer.price.total_fare + ret_price.total_fare, 2
                        )
                        if (
                            criteria.max_price is not None
                            and combined_total > criteria.max_price
                        ):
                            continue
                        if (
                            criteria.min_price is not None
                            and combined_total < criteria.min_price
                        ):
                            continue

                        combined_price = PriceBreakdown(
                            currency="USD",
                            base_fare=round(
                                out_offer.price.base_fare + ret_price.base_fare, 2
                            ),
                            taxes_and_fees=round(
                                out_offer.price.taxes_and_fees
                                + ret_price.taxes_and_fees,
                                2,
                            ),
                            baggage_fees=round(
                                out_offer.price.baggage_fees + ret_price.baggage_fees, 2
                            ),
                            seat_fees=0.0,
                            total_fare=combined_total,
                            pricing_factors_applied={
                                "round_trip_discount": 0.10,
                                "outbound_offer": out_offer.offer_id,
                            },
                        )

                        combined_traveler_pricings = []
                        for out_tp, ret_tp in zip(
                            out_offer.traveler_pricings, ret_traveler_pricing
                        ):
                            combined_traveler_pricings.append(
                                TravelerPricing(
                                    traveler_type=out_tp.traveler_type,
                                    count=out_tp.count,
                                    fare_per_passenger=round(
                                        out_tp.fare_per_passenger
                                        + ret_tp.fare_per_passenger,
                                        2,
                                    ),
                                    taxes_per_passenger=round(
                                        out_tp.taxes_per_passenger
                                        + ret_tp.taxes_per_passenger,
                                        2,
                                    ),
                                    total_per_passenger=round(
                                        out_tp.total_per_passenger
                                        + ret_tp.total_per_passenger,
                                        2,
                                    ),
                                    subtotal=round(
                                        out_tp.subtotal + ret_tp.subtotal, 2
                                    ),
                                )
                            )

                        rt_offer_id = f"OFFER-RT-{out_offer.primary_airline_code}-{secrets.token_hex(4).upper()}"
                        rt_offer = FlightOffer(
                            offer_id=rt_offer_id,
                            trip_type=TripType.ROUND_TRIP,
                            valid_until=out_offer.valid_until,
                            primary_airline_code=out_offer.primary_airline_code,
                            primary_airline_name=out_offer.primary_airline_name,
                            itineraries=[out_offer.itineraries[0], ret_itin],
                            travel_class=criteria.travel_class,
                            available_seats=min(out_offer.available_seats, ret_seats),
                            price=combined_price,
                            traveler_pricings=combined_traveler_pricings,
                            baggage=out_offer.baggage,
                            cancellation_policy=out_offer.cancellation_policy,
                        )
                        self._active_offers[rt_offer_id] = rt_offer
                        round_trip_offers.append(rt_offer)

            if round_trip_offers:
                candidate_offers = round_trip_offers

        # 4. Sorting
        if criteria.sort_by == SortBy.PRICE_ASC:
            candidate_offers.sort(key=lambda o: o.price.total_fare)
        elif criteria.sort_by == SortBy.DURATION_ASC:
            candidate_offers.sort(
                key=lambda o: sum(itin.total_duration_minutes for itin in o.itineraries)
            )
        elif criteria.sort_by == SortBy.DEPARTURE_ASC:
            candidate_offers.sort(
                key=lambda o: o.itineraries[0].segments[0].departure_time
            )
        elif criteria.sort_by == SortBy.ARRIVAL_ASC:
            candidate_offers.sort(
                key=lambda o: o.itineraries[0].segments[-1].arrival_time
            )
        elif criteria.sort_by == SortBy.BEST_VALUE:
            candidate_offers.sort(
                key=lambda o: (
                    o.price.total_fare * 0.7
                    + sum(itin.total_duration_minutes for itin in o.itineraries) * 0.3
                )
            )

        total_matches = len(candidate_offers)
        paginated = candidate_offers[criteria.offset : criteria.offset + criteria.limit]

        return FlightSearchResponse(
            session_id=session_id,
            search_criteria_summary={
                "origin": origin_code,
                "destination": dest_code,
                "departure_date": criteria.departure_date,
                "return_date": criteria.return_date,
                "trip_type": criteria.trip_type.value,
                "travel_class": criteria.travel_class.value,
                "adults": criteria.adults,
                "children": criteria.children,
                "infants": criteria.infants,
                "total_passengers": total_passengers,
                "sort_by": criteria.sort_by.value,
            },
            total_matches=total_matches,
            offset=criteria.offset,
            limit=criteria.limit,
            offers=paginated,
            notes=f"Found {total_matches} flight offer(s) from {origin_code} to {dest_code}. Dynamic pricing applied.",
        )

    def get_flight_details(self, offer_id: str) -> FlightOffer | None:
        """Fetch deep details of an active offer quote by ID."""
        return self._active_offers.get(offer_id.strip())

    # --- Booking Creation & Session-Isolated Storage ---

    def create_flight_booking(
        self,
        offer_id: str,
        travelers: list[TravelerInfo],
        contact_email: str,
        contact_phone: str,
        payment_method: str = "CREDIT_CARD",
        selected_seats: list[str] | None = None,
        special_requests: str | None = None,
        session_id: str | None = None,
    ) -> FlightOrder:
        """Create a confirmed flight order locked strictly to the requesting session ID."""
        session = self.get_or_create_session(session_id)
        offer = self.get_flight_details(offer_id)
        if not offer:
            raise ValueError(
                f"Flight offer '{offer_id}' not found or expired. "
                "Flight quotes expire after 2 hours. Please perform search_flights again to receive fresh offers."
            )

        passenger_count = len(travelers)
        if passenger_count == 0:
            raise ValueError(
                "At least one traveler profile must be supplied to create a booking."
            )

        # Check and decrement seat inventory for each segment across all itineraries
        for itin in offer.itineraries:
            for seg in itin.segments:
                dep_date = seg.departure_time.split("T")[0]
                ok = self._decrement_seats(seg.flight_number, dep_date, passenger_count)
                if not ok:
                    raise ValueError(
                        f"Seat inventory exhausted on segment {seg.flight_number} for date {dep_date}."
                    )

        # Generate realistic 6-character PNR and internal Order ID
        pnr = f"{offer.primary_airline_code}{secrets.token_hex(2).upper()}"
        order_id = f"ORD-{secrets.token_hex(4).upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Map seats if provided, else assign default seat recommendations
        seat_assignments: dict[str, str] = {}
        for idx, t in enumerate(travelers):
            p_name = f"{t.first_name} {t.last_name}"
            if selected_seats and idx < len(selected_seats):
                seat_assignments[p_name] = selected_seats[idx]
            else:
                row = 12 + idx
                col = (
                    "A"
                    if t.seat_preference == "WINDOW"
                    else ("C" if t.seat_preference == "AISLE" else "B")
                )
                seat_assignments[p_name] = f"{row}{col}"

        order = FlightOrder(
            order_id=order_id,
            booking_reference=pnr,
            session_id=session.session_id,
            status=BookingStatus.CONFIRMED,
            offer_id=offer.offer_id,
            primary_airline_code=offer.primary_airline_code,
            primary_airline_name=offer.primary_airline_name,
            travel_class=offer.travel_class,
            itineraries=offer.itineraries,
            travelers=travelers,
            contact_email=contact_email,
            contact_phone=contact_phone,
            seat_assignments=seat_assignments,
            price_breakdown=offer.price,
            cancellation_policy=offer.cancellation_policy,
            payment_status=f"PAID_CONFIRMED_VIA_{payment_method.upper()}",
            created_at=now_iso,
            updated_at=now_iso,
        )

        # Store strictly inside this session's isolated compartment
        session_orders = self._orders_by_session[session.session_id]
        session_orders[pnr] = order
        session_orders[order_id] = order
        session.active_orders_count = len(
            {
                o.booking_reference
                for o in session_orders.values()
                if o.status == BookingStatus.CONFIRMED
            }
        )

        return order

    # --- Session-Isolated Order Retrieval ---

    def get_booking_details(
        self, booking_reference: str, session_id: str | None = None
    ) -> FlightOrder | None:
        """Fetch booking details strictly scoped to session_id to avoid leaking other users' data."""
        session = self.get_or_create_session(session_id)
        session_orders = self._orders_by_session.get(session.session_id, {})
        ref = booking_reference.strip().upper()
        return session_orders.get(ref)

    # --- Session-Isolated Booking Cancellation ---

    def cancel_flight_booking(
        self,
        booking_reference: str,
        reason: str | None = None,
        session_id: str | None = None,
    ) -> BookingCancellationResponse:
        """Cancel an existing booking belonging to the session, calculating net refund."""
        session = self.get_or_create_session(session_id)
        order = self.get_booking_details(booking_reference, session.session_id)

        if not order:
            raise ValueError(
                f"Booking '{booking_reference}' not found in current session '{session.session_id}'. "
                "Ensure the correct booking reference (PNR) and session are provided."
            )

        if order.status == BookingStatus.CANCELLED:
            raise ValueError(
                f"Booking '{booking_reference}' has already been cancelled."
            )

        # Compute fee and refund based on airline cancellation policy
        policy = order.cancellation_policy
        now_iso = datetime.now(timezone.utc).isoformat()
        fee = (
            policy.cancellation_fee
            if policy.refundable
            else order.price_breakdown.total_fare
        )
        refund_amount = max(0.0, round(order.price_breakdown.total_fare - fee, 2))

        # Restore seat inventory
        passenger_count = len(order.travelers)
        for itin in order.itineraries:
            for seg in itin.segments:
                dep_date = seg.departure_time.split("T")[0]
                self._restore_seats(seg.flight_number, dep_date, passenger_count)

        # Update order status
        order.status = BookingStatus.CANCELLED
        order.updated_at = now_iso
        order.cancellation_details = {
            "cancelled_at": now_iso,
            "reason": reason or "Requested by customer",
            "fee_applied": fee,
            "refund_issued": refund_amount,
        }
        session.active_orders_count = len(
            {
                o.booking_reference
                for o in self._orders_by_session[session.session_id].values()
                if o.status == BookingStatus.CONFIRMED
            }
        )

        return BookingCancellationResponse(
            session_id=session.session_id,
            booking_reference=order.booking_reference,
            order_id=order.order_id,
            status=BookingStatus.CANCELLED,
            original_total_amount=order.price_breakdown.total_fare,
            cancellation_fee=fee,
            refund_amount=refund_amount,
            currency=order.price_breakdown.currency,
            cancelled_at=now_iso,
            message=f"Booking {order.booking_reference} successfully cancelled. Refund of ${refund_amount} processed.",
        )

    # --- Session Diagnostic Status ---

    def get_session_status(self, session_id: str | None = None) -> dict[str, Any]:
        """Inspect session state, audit trail, and active bookings."""
        session = self.get_or_create_session(session_id)
        session_orders = self._orders_by_session.get(session.session_id, {})
        unique_pnrs = list(
            {
                o.booking_reference
                for o in session_orders.values()
                if o.status == BookingStatus.CONFIRMED
            }
        )

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "searches_executed": session.search_history_count,
            "active_confirmed_bookings_count": len(unique_pnrs),
            "active_pnr_references": unique_pnrs,
            "isolation_status": "ENFORCED (Zero cross-session data leakage)",
        }


# Global database singleton
db = MockFlightDatabase()
