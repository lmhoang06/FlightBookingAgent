"""Unit tests for flight search and quotation tools."""

import json

from tools.flight_search_tools import get_flight_details, search_flights


def test_search_flights_city_name_validation_guidance() -> None:
    """Passing a city name like 'London' instead of IATA code must return clear guidance."""
    raw = search_flights.invoke(
        {
            "origin": "London",
            "destination": "CDG",
            "departure_date": "2026-10-15",
        }
    )
    data = json.loads(raw)

    assert data["status"] == "error"
    assert "multiple airports" in data["message"]
    assert "LHR" in data["message"]


def test_search_flights_direct_london_to_paris() -> None:
    """Searching LHR to CDG must return direct British Airways flights."""
    raw = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-14",  # Wednesday (no weekend surcharge)
            "travel_class": "ECONOMY",
            "adults": 1,
        }
    )
    data = json.loads(raw)

    assert data["total_matches"] >= 1
    offer = data["offers"][0]
    assert offer["primary_airline_code"] == "BA"
    assert offer["itineraries"][0]["segments"][0]["departure_airport"] == "LHR"
    assert offer["itineraries"][0]["segments"][0]["arrival_airport"] == "CDG"

    # Verify dynamic pricing breakdown
    price = offer["price"]
    assert price["currency"] == "USD"
    assert price["base_fare"] > 0
    assert price["taxes_and_fees"] > 0
    assert price["total_fare"] == round(price["base_fare"] + price["taxes_and_fees"], 2)


def test_dynamic_pricing_cabin_multiplier() -> None:
    """Business class price must be derived from the airline's policy multiplier."""
    # Economy search
    raw_econ = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-14",
            "travel_class": "ECONOMY",
            "adults": 1,
        }
    )
    data_econ = json.loads(raw_econ)
    base_econ = data_econ["offers"][0]["price"]["base_fare"]

    # Business search on same flight
    raw_biz = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-14",
            "travel_class": "BUSINESS",
            "adults": 1,
        }
    )
    data_biz = json.loads(raw_biz)
    base_biz = data_biz["offers"][0]["price"]["base_fare"]

    # BA Business multiplier is 2.50
    assert round(base_biz, 2) == round(base_econ * 2.50, 2)


def test_dynamic_pricing_child_and_infant_discounts() -> None:
    """Child and infant fares must apply airline discount ratios."""
    raw = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-14",
            "travel_class": "ECONOMY",
            "adults": 1,
            "children": 1,
            "infants": 1,
        }
    )
    data = json.loads(raw)
    offer = data["offers"][0]
    pricing = offer["traveler_pricings"]

    adult_p = next(p for p in pricing if p["traveler_type"] == "ADULT")
    child_p = next(p for p in pricing if p["traveler_type"] == "CHILD")
    infant_p = next(p for p in pricing if p["traveler_type"] == "INFANT")

    # BA child discount is 20% off base fare (ratio=0.20), infant is 90% off (ratio=0.90)
    expected_child_base = round(adult_p["fare_per_passenger"] * 0.80, 2)
    expected_infant_base = round(adult_p["fare_per_passenger"] * 0.10, 2)

    assert child_p["fare_per_passenger"] == expected_child_base
    assert infant_p["fare_per_passenger"] == expected_infant_base


def test_search_flights_round_trip() -> None:
    """Searching round trip pairing LHR <-> CDG returns outbound and inbound itineraries."""
    raw = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-14",
            "return_date": "2026-10-20",
            "trip_type": "ROUND_TRIP",
        }
    )
    data = json.loads(raw)

    assert data["total_matches"] >= 1
    offer = data["offers"][0]
    assert offer["trip_type"] == "ROUND_TRIP"
    assert len(offer["itineraries"]) == 2
    assert offer["itineraries"][0]["direction"] == "OUTBOUND"
    assert offer["itineraries"][1]["direction"] == "INBOUND"


def test_search_flights_filtering_and_sorting() -> None:
    """Filters for max_price and departure window must filter accurately."""
    raw = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-14",
            "max_price": 500.0,
            "sort_by": "PRICE_ASC",
            "limit": 2,
        }
    )
    data = json.loads(raw)

    assert len(data["offers"]) <= 2
    fares = [o["price"]["total_fare"] for o in data["offers"]]
    assert all(f <= 500.0 for f in fares)
    # Check ascending order
    assert fares == sorted(fares)


def test_get_flight_details() -> None:
    """Valid offer ID inspection must retrieve full specifications."""
    search_raw = search_flights.invoke(
        {
            "origin": "LHR",
            "destination": "CDG",
            "departure_date": "2026-10-14",
        }
    )
    search_data = json.loads(search_raw)
    offer_id = search_data["offers"][0]["offer_id"]

    details_raw = get_flight_details.invoke({"offer_id": offer_id})
    details_data = json.loads(details_raw)

    assert details_data["status"] == "success"
    assert details_data["offer"]["offer_id"] == offer_id


def test_get_flight_details_non_existent() -> None:
    """Non-existent offer ID returns an informative error."""
    raw = get_flight_details.invoke({"offer_id": "OFFER-FAKE-9999"})
    data = json.loads(raw)

    assert data["status"] == "error"
    assert "not found or expired" in data["message"]


def test_flight_search_timezone_transpacific_hnd_to_sfo() -> None:
    """Transpacific flight HND -> SFO must land earlier on the same calendar day due to timezone difference."""
    raw = search_flights.invoke(
        {
            "origin": "HND",
            "destination": "SFO",
            "departure_date": "2026-10-15",
        }
    )
    data = json.loads(raw)

    assert data["total_matches"] >= 1
    segment = data["offers"][0]["itineraries"][0]["segments"][0]

    # Verify timezones
    assert segment["departure_timezone"] == "Asia/Tokyo"
    assert segment["arrival_timezone"] == "America/Los_Angeles"

    # Verify ISO-8601 offset strings
    assert "+09:00" in segment["departure_time"]
    assert "-07:00" in segment["arrival_time"]

    # Departs 19:45 in Tokyo, arrives 13:20 same day in San Francisco
    assert "2026-10-15T19:45:00+09:00" == segment["departure_time"]
    assert "2026-10-15T13:20:00-07:00" == segment["arrival_time"]


def test_flight_search_timezone_transatlantic_jfk_to_lhr() -> None:
    """Transatlantic flight JFK -> LHR departs evening and arrives next morning with correct offsets."""
    raw = search_flights.invoke(
        {
            "origin": "JFK",
            "destination": "LHR",
            "departure_date": "2026-10-15",
        }
    )
    data = json.loads(raw)

    assert data["total_matches"] >= 1
    segment = data["offers"][0]["itineraries"][0]["segments"][0]

    assert segment["departure_timezone"] == "America/New_York"
    assert segment["arrival_timezone"] == "Europe/London"
    assert "-04:00" in segment["departure_time"]  # EDT
    assert "+01:00" in segment["arrival_time"]  # BST
    # Arrives next morning
    assert "2026-10-16" in segment["arrival_time"]
