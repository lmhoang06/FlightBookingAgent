"""Unit tests for airport and airline discovery tools."""

import json

from tools.airport_tools import get_airline_info, search_airports_and_cities


def test_search_airports_multi_airport_city_london() -> None:
    """Searching for 'London' must return all 5 London metropolitan airports."""
    raw = search_airports_and_cities.invoke({"query": "London"})
    data = json.loads(raw)

    assert data["matches_found"] == 5
    codes = [a["iata_code"] for a in data["airports"]]
    assert "LHR" in codes
    assert "LGW" in codes
    assert "STN" in codes
    assert "LCY" in codes
    assert "LTN" in codes


def test_search_airports_multi_airport_city_paris() -> None:
    """Searching for 'Paris' must return Paris airports (CDG, ORY, BVA)."""
    raw = search_airports_and_cities.invoke({"query": "Paris"})
    data = json.loads(raw)

    assert data["matches_found"] >= 2
    codes = [a["iata_code"] for a in data["airports"]]
    assert "CDG" in codes
    assert "ORY" in codes


def test_search_airports_exact_iata_code() -> None:
    """Querying by exact 3-letter IATA code must return the specific airport."""
    raw = search_airports_and_cities.invoke({"query": "LHR"})
    data = json.loads(raw)

    assert data["matches_found"] == 1
    assert data["airports"][0]["iata_code"] == "LHR"
    assert "Heathrow" in data["airports"][0]["name"]
    assert data["suggestion"] is None


def test_search_airports_fuzzy_typo_londo() -> None:
    """Querying 'Londo' must trigger fuzzy correction to 'London' and return its airports."""
    raw = search_airports_and_cities.invoke({"query": "Londo"})
    data = json.loads(raw)

    assert data["suggestion"] == "London"
    assert data["matches_found"] == 5
    assert "Did you mean 'London'?" in data["message"]


def test_search_airports_unknown_query() -> None:
    """Querying an unknown term should return 0 matches with helpful guidance."""
    raw = search_airports_and_cities.invoke({"query": "NonExistentCityXYZ"})
    data = json.loads(raw)

    assert data["matches_found"] == 0
    assert len(data["airports"]) == 0
    assert "Supported cities include" in data["message"]


def test_get_airline_info_exact_code() -> None:
    """Querying airline by 2-letter IATA code returns its profile and policies."""
    raw = get_airline_info.invoke({"query": "BA"})
    data = json.loads(raw)

    assert data["status"] == "success"
    airline = data["airline"]
    assert airline["iata_code"] == "BA"
    assert airline["alliance"] == "oneworld"
    assert "pricing_policy" in airline
    assert airline["pricing_policy"]["cabin_multipliers"]["BUSINESS"] == 2.50


def test_get_airline_info_typo_suggestion() -> None:
    """Querying with a typo like 'Britsh Airways' should suggest 'British Airways'."""
    raw = get_airline_info.invoke({"query": "Britsh Airways"})
    data = json.loads(raw)

    assert data["status"] == "suggestion"
    assert "Did you mean 'British Airways'?" in data["message"]
    assert data["airline"]["iata_code"] == "BA"
