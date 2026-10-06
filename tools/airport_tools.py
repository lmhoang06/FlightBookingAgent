"""LangChain tools for airport and airline discovery.

Follows the Single Responsibility Principle (SRP):
- Enables agentic models to resolve city names (especially multi-airport metros like London, Paris, NYC)
  into specific 3-letter IATA codes.
- Provides proactive fuzzy typo detection (e.g. 'Londo' -> 'London').
- Provides airline fleet, alliance, and policy discovery.
"""

import json

from langchain_core.tools import tool

from .db import db


@tool
def search_airports_and_cities(
    query: str,
    country: str | None = None,
) -> str:
    """Discover airports, multi-airport metropolitan areas, and 3-letter IATA codes.

    Agents MUST use this tool when the user provides a city name, airport name, or colloquial location
    (e.g., 'London', 'Paris', 'Tokyo', 'Londo') instead of an explicit 3-letter IATA code.
    If the city has multiple airports (like London with LHR, LGW, STN, LCY, LTN), this tool
    returns all corresponding airports with terminals and timezones.
    If a typo is detected (e.g. 'Londo' instead of 'London'), it proactively suggests the correction.

    Args:
        query: Name of the city (e.g. 'London', 'Paris', 'Tokyo', 'Ho Chi Minh City'),
               airport name (e.g. 'Heathrow', 'Charles de Gaulle'),
               or 3-letter IATA code (e.g. 'LHR', 'CDG', 'SGN').
        country: Optional country name or code to narrow down search (e.g. 'United Kingdom', 'France').

    Returns:
        JSON string representing AirportSearchResult with matching airports, IATA codes, and suggestions.
    """
    try:
        result = db.search_airports_and_cities(query=query, country=country)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to search airports: {e!s}",
            },
            indent=2,
        )


@tool
def get_airline_info(
    query: str,
) -> str:
    """Retrieve airline company profile, baggage allowances, pricing policies, and alliances.

    Useful for agents looking up carrier capabilities, checking whether an airline is part of
    Star Alliance, SkyTeam, or oneworld, or inspecting baggage allowances.

    Args:
        query: 2-letter IATA airline code (e.g. 'BA', 'AF', 'VN', 'SQ', 'JL') or airline name (e.g. 'British Airways').

    Returns:
        JSON string with airline details, baggage limits, pricing multipliers, or typo suggestions.
    """
    try:
        result = db.get_airline_info(query=query)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to retrieve airline info: {e!s}",
            },
            indent=2,
        )
