"""Agent configuration and LLM factory for FlightBookingAgent."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class AgentConfig:
    """Configuration and guard limits for flight booking agents."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_BASE_URL", "http://localhost:20128/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_MODEL", "openrouter/dots-studio/dots-3-note-preview:free"
        )
    )
    temperature: float = 0.0

    # Runtime guard thresholds
    max_iterations: int = 10
    max_plan_steps: int = 8
    max_replans: int = 3
    micro_max_steps: int = 4
    max_macro_steps: int = 5


def get_llm(config=None, **kwargs):
    """Factory creating an OpenAI-compatible ChatOpenAI instance.

    Uses config if provided, otherwise loads defaults from AgentConfig.
    Allows overriding parameters via kwargs.
    """
    cfg = config or AgentConfig()
    api_key = kwargs.get("api_key", cfg.api_key)
    base_url = kwargs.get("base_url", cfg.base_url)
    model = kwargs.get("model", cfg.model)
    temperature = kwargs.get("temperature", cfg.temperature)

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ("api_key", "base_url", "model", "temperature")
        },
    )
