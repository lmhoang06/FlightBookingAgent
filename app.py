"""Streamlit Interactive Application for FlightBookingAgent.

Provides an interface to test ReAct, Plan-then-Execute, and Hybrid patterns,
inspect tool call executions, trace states, and review mock database bookings.
"""

import json
import os

import streamlit as st

from agents import get_flight_agent
from agents.config import AgentConfig
from tools.db import db

st.set_page_config(
    page_title="FlightBookingAgent Playground",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-completed { background-color: #d1fae5; color: #065f46; }
    .badge-input { background-color: #fef3c7; color: #92400e; }
    .badge-max { background-color: #fee2e2; color: #991b1b; }
    .badge-other { background-color: #e0e7ff; color: #3730a3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_streamlit"
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Sidebar Configuration
st.sidebar.title("✈️ Flight Agent Settings")

pattern_option = st.sidebar.selectbox(
    "Agent Architecture Pattern",
    ["react", "plan_then_execute", "hybrid"],
    format_func=lambda x: {
        "react": "1. ReAct (Reasoning + Acting)",
        "plan_then_execute": "2. Plan-then-Execute (StateGraph)",
        "hybrid": "3. Hybrid (Macro Milestone + Micro ReAct)",
    }.get(x, x),
)

# Read environment settings from .env
env_model = os.getenv("OPENAI_MODEL", "openrouter/dots-studio/dots-3-note-preview:free")
env_base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")

with st.sidebar.expander("Model Configuration (.env)", expanded=False):
    st.markdown(f"**Model:** `{env_model}`")
    st.markdown(f"**Base URL:** `{env_base_url}`")
    st.caption("ℹ️ Model configuration is read-only. Edit `.env` to change settings.")

with st.sidebar.expander("Runtime Guard Config", expanded=False):
    max_iters = st.number_input("Max Iterations", 2, 20, 10)
    max_plan_steps = st.number_input("Max Plan Steps", 2, 15, 8)
    max_replans = st.number_input("Max Replans", 1, 5, 3)
    micro_max_steps = st.number_input("Micro Max Steps", 1, 8, 4)

custom_config = AgentConfig(
    model=env_model,
    base_url=env_base_url,
    max_iterations=max_iters,
    max_plan_steps=max_plan_steps,
    max_replans=max_replans,
    micro_max_steps=micro_max_steps,
)

st.sidebar.divider()
st.sidebar.subheader("Quick Presets")
presets = [
    "Find airport code for Da Nang and Hanoi",
    "Search flights from SGN to HAN for 1 adult on 2026-10-15",
    "What is Vietnam Airlines baggage policy?",
    "Check session status",
]
for p in presets:
    if st.sidebar.button(p, key=f"preset_{p}"):
        st.session_state.preset_query = p

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_response = None
    # Remove cached agent instances so memory resets cleanly
    for k in list(st.session_state.keys()):
        if k.startswith("agent_"):
            del st.session_state[k]
    st.rerun()

# Main Header
st.title("🛫 FlightBookingAgent Playground")
st.caption(
    f"Active Pattern: **{pattern_option.upper()}** | Session ID: `{st.session_state.session_id}`"
)

# Layout: Split Chat and Execution Inspection
col_chat, col_inspect = st.columns([3, 2])

with col_chat:
    st.subheader("💬 Conversation")

    # Scrollable container for chat history
    chat_container = st.container()

    # Process new prompt before rendering so the new message and answer render inside history
    new_query = None
    if "preset_query" in st.session_state and st.session_state.preset_query:
        new_query = st.session_state.pop("preset_query")

    chat_input_val = st.chat_input("Ask about flights, airports, or bookings...")
    if chat_input_val:
        new_query = chat_input_val

    if new_query:
        st.session_state.messages.append({"role": "user", "content": new_query})

        # Retain agent in st.session_state so multi-turn checkpoint memory persists
        agent_key = f"agent_{pattern_option}_{st.session_state.session_id}"
        if agent_key not in st.session_state:
            st.session_state[agent_key] = get_flight_agent(
                pattern=pattern_option,
                config=custom_config,
                session_id=st.session_state.session_id,
            )
        agent = st.session_state[agent_key]

        with st.spinner(f"Agent ({pattern_option}) reasoning..."):
            resp = agent.invoke(new_query, session_id=st.session_state.session_id)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "stop_reason": resp.stop_reason,
            }
        )
        st.session_state.last_response = resp
        st.rerun()

    # Render all conversation turns in order inside container
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "stop_reason" in msg:
                    stop_r = msg["stop_reason"]
                    badge_class = (
                        "badge-completed"
                        if stop_r == "completed"
                        else "badge-input"
                        if stop_r == "awaiting_user_input"
                        else "badge-max"
                        if stop_r in ("max_iterations", "unachievable_goal")
                        else "badge-other"
                    )
                    st.markdown(
                        f'<span class="metric-badge {badge_class}">Status: {stop_r.upper()}</span>',
                        unsafe_allow_html=True,
                    )

with col_inspect:
    st.subheader("🔍 Execution Trace & Diagnostics")

    resp = st.session_state.last_response
    if resp:
        st.markdown(
            f"**Pattern:** `{resp.pattern}` | **Stop Reason:** `{resp.stop_reason}`"
        )
        st.markdown(f"**Total Trace Steps:** `{len(resp.steps)}`")

        with st.expander("Detailed Reasoning Steps", expanded=True):
            if not resp.steps:
                st.info("Direct generation without tool or sub-step calls.")
            else:
                for idx, step in enumerate(resp.steps, 1):
                    st.markdown(f"**{idx}. [{step.step_type.upper()}]** `{step.name}`")
                    if step.input_data:
                        st.caption("Input:")
                        st.code(
                            json.dumps(step.input_data, indent=2)
                            if isinstance(step.input_data, (dict, list))
                            else str(step.input_data),
                            language="json",
                        )
                    if step.output_data:
                        st.caption("Output:")
                        st.code(
                            json.dumps(step.output_data, indent=2)
                            if isinstance(step.output_data, (dict, list))
                            else str(step.output_data),
                            language="json",
                        )
                    st.divider()

    with st.expander("📦 Mock Database Inspector", expanded=False):
        sess_status = db.get_session_status(st.session_state.session_id)
        session_orders = db._orders_by_session.get(st.session_state.session_id, {})
        unique_orders = list(
            {o.booking_reference: o for o in session_orders.values()}.values()
        )
        st.markdown(
            f"**Active Bookings for Session (`{st.session_state.session_id}`):** {sess_status['active_confirmed_bookings_count']}"
        )
        st.caption(f"Searches executed in session: {sess_status['searches_executed']}")
        if unique_orders:
            for b in unique_orders:
                st.json(b.model_dump())
        else:
            st.caption("No bookings created yet for this session.")
