"""Streamlit Interactive Application for FlightBookingAgent.

Features the 4-layer Harness Architecture:
1. Constraints as DATA (Dynamic Sidebar Control Panel)
2. Permission Check (Pre-tool safety interception)
3. Done checked by CODE (Real-time DB verification badge)
4. Structured Human Handoff (done_so_far, tried, question)

Supports ReAct, Plan-then-Execute (with Human Reviewer approval gate),
and Hybrid patterns, plus an Offline Scripted Demo Switcher.
"""

import json
import os

import streamlit as st

from agents import (
    Constraints,
    ScriptedFakeChatModel,
    get_flight_agent,
)
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
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .badge-completed { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
    .badge-input { background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
    .badge-max { background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
    .badge-other { background-color: #e0e7ff; color: #3730a3; border: 1px solid #6366f1; }
    .badge-harness-ok { background-color: #dcfce7; color: #166534; font-weight: bold; }
    .badge-harness-no { background-color: #f1f5f9; color: #475569; }
    .harness-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .handoff-box {
        background-color: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
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
if "constraints" not in st.session_state:
    st.session_state.constraints = Constraints()
if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = None

# Sidebar Configuration
st.sidebar.title("✈️ Flight Agent Settings")

pattern_option = st.sidebar.selectbox(
    "Agent Architecture Pattern",
    ["react", "plan_then_execute", "hybrid"],
    format_func=lambda x: {
        "react": "1. ReAct (Reasoning + Acting)",
        "plan_then_execute": "2. Plan-then-Execute (One-Shot Plan + Reviewer)",
        "hybrid": "3. Hybrid (Bounded k-Steps + Delta Replan)",
    }.get(x, x),
)

backend_option = st.sidebar.selectbox(
    "Execution Backend",
    [
        "live_api",
        "offline_success",
        "offline_fail_drift",
        "offline_fail_loop",
    ],
    format_func=lambda x: {
        "live_api": "🌐 Live LLM API (.env)",
        "offline_success": "🧪 Offline Demo: Happy Path Success",
        "offline_fail_drift": "🛡️ Offline Demo: Goal Drift Blocked",
        "offline_fail_loop": "🔄 Offline Demo: Loop Interrupted",
    }.get(x, x),
)

# Constraints as DATA Control Panel
st.sidebar.divider()
st.sidebar.subheader("📋 Constraints as DATA")

col_orig, col_dest = st.sidebar.columns(2)
with col_orig:
    c_origin = st.text_input("Origin", value=st.session_state.constraints.origin)
with col_dest:
    c_dest = st.text_input("Destination", value=st.session_state.constraints.destination)

c_date = st.sidebar.text_input("Travel Date", value=st.session_state.constraints.date)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    c_dep_after = st.text_input("Depart After", value=st.session_state.constraints.depart_after)
with col_t2:
    c_dep_before = st.text_input("Depart Before", value=st.session_state.constraints.depart_before)

c_max_price = st.sidebar.number_input(
    "Max Price (VND)",
    min_value=100_000,
    max_value=100_000_000,
    value=st.session_state.constraints.max_price,
    step=200_000,
)

c_pax_name = st.sidebar.text_input("Passenger Name", value=st.session_state.constraints.passenger_name)
c_pax_count = st.sidebar.number_input("Passenger Count", min_value=1, max_value=9, value=st.session_state.constraints.passenger_count)

if st.sidebar.button("🔄 Sync Constraints as DATA"):
    st.session_state.constraints = Constraints(
        origin=c_origin.strip().upper(),
        destination=c_dest.strip().upper(),
        date=c_date.strip(),
        depart_after=c_dep_after.strip(),
        depart_before=c_dep_before.strip(),
        max_price=int(c_max_price),
        passenger_name=c_pax_name.strip(),
        passenger_count=int(c_pax_count),
    )
    st.sidebar.success("Constraints updated!")

st.sidebar.caption(f"**Deterministic Prompt:**\n`{st.session_state.constraints.to_prompt()}`")

if st.sidebar.button("📝 Copy Constraint as Prompt"):
    st.session_state.preset_query = st.session_state.constraints.to_prompt()

# Model and Guards
st.sidebar.divider()
env_model = os.getenv("OPENAI_MODEL", "openrouter/dots-studio/dots-3-note-preview:free")
env_base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")

with st.sidebar.expander("Runtime Guards", expanded=False):
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

if st.sidebar.button("🧹 Clear Chat & Reset Session"):
    st.session_state.messages = []
    st.session_state.last_response = None
    st.session_state.pending_plan = None
    for k in list(st.session_state.keys()):
        if k.startswith("agent_"):
            del st.session_state[k]
    st.rerun()

# Main Header
st.title("🛫 FlightBookingAgent Playground")
st.caption(
    f"Pattern: **{pattern_option.upper()}** | Backend: `{backend_option}` | Session ID: `{st.session_state.session_id}`"
)

col_chat, col_inspect = st.columns([3, 2])

# Helper function to instantiate agent with active backend
def create_active_agent():
    if backend_option == "offline_success":
        fake_llm = ScriptedFakeChatModel(case="success")
        return get_flight_agent(pattern=pattern_option, config=custom_config, llm=fake_llm, session_id=st.session_state.session_id)
    elif backend_option == "offline_fail_drift":
        fake_llm = ScriptedFakeChatModel(case="fail_drift")
        return get_flight_agent(pattern=pattern_option, config=custom_config, llm=fake_llm, session_id=st.session_state.session_id)
    elif backend_option == "offline_fail_loop":
        fake_llm = ScriptedFakeChatModel(case="fail_loop")
        return get_flight_agent(pattern=pattern_option, config=custom_config, llm=fake_llm, session_id=st.session_state.session_id)
    else:
        return get_flight_agent(pattern=pattern_option, config=custom_config, session_id=st.session_state.session_id)

with col_chat:
    st.subheader("💬 Conversation")
    chat_container = st.container()

    new_query = None
    if "preset_query" in st.session_state and st.session_state.preset_query:
        new_query = st.session_state.pop("preset_query")

    chat_input_val = st.chat_input("Ask about flights, bookings, or enter request...")
    if chat_input_val:
        new_query = chat_input_val

    if new_query:
        st.session_state.messages.append({"role": "user", "content": new_query})
        agent = create_active_agent()

        with st.spinner(f"Agent ({pattern_option}) executing with Harness protection..."):
            resp = agent.invoke(
                new_query,
                session_id=st.session_state.session_id,
                constraints=st.session_state.constraints,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "stop_reason": resp.stop_reason,
            }
        )
        st.session_state.last_response = resp
        st.rerun()

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
    st.subheader("🔍 Harness Inspector & Trace")

    resp = st.session_state.last_response
    if resp:
        # 1. Done checked by CODE Badge
        is_done_val = resp.metadata.get("is_done", False)
        st.markdown("#### 🛡️ 4-Layer Harness Status")
        if is_done_val:
            st.markdown(
                '<span class="metric-badge badge-harness-ok">✅ CODE-CHECKED DONE: TRUE (Confirmed in DB)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="metric-badge badge-harness-no">⚪ CODE-CHECKED DONE: FALSE (No valid DB order)</span>',
                unsafe_allow_html=True,
            )

        # 2. Structured Handoff Card
        handoff_data = resp.metadata.get("handoff")
        if handoff_data:
            st.markdown(
                f"""
                <div class="handoff-box">
                    <strong>🤝 Structured Handoff to Human Initiated</strong><br/>
                    <em>{handoff_data.get('question')}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Handoff Data Details", expanded=False):
                st.write("**Done so far:**", handoff_data.get("done_so_far", []))
                st.write("**Tried steps:**", len(handoff_data.get("tried", [])))

        # 3. Permission & Failure Mode Checks
        hallucinations = resp.metadata.get("hallucinations")
        if hallucinations:
            st.error(f"🚨 Hallucination Detected: Flight numbers not in DB tools: {hallucinations}")

        st.divider()
        st.markdown(
            f"**Pattern:** `{resp.pattern}` | **Stop Reason:** `{resp.stop_reason}`"
        )
        st.markdown(f"**Total Trace Steps:** `{len(resp.steps)}`")

        with st.expander("Detailed Execution Trace Steps", expanded=True):
            if not resp.steps:
                st.info("Direct execution without tool steps.")
            else:
                for idx, step in enumerate(resp.steps, 1):
                    s_type = getattr(step, "step_type", "step")
                    s_name = getattr(step, "name", "tool")
                    st.markdown(f"**{idx}. [{s_type.upper()}]** `{s_name}`")
                    inp = getattr(step, "input_data", None)
                    out = getattr(step, "output_data", None)
                    if inp:
                        st.caption("Input:")
                        st.code(
                            json.dumps(inp, indent=2) if isinstance(inp, (dict, list)) else str(inp),
                            language="json",
                        )
                    if out:
                        st.caption("Output:")
                        st.code(
                            json.dumps(out, indent=2) if isinstance(out, (dict, list)) else str(out),
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
