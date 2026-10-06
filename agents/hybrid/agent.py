"""Hybrid agent orchestrator combining Macro-Milestones with Micro-ReAct loops."""

from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from tools import FLIGHT_BOOKING_TOOLS

from ..base import AgentResponse, BaseFlightAgent, StepRecord
from .schemas import HybridPlan


class HybridStateDict(TypedDict):
    """Dictionary representation of Hybrid Agent state."""

    query: str
    milestones: list
    current_milestone_index: int
    milestone_results: list
    conversation_history: list
    response: str
    stop_reason: str
    session_id: str
    records: list


class HybridFlightAgent(BaseFlightAgent):
    """Hybrid agent: High-level Milestone planning with bounded Micro-ReAct execution."""

    def __init__(self, config=None, llm=None, session_id="default_session", tools=None):
        super().__init__(config=config, llm=llm, session_id=session_id)
        self.tools = tools if tools is not None else FLIGHT_BOOKING_TOOLS
        self.micro_agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )
        self._session_memories: dict[str, dict] = {}
        self._build_graph()

    def _get_session_state(self, session_id: str) -> dict:
        """Fetch or initialize multi-turn memory state for a session."""
        if session_id not in self._session_memories:
            self._session_memories[session_id] = {
                "milestones": [m.model_dump() for m in HybridPlan().milestones],
                "current_milestone_index": 0,
                "milestone_results": [],
                "conversation_history": [],
            }
        return self._session_memories[session_id]

    def _init_plan_node(self, state: HybridStateDict):
        """Initializes or restores standard flight milestones and conversation context."""
        sess_mem = self._get_session_state(state["session_id"])
        records = list(state.get("records", []))

        milestones = sess_mem["milestones"]
        curr_idx = sess_mem["current_milestone_index"]
        prev_results = list(sess_mem["milestone_results"])
        conv_hist = list(sess_mem["conversation_history"])

        # Record initiation or continuation step
        records.append(
            StepRecord(
                step_type="milestone",
                name="macro_init",
                input_data={
                    "query": state["query"],
                    "current_milestone_index": curr_idx,
                },
                output_data={"milestones": [m["name"] for m in milestones]},
            )
        )
        return {
            "milestones": milestones,
            "current_milestone_index": curr_idx,
            "milestone_results": prev_results,
            "conversation_history": conv_hist,
            "records": records,
        }

    def _micro_react_node(self, state: HybridStateDict):
        """Executes a bounded micro-ReAct loop for the current active milestone."""
        idx = state["current_milestone_index"]
        milestones = state["milestones"]
        records = list(state.get("records", []))
        results = list(state.get("milestone_results", []))
        conv_hist = state.get("conversation_history", [])

        if idx >= len(milestones):
            return {"response": "All milestones processed."}

        current_ms = milestones[idx]
        ms_name = current_ms.get("name", "UNKNOWN")
        ms_desc = current_ms.get("description", "")

        records.append(
            StepRecord(
                step_type="milestone",
                name=f"start_milestone_{ms_name}",
                input_data={"milestone": ms_name, "description": ms_desc},
            )
        )

        # Build comprehensive multi-turn dialogue context
        hist_context_lines = []
        for h in conv_hist[-6:]:  # last 3 turns
            hist_context_lines.append(f"{h['role'].upper()}: {h['content']}")
        hist_context_str = (
            "\n".join(hist_context_lines) if hist_context_lines else "None (first turn)"
        )

        micro_prompt = (
            f"You are executing milestone: {ms_name} ({ms_desc}).\n\n"
            f"Conversation History:\n{hist_context_str}\n\n"
            f"Latest User Message: {state['query']}\n"
            f"Prior Milestone Observations: {results}\n\n"
            "OPERATIONAL GUIDELINES:\n"
            "- If the user's latest query refers to flights, offers, or choices already provided in the Conversation History or Prior Observations (e.g. 'earliest arrival time', 'cheapest', 'book the first flight'), DO NOT ask for origin/destination again! Instead, inspect the existing flight details from history and resolve the user request directly.\n"
            "- Only invoke search/airport tools if new flight parameters are introduced or missing.\n"
            "- If the milestone requires user confirmation or traveler details, summarize the options and stop concisely."
        )

        micro_summary = ""
        try:
            micro_res = self.micro_agent.invoke(
                {"messages": [HumanMessage(content=micro_prompt)]},
                config={"recursion_limit": self.config.micro_max_steps * 2 + 2},
            )
            for m in micro_res.get("messages", []):
                if isinstance(m, AIMessage):
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            records.append(
                                StepRecord(
                                    step_type="tool_call",
                                    name=f"[{ms_name}] {tc.get('name')}",
                                    input_data=tc.get("args", {}),
                                )
                            )
                    elif m.content:
                        micro_summary = m.content
                elif isinstance(m, ToolMessage):
                    records.append(
                        StepRecord(
                            step_type="tool_result",
                            name=f"[{ms_name}] {m.name}",
                            output_data=m.content,
                        )
                    )
        except Exception as e:
            micro_summary = f"Milestone {ms_name} encountered error: {e!s}"

        if not micro_summary:
            micro_summary = f"Milestone {ms_name} completed."

        results.append(
            {
                "milestone": ms_name,
                "summary": micro_summary,
            }
        )

        return {
            "current_milestone_index": idx + 1,
            "milestone_results": results,
            "response": micro_summary,
            "records": records,
        }

    def _check_macro_progress(self, state: HybridStateDict):
        """Evaluates whether to continue to next milestone or terminate."""
        idx = state["current_milestone_index"]
        milestones = state["milestones"]
        results = state.get("milestone_results", [])

        # If a milestone ended requiring user decision (e.g. search options presented)
        if results:
            latest_text = results[-1].get("summary", "").lower()
            if any(
                term in latest_text
                for term in [
                    "which flight",
                    "please confirm",
                    "please provide",
                    "choose",
                    "select",
                    "next step",
                    "earliest",
                ]
            ):
                return "halt_user_input"

        if idx >= len(milestones) or idx >= self.config.max_macro_steps:
            return "finish_macro"

        return "continue_macro"

    def _finalize_node(self, state: HybridStateDict):
        """Finalizes response and stop reason."""
        results = state.get("milestone_results", [])
        records = list(state.get("records", []))
        stop_reason = state.get("stop_reason", "completed")

        if results:
            final_content = results[-1].get("summary", "")
        else:
            final_content = state.get("response", "Hybrid process finished.")

        return {
            "response": final_content,
            "stop_reason": stop_reason,
            "records": records,
        }

    def _halt_input_node(self, state: HybridStateDict):
        """Sets stop reason to awaiting_user_input."""
        results = state.get("milestone_results", [])
        return {
            "response": results[-1].get("summary", "")
            if results
            else "Awaiting input.",
            "stop_reason": "awaiting_user_input",
        }

    def _build_graph(self):
        """Builds hierarchical LangGraph StateGraph."""
        builder = StateGraph(HybridStateDict)

        builder.add_node("init_plan", self._init_plan_node)
        builder.add_node("micro_react", self._micro_react_node)
        builder.add_node("halt_input", self._halt_input_node)
        builder.add_node("finalizer", self._finalize_node)

        builder.set_entry_point("init_plan")
        builder.add_edge("init_plan", "micro_react")

        builder.add_conditional_edges(
            "micro_react",
            self._check_macro_progress,
            {
                "continue_macro": "micro_react",
                "halt_user_input": "halt_input",
                "finish_macro": "finalizer",
            },
        )

        builder.add_edge("halt_input", END)
        builder.add_edge("finalizer", END)

        self.graph = builder.compile()

    def invoke(self, query: str, session_id=None):
        """Synchronously execute Hybrid agent with multi-turn memory retention."""
        active_session = session_id or self.session_id
        sess_mem = self._get_session_state(active_session)

        initial_state = {
            "query": query,
            "milestones": sess_mem["milestones"],
            "current_milestone_index": sess_mem["current_milestone_index"],
            "milestone_results": sess_mem["milestone_results"],
            "conversation_history": sess_mem["conversation_history"],
            "response": "",
            "stop_reason": "completed",
            "session_id": active_session,
            "records": [],
        }

        try:
            result = self.graph.invoke(initial_state)
            final_response_text = result.get("response", "Hybrid execution complete.")
            stop_reason = result.get("stop_reason", "completed")

            # Persist session state across turns
            sess_mem["current_milestone_index"] = result.get(
                "current_milestone_index", sess_mem["current_milestone_index"]
            )
            sess_mem["milestone_results"] = result.get(
                "milestone_results", sess_mem["milestone_results"]
            )
            sess_mem["conversation_history"].append({"role": "user", "content": query})
            sess_mem["conversation_history"].append(
                {"role": "assistant", "content": final_response_text}
            )

            return AgentResponse(
                content=final_response_text,
                steps=result.get("records", []),
                session_id=active_session,
                pattern="hybrid",
                stop_reason=stop_reason,
                metadata={"milestones_executed": len(sess_mem["milestone_results"])},
            )
        except Exception as e:
            return AgentResponse(
                content=f"Hybrid Agent error: {e!s}",
                steps=[],
                session_id=active_session,
                pattern="hybrid",
                stop_reason="error",
            )

    def stream(self, query: str, session_id=None):
        """Stream state updates across milestones."""
        active_session = session_id or self.session_id
        sess_mem = self._get_session_state(active_session)
        initial_state = {
            "query": query,
            "milestones": sess_mem["milestones"],
            "current_milestone_index": sess_mem["current_milestone_index"],
            "milestone_results": sess_mem["milestone_results"],
            "conversation_history": sess_mem["conversation_history"],
            "response": "",
            "stop_reason": "completed",
            "session_id": active_session,
            "records": [],
        }
        yield from self.graph.stream(initial_state)

    def reset(self, session_id=None):
        """Reset agent session memory."""
        active_session = session_id or self.session_id
        if active_session in self._session_memories:
            del self._session_memories[active_session]

    def get_history(self, session_id=None):
        """Return history."""
        active_session = session_id or self.session_id
        sess_mem = self._get_session_state(active_session)
        return sess_mem.get("conversation_history", [])
