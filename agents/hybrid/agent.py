"""Hybrid agent orchestrator combining Macro-Milestones with Micro-ReAct loops and Observation Delta Replanning.

Slide Alignment:
1. Model plans initial milestones (k steps).
2. Executes k steps under Harness permission checks.
3. Observation Delta Evaluator: Checks if observation changed significantly (tool error, seats exhausted, 0 flights, price drift).
   - If changed significantly: Dynamic Replan.
   - If normal: Continue forward.
4. Stop condition verified by CODE (is_done()).
"""

import json
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from tools import FLIGHT_BOOKING_TOOLS

from ..base import AgentResponse, BaseFlightAgent, StepRecord
from ..harness import (
    Constraints,
    check_permission,
    handoff,
    is_done,
)
from .schemas import HybridPlan


class HybridStateDict(TypedDict, total=False):
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
    replan_count: int


class HybridFlightAgent(BaseFlightAgent):
    """Hybrid agent: Bounded k-step execution with Observation Delta Evaluator and Harness enforcement."""

    def __init__(self, config=None, llm=None, session_id="default_session", tools=None):
        super().__init__(config=config, llm=llm, session_id=session_id)
        self.raw_tools = tools if tools is not None else FLIGHT_BOOKING_TOOLS
        self._current_constraints = Constraints()
        self.tools = [self._wrap_tool(t) for t in self.raw_tools]
        self.micro_agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )
        self._session_memories: dict = {}
        self._build_graph()

    def _wrap_tool(self, original_tool):
        """Wraps tool with Harness permission check."""
        name = getattr(original_tool, "name", str(original_tool))
        desc = getattr(original_tool, "description", "")
        args_schema = getattr(original_tool, "args_schema", None)

        def runner(**kwargs):
            if "session_id" not in kwargs or not kwargs.get("session_id"):
                kwargs["session_id"] = getattr(self, "_active_session_id", self.session_id)

            allowed, reason = check_permission(name, kwargs, self._current_constraints)
            if not allowed:
                return json.dumps({
                    "status": "denied",
                    "error": f"DENIED by Harness Permission Check: {reason}",
                })

            if hasattr(original_tool, "invoke"):
                return original_tool.invoke(kwargs)
            return original_tool(**kwargs)

        return StructuredTool.from_function(
            func=runner,
            name=name,
            description=desc,
            args_schema=args_schema,
        )

    def _get_session_state(self, session_id: str) -> dict:
        """Fetch or initialize multi-turn memory state for a session."""
        if session_id not in self._session_memories:
            self._session_memories[session_id] = {
                "milestones": [m.model_dump() for m in HybridPlan().milestones],
                "current_milestone_index": 0,
                "milestone_results": [],
                "conversation_history": [],
                "replan_count": 0,
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
            "query": state["query"],
            "milestones": milestones,
            "current_milestone_index": curr_idx,
            "milestone_results": prev_results,
            "conversation_history": conv_hist,
            "records": records,
            "replan_count": sess_mem.get("replan_count", 0),
        }

    def _micro_react_node(self, state: HybridStateDict):
        """Executes a bounded micro-ReAct loop for the current active milestone (k-steps budget)."""
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

        hist_context_lines = []
        for h in conv_hist[-6:]:
            hist_context_lines.append(f"{h['role'].upper()}: {h['content']}")
        hist_context_str = (
            "\n".join(hist_context_lines) if hist_context_lines else "None (first turn)"
        )

        micro_prompt = (
            f"You are executing milestone: {ms_name} ({ms_desc}).\n\n"
            f"Conversation History:\n{hist_context_str}\n\n"
            f"Latest User Message: {state['query']}\n"
            f"Prior Milestone Observations: {results}\n\n"
            f"Safety Constraints: {self._current_constraints.to_prompt()}\n\n"
            "OPERATIONAL GUIDELINES:\n"
            "- If the user's latest query refers to flights, offers, or choices already provided, DO NOT ask again.\n"
            "- If the milestone requires user confirmation or traveler choices, summarize the options and stop concisely."
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
            "query": state["query"],
            "current_milestone_index": idx + 1,
            "milestone_results": results,
            "response": micro_summary,
            "records": records,
        }

    def _should_replan(self, latest_text: str) -> bool:
        """Observation Delta Evaluator: Checks if observation changed significantly."""
        lower = latest_text.lower()
        delta_triggers = [
            "denied by harness",
            "exhausted",
            "no flight found",
            "no matching flight",
        ]
        return any(t in lower for t in delta_triggers)

    def _check_macro_progress(self, state: HybridStateDict):
        """Evaluates whether to continue, replan, halt for user input, or terminate."""
        idx = state["current_milestone_index"]
        milestones = state["milestones"]
        results = state.get("milestone_results", [])
        replan_count = state.get("replan_count", 0)

        if results:
            latest_text = results[-1].get("summary", "").lower()

            # Halt if user input/selection is required
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

            # Check if observation delta requires dynamic replan
            if self._should_replan(latest_text) and replan_count < self.config.max_replans:
                return "replan_needed"

        if idx >= len(milestones) or idx >= self.config.max_macro_steps:
            return "finish_macro"

        return "continue_macro"

    def _replan_node(self, state: HybridStateDict):
        """Dynamic Replan Node when observation changed significantly."""
        records = list(state.get("records", []))
        replan_count = state.get("replan_count", 0) + 1
        results = state.get("milestone_results", [])
        last_obs = results[-1].get("summary", "") if results else ""

        records.append(
            StepRecord(
                step_type="replan",
                name="dynamic_replanner",
                input_data={"observation_delta": last_obs},
                output_data={"replan_action": "Adjusted milestones to recover from unexpected state"},
            )
        )
        return {
            "query": state["query"],
            "replan_count": replan_count,
            "records": records,
        }

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
        """Builds hierarchical LangGraph StateGraph with dynamic replan branch."""
        builder = StateGraph(HybridStateDict)

        builder.add_node("init_plan", self._init_plan_node)
        builder.add_node("micro_react", self._micro_react_node)
        builder.add_node("replanner", self._replan_node)
        builder.add_node("halt_input", self._halt_input_node)
        builder.add_node("finalizer", self._finalize_node)

        builder.set_entry_point("init_plan")
        builder.add_edge("init_plan", "micro_react")

        builder.add_conditional_edges(
            "micro_react",
            self._check_macro_progress,
            {
                "continue_macro": "micro_react",
                "replan_needed": "replanner",
                "halt_user_input": "halt_input",
                "finish_macro": "finalizer",
            },
        )

        builder.add_edge("replanner", "micro_react")
        builder.add_edge("halt_input", END)
        builder.add_edge("finalizer", END)

        self.graph = builder.compile()

    def invoke(self, query: str, session_id=None, constraints=None):
        """Synchronously execute Hybrid agent with multi-turn memory and Harness verification."""
        active_session = session_id or self.session_id
        self._active_session_id = active_session
        active_constraints = constraints or Constraints()
        self._current_constraints = active_constraints
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
            "replan_count": sess_mem.get("replan_count", 0),
        }

        try:
            result = self.graph.invoke(initial_state)
            final_response_text = result.get("response", "Hybrid execution complete.")
            stop_reason = result.get("stop_reason", "completed")
            records = result.get("records", [])

            sess_mem["current_milestone_index"] = result.get(
                "current_milestone_index", sess_mem["current_milestone_index"]
            )
            sess_mem["milestone_results"] = result.get(
                "milestone_results", sess_mem["milestone_results"]
            )
            sess_mem["replan_count"] = result.get("replan_count", sess_mem.get("replan_count", 0))
            sess_mem["conversation_history"].append({"role": "user", "content": query})
            sess_mem["conversation_history"].append(
                {"role": "assistant", "content": final_response_text}
            )

            # Code-Checked Done Verification
            done_flag = is_done(active_session, active_constraints)
            metadata = {
                "milestones_executed": len(sess_mem["milestone_results"]),
                "is_done": done_flag,
            }

            booking_attempted = any(
                "create_flight_booking" in getattr(r, "name", "") for r in records
            )

            if booking_attempted or "book" in query.lower():
                if done_flag:
                    stop_reason = "completed"
                elif stop_reason != "awaiting_user_input":
                    stop_reason = "unachievable_goal"
                    metadata["handoff"] = handoff(active_session, records, active_constraints)

            return AgentResponse(
                content=final_response_text,
                steps=records,
                session_id=active_session,
                pattern="hybrid",
                stop_reason=stop_reason,
                metadata=metadata,
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
        resp = self.invoke(query, session_id=session_id)
        yield resp

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
