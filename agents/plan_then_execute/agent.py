"""Plan-then-Execute agent orchestration using LangGraph StateGraph."""

from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from tools import FLIGHT_BOOKING_TOOLS

from ..base import AgentResponse, BaseFlightAgent, StepRecord
from .planner import Planner, Replanner


class PlanStateDict(TypedDict):
    """Dictionary representation of state for LangGraph nodes."""

    query: str
    plan: list
    completed_steps: list
    current_step: str
    current_step_result: str
    response: str
    replan_count: int
    step_count: int
    stop_reason: str
    session_id: str
    records: list


class PlanThenExecuteFlightAgent(BaseFlightAgent):
    """Plan-then-Execute agent with structured decomposition and replanning."""

    def __init__(self, config=None, llm=None, session_id="default_session", tools=None):
        super().__init__(config=config, llm=llm, session_id=session_id)
        self.tools = tools if tools is not None else FLIGHT_BOOKING_TOOLS
        self.planner = Planner(self.llm)
        self.replanner = Replanner(self.llm)

        # Micro-executor for single steps
        self.executor_llm = self.llm
        self.step_agent = create_react_agent(
            model=self.executor_llm,
            tools=self.tools,
        )
        self._build_graph()

    def _plan_node(self, state: PlanStateDict):
        """Planner node generates initial plan."""
        query = state["query"]
        steps = self.planner.plan(query)
        records = list(state.get("records", []))
        records.append(
            StepRecord(
                step_type="plan",
                name="planner",
                input_data={"query": query},
                output_data={"plan": steps},
            )
        )
        return {
            "plan": steps,
            "records": records,
        }

    def _execute_step_node(self, state: PlanStateDict):
        """Executes the current head step of the plan."""
        plan = state["plan"]
        records = list(state.get("records", []))
        if not plan:
            return {
                "current_step": "",
                "current_step_result": "No pending steps.",
                "step_count": state["step_count"] + 1,
            }

        current_step = plan[0]
        remaining = plan[1:]

        # Execute step using tool-augmented agent
        context = (
            f"Overall Objective: {state['query']}\n"
            f"Current Sub-Task to fulfill: {current_step}\n"
            f"Previously completed sub-tasks: {state.get('completed_steps', [])}\n"
            f"Execute the appropriate tools to accomplish this sub-task."
        )

        try:
            exec_res = self.step_agent.invoke(
                {"messages": [HumanMessage(content=context)]},
                config={"recursion_limit": self.config.micro_max_steps * 2 + 2},
            )
            msgs = exec_res.get("messages", [])
            last_msg = ""
            for m in msgs:
                if isinstance(m, AIMessage):
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            records.append(
                                StepRecord(
                                    step_type="tool_call",
                                    name=tc.get("name", "tool"),
                                    input_data=tc.get("args", {}),
                                )
                            )
                    elif m.content:
                        last_msg = m.content
                elif isinstance(m, ToolMessage):
                    records.append(
                        StepRecord(
                            step_type="tool_result",
                            name=m.name or "tool_result",
                            output_data=m.content,
                        )
                    )
            step_result = last_msg or "Step executed."
        except Exception as e:
            step_result = f"Error during step execution: {e!s}"

        records.append(
            StepRecord(
                step_type="thought",
                name=f"execute_step: {current_step}",
                input_data=current_step,
                output_data=step_result,
            )
        )

        completed = list(state.get("completed_steps", []))
        completed.append({"step": current_step, "result": step_result})

        return {
            "plan": remaining,
            "completed_steps": completed,
            "current_step": current_step,
            "current_step_result": step_result,
            "step_count": state["step_count"] + 1,
            "records": records,
        }

    def _replan_node(self, state: PlanStateDict):
        """Replanner evaluates if goal is satisfied or updates remaining plan."""
        query = state["query"]
        completed = state.get("completed_steps", [])
        remaining = state.get("plan", [])
        latest_res = state.get("current_step_result", "")
        replan_count = state.get("replan_count", 0) + 1
        records = list(state.get("records", []))

        replanned = self.replanner.replan(
            query=query,
            completed_steps=completed,
            remaining_steps=remaining,
            latest_result=latest_res,
        )

        records.append(
            StepRecord(
                step_type="replan",
                name="replanner",
                input_data={"remaining": remaining},
                output_data={
                    "is_complete": replanned.is_complete,
                    "remaining_steps": replanned.remaining_steps,
                    "final_response": replanned.final_response,
                },
            )
        )

        new_plan = replanned.remaining_steps
        response = (
            replanned.final_response
            if replanned.is_complete
            else state.get("response", "")
        )

        return {
            "plan": new_plan,
            "response": response,
            "replan_count": replan_count,
            "records": records,
        }

    def _should_continue(self, state: PlanStateDict):
        """Conditional routing after replanning."""
        if state.get("step_count", 0) >= self.config.max_plan_steps:
            return "finish_max_steps"
        if state.get("replan_count", 0) >= self.config.max_replans:
            return "finish_max_replans"
        if not state.get("plan"):
            return "finish_complete"
        return "continue"

    def _finish_node(self, state: PlanStateDict):
        """Synthesizes final response and sets appropriate stop reason."""
        plan = state.get("plan", [])
        step_count = state.get("step_count", 0)
        replan_count = state.get("replan_count", 0)
        completed = state.get("completed_steps", [])
        response = state.get("response", "")

        stop_reason = "completed"
        if (
            step_count >= self.config.max_plan_steps
            or replan_count >= self.config.max_replans
        ):
            stop_reason = "max_iterations"
        elif not completed:
            stop_reason = "unachievable_goal"

        if not response:
            if completed:
                response = completed[-1].get("result", "Plan execution finished.")
            else:
                response = "Unable to fulfill the flight booking plan."

        return {
            "response": response,
            "stop_reason": stop_reason,
        }

    def _build_graph(self):
        """Constructs the Plan-then-Execute StateGraph."""
        builder = StateGraph(PlanStateDict)

        builder.add_node("planner", self._plan_node)
        builder.add_node("executor", self._execute_step_node)
        builder.add_node("replanner", self._replan_node)
        builder.add_node("finalizer", self._finish_node)

        builder.set_entry_point("planner")
        builder.add_edge("planner", "executor")
        builder.add_edge("executor", "replanner")

        builder.add_conditional_edges(
            "replanner",
            self._should_continue,
            {
                "continue": "executor",
                "finish_max_steps": "finalizer",
                "finish_max_replans": "finalizer",
                "finish_complete": "finalizer",
            },
        )
        builder.add_edge("finalizer", END)

        self.graph = builder.compile()

    def invoke(self, query: str, session_id=None):
        """Executes Plan-then-Execute graph synchronously."""
        active_session = session_id or self.session_id
        initial_state = {
            "query": query,
            "plan": [],
            "completed_steps": [],
            "current_step": "",
            "current_step_result": "",
            "response": "",
            "replan_count": 0,
            "step_count": 0,
            "stop_reason": "completed",
            "session_id": active_session,
            "records": [],
        }

        try:
            result = self.graph.invoke(initial_state)
            return AgentResponse(
                content=result.get("response", "Execution complete."),
                steps=result.get("records", []),
                session_id=active_session,
                pattern="plan_then_execute",
                stop_reason=result.get("stop_reason", "completed"),
                metadata={
                    "completed_steps": len(result.get("completed_steps", [])),
                    "replan_count": result.get("replan_count", 0),
                },
            )
        except Exception as e:
            return AgentResponse(
                content=f"Plan-then-Execute failed: {e!s}",
                steps=[],
                session_id=active_session,
                pattern="plan_then_execute",
                stop_reason="error",
            )

    def stream(self, query: str, session_id=None):
        """Streams state transitions across nodes."""
        active_session = session_id or self.session_id
        initial_state = {
            "query": query,
            "plan": [],
            "completed_steps": [],
            "current_step": "",
            "current_step_result": "",
            "response": "",
            "replan_count": 0,
            "step_count": 0,
            "stop_reason": "completed",
            "session_id": active_session,
            "records": [],
        }
        yield from self.graph.stream(initial_state)

    def reset(self, session_id=None):
        """Resets agent."""

    def get_history(self, session_id=None):
        """Returns empty history or last runs."""
        return []
