"""ReAct pattern flight booking agent using LangGraph with integrated Harness Architecture.

Slide Pattern Alignment:
01 Dựng ngữ cảnh (Setup Context) ->
02 Model đề xuất tool (Model Proposes Tool) ->
03 Harness gọi tool (Permission & Loop Check) ->
04 Ghi kết quả (Log Outcomes) ->
05 Xét điều kiện dừng bằng code (Code-checked Done via is_done()).
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from tools import FLIGHT_BOOKING_TOOLS

from ..base import AgentResponse, BaseFlightAgent, StepRecord
from ..harness import (
    Constraints,
    StopAgentException,
    check_hallucination,
    check_permission,
    handoff,
    is_done,
    is_looping,
)
from ..prompts import REACT_SYSTEM_PROMPT


class ReActFlightAgent(BaseFlightAgent):
    """Reasoning + Acting (ReAct) agent implementation for flight booking with Harness protection."""

    def __init__(
        self,
        config=None,
        llm=None,
        session_id="default_session",
        checkpointer=None,
        tools=None,
    ):
        super().__init__(config=config, llm=llm, session_id=session_id)
        self.raw_tools = tools if tools is not None else FLIGHT_BOOKING_TOOLS
        self.checkpointer = checkpointer or MemorySaver()
        self.system_prompt = REACT_SYSTEM_PROMPT
        self._current_constraints = Constraints()
        self._execution_log = []
        self._build_graph()

    def _wrap_tool(self, original_tool):
        """Wraps a tool with Harness permission check and loop detector."""
        name = getattr(original_tool, "name", str(original_tool))
        desc = getattr(original_tool, "description", "")
        args_schema = getattr(original_tool, "args_schema", None)

        def runner(**kwargs):
            if "session_id" not in kwargs or not kwargs.get("session_id"):
                kwargs["session_id"] = getattr(self, "_active_session_id", self.session_id)

            # 1. Harness Permission Check (Blocks forbidden operations before execution)
            allowed, reason = check_permission(name, kwargs, self._current_constraints)
            if not allowed:
                return json.dumps({
                    "status": "denied",
                    "error": f"DENIED by Harness Permission Check: {reason}",
                })

            # 2. Infinite Loop Check
            if is_looping(self._execution_log, threshold=3):
                raise StopAgentException("LOOP: Repeated tool call sequence detected by Harness.")

            # 3. Plain execution of the tool
            if hasattr(original_tool, "invoke"):
                result = original_tool.invoke(kwargs)
            else:
                result = original_tool(**kwargs)

            return result

        return StructuredTool.from_function(
            func=runner,
            name=name,
            description=desc,
            args_schema=args_schema,
        )

    def _build_graph(self):
        """Constructs the prebuilt react agent graph with memory checkpointer and wrapped tools."""
        self.tools = [self._wrap_tool(t) for t in self.raw_tools]
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )

    def invoke(self, query: str, session_id=None, constraints=None):
        """Execute ReAct reasoning loop synchronously with Harness validation."""
        active_session = session_id or self.session_id
        self._active_session_id = active_session
        active_constraints = constraints or self._current_constraints or Constraints()
        self._current_constraints = active_constraints
        self._execution_log = []

        thread_config = {
            "configurable": {"thread_id": active_session},
            "recursion_limit": self.config.max_iterations * 2 + 2,
        }

        steps = []
        stop_reason = "completed"
        final_text = ""
        seen_actions = set()
        booking_attempted = False

        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=thread_config,
            )

            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tc_sig = (
                                tc.get("name"),
                                json.dumps(tc.get("args"), sort_keys=True),
                            )
                            if tc_sig in seen_actions:
                                stop_reason = "duplicate_action_prevented"
                            seen_actions.add(tc_sig)

                            if tc.get("name") == "create_flight_booking":
                                booking_attempted = True

                            record = StepRecord(
                                step_type="tool_call",
                                name=tc.get("name", "unknown_tool"),
                                input_data=tc.get("args", {}),
                            )
                            steps.append(record)
                            self._execution_log.append(record)

                            # Early loop detection check
                            if is_looping(self._execution_log, threshold=3):
                                stop_reason = "duplicate_action_prevented"

                    elif msg.content:
                        final_text = msg.content
                elif isinstance(msg, ToolMessage):
                    record = StepRecord(
                        step_type="tool_result",
                        name=msg.name or "tool_result",
                        output_data=msg.content,
                    )
                    steps.append(record)
                    self._execution_log.append(record)
                    if "denied" in str(msg.content).lower():
                        booking_attempted = True

            if not final_text and messages:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_text = msg.content
                        break

        except StopAgentException:
            stop_reason = "duplicate_action_prevented"
            final_text = (
                "Harness Safety Interruption: Repeated tool loop detected. "
                "Agent stopped to prevent runaway resource consumption."
            )
        except GraphRecursionError:
            stop_reason = "max_iterations"
            final_text = (
                "I reached the maximum operational step limit while processing your flight request. "
                "Here is what I gathered so far. Please refine or clarify your request."
            )
        except Exception as e:
            stop_reason = "error"
            final_text = f"An error occurred during ReAct execution: {e!s}"

        # Code-Checked Done Verification
        done_flag = is_done(active_session, active_constraints)
        metadata = {
            "total_steps": len(steps),
            "is_done": done_flag,
        }

        # Check hallucinations
        hallucinations = check_hallucination(final_text, steps)
        if hallucinations:
            metadata["hallucinations"] = hallucinations

        # If booking was attempted or stopped due to loop/denial, verify done and attach handoff
        if booking_attempted or stop_reason == "duplicate_action_prevented":
            if done_flag:
                stop_reason = "completed"
            else:
                if stop_reason == "completed":
                    stop_reason = "unachievable_goal"
                metadata["handoff"] = handoff(active_session, steps, active_constraints)

        return AgentResponse(
            content=final_text or "No response generated.",
            steps=steps,
            session_id=active_session,
            pattern="react",
            stop_reason=stop_reason,
            metadata=metadata,
        )

    def stream(self, query: str, session_id=None):
        """Stream step updates during ReAct execution."""
        active_session = session_id or self.session_id
        thread_config = {
            "configurable": {"thread_id": active_session},
            "recursion_limit": self.config.max_iterations * 2 + 2,
        }

        yield from self.graph.stream(
            {"messages": [HumanMessage(content=query)]},
            config=thread_config,
            stream_mode="updates",
        )

    def reset(self, session_id=None):
        """Resets thread state by instantiating a fresh checkpointer or clearing thread."""
        self.checkpointer = MemorySaver()
        self._build_graph()

    def get_history(self, session_id=None):
        """Retrieve thread state messages from checkpointer."""
        active_session = session_id or self.session_id
        state = self.graph.get_state({"configurable": {"thread_id": active_session}})
        if state and state.values:
            return state.values.get("messages", [])
        return []
