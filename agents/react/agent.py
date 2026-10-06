"""ReAct pattern flight booking agent using LangGraph and standard flight tools."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from tools import FLIGHT_BOOKING_TOOLS

from ..base import AgentResponse, BaseFlightAgent, StepRecord
from ..prompts import REACT_SYSTEM_PROMPT


class ReActFlightAgent(BaseFlightAgent):
    """Reasoning + Acting (ReAct) agent implementation for flight booking."""

    def __init__(
        self,
        config=None,
        llm=None,
        session_id="default_session",
        checkpointer=None,
        tools=None,
    ):
        super().__init__(config=config, llm=llm, session_id=session_id)
        self.tools = tools if tools is not None else FLIGHT_BOOKING_TOOLS
        self.checkpointer = checkpointer or MemorySaver()
        self.system_prompt = REACT_SYSTEM_PROMPT
        self._build_graph()

    def _build_graph(self):
        """Constructs the prebuilt react agent graph with memory checkpointer."""
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )

    def invoke(self, query: str, session_id=None):
        """Execute ReAct reasoning loop synchronously."""
        active_session = session_id or self.session_id
        thread_config = {
            "configurable": {"thread_id": active_session},
            "recursion_limit": self.config.max_iterations * 2 + 2,
        }

        steps = []
        stop_reason = "completed"
        final_text = ""
        seen_actions = set()

        try:
            # Execute graph
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

                            steps.append(
                                StepRecord(
                                    step_type="tool_call",
                                    name=tc.get("name", "unknown_tool"),
                                    input_data=tc.get("args", {}),
                                )
                            )
                    elif msg.content:
                        final_text = msg.content
                elif isinstance(msg, ToolMessage):
                    steps.append(
                        StepRecord(
                            step_type="tool_result",
                            name=msg.name or "tool_result",
                            output_data=msg.content,
                        )
                    )

            if not final_text and messages:
                # Find last AIMessage content
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_text = msg.content
                        break

        except GraphRecursionError:
            stop_reason = "max_iterations"
            final_text = (
                "I reached the maximum operational step limit while processing your flight request. "
                "Here is what I gathered so far. Please refine or clarify your request."
            )
        except Exception as e:
            stop_reason = "error"
            final_text = f"An error occurred during ReAct execution: {e!s}"

        return AgentResponse(
            content=final_text or "No response generated.",
            steps=steps,
            session_id=active_session,
            pattern="react",
            stop_reason=stop_reason,
            metadata={"total_steps": len(steps)},
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
