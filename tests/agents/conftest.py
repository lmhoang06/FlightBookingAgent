"""Mock test chat model supporting bind_tools and structured output for unit tests."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class ToolCallingFakeChatModel(FakeMessagesListChatModel):
    """FakeChatModel that implements bind_tools for test environments."""

    def bind_tools(self, tools, **kwargs):
        """Returns self so that test mock model can be passed to create_react_agent."""
        return self

    def with_structured_output(self, schema, **kwargs):
        """Returns self for structured output test mocking."""
        return self
