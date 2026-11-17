class FakeStructuredLLM:
    """Stands in for `llm.with_structured_output(Model).invoke(prompt)` in unit tests,
    so agent node tests don't need a real GOOGLE_API_KEY or network call."""

    def __init__(self, response):
        self._response = response

    def invoke(self, _prompt):
        return self._response


class FakeChatModel:
    """Stands in for ChatGoogleGenerativeAI(...) — returns a FakeStructuredLLM regardless
    of which Pydantic schema is requested, since tests control the canned response directly."""

    def __init__(self, response=None, **_kwargs):
        self._response = response

    def with_structured_output(self, _schema):
        return FakeStructuredLLM(self._response)
