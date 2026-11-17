from shopunow.config import settings
from shopunow.tools import web_search_tool


def test_no_api_key_returns_empty_list_without_calling_tavily(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "")
    assert web_search_tool.web_search.invoke({"query": "anything"}) == []


def test_tavily_network_failure_degrades_to_empty_list_not_a_crash(monkeypatch):
    """The actual bug: an SSLError (or any other network/API failure) from the live Tavily
    call must not propagate and crash the turn — it should look like "no results" instead."""
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key-for-test")

    class _ExplodingClient:
        def __init__(self, api_key):
            pass

        def search(self, query, max_results=3):
            import requests

            raise requests.exceptions.SSLError("certificate verify failed")

    import tavily

    monkeypatch.setattr(tavily, "TavilyClient", _ExplodingClient)

    assert web_search_tool.web_search.invoke({"query": "classic denim jacket stock"}) == []


def test_successful_search_shapes_results(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key-for-test")

    class _FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, query, max_results=3):
            return {"results": [{"title": "T", "url": "https://example.com", "content": "C"}]}

    import tavily

    monkeypatch.setattr(tavily, "TavilyClient", _FakeClient)

    result = web_search_tool.web_search.invoke({"query": "anything"})
    assert result == [{"title": "T", "url": "https://example.com", "content": "C"}]
