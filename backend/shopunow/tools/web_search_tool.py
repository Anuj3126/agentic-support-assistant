"""Web search, used as a fallback whenever the internal knowledge base has nothing relevant.

Requires TAVILY_API_KEY. If it's not configured, or the live search call itself fails for any
reason (network, SSL, rate limit, API error), `web_search` returns an empty list instead of
raising, so callers can fall through to "no relevant documents found" rather than crashing
the whole turn.
"""

from langchain_core.tools import tool

from shopunow.config import settings


@tool
def web_search(query: str) -> list[dict]:
    """Search the public web for an answer when the internal knowledge base has nothing
    relevant. Returns a list of {"title", "url", "content"} results."""
    if not settings.tavily_api_key:
        return []
    from tavily import TavilyClient

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query, max_results=3)
    except Exception:
        return []
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in response.get("results", [])
    ]


def to_docs(results: list[dict], department: str) -> list[dict]:
    """Reshapes web_search's output into the same {"text", "department", "score"} shape
    Chroma retrieval returns, so downstream code can treat both sources identically.
    Score is None since web results aren't ranked on the same scale as vector similarity.
    """
    return [
        {"text": f"{r['title']}: {r['content']}", "department": department, "score": None}
        for r in results
    ]
