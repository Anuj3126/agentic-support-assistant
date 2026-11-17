"""Everything a department agent needs to get relevant context for a query: look in that
department's slice of the shared Chroma knowledge base first, and fall back to a live web
search only if the knowledge base comes up empty.
"""

from shopunow.config import settings
from shopunow.rag.vectorstore import get_query_store
from shopunow.tools.web_search_tool import to_docs, web_search


def retrieve_for_department(query: str, department: str, k: int = None) -> list[dict]:
    """Chroma similarity search, filtered to one department's documents.

    Returns a list of {"text", "department", "score"} dicts, best match first. Score is
    Chroma's relevance score (higher = more relevant, roughly 0-1).
    """
    store = get_query_store()
    results = store.similarity_search_with_relevance_scores(
        query, k=k or settings.rag_top_k, filter={"department": department}
    )
    return [
        {"text": doc.page_content, "department": doc.metadata.get("department"), "score": score}
        for doc, score in results
    ]


def retrieve_with_fallback(query: str, department: str, k: int = None) -> tuple[list[dict], bool]:
    """The knowledge base is tried first; if it has nothing for this department, a web search
    fills in instead. Returns (docs, used_web_search)."""
    docs = retrieve_for_department(query, department, k)
    if docs:
        return docs, False

    web_docs = to_docs(web_search.invoke({"query": query}), department)
    return web_docs, bool(web_docs)
