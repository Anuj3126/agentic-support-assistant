"""Owns the one Chroma collection that backs every department's knowledge base.

Design, in short:
- A SINGLE collection ("shopunow_faqs") holds every department's FAQ docs together, each
  tagged with a "department" metadata field. Retrieval (rag/retriever.py) filters on that
  field, so this is "multiple knowledge bases" in effect without needing multiple databases.
- Gemini's embedding model wants to know whether it's embedding something to be STORED
  (`task_type="RETRIEVAL_DOCUMENT"`) or something being SEARCHED FOR (`"RETRIEVAL_QUERY"`) —
  using the right one on each side measurably improves retrieval quality. That's the only
  reason there are two embedding-function builders below instead of one.
- build_index() is the one-time write path (run once; safe to re-run, it's a no-op once the
  collection already has data unless force=True). get_query_store() is the read path used on
  every search.
"""

import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from shopunow.config import CHROMA_DIR, DATA_DIR, settings
from shopunow.data.departments import DEPARTMENT_IDS

COLLECTION_NAME = "shopunow_faqs"


def _embeddings(task_type: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.google_api_key, task_type=task_type
    )


def _load_qa_documents() -> tuple[list[Document], list[str]]:
    """Reads shopunow/data/qa/<department>.json into Chroma Documents, tagging each with its
    department so retrieval can later filter on it. Returns (documents, ids) in lockstep."""
    documents, ids = [], []
    qa_dir = DATA_DIR / "qa"
    for dept_id in DEPARTMENT_IDS:
        qa_file = qa_dir / f"{dept_id}.json"
        if not qa_file.exists():
            continue
        for i, pair in enumerate(json.loads(qa_file.read_text())):
            documents.append(
                Document(
                    page_content=pair["doc"],
                    metadata={"department": dept_id, "category": pair.get("category", dept_id)},
                )
            )
            ids.append(pair.get("id", f"{dept_id}-{i}"))
    return documents, ids


def build_index(force: bool = False) -> Chroma:
    """Embeds and stores every department's QA docs. Safe to call every startup — it's a
    no-op if the collection is already populated, unless force=True wipes and rebuilds it."""
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings("RETRIEVAL_DOCUMENT"),
        persist_directory=str(CHROMA_DIR),
    )

    existing_ids = store.get()["ids"]
    if existing_ids:
        if not force:
            return store
        store.delete(ids=existing_ids)

    documents, ids = _load_qa_documents()
    if not documents:
        raise RuntimeError("No QA documents found under shopunow/data/qa/ to index.")
    store.add_documents(documents, ids=ids)
    return store


def get_query_store() -> Chroma:
    """The read-side handle used by every search — same collection, query-tuned embeddings."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings("RETRIEVAL_QUERY"),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    store = build_index()
    print(f"index ready with {store._collection.count()} documents")
