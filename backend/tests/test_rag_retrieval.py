"""Needs a real GOOGLE_API_KEY (Chroma similarity search embeds the query live) and the
index already built via `python -m shopunow.rag.vectorstore`."""

import pytest

from shopunow.data.departments import DEPARTMENT_IDS
from shopunow.rag.retriever import retrieve_for_department

pytestmark = pytest.mark.live


@pytest.mark.parametrize("department_id", DEPARTMENT_IDS)
def test_retrieval_only_returns_docs_for_the_requested_department(department_id):
    results = retrieve_for_department("What is the policy here?", department_id, k=4)
    assert all(doc["department"] == department_id for doc in results)
