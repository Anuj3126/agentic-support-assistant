"""Per-user session memory, keyed by thread_id, so multiple users can each hold their own
conversation state in the same running process (Stretch Option 1). In-process only — no
database file — so history resets on restart, which is an accepted trade-off for staying
free of any SQL/SQLite dependency.
"""

from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer() -> MemorySaver:
    return MemorySaver()


def make_thread_id(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"
