"""Per-user session memory, keyed by thread_id, so each of the 5 login names holds its own
conversation state in the same running process (Stretch Option 1). In-process only — no
database file — so history resets on restart, which is an accepted trade-off for staying
free of a SQLite dependency for session memory specifically (the business data has its own
SQLite db, seeded separately — see db/init_db.py).
"""

from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer() -> MemorySaver:
    return MemorySaver()


def make_thread_id(name: str) -> str:
    return name.strip().lower().replace(" ", "_")
