from shopunow.db import identity


def test_resolves_a_real_customer(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(identity, "BUSINESS_DB_PATH", seeded_sqlite_db)
    result = identity.resolve_identity("returns", "Priya Sharma")
    assert result == {"kind": "customer", "id": 1, "name": "Priya Sharma"}


def test_resolves_a_real_employee(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(identity, "BUSINESS_DB_PATH", seeded_sqlite_db)
    result = identity.resolve_identity("hr", "Asha Nair")
    assert result == {"kind": "employee", "id": 1, "name": "Asha Nair"}


def test_unrecognized_name_resolves_to_none_not_a_fabricated_identity(seeded_sqlite_db, monkeypatch):
    """This is the actual fix for "I'm Priya... I'm Jackson both being accepted" — a name
    that isn't a real row must resolve to nothing, not be trusted at face value."""
    monkeypatch.setattr(identity, "BUSINESS_DB_PATH", seeded_sqlite_db)
    assert identity.resolve_identity("returns", "Jackson Nobody") is None


def test_department_without_an_identity_concept_always_returns_none(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(identity, "BUSINESS_DB_PATH", seeded_sqlite_db)
    assert identity.resolve_identity("facilities", "Priya Sharma") is None


def test_no_login_name_resolves_to_none(seeded_sqlite_db, monkeypatch):
    monkeypatch.setattr(identity, "BUSINESS_DB_PATH", seeded_sqlite_db)
    assert identity.resolve_identity("returns", None) is None
