from tests.conftest import test_database_url as database_url_fixture


def test_database_fixture_never_falls_back_to_application_database(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://app/application")
    assert database_url_fixture.__wrapped__() is None


def test_database_fixture_accepts_explicit_test_database(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+asyncpg://test/dedicated")
    assert database_url_fixture.__wrapped__() == "postgresql+asyncpg://test/dedicated"
