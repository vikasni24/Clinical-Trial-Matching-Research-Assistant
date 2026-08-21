"""GET /health additionally reports whether an LLM provider is configured
— a plain boolean plus the (non-secret) provider name — so the frontend
can show a real AI Assistant status. The API key itself must never appear
in this response under any configuration."""

from app.main import app


def test_health_reports_llm_configured_true_when_key_present(api_client, monkeypatch):
    from app.config import Settings

    monkeypatch.setattr("app.main.get_settings", lambda: Settings(llm_provider="groq", llm_api_key="fake-key-for-test"))

    resp = api_client.get("/health")

    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True
    assert body["llm_provider"] == "groq"


def test_health_reports_llm_configured_false_when_key_absent(api_client, monkeypatch):
    from app.config import Settings

    monkeypatch.setattr("app.main.get_settings", lambda: Settings(llm_api_key=None))

    resp = api_client.get("/health")

    body = resp.json()
    assert body["llm_configured"] is False


def test_health_never_leaks_the_api_key(api_client, monkeypatch):
    from app.config import Settings

    monkeypatch.setattr(
        "app.main.get_settings", lambda: Settings(llm_provider="groq", llm_api_key="gsk_should-never-appear-anywhere")
    )

    resp = api_client.get("/health")

    assert "gsk_should-never-appear-anywhere" not in resp.text
    assert set(resp.json().keys()) == {"status", "llm_configured", "llm_provider"}
