from unittest.mock import Mock

import pytest
import app as app_module


@pytest.fixture
def config(monkeypatch):
    settings = dict(app_module.DEFAULT_CONFIG)
    settings.update(llm_provider="anthropic", anthropic_api_key="test-key",
                    anthropic_model="claude-test")
    monkeypatch.setattr(app_module, "load_config", lambda: settings.copy())
    return settings


def test_provider_chain_routes_to_native_api_and_ignores_shared_sampling(config, monkeypatch):
    from src import anthropic_processor
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return Mock(status_code=200, json=lambda: {
            "stop_reason": "end_turn", "content": [{"type": "text", "text": "prepared"}]})

    monkeypatch.setattr(anthropic_processor.requests, "request", request)
    config.update(llm_local_temperature=0.2, llm_local_top_p=1)
    result, used, failures = app_module._run_llm_prompt_with_failover("source", config)
    assert result == "prepared"
    assert used["provider"] == "anthropic"
    assert used["model"] == "claude-test"
    assert "api_key" not in used
    assert not failures
    assert "temperature" not in calls[0][2]["json"]


def test_model_endpoint_and_saved_key(config, monkeypatch):
    captured = {}

    def discover(**kwargs):
        captured.update(kwargs)
        return ["claude-test"]

    monkeypatch.setattr(app_module.AnthropicProcessor, "list_available_models", discover)
    result = app_module.app.test_client().post("/api/anthropic/models", json={})
    assert result.status_code == 200
    assert result.json == {"success": True, "models": ["claude-test"]}
    assert captured["api_key"] == "test-key"


def test_missing_key_endpoint_is_actionable(config):
    config["anthropic_api_key"] = ""
    result = app_module.app.test_client().post("/api/anthropic/models", json={})
    assert result.status_code == 400
    assert "API key is required" in result.json["error"]


def test_anthropic_429_can_use_configured_backup(config, monkeypatch):
    from src.anthropic_processor import AnthropicProcessorError
    config["llm_backup_profiles"] = [{"id": "backup", "provider": "local", "model": "local-model"}]
    calls = []

    def run(prompt, config, provider, **kwargs):
        calls.append(provider)
        if provider == "anthropic":
            raise AnthropicProcessorError("Claude API error (HTTP 429): rate limit")
        return "prepared by backup"

    monkeypatch.setattr(app_module, "_run_llm_prompt_for_provider", run)
    result, used, failures = app_module._run_llm_prompt_with_failover("source", config)
    assert result == "prepared by backup"
    assert calls == ["anthropic", "local"]
    assert used["provider"] == "local"
    assert len(failures) == 1


def test_key_is_redacted_from_export(config):
    assert "anthropic_api_key" in app_module.SECRET_CONFIG_KEYS
    assert app_module._redact_config_secrets(config)["anthropic_api_key"] == ""


def test_structured_prep_routes_schema_to_claude(config, monkeypatch):
    from src import anthropic_processor
    calls = []
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}

    def request(method, url, **kwargs):
        calls.append(kwargs["json"])
        return Mock(status_code=200, json=lambda: {
            "stop_reason": "end_turn", "content": [{"type": "text", "text": '{"text":"prepared"}'}]})

    monkeypatch.setattr(anthropic_processor.requests, "request", request)
    result, used, failures = app_module._run_llm_prompt_with_failover("source", config, response_schema=schema)
    assert result == '{"text":"prepared"}'
    assert used["provider"] == "anthropic"
    assert not failures
    assert calls[0]["output_config"]["format"]["schema"] == schema
