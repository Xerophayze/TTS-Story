from __future__ import annotations

from pathlib import Path

import pytest

import app as app_module
from src.gemini_processor import GeminiProcessorError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_profile_chain_supports_multiple_models_and_api_keys_for_one_provider():
    profiles = app_module._resolve_llm_profile_chain({
        "llm_provider": "openrouter",
        "openrouter_model": "model/primary",
        "openrouter_api_key": "primary-key",
        "llm_backup_profiles": [
            {
                "id": "fast-backup",
                "name": "Fast Backup",
                "provider": "openrouter",
                "model": "model/fast",
                "api_key": "fast-key",
            },
            {
                "id": "quality-backup",
                "name": "Quality Backup",
                "provider": "openrouter",
                "model": "model/quality",
                "api_key": "quality-key",
            },
        ],
    })

    assert [(profile["provider"], profile["model"]) for profile in profiles] == [
        ("openrouter", "model/primary"),
        ("openrouter", "model/fast"),
        ("openrouter", "model/quality"),
    ]
    assert profiles[1]["api_key"] == "fast-key"
    assert profiles[2]["api_key"] == "quality-key"


def test_retryable_primary_failure_uses_profile_model_and_key(monkeypatch):
    attempts = []

    def fake_run(prompt, config, provider, model_override=None, api_key_override=None):
        attempts.append((provider, model_override, api_key_override))
        if len(attempts) == 1:
            raise GeminiProcessorError("Gemini API error: 429 RESOURCE_EXHAUSTED")
        return "prepared text"

    monkeypatch.setattr(app_module, "_run_llm_prompt_for_provider", fake_run)
    text, profile, failures = app_module._run_llm_prompt_with_failover(
        "prompt",
        {
            "llm_provider": "gemini",
            "gemini_model": "gemini-primary",
            "llm_backup_profiles": [{
                "id": "router-backup",
                "name": "Router Backup",
                "provider": "openrouter",
                "model": "vendor/backup",
                "api_key": "backup-key",
            }],
        },
    )

    assert text == "prepared text"
    assert profile == {
        "id": "router-backup",
        "name": "Router Backup",
        "provider": "openrouter",
        "model": "vendor/backup",
    }
    assert attempts == [
        ("gemini", "gemini-primary", ""),
        ("openrouter", "vendor/backup", "backup-key"),
    ]
    assert failures[0]["profile_id"] == "primary"
    assert "api_key" not in profile
    assert "api_key" not in failures[0]


def test_configuration_failure_does_not_silently_switch_profile(monkeypatch):
    attempts = []

    def fake_run(prompt, config, provider, model_override=None, api_key_override=None):
        attempts.append(provider)
        raise GeminiProcessorError("Gemini API key not configured")

    monkeypatch.setattr(app_module, "_run_llm_prompt_for_provider", fake_run)

    with pytest.raises(GeminiProcessorError, match="not configured"):
        app_module._run_llm_prompt_with_failover(
            "prompt",
            {
                "llm_provider": "gemini",
                "llm_backup_profiles": [{
                    "provider": "openrouter",
                    "model": "vendor/backup",
                    "api_key": "backup-key",
                }],
            },
        )

    assert attempts == ["gemini"]


def test_preferred_profile_skips_earlier_failures_for_remaining_sections(monkeypatch):
    attempts = []

    def fake_run(prompt, config, provider, model_override=None, api_key_override=None):
        attempts.append((provider, model_override))
        return "next section"

    monkeypatch.setattr(app_module, "_run_llm_prompt_for_provider", fake_run)
    text, profile, failures = app_module._run_llm_prompt_with_failover(
        "prompt",
        {
            "llm_provider": "gemini",
            "llm_backup_profiles": [
                {"id": "router-a", "provider": "openrouter", "model": "model/a"},
                {"id": "router-b", "provider": "openrouter", "model": "model/b"},
            ],
        },
        preferred_profile="router-a",
    )

    assert text == "next section"
    assert profile["id"] == "router-a"
    assert failures == []
    assert attempts == [("openrouter", "model/a")]


def test_job_snapshot_redacts_backup_profile_api_keys():
    snapshot = app_module._redact_config_secrets({
        "openrouter_api_key": "primary-secret",
        "llm_backup_profiles": [{
            "id": "router-backup",
            "provider": "openrouter",
            "model": "model/backup",
            "api_key": "backup-secret",
        }],
    })

    assert snapshot["openrouter_api_key"] == ""
    assert snapshot["llm_backup_profiles"][0]["api_key"] == ""


def test_frontend_builds_dynamic_profiles_and_resumes_active_profile():
    main_source = (PROJECT_ROOT / "static" / "js" / "main.js").read_text(encoding="utf-8")
    settings_source = (PROJECT_ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    template = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert "active_profile: activeProfile || ''" in main_source
    assert "payload.preferred_profile = activeProfile" in main_source
    assert "llm_backup_profiles" in settings_source
    assert "data-profile-field=\"model\"" in settings_source
    assert "data-profile-field=\"api_key\"" in settings_source
    assert 'id="llm-backup-profile-count"' in template
    assert 'id="llm-backup-profiles-list"' in template
