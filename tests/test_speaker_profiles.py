from __future__ import annotations

from pathlib import Path

import app as app_module
from app import (
    build_speaker_profile_excerpts,
    compose_gemini_speaker_profile_prompt,
    parse_gemini_speaker_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_profile_excerpts_are_bounded_and_grouped_by_exact_speaker():
    text = (
        "[narrator]The storm surrounded the old house.[/narrator]\n"
        "[alice-female]We should leave now.[/alice-female]\n"
        "[narrator]Alice watched the windows shake.[/narrator]"
    )

    result = build_speaker_profile_excerpts(
        text,
        ["narrator", "alice-female"],
        max_chars_per_speaker=30,
        max_total_chars=500,
    )

    assert "Speaker: narrator" in result
    assert "Speaker: alice-female" in result
    assert "We should leave now." in result
    assert len(result) < 500


def test_profile_prompt_enforces_complete_voice_table():
    prompt = compose_gemini_speaker_profile_prompt(
        "Analyze the characters.",
        ["narrator", "alice-female"],
        processed_text="Speaker: alice-female\nExcerpts: Hello.",
    )

    assert "Create one row for every exact speaker ID" in prompt
    assert "Character Name | Full Description | Voice Profile" in prompt
    assert "do not return blank cells" in prompt


def test_profile_parser_accepts_markdown_table_and_skips_headers():
    response = """
| Character Name | Full Description | Voice Profile |
| :--- | :--- | :--- |
| narrator | A measured adult storyteller with restrained dramatic warmth. | Warm, steady baritone |
| alice-female | A determined young woman whose urgency remains articulate. | Clear, urgent alto |
"""

    profiles = parse_gemini_speaker_table(response)

    assert set(profiles) == {"narrator", "alice-female"}
    assert profiles["alice-female"]["voice"] == "Clear, urgent alto"


def test_profile_parser_accepts_json_field_variations():
    response = """```json
[
  {
    "speaker": "alice-female",
    "profile": "A thoughtful young woman with a careful delivery.",
    "voice_type": "Soft, reflective alto"
  }
]
```"""

    profiles = parse_gemini_speaker_table(response)

    assert profiles["alice-female"] == {
        "name": "alice-female",
        "description": "A thoughtful young woman with a careful delivery.",
        "voice": "Soft, reflective alto",
    }


def test_profile_parser_rejects_blank_profile_cells():
    response = "| alice-female |  | Soft alto |"

    assert parse_gemini_speaker_table(response) == {}


def test_generate_page_offers_profile_retry_without_reprocessing_text():
    template = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "static" / "js" / "main.js").read_text(encoding="utf-8")

    assert 'id="build-speaker-profiles-btn"' in template
    assert "await fetchSpeakerProfiles();" in javascript
    assert "buildProfilesBtn.disabled = !hasDetectedSpeakers" in javascript


def test_speaker_properties_offers_single_profile_generation():
    javascript = (PROJECT_ROOT / "static" / "js" / "main.js").read_text(encoding="utf-8")

    assert 'data-role="speaker-build-profile"' in javascript
    assert "async function buildSingleSpeakerProfile(speaker)" in javascript
    assert "speakers: [speaker]" in javascript
    assert "processed_text: processedText" in javascript
    assert "updateSpeakerProfileEntry(speaker" in javascript


def test_single_profile_request_sends_only_the_selected_speakers_excerpt(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        app_module,
        "load_config",
        lambda: {
            "llm_provider": "local",
            "gemini_speaker_profile_prompt": "Analyze this speaker.",
        },
    )

    def fake_run(prompt, _config):
        captured["prompt"] = prompt
        return (
            "| Character Name | Full Description | Voice Profile |\n"
            "| --- | --- | --- |\n"
            "| alice-female | A decisive investigator who speaks with calm urgency. | Clear, focused alto |",
            {"id": "primary", "name": "Primary", "provider": "local", "model": "test-model"},
            [],
        )

    monkeypatch.setattr(app_module, "_run_llm_prompt_with_failover", fake_run)
    response = app_module.app.test_client().post(
        "/api/gemini/speaker-profiles",
        json={
            "speakers": ["alice-female"],
            "processed_text": (
                "[alice-female]We leave before sunrise.[/alice-female]\n"
                "[bob-male]I will stay here.[/bob-male]"
            ),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload["profiles"]) == {"alice-female"}
    assert "We leave before sunrise." in captured["prompt"]
    assert "I will stay here." not in captured["prompt"]
