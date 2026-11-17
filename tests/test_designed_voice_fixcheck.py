"""Fix-checking verification tests for the designed-voice approval lifecycle bugfix.

Spec: .kiro/specs/designed-voice-approval-lifecycle/

**Property 1: Expected Behavior - Designed Voice Lifecycle Completeness**

Companion to ``tests/test_designed_voice_approval.py`` (the task-1 bug-condition
tests, which now encode and PASS the expected behavior). This file EXTENDS that
verification with the full post-fix lifecycle:

  * PBT (fix-checking): for designed-voice records with random valid (>=5s)
    durations, names, languages, and genders assert a non-null deterministic
    ``voice_code``; approval to ``approval_status == "approved"``; appearance in
    both ``GET /api/voices`` ``custom_voices`` and ``GET /api/custom-voices``; and
    a sample route that returns the exact stored bytes.
  * Code determinism unit test for ``_designed_voice_code``.
  * Full lifecycle integration test (save -> result shape + voice_code -> approve
    -> both listings -> byte-for-byte sample).
  * Generation acceptance: the approved ``voice_code`` resolves to the stored
    ``file_name`` at the resolution layer (robust to missing TTS engines).
  * Restart persistence: reload the registry from disk and confirm the voice is
    still listed and previewable from persisted ``approval_status`` alone.
  * Pending visibility: a pending record is previewable (sample) and inspectable
    (detail) but absent from both usable-voice listings until approved.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import app as app_module


SAMPLE_RATE = 24_000  # 24 kHz mono, matching the designed-voice WAV contract
MIN_SECONDS = app_module.MIN_CHATTERBOX_PROMPT_SECONDS  # 5.0


# ---------------------------------------------------------------------------
# WAV synthesis helpers (reused from the sibling test files)
# ---------------------------------------------------------------------------
def _synth_wav_bytes(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Synthesize a mono PCM16 WAV of the requested duration as raw bytes."""
    n_samples = max(1, int(round(duration_seconds * sample_rate)))
    t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)
    signal = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _synth_wav_base64(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> str:
    return base64.b64encode(_synth_wav_bytes(duration_seconds, sample_rate)).decode("ascii")


# ---------------------------------------------------------------------------
# Fixtures: isolate the registry + prompt dir to a temp path per test
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_voice_store(tmp_path, monkeypatch):
    """Point the chatterbox registry and voice-prompt dir at a fresh temp path."""
    prompt_dir = tmp_path / "voice_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "chatterbox_voices.json"

    monkeypatch.setattr(app_module, "VOICE_PROMPT_DIR", prompt_dir)
    monkeypatch.setattr(app_module, "CHATTERBOX_VOICE_REGISTRY", registry)
    monkeypatch.setattr(app_module, "_voice_prompt_transcript_cache", None, raising=False)
    return {"prompt_dir": prompt_dir, "registry": registry}


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _seed_designed_voice(
    duration_seconds: float,
    name: str = "Fixcheck Voice",
    language: str = "English",
    gender: str = "Female",
):
    """Seed a designed-voice record via the real save helper.

    Returns ``(record, raw_entry)`` where ``record`` is the serialized save
    result and ``raw_entry`` is the persisted registry entry.
    """
    payload = {
        "name": name,
        "text": (
            "This is a synthesized reference clip used to seed a designed voice "
            "record for the approval lifecycle fix-checking tests."
        ),
        "gender": gender,
        "language": language,
        "description": "Seeded by fix-checking verification tests.",
        "engine": "qwen3_voice_design",
        "audio_base64": _synth_wav_base64(duration_seconds),
        # Skip the cleanup pass so the stored bytes match the synthesized input.
        "cleanup_applied": True,
    }
    record = app_module._save_voice_design_payload(payload)
    entries = app_module._load_chatterbox_voice_entries()
    raw_entry = app_module._resolve_chatterbox_voice(record["id"], entries)
    return record, raw_entry


def _listed_in_custom_voices(client, record_id: str, voice_code: str) -> bool:
    coll = client.get("/api/custom-voices")
    assert coll.status_code == 200
    voices = coll.get_json().get("voices", [])
    return any(
        v.get("id") == record_id
        or v.get("voice_code") == voice_code
        or v.get("code") == voice_code
        for v in voices
    )


def _collect_custom_voices_from_voices_payload(client) -> list:
    resp = client.get("/api/voices")
    assert resp.status_code == 200
    payload = resp.get_json().get("voices", {})
    collected: list = []
    if isinstance(payload, dict):
        for group in payload.values():
            if isinstance(group, dict):
                collected.extend(group.get("custom_voices", []) or [])
            elif isinstance(group, list):
                for lang_group in group:
                    if isinstance(lang_group, dict):
                        collected.extend(lang_group.get("custom_voices", []) or [])
    return collected


def _listed_in_voices(client, record_id: str, voice_code: str) -> bool:
    return any(
        v.get("id") == record_id
        or v.get("code") == voice_code
        or v.get("voice_code") == voice_code
        for v in _collect_custom_voices_from_voices_payload(client)
    )


# ===========================================================================
# Code determinism unit test (req 2.1)
# ===========================================================================
def test_designed_voice_code_is_deterministic_unique_and_prefixed():
    """``_designed_voice_code`` is stable per id, unique across ids, ``designed_``.

    - Stable across repeated calls for the same record id.
    - Unique per record id (different ids -> different codes).
    - Prefixed ``designed_`` and never ``custom_`` (no Kokoro collision).

    **Validates: Requirements 2.1**
    """
    entry_a = {"id": "11111111-aaaa", "voice_design": {"engine": "qwen3_voice_design"}}
    entry_b = {"id": "22222222-bbbb", "voice_design": {"engine": "qwen3_voice_design"}}

    code_a1 = app_module._designed_voice_code(entry_a)
    code_a2 = app_module._designed_voice_code(entry_a)
    code_b = app_module._designed_voice_code(entry_b)

    # Stable across repeated calls.
    assert code_a1 == code_a2 == "designed_11111111-aaaa"
    # Unique per record id.
    assert code_a1 != code_b
    assert code_b == "designed_22222222-bbbb"
    # Prefixed designed_ and no custom_ collision.
    assert code_a1.startswith(app_module.DESIGNED_VOICE_CODE_PREFIX)
    assert code_a1.startswith("designed_")
    assert not code_a1.startswith(app_module.CUSTOM_CODE_PREFIX)
    assert not code_a1.startswith("custom_")


# ===========================================================================
# PBT (fix-checking): Property 1 - full lifecycle holds for random records
# ===========================================================================
_LANGUAGES = ["English", "British English", "Brazilian Portuguese", "Spanish", "Auto"]
_GENDERS = ["Female", "Male"]


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    duration_seconds=st.floats(min_value=float(MIN_SECONDS), max_value=12.0),
    name=st.text(
        alphabet=st.characters(min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=24,
    ).map(lambda s: s.strip()).filter(lambda s: bool(s)),
    language=st.sampled_from(_LANGUAGES),
    gender=st.sampled_from(_GENDERS),
)
def test_property1_designed_voice_lifecycle_completeness(
    duration_seconds, name, language, gender, isolated_voice_store, client
):
    """For any valid designed-voice record the full lifecycle holds.

    FOR ALL X WHERE isBugCondition(X):
      - voiceCodeOf(X) IS NOT NULL and deterministic (``designed_<id>``)
      - approve(X) -> approval_status == "approved"
      - X appears in GET /api/voices custom_voices AND GET /api/custom-voices
      - sample bytes == stored WAV bytes (exact)

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    record, raw_entry = _seed_designed_voice(
        duration_seconds, name=name, language=language, gender=gender
    )
    record_id = record["id"]

    # 2.1 - non-null deterministic voice_code.
    assert record.get("voice_code") == f"designed_{record_id}"
    voice_code = record["voice_code"]

    # Before approval the voice is NOT listed (usable-voice surface).
    assert not _listed_in_custom_voices(client, record_id, voice_code)
    assert not _listed_in_voices(client, record_id, voice_code)

    # 2.2 - approval flips status to "approved".
    approve = client.post(f"/api/custom-voices/{record_id}/approve")
    assert approve.status_code == 200, approve.get_data(as_text=True)
    approved_voice = approve.get_json()["voice"]
    assert approved_voice["voice_design"]["approval_status"] == "approved"
    assert approved_voice["voice_code"] == voice_code

    # 2.3 - appears in both listings exposing voice_code.
    assert _listed_in_custom_voices(client, record_id, voice_code)
    assert _listed_in_voices(client, record_id, voice_code)

    # 2.4 - sample serves the exact stored bytes.
    stored_bytes = (isolated_voice_store["prompt_dir"] / raw_entry["file_name"]).read_bytes()
    sample = client.get(f"/api/custom-voices/{record_id}/sample")
    assert sample.status_code == 200
    assert sample.data == stored_bytes


# ===========================================================================
# Full lifecycle integration test (req 2.1, 2.2, 2.3, 2.4)
# ===========================================================================
def test_full_lifecycle_integration(isolated_voice_store, client):
    """save -> result shape + voice_code -> approve -> both listings -> sample.

    NOTE on "poll task to completion": invoking the real async save worker and
    the ``GET /api/qwen3/voice-design/tasks/<task_id>`` polling loop requires the
    qwen3 voice-design engine, which is unavailable in CI. ``_save_voice_design_payload``
    is the helper the save worker calls and returns the SAME completed,
    record-shaped ``result`` the worker stores as the task result, so driving it
    directly faithfully exercises the save-to-completion step. We assert the
    result shape (``id`` + ``voice_design.approval_status``) and ``voice_code``,
    matching what a completed task poll would yield.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    stored_bytes_input = _synth_wav_bytes(6.0)
    payload = {
        "name": "Lifecycle Voice",
        "text": "A reference clip exercising the full designed-voice lifecycle.",
        "gender": "Male",
        "language": "English",
        "engine": "qwen3_voice_design",
        "audio_base64": base64.b64encode(stored_bytes_input).decode("ascii"),
        "cleanup_applied": True,
    }

    # save -> completed record-shaped result.
    result = app_module._save_voice_design_payload(payload)
    assert "id" in result
    assert result["voice_design"]["approval_status"] == "pending"
    assert result["voice_code"] == f"designed_{result['id']}"
    record_id = result["id"]
    voice_code = result["voice_code"]

    # approve.
    approve = client.post(f"/api/custom-voices/{record_id}/approve")
    assert approve.status_code == 200
    assert approve.get_json()["voice"]["voice_design"]["approval_status"] == "approved"

    # appears in both listings exposing voice_code.
    assert _listed_in_custom_voices(client, record_id, voice_code)
    assert _listed_in_voices(client, record_id, voice_code)

    # sample serves storedWavBytes(X) byte-for-byte.
    entries = app_module._load_chatterbox_voice_entries()
    entry = app_module._resolve_chatterbox_voice(record_id, entries)
    stored_bytes = (isolated_voice_store["prompt_dir"] / entry["file_name"]).read_bytes()
    sample = client.get(f"/api/custom-voices/{record_id}/sample")
    assert sample.status_code == 200
    assert sample.data == stored_bytes


# ===========================================================================
# Generation acceptance test (req 2.5)
# ===========================================================================
def test_generation_accepts_approved_voice_code(isolated_voice_store, client):
    """An approved ``voice_code`` resolves to the stored ``file_name`` in generation.

    Requirement 2.5 says referencing the code in a chapter/preview generation run
    must not fail on an "unknown voice". Since the actual TTS engine may be
    unavailable, we assert at the resolution layer that the preview/generate path
    uses: ``_resolve_designed_voice_prompt_file(voice_code)`` returns the entry's
    ``file_name`` (the ``audio_prompt_path`` forwarded to prompt engines). Unknown
    codes still return ``None`` (fall through -> fail later as an unknown voice,
    preserving prior behavior).

    **Validates: Requirements 2.5**
    """
    record, raw_entry = _seed_designed_voice(7.0, name="Generation Voice")
    record_id = record["id"]
    voice_code = record["voice_code"]

    # Approve so the voice is a usable reference.
    assert client.post(f"/api/custom-voices/{record_id}/approve").status_code == 200

    # Resolution layer: the designed code maps to the stored prompt filename.
    resolved = app_module._resolve_designed_voice_prompt_file(voice_code)
    assert resolved == raw_entry["file_name"]

    # Robustness: an unknown designed code resolves to None (not an exception),
    # so generation falls through and fails later as an unknown voice as before.
    assert app_module._resolve_designed_voice_prompt_file("designed_unknown-00000000") is None
    # A non-designed voice string is untouched by the resolver.
    assert app_module._resolve_designed_voice_prompt_file("af_heart") is None


# ===========================================================================
# Restart persistence test (req 2.6)
# ===========================================================================
def test_restart_persistence_from_disk(isolated_voice_store, client):
    """Approval survives a simulated restart (reload registry from disk).

    Approve a voice, then reload the registry via ``_load_chatterbox_voice_entries``
    (no in-memory state carried over) and assert the persisted ``approval_status``
    alone keeps the voice listed and previewable.

    **Validates: Requirements 2.6**
    """
    record, _ = _seed_designed_voice(6.5, name="Persistent Voice")
    record_id = record["id"]
    voice_code = record["voice_code"]

    assert client.post(f"/api/custom-voices/{record_id}/approve").status_code == 200

    # Simulate a restart: nothing in memory, read approval_status from disk.
    reloaded = app_module._load_chatterbox_voice_entries()
    entry = app_module._resolve_chatterbox_voice(record_id, reloaded)
    assert entry is not None
    assert entry["voice_design"]["approval_status"] == "approved"

    # Promotion is driven solely by the persisted approval_status, so the voice
    # is still listed and previewable after the "restart".
    assert _listed_in_custom_voices(client, record_id, voice_code)
    assert _listed_in_voices(client, record_id, voice_code)
    assert client.get(f"/api/custom-voices/{record_id}/sample").status_code == 200


# ===========================================================================
# Pending visibility test (req 2.4, 2.7)
# ===========================================================================
def test_pending_visibility_previewable_inspectable_but_unlisted(
    isolated_voice_store, client
):
    """A pending record is previewable + inspectable but absent from listings.

    - Sample route serves the stored WAV for a PENDING record (reviewer preview).
    - Detail route returns the pending record with its ``approval_status`` and is
      distinguishable from an unknown-id 404 (``reason: "unknown_id"``).
    - The pending record is NOT in GET /api/custom-voices nor GET /api/voices
      custom_voices until approved.

    **Validates: Requirements 2.4, 2.7**
    """
    record, raw_entry = _seed_designed_voice(5.5, name="Pending Voice")
    record_id = record["id"]
    voice_code = record["voice_code"]

    assert record["voice_design"]["approval_status"] == "pending"

    # Previewable while pending.
    stored_bytes = (isolated_voice_store["prompt_dir"] / raw_entry["file_name"]).read_bytes()
    sample = client.get(f"/api/custom-voices/{record_id}/sample")
    assert sample.status_code == 200
    assert sample.data == stored_bytes

    # Inspectable while pending, and distinguishable from unknown-id.
    detail = client.get(f"/api/custom-voices/{record_id}")
    assert detail.status_code == 200
    voice = detail.get_json()["voice"]
    assert voice["voice_code"] == voice_code
    assert voice["voice_design"]["approval_status"] == "pending"

    unknown = client.get("/api/custom-voices/does-not-exist-00000000")
    assert unknown.status_code == 404
    assert unknown.get_json().get("reason") == "unknown_id"

    # Absent from the usable-voice listings until approved.
    assert not _listed_in_custom_voices(client, record_id, voice_code)
    assert not _listed_in_voices(client, record_id, voice_code)
