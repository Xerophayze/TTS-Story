"""Bug-condition exploration tests for the designed-voice approval lifecycle.

Spec: .kiro/specs/designed-voice-approval-lifecycle/

These tests encode the EXPECTED (post-fix) lifecycle behavior for a saved
``qwen3_voice_design`` record. On the UNFIXED code they are EXPECTED TO FAIL:
each failure is a counterexample proving one of the lifecycle gaps described in
the bugfix document (no surfaced ``voice_code``; no single-record approval route;
never promoted into the voice listings; no sample route; indistinguishable 404).

The bug is deterministic for any saved designed-voice record, so the property is
scoped to a concrete seeded record while the WAV duration is parameterized over a
range of values >= ``MIN_CHATTERBOX_PROMPT_SECONDS`` to show the gap is
independent of clip content.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

import app as app_module


SAMPLE_RATE = 24_000  # 24 kHz mono, matching the designed-voice WAV contract
MIN_SECONDS = app_module.MIN_CHATTERBOX_PROMPT_SECONDS  # 5.0


# ---------------------------------------------------------------------------
# WAV synthesis helpers
# ---------------------------------------------------------------------------
def _synth_wav_bytes(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Synthesize a mono PCM16 WAV of the requested duration as raw bytes."""
    n_samples = max(1, int(round(duration_seconds * sample_rate)))
    t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)
    # A gentle 220 Hz tone at moderate amplitude - deterministic, non-silent audio.
    signal = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _synth_wav_base64(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> str:
    import base64

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
    # The transcript sidecar lives under VOICE_PROMPT_DIR/transcripts.json, which
    # the monkeypatch above already isolates. Reset the module-level cache so the
    # temp transcripts file is read fresh and nothing leaks between tests.
    monkeypatch.setattr(app_module, "_voice_prompt_transcript_cache", None, raising=False)
    return {"prompt_dir": prompt_dir, "registry": registry}


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _seed_designed_voice(duration_seconds: float, name: str = "Explorer Voice"):
    """Seed a designed-voice record via the real save helper and return (record, raw_entry)."""
    payload = {
        "name": name,
        "text": (
            "This is a synthesized reference clip used to seed a designed voice "
            "record for the approval lifecycle exploration tests."
        ),
        "gender": "Female",
        "language": "English",
        "description": "Seeded by bug-condition exploration tests.",
        "engine": "qwen3_voice_design",
        "audio_base64": _synth_wav_base64(duration_seconds),
        # Skip the cleanup pass so the stored bytes match the synthesized input.
        "cleanup_applied": True,
    }
    record = app_module._save_voice_design_payload(payload)
    entries = app_module._load_chatterbox_voice_entries()
    raw_entry = app_module._resolve_chatterbox_voice(record["id"], entries)
    return record, raw_entry


# ---------------------------------------------------------------------------
# Case 1 - No code surfaced (Requirement 1.1 -> 2.1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration_seconds", [5.0, 5.5, 7.3, 12.0])
def test_case1_voice_code_surfaced_on_save_and_serialization(
    duration_seconds, isolated_voice_store, client
):
    """A saved designed-voice record SHALL surface a non-null ``voice_code``.

    FAILS on unfixed code: neither the save ``result`` nor the re-serialized
    record carries a ``voice_code`` key.

    **Validates: Requirements 1.1**
    """
    record, raw_entry = _seed_designed_voice(duration_seconds)

    # Sanity: the record was persisted as a pending designed voice.
    assert record["voice_design"]["approval_status"] == "pending"
    assert record["voice_design"]["engine"] == "qwen3_voice_design"

    serialized = app_module._serialize_chatterbox_voice(raw_entry)

    assert "voice_code" in record, "save result is missing voice_code"
    assert record["voice_code"], "save result voice_code is empty/None"
    assert "voice_code" in serialized, "serialized record is missing voice_code"
    assert serialized["voice_code"], "serialized voice_code is empty/None"


# ---------------------------------------------------------------------------
# Case 2 - No approval route (Requirement 1.2 -> 2.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration_seconds", [5.0, 6.4, 9.1])
def test_case2_single_record_approve_route_exists(
    duration_seconds, isolated_voice_store, client
):
    """A single-record approval route SHALL exist and flip approval_status.

    FAILS on unfixed code: ``POST /api/custom-voices/<id>/approve`` returns 404.

    **Validates: Requirements 1.2**
    """
    record, _ = _seed_designed_voice(duration_seconds)
    record_id = record["id"]

    resp = client.post(f"/api/custom-voices/{record_id}/approve")
    assert resp.status_code != 404, (
        "approve route is missing (404) - no single-record approval mechanism"
    )


@pytest.mark.parametrize("duration_seconds", [5.0, 6.4])
def test_case2_single_record_reject_route_exists(
    duration_seconds, isolated_voice_store, client
):
    """A single-record reject route SHALL exist.

    FAILS on unfixed code: ``POST /api/custom-voices/<id>/reject`` returns 404.

    **Validates: Requirements 1.2**
    """
    record, _ = _seed_designed_voice(duration_seconds)
    record_id = record["id"]

    resp = client.post(f"/api/custom-voices/{record_id}/reject")
    assert resp.status_code != 404, (
        "reject route is missing (404) - no single-record reject mechanism"
    )


# ---------------------------------------------------------------------------
# Case 3 - Not promoted (Requirement 1.3 -> 2.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration_seconds", [5.0, 7.0, 11.5])
def test_case3_approved_voice_promoted_into_listings(
    duration_seconds, isolated_voice_store, client
):
    """An approved designed voice SHALL appear in both voice listings.

    FAILS on unfixed code: there is no approval route to promote the record, and
    ``VoiceManager`` never reads the chatterbox registry, so ``GET /api/voices``
    custom_voices groups are empty and ``GET /api/custom-voices`` returns ``[]``
    for the designed record.

    **Validates: Requirements 1.3**
    """
    record, _ = _seed_designed_voice(duration_seconds)
    record_id = record["id"]
    voice_code = record.get("voice_code")

    # Approve the record (the route must exist for promotion to be possible).
    approve_resp = client.post(f"/api/custom-voices/{record_id}/approve")
    assert approve_resp.status_code != 404, (
        "cannot promote - approve route missing (404)"
    )

    # GET /api/custom-voices must now list the designed voice.
    coll = client.get("/api/custom-voices")
    assert coll.status_code == 200
    coll_voices = coll.get_json().get("voices", [])
    listed_in_collection = any(
        (v.get("id") == record_id)
        or (voice_code and v.get("voice_code") == voice_code)
        or (voice_code and v.get("code") == voice_code)
        for v in coll_voices
    )
    assert listed_in_collection, (
        "approved designed voice is absent from GET /api/custom-voices"
    )

    # GET /api/voices must surface it under some language group's custom_voices.
    voices_resp = client.get("/api/voices")
    assert voices_resp.status_code == 200
    voices_payload = voices_resp.get_json().get("voices", {})
    all_custom = []
    if isinstance(voices_payload, dict):
        for group in voices_payload.values():
            if isinstance(group, dict):
                all_custom.extend(group.get("custom_voices", []) or [])
            elif isinstance(group, list):
                for lang_group in group:
                    if isinstance(lang_group, dict):
                        all_custom.extend(lang_group.get("custom_voices", []) or [])
    listed_in_voices = any(
        (v.get("id") == record_id)
        or (voice_code and v.get("code") == voice_code)
        or (voice_code and v.get("voice_code") == voice_code)
        for v in all_custom
    )
    assert listed_in_voices, (
        "approved designed voice is absent from GET /api/voices custom_voices groups"
    )


# ---------------------------------------------------------------------------
# Case 4 - No sample route (Requirement 1.4 -> 2.4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration_seconds", [5.0, 6.0, 8.8])
def test_case4_sample_route_serves_stored_wav(
    duration_seconds, isolated_voice_store, client
):
    """A sample route SHALL serve the exact stored WAV for the record.

    FAILS on unfixed code: ``GET /api/custom-voices/<id>/sample`` returns 404 even
    though ``missing_file`` is false (the WAV exists on disk).

    **Validates: Requirements 1.4**
    """
    record, raw_entry = _seed_designed_voice(duration_seconds)
    record_id = record["id"]

    # Precondition: the stored file exists (missing_file is false).
    assert record["missing_file"] is False
    stored_path = isolated_voice_store["prompt_dir"] / raw_entry["file_name"]
    assert stored_path.is_file()
    stored_bytes = stored_path.read_bytes()

    resp = client.get(f"/api/custom-voices/{record_id}/sample")
    assert resp.status_code == 200, (
        f"sample route missing/failed (status {resp.status_code}) though the WAV exists"
    )
    assert resp.data == stored_bytes, "sample route did not serve the exact stored WAV bytes"


# ---------------------------------------------------------------------------
# Case 5 - Edge: distinguishable 404 (Requirement 1.5 -> 2.7)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration_seconds", [5.0, 7.7])
def test_case5_pending_record_detail_distinguishable_from_unknown(
    duration_seconds, isolated_voice_store, client
):
    """Looking up a pending record SHALL be distinguishable from an unknown id.

    Expected (post-fix): ``GET /api/custom-voices/<pending_id>`` either returns the
    record (with ``approval_status``) or returns a 404 whose body is distinguishable
    from the unknown-id 404.

    FAILS on unfixed code: a pending record and a wholly unknown id both return the
    SAME generic 404 ("Custom voice not found."), so the two cases are
    indistinguishable.

    **Validates: Requirements 1.5**
    """
    record, _ = _seed_designed_voice(duration_seconds)
    record_id = record["id"]

    pending_resp = client.get(f"/api/custom-voices/{record_id}")
    unknown_resp = client.get("/api/custom-voices/does-not-exist-00000000")

    if pending_resp.status_code == 200:
        body = pending_resp.get_json()
        voice = body.get("voice", body)
        assert "approval_status" in voice or (
            isinstance(voice.get("voice_design"), dict)
            and "approval_status" in voice["voice_design"]
        ), "pending record returned 200 but without an approval_status to inspect"
        return

    # Both 404 - they MUST be distinguishable.
    assert pending_resp.status_code == 404
    assert unknown_resp.status_code == 404
    assert pending_resp.get_json() != unknown_resp.get_json(), (
        "pending-record 404 is indistinguishable from unknown-id 404"
    )
