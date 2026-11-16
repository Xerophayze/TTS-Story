"""Preservation property tests for the designed-voice approval lifecycle bugfix.

Spec: .kiro/specs/designed-voice-approval-lifecycle/

**Property 2: Preservation - Non-Lifecycle Behavior**

These tests capture the behavior that MUST remain unchanged by the fix, for every
input where the bug condition does NOT hold (``isBugCondition`` returns false).

Methodology: observation-first. Each test exercises the UNFIXED code for a
preserved surface, records the actual output, and asserts it. On the UNFIXED code
these tests are EXPECTED TO PASS (they establish the baseline to preserve); after
the fix they must still pass (no regression).

Preserved surfaces (bugfix.md "Unchanged Behavior"):
  3.1 Sub-5s clip rejection with the exact message
      ``"Clip is only {d:.2f}s. Voice prompts require at least 5 seconds."``
  3.2 Save ``result`` record shape exposes ``id`` and
      ``voice_design.approval_status`` (keys ``TTSStoryClient.voice_design_save()`` reads).
  3.3 The non-save PREVIEW task result still contains ``audio_base64``.
  3.4 Candidate-group approval via ``POST /api/qwen3/voice-design/candidates/approve``
      approves the selected candidate and rejects/archives the rest.
  3.5 Existing Kokoro custom blends are listed/detailed identically for records that
      are not designed-voice records.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""
from __future__ import annotations

import base64
import io
import uuid

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import app as app_module
import src.custom_voice_store as custom_voice_store


SAMPLE_RATE = 24_000  # 24 kHz mono, matching the designed-voice WAV contract
MIN_SECONDS = app_module.MIN_CHATTERBOX_PROMPT_SECONDS  # 5.0


# ---------------------------------------------------------------------------
# WAV synthesis helpers
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
# Fixtures: isolate every persisted surface to a temp path per test
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_voice_store(tmp_path, monkeypatch):
    """Point the chatterbox registry, voice-prompt dir, and Kokoro store at temp paths."""
    prompt_dir = tmp_path / "voice_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "chatterbox_voices.json"
    kokoro_store = tmp_path / "custom_voices.json"

    monkeypatch.setattr(app_module, "VOICE_PROMPT_DIR", prompt_dir)
    monkeypatch.setattr(app_module, "CHATTERBOX_VOICE_REGISTRY", registry)
    # Isolate the separate Kokoro custom-voice store (src/custom_voice_store.py).
    monkeypatch.setattr(custom_voice_store, "CUSTOM_VOICE_PATH", kokoro_store)
    # Reset the transcript sidecar cache so the temp file is read fresh per test.
    monkeypatch.setattr(app_module, "_voice_prompt_transcript_cache", None, raising=False)
    return {
        "prompt_dir": prompt_dir,
        "registry": registry,
        "kokoro_store": kokoro_store,
    }


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _save_payload(duration_seconds: float, name: str = "Preserve Voice") -> dict:
    """Build a valid save payload for ``_save_voice_design_payload``."""
    return {
        "name": name,
        "text": (
            "This is a synthesized reference clip used to exercise the preserved "
            "save behavior of the designed-voice lifecycle."
        ),
        "gender": "Female",
        "language": "English",
        "engine": "qwen3_voice_design",
        "audio_base64": _synth_wav_base64(duration_seconds),
        # Skip the cleanup pass so the stored bytes match the synthesized input.
        "cleanup_applied": True,
    }


# ===========================================================================
# 3.1 - Sub-5s rejection wording (PBT over random durations in (0, 5))
# ===========================================================================
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(duration_seconds=st.floats(min_value=0.05, max_value=4.99))
def test_sub5s_rejection_message_verbatim(duration_seconds, isolated_voice_store):
    """A sub-5s clip is rejected with the exact, duration-formatted message.

    Observation-first: the message is built as
    ``f"Clip is only {d:.2f}s. Voice prompts require at least 5 seconds."``
    using the MEASURED duration of the stored clip. We synthesize the clip at
    ``duration_seconds`` and assert the raised message matches that pattern with
    the measured duration (which is what the unfixed code reports).

    **Validates: Requirements 3.1**
    """
    payload = _save_payload(duration_seconds, name=f"Short {duration_seconds:.3f}")

    with pytest.raises(ValueError) as excinfo:
        app_module._save_voice_design_payload(payload)

    message = str(excinfo.value)
    # The measured duration is derived from the stored file; recover it from the
    # message and confirm the message is EXACTLY the documented template for it.
    assert message.startswith("Clip is only "), message
    assert message.endswith("s. Voice prompts require at least 5 seconds."), message

    measured = float(message[len("Clip is only ") : message.index("s. Voice prompts")])
    expected = (
        f"Clip is only {measured:.2f}s. Voice prompts require at least 5 seconds."
    )
    assert message == expected
    # The measured duration must be sub-5s (the rejection branch condition).
    assert measured < MIN_SECONDS


def test_sub5s_rejection_message_exact_030(isolated_voice_store):
    """The documented 0.30s counterexample yields the verbatim message.

    This is the concrete example from the bugfix doc / design test plan.

    **Validates: Requirements 3.1**
    """
    payload = _save_payload(0.30, name="ThirtyHundredths")
    with pytest.raises(ValueError) as excinfo:
        app_module._save_voice_design_payload(payload)
    assert str(excinfo.value) == (
        "Clip is only 0.30s. Voice prompts require at least 5 seconds."
    )


# ===========================================================================
# 3.2 - Save result record shape (observe actual output on unfixed code)
# ===========================================================================
@pytest.mark.parametrize("duration_seconds", [5.0, 5.5, 8.2, 12.0])
def test_save_result_shape_exposes_id_and_approval_status(
    duration_seconds, isolated_voice_store
):
    """The completed save ``result`` exposes ``id`` and ``voice_design.approval_status``.

    These are the keys ``TTSStoryClient.voice_design_save()`` reads. Observed on the
    unfixed code: the serialized record carries both. The fix must keep them.

    **Validates: Requirements 3.2**
    """
    result = app_module._save_voice_design_payload(_save_payload(duration_seconds))

    assert "id" in result and result["id"], "save result missing 'id'"
    assert isinstance(result.get("voice_design"), dict), "missing voice_design dict"
    assert result["voice_design"].get("approval_status") == "pending", (
        "save result voice_design.approval_status is not the baseline 'pending'"
    )
    # Baseline record keys that downstream readers rely on (observed, not exhaustive).
    for key in ("id", "name", "file_name", "voice_design"):
        assert key in result, f"save result missing baseline key {key!r}"


# ===========================================================================
# 3.3 - PREVIEW task contract: result still contains audio_base64
# ===========================================================================
def test_preview_task_result_contains_audio_base64(isolated_voice_store, client):
    """The non-save PREVIEW task delivers ``audio_base64`` in its completed result.

    The audio generation itself needs the qwen3 isolated engine, which is not
    available in the test environment. We therefore exercise the stable CONTRACT
    DELIVERY path: a completed ``qwen3_voice_design_preview`` job whose result dict
    carries ``audio_base64`` is surfaced verbatim by the real task-status route
    ``GET /api/qwen3/voice-design/tasks/<task_id>`` (app.py ~14465). This is exactly
    the shape ``process_qwen3_voice_design_preview_task`` (~6630) stores via
    ``_generate_voice_design_preview`` (~7925), which returns ``{"audio_base64": ...}``.

    **Validates: Requirements 3.3**
    """
    task_id = f"preview-{uuid.uuid4()}"
    encoded = base64.b64encode(_synth_wav_bytes(10.5)).decode("ascii")
    result_payload = {
        "audio_base64": encoded,
        "mime_type": "audio/wav",
        "engine": "qwen3_voice_design",
        "task_id": task_id,
    }
    with app_module.queue_lock:
        app_module.jobs[task_id] = {
            "job_id": task_id,
            "job_type": "qwen3_voice_design_preview",
            "status": "completed",
            "progress": 100,
            "result": result_payload,
        }
    try:
        resp = client.get(f"/api/qwen3/voice-design/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["status"] == "completed"
        assert "result" in body, "completed preview task did not surface a result"
        assert "audio_base64" in body["result"], (
            "PREVIEW task result is missing audio_base64 (contract broken)"
        )
        assert body["result"]["audio_base64"] == encoded
    finally:
        with app_module.queue_lock:
            app_module.jobs.pop(task_id, None)


def test_preview_generator_result_shape_includes_audio_base64():
    """The preview generator's documented result shape keys include ``audio_base64``.

    A lightweight structural guard: the return dict assembled by
    ``_generate_voice_design_preview`` (~7925) includes an ``audio_base64`` field.
    Verified by source inspection to avoid requiring the heavy engine here; the
    route-level test above exercises delivery of that field.

    **Validates: Requirements 3.3**
    """
    import inspect

    source = inspect.getsource(app_module._generate_voice_design_preview)
    assert '"audio_base64"' in source, (
        "preview generator no longer returns an audio_base64 field"
    )


# ===========================================================================
# 3.4 - Candidate-group approval via candidates/approve (real route)
# ===========================================================================
def _seed_candidate_group(group_id: str, candidate_ids):
    """Seed a chatterbox candidate group directly into the isolated registry."""
    entries = []
    for cid in candidate_ids:
        entries.append(
            {
                "id": cid,
                "name": f"Candidate {cid[:6]}",
                "file_name": f"{cid}.wav",
                "voice_design": {
                    "engine": "qwen3_voice_design",
                    "candidate_group_id": group_id,
                    "approval_status": "pending",
                },
            }
        )
    app_module._save_chatterbox_voice_entries(entries)
    return entries


def test_candidate_group_approval_selects_one_rejects_rest(isolated_voice_store, client):
    """Candidate-group approval approves the selected candidate and rejects the rest.

    Exercises the real ``POST /api/qwen3/voice-design/candidates/approve`` route
    (``qwen3_voice_design_approve_candidate`` ~14547) end-to-end against a seeded
    candidate group. Observed behavior on unfixed code: selected -> approved
    (archived False); others -> rejected (archived True).

    **Validates: Requirements 3.4**
    """
    group_id = f"group-{uuid.uuid4()}"
    candidate_ids = [str(uuid.uuid4()) for _ in range(4)]
    _seed_candidate_group(group_id, candidate_ids)
    selected_id = candidate_ids[1]

    resp = client.post(
        "/api/qwen3/voice-design/candidates/approve",
        json={"candidate_group_id": group_id, "selected_id": selected_id},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["success"] is True
    assert body["voice"]["id"] == selected_id

    entries = app_module._load_chatterbox_voice_entries()
    by_id = {e["id"]: e for e in entries}
    for cid in candidate_ids:
        entry = by_id[cid]
        status = entry["voice_design"]["approval_status"]
        archived = entry.get("archived")
        if cid == selected_id:
            assert status == "approved"
            assert archived is False
        else:
            assert status == "rejected"
            assert archived is True


def test_candidate_group_approval_requires_ids(isolated_voice_store, client):
    """Candidate-group approval rejects missing ids with a 400 (baseline guard).

    **Validates: Requirements 3.4**
    """
    resp = client.post("/api/qwen3/voice-design/candidates/approve", json={})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


# ===========================================================================
# 3.5 - Kokoro custom blends listed/detailed identically (PBT)
# ===========================================================================
_LANG_CODES = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]


def _stored_kokoro_blend(name, lang_code, components, notes):
    """Build a raw Kokoro custom-voice record (stored/internal format)."""
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "lang_code": lang_code,
        "components": components,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-02T00:00:00",
        "notes": notes,
    }


_component_strategy = st.fixed_dictionaries(
    {
        "voice": st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=2, max_size=10
        ),
        "weight": st.floats(min_value=0.1, max_value=5.0),
    }
)

_blend_strategy = st.builds(
    _stored_kokoro_blend,
    name=st.text(alphabet="ABCDEFGHIJ abcdefghij", min_size=1, max_size=20).map(
        lambda s: s.strip() or "Blend"
    ),
    lang_code=st.sampled_from(_LANG_CODES),
    components=st.lists(_component_strategy, min_size=1, max_size=4),
    notes=st.one_of(st.none(), st.text(max_size=30)),
)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(blends=st.lists(_blend_strategy, min_size=1, max_size=5))
def test_kokoro_blends_listing_identical_to_direct_serialization(
    blends, isolated_voice_store, client
):
    """``GET /api/custom-voices`` returns Kokoro blends identically to direct serialization.

    PBT over random non-designed (Kokoro) blend records. We seed the isolated store
    directly, then assert the collection route's output is byte-for-byte identical to
    the recorded behavior produced by the same serialization helpers
    (``list_custom_voice_entries`` -> ``_to_public_custom_voice``) the route uses.

    **Validates: Requirements 3.5**
    """
    # Reset the isolated store and seed the generated blends.
    custom_voice_store._save_payload(list(blends))

    # Recorded (expected) behavior: the exact serialization the GET branch applies.
    entries = app_module.list_custom_voice_entries()
    expected_voices = [app_module._to_public_custom_voice(e) for e in entries]

    resp = client.get("/api/custom-voices")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["voices"] == expected_voices, (
        "custom-voices listing diverged from the recorded Kokoro serialization"
    )
    # Every seeded blend appears, exposing its custom_ code (non-designed surface).
    listed_ids = {v.get("id") for v in body["voices"]}
    for blend in blends:
        assert blend["id"] in listed_ids


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(blend=_blend_strategy)
def test_kokoro_blend_detail_identical_to_direct_serialization(
    blend, isolated_voice_store, client
):
    """``GET /api/custom-voices/<voice_id>`` returns a Kokoro blend identically.

    PBT over a random non-designed blend. The detail route's body must match the
    recorded serialization of the raw record for both the bare id and the
    ``custom_<id>`` code form.

    **Validates: Requirements 3.5**
    """
    custom_voice_store._save_payload([blend])

    raw = app_module._get_raw_custom_voice(blend["id"])
    expected = app_module._to_public_custom_voice(raw)

    # Lookup by bare id.
    resp_id = client.get(f"/api/custom-voices/{blend['id']}")
    assert resp_id.status_code == 200
    body_id = resp_id.get_json()
    assert body_id["success"] is True
    assert body_id["voice"] == expected

    # Lookup by public code (custom_<id>).
    code = f"{app_module.CUSTOM_CODE_PREFIX}{blend['id']}"
    resp_code = client.get(f"/api/custom-voices/{code}")
    assert resp_code.status_code == 200
    assert resp_code.get_json()["voice"] == expected


def test_kokoro_detail_unknown_id_is_404(isolated_voice_store, client):
    """An unknown custom-voice id returns the baseline 404 (non-designed surface).

    **Validates: Requirements 3.5**
    """
    resp = client.get("/api/custom-voices/no-such-kokoro-id")
    assert resp.status_code == 404
    assert resp.get_json()["success"] is False
