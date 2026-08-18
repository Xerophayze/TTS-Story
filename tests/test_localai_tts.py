from __future__ import annotations

import io
import hashlib
import json
import wave
from pathlib import Path

from src.engines.localai_tts_engine import LocalAITTSEngine
from src.localai_tts_client import discover_localai_tts_catalog, normalize_localai_urls
from src.localai_voice_profiles import (
    LocalAIVoiceProfileError,
    LocalAIVoiceProfileManager,
    build_tts_story_voice_reference,
)
import pytest
import app as app_module


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 120)
    return output.getvalue()


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""
        self.headers = {}
        self.content = _wav_bytes()

    def json(self):
        return self._payload


def test_localai_engine_sends_voice_profile_uri_without_api_key():
    calls = []
    engine = LocalAITTSEngine(
        base_url="http://localhost:8080/v1",
        model_id="omnivoice-cpp-hq",
        default_voice="localai://voice-profiles/profile-1",
        request_func=lambda method, url, **kwargs: calls.append((method, url, kwargs)) or FakeResponse({}),
        audio_converter=lambda payload, **kwargs: payload,
    )
    engine.generate_audio("Hello from LocalAI.")
    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == "http://localhost:8080/v1/audio/speech"
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["json"]["model"] == "omnivoice-cpp-hq"
    assert kwargs["json"]["voice"] == "localai://voice-profiles/profile-1"


def test_localai_engine_allows_model_default_voice():
    calls = []
    engine = LocalAITTSEngine(
        model_id="tts-model",
        request_func=lambda method, url, **kwargs: calls.append(kwargs) or FakeResponse({}),
        audio_converter=lambda payload, **kwargs: payload,
    )
    engine.generate_audio("Hello.")
    assert "voice" not in calls[0]["json"]


def test_localai_engine_sends_freeform_voice_and_language():
    calls = []
    engine = LocalAITTSEngine(
        model_id="custom-tts-model",
        default_voice="voice-from-model-config",
        default_language="fr-FR",
        request_func=lambda method, url, **kwargs: calls.append(kwargs) or FakeResponse({}),
        audio_converter=lambda payload, **kwargs: payload,
    )
    engine.generate_audio(
        "Bonjour depuis LocalAI.",
        voice="speaker-name-not-in-catalog",
        lang_code="French",
    )
    assert calls[0]["json"]["voice"] == "speaker-name-not-in-catalog"
    assert calls[0]["json"]["language"] == "French"


def test_localai_engine_uses_configured_default_language():
    calls = []
    engine = LocalAITTSEngine(
        model_id="custom-tts-model",
        default_language="ja",
        request_func=lambda method, url, **kwargs: calls.append(kwargs) or FakeResponse({}),
        audio_converter=lambda payload, **kwargs: payload,
    )
    engine.generate_audio("LocalAI language fallback.")
    assert calls[0]["json"]["language"] == "ja"


def test_localai_settings_offer_freeform_voice_and_language_fields():
    project_root = Path(__file__).resolve().parents[1]
    template = (project_root / "templates" / "index.html").read_text(encoding="utf-8")
    main_js = (project_root / "static" / "js" / "main.js").read_text(encoding="utf-8")
    settings_js = (project_root / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    assert 'id="localai-tts-default-voice" list="localai-tts-voice-options"' in template
    assert 'id="localai-tts-default-language"' in template
    assert 'class="localai-voice-input"' in main_js
    assert 'class="localai-language-input"' in main_js
    assert "localai_tts_default_language" in settings_js
    library_js = (project_root / "static" / "js" / "library.js").read_text(encoding="utf-8")
    assert "Type a custom voice / speaker ID" in library_js
    assert "resolveLocalAIFreeformSelection" in library_js


def test_unix_installers_do_not_require_sudo_when_running_as_root():
    project_root = Path(__file__).resolve().parents[1]
    setup_script = (project_root / "setup.sh").read_text(encoding="utf-8")
    update_script = (project_root / "install-update.sh").read_text(encoding="utf-8")
    for script in (setup_script, update_script):
        assert 'if [ "$(id -u)" -eq 0 ]; then' in script
        assert "elif command -v sudo" in script
        assert "run_as_root" in script


def test_localai_engine_accepts_server_root_without_v1():
    calls = []
    engine = LocalAITTSEngine(
        base_url="http://localhost:8080",
        model_id="tts-model",
        request_func=lambda method, url, **kwargs: calls.append(url) or FakeResponse({}),
        audio_converter=lambda payload, **kwargs: payload,
    )
    engine.generate_audio("Hello.")
    assert calls == ["http://localhost:8080/v1/audio/speech"]


def test_localai_engine_resolves_tts_story_reference_before_synthesis():
    calls = []
    resolved = []
    engine = LocalAITTSEngine(
        model_id="clone-model",
        default_voice=build_tts_story_voice_reference("narrator.wav"),
        voice_resolver=lambda voice: resolved.append(voice) or "localai://voice-profiles/new-profile",
        request_func=lambda method, url, **kwargs: calls.append(kwargs) or FakeResponse({}),
        audio_converter=lambda payload, **kwargs: payload,
    )
    engine.generate_audio("Use the synchronized sample.")
    assert resolved == [build_tts_story_voice_reference("narrator.wav")]
    assert calls[0]["json"]["voice"] == "localai://voice-profiles/new-profile"


def test_localai_catalog_filters_tts_models_and_maps_voice_profiles():
    payloads = {
        "http://localhost:8080/.well-known/localai.json": {
            "version": "v4.8.2", "capabilities": {"voice_profiles": True}
        },
        "http://localhost:8080/v1/models/capabilities": {"data": [
            {"id": "omnivoice", "capabilities": ["tts"]},
            {"id": "chat-model", "capabilities": ["chat"]},
        ]},
        "http://localhost:8080/api/models/config-json/omnivoice": {
            "tts": {"voice_cloning": True}
        },
        "http://localhost:8080/api/voice-profiles": {"profiles": [{
            "id": "profile-1", "name": "Narrator", "language": "EN-US"
        }]},
    }

    def request(method, url, **kwargs):
        return FakeResponse(payloads[url])

    catalog = discover_localai_tts_catalog("http://localhost:8080", request_func=request)
    assert [model["model_id"] for model in catalog["models"]] == ["omnivoice"]
    assert catalog["models"][0]["voice_cloning"] is True
    assert catalog["voice_profiles_supported"] is True
    assert catalog["voices"][0]["voice_id"] == "localai://voice-profiles/profile-1"
    assert catalog["voices"][0]["display_name"] == "Narrator"


def test_localai_url_normalization_accepts_root_v1_and_speech_endpoint():
    expected = ("http://localhost:8080", "http://localhost:8080/v1")
    assert normalize_localai_urls("http://localhost:8080") == expected
    assert normalize_localai_urls("http://localhost:8080/v1") == expected
    assert normalize_localai_urls("http://localhost:8080/v1/audio/speech") == expected


def _write_voice_fixture(tmp_path, *, transcript="Exact words in the reference sample."):
    prompts = tmp_path / "voice_prompts"
    prompts.mkdir()
    audio_path = prompts / "narrator.wav"
    audio_path.write_bytes(_wav_bytes())
    stat = audio_path.stat()
    transcript_key = hashlib.md5(
        f"{audio_path.name}:{stat.st_size}:{stat.st_mtime}".encode()
    ).hexdigest()[:16]
    (prompts / "transcripts.json").write_text(json.dumps({
        "transcripts": {transcript_key: transcript} if transcript else {}
    }), encoding="utf-8")
    registry = tmp_path / "chatterbox_voices.json"
    registry.write_text(json.dumps([{
        "id": "voice-1", "name": "Narrator", "file_name": audio_path.name,
        "language": "en-US",
    }]), encoding="utf-8")
    return prompts, registry


def test_localai_voice_profile_uploads_transcript_once_and_reuses_profile(tmp_path):
    prompts, registry = _write_voice_fixture(tmp_path)
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return FakeResponse({"data": []})
        assert kwargs["data"]["transcript"] == "Exact words in the reference sample."
        assert kwargs["data"]["consent_confirmed"] == "true"
        assert kwargs["files"]["audio"][0] == "narrator.wav"
        return FakeResponse({
            "id": "profile-1", "voice": "localai://voice-profiles/profile-1"
        }, status=201)

    manager = LocalAIVoiceProfileManager(
        "http://localhost:8080/v1",
        consent_confirmed=True,
        voice_prompt_dir=prompts,
        registry_path=registry,
        mappings_path=tmp_path / "mappings.json",
        request_func=request,
    )
    reference = build_tts_story_voice_reference("narrator.wav")
    assert manager.resolve(reference) == "localai://voice-profiles/profile-1"
    assert manager.resolve(reference) == "localai://voice-profiles/profile-1"
    assert [call[0] for call in calls].count("POST") == 1


def test_localai_voice_profile_requires_transcript_and_consent(tmp_path):
    prompts, registry = _write_voice_fixture(tmp_path, transcript="")
    reference = build_tts_story_voice_reference("narrator.wav")
    manager = LocalAIVoiceProfileManager(
        "http://localhost:8080",
        consent_confirmed=False,
        voice_prompt_dir=prompts,
        registry_path=registry,
        mappings_path=tmp_path / "mappings.json",
    )
    with pytest.raises(LocalAIVoiceProfileError, match="rights/consent"):
        manager.resolve(reference)
    manager.consent_confirmed = True
    with pytest.raises(LocalAIVoiceProfileError, match="needs an exact transcript"):
        manager.resolve(reference)


def test_localai_catalog_includes_only_transcript_ready_tts_story_samples(monkeypatch):
    monkeypatch.setattr(app_module, "discover_localai_tts_catalog", lambda *args, **kwargs: {
        "voice_profiles_supported": True,
        "models": [{"model_id": "clone-model", "voice_cloning": True}],
        "voices": [{"voice_id": "localai://voice-profiles/server", "short_name": "localai://voice-profiles/server"}],
    })
    monkeypatch.setattr(app_module, "_auto_register_voice_prompt_files", lambda: None)
    monkeypatch.setattr(app_module, "_backfill_generated_voice_transcripts", lambda entries: 0)
    monkeypatch.setattr(app_module, "_load_chatterbox_voice_entries", lambda: [
        {"id": "ready", "file_name": "ready.wav"},
        {"id": "missing", "file_name": "missing.wav"},
    ])
    monkeypatch.setattr(app_module, "_serialize_chatterbox_voice", lambda entry: {
        "id": entry["id"],
        "file_name": entry["file_name"],
        "name": entry["id"].title(),
        "transcript": "Exact transcript." if entry["id"] == "ready" else "",
        "missing_file": False,
        "archived": False,
    })
    response = app_module.app.test_client().post('/api/localai-tts/catalog', json={
        "model": "clone-model",
        "consent_confirmed": True,
    })
    data = response.get_json()
    assert response.status_code == 200
    assert len(data["voices"]) == 2
    assert data["voices"][1]["voice_id"] == build_tts_story_voice_reference("ready.wav")
    assert all("missing.wav" not in voice.get("voice_id", "") for voice in data["voices"])


def test_voice_library_transcription_endpoint_persists_generated_text(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(_wav_bytes())
    entry = {"id": "voice-1", "file_name": audio_path.name, "name": "Sample"}
    saved = {}

    class FakeGenerator:
        def transcribe(self, path):
            assert path == audio_path
            return "The generated transcript."

    monkeypatch.setattr(app_module, "VOICE_PROMPT_DIR", tmp_path)
    monkeypatch.setattr(app_module, "_load_chatterbox_voice_entries", lambda: [entry])
    monkeypatch.setattr(app_module, "voice_transcript_generator", FakeGenerator())
    monkeypatch.setattr(
        app_module,
        "_update_voice_prompt_transcript",
        lambda path, transcript: saved.update(path=path, transcript=transcript),
    )
    monkeypatch.setattr(
        app_module,
        "_serialize_chatterbox_voice",
        lambda value: {"id": value["id"], "transcript": saved.get("transcript", "")},
    )
    response = app_module.app.test_client().post(
        "/api/chatterbox-voices/voice-1/transcribe"
    )
    data = response.get_json()
    assert response.status_code == 200
    assert data["transcript"] == "The generated transcript."
    assert saved == {"path": audio_path, "transcript": "The generated transcript."}
