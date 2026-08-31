from __future__ import annotations

import json
from pathlib import Path

import pytest

import app as app_module
from scripts import install_engine
from src.engines.audio8_tts_engine import Audio8TTSEngine
from src.engines.isolated_proxy import ENGINE_DIRS, SAMPLE_RATES


ROOT = Path(__file__).resolve().parents[1]


def test_audio8_is_an_isolated_installable_engine():
    assert install_engine.ISOLATED_ENGINES["audio8_tts"] == "audio8_tts.txt"
    assert "audio8_tts" in install_engine.TORCH_ENGINES
    assert ENGINE_DIRS["audio8_tts"] == "audio8_tts"
    assert SAMPLE_RATES["audio8_tts"] == 44100
    requirements = (ROOT / "requirements-engines" / "audio8_tts.txt").read_text(encoding="utf-8")
    assert "transformers>=4.57.0,<5" in requirements
    assert "torch>=2.5.0" in requirements


def test_audio8_catalog_reports_install_action(monkeypatch):
    real_available = app_module.isolated_engine_available
    monkeypatch.setattr(
        app_module, "isolated_engine_available",
        lambda engine: False if engine == "audio8_tts" else real_available(engine),
    )
    catalog = {entry["id"]: entry for entry in app_module._engine_setup_catalog(dict(app_module.DEFAULT_CONFIG))}
    assert catalog["audio8_tts"]["install_target"] == "audio8_tts"
    assert catalog["audio8_tts"]["settings_tab"] == "audio8-tts"
    assert catalog["audio8_tts"]["action"] == "install"


def test_audio8_text_hard_limit_is_configurable():
    engine = Audio8TTSEngine.__new__(Audio8TTSEngine)
    engine.max_input_chars = 400
    assert engine._validate_text("  A short sentence.  ") == "A short sentence."
    assert engine._validate_text("x" * 151) == "x" * 151
    with pytest.raises(ValueError, match="400"):
        engine._validate_text("x" * 401)


def test_audio8_processor_preserves_reasonable_sentence_overflow_and_abbreviations():
    processor = app_module._create_text_processor_for_engine(
        "audio8_tts",
        app_module.DEFAULT_CONFIG["chunk_size"],
        {
            **app_module.DEFAULT_CONFIG,
            "audio8_tts_chunk_size": 140,
            "audio8_tts_hard_chunk_size": 400,
        },
    )
    text = (
        "Holding communist beliefs or joining the Communist Party USA is fully legal for U.S. "
        "citizens under First Amendment speech and association protections. "
        "However, federal law addresses communism through several historical statutes."
    )

    chunks = processor.process_text(text)[0]["chunks"]

    assert chunks == [
        "Holding communist beliefs or joining the Communist Party USA is fully legal for U.S. "
        "citizens under First Amendment speech and association protections.",
        "However, federal law addresses communism through several historical statutes.",
    ]
    assert " ".join(chunks) == text


def test_audio8_processor_splits_extreme_sentence_at_clause_boundary():
    processor = app_module._create_text_processor_for_engine(
        "audio8_tts",
        app_module.DEFAULT_CONFIG["chunk_size"],
        {
            **app_module.DEFAULT_CONFIG,
            "audio8_tts_chunk_size": 140,
            "audio8_tts_hard_chunk_size": 200,
        },
    )
    text = (
        "This sentence begins with a substantial independent clause, "
        "then it continues with enough descriptive language to pass the configured hard limit "
        "while still offering punctuation that can provide a natural place to divide the performance "
        "without resorting to an arbitrary split in the middle of a word."
    )
    chunks = processor.process_text(text)[0]["chunks"]
    assert len(chunks) >= 2
    assert max(map(len, chunks)) <= 200
    assert " ".join(chunks) == text


def test_audio8_uses_voice_library_transcript_key(tmp_path):
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFF-test-audio")
    stat = voice.stat()
    import hashlib
    key = hashlib.md5(f"{voice.name}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:16]
    engine = Audio8TTSEngine.__new__(Audio8TTSEngine)
    engine._transcripts = {key: "Exact reference words."}
    assert engine._transcript_for(voice) == "Exact reference words."


def test_audio8_defaults_and_frontend_controls_are_present():
    assert app_module.DEFAULT_CONFIG["audio8_tts_chunk_size"] == 140
    assert app_module.DEFAULT_CONFIG["audio8_tts_hard_chunk_size"] == 400
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    settings = (ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    assert 'value="audio8_tts"' in template
    assert 'id="engine-panel-audio8-tts"' in template
    assert "audio8_tts_model_id" in settings
    assert "audio8_tts_retry_max_new_tokens" in settings
    assert "audio8_tts_hard_chunk_size" in settings
