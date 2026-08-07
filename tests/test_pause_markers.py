from __future__ import annotations

import wave
from pathlib import Path

import pytest

from src.pause_markers import pause_seconds_for_text, sanitize_display_title, write_silence_wav
from src.text_processor import TextProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_three_star_groups_become_ordered_standalone_chunks():
    chunks = TextProcessor(chunk_size=100).chunk_text(
        "First sentence.\n\n***\n\nSecond sentence.\n******\nThird sentence."
    )

    assert chunks == [
        "First sentence.",
        "***",
        "Second sentence.",
        "******",
        "Third sentence.",
    ]


def test_inline_pause_markers_after_periods_become_ordered_chunks():
    chunks = TextProcessor(chunk_size=100).chunk_text(
        "The man could not see.*** Then he heard a sound.****** Afterward, he ran."
    )

    assert chunks == [
        "The man could not see.",
        "***",
        "Then he heard a sound.",
        "******",
        "Afterward, he ran.",
    ]


def test_inline_pause_marker_inside_speaker_tag_is_preserved():
    segments = TextProcessor(chunk_size=100).process_text(
        "[narrator]Chapter 1.****** The man could not see.*** He waited.[/narrator]"
    )

    assert [chunk for segment in segments for chunk in segment["chunks"]] == [
        "Chapter 1.",
        "******",
        "The man could not see.",
        "***",
        "He waited.",
    ]


def test_pause_marker_between_speaker_tags_is_not_discarded():
    segments = TextProcessor(chunk_size=100).process_text(
        "[narrator]First sentence.[/narrator]\n***\n"
        "[alice]Second sentence.[/alice]"
    )

    assert [chunk for segment in segments for chunk in segment["chunks"]] == [
        "First sentence.",
        "***",
        "Second sentence.",
    ]


def test_pause_marker_duration_scales_by_quarter_second_per_three_stars():
    assert pause_seconds_for_text("***") == 0.25
    assert pause_seconds_for_text("******") == 0.5
    assert pause_seconds_for_text("*********") == 0.75
    assert pause_seconds_for_text("**") is None
    assert pause_seconds_for_text("This is *** emphasis.") is None


def test_markdown_style_closing_asterisks_are_not_pause_markers():
    chunks = TextProcessor(chunk_size=100).chunk_text("This is important***")

    assert chunks == ["This is important***"]


def test_display_title_removes_only_trailing_recognized_pause_groups():
    assert sanitize_display_title("Chapter 1.******") == "Chapter 1."
    assert sanitize_display_title("The Preface.  ***  ") == "The Preface."
    assert sanitize_display_title("Stars *** inside a title") == "Stars *** inside a title"
    assert sanitize_display_title("Chapter 1.**") == "Chapter 1.**"


def test_silence_wav_has_requested_duration(tmp_path):
    destination = write_silence_wav(tmp_path / "pause.wav", 1.0, sample_rate=16000)

    with wave.open(str(destination), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() / wav_file.getframerate() == pytest.approx(1.0)


def test_audio_job_filters_pause_markers_before_calling_any_tts_engine():
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    start = source.index("def generate_chunks(")
    end = source.index("def _prebuild_subprocess_engine_all_chapters", start)
    generation = source[start:end]

    assert "pause_seconds_for_text(chunk_text)" in generation
    assert '"segments": render_segments if has_pause_markers else segments' in generation
    assert "write_silence_wav(" in generation
    assert "commit_pauses_before" in generation
