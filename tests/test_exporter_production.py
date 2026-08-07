from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import app as app_module
import src.audio_merger as audio_merger_module
from src.audio_merger import AudioMerger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_acx_mp3_command_uses_only_192k_cbr_mode(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"placeholder")
    output = tmp_path / "output.mp3"
    captured = {}

    monkeypatch.setattr(audio_merger_module, "_find_ffmpeg", lambda: "ffmpeg")

    def fake_run(command, **kwargs):
        captured["command"] = [str(value) for value in command]
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio_merger_module.subprocess, "run", fake_run)

    assert AudioMerger(acx_compliance=True)._merge_with_ffmpeg(
        [str(source)], output, "mp3"
    )

    command = captured["command"]
    assert command.count("-b:a") == 1
    bitrate_index = command.index("-b:a")
    assert command[bitrate_index + 1] == "192k"
    assert "-q:a" not in command
    assert "-qscale:a" not in command
    assert "-abr" not in command


def test_library_rebuild_prefers_saved_export_settings(monkeypatch, tmp_path):
    job_dir = tmp_path / "saved-export-job"
    job_dir.mkdir()
    (job_dir / "metadata.json").write_text(
        json.dumps({
            "output_format": "mp3",
            "intro_silence_ms": 1500,
            "inter_chunk_silence_ms": 400,
            "acx_compliance": True,
            "output_bitrate_kbps": 0,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "load_config", lambda: {
        "output_format": "wav",
        "crossfade_duration": 1.0,
        "intro_silence_ms": 0,
        "inter_chunk_silence_ms": 0,
        "output_bitrate_kbps": 64,
        "acx_compliance": False,
    })

    options = app_module._load_library_merge_options(
        job_dir,
        {"output_format": "ogg", "crossfade_duration": 0.25},
    )

    assert options["output_format"] == "mp3"
    assert options["crossfade_duration"] == 0.25
    assert options["intro_silence_ms"] == 1500
    assert options["inter_chunk_silence_ms"] == 400
    assert options["output_bitrate_kbps"] == 0
    assert options["acx_compliance"] is True


def _write_legacy_m4b_job(job_dir: Path) -> None:
    job_dir.mkdir(parents=True)
    (job_dir / "chapter.mp3").write_bytes(b"chapter-audio")
    (job_dir / "metadata.json").write_text(
        json.dumps({
            "chapter_mode": True,
            "output_format": "mp3",
            "chapters": [{
                "index": 0,
                "title": "Chapter 1.******",
                "relative_path": "chapter.mp3",
            }],
        }),
        encoding="utf-8",
    )


def test_m4b_route_sanitizes_legacy_stored_chapter_titles(monkeypatch, tmp_path):
    job_id = "legacy-title-job"
    job_dir = tmp_path / job_id
    _write_legacy_m4b_job(job_dir)
    captured = {}

    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path)

    def fake_merge(self, *, output_path, chapter_metadata, **kwargs):
        captured["chapter_metadata"] = chapter_metadata
        Path(output_path).write_bytes(b"m4b-output")
        return output_path

    monkeypatch.setattr(AudioMerger, "merge_to_m4b", fake_merge)

    response = app_module.app.test_client().post(
        f"/api/download/{job_id}/m4b",
        json={"bitrate": 128},
    )

    assert response.status_code == 200
    assert captured["chapter_metadata"][0]["title"] == "Chapter 1."


def test_m4b_route_requires_confirmation_for_non_square_cover(monkeypatch, tmp_path):
    job_id = "cover-guard-job"
    job_dir = tmp_path / job_id
    _write_legacy_m4b_job(job_dir)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path)

    cover = io.BytesIO()
    Image.new("RGB", (1600, 900), "navy").save(cover, format="PNG")
    cover_data = "data:image/png;base64," + base64.b64encode(cover.getvalue()).decode("ascii")

    response = app_module.app.test_client().post(
        f"/api/download/{job_id}/m4b",
        json={"bitrate": 128, "cover_art": cover_data},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["requires_non_square_confirmation"] is True
    assert payload["cover_width"] == 1600
    assert payload["cover_height"] == 900


def test_m4b_dialog_previews_dimensions_and_sends_cover_confirmation():
    source = (PROJECT_ROOT / "static" / "js" / "library.js").read_text(encoding="utf-8")

    assert 'id="m4b-cover-preview"' in source
    assert "image.naturalWidth" in source
    assert "Explicit confirmation will be required" in source
    assert "allow_non_square_cover: allowNonSquareCover" in source
    assert "TTS-Story will not crop it" in source
