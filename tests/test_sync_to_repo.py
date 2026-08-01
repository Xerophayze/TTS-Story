from __future__ import annotations

import json

from sync_to_repo import CONFIG_SECRET_KEYS, scrub_config


def test_scrub_config_removes_every_supported_cloud_credential(tmp_path):
    source = tmp_path / "config.local.json"
    destination = tmp_path / "config.scrubbed.json"
    payload = {key: f"secret-for-{key}" for key in CONFIG_SECRET_KEYS}
    payload["tts_engine"] = "openai_tts"
    source.write_text(json.dumps(payload), encoding="utf-8")

    assert scrub_config(str(source), str(destination)) is True

    scrubbed = json.loads(destination.read_text(encoding="utf-8"))
    assert scrubbed["openai_tts_api_key"] == ""
    assert all(scrubbed[key] == "" for key in CONFIG_SECRET_KEYS)
    assert scrubbed["tts_engine"] == "openai_tts"


def test_safety_scanner_and_sync_scrubber_cover_the_same_secret_keys():
    from scripts.check_repo_safety import SECRET_CONFIG_KEYS

    assert CONFIG_SECRET_KEYS == SECRET_CONFIG_KEYS
