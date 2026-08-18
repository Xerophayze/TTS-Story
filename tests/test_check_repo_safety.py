import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_safety.py"
SPEC = importlib.util.spec_from_file_location("check_repo_safety", MODULE_PATH)
assert SPEC and SPEC.loader
SAFETY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFETY)


def test_blocks_runtime_and_credential_paths():
    assert SAFETY.path_problem("data/prep/story.json")
    assert SAFETY.path_problem("static/samples/voice.wav")
    assert SAFETY.path_problem("jobs.db")
    assert SAFETY.path_problem("secrets/private.key")
    assert SAFETY.path_problem("engines/chatterbox/.venv/Scripts/python.exe")
    assert SAFETY.path_problem("engines/new-engine/.engine_ready")
    assert SAFETY.path_problem("engines/new-engine/models/model.safetensors")
    assert SAFETY.path_problem(".codex-remote-attachments/private/screenshot.png")
    assert SAFETY.path_problem(".cache/huggingface/model.bin")
    assert SAFETY.path_problem("data/engine-first-run.json")
    assert SAFETY.path_problem("temp_requirements_filtered5.txt")


def test_allows_placeholders_and_tracked_engine_worker():
    assert SAFETY.path_problem("static/samples/.gitkeep") is None
    assert SAFETY.path_problem("engines/index-tts/tts_worker.py") is None
    assert SAFETY.path_problem(".env.example") is None


def test_local_config_is_always_rejected_from_repository_content():
    assert SAFETY.path_problem("config.json") == "runtime or backup file"


def test_detects_populated_config_secrets():
    content = b'{"atlas_cloud_api_key": "atlas-value", "openrouter_api_key": "router-value", "azure_speech_key": "azure-value", "elevenlabs_api_key": "eleven-value", "openai_tts_api_key": "openai-value", "gemini_api_key": "real-value", "replicate_api_key": ""}'
    problems = SAFETY.content_problems("config.json", content)
    assert any("atlas_cloud_api_key" in problem for problem in problems)
    assert any("openrouter_api_key" in problem for problem in problems)
    assert any("azure_speech_key" in problem for problem in problems)
    assert any("elevenlabs_api_key" in problem for problem in problems)
    assert any("openai_tts_api_key" in problem for problem in problems)
    assert any("gemini_api_key" in problem for problem in problems)


def test_accepts_scrubbed_config():
    content = b'{"atlas_cloud_api_key": "", "openrouter_api_key": "", "azure_speech_key": "", "elevenlabs_api_key": "", "openai_tts_api_key": "", "gemini_api_key": "", "replicate_api_key": ""}'
    assert SAFETY.content_problems("config.json", content) == []


def test_detects_high_confidence_token_pattern():
    token = b"AKIA" + (b"A" * 16)
    assert "AWS access key" in SAFETY.content_problems("settings.txt", token)


def test_detects_openrouter_token_pattern():
    token = b"sk-or-v1-" + (b"A" * 40)
    assert "OpenAI-style key" in SAFETY.content_problems("settings.txt", token)


def test_detects_opaque_api_key_by_assignment_context():
    content = b'azure_speech_key = "0123456789abcdef0123456789abcdef"'
    assert "credential-like populated key or token setting" in SAFETY.content_problems(
        "settings.py", content
    )


def test_allows_documented_placeholder_key_assignment():
    content = b'api_key = "your-api-key-placeholder"'
    assert SAFETY.content_problems("README.md", content) == []
