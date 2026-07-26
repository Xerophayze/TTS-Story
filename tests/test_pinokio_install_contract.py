import json
from pathlib import Path

from scripts.setup_state import calculate_fingerprint, write_state


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def load_json(name: str) -> dict:
    return json.loads(read(name))


def test_unix_installers_do_not_request_nonexistent_cpu_suffix_packages():
    setup = read("setup.sh")
    run = read("run.sh")

    assert "torch==${TORCH_VERSION}+cpu" not in setup
    assert "torch==${TORCH_PIN}+cpu" not in run
    assert "torchvision==${TORCHVISION_VERSION}+cpu" not in setup
    assert "torchaudio==${TORCHAUDIO_PIN}+cpu" not in run
    assert "tts_story_install_cpu_torch python" in setup
    assert "tts_story_install_cpu_torch python" in run


def test_shared_torch_helper_routes_macos_to_pypi():
    helper = read("scripts/unix_torch.sh")
    darwin_branch, separator, linux_branch = helper.partition("\n    else\n")

    assert separator
    assert '"$(uname -s)" = "Darwin"' in darwin_branch
    assert "download.pytorch.org/whl/cpu" not in darwin_branch
    assert "download.pytorch.org/whl/cpu" in linux_branch
    assert "(3, 9) <= sys.version_info[:2] < (3, 13)" in helper


def test_setup_completion_marker_only_follows_verification():
    unix_setup = read("setup.sh")
    windows_setup = read("setup.bat")

    assert unix_setup.index("python scripts/torch_cuda_probe.py", unix_setup.index("[12/12]")) < unix_setup.index(
        "touch .setup_complete"
    )
    assert unix_setup.index("touch .setup_complete") < unix_setup.rindex('echo "Setup Complete!"')
    assert windows_setup.index('type nul > ".setup_complete"') < windows_setup.rindex("echo Setup Complete!")
    assert unix_setup.index('setup_state.py write') < unix_setup.index("touch .setup_complete")
    assert windows_setup.index("setup_state.py write") < windows_setup.index('type nul > ".setup_complete"')
    assert 'rm -f .setup_complete' in unix_setup
    assert 'del /q ".setup_complete"' in windows_setup


def test_pinokio_launcher_uses_verified_install_state_and_ai_bundle():
    launcher = read("pinokio.js")
    install = load_json("install.json")
    reset = load_json("reset.json")

    assert 'info.exists("venv") && info.exists(".setup_complete")' in launcher
    assert install["requires"] == {"bundle": "ai"}
    assert reset["run"][0]["method"] == "fs.rm"
    assert reset["run"][0]["params"]["path"] == ".setup_complete"
    assert reset["run"][1]["params"]["path"] == ".setup_state.json"


def test_pinokio_start_captures_the_actual_server_url():
    start = load_json("start.json")
    event = start["run"][0]["params"]["on"][0]

    assert event == {"event": r"/(http:\/\/[0-9.:]+)/", "done": True}
    assert start["run"][1]["method"] == "local.set"
    assert start["run"][1]["params"]["url"] == "{{input.event[1]}}"


def test_windows_venv_probe_does_not_use_fragile_quoted_for_command():
    setup = read("setup.bat")

    assert """in ('"%VENV_PYTHON%" -c""" not in setup
    assert '"%VENV_PYTHON%" -c "import sys;' in setup
    assert "VENV_VERSION_FILE" in setup


def test_existing_native_installs_use_dependency_aware_update_mode():
    installer = read("install-update.bat")

    assert installer.count("call setup.bat --update") == 2
    assert 'set "EXISTING_INSTALL=1"' in installer
    assert "call setup.bat\n" in installer


def test_setup_scripts_have_fast_update_and_explicit_repair_paths():
    windows_setup = read("setup.bat")
    unix_setup = read("setup.sh")

    for setup in (windows_setup, unix_setup):
        assert "setup_state.py matches" in setup
        assert "--repair" in setup
        assert "Unknown setup option" in setup
        assert "No setup work is required for this update." in setup


def test_setup_fingerprint_ignores_docs_but_tracks_dependency_definitions(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "requirements.txt").write_text("flask==1\n", encoding="utf-8")
    (tmp_path / "setup.bat").write_text("setup-v1\n", encoding="utf-8")
    (tmp_path / "scripts" / "setup_state.py").write_text("schema-v1\n", encoding="utf-8")

    original = calculate_fingerprint(tmp_path, "windows")
    (tmp_path / "README.md").write_text("documentation changed\n", encoding="utf-8")
    assert calculate_fingerprint(tmp_path, "windows") == original

    (tmp_path / "requirements.txt").write_text("flask==2\n", encoding="utf-8")
    assert calculate_fingerprint(tmp_path, "windows") != original


def test_setup_state_is_written_outside_the_venv(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "setup.bat").write_text("setup\n", encoding="utf-8")
    (tmp_path / "scripts" / "setup_state.py").write_text("state\n", encoding="utf-8")
    state_path = tmp_path / ".setup_state.json"

    write_state(tmp_path, state_path, "windows")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert state["platform"] == "windows"
    assert state["fingerprint"] == calculate_fingerprint(tmp_path, "windows")


def test_kitten_prefetch_uses_portable_timeout_runner():
    setup = read("setup.bat")
    runner = read("scripts/run_with_timeout.py")

    assert "Start-Process" not in setup
    assert "scripts\\run_with_timeout.py --timeout 300" in setup
    assert "subprocess.run" in runner
    assert "subprocess.TimeoutExpired" in runner
