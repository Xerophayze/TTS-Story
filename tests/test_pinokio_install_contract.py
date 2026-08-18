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


def test_unix_setup_discovers_compatible_macos_python_instead_of_using_default_only():
    helper = read("scripts/unix_torch.sh")
    setup = read("setup.sh")
    updater = read("install-update.sh")

    assert "tts_story_find_compatible_python" in helper
    assert "python3.11 python3.12 python3.10 python3.9" in helper
    assert "/opt/homebrew/opt/python@3.$minor/bin/python3.$minor" in helper
    assert "/usr/local/opt/python@3.$minor/bin/python3.$minor" in helper
    assert "/Library/Frameworks/Python.framework/Versions/3.$minor/bin/python3" in helper
    assert '"$pyenv_root"/versions/3."$minor"*/bin/python3' in helper
    assert "TTS_STORY_PYTHON" in helper

    assert 'PYTHON_BIN="$(tts_story_find_compatible_python || true)"' in setup
    assert '"$PYTHON_BIN" -m venv venv' in setup
    assert "python3 -m venv venv" not in setup
    assert "Python selection and venv support will be verified by setup.sh." in updater
    assert 'TEST_VENV="/tmp/venv_test_$$"' not in updater


def test_setup_completion_marker_only_follows_verification():
    unix_setup = read("setup.sh")
    windows_setup = read("setup.bat")

    assert unix_setup.index("python scripts/torch_cuda_probe.py", unix_setup.index("[10/10]")) < unix_setup.index(
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
    assert reset["run"][2]["params"]["path"] == ".pinokio"


def test_linux_pinokio_install_uses_managed_dependencies_without_sudo():
    install = load_json("install.json")
    update = load_json("update.json")
    setup = read("setup.sh")

    for launcher in (install, update):
        linux_steps = [step for step in launcher["run"] if "platform === 'linux'" in step.get("when", "")]
        managed_steps = [step for step in linux_steps if step.get("when") == "{{platform === 'linux'}}"]
        assert len(linux_steps) == 3

        dependency_params = managed_steps[0]["params"]
        assert dependency_params["conda"] == {"path": ".pinokio", "python": "python=3.11"}
        assert "conda install -y -c conda-forge" in dependency_params["message"]
        for package in ("git", "ffmpeg", "sox"):
            assert package in dependency_params["message"]

        brew_steps = [step for step in linux_steps if "which('brew')" in step.get("when", "")]
        assert len(brew_steps) == 1
        assert brew_steps[0]["params"]["message"] == "brew install espeak-ng rubberband"

        setup_params = managed_steps[1]["params"]
        assert setup_params["conda"] == ".pinokio"
        assert setup_params["env"]["TTS_STORY_PINOKIO"] == "1"
        assert all("sudo" not in str(step) for step in linux_steps)

    assert 'PINOKIO_MODE="${TTS_STORY_PINOKIO:-0}"' in setup
    assert 'if [ "$PINOKIO_MODE" = "1" ]; then' in setup
    assert "Pinokio managed-environment mode enabled (no sudo prompts)." in setup


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


def test_setup_scripts_manage_optional_flash_attention_without_making_it_required():
    windows_setup = read("setup.bat")
    unix_setup = read("setup.sh")
    helper = read("scripts/flash_attention_setup.py")

    assert "scripts\\flash_attention_setup.py install" in windows_setup
    assert "scripts/flash_attention_setup.py install" in unix_setup
    assert "INSTALL_FLASH_ATTN" in windows_setup
    assert "INSTALL_FLASH_ATTN" in unix_setup
    assert "--no-build-isolation" in helper
    assert "TTS-Story will use PyTorch SDPA acceleration instead." in helper
    assert "offer-prerequisites" in windows_setup
    assert "Nvidia.CUDA" in helper
    assert "Microsoft.VisualStudio.2022.BuildTools" in helper
    assert "Microsoft.VisualStudio.Workload.VCTools" in helper


def test_pinokio_disables_interactive_system_prerequisite_prompts():
    install = load_json("install.json")
    update = load_json("update.json")

    for launcher in (install, update):
        non_linux = [step for step in launcher["run"] if step.get("when") == "{{platform !== 'linux'}}"]
        assert len(non_linux) == 1
        assert non_linux[0]["params"]["env"]["TTS_STORY_PINOKIO"] == "1"


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
