import json
from pathlib import Path


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
    assert unix_setup.index("touch .setup_complete") < unix_setup.index('echo "Setup Complete!"')
    assert windows_setup.index('type nul > ".setup_complete"') < windows_setup.index("echo Setup Complete!")
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


def test_pinokio_start_captures_the_actual_server_url():
    start = load_json("start.json")
    event = start["run"][0]["params"]["on"][0]

    assert event == {"event": r"/(http:\/\/[0-9.:]+)/", "done": True}
    assert start["run"][1]["method"] == "local.set"
    assert start["run"][1]["params"]["url"] == "{{input.event[1]}}"
