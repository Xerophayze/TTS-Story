from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from scripts import flash_attention_setup


def test_windows_prerequisite_installer_pins_matching_cuda_and_cpp_workload(monkeypatch):
    commands = []

    monkeypatch.setattr(flash_attention_setup.os, "name", "nt")
    monkeypatch.setattr(flash_attention_setup.shutil, "which", lambda name: "winget.exe" if name == "winget" else None)
    monkeypatch.setattr(flash_attention_setup, "_cuda_compiler_path", lambda: None)
    monkeypatch.setattr(flash_attention_setup, "_compiler_details", lambda: ("", None))

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(flash_attention_setup.subprocess, "run", fake_run)

    assert flash_attention_setup.install_windows_prerequisites() == 0
    assert len(commands) == 2
    cuda_command, build_tools_command = commands
    assert "Nvidia.CUDA" in cuda_command
    assert cuda_command[cuda_command.index("--version") + 1] == "12.4"
    assert "Microsoft.VisualStudio.2022.BuildTools" in build_tools_command
    override = build_tools_command[build_tools_command.index("--override") + 1]
    assert "Microsoft.VisualStudio.Workload.VCTools" in override


def test_prerequisite_offer_defaults_to_safe_skip(monkeypatch):
    monkeypatch.setattr(flash_attention_setup.os, "name", "nt")
    monkeypatch.setattr(flash_attention_setup, "toolchain_needed", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert flash_attention_setup.offer_windows_prerequisites() == 0


def test_visual_studio_environment_handles_install_path_with_spaces(monkeypatch, tmp_path):
    vcvars = tmp_path / "Microsoft Visual Studio" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    vcvars.parent.mkdir(parents=True)
    vcvars.write_text("@echo off\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(flash_attention_setup.os, "name", "nt")
    monkeypatch.setattr(flash_attention_setup.shutil, "which", lambda _name, **_kwargs: None)
    monkeypatch.setattr(Path, "glob", lambda _self, _pattern: [vcvars])
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["shell"] = kwargs.get("shell")
        return SimpleNamespace(returncode=0, stdout="Path=C:\\MSVC\\bin\nVSCMD_ARG_TGT_ARCH=x64\n", stderr="")

    monkeypatch.setattr(flash_attention_setup.subprocess, "run", fake_run)
    environment = flash_attention_setup._visual_studio_environment()

    assert captured["shell"] is True
    assert str(vcvars) in captured["command"]
    assert environment["PATH"] == r"C:\MSVC\bin"


def test_flash_build_limits_cuda_architecture_to_detected_gpu_family():
    helper = Path(flash_attention_setup.__file__).read_text(encoding="utf-8")
    assert 'env.setdefault("FLASH_ATTN_CUDA_ARCHS", flash_arch)' in helper
    assert '{8: "80", 9: "90", 10: "100", 11: "110", 12: "120"}' in helper
