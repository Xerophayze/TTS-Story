#!/usr/bin/env python3
"""Detect, install, and validate the optional FlashAttention 2 runtime."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CUDA_TOOLKIT_VERSION = "12.4"
CUDA_DOWNLOAD_URL = "https://developer.nvidia.com/cuda-12-4-0-download-archive"
MSVC_INSTRUCTIONS_URL = (
    "https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170"
)


def _command_path(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    return None


def _cuda_compiler_path() -> str | None:
    """Prefer a CUDA Toolkit matching the CUDA runtime bundled with PyTorch."""
    preferred = CUDA_TOOLKIT_VERSION
    torch, _error = _torch_details()
    if torch is not None and getattr(torch.version, "cuda", None):
        preferred = str(torch.version.cuda)
    normalized = preferred.replace(".", "_")
    candidates: list[Path] = []
    for variable in (f"CUDA_PATH_V{normalized}", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"))
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        candidates.insert(0, root / f"v{preferred}" / "bin" / "nvcc.exe")
        if root.is_dir():
            candidates.extend(sorted(root.glob("v*/bin/nvcc.exe"), reverse=True))
    direct = shutil.which("nvcc")
    if direct:
        candidates.append(Path(direct))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


def _visual_studio_environment() -> dict[str, str] | None:
    """Load the x64 MSVC environment even outside a Developer Command Prompt."""
    if os.name != "nt":
        return None
    candidates: list[Path] = []
    vswhere = shutil.which("vswhere")
    if not vswhere:
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        if program_files_x86:
            candidate = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
            if candidate.is_file():
                vswhere = str(candidate)
    if vswhere:
        result = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            candidates.append(Path(result.stdout.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
    for root_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(root_name)
        if root:
            candidates.extend(Path(root).glob("Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat"))

    vcvars = next((candidate for candidate in candidates if candidate.is_file()), None)
    if not vcvars:
        return None
    # shell=True is intentional here: CALL must run inside cmd.exe so the
    # vcvars64.bat environment changes can be captured. Passing this command
    # through list2cmdline adds another quote layer and breaks paths containing
    # spaces, which is the normal Visual Studio installation location.
    result = subprocess.run(
        f'call "{vcvars}" >nul && set',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    environment = os.environ.copy()
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            # Environment names are case-insensitive on Windows, while the
            # copied dict is not. Normalize PATH so shutil.which sees MSVC.
            environment[key.upper()] = value
    return environment


def _compiler_details() -> tuple[str, dict[str, str] | None]:
    if os.name != "nt":
        return _command_path("c++") or _command_path("g++") or "", None
    direct = _command_path("cl")
    if direct:
        return direct, None
    environment = _visual_studio_environment()
    if environment:
        compiler = shutil.which("cl", path=environment.get("PATH"))
        if compiler:
            return compiler, environment
    return "", None


def _torch_details() -> tuple[object | None, str | None]:
    try:
        import torch

        return torch, None
    except Exception as exc:
        return None, str(exc)


def _print_runtime(torch: object) -> tuple[int, int]:
    capability = torch.cuda.get_device_capability(0)  # type: ignore[attr-defined]
    print(f"PyTorch: {torch.__version__}")  # type: ignore[attr-defined]
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'none'}")  # type: ignore[attr-defined]
    print(f"GPU: {torch.cuda.get_device_name(0)}")  # type: ignore[attr-defined]
    print(f"GPU compute capability: {capability[0]}.{capability[1]}")
    return capability


def _validate(*, force: bool = False) -> tuple[bool, str]:
    from src.attention_backend import validate_flash_attention

    return validate_flash_attention(force=force)


def diagnose() -> int:
    print("FlashAttention 2 compatibility check")
    print(f"Platform: {platform.system()} {platform.machine()}")
    torch, error = _torch_details()
    if torch is None:
        print(f"Status: unavailable (PyTorch import failed: {error})")
        return 0
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        print("Status: not applicable (no NVIDIA CUDA device is available)")
        print("TTS-Story will use PyTorch SDPA or eager attention instead.")
        return 0

    capability = _print_runtime(torch)
    if capability[0] < 8:
        print("Status: unsupported GPU (FlashAttention 2 requires Ampere or newer NVIDIA hardware)")
        return 0

    valid, detail = _validate(force=True)
    if valid:
        print(f"Status: ready (flash-attn {detail}; CUDA kernel test passed)")
        return 0

    print(f"Status: not installed or unusable ({detail})")
    print(f"CUDA compiler (nvcc): {_cuda_compiler_path() or 'not found'}")
    compiler_name = "cl" if os.name == "nt" else "c++"
    compiler_path, _compiler_env = _compiler_details()
    print(f"C++ compiler ({compiler_name}): {compiler_path or 'not found'}")
    if os.name == "nt":
        print(
            "Windows note: upstream FlashAttention does not provide dependable official Windows wheels. "
            "A source build requires the CUDA Toolkit and Visual Studio Build Tools with C++."
        )
    return 0


def install(*, required: bool = False) -> int:
    torch, error = _torch_details()
    if torch is None:
        print(f"FlashAttention skipped: PyTorch import failed ({error}).")
        return 1 if required else 0
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        print("FlashAttention skipped: no NVIDIA CUDA device is available.")
        return 0

    capability = _print_runtime(torch)
    if capability[0] < 8:
        print("FlashAttention skipped: the detected GPU is older than NVIDIA Ampere.")
        return 1 if required else 0

    valid, detail = _validate()
    if valid:
        print(f"FlashAttention is already installed and validated (version {detail}).")
        return 0

    if platform.system() == "Darwin":
        print("FlashAttention skipped: CUDA FlashAttention is not available on macOS.")
        return 0

    nvcc = _cuda_compiler_path()
    compiler, compiler_env = _compiler_details()
    if not nvcc or not compiler:
        print("FlashAttention could not be built automatically because its compiler toolchain is incomplete.")
        if not nvcc:
            print("  - CUDA Toolkit compiler nvcc was not found. The NVIDIA driver alone is insufficient.")
        if not compiler:
            if os.name == "nt":
                print("  - MSVC cl.exe was not found. Install Visual Studio Build Tools: Desktop development with C++.")
            else:
                print("  - A C++ compiler was not found (install the platform build-essential tools).")
        print("TTS-Story will use PyTorch SDPA acceleration instead.")
        return 1 if required else 0

    print(f"Building FlashAttention with nvcc={nvcc} and compiler={compiler}")
    env = compiler_env or os.environ.copy()
    env.setdefault("MAX_JOBS", "4")
    # Upstream otherwise builds kernels for every supported NVIDIA generation
    # (80, 90, 100, and 120). Limit the wheel to this machine's GPU family so a
    # consumer Windows build does not spend tens of minutes compiling kernels
    # it can never use.
    flash_arch = {8: "80", 9: "90", 10: "100", 11: "110", 12: "120"}.get(capability[0])
    if flash_arch:
        env.setdefault("FLASH_ATTN_CUDA_ARCHS", flash_arch)
        print(f"FlashAttention CUDA architecture target: {flash_arch}")
    cuda_home = str(Path(nvcc).resolve().parents[1])
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    env["PATH"] = str(Path(nvcc).resolve().parent) + os.pathsep + env.get("PATH", "")
    if os.name == "nt":
        env.setdefault("DISTUTILS_USE_SDK", "1")
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "packaging",
            "psutil",
            "ninja",
            "wheel",
            "setuptools",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            "flash-attn",
            "--no-build-isolation",
        ],
    ]
    try:
        for command in commands:
            subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: FlashAttention build failed with exit code {exc.returncode}.")
        print("TTS-Story will continue using PyTorch SDPA acceleration.")
        return 1 if required else 0

    valid, detail = _validate(force=True)
    if not valid:
        print(f"WARNING: FlashAttention installed but failed its CUDA kernel test: {detail}")
        return 1 if required else 0
    print(f"FlashAttention installation validated successfully (version {detail}).")
    return 0


def toolchain_needed() -> bool:
    torch, _error = _torch_details()
    if torch is None or not torch.cuda.is_available():  # type: ignore[attr-defined]
        return False
    capability = torch.cuda.get_device_capability(0)  # type: ignore[attr-defined]
    if capability[0] < 8:
        return False
    valid, _detail = _validate(force=True)
    if valid:
        return False
    compiler, _environment = _compiler_details()
    return not (_cuda_compiler_path() and compiler)


def open_prerequisite_pages() -> int:
    print("Opening official FlashAttention prerequisite pages...")
    print(f"CUDA Toolkit {CUDA_TOOLKIT_VERSION}: {CUDA_DOWNLOAD_URL}")
    print(f"Visual Studio C++ Build Tools instructions: {MSVC_INSTRUCTIONS_URL}")
    print("For Visual Studio, select the 'Desktop development with C++' workload.")
    webbrowser.open(CUDA_DOWNLOAD_URL)
    webbrowser.open(MSVC_INSTRUCTIONS_URL)
    return 0


def install_windows_prerequisites() -> int:
    if os.name != "nt":
        print("Automated system prerequisite installation is currently available only on Windows.")
        return 1
    winget = shutil.which("winget")
    if not winget:
        print("Windows Package Manager (winget) is unavailable. Opening the official download pages instead.")
        open_prerequisite_pages()
        return 1

    commands: list[list[str]] = []
    if not _cuda_compiler_path():
        commands.append([
            winget,
            "install",
            "--exact",
            "--id",
            "Nvidia.CUDA",
            "--version",
            CUDA_TOOLKIT_VERSION,
            "--source",
            "winget",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ])
    compiler, _environment = _compiler_details()
    if not compiler:
        commands.append([
            winget,
            "install",
            "--exact",
            "--id",
            "Microsoft.VisualStudio.2022.BuildTools",
            "--source",
            "winget",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--override",
            "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended",
        ])
    if not commands:
        print("The CUDA Toolkit and Visual Studio C++ Build Tools are already present.")
        return 0

    print("Installing system prerequisites. Windows may request administrator approval.")
    print("These developer tools require a large download and can use 10 GB or more of disk space.")
    for command in commands:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: prerequisite installer failed with exit code {exc.returncode}.")
            print("Use the official download-page option for manual installation.")
            return 1
    print("System prerequisite installers completed.")
    return 0


def offer_windows_prerequisites() -> int:
    if os.name != "nt" or not toolchain_needed():
        return 0
    print()
    print("Optional FlashAttention 2 system prerequisites are missing.")
    print("  [A] Install CUDA Toolkit 12.4 and Visual Studio 2022 C++ Build Tools with WinGet")
    print("  [O] Open the official download and instruction pages")
    print("  [S] Skip and continue using PyTorch SDPA (recommended if unsure)")
    try:
        choice = input("Choose A, O, or S [S]: ").strip().lower() or "s"
    except (EOFError, KeyboardInterrupt):
        choice = "s"
        print()
    if choice == "a":
        if install_windows_prerequisites() != 0:
            return 0
        print("Attempting the FlashAttention build with the newly installed toolchain...")
        install(required=False)
        return 0
    if choice == "o":
        return open_prerequisite_pages()
    print("Skipping system prerequisites. Qwen3 will use PyTorch SDPA acceleration.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("diagnose", "install", "offer-prerequisites", "open-prerequisites", "install-prerequisites"),
        nargs="?",
        default="diagnose",
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help="Return an error when FlashAttention cannot be installed (normally it is optional).",
    )
    args = parser.parse_args()
    if args.action == "install":
        return install(required=args.required)
    if args.action == "offer-prerequisites":
        return offer_windows_prerequisites()
    if args.action == "open-prerequisites":
        return open_prerequisite_pages()
    if args.action == "install-prerequisites":
        return install_windows_prerequisites()
    return diagnose()


if __name__ == "__main__":
    raise SystemExit(main())
