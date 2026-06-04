#!/usr/bin/env python
"""Validate that the installed PyTorch build can actually run on CUDA."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe PyTorch CUDA compatibility.")
    parser.add_argument(
        "--require-arch",
        default="",
        help="Required compiled CUDA architecture, for example sm_120.",
    )
    parser.add_argument(
        "--test-cuda",
        action="store_true",
        help="Run a tiny CUDA tensor operation when CUDA is available.",
    )
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"ERROR: Unable to import torch: {exc}")
        return 2

    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Runtime: {torch.version.cuda or 'none'}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if not cuda_available:
        print("CUDA Device: CPU-only")
        return 0 if not args.require_arch and not args.test_cuda else 3

    try:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"CUDA Device: {device_name}")
        print(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
    except Exception as exc:
        print(f"ERROR: Unable to query CUDA device: {exc}")
        return 4

    arch_list = []
    try:
        arch_list = list(torch.cuda.get_arch_list())
    except Exception as exc:
        print(f"WARNING: Unable to read compiled CUDA architecture list: {exc}")
    print(f"Compiled CUDA Architectures: {', '.join(arch_list) if arch_list else 'unknown'}")

    if args.require_arch and arch_list and args.require_arch not in arch_list:
        print(
            "ERROR: Installed PyTorch build does not include "
            f"{args.require_arch}. Install a CUDA build that supports this GPU."
        )
        return 5

    if args.test_cuda:
        try:
            value = (torch.ones((1,), device="cuda") + 1).item()
            torch.cuda.synchronize()
            print(f"CUDA Tensor Test: OK ({value:g})")
        except Exception as exc:
            print(f"ERROR: CUDA tensor test failed: {exc}")
            return 6

    return 0


if __name__ == "__main__":
    sys.exit(main())
