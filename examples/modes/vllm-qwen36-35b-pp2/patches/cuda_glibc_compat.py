"""Reproduce the CUDA 13.0 / glibc >= 2.42 rsqrt declaration fix locally.

Only touches this mode's CUDA files. Header replacement is atomic to avoid
changing uv's hard-linked package cache. This helper is for the recorded
nvidia-cuda-crt 13.0.88 environment; re-evaluate it when upgrading.
"""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import sysconfig
import tempfile


def main():
    expected = Path(__file__).resolve().parent.parent / ".venv"
    if Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError("Run this helper with the mode-local .venv/bin/python")
    cuda_home = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    # NVIDIA's pip toolkit uses lib/, while FlashInfer's linker expects lib64/.
    for path, target in ((cuda_home / "lib64", "lib"),
                         (cuda_home / "lib/libcudart.so", "libcudart.so.13")):
        if not path.exists() and not path.is_symlink():
            path.symlink_to(target)
        if not path.exists():
            raise RuntimeError(f"Missing mode-local CUDA library target: {path}")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 42):
        print("CUDA/glibc compatibility patch is not needed")
        return
    if importlib.metadata.version("nvidia-cuda-crt") != "13.0.88":
        raise RuntimeError("Review the compatibility patch for the new CUDA CRT version")
    header = cuda_home / "include/crt/math_functions.h"
    original = header.read_bytes()
    updated = original
    for declaration in (b"rsqrt(double x)", b"rsqrtf(float x)"):
        before = declaration + b";"
        after = declaration + b" noexcept(true);"
        if updated.count(after) == 1:
            continue
        if updated.count(before) != 1:
            raise RuntimeError(f"Unexpected CUDA header declaration: {declaration!r}")
        updated = updated.replace(before, after, 1)
    if updated != original:
        with tempfile.NamedTemporaryFile(dir=header.parent, delete=False) as stream:
            stream.write(updated)
            temporary = Path(stream.name)
        try:
            temporary.chmod(header.stat().st_mode & 0o777)
            os.replace(temporary, header)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"CUDA/glibc header ready: {header}; sha256={hashlib.sha256(updated).hexdigest()}")


if __name__ == "__main__":
    main()
