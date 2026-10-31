"""Optional stdlib helpers for adapters; model-specific behavior stays in modes."""
import importlib.metadata
import json
import os
from pathlib import Path
import tomllib
import urllib.request


def load(directory):
    directory = Path(directory).resolve()
    project = directory.parent.parent
    cfg = tomllib.loads((directory / "mode.toml").read_text())
    machine = tomllib.loads((project / "machine.toml").read_text())
    return directory, project, cfg, machine


def local_path(directory, value):
    path = Path(value)
    return path if path.is_absolute() else Path(directory) / path


def require_files(paths):
    for path in paths:
        if not Path(path).is_file():
            raise ValueError("Required local file missing: " + str(path))


def require_versions(versions):
    for name, expected in versions.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Expected {name} {expected}, found {actual}; prepare explicitly")


def environment(directory, cfg, machine):
    directory = Path(directory)
    runtime = directory.parent.parent / ".runtime" / cfg["id"]
    # Called only while constructing a launch, never by status/check.
    directories = {"HF_HOME": runtime / "cache/huggingface",
                   "TORCH_HOME": runtime / "cache/torch",
                   "XDG_CACHE_HOME": runtime / "cache/xdg",
                   "XDG_STATE_HOME": runtime / "state",
                   "TRITON_CACHE_DIR": runtime / "cache/triton",
                   "TMPDIR": runtime / "cache/tmp"}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({key: str(value) for key, value in cfg.get("environment", {}).items()})
    env.update({key: str(value) for key, value in directories.items()})
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(machine["gpu"]["uuids"]),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PATH=str(directory / ".venv/bin") + os.pathsep + env.get("PATH", ""))
    return env


def get_json(url, headers=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers=headers or {}), timeout=5) as response:
        return json.load(response)
