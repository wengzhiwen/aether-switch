"""Strata recipe and adapter hooks; lifecycle lives in server_manager."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib
import urllib.request

from server_manager.recipe import require_files, require_versions

ROOT = Path(__file__).resolve().parent


def config() -> dict:
    cfg = tomllib.loads((ROOT / "mode.toml").read_text())
    machine = tomllib.loads((ROOT.parent.parent / "machine.toml").read_text())
    cfg["gpu"]["device_uuids"] = machine["gpu"]["uuids"]
    cfg["paths"] = {"runtime": str(ROOT.parent.parent / ".runtime" / cfg["id"])}
    return cfg


def runtime_config(cfg: dict) -> dict:
    model, gpu, inference = cfg["model"], cfg["gpu"], cfg["inference"]
    runtime = Path(cfg["paths"]["runtime"])
    argv = ["--pack", model["pack"], "--native", model["gguf"], "--ple-gguf", model["shard2"],
            "--expert-profile", str(ROOT / "app/data/expert-profile.bin"),
            "--expert-cache", inference["expert_cache"], "--prefill", inference["prefill"],
            "--max-context", str(gpu["context_tokens"]), "--kv", inference["kv"],
            "--ple-io", inference["ple_io"], "--vram-reserve-mib", str(gpu["vram_reserve_mib"]),
            "--vram-reserve-later-mib", str(gpu["vram_reserve_later_mib"])]
    if model["mtp_enabled"]:
        argv += ["--mtp", str(Path(model["mtp"]) / "rt"), "--spec", str(inference["speculative_window"]),
                 "--spec-min-p", str(inference["speculative_min_probability"])]
    if inference.get("batch_mtp"):
        argv.append("--batch-mtp")
    if inference.get("batch_groups") is not None:
        argv += ["--batch-groups", str(inference["batch_groups"])]
    if inference.get("trim_stage_weights"):
        if gpu["layer_split"] == "auto":
            raise ValueError("trim_stage_weights requires an explicit layer_split")
        argv.append("--trim-stage-weights")
    if inference.get("conversation_cache_slots") is not None:
        argv += ["--conversation-cache-slots", str(inference["conversation_cache_slots"])]
    result = {"exe": str(ROOT / cfg["engine"]["executable"]), "args": argv, "cwd": str(ROOT / "app"),
              "tokenizer": str(Path(model["pack"]) / "tokenizer"), "model_name": cfg["api"]["model_name"],
              "host": cfg["api"]["host"], "port": cfg["api"]["port"], "api_key": "",
              "gpu": [0, 1], "gpu_roles": {"0": "helper", "1": "main"}, "gpu_order": "as_given",
              "layer_split": gpu["layer_split"], "parallel": gpu["parallel_requests"],
              "mtp": model["mtp_enabled"], "drafting": "mtp" if model["mtp_enabled"] else "off",
              "draft_vocab": model["draft_vocab"], "open_browser": False,
              "engine_ready_s": cfg["control"]["startup_timeout_seconds"],
              "log": str(runtime / "engine.log"), "lib_dirs": [str(ROOT / "engine")],
              "env": {**cfg.get("environment", {}), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                      "CUDA_VISIBLE_DEVICES": ",".join(gpu["device_uuids"]),
                      "STRATA_KEEP_LEFTOVERS": "1", "STRATA_START_RETRIES": "0"}}
    if cfg["vision"]["enabled"]:
        argv.append("--vision")
        result["vision"] = {"exe": str(ROOT / "engine/strata-vision"), "mmproj": model["mmproj"],
                            "model": model["gguf"], "gpu": cfg["vision"]["gpu"],
                            "max_tokens": cfg["vision"]["max_tokens"]}
    return result


def preflight(cfg: dict) -> dict:
    if not isinstance(cfg["gpu"]["parallel_requests"], int) or not 1 <= cfg["gpu"]["parallel_requests"] <= 8:
        raise ValueError("parallel_requests must be 1..8")
    runtime_config(cfg)  # Validate the resolved flags before a supervisor stops a running mode.
    pins = {}
    for line in (ROOT / "requirements.in").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==", 1)
            pins[name] = version
    require_versions(pins)
    required = [ROOT / cfg["engine"]["executable"], ROOT / "engine/engine-component.json",
                ROOT / "app/serve/server.py", ROOT / "app/data/expert-profile.bin"]
    required += [Path(cfg["model"][key]) for key in ("gguf", "shard2")]
    required += [Path(cfg["model"]["pack"]) / name for name in
                 ("index.txt", "native_experts.txt", "dense.bin", "tokenizer/tokenizer.json",
                  "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/token_type.json",
                  "tokenizer/chat_template.jinja")]
    if cfg["model"]["mtp_enabled"]:
        required += [Path(cfg["model"]["mtp"]) / "rt" / name for name in
                     ("experts.bin", "dense.bin", "dense.txt", "draft_vocab.bin")]
    if cfg["vision"]["enabled"]:
        required += [Path(cfg["model"]["mmproj"]), ROOT / "engine/strata-vision"]
    require_files(required)
    component = json.loads((ROOT / "engine/engine-component.json").read_text())
    if component["version"] != cfg["engine"]["version"]:
        raise ValueError("Native engine version differs from the prepared recipe")
    return {"files_checked": len(required), "scope": "files/final flags; no model loaded"}


def command(cfg: dict) -> list[str]:
    return [str(ROOT / ".venv/bin/python"), "-B", str(ROOT / "app/serve/server.py"),
            "--engine", "strata", "--config", str(Path(cfg["paths"]["runtime"]) / "server.json"),
            "--host", cfg["api"]["host"], "--port", str(cfg["api"]["port"]),
            "--offline", "--no-api-key", "--shutdown-timeout", "0"]


def launch_spec(cfg: dict) -> dict:
    runtime = Path(cfg["paths"]["runtime"])
    for name in ("cache/tmp", "cache/xdg", "cache/cuda", "state"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    path = runtime / "server.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(runtime_config(cfg), ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    env = dict(os.environ)
    env.pop("STRATA_API_KEY", None)
    env.update(cfg.get("environment", {}))
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": ",".join(cfg["gpu"]["device_uuids"]),
                "STRATA_KEEP_LEFTOVERS": "1", "STRATA_START_RETRIES": "0",
                "XDG_STATE_HOME": str(runtime / "state"), "XDG_CACHE_HOME": str(runtime / "cache/xdg"),
                "CUDA_CACHE_PATH": str(runtime / "cache/cuda"),
                "TMPDIR": str(runtime / "cache/tmp"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    argv = command(cfg)
    return {"argv": argv, "cwd": str(ROOT / "app"), "env": env}


def ready(cfg: dict) -> dict:
    base = f"http://127.0.0.1:{cfg['api']['port']}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/health", timeout=5) as response:
        health = json.load(response)
    if not health.get("loaded") or health.get("service") != "strata":
        raise RuntimeError(f"Model is not loaded: {health}")
    if health.get("api_key"):
        raise RuntimeError("Authentication differs from the requested unauthenticated service")
    if health.get("max_context") != cfg["gpu"]["context_tokens"]:
        raise RuntimeError(f"Unexpected context: {health}")
    if bool(health.get("images")) != cfg["vision"]["enabled"]:
        raise RuntimeError("Vision state differs from the requested profile")
    with opener.open(base + "/v1/models", timeout=5) as response:
        models = json.load(response)
    ids = [item["id"] for item in models["data"]]
    if cfg["api"]["model_name"] not in ids:
        raise RuntimeError(f"Unexpected models: {ids}")
    with opener.open(base + "/v1/status", timeout=5) as response:
        status = json.load(response)
    concurrency = status["concurrency"]
    if concurrency["serving"] != cfg["gpu"]["parallel_requests"]:
        raise RuntimeError(f"Effective parallelism differs from requested profile: {concurrency}")
    return {"ready": True, "healthy": True, "accepting_requests": True, "busy": False,
            "health": health, "models": ids, "concurrency": concurrency}


def check():
    return preflight(config())


def listen_ports():
    cfg = config()
    return [(cfg["api"]["host"], cfg["api"]["port"])]


def build_launch():
    return launch_spec(config())


def probe():
    return ready(config())
