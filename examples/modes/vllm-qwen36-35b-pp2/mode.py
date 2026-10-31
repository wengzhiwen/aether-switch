"""Plain vLLM adapter: one command, local weights, OpenAI-compatible readiness."""
import json
from pathlib import Path
import sysconfig

from server_manager.recipe import environment, get_json, load, require_files, require_versions

ROOT = Path(__file__).resolve().parent


def check():
    _, _, cfg, machine = load(ROOT)
    base = Path(cfg["model"]["path"])
    if not base.resolve().is_relative_to(Path(machine["storage"]["mountpoint"]).resolve()):
        raise ValueError("Model path must be inside the configured model mount")
    require_files([base / name for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                                           "model.safetensors.index.json"]])
    model = json.loads((base / "config.json").read_text())
    if model.get("model_type") != cfg["model"]["expected_model_type"]:
        raise ValueError("Unexpected model architecture")
    if model.get("quantization_config", {}).get("quant_method") != "fp8":
        raise ValueError("Expected local FP8 weights")
    shards = set(json.loads((base / "model.safetensors.index.json").read_text())["weight_map"].values())
    for shard in shards:
        path = base / shard
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError("Weight shard escapes its checkpoint directory")
        require_files([path])
        if path.stat().st_size == 0:
            raise ValueError("Empty weight shard")
    require_files([ROOT / ".venv/bin/vllm"])
    require_versions({"vllm": "0.32.0"})
    if cfg["vllm"]["pipeline_parallel_size"] != len(machine["gpu"]["uuids"]):
        raise ValueError("This recipe assigns one pipeline stage to each configured GPU")
    return {"local_shards": len(shards), "scope": "files/configuration; no model loaded"}


def listen_ports():
    cfg = load(ROOT)[2]
    return [(cfg["vllm"]["host"], cfg["vllm"]["port"])]


def build_launch():
    _, project, cfg, machine = load(ROOT)
    env = environment(ROOT, cfg, machine)
    env.pop("VLLM_API_KEY", None)
    env["VLLM_CACHE_ROOT"] = str(project / ".runtime" / cfg["id"] / "cache/vllm")
    # The historical cu130 recipe uses the mode-local pip CUDA SDK, if installed.
    cuda = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    if (cuda / "bin/nvcc").is_file():
        env.update(CUDA_HOME=str(cuda), CUDA_PATH=str(cuda), PATH=str(cuda / "bin") + ":" + env["PATH"])
    argv = [str(ROOT / ".venv/bin/vllm"), "serve", cfg["model"]["path"]]
    for key, value in cfg["vllm"].items():
        flag = "--" + key.replace("_", "-")
        argv += [flag if value else "--no-" + key.replace("_", "-")] if isinstance(value, bool) else [flag, str(value)]
    return {"argv": argv, "cwd": str(ROOT), "env": env}


def probe():
    cfg = load(ROOT)[2]
    base = f"http://127.0.0.1:{cfg['vllm']['port']}"
    # /health is an empty 200 response in this vLLM release.
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/health", timeout=5):
        pass
    ids = [item["id"] for item in get_json(base + "/v1/models")["data"]]
    if cfg["vllm"]["served_model_name"] not in ids:
        raise ValueError("Unexpected served model")
    return {"healthy": True, "accepting_requests": True, "busy": False, "models": ids}
