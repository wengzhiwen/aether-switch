"""Workflow-node adapter: an authenticated API owns all on-demand GPU workers.

This example defines the HTTP contract in README.md; it does not bundle a private
business server, job database, transport integration or a ComfyUI workflow.
"""
import json
from pathlib import Path
import urllib.error
import urllib.request

from server_manager.recipe import environment, get_json, load, local_path, require_files

ROOT = Path(__file__).resolve().parent


def token(cfg):
    path = local_path(ROOT, cfg["worker"]["token_file"])
    if path.stat().st_mode & 0o077:
        raise ValueError("API token file must be private (0600)")
    value = path.read_text().strip()
    if not value:
        raise ValueError("API token file is empty")
    return value


def check():
    cfg = load(ROOT)[2]
    paths = [local_path(ROOT, cfg["worker"][key]) for key in ("script", "config", "token_file")]
    paths += [Path(path) for path in cfg["models"]["required_files"]]
    require_files(paths)
    token(cfg)
    private = local_path(ROOT, cfg["worker"]["config"])
    if private.stat().st_mode & 0o077:
        raise ValueError("Worker config must be private (0600)")
    json.loads(private.read_text())
    return {"local_files_checked": len(paths), "model_loading": "on demand per request"}


def listen_ports():
    cfg = load(ROOT)[2]
    return [(cfg["api"]["host"], cfg["api"]["port"]),
            (cfg["api"]["comfy_internal_host"], cfg["api"]["comfy_internal_port"])]


def build_launch():
    _, _, cfg, machine = load(ROOT)
    argv = [str(ROOT / ".venv/bin/python"), "-B", str(local_path(ROOT, cfg["worker"]["script"])),
            "--config", str(local_path(ROOT, cfg["worker"]["config"]))]
    return {"argv": argv, "cwd": str(ROOT), "env": environment(ROOT, cfg, machine)}


def health_state(status, health, protocol):
    if status == 200 and health == {"ready": True, "protocol_version": protocol}:
        return False
    if status == 503 and health == {"ready": False, "protocol_version": protocol}:
        return True
    raise ValueError("Unexpected resource health response")


def probe():
    cfg = load(ROOT)[2]
    base = f"http://127.0.0.1:{cfg['api']['port']}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(base + "/health", timeout=5)
    except urllib.error.HTTPError as error:
        if error.code != 503:
            raise
        response = error
    with response:
        busy = health_state(response.code, json.load(response), cfg["api"]["protocol"])
    capabilities = get_json(base + "/capabilities", {"Authorization": "Bearer " + token(cfg)})
    expected = {"capability_id": cfg["api"]["capability"], "profile_version": cfg["api"]["profile_version"]}
    if capabilities.get("protocol_version") != cfg["api"]["protocol"] or expected not in capabilities.get("capabilities", []):
        raise ValueError("Expected protocol/capability is unavailable")
    return {"healthy": True, "accepting_requests": not busy, "busy": busy,
            "model_loading": "on demand per request"}
