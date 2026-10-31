"""Public examples run against temporary fixtures, never an operator deployment."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_controller import ROOT, load_file, make_project
from server_manager.core import Controller, ManagerError


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = make_project(self.temp.name)

    def adapter(self, identifier):
        return load_file("example_" + identifier.replace("-", "_"), self.project / "modes" / identifier / "mode.py")

    def test_examples_cannot_start_without_explicit_enablement(self):
        controller = Controller(self.project)
        for mode in controller.modes():
            with self.assertRaisesRegex(ManagerError, "disabled"):
                controller.check(mode)

    def test_missing_local_machine_config_has_actionable_error(self):
        (self.project / "machine.toml").unlink()
        with self.assertRaisesRegex(ManagerError, "examples/machine.toml"):
            Controller(self.project)

    def test_vllm_build_preserves_sample_flags_and_explicit_gpu_order(self):
        adapter = self.adapter("vllm-qwen36-35b-pp2")
        result = adapter.build_launch()
        argv = result["argv"]
        self.assertEqual(argv[argv.index("--pipeline-parallel-size") + 1], "2")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "10")
        self.assertEqual(result["env"]["CUDA_VISIBLE_DEVICES"], "GPU-REPLACE-FIRST,GPU-REPLACE-SECOND")
        self.assertEqual(result["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(json.loads(argv[argv.index("--default-chat-template-kwargs") + 1]), {"enable_thinking": False})

    def test_strata_generates_project_relative_native_engine_config(self):
        adapter = self.adapter("strata-qwen38-flash-next")
        launch = adapter.build_launch()
        path = self.project / ".runtime/strata-qwen38-flash-next/server.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["parallel"], 4)
        self.assertEqual(config["layer_split"], "24")
        self.assertIn("--trim-stage-weights", config["args"])
        self.assertTrue(config["mtp"])
        self.assertIn("--mtp", config["args"])
        self.assertEqual(config["env"]["STRATA_START_RETRIES"], "0")
        self.assertEqual(launch["cwd"], str(self.project / "modes/strata-qwen38-flash-next/app"))

    def test_workflow_token_requires_local_private_file(self):
        adapter = self.adapter("comfyui-wan-animate2-rgba")
        cfg = adapter.load(adapter.ROOT)[2]
        path = adapter.ROOT / cfg["worker"]["token_file"]
        path.parent.mkdir()
        path.write_text("fixture-only-value")
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "0600"):
            adapter.token(cfg)
        path.chmod(0o600)
        self.assertEqual(adapter.token(cfg), "fixture-only-value")

    def test_cuda_helper_refuses_to_patch_outside_its_mode_environment(self):
        path = self.project / "modes/vllm-qwen36-35b-pp2/patches/cuda_glibc_compat.py"
        helper = load_file("cuda_helper_fixture", path)
        with patch.object(helper.sys, "prefix", str(self.project / "unrelated-python")):
            with self.assertRaisesRegex(RuntimeError, "mode-local"):
                helper.main()

    def test_workflow_probe_authenticates_and_reports_busy_without_submitting_work(self):
        self.probe_workflow(busy=True)

    def test_workflow_probe_reports_idle_without_submitting_work(self):
        self.probe_workflow(busy=False)

    def probe_workflow(self, busy):
        adapter = self.adapter("comfyui-wan-animate2-rgba")
        cfg = adapter.load(adapter.ROOT)[2]
        path = adapter.ROOT / cfg["worker"]["token_file"]
        path.parent.mkdir()
        path.write_text("fixture-only-value")
        path.chmod(0o600)
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                status = 200
                if self.path == "/health":
                    payload = {"ready": not busy, "protocol_version": cfg["api"]["protocol"]}
                    status = 503 if busy else 200
                else:
                    self.server.testcase.assertEqual(self.headers.get("Authorization"), "Bearer fixture-only-value")
                    payload = {"protocol_version": cfg["api"]["protocol"], "capabilities": [{
                        "capability_id": cfg["api"]["capability"], "profile_version": "v1"}]}
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.testcase = self
        cfg["api"]["port"] = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(adapter, "load", return_value=(adapter.ROOT, self.project, cfg, {})):
                result = adapter.probe()
            self.assertEqual(result["busy"], busy)
            self.assertEqual(result["accepting_requests"], not busy)
            self.assertTrue(result["healthy"])
            self.assertNotIn("fixture-only-value", json.dumps(result))
            self.assertEqual(requests, ["/health", "/capabilities"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_status_does_not_wait_for_operation_lock(self):
        controller = Controller(self.project)
        controller.status_one = lambda mode: {"mode": mode.id, "state": "stopped", "systemd": {}}
        controller.devices = lambda: {}
        controller.compute = lambda: []
        controller.backend.live_names = lambda: set()
        with controller.lock("switch", "test"):
            result = controller.status()
        self.assertEqual(len(result["modes"]), 3)
        self.assertTrue(result["operation"]["in_progress"])


if __name__ == "__main__":
    unittest.main()
