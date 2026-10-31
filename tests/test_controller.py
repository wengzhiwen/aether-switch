"""CPU-only failure and ownership tests. No GPU model or real unit is started."""
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from server_manager.core import Controller, ManagerError, boot_id, in_group, pid_identity, write_json

ROOT = Path(__file__).resolve().parent.parent


def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def make_project(directory):
    project = Path(directory)
    shutil.copy2(ROOT / "examples/machine.toml", project / "machine.toml")
    (project / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/mode-entry.sh", project / "scripts/mode-entry.sh")
    shutil.copytree(ROOT / "examples/modes", project / "modes")
    for mode in (project / "modes").iterdir():
        python = mode / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        for action in ["start", "status", "stop", "force-stop", "launch"]:
            (mode / (action + ".sh")).symlink_to("../../scripts/mode-entry.sh")
    return project


def add_docker_mode(project, identifier="docker-test"):
    mode = project / "modes" / identifier
    mode.mkdir()
    (mode / "mode.toml").write_text(
        f'id = "{identifier}"\nname = "docker test"\nenabled = true\n\n'
        "[control]\nversion = 1\nruntime = \"docker\"\ninterpreter = \".venv/bin/python\"\nadapter = \"mode.py\"\n")
    (mode / "mode.py").write_text(
        "def check():\n    return {}\n\n\ndef listen_ports():\n    return [('127.0.0.1', 18300)]\n\n\n"
        "def build_launch():\n    raise NotImplementedError\n\n\ndef probe():\n"
        "    return {'healthy': True, 'accepting_requests': True, 'busy': False}\n")
    python = mode / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    for action in ["start", "status", "stop", "force-stop", "launch"]:
        (mode / (action + ".sh")).symlink_to("../../scripts/mode-entry.sh")
    return mode


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = make_project(self.temp.name)
        self.controller = Controller(self.project, backend=Mock())
        self.a = self.controller.mode("1")
        self.b = self.controller.mode("3")
        self.a.runtime = self.controller.state / "a"
        self.b.runtime = self.controller.state / "b"

    def live(self, mode):
        return {"Id": mode.unit, "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
                "Transient": "yes", "KillMode": "control-group", "Restart": "no", "TimeoutStopUSec": "5s",
                "ExecStart": f"{{ path={mode.directory / 'launch.sh'} ; argv[]=ignored ; }}",
                "InvocationID": "original", "MainPID": "0",
                "ControlGroup": f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{mode.unit}"}

    def test_failed_target_check_keeps_old_service(self):
        c = self.controller
        c.inventory = Mock(return_value=[(self.a, self.live(self.a))])
        c.assert_no_unknown_compute = Mock()
        c.check = Mock(side_effect=ManagerError("missing model"))
        c.stop_one = Mock()
        with self.assertRaisesRegex(ManagerError, "missing model"):
            c.switch(self.b)
        c.stop_one.assert_not_called()
        c.backend.start.assert_not_called()

    def test_unknown_gpu_is_rejected_before_any_stop(self):
        c = self.controller
        c.inventory = Mock(return_value=[(self.a, self.live(self.a))])
        c.compute = Mock(return_value=[{"cgroup": "/unrelated", "pid": 123, "gpu": "test"}])
        c.stop_one = Mock()
        with self.assertRaisesRegex(ManagerError, "Unmanaged GPU"):
            c.switch(self.b)
        c.stop_one.assert_not_called()

    def test_healthy_same_mode_is_noop_including_when_busy(self):
        c = self.controller
        c.inventory = Mock(return_value=[(self.b, self.live(self.b))])
        c.assert_no_unknown_compute = Mock()
        c.status_one = Mock(return_value={"mode": self.b.id, "state": "busy", "healthy": True})
        c.check = Mock()
        self.assertTrue(c.switch(self.b)["unchanged"])
        c.check.assert_not_called()
        c.backend.start.assert_not_called()

    def test_switch_order_and_launch_grant(self):
        c = self.controller
        order = []
        c.inventory = Mock(return_value=[(self.a, self.live(self.a))])
        c.assert_no_unknown_compute = Mock()
        c.check = Mock(side_effect=lambda _: order.append("check"))
        c.stop_one = Mock(side_effect=lambda *_, **__: order.append("stop"))
        c.wait_idle = Mock(side_effect=lambda: order.append("idle"))
        def start(*_):
            self.assertTrue((c.state / "grants" / (self.b.id + ".json")).exists())
            order.append("start")
        c.backend.start.side_effect = start
        c.backend.show.return_value = self.live(self.b)
        c.status_one = Mock(return_value={"mode": self.b.id, "state": "ready", "healthy": True})
        c.switch(self.b)
        self.assertEqual(order, ["check", "stop", "idle", "start"])
        self.assertFalse((c.state / "grants" / (self.b.id + ".json")).exists())

    def test_failed_start_cleans_target_without_restarting_old(self):
        c = self.controller
        c.inventory = Mock(return_value=[])
        c.assert_no_unknown_compute = Mock()
        c.check = Mock()
        c.wait_idle = Mock()
        c.stop_one = Mock()
        c.backend.start.side_effect = ManagerError("exec failure")
        c.backend.show.return_value = {"ActiveState": "failed"}
        with self.assertRaisesRegex(ManagerError, "exec failure"):
            c.switch(self.b)
        c.stop_one.assert_called_once_with(self.b, force=True, expected=None)
        self.assertEqual(c.backend.start.call_count, 1)

    def test_cli_interruption_during_start_cleans_target(self):
        c = self.controller
        c.inventory = Mock(return_value=[])
        c.assert_no_unknown_compute = Mock()
        c.check = Mock()
        c.wait_idle = Mock()
        c.stop_one = Mock()
        c.backend.start.side_effect = KeyboardInterrupt()
        c.backend.show.return_value = {"ActiveState": "active"}
        with self.assertRaises(KeyboardInterrupt):
            c.switch(self.b)
        c.stop_one.assert_called_once()

    def test_invocation_change_refuses_stop(self):
        c = self.controller
        c.backend.show.return_value = self.live(self.a)
        with self.assertRaisesRegex(ManagerError, "invocation changed"):
            c.stop_one(self.a, expected="other-invocation")
        c.backend.stop.assert_not_called()

    def test_foreign_unit_name_is_insufficient_ownership(self):
        props = self.live(self.a)
        props["ExecStart"] = "{ path=/usr/bin/python3 ; }"
        with self.assertRaisesRegex(ManagerError, "ownership"):
            self.controller.ownership(self.a, props)

    def test_stop_an_inactive_mode_does_not_touch_another(self):
        c = self.controller
        c.backend.show.return_value = {"LoadState": "not-found", "ActiveState": "inactive"}
        c.wait_idle = Mock()
        result = c.stop(self.a)
        self.assertTrue(result["already_stopped"])
        c.backend.stop.assert_not_called()
        c.wait_idle.assert_not_called()

    def test_force_stop_is_forwarded_to_same_shared_cleanup(self):
        c = self.controller
        c.backend.show.side_effect = [self.live(self.b), {"ActiveState": "failed", "Result": "signal", "ExecMainStatus": "9"}]
        c.compute = Mock(return_value=[])
        with patch("server_manager.core.cgroup_pids", return_value=[]), patch("server_manager.core.cgroup_empty", return_value=True):
            result = c.stop_one(self.b, force=True)
        c.backend.stop.assert_called_once_with(self.b, True)
        self.assertTrue(result["forced"])
        self.assertEqual(result["exit_status"], "9")

    def test_mutation_lock_rejects_competing_client(self):
        other = Controller(self.project)
        other.state = self.controller.state
        with self.controller.lock("test", "a"):
            with self.assertRaisesRegex(ManagerError, "Another lifecycle"):
                with other.lock("test", "b"):
                    self.fail("must not enter")
            self.assertTrue(other.operation_status()["in_progress"])

    def test_stale_boot_record_is_not_authoritative(self):
        write_json(self.controller.state / "operation.json", {
            "stage": "starting", "boot_id": "previous-boot", "pid": os.getpid(),
            "pid_start": pid_identity(os.getpid()), "target": self.b.id})
        self.assertFalse(self.controller.operation_status()["in_progress"])

    def test_all_lifecycle_entries_are_shared_links(self):
        for mode in self.controller.modes():
            self.controller.registered(mode)

    def test_cgroup_descendant_match_has_path_boundary(self):
        self.assertTrue(in_group("/mode/worker", "/mode"))
        self.assertFalse(in_group("/mode-other", "/mode"))
        self.assertFalse(in_group("/anything", ""))

    def test_unknown_listener_fails_before_stopping_old_mode(self):
        self.controller.inventory = Mock(return_value=[])
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(ManagerError, "before stopping old mode"):
                self.controller.preflight_ports([("127.0.0.1", listener.getsockname()[1])])


class TimespanAndUnifiedMemoryTests(ControllerTests):
    def test_verbose_timespan_still_proves_unit_ownership(self):
        c = self.controller
        c.machine["lifecycle"]["stop_timeout_seconds"] = 90
        props = self.live(self.a)
        props["TimeoutStopUSec"] = "1min 30s"
        c.ownership(self.a, props)  # must not raise
        props["TimeoutStopUSec"] = "5s"
        with self.assertRaisesRegex(ManagerError, "ownership"):
            c.ownership(self.a, props)

    def test_na_memory_fields_fall_back_to_compute_app_sums(self):
        c = self.controller
        c.machine["gpu"]["uuids"] = ["GPU-X"]
        c.machine["gpu"]["memory_total_mib"] = 131072
        c.compute = Mock(return_value=[{"gpu": "GPU-X", "used_mib": 400}, {"gpu": "GPU-X", "used_mib": 100},
                                       {"gpu": "GPU-OTHER", "used_mib": 50}])
        with patch("server_manager.core.run") as nvidia:
            nvidia.return_value = Mock(stdout="GPU-X, [N/A], [N/A]\n")
            result = c.devices()
        self.assertEqual(result["GPU-X"]["memory_used_mib"], 500)
        self.assertEqual(result["GPU-X"]["memory_total_mib"], 131072)
        self.assertEqual(result["GPU-X"]["memory_source"], "compute-apps")


class DockerModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = make_project(self.temp.name)
        add_docker_mode(self.project)
        self.controller = Controller(self.project, backend=Mock())
        self.m = self.controller.mode("docker-test")
        self.m.runtime = self.controller.state / "m"

    def live(self, mode):
        return {"Id": mode.unit, "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
                "Transient": "yes", "KillMode": "control-group", "Restart": "no", "TimeoutStopUSec": "5s",
                "ExecStart": f"{{ path={mode.directory / 'launch.sh'} ; argv[]=ignored ; }}",
                "InvocationID": "original", "MainPID": "0",
                "ControlGroup": f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{mode.unit}"}

    def running_container(self):
        return {"id": "abc123def456", "running": True, "label_ok": True,
                "cgroup": "/system.slice/docker-abc123def456.scope"}

    def test_docker_stop_removes_container_and_verifies_release(self):
        c = self.controller
        c.backend.show.side_effect = [self.live(self.m), {"ActiveState": "failed", "Result": "signal", "ExecMainStatus": "137"}]
        c.backend.docker_state.side_effect = [self.running_container(), None, None]
        c.compute = Mock(return_value=[])
        with patch("server_manager.core.cgroup_pids", return_value=[]), patch("server_manager.core.cgroup_empty", return_value=True):
            result = c.stop_one(self.m)
        c.backend.stop.assert_called_once_with(self.m, False)
        c.backend.docker_remove.assert_called_once_with(self.m)
        self.assertTrue(result["container_removed"])
        self.assertEqual(result["exit_status"], "137")

    def test_foreign_container_is_never_touched(self):
        c = self.controller
        c.backend.show.return_value = self.live(self.m)
        foreign = self.running_container()
        foreign["label_ok"] = False
        c.backend.docker_state.return_value = foreign
        with self.assertRaisesRegex(ManagerError, "Foreign container"):
            c.stop_one(self.m)
        c.backend.docker_remove.assert_not_called()
        c.backend.stop.assert_not_called()

    def test_stale_container_without_unit_is_cleaned(self):
        c = self.controller
        c.backend.show.return_value = {"LoadState": "not-found", "ActiveState": "inactive"}
        c.backend.docker_state.side_effect = [self.running_container(), None]
        c.compute = Mock(return_value=[])
        result = c.stop_one(self.m)
        c.backend.stop.assert_not_called()
        c.backend.docker_remove.assert_called_once_with(self.m)
        self.assertTrue(result["stopped"])
        self.assertNotIn("already_stopped", result)

    def test_declared_port_of_old_docker_mode_is_reusable(self):
        c = self.controller
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            c.inventory = Mock(return_value=[(self.m, self.live(self.m))])
            c.adapter = Mock(return_value={"listen_ports": [("127.0.0.1", port)]})
            c.preflight_ports([("127.0.0.1", port)])  # owned by the identified old docker mode
            c.adapter = Mock(return_value={"listen_ports": [("127.0.0.1", port + 1)]})
            with self.assertRaisesRegex(ManagerError, "before stopping old mode"):
                c.preflight_ports([("127.0.0.1", port)])

    def test_container_compute_is_managed_only_with_matching_scope(self):
        c = self.controller
        live = [(self.m, self.live(self.m))]
        c.backend.docker_state.return_value = self.running_container()
        c.compute = Mock(return_value=[{"gpu": "GPU-X", "pid": 9, "used_mib": 1, "name": "vllm",
                                        "cgroup": "/system.slice/docker-abc123def456.scope/vllm"}])
        c.assert_no_unknown_compute(live)
        c.compute = Mock(return_value=[{"gpu": "GPU-X", "pid": 9, "used_mib": 1, "name": "other",
                                        "cgroup": "/system.slice/docker-OTHER.scope/worker"}])
        with self.assertRaisesRegex(ManagerError, "Unmanaged GPU"):
            c.assert_no_unknown_compute(live)

    def test_internal_run_refuses_stale_container(self):
        c = self.controller
        c.backend.show.return_value = self.live(self.m)
        c.backend.docker_state.return_value = self.running_container()
        with self.assertRaisesRegex(ManagerError, "Stale container"):
            c.internal_run(self.m)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = make_project(self.temp.name)

    def test_wan_only_documented_503_is_healthy_busy(self):
        wan = load_file("wan_test", self.project / "modes/comfyui-wan-animate2-rgba/mode.py")
        protocol = "media-resource-v1"
        self.assertTrue(wan.health_state(503, {"ready": False, "protocol_version": protocol}, protocol))
        self.assertFalse(wan.health_state(200, {"ready": True, "protocol_version": protocol}, protocol))
        for status, health in [(503, {}), (503, {"ready": False, "protocol_version": "foreign"}),
                               (200, {"ready": False, "protocol_version": protocol})]:
            with self.assertRaises(ValueError):
                wan.health_state(status, health, protocol)

    def test_shared_port_probe_rejects_live_listener_and_accepts_closed_port(self):
        runner = load_file("runner_test", ROOT / "scripts/adapter-runner.py")
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaises(OSError):
                runner.check_ports([("127.0.0.1", port)])
            with socket.create_connection(("127.0.0.1", port)) as client:
                server, _ = listener.accept()
                server.close()  # creates a real TIME_WAIT entry
        runner.check_ports([("127.0.0.1", port)])

    def test_strata_preflight_validates_trim_before_external_checks(self):
        strata = load_file("strata_test", self.project / "modes/strata-qwen38-flash-next/mode.py")
        cfg = strata.config()
        cfg["gpu"]["layer_split"] = "auto"
        with self.assertRaisesRegex(ValueError, "explicit layer_split"):
            strata.preflight(cfg)


if __name__ == "__main__":
    unittest.main()
