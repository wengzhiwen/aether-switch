"""One lifecycle, backed by user systemd cgroups. State files are evidence only."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import time
import tomllib

LINKS = ("start", "status", "stop", "force-stop", "launch")
LIVE = {"active", "activating", "deactivating", "reloading"}
PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup",
              "InvocationID", "ExecStart", "Transient", "KillMode", "TimeoutStopUSec",
              "Restart", "Result", "ExecMainCode", "ExecMainStatus")


class ManagerError(RuntimeError):
    pass


def run(argv, timeout=30, check=True, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kwargs)
    if check and result.returncode:
        raise ManagerError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    return result


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def pid_identity(pid):
    try:
        # comm can contain spaces and parentheses; starttime is field 22.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def pid_group(pid):
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return line[3:]
    except (FileNotFoundError, ProcessLookupError):
        pass
    return ""


def in_group(group, parent):
    return bool(parent) and (group == parent or group.startswith(parent + "/"))


def cgroup_pids(group):
    directory = Path("/sys/fs/cgroup" + group)
    if not group or not directory.exists():
        return []
    values = set()
    for path in directory.rglob("cgroup.procs"):
        try:
            values.update(int(pid) for pid in path.read_text().split())
        except FileNotFoundError:
            pass
    return sorted(values)


def cgroup_empty(group):
    if not group:
        return True
    try:
        return "populated 0" in Path("/sys/fs/cgroup" + group, "cgroup.events").read_text()
    except FileNotFoundError:
        return True


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Mode:
    def __init__(self, root, identifier):
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identifier):
            raise ManagerError("Invalid mode ID")
        self.id = identifier
        self.directory = root / "modes" / identifier
        self.cfg = tomllib.loads((self.directory / "mode.toml").read_text())
        if self.cfg.get("id") != identifier:
            raise ManagerError("mode.toml ID must equal directory name: " + identifier)
        self.control = self.cfg.get("control", {})
        if self.control.get("version") != 1:
            raise ManagerError("Missing supported [control] contract: " + identifier)
        self.unit = "server-mgr-" + identifier + ".service"
        self.runtime = root / ".runtime" / identifier
        self.python = self.directory / self.control["interpreter"]
        self.adapter = self.control["adapter"]
        if not self.python.is_absolute() or not self.python.is_relative_to(self.directory):
            raise ManagerError("interpreter must be inside the mode")
        if not (self.directory / self.adapter).resolve().is_relative_to(self.directory.resolve()):
            raise ManagerError("adapter must be inside the mode")


class Systemd:
    def show(self, mode):
        argv = ["systemctl", "--user", "show", mode.unit]
        for prop in PROPERTIES:
            argv += ["-p", prop]
        result = run(argv, check=False)
        props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if not props:
            raise ManagerError("Cannot query user systemd: " + result.stderr.strip())
        return props

    def live_names(self):
        output = run(["systemctl", "--user", "list-units", "server-mgr-*.service", "--all",
                      "--plain", "--no-legend", "--no-pager"]).stdout
        return {parts[0] for line in output.splitlines() if len(parts := line.split()) >= 4 and parts[2] in LIVE}

    def start(self, mode, nonce, stop_seconds):
        mode.runtime.mkdir(parents=True, exist_ok=True)
        logfile = mode.runtime / "server.log"
        # Keep failed units until an explicit stop/reset so exit evidence survives.
        run(["systemctl", "--user", "reset-failed", mode.unit], check=False)
        return run(["systemd-run", "--user", "--quiet", "--unit=" + mode.unit,
                    "--description=server_mgr mode " + mode.id,
                    "--property=Type=exec", "--property=KillMode=control-group",
                    f"--property=TimeoutStopSec={stop_seconds}s", "--property=Restart=no",
                    "--property=SendSIGKILL=yes", "--property=Delegate=no", "--property=UMask=0077",
                    "--property=StandardOutput=append:" + str(logfile),
                    "--property=StandardError=append:" + str(logfile),
                    "--setenv=SERVER_MGR_LAUNCH_TOKEN=" + nonce, str(mode.directory / "launch.sh")])

    def stop(self, mode, force):
        if force:
            result = run(["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", mode.unit], check=False)
            # A unit may finish between show and kill. A populated cgroup still fails below.
            if result.returncode and self.show(mode).get("ActiveState") in LIVE:
                raise ManagerError(result.stderr.strip())
        run(["systemctl", "--user", "stop", mode.unit], timeout=30, check=False)


class Controller:
    def __init__(self, root, backend=None):
        self.root = Path(root).resolve()
        try:
            self.machine = tomllib.loads((self.root / "machine.toml").read_text())
        except FileNotFoundError as error:
            raise ManagerError("Create machine.toml from examples/machine.toml and configure your storage/GPU UUIDs") from error
        self.state = self.root / ".runtime/controller"
        self.backend = backend or Systemd()
        self.operation = None

    def modes(self):
        return [Mode(self.root, p.parent.name) for p in sorted((self.root / "modes").glob("*/mode.toml"))]

    def mode(self, value):
        identifier = self.machine.get("aliases", {}).get(value, value)
        try:
            return Mode(self.root, identifier)
        except FileNotFoundError as error:
            raise ManagerError("Unknown mode: " + str(value)) from error

    def registered(self, mode):
        for name in LINKS:
            path = mode.directory / (name + ".sh")
            if not path.is_symlink() or path.resolve() != self.root / "scripts/mode-entry.sh":
                raise ManagerError(f"Lifecycle entry must be shared symlink: {path}; use mgr register")
        if not mode.python.is_file() or not os.access(mode.python, os.X_OK):
            raise ManagerError("Mode interpreter missing; explicitly prepare first: " + str(mode.python))

    def register(self, mode):
        # Never overwrite an independent lifecycle script silently.
        for name in LINKS:
            path = mode.directory / (name + ".sh")
            if path.exists() and (not path.is_symlink() or path.resolve() != self.root / "scripts/mode-entry.sh"):
                raise ManagerError("Back up/remove the independent lifecycle script first: " + str(path))
        for name in LINKS:
            path = mode.directory / (name + ".sh")
            if not path.is_symlink():
                path.symlink_to("../../scripts/mode-entry.sh")
        self.registered(mode)
        return {"mode": mode.id, "registered": True, "shared_entry": "scripts/mode-entry.sh"}

    def note(self, stage, **details):
        if self.operation is None:
            return
        self.operation.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **details)
        write_json(self.state / "operation.json", self.operation)
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(self.operation, ensure_ascii=False) + "\n")
        print(f"[{stage}] {details.get('mode', self.operation.get('target', ''))}", file=sys.stderr, flush=True)

    @contextmanager
    def lock(self, action, target):
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.state / "lock").open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ManagerError("Another lifecycle operation is running; use mgr status") from error
            self.operation = {"action": action, "target": target, "boot_id": boot_id(),
                              "pid": os.getpid(), "pid_start": pid_identity(os.getpid()),
                              "started_at": datetime.now(timezone.utc).isoformat()}
            self.note("begin")
            try:
                yield
            except BaseException as error:
                self.note("failed", error=str(error) or type(error).__name__)
                raise
            else:
                self.note("complete")
            finally:
                self.operation = None

    def operation_status(self):
        try:
            record = json.loads((self.state / "operation.json").read_text())
        except (FileNotFoundError, ValueError):
            return None
        record["in_progress"] = (record.get("stage") not in {"complete", "failed"} and
                                  record.get("boot_id") == boot_id() and
                                  pid_identity(record.get("pid")) == record.get("pid_start") and
                                  record.get("pid_start") is not None)
        return record

    def ownership(self, mode, props):
        if props.get("LoadState") == "not-found":
            return
        expected = f"path={mode.directory / 'launch.sh'} ;"
        valid = (props.get("Id") == mode.unit and props.get("Transient") == "yes" and
                 props.get("KillMode") == "control-group" and props.get("Restart") == "no" and
                 props.get("TimeoutStopUSec") == f"{self.machine['lifecycle']['stop_timeout_seconds']}s" and
                 expected in props.get("ExecStart", ""))
        group = props.get("ControlGroup", "")
        if group:
            valid = valid and group.endswith("/" + mode.unit) and f"/user@{os.getuid()}.service/" in group
        if props.get("ActiveState") in LIVE:
            valid = valid and bool(group)
        if not valid:
            raise ManagerError("Unit ownership/properties cannot be confirmed; inspect without killing: " + mode.unit)
        pid = int(props.get("MainPID", 0))
        if pid and pid_identity(pid) and not in_group(pid_group(pid), group):
            raise ManagerError("MainPID is outside its expected cgroup: " + mode.unit)

    def inventory(self):
        pairs = [(mode, self.backend.show(mode)) for mode in self.modes()]
        unknown = self.backend.live_names() - {mode.unit for mode, _ in pairs}
        if unknown:
            raise ManagerError("Unregistered server_mgr units; inspect without killing: " + ", ".join(sorted(unknown)))
        live = []
        for mode, props in pairs:
            if props.get("ActiveState") in LIVE or not cgroup_empty(props.get("ControlGroup", "")):
                self.ownership(mode, props)
                live.append((mode, props))
        if len(live) > 1:
            raise ManagerError("Multiple modes are running; stop an explicitly identified mode before switching")
        return live

    def devices(self):
        result = run(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=15)
        found = {}
        for line in result.stdout.splitlines():
            uuid, used, total = [item.strip() for item in line.split(",")]
            found[uuid] = {"memory_used_mib": int(used), "memory_total_mib": int(total)}
        required = self.machine["gpu"]["uuids"]
        if set(required) - found.keys():
            raise ManagerError("Required GPUs missing: " + str(set(required) - found.keys()))
        return {uuid: found[uuid] for uuid in required}

    def compute(self):
        result = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], timeout=15)
        found = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",", 2)]
            if len(parts) == 3 and parts[0] in self.machine["gpu"]["uuids"]:
                pid = int(parts[1])
                found.append({"gpu": parts[0], "pid": pid, "name": parts[2], "cgroup": pid_group(pid)})
        return found

    def machine_check(self):
        storage = self.machine["storage"]
        mounted = run(["findmnt", "-rn", "-M", storage["mountpoint"], "-o", "UUID"], timeout=10).stdout.strip()
        if mounted != storage["filesystem_uuid"]:
            raise ManagerError(f"Wrong model mount UUID: {mounted!r}")
        self.devices()

    def adapter(self, mode, action):
        timeout = mode.control.get("check_timeout_seconds", 300) if action == "check" else mode.control.get("probe_timeout_seconds", 35)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("SERVER_MGR_LAUNCH_TOKEN", None)
        if action == "check":
            env["CUDA_VISIBLE_DEVICES"] = ""
        result = run([str(mode.python), "-B", str(self.root / "scripts/adapter-runner.py"),
                      str(mode.directory), mode.adapter, action], timeout=timeout, env=env, cwd=mode.directory)
        try:
            value = json.loads(result.stdout)
        except ValueError as error:
            raise ManagerError("Adapter must emit one JSON object: " + mode.id) from error
        if not isinstance(value, dict):
            raise ManagerError("Invalid adapter response: " + mode.id)
        return value

    def check(self, mode):
        self.registered(mode)
        if not mode.cfg.get("enabled", True):
            raise ManagerError("Mode is disabled: " + mode.id)
        self.machine_check()
        result = self.adapter(mode, "check")
        self.preflight_ports(result["listen_ports"])
        return {"mode": mode.id, "check_passed": True, "details": result}

    def preflight_ports(self, ports):
        # Permit ports held by the positively identified old mode; a new mode can
        # legitimately reuse its endpoint. Unknown listeners must preserve the old mode.
        live = self.inventory()
        allowed_inodes = set()
        for _, props in live:
            for pid in cgroup_pids(props.get("ControlGroup", "")):
                try:
                    for fd in Path(f"/proc/{pid}/fd").iterdir():
                        try:
                            target = os.readlink(fd)
                            if target.startswith("socket:["):
                                allowed_inodes.add(target[8:-1])
                        except (FileNotFoundError, PermissionError):
                            pass
                except FileNotFoundError:
                    pass
        for host, port in ports:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind((host, int(port)))
                except OSError as error:
                    listeners = set()
                    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
                        for line in Path(table).read_text().splitlines()[1:]:
                            columns = line.split()
                            if columns[3] == "0A" and int(columns[1].split(":")[1], 16) == int(port):
                                listeners.add(columns[9])
                    if not listeners or not listeners.issubset(allowed_inodes):
                        raise ManagerError(f"Target port unavailable before stopping old mode: {host}:{port}: {error}") from error

    def status_one(self, mode, probe=True):
        props = self.backend.show(mode)
        report = {"mode": mode.id, "unit": mode.unit, "systemd": props,
                  "state": "stopped", "healthy": False, "accepting_requests": False, "busy": False}
        if props.get("ActiveState") in LIVE:
            try:
                self.ownership(mode, props)
            except ManagerError as error:
                report.update(state="conflict", error=str(error))
                return report
            if props.get("ActiveState") == "deactivating":
                report["state"] = "stopping"
            elif probe:
                try:
                    result = self.adapter(mode, "probe")
                    fresh = self.backend.show(mode)
                    if fresh.get("InvocationID") != props.get("InvocationID") or fresh.get("ActiveState") not in {"active", "activating"}:
                        raise ManagerError("Unit changed while probing")
                    healthy = result.get("healthy") is True
                    report.update(healthy=healthy, accepting_requests=result.get("accepting_requests") is True,
                                  busy=result.get("busy") is True, probe=result,
                                  state=("busy" if result.get("busy") else "ready") if healthy else "unready")
                except (ManagerError, OSError, subprocess.SubprocessError) as error:
                    operation = self.operation_status()
                    loading = operation and operation["in_progress"] and operation.get("target") == mode.id and operation.get("stage") in {"starting", "loading"}
                    report.update(state="loading" if loading else "unready", error=str(error)[-2000:])
            else:
                report["state"] = "running"
        elif props.get("ActiveState") == "failed":
            report["state"] = "failed"
        elif not cgroup_empty(props.get("ControlGroup", "")):
            report["state"] = "conflict"
        return report

    def status(self, mode=None):
        modes = [mode] if mode else self.modes()
        reports = [self.status_one(item) for item in modes]
        active = [item for item in reports if item["systemd"].get("ActiveState") in LIVE]
        processes = self.compute()
        conflicts = []
        if len(active) > 1:
            conflicts.append("multiple managed modes are running")
        unknown_units = self.backend.live_names() - {item.unit for item in self.modes()}
        if unknown_units:
            conflicts.append("unregistered units: " + ", ".join(sorted(unknown_units)))
        if mode is None:
            groups = [item["systemd"].get("ControlGroup", "") for item in active if item["state"] != "conflict"]
            if any(not any(in_group(p["cgroup"], group) for group in groups) for p in processes):
                conflicts.append("unmanaged GPU compute processes")
        return {"modes": reports, "operation": self.operation_status(), "conflicts": conflicts,
                "current_mode": active[0]["mode"] if len(active) == 1 else None,
                "gpus": self.devices(), "compute_processes": processes}

    def assert_no_unknown_compute(self, live):
        groups = [props.get("ControlGroup", "") for _, props in live]
        conflicts = [item for item in self.compute() if not any(in_group(item["cgroup"], group) for group in groups)]
        if conflicts:
            raise ManagerError("Unmanaged GPU compute processes; no processes stopped: " + json.dumps(conflicts))

    def wait_idle(self):
        deadline = time.monotonic() + self.machine["gpu"]["release_timeout_seconds"]
        while True:
            processes, devices = self.compute(), self.devices()
            if not processes and all(gpu["memory_used_mib"] <= self.machine["gpu"]["released_memory_max_mib"] for gpu in devices.values()):
                return {"compute_processes": [], "gpus": devices}
            if time.monotonic() >= deadline:
                raise ManagerError("GPU release not confirmed; no next mode started: " + json.dumps({"processes": processes, "gpus": devices}))
            time.sleep(0.5)

    def stop_one(self, mode, force=False, expected=None):
        props = self.backend.show(mode)
        if props.get("LoadState") == "not-found" or (props.get("ActiveState") not in LIVE and cgroup_empty(props.get("ControlGroup", ""))):
            return {"mode": mode.id, "stopped": True, "already_stopped": True}
        self.ownership(mode, props)
        if expected and props.get("InvocationID") != expected:
            raise ManagerError("Unit invocation changed; refusing to stop: " + mode.id)
        group = props.get("ControlGroup", "")
        identities = {pid: pid_identity(pid) for pid in cgroup_pids(group)}
        self.note("force-stopping" if force else "stopping", mode=mode.id)
        self.backend.stop(mode, force)
        deadline = time.monotonic() + self.machine["gpu"]["release_timeout_seconds"]
        while True:
            fresh = self.backend.show(mode)
            if fresh.get("ActiveState") in LIVE and fresh.get("InvocationID") != props.get("InvocationID"):
                raise ManagerError("A new invocation appeared during cleanup; inspect without killing")
            same_pids = [pid for pid, identity in identities.items() if identity and pid_identity(pid) == identity]
            gpu_pids = [item for item in self.compute() if item["pid"] in identities or in_group(item["cgroup"], group)]
            if fresh.get("ActiveState") not in LIVE and cgroup_empty(group) and not same_pids and not gpu_pids:
                break
            if time.monotonic() >= deadline:
                raise ManagerError("Owned processes have not exited: " + json.dumps({"pids": same_pids, "gpu": gpu_pids}))
            time.sleep(0.5)
        proof = {"mode": mode.id, "stopped": True, "forced": force, "boot_id": boot_id(),
                 "invocation_id": props.get("InvocationID"), "cgroup": group, "cgroup_empty": True,
                 "process_identities": identities, "result": fresh.get("Result"),
                 "exit_status": fresh.get("ExecMainStatus")}
        write_json(mode.runtime / "last-stop.json", proof)
        return proof

    def switch(self, mode, restart=False):
        with self.lock("restart" if restart else "switch", mode.id):
            self.registered(mode)
            live = self.inventory()
            self.assert_no_unknown_compute(live)
            if not restart and live and live[0][0].id == mode.id:
                current = self.status_one(mode)
                if current["healthy"]:
                    return {"unchanged": True, **current}
            self.note("checking", mode=mode.id)
            self.check(mode)  # No old-mode mutation until this succeeds.
            if live:
                old, props = live[0]
                self.stop_one(old, expected=props.get("InvocationID"))
            self.note("releasing", mode=mode.id)
            self.wait_idle()
            nonce = secrets.token_hex(32)
            grant = self.state / "grants" / (mode.id + ".json")
            write_json(grant, {"boot_id": boot_id(), "token": nonce})
            self.note("starting", mode=mode.id)
            attempted = False
            invocation = None
            try:
                attempted = True
                self.backend.start(mode, nonce, self.machine["lifecycle"]["stop_timeout_seconds"])
                deadline = time.monotonic() + mode.control.get("startup_timeout_seconds", 900)
                announced = time.monotonic()
                while time.monotonic() < deadline:
                    props = self.backend.show(mode)
                    invocation = invocation or props.get("InvocationID")
                    if props.get("ActiveState") not in {"active", "activating"}:
                        raise ManagerError("Mode exited before readiness: " + json.dumps(props))
                    current = self.status_one(mode)
                    if current["healthy"]:
                        self.note("ready", mode=mode.id, invocation_id=props.get("InvocationID"))
                        return {"unchanged": False, **current}
                    if time.monotonic() - announced >= 25:
                        self.note("loading", mode=mode.id)
                        announced = time.monotonic()
                    time.sleep(2)
                raise ManagerError("Startup timed out: " + current.get("error", mode.id))
            except BaseException as error:
                self.note("startup-failed", mode=mode.id, error=str(error) or type(error).__name__,
                          unit_snapshot=self.backend.show(mode))
                if attempted:
                    try:
                        self.stop_one(mode, force=True, expected=invocation)
                        self.wait_idle()
                    except BaseException as cleanup:
                        raise ManagerError(f"Startup failed: {error}; cleanup also failed: {cleanup}") from cleanup
                raise
            finally:
                grant.unlink(missing_ok=True)

    def stop(self, mode=None, force=False):
        with self.lock("force-stop" if force else "stop", mode.id if mode else "current"):
            if mode is None:
                live = self.inventory()
                if not live:
                    return {"stopped": True, "already_stopped": True}
                mode = live[0][0]
            result = self.stop_one(mode, force=force)
            # Other unknown processes are never killed; report that resources remain occupied.
            if not result.get("already_stopped"):
                result["resource_release"] = self.wait_idle()
            return result

    def internal_run(self, mode):
        self.registered(mode)
        props = self.backend.show(mode)
        self.ownership(mode, props)
        own_group = pid_group(os.getpid())
        if not in_group(own_group, props.get("ControlGroup", "")) or not own_group.endswith("/" + mode.unit):
            raise ManagerError("launch.sh is internal; use start.sh or mgr switch")
        grant = self.state / "grants" / (mode.id + ".json")
        try:
            value = json.loads(grant.read_text())
        except FileNotFoundError as error:
            raise ManagerError("No controller launch grant; use start.sh or mgr switch") from error
        token = os.environ.pop("SERVER_MGR_LAUNCH_TOKEN", "")
        if value.get("boot_id") != boot_id() or not token or not secrets.compare_digest(token, value.get("token", "")):
            raise ManagerError("Invalid controller launch grant")
        grant.unlink()
        live = self.inventory()
        if any(other.id != mode.id for other, _ in live):
            raise ManagerError("Another mode owns the GPUs")
        self.machine_check()
        if self.compute():
            raise ManagerError("GPU processes appeared before exec; inspect without killing")
        argv = [str(mode.python), "-B", str(self.root / "scripts/adapter-runner.py"),
                str(mode.directory), mode.adapter, "run"]
        os.execve(argv[0], argv, dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
