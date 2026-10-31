# Controller and adapter contract

The project entry point is `mgr` → `server_manager/`. The controller uses Python's standard library and user systemd transient services. There is no extra always-running controller daemon.

## Switch sequence

1. Take a nonblocking project operation lock. A competing mutation fails immediately; status/log queries do not wait for it.
2. Identify managed modes using unit properties, absolute ExecStart, user cgroup, InvocationID and process identities. Unknown units/GPU compute processes block a switch.
3. Check model mount UUID, required GPUs, target recipe/environment and listener conflicts. Ports positively owned by the old mode can be reused. Preflight failure leaves the old service running.
4. TERM the whole old cgroup; systemd KILLs leftovers after the shared timeout (default five seconds).
5. Verify the old cgroup is empty, original PID/starttime identities have exited, NVML compute PIDs are gone, and configured GPU memory release thresholds are met.
6. Create the target transient unit; a one-use launch grant authorizes the internal foreground entry. Wait for the mode-specific probe before reporting readiness.

`force-stop` immediately KILLs the owned cgroup, then uses the same release verification. A failed start or interrupted startup wait cleans only the attempted mode, preserves logs, and does not replay work or reload the old mode. Arbitrary admin commands, driver faults and power loss can bypass normal control; they are not repaired automatically.

Selecting a healthy current mode is a no-op even if it is busy. `restart` explicitly stops/reloads. A mode-specific stop only affects that mode; `mgr stop` selects the currently managed one.

## Deployment layout

```text
machine.toml                 # Operator hardware configuration; ignored
modes/<mode-id>/              # Operator application/environment; ignored
  README.md
  mode.toml
  mode.py                    # Adapter, configurable name
  .venv/                     # Mode-local Python
  start.sh       -> ../../scripts/mode-entry.sh
  status.sh      -> ../../scripts/mode-entry.sh
  stop.sh        -> ../../scripts/mode-entry.sh
  force-stop.sh  -> ../../scripts/mode-entry.sh
  launch.sh      -> ../../scripts/mode-entry.sh
.runtime/<mode-id>/           # Logs/state/cache/inputs/outputs; ignored
.runtime/controller/          # Shared operation evidence; ignored
```

Prepare the interpreter and run `mgr register <mode-id>` to create links. Registration refuses independent lifecycle scripts instead of silently replacing them. Start validates that all links still point to the shared implementation. `launch.sh` is internal: use `start.sh` or `mgr switch`, never another systemd supervisor.

A minimum control declaration in the mode's TOML:

```toml
id = "example-mode"
enabled = false

[control]
version = 1
interpreter = ".venv/bin/python"
adapter = "mode.py"
startup_timeout_seconds = 900
check_timeout_seconds = 300
probe_timeout_seconds = 35
```

## Docker modes

A mode may declare `[control] runtime = "docker"` when its service runs containerized. The adapter contract is unchanged: `build_launch()` returns a foreground `docker run --rm --name server-mgr-<mode-id> --label server_mgr.mode=<mode-id> …` argv, so container stdout streams into the unit log and the unit lifecycle follows the client. The mode-local interpreter still runs the adapter hooks on the host.

Differences from native modes, all in the controller:

- The container name and label are the ownership proof for the container; a foreign container with the mode's name is reported, never removed.
- Stopping stops the unit (TERM reaches the container through the client), then removes the labeled container (`docker rm -f`) and verifies the container is gone, original identities exited and GPU release exactly like a native mode. A stale labeled container is cleaned even when its unit already disappeared.
- Container processes live in a `system.slice/docker-<id>.scope` cgroup outside the user unit. GPU processes in that scope count as managed only while the identified mode is live, and a container that outlives its unit surfaces as a status conflict.
- Its listeners are root-owned, so inode-level port attribution is impossible from a user controller. Preflight instead reuses the ports the old docker mode itself declares through `listen_ports()`; unknown listeners still block a switch.

## Unified-memory GPUs

Some devices (for example GB10-class unified memory) answer `[N/A]` for per-GPU memory accounting. When that happens the controller sums `nvidia-smi --query-compute-apps` allocations per GPU and uses the operator-declared `[gpu] memory_total_mib` from `machine.toml` as the capacity; release checks then mean "no compute processes hold allocations". `TimeoutStopSec` unit properties are compared as parsed timespans, not strings, so distributions that render `90s` as `1min 30s` verify correctly.

Machine-wide stop/release policy comes from `machine.toml`. The existing internal unit prefix `server-mgr-<mode-id>.service` and `SERVER_MGR_LAUNCH_TOKEN` are retained for compatibility; public branding does not require renaming running units.

## Adapter hooks

`adapter-runner.py` imports the adapter with its own interpreter, in a separate process. Hook output must be a dictionary; the runner emits one JSON object. Do not print secrets or unrelated text to stdout.

| Hook | Responsibility |
| --- | --- |
| `check() -> dict` | Validate prepared files, pinned versions, frozen application resources and final parameter combinations before stopping the old mode. No CUDA initialization, preparation, generation or downloads. |
| `listen_ports() -> list` | Static `(host, port)` listeners, including request-scoped workers if needed. Used before stopping and again before exec. |
| `build_launch() -> dict` | Return `argv` (absolute executable first), absolute `cwd`, and a string-to-string `env`; writing this invocation's runtime config is allowed. No systemd management, killing or exec. |
| `probe() -> dict` | Read-only application/API checks; return boolean `healthy`, `accepting_requests`, `busy`. May include non-secret diagnostics. |
| `recover_before_launch()` | Optional application-owned settlement of positively identified interrupted jobs. Preserve original data; never replay. |

`healthy` means initialized and matching the intended protocol/configuration. `accepting_requests` means currently able to accept another job; `busy` reports active work if the application exposes it. The vLLM/Strata samples do not report per-request occupancy: `busy=false` there does not mean there is no active generation.

The shared runner handles port binding with SO_REUSEADDR, umask, chdir and execve. The controller handles locking, mount/device identity, ownership, cgroup stop/kill, timeouts and resource handoff. Never source a mode environment or activate its venv in the controller parent.

Optional `server_manager.recipe` helpers implement TOML loading, local paths, dependency/file checks, explicit cache environments and proxy-free local JSON HTTP requests. They do not know any model family or business protocol. They expect an installed `modes/<id>/` layout; copy examples there before using them.

## Status and persistence

- Actual unit/cgroup/API state is authoritative. `operation.json` is progress evidence with boot ID, PID and starttime; a stale record is not treated as a running operation.
- stdout/stderr append `.runtime/<id>/server.log`; engine/task logs stay mode-owned. `mgr logs <id> -f` follows the common log.
- Stop evidence is `.runtime/<id>/last-stop.json`; transitions append `.runtime/controller/events.jsonl`. Failed startup preserves a unit snapshot before cleanup.
- User systemd services depend on the user's manager. Decide separately whether linger is required after the last logout; AetherSwitch never changes linger or boot policy.
- Mount checks do not edit fstab, mount filesystems or alter DNS. Model files and GPU driver setup remain explicit operator work.
- The tool assumes a trusted operator and trusted adapter code. It is not an isolation boundary for untrusted tenants. API bind addresses and authentication are application decisions.
