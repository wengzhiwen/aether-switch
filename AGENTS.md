# AetherSwitch contributor rules

Read this file before changing the project. Read `docs/controller.md` and `templates/mode/README.md` before adding an adapter; read its README, configuration and version references before editing an example. Consult `docs/deployment-notes.md` for deployment work.

## Scope and ownership

- Shared lifecycle belongs to `mgr`, `server_manager/` and `scripts/`: locking, mount identity, unit/cgroup ownership, startup/stop/force-stop, resource handoff and status.
- Mode adapters own models, environments, PP/TP, component placement, workflows, API compatibility, authentication and readiness.
- One mode exclusively owns the configured GPU pool, even with no resident GPU model. Switching interrupts old work after the target preflight succeeds.
- Preserve inputs, results and logs. Never replay interrupted jobs or automatically install/download/build during normal lifecycle operations.
- Never stop unknown processes, use broad `pkill`, reset GPUs, or restart/restore a mode after boot automatically.

## Modes and public examples

- Active deployments live in ignored `modes/<mode-id>/`; real hardware configuration is ignored `machine.toml`. Do not overwrite a user's deployment with a public example.
- Publish portable, sanitized examples in `examples/`. Do not publish credentials, personal paths, real hardware UUIDs, network identities, local evidence, caches, model weights or private business integrations.
- Each mode needs README, mode.toml and an adapter implementing `check`, `listen_ports`, `build_launch`, `probe`, plus optional `recover_before_launch`.
- Generate start/status/stop/force-stop/launch links with `mgr register`; all must point to `../../scripts/mode-entry.sh`. Do not copy lifecycle implementations or create a second systemd supervisor in a mode.
- Keep Python environments under each mode's `.venv/`; keep weights on the configured model mount. Record pinned framework/engine versions and explicit preparation instructions. Do not vendor engines or private workflows.
- Keep all examples disabled until the operator configures and prepares them. Mark placeholders and missing application components explicitly.

## Validation and documentation

- Preflight must run before stopping the old mode. Startup failure cleans only the attempted, positively identified unit and preserves logs.
- Status and probes are read-only: no model loading or generation tasks. Distinguish healthy, accepting requests and busy.
- Tests must use temporary fixtures and run without operator configuration, models, GPU or a live user systemd manager.
- Update relevant READMEs and CHANGELOG with behavior/interface changes. Shared deployment lessons belong in `docs/deployment-notes.md`; model-specific preferences stay in that example.
- Distinguish prepared files, service readiness, model loading and actual inference. Do not describe a sanitized example or CPU mock as a fresh GPU deployment validation.
- Before publishing, review the Git index and run `scripts/audit-public.py`. It supplements human review; it is not a complete secret detector.
