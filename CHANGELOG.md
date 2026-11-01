# Changelog

## Unreleased

- Add a standalone sanitized Strata deployment guide for Linux and two NVIDIA GPUs: pinned dependencies and model preparation, pipeline/MTP choices, historical benchmark methods and summaries, memory diagnostics, Claude Code integration and the shared lifecycle boundary.
- Link the guide from both repository READMEs, deployment lessons and the disabled Strata example. No operator deployment, raw runtime evidence or application binaries are published.
- Add the companion sanitized Strata guide for Linux on GB10-class unified memory: the per-release aarch64 engine matrix, single-GPU recipe deltas (mmap experts, PLE table in RAM, 32K prefill chunks, MTP q4/hnorm, 8 serving slots), unified-memory accounting with the controller's compute-app fallback, adapter pitfalls and historical acceptance summaries.
- Link the GB10 guide from both repository READMEs, deployment lessons and the disabled Strata example; it documents the same public example's adapter contract with a machine-specific recipe, not a new GPU validation.

## 0.2.0 — 2026-11-01

- Docker modes: `[control] runtime = "docker"` runs a foreground labeled container through the same launch grant; stopping stops the unit, removes the labeled container and verifies release. Container-scope GPU processes are managed attribution; declared `listen_ports()` authorize port reuse by an old docker mode.
- Unified-memory GPU support: `[N/A]` per-GPU accounting falls back to compute-app allocation sums plus operator-declared `memory_total_mib`.
- Unit ownership compares `TimeoutStopSec` as a parsed timespan instead of a formatted string.

## 0.1.0 — 2026-10-31

Initial public AetherSwitch source release.

- Shared mode switching, status, normal stop and immediate force-stop through user systemd cgroups.
- Target preflight, launch grants, operation locking, ownership checks, GPU release checks and read-only application probes.
- Sanitized examples for plain vLLM, a Strata native-engine recipe, and a Wan workflow-node adapter contract.
- Local hardware configuration, deployed modes, runtime evidence, environments and private business integrations excluded from Git.
- CPU-only fixture-based tests, publication audit, CI and bilingual documentation.
- MIT license; external frameworks, engines, workflows and model weights are not redistributed.

The existing internal `server-mgr-` unit prefix is retained for compatibility with already-running local deployments. Public branding does not rename or restart those services.
