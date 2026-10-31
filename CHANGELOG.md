# Changelog

## 0.1.0 — 2026-10-31

Initial public AetherSwitch source release.

- Shared mode switching, status, normal stop and immediate force-stop through user systemd cgroups.
- Target preflight, launch grants, operation locking, ownership checks, GPU release checks and read-only application probes.
- Sanitized examples for plain vLLM, a Strata native-engine recipe, and a Wan workflow-node adapter contract.
- Local hardware configuration, deployed modes, runtime evidence, environments and private business integrations excluded from Git.
- CPU-only fixture-based tests, publication audit, CI and bilingual documentation.
- MIT license; external frameworks, engines, workflows and model weights are not redistributed.

The existing internal `server-mgr-` unit prefix is retained for compatibility with already-running local deployments. Public branding does not rename or restart those services.
