# Mode README template

Copy this outline to an operator mode's README. Keep unknowns explicit. Publish sanitized reusable versions under examples/, not a live deployment directory.

## Purpose and status

- Mode ID, integration pattern and intended callers.
- Framework/native engine, source commit, package version and dependency pins.
- Prepared / API-ready / model-loaded / inference-tested: state each scope separately.

## Recipe

- All models/components, existing local weight paths, quantization and required derived artifacts.
- GPU roles or PP/TP, mapped to machine.toml GPU UUID order.
- Context, concurrency/headroom, workflows and required patches.

## Explicit preparation

- Mode-local interpreter/.venv, source/native bundle setup, dependency versions and verification.
- Any model preparation/download must be explicit and separately authorized.
- No installation, compilation, model fetching or inference in normal status/switch.

## Shared lifecycle

Configure [control], then run `mgr register <id>` after preparing the interpreter. The five lifecycle entries must link to the shared project script; do not copy their implementation.

Use `start.sh` / `mgr switch <id>`, `status.sh`, `stop.sh`, `force-stop.sh`; internal launch.sh is not a public foreground bypass. Describe check/listen_ports/build_launch/probe and optional recovery behavior according to docs/controller.md.

## API, data and limits

- Bind addresses, ports, auth, expected model/protocol and compatibility requirements.
- Input/output/cache/log locations, retention and interrupted-job settlement with no replay.
- Readiness versus busy/acceptance semantics, without generation in status.
- Actual validation environment/date, tasks, results and known/unvalidated limits.
