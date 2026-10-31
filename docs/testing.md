# Validation scope

## Public source tests

```bash
python3 -B -m unittest discover -s tests -v
bash -n mgr scripts/mode-entry.sh
python3 -B scripts/audit-public.py
```

Unit tests create temporary projects from public example files, with disposable mode interpreters and lifecycle links. They do not read the operator's machine.toml/modes, load models, query real GPUs, start systemd services or submit generation jobs.

Coverage includes target-check failure preserving an old mode, unknown GPU/port conflicts, same-mode idempotence, switch order, launch grants, failed start and CLI interruption cleanup, invocation/ownership checks, force-stop routing, lock competition, stale boot records, status during a lock, disabled examples, recipe construction and the workflow busy/health contract. These tests do not prove driver behavior or actual GPU inference.

GitHub Actions runs the suite on Python 3.11, 3.12 and 3.14. Local checks and configured CI are distinct from a completed remote CI result.

## Original deployment provenance

The controller grew from one Ubuntu 26.04.1 / glibc 2.43 workstation, with two RTX PRO 4000 Blackwell 24 GB GPUs, driver 595.99.02, systemd 259 and mode-local Python 3.12.16 environments.

On that deployment, the shared controller completed Wan → vLLM → Strata → Wan switching, API probes, vLLM short code/tool/stream/concurrency requests, Strata short OpenAI/Anthropic requests, a target-port conflict preserving the old InvocationID, and workflow interruption/restart without replay. A running four-process Wan group was force-stopped and GPU release confirmed in about 0.66 seconds. The workflow's earlier private media test produced and decoded a five-second RGBA WebM.

These are historical, environment-specific observations, not a benchmark guarantee. No personal paths, device UUIDs, request IDs, tokens, raw logs, videos or private business transport data are redistributed. The public example adapters are adapted/simplified versions and have not each been newly certified by GPU generation after sanitization.

## Release review

Before publication, examine the tracked file list, run the index audit, and export/clone only tracked files to rerun tests without local deployment files. Review doc links, Python syntax and example placeholders. Audit the committed tree as well as the staging index. The scanner catches common mistakes but does not replace review or constitute a complete secret detector.
