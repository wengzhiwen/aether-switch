# Contributing

Keep lifecycle behavior in the shared controller. An engine-specific adapter should only validate its recipe, declare ports, return a launch specification and probe its own API. See [the adapter contract](docs/controller.md).

For a new integration, add a disabled, sanitized example under `examples/modes/` and describe preparation, pinned dependencies, model placement, API behavior and validation limits. Do not bundle model weights, application checkouts, private workflows or a ready-made secret.

Tests use temporary project fixtures and only the Python standard library:

```bash
python3 -B -m unittest discover -s tests -v
bash -n mgr scripts/mode-entry.sh
```

Review and stage only public files. Then run `python3 -B scripts/audit-public.py`. The command checks the Git index, so it also works before an initial commit. Local `machine.toml`, `modes/`, runtime records and `.local/` must remain ignored.

For lifecycle changes, cover meaningful failure cases: target preflight preserves the old service, unknown processes are never killed, failed startup cleans only the target, operation locks cannot overlap, and normal status never starts work. Do not run GPU inference automatically in CI.

Report the exact environment and scope of any GPU validation. Original deployments do not prove that changed examples work unchanged on another workstation.

Contributions to this repository are submitted under its [MIT license](LICENSE). Refer to upstream licenses when integrating external frameworks; this repository does not grant rights to their code or weights.
