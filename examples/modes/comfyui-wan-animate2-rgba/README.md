# Workflow node: Wan-Animate2 → RGBA WebM

This is the third pattern: AetherSwitch manages a **resource node**, while a larger application submits media work and handles results. The API stays running; each request launches GPU workers and releases models after its phases. Idle GPU memory does not end the mode's exclusive ownership.

## Application boundary

This repository provides the adapter and a sanitized recipe, **not** the resource server, private business workflow, transport bindings or job database. Supply `app/resource_api.py` (or adapt `build_launch`) and your own private JSON config. The worker owns its ComfyUI graph, queue, files, authentication, job recovery and outputs.

The illustrative `media-resource-v1` contract is:

- `GET /health`: HTTP 200 with `{"ready": true, "protocol_version": "media-resource-v1"}` when idle; HTTP 503 with the same protocol and `ready=false` when busy.
- Authenticated `GET /capabilities`: matching `protocol_version` and `capabilities=[{"capability_id": "wan-animate2-semantic720-rgb-v1", "profile_version": "v1"}]`.
- Bearer token read by the adapter from `secrets/api-token`, mode 0600. Your worker's private config must use the same token and be 0600 too.

`probe()` maps only this exact, authenticated contract's busy response to healthy/busy. An arbitrary HTTP 503 is not considered healthy. Adapt the probe to your real application rather than expecting this sample contract to exist in stock ComfyUI.

All workers must remain in the mode's systemd cgroup, including workers creating new process sessions. If your server needs interrupted-job recovery, add a mode-owned `recover_before_launch()` that confirms identities, records failure and preserves original inputs/results. Never automatically replay interrupted jobs.

## Reference media recipe

RGBA character PNG + RGB motion video + appearance/motion text → Wan-Animate2 distill INT8 ConvRot → RGB ×4 super-resolution → BiRefNet HR + MatAnyone2 temporal alpha → VP9 Alpha WebM.

- GPU0: text/vision/VAE; later RGB super-resolution and alpha. GPU1: DiT. Component placement and sequential phases, no TP/PP collective.
- Internal 352×624, 81 latent/80 RGB frames, 16 fps, ten LCM/simple steps, CFG 1, shift 5.
- Output 720×1280, 120 frames, 24 fps, five seconds. Alpha is semantic matting, not native diffusion alpha.
- API `0.0.0.0:8191`; per-request internal ComfyUI at `127.0.0.1:8192`; one request at a time.
- Nine model components listed under `[models]` use anonymous `/srv/models/` paths. The worker's own configuration must reference the same prepared files; this adapter does not generate a workflow from that list.

Original environment versions are recorded under `[reference_versions]`. Prepare compatible application code, `.venv`, workflow and weights explicitly; no application code, pretrained weights or credentials are bundled. Worker inputs/outputs and caches should live under `.runtime/<mode-id>/` and survive stops.

Then enable the configured mode, run `mgr register comfyui-wan-animate2-rgba`, `mgr check wan`, and `mgr switch wan`. Status/probes never submit generation.

The original private workflow completed an end-to-end media test and interruption tests. This public contract example has CPU-level checks; it is not an installable replacement for that private workflow or a new video-quality acceptance.
