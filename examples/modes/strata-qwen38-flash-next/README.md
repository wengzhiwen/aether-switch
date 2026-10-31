# Native engine: Strata / Qwen3.8-Flash-Next IQ3_S

This adapter demonstrates a third-party engine with a richer preparation boundary: native binaries and shared libraries, a local GGUF/native pack, MTP artifacts, an application server and a generated runtime configuration.

## Versions and preparation

- [Strata v0.2.8](https://github.com/Niko1221/Strata/releases/tag/v0.2.8), source commit `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`.
- Native CUDA engine 0.1.54, official build ID `8c1aea4bca1b+cuda13.0`; distinguish package build identity from source tag commit.
- Original adapter/server environment: Python 3.12.16. `requirements.in` records its dependency pins.

Prepare the upstream checkout under `app/`, compatible native bundle under `engine/`, and a mode-local `.venv/`. Verify upstream package hashes, loader dependencies, driver and GPU architecture support explicitly. They are not bundled or downloaded by AetherSwitch.

Prepare existing GGUF shards, native pack/tokenizer and MTP under the configured model mount. All sample paths use `/srv/models/Strata/`. Conversion/model preparation is an explicit upstream step; a GGUF file alone is insufficient. The sample preflight checks resolved parameter combinations, required artifacts and engine metadata; add full integrity validation for your chosen artifacts.

## Recipe

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` split over two files. Experts use host memory and a GPU cache; the second shard supplies PLE data from SSD.
- GPU0 helper layers 0–23; GPU1 main layers 24–47, output and MTP. This is pipeline layer splitting, not tensor parallelism or NCCL collectives.
- Fixed `layer_split=24`, trimmed stage weights, 32K context, INT8 KV, four serving slots, automatic batch groups; MTP enabled for eligible requests.
- Per-GPU reserves 2048/2560 MiB, eight conversation cache slots; measure real task memory, not only the post-load snapshot.
- `STRATA_ROUTE_TAIL_SKIP=0`; framework-specific behavior stays in this mode.
- `0.0.0.0:8080`, no authentication, OpenAI/Anthropic-compatible API, model alias `qwen3.8-flash-next-iq3_s`.

The adapter writes `.runtime/<id>/server.json` from the mode recipe. Logs, engine state and temporary caches stay under that runtime directory. `STRATA_KEEP_LEFTOVERS=1` and zero startup retries prevent the upstream server from replacing shared lifecycle ownership.

After editing paths/preparing/enabling, use `mgr register strata-qwen38-flash-next`, `mgr check strata`, and `mgr switch strata`. Readiness verifies model, context, vision state, authentication and actual four-slot serving configuration without submitting work.

The original deployment had GPU/API validation. The sanitized example is checked with CPU fixtures, not re-certified as a universally reproducible GPU deployment. Engine/source/model licenses and preparation remain upstream responsibilities.
