# Deployment lessons

These lessons come from a Linux workstation running three different GPU modes. They describe observations and boundaries, not defaults that every engine should inherit. The sanitized recipes are in `examples/`; raw operator records remain private.

## Storage and environment preparation

Check the mounted filesystem UUID, not just whether a path exists. Keep weights on the configured model mount and environments/source/build outputs on a suitable Linux filesystem. Verify checkpoint architecture, quantization, shards and required tokenizer/native artifacts before switching. Missing files must fail rather than trigger an implicit model download.

Pin Python, framework, CUDA runtime/compiler and native-engine versions independently. A native engine can ship its own CUDA libraries and need none of a vLLM mode's PyTorch/NVCC/NCCL setup. A pack/tokenizer or MTP conversion is a separate preparation artifact, not proof that a GGUF checkpoint is deployable.

Use explicit cache variables and each mode's child environment. HF offline mode does not prevent Torch Hub downloads: prepare required Torch Hub weights and point TORCH_HOME at them. Never modify HOME or shared parent environments to switch modes.

## CUDA and communication

The reference vLLM recipe used a CUDA 13.0 SDK with glibc 2.43. Mixing a newer compiler with older headers caused compile errors. CUDA CRT rsqrt declarations needed a local compatibility adjustment; pip SDK library layout needed developer linker symlinks. The example includes an explicit, version-checked helper for that historical environment. Keep such fixes mode-local and re-evaluate after upgrades; never blindly patch system headers or hard-linked package caches.

Foundation checks should isolate computation, JIT compilation and communication. On the reference machine, NCCL P2P initialization hung and disabling that transport let a bounded two-process all-reduce and later PP inference complete. This does not establish a global no-P2P rule. Similarly, the Qwen FP8 MoE recipe selected a Triton backend after backend-specific precision warnings; other engines retain their own choices.

FP8 weights and FP8 KV caches are separate decisions. Checkpoint scales, framework flags and calibration support vary. Record actual KV capacity: ten scheduled sequences do not guarantee ten full-context simultaneous requests.

## Native engines and host memory

Check package hash, native build ID, source commit, architecture support, driver and linked libraries separately. A precompiled build ID can differ from a release tag's source commit. Avoid claiming a local build or identical provenance without evidence.

Strata's trimming flag requires explicit layer splitting. Validate resolved flag combinations before stopping a current mode. Confirm effective concurrency from the running engine, not just a requested CLI number. CPU expert arenas, GPU hot-expert caches and SSD-backed components can intentionally coexist; measure MemAvailable, swap growth, OOM and driver errors before interpreting a memory graph as a crash.

## Lifecycle and application status

An API can remain idle with GPU memory at driver baseline and still own the mode. A workflow resource can report a documented 503/ready=false for busy while remaining healthy. Match that specific protocol, authenticate capabilities and verify process identity when the application exposes it; generic 503 is not success.

Workers using start_new_session stay in their inherited systemd cgroup. Stop the whole cgroup and then verify process identities, NVML PID disappearance and measured idle memory. Transient NVML stale PIDs can need bounded waiting; do not reset a GPU or kill unrelated applications.

Port probes must match the HTTP server's SO_REUSEADDR behavior. TIME_WAIT is not a live listener. Preflight can reuse an old mode's positively identified listeners but must reject unknown listeners before interrupting it.

Numeric PIDs/PGIDs can be reused after a restart or reboot. Use boot identity, starttime, unit InvocationID and application job identity, not a bare number, when settling interrupted work. Preserve prior failure/results and never auto-replay. An exit code after an intentionally interrupted generation is not by itself proof that resources remain allocated, nor should every instance of that exit code be globally declared successful.

## Validate the claimed layer

Distinguish prepared files, API readiness, model loading, real inference, client integration and production quality. Status/probes must not submit generation. A short API test does not validate full-context concurrency; an encoded alpha flag does not establish correct video alpha. A full media acceptance should decode and compare actual alpha frames and then separately evaluate visual quality.

CPU fixture tests validate control logic. Historical GPU runs validate their recorded environment. Rewriting an adapter into a sanitized example does not magically produce a new GPU acceptance for it; record that limit explicitly. Application DNS, transport, storage, worker authentication and business receipts belong to that application and should not be embedded into the generic controller.
