# Plain vLLM: Qwen3.6-35B-A3B FP8, PP=2

A standard model server is the simplest adapter: validate local files, build one foreground command and check `/health` plus `/v1/models`.

## Recipe

- Existing Hugging Face checkpoint: `/srv/models/Qwen/Qwen3.6-35B-A3B-FP8`; `qwen3_5_moe`, FP8 weights.
- Two configured GPUs in `machine.toml` UUID order; PP=2, TP=1.
- 65,536-token context; up to ten scheduled sequences, 4,096 batched tokens; BF16 KV baseline. Ten scheduling slots do not mean ten simultaneous full-context requests.
- Text/programming only; Qwen reasoning/tool parsers. The sample's default no-thinking is a recipe preference, not a controller policy.
- API `0.0.0.0:8000`, OpenAI base `/v1`, model alias `qwen3.6-35b-a3b`, no API authentication. Configure the listener/authentication for your deployment.

## Explicit preparation

Prepare Python 3.12 and a private `.venv`, then install `requirements.in` explicitly. It records the original vLLM/CUDA compiler combination; it is not a portable universal lock. Review the framework's own compatible PyTorch, CUDA and driver requirements and freeze your resolved environment.

The original recipe used vLLM 0.32.0, PyTorch 2.13.0+cu130 and glibc 2.43. CUDA 13.0 headers needed a compatibility fix and pip SDK linker symlinks there. `patches/cuda_glibc_compat.py` is an **explicit preparation helper** for that pinned combination; inspect it before use, and do not apply it blindly to another toolkit. It is never run by switch/status. See [deployment lessons](../../../docs/deployment-notes.md).

Edit `mode.toml` and `machine.toml`, supply weights already present on the model mount, and set `enabled = true`. Then, from the project root:

```bash
./mgr register vllm-qwen36-35b-pp2
./mgr check vllm
./mgr switch vllm
./mgr status vllm
./mgr stop vllm
```

Environment/kernel caches go to `.runtime/<mode-id>/`; requests and results use the model API. Normal checks do not run inference or download missing weights. The historical NCCL/DeepGEMM environment choices are specific to this recipe, not defaults for other modes.

Original private deployment passed model loading and short API tests. This sanitized public adapter has CPU fixture validation; no new portable GPU acceptance is claimed. Long-context quality, sustained throughput and ten full-context concurrent requests remain separate tests.
