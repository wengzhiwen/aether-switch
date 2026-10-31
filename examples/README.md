# Sanitized examples

These examples illustrate three integration patterns, derived from a workstation with two RTX PRO 4000 Blackwell GPUs (24 GB each). Hardware UUIDs, personal paths, addresses, client identities and credentials have been removed. Model names and useful tuning parameters are preserved.

| Directory | Integration |
| --- | --- |
| `vllm-qwen36-35b-pp2` | Standard vLLM service; FP8 model, PP=2, 64K context, ten scheduling slots |
| `strata-qwen38-flash-next` | Native third-party engine; split layers, CPU expert memory, MTP and generated server config |
| `comfyui-wan-animate2-rgba` | An on-demand GPU resource node inside a larger media workflow |

Copy `machine.toml` to the repository root and an example directory into `modes/` in a new checkout. Replace placeholders, prepare the mode's `.venv` and software explicitly, provide existing local weights, and enable the mode. Run `mgr register` to create all lifecycle links; examples never contain copied lifecycle scripts.

The original deployment used Python 3.12.16, Ubuntu 26.04.1, glibc 2.43, NVIDIA driver 595.99.02 and systemd 259. These are provenance, not controller requirements or a claim of cross-machine compatibility. GPU UUID order and memory headroom must be measured on the target machine.

Public adapters have been refactored and checked with CPU fixtures. They are not new GPU acceptance runs. The workflow example does not include an application: supply a resource server that implements the documented contract. See each README for exact dependencies, missing pieces and limits.
