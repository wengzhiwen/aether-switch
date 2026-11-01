# AetherSwitch

**一机多态，一令切换。**

用于个人 Linux 推理工作站的轻量中控。把 vLLM、第三方原生推理引擎、按请求加载的媒体工作流统一为“模式”，由同一份生命周期代码管理切换。

[English](README.md) · [日常使用速查](中控使用说明.md) · [MIT 许可证](LICENSE)

## 基本行为

一次只运行一个模式，独占机器配置中的 GPU。先检查目标条件，再中断旧任务、停止整个旧 cgroup、核验资源释放，最后启动目标并等待它自己的 API 就绪。

- API 空闲、显存回到基线，不代表模式已经停止。
- 正常停止立即中断任务，默认 5 秒后清理残留；强停直接 KILL 整个受管理 cgroup。
- 保留输入、结果和日志，不自动重投任务。
- 重启电脑后手动选模式，不自动启动／恢复 GPU 服务。
- 安装依赖、准备权重和编译均为显式准备步骤，不在切换过程中运行。

中控只管理生命周期。API 地址、鉴权、PP／TP、分卡方式、模型参数、工作流和业务恢复仍归各个模式。

## 首次使用

需要 Linux、cgroup v2、用户级 systemd、NVIDIA 驱动／nvidia-smi、findmnt 和 `/usr/bin/python3`（3.11+）。中控自身只有标准库依赖。原部署使用 systemd 259；其他发行版需验证 `--kill-whom` 等选项支持情况。

以下命令用于新 checkout；保留已有部署的配置，不要覆盖正在使用的模式：

```bash
cp -n examples/machine.toml machine.toml
# 编辑模型挂载路径、文件系统 UUID、GPU UUID。
mkdir -p modes
cp -Rn examples/modes/vllm-qwen36-35b-pp2 modes/
```

按模式 README 显式准备 `.venv`、引擎和本地权重，调整路径，再把模式配置的 `enabled` 设为 `true`：

```bash
./mgr register vllm-qwen36-35b-pp2
./mgr check vllm
./mgr switch vllm
./mgr status
```

三个样例默认禁用，硬件身份均为占位值。它们不是跨机器直接可运行的部署包。

## 三个典型样例

| 模式 | 代表的接入方式 |
| --- | --- |
| [vLLM / Qwen3.6 FP8 PP=2](examples/modes/vllm-qwen36-35b-pp2) | 普通推理服务，启动命令和模型 API 就绪检查 |
| [Strata / Qwen3.8-Flash-Next](examples/modes/strata-qwen38-flash-next) | 第三方原生引擎、派生配置、MTP 和复杂双卡配方 |
| [Wan-Animate2 工作流节点](examples/modes/comfyui-wan-animate2-rgba) | 一整套业务工作流中的算力节点；API 常驻、模型按请求加载、忙碌与健康分开 |

Wan 样例展示适配器和接口契约，需自行提供资源 API、工作器和图。私有业务代码、传输接线及上游应用源码不包含在仓库中。

## 公共代码与本机部署

- `mgr`、`server_manager/`、`scripts/`：真正属于中控的共用实现。
- `examples/`：脱敏后的硬件与模型配方，保留模型名称和接入模式。
- `templates/`、`tests/`、`docs/`：接口约定、独立 CPU 测试和公共文档。
- `machine.toml`、`modes/`、`.runtime/`、`.local/`：本机配置、环境、业务和运行数据，Git 忽略。

各模式的五个生命周期脚本通过 `mgr register` 生成符号链接，指向同一份项目脚本，不能复制后分别维护。

测试命令：`python3 -B -m unittest discover -s tests -v`。测试不需要本机配置、GPU、模型或运行中的 systemd 服务。公开前运行 `python3 -B scripts/audit-public.py` 检查 Git 暂存内容。

更多内容：[中控与适配器接口](docs/controller.md)、[部署经验](docs/deployment-notes.md)、[Strata：Linux + NVIDIA 双卡部署实战](docs/strata-linux-nvidia-dual-gpu.md)、[测试范围](docs/testing.md)、[贡献说明](CONTRIBUTING.md)。
