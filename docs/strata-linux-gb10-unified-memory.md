# Linux + GB10（统一内存）部署 Strata：单卡配方与平台差异

本文整理在一台 NVIDIA GB10（DGX Spark 类）上以 AetherSwitch 部署 Strata、服务多个并发编码 agent 的经验，模型与[双 NVIDIA GPU 部署指南](strata-linux-nvidia-dual-gpu.md)相同：Qwen3.8-Flash-Next GSQ-RCO IQ3_S，主要调用方为 Claude Code。

这是脱敏后的独立技术记录：路径使用通用示例，不包含部署者身份、实际硬件 UUID、网络身份、凭据、原始日志、请求内容或本机运行证据。文中的速度和资源数字是历史实测摘要；公开文档和适配器示例没有因此获得一次新的 GPU 部署验证。

整理日期：2026-11-01。以下判断限定于所列版本。**通用内容不重复**：环境固定方法、运行依赖清单、权重/native pack/MTP 的显式准备、Claude Code 接入环境和 AetherSwitch 生命周期边界均见双卡指南对应章节，本文只记录 GB10 与其不同的部分。

## 1. 先核对平台矩阵：ARM64 结论会按版本过期

| 项目 | 历史验证组合 |
| --- | --- |
| 系统 | Ubuntu 24.04.5 LTS，glibc 2.39（引擎包声明的下限，逐包核对） |
| CPU / 内存 | aarch64，20 核；CPU 与 GPU 共享约 121.6 GiB 统一内存 |
| GPU | 单张 GB10，计算能力 12.1（sm_121），无独立显存 |
| NVIDIA 驱动 | 580 系列（包要求 ≥ 580.65） |
| Python | 模式独立 CPython 3.12.3 / `.venv` |
| Strata 源码 | v0.2.9 官方 tag 源码包 |
| 官方 native engine | 0.1.55，build ID `release-0.1.55-cuda13-13.0-aarch64-99866a47a2b8` |
| 引擎包 | `strata-engine-0.1.55-linux-aarch64-cuda-cuda13.tar.gz`，656,111,027 字节 |
| 编码客户端 | Claude Code（Anthropic 协议直连） |

历史核对的 SHA256：

```text
strata-engine-0.1.55-linux-aarch64-cuda-cuda13.tar.gz
54efce60e58f1ec9c529bb4c3cc1ee0ab02f57a73af74206948b03fbd9582034

strata-v0.2.9 源码 tag 包
27fbaeed7247f5f22491a3a2d7eb0fde43a1375e89f1bdfde5b54f3ca0c44810
```

来源：[v0.2.9 发行页](https://github.com/Niko1221/Strata/releases/tag/v0.2.9)。

两条平台教训：

- **"该项目没有 ARM 构建"的结论按发行版过期。** 此前版本确实只有 x86_64 引擎，v0.2.9 起提供 `linux-aarch64` 官方包并以 `sm_121`/`compute_121` 编译。为别的架构下结论前，先读当版资产列表和包内 `engine-component.json` 的 `platform`/`archs`/`min_driver` 字段，把它们写进模式预检：装错平台或驱动过旧的包应在静态检查阶段被拒绝，而不是启动时才失败。
- 上游在该版本自带 [DGX Spark (Linux, ARM64) 官方说明](https://github.com/Niko1221/Strata/blob/v0.2.9/docs/DGX_SPARK_LINUX.md)，其中的逐开关实测是本配方的主要依据；本文的数字是对其的本地复核摘要。

## 2. 单卡配方：相对双卡篇改了什么

```toml
[gpu]
devices = [0]                # 单卡；无 roles、无 layer_split
context_tokens = 65536       # 统一内存余量下从 32K 提到 64K；KV 仍 int8
parallel_requests = 8        # 多 agent 并发；以 /v1/status 实际 serving 为准
vram_reserve_mib = 2048
vram_reserve_later_mib = 2560

[environment]
STRATA_ROUTE_TAIL_SKIP = "0" # 与双卡篇一致，完整路由专家和
STRATA_HC_UPMIX = "0"        # 8 槽下这两个 decode 开关各损约 3.5%（上游实测）
STRATA_SPEC_COST = "0"

[inference]
kv = "int8"
ple_io = "ram"               # PLE n-gram 表 28.8G 锁进内存；双卡篇为 direct/SSD
expert_cache = "auto"
prefill = "auto:32768"       # 32K 预取块比默认 8K 快 2–6%
mmap_experts = true          # 专家单份驻页缓存，见下
mtp_q4 = "all"               # draft 层 4-bit：+3.5% decode，target 校验每个 draft
mtp_hnorm = "stream"         # 每流一归一化：+4.3% decode
speculative_window = 4
speculative_min_probability = 0.5
batch_mtp = false
batch_groups = "auto"
conversation_cache_slots = 8
trim_stage_weights = false   # 单卡无多 stage 重复 dense，裁剪与显式切分一并省略
```

选择依据（均为上游在该部件上的实测，本地复核方向一致）：

- **`--mmap-experts`：专家只存一份。** 双卡篇的布局是"主机 RAM 注册约 46.8 GiB expert arena + 两卡各存热点专家"；GB10 统一内存下直接以页缓存持有模型文件本身，24,576/24,576 个专家槽全部可缓存，不再有第二份 RAM 拷贝。专家字节约 50.3 GiB。
- **`--ple-io ram`：能锁进内存就锁。** IQ3_S 的 PLE 表约 28.8 GiB，锁内存后 4K prompt 读取约 137 tok/s、对照 SSD 直读约 39 tok/s。注意这是 IQ3_S 的数字：更大的表（如 NVFP4 包的 51.2 GiB）与专家页缓存放不进同一份统一内存，官方对该组合明确留在 SSD。**按包型决定，不要照抄。**
- **`prefill auto:32768`**：该部件上默认 8,192 块慢 2–6%。
- **MTP 两个 decode 开关**（q4 draft 层、stream hnorm）合计约 +8% decode，且 target 逐个校验 draft，答案不变；它们与 `HC_UPMIX`/`SPEC_COST` 是四组独立开关，后两者在 8 槽下为负收益故显式关闭。
- **8 个请求槽**：上游在同一部件上测得 32 客户端聚合 54.2 tok/s（8 槽）对 47.3 tok/s（4 槽）；多 agent 场景取 8。槽数仍必须从 `/v1/status` 读回实际值。
- **64K 上下文**：统一内存下 KV 不再挤双 24G 显存，上下文翻倍；会话缓存槽数维持 8（引擎对该缓存另有 RAM 字节预算，本版本为 8192 MiB，见双卡篇同样提醒）。

## 3. 统一内存的账，以及和中控的配合

- 启动完成后的内存形态：约 50 GiB 专家页（page cache，可回收但保持热）+ 28.8 GiB 锁定 PLE 表 + KV/dense/MTP/草稿层。`MemFree` 降到 1–3 GiB 属预期，观察 `MemAvailable` 与换页/OOM/Xid，方法同双卡篇第 7 节。
- GB10 的 `nvidia-smi --query-gpu=memory.used,memory.total` 返回 `[N/A]`。AetherSwitch 0.2.0 起对此类统一内存设备改按 `--query-compute-apps` 逐进程求和，容量用 `machine.toml` 声明值；"释放"的判定即"无计算进程持有分配"。**页缓存不计入占用**：模式停止后专家页仍在 cache 里，属正常残留，不是泄漏。
- 页缓存被其它 GPU 模式（例如一个占满内存的 vLLM）逐出后，重新切回 Strata 需要重读专家文件，重启时长见第 5 节；这不是缓存损坏。

## 4. 适配器差异与两个实测坑

- **模型路径必须绝对。** 上游 server 以自身工作目录解析相对路径，而不是配置文件所在目录；适配器生成 run config 时应把所有模型/包/MTP 路径归一为绝对路径，模式 TOML 内可继续写相对形式。
- **该模型是 reasoning 输出。** 短 `max_tokens` 的探测请求会把预算全部消耗在推理段，`content` 为空属正常，不是服务故障。健康/就绪探针不做任何生成；性能基准显式关闭 thinking。
- 预检除双卡篇清单外，增加两条：引擎包 `platform` 必须是本机架构（防止 x86 包混入）；`--mmap-experts`/`--ple-io ram` 组合只对已核对过内存形态的包型启用。

## 5. 实测摘要（历史数字，脱敏）

计时方法与双卡篇第 6 节相同（Anthropic SSE、温度 0、关闭 thinking、固定编码任务），另加真实 Claude Code 客户端并发。

| 项目 | 历史实测 |
| --- | ---: |
| 单路 decode | ≈54 tok/s（上游 0.2.7.1 在同部件为 ≈46–47；0.2.9 引擎更快） |
| 4 路聚合输出 | ≈69 tok/s |
| 4 请求首字等待 p95 | < 2 s |
| 24,172-token 仓库摘录 | 首字 ≈22.5 s，首/中/尾三处配置值召回正确 |
| 3 个并发 Claude Code（完整工具 + 子 agent） | 各自修复带缺陷 fixture 并跑过测试，单任务 150–180 s 量级 |
| force-stop（原生模式） | ≈2.3 s |
| 从其它 GPU 模式切入到就绪 | ≈60 s（含重读专家页） |
| 对照：同机 vLLM docker 模式冷启动 | ≈350 s（`--rm` 容器冷 torch.compile） |

没有测过的同双卡篇声明：单流填满 64K、8 路满上下文、长时浸泡、其它统一内存部件。多 agent 并发下吞吐与前台延迟一起看，不要只报聚合值。

## 6. 验收差异

验收四问沿用双卡篇第 10 节。GB10 追加：

1. `/v1/status` 实际 `serving` 必须等于请求的 8，且 `/health` 的 `max_context` 等于 65536——这两项最常暴露"配置写了但引擎没吃到"。
2. 首次启动会构建引擎启动缓存与 CUDA 缓存；把它们留在模式 runtime 目录内，重启才会保持约 60 s 量级。清空 runtime 后第一次变慢属预期，不当作回归。
3. 与其它 GPU 模式做一轮真实互切：中断在途请求 → 停止/释放验证 → 启动到就绪 → 切回重载。统一内存设备的释放判定看计算进程归零，而不是显存曲线。
4. 权重跨机搬运按双卡篇第 3 节整套核对（两片 GGUF + native pack + MTP runtime，缺一不可）；搬运用字节级加 SHA256 双侧核对，仅有目录树一致不算数。

## 参考资料与可复用示例

- [Strata v0.2.9 发布页](https://github.com/Niko1221/Strata/releases/tag/v0.2.9)、[官方 DGX Spark (Linux, ARM64) 说明](https://github.com/Niko1221/Strata/blob/v0.2.9/docs/DGX_SPARK_LINUX.md)
- [双 NVIDIA GPU 部署指南](strata-linux-nvidia-dual-gpu.md)（通用准备、权重清单、客户端接入与生命周期边界）
- [中控协议](controller.md)、[共用部署经验](deployment-notes.md)、[公开 Strata 配方和适配器](../examples/modes/strata-qwen38-flash-next/README.md)

公开示例仍默认禁用；上游代码、引擎和模型各自保留许可证与使用条件，本文不分发这些产物。
