# Linux + NVIDIA 双卡部署 Strata：准备、调优与排障

本文整理一套 Linux 原生双 NVIDIA GPU 部署 Strata、服务单人和数个编码 agent 的经验。模型为 Qwen3.8-Flash-Next GSQ-RCO IQ3_S，主要调用方为 Claude Code。

这是脱敏后的独立技术记录：路径使用通用示例，设备仅按逻辑编号说明，不包含部署者身份、实际硬件 UUID、网络身份、凭据、原始日志、请求内容或本机运行证据。文中的速度和资源数字是历史实测摘要；公开文档和适配器示例没有因此获得一次新的 GPU 部署验证。

整理日期：2026-11-01。以下判断限定于所列版本，不代表 Strata 当前所有版本或其他模型的行为。

## 1. 先固定环境与来源

| 项目 | 历史验证组合 |
| --- | --- |
| 系统 | Ubuntu 26.04.1 LTS，glibc 2.43，原生 Linux |
| CPU / RAM | x86_64，8 个物理核心，系统可见物理内存约 90 GiB |
| GPU | 两张 RTX PRO 4000 Blackwell，各约 24 GiB 显存 |
| NVIDIA 驱动 | 595.99.02 |
| Python | 模式独立的 CPython 3.12.16 / `.venv` |
| Strata 源码 | v0.2.8，提交 `870aedd9e1ebe1c91f94da93b59af17729f6c5c0` |
| 官方 native engine | 0.1.54，build ID `8c1aea4bca1b+cuda13.0` |
| CUDA 库 | 官方引擎包自带 CUDA 13 runtime / cuBLAS |
| 模型 | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`，IQ3_S 混合量化 |
| 编码客户端 | Claude Code 2.1.314 |

官方发行包、源码 tag 和模型 revision 分别固定，不把 engine 的短 build ID 当成源码 tag 的完整提交。此处使用官方预编译包，未宣称由本地源码编译；短 build ID 对应的完整源码 SHA 未单独确认。

本次 CUDA 包含 `sm_120`，包内清单要求 NVIDIA 驱动至少 580.65。换硬件时重新核对架构支持、最低驱动、包内库和 `ldd`，不要只看系统报告的 CUDA 版本。

官方包名为 `strata-engine-0.1.54-linux-x86_64-cuda-cuda13.tar.gz`，历史核对的 SHA256 为：

```text
a183c298e26145a11b87baaeda5fe2ec32b305c600e6886206fd225bf3070726
```

来源：[固定版本发行页](https://github.com/Niko1221/Strata/releases/tag/v0.2.8)、[固定源码](https://github.com/Niko1221/Strata/tree/870aedd9e1ebe1c91f94da93b59af17729f6c5c0)。

## 2. 依赖：native engine 不需要照搬 vLLM 环境

这一组合没有依赖 PyTorch、vLLM、额外 CUDA pip SDK、NVCC、CMake 或 Ninja 来启动服务。官方引擎带运行库；Python 环境负责应用服务、tokenizer 和准备工具。只有选择自行编译时才重新准备上游构建链。

Python 3.12 / Linux 对应的运行依赖如下，已包含其锁定的依赖项：

```text
numpy==2.5.3
jinja2==3.1.6
regex==2026.9.10
pyyaml==6.0.3
tqdm==4.70.1
requests==2.34.2
pillow==12.3.0
psutil==7.2.2
markupsafe==3.0.3
certifi==2026.10.19
truststore==0.10.5
charset-normalizer==3.5.1
idna==3.20
urllib3==2.8.0
```

依据为该提交的 [requirements.txt](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/requirements.txt)，Python 条件依赖已按 3.12 解析。不要把不同 Python 版本的 numpy 条件分支全部装上。

准备顺序：

1. 将固定源码放到模式的 `app/`，官方兼容引擎和清单放到 `engine/`。
2. 用已显式准备的 Python 3.12.16 创建模式自己的 `.venv`，安装并锁定依赖。解释器、源码和构建产物留在 Linux 文件系统。
3. 核对发行包 SHA256、包内组件清单和共享库解析，例如 `ldd engine/strata`；此检查不是 CUDA 计算或模型加载验证。
4. 显式设置子进程的 CUDA 库目录和缓存路径，不修改系统头文件、全局 Python 或其他模式的环境。

安装、下载、转换、编译都在准备阶段执行。普通启动或状态查询遇到缺少组件时，应明确失败。

## 3. 权重齐全：不止两片 GGUF

本文统一用 `/srv/models/Strata` 作为**示例模型目录**，应替换为操作者已挂载并核对身份的模型分区。不要仅凭目录存在就认定正确磁盘已挂载。

主模型固定 revision 为 `1868010e27505d4b8cc15970cdcc4dd76005d7de`。MTP 来源为 `Qwen/Qwen3.8-Flash-Next`，固定 revision 为 `de4b8e4d43b917e7706784d8bb445c9af86a3540`。

| 组件 | 示例位置或用途 |
| --- | --- |
| GGUF 第一片 | `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` |
| GGUF 第二片 | `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`，含 PLE 表 |
| Native pack | `packs/iq3_s/`，含索引、派生 dense 权重和 tokenizer |
| MTP 原始及转换文件 | `mtp/`，含原始张量、校验清单和转换后的 GGUF |
| MTP runtime pack | `mtp/rt/`，含 `experts.bin`、`dense.bin`、`dense.txt`、`draft_vocab.bin` |
| 视觉投影 | `mmproj-Qwen3.8-Flash-Next-BF16.gguf`，仅启用视觉时需要 |

两片主 GGUF 总计约 83.6 GB / 77.9 GiB。IQ3_S 指这个 checkpoint 的混合量化配方，不表示所有张量的存储类型都相同。第二片约 26.8 GiB 的 PLE 数据在此配方中通过 SSD direct I/O 读取。

### Native pack 和 MTP 的显式准备

以下命令对应固定源码版本的工具，**会生成文件，MTP fetch 还会下载数据**。它们仅用于准备阶段；先准备好源码、虚拟环境、目标模型目录和已有主权重。`MODE_DIR` 应指向自己的已安装模式目录。

```bash
export MODE_DIR="$PWD/modes/strata-qwen38-flash-next"
export MODEL_ROOT=/srv/models/Strata
export STRATA_GGUF_PY="$MODE_DIR/app/third_party/llama.cpp/gguf-py"

"$MODE_DIR/.venv/bin/python" "$MODE_DIR/app/tools/iq_pack.py" \
  --gguf "$MODEL_ROOT/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  --out "$MODEL_ROOT/packs/iq3_s"

"$MODE_DIR/.venv/bin/python" "$MODE_DIR/app/tools/mtp_fetch.py" \
  fetch --out "$MODEL_ROOT/mtp"
"$MODE_DIR/.venv/bin/python" "$MODE_DIR/app/tools/mtp_fetch.py" \
  verify --out "$MODEL_ROOT/mtp"
"$MODE_DIR/.venv/bin/python" "$MODE_DIR/app/tools/mtp_pack.py" \
  --src "$MODEL_ROOT/mtp" --experts q2_0 --out "$MODEL_ROOT/mtp/mtp-q2_0.gguf"
"$MODE_DIR/.venv/bin/python" "$MODE_DIR/app/tools/mtp_rt.py" \
  --gguf "$MODEL_ROOT/mtp/mtp-q2_0.gguf" --out "$MODEL_ROOT/mtp/rt"
cp -n "$MODE_DIR/app/data/draft_vocab.bin" "$MODEL_ROOT/mtp/rt/draft_vocab.bin"
```

运行 MTP 转换前，应在 `STRATA_GGUF_PY` 指向的位置准备上游固定 llama.cpp 提交 `3cf03257f219afbe7334045ff7c6a06ac68c627d` 的 `gguf-py`。任意 PyPI `gguf` 包不能替代这项版本要求。上游校验依赖默认固定来源；不要用 revision/endpoint 覆盖变量绕过张量校验。历史准备验证了 31 个 MTP 张量的哈希。

这里没有给 `iq_pack.py` 传 `--experts-bin`，专家继续引用原 GGUF，避免额外复制约 50 GB 专家数据。派生 dense 权重、MTP 原始/转换文件仍属于模型产物，应和主权重一起留在模型分区。

本次 MTP 使用 Q2_0 专家和默认 CJK draft vocabulary；提议由主模型验证。Native pack 的索引、tokenizer、聊天模板以及 MTP runtime 文件必须各自检查，不能仅凭 GGUF 大小正确就报告“准备完成”。

## 4. 双卡与主机内存怎样分工

| 逻辑设备 | 分工 |
| --- | --- |
| CUDA0 / Helper | 层 0–23，持有自己层的专家缓存和会话状态 |
| CUDA1 / Main | 层 24–47、输出头、MTP 和自己层的专家缓存 |
| 主机 RAM | 一份共享 expert arena；CPU 计算 GPU 未驻留的专家 |
| 模型 SSD | 主 GGUF、派生权重和按需读取的 PLE 数据 |

这是连续层切分的流水线并行，不是 tensor parallel；不需要沿用其他引擎的 NCCL 配置。显卡编号必须按自己的设备映射核对，本机 UUID 仅进入私有机器配置，不写进公共配方。

历史运行在主机 RAM 中读取并注册了约 46.8 GiB expert arena，再由两卡缓存热点专家。因此 RAM 和显存同时较高是这套架构的预期行为，不能理解为“本应全部放进显存，却意外卸载到内存”。PLE direct I/O 也不是操作系统的 swap。

Linux 上该组合成功注册普通主机页并请求透明大页；未配置显式 huge pages。没有仅因日志提到缺少显式大页就修改系统内存设置。

来源：[固定版本的双卡说明](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/MULTI_GPU.md)。

## 5. 最终采用的编码配方

下面是配方片段，不是完整的启动配置；模型路径、组件、硬件和 `[control]` 等仍需按安装环境补齐。完整的禁用示例在文末链接中。

```toml
[gpu]
layer_split = "24"
context_tokens = 32768
parallel_requests = 4
vram_reserve_mib = 2048
vram_reserve_later_mib = 2560

[environment]
STRATA_ROUTE_TAIL_SKIP = "0"

[inference]
kv = "int8"
ple_io = "direct"
expert_cache = "auto"
prefill = "auto"
speculative_window = 4
speculative_min_probability = 0.5
batch_mtp = false
batch_groups = "auto"
trim_stage_weights = true
conversation_cache_slots = 8

[vision]
enabled = false
```

这套配置的选择依据：

- **4 个有效请求槽、两个流水线组**：减少几个 agent 同时工作时的排队。必须从引擎和 `/v1/status` 确认实际为 4 槽，不能只相信 requested 值。
- **分开看单流 MTP 和 batch MTP**：此版本在两个流水线组时没有使用 batch MTP；只有一个活跃请求时仍自动使用单流 MTP。双路吞吐与四路/前台延迟有取舍。
- **固定 K=24 并裁剪重复权重**：先由自动分层得到 K=24，再固定该切分配合 `--trim-stage-weights`。该选项需要显式切分，不能与 `auto` 搭配；输出也不能保证逐位相同，仍要验证实际代码任务。
- **保留 INT8 KV、32K 上下文**：这些精度和容量足够完成本轮任务，没有为了腾显存改用 Q4 KV 或提高上下文长度。
- **关闭视觉**：编码文本用途释放 encoder 空间，原有 mmproj 可以保留。日后启用图像需重新评估并发和显存。
- **保留 prefill auto**：固定 2048 没有降低测试中长输入对短请求的阻塞，反而延长了长输入读取；有并发槽也不保证长 prefill 不影响其他请求。
- **8 个会话缓存槽**：有利于多个对话轮换。实际还受 RAM 字节预算约束，本次引擎给出的上限为 8192 MiB；不是八个任意长会话保证驻留，更不等于八路同时推理。
- **每卡多留 512 MiB**：早期预留为 1536/2048 MiB，24K 输入时采样空闲仅约 0.7/0.8 GiB。提高到 2048/2560 MiB 后，单流和四路合计速度损失都不到约 1%，选择更充足的余量。

`STRATA_ROUTE_TAIL_SKIP=0` 保留完整路由专家和，用作质量基线。不同 CPU/GPU 路径及缓存变化仍可能带来浮点舍入差异，不将此设置等同于任意配方之间的逐位一致。

### 这一版本没有采用的建议

**DFlash2**：固定版本的 [drafting.py](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/strata_control/drafting.py#L141) 明确为 `ENGINE_SERVES_DFLASH2=False`。服务端/分层路径会回退 MTP；因此没有下载一个在该 API 模式下不能生效的 drafter。不能把别的版本、单卡 CLI 或自动回退当作本配方已获得 DFlash2 提速。

**thorough tune、更多上下文和进一步压低 KV 精度**：本轮未进行最长两小时的 thorough 搜索，也没有证明所有参数组合的极限。目标是单人、数个编码 agent 的可用折中，而非最大跑分。

## 6. 实测摘要：延迟和吞吐要一起看

历史上比较了 14 次配置运行，包含基线及重复确认；以下是整体方案的摘要，不能把差异全部归因于某个单项开关。模型、引擎版本、INT8 KV、32K 配置和完整路由专家基线保持一致。

计时使用 Anthropic Messages SSE、温度 0、关闭 thinking，固定编码文本任务，每请求上限 256 个输出 tokens，分别发出 1/2/4 个同时请求。下面两组各测三轮：吞吐取每轮总输出 tokens / 整轮墙钟时间的中位数；首字等待包含排队、prefill 和服务端开销。四请求 p95 来自每组 12 个请求，是小样本描述，不是生产 SLA。

| 项目 | 初始配置复测：单请求、视觉开启 | 最终配方 |
| --- | ---: | ---: |
| 单路聚合输出 | 108.1 tok/s | 111.6 tok/s |
| 两路合计输出 | 134.0 tok/s | 122.4 tok/s |
| 四路合计输出 | 129.7 tok/s | 177.0 tok/s |
| 四请求首字等待 p95 | 6.39 s | 1.37 s |
| 三后台生成期间，前台首字中位数 | 10.48 s | 0.59 s |

最终两路合计吞吐略低，但四路吞吐和交互延迟更符合几位编码 worker 同时工作的目标。三后台加前台场景各测三轮，后台每请求上限 512 tokens，开始输出后再加入一个上限 96 tokens 的短前台请求。

在另一个混合样例中，一段 14,862-token 仓库摘录和后台生成运行期间，短请求首字等待从约 10.05 s 降至 5.51 s；该场景只各测一次，仍然存在明显 prefill 干扰。

顺序轮换八个会话时，4 槽缓存会在这组顺序下互相淘汰；同为 prefill=2048，8 槽配置的第二轮缓存复用约 98%，首字中位数由约 1.46 s 降至 0.24 s。最终 auto 配方复验约为 98.43%、0.23 s。多请求流水线下某些槽可能因没有逐槽 draft KV 而不停车保存，因此不能承诺每次都命中。

性能任务按输出长度截断，仅用于计时，不能当作完整代码正确性的证明。正确性另行检查：

- flatten、区间合并、LRU、拓扑排序：生成结果实际执行，共 50 项断言通过；另验证 Anthropic 工具调用和工具结果回传。
- Claude Code 完整默认工具和 reviewer 子 agent：子任务诊断后由主会话编辑、运行测试；测试文件未修改。
- 三个独立 Claude Code 客户端并发：分别修复不同的小型 Python fixture，共 12 项测试通过，测试文件未修改。
- 单个 24,172 输入 tokens 的仓库摘录：首部、中部、尾部三个配置值召回正确，输出 22 tokens，首字约 10.92 s。

这些结果没有证明单路填满 32K、四路满上下文、全天持续负载、大型仓库的成功率或其他 GPU 的性能。原始请求/日志/本机验收文件保持私有；这里仅发布脱敏后的方法和汇总。

## 7. 分清显存重载、混合执行和真正故障

| 现象 | 本次可确认的解释或检查方法 |
| --- | --- |
| 两卡约 22 GiB → 接近零 → 数 GiB → 约 22 GiB | 调参时受控停止与重载的典型过程；先对齐 PID、服务生命周期和日志，不能仅凭曲线认定崩溃 |
| RAM 与两卡显存都很高 | 主机 expert arena 与 GPU 热点专家缓存共存；这套配方常驻服务不会每请求清空整个模型 |
| 空闲时利用率、功耗和频率下降 | 与是否仍驻留模型是不同问题，应一起看显存与进程 |
| `MemFree` 很低 | 同时看 `MemAvailable`、swap-in/out 增量、OOM kill 和驱动错误 |
| 引擎报告启动后还有显存 | 启动快照不能代表长 prefill、图捕获和客户端并发期间的最低余量 |
| 停止后短暂出现旧 GPU PID | 有界等待并核验 cgroup/PID 身份和释放阈值，不 reset GPU、不杀未知进程 |

最终配方运行真实客户端、长输入及会话复用时，每 2 秒采样，共 49 次：两卡采样最低空闲约 992/1128 MiB，主机 `MemAvailable` 最低约 28.5 GiB；没有 OOM kill、NVIDIA Xid 或服务自动重启。观察到 swap-in，swap-out 增量为零，不能将历史 swap 占用直接描述为正在持续换出。

这些是采样观察值，不能保证捕获毫秒级瞬态峰值。主机 RAM 预算也要容纳操作系统、agent 客户端、会话快照和文件缓存，不能只按 GGUF 大小算。

## 8. Claude Code 接入要点

Anthropic base URL 使用服务根路径，**不附加 `/v1`**。以下示例只访问 loopback；远端部署应使用操作者自己的地址，监听和鉴权由模式决定。本配方历史上选择免鉴权服务，SDK 的占位字符串并不是服务端验证的凭据。

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_API_KEY=strata-local-placeholder
export ANTHROPIC_MODEL=qwen3.8-flash-next-iq3_s
export ANTHROPIC_SMALL_FAST_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
export MAX_THINKING_TOKENS=0
unset CLAUDE_CODE_SIMPLE
unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY
claude
```

该环境用于选择本地端点和测试过的客户端限制，不修改用户持久配置或权限策略。验证需要真实工具循环：读取文件、编辑、执行测试并检查结果，不能只靠一句文本回复。独立 CLI 客户端和 CLI 内的子 agent 是不同场景，两者都要测。

该版本上游的低上下文快捷接入建议可能设置 SIMPLE；本次为了保留完整工具和子 agent，没有采用该开关。长会话应在窗口不足前压缩，会话缓存复用不能扩大模型上下文。关闭 thinking 是本轮验证条件，不是关于复杂任务质量的普遍建议。

## 9. 在 AetherSwitch 中接入，复用唯一生命周期

公开示例默认禁用。以下步骤适用于**新安装目录**；已有部署应先核对目录，不能拿公共样例覆盖。

```bash
cp -n examples/machine.toml machine.toml
mkdir -p modes
cp -Rn examples/modes/strata-qwen38-flash-next modes/
# 编辑 machine.toml 的实际挂载身份和 GPU UUID。
# 显式准备模式环境、引擎和权重，修改示例路径，最后设 enabled=true。
./mgr register strata-qwen38-flash-next
./mgr check strata
./mgr switch strata
./mgr status
```

`register` 生成 start/status/stop/force-stop/launch 五个链接，全部指向共用 `scripts/mode-entry.sh`。平时使用 `mgr switch` 或模式 `start.sh`；内部 `launch.sh` 需要中控授权，不另外包一层 `systemd-run` 或模式自管 supervisor。

适配器负责 `check`、`listen_ports`、`build_launch`、`probe`，分别核对文件/参数、声明端口、生成前台命令、只读查询协议状态。共享中控负责挂载身份、GPU 池独占、锁、unit/cgroup 所有权、停止和资源交接。先确认目标可部署，再中断旧模式；失败只清理已确认属于本次启动的资源，保留输入、结果和日志，不重投任务。

Strata 配方设置 `STRATA_KEEP_LEFTOVERS=1`、`STRATA_START_RETRIES=0`，避免上游自行接管进程清理或启动重试；它们不是另一套中控。正常启动不执行上面的模型准备命令，重启机器后也不自动恢复 GPU 模式。

## 10. 验收时分别回答四个问题

1. **文件准备好了吗？** 核对正确挂载、版本、哈希、分片、pack、tokenizer、MTP 及最终参数组合；不要初始化 GPU 或下载补缺。
2. **服务和模型可用吗？** 只读检查 `/health` 的 loaded/model/context/images/api_key、`/v1/models` 和 `/v1/status` 的实际 serving。此配方期望 `loaded=true`、正确模型/32K、视觉关闭、免鉴权和实际四槽。槽数不是实时空闲数量，探针未报告 busy 也不证明没有生成请求。
3. **实际任务成功了吗？** 显式提交最小文本、SSE、工具结果回传、真实编码客户端和必要的长输入样例。Anthropic 输入总量合并 `input_tokens`、`cache_creation_input_tokens`、`cache_read_input_tokens`，避免把缓存后的长输入误记成一个 token。
4. **中断和资源交接正确吗？** 在有界真实请求中中断，核对整个受管理 cgroup、进程身份和 GPU 释放，保留结果；不要把故意中断的退出码单独当作残留，也不把该退出码全局视为成功。

端口探测需要与 HTTP 服务一致的 `SO_REUSEADDR` 行为：旧连接的 TIME_WAIT 不等于仍有监听者。探针保持只读，不能为了报告健康而触发一次生成。

## 参考资料与可复用示例

- [Strata v0.2.8 发布页](https://github.com/Niko1221/Strata/releases/tag/v0.2.8)
- [该版本依赖](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/requirements.txt)、[双卡说明](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/MULTI_GPU.md)、[drafter 选择实现](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/strata_control/drafting.py)、[客户端接入实现](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/strata_cli/launch/harness.py)
- [主模型固定 revision](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/1868010e27505d4b8cc15970cdcc4dd76005d7de)、[MTP 原模型固定 revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540)
- [公开 Strata 配方和适配器](../examples/modes/strata-qwen38-flash-next/README.md)、[中控协议](controller.md)、[共用部署经验](deployment-notes.md)

公开示例仅携带配方和适配器，仍需操作者准备软件、模型和自己的硬件配置。上游代码、引擎和模型各自保留许可证与使用条件，本文不分发这些产物。
