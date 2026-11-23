# DeepSeek-V2-Lite TP8 Acceptance

Acceptance Date 2026-11-23 (Asia/Shanghai). Source code baseline `8e134acac4d0c2d991c32059db09a5fbf2244422`, with workspace chat fixes and regression cases; patches saved with this evidence. Model uses specified four safetensors shards. TP8 uses fixed Vendor SDK image, eight MLU590-M9 cards, BF16, CNCL and existing Stage C.3 native computation path. [Current acceptance evidence](../artifacts/logs/deepseek/acceptance.SjYNvMQF)

## Previous Blocked Reproduction and Resolution

- By default device nodes are not visible in sandbox; `cnmon` returns `CNDEV_ERROR_NO_DRIVER`. After authorized execution eight cards are visible, each with 96 GiB free; fixed image, device, model preflight pass. Previous device unavailability conclusion only applies to corresponding execution environment. [Preflight](../artifacts/logs/deepseek/deepseek-check.HiSvKfus)
- Chat emitter applies complete notification regex to text sequentially. Current local regression reproduced notification leak written in three times to stdout. Fix preserves possible notification tail; after identifying complete notification, only matches are written to diagnostics file; remaining text continues streaming to terminal and session log. 12 Chat CPU use cases, routing check, and GEMM CPU CTest passed. [Fix](../integration/vllm_native_mlu/phase16/deepseek_chat.py) · [Regression](../integration/vllm_native_mlu/phase16/test_chat.py)
- Production entry fixes `max_model_len=128` (input + output) and `max_num_seqs=1`. Minimum performance combination 128→32 for this task already exceeds length limit; all 48 combinations **BLOCKED** under current production config. Concurrency 4/8/16 also exceeds current scheduling config. TP8 model constructor explicitly rejects >128; cache scheduler uses same limit; archived YaRN RoPE table has only 131 positions. Only adjusting outer Make variables cannot open this matrix. Current evidence insufficient to attribute these limits to device VRAM. [Inference Entry](../integration/vllm_native_mlu/phase15/deepseek_infer.py) · [TP8 Model Constraints](../integration/vllm_native_mlu/phase15/tp8/model.py) · [Chat Config](../integration/vllm_native_mlu/phase16/deepseek_chat.py) · [Exit Code 2 Config Rejection Log](../artifacts/logs/deepseek/acceptance.SjYNvMQF/unsupported-configurations.log)
- Explicit requirements from this task and archived technical gates. Repository provides no attributable original signature from project lead, minimum throughput, latency, or new tolerance requirements; status **UNKNOWN**. 20% TPOT/Prefill, 15% MoE, and TP8 one hour are engineering goals proposed by this task. [Historical requirement audit](FINAL_ACCEPTANCE_GAP.md)

## Implementation Scope

TP8 entry uses Vendor vLLM to manage workers, KV paging metadata, and request scheduling; uses CNCL to exchange rank data. Stage C.3 uses native GEMM, RMSNorm, RoPE, QK, Softmax, AV, device KV Cache, output-axis MLA, and fused/batched MoE. 26 of 27 layers are MoE. Chat re-renders preserved history per request within single engine; records valid history, truncation, and token timing. For detailed source code, current tests, early failures, and historical evidence see indices below. [Implementation Audit](PROJECT_FUNCTION_AUDIT.md) · [Stage D Architecture](STAGE_D_ARCHITECTURE.md) · [Phase Acceptance Metrics](phase15_gate_plan.md)

## Current Features and Numerical Matrix

| Acceptance Item | Status | Evidence and Applicable Scope |
|---|---|---|
| Build artifact routing and frozen C.1–C.3 source/evidence integrity | PASS | `make deepseek-verify`, routing tests and current real worker identity |
| Real weights, 27 layers, TP8/CNCL, Prefill and incremental Decode | PASS | Default 12→32 requests pass; all ranks bound to devices 0–7; verify each rank has 27 native output projection modules; 32 IDs individually match historical Stage D with same prompt; eight workers exit code 0 |
| BF16/FP16 QK → Softmax → AV | PASS | After current Softmax recompilation, both precisions have 144 stage records each with zero overshoot. Historical baseline reproduced 4 FP16 overshoots |
| BF16/FP16 MLA, KV Cache, Prefill, continuous Decode | PASS | Phase 10 current source recompiled; 28 configurations normal exit; 2,304 MLA and 480 cache stage records zero overshoot |
| TP8 FP16 full model | BLOCKED | Current TP8 model and launcher only support BF16; Phase 9/10 two-precision operators accepted independently |
| Fixed TP8 BF16 full model logits | PASS | 512 rank/steps bit-level identical to frozen C.2; scope is same path 17→32 |
| Fixed 17→32 end-to-end independent BF16 reference | PASS | stock Vendor uses original architecture; four times individually matched 32 Stage C.3 greedy tokens |
| CPU error handling and regression | PASS | `make test` CTest, routing, 12 Chat tests all passed. Illegal lengths and variants rejected |
| Real hardware TTY Chat, truncation and reset | PASS | Single engine completes 15/15; nine rounds actual truncation; no old notifications in TTY and disk transcript; zero valid history after reset |
| TP8 repeated startup and normal shutdown | PASS | Inference, benchmark, instrumentation, and two independent Chat engines all have eight worker cleanup records |
| TP8 fault injection and one-hour stability | NOT RUN | B2, TP1 one hour, TP2 repeated start/stop, and some other operators are historical evidence |
| Production performance, specified workload matrix | BLOCKED | Current 128-token total budget and single-sequence |
| Stock Vendor same-condition A/B | PASS | Group of comparisons with same model, image, cards, TP, precision, context, input, output, and logical cache config; four sequences identical |
| Overall acceptance | BLOCKED | Specified workloads cannot run; complete multi-shape numerical coverage and stability still to close |

Default entry original records in [12→32 TP8 run directory](../artifacts/logs/deepseek/deepseek.Rx1PBgke). First request TTFT 27.002 s, average TPOT 276.49 ms, p95 ITL 446.48 ms, Decode 3.617 token/s, end-to-end 0.900 output token/s. This is first request after initialization, including first JIT overhead. Use `make deepseek` to test single request. Warm request sampling provided by `make deepseek-benchmark`.

## BF16 / FP16 QK → Softmax → AV

Phase 9.4 real hardware replay uses card 0, original fixture, frozen QK/AV objects and old baseline comparison; current Softmax recompiled from source, using original threshold `1e-5 + 1e-5 * abs(reference)`. Old FP16 Softmax low-precision probability shows ULP-level difference; subsequent AV output rounding amplifies to overshoot. Old path has one AV output overshoot in 128-token Prefill and subsequent 129-key Decode each; reproducible with one and two queues; total four. QK matches input fixture. Current corrected BF16, FP16 each have 144 stage records with zero overshoot; low-precision QK, probability, and AV output all bit-level identical; FP32 AV max absolute error 3.5763e-7. [Replay Log](../artifacts/logs/deepseek/acceptance.SjYNvMQF/phase9_4/runtime.log) · [Per-Stage Numerical](../artifacts/logs/deepseek/acceptance.SjYNvMQF/phase9_4/chain.csv)

Phase 10 same real hardware recompiled and replayed BF16/FP16 MLA, device KV Cache and three-step continuous Decode. In 28 configurations of Prefill 1, 7, 32, 128 and mixed 7/128, single/dual requests and single/dual queues; 2,304 MLA and 480 cache stage records all zero overshoot, zero abnormal non-finite numbers. Covers reset/reuse and cache protection. Uses frozen reference and original threshold. [Recompile and Replay Log](../artifacts/logs/deepseek/acceptance.SjYNvMQF/phase10/runtime.log) · [Original Matrix](../artifacts/logs/deepseek/acceptance.SjYNvMQF/phase10/matrix)

## Real Hardware Chat

First TTY regression uses eight output token budget. Seven requests use single engine and generate 42 tokens; two reach output limit, four end at role boundary with no saveable assistant text, last one completes. Terminal executed `/stats`, `/reset`, `/exit`; eight workers exit. This input set did not trigger history truncation; complete context acceptance requires inputs known to fill history. [Session Original Record](../artifacts/logs/deepseek/acceptance.SjYNvMQF/chat.bPiqPRMN/interactive_requests.csv) · [TTY Output](../artifacts/logs/deepseek/acceptance.SjYNvMQF/chat.tty.log)

Re-run uses archived 14-round prompts with added 15th round after reset. Single engine in real TTY completes **15/15** requests, generates **139** tokens. Rounds 6–14 each truncate one complete history round; CSV has total nine positive `dropped_turns`; terminal and transcript have no old prompts. Round 15 has zero valid history. Eight workers exit 0; CNMON no residual device processes. This test verifies history concatenation, truncation, and resource lifecycle; output content quality has no independent metrics. [Session Summary](../artifacts/logs/deepseek/acceptance.SjYNvMQF/chat.77ZidgAI/session_summary.json) · [Truncation CSV](../artifacts/logs/deepseek/acceptance.SjYNvMQF/chat.77ZidgAI/interactive_requests.csv) · [TTY Original Output](../artifacts/logs/deepseek/acceptance.SjYNvMQF/chat-truncation.tty.log) · [Final State](../artifacts/logs/deepseek/acceptance.SjYNvMQF/final-chat-device-state.log)

## Current Performance Baseline

固定 17 个输入 token、32 个输出 token，八卡 BF16 TP8、batch/concurrency 1、greedy、忽略 EOS、eager、关闭 prefix cache。一个新引擎，先测首次请求，再测三个热请求。基准走同步 `engine.step()`，从提交到接收 token 记录主机墙钟时间。每请求 31 个 ITL，p95 为 nearest-rank，表中热请求结果取三个请求的中位数。单引擎初始化 **130.846 s**，包含权重加载、起 worker、KV cache 及 warmup 等。rank 0 的 loader 日志单独报告权重加载 0.49 s。基准的首次 TTFT 从引擎初始化结束后计时。[原始请求](../artifacts/logs/deepseek/acceptance.SjYNvMQF/deepseek-benchmark.CoyjTiy8/performance.csv) · [逐 token ITL](../artifacts/logs/deepseek/acceptance.SjYNvMQF/deepseek-benchmark.CoyjTiy8/token_intervals.csv)

| 指标 | 首次 | 热请求中位数 |
|---|---:|---:|
| TTFT | 15,764.10 ms | 436.02 ms |
| 平均 TPOT | 961.79 ms | 155.53 ms |
| p95 ITL | 376.75 ms | 174.85 ms |
| Decode 吞吐 | 1.040 token/s | 6.429 token/s |
| 端到端输出吞吐 | 0.702 token/s | 6.086 token/s |
| 端到端延迟 | 45.580 s | 5.258 s |
| 请求提交时间 | 55.59 ms | 1.23 ms |

首次请求的最大 ITL 为 22.736 s，热请求最大 ITL 为 192.59 ms。首次长尾的归因待内核和主机时间线证实。串行条件下聚合输出吞吐等于该单请求端到端输出吞吐。17 除以 TTFT 得到热请求 39.0 输入 token/s 的端到端 prefill 代理值，包含请求提交、首 token 采样和返回。独立同步插桩的热 prefill forward 区间为 436.80 ms（最慢 rank），即 38.92 输入 token/s，区间覆盖 27 层 forward，首 token logits 另计。

- 请求后各卡占用（GiB，rank 0–7）为 6.071、6.103、6.071、6.103、6.165、6.165、6.290、6.040。退出后八卡均回到 0。48 次主机采样中各卡至少有一次 100% 利用率，最高采样显存 6441 MiB。6441 MiB 为本轮采样最大值，可作为真实峰值的下界。真实分配峰值需设备或 allocator 高水位计数器。[显存快照](../artifacts/logs/deepseek/acceptance.SjYNvMQF/deepseek-benchmark.CoyjTiy8/memory_per_card.csv) · [CNMON 与进程采样](../artifacts/logs/deepseek/acceptance.SjYNvMQF/baseline-telemetry.log)
- CPU 进程的生命周期 `%CPU` 和 RSS 已保留。CPU 调度、kernel launch 和同步的纯开销需要精确的时间线证据。当前 wall time 和外部采样无法单独归因。

## TP8 关键路径

Stage C.3 原有的同步 instrumentation 完成一次首次请求和一次热请求。512 个 rank/step 的 102,400 维 FP32 logits 与冻结 C.2 逐位一致。两次请求的 32 个 greedy token 都通过 C.1 序列校验，八 worker 退出 0。运行退出阶段记录了 TCPStore 收发告警，退出码和生命周期正常。[当前 logits 校验](../artifacts/logs/deepseek/acceptance.SjYNvMQF/profile/logit-parity.csv) · [profiling 原始记录](../artifacts/logs/deepseek/acceptance.SjYNvMQF/profile)

每个 token step 取 forward + logits 总墙钟最长的 rank，以该 rank 为该 step 的关键 worker 近似。热 Decode 的 31 步平均为 **177.00 ms**。下表按可归因的区间排序，包含同步及 collective 到达等待。其余 **59.19 ms（33.44%）** 未独立归因，可能包含分配、调度、拷贝、同步和 instrumentation 自身耗时。热 Prefill 的一个 17-token step 为 **448.04 ms**，约 37.94 输入 token/s，含首 token logits 和插桩同步。最大可归因区间是 fused expert gate/up，103.03 ms（23.00%）。此表不与无插桩 TPOT 混算。以下 CNPerf 独立运行提供跨进程 API 和设备活动时间线，未启用 fabric PMU。[逐阶段 CSV](../artifacts/logs/deepseek/acceptance.SjYNvMQF/profile/critical-path.csv)

| 排名 / 实测区间 | ms/step | worker 占比 | 源码与耗时来源 | 优化假设、复杂度和数值风险 |
|---|---:|---:|---|---|
| 1 · MoE AllReduce | 21.25 | 12.01% | `stage_c3/moe.py::combine_slots`，每层归并稀疏 routed slots，含等待其他 rank | 与共享专家计算重叠可掩盖部分等待，预计数 ms。复杂度中，保持 SUM、slot 顺序和使用前 wait，需核对 logits 与退出 |
| 2 · MLA O 的 AllGather | 21.00 | 11.86% | `stage_c2/output_projection.py::OutputAxisProjection.__call__`，每层两个 gather 分隔全 K GEMM | 优化 gather 布局或设备拷贝可能降低重复通信开销。复杂度高，GEMM 累加顺序和 BF16 cast 是数值门槛 |
| 3 · MLA Q/KV 投影 | 16.86 | 9.53% | `stage_c2/model.py::WorkerProfile.attach` 插桩每层 `q`、`kv_a`、`kv_b` | 融合能共享输入的投影可减少发射。复杂度高，KV latent norm 依赖与 TP 分片需保留 |
| 4 · Fused expert gate/up | 10.91 | 6.16% | `stage_c3/moe.py::TunableDecoder.project_gate_up`，小行数专家 GEMM | 异构 grouped GEMM 可降低 per-expert 发射。复杂度高，保留 FP32 累加和 BF16 边界 |
| 5 · MoE 路由检查 AllGather | 10.33 | 5.84% | `phase13_4/mla.py::NativeCalls.agree`，交换各 rank 的路由摘要 | 批量校验可能减少小 collective。复杂度高，必须保持跨 rank 不一致检测及 collective 顺序 |
| 6 · MLA O 非 collective | 9.18 | 5.19% | `stage_c2/output_projection.py::OutputAxisProjection.__call__`，包括投影、布局和残差 | 优化小批 GEMM 和布局转换可能减少该区间。复杂度中到高，累加和 cast 顺序敏感 |
| 7 · Shared expert GEMM | 5.55 | 3.14% | `stage_c3/moe.py::TunableDecoder._ffn` 的共享专家剩余投影 | 与 routed slots 通信重叠可能隐藏该区间。复杂度中，依赖等待正确性 |
| 8 · Logits GEMM | 3.72 | 2.10% | `stage_c2/model.py::WorkerProfile.attach` 的 `lm_head.weight` 路由 | 测试单行 GEMM tiling 可减小小批开销。复杂度中，argmax 及近邻 logits 敏感 |
| 9 · Routed expert 剩余 GEMM | 2.92 | 1.65% | `stage_c3/moe.py::TunableDecoder.routed_outputs` 与 `_ffn` 的专家投影 | 合并专家提交可减少发射。复杂度高，保留权重、token 与 slot 对应 |
| 10 · QK | 2.81 | 1.59% | `phase13_4/mla.py::NativeCalls.attention`，读取 K 并生成 logits | QK–Softmax–AV 融合可减小中间缓冲流量。复杂度高，FP16 舍入、mask 和低精度概率边界风险高 |

MoE 记录了 13,312 条 rank/layer 路由观测。热 Decode 按 rank 统计的无本地 routed slot 比例为 37.2–46.3%，最忙 rank 在单层最多处理 6 个 slot 中的 4 个。热 Prefill 单 rank 单层最多 34 个 slot。负载不均是 collective 到达等待的可测线索，尚需设备时间线验证具体因果关联。

Cambricon CNPerf 6.7.5 另采集了默认 C.3 的启动、一个首次请求、一个热请求和清理，含所有八 worker。按整段跨 rank 的聚合统计计，`cnInvokeKernel` 调用 **799,056** 次，API 耗时平均 **7.36 µs**、p90 **9.48 µs**，总计 **5.878 s**。`cnQueueSync` 共 **449,528** 次，累计 **210.242 s**（API 统计 33.13%），含 device idle、wait 和启动期长尾。设备累计时间中，三个最重的 collective kernel 为 BF16 AllReduce、INT32 ring AllReduce 和 BF16 AllGather。设备 memcpy 的 DtoH、HtoD 累计分别为 23.289 s、9.992 s。以上为整段多 rank 累加并可能重叠，包含初始化、首次请求、运行时插桩和 teardown。逐 token 样本以无 CNPerf 基线为准。[CNPerf 原始录制](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cnperf/native.cnperf-rep) · [API 统计](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cnperf/csv/FunctionStatistics.csv) · [Kernel 统计](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cnperf/csv/KernelStatistics.csv) · [拷贝统计](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cnperf/csv/MemoryOpTimeStatistics.csv)

CNPerf 记录的设备分配高水位（GiB，rank 0–7）为 5.194、5.242、5.194、5.226、5.291、5.274、5.403、5.178。每卡还存在 14 条未匹配的 free，且 context 退出的隐式释放未完整呈现。该值仅覆盖录制能匹配的地址分配，设备总占用仍参考 CNMON 采样。[地址配对原始计算](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cnperf/allocator-high-water.csv) · [JSON 时间线](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cnperf/native-timeline.json)

异步候选使用原有 `overlap` 实验开关，只改变 routed-slot SUM 的提交时序，保留 combine 前的 wait。选择依据是本轮可归因 Decode 排名及共享专家的独立计算。历史 probe 显示该变体收益接近测量波动，本轮用完整模型重新测量。[历史 probe](../artifacts/logs/phase15/stage_c3/final_report.md)

## MoE AllReduce overlap A/B

单变量为现有异步 SUM 开关。基线和候选各两个独立引擎启动，每次同样一个首次请求、三个热请求。第二对以候选先行，使用相同的 17→32 固定输入和 BF16 TP8 配置。四次运行都通过冻结序列校验，八 worker 各自退出 0。两种变体的请求后显存快照均为 6.040–6.290 GiB。 候选单独插桩的 512 个 rank/step、每对 102,400 个 FP32 logits，与本轮默认路径逐位一致。[候选数值校验](../artifacts/logs/deepseek/acceptance.SjYNvMQF/overlap_profile/logit-parity.csv)

| 热请求中位数 | 第一对基线 | 第一对 overlap | 第二对基线 | 第二对 overlap |
|---|---:|---:|---:|---:|
| TTFT (ms) | 436.02 | 427.20 | 416.99 | 426.64 |
| 平均 TPOT (ms) | 155.53 | 149.32 | 151.50 | 150.16 |
| p95 ITL (ms) | 174.85 | 164.47 | 170.98 | 172.65 |
| Decode (token/s) | 6.429 | 6.697 | 6.601 | 6.659 |
| 端到端输出 (token/s) | 6.086 | 6.334 | 6.260 | 6.297 |

热 TPOT 两对分别下降 **3.99%**、**0.88%**；合并六个热样本的中位数从 152.94 到 149.74 ms，下降 **2.09%**。合并热 TTFT 从 418.46 到 426.92 ms，上升 **2.02%**。复测中 TTFT 与 p95 ITL 同时回退，冷请求受 22.736 s 单步长尾影响较大，无法建立稳定冷收益。当前生产入口未启用 overlap，继续保留实验开关；可重复且足以支持发布的综合收益尚未建立。20% TPOT 目标未达到，Prefill 与 MoE 目标没有同口径达标证据。[初测基线](../artifacts/logs/deepseek/acceptance.SjYNvMQF/deepseek-benchmark.CoyjTiy8/performance.csv) · [初测 overlap](../artifacts/logs/deepseek/acceptance.SjYNvMQF/overlap/performance.csv) · [复测 overlap](../artifacts/logs/deepseek/acceptance.SjYNvMQF/overlap_repeat/performance.csv) · [复测基线](../artifacts/logs/deepseek/acceptance.SjYNvMQF/baseline_repeat/performance.csv)

## Stock Vendor 对照

stock Vendor 跑固定镜像的 `DeepseekV2ForCausalLM`，没有 NeurX 架构覆盖。同一 checkpoint、八卡 BF16、TP8 CNCL、eager、17→32、并发 1、greedy、禁用 prefix cache，逻辑 cache 配置与 Stage C.3 一致。两边各启动一次，取三个热请求的中位数。两边四次请求均逐个匹配冻结 32 个 token，stock 可作为这个配置的独立 BF16 token oracle。两套栈的内部 KV 表示与 kernel 实现各自由其生产路径决定。[Stock 脚本](../artifacts/logs/deepseek/acceptance.SjYNvMQF/vendor/run.py) · [原始时序](../artifacts/logs/deepseek/acceptance.SjYNvMQF/vendor/performance.csv) · [逐请求 token](../artifacts/logs/deepseek/acceptance.SjYNvMQF/vendor/generation.json)

| 指标 | Stage C.3 | Stock Vendor |
|---|---:|---:|
| 引擎初始化 | 130.846 s | 139.207 s |
| 首次 TTFT | 15,764.10 ms | 5,575.52 ms |
| 热 TTFT | 436.02 ms | 78.92 ms |
| 热平均 TPOT | 155.53 ms | 69.17 ms |
| 热 p95 ITL | 174.85 ms | 75.15 ms |
| 热 Decode 吞吐 | 6.429 token/s | 14.457 token/s |
| 热端到端输出吞吐 | 6.086 token/s | 14.403 token/s |
| 请求后各卡显存快照 | 6.040–6.290 GiB | 4.934 GiB |

本轮 stock 的热 Decode 为原生路径的 2.25 倍，热端到端输出吞吐为 2.37 倍。双方的样本覆盖一个引擎启动和固定请求，无法外推到其它并发或长度。

额外 CPU FP32 eager 对照首次因容器数值 UID 无用户名而退出。补设运行时缓存与用户名后成功加载，并匹配前六个 greedy token。独立 BF16 gate 已由 stock Vendor 完成，CPU 对照在生成中主动停止，保留部分 logits 和日志。[首次尝试](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cpu_reference/runtime.log) · [CPU 重试](../artifacts/logs/deepseek/acceptance.SjYNvMQF/cpu_reference_retry/runtime.log)

## 复现入口

```bash
make deepseek-verify
make test
make deepseek-check
make deepseek TP=8 MAX_TOKENS=32
make deepseek-benchmark WARM_REQUESTS=3
bash integration/vllm_native_mlu/phase15/stage_c3/run.sh /absolute/new/evidence/profile benchmark --experiment fused,batched --warm-requests 1 --profile
make deepseek-chat MAX_TOKENS=8 MAX_MODEL_LEN=128 TEMPERATURE=0
evidence=$PWD/artifacts/logs/deepseek/acceptance.SjYNvMQF
run_dir=$(mktemp -d "$PWD/artifacts/logs/deepseek/reproduce.XXXXXXXX")
DEEPSEEK_CHAT_LOG_ROOT="$run_dir" script -qefc \
  'make deepseek-chat MAX_TOKENS=32 MAX_MODEL_LEN=128 TEMPERATURE=0.7' \
  "$run_dir/chat.tty.log" < "$evidence/chat-truncation-input.txt"
bash "$evidence/vendor/reproduce.sh" "$run_dir/vendor" --warm-requests 3
bash "$evidence/cnperf/run.sh" "$run_dir/cnperf" benchmark --experiment fused,batched --warm-requests 1
bash integration/vllm_native_mlu/phase15/stage_c3/run.sh \
  "$run_dir/overlap" benchmark --experiment fused,batched,overlap --warm-requests 3
```

对比候选全量 logits 时，通过 Python `runpy.run_path` 调用 `stage_c2/report.py` 的 `compare_logits`，输入 profile 目录及候选 profile 目录，输出到新的 CSV。两边的 `--profile` 运行均固定 `--warm-requests 1`。Assembly/SDK 位于冻结镜像。

Phase 9/10 的容器内重编译、fixture、输出及运行参数保存在 [Phase 9.4 日志](../artifacts/logs/deepseek/acceptance.SjYNvMQF/phase9_4/runtime.log) 和 [Phase 10 日志](../artifacts/logs/deepseek/acceptance.SjYNvMQF/phase10/runtime.log)。GPU 入口使用上述固定镜像及空闲设备。

## 分项结论

| 领域 | 判定 | 范围 |
|---|---|---|
| 功能 | PASS | 已验证短上下文 BF16 TP8 真模型、Prefill、增量 Decode、TTY Chat 和历史管理 |
| 数值 | PASS | BF16/FP16 冻结算子矩阵、BF16 TP8 17→32 的 512 个 logits 对照及独立 stock Vendor greedy 序列 |
| 稳定性 | UNKNOWN | 重复初始化及正常关闭通过。TP8 一小时、故障注入、并发压力和内存高水位未闭环 |
| 性能 | BLOCKED | 17→32 基线和独立 Vendor 对照有效。原生慢于该配置的 Vendor，生产入口的长度和单序列约束阻止指定矩阵 |
| 项目整体 | BLOCKED | 先为 TP8 补齐最小的 128→32 矩阵用例及配套 RoPE/cache/scheduler 数值验证，再逐步扩大长度和并发 |

## 上次审计记录

下表保留上次记录的判定口径。当前验收状态见上文。

**Conclusion: Real eight-card inference and multi-turn dialogue have historical PASS, current version final sign-off BLOCKED.** Audit baseline `main@8e134ac`. Current environment has no available MLU, and current console filtering still has reproducible boundary counterexamples.

| Issue | Conclusion |
|---|---|
| Which features are implemented? | Native GEMM, RMSNorm, RoPE/YaRN, QK, Softmax, AV, MLA, device KV Cache, MoE, model loading, prefill, continuous decode, Vendor scheduling/TP, interactive DELTA Chat, historical truncation, statistics, and lifecycle handling. [Source code, reproduction commands, and phase table](PROJECT_FUNCTION_AUDIT.md) |
| Which ones passed on real MLU590? | Native operators and fixed numerical matrices for single-card MLA/KV, Phase 12.5.4 single-card 27-layer generation, Phase 14 TP2, Stage B2 eight-card communication, Stage C.1–C.3 eight-card full model, and Phase 16 multi-turn session. Early Layer 0, old FP16 chain, Phase 11 per-tensor, early Stage C, and some fault attempts still recorded as FAIL/BLOCKED. [Evidence table](PROJECT_FUNCTION_AUDIT.md#phase-coverage) |
| Did TP=8 truly succeed? | **Yes, historical full-chain PASS.** Stage C.1 loaded real checkpoint on eight cards, 17-token prefill followed by 31 incremental decodes, generated 32 IDs total. C.3 against C.2 has 512 rank/step logits bit-level identical, timed requests' 32 generated IDs identical to C.1. B2 alone only proved three rounds of eight-card communication and lifecycle. [C.1 original run](../artifacts/logs/phase15/stage_c1/inference.log) · [C.3 numerical](../artifacts/logs/phase15/stage_c3/logit_parity.csv) · [B2 communication](../artifacts/logs/phase15/stage_b2/cncl_acceptance.csv) |
| Can it do complete inference and multi-turn dialogue? | **Yes, historical BF16, TP8, 128-token total budget path passed.** Four rounds reusing single engine, 32 actual output tokens; also five rounds of real TTY and 14 completed requests session. Current version and longer context each require separate acceptance. [Multi-turn test results](../artifacts/logs/phase16/test_results.md) · [14 requests](../artifacts/logs/phase16/runs/chat.BXuog9ba/interactive_requests.csv) |
| Does Phase 16.3 truncate silently and preserve statistics? | Current local truncation branch silent, CPU regression proves `dropped_turns` preserved and written to request record. 16.3's formal hardware run was once blocked by resource contention. Subsequent 16.4.1 filtering single complete notification passed, but audit's two-segment output reproduced original text, strict silence **FAIL**; final hardware TTY re-verify **BLOCKED**. [Truncation and filtering source](../integration/vllm_native_mlu/phase16/deepseek_chat.py) · [16.3 formal acceptance](../artifacts/logs/phase16/phase16_3/PHASE16_3_ACCEPTANCE.md) |
| What level of performance achieved? | C.3 fixed warm request median TTFT 424.3 ms, average TPOT 153.5 ms, p95 TPOT 170.3 ms, decode 6.515 token/s, end-to-end 6.155 token/s. Per-card 6.040–6.290 GiB after request. Performance improvement over C.2 warm, cold end-to-end throughput declined. Missing stock Vendor same-condition full-model A/B, TP8 long-run/fault matrix, and acceptance thresholds. [Performance raw table and method](../artifacts/logs/phase15/stage_c3/final_report.md) |
| Historical features provable; current strict silence, board re-verify, and performance comparison still have gaps. [Requirements and gaps](FINAL_ACCEPTANCE_GAP.md) |

- **P1**: Complete same-condition stock Vendor A/B, per-card peak memory, and TP8 long-run/concurrent/anomaly matrix.
- **P2**: Incorporate real emitter cross-segment counterexamples into regression, improve version, evidence, and phase numbering traceability.

This submission includes only three reports; no changes to Vendor source, model weights, or historical acceptance evidence.
