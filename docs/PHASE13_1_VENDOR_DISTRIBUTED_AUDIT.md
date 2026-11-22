# Phase 13.1 — Vendor vLLM MLU distributed architecture

The audit uses the Phase 12 Vendor image digest `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. The hardware run uses eight MLU590-M9 devices. The runtime reports CNCL 1.30.10, CNRT 7.7.3 and driver 6.5.52.

## Worker and process groups

- `MLUPlatform` advertises `cncl` and resolves the device communicator to `MLUCommunicator`.
- `MLUWorker.init_device` maps the local rank to a visible MLU and sets it before calling the base worker distributed initializer. It then calls `initialize_model_parallel_mlu`, which manages additional Vendor groups for attention sequence parallelism, embedding/logits, dense MLP, and MoE TP/EP. Vendor MoE group creation consults `enable_expert_parallel`.
- Base vLLM `GroupCoordinator` creates device and CPU groups. The MLU communicator implements AllReduce using `torch.distributed.all_reduce` on the device group. Fixed-size AllGather and ReduceScatter inherit implementations that call `all_gather_into_tensor` and `reduce_scatter_tensor`. The Vendor variable-size paths have separate code.
- The experiment calls the same base vLLM group code with a TP=8 configuration, disables custom AllReduce, and validates the actual TP backend, communicator and local MLU on each rank. It exercises the three fixed-size collectives without loading model weights. Vendor-specific auxiliary groups and the full engine-worker lifecycle are covered by the source audit only.

[Vendor source hashes and symbol locations](../artifacts/logs/phase13_1_source_audit.log)

## Implications for NeurX DeepSeek-V2-Lite

- **Linear layers.** Vendor DeepSeek imports its ColumnParallel, RowParallel and replicated linear paths. For the native model, column-parallel checkpoint loading must slice output channels, including the separate gate/up portions. Row parallel must slice inputs and reduce partial outputs before applying any replicated bias or residual.
- **MLA and cache.** Vendor attention reshapes Q/KV using the local head count and has compressed latent-cache decode paths. The checkpoint has 16 attention heads, giving two per rank at TP=8. The frozen native attention path uses materialized K/V. A native TP path must size per-rank cache pages for two heads and preserve consistent scheduler slot/block ownership across ranks.
- **MoE.** The checkpoint specifies 64 routed and two shared experts. Eight-way EP can assign eight routed experts to each rank. Routing and combine must preserve checkpoint expert IDs and top-k weights across ranks. Shared-expert execution and its final contribution require a separate test.
- **Acceptance.** The collective fixture can establish device communication and rank participation. Weight ownership, prefill/decode logits, greedy tokens, cache lifecycle, and model performance require subsequent hardware runs against the frozen single-card reference.

## Reproduction

Run on the MLU host where the eight devices and pinned local Docker image are available.

```bash
set -o pipefail
bash integration/vllm_native_mlu/phase13_1/run_cncl_probe.sh 2>&1 | tee artifacts/logs/phase13_1_cncl_final.log
```

## Hardware result — 2026-11-22

**PASS for the TP communication probe.** The launcher exited 0. Ranks 0–7 each used their matching MLU ordinal and reported `cncl` and `MLUCommunicator`. AllReduce, AllGather and ReduceScatter matched the expected tensors exactly on every rank for FP32 and BF16, a total of 48 checks. CNCL reported MLU-Link rings. This is a correctness fixture using 2×16 elements per rank for AllReduce and AllGather and 16×16 input per rank for ReduceScatter.

Startup retries resolved hostname resolution in the isolated container with loopback rendezvous, Python cache initialization with writable temporary cache paths, and TP initialization with an active vLLM config. The experiment used the frozen image and left the Phase 12.5.4 model untouched. The RMSNorm/GEMM scaling CSVs and the Phase 12.5.4 token fixture retained their pre-run SHA-256 hashes.

[Per-rank results](../artifacts/logs/phase13_1_cncl_accuracy.csv) · [Raw hardware log](../artifacts/logs/phase13_1_cncl_final.log)
