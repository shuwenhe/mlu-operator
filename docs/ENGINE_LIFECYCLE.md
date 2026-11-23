# Engine lifecycle and fault recovery

- [GLM53 shutdown and MLU release](#glm53-graceful-shutdown-report)
- [Phase 14.2 — Vendor lifecycle audit](#phase14-2-vendor-lifecycle-audit)
- [Phase 14.2 — Lifecycle correction](#phase14-2-implementation)
- [Phase 14.2 — Numerical regression](#phase14-2-numerical-regression)
- [Phase 14.2 — Repeated TP=2 Engine Lifecycle Acceptance](#phase14-2-repeated-lifecycle-report)

<a id="glm53-graceful-shutdown-report"></a>

## GLM53 shutdown and MLU release

Executed on `tza-0a09-ai01-em9` on 2026-11-21 (Asia/Shanghai). The user relayed Li Tianchu's authorization to stop the GLM53 service. Scope was `glm53_serve` only.

### Pre-shutdown

| Check | Result | Evidence |
| --- | --- | --- |
| Container identity and status | PASS | Running container `6361c56e3aea1a4388776ad090634480587f8a21c760b4ff34131173be76d147`, image `teleavatar-mlu:v0.1`, ID `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. Restart policy `no`. Default stop signal SIGTERM. |
| vLLM process identity | PASS | Container PID 1 was Bash (host PID 3637335). Server was namespace PID 9 (host PID 3637351). EngineCore host PID 3640298 and eight workers 3642753–3642760. Model path and port matched the request. |
| Active request snapshot | PASS | Two read-only queries to port 8000 metrics reported zero running and zero waiting requests. The container log showed no recent POST entries in the five-minute window. The separate serving log had historical POST entries. The metrics are snapshots. Request activity between the final sample and signal was not independently monitored. |
| Graceful signal path | PASS | Launcher script uses `exec` for vLLM beneath Bash. SIGTERM was directed at the verified server process. A separate drain endpoint was not verified. |
| MLU baseline | PASS | CNMON showed all eight initialized and 93,556 MiB charged to a vLLM worker on each card. |

Full pre-shutdown outputs are in [glm53_before_shutdown.log](../artifacts/logs/glm53_before_shutdown.log). The initial template lookup for the optional stop signal failed. The follow-up JSON inspection recovered the identity, restart policy and default signal.

### Shutdown and release

| Check | Result | Evidence |
| --- | --- | --- |
| API shutdown signal | PASS | At 14:16:17, `docker exec --user 0 glm53_serve sh -ec 'test "$(cat /proc/9/comm)" = vllm; kill -TERM 9'` exited 0. |
| Engine shutdown behavior | NOT VERIFIED | The application log confirms API shutdown and EngineCore receipt of SIGTERM, completion of its request-processing phase and start of its resource teardown. The deployed vLLM used `mode=abort timeout=0s`, and its own process manager logged `force killing remaining processes count=1`. No external force-kill command was sent. The logs do not prove every child completed graceful teardown. |
| Graceful Docker stop | PASS | After the vLLM server exited, `timeout 120 docker stop --time -1 glm53_serve` returned 0. The outer timeout limits the CLI wait. Docker's indefinite timeout avoids an automatic SIGKILL if the grace period expires. Container state is `exited`, exit code 0, PID 0. |
| Worker and port check | PASS | Server, EngineCore and all eight recorded worker PIDs were absent. No remaining listener on port 8000 was observed. |
| MLU release and CNDEV | PASS | CNMON exited 0 and reported all eight cards with 0 MiB used and 97,061 MiB free of 98,304 MiB. No residual GLM53 allocation was observed. |
| Unrelated runtime status | PASS | `ta-mlu-team1` and `ta-mlu-team2` remained running with their existing device mappings. Neither received a shutdown or configuration change. |

See the [shutdown transcript](../artifacts/logs/glm53_shutdown.log) and [post-shutdown CNMON and process log](../artifacts/logs/glm53_after_shutdown.log).

### Recovery and operator gates

| Check | Result | Evidence |
| --- | --- | --- |
| Container and image recovery | PASS | Same container and image IDs remain. The bind mount, restart policy and device settings were preserved. No restart, removal or reconfiguration was issued. |
| Model preservation | PASS | The bind-mounted model directory remains accessible. The post-check found 88 matching config, tokenizer or safetensors files. No model edits or deletions were issued. Content checksums and a restart test were not performed. |
| Exclusive single-card context | BLOCKED | The two unrelated live containers still have explicit device mappings covering cards 0–3 and 4–7. Released GLM53 memory does not establish exclusive use by the acceptance process. No existing mapping was changed. |
| CNRT initialization and SDK | BLOCKED | Host CNDEV passed. Host glibc is 2.31 and only CNDev was found in the vendor loader cache. The previously linked Ubuntu 24.04 SDK image is available. CNRT visibility and transfer correctness in an approved single-card context are pending. |
| RMSNorm hardware checks | BLOCKED | No BANG C or CNNL fixture was launched. FP16/BF16 correctness for widths 512, 2048, 65, 127 and 193, canaries, PyTorch comparison, kernel and transfer-inclusive latency remain pending. Follow the existing [hardware acceptance procedure](MLU590_M9_RMSNORM_HARDWARE_ACCURACY.md#deepseek-rmsnorm-hardware-acceptance) once the device access gates are satisfied. |

<a id="phase14-2-vendor-lifecycle-audit"></a>

## Phase 14.2 — Vendor lifecycle audit

The frozen image contains Vendor `vllm_mlu 0.25.1+mlu0.16.0.pt212`, `torch_mlu 1.34.1+torch2.12.1`, and PyZMQ 27.1.0. The installed source snapshots and package manifest are in the Phase 14.2 evidence directory.

### Same-process EngineCore

`InprocClient.shutdown(timeout)` calls `EngineCore.shutdown()` on the caller's thread. It does not forward or enforce the timeout. EngineCore clears structured-output state, shuts down the executor, shuts down the scheduler, then runs Vendor distributed and allocator cleanup. That cleanup calls `gc.collect()` after destroying the process groups.

The executor owns its broadcast queue and references each rank's response queue. Each `MessageQueue` creates its own PyZMQ context. The queue retains its data sockets and `SpinCondition` retains its notify and reader cancellation sockets. Context socket tracking uses weak references. The Vendor queue's `shutdown()` sets a flag and wakes blocked readers. It has no explicit socket-close or context-terminate operation. The executor signals child processes over their death pipes, waits for them, calls queue shutdown and clears its queue references. Each worker handles the death event, shuts down its local queues and destroys its process groups.

`Context.destroy()` takes the sockets discoverable through its tracking set, closes them and calls the synchronous `term()`. Destruction must run after socket users have finished. Worker-exit and monitor-thread ordering matter when deciding where the owning process can safely close its sockets.

### Independent EngineCore

The multiprocess frontend client owns its own IPC endpoints and finalizer and waits on a separate EngineCore process through its process manager. It has a separate timeout and resource cleanup sequence. The executor and its shared-memory queue contexts still reside with the EngineCore. Frontend shutdown completion therefore does not establish orderly executor queue termination. The Phase 14.1 separate-process fault attempt illustrates this distinction.

<a id="phase14-2-implementation"></a>

## Phase 14.2 — Lifecycle correction

The new `LifecycleExecutor` is selected through the existing Vendor `distributed_executor_backend` argument. It uses Vendor worker creation, request execution, failure monitoring and shutdown. It captures strong references to the existing parent-owned queue sockets before the Vendor monitor can release the queue references.

The lifecycle executor calls Vendor shutdown first. Once Vendor has signalled and waited for workers, the parent owner thread closes its sockets with zero linger and terminates their contexts. The monitor thread may request Vendor worker teardown and wake pending readers. It defers socket closure to the owner thread. A lock serializes shutdown across the two callers. A subsequent owner shutdown has no sockets left to close.

The Phase 14.1 launcher and model sources are preserved. The Phase 14.2 replay supplies the lifecycle executor to the same frozen inference driver. The launcher uses the same image and original configuration.

Select the fix with `LLM(..., distributed_executor_backend=LifecycleExecutor)`. The original default executor remains available for exact reproduction. Keep Engine creation, request stepping, and explicit client shutdown on its owner thread. The Vendor worker monitor continues to start after worker readiness and before the queue handshakes.

The first hardware retry closed the ZeroMQ sockets and exited, but repeated engines left four descriptors per round. The saved parent descriptor list shows two pipe pairs per cycle. The executor now waits for its Vendor liveness monitor to finish, records both worker exit codes, closes the exited Python process handles, and detaches the bound Vendor exit finalizer. The failure probe's forced container cleanup was used to end that failed experiment and is outside accepted shutdown evidence.

<a id="phase14-2-numerical-regression"></a>

## Phase 14.2 — Numerical regression

The lifecycle executor leaves the original TP=1 route and the Phase 14.1 model, Native MLA, Ordered O and KV cache unchanged. The TP=2 normal-shutdown acceptance ran the frozen length-one prompt through the production `LLM.generate()` path. Both prefill and incremental decode produced the frozen greedy tokens and passed four rank/logit comparisons.

The recovery driver uses the same frozen fixture and tolerance on a newly initialized engine after each fault. The separate TP=1 launch uses the pinned image, original checkpoint and original implementation snapshots. The final gate record reports the new run counts and distinguishes prior Phase 14.1 results from Phase 14.2 evidence.

<a id="phase14-2-repeated-lifecycle-report"></a>

## Phase 14.2 — Repeated TP=2 Engine Lifecycle Acceptance

- Run timestamp: `2026-11-22T20:49:46+08:00`
- Evidence root: `artifacts/logs/phase14_2/repeated_lifecycle/run_20261122_204946`
- Model: DeepSeek-V2-Lite
- Hardware: 2 × MLU590-M9
- Runtime: frozen Vendor vLLM image (`sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`)

### Gate Result

Overall repeated lifecycle gate: **PASS**

### Acceptance Summary (10/10)

| Gate | Result |
|---|---|
| Fresh Engine creation | PASS (10/10) |
| Real TP=2 inference (`LLM.generate`) | PASS (10/10) |
| Frozen logits parity | PASS (10/10) |
| Greedy token parity | PASS (10/10) |
| Rank 0 cleanup | PASS (10/10, exit code 0) |
| Rank 1 cleanup | PASS (10/10, exit code 0) |
| EngineCore cleanup | PASS (10/10) |
| Parent cleanup | PASS (10/10) |
| ZeroMQ cleanup | PASS (10/10, socket count 0) |
| FD stability | PASS (no cumulative growth) |
| Thread stability | PASS (no unexplained growth) |
| Process stability | PASS (no orphan worker/engine processes observed at boundaries) |
| MLU resource stability | PASS (`cnmon` boundary checks clean) |

### Resource Trend

| Iteration | FD | Threads | Child Processes | ZMQ Sockets | MLU Memory | Result |
|---|---:|---:|---:|---:|---|---|
| Baseline | 6 | 2 | 0 | 0 | N/A | N/A |
| 1 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 2 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 3 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 5 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 10 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |

### FD Stability Conclusion

- In-process baseline: FD=6
- Iteration 1/2/3/5/10 after-cleanup: FD=10 (+4 from baseline)
- No monotonic growth across 10 cycles (stable at 10 after first-cycle warm retention)
- ZeroMQ sockets remained 0 at every cycle boundary
- No unreleased worker process handles (worker exit codes consistently `[0, 0]`)

### Numerical Regression Conclusion

- Frozen logits checks remained within tolerance across all cycles.
- Greedy token IDs matched frozen references in all cycles (`[254, 4334]` for the fixed prompt in this gate).
- No NaN/Inf reported in logged logits parity metrics.
- No CPU fallback evidence was observed in this gate.

### Evidence Files

All required deliverables are under:

`artifacts/logs/phase14_2/repeated_lifecycle/run_20261122_204946/`

- `repeated_lifecycle.csv`
- `repeated_lifecycle_summary.json`
- `numerical_results.csv`
- `fd_inventory.csv`
- `fd_diff.csv`
- `thread_inventory.csv`
- `process_inventory.csv`
- `zmq_cleanup.csv`
- `cnmon_snapshots.log`
- `cleanup.log`
- `resource_trend.md`
- `source_hashes.txt`
- `phase14_2.sha256`
