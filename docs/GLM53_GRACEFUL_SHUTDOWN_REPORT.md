# GLM53 shutdown and MLU release

Executed on `tza-0a09-ai01-em9` on 2026-11-21 (Asia/Shanghai). The user relayed Li Tianchu's authorization to stop the GLM53 service. Scope was `glm53_serve` only.

## Pre-shutdown

| Check | Result | Evidence |
| --- | --- | --- |
| Container identity and status | PASS | Running container `6361c56e3aea1a4388776ad090634480587f8a21c760b4ff34131173be76d147`, image `teleavatar-mlu:v0.1`, ID `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. Restart policy `no`. Default stop signal SIGTERM. |
| vLLM process identity | PASS | Container PID 1 was Bash (host PID 3637335). Server was namespace PID 9 (host PID 3637351). EngineCore host PID 3640298 and eight workers 3642753–3642760. Model path and port matched the request. |
| Active request snapshot | PASS | Two read-only queries to port 8000 metrics reported zero running and zero waiting requests. The container log showed no recent POST entries in the five-minute window. The separate serving log had historical POST entries. The metrics are snapshots. Request activity between the final sample and signal was not independently monitored. |
| Graceful signal path | PASS | Launcher script uses `exec` for vLLM beneath Bash. SIGTERM was directed at the verified server process. A separate drain endpoint was not verified. |
| MLU baseline | PASS | CNMON showed all eight initialized and 93,556 MiB charged to a vLLM worker on each card. |

Full pre-shutdown outputs are in [glm53_before_shutdown.log](../artifacts/logs/glm53_before_shutdown.log). The initial template lookup for the optional stop signal failed. The follow-up JSON inspection recovered the identity, restart policy and default signal.

## Shutdown and release

| Check | Result | Evidence |
| --- | --- | --- |
| API shutdown signal | PASS | At 14:16:17, `docker exec --user 0 glm53_serve sh -ec 'test "$(cat /proc/9/comm)" = vllm; kill -TERM 9'` exited 0. |
| Engine shutdown behavior | NOT VERIFIED | The application log confirms API shutdown and EngineCore receipt of SIGTERM, completion of its request-processing phase and start of its resource teardown. The deployed vLLM used `mode=abort timeout=0s`, and its own process manager logged `force killing remaining processes count=1`. No external force-kill command was sent. The logs do not prove every child completed graceful teardown. |
| Graceful Docker stop | PASS | After the vLLM server exited, `timeout 120 docker stop --time -1 glm53_serve` returned 0. The outer timeout limits the CLI wait. Docker's indefinite timeout avoids an automatic SIGKILL if the grace period expires. Container state is `exited`, exit code 0, PID 0. |
| Worker and port check | PASS | Server, EngineCore and all eight recorded worker PIDs were absent. No remaining listener on port 8000 was observed. |
| MLU release and CNDEV | PASS | CNMON exited 0 and reported all eight cards with 0 MiB used and 97,061 MiB free of 98,304 MiB. No residual GLM53 allocation was observed. |
| Unrelated runtime status | PASS | `ta-mlu-team1` and `ta-mlu-team2` remained running with their existing device mappings. Neither received a shutdown or configuration change. |

See the [shutdown transcript](../artifacts/logs/glm53_shutdown.log) and [post-shutdown CNMON and process log](../artifacts/logs/glm53_after_shutdown.log).

## Recovery and operator gates

| Check | Result | Evidence |
| --- | --- | --- |
| Container and image recovery | PASS | Same container and image IDs remain. The bind mount, restart policy and device settings were preserved. No restart, removal or reconfiguration was issued. |
| Model preservation | PASS | The bind-mounted model directory remains accessible. The post-check found 88 matching config, tokenizer or safetensors files. No model edits or deletions were issued. Content checksums and a restart test were not performed. |
| Exclusive single-card context | BLOCKED | The two unrelated live containers still have explicit device mappings covering cards 0–3 and 4–7. Released GLM53 memory does not establish exclusive use by the acceptance process. No existing mapping was changed. |
| CNRT initialization and SDK | BLOCKED | Host CNDEV passed. Host glibc is 2.31 and only CNDev was found in the vendor loader cache. The previously linked Ubuntu 24.04 SDK image is available. CNRT visibility and transfer correctness in an approved single-card context are pending. |
| RMSNorm hardware checks | BLOCKED | No BANG C or CNNL fixture was launched. FP16/BF16 correctness for widths 512, 2048, 65, 127 and 193, canaries, PyTorch comparison, kernel and transfer-inclusive latency remain pending. Follow the existing [hardware acceptance procedure](DEEPSEEK_RMSNORM_HARDWARE_ACCEPTANCE.md) once the device access gates are satisfied. |
