# Final Acceptance Gap

Audit baseline is `main@8e134ac`. Current conclusion: **Historical features PASS, final sign-off BLOCKED**. Current environment has no available MLU. For complete technical index, see [Project Function Audit](PROJECT_FUNCTION_AUDIT.md).

## Explicit Requirements and Evidence

| Requirement | Current Determination | Evidence and Closure Conditions |
|---|---|---|
| Only check specified repo and DeepSeek-V2-Lite | PASS | Checked four shards/index with 5,291 BF16 tensors; historical launcher corresponds to same model path. Phase records detailed in function audit |
| Phase 1–16.3 and Stage B2 archived item-by-item | PASS (current evidence inventory) | Phases 1–16.3 and early baseline in function audit. 16.1, 16.2 each have independent acceptance criteria **UNKNOWN**; requires corresponding original tasks or historical logs |
| QK/Softmax/AV/MLA/KV/MoE/Prefill/Decode hardware numerical values | PASS (historical frozen matrix) | Phase 9–10 device replay, Phase 12.5.4, and Stages C.1–C.3. Early FP16 chain/Layer 0 and Phase 11 full-network tensor FAIL preserved; no evidence covering all inputs and dtypes. [Detailed operator matrix](PROJECT_FUNCTION_AUDIT.md#operator-evidence) |
| TP=8 must truly use eight-card inference | PASS (historical), current re-verification BLOCKED | B2 is real eight-card communication; C.1–C.3 is actual eight-card model inference. When verifying current product packaging, after authorized SDK runs `make deepseek-check`, run `make deepseek TP=8 MAX_TOKENS=32`. Preserve per-rank model/device, real weights, forward, 32 IDs, logits comparison, exit, and memory. [C.1](../artifacts/logs/phase15/stage_c1/final_report.md) · [C.3](../artifacts/logs/phase15/stage_c3/final_report.md) · [Historical stable default path](../artifacts/logs/phase15/stage_d/make_runs/deepseek.18glKfkW/launch.txt) |
| Real weights, complete forward, continuous tokens, multi-turn Chat | PASS (historical BF16 TP8 short context) | 27-layer real generation with single engine load and multi-turn session has original records. Applies to current 128-token total budget and controlled test cases. Maintain `--model-dir` and C.3 routing; re-test Chinese/English, cross-turn memory, greedy sequence, reset/exit. [Device actual run and commands](../artifacts/logs/phase16/test_results.md) |
| Console silent on truncation, preserve `dropped_turns` | Self truncation and CSV **PASS (CPU)**; any fragmentation strictly silent **FAIL (CPU counterexample)**; current hardware **BLOCKED** | Old hardware records have actual truncation and old notifications; current full string filtering can intercept single emit, but two output fragments can combine to show original text. Use actual production emitter for single, fragmented, and normal user text testing, complete all stdout/stderr TTY acceptance. [Current source](../integration/vllm_native_mlu/phase16/deepseek_chat.py) · [Historical notifications and generation](../artifacts/logs/phase16/runs/chat.BXuog9ba/chat_smoke_test.log) · [Statistics CSV](../artifacts/logs/phase16/runs/chat.BXuog9ba/interactive_requests.csv) |
| TTFT, TPOT, throughput, memory, stability | Partial measurements PASS, production performance and stability overall unaccepted | C.3 has same-parameter component variants and per-card snapshots; TP1 one hour, TP2 ten iterations, TP8 multiple rounds each independent. Requires TP8 long-run and anomaly tests with same config, peak memory per card, and clear measurement criteria. [C.3 performance](../artifacts/logs/phase15/stage_c3/benchmark_summary.csv) · [TP1 long run](../artifacts/logs/phase12_6_reliability_1h_summary.json) · [TP2 ten iterations](../artifacts/logs/phase14_2/repeated_lifecycle/run_20261122_204946/repeated_lifecycle.csv) |
| A/B comparison with Vendor vLLM under same conditions | BLOCKED (evidence missing and no device this time) | Vendor original smoke, CNNL operator comparison, C.2/C.3 native variant comparison have records. Requires running stock Vendor comparison under same checkpoint, image, eight cards, precision, sampling, prompt, output length, batch, cache settings and cold/warm criteria. [Vendor smoke](../artifacts/logs/vllm_mlu_hardware.log) · [Current C.3 method](../artifacts/logs/phase15/stage_c3/final_report.md) |

## Requirement Attribution

This audit checked README, Makefile, docs, phase reports, source code, scripts, and Git commit messages within permitted scope, but found no attributable original text. Below are technical proxy evidence items found. Historical engineering report gates are not personal approval records.

| Unconfirmed Personal Acceptance Terms | Original Requirement | Cross-checkable Technical Evidence |
|---|---|---|
| Whether specified model, device type, and TP=8 are personal delivery criteria | UNKNOWN | This request clarified these technical items; C.1–C.3 historical hardware already covers |
| Accept BF16, max 128 tokens, max 32 tokens per round by default, or must be longer | UNKNOWN | Current Stage C.3 and Chat constraints; longer model config does not mean current native path supports it |
| Require all-native arithmetic plus eliminate host route scheduling, or allow Vendor scheduler | UNKNOWN | C.3's Vendor engine/CNCL coexists with native math path and host route ID scheduling |
| Token, logits, error threshold, and independent oracle criteria | UNKNOWN | Historically frozen `1e-5 + 1e-5*abs(ref)`, C.3 bit-level to C.2 and fixed sequence; early Phase 11 independent full-tensor has FAIL |
| TTFT/TPOT, throughput, peak memory, and Vendor minimum multiple | UNKNOWN | Repository has comparison data and measurement definitions; lacks explicit personal thresholds and stock Vendor A/B under same conditions |
| Concurrency, continuous operation, failure, duration, and console experience sign-off scope | UNKNOWN | Have single-card, dual-card, eight-card records at different scopes; this request additionally requires truncation silence and statistics preservation |
| Whether personal final sign-off has already passed | UNKNOWN | No sign-off original or corresponding commit record |

## Priorities

### P0 · Final Sign-off Threshold

1. On eight explicitly allocated and pre-checked MLU590-M9 cards, lock current code, checkpoint, image, and actual routing, then sequentially re-verify `make deepseek-check`, `make deepseek TP=8 MAX_TOKENS=32`, `make deepseek-chat MAX_TOKENS=32 MAX_MODEL_LEN=128`. Separately preserve evidence and verify all eight cards' rank, forward, continuous tokens, logs, on-card memory, and exit. Current device conditions **BLOCKED**.
2. Close console silence gap. First distinguish application self-reporting, engine/worker direct writes, and model-generated text; cover production emitter, fragment splitting, and actual TTY. Acceptance must construct actual `dropped_turns > 0` complete rounds, collect stdout/stderr, request CSV, and worker cleanup. Currently have reproducible CPU counterexample ensuring strict silence **FAIL**.
3. For obtained original text and sign-off criteria, bind each condition to command, archive path, threshold. Clarify whether stock Vendor performance is required before sign-off, and whether 128 tokens acceptable. Currently **UNKNOWN**.

### P1 · Production Comparison and Stability

1. Test stock Vendor vLLM against same-version Stage C.3 under same eight-card TP8, BF16, checkpoint, sampling, and request matrix; separately report initialization, cold/warm TTFT, TPOT p50/p95, decode/end-to-end throughput, input/output tokens, per-card snapshots and peak. Use 17→32, repeat three times alternately, cover concurrency and session distribution in sign-off requirements. 
2. Complete final TP8 continuous operation, long session, concurrent scheduling, cache release/reuse, anomaly cancellation, worker loss, and fault recovery evidence. Current Phase 12.6 TP1 long run and Phase 14.2 TP2 ten rounds cannot substitute. If providing OpenAI interface, cross-round prefix reuse, or longer context are explicit requirements, implement and accept each separately.

### P2 · Maintainability and Traceability

1. Each acceptance records precise commit, active script SHA, image, model inventory, BDF/ordinal, actual command, exit code, reference result, and original evidence path. Preserve failed and BLOCKED runs, complete historical 16.1/16.2 numbering mapping.
2. Incorporate discovered cross-fragment counterexamples into production emitter testing, periodically re-run frozen Phase 9/10/12/C-stage numerical regression. Historical inventory preserved by capture-time path, specifically mark live file drift.
