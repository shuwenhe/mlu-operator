# Phase 13.5 — DeepSeek-V2-Lite Native MLU TP=8 Production Integration Plan

## Scope and constraints
- Repository: `/app/shuwen/mlu-operator`
- Frozen runtime/image: `teleavatar-mlu:v0.1` (`sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`)
- Hardware target: 8 × MLU590-M9
- Frozen tolerance: `abs(error) <= 1e-5 + 1e-5 * abs(reference)`
- No modification to Phase 12.5.4 frozen references/fixtures.
- No silent fallback and no PASS without hardware evidence.

## Preflight audit (completed before code modification)
- Audit log: `artifacts/logs/phase13_5/preflight_audit.log`
- SHA manifest: `artifacts/logs/phase13_5/preflight_sha256.txt`
- Verified presence of mandatory baseline evidence:
  - Phase 12.5.4 acceptance and source hashes
  - Phase 13.1 CNCL probe evidence
  - Phase 13.2 acceptance
  - Phase 13.3.1 architecture/mapping docs
  - Phase 13.3.2 failure evidence
  - Phase 13.3.3 ordered-O acceptance
  - Phase 13.4 corrected replay closure

## Integration strategy (incremental)
1. **Boundary audit and wiring design**
   - Locate production entry points used by Vendor vLLM worker init and model execution.
   - Define TP=8-only integration boundary; preserve TP=1 path behavior.
   - Confirm process-group/communicator ownership and stream ordering invariants.

2. **Production path integration (minimal/surgical)**
   - Integrate validated native MLA path into actual Vendor vLLM execution path:
     - Query/KV-A/KV-B projections
     - RoPE
     - two-head local attention
     - rank-local KV cache write/read
     - ordered O compatibility
     - FP32 accumulation, BF16 cast, residual add
   - Keep TP=1 production path unchanged and protected by regression gate.

3. **Distributed KV lifecycle validation**
   - Validate block/slot ownership and metadata consistency across ranks.
   - Validate prefill/decode mixed scheduling, block release/reuse.
   - Inject invalid metadata and verify coherent failure without collective deadlock.

4. **Numerical acceptance (frozen references)**
   - Context lengths: 1/7/17/65
   - Prefill + ≥3 decode steps
   - Compare required boundaries (Q/K/V, RoPE, QK/softmax/AV, O FP32 partial, ordered FP32 sum, BF16 O, residual, logits, token IDs).
   - Stop at first unexplained mismatch and preserve logs.

5. **End-to-end Vendor vLLM engine acceptance**
   - Real engine run (not isolated probe): TP=1/2/4/8
   - Single + concurrent + mixed requests, restart cycle
   - Deterministic token checks where supported.

6. **Performance and reliability**
   - Performance: ≥100 repetitions/config where supported; report slowest-rank p50/p95.
   - Reliability: continuous ≥1h hardware run, no deadlock/divergence/leak/fallback.

7. **Independent review**
   - Conduct a dedicated read-only review of integration boundaries, TP ownership, CNCL ordering, BF16 rounding, RoPE positions, KV lifecycle, failure handling, TP=1 protection.

## Planned outputs for this phase
- `docs/PHASE13_5_NUMERICAL_ACCEPTANCE.md`
- `docs/PHASE13_5_PERFORMANCE_REPORT.md`
- `docs/PHASE13_5_RELIABILITY_REPORT.md`
- `docs/PHASE13_5_ARCHITECTURE_AUDIT.md`
- `artifacts/logs/phase13_5/` (raw logs, per-rank CSV, summaries, cleanup evidence, SHA256 manifests)

## Current status
- **Phase 13.5:** IN PROGRESS (audit complete, implementation not started in this document).
