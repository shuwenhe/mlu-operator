# Phase 13.4 Final Numerical Evidence Closure

## Scope
- Repository: `/app/shuwen/mlu-operator`
- Runtime image: `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756` (`teleavatar-mlu:v0.1`)
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Hardware: MLU TP=8 (`mlu:0..7`)
- Frozen fixture policy:
  - required: `artifacts/build/phase12_5_4/native-dump-yarn-scale`
  - forbidden: `artifacts/build/phase12_5_4/native-dump-yarn`

## What was closed in this pass
1. Verified 27-layer execution evidence from corrected replay outputs.
2. Recorded runtime-resolved fixture absolute path + SHA256 per rank.
3. Added explicit O boundaries:
   - local FP32 O partial
   - CNCL FP32 SUM
   - BF16 O output
   - BF16 residual output
4. Verified rank/device mapping for all 8 ranks.
5. Verified cleanup events for all ranks and captured final CNMON snapshot attempt.

## Evidence (non-overwriting)
- Corrected closure layered replay: `artifacts/logs/phase13_4/corrected_replay_closure/tp8_test_gate8_closure3.log`
- Corrected closure full-model replay: `artifacts/logs/phase13_4/corrected_replay_closure/tp8_full_model_closure3.log`
- Closure summary: `artifacts/logs/phase13_4/corrected_replay_closure/closure3_summary.json`
- CNMON snapshot: `artifacts/logs/phase13_4/corrected_replay_closure/cnmon_final_state.log`
- Preserved previous failed closure attempt: `artifacts/logs/phase13_4/corrected_replay_closure/tp8_test_gate8_closure2.log`

## Required distinction
### A) Original diagnostic failure cause
- The original divergence investigation used wrong fixture lineage (`native-dump-yarn`) in older launcher paths.
- The corrected acceptance path explicitly enforces `native-dump-yarn-scale` (see precheck/audit evidence).

### B) Corrected fixture validation result
- `closure3_summary.json` reports:
  - `fail_event_count = 0`
  - `all_27_layer_gates_passed_on_8_ranks = true`
  - `all_27_layers_seen_in_full_model = true`
  - `fixture_recorded_for_all_ranks = true`
- Runtime fixture path + SHA256 is recorded on all ranks in accuracy rows (`fixture_path`, `fixture_sha256`).

### C) Remaining genuine failures
- No numerical tolerance failure was observed in corrected closure replay logs.
- No runtime gate failure was observed in corrected closure replay logs.

## Explicit gate outcomes for this closure
- Corrected fixture validation: **PASS**
- Layer-level numerical acceptance (TP8, all 27 layers): **PASS**
- Full 27-layer prefill/decode acceptance: **PASS**
- End-to-end logits acceptance (`model_logits` rows): **PASS**
- End-to-end token-id acceptance (`token` rows): **PASS**
- Rank/device mapping (0..7 -> mlu:0..7): **PASS**
- Rank cleanup events (0..7): **PASS**
- Final CNMON state capture command: **PASS (command exit code 0; command output empty in this image)**

## Notes
- No production kernel logic was modified for numerical “compensation”.
- Existing frozen references and tolerances were preserved.
- Prior failed closure logs were preserved and not overwritten.
