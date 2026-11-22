# Phase 12.5.1 / 12.5.2 runtime acceptance status

| Gate | Result | Evidence |
|---|---|---|
| Vendor Environment | **PASS** | `artifacts/logs/phase12_5_2_environment.log` |
| Native Model Registration | **PASS** | Architecture resolved as `NeurXDeepSeekV2ForCausalLM` in `phase12_5_2_registration_hardware.log` |
| Model Instantiation | **PASS** | Native model constructed and weights loaded in Vendor worker runtime |
| 27-layer Cache Discovery | **PASS** | Static context reached 27 layers (`static_context_layer_count`) |
| Worker Cache Allocation | **PASS** | Worker allocated all native cache groups; per-layer allocation IDs logged |
| Reshape Hook | **PASS** | Reshape hook invoked for all layers (`reshape_hook_invoked=true`) |
| K/V Zero-copy | **PASS** | Zero-copy storage checks logged (`zero_copy_verified=true`) |
| Scheduler Metadata | **PASS** | Real scheduler schedule events and native metadata consumption logged |
| Native Forward | **PASS** | Real prefill + decode requests executed; native request outputs logged |
| Hardware Acceptance | **BLOCKED** | Runtime integration passed, but fixture-based regression controls from prior phases were not rerun in this pass due unavailable mounted fixture set |

## Notes

- No package/environment upgrades were performed.
- Vendor baseline (`DeepseekV2ForCausalLM`) remained runnable and unchanged during this phase.
- Final numerical follow-up is tracked in `docs/PHASE12_5_3_FINAL_ACCEPTANCE.md`.
- Root-cause closure and final numerical pass are recorded in `docs/PHASE12_5_4_ACCEPTANCE.md`.
- Detailed runtime evidence is captured in:
  - `artifacts/logs/phase12_5_2_registration_hardware.log`
  - `artifacts/logs/phase12_5_2_worker_cache_hardware.log`
  - `artifacts/logs/phase12_5_2_scheduler_metadata.log`

