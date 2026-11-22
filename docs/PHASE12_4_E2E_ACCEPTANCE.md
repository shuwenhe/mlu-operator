# Phase 12.4 end-to-end acceptance

| Gate | Status | Evidence or dependency |
|---|---|---|
| A. DeepSeek architecture | **PASS** | All 5,291 BF16 tensor headers checked. |
| B. Vendor MLA layout | **PASS** | Actual prefill and decode cache traces for all 27 vendor layers. |
| C. Native model registration | **BLOCKED** | Registry entry and native decoder module are pending. |
| D. Per-layer cache specification binding | **BLOCKED** | All 27 specs and zero-copy device views passed the component test. Actual worker binding remains pending. |
| E. Native projection integration | **BLOCKED** | Actual vendor requests invoked the native Layer 0 O projection. All-layer native projection binding is pending. |
| F. Native attention integration | **BLOCKED** | The native cache and attention chain are validated with fixtures in Phase 12.3. Actual worker invocation remains pending. |
| G. Actual metadata dispatch to native backend | **BLOCKED** | Requires registered native model and materialized cache ownership. |
| H. Full 27-layer native execution | **NOT RUN** | Vendor control used the vendor model. |
| I. Full-model numerical parity | **NOT RUN** | Scheduled native model output is unavailable. |
| J. Incremental native decode | **NOT RUN** | Vendor control generated four tokens, including three incremental decode steps. |
| K. Scheduler-managed native reclamation | **NOT RUN** | Requires native cache allocation from the scheduler. |

[Model binding](PHASE12_4_NATIVE_MODEL_BINDING.md) · [Source verification](../artifacts/logs/phase12_4_source_hashes.txt) · [Phase 12.5.1 acceptance](PHASE12_5_1_ACCEPTANCE.md)
