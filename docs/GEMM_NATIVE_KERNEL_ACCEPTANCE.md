# Native GEMM hardware acceptance

**PASS for the 14 native FP16/BF16 correctness cases on MLU 0.** CNNL also passed the same 14 cases using the same inputs. Both paths repeated the five required shapes per dtype in the timed run on MLU 2 (`0000:6f:00.0`). The device builds emit the expected typed matrix instructions and fatbins. This is actual execution evidence.

| Check | Native | CNNL |
| --- | ---: | ---: |
| Correctness cases on MLU 0 | 14 | 14 |
| Maximum absolute, relative and mean absolute error | 0 | 0 |
| Mismatches at existing `1e-5 + 1e-5*abs(reference)` gate | 0 | 0 |
| NaN/Inf count (combined) | 0 | 0 |
| Adjacent output canary errors | 0 | 0 |
| Timed cases on MLU 2 | 10 | 10 |

The benchmark uses deterministic LCG seeds 7 and 43 and dyadic values exact in both input formats. Correctness includes five requested shapes, a 1×1×1 small-magnitude case and a 17×129×257 paired-sign cancellation case using larger magnitude A and small B. Reference accumulation is independent serial FP32. The result applies to those fixtures. Broader dynamic ranges, memory-checker coverage and arbitrary alignment beyond tested packed matrices remain open. No correctness tolerance was loosened.

The one-card test container maps the selected compute/IPCM node and shared control using the already authorized disposable SDK context. The existing eight-card CNRT smoke gate was reused. Neither Team container, GLM53, NeurX, nor the earlier operator implementations were modified. Post-run CNMON reported 0 MiB used on every card.

[MLU 0 accuracy CSV](../artifacts/logs/gemm_native_matrix_accuracy.csv) · [Build and correctness transcript](../artifacts/logs/gemm_matrix_build_and_check.log) · [MLU 2 rerun](../artifacts/logs/gemm_matrix_performance_run.log)
