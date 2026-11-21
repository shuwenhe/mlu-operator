# Technical acceptance

| Evidence | Result |
| --- | --- |
| Implemented | Standalone CNNL GEMM and RMSNorm wrappers, CPU reference, native harnesses, CNRT probe and build scripts. GEMM and its reference were adapted from the earlier local prototype into self-contained sources. |
| Compiled | C translation units for GEMM, RMSNorm, their test harnesses and the CNRT probe compiled against the deployed CNRT 7.7.3 / CNNL 2.2.18 headers. |
| CPU-tested | Twenty GEMM reference cases and the FP16/BF16 boundary checks passed. |
| Linked native executable | Blocked by deployed SDK glibc dependencies on Ubuntu 20.04. |
| Hardware execution and numerical acceptance | Blocked by occupancy, missing sandbox device nodes, and lack of an authorized reservation. No MLU operator was submitted. |
| Performance | No MLU timings or throughput. No valid production baseline was measured. |

The GEMM harness covers FP16/BF16, 128–1024 squares, an odd rectangle and two projection shapes. It rejects nonfinite outputs and checks absolute plus relative tolerance at 1e-5, with a 1e-6 relative-error denominator floor. It records CPU reference time, transfer times, one warm-up, and 20 measured launches bracketed by device notifiers. Kernel TFLOPS is 2MNK divided by the mean kernel seconds and 10^12. Session end-to-end wall time includes setup, all iterations, copies and cleanup. It should be compared only with a baseline under the same scope. Hardware utilization and p95 are pending an authorized performance run.

RMSNorm checks widths 1, 65, 512, 2048 and 4096, both input types, zero rows, signed inputs, nonconstant FP32 weights and two repeated calls. Its scalar reference uses FP32. The observed CNNL header documents supported input, scale, output and saved-RMS types, and the vendor internals' accumulation precision remains unverified. Before production integration, validate reduction precision with dynamic-range fixtures and confirm the selected output dtype contract with the target model.

Shell builds and script syntax checks were verified. CMake configuration was not exercised because CMake is absent from this host. Local diagnostic artifacts are in `artifacts/logs/cpu-test.log`, `artifacts/logs/native-link.log` and `artifacts/accuracy/gemm_cpu_reference.csv`. The existing README was backed up before replacement in `artifacts/logs/README.original.md`.
