# Environment audit

Host Ubuntu 20.04, kernel 5.15.0-139-generic, glibc 2.31 and driver 6.5.52. Host Neuware prefix contains CNDev 3.12.4 and CNDev examples. The host toolkit 3.13.6 package stages Jammy archives. Cached CNRT 6.12.1 requires glibc 2.34. Host CNCC and CMake were absent from PATH. GCC is available.

The inspected production image is Ubuntu 24.04 with installed toolkit 4.7.6, CNRT 7.7.3, CNNL 2.2.18, CNCC/CNAS 5.7.6 and CNCL 1.30.10. CNRT, CNNL and CNDrv in this bundle need glibc 2.38 symbols. CNCC documents the MLU590 family target as `mtp_592`. This project uses CNNL, so the host C client requires matching headers and libraries but does not invoke CNCC. Installed BANG examples include device link, launch, addition and convolution. Their presence establishes source availability.

Earlier read-only production inspection found installed torch_mlu 1.34.1, torch_mlu_ops 1.13.0, vLLM 0.25.1 and vLLM-MLU 0.25.1+mlu0.16.0.pt212. The compiled TMO extension references CNNL MatMulEx v2, BatchMatMulEx and GroupGemm. The RMSNorm Python wrapper dispatches to the compiled TMO fused layernorm family. Its precise underlying kernel was not established. The deployed GLM workload establishes aggregate hardware use, with no operator-level timings or correctness trace available.

Compile-only validation used previously staged copies of the deployed headers via a caller-supplied SDK prefix. The project has no runtime dependency on the staging location. A dedicated compatible userspace must be approved independently of the production serving container.
