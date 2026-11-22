# vLLM MLU compatibility audit

The pinned Cambricon SDK image `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756` exposes PCI `0000:4f:00.0` as MLU device 0. The host Python lacks the required packages. The image contains Python 3.12.3, PyTorch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, and Transformers 5.11.0. The vLLM module reports 0.25.1, the distribution is 0.25.1+empty, and the vendor distribution is vllm_mlu 0.25.1+mlu0.16.0.pt212. Runtime versions on card 0 were driver 6.5.52, CNRT 7.7.3 and CNNL 2.2.18. CNCC 5.7.6 compiled the shared bridge for `mtp_592`. The vendor source snapshot under `/workspace/vllm-v0.25.1+mlu0.16.0.pt212` has no Git metadata. The image digest and interface file hashes pin the inspected implementation.

| Surface | Evidence and adaptation |
|---|---|
| Platform and PyTorch device | The vendor `vllm.platform_plugins` entry point resolves to `vllm_mlu.platforms.mlu.MLUPlatform`. It enumerated one MLU590-M9 and allocated an MLU tensor. Import `torch_mlu` before `vllm` in standalone probes to register the dispatch key. |
| Worker and runner | The vendor platform sets `vllm_mlu.v1.worker.gpu_worker.MLUWorker`. Retain its device initialization, allocation accounting, queue handling and engine process lifecycle. Select a project model explicitly through the model registry in each process. |
| Operator hooks | Installed `CustomOp` supports registration and OOT dispatch under an active `VllmConfig`. The vendor also applies its own RMSNorm hijack with a fused-residual signature. The project bridge registers a distinct op so selection is explicit. |
| Attention and KV | `MLUPlatform.get_attn_backend_cls` selects the vendor backend. The MLA metadata builder provides `block_table_tensor` and `slot_mapping`. It splits prefills and decodes. A project cache adapter must consume this metadata and use a matching cache specification. |
| MoE | The vendor DeepSeek model defines its own MoE path. A project model adapter must call the existing native routing and expert kernels explicitly. |
| CNRT queue | torch_mlu exposes the current stream's `mlu_stream` handle. A real nondefault handle was passed to the native launcher. The default stream reported sentinel value 2 and was outside the probe. |
| Standalone runner | Builds with CNCC, g++, CNRT and accepted packed-weight objects. It owns a worker, request buffers, weight registry and queue. It has no vLLM worker lifecycle or paged allocator binding. |

An offline vLLM `LLM.generate` call loaded the four original checkpoint shards and generated four tokens per request for two prompts on the vendor MLU backend. The read-only sandbox requires writable Triton, HF, inductor and vLLM caches. The initial import order caused an unknown MLU dispatch key and the first retry tried to write `/.triton`. Setting the cache paths fixed initialization. The successful vendor run logged a nonfatal usage-statistics write failure under `/.config`.

[Installed-source and build evidence](../artifacts/logs/vllm_mlu_build.log) · [Card and model execution](../artifacts/logs/vllm_mlu_hardware.log) · [Upstream plugin contracts](https://docs.vllm.ai/en/stable/design/plugin_system/) · [Upstream CustomOp contracts](https://docs.vllm.ai/en/stable/design/custom_op/)

## Reproduction in the pinned image

Mount the repository read-only at `/project`, the original model read-only at `/checkpoint`, the Phase 11 build at `/phase11`, the router captures at `/p115`, the Phase 9.1.2 fixtures at `/out`, the Phase 9.3 fixtures at `/av-out`, and an empty writable output directory at `/vout`. Expose the card 0 CNRT device nodes only. From `/project`:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCHINDUCTOR_CACHE_DIR=/tmp/phase-inductor TRITON_CACHE_DIR=/tmp/triton
export HF_HOME=/tmp/hf VLLM_CACHE_ROOT=/tmp/vllm LOGNAME=phase_audit
py=/torch/venv3/pytorch_infer/bin/python3
bash integration/vllm_native_mlu/build.sh /vout/native
"$py" -m integration.vllm_native_mlu.probe /vout/native/libnative_mlu.so \
    /vout/accuracy.csv /checkpoint /phase11/boundaries /p115
"$py" -m integration.vllm_native_mlu.attention_probe /vout/native/libnative_mlu.so \
    /vout/attention.csv /out /av-out
"$py" -m integration.vllm_native_mlu.vendor_baseline
```
