# Device-resident packed GEMM weights

The new [packed-weight interface](../operators/gemm/native/prepacked/packed_weight.h) owns immutable device RHS storage. It reuses the existing matrix-instruction GEMM launchers without altering them.

| API | Contract |
| --- | --- |
| `pack_rhs_host` | Takes packed logical row-major host `B[K,N]`, FP16 or BF16 words. Allocates on the current device, packs on the CPU and completes one synchronous H2D copy before returning. |
| `pack_rhs_device` | Takes device row-major `B[K,N]` and a queue for the current device. Allocates packed storage and enqueues device packing. It optionally brackets packing with caller-provided notifiers. B must stay live until the queue completes. |
| `gemm_prepacked` | Enqueues the existing matrix GEMM with packed RHS and known K/N/dtype. A is device row-major `[M,K]`; output is device FP32 `[M,N]`. |
| `release_packed_weight` | Frees the packed storage on its owning current device and clears the caller's handle on success. The caller must complete all referencing queues first. |

The handle records device ordinal, dimensions, byte count and input type. There is no global cache, automatic device switch or implicit queue synchronization in the asynchronous path. The ABI checks input pointer alignment to scalar type alignment, positive shapes, valid dtype, queue presence and current-device identity. Callers provide valid host/device ranges, nonoverlapping buffers and queues belonging to that device. The runtime handles launch/queue errors. Device pointer provenance and queue ownership are not inferred from raw addresses.

Tiles are 128×128, ordered by N tile then K tile. The device packer bulk-loads valid rows, zero-fills K/N tails, transposes in NRAM and copies each column into the hardware-verified WRAM slot. It uses two 32 KiB NRAM tiles per task. Its logical mapping agrees with the prior one-hot hardware map:

```text
offset(k,n) = (8*floor((n mod 64)/4) + 4*floor(n/64) + n mod 4)*128 + k
```

Persistent storage is `32768 * ceil(K/128) * ceil(N/128)` bytes. The host route uses one temporary host buffer of that size. The device route needs the original `2*K*N` device bytes until packing completes, in addition to the persistent allocation. Existing model-resident row-major weights can be passed directly. The benchmark uploads them once to separate preparation and invocation costs.

Build from the existing authorized Ubuntu 24.04 SDK context with [build.sh](../operators/gemm/native/prepacked/build.sh). It records matrix and packer compilation at `--bang-mlu-arch=mtp_592 -O2 -std=c++11` and the host/link flags. No SDK installation, NeurX change, or production integration is required.
