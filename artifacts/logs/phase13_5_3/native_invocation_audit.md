# Native Invocation Audit (Phase 13.5.3)

- Source trace: `artifacts/logs/phase13_5_2/tp1_trace`
- Requests: 5
- Total prefill/decode steps: 20
- Layer op evidence files: key=540, value=540, attention=540, residual=540

## Invocation Matrix

| Operator | Library | Symbol | Exported | Invocation Count | Input Shape | Output Shape | Dtype | MLU |
|---|---|---|---|---:|---|---|---|---|
| Q projection | `libprojection.so` | `phase124_project` | TRUE | 540 | [rows, 2048] | [rows, q_proj_out] fp32 (rows<=128) | bf16 input / fp32 out | YES |
| KV-A projection | `libprojection.so` | `phase124_project` | TRUE | 540 | [rows, 2048] | [rows, kv_a_out] fp32 | bf16 input / fp32 out | YES |
| KV-A RMSNorm | `libnative_mlu.so` | `phase82_native_bf16` | TRUE | 540 | [rows, 512] | [rows, 512] | bf16 | YES |
| KV-B projection | `libprojection.so` | `phase124_project` | TRUE | 540 | [rows, 512] | [rows, kv_b_out] fp32 | bf16 input / fp32 out | YES |
| RoPE | `libdecoder.so` | `worker_rope_bf16` | TRUE | 540 | rope=[17, rows, 64], pos=[rows] | rotated=[17, rows, 64] | bf16+int32 | YES |
| QK | `libnative_mlu.so` | `phase9_launch_qk_bf16` | TRUE | 540 | q=[16,rows,192], k=[16,len,192] | scores=[16,rows,len] | bf16/fp32 mixed | YES |
| Softmax | `libnative_mlu.so` | `phase9_softmax_bf16` | TRUE | 540 | scores=[16,rows,len] | probs=[16,rows,len] | bf16/fp32 mixed | YES |
| AV | `libnative_mlu.so` | `phase9_av_bf16` | TRUE | 540 | probs=[16,rows,len], v=[16,len,128] | av=[16,rows,128] | bf16/fp32 mixed | YES |
| Ordered O accumulation | `libdecoder.so` | `phase10_merge` | TRUE | 540 | av=[16,rows,128] | merged=[rows,2048] then o_proj->[rows,2048] | bf16/fp32 mixed | YES |
| Residual | `libdecoder.so` | `phase11_residual` | TRUE | 1080 | lhs=[rows,2048], rhs=[rows,2048] | [rows,2048] | bf16 | YES |

## Notes
- Residual invocation count is 1080 (= 2 residual adds × 27 layers × 20 steps), based on `_add` being called twice per layer in `phase12_5/model.py`.
- No silent fallback evidence found in TP1 trace/log path for the above operators.
- This audit reflects TP1 production path invocation evidence only; TP>1 remains separately gated.
