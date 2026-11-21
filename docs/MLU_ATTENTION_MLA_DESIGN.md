# DeepSeek-V2-Lite MLA execution design and hardware checks

## Eager checkpoint contract

The contiguous Q projection `[B,S,16,192]` has strides `[S*3072,3072,192,1]`. The head-first view `[B,16,S,192]` has strides `[S*3072,192,3072,1]` until materialization. Q splits into 128 nonrotary and 64 rotary values. KV-A yields one normalized latent of width 512 and one shared 64-wide rotary key per token. KV-B yields a head-first view of `[B,16,S,256]` with strides `[S*4096,256,4096,1]` before splitting. The reference assembles K with nonrotary keys and the broadcast rotary key. It caches expanded K `[B,16,L,192]` and V `[B,16,L,128]` through the configured cache object. A contiguous implementation may allocate row-major versioned tensors, while the incoming V split has the parent KV-B stride. Cache layout must be explicit at the adapter boundary.

QK scores are scaled by `192^(-1/2) * yarn_get_mscale(40,0.707)^2`. Eager attention accepts additive mask `[B,1,S,L]`, uses FP32 softmax then casts probabilities to activation dtype, and multiplies by V. Padding and causal offset must agree with the reference, especially for unequal query/cache lengths. Attention dropout is zero at inference. Reference RoPE is applied to Q and the shared K rotary component before cache update.

## Compressed alternative

A development cache can keep normalized latents `[B,L,512]` and rotated shared keys `[B,L,64]`, each contiguous in the last dim. With per-head KV-B matrices `Wk,Wv[128,512]`, absorb the key projection into Q as `(Q_nope·Wk)·latent^T` plus the rotary dot. Apply the same scale and mask. Mix latents with attention probabilities, then apply `Wv^T`. This is an FP32 algebraic identity. Low-precision intermediate rounding differs and must be evaluated under the matched semantic contract before replacing eager/cache behavior.

## Standalone hardware test

On MLU 5 (`0000:b3:00.0`), the fixture used real layer-zero KV-B weights, deterministic normalized-latent and rotated Q/K inputs, all 16 heads, and query/cache lengths 7/7, 1/65 and 3/65. Torch MLU operations exercised causal prefill and offset cache read shapes. It did not allocate or append a serving cache or run a custom fused MLA kernel. The FP32 materialized and absorbed formulations passed all 12 cases at the fixed `1e-5 + 1e-5*abs(ref)` gate. The largest FP32 difference was 5.96046448e-8.

**Low-precision model-order parity is not accepted.** Five of six cases passed. One FP16 7/7 prefill output differed by 1.52587890625e-5, beyond the same threshold. The tolerance was preserved, and the harness exited 1. Stage diagnostics found the first differences after KV-B rounding (30 values, max 0.0001220703125). Rounded probabilities agreed. The FP32 AV step differed by up to 7.7560544e-6 before final FP16 rounding. These observations identify a boundary for debugging. They do not establish a hardware defect or justify loosening the tolerance.

[MLA accuracy CSV](../artifacts/logs/phase5_mla_accuracy.csv) · [Stage diagnostics and exit log](../artifacts/logs/phase5_mla_run.log)
