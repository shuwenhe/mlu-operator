# Phase 9.3 AV execution plan

- Record and check the accepted QK, Softmax and Phase 8.2 source hashes. Preserve historical fixtures and logs.
- Capture the eager checkpoint's V, low-precision probabilities and AV outputs in both dtypes for the same two requests and four lengths. Verify the captured QK scores against the Phase 9.1.2 fixture bits.
- Implement the device probability × V product with FP32 accumulation and one final dtype conversion. Use flattened head-major buffers with explicit physical cache capacity. Replace the existing AV dispatch stub and retain owned buffers until the request queue completes.
- Run the eight documented two-head synthetic cases and the 32 checkpoint cases on card 0, with prefill causal masks and nonzero decode offsets. Replay AV with both exact checkpoint probabilities and actual native Softmax probabilities. Compare FP32 with widened PyTorch and the converted result with original eager PyTorch, keeping matched-input and chained metrics explicit.
- Re-run the accepted QK and Softmax checks. Record build, hardware, numerical and hash results without changing their acceptance artifacts. Keep Phase 9.4 outside this change.
