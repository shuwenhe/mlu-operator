# MLU590-M9 matrix instruction analysis

CNCC 5.7.6 on `mtp_592` declares FP16 and BF16 overloads of `__bang_matmul(float *dst, const Low *src0, const Low *src1, uint32_t M, uint32_t K, uint32_t N)` in `/usr/local/neuware/lib/clang/11.1.0/include/bang_device_functions_decls.h`. The device assembly uses `conv.nram.f32.f16.f16` and `conv.nram.f32.bf16.bf16` for the verified 16×128×128 tile. A and C occupy NRAM, B occupies WRAM. The hardware probe confirms the 16-row A and output order. The overload has no transpose or stride arguments.

The installed SDK convolution example establishes the NRAM/WRAM staging model. The vendor GEMM lab describes rearranging weights for WRAM. Its data layout is not used as evidence for the exact modern overload. [Cambricon GEMM lab](https://developer.cambricon.com/index/curriculum/expdetails/id/8/classid/)

## Verified RHS layout

On host MLU 0 (`0000:4f:00.0`), [layout_probe.mlu](../operators/gemm/native/matrix/layout_probe.mlu) swept all 128 K positions in groups of 16 one-hot A rows. Two passes assigned exact base-128 digits to each physical WRAM word. Decoding the output recovered all 16,384 physical offsets for both dtypes. The maps were identical and every entry matched this word-offset rule:

```text
offset(k,n) = (8*floor((n mod 64)/4) + 4*floor(n/64) + (n mod 4))*128 + k
```

The output is FP32 and the later full kernel accumulates across K slabs in FP32. One-hot probes establish addressing. Cancellation and magnitude tests establish the observed numerical behavior for those fixtures. Neither establishes undocumented inner accumulator behavior for arbitrary exponent patterns. Compiler-emitted local arrays use 64-byte alignment. The verified tile uses 32 KiB WRAM, and exact supported shape/alignment limits beyond this tile were not determined.

The first probe build rejected host `_Float16` under CNCC. Integer bit encoding and explicit `-lstdc++` fixed the compilation and linking issues. Compile and execution commands are in the [layout log](../artifacts/logs/gemm_matrix_layout.log). [FP16 map](../artifacts/logs/matrix-layout-fp16.csv) · [BF16 map](../artifacts/logs/matrix-layout-bf16.csv)
