# Phase 12.4 numerical acceptance

**Full-model three-way comparison NOT RUN.** The independent CPU PyTorch and standalone Phase 11.5 traces are preserved. The scheduled native model needed for the third path is pending. Vendor control tokens establish runtime execution of the vendor model.

The accuracy CSV contains component comparisons. Cache tests require bitwise equality for the preserved BF16 words. Projection tests compare FP32 output from accepted native code with independent CPU PyTorch on the same BF16 inputs and checkpoint weights. They use `1e-5 + 1e-5 * abs(reference)`. First recorded indices are flattened and refer to the first bit difference.

[Component accuracy](../artifacts/logs/phase12_4_accuracy.csv) · [Frozen standalone comparison](PHASE11_5_ROUTER_NUMERICAL_ACCEPTANCE.md)

The cache component matched 2,350,080 BF16 words bit-for-bit across 54 K/V comparisons. Four one-row Q and KV-A projection cases at layers 0 and 26 passed across 7,296 FP32 values. The maximum absolute error was 2.8611e-6, the maximum relative error was 1.6542e-4, and the maximum ULP distance was 2,048 near zero. Every projection value met the existing combined tolerance.
