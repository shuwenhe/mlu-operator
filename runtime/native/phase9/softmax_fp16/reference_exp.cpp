// CPU-only diagnostic shim for the pinned PyTorch wheel's exact SIMD exponential.
#include <immintrin.h>
#include <cstddef>

extern "C" __m256 Sleef_expf8_u10avx2(__m256);

extern "C" void phase94_pinned_exp(const float *input, float *output, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8)
        _mm256_storeu_ps(output + i, Sleef_expf8_u10avx2(_mm256_loadu_ps(input + i)));
    if (i < size) {
        float tail[8]{};
        for (size_t j = 0; i + j < size; ++j) tail[j] = input[i + j];
        _mm256_storeu_ps(tail, Sleef_expf8_u10avx2(_mm256_loadu_ps(tail)));
        for (size_t j = 0; i + j < size; ++j) output[i + j] = tail[j];
    }
}
