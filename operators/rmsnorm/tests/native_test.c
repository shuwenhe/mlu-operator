#include "mlu_rmsnorm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int run_case(int device, mlu_dtype dtype, int width) {
    const int rows = 3;
    const size_t count = (size_t)rows * width;
    const float epsilon = 1e-6f;
    uint16_t *x = calloc(count, sizeof(*x));
    float *weights = malloc((size_t)width * sizeof(*weights));
    float *out = malloc(count * sizeof(*out));
    int failed = 1;
    if (!x || !weights || !out)
        goto cleanup;

    const uint16_t one = dtype == MLU_OPERATOR_BF16 ? 0x3f80 : 0x3c00;
    for (int j = 0; j < width; ++j)
        weights[j] = 0.5f + (float)(j % 7) / 8.0f;
    for (size_t i = 0; i < count; ++i) {
        if (i >= (size_t)width && i % 3 != 0)
            x[i] = (uint16_t)(one | (i % 2 ? 0x8000 : 0));
        out[i] = NAN;
    }

    for (int repeat = 0; repeat < 2; ++repeat) {
        mlu_status status = mlu_rmsnorm(device, dtype, rows, width, epsilon, x, weights, out);
        if (status.domain != MLU_OPERATOR_OK) {
            fprintf(stderr, "%s domain=%d code=%d\n", status.operation, status.domain, status.code);
            goto cleanup;
        }
        float max_abs = 0, max_rel = 0, sum_abs = 0;
        size_t mismatches = 0;
        for (int row = 0; row < rows; ++row) {
            const size_t start = (size_t)row * width;
            float sum = 0;
            for (int j = 0; j < width; ++j)
                sum += x[start + j] != 0 ? 1.0f : 0.0f;
            const float inv = 1.0f / sqrtf(sum / (float)width + epsilon);
            for (int j = 0; j < width; ++j) {
                const uint16_t input = x[start + j];
                const float value = input == 0 ? 0 : (input & 0x8000 ? -1.0f : 1.0f);
                const float expected = value * inv * weights[j];
                const float error = fabsf(out[start + j] - expected);
                if (!isfinite(error)) {
                    ++mismatches;
                    continue;
                }
                max_abs = fmaxf(max_abs, error);
                max_rel = fmaxf(max_rel, error / fmaxf(fabsf(expected), 1e-6f));
                sum_abs += error;
                mismatches += error > 1e-5f + 1e-5f * fabsf(expected);
            }
        }
        printf("%s,%d,%d,%.8g,%.8g,%.8g,%zu\n",
               dtype == MLU_OPERATOR_BF16 ? "bf16" : "fp16", width, repeat,
               max_abs, max_rel, sum_abs / (float)count, mismatches);
        if (mismatches)
            goto cleanup;
    }
    failed = 0;

cleanup:
    free(out);
    free(weights);
    free(x);
    return failed;
}

int main(int argc, char **argv) {
    if (argc != 2 || argv[1][0] < '0' || argv[1][0] > '7' || argv[1][1]) {
        fprintf(stderr, "usage: %s <allocated CNRT ordinal 0-7>\n", argv[0]);
        return 2;
    }
    const int widths[] = {1, 65, 512, 2048, 4096};
    puts("dtype,width,repeat,max_abs,max_rel,mean_abs,mismatches");
    for (int dtype = MLU_OPERATOR_FP16; dtype <= MLU_OPERATOR_BF16; ++dtype)
        for (size_t i = 0; i < sizeof(widths) / sizeof(*widths); ++i)
            if (run_case(argv[1][0] - '0', (mlu_dtype)dtype, widths[i]))
                return 1;
    return 0;
}
