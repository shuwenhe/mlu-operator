#include "gemm_common.h"
#include "mlu_gemm_reference.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Launch = decltype(&mlu_bang_gemm_fp16);
static constexpr int warmups = 2, repeats = 20;

static void check(cnrtRet_t status, const char *operation) {
    if (status != cnrtSuccess) {
        std::fprintf(stderr, "%s: %u (%s)\n", operation, unsigned(status), cnrtGetErrorStr(status));
        std::exit(1);
    }
}
#define CHECK(call) check((call), #call)

static int run_case(bool bf16, int m, int n, int k, cnrtQueue_t queue) {
    std::vector<uint16_t> a(size_t(m)*k), b(size_t(k)*n);
    std::vector<float> af(a.size()), bf(b.size()), c(size_t(m)*n), ref(c.size());
    const uint16_t one = bf16 ? 0x3f80 : 0x3c00;
    const uint16_t half_value = bf16 ? 0x3f00 : 0x3800;
    auto fill = [=](std::vector<uint16_t> &bits, std::vector<float> &values, unsigned stride) {
        for (size_t i = 0; i < bits.size(); ++i) {
            int x = int((i % 31) * stride % 31) - 15;
            uint16_t magnitude = x == 0 ? 0 : (x % 2 ? half_value : one);
            bits[i] = magnitude | (x < 0 ? 0x8000 : 0);
            values[i] = x == 0 ? 0.0f : (x < 0 ? -1.0f : 1.0f) * (x % 2 ? 0.5f : 1.0f);
        }
    };
    fill(a, af, 7);
    fill(b, bf, 11);
    mlu_gemm_reference(af.data(), bf.data(), ref.data(), m, n, k, k, n, n);

    void *da = nullptr, *db = nullptr, *dc = nullptr;
    cnrtNotifier_t start = nullptr, end = nullptr;
    CHECK(cnrtMalloc(&da, a.size()*sizeof(uint16_t)));
    CHECK(cnrtMalloc(&db, b.size()*sizeof(uint16_t)));
    CHECK(cnrtMalloc(&dc, c.size()*sizeof(float)));
    CHECK(cnrtNotifierCreate(&start));
    CHECK(cnrtNotifierCreate(&end));
    const Launch launch = bf16 ? mlu_bang_gemm_bf16 : mlu_bang_gemm_fp16;
    const auto enqueue = [&] {
        CHECK(launch(queue, static_cast<const uint16_t *>(da), static_cast<const uint16_t *>(db),
                     static_cast<float *>(dc), m, n, k));
    };
    CHECK(cnrtMemcpy(da, a.data(), a.size()*sizeof(uint16_t), cnrtMemcpyHostToDev));
    CHECK(cnrtMemcpy(db, b.data(), b.size()*sizeof(uint16_t), cnrtMemcpyHostToDev));
    for (int i = 0; i < warmups; ++i) enqueue();
    CHECK(cnrtPlaceNotifier(start, queue));
    for (int i = 0; i < repeats; ++i) enqueue();
    CHECK(cnrtPlaceNotifier(end, queue));
    CHECK(cnrtQueueSync(queue));
    float total_us = 0;
    CHECK(cnrtNotifierDuration(start, end, &total_us));

    const auto wall_start = Clock::now();
    CHECK(cnrtMemcpy(da, a.data(), a.size()*sizeof(uint16_t), cnrtMemcpyHostToDev));
    CHECK(cnrtMemcpy(db, b.data(), b.size()*sizeof(uint16_t), cnrtMemcpyHostToDev));
    enqueue();
    CHECK(cnrtQueueSync(queue));
    CHECK(cnrtMemcpy(c.data(), dc, c.size()*sizeof(float), cnrtMemcpyDevToHost));
    const double wall_us = std::chrono::duration<double, std::micro>(Clock::now() - wall_start).count();
    CHECK(cnrtNotifierDestroy(end));
    CHECK(cnrtNotifierDestroy(start));
    CHECK(cnrtFree(dc));
    CHECK(cnrtFree(db));
    CHECK(cnrtFree(da));

    double max_abs = 0, max_rel = 0, sum_abs = 0;
    size_t nan = 0, inf = 0, bad = 0;
    for (size_t j = 0; j < c.size(); ++j) {
        if (!std::isfinite(c[j])) {
            nan += std::isnan(c[j]);
            inf += std::isinf(c[j]);
            continue;
        }
        const double error = std::abs(double(c[j]) - ref[j]);
        const double scale = std::abs(double(ref[j]));
        max_abs = std::fmax(max_abs, error);
        max_rel = std::fmax(max_rel, error / std::fmax(1e-6, scale));
        sum_abs += error;
        bad += error > 1e-5 + 1e-5 * scale;
    }
    const double kernel_us = double(total_us) / repeats;
    std::printf("%s,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%zu,%zu,%zu,%.6g,%.6g,%.6g\n",
        bf16 ? "bf16" : "fp16", m,n,k,warmups,repeats,max_abs,max_rel,sum_abs/c.size(),
        nan,inf,bad,kernel_us,wall_us,kernel_us > 0 ? 2.0*m*n*k/(kernel_us*1e6) : NAN);
    return bad || nan || inf;
}

int main(int argc, char **argv) {
    if (argc != 3 || std::string(argv[1]) != "--authorized" || std::string(argv[2]) != "0") {
        std::fprintf(stderr, "usage: %s --authorized 0 (only the reserved MLU may be visible)\n", argv[0]);
        return 2;
    }
    unsigned int count = 0;
    CHECK(cnrtGetDeviceCount(&count));
    if (count != 1) {
        std::fprintf(stderr, "expected exactly one visible device, got %u\n", count);
        return 2;
    }
    CHECK(cnrtSetDevice(0));
    cnrtQueue_t queue = nullptr;
    CHECK(cnrtQueueCreate(&queue));
    const int shapes[][3] = {{16,16,16},{32,64,128},{128,256,512},{256,128,512},{127,193,65},{1,129,129}};
    std::puts("dtype,m,n,k,warmups,repeats,max_abs,max_rel,mean_abs,nan,inf,mismatches,kernel_us,e2e_us,tflops");
    int failed = 0;
    for (int dtype = 0; dtype < 2; ++dtype)
        for (const auto &shape : shapes)
            failed |= run_case(dtype != 0, shape[0], shape[1], shape[2], queue);
    CHECK(cnrtQueueDestroy(queue));
    return failed;
}
