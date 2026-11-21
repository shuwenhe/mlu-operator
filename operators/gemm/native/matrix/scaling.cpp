#include "matrix_gemm.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr int rows = 128, width = 256, inner = 512, warmups = 5, repeats = 100;
constexpr size_t elements = size_t(rows)*width;

struct Result {
    Clock::time_point finish;
    bool ok = true;
};

static void worker(int ordinal, std::atomic<int> &ready, std::atomic<bool> &start,
                   Result &result) {
    cnrtQueue_t queue = nullptr;
    void *x = nullptr, *gamma = nullptr, *out = nullptr;
    const uint16_t weight = uint16_t(0x3f80 + ordinal*16);
    // All entries of the padded RHS are equal for this aligned shape.
    std::vector<uint16_t> host_x(size_t(rows)*inner, 0x3f80), host_gamma(size_t(inner)*width, weight);
    std::vector<float> host_out(elements);
    const float expected = inner * (1.0f + ordinal/8.0f);
    const auto check = [&](cnrtRet_t status, const char *op) {
        if (status != cnrtSuccess) {
            std::fprintf(stderr, "ordinal %d %s: %u %s\n", ordinal, op, unsigned(status),
                         cnrtGetErrorStr(status));
            result.ok = false;
            return false;
        }
        return true;
    };
    const auto launch = [&] {
        return check(matrix_gemm_bf16(queue, static_cast<uint16_t *>(x),
                     static_cast<uint16_t *>(gamma), static_cast<float *>(out),
                     rows, width, inner), "launch");
    };
    if (check(cnrtSetDevice(ordinal), "select") &&
        check(cnrtQueueCreate(&queue), "create queue") &&
        check(cnrtMalloc(&x, host_x.size()*2), "allocate X") &&
        check(cnrtMalloc(&gamma, host_gamma.size()*2), "allocate gamma") &&
        check(cnrtMalloc(&out, elements*4), "allocate output") &&
        check(cnrtMemcpy(x, host_x.data(), host_x.size()*2, cnrtMemcpyHostToDev), "upload X") &&
        check(cnrtMemcpy(gamma, host_gamma.data(), host_gamma.size()*2, cnrtMemcpyHostToDev), "upload gamma")) {
        for (int i = 0; i < warmups && launch(); ++i) {}
        check(cnrtQueueSync(queue), "warmup sync");
    }
    ++ready;
    while (!start.load()) std::this_thread::yield();
    if (result.ok) {
        for (int i = 0; i < repeats && launch(); ++i) {}
        check(cnrtQueueSync(queue), "timed sync");
    }
    result.finish = Clock::now();
    if (result.ok && check(cnrtMemcpy(host_out.data(), out, elements*4, cnrtMemcpyDevToHost), "readback") &&
        !std::all_of(host_out.begin(), host_out.end(), [=](float v) { return v == expected; })) {
        std::fprintf(stderr, "ordinal %d output mismatch\n", ordinal);
        result.ok = false;
    }
    if (out) check(cnrtFree(out), "free output");
    if (gamma) check(cnrtFree(gamma), "free gamma");
    if (x) check(cnrtFree(x), "free X");
    if (queue) check(cnrtQueueDestroy(queue), "destroy queue");
}

int main(int argc, char **argv) {
    if (argc != 2 || std::string(argv[1]) != "--authorized") return 2;
    unsigned count = 0;
    if (cnrtGetDeviceCount(&count) != cnrtSuccess || count != 8) return 1;
    std::puts("cards,dtype,m,n,k,warmups,repeats,elapsed_ms,aggregate_tflops,mean_per_card_tflops,speedup,efficiency");
    double baseline = 0;
    for (int cards : {1, 2, 4, 8}) {
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        std::array<Result, 8> results;
        std::vector<std::thread> threads;
        for (int i = 0; i < cards; ++i)
            threads.emplace_back(worker, i, std::ref(ready), std::ref(start), std::ref(results[i]));
        while (ready.load() != cards) std::this_thread::yield();
        const auto begin = Clock::now();
        start.store(true);
        for (auto &thread : threads) thread.join();
        auto finish = begin;
        for (int i = 0; i < cards; ++i) {
            if (!results[i].ok) return 1;
            finish = std::max(finish, results[i].finish);
        }
        const double seconds = std::chrono::duration<double>(finish - begin).count();
        const double throughput = 2.*cards*rows*width*inner*repeats/(seconds*1e12);
        if (cards == 1) baseline = throughput;
        std::printf("%d,bf16,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", cards, rows, width, inner,
                    warmups, repeats, seconds*1000, throughput, throughput/cards, throughput/baseline, throughput/(baseline*cards));
    }
}
