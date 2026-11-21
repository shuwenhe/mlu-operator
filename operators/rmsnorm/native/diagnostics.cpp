#include "rmsnorm_native.h"
extern "C" {
#include "mlu_rmsnorm.h"
}
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void read(std::istream &s, T *p, size_t n) {
    if (!s.read(reinterpret_cast<char *>(p), n * sizeof(T)))
        throw std::runtime_error("truncated fixture");
}

template<class T> void write(std::ostream &s, const std::vector<T> &v) {
    if (!s.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T)))
        throw std::runtime_error("write diagnostics failed");
}

static void check(cnrtRet_t status, const char *op) {
    if (status != cnrtSuccess)
        throw std::runtime_error(std::string(op) + ": " + std::to_string(status) +
                                 " (" + cnrtGetErrorStr(status) + ")");
}
#define CHECK(call) check((call), #call)

struct Queue {
    cnrtQueue_t value = nullptr;
    Queue() { CHECK(cnrtQueueCreate(&value)); }
    ~Queue() { if (value) cnrtQueueDestroy(value); }
    void close() { CHECK(cnrtQueueDestroy(value)); value = nullptr; }
};

struct Buffer {
    void *ptr = nullptr;
    static constexpr size_t guard = 128;
    size_t bytes;
    explicit Buffer(size_t size) : bytes(size) {
        CHECK(cnrtMalloc(&ptr, size + 2*guard));
        std::vector<unsigned char> initial(size + 2*guard, 0xa5);
        try {
            CHECK(cnrtMemcpy(ptr, initial.data(), initial.size(), cnrtMemcpyHostToDev));
        } catch (...) {
            cnrtFree(ptr);
            throw;
        }
    }
    ~Buffer() { if (ptr) cnrtFree(ptr); }
    void close() { CHECK(cnrtFree(ptr)); ptr = nullptr; }
    template<class T> T *as() const {
        return reinterpret_cast<T *>(static_cast<unsigned char *>(ptr) + guard);
    }
    void verify_guards() const {
        auto *base = static_cast<unsigned char *>(ptr);
        unsigned char borders[2*guard];
        CHECK(cnrtMemcpy(borders, base, guard, cnrtMemcpyDevToHost));
        CHECK(cnrtMemcpy(borders + guard, base + guard + bytes, guard, cnrtMemcpyDevToHost));
        for (const auto value : borders)
            if (value != 0xa5) throw std::runtime_error("device buffer guard was modified");
    }
    template<class T> void upload(std::vector<T> &v) {
        CHECK(cnrtMemcpy(as<T>(), v.data(), v.size()*sizeof(T), cnrtMemcpyHostToDev));
    }
    template<class T> void download(std::vector<T> &v) {
        CHECK(cnrtMemcpy(v.data(), as<T>(), v.size()*sizeof(T), cnrtMemcpyDevToHost));
    }
};

static uint16_t weight_bits(float value, bool bf16) {
    if (!std::isfinite(value))
        throw std::runtime_error("nonfinite gamma");
    if (!bf16) {
        const _Float16 low = static_cast<_Float16>(value);
        uint16_t result;
        std::memcpy(&result, &low, sizeof(result));
        return result;
    }
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static void run(const std::string &path, const std::string &out_dir, cnrtQueue_t queue, bool benchmark) {
    std::ifstream in(path, std::ios::binary);
    char magic[8];
    uint32_t fields[5];
    float eps;
    read(in, magic, 8);
    read(in, fields, 5);
    read(in, &eps, 1);
    if (std::memcmp(magic, "DSV2OP01", 8) || fields[0] != 1 || fields[1] > 1 ||
        !fields[2] || !fields[3] || fields[2] > 64 || fields[3] > 16384 ||
        fields[3] != fields[4] || !std::isfinite(eps) || eps <= 0)
        throw std::runtime_error("invalid RMSNorm fixture");
    const int rows = fields[2], width = fields[3];
    const bool bf16 = fields[1];
    const size_t count = size_t(rows)*width;
    std::vector<uint16_t> x(count), gamma_bits(width), rounded(count), output(count);
    std::vector<float> gamma(width), normalized(count), cnnl(count);
    read(in, x.data(), count);
    read(in, gamma.data(), gamma.size());
    if (in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("unexpected trailing fixture data");
    for (int i = 0; i < width; ++i)
        gamma_bits[i] = weight_bits(gamma[i], bf16);

    Buffer dx(2*count), dg(2*width), dy(2*count), dn(4*count), dr(2*count);
    dx.upload(x);
    dg.upload(gamma_bits);
    const auto launch = bf16 ? bang_rmsnorm_bf16 : bang_rmsnorm_fp16;
    const auto enqueue = [&](bool diagnostics) {
        CHECK(launch(queue, dx.as<uint16_t>(), dg.as<uint16_t>(), dy.as<uint16_t>(),
                     diagnostics ? dn.as<float>() : nullptr,
                     diagnostics ? dr.as<uint16_t>() : nullptr, rows, width, eps));
    };
    enqueue(true);
    CHECK(cnrtQueueSync(queue));
    dy.download(output);
    dn.download(normalized);
    dr.download(rounded);

    if (benchmark) {
        constexpr int warmups = 5, repeats = 30;
        for (int i = 0; i < warmups; ++i) enqueue(false);
        CHECK(cnrtQueueSync(queue));
        cnrtNotifier_t start = nullptr, end = nullptr;
        CHECK(cnrtNotifierCreate(&start));
        try {
            CHECK(cnrtNotifierCreate(&end));
            CHECK(cnrtPlaceNotifier(start, queue));
            for (int i = 0; i < repeats; ++i) enqueue(false);
            CHECK(cnrtPlaceNotifier(end, queue));
            CHECK(cnrtQueueSync(queue));
            float kernel_us = 0;
            CHECK(cnrtNotifierDuration(start, end, &kernel_us));
            std::vector<uint16_t> timed_output(count);
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) {
                dx.upload(x);
                dg.upload(gamma_bits);
                enqueue(false);
                CHECK(cnrtQueueSync(queue));
                dy.download(timed_output);
            }
            const auto us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - begin).count();
            if (timed_output != output)
                throw std::runtime_error("timed output differs from diagnostic output");
            std::cout << (bf16 ? "bf16" : "fp16") << ',' << rows << ',' << width
                      << ',' << warmups << ',' << repeats << ',' << kernel_us/repeats
                      << ',' << us/repeats << '\n';
        } catch (...) {
            cnrtQueueSync(queue);
            if (end) cnrtNotifierDestroy(end);
            cnrtNotifierDestroy(start);
            throw;
        }
        CHECK(cnrtNotifierDestroy(end));
        CHECK(cnrtNotifierDestroy(start));
    }
    dx.verify_guards(); dg.verify_guards(); dy.verify_guards();
    dn.verify_guards(); dr.verify_guards();
    dr.close(); dn.close(); dy.close(); dg.close(); dx.close();

    const mlu_status status = mlu_rmsnorm(0, bf16 ? MLU_OPERATOR_BF16 : MLU_OPERATOR_FP16,
                                         rows, width, eps, x.data(), gamma.data(), cnnl.data());
    if (status.domain != MLU_OPERATOR_OK)
        throw std::runtime_error(std::string(status.operation) + ": domain=" +
            std::to_string(status.domain) + ", code=" + std::to_string(status.code));

    const auto filename = path.substr(path.find_last_of('/') + 1);
    std::ofstream out(out_dir + "/" + filename + ".actual", std::ios::binary);
    write(out, normalized);
    write(out, rounded);
    write(out, output);
    write(out, cnnl);
    out.close();
    if (!out) throw std::runtime_error("close diagnostics failed");
}

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool benchmark = mode == "--authorized-benchmark";
    if (argc < 4 || (mode != "--authorized" && !benchmark)) {
        std::cerr << "usage: rmsnorm_diagnostics --authorized|--authorized-benchmark <output-dir> <norm-fixture>...\n";
        return 2;
    }
    try {
        unsigned int devices = 0;
        CHECK(cnrtGetDeviceCount(&devices));
        if (devices != 1) throw std::runtime_error("requires one visible, exclusively allocated MLU");
        CHECK(cnrtSetDevice(0));
        Queue queue;
        if (benchmark)
            std::cout << "dtype,rows,width,warmups,repeats,kernel_mean_us,e2e_mean_us\n";
        bool failed = false;
        for (int i = 3; i < argc; ++i) {
            try { run(argv[i], argv[2], queue.value, benchmark); }
            catch (const std::exception &err) {
                std::cerr << argv[i] << ": " << err.what() << '\n';
                failed = true;
            }
        }
        queue.close();
        return failed ? 1 : 0;
    } catch (const std::exception &err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
}
