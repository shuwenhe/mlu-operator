#include <cnrt.h>
#include "gemm_common.h"
extern "C" {
#include "mlu_gemm.h"
#include "mlu_rmsnorm.h"
}
#include "mlu_gemm_reference.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T> void read(std::istream &stream, T *data, size_t count) {
    if (!stream.read(reinterpret_cast<char *>(data), count * sizeof(T)))
        throw std::runtime_error("truncated fixture");
}

static float unpack(uint16_t bits, mlu_dtype dtype) {
    if (dtype == MLU_OPERATOR_BF16) {
        uint32_t wide = uint32_t(bits) << 16;
        float value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    }
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    if (exponent == 31)
        throw std::runtime_error("nonfinite fixture input");
    const float value = std::ldexp(float(exponent ? 1024 + fraction : fraction),
                                  (exponent ? int(exponent) : 1) - 25);
    return bits & 0x8000 ? -value : value;
}

static mlu_status bang_gemm(mlu_dtype dtype, int m, int n, int k,
                           const std::vector<uint16_t> &a,
                           const std::vector<uint16_t> &b, std::vector<float> &c) {
    cnrtQueue_t queue = nullptr;
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    mlu_status status{MLU_OPERATOR_OK, 0, nullptr};
    auto check = [&](cnrtRet_t code, const char *operation) {
        if (code != cnrtSuccess && status.domain == MLU_OPERATOR_OK)
            status = {MLU_OPERATOR_CNRT, int(code), operation};
        return code == cnrtSuccess;
    };
    const auto launch = dtype == MLU_OPERATOR_BF16 ? mlu_bang_gemm_bf16 : mlu_bang_gemm_fp16;
    const auto ab = a.size()*sizeof(uint16_t), bb = b.size()*sizeof(uint16_t);
    const auto cb = c.size()*sizeof(float);
    if (check(cnrtSetDevice(0), "set device") &&
        check(cnrtQueueCreate(&queue), "create queue") &&
        check(cnrtMalloc(&da, ab), "allocate A") &&
        check(cnrtMalloc(&db, bb), "allocate B") &&
        check(cnrtMalloc(&dc, cb), "allocate C") &&
        check(cnrtMemcpy(da, const_cast<uint16_t *>(a.data()), ab, cnrtMemcpyHostToDev), "copy A") &&
        check(cnrtMemcpy(db, const_cast<uint16_t *>(b.data()), bb, cnrtMemcpyHostToDev), "copy B")) {
        for (int i = 0; i < 3; ++i)
            if (!check(launch(queue, static_cast<const uint16_t *>(da),
                              static_cast<const uint16_t *>(db), static_cast<float *>(dc),
                              m, n, k), "launch BANG GEMM"))
                break;
        if (check(cnrtQueueSync(queue), "synchronize BANG GEMM") &&
            status.domain == MLU_OPERATOR_OK)
            check(cnrtMemcpy(c.data(), dc, cb, cnrtMemcpyDevToHost), "copy result");
    }
    if (dc) check(cnrtFree(dc), "free C");
    if (db) check(cnrtFree(db), "free B");
    if (da) check(cnrtFree(da), "free A");
    if (queue) check(cnrtQueueDestroy(queue), "destroy queue");
    return status;
}

static bool run(const char *path, bool reference_only, bool bang) {
    std::ifstream stream(path, std::ios::binary);
    char magic[8];
    uint32_t fields[5];
    float epsilon;
    read(stream, magic, 8);
    read(stream, fields, 5);
    read(stream, &epsilon, 1);
    if (std::memcmp(magic, "DSV2OP01", 8) || fields[0] > 1 || fields[1] > 1 ||
        !fields[2] || !fields[3] || !fields[4] ||
        fields[2] > 64 || fields[3] > 16384 || fields[4] > 16384 ||
        (fields[0] && fields[3] != fields[4]))
        throw std::runtime_error("invalid fixture header");
    const bool norm = fields[0];
    const auto dtype = static_cast<mlu_dtype>(fields[1]);
    const int m = fields[2], n = fields[3], k = fields[4];
    std::vector<uint16_t> a(size_t(m)*k);
    std::vector<float> af(a.size()), expected(size_t(m)*n), actual(expected.size(), NAN);
    read(stream, a.data(), a.size());
    for (size_t i = 0; i < a.size(); ++i)
        af[i] = unpack(a[i], dtype);

    mlu_status status{MLU_OPERATOR_OK, 0, nullptr};
    mlu_gemm_metrics metrics{};
    if (norm) {
        std::vector<float> gamma(k);
        read(stream, gamma.data(), gamma.size());
        for (int i = 0; i < m; ++i) {
            const size_t start = size_t(i)*k;
            float sum = 0;
            for (int j = 0; j < k; ++j)
                sum += af[start+j]*af[start+j];
            const float inv = 1.0f/std::sqrt(sum/k + epsilon);
            for (int j = 0; j < k; ++j)
                expected[start+j] = af[start+j]*inv*gamma[j];
        }
        if (!reference_only) status = mlu_rmsnorm(0, dtype, m, k, epsilon, a.data(), gamma.data(), actual.data());
    } else {
        std::vector<uint16_t> b(size_t(k)*n);
        std::vector<float> bf(b.size());
        read(stream, b.data(), b.size());
        for (size_t i = 0; i < b.size(); ++i)
            bf[i] = unpack(b[i], dtype);
        mlu_gemm_reference(af.data(), bf.data(), expected.data(), m, n, k, k, n, n);
        if (!reference_only)
            status = bang ? bang_gemm(dtype, m, n, k, a, b, actual)
                          : mlu_gemm(0, dtype, m, n, k, a.data(), b.data(), actual.data(), 3, &metrics);
    }
    if (stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("unexpected trailing fixture data");
    if (reference_only) {
        double checksum = 0;
        for (const float value : expected) {
            if (!std::isfinite(value))
                throw std::runtime_error("nonfinite CPU reference");
            checksum += value;
        }
        std::cout << path << ',' << expected.size() << ',' << checksum << '\n';
        return true;
    }
    if (status.domain != MLU_OPERATOR_OK) {
        std::cerr << path << ": " << status.operation << " domain=" << status.domain
                  << " code=" << status.code << '\n';
        return false;
    }

    double max_abs = 0, max_rel = 0, sum_abs = 0;
    size_t nan = 0, inf = 0, mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            nan += std::isnan(actual[i]);
            inf += std::isinf(actual[i]);
            continue;
        }
        const double error = std::abs(double(actual[i]) - expected[i]);
        const double scale = std::abs(double(expected[i]));
        max_abs = std::fmax(max_abs, error);
        max_rel = std::fmax(max_rel, error / std::fmax(scale, 1e-6));
        sum_abs += error;
        mismatches += error > 1e-4 + 1e-4 * scale;
    }
    std::cout << path << ',' << max_abs << ',' << max_rel << ',' << sum_abs/actual.size()
              << ',' << nan << ',' << inf << ',' << mismatches << ',';
    if (!norm && !bang) std::cout << metrics.kernel_us;
    std::cout << '\n';
    return !(nan || inf || mismatches);
}

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool reference_only = mode == "--reference-only";
    if (argc < 3 || (!reference_only && mode != "--authorized-cnnl" && mode != "--authorized-bang")) {
        std::cerr << "usage: runner --reference-only|--authorized-cnnl|--authorized-bang <fixture>...\n";
        return 2;
    }
    if (!reference_only) {
        unsigned count = 0;
        const auto status = cnrtGetDeviceCount(&count);
        if (status != cnrtSuccess || count != 1) {
            std::cerr << "requires exactly one reserved visible MLU; CNRT status=" << status
                      << ", count=" << count << '\n';
            return 2;
        }
        std::cout << "fixture,max_abs,max_rel,mean_abs,nan,inf,mismatches,kernel_us\n";
    } else {
        std::cout << "fixture,cpu_reference_elements,checksum\n";
    }
    bool passed = true;
    for (int i = 2; i < argc; ++i) {
        try {
            if (!run(argv[i], reference_only, mode == "--authorized-bang")) passed = false;
        } catch (const std::exception &error) {
            std::cerr << argv[i] << ": " << error.what() << '\n';
            passed = false;
        }
    }
    return passed ? 0 : 1;
}
