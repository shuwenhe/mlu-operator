#include "attention_qk.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static void check(cnrtRet_t status) {
    if (status != cnrtSuccess) {
        std::fprintf(stderr, "CNRT error %d\n", static_cast<int>(status));
        std::exit(1);
    }
}

template<class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        void *storage = nullptr;
        check(cnrtMalloc(&storage, count * sizeof(T)));
        data = static_cast<T *>(storage);
    }
    ~DeviceBuffer() { check(cnrtFree(data)); }

    void upload(const std::vector<T> &host) {
        check(cnrtMemcpy(data, const_cast<T *>(host.data()), count_ * sizeof(T), cnrtMemcpyHostToDev));
    }
    std::vector<T> download() const {
        std::vector<T> host(count_);
        check(cnrtMemcpy(host.data(), data, count_ * sizeof(T), cnrtMemcpyDevToHost));
        return host;
    }

    T *data;
private:
    size_t count_;
};

static bool test(cnrtQueue_t queue, bool bf16, uint32_t cache_base) {
    constexpr uint32_t heads = 2, seq = 2, length = 2, capacity = 3, dim = 192;
    const uint16_t one = bf16 ? 0x3f80 : 0x3c00;
    const uint16_t half_ulp = bf16 ? 0x3b80 : 0x1000;
    const uint16_t three = bf16 ? 0x4040 : 0x4200;
    const float eps = bf16 ? 1.f / 256 : 1.f / 2048;
    std::vector<uint16_t> q(heads * seq * dim), k(heads * capacity * dim);
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t s = 0; s < seq; ++s) {
            q[(head * seq + s) * dim] = one;
            q[(head * seq + s) * dim + 1] = one;
        }
        for (uint32_t key = 0; key < length; ++key) {
            const size_t offset = (head * capacity + key) * dim;
            k[offset] = head == key ? one : 0;
            k[offset + 1] = head == key ? half_ulp : 0;
        }
    }

    DeviceBuffer<uint16_t> dq(q.size()), dk(k.size()), scaled(heads * seq * length);
    DeviceBuffer<uint32_t> positions(seq);
    DeviceBuffer<float> raw(heads * seq * length), masked(heads * seq * length);
    dq.upload(q);
    dk.upload(k);
    positions.upload({0, 1});
    const auto launch = bf16 ? phase9_launch_qk_bf16 : phase9_launch_qk_fp16;
    check(launch(queue, dq.data, dk.data, positions.data, cache_base, 3.0f,
                 raw.data, scaled.data, masked.data, heads, seq, length, capacity));
    check(cnrtQueueSync(queue));

    const auto dots = raw.download();
    const auto scores = scaled.download();
    const auto masks = masked.download();
    unsigned mismatches = 0;
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t s = 0; s < seq; ++s) {
            for (uint32_t key = 0; key < length; ++key) {
                const size_t i = (head * seq + s) * length + key;
                const float expected_mask = key > cache_base + s ? -INFINITY :
                    (head == key ? 3.f : 0.f);
                mismatches += (dots[i] != (head == key ? 1.f + eps : 0.f)
                    || scores[i] != (head == key ? three : 0)
                    || masks[i] != expected_mask);
            }
        }
    }
    std::printf("dtype=%s heads=%u seq=%u length=%u capacity=%u base=%u mismatches=%u\n",
                bf16 ? "BF16" : "FP16", heads, seq, length, capacity, cache_base, mismatches);
    return mismatches == 0;
}

template<class T>
static std::vector<T> read_input(size_t count) {
    std::vector<T> result(count);
    if (std::fread(result.data(), sizeof(T), count, stdin) != count) {
        std::fputs("Incomplete QK input\n", stderr);
        std::exit(1);
    }
    return result;
}

template<class T>
static bool write_output(const std::vector<T> &data) {
    return std::fwrite(data.data(), sizeof(T), data.size(), stdout) == data.size();
}

static bool replay(cnrtQueue_t queue, char **argv) {
    constexpr uint32_t heads = 16, dim = 192;
    const auto launch = std::strcmp(argv[2], "bf16") == 0 ?
        phase9_launch_qk_bf16 : phase9_launch_qk_fp16;
    const uint32_t seq = std::strtoul(argv[3], nullptr, 10);
    const uint32_t length = std::strtoul(argv[4], nullptr, 10);
    const uint32_t base = std::strtoul(argv[5], nullptr, 10);
    const float scale = std::strtof(argv[6], nullptr);
    const size_t q_count = size_t(heads) * seq * dim;
    const size_t k_count = size_t(heads) * length * dim;
    const size_t scores = size_t(heads) * seq * length;
    DeviceBuffer<uint16_t> q(q_count), k(k_count), scaled(scores);
    DeviceBuffer<uint32_t> pos(seq);
    DeviceBuffer<float> raw(scores), masked(scores);
    q.upload(read_input<uint16_t>(q_count));
    k.upload(read_input<uint16_t>(k_count));
    pos.upload(read_input<uint32_t>(seq));
    check(launch(queue, q.data, k.data, pos.data, base, scale, raw.data,
                 scaled.data, masked.data, heads, seq, length, length));
    check(cnrtQueueSync(queue));
    return write_output(raw.download()) && write_output(scaled.download()) &&
        write_output(masked.download()) && std::fflush(stdout) == 0;
}

int main(int argc, char **argv) {
    if (argc != 1 && argc != 2 && argc != 7) {
        std::fprintf(stderr, "usage: %s [device [bf16|fp16 seq length base scale]]\n", argv[0]);
        return 2;
    }
    check(cnrtSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
    cnrtQueue_t queue;
    check(cnrtQueueCreate(&queue));
    if (argc == 7) {
        const bool ok = replay(queue, argv);
        check(cnrtQueueDestroy(queue));
        return ok ? 0 : 1;
    }
    bool ok = true;
    for (bool bf16 : {false, true}) {
        for (uint32_t base : {0u, 1u}) {
            ok = test(queue, bf16, base) && ok;
        }
    }
    check(cnrtQueueDestroy(queue));
    return ok ? 0 : 1;
}
