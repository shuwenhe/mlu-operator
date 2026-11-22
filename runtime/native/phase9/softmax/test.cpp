#include "softmax.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cnrtRet_t rc) {
    if (rc != cnrtSuccess) {
        std::fprintf(stderr, "CNRT error %d\n", int(rc));
        std::exit(1);
    }
}

template<class T>
class Buffer {
public:
    explicit Buffer(size_t count) : host(count) {
        void *p = nullptr;
        check(cnrtMalloc(&p, count * sizeof(T)));
        data = static_cast<T *>(p);
    }
    ~Buffer() { check(cnrtFree(data)); }
    bool read() {
        if (std::fread(host.data(), sizeof(T), host.size(), stdin) != host.size()) return false;
        check(cnrtMemcpy(data, host.data(), host.size() * sizeof(T), cnrtMemcpyHostToDev));
        return true;
    }
    bool write() {
        check(cnrtMemcpy(host.data(), data, host.size() * sizeof(T), cnrtMemcpyDevToHost));
        return std::fwrite(host.data(), sizeof(T), host.size(), stdout) == host.size();
    }
    T *data;
    std::vector<T> host;
};

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const auto launch = std::strcmp(argv[1], "bf16") == 0 ?
        phase9_softmax_bf16 : phase9_softmax_fp16;
    const auto rows = std::strtoul(argv[2], nullptr, 10);
    const auto cols = std::strtoul(argv[3], nullptr, 10);
    const auto valid = std::strtoul(argv[4], nullptr, 10);
    const auto mask_rows = std::strtoul(argv[5], nullptr, 10);
    check(cnrtSetDevice(0));
    cnrtQueue_t queue;
    check(cnrtQueueCreate(&queue));
    Buffer<float> input(rows * cols), probs(rows * cols), maxes(rows), denom(rows);
    Buffer<uint16_t> low(rows * cols), mask(mask_rows ? mask_rows * cols : 1);
    if (!input.read() || (mask_rows && !mask.read())) return 2;
    check(launch(queue, input.data, mask_rows ? mask.data : nullptr, mask_rows,
                 maxes.data, denom.data, probs.data, low.data, rows, cols, valid));
    check(cnrtQueueSync(queue));
    const bool written = maxes.write() && denom.write() && probs.write() && low.write();
    check(cnrtQueueDestroy(queue));
    return written ? 0 : 1;
}
