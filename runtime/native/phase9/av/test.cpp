#include "av.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cnrtRet_t rc) {
    if (rc != cnrtSuccess) {
        std::fprintf(stderr, "AV CNRT error %d\n", int(rc));
        std::exit(1);
    }
}

template<class T>
struct Buffer {
    T *data;
    std::vector<T> host;
    explicit Buffer(size_t count) : host(count) {
        void *pointer = nullptr;
        check(cnrtMalloc(&pointer, count * sizeof(T)));
        data = static_cast<T *>(pointer);
    }
    ~Buffer() { check(cnrtFree(data)); }
    bool upload() {
        if (std::fread(host.data(), sizeof(T), host.size(), stdin) != host.size()) return false;
        check(cnrtMemcpy(data, host.data(), host.size() * sizeof(T), cnrtMemcpyHostToDev));
        return true;
    }
    bool download() {
        check(cnrtMemcpy(host.data(), data, host.size() * sizeof(T), cnrtMemcpyDevToHost));
        return std::fwrite(host.data(), sizeof(T), host.size(), stdout) == host.size();
    }
};

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const auto launch = std::strcmp(argv[1], "bf16") == 0 ? phase9_av_bf16 : phase9_av_fp16;
    const uint32_t heads = std::strtoul(argv[2], nullptr, 10);
    const uint32_t seq = std::strtoul(argv[3], nullptr, 10);
    const uint32_t length = std::strtoul(argv[4], nullptr, 10);
    const uint32_t capacity = std::strtoul(argv[5], nullptr, 10);
    check(cnrtSetDevice(0));
    cnrtQueue_t queue;
    check(cnrtQueueCreate(&queue));
    Buffer<uint16_t> probs(size_t(heads) * seq * length), values(size_t(heads) * capacity * 128),
                     low(size_t(heads) * seq * 128);
    Buffer<float> raw(size_t(heads) * seq * 128);
    if (!probs.upload() || !values.upload()) return 2;
    check(launch(queue, probs.data, values.data, raw.data, low.data, heads, seq, length, capacity));
    check(cnrtQueueSync(queue));
    const bool ok = raw.download() && low.download();
    check(cnrtQueueDestroy(queue));
    return ok ? 0 : 1;
}
