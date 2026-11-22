#include "../attention_qk.h"
#include "../av/av.h"
#include "../softmax/softmax.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

static void checked(cnrtRet_t result) {
    if (result != cnrtSuccess) {
        std::fprintf(stderr, "native chain CNRT error %d\n", int(result));
        std::exit(1);
    }
}
static void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

template <typename T> struct Device {
    static constexpr size_t guard_count = 16;
    size_t count;
    T *data;
    explicit Device(size_t n): count(n) {
        void *allocation = nullptr;
        checked(cnrtMalloc(&allocation, (count + guard_count) * sizeof(T)));
        data = static_cast<T *>(allocation);
        std::array<unsigned char, guard_count * sizeof(T)> guard;
        guard.fill(0xa5);
        checked(cnrtMemcpy(data + count, guard.data(), guard.size(), cnrtMemcpyHostToDev));
    }
    Device(const Device &) = delete;
    ~Device() { checked(cnrtFree(data)); }
    void upload(std::vector<T> &host) {
        require(host.size() == count, "unexpected input size");
        checked(cnrtMemcpy(data, host.data(), count * sizeof(T), cnrtMemcpyHostToDev));
    }
    std::vector<T> download() const {
        std::vector<T> result(count);
        checked(cnrtMemcpy(result.data(), data, count * sizeof(T), cnrtMemcpyDevToHost));
        return result;
    }
    void verify_guard() const {
        std::array<unsigned char, guard_count * sizeof(T)> guard;
        checked(cnrtMemcpy(guard.data(), data + count, guard.size(), cnrtMemcpyDevToHost));
        for (auto byte : guard) require(byte == 0xa5, "device allocation guard changed");
    }
    size_t bytes() const { return (count + guard_count) * sizeof(T); }
};

template <typename T> static std::vector<T> input(size_t count) {
    std::vector<T> result(count);
    require(std::fread(result.data(), sizeof(T), count, stdin) == count, "truncated chain input");
    return result;
}
template <typename T> static void output(const std::vector<T> &data) {
    require(std::fwrite(data.data(), sizeof(T), data.size(), stdout) == data.size(),
            "could not write chain output");
}

struct Shape {
    uint32_t seq, keys, offset;
    float scale;
    bool bf16, unmasked;
};
struct Chain {
    static constexpr uint32_t heads = 16;
    Shape shape;
    cnrtQueue_t queue;
    cnrtNotifier_t marks[4];
    Device<uint16_t> q, k, v, scaled, low_prob, low_av;
    Device<uint32_t> positions;
    Device<float> raw_qk, masked, maximum, denominator, probability, raw_av;

    explicit Chain(Shape s)
        : shape(s), q(heads*s.seq*192), k(heads*s.keys*192), v(heads*s.keys*128),
          scaled(heads*s.seq*s.keys), low_prob(scaled.count), low_av(heads*s.seq*128),
          positions(s.seq), raw_qk(scaled.count), masked(scaled.count), maximum(heads*s.seq),
          denominator(maximum.count), probability(scaled.count), raw_av(low_av.count) {
        checked(cnrtQueueCreate(&queue));
        for (auto &mark : marks) checked(cnrtNotifierCreate(&mark));
    }
    ~Chain() {
        for (auto mark : marks) checked(cnrtNotifierDestroy(mark));
        checked(cnrtQueueDestroy(queue));
    }
    void submit() {
        const auto s = shape;
        const auto qk = s.bf16 ? phase9_launch_qk_bf16 : phase9_launch_qk_fp16;
        const auto softmax = s.bf16 ? phase9_softmax_bf16 : phase9_softmax_fp16;
        const auto av = s.bf16 ? phase9_av_bf16 : phase9_av_fp16;
        checked(cnrtPlaceNotifier(marks[0], queue));
        checked(qk(queue, q.data, k.data, positions.data, s.offset, s.scale, raw_qk.data,
                   scaled.data, masked.data, heads, s.seq, s.keys, s.keys));
        checked(cnrtPlaceNotifier(marks[1], queue));
        checked(softmax(queue, masked.data, nullptr, 0, maximum.data, denominator.data,
                        probability.data, low_prob.data, heads*s.seq, s.keys, s.keys));
        checked(cnrtPlaceNotifier(marks[2], queue));
        checked(av(queue, low_prob.data, v.data, raw_av.data, low_av.data,
                   heads, s.seq, s.keys, s.keys));
        checked(cnrtPlaceNotifier(marks[3], queue));
    }
    void sync() { checked(cnrtQueueSync(queue)); }
    void timing(unsigned slot, int sample, double wall) {
        float qk, softmax, av, chain;
        checked(cnrtNotifierDuration(marks[0], marks[1], &qk));
        checked(cnrtNotifierDuration(marks[1], marks[2], &softmax));
        checked(cnrtNotifierDuration(marks[2], marks[3], &av));
        checked(cnrtNotifierDuration(marks[0], marks[3], &chain));
        const size_t bytes = q.bytes()+k.bytes()+v.bytes()+scaled.bytes()+low_prob.bytes()+
            low_av.bytes()+positions.bytes()+raw_qk.bytes()+masked.bytes()+maximum.bytes()+
            denominator.bytes()+probability.bytes()+raw_av.bytes();
        std::fprintf(stderr, "TIMING,%u,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%zu\n",
                     slot, sample, qk, softmax, av, chain, wall, bytes);
    }
    void output_checked() {
        q.verify_guard(); k.verify_guard(); v.verify_guard(); scaled.verify_guard();
        low_prob.verify_guard(); low_av.verify_guard(); positions.verify_guard();
        raw_qk.verify_guard(); masked.verify_guard(); maximum.verify_guard();
        denominator.verify_guard(); probability.verify_guard(); raw_av.verify_guard();
        output(masked.download()); output(low_prob.download());
        output(raw_av.download()); output(low_av.download());
    }
};

int main(int argc, char **argv) {
    if (argc != 9) return 2;
    checked(cnrtSetDevice(0));
    const auto u32 = [&](int n) { return uint32_t(std::strtoul(argv[n], nullptr, 10)); };
    const unsigned slots = u32(6), repeats = u32(7);
    require((slots == 1 || slots == 2) && repeats > 0, "invalid queue or repeat count");
    const Shape s{u32(2), u32(3), u32(4), std::strtof(argv[5], nullptr),
                  std::strcmp(argv[1], "bf16") == 0, u32(8) != 0};
    Chain first(s);
    std::vector<Chain *> queues{&first};
    std::unique_ptr<Chain> second;
    if (slots == 2) {
        second.reset(new Chain(s));
        queues.push_back(second.get());
    }
    for (auto *chain : queues) {
        auto q = input<uint16_t>(chain->q.count), k = input<uint16_t>(chain->k.count),
             v = input<uint16_t>(chain->v.count);
        auto pos = input<uint32_t>(chain->positions.count);
        if (s.unmasked)
            for (auto &p : pos) p = s.keys - 1;
        chain->q.upload(q); chain->k.upload(k); chain->v.upload(v);
        chain->positions.upload(pos);
    }
    std::vector<std::vector<uint16_t>> baseline(slots);
    for (unsigned sample = 0; sample <= repeats; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (auto *chain : queues) chain->submit();
        for (auto *chain : queues) chain->sync();
        const double wall = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        for (unsigned slot = 0; slot < slots; ++slot) {
            auto actual = queues[slot]->low_av.download();
            if (!sample) baseline[slot] = actual;
            else {
                require(baseline[slot] == actual, "AV output changed across repeated submissions");
                queues[slot]->timing(slot, sample, wall);
            }
        }
    }
    for (auto *chain : queues) chain->output_checked();
    return 0;
}
