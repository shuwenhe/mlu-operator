#include "dispatch.h"
#include <cstdio>

using namespace native_mlu;

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    Worker worker(argv[1]);
    Memory memory(worker);
    Workspace slots(worker, memory);
    Dispatch dispatch(worker, memory, slots);
    struct Request {
        size_t slot;
        View probs, values, raw, low;
        uint16_t one;
    };
    Request requests[2]{};
    auto allocate = [&](int rows, int cols, Type type) {
        const size_t bytes = size_t(rows) * cols * (type == Type::fp32 ? 4 : 2);
        return View{memory.allocate(bytes, Kind::workspace), rows, cols, cols, type};
    };
    for (unsigned n = 0; n < 2; ++n) {
        const auto type = n == 0 ? Type::bf16 : Type::fp16;
        auto &r = requests[n];
        r = {slots.add(worker.add_queue(), 2, 4), allocate(4, 2, type),
             allocate(6, 128, type), allocate(4, 128, Type::fp32),
             allocate(4, 128, type), uint16_t(n == 0 ? 0x3f80 : 0x3c00)};
        uint16_t p[] = {r.one, 0, 0, r.one, r.one, 0, 0, r.one};
        std::vector<uint16_t> v(6 * 128, 0);
        for (unsigned row : {1u, 3u})
            for (unsigned col = 0; col < 128; ++col) v[row * 128 + col] = r.one;
        check(cnrtMemcpy(r.probs.data, p, sizeof(p), cnrtMemcpyHostToDev), "upload AV probabilities");
        check(cnrtMemcpy(r.values.data, v.data(), v.size() * sizeof(uint16_t), cnrtMemcpyHostToDev),
              "upload AV values");
        slots.start(r.slot);
        dispatch.attention_av(r.slot, r.probs, r.values, r.raw, r.low, 2);
    }
    for (auto &r : requests) {
        auto *early = r.values.data;
        bool pinned = false;
        try { memory.release(early); }
        catch (const std::logic_error &) { pinned = true; }
        if (!pinned) throw std::runtime_error("AV buffer released while slot was active");
        slots.finish(r.slot);
        uint16_t result[4 * 128];
        check(cnrtMemcpy(result, r.low.data, sizeof(result), cnrtMemcpyDevToHost), "read AV output");
        for (unsigned row = 0; row < 4; ++row)
            for (unsigned col = 0; col < 128; ++col)
                if (result[row * 128 + col] != ((row == 1 || row == 2) ? r.one : 0))
                    throw std::runtime_error("AV dispatch output or head stride mismatch");
        for (auto view : {r.probs, r.values, r.raw, r.low}) memory.release(view.data);
    }
    slots.close();
    memory.close();
    worker.close();
    std::puts("PASS native AV dispatch, two queues, padded head stride and request buffer retention");
}
