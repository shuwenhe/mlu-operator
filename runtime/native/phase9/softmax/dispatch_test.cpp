#include "dispatch.h"
#include <cmath>
#include <cstdio>

using namespace native_mlu;

static void require(bool ok) {
    if (!ok) throw std::runtime_error("Softmax dispatch assertion failed");
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    Worker worker(argv[1]);
    Memory memory(worker);
    Workspace slots(worker, memory);
    Dispatch dispatch(worker, memory, slots);
    auto allocate = [&](int rows, int cols, Type type) {
        const size_t bytes = size_t(rows) * cols * (type == Type::fp32 ? 4 : 2);
        return View{memory.allocate(bytes, Kind::workspace), rows, cols, cols, type};
    };
    for (auto type : {Type::bf16, Type::fp16}) {
        const auto slot = slots.add(worker.add_queue(), 2, 4);
        auto scores = allocate(2, 3, Type::fp32), maxes = allocate(2, 1, Type::fp32),
             denom = allocate(2, 1, Type::fp32), probs = allocate(2, 3, Type::fp32),
             output = allocate(2, 3, type), mask = allocate(1, 3, type);
        float input[] = {0.f, -INFINITY, 16.f, -INFINITY, 0.f, 16.f};
        uint16_t zero_mask[] = {0, 0, 0};
        check(cnrtMemcpy(scores.data, input, sizeof(input), cnrtMemcpyHostToDev), "upload scores");
        check(cnrtMemcpy(mask.data, zero_mask, sizeof(zero_mask), cnrtMemcpyHostToDev), "upload mask");
        slots.start(slot);
        dispatch.attention_softmax(slot, scores, 2, maxes, denom, probs, output, mask);
        auto *pinned = mask.data;
        bool release_blocked = false;
        try { memory.release(pinned); }
        catch (const std::logic_error &) { release_blocked = true; }
        require(release_blocked);
        slots.finish(slot);
        uint16_t actual[6];
        check(cnrtMemcpy(actual, output.data, sizeof(actual), cnrtMemcpyDevToHost), "read Softmax");
        const uint16_t one = type == Type::bf16 ? 0x3f80 : 0x3c00;
        require(actual[0] == one && actual[4] == one &&
                actual[1] == 0 && actual[2] == 0 && actual[3] == 0 && actual[5] == 0);
        for (auto view : {scores, maxes, denom, probs, output, mask}) memory.release(view.data);
    }
    slots.close();
    memory.close();
    worker.close();
    std::puts("PASS native Softmax dispatch for both dtypes, queue completion and buffer retention");
}
