#ifndef PHASE10_NATIVE_MLA_H
#define PHASE10_NATIVE_MLA_H
#include "../workspace.h"
#include <array>
#include <cstdint>
#include <vector>

namespace native_mlu {

struct MlaWeights {
    std::array<Registry::Lease, 4> projections; // Q, KV-A, KV-B, O.
    const uint16_t *latent_gamma;
    const uint16_t *cosine;
    const uint16_t *sine;
    int table_rows;
    float scale;
};

// One independently committed materialized cache per request. All table pointers
// must remain valid for the request lifetime. Use the worker's owning host thread.
class MlaRequest {
public:
    struct Tensors {
        uint16_t *input, *compressed, *latent, *norm_round, *rope_in, *rope_out;
        uint16_t *query, *key, *value, *scaled, *prob, *av, *merged, *output;
        float *q_projection, *kv_a, *kv_b, *norm_fp32, *raw_qk, *masked;
        float *maximum, *denominator, *prob_fp32, *av_fp32, *output_fp32;
        uint32_t *positions;
    };

    MlaRequest(Worker &, Memory &, size_t queue, MlaWeights, int max_chunk,
               int capacity, bool bf16, size_t cache_guard_bytes = 0);
    ~MlaRequest();
    MlaRequest(const MlaRequest &) = delete;

    // Input is written by the caller before enqueue on the request's queue.
    // Only successfully completed appends update length().
    // Optional caller-owned notifiers delimit start, pre-append, post-append,
    // post-attention, and end. Keep them alive through complete().
    void enqueue(int cursor, int count, const std::array<cnrtNotifier_t, 5> *markers = nullptr);
    void complete();
    void reset();
    int length() const { return length_; }
    int capacity() const { return capacity_; }
    const Tensors &tensors() const { return t_; }

private:
    void *allocate(size_t bytes, Kind kind);
    void release();
    Worker &worker_;
    Memory &memory_;
    size_t queue_;
    MlaWeights weights_;
    int max_chunk_, capacity_, length_ = 0, next_ = 0;
    bool bf16_, pending_ = false;
    std::vector<void *> owned_;
    Tensors t_{};
};
}
#endif
