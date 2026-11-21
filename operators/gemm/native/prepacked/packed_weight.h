#ifndef MLU_PACKED_WEIGHT_H
#define MLU_PACKED_WEIGHT_H
#include <cnrt.h>
#include <stddef.h>
#include <stdint.h>

namespace mlu {
enum class WeightType { fp16, bf16 };
class PackedWeight {
public:
    PackedWeight(const PackedWeight&) = delete;
    PackedWeight& operator=(const PackedWeight&) = delete;
    const uint16_t *data() const { return data_; }
    size_t bytes() const { return bytes_; }
    int k() const { return k_; }
    int n() const { return n_; }
    int device() const { return device_; }
    WeightType type() const { return type_; }
private:
    PackedWeight() = default;
    ~PackedWeight() = default;
    uint16_t *data_ = nullptr;
    size_t bytes_ = 0;
    int k_ = 0, n_ = 0, device_ = -1;
    WeightType type_ = WeightType::fp16;
    friend cnrtRet_t allocate_weight(int, int, WeightType, PackedWeight *&);
    friend cnrtRet_t release_packed_weight(PackedWeight *&);
};

// Host path is synchronous. Device path enqueues packing on q. Keep device B
// alive until q completes. Publish the weight to other queues only after that.
struct HostPackTiming { double pack_us = 0, upload_us = 0; };
cnrtRet_t pack_rhs_host(const uint16_t *b, int k, int n, WeightType, PackedWeight *&, HostPackTiming * = nullptr);
cnrtRet_t pack_rhs_device(const uint16_t *b, int k, int n, WeightType, cnrtQueue_t, PackedWeight *&,
                           cnrtNotifier_t begin = nullptr, cnrtNotifier_t end = nullptr);
cnrtRet_t gemm_prepacked(const uint16_t *a, const PackedWeight &, float *c, int m, cnrtQueue_t);
// Caller must synchronize every queue that can still refer to the weight.
cnrtRet_t release_packed_weight(PackedWeight *&);
}
#endif
