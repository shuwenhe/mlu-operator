#include "packed_weight.h"

extern "C" cnrtRet_t phase8_candidate_bf16(cnrtQueue_t, const uint16_t *,
    const uint16_t *, float *, int, int, int);

extern "C" cnrtRet_t phase124_pack(cnrtQueue_t queue, const uint16_t *kn, int k, int n,
    mlu::PackedWeight **packed) {
    if (!packed || *packed || !queue || !kn || k <= 0 || k > 10944 || k % 64 ||
        n <= 0 || n > 10944 || n % 64)
        return cnrtErrorArgsInvalid;
    return mlu::pack_rhs_device(kn, k, n, mlu::WeightType::bf16, queue, *packed);
}

extern "C" cnrtRet_t phase124_project(cnrtQueue_t queue, const uint16_t *x, float *out,
    const mlu::PackedWeight *packed, int rows) {
    if (!packed) return cnrtErrorArgsInvalid;
    return phase8_candidate_bf16(queue, x, packed->data(), out, rows, packed->n(), packed->k());
}

extern "C" cnrtRet_t phase124_release(mlu::PackedWeight **packed) {
    if (!packed) return cnrtErrorArgsInvalid;
    return mlu::release_packed_weight(*packed);
}
