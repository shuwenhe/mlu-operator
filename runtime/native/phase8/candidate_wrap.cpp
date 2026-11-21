// Linked only into the experimental Layer 0 executable. Never retries production GEMM.
#include "packed_weight.h"

extern "C" cnrtRet_t phase8_candidate_fp16(cnrtQueue_t, const uint16_t *, const uint16_t *, float *,
                                            int, int, int);
extern "C" cnrtRet_t phase8_candidate_bf16(cnrtQueue_t, const uint16_t *, const uint16_t *, float *,
                                            int, int, int);

extern "C" cnrtRet_t __wrap__ZN3mlu14gemm_prepackedEPKtRKNS_12PackedWeightEPfiP10CNqueue_st(
    const uint16_t *a, const mlu::PackedWeight &weight, float *out, int rows, cnrtQueue_t queue) {
    int selected = -1;
    const auto status = cnrtGetDevice(&selected);
    if (status != cnrtSuccess) return status;
    if (selected != weight.device()) return cnrtErrorDeviceInvalid;
    const auto launch = weight.type() == mlu::WeightType::bf16 ? phase8_candidate_bf16 :
                                                               phase8_candidate_fp16;
    return launch(queue, a, weight.data(), out, rows, weight.n(), weight.k());
}
