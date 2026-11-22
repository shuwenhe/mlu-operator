#ifndef PHASE11_DENSE_DECODER_H
#define PHASE11_DENSE_DECODER_H
#include "../phase10/mla.h"

namespace native_mlu {

struct DenseWeights {
    MlaWeights attention;
    std::array<Registry::Lease,3> ffn;
    const uint16_t *input_gamma;
    const uint16_t *post_gamma;
};

// BF16 Layer 0. The caller supplies device-resident embeddings or hidden states.
class DenseDecoder {
public:
    struct Buffers {
        uint16_t *input, *residual, *post_norm, *swiglu, *ffn, *output;
        float *gate, *up, *down;
    };
    DenseDecoder(Worker &,Memory &,size_t,DenseWeights,int max_chunk,int capacity);
    ~DenseDecoder();
    DenseDecoder(const DenseDecoder &)=delete;
    void run(int cursor,int count);
    const Buffers &buffers() const {return b_;}
    const MlaRequest &attention() const {return mla_;}
private:
    Worker &worker_;
    Memory &memory_;
    size_t queue_;
    DenseWeights weights_;
    MlaRequest mla_;
    Buffers b_{};
    std::vector<void*> owned_;
    uint16_t *rounded_=nullptr;
    float *norm_=nullptr;
    bool failed_=false;
};
}
#endif
