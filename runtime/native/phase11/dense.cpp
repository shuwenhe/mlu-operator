#include "dense.h"
#include "../phase10/layout.h"
#include <cstdio>
#include <utility>

extern "C" {
cnrtRet_t phase8_candidate_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,float*,int,int,int);
cnrtRet_t phase82_native_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,int,int,float);
cnrtRet_t phase11_residual(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,int);
cnrtRet_t phase11_swiglu(cnrtQueue_t,const float*,const float*,uint16_t*,int);
}
namespace native_mlu {
namespace {
constexpr int hidden=2048, width=10944;
}

DenseDecoder::DenseDecoder(Worker &worker,Memory &memory,size_t queue,DenseWeights weights,
                           int max_chunk,int capacity)
    :worker_(worker),memory_(memory),queue_(queue),weights_(std::move(weights)),
     mla_(worker,memory,queue,weights_.attention,max_chunk,capacity,true) {
    const int k[]={hidden,hidden,width}, n[]={width,width,hidden};
    for (size_t i=0;i<3;++i) {
        const auto &w=weights_.ffn[i];
        if (!w||w->device()!=worker_.ordinal()||w->type()!=mlu::WeightType::bf16||
            w->k()!=k[i]||w->n()!=n[i]) throw std::invalid_argument("invalid dense FFN weights");
    }
    for (auto *g:{weights_.input_gamma,weights_.post_gamma})
        if (memory_.capacity(g)<hidden*2) throw std::invalid_argument("invalid decoder norm weight");
    owned_.reserve(11);
    const auto alloc=[&](size_t size) {
        auto *p=memory_.allocate(size,Kind::workspace);
        owned_.push_back(p);
        return p;
    };
    const auto low=[&](size_t cols){return static_cast<uint16_t*>(alloc(cols*max_chunk*2));};
    const auto wide=[&](size_t cols){return static_cast<float*>(alloc(cols*max_chunk*4));};
    try {
        b_=Buffers{low(hidden),low(hidden),low(hidden),low(width),low(hidden),low(hidden),
                   wide(width),wide(width),wide(hidden)};
        rounded_=low(hidden);
        norm_=wide(hidden);
        memory_.pin(weights_.input_gamma);
        memory_.pin(weights_.post_gamma);
    } catch (...) {
        for (auto &p:owned_) memory_.release(p);
        throw;
    }
}

DenseDecoder::~DenseDecoder() {
    try {
        worker_.sync(queue_);
        for (auto &p:owned_) memory_.release(p);
        memory_.unpin(weights_.post_gamma);
        memory_.unpin(weights_.input_gamma);
    } catch (const std::exception &e) {
        std::fprintf(stderr,"dense decoder teardown failed: %s\n",e.what());
        std::terminate();
    }
}

void DenseDecoder::run(int cursor,int count) {
    if (failed_) throw std::logic_error("decoder request failed");
    if (cursor!=mla_.length()||count<=0||count>mla_.capacity()-cursor||
        size_t(count)*hidden*2>memory_.capacity(b_.input))
        throw std::invalid_argument("invalid dense decoder positions");
    const auto q=worker_.queue(queue_);
    const auto normalize=[&](const uint16_t *in,const uint16_t *gamma,uint16_t *out) {
        check(phase82_native_bf16(q,in,gamma,out,norm_,rounded_,count,hidden,1e-6f),"decoder RMSNorm");
    };
    const auto project=[&](size_t slot,const uint16_t *input,float *out) {
        auto &w=*weights_.ffn[slot];
        check(phase8_candidate_bf16(q,input,w.data(),out,count,w.n(),w.k()),"dense FFN projection");
    };
    try {
        normalize(b_.input,weights_.input_gamma,mla_.tensors().input);
        mla_.enqueue(cursor,count);
        check(phase11_residual(q,b_.input,mla_.tensors().output,b_.residual,count*hidden),"attention residual");
        normalize(b_.residual,weights_.post_gamma,b_.post_norm);
        project(0,b_.post_norm,b_.gate);
        project(1,b_.post_norm,b_.up);
        check(phase11_swiglu(q,b_.gate,b_.up,b_.swiglu,count*width),"dense SwiGLU");
        project(2,b_.swiglu,b_.down);
        check(phase10_cast_bf16(q,b_.down,b_.ffn,count),"dense FFN cast");
        check(phase11_residual(q,b_.residual,b_.ffn,b_.output,count*hidden),"FFN residual");
        mla_.complete();
    } catch (...) {
        failed_=true;
        throw;
    }
}
}
