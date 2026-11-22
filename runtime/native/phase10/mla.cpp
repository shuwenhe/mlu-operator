#include "mla.h"
#include "layout.h"
#include "../phase9/attention_qk.h"
#include "../phase9/av/av.h"
#include "../phase9/softmax/softmax.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

extern "C" {
cnrtRet_t phase8_candidate_fp16(cnrtQueue_t,const uint16_t*,const uint16_t*,float*,int,int,int);
cnrtRet_t phase8_candidate_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,float*,int,int,int);
cnrtRet_t phase82_native_fp16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,int,int,float);
cnrtRet_t phase82_native_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,int,int,float);
cnrtRet_t phase5_cast_kv(cnrtQueue_t,const float*,uint16_t*,int,bool);
cnrtRet_t worker_rope_fp16(cnrtQueue_t,const uint16_t*,const uint16_t*,const uint16_t*,const uint32_t*,uint16_t*,int);
cnrtRet_t worker_rope_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,const uint16_t*,const uint32_t*,uint16_t*,int);
}

namespace native_mlu {
namespace {
constexpr int heads=16, hidden=2048, qwidth=3072, narrow=512, wide=576, expanded=4096;
}

MlaRequest::MlaRequest(Worker &worker, Memory &memory, size_t queue, MlaWeights weights,
                       int max_chunk, int capacity, bool bf16, size_t guard)
    :worker_(worker), memory_(memory), queue_(queue), weights_(std::move(weights)),
     max_chunk_(max_chunk), capacity_(capacity), bf16_(bf16) {
    worker_.queue(queue_);
    const auto type=bf16?mlu::WeightType::bf16:mlu::WeightType::fp16;
    const size_t largest_cache=size_t(capacity)*qwidth*sizeof(uint16_t);
    if (max_chunk<=0 || max_chunk>128 || capacity<max_chunk || capacity>weights_.table_rows ||
        !std::isfinite(weights_.scale) || weights_.scale<=0 ||
        guard>std::numeric_limits<size_t>::max()-largest_cache ||
        memory_.capacity(weights_.latent_gamma)<narrow*2 ||
        memory_.capacity(weights_.cosine)<size_t(capacity)*128 ||
        memory_.capacity(weights_.sine)<size_t(capacity)*128)
        throw std::invalid_argument("invalid MLA configuration");
    constexpr int k[] = {hidden,hidden,narrow,hidden};
    constexpr int n[] = {qwidth,wide,expanded,hidden};
    for (size_t i=0;i<4;++i) {
        const auto &w=weights_.projections[i];
        if (!w || w->type()!=type || w->k()!=k[i] || w->n()!=n[i] || w->device()!=worker_.ordinal())
            throw std::invalid_argument("invalid MLA weight lease");
    }
    owned_.reserve(32);
    try {
        const auto low=[&](size_t count){return static_cast<uint16_t*>(allocate(count*2,Kind::workspace));};
        const auto fp=[&](size_t count){return static_cast<float*>(allocate(count*4,Kind::workspace));};
        const size_t s=max_chunk, scores=heads*s*capacity;
        t_.input=low(s*hidden); t_.compressed=low(s*narrow); t_.latent=low(s*narrow);
        t_.norm_round=low(s*narrow); t_.rope_in=low(17*s*64); t_.rope_out=low(17*s*64);
        t_.query=low(s*qwidth); t_.scaled=low(scores); t_.prob=low(scores);
        t_.av=low(s*hidden); t_.merged=low(s*hidden); t_.output=low(s*hidden);
        t_.key=static_cast<uint16_t*>(allocate(largest_cache+guard,Kind::cache));
        t_.value=static_cast<uint16_t*>(allocate(size_t(capacity)*hidden*2+guard,Kind::cache));
        t_.q_projection=fp(s*qwidth); t_.kv_a=fp(s*wide); t_.kv_b=fp(s*expanded);
        t_.norm_fp32=fp(s*narrow); t_.raw_qk=fp(scores); t_.masked=fp(scores);
        t_.maximum=fp(heads*s); t_.denominator=fp(heads*s); t_.prob_fp32=fp(scores);
        t_.av_fp32=fp(s*hidden); t_.output_fp32=fp(s*hidden);
        t_.positions=static_cast<uint32_t*>(allocate(s*4,Kind::workspace));
        for (auto *table:{weights_.latent_gamma,weights_.cosine,weights_.sine}) {
            memory_.pin(table);
        }
    } catch (...) {
        release();
        throw;
    }
}

void *MlaRequest::allocate(size_t bytes, Kind kind) {
    auto *ptr=memory_.allocate(bytes,kind);
    owned_.push_back(ptr);
    return ptr;
}

void MlaRequest::release() {
    while (!owned_.empty()) {
        memory_.release(owned_.back());
        owned_.pop_back();
    }
}

MlaRequest::~MlaRequest() {
    try {
        complete();
        release();
        for (auto *table:{weights_.latent_gamma,weights_.cosine,weights_.sine}) memory_.unpin(table);
    } catch (const std::exception &e) {
        // Continuing would let packed-weight leases expire while a queue may
        // still reference them. Report the unrecoverable device failure.
        std::fprintf(stderr,"MLA request teardown failed: %s\n",e.what());
        std::terminate();
    }
}

void MlaRequest::enqueue(int cursor,int count,const std::array<cnrtNotifier_t,5> *markers) {
    if (pending_) throw std::logic_error("MLA append already pending");
    if (cursor!=length_ || count<=0 || count>max_chunk_ || count>capacity_-length_)
        throw std::invalid_argument("invalid MLA cache append");
    const auto queue=worker_.queue(queue_);
    const int length=cursor+count;
    const auto gemm=bf16_?phase8_candidate_bf16:phase8_candidate_fp16;
    const auto norm=bf16_?phase82_native_bf16:phase82_native_fp16;
    const auto rope=bf16_?worker_rope_bf16:worker_rope_fp16;
    const auto prepare=bf16_?phase10_prepare_bf16:phase10_prepare_fp16;
    const auto append=bf16_?phase10_append_bf16:phase10_append_fp16;
    const auto cast=bf16_?phase10_cast_bf16:phase10_cast_fp16;
    const auto qk=bf16_?phase9_launch_qk_bf16:phase9_launch_qk_fp16;
    const auto softmax=bf16_?phase9_softmax_bf16:phase9_softmax_fp16;
    const auto av=bf16_?phase9_av_bf16:phase9_av_fp16;
    const auto project=[&](size_t i,const uint16_t *input,float *output) {
        const auto &w=*weights_.projections[i];
        check(gemm(queue,input,w.data(),output,count,w.n(),w.k()),"MLA projection");
    };

    const auto mark=[&](size_t i) {
        if (markers) check(cnrtPlaceNotifier((*markers)[i],queue),"MLA profiling marker");
    };
    pending_=true;
    next_=length_;
    mark(0);
    project(0,t_.input,t_.q_projection);
    project(1,t_.input,t_.kv_a);
    check(phase5_cast_kv(queue,t_.kv_a,t_.compressed,count,bf16_),"MLA compact latent");
    check(norm(queue,t_.compressed,weights_.latent_gamma,t_.latent,t_.norm_fp32,
               t_.norm_round,count,narrow,1e-6f),"MLA latent norm");
    project(2,t_.latent,t_.kv_b);
    check(prepare(queue,t_.q_projection,t_.kv_a,t_.rope_in,t_.positions,count,cursor),"MLA prepare rotary");
    check(rope(queue,t_.rope_in,weights_.cosine,weights_.sine,t_.positions,t_.rope_out,count),"MLA RoPE");
    mark(1);
    check(append(queue,t_.q_projection,t_.kv_b,t_.rope_out,t_.query,t_.key,t_.value,
                 count,cursor,capacity_),"MLA materialize and append");
    mark(2);
    check(qk(queue,t_.query,t_.key,t_.positions,0,weights_.scale,t_.raw_qk,t_.scaled,t_.masked,
             heads,count,length,capacity_),"MLA QK cache read");
    check(softmax(queue,t_.masked,nullptr,0,t_.maximum,t_.denominator,t_.prob_fp32,t_.prob,
                  heads*count,length,length),"MLA Softmax");
    check(av(queue,t_.prob,t_.value,t_.av_fp32,t_.av,heads,count,length,capacity_),"MLA AV cache read");
    mark(3);
    check(phase10_merge(queue,t_.av,t_.merged,count),"MLA merge heads");
    project(3,t_.merged,t_.output_fp32);
    check(cast(queue,t_.output_fp32,t_.output,count),"MLA final cast");
    mark(4);
    next_=length;
}

void MlaRequest::complete() {
    if (!pending_) return;
    worker_.sync(queue_);
    length_=next_;
    pending_=false;
}

void MlaRequest::reset() {
    if (pending_) throw std::logic_error("cannot reset an active MLA request");
    length_=0;
}
}
