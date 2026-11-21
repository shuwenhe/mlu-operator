#include "dispatch.h"
#include "rmsnorm_native.h"
#include <cnnl.h>
#include <cmath>
#include <limits>

extern "C" cnrtRet_t phase5_cast_kv(cnrtQueue_t,const float *,uint16_t *,int,bool);
extern "C" cnrtRet_t worker_rope_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,const uint16_t*,const uint32_t*,uint16_t*,int);
extern "C" cnrtRet_t worker_rope_fp16(cnrtQueue_t,const uint16_t*,const uint16_t*,const uint16_t*,const uint32_t*,uint16_t*,int);

namespace native_mlu {
namespace {
bool low(Type t){return t==Type::bf16 || t==Type::fp16;}
bool same(View a,View b){return a.rows==b.rows && a.cols==b.cols;}
void require(bool b,const char *msg){if(!b) throw std::invalid_argument(msg);}
void cnnl(cnnlStatus_t s,const char *op) {
    if(s!=CNNL_STATUS_SUCCESS) throw std::runtime_error(std::string(op)+": CNNL "+std::to_string(s));
}
void distinct(std::initializer_list<View> views) {
    for(auto i=views.begin();i!=views.end();++i)
        for(auto j=i+1;j!=views.end();++j) require(i->data!=j->data,"overlapping device buffers");
}
#define CNNL(call) cnnl((call),#call)
struct CnnlCall {
    cnnlHandle_t handle=nullptr;
    cnnlTensorDescriptor_t tensors[4]{};
    cnnlMatMulDescriptor_t matmul=nullptr;
    cnnlMatMulAlgo_t algo=nullptr;
    cnnlMatMulHeuristicResult_t heuristic=nullptr;
    cnnlNormDescriptor_t norm=nullptr;
    explicit CnnlCall(cnrtQueue_t q) {
        CNNL(cnnlCreate(&handle));
        auto rc=cnnlSetQueue(handle,q);
        if(rc!=CNNL_STATUS_SUCCESS){cnnlDestroy(handle);cnnl(rc,"bind CNNL queue");}
    }
    ~CnnlCall() {
        if(norm) cnnlDestroyNormDesc(norm);
        if(heuristic) cnnlDestroyMatMulHeuristicResult(heuristic);
        if(algo) cnnlDestroyMatMulAlgo(algo);
        if(matmul) cnnlDestroyMatMulDescriptor(matmul);
        for(auto t:tensors) if(t) cnnlDestroyTensorDescriptor(t);
        if(handle) cnnlDestroy(handle);
    }
    cnnlTensorDescriptor_t tensor(size_t i,View v,bool vector=false) {
        const int dims[]={vector?v.cols:v.rows,v.cols};
        const auto type=v.type==Type::bf16?CNNL_DTYPE_BFLOAT16:v.type==Type::fp16?CNNL_DTYPE_HALF:CNNL_DTYPE_FLOAT;
        CNNL(cnnlCreateTensorDescriptor(&tensors[i]));
        CNNL(cnnlSetTensorDescriptor(tensors[i],CNNL_LAYOUT_ARRAY,type,vector?1:2,dims));
        return tensors[i];
    }
};
struct Scratch {
    Memory &memory;
    void *p=nullptr;
    Scratch(Memory& m,size_t bytes):memory(m) {if(bytes) p=m.allocate(bytes,Kind::workspace);}
    ~Scratch() {try {memory.release(p);} catch(...) {}}
};
}
void Dispatch::view(View v) const {
    const size_t elem=low(v.type)?2:4;
    require(v.rows>0 && v.cols>0 && v.stride==v.cols &&
        size_t(v.rows)<=std::numeric_limits<size_t>::max()/elem/size_t(v.cols) &&
        memory_.capacity(v.data)>=size_t(v.rows)*v.cols*elem,"invalid shape, stride, or buffer ownership");
}
void Dispatch::active(size_t id) {
    require(slots_.slot(id).active,"inactive dispatch slot");
    worker_.queue(slots_.slot(id).queue);
}
void Dispatch::gemm(size_t id,Backend backend,View a,Registry::Lease weight,View b,View c) {
    active(id);view(a);view(c);
    require(backend==Backend::Native||backend==Backend::CNNL,"unregistered GEMM backend");
    require(low(a.type)&&c.type==Type::fp32&&a.rows==c.rows,"invalid GEMM dtype or rows");
    const auto queue_id=slots_.slot(id).queue;
    const auto q=worker_.queue(queue_id);
    if(backend==Backend::Native) {
        const auto type=a.type==Type::bf16?mlu::WeightType::bf16:mlu::WeightType::fp16;
        require(weight&&weight->device()==worker_.ordinal()&&weight->type()==type&&
                weight->k()==a.cols&&weight->n()==c.cols&&a.data!=c.data,"invalid native GEMM weight");
        slots_.retain(id,weight);
        for(auto v:{a,c}) slots_.retain_buffer(id,v.data);
        slots_.pending(id);
        check(mlu::gemm_prepacked(static_cast<uint16_t*>(a.data),*weight,static_cast<float*>(c.data),a.rows,q),
              "native GEMM submit");
        return;
    }
    view(b);distinct({a,b,c});
    require(!weight && b.type==a.type && b.rows==a.cols && b.cols==c.cols,"invalid CNNL GEMM weight");
    auto resource=std::make_shared<CnnlCall>(q);
    auto &lib=*resource;
    auto ad=lib.tensor(0,a),bd=lib.tensor(1,b),cd=lib.tensor(2,c);
    CNNL(cnnlCreateMatMulDescriptor(&lib.matmul));
    const auto compute=CNNL_DTYPE_FLOAT;const bool beta_enabled=false;
    CNNL(cnnlSetMatMulDescAttr(lib.matmul,CNNL_MATMUL_DESC_COMPUTE_TYPE,&compute,sizeof(compute)));
    CNNL(cnnlSetMatMulDescAttr(lib.matmul,CNNL_MATMUL_USE_BETA,&beta_enabled,sizeof(beta_enabled)));
    CNNL(cnnlCreateMatMulAlgo(&lib.algo));
    CNNL(cnnlCreateMatMulHeuristicResult(&lib.heuristic));
    int count=0;size_t bytes=0;
    CNNL(cnnlGetMatMulAlgoHeuristic(lib.handle,lib.matmul,ad,bd,cd,cd,nullptr,1,&lib.heuristic,&count));
    require(count==1,"CNNL returned no GEMM algorithm");
    CNNL(cnnlGetMatMulHeuristicResult(lib.heuristic,lib.algo,&bytes));
    auto work=std::make_shared<Scratch>(memory_,bytes);
    auto residual=std::make_shared<Scratch>(memory_,size_t(c.rows)*c.cols*4);
    for(auto v:{a,b,c}) slots_.retain_buffer(id,v.data);
    slots_.retain_resource(id,resource); slots_.retain_resource(id,work);
    slots_.retain_resource(id,residual);
    const float alpha=1,beta=0;
    slots_.pending(id);
    auto rc=cnnlMatMul_v2(lib.handle,lib.matmul,lib.algo,&alpha,ad,a.data,bd,b.data,
                          &beta,cd,residual->p,work->p,bytes,cd,c.data);
    worker_.sync(queue_id);
    cnnl(rc,"CNNL GEMM submit");
}
void Dispatch::rmsnorm(size_t id,Backend backend,View a,View gamma,View c,float eps) {
    active(id);view(a);view(gamma);view(c);distinct({a,gamma,c});
    require(backend==Backend::Native||backend==Backend::CNNL,"unregistered RMSNorm backend");
    require(low(a.type)&&same(a,c)&&gamma.rows==1&&gamma.cols==a.cols&&
            std::isfinite(eps)&&eps>0,"invalid RMSNorm parameters");
    const auto queue_id=slots_.slot(id).queue;
    const auto q=worker_.queue(queue_id);
    if(backend==Backend::Native) {
        require(c.type==a.type&&gamma.type==a.type,"native RMSNorm uses matching low dtype");
        const auto fn=a.type==Type::bf16?bang_rmsnorm_bf16:bang_rmsnorm_fp16;
        for(auto v:{a,gamma,c}) slots_.retain_buffer(id,v.data);
        slots_.pending(id);
        check(fn(q,static_cast<uint16_t*>(a.data),static_cast<uint16_t*>(gamma.data),
                 static_cast<uint16_t*>(c.data),nullptr,nullptr,a.rows,a.cols,eps),"native RMSNorm submit");
        return;
    }
    require(c.type==Type::fp32&&gamma.type==Type::fp32,"CNNL RMSNorm uses FP32 gamma and output");
    auto resource=std::make_shared<CnnlCall>(q);
    auto &lib=*resource;
    auto ad=lib.tensor(0,a), gd=lib.tensor(1,gamma,true), cd=lib.tensor(2,c);
    auto rd=lib.tensor(3,View{nullptr,1,a.rows,a.rows,Type::fp32},true);
    CNNL(cnnlCreateNormDesc(&lib.norm));
    size_t bytes=0;
    CNNL(cnnlGetRmsNormOpWorkspaceSize(lib.handle,1,ad,&bytes));
    auto work=std::make_shared<Scratch>(memory_,bytes);
    auto saved=std::make_shared<Scratch>(memory_,size_t(a.rows)*4);
    for(auto v:{a,gamma,c}) slots_.retain_buffer(id,v.data);
    slots_.retain_resource(id,resource); slots_.retain_resource(id,work);
    slots_.retain_resource(id,saved);
    slots_.pending(id);
    auto rc=cnnlRmsNormForward_v2(lib.handle,1,eps,lib.norm,ad,a.data,gd,gamma.data,nullptr,
                                  work->p,bytes,cd,c.data,rd,saved->p);
    worker_.sync(queue_id);
    cnnl(rc,"CNNL RMSNorm submit");
}
void Dispatch::rope(size_t id,Backend backend,View a,View cos,View sin,View pos,View c,int seq) {
    active(id);
    require(backend==Backend::Native,"CNNL RoPE is not registered");
    view(a);view(cos);view(sin);view(pos);view(c);distinct({a,cos,sin,pos,c});
    require(seq>0 && seq<=1024 && a.rows==17*seq && a.cols==64 && same(a,c) &&
            same(cos,sin) && cos.cols==64 && pos.rows==1 && pos.cols==seq && pos.type==Type::u32 &&
            low(a.type)&&cos.type==a.type&&sin.type==a.type&&c.type==a.type,"invalid checkpoint RoPE view");
    // Validate externally supplied positions before device indexing.
    worker_.sync(slots_.slot(id).queue);
    std::vector<uint32_t> slots(seq);
    check(cnrtMemcpy(slots.data(),pos.data,slots.size()*4,cnrtMemcpyDevToHost),"validate RoPE position table");
    for(auto slot:slots) require(slot<static_cast<unsigned>(cos.rows),"RoPE slot out of range");
    const auto fn=a.type==Type::bf16?worker_rope_bf16:worker_rope_fp16;
    for(auto v:{a,cos,sin,pos,c}) slots_.retain_buffer(id,v.data);
    slots_.pending(id);
    check(fn(worker_.queue(slots_.slot(id).queue),static_cast<uint16_t*>(a.data),
             static_cast<uint16_t*>(cos.data),static_cast<uint16_t*>(sin.data),
             static_cast<uint32_t*>(pos.data),static_cast<uint16_t*>(c.data),seq),"native RoPE submit");
}
void Dispatch::kv_chain(size_t id,View a,Registry::Lease wa,View gamma,Registry::Lease wb,
                         View wide,View latent,View norm,View output) {
    active(id);
    view(a);view(gamma);view(wide);view(latent);view(norm);view(output);
    distinct({a,gamma,wide,latent,norm,output});
    require(a.cols==2048&&wide.cols==576&&latent.cols==512&&same(latent,norm)&&
            latent.rows==a.rows&&wide.rows==a.rows&&output.rows==a.rows&&output.cols==4096&&
            gamma.rows==1&&gamma.cols==512&&gamma.type==a.type&&wide.type==Type::fp32&&
            output.type==Type::fp32&&latent.type==a.type&&norm.type==a.type&&low(a.type),
            "invalid deepseek KV chain");
    const auto dtype=a.type==Type::bf16?mlu::WeightType::bf16:mlu::WeightType::fp16;
    require(wa&&wb&&wa->device()==worker_.ordinal()&&wb->device()==worker_.ordinal()&&
            wa->type()==dtype&&wb->type()==dtype&&wa->k()==2048&&wa->n()==576&&
            wb->k()==512&&wb->n()==4096,"invalid deepseek KV weights");
    slots_.retain_buffer(id,latent.data);
    gemm(id,Backend::Native,a,std::move(wa),{},wide);
    check(phase5_cast_kv(worker_.queue(slots_.slot(id).queue),static_cast<float*>(wide.data),
                        static_cast<uint16_t*>(latent.data),a.rows,a.type==Type::bf16),"compact latent");
    rmsnorm(id,Backend::Native,latent,gamma,norm,1e-6f);
    gemm(id,Backend::Native,norm,std::move(wb),{},output);
}

/*
 * Phase 9: Native MLU Attention Kernels
 *
 * Query-Key multiplication with scale and causal masking.
 * Input: q [rows, seq, 192], k [rows, cache_len, 192] (BF16/FP16)
 * Output: qk [rows, seq, cache_len] FP32 raw
 *         scaled [rows, seq, cache_len] BF16/FP16 scaled
 *         masked [rows, seq, cache_len] FP32 with -INF at masked positions
 */
void Dispatch::attention_qk(size_t id,View q,View k,View positions,int cache_base,
                            float scale,View qk_out,View scaled_out,View masked_out) {
    active(id);view(q);view(k);view(positions);view(qk_out);view(scaled_out);view(masked_out);
    distinct({q,k,positions,qk_out,scaled_out,masked_out});
    require(q.cols==192&&k.cols==192&&q.rows==k.rows&&q.type==k.type&&low(q.type)&&
            positions.rows==1&&positions.type==Type::u32&&
            qk_out.rows==q.rows&&qk_out.cols==k.rows&&qk_out.type==Type::fp32&&
            scaled_out.rows==qk_out.rows&&scaled_out.cols==qk_out.cols&&scaled_out.type==q.type&&
            masked_out.rows==qk_out.rows&&masked_out.cols==qk_out.cols&&masked_out.type==Type::fp32,
            "invalid attention QK view");
    // Phase 9: Placeholder for BANG C kernel call
    // TODO: Invoke phase9_attention_qk_bf16/fp16 kernel
    // For now, this is a noop; actual implementation requires BANG C kernel
    require(false,"attention_qk: native MLU kernel not yet implemented");
}

/*
 * Row-wise softmax with FP32 precision.
 * Input: masked [rows, seq, cache_len] FP32 with -INF at masked positions
 * Output: maxes [rows, seq] FP32
 *         denom [rows, seq] FP32
 *         probs [rows, seq, cache_len] FP32 normalized
 *         probs_low [rows, seq, cache_len] BF16/FP16 converted
 */
void Dispatch::attention_softmax(size_t id,View masked,int cache_len,
                                 View maxes_out,View denom_out,View probs_out,View probs_low_out) {
    active(id);view(masked);view(maxes_out);view(denom_out);view(probs_out);view(probs_low_out);
    distinct({masked,maxes_out,denom_out,probs_out,probs_low_out});
    require(masked.type==Type::fp32&&
            maxes_out.rows==masked.rows&&maxes_out.cols==masked.rows&&maxes_out.type==Type::fp32&&
            denom_out.rows==masked.rows&&denom_out.cols==masked.rows&&denom_out.type==Type::fp32&&
            probs_out.rows==masked.rows&&probs_out.cols==masked.cols&&probs_out.type==Type::fp32&&
            probs_low_out.rows==masked.rows&&probs_low_out.cols==masked.cols&&low(probs_low_out.type),
            "invalid attention softmax view");
    // Phase 9: Placeholder for BANG C kernel call
    // TODO: Invoke phase9_attention_softmax_bf16/fp16 kernel
    require(false,"attention_softmax: native MLU kernel not yet implemented");
}

/*
 * Attention-weighted value summation.
 * Input: probs [rows, seq, cache_len] BF16/FP16
 *        v [rows, cache_len, 128] BF16/FP16
 * Output: av [rows, seq, 128] FP32
 *         out [rows, seq, 128] BF16/FP16
 */
void Dispatch::attention_av(size_t id,View probs,View v,View av_out,View out) {
    active(id);view(probs);view(v);view(av_out);view(out);
    distinct({probs,v,av_out,out});
    require(low(probs.type)&&low(v.type)&&probs.type==v.type&&
            v.cols==128&&probs.rows==v.rows&&
            av_out.rows==probs.rows&&av_out.cols==128&&av_out.type==Type::fp32&&
            out.rows==av_out.rows&&out.cols==av_out.cols&&out.type==probs.type,
            "invalid attention AV view");
    // Phase 9: Placeholder for BANG C kernel call
    // TODO: Invoke phase9_attention_av_bf16/fp16 kernel
    require(false,"attention_av: native MLU kernel not yet implemented");
}
}
}
