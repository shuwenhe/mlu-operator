// Explicit experimental ATen reference paths. Only the isolated fixture links this file.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cnrt.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" cnrtRet_t phase82_finish_fp16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,const float*,int,int);
extern "C" cnrtRet_t phase82_finish_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,const float*,int,int);

#ifdef PHASE82_DEVICE_ONLY
extern "C" cnrtRet_t phase82_native_fp16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,int,int,float);
extern "C" cnrtRet_t phase82_native_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,int,int,float);
extern "C" cnrtRet_t __wrap_bang_rmsnorm_fp16(cnrtQueue_t q,const uint16_t*x,const uint16_t*g,
    uint16_t*y,float*n,uint16_t*r,int rows,int width,float eps) {
    return phase82_native_fp16(q,x,g,y,n,r,rows,width,eps);
}
extern "C" cnrtRet_t __wrap_bang_rmsnorm_bf16(cnrtQueue_t q,const uint16_t*x,const uint16_t*g,
    uint16_t*y,float*n,uint16_t*r,int rows,int width,float eps) {
    return phase82_native_bf16(q,x,g,y,n,r,rows,width,eps);
}
#else
namespace {
void check(cnrtRet_t status) {
    if (status != cnrtSuccess) throw std::runtime_error("Phase 8.2 CNRT " + std::to_string(status));
}
struct Buffer {
    void* data{};
    explicit Buffer(size_t bytes) {check(cnrtMalloc(&data,bytes));}
    ~Buffer() { if (data) cnrtFree(data); }
    void close() {check(cnrtFree(data));data=nullptr;}
};

cnrtRet_t norm(bool bf16, cnrtQueue_t queue, const uint16_t* x, const uint16_t* gamma,
              uint16_t* output, float* normalized, uint16_t* rounded, int rows, int width, float eps) {
    if (!queue || !x || !gamma || !output || rows<=0 || width<=0 || !(eps>0.f))
        return cnrtErrorArgsInvalid;
    // Explicit CPU reference scale and device-resident normalize/cast/gamma/cast.
    at::set_num_threads(2);
    std::vector<uint16_t> host(size_t(rows)*width);
    check(cnrtQueueSync(queue));
    check(cnrtMemcpy(host.data(), const_cast<uint16_t*>(x),host.size()*2,cnrtMemcpyDevToHost));
    auto h = at::from_blob(host.data(),{rows,width},bf16?at::kBFloat16:at::kHalf).to(at::kFloat);
    auto inverse = (h.pow(2).mean(-1)+eps).rsqrt().contiguous();
    Buffer buffer(size_t(rows)*sizeof(float));
    check(cnrtMemcpy(buffer.data,inverse.data_ptr<float>(),rows*sizeof(float),cnrtMemcpyHostToDev));
    auto finish=bf16?phase82_finish_bf16:phase82_finish_fp16;
    auto rc=finish(queue,x,gamma,output,normalized,rounded,static_cast<float*>(buffer.data),rows,width);
    check(cnrtQueueSync(queue));
    buffer.close();
    return rc;
}
}
#define WRAPPER(TYPE,BF) \
extern "C" cnrtRet_t __wrap_bang_rmsnorm_##TYPE(cnrtQueue_t q,const uint16_t*x,const uint16_t*g, \
    uint16_t*y,float*n,uint16_t*r,int rows,int width,float eps) { \
    return norm(BF,q,x,g,y,n,r,rows,width,eps); \
}
WRAPPER(fp16,false)
WRAPPER(bf16,true)
#endif

std::vector<float> checkpoint_attention(const std::vector<float>& q,const std::vector<float>& kv,
        const std::vector<uint16_t>& rotated,int pos,int m,bool bf16) {
    at::set_num_threads(2);
    const auto dtype=bf16?at::kBFloat16:at::kHalf;
    const int n=pos+m;
    auto qp=at::from_blob(const_cast<float*>(q.data()),{1,m,16,192},at::kFloat)
        .to(dtype).transpose(1,2).contiguous();
    auto cache=at::from_blob(const_cast<float*>(kv.data()),{1,n,16,256},at::kFloat).to(dtype).transpose(1,2);
    auto rp=at::from_blob(const_cast<uint16_t*>(rotated.data()),{17,n,64},dtype);
    qp.slice(-1,128).copy_(rp.slice(0,0,16).slice(1,pos,n).unsqueeze(0));
    auto kp=at::empty({1,16,n,192},dtype);
    kp.slice(-1,0,128).copy_(cache.slice(-1,0,128));
    kp.slice(-1,128).copy_(rp.select(0,16).expand({1,16,n,64}));
    auto allowed=at::arange(n).unsqueeze(0).le(at::arange(pos,n).unsqueeze(1));
    auto mask=at::zeros({1,1,m,n},dtype).masked_fill(allowed.logical_not(),-INFINITY);
    const double multiplier=1.0+0.1*0.707*std::log(40.0);
    const double scale=std::pow(192.,-0.5)*multiplier*multiplier;
    auto logits=at::matmul(qp,kp.transpose(2,3))*scale+mask;
    auto p=at::softmax(logits,-1,at::kFloat).to(dtype);
    auto a=at::matmul(p,cache.slice(-1,128)).transpose(1,2).contiguous().to(at::kFloat);
    const auto* data=a.data_ptr<float>();
    return {data,data+a.numel()};
}
