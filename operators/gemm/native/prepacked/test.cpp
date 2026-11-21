#include "matrix_gemm.h"
#include "packed_weight.h"
#include "mlu_gemm_reference.h"
#include <cnnl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call) do { auto s=(call); if(s!=0) { \
    std::fprintf(stderr,"%s status=%u\n",#call,unsigned(s)); std::exit(1); } } while(0)
using Clock = std::chrono::steady_clock;
constexpr int warmups=5, samples=30;
constexpr size_t guard_words=32;

struct Buffer {
    void *data=nullptr;
    explicit Buffer(size_t n) { if(n) CHECK(cnrtMalloc(&data,n)); }
    ~Buffer() { if(data) CHECK(cnrtFree(data)); }
    void upload(void *src,size_t n) { CHECK(cnrtMemcpy(data,src,n,cnrtMemcpyHostToDev)); }
};

struct Library {
    cnnlHandle_t h;
    cnnlTensorDescriptor_t a,b,c;
    cnnlMatMulDescriptor_t d;
    cnnlMatMulAlgo_t algo;
    size_t bytes;
    float alpha=1,beta=0;
    Library(cnrtQueue_t q,bool bf16,int m,int n,int k) {
        CHECK(cnnlCreate(&h)); CHECK(cnnlSetQueue(h,q));
        CHECK(cnnlCreateTensorDescriptor(&a)); CHECK(cnnlCreateTensorDescriptor(&b));
        CHECK(cnnlCreateTensorDescriptor(&c));
        const int ad[]={m,k},bd[]={k,n},cd[]={m,n};
        const auto low=bf16?CNNL_DTYPE_BFLOAT16:CNNL_DTYPE_HALF;
        const auto high=CNNL_DTYPE_FLOAT;
        CHECK(cnnlSetTensorDescriptor(a,CNNL_LAYOUT_ARRAY,low,2,ad));
        CHECK(cnnlSetTensorDescriptor(b,CNNL_LAYOUT_ARRAY,low,2,bd));
        CHECK(cnnlSetTensorDescriptor(c,CNNL_LAYOUT_ARRAY,high,2,cd));
        CHECK(cnnlCreateMatMulDescriptor(&d));
        const bool use_beta=false;
        CHECK(cnnlSetMatMulDescAttr(d,CNNL_MATMUL_DESC_COMPUTE_TYPE,&high,sizeof(high)));
        CHECK(cnnlSetMatMulDescAttr(d,CNNL_MATMUL_USE_BETA,&use_beta,sizeof(use_beta)));
        CHECK(cnnlCreateMatMulAlgo(&algo));
        cnnlMatMulHeuristicResult_t heuristic;
        CHECK(cnnlCreateMatMulHeuristicResult(&heuristic));
        int count=0;
        CHECK(cnnlGetMatMulAlgoHeuristic(h,d,a,b,c,c,nullptr,1,&heuristic,&count));
        if(count!=1) { std::fputs("no CNNL algorithm\n",stderr); std::exit(1); }
        CHECK(cnnlGetMatMulHeuristicResult(heuristic,algo,&bytes));
        CHECK(cnnlDestroyMatMulHeuristicResult(heuristic));
    }
    ~Library() {
        CHECK(cnnlDestroyMatMulAlgo(algo)); CHECK(cnnlDestroyMatMulDescriptor(d));
        CHECK(cnnlDestroyTensorDescriptor(c)); CHECK(cnnlDestroyTensorDescriptor(b));
        CHECK(cnnlDestroyTensorDescriptor(a)); CHECK(cnnlDestroy(h));
    }
    void launch(void *x,void *y,void *z,void *out,void *work) {
        CHECK(cnnlMatMul_v2(h,d,algo,&alpha,a,x,b,y,&beta,c,z,work,bytes,c,out));
    }
};

static void fill(std::vector<uint16_t>&bits,std::vector<float>&values,bool bf16,
                 uint32_t seed,float scale) {
    for(size_t i=0;i<bits.size();++i) {
        seed=1664525u*seed+1013904223u;
        const float x=(int(seed>>28)-8)*scale;
        values[i]=x;
        if(bf16) { uint32_t u; std::memcpy(&u,&x,4); bits[i]=uint16_t(u>>16); }
        else { _Float16 f=static_cast<_Float16>(x); std::memcpy(&bits[i],&f,2); }
    }
}

static std::vector<uint16_t> pack(const std::vector<uint16_t>&b,int k,int n) {
    const int kt=(k+matrix_k-1)/matrix_k, nt=(n+matrix_n-1)/matrix_n;
    std::vector<uint16_t> packed(size_t(kt)*nt*matrix_k*matrix_n);
    for(int p=0;p<k;++p) for(int j=0;j<n;++j) {
        const size_t tile=size_t(j/matrix_n)*kt+p/matrix_k;
        packed[tile*matrix_k*matrix_n+matrix_rhs_slot(p%matrix_k,j%matrix_n)]=b[size_t(p)*n+j];
    }
    return packed;
}

// Linker injection exercises the error path without consuming device memory.
static bool fail_next_allocation = false;
extern "C" cnrtRet_t __real_cnrtMalloc(void **, size_t);
extern "C" cnrtRet_t __wrap_cnrtMalloc(void **p, size_t bytes) {
    if (fail_next_allocation) {
        fail_next_allocation = false;
        return cnrtErrorNoMem;
    }
    return __real_cnrtMalloc(p, bytes);
}

using mlu::PackedWeight;
using mlu::WeightType;
enum class Path { fresh, host, device, library };
static double us(Clock::time_point start) {
    return std::chrono::duration<double,std::micro>(Clock::now()-start).count();
}
struct Events {
    cnrtNotifier_t begin,end;
    Events() { CHECK(cnrtNotifierCreate(&begin)); CHECK(cnrtNotifierCreate(&end)); }
    ~Events() { CHECK(cnrtNotifierDestroy(end)); CHECK(cnrtNotifierDestroy(begin)); }
    double duration() const { float time=0; CHECK(cnrtNotifierDuration(begin,end,&time)); return time; }
};
struct Prep { double total=0, host_pack=0, device_pack=0, upload=0; };

static Prep prepare(Path path, std::vector<uint16_t>&b, int k, int n, WeightType type,
                    cnrtQueue_t q, Buffer &raw, Events &event, PackedWeight *&weight) {
    Prep prep;
    const auto begin=Clock::now();
    if(path==Path::device) {
        raw.upload(b.data(),b.size()*2);
        prep.upload=us(begin);
        CHECK(mlu::pack_rhs_device(static_cast<uint16_t*>(raw.data),k,n,type,q,weight,event.begin,event.end));
        CHECK(cnrtQueueSync(q));
        prep.device_pack=event.duration();
    } else if(path==Path::library) {
        raw.upload(b.data(),b.size()*2);
        prep.upload=us(begin);
    } else {
        mlu::HostPackTiming timing;
        CHECK(mlu::pack_rhs_host(b.data(),k,n,type,weight,&timing));
        prep.host_pack=timing.pack_us;
        prep.upload=timing.upload_us;
    }
    prep.total=us(begin);
    return prep;
}

static bool fixture(cnrtQueue_t q,cnrtQueue_t alternate,bool bf16,int m,int n,int k,bool two_cards) {
    const auto type=bf16?WeightType::bf16:WeightType::fp16;
    std::vector<uint16_t> a(size_t(m)*k),b(size_t(k)*n);
    std::vector<float> af(a.size()),bf(b.size()),expected(size_t(m)*n);
    Buffer da(a.size()*2),raw(b.size()*2),out((expected.size()+2*guard_words)*4),
           second((expected.size()+2*guard_words)*4);
    Events event;
    constexpr float sentinel=-123456.25f;
    std::vector<float> got(expected.size()+2*guard_words,sentinel);
    PackedWeight *host=nullptr,*device=nullptr;
    fill(b,bf,bf16,43,0.125f);
    prepare(Path::host,b,k,n,type,q,raw,event,host);
    prepare(Path::device,b,k,n,type,q,raw,event,device);
    const auto oracle=pack(b,k,n);
    size_t pack_bad=0;
    for(auto *w:{host,device}) {
        std::vector<uint16_t> bytes(w->bytes()/2);
        CHECK(cnrtMemcpy(bytes.data(),const_cast<uint16_t*>(w->data()),w->bytes(),cnrtMemcpyDevToHost));
        for(size_t i=0;i<bytes.size();++i) pack_bad+=bytes[i]!=oracle[i];
    }
    bool pass=!pack_bad;
    PackedWeight *invalid=nullptr;
    fail_next_allocation=true;
    const auto no_memory=mlu::pack_rhs_host(b.data(),k,n,type,invalid);
    pass&=no_memory==cnrtErrorNoMem && !invalid && !fail_next_allocation;
    std::fprintf(stderr,"allocation_failure_injection status=%u handle_null=%d\n",
                 unsigned(no_memory),!invalid);
    if(two_cards) {
        CHECK(cnrtSetDevice(1));
        const auto wrong=mlu::release_packed_weight(host);
        pass&=wrong==cnrtErrorDeviceInvalid && host;
        CHECK(cnrtSetDevice(0));
        std::fprintf(stderr,"cross_device_release status=%u handle_preserved=%d\n",
                     unsigned(wrong),host!=nullptr);
    }
    for(int i=0;i<8;++i) {
        const uint16_t word=uint16_t(i+1);
        CHECK(mlu::pack_rhs_host(&word,1,1,type,invalid));
        uint16_t got_word=0;
        CHECK(cnrtMemcpy(&got_word,const_cast<uint16_t*>(invalid->data()),2,cnrtMemcpyDevToHost));
        pass&=got_word==word;
        CHECK(mlu::release_packed_weight(invalid));
    }
    pass&=mlu::pack_rhs_host(b.data(),0,n,type,invalid)==cnrtErrorArgsInvalid && !invalid;
    pass&=mlu::pack_rhs_host(b.data(),k,n,static_cast<WeightType>(99),invalid)==cnrtErrorArgsInvalid && !invalid;
    pass&=mlu::pack_rhs_device(static_cast<uint16_t*>(raw.data),k,n,type,nullptr,invalid)==cnrtErrorArgsInvalid && !invalid;
    pass&=mlu::pack_rhs_host(b.data(),k,n,type,host)==cnrtErrorArgsInvalid && host;
    auto *dest=static_cast<float*>(out.data)+guard_words;
    pass&=mlu::gemm_prepacked(static_cast<uint16_t*>(da.data),*host,dest,0,q)==cnrtErrorArgsInvalid;
    pass&=mlu::gemm_prepacked(reinterpret_cast<uint16_t*>(static_cast<char*>(da.data)+1),*host,dest,m,q)==cnrtErrorArgsInvalid;
    pass&=mlu::gemm_prepacked(static_cast<uint16_t*>(da.data),*host,dest,m,nullptr)==cnrtErrorArgsInvalid;
    double max_abs=0, max_rel=0, sum_abs=0;
    size_t mismatches=0,nonfinite=0,guards=0,checked=0;
    constexpr float tolerance=1e-5f;
    for(int reuse:{1,10,100}) {
        for(int i=0;i<reuse;++i) {
            fill(a,af,bf16,7+i,0.125f);
            da.upload(a.data(),a.size()*2);
            mlu_gemm_reference(af.data(),bf.data(),expected.data(),m,n,k,k,n,n);
            std::fill(got.begin(),got.end(),sentinel);
            out.upload(got.data(),got.size()*4); second.upload(got.data(),got.size()*4);
            // Submit both weights to independent queues before either synchronize.
            CHECK(mlu::gemm_prepacked(static_cast<uint16_t*>(da.data),*host,dest,m,q));
            CHECK(mlu::gemm_prepacked(static_cast<uint16_t*>(da.data),*(i%2 ? host : device),
                                     static_cast<float*>(second.data)+guard_words,m,alternate));
            CHECK(cnrtQueueSync(q)); CHECK(cnrtQueueSync(alternate));
            for(auto *buffer:{&out,&second}) {
                CHECK(cnrtMemcpy(got.data(),buffer->data,got.size()*4,cnrtMemcpyDevToHost));
                for(size_t j=0;j<guard_words;++j)
                    guards+=(got[j]!=sentinel)+(got[guard_words+expected.size()+j]!=sentinel);
                for(size_t j=0;j<expected.size();++j) {
                    const float value=got[guard_words+j];
                    if(!std::isfinite(value)) { ++nonfinite; continue; }
                    const double error=std::abs(double(expected[j])-value);
                    const double scale=std::abs(double(expected[j]));
                    max_abs=std::max(max_abs,error); max_rel=std::max(max_rel,error/std::max(scale,1e-6));
                    sum_abs+=error; mismatches+=error>tolerance+tolerance*scale; ++checked;
                }
            }
        }
    }
    CHECK(mlu::release_packed_weight(host)); CHECK(mlu::release_packed_weight(device));
    CHECK(mlu::release_packed_weight(host));
    pass&=!host && !device && !mismatches && !nonfinite && !guards;
    std::printf("%s,%d,%d,%d,%zu,%zu,%.9g,%.9g,%.9g,%zu,%zu,%zu,%s\n",
                bf16?"bf16":"fp16",m,n,k,oracle.size(),pack_bad,max_abs,max_rel,
                checked?sum_abs/checked:0,mismatches,nonfinite,guards,pass?"PASS":"FAIL");
    return pass;
}

static double quantile(std::vector<double> values,double p) {
    std::sort(values.begin(),values.end());
    if(p==0.5) return (values[(values.size()-1)/2]+values[values.size()/2])/2;
    return values[size_t(std::ceil(p*values.size()))-1];
}
static bool benchmark(cnrtQueue_t q,bool bf16,int m,int n,int k,bool device_only) {
    const auto type=bf16?WeightType::bf16:WeightType::fp16;
    std::vector<uint16_t> a(size_t(m)*k),b(size_t(k)*n);
    std::vector<float> af(a.size()),bf(b.size()),ref(size_t(m)*n),got(ref.size());
    fill(a,af,bf16,7,0.125f); fill(b,bf,bf16,43,0.125f);
    mlu_gemm_reference(af.data(),bf.data(),ref.data(),m,n,k,k,n,n);
    Library lib(q,bf16,m,n,k);
    Buffer da(a.size()*2),raw(b.size()*2),dc(got.size()*4),dst(got.size()*4),work(lib.bytes);
    Events pack_event,kernel_event;
    constexpr int trials=9;
    bool passed=true;
    for(auto path:{Path::fresh,Path::host,Path::device,Path::library}) {
        if(device_only && path!=Path::device) continue;
        const auto label=path==Path::fresh?"host_every_call":path==Path::host?"host_resident":
                         path==Path::device?"device_resident":"cnnl_resident";
        PackedWeight *weight=nullptr;
        const auto enqueue=[&] {
            if(path==Path::library) lib.launch(da.data,raw.data,dc.data,dst.data,work.data);
            else CHECK(mlu::gemm_prepacked(static_cast<uint16_t*>(da.data),*weight,static_cast<float*>(dst.data),m,q));
        };
        prepare(path,b,k,n,type,q,raw,pack_event,weight);
        const size_t resident_bytes=weight?weight->bytes():b.size()*2;
        da.upload(a.data(),a.size()*2);
        for(int i=0;i<warmups;++i) enqueue();
        CHECK(cnrtQueueSync(q));
        std::vector<double> kernel;
        for(int i=0;i<samples;++i) {
            CHECK(cnrtPlaceNotifier(kernel_event.begin,q)); enqueue(); CHECK(cnrtPlaceNotifier(kernel_event.end,q));
            CHECK(cnrtQueueSync(q)); kernel.push_back(kernel_event.duration());
        }
        CHECK(mlu::release_packed_weight(weight));
        const double kernel_p50=quantile(kernel,0.5),kernel_p95=quantile(kernel,0.95);
        for(int reuse:{1,10,100}) {
            std::vector<double> totals,invocations,preps,hp,dp,h2d;
            for(int t=0;t<trials;++t) {
                Prep prep;
                const auto start=Clock::now();
                if(path!=Path::fresh) prep=prepare(path,b,k,n,type,q,raw,pack_event,weight);
                double invocation=0;
                for(int i=0;i<reuse;++i) {
                    if(path==Path::fresh) {
                        const auto one=prepare(path,b,k,n,type,q,raw,pack_event,weight);
                        prep.total+=one.total; prep.host_pack+=one.host_pack; prep.upload+=one.upload;
                    }
                    const auto invoke=Clock::now();
                    da.upload(a.data(),a.size()*2);
                    enqueue(); CHECK(cnrtQueueSync(q));
                    CHECK(cnrtMemcpy(got.data(),dst.data,got.size()*4,cnrtMemcpyDevToHost));
                    invocation+=us(invoke);
                    if(path==Path::fresh) CHECK(mlu::release_packed_weight(weight));
                }
                CHECK(mlu::release_packed_weight(weight));
                totals.push_back(us(start)/reuse); invocations.push_back(invocation/reuse);
                preps.push_back(prep.total); hp.push_back(prep.host_pack);
                dp.push_back(prep.device_pack); h2d.push_back(prep.upload);
                for(size_t i=0;i<got.size();++i)
                    passed&=std::isfinite(got[i]) && std::abs(got[i]-ref[i])<=1e-5+1e-5*std::abs(ref[i]);
            }
            std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                label,bf16?"bf16":"fp16",m,n,k,reuse,warmups,samples,trials,b.size()*2,resident_bytes,
                quantile(hp,0.5),quantile(dp,0.5),quantile(h2d,0.5),quantile(preps,0.5),
                kernel_p50,kernel_p95,quantile(invocations,0.5),quantile(invocations,0.95),
                quantile(totals,0.5),quantile(totals,0.95),2.*m*n*k/(kernel_p50*1e6));
        }
    }
    return passed;
}

int main(int argc,char **argv) {
    if(argc!=2) return 2;
    const std::string mode=argv[1];
    const bool device_only=mode=="--authorized-device-pack";
    if(mode!="--authorized-check" && mode!="--authorized-benchmark" && !device_only) return 2;
    unsigned count=0; CHECK(cnrtGetDeviceCount(&count)); if(count!=1 && !(count==2 && mode=="--authorized-check")) return 2;
    CHECK(cnrtSetDevice(0));
    cnrtQueue_t queue,alternate; CHECK(cnrtQueueCreate(&queue)); CHECK(cnrtQueueCreate(&alternate));
    bool pass=true;
    if(mode=="--authorized-check") {
        std::puts("dtype,m,n,k,packed_words_per_path,packing_mismatches,max_abs,max_rel,mean_abs,mismatches,nonfinite,guard_errors,status");
        for(bool bf16:{false,true}) {
            pass&=fixture(queue,alternate,bf16,3,193,65,count==2);
            pass&=fixture(queue,alternate,bf16,1,129,257,count==2);
        }
    } else {
        std::puts("path,dtype,m,n,k,reuse,warmups,kernel_samples,trials,logical_rhs_bytes,resident_rhs_bytes,host_pack_p50_us,device_pack_p50_us,rhs_h2d_p50_us,prep_total_p50_us,kernel_p50_us,kernel_p95_us,invocation_p50_us,invocation_p95_us,amortized_p50_us,amortized_p95_us,tflops");
        for(bool bf16:{false,true}) {
            pass&=benchmark(queue,bf16,3,193,65,device_only);
            pass&=benchmark(queue,bf16,1,3072,2048,device_only);
            pass&=benchmark(queue,bf16,128,3072,2048,device_only);
        }
    }
    CHECK(cnrtQueueDestroy(alternate)); CHECK(cnrtQueueDestroy(queue));
    return !pass;
}
