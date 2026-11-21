#include "packed_weight.h"
#include "mlu_gemm_reference.h"
#include <cnnl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

static double micros(Clock::time_point start) {
    return std::chrono::duration<double,std::micro>(Clock::now()-start).count();
}
static float decode(uint16_t x,bool bf16) {
    float value;
    if(bf16) { uint32_t bits=uint32_t(x)<<16; std::memcpy(&value,&bits,4); }
    else { _Float16 low; std::memcpy(&low,&x,2); value=static_cast<float>(low); }
    return value;
}
static uint16_t encode(float value,bool bf16) {
    uint16_t bits;
    if(bf16) { uint32_t u; std::memcpy(&u,&value,4); bits=uint16_t((u+0x7fff+((u>>16)&1))>>16); }
    else { const _Float16 f=static_cast<_Float16>(value); std::memcpy(&bits,&f,2); }
    return bits;
}
static bool test(cnrtQueue_t q,const std::string &directory,const char *label,bool bf16) {
    std::ifstream in(directory+"/"+label+".bin",std::ios::binary);
    uint32_t dims[2];
    if(!in.read(reinterpret_cast<char*>(dims),sizeof(dims)) || !dims[0] || !dims[1] ||
       dims[0]>4096 || dims[1]>2048) return false;
    const int n=dims[0], k=dims[1];
    std::vector<uint16_t> raw(size_t(k)*n), b(raw.size());
    if(!in.read(reinterpret_cast<char*>(raw.data()),raw.size()*2) || in.peek()!=EOF) return false;
    std::vector<float> bf(b.size());
    for(int j=0;j<n;++j) for(int p=0;p<k;++p) {
        const auto source=raw[size_t(j)*k+p];
        const size_t offset=size_t(p)*n+j;
        b[offset]=bf16?source:encode(decode(source,true),false);
        bf[offset]=decode(b[offset],bf16);
    }
    Buffer logical(b.size()*2);
    const auto upload_start=Clock::now();
    logical.upload(b.data(),b.size()*2);
    const double weight_upload=micros(upload_start);
    mlu::PackedWeight *packed=nullptr;
    auto started=Clock::now();
    CHECK(mlu::pack_rhs_device(static_cast<uint16_t*>(logical.data),k,n,
        bf16?mlu::WeightType::bf16:mlu::WeightType::fp16,q,packed));
    CHECK(cnrtQueueSync(q));
    const double pack_us=micros(started);
    std::fprintf(stderr,"%s,%s,weight_h2d_us=%.3f,pack_ready_wall_us=%.3f,logical_bytes=%zu,packed_bytes=%zu\n",
        label,bf16?"bf16":"fp16",weight_upload,pack_us,b.size()*2,packed->bytes());
    bool passed=true;
    for(int m:{1,2,4,8,32,64,128}) {
        std::vector<uint16_t> a(size_t(m)*k);
        std::vector<float> af(a.size()),ref(size_t(m)*n),got(ref.size());
        uint32_t state=71;
        for(size_t i=0;i<a.size();++i) {
            state=1664525u*state+1013904223u;
            a[i]=encode((int(state>>27)-16)/128.0f,bf16);
            af[i]=decode(a[i],bf16);
        }
        mlu_gemm_reference(af.data(),bf.data(),ref.data(),m,n,k,k,n,n);
        Buffer da(a.size()*2),dc(ref.size()*4),dy(ref.size()*4);
        Library lib(q,bf16,m,n,k);
        Buffer workspace(lib.bytes);
        for(bool native:{true,false}) {
            auto launch=[&] {
                if(native) CHECK(mlu::gemm_prepacked(static_cast<uint16_t*>(da.data),*packed,
                                                     static_cast<float*>(dy.data),m,q));
                else lib.launch(da.data,logical.data,dc.data,dy.data,workspace.data);
            };
            da.upload(a.data(),a.size()*2);
            for(int i=0;i<warmups;++i) launch();
            CHECK(cnrtQueueSync(q));
            cnrtNotifier_t begin,end;
            CHECK(cnrtNotifierCreate(&begin)); CHECK(cnrtNotifierCreate(&end));
            std::vector<double> device,wall;
            for(int i=0;i<samples;++i) {
                CHECK(cnrtPlaceNotifier(begin,q)); launch(); CHECK(cnrtPlaceNotifier(end,q));
                CHECK(cnrtQueueSync(q));
                float us=0; CHECK(cnrtNotifierDuration(begin,end,&us)); device.push_back(us);
                started=Clock::now();
                da.upload(a.data(),a.size()*2); launch(); CHECK(cnrtQueueSync(q));
                CHECK(cnrtMemcpy(got.data(),dy.data,got.size()*4,cnrtMemcpyDevToHost));
                wall.push_back(micros(started));
            }
            CHECK(cnrtNotifierDestroy(end)); CHECK(cnrtNotifierDestroy(begin));
            double max_abs=0,max_rel=0,sum=0;
            size_t bad=0,nonfinite=0;
            for(size_t i=0;i<got.size();++i) {
                if(!std::isfinite(got[i])) { ++nonfinite; continue; }
                const double abs_ref=std::abs(double(ref[i])), e=std::abs(double(got[i])-ref[i]);
                max_abs=std::max(max_abs,e); max_rel=std::max(max_rel,e/std::max(abs_ref,1e-6));
                sum+=e; bad+=e>1e-5+1e-5*abs_ref;
            }
            passed&=!bad&&!nonfinite;
            std::sort(device.begin(),device.end()); std::sort(wall.begin(),wall.end());
            const double p50=(device[14]+device[15])/2;
            std::printf("%s,%s,%s,%d,%d,%d,%zu,%zu,%.9g,%.9g,%.9g,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                label,native?"bang":"cnnl",bf16?"bf16":"fp16",m,n,k,bad,nonfinite,max_abs,max_rel,
                sum/got.size(),warmups,samples,p50,device[28],(wall[14]+wall[15])/2,wall[28],2.*m*n*k/(p50*1e6));
        }
    }
    CHECK(mlu::release_packed_weight(packed));
    return passed;
}
int main(int argc,char **argv) {
    if(argc!=3 || std::string(argv[1])!="--authorized") return 2;
    unsigned count=0; CHECK(cnrtGetDeviceCount(&count)); if(count!=1) return 2;
    CHECK(cnrtSetDevice(0));
    cnrtQueue_t q; CHECK(cnrtQueueCreate(&q));
    std::puts("projection,backend,dtype,m,n,k,mismatches,nonfinite,max_abs,max_rel,mean_abs,warmups,samples,kernel_p50_us,kernel_p95_us,io_p50_us,io_p95_us,tflops");
    bool passed=true;
    for(bool bf16:{true,false}) for(const char *p:{"q","kv_a","kv_b"}) passed&=test(q,argv[2],p,bf16);
    CHECK(cnrtQueueDestroy(q));
    return !passed;
}
