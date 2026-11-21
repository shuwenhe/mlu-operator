#include "matrix_gemm.h"
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

static bool run(cnrtQueue_t q,bool bf16,int m,int n,int k,int fixture,bool measure) {
    const size_t na=size_t(m)*k, nb=size_t(k)*n,nc=size_t(m)*n;
    std::vector<uint16_t> a(na),b(nb);
    std::vector<float> af(na),bf(nb),ref(nc),readback(nc+2*guard_words);
    fill(a,af,bf16,7,fixture==1?32.f:fixture==2?(1.f/4096):0.125f);
    fill(b,bf,bf16,43,fixture==1?(1.f/256):0.125f);
    if(fixture==1)
        for(int i=0;i<k/2;++i) for(int j=0;j<n;++j) {
            const size_t first=size_t(i)*n+j,last=size_t(k-1-i)*n+j;
            b[last]=b[first]^0x8000; bf[last]=-bf[first];
            for(int r=0;j==0 && r<m;++r) {
                a[size_t(r)*k+k-1-i]=a[size_t(r)*k+i]; af[size_t(r)*k+k-1-i]=af[size_t(r)*k+i];
            }
        }
    mlu_gemm_reference(af.data(),bf.data(),ref.data(),m,n,k,k,n,n);
    const auto pack_start=Clock::now();
    auto packed=pack(b,k,n);
    const double pack_us=std::chrono::duration<double,std::micro>(Clock::now()-pack_start).count();
    Library lib(q,bf16,m,n,k);
    Buffer da(na*2),db(nb*2),dp(packed.size()*2),dc(nc*4),work(lib.bytes),
           output(readback.size()*4);
    const auto launch=bf16?matrix_gemm_bf16:matrix_gemm_fp16;
    constexpr float sentinel=-123456.25f;
    bool pass=true;
    for(bool native:{true,false}) {
        const auto upload=[&] {
            da.upload(a.data(),na*2);
            if(native) dp.upload(packed.data(),packed.size()*2);
            else db.upload(b.data(),nb*2);
        };
        auto *dest=static_cast<float*>(output.data)+guard_words;
        const auto enqueue=[&] {
            if(native) CHECK(launch(q,static_cast<uint16_t*>(da.data),static_cast<uint16_t*>(dp.data),dest,m,n,k));
            else lib.launch(da.data,db.data,dc.data,dest,work.data);
        };
        std::fill(readback.begin(),readback.end(),sentinel);
        output.upload(readback.data(),readback.size()*4);
        upload();
        enqueue(); CHECK(cnrtQueueSync(q));
        std::vector<double> kernel,e2e;
        if(measure) {
            for(int i=0;i<warmups;++i) enqueue();
            CHECK(cnrtQueueSync(q));
            cnrtNotifier_t start,end;
            CHECK(cnrtNotifierCreate(&start)); CHECK(cnrtNotifierCreate(&end));
            for(int i=0;i<samples;++i) {
                CHECK(cnrtPlaceNotifier(start,q)); enqueue(); CHECK(cnrtPlaceNotifier(end,q));
                CHECK(cnrtQueueSync(q));
                float us=0; CHECK(cnrtNotifierDuration(start,end,&us)); kernel.push_back(us);
            }
            CHECK(cnrtNotifierDestroy(end)); CHECK(cnrtNotifierDestroy(start));
            for(int i=0;i<samples;++i) {
                const auto begin=Clock::now();
                upload(); enqueue(); CHECK(cnrtQueueSync(q));
                CHECK(cnrtMemcpy(readback.data()+guard_words,dest,nc*4,cnrtMemcpyDevToHost));
                e2e.push_back(std::chrono::duration<double,std::micro>(Clock::now()-begin).count());
            }
        }
        CHECK(cnrtMemcpy(readback.data(),output.data,readback.size()*4,cnrtMemcpyDevToHost));
        size_t guards=0,bad=0,nonfinite=0;
        double ma=0,mr=0,mean=0;
        for(size_t i=0;i<guard_words;++i)
            guards+=(readback[i]!=sentinel)+(readback[guard_words+nc+i]!=sentinel);
        for(size_t i=0;i<nc;++i) {
            const float got=readback[i+guard_words];
            if(!std::isfinite(got)) { ++nonfinite; continue; }
            const double e=std::abs(double(ref[i])-got), scale=std::abs(double(ref[i]));
            ma=std::max(ma,e); mr=std::max(mr,e/std::max(scale,1e-6)); mean+=e;
            bad+=e>1e-5+1e-5*scale;
        }
        pass&=!(bad||nonfinite||guards);
        const auto median=[](std::vector<double>&v) {
            if(v.empty()) return 0.;
            std::sort(v.begin(),v.end()); return (v[(v.size()-1)/2]+v[v.size()/2])/2;
        };
        const double med=median(kernel),emed=median(e2e),flops=2.*m*n*k;
        std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%zu,%zu,%zu,%.6g,%.6g,%.6g,%.6g,%.8g,%.6g\n",
            native?"bang":"cnnl",bf16?"bf16":"fp16",m,n,k,fixture,measure?warmups:0,
            measure?samples:0,ma,mr,mean/nc,bad,nonfinite,guards,med,
            measure?kernel[28]:0,emed,measure?e2e[28]:0,med>0?flops/(med*1e6):0,native?pack_us:0);
    }
    return pass;
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    const std::string mode=argv[1];
    if(mode!="--authorized-check" && mode!="--authorized-benchmark") return 2;
    unsigned count=0; CHECK(cnrtGetDeviceCount(&count)); if(count!=1) return 2;
    CHECK(cnrtSetDevice(0));
    cnrtQueue_t queue; CHECK(cnrtQueueCreate(&queue));
    std::puts("backend,dtype,m,n,k,fixture,warmups,samples,max_abs,max_rel,mean_abs,mismatches,nonfinite,guard_errors,kernel_p50_us,kernel_p95_us,e2e_p50_us,e2e_p95_us,tflops,host_pack_us");
    const int shapes[][3]={{127,193,65},{32,64,128},{128,256,512},{256,128,512},{1024,1024,1024}};
    bool pass=true;
    for(bool bf16:{false,true}) {
        for(const auto &s:shapes) pass&=run(queue,bf16,s[0],s[1],s[2],0,mode=="--authorized-benchmark");
        if(mode=="--authorized-check") {
            pass&=run(queue,bf16,1,1,1,2,false);
            pass&=run(queue,bf16,17,129,257,1,false);
        }
    }
    CHECK(cnrtQueueDestroy(queue));
    return !pass;
}
