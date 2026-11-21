#include "packed_weight.h"
#include "rmsnorm_native.h"
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
extern "C" cnrtRet_t phase5_cast_kv(cnrtQueue_t,const float *,uint16_t *,int,bool);

static bool chain(cnrtQueue_t q,const std::string& dir,bool bf16) {
    constexpr int hidden=2048,latent=512,split=576,out_size=4096;
    auto load=[&](const char *name,int k,int n,std::vector<float>&ref) {
        std::ifstream f(dir+"/"+name+".bin",std::ios::binary);
        uint32_t dims[2];
        std::vector<uint16_t> b(size_t(k)*n),raw(b.size());
        if(!f.read(reinterpret_cast<char*>(dims),8) || dims[0]!=unsigned(n)||dims[1]!=unsigned(k) ||
           !f.read(reinterpret_cast<char*>(raw.data()),raw.size()*2)) std::exit(2);
        for(int j=0;j<n;++j) for(int p=0;p<k;++p) {
            const size_t i=size_t(p)*n+j;
            b[i]=bf16?raw[size_t(j)*k+p]:encode(decode(raw[size_t(j)*k+p],true),false);
            ref[i]=decode(b[i],bf16);
        }
        return b;
    };
    std::vector<float> wa(size_t(hidden)*split), wb(size_t(latent)*out_size);
    auto a=load("kv_a",hidden,split,wa),b=load("kv_b",latent,out_size,wb);
    std::vector<uint16_t> gamma(latent);
    std::ifstream gamma_file(dir+"/kv_gamma.bin",std::ios::binary);
    if(!gamma_file.read(reinterpret_cast<char*>(gamma.data()),gamma.size()*2)) return false;
    if(!bf16) for(auto& w:gamma) w=encode(decode(w,true),false);
    const auto type=bf16?mlu::WeightType::bf16:mlu::WeightType::fp16;
    mlu::PackedWeight *pa=nullptr,*pb=nullptr;
    CHECK(mlu::pack_rhs_host(a.data(),hidden,split,type,pa));
    CHECK(mlu::pack_rhs_host(b.data(),latent,out_size,type,pb));
    Buffer dg(gamma.size()*2); dg.upload(gamma.data(),gamma.size()*2);
    bool pass=true;
    for(int rows:{1,32}) {
        std::vector<uint16_t> x(size_t(rows)*hidden);
        std::vector<float> xf(x.size()),kv(size_t(rows)*split),norm(size_t(rows)*latent),
            output(size_t(rows)*out_size),ref(output.size());
        for(size_t i=0;i<x.size();++i) {
            xf[i]=(int(i%31)-15)/128.0f;
            x[i]=encode(xf[i],bf16);
        }
        mlu_gemm_reference(xf.data(),wa.data(),kv.data(),rows,split,hidden,hidden,split,split);
        for(int j=0;j<rows;++j) {
            float sum=0;
            for(int p=0;p<latent;++p) {
                const float v=decode(encode(kv[size_t(j)*split+p],bf16),bf16);
                norm[size_t(j)*latent+p]=v; sum+=v*v;
            }
            const float inv=1/std::sqrt(sum/latent+1e-6f);
            for(int p=0;p<latent;++p) {
                auto& v=norm[size_t(j)*latent+p];
                v=decode(encode(decode(encode(v*inv,bf16),bf16)*decode(gamma[p],bf16),bf16),bf16);
            }
        }
        mlu_gemm_reference(norm.data(),wb.data(),ref.data(),rows,out_size,latent,latent,out_size,out_size);
        Buffer dx(x.size()*2),dka(kv.size()*4),dl(norm.size()*2),dn(norm.size()*2),dy(output.size()*4);
        dx.upload(x.data(),x.size()*2);
        CHECK(mlu::gemm_prepacked(static_cast<uint16_t*>(dx.data),*pa,static_cast<float*>(dka.data),rows,q));
        CHECK(phase5_cast_kv(q,static_cast<float*>(dka.data),static_cast<uint16_t*>(dl.data),rows,bf16));
        const auto rms=bf16?bang_rmsnorm_bf16:bang_rmsnorm_fp16;
        CHECK(rms(q,static_cast<uint16_t*>(dl.data),static_cast<uint16_t*>(dg.data),
                  static_cast<uint16_t*>(dn.data),nullptr,nullptr,rows,latent,1e-6f));
        CHECK(mlu::gemm_prepacked(static_cast<uint16_t*>(dn.data),*pb,static_cast<float*>(dy.data),rows,q));
        CHECK(cnrtQueueSync(q));
        CHECK(cnrtMemcpy(output.data(),dy.data,output.size()*4,cnrtMemcpyDevToHost));
        double maximum=0,sum=0;
        size_t bad=0;
        for(size_t i=0;i<output.size();++i) {
            const double error=std::abs(double(output[i])-ref[i]);
            maximum=std::max(maximum,error);sum+=error;
            bad+=!std::isfinite(output[i])||error>1e-5+1e-5*std::abs(ref[i]);
        }
        pass&=!bad;
        std::printf("%s,%d,%zu,%.9g,%.9g,%zu\n",bf16?"bf16":"fp16",rows,output.size(),maximum,sum/output.size(),bad);
    }
    CHECK(mlu::release_packed_weight(pb));CHECK(mlu::release_packed_weight(pa));
    return pass;
}
int main(int argc,char **argv) {
    if(argc!=3) return 2;
    const std::string mode=argv[1];
    if(mode!="--authorized" && mode!="--authorized-chain") return 2;
    unsigned count=0; CHECK(cnrtGetDeviceCount(&count)); if(count!=1) return 2;
    CHECK(cnrtSetDevice(0));
    cnrtQueue_t q; CHECK(cnrtQueueCreate(&q));
    bool passed=true;
    if(mode=="--authorized-chain") {
        std::puts("dtype,m,elements,max_abs,mean_abs,mismatches");
        for(bool bf16:{true,false}) passed&=chain(q,argv[2],bf16);
    } else {
    std::puts("projection,backend,dtype,m,n,k,mismatches,nonfinite,max_abs,max_rel,mean_abs,warmups,samples,kernel_p50_us,kernel_p95_us,io_p50_us,io_p95_us,tflops");
    for(bool bf16:{true,false}) for(const char *p:{"q","kv_a","kv_b"}) passed&=test(q,argv[2],p,bf16);
    }
    CHECK(cnrtQueueDestroy(q));
    return !passed;
}
