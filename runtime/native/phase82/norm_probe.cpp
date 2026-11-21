#include "rmsnorm_native.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cnrtRet_t status) {
    if (status != cnrtSuccess) throw std::runtime_error("CNRT " + std::to_string(status));
}
template<class T> static std::vector<T> read(const std::string& name, size_t n) {
    std::vector<T> result(n);
    std::ifstream in(name, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(result.data()), n*sizeof(T)) || in.peek()!=EOF)
        throw std::runtime_error(name);
    return result;
}
template<class T> static void write(const std::string& name,const std::vector<T>& data) {
    std::ofstream out(name, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T)))
        throw std::runtime_error(name);
}
template<class T> struct Buf {
    T *p{};
    explicit Buf(size_t n) { check(cnrtMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
    ~Buf(){ if(p) cnrtFree(p); }
    void upload(const std::vector<T>& v) {check(cnrtMemcpy(p,const_cast<T*>(v.data()),v.size()*sizeof(T),cnrtMemcpyHostToDev));}
    std::vector<T> fetch(size_t n) {
        std::vector<T> r(n);
        check(cnrtMemcpy(r.data(),p,r.size()*sizeof(T),cnrtMemcpyDevToHost));
        return r;
    }
};
extern "C" cnrtRet_t phase82_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,float*,float*,int,int,float);
extern "C" cnrtRet_t phase82_fp16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,float*,float*,int,int,float);
int main(int argc,char**argv) try {
    if(argc!=8 || std::string(argv[1])!="--authorized")return 2;
    int dev=-1;
    check(cnrtDeviceGetByPCIBusId(&dev,argv[2]));check(cnrtSetDevice(dev));
    std::string dtype=argv[3], prefix=argv[7]; int m=std::stoi(argv[4]),w=2048;
    if((dtype!="bf16"&&dtype!="fp16")||m<=0||m>128)return 2;
    size_t n=size_t(m)*w;
    auto input=read<uint16_t>(argv[5],n), gamma=read<uint16_t>(argv[6],w);
    Buf<uint16_t> x(n),g(w),y(n),rounded(n),baseline(n);
    Buf<float> norm(n),squares(n),stats(size_t(m)*4);
    x.upload(input);g.upload(gamma);
    cnrtQueue_t q=nullptr; check(cnrtQueueCreate(&q));
    auto probe=dtype=="bf16"?phase82_bf16:phase82_fp16;
    auto accepted=dtype=="bf16"?bang_rmsnorm_bf16:bang_rmsnorm_fp16;
    check(probe(q,x.p,g.p,y.p,norm.p,rounded.p,squares.p,stats.p,m,w,1e-6f));
    check(accepted(q,x.p,g.p,baseline.p,nullptr,nullptr,m,w,1e-6f));
    check(cnrtQueueSync(q));check(cnrtQueueDestroy(q));
    auto result=y.fetch(n);
    if(result!=baseline.fetch(n))throw std::runtime_error("traced norm differs from accepted device output");
    write(prefix+".out.u16",result); write(prefix+".normalized.f32",norm.fetch(n));
    write(prefix+".rounded.u16",rounded.fetch(n)); write(prefix+".squares.f32",squares.fetch(n));
    write(prefix+".stats.f32",stats.fetch(size_t(m)*4));
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
