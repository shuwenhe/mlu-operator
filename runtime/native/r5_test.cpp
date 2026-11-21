#include "dispatch.h"
#include <algorithm>
#include <cstdio>
#include <vector>

using namespace native_mlu;
static void require(bool ok) { if(!ok) throw std::runtime_error("R5 assertion"); }
int main(int argc,char**argv) {
    if(argc!=3 || std::string(argv[1])!="--authorized")return 2;
    Worker worker(argv[2]); Registry weights(worker); Memory mem(worker); Workspace ws(worker,mem);
    const auto q=worker.add_queue();
    Dispatch op(worker,mem,ws);
    const auto id=ws.add(q,2,4);
    auto make=[&](int rows,int cols,Type t){
        return View{mem.allocate(size_t(rows)*cols*(t==Type::fp32||t==Type::u32?4:2),Kind::workspace),rows,cols,cols,t};
    };
    auto a=make(1,1,Type::bf16), b=make(1,1,Type::bf16), c=make(1,1,Type::fp32);
    auto low=make(1,1,Type::bf16), gamma=make(1,1,Type::fp32);
    uint16_t one=0x3f80;float fp=1;
    for(auto v:{a,b}) check(cnrtMemcpy(v.data,&one,2,cnrtMemcpyHostToDev),"upload one");
    check(cnrtMemcpy(gamma.data,&fp,4,cnrtMemcpyHostToDev),"upload FP32 gamma");
    WeightKey key{"r5","one","synthetic","kn-row-major",worker.pci(),0,mlu::WeightType::bf16,1,1};
    weights.host(key,&one);
    for(auto backend:{Backend::Native,Backend::CNNL}) {
        ws.start(id);
        op.gemm(id,backend,a,backend==Backend::Native?weights.acquire(key):Registry::Lease{},
                backend==Backend::CNNL?b:View{},c);
        if(backend==Backend::Native) {
            weights.invalidate("r5");
            weights.collect();
            require(!weights.acquire(key)&&weights.bytes()>0);
        }
        auto *early=c.data;
        bool release_blocked=false;
        try { mem.release(early); } catch(const std::logic_error&) { release_blocked=true; }
        require(release_blocked && early==c.data);
        ws.finish(id);
        if(backend==Backend::Native) {weights.collect(); require(weights.bytes()==0);}
        check(cnrtMemcpy(&fp,c.data,4,cnrtMemcpyDevToHost),"read GEMM");
        require(fp==1);
        ws.start(id);
        op.rmsnorm(id,backend,a,backend==Backend::Native?b:gamma,backend==Backend::Native?low:c,1e-6f);
        ws.finish(id);
        if(backend==Backend::Native) {
            uint16_t bits=0;check(cnrtMemcpy(&bits,low.data,2,cnrtMemcpyDevToHost),"read native norm");
            require(bits==one);
        } else {
            check(cnrtMemcpy(&fp,c.data,4,cnrtMemcpyDevToHost),"read CNNL norm");
            require(fp>0.999f&&fp<=1.f);
        }
    }
    auto x=make(17,64,Type::bf16), cs=make(1,64,Type::bf16), sn=make(1,64,Type::bf16),
         pos=make(1,1,Type::u32), y=make(17,64,Type::bf16);
    std::vector<uint16_t> source(17*64,one), cos(64,one), sin(64);
    uint32_t slot=0;
    check(cnrtMemcpy(x.data,source.data(),source.size()*2,cnrtMemcpyHostToDev),"upload RoPE input");
    check(cnrtMemcpy(cs.data,cos.data(),128,cnrtMemcpyHostToDev),"upload cosine");
    check(cnrtMemcpy(sn.data,sin.data(),128,cnrtMemcpyHostToDev),"upload sine");
    check(cnrtMemcpy(pos.data,&slot,4,cnrtMemcpyHostToDev),"upload position");
    ws.start(id);op.rope(id,Backend::Native,x,cs,sn,pos,y,1);ws.finish(id);
    std::vector<uint16_t> got(source.size());
    check(cnrtMemcpy(got.data(),y.data,got.size()*2,cnrtMemcpyDevToHost),"read RoPE");
    require(got==source);
    ws.start(id);
    try {op.rope(id,Backend::CNNL,x,cs,sn,pos,y,1);require(false);}
    catch(const std::invalid_argument&) {}
    ws.finish(id);
    for(auto v:{a,b,c,low,gamma,x,cs,sn,pos,y}) mem.release(v.data);
    ws.close();weights.close();mem.close();worker.close();
    std::puts("PASS R5: native and CNNL GEMM/RMSNorm; native RoPE and unsupported route rejection");
}
