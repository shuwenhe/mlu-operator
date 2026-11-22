#include "../workspace.h"
#include "../phase9/softmax/softmax.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

extern "C" {
cnrtRet_t phase8_candidate_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,float*,int,int,int);
cnrtRet_t phase10_cast_bf16(cnrtQueue_t,const float*,uint16_t*,int);
cnrtRet_t phase11_swiglu(cnrtQueue_t,const float*,const float*,uint16_t*,int);
cnrtRet_t phase11_route(cnrtQueue_t,const float*,uint32_t*,float*,int);
cnrtRet_t phase11_expert_combine(cnrtQueue_t,const uint16_t*,const float*,const uint16_t*,uint16_t*,int);
}

using namespace native_mlu;
namespace {
constexpr int hidden=2048, n_experts=64, k_routes=6, shared_width=2816;
int layer=1;
struct Matrix { std::string file; uint64_t offset; int n,k; };

void require(bool pass,const char *message) {
    if (!pass) throw std::runtime_error(message);
}
template<typename T> std::vector<T> read(const std::string &filename,size_t count) {
    std::vector<T> result(count);
    std::ifstream in(filename,std::ios::binary);
    require(bool(in.read(reinterpret_cast<char*>(result.data()),count*sizeof(T)))&&in.peek()==EOF,
            "invalid MoE fixture");
    return result;
}
template<typename T> auto fetch(T *device,size_t count) {
    std::vector<T> result(count);
    check(cnrtMemcpy(result.data(),device,count*sizeof(T),cnrtMemcpyDevToHost),"read MoE stage");
    return result;
}
double value(uint16_t word) {
    const uint32_t bits=uint32_t(word)<<16;
    float f;
    std::memcpy(&f,&bits,sizeof(f));
    return f;
}
double value(float f) {return f;}
double value(uint32_t i) {return i;}

template<typename T> bool compare(const char *name,int rows,const std::vector<T> &actual,
                                  const std::vector<T> &reference) {
    size_t mismatches=0,nonfinite=0,bit=0;
    double max_abs=0,max_rel=0;
    for (size_t j=0;j<reference.size();++j) {
        const double a=value(actual[j]),b=value(reference[j]);
        const double error=std::abs(a-b);
        const bool invalid=!std::isfinite(a)||!std::isfinite(b);
        const bool fail=invalid||error>1e-5+1e-5*std::abs(b);
        mismatches+=fail;
        nonfinite+=invalid;
        bit+=actual[j]!=reference[j];
        max_abs=std::max(max_abs,error);
        max_rel=std::max(max_rel,error/std::max(1e-30,std::abs(b)));
    }
    std::cout<<"layer="<<layer<<" stage="<<name<<" rows="<<rows<<" elements="<<reference.size()
             <<" mismatches="<<mismatches<<" unequal="<<bit<<" nonfinite="<<nonfinite
             <<" max_abs="<<max_abs<<" max_rel="<<max_rel<<'\n';
    return mismatches==0;
}
}

int main(int argc,char **argv) try {
    if (argc<4||argc>6) return 2;
    if (argc>=5) layer=std::stoi(argv[4]);
    if (layer<1||layer>=27) return 2;
    Worker worker(argv[1]);
    Memory memory(worker);
    Registry registry(worker);
    const auto qid=worker.add_queue();
    const auto queue=worker.queue(qid);
    const std::string checkpoint=argv[2], fixture=argv[3];
    std::map<std::string,Matrix> metadata;
    std::ifstream manifest(fixture+"/weights.txt");
    for (std::string name;manifest>>name;) {
        auto &item=metadata[name];
        require(bool(manifest>>item.file>>item.offset>>item.n>>item.k)&&item.n>0&&item.k>0,"invalid MoE matrix");
    }
    require(metadata.size()==196,"wrong MoE tensor count");
    std::map<std::string,Registry::Lease> weights;
    const auto packed=[&](const std::string &name)->const mlu::PackedWeight& {
        auto known=weights.find(name);
        if (known!=weights.end()) return *known->second;
        const auto &item=metadata.at(name);
        std::ifstream shard(checkpoint+'/'+item.file,std::ios::binary);
        require(bool(shard.seekg(item.offset)),"bad MoE tensor offset");
        std::vector<uint16_t> source(size_t(item.k)*item.n), matrix(source.size());
        require(bool(shard.read(reinterpret_cast<char*>(source.data()),source.size()*2)),
                "truncated MoE tensor");
        for (int col=0;col<item.n;++col)
            for (int row=0;row<item.k;++row) matrix[size_t(row)*item.n+col]=source[size_t(col)*item.k+row];
        const WeightKey key{"phase11-moe",name,item.file,"kn-row-major",worker.pci(),item.offset,
                            mlu::WeightType::bf16,item.k,item.n};
        registry.host(key,matrix.data());
        auto &weight=weights[name];
        weight=registry.acquire(key);
        return *weight;
    };
    const auto gemm=[&](const std::string &name,const uint16_t *src,float *out,int rows) {
        const auto &w=packed(name);
        check(phase8_candidate_bf16(queue,src,w.data(),out,rows,w.n(),w.k()),"MoE projection");
    };
    bool passed=true;
    const std::vector<int> cases=argc==6?std::vector<int>{std::stoi(argv[5])}:std::vector<int>{1,7};
    for (const int rows:cases) {
        require(rows>0&&rows<=128,"unsupported MoE test length");
        std::vector<void*> storage;
        const auto alloc=[&](size_t bytes) {
            auto *p=memory.allocate(bytes,Kind::workspace);
            storage.push_back(p);
            return p;
        };
        const auto low=[&](size_t n){return static_cast<uint16_t*>(alloc(n*2));};
        const auto f32=[&](size_t n){return static_cast<float*>(alloc(n*4));};
        auto *input=low(rows*hidden),*unused_probs=low(rows*n_experts);
        const auto scores=rows*n_experts;
        auto *logits=f32(scores),*prob=f32(scores),*max=f32(rows),*denom=f32(rows);
        auto *route_ids=static_cast<uint32_t*>(alloc(rows*k_routes*4));
        auto *route_weights=f32(rows*k_routes);
        auto *routed=low(rows*k_routes*hidden),*shared=low(rows*hidden),*result=low(rows*hidden);
        auto *mid=low(rows*shared_width);
        auto *gate=f32(rows*shared_width),*up=f32(rows*shared_width),
             *down=f32(rows*hidden);
        const auto file=[&](std::string name){return fixture+'/'+name+'-'+std::to_string(rows)+".bin";};
        auto host=read<uint16_t>(file("input"),rows*hidden);
        check(cnrtMemcpy(input,host.data(),host.size()*2,cnrtMemcpyHostToDev),"upload matched MoE input");
        gemm("gate.weight",input,logits,rows);
        check(phase9_softmax_bf16(queue,logits,nullptr,0,max,denom,prob,unused_probs,
                                  rows,n_experts,n_experts),"router FP32 Softmax");
        check(phase11_route(queue,prob,route_ids,route_weights,rows),"router top-six");
        worker.sync(qid);
        // Only integer expert IDs cross to the scheduling thread. Weights,
        // activations and FP32 reduction stay on device.
        const auto chosen=fetch(route_ids,rows*k_routes);
        auto checkpoint_logits=read<float>(file("logits"),scores);
        passed &= compare("logits",rows,fetch(logits,scores),checkpoint_logits);
        passed &= compare("prob",rows,fetch(prob,scores),read<float>(file("prob"),scores));
        passed &= compare("ids",rows,chosen,read<uint32_t>(file("ids"),rows*k_routes));
        auto checkpoint_weights=read<float>(file("weights"),rows*k_routes);
        passed &= compare("weights",rows,fetch(route_weights,rows*k_routes),checkpoint_weights);
        const auto run_expert=[&](const std::string &name,const uint16_t *x,uint16_t *out,int count,int width) {
            gemm(name+".gate_proj.weight",x,gate,count);
            gemm(name+".up_proj.weight",x,up,count);
            check(phase11_swiglu(queue,gate,up,mid,count*width),"expert SwiGLU");
            gemm(name+".down_proj.weight",mid,down,count);
            check(phase10_cast_bf16(queue,down,out,count),"expert output cast");
        };
        for (int slot=0;slot<rows*k_routes;++slot) {
            require(chosen[slot]<n_experts,"native router id out of bounds");
            run_expert("experts."+std::to_string(chosen[slot]),input+(slot/k_routes)*hidden,
                       routed+slot*hidden,1,1408);
        }
        run_expert("shared_experts",input,shared,rows,shared_width);
        check(phase11_expert_combine(queue,routed,route_weights,shared,result,rows),"device expert reduction");
        worker.sync(qid);
        passed &= compare("routed",rows,fetch(routed,rows*k_routes*hidden),
                          read<uint16_t>(file("routed"),rows*k_routes*hidden));
        passed &= compare("shared",rows,fetch(shared,rows*hidden),read<uint16_t>(file("shared"),rows*hidden));
        const auto expected=read<uint16_t>(file("output"),rows*hidden);
        passed &= compare("output",rows,fetch(result,rows*hidden),expected);
        if (argc==6) {
            check(cnrtMemcpy(logits,checkpoint_logits.data(),checkpoint_logits.size()*4,
                             cnrtMemcpyHostToDev),"diagnostic checkpoint router logits");
            check(phase9_softmax_bf16(queue,logits,nullptr,0,max,denom,prob,unused_probs,
                                      rows,n_experts,n_experts),"matched-logit device Softmax");
            check(phase11_route(queue,prob,route_ids,route_weights,rows),"matched-logit device top-six");
            worker.sync(qid);
            require(fetch(route_ids,rows*k_routes)==chosen,"diagnostic route changed the expert order");
            compare("weights_matched_checkpoint_logits",rows,fetch(route_weights,rows*k_routes),
                    checkpoint_weights);
            check(phase11_expert_combine(queue,routed,route_weights,shared,result,rows),
                  "matched-logit device expert reduction");
            worker.sync(qid);
            compare("output_matched_checkpoint_logits",rows,fetch(result,rows*hidden),expected);
            // Diagnostic control only. The full model never uploads router weights.
            check(cnrtMemcpy(route_weights,checkpoint_weights.data(),checkpoint_weights.size()*4,
                             cnrtMemcpyHostToDev),"diagnostic checkpoint route weights");
            check(phase11_expert_combine(queue,routed,route_weights,shared,result,rows),
                  "matched-weight device reduction");
            worker.sync(qid);
            compare("output_matched_checkpoint_weights",rows,fetch(result,rows*hidden),expected);
        }
        for (auto &p:storage) memory.release(p);
    }
    weights.clear();
    registry.close();
    memory.close();
    worker.close();
    std::cout<<"layer="<<layer<<" moe="<<(passed?"PASS":"FAIL")<<" live_managed_bytes="<<memory.usage().live<<'\n';
    return passed?0:1;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 2;
}
