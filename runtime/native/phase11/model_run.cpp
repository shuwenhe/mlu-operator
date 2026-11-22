#include "dense.h"
#include "../phase10/layout.h"
#include "../phase9/softmax/softmax.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

extern "C" {
cnrtRet_t phase8_candidate_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,float*,int,int,int);
cnrtRet_t phase82_native_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,int,int,float);
cnrtRet_t phase11_residual(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,int);
cnrtRet_t phase11_swiglu(cnrtQueue_t,const float*,const float*,uint16_t*,int);
cnrtRet_t phase11_embedding(cnrtQueue_t,const uint16_t*,const uint32_t*,uint16_t*,int);
cnrtRet_t phase11_logits(cnrtQueue_t,const float*,float*,int,int,int);
cnrtRet_t phase11_greedy(cnrtQueue_t,const float*,uint32_t*,int);
cnrtRet_t phase11_route(cnrtQueue_t,const float*,uint32_t*,float*,int);
cnrtRet_t phase11_expert_combine(cnrtQueue_t,const uint16_t*,const float*,const uint16_t*,uint16_t*,int);
}
using namespace native_mlu;
namespace {
constexpr int hidden=2048,vocab=102400,head_tile=10240,experts=64,routes=6,table_rows=131;
using Clock=std::chrono::steady_clock;
struct Tensor {std::string file; uint64_t offset; int n,k;};
void require(bool ok,const char *msg) {
    if (!ok) throw std::runtime_error(msg);
}
template<typename T> void trace(const std::string& file,const T *device,size_t n) {
    std::vector<T> host(n);
    check(cnrtMemcpy(host.data(),const_cast<T*>(device),n*sizeof(T),cnrtMemcpyDevToHost),"model trace download");
    std::ofstream out(file,std::ios::binary);
    require(bool(out.write(reinterpret_cast<const char*>(host.data()),n*sizeof(T))),"write model trace");
}
class Weights {
public:
    Weights(Worker &worker,Memory &memory,Registry &registry,std::string root,const std::string &manifest)
      :worker_(worker),memory_(memory),registry_(registry),root_(std::move(root)) {
        std::ifstream input(manifest);
        for (std::string name;input>>name;) {
            auto &t=tensors_[name];
            require(bool(input>>t.file>>t.offset>>t.n>>t.k)&&t.n>0&&t.k>=0&&
                    std::filesystem::path(t.file).filename()==t.file,"bad model manifest");
        }
        require(tensors_.size()==5291,"unexpected model tensor count");
    }
    uint16_t *resident(const std::string &name,int n,int k) {
        const auto& t=tensors_.at(name);
        require(t.n==n&&t.k==k,"resident tensor shape mismatch");
        std::ifstream in(root_+'/'+t.file,std::ios::binary);
        require(bool(in.seekg(t.offset)),"seek resident tensor");
        const size_t size=size_t(n)*std::max(k,1)*2;
        return upload(in,size);
    }
    uint16_t *table(const std::string &path,size_t bytes) {
        std::ifstream in(path,std::ios::binary);
        require(in.good(),"missing rotary table");
        return upload(in,bytes);
    }
    Registry::Lease packed(const std::string &name,int start=0,int width=0) {
        const auto& t=tensors_.at(name);
        const int n=width?width:t.n;
        require(start>=0&&n>0&&n<=t.n-start&&t.k>0,"packed tensor extent mismatch");
        const auto off=t.offset+size_t(start)*t.k*2;
        const WeightKey key{"phase11-model",name,t.file,"kn-row-major",worker_.pci(),off,
                            mlu::WeightType::bf16,t.k,n};
        if (auto existing=registry_.acquire(key)) return existing;
        std::vector<uint16_t> nk(size_t(n)*t.k),kn(nk.size());
        std::ifstream in(root_+'/'+t.file,std::ios::binary);
        require(bool(in.seekg(off))&&bool(in.read(reinterpret_cast<char*>(nk.data()),nk.size()*2)),
                "missing model projection bytes");
        for (int col=0;col<n;col+=32)
            for (int row=0;row<t.k;++row)
                for (int j=col;j<std::min(n,col+32);++j) kn[size_t(row)*n+j]=nk[size_t(j)*t.k+row];
        registry_.host(key,kn.data());
        return registry_.acquire(key);
    }
    uint64_t bytes() const {return registry_.bytes();}
    void close() {
        for (auto &p:tables_) memory_.release(p);
    }
private:
    uint16_t *upload(std::istream &in,size_t bytes) {
        auto *p=memory_.allocate(bytes,Kind::weight);
        tables_.push_back(p);
        constexpr size_t window=16*1024*1024;
        std::vector<char> host(std::min(window,bytes));
        for (size_t offset=0;offset<bytes;) {
            const size_t n=std::min(host.size(),bytes-offset);
            require(bool(in.read(host.data(),n)),"truncated model tensor bytes");
            check(cnrtMemcpy(static_cast<char*>(p)+offset,host.data(),n,cnrtMemcpyHostToDev),"upload model tensor");
            offset+=n;
        }
        return static_cast<uint16_t*>(p);
    }
    Worker &worker_;
    Memory &memory_;
    Registry &registry_;
    std::string root_;
    std::map<std::string,Tensor> tensors_;
    std::vector<void*> tables_;
};

struct Layer {
    std::string prefix;
    const uint16_t *input_gamma;
    const uint16_t *post_gamma;
    std::unique_ptr<MlaRequest> attention;
};

class Model {
public:
    Model(Worker &worker,Memory &memory,size_t qid,Weights &weights,
          const std::string &rope,int chunk,int capacity)
      :worker_(worker),memory_(memory),qid_(qid),queue_(worker.queue(qid)),weights_(weights) {
        embedding_=weights.resident("model.embed_tokens.weight",vocab,hidden);
        gamma_=weights.resident("model.norm.weight",hidden,0);
        const auto *cosine=weights.table(rope+"/cos.bin",table_rows*64*2);
        const auto *sine=weights.table(rope+"/sin.bin",table_rows*64*2);
        std::ifstream scale_file(rope+"/scale.txt");
        float scale=0;
        require(bool(scale_file>>scale),"missing model attention scale");
        for (int layer=0;layer<27;++layer) {
            const auto name="model.layers."+std::to_string(layer)+'.';
            const uint16_t *input_gamma=weights.resident(name+"input_layernorm.weight",hidden,0);
            const uint16_t *post_gamma=weights.resident(name+"post_attention_layernorm.weight",hidden,0);
            const auto a=name+"self_attn.";
            MlaWeights attention{{weights.packed(a+"q_proj.weight"),
                                  weights.packed(a+"kv_a_proj_with_mqa.weight"),
                                  weights.packed(a+"kv_b_proj.weight"),
                                  weights.packed(a+"o_proj.weight")},
                                 weights.resident(a+"kv_a_layernorm.weight",512,0),
                                 cosine,sine,table_rows,scale};
            if (layer==0) {
                DenseWeights dense{attention,{weights.packed(name+"mlp.gate_proj.weight"),
                    weights.packed(name+"mlp.up_proj.weight"),weights.packed(name+"mlp.down_proj.weight")},
                    input_gamma,post_gamma};
                dense_=std::make_unique<DenseDecoder>(worker,memory,qid,std::move(dense),chunk,capacity);
            } else {
                layers_.push_back({name,input_gamma,post_gamma,std::make_unique<MlaRequest>(
                    worker,memory,qid,std::move(attention),chunk,capacity,true)});
            }
        }
        for (int col=0;col<vocab;col+=head_tile)
            lm_.push_back(weights.packed("lm_head.weight",col,head_tile));
        const auto low=[&](size_t width){return static_cast<uint16_t*>(allocate(size_t(chunk)*width*2));};
        const auto f32=[&](size_t width){return static_cast<float*>(allocate(size_t(chunk)*width*4));};
        input_ids_=static_cast<uint32_t*>(allocate(chunk*4));
        token_=static_cast<uint32_t*>(allocate(4));
        post_=low(hidden); residual_=low(hidden); result_=low(hidden); normalized_=low(hidden);
        norm_round_=low(hidden); norm_wide_=f32(hidden);
        route_ids_=static_cast<uint32_t*>(allocate(chunk*routes*4));
        router_logits_=f32(experts); router_probs_=f32(experts); router_low_=low(experts);
        router_max_=f32(1); router_denom_=f32(1); row_weights_=f32(routes);
        routed_=low(routes*hidden); shared_=low(hidden); mixed_=low(hidden);
        gate_=f32(2816); up_=f32(2816); mid_=low(2816); down_=f32(hidden);
        partial_=static_cast<float*>(allocate(head_tile*4));
        logits_=static_cast<float*>(allocate(vocab*4));
    }

    void prompt(std::vector<uint32_t> &ids) {
        check(cnrtMemcpy(input_ids_,ids.data(),ids.size()*4,cnrtMemcpyHostToDev),"upload prompt");
    }
    uint32_t run(int cursor,int count,const std::string &dump) {
        auto *const embedded=dense_->buffers().input;
        check(phase11_embedding(queue_,embedding_,cursor?token_:input_ids_,embedded,count),
              "model embedding");
        if (!dump.empty()) record(dump+"embedding.bin",embedded,count*hidden);
        dense_->run(cursor,count);
        const uint16_t *x=dense_->buffers().output;
        if (!dump.empty()) {
            record(dump+"layer0.bin",x,count*hidden);
            record_attention(dump,0,dense_->attention(),count);
        }
        for (auto &layer:layers_) {
            const auto number=&layer-layers_.data()+1;
            auto &request=*layer.attention;
            normalize(x,layer.input_gamma,request.tensors().input,count);
            request.enqueue(cursor,count);
            check(phase11_residual(queue_,x,request.tensors().output,residual_,count*hidden),
                  "model attention residual");
            normalize(residual_,layer.post_gamma,post_,count);
            run_moe(layer.prefix+"mlp.",count);
            check(phase11_residual(queue_,residual_,mixed_,result_,count*hidden),"model MoE residual");
            request.complete();
            x=result_;
            if (!dump.empty()) {
                record(dump+"layer"+std::to_string(number)+".bin",result_,count*hidden);
                record_attention(dump,number,request,count);
            }
        }
        normalize(x,gamma_,normalized_,count);
        const auto *last=normalized_+size_t(count-1)*hidden;
        int col=0;
        for (const auto &w:lm_) {
            project(*w,last,partial_,1);
            check(phase11_logits(queue_,partial_,logits_,1,head_tile,col),"model LM head");
            col+=head_tile;
        }
        check(phase11_greedy(queue_,logits_,token_,1),"model token selection");
        worker_.sync(qid_);
        if (!dump.empty()) trace(dump+"logits.bin",logits_,vocab);
        uint32_t next;
        check(cnrtMemcpy(&next,token_,sizeof(next),cnrtMemcpyDevToHost),"read selected token");
        return next;
    }
    void close() {
        worker_.sync(qid_);
        layers_.clear();
        dense_.reset();
        lm_.clear();
        for (auto &p:scratch_) memory_.release(p);
    }
private:
    void *allocate(size_t bytes) {
        auto *p=memory_.allocate(bytes,Kind::workspace);
        scratch_.push_back(p);
        return p;
    }
    template<typename T> void record(const std::string &file,T *data,size_t n) {
        worker_.sync(qid_);
        trace(file,data,n);
    }
    void record_attention(const std::string &dump,int index,const MlaRequest &request,int count) {
        const auto &t=request.tensors();
        const auto prefix=dump+"layer"+std::to_string(index);
        trace(prefix+"-attention.bin",t.output,count*hidden);
        const auto cache=[&](const char *name,uint16_t *data,int width) {
            const size_t length=request.length(), physical=request.capacity();
            std::vector<uint16_t> host(length*width);
            std::ofstream file(prefix+name,std::ios::binary);
            for (size_t head=0;head<16;++head) {
                check(cnrtMemcpy(host.data(),data+head*physical*width,host.size()*2,
                                 cnrtMemcpyDevToHost),"trace committed cache prefix");
                require(bool(file.write(reinterpret_cast<const char*>(host.data()),host.size()*2)),
                        "write model cache trace");
            }
        };
        cache("-key.bin",t.key,192);
        cache("-value.bin",t.value,128);
    }
    void normalize(const uint16_t *x,const uint16_t *gamma,uint16_t *out,int count) {
        check(phase82_native_bf16(queue_,x,gamma,out,norm_wide_,norm_round_,count,hidden,1e-6f),
              "model RMSNorm");
    }
    void project(const mlu::PackedWeight &w,const uint16_t *x,float *out,int count) {
        check(phase8_candidate_bf16(queue_,x,w.data(),out,count,w.n(),w.k()),"model GEMM");
    }
    void expert(const std::string &name,const uint16_t *x,uint16_t *out,int count,int width) {
        project(*weights_.packed(name+".gate_proj.weight"),x,gate_,count);
        project(*weights_.packed(name+".up_proj.weight"),x,up_,count);
        check(phase11_swiglu(queue_,gate_,up_,mid_,count*width),"model expert SwiGLU");
        project(*weights_.packed(name+".down_proj.weight"),mid_,down_,count);
        check(phase10_cast_bf16(queue_,down_,out,count),"model expert cast");
    }
    void run_moe(const std::string &prefix,int count) {
        project(*weights_.packed(prefix+"gate.weight"),post_,router_logits_,count);
        check(phase9_softmax_bf16(queue_,router_logits_,nullptr,0,router_max_,router_denom_,
                router_probs_,router_low_,count,experts,experts),"model router FP32 Softmax");
        check(phase11_route(queue_,router_probs_,route_ids_,row_weights_,count),"model route top-six");
        worker_.sync(qid_);
        std::vector<uint32_t> selected(count*routes);
        check(cnrtMemcpy(selected.data(),route_ids_,selected.size()*4,cnrtMemcpyDevToHost),
              "read expert IDs for scheduling");
        for (size_t i=0;i<selected.size();++i) {
            require(selected[i]<experts,"router selected invalid expert");
            expert(prefix+"experts."+std::to_string(selected[i]),post_+(i/routes)*hidden,
                   routed_+i*hidden,1,1408);
        }
        expert(prefix+"shared_experts",post_,shared_,count,2816);
        check(phase11_expert_combine(queue_,routed_,row_weights_,shared_,mixed_,count),
              "model weighted expert combine");
    }
    Worker &worker_;
    Memory &memory_;
    size_t qid_;
    cnrtQueue_t queue_;
    Weights &weights_;
    std::unique_ptr<DenseDecoder> dense_;
    std::vector<Layer> layers_;
    std::vector<Registry::Lease> lm_;
    std::vector<void*> scratch_;
    const uint16_t *embedding_,*gamma_;
    uint16_t *post_,*residual_,*result_,*normalized_,*norm_round_,*router_low_,*routed_,*shared_,*mixed_,*mid_;
    float *norm_wide_,*router_logits_,*router_probs_,*router_max_,*router_denom_,*row_weights_;
    float *gate_,*up_,*down_,*partial_,*logits_;
    uint32_t *input_ids_,*token_,*route_ids_;
};
}

int main(int argc,char **argv) try {
    if (argc!=7&&argc!=8) return 2;
    const std::string files=argv[3],rope=argv[4],prompt=argv[5],dump=argc==8?argv[7]:"";
    const int steps=std::stoi(argv[6]);
    const auto bytes=std::filesystem::file_size(prompt);
    require(bytes%4==0&&bytes/4>0&&bytes/4<=128&&steps>0&&steps<=table_rows&&
            bytes/4+steps-1<=table_rows,"unsupported prompt or generation length");
    std::vector<uint32_t> ids(bytes/4);
    std::ifstream tokens(prompt,std::ios::binary);
    require(bool(tokens.read(reinterpret_cast<char*>(ids.data()),bytes)),"truncated prompt");
    for (auto id:ids) require(id<vocab,"invalid prompt token");
    Worker worker(argv[1]);
    Memory memory(worker);
    Registry registry(worker);
    const auto queue=worker.add_queue();
    Weights weights(worker,memory,registry,argv[2],files+"/tensors.txt");
    const auto start=Clock::now();
    Model model(worker,memory,queue,weights,rope,ids.size(),ids.size()+steps-1);
    const auto seconds=std::chrono::duration<double>(Clock::now()-start).count();
    std::cout<<"model_startup_seconds="<<seconds<<" packed_bytes="<<weights.bytes()
             <<" managed_bytes="<<memory.usage().live<<'\n'<<std::flush;
    model.prompt(ids);
    int emitted=0;
    for (int step=0;step<steps;++step) {
        const auto begin=Clock::now();
        const int cursor=step?int(ids.size())+step-1:0;
        const auto output=dump.empty()?"":dump+"/"+std::to_string(step)+"-";
        const auto id=model.run(cursor,step?1:ids.size(),output);
        const double ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        ++emitted;
        std::cout<<"generated "<<step<<' '<<id<<' '<<ms<<'\n'<<std::flush;
        if (id==100001) break;
    }
    rusage host{};
    require(getrusage(RUSAGE_SELF,&host)==0,"model host resource usage");
    std::cout<<"generated_count="<<emitted<<" host_peak_kib="<<host.ru_maxrss
             <<" managed_peak_bytes="<<memory.usage().peak<<" packed_bytes="<<weights.bytes()<<'\n';
    model.close();
    registry.close();
    weights.close();
    memory.close();
    worker.close();
    return 0;
} catch (const std::exception &error) {
    std::cerr<<error.what()<<'\n';
    return 2;
}
