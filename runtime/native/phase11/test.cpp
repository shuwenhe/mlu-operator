#include "dense.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace native_mlu;
namespace {
constexpr int hidden=2048, width=10944;
template<typename T> auto read(const std::string& file,size_t count) {
    std::vector<T> result(count);
    std::ifstream in(file,std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(result.data()),count*sizeof(T))||in.peek()!=EOF)
        throw std::runtime_error("invalid fixture: "+file);
    return result;
}
uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits,&value,4);
    return (bits+0x7fff+((bits>>16)&1))>>16;
}
float expand(uint16_t value) {
    uint32_t bits=uint32_t(value)<<16;
    float result;
    std::memcpy(&result,&bits,4);
    return result;
}
uint16_t quantize(float value) {return bf16(value);}
uint16_t quantize(uint16_t value) {return value;}
template<typename T> bool compare(std::string file,T *buffer,size_t count,int cursor,int request,
                                  const char *stage) {
    const auto ref=read<float>(file,count);
    std::vector<T> actual(count);
    check(cnrtMemcpy(actual.data(),buffer,count*sizeof(T),cnrtMemcpyDevToHost),"download stage");
    size_t failures=0,differ=0,nonfinite=0;
    double max_abs=0,max_rel=0;
    unsigned ulp=0;
    for (size_t i=0;i<count;++i) {
        const auto bits=quantize(actual[i]), expected=bf16(ref[i]);
        const double val=expand(bits), wanted=ref[i];
        const double error=std::abs(val-wanted);
        const auto ordered=[](uint16_t b){return b&0x8000?-int(b&0x7fff):int(b);};
        const bool invalid=!std::isfinite(val)||!std::isfinite(wanted);
        nonfinite+=invalid;
        failures+=invalid||error>1e-5+1e-5*std::abs(wanted);
        differ+=bits!=expected;
        ulp=std::max(ulp,unsigned(std::abs(ordered(bits)-ordered(expected))));
        max_abs=std::max(max_abs,error);
        max_rel=std::max(max_rel,error/std::max(std::abs(wanted),1e-30));
    }
    std::cout<<cursor<<','<<request<<','<<stage<<','<<count<<','<<failures<<','<<differ<<','
             <<nonfinite<<','<<max_abs<<','<<max_rel<<','<<ulp<<'\n';
    return failures==0;
}
}

int main(int argc,char **argv) try {
    if (argc!=6) return 2;
    Worker worker(argv[1]);
    Memory memory(worker);
    Registry registry(worker);
    const auto queue=worker.add_queue();
    const std::string mla=argv[2],ffn=argv[3],reference=argv[4];
    const int prefill=std::stoi(argv[5]);
    if (prefill!=1&&prefill!=7&&prefill!=32&&prefill!=128) return 2;
    std::vector<void*> shared;
    const auto load_table=[&](const std::string &path,size_t count) {
        auto words=read<uint16_t>(path,count);
        auto *device=static_cast<uint16_t*>(memory.allocate(count*2,Kind::weight));
        shared.push_back(device);
        check(cnrtMemcpy(device,words.data(),count*2,cnrtMemcpyHostToDev),"upload weights");
        return device;
    };
    const auto load_gemm=[&](const std::string &path,const std::string &name,int k,int n) {
        auto data=read<uint16_t>(path,size_t(k)*n);
        const WeightKey key{"phase11",name,"checkpoint","kn-row-major",worker.pci(),0,
                            mlu::WeightType::bf16,k,n};
        registry.host(key,data.data());
        return registry.acquire(key);
    };
    bool passed=true;
    {
        DenseWeights weights{};
        const char *names[]={"q","kv_a","kv_b","o"};
        const int ks[]={hidden,hidden,512,hidden}, ns[]={3072,576,4096,hidden};
        for (size_t i=0;i<4;++i)
            weights.attention.projections[i]=load_gemm(mla+"/weight_"+names[i]+".bin",names[i],ks[i],ns[i]);
        weights.attention.latent_gamma=load_table(mla+"/weight_gamma.bin",512);
        weights.attention.cosine=load_table(mla+"/cos.bin",131*64);
        weights.attention.sine=load_table(mla+"/sin.bin",131*64);
        weights.attention.table_rows=131;
        std::ifstream scale(mla+"/scale.txt");
        if (!(scale>>weights.attention.scale)) throw std::runtime_error("invalid scale");
        weights.ffn={load_gemm(ffn+"/gate.bin","gate",hidden,width),
                     load_gemm(ffn+"/up.bin","up",hidden,width),
                     load_gemm(ffn+"/down.bin","down",width,hidden)};
        weights.input_gamma=load_table(ffn+"/input_gamma.bin",hidden);
        weights.post_gamma=load_table(ffn+"/post_gamma.bin",hidden);
        std::cout<<"cursor,request,stage,count,mismatches,bit_differences,nonfinite,max_abs,max_rel,max_ulp\n";
        for (int request=0;request<2;++request) {
            DenseDecoder decoder(worker,memory,queue,weights,prefill,prefill+1);
            const auto &b=decoder.buffers();
            for (int cursor:{0,prefill}) {
                const int count=cursor?1:prefill;
                const std::string prefix=reference+"/"+std::to_string(request)+"_"+std::to_string(cursor)+"_";
                const auto src=read<float>(prefix+"original_hidden.bin",count*hidden);
                std::vector<uint16_t> input;
                input.reserve(src.size());
                for (auto value:src) input.push_back(bf16(value));
                check(cnrtMemcpy(b.input,input.data(),input.size()*2,cnrtMemcpyHostToDev),"upload hidden");
                decoder.run(cursor,count);
                const auto &mla_t=decoder.attention().tensors();
                const auto stage=[&](const char *name,auto *ptr,int cols) {
                    passed &= compare(prefix+name+".bin",ptr,size_t(count)*cols,cursor,request,name);
                };
                stage("input_norm",mla_t.input,hidden);
                stage("attention",mla_t.merged,hidden);
                stage("output_projection",mla_t.output,hidden);
                stage("residual",b.residual,hidden);
                stage("post_norm",b.post_norm,hidden);
                stage("gate",b.gate,width);
                stage("up",b.up,width);
                stage("swiglu",b.swiglu,width);
                stage("down",b.ffn,hidden);
                stage("final",b.output,hidden);
                if (decoder.attention().length()!=cursor+count)
                    throw std::runtime_error("wrong decoder cache length");
            }
        }
    }
    for (auto &p:shared) memory.release(p);
    registry.close();
    memory.close();
    worker.close();
    std::cerr<<"decoder="<<(passed?"PASS":"FAIL")<<" prefill="<<prefill
             <<" peak_managed_bytes="<<memory.usage().peak<<" live_bytes="<<memory.usage().live<<'\n';
    return passed?0:1;
} catch (const std::exception &error) {
    std::cerr<<error.what()<<'\n';
    return 2;
}
