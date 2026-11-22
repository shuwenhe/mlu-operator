#include "../runtime.h"
#include "../workspace.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

extern "C" {
cnrtRet_t phase11_embedding(cnrtQueue_t,const uint16_t*,const uint32_t*,uint16_t*,int);
cnrtRet_t phase11_logits(cnrtQueue_t,const float*,float*,int,int,int);
cnrtRet_t phase11_greedy(cnrtQueue_t,const float*,uint32_t*,int);
cnrtRet_t phase8_candidate_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,float*,int,int,int);
cnrtRet_t phase82_native_bf16(cnrtQueue_t,const uint16_t*,const uint16_t*,uint16_t*,float*,uint16_t*,int,int,float);
}

using namespace native_mlu;
namespace {
constexpr int hidden=2048,vocab=102400,block=10240;
using Clock=std::chrono::steady_clock;

void require(bool value,const char *message) {
    if (!value) throw std::runtime_error(message);
}
void read_bytes(std::istream& in,void *dest,size_t bytes) {
    require(bool(in.read(static_cast<char*>(dest),bytes)),"truncated checkpoint or fixture");
}
template<typename T> auto read(const std::string &path,size_t count) {
    std::vector<T> result(count);
    std::ifstream in(path,std::ios::binary);
    read_bytes(in,result.data(),count*sizeof(T));
    require(in.peek()==EOF,"unexpected fixture trailing bytes");
    return result;
}
template<typename T> auto download(T *device,size_t count) {
    std::vector<T> result(count);
    check(cnrtMemcpy(result.data(),device,count*sizeof(T),cnrtMemcpyDevToHost),"download native boundary");
    return result;
}
struct Shard {
    std::string path;
    size_t base,bytes;
    void *device;
};
struct Slice {size_t shard,offset;};

class Checkpoint {
public:
    Checkpoint(Memory &memory,const std::string &dir,std::istream &manifest):memory_(memory) {
        size_t count=0;
        require(bool(manifest>>count)&&count==4,"invalid checkpoint manifest shard count");
        std::vector<char> staging(16*1024*1024);
        shards_.reserve(count);
        for (size_t i=0;i<count;++i) {
            Shard shard{};
            require(bool(manifest>>shard.path>>shard.base>>shard.bytes)&&shard.bytes>0,
                    "invalid checkpoint shard metadata");
            require(std::filesystem::path(shard.path).filename()==shard.path,"invalid shard name");
            shard.path=dir+"/"+shard.path;
            std::ifstream source(shard.path,std::ios::binary);
            require(bool(source.seekg(shard.base))&&std::filesystem::file_size(shard.path)==shard.base+shard.bytes,
                    "invalid shard payload");
            shard.device=memory_.allocate(shard.bytes,Kind::weight);
            shards_.push_back(shard);
            for (size_t offset=0;offset<shard.bytes;) {
                const size_t size=std::min(staging.size(),shard.bytes-offset);
                read_bytes(source,staging.data(),size);
                check(cnrtMemcpy(static_cast<char*>(shard.device)+offset,staging.data(),size,cnrtMemcpyHostToDev),
                      "upload BF16 checkpoint shard");
                offset+=size;
            }
            std::cout<<"shard_resident="<<i<<" payload_bytes="<<shard.bytes<<'\n'<<std::flush;
        }
    }
    void close() {
        for (auto &shard:shards_) memory_.release(shard.device);
    }
    const Shard &shard(Slice slice,size_t bytes) const {
        const auto &entry=shards_.at(slice.shard);
        require(slice.offset<=entry.bytes&&bytes<=entry.bytes-slice.offset&&!(slice.offset%2),
                "checkpoint slice out of bounds");
        return entry;
    }
    const uint16_t *device(Slice slice,size_t bytes) const {
        const auto &entry=shard(slice,bytes);
        return reinterpret_cast<const uint16_t*>(static_cast<const char*>(entry.device)+slice.offset);
    }
private:
    Memory &memory_;
    std::vector<Shard> shards_;
};
Slice slice(std::istream& info) {
    Slice result{};
    require(bool(info>>result.shard>>result.offset),"missing tensor metadata");
    return result;
}

template<typename T> size_t compare(const char *stage,int rows,const std::vector<T>& actual,
                                    const std::vector<T>& ref) {
    size_t bits=0,failures=0,nonfinite=0;
    double max_abs=0,max_rel=0;
    for (size_t i=0;i<ref.size();++i) {
        bits+=actual[i]!=ref[i];
        const double got=actual[i], expected=ref[i];
        const double error=std::abs(got-expected);
        const bool invalid=!std::isfinite(got)||!std::isfinite(expected);
        nonfinite+=invalid;
        failures+=invalid||error>1e-5+1e-5*std::abs(expected);
        max_abs=std::max(max_abs,error);
        max_rel=std::max(max_rel,error/std::max(1e-30,std::abs(expected)));
    }
    std::cout<<"stage="<<stage<<" rows="<<rows<<" elements="<<ref.size()<<" differences="<<bits
             <<" mismatches="<<failures<<" nonfinite="<<nonfinite
             <<" max_abs="<<max_abs<<" max_rel="<<max_rel<<'\n';
    return std::is_same_v<T,float>?failures:bits;
}
}

int main(int argc,char **argv) try {
    if (argc!=4) return 2;
    Worker worker(argv[1]);
    Memory memory(worker);
    Registry registry(worker);
    const auto queue_id=worker.add_queue();
    const auto queue=worker.queue(queue_id);
    const std::string root=argv[2],fixtures=argv[3];
    std::ifstream info(fixtures+"/checkpoint.txt");
    const auto start=Clock::now();
    Checkpoint checkpoint(memory,root,info);
    const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
    const auto embed=slice(info), gamma=slice(info), lm=slice(info);
    const auto *embedding=checkpoint.device(embed,size_t(vocab)*hidden*2);
    const auto *norm=checkpoint.device(gamma,hidden*2);
    const auto &head=checkpoint.shard(lm,size_t(vocab)*hidden*2);
    std::cout<<"raw_bf16_checkpoint=PASS resident_bytes="<<memory.usage().live
             <<" load_seconds="<<seconds<<'\n'<<std::flush;
    const auto id_path=fixtures+"/ids.bin";
    const int count=std::filesystem::file_size(id_path)/sizeof(uint32_t);
    require(count>=2&&count<=128,"invalid fixture token count");
    auto ids=read<uint32_t>(id_path,count);
    for (auto id:ids) require(id<vocab,"fixture token outside vocabulary");
    std::vector<void*> scratch;
    const auto alloc=[&](size_t bytes) {
        auto *ptr=memory.allocate(bytes,Kind::workspace);
        scratch.push_back(ptr);
        return ptr;
    };
    auto *device_ids=static_cast<uint32_t*>(alloc(count*4)), *tokens=static_cast<uint32_t*>(alloc(count*4));
    auto *embedded=static_cast<uint16_t*>(alloc(count*hidden*2)),
         *normalized=static_cast<uint16_t*>(alloc(count*hidden*2)),
         *rounded=static_cast<uint16_t*>(alloc(count*hidden*2));
    auto *norm_wide=static_cast<float*>(alloc(count*hidden*4)),
         *partial=static_cast<float*>(alloc(count*block*4)),
         *logits=static_cast<float*>(alloc(count*vocab*4));
    check(cnrtMemcpy(device_ids,ids.data(),count*4,cnrtMemcpyHostToDev),"upload tokenizer IDs");
    size_t failures=0,peak_packed=0;
    for (int rows:{1,count}) {
        check(phase11_embedding(queue,embedding,device_ids,embedded,rows),"embedding lookup");
        check(phase82_native_bf16(queue,embedded,norm,normalized,norm_wide,rounded,rows,hidden,1e-6f),
              "final RMSNorm boundary");
        worker.sync(queue_id);
        const auto tag=std::to_string(rows)+".bin";
        failures+=compare("embedding",rows,download(embedded,rows*hidden),
                          read<uint16_t>(fixtures+"/embedding-"+tag,rows*hidden));
        failures+=compare("final_norm",rows,download(normalized,rows*hidden),
                          read<uint16_t>(fixtures+"/normalized-"+tag,rows*hidden));
        std::ifstream weights(head.path,std::ios::binary);
        require(bool(weights.seekg(head.base+lm.offset)),"seek LM head");
        std::vector<uint16_t> nk(size_t(hidden)*block),kn(nk.size());
        for (int col=0;col<vocab;col+=block) {
            read_bytes(weights,nk.data(),nk.size()*2);
            for (int j=0;j<block;j+=32)
                for (int i=0;i<hidden;++i)
                    for (int t=j;t<j+32;++t)
                        kn[size_t(i)*block+t]=nk[size_t(t)*hidden+i];
            const auto part=std::to_string(col);
            const WeightKey key{part,"lm_head",head.path,"kn-row-major",worker.pci(),
                                uint64_t(lm.offset)+uint64_t(col)*hidden*2,
                                mlu::WeightType::bf16,hidden,block};
            registry.host(key,kn.data());
            peak_packed=std::max(peak_packed,registry.bytes());
            {
                const auto w=registry.acquire(key);
                check(phase8_candidate_bf16(queue,normalized,w->data(),partial,rows,block,hidden),
                      "vocabulary GEMM");
                check(phase11_logits(queue,partial,logits,rows,block,col),"widen checkpoint logits");
                worker.sync(queue_id);
            }
            registry.invalidate(part);
            registry.collect();
        }
        check(phase11_greedy(queue,logits,tokens,rows),"greedy argmax");
        worker.sync(queue_id);
        failures+=compare("logits",rows,download(logits,size_t(rows)*vocab),
                          read<float>(fixtures+"/logits-"+tag,size_t(rows)*vocab));
        const auto actual=download(tokens,rows);
        failures+=compare("argmax",rows,actual,read<uint32_t>(fixtures+"/argmax-"+tag,rows));
        std::cout<<"isolated_argmax_ids";
        for (auto id:actual) std::cout<<' '<<id;
        std::cout<<'\n';
    }
    for (auto &ptr:scratch) memory.release(ptr);
    registry.close();
    checkpoint.close();
    memory.close();
    worker.close();
    rusage usage{};
    require(getrusage(RUSAGE_SELF,&usage)==0,"read peak host process memory");
    std::cout<<"peak_managed_mlu_bytes="<<memory.usage().peak<<" peak_extra_packed_bytes="<<peak_packed
             <<" host_peak_kib="<<usage.ru_maxrss<<" live_managed_bytes="<<memory.usage().live
             <<" isolated_boundaries="<<(failures?"FAIL":"PASS")<<'\n';
    return failures?1:0;
} catch (const std::exception &error) {
    std::cerr<<error.what()<<'\n';
    return 2;
}
