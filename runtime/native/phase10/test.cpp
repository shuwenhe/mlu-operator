#include "mla.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <tuple>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using namespace native_mlu;
namespace {
constexpr int heads=16, hd=2048, kw=192, vw=128, table_rows=131;
constexpr size_t guard_words=64;
constexpr uint16_t poison=0x5a5a;
bool bf16;

void require(bool value,const char *message) {
    if (!value) throw std::runtime_error(message);
}
template<typename T> std::vector<T> load(const std::string& path,size_t count) {
    std::vector<T> result(count);
    std::ifstream source(path,std::ios::binary);
    if (!source.read(reinterpret_cast<char*>(result.data()),count*sizeof(T)) || source.peek()!=EOF)
        throw std::runtime_error("invalid fixture: "+path);
    return result;
}
template<typename T> void upload(T *device,std::vector<T> &host) {
    check(cnrtMemcpy(device,host.data(),host.size()*sizeof(T),cnrtMemcpyHostToDev),"upload MLA fixture");
}
template<typename T> std::vector<T> download(T *device,size_t size) {
    std::vector<T> host(size);
    check(cnrtMemcpy(host.data(),device,host.size()*sizeof(T),cnrtMemcpyDevToHost),"download MLA stage");
    return host;
}
float decode(uint16_t x) {
    if (bf16) {
        uint32_t bits=uint32_t(x)<<16;
        float value;
        std::memcpy(&value,&bits,4);
        return value;
    }
    _Float16 value;
    std::memcpy(&value,&x,2);
    return float(value);
}
uint16_t encode(float x) {
    if (!bf16) {
        const _Float16 rounded=static_cast<_Float16>(x);
        uint16_t bits;
        std::memcpy(&bits,&rounded,2);
        return bits;
    }
    uint32_t bits;
    std::memcpy(&bits,&x,4);
    return (bits+0x7fff+((bits>>16)&1))>>16;
}
uint16_t word(uint16_t x) { return x; }
uint16_t word(float x) { return encode(x); }
int ordered(uint16_t bits) {
    return bits&0x8000 ? -int(bits&0x7fff) : int(bits);
}

template<typename T> bool compare(std::ostream &csv,const std::string& label,
    const std::vector<uint16_t>& ref,const std::vector<T>& actual) {
    require(ref.size()==actual.size(),"stage comparison shape mismatch");
    size_t fails=0,bits=0,nan=0,inf=0,unexpected=0,finite=0,first=ref.size();
    double max_abs=0,max_rel=0,sum=0,sum_sq=0;
    std::map<int,size_t> ulps;
    for (size_t i=0;i<ref.size();++i) {
        const uint16_t native=word(actual[i]);
        const float expected=decode(ref[i]), got=decode(native);
        bits+=native!=ref[i];
        nan+=std::isnan(got); inf+=std::isinf(got);
        bool fail;
        if (!std::isfinite(expected)||!std::isfinite(got)) {
            fail=expected!=got;
            unexpected+=fail;
        } else {
            ++finite;
            const double error=std::abs(double(expected)-got);
            max_abs=std::max(max_abs,error);
            max_rel=std::max(max_rel,error/std::max(std::abs(double(expected)),1e-30));
            sum+=error; sum_sq+=error*error;
            ++ulps[std::abs(ordered(ref[i])-ordered(native))];
            fail=error>1e-5+1e-5*std::abs(expected);
        }
        if (fail) { ++fails; first=std::min(first,i); }
    }
    csv<<label<<','<<ref.size()<<','<<fails<<','<<bits<<','<<max_abs<<','
       <<sum/std::max<size_t>(1,finite)<<','<<max_rel<<','<<std::sqrt(sum_sq/std::max<size_t>(1,finite))
       <<','<<(ulps.empty()?0:ulps.rbegin()->first)<<','<<nan<<','<<inf<<','<<unexpected<<',';
    if (first!=ref.size()) csv<<first<<':'<<ref[first]<<':'<<word(actual[first]);
    csv<<',';
    for (const auto &[ulp,count]:ulps) csv<<ulp<<':'<<count<<' ';
    csv<<'\n';
    return fails==0;
}

template<typename T, typename Exception> void rejects(T call) {
    try {call();} catch (const Exception&) {return;}
    throw std::runtime_error("MLA cache operation was accepted unexpectedly");
}
template<typename T> void bad_argument(T call) {rejects<T,std::invalid_argument>(call);}
template<typename T> void bad_state(T call) {rejects<T,std::logic_error>(call);}

uint16_t *table(Memory& memory,const std::string &path,size_t count) {
    auto host=load<uint16_t>(path,count);
    auto *device=static_cast<uint16_t*>(memory.allocate(count*2,Kind::weight));
    upload(device,host);
    return device;
}
MlaWeights load_weights(Worker& worker,Registry& registry,Memory& memory,const std::string& folder) {
    const auto type=bf16?mlu::WeightType::bf16:mlu::WeightType::fp16;
    MlaWeights set{};
    const char *names[]={"q","kv_a","kv_b","o"};
    const int k[]={hd,hd,512,hd}, n[]={3072,576,4096,hd};
    for (size_t i=0;i<4;++i) {
        const WeightKey key{"layer0",names[i],"checkpoint-fixture","kn-row-major",
                            worker.pci(),0,type,k[i],n[i]};
        auto host=load<uint16_t>(folder+"/weight_"+names[i]+".bin",size_t(k[i])*n[i]);
        registry.host(key,host.data());
        set.projections[i]=registry.acquire(key);
    }
    set.latent_gamma=table(memory,folder+"/weight_gamma.bin",512);
    set.cosine=table(memory,folder+"/cos.bin",table_rows*64);
    set.sine=table(memory,folder+"/sin.bin",table_rows*64);
    set.table_rows=table_rows;
    std::ifstream scale(folder+"/scale.txt");
    require(bool(scale>>set.scale),"missing MLA scale");
    return set;
}

void poison_cache(MlaRequest& request) {
    const auto& t=request.tensors();
    for (auto [device,width]:{std::pair{t.key,kw},std::pair{t.value,vw}}) {
        std::vector<uint16_t> host(size_t(heads)*request.capacity()*width+guard_words,poison);
        upload(device,host);
    }
}

bool stages(MlaRequest& request,const std::string &folder,int cursor,int count,
            const std::string& id,std::ostream& accuracy,std::ostream& cache) {
    const auto& t=request.tensors();
    bool ok=true;
    auto stage=[&](const char *name,auto *data,size_t size) {
        const auto reference=load<uint16_t>(folder+"/"+name+".bin",size);
        ok &= compare(accuracy,id+','+name,reference,download(data,size));
    };
    stage("q_projection",t.q_projection,count*3072);
    stage("kv_a",t.kv_a,count*576);
    stage("latent",t.latent,count*512);
    stage("kv_b",t.kv_b,count*4096);
    stage("q",t.query,count*3072);
    stage("masked",t.masked,heads*count*request.length());
    stage("prob",t.prob,heads*count*request.length());
    stage("merged",t.merged,count*hd);
    stage("output",t.output,count*hd);
    const auto positions=download(t.positions,count);
    for (int j=0;j<count;++j) require(positions[j]==uint32_t(cursor+j),"wrong MLA position");
    for (auto [name,device,width]:{std::tuple{"k",t.key,kw},std::tuple{"v",t.value,vw}}) {
        const int cap=request.capacity(), len=request.length();
        const auto backing=download(device,size_t(heads)*cap*width+guard_words);
        const auto reference=load<uint16_t>(folder+"/"+name+".bin",size_t(heads)*len*width);
        std::vector<uint16_t> actual;
        actual.reserve(reference.size());
        for (int head=0;head<heads;++head) {
            auto base=backing.begin()+size_t(head)*cap*width;
            actual.insert(actual.end(),base,base+len*width);
            // Reuse keeps the old suffix, so poison is required only before reset.
            if (id.find(",reuse,")==std::string::npos)
                require(std::all_of(base+len*width,base+cap*width,
                                    [](uint16_t x){return x==poison;}),
                        "MLA cache write touched unused suffix");
        }
        require(std::all_of(backing.end()-guard_words,backing.end(),
                            [](uint16_t x){return x==poison;}),"MLA cache guard modified");
        ok &= compare(cache,id+','+name,reference,actual);
    }
    return ok;
}
}

namespace {
struct Profile {
    std::array<cnrtNotifier_t,5> markers{};
    Profile() {
        size_t created=0;
        try {
            for (auto &marker:markers) {
                check(cnrtNotifierCreate(&marker),"allocate MLA profiling marker");
                ++created;
            }
        } catch (...) {
            while (created) cnrtNotifierDestroy(markers[--created]);
            throw;
        }
    }
    ~Profile() { for (auto marker:markers) cnrtNotifierDestroy(marker); }
    float duration(size_t start,size_t end) const {
        float us=0;
        check(cnrtNotifierDuration(markers[start],markers[end],&us),"read MLA profiling marker");
        return us;
    }
    float read_prefix(Worker &worker, Memory &memory, size_t queue_id,
                      const MlaRequest &request) const {
        const auto queue=worker.queue(queue_id);
        const auto &t=request.tensors();
        const size_t len=request.length(), capacity=request.capacity();
        void *buffer=memory.allocate(heads*len*(kw+vw)*2,Kind::workspace);
        auto *dst=static_cast<uint16_t*>(buffer);
        try {
            check(cnrtPlaceNotifier(markers[0],queue),"cache read start");
            for (auto [source,width]:{std::pair{t.key,kw},std::pair{t.value,vw}}) {
                for (int head=0;head<heads;++head) {
                    check(cnrtMemcpyAsync(dst,source+head*capacity*width,len*width*2,queue,
                                         cnrtMemcpyDevToDev),"gather cache prefix");
                    dst+=len*width;
                }
            }
            check(cnrtPlaceNotifier(markers[1],queue),"cache read end");
            worker.sync(queue_id);
            const float time=duration(0,1);
            memory.release(buffer);
            return time;
        } catch (...) {
            worker.sync(queue_id);
            memory.release(buffer);
            throw;
        }
    }

};
}
int main(int argc,char **argv) try {
    if (argc!=8) return 2;
    const std::string root=argv[2], dtype=argv[3], kind=argv[6];
    bf16=dtype=="bf16";
    require(bf16||dtype=="fp16","unsupported dtype");
    const int requests=std::stoi(argv[5]), queue_count=std::stoi(kind);
    require(requests>=1&&requests<=2&&queue_count>=1&&queue_count<=requests,
            "invalid fixture configuration");
    const std::string lengths=argv[4];
    const auto separator=lengths.find(':');
    require(separator==std::string::npos||requests==2,"mixed lengths require two requests");
    std::array<int,2> prefill{std::stoi(lengths),std::stoi(lengths.substr(
        separator==std::string::npos?0:separator+1))};
    for (const int count:prefill)
        require(count==1||count==7||count==32||count==128,"invalid prefill length");
    const std::string weights_path=root+'/'+dtype, output=argv[7];
    Worker worker(argv[1]); Memory memory(worker); Registry registry(worker);
    auto weights=load_weights(worker,registry,memory,weights_path);
    std::vector<size_t> queues;
    for (int q=0;q<queue_count;++q) queues.push_back(worker.add_queue());
    const auto base_bytes=memory.usage().live;
    std::vector<std::unique_ptr<MlaRequest>> active;
    for (int i=0;i<requests;++i) {
        active.push_back(std::make_unique<MlaRequest>(worker,memory,queues[i%queue_count],weights,
                         prefill[i],prefill[i]+3,bf16,guard_words*sizeof(uint16_t)));
        bad_argument([&]{active.back()->enqueue(1,1);});
    }
    std::ofstream accuracy(output+".mla.csv"),cache(output+".cache.csv");
    require(accuracy.good()&&cache.good(),"cannot write MLA case output");
    const char *header="dtype,prefill,requests,queues,mode,request,cursor,count,stage,elements,mismatches,bit_diffs,max_abs,mean_abs,max_rel,rms,max_ulp,nan,inf,unexpected_nonfinite,first_mismatch,ulp_histogram\n";
    accuracy<<header; cache<<header;
    const bool benchmark=std::getenv("MLA_PROFILE")!=nullptr;
    std::vector<std::unique_ptr<Profile>> profiles(requests);
    std::ofstream performance;
    if (benchmark) {
        performance.open(output+".perf.csv");
        require(performance.good(),"cannot write MLA profiling output");
        performance<<"dtype,prefill,requests,queues,mode,request,cursor,count,sample,"
                      "project_rope_us,cache_append_us,cache_read_attention_us,output_us,device_us,"
                      "batch_wall_us,query_tokens_per_sec,cache_bytes,managed_bytes,weight_bytes,cache_gather_us\n";
        for (auto &profile:profiles) profile=std::make_unique<Profile>();
    }
    bool passed=true;
    for (int sample=0;sample<(benchmark?4:1);++sample) {
        for (auto &request:active) {
            request->reset();
            poison_cache(*request);
        }
        for (int step=0;step<5;++step) {
            const bool reuse=step==4;
            std::array<int,2> cursors{},counts{};
            int total_queries=0;
            for (int n=0;n<requests;++n) {
                cursors[n]=step==0||reuse?0:prefill[n]+step-1;
                counts[n]=cursors[n]?1:prefill[n];
                total_queries+=counts[n];
                if (reuse) {
                    bad_argument([&]{active[n]->enqueue(active[n]->length(),1);});
                    active[n]->reset();
                    require(active[n]->length()==0,"reset did not clear logical length");
                }
                const int fixture_request=reuse?1-n:n;
                const std::string fixture=weights_path+'/'+std::to_string(prefill[n])+'/'+
                    std::to_string(fixture_request)+'/'+std::to_string(cursors[n]);
                auto input=load<uint16_t>(fixture+"/input.bin",counts[n]*hd);
                upload(active[n]->tensors().input,input);
            }
            const auto start=std::chrono::steady_clock::now();
            for (int n=0;n<requests;++n) {
                auto &request=*active[n];
                request.enqueue(cursors[n],counts[n],benchmark?&profiles[n]->markers:nullptr);
                require(request.length()==cursors[n],"uncommitted cache length became visible");
                if (!benchmark) {
                    bad_state([&]{request.reset();});
                    bad_state([&]{request.enqueue(cursors[n],counts[n]);});
                }
            }
            for (auto &request:active) request->complete();
            const double wall=std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-start).count();
            for (int n=0;n<requests;++n) {
                auto &request=*active[n];
                const int cursor=cursors[n],count=counts[n];
                require(request.length()==cursor+count,"cache commit length mismatch");
                std::ostringstream id;
                id<<dtype<<','<<prefill[n]<<','<<requests<<','<<queue_count<<','
                  <<(reuse?"reuse":cursor?"decode":"prefill")<<','<<n<<','<<cursor<<','<<count;
                if (!benchmark || !sample) {
                    const int fixture_request=reuse?1-n:n;
                    const std::string parent=weights_path+'/'+std::to_string(prefill[n])+'/'+
                        std::to_string(fixture_request);
                    passed &= stages(request,parent+'/'+std::to_string(cursor),cursor,count,id.str(),accuracy,cache);
                    if (cursor) {
                        const auto reference=load<uint16_t>(parent+"/full_output.bin",size_t(prefill[n]+3)*hd);
                        const std::vector<uint16_t> token(reference.begin()+cursor*hd,
                                                           reference.begin()+(cursor+count)*hd);
                        passed &= compare(accuracy,id.str()+",full_continuation",token,
                                          download(request.tensors().output,count*hd));
                    }
                    std::cout<<"case "<<id.str()<<" cache_length="<<request.length()
                             <<" guard=PASS position=PASS\n";
                }
                if (benchmark && sample) {
                    const auto& p=*profiles[n];
                    performance<<id.str()<<','<<sample<<','<<p.duration(0,1)<<','<<p.duration(1,2)
                               <<','<<p.duration(2,3)<<','<<p.duration(3,4)<<','<<p.duration(0,4)
                               <<','<<wall<<','<<total_queries*1e6/wall
                               <<','<<memory.usage().by_kind[size_t(Kind::cache)]<<','
                               <<memory.usage().live<<','<<registry.bytes();
                    performance<<','<<p.read_prefix(worker,memory,queues[n%queue_count],request)<<'\n';
                }
            }
        }
    }
    profiles.clear();
    active.clear();
    require(memory.usage().live==base_bytes,"MLA workspace or cache leaked");
    for (auto ptr:{weights.latent_gamma,weights.cosine,weights.sine}) {
        void *buffer=const_cast<uint16_t*>(ptr);
        memory.release(buffer);
    }
    weights.projections={};
    registry.close(); memory.close(); worker.close();
    std::cout<<"cache lifecycle PASS, final managed bytes="<<memory.usage().live
             <<" peak managed bytes="<<memory.usage().peak<<" numerical="<<(passed?"PASS":"FAIL")<<'\n';
    return passed?0:1;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 2;
}
