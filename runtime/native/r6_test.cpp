#include "dispatch.h"
#include "mlu_gemm_reference.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace native_mlu;
namespace {
constexpr int h=2048,qwidth=3072,kva=576,latent=512,kvb=4096,ffn=10944;
int capacity=3;
bool fp16=false;
Type dtype() {return fp16?Type::fp16:Type::bf16;}
mlu::WeightType weight_dtype(){return fp16?mlu::WeightType::fp16:mlu::WeightType::bf16;}
static void require(bool ok,const char *message){if(!ok) throw std::runtime_error(message);}
template<class T> auto load(const std::string& path,size_t count) {
    std::ifstream f(path,std::ios::binary);
    std::vector<T> result(count);
    require(bool(f.read(reinterpret_cast<char*>(result.data()),count*sizeof(T))) && f.peek()==EOF,
            "invalid fixture file");
    return result;
}
static float decode_bf16(uint16_t x) {
    uint32_t bits=uint32_t(x)<<16; float y;
    std::memcpy(&y,&bits,4);return y;
}
static float decode(uint16_t x) {
    if(!fp16) return decode_bf16(x);
    _Float16 value;
    std::memcpy(&value,&x,2);
    return static_cast<float>(value);
}
static uint16_t encode(float x) {
    if(fp16) {
        const _Float16 value=static_cast<_Float16>(x);
        uint16_t bits; std::memcpy(&bits,&value,2); return bits;
    }
    uint32_t bits;std::memcpy(&bits,&x,4);
    return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
static void normalize(const float *in,float *out,const std::vector<uint16_t>& gamma,int size) {
    float sum=0;
    for(int j=0;j<size;++j)sum+=in[j]*in[j];
    const float scale=1.f/std::sqrt(sum/size+1e-6f);
    for(int j=0;j<size;++j)out[j]=decode(encode(decode(encode(in[j]*scale))*decode(gamma[j])));
}
static View make(Memory& mem,int rows,int cols,Type type,Kind kind=Kind::workspace) {
    return {mem.allocate(size_t(rows)*cols*(type==Type::fp32||type==Type::u32?4:2),kind),rows,cols,cols,type};
}
static View rows(View view,int count){view.rows=count;return view;}
static void upload(View v,const uint16_t *data) {
    check(cnrtMemcpy(v.data,const_cast<uint16_t*>(data),size_t(v.rows)*v.cols*2,cnrtMemcpyHostToDev),"upload BF16");
}
struct Weight {
    WeightKey key;
    std::vector<uint16_t> logical;
    std::vector<float> reference;
};
struct Request {
    size_t slot=0;
    int length=0;
    View input{},norm{},query{},split{},compressed{},normalized{},expanded{},cache{};
    std::vector<float> expected_cache, expected_pe, actual_pe;
};
static void trace(const char *stage,int request,int position,const std::vector<float>& data) {
    const auto* path=std::getenv("LAYER0_TRACE_DIR");
    if(!path) return;
    std::ofstream output(std::string(path)+"/"+std::to_string(request)+"_"+
                         std::to_string(position)+"_"+stage+".bin",std::ios::binary);
    require(bool(output.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float))),
            "write stage trace failed");
}
static std::vector<float> rounded(std::vector<float> values) {
    for(auto& v:values)v=decode(encode(v));
    return values;
}
static bool compare(const char *stage,int request,int position,const float *ref,const float *actual,size_t n) {
    double max_abs=0,mean=0;size_t bad=0;
    for(size_t j=0;j<n;++j){
        const double error=std::abs(double(actual[j])-ref[j]);
        max_abs=std::max(max_abs,error);mean+=error;
        bad+=!std::isfinite(actual[j])||error>1e-5+1e-5*std::abs(ref[j]);
    }
    std::printf("%s,%d,%d,%zu,%.9g,%.9g,%zu\n",stage,request,position,n,max_abs,mean/n,bad);
    std::fflush(stdout);
    return bad==0;
}
// Positions 0..2 from the checkpoint's YaRN configuration. The two mscale
// factors are equal, so the rotary table multiplier is one.
static void fill_rope_tables(std::vector<uint16_t>& cos,std::vector<uint16_t>& sin) {
    const double pi=std::acos(-1.0), base=10000.0;
    const auto correction=[&](double rotations){
        return 64*std::log(4096/(rotations*2*pi))/(2*std::log(base));
    };
    const float low=std::max(0.,std::floor(correction(32)));
    const float high=std::min(63.,std::ceil(correction(1)));
    for(int i=0;i<32;++i) {
        const float divisor=std::pow(10000.f,float(2*i)/64);
        const float ramp=std::clamp((i-low)/(high-low),0.f,1.f);
        const float freq=(1-ramp)/divisor+ramp/(40.f*divisor);
        for(int pos=0;pos<capacity;++pos) {
            cos[pos*64+i]=cos[pos*64+32+i]=encode(std::cos(pos*freq));
            sin[pos*64+i]=sin[pos*64+32+i]=encode(std::sin(pos*freq));
        }
    }
}
static void rotate_cpu(const std::vector<uint16_t>& in,const std::vector<uint16_t>& cos,
                       const std::vector<uint16_t>& sin,std::vector<uint16_t>& out,int seq) {
    for(int head=0;head<17;++head)for(int pos=0;pos<seq;++pos)for(int lane=0;lane<64;++lane) {
        const size_t row=size_t(head*seq+pos)*64;
        const int i=lane%32;
        const float a=decode(in[row+2*i+(lane>=32)]);
        const float b=decode(in[row+2*i+(lane<32)])*(lane<32?-1.f:1.f);
        out[row+lane]=encode(decode(encode(a*decode(cos[pos*64+lane])))+
                             decode(encode(b*decode(sin[pos*64+lane]))));
    }
}
static std::vector<float> reference_mla(const std::vector<float>& q,const std::vector<float>& kv,
                                       const std::vector<uint16_t>& rotated,int cursor,int batch) {
    const float scale=std::pow(192.f,-0.5f)*std::pow(1.f+0.1f*0.707f*std::log(40.f),2.f);
    const int seq=cursor+batch;
    std::vector<float> result(size_t(batch)*h);
    for(int row=0;row<batch;++row)for(int head=0;head<16;++head) {
        std::vector<float> score(capacity);
        float maximum=-INFINITY;
        const int pos=cursor+row;
        for(int token=0;token<=pos;++token) {
            float sum=0;
            for(int j=0;j<128;++j)
                sum+=decode(encode(q[size_t(row)*qwidth+head*192+j]))*
                     decode(encode(kv[size_t(token)*kvb+head*256+j]));
            for(int j=0;j<64;++j)
                sum+=decode(rotated[size_t(head*seq+pos)*64+j])*
                     decode(rotated[size_t(16*seq+token)*64+j]);
            score[token]=decode(encode(decode(encode(sum))*scale));
            maximum=std::max(maximum,score[token]);
        }
        float sum=0;
        for(int token=0;token<=pos;++token) {
            score[token]=std::exp(score[token]-maximum);
            sum+=score[token];
        }
        for(int j=0;j<128;++j) {
            float value=0;
            for(int token=0;token<=pos;++token)
                value+=decode(encode(score[token]/sum))*
                       decode(encode(kv[size_t(token)*kvb+head*256+128+j]));
            result[size_t(row)*h+head*128+j]=decode(encode(value));
        }
    }
    return result;
}

}
int main(int argc,char **argv) try {
    if((argc!=4 && argc!=6) || std::string(argv[1])!="--authorized-reference-mla")return 2;
    int prefill=2;
    if(argc==6) {
        const std::string requested=argv[4];
        if(requested!="fp16" && requested!="bf16")return 2;
        fp16=requested=="fp16";
        prefill=std::stoi(argv[5]);
        if(prefill!=1 && prefill!=7 && prefill!=32 && prefill!=128)return 2;
    }
    capacity=prefill+1;
    std::fprintf(stderr,"case_dtype=%s prefill=%d decode=1\n",fp16?"fp16":"bf16",prefill);
    const std::string dir=argv[3];
    Worker worker(argv[2]);Registry registry(worker);Memory mem(worker,4096);Workspace ws(worker,mem);
    Dispatch dispatch(worker,mem,ws);
    std::ifstream manifest(dir+"/manifest");
    std::string revision;require(bool(manifest>>revision),"missing manifest identity");
    std::map<std::string,Weight> weights;
    for(std::string alias,name,shard;manifest>>alias>>name>>shard;) {
        uint64_t offset;require(bool(manifest>>offset),"invalid checkpoint offset");
        const std::map<std::string,std::pair<int,int>> shapes{
            {"q",{h,qwidth}},{"kv_a",{h,kva}},{"kv_b",{latent,kvb}},{"o",{h,h}},
            {"gate",{h,ffn}},{"up",{h,ffn}},{"down",{ffn,h}}};
        auto found=shapes.find(alias);
        if(found==shapes.end())continue;
        const auto [k,n]=found->second;
        const auto raw=load<uint16_t>(dir+"/"+alias+".bin",size_t(k)*n);
        Weight entry{{revision,name,shard,"kn-row-major",worker.pci(),offset,weight_dtype(),k,n},
                     std::vector<uint16_t>(raw.size()),std::vector<float>(raw.size())};
        for(int j=0;j<n;++j)for(int p=0;p<k;++p){
            const size_t i=size_t(p)*n+j;
            const auto word=raw[size_t(j)*k+p];
            entry.logical[i]=fp16?encode(decode_bf16(word)):word;
            entry.reference[i]=decode(entry.logical[i]);
        }
        registry.host(entry.key,entry.logical.data());
        weights.emplace(alias,std::move(entry));
    }
    auto& q=weights.at("q");auto& ka=weights.at("kv_a");auto& kb=weights.at("kv_b");
    auto ig=load<uint16_t>(dir+"/input_gamma.bin",h),lg=load<uint16_t>(dir+"/gamma.bin",latent);
    auto pg=load<uint16_t>(dir+"/post_gamma.bin",h);
    for(auto *gamma:{&ig,&lg,&pg}) if(fp16) for(auto& v:*gamma)v=encode(decode_bf16(v));
    auto d_pg=make(mem,1,h,dtype(),Kind::weight);
    upload(d_pg,pg.data());
    auto d_ig=make(mem,1,h,dtype(),Kind::weight),d_lg=make(mem,1,latent,dtype(),Kind::weight);
    upload(d_ig,ig.data());upload(d_lg,lg.data());
    std::array<std::vector<uint16_t>,2> embeddings{std::vector<uint16_t>(capacity*h),std::vector<uint16_t>(capacity*h)};
    std::ifstream token_file(dir+"/tokens.bin",std::ios::binary);
    const uint32_t ids[][3]={{100000,42,100},{777,512,16}};
    for(int i=0;i<2;++i) {
        uint16_t original[3][h];
        for(int j=0;j<3;++j) {
            uint32_t token;
            require(bool(token_file.read(reinterpret_cast<char*>(&token),4)) && token==ids[i][j],"unexpected token");
            require(bool(token_file.read(reinterpret_cast<char*>(original[j]),h*2)),"missing embedding row");
        }
        for(int j=0;j<capacity;++j)for(int col=0;col<h;++col) {
            const auto word=original[j%3][col];
            embeddings[i][j*h+col]=fp16?encode(decode_bf16(word)):word;
        }
    }
    std::vector<uint16_t> cos(capacity*64),sin(capacity*64);
    fill_rope_tables(cos,sin);
    auto d_cos=make(mem,capacity,64,dtype()),d_sin=make(mem,capacity,64,dtype()),
         d_rot_in=make(mem,17*capacity,64,dtype()),d_rot_out=make(mem,17*capacity,64,dtype());
    auto d_pos=make(mem,1,capacity,Type::u32);
    upload(d_cos,cos.data());upload(d_sin,sin.data());
    std::vector<uint32_t> pos(capacity);
    for(int i=0;i<capacity;++i)pos[i]=i;
    check(cnrtMemcpy(d_pos.data,pos.data(),pos.size()*4,cnrtMemcpyHostToDev),"upload positions");
    Request requests[2];
    for(auto& r:requests){
        r.slot=ws.add(worker.add_queue(),h*2,qwidth*4);
        r.input=make(mem,prefill,h,dtype());r.norm=make(mem,prefill,h,dtype());
        r.query=make(mem,prefill,qwidth,Type::fp32);r.split=make(mem,prefill,kva,Type::fp32);
        r.compressed=make(mem,prefill,latent,dtype());r.normalized=make(mem,prefill,latent,dtype());
        r.expanded=make(mem,prefill,kvb,Type::fp32);r.cache=make(mem,capacity,kvb,Type::fp32,Kind::cache);
        r.expected_cache.resize(size_t(capacity)*kvb);
        r.expected_pe.resize(capacity*64); r.actual_pe.resize(capacity*64);
    }

    bool passed=true;
    std::puts("stage,request,position,elements,max_abs,mean_abs,mismatches");
    for(int repeat=0;repeat<2;++repeat){
        for(auto& r:requests) r.length=0;
        for(int batch:{prefill,1}) {
            // Submit two independent requests before either queue is synchronized.
            for(int i=0;i<2;++i) {
                auto& r=requests[i];
                ws.start(r.slot);
                upload(rows(r.input,batch),embeddings[i].data()+r.length*h);
                dispatch.rmsnorm(r.slot,Backend::Native,rows(r.input,batch),d_ig,rows(r.norm,batch),1e-6f);
                dispatch.gemm(r.slot,Backend::Native,rows(r.norm,batch),registry.acquire(q.key),{},rows(r.query,batch));
                dispatch.kv_chain(r.slot,rows(r.norm,batch),registry.acquire(ka.key),d_lg,registry.acquire(kb.key),
                                  rows(r.split,batch),rows(r.compressed,batch),rows(r.normalized,batch),rows(r.expanded,batch));
            }
            for(int i=0;i<2;++i) {
                auto& r=requests[i];ws.finish(r.slot);
                std::vector<float> xf(size_t(batch)*h),norm(xf.size()),qr(size_t(batch)*qwidth),
                                   compressed(size_t(batch)*kva),nr(size_t(batch)*latent),kv(size_t(batch)*kvb);
                std::vector<float> actual_q(qr.size()),actual_cache(size_t(r.length+batch)*kvb);
                for(size_t j=0;j<xf.size();++j)xf[j]=decode(embeddings[i][r.length*h+j]);
                for(int j=0;j<batch;++j)normalize(xf.data()+j*h,norm.data()+j*h,ig,h);
                mlu_gemm_reference(norm.data(),q.reference.data(),qr.data(),batch,qwidth,h,h,qwidth,qwidth);
                mlu_gemm_reference(norm.data(),ka.reference.data(),compressed.data(),batch,kva,h,h,kva,kva);
                for(int row=0;row<batch;++row) {
                    float compact[latent];
                    for(int col=0;col<latent;++col)
                        compact[col]=decode(encode(compressed[size_t(row)*kva+col]));
                    normalize(compact,nr.data()+row*latent,lg,latent);
                }
                mlu_gemm_reference(nr.data(),kb.reference.data(),kv.data(),batch,kvb,latent,latent,kvb,kvb);
                check(cnrtMemcpy(actual_q.data(),r.query.data,actual_q.size()*4,cnrtMemcpyDevToHost),"copy native Q");
                passed&=compare("q",i,r.length,qr.data(),actual_q.data(),qr.size());
                require(r.length+batch<=capacity,"cache capacity exceeded");
                check(cnrtMemcpy(static_cast<float*>(r.cache.data)+size_t(r.length)*kvb,
                                 r.expanded.data,kv.size()*4,cnrtMemcpyDevToDev),"append projected KV");
                std::copy(kv.begin(),kv.end(),r.expected_cache.begin()+size_t(r.length)*kvb);
                check(cnrtMemcpy(actual_cache.data(),r.cache.data,actual_cache.size()*4,cnrtMemcpyDevToHost),"read projected KV cache");
                passed&=compare("projected_kv_cache",i,r.length,r.expected_cache.data(),actual_cache.data(),actual_cache.size());
                std::vector<float> actual_split(compressed.size());
                check(cnrtMemcpy(actual_split.data(),r.split.data,actual_split.size()*4,cnrtMemcpyDevToHost),"read KV-A rotary components");
                for(int row=0;row<batch;++row) {
                    std::copy_n(compressed.data()+row*kva+latent,64,r.expected_pe.data()+(r.length+row)*64);
                    std::copy_n(actual_split.data()+row*kva+latent,64,r.actual_pe.data()+(r.length+row)*64);
                }
                const int seq=r.length+batch;
                auto pack_rotary=[&](const std::vector<float>& query,const std::vector<float>& keys) {
                    std::vector<uint16_t> packed(17*seq*64);
                    for(int head=0;head<16;++head)for(int row=0;row<batch;++row)
                        for(int j=0;j<64;++j)
                            packed[(head*seq+r.length+row)*64+j]=encode(query[size_t(row)*qwidth+head*192+128+j]);
                    for(int j=0;j<seq*64;++j)packed[16*seq*64+j]=encode(keys[j]);
                    return packed;
                };
                auto gold_input=pack_rotary(qr,r.expected_pe), actual_input=pack_rotary(actual_q,r.actual_pe);
                std::vector<uint16_t> gold_rot(gold_input.size()),actual_rot(gold_input.size());
                rotate_cpu(gold_input,cos,sin,gold_rot,seq);
                upload(rows(d_rot_in,17*seq),actual_input.data());
                auto positions=d_pos; positions.cols=positions.stride=seq;
                ws.start(r.slot);
                dispatch.rope(r.slot,Backend::Native,rows(d_rot_in,17*seq),d_cos,d_sin,positions,
                              rows(d_rot_out,17*seq),seq);
                ws.finish(r.slot);
                check(cnrtMemcpy(actual_rot.data(),d_rot_out.data,actual_rot.size()*2,cnrtMemcpyDevToHost),"read native rotary output");
                const auto ref_attention=reference_mla(qr,r.expected_cache,gold_rot,r.length,batch);
                const auto out_attention=reference_mla(actual_q,actual_cache,actual_rot,r.length,batch);
                passed&=compare("cpu_reference_mla",i,r.length,ref_attention.data(),out_attention.data(),ref_attention.size());
                const auto run_linear=[&](const char* name,const std::vector<float>& host,
                                          const std::vector<float>& gold) {
                    const auto& w=weights.at(name);
                    const int k=w.key.k,n=w.key.n;
                    std::vector<float> ref(size_t(batch)*n), actual(ref.size());
                    mlu_gemm_reference(gold.data(),w.reference.data(),ref.data(),batch,n,k,k,n,n);
                    std::vector<uint16_t> words(host.size());
                    std::transform(host.begin(),host.end(),words.begin(),encode);
                    auto in=make(mem,batch,k,dtype()),out=make(mem,batch,n,Type::fp32);
                    upload(in,words.data());
                    ws.start(r.slot);
                    dispatch.gemm(r.slot,Backend::Native,in,registry.acquire(w.key),{},out);
                    ws.finish(r.slot);
                    check(cnrtMemcpy(actual.data(),out.data,actual.size()*4,cnrtMemcpyDevToHost),"read decoder projection");
                    passed&=compare(name,i,r.length,ref.data(),actual.data(),ref.size());
                    mem.release(out.data);mem.release(in.data);
                    return std::make_pair(std::move(actual),std::move(ref));
                };
                trace("original_hidden",i,r.length,xf);
                trace("input_norm",i,r.length,[&]{
                    std::vector<uint16_t> words(size_t(batch)*h);
                    check(cnrtMemcpy(words.data(),r.norm.data,words.size()*2,cnrtMemcpyDevToHost),"trace norm");
                    std::vector<float> values(words.size());
                    std::transform(words.begin(),words.end(),values.begin(),decode);
                    return values;
                }());
                trace("q",i,r.length,rounded(actual_q));
                trace("kv_a",i,r.length,rounded(actual_split));
                trace("kv_b",i,r.length,rounded(std::vector<float>(actual_cache.end()-kv.size(),actual_cache.end())));
                trace("attention",i,r.length,out_attention);
                auto projected=run_linear("o",out_attention,ref_attention);
                trace("output_projection_fp32",i,r.length,projected.first);
                trace("output_projection",i,r.length,rounded(projected.first));
                trace("serial_output_projection_fp32",i,r.length,projected.second);
                std::vector<float> residual_fp32(xf.size());
                for(size_t j=0;j<xf.size();++j)residual_fp32[j]=xf[j]+decode(encode(projected.first[j]));
                trace("residual_fp32",i,r.length,residual_fp32);
                for(size_t j=0;j<xf.size();++j) {
                    projected.first[j]=decode(encode(xf[j]+decode(encode(projected.first[j]))));
                    projected.second[j]=decode(encode(xf[j]+decode(encode(projected.second[j]))));
                }
                trace("residual",i,r.length,projected.first);
                passed&=compare("attention_residual",i,r.length,projected.second.data(),projected.first.data(),xf.size());
                std::vector<uint16_t> residual_words(xf.size());
                std::transform(projected.first.begin(),projected.first.end(),residual_words.begin(),encode);
                std::vector<float> gold_norm(xf.size());
                for(int j=0;j<batch;++j)normalize(projected.second.data()+j*h,gold_norm.data()+j*h,pg,h);
                upload(rows(r.input,batch),residual_words.data());
                ws.start(r.slot);
                dispatch.rmsnorm(r.slot,Backend::Native,rows(r.input,batch),d_pg,rows(r.norm,batch),1e-6f);
                ws.finish(r.slot);
                check(cnrtMemcpy(residual_words.data(),r.norm.data,residual_words.size()*2,cnrtMemcpyDevToHost),"read post norm");
                std::vector<float> native_norm(residual_words.size());
                std::transform(residual_words.begin(),residual_words.end(),native_norm.begin(),decode);
                std::vector<float> matched_input_norm(xf.size());
                for(int j=0;j<batch;++j)normalize(projected.first.data()+j*h,matched_input_norm.data()+j*h,pg,h);
                passed&=compare("post_norm_matched_input",i,r.length,matched_input_norm.data(),native_norm.data(),gold_norm.size());
                passed&=compare("post_attention_norm",i,r.length,gold_norm.data(),native_norm.data(),gold_norm.size());
                trace("post_norm",i,r.length,native_norm);
                auto gate=run_linear("gate",native_norm,gold_norm),up=run_linear("up",native_norm,gold_norm);
                trace("gate",i,r.length,rounded(gate.first));
                trace("up",i,r.length,rounded(up.first));
                const auto swiglu=[](std::vector<float>& g,const std::vector<float>& u) {
                    for(size_t j=0;j<g.size();++j) {
                        const float x=decode(encode(g[j]));
                        const float activation=decode(encode(x/(1.f+std::exp(-x))));
                        g[j]=decode(encode(activation*decode(encode(u[j]))));
                    }
                };
                swiglu(gate.first,up.first); swiglu(gate.second,up.second);
                trace("swiglu",i,r.length,gate.first);
                auto down=run_linear("down",gate.first,gate.second);
                trace("down",i,r.length,rounded(down.first));
                for(size_t j=0;j<xf.size();++j) {
                    down.first[j]=decode(encode(projected.first[j]+decode(encode(down.first[j]))));
                    down.second[j]=decode(encode(projected.second[j]+decode(encode(down.second[j]))));
                }
                trace("final",i,r.length,down.first);
                passed&=compare("decoder_layer_output",i,r.length,down.second.data(),down.first.data(),down.first.size());
                r.length+=batch;
            }
        }
    }

    const size_t packed_bytes=registry.bytes();
    const auto peak=mem.usage().peak,cache_bytes=mem.usage().by_kind[size_t(Kind::cache)];
    require(requests[0].cache.data!=requests[1].cache.data,"request caches aliased");
    for(auto& r:requests)
        for(auto v:{r.input,r.norm,r.query,r.split,r.compressed,r.normalized,r.expanded,r.cache}) mem.release(v.data);
    mem.release(d_ig.data);mem.release(d_lg.data);mem.release(d_pg.data);
    for(auto v:{d_cos,d_sin,d_rot_in,d_rot_out,d_pos})mem.release(v.data);
    ws.close();registry.close();mem.close();worker.close();
    std::fprintf(stderr,"packed_weight_bytes=%zu cache_bytes=%zu peak_managed_bytes=%zu remaining=%zu\n",
                 packed_bytes,cache_bytes,peak,mem.usage().live+registry.bytes());
    std::fprintf(stderr,"%s R6 layer parity; reclamation completed\n",passed?"PASS":"FAIL");
    return passed?0:1;
} catch(const std::exception& error) {
    std::fprintf(stderr,"R6 failure: %s\n",error.what());
    return 1;
}
