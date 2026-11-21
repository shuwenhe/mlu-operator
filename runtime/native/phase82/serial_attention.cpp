#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

static float round_bf16(float x) {
    uint32_t bits; std::memcpy(&bits,&x,4);
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000U;
    std::memcpy(&x,&bits,4);
    return x;
}

extern "C" void serial_attention(const float *q,const float *k,const float *v,int m,int n,int pos,
    float *qk,float *scaled,float *masked,float *maxes,float *denom,float *prob,float *prob_low,
    float *av,float *out) {
    const float scale=std::pow(192.f,-0.5f)*std::pow(1.f+0.1f*0.707f*std::log(40.f),2.f);
    for(int h=0;h<16;++h)for(int i=0;i<m;++i){
        const size_t row=size_t(h)*m+i;
        const size_t start=row*n; float maximum=-INFINITY;
        for(int j=0;j<n;++j) {
            float x=0;
            for(int p=0;p<192;++p)x+=q[row*192+p]*k[(size_t(h)*n+j)*192+p];
            qk[start+j]=x;scaled[start+j]=round_bf16(round_bf16(x)*scale);
            masked[start+j]=j<=pos+i?scaled[start+j]:-INFINITY;
            maximum=std::max(maximum,masked[start+j]);
        }
        maxes[row]=maximum;float sum=0;
        for(int j=0;j<n;++j){prob[start+j]=std::exp(masked[start+j]-maximum);sum+=prob[start+j];}
        denom[row]=sum;
        for(int j=0;j<n;++j){prob[start+j]/=sum;prob_low[start+j]=round_bf16(prob[start+j]);}
        for(int p=0;p<128;++p){
            float x=0;
            for(int j=0;j<n;++j)x+=prob_low[start+j]*v[(size_t(h)*n+j)*128+p];
            av[row*128+p]=x;out[row*128+p]=round_bf16(x);
        }
    }
}
