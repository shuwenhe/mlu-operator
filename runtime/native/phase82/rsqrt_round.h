#ifndef PHASE82_RSQRT_ROUND_H
#define PHASE82_RSQRT_ROUND_H
#include <stdint.h>

// Positive normal checkpoint variance. Two IEEE binary32 RN-even stages.
__mlu_func__ float phase82_rsqrt_round(float value) {
    union Word { float f; uint32_t u; };
    Word val; val.f=value;
    const int exponent=int((val.u>>23)&255)-127;
    if (exponent < -126 || exponent > 126) return value;
    const int parity=(exponent % 2 + 2) % 2;
    int root_exp=(exponent-parity)/2;
    const uint64_t radicand=uint64_t((val.u&0x7fffffU)|0x800000U)<<(23+parity);
    uint32_t root=0x800000U;
    for(uint32_t bit=0x400000U;bit;bit>>=1){
        const uint32_t trial=root|bit;
        if(uint64_t(trial)*trial<=radicand)root=trial;
    }
    if(radicand-uint64_t(root)*root>root)++root;
    if(root==0x1000000U){root>>=1;++root_exp;}
    int inv_exp=-root_exp;
    uint32_t reciprocal=0x800000U;
    if(root!=0x800000U){
        --inv_exp;
        const uint64_t numerator=uint64_t(1)<<47;
        for(uint32_t bit=0x400000U;bit;bit>>=1){
            const uint32_t trial=reciprocal|bit;
            if(uint64_t(trial)*root<=numerator)reciprocal=trial;
        }
        const uint64_t excess=2*(numerator-uint64_t(reciprocal)*root);
        if(excess>root || (excess==root && (reciprocal&1)))++reciprocal;
        if(reciprocal==0x1000000U){reciprocal>>=1;++inv_exp;}
    }
    Word result;
    result.u=(uint32_t(inv_exp+127)<<23)|(reciprocal&0x7fffffU);
    return result.f;
}
#endif
