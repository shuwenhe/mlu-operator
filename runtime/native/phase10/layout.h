#ifndef PHASE10_MLA_LAYOUT_H
#define PHASE10_MLA_LAYOUT_H
#include <cnrt.h>
#include <cstdint>

extern "C" {
cnrtRet_t phase10_prepare_fp16(cnrtQueue_t,const float*,const float*,uint16_t*,uint32_t*,int,int);
cnrtRet_t phase10_prepare_bf16(cnrtQueue_t,const float*,const float*,uint16_t*,uint32_t*,int,int);
cnrtRet_t phase10_append_fp16(cnrtQueue_t,const float*,const float*,const uint16_t*,
                            uint16_t*,uint16_t*,uint16_t*,int,int,int);
cnrtRet_t phase10_append_bf16(cnrtQueue_t,const float*,const float*,const uint16_t*,
                            uint16_t*,uint16_t*,uint16_t*,int,int,int);
cnrtRet_t phase10_cast_fp16(cnrtQueue_t,const float*,uint16_t*,int);
cnrtRet_t phase10_cast_bf16(cnrtQueue_t,const float*,uint16_t*,int);
cnrtRet_t phase10_merge(cnrtQueue_t,const uint16_t*,uint16_t*,int);
}
#endif
