#ifndef NATIVE_MLU_DISPATCH_H
#define NATIVE_MLU_DISPATCH_H
#include "workspace.h"

namespace native_mlu {
enum class Backend { Native, CNNL };
enum class Type { fp16,bf16,fp32,u32 };
struct View {void *data; int rows,cols,stride; Type type;};
class Dispatch {
public:
    Dispatch(Worker &worker, Memory &memory, Workspace &slots)
        :worker_(worker),memory_(memory),slots_(slots){}
    void gemm(size_t slot,Backend,View a,Registry::Lease weight,View logical_rhs,View output);
    void rmsnorm(size_t slot,Backend,View a,View gamma,View output,float epsilon);
    void rope(size_t slot,Backend,View qk,View cosine,View sine,View positions,View output,int sequence);
    void kv_chain(size_t slot,View input,Registry::Lease kvaa,View gamma,
                  Registry::Lease kvb,View wide,View latent,View norm,View output);
private:
    void view(View) const;
    void active(size_t);
    Worker &worker_;
    Memory &memory_;
    Workspace &slots_;
};
}
#endif
