#include "workspace.h"
#include <cstdio>

using namespace native_mlu;
static int fail_alloc_after=-1;
static bool fail_sync=false;
extern "C" cnrtRet_t __real_cnrtMalloc(void**,size_t);
extern "C" cnrtRet_t __real_cnrtQueueSync(cnrtQueue_t);
extern "C" cnrtRet_t __wrap_cnrtMalloc(void **p,size_t bytes) {
    if(fail_alloc_after>=0 && fail_alloc_after--==0) return cnrtErrorNoMem;
    return __real_cnrtMalloc(p,bytes);
}
extern "C" cnrtRet_t __wrap_cnrtQueueSync(cnrtQueue_t q) {
    if(fail_sync) {fail_sync=false;return cnrtErrorQueue;}
    return __real_cnrtQueueSync(q);
}
static void require(bool condition) {if(!condition) throw std::runtime_error("R3/R4 assertion");}
int main(int argc,char **argv) {
    if(argc!=3 || std::string(argv[1])!="--authorized") return 2;
    Worker worker(argv[2]); Memory mem(worker,4096); Workspace ws(worker,mem);
    const auto q0=worker.add_queue(), q1=worker.add_queue();
    fail_alloc_after=1;
    try {ws.add(q0,32,64); require(false);} catch(const Error &e) {require(e.code==cnrtErrorNoMem);}
    require(mem.usage().live==0);
    const auto a=ws.add(q0,32,64), b=ws.add(q1,32,64);
    auto& aa=ws.start(a);auto& bb=ws.start(b);
    require(aa.activation!=bb.activation && aa.fp32!=bb.fp32);
    try {ws.start(a);require(false);} catch(const std::logic_error&) {}
    ws.pending(a);ws.pending(b);
    fail_sync=true;
    try {ws.finish(a);require(false);} catch(const Error &e) {require(e.code==cnrtErrorQueue);}
    require(ws.slot(a).pending && worker.fault()==cnrtErrorQueue);
    worker.recover();ws.finish(a);ws.finish(b);
    auto *cache=mem.allocate(64,Kind::cache);
    auto *weight=mem.allocate(32,Kind::weight);
    require(mem.usage().by_kind[size_t(Kind::cache)]==64);
    require(mem.usage().by_kind[size_t(Kind::weight)]==32);
    size_t free=0,total=0;check(cnrtMemGetInfo(&free,&total),"runtime memory");
    std::printf("peak_managed=%zu reserve_floor=%zu driver_free=%zu driver_total=%zu\n",
      mem.usage().peak,mem.usage().reserve,free,total);
    mem.release(cache); mem.release(weight);ws.close();mem.close();worker.close();
    require(mem.usage().live==0);
    std::puts("PASS R3/R4: two queues/slots, allocation rollback, injected sync fault/recovery and zero live bytes");
}
