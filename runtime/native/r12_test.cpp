#include "runtime.h"
#include <cstdio>
#include <stdexcept>

using namespace native_mlu;
static void require(bool ok) { if(!ok) throw std::runtime_error("R1/R2 assertion"); }
int main(int argc,char **argv) {
    if(argc!=3 || std::string(argv[1])!="--authorized") return 2;
    Worker worker(argv[2]);
    std::printf("pci=%s ordinal=%d name=%s memory_mb=%d\n",worker.pci().c_str(),
                worker.ordinal(),worker.properties().name,worker.properties().totalMem);
    const auto queue=worker.add_queue();
    Registry registry(worker);
    WeightKey key{"revision-a","fixture","shard1","kn-row-major",worker.pci(),0,
                  mlu::WeightType::bf16,1,1};
    uint16_t word=0x3f80;
    registry.host(key,&word);
    auto a=registry.acquire(key);
    require(a && a->data() && a->bytes()==32768);
    void *source=nullptr;
    check(cnrtMalloc(&source,2),"source allocation");
    check(cnrtMemcpy(source,&word,2,cnrtMemcpyHostToDev),"upload");
    auto next=key; next.revision="revision-b";
    registry.device(next,static_cast<uint16_t*>(source),queue);
    require(!registry.acquire(next));
    registry.publish(next);
    auto b=registry.acquire(next);
    require(b && b->data() != a->data());
    check(cnrtFree(source),"source free");
    registry.invalidate("revision-a");
    require(!registry.acquire(key));
    registry.collect();
    require(registry.bytes()==65536);
    a.reset(); registry.collect(); require(registry.bytes()==32768);
    registry.invalidate("revision-b"); b.reset(); registry.collect();
    require(registry.bytes()==0);
    registry.close();
    worker.close();
    std::puts("PASS R1/R2: PCI resolution, publication, revision reload, pinned invalidation, reclamation");
}
