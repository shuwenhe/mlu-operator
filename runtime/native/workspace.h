#ifndef NATIVE_MLU_WORKSPACE_H
#define NATIVE_MLU_WORKSPACE_H
#include "runtime.h"
#include <array>

namespace native_mlu {
enum class Kind : size_t { weight, cache, workspace };
struct Usage {
    std::array<size_t,3> by_kind{};
    size_t reserve=0, live=0, peak=0;
};
class Memory {
    struct Block { size_t bytes; Kind kind; size_t pins=0; };
public:
    explicit Memory(Worker &worker,size_t reserve=0):worker_(worker) {usage_.reserve=reserve;}
    Memory(const Memory&)=delete;
    ~Memory();
    void *allocate(size_t bytes,Kind kind);
    void release(void *&pointer);
    void pin(const void *pointer);
    void unpin(const void *pointer);
    size_t capacity(const void *pointer) const;
    const Usage& usage() const {return usage_;}
    void close();
private:
    Worker &worker_;
    Usage usage_;
    std::map<const void*,Block> blocks_;
};

class Workspace {
public:
    struct Slot {
        size_t queue;
        void *activation=nullptr,*fp32=nullptr;
        bool active=false, pending=false;
        std::vector<Registry::Lease> weights;
        std::vector<const void*> buffers;
        std::vector<std::shared_ptr<void>> resources;
    };
    Workspace(Worker &worker,Memory &memory):worker_(worker),memory_(memory){}
    Workspace(const Workspace&)=delete;
    ~Workspace();
    size_t add(size_t queue,size_t low_bytes,size_t fp32_bytes);
    Slot& start(size_t id);
    void retain(size_t id,Registry::Lease lease);
    void retain_buffer(size_t id,const void *pointer);
    void retain_resource(size_t id,std::shared_ptr<void> resource);
    void pending(size_t id);
    void finish(size_t id);
    void close();
    const Slot& slot(size_t id) const {return slots_.at(id);}
private:
    Worker &worker_;
    Memory &memory_;
    std::vector<Slot> slots_;
};
}
#endif
