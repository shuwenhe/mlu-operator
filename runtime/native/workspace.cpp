#include "workspace.h"
#include <algorithm>
#include <cstdio>
#include <limits>

namespace native_mlu {
Memory::~Memory() {
    if(blocks_.empty()) return;
    try {
        worker_.recover();
        while(!blocks_.empty()) {
            void *p=const_cast<void*>(blocks_.begin()->first);
            release(p);
        }
    } catch(const std::exception& e) {
        std::fprintf(stderr,"memory cleanup withheld: %s\n",e.what());
    }
}
void *Memory::allocate(size_t bytes,Kind kind) {
    if(!bytes || size_t(kind)>=usage_.by_kind.size() || bytes>std::numeric_limits<size_t>::max()-usage_.live)
        throw std::invalid_argument("invalid buffer allocation");
    worker_.select();
    check(worker_.fault(),"worker allocation blocked by queue fault");
    if(usage_.reserve) {
        size_t free=0,total=0;
        check(cnrtMemGetInfo(&free,&total),"read free memory");
        if(free<usage_.reserve || bytes>free-usage_.reserve) throw Error(cnrtErrorNoMem,"reserved memory floor");
    }
    void *p=nullptr;
    check(cnrtMalloc(&p,bytes),"allocate managed device buffer");
    try { blocks_.emplace(p,Block{bytes,kind}); }
    catch(...) { check(cnrtFree(p),"rollback allocation"); throw; }
    usage_.by_kind.at(size_t(kind))+=bytes;
    usage_.live+=bytes;
    usage_.peak=std::max(usage_.peak,usage_.live);
    return p;
}
void Memory::release(void *&p) {
    if(!p) return;
    worker_.select();
    check(worker_.fault(),"memory reclamation requires queue recovery");
    const auto it=blocks_.find(p);
    if(it==blocks_.end()) throw std::invalid_argument("buffer not owned by memory manager");
    if(it->second.pins) throw std::logic_error("buffer is referenced by a request slot");
    check(cnrtFree(p),"free managed device buffer");
    usage_.live-=it->second.bytes;
    usage_.by_kind.at(size_t(it->second.kind))-=it->second.bytes;
    blocks_.erase(it);
    p=nullptr;
}
void Memory::pin(const void *p) { ++blocks_.at(p).pins; }
void Memory::unpin(const void *p) { --blocks_.at(p).pins; }
size_t Memory::capacity(const void *p) const {
    const auto it=blocks_.find(p);
    return it==blocks_.end()?0:it->second.bytes;
}
void Memory::close() {
    if(!blocks_.empty()) throw std::logic_error("managed buffers remain live");
}
Workspace::~Workspace() {
    if(slots_.empty()) return;
    try { close(); } catch(const std::exception& e) {
        std::fprintf(stderr,"unsafe workspace teardown: %s\n",e.what());
        std::terminate();
    }
}
size_t Workspace::add(size_t queue,size_t low_bytes,size_t fp32_bytes) {
    worker_.queue(queue);
    slots_.reserve(slots_.size()+1);
    Slot slot{queue,nullptr,nullptr,false,false,{},{},{}};
    slot.activation=memory_.allocate(low_bytes,Kind::workspace);
    try { slot.fp32=memory_.allocate(fp32_bytes,Kind::workspace); }
    catch(...) { memory_.release(slot.activation); throw; }
    slots_.push_back(std::move(slot));
    return slots_.size()-1;
}
Workspace::Slot& Workspace::start(size_t id) {
    auto& slot=slots_.at(id);
    worker_.queue(slot.queue);
    if(slot.active || slot.pending) throw std::logic_error("request slot is in use");
    slot.active=true;
    return slot;
}
void Workspace::retain(size_t id,Registry::Lease lease) {
    auto& slot=slots_.at(id);
    if(!slot.active || !lease) throw std::invalid_argument("invalid request weight retention");
    slot.weights.push_back(std::move(lease));
}
void Workspace::retain_buffer(size_t id,const void *p) {
    auto& slot=slots_.at(id);
    if(!slot.active || !memory_.capacity(p)) throw std::invalid_argument("unowned request buffer");
    if(std::find(slot.buffers.begin(),slot.buffers.end(),p)!=slot.buffers.end()) return;
    slot.buffers.push_back(p);
    memory_.pin(p);
}
void Workspace::retain_resource(size_t id,std::shared_ptr<void> resource) {
    auto& slot=slots_.at(id);
    if(!slot.active || !resource) throw std::invalid_argument("invalid request resource");
    slot.resources.push_back(std::move(resource));
}
void Workspace::pending(size_t id) {
    auto& slot=slots_.at(id);
    if(!slot.active) throw std::logic_error("inactive request slot");
    slot.pending=true;
}
void Workspace::finish(size_t id) {
    auto& slot=slots_.at(id);
    if(!slot.active) throw std::logic_error("inactive request slot");
    if(slot.pending) worker_.sync(slot.queue);
    slot.pending=false;
    slot.active=false;
    slot.weights.clear();
    for(auto p:slot.buffers) memory_.unpin(p);
    slot.buffers.clear();
    slot.resources.clear();
}
void Workspace::close() {
    for(auto& slot:slots_) {
        if(slot.active) finish(&slot-slots_.data());
        memory_.release(slot.fp32);
        memory_.release(slot.activation);
    }
    slots_.clear();
}
}
