#include "runtime.h"
#include <cstdio>
#include <tuple>

namespace native_mlu {
Error::Error(cnrtRet_t status,const char *op)
    :std::runtime_error(std::string(op)+": CNRT "+std::to_string(status)),code(status){}
void check(cnrtRet_t status,const char *op) {
    if(status!=cnrtSuccess) throw Error(status,op);
}
Worker::Worker(const std::string& requested) {
    check(cnrtDeviceGetByPCIBusId(&ordinal_,requested.c_str()),"resolve PCI identity");
    char found[64]={};
    check(cnrtDeviceGetPCIBusId(found,sizeof(found),ordinal_),"read PCI identity");
    pci_=found;
    check(cnrtGetDeviceProperties(&properties_,ordinal_),"get device properties");
    select();
}
Worker::~Worker() {
    try { close(); } catch(const std::exception &e) {
        std::fprintf(stderr,"worker cleanup failed: %s\n",e.what());
    }
}
void Worker::select() const {
    if(closed_ || owner_!=std::this_thread::get_id()) throw std::logic_error("inactive worker or owner thread mismatch");
    check(cnrtSetDevice(ordinal_),"select worker device");
}
size_t Worker::add_queue() {
    select();
    check(fault_,"worker has a queue fault");
    queues_.reserve(queues_.size()+1);
    cnrtQueue_t q=nullptr;
    check(cnrtQueueCreate(&q),"create worker queue");
    queues_.push_back(q);
    return queues_.size()-1;
}
cnrtQueue_t Worker::queue(size_t id) const {
    select(); check(fault_,"worker has a queue fault");
    auto q=queues_.at(id);
    if(!q) throw std::logic_error("queue closed");
    return q;
}
void Worker::sync(size_t id) {
    select();
    auto q=queues_.at(id);
    if(!q) throw std::logic_error("queue closed");
    const auto status=cnrtQueueSync(q);
    if(status!=cnrtSuccess) fault_=status;
    check(status,"synchronize worker queue");
}
void Worker::recover() {
    select();
    for(size_t i=0;i<queues_.size();++i) if(queues_[i]) sync(i);
    fault_=cnrtSuccess;
}
void Worker::close() {
    if(closed_) return;
    recover();
    for(auto &q:queues_) if(q) {
        check(cnrtQueueDestroy(q),"destroy worker queue");
        q=nullptr;
    }
    closed_=true;
}
bool WeightKey::operator<(const WeightKey& b) const {
    return std::tie(revision,tensor,shard,layout,pci,offset,type,k,n)<
           std::tie(b.revision,b.tensor,b.shard,b.layout,b.pci,b.offset,b.type,b.k,b.n);
}
Registry::~Registry() {
    if(entries_.empty()) return;
    try { close(); } catch(const std::exception &e) {
        std::fprintf(stderr,"registry cleanup failed: %s\n",e.what());
    }
}
Registry::Entry& Registry::insert(const WeightKey &key) {
    worker_.select();
    if(key.pci!=worker_.pci() || key.layout!="kn-row-major" || key.revision.empty() ||
       key.tensor.empty() || key.shard.empty()) throw std::invalid_argument("invalid weight metadata");
    auto inserted=entries_.emplace(key,std::make_shared<Entry>());
    if(!inserted.second) throw std::logic_error("duplicate weight key");
    return *inserted.first->second;
}
void Registry::host(const WeightKey &key,const uint16_t *source) {
    auto &entry=insert(key);
    const auto rc=mlu::pack_rhs_host(source,key.k,key.n,key.type,entry.weight);
    if(rc!=cnrtSuccess) {
        entry.valid=false;
        if(!entry.weight) entries_.erase(key);
        check(rc,"host weight preparation");
    }
    entry.published=true;
}
void Registry::device(const WeightKey &key,const uint16_t *source,size_t id) {
    const auto q=worker_.queue(id);
    auto &entry=insert(key);
    entry.queue=id;
    const auto rc=mlu::pack_rhs_device(source,key.k,key.n,key.type,q,entry.weight);
    if(rc!=cnrtSuccess) {
        entry.valid=false;
        if(!entry.weight) entries_.erase(key);
        check(rc,"enqueue weight packing");
    }
}
void Registry::publish(const WeightKey &key) {
    auto &entry=*entries_.at(key);
    if(!entry.valid || !entry.weight) throw std::logic_error("weight is invalid");
    if(!entry.published) {
        worker_.sync(entry.queue);
        entry.published=true;
    }
}
Registry::Lease Registry::acquire(const WeightKey &key) const {
    const auto it=entries_.find(key);
    if(it==entries_.end() || !it->second->valid || !it->second->published) return {};
    return Lease(it->second,it->second->weight);
}
void Registry::invalidate(const std::string &revision) {
    for(auto &pair:entries_) if(pair.first.revision==revision) pair.second->valid=false;
}
void Registry::collect() {
    worker_.select();
    for(auto it=entries_.begin();it!=entries_.end();) {
        auto &entry=it->second;
        if(entry->valid || entry.use_count()!=1) { ++it; continue; }
        if(entry->weight) {
            if(!entry->published) worker_.sync(entry->queue);
            check(mlu::release_packed_weight(entry->weight),"release packed weight");
        }
        it=entries_.erase(it);
    }
}
void Registry::close() {
    for(auto &pair:entries_) pair.second->valid=false;
    collect();
    if(!entries_.empty()) throw std::logic_error("outstanding weight leases");
}
size_t Registry::bytes() const {
    size_t n=0;
    for(const auto &pair:entries_) if(pair.second->weight) n+=pair.second->weight->bytes();
    return n;
}
}
